#!/bin/bash
# NanoSeek Miniseries: sweep depths at compute-optimal settings
# Adapted from nanochat/runs/miniseries.sh for MoE architecture
#
# Usage: ./runs/miniseries.sh [series_name]
# Example: ./runs/miniseries.sh apr06
# Default series name: today's date (e.g., apr06)
#
# Run from nanoseek/ directory (NOT nanoseek/nanoseek/)

set -euo pipefail
export OMP_NUM_THREADS=1

SERIES_NAME="${1:-$(date +%b%d | tr '[:upper:]' '[:lower:]')}"
DEPTHS=(12 14 16 18 20 24)
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
PARAM_DATA_RATIO="${PARAM_DATA_RATIO:-20}"
WANDB_RUN="${WANDB_RUN:-${SERIES_NAME}_miniseries}"

# Tuned HPs from Phase 1 — REQUIRED
MATRIX_LR="${MATRIX_LR:?'Set MATRIX_LR from Phase 1 HP search (e.g., MATRIX_LR=0.01)'}"
EMBEDDING_LR="${EMBEDDING_LR:?'Set EMBEDDING_LR from Phase 1 HP search (e.g., EMBEDDING_LR=0.3)'}"

RESULTS_DIR="results/${SERIES_NAME}_miniseries"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/results.csv"

if [ ! -f "$RESULTS_FILE" ]; then
    echo "depth,model_dim,active_params,total_params,scaling_params,num_iterations,tokens_trained,val_bpb,H_load,train_time_sec" > "$RESULTS_FILE"
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

run_exists() {
    grep -q "^${1}," "$RESULTS_FILE" 2>/dev/null
}

log "=============================================="
log "${SERIES_NAME} Miniseries (depths: ${DEPTHS[*]})"
log "=============================================="

for d in "${DEPTHS[@]}"; do
    if run_exists "$d"; then
        log "Skipping d=$d (already in results)"
        continue
    fi

    log "Training d=$d..."
    TAG="${SERIES_NAME}_d${d}"

    # OOM prevention: reduce device batch at larger depths
    DBS=32
    [ "$d" -ge 20 ] && DBS=16
    [ "$d" -ge 28 ] && DBS=8

    START_TIME=$(date +%s)

    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
        -m nanoseek.scripts.pre_train \
        --depth="$d" \
        --target-param-data-ratio="$PARAM_DATA_RATIO" \
        --matrix-lr="$MATRIX_LR" \
        --embedding-lr="$EMBEDDING_LR" \
        --run="${WANDB_RUN}_d${d}" \
        --device-batch-size="$DBS" \
        --save-every=-1 \
        --seed=42 \
        2>&1 | tee "$RESULTS_DIR/${TAG}.log"

    END_TIME=$(date +%s)
    TRAIN_TIME=$((END_TIME - START_TIME))

    # Extract metrics from parseable log lines
    LOG="$RESULTS_DIR/${TAG}.log"
    ACTIVE=$(grep "^active " "$LOG" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
    TOTAL=$(grep "^total " "$LOG" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
    SCALING=$(grep "^scaling " "$LOG" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
    MODEL_DIM=$(grep "hidden_size:" "$LOG" | tail -1 | grep -oP '\d+')
    ITERS=$(grep "Calculated iterations" "$LOG" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
    BATCH_TOKENS=$(grep "Training plan:" "$LOG" | tail -1 | grep -oP '[\d,]+ tok/step' | grep -oP '[\d,]+' | tr -d ',')
    TOKENS=$((ITERS * BATCH_TOKENS))
    VAL_BPB=$(grep "EMA Validation bpb:" "$LOG" | tail -1 | grep -oP '[\d.]+$')
    H_LOAD=$(grep "H_load:" "$LOG" | tail -1 | grep -oP 'H_load: [\d.]+' | grep -oP '[\d.]+')

    [ -z "$VAL_BPB" ] && VAL_BPB="0.0"
    [ -z "$H_LOAD" ] && H_LOAD="0.0"

    log "  d=$d: active=$ACTIVE, scaling=$SCALING, bpb=$VAL_BPB, H_load=$H_LOAD, time=${TRAIN_TIME}s"
    echo "$d,$MODEL_DIM,$ACTIVE,$TOTAL,$SCALING,$ITERS,$TOKENS,$VAL_BPB,$H_LOAD,$TRAIN_TIME" >> "$RESULTS_FILE"
done

log "=============================================="
log "${SERIES_NAME} Miniseries Complete!"
log "=============================================="
log "Results saved to: $RESULTS_FILE"
echo ""
echo "Results:"
column -t -s',' "$RESULTS_FILE"
