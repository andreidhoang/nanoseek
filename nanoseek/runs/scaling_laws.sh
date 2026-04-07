#!/bin/bash
# NanoSeek IsoFLOP Scaling Laws: sweep depths × FLOPs budgets
# Adapted from nanochat/runs/scaling_laws.sh for MoE architecture
#
# Usage: ./runs/scaling_laws.sh [label]
# Example: ./runs/scaling_laws.sh apr06
#
# Run from nanoseek/ directory (NOT nanoseek/nanoseek/)

set -euo pipefail
export OMP_NUM_THREADS=1

LABEL="${1:-$(date +%b%d | tr '[:upper:]' '[:lower:]')}"

FLOPS_BUDGETS=(
    1e18
    3e18
    1e19
    3e19
)
DEPTHS=(12 14 16 18 20)

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
WANDB_RUN="${WANDB_RUN:-scaling_${LABEL}}"
EVAL_TOKENS=$((100 * 524288))  # ~52M tokens for final eval

# Tuned HPs from Phase 1 — REQUIRED (the whole point of HP-before-IsoFLOP)
MATRIX_LR="${MATRIX_LR:?'Set MATRIX_LR from Phase 1 HP search (e.g., MATRIX_LR=0.01)'}"
EMBEDDING_LR="${EMBEDDING_LR:?'Set EMBEDDING_LR from Phase 1 HP search (e.g., EMBEDDING_LR=0.3)'}"

RESULTS_DIR="results/scaling_laws_${LABEL}"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/results.csv"

if [ ! -f "$RESULTS_FILE" ]; then
    echo "flops_budget,depth,model_dim,active_params,total_params,scaling_params,num_iterations,tokens_trained,val_bpb,H_load,train_time_sec" > "$RESULTS_FILE"
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

run_exists() {
    grep -q "^${1},${2}," "$RESULTS_FILE" 2>/dev/null
}

for flops in "${FLOPS_BUDGETS[@]}"; do
    log "=============================================="
    log "Compute budget: $flops FLOPs"
    log "=============================================="

    for d in "${DEPTHS[@]}"; do
        if run_exists "$flops" "$d"; then
            log "Skipping d=$d at $flops FLOPs (already in results)"
            continue
        fi

        log "Training d=$d at $flops FLOPs..."
        TAG="scaling_${flops}_d${d}"

        # OOM prevention
        DBS=32
        [ "$d" -ge 20 ] && DBS=16
        [ "$d" -ge 28 ] && DBS=8

        START_TIME=$(date +%s)

        torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
            -m nanoseek.scripts.pre_train \
            --depth="$d" \
            --target-flops="$flops" \
            --matrix-lr="$MATRIX_LR" \
            --embedding-lr="$EMBEDDING_LR" \
            --run="${WANDB_RUN}_${TAG}" \
            --device-batch-size="$DBS" \
            --eval-tokens="$EVAL_TOKENS" \
            --eval-every=999999 \
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

        log "  d=$d: active=$ACTIVE, iters=$ITERS, bpb=$VAL_BPB, H_load=$H_LOAD"
        echo "$flops,$d,$MODEL_DIM,$ACTIVE,$TOTAL,$SCALING,$ITERS,$TOKENS,$VAL_BPB,$H_LOAD,$TRAIN_TIME" >> "$RESULTS_FILE"
    done
done

log "=============================================="
log "Scaling Laws Sweep Complete"
log "=============================================="
log "Results saved to: $RESULTS_FILE"
echo ""
echo "Results:"
column -t -s',' "$RESULTS_FILE"
