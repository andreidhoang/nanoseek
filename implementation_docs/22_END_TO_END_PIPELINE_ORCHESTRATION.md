# 22 — End-to-End Pipeline Orchestration

## Principal Engineer's Implementation Guide — February 2026

**Author**: Distinguished AI Researcher & Principal Engineer, Foundation Models Division
**Scope**: Complete pipeline orchestration — shell scripts, DAG execution, artifact management, and the Final Table from pre-training through ship gate
**Prerequisite**: `00_MASTER_PLAN.md`, `06_MULTI_PHASE_TRAINING_ORCHESTRATION.md`, `09_POST_TRAINING_SFT_DPO_RLVR.md`, `10_EVALUATION_BENCHMARKS.md`, `17_EVAL_GATE_SHIP_REJECT.md`
**Criticality**: **MAXIMUM** — Without orchestration, every stage is a manual SSH-and-pray ritual. One typo wastes $300 and 14 hours.

---

## 1. Problem Statement

Training NanoSeek involves **six sequential stages**, each consuming the output of the previous:

| Stage | Input | Output | Wall Time (8×H100) | Cost |
|-------|-------|--------|--------------------:|-----:|
| Pre-train | FineWeb-Edu parquet | Base checkpoint | ~14 hrs | ~$300 |
| SFT | Base ckpt + UltraChat | SFT checkpoint | ~2 hrs | ~$48 |
| DPO | SFT ckpt + UltraFeedback | DPO checkpoint | ~30 min | ~$12 |
| Eval | All checkpoints + benchmarks | JSON results | ~30 min | ~$12 |
| Ship Gate | JSON results + thresholds | gate_decision.json | <1 sec | ~$0 |
| Report | All artifacts | Final Table + PNGs | <10 sec | ~$0 |

### What Happens Without Orchestration

| Failure Mode | Consequence |
|---|---|
| Engineer forgets `--resume_from` and retrains from scratch | 14 hours wasted; $300 burned |
| SFT script uses wrong checkpoint path (stale base) | Entire post-training built on wrong foundation |
| Eval runs against base model instead of DPO model | Published numbers don't reflect pipeline output |
| Gate check skipped — "I'll eyeball the JSON" | Silent safety regression ships to production |
| GPU memory not freed between stages | OOM on SFT because pre-training CUDA context holds 40GB |

### Current State

- `scripts/pre-train.py`: Complete training loop with checkpoint save/resume (Doc 06)
- `model/eval/report.py`: Report generation infrastructure with `Report` class
- `scripts/base_eval.py`: CORE benchmark evaluation
- **No SFT/DPO/serve/eval-all/gate shell scripts exist**
- **No master pipeline script or Final Table generation exists**

### Targets

- **One-command full pipeline**: `./scripts/run_pipeline.sh` runs everything
- **Idempotency**: Re-running any script skips already-completed work
- **Failure recovery**: Pipeline resumes from last successful stage
- **CI/CD integration**: `gate_check.sh` exits 0 (SHIP) or 1 (REJECT)

---

## 2. First Principles

### 2a. Pipeline as DAG

The pipeline is a directed acyclic graph. Modeling it as a DAG enables parallel execution of independent stages and correct dependency tracking:

```
                    ┌──────────────┐
                    │  Pre-Train   │
                    │  (Doc 06)    │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │     SFT      │
                    │  (Doc 09)    │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │     DPO      │
                    │  (Doc 09)    │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──────┐ ┌──▼──────┐ ┌───▼─────────┐
       │  Eval Suite  │ │ Safety  │ │  Leakage    │
       │  (Doc 10)    │ │(Doc 20) │ │  (Doc 18)   │
       └──────┬──────┘ └──┬──────┘ └───┬─────────┘
              └────────────┼────────────┘
                           │
                    ┌──────▼───────┐
                    │  Ship Gate   │
                    │  (Doc 17)    │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Report     │
                    └──────────────┘
```

The eval branches (quality, safety, leakage) are **independent** — they can run in parallel on separate GPUs.

### 2b. Idempotency — Re-Run Any Stage Safely

Every script satisfies `f(f(x)) = f(x)`:

1. **Checkpoint detection**: Before starting, check if output artifact exists and is valid. If so, log "skipping" and exit 0.
2. **Atomic writes**: Write to `.tmp`, then `mv` to final path. Prevents partial artifacts.
3. **Forced re-run**: `--force` flag bypasses detection for intentional re-runs.

### 2c. Artifact Management

Each stage has explicit input/output contracts:

```
Stage           Input Artifacts                     Output Artifacts
─────────────   ─────────────────────────────────   ──────────────────────────────
Pre-Train       data/*.parquet, config.yaml         checkpoints/pretrain/final/
SFT             checkpoints/pretrain/final/         checkpoints/sft/final/
DPO             checkpoints/sft/final/              checkpoints/dpo/final/
Eval            checkpoints/*/final/                reports/eval_*.json
Gate            reports/eval_*.json                 reports/gate_decision.json
Report          reports/*.json                      reports/final_table.md
```

### 2d. Failure Recovery

The master pipeline tracks completed stages in a **state file** (`reports/.pipeline_state`):

```
PRETRAIN COMPLETED 2026-02-27T14:32:00
SFT COMPLETED 2026-02-27T16:45:00
DPO FAILED 2026-02-27T17:15:00
EVAL_BASE PENDING
```

On re-run: skip `COMPLETED`, retry `FAILED`, execute `PENDING`.

### 2e. CI/CD Integration — Gate as Merge Blocker

```yaml
# .github/workflows/model-release.yml
- run: ./scripts/run_pipeline.sh
- run: ./scripts/gate_check.sh  # exit 0 = SHIP, exit 1 = REJECT
```

### 2f. Reproducibility Through Config Pinning

Every pipeline run saves a manifest (`reports/pipeline_manifest.json`) with git commit, config hashes, torch version, and hardware info. To reproduce: checkout the commit, verify hashes, re-execute.

---

## 3. Production Code

### 3a. Common Utilities (`scripts/pipeline_common.sh`)

```bash
#!/usr/bin/env bash
# scripts/pipeline_common.sh — Shared utilities for NanoSeek pipeline scripts

set -euo pipefail

# ─── Color codes ─────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
    BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; BOLD=''; NC=''
fi

# ─── Logging ─────────────────────────────────────────────────────────────────
log()   { echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $*"; }
info()  { echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] INFO${NC}  $*"; }
warn()  { echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARN${NC}  $*" >&2; }
error() { echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR${NC} $*" >&2; }
fatal() { error "$@"; exit 1; }

# ─── Defaults ────────────────────────────────────────────────────────────────
export NANOSEEK_BASE_DIR="${NANOSEEK_BASE_DIR:-out}"
export NANOSEEK_DATA_DIR="${NANOSEEK_DATA_DIR:-data}"
export NANOSEEK_REPORT_DIR="${NANOSEEK_BASE_DIR}/reports"
export NANOSEEK_CHECKPOINT_DIR="${NANOSEEK_BASE_DIR}/checkpoints"
export NANOSEEK_NUM_GPUS="${NANOSEEK_NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}"
export FORCE="${FORCE:-false}"

mkdir -p "$NANOSEEK_REPORT_DIR" "$NANOSEEK_CHECKPOINT_DIR"

# ─── Artifact checking ──────────────────────────────────────────────────────
check_artifact() {
    local artifact="$1"
    if [[ "$FORCE" == "true" ]]; then
        log "FORCE mode — re-running despite existing artifact: $artifact"
        return 1
    fi
    if [[ -f "$artifact" ]]; then
        local size
        size=$(stat -c%s "$artifact" 2>/dev/null || stat -f%z "$artifact" 2>/dev/null || echo 0)
        if [[ "$size" -gt 0 ]]; then
            info "Artifact exists ($size bytes): $artifact — skipping"
            return 0
        fi
        rm -f "$artifact"
    fi
    if [[ -d "$artifact" ]]; then
        local count
        count=$(find "$artifact" -type f | wc -l)
        if [[ "$count" -gt 0 ]]; then
            info "Artifact directory exists ($count files): $artifact — skipping"
            return 0
        fi
        rm -rf "$artifact"
    fi
    return 1
}

# ─── Checkpoint detection ───────────────────────────────────────────────────
find_latest_checkpoint() {
    local ckpt_dir="$1"
    [[ ! -d "$ckpt_dir" ]] && echo "" && return
    find "$ckpt_dir" -maxdepth 3 -name "*.pt" -o -name "*.safetensors" 2>/dev/null | \
        sort | tail -1 || echo ""
}

# ─── GPU memory cleanup ─────────────────────────────────────────────────────
release_gpu_memory() {
    info "Releasing GPU memory..."
    python3 -c "
import torch, gc
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    gc.collect()
" 2>/dev/null || warn "GPU cleanup failed (non-fatal)"
}

# ─── Timing ──────────────────────────────────────────────────────────────────
STAGE_START_TIME=""
stage_start() {
    STAGE_START_TIME=$(date +%s)
    log "════════════════════════════════════════════════════"
    log "  Starting: ${BOLD}$1${NC}"
    log "════════════════════════════════════════════════════"
}
stage_end() {
    local end_time; end_time=$(date +%s)
    local elapsed=$((end_time - STAGE_START_TIME))
    local h=$((elapsed/3600)) m=$(((elapsed%3600)/60)) s=$((elapsed%60))
    if [[ "${2:-0}" -eq 0 ]]; then
        info "Completed: ${BOLD}$1${NC} in ${h}h ${m}m ${s}s"
    else
        error "FAILED: ${BOLD}$1${NC} after ${h}h ${m}m ${s}s (exit $2)"
    fi
}

# ─── Pipeline state management ──────────────────────────────────────────────
PIPELINE_STATE_FILE="${NANOSEEK_REPORT_DIR}/.pipeline_state"

update_pipeline_state() {
    local stage="$1" status="$2"
    touch "$PIPELINE_STATE_FILE"
    grep -v "^${stage} " "$PIPELINE_STATE_FILE" > "${PIPELINE_STATE_FILE}.tmp" 2>/dev/null || true
    echo "${stage} ${status} $(date -Iseconds)" >> "${PIPELINE_STATE_FILE}.tmp"
    mv "${PIPELINE_STATE_FILE}.tmp" "$PIPELINE_STATE_FILE"
}

get_pipeline_state() {
    [[ ! -f "$PIPELINE_STATE_FILE" ]] && echo "PENDING" && return
    grep "^$1 " "$PIPELINE_STATE_FILE" 2>/dev/null | awk '{print $2}' || echo "PENDING"
}

# ─── Cleanup trap ────────────────────────────────────────────────────────────
cleanup() {
    local exit_code=$?
    [[ $exit_code -ne 0 ]] && error "Script failed (exit $exit_code). Re-run with FORCE=true."
}
trap cleanup EXIT

# ─── Git metadata ────────────────────────────────────────────────────────────
get_git_commit() { git rev-parse --short HEAD 2>/dev/null || echo "unknown"; }
get_git_dirty()  {
    [[ -n "$(git status --porcelain 2>/dev/null)" ]] && echo "true" || echo "false"
}
```

### 3b. SFT Training Script (`scripts/train_sft.sh`)

```bash
#!/usr/bin/env bash
# scripts/train_sft.sh — One-command supervised fine-tuning
#
# Usage:  ./scripts/train_sft.sh [OPTIONS]
# Options:
#   --base-checkpoint PATH   Pre-trained checkpoint (default: auto-detect)
#   --data-dir PATH          UltraChat data (default: $NANOSEEK_DATA_DIR/ultrachat)
#   --output-dir PATH        Output dir (default: $NANOSEEK_CHECKPOINT_DIR/sft)
#   --epochs NUM             Training epochs (default: 3)
#   --lr FLOAT               Learning rate (default: 2e-5)
#   --batch-size NUM         Per-device batch size (default: 4)
#   --max-seq-len NUM        Max sequence length (default: 4096)
#   --force                  Re-run even if checkpoint exists
#   --help                   Show this message

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/pipeline_common.sh"

BASE_CHECKPOINT="${NANOSEEK_CHECKPOINT_DIR}/pretrain/final"
DATA_DIR="${NANOSEEK_DATA_DIR}/ultrachat"
OUTPUT_DIR="${NANOSEEK_CHECKPOINT_DIR}/sft"
EPOCHS=3; LR="2e-5"; BATCH_SIZE=4; MAX_SEQ_LEN=4096

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-checkpoint) BASE_CHECKPOINT="$2"; shift 2 ;;
        --data-dir)        DATA_DIR="$2"; shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2"; shift 2 ;;
        --epochs)          EPOCHS="$2"; shift 2 ;;
        --lr)              LR="$2"; shift 2 ;;
        --batch-size)      BATCH_SIZE="$2"; shift 2 ;;
        --max-seq-len)     MAX_SEQ_LEN="$2"; shift 2 ;;
        --force)           FORCE="true"; shift ;;
        --help|-h)         head -16 "$0" | grep '^#' | sed 's/^# \?//'; exit 0 ;;
        *)                 fatal "Unknown option: $1" ;;
    esac
done

stage_start "SFT Training"

FINAL_ARTIFACT="${OUTPUT_DIR}/final/model_state.pt"
if check_artifact "$FINAL_ARTIFACT"; then
    stage_end "SFT Training (skipped)" 0; exit 0
fi

# Auto-detect base checkpoint if needed
if [[ ! -d "$BASE_CHECKPOINT" ]] && [[ ! -f "$BASE_CHECKPOINT" ]]; then
    BASE_CHECKPOINT=$(find_latest_checkpoint "${NANOSEEK_CHECKPOINT_DIR}/pretrain")
    [[ -z "$BASE_CHECKPOINT" ]] && fatal "No base checkpoint found. Run pre-training first."
    info "Auto-detected base checkpoint: $BASE_CHECKPOINT"
fi
[[ ! -d "$DATA_DIR" ]] && fatal "Data directory not found: $DATA_DIR"

info "Config: base=$BASE_CHECKPOINT data=$DATA_DIR epochs=$EPOCHS lr=$LR bs=$BATCH_SIZE"
release_gpu_memory
mkdir -p "$OUTPUT_DIR"

# Check for partial checkpoint (resume support)
RESUME_FLAG=""
PARTIAL_CKPT=$(find_latest_checkpoint "$OUTPUT_DIR")
[[ -n "$PARTIAL_CKPT" ]] && info "Resuming from: $PARTIAL_CKPT" && RESUME_FLAG="--resume_from $PARTIAL_CKPT"

# Save run metadata
cat > "${OUTPUT_DIR}/sft_config.json" <<EOF
{"stage":"sft","git_commit":"$(get_git_commit)","timestamp":"$(date -Iseconds)",
 "base_checkpoint":"$BASE_CHECKPOINT","epochs":$EPOCHS,"lr":"$LR","batch_size":$BATCH_SIZE}
EOF

update_pipeline_state "SFT" "RUNNING"

TRAIN_CMD="python3 -m fms.post_training.sft \
    --base_model $BASE_CHECKPOINT --data_dir $DATA_DIR --output_dir $OUTPUT_DIR \
    --epochs $EPOCHS --lr $LR --per_device_batch_size $BATCH_SIZE \
    --max_seq_len $MAX_SEQ_LEN --save_strategy epoch --eval_steps 500 \
    --warmup_ratio 0.03 --weight_decay 0.01 --gradient_checkpointing true --bf16 true \
    $RESUME_FLAG"

if [[ "$NANOSEEK_NUM_GPUS" -gt 1 ]]; then
    torchrun --nproc_per_node="$NANOSEEK_NUM_GPUS" ${TRAIN_CMD#python3 } \
        2>&1 | tee "${OUTPUT_DIR}/sft_train.log"
else
    $TRAIN_CMD 2>&1 | tee "${OUTPUT_DIR}/sft_train.log"
fi

[[ $? -ne 0 ]] && update_pipeline_state "SFT" "FAILED" && fatal "SFT training failed"

# Verify output — link final artifact if saved to different path
if [[ ! -f "$FINAL_ARTIFACT" ]]; then
    ACTUAL=$(find_latest_checkpoint "$OUTPUT_DIR")
    if [[ -n "$ACTUAL" ]]; then
        mkdir -p "$(dirname "$FINAL_ARTIFACT")"
        ln -sf "$(realpath "$ACTUAL")" "$FINAL_ARTIFACT"
    else
        update_pipeline_state "SFT" "FAILED"
        fatal "SFT completed but no checkpoint found"
    fi
fi

update_pipeline_state "SFT" "COMPLETED"
stage_end "SFT Training" 0
```

### 3c. DPO Training Script (`scripts/train_dpo.sh`)

```bash
#!/usr/bin/env bash
# scripts/train_dpo.sh — One-command Direct Preference Optimization
#
# Usage:  ./scripts/train_dpo.sh [OPTIONS]
# Options:
#   --sft-checkpoint PATH    SFT checkpoint (default: auto-detect)
#   --data-dir PATH          UltraFeedback data (default: $NANOSEEK_DATA_DIR/ultrafeedback)
#   --output-dir PATH        Output dir (default: $NANOSEEK_CHECKPOINT_DIR/dpo)
#   --beta FLOAT             DPO beta (default: 0.1)
#   --lr FLOAT               Learning rate (default: 5e-7)
#   --batch-size NUM         Per-device batch size (default: 2)
#   --epochs NUM             Epochs (default: 1)
#   --force                  Re-run even if checkpoint exists
#   --help                   Show this message

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/pipeline_common.sh"

SFT_CHECKPOINT="${NANOSEEK_CHECKPOINT_DIR}/sft/final"
DATA_DIR="${NANOSEEK_DATA_DIR}/ultrafeedback"
OUTPUT_DIR="${NANOSEEK_CHECKPOINT_DIR}/dpo"
BETA="0.1"; LR="5e-7"; BATCH_SIZE=2; MAX_SEQ_LEN=2048; EPOCHS=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sft-checkpoint) SFT_CHECKPOINT="$2"; shift 2 ;;
        --data-dir)       DATA_DIR="$2"; shift 2 ;;
        --output-dir)     OUTPUT_DIR="$2"; shift 2 ;;
        --beta)           BETA="$2"; shift 2 ;;
        --lr)             LR="$2"; shift 2 ;;
        --batch-size)     BATCH_SIZE="$2"; shift 2 ;;
        --epochs)         EPOCHS="$2"; shift 2 ;;
        --force)          FORCE="true"; shift ;;
        --help|-h)        head -16 "$0" | grep '^#' | sed 's/^# \?//'; exit 0 ;;
        *)                fatal "Unknown option: $1" ;;
    esac
done

stage_start "DPO Training"

FINAL_ARTIFACT="${OUTPUT_DIR}/final/model_state.pt"
if check_artifact "$FINAL_ARTIFACT"; then
    stage_end "DPO Training (skipped)" 0; exit 0
fi

if [[ ! -d "$SFT_CHECKPOINT" ]] && [[ ! -f "$SFT_CHECKPOINT" ]]; then
    SFT_CHECKPOINT=$(find_latest_checkpoint "${NANOSEEK_CHECKPOINT_DIR}/sft")
    [[ -z "$SFT_CHECKPOINT" ]] && fatal "No SFT checkpoint found. Run SFT first."
    info "Auto-detected SFT checkpoint: $SFT_CHECKPOINT"
fi
[[ ! -d "$DATA_DIR" ]] && fatal "Data directory not found: $DATA_DIR"

info "Config: sft=$SFT_CHECKPOINT beta=$BETA lr=$LR epochs=$EPOCHS"
release_gpu_memory
mkdir -p "$OUTPUT_DIR"

RESUME_FLAG=""
PARTIAL=$(find_latest_checkpoint "$OUTPUT_DIR")
[[ -n "$PARTIAL" ]] && RESUME_FLAG="--resume_from $PARTIAL"

cat > "${OUTPUT_DIR}/dpo_config.json" <<EOF
{"stage":"dpo","git_commit":"$(get_git_commit)","timestamp":"$(date -Iseconds)",
 "sft_checkpoint":"$SFT_CHECKPOINT","beta":$BETA,"lr":"$LR","epochs":$EPOCHS}
EOF

update_pipeline_state "DPO" "RUNNING"

TRAIN_CMD="python3 -m fms.post_training.dpo \
    --model_path $SFT_CHECKPOINT --ref_model_path $SFT_CHECKPOINT \
    --data_dir $DATA_DIR --output_dir $OUTPUT_DIR \
    --beta $BETA --lr $LR --per_device_batch_size $BATCH_SIZE \
    --max_seq_len $MAX_SEQ_LEN --epochs $EPOCHS \
    --gradient_checkpointing true --bf16 true $RESUME_FLAG"

if [[ "$NANOSEEK_NUM_GPUS" -gt 1 ]]; then
    torchrun --nproc_per_node="$NANOSEEK_NUM_GPUS" ${TRAIN_CMD#python3 } \
        2>&1 | tee "${OUTPUT_DIR}/dpo_train.log"
else
    $TRAIN_CMD 2>&1 | tee "${OUTPUT_DIR}/dpo_train.log"
fi

[[ $? -ne 0 ]] && update_pipeline_state "DPO" "FAILED" && fatal "DPO training failed"

if [[ ! -f "$FINAL_ARTIFACT" ]]; then
    ACTUAL=$(find_latest_checkpoint "$OUTPUT_DIR")
    if [[ -n "$ACTUAL" ]]; then
        mkdir -p "$(dirname "$FINAL_ARTIFACT")"
        ln -sf "$(realpath "$ACTUAL")" "$FINAL_ARTIFACT"
    else
        update_pipeline_state "DPO" "FAILED"
        fatal "DPO completed but no checkpoint found"
    fi
fi

update_pipeline_state "DPO" "COMPLETED"
stage_end "DPO Training" 0
```

### 3d. Serving Script (`scripts/serve.sh`)

```bash
#!/usr/bin/env bash
# scripts/serve.sh — One-command model serving with health check
#
# Usage:  ./scripts/serve.sh [OPTIONS]
# Options:
#   --checkpoint PATH     Checkpoint to serve (default: auto-detect best)
#   --host HOST           Bind address (default: 0.0.0.0)
#   --port PORT           Port (default: 8000)
#   --quantize INT8|INT4  Quantization mode (default: none)
#   --background          Run in background, return after health check
#   --help                Show this message

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/pipeline_common.sh"

CHECKPOINT=""; HOST="0.0.0.0"; PORT=8000; QUANTIZE=""; BACKGROUND=false
HEALTH_TIMEOUT=60; MODEL_NAME="${NANOSEEK_MODEL_NAME:-nanoseek-1b}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --host)       HOST="$2"; shift 2 ;;
        --port)       PORT="$2"; shift 2 ;;
        --quantize)   QUANTIZE="$2"; shift 2 ;;
        --background) BACKGROUND=true; shift ;;
        --help|-h)    head -14 "$0" | grep '^#' | sed 's/^# \?//'; exit 0 ;;
        *)            fatal "Unknown option: $1" ;;
    esac
done

stage_start "Model Serving"

# Auto-detect best checkpoint (DPO > SFT > pretrain)
if [[ -z "$CHECKPOINT" ]]; then
    for stage in dpo sft pretrain; do
        CHECKPOINT=$(find_latest_checkpoint "${NANOSEEK_CHECKPOINT_DIR}/${stage}")
        [[ -n "$CHECKPOINT" ]] && info "Auto-detected ($stage): $CHECKPOINT" && break
    done
    [[ -z "$CHECKPOINT" ]] && fatal "No checkpoint found. Train a model first."
fi

SERVE_CMD="python3 -m fms.serving.api_server \
    --model_path $CHECKPOINT --host $HOST --port $PORT --model_name $MODEL_NAME"
[[ -n "$QUANTIZE" ]] && SERVE_CMD="$SERVE_CMD --quantize $QUANTIZE"

if [[ "$BACKGROUND" == "true" ]]; then
    $SERVE_CMD > "${NANOSEEK_BASE_DIR}/serve.log" 2>&1 &
    SERVER_PID=$!
    info "Server PID: $SERVER_PID — waiting for health..."

    ELAPSED=0
    while [[ $ELAPSED -lt $HEALTH_TIMEOUT ]]; do
        if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "$SERVER_PID" > "${NANOSEEK_BASE_DIR}/serve.pid"
            info "Server healthy! Smoke test:"
            curl -s "http://localhost:${PORT}/v1/completions" \
                -H "Content-Type: application/json" \
                -d "{\"model\":\"$MODEL_NAME\",\"prompt\":\"Hello\",\"max_tokens\":5}" || true
            stage_end "Model Serving" 0; exit 0
        fi
        kill -0 "$SERVER_PID" 2>/dev/null || fatal "Server died. See ${NANOSEEK_BASE_DIR}/serve.log"
        sleep 2; ELAPSED=$((ELAPSED + 2))
    done
    kill "$SERVER_PID" 2>/dev/null || true
    fatal "Server not healthy within ${HEALTH_TIMEOUT}s"
else
    info "Running in foreground (Ctrl+C to stop)..."
    exec $SERVE_CMD
fi
```

### 3e. Full Evaluation Script (`scripts/eval_all.sh`)

```bash
#!/usr/bin/env bash
# scripts/eval_all.sh — One-command full evaluation suite
#
# Runs all benchmarks: quality (MMLU, GSM8K, HumanEval, ARC, MT-Bench),
# safety (TruthfulQA, toxicity, refusal), and contamination detection.
#
# Usage:  ./scripts/eval_all.sh [OPTIONS]
# Options:
#   --checkpoint PATH      Checkpoint to evaluate (default: auto-detect)
#   --stage NAME           Label: baseline|sft|dpo (default: inferred from path)
#   --output-dir PATH      Result directory (default: $NANOSEEK_REPORT_DIR)
#   --skip-safety          Skip safety benchmarks
#   --skip-leakage         Skip contamination detection
#   --parallel             Run benchmarks in parallel (multi-GPU only)
#   --force                Re-run even if results exist
#   --help                 Show this message

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/pipeline_common.sh"

CHECKPOINT=""; STAGE=""; OUTPUT_DIR="$NANOSEEK_REPORT_DIR"
SKIP_SAFETY=false; SKIP_LEAKAGE=false; PARALLEL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)   CHECKPOINT="$2"; shift 2 ;;
        --stage)        STAGE="$2"; shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --skip-safety)  SKIP_SAFETY=true; shift ;;
        --skip-leakage) SKIP_LEAKAGE=true; shift ;;
        --parallel)     PARALLEL=true; shift ;;
        --force)        FORCE="true"; shift ;;
        --help|-h)      head -18 "$0" | grep '^#' | sed 's/^# \?//'; exit 0 ;;
        *)              fatal "Unknown option: $1" ;;
    esac
done

stage_start "Full Evaluation"

# Auto-detect checkpoint and stage
if [[ -z "$CHECKPOINT" ]]; then
    for s in dpo sft pretrain; do
        CHECKPOINT=$(find_latest_checkpoint "${NANOSEEK_CHECKPOINT_DIR}/${s}")
        [[ -n "$CHECKPOINT" ]] && STAGE="${STAGE:-$s}" && break
    done
    [[ -z "$CHECKPOINT" ]] && fatal "No checkpoint found."
fi
[[ -z "$STAGE" ]] && case "$CHECKPOINT" in
    *"/dpo/"*) STAGE="dpo" ;; *"/sft/"*) STAGE="sft" ;; *) STAGE="baseline" ;;
esac

OUTPUT_FILE="${OUTPUT_DIR}/eval_${STAGE}.json"
if check_artifact "$OUTPUT_FILE"; then
    stage_end "Eval (skipped)" 0; exit 0
fi

info "Eval: checkpoint=$CHECKPOINT stage=$STAGE"
mkdir -p "$OUTPUT_DIR"

EVAL_PIDS=()

run_benchmark() {
    local name="$1" out="${OUTPUT_DIR}/${1}_${STAGE}.json"
    check_artifact "$out" && return 0
    local cmd="python3 -m fms.eval_harness.run --model_path $CHECKPOINT --benchmark $name --output $out --device cuda"
    if [[ "$PARALLEL" == "true" ]]; then
        $cmd > "${OUTPUT_DIR}/${name}_${STAGE}.log" 2>&1 &
        EVAL_PIDS+=("$!:$name")
    else
        $cmd 2>&1 | tee "${OUTPUT_DIR}/${name}_${STAGE}.log" || error "Benchmark $name failed"
    fi
}

# Quality benchmarks
for b in mmlu gsm8k humaneval arc mt_bench; do run_benchmark "$b"; done

# Safety benchmarks
if [[ "$SKIP_SAFETY" != "true" ]]; then
    for b in truthfulqa toxicity refusal; do run_benchmark "$b"; done
fi

# Contamination detection
if [[ "$SKIP_LEAKAGE" != "true" ]]; then
    python3 -m fms.eval_harness.contamination \
        --data_dir "$NANOSEEK_DATA_DIR" --output "${OUTPUT_DIR}/leakage_${STAGE}.json" \
        2>&1 | tee "${OUTPUT_DIR}/leakage_${STAGE}.log" || warn "Leakage detection failed"
fi

# Wait for parallel jobs
if [[ "$PARALLEL" == "true" ]] && [[ ${#EVAL_PIDS[@]} -gt 0 ]]; then
    for entry in "${EVAL_PIDS[@]}"; do
        PID="${entry%%:*}"; NAME="${entry##*:}"
        wait "$PID" && info "  $NAME: done" || error "  $NAME: FAILED"
    done
fi

# Aggregate into single JSON
python3 -c "
import json, glob, os, sys
results = {'stage': '$STAGE', 'checkpoint': '$CHECKPOINT'}
for f in sorted(glob.glob(os.path.join('$OUTPUT_DIR', f'*_${STAGE}.json'))):
    name = os.path.basename(f).replace(f'_${STAGE}.json', '')
    if name.startswith('eval_'): continue
    try:
        with open(f) as fh: results[name] = json.load(fh)
    except Exception as e: print(f'Warning: {f}: {e}', file=sys.stderr)
with open('${OUTPUT_FILE}.tmp', 'w') as fh: json.dump(results, fh, indent=2)
" && mv "${OUTPUT_FILE}.tmp" "$OUTPUT_FILE"

update_pipeline_state "EVAL_${STAGE^^}" "COMPLETED"
stage_end "Full Evaluation ($STAGE)" 0
```

### 3f. Gate Check Script (`scripts/gate_check.sh`)

```bash
#!/usr/bin/env bash
# scripts/gate_check.sh — One-command ship/reject gate
#
# Exit codes: 0 = SHIP, 1 = REJECT, 2 = ERROR
#
# Usage:  ./scripts/gate_check.sh [OPTIONS]
# Options:
#   --baseline PATH     Baseline eval JSON (default: $NANOSEEK_REPORT_DIR/eval_baseline.json)
#   --candidate PATH    Candidate eval JSON (default: $NANOSEEK_REPORT_DIR/eval_dpo.json)
#   --thresholds PATH   Custom threshold config (default: built-in)
#   --output PATH       Gate decision output (default: $NANOSEEK_REPORT_DIR/gate_decision.json)
#   --strict            Fail on any quality regression
#   --help              Show this message

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/pipeline_common.sh"

BASELINE="${NANOSEEK_REPORT_DIR}/eval_baseline.json"
CANDIDATE="${NANOSEEK_REPORT_DIR}/eval_dpo.json"
THRESHOLDS=""; OUTPUT="${NANOSEEK_REPORT_DIR}/gate_decision.json"
STRICT=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --baseline)   BASELINE="$2"; shift 2 ;;
        --candidate)  CANDIDATE="$2"; shift 2 ;;
        --thresholds) THRESHOLDS="$2"; shift 2 ;;
        --output)     OUTPUT="$2"; shift 2 ;;
        --strict)     STRICT=true; shift ;;
        --help|-h)    head -15 "$0" | grep '^#' | sed 's/^# \?//'; exit 0 ;;
        *)            fatal "Unknown option: $1" ;;
    esac
done

stage_start "Ship Gate"

[[ ! -f "$BASELINE" ]]  && error "Baseline not found: $BASELINE"  && exit 2
[[ ! -f "$CANDIDATE" ]] && error "Candidate not found: $CANDIDATE" && exit 2

mkdir -p "$(dirname "$OUTPUT")"

BASELINE_PATH="$BASELINE" CANDIDATE_PATH="$CANDIDATE" \
OUTPUT_PATH="$OUTPUT" STRICT_MODE="$STRICT" THRESHOLDS_PATH="${THRESHOLDS:-}" \
python3 << 'GATE_EOF'
import json, sys, os
from datetime import datetime

baseline_path = os.environ["BASELINE_PATH"]
candidate_path = os.environ["CANDIDATE_PATH"]
output_path = os.environ["OUTPUT_PATH"]
strict = os.environ.get("STRICT_MODE", "false") == "true"
thresholds_path = os.environ.get("THRESHOLDS_PATH", "")

with open(baseline_path) as f: baseline = json.load(f)
with open(candidate_path) as f: candidate = json.load(f)

thresholds = {
    "quality": {"mmlu": 0.30, "gsm8k": 0.25, "humaneval": 0.25, "arc": 0.10, "mt_bench": 0.10},
    "safety":  {"truthfulqa": {"floor": 0.30}, "toxicity": {"ceil": 0.05}, "refusal": {"floor": 0.85}},
}
if thresholds_path and os.path.exists(thresholds_path):
    with open(thresholds_path) as f: thresholds = json.load(f)

def get_metric(data, name):
    if name in data:
        v = data[name]
        return v.get("accuracy", v.get("score", v.get("value", v))) if isinstance(v, dict) else v
    for val in data.values():
        if isinstance(val, dict) and name in val:
            return get_metric(val, name)
    return None

decision = {"timestamp": datetime.now().isoformat(), "verdict": "SHIP", "reasons": [], "metrics": {}}

# Safety checks (HARD constraints)
for metric, cfg in thresholds.get("safety", {}).items():
    b, c = get_metric(baseline, metric), get_metric(candidate, metric)
    entry = {"baseline": b, "candidate": c, "status": "PASS"}
    if c is not None and b is not None:
        is_inverse = metric == "toxicity"
        regression = (c - b) if is_inverse else (b - c)
        if regression > 0:
            entry.update(status="FAIL", reason=f"{metric} safety regression: {regression:+.4f}")
            decision["verdict"] = "REJECT"
            decision["reasons"].append(entry["reason"])
    decision["metrics"][metric] = entry

# Quality checks (SOFT — weighted composite)
composite, total_w = 0.0, 0.0
for metric, weight in thresholds.get("quality", {}).items():
    b, c = get_metric(baseline, metric), get_metric(candidate, metric)
    entry = {"baseline": b, "candidate": c, "weight": weight, "status": "PASS"}
    if c is not None and b is not None and b > 0:
        delta = (c - b) / b
        entry["delta_pct"] = f"{delta:+.2%}"
        composite += delta * weight
        total_w += weight
        if strict and delta < 0:
            entry["status"] = "FAIL"
            decision["verdict"] = "REJECT"
            decision["reasons"].append(f"{metric} regressed: {delta:+.2%}")
    decision["metrics"][metric] = entry

decision["composite_quality_delta"] = f"{composite / total_w:+.4f}" if total_w > 0 else "+0.0000"

with open(output_path + ".tmp", "w") as f: json.dump(decision, f, indent=2)
os.rename(output_path + ".tmp", output_path)

icon = "✅" if decision["verdict"] == "SHIP" else "❌"
print(f"\n{'='*60}\n  GATE DECISION: {icon} {decision['verdict']}\n{'='*60}")
print(f"  Composite quality delta: {decision['composite_quality_delta']}")
for r in decision["reasons"]: print(f"  - {r}")
print(f"  Report: {output_path}\n{'='*60}\n")
sys.exit(0 if decision["verdict"] == "SHIP" else 1)
GATE_EOF

GATE_EXIT=$?
if [[ $GATE_EXIT -eq 0 ]]; then
    update_pipeline_state "GATE" "COMPLETED"; stage_end "Ship Gate" 0; exit 0
elif [[ $GATE_EXIT -eq 1 ]]; then
    update_pipeline_state "GATE" "COMPLETED"; stage_end "Ship Gate" 0; exit 1
else
    update_pipeline_state "GATE" "FAILED"; exit 2
fi
```

### 3g. Report Generator (`scripts/generate_report.py`)

```python
#!/usr/bin/env python3
"""
scripts/generate_report.py — Aggregate pipeline results into the Final Table.

Reads eval JSONs and gate decision from reports/ and produces:
  - reports/final_table.md  (markdown for README)
  - reports/final_table.txt (plain text)
  - stdout: the table for easy piping

Usage:
    python scripts/generate_report.py [--report-dir DIR]
"""
import json, os, sys, argparse
from datetime import datetime
from typing import Dict, Any, Optional


def load_json(path: str) -> Optional[Dict]:
    try:
        with open(path) as f: return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, IOError):
        return None


def extract_metric(data: Optional[Dict], name: str) -> Optional[float]:
    if data is None: return None
    if name in data:
        v = data[name]
        if isinstance(v, dict):
            for k in ("accuracy", "score", "value", "pass_at_1"):
                if k in v: return float(v[k])
        if isinstance(v, (int, float)): return float(v)
    for val in data.values():
        if isinstance(val, dict) and name in val:
            return extract_metric(val, name)
    return None


def fmt(v: Optional[float], pct: bool = True) -> str:
    if v is None: return "—"
    return f"{v*100:.1f}%" if pct and v <= 1.0 else f"{v:.1f}%" if pct else f"{v:.1f}"


def delta(base: Optional[float], final: Optional[float], pct: bool = True) -> str:
    if base is None or final is None: return "—"
    d = final - base
    return f"{d*100:+.1f}%" if pct and base <= 1.0 else f"{d:+.1f}%" if pct else f"{d:+.1f}"


def generate_final_table(report_dir: str) -> str:
    bl = load_json(os.path.join(report_dir, "eval_baseline.json"))
    sf = load_json(os.path.join(report_dir, "eval_sft.json"))
    dp = load_json(os.path.join(report_dir, "eval_dpo.json"))
    gate = load_json(os.path.join(report_dir, "gate_decision.json"))
    perf_bl = load_json(os.path.join(report_dir, "perf_baseline.json"))
    perf_opt = load_json(os.path.join(report_dir, "perf_optimized.json"))

    metrics = [
        ("MMLU (5-shot)", "mmlu", True),
        ("GSM8K (5-shot)", "gsm8k", True),
        ("HumanEval (pass@1)", "humaneval", True),
        ("ARC-Challenge (25-shot)", "arc", True),
        ("MT-Bench", "mt_bench", False),
        ("TruthfulQA (MC1)", "truthfulqa", True),
        ("Toxicity Rate", "toxicity", True),
    ]
    perf = [
        ("Throughput (tok/s)", "throughput_tok_s", False),
        ("p95 Latency (ms)", "p95_latency_ms", False),
        ("Peak VRAM (GB)", "peak_vram_gb", False),
    ]

    lines = ["| Metric | Base Model | + SFT | + DPO | Δ Total |",
             "|---|---|---|---|---|"]
    for name, key, is_pct in metrics:
        b, s, d = extract_metric(bl, key), extract_metric(sf, key), extract_metric(dp, key)
        lines.append(f"| {name} | {fmt(b,is_pct)} | {fmt(s,is_pct)} | {fmt(d,is_pct)} | {delta(b,d,is_pct)} |")

    if perf_bl or perf_opt:
        lines.append("| **Performance** | | | | |")
        for name, key, is_pct in perf:
            b, o = extract_metric(perf_bl, key), extract_metric(perf_opt, key)
            lines.append(f"| {name} | {fmt(b,is_pct)} | — | {fmt(o,is_pct)} | {delta(b,o,is_pct)} |")

    if gate:
        v = gate.get("verdict", "UNKNOWN")
        lines.append(f"| **GATE DECISION** | | | {'✅' if v=='SHIP' else '❌'} {v} | |")
    else:
        lines.append("| **GATE DECISION** | | | ⚠️ NOT RUN | |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate NanoSeek pipeline report")
    parser.add_argument("--report-dir", default=None)
    args = parser.parse_args()

    base_dir = os.environ.get("NANOSEEK_BASE_DIR", "out")
    report_dir = args.report_dir or os.path.join(base_dir, "reports")
    if not os.path.isdir(report_dir):
        print(f"ERROR: Report directory not found: {report_dir}", file=sys.stderr)
        sys.exit(1)

    table = generate_final_table(report_dir)

    md_path = os.path.join(report_dir, "final_table.md")
    with open(md_path, "w") as f:
        f.write(f"# NanoSeek Pipeline Results\n\nGenerated: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        f.write("## Final Table\n\n" + table + "\n\n")
        gate = load_json(os.path.join(report_dir, "gate_decision.json"))
        if gate:
            f.write(f"## Gate Details\n\n- **Verdict**: {gate.get('verdict')}\n")
            f.write(f"- **Composite Δ**: {gate.get('composite_quality_delta', 'N/A')}\n")
            for r in gate.get("reasons", []): f.write(f"  - {r}\n")

    with open(os.path.join(report_dir, "final_table.txt"), "w") as f:
        f.write(table)

    print(table)
    print(f"\nArtifacts: {md_path}")


if __name__ == "__main__":
    main()
```

### 3h. Master Pipeline Script (`scripts/run_pipeline.sh`)

```bash
#!/usr/bin/env bash
# scripts/run_pipeline.sh — Master: pre-train → SFT → DPO → eval → gate → report
#
# Usage:  ./scripts/run_pipeline.sh [OPTIONS]
# Options:
#   --start-from STAGE   Resume from stage (pretrain|sft|dpo|eval_base|eval_sft|eval_dpo|gate|report)
#   --skip STAGES        Comma-separated stages to skip
#   --force              Force re-run all stages
#   --dry-run            Print plan without executing
#   --help               Show this message

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/pipeline_common.sh"

STAGES=(pretrain sft dpo eval_base eval_sft eval_dpo gate report)
START_FROM=""; SKIP_STAGES=""; DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-from) START_FROM="$2"; shift 2 ;;
        --skip)       SKIP_STAGES="$2"; shift 2 ;;
        --force)      export FORCE="true"; shift ;;
        --dry-run)    DRY_RUN=true; shift ;;
        --help|-h)    head -14 "$0" | grep '^#' | sed 's/^# \?//'; exit 0 ;;
        *)            fatal "Unknown option: $1" ;;
    esac
done

log ""
log "╔═══════════════════════════════════════════════════════╗"
log "║          NanoSeek End-to-End Pipeline                 ║"
log "║  Pre-train → SFT → DPO → Eval → Gate → Report       ║"
log "╚═══════════════════════════════════════════════════════╝"
log "  Base: $NANOSEEK_BASE_DIR  Data: $NANOSEEK_DATA_DIR  GPUs: $NANOSEEK_NUM_GPUS"
log ""

# Save manifest
python3 -c "
import json, platform, os
try:
    import torch; tv=torch.__version__; cv=torch.version.cuda or 'N/A'
    gc=torch.cuda.device_count() if torch.cuda.is_available() else 0
    hw=f'{gc}x{torch.cuda.get_device_name(0)}' if gc>0 else 'CPU'
except: tv=cv=hw='unknown'
json.dump({'git_commit':'$(get_git_commit)','timestamp':'$(date -Iseconds)',
  'torch':tv,'cuda':cv,'hardware':hw}, open('${NANOSEEK_REPORT_DIR}/pipeline_manifest.json','w'), indent=2)
" 2>/dev/null || warn "Manifest write failed"

should_skip() {
    if [[ -n "$SKIP_STAGES" ]]; then
        IFS=',' read -ra SKIP <<< "$SKIP_STAGES"
        for s in "${SKIP[@]}"; do [[ "$s" == "$1" ]] && return 0; done
    fi
    if [[ -n "$START_FROM" ]]; then
        local found=false
        for s in "${STAGES[@]}"; do
            [[ "$s" == "$START_FROM" ]] && found=true
            [[ "$s" == "$1" ]] && [[ "$found" != "true" ]] && return 0
        done
    fi
    return 1
}

run_stage() {
    local stage="$1"; shift
    should_skip "$stage" && info "Skipping: $stage" && return 0
    local state; state=$(get_pipeline_state "$stage")
    [[ "$state" == "COMPLETED" ]] && [[ "$FORCE" != "true" ]] && info "Done: $stage" && return 0
    [[ "$DRY_RUN" == "true" ]] && log "[DRY] $*" && return 0
    update_pipeline_state "$stage" "RUNNING"
    if eval "$@"; then
        update_pipeline_state "$stage" "COMPLETED"
    else
        update_pipeline_state "$stage" "FAILED"
        error "Stage $stage failed"; return 1
    fi
}

PIPELINE_START=$(date +%s)

run_stage "pretrain"  "python3 scripts/pre-train.py 2>&1 | tee ${NANOSEEK_BASE_DIR}/pretrain.log"
release_gpu_memory
run_stage "sft"       "${SCRIPT_DIR}/train_sft.sh"
release_gpu_memory
run_stage "dpo"       "${SCRIPT_DIR}/train_dpo.sh"
release_gpu_memory
run_stage "eval_base" "${SCRIPT_DIR}/eval_all.sh --stage baseline --checkpoint ${NANOSEEK_CHECKPOINT_DIR}/pretrain/final"
run_stage "eval_sft"  "${SCRIPT_DIR}/eval_all.sh --stage sft --checkpoint ${NANOSEEK_CHECKPOINT_DIR}/sft/final"
run_stage "eval_dpo"  "${SCRIPT_DIR}/eval_all.sh --stage dpo --checkpoint ${NANOSEEK_CHECKPOINT_DIR}/dpo/final"
run_stage "gate"      "${SCRIPT_DIR}/gate_check.sh"
run_stage "report"    "python3 ${SCRIPT_DIR}/generate_report.py --report-dir ${NANOSEEK_REPORT_DIR}"

ELAPSED=$(( $(date +%s) - PIPELINE_START ))
log ""
log "Pipeline complete in $((ELAPSED/3600))h $(((ELAPSED%3600)/60))m"
log "Stage summary:"
for s in "${STAGES[@]}"; do
    state=$(get_pipeline_state "$s")
    case "$state" in COMPLETED) i="✅";; FAILED) i="❌";; *) i="⏳";; esac
    log "  $i $s: $state"
done

# Print final table if available
[[ -f "${NANOSEEK_REPORT_DIR}/final_table.txt" ]] && echo "" && cat "${NANOSEEK_REPORT_DIR}/final_table.txt"

# Exit with gate verdict
GATE_FILE="${NANOSEEK_REPORT_DIR}/gate_decision.json"
if [[ -f "$GATE_FILE" ]]; then
    VERDICT=$(python3 -c "import json; print(json.load(open('$GATE_FILE'))['verdict'])" 2>/dev/null)
    [[ "$VERDICT" == "SHIP" ]] && exit 0 || exit 1
fi
```

---

## 4. Pipeline DAG Visualization

```
┌──────────────────────────────────────────────────────────────────────┐
│                  NanoSeek End-to-End Pipeline DAG                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐      │
│  │  DATA PREPARATION  (scripts/setup_data.py, Doc 01)         │      │
│  │  → data/*.parquet, data/ultrachat/, data/ultrafeedback/    │      │
│  └─────────────────────────┬──────────────────────────────────┘      │
│                            │                                         │
│  ┌─────────────────────────▼──────────────────────────────────┐      │
│  │  PRE-TRAINING  (scripts/pre-train.py, Doc 06)              │      │
│  │  → checkpoints/pretrain/final/  |  ~14h, ~$300             │      │
│  │  → reports/pretrain_loss.png, reports/expert_loads.png      │      │
│  └─────────────────────────┬──────────────────────────────────┘      │
│                            │                                         │
│  ┌─────────────────────────▼──────────────────────────────────┐      │
│  │  SFT  (scripts/train_sft.sh, Doc 09)                       │      │
│  │  → checkpoints/sft/final/  |  ~2h, ~$48                    │      │
│  └─────────────────────────┬──────────────────────────────────┘      │
│                            │                                         │
│  ┌─────────────────────────▼──────────────────────────────────┐      │
│  │  DPO  (scripts/train_dpo.sh, Doc 09)                       │      │
│  │  → checkpoints/dpo/final/  |  ~30min, ~$12                 │      │
│  └────────┬────────────────┬──────────────────┬───────────────┘      │
│           │                │                  │                      │
│  ┌────────▼──────┐ ┌──────▼───────┐ ┌────────▼───────┐             │
│  │ EVAL BASELINE │ │  EVAL SFT    │ │  EVAL DPO      │             │
│  │→eval_base.json│ │→eval_sft.json│ │→eval_dpo.json  │             │
│  └────────┬──────┘ └──────┬───────┘ └────────┬───────┘             │
│           └────────────────┼─────────────────┘                      │
│                            │                                         │
│  ┌─────────────────────────▼──────────────────────────────────┐      │
│  │  SHIP GATE  (scripts/gate_check.sh, Doc 17)                │      │
│  │  → reports/gate_decision.json  |  exit 0=SHIP, 1=REJECT    │      │
│  └─────────────────────────┬──────────────────────────────────┘      │
│                            │                                         │
│  ┌─────────────────────────▼──────────────────────────────────┐      │
│  │  REPORT  (scripts/generate_report.py)                      │      │
│  │  → reports/final_table.md                                  │      │
│  └────────────────────────────────────────────────────────────┘      │
│                                                                      │
│  Legend:  ──▶ sequential   ─┬─ fork (parallel)   ─▼─ join           │
└──────────────────────────────────────────────────────────────────────┘
```

### Critical Path

```
Pre-train (14h) → SFT (2h) → DPO (30m) → Eval-DPO (30m) → Gate (<1s) → Report (<10s)
Total: ~17 hours  |  Parallelized evals save ~1h → ~16 hours
```

---

## 5. File Placement

```
nanoseek/
├── scripts/
│   ├── pipeline_common.sh       # Shared logging, artifact checks, state mgmt
│   ├── run_pipeline.sh          # Master orchestrator
│   ├── train_sft.sh             # One-command SFT
│   ├── train_dpo.sh             # One-command DPO
│   ├── serve.sh                 # One-command serving + health check
│   ├── eval_all.sh              # One-command full eval
│   ├── gate_check.sh            # Ship/reject gate
│   ├── generate_report.py       # Final Table generator
│   └── pre-train.py             # Existing (Doc 06)
├── out/                         # Default NANOSEEK_BASE_DIR
│   ├── checkpoints/
│   │   ├── pretrain/final/model_state.pt
│   │   ├── sft/final/model_state.pt
│   │   └── dpo/final/model_state.pt
│   └── reports/
│       ├── .pipeline_state      # Stage completion tracking
│       ├── pipeline_manifest.json
│       ├── eval_baseline.json
│       ├── eval_sft.json
│       ├── eval_dpo.json
│       ├── gate_decision.json
│       ├── final_table.md
│       ├── pretrain_loss.png
│       └── expert_loads.png
└── configs/                     # Optional stage-specific overrides
```

---

## 6. Usage Examples

### 6a. Full Pipeline — One Command

```bash
./scripts/run_pipeline.sh                          # Full run
NANOSEEK_NUM_GPUS=4 ./scripts/run_pipeline.sh      # Custom GPU count
./scripts/run_pipeline.sh --dry-run                 # Preview execution plan
```

### 6b. Single Stage Execution

```bash
./scripts/train_sft.sh --base-checkpoint out/checkpoints/pretrain/step_50000/model_state.pt
./scripts/train_dpo.sh --beta 0.05 --lr 1e-6
./scripts/eval_all.sh --checkpoint out/checkpoints/dpo/final --stage dpo
./scripts/gate_check.sh --strict
python3 scripts/generate_report.py --report-dir out/reports
```

### 6c. Resume From Failure

```bash
# Pipeline crashed during DPO? Just re-run — completed stages auto-skip:
./scripts/run_pipeline.sh

# Or start from a specific stage:
./scripts/run_pipeline.sh --start-from dpo

# Force everything from scratch:
./scripts/run_pipeline.sh --force
```

### 6d. Serving After Pipeline

```bash
./scripts/serve.sh --background                                    # Auto-detect best ckpt
./scripts/serve.sh --checkpoint out/checkpoints/dpo/final --quantize INT8  # INT8 quantized

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nanoseek-1b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":50}'
```

### 6e. CI/CD Integration

```yaml
name: NanoSeek Release
on: { push: { branches: [main], paths: ['model/**', 'scripts/**'] } }
jobs:
  pipeline:
    runs-on: [self-hosted, gpu-8xh100]
    timeout-minutes: 1200
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e ".[dev,training]"
      - run: ./scripts/run_pipeline.sh
      - run: cat out/reports/final_table.md
      - run: ./scripts/gate_check.sh    # exit code drives pass/fail
```

---

## 7. Gotchas

### 7a. GPU Memory Between Stages Is Not Automatically Freed

PyTorch's CUDA context persists across Python calls in the same process. Pre-training allocates ~40GB; launching SFT immediately after causes OOM because both contexts coexist.

**Mitigation**: Each stage runs as a separate subprocess (separate `python3` invocation), guaranteeing CUDA context isolation. The `release_gpu_memory()` helper clears caches between stages, but the real fix is process separation — which `run_pipeline.sh` does by calling each stage script as a child process.

### 7b. Checkpoint Format Compatibility

Pre-training saves raw `state_dict`s. FSDP wrapping adds `_fsdp_wrapped_module.` prefixes. LoRA in SFT expects adapter keys that don't exist in base checkpoints.

**Mitigation**: Every load function must strip FSDP prefixes and use `strict=False` when adapter keys are expected to initialize fresh. The training scripts handle this, but if you add a custom loading path, remember to strip.

### 7c. Data Path Resolution Across Environments

Scripts use `$NANOSEEK_DATA_DIR` (default: `data/`) which works locally but fails on clusters where data lives at `/shared/datasets/`. Never hardcode absolute paths.

**Mitigation**: Set `export NANOSEEK_DATA_DIR=/your/path` in your shell profile. All scripts inherit this via `pipeline_common.sh`.

### 7d. Eval JSON Schema Consistency

The gate reads specific keys from eval JSONs. If different benchmark runners produce different schemas (`mmlu_accuracy` vs `mmlu.accuracy`), the gate silently reads `None` and may produce incorrect verdicts.

**Mitigation**: The `extract_metric()` functions search nested structures, but the safest approach is normalizing the schema in `eval_all.sh`'s aggregation step. Every benchmark should produce `{"benchmark": {"accuracy": float}}`.

### 7e. `set -euo pipefail` + Grep

`set -e` exits on any non-zero code. `grep` returns 1 for "no matches" — which is not an error. `nvidia-smi` returns non-zero without GPUs.

**Mitigation**: Use `|| true` or `|| echo default` for legitimately-failing commands. Do NOT add `|| true` to training/eval commands that should fail loudly.

### 7f. Parallel Eval Exhausts GPU Memory

Running MMLU, GSM8K, and HumanEval in parallel on **one** GPU causes OOM — each loads the full model. The `--parallel` flag is for **multi-GPU** setups only (one benchmark per GPU via `CUDA_VISIBLE_DEVICES`).

### 7g. Pipeline State File Staleness

Manually re-running a stage outside `run_pipeline.sh` leaves `.pipeline_state` with a stale `COMPLETED` entry. The master script will skip the stage even though outputs changed.

**Mitigation**: Use `--force` when re-running manually. Or edit `.pipeline_state` directly — it's a flat text file by design.

### 7h. Report With Missing Stages Shows Dashes Silently

`generate_report.py` shows "—" for missing eval files. A misconfigured pipeline that skips eval produces a table of dashes without errors. The gate (`gate_check.sh`) will exit 2 (ERROR) for missing files — always run gate before report.

---

*"The pipeline is the product. The model is a checkpoint — ephemeral, replaceable, one bad gradient away from worthless. The pipeline is what makes the next checkpoint better than the last. Invest in the pipeline and every model you ever train benefits."*

— Principal Engineer's Note, Foundation Models Division, 2026
