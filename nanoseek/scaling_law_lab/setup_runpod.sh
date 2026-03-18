#!/bin/bash
# ============================================================================
# NanoSeek RunPod Environment Setup
# ============================================================================
#
# Run ONCE after pod starts. Handles:
#   1. System dependencies
#   2. Repo clone + package install
#   3. GPU verification
#   4. Training data download
#   5. W&B authentication
#   6. Test suite verification
#
# Usage:
#   # First time setup (downloads everything)
#   bash setup_runpod.sh --full
#
#   # Quick setup (skip data download — use existing volume data)
#   bash setup_runpod.sh
#
#   # Specify number of data shards (more shards = more training data)
#   bash setup_runpod.sh --full --num-shards 170
#
# Prerequisites:
#   - RunPod pod with NVIDIA GPU (A6000/H100)
#   - Network volume mounted at /workspace
#   - WANDB_API_KEY environment variable set
#
# Expected pod env vars (set in RunPod template):
#   WANDB_API_KEY=<your-key>
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#
# ============================================================================

set -euo pipefail

# ── Parse arguments ──
FULL_SETUP=false
NUM_SHARDS=170  # ~170 shards is enough for anchor + 500M training
REPO_URL="${NANOSEEK_REPO_URL:-https://github.com/andreidhoang/nanoseek.git}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full) FULL_SETUP=true; shift ;;
        --num-shards) NUM_SHARDS="$2"; shift 2 ;;
        --repo) REPO_URL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "  NanoSeek RunPod Setup"
echo "  Full setup: ${FULL_SETUP}"
echo "  Data shards: ${NUM_SHARDS}"
echo "============================================"
echo ""

# ── Step 1: System dependencies ──
echo "[1/6] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq git htop nvtop tmux > /dev/null 2>&1
echo "  Done."

# ── Step 2: Clone repo + install packages ──
echo "[2/6] Setting up repository..."
cd /workspace

if [ ! -d "nanoseek" ]; then
    if [ -z "${REPO_URL}" ]; then
        echo "  ERROR: No repo found at /workspace/nanoseek and NANOSEEK_REPO_URL not set."
        echo "  Either:"
        echo "    1. Set NANOSEEK_REPO_URL env var in RunPod template"
        echo "    2. Run: bash setup_runpod.sh --repo <your-git-url>"
        echo "    3. Manually clone: cd /workspace && git clone <url> nanoseek"
        exit 1
    fi
    echo "  Cloning from ${REPO_URL}..."
    git clone "${REPO_URL}" nanoseek
else
    echo "  Repo already exists at /workspace/nanoseek"
    cd nanoseek && git pull && cd /workspace
fi

cd /workspace/nanoseek

# Install Python dependencies
echo "  Installing Python packages..."
pip install --upgrade pip -q

# PyTorch with CUDA 12.4 (matches RunPod's CUDA driver)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q

# Training dependencies
pip install wandb pyarrow scipy filelock requests -q

# nanochat (sibling package — needed for tokenizer)
# Install in development mode so imports work
pip install -e nanochat/ -q 2>/dev/null || echo "  nanochat editable install skipped (may need manual setup)"

# nanoseek itself
pip install -e nanoseek/ -q

echo "  Done."

# ── Step 3: Verify GPU ──
echo "[3/6] Verifying GPU..."
python3 -c "
import torch
n_gpus = torch.cuda.device_count()
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  GPU count: {n_gpus}')
for i in range(n_gpus):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_mem / 1e9
    print(f'  GPU {i}: {name} ({mem:.1f} GB)')
print(f'  BF16 supported: {torch.cuda.is_bf16_supported()}')
if not torch.cuda.is_available():
    raise RuntimeError('No CUDA device found!')
if not torch.cuda.is_bf16_supported():
    print('  WARNING: BF16 not supported — training will be slower in FP32')
"

# ── Step 4: Download training data ──
if [ "${FULL_SETUP}" = true ]; then
    echo "[4/6] Downloading training data (${NUM_SHARDS} shards)..."
    echo "  This downloads from HuggingFace (ClimbMix-400B)."
    echo "  Each shard is ~60MB. Total: ~$((NUM_SHARDS * 60 / 1024)) GB"
    echo "  Data dir: $(python3 -c 'from nanoseek.nanoseek.common import get_base_dir; print(get_base_dir())')/base_data_climbmix/"

    # Set base dir to /workspace so data persists across pod restarts
    export NANOCHAT_BASE_DIR="/workspace/data"

    cd /workspace/nanoseek
    python3 -m nanoseek.nanoseek.dataset -n "${NUM_SHARDS}" -w 8

    echo "  Data download complete."
else
    echo "[4/6] Skipping data download (use --full for first-time setup)"

    # Check if data already exists
    DATA_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}/base_data_climbmix"
    if [ -d "${DATA_DIR}" ]; then
        SHARD_COUNT=$(ls "${DATA_DIR}"/*.parquet 2>/dev/null | wc -l)
        echo "  Found ${SHARD_COUNT} existing shards at ${DATA_DIR}"
    else
        echo "  WARNING: No data found. Run with --full to download."
    fi
fi

# ── Step 5: W&B authentication ──
echo "[5/6] Setting up W&B..."
if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login "${WANDB_API_KEY}" 2>/dev/null
    echo "  W&B authenticated."
else
    echo "  WARNING: WANDB_API_KEY not set. Runs will use --run dummy (no logging)."
    echo "  Set it in RunPod template or run: export WANDB_API_KEY=<your-key>"
fi

# ── Step 6: Run test suite ──
echo "[6/6] Running test suite..."
cd /workspace/nanoseek/nanoseek
python3 -m pytest tests/ -v --tb=short 2>&1 | tail -10

echo ""
echo "============================================"
echo "  SETUP COMPLETE"
echo "============================================"
echo ""
echo "  Data dir: ${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}/base_data_climbmix/"
echo "  Checkpoints: /workspace/nanoseek/nanoseek/checkpoints/"
echo "  W&B project: nanoseek"
echo ""
echo "  Quick commands:"
echo "    # Gate 1 smoke test"
echo "    cd /workspace/nanoseek/nanoseek"
echo "    python -m nanoseek.scripts.pre_train --run gate1-smoke --scale anchor --num-iterations 100 --eval-every 50 --save-every 100 --seed 42"
echo ""
echo "    # Full Day 1 pipeline (dry run first)"
echo "    python -m nanoseek.scripts.run_phase3 --dry-run --stop-after day1"
echo ""
echo "    # Full Day 1 pipeline (real)"
echo "    python -m nanoseek.scripts.run_phase3 --stop-after day1"
echo ""
echo "  IMPORTANT: Set NANOCHAT_BASE_DIR=/workspace/data in your shell"
echo "  to ensure data persists on the network volume."
echo ""
