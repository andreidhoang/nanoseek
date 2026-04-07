#!/bin/bash
# NanoSeek CPU/MPS demo: exercise code paths on a Mac
# Adapted from nanochat/runs/runcpu.sh for MoE + MLA + MTP architecture
#
# Usage: bash runs/runcpu.sh
# Run from nanoseek/ directory (NOT nanoseek/nanoseek/)
#
# NOTE: MoE models are expensive. This is educational/debugging only.
# 64 experts × top-8 means even the smallest config has ~1B total params in memory.

set -euo pipefail

# ─── Setup ───
export NANOSEEK_BASE_DIR="${NANOSEEK_BASE_DIR:-$HOME/.cache/nanoseek}"
mkdir -p "$NANOSEEK_BASE_DIR"

# Python environment: use existing venv or create + install deps
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev,training]"
fi

if [ -z "${WANDB_RUN:-}" ]; then
    WANDB_RUN=dummy
fi

# ─── Download a few data shards (~2 min) ───
echo "=== Downloading data shards ==="
python -m nanoseek.nanoseek.dataset -n 4

# ─── Train tokenizer if not already built (~30s on M3 Max) ───
TOKENIZER_DIR="$(python -c 'from nanoseek.nanoseek.tokenizer import _default_tokenizer_dir; print(_default_tokenizer_dir())')"
if [ ! -f "$TOKENIZER_DIR/tokenizer.pkl" ]; then
    echo ""
    echo "=== Training tokenizer (vocab=32768) ==="
    python -m nanoseek.scripts.tok_train --max-chars=500000000
else
    echo "=== Tokenizer already exists at $TOKENIZER_DIR ==="
fi

# ─── Train smallest MoE model (d12 = 240M active / 967M total) ───
# d12: H=1024, 8 heads, 12 layers
# ~30 min on M3 Max with MPS, longer on CPU
# --no-compile: torch.compile not supported on MPS
# --device-batch-size 2: MoE needs more memory than dense (64 experts in memory)
# --total-batch-size 8192: small batch for fast iterations
# --target-param-data-ratio -1: we set iterations explicitly
echo ""
echo "=== Training d12 (240M active, 967M total) ==="
python -m nanoseek.scripts.pre_train \
    --depth=12 \
    --num-iterations=500 \
    --device-batch-size=2 \
    --total-batch-size=8192 \
    --eval-every=50 \
    --eval-tokens=524288 \
    --save-every=-1 \
    --no-compile \
    --seed=42 \
    --run="$WANDB_RUN"

echo ""
echo "=== Done ==="
echo "Check wandb (if enabled) or terminal output for:"
echo "  - val_bpb should decrease (learning signal)"
echo "  - H_load should be ~4.5-5.5 (healthy expert balance)"
echo "  - I_spec should increase (expert specialization emerging)"
