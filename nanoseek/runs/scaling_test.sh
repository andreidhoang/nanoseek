#!/bin/bash
# NanoSeek DDP Scaling Test — measure 1/2/4/8 GPU throughput
#
# Usage: ./runs/scaling_test.sh [scale] [device_batch_size]
# Example: ./runs/scaling_test.sh ablation 4
#
# Run from nanoseek/ directory (NOT nanoseek/nanoseek/)

set -euo pipefail

SCALE="${1:-ablation}"
DBS="${2:-4}"
STEPS=30

echo "=============================================="
echo "NanoSeek Scaling Test: ${SCALE}, DBS=${DBS}"
echo "=============================================="
echo ""

for NGPU in 1 2 4 8; do
    # Check if we have enough GPUs
    AVAIL=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$NGPU" -gt "$AVAIL" ]; then
        echo "--- ${NGPU} GPU(s): SKIPPED (only ${AVAIL} available) ---"
        echo ""
        continue
    fi

    echo "--- ${NGPU} GPU(s) ---"
    if [ "$NGPU" -eq 1 ]; then
        python -m nanoseek.scripts.pre_train \
            --run "scale-${NGPU}gpu" --scale "$SCALE" --seed 42 \
            --num-iterations $STEPS --eval-every -1 --save-every -1 \
            --device-batch-size "$DBS" 2>&1 | grep -E "(step 00(1[5-9]|2[0-9]|30)|mfu)" | tail -5
    else
        torchrun --nproc_per_node=$NGPU -m nanoseek.scripts.pre_train \
            --run "scale-${NGPU}gpu" --scale "$SCALE" --seed 42 \
            --num-iterations $STEPS --eval-every -1 --save-every -1 \
            --device-batch-size "$DBS" 2>&1 | grep -E "(step 00(1[5-9]|2[0-9]|30)|mfu)" | tail -5
    fi
    echo ""
done

echo "=============================================="
echo "Compare tok/s across GPU counts for scaling efficiency:"
echo "  efficiency = tok/s(N) / (N * tok/s(1))"
echo "  Target: >85% with NVLink"
echo "=============================================="
