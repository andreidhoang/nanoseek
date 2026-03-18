#!/bin/bash
# ============================================================================
# Day 1: Anchor-Scale Ablations (~4 hours on 1× A6000)
# ============================================================================
# Prerequisites:
#   - setup_runpod.sh has been run
#   - pytest passes
#   - W&B is logged in
#
# Execution order:
#   1. Gate 1 smoke test (100 steps, ~2 min)
#   2. HP grid search (12 runs, ~2.5 hrs)
#   3. Stability ablations A/C/D/E/F (5 runs, ~1 hr)
#   4. Architecture ablations (3 runs, ~45 min)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."  # nanoseek project root

echo "============================================"
echo "PHASE 1: Gate 1 Smoke Test"
echo "============================================"

python -m nanoseek.scripts.pre_train \
    --run "gate1-smoke" \
    --scale anchor \
    --num-iterations 100 \
    --eval-every 50 \
    --save-every 100 \
    --device-batch-size 16 \
    --seed 42

echo ""
echo "Gate 1 complete. Check W&B for:"
echo "  ✅ H_load > 4 bits at step 0"
echo "  ✅ MTP acceptance ~50% at step 0"
echo "  ✅ FIM fraction ~10%"
echo "  ✅ No NaN/Inf"
echo ""
read -p "Gate 1 passed? Press Enter to continue or Ctrl+C to abort..."

echo "============================================"
echo "PHASE 2: HP Grid Search (12 runs)"
echo "============================================"

for mlr in 0.005 0.01 0.02 0.04; do
  for elr in 0.1 0.3 0.5; do
    echo "--- Launching: matrix_lr=${mlr}, embedding_lr=${elr} ---"
    python -m nanoseek.scripts.pre_train \
        --run "hp-anchor-mlr${mlr}-elr${elr}" \
        --scale anchor \
        --matrix-lr $mlr \
        --embedding-lr $elr \
        --eval-every 100 \
        --save-every -1 \
        --seed 42
  done
done

echo ""
echo "HP grid complete. Pick best (matrix_lr, embedding_lr) from W&B."
echo ""

echo "============================================"
echo "PHASE 3: Stability Ablations (5 runs)"
echo "============================================"

# stab-A: Baseline (all V3.2 techniques ON)
python -m nanoseek.scripts.pre_train \
    --run "stab-A-baseline" \
    --scale anchor \
    --seed 42

# stab-C: No seq_aux
python -m nanoseek.scripts.pre_train \
    --run "stab-C-no-seq-aux" \
    --scale anchor \
    --no-seq-aux \
    --seed 42

# stab-D: No grad clipping
python -m nanoseek.scripts.pre_train \
    --run "stab-D-no-gradclip" \
    --scale anchor \
    --no-grad-clip \
    --seed 42

# stab-E: Classic aux loss
python -m nanoseek.scripts.pre_train \
    --run "stab-E-classic-aux" \
    --scale anchor \
    --aux-loss-type classic \
    --seed 42

# stab-F: Bad batch injection at step 1500
python -m nanoseek.scripts.pre_train \
    --run "stab-F-bad-batch" \
    --scale anchor \
    --inject-bad-batch 1500 \
    --seed 42

echo ""
echo "Stability ablations complete."
echo ""

echo "============================================"
echo "PHASE 4: Architecture Ablations (3 runs)"
echo "============================================"

# arch-no-mtp
python -m nanoseek.scripts.pre_train \
    --run "arch-no-mtp" \
    --scale anchor \
    --no-mtp \
    --seed 42

# arch-no-shared
python -m nanoseek.scripts.pre_train \
    --run "arch-no-shared" \
    --scale anchor \
    --no-shared-experts \
    --seed 42

# arch-fewer-experts (16 experts, top-2)
python -m nanoseek.scripts.pre_train \
    --run "arch-fewer-experts" \
    --scale anchor \
    --num-experts 16 --top-k 2 --n-group 4 --topk-group 2 \
    --seed 42

echo ""
echo "============================================"
echo "DAY 1 COMPLETE"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Analyze HP grid → pick best (matrix_lr, embedding_lr)"
echo "  2. Compare stab-A vs C/D/E: check pre-registered hypotheses"
echo "  3. Compare arch ablations vs stab-A baseline"
echo "  4. Proceed to Day 2: hp-500m-transfer"
