#!/bin/bash
# NanoSeek Phase 3: Ablation Campaign Launcher
#
# Usage:
#   ./scaling_law_lab/launch_ablations.sh gate1          # Gate 1 smoke test
#   ./scaling_law_lab/launch_ablations.sh hp-grid        # HP grid search (12 runs)
#   ./scaling_law_lab/launch_ablations.sh stability      # Stability ablations (5 runs)
#   ./scaling_law_lab/launch_ablations.sh architecture   # Architecture ablations (4 runs)
#   ./scaling_law_lab/launch_ablations.sh 500m-transfer  # 500M validation (1 run)
#   ./scaling_law_lab/launch_ablations.sh day1           # All Day 1 runs
#
# Each run logs to W&B with proper group/tags for automated analysis.
# Checkpoint dirs are per-run (no overwrites).

set -euo pipefail

PHASE="${1:-help}"
SEED="${SEED:-42}"
SCALE="${SCALE:-anchor}"

echo "═══════════════════════════════════════════════════"
echo "  NanoSeek Phase 3 Ablation Launcher"
echo "  Phase: ${PHASE} | Scale: ${SCALE} | Seed: ${SEED}"
echo "═══════════════════════════════════════════════════"

case "${PHASE}" in

  gate1)
    echo "▶ Gate 1: 100-step smoke test"
    python -m nanoseek.scripts.pre_train \
        --run "gate1-smoke" \
        --scale anchor \
        --num-iterations 100 \
        --eval-every 50 \
        --save-every 100 \
        --device-batch-size 16 \
        --seed "${SEED}"
    ;;

  hp-grid)
    echo "▶ HP Grid Search: 12 runs at anchor scale"
    for mlr in 0.005 0.01 0.02 0.04; do
      for elr in 0.1 0.3 0.5; do
        echo "  Launching hp-anchor-mlr${mlr}-elr${elr}..."
        python -m nanoseek.scripts.pre_train \
            --run "hp-anchor-mlr${mlr}-elr${elr}" \
            --scale anchor \
            --matrix-lr "${mlr}" \
            --embedding-lr "${elr}" \
            --eval-every 100 \
            --save-every -1 \
            --seed "${SEED}"
      done
    done
    echo "▶ HP Grid complete. Run analyze_ablations.py --group hp-anchor --hp-transfer"
    ;;

  stability)
    echo "▶ Stability Ablations: 5 runs at anchor scale"

    echo "  [1/5] stab-A: Full V3.2 baseline"
    python -m nanoseek.scripts.pre_train \
        --run "stab-A-baseline" \
        --scale anchor \
        --eval-every 50 \
        --seed "${SEED}"

    echo "  [2/5] stab-C: No seq_aux"
    python -m nanoseek.scripts.pre_train \
        --run "stab-C-no-seq-aux" \
        --scale anchor \
        --no-seq-aux \
        --eval-every 50 \
        --seed "${SEED}"

    echo "  [3/5] stab-D: No grad clipping"
    python -m nanoseek.scripts.pre_train \
        --run "stab-D-no-gradclip" \
        --scale anchor \
        --no-grad-clip \
        --eval-every 50 \
        --seed "${SEED}"

    echo "  [4/5] stab-E: Classic aux loss"
    python -m nanoseek.scripts.pre_train \
        --run "stab-E-classic-aux" \
        --scale anchor \
        --aux-loss-type classic \
        --eval-every 50 \
        --seed "${SEED}"

    echo "  [5/5] stab-F: Bad batch injection at step 1500"
    python -m nanoseek.scripts.pre_train \
        --run "stab-F-bad-batch" \
        --scale anchor \
        --inject-bad-batch 1500 \
        --eval-every 50 \
        --seed "${SEED}"

    echo "▶ Stability ablations complete. Run analyze_ablations.py --group stability-anchor"
    ;;

  architecture)
    echo "▶ Architecture Ablations: 4 runs at anchor scale"

    echo "  [1/4] arch-no-mtp: MTP disabled"
    python -m nanoseek.scripts.pre_train \
        --run "arch-no-mtp" \
        --scale anchor \
        --no-mtp \
        --eval-every 50 \
        --seed "${SEED}"

    echo "  [2/4] arch-no-shared: Shared experts removed"
    python -m nanoseek.scripts.pre_train \
        --run "arch-no-shared" \
        --scale anchor \
        --no-shared-experts \
        --eval-every 50 \
        --seed "${SEED}"

    echo "  [3/4] arch-fewer-experts: 16 experts, top-2"
    python -m nanoseek.scripts.pre_train \
        --run "arch-fewer-experts" \
        --scale anchor \
        --num-experts 16 \
        --top-k 2 \
        --n-group 4 \
        --topk-group 2 \
        --eval-every 50 \
        --seed "${SEED}"

    echo "  [4/4] arch-no-mla: SKIPPED (MHA fallback not yet implemented)"
    # python -m nanoseek.scripts.pre_train \
    #     --run "arch-no-mla" \
    #     --scale anchor \
    #     --no-mla \
    #     --eval-every 50 \
    #     --seed "${SEED}"

    echo "▶ Architecture ablations complete. Run analyze_ablations.py --group architecture-anchor"
    ;;

  500m-transfer)
    echo "▶ 500M Transfer Validation"
    echo "  IMPORTANT: Update --matrix-lr and --embedding-lr with best anchor HP first!"
    python -m nanoseek.scripts.pre_train \
        --run "hp-500m-transfer" \
        --scale 500m \
        --matrix-lr "${BEST_MLR:-0.02}" \
        --embedding-lr "${BEST_ELR:-0.3}" \
        --eval-every 500 \
        --save-every 2000 \
        --seed "${SEED}"
    ;;

  day1)
    echo "▶ Running all Day 1 ablations (gate1 → hp-grid → stability → architecture)"
    bash "$0" gate1
    bash "$0" hp-grid
    bash "$0" stability
    bash "$0" architecture
    echo "▶ Day 1 complete! Total time and cost in W&B."
    echo "  Next: python -m nanoseek.scripts.analyze_ablations --full-report"
    ;;

  help|*)
    echo "Usage: $0 {gate1|hp-grid|stability|architecture|500m-transfer|day1}"
    echo ""
    echo "  gate1          Gate 1 smoke test (100 steps, ~2 min)"
    echo "  hp-grid        HP grid search (12 runs, ~2.6 hrs)"
    echo "  stability      Stability ablations (5 runs, ~1.1 hrs)"
    echo "  architecture   Architecture ablations (4 runs, ~52 min)"
    echo "  500m-transfer  500M muP validation (1 run, ~14 hrs)"
    echo "  day1           All Day 1 runs sequentially"
    echo ""
    echo "Environment variables:"
    echo "  SEED=42        Random seed (default: 42)"
    echo "  SCALE=anchor   Model scale (default: anchor)"
    echo "  BEST_MLR=0.02  Best matrix LR from HP grid (for 500m-transfer)"
    echo "  BEST_ELR=0.3   Best embedding LR from HP grid (for 500m-transfer)"
    ;;

esac
