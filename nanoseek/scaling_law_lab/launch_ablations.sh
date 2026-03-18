#!/bin/bash
# NanoSeek Phase 3: Ablation Campaign Launcher
#
# Usage:
#   ./scaling_law_lab/launch_ablations.sh gate1          # Gate 1 smoke test
#   ./scaling_law_lab/launch_ablations.sh hp-grid        # HP Round 1: coarse grid (12 runs)
#   ./scaling_law_lab/launch_ablations.sh hp-round2      # HP Round 2: fine grid (9 runs)
#   ./scaling_law_lab/launch_ablations.sh hp-seeds       # HP Round 3: multi-seed top-3 (9 runs)
#   ./scaling_law_lab/launch_ablations.sh hp-wd          # HP Round 4: weight decay check (2 runs)
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
    # ═══════════════════════════════════════════════════════════════
    # Round 1: Coarse grid (12 runs, single seed)
    # matrix_lr:    [0.005, 0.01, 0.02, 0.04] — geometric 2× spacing
    #   Range: Muon typically 0.01-0.05 (Newton-style); 8× span covers ±2× of sweet spot
    # embedding_lr: [0.1, 0.3, 0.5] — embeddings need aggressive LR (muP lookup table)
    # Why these values: see EXPERIMENT_REASONING.md Section 2 (Exp 2)
    # ═══════════════════════════════════════════════════════════════
    echo "▶ HP Grid Round 1 (Coarse): 12 runs at anchor scale"
    for mlr in 0.005 0.01 0.02 0.04; do
      for elr in 0.1 0.3 0.5; do
        echo "  Launching hp-r1-mlr${mlr}-elr${elr}..."
        python -m nanoseek.scripts.pre_train \
            --run "hp-r1-mlr${mlr}-elr${elr}" \
            --scale anchor \
            --matrix-lr "${mlr}" \
            --embedding-lr "${elr}" \
            --eval-every 100 \
            --save-every -1 \
            --seed "${SEED}"
      done
    done
    echo "▶ Round 1 complete."
    echo "  Next: python -m nanoseek.scripts.analyze_ablations --group hp-anchor --hp-transfer"
    echo "  Then: BEST_MLR=X BEST_ELR=Y ./launch_ablations.sh hp-round2"
    ;;

  hp-round2)
    # ═══════════════════════════════════════════════════════════════
    # Round 2: Fine grid around Round 1 winner (9 runs, single seed)
    # Zooms ±25% around the best (matrix_lr, embedding_lr) from Round 1.
    # Why: Round 1's 2× jumps may miss the actual optimum. Round 2
    # with 1.25× spacing finds it within ~12% accuracy.
    # ═══════════════════════════════════════════════════════════════
    if [[ -z "${BEST_MLR:-}" || -z "${BEST_ELR:-}" ]]; then
      echo "ERROR: Set BEST_MLR and BEST_ELR from Round 1 results first."
      echo "  Example: BEST_MLR=0.02 BEST_ELR=0.3 $0 hp-round2"
      exit 1
    fi
    echo "▶ HP Grid Round 2 (Fine): 9 runs around mlr=${BEST_MLR}, elr=${BEST_ELR}"
    # Generate ±25% grid: [0.75×best, 1.0×best, 1.25×best]
    for mlr_mult in 0.75 1.0 1.25; do
      for elr_mult in 0.75 1.0 1.25; do
        mlr=$(python3 -c "print(round(${BEST_MLR} * ${mlr_mult}, 6))")
        elr=$(python3 -c "print(round(${BEST_ELR} * ${elr_mult}, 4))")
        echo "  Launching hp-r2-mlr${mlr}-elr${elr}..."
        python -m nanoseek.scripts.pre_train \
            --run "hp-r2-mlr${mlr}-elr${elr}" \
            --scale anchor \
            --matrix-lr "${mlr}" \
            --embedding-lr "${elr}" \
            --eval-every 100 \
            --save-every -1 \
            --seed "${SEED}"
      done
    done
    echo "▶ Round 2 complete."
    echo "  Next: python -m nanoseek.scripts.analyze_ablations --group hp-anchor --hp-transfer"
    echo "  Then: BEST_MLR=X BEST_ELR=Y $0 hp-seeds"
    ;;

  hp-seeds)
    # ═══════════════════════════════════════════════════════════════
    # Round 3: Multi-seed validation of top-3 configs (9 runs)
    # Why: A single seed can't distinguish signal from noise. A 0.02 BPB
    # difference could be real or random. 3 seeds gives mean ± std so we
    # can tell if the winner is statistically significant.
    # Practical: 3 seeds × 3 configs = 9 runs (not 3 seeds × 21 configs)
    # ═══════════════════════════════════════════════════════════════
    if [[ -z "${TOP3_CONFIGS:-}" ]]; then
      echo "ERROR: Set TOP3_CONFIGS as semicolon-separated 'mlr,elr' pairs."
      echo "  Example: TOP3_CONFIGS='0.02,0.3;0.025,0.3;0.02,0.225' $0 hp-seeds"
      exit 1
    fi
    echo "▶ HP Multi-Seed Validation: top-3 configs × 3 seeds"
    IFS=';' read -ra CONFIGS <<< "${TOP3_CONFIGS}"
    for config_pair in "${CONFIGS[@]}"; do
      IFS=',' read -r mlr elr <<< "${config_pair}"
      for seed in 42 137 2024; do
        echo "  Launching hp-seed-mlr${mlr}-elr${elr}-s${seed}..."
        python -m nanoseek.scripts.pre_train \
            --run "hp-seed-mlr${mlr}-elr${elr}-s${seed}" \
            --scale anchor \
            --matrix-lr "${mlr}" \
            --embedding-lr "${elr}" \
            --eval-every 100 \
            --save-every -1 \
            --seed "${seed}"
      done
    done
    echo "▶ Multi-seed validation complete."
    echo "  Next: python -m nanoseek.scripts.analyze_ablations --group hp-anchor --hp-transfer --multiseed"
    echo "  Then: BEST_MLR=X BEST_ELR=Y $0 hp-wd"
    ;;

  hp-wd)
    # ═══════════════════════════════════════════════════════════════
    # Round 4: Weight decay check on confirmed winner (2 runs)
    # Why: Muon's interaction with weight decay is non-trivial.
    # WD=0.1 is the default but WD=0.0 may work better with Muon's
    # Newton-style updates. Check both extremes against the winner.
    # ═══════════════════════════════════════════════════════════════
    if [[ -z "${BEST_MLR:-}" || -z "${BEST_ELR:-}" ]]; then
      echo "ERROR: Set BEST_MLR and BEST_ELR from multi-seed winner."
      echo "  Example: BEST_MLR=0.02 BEST_ELR=0.3 $0 hp-wd"
      exit 1
    fi
    echo "▶ HP Weight Decay Check: 2 runs at winner config"
    for wd in 0.0 0.3; do
      echo "  Launching hp-wd-mlr${BEST_MLR}-elr${BEST_ELR}-wd${wd}..."
      python -m nanoseek.scripts.pre_train \
          --run "hp-wd-mlr${BEST_MLR}-elr${BEST_ELR}-wd${wd}" \
          --scale anchor \
          --matrix-lr "${BEST_MLR}" \
          --embedding-lr "${BEST_ELR}" \
          --weight-decay "${wd}" \
          --eval-every 100 \
          --save-every -1 \
          --seed "${SEED}"
    done
    echo "▶ Weight decay check complete."
    echo "  Compare against hp-seed-mlr${BEST_MLR}-elr${BEST_ELR}-s42 (WD=0.1 baseline)"
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
    echo "▶ Running Day 1 ablations (gate1 → hp-grid R1 → stability → architecture)"
    bash "$0" gate1
    bash "$0" hp-grid
    bash "$0" stability
    bash "$0" architecture
    echo "▶ Day 1 complete! Total time and cost in W&B."
    echo "  Next steps (sequential — each depends on prior results):"
    echo "  1. python -m nanoseek.scripts.analyze_ablations --group hp-anchor --hp-transfer"
    echo "     → Get BEST_MLR, BEST_ELR from Round 1"
    echo "  2. BEST_MLR=X BEST_ELR=Y $0 hp-round2"
    echo "     → Fine-tune around winner"
    echo "  3. Analyze Round 2, identify top-3 configs"
    echo "  4. TOP3_CONFIGS='a,b;c,d;e,f' $0 hp-seeds"
    echo "     → Multi-seed validation"
    echo "  5. BEST_MLR=X BEST_ELR=Y $0 hp-wd"
    echo "     → Weight decay sensitivity check"
    echo "  6. python -m nanoseek.scripts.analyze_ablations --full-report"
    ;;

  help|*)
    echo "Usage: $0 {gate1|hp-grid|hp-round2|hp-seeds|hp-wd|stability|architecture|500m-transfer|day1}"
    echo ""
    echo "  HP search (run sequentially, each informs the next):"
    echo "    hp-grid        Round 1: Coarse grid search (12 runs, ~2.6 hrs)"
    echo "    hp-round2      Round 2: Fine grid around R1 winner (9 runs, ~2 hrs)"
    echo "    hp-seeds       Round 3: Multi-seed validation of top-3 (9 runs, ~2 hrs)"
    echo "    hp-wd          Round 4: Weight decay sensitivity on winner (2 runs, ~26 min)"
    echo ""
    echo "  Other phases:"
    echo "    gate1          Gate 1 smoke test (100 steps, ~2 min)"
    echo "    stability      Stability ablations (5 runs, ~1.1 hrs)"
    echo "    architecture   Architecture ablations (4 runs, ~52 min)"
    echo "    500m-transfer  500M muP validation (1 run, ~14 hrs)"
    echo "    day1           gate1 + hp-grid R1 + stability + architecture"
    echo ""
    echo "Environment variables:"
    echo "  SEED=42                           Random seed (default: 42)"
    echo "  SCALE=anchor                      Model scale (default: anchor)"
    echo "  BEST_MLR=0.02                     Best matrix_lr (for hp-round2, hp-wd, 500m-transfer)"
    echo "  BEST_ELR=0.3                      Best embedding_lr (for hp-round2, hp-wd, 500m-transfer)"
    echo "  TOP3_CONFIGS='mlr,elr;mlr,elr;..' Top-3 configs for multi-seed (for hp-seeds)"
    ;;

esac
