# NanoSeek Phase 3 — RunPod Training Agent Context

**Read this FIRST before doing anything.** This file gives you everything you need to run NanoSeek's Phase 3 training ablation campaign on RunPod.

---

## What This Project Is

NanoSeek is a from-scratch DeepSeek V3.2 reimplementation at 1.08B active / 4.75B total parameters. You are running **Phase 3: ablation experiments** — the first time real GPU training happens.

## What You Are Doing

Running 21 training runs at anchor scale (~55M active params) on a single RTX 4090, then analyzing the results. These runs take ~2-4 hours total.

The runs fall into 4 groups:
1. **Gate 1 smoke test** (1 run, 100 steps) — verify everything works
2. **HP grid search** (up to 32 runs across 4 rounds) — find best learning rates
   - Round 1: coarse 4×3 grid (12 runs)
   - Round 2: fine ±25% grid around R1 winner (9 runs)
   - Round 3: multi-seed validation of top-3 (9 runs)
   - Round 4: weight decay sensitivity (2 runs)
3. **Stability ablations** (5 runs) — test DeepSeek V3 training techniques
4. **Architecture ablations** (3 runs) — test MoE architectural choices

---

## Setup (Run Once)

```bash
cd /workspace/nanoseek/nanoseek/scaling_law_lab
bash setup_runpod.sh --full --num-shards 170
```

This clones the repo, installs deps, downloads ~10GB of training data, authenticates W&B, and runs the test suite. Takes ~10-15 min.

**Critical environment variables** (must be set):
```bash
export NANOCHAT_BASE_DIR=/workspace/data    # data on persistent volume
export WANDB_API_KEY=<set in RunPod env>    # W&B logging
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## Run Training

```bash
cd /workspace/nanoseek/nanoseek
python -m nanoseek.scripts.run_phase3 --stop-after day1
```

This is the **pipeline orchestrator**. It runs all 21 runs sequentially with:
- **Crash recovery**: state saved to `scaling_law_lab/pipeline_state.json`. If the pod dies, re-run the same command — it skips completed stages.
- **Monitored training**: each run writes `logs/{run_name}.log` and `logs/{run_name}_status.json` with PID, metrics, and health alerts.
- **Automated health monitoring**: TrainingHealthMonitor checks every step for NaN, gradient spikes, loss spikes, expert collapse.

---

## How to Monitor

```bash
# Check status of all runs
python -m nanoseek.scripts.monitored_train --list

# Check a specific run (works while training)
python -m nanoseek.scripts.monitored_train --check gate1-smoke

# Read the full log of a run
cat logs/gate1-smoke.log

# Read last 50 lines
tail -50 logs/stab-A-baseline.log

# Kill a running training process (saves emergency checkpoint first)
python -m nanoseek.scripts.monitored_train --kill <run_name>
```

---

## What to Verify After Gate 1 (First Run)

Gate 1 is a 100-step smoke test. After it completes, verify:

```bash
python -m nanoseek.scripts.monitored_train --check gate1-smoke
```

Check these criteria:
- **state**: "completed" (not "failed")
- **last_loss**: decreased from ~10.4 (should be < 10.0 by step 100)
- **last_h_load**: > 4.0 bits (random routing should be near-uniform)
- **critical_count**: 0 (no NaN, no expert collapse)
- **last_mfu**: > 30% on RTX 4090

Also verify checkpoint exists:
```bash
ls checkpoints/nanoseek_anchor/gate1-smoke/ema_000100.pt
```

**If Gate 1 fails**: the pipeline stops automatically. Debug before proceeding.

---

## What to Do When a Run Fails

**DO NOT retry blindly.** Follow this procedure:

1. Check what failed:
```bash
python -m nanoseek.scripts.monitored_train --check <failed_run>
tail -50 logs/<failed_run>.log
```

2. Look for health alerts:
```bash
cat logs/<failed_run>_status.json | python -m json.tool | grep -A2 CRITICAL
```

3. Diagnose root cause:

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Return code -9 | OOM killed | Reduce `--device-batch-size` to 8 |
| Return code -15 | SIGTERM (preemption) | Resume: pipeline auto-skips completed runs |
| NaN in loss | Numerical overflow in bf16 | Check LR, check data batch |
| H_load < 2.0 | Expert collapse | Check routing config, check aux loss |
| `ModuleNotFoundError: nanochat` | Missing dependency | `pip install -e /workspace/nanoseek/nanochat/` |
| `FileNotFoundError: parquet` | No training data | `export NANOCHAT_BASE_DIR=/workspace/data` |
| `total_batch_tokens not divisible` | Wrong batch config | Reduce `--device-batch-size` |
| CUDA error | GPU issue | Check `nvidia-smi`, restart pod if needed |

4. Fix the code, test with 10-step smoke test:
```bash
python -m nanoseek.scripts.monitored_train \
    --run debug-test --scale anchor \
    --num-iterations 10 --eval-every 5 --save-every -1
```

5. Resume the pipeline:
```bash
python -m nanoseek.scripts.run_phase3 --stop-after day1
```

---

## After Day 1 Completes

1. Run analysis:
```bash
python -m nanoseek.scripts.analyze_ablations --full-report --output results/phase3_day1.json
```

2. Check hypothesis test results — the script auto-compares against pre-registered hypotheses.

3. Commit and push:
```bash
git add results/ logs/ scaling_law_lab/pipeline_state.json
git commit -m "Phase 3 Day 1: anchor ablation results"
git push
```

4. Find the best HP from the grid search:
```bash
python -m nanoseek.scripts.monitored_train --list | grep hp-r1
```
The pipeline also saves this to `scaling_law_lab/pipeline_state.json` under `best_hp`.

---

## Key Architecture Decisions (So You Understand What You're Training)

- **Model**: MoE + MLA + MTP (DeepSeek V3.2 architecture at nano scale)
- **Anchor model**: 55M active / 240M total params, 16 layers, 480 hidden, 64 experts (top-8)
- **Training data**: ClimbMix-400B (Karpathy), 170 parquet shards downloaded
- **Tokenizer**: From nanochat (sibling package in repo)
- **Optimizer**: MuonAdamW with muP scaling (LR transfers across model widths)
- **Loss**: main CE + MTP auxiliary + MoE load-balance (seq_aux + bias-based)
- **Evaluation**: EMA weights only (never raw weights), bits-per-byte metric

## Ablation Flags (What Each Run Tests)

| Flag | What It Disables | Run Name |
|------|-----------------|----------|
| (none) | Baseline — all V3.2 techniques ON | stab-A-baseline |
| `--no-seq-aux` | Sequence-level auxiliary loss | stab-C-no-seq-aux |
| `--no-grad-clip` | Gradient clipping | stab-D-no-gradclip |
| `--aux-loss-type classic` | Switches to traditional aux loss | stab-E-classic-aux |
| `--inject-bad-batch 1500` | 10× gradient spike at step 1500 | stab-F-bad-batch |
| `--no-mtp` | Multi-token prediction | arch-no-mtp |
| `--no-shared-experts` | Shared expert output | arch-no-shared |
| `--num-experts 16 --top-k 2` | Fewer, coarser experts | arch-fewer-experts |

## Health Monitor Alerts

The `TrainingHealthMonitor` in `pre_train.py` runs every step and emits:

| Alert | Severity | Meaning |
|-------|----------|---------|
| NaN/Inf detected | CRITICAL | Immediate — restore checkpoint |
| Severe loss spike (>3×) | CRITICAL | Consider checkpoint restore |
| Expert collapse (H_load < 2.0) | CRITICAL | Routing is degenerate, run is dead |
| Gradient norm spike (z > 4.0) | WARNING | Isolated spike, watch for recurrence |
| Grad norm EMA > 0.5 | WARNING | Trending toward blowup |
| Loss spike (>2×) | WARNING | Monitor — usually recovers |
| Spike frequency increasing | WARNING | Multiple spikes = deteriorating |
| H_load declining < 4.0 | WARNING | Early sign of collapse |

---

## File Map (Key Files Only)

```
nanoseek/
├── scripts/
│   ├── pre_train.py              # THE training loop (ablation flags, health monitor, W&B)
│   ├── run_phase3.py             # Pipeline orchestrator (crash recovery, gate checks)
│   ├── monitored_train.py        # Wrapper: PID tracking, log capture, status JSON
│   └── analyze_ablations.py      # Post-training hypothesis testing + reports
├── nanoseek/
│   ├── config.py                 # All model/training configs (anchor, 500M, 1B)
│   ├── model.py                  # NanoSeekModel (MoE + MLA + MTP)
│   ├── dataloader.py             # Tokenizing data loader with FIM
│   └── checkpoint_manager.py     # Atomic checkpoint save/load
├── scaling_law_lab/
│   ├── setup_runpod.sh           # One-command pod setup
│   ├── pipeline_state.json       # Crash recovery state (auto-generated)
│   ├── configs/                  # YAML configs for each ablation group
│   └── runpod_templates.yaml     # Pod specs (RTX 4090, H100, 8×H100)
├── eval/
│   ├── information_metrics.py    # I_spec (expert specialization)
│   ├── moe_diagnostics.py        # MTP acceptance, dead experts
│   ├── domain_bpb.py             # Per-domain evaluation
│   └── scaling_law.py            # Chinchilla fit with LOOCV
├── logs/                         # Training logs (auto-generated)
├── checkpoints/                  # Model checkpoints (auto-generated)
├── results/                      # Analysis output (auto-generated)
└── CLAUDE.md                     # Full project context (read for deep understanding)
```

---

## W&B Organization

All runs log to W&B project `nanoseek` with automatic grouping:
- **Groups**: `gate1-anchor`, `hp-anchor`, `stability-anchor`, `architecture-anchor`
- **Tags**: `scale:anchor`, `ablation:stability`, `variant:no-seq-aux`, `hp-round:1`, etc.
- **HP round tags**: `hp-round:1` (coarse), `hp-round:2` (fine), `hp-round:3` (multi-seed), `hp-round:4` (weight decay)
- **Config**: both CLI args and `effective/*` post-override values

**HP run name convention:**
- Round 1: `hp-r1-mlr{mlr}-elr{elr}` (12 runs, coarse grid)
- Round 2: `hp-r2-mlr{mlr}-elr{elr}` (9 runs, fine grid around R1 winner)
- Round 3: `hp-seed-mlr{mlr}-elr{elr}-s{seed}` (9 runs, 3 configs × 3 seeds)
- Round 4: `hp-wd-mlr{mlr}-elr{elr}-wd{wd}` (2 runs, weight decay sensitivity)

Dashboard URL: https://wandb.ai (check your entity)

---

## What Happens After Day 1

The pipeline state file tells the next agent what to do:

- **Day 2** (1× H100): Run `python -m nanoseek.scripts.run_phase3 --start-from 500m-transfer --stop-after day2`
  - Uses best HP from Day 1 grid search (auto-extracted from checkpoint metadata)
  - Validates muP transfer hypothesis at 500M scale

- **Day 3** (8× H100): Full NanoSeek-1B training (22B tokens)
  - Use `torchrun --nproc_per_node=8` for multi-GPU
  - ~12 hours, ~$316

---

## Rules

1. **Never modify the pre-registered hypotheses** in `analyze_ablations.py` after seeing results
2. **Always use monitored_train.py** (via run_phase3.py) — never run pre_train.py directly
3. **Commit results and push** after each completed stage
4. **If you fix a bug**, test with 10-step smoke test before resuming the pipeline
5. **Rotate the W&B API key** if it was ever exposed in logs or code
