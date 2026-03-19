# NanoSeek Training Execution Plan
## Current State + Next Steps — For Any Engineer to Pick Up and Run
### Updated: March 19, 2026

---

## 1. Where We Are Right Now

### Project in One Sentence
NanoSeek is a from-scratch DeepSeek V3.2 reimplementation at 1B scale, built as a scaling science lab.

### What's Done
- Model architecture: Sections 1-7 complete (MLA, Gate, MoE, MTP, full NanoSeekModel)
- Training infrastructure: optimizer (MuonAdamW), dataloader (FIM 10%), EMA tracking, eval wiring, checkpoint resume
- 120 tests passing, config validation enforces all rules
- 7 critical training bugs found and fixed (see `docs/TRAINING_BUGS_POSTMORTEM.md`)

### What's NOT Done
- Sections 8-9 (DSA/Indexer) — deferred to Phase 2, not needed for 4K training
- 3-stage RL pipeline — only single-stage GRPO exists
- No completed full training run yet — only smoke tests (100 steps)

### Last Training Run (gate1-smoke-v5, 38 steps before kill)
```
Step 00 | main=10.89  mtp=10.88  H_load=5.97  grad=5.45
Step 10 | main= 9.05  mtp=10.19  H_load=5.84  grad=3.24
Step 20 | main= 7.13  mtp= 7.95  H_load=5.28  grad=2.75
Step 30 | main= 6.53  mtp= 6.86  H_load=5.70  grad=1.15
Step 38 | main= 6.38  mtp= 6.56  H_load=5.78  grad=1.08
```
This is the FIRST correctly-initialized run. All v1-v4 runs had broken init.

---

## 2. The Training Path (3 Scales, Bottom-Up)

```
Anchor (~55M active)  ──→  nano-500M (~441M active)  ──→  NanoSeek-1B (1.08B active)
    HP search here            Validate HP transfer            Full training
    ~4 GPU-hours              ~14 GPU-hours                   ~12 GPU-hours (8×H100)
    ~$10                      ~$46                            ~$316
```

All three scales share: 64 experts, top-8, 16 layers, same MoE ratio (0.375×hidden).
Only `hidden_size` changes: 480 → 1280 → 2048.
Learning rates scale via muP: `LR_scaled = LR_base × √(B/B_ref) × (w_ref / w)`.

---

## 3. Phase 3A: Anchor HP Search (YOUR NEXT STEP)

### 3A.0 Gate 1 Smoke Test (Do This First)

**Purpose**: Verify training works end-to-end on GPU before burning compute.

**Command**:
```bash
cd /workspace/nanoseek
WANDB_API_KEY="<your-key>" python -m nanoseek.scripts.pre_train \
    --run gate1-smoke \
    --scale anchor \
    --seed 42 \
    --num-iterations 100 \
    --eval-every 50 \
    --save-every 100 \
    --device-batch-size 4
```

**Pass Criteria** (check in logs + W&B):
| Check | Threshold | Where to Find |
|-------|-----------|---------------|
| H_load at init | > 4 bits | Step 0 log line: `H_load: X.XX` |
| H_load maintained | > 2 bits throughout | All log lines |
| MTP loss decreasing | Not stuck at 10.3972 | `mtp:` field in log lines |
| Main loss decreasing | Drops from ~10.9 | `main:` field in log lines |
| EMA val BPB changes | Different at step 0 vs 100 | `EMA Validation bpb:` lines |
| Dead expert check | No crash | `Milestone eval` output at step 50, 100 |
| I_spec computes | Shows a number | `I_spec:` at milestone evals |
| FIM fraction | ~0.10 | W&B `train/fim_fraction` |
| MTP acceptance | ~50% at init | `MTP acceptance:` at milestone |
| No NaN/Inf | Zero | No `[CRITICAL]` warnings |
| No false alarm spam | Zero `[WARNING]` in first 20 steps | Clean log output |

**If any check fails**: STOP. Do not proceed to HP search. Debug first.

### 3A.1 HP Grid Search (12 Runs)

**Purpose**: Find best learning rates at anchor scale. These transfer to larger scales via muP.

**What we're sweeping**:
- `matrix_lr` (Muon LR for 2D weights): {0.005, 0.01, 0.02, 0.04}
- `embedding_lr` (AdamW LR for embeddings): {0.1, 0.3, 0.5}
- Total: 4 × 3 = 12 runs

**Command** (run all 12):
```bash
for mlr in 0.005 0.01 0.02 0.04; do
  for elr in 0.1 0.3 0.5; do
    WANDB_API_KEY="<your-key>" python -m nanoseek.scripts.pre_train \
        --run "hp-r1-mlr${mlr}-elr${elr}" \
        --scale anchor \
        --matrix-lr $mlr \
        --embedding-lr $elr \
        --eval-every 100 \
        --save-every -1 \
        --seed 42 &
    sleep 5
  done
  wait  # wait for this batch of 3 before starting next
done
```

**Each run**: ~13 minutes on A6000 (full anchor training = 1.1B tokens).
**Total**: ~4 GPU-hours, ~$10 on RunPod A6000.

**How to pick the winner**: Sort by `ema_val_bpb` at the end. Lowest wins.

### 3A.2 Stability Ablations (5 Runs)

**Purpose**: Test which stability techniques actually matter.

| Run Name | What's Changed | CLI Flags |
|----------|---------------|-----------|
| `stab-A` | Full V3.2 baseline | (none — default) |
| `stab-C` | Remove seq_aux | `--no-seq-aux` |
| `stab-D` | Remove grad clip | `--no-grad-clip` |
| `stab-E` | Classic aux loss | `--aux-loss-type classic` |
| `stab-F` | Bad batch injection | `--inject-bad-batch 1500` |

**Use best HP from 3A.1** for all runs. Each run uses the same command as the gate1 smoke test but with the ablation flag added and the winning LR values.

**What to measure**: Compare `ema_val_bpb`, `I_spec`, spike recovery across runs.

---

## 4. Phase 3B: nano-500M Validation

**Purpose**: Verify that HP from anchor scale transfers to 500M via muP scaling.

**Prerequisites**: Phase 3A complete. Best `matrix_lr` and `embedding_lr` identified.

**Command** (single run, ~14 hours):
```bash
WANDB_API_KEY="<your-key>" python -m nanoseek.scripts.pre_train \
    --run hp-500m-transfer \
    --scale 500m \
    --matrix-lr <best_from_3A> \
    --embedding-lr <best_from_3A> \
    --eval-every 500 \
    --save-every 2000 \
    --device-batch-size 8 \
    --seed 42
```

Note: muP scaling happens automatically — `pre_train.py` computes
`hidden_lr = matrix_lr × √(B/B_ref) × (w_ref / w)` from the scale config.

**Gate 2 Pass Criteria**:
- [ ] nano-500M converged (no NaN, no divergence)
- [ ] ema_val_bpb reasonable for 500M-class MoE
- [ ] H_load > 2 bits throughout
- [ ] MTP acceptance increased over training
- [ ] gamma_freeze_ratio = 0.95 in config
- [ ] No per-scale HP tuning needed (if tuning was needed, document which rule broke)

**If Gate 2 fails**: Run a 4-point grid at 500M scale to find correct HP, then investigate why muP transfer broke.

---

## 5. Phase 3D: NanoSeek-1B Full Training

**Purpose**: Train the target model at full scale.

**Prerequisites**: Gate 2 passed. muP HP transfer validated.

**Command** (8×H100, ~12 hours):
```bash
torchrun --nproc_per_node=8 -m nanoseek.scripts.pre_train \
    --run nanoseek-1b-v1 \
    --scale 1b \
    --matrix-lr <best_from_3A> \
    --embedding-lr <best_from_3A> \
    --eval-every 500 \
    --save-every 2000 \
    --device-batch-size 16 \
    --seed 42
```

**Key numbers**:
- 22B tokens, 1.08B active / 4.75B total parameters
- ~75 GB GPU memory (tight on 1×H100, comfortable on 8×H100 with DDP)
- Expected final ema_val_bpb: from scaling law fit using anchor + 500M points

---

## 6. GPU & Cost Requirements

| Phase | GPU | GPU-Hours | Cost | Wall Time |
|-------|-----|-----------|------|-----------|
| 3A.0 Gate 1 smoke | 1× A6000 (24GB) | 0.2 | $0.16 | 13 min |
| 3A.1 HP grid (12 runs) | 1× A6000 | 2.6 | $8.55 | ~3 hrs |
| 3A.2 Stability (5 runs) | 1× A6000 | 1.1 | $3.62 | ~1.5 hrs |
| 3B nano-500M | 1× H100 (80GB) | 14 | $46 | ~14 hrs |
| 3D NanoSeek-1B | 8× H100 | 96 | $316 | ~12 hrs |
| **Total** | | **~114** | **~$374** | **~3 days** |

### RunPod Pod Specs

**Day 1 — Anchor (A6000)**:
```
GPU: NVIDIA A6000 (48GB), 1×
Image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
Volume: 50 GB
Cost: ~$0.79/hr
```

**Day 2 — nano-500M (H100)**:
```
GPU: NVIDIA H100 80GB HBM3, 1×
Image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
Volume: 100 GB
Cloud: SECURE (for NVLink when scaling to 8×)
Cost: ~$3.29/hr
```

**Day 3 — NanoSeek-1B (8×H100)**:
```
GPU: NVIDIA H100 80GB HBM3, 8×
Image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
Volume: 200 GB
Cloud: SECURE (MUST have NVLink)
Cost: ~$26.32/hr
```

---

## 7. Environment Setup (Run on Every New Pod)

```bash
# 1. Clone repo
git clone <repo-url> /workspace/nanoseek
cd /workspace/nanoseek/nanoseek

# 2. Install dependencies
pip install wandb tiktoken scikit-learn pyarrow

# 3. Download training data (ClimbMix)
python -m nanoseek.nanoseek.dataset --num-shards 170  # ~170 shards = enough for anchor+500M

# 4. Set W&B key
export WANDB_API_KEY="<your-key>"

# 5. Verify setup
python -m pytest tests/ -v  # all 120 tests should pass

# 6. Quick sanity check (10 steps on GPU, should take <2 min)
python -m nanoseek.scripts.pre_train \
    --run sanity-check \
    --scale anchor \
    --num-iterations 10 \
    --eval-every -1 \
    --save-every -1 \
    --device-batch-size 4
# Check: loss decreasing, H_load > 4, no errors
```

---

## 8. Key Hyperparameters Reference

### Anchor Scale (what you'll use for Phase 3A)
| Parameter | Value | Notes |
|-----------|-------|-------|
| hidden_size | 480 | |
| num_layers | 16 | Same across all scales |
| n_experts | 64 | Same across all scales |
| top_k | 8 | Same across all scales |
| sequence_length | 4096 | |
| total_tokens | 1.1B | Chinchilla-optimal for ~55M active |
| matrix_lr (base) | 0.02 | Sweep in 3A.1 |
| embedding_lr (base) | 0.3 | Sweep in 3A.1 |
| lm_head_lr | 0.008 | |
| router_lr | 3e-4 | Constant (no muP scaling) |
| weight_decay | 0.1 | |
| grad_clip | 1.0 | |
| warmup_steps | 200 | |
| ema_decay | 0.9999 | Karras warmup applied automatically |
| fim_rate | 0.10 | 10% fill-in-middle |
| mtp_lambda | 0.3 → 0.1 | Transitions at 60% of training |
| gamma_freeze_ratio | **0.95** | NEVER use 0.80 |
| beta2 | 0.95 | NOT 0.999 |

### muP Scaling Rules (automatic in pre_train.py)
```
Hidden weight LR  = base × √(B/B_ref) × (w_ref / w)
Embedding LR      = base × √(B/B_ref)
LM head LR        = base × √(B/B_ref)
Router LR          = constant (no scaling)
Norm LR            = constant (no scaling)
Weight decay       = base × √B × (D_ref / D)
```
Where `w_ref = 480` (anchor hidden), `B_ref = 262,144` (anchor batch tokens).

---

## 9. Monitoring & Debugging

### W&B Dashboard
All runs log to project `nanoseek` at https://wandb.ai/<your-org>/nanoseek.

**Key metrics to watch**:
| Metric | Healthy Range | Red Flag |
|--------|--------------|----------|
| `ema_val/bpb` | Decreasing over training | Flat or increasing |
| `train/h_load` | > 4.0 bits | < 2.0 = expert collapse |
| `eval/i_spec_mean` | 0.1-0.7 (increases over training) | Stuck at 0 |
| `train/mtp_loss` | Decreasing, tracks main_loss | Frozen at 10.3972 |
| `train/grad_norm` | Stable or slowly decreasing | Spikes > 10× or NaN |
| `health/grad_spikes_last_100` | < 5 | > 10 = instability |
| `train/fim_fraction` | ~0.10 | 0.0 = FIM broken |

### Log Files
Training logs are saved to `logs/<run-name>.log`. Key patterns to grep:
```bash
grep "CRITICAL" logs/<run>.log          # Must be zero
grep "WARNING" logs/<run>.log | wc -l   # Should be minimal after warmup
grep "EMA Validation" logs/<run>.log    # Should change between evals
grep "Dead expert" logs/<run>.log       # Should complete, not crash
grep "I_spec" logs/<run>.log            # Should show non-zero eventually
```

### Common Failure Modes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Loss = 10.3972 at step 0 | RMSNorm/RoPE not initialized | Re-run `init_weights()` — check `_reinit_buffers()` |
| MTP loss frozen | concat_proj zero or init bug | Check MTP param stats: `model.mtp.mtp_modules.0.concat_proj.weight.std()` |
| H_load = 3.0 (stuck) | Broken init (Bugs 1-2) | Same as loss=10.3972 |
| EMA BPB never changes | EMA decay too high for run length | Karras warmup should fix; check `ema_tracker.update_count` |
| Dead expert crash | Tensor on GPU passed to numpy | Add `.cpu().float()` before `.numpy()` |
| I_spec = 0 | sklearn not installed | `pip install scikit-learn` |
| OOM at 500M/1B | Batch size too large | Reduce `--device-batch-size`, gradient accumulation compensates |
| NaN loss | Gradient explosion | Check `--no-grad-clip` isn't set; reduce LR |

---

## 10. Critical Rules (NEVER Violate)

1. **EMA weights for ALL evaluation** — never eval on raw checkpoint weights
2. **gamma_freeze_ratio = 0.95** — the old 0.80 was wrong
3. **FIM from token 1** — never add FIM via fine-tuning
4. **H_load + I_spec logged for ALL runs** — these are the primary scientific output
5. **Test meta-device init** — always run `TestMetaDeviceInit` tests before training
6. **Check all 120 tests pass** before any GPU training run
7. **Read `docs/TRAINING_BUGS_POSTMORTEM.md`** before modifying init or buffer code

---

## 11. File Map (Where Things Live)

```
nanoseek/
├── nanoseek/nanoseek/
│   ├── model.py              ← Model architecture (1,999 lines)
│   ├── config.py             ← Configs: anchor/500m/1b
│   ├── dataloader.py         ← BOS-aligned packing + FIM
│   ├── optim.py              ← MuonAdamW optimizer
│   └── eval/                 ← I_spec, dead experts, domain BPB, scaling law
├── nanoseek/scripts/
│   ├── pre_train.py          ← Main training script (run this)
│   ├── run_phase3.py         ← Phase 3 orchestration
│   └── analyze_ablations.py  ← Ablation analysis
├── nanoseek/tests/           ← 120 tests (run before training)
├── docs/
│   ├── TRAINING_BUGS_POSTMORTEM.md  ← What went wrong in v1-v4
│   ├── TRAINING_EXECUTION_PLAN.md   ← THIS FILE
│   └── PROJECT_STATE.md             ← Status dashboard
├── checkpoints/              ← Saved model weights
├── logs/                     ← Training logs
└── wandb/                    ← W&B run data
```
