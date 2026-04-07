# NanoSeek Scaling Laws — Experiment Plan

## Context

NanoSeek is a from-scratch MoE reimplementation of DeepSeek V3.2 at nano scale (1.08B active / 4.75B total). The scaling law infrastructure answers: **"How should we allocate compute for MoE models at this scale?"** — a question nobody has systematically studied below 1B active params with full routing diagnostics.

We borrow nanochat's "single dial" philosophy (one `--depth` flag controls everything) but adapt it for MoE where active != total params, routing health matters, and expert specialization is a first-class metric.

---

## First Principles: Why This Ordering

The ordering of experiments follows a single rule: **never measure properties of a system you haven't calibrated.**

1. **Architecture is settled.** DeepSeek V3 proved bias routing, shared experts, and MTP at 671B scale. DeepSeekMoE proved shared experts at 16B scale (+33% loss without them). NanoSeek is a **reimplementation**, not a new design. Re-ablating proven features is spending compute to rediscover known results. The scientific question is "how does this architecture scale down?" — not "is this architecture right?"

2. **Calibrate hyperparameters first.** IsoFLOP (Phase 2) will spend ~$200 measuring `L(N, D) = A/N^alpha + B/D^beta + E`. The curve shape depends on HP quality — if LRs are far off, some depths may be differentially hurt (e.g., d20 unstable while d12 is fine), corrupting the IsoFLOP minimum. By tuning HPs first (~$30), we eliminate this risk entirely.

3. **Then measure how it scales.** With tuned HPs, IsoFLOP discovers the optimal compute allocation. Every run produces reliable data points — no "HP/IsoFLOP bet" needed.

4. **Then validate end-to-end.** Miniseries checks that everything composes correctly — muP transfer, Power Lines batch scaling, weight decay T_epoch framework, and routing stability across all depths.

**The previous ordering (Ablations -> IsoFLOP -> HP Search -> Miniseries) had two problems:**
1. Architecture ablation was redundant — re-proving what DeepSeek already validated at multiple scales.
2. IsoFLOP ran before HP calibration — a risky bet that default HPs were close enough. If they weren't, $200 wasted.

**The correct ordering eliminates both risks:** Lock architecture from V3 paper → calibrate HPs cheaply → measure scaling laws with tuned HPs → validate → train 1B.

**Why architecture ablation is wrong for this project:**
- HPs found during HP search are only valid for the architecture used. If ablation later changes the architecture (e.g., drops shared experts), HP search must be redone — combinatorial explosion.
- Architecture ablations at nano scale can give misleading signals. A feature that seems marginal at 400M active params might be critical at 1B. DeepSeek already tested at scales bracketing ours (16B, 671B).
- Every dollar spent on architecture ablation produces results about the wrong model (the ablated variant), while every dollar on HP search directly informs the final model.

**nanochat doesn't have this problem** because it's a dense transformer with no architectural variants — the architecture is GPT-2 (fixed by 7 years of literature). NanoSeek's architecture is also fixed — by DeepSeek V3 (validated at 671B) and DeepSeekMoE (validated at 16B).

---

## Pipeline Overview

```
Phase 0 ──> Phase 1 ──> Phase 2 ──> Phase 3 ──> Phase 4
Gate 1      HP Search    IsoFLOP     Miniseries   1B Grad.
(verify)    (calibrate)  (discover)  (validate)   (produce)
$0.50       ~$30         ~$200       ~$120        ~$350
5 min       4 hours      2-3 days    1-2 days     12 hours
```

**Design principle**: Each phase answers **one question** whose output is a **hard dependency** for the next phase. No skipping, no reordering.

**Architecture is locked from Phase 0**: Full DeepSeek V3 — MLA + MoE (64 experts top-8, bias routing, 2 shared) + MTP. All phases run on the exact same architecture.

---

## Phase 0: Gate 1 Smoke Test

### What to run
```bash
python -m nanoseek.scripts.pre_train \
    --run gate1-smoke --scale ablation --seed 42 \
    --num-iterations 100 --eval-every 50 --save-every 100 \
    --device-batch-size 4
```
100 steps at d16 ablation scale (~5 minutes, ~$0.50).

### Question answered
**"Does the code actually work on a real GPU?"**

### Pass/fail checklist

| Check | Pass | Fail means |
|-------|------|------------|
| Loss step 0 ~ 10.9 (~ ln(32768)) | Init correct | Bug #1/#2 from POSTMORTEM: RMSNorm/RoPE zeros |
| Loss step 100 < 8.0 | Model is learning | Gradient flow broken |
| H_load > 4.0 bits | Routing balanced | Dead routing — all tokens hitting one expert |
| MTP loss decreasing | MTP head working | Bug #3: MTP zero-init |
| No OOM | Batch size fits | Reduce --device-batch-size |
| Gradient norm < 10 | Stable training | MoE gradient spikes, clipping needed |

### Why this must come first
7 bugs were found during development (see `docs/TRAINING_BUGS_POSTMORTEM.md`) — 3 were CRITICAL, causing the model to **not learn at all** (loss stuck at 10.3972 = ln(vocab)). Without a smoke test, you could burn the entire budget with broken code and get plausible-looking but meaningless results.

### Output
Binary: **PASS** (continue) or **FAIL** (debug first).

---

## Phase 1: HP Search + muP Validation — Calibrate the System

### Sub-phase 1a: Learning Rate Sweep at Reference Scale

### What to run
```bash
for mlr in 0.005 0.01 0.02; do
  for elr in 0.2 0.5; do
    python -m nanoseek.scripts.pre_train \
        --run "hp-mlr${mlr}-elr${elr}" --scale ablation \
        --matrix-lr $mlr --embedding-lr $elr \
        --num-iterations 500 --eval-every 100 \
        --save-every -1 --seed 42
  done
done
```
6 runs x 500 steps at d16 ablation scale.

### Question answered
**"What are the optimal base learning rates for matrix and embedding parameters?"**

### Why only 2 hyperparameters?

NanoSeek uses **MuonAdamW** — a hybrid optimizer with distinct parameter groups, each with its own LR (`pre_train.py:545-631`):

```python
hidden_lr  = matrix_lr  * sqrt(B/B_ref) * (w_ref/w)    # Muon — auto-scales
embed_lr   = embedding_lr * sqrt(B/B_ref)                # AdamW — auto-scales
lm_head_lr = unembedding_lr * sqrt(B/B_ref)              # AdamW — auto-scales
router_lr  = 3e-4                                        # CONSTANT (no scaling)
norm_lr    = 3e-4                                        # CONSTANT (no scaling)
```

Router and norm LRs are constant by muP prescription (output weights and 1D params don't scale with width). lm_head is typically proportional to embed_lr. Only **2 base rates** need tuning: `matrix_lr` (Muon) and `embedding_lr` (AdamW).

### Why 6 runs is enough (vs nanochat's 320)

nanochat **doesn't use muP** — Karpathy found that *"What works at d12 doesn't transfer to d20"* (LOG.md, Jan 19-22, 2026). He had to brute-force sweep at every target scale across 320 experiments.

NanoSeek uses muP explicitly (Tensor Programs V + Complete(d)P, `pre_train.py:449-452`):
```
eta_d24 = eta_d16 * (1280/1920)  <- auto-computed, no re-tuning needed
```
Tune at d16 -> transfer to d12, d14, d18, d20, d24, 1B by formula. 6 runs instead of 320 — **50x cheaper**.

### Why HP search doesn't need ratio*

The previous plan argued HP search must come after IsoFLOP because weight decay scaling depends on total tokens D, and D depends on ratio* from IsoFLOP. This sounds rigorous but overestimates the coupling:

1. **HP search is a relative comparison at fixed budget.** All 6 runs use the same training horizon (500 steps). Weight decay is identical across runs. The ranking of (matrix_lr, embedding_lr) pairs is stable regardless of D.

2. **Weight decay sensitivity is low.** WD scales as `lambda_ref * sqrt(B/B_ref) * (D_ref/D)`. A 2x error in D changes WD by 2x, but WD is second-order compared to LR — the optimal LR is the same within our sweep grid.

3. **The Chinchilla 20x default is a reasonable D estimate.** We use `D = 20 * N_scaling` for batch sizing. The true ratio* might be 15x or 25x — neither changes the LR ranking.

### Sub-phase 1b: muP Transfer Validation

### What to run
```bash
# Best (mlr, elr) from 1a — muP auto-scales to other depths
python -m nanoseek.scripts.pre_train \
    --run mup-d12 --depth 12 --seed 42 \
    --matrix-lr <best_mlr> --embedding-lr <best_elr> \
    --num-iterations 500 --eval-every 100 --save-every -1

python -m nanoseek.scripts.pre_train \
    --run mup-d18 --depth 18 --seed 42 \
    --matrix-lr <best_mlr> --embedding-lr <best_elr> \
    --num-iterations 500 --eval-every 100 --save-every -1

python -m nanoseek.scripts.pre_train \
    --run mup-d20 --depth 20 --seed 42 \
    --matrix-lr <best_mlr> --embedding-lr <best_elr> \
    --num-iterations 500 --eval-every 100 --save-every -1
```
3 runs x 500 steps at d12, d18, d20.

### Question answered
**"Does muP HP transfer actually work for MoE?"**

This is the most scientifically novel question in the entire project. muP (Tensor Programs V) was designed for dense transformers. MoE adds routing dynamics that muP doesn't account for — router weights, expert load balancing, and bias updates are all outside muP's theoretical guarantees.

### What to check

If muP transfer works:
- Loss curves at d12, d18, d20 should be **stable** (no divergence, no NaN)
- Relative loss ordering: d20 < d18 < d12 (more params = lower loss at same step count)
- H_load > 4.0 bits at all depths (routing isn't destabilized by LR scaling)
- No depth shows gradient instability that d16 didn't show

If muP transfer fails:
- Some depth diverges or shows dramatically worse loss/step than expected
- H_load collapses at large or small depth (routing sensitive to LR)
- **Mitigation**: Run a small LR sweep at the failing depth to find a correction factor. If the factor is consistent (e.g., "d20 needs 0.7x the muP-predicted LR"), incorporate it. If random, muP doesn't work for MoE and we fall back to per-depth sweeps (more expensive but still cheaper than nanochat's 320 runs).

### Sub-phase 1c (optional): MTP Cost-Benefit Check

### What to run
```bash
# With MTP (default)
python -m nanoseek.scripts.pre_train \
    --run mtp-on --scale ablation --seed 42 \
    --matrix-lr <best> --embedding-lr <best> \
    --num-iterations 2000 --eval-every 500 --save-every -1

# Without MTP
python -m nanoseek.scripts.pre_train \
    --run mtp-off --scale ablation --seed 42 \
    --matrix-lr <best> --embedding-lr <best> \
    --no-mtp \
    --num-iterations 2000 --eval-every 500 --save-every -1
```
2 runs x 2000 steps.

### Why this is optional and NOT a full architecture ablation

This is NOT asking "should we use MTP?" — DeepSeek V3 already answered yes. This is asking a **compute allocation question**: "At nano scale, does MTP's ~15% compute overhead pay for itself in BPB improvement?"

At 671B / 14.8T tokens, 15% overhead is negligible. At 410M / 8.2B tokens, every token matters more. If MTP costs 15% effective tokens but improves BPB by less than what 15% more tokens would give, it's a net loss at this scale.

**Decision rule:**
- `delta_bpb > 0.02`: MTP clearly helps → KEEP (expected outcome)
- `delta_bpb < 0.005`: MTP overhead exceeds benefit at nano scale → DROP and reclaim compute
- `0.005 < delta_bpb < 0.02`: Borderline → KEEP (V3 proved it helps at scale, trust the paper)

**Note**: This is not testing MTP lambda annealing (0.3→0.1 at 60%). That schedule cannot be validated in 2000 steps — it gets tested during Miniseries (Phase 3).

### Phase 1 Output
- Best (matrix_lr, embedding_lr) pair. Example: `(0.01, 0.3)`
- muP transfer: WORKS / NEEDS_CORRECTION / FAILS
- MTP: KEEP / DROP (if sub-phase 1c was run)

### Phase 1 Cost
~$30 total (LR sweep $18 + muP validation $9 + optional MTP check $3)

---

## Phase 2: IsoFLOP Sweep — Discover the Scaling Law

### What to run
```bash
./runs/scaling_laws.sh apr06
```

20 runs: 5 depths x 4 FLOPs budgets, using **tuned HPs from Phase 1**.

```
           |  1e18    3e18    1e19    3e19
-----------+--------------------------------
  d12      |  run_1   run_2   run_3   run_4
  d14      |  run_5   run_6   run_7   run_8
  d16      |  run_9   run_10  run_11  run_12
  d18      |  run_13  run_14  run_15  run_16
  d20      |  run_17  run_18  run_19  run_20
```

Each run uses `--depth=$d --target-flops=$flops --matrix-lr=<best> --embedding-lr=<best>`. The auto-compute cascade in `pre_train.py:346-421` derives everything else:

```
depth -> hidden_size = ceil(depth * 80 / 128) * 128
      -> build model -> count N_active, N_scaling
      -> tokens = 20 * N_scaling (Chinchilla default, for batch sizing only)
      -> batch = B_REF * (tokens/D_REF)^0.383 (Power Lines paper)
      -> iterations = target_flops / (6 * N_active * batch_tokens)
      -> actual tokens = iterations * batch_tokens
```

When using `--target-flops`, tokens are computed **backwards from FLOPs**, not from the ratio. The Chinchilla 20x default is only used to size the batch via the Power Lines formula.

### Question answered
**"For a fixed compute budget C, which MoE model size gives the lowest loss?"**

### Why this is now reliable (no HP bet)

The previous plan ran IsoFLOP with default HPs and hoped they were close enough. This was a bet with $200 at stake. Now Phase 1 has already calibrated HPs:

- **Tuned LRs** → no risk of differential instability across depths
- **muP validated** → confident that LR scaling across the depth ladder is correct
- **Every data point is trustworthy** → no need to re-run if HPs turn out to be wrong

### How to analyze results

**Step 1: Plot IsoFLOP curves.** For each FLOPs budget, plot `N_scaling vs val_bpb`. Each curve is U-shaped — the minimum is the optimal model size N* for that budget.

**Step 2: Fit the Chinchilla power law.**
```
L(N, D) = A/N^alpha + B/D^beta + E
```
20 data points (N, D, L) -> fit A, B, alpha, beta, E.

**Step 3: Derive the optimal ratio.** Minimize L(N, D) subject to C = 6ND:
```
N* proportional to C^(beta/(alpha+beta))
D* proportional to C^(alpha/(alpha+beta))
ratio* = D*/N*
```
If alpha ~ beta, ratio* ~ constant. Chinchilla found alpha ~ 0.34, beta ~ 0.28 -> ratio ~ 20 for dense. MoE may differ because N_active << N_total changes the L(N, D) relationship.

**Step 4: MoE-specific analysis.** From H_load data:
- H_load vs depth: Does routing stay stable as we scale up?
- H_load vs FLOPs: Do longer-trained models develop better routing?
- If H_load collapses at large depth -> architecture problem, STOP.

### Cost
~$200 (20 runs x ~$10/run average). Small runs (d12 x 1e18) take minutes; large ones (d20 x 3e19) take hours.

---

## Phase 3: Miniseries Validation — Verify the Whole System

### What to run
```bash
PARAM_DATA_RATIO=<ratio* from Phase 2> ./runs/miniseries.sh apr06
```

6 runs: depths 12, 14, 16, 18, 20, 24. Each model trains at **compute-optimal** with ratio* from Phase 2, best HPs from Phase 1, on the **locked V3 architecture**.

### Question answered
**"Does the single-dial auto-compute cascade actually work for MoE across all depths?"**

### What gets auto-computed

When you pass `--depth=14`, the cascade (`pre_train.py:346-421`) computes:
```
depth=14, aspect_ratio=80
  -> hidden_size = ceil(14*80/128)*128 = 1152
  -> num_heads = 1152/128 = 9
  -> build model, count params: scaling = 284.5M
  -> tokens = ratio* * 284.5M
  -> batch = B_REF * (tokens/D_REF)^0.383, rounded to power of 2
  -> iterations = tokens / batch
  -> muP LR: matrix_lr* * sqrt(B/B_ref) * (1280/1152)
  -> WD: 0.1 * sqrt(B/B_ref) * (D_ref/tokens)
```

### This validates 4 things simultaneously

**1. muP transfer works for MoE at full training horizon?** Phase 1 validated muP at 500 steps. Here we test at compute-optimal duration (thousands of steps). If muP breaks at longer horizons — e.g., d24 diverges at step 5000 but was fine at step 500 — this catches it.

**2. Power Lines batch scaling is correct?** `B_opt proportional to D^0.383` (Bergsma et al., arXiv:2505.13738). If correct, each model at its auto-computed batch -> comparable loss/step efficiency across depths.

**3. Weight decay scaling is correct?** T_epoch framework (arXiv:2405.13698): `lambda = lambda_ref * sqrt(B/B_ref) * (D_ref/D)`. If correct, regularization strength is approximately constant across scales.

**4. MoE routing stays stable at scale?**
- H_load vs depth: should remain > 4.0 bits for all depths
- Dead experts vs depth: should be <= 2-3 at all depths
- MTP lambda annealing: validated here for the first time (Phase 1's 2000-step check didn't cover the 0.3 -> 0.1 transition at 60% of training)

### Pass/fail criteria

| Check | Pass | Fail means |
|-------|------|------------|
| val_bpb vs N_scaling: smooth curve | Monotonically decreasing | Auto-compute broken at some depth |
| d16 val_bpb ~ Phase 1 baseline | Within 1% | Ratio or HP mismatch |
| H_load stable across all depths | > 4.0 bits everywhere | Routing fails at certain scale |
| No OOM at d24 | Runs to completion | Batch auto-sizing too aggressive |
| Training time proportional to tokens | Linear relationship | Efficiency problem |

### Output
CSV: `(depth, active_params, scaling_params, tokens, val_bpb, H_load)`.

Plot: val_bpb vs N_scaling -> **MoE compute-optimal frontier**. Compare against IsoFLOP predictions — if they match, the system works end-to-end. If d12 or d24 deviates, investigate whether routing dynamics or muP transfer caused the deviation.

### Cost
~$120 (6 runs, from ~$10 for d12 to ~$50 for d24).

---

## Phase 4: NanoSeek-1B Graduation

### What to run
```bash
torchrun --nproc_per_node=8 -m nanoseek.scripts.pre_train \
    --run nanoseek-1b-v1 --scale 1b \
    --matrix-lr <best> --embedding-lr <best> \
    --eval-every 500 --save-every 2000 \
    --device-batch-size 16 --seed 42
```

1B scale: hidden=2048, 16 heads, 16 layers, ~1.08B active / ~4.75B total, 22B tokens, 8xH100.

### Question answered
**"What quality level does NanoSeek-1B achieve?"**

### Why we can run this with confidence

Every decision has been validated:

| Decision | Source | Phase |
|----------|--------|-------|
| Architecture | DeepSeek V3 paper (proven at 671B) | Locked from start |
| Learning rates | HP Search -> matrix_lr*, embed_lr* | Phase 1 |
| muP transfer | Validated at d12, d18, d20 | Phase 1 |
| Param-data ratio | IsoFLOP -> ratio* | Phase 2 |
| Auto-compute cascade (batch, WD) | Miniseries -> verified at 6 depths | Phase 3 |

muP automatically transfers HPs from d16 (1280) to 1B (2048):
```python
matrix_lr_1b = matrix_lr_d16 * sqrt(B_1b/B_ref) * (1280/2048)
embed_lr_1b  = embedding_lr_d16 * sqrt(B_1b/B_ref)
weight_decay = 0.1 * sqrt(B_1b/B_ref) * (D_ref/22B)
```

### Metrics tracked during training

| Metric | Frequency | Purpose |
|--------|-----------|---------|
| train_loss | Every step | Convergence monitoring |
| val_bpb (EMA) | 500 steps | **Primary quality metric** |
| domain_bpb | 500 steps | Per-domain breakdown (code, math, web, books, science) |
| H_load | 500 steps | Routing health |
| I_spec (labeled) | 2000 steps | Expert specialization by domain |
| Dead experts | 2000 steps | Expert utilization |
| MTP acceptance | 2000 steps | Speculative decoding readiness (target > 75%) |
| MFU | Every step | Hardware efficiency |

### Validation checkpoints

| Checkpoint | What to verify |
|------------|----------------|
| Step 0 | Loss ~ 10.9 (random init correct) |
| Step 100 | Loss < 8.0 (learning happening) |
| 10% training | val_bpb within 10% of IsoFLOP prediction |
| 60% training | MTP lambda transition 0.3 -> 0.1 (verify smooth) |
| 95% training | gamma_freeze kicks in (load bias stops updating) |
| Final | val_bpb matches/beats prediction, H_load > 4.0, I_spec > 0.3 |

### Cost
~$350 (8xH100 x ~12 hours).

---

## Dependency Graph

```
Phase 0: Gate 1
  | output: code works on GPU
  v
Phase 1: HP Search + muP Validation
  | needs: working code
  | output: best (matrix_lr, embedding_lr), muP transfer confirmed
  | NOT needed: ratio* (HP ranking is stable across D values)
  v
Phase 2: IsoFLOP Sweep (with TUNED HPs — no bet)
  | needs: tuned HPs + confirmed muP transfer
  | output: ratio*, alpha, beta, L(N,D) fit, H_load vs scale
  | advantage: every data point is reliable (no default-HP risk)
  v
Phase 3: Miniseries (the safety net)
  | needs: ratio* + best HPs + locked architecture
  | output: validated auto-compute cascade, compute-optimal frontier
  | catches: muP long-horizon failures, batch/WD scaling errors
  v
Phase 4: 1B Graduation
  | needs: everything above
  | output: trained NanoSeek-1B model + scientific metrics
```

**Every arrow is a hard dependency.** Architecture is locked from Phase 0 — it's not a phase output, it's a precondition.

**The key improvement over the previous plan:** No HP/IsoFLOP bet. Phase 1 calibrates HPs cheaply ($30), then Phase 2 uses them. No risk of spending $200 on IsoFLOP with bad HPs, no cross-check needed, no worst-case $200 re-run.

---

## Cost Summary

Total FLOPs: **7.37e20** → **591 GPU-hours** on H100 at 35% MFU.

| Phase | Runs | FLOPs | GPU-hrs | @ $1/hr | @ $2.50/hr |
|-------|------|-------|---------|---------|------------|
| 0. Gate 1 | 1 | 1.4e17 | 0.1 | $0.10 | $0.30 |
| 1. HP Search + muP | 9-11 | 1.2e19 | 10 | $10 | $25 |
| 2. IsoFLOP | 20 | 2.2e20 | 176 | $176 | $440 |
| **Research total** | **~30** | **2.3e20** | **186** | **$186** | **$465** |
| 3. Miniseries | 6 | 3.6e20 | 290 | $290 | $725 |
| 4. 1B Graduation | 1 | 1.4e20 | 114 | $114 | $286 |
| **Grand total** | **~37** | **7.4e20** | **591** | **$591** | **$1,476** |

**Spot pricing at $1/hr achieves the ~$600 target.** See "GPU Specification" section below for exact memory requirements, per-run breakdowns, and rental strategies.

**Note on Phase 3:** d24 alone is 155 GPU-hours (53% of Phase 3). If the scaling law fits well from d12-d20, d24 can be treated as optional validation — saving ~$155-390.

---

## GPU Specification — Exact Compute Requirements

### First Principles

Every run's compute cost is determined by one equation:

```
Total FLOPs = 6 × N_active × D
```

Where `6` = 2 (multiply-add) × 3 (forward + 2× backward). This is a physical constant — it doesn't change with GPU type, batch size, or number of GPUs. Everything else (wall time, cost) derives from this.

```
GPU-hours = Total_FLOPs / (peak_TFLOPS × MFU × 3.6e15)
Cost = GPU-hours × num_GPUs × price_per_GPU_hour
```

### GPU Selection: H100 80GB SXM

**Why H100 80GB, not A100:**
- H100 bf16 peak: 989 TFLOPS vs A100's 312 TFLOPS (3.2× faster)
- 80GB HBM3 vs 80GB HBM2e (3.35 TB/s vs 2.0 TB/s bandwidth)
- MoE is memory-bandwidth bound (64 experts loaded, 8 used) → H100's bandwidth advantage matters more than raw FLOPS
- Cost-efficiency: H100 at $2.50/hr does work 3.2× faster than A100 at $1.50/hr → H100 is 1.9× more cost-efficient

**MFU for MoE at nano scale:**
- Dense transformers achieve 40-55% MFU on H100 (nanochat reports ~50%)
- MoE is lower because: expert routing overhead, all 64 experts in memory but only 8 active, token dispatch/combine ops, load imbalance
- Conservative estimate: **30% MFU** (297 TFLOPS effective)
- Optimistic with torch.compile: **40% MFU** (396 TFLOPS effective)
- We use **35% MFU** (346 TFLOPS) as baseline for planning

### Memory Budget Per Depth

Training memory = model weights (bf16) + optimizer state (fp32) + gradients (bf16) + activations.

```
Static memory = N_total × 12 bytes
  Breakdown: 2 (bf16 params) + 2 (bf16 grads) + 8 (fp32 optimizer: momentum + variance)

Activation memory ≈ 2 × num_layers × DBS × seq_len × hidden_size × 2 bytes
  (With FlashAttention — no attention weight materialization)

Multi-GPU (DistMuonAdamW = ZeRO-2):
  Optimizer sharded across GPUs: 8 bytes/param → 8/num_gpus bytes/param
  Params + grads: full copy on each GPU (4 bytes/param)
  Per-GPU static = N_total × (4 + 8/num_gpus)
```

| Depth | N_total | Static (1 GPU) | Activation (DBS=4) | Total | Fits 1× H100? |
|-------|---------|---------------|-------------------|-------|---------------|
| d12 | 965M | 11.6 GB | 0.8 GB | **12.4 GB** | Yes |
| d14 | 1.41B | 16.9 GB | 1.1 GB | **18.0 GB** | Yes |
| d16 | 1.99B | 23.9 GB | 1.3 GB | **25.2 GB** | Yes |
| d18 | 3.20B | 38.4 GB | 2.3 GB | **40.7 GB** | Yes |
| d20 | 4.17B | 50.0 GB | 3.3 GB | **53.3 GB** | Yes |
| d24 | 6.65B | 79.8 GB | 4.5 GB | **84.3 GB** | **No** (need 2+) |
| 1b | 4.75B | 57.0 GB | 3.4 GB | **60.4 GB** | Yes |

With 2 GPUs + ZeRO-2 optimizer sharding:
- d24: per-GPU = 6.65B × (4 + 4) + 2.3 GB = **55.5 GB** → fits

With 8 GPUs + ZeRO-2:
- 1b: per-GPU = 4.75B × (4 + 1) + 3.4 GB = **27.2 GB** → comfortable, allows DBS=16

### Phase-by-Phase FLOPs and GPU-Hours

**Reference throughput: 1× H100 at 35% MFU = 346 TFLOPS → 1.246e18 FLOPs/GPU-hour**

#### Phase 0: Gate 1 Smoke Test

| Run | Config | Tokens | FLOPs | GPU-hrs | GPUs | Wall time |
|-----|--------|--------|-------|---------|------|-----------|
| gate1-smoke | d16, 100 steps | 52.4M | 1.37e17 | 0.11 | 1 | **7 min** |

**Cost: ~$0.30**

#### Phase 1: HP Search + muP Validation

| Run | Config | Steps | Tokens | FLOPs | GPU-hrs |
|-----|--------|-------|--------|-------|---------|
| **1a: LR sweep** (×6) | d16, 500 steps each | 3,000 | 1.57B | 4.12e18 | 3.3 |
| **1b: muP d12** | d12, 500 steps | 500 | 131M | 1.88e17 | 0.15 |
| **1b: muP d18** | d18, 500 steps | 500 | 262M | 1.05e18 | 0.84 |
| **1b: muP d20** | d20, 500 steps | 500 | 262M | 1.34e18 | 1.07 |
| **1c: MTP on** (opt.) | d16, 2000 steps | 2,000 | 1.05B | 2.75e18 | 2.21 |
| **1c: MTP off** (opt.) | d16, 2000 steps | 2,000 | 1.05B | 2.75e18 | 2.21 |

**Phase 1 total: 9.8 GPU-hours → ~$25 on 1× H100**
All runs fit on **1 GPU**. Sequential wall time: ~10 hours.
Run 1a in parallel (3 at a time) to finish in ~4 hours.

#### Phase 2: IsoFLOP Sweep (20 runs)

Each cell: `FLOPs_budget / (346e12 × 3600)` GPU-hours.

| Depth | 1e18 FLOPs | 3e18 FLOPs | 1e19 FLOPs | 3e19 FLOPs | Subtotal |
|-------|-----------|-----------|-----------|-----------|----------|
| d12 | 0.80 hrs | 2.41 hrs | 8.03 hrs | 24.1 hrs | 35.3 hrs |
| d14 | 0.80 | 2.41 | 8.03 | 24.1 | 35.3 |
| d16 | 0.80 | 2.41 | 8.03 | 24.1 | 35.3 |
| d18 | 0.80 | 2.41 | 8.03 | 24.1 | 35.3 |
| d20 | 0.80 | 2.41 | 8.03 | 24.1 | 35.3 |

**Phase 2 total: 176 GPU-hours**

Note: GPU-hours per cell depend only on FLOPs budget, not depth (FLOPs are fixed per cell).
All d12-d18 runs fit on **1 GPU**. d20 fits on **1 GPU** with DBS=4.

**Optimal strategy:** Rent 1× H100. Run small FLOPs budgets first (fast iteration).
Sequential wall time: 176 hours (~7.3 days).
With 4× H100 running 4 depths in parallel: **~1.8 days** wall time × 4 GPUs = 176 GPU-hrs (same cost).

**Cost: 176 × $2.50 = $440** (or $176 at $1/hr spot pricing)

#### Phase 3: Miniseries (6 runs at compute-optimal)

| Run | N_active | Tokens (ratio×N_sc) | FLOPs | GPU-hrs | Min GPUs |
|-----|----------|-------------------|-------|---------|----------|
| d12 | 239M | 4.12B | 5.91e18 | 4.7 | 1 |
| d14 | 327M | 5.80B | 1.14e19 | 9.1 | 1 |
| d16 | 437M | 7.90B | 2.07e19 | 16.6 | 1 |
| d18 | 671M | 12.4B | 4.99e19 | 40.0 | 1 |
| d20 | 851M | 15.9B | 8.12e19 | 65.1 | 1 |
| d24 | 1.30B | 24.8B | 1.93e20 | 154.9 | **2** |

**Phase 3 total: 290 GPU-hours**

d24 dominates (53% of Phase 3 compute). Run d24 on 2× H100 for memory, 4× for speed.

Sequential wall time on 1-2 GPUs: d12-d20 = 135 hrs, d24 = 77 hrs (2 GPUs) → ~212 hrs (~8.9 days).
With 4× H100 (all running in parallel where possible): **~3 days**.

**Cost: 290 × $2.50 = $725** (or $290 at $1/hr spot)

#### Phase 4: 1B Graduation

| Run | N_active | Tokens | FLOPs | GPU-hrs | GPUs | Wall time |
|-----|----------|--------|-------|---------|------|-----------|
| nanoseek-1b | 1.08B | 22.0B | 1.43e20 | 114.4 | 8 | **14.3 hrs** |

**Cost: 114.4 × $2.50 = $286** (or $114 at $1/hr spot)

### Total Compute Budget

| Phase | FLOPs | GPU-hours | @ $2.50/hr | @ $1.00/hr |
|-------|-------|----------|-----------|-----------|
| 0. Gate 1 | 1.37e17 | 0.1 | $0.30 | $0.10 |
| 1. HP Search + muP | 1.22e19 | 9.8 | $25 | $10 |
| 2. IsoFLOP | 2.20e20 | 176 | $440 | $176 |
| **Research subtotal** | **2.32e20** | **186** | **$465** | **$186** |
| 3. Miniseries | 3.62e20 | 290 | $725 | $290 |
| 4. 1B Graduation | 1.43e20 | 114 | $286 | $114 |
| **Grand total** | **7.37e20** | **591** | **$1,476** | **$591** |

### Recommended GPU Rental Strategy

**Budget-optimized ($591):** Use spot H100 at ~$1/hr (RunPod spot, Lambda Labs)
```
Phase 0-1:  1× H100 80GB    ~10 hrs     ~$10
Phase 2:    1× H100 80GB    ~176 hrs    ~$176    (7.3 days sequential)
Phase 3:    2× H100 80GB    ~145 hrs    ~$290    (d24 needs 2 GPUs for memory)
Phase 4:    8× H100 80GB    ~14 hrs     ~$114
                            ─────────   ────────
                            ~345 hrs    ~$591
```

**Speed-optimized ($1,476 but 5 days total):**
```
Phase 0-1:  1× H100         ~10 hrs     ~$25     Day 1
Phase 2:    4× H100          ~44 hrs    ~$440    Days 2-3
Phase 3:    4× H100          ~73 hrs    ~$725    Days 3-6
Phase 4:    8× H100          ~14 hrs    ~$286    Day 6
                             ─────────   ────────
                             ~141 hrs    ~$1,476
```

**Balanced recommendation ($700-$900):** Spot H100 at $1.00-$1.50/hr
```
Phase 0-1:  1× H100         10 hrs      $10-15
Phase 2:    2× H100         88 hrs      $176-264   (3.7 days)
Phase 3:    2× H100         145 hrs     $290-435   (6 days, sequential)
Phase 4:    8× H100         14 hrs      $114-171
                                        ─────────
                                        $591-$885
```

### Key Decisions

**Why not A100?** H100 is 3.2× faster (989 vs 312 TFLOPS bf16) but only 1.7× the price → 1.9× more cost-efficient. For MoE, H100's 3.35 TB/s HBM3 bandwidth is critical because all 64 experts are in memory.

**Why 35% MFU?** Conservative estimate for MoE. Dense transformers achieve 45-55% on H100. MoE loses ~15% to: expert routing overhead (top-k selection + token dispatch), memory bandwidth (load 8× more params than a dense FFN uses), and load imbalance. If you achieve 40%+ MFU with torch.compile, costs decrease proportionally.

**Can Phase 3 be cheaper?** d24 alone is 155 GPU-hours (53% of Phase 3). If IsoFLOP (Phase 2) shows the scaling law is well-behaved at d12-d20, you could skip d24 and save ~$155-390. Only include d24 if the d20 data point suggests the curve hasn't converged.

**Phase 3 cost vs plan estimate:** The plan estimated ~$120 for Phase 3. The FLOPs calculation shows ~$290 at $1/hr. The discrepancy is because d24 (24.8B tokens × 1.30B active) dominates. Consider d24 as optional validation — the scaling law can be fit from d12-d20 alone.

---

## Scientific Output

Upon completing all phases, NanoSeek will produce:

### 1. MoE Scaling Laws at Nano Scale (Phase 2 + 3)
- L(N, D) power law for MoE architecture — first systematic study below 1B active params with routing diagnostics
- Compare alpha_MoE vs alpha_dense (from nanochat) -> quantify how much more efficiently MoE scales
- Prior work: OLMoE (Muennighoff et al.) and Krajewski et al. studied MoE scaling but without IsoFLOP + H_load + I_spec at sub-1B scale

### 2. muP Transfer for MoE — Novel Contribution (Phase 1 + 3 + 4)
- Does muP (Tensor Programs V) work for MoE models?
- MoE adds routing dynamics (router weights, bias updates, expert load balancing) outside muP's theoretical guarantees
- Empirical test: HPs tuned at d16 (400M active) → transferred to d12, d18, d20 (Phase 1) → validated at compute-optimal training (Phase 3) → deployed at 1B (Phase 4)
- If muP works for MoE: first published evidence at this scale. If it fails: first published negative result, equally valuable.

### 3. MoE Routing Science (Phase 2 + 3 + 4)
- H_load trajectory over 22B tokens -> routing dynamics through training
- I_spec (labeled): I(Expert; Domain) mutual information (`eval/information_metrics.py:250-461`) -> which domains do experts specialize in? When does specialization emerge?
- Dead expert analysis -> how many of 64 experts are actually useful?

### 4. MTP Cost-Benefit at Nano Scale (Phase 1c, optional)
- Does MTP's ~15% compute overhead pay for itself in BPB at 400M active params?
- First compute-allocation analysis of MTP at sub-1B scale

---

## Architecture Reference

### Depth Ladder (aspect_ratio=80, round up to 128)

| Depth | H | Heads | Active | Total | Scaling | Tokens@20 | Est. cost |
|-------|------|-------|---------|---------|---------|-----------|-----------|
| d12 | 1024 | 8 | ~239M | ~965M | ~206M | 4.1B | ~$15 |
| d14 | 1152 | 9 | ~327M | ~1.41B | ~290M | 5.8B | ~$22 |
| d16 | 1280 | 10 | ~437M | ~1.99B | ~395M | 7.9B | ~$35 |
| d18 | 1536 | 12 | ~671M | ~3.20B | ~621M | 12.4B | ~$65 |
| d20 | 1664 | 13 | ~851M | ~4.17B | ~796M | 15.9B | ~$110 |
| d24 | 1920 | 15 | ~1.30B | ~6.65B | ~1.24B | 24.8B | ~$250 |

### Key Invariants (constant across all depths)

- Architecture: Full DeepSeek V3 (locked, not ablated)
- MoE topology: 64 routed experts, top-8, 2 shared, 8 groups, topk_group=4
- Layer 0 is dense FFN, layers 1..D-1 are MoE
- MTP: 1 module with dense FFN (not MoE), shares embed_tokens + lm_head
- Head dimensions: qk_nope=128, qk_rope=64, v=128
- Ratios: dense_FFN=2.56xH, moe_inter=0.375xH, q_lora=0.215xH, kv_lora=0.070xH

### Parameter Counting (Kaplan Convention)

```python
# model.py:1831-1847
scaling_params = active_params - embedding_params
```

Why exclude embeddings? Embeddings are a lookup table — O(1) FLOPs per token regardless of vocab size. Transformer weights do O(1) FLOPs **per parameter per token** (matrix multiply). Only the latter contributes to the compute-capacity relationship L(N, D).

Why "active" not "total"? In MoE, each token only routes through 8 of 64 experts. The 56 inactive experts per layer don't contribute compute for that token. Active params = total - inactive expert params.

### Auto-Compute Cascade (pre_train.py:346-421)

```
--depth -> hidden_size = ceil(depth * 80 / 128) * 128
        -> num_heads = hidden_size / 128
        -> build model -> N_active, N_scaling

--target-param-data-ratio -> tokens = ratio * N_scaling
  OR --target-flops       -> iterations = C / (6 * N_active * batch)

Batch: B = B_REF * (tokens/D_REF)^0.383  (Power Lines paper)
       Round to nearest power of 2

Iterations: tokens / batch
```

Reference point: d16 ablation (D_REF=8.2B tokens, B_REF=524,288 tokens/step).

### muP Hyperparameter Transfer (pre_train.py:423-491)

Two independent scaling factors compose multiplicatively:

```
Factor 1: sqrt(B/B_ref)     — Complete(d)P batch scaling
  Larger batch -> cleaner gradient -> can take bigger step
  Gradient noise proportional to 1/sqrt(B), so LR proportional to sqrt(B)

Factor 2: w_ref/w           — Tensor Programs V width scaling
  Wider network -> each weight contributes less to activation
  LR proportional to 1/width to keep ||delta_h|| = Theta(1) across widths
```

| Parameter group | LR formula | Scales with width? |
|----------------|------------|-------------------|
| Hidden weights (Muon) | base * sqrt(B/B_ref) * (w_ref/w) | Yes |
| Embeddings (AdamW) | base * sqrt(B/B_ref) | No |
| LM head (AdamW) | base * sqrt(B/B_ref) | No |
| Router (AdamW) | constant (3e-4) | No |
| Norms (AdamW) | constant (3e-4) | No |

Weight decay scaling (T_epoch framework, arXiv:2405.13698):
```
lambda = lambda_ref * sqrt(B/B_ref) * (D_ref/D)
```
Keeps `T_epoch = B/(eta * lambda * D)` constant across scales.

### Evaluation Metrics

| Metric | Definition | Healthy range | Source |
|--------|-----------|---------------|--------|
| val_bpb | Bits per byte on validation set (EMA weights) | Lower = better | eval/domain_bpb.py |
| H_load | Shannon entropy of expert load: -sum p_e log(p_e) | > 4.0 / 6.0 bits | model.py:631-638 |
| I_spec | I(Expert; Domain) mutual information | 0.3-0.7 bits | eval/information_metrics.py:250-461 |
| Dead experts | Experts receiving < 1% of token traffic | <= 2-3 / 64 | eval/moe_diagnostics.py:26-116 |
| MTP acceptance | Fraction of MTP predictions matching ground truth | > 75% at end | eval/moe_diagnostics.py:132-228 |
