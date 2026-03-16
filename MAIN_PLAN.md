# NanoSeek Option B — Build-First, Light Validation
## "Karpathy mode with MoE rigor"
### March 2026

---

## Philosophy

Karpathy's insight: at 1B scale, you learn more from building and training than from
fitting power laws. The scaling law sweep in our original plan solves a problem we don't
have — we're not deciding between a $10M 7B run and a $10M 13B run. We're spending $300.

But we're not Karpathy either. We're training MoE, which has failure modes dense models
don't: expert collapse, routing instability, gradient sparsity. These need validation
that dense models can skip.

**Option B = Karpathy's directness + MoE-specific validation.**

---

## What changes from the original plan

| Original (15-run science mode) | Option B (build-first) | Why |
|---|----|---|
| 7 Series A configs (nano-20M through 1B) | 3 configs: muP anchor (~55M active), nano-500M (~441M), NanoSeek-1B (1.08B) | 3 points validate muP HP transfer. No curve fitting. |
| 4 Series B configs (expert count sweep) | **CUT** | log(E) correction is mild (~0.05-0.10). Not worth 4 runs to measure. |
| 5 Series C configs (IsoFLOP sweep) | **CUT** | IsoFLOP determines optimal D/N ratio. At our budget, we train one model — there's no allocation decision to optimize. |
| 7-parameter scaling law fit | **CUT** | 6 data points for 7 parameters was overfitting anyway. |
| fit_scaling_law.py, predict_and_validate.py | **CUT** as mandatory deliverables | Can still write them post-hoc from the 3 data points if curious. |
| 3 stability ablations (A, C, D) | **KEEP** — but Run D IS the anchor model, so effectively 2 extra runs | These answer the one genuinely open question (aux-loss-free vs traditional). |
| RL post-training (3-stage, 3 budgets) | **KEEP** — this is the project's real differentiator | MoE + RL is where the novel science lives. |
| 4 W&B dashboards | **KEEP** — observability is not optional for MoE | Expert collapse is silent. Dashboards catch it. |

**Runs saved: 12.** Cost saved: ~$250-300. Timeline saved: ~4 weeks.

---

## The Plan

### Phase 1: Build the model (Week 1-2)

No change from original plan. This is where the real learning happens.

```
config.py → RMSNorm → RoPE → MLA → Gate → MoE → MTP → Indexer → DSA
→ DecoderLayer → NanoSeekModel → Factory + tests

Quality gate: python -m nanoseek.model.model passes, all 145+ tests pass.
```

Deliverable: correct, tested MoE + MLA + MTP + DSA model.

### Phase 2: Training infrastructure (Week 3)

No change. All components needed for ANY training run.

```
ema_tracker.py → expert_specialization.py → checkpoint_manager_async.py
→ dataset.py (FIM 10%) → pre-train.py (Muon+AdamW, batch warmup, EMA,
   grad clip, aux-loss-free balancing, H_load/I_spec logging)
```

Deliverable: training loop that can train any config from nano-150M to NanoSeek-1B.

### Phase 3: Anchor model + stability ablations (Week 4-5)

**This is where Option B diverges from the original.**

#### Step 1: HP grid search at anchor model (~$40, 1-2 days)

```
Anchor model (muP-corrected):
  16 layers, 480 hidden, 64 experts, top-8
  N_active ≈ 55M, N_total ≈ 350M
  κ = top_k/n_experts = 8/64 = 12.5% (matches NanoSeek-1B exactly)
  Depth = 16 layers (matches NanoSeek-1B exactly — required by muP)

Why this anchor (from Tensor Programs V + μP-MoE + CompleteP-MoE):
  1. Depth MUST match target (muP transfers across width, NOT depth)
  2. Width ≥ 256 for CLT convergence (480 > 256 ✓)
  3. κ = sparsity ratio must be constant (12.5% at both scales ✓)
  4. Only width changes: 480 → 2048 (4.27× width ratio; param expansion 4.4×)

Grid search: ~15-20 short runs (1000 steps each)
Tune: η_attn, η_expert, η_embed, η_unembed, σ_attn, σ_expert, B_ref, λ_ref

  σ_attn, σ_expert = constant-scale multipliers (from CompleteP-MoE)
  These control initialization and forward-pass scaling per parameter group.
  Tuned ONCE at anchor scale, then held constant during transfer.

Transfer rules (muP-corrected, validated by Complete(d)P):
  B = B_ref × (D_target / D_ref)^0.383              # Power Lines (batch size)
  η_hidden = η_ref × √(B / B_ref) × (w_ref / w)    # √B + 1/width (muP hidden weights)
  η_embed  = η_ref_embed × √(B / B_ref)             # √B only (muP input weights)
  η_unembed = η_ref_unembed × √(B / B_ref)          # √B only (muP output weights)
  λ = λ_ref × √(B/B_ref) × (D_ref/D_target)         # T_epoch (weight decay)

  KEY CORRECTION vs nanochat: Hidden weight LRs (attention, expert FFN) get an
  extra 1/width factor. Without this, expert LRs at 1B scale are ~4× too high.
  Source: Tensor Programs V §3 — hidden weights need η ∝ 1/fan_in for Θ(1) updates.
  At anchor (w=480) → 1B (w=2048): correction factor = 480/2048 = 0.234.

MoE-specific constants (don't tune, don't scale):
  Router LR: 3e-4    (muP: router is "output weight" — LR stays constant across width)
  gamma: 0.001, γ_freeze: 0.95, β₂: 0.95, grad_clip: 1.0

Selection criterion: lowest ema_val_bpb at step 1000.

Coordinate check (verify before proceeding to 500M):
  Run anchor at 2 widths (480, 960) for 500 steps.
  If ema_val_bpb differs by >0.05 BPB → transfer rules are broken.
  This costs ~$5 and catches errors before the $30 validation run.
```

#### Step 2: Stability ablations at anchor scale (~$6, same day)

Run D is the anchor model winner from Step 1 — zero marginal cost.

```
Run A: Traditional aux_loss (α=0.001), NO QK-norm    — Mixtral baseline
Run C: Aux-loss-free (γ=0.001), NO QK-norm           — DeepSeek-V3 style
Run D: Aux-loss-free (γ=0.001), QK-norm ON           — nanoseek default (= anchor winner)

All 3 runs: 3000 steps, bad batch injected at step 1500.
Log: H_load, I_spec, loss, spike recovery time.

Decision: A vs C determines load balancing method for 1B run.
          C vs D determines whether QK-norm is worth the complexity.
          (Note SwiGLU confound on C vs D — interpret cautiously.)
```

#### Step 3: Validation run at nano-500M (~$30, 1-2 days)

**This is the key Option B addition.** One intermediate-scale run to validate HP transfer.

```
Config: nano-500M (16 layers, 1280 hidden, 64 experts, top-8)
HPs: auto-scaled from anchor model via Power Lines + √B + 1/width + T_epoch rules
Duration: full Chinchilla-optimal training (8.8B tokens)

What we check:
  1. Does training converge? (not guaranteed — MoE gradient sparsity may break √B scaling)
  2. Is ema_val_bpb reasonable? (compare to published 500M-class MoE results)
  3. Is H_load stable throughout? (expert collapse would appear by now)
  4. Does MTP acceptance rate increase over training? (should reach ~60-70%)

If nano-500M FAILS (diverges, collapses, or loss is unreasonable):
  → The HP transfer rules don't work for MoE at this scale
  → Fall back to grid search at nano-500M (~$50 extra, still cheaper than 15-run sweep)
  → Identify WHICH scaling rule broke (compare attention vs expert weight dynamics)

If nano-500M SUCCEEDS:
  → muP HP transfer works for MoE. Proceed to 1B with high confidence.
  → This is itself a finding: muP-corrected scaling rules (√B + 1/width) transfer to MoE.
```

### Phase 4: NanoSeek-1B training (Week 6-8)

```
Config: NanoSeek-1B (16 layers, 2048 hidden, 64 experts, top-8)
HPs: auto-scaled from anchor model (same muP rules validated at 500M)
Tokens: 22B (Phase 1: 4K dense, Phase 2: 8K DSA)
Duration: ~$300, ~3 days on 8×A100

Monitoring (non-negotiable for MoE):
  - ema_val_bpb every 250 steps
  - H_load every eval step (alert < 2 bits)
  - I_spec at 20%, 50%, 80%, 100%
  - MTP acceptance rate every 2000 steps
  - MFU tracking (target: 47%)
  - W&B dashboards: Training Health, MoE Health, Stability Monitor

Phase transition at 80% of steps:
  → Switch to 8K context, enable DSA
  → Indexer warmup: 1K steps frozen backbone, LR=1e-3
  → YaRN RoPE extension
```

### Phase 5: RL post-training (Week 9-11)

No change from original plan. This is where NanoSeek's novel science lives.

```
3-stage pipeline:
  Stage 1: Reasoning RL (GRPO, 60% budget) — GSM8K, MATH
  Stage 2: Agent RL (GRPO, 25% budget) — tool-use, code exec
  Stage 3: General Alignment (DPO, 15% budget) — safety, helpfulness
  Cross-stage distillation (500 steps)

3 compute budgets: 2%, 5%, 10% of pre-training FLOPs
  → 3 complete pipeline runs

Staging ablation: single-stage vs three-stage at Budget 2

All 4 V3.2 MoE stabilization techniques active in every stage:
  1. Unbiased KL penalty
  2. Off-policy masking
  3. Keep Routing (most critical in Stage 2)
  4. Keep Sampling Mask

Novel measurements:
  - MTP acceptance rate at each stage boundary
  - Test-time scaling curve: accuracy vs tokens at 256/512/1024/2048
  - H_load and I_spec preservation across stages
  - Routing divergence between stages
```

### Phase 6: Reports + packaging (Week 12)

```
Deliverables:
  1. NanoSeek-1B weights (EMA) on HuggingFace
  2. HP_TRANSFER_REPORT.md — do muP-corrected scaling rules transfer to MoE? (3-point validation)
  3. STABILITY_PLAYBOOK.md — aux-loss-free vs traditional at 1B scale
  4. RL_SCALING_REPORT.md — 3-stage pipeline results, test-time scaling curve
  5. W&B dashboards archive (Training Health, MoE Health, Stability, RL Health)
```

---

## Timeline comparison

| Week | Original (16 weeks) | Option B (12 weeks) |
|------|--------------------|--------------------|
| 1-2 | Model rewrite | Model rewrite (same) |
| 3 | Training infra | Training infra (same) |
| 4-5 | Series A sweep (6 runs) | Anchor HP search + stability ablations + nano-500M validation |
| 6-8 | Series B+C + scaling law fit | **NanoSeek-1B training** |
| 8-9 | Stability ablations | (already done in Week 4-5) |
| 9-11 | — | RL post-training |
| 10-12 | NanoSeek-1B training | Reports + packaging |
| 13-14 | RL post-training | **DONE** |
| 15-16 | Reports + packaging | — |

**4 weeks faster. Same model. Same RL. Same stability science.**

---

## Budget comparison

| Item | Original | Option B | Savings |
|------|----------|----------|---------|
| HP grid search (anchor ~55M active) | $40 | $40 | $0 |
| Series A sweep (6 runs) | ~$100 | — | $100 |
| Series B sweep (4 runs) | ~$30 | — | $30 |
| Series C sweep (5 runs) | ~$50 | — | $50 |
| nano-500M validation | — | ~$30 | -$30 |
| Stability ablations (3 runs) | ~$6 | ~$6 | $0 |
| NanoSeek-1B training | ~$300 | ~$300 | $0 |
| RL post-training | ~$80 | ~$80 | $0 |
| **Total** | **~$606** | **~$456** | **~$150** |

---

## What we lose (honestly)

1. **No scaling law fit.** We can't say "we predicted 1B loss within 2% from small runs."
   This was the original plan's headline deliverable. But it was also its weakest science —
   fitting 7 parameters from 6 data points at 20M-800M scale and claiming it predicts 1B
   is a stretch, not a "killer differentiator."

2. **No IsoFLOP curve.** We don't know the compute-optimal D/N_active ratio for our
   architecture. We use Chinchilla's 20:1 as default. If the true optimal is 40:1,
   our model is undertrained by 2×. But at $300, the fix is "train longer next time,"
   not "run 5 more IsoFLOP experiments first."

3. **No expert count sweep.** We don't measure the log(E) coefficient independently.
   The literature says it's mild (γ ≈ 0.05-0.10). We trust the literature.

4. **Weaker portfolio narrative** for "pretraining team" job applications. But stronger
   narrative for "I built, trained, and RL-tuned an MoE from scratch in 12 weeks."

## What we keep (the real value)

1. **Built MoE + MLA + MTP + DSA from scratch.** The deepest possible understanding.
2. **muP HP transfer validated for MoE.** Novel finding: do muP-corrected rules (√B + 1/width) work across MoE scales?
3. **Stability ablation.** The one genuinely open question (aux-loss-free at our scale).
4. **Full 3-stage RL pipeline.** MoE + RL is the frontier. This is where the novel science is.
5. **Production observability.** W&B dashboards, H_load monitoring, MTP tracking.
6. **Trained 1B MoE model.** The actual artifact.

---

## Decision criteria: when to upgrade back to full sweep

If nano-500M validation **fails** (HP transfer doesn't work for MoE), we have two choices:
- **Fix and continue Option B**: Grid search at 500M scale (~$50 extra), then proceed to 1B.
  Total extra cost: ~$50, ~3 days. Still cheaper than the full sweep.
- **Upgrade to full sweep**: If the failure reveals something fundamental about MoE scaling
  that requires systematic investigation (e.g., expert gradient sparsity causes non-monotonic
  loss scaling), then the full Series A sweep becomes scientifically motivated — not as a
  portfolio piece, but as a diagnostic tool. At that point, pivot to the original plan with
  a genuine research question driving it.

The key insight: **run the experiment first, design the investigation second.** Karpathy's
approach works because you discover what actually breaks, not what you predicted would break
from theory alone.

---

## Files affected

### New/Modified
| File | Action | Purpose |
|------|--------|---------|
| `OPTION_B_PLAN.md` | **THIS FILE** | Replaces SCALING_LAB_PLAN.md Pillar 1 sweep design |
| `CLAUDE.md` | **UPDATE** | Change Phase 3 from "15-run sweep" to "anchor + validation" |
| `SCALING_LAB_PLAN.md` | **KEEP as reference** | Original sweep design preserved for potential upgrade |

### Deleted from critical path (but kept in codebase for potential use)
| File | Status | Notes |
|------|--------|-------|
| `fit_scaling_law.py` | Optional | Can write post-hoc from 3 data points if desired |
| `predict_and_validate.py` | Optional | No prediction to validate without full sweep |
| `series_b_experts.yaml` | Cut | Expert count sweep eliminated |
| `series_c_isoflop.yaml` | Cut | IsoFLOP sweep eliminated |
| `run_sweep.sh` | Simplified | 3 runs instead of 15 |

### Unchanged
Everything in Phase 1 (model/), Phase 2 (training_ops/), Phase 5 (RL), and Phase 6 (reports).

---

## Quality gates (updated for Option B)

### Gate 1: Before any training (unchanged)
```
python -m nanoseek.model.model            # all shapes, loss finite
python -m pytest nanoseek/tests/ -v       # all 145+ tests pass
Expert load entropy > 4 bits at init
EMA checkpoint saved at step 100
FIM at ~10% rate
MTP acceptance ~50% at init
```

### Gate 2: Before proceeding from nano-500M to NanoSeek-1B
```
✅ nano-500M converged (no divergence, no NaN)
✅ ema_val_bpb is reasonable for 500M-class MoE (compare to literature)
✅ H_load stayed > 2 bits throughout (no expert collapse)
✅ MTP acceptance rate increased over training (shows MTP is learning)
✅ muP HP transfer (√B + 1/width) produced these results WITHOUT per-scale tuning
   (if manual tuning was needed, document which scaling rule broke)
✅ Coordinate check passed at anchor scale (2 widths within 0.05 BPB)
```

### Gate 3: Stability ablation (unchanged)
```
✅ All 3 runs (A, C, D) completed at anchor scale (~55M active)
✅ Bad batch injected at step 1500
✅ H_load and I_spec logged for all runs
✅ A vs C: I_spec comparison documented
✅ C vs D: spike recovery comparison documented (SwiGLU confound noted)
```

### Gate 4: RL post-training (unchanged from original Gate 5)
```
✅ GSM8K and HumanEval baselines on EMA weights
✅ 3 budgets × 3-stage pipeline
✅ All 4 V3.2 stabilization techniques active
✅ Staging ablation at Budget 2
✅ H_load and I_spec preserved across stages
✅ MTP acceptance rate measured at stage boundaries
✅ Test-time scaling curve plotted
```

---

## The honest trade-off

**Original plan**: More science, less building. 15 sweep runs produce a scaling law paper
but delay the actual model training by 4 weeks. The "science" is a 7-parameter fit on 6
data points — statistically questionable and scientifically narrow (only valid at 20M-1B
scale, doesn't extrapolate to 7B+ where it would matter).

**Option B**: More building, targeted science. You build the model, validate HP transfer
with one intermediate checkpoint, answer the one open stability question, then spend your
time on RL — which is where MoE science is actually underexplored. The 3-point HP transfer
validation (anchor 55M → 500M → 1B) is arguably stronger evidence than a 7-parameter curve fit,
because it tests a MECHANISM (do muP-corrected scaling rules transfer to MoE?) rather than
fitting a CURVE (can we interpolate between points we already measured?).

**Bottom line**: Option B trades a weak scaling law for a strong HP transfer finding,
saves 4 weeks and $150, and gets you to RL faster — where the real open questions live.
