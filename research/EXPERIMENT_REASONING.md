# NanoSeek Experiment Reasoning — First Principles

## Why This Document Exists

Every experiment in NanoSeek costs GPU-hours and calendar time. This document captures
the **first-principles reasoning** behind each experiment: not just WHAT we run, but
WHY that experiment is the right one, what physics of neural networks makes it necessary,
what we expect to learn, what failure looks like, and what we'd do if it fails.

This is the missing OPTION_B_PLAN.md — the active experimental plan for NanoSeek.

---

## The Central Problem

We want to train a 1.08B-active MoE model (NanoSeek-1B). The training costs ~$300.
We have ~$400 total GPU budget. This means we get ONE shot at the 1B run.

**The fundamental question:** How do we set ~8 continuous hyperparameters (LRs, weight
decay, batch size, momentum) for an architecture (MoE+MLA+MTP+Muon) that no paper has
trained at exactly this configuration?

The naive approach — just pick "reasonable" values and hope — has a ~50% chance of
wasting $300 on a diverged or suboptimal run. The smart approach is to **validate
at cheap scale first**, which costs ~$75 total but reduces the risk of the $300 run
from ~50% to ~5%.

---

## The Validation Path: 5 Experiments in Dependency Order

```
Experiment 1: Anchor HP Grid Search      (~$40, 15-20 runs × 3000 steps)
     ↓ produces: best HP ratios
Experiment 2: Coordinate Check            (~$5, 2 runs × 500 steps)
     ↓ validates: muP scaling rules work
Experiment 3: Stability Ablations A,C,D   (~$6, 3 runs × 3000 steps)
     ↓ validates: which stabilizers to use
Experiment 4: nano-500M Validation Run    (~$30, 1 full training run)
     ↓ validates: HP transfer at intermediate scale
Experiment 5: NanoSeek-1B Training        (~$300, 1 full training run)
     ↓ produces: final model
```

Total pre-validation cost: ~$81. Total project: ~$381.
Without validation: $300 gamble. With validation: $381 for high-confidence 1B HPs.

Each experiment below is justified from first principles of optimization theory,
MoE dynamics, and scaling laws.

---

# Experiment 1: Anchor HP Grid Search

## The Physics

A neural network's training dynamics are governed by the **learning rate to curvature
ratio**. If η (learning rate) is too high relative to the loss surface curvature,
gradient steps overshoot the minimum → loss spikes or divergence. If η is too low,
convergence is slow → wasted tokens.

The optimal η depends on:
1. **Model width** — wider networks have smoother loss surfaces (CLT: more parameters
   averaging → lower gradient variance). This is the basis of muP (Tensor Programs V).
2. **Batch size** — larger batches give cleaner gradient estimates → can take bigger
   steps. This is the basis of Complete(d)P's √B scaling.
3. **Architecture-specific factors** — MoE has 8× gradient sparsity per expert (only
   1/top_k of tokens flow through each expert), MLA has low-rank bottlenecks that
   amplify gradient norms, Muon optimizer uses spectral normalization that interacts
   differently with these structures than Adam.

**No paper has tuned HPs for the combination (MoE + MLA + MTP + Muon + SwiGLU) at
any scale.** DeepSeek V3 uses Adam, not Muon. Nanochat uses Muon, but on dense GPT,
not MoE. We're in uncharted territory for the optimizer-architecture pairing.

## Why a Grid Search (Not Random Guessing)

The HP landscape for this architecture has **correlated ridges**, not independent axes:

- `matrix_lr` (Muon LR for hidden weights) and `weight_decay` are anticorrelated —
  higher LR needs lower WD to avoid over-regularization
- `embedding_lr` and `unembedding_lr` have a natural ratio (embeddings see every token,
  lm_head sees every token → both scale with vocab, but embed is input and lm_head is
  output in muP theory, so their optimal values are linked but not identical)
- `matrix_lr` and `router_lr` interact through the MoE routing: if expert weights change
  fast (high matrix_lr) but router changes slowly (low router_lr), routing decisions
  become stale → expert thrashing

A grid search over the **correlated** HPs, with the **uncorrelated** ones fixed, is
the right approach. Random search would waste budget exploring irrelevant corners.

## What We Search

### Tuned at anchor (4 variables, the "hidden weight" family):

| HP | Range | Why This Range | First Principle |
|----|-------|----------------|-----------------|
| `matrix_lr` | [0.01, 0.02, 0.04] | Nanochat's dense GPT uses 0.02. MoE experts have 8× gradient sparsity → could need higher LR. But MLA bottleneck amplifies gradients → could need lower. Bracket around 0.02. | Gradient variance ∝ 1/effective_batch. Expert sparsity = 1/8. |
| `embedding_lr` | [0.15, 0.3, 0.6] | Nanochat uses 0.3. Embeddings see ALL tokens (no sparsity). But vocab=32K (not 50K) → embeddings are denser, may tolerate higher LR. | Input weight in muP — LR scales with √B only, no 1/width. |
| `unembedding_lr` | [0.004, 0.008, 0.016] | Nanochat uses 0.008. Same reasoning as embedding but output weight in muP. | Output weight in muP — same scaling as embedding. |
| `weight_decay` | [0.05, 0.1, 0.2] | Nanochat uses 0.28 for dense. MoE has sparser gradients → WD has proportionally larger effect per expert → might need lower WD. | T_epoch framework: λ ∝ √(B/B_ref) × (D_ref/D). |

### Fixed at anchor (not searched — either settled science or muP-constant):

| HP | Value | Why Fixed | Source |
|----|-------|-----------|--------|
| `router_lr` | 3e-4 | muP theory: router is "output weight" → LR constant across scales. Searching it at anchor then fixing is wrong — it must be THE SAME at all scales. | μP-MoE (arXiv:2508.09752) |
| `norm_lr` | 3e-4 | 1D parameters — constant in muP. | Tensor Programs V |
| `β₂` | 0.95 | Settled science — universal across frontier MoE models. No value in searching. | DeepSeek V2/V3, Ling MoE |
| `grad_clip` | 1.0 | Settled science — universal across frontier MoE models. | DeepSeek V2/V3, Chameleon |
| `gamma` | 0.001 | Load-balance bias update rate — not gradient-based, doesn't interact with LR. | DeepSeek V3 paper |
| `MTP λ schedule` | 0.3→0.1 at 60% | Architecture parameter, not optimizer HP. Same across scales. | DeepSeek V3 paper |

### Why 3 values per HP (not 5, not 2)?

- **2 values**: can't distinguish monotonic trend from U-shape. If [0.01, 0.04] both
  give similar loss, is 0.02 better or worse? Can't tell.
- **3 values**: minimum to detect curvature. If [0.01, 0.02, 0.04] gives losses
  [3.1, 2.9, 3.0], we know the optimum is near 0.02 (U-shape detected).
- **5 values**: 5^4 = 625 combos. At 3000 steps each, this is ~$160. Exceeds our
  pre-validation budget.
- **3 values**: 3^4 = 81 combos. Random sample 15-20 → ~$40. Fits budget.

### Why Random Sample 15-20 from 81 (Not Full Grid)?

Bergstra & Bengio (2012) proved random search finds good HPs in fewer trials than
grid search when some dimensions matter more than others. In our case, `matrix_lr`
probably matters most (it controls the majority of parameters), while `unembedding_lr`
matters least (only 1 parameter matrix). Random sampling concentrates more trials
near the effective dimensionality of the search space.

15-20 runs gives ~95% probability of sampling within 0.05 BPB of the global optimum
for a 4D grid with 3 levels per dimension (by coupon collector analysis).

## Anchor Config Details

```
Architecture: 16 layers, 480 hidden, 64 experts, top-8
N_active ≈ 77M, N_total ≈ 282M
```

### Why 16 layers (matching 1B)?

muP (Tensor Programs V) transfers HPs across WIDTH, not depth. The theory derives
that activation updates scale as Θ(1) when η ∝ 1/fan_in — but this holds only when
the computational graph structure is the same. Changing depth changes the residual
stream dynamics:

- 16 layers → 16 residual additions → gradient flow through 16 sequential transformations
- 8 layers → 8 residual additions → fundamentally different gradient flow

If anchor has 8 layers but target has 16, the optimal LR for the anchor includes
a "depth factor" that doesn't transfer. Cerebras muP guide confirms: "depth mismatch
invalidates transfer."

**Cost of matching depth**: 16 layers at 480 hidden is ~2× more expensive per step
than 8 layers. But each grid search run is only 3000 steps — the per-run cost is
still small (~$2-3).

### Why 480 hidden (not 256, not 768)?

- **Minimum 256**: Cerebras muP guide says CLT convergence requires ≥256 hidden for
  the gradient noise to be approximately Gaussian (prerequisite for muP scaling laws).
  Below 256, the scaling rules break.
- **480 gives 4.27× expansion to 2048**: The muP width scaling factor is w_ref/w =
  480/2048 = 0.234. This is large enough to be meaningful (if transfer works at 4.27×,
  it probably works at any reasonable factor) but small enough that the anchor is cheap.
- **768 would be too expensive**: At 768 hidden, the anchor would be ~3× larger,
  making grid search cost ~$120 instead of ~$40.

### Why 64 experts, top-8 (not 32/top-4 or 16/top-2)?

μP-MoE (arXiv:2508.09752) proves that muP transfer works across width AND expert
count, but **NOT across granularity** (κ = top_k/n_experts). If κ changes between
anchor and target, the effective batch size per expert changes, which invalidates
the gradient variance assumptions in the scaling rules.

```
Anchor:    κ = 8/64 = 12.5%
nano-500M: κ = 8/64 = 12.5%  ← same
NanoSeek-1B: κ = 8/64 = 12.5%  ← same
```

If we used 32/top-4 at anchor: κ = 4/32 = 12.5% (same ratio), but the per-expert
effective batch is different because n_group and topk_group routing dynamics change
with E. CompleteP-MoE's constant-κ requirement is about the ROUTING MECHANISM
behavior, not just the numerical ratio.

### Training length: 3000 steps

**Why 3000?**

- At 64 × 4096 = 262K tokens per step, 3000 steps = ~786M tokens = ~10× N_active.
- Chinchilla-optimal is 20× — but we're not trying to train a good model, just find
  good HP ratios. The relative ranking of HPs stabilizes much earlier than convergence.
- Empirically (nanochat experience, corroborated by muP literature): HP rankings are
  stable after 5-10× N_active tokens. Beyond that, more steps add precision but
  don't change which config wins.
- At 1000 steps: rankings are noisy (EMA hasn't converged, batch warmup still active).
- At 3000 steps: rankings are stable, EMA has had 300 updates, LR schedule has
  completed warmup + entered constant phase.

**What we measure**: `ema_val_bpb` at step 3000 (RULE 3). Not train loss (noisy),
not raw val loss (optimizer-state-dependent, RULE 1).

## Selection Criteria

After 15-20 runs complete:

1. **Filter**: Remove any run where H_load < 2 bits at any point (expert collapse)
2. **Filter**: Remove any run with loss spike > 5× moving average (instability)
3. **Filter**: Remove any run with NaN loss (divergence)
4. **Rank**: Sort remaining by final `ema_val_bpb` (lower is better)
5. **Select**: Top config becomes the reference HPs for muP transfer

**If all runs are filtered out**: The search space is wrong. Possible causes:
- `matrix_lr` range is entirely too high for MoE+Muon → extend down to [0.005, 0.01, 0.02]
- Expert gradient sparsity needs the √(1/8) correction that μP-MoE doesn't include
  → add correction and re-run grid
- Architecture has a bug (e.g., norm_topk_prob not applied) → check model code first

**If multiple runs tie within noise**: Pick the one with highest H_load (most balanced
expert utilization), as this indicates the most robust routing behavior.

## What This Experiment Produces

- The 4 reference HPs: `matrix_lr_ref`, `embedding_lr_ref`, `unembedding_lr_ref`, `wd_ref`
- These get auto-scaled to any width via muP rules (Experiment 2 validates this)
- A W&B table of all grid search runs for the paper appendix

---

# Experiment 2: Coordinate Check

## The Physics

muP says that if we scale the learning rate as η ∝ 1/width for hidden weights, the
activation updates ||Δh|| stay Θ(1) across widths. This is a **necessary condition**
for HP transfer — if it fails, the LRs we found at anchor scale will be wrong at 1B.

The coordinate check tests this directly:

```
1. Train at width=480 for 500 steps → record ema_val_bpb
2. Train at width=960 for 500 steps → record ema_val_bpb
   (with η_hidden auto-scaled: η × 480/960 = η/2)
3. Compare: |bpb_480 - bpb_960| should be < 0.05
```

### Why 500 steps (not 3000)?

The coordinate check doesn't need convergence. It tests whether the **trajectory**
is similar — are both models moving through loss landscape at similar rates? 500 steps
is enough to see if one model is diverging while the other converges, or if one is
learning 5× faster than the other.

### Why width=960 (not 1024 or 1280)?

960 = 2 × 480. This gives the cleanest scaling factor (exactly 2×). If muP works
at 2× scaling, it will work at 4.27× (480→2048). If it fails at 2×, it definitely
fails at 4.27×.

Also, 960 is small enough that the run is cheap (~$2.50), but large enough that
the muP correction is meaningful (η_hidden is halved, not just slightly reduced).

### What "failure" means

**|bpb_480 - bpb_960| > 0.05**: The muP scaling rules are WRONG for this architecture.

Possible causes:
1. **MoE expert sparsity needs additional correction**: Each expert sees 1/8 of tokens.
   The effective gradient variance per expert is 8× higher. Maybe η_expert should be
   η_ref × √(B/B_ref) × (w_ref/w) × √(1/8). μP-MoE says no, but their theory
   was validated on standard MoE without MLA.

2. **MLA bottleneck norms interact with muP**: MLA has RMSNorm at the compressed
   latent stage. This changes the gradient flow compared to standard attention.
   muP was derived for standard attention — MLA might need different scaling.

3. **Muon optimizer doesn't obey muP**: muP was derived for SGD and Adam. Muon uses
   spectral normalization (Newton-Schulz iterations) which changes the effective
   learning rate in a width-dependent way. Muon's spectral normalization already
   incorporates an implicit 1/√fan_in — does this stack with muP's 1/width?

**Diagnostic**: If the coordinate check fails:
- Check whether attention weights diverge but expert weights are fine → MLA issue
- Check whether expert weights diverge but attention is fine → sparsity correction needed
- Check whether everything diverges → Muon + muP interaction

**Cost of failure**: $5 wasted on the coordinate check + need to debug before the
$30 validation run. This is exactly why the coordinate check exists — it catches
muP implementation errors at $5, not $30 or $300.

## What This Experiment Produces

- Binary: muP transfer works / doesn't work at 2× scaling
- If works: green light for Experiment 4 (500M validation)
- If fails: diagnostic information about which component breaks muP

---

# Experiment 3: Stability Ablations (Runs A, C, D)

## The Central Question

NanoSeek's training recipe includes two "belt and suspenders" stabilizers:
1. **Aux-loss-free load balancing** (bias-based, γ=0.001) — from DeepSeek V3
2. **QK-norm** (RMSNorm on Q,K after up-projection) — from Qwen3/Llama 4

Are BOTH necessary? Is EITHER sufficient? What happens without them?

This matters because each stabilizer has a cost:
- Aux-loss-free: requires bias tracking per expert, complicates checkpointing
- QK-norm: adds 2 RMSNorm calls per layer per forward pass (~2% compute overhead)

If we can remove one without compromising stability, we save complexity and compute.
More importantly: the answer tells us something about MoE training physics at
small scale that no paper has measured.

## Run A: Traditional Aux Loss + No QK-Norm (Mixtral Baseline)

### What this configuration is

The "industry default" — what Mixtral, Qwen3, Llama 4 all use:
- Traditional auxiliary loss pushes load toward uniform via gradient signal
- No QK-norm (Mixtral) or QK-norm (Qwen3/Llama4) — we test the "no" case

### The physics of traditional aux loss

Traditional aux loss adds a term to the training objective:

```
L_total = L_main + α × L_aux
L_aux = Σ_i (f_i × p_i)    where f_i = fraction of tokens routed to expert i
                                   p_i = average routing probability for expert i
```

This creates a **gradient conflict**: the model wants to minimize L_main (route tokens
to the best expert) but also minimize L_aux (route tokens uniformly). These objectives
are directly opposed — the optimal expert for a token is usually NOT the least-loaded one.

The gradient of L_aux interferes with the gradient of L_main at every step. DeepSeek
showed this interference costs ~0.06 PPL at their scale (arXiv:2408.15664).

### Why test it

Run A is the **control group**. Without it, we can't measure whether aux-loss-free
actually helps. If Run C (aux-loss-free, no QK-norm) beats Run A, we know the
improvement comes from removing gradient interference, not from some other factor.

### Expected outcome

- **Loss**: Slightly worse than C and D (~0.06 PPL penalty from gradient interference)
- **H_load**: Should still be > 2 bits (traditional aux loss does prevent collapse)
- **I_spec**: Potentially LOWER than C/D (gradient interference pushes toward uniformity,
  suppressing natural expert specialization)
- **Spike recovery**: Worst of the three — no QK-norm means attention logits can grow
  unboundedly after the bad batch injection at step 1500

## Run C: Aux-Loss-Free + No QK-Norm (DeepSeek V3 Style)

### What this configuration is

Exactly what DeepSeek V3 uses — the paper's recommended recipe:
- Aux-loss-free: bias-based load balancing with γ=0.001
- No QK-norm: DeepSeek V3 achieved zero irrecoverable spikes without it

### The physics of aux-loss-free

Instead of adding a loss term, aux-loss-free adjusts expert biases directly:

```
After each step:
  b_i -= γ × (load_i - mean_load) / mean_load

Where b_i is added to expert i's routing score before top-k selection.
```

This is a **control system**, not an optimization objective. It's like a thermostat:
if expert i is overloaded, reduce its routing score slightly. If underloaded, increase it.

**Why this is better in theory**:
- No gradient interference — L_main gradients are never contaminated
- The bias update is decoupled from backpropagation entirely
- At γ=0.001, the bias changes are tiny — they nudge routing without disrupting learning
- γ freezes at 0 after 95% of training (RULE 2) to avoid late-stage routing noise

### The risk at our scale

DeepSeek V3 validated this at 671B total parameters, 14.8T tokens. At our scale
(4.75B total, 22B tokens):
- We have 64 experts vs V3's 256 → each expert gets 8× more tokens → load imbalance
  is naturally lower → bias correction may be less critical
- We train for 22B tokens vs 14.8T → gradient interference has less time to accumulate
  → the advantage of aux-loss-free over traditional may be smaller

**This is the hypothesis**: at small scale and short training, does aux-loss-free
still outperform? If not, at what scale does the crossover happen?

### Expected outcome

- **Loss**: Better than A (~0.06 PPL improvement from no gradient interference)
- **H_load**: Should be > 3 bits (bias-based balancing is more precise than loss-based)
- **I_spec**: Higher than A (experts can specialize freely without uniformity pressure)
- **Spike recovery**: Moderate — no QK-norm means attention logits can still grow

## Run D: Aux-Loss-Free + QK-Norm (NanoSeek Default)

### What this configuration is

Our "belt and suspenders" approach — combining DeepSeek's routing innovation with
Qwen3/Llama4's attention stabilizer.

### The physics of QK-norm with MLA

MLA has two stages of normalization:
1. **Bottleneck RMSNorm**: Applied to compressed latent vectors c_q and c_kv
   - Prevents magnitude drift in the compressed space
   - Always present (removing it breaks MLA entirely)

2. **QK-norm** (what we're ablating): Applied to reconstructed Q and K after up-projection
   - Prevents attention logit growth: score_ij = (Q_i · K_j) / √d
   - Without QK-norm: ||Q|| and ||K|| can grow over training → score_ij grows → softmax saturates → attention becomes a one-hot → loss of gradient signal

**The question**: Does MLA's bottleneck norm already prevent attention logit growth?

**Argument for "yes, QK-norm is redundant"**:
- The bottleneck norm constrains ||c_q|| and ||c_kv||
- Since Q = W_uq(c_q) and K = W_uk(c_kv), if ||c_q|| is bounded and W_uq doesn't grow,
  then ||Q|| is bounded
- But W_uq CAN grow — it's a learned weight matrix, and nothing constrains its spectral norm
- So bottleneck norm is necessary but not sufficient

**Argument for "yes, QK-norm IS needed"**:
- Wortsman et al. (arXiv:2309.14322) shows that attention logit growth is the #1 cause
  of training instability in transformers at scale
- Even with bottleneck norms, the up-projection matrices can amplify the signal
- At 16 layers, there are 16 sequential attention computations — logit growth compounds
- QK-norm adds a hard bound: ||Q_normed|| = 1, regardless of W_uq's spectral norm

### What this run answers about MLA specifically

If Run D (QK-norm ON) ≈ Run C (QK-norm OFF) in stability:
→ MLA's bottleneck norms ARE sufficient for attention stability
→ QK-norm is a free 2% compute saving for MLA architectures
→ DeepSeek's choice to skip QK-norm was correct even at small scale

If Run D > Run C in stability (especially after spike injection):
→ QK-norm adds value BEYOND bottleneck norms
→ The up-projection matrices amplify signal enough to cause logit growth
→ Qwen3/Llama4's choice to add QK-norm was correct
→ At small scale (16 layers, 480 hidden), this effect is large enough to measure

### Expected outcome

- **Loss**: Same as C (QK-norm doesn't affect final loss, only stability)
- **H_load**: Same as C (QK-norm doesn't affect routing)
- **I_spec**: Same as C (QK-norm doesn't affect expert specialization)
- **Spike recovery**: BEST — QK-norm prevents attention logit explosion after bad batch

### SwiGLU confound (Allen-Zhu Part 3.3) — documented here for honesty

Allen-Zhu's Physics of Language Models (Part 3.3) shows that SwiGLU (GatedMLP) has
harder early-training dynamics than standard MLP. The gating mechanism creates regions
of near-zero gradient that slow early learning.

This creates a confound for the C vs D comparison: if Run D (QK-norm ON) shows better
early-training stability, it could be because:
1. QK-norm genuinely stabilizes attention → the conclusion we want
2. QK-norm compensates for SwiGLU's early instability → a different mechanism

**We cannot distinguish these two causes with only 3 runs.** To isolate them, we'd
need a 4th run with standard MLP instead of SwiGLU — cost: ~$2 extra.

**Decision**: Document the confound. Do NOT add the 4th run unless C vs D shows a
large stability difference (>0.1 BPB gap). If the gap is small, the confound
doesn't matter.

## Bad Batch Injection at Step 1500

### Why inject failure

Training stability isn't just about normal conditions — it's about recovery from
perturbation. A model that trains smoothly but can't recover from a data corruption
is fragile. In production, data corruption happens (disk errors, corrupt shards,
encoding bugs).

### Why step 1500 (not step 500, not step 2500)

- **Step 500**: Too early. Model is still in warmup phase. LR hasn't reached peak.
  Batch warmup may still be active. Testing recovery here tests warmup robustness,
  not steady-state resilience.
- **Step 1500**: Mid-training. LR is at peak or near peak. Expert routing has
  stabilized (H_load should be > 3 bits by now). Model has learned enough that
  a perturbation tests recovery of LEARNED representations.
- **Step 2500**: Too late. Only 500 steps of recovery time before we measure at
  step 3000. May not be enough to distinguish "recovered" from "still recovering."

### What "bad batch" means physically

Replace 100% of tokens in one batch with uniform random token IDs.

This is the **worst-case data perturbation** — every token is wrong. The gradient
from this batch is pure noise. The model's response to this gradient tells us:

1. **Gradient clipping (always active)**: Clips the noise gradient to norm 1.0.
   Without clipping, the noise gradient could be 10-100× normal, causing weight
   updates that destroy learned representations.

2. **EMA (always active)**: The EMA weights are barely affected by one bad step
   (decay=0.9999 → one step contributes 0.01% to EMA). This is a safety net —
   even if raw weights take a hit, EMA weights preserve the pre-perturbation state.

3. **Momentum buffer (β₂=0.95)**: The bad gradient enters the momentum buffer
   and takes ~20 steps to decay (1/[1-0.95] = 20). With β₂=0.999, it would take
   ~1000 steps — this is why β₂=0.95 is universal for MoE.

4. **Expert routing**: The bad batch routes randomly across experts (uniform tokens
   → uniform routing). This briefly disrupts the routing bias b_i. With aux-loss-free
   (Run C, D), the bias self-corrects in ~10 steps (γ=0.001). With traditional aux
   loss (Run A), recovery depends on the aux loss gradient overpowering the noise.

## Measurements for ALL 3 Runs

```
Logged every 10 steps (via W&B):
  - train_loss, main_loss, mtp_loss, aux_loss
  - H_load (load-balance entropy)
  - grad_norm (detects gradient spikes)
  - per-group LRs (verify muP scaling is correct)

Logged every 500 steps:
  - ema_val_bpb (RULE 3)
  - I_spec (specialization MI: do experts develop semantic roles?)
  - Domain routing heatmap (which tokens go to which experts)
  - load_per_expert histogram (distribution of token counts across 64 experts)

Post-hoc analysis:
  - A vs C comparison: ema_val_bpb, I_spec, H_load at step 3000
  - C vs D comparison: ema_val_bpb, H_load trajectory around step 1500 (spike recovery)
  - Recovery time: steps to return within 10% of pre-spike loss
```

## Decision Tree After Ablations

```
IF Run D best (aux-loss-free + QK-norm):
  → Use as NanoSeek default (already is)
  → QK-norm adds measurable value at 16 layers
  → Report: "QK-norm recommended for MoE+MLA below 32 layers"

IF Run C ≈ Run D:
  → Remove QK-norm (save 2% compute)
  → Report: "MLA bottleneck norms sufficient for attention stability"
  → DeepSeek's no-QK-norm choice validated at small scale

IF Run A ≈ Run C:
  → Aux-loss-free doesn't help at this scale/duration
  → Hypothesis: 22B tokens is too short for gradient interference to accumulate
  → Use simpler traditional aux loss for NanoSeek (less code complexity)
  → Report: "Aux-loss-free advantage may require >100B tokens to manifest"

IF Run A best (unexpected):
  → Something is wrong with our aux-loss-free implementation
  → Check: is gamma_freeze_ratio=0.95? Is bias update formula correct?
  → This would be a debugging signal, not a scientific finding
```

---

# Experiment 4: nano-500M Validation Run

## The Physics

muP (Tensor Programs V) proves that for a specific class of parameterizations
(the "maximal update parameterization"), hyperparameters found at one width
transfer to any other width with zero retuning. The theory derives exact
scaling rules:

```
For hidden weights:  η_target = η_ref × (w_ref / w_target)
For input weights:   η_target = η_ref (constant)
For output weights:  η_target = η_ref (constant)
```

Complete(d)P extends this to batch size: η ∝ √B.

Combined: η_hidden = η_ref × √(B/B_ref) × (w_ref/w)

**The theory is proven for dense transformers with Adam optimizer.**
It is NOT proven for:
- MoE architectures (μP-MoE extends it, but only validated up to ~1B total)
- MLA attention (non-standard attention mechanism)
- Muon optimizer (spectral normalization, not Adam)
- The combination of all three

The nano-500M validation run tests whether the theory holds for our specific
architecture-optimizer combination.

## Why 500M (Not 200M or 800M)

### Width = 1280 (midpoint on log scale)

```
Anchor:  480  →  log(480)  = 6.17
nano-500M: 1280 → log(1280) = 7.15
NanoSeek-1B: 2048 → log(2048) = 7.62
```

The 500M sits at 67% of the way from anchor to 1B on the log scale. This means:
- If transfer works at 480→1280 (2.67× scaling), it strongly predicts 480→2048 (4.27×)
- The 500M is different enough from the anchor to catch real transfer failures
- But not so different that a failure is uninformative about what went wrong

### N_active ≈ 441M

At 441M active parameters, Chinchilla-optimal training is ~8.8B tokens (20× N_active).
This is expensive enough to be meaningful (~$30) but cheap enough to be affordable.

### All architectural ratios constant

```
nano-500M: 16 layers, 1280 hidden, 64 experts, top-8
  κ = 8/64 = 12.5%           (same as anchor and 1B)
  moe_inter/hidden = 0.375   (same)
  n_shared_experts = 2       (same)
  n_group = 8                (same)
  first_k_dense = 2          (same)
```

ONLY width changes: 480 → 1280. This is the muP requirement.

## What We Measure

### Primary metric: ema_val_bpb at convergence

The HP transfer prediction is:

```
Given: η_ref (from anchor grid search)
Compute: η_500m = η_ref × √(B_500m/B_ref) × (480/1280)
Train nano-500M with η_500m
Measure: ema_val_bpb_500m
```

If muP works: ema_val_bpb_500m should be "reasonable for a 500M-class MoE model."

**What "reasonable" means**: We don't have a precise prediction for the absolute BPB
(that would require a scaling law we haven't fit yet). But we can sanity-check against:
- OLMoE-1B at similar active parameter count
- Dense models at 441M parameters (should be worse than MoE due to lower total params)
- The anchor's ema_val_bpb (should be better than anchor — more capacity)

### Secondary metrics

- **H_load > 2 bits throughout**: Expert collapse detection
- **MTP acceptance rate increases over training**: MTP module is learning
- **No divergence, no NaN**: Basic training health
- **MFU within expected range**: Training loop is efficient

### The go/no-go decision

```
IF nano-500M converges AND ema_val_bpb is reasonable AND no manual HP tuning needed:
  → muP transfer WORKS for MoE+MLA+Muon
  → Proceed to 1B with auto-scaled HPs
  → This is a publishable result (muP validated for novel architecture combo)

IF nano-500M converges BUT ema_val_bpb is unexpectedly bad:
  → muP transfer PARTIALLY works (no divergence, but suboptimal)
  → Diagnose: which parameter group's LR is wrong?
    - If expert FFN LR too high → add √(1/8) sparsity correction, re-run
    - If attention LR too high → MLA bottleneck changes the effective fan_in
    - If embedding LR wrong → vocab=32K interaction
  → Fix the scaling rule, re-run 500M (~$30 extra)
  → Still cheaper than blind 1B run

IF nano-500M diverges:
  → muP transfer FAILS for this architecture
  → Fall back to: run 3-5 configs at 500M scale with different LR multipliers
  → This is more expensive (~$90-150) but still cheaper than 1B guessing
  → The failure itself is a research finding: "muP does not transfer to
    MoE+MLA+Muon — here's what breaks and why"
```

## Why We Don't Skip to 1B

The expected value calculation:

```
Without 500M validation:
  P(success) ≈ 0.6  (muP works, HPs happen to be right)
  P(failure) ≈ 0.4  (muP fails, or HPs slightly wrong → bad model)
  E[cost] = 0.6 × $300 + 0.4 × $600 = $420  (need to re-run on failure)

With 500M validation:
  P(validation catches problem) ≈ 0.9
  E[cost] = $30 + 0.9 × $300 + 0.1 × $600 = $360

Savings: $60 expected value + avoids 2-week delay from failed 1B run.
```

The $30 validation run is cheap insurance.

---

# Experiment 5: NanoSeek-1B Training

## Prerequisites

ALL of these must be true before starting:

1. Anchor grid search complete → reference HPs selected
2. Coordinate check passed → muP scaling validated at 2×
3. Stability ablations complete → stabilizer configuration decided
4. nano-500M validation passed → HP transfer confirmed at 2.67×

## What's different from 500M

```
Width: 1280 → 2048 (1.6× wider)
N_active: 441M → 1.08B (2.4× more active params)
N_total: ~1.7B → ~4.75B (2.8× more total params)
Training tokens: ~8.8B → 22B (2.5× more tokens)
Training time: ~4 hours → ~14 hours
Cost: ~$30 → ~$300
```

### Phase 2 (DSA at 8K) — 1B only

At 80% of training (step ~4300), switch to Phase 2:
- Context length: 4096 → 8192
- DSA: enabled (indexer selects top-k tokens)
- YaRN: enabled (interpolate RoPE to 8K)
- Batch size: halved (2× context = 2× memory per sequence)

**Why only 1B gets Phase 2**: See earlier discussion. DSA is a capability feature
(long context), not a scaling variable. The anchor and 500M don't need it.

**Indexer warmup**: 1K steps with frozen backbone, indexer-only training at LR=1e-3.
Then unfreeze backbone for joint training. This gives the indexer time to learn
attention patterns before the backbone starts changing under it.

## Risk Factors

1. **OOM at 8K context**: Phase 2 doubles sequence length. Memory requirement
   approximately doubles for attention. Mitigation: gradient checkpointing is
   already enabled; reduce device_batch_size if needed.

2. **Loss spike at phase transition**: Switching context length + attention mode
   simultaneously is a large perturbation. Mitigation: 100-step warmup at Phase 2
   entry with reduced LR.

3. **Expert routing shift at Phase 2**: Longer sequences may route differently
   (more diverse content per sequence → different expert specialization).
   This is expected and not a problem — monitor H_load to ensure no collapse.

## What "Done" Looks Like

- Final `ema_val_bpb` on held-out validation set
- No expert collapse (H_load > 2 bits throughout)
- MTP acceptance rate > 75% by end of training
- Checkpoint uploaded to HuggingFace
- W&B dashboards show complete training trajectory

---

# Summary: The Scientific Story

Each experiment answers one question. Together, they build a narrative:

```
Q1 (Grid Search):  What HPs work for MoE+MLA+Muon at anchor scale?
Q2 (Coord Check):  Do muP scaling rules hold for this architecture?
Q3 (Ablations):    Which stabilizers does MoE+MLA need at small scale?
Q4 (500M):         Does HP transfer work at intermediate scale?
Q5 (1B):           Does the full recipe produce a good model?
```

**If all go well**: We've validated muP HP transfer for MoE+MLA (novel finding),
determined the minimal stabilizer set for small-scale MoE+MLA (applied finding),
and produced a trained 1B model (practical output).

**If something fails**: The failure is itself informative. muP failing for MoE
would be a contribution (negative result). Stability ablations showing unexpected
results would revise best practices. Each experiment is designed so that failure
teaches us something publishable.

---

# Appendix: Decision Log

| Decision | What | Why | Alternative Considered | Why Rejected |
|----------|------|-----|----------------------|--------------|
| 3-point validation (not 15-run sweep) | Anchor → 500M → 1B | muP theory + budget constraint ($400 total) | Full 15-run Series A+B+C sweep from SCALING_LAB_PLAN.md | Costs ~$200 for sweep alone, leaving insufficient budget for 1B run. Archived in SCALING_LAB_PLAN.md for potential upgrade. |
| Grid search (not Bayesian optimization) | 15-20 random samples from 3^4 grid | Simple, parallelizable, sufficient for 4D search | W&B Sweeps with Bayesian agent | Bayesian optimization benefits most with >50 trials. At 15-20 trials, random search is nearly as good and much simpler. |
| 3 ablation runs (not 5 or 8) | A, C, D only | Answers the 2 real open questions (aux-loss-free value, QK-norm value) | Original 5-run matrix (+ F: no load balance, G: all stabilizers) | F tests settled science (expert collapse without LB). G tests an interaction (QK-norm + z-loss) that 4/6 frontier models already skip. Both cost $4 for minimal new information. |
| Anchor depth=16 (matching 1B) | 16 layers at all scales | muP transfers across width only. Depth mismatch invalidates transfer. | 8-layer anchor (cheaper, faster) | Cerebras muP guide: "depth mismatch invalidates transfer." 8→16 layer transfer would require a separate depth scaling rule (not established for MoE). |
| Phase 2 (DSA) on 1B only | Skip DSA for anchor and 500M | DSA is a capability feature, not a scaling variable. Anchor/500M validate HPs, not capabilities. | DSA on all 3 scales | Phase 2 adds significant complexity and cost. Running it on anchor would make grid search 4× more expensive for no HP transfer benefit. |
| Random sample 15-20 from grid (not full 81) | Bergstra & Bengio + budget | 95% probability of finding near-optimal in 4D space with 3 levels | Full 81-run grid | $160+ cost exceeds pre-validation budget. Diminishing returns after ~15 trials in low-dimensional grid. |
| Bad batch injection at step 1500 | Mid-training perturbation test | Tests recovery of learned representations, not warmup robustness | Step 500 (tests warmup), Step 2500 (tests late-stage) | Step 500: model hasn't learned enough for recovery to be meaningful. Step 2500: only 500 steps of recovery observation. |

---

# Appendix: Failure Mode Catalog

| Failure | How We'd Detect It | What It Means | What To Do |
|---------|-------------------|---------------|------------|
| All grid search runs diverge | NaN loss in all 15-20 runs | LR ranges are entirely too high for MoE+Muon | Extend grid downward: [0.005, 0.01, 0.02] for matrix_lr |
| Grid search runs have low H_load | H_load < 2 bits in best runs | Expert collapse despite good loss | Router LR (3e-4) may be too low relative to expert LR → experts change faster than routing can follow |
| Coordinate check fails | |bpb_480 - bpb_960| > 0.05 | muP scaling rules wrong for this architecture | Diagnose which parameter group breaks (attention vs expert vs embedding). Add corrections. |
| 500M diverges | NaN loss during training | HP transfer fails | Fall back to 3-5 manual HP configs at 500M scale (~$90-150) |
| 500M converges but BPB is bad | ema_val_bpb >> expected for 500M MoE | One or more scaling rules have wrong exponent | Check each parameter group's effective LR. Likely: expert LR needs √(1/8) sparsity correction. |
| 1B Phase 2 OOM | CUDA OOM at phase transition | 8K context exceeds GPU memory with current batch size | Reduce device_batch_size, increase grad accumulation |
| 1B expert collapse mid-training | H_load drops below 2 bits after step 1000 | Winner-take-all dynamics. Load balancing insufficient. | Check gamma_freeze_ratio, check if bias update is running, compare with anchor H_load trajectory |
| Stability ablation: Run A beats Run C | ema_val_bpb(A) < ema_val_bpb(C) at step 3000 | Aux-loss-free doesn't help at this scale | Check implementation. If correct: use traditional aux loss (simpler). Report as scale-dependent finding. |
