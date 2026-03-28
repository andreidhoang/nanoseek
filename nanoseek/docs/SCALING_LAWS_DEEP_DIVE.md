# Scaling Laws Deep Dive: From Chinchilla to NanoSeek
## Why a 410M MoE Trains Differently Than a 124M Dense GPT
### First-Principles Theory, Intuitive Explanations, and Every Design Decision

---

## Part 1: What Is a Scaling Law?

A scaling law is an empirical equation that predicts how good a model will be before you
train it. You plug in model size and training tokens, and it tells you the final loss.

The most famous one (Hoffmann et al., 2022 — "Chinchilla"):

```
L(N, D) = E + A/N^α + B/D^β

Where:
  L = final validation loss (lower = better)
  N = number of parameters
  D = number of training tokens
  E = irreducible entropy (the floor — even an infinite model can't beat this)
  A, α = how loss improves with more parameters
  B, β = how loss improves with more data
```

### Why This Matters — The $1M Question

Without scaling laws, training a large model is a gamble:

```
"Let's train a 7B model on 200B tokens for $50,000"
→ What if 7B is too big and we should've used 3B on 500B tokens?
→ What if 200B tokens isn't enough and we plateau at 150B?
→ What if the architecture doesn't scale and we wasted everything?
```

With scaling laws, you can answer these BEFORE spending money:

```
1. Train 3 small models ($100 total)
2. Fit the scaling law
3. Predict: "A 7B model on 200B tokens will achieve loss = 1.42"
4. Compare to: "A 3B model on 500B tokens will achieve loss = 1.39"
5. Decision: train 3B on 500B tokens → save $30,000
```

This is why every serious lab (OpenAI, DeepMind, Anthropic, DeepSeek) runs scaling law
experiments first. The small experiments steer the big investments.

---

## Part 2: How Nanochat Does It — The IsoFLOP Sweep

Nanochat is a **dense GPT-2-style transformer**. Every parameter is active on every token.
The architecture has one continuous knob: depth (which controls both depth and width via
`model_dim = depth × 64`).

### The Sweep Design

```
                    4 FLOPs Budgets
                    ─────────────────────────────────────
                    1e18     2.15e18   4.64e18   1e19
Depths:             │        │         │         │
  d=8  (180M)      ●────────●─────────●─────────●
  d=10 (275M)      ●────────●─────────●─────────●
  d=12 (394M)      ●────────●─────────●─────────●  ← reference
  d=14 (541M)      ●────────●─────────●─────────●
  d=16 (716M)      ●────────●─────────●─────────●
  d=18 (924M)      ●────────●─────────●─────────●
  d=20 (1.17B)     ●────────●─────────●─────────●

  28 runs total. Each ● is one training run to completion.
```

For each FLOPs budget, every depth gets the same total compute. A bigger model trains for
fewer steps (same compute / more FLOPs-per-step = fewer steps). A smaller model trains for
more steps.

### What the Sweep Found

For each FLOPs budget, pick the depth with the lowest val_bpb:

```
FLOPs = 1e18:   best depth ≈ d=12  (394M params, ~1.2B tokens, ratio ≈ 3.0)
FLOPs = 2.15e18: best depth ≈ d=14 (541M params, ~1.8B tokens, ratio ≈ 3.2)
FLOPs = 4.64e18: best depth ≈ d=16 (716M params, ~2.6B tokens, ratio ≈ 3.6)
FLOPs = 1e19:   best depth ≈ d=18 (924M params, ~3.9B tokens, ratio ≈ 4.0)
```

The compute-optimal frontier traces a curve: as compute grows, both the optimal model
size and the optimal token count grow, but at different rates.

### The Token:Param Ratio

```
Convention matters!

Chinchilla (Hoffmann 2022): counted ALL parameters
  → Optimal ratio: D ≈ 20 × N
  → "For every parameter, train on 20 tokens"

Kaplan (2020): excluded embeddings
  → Optimal ratio: D ≈ 10 × N
  → Different answer because N is smaller (no embeddings)

Nanochat (empirical): used "transformer matrices + lm_head" (Kaplan-style)
  → Found ratio ≈ 10.5 × N
  → This is their default: --target-param-data-ratio=10.5
```

The ratio depends on what you count as "N". Nanochat uses Kaplan-style because it
produces the most consistent scaling exponents (N ∝ C^0.54, D ∝ C^0.49 — both ≈0.5,
meaning compute splits roughly equally between model size and data).

### Why Nanochat CAN Do This

A dense transformer has a **smooth, continuous** parameter-FLOPs relationship:

```
FLOPs per token ≈ 6 × N_total

Change --depth from 8 to 36:
  d=8:  N=180M,  FLOPs/tok=1.08 GFLOPs
  d=12: N=394M,  FLOPs/tok=2.36 GFLOPs
  d=20: N=1.17B, FLOPs/tok=7.00 GFLOPs
  d=36: N=4.61B, FLOPs/tok=27.7 GFLOPs

Smooth curve. Any size works. No minimum viable scale.
```

You can build a 16M dense transformer (d=4, dim=256) and it's a perfectly valid model —
just small. The attention mechanism, the FFN, the residual connections — they all work
at any scale. There are no architectural components that require a minimum size.

---

## Part 3: Why NanoSeek Can't Do the Same — MoE Has a Floor

NanoSeek is a **Mixture of Experts** transformer. It has the same attention (MLA) and
residual structure, but the FFN is replaced by 64 parallel expert FFNs with a learned
router that picks 8 of 64 for each token.

This creates **hard architectural constraints** that don't exist in dense models:

### Constraint 1: Expert Count Has a Minimum

```
Dense FFN:   Every token → one FFN → output
MoE FFN:     Every token → router picks 8 of 64 experts → 8 FFN outputs → weighted sum

The router must learn MEANINGFUL routing — which expert is best for which tokens.
With too few experts, there's nothing to specialize:
  2 experts:  "expert 0 = everything" and "expert 1 = also everything"
  8 experts:  barely enough for broad categories (code, math, text, ...)
  64 experts: enough for fine-grained specialization (Python, LaTeX, dialogue, ...)

DeepSeek found 64 experts with top-8 selection (κ=12.5%) works.
Reducing to 16 experts saves parameters but routing dynamics change fundamentally.
We keep 64 because our research goal IS routing dynamics.
```

### Constraint 2: Each Expert Has a Minimum Useful Size

An expert is a SwiGLU FFN: `output = W_down(SiLU(W_gate(x)) × W_up(x))`

It has 3 weight matrices: `[inter, hidden]`, `[inter, hidden]`, `[hidden, inter]`.
Parameters per expert = `3 × hidden × inter`.

```
At d=512 (hypothetical small NanoSeek):
  inter = 0.375 × 512 = 192
  Expert params = 3 × 512 × 192 = 295K

  A 295K-parameter FFN has 192 intermediate neurons.
  Each neuron can only represent one simple feature.
  SiLU gating on 192 features = very limited nonlinearity.

  For comparison, GPT-2 Small (124M) has a 3072-neuron FFN.
  Our tiny expert has 16× fewer neurons than GPT-2 Small's single FFN.
  An expert this small can barely learn anything beyond linear transformations.

At d=1280 (our ablation scale):
  inter = 0.375 × 1280 = 480
  Expert params = 3 × 1280 × 480 = 1.84M

  480 intermediate neurons — enough for meaningful feature detection.
  Each expert can develop genuine specialization (code syntax, math notation, etc.)
```

This is quantified by the **Krajewski granularity** G:

```
G = N_active / (top_k × expert_params)

G is "how many expert-sizes fit in the active model."
Krajewski et al. (ICML 2024) found G ∈ [16, 32] is optimal.

At d=512:  G = 96M / (8 × 295K) = 41  — too granular, experts too small
At d=768:  G = 175M / (8 × 663K) = 33  — borderline
At d=1280: G = 410M / (8 × 1.84M) = 28 — sweet spot ✓
At d=2048: G = 1.08B / (8 × 4.72M) = 29 — sweet spot ✓
```

### Constraint 3: MLA Head Dimensions Are Fixed Constants

Multi-Head Latent Attention uses fixed head dimensions from the DeepSeek family:

```
qk_nope_head_dim = 128   (non-positional query/key dimension per head)
qk_rope_head_dim = 64    (RoPE dimension per head)
v_head_dim = 128          (value dimension per head)
```

These are NOT ratios of hidden_size — they're absolute constants. The output projection
`W_O` maps `n_heads × v_head_dim → hidden_size`. For this to be a square (non-lossy)
matrix:

```
n_heads × 128 = hidden_size

Valid configurations:
  d=256:   n_heads=2  → W_O is 256→256   (but 2 heads is too few for attention)
  d=384:   n_heads=3  → W_O is 384→384   (3 heads barely viable)
  d=512:   n_heads=4  → W_O is 512→512   (4 heads = minimum for diverse attention)
  d=768:   n_heads=6  → W_O is 768→768   ← anchor (smallest valid config)
  d=1024:  n_heads=8  → W_O is 1024→1024
  d=1280:  n_heads=10 → W_O is 1280→1280 ← ablation
  d=2048:  n_heads=16 → W_O is 2048→2048 ← 1B
```

Hidden_size must be a multiple of 128. You can't smoothly sweep from 100M to 1B like
nanochat does with `--depth=4` through `--depth=36`.

### Constraint 4: MLA Lora Ranks Create Compression Limits

MLA compresses K/V into a low-rank latent. The compression ratios are:

```
q_lora_rank = 0.215 × hidden_size
kv_lora_rank = 0.070 × hidden_size

At d=512:  kv_lora_rank = 36  → 512→36 = 14× compression
           This is so aggressive that attention quality suffers.
           The latent can't represent enough information.

At d=768:  kv_lora_rank = 54  → 768→54 = 14× compression
           Still very aggressive but borderline workable.

At d=1280: kv_lora_rank = 90  → 1280→90 = 14× compression
           Same ratio but more absolute dimensions = richer latent.

At d=2048: kv_lora_rank = 143 → 2048→143 = 14× compression
           Comfortable. DeepSeek V3 (671B) uses 512/7168 = 14× too.
```

The compression ratio is constant (~14×) across scales, but the absolute dimension
matters. A 36-dimensional latent can represent at most 36 linearly independent
features. A 143-dimensional latent can represent 143 — 4× richer representation.

### Putting It All Together: The NanoSeek Size Floor

```
                    Dense (nanochat)          MoE (NanoSeek)
                    ────────────────          ──────────────
Minimum viable      Any size (even 1M)        ~175M active (d=768)
  params

Why                 All components scale       MLA needs ≥6 heads (d=768)
                    smoothly to zero           Experts need ≥480 inter dim
                                               Router needs ≥16 experts
                                               G must be in [16, 32]

Scaling knob        depth (continuous)         width (discrete: 768, 1024,
                    d=4 to d=36               1280, 1536, 2048, ...)
                    model_dim = d × 64        must be multiple of 128

Smallest useful     d=4: 55M total             d=768: 175M active / 730M total
  for scaling       ~$0.50/run                 ~$10/run
  experiments

Sweep range         ~50× (55M → 2.7B)         ~6× (175M → 1.08B)
  in params                                    Limited by MLA geometry

Cost for a          28 runs × ~$2 = ~$56       5 runs × $10-350 = ~$485
  full sweep
```

---

## Part 4: How NanoSeek Applies Chinchilla Differently

### Nanochat: IsoFLOP → Find Optimal N/D Split → Token Ratio

```
Step 1: Run 28 models at 4 FLOPs budgets × 7 depths
Step 2: For each budget, pick the best depth
Step 3: Plot optimal N vs optimal D → fit power law
Step 4: Find: D_optimal ≈ 10.5 × N_scaling_params
Step 5: Use this ratio for all future models
```

This is the gold standard. It derives the optimal ratio from YOUR architecture on YOUR data.

### NanoSeek: Adopt Chinchilla Ratio → Apply to N_active

We can't afford 28 MoE training runs. Instead:

```
Step 1: Adopt D = 20 × N_active (Chinchilla's original finding)
Step 2: Count N_active carefully (only parameters that see each token):
          - Embeddings: always active
          - MLA projections: always active
          - Shared expert FFN: always active
          - 8 of 64 routed experts: active (not all 64!)
          - Router, norms: always active

Step 3: Compute token budget:
          Ablation: N_active ≈ 410M → D = 20 × 410M = 8.2B tokens
          1B:       N_active ≈ 1.08B → D = 20 × 1.08B = 22B tokens
```

### Why N_active, Not N_total?

This is the critical MoE-specific insight:

```
Dense model:  N_active = N_total = 394M
  Every parameter does work on every token.
  FLOPs = 6 × 394M per token.

MoE model:    N_active = 410M, N_total = 1.95B
  Only 410M parameters do work on each token.
  The other 1.54B parameters sit idle (wrong experts for this token).
  FLOPs = 6 × 410M per token (NOT 6 × 1.95B).

If we used D = 20 × N_total:
  D = 20 × 1.95B = 39B tokens
  FLOPs = 6 × 410M × 39B = 9.6e19
  But the model only needed 6 × 410M × 8.2B = 2.0e19 FLOPs to converge.
  We'd waste 4.8× more compute than necessary!

If we used D = 20 × N_active:
  D = 20 × 410M = 8.2B tokens
  FLOPs = 6 × 410M × 8.2B = 2.0e19
  Compute-optimal. Each active parameter sees enough data to converge.
```

The intuition: each token trains 410M parameters. After 8.2B tokens, each parameter
has seen 8.2B training signals. That's 20 tokens per parameter — exactly Chinchilla
optimal. The inactive 1.54B parameters get trained indirectly (when their expert IS
selected), but they share training signal with the active set.

### Why 20:1 and Not 10.5:1 Like Nanochat?

```
Nanochat found ratio ≈ 10.5 using Kaplan-style param counting (exclude embeddings).
Chinchilla found ratio ≈ 20 using all parameters.

NanoSeek uses Chinchilla's 20:1 because:

1. MoE active params are already "Kaplan-like" — embeddings are a small fraction
   of N_active (32768 × 1280 = 42M out of 410M = 10%). In dense models,
   embeddings can be 30-50% of total params, which is why Kaplan excluded them.

2. DeepSeek V3 used ~14.8T tokens for 37B active params → ratio = 400:1.
   But they were pursuing OVERTRAINED models (more data than Chinchilla-optimal)
   because inference cost, not training cost, dominates at their scale.

3. At our budget ($35/run), 20:1 is the sweet spot:
   - 10:1 would undertrain (4.1B tokens for 410M = not enough data)
   - 20:1 is Chinchilla-optimal (8.2B tokens for 410M)
   - 40:1 would cost 2× ($70/run) for diminishing returns
```

---

## Part 5: The Scaling Law Fitter — What It Can and Can't Tell Us

### With Nanochat's 28 Runs

```
28 data points → fit L(N, D) with 5 parameters → robust extrapolation

Can predict:
  ✓ Optimal model size for any FLOPs budget
  ✓ Optimal token count for any model size
  ✓ Expected loss at 3B, 7B, 13B (extrapolation)
  ✓ Whether architecture is scaling efficiently
  ✓ The irreducible entropy E (data quality signal)

Validation: LOOCV on 28 points → reliable error estimates
```

### With NanoSeek's 2 Runs

```
2 data points → fit L(N) = E + A × N^(-α) with E fixed → exact fit, no validation

The fit:
  Fix E = 1.69 (literature prior for English text)
  Two points: (410M, loss_ablation) and (1.08B, loss_1b)
  Two unknowns: A and α
  → Exactly one solution (2 equations, 2 unknowns)

Can predict:
  ✓ Expected loss at 3B, 7B (extrapolation — but unvalidated)
  ✓ Whether α is in literature range [0.15, 0.7] (sanity check)
  ✗ Cannot validate predictions (no held-out data point)
  ✗ Cannot estimate E (it's fixed, not fitted)
  ✗ Cannot detect architecture-specific scaling bottlenecks

This is a SANITY CHECK, not a discovery tool:
  "Does NanoSeek-MoE follow roughly the same scaling law as dense transformers?"
  If yes → architecture is healthy, proceed to larger scales.
  If no (α outside range) → something is broken (dead experts, routing collapse, etc.)
```

### How We Implemented This

```python
# eval/scaling_law.py — handles 2-point and 3+-point cases

def fit_scaling_law(data_points, E_prior=1.69):
    n_points = len(data_points)

    if n_points == 2:
        # Fix E, fit A and α from 2 points (exactly determined)
        # Validate: is α in [0.15, 0.7]?
        # Report residuals (should be ~0 for exact fit)
        # NO LOOCV possible

    elif n_points == 3:
        # Fix E, fit A and α from 3 points (1 degree of freedom)
        # LOOCV: leave each out, fit on 2, predict held-out
        # Mean LOOCV error < 15% = healthy

    elif n_points >= 4:
        # Full 3-param fit (E, A, α all free)
        # LOOCV on all points
        # Can estimate E from data (compare to literature 1.69)
```

### Adding the Anchor: Cheapest Path to 3-Point Validation

```
If we train the anchor config (d=768, ~175M active, ~$10):
  3 data points: (175M, loss_anchor), (410M, loss_ablation), (1.08B, loss_1b)
  Fix E = 1.69, fit A and α from 3 points (1 DOF)
  LOOCV: leave each out, fit on 2, predict held-out

  This tells us:
  ✓ Does the scaling law HOLD across 6× range? (LOOCV < 15%)
  ✓ Is the architecture efficient at all 3 scales?
  ✓ Reliable extrapolation to 3B, 7B

  Cost: $10 extra (1 anchor run) for validated scaling law.
  This is by far the cheapest improvement to our experimental design.
```

---

## Part 6: Beyond Token Count — The Other Scaling Laws in NanoSeek

Chinchilla tells us HOW MANY tokens. But there are other scaling laws that tell us
HOW to train — learning rate, batch size, weight decay. NanoSeek uses three more.

### 6.1 muP Width Scaling (Tensor Programs V)

**The problem**: HPs tuned at ablation scale (d=1280) don't work at 1B scale (d=2048)
if you just copy them. The gradients have different magnitudes at different widths.

**The insight** (Yang et al., 2022): If you scale learning rate as `η ∝ 1/width`,
then the effective update `||Δh||` (how much hidden states change per step) stays
constant across widths. This is called **maximal update parametrization (muP)**.

```
Without muP:
  Ablation (d=1280): lr=0.02 → ||Δh|| = 0.01 → works great
  1B (d=2048):       lr=0.02 → ||Δh|| = 0.016 → updates too large → unstable!

With muP:
  Ablation (d=1280): lr=0.02 → ||Δh|| = 0.01 → works great
  1B (d=2048):       lr=0.02 × (1280/2048) = 0.0125 → ||Δh|| = 0.01 → same! ✓
```

In our code:

```python
# pre_train.py:508-509
width_lr_scale = w_ref / w   # 1280 / 2048 = 0.625 for 1B
# Hidden weight LRs get multiplied by this factor
```

### 6.2 Batch Size Scaling (Complete(d)P)

**The problem**: if ablation uses batch=524K tokens and 1B uses the same batch, the
gradient noise is the same — but the model is bigger, so updates should be cleaner.

**The insight**: gradient noise scales as `1/√B`. To get equivalent noise:

```
η ∝ √(B/B_ref)

If batch doubles: η scales by √2 ≈ 1.41
  → Bigger batch = cleaner gradient = can take bigger steps
```

In practice, NanoSeek uses the same batch size at both scales (524K tokens),
so this factor is 1.0. But the code handles it correctly for future experiments.

### 6.3 Weight Decay Scaling (T_epoch Framework)

**The problem**: weight decay `λ` regularizes the model by pulling weights toward zero.
If you change batch size or training duration, the TOTAL amount of regularization changes.

**The insight** (Lewkowycz et al., 2024): define `T_epoch = B / (η × λ × D)`.
This is "how many effective epochs of regularization the model sees." Keep T_epoch
constant across scales:

```
λ = λ_ref × √(B/B_ref) × (D_ref/D)

Where D_ref is the training tokens at the reference scale (ablation = 8.2B).

At ablation: λ = λ_ref × 1.0 × 1.0 = λ_ref        (no scaling, this IS reference)
At 1B:       λ = λ_ref × 1.0 × (8.2/22) = 0.37 × λ_ref  (less decay, longer training)
```

Why less decay for longer training? The model sees 22B tokens instead of 8.2B — more
data means less need for regularization. Weight decay's job is to prevent overfitting;
with 2.7× more data, there's less overfitting to prevent.

**Critical bug we found**: the old code used `D_ref = anchor_config.total_tokens = 2.5B`
(anchor was never trained!). This made weight decay 3.3× too small at ablation scale:

```
OLD (wrong): λ = λ_ref × (2.5B / 8.2B) = 0.305 × λ_ref  ← 3× too weak!
NEW (fixed): λ = λ_ref × (8.2B / 8.2B) = 1.0 × λ_ref    ← correct
```

---

## Part 7: The Complete Decision Chain — From Theory to Config

Here's how every number in `config.py` connects to a scaling law or design principle:

```
STEP 1: Budget → N_active
  ──────────────────────────────────────────────────────────────
  Budget constraint: ~$350 for graduation run
  GPU: 8×H100 at ~$2.49/hr → ~14 hours of compute
  Compute: 8 × 989 TFLOPS × 14hr × 3600s × 0.45 MFU ≈ 1.6e20 FLOPs

  Chinchilla: C = 6 × N × D, and D = 20 × N
  So: C = 6 × N × 20N = 120 × N²
  N = √(C / 120) = √(1.6e20 / 120) ≈ 1.15B
  Round to clean architecture: N_active ≈ 1.08B
                                                    ↓
STEP 2: N_active → Token Budget
  ──────────────────────────────────────────────────────────────
  D = 20 × N_active = 20 × 1.08B = 21.6B → config: total_tokens = 22B
  D_ablation = 20 × 410M = 8.2B → config: total_tokens = 8.2B
                                                    ↓
STEP 3: N_active → Architecture (d, L, experts)
  ──────────────────────────────────────────────────────────────
  Depth: L = 16 (Allen-Zhu: depth = reasoning hops; matches OLMoE-1B)
  Width: d/L = 128 → d = 2048 (d must be multiple of 128 for MLA)
  Ablation: same L=16, d=1280 (muP: only width varies between scales)

  MoE: Krajewski G ≈ 29 → expert_params = N_active/(top_k × G)
       = 1.08B / (8 × 29) = 4.66M → moe_inter = 4.66M / (3 × 2048) = 758 → 768

  Expert count: 64 routed, 2 shared (OLMoE precedent, DeepSeek V3 routing)
  κ = 8/64 = 12.5% (OLMoE ratio, not DeepSeek's 3.1%)
                                                    ↓
STEP 4: Architecture → MLA Dimensions
  ──────────────────────────────────────────────────────────────
  Head dims (DeepSeek constants): qk_nope=128, qk_rope=64, v=128
  n_heads = d / v_head_dim = 2048/128 = 16

  Compression (DeepSeek ratios):
    q_lora_rank = 0.215 × 2048 = 440
    kv_lora_rank = 0.070 × 2048 = 143
    MLA compression = d / kv_lora_rank = 2048/143 ≈ 14× (matches V3!)
                                                    ↓
STEP 5: Token Budget → Training Steps
  ──────────────────────────────────────────────────────────────
  Batch = global_batch_size × sequence_length = 128 × 4096 = 524,288 tokens
  Steps = total_tokens / batch = 22B / 524,288 ≈ 41,962
  Steps_ablation = 8.2B / 524,288 ≈ 15,640
                                                    ↓
STEP 6: Training Steps → LR Schedule
  ──────────────────────────────────────────────────────────────
  Warmup: 1000 steps (2.4% of ablation, 6.4% of ablation)
  Constant: until 70% of total steps
  Cosine decay: 70% → 95% of total steps
  lr_min: 10% of peak (remaining 5% of steps)
                                                    ↓
STEP 7: muP → LR Transfer (ablation → 1B)
  ──────────────────────────────────────────────────────────────
  Hidden weights:  η_1b = η_abl × (1280/2048) = 0.625 × η_abl
  Embed weights:   η_1b = η_abl × 1.0 (no width scaling for input/output)
  Router weights:  η_1b = η_abl × 1.0 (constant — output layer in muP)
  Norms/scalars:   η_1b = η_abl × 1.0 (scale-independent)
  Weight decay:    λ_1b = λ_abl × (8.2B/22B) = 0.373 × λ_abl
                                                    ↓
STEP 8: FLOPs → MFU Target
  ──────────────────────────────────────────────────────────────
  FLOPs per token = 6 × N_active = 6 × 1.08B = 6.48 GFLOPs
  Total FLOPs = 6.48G × 22B = 1.43e20
  Peak hardware: 8 × 989 TFLOPS = 7.91 PFLOPS
  MFU target = actual / peak ≈ 45% (typical for MoE on H100)
  Wall time = 1.43e20 / (7.91e15 × 0.45) ≈ 40,000s ≈ 11 hours
```

---

## Part 8: The Scaling Law Fitter — Usage After Training

After training completes at both scales:

```bash
# Fit scaling law from ablation + 1B results
python -m nanoseek.eval.scaling_law \
    --runs ablation:410e6:<ema_val_bpb> 1b:1.08e9:<ema_val_bpb> \
    --predict 3e9 7e9

# Expected output:
# L(N) = 1.6900 + A * N^(-α)
# α should be in [0.15, 0.7] (literature range for well-behaved architectures)
# Predictions for 3B and 7B extrapolated from the fitted law
```

### What Each Outcome Means

```
α ∈ [0.2, 0.5]:  Normal scaling. MoE architecture is healthy.
                   Safe to extrapolate to 3B, 7B.

α < 0.15:         Weak scaling. Loss barely improves with more params.
                   Possible causes:
                   - Expert routing collapse (check H_load)
                   - Dead experts (check dead expert count)
                   - MLA compression too aggressive
                   - Data quality limiting (check domain BPB)

α > 0.7:          Unusually strong scaling. Suspicious.
                   Possible causes:
                   - Ablation was undertrained (loss too high)
                   - 1B had a lucky seed / data ordering
                   - Measurement error in ema_val_bpb

R² ≈ 1.0:         Expected with 2 points (exactly determined).
                   Not informative. Add anchor ($10) for 3-point LOOCV.

LOOCV < 15%:      (With 3+ points) Scaling law extrapolates reliably.
                   Predictions at 3B, 7B are trustworthy.

LOOCV > 15%:      Scaling law doesn't fit well.
                   Architecture may have a scale-dependent bottleneck.
                   Investigate per-scale training dynamics before scaling up.
```

---

## Part 9: Nanochat vs NanoSeek — Side-by-Side Summary

```
                        Nanochat (Dense GPT)         NanoSeek (MoE)
                        ─────────────────────        ──────────────────
Architecture            Dense transformer             MLA + MoE + MTP
Param/FLOP relation     N_total = N_active            N_total ≈ 4.4× N_active
Scaling knob            depth (continuous)             width (discrete, ×128)
Minimum viable scale    ~16M (d=4)                    ~175M active (d=768)
Scaling sweep cost      ~$56 (28 runs)                ~$485 (5 runs)
Token:param ratio       10.5 (empirically derived)    20 (Chinchilla, applied to N_active)
HP transfer method      muP across depths             muP across widths (same depth)
Training scales         7 depths × 4 budgets          2 scales (ablation + 1B)
Scaling law quality     28-point fit, validated        2-point fit, sanity check
Reference model         d=12 (394M total)             ablation (1280h, 410M active)
Research output         Optimal N/D split, ratio       MoE dynamics, expert specialization
```

### Why These Are Different Projects, Not Competitors

Nanochat asks: **"What is the compute-optimal way to train a dense GPT?"**
→ Answer: sweep, measure, fit, extrapolate.

NanoSeek asks: **"How do MoE routing dynamics work during training?"**
→ Answer: instrument everything, train at the smallest MoE scale where dynamics
are meaningful, study expert specialization and collapse.

The scaling law is a TOOL in both projects, but it serves different purposes:
- Nanochat: the scaling law IS the research output
- NanoSeek: the scaling law is a sanity check that the architecture is healthy

---

## Appendix A: The 6× FLOPs Rule — Where "6" Comes From

```
A Linear layer: output = input @ weight.T

FORWARD (1 GEMM):
  For each output element: multiply K values and sum them.
  That's K multiplies + K additions = 2K FLOPs.
  Output has M × N elements → 2MNK FLOPs.
  Per parameter (MK weights): 2MNK / (M×K) = 2N FLOPs per param.
  Per token (N=1): 2 FLOPs per parameter.

BACKWARD (2 GEMMs):
  grad_input = grad_output @ weight     → same shape as forward → 2 FLOPs/param
  grad_weight = grad_output.T @ input   → same size matmul → 2 FLOPs/param

TOTAL: 2 + 2 + 2 = 6 FLOPs per parameter per token.

For NanoSeek:
  FLOPs per token = 6 × N_active = 6 × 410M = 2.46 GFLOPs
  Total training FLOPs = 2.46G × 8.2B = 2.02 × 10^19
```

## Appendix B: Why Embeddings Are Special

```
Embedding:  lookup (not a matmul) → no "6× per param" rule
  Forward: index into [vocab, d] table → 0 FLOPs (memory access only)
  Backward: scatter-add gradient to the looked-up row → minimal FLOPs

LM Head:    IS a matmul: hidden @ embedding.T → [batch, vocab]
  Forward: 2 × batch × vocab × d FLOPs
  But this is a TINY fraction of total (one layer vs 16 layers × MoE)

This is why:
  - Kaplan excluded embeddings from N (they don't contribute proportional FLOPs)
  - Chinchilla included everything (simpler, works for total compute accounting)
  - Nanochat uses "transformer matrices + lm_head" (Kaplan-ish, cleanest fit)
  - NanoSeek uses N_active (includes embeddings because they're small: 42M/410M = 10%)
```

## Appendix C: How to Add More Scales (Future Work)

```
If budget allows, add these training runs for better scaling law:

  $10:  Anchor (d=768, 175M active, 2.5B tokens)
        → 3-point fit with LOOCV validation

  $20:  d=1024 (280M active, 5.6B tokens)
        → 4-point fit, full 3-param fit possible

  $70:  d=1536 (600M active, 12B tokens)
        → 5-point fit, robust extrapolation to 3B+

Total with all 5 scales: ~$485
Compare: nanochat's 28-run sweep: ~$56

The 10× cost difference is the price of MoE:
  - Each expert adds parameters but not FLOPs
  - N_total/N_active ≈ 4.4× → 4.4× more memory per GPU
  - Expert routing overhead (sort, scatter, gather) adds wall time
  - Larger batch needed for stable routing statistics

This is why we do 2 scales (not 28) and rely on Chinchilla's 20:1 ratio
rather than deriving our own. The ratio has been validated across
architectures (GPT, LLaMA, PaLM, Chinchilla, T5) — it transfers.
```
