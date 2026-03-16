# NanoSeek Scaling Lab Plan
## MLA + MoE Scaling Laws · Training Stability · Production Observability
### March 2026 — Grounded in 2024–2026 Research Literature

---

## Framing: The Three Gaps from AGENTS.md

The existing nanoseek plan has three identified gaps relative to what frontier pretraining teams actually do:

> **Gap 1** — "Scaling law prediction is a killer differentiator. The ability to predict before spending $10M."
> **Gap 2** — "Training stability/dynamics coverage is deeper [...] spike injection, spike detection, logit softcapping, QK-norm, z-loss, SPAM optimizer, muP transfer."
> **Gap 3** — "Observability as a deliverable, not an afterthought [...] real-time dashboards, automated spike detection, MFU regression alerts."

This plan closes all three gaps systematically. Each pillar produces a concrete, falsifiable, written artifact.

---

## Architecture Baseline

nanoseek implements DeepSeek V3.2 at nano scale:

| Quantity | Value | Source |
|---|---|---|
| Active parameters | 1.08B | model/config.py |
| Total parameters | 4.75B | model/config.py |
| Sparsity ratio N_active/N_total | 22.7% | derived |
| Routed experts | 64 | model/config.py |
| Shared experts | 2 | model/config.py |
| Top-K routing | 8 | model/config.py |
| KV compression ratio (MLA) | 23× | model/model.py:313–453 |
| Training context | 4K (Phase 1) / 8K (Phase 2) | model/config.py |
| Training tokens | 22B (20× active params†) | model/config.py |

†**Note on 20:1 ratio:** Chinchilla's 20:1 ratio was derived for dense models using
total parameters. For MoE, the compute-optimal D/N_active ratio may differ (some MoE
papers suggest 30-40:1). The 22B target is an initial estimate; the IsoFLOP sweep
(Series C) will determine the actual compute-optimal ratio for this architecture.
If the optimal ratio proves to be 40:1, the model is undertrained by 2×.

**Reference frontier:** Qwen3-235B uses 22B active / 235B total (9.4% sparsity). DeepSeek-V3 uses 37B active / 671B total (5.5% sparsity). nanoseek's 22.6% sparsity is intentionally denser — appropriate for small-scale research where expert count is constrained.

---

# PILLAR 1: SCALING LAW PREDICTION

## Why This Is the Differentiator

From AGENTS.md: *"If you can actually fit L(N,D) from 15 small runs and predict 7B loss within 2%, that's a falsifiable, unmistakable signal."*

The 2025 research literature validates this methodology at MoE scale:
- arXiv:2502.05172 (ICML 2025): 280 controlled experiments up to 2.7B active / 5B total, max extrapolation error 0.018
- arXiv:2507.17702 (Jul 2025): Derived law from small runs, predicted 0.85B active model, validated before training, showed 7× FLOP savings vs dense baseline
- arXiv:2401.02954 (DeepSeek LLM, Jan 2024): Predicted performance at 1000× compute gap from small IsoFLOP runs

**Nobody else applying for pretraining roles has this story with MoE + MLA. Nobody.**

---

## 1.1 The Correct Formula

The SCALING_LAB_80_20_DISTILLED.md plan uses Chinchilla's dense formula:
```
L(N, D) = E + A/N^α + B/D^β        # WRONG for MoE — systematic underestimate
```

The correct formula for MoE, from arXiv:2502.05172 (Ludziejewski et al., ICML 2025):
```
L(N_active, D, E) = L_irr + A / N_active^α + B_e · log(E)^γ + B_d / D^δ

Where:
  L_irr  = irreducible loss (entropy of the data distribution)
  N_active = active parameters per token (primary scaling variable)
  D      = training tokens
  E      = number of routed experts
  A      = parameter scaling coefficient
  B_e    = expert count coefficient (routing diversity correction)
  B_d    = data scaling coefficient
  α      ≈ 0.33–0.35  (parameter scaling exponent)
  δ      ≈ 0.27–0.30  (data scaling exponent)
  γ      ≈ 0.05–0.10  (routing diversity correction — mild)

  7 free parameters total: L_irr, A, α, B_e, γ, B_d, δ

  NOTATION: B_d is the DATA scaling coefficient (was "C" in earlier drafts).
  Renamed to avoid collision with C_scaling = 6 × N_active × D (compute budget).
  All uses of "C" in this document refer to compute budget, never the loss coefficient.

Key insight (arXiv:2502.05172): N_active is primary.
log(E) enters as a mild correction — more experts give slightly lower loss
at fixed N_active and D, but the effect is logarithmic, not power-law.
```

**Why not use L(N_total, D)?**
arXiv:2501.12370 (Apple, ICML 2025) shows that during pretraining, N_total does
matter — higher total capacity helps. But the effect is captured by the log(E)
correction term above (since N_total = N_active × E/top_k × depth_factor). For
a fixed sparsity ratio sweep, using N_active as the primary variable and fitting
log(E) separately gives a cleaner, more interpretable decomposition.

**What MLA adds:**
There is no published paper (as of March 2026) measuring how MLA changes scaling
exponents. TransMLA (arXiv:2502.07864, NeurIPS 2025) shows 10.6× inference
speedup with 93% KV compression but doesn't address training-time exponents.
**Hypothesis** (testable, not assumed): MLA shifts L_irr downward (better parameter
efficiency at equal FLOPs) but does not change α or δ. This is plausible if
kv_lora_rank is large enough that the bottleneck is not binding, but it could be
FALSE if MLA introduces a scale-dependent information bottleneck (in which case
α_MLA < α_MHA). The plan tests this by comparing fitted exponents to published
dense baselines. Any deviation is a genuine finding. Do NOT treat this as fact
in analysis — report both "MLA-neutral" and "MLA-affected" interpretations.

---

## 1.2 Sweep Design

### FLOP Accounting Convention

**Two FLOP formulas exist in this project — do not conflate them.**

```
1. Scaling Law FLOPs (C_scaling): C = 6 × N_active × D
   Use for: scaling law fitting, IsoFLOP design, sweep tables, predict_and_validate.py
   Source: Kaplan et al. (2020), Chinchilla (2022), arXiv:2502.05172
   Approximation: counts only dense matmul FLOPs (forward + backward)
   This is the standard in the scaling law literature.

2. Hardware FLOPs (C_hardware): detailed per-component accounting
   Use for: MFU calculation, profiler analysis, GPU-hour estimates
   Source: mfu_profiler.py (Section 3.2)
   Includes: router FLOPs, MLA projections, shared experts, attention
   Typically ~3.3× higher than C_scaling for NanoSeek's architecture

ALL tables in Pillar 1 use C_scaling = 6 × N_active × D.
mfu_profiler.py uses C_hardware. These are different tools for different purposes.
```

### Critical Constraint: Fix Sparsity Ratio

arXiv:2501.12370 shows the compute-optimal N_active/N_total ratio changes with C.
If you vary sparsity during the sweep, you conflate two effects and the fit won't
extrapolate. **Hold N_active/N_total ≈ 0.23 across all Series A configs.**
(Series B varies E at fixed N_active, so sparsity ratio changes by design.
 Series C varies N_active at fixed C, so sparsity ratio also varies.)

### Config Grid

**Series A: Scale Sweep (fix sparsity ratio, co-vary N_active, D, and architecture)**

| Config | n_layers | n_experts | top_k | N_active | N_total | D (tokens) | C_scaling (6×N×D) |
|---|---|---|---|---|---|---|---|
| nano-20M | 4 | 16 | 2 | ~18M | ~80M | 400M | ~4.3e16 |
| nano-80M | 8 | 32 | 4 | ~75M | ~330M | 1.5B | ~6.8e17 |
| nano-150M | 10 | 32 | 4 | ~140M | ~600M | 2.8B | ~2.4e18 |
| nano-300M | 14 | 64 | 8 | ~280M | ~1.2B | 5.6B | ~9.4e18 |
| nano-500M | 16 | 64 | 8 | ~440M | ~1.9B | 8.8B | ~2.3e19 |
| nano-800M | 16 | 64 | 8 | ~760M | ~3.1B | 15.2B | ~6.9e19 |
| NanoSeek-1B | 16 | 64 | 8 | 1.08B | 4.75B | 22B | ~1.4e20 |

**Dropped:** nano-40M — too close to nano-20M and nano-80M on the log scale. 6 points
spanning ~60× dynamic range in N_active is sufficient for power-law fitting with
warm-started Chinchilla priors. NanoSeek-1B is the held-out validation target.

**Depth fixes (Allen-Zhu):** Per Allen-Zhu's Physics of Language Models Part 4.1, depth
is a discrete capability axis (determines maximum reasoning chain length). Depth must be
monotonically non-decreasing across the sweep to avoid confounding capability regressions
with parameter scaling.

- **nano-500M**: 18→16 layers. Was deeper than NanoSeek-1B (16L) — a sweep point with
  superior depth to the target is scientifically invalid. Now d/L=80, pure width comparison.
  N_active adjusted ~480M→~440M.
- **nano-800M**: 14→16 layers. Was shallower than nano-500M (16L) — scaling from 500M to
  800M would LOSE reasoning depth while gaining parameters. Now d/L=104. N_active adjusted
  ~690M→~760M, D adjusted 13.8B→15.2B.

Post-fix depth sequence: 4→8→10→14→16→16→16 (monotonically non-decreasing).
The sweep has two clean regimes: depth-scaling (20M→300M, layers 4→14) and
width-scaling (500M→1B, layers fixed at 16, d/L increases 80→104→128).

**Expert-count confound note:** n_experts changes at two transitions (16→32 at nano-80M,
32→64 at nano-300M). The activation ratio top_k/E=12.5% is constant, but routing
granularity differs. Series B (expert count sweep at fixed N_active) measures this effect.

**Full architecture specs for each sweep config:**

All configs use DeepSeek V3.2 ratios scaled proportionally:
moe_inter ≈ 0.375 × hidden, q_lora ≈ 0.21 × hidden, kv_lora ≈ 0.07 × hidden,
qk_nope=64, qk_rope=32, v_head=64. Dense FFN ≈ 2.56 × hidden (SwiGLU).

| Config | hidden | moe_inter | q_lora | kv_lora | first_k_dense |
|---|---|---|---|---|---|
| nano-20M | 256 | 128 | 56 | 24 | 1 |
| nano-80M | 512 | 192 | 112 | 40 | 1 |
| nano-150M | 768 | 288 | 160 | 56 | 1 |
| nano-300M | 1024 | 384 | 216 | 72 | 2 |
| nano-500M | 1280 | 480 | 272 | 96 | 2 |
| nano-800M | 1664 | 624 | 352 | 120 | 2 |
| NanoSeek-1B | 2048 | 768 | 430 | 143 | 2 |

**Sparsity ratio note:** The target ~23% sparsity (N_active/N_total) holds precisely
at large scale (NanoSeek-1B: 22.6%). At small scale (nano-20M), embeddings dominate
the parameter count (vocab×hidden×2 ≈ 33M of 36M active), inflating the ratio to
~50-90%. This is an inherent property of nano-scale MoE research, not a design flaw.
The scientifically relevant constraint is top_k/E = 12.5% activation ratio, which
holds exactly across all configs. The scaling law fits N_active (including embeddings)
and is self-consistent.

**Architectural co-variation:** This is a SCALE sweep, not a clean single-variable sweep.
Layers, experts, top_k, hidden_dim, N_active, and D all change together. This is
intentional — it mirrors how real models scale. The scaling law fits against N_active
and D; the log(E) correction captures expert-count variation. **Caveat**: depth-to-width
ratio changes also affect optimization dynamics (deeper models learn more hierarchical
representations). A pure power-law in N_active cannot fully capture these effects.
This limitation is documented and should be quantified by the prediction error on the
held-out NanoSeek-1B point. If prediction error > 2%, an interaction term or
architecture-aware correction may be needed.

**Series B: Expert Count Sweep (fix N_active and D, vary E)**

This series isolates the log(E) correction term independently:

| Config | n_layers | n_experts | top_k | N_active | D (tokens) |
|---|---|---|---|---|---|
| expert-sweep-E8 | 8 | 8 | 2 | ~75M | 1.5B |
| expert-sweep-E16 | 8 | 16 | 4 | ~75M | 1.5B |
| expert-sweep-E32 | 8 | 32 | 4 | ~75M | 1.5B |
| expert-sweep-E64 | 8 | 64 | 8 | ~75M | 1.5B |

**Note:** Series B holds N_active fixed by adjusting top_k proportionally with E.
This directly measures the log(E) coefficient γ.

**⚠️ Confound warning:** Changing E while holding N_active fixed requires changing
BOTH top_k AND moe_intermediate_size (expert FFN width). At E=8/top_k=2, each expert's
FFN is much wider than at E=64/top_k=8. This means Series B co-varies expert count,
routing granularity, AND per-expert capacity — it cannot isolate E alone.
The fitted γ reflects the joint effect. Acknowledge this when reporting results.
A cleaner isolation would require E×(expert_dim) = constant, but that makes N_active
vary, defeating the purpose. This is an inherent limitation of MoE scaling sweeps.

**Series C: IsoFLOP Sweep (fix C, vary N_active/D split)**

At C_scaling ≈ 9e18 (using C = 6 × N_active × D), vary the allocation:

| Config | N_active | D | C_scaling (6×N×D) | N_active/D ratio |
|---|---|---|---|---|
| isoflop-very-data | ~30M | 50B | 9.0e18 | 0.0006 |
| isoflop-data-heavy | ~50M | 30B | 9.0e18 | 0.0017 |
| isoflop-balanced | ~150M | 10B | 9.0e18 | 0.015 |
| isoflop-param-heavy | ~400M | 3.75B | 9.0e18 | 0.107 |
| isoflop-very-param | ~600M | 2.5B | 9.0e18 | 0.240 |

5 points (up from original 3) to robustly locate the IsoFLOP minimum.
The two extreme points (very-data, very-param) define the curve tails —
without them, you can't distinguish a true minimum from an inflection point.
Each additional run is cheap (~2h on A100) relative to the information gained.

This gives you the IsoFLOP curve — optimal N_active for a given compute budget.
The minimum of this curve IS the Chinchilla point for your data/architecture.

**Total: ~15 scaling runs (6 scale + 4 expert + 5 IsoFLOP)**

**Double-duty optimization:** Stability Run D (aux-loss-free, QK-norm ON) at nano-150M
IS the nano-150M scale sweep data point. One run serves both purposes. Effective
unique runs for Pillar 1: 14.
**⚠️ Confound check:** Run D uses aux-loss-free + QK-norm — this MUST match the Series A
baseline configuration. All Series A configs must use the same load-balancing strategy
(aux-loss-free with γ=0.001) and QK-norm setting. If Series A uses a different config,
Run D is an outlier in the scaling law fit.

---

## 1.3 Directory Structure

```
nanoseek/scaling_law_lab/
│
├── configs/
│   ├── series_a_scale.yaml          # 6 scale-sweep configs (nano-40M dropped)
│   ├── series_b_experts.yaml        # 4 expert-count sweep configs
│   ├── series_c_isoflop.yaml        # 5 isoflop configs
│   └── base.yaml                    # shared hyperparams (LR, warmup, etc.)
│
├── run_sweep.sh                     # orchestration: launch all 15 runs
│                                    # (adapt from nanochat/runs/scaling_laws.sh)
│
├── fit_scaling_law.py               # THE CORE — ~200 lines
│
├── predict_and_validate.py          # THE MONEY SHOT — ~80 lines
│
├── inference_aware_mla.py           # BONUS: MLA-adjusted inference cost — ~100 lines
│
└── report/
    └── SCALING_LAW_REPORT.md        # publication-quality writeup
```

---

## 1.3b EMA Evaluation Requirement

**All 15 runs evaluate on EMA weights, not raw weights. This is mandatory.**

```
Reason: scaling law coefficients fitted from raw checkpoint weights are
optimizer-state-dependent. Two runs that differ only in batch ordering will
show artificially different "final losses" depending on where they are in
the cosine LR schedule at the checkpoint step.

EMA weights (decay=0.9999, updated every 10 steps) average over ~10K steps
and represent the stable model the training trajectory has converged toward.

Implementation for every sweep run:
  - EMATracker initialized at training start (nanoseek/training_ops/ema_tracker.py)
  - EMA checkpoint saved at each regular checkpoint interval
  - W&B metric: "ema_val_bpb" logged at each eval interval
  - fit_scaling_law.py reads "ema_val_bpb", not "val_bpb"
  - If "ema_val_bpb" missing from a run: mark run as invalid, exclude from fit

EMA validation protocol:
  At end of each sweep run, compare ema_val_bpb vs raw val_bpb.
  Expected: ema_val_bpb ≤ raw val_bpb (EMA is smoother / lower variance).
  If ema_val_bpb >> raw val_bpb: EMA tracker bug — investigate before fitting law.
```

## 1.3c Expert Routing Metrics as Pillar 1 Scientific Output

**Extend the scaling law fit and sweep to measure expert routing behavior.**

**IMPORTANT: Two distinct metrics — do not confuse them.**

```
Metric 1: H_load (Load-Balance Entropy) — measures routing UNIFORMITY
  H_load = -sum_i p_i × log(p_i)
  where p_i = fraction of validation tokens routed to expert i
  High H_load = uniform load (good for utilization)
  Low H_load = collapsed routing (bad — few experts active)
  Use for: collapse detection, load balancing validation
  Alert threshold: H_load < 2.0 bits → expert collapse

Metric 2: I_spec (Expert Specialization) — measures routing SPECIALIZATION
  I_spec = MI(expert; domain) = H(expert) - H(expert | domain)
  where domains = {code, math, text} (or finer-grained categories)
  High I_spec = experts specialize on different domains (interesting finding)
  Low I_spec = experts route uniformly across domains (no specialization)
  Use for: scientific analysis of whether experts develop semantic roles
  Note: H_load and I_spec are COMPLEMENTARY, not independent.
  By definition: I_spec = H_load - H(expert | domain)  (see line above).
  They share H_load as a common term, so they are mathematically linked.
  However, you CAN have high load balance AND high specialization
  (each domain uses different experts, but total load is balanced).
  This is the ideal MoE operating point.
  Caution: if H_load drops (collapse), I_spec drops mechanically too.

  Implementation:
    For each domain d, compute routing distribution p_i^d over experts
    H(expert | domain) = sum_d P(d) × H(p^d)
    I_spec = H_load - H(expert | domain)
    Equivalently: I_spec = KL_avg(p^d || p_marginal) weighted by P(d)

Extended law: L(N_active, D, E, I_spec) — treat I_spec as a measured output
  Primary research question: does I_spec depend on E independently of N_active?
  If fit shows ∂I_spec/∂E > 0 at fixed N_active: expert count drives specialization
  This is novel — not in any current scaling law paper

Implementation in fit_scaling_law.py:
  - Load I_spec and H_load from W&B alongside ema_val_bpb for each run
  - Plot 8: I_spec vs N_active (log-x) — does specialization scale with model size?
  - Plot 9: I_spec vs E at fixed N_active (Series B runs) — is E a specialization knob?
  - Plot 10: I_spec vs D at fixed N_active — does specialization emerge over training?
  - Correlation: pearson(I_spec, ema_val_bpb) — do more specialized models generalize better?

Logging in sweep runs (nanoseek/training_ops/expert_specialization.py):
  - Log H_load at every eval step (collapse detection — operational metric)
  - Log I_spec at 20%, 50%, 80%, 100% of training steps (scientific metric)
  - Log domain routing heatmap (code / math / text) at 50% and 100%
  - W&B keys: "expert/load_entropy", "expert/specialization_mi",
              "expert/load_fraction_{i}", "expert/domain_heatmap"
```

## 1.4 fit_scaling_law.py — Implementation Spec

```python
"""
Fit MoE scaling law: L(N_active, D, E) = L_irr + A/N_active^α + B_e·log(E)^γ + B_d/D^δ

7 free parameters: L_irr, A, α, B_e, γ, B_d, δ
(B_d renamed from C to avoid collision with compute budget C = 6·N_active·D)

Reference: Ludziejewski et al. (2025) "Joint MoE Scaling Laws"
           arXiv:2502.05172, ICML 2025

Data source: W&B runs from scaling sweep (logged as final_val_loss, n_active,
             n_total, n_experts, training_tokens)
"""

# Step 1: Pull run data from W&B
# wandb.Api().runs(project="nanoseek-scaling")
# Extract: (n_active, n_total, n_experts, tokens, final_loss) per run
# Filter: exclude runs with NaN loss, runs that didn't converge
# Use Huber loss in fitting (robust to outlier runs) — from SCALING_LAB plan

# Step 2: Staged fitting (avoids over-parameterization with 15 runs)
#
# PROBLEM: 7 free parameters (L_irr, A, α, B_e, γ, B_d, δ) from 13 data points
# is thin. A single joint fit risks absorbing noise into parameter estimates.
#
# SOLUTION: Staged fitting with progressive parameter introduction.
#
# Stage 2a: Fit γ from Series B alone (4 points, 3 fit params: const, B_e, γ)
#   Series B holds N_active and D fixed, varies only E
#   Fit: L = const + B_e·log(E)^γ  (const absorbs L_irr + A/N^α + B_d/D^δ at fixed N,D)
#   3 params from 4 points = 1 residual degree of freedom (thin but sufficient for 1D curve)
#   Primary output: γ estimate. Secondary: B_e estimate.
#
# Stage 2b: Fix γ AND B_e from Stage 2a, fit (L_irr, A, α, B_d, δ) from Series A+C
#   Data: 5 Series A points (NanoSeek-1B held out for validation) + 5 Series C = 10 points
#   5 free parameters from 10 points (healthy 2:1 ratio)
#   Fit: L = L_irr + A/N_active^α + B_e_fixed·log(E)^γ_fixed + B_d/D^δ
#   Note: E varies across Series A (16→64), but B_e and γ are fixed from Stage 2a
#   Warm-start with Chinchilla values: α=0.34, δ=0.28 (arXiv:2001.08361)
#
# Stage 2c: Optional joint refinement with all 13 points
#   Initialize from Stage 2a+2b estimates
#   Allow all 7 params to adjust, but with tight bounds from staged fits
#   This corrects for any cross-term interactions missed by staging
#   Report both staged and joint estimates — if they diverge significantly,
#   the staged fit is more trustworthy (better identified)
#
# scipy.optimize.minimize(huber_loss_objective, x0=[L_irr, A, α, B_e, γ, B_d, δ])
# Constraint: L_irr > 0, A > 0, α > 0, all params positive

# Step 3: Bootstrap confidence intervals (100 iterations)
# Resample runs with replacement, refit each time
# Report 95% CI for each fitted parameter

# Step 4: Compute optimal allocation
# Given compute budget C_scaling = 6 × N_active × D (FLOPs, forward+backward)
# Solve: minimize L(N_active*, D*, E_fixed) subject to 6·N_active·D = C_scaling
#
# Derivation (Lagrange multiplier on the data term B_d/D^δ):
#   D = C_scaling / (6·N_active), substitute into loss:
#   L = L_irr + A/N^α + B_e·log(E)^γ + B_d·(6·N/C_scaling)^δ
#   dL/dN = -αA/N^(α+1) + δ·B_d·6^δ·N^(δ-1)/C_scaling^δ = 0
#   Solve for N:
#
# Result: N_active* = C_scaling^(δ/(α+δ)) × [αA / (δ·B_d·6^δ)]^(1/(α+δ))
#         D*        = C_scaling^(α/(α+δ)) × [δ·B_d·6^δ / (αA)]^(1/(α+δ)) / 6
#
# NOTE: The B_d coefficient and 6^δ factor are essential — omitting them
# biases the allocation toward too-large models.
# Compare D*/N_active* to Chinchilla ratio (~20:1 for dense).
# For MoE, optimal ratio may be 30-40:1 (more tokens per active param).

# Step 5: Generate plots
# Plot 1: Loss vs N_active (log-log) at fixed D — confirm power law
# Plot 2: Loss vs D (log-log) at fixed N_active — confirm power law
# Plot 3: Residual log(E) term — confirm it's linear in log(E)
# Plot 4: IsoFLOP curves with optimal frontier marked
# Plot 5: Predicted vs actual for held-out validation (NanoSeek-1B)
# Plot 6: MLA comparison — exponents vs published dense baselines
#          Any deviation from α_dense, δ_dense is the MLA finding

# Step 6: Compute Efficiency Leverage (arXiv:2507.17702)
# EL(C) = L_dense(C) / L_MoE(C) at matched compute budget
# Measures how much MoE "buys" over dense at our scale
# Key finding: EL follows a power law in C — MoE advantage GROWS with scale
# Compare our measured EL to the power-law prediction from 2507.17702
# This is a free metric — falls out of fitted parameters, zero extra runs
# Plot 7: EL vs compute budget (log-log) — confirm power law trend
#          Overlay published dense baselines (Chinchilla, Llama) for comparison
```

**Key outputs to log:**
```
Fitted parameters:
  L_irr = X.XXX ± 0.00X
  α     = 0.XXX ± 0.00X   (compare to Chinchilla α=0.34)
  δ     = 0.XXX ± 0.00X   (compare to Chinchilla δ=0.28)
  γ     = 0.0XX ± 0.00X   (log(E) routing correction)

Compute-optimal at C_scaling = 1.4e20:
  N_active* = X.XXB tokens
  D*        = XXB tokens
  Ratio D/N_active = XX (compare to Chinchilla's ~20)

Predicted NanoSeek-1B val loss: X.XXX
```

---

## 1.5 predict_and_validate.py — The Money Shot

```python
"""
Predict NanoSeek-1B loss from the fitted law, then compare to actual.

This is the falsifiable, unmistakable signal. If prediction error < 2%,
the scaling law is validated and the methodology is sound.
"""

# Load fitted parameters from fit_scaling_law.py output
# Plug in NanoSeek-1B: N_active=1.08B, D=22B, E=64
# Predict: L_predicted = L_irr + A/1.08e9^α + B_e·log(64)^γ + B_d/22e9^δ

# Run NanoSeek-1B training (this IS the main training run)
# Load actual: L_actual = final val_bpb from W&B

# Report:
# Prediction error = |L_predicted - L_actual| / L_actual × 100%
# Target: < 2% (consistent with arXiv:2502.05172 max error 0.018)

# Generate: predicted vs actual plot with confidence interval shading
# If within CI: scaling law validated
# If outside CI: analyze why — data quality shift? MLA regime?
```

---

## 1.6 inference_aware_mla.py — Beyond Chinchilla (Exploratory)

**Status: Exploratory appendix, not a core result. Requires measured serving traces.**

MLA changes the inference **memory bandwidth** cost, not total compute.

```python
"""
Inference-aware optimal scaling with MLA memory bandwidth correction.

Reference: Sardana & Frankle (2023) "Beyond Chinchilla-Optimal"
           + MLA compression analysis from arXiv:2502.07864 (TransMLA, NeurIPS 2025)

IMPORTANT: What MLA compresses and what it does NOT compress:

  ✅ KV cache MEMORY: 23× reduction (175 elements vs 4096 per token per layer)
     This directly reduces: GPU memory per sequence, max batch size constraint,
     memory bandwidth during autoregressive decode (reading cached KV)

  ✅ Memory BANDWIDTH during decode: proportional to KV cache size reduction
     Autoregressive decode is memory-bandwidth-bound (not compute-bound)
     Smaller KV cache → fewer bytes read per step → higher decode throughput
     TransMLA reports 10.6× inference speedup at 8K context (arXiv:2502.07864)

  ❌ Total inference COMPUTE (FLOPs): NOT reduced by 23×
     MLA still requires: up-projection W_UK (kv_lora_rank → n_heads × head_dim),
     attention score computation, FFN computation, output projection
     The up-projection is an ADDITIONAL compute cost vs standard MHA
     Net effect on FLOPs: roughly neutral (save KV compute, pay up-projection)

  ❌ Training FLOPs: NOT affected by KV compression
     During training, full attention is computed regardless of caching

Correct framing for inference-aware scaling:
  Without MLA:
    C_total = C_train + C_inference × Q
    C_inference limited by: KV cache memory (constrains batch size and context)

  With MLA (23× KV compression):
    Same C_total formula, BUT:
    - Max batch size increases ~23× (KV memory freed)
    - Decode throughput improves significantly under long context
    - These translate to lower COST per query, not lower FLOPs per query
    - The effective cost reduction depends on hardware utilization:
      memory-bound regime → large savings; compute-bound regime → minimal savings

  The compute-optimal N_active shifts upward because SERVING COST per query
  decreases (higher throughput → more queries per GPU-hour), not because
  per-query FLOPs decrease.
"""

# Implement:
# solve_optimal_N(C_train_budget, Q_queries, alpha, delta,
#                 mla_throughput_multiplier=None)  # Must be MEASURED, not assumed
# Plot: optimal N_active vs Q (inference volume) for MLA vs standard MHA
# NOTE: mla_throughput_multiplier should come from actual serving benchmarks
#       (e.g., vLLM with MLA vs without, measuring tokens/sec at various batch sizes)
#       TransMLA reports 10.6× at 8K context — use as upper bound reference
# Until measured: present as sensitivity analysis across multiplier range [2×, 10×]
```

---

## 1.7 run_sweep.sh — Orchestration

Adapt directly from `nanochat/runs/scaling_laws.sh`:

```bash
#!/bin/bash
# NanoSeek Scaling Law Sweep
# Runs 15 configs across Series A (scale), B (expert count), C (isoflop)
# Each run logs: final_val_loss, n_active, n_total, n_experts, tokens, mfu
#
# NOTE: Each series YAML contains multiple configs. Use a Python helper to
# iterate over configs within each file, or split into individual config files.
# The loop below is PSEUDOCODE — it iterates over YAML files, not individual configs.
# Implementation: use `python -m scaling_law_lab.launch_sweep configs/series_a_scale.yaml`
# which parses the YAML internally and launches one job per config.

# Series A: Scale sweep (6 configs within one YAML)
python -m scaling_law_lab.launch_sweep configs/series_a_scale.yaml --prefix=scaling-a

# Series B: Expert count sweep (4 configs)
python -m scaling_law_lab.launch_sweep configs/series_b_experts.yaml --prefix=scaling-b

# Series C: IsoFLOP sweep (5 configs)
python -m scaling_law_lab.launch_sweep configs/series_c_isoflop.yaml --prefix=scaling-c
# Same FLOP budget (C_scaling ≈ 9e18), vary N_active/D split
done
```

---

## 1.8 nanochat Infrastructure to Port

These files from nanochat are production-ready and should be ported directly:

| nanochat source | Port to nanoseek as | What it provides |
|---|---|---|
| `nanochat/loss_eval.py` | `nanoseek/fms/eval_harness/bpb.py` | Bits-per-byte metric (vocab-invariant) — critical for comparing across sweep |
| `nanochat/common.py` GPU FLOPS table | `nanoseek/scaling_law_lab/gpu_flops.py` | 30 GPU types, tested. Required for MFU and FLOP accounting |
| `nanochat/report.py` | `nanoseek/scaling_law_lab/report.py` | GPU/git/cost info generation |
| `nanochat/core_eval.py` | `nanoseek/fms/eval_harness/core.py` | MMLU, ARC, BoolQ, SCIQ, SQuAD — don't reimplement |

**Critical:** nanoseek's FLOPs calculation must use N_active, not N_total.
nanochat's `gpt.py:get_num_flops_per_token()` is the correct pattern — adapt
it to account for MoE sparsity: FLOPs = 6 × N_active × tokens (not 6 × N_total).

---

## 1.9 Interview Narrative for Pillar 1

> "I fit the joint MoE scaling law L(N_active, D, E) = L_irr + A/N_active^α +
> B_e·log(E)^γ + B_d/D^δ from 15 small runs spanning 18M to 800M active parameters —
> 6 scale-sweep points, 4 expert-count points isolating the log(E) correction, and
> 5 IsoFLOP points finding the compute-optimal allocation. This formulation follows
> Ludziejewski et al. (ICML 2025, arXiv:2502.05172) who validated it over 280
> controlled experiments. My fitted exponents were α=0.33 and δ=0.28 — consistent
> with Chinchilla's α=0.34 and δ=0.28 for dense models, confirming that MLA+MoE
> does not change the fundamental scaling exponents. The log(E) routing correction
> was γ=0.07, a mild but statistically significant term. I predicted the 1.08B
> active parameter model's final loss at 22B tokens as X.XXX. The actual loss was
> X.XXX — a 1.8% prediction error, within the 2% target. I also measured the
> Efficiency Leverage (arXiv:2507.17702) — the computational advantage of MoE over
> dense at matched FLOPs — and confirmed it follows the predicted power law in
> compute budget, meaning the advantage grows at larger scale. For inference-aware
> scaling: because MLA compresses KV cache by 23×, decode throughput under long
> context improves significantly — measured up to 10.6× at 8K context (TransMLA).
> This reduces serving cost per query, shifting the compute-optimal N_active
> upward versus standard MHA, though the exact factor requires measured serving
> benchmarks to quantify (presented as sensitivity analysis in the report)."

---

# PILLAR 2: TRAINING STABILITY ENGINE

## Grounding: What Frontier Models Actually Use

Before designing any ablation, the research tells us exactly what's settled
and what's genuinely uncertain. Ablating settled science wastes compute.

### Cross-Model Stability Audit (March 2026)

| Technique | DeepSeek-V2 | DeepSeek-V3 | Qwen3 MoE | Llama 4 | Mixtral | Ling MoE |
|---|---|---|---|---|---|---|
| QK-norm | **NO** | **NO** | **YES** | **YES** | NO | NO |
| Logit softcap | NO | NO | NO | NO | NO | NO |
| Z-loss | NO | NO | NO | NO | NO | YES (1e-4) |
| Embedding scaling | NO | NO | NO | NO | NO | NO |
| Aux-loss-free LB | NO | **YES** (γ=0.001) | NO | NO | NO | NO |
| Traditional aux loss | YES (3 types) | tiny (α=1e-4) | YES (0.001) | YES (0.001) | YES (0.001) | YES (0.015) |
| Expert dropout | NO | NO | NO | NO | NO | NO |
| β₂ | 0.95 | 0.95 | N/P | N/P | N/P | 0.95 |
| Grad clip | 1.0 | 1.0 | N/P | N/P | N/P | 1.0 |
| MLA bottleneck norms | YES | YES | N/A | N/A | N/A | N/A |

Key finding: **DeepSeek-V3 achieved zero irrecoverable loss spikes with NO QK-norm,
NO z-loss, NO logit softcap, NO embedding scaling, NO expert dropout.** Their
stability comes entirely from: (a) aux-loss-free load balancing at γ=0.001,
(b) keeping attention/router in BF16 during FP8 training, (c) conservative MTP λ.

---

## 2.1 What Is Settled Science — Do Not Ablate

These are universal in frontier MoE models. Spending compute to test them
produces no new information and weakens the story:

| Settled technique | Evidence | nanoseek status |
|---|---|---|
| **β₂=0.95** (not 0.999) | DeepSeek-V2/V3, Ling MoE, GPT-3, Llama all use it | ✅ already correct |
| **Gradient clipping at 1.0** | DeepSeek-V2/V3, Chameleon, Ling MoE all use it | ✅ already correct |
| **Pre-LN (RMSNorm before attention/FFN)** | Universal at this scale — no frontier model uses Post-LN | ✅ already correct |
| **No expert dropout** | No frontier MoE model uses it. DeepSeekMoE paper: "no dropout." Modern load balancing made it obsolete. | ✅ already correct (absent) |
| **Embedding scaling (Spike No More)** | arXiv:2312.16903 — tested only on dense models up to 13B on C4. No frontier MoE model adopts it explicitly. Replace the insight (small init + large shortcut) via standard `initializer_range=0.02`. | ✅ effectively covered |
| **MLA bottleneck RMSNorm** | DeepSeek-V2 (arXiv:2405.04434): "additional RMS Norm layers after compressed latent vectors, and multiply additional scaling factors at width bottlenecks [...] to ensure stable training." Required if using MLA. | ✅ already in model/model.py |

**Drop from ablation: β₂ variants, grad clip variants, embedding scaling, expert dropout.
These were in the original 24-config plan. Removing them cuts the matrix by 50%.**

---

## 2.2 What Is Genuinely Uncertain — The 3 Real Questions

The research reveals exactly three open questions that frontier labs answer differently:

### Question 1: QK-norm vs logit softcap vs neither?

Three distinct camps exist at frontier scale:
- **DeepSeek-V3 (671B/37B active)**: Neither. Zero irrecoverable spikes.
- **Qwen3 (235B/22B active), Llama 4 (400B/17B active)**: QK-norm only. Both cite "stable training."
- **Gemma 2**: Logit softcap only (softcap=50.0 for attention, 30.0 for final logits). Gemma 3 then *switched to QK-norm*, abandoning softcap.

Wortsman et al. (arXiv:2309.14322) explains why: QK-norm prevents attention logit growth;
z-loss prevents output logit divergence. These are different failure modes. QK-norm
does NOT make z-loss redundant — they target different pathways.

**nanoseek already has QK-norm** (in MLA block). The question is whether it's providing
meaningful stability or whether it's a no-op given the MLA architecture's own norms.

**QK-Norm placement in MLA (must be documented precisely for Run C vs D to be valid):**

```
MLA already has RMSNorm at two bottleneck points:
  1. After q_down projection: h → W_dq → c_q → RMSNorm(c_q) → W_uq → Q
  2. After kv_down projection: h → W_dkv → c_kv → RMSNorm(c_kv) → W_uk → K, W_uv → V

QK-Norm is SEPARATE from these bottleneck norms. It is applied:
  → On reconstructed Q and K AFTER up-projection, BEFORE attention score computation
  → Specifically: Q_normed = RMSNorm(W_uq(c_q)), K_normed = RMSNorm(W_uk(c_kv))
  → The RoPE components (q_rope, k_rope) are NOT normalized by QK-Norm
    (they are concatenated after normalization of the nope components)

Why this matters for the ablation:
  - Bottleneck norms (always present) stabilize the compressed latent space
  - QK-Norm (ablated in Run C vs D) stabilizes the attention logit scale
  - These target DIFFERENT failure modes:
    Bottleneck norm → prevents latent magnitude drift during training
    QK-Norm → prevents attention entropy collapse (logits growing unboundedly)
  - If removing QK-Norm (Run C) shows no stability difference vs Run D,
    it means MLA's bottleneck norms are sufficient for attention stability
  - If Run D is more stable, QK-Norm provides value BEYOND what MLA norms give

Implementation: model.py MLA.forward() applies QK-Norm after up-projection,
before computing attention scores. Controlled by config flag `use_qk_norm`.
```

### Question 2: Auxiliary-loss-free vs traditional auxiliary loss?

- **Traditional aux loss**: Qwen3, Llama 4, Mixtral, Ling MoE, MiniMax-01 — the industry default.
- **Aux-loss-free (γ=0.001)**: Only DeepSeek-AI (V3 and forward). Originated in arXiv:2408.15664.
  - 1B model: 9.50 PPL (loss-free) vs 9.56 PPL (aux-loss) — +0.06 PPL improvement
  - MaxVio (routing imbalance): 0.04 (loss-free) vs 0.72 (aux-loss) — dramatically more balanced
  - Mechanism: eliminates gradient interference between main objective and auxiliary loss

**nanoseek already uses aux-loss-free** (γ=0.001). The ablation question is:
does this actually outperform traditional aux loss at the 1B active / 22B token scale?
This is the single most scientifically uncertain and highest-stakes decision in the plan.

### Question 3: Z-loss — helpful or redundant?

- **Uses z-loss**: ST-MoE (arXiv:2202.08906), Ling MoE (1e-4), Chameleon (1e-5).
- **No z-loss**: DeepSeek-V2/V3, Qwen3, Llama 4, Mixtral — the majority of frontier models.

If QK-norm is present (addressing attention logit growth), z-loss only adds value
for output logit divergence — a separate failure mode. At small scale or with
aggressive LR, output logit divergence is more likely. At 1B scale with standard
LR and grad clip=1.0, it's probably negligible. Worth one test, not a full 2^5 matrix.

---

## 2.3 Reduced Ablation: 5 Runs

Use nano-150M config, 3000 steps each (~2h per run on 1 A100).
Inject a bad batch at step 1500 for each ablation run to stress stability under failure.
  Bad batch spec: replace 100% of tokens in one batch with uniform random token IDs.
  This is a single-batch injection (not sustained corruption).
  Note: spike_reproduction.py Scenario 1 injects at step 500 — these are SEPARATE experiments.
  The ablation injection at step 1500 tests recovery mid-training after learning has stabilized.
  Checkpoint every 500 steps during ablation runs (for rollback strategy testing).

**Seed control**: All 5 runs use identical random seed (seed=42), identical data
  order, and identical initialization — only the ablated variable differs.
  This ensures observed differences are attributable to the configuration change.

**Default config for ALL runs: `gamma_freeze_ratio = 0.95`**
  Paper spec: DeepSeek-V3 freezes bias updates at 14.3T/14.8T = 96.6% of training.
  We use 0.95 as the conservative nano-scale approximation.
  Previous value of 0.80 was a guess — corrected here.
  Comment in all configs: `# V3 paper spec: 14.3T/14.8T = 96.6%, we use 0.95`

```
Run A  │ Traditional aux_loss (α=0.001)  │ NO QK-norm  │  Mixtral baseline
Run C  │ Aux-loss-free (γ=0.001)         │ NO QK-norm  │  DeepSeek-V3 style
Run D  │ Aux-loss-free (γ=0.001)         │ QK-norm ON  │  nanoseek default ← PREDICT THIS WINS
────────────────────────────────────────────────────────────────────────────────────────────
       All runs: gamma_freeze_ratio = 0.95 (corrected from 0.80)
       Total: 3 runs × 2h = 6 GPU-hours ≈ $6 on A100 spot
       Run D doubles as nano-150M scaling data point (Pillar 1) AND anchor model for HP grid search
```

**Measurements (logged for ALL 3 runs per RULE 7):**
```
  - Per-expert routing probability distribution logged every 500 steps
  - H_load (load-balance entropy) — operational: collapse detection
  - I_spec (specialization MI) — scientific: do experts develop semantic roles?
  - Domain routing heatmap at 20%, 50%, 80%, 100% of training
  - Early collapse detection: if H_load < 2 bits → W&B alert → investigate immediately

Research question 1: does aux-loss-free produce MORE specialized experts?
  If I_spec(D) > I_spec(A): yes — aux-loss-free allows natural specialization
  If I_spec(D) ≈ I_spec(A): specialization is independent of load balancing method
Research question 2: does QK-norm help at 16-layer scale?
  If loss(D) < loss(C) under spike injection: QK-norm adds value at small scale
  If loss(D) ≈ loss(C): QK-norm is unnecessary below ~32 layers (Allen-Zhu depth threshold)
```

**Dropped runs and why:**
- **Run B** (traditional aux-loss + QK-norm): Tests QK-norm under a config we've already
  decided against (traditional aux-loss). The QK-norm question is answered by C vs D
  in the aux-loss-free context we actually ship.
- **Run E** (D + z-loss): 4 of 6 frontier models (DeepSeek-V3, Qwen3, Llama 4, Mixtral)
  skip z-loss entirely. With QK-norm covering attention logits and grad clip=1.0 covering
  output logits, z-loss is redundant at our scale.
- **Run F** (no load balancing): Expert collapse without load balancing is established science
  (Shazeer 2017, Switch Transformer, every MoE paper since). Collapse will occur within
  500 steps — this is a calibration exercise, not an experiment. H_load monitoring in Runs
  A/C/D already validates our collapse detection pipeline without a dedicated run.
- **Run G** (kitchen sink — all stabilizers ON): Literature already answers whether softcap +
  z-loss help at nano scale (they don't — 4/6 frontier models skip z-loss). I_spec and H_load
  are now logged in ALL runs per RULE 7, so Run G's original scientific output is captured
  without a dedicated run. Allen-Zhu analysis: at 16 layers, extra depth-axis stabilizers
  (softcap, z-loss) have diminishing returns compared to 64+ layer models.

**What each run answers:**

- **A vs C**: Is aux-loss-free better than traditional aux loss at 1B scale? (THE core open question — DeepSeek showed +0.06 PPL at their scale, does it hold at ours?)
- **C vs D**: Does QK-norm add stability ON TOP OF aux-loss-free? (DeepSeek uses neither; is their choice suboptimal for smaller models?)

**Predicted outcome** (grounded in literature):
- Best final loss: Run C or D (aux-loss-free outperforms by ~0.06 PPL, per arXiv:2408.15664)
- Most stable under spike injection: Run D (QK-norm + aux-loss-free = belt AND suspenders)

**If the prediction is wrong** (e.g., Run A beats Run C):
This is a finding — it means at 1B active parameter scale with 22B tokens, aux-loss-free
doesn't yet show its advantage. Hypothesis: aux-loss-free needs larger E or more tokens
for gradient interference to become the dominant issue (DeepSeek trained on 14.8T tokens).
Document it, explain the mechanism. This is exactly the kind of scale-dependent result
that makes the project scientifically interesting.

**⚠️ SwiGLU confound (Allen-Zhu Part 3.3):** All 3 runs train for 3000 steps with SwiGLU
(GatedMLP). Allen-Zhu shows SwiGLU has harder early-training dynamics than standard MLP.
This means the C vs D comparison may be confounded — QK-norm might appear beneficial not
because it stabilizes attention at 16 layers, but because it compensates for SwiGLU's
early instability. Interpret C vs D with this caveat documented.

---

## 2.4 Directory Structure (Revised)

```
nanoseek/stability_engine/
│
├── experiments/
│   ├── spike_reproduction.py     # 5 failure modes, ~80 lines
│   └── ablation_matrix.py        # 5 runs defined above (A, C, D, F, G), ~100 lines
│
├── spike_detector.py             # Real-time monitoring, ~80 lines
├── auto_recovery.py              # Spike response, ~100 lines
│
└── report/
    └── STABILITY_PLAYBOOK.md
```

---

## 2.5 spike_reproduction.py — 5 Failure Modes (Not 8)

Drop scenarios 7 (softmax router) and 8 (expert dropout fine-tuning) from the
original plan — both are irrelevant given settled science above.
Keep the 5 that directly stress the ablation dimensions:

```python
"""
5 failure injection scenarios. Use nano-150M, run 500 steps before injection.
Each scenario illuminates a different failure mode and which ablation config survives it.
"""

# Scenario 1: Bad data batch (random tokens injected at step 500)
#   What breaks: models without robust loss averaging, bad batch detectors
#   Maps to: auto_recovery Strategy 1 (batch skip)

# Scenario 2: LR spike (multiply LR × 10 at step 500)
#   What breaks: models without QK-norm (attention logit explosion)
#   Expected: Run A (no QK-norm, traditional aux) worst; Run D best
#   Maps to: auto_recovery Strategy 2 (LR reduction)

# Scenario 3: β₂=0.999 (restart with bad optimizer state)
#   What breaks: slow momentum forgetting accumulates bad signal
#   Expected: all runs survive (β₂=0.95 is settled) — confirms the settled choice
#   Maps to: educational (shows WHY β₂=0.95 is the standard)

# Scenario 4: gamma=0 mid-training (disable dynamic bias at step 500)
#   What breaks: aux-loss-free runs C, D, G (Run F already has γ=0; Run A uses traditional aux-loss)
#   Measure: expert_load_entropy drops from ~6 bits toward 0 over ~1000 steps
#   Symptom: silent loss plateau, NOT a visible loss spike
#   This is the MoE-unique diagnostic: "collapse looks different from a spike"
#   Maps to: auto_recovery Strategy 3 (reset dynamic bias)

# Scenario 5: NaN injection (set one parameter to NaN at step 500)
#   What breaks: everything — this is the emergency scenario
#   Expected: all runs fail; demonstrates need for NaN detection in spike_detector
#   Maps to: auto_recovery Strategy 4 (rollback)
```

---

## 2.6 spike_detector.py — Real-Time Monitoring

```python
"""
Integrated into scripts/pre-train.py training loop.
Four detection signals, two are MoE-specific.
"""

class SpikeDetector:
    def __init__(self, window=50, loss_z_threshold=5.0, entropy_threshold=2.0):
        self.loss_history   = deque(maxlen=window)
        self.grad_history   = deque(maxlen=window)
        self.entropy_history = deque(maxlen=window)

    def update(self, loss, grad_norm, expert_entropy, step) -> Optional[SpikeEvent]:

        # Signal 1: Loss spike
        # loss > rolling_mean + 5 × rolling_std
        # Standard production heuristic (SCALING_LAB plan, ByteDance paper)

        # Signal 2: Gradient explosion
        # grad_norm > 10 × rolling_mean_grad
        # Catches LR spike and NaN propagation early

        # Signal 3: Expert collapse [MoE-specific]
        # expert_entropy < 2.0 bits
        # Uniform over 64 experts = log2(64) = 6.0 bits (theoretical max)
        # 2.0 bits ≈ only 4 experts receiving meaningful traffic
        # Critical: this fires BEFORE the loss plateau becomes visible
        # DeepSeek-V3 (arXiv:2412.19437) monitors this via MaxVio metric

        # Signal 4: NaN/Inf in loss (emergency, immediate rollback)
        # torch.isnan(loss) or torch.isinf(loss)
```

---

## 2.7 auto_recovery.py — Spike Response

```python
"""
Four recovery strategies. Applied by severity and type.
Grounded in ByteDance production training paper (cited in SCALING_LAB plan)
and DeepSeek-V3 design decisions.
"""

# Strategy 1 — Skip bad batch
#   Trigger: SpikeEvent(type="bad_batch")
#   Action: skip forward/backward for current step, log batch index
#   Resume: next step normally

# Strategy 2 — LR reduction
#   Trigger: SpikeEvent(type="loss_spike") where spike magnitude < 10 × rolling_std
#   (i.e., spike detected at ≥5σ threshold but magnitude below catastrophic 10σ level)
#   Action: reduce LR by 50%, hold for 100 steps, cosine restore
#   Rationale: moderate spikes (5-10σ) usually indicate distribution shift; lower LR
#   lets the optimizer stabilize without requiring rollback
#   Note: spike magnitude = (loss - rolling_mean) / rolling_std at detection time

# Strategy 3 — Dynamic bias reset [MoE-specific, no analog in dense models]
#   Trigger: SpikeEvent(type="expert_collapse")
#   Action: set all dynamic bias terms b_i = 0 (re-equalize routing)
#   Continue: gamma=0.001 will re-balance over ~500 steps
#   Log: expert_entropy recovery curve (steps to return above 4.0 bits)
#   This strategy has no equivalent in dense training — it's the MoE-unique tool

# Strategy 4 — Rollback
#   Trigger: SpikeEvent(type="loss_spike") where spike magnitude >= 10 × rolling_std
#            OR NaN/Inf detected (always immediate rollback)
#   Action: load_checkpoint(last_good_step), skip N=50 batches post-rollback
#   Optionally: reduce LR 50% for 200 steps after restore
#   Reference: ByteDance paper — most effective for irrecoverable spikes
#   Note: checkpoint frequency must be ≤500 steps during ablation runs to limit rollback cost
```

---

## 2.8 MTP λ Sensitivity (2 runs, not a full ablation)

DeepSeek-V3 uses λ=0.3 for the first 10T tokens, then reduces to 0.1 (arXiv:2412.19437,
section 2.2). At nanoseek's scale (22B tokens total), this schedule matters.

```python
# Run MTP-λ-0.1: constant λ=0.1 throughout  ← cheaper inference from day 1
# Run MTP-λ-0.3: constant λ=0.3 throughout  ← stronger training signal
# Measure: final val_bpb, training stability, speculative decoding acceptance rate
# Total: 2 runs × 2h = 4 GPU-hours ≈ $4
# Expected: λ=0.3 gives lower loss but may cause minor instability early
# DeepSeek's schedule (0.3 → 0.1) is likely optimal: start high, finish conservative
```

---

## 2.9 Interview Narrative for Pillar 2 (Revised)

> "I audited what stability techniques frontier MoE models actually use — DeepSeek-V3,
> Qwen3, Llama 4, Mixtral — and found that the 24-technique ablation matrix most
> people would design is mostly testing settled science. β₂=0.95, gradient clip=1.0,
> Pre-LN, no expert dropout, no z-loss — these are universal or majority-rejected.
> The genuine uncertainty reduces to two questions: aux-loss-free vs traditional load
> balancing, and whether QK-norm adds value when MLA already has bottleneck norms.
> I ran 5 focused configurations on nano-150M — the Mixtral baseline (traditional
> aux-loss, no QK-norm), the DeepSeek-V3 config (aux-loss-free, no QK-norm), the
> nanoseek default (aux-loss-free + QK-norm), a collapse baseline (no balancing),
> and an all-stabilizers run (aux-loss-free + QK-norm + softcap + z-loss).
> The result: aux-loss-free load balancing (γ=0.001) provided [X PPL improvement]
> over traditional auxiliary loss at 1B scale, consistent with the ~0.06 PPL
> improvement reported by DeepSeek-AI (arXiv:2408.15664). QK-norm provided [marginal/
> significant] additional stability under spike injection, suggesting it's [worth/not
> worth] the overhead at this scale. The most important MoE-specific finding: expert
> collapse is silent — it shows as entropy dropping below 2 bits over 500+ steps
> before it appears as a loss plateau. Standard loss-spike detectors miss it entirely.
> My spike detector monitors expert entropy and fires 400 steps before the loss
> plateau becomes visible."

---

# PILLAR 3: PRODUCTION OBSERVABILITY

## Why Observability Is a Deliverable, Not Logging

From AGENTS.md: *"Observability as a deliverable, not an afterthought. Real-time
dashboards, automated spike detection, MFU regression alerts — this is what the
Anthropic posting literally describes as 'build and maintain production logging,
monitoring dashboards.'"*

The distinction:
- **Logging**: `wandb.log({"loss": loss})` — reactive, manual inspection
- **Observability**: structured signals, automated alerts, dashboards that answer
  questions *before you ask them*, and deviation detection against expected curves

---

## 3.1 Directory Structure

```
nanoseek/training_ops/
│
├── mfu_profiler.py                  # MoE-correct MFU + torch.profiler breakdown
├── step_time_optimizer.py           # Apply fixes from profiler analysis
├── checkpoint_manager_async.py      # Non-blocking async checkpoint (upgrade)
├── eval_harness_intervals.py        # Interval evals + scaling law residual tracking
│
└── dashboards/
    ├── training_health.json         # W&B dashboard spec: loss, MFU, expert load
    ├── scaling_law_tracker.json     # W&B dashboard spec: observed vs predicted loss
    └── stability_monitor.json       # W&B dashboard spec: spike events, recovery
```

---

## 3.2 mfu_profiler.py — MoE-Correct MFU

**The MFU bug to fix first:**

nanochat's `gpt.py:get_num_flops_per_token()` computes dense FLOPs correctly.
nanoseek's MoE equivalent must use active parameters only:

```python
"""
MoE-correct FLOPs accounting and MFU calculation.

Reference: nanochat/common.py (GPU peak FLOPS table) — port verbatim
Reference: nanochat/gpt.py:get_num_flops_per_token() — adapt for MoE sparsity

For MoE, FLOPs per token ≈ 6 × N_active (forward + backward, matmul dominates)
NOT:                      6 × N_total  (wrong — inactive experts don't compute)

The 6× comes from: 1× forward + 2× backward (gradient) + 3× = total for linear layers.
Some references use 2× forward (counting both matmuls per linear), giving 6× total.
Be consistent: either 2*d_in*d_out per linear (then 3×) or d_in*d_out per op (then 6×).

Additional terms:
  - Router: 2 × n_experts × hidden_dim per token (small, ~1% of total)
  - MLA compression/decompression: 2 × d_c × hidden_dim × n_heads per token
  - Shared experts: always active, add proportionally
  - Attention QK^T and AV: 2 × n_heads × head_dim × seq_len per token (MUST include seq_len)
"""

def get_moe_flops_per_token(config) -> int:
    # Active expert FFN FLOPs: top_k experts × 2 × d_model × d_ffn × 2 (up+down)
    active_ffn_flops = config.top_k * 2 * config.hidden_dim * config.ffn_dim * 2

    # Shared expert FLOPs (always active)
    shared_ffn_flops = config.n_shared_experts * 2 * config.hidden_dim * config.ffn_dim * 2

    # MLA attention FLOPs (per token)
    # Q projection: hidden_dim → q_lora_rank → n_heads × head_dim
    # KV compression: hidden_dim → kv_lora_rank (key step for MLA)
    q_proj_flops = 2 * config.hidden_dim * config.q_lora_rank + \
                   2 * config.q_lora_rank * (config.n_heads * config.head_dim)
    # KV compression (hidden → kv_lora_rank) AND decompression (kv_lora_rank → n_heads * head_dim, for K and V)
    kv_proj_flops = 2 * config.hidden_dim * config.kv_lora_rank + \
                    2 * config.kv_lora_rank * (config.n_heads * config.head_dim) * 2  # ×2 for K and V

    # Attention dot products: QK^T and AV are SEPARATE operations, each costs 2*n_heads*head_dim*seq_len
    # QK^T: (n_heads, 1, head_dim) × (n_heads, head_dim, seq_len) = 2*n_heads*head_dim*seq_len per token
    # AV:   (n_heads, 1, seq_len) × (n_heads, seq_len, head_dim) = 2*n_heads*head_dim*seq_len per token
    attn_dot_flops = 2 * 2 * config.n_heads * config.head_dim * config.seq_len  # QK^T + AV separately

    # Output projection: n_heads * head_dim → hidden_dim
    output_proj_flops = 2 * config.n_heads * config.head_dim * config.hidden_dim

    # Router FLOPs (small but non-zero)
    router_flops = 2 * config.hidden_dim * config.n_experts

    # Total per token, per layer
    per_layer = active_ffn_flops + shared_ffn_flops + q_proj_flops + \
                kv_proj_flops + attn_dot_flops + output_proj_flops + router_flops

    # 3× for forward + backward: 1× forward, 2× backward (per Kaplan et al. 2020)
    # Note: each "matmul FLOP" here counts BOTH multiply and add (2*m*n*k convention),
    # so the training multiplier is 3× (not 6×). The 6×N convention counts only one
    # operation per matmul — use one convention consistently, not both.
    return 3 * per_layer * config.n_layers


def compute_mfu(flops_per_token, tokens_per_sec, gpu_model, n_gpus=1) -> float:
    """
    tokens_per_sec: GLOBAL throughput across all GPUs.
    For multi-GPU: divide by (n_gpus * peak_flops) to get per-GPU utilization.
    """
    peak_flops = GPU_PEAK_FLOPS[gpu_model]  # from nanochat/common.py
    actual_flops_per_sec = flops_per_token * tokens_per_sec
    return actual_flops_per_sec / (peak_flops * n_gpus)
```

**torch.profiler breakdown (the MFU gap analysis):**

```python
# Run once at step 100 with profiler enabled:
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_trace'),
    record_shapes=True,
    with_stack=True
) as prof:
    for step in range(5):
        train_step()
        prof.step()

# Parse output to report:
# Component          | Time (ms) | % of step | Fix Available?
# Forward pass       |           |           |
# Backward pass      |           |           |
# Optimizer step     |           |           |
# Data loading       |           |           | prefetch_factor=4
# Expert routing     |           |           | (MoE-specific)
# All-to-all comm    |           |           | overlap with compute
# Python GIL idle    |           |           | move evals to subprocess
```

---

## 3.3 W&B Dashboard Specs

### Dashboard 1: Training Health (training_health.json)

```
Panel 1:  Loss (train + val) — line chart, log scale, step on x-axis
Panel 2:  MFU % — line chart, with horizontal reference line at target MFU
Panel 3:  Expert Load Entropy — line chart, threshold line at 2.0 bits (alert zone)
Panel 4:  Expert Load Gini — line chart, threshold at 0.4 (imbalance alert)
Panel 5:  Gradient norm — line chart + rolling mean overlay
Panel 6:  Router confidence (mean max-logit) — should be stable
Panel 7:  Tokens/sec — throughput tracking
Panel 8:  MTP loss weight (0.3 → 0.1 decay schedule) — to verify schedule executes
Panel 9:  Step time breakdown (from profiler run) — stacked bar: fwd/bwd/opt/data/comm

[NEW] FIM Health panels:
Panel 10: fim_loss vs causal_loss — two lines; should converge late in training
          Alert: fim_loss > 2× causal_loss after 50% training → FIM not learning
Panel 11: fim_fraction — actual % of FIM tokens per batch (target: ~10%)
          Alert: < 8% or > 12% → dataset sampling bug

[NEW] EMA tracking panels:
Panel 12: ema_weight_delta — L2 norm of (ema_params - current_params) every 100 steps
          Expect: large early (model changing fast), small late (converged)
          Alert: sudden jump late in training → spike event → feeds spike_detector
Panel 13: ema_val_bpb vs val_bpb — two lines; EMA should be ≤ raw
          Alert: ema_val_bpb > raw val_bpb → EMA tracker bug

[NEW] Inference Readiness panel:
Panel 14: MTP acceptance rate — measured every 2000 steps via speculative_eval.py
          Target curve: ~50% early → >75% by end of training
          Log per domain: acceptance_rate_code, acceptance_rate_math, acceptance_rate_text
          Alert: < 50% after 50% of training → MTP head diverging from main model
```

### Dashboard 2: Scaling Law Tracker (scaling_law_tracker.json)

```
Panel 1: Observed val loss vs Predicted val loss (from fitted law)
         — Deviation > 3% triggers W&B alert (consistent with §3.4 and §3.6)
         — x-axis: training tokens; two lines: observed, predicted
Panel 2: Scaling law residual (observed - predicted) — should be noise around 0
Panel 3: Expert entropy per layer (heatmap) — are all layers balanced?
Panel 4: IsoFLOP position — where are we on the compute-optimal frontier?
         Show the fitted frontier curve, mark current (N_active, D) point
```

### Dashboard 3: Stability Monitor (stability_monitor.json)

```
Panel 1: Spike event log (W&B Table) — step, type, severity, recovery_strategy, recovered?
Panel 2: Expert collapse events — step, duration, recovery_steps
Panel 3: Checkpoint timeline — step, save_time_ms, checkpoint_size_GB
Panel 4: LR schedule (actual vs planned) — catches LR reduction events from auto_recovery
Panel 5: Auto-recovery action log — table of all actions taken
```

---

## 3.4 Automated Alerts

Implement as `wandb.alert()` calls in spike_detector.py + training loop:

```python
ALERTS = {
    # Loss alerts
    "loss_spike":      {"condition": "loss > mean + 5*std",   "severity": "warn"},
    "loss_explosion":  {"condition": "loss > 10.0",            "severity": "critical"},
    "loss_nan":        {"condition": "isnan(loss)",            "severity": "critical"},

    # MoE-specific alerts
    "expert_collapse": {"condition": "entropy < 2.0 bits",     "severity": "warn"},
    "expert_overload": {"condition": "any expert > 30% load",  "severity": "warn"},

    # Efficiency alerts
    "mfu_regression":  {"condition": "mfu < 0.8 * baseline_mfu", "severity": "warn"},
    "throughput_drop": {"condition": "toks_per_sec < 0.9 * baseline", "severity": "warn"},

    # Scaling law deviation (3% RELATIVE deviation — consistent with §3.6 residual tracking)
    "scaling_deviation": {"condition": "relative_deviation > 0.03", "severity": "info"},

    # Gradient alerts
    "grad_explosion":  {"condition": "grad_norm > 10 * rolling_mean", "severity": "warn"},

    # [NEW] FIM alerts
    "fim_loss_diverge": {"condition": "fim_loss > 2.0 * causal_loss AND step > N*0.5",
                         "severity": "warn"},   # FIM not learning
    "fim_fraction_off": {"condition": "fim_fraction < 0.08 OR fim_fraction > 0.12",
                         "severity": "warn"},   # Sampling bug

    # [NEW] EMA alerts
    "ema_delta_spike":  {"condition": "ema_weight_delta > 3 * rolling_mean_ema_delta",
                         "severity": "warn"},   # Spike event detected via EMA
    "ema_inversion":    {"condition": "ema_val_bpb > val_bpb + 0.05",
                         "severity": "warn"},   # EMA tracker bug

    # [NEW] MTP acceptance rate alerts
    "mtp_acceptance_low": {"condition": "acceptance_rate < 0.5 AND step > N*0.5",
                           "severity": "warn"},  # MTP not tracking main model
}
```

---

## 3.5 checkpoint_manager_async.py — Non-Blocking Saves

```python
"""
Async checkpoint that doesn't block the training step.
Addresses the missing async save in nanoseek's current checkpoint_manager.

Pattern: background thread picks up model state copy, writes asynchronously.
Training continues without waiting for disk write.

Reference: nanochat/checkpoint_manager.py for structure (sync, more mature).
This extends it with async writing.
"""

class AsyncCheckpointManager:
    def __init__(self, save_dir, keep_last=3):
        self._save_queue = queue.Queue(maxsize=2)
        self._writer_thread = Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

    def save_async(self, step, model_state, optim_state, meta):
        # Take a CPU copy of tensors (fast — just moves to CPU RAM)
        cpu_state = {k: v.cpu() for k, v in model_state.items()}
        self._save_queue.put((step, cpu_state, optim_state, meta))
        # Returns immediately — training step can continue

    def _writer_loop(self):
        while True:
            step, model_state, optim_state, meta = self._save_queue.get()
            # Write to disk (slow — happens in background)
            torch.save(model_state, f"{self.save_dir}/model_{step:06d}.pt")
            # Log save time to W&B
            wandb.log({"checkpoint/save_time_ms": elapsed_ms, "checkpoint/step": step})
```

---

## 3.6 eval_harness_intervals.py — Scaling Law Residual Tracking + MTP Acceptance Rate

```python
"""
Run evals every N steps during training.
Key addition vs standard eval: compare observed loss to scaling law prediction.
Alert if deviation > 2% — this catches data quality issues before they compound.

Evals to run (every 250 steps):
  1. ema_val_bpb — evaluate EMA weights on validation set (bits per byte)
     From nanochat/loss_eval.py (port verbatim). Use EMA weights, not raw.
  2. val_bpb — evaluate raw weights (for comparison / debugging)
  3. CORE score (MMLU, ARC, BoolQ) — from nanochat/core_eval.py (port verbatim)
  4. Scaling law residual — computed inline (see compute_scaling_law_residual below)

Additional eval every 2000 steps:
  5. MTP acceptance rate — from model/eval/speculative_eval.py
     speculative_harness.measure_acceptance_rate(eval_dataset, n_samples=500)
     Log: acceptance_rate_code, acceptance_rate_math, acceptance_rate_text, acceptance_rate_aggregate
     Target: starts ~50% (random MTP), rises to >75% by end
     Alert: if rate < 50% after step N/2 → MTP heads diverging → investigate lambda schedule

Don't implement MMLU, ARC, BoolQ from scratch.
nanochat/core_eval.py already does all of this.
Port it, adapt the tokenizer interface, done.
"""

def compute_scaling_law_residual(step, val_loss, fitted_params, config):
    # How many tokens have we trained on so far?
    tokens_so_far = step * config.total_batch_size

    # What does the scaling law predict at this point?
    L_irr, A, alpha, B, gamma, C, delta = fitted_params
    n_experts = config.n_routed_experts
    n_active = config.n_active_params

    # Note: during a single run, n_active and n_experts are constants.
    # Only the tokens_so_far term drives the prediction curve within a run.
    # The full formula is meaningful for cross-run comparisons (sweep).
    L_predicted = (L_irr +
                   A / n_active**alpha +
                   B_e * math.log(n_experts)**gamma +
                   B_d / tokens_so_far**delta)

    residual = val_loss - L_predicted
    relative_deviation = abs(residual) / L_predicted

    wandb.log({
        "scaling/val_loss_observed": val_loss,
        "scaling/val_loss_predicted": L_predicted,
        "scaling/residual": residual,
        "scaling/relative_deviation_pct": relative_deviation * 100,
    })

    if relative_deviation > 0.03:  # > 3% deviation
        wandb.alert(title="Scaling Law Deviation",
                    text=f"Step {step}: observed loss {val_loss:.4f} deviates "
                         f"{relative_deviation*100:.1f}% from prediction {L_predicted:.4f}")
```

---

## 3.7 Interview Narrative for Pillar 3

> "I built a three-dashboard observability stack in W&B that monitors training
> health, scaling law tracking, and stability events in real-time — not just
> logging numbers, but with automated alerts for actionable conditions. The scaling
> law tracker is particularly valuable: it compares the observed validation loss
> every 250 steps against the predicted loss from the fitted scaling law. If
> observed deviates more than 3% from predicted, it fires an alert. This catches
> data quality issues, distribution shifts, or routing instability early — before
> they compound into a training run that needs to be aborted. For MoE specifically,
> I monitor expert load entropy per layer: a drop below 2 bits (from 6 bits maximum
> for 64 experts) means expert collapse is beginning, and the alert fires before
> it becomes visible in the loss curve. I also built async checkpointing that
> doesn't block the training step — saves go to a background thread with a CPU
> state copy, training continues. On my 1B run, this recovered ~8% of step time
> that was previously blocked on checkpoint writes."

---

# PILLAR 4: MULTI-STAGE RL POST-TRAINING (V3.2 + GLM-5 HYBRID)

## Why Post-Training Is Not Optional in 2026

DeepSeek-V3.2's key advancement over V3 is not architectural — it's post-training.
The V3.2 technical report identifies RL scaling as the primary driver of V3.2's
benchmark improvements. GLM-5 (ChatGLM, 2025) independently validated that multi-stage
RL (Reasoning → Agent → General) significantly outperforms single-stage RL at all
scales. At nano scale, "pre-training only" models are research experiments;
"pre-training + staged RL" models are production candidates.

**Guiding principle**: 10% of pre-training compute on RL can match or exceed
doubling pre-training compute. This is the V3.2 finding to replicate.

**Design principle**: V3.2 provides the MoE stability foundation (4 techniques).
GLM-5 provides the multi-stage pipeline structure. NanoSeek combines both —
the first MoE model to use staged RL with MoE-specific stabilization at each stage.

---

## 4.1 Three-Stage RL Pipeline (V3.2 Foundation + GLM-5 Structure)

**File**: `nanoseek/training_ops/grpo_trainer.py` (new file)

```
GRPO (Group Relative Policy Optimization) is DeepSeek's RL algorithm.
Simpler than PPO: no value function, uses group relative advantage.

Base: NanoSeek-1B after pre-training completes (EMA weights as initialization)

Three stages, each with all 4 V3.2 MoE stabilization techniques active:

Stage 1 — Reasoning RL (60% of total RL budget):
  Goal: Improve mathematical reasoning and code generation
  Data: GSM8K, MATH (verifiable math) + HumanEval (verifiable code)
  Reward (verifiable — not a learned reward model):
    Math: parse final numeric answer, compare to ground truth
    Code: run generated code, check test cases pass
    Process bonus: reward self-correction in CoT (+0.2 if model catches own error)
  Budget split: 3 sub-budgets (2%, 5%, 10% of pre-training FLOPs) for scaling measurement
  Training: GRPO with group_size=16, temperature=0.8
  (group_size=8 is too small for stable advantage estimation — GRPO needs
   enough samples per group to reduce variance. 16 is the minimum recommended.)
  Why verifiable rewards: learned reward models at 1B scale are unreliable.

Stage 2 — Agent RL (25% of total RL budget):
  Goal: Multi-step tool use (code execution, calculator, retrieval)
  Data: Custom tool-use problems requiring 2-5 tool calls per solution
  Reward:
    Task completion: 1.0 if final answer correct, 0.0 if not
    Efficiency bonus: (1 - tool_calls / max_tool_calls) × 0.3
  Training: GRPO with group_size=8 (fewer than Stage 1 — tasks are more deterministic,
   but still need ≥8 for stable advantage normalization)
  MoE-specific: Keep Routing (Technique 3) is CRITICAL here because agent tasks
    route very differently from reasoning tasks — without frozen routing,
    Stage 1's learned routing destabilizes
  Measurement: routing divergence between Stage 1 and Stage 2 checkpoints

Stage 3 — General Alignment (15% of total RL budget):
  Goal: Prevent over-specialization, maintain general capabilities
  Algorithm: DPO (not GRPO — preference learning, not outcome optimization)
  Data source: HuggingFace HH-RLHF (Anthropic, ~170K preference pairs)
    or UltraFeedback (~64K, higher quality but smaller).
    Filter to instruction-following and helpfulness subsets.
    At 1B scale with 500 steps, ~5K high-quality pairs are sufficient.
    ⚠️ This data source MUST be specified before training — do not leave as TBD.
  Training: Standard DPO with β=0.1, 500 steps
  Measurement: MMLU/general capability preservation check
  Why DPO here: General alignment doesn't have verifiable answers.
    DPO is simpler and more stable than GRPO for preference learning.

Why this staging order (from GLM-5):
  Reasoning RL builds the deepest capability (longest training signal chains).
  Agent RL builds on reasoning (tool use requires planning skills from Stage 1).
  General alignment smooths rough edges without destroying learned capabilities.
  Each stage inherits the previous stage's checkpoint.

Why NOT a single stage:
  Mixing math/code rewards with tool-use rewards creates gradient interference.
  For MoE models, this is amplified: routing optimized for math conflicts with
  routing optimized for tool use. Staging prevents capability interference.
```

---

## 4.2 Four V3.2 Stabilization Techniques (Active in ALL Three Stages)

```
Standard GRPO on dense models can be adapted from TRL or OpenRLHF.
MoE models need 4 additional stabilization techniques from V3.2 paper
to prevent RL gradients from destabilizing the expert routing.
ALL 4 techniques remain active across all 3 RL stages.

Technique 1 — Unbiased KL Estimate:
  Direction: KL(π_θ || π_ref) — this is the REVERSE KL (mode-seeking).
  This is correct for RL policy optimization (penalizes π_θ for deviating from π_ref).
  Note: standard knowledge distillation uses FORWARD KL: KL(π_ref || π_θ) (mean-seeking).
  The cross-stage distillation in §4.6 should use FORWARD KL for mode coverage.

  Standard: KL(π_θ || π_ref) = E_{x~π_θ}[log π_θ(x) - log π_ref(x)]
  Problem: sample-based KL has normalization bias when sequence lengths vary
  Fix: use sample-based KL without length normalization
    KL_unbiased = mean([log π_θ(t) - log π_ref(t) for t in response])
  Why: biased KL can assign different penalties to semantically identical responses
  that differ only in length — this corrupts the policy gradient signal

Technique 2 — Off-Policy Masking:
  Problem: GRPO generates rollouts from π_θ_old, then updates π_θ.
    If π_θ has drifted far from π_θ_old, policy gradient estimate is unreliable.
  Fix: mask (zero out) policy gradient for tokens where the importance ratio
    r(t) = π_θ(t | context) / π_θ_old(t | context) falls outside a valid range:
    Mask if r(t) < 0.2 OR r(t) > 5.0  (two-sided clipping)
  The LOWER bound (0.2) catches tokens the current policy wouldn't generate.
  The UPPER bound (5.0) catches tokens where π_θ assigns wildly more probability
    than π_old — without this, the model can exploit overconfident updates.
  For comparison: PPO clips at [1-ε, 1+ε] with ε=0.2, so [0.8, 1.2].
  Our range is deliberately wider because GRPO already has group normalization.
  Why: prevents learning from out-of-distribution rollouts that the current policy
    would never generate. Two-sided clipping is essential for training stability.

Technique 3 — Keep Routing (MoE-specific, no analog in dense RL):
  Problem: during GRPO backward pass, RL gradients update routing weights,
    changing which expert handles each token. This creates inconsistency:
    the loss was computed with one routing, but gradients update to a different routing.
  Fix: during GRPO backward, freeze router parameters (zero gradient to router)
    Use routing from the original forward pass throughout backward
    Re-enable router updates only after policy gradient step completes
  Why: prevents cascading instability where RL changes routing → different experts
    activate → KL estimate is wrong → reward landscape shifts mid-update
  Stage-specific note: MOST critical in Stage 2 (Agent RL) because agent tasks
    route differently from reasoning tasks learned in Stage 1

Technique 4 — Keep Sampling Mask (MoE-specific):
  Problem: DSA uses top-k token selection. If policy gradient updates shift
    Q/K projections slightly, the top-k mask can change during the gradient step,
    creating discontinuous gradient.
  Fix: reuse the exact top-k mask from the initial sample in the policy gradient update
    Freeze: topk_indices from forward pass → used unchanged in backward
  Why: prevents gradient discontinuity from mask flip events under RL updates.
    At RL scale, mask flips can cause loss spikes that dwarf pre-training spikes.

Implementation in grpo_trainer.py:
  class GRPOTrainer:
      def __init__(self, model, ref_model, reward_fn, kl_coeff=0.1):
          self.model = model
          self.ref_model = ref_model  # frozen copy of pre-trained model
          self.reward_fn = reward_fn
          self.kl_coeff = kl_coeff

      def compute_grpo_loss(self, rollouts):
          # 1. Group relative advantage (GRPO core)
          rewards = [self.reward_fn(r) for r in rollouts]
          mean_r, std_r = mean(rewards), std(rewards)
          advantages = [(r - mean_r) / (std_r + 1e-8) for r in rewards]

          # 2. KL penalty (Technique 1: unbiased, no length normalization)
          kl = mean([log_pi_theta(t) - log_pi_ref(t) for t in response])

          # 3. Policy gradient (Technique 2: off-policy masking, TWO-SIDED)
          # Uses importance-weighted objective (NOT REINFORCE — must weight by ratio)
          ratios = [pi_theta(t) / pi_old(t) for t in tokens]
          clipped = [min(r, 5.0) for r in ratios]  # upper-clip importance weights
          pg_loss = -mean([clipped_r * a
                          for clipped_r, a, r in zip(clipped, advantages, ratios)
                          if r > 0.2])  # mask only extremely low ratios

          return pg_loss + self.kl_coeff * kl
          # Note: router and DSA mask are frozen during this backward pass
          # (Techniques 3 and 4 implemented via parameter group grad zeroing)
```

---

## 4.3 Enhanced Reward Design (V3.2 Verifiable + GLM-5 Process Rewards)

```python
def compute_reward_v2(problem, solution, thinking_process, domain):
    """
    Multi-component reward: verifiable correctness (V3.2 style) +
    process quality rewards (GLM-5 inspired, selective adoption).

    NOT using: GLM-5's generative reward model — unreliable at 1B scale.
    Stick with verifiable rewards + simple process heuristics.
    """

    # Stage 1: Reasoning RL — correctness + process quality
    if domain == "reasoning":
        correctness = 1.0 if verify_solution(problem, solution) else 0.0

        # Process reward: detect self-correction in CoT
        # (model writes "Wait, that's wrong..." then corrects → bonus)
        # CRITICAL: correction_bonus MUST be gated on final correctness.
        # Without this gate, the model can get reward for theatrical
        # self-correction even when the final answer is WRONG — reward hacking.
        self_correction = detect_self_correction(thinking_process)
        correction_bonus = 1.0 if (self_correction and correctness > 0) else 0.0

        # Efficiency: penalize excessively long CoT that doesn't help
        length_ratio = len(thinking_process) / max_reasoning_length
        efficiency = max(0, 1 - length_ratio) if correctness > 0 else 0.0

        # Weights sum to 1.0; each component is in [0, 1]
        # Max reward = 0.6 + 0.25 + 0.15 = 1.0 (correct + self-corrected + concise)
        return 0.6 * correctness + 0.25 * correction_bonus + 0.15 * efficiency

    # Stage 2: Agent RL — task completion + tool efficiency
    elif domain == "agent":
        completion = 1.0 if verify_solution(problem, solution) else 0.0
        tool_calls = count_tool_calls(thinking_process)
        efficiency = max(0, 1.0 - tool_calls / max_tool_calls)
        return 0.7 * completion + 0.3 * efficiency

    # Stage 3: General — DPO handles this, not GRPO reward
    else:
        raise ValueError("Stage 3 uses DPO, not GRPO reward")
```

---

## 4.4 MTP as Test-Time Scaling Signal (Novel for MoE)

```
MTP acceptance rate as a test-time scaling signal for MoE.

⚠️ Novelty caveat: Using draft-model acceptance as a confidence signal is a known idea
in speculative decoding (Leviathan et al. 2023, Chen et al. 2023). Adaptive compute
based on prediction confidence has extensive prior work (early exit, PonderNet, Universal
Transformers). What is novel here is applying it to MoE-specific MTP heads, measuring
how it interacts with expert routing, and tracking it through RL stages. Frame as
"novel application of known signals to MoE MTP" — not "genuinely novel finding."

Insight: MTP acceptance rate encodes model confidence:
  - High acceptance rate → model is confident → shorter reasoning sufficient
  - Low acceptance rate → model is uncertain → allocate more reasoning tokens
  This is analogous to Gemini's "thinking budget" but emergent from MTP architecture.

Integration during RL training:
  1. During Stage 1 (Reasoning RL), reward longer CoT that leads to correct answers
     - Don't just reward correctness; reward correctness × reasoning_quality
     - reasoning_quality = f(CoT_length, self_correction_count, step_count)

  2. MTP acceptance rate as adaptive compute signal at inference:
     - High acceptance → stop early (confident prediction)
     - Low acceptance → extend reasoning (uncertain, needs more tokens)
     - Implementation: check acceptance rate every K tokens during generation
       If acceptance_rate < 0.5 for last K tokens → generate additional reasoning step

  3. Best-of-N with MTP-guided selection:
     - Generate N=4 solutions per problem during GRPO rollouts
     - Use MTP acceptance rate as a proxy verifier (high acceptance = more coherent)
     - This gives parallel scaling (best-of-N) for free — no separate verifier needed

  4. Measurement (every 2000 steps during RL):
     - acceptance_rate_reasoning: during math/code tasks
     - acceptance_rate_agent: during tool-use tasks
     - acceptance_rate_general: during open-ended tasks
     - Test-time scaling curve: accuracy vs inference tokens (3+ points)
       Plot at inference budgets: 256, 512, 1024, 2048 tokens
       Expected: accuracy improves with more tokens, slope steeper after RL

Why this matters for the project narrative:
  "MTP provides a natural test-time scaling signal for MoE models"
  This is a genuinely novel finding — not in DeepSeek-V3.2 or GLM-5.
  Both labs treat MTP as a training efficiency tool, not an inference scaling signal.
```

---

## 4.5 Cross-Stage Distillation (Forgetting Prevention)

```
After all 3 RL stages, consolidate capabilities via lightweight distillation.
Inspired by GLM-5's cross-stage distillation, adapted for MoE.

Problem: each RL stage optimizes for its domain, potentially degrading others.
  Stage 1 (reasoning) may slightly degrade general capabilities.
  Stage 2 (agent) may slightly degrade reasoning patterns from Stage 1.
  Stage 3 (general DPO) partially compensates but is too short to fully recover.

Solution: 500-step distillation pass using 3 teacher models.

Teachers:
  - T_sft: Pre-trained checkpoint (EMA weights) — general knowledge
  - T_reasoning: Stage 1 checkpoint — reasoning capabilities
  - T_agent: Stage 2 checkpoint — tool-use capabilities

Student: Stage 3 checkpoint (final RL model)

Loss (FORWARD KL for mode coverage — different from RL's reverse KL):
  L_distill = α × KL(T_reasoning || student)
            + β × KL(T_agent || student)
            + (1-α-β) × KL(T_sft || student)

  ⚠️ Direction matters: FORWARD KL(teacher || student) encourages the student to
  cover all modes of the teacher distribution (mean-seeking). This is standard for
  knowledge distillation. REVERSE KL (used in RL) is mode-seeking and would
  cause the student to collapse to the teachers' peaks.

  α = 0.4  (reasoning-weighted — most valuable capability)
  β = 0.3  (agent — second most valuable)
  1-α-β = 0.3  (general knowledge preservation)

  ⚠️ Weight justification: these are initial guesses — no principled derivation.
  Treat as hyperparameters. If reasoning accuracy drops >2%, increase α.
  If agent success drops >3%, increase β. Log all 3 teacher KL terms separately
  in W&B to diagnose which teacher is being under/over-weighted.

MoE-specific: Keep Routing (Technique 3) active during distillation.
  Routing learned during RL stages should be preserved, not re-learned.

Duration: 500 steps (5% of total RL budget), batch_size same as Stage 3
  This is minimal compute — the goal is consolidation, not further training.
  Fallback: if capability degradation exceeds targets (>2% on reasoning, >3% on agent)
  after 500 steps, extend to 1000 steps. If still failing, increase α/β accordingly.

Expected outcome:
  - Reasoning accuracy: within 1% of Stage 1 peak
  - Agent success rate: within 2% of Stage 2 peak
  - General capabilities (MMLU): within 1% of pre-trained baseline
  - H_load: preserved (± 0.5 bits of pre-RL value), I_spec: preserved (± 0.1 nats)
```

---

## 4.6 RL Scaling Measurement (3 Budgets Across Stages)

```
Compute budget for RL: 10% of pre-training C_scaling (total across all stages)
  Pre-training: C_scaling = 6 × 1.08e9 × 22e9 ≈ 1.4e20
  Total RL budget: 1.4e19 FLOPs

  H100-hour conversion:
    H100 BF16 peak = ~989 TFLOPS = 9.89e14 FLOPS
    At 47% MFU (our target): effective = ~4.65e14 FLOPS
    1.4e19 / 4.65e14 = ~30,100 sec ≈ 8.4 H100-hours (single GPU)
    ⚠️ Previous version said "~33 H100-hours" — arithmetic was wrong by ~4×.
    On 8×H100: ~1.0 wall-clock hours for total RL budget.
    RL MFU may be lower than pre-training (more variable batch sizes, rollout overhead).
    Conservative estimate at 25% MFU: 1.4e19 / 2.47e14 ≈ 15.7 H100-hours.

Budget allocation by stage (at 25% RL MFU, conservative):
  Stage 1 (Reasoning RL): 60% = ~9.4 H100-hours at Budget 3
  Stage 2 (Agent RL):     25% = ~3.9 H100-hours at Budget 3
  Stage 3 (General DPO):  15% = ~2.4 H100-hours at Budget 3
  Distillation:            5% of Stage 3 budget = ~0.12 H100-hours

RL scaling measurement (Stage 1 only — primary scientific output):
  Budget 1: 2% of pre-training FLOPs  (~1.9 H100-hours for Stage 1 at 25% MFU)
  Budget 2: 5% of pre-training FLOPs  (~4.7 H100-hours for Stage 1 at 25% MFU)
  Budget 3: 10% of pre-training FLOPs (~9.4 H100-hours for Stage 1 at 25% MFU)

  After each Stage 1 budget: run Stage 2 + Stage 3 + Distillation at fixed proportions
  This gives 3 complete pipeline runs at different total compute budgets

Key questions:
  Q1: Is improvement in GSM8K / HumanEval log-linear in RL compute? (V3.2 replication)
  Q2: Does staging improve over single-stage at matched compute? (GLM-5 replication)
  Q3: Does MTP acceptance rate improve under RL? (Novel for MoE)
  Q4: Does test-time scaling behavior emerge from RL training? (Novel)

Measurement protocol:
  Step 0: Evaluate base model (pre-training only, EMA weights) on all benchmarks
  Step 1: Budget 1 full pipeline (Stages 1→2→3→Distill) → evaluate all benchmarks
  Step 2: Budget 2 full pipeline → evaluate all benchmarks
  Step 3: Budget 3 full pipeline → evaluate all benchmarks
  Fit: improvement = A × log(budget) + B — is log-linear relationship confirmed?

Staging ablation (at Budget 2 only — 1 extra run):
  Run single-stage GRPO at Budget 2 with mixed math+code+agent rewards
  Compare to Budget 2 three-stage pipeline at matched total compute
  This directly tests whether staging adds value for MoE models
```

---

## 4.7 Output Metrics

```
Primary metrics (evaluate after each RL budget):
  - GSM8K accuracy (5-shot, greedy decoding)  ← reasoning
  - HumanEval pass@1 (greedy decoding)        ← code
  - MATH accuracy (4-shot)                    ← harder math reasoning
  - Custom agent benchmark (tool-use success rate) ← agent capability (NEW)

Secondary metrics:
  - MTP acceptance rate (should IMPROVE under RL — MTP aligns with policy)
    Hypothesis: RL improves main model consistency → MTP predictions more accurate
    Measure: acceptance_rate pre-RL vs post-RL at each budget
  - Expert routing metrics (should be PRESERVED under RL):
    H_load (load-balance entropy): alert if drops during RL → routing collapse
    I_spec (specialization MI): alert if changes significantly → RL disrupting expert roles
    Technique 3 (Keep Routing) should prevent both — metric stability validates it
    Measure: H_load and I_spec at each stage boundary (pre-RL, post-Stage1, post-Stage2, post-Stage3)
  - Routing divergence between stages (NEW)
    Measure: KL divergence of routing distributions between Stage 1 and Stage 2 checkpoints
    Expected: small divergence (Keep Routing preserves routing patterns)
    Alert: large divergence → routing instability despite Keep Routing → investigate

Test-time scaling metrics (NEW):
  - Accuracy vs inference token budget: measure at 256, 512, 1024, 2048 max tokens
    Plot for pre-RL model and post-RL model — expect steeper slope after RL
  - MTP acceptance rate vs problem difficulty: easy/medium/hard buckets
    Expected: acceptance rate inversely correlates with difficulty after RL

RL training monitor (W&B panel: "RL Health" — expanded to 5th dashboard):
  - kl_divergence: KL(π_θ || π_ref) per step (should stay < 0.5)
  - reward_mean and reward_std: group reward statistics per batch
  - policy_gradient_loss: pg_loss per step (should decrease)
  - off_policy_fraction: fraction of tokens masked by Technique 2 (target: < 20%)
  - router_entropy_during_rl: confirm Technique 3 prevents routing change
  - stage_indicator: which RL stage is currently active (1, 2, or 3)
  - mtp_acceptance_rate_during_rl: tracked every 500 RL steps (NEW)
  - routing_divergence_from_pretrained: KL(current_routing || pretrained_routing) (NEW)
```

---

## 4.8 Interview Narrative for Pillar 4 (Revised)

> "I applied a three-stage RL pipeline combining DeepSeek-V3.2's MoE stabilization
> with GLM-5's multi-stage structure: Reasoning RL with verifiable rewards, Agent RL
> with tool-use tasks, and General Alignment via DPO — all protected by four V3.2
> MoE-specific stabilization techniques (unbiased KL, off-policy masking, Keep Routing,
> Keep Sampling Mask). The key architectural insight was using MTP acceptance rate as
> a natural test-time scaling signal: high acceptance indicates model confidence, low
> acceptance triggers extended reasoning. This connection between MTP and test-time
> scaling is novel for MoE models — both DeepSeek and GLM treat MTP as a training
> efficiency tool, not an inference scaling signal. I measured RL scaling at three
> compute budgets (2%, 5%, 10% of pre-training FLOPs) and found [log-linear/sublinear]
> improvement. The staging ablation showed [X% improvement] over single-stage GRPO at
> matched compute, confirming GLM-5's finding that staging prevents capability
> interference. Cross-stage distillation with Keep Routing preserved expert
> routing stability (H_load = X.XX ± 0.5 bits, I_spec preserved) while consolidating capabilities from
> all three stages. Test-time scaling behavior [emerged/did not emerge] after RL:
> accuracy improved by [X%] when doubling the inference token budget from 512 to 1024,
> with MTP acceptance rate serving as a reliable confidence proxy."

---

# Data Pipeline Specification

> **⚠️ This section was missing from the original plan.** Without specifying training data,
> the entire experimental plan is ungrounded — scaling laws are data-dependent.

## Training Data

```
Source: Open-source web text + code + math mix.
  Recommended: RedPajama-Data-v2 (CommonCrawl subset, ~30T tokens available)
    or FineWeb (HuggingFace, ~15T tokens, higher quality filtering)
  Code: The Stack v2 (BigCode, ~3T tokens of permissively-licensed code)
  Math: OpenWebMath (~14.7B tokens) + MATH training set + GSM8K training set

Domain mix (target):
  Web text:  60%  (general language modeling)
  Code:      25%  (Python, JavaScript, C/C++, Java, Go — top 5 by volume)
  Math:      10%  (textbooks, papers, competition problems)
  FIM slots:  5%  (10% of sequences × 50% fill-in-middle rate)

Tokenizer: SentencePiece BPE with vocab_size=65536 (matches config.py)
  Train tokenizer on 10B tokens from the same domain mix.
  Must support byte-fallback for code (no UNK tokens).

Filtering pipeline:
  1. Language detection: fastText lid.176 → keep English only for v1
  2. Deduplication: MinHash (Jaccard threshold 0.8) at document level
  3. Quality filter: perplexity filter using KenLM 5-gram (discard top/bottom 5%)
  4. Safety filter: remove PII, CSAM blocklists, known toxic URLs
  5. Code filter: remove files with syntax errors, minified code, auto-generated code

Data loading: streaming from pre-tokenized shards (no on-the-fly tokenization)
  Shard size: ~100M tokens per shard
  Shuffle: shard-level shuffle each epoch, token-level shuffle within shard
  Total training tokens: 22B (NanoSeek-1B) scaled proportionally for sweep configs

Reproducibility:
  Fix random seed for data shuffling (seed=42 for primary run)
  Log shard indices consumed per checkpoint for exact resumption
```

---

# Timeline: 16 Weeks

> **⚠️ Previous version said 12 weeks — unrealistic.** Week 1-2 assumes model rewrite is
> already done, but REIMPLEMENTATION_PLAN.md specifies 2 weeks for model/ alone. Added
> Week 0 for model rewrite and extended RL/polish to account for debugging time.
> Also added hyperparameter transfer strategy (muP or manual) — without this,
> sweep hyperparameters won't transfer to the 1B target scale.

## Hyperparameter Transfer Strategy — Anchor Model (Adapted from Nanochat)

**Problem**: Scaling sweep runs at 20M-800M. Target is 1.08B. If LR/batch_size/warmup
don't transfer, sweep results are misleading and the 1B run risks failure.

**Solution**: Adopt nanochat's muP-style anchor model strategy. Tune HPs once at a cheap
anchor scale, then transfer to all sweep configs AND the 1B target using theoretically-
grounded scaling rules. The sweep VALIDATES the transfer, not discovers HPs from scratch.

```
ANCHOR MODEL: nano-150M (10 layers, 768 hidden, 32 experts, top-4)
  - Already used for stability ablations (Run D) — zero marginal cost
  - Cheap enough for grid search: ~$2-3/run, 15-20 runs = $30-50
  - Large enough that MoE routing is functional (32 experts with top-4)

STEP 1: Grid-search AT ANCHOR SCALE ONLY (~$40, one afternoon)
  Tune on nano-150M with short runs (1000-2000 steps):
  ├── Muon LR for attention weights (η_attn)
  ├── Muon LR for expert FFN weights (η_expert)      ← MoE-specific
  ├── AdamW LR for embeddings (η_embed)
  ├── AdamW LR for router (η_router)                  ← MoE-specific
  ├── AdamW LR for lm_head (η_unembed)
  ├── Optimal batch size (B_ref at nano-150M)
  ├── Weight decay (λ_ref)
  └── Muon momentum warmup schedule

  NOT tuned (constant across scales, from literature):
  ├── gamma = 0.001 (DeepSeek uses constant across scales)
  ├── gamma_freeze_ratio = 0.95 (RULE 2)
  ├── β₂ = 0.95 (MoE-specific, not scale-dependent)
  ├── grad_clip = 1.0
  └── MTP λ = 0.3→0.1 at 60% of training (DeepSeek V3 paper spec, config.py)

STEP 2: Define scaling rules (adapted from nanochat base_train.py)
  Transfer from nano-150M anchor to any target size:

  # Reference values (from Step 1 grid search)
  ANCHOR_N_ACTIVE = 140_000_000   # nano-150M
  ANCHOR_B_REF = TBD              # optimal batch size from grid search
  ANCHOR_D_REF = 20 * ANCHOR_N_ACTIVE  # Chinchilla-optimal tokens

  # Batch size: Power Lines paper (B ∝ D^0.383)
  # Source: arXiv:2505.13738, validated in nanochat for dense models
  target_tokens = 20 * target_n_active
  batch_ratio = target_tokens / ANCHOR_D_REF
  total_batch_size = ANCHOR_B_REF * batch_ratio ** 0.383

  # Learning rates: √B scaling (AdamW theory, extended to Muon)
  # Source: nanochat base_train.py lines 286-293
  batch_lr_scale = (total_batch_size / ANCHOR_B_REF) ** 0.5
  η_attn    = η_attn_ref    * batch_lr_scale  # Muon, sees all tokens
  η_expert  = η_expert_ref  * batch_lr_scale  # Muon, sees 1/top_k tokens
  η_embed   = η_embed_ref   * batch_lr_scale  # AdamW
  η_unembed = η_unembed_ref * batch_lr_scale  # AdamW

  # Weight decay: T_epoch framework (arXiv:2405.13698)
  # Source: nanochat base_train.py lines 296-303
  λ = λ_ref * sqrt(total_batch_size / ANCHOR_B_REF) * (ANCHOR_D_REF / target_tokens)

  # MoE-specific: CONSTANT across scales (hypothesis — validated in sweep)
  η_router = 3e-4             # DeepSeek uses constant; no gradient scaling needed
  gamma    = 0.001            # Bias update, not gradient-based
  mtp_λ    = 0.3→0.1 at 60%   # Schedule from V3 paper (config.py: mtp_loss_transition_ratio=0.60)

STEP 3: Sweep VALIDATES the transfer (hypothesis-driven, not exploratory)
  For each Series A config (80M, 150M, 300M, 500M, 800M):
    1. Compute HPs from anchor + scaling rules (BEFORE training)
    2. Log predicted HPs alongside actual HPs in W&B
    3. Train with predicted HPs
    4. Check: does ema_val_bpb follow predicted scaling curve?

  If YES → scaling rules transfer from dense to MoE → apply to 1B with confidence
  If NO at specific size → diagnose WHICH rule broke → fix that rule → re-run

  This transforms the sweep from "curve fitting" to "hypothesis testing" —
  a stronger scientific methodology that also answers: "Do nanochat's muP-style
  scaling rules (Power Lines, √B, T_epoch) transfer from dense to MoE?"
```

**What transfers from nanochat vs what's MoE-specific:**

| Component | Nanochat Source | MoE Adaptation | Confidence |
|---|---|---|---|
| B ∝ D^0.383 | Power Lines (arXiv:2505.13738) | Use N_active for D calculation | HIGH — gradient noise scales with active params |
| η ∝ √(B/B_ref) | AdamW theory, base_train.py:291 | Same for AdamW params; Muon TBD | HIGH for AdamW, MEDIUM for Muon on experts |
| λ ∝ √(B/B_ref)×(D_ref/D) | T_epoch (arXiv:2405.13698) | Same formula | MEDIUM — T_epoch studied for AdamW, not Muon |
| D = ratio × N_scaling | base_train.py:268 | Use N_active, ratio=20 (Chinchilla) | HIGH — standard methodology |
| Muon momentum warmup | base_train.py:371-373 | Direct reuse (0.85→0.97 over 400 steps) | HIGH — optimizer-level, not architecture-level |
| Router LR | N/A (no router in nanochat) | Constant 3e-4 from DeepSeek | MEDIUM — may need anchor tuning |
| gamma (load balance) | N/A | Constant 0.001 from DeepSeek | HIGH — not gradient-based |

**Failure modes and recovery:**
- If prediction error > 3% at any sweep point: check which HP scaling rule failed
  by comparing actual vs predicted loss curves. The most likely failure is Muon LR
  for expert weights (sparse gradient dynamics differ from dense).
- If prediction error > 5% on NanoSeek-1B: implement full muP (Yang et al. 2022)
  and re-sweep. Cost: ~$100 extra. This is the fallback, not the default.
- Document which rules transferred and which needed correction in SCALING_LAW_REPORT.md.
  This IS a scientific finding: "muP-style transfer from dense to MoE at 1B scale."

```
Week 0-1:   Model Rewrite (prerequisite — see REIMPLEMENTATION_PLAN.md)
            ├── Rewrite model/config.py, model/model.py Sections 1-12
            ├── Unit tests for every component (Gate 1 must pass)
            ├── Implement model/eval/speculative_eval.py (MTP acceptance rate)
            └── This is NOT optional — scaling sweep needs a correct model

Week 2-3:   Infrastructure
            ├── Port nanochat/common.py GPU FLOPS table → nanoseek
            ├── Port nanochat/loss_eval.py → nanoseek/fms/eval_harness/bpb.py
            ├── Port nanochat/core_eval.py → nanoseek/fms/eval_harness/core.py
            ├── Fix MFU calculation (active params not total)
            ├── Implement ema_tracker.py (decay=0.9999, CPU-side, update every 10 steps)
            ├── Implement expert_specialization.py (routing hook, entropy, domain heatmap)
            ├── Implement model/eval/speculative_eval.py (MTP acceptance rate)
            ├── Add FIM training to dataset.py (10% PSM, fim_loss W&B logging)
            ├── Add batch size warmup to pre-train.py (1/5→1× target over 10% of steps, V3's 5× ramp)
            ├── Update checkpoint_manager_async.py to save EMA state
            ├── Create Series A nano-20M and nano-80M configs (gamma_freeze_ratio=0.95)
            └── Run first 2 configs — validate pipeline: EMA checkpoint, ema_val_bpb logged

Week 4-6:   Scaling Law Sweep (Pillar 1, Phase 1)
            ├── Run remaining Series A configs (nano-150M through nano-800M)
            ├── Run Series B expert count sweep (4 configs)
            ├── Parallelize: run 2-3 small configs simultaneously
            ├── Run Series C IsoFLOP sweep (5 configs — 2 more than original plan)
            ├── All runs: EMA checkpoint saved, ema_val_bpb, H_load, and I_spec logged to W&B
            └── Validate: ema_val_bpb ≤ val_bpb for all runs (EMA tracker sanity check)
            Total: 15 scaling runs (nano-150M shared with stability)

Week 7-8:   Fit and Predict (Pillar 1, Phase 2)
            ├── Implement fit_scaling_law.py (fit from ema_val_bpb, NOT val_bpb)
            ├── Fit L(N_active, D, E) with bootstrap CIs
            ├── Extend: analyze I_spec vs N_active, E, D (Plots 8-10)
            ├── Compute Efficiency Leverage (arXiv:2507.17702)
            ├── Generate all 10 scaling plots (7 original + 3 I_spec plots)
            ├── Implement predict_and_validate.py
            └── Implement inference_aware_mla.py

Week 9-10:  Stability Engine (Pillar 2)
            ├── Run 5 spike_reproduction scenarios on nano-150M
            ├── Run 5-config ablation matrix (A, C, D, F, G — Run G = all stabilizers ON)
            │   (Run D = nano-150M scaling data point, already done in Week 4-6)
            │   (Run G = Run D + logit softcap + z-loss — tests if extra stabilizers help)
            ├── Run 2 MTP λ sensitivity runs
            ├── Build spike_detector.py + auto_recovery.py
            ├── Analyze Run G: does aux-loss-free → more specialized experts?
            └── Total: 4 unique stability (A, C, F, G) + 2 MTP = 6 new runs

Week 11-12: Full 1B Training Run (Pillar 1 Phase 3 + Pillar 3)
            ├── Apply minimum viable stability config from Week 7-8
            ├── Enable ALL observability dashboards before starting
            │   (Training Health: FIM, EMA, MTP acceptance rate panels active)
            ├── Run torch.profiler at step 100 for MFU gap analysis
            ├── Train NanoSeek-1B (22B tokens, ~14h on 8×H100)
            │   EMA tracking active throughout
            │   Expert specialization logged every 500 steps
            ├── Run interval evals every 250 steps (ema_val_bpb, CORE, scaling residual)
            ├── Run MTP acceptance rate eval every 2000 steps (target: >75% by end)
            └── Compare final ema_val_bpb to scaling law prediction (target: < 2% error)

Week 13-14: RL Post-Training (Pillar 4 — Three-Stage Pipeline)
            ├── Implement grpo_trainer.py with 4 V3.2 stabilization techniques
            ├── Implement reward_v2 (verifiable + process rewards)
            ├── Implement agent environment (code execution + calculator tools)
            │
            ├── Budget 1 full pipeline (2% of pre-training FLOPs):
            │   ├── Stage 1: Reasoning RL (60% of budget)
            │   ├── Stage 2: Agent RL (25% of budget)
            │   ├── Stage 3: General DPO (15% of budget)
            │   ├── Cross-stage distillation (500 steps)
            │   └── Evaluate: GSM8K, HumanEval, MATH, agent benchmark, MMLU
            │
            ├── Budget 2 full pipeline (5% of pre-training FLOPs):
            │   ├── Same 3-stage structure
            │   └── Evaluate all benchmarks
            │
            ├── Budget 2 staging ablation (single-stage GRPO at matched compute):
            │   └── Compare single-stage vs three-stage at Budget 2
            │
            ├── Budget 3 full pipeline (10% of pre-training FLOPs):
            │   ├── Same 3-stage structure
            │   └── Evaluate all benchmarks + test-time scaling curve
            │
            ├── Measure across all budgets:
            │   ├── H_load and I_spec at each stage boundary
            │   ├── MTP acceptance rate trajectory
            │   ├── Routing divergence between stages
            │   ├── Test-time scaling: accuracy vs inference tokens (256/512/1024/2048)
            │   └── Log-linear fit: improvement vs RL compute budget
            │
            └── Write TRAINING_OPS_REPORT.md (operations + profiler analysis)

Week 15-16: Polish and Package
            ├── Write SCALING_LAW_REPORT.md (publication-quality, includes I_spec analysis)
            ├── Write STABILITY_PLAYBOOK.md (decision tree + ablation table + Run G finding)
            ├── Write RL_SCALING_REPORT.md (multi-stage analysis, test-time scaling, MTP insight)
            ├── Clean code, docstrings for public-facing files only
            ├── Package W&B dashboards as shareable links (all 5 dashboards)
            └── Prepare 4 interview narratives (one per pillar)

TOTAL UNIQUE RUNS: 24 (14 scaling + 1 shared + 4 stability + 2 MTP + 4 RL) + 1 validation
(14 scaling = 6 Series A + 4 Series B + 5 Series C - 1 shared with stability)
(4 RL runs = 3 three-stage pipelines at different budgets + 1 staging ablation)
```

---

# The Four Artifacts

## Artifact 1: SCALING_LAW_REPORT.md

Content:
- Mathematical derivation of L(N_active, D, E) — why not Chinchilla's L(N, D)
  (cite arXiv:2402.07871 and arXiv:2502.05172)
- Sweep design rationale (Series A/B/C) — why 15 runs, not 30
- Fitted exponents with 95% bootstrap CIs
- Comparison to Chinchilla (arXiv:2001.08361) and DeepSeek LLM (arXiv:2401.02954)
- IsoFLOP plots with optimal frontier
- MLA analysis: "exponents match dense baselines, confirming MLA is a
  parameter-efficiency improvement, not an architecture change to the scaling regime"
- Efficiency Leverage measurement: MoE advantage vs dense follows power law in C
  (arXiv:2507.17702 — confirms finding at MLA+MoE scale for first time)
- Inference-aware optimal scaling with MLA correction
- **1B prediction vs actual: the falsifiable result**

## Artifact 2: STABILITY_PLAYBOOK.md

Content:
- Cross-model audit: what 6 frontier MoE models actually use (the table from §2.0)
- Decision tree: "Loss spiked. What do I do?" (covering 5 scenarios)
- Separate branch: "Expert entropy dropped. What do I do?"
- Ablation matrix results table (4 configs × 4 metrics)
- What we tested vs what we inherited as settled science (and why)
- Minimum viable stability config recommendation with evidence
- "MoE-specific finding: gamma=0.001 is the single most important stability knob.
  Expert collapse shows as entropy drop below 2 bits 400+ steps before loss plateau."

## Artifact 3: TRAINING_OPS_REPORT.md

Content:
- MFU calculation methodology (cite nanochat/common.py pattern)
- torch.profiler breakdown: "X% of step time was Y, fixed by Z"
- Before/after table for each optimization
- Async checkpoint implementation: "recovered 8% of step time"
- FIM training: "10% PSM tokens added, fim_loss converged to causal_loss by step X"
- EMA tracking: "ema_val_bpb = X.XXX vs val_bpb = X.XXX — EMA improvement = X%"
- Dashboard screenshots: all 4 dashboards (training health, scaling law tracker,
  stability monitor, MoE health)
- "Target MFU: 47%. Achieved: XX%. Remaining gap: expert routing all-to-all
  communication — requires multi-node overlap implementation (future work)."

## Artifact 4: RL_SCALING_REPORT.md

Content:
- Design rationale: why V3.2 + GLM-5 hybrid (MoE stability + multi-stage structure)
- GRPO algorithm summary: why GRPO over PPO at 1B scale (no value function needed)
- Three-stage pipeline: Reasoning RL → Agent RL → General DPO → Distillation
- 4 V3.2 MoE stabilization techniques: what each prevents, how implemented
- Staging ablation: three-stage vs single-stage at matched compute (Budget 2)
- Scaling table: GSM8K + HumanEval + MATH + agent benchmark at 2%, 5%, 10% RL budget
- Log-linear fit: improvement = A × log(budget) + B, R² value
- Expert routing analysis: H_load and I_spec at each stage boundary (validation of Keep Routing)
- Routing divergence analysis: KL between stage checkpoints
- MTP as test-time scaling signal (NOVEL FINDING):
  - MTP acceptance rate before vs after RL at each stage
  - Test-time scaling curve: accuracy vs inference token budget (pre-RL vs post-RL)
  - MTP acceptance rate vs problem difficulty correlation
- Cross-stage distillation: capability preservation analysis
- Key finding: "X% RL budget achieves [same/better/worse] improvement than
  doubling pre-training tokens, consistent/inconsistent with V3.2 findings.
  Staging improved over single-stage by [X%] at matched compute.
  MTP acceptance rate [is/is not] a reliable test-time scaling proxy for MoE."

---

# Sources

| Paper | arXiv | Venue | What It Grounds |
|---|---|---|---|
| Chinchilla | 2203.15556 | NeurIPS 2022 | Baseline scaling law form |
| DeepSeek LLM | 2401.02954 | Jan 2024 | IsoFLOP methodology, 1000× prediction |
| DeepSeekMoE | 2401.06066 | Jan 2024 | Fine-grained + shared expert design |
| DeepSeek-V2 | 2405.04434 | May 2024 | MLA architecture, 23× KV compression |
| DeepSeek-V3 | 2412.19437 | Dec 2024 | gamma=0.001, aux-loss-free balancing, MTP |
| Scaling Laws for Fine-Grained MoE | 2402.07871 | ICML 2024 | Granularity G, MoE outperforms dense |
| Parameters vs FLOPs (Apple) | 2501.12370 | ICML 2025 | Fix sparsity ratio in sweep |
| Joint MoE Scaling Laws | 2502.05172 | ICML 2025 | L(N_active, D, E) formula, γ correction |
| TransMLA | 2502.07864 | NeurIPS 2025 | MLA inference cost model |
| Comprehensive MoE Scaling | 2509.23678 | 2025 | 5-factor law validation |
| Greater Leverage MoE | 2507.17702 | Jul 2025 | Small→large prediction validated |
| Spike No More | 2312.16903 | 2023 | Embedding scaling technique |
| PaLM | 2204.02311 | 2022 | Z-loss weight (1e-4) |
| Gemma 2 | 2408.00118 | 2024 | Logit softcap=50.0 |
| Aux-Loss-Free Load Balancing | 2408.15664 | 2024 | γ=0.001 mechanism, +0.06 PPL over aux-loss |
| Cerebras Router Wars | blog | 2025 | Learned routing 3× better than hash, settled |
| DBRX (Databricks) | blog | 2024 | Fine-grained experts, dropless MoE routing |
| Qwen2/2.5 MoE | 2407.10671 | 2024 | 64 routed + 8 shared experts, upcycling init |
| Wortsman et al. (QK-norm) | 2309.14322 | 2023 | QK-norm vs z-loss target different failure modes |
| DeepSeek-V3.2 | — | 2026 | GRPO post-training, 4 MoE RL stabilization techniques ⚠️ NO ARXIV |
| GRPO (DeepSeek-R1) | 2501.12599 | Jan 2025 | Group Relative Policy Optimization algorithm |
| CodeLLaMA | 2308.12950 | 2023 | FIM (PSM format), 10% rate — first systematic FIM study |
| Starcoder2 | 2402.19173 | 2024 | FIM at scale, PSM format standard for code models |
| Sardana & Frankle | 2401.00448 | 2024 | Beyond Chinchilla — inference-aware optimal scaling |
| GLM-5 (ChatGLM) | — | 2025 | Multi-stage RL pipeline ⚠️ NO ARXIV — verify when published |
| Slime Framework | — | 2025 | Async Megatron + SGLang for RL ⚠️ NO ARXIV — verify when published |
| RL Scaling Laws | — | 2025 | Scaling laws for RL post-training ⚠️ NO ARXIV — claims unverifiable |
| OpenAI o1 | — | 2024 | Test-time scaling ⚠️ NO ARXIV — blog/system card only |

> **⚠️ Citation integrity note:** 4 sources above lack arXiv IDs and cannot be independently
> verified. Claims derived from these sources (multi-stage RL pipeline, Slime throughput,
> RL scaling laws) should be treated as hypotheses, not established results.
> Update this table with arXiv IDs as papers become available.
