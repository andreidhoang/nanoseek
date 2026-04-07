# NanoSeek Pretraining Masterplan
## MoE Training Dynamics at Nano Scale

**Version**: 2026-04-06-v8  
**Status**: Revised after 5-agent deep review (information theory, muP theory, experimental design, MoE scale validity, training infrastructure). Corrected metric limitations, statistical power, and theoretical gaps. All novelty claims verified against published literature as of April 2026.  
**Thesis**: NanoSeek's contribution is **I_spec as the first online MoE training diagnostic for expert specialization**, combined with a controlled aux-loss comparison and a replication-with-extension of muP-MoE transfer under DeepSeek V3-style architecture with MuonAdamW.

---

## Goal Statement: Why This Project Exists

### The Core Bet

The frontier AI industry trains MoE models with hundreds of billions of parameters (DeepSeek V3 671B, Qwen3.5 397B, DBRX 132B). These models use expert routing — a discrete, combinatorial mechanism whose training dynamics are poorly instrumented. Teams monitor loss curves, gradient norms, and load balance entropy. But no one systematically monitors **whether experts are learning meaningful specializations during training**, despite this being the entire purpose of the MoE architecture.

NanoSeek exists to build and validate **I_spec** — expert-domain mutual information tracked as an online training diagnostic — and to demonstrate that it reveals MoE training dynamics invisible to loss curves. This is a measurement tool, not a model. Its value is that it can be deployed at any scale.

### How Diagnostics Are Actually Used (Operational Model)

Frontier teams do **not** stop million-dollar production runs to tune hyperparameters based on diagnostic signals. Production runs have optimizer state, learning rate schedules, and data mixtures locked before launch. Mid-run interventions are reserved for emergencies: loss spikes, NaN, hardware failure. DeepSeek V3 reported ~10 checkpoint rollbacks during training — all for stability failures, not routing tuning.

The actual workflow is:

```
Cheap ablations (select best config) → Lock everything → Expensive production run (monitor only)
```

**I_spec's primary value is in the left box, not the right.** It is a configuration selection metric that helps you choose between routing strategies, aux-loss coefficients, and optimizer settings during cheap ablation runs — where you can afford to run 6+ configurations and compare. At production scale, I_spec is logged for post-hoc analysis and scientific understanding, but not for mid-run intervention.

This distinction matters for honest framing. I_spec is not an "early warning system that saves your training run." It is a **selection criterion that prevents you from starting the wrong training run** — which is more valuable, because the cost of a bad configuration is the entire production budget, not just the steps after a warning fires.

### Why Small Scale Produces Transferable Methodology

NanoSeek does not claim that 410M-parameter results predict 671B-parameter loss. It claims something more specific and more defensible:

1. **I_spec is a scale-invariant metric.** Mutual information I(E; D) measures statistical dependence between expert routing and input domain. This quantity is defined by the joint distribution of routing decisions and domains, not by parameter count. If I_spec is informative at 410M, the same measurement can be taken at 671B. We validate that it is informative; the user deploys it at their scale.

2. **muP-MoE transfer has been theoretically derived and empirically validated by three recent papers** (Malasnicki et al. Aug 2025, Jiang et al. Jan 2026, HyperP Mar 2026). NanoSeek does not claim to pioneer this. Instead, we replicate muP-MoE transfer under a specific combination — DeepSeek V3 architecture, MuonAdamW optimizer, aux-loss-free routing — that no existing paper covers. This is replication-with-extension, honestly labeled.

3. **Routing strategy comparisons (aux-loss vs aux-loss-free) are mechanism studies.** If aux-loss gradient interference suppresses expert specialization (measurable via I_spec), that mechanism operates identically at any parameter count. We identify the mechanism at small scale; it applies at large scale because the gradient math is the same.

### What muP Actually Guarantees (Precise Statement)

muP (Yang et al., Tensor Programs V, 2022) is a theorem about the infinite-width limit of neural networks. Under specific parameterization rules:
- In the infinite-width limit, optimal learning rates and feature learning dynamics are width-independent
- At finite width, this is an **approximation** whose quality improves with width
- The theorem applies under **fixed depth** and standard dense layer structure
- It says nothing directly about discrete routing decisions in MoE

Three recent papers extend muP theory to MoE:
- Malasnicki et al. (arXiv:2508.09752): Classifies expert weights as "hidden" and router weights as "output" within TP5. Finds LR transfer works across width but **breaks across top-k values and expert granularity**.
- Jiang et al. (arXiv:2601.20205): Uses DMFT to derive MoE parameterization. Validates from 51M to 2B+.
- HyperP (arXiv:2603.28743): Extends to Muon optimizer with SqrtGate. Transfers across width, depth, tokens, and MoE granularity.

NanoSeek builds on these results. It does not claim priority.

### Why MoE Diagnostics Are the Right Thing to Study in 2026

| Property | Current State | NanoSeek Contribution |
|---|---|---|
| muP for MoE | Published (3 papers) | Replicate under DeepSeek V3 + MuonAdamW + aux-loss-free |
| Expert specialization measurement | Post-hoc heatmaps (DeepSeek V3) | **First online training diagnostic (I_spec vs step)** |
| Aux-loss vs aux-loss-free mechanism | "Aux-loss-free is better" (DeepSeek V3) | **First controlled comparison with specialization metrics as primary outcome** |
| MoE routing quality signal | H_load (load balance entropy) | **I_spec as complementary signal: specialization, not just balance** |

The novel contributions are in the last column. The muP component is honest replication. The I_spec and aux-loss components are genuinely new.

### What This Project Is NOT

- It is not a frontier model. 1.08B active parameters is not competitive.
- It is not a benchmark chaser. We do not report MMLU or HumanEval.
- It is not a scaling law paper. We do not fit scaling laws from 3 points.
- It is not the first muP-MoE paper. Three exist. We replicate and extend.
- It is not a reproduction of DeepSeek V3. We measure quantities DeepSeek did not report.

### What This Project IS

A **measurement program** that produces:
1. The first systematic I_spec training curves for MoE (expert-domain MI tracked every 250 steps)
2. The first controlled aux-loss vs aux-loss-free comparison with specialization metrics as the primary outcome variable
3. A muP-MoE replication under DeepSeek V3 + MuonAdamW + aux-loss-free (novel combination)
4. A replayable, restartable, profiler-backed execution

### Contribution to Fast-Moving 2026 Frontier

Architecture winners change fast (Transformers → SSM hype → hybrid consensus in 18 months). But **training diagnostics** compound:

- H_load (load balance entropy) was introduced in Switch Transformer (2022) and is still the primary MoE monitoring tool in 2026
- Gradient norm monitoring, introduced decades ago, is still used in every training run
- If I_spec proves informative, it joins this permanent toolkit

We contribute a measurement tool, not an architecture. Measurement tools are durable because they are architecture-agnostic. I_spec works for any MoE variant — DeepSeek, Mixtral, Qwen, or whatever comes next.

---

## 0. Why the Previous Plans Were Killed

### v4 → v5: Hybrid Attention Study Killed
The 3-arm hybrid attention study (MLA vs KDA+MLA vs GatedDeltaNet+MLA) asked a question answered by Kimi Linear, Qwen3.5, and OLMo Hybrid. KDA was stubbed; GatedDeltaNet didn't exist in the codebase.

### v5 → v6: Novelty Claims Corrected
The v5 plan claimed "first clean empirical test of muP for MoE." A thorough literature review found 3 published papers (Aug 2025, Jan 2026, Mar 2026) that address this directly. The plan was rewritten to honestly position the muP component as replication-with-extension and to elevate the genuinely novel components (I_spec diagnostic, aux-loss specialization comparison).

**This is how science works: you check the literature before you claim novelty, and you correct your claims when the literature has moved.**

### v6 → v7: I_spec Operational Model Corrected
The v6 plan framed I_spec as a "real-time diagnostic" and "early warning system" for production-scale training, implying teams would intervene mid-run based on I_spec signals. Analysis of actual frontier workflows (DeepSeek V3 rollback reports, Llama 3 ablation methodology) revealed that diagnostic metrics are used for **configuration selection during cheap ablation campaigns**, not for mid-run intervention. Production runs have locked HPs and are only interrupted for stability emergencies. The plan was rewritten to honestly frame I_spec as an ablation-phase selection metric.

### v7 → v8: 5-Agent Deep Review
Five specialized research agents (information theory, muP scaling theory, experimental methodology, MoE scale validity, training infrastructure) stress-tested every aspect of the plan from first principles. Key corrections:
1. **I_spec metric limitations**: MI measures routing preference, not functional specialization. MI ceiling at H(D)=ln(5)=1.609 nats must be normalized. Added I_spec/H(D), bootstrap CIs, chi-squared significance tests.
2. **Statistical power**: n=3 per group has power ~0.15 for moderate effects. H2 downgraded from confirmatory to exploratory. Effect sizes + CIs replace binary significance.
3. **muP theoretical gaps**: No SqrtGate = most likely transfer failure point. Batch scaling for Muon unvalidated. Added SqrtGate fallback and batch scaling control run.
4. **Step count risk**: 8000 steps may not be enough for specialization convergence. Added 1 full-budget convergence pilot ($35).
5. **Infrastructure gaps**: Missing RNG state in checkpoints (replay proof impossible). Domain eval dataset doesn't exist. Cluster-based I_spec too expensive at 1.08B.
6. **Scale validity confirmed**: Expert intermediate is 480 (not 120). Each expert sees 28x Chinchilla-optimal tokens. Bias updates stable at batch 128. MLA doesn't affect routing.
Budget updated from $600 → $700 to accommodate additional controls.

---

## I. Problem Statement

NanoSeek has a complete, tested MoE + MLA + MTP + FIM implementation with muP scaling, MuonAdamW optimizer, aux-loss-free routing, and expert specialization instrumentation at two scales (410M active, 1.08B active).

**Objective**:
- Introduce I_spec as an online MoE training diagnostic and demonstrate it reveals dynamics invisible to loss curves
- Produce the first controlled comparison of aux-loss vs aux-loss-free routing with specialization metrics as the primary outcome
- Validate muP-MoE transfer under a DeepSeek V3 + MuonAdamW configuration not covered by existing papers
- Demonstrate scientific rigor, systems execution, and honest interpretation

**Constraints**:
| Resource | Constraint | Planning implication |
|---|---|---|
| Scale | 410M anchor (1.95B total), 1.08B validation (4.75B total) | MoE dynamics are real at these scales; absolute quality is not the point |
| Hardware | Single-node multi-GPU (4x A6000 or 8x H100) | Enough for MoE with 64 experts |
| Sequence length | 4K training | Standard; no need for PP or CP |
| Budget | ~$700 total | ~15 ablation runs + convergence pilot + 1-2 graduation runs + controls |
| Timeline | 4-6 weeks | Executable because core infrastructure exists |

**Success criteria**:
1. I_spec training curves that reveal at least one MoE dynamic invisible to loss curves (e.g., specialization onset, collapse prediction, arm differentiation before loss diverges) — **exploratory, H1**
2. Descriptive I_spec difference between aux-loss and aux-loss-free routing, reported with effect sizes and confidence intervals (n=3 per group cannot support confirmatory claims) — **exploratory, H2**
3. muP-predicted LR at 1.08B within 1.5x of grid-optimal AND better than val-naive — **confirmatory, pre-specified criterion**
4. Replayable, restartable, profiler-backed execution — **engineering requirement**

**Assumptions**:
- [ASSUMPTION-1] MoE dynamics at 410M-1.08B active parameters are scientifically real. **v8 evidence**: Expert intermediate is 480 hidden units (SwiGLU), ~1.8M params per expert. Each expert sees ~28x Chinchilla-optimal tokens for its size. OLMoE demonstrated specialization at comparable scale (1.3B total, 1024 expert intermediate). Bias update stability confirmed: 65K activations per expert per step at batch 128. **Risks**: Shared expert dominance at small scale (monitored via norm ratio). Group routing artifacts (monitored via per-group I_spec).
- [ASSUMPTION-2] I_spec (expert-domain MI computed against labeled domains) is a meaningful training diagnostic that adds information beyond H_load. **v8 limitations**: MI measures routing preference, not functional specialization. MI ceiling at ln(5)=1.609 nats. Mitigated by normalization, bootstrap CIs, chi-squared tests, and domain label validation.
- [ASSUMPTION-3] The existing muP-MoE theory (Malasnicki 2025, Jiang 2026, HyperP 2026) extends to DeepSeek V3 + MuonAdamW + aux-loss-free routing. **v8 gaps**: No SqrtGate (most likely failure point), Muon batch scaling unvalidated, depth-to-width ratio changes 2.67x. Mitigated by SqrtGate fallback and batch scaling control run.

**Out of scope**:
- Hybrid attention comparisons (answered)
- Scaling-law fits from fewer than ~10 points (underdetermined)
- Production-safety claims
- Post-training algorithm comparisons

---

## II. Evidence Base

### Published and Settled
- Hybrid attention beats pure attention (Kimi, Qwen, OLMo Hybrid — settled)
- MLA is superior to GQA for KV compression (DeepSeek V2/V3 — settled)
- muP extends to MoE (Malasnicki 2025, Jiang 2026, HyperP 2026 — published)
- Aux-loss-free routing is better on final quality (DeepSeek V3 — published)

### Open and Worth Paying For
- **I_spec as training diagnostic**: No published work tracks expert-domain MI over training steps as an online diagnostic. Post-hoc heatmaps exist (DeepSeek V3). Orthogonality/variance metrics exist (Advancing Expert Specialization, 2505.22323). But real-time I_spec curves during training? **Genuinely novel.**
- **Controlled specialization comparison**: Papers compare aux-loss vs aux-loss-free on perplexity. Papers propose new specialization losses vs aux-loss baselines. But a clean head-to-head with MI-based specialization as primary outcome? **Not published.**
- **muP-MoE with MuonAdamW + DeepSeek architecture**: HyperP covers Muon but with SqrtGate (not standard gate). Malasnicki and Jiang use standard Adam. The specific combination of MuonAdamW + bias-based routing + DeepSeek V3 MLA is not covered. **Extension of published work.**

---

## III. Core Questions

Ordered by novelty and value:

1. **I_spec Diagnostic**: Does expert-domain mutual information, tracked during training, reveal MoE dynamics invisible to loss curves and H_load? (Novel)
2. **Specialization Mechanism**: Does aux-loss-free routing produce measurably higher expert specialization than aux-loss routing? Does the I_spec gap predict quality gaps? (Novel comparison)
3. **muP Replication**: Does muP-MoE HP transfer work for DeepSeek V3 architecture with MuonAdamW and aux-loss-free routing? (Replication-with-extension)

---

## IV. Program Overview

| Workstream | Type | Novelty |
|---|---|---|
| 1. I_spec as training diagnostic | Primary, novel | First online tracking of expert-domain MI |
| 2. Aux-loss specialization comparison | Primary, novel comparison | First controlled head-to-head with MI metrics |
| 3. muP-MoE transfer replication | Secondary, extension | Validates published theory under new optimizer/architecture |
| 4. Systems reliability harness | Must-have | Frontier engineering discipline |

Scaling law fitting is **not a workstream**. If we have 3 scale points, we plot L vs N_active in an appendix figure with an explicit note that 3 points cannot distinguish functional forms.

---

## V. Workstream 1: I_spec as Online Training Diagnostic

### Research Question
Does I_spec (expert-domain mutual information), computed against labeled domains and tracked every 250 steps, reveal MoE training dynamics that are invisible to loss curves and H_load?

### Why This Is Novel
- DeepSeek V3: published expert-domain heatmaps (post-hoc, qualitative, not tracked during training)
- OLMoE: published expert utilization analysis (post-hoc)
- Switch Transformer: introduced H_load (measures balance, not specialization)
- Mod-Squad: used I(E; Y) as a training *loss*, not a passive diagnostic
- **No one has published I_spec tracked over training steps as an online diagnostic.**

### Critical Design Detail: Domain Definition
I_spec = I(E; D) where D is the input domain. We use **labeled domains** from the training data:

| Domain | Source | Label |
|---|---|---|
| code | Programming files | Heuristic: file extension + syntax markers |
| math | Mathematical text | Heuristic: equation density + math vocabulary |
| science | Scientific papers | Heuristic: citation patterns + technical vocabulary |
| web | General web text | Default category |
| books | Long-form prose | Heuristic: paragraph structure + length |

These are the same 5 domains used in `eval/domain_bpb.py`. I_spec is computed from the joint distribution P(expert_id, domain_label) over a fixed evaluation set with balanced domain representation.

**Important**: The current implementation in `information_metrics.py` uses k-means clusters on hidden states, NOT labeled domains. This must be fixed before launch. The k-means version may be reported as a secondary metric ("representation-based I_spec") but the primary metric uses labeled domains.

### Metric Limitations (Honest Accounting)

**What I_spec measures**: Statistical dependence between expert routing decisions and input domain labels. If experts route differently for different domains, I_spec is high. This is **routing preference**, not **functional specialization**.

**What I_spec does NOT measure**:
- **Functional specialization**: An expert routed preferentially to "code" may compute the same function as a "math" expert. I_spec measures *who gets which tokens*, not *what computation each expert performs*. Distinguishing routing from functional specialization would require mechanistic interpretability tools (e.g., SAEs on expert outputs).
- **Within-domain structure**: If "code" has Python-specialist and C++-specialist experts, I_spec with 5 coarse domains cannot see this. The k-means secondary metric partially addresses this.
- **Expert co-activation patterns**: I_spec is computed over marginal P(expert, domain), not over which *combinations* of top-8 experts fire together. Two systems with identical marginal distributions but very different co-activation structure would have the same I_spec.
- **Quality of specialization**: An expert that specializes on whitespace tokens (degenerate) looks the same as one specializing on semantic code structure (useful). I_spec cannot distinguish useful from degenerate specialization.

**MI ceiling**: With 5 domains, I_spec ≤ H(D) = ln(5) ≈ 1.609 nats, regardless of expert count or specialization quality. This ceiling is identical across all NanoSeek scales (same 5 domains, same 64 experts), so cross-scale comparisons are valid. But raw I_spec values are uninterpretable without context — **always report I_spec/H(D)** as the normalized [0,1] "fraction of domain information captured."

**Finite-sample bias**: The Miller-Madow bias for MI from a contingency table is approximately (|E_eff|-1)(|D|-1)/(2N) nats. With 64 experts, 5 domains, and N=80,000 observations (10K tokens × top-8): bias ≈ 0.0016 nats. This is negligible (<0.1% of any scientifically interesting I_spec value). Report the bias estimate alongside I_spec for transparency.

**Domain label noise**: Heuristic domain labels have estimated 5-20% error rate (science papers with math equations, code documentation that is mostly prose). Label noise systematically *deflates* I_spec, making the metric conservative — harder to show specialization, not easier. This is acceptable bias direction. Phase 0 includes domain label validation (see Section IX).

**These limitations apply to ANY routing-based specialization metric, not just MI.** I_spec is the natural information-theoretic choice, and its limitations are well-understood and bounded.

### Instrumentation Schedule

**Primary metrics** (every 250 steps on fixed 10K-token eval set):
- **I_spec (labeled-domain)**: Raw MI in nats, per-layer
- **I_spec/H(D) normalized**: Fraction of domain information captured, [0,1] range — the primary interpretable number
- **I_spec_max**: Maximum I_spec across layers (avoids dilution from low-specialization early layers)
- **Chi-squared p-value**: `scipy.stats.chi2_contingency(joint)` — is the expert-domain association statistically significant?
- **Bootstrap 95% CI**: 100 resamples of the contingency table per layer — error bars on every I_spec measurement
- **Miller-Madow bias estimate**: (|E_eff|-1)(|D|-1)/(2N) per layer — verify bias is negligible
- **Per-group I_spec**: I_spec computed within each of the 8 routing groups — distinguish genuine specialization from group-routing artifacts
- **Shared/routed output norm ratio**: ||shared_expert_output|| / ||routed_expert_output|| — detect shared expert dominance
- **Effective expert support per domain**: n_eff_d = exp(H(E|D=d)) — flag domains with <5 effective experts

**Standard metrics** (every 250 steps):
- H_load: already logged every step (also per-layer at eval)
- Dead expert count
- Per-domain BPB
- Routing entropy per layer
- MTP acceptance rate

**Secondary metrics** (every 2000 steps, not every 250 — too expensive at 1.08B):
- I_spec (cluster-based / k-means): Limited to 10K-token sample, not full val_loader

### Analyses
1. **I_spec trajectory**: Plot I_spec vs step. When does specialization onset? Is it monotonic?
2. **I_spec vs H_load**: Do they provide independent information? Can I_spec drop while H_load stays high? (This would mean experts are balanced but not specialized — a known failure mode.)
3. **I_spec layer-wise**: Does specialization increase with layer depth? (Qualitatively suggested by DeepSeek V3 heatmaps, never quantified.)
4. **I_spec leading indicator**: Does I_spec change direction before loss does? (If yes, this makes I_spec a faster configuration discriminator during ablations — you can distinguish good from bad routing strategies in fewer steps. Note: this is about shortening ablation runs, not about intervening in production runs.)
5. **I_spec at scale**: Do patterns persist from 410M to 1.08B?

### Hypothesis H1: I_spec Informativeness
**Statement**: I_spec training curves reveal at least one MoE dynamic that is not visible in loss curves or H_load.

**Status**: Exploratory (this is a measurement study — we are establishing whether the metric is informative, not testing a directional prediction).

**Evidence threshold**: At least one of the following is observed:
- I_spec differentiates between routing arms (R0 vs R1) at a timepoint where loss curves do not
- I_spec predicts expert collapse (dead expert onset) before H_load drops below threshold
- I_spec layer-wise pattern reveals qualitative structure (e.g., deeper = more specialized) that is not visible in layer-wise H_load

---

## VI. Workstream 2: Aux-Loss Specialization Comparison

### Research Question
Does aux-loss-free routing (DeepSeek V3 style) produce measurably higher expert specialization (I_spec) than traditional aux-loss routing, under controlled conditions?

### Arms

| Arm | Routing Strategy | Details |
|---|---|---|
| R0 | Aux-loss-free (bias-based) | NanoSeek default: dynamic bias update, seq_aux_loss_alpha=0.0 (`--no-seq-aux` flag) |
| R1 | Traditional aux-loss | seq_aux_loss_alpha=0.01 (`--aux-loss-type classic` flag) |

### Aux-Loss Coefficient Justification
The R1 coefficient (alpha=0.01) requires calibration, not literature citation. Before locking:
1. Run a 500-step pilot at 410M with alpha values [0.001, 0.01, 0.1]
2. Select the alpha that produces H_load > 4.0 bits (healthy balance) without dominating the total loss gradient
3. Log gradient norms for router parameters in both arms to verify comparable optimization dynamics
4. Lock the chosen alpha in the pre-registration before Phase 2

### Fixed Parameters
| Parameter | Value |
|---|---|
| Scale | 410M active (ablation config) |
| Layers | 16 |
| Experts | 64 routed + 2 shared |
| Top-k | 8 |
| Steps | 8,000 (+ 1 convergence pilot to full budget, see below) |
| Optimizer | MuonAdamW with Phase 1 best HPs |
| Seeds | [42, 137, 2026] |
| Sequence length | 4096 |
| EMA | enabled, decay=0.9999 |

### Convergence Pilot (CRITICAL — added v8)
**Problem**: 8000 steps = 4.2B tokens = 6.4% of full ablation budget. Expert specialization timing at small scale is unknown. If specialization develops after 8000 steps, the study reports "no difference" when the real answer is "not enough training."

**Solution**: Before Phase 2 proper, run **1 seed of R0 to full ablation budget** (~125K steps, ~$35). Plot I_spec vs step. If I_spec has converged by step 6000-8000, the 8000-step budget is adequate. If I_spec is still monotonically increasing at step 8000, extend Phase 2 runs accordingly.

This resolves the "null result ambiguity" — a null result at 8000 steps with evidence that I_spec has converged means "no difference." A null result at 8000 steps with I_spec still rising means "not enough training."

### Co-Primary Metrics
1. **I_spec (labeled-domain)**: Expert-domain MI using 5 labeled domains
2. **Dead expert count**: Number of experts receiving < 1% of expected load
3. **ema_val_bpb**: Final quality

Dead experts are co-primary because if R0 has 20 dead experts and R1 has 0, the I_spec comparison is confounded — you are measuring specialization over different effective expert counts.

### Hypothesis H2: Specialization Advantage
**Statement**: Aux-loss-free routing (R0) achieves higher final I_spec than aux-loss routing (R1) at 410M scale, with comparable or lower dead expert count.

**Status**: **Exploratory** (downgraded from confirmatory in v8). Rationale: with n=3 seeds per group, a two-sample t-test has power ~0.15 for a moderate effect (Cohen's d=1.0). The study is almost guaranteed to fail to detect real moderate effects. Binary significance claims are inappropriate at this sample size.

**What we report instead of p-values**:
- Effect size (Cohen's d) with 95% confidence interval
- Raw I_spec/H(D) trajectories for all 6 runs, overlaid
- Bayesian credible intervals (valid at any sample size)
- Descriptive comparison: "R0 I_spec was X±Y, R1 was A±B, difference = D [95% CI: L, U]"

**Threshold (for descriptive framing only)**: Determined after pilot measurement. Run R0 for 500 steps, measure I_spec, compute standard deviation. The 1.5x pilot SD is a descriptive benchmark, not a confirmatory test threshold.

**Additional metrics to report**:
- H(E|D): conditional entropy (how much uncertainty about expert remains after knowing domain)
- Per-layer routing entropy
- Router gradient norms for **ALL parameter groups** (not just router — aux loss gradient propagates through entire computation graph)
- Expert-domain affinity heatmaps at steps 2000, 4000, 6000, 8000
- Optimizer state statistics: mean/variance of Adam second moments for router vs dense (to check for optimization confound)

### Pre-Specified Sensitivity Analyses (added v8)
1. **I_spec-quality correlation**: Report Pearson r between final I_spec and final ema_val_bpb across all 6 runs. If r < -0.3 (higher specialization → worse loss), flag the "specialization is good" assumption for scrutiny.
2. **Dead expert confound**: If dead expert counts differ by >5 between R0 and R1, report I_spec both raw and normalized by effective expert count (64 minus dead experts). The comparison is confounded when arms have different effective architectures.
3. **Domain robustness**: Compute I_spec with both 5 domains and a coarser 3-domain scheme (code / STEM / general) as a sensitivity check.

### Scale Validation
Run winning routing arm at 1.08B with **2 seeds**, **4000 steps** (extended from 1000 in v8 — 1000 steps is deep in warmup/early-training where I_spec patterns have not developed). Check:
- Does I_spec layer-wise pattern persist?
- Does final I_spec at 1.08B exceed 410M I_spec?
- Is the I_spec trajectory shape similar (onset timing, growth rate)?

---

## VII. Workstream 3: muP-MoE Transfer Replication

### Research Question
Does muP HP transfer work for DeepSeek V3-style MoE architecture with MuonAdamW optimizer and aux-loss-free routing?

### Positioning
This is **replication-with-extension** of published results:
- Malasnicki et al. (Aug 2025): validated muP-MoE with standard Adam
- Jiang et al. (Jan 2026): validated with DMFT-based parameterization
- HyperP (Mar 2026): validated with Muon + SqrtGate

NanoSeek tests the specific combination: **MuonAdamW + standard gate (no SqrtGate) + aux-loss-free routing + MLA**. If transfer works, this extends the published results. If it fails, we identify which component breaks.

### Theoretical Gaps (Honest Accounting — added v8)

**1. No SqrtGate = most likely failure point.** HyperP's muP derivation requires SqrtGate (router logits divided by √d_model) to normalize the router's output scale across widths. Without SqrtGate, sigmoid inputs scale as O(√d_model) with width, causing increased saturation at wider models. This breaks the width-invariance that muP requires for the router. **SqrtGate is planned as a fallback** (see Phase 1D below).

**2. Batch scaling for Muon is unvalidated.** The √(B/B_ref) LR scaling rule was derived for SGD (Goyal et al. 2017) and extended to Adam. Muon uses Newton-Schulz orthogonalization of gradients — a fundamentally different update mechanism where batch noise interacts differently with the update direction. No published work validates batch scaling for Muon. **A batch scaling control run is included** (see Phase 1 below).

**3. Weight decay scaling for Muon is approximate.** The T_epoch framework (arXiv:2405.13698) derives WD scaling for AdamW. The code applies `wd_scaled = wd × √(B/B_ref) × (D_ref/D)` but omits the width factor `(w_ref/w)` for Muon groups. Standard muP does not width-scale WD, so the code may be accidentally correct, but the theoretical justification is incomplete for orthogonalized optimizers.

**4. Depth-to-width ratio changes across scales.** muP assumes fixed depth. NanoSeek fixes depth at 16 layers, but the depth-to-width ratio changes from 0.021 (768h) to 0.008 (2048h) — a 2.67x change. At 16 layers this effect is likely small, but it is an uncontrolled variable. Depth-muP extensions (Yang et al. 2024, Tensor Programs VI) exist but are not implemented here.

### Design

#### Phase 1A: HP Grid at 175M Anchor (3-point transfer)
| Run | matrix_lr | embedding_lr | Steps | Seed |
|---|---|---|---|---|
| hp-175m-1 through hp-175m-8 | [0.003, 0.005, 0.01, 0.02, 0.03, 0.04] × [0.2, 0.5] | 1000 | 42 |

**Why 175M anchor**: Two-point transfer (410M → 1.08B) cannot distinguish "muP works" from "LR is insensitive." Adding 175M gives a 3-point transfer curve (175M → 410M → 1.08B) for ~$12 extra.

**Why 1000 steps**: 500 steps is marginal (warmup may consume 100+ steps). 1000 steps at 175M costs ~$1.50/run.

#### Phase 1B: HP Grid at 410M
Same 8-point grid, 1000 steps, seed 42. Select best (matrix_lr*, embedding_lr*).

#### Phase 1C: Router LR Isolation at 410M
| Run | Router LR | Other HPs |
|---|---|---|
| router-matrix | matrix_lr* (default: same as dense layers) | Best from 1B |
| router-emb | embedding_lr* (treat router as output head) | Best from 1B |
| router-0.5x | 0.5 × matrix_lr* | Best from 1B |
| router-2x | 2.0 × matrix_lr* | Best from 1B |

**Why router-emb arm**: Under muP, the router is a projection from d_model to n_experts. Its structure is analogous to the output head (d_model → vocab), which under muP gets the embedding LR rule, not the matrix LR rule. The plan must test both hypotheses.

#### Phase 1D: muP Transfer Predictions
```
matrix_lr_410m = matrix_lr_175m* × (768 / 1280)    # width ratio
matrix_lr_1b   = matrix_lr_175m* × (768 / 2048)    # width ratio
```
Compare predicted vs actual grid-optimal at 410M and 1.08B.

**SqrtGate fallback** (added v8): If Phase 1D transfer fails by >1.5x at 410M OR 1.08B, re-run the failing scale with SqrtGate enabled (router logits /= √d_model). If SqrtGate fixes transfer, report: "muP-MoE transfer requires SqrtGate for standard sigmoid routing, consistent with HyperP." This is a cheap test (~$5-10) that distinguishes "muP doesn't work for MoE" from "muP requires SqrtGate."

#### Phase 1E: Batch Scaling Control (added v8)
Run 1 additional config at 410M: best anchor HPs (175M) applied **without** √(B/B_ref) batch scaling correction. Compare ema_val_bpb against the muP-predicted (with batch scaling) run. If the un-scaled version performs equally well, batch scaling adds no value for Muon and should be dropped. Cost: ~$5.

**Reference scale note**: The masterplan tunes HPs at 175M anchor (w=768, batch=64). When running Phase 1A, pass `--mup-ref-width=768 --mup-ref-batch-tokens=262144` explicitly. The code defaults (`--mup-ref-width=1280`) assume ablation-as-reference, which is incorrect for the 3-point transfer design.

#### Phase 3: Validation at 1.08B
| Run | Config | Steps | Seeds |
|---|---|---|---|
| val-mup | muP-predicted HPs from 175M | 1000 | 42, 137 |
| val-grid-1 | 0.5x muP-predicted matrix_lr | 1000 | 42 |
| val-grid-2 | 2.0x muP-predicted matrix_lr | 1000 | 42 |
| val-naive | Best 175M HPs without muP adjustment (control) | 1000 | 42 |

### Success Criteria
**Criterion 1 (within-1.5x)**: muP-predicted LR within 1.5x of grid-optimal at both 410M and 1.08B.

`|log2(predicted / optimal)| <= 0.585`

**Why 1.5x not 2x**: Published muP papers demonstrate transfer within ~1.2-1.5x. A 2x threshold could pass trivially on a flat loss landscape. 1.5x is the standard in the literature.

**Criterion 2 (better-than-naive, added v8)**: muP-predicted ema_val_bpb must be **lower** than val-naive ema_val_bpb (best 175M HPs without muP adjustment). If val-naive also passes the 1.5x criterion, muP adds no value over simply reusing small-scale HPs — the loss landscape is flat and muP is unnecessary at these scales.

**Falsification**: muP-predicted LR at 1.08B produces `ema_val_bpb` more than 0.03 bpb worse than grid-optimal (relative threshold: state expected bpb range at launch).

### What We Report That's New
- Whether transfer works with MuonAdamW (not covered by Malasnicki or Jiang who use Adam)
- Whether transfer works with aux-loss-free routing (not tested in any published paper)
- Whether the router follows the matrix LR rule or the embedding LR rule under muP

---

## VIII. Workstream 4: Systems Reliability Harness

Unchanged from v5. This is correct.

### Systems Contract
Freeze before step 0:
- GPU SKU and count
- Interconnect class
- World size
- Parallelism mode (DDP or FSDP2)
- Precision policy (BF16 baseline; FP8 only on H100+ after BF16 parity)
- Activation checkpointing scope
- Fused-kernel policy
- `torch.compile` policy
- Dataloader worker count, prefetch, and packing implementation
- Benchmarking warmup window and measurement window

### Checkpoint Contract
Every valid checkpoint: model state, optimizer state, EMA state, scheduler state, CPU RNG, CUDA RNG, sampler/dataloader cursor, grad-accum microstep, exact consumed-token count.

### Must-Have Proofs
1. Exact replay with token-order continuity
2. Atomic checkpoint writes
3. Fault-injection restart-equivalence
4. Fixed-harness inference measurements
5. Profiler-backed bottleneck attribution

---

## IX. Execution Sequence

### Phase 0: Infrastructure Gate (~3 days)

**Code fixes (must complete first)**:
- **Fix RNG state in checkpoints**: Add `torch.get_rng_state()`, `torch.cuda.get_rng_state_all()`, `random.getstate()`, `numpy.random.get_state()` to checkpoint save AND resume paths (~10 lines). Without this, replay proof is impossible.
- **Fix I_spec to use labeled domains** (not k-means clusters) as primary metric
- **Add I_spec/H(D) normalization** to output dict (1 line per layer)
- **Add chi-squared independence test** to I_spec output (`scipy.stats.chi2_contingency`)
- **Add bootstrap 95% CI** to I_spec (100 resamples per layer, ~1 second compute)
- **Add Miller-Madow bias estimate** to I_spec output
- **Add I_spec_max** (maximum across layers) alongside mean
- **Add per-group I_spec** computation (I_spec within each of 8 routing groups)
- **Add shared/routed output norm ratio** logging
- **Increase I_spec measurement frequency** to every 250 steps
- **Fix h_expert bug** (overwrites per-layer, only last layer retained)
- **Limit cluster-based I_spec** to 10K-token sample and every 2000 steps (not 250 — too expensive at 1.08B)
- **Fix reference scale defaults** in `pre_train.py`: `--mup-ref-width` and `--mup-ref-batch-tokens` must be consistent with masterplan (768, 262144 for anchor-as-reference)
- **Add data shard validation**: Pre-flight check that reads each parquet file footer and verifies row counts

**Data preparation**:
- Prepare external domain eval data (balanced, 2K tokens per domain, 10K total)
- **Domain label validation** (added v8): Manually label 200 randomly sampled documents (40 per domain). Compute accuracy for each heuristic. Report confusion matrix. If any domain accuracy <80%, fix the heuristic or merge with its most-confused neighbor.
- Use **high-confidence exemplars**: pure Python scripts for code, pure fiction for books, pure arXiv abstracts for science. Minimize cross-domain contamination.

**Infrastructure validation**:
- Lock configs, seeds, systems contract, data manifest
- Validate checkpoint save/load (including RNG state)
- **Write and run replay test** (`test_replay.py`): Run 20 steps, save checkpoint, run 20 more steps, resume from checkpoint, compare — model weights must be bitwise identical, loss at step 21 must match
- **Write and run fault-injection test**: Save at step 50, kill process (SIGKILL), resume, verify loss continuity and no checkpoint corruption
- Run Gate 1 smoke test (100 steps)
- Verify I_spec produces valid, non-degenerate values on 100-step run
- Produce one profiler trace

**Pilot runs**:
- Run aux-loss coefficient pilot (R1: alpha in [0.001, 0.01, 0.1], 500 steps each)
- Run I_spec pilot (R0: 500 steps, measure I_spec standard deviation for descriptive threshold)
- Verify I_spec/H(D) > 0.02 after 500 steps (if not, investigate before proceeding — metric may be uninformative at this scale)

### Phase 1: HP Transfer Study (~$50, ~2 days)
- Run 8-point HP grid at 175M anchor (1000 steps each)
- Run 8-point HP grid at 410M (1000 steps each)
- Run 4 router LR configs at 410M (1000 steps each)
- Select best HPs, compute muP transfer predictions

### Phase 1.5: Convergence Pilot (~$35, ~1 day) — added v8
- Run 1 seed of R0 (aux-loss-free) at 410M to **full ablation budget** (~125K steps)
- Plot I_spec/H(D) vs step. Determine convergence point.
- If I_spec converges by step 6000-8000: proceed with 8000-step Phase 2
- If I_spec is still rising at step 8000: extend Phase 2 step budget accordingly
- This single run resolves the "null result ambiguity" for $35

### Phase 2: Specialization Study (~$70, ~3 days)
- Run R0 (aux-loss-free) at 410M with 3 seeds, 8,000 steps each (may extend based on convergence pilot)
- Run R1 (aux-loss) at 410M with 3 seeds, 8,000 steps each
- Full instrumentation every 250 steps: I_spec (labeled + normalized), H_load, domain BPB, dead experts, per-group I_spec, shared/routed norm ratio, bootstrap CIs
- Log gradient norms for **ALL parameter groups** (not just router) to control for optimization confound
- Log optimizer state statistics (Adam second moments for router vs dense)
- **All comparisons are exploratory**: report effect sizes, CIs, Bayesian credible intervals. No binary significance claims.

### Phase 3: 1.08B Validation (~$400, ~2 days)
- Run muP transfer validation (muP-predicted + grid + naive control, 2 seeds for muP-predicted)
- Run winning routing arm at 1.08B (2 seeds, **4000 steps** — extended from 1000 in v8)
- Run batch scaling control: 1 run with anchor HPs without √(B/B_ref) correction
- If muP transfer fails: run SqrtGate fallback test (~$5-10)
- Compare muP prediction vs actual, I_spec patterns across scales

### Phase 4: Analysis and Writeup (~1 week)
- Produce all artifacts (see Mandatory Outputs)
- Plot L vs N_active for the 3 scales (appendix figure only, NOT a scaling law claim)
- Write 3-5 page research report

**Total budget**: ~$700 (increased from $600 in v8 to accommodate convergence pilot +$35, batch scaling control +$5, extended Phase 3 +$60)  
**Total timeline**: ~5-6 weeks

If Phase 0 shows I_spec is degenerate or uninformative, stop Workstream 1 and report that.  
If Phase 1 shows muP transfer fails entirely, report the failure mode (still valuable).  
If Phase 2 shows no I_spec difference between R0 and R1, report the null result.

---

## X. Kill Criteria

Kill or invalidate runs if any persist:
- Repeated NaNs or irrecoverable divergence
- Checkpoint corruption
- Replay failure
- Restart mismatch beyond declared tolerance
- **Expert collapse**: H_load < **3.0 bits** across all seeds (tightened from 2.0 in v8 — 2.0 bits = only 4 effective experts, collapse is obvious long before this). **Warning threshold**: H_load < 4.0 bits (16 effective experts, 75% utilization — flag for monitoring).
- I_spec measurement produces degenerate values (exactly 0 or NaN) after code fix
- **I_spec floor** (added v8): If pilot I_spec/H(D) < 0.02 after 500 steps, investigate before proceeding. The metric may not be measurable at this scale/step count.
- Router gradient norms differ by >10x between R0 and R1 (calibrate this threshold from Phase 0 pilot data — if normal ratio is 3-5x, the 10x kill is reasonable; if normal is 1-2x, tighten to 5x)
- **I_spec-quality anti-correlation** (added v8): If Pearson r between final I_spec and final ema_val_bpb across all Phase 2 runs is < -0.3, the "specialization is good" assumption is wrong. Flag for scrutiny before interpreting results.

---

## XI. Mandatory Outputs

Each run:
1. Exact config artifact (including muP parameterization confirmation)
2. Exact code commit
3. Exact data manifest
4. Exact systems contract
5. Final checkpoint path
6. Quality metrics (ema_val_bpb, per-domain BPB)
7. Specialization metrics:
   - I_spec labeled-domain per layer (raw nats)
   - **I_spec/H(D) normalized per layer** ([0,1] range)
   - **I_spec_max** (maximum across layers)
   - **Bootstrap 95% CI** per layer
   - **Chi-squared p-value** per layer
   - **Miller-Madow bias estimate** per layer
   - **Per-group I_spec** (8 groups)
   - I_spec k-means per layer (secondary, every 2000 steps)
   - H_load, dead expert count, routing entropy, H(E|D)
   - **Shared/routed output norm ratio**
   - **Effective expert support per domain** (n_eff_d)
8. Gradient norms for **ALL parameter groups** (router, dense, embeddings, norms — for R0/R1 confound control)
8b. **Optimizer state statistics**: Adam second moments for router vs dense groups
9. Throughput / MFU / step_time_ms
10. Restart-proof log
11. Replay-proof log

The study must produce:
- `results/i_spec_diagnostic_report.md` — I_spec training curves and informativeness analysis
- `results/specialization_comparison_report.md` — R0 vs R1 with all metrics
- `results/mup_transfer_report.md` — muP prediction vs actual, router LR findings
- `results/i_spec_results.json` — machine-readable I_spec data
- `results/mup_results.json` — machine-readable HP transfer data
- Expert-domain affinity heatmaps at 4 timepoints per arm
- A profiler summary per scale

---

## XII. What Frontier Labs Would Actually Respect

1. **Checking the literature before claiming novelty** — and correcting when the literature has moved (v5 → v6)
2. **Understanding how the tool would actually be used** — I_spec is an ablation selection metric, not a production intervention trigger. Framing matches real frontier workflows (v6 → v7)
3. **Honest accounting of metric limitations** — MI measures routing preference, not functional specialization. The ceiling, bias, and domain noise are quantified and reported, not hidden (v8)
4. **Statistical honesty** — n=3 per group cannot support confirmatory claims. H2 is labeled exploratory. Effect sizes and CIs replace p-values. The study does not pretend to have power it lacks (v8)
5. **Theoretical gap acknowledgment** — SqrtGate omission, Muon batch scaling, depth-to-width ratio are listed as known limitations with planned fallbacks, not swept under the rug (v8)
6. **Introducing a genuinely new measurement tool** (I_spec as online diagnostic) rather than re-discovering known results
7. **Honest labeling**: replication is labeled replication, exploratory is labeled exploratory, novel is labeled novel
8. **A negative result is still a result** — if I_spec is uninformative, that saves other teams from instrumenting it
9. **Engineering discipline through systems integrity** — replay, restart, profiling
10. **Scientific maturity** — pilot measurements before locking thresholds, convergence check before main study, gradient norm controls, co-primary metrics, pre-specified sensitivity analyses

---

## XIII. How This Transfers to Billion+ Parameter Models

### Operational Reality at Frontier Scale

Frontier teams do not tune MoE routing mid-run based on diagnostic signals. A 671B production run costs $1-10M+, has locked hyperparameters, and is only interrupted for emergencies (loss spikes, NaN, hardware failure). The value of MoE diagnostics is in the **ablation phase before launch**, where teams run hundreds of small-scale experiments to select the configuration they commit to.

### Transfer Table (Honest Framing)

| NanoSeek Result | Frontier Application | Phase Where It Matters |
|---|---|---|
| I_spec is informative during training | Configuration selection metric: discriminate routing strategies during ablations before committing to production run | **Ablation phase** (cheap) |
| I_spec differentiates arms before loss does | Faster ablation runs: fewer steps needed to distinguish good from bad configurations, saving ablation budget | **Ablation phase** (cheap) |
| I_spec layer-wise patterns persist across scale | Qualitative validation that the metric measures a real property, not a scale artifact | **Scale validation** |
| Aux-loss-free achieves higher I_spec | Quantitative mechanism for why it works (gradient interference hypothesis), not just "DeepSeek says so" | **Architecture decisions** (pre-launch) |
| muP-MoE works with MuonAdamW | Extend published muP-MoE results to popular optimizer — reduce HP search cost at scale | **HP selection** (pre-launch) |
| muP-MoE fails for router with standard gate | Tells labs to adopt SqrtGate (HyperP) or tune router separately | **Architecture decisions** (pre-launch) |

### What I_spec Does NOT Do at Scale

- It does not "save" a production run by triggering early stopping or HP adjustment
- It does not replace loss curves or H_load as the primary monitoring signal during production
- It is logged during production for **post-hoc scientific understanding**, not for intervention

### What I_spec DOES Do at Scale

- During ablation campaigns (which every frontier lab runs): adds a selection criterion that loss curves and H_load cannot provide
- Prevents committing $1M+ to a routing configuration where experts are balanced but not specialized (the "balanced-but-useless" failure mode invisible to H_load)
- Provides mechanism evidence for architecture decisions (aux-loss choice, router design) that propagate to all future runs

---

## XIV. Codebase Fixes Required Before Launch

### P0: Must Fix (blocks all training)

| Fix | File | What |
|---|---|---|
| **RNG state in checkpoints** | `scripts/pre_train.py` | Add torch/CUDA/random/numpy RNG state to save AND resume (~10 lines). Replay proof impossible without this. |
| **Domain eval dataset** | External | Curate balanced 10K-token eval set (2K per domain) from high-confidence exemplars. Single point of failure for I_spec. |
| I_spec domain source | `eval/information_metrics.py` | Replace k-means clustering with labeled-domain computation as primary |
| h_expert bug | `eval/information_metrics.py` | Per-layer accumulation overwrites — fix to collect all layers |
| **Reference scale defaults** | `scripts/pre_train.py` | Reconcile `--mup-ref-width` (currently 1280) with masterplan (768 for anchor-as-reference). Fix comment on line 559. |

### P1: Must Fix (blocks Phase 2)

| Fix | File | What |
|---|---|---|
| I_spec/H(D) normalization | `eval/information_metrics.py` | Add `i_spec_normalized = i_spec / h_domain` to output dict (1 line per layer) |
| I_spec_max | `eval/information_metrics.py` | Add maximum I_spec across layers to output |
| Chi-squared test | `eval/information_metrics.py` | Add `scipy.stats.chi2_contingency(joint)` p-value to output |
| Bootstrap CIs | `eval/information_metrics.py` | 100 resamples of contingency table, report 2.5th/97.5th percentiles |
| Miller-Madow bias | `eval/information_metrics.py` | Compute (n_eff-1)(n_domains-1)/(2N) per layer |
| Per-group I_spec | `eval/information_metrics.py` | Compute I_spec within each of 8 routing groups |
| Shared/routed norm ratio | `scripts/pre_train.py` or `eval/moe_diagnostics.py` | Log ||shared_output|| / ||routed_output|| |
| Cluster I_spec perf fix | `scripts/pre_train.py` | Limit cluster-based I_spec to 10K sample, every 2000 steps (not 250) |
| I_spec frequency | `scripts/pre_train.py` | Change from milestone-only (4 points) to every 250 steps |
| MTP frequency | `scripts/pre_train.py` | Change from every 2000 steps to every 250 steps |
| sklearn + scipy deps | `pyproject.toml` | Add scikit-learn and scipy to requirements |

### P2: Must Fix (blocks Phase 0 completion)

| Fix | File | What |
|---|---|---|
| **Replay test** | `tests/test_replay.py` | New file: run 20 steps → save → run 20 more → resume from checkpoint → compare weights and loss |
| **Fault-injection test** | `tests/test_fault_injection.py` | New file: save at step 50 → kill → resume → verify loss continuity |
| **Data shard validation** | `nanoseek/dataset.py` | Pre-flight check: read each parquet file footer, verify row counts, check for corruption |
| **Domain label validation** | External script | Manual labeling of 200 docs (40/domain), confusion matrix, accuracy check |
| Aux-loss coefficient | Config | Calibrate via pilot before locking |

---

## XV. Source Links

### muP-MoE (Published — NanoSeek Builds On These)
- mu-Parameterization for MoE (Aug 2025): https://arxiv.org/abs/2508.09752
- HP Transfer with MoE Layers (Jan 2026): https://arxiv.org/abs/2601.20205
- HyperP (Mar 2026): https://arxiv.org/abs/2603.28743

### muP Foundation
- Tensor Programs V (Yang et al. 2022): https://arxiv.org/abs/2203.03466
- Cerebras muP Guide: https://www.cerebras.ai/blog/the-practitioners-guide-to-the-maximal-update-parameterization
- Sparse muP (NeurIPS 2024): https://proceedings.neurips.cc/paper_files/paper/2024/file/3b6afffec941f98930753fa6d6de7263-Paper-Conference.pdf

### MoE Routing and Specialization
- DeepSeek V3: https://arxiv.org/abs/2412.19437
- Auxiliary-Loss-Free Load Balancing: https://arxiv.org/abs/2408.15664
- Advancing Expert Specialization: https://arxiv.org/abs/2505.22323
- On Implementing Load Balancing Loss: https://arxiv.org/abs/2501.11873
- OLMoE: https://arxiv.org/abs/2409.02060

### Hybrid Attention (Settled — Not Studied Here)
- Kimi Linear: https://arxiv.org/abs/2510.26692
- OLMo Hybrid: https://allenai.org/blog/olmohybrid
- Qwen3.5: https://huggingface.co/blog/mlabonne/qwen35

### Systems
- TorchTitan: https://github.com/pytorch/torchtitan
- PyTorch async checkpointing: https://pytorch.org/blog/6x-faster-async-checkpointing/
- PyTorch fault-tolerant training: https://pytorch.org/blog/fault-tolerant-llama-training-with-2000-synthetic-failures-every-15-seconds-and-no-checkpoints-on-crusoe-l40s/

### Frontier Lab Context
- Anthropic Pretraining RE: https://job-boards.greenhouse.io/anthropic/jobs/5135168008
- Anthropic Pretraining Scaling RE: https://job-boards.greenhouse.io/anthropic/jobs/4938432008

---

## XVI. Final Narrative

> NanoSeek is a small MoE measurement program. Its primary contribution is **I_spec** — expert-domain mutual information tracked as an online training diagnostic — a metric that has never been systematically monitored during MoE training despite expert specialization being the entire purpose of the architecture. I_spec measures routing preference, not functional specialization; its MI ceiling is bounded by the number of domains; and its value lies in ablation-phase configuration selection, not mid-run intervention. These limitations are quantified and reported. The secondary contribution is a controlled comparison of aux-loss vs aux-loss-free routing, framed as exploratory with effect sizes rather than binary significance (because n=3 per group cannot support confirmatory claims). The muP-MoE component is replication-with-extension, with honest acknowledgment that SqrtGate omission is the most likely failure point and batch scaling for Muon is theoretically unvalidated. The plan was revised four times: to kill an answered question, to correct novelty claims, to correct the operational model, and to address metric limitations, statistical power, and theoretical gaps surfaced by a 5-agent deep review. That rigor — stress-testing your own plan, downgrading claims when the evidence doesn't support them, and adding controls when theory has gaps — is itself the strongest signal of scientific maturity.
