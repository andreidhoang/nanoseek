# NanoSeek: Research Rationale from First Principles

**A complete account of why every major decision was made, what research it rests on, and what the project ultimately aims to prove.**

*Last updated: 2026-03-13*

---

## Table of Contents

1. [The Problem We Solve](#1-the-problem-we-solve)
2. [The Ultimate Goal](#2-the-ultimate-goal)
3. [The 7 Fundamental Laws That Make This Possible](#3-the-7-fundamental-laws-that-make-this-possible)
4. [From Budget to Architecture: The Complete Decision Chain](#4-from-budget-to-architecture-the-complete-decision-chain)
5. [MoE Sizing: The Krajewski-OLMoE Derivation](#5-moe-sizing-the-krajewski-olmoe-derivation)
6. [MLA: Why Compressed Attention Works](#6-mla-why-compressed-attention-works)
7. [Hyperparameter Transfer: muP for MoE](#7-hyperparameter-transfer-mup-for-moe)
8. [Training Recipe: What and Why](#8-training-recipe-what-and-why)
9. [Depth vs Width: The Allen-Zhu Analysis](#9-depth-vs-width-the-allen-zhu-analysis)
10. [Honest Provenance: What Comes From Where](#10-honest-provenance-what-comes-from-where)
11. [Risk Register: What Could Fail](#11-risk-register-what-could-fail)
12. [What "Done" Looks Like](#12-what-done-looks-like)
13. [Complete Citation List](#13-complete-citation-list)

---

## 1. The Problem We Solve

Modern frontier MoE models (DeepSeek V3, Mixtral, GPT-4) demonstrate that sparse expert architectures achieve superior quality-per-FLOP compared to dense models. But the research community lacks:

1. **Reproducible scaling science for MoE at accessible scale.** DeepSeek V3's scaling experiments required >$5M compute. No published work validates MoE scaling laws at the $50-500 budget range where independent researchers can iterate.

2. **muP transfer validation for MoE.** Maximal Update Parameterization (Yang 2022) enables hyperparameter transfer across model widths — proven for dense transformers but **unvalidated for MoE architectures**. μP-MoE (arXiv:2508.09752) provides the theory; we provide the empirical test.

3. **A complete, educational implementation** of DeepSeek V3's innovations (MLA, auxiliary-loss-free routing, MTP, DSA) that a single researcher can understand, train, and experiment with.

NanoSeek addresses all three by building a 1.08B-active / 4.75B-total MoE from first principles, with every design choice traceable to a specific paper and derivation.

---

## 2. The Ultimate Goal

The project produces five artifacts, each with falsifiable success criteria:

| Artifact | Success Criterion | Why It Matters |
|----------|------------------|----------------|
| **Trained NanoSeek-1B checkpoint** | Converged ema_val_bpb, H_load > 2 bits | Proves the architecture works at this scale |
| **muP HP transfer report** | 3-point validation (55M → 441M → 1.08B) without per-scale tuning | First empirical test of muP for MoE |
| **Stability playbook** | 3 ablations (A, C, D) with documented H_load/I_spec trajectories | Actionable knowledge for MoE training stability |
| **Scaling law fit** | L(N_active, D, E) with R² > 0.95 from EMA validation BPB | Quantitative model for MoE compute-optimal training |
| **3-stage RL pipeline** | GSM8K/HumanEval improvement at 3 compute budgets | Validates staged RL for small MoE |

The overarching scientific question: **Can muP-corrected scaling rules transfer to MoE architectures, enabling cheap hyperparameter search at small scale to predict large-scale training behavior?**

If yes, this makes MoE training accessible to researchers without frontier-scale compute.

---

## 3. The 7 Fundamental Laws That Make This Possible

Every engineering decision in NanoSeek rests on one or more of these established results. Each law is stated precisely, with its consequence for our design.

### Law 1: The Neural Scaling Law

**Statement** (Kaplan et al. 2020; Hoffmann et al. 2022):
Language model cross-entropy loss follows a power law in model size and data:

```
L(N, D) = L_irr + A/N^α + B/D^β
```

where L_irr is the irreducible loss (entropy of natural language), N is parameter count, D is training tokens, and α ≈ 0.34, β ≈ 0.28 (Chinchilla estimates).

**For MoE** (Ludziejewski et al., ICML 2025):
The law extends to:

```
L(N_active, D, E) = L_irr + A/N_active^α + B/D^β + γ·log(E)
```

The critical insight: **N_active (not N_total) is the primary scaling variable.** Total parameters matter only logarithmically through expert count E. This means MoE's compute advantage comes from the fact that N_total >> N_active, giving more knowledge capacity at the same FLOP budget.

**Consequence for NanoSeek:** We size the model by N_active (1.08B) and compute by FLOPs = 6 × N_active × D. The 4.75B total parameters are "free" in compute terms — they provide knowledge capacity without proportional FLOP cost.

### Law 2: Chinchilla Compute-Optimal Training

**Statement** (Hoffmann et al. 2022):
For a given compute budget C, the optimal allocation between model size N and data D is approximately:

```
D_optimal ≈ 20 × N
```

Training with fewer tokens leaves model capacity underutilized. Training with more tokens wastes compute on diminishing returns.

**Derivation for NanoSeek:**

```
Budget constraint: 8×H100, ~14 hours → C ≈ 1.3 × 10²⁰ FLOPs
From C = 6 × N_active × D and D = 20 × N_active:
  C = 6 × N_active × 20 × N_active = 120 × N_active²
  N_active = √(C / 120) = √(1.3e20 / 120) ≈ 1.04B → rounded to 1.08B
  D = 20 × 1.08B = 21.6B → rounded to 22B tokens
```

**Consequence:** N_active ≈ 1.08B and D = 22B are not arbitrary — they are the compute-optimal allocation for our hardware budget.

### Law 3: MoE Compute Advantage

**Statement** (Fedus et al. 2022; Ludziejewski et al. 2025):
In a top-k MoE, only k of E experts execute per token. The FLOPs per forward pass scale with N_active, but the model stores N_total parameters:

```
FLOPs = 6 × N_active × D    (same as dense model with N_active params)
Knowledge capacity ∝ N_total  (all expert weights store learned patterns)
```

This creates a "free lunch": N_total/N_active = expansion ratio gives that factor more knowledge storage at the same compute cost.

**For NanoSeek:**
- N_active = 1.08B, N_total = 4.75B
- Expansion = 4.4×
- The model stores knowledge as if it were 4.75B dense, but trains at the cost of 1.08B dense.

**This is why MoE exists.** Not for speed — for knowledge capacity per FLOP.

### Law 4: Expert Collapse (The Fundamental MoE Failure Mode)

**Statement** (Shazeer et al. 2017; Fedus et al. 2022):
Without explicit load balancing, MoE training converges to a degenerate equilibrium where 3-5 experts handle all tokens while the rest go unused. This is a stable attractor — once an expert receives more tokens, it trains faster, receives higher routing scores, and captures even more tokens.

The result: a 4.75B-parameter model that behaves like a ~200M dense model, wasting 95% of its capacity.

**Prevention mechanisms:**
- **Auxiliary loss (traditional):** Add a term to the training loss that penalizes uneven load. Problem: this distorts the main language modeling objective.
- **Aux-loss-free bias balancing (DeepSeek V3):** Maintain per-expert bias terms updated by a simple rule: if an expert is overloaded, decrease its bias; if underloaded, increase it. Learning rate γ = 0.001, frozen at 95% of training.

**Consequence for NanoSeek:** We use DeepSeek V3's aux-loss-free approach because it prevents collapse WITHOUT distorting the main loss. The bias terms act as a thermodynamic regulator — they don't change what the model learns, only which experts learn it.

**Monitoring:** Expert load entropy H_load must be tracked continuously. H_load < 2 bits is the alert threshold (random routing gives H_load = log₂(64) ≈ 6 bits). Collapse is silent — it develops over 400+ steps before affecting loss.

### Law 5: Krajewski Granularity Theory

**Statement** (Krajewski et al., ICML 2024):
MoE performance depends on expert granularity G, defined as:

```
G = N_active / (top_k × expert_params)
```

At a given compute budget, there exists an optimal G range. For compute ~10²⁰ FLOPs (our range), G = 16-32 is optimal. Too-large experts (low G) waste capacity on redundancy. Too-small experts (high G) lack capacity for meaningful specialization.

**Forward derivation for NanoSeek:**

```
Target: G ≈ 29 (mid-range of 16-32 optimal)
Given: N_active = 1.08B, top_k = 8

expert_params = N_active / (top_k × G)
             = 1.08B / (8 × 29)
             = 4.66M

With SwiGLU architecture: expert_params = 3 × hidden × moe_inter
moe_inter = 4.66M / (3 × 2048) = 758 → rounded to 768

Actual G = 1.08B / (8 × 3 × 2048 × 768) = 28.6 ✓
```

**This derivation is forward from the principle, not backward from a desired number.** The moe_inter/hidden ratio of 0.375 is a consequence of targeting G≈29 — it is not "close to DeepSeek's ratio" (which is 0.286 for a different G target at different scale).

### Law 6: muP Hyperparameter Transfer

**Statement** (Yang 2022 — Tensor Programs V):
Under Maximal Update Parameterization, optimal hyperparameters transfer across model widths if:

1. Hidden-layer weight learning rates scale as 1/fan_in
2. Output-layer learning rates remain constant
3. Initialization scales as 1/√fan_in for hidden, 1/fan_in for output
4. Embedding learning rates scale independently (typically constant)

**Extension to MoE** (arXiv:2508.09752, μP-MoE):
For MoE architectures, additional rules apply:
- Expert FFN weights are "hidden weights" → LR ∝ 1/width
- Router weights are "output weights" → LR constant
- κ (sparsity ratio = top_k/E) must remain constant across scales
- Expert count E should remain constant; only hidden_size changes

**Extension to batch size and duration** (arXiv:2512.22382, Complete(d)P):
- Optimal batch size scales as B ∝ √(model_size)
- Training duration T_epoch can be transferred with appropriate LR schedule adjustment

**Consequence for NanoSeek:**
Our 3-point muP transfer path (anchor 55M → 500M → 1.08B) holds constant:
- Depth: 16 layers (all three configs)
- Expert count: 64 (all three)
- top_k: 8 (all three)
- κ: 12.5% (all three)
- moe_inter/hidden: 0.375 (all three)
- n_shared_experts: 2 (all three)
- MLA ratios: q_lora/h=0.215, kv_lora/h=0.070 (all three)

Only hidden_size changes: 480 → 1280 → 2048. This is exactly what muP requires — vary the "width" axis while holding everything else constant.

The research question: **Does this actually work for MoE?** No published empirical validation exists. Our 3-point path tests it.

### Law 7: Knowledge Capacity

**Statement** (Allen-Zhu, Physics of Language Models Part 3.3):
A transformer can store at most 2 bits of knowledge per parameter. This is a fundamental limit, not an empirical observation. Moreover, knowledge accessibility (ability to retrieve stored knowledge) depends on training data diversity, not model capacity.

**For NanoSeek:**
- N_total = 4.75B → theoretical capacity = 9.5 billion bits of knowledge
- N_active = 1.08B → accessible per forward pass: 2.16 billion bits
- The MoE expansion gives us 4.4× more storage than a dense 1.08B model

This has a concrete implication for our scaling law: if our irreducible loss L_irr is higher than expected, the bottleneck may be data diversity (knowledge accessibility), not model capacity (knowledge storage). Allen-Zhu's framework lets us distinguish these two failure modes.

---

## 4. From Budget to Architecture: The Complete Decision Chain

Every architectural parameter traces back to the hardware budget through a chain of principled derivations. Here is the complete chain:

```
HARDWARE BUDGET: 8×H100, ~14 hours, $275-350
         │
         ▼
COMPUTE BUDGET: C ≈ 1.3 × 10²⁰ FLOPs
         │
         ▼  (Chinchilla: D = 20 × N)
TWO CONSTRAINTS:
  N_active ≈ 1.08B
  D = 22B tokens
         │
         ▼  (Allen-Zhu: depth=reasoning, width=knowledge)
DEPTH × WIDTH:
  16 layers × 2048 hidden
  d/L = 128 (matches OLMoE-1B precedent)
  16 layers → sufficient reasoning depth for 1B-class model
  2048 width → sufficient knowledge capacity
         │
         ▼  (Krajewski: G=16-32 optimal at 10²⁰ FLOPs)
EXPERT GRANULARITY:
  G ≈ 29 (mid-range optimal)
  expert_params = 1.08B / (8 × 29) = 4.66M
  moe_inter = 4.66M / (3 × 2048) = 768
         │
         ▼  (OLMoE: validated E=64 at 1B scale)
EXPERT COUNT:
  E = 64, top_k = 8, κ = 12.5%
  (not DeepSeek's 256 — too small per-expert at our scale)
         │
         ▼  (DeepSeek V3: MLA compression ratios)
MLA SIZING:
  q_lora_rank = 0.215 × 2048 = 440
  kv_lora_rank = 0.070 × 2048 = 143
  Compression: 23× KV cache reduction
         │
         ▼  (Expert collapse prevention)
LOAD BALANCING:
  Aux-loss-free (DeepSeek V3 method)
  γ = 0.001, frozen at 95% of training
  Monitor: H_load > 2 bits throughout
         │
         ▼  (muP: width-only scaling)
TRANSFER PATH:
  anchor (480h) → 500M (1280h) → 1B (2048h)
  Depth=16, E=64, κ=12.5% constant everywhere
  HP search at anchor ($5), validate at 500M ($30), run at 1B ($300)
         │
         ▼
FINAL ARCHITECTURE:
  1.08B active / 4.75B total MoE
  MLA + SwiGLU + aux-loss-free routing + MTP + DSA
```

Every arrow in this chain represents a paper-backed derivation, not a guess.

---

## 5. MoE Sizing: The Krajewski-OLMoE Derivation

This section explains the MoE configuration in detail because it is where most projects make ad hoc choices.

### Why E=64 and not 256 (like DeepSeek V3)

DeepSeek V3 uses 256 experts with hidden_size=7168. Each expert has:
```
expert_params_V3 = 3 × 7168 × 2048 = 44M
```

If we used E=256 with hidden_size=2048:
```
expert_params_ours = 3 × 2048 × moe_inter
```

For our N_active=1.08B with top_k=8:
```
expert_params = 1.08B / (8 × G)
```

At G=29: expert_params = 4.66M → moe_inter = 758.

With E=256: we'd need 256 experts of 4.66M each, giving N_total = 256 × 4.66M + shared ≈ 1.4B. This is problematic:
- Expansion ratio only 1.3× (barely MoE)
- At anchor scale (480h), each expert would be ~65K params — too small for SwiGLU to learn meaningful patterns

With E=64: N_total = 64 × 4.66M + shared ≈ 4.75B. Expansion = 4.4×.
- Matches OLMoE-1B's validated configuration
- At anchor scale, each expert has ~255K params — still viable for specialization

**The expert count is constrained by the anchor scale viability requirement for muP transfer.** This is a practical constraint, not a theoretical preference.

### Sparsity ratio κ = 12.5%

```
κ = top_k / E = 8 / 64 = 12.5%
```

Compared to DeepSeek V3: κ = 8 / 256 = 3.1%. We are 4× denser.

Combinatorial diversity:
```
C(64, 8) ≈ 4.4 × 10⁹ unique expert combinations
C(256, 8) ≈ 4.4 × 10¹³ unique combinations (10,000× more)
```

4.4 billion combinations is more than sufficient for 22B tokens. Each token sees a unique expert combination even if we never repeat.

This κ is validated by OLMoE-1B-7B, which published competitive results with the same E=64, top_k=8 configuration.

### Why 2 shared experts

DeepSeek V3 uses 1 shared expert at 671B total. DeepSeekMoE (arXiv:2401.06066) used 2 at 16B total. We use 2 at 4.75B total.

Rationale: shared experts handle "common knowledge" that all tokens need (syntax, frequent patterns). At our smaller scale, the shared/routed ratio matters more:
- 2 shared: 2/(2+8) = 20% shared capacity
- 1 shared: 1/(1+8) = 11% shared capacity

**This is explicitly an unvalidated hypothesis.** No published ablation proves 2 > 1 at our scale. It could be tested as a single-variable experiment: switching to 1 shared would free ~66M params for additional routed experts.

---

## 6. MLA: Why Compressed Attention Works

Multi-head Latent Attention (DeepSeek V3) replaces standard multi-head attention's per-head KV cache with a shared low-rank representation.

### The KV cache problem

Standard MHA with n_heads=16, head_dim=128:
```
KV cache per token = 2 × n_heads × head_dim = 2 × 16 × 128 = 4096 floats
```

For 8K context at batch_size=1 in BF16:
```
KV cache = 16_layers × 8192 × 4096 × 2 bytes = 1.07 GB
```

This dominates inference memory at long contexts.

### MLA solution

Compress KV into a single low-rank vector:
```
kv_lora_rank = 143 (vs 4096 for standard)
Compression = 4096 / 143 ≈ 23×

MLA KV cache = 16 × 8192 × 143 × 2 = 37 MB  (vs 1.07 GB)
```

The key insight from DeepSeek V3: the compressed representation c_t is a shared "latent" that is expanded via learned projections into keys and values at inference time. Training learns what to compress.

### MLA ratios (direct from DeepSeek V3 paper)

```
q_lora_rank / hidden = 440 / 2048 = 0.215
kv_lora_rank / hidden = 143 / 2048 = 0.070
```

These ratios are maintained across our muP transfer path:
- Anchor: q_lora=103, kv_lora=34 (0.215 × 480, 0.070 × 480)
- 500M: q_lora=275, kv_lora=90 (0.215 × 1280, 0.070 × 1280)
- 1B: q_lora=440, kv_lora=143 (0.215 × 2048, 0.070 × 2048)

**Why trust DeepSeek's ratios?** They trained at 671B total parameters with extensive ablations. At our scale, the compression ratio matters less (the cache is already small), but maintaining the ratio is necessary for muP transfer validity.

---

## 7. Hyperparameter Transfer: muP for MoE

This is the project's most novel scientific contribution. No published work empirically validates muP transfer for MoE architectures.

### What muP does (dense models)

Standard practice: train a large model, tune hyperparameters at that scale. Cost: proportional to model size × number of HP trials.

muP insight: under specific parameterization rules, the optimal learning rate at width W₁ predicts the optimal learning rate at width W₂. You search HPs at small scale and transfer them to large scale for free.

### What muP requires for MoE (μP-MoE, arXiv:2508.09752)

Additional constraints beyond dense muP:

1. **Expert weights are "hidden weights"**: LR ∝ 1/width (same as MLP layers)
2. **Router weights are "output weights"**: LR constant across widths
3. **κ must be constant**: top_k/E is an architectural ratio, not a scaling variable
4. **E must be constant**: Expert count is not a width; varying it changes the model class

### Our 3-point validation design

```
Scale        │ hidden │ N_active │ Compute │ Purpose
─────────────┼────────┼──────────┼─────────┼───────────────
Anchor (55M) │  480   │  ~55M    │   ~$5   │ HP search (grid)
500M         │  1280  │  ~441M   │  ~$30   │ Transfer validation
1B (target)  │  2048  │  ~1.08B  │  ~$300  │ Final model
```

Constants across all three:
- num_layers = 16
- n_routed_experts = 64
- num_experts_per_tok = 8
- κ = 12.5%
- moe_inter/hidden = 0.375
- n_shared_experts = 2
- MLA ratios (q_lora/h, kv_lora/h)
- Head dimension structure (qk_nope:qk_rope:v = 3:1:3)

**Success criterion:** nano-500M trains to reasonable ema_val_bpb using auto-scaled HPs from anchor, without manual tuning. If manual tuning is needed, we document which scaling rule broke and why.

### What this would prove

If muP transfer works for MoE:
- HP search cost drops from ~$300 (at 1B) to ~$5 (at anchor)
- Any researcher with a single GPU can find good HPs for a larger MoE
- Scaling law predictions become more reliable (HPs are consistent across scales)

If it fails:
- We document which muP assumption breaks for MoE (still valuable)
- We fall back to per-scale tuning (more expensive but still feasible)

---

## 8. Training Recipe: What and Why

Each training decision has a specific rationale rooted in the MoE or MLA architecture.

### Optimizer: Muon + AdamW

**Why not just AdamW?** MoE experts receive sparse gradients — only 1/8 of tokens activate each expert. This means gradient statistics are noisy and change rapidly as routing evolves.

Muon (spectral normalization optimizer) prevents rank collapse in weight matrices by maintaining spectral properties. It's applied to 2D weight matrices (attention projections, expert FFNs). AdamW handles everything else (embeddings, norms, scalars).

Expert FFN weights create shape groups for Muon: `[768, 2048]` (gate/up projections) and `[2048, 768]` (down projections). MLA projections add additional shapes.

### β₂ = 0.95 (not the standard 0.999)

Standard Adam uses β₂ = 0.999, which means the second moment estimate has an effective window of ~1000 steps. For MoE experts that receive sparse, rapidly-changing gradients due to routing dynamics, this carries stale gradient statistics.

β₂ = 0.95 adapts in ~20 steps, tracking the current routing regime. This is especially important during early training when routing patterns are still forming.

### Gradient clipping at 1.0

MoE sparse activation + MLA low-rank amplification create high gradient variance. In BF16, unclipped gradients produce NaN within ~1000 steps. This is not optional.

### Batch size warmup (1/5 → 1× target over 10% of steps)

From Complete(d)P (arXiv:2512.22382): optimal batch size scales with training progress. Small batches early give more gradient updates per token (more exploration). Large batches late provide stable convergence.

DeepSeek V3 uses 3072→15360 SEQUENCES over first 3% of training (5× ramp). For NanoSeek, we apply the same 5× ratio scaled to our batch size: start at 1/5 of target global_batch_size, ramp to full over first 10% of steps (conservative vs V3's 3%). For NanoSeek-1B with global_batch_size=128 sequences: ~26→128 sequences.

### Learning rate schedule: warmup → constant → cosine decay (DeepSeek V3 style)

- **Warmup** (first 1000 steps, config.py: warmup_steps=1000): Linear increase from 0 to peak LR. Prevents early instability.
- **Constant** (70% of training, config.py: constant_phase_ratio=0.70): Stable learning at peak rate. This is where most learning happens.
- **Cosine decay** (70%→95% of training, config.py: cosine_decay_end_ratio=0.95): Cosine annealing from peak LR to lr_min. Smooth decay enables convergence into a loss minimum.
- **Minimum LR** (final 5%): Hold at lr_min (0.1× peak). Stabilizes final convergence.

### FIM at 10% (from token 1)

Fill-in-the-Middle (FIM) trains the model on P(middle | prefix, suffix) using Prefix-Suffix-Middle (PSM) format. This enables code infilling — a capability that cannot be added via fine-tuning (catastrophic interference with autoregressive distribution).

10% of training sequences are FIM-formatted. This rate is from established practice (Bavarian et al. 2022) and must be present from the first training token.

### MTP loss schedule (λ = 0.3 → 0.1)

Multi-Token Prediction auxiliary loss forces the model to plan ahead (predict the next token AND the token after that).

- **λ = 0.3 for first 60% of training**: Strong planning signal builds long-range representations early when the model benefits most from structural scaffolding.
- **λ = 0.1 for remaining 40%**: Reduced weight prevents MTP from interfering with main autoregressive convergence.

### Two-phase training (4K → 8K context)

**Phase 1 (80% of tokens): Dense attention at 4K context**
- Dense attention at 4K is O(4096²) = O(16M) per layer
- Dense attention at 8K would be O(8192²) = O(67M) — 4× more compute
- For the same token budget, 4K gives 4× more gradient updates
- The DSA indexer needs dense attention patterns to learn from

**Phase 2 (20% of tokens): Sparse DSA at 8K context**
- Enables long-context capability
- YaRN (Yet another RoPE extensioN) interpolates position encodings to 8K
- DSA indexer warms up: 1K steps frozen backbone, then joint training

### EMA tracking

Exponential Moving Average of model weights (Polyak averaging), not to be confused with EMA of scalar loss values (which is only for log smoothing).

**Formula:** θ_ema = α × θ_ema + (1 - α) × θ_model, where α = 0.9999, updated every 10 steps.

**Why it works (3 independent lines of evidence):**

1. **Polyak & Juditsky (1992):** Near a minimum, SGD oscillates. The time-average of these oscillations converges at rate O(1/T) regardless of learning rate, while the last iterate's error scales with η/B. EMA is a practical approximation of this average.

2. **Morales-Brotons et al. (2024, TMLR):** Systematic study across 7 architectures showing EMA models generalize better, are more robust to noisy labels, improve calibration, and learn more transferable representations. Key insight: EMA acts as an implicit regularizer — it can replace the final phase of LR decay because averaging naturally reduces gradient noise.

3. **Sanyal et al. (2023, COLM 2024):** Weight averaging specifically for LLM pretraining speeds up training across all workloads and enables higher learning rates (the averaging compensates for increased noise). Directly relevant because we use aggressive muP-scaled LRs.

**Why CPU-side?** DeepSeek V3 uses the same design (arXiv:2412.19437): EMA weights offloaded to CPU RAM, updated asynchronously. At 4.75B params × 2 bytes (BF16) = 9.5 GB, keeping EMA on GPU would double parameter memory. CPU RAM is effectively free for this purpose. For eval every 250 steps, the GPU←CPU copy cost is negligible.

**Why α = 0.9999?** Effective window ≈ 1/(1-α) = 10,000 steps. With ema_every=10: ~1,000 EMA updates over a typical run. Too small (0.99, window=100): too reactive, doesn't smooth. Too large (0.99999, window=100K): too stale, weights lag behind training. 0.9999 is standard practice (used in diffusion model training, validated in Morales-Brotons et al.).

**Monitoring note:** We should track both raw val_bpb and ema_val_bpb during early runs to verify EMA is actually better at our training length. For the anchor (~2,100 steps), the EMA window covers the full run. For 1B (~5,300 steps), the window covers ~50% — borderline but sufficient per Sanyal et al.'s findings.

**Critical rule:** ALL evaluation uses EMA weights. Raw checkpoint weights are noisier and produce less reliable BPB measurements. Scaling law fits depend on clean evaluation metrics.

### Aux-loss-free load balancing

Per-expert bias terms b_e, updated every step:

```python
if expert_load[e] > average_load:
    b_e -= γ  # Discourage overloaded expert
else:
    b_e += γ  # Encourage underloaded expert
```

γ = 0.001 for the first 95% of training. γ = 0 for the final 5% (freeze biases to avoid noise near convergence).

This is strictly superior to auxiliary loss approaches because it does not distort the main language modeling objective.

---

## 9. Depth vs Width: The Allen-Zhu Analysis

Allen-Zhu's "Physics of Language Models" (Parts 1-4) establishes that depth and width serve fundamentally different purposes in transformers. This has direct implications for our d/L = 128 architecture.

### The fundamental separation

```
WIDTH (hidden_size)  → Knowledge capacity: 2 bits per parameter
                     → Breadth of parallel reasoning
                     → CANNOT compensate for missing depth

DEPTH (num_layers)   → Maximum reasoning chain length
                     → Hierarchical structure processing
                     → Has hard minimum thresholds
                     → CANNOT be compensated by width
```

A 4-layer model with 10,000 hidden dims CANNOT do 5-hop reasoning. A 20-layer model with 576 hidden dims CAN — and outperforms the wider model on reasoning tasks despite having fewer total parameters.

### NanoSeek's position

| Model | d/L | Depth | Width | Character |
|-------|-----|-------|-------|-----------|
| GPT-2 Large (774M) | 36 | 36 | 1280 | Deep |
| OLMoE-1B (1B) | 128 | 16 | 2048 | Wide |
| NanoSeek-1B (1.08B) | 128 | 16 | 2048 | Wide |
| LLaMA-7B (7B) | 128 | 32 | 4096 | Wide + Deep |

NanoSeek is a "wide" architecture: high knowledge capacity (2048 width + 4.75B total params for MoE knowledge storage), moderate reasoning depth (16 layers → ~16-hop ceiling).

**Why this is the right trade-off at 1B scale:**
1. MoE already provides 4.4× parameter expansion for knowledge — width serves this well
2. Going deeper (e.g., 32 layers) would require narrowing to ~1024 hidden to stay at 1.08B active — this halves knowledge capacity per layer
3. 16 layers provides adequate reasoning depth for a 1B-class model
4. The d/L = 128 ratio matches OLMoE-1B, which published competitive results

### Implications for the muP transfer path

Our transfer path uses constant depth (16 layers) across all scales:

| Config | Layers | Hidden | d/L |
|--------|--------|--------|-----|
| Anchor | 16 | 480 | 30 |
| 500M | 16 | 1280 | 80 |
| 1B | 16 | 2048 | 128 |

The d/L ratio changes — this means the anchor has relatively more depth (reasoning-oriented) while the 1B has relatively more width (knowledge-oriented). Allen-Zhu's framework predicts:
- The anchor may show disproportionately good reasoning for its size
- The 1B may show disproportionately good knowledge recall for its size
- BPB (which combines both) should still follow the scaling law smoothly

This is a known confound — we document it rather than trying to eliminate it (which would require varying depth, breaking muP).

---

## 10. Honest Provenance: What Comes From Where

One of this project's principles is intellectual honesty about the source of each design choice. This section provides a complete accounting.

### From DeepSeek V3 (arXiv:2412.19437) — mechanisms and techniques

| Component | What we take | Confidence |
|-----------|-------------|------------|
| MLA architecture | Compression ratios (q_lora/h=0.215, kv_lora/h=0.070), decoupled RoPE | High — validated at 671B scale |
| Dense FFN sizing | SwiGLU with intermediate = 2.56 × hidden | High — standard practice |
| Routing mechanism | Sigmoid with grouped expert selection (n_group=8, topk_group=4) | High — published with ablations |
| Load balancing | Aux-loss-free bias adjustment, γ=0.001 | High — principled advantage over aux-loss |
| MTP | Multi-token prediction with k=1 additional head | Medium — beneficial but λ schedule is our choice |
| DSA | Dynamic sparse attention concept and indexer design | Medium — adapted for our scale |
| First-2-dense | First k layers use dense FFN instead of MoE | Medium — heuristic, not deeply ablated |
| top-8 activation | k=8 experts per token | High — same as V3, though κ differs |
| CPU-side EMA | EMA weights offloaded to CPU RAM, updated asynchronously | High — same design principle at 671B |

### From OLMoE (Muennighoff et al., 2024) — scale-class sizing

| Component | What we take | Confidence |
|-----------|-------------|------------|
| E = 64 | Expert count at 1B-class scale | High — published competitive results |
| κ = 12.5% | Sparsity ratio (consequence of E=64, k=8) | High — same as OLMoE-1B-7B |
| d/L = 128 | Depth-to-width ratio at 1B active | Medium — matches but may not be optimal |

### From Krajewski et al. (ICML 2024) — granularity theory

| Component | What we take | Confidence |
|-----------|-------------|------------|
| G ≈ 29 | Expert granularity target | High — within published optimal range |
| moe_inter = 768 | Forward-derived from G target | High — principled derivation |
| 0.375 ratio | moe_inter/hidden (consequence of G≈29) | High — derived, not assumed |

### From μP-MoE (arXiv:2508.09752) — transfer theory

| Component | What we take | Confidence |
|-----------|-------------|------------|
| κ-constant rule | Keep top_k/E constant across scales | Medium — theoretical, we test empirically |
| Expert LR scaling | Expert weights scale as 1/width | Medium — extends dense muP to MoE |
| Router LR constant | Router is "output weight" class | Medium — theoretical prediction |

### From DeepSeekMoE (arXiv:2401.06066) — small-scale MoE precedent

| Component | What we take | Confidence |
|-----------|-------------|------------|
| 2 shared experts | Small-scale shared expert count | LOW — unvalidated hypothesis at our scale |

### EMA weight averaging — multi-source provenance

| Component | Source | Confidence |
|-----------|--------|------------|
| Polyak averaging theory | Polyak & Juditsky (1992) — O(1/T) convergence of averaged iterates | High — foundational result, 30+ years validated |
| CPU-side EMA design | DeepSeek V3 (arXiv:2412.19437) — offload to CPU RAM, async update | High — same design at 671B scale |
| EMA as implicit regularizer | Morales-Brotons et al. (2024, TMLR) arXiv:2411.18704 | High — systematic study across 7 architectures |
| EMA for LLM pretraining | Sanyal et al. (2023, COLM 2024) arXiv:2306.03241 | High — speeds up training, enables higher LR |
| decay = 0.9999 | Standard practice in diffusion + LLM training; validated in Morales-Brotons | Medium — task-dependent, monitor ema vs raw |
| ema_every = 10 | Cost trade-off (4.75B param copy is expensive); DeepSeek V3 also updates async | Medium — practical choice, not deeply ablated |

### Backward-derived (not from any single source)

| Component | Derivation path | Confidence |
|-----------|----------------|------------|
| N_active = 1.08B | Budget → Chinchilla | High — standard derivation |
| 16 layers | OLMoE precedent + Allen-Zhu analysis | Medium — trade-off, not unique optimum |
| 2048 hidden | N_active / (depth × overhead) | Medium — follows from depth choice |

---

## 11. Risk Register: What Could Fail

### Critical risks (would invalidate core results)

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Expert collapse despite aux-loss-free** | Low (10%) | Catastrophic — model is 200M dense | H_load monitoring every step; alert at H_load < 2 bits; manual bias reset protocol |
| **muP transfer fails for MoE** | Medium (30%) | Scientific — no HP transfer validation | Document which rule breaks; fall back to per-scale tuning; still a novel negative result |
| **BF16 numerical instability** | Low (15%) | Training diverges | Grad clip 1.0; QK-norm in MLA; stability ablations (runs A, C, D) |

### Important risks (degrade quality but don't invalidate)

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Krajewski G≈29 is suboptimal at our scale** | Medium (25%) | ~0.02-0.05 BPB penalty | G is within optimal range; post-hoc analysis can quantify |
| **2 shared experts is worse than 1** | Medium (30%) | ~0.01 BPB penalty + wasted params | Can ablate as single-variable experiment |
| **MTP λ schedule suboptimal** | Medium (25%) | Reduced MTP acceptance rate | Schedule 0.3→0.1 at 60% (config.py default); fallback: constant λ=0.3 |
| **DSA indexer doesn't converge at 1B** | Medium (20%) | No long-context capability | Phase 1 (dense, 4K) still produces a useful model |

### Dependency risks (external factors)

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **H100 availability/pricing changes** | Low (10%) | Budget overrun | Can train on A100 (slower); spot pricing buffer |
| **Training data quality issues** | Medium (20%) | Higher L_irr than expected | Pre-training diversity audit (Allen-Zhu Part 3.1) |
| **Chinchilla ratio is wrong for MoE** | Low (15%) | Over/under-training | Ludziejewski MoE scaling law accounts for this; monitor loss curvature |

---

## 12. What "Done" Looks Like

The project is complete when these five conditions hold simultaneously:

### 1. Trained NanoSeek-1B checkpoint
- EMA weights available
- Final ema_val_bpb measured on held-out set
- H_load > 2 bits at final step (no expert collapse)
- MTP acceptance rate > 75%

### 2. muP HP transfer validated
- 3-point path: anchor (55M) → 500M → 1B
- 500M trained with auto-scaled HPs from anchor
- 1B trained with auto-scaled HPs from anchor
- Report documenting which rules held and which (if any) broke
- File: `scaling_law_lab/report/HP_TRANSFER_REPORT.md`

### 3. Stability ablations complete
- Run A: Full config (baseline)
- Run C: No QK-norm (isolates QK-norm contribution)
- Run D: No aux-loss-free (isolates load balancing contribution)
- Each with H_load and I_spec trajectories
- SwiGLU confound documented (Allen-Zhu Part 3.3)
- File: `stability_engine/report/STABILITY_PLAYBOOK.md`

### 4. Scaling law fit
- L(N_active, D, E) = L_irr + A/N_active^α + B/D^β + γ·log(E)
- Fit from EMA validation BPB across sweep points
- R² > 0.95
- Knowledge capacity analysis (Allen-Zhu Part 3.3)
- Depth-stratified analysis (Allen-Zhu Parts 1, 2.1)

### 5. 3-stage RL post-training
- Stage 1: Reasoning RL (GRPO, 60% budget)
- Stage 2: Agent RL (GRPO, 25% budget)
- Stage 3: General Alignment (DPO, 15% budget)
- 3 compute budgets: 2%, 5%, 10% of pre-training compute
- All 4 V3.2 MoE stabilization techniques active
- Test-time scaling curve: accuracy vs inference tokens
- MTP acceptance as test-time scaling signal
- File: `docs/RL_SCALING_REPORT.md`

---

## 13. Complete Citation List

### Primary architectural references

1. **DeepSeek V3** — DeepSeek-AI (2024). "DeepSeek-V3 Technical Report." arXiv:2412.19437.
   *Source for: MLA, routing, load balancing, MTP, DSA, SwiGLU sizing.*

2. **Chinchilla** — Hoffmann, J. et al. (2022). "Training Compute-Optimal Large Language Models." arXiv:2203.15556.
   *Source for: N_active and D derivation from compute budget.*

3. **Krajewski Granularity** — Krajewski, J. et al. (2024). "Scaling Laws for Fine-Grained Mixture of Experts." ICML 2024.
   *Source for: Expert granularity G, moe_intermediate_size derivation.*

4. **OLMoE** — Muennighoff, N. et al. (2024). "OLMoE: Open Mixture-of-Experts Language Models." arXiv:2409.02060.
   *Source for: E=64, κ=12.5% at 1B scale, d/L=128.*

5. **muP** — Yang, G. et al. (2022). "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer." arXiv:2203.03466.
   *Source for: HP transfer rules across widths.*

6. **μP-MoE** — (2025). "Maximal Update Parameterization for Mixture of Experts." arXiv:2508.09752.
   *Source for: MoE-specific muP rules (κ-constant, expert LR scaling).*

7. **Complete(d)P** — (2025). "Completing muP: Batch Size and Training Duration Transfer." arXiv:2512.22382.
   *Source for: Batch size √B scaling, training duration transfer.*

### Scaling law and MoE theory

8. **Neural Scaling Laws** — Kaplan, J. et al. (2020). "Scaling Laws for Neural Language Models." arXiv:2001.08361.
   *Source for: Power-law relationship between loss, parameters, and data.*

9. **MoE Scaling Laws** — Ludziejewski, K. et al. (2025). "Scaling Laws for Mixture of Experts." ICML 2025.
   *Source for: L(N_active, D, E) formulation; N_active as primary scaling variable.*

10. **Switch Transformer** — Fedus, W. et al. (2022). "Switch Transformers: Scaling to Trillion Parameter Models." JMLR.
    *Source for: Expert collapse analysis, MoE compute advantage.*

11. **DeepSeekMoE** — DeepSeek-AI (2024). "DeepSeekMoE: Towards Ultimate Expert Specialization." arXiv:2401.06066.
    *Source for: Shared expert design at smaller scale (2 shared experts precedent).*

### Depth, width, and knowledge capacity

12. **Physics of Language Models Part 1** — Allen-Zhu, Z. & Li, Y. (2024). "Physics of Language Models: Knowledge Storage, Extraction, and Manipulation."
    *Source for: 2 bits/param knowledge capacity limit.*

13. **Physics of Language Models Part 2.1** — Allen-Zhu, Z. & Li, Y. (2024). "Physics of Language Models: Grade-School Math and the Hidden Reasoning Process."
    *Source for: Depth determines reasoning hops; internal planning precedes token generation.*

14. **Physics of Language Models Part 3.3** — Allen-Zhu, Z. & Li, Y. (2024). "Physics of Language Models: Knowledge Capacity Scaling Laws."
    *Source for: Knowledge capacity is 2 bits/param; accessibility depends on data diversity.*

15. **Physics of Language Models Part 4.1** — Allen-Zhu, Z. & Li, Y. (2025). "Physics of Language Models: Depo, Brevo, Capo, Mano, Lano."
    *Source for: Depth vs width benchmarks; canon layers.*

### Training techniques

16. **FIM** — Bavarian, M. et al. (2022). "Efficient Training of Language Models to Fill in the Middle." arXiv:2207.14255.
    *Source for: 10% PSM format for fill-in-the-middle capability.*

17. **EMA (Polyak Averaging)** — Polyak, B. T. & Juditsky, A. B. (1992). "Acceleration of Stochastic Approximation by Averaging." SIAM J. Control Optim.
    *Source for: Theoretical foundation — averaged iterates converge at O(1/T) regardless of LR.*

18. **EMA Dynamics** — Morales-Brotons, D. et al. (2024). "Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits." TMLR. arXiv:2411.18704.
    *Source for: EMA as implicit regularizer; generalization, calibration, and transfer learning benefits across 7 architectures.*

19. **Early Weight Averaging for LLMs** — Sanyal, S. et al. (2023). "Early Weight Averaging meets High Learning Rates for LLM Pre-training." COLM 2024. arXiv:2306.03241.
    *Source for: Weight averaging speeds up LLM pretraining across all workloads; enables higher learning rates.*

20. **YaRN** — Peng, B. et al. (2023). "YaRN: Efficient Context Window Extension of Large Language Models." arXiv:2309.00071.
    *Source for: RoPE interpolation for Phase 2 context extension.*

21. **Gemma 2** — Google DeepMind (2024). "Gemma 2: Improving Open Language Models at a Practical Size."
    *Source for: Logit softcap (tanh squash), post-embedding normalization, CastLinear pattern.*

22. **GPT-3** — Brown, T. B. et al. (2020). "Language Models are Few-Shot Learners." NeurIPS 2020. arXiv:2005.14165.
    *Source for: Width-dependent initialization std = 1/√(d_model) for variance-preserving forward pass.*

---

## 14. Modern Training Techniques (nanochat-derived)

Five techniques adapted from nanochat/gpt.py (Karpathy's training codebase) after compatibility analysis against our MLA + MoE + muP architecture. Each addresses an orthogonal failure mode; together they compose without interference.

### 14.1 CastLinear (Dtype-Casting Linear)

**What**: `nn.Linear` subclass that keeps master weights in fp32 for optimizer precision but casts to activation dtype (bf16) in `forward()` via `F.linear(x, self.weight.to(x.dtype))`.

**Why**: With 192+ Linear layers in our MoE architecture (64 experts × 3 weights each + MLA projections + router + MTP), autocast overhead is non-trivial. CastLinear eliminates the autocast bookkeeping entirely — the cast is explicit, one-shot, and fused into the matmul. Master weights stay fp32 so AdamW moments don't lose precision, which matters especially for the router weight where gradient signal is sparse (only top-8 experts get nonzero gradients per token).

**Provenance**: nanochat/gpt.py lines 45–50. Similar pattern used in Gemma 2 and PaLM for large-scale training stability.

**muP compatibility**: Transparent — muP scaling factors apply to optimizer LR, not to the Linear forward. CastLinear doesn't change the mathematical operation, only the precision path.

### 14.2 Zero-Init Output Projections

**What**: Initialize `wo.weight` (attention output), `w_down.weight` (FFN/expert output), and `concat_proj.weight` (MTP fusion) to zeros. Every other weight uses normal init.

**Why**: At initialization, each transformer block becomes an identity function (residual + 0 = residual). This is especially critical for MoE: without zero-init, you have random routing (router is untrained) selecting random experts whose random projections perturb the residual stream — double randomness that can cause early training instability. Zero-init means the model starts as an embedding → norm → lm_head pipeline, and experts "fade in" as training progresses.

**Provenance**: GPT-2 (Radford et al., 2019) used scaled residual init; nanochat/gpt.py lines 223–225 uses explicit zero-init. The MoE justification follows from DeepSeek V3's analysis of expert load collapse in early training.

**muP compatibility**: Zero-init is a constant initializer, so it composes with muP's per-layer LR scaling — the optimizer will grow these weights from zero at the muP-prescribed rate.

### 14.3 Logit Softcap

**What**: After `lm_head`, apply `cap × tanh(logits / cap)` with `cap = 15.0`, computed in fp32. This squashes logits to [−15, 15] with a smooth tanh boundary.

**Why**: MoE expert specialization can cause logit explosion — when a subset of experts becomes highly confident on certain tokens, their concentrated output through `lm_head` produces extreme logits. Unlike simple clipping, tanh softcap has smooth gradients everywhere (no gradient discontinuity at the boundary), so it doesn't interfere with learning. The fp32 upcast prevents bf16 overflow in the tanh computation.

**Provenance**: Gemma 2 (Google DeepMind, 2024). nanochat/gpt.py lines 421–425. Cap value of 15.0 is empirically standard — large enough not to constrain normal logits (typical range ±5–8) but small enough to prevent explosion.

**muP compatibility**: Logit softcap acts after the forward pass, before loss. muP doesn't prescribe logit scaling (it scales learning rates and init), so there's no interference.

### 14.4 Post-Embedding Norm

**What**: Parameterless `F.rms_norm` applied immediately after `embed_tokens` lookup, before entering layer 0. No learnable parameters — pure normalization.

**Why**: Embedding vectors have magnitude that varies with vocabulary frequency distribution. High-frequency tokens (articles, punctuation) get larger gradient updates and develop larger norms than rare tokens. Without normalization, layer 0 receives inputs with inconsistent scale, which the first RMSNorm (pre-attention) must compensate for. A parameterless norm before layer 0 gives consistent activation scale from token 1, reducing the burden on learned norms.

**Provenance**: nanochat/gpt.py line 412. Gemma 2 uses a similar approach. We use parameterless (no learnable γ) to avoid adding parameters that could interact with muP scaling.

**muP compatibility**: Parameterless norm has no learnable parameters, so there's nothing for muP to scale. It normalizes the embedding output to unit RMS, giving a consistent starting point regardless of hidden_size — which actually helps muP transfer by removing a source of width-dependent variance.

### 14.5 Width-Dependent Init Std

**What**: Initialize all Linear and Embedding weights with `std = 1/√hidden_size` instead of fixed `std = 0.02`.

**Why**: The variance-preserving principle: if input has unit variance and weights are N(0, 1/√fan_in), the output also has approximately unit variance. With fixed `std = 0.02`, the mismatch is width-dependent: at anchor scale (hidden=480), `1/√480 ≈ 0.0456` vs `0.02` — a 2.3× gap. At 1B scale (hidden=2048), `1/√2048 ≈ 0.0221` vs `0.02` — nearly matched. This means fixed init works differently at different scales, which is exactly what muP transfer wants to avoid. Width-dependent init makes the activation scale consistent across all widths.

**Provenance**: GPT-3 (Brown et al., 2020) used `1/√(2·n_layers·hidden)` for residual projections. nanochat/gpt.py lines 217–224. We use the simpler `1/√hidden_size` combined with zero-init output projections (which handles the depth scaling).

**muP compatibility**: Strongly positive. muP's width scaling rule `η ∝ 1/width` for hidden weights assumes variance-preserving init. Fixed `std = 0.02` violates this assumption at non-target widths. Width-dependent init means the anchor, 500M, and 1B configs all start with the same activation dynamics, making muP transfer more reliable.

### Summary of Interaction Effects

| Technique | Failure mode addressed | Interacts with |
|-----------|----------------------|----------------|
| CastLinear | Autocast overhead, optimizer precision | None (transparent) |
| Zero-init outputs | MoE double-random perturbation at init | Width-dependent init (complementary) |
| Logit softcap | MoE expert specialization → logit explosion | None (post-forward) |
| Post-embedding norm | Width-dependent embedding magnitude | Width-dependent init (complementary) |
| Width-dependent init | Scale-dependent activation dynamics | muP transfer (enabling) |

The five techniques compose cleanly: CastLinear and logit softcap are isolated (precision path and post-forward respectively). Zero-init + width-dependent init are complementary (zero-init handles output projections, width-dependent init handles everything else). Post-embedding norm + width-dependent init both serve scale-consistency but at different points (embedding output vs all weights).

---

## 15. Tokenizer Strategy: 32K Vocab for Nano-Scale MoE

### The Problem: Embedding Tax

The embedding table (input `embed_tokens` + output `lm_head`, untied) scales as `vocab_size × hidden_size × 2`. At nano scale, this can dominate the model:

| Scale | hidden | 65K vocab embed | % of active | 32K vocab embed | % of active |
|-------|--------|-----------------|-------------|-----------------|-------------|
| Anchor | 480 | 63M | **114%** | 31M | **57%** |
| 500M | 1280 | 168M | 38% | 84M | 21% |
| 1B | 2048 | 268M | 24.8% | 134M | 14.2% |

At 65K vocab, the anchor's embedding table is *larger than the entire model's active capacity*. This distorts muP transfer because the anchor measures embedding learning dynamics, not transformer dynamics.

### Decision: vocab_size = 32,768

**Why 32K over 65K:**
1. **muP transfer reliability** — Embedding:transformer ratio is more consistent across scales. At 32K, anchor embed is 57% of active (still high, but manageable) vs 114% at 65K.
2. **134M freed params at 1B** — These params go into transformer capacity (experts, MLA) rather than storing rare token embeddings.
3. **MTP acceptance rate** — Smaller vocab → fewer possible continuations → easier next-token prediction → higher speculative decoding acceptance rate. MTP acceptance is a measured scientific output (RULE 9).
4. **Number tokenization** — The `\p{N}{1,2}` split pattern (grouping at most 2 digits) is validated optimal for ≤32K vocab (nanochat LOG.md). 3-digit grouping wastes token space on rare combos at this vocab size.

**Why not 16K:**
At 16K vocab, bytes-per-token would drop below 2.5, meaning 22B tokens covers less text. The compression efficiency loss exceeds the embedding savings. 32K is the sweet spot where compression remains adequate (~2.8-3.2 bytes/token) and embedding tax is manageable.

**Trade-off accepted:** ~15% fewer bytes per token vs 65K. At 22B training tokens, this means covering less raw text. However, the freed 134M params provide more model capacity per byte seen, which should compensate.

### Implementation Details

1. **GPT-4 style BPE** with `byte_fallback=True` — no OOV tokens
2. **Split pattern**: `\p{N}{1,2}` (nanochat-validated for 32K, not GPT-4's `\p{N}{1,3}`)
3. **Vocab padding**: Model pads to multiple of 64 at init for matmul alignment (nanochat pattern). `lm_head` output is sliced back to `config.vocab_size` in forward.
4. **14 special tokens**: bos, eos, 3 FIM (prefix/suffix/middle), 8 chat/RL (user/assistant/python/output), pad
5. **FIM from token 1** (RULE 6): PSM format tokens are in the tokenizer from the start
6. **Two backends**: HuggingFace (for BPE training), RustBPE+tiktoken (for fast inference)

### Provenance

- **Vocab size decision**: First-principles analysis of embedding tax at nano scale
- **Split pattern**: nanochat/tokenizer.py (Karpathy), validated in nanochat/dev/LOG.md
- **Vocab padding**: nanochat/gpt.py lines 168-170 (pad to multiple of 64, slice in forward)
- **BPE architecture**: GPT-4 style with byte fallback (Radford et al., 2019)

---

*This document is a living record. As training proceeds and results come in, sections will be updated with empirical findings. The honest provenance audit (Section 10) will be revised if any attribution is found to be incorrect.*
