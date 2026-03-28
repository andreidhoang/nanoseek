# MiniMax M2.7 RL Pipeline — Deep Technical Reconstruction
## Comprehensive Analysis from Text-01 through M2.7
### Compiled: 2026-03-24 | Classification: [VERIFIED], [INFERRED-STRONG], [INFERRED-WEAK], [UNKNOWN]

---

## Sources Used

### Primary Sources (Official Papers & Blogs)
- [MiniMax-01 (Text-01): arXiv 2501.08313](https://arxiv.org/abs/2501.08313)
- [MiniMax-M1: arXiv 2506.13585](https://arxiv.org/abs/2506.13585)
- [MiniMax M2.1 Post-Training Insights](https://www.minimax.io/news/post-training-experience-and-insights-for-agent-models)
- [MiniMax M2.5 Announcement](https://www.minimax.io/news/minimax-m25)
- [MiniMax M2.7 Announcement](https://www.minimax.io/news/minimax-m27-en)
- [Forge Framework Blog](https://www.minimax.io/news/forge-scalable-agent-rl-framework-and-algorithm)
- [CISPO Documentation (Swift)](https://swift.readthedocs.io/en/latest/Instruction/GRPO/AdvancedResearch/CISPO.html)

### Comparison Sources (Competitor Algorithms)
- [DISPO: arXiv 2602.00983](https://arxiv.org/abs/2602.00983)
- [STAPO: arXiv 2602.15620](https://arxiv.org/abs/2602.15620)
- [DAPO: arXiv 2503.14476](https://arxiv.org/abs/2503.14476)

---

# 1. FACTS — Verified Information with Sources

## Architecture Facts

| Fact | Value | Source | Classification |
|------|-------|--------|----------------|
| Text-01 total params | 456B | arXiv 2501.08313 | [VERIFIED] |
| Text-01 active params/token | 45.9B | arXiv 2501.08313 | [VERIFIED] |
| M2.x total params | 230B | HuggingFace model card, API docs | [VERIFIED] |
| M2.x active params/token | ~10B | MiniMax blog, API pricing analysis | [VERIFIED] |
| Attention mechanism | Hybrid: 7 lightning + 1 softmax per 8 layers | arXiv 2501.08313 | [VERIFIED] |
| Total experts | 32 | arXiv 2501.08313 | [VERIFIED] |
| Active experts (top-k) | 2 | arXiv 2501.08313 | [VERIFIED] |
| Expert hidden dim | 9,216 | arXiv 2501.08313 | [VERIFIED] |
| Shared experts | None | arXiv 2501.08313 | [VERIFIED] |
| Layers (Text-01) | 80 | arXiv 2501.08313 | [VERIFIED] |
| Hidden dim (Text-01) | 6,144 | arXiv 2501.08313 | [VERIFIED] |
| Vocab size | 200K | arXiv 2501.08313 | [VERIFIED] |
| Load balancing | Aux loss (α=0.001) + token drop | arXiv 2501.08313 | [VERIFIED] |
| Context (train) | 1M tokens | arXiv 2501.08313 | [VERIFIED] |
| Context (inference) | 4M tokens | arXiv 2501.08313 | [VERIFIED] |
| M2.7 context | 205K tokens | API docs | [VERIFIED] |
| Positional encoding | RoPE (softmax layers only, half head dim) | arXiv 2501.08313 | [VERIFIED] |

## RL Training Facts

| Fact | Value | Source | Classification |
|------|-------|--------|----------------|
| RL algorithm | CISPO | arXiv 2506.13585 | [VERIFIED] |
| RL GPUs (M1) | 512 H800 | arXiv 2506.13585 | [VERIFIED] |
| RL duration (M1) | 3 weeks | arXiv 2506.13585 | [VERIFIED] |
| RL cost (M1) | ~$534,700 | arXiv 2506.13585 | [VERIFIED] |
| Adam epsilon | 1e-15 | arXiv 2506.13585 | [VERIFIED] |
| Adam beta1, beta2 | 0.9, 0.95 | arXiv 2506.13585 | [VERIFIED] |
| FP32 LM head | Required for RL stability | arXiv 2506.13585 | [VERIFIED] |
| IS weight clip (ε_high) | 5.0 | CISPO docs + paper | [VERIFIED] |
| Repetition detection | 3000 tokens > 0.99 prob → truncate | arXiv 2506.13585 | [VERIFIED] |
| CISPO vs DAPO speed | 2× faster (50% steps) | arXiv 2506.13585 | [VERIFIED] |
| M1 thinking budgets | 40K and 80K | arXiv 2506.13585 | [VERIFIED] |
| Gradient magnitude range | 1e-18 to 1e-5 | arXiv 2506.13585 | [VERIFIED] |
| FP32 fix correlation improvement | ~0.9x → 0.99x (train vs inference probs) | arXiv 2506.13585 | [VERIFIED] |

## M2.7 Self-Evolution Facts

| Fact | Value | Source | Classification |
|------|-------|--------|----------------|
| Self-evolving rounds | 100+ autonomous rounds | M2.7 blog | [VERIFIED] |
| Performance improvement | 30% from self-evolution alone | M2.7 blog | [VERIFIED] |
| RL workflow automation | 30-50% of RL research workflow | M2.7 blog, VentureBeat | [VERIFIED] |
| Complex skills built | 40+ with 97% compliance | M2.7 blog | [VERIFIED] |
| Self-evolving loop | Analyze→Plan→Modify→Evaluate→Compare→Keep/Revert | M2.7 blog | [VERIFIED] |

---

# 2. RL ALGORITHM — CISPO

## 2.1 Mathematical Formulation

[VERIFIED — arXiv 2506.13585, CISPO Swift docs]

### Standard GRPO Loss (for comparison)

```
L_GRPO(θ) = -E[ min( r_t(θ) · Â_t,  clip(r_t(θ), 1-ε, 1+ε) · Â_t ) ]
```

Where `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)` is the importance sampling ratio.

**Problem**: When `r_t(θ)` is large (token's probability changed significantly between old
and new policy), the `clip()` operation kills the gradient entirely. This disproportionately
affects rare but important reasoning tokens ("However," "Wait," "Recheck") which have
high policy ratio variance.

### CISPO Loss

```
L_CISPO(θ) = -E[ detach(min(r_t(θ), ε_high)) · Â_t · log π_θ(a_t|s_t) ]
```

Where:
- `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)` — importance sampling ratio
- `ε_high = 5.0` — upper clip bound (much larger than PPO's 0.2)
- `detach()` — stops gradient flow through the IS weight
- `Â_t` — advantage (mean reward baseline, no value network)
- `log π_θ(a_t|s_t)` — token log probability (where gradients flow)

### Python Pseudocode

```python
# CISPO core (simplified from arXiv 2506.13585)
log_ratio = per_token_logps - old_per_token_logps          # shape: [B, T]
importance_weights = torch.exp(log_ratio)                   # IS ratio
clamped_weights = torch.clamp(importance_weights, max=5.0)  # upper clip only
clamped_weights = clamped_weights.detach()                  # KEY: no gradient through weights

# Advantage: mean reward baseline (no value network)
advantages = rewards - rewards.mean()                       # shape: [B]
advantages = advantages.unsqueeze(-1).expand_as(per_token_logps)  # broadcast

# Loss: gradients flow ONLY through log_probs
per_token_loss = -clamped_weights * advantages * per_token_logps
loss = per_token_loss.sum() / valid_token_count
```

### Why .detach() Matters

[VERIFIED — arXiv 2506.13585 Section on CISPO]

In GRPO/PPO, the gradient of `clip(r_t, 1-ε, 1+ε) * Â_t` is:
- **Zero** when r_t is outside the clip range
- This kills gradient signal for any token whose probability shifted a lot

In CISPO, `detach(min(r_t, ε_high))` acts as a **constant scalar multiplier**:
- Gradients always flow through `log π_θ(a_t|s_t)` for EVERY token
- The IS weight just re-scales the gradient magnitude (like a learning rate per token)
- Rare tokens that shifted a lot still get gradients, just bounded by ε_high=5.0

### Key Properties

| Property | CISPO | GRPO/PPO |
|----------|-------|----------|
| Value network | No | No (GRPO) / Yes (PPO) |
| Trust region | No (abandoned) | Yes (core mechanism) |
| Gradient for all tokens | Yes (main innovation) | No (clipped tokens get zero) |
| IS weight gradient | Detached (constant) | Part of gradient computation |
| KL regularization | Implicit (IS weight clipping) | Explicit (PPO) / Implicit (GRPO) |
| Clip bounds | Upper only: ε_high=5.0 | Symmetric: [1-ε, 1+ε], ε=0.2 |
| Bias | Slightly biased (acknowledged) | Less biased within trust region |
| Long response handling | Better (all tokens contribute) | Worse (rare tokens suppressed) |

## 2.2 Comparison with DISPO

[VERIFIED — arXiv 2602.00983]

DISPO (Decoupled Importance Sampling-weighted Policy Optimization) is a direct improvement
on CISPO that identifies and fixes specific failure modes.

### DISPO's Four Regimes

DISPO assigns **distinct clipping bounds** based on two axes:
1. Reward sign: correct vs incorrect response
2. IS weight relative to 1: above vs below

| Regime | Condition | Controls | Effect |
|--------|-----------|----------|--------|
| R1 | Correct + IS < 1 | Lower clip for correct | Distillation (reinforce known-good) |
| R2 | Correct + IS > 1 | Upper clip for correct | Exploration (try new correct paths) |
| R3 | Incorrect + IS < 1 | Lower clip for incorrect | Forgetting bad patterns |
| R4 | Incorrect + IS > 1 | Upper clip for incorrect | Suppression of bad exploration |

### DISPO vs CISPO Performance

[VERIFIED — arXiv 2602.00983]

| Benchmark | CISPO | DAPO | DISPO |
|-----------|-------|------|-------|
| AIME'24 (Avg@16) | 55.42% | 50.21% | **61.04%** |

DISPO achieves 10% absolute improvement over CISPO on AIME'24 by properly balancing
exploration and distillation regimes.

### Why CISPO Fails (DISPO's Analysis)

[VERIFIED — arXiv 2602.00983]

DISPO identifies that CISPO's uniform upper clipping causes:
1. **Insufficient exploration**: Same ε_high for both correct and incorrect responses
   doesn't distinguish between "try new correct paths" and "suppress bad exploration"
2. **Imbalanced distillation**: CISPO treats all IS<1 tokens the same regardless of
   reward sign, missing the opportunity to reinforce known-good patterns more strongly

## 2.3 Comparison with STAPO

[VERIFIED — arXiv 2602.15620]

STAPO (Stabilizing Reinforcement Learning for LLMs by Silencing Rare Spurious Tokens)
identifies a different failure mode that affects CISPO.

### The Spurious Token Problem

[VERIFIED — arXiv 2602.15620]

- ~0.01% of tokens are "spurious": low probability, low entropy, positive advantage
- These tokens contribute nothing to reasoning but inherit full sequence-level reward
- Because they're rare (low prob), their IS weights are very large
- CISPO's detach(min(r_t, 5.0)) still allows weights up to 5.0 for these tokens
- Result: **abnormally amplified gradient updates** from meaningless tokens

### STAPO's Solution: S2T (Silencing Spurious Tokens)

```python
# STAPO spurious token detection
is_spurious = (
    (token_prob < threshold_prob) &      # low probability
    (token_entropy < threshold_entropy) & # low entropy
    (advantage > 0)                       # positive advantage (correct response)
)
# Suppress gradients from spurious tokens
per_token_loss[is_spurious] = 0.0
```

### CISPO's Entropy Collapse (STAPO's Finding)

[VERIFIED — arXiv 2602.15620]

CISPO exhibits **model collapse** in later training stages:
- Performance declines sharply
- Entropy drops rapidly
- Root cause: "constraining gradient norms alone is insufficient to ensure training
  stability when gradients from all tokens are retained"
- The very property that makes CISPO good (all tokens get gradients) also makes it
  vulnerable to spurious token amplification

### STAPO Performance

[VERIFIED — arXiv 2602.15620]

Average improvement over GRPO: +7.13% and +3.69% across six math benchmarks.
Superior entropy stability compared to CISPO, GRPO, and JustRL.

## 2.4 Algorithm Family Tree

```
REINFORCE (Williams 1992)
  └── PPO (Schulman 2017) — trust region + value network
       └── GRPO (DeepSeek, 2024) — no value network, group relative advantage
            ├── DAPO (ByteDance, 2025) — decoupled clip + dynamic sampling
            ├── CISPO (MiniMax, 2025) — detached IS weight clipping
            │    ├── DISPO (2026) — 4-regime decoupled clipping (fixes CISPO exploration)
            │    └── STAPO (2026) — spurious token silencing (fixes CISPO collapse)
            └── ESPO/ASPO/CE-GPPO — other variants addressing entropy
```

---

# 3. TRAINING PIPELINE — Stages, Self-Evolving Loop, Multi-Scaffold

## 3.1 M1 RL Training Pipeline (Most Detailed)

[VERIFIED — arXiv 2506.13585]

### Stage 1: Pre-Training (Text-01 Base)

- 7.5T tokens from reasoning-intensive corpus
- Context length extended in 4 stages: 32K → 128K → 512K → 1M tokens
- Gradient clipping decreased during 80K extension phase
- Lightning attention: O(n) for 87.5% of layers, O(n²) for 12.5%
- Training on H800 GPUs (pre-training cluster size ~2,000 H800) [INFERRED-STRONG]

### Stage 2: Cold-Start SFT

- Inject Chain-of-Thought (CoT) patterns
- Reflection-based reasoning from high-quality, long CoT examples
- Diverse domains covered
- Purpose: give the model the format/structure for reasoning before RL

### Stage 3: RL Training (CISPO)

**Phase 3a: Reasoning-Only RL**
- Start exclusively with reasoning-intensive tasks
- Math: ~50K curated problems (filtered: 0 < pass@10 < 0.9)
- Logic: ~53K problems via SynLogic (41 task types)
- Code: ~30K competitive programming problems
- Rule-based rewards (binary correct/incorrect)

**Phase 3b: Gradual Domain Mixing**
- Progressively add general domain tasks
- SWE: GitHub issues, containerized sandbox execution
- General domain: ~25K samples (STEM, factual, instruction, creative)
- Introduce GenRM for non-verifiable tasks (5-grade scale)

**Phase 3c: Full RL with GenRM Monitoring**
- Open-ended conversations
- Continuous online monitoring of length bias
- GenRM recalibration triggered on detecting length-seeking behavior
- Reward shaping + value clipping + normalization

### Critical Stability Fixes

[VERIFIED — arXiv 2506.13585]

1. **Adam ε = 1e-15** (not default 1e-8): Gradient magnitudes range 1e-18 to 1e-5;
   default ε would zero out the smallest gradients entirely
2. **FP32 LM head**: Train-inference probability correlation was ~0.9x with BF16/FP16;
   upgrading to FP32 improved to 0.99x. Without this fix, rewards plateau.
3. **Repetition truncation**: 3000 consecutive tokens with >0.99 probability → halt
   generation. Prevents instability from pathological loops.

## 3.2 M2.x Agentic RL Pipeline

[VERIFIED — M2.1 blog, Forge blog, M2.5 blog]

### Key Discovery: Multi-Scaffold Training is Essential

[VERIFIED — M2.1 Post-Training blog]

Training on a single scaffold (e.g., simple ReAct loop) severely limits generalization.
Different scaffolds introduce different context management and execution logic. The model
must adapt to:
- Scaffolds that discard historical thinking content
- Scaffolds with different tool-calling formats
- Scaffolds with varying context compression strategies

**Solution**: Train across hundreds of scaffold types simultaneously.

### SFT Stage: Multi-Scaffold Rejection Sampling

[VERIFIED — M2.1 blog]

- Generate trajectories across multiple scaffold types
- Reject poor trajectories using execution-based verification
- Keep trajectories where agent successfully completes task regardless of scaffold

### RL Stage: Forge Framework

[VERIFIED — Forge blog, M2.5 blog]

**Scale**:
- 100K+ distinct real-world agent scaffolds and environments
- Context lengths up to 200K tokens
- Millions of samples processed daily
- Hundreds of scaffold types, thousands of tool invocation formats

**Four Required Interfaces per Scaffold**:
1. `reprocess` — prepare trajectory for training
2. `run` — execute agent in environment
3. `postprocess` — format results
4. `calculate_reward` — compute reward signal

**Reward Types for Agentic RL**:
- F2P (Fail-to-Pass): Tests that were failing now pass
- P2P (Pass-to-Pass): Tests that were passing still pass
- Sandbox execution: containerized code execution for SWE tasks
- Process reward: end-to-end monitoring of generation quality [VERIFIED — M2.1 blog]

### Data Synthesis Pipeline

[VERIFIED — M2.1 blog, M2.5 blog]

| Source | Method | Scale |
|--------|--------|-------|
| GitHub PRs (real) | Bug injection, difficulty merging | 10K+ PRs |
| Expert-designed tasks | Human-in-the-loop | Thousands |
| SWE-Test pipeline | BugFix→SWE-Test transformation | 140K+ tasks |
| Multi-language | 10+ programming languages | Diverse |
| AppDev tasks | Expert + Playwright verification | [INFERRED-STRONG] |

### Context Rot Solution

[VERIFIED — M2.1 blog]

MiniMax identified "context rot" — attention dilution from accumulated intermediate steps
in long agentic trajectories. Solution: Make context management an **explicit RL action**.
The agent learns when to summarize/compress/discard context, rather than the infrastructure
managing context for it.

## 3.3 M2.7 Self-Evolving Loop

[VERIFIED — M2.7 blog, VentureBeat]

### The Loop

```
1. ANALYZE failure trajectories from current RL training
2. PLAN changes to scaffold code, sampling parameters, workflow
3. MODIFY scaffold code autonomously
4. RUN evaluations on modified scaffold
5. COMPARE results against baseline
6. KEEP improvements or REVERT failed changes
7. REPEAT for 100+ rounds
```

### What M2.7 Self-Optimized

[VERIFIED — M2.7 blog]

1. **Sampling parameters**: Systematically searched optimal temperature, frequency penalty,
   presence penalty combinations
2. **Workflow guidelines**: Designed more specific instructions for itself
3. **Loop detection**: Added optimizations to detect and break infinite agent loops
4. **Agent harness architecture**: Built research agent harness supporting data pipelines,
   training environments, infrastructure, cross-team collaboration, persistent memory
5. **40+ complex skills**: Built with 97% compliance rate

### Quantitative Results

- **30% performance improvement** on internal evaluation sets from self-evolution alone
- **30-50% of RL research workflow** handled autonomously
- **100+ autonomous rounds** without human intervention

---

# 4. INFRASTRUCTURE

## 4.1 Pre-Training Infrastructure

[VERIFIED — arXiv 2501.08313, arXiv 2506.13585]

| Component | Detail |
|-----------|--------|
| GPU type | NVIDIA H800 (also H20 mentioned for some experiments) |
| Pre-training cluster | ~2,000 H800 [INFERRED-STRONG from blog mentions] |
| MFU | >75% on H20 GPUs |
| Sequence parallelism | LASP+ (Linear Attention Sequence Parallelism) |
| Training tokens | 7.5T for Text-01 base |
| Context stages | 32K → 128K → 512K → 1M |
| CUDA kernels | Custom optimized for lightning attention |

## 4.2 RL Training Infrastructure (M1)

[VERIFIED — arXiv 2506.13585]

| Component | Detail |
|-----------|--------|
| GPU count | 512 H800 |
| Duration | 3 weeks |
| Cost | $534,700 (rental) |
| Optimizer | AdamW (β1=0.9, β2=0.95, ε=1e-15) |
| LM head precision | FP32 (critical fix) |
| Thinking budgets | 40K tokens (intermediate), 80K tokens (final) |

## 4.3 Forge RL Framework Infrastructure (M2.x)

[VERIFIED — Forge blog]

### Architecture

```
Agent Side ──► Middleware (Gateway + FIFO Scheduler) ──► Training/Inference Engine
                     │                                         │
              Windowed FIFO                              Global L3 KV Cache
              Scheduling                                 (DFS structure)
                     │                                         │
              Prefix Tree                               Magi Attention
              Merging (40×)                              Primitives
```

### Key Infrastructure Innovations

**Windowed FIFO Scheduling**:
- Prevents "straggler effect" (easy tasks dominating training queue)
- 30% visibility window with local greedy disorder + global FIFO ordering
- Middle ground between strict synchronous and greedy asynchronous

**Prefix Tree Merging**:
- 40× training speedup
- Multiple completions sharing prefix merged into single tree
- Uses Magi Attention primitives for consistent forward pass
- Tree deconstructed post-forward for normal loss computation
- Reduces memory overhead → longer sequences or larger batches

**Global L3 KV Cache**:
- DFS-structured cache for maximum prefix cache hit rate
- Group-level rollout optimization
- Prevents redundant prefilling in multi-turn agent RL

### Scale

| Metric | Value |
|--------|-------|
| Scaffold types | Hundreds integrated |
| Tool formats | Thousands |
| Real-world scaffolds | 100K+ |
| Context length | Up to 200K tokens |
| Daily throughput | Millions of samples |

## 4.4 M2.7 Infrastructure

[INFERRED-STRONG — not explicitly disclosed]

| Component | Estimate | Basis |
|-----------|----------|-------|
| Active params | ~10B | API analysis, model card |
| Total params | ~230B | HuggingFace, same as M2.x family |
| Context window | 205K tokens | API docs |
| Pricing | $0.30/$1.20 per M tokens (in/out) | API pricing |
| Inference speed | ~100 TPS | Lightning attention |
| RL training GPUs | [UNKNOWN] | Not disclosed for M2.7 |
| RL training duration | [UNKNOWN] | Not disclosed for M2.7 |

---

# 5. BENCHMARK RESULTS

## 5.1 MiniMax M2.7 Benchmarks

[VERIFIED — M2.7 blog, Artificial Analysis]

| Benchmark | M2.7 Score | Comparison | Source |
|-----------|-----------|------------|--------|
| SWE-Pro | 56.22% | Matches GPT-5.3-Codex | M2.7 blog |
| SWE-bench Verified | ~78% | vs Opus 55% | M2.7 blog |
| SWE Multilingual | 76.5% | — | M2.7 blog |
| Multi-SWE-Bench | 52.7% | — | M2.7 blog |
| MLE Bench Lite | 66.6% medal rate | Ties Gemini 3.1, near Opus 4.6 | M2.7 blog |
| AA Intelligence Index | 50 | Tier-1 | Artificial Analysis |
| Hallucination rate | 34% | vs 46% Claude Sonnet | M2.7 blog |
| AIME 2025 | 78.3% | [INFERRED-STRONG from analysis sites] | Third-party |

## 5.2 MiniMax M2.5 Benchmarks

[VERIFIED — M2.5 blog]

| Benchmark | M2.5 Score | Notes |
|-----------|-----------|-------|
| SWE-bench Verified | 80.2% | SOTA at time of release |
| Multi-SWE-Bench | 51.3% | — |
| BrowseComp | 76.3% | With context management |
| Avg runtime | 22.8 min | 37% faster than previous (was 31.3 min) |

## 5.3 MiniMax M2.1 Benchmarks

[VERIFIED — M2.1 blog]

| Benchmark | M2.1 Score | Notes |
|-----------|-----------|-------|
| SWE-bench Verified | >67% | Stable across scaffolds (mini-swe-agent, Droid, Claude Code) |

## 5.4 MiniMax M1 Benchmarks

[VERIFIED — arXiv 2506.13585]

| Benchmark | M1 Score | Notes |
|-----------|---------|-------|
| Context window | 1M tokens (train), 4M (inference) | 8× DeepSeek R1 |
| Thinking budget | 40K / 80K tokens | Two released versions |
| RL training cost | $534,700 | 512 H800, 3 weeks |

---

# 6. UNCERTAINTY MAP

## What Is Known (High Confidence)

| Topic | Confidence | Source Quality |
|-------|-----------|---------------|
| CISPO algorithm math | Very High | Paper + open-source implementation |
| Text-01/M1 architecture | Very High | Full arXiv paper |
| M1 RL training details | Very High | Full arXiv paper |
| Adam ε=1e-15 fix | Very High | Paper + seminar |
| FP32 LM head fix | Very High | Paper + seminar |
| Forge framework design | High | Official blog with diagrams |
| Multi-scaffold training | High | M2.1 blog |
| M2.7 self-evolving concept | High | Official blog |
| M2.7 benchmark numbers | High | Official blog + third-party verification |
| Prefix tree 40× speedup | High | Forge blog |

## What Is Partially Known (Medium Confidence)

| Topic | What We Know | What We Don't |
|-------|-------------|---------------|
| M2.x architecture | 230B total, ~10B active, MoE | Exact layer count, dims for M2.x vs Text-01 |
| M2.7 RL training | Uses CISPO + Forge | Specific RL hyperparams for M2.7 |
| Self-evolving loop | Concept + results (30% gain) | Exact implementation, guardrails, failure rate |
| GenRM details | 5-grade scale, length monitoring | Architecture, training data, accuracy |
| Reward hacking mitigation | GenRM recalibration | Specific detection thresholds |
| M2.x pre-training | Same base as Text-01 presumably | Whether M2.x was retrained from scratch |

## What Is Unknown (Low/No Confidence)

| Topic | Status |
|-------|--------|
| M2.7 RL training compute (GPUs, duration, cost) | [UNKNOWN] |
| M2.7 RL training data composition | [UNKNOWN] |
| M2.7 specific CISPO hyperparameters | [UNKNOWN] |
| Whether M2.7 uses DISPO/STAPO improvements | [UNKNOWN] |
| M2.x MoE expert count (same 32 as Text-01?) | [UNKNOWN — likely same] |
| Value function experiments | [UNKNOWN — paper says no value network] |
| Specific self-evolving loop failure modes | [UNKNOWN] |
| M2.7 safety training details | [UNKNOWN] |
| Whether M2.7 used Muon optimizer (like Kimi) | [UNKNOWN — likely still Adam] |
| Internal evaluation set composition | [UNKNOWN] |
| M2.7 forgetting prevention method | [UNKNOWN — M1 used curriculum mixing] |
| Process reward model details for agent RL | [UNKNOWN — mentioned but not detailed] |

---

# 7. FAILURE MODES

## 7.1 CISPO-Specific Failure Modes

### Entropy Collapse (Late-Stage)

[VERIFIED — STAPO paper, arXiv 2602.15620]

- CISPO exhibits model collapse in later training stages
- Performance declines sharply alongside rapid entropy drop
- Root cause: retaining gradients from ALL tokens (including spurious ones)
  amplifies noise that destabilizes training
- "Constraining gradient norms alone is insufficient when gradients from all
  tokens are retained"

### Spurious Token Amplification

[VERIFIED — STAPO paper]

- ~0.01% of tokens are "spurious" (low prob, low entropy, positive advantage)
- These inherit full sequence-level reward despite contributing nothing to reasoning
- IS weights up to 5.0 amplify their already problematic gradient contributions
- Result: gradient spikes that compound over training

### Exploration-Distillation Imbalance

[VERIFIED — DISPO paper, arXiv 2602.00983]

- CISPO's uniform ε_high=5.0 applies to both correct and incorrect responses
- Doesn't distinguish between:
  - Exploring new correct solution paths (should encourage)
  - Reinforcing known-bad patterns (should suppress harder)
  - Distilling known-good patterns (should reinforce)
- Result: suboptimal learning efficiency (~10% below DISPO on AIME'24)

## 7.2 Agent RL Failure Modes

### Context Rot

[VERIFIED — M2.1 blog]

- Attention dilution from accumulated intermediate steps in long trajectories
- Model's attention spreads across increasingly irrelevant historical context
- Solution: teach model to actively manage context (summarize/compress/discard)

### Scaffold Overfitting

[VERIFIED — M2.1 blog]

- Training on single scaffold severely limits generalization
- Model learns scaffold-specific patterns rather than general agent capabilities
- Some scaffolds discard historical thinking content, causing performance drops
- Solution: multi-scaffold training across hundreds of scaffold types

### Straggler Effect

[VERIFIED — Forge blog]

- In asynchronous RL, easy tasks complete faster and dominate training distribution
- Training skews toward "easy" samples, neglecting hard problems
- Solution: Windowed FIFO scheduling (30% visibility window)

### GenRM Length Bias / Reward Hacking

[VERIFIED — arXiv 2506.13585]

- GenRM develops preference for longer outputs regardless of quality
- Model learns to generate verbose responses to hack higher rewards
- Solution: continuous online monitoring + GenRM recalibration when detected

### Train-Inference Probability Mismatch

[VERIFIED — arXiv 2506.13585]

- BF16/FP16 precision in LM head causes ~10% probability mismatch
- RL reward signal based on training-mode probabilities doesn't transfer to inference
- Result: reward plateau (model can't improve because signals are misaligned)
- Solution: FP32 LM head

### Gradient Magnitude Extremes

[VERIFIED — arXiv 2506.13585]

- Gradient magnitudes span 1e-18 to 1e-5 (13 orders of magnitude)
- Default Adam ε=1e-8 would zero out gradients smaller than 1e-14
- Result: most gradient information lost
- Solution: ε=1e-15

---

# 8. EVOLUTION: Text-01 → M1 → M2.x → M2.7

## Timeline

```
Jan 2025:  Text-01 (MiniMax-01) released — arXiv 2501.08313
           ├── 456B total / 45.9B active
           ├── Hybrid lightning + softmax attention
           ├── 1M context training, 4M inference
           ├── MoE: 32 experts, top-2
           └── Open-source weights

Jun 2025:  M1 released — arXiv 2506.13585
           ├── Same Text-01 base (456B/45.9B)
           ├── CISPO algorithm introduced
           ├── Cold-start SFT → RL pipeline
           ├── 40K and 80K thinking budgets
           ├── 512 H800, 3 weeks, $534K
           └── Open-source weights (first open hybrid-attention reasoning model)

~Sep 2025: M2 released [INFERRED — from blog timeline]
           ├── 230B total / ~10B active (NEW, smaller architecture)
           ├── Focus on agentic capabilities
           └── Introduction of Forge framework (internal)

~Nov 2025: M2.1 released
           ├── 229B params (same M2.x architecture)
           ├── Multi-scaffold training discovery
           ├── Context rot identification + fix
           ├── Multi-language coding (10+ languages)
           ├── SWE-bench >67% (stable across scaffolds)
           └── Open-source weights

Feb 2026:  M2.5 released
           ├── Same M2.x architecture (230B/10B)
           ├── Forge framework publicly documented
           ├── 100K+ scaffolds, 200K context
           ├── SWE-bench Verified 80.2% (SOTA)
           ├── BrowseComp 76.3%
           ├── 37% faster end-to-end runtime
           └── Open-source weights

Mar 2026:  M2.7 released (March 18)
           ├── Same M2.x architecture (230B/10B)
           ├── Self-evolving loop (100+ autonomous rounds)
           ├── 30% improvement from self-evolution
           ├── 30-50% RL research workflow automated
           ├── SWE-Pro 56.22% (matches GPT-5.3-Codex)
           ├── 205K context window
           ├── $0.30/$1.20 per M tokens
           └── PROPRIETARY (not open-source)
```

## Key Evolution Patterns

### Architecture Evolution

```
Text-01/M1 (456B/45.9B) ──► M2.x (230B/10B)
     ↑                            ↑
  Huge, open-source           Smaller, more efficient
  Prove architecture works    Optimize for production
  1M context training         200K context for agents
```

[INFERRED-STRONG] The M2.x series appears to be a separate, smaller model trained
from scratch (or significantly pruned/retrained from Text-01). The 230B/10B vs 456B/45.9B
size difference suggests a deliberate efficiency optimization — 10B active params is
enough for agent tasks at 1/50th the cost of competitors.

### RL Algorithm Evolution

```
Cold-start SFT (M1)
  └── Math/code RLVR only (M1 Phase 3a)
       └── + General domain + GenRM (M1 Phase 3b-c)
            └── + Multi-scaffold training (M2.1)
                 └── + Forge framework + 100K scaffolds (M2.5)
                      └── + Self-evolving loop (M2.7)
```

### Infrastructure Evolution

```
512 H800 for 3 weeks (M1, brute force)
  └── Forge middleware + prefix tree merging (M2.1-M2.5, efficiency)
       └── Self-optimizing training loops (M2.7, automation)
```

### Philosophy Evolution

```
Text-01: "Build the architecture" (lightning attention, 1M context)
M1:      "Build the RL algorithm" (CISPO, training stability)
M2.1:    "Build the agent framework" (multi-scaffold, Forge)
M2.5:    "Scale the framework" (100K scaffolds, production quality)
M2.7:    "Let the model improve itself" (self-evolving, 30-50% autonomous)
```

---

# 9. IMPLICATIONS FOR NANOSEEK

## Directly Adoptable Techniques

| Technique | Effort | Impact | Source |
|-----------|--------|--------|--------|
| CISPO `.detach()` loss | ~1 hour | All tokens get gradients in RL | M1 paper |
| Adam ε=1e-15 | 1 line | Prevents gradient death | M1 paper |
| FP32 LM head during RL | 1 line | Prevents reward plateau | M1 paper |
| Repetition truncation (3K tokens) | ~20 lines | Training stability | M1 paper |
| No value network | Already done | Both MiniMax & Kimi validate this | M1 paper |
| Mean reward baseline | Already done | Simple, effective | M1 paper |

## Worth Investigating

| Technique | Complexity | Rationale |
|-----------|-----------|-----------|
| DISPO 4-regime clipping | Medium | +10% over CISPO on AIME |
| STAPO spurious token silencing | Medium | Fixes CISPO's entropy collapse |
| Multi-scaffold training | High | Essential for agent generalization |
| Prefix tree merging | Very High | 40× speedup but needs custom attention |

## Novel Research Opportunity

**MoE + RL interaction**: Neither MiniMax nor any competitor has published detailed analysis
of how RL training affects expert specialization (I_spec). NanoSeek's MoE diagnostics
(dead expert detection, Gini coefficient, I_spec tracking) could provide the first
systematic study of whether CISPO/DISPO/STAPO variants preserve or disrupt expert
routing during RL post-training.

---

# 10. OPEN QUESTIONS

1. **Does M2.7 use DISPO or STAPO improvements?** Both papers postdate M1's CISPO.
   M2.7 was released March 2026, DISPO/STAPO were February 2026. Timeline is tight.
   [UNKNOWN]

2. **What prevents the self-evolving loop from diverging?** 100+ autonomous rounds of
   self-modification without human intervention risks compounding errors. What guardrails
   exist? [UNKNOWN]

3. **Why did M2.x shrink from 456B to 230B?** Was this efficiency-driven (10B active is
   enough) or did training the full 456B model with agent RL prove too expensive?
   [UNKNOWN — likely efficiency-driven given $0.30/M pricing goal]

4. **Does M2.7's self-evolution generalize?** The 30% improvement was on internal
   evaluation sets. Does this transfer to held-out benchmarks? The SWE-Pro score
   (56.22%) matches GPT-5.3-Codex, suggesting yes. [INFERRED-STRONG]

5. **Why is M2.7 proprietary when M2.5 was open-source?** The self-evolving capability
   may be considered too sensitive/dangerous to open-source, or the self-modified
   scaffolds constitute proprietary training methodology. [INFERRED-WEAK]
