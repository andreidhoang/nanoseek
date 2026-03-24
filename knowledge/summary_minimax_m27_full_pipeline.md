# MiniMax Model Family: Complete Technical Breakdown
## From MiniMax-01 Foundation → M1 Reasoning → M2.1/M2.5 Agent → M2.7 Self-Evolving
### Compiled: March 2026 | Sources: arXiv:2501.08313, arXiv:2506.13585, official blog posts

---

## Model Lineage

```
MiniMax-Text-01 (Jan 2025)  — Foundation: 456B MoE, lightning attention, 1M context
    │
MiniMax-M1 (Jun 2025)       — Reasoning: +7.5T continued PT, CISPO RL, 80K output
    │
MiniMax-M2.1 (late 2025)    — Agent: Forge framework, agentic CISPO, multi-scaffold
    │
MiniMax-M2.5 (early 2026)   — Production agent: 80.2% SWE-Verified, process rewards
    │
MiniMax-M2.7 (Mar 18, 2026) — Self-evolving: autonomous RL research, 100+ iteration loops
```

---

## 1. MiniMax-Text-01 — Foundation Model (arXiv:2501.08313)

### 1.1 Architecture Summary

| Parameter | Value |
|-----------|-------|
| Total parameters | 456B |
| Activated params/token | 45.9B |
| Hidden dimension | 6,144 |
| Layers | 80 |
| Attention heads | 64 (dim 128 each) |
| KV heads | 8 (GQA) |
| Vocabulary | 200,064 tokens (byte-level BPE) |
| MoE experts | 32 routed, **no shared expert** |
| Top-k routing | 2 |
| Expert FFN hidden dim | 9,216 |
| RoPE base | 10,000,000 (applied to half of head dim: 64/128) |
| Normalization | DeepNorm (post-norm with scaled residuals) |
| Context (train) | 1M tokens |
| Context (inference) | 4M tokens (extrapolation) |

### 1.2 Hybrid Attention: Lightning Attention + Softmax

The core architectural innovation. Within every 8 layers:
- **7 layers**: Lightning Attention-2 (linear attention) — "TransNormer" blocks
- **1 layer**: Standard softmax attention — "Transformer" blocks

This gives 70 lightning layers + 10 softmax layers across 80 total.

#### Lightning Attention-2 (Linear Attention)

**Mathematical formulation:**

Standard attention: `O = Norm((Q K^T) V)` — O(n² d) complexity.

Linear variant uses the "right product kernel trick":
```
O = Norm(Q (K^T V))     — O(n d²) complexity
```

During inference, this enables **constant O(d²) complexity per token** via recurrent state:
```
KV_state_t = KV_state_{t-1} + k_t v_t^T     (d × d accumulator)
o_t = Norm(q_t · KV_state_t)
```

**Information capacity**: O(d²/h) vs O(d) for softmax — strictly larger when d > h.

**Divide-and-conquer tiling strategy** (for training):
- **Intra-block**: Standard quadratic attention within each tile (block_size = 256). Handles local interactions accurately.
- **Inter-block**: Linear attention with cumulative KV state across blocks. `O_inter = Lambda · Q_i · KV_cumsum`.

**Lightning attention layer forward pass:**
```python
Q, K, V = SiLU(W_q @ X), SiLU(W_k @ X), SiLU(W_v @ X)  # SiLU activation
Y = lightning_attn(Q, K, V)        # Linear attention (intra+inter tiled)
Output = RMSNorm(Y) * sigmoid(X)   # Gated output
```

Implemented in Triton for I/O-aware GPU optimization.

#### Why Hybrid (Not Pure Linear)

Pure linear attention fails on Needle-in-a-Haystack (NIAH) retrieval. Softmax attention
excels at precise long-range retrieval. MiniMax tested multiple ratios and found **7:1
was optimal** for the speed/quality tradeoff.

**RoPE on softmax layers only**: Applied to half the head dimension (64 of 128 dims),
base frequency 10M. Lightning attention layers do not use positional encodings —
position information comes from the causal structure of the linear recurrence.

### 1.3 MoE Design

| Parameter | MiniMax-01 | DeepSeek V3 (comparison) |
|-----------|-----------|--------------------------|
| Total experts | 32 | 256 + 1 shared |
| Activated (top-k) | 2 | 8 + 1 shared |
| Expert hidden dim | 9,216 | 2,048 |
| Shared experts | **None** | 1 |
| Hidden size | 6,144 | 7,168 |
| Active compute/MoE layer | 2 × 9,216 = 18,432 | (8+1) × 2,048 = 18,432 |
| Total layers | 80 | 61 |
| Routing | Top-2 with token drop | Top-8 grouped, aux-loss-free |
| Load balancing | Auxiliary loss (α=0.001) | Dynamic bias (aux-loss-free) |

**Key design choices:**
- **No shared expert** — deliberate departure from DeepSeek V3
- **Fewer, larger experts** (32 × 9216) vs DeepSeek's many small experts (256 × 2048)
- Same active compute per MoE layer despite radically different expert topology
- **Token-drop strategy**: Capacity limits per expert, excess tokens discarded during training
- **Global Router**: Balances tokens per Expert Parallelism (EP) group, not just per-expert
- Activation function: SiLU

### 1.4 Pre-Training

#### Data
- **~12 trillion tokens** total
- Sources: academic literature, books, web content, code
- Quality filtering via reward labeler (trained from prior MiniMax 5B-active/60B-total MoE model)
- Metrics tracked: knowledge depth, practical helpfulness, categorical distribution
- High-quality data: 4× deduplication; low-quality: 2× deduplication
- Format: mix of QA format and natural distribution
- Byte normalization: acc_norm² used for tracking

#### Optimization
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Beta1 | 0.9 |
| Beta2 | 0.95 |
| LR schedule | WSD-like (warmup-stable-decay), final LR = 10% of peak |
| Batch size warmup | 16M → 128M tokens (critical batch size scaling via power-law) |
| Initialization | Xavier with DeepNorm modifications |
| Infrastructure | ~2,000 H800 GPUs |
| MFU | >75% on H20 GPU |

#### Long Context Training (3-Phase Progressive Extension)

| Phase | Context | RoPE Base | Data Mix | Notes |
|-------|---------|-----------|----------|-------|
| Main training | 8K | 10,000 | Standard | Bulk of 12T tokens |
| Phase 2 | 128K | 5,000,000 | 30% short (<32K) + 70% medium (<128K) | 300B tokens |
| Phase 3a | 512K | 10,000,000 | 35% short + 35% medium + 30% long | |
| Phase 3b | 1M | 10,000,000 | 30% short + 30% medium + 40% long | |

**Linear weight interpolation** between phases to mitigate distribution shift:
```
W_t = alpha × W_prev + (1 - alpha) × W_current
```

### 1.5 Post-Training Pipeline (Iterative)

Four-stage process, run iteratively:
1. **Short-context SFT**
2. **Long-context SFT**
3. **Short-context RL** (offline DPO + online GRPO)
4. **Long-context RL** (offline DPO + online GRPO)

Key finding: **iterative alternation** between short and long context training is essential.
Running short-only then long-only degrades short-context performance. The alternating
pattern preserves both capabilities.

Safety alignment via harmless reward model balancing utility with content reliability.

### 1.6 Infrastructure Innovations

- **LASP+** (Linear Attention Sequence Parallelism Plus): replaces send-recv with AllGather for linear attention
- **Varlen Ring Attention**: ring attention on concatenated variable-length sequences without padding
- **Expert Tensor Parallel (ETP)**: decouples MoE parallel strategy from non-MoE parallelism
- **EP-ETP overlap**: 50% reduction in MoE communication overhead
- **Batched kernel fusion**: 10% latency reduction
- **Separated prefill/decode**: independent parallelism for each phase
- **Multi-level padding**: reduces padding waste in heterogeneous batches

---

## 2. MiniMax-M1 — Reasoning Model (arXiv:2506.13585)

Built on MiniMax-Text-01. First open-weight large-scale hybrid-attention reasoning model.

### 2.1 Continued Pre-Training

| Parameter | Value |
|-----------|-------|
| Additional tokens | 7.5T |
| Corpus composition | 70% STEM, code, books, reasoning; 30% natural QA from web/forums/textbooks |
| QA data processing | Semantic deduplication |
| LR schedule | Constant 8e-5 for first 2.5T tokens, decay to 8e-6 over remaining 5T tokens |

**Long context extension (4-phase staged)**:
Progressive 32K → 1M to prevent gradient explosions during context extension.

### 2.2 SFT Stage

- CoT (chain-of-thought) injection
- Data: math, coding, STEM, writing, QA, multi-turn chat
- Math and coding: ~60% of SFT samples
- Output format trains the model to produce explicit reasoning chains

### 2.3 CISPO Algorithm — The Core RL Innovation

**Clipped Importance-weight Sampling Policy Optimization** — MiniMax's replacement for
PPO/GRPO/DAPO. This is their most important algorithmic contribution.

#### The Problem with PPO/GRPO

PPO clips the policy ratio in the multiplicative loss term:
```
L_PPO = min(r(θ) × A, clip(r(θ), 1-ε, 1+ε) × A)
```

When `r(θ)` is clipped and A is small, the gradient approaches zero. This **kills gradient
signal for rare but crucial reasoning tokens** — words like "However," "Wait," "Recheck"
that appear infrequently but drive correct reasoning. PPO/GRPO systematically suppresses
exactly the tokens that matter most for reasoning.

#### CISPO Formulation

```
J_CISPO(θ) = E[ 1/Σ|o_i| × Σ_i Σ_t sg(r̂_i,t(θ)) × Â_i,t × log π_θ(o_i,t | q, o_i,<t) ]
```

Where:
- `r̂_i,t(θ) = clip(r_i,t(θ), 1 - ε_low, 1 + ε_high)` — clipped importance sampling weight
- `sg()` = stop-gradient (detach) — weights do NOT participate in backprop
- Gradients flow **only** through `log π_θ` term

**The critical difference**: Instead of clipping the policy ratio in the loss, CISPO clips
importance sampling weights as **detached scalar coefficients**. This ensures ALL tokens
receive gradients, including rare reasoning tokens that PPO would suppress.

#### Implementation

```python
# CISPO core (pseudocode from Swift/ms-swift documentation)
log_ratio = per_token_logps - old_per_token_logps
importance_weights = torch.exp(log_ratio)
clamped_ratios = torch.clamp(importance_weights, max=epsilon_high).detach()  # KEY: .detach()
per_token_loss = -clamped_ratios * advantages.unsqueeze(1) * per_token_logps
```

The `.detach()` is the entire innovation. Clipped weights don't backpropagate —
gradients derive solely from the log probability term.

#### Hyperparameters

| Parameter | Value |
|-----------|-------|
| ε_high (IS clip) | 5.0 (much larger than PPO's typical 0.2) |
| ε_low (IS clip) | Typically disabled (lower bound not needed) |
| Optimizer | AdamW |
| Beta1 | 0.9 |
| Beta2 | 0.95 |
| Epsilon | **1e-15** (NOT 1e-8 — standard VeRL default caused non-convergence) |
| Output length | First 40K tokens, then extended to 80K |
| Gradient clipping | Decreased during 80K length extension phase |

**Why ε = 1e-15**: Gradient magnitudes in reasoning RL span 1e-18 to 1e-5. Standard
Adam epsilon (1e-8) dominates the denominator for small gradients, effectively zeroing
them. Using 1e-15 preserves gradient signal across the full range.

#### Performance

- **2× convergence speedup** vs DAPO on AIME 2024
- Reaches DAPO-equivalent performance in **50% of training steps**
- Significantly superior to early GRPO

#### Critical Precision Fix

A training-inference inconsistency was discovered in the LM prediction head.
Restoring the output layer to **FP32** improved probability correlation from ~0.9x to
~0.99x, enabling stable sustained training gains. Without this fix, reward curves
plateau prematurely.

### 2.4 Reward System

#### Rule-Based Verification (Verifiable Tasks)

| Domain | Data | Verification |
|--------|------|-------------|
| Math | ~50K curated problems (filtered: 0 < pass@10 < 0.9) | Exact match against ground truth |
| Logic | 41 tasks via SynLogic framework, ~53K synthesized | Programmatic verification |
| Competitive programming | 30K problems from online judges | LLM-generated test suites |
| Software engineering | Several thousand sandbox-based samples | GitHub issue/PR resolution, test pass/fail |

**Pass-rate filtering**: Only include problems where `0 < pass@10 < 0.9`. This filters
out trivial (always solved) and impossible (never solved) problems.

#### Model-Based Rewards (GenRM)

For tasks without verifiable ground truth:
- **Five-grade scale** evaluating response-ground-truth alignment
- **Pairwise comparison** framework: scores of -1, 0, or +1 for open-ended tasks
- **Combined** rule + model-based rewards for instruction-following

**Length bias mitigation**: Online monitoring during RL detects reward hacking exploiting
length preferences. When detected, triggers immediate GenRM recalibration.

#### General Domain

~25K complex samples covering writing, reasoning, multi-turn dialog.

### 2.5 Curriculum RL

**Progressive mixing**: Start with reasoning-intensive tasks (rule-based reward only),
gradually mix in general domain tasks (model-based reward).

This prevents catastrophic forgetting while fostering cross-domain generalization.

### 2.6 Repetition Detection

Early truncation when **3,000 consecutive tokens exceed 0.99 probability**.
This catches degenerate repetition loops that waste compute during RL rollouts.

### 2.7 Training Scale

| Parameter | Value |
|-----------|-------|
| Infrastructure | 512 H800 GPUs |
| Duration | 3 weeks |
| Cost | ~$534,700 |
| Max output length | 80K tokens |

### 2.8 Efficiency Advantage

At 100K token generation: M1 consumes **~25% of FLOPs** vs DeepSeek-R1 due to near-linear
attention scaling from lightning attention. First-token latency reduced from 60s to 4-5s
for 100K-token inputs.

---

## 3. MiniMax-M2.1 — Agent Model

### 3.1 Architecture
- ~230B total parameters, ~10B activated per token (same MoE base, different scale)
- Context window: 200K+ tokens

### 3.2 Core Insight: Multi-Scaffold Training

MiniMax's critical discovery for agent training: **training on a single scaffold severely
limits generalization**. Agents trained on one tool-use framework fail to transfer to others.

**Solution**: Train across **hundreds of scaffold types** simultaneously.

**For SFT**: Multi-scaffold rejection sampling — generate trajectories across multiple
agent scaffolds, select best ones per task.

**For RL**: Train directly on diverse scaffolds. Forge framework (see §3.3) supports both:
- **White-box agents**: Full scaffold design with direct performance optimization
- **Black-box agents**: Complete agnosticism to internal implementation

### 3.3 Agentic Training Data Construction

#### SWE Scaling (Real-Data-Driven)

Source pipeline:
1. Filter GitHub PRs for quality (merged PRs with relevant test cases)
2. Build Docker sandbox environments with **iterative agent-based construction + self-correction**
3. Categorize PRs: bug fixes, features, optimizations, refactoring
4. Apply task diversification transforms:
   - F2P (Fail-to-Pass): extract failing tests, verify fix restores them
   - P2P (Pass-to-Pass): verify existing tests still pass
   - Bug injection: deliberately introduce bugs, create fix tasks
   - Difficulty merging: combine multiple commits into harder tasks
   - BugFix → SWE-Test conversion: turn fixes into test-writing tasks

**Scale**: 10+ programming languages, 10,000+ runnable PRs, 140,000+ variable-difficulty tasks.

#### AppDev (Expert-Driven Synthesis)

Full-stack application development from scratch:
- Expert-in-the-loop: domain specialists design prompts, meta-queries, rubric-based rewards
- **"Agent-as-a-Verifier"**: Playwright in sandboxes validates dynamic app behavior
- Rubric-based scoring against functional requirements

#### WebExplorer (Synthetic Long-Horizon)

Two-step construction:
1. **Exploration phase**: Agents explore web environments, construct seed questions
2. **Evolution phase**: Queries evolved through removal/obfuscation/substitution
- Average reasoning turns: 7.9 (original) → 9.9 (evolved)

### 3.4 Context Rot Problem

In multi-turn agent interactions, intermediate reasoning steps accumulate and create
**"attention dilution"** — the model attends to irrelevant historical context instead
of the current task state.

**MiniMax's solution**: Integrate context management directly into the RL loop as an
**explicit agent action**. The agent learns when to summarize/compress/discard context,
preventing inference-training mismatch.

---

## 4. Forge: Scalable Agent RL Framework

MiniMax's proprietary infrastructure for training agents with RL at scale. Solves the
"impossible triangle" of **system throughput**, **training stability**, and **agent flexibility**.

### 4.1 Three-Module Architecture

```
┌─────────────────┐    ┌────────────────────────┐    ┌─────────────────────┐
│   Agent Side     │    │  Middleware Abstraction  │    │  Training/Inference  │
│                  │    │                          │    │                     │
│  agent_reprocess │───▶│  Gateway Server          │───▶│  LLM Engine         │
│  agent_run       │    │  Data Pool               │    │  Train Engine       │
│  agent_postprocess│◀──│  Windowed FIFO Scheduler │◀──│                     │
│  calculate_reward│    │                          │    │                     │
└─────────────────┘    └────────────────────────┘    └─────────────────────┘
```

**Four required interfaces** per agent scaffold:
1. `agent_reprocess` — initialization and setup
2. `agent_run` — execution (tool calls, reasoning, environment interaction)
3. `agent_postprocess` — trajectory extraction and cleanup
4. `calculate_reward` — reward computation from outcomes

### 4.2 Windowed FIFO Scheduling

**Problem**: Naive asynchronous RL training drifts toward "fast and easy" samples
(the **Straggler Effect**). Hard tasks take longer to complete, so easy tasks dominate
the training queue, biasing the policy toward simple behavior.

**Solution**: Windowed FIFO with local disorder + global ordering.

```
Generation queue: Q = [T_0, T_1, ..., T_{n-1}]
Visibility window: W = 0.3N (30% of queue)

Rules:
1. Within window: any completed trajectory consumed immediately (local greedy disorder)
2. At window boundary: STRICT BLOCKING — tasks outside window forbidden even if complete
3. Window slides forward ONLY as head tasks are consumed
```

This preserves data distribution while tolerating completion time variance.

### 4.3 Prefix Tree Merging (40× Training Speedup)

**Problem**: Multi-turn agent interactions share common prefixes (system prompt + initial
context + early tool calls). Naive training recomputes these prefixes for every sample.

**Solution**: Merge multiple completions sharing common prefixes into a single prefix tree.

```
Sample 1: [system_prompt][context][tool_call_1][response_A]
Sample 2: [system_prompt][context][tool_call_1][response_B]
Sample 3: [system_prompt][context][tool_call_2][response_C]

Merged tree:
[system_prompt][context] ─┬─ [tool_call_1] ─┬─ [response_A]
                          │                  └─ [response_B]
                          └─ [tool_call_2] ── [response_C]
```

- Uses **Magi Attention** primitives for logical consistency within tree structure
- Post-forward-pass: tree deconstructed for standard loss computation
- **Strict mathematical equivalence guaranteed** — zero downstream impact on training
- Eliminates redundant prefix prefilling in long-context multi-turn scenarios
- **40× training speedup** in practice

### 4.4 Inference Acceleration

- **MTP-based speculative decoding** with Top-K KL loss maintaining acceptance rates
  despite evolving policy during RL
- **Heterogeneous prefill/decode disaggregation** for independent parallelism
- **Global L3 KV Cache Pool** with DFS-backed cost-aware scheduler

### 4.5 Scale

- 100,000+ distinct real-world agent scaffolds and environments
- Context lengths up to 200K tokens
- Millions of samples/day throughput

---

## 5. CISPO for Agents (Multi-Turn Extension)

CISPO (§2.3) was designed for single-turn reasoning. Agent training requires multi-turn
adaptation:

### 5.1 Multiple Importance Sampling (MIS)

Multi-turn agent trajectories have different numbers of LLM calls per trajectory. Standard
IS weight computation doesn't account for this heterogeneity.

**MIS adaptation**: Properly normalizes importance weights across trajectories with
different numbers of generation steps, preventing trajectories with more steps from
dominating the gradient.

### 5.2 PPO-Based Trajectory Filtering

Long-tail trajectory filtering prevents excessive gradient fluctuations from extreme
outlier trajectories (very long or very short completions).

### 5.3 Process Rewards for Agents

**Composite Reward Framework** addressing credit assignment across 200K-token contexts:

| Reward Type | Purpose | Signal |
|-------------|---------|--------|
| Task completion | Did the agent solve the task? | Binary: pass/fail from test execution |
| Process reward | Dense feedback on intermediate behavior | Language mixing penalties, tool errors, format violations |
| Completion time | Incentivize efficiency | Relative completion time vs baseline (rewards parallelism) |
| Reward-to-go | Reduce variance | Normalize returns for long-horizon credit assignment |

### 5.4 Unified Mixed-Domain Training

Train simultaneously across Reasoning, General QA, and Agent domains.
**Avoids negative transfer** from sequential domain-specific training.

---

## 6. MiniMax-M2.5 — Production Agent

- Trained with RL in **hundreds of thousands** of complex real-world environments
- 10+ programming languages
- Continued CISPO algorithm for MoE stability
- Enhanced process reward mechanisms for long-context agent rollouts

### Benchmarks

| Benchmark | Score |
|-----------|-------|
| SWE-Bench Verified | 80.2% |
| Multi-SWE-Bench | 51.3% |
| BrowseComp | 76.3% |
| Speed | 100 TPS (Lightning variant) |
| SWE task time | ~22.8 min average |

---

## 7. MiniMax-M2.7 — Self-Evolving Model (March 18, 2026)

### 7.1 Architecture

| Parameter | Value |
|-----------|-------|
| Total parameters | ~230B |
| Activated per token | ~10B |
| Context window | 205K tokens |
| Speed | 100 TPS (3× faster than Claude Opus) |
| Base | Same MoE transformer as M2.1/M2.5 |

### 7.2 The Self-Evolving Training Loop (Headline Innovation)

M2.7 is the first model that **"deeply participates in its own evolution."** The model
functions as an autonomous research agent within its own training loop.

#### Autonomous RL Research Loop

The model executes iterative cycles, each consisting of:

```
┌─────────────────────────────────────────────────┐
│  1. Analyze failure trajectories (why task failed) │
│  2. Plan changes (strategy to fix)                 │
│  3. Modify scaffold code (edit own agent harness)  │
│  4. Run evaluations (test new version)             │
│  5. Compare results (metric analysis)              │
│  6. Decide: keep or revert changes                 │
└─────────────────────────────────────────────────┘
                    ↓ repeat
```

This loop ran for **100+ rounds entirely autonomously**, achieving **30% performance
improvement** on internal benchmarks without human intervention.

#### Discovered Optimizations (By the Model Itself)

The model autonomously discovered and implemented:
- Systematic parameter search (temperature, frequency penalty, presence penalty)
- Workflow guideline refinement (e.g., auto-searching for same bug patterns in other files)
- Loop detection and prevention in agent scaffold
- Context management improvements
- Failure pattern recognition and mitigation strategies

#### Scope of RL Research Automation

M2.7 handles **30-50% of the RL research workflow** autonomously:

| Automated Task | Description |
|----------------|-------------|
| Log reading & debugging | Parse training logs, identify anomalies |
| Metric analysis | Compute, visualize, interpret training metrics |
| Code fixes & merge requests | Fix bugs in training code, submit changes |
| Smoke testing | Run quick validation tests on code changes |
| Synthetic data generation | Generate new training examples |
| Training environment optimization | Improve Docker/sandbox configurations |
| Literature review assistance | Search and summarize relevant papers |
| Experiment specification tracking | Maintain experiment configs and results |
| Data pipeline management | Monitor and fix data processing issues |

#### Harness Skills

The model builds **reusable instruction sets** (2,000+ tokens each), maintains a
persistent memory store, updates its own capabilities across iterations, and runs
RL experiments to optimize its own performance. It constructed **40+ complex skills**
with **97% compliance rate**.

### 7.3 Agent Teams

Multi-agent collaboration with:
- Role boundaries and specialization
- Adversarial reasoning (agents critique each other)
- Protocol adherence enforcement
- Behavioral differentiation (different agents, different strategies)

### 7.4 Benchmarks

| Benchmark | M2.7 Score | Notes |
|-----------|-----------|-------|
| SWE-Pro | 56.22% | Matches GPT-5.3-Codex |
| VIBE-Pro | 55.6% | |
| Terminal Bench 2 | 57.0% | |
| SWE Multilingual | 76.5% | |
| Toolathon | 46.3% | |
| GDPval-AA ELO | 1,495 | Highest open-API model |
| MLE Bench Lite | 66.6% medal rate | Best run: 9 gold, 5 silver, 1 bronze |
| MM Claw | 62.7% accuracy | 97% adherence on 40+ complex skills |
| Hallucination rate | 34% | vs 46% Claude Sonnet, 50% Gemini 3.1 |
| AI Intelligence Index | 50 | +8 over M2.5 |

### 7.5 Pricing

- $0.30/M input tokens, $1.20/M output tokens (API-only, proprietary)

---

## 8. Key Technical Lessons for NanoSeek

### 8.1 Architecture Lessons

1. **Hybrid attention works at scale**: 7:1 linear:softmax ratio preserves retrieval
   while gaining near-linear scaling. Pure linear attention fails NIAH.

2. **Fewer, larger experts can match many, smaller experts**: MiniMax uses 32×9216
   vs DeepSeek's 256×2048 with identical active compute per layer. The expert topology
   is a design choice, not a constraint.

3. **No shared expert is viable**: MiniMax's MoE omits the shared expert that DeepSeek
   considers essential. Both approaches produce competitive models.

4. **Auxiliary loss vs aux-loss-free**: MiniMax uses standard auxiliary loss (α=0.001)
   with token drop, while DeepSeek uses dynamic bias (aux-loss-free). Both work at scale.
   This is directly testable in NanoSeek's stability experiments.

### 8.2 Training Pipeline Lessons

1. **Iterative short/long context alternation** is essential for preserving capabilities.
   Sequential training (all short, then all long) degrades short-context performance.

2. **Batch size warmup via critical batch size scaling**: MiniMax warms from 16M → 128M
   tokens. NanoSeek already does 1/5→1× warmup following a similar principle.

3. **Linear weight interpolation** between training phases prevents distribution shift.
   Formula: `W_t = α × W_prev + (1-α) × W_current`.

4. **WSD learning rate schedule** (warmup-stable-decay) with final LR = 10% of peak.

### 8.3 CISPO vs GRPO for NanoSeek

CISPO's key insight is directly relevant to NanoSeek's RL post-training:

| Aspect | GRPO/PPO | CISPO |
|--------|----------|-------|
| Clipping target | Policy ratio in loss | IS weight as detached scalar |
| Gradient for rare tokens | Often zero (clipped away) | Always non-zero |
| Reasoning token preservation | Poor | Excellent |
| Convergence speed | Baseline | 2× faster than DAPO |
| Implementation complexity | Standard | Minimal change (add .detach()) |
| Adam epsilon | 1e-8 (standard) | **1e-15** (critical for stability) |
| LM head precision | BF16 | **FP32** (critical for reward correlation) |

**For NanoSeek Phase 5 (RL)**: Consider implementing CISPO instead of or alongside GRPO.
The implementation difference is ~10 lines. The epsilon and FP32 head fixes alone could
prevent premature reward plateaus.

### 8.4 Agent Training Lessons (Future Work)

1. **Multi-scaffold training** is essential for agent generalization
2. **Context management must be an explicit RL action**, not implicit
3. **Windowed FIFO scheduling** prevents easy-sample bias in async RL
4. **Prefix tree merging** gives 40× speedup for multi-turn RL training
5. **Process rewards** (dense intermediate feedback) are needed for long-horizon credit assignment
6. **Unified mixed-domain training** (reasoning + general + agent simultaneously) avoids negative transfer

### 8.5 Self-Evolving Training (Frontier Direction)

M2.7's self-evolving loop is the beginning of **recursive self-improvement** in production:
- The model modifies its own scaffold, evaluates, decides to keep or revert
- 100+ autonomous rounds, 30% improvement without human intervention
- Currently covers 30-50% of RL research workflow

This is the direction NanoSeek should prepare for in later phases:
instrumenting training dynamics deeply enough that an agent could read the W&B
dashboard and make informed training decisions.

---

## 9. Comparison: MiniMax vs DeepSeek vs Kimi

| Dimension | MiniMax | DeepSeek V3 | Kimi (Moonshot) |
|-----------|---------|-------------|-----------------|
| Attention | Hybrid lightning+softmax 7:1 | MLA (latent compression) | KDA (delta attention) |
| MoE experts | 32 large (top-2) | 256 small + 1 shared (top-8) | Variable, high-sparsity tested |
| Load balancing | Aux loss + token drop | Aux-loss-free (dynamic bias) | Aux-loss-free |
| Pre-training tokens | 12T | 14.8T | ~13T |
| Context (train) | 1M | 128K | 128K (Kimi-VL: 256K) |
| RL algorithm | **CISPO** | GRPO | GRPO-variant |
| Agent training | Forge framework + multi-scaffold | Not public | Swarm RL (multi-agent) |
| Self-evolving | Yes (M2.7, 100+ autonomous rounds) | Not public | Not public |
| Key innovation | Lightning attention + CISPO | MLA + aux-loss-free routing | Linear attention + delta attention |

---

## Sources

- [MiniMax-01: Scaling Foundation Models with Lightning Attention (arXiv:2501.08313)](https://arxiv.org/abs/2501.08313)
- [MiniMax-M1: Scaling Test-Time Compute Efficiently (arXiv:2506.13585)](https://arxiv.org/abs/2506.13585)
- [MiniMax M2.7: Early Echoes of Self-Evolution (official)](https://www.minimax.io/news/minimax-m27-en)
- [Forge: Scalable Agent RL Framework (official)](https://www.minimax.io/news/forge-scalable-agent-rl-framework-and-algorithm)
- [MiniMax M2.1: Post-Training Experience and Insights (official)](https://www.minimax.io/news/post-training-experience-and-insights-for-agent-models)
- [MiniMax M2.5: Built for Real-World Productivity (official)](https://www.minimax.io/news/minimax-m25)
- [MiniMax-M1 Technical Seminar (official)](https://www.minimax.io/news/minimax-m1-technical-seminar-2)
- [Diving into MiniMax-01 405B MoE (HuggingFace blog)](https://huggingface.co/blog/eliebak/minimax01-deepdive)
- [CISPO documentation (Swift/ms-swift)](https://swift.readthedocs.io/en/latest/Instruction/GRPO/AdvancedResearch/CISPO.html)
- [Lightning Attention-2 (arXiv:2401.04658)](https://arxiv.org/abs/2401.04658)
- [MiniMax-01 Summary (Jianyu Huang)](https://jianyuh.github.io/minimax-01/2025/01/18/minimax-01.html)
