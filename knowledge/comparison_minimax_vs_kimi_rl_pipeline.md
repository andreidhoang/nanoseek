# MiniMax M2.7 vs Kimi K2: RL Training Pipeline Comparison
## Head-to-Head Technical Breakdown
### Compiled: March 2026 | Sources: arXiv:2501.08313, arXiv:2506.13585, arXiv:2507.20534, arXiv:2501.12599, arXiv:2602.02276, official blogs

---

## Overview

Both MiniMax and Moonshot (Kimi) represent the Chinese AI frontier pushing RL-trained
agentic models. They took **radically different architectural and algorithmic paths**
to arrive at competitive results. This comparison dissects every layer.

```
MiniMax lineage:   Text-01 → M1 → M2.1 → M2.5 → M2.7  (Jan 2025 → Mar 2026)
Kimi lineage:      K1.5 → K2 → K2-Thinking → K2.5       (Jan 2025 → Feb 2026)
```

---

## 1. Architecture Comparison

| Dimension | MiniMax (M2.7 / Text-01) | Kimi K2 |
|-----------|--------------------------|---------|
| Total params | 456B (Text-01) / ~230B (M2.x) | 1.04T |
| Active params/token | 45.9B (Text-01) / ~10B (M2.x) | 32.6B |
| Layers | 80 | 61 |
| Hidden dim | 6,144 | 7,168 |
| Attention | **Hybrid: 7 lightning + 1 softmax per 8 layers** | **MLA (Multi-Head Latent Attention)** |
| Total experts | 32 | 384 |
| Active experts (top-k) | 2 | 8 |
| Shared experts | **None** | 1 |
| Expert hidden dim | 9,216 | 2,048 |
| Active compute/MoE layer | 2 × 9,216 = 18,432 | (8+1) × 2,048 = 18,432 |
| Sparsity ratio | 16 (32/2) | **48 (384/8)** |
| Load balancing | Aux loss (α=0.001) + token drop | Aux-loss-free (dynamic bias) |
| Context (train) | **1M tokens** | 128K tokens |
| Context (inference) | 4M tokens | 128K tokens |
| Positional encoding | RoPE (softmax layers only, half head dim) | RoPE (via MLA, YaRN extension) |
| Vocab | 200K | ~160K |

**Key architectural divergences:**

1. **Attention**: MiniMax's lightning attention gives O(n) inference for 87.5% of layers, enabling 1M context training. Kimi uses standard MLA (like DeepSeek V3) with 23× KV compression but O(n²) complexity. MiniMax wins on context length; Kimi's MLA is more proven at scale.

2. **MoE topology**: Opposite philosophies. MiniMax: fewer, larger experts (32 × 9216). Kimi: many, smaller experts (384 × 2048). Same active compute per layer. Kimi's high sparsity (48×) gives 1.69× fewer FLOPs for same loss — but MiniMax's fewer experts may be easier to stabilize during RL.

3. **Shared expert**: Kimi has 1 shared expert (always active, handles common patterns). MiniMax deliberately omits it. Both approaches produce competitive models, suggesting the shared expert is not critical.

---

## 2. RL Algorithm: CISPO vs Squared-Loss Mirror Descent

This is the deepest divergence. Both labs rejected standard PPO/GRPO but for **different reasons** and arrived at **different solutions**.

### MiniMax: CISPO (Clipped Importance-weight Sampling Policy Optimization)

**Core idea**: Clip importance sampling weights as **detached scalars**, not the policy ratio in the loss.

```python
# CISPO (simplified)
log_ratio = per_token_logps - old_per_token_logps
importance_weights = torch.exp(log_ratio)
clamped_ratios = torch.clamp(importance_weights, max=5.0).detach()   # KEY: .detach()
per_token_loss = -clamped_ratios * advantages * per_token_logps
```

**Why**: PPO/GRPO clips the policy ratio in the multiplicative loss, which kills gradient
signal for rare but crucial reasoning tokens ("However," "Wait," "Recheck"). These
tokens have high policy ratio variance but low advantage magnitude, so PPO systematically
suppresses them. CISPO's `.detach()` ensures ALL tokens get gradients.

**Properties**:
- No value network needed (uses mean-reward baseline like GRPO)
- ε_high = 5.0 (much larger than PPO's 0.2 — allows more exploration)
- 2× faster convergence than DAPO
- Critical fix: Adam ε = 1e-15 (not 1e-8), FP32 LM head

### Kimi: Squared-Loss Online Mirror Descent

**Core idea**: Replace PPO's clipped surrogate with a **squared-loss** surrogate derived from online mirror descent.

```python
# Kimi K2 RL loss (simplified)
log_ratio = per_token_logps - old_per_token_logps
reward_advantage = rewards - mean_rewards  # mean baseline, no value network
squared_loss = (reward_advantage - tau * log_ratio) ** 2
loss = squared_loss.mean()
```

**Why**: Derived from first principles as the optimal policy update under KL-regularized
reward maximization. The squared loss naturally penalizes large deviations from the
reference policy without needing explicit clipping. tau controls exploration/exploitation.

**Properties**:
- No value network (deliberate: value networks penalize wrong intermediate reasoning steps)
- Derived from optimization theory (mirror descent), not heuristic clipping
- tau > 0 as explicit KL regularization (vs CISPO's implicit clipping)
- Muon optimizer (not Adam)

### Head-to-Head

| Dimension | CISPO (MiniMax) | Squared-Loss MD (Kimi) |
|-----------|-----------------|------------------------|
| **Theoretical basis** | Importance sampling with detached weights | Online mirror descent, closed-form optimal policy |
| **Clipping mechanism** | Explicit: clamp IS weights, detach | Implicit: squared loss naturally bounds updates |
| **Gradient for rare tokens** | Always non-zero (main innovation) | Proportional to squared deviation (may still diminish) |
| **KL regularization** | Implicit via IS weight clipping | Explicit via tau parameter |
| **Value network** | No | No |
| **Advantage baseline** | Mean reward across K samples | Mean reward across K samples |
| **Optimizer** | AdamW (ε=1e-15) | **Muon** |
| **Convergence claim** | 2× faster than DAPO | Not directly compared to DAPO |
| **Precision fix** | FP32 LM head (critical) | Not disclosed |
| **Token-level control** | Via detached IS weights | K2.5: explicit token-level clipping of log-ratio |
| **Elegance** | Pragmatic (one-line .detach() fix) | Principled (derived from optimization theory) |
| **Risk** | ε_high=5.0 allows very large IS weights (variance?) | Squared loss may over-penalize large policy shifts |

**Analysis**: Both reject value networks for the same reason — they punish exploratory
reasoning steps. CISPO is more pragmatic (just detach the weights), Kimi's is more
principled (derive the loss from scratch). The Muon vs Adam choice may matter more than
the loss function — Muon's Newton-Schulz orthogonalization could stabilize MoE routing
during RL better than Adam.

---

## 3. Reward System

### Verifiable Rewards

| Domain | MiniMax M1 | Kimi K2 |
|--------|-----------|---------|
| Math | ~50K problems (0 < pass@10 < 0.9) | QA + NuminaMath + AIMO-2 + expert annotations |
| Logic | 53K via SynLogic (41 task types) | 24-game, Sudoku, cryptarithms, Morse code |
| Code | 30K competitive programming + LLM test suites | Competition + GitHub PRs + human unit tests |
| SWE | Several thousand sandbox-based | GitHub PRs, 10K+ concurrent K8s sandboxes |
| Safety | Not disclosed in detail | Adversarial self-play: Attack→Target→Judge |
| Instruction | Combined rule + model-based | Hybrid: code verifier + LLM judge + hack detection |

**Key difference**: Kimi has a **Gym-like extensible framework** (RLVR) that unifies all
verifiable reward types under one API. MiniMax uses domain-specific reward functions
without a unified framework.

### Model-Based Rewards (Non-Verifiable)

| Aspect | MiniMax (GenRM) | Kimi (Self-Critic Rubric) |
|--------|----------------|---------------------------|
| **Architecture** | Separate reward model, 5-grade scale | **Self-critic: the policy model evaluates itself** |
| **Pairwise** | Yes (scores -1, 0, +1) | Yes (pairwise against rubrics) |
| **Rubrics** | Not disclosed | 3 types: core + prescriptive + human-annotated |
| **Anti-hack** | Length bias monitoring + recalibration | N=8 hack detection + prescriptive rubrics ("no initial praise") |
| **Calibration** | Online monitoring, manual recalibration | **Closed-loop**: verifiable rewards continuously calibrate critic |
| **CoT reward model** | Not disclosed | **98.5% accuracy** (vs 84.4% for classic RM) |

**Key difference**: Kimi's self-critic approach is more elegant — the model evaluates
itself, and verifiable rewards continuously ground the critic in reality. MiniMax uses a
separate GenRM. Kimi's closed-loop calibration is a form of self-improvement.

MiniMax's CoT reward model accuracy is not disclosed; Kimi's 98.5% vs 84.4% improvement
from adding chain-of-thought to the RM is a significant finding.

---

## 4. Agentic RL Training

### Data Synthesis

| Stage | MiniMax | Kimi |
|-------|---------|------|
| **Tool sources** | GitHub PRs (real) + expert-designed | **3,000+ real MCP tools + 20,000+ synthetic** (WizardLM evolution) |
| **Task generation** | Bug injection, difficulty merging, BugFix→SWE-Test | Thousands of agents via system prompt × tool combinations |
| **Trajectory generation** | Rejection sampling across scaffolds | Multi-turn with LLM user personas + controlled stochasticity |
| **Quality filter** | F2P/P2P test verification | LLM judge against rubrics |
| **Scale** | 10+ languages, 10K+ PRs, 140K+ tasks | 23K+ tools, tens of thousands of examples |
| **WebExplorer** | Not disclosed | Exploration → evolution: 7.9 → 9.9 reasoning turns |
| **AppDev** | Expert-in-the-loop + Playwright verification | Not specifically named (but covered by tool synthesis) |

**Key difference**: Kimi has more tools (23K+ vs ~10K+) with a synthetic evolution pipeline.
MiniMax has deeper SWE coverage with real GitHub PR transformations (bug injection,
difficulty merging). Both achieve comparable SWE-bench scores.

### Multi-Scaffold vs Standard Training

**MiniMax**: Critical discovery — training on a single scaffold severely limits generalization.
Solution: Train across **hundreds of scaffold types simultaneously**. Forge framework
supports both white-box and black-box agent scaffolds.

**Kimi**: Uses a unified Gym-like RLVR interface but doesn't emphasize scaffold diversity
the same way. Their agentic training uses standard tool-calling scaffolds with MCP integration.

### Context Management

**MiniMax**: Identified "context rot" (attention dilution from accumulated intermediate steps).
Solution: Make context management an **explicit RL action** — the agent learns when to
summarize/compress/discard context.

**Kimi**: Uses **partial rollouts** — long-horizon tasks that exceed token limits are paused
and resumed in the next RL iteration. Previous trajectory segments are reused without
re-generation. Also has repeat detection for early termination.

Both address the long-context RL problem but differently: MiniMax teaches the model to
manage context; Kimi manages context at the infrastructure level.

---

## 5. Agent RL Framework: Forge vs Colocated Hybrid

### MiniMax: Forge Framework

```
Agent Side → Middleware (Gateway + FIFO Scheduler) → Training/Inference Engine
```

**Key innovations**:
- **Windowed FIFO Scheduling**: Prevents "straggler effect" (easy tasks dominating queue).
  30% visibility window with local greedy disorder + global FIFO ordering.
- **Prefix Tree Merging**: 40× speedup by merging shared prefixes across multi-turn
  trajectories into a single tree structure. Uses Magi Attention primitives.
- **MTP-based speculative decoding** with Top-K KL loss maintaining acceptance rates
  despite policy evolution during RL.
- 4 required interfaces per scaffold: reprocess, run, postprocess, calculate_reward.

### Kimi: Colocated Hybrid Architecture

```
Centralized Controller → Inference Engine (rollouts) ↔ Training Engine (updates)
                                      ↕
                        Distributed Checkpoint Engine
```

**Key innovations**:
- **Colocated training/inference**: Same GPU workers alternate between roles. When one
  is active, the other releases GPU memory. Transition: <1 min (train→infer), ~10s (infer→train).
- **Distributed Checkpoint Engine**: Full K2 parameter update broadcast in **<30 seconds**
  across all workers. Pipelined parameter-by-parameter for minimal memory.
- **Partial rollouts**: Long-horizon tasks pause mid-trajectory and resume next iteration.
- **10K+ concurrent K8s sandboxes** for code execution.
- Uses vLLM for inference, Megatron for training, Mooncake for RDMA checkpoint transfer.

### Head-to-Head

| Aspect | Forge (MiniMax) | Colocated Hybrid (Kimi) |
|--------|----------------|-------------------------|
| **GPU utilization** | Separate inference/training | **Same GPUs, alternating** (better utilization) |
| **Transition time** | Not disclosed | <1 min / ~10s |
| **Scheduling** | Windowed FIFO (prevents easy-sample bias) | Standard with partial rollouts |
| **KV cache** | Global L3 pool with DFS scheduler | vLLM PagedAttention |
| **Multi-turn speedup** | **40× via prefix tree merging** | Partial rollout reuse |
| **Speculative decoding** | MTP-based with KL loss | Not disclosed for RL |
| **Sandbox scale** | 100K+ scaffolds | **10K+ concurrent K8s instances** |
| **Scaffold flexibility** | White-box + black-box via 4 interfaces | Unified Gym-like RLVR interface |

**Analysis**: MiniMax's Forge is optimized for **diverse scaffold training** (hundreds of
agent types). Kimi's colocated architecture is optimized for **GPU efficiency** (no
wasted memory from separate inference/training pools). MiniMax's prefix tree merging
is a bigger speedup for multi-turn training; Kimi's partial rollouts handle the long-tail
better.

---

## 6. Multi-Agent / Swarm Training

### MiniMax M2.7: Agent Teams + Self-Evolving

**Agent Teams**: Multi-agent collaboration with role boundaries, adversarial reasoning
(agents critique each other), protocol adherence enforcement.

**Self-Evolving Loop** (headline innovation):
```
Analyze failures → Plan changes → Modify scaffold → Evaluate → Compare → Keep/Revert
```
- 100+ autonomous rounds, 30% improvement without human intervention
- Model modifies its own agent harness code
- Handles 30-50% of RL research workflow autonomously
- Built 40+ complex skills with 97% compliance

### Kimi K2.5: PARL (Parallel Agent Reinforcement Learning)

**Architecture**: Trainable orchestrator + **frozen subagents** (from fixed intermediate
policy checkpoints).

**Reward**:
```
r_PARL = λ₁·r_parallel + λ₂·r_finish + r_perf(x, y)
```
- `r_parallel`: Prevents "serial collapse" (defaulting to single-agent mode)
- `r_finish`: Prevents "spurious parallelism" (spinning up useless subagents)
- λ₁, λ₂ **annealed to zero** — auxiliary rewards are training wheels only

**Credit assignment**: Subagent outputs = environmental observations (not differentiable
decision points). Clean separation of coordination logic from execution.

**Scale**: Up to 100 subagents, 1,500 coordinated steps, 4.5× latency reduction.

### Head-to-Head

| Aspect | MiniMax M2.7 | Kimi K2.5 PARL |
|--------|-------------|----------------|
| **Multi-agent architecture** | Agent teams with role boundaries | Orchestrator + frozen subagents |
| **Credit assignment** | Not formally specified | Subagent outputs as environment observations |
| **Anti-collapse** | Not disclosed | r_parallel + r_finish (annealed) |
| **Self-modification** | **Yes — model edits own scaffold** | No — orchestrator trained, subagents frozen |
| **Autonomous research** | **100+ rounds, 30% improvement** | Not claimed |
| **Scale** | Not quantified (agent teams) | **100 subagents, 1,500 steps, 4.5× speedup** |
| **Innovation type** | Recursive self-improvement | Principled multi-agent RL with auxiliary rewards |

**Analysis**: These are fundamentally different approaches. MiniMax's self-evolving loop
is more ambitious — the model improves its own training process. Kimi's PARL is more
rigorous — clean credit assignment, principled auxiliary rewards, quantified scaling.
MiniMax is pushing toward AGI-like self-improvement; Kimi is pushing toward reliable
multi-agent coordination.

---

## 7. Training Stability Techniques

| Technique | MiniMax | Kimi |
|-----------|---------|------|
| **Optimizer** | AdamW (ε=1e-15) | **Muon** (Newton-Schulz orthogonalization) |
| **Precision** | FP32 LM head (critical fix) | FP8 storage (E4M3) but no FP8 compute |
| **KL regularization** | Implicit (IS weight clipping) | Explicit (tau parameter) |
| **Repetition detection** | 3,000 tokens > 0.99 probability → truncate | Repeat detection → early termination |
| **Length control** | Not disclosed | Per-task token budgets + truncation penalty |
| **Token efficiency** | Not disclosed | K2.5: toggle heuristic (25-30% token reduction) |
| **Forgetting prevention** | Curriculum mixing (reasoning → general) | **PTX auxiliary loss** on high-quality samples |
| **Gradient clipping** | Decreased during 80K extension | Not specifically disclosed |
| **Reward hacking** | GenRM length bias monitoring | N=8 hack detection + prescriptive rubrics |

**Key differences**:
- **Optimizer**: Muon vs Adam is potentially the most impactful difference. Muon's
  orthogonalization could maintain expert routing stability during RL updates better
  than Adam's diagonal scaling.
- **Forgetting**: Kimi uses PTX loss (explicit); MiniMax uses curriculum mixing (implicit).
  PTX is more principled; curriculum mixing is simpler.
- **Token efficiency**: Kimi's K2.5 toggle heuristic saves 25-30% tokens — a significant
  cost reduction for large-scale RL.

---

## 8. Self-Improving / Self-Evolving Comparison

| Aspect | MiniMax M2.7 | Kimi K2 |
|--------|-------------|---------|
| **Model modifies own code** | **Yes** — edits scaffold, evaluates, keeps/reverts | No |
| **Self-critique** | GenRM (separate model) | **Self-critic**: policy model evaluates itself |
| **Closed-loop calibration** | Not disclosed | **RLVR → critic calibration loop** |
| **Adversarial self-play** | Agent teams critique each other | Safety: Attack→Target→Judge pipeline |
| **Autonomous rounds** | **100+** | Not claimed |
| **Performance gain** | **30%** from self-evolution alone | Not separated from overall RL gains |
| **RL research automation** | **30-50% of workflow** | Not claimed |
| **Persistent memory** | Yes — builds reusable skills | Not disclosed |
| **Recursive improvement** | Yes — model improves own training | Partial — critic improves via verifiable grounding |

**Analysis**: MiniMax's self-evolving loop is genuinely novel — the model is an autonomous
RL researcher modifying its own training. Kimi's self-critic with closed-loop calibration
is more conservative but more rigorous — it uses verifiable rewards as an anchor to
prevent the self-evaluation from drifting.

The risk profiles differ: MiniMax risks compounding errors through recursive
self-modification. Kimi risks missing optimization opportunities by keeping the
model out of its own training loop.

---

## 9. Benchmarks Comparison

| Benchmark | MiniMax M2.7 | Kimi K2 | Kimi K2.5 | Notes |
|-----------|-------------|---------|-----------|-------|
| SWE-Bench Verified | 80.2% (M2.5) | 65.4% | — | MiniMax leads |
| SWE-Pro | 56.2% | — | — | M2.7 flagship |
| SWE Multilingual | 76.5% | — | — | M2.7 |
| Multi-SWE-Bench | 51.3% (M2.5) | — | — | |
| AIME 2024 | — | 77.0% (K2-Think) | — | Reasoning |
| MATH-500 | — | 96.8% (K2-Think) | — | Reasoning |
| LiveCodeBench | — | 65.8% | — | Code |
| BrowseComp | 76.3% (M2.5) | — | — | Web browsing |
| Latency reduction | — | — | **4.5× via PARL** | Multi-agent |
| Speed | 100 TPS | — | — | MiniMax Lightning |
| Hallucination | **34%** | — | — | vs 46% Claude Sonnet |
| Context | 205K (M2.7) | **128K** | — | MiniMax wins |
| Throughput cost | $0.30/$1.20 per M tokens | Open-weight | — | Kimi is free |

**Note**: Direct comparison is difficult because:
1. MiniMax M2.7 is proprietary (API-only); Kimi K2 is open-weight
2. They optimize for different benchmarks (MiniMax: SWE; Kimi: math/reasoning + SWE)
3. M2.7 is ~10B active; K2 is 32.6B active (3× more compute per token)

---

## 10. Cost & Infrastructure

| Dimension | MiniMax | Kimi |
|-----------|---------|------|
| **Pre-training GPUs** | ~2,000 H800 | H800 cluster (size not disclosed) |
| **RL training GPUs** | 512 H800 (M1) | Multiples of 32 nodes (256 GPU model-parallel group) |
| **RL duration** | 3 weeks (M1) | Not disclosed for K2 |
| **RL cost** | ~$535K (M1) | Not disclosed |
| **Inference optimization** | Lightning attention → 100 TPS | Standard MLA |
| **GPU per token** | ~10B active (M2.x) = cheaper | ~32.6B active = 3× more compute |
| **Weight availability** | Proprietary (API-only) | **Open-weight** (Apache 2.0) |
| **Sandbox scale** | 100K+ scaffolds | 10K+ concurrent K8s instances |

---

## 11. Summary: Strategic Philosophy

### MiniMax Philosophy: "Infrastructure-First Self-Evolution"

1. Build the best inference engine (lightning attention → 100 TPS, 1M context)
2. Build the best agent training framework (Forge, prefix tree merging, windowed FIFO)
3. Let the model improve itself (self-evolving loop, 100+ autonomous rounds)

**Bet**: The model that can improve its own training pipeline will eventually win,
regardless of the starting RL algorithm.

### Kimi Philosophy: "Principled Optimization with Verifiable Grounding"

1. Derive the optimal RL algorithm from first principles (mirror descent, no value network)
2. Ground all rewards in verifiable signals (self-critic calibrated by RLVR)
3. Scale multi-agent coordination with clean theory (PARL, credit assignment)

**Bet**: Principled algorithms with verifiable grounding will scale more reliably
than self-modifying systems, and the resulting model will be trustworthy enough
to open-source.

---

## 12. Implications for NanoSeek

### What to Adopt

| Technique | Source | Why | Effort |
|-----------|--------|-----|--------|
| **CISPO .detach()** | MiniMax | 10 lines, preserves rare token gradients | ~1 hour |
| **Adam ε=1e-15** | MiniMax | Prevents gradient death in RL | 1 line |
| **FP32 LM head during RL** | MiniMax | Prevents reward plateau | 1 line |
| **No value network** | Both | Both labs independently validated this for long-CoT | Already in GRPO |
| **Mean reward baseline** | Both | Simple, effective, no extra model | Already in GRPO |
| **Self-critic calibration** | Kimi | Verifiable rewards → calibrate subjective judge | Medium effort |
| **Token-level clipping** | Kimi K2.5 | Bounds off-policy drift per token | ~20 lines |
| **PTX loss** | Kimi | Prevents forgetting during RL | ~10 lines |
| **Toggle heuristic** | Kimi K2.5 | 25-30% token reduction in RL | ~30 lines |

### What to Watch

| Technique | Source | Why Watch |
|-----------|--------|-----------|
| **Self-evolving training** | MiniMax | Could be the future, but unproven at smaller scales |
| **PARL multi-agent** | Kimi K2.5 | Clean theory, 4.5× speedup, but requires infrastructure |
| **Muon for RL** | Kimi | May stabilize MoE routing better than Adam during RL |
| **Prefix tree merging** | MiniMax | 40× speedup but needs Magi Attention primitives |
| **Partial rollouts** | Kimi | Essential for long-horizon agent RL |

### For NanoSeek Phase 5 (RL Post-Training)

**Recommended RL algorithm**: Implement both CISPO and Kimi's squared-loss as ablation:

```python
# CISPO variant
clamped_ratios = torch.clamp(torch.exp(log_ratio), max=5.0).detach()
loss_cispo = -(clamped_ratios * advantages * log_probs).mean()

# Kimi squared-loss variant
loss_kimi = ((advantages - tau * log_ratio) ** 2).mean()
```

Both are ~10 lines on top of existing GRPO. Compare convergence speed, reward curves,
and — critically — **I_spec stability during RL**. If one algorithm disrupts expert
specialization less than the other, that's a novel finding about MoE + RL interaction.

---

## Sources

### MiniMax
- [MiniMax-01: Scaling Foundation Models with Lightning Attention (arXiv:2501.08313)](https://arxiv.org/abs/2501.08313)
- [MiniMax-M1: Scaling Test-Time Compute Efficiently (arXiv:2506.13585)](https://arxiv.org/abs/2506.13585)
- [MiniMax M2.7: Early Echoes of Self-Evolution](https://www.minimax.io/news/minimax-m27-en)
- [Forge: Scalable Agent RL Framework](https://www.minimax.io/news/forge-scalable-agent-rl-framework-and-algorithm)
- [MiniMax M2.1: Post-Training Insights](https://www.minimax.io/news/post-training-experience-and-insights-for-agent-models)
- [CISPO documentation (Swift)](https://swift.readthedocs.io/en/latest/Instruction/GRPO/AdvancedResearch/CISPO.html)

### Kimi
- [Kimi K2: Open Agentic Intelligence (arXiv:2507.20534)](https://arxiv.org/abs/2507.20534)
- [Kimi K1.5: Scaling RL with LLMs (arXiv:2501.12599)](https://arxiv.org/abs/2501.12599)
- [Kimi K2.5: Visual Agentic Intelligence (arXiv:2602.02276)](https://arxiv.org/abs/2602.02276)
