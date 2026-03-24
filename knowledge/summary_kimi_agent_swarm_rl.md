# Kimi RL Training & Agent Swarm: Deep Analysis
## K1.5 (Jan 2025) -> K2 (Jul 2025) -> K2.5 (Feb 2026)

**Papers**:
- K1.5: arXiv:2501.12599 — "Scaling Reinforcement Learning with LLMs"
- K2: arXiv:2507.20534 — "Open Agentic Intelligence"
- K2.5: arXiv:2602.02276 — "Visual Agentic Intelligence"

---

## 1. Evolution of Kimi's RL Framework

### The Big Picture

Kimi's RL journey represents a **progressive scaling** from single-model reasoning RL (K1.5) to multi-domain agentic RL with self-critique (K2) to parallel multi-agent swarm RL (K2.5):

```
K1.5 (Jan 2025):  Single-model RL for long-CoT reasoning
                   Key innovation: Long context scaling (128K) + Partial Rollouts
                   Result: 77.5 AIME, matches o1

K2 (Jul 2025):    Multi-domain RL with verifiable + self-critique rewards
                   Key innovation: Verifiable Rewards Gym + Self-Critique Rubric
                   Scale: 1.04T params, 32B active, 15.5T tokens pretrained
                   Result: #1 open-source on LMSYS Arena

K2.5 (Feb 2026):  Agent Swarm with Parallel Agent RL (PARL)
                   Key innovation: Frozen subagents + trainable orchestrator
                   Result: 4.5x latency reduction, 100 parallel sub-agents
```

---

## 2. K1.5: The Foundation — RL for Long-CoT Reasoning

### 2.1 Core RL Algorithm: Online Policy Mirror Descent

K1.5 derives RL from **online mirror descent** with KL regularization. The key objective:

```
max_theta E_{x,y*~D}[ E_{(y,z)~pi_theta}[r(x,y,y*)] - tau * KL(pi_theta || pi_theta_i) ]
```

This has a closed-form optimal solution:
```
pi*(y,z|x) = pi_theta_i(y,z|x) * exp(r(x,y,y*)/tau) / Z
```

The **surrogate loss** (what they actually optimize):
```
L(theta) = E_D[ E_{pi_theta_i}[ (r(x,y,y*) - tau*log(Z) - tau*log(pi_theta/pi_theta_i))^2 ] ]
```

They approximate `tau*log(Z)` with the **empirical mean reward** `r_bar = mean(r(x,y_1,y*), ..., r(x,y_k,y*))`.

The gradient:
```
(1/k) sum_j [ grad_theta log(pi_theta(y_j,z_j|x)) * (r(x,y_j,y*) - r_bar)
              - (tau/2) * grad_theta (log(pi_theta/pi_theta_i))^2 ]
```

**Key insight**: This resembles policy gradient with mean reward as baseline + L2 regularization on log-ratio, but responses are sampled from `pi_theta_i` (off-policy), not on-policy. The reference policy is reset every iteration.

### 2.2 No Value Network — A Deliberate Choice

K1.5 explicitly **rejects value functions** for credit assignment in CoT settings:

**Argument**: If the model generates partial CoT `(z_1, ..., z_t)` and there are two next steps:
- `z_{t+1}`: directly leads to correct answer (high value)
- `z'_{t+1}`: contains errors but model recovers later (low value)

Standard RL would penalize `z'_{t+1}` via negative advantage. But exploring `z'_{t+1}` is **critical** for learning trial-and-error patterns in long CoT. By using only outcome-based rewards on the entire trajectory, the model learns to recover from mistakes — a crucial skill for reasoning.

**This is philosophically important**: The goal is not maximizing training accuracy, but equipping the model with **problem-solving strategies** that generalize to test problems.

### 2.3 Long Context Scaling: The Core Innovation

K1.5 scales RL context to **128K tokens** and shows performance increases with context length on hard reasoning benchmarks. Key technique:

**Partial Rollouts**: When a trajectory exceeds the token budget:
1. Save the unfinished portion to a replay buffer
2. Continue in the next RL iteration
3. Only the current iteration requires on-policy computation
4. Previous segments are reused from buffer (no re-generation)

This is analogous to **experience replay** but for long-form generation. It enables training on extremely long CoT without blocking GPU resources.

**The key insight**: Longer CoT enables implicit planning/search within a single autoregressive pass, eliminating the need for MCTS, value functions, or process reward models. The model learns to plan, reflect, backtrack, and explore — all within the text stream.

### 2.4 Length Penalty

To combat **overthinking** (response length explosion during RL):

```python
# For k sampled responses of problem x:
lambda_i = 0.5 - (len(i) - min_len) / (max_len - min_len)

len_reward(i) = lambda_i     if correct
                min(0, lambda_i)  if incorrect
```

- Correct responses: shorter = bonus, longer = penalty
- Incorrect responses: only penalize if longer than shortest
- Warmed up gradually (no penalty initially, then constant)

### 2.5 Sampling Strategies

**Curriculum Sampling**: Start with easy problems, progress to harder ones. Early RL model has limited capability; spending compute on very hard problems yields few correct samples.

**Prioritized Sampling**: Track per-problem success rates `s_i`, sample proportional to `1 - s_i`. Focus effort on weakest areas.

### 2.6 Reward Modeling

**For math**: Two approaches compared:
- Classic RM (value-head scalar): 84.4% accuracy
- **CoT RM** (step-by-step reasoning then judgment): **98.5% accuracy**

They use CoT RM for RL training. This is a huge difference — nearly perfect reward signal vs 15% error rate.

**For coding**: Auto-generated test cases using CYaRon library + cross-validation against ground truth submissions.

### 2.7 Long2Short Transfer

Four methods to distill long-CoT model into short-CoT:
1. **Model merging**: Average weights of long-CoT and short-CoT models (surprisingly effective)
2. **Shortest rejection sampling**: Sample 8x, keep shortest correct for SFT
3. **DPO**: Shortest correct as positive, longer as negative
4. **Long2short RL**: Standard RL first, then RL with length penalty + reduced max rollout

Long2short RL gives best token efficiency. k1.5-short achieves 60.8 AIME with only 3,272 tokens average.

### 2.8 Infrastructure

**Synchronous iterative RL framework**:
- Rollout phase: workers generate trajectories (coordinated by central master)
- Training phase: trainer workers update model from replay buffer
- Hybrid deployment: Megatron (training) + vLLM (inference) on same GPUs
  - Training -> offload GPU memory -> inference runs
  - Inference done -> release memory -> training resumes
  - Switching time: <1 minute training->inference, ~10 seconds inference->training
- Code sandbox: Kubernetes-based, using `crun` for fast container startup, cgroup reuse

---

## 3. K2: Multi-Domain Agentic RL

### 3.1 Scale and Architecture

- **1.04T total params, 32B active** (MoE: 384 experts, top-8, sparsity 48x)
- Pre-trained with **MuonClip optimizer** on **15.5T tokens** with **zero loss spikes**
- MuonClip = Muon + QK-Clip (novel technique for training stability)

### 3.2 Large-Scale Agentic Data Synthesis

Three-stage pipeline for tool-use data:

**Stage 1: Tool Spec Generation**
- 3000+ real MCP tools from GitHub
- 20,000+ synthetic tools via hierarchical domain evolution
- t-SNE visualization shows complementary coverage

**Stage 2: Agent & Task Generation**
- Thousands of distinct agents (varied system prompts + tool combinations)
- Rubric-based tasks (simple to complex, with explicit success criteria)

**Stage 3: Trajectory Generation**
- User simulation (LLM-generated personas)
- Tool execution environment (world model with state + controlled stochasticity)
- Quality filtering via LLM judge against rubrics
- **Hybrid approach**: Simulated environments + real execution sandboxes for coding

This is essentially **large-scale rejection sampling** through quality filtering.

### 3.3 Verifiable Rewards Gym

K2 extends RL to many more domains with verifiable rewards:

| Domain | Reward Signal |
|--------|--------------|
| Math/STEM/Logic | Rule-based correctness, pass@k difficulty filtering |
| Coding | Test suite pass rates, real sandbox execution |
| Software Engineering | GitHub PRs with unit tests |
| Instruction Following | Hybrid: code interpreter (deterministic) + LLM judge (nuanced) + hack-check layer |
| Faithfulness | Sentence-level faithfulness judge model |
| Safety | Attack model + target model + judge model pipeline |

**Difficulty calibration**: Use SFT model's pass@k as difficulty proxy, select **moderate difficulty** (neither too easy nor too hard).

### 3.4 Self-Critique Rubric Reward (Beyond Verification)

For tasks without verifiable rewards (creative writing, open-ended QA), K2 introduces **self-critique**:

**Process**:
1. K2 actor generates responses for general prompts
2. K2 critic performs **pairwise evaluations** against rubrics:
   - **Core rubrics**: Fundamental values (helpfulness, accuracy, etc.)
   - **Prescriptive rubrics**: Anti-reward-hacking rules
   - **Human-annotated rubrics**: Task-specific evaluation criteria
3. Critic selects best response for RL training

**Closed-loop refinement**: The critic is continuously updated using verifiable signals from RLVR prompts. This **distills objective performance signals into the critic**, grounding subjective judgments in verifiable data.

### 3.5 RL Algorithm (K2's Extension of K1.5)

Same base algorithm as K1.5, with additions:

**Budget Control**: Per-task maximum token budget. Responses exceeding budget are truncated + penalized. Forces concise reasoning per task type.

**PTX Loss**: Auxiliary SFT loss on curated high-quality samples during RL. Prevents catastrophic forgetting and improves generalization.

**Temperature Decay**: High temperature initially (exploration) -> decay over training (exploitation). Prevents premature convergence on creative/reasoning tasks.

### 3.6 K2 RL Infrastructure

**Colocated Architecture**: Training and inference engines on same workers (same as K1.5 but scaled to 1T model).

**Efficient Engine Switching**:
- Challenge: 1T model requires petabytes/sec of bandwidth for naive parameter transfer
- Solution: **Distributed checkpoint engine** co-located on training nodes
  1. Each checkpoint engine worker gets local copy from training engine
  2. Broadcasts full parameters across all checkpoint engine workers
  3. Inference engine retrieves only its shard
  4. Pipelined parameter-by-parameter updates (minimizes memory footprint)
- Result: Full parameter update for 1T K2 in **<30 seconds**
- Open-sourced: github.com/MoonshotAI/checkpoint-engine

**Efficient System Startup**:
- Training: Each worker reads partial checkpoint, broadcasts to peers (collective read = 1x)
- Inference: Reuses checkpoint engine (no cross-replica synchronization needed)
- Result: Robust to single-point failures

**Agentic Rollout**:
- Heavy environments deployed as dedicated scalable services
- Large number of concurrent rollouts to amortize environment latency
- **Partial rollout** from K1.5: Long-tail tasks pause and resume across iterations
- Unified OpenAI Gym-like interface for new environments

---

## 4. K2.5: Agent Swarm & PARL

### 4.1 Parallel Agent Reinforcement Learning (PARL)

**The Problem**: Sequential agent execution becomes a bottleneck as tasks get complex. Even 100-step reasoning chains hit practical limits for multi-branch tasks requiring wide search.

**The Solution**: Agent Swarm — dynamic task decomposition + parallel sub-agent execution.

**Architecture**:
```
┌─────────────────────────────────────────┐
│         TRAINABLE ORCHESTRATOR          │ ← Updated via RL
│  (decomposes, delegates, synthesizes)   │
└───────┬──────────┬──────────┬───────────┘
        │          │          │
   ┌────▼────┐ ┌───▼────┐ ┌──▼─────┐
   │ FROZEN  │ │ FROZEN │ │ FROZEN │  ← NOT updated
   │Subagent1│ │Subagt2 │ │Subagt3 │    (fixed checkpoints)
   │(research│ │(coding)│ │(analysis)
   └─────────┘ └────────┘ └────────┘
```

**Critical Design Decision**: Subagents are **frozen** (from intermediate policy checkpoints). Only the orchestrator is trained. This solves:

1. **Credit assignment ambiguity**: In multi-agent settings, outcome-based rewards are noisy. A correct final answer doesn't mean all subagents performed well (and vice versa). Freezing subagents treats their outputs as **environmental observations**, not differentiable decision points.

2. **Training instability**: End-to-end co-optimization of orchestrator + subagents creates non-stationary optimization landscape. Decoupling stabilizes training.

**Efficiency trick**: Train orchestrator first with **small-size subagents**, then transition to larger models. Framework dynamically adjusts inference instance ratios between subagents and orchestrator.

### 4.2 PARL Reward Function

```
r_PARL(x, y) = lambda_1 * r_parallel + lambda_2 * r_finish + r_perf(x, y)
```

Three components addressing distinct challenges:

| Reward | Purpose | Addresses |
|--------|---------|-----------|
| `r_parallel` (instantiation reward) | Incentivize creating subagents | **Serial collapse**: orchestrator defaults to single-agent |
| `r_finish` (sub-agent finish rate) | Reward completed subtasks | **Spurious parallelism**: spawning many agents without real decomposition |
| `r_perf` (task-level outcome) | Overall task success | Primary objective |

**Annealing**: `lambda_1` and `lambda_2` are **annealed to zero** over training. This ensures the final policy optimizes for the primary objective, not auxiliary rewards. Early training uses auxiliary rewards to overcome exploration challenges; late training optimizes pure task performance.

### 4.3 Critical Steps: The Key Metric

**Definition**: By analogy to critical path in computation graphs:

```
CriticalSteps = sum_{t=1}^{T} (S_main^(t) + max_i S_sub,i^(t))
```

Where:
- `S_main^(t)`: Steps by main agent in stage t (typically 1)
- `S_sub,i^(t)`: Steps by i-th subagent in parallel group at stage t
- Duration of each stage = the **longest-running subagent** (critical path)

**Why this matters**: Using critical steps (not total steps) explicitly incentivizes **effective parallelization**:
- Excessive subtask creation that doesn't reduce max execution time = no benefit
- Well-balanced task decomposition = direct latency reduction
- Orchestrator learns to allocate work to minimize **end-to-end latency**, not just maximize concurrency

### 4.4 Prompt Construction for PARL

Tasks are designed to stress sequential execution limits:
- **Wide search**: Simultaneous exploration of many independent sources
- **Deep search**: Multiple reasoning branches with delayed aggregation
- Real-world workloads: Long-context document analysis, large-scale file downloading

Tasks are **not explicitly instructed to parallelize**. Instead, they're constructed so sequential execution is impractical within fixed step/tool budgets, making parallel decomposition the natural winning strategy.

### 4.5 K2.5 RL Algorithm Evolution

K2.5 introduces **token-level clipping** to the K1.5/K2 policy optimization:

```
L_RL(theta) = E_x[ (1/N) sum_j sum_i
    Clip(pi_theta(y_j^i|x,prefix) / pi_old(y_j^i|x,prefix), alpha, beta)
    * (r(x,y_j) - r_bar(x))
    - tau * (log(pi_theta/pi_old))^2 ]
```

**Token-level clipping**: Gradients are zeroed for tokens with log-ratios outside `[alpha, beta]`. This differs from PPO clipping — it bounds off-policy drift regardless of advantage sign. Essential for stability in **long-horizon multi-step tool-use reasoning** where train-inference precision mismatches accumulate.

### 4.6 Toggle: Token-Efficient RL

K2.5 proposes **Toggle** — alternating between scaling and budget optimization:

```
Phase 0 (budget-limited):
    r_tilde(x,y) = r(x,y) * I{ mean_accuracy < lambda OR |y| <= budget(x) }

Phase 1 (standard scaling):
    r_tilde(x,y) = r(x,y)
```

Alternates every `m` iterations between:
- **Phase 0**: Train within token budget (only enforced when mean accuracy > threshold `lambda`)
- **Phase 1**: Generate up to max length (standard test-time scaling)

Budget = `rho`-th percentile of correct response lengths (computed once, fixed).

**Result**: 25-30% token reduction with negligible performance impact. Redundant CoT patterns (repeated verifications, mechanical calculations) decrease substantially. Strong domain generalization — training on math+code reduces tokens on GPQA/MMLU-Pro too.

**Why not just budget-constrain?** Pure budget constraints cause **length-overfitting**: models trained under rigid budgets fail to leverage additional tokens for hard problems.

### 4.7 Joint Multimodal RL

**Surprising finding: Visual RL improves text performance**

| Benchmark | Before Vision-RL | After Vision-RL | Delta |
|-----------|-------------------|------------------|-------|
| MMLU-Pro | 84.7 | 86.4 | **+1.7** |
| GPQA-Diamond | 84.3 | 86.4 | **+2.1** |
| LongBench v2 | 56.7 | 58.9 | **+2.2** |

Analysis: Visual RL enhances calibration in structured information extraction, reducing uncertainty on queries resembling visually grounded reasoning.

**Joint RL paradigm**: Organize RL domains by **ability** (knowledge, reasoning, coding, agentic), NOT by modality. Domain experts learn from both text and multimodal queries. GRM optimizes across heterogeneous traces without modality barriers.

### 4.8 K2.5 Unified Agentic RL Infrastructure

**Gym-like interface** with pluggable components:
- `Toolset` module for tools + sandboxes
- `Judge` module for multi-faceted rewards
- Modules for prompt diversification, instruction-following

**Execution model**: Every agent task = independent async coroutine. Tasks can recursively trigger sub-task rollouts (enabling PARL and Agent-as-Judge).

**Scale**: `Rollout Manager` orchestrates up to **100,000 concurrent agent tasks** during RL, with fine-grained control for partial rollout.

**Token-in-Token-out paradigm**: Record log probabilities for all outputs. Co-designed inference engine with custom APIs for RL requirements.

**LLM Gateway**: Proxy service for black-box environments that can't use custom protocol. Records rollout requests/responses under custom protocol for later optimization.

---

## 5. Monitoring, Measurement & Evaluation Techniques

### 5.1 Metrics Tracked During RL Training

| Metric | What It Measures | Used In |
|--------|------------------|---------|
| Training accuracy | Success rate on RL prompts | All (K1.5, K2, K2.5) |
| Response length | Token count of generated CoT | All |
| Per-problem success rate | Difficulty proxy + prioritized sampling | K1.5, K2 |
| Length penalty effect | Token efficiency vs quality tradeoff | K1.5, K2.5 |
| Critical steps | Parallel execution efficiency | K2.5 PARL |
| Parallelism level | Number of active subagents | K2.5 PARL |
| Sub-agent finish rate | Task decomposition quality | K2.5 PARL |
| KL divergence | Policy drift from reference | All |
| Temperature | Exploration vs exploitation balance | K2 |
| Token efficiency | Performance per token | K2.5 Toggle |
| Cross-modal transfer | Vision RL effect on text benchmarks | K2.5 |
| Reward hacking detection | Hack-check layer + adversarial probes | K2 |
| Entropy of generated outputs | Collapse detection during RL | K2 (via dynamic KL penalty) |

### 5.2 Evaluation Benchmarks Across Papers

**K1.5**: AIME 2024, MATH-500, Codeforces, HumanEval-Mul, LiveCodeBench, MMMU, MathVista

**K2**: Tau2-bench, ACEBench, SWE-bench Verified/Multilingual, LiveCodeBench v6, AIME 2025, GPQA-Diamond, OJBench, LMSYS Arena

**K2.5 (comprehensive)**: HLE, AIME 2025, HMMT 2025, IMO-AnswerBench, GPQA-Diamond, MMLU-Pro, SWE-bench Verified/Pro/Multilingual, Terminal Bench 2.0, PaperBench, CyberGym, SciCode, BrowseComp, WideSearch, DeepSearchQA, OSWorld, WebArena, MMMU-Pro, VideoMMMU, ZeroBench

### 5.3 Stability Monitoring Techniques

**From K2**:
- Dynamic KL penalty: Prevents entropy collapse during RL
- Dynamic mini-batch size: Adjusts updates per iteration for stability
- Truncated importance sampling: Mitigates policy mismatch between rollout and training
- PTX loss: Prevents catastrophic forgetting

**From K2.5**:
- Token-level clipping: Bounds off-policy drift from train-inference precision mismatch
- Log probability recording: Detects train-inference mismatches
- Reward annealing: Lambda_1, lambda_2 -> 0 prevents reward hacking on auxiliary rewards

### 5.4 Infrastructure Monitoring

- Rollout Manager tracking 100,000 concurrent tasks
- Parameter update time monitoring (<30s for 1T model)
- GPU utilization optimization (concurrent rollouts amortize environment latency)
- Partial rollout tracking (pause/resume long-tail tasks)
- Performance profiling, data visualization, data verification tools

---

## 6. Key Design Principles Emerging Across Papers

### 6.1 Simplicity Over Complexity

K1.5 explicitly argues for a **simplistic framework**:
- No MCTS, no value functions, no process reward models
- Long context + improved policy optimization is sufficient
- Models learn implicit planning via long CoT

K2 extends this: Self-critique replaces separate reward models for subjective tasks.

### 6.2 Outcome-Based > Process-Based Rewards

All three papers use **outcome-based rewards** (correctness of final answer), NOT process rewards. This encourages exploration of diverse reasoning paths including incorrect intermediate steps.

### 6.3 Decouple What You Can't Co-Optimize

- K1.5: Decouple value estimation from policy (no value network)
- K2: Decouple critic refinement (closed-loop with verifiable signals)
- K2.5: Decouple orchestrator from subagents (freeze subagents)

### 6.4 Progressive Scaling

- Start with easy problems (curriculum)
- Start with small subagents (K2.5 orchestrator training)
- Anneal auxiliary rewards to zero
- Temperature decay (exploration -> exploitation)
- Budget constraints only when model is already accurate

### 6.5 Hybrid Real + Simulated Environments

K2's agentic data synthesis combines:
- Scalable simulation (LLM-as-world-model + tool simulator)
- Real execution sandboxes (coding, software engineering)
- Quality filtering as implicit rejection sampling

---

## 7. Connection to NanoSeek Project

### 7.1 RL Algorithm for NanoSeek

NanoSeek currently has single-stage GRPO. The Kimi papers suggest:

**Adopt K1.5's loss function**: The online mirror descent surrogate loss with mean reward baseline is simpler and more principled than standard GRPO:
```
L = E[(r(x,y) - r_bar - tau*log(pi/pi_old))^2]
```

This is essentially what NanoSeek already does (GRPO is similar), but Kimi's formulation is derived more rigorously from mirror descent.

**No value network**: Kimi validates that outcome-based rewards without value functions work well for reasoning RL. This aligns with NanoSeek's current approach.

### 7.2 Training Stability Techniques for MoE

From Kimi K2/K2.5, applicable to NanoSeek's MoE RL:

1. **PTX loss**: Concurrent SFT during RL prevents forgetting — critical for MoE where some experts may collapse
2. **Dynamic KL penalty**: Prevents entropy collapse (directly relevant to NanoSeek's H_load monitoring)
3. **Token-level clipping**: For NanoSeek's train-inference alignment
4. **Temperature decay**: Start high for exploration, decay for exploitation
5. **Budget control**: Per-task token budgets to prevent overthinking

### 7.3 Reward Design Lessons

1. **CoT RM >> Classic RM** (98.5% vs 84.4%): If building reward models, use chain-of-thought judgment
2. **Self-critique works**: Model can judge its own outputs when grounded in verifiable signals
3. **Difficulty calibration**: Use model's own pass@k as difficulty proxy, train on moderate difficulty
4. **Anti-hack filtering**: Remove easy-to-guess problems (N=8 guesses), add hack-check layers

### 7.4 Partial Rollouts for Long Context

NanoSeek's 4K context is short, but for Phase 2 (8K) and potential future extensions, partial rollouts enable:
- Training on longer CoT than the context window
- No single trajectory blocks GPU resources
- Replay buffer reuse of previous segments

### 7.5 Agent Swarm for Future NanoSeek Extensions

While NanoSeek is a base model, the PARL paradigm is architecturally interesting:
- **Frozen subagents**: Could be applied to NanoSeek's multi-stage GRPO pipeline
- **Critical steps metric**: Useful if NanoSeek trains on multi-step tool-use tasks
- **Reward annealing**: Applicable to NanoSeek's auxiliary losses (load balance, MTP)

### 7.6 Infrastructure Lessons

1. **Colocated training + inference**: Share GPUs, switch between Megatron and vLLM
2. **Checkpoint engine**: For 1T models, broadcast full params, let inference pick shards (<30s update)
3. **Gym-like interface**: Standardize RL environments for easy extension
4. **100K concurrent rollouts**: Amortize environment latency with massive parallelism
5. **Partial rollout**: Pause/resume long-tail tasks across RL iterations

### 7.7 Toggle for NanoSeek's Token Efficiency

K2.5's Toggle method is directly applicable to NanoSeek's RL post-training:
- Alternate between budget-constrained and scaling phases
- Prevents length-overfitting while maintaining scaling ability
- 25-30% token reduction with minimal quality loss
- Generalizes across domains

---

## 8. Critical Numbers & Hyperparameters

### K1.5
- RL context: up to **128K tokens**
- CoT RM accuracy: **98.5%** (vs 84.4% classic RM)
- Hack detection threshold: **N=8 guesses**
- Curriculum: easy -> hard over training
- Length penalty warmup: none initially, then constant
- Long2short: model merging (simple weight averaging) surprisingly effective

### K2
- Model: **1.04T total, 32B active** (384 experts, top-8)
- Pre-training: **15.5T tokens**, zero loss spikes (MuonClip)
- MuonClip optimizer for both pre-training and RL
- Agentic data: 3000+ real MCP tools + 20,000+ synthetic tools
- Parameter update time: **<30 seconds** for 1T model
- SFT data: ~1M text + ~1M vision examples
- RL: Gym framework with 10,000+ concurrent sandboxes
- LMSYS Arena: **#1 open-source** (July 2025)

### K2.5
- Architecture: **1.04T total, 32B active** (same base as K2)
- Pre-training: **15T additional tokens** on top of K2 (vision-text joint)
- Agent Swarm: up to **100 parallel sub-agents, 1,500 coordinated steps**
- Rollout Manager: **100,000 concurrent agent tasks**
- Toggle token reduction: **25-30%** with negligible performance impact
- Wide-search speedup: **4.5x** over sequential
- F1 improvement: 72.8% -> **79.0%** over single-agent
- AIME 2025: **96.1** (with 25K tokens avg, vs K2-Thinking's 30K)

---

## 9. Open Questions and Research Frontiers

1. **Orchestrator generalization**: Does the PARL-trained orchestrator generalize to new tool types not seen during training?
2. **Subagent unfreezing**: Can we eventually fine-tune subagents with proper credit assignment?
3. **RL scaling laws**: K1.5 shows performance scales with context length. Is there a predictable relationship?
4. **Self-critique calibration**: How to prevent the critic from drifting? Closed-loop refinement helps but may have limits.
5. **Cross-modal RL transfer**: K2.5 shows vision RL helps text. How far does this extend? Audio? Code execution traces?
6. **Toggle for non-reasoning**: Does Toggle work for creative writing and open-ended tasks, or only reasoning?
7. **PARL at smaller scales**: Can Agent Swarm work with <7B subagents? What's the minimum competence threshold?

---

*Analysis generated from TeX sources of arXiv:2501.12599 (K1.5), arXiv:2507.20534 (K2), and arXiv:2602.02276 (K2.5).*

Sources:
- [Kimi K1.5 Paper](https://arxiv.org/abs/2501.12599)
- [Kimi K2 Paper](https://arxiv.org/abs/2507.20534)
- [Kimi K2.5 Paper](https://arxiv.org/abs/2602.02276)
- [Kimi K2 GitHub](https://github.com/MoonshotAI/Kimi-K2)
- [Kimi K2.5 GitHub](https://github.com/MoonshotAI/Kimi-K2.5)
- [Checkpoint Engine](https://github.com/MoonshotAI/checkpoint-engine)
