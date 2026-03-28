# Deep Multi-Agent Analysis: Kimi K1.5 → K2 → K2.5 RL System
## Reverse-Engineering from First Principles

**Source papers:**
- K1.5: arXiv 2501.12599 (Jan 2025)
- Kimi-VL: arXiv 2504.07491 (Apr 2025)
- K2: arXiv 2507.20534 (Jul 2025)
- K2.5: arXiv 2602.02276 (Feb 2026)

**Analysis date:** 2026-03-24
**Analysis scope:** Full RL pipeline — math, algorithm, training, infrastructure, evidence, critique

---

# STEP 1 — PROBLEM DEFINITION

## 1.1 What Is Being Solved?

The Kimi family solves the problem of **post-training alignment of large language models** through reinforcement learning, across three progressive scopes:

| Generation | Core Problem | Environment | Action Space |
|-----------|-------------|-------------|-------------|
| K1.5 | Teach LLM long-chain reasoning | Text prompts (math, code, general) | Token sequences (up to 128K) |
| K2 | Add agentic tool use + faithfulness | Text prompts + tool APIs (3K+ real MCP tools) | Token sequences + tool calls |
| K2.5 | Add vision + multi-agent coordination | Multimodal prompts + browser/OS | Token sequences + tool calls + subagent spawning |

## 1.2 What Is the Objective Function?

The core RL objective across all generations is a **KL-regularized expected reward**:

```
max_θ  E_{(y,z) ~ π_θ(·|x)} [ r(x, y, y*) ]  -  τ · KL( π_θ(·|x) || π_ref(·|x) )
```

Where:
- `π_θ` = current policy (the LLM)
- `π_ref` = reference policy (updated periodically, NOT frozen at SFT)
- `r(x, y, y*)` = task-specific reward (verifiable or model-judged)
- `τ` = KL penalty coefficient
- `x` = prompt, `y` = response, `y*` = ground truth (when available)
- `z` = chain-of-thought (latent reasoning trace)

**Critical distinction from standard RLHF:** The reference policy `π_ref` is the policy from the *previous RL iteration* (π_{θ_i}), NOT the original SFT model. This makes it **online policy mirror descent**, not PPO-style trust region optimization.

## 1.3 What Is the Environment?

The "environment" is **non-Markovian and episodic**:
- State = (prompt, tokens generated so far)
- Action = next token
- Transition = deterministic (append token to sequence)
- Reward = given only at episode end (outcome-based)
- Episode termination = EOS token or max length

For agentic tasks (K2, K2.5), the environment includes:
- Tool execution results (code sandbox, web browser, file system)
- Subagent responses (K2.5 Agent Swarm)
- These inject new observations mid-episode, making it a **partially observable** environment

---

# STEP 2 — MATHEMATICAL FOUNDATION (📐 MATH AGENT)

## 2.1 Policy Gradient Theorem — The Foundation

### FACTS (arXiv 2501.12599, Section 3)
K1.5 uses **online policy mirror descent**, which is a specific instantiation of the policy gradient theorem for language models.

### MECHANISM

**Standard policy gradient for LLMs:**

Given a prompt x, the LLM generates a complete response y = (y_1, ..., y_T) autoregressively:

```
π_θ(y|x) = Π_{t=1}^{T} π_θ(y_t | x, y_{<t})
```

The objective is to maximize expected reward:

```
J(θ) = E_{y ~ π_θ(·|x)} [r(x, y)]
```

By the REINFORCE theorem (Williams 1992):

```
∇_θ J(θ) = E_{y ~ π_θ} [ r(x, y) · ∇_θ log π_θ(y|x) ]
```

Expanding the log probability:

```
∇_θ log π_θ(y|x) = Σ_{t=1}^{T} ∇_θ log π_θ(y_t | x, y_{<t})
```

**Problem:** This has **high variance** because:
1. The reward is scalar but applied to all T token decisions
2. No per-token credit assignment
3. Long sequences amplify variance linearly with T

### DERIVATION — K1.5's Solution: Policy Mirror Descent

K1.5 addresses variance through **KL-regularized optimization with a moving reference**:

```
θ_{i+1} = argmax_θ  E_{y ~ π_θ} [r(x, y)]  -  τ · KL(π_θ || π_{θ_i})
```

The KL divergence for autoregressive models:

```
KL(π_θ || π_{θ_i}) = E_{y ~ π_θ} [ Σ_{t=1}^{T} log(π_θ(y_t|x,y_{<t}) / π_{θ_i}(y_t|x,y_{<t})) ]
```

Taking the gradient of the regularized objective:

```
∇_θ L(θ) = E_{y ~ π_θ} [ (r(x,y) - b(x)) · ∇_θ log π_θ(y|x) ]
            - τ · ∇_θ KL(π_θ || π_{θ_i})
```

Where `b(x)` is the baseline (mean reward over sampled responses for prompt x).

**Key insight — why NO value network:**
- A value function V(s_t) estimates expected future reward at each token position
- For long-CoT reasoning, intermediate states that look "unpromising" may lead to breakthroughs
- A value function trained on outcome rewards would assign LOW value to exploratory reasoning steps
- This **penalizes exploration** — the value function acts as a premature pruner
- K1.5 instead uses the simple mean baseline: `b(x) = (1/K) Σ_{k=1}^{K} r(x, y_k)`

**Tensor shapes for K sampled responses:**

```
prompts:     [B]           — B prompts per batch
responses:   [B, K, T_max] — K samples per prompt, max T tokens each
rewards:     [B, K]        — scalar reward per response
baseline:    [B]           — mean reward per prompt
advantages:  [B, K]        — r(x, y_k) - b(x)
log_probs:   [B, K, T_max] — per-token log probabilities
```

### NUMERIC EXAMPLE

Consider a math prompt with K=4 sampled responses:

```
rewards = [1.0, 0.0, 1.0, 0.0]  (binary correct/incorrect)
baseline = mean = 0.5
advantages = [0.5, -0.5, 0.5, -0.5]
```

The gradient pushes UP probability of correct responses (advantage > 0) and DOWN probability of incorrect ones. With only K=4 samples, variance is high — but this is exactly what K1.5 uses in practice.

### FAILURE MODES

1. **Reward hacking**: Policy learns surface patterns that correlate with reward but don't generalize
2. **Mode collapse**: KL penalty too weak → policy converges to single response pattern
3. **KL penalty too strong**: Training barely moves from reference → slow progress
4. **Baseline variance**: With small K, the mean baseline is noisy → high gradient variance
5. **Length bias**: Longer correct responses get same reward as shorter ones → no length incentive

## 2.2 The L2 Regularization on Log-Ratio

### FACTS (K1.5 paper, Section 3.1)
K1.5 uses L2 regularization on the log-probability ratio instead of explicit KL:

```
L_reg = (1/2) · E_{y ~ π_θ} [ Σ_t (log π_θ(y_t|·) - log π_{θ_i}(y_t|·))² ]
```

### MECHANISM
- KL divergence is **asymmetric**: KL(π_θ || π_ref) ≠ KL(π_ref || π_θ)
- Forward KL (mode-covering) can cause the policy to spread mass over bad regions
- L2 on log-ratio is **symmetric** and penalizes large deviations in either direction
- This is effectively a **squared KL** approximation, which is more conservative than standard KL
- It prevents both mode collapse (pulling too close) and mode explosion (pushing too far)

### WHY IT WORKS
For small deviations, KL ≈ (1/2) · E[(log ratio)²] (second-order Taylor expansion). The L2 formulation keeps this quadratic penalty even for larger deviations, providing stronger regularization against catastrophic policy shifts while allowing smooth optimization.

## 2.3 Length Penalty — Anti-Overthinking

### FACTS (K1.5 paper, Section 3.2)

```
len_reward(i) = 0.5 - (len(i) - min_len) / (max_len - min_len)   [for correct responses]
len_reward(i) = min(0, above)                                      [for incorrect responses]
```

### MECHANISM

This creates an **asymmetric** length incentive:
- **Correct responses**: Shorter = higher reward bonus, longer = lower (but still positive if shortest)
- **Incorrect responses**: Only penalized if longer than average (no reward for being short and wrong)
- The 0.5 offset ensures the shortest correct response gets a positive bonus
- `min_len`, `max_len` are computed over the batch of sampled responses for the same prompt

### DERIVATION

Normalized length position: `p(i) = (len(i) - min_len) / (max_len - min_len)` ∈ [0, 1]

For correct: `len_reward = 0.5 - p(i)` ∈ [-0.5, 0.5]
For incorrect: `len_reward = min(0, 0.5 - p(i))` ∈ [-0.5, 0]

The asymmetry is critical: we want to encourage concise **correct** responses, but incorrect responses should never be rewarded for brevity (that would incentivize giving up quickly).

### FAILURE MODES

- If all responses are the same length, `max_len = min_len` → division by zero
- If the length penalty weight is too high, the model learns to give terse, low-quality answers
- The penalty doesn't account for problem difficulty: complex problems legitimately need more tokens

## 2.4 MuonClip — Spectral Stability Theory

### FACTS (K2 paper, Section 2.3)
Muon optimizer produces updates where **all singular values are equal** (full effective rank). This causes attention logit explosion through spectral norm compounding.

### DERIVATION

**Step 1: Why Muon has uniform singular values**

Muon applies Newton-Schulz orthogonalization to the gradient:
```
G → G / ||G||_F  (normalize)
Iterate 5x: X ← aX + bX@X^T@X + cX@X^T@X@X^T@X  (NS5)
```
This converges to the **matrix sign function** msign(G), which has the property:
```
σ_i(msign(G)) = 1  for all i
```
All singular values are exactly 1. The effective rank = full rank.

**Step 2: Why full rank causes instability**

For attention, the logit matrix is:
```
A = (1/√d) · Q · K^T = (1/√d) · (X · W_q) · (X · W_k)^T
```

The spectral norm of A depends on:
```
||A||_2 ≤ (1/√d) · ||X||_2² · ||W_q||_2 · ||W_k||_2
```

When updating W_q with an update ΔW of full effective rank:
```
||W_q + ΔW||_2 ≥ ||W_q||_2 + σ_min(ΔW) · cos(θ)
```
where θ is the principal angle between the singular subspaces.

With Adam, σ_min(ΔW) ≈ 0 (low effective rank), so most updates don't increase ||W_q||_2.
With Muon, σ_min(ΔW) = σ_max(ΔW) = α (all equal), so **every** update has probability of increasing the spectral norm.

**Step 3: The bilinear compounding**

Since logits involve W_q · W_k^T, the spectral norm is **multiplicative**:
```
||W_q · W_k^T||_2 = ||W_q||_2 · ||W_k||_2
```

If both grow by factor (1+ε) per step:
```
||W_q · W_k^T||_2 grows as (1+ε)^{2t}  — squared exponential growth
```

This explains why attention logits explode faster than other weight norms.

**Step 4: The QK-Clip fix**

After each optimizer step, for each attention head h:
```
S_max^h = (1/√d) · max_{x ∈ batch} max_{i,j} |Q_i^h · K_j^{hT}|
```

If S_max^h > τ (τ = 100 for K2):
```
γ_h = min(1, τ / S_max^h)

W_qc^h ← √γ_h · W_qc^h    (compressed query)
W_kc^h ← √γ_h · W_kc^h    (compressed key)
W_qr^h ← γ_h · W_qr^h      (rotary query — full scale because not bilinear)
W_kr^h ← unchanged           (shared across heads — avoid cross-head interference)
```

### WHY √γ for compressed but γ for rotary?

The compressed Q/K contribute to logits via bilinear form: Q_c · K_c^T. Scaling both by √γ gives:
```
(√γ · Q_c) · (√γ · K_c)^T = γ · Q_c · K_c^T
```
Exactly γ scaling on the logit contribution.

The rotary Q/K contribute via: Q_r · K_r^T where K_r is shared across heads. Scaling only Q_r by γ:
```
(γ · Q_r) · K_r^T = γ · Q_r · K_r^T
```
Same effect, but without touching the shared K_r.

### NUMERIC EXAMPLE

Head h=3 has S_max = 150, τ = 100:
```
γ = min(1, 100/150) = 0.667
√γ = 0.816

W_qc^3 *= 0.816
W_kc^3 *= 0.816
W_qr^3 *= 0.667
W_kr (shared) unchanged

New S_max ≈ 0.667 * 150 = 100  ✓  (exactly at threshold)
```

### FAILURE MODES

1. τ too low → excessive clipping → information loss → degraded training
2. τ too high → clipping never triggers → instability not prevented
3. Batch-dependent S_max → clipping can oscillate if batch composition varies
4. Cross-head effects through shared K_r are not fully addressed

### OPEN QUESTIONS

- Why τ = 100 specifically? No ablation provided for different thresholds
- Does the spectral norm growth rate depend on model width? (Relevant for NanoSeek's smaller scale)
- Could pre-emptive spectral normalization (Miyato et al. 2018) be cheaper than post-hoc clipping?

---

# STEP 3 — ALGORITHM CONSTRUCTION (⚙️ ALGORITHM AGENT)

## 3.1 K1.5 RL Algorithm — Online Policy Mirror Descent

### Pseudocode

```python
# Initialize
θ = SFT_checkpoint
θ_ref = copy(θ)

for iteration i = 1, 2, ...:
    # 1. Sample prompts (curriculum + prioritized)
    prompts = sample_prompts(difficulty=f(i), weights=1-success_rates)

    # 2. Generate K responses per prompt
    for x in prompts:
        responses[x] = [sample(π_θ, x) for _ in range(K)]  # K=4 typically

    # 3. Compute rewards
    for x in prompts:
        for y in responses[x]:
            rewards[x][y] = compute_reward(x, y)  # binary or continuous

    # 4. Compute advantages with mean baseline
    for x in prompts:
        baseline[x] = mean(rewards[x])
        advantages[x] = [r - baseline[x] for r in rewards[x]]

    # 5. Add length penalty (for correct responses)
    for x in prompts:
        lens = [len(y) for y in responses[x]]
        min_len, max_len = min(lens), max(lens)
        for k, y in enumerate(responses[x]):
            if rewards[x][k] > 0:  # correct
                lp = 0.5 - (lens[k] - min_len) / (max_len - min_len + eps)
            else:  # incorrect
                lp = min(0, 0.5 - (lens[k] - min_len) / (max_len - min_len + eps))
            advantages[x][k] += λ_len * lp

    # 6. Policy gradient with L2 log-ratio regularization
    loss = 0
    for x in prompts:
        for k, y in enumerate(responses[x]):
            log_ratio = log_prob(π_θ, y|x) - log_prob(π_ref, y|x)
            policy_loss = -advantages[x][k] * log_prob(π_θ, y|x)
            reg_loss = (1/2) * τ * sum(log_ratio_per_token ** 2)
            loss += policy_loss + reg_loss

    loss.backward()
    optimizer.step()  # Muon optimizer
    optimizer.zero_grad()  # Reset at each iteration

    # 7. Update reference policy
    θ_ref = copy(θ)
```

### Step-by-Step Execution Trace (1 iteration, 1 prompt)

```
Input: x = "What is 17 × 23?"

Step 1: Sample K=4 responses from π_θ
  y1 = "17 × 23 = 17 × 20 + 17 × 3 = 340 + 51 = 391"  [T=20 tokens]
  y2 = "Let me think step by step. 17 × 23. First, 17 × 20 = 340..."  [T=45 tokens]
  y3 = "17 × 23 = 391"  [T=8 tokens]
  y4 = "17 × 23 = 401"  [T=8 tokens, WRONG]

Step 2: Compute rewards
  r = [1.0, 1.0, 1.0, 0.0]  (binary correct/incorrect)

Step 3: Baseline
  b = mean(r) = 0.75
  advantages = [0.25, 0.25, 0.25, -0.75]

Step 4: Length penalty (correct only)
  lengths = [20, 45, 8, 8]
  min_len = 8, max_len = 45
  lp1 = 0.5 - (20-8)/(45-8) = 0.5 - 0.324 = +0.176  (moderate bonus)
  lp2 = 0.5 - (45-8)/(45-8) = 0.5 - 1.0 = -0.500   (longest penalty)
  lp3 = 0.5 - (8-8)/(45-8) = 0.5 - 0.0 = +0.500    (shortest bonus!)
  lp4 = min(0, 0.5 - 0.0) = 0.0                      (wrong but short, no bonus)

Step 5: Final advantages (with λ_len = 0.1)
  a = [0.25 + 0.018, 0.25 - 0.05, 0.25 + 0.05, -0.75 + 0.0]
  a = [0.268, 0.200, 0.300, -0.750]

Step 6: Gradient update
  → y3 gets HIGHEST advantage (correct AND concise)
  → y2 gets LOWEST positive advantage (correct but verbose)
  → y4 gets strong negative (wrong)
  → Model learns: prefer concise correct answers
```

### Comparison to Baseline Algorithms

| Feature | PPO | GRPO | K1.5 Policy Mirror Descent |
|---------|-----|------|---------------------------|
| Value network | Yes | No | No |
| Reference policy | Frozen SFT | Frozen SFT | Moving (previous iteration) |
| KL regularization | Clip ratio | KL penalty | L2 on log-ratio |
| Baseline | Value function | Group mean | Group mean |
| Optimizer | Adam | Adam | Muon |
| Length control | None | None | Explicit length penalty |
| Credit assignment | GAE (per-token) | Outcome-only | Outcome-only |

**What is NEW vs prior work:**
1. Moving reference policy (online mirror descent vs fixed reference)
2. L2 log-ratio regularization (symmetric vs asymmetric KL)
3. Optimizer reset each iteration (prevents momentum from old policy)
4. Explicit length penalty (separate from reward)
5. No value network by design (not a simplification — a deliberate choice)

## 3.2 K2 Extensions — Budget Control + Self-Critique

### Budget Control Algorithm

```python
def apply_budget_control(response, max_budget):
    """Per-sample token budget enforcement."""
    if len(response) > max_budget:
        response = response[:max_budget]  # Truncate
        reward -= budget_penalty           # Penalty for exceeding budget
    return response, reward
```

This is simpler than K2.5's Toggle but effective: hard truncation teaches the model to front-load important content.

### Self-Critique Reward Pipeline

```python
def compute_self_critique_reward(prompt, response, model):
    """Bootstrap self-evaluation capability."""

    # During SFT: train model to evaluate (prompt, response) pairs
    # During RL: use the model's own evaluation as reward

    # 1. Generate rubric evaluation
    rubric_prompt = format_rubric(prompt, response, rubric_type)
    evaluation = model.generate(rubric_prompt)

    # 2. Extract score from evaluation
    score = parse_score(evaluation)  # 0-10 scale

    # 3. Normalize to reward signal
    reward = (score - 5) / 5  # Center at 0, range [-1, 1]

    # 4. Closed-loop refinement
    # On verifiable tasks: compare self-critique score with ground truth
    # If self-critique is wrong: use this as training signal for the critic

    return reward
```

**Three rubric types:**
1. **Core rubric**: General quality (coherence, helpfulness, accuracy)
2. **Prescriptive rubric**: Task-specific criteria (format, length, style)
3. **Human-annotated rubric**: Gold-standard evaluations for calibration

### PTX Loss (Prevent Catastrophic Forgetting)

```python
# During RL, add auxiliary loss on curated text samples
total_loss = rl_loss + λ_ptx * cross_entropy(model, curated_text_batch)
```

This prevents the model from "forgetting" general capabilities while optimizing for RL rewards. Similar to RLHF's pretraining loss mixing.

## 3.3 K2.5 Extensions — Toggle + Agent Swarm

### Toggle Algorithm (Token-Efficient RL)

```python
def toggle_rl_step(iteration, m=2, ρ=0.9, λ_acc=7/8):
    """Alternating budget-limited and standard RL phases."""

    # Compute current accuracy across batch
    mean_accuracy = compute_batch_accuracy()

    if mean_accuracy > λ_acc and (iteration // m) % 2 == 1:
        # BUDGET-LIMITED PHASE
        # Compute budget as ρ-th percentile of correct response lengths
        correct_lengths = [len(y) for y in responses if reward(y) > 0]
        budget = percentile(correct_lengths, ρ * 100)  # 90th percentile

        for response in responses:
            if len(response) > budget:
                response = truncate(response, budget)
                reward -= truncation_penalty
    else:
        # STANDARD PHASE — no budget constraint
        pass

    # Regular RL update with (possibly truncated) responses
    compute_policy_gradient(responses, rewards)
```

**Why alternating, not permanent budget?**
- Permanent budget → model can't explore long-form reasoning
- Alternating → model learns BOTH long and short strategies
- Budget only when accuracy is high (λ > 7/8) → don't constrain when model is still learning
- ρ=90th percentile → allows top 10% of responses to be longer than budget

### Agent Swarm (PARL) Algorithm

```python
def parl_training_step(orchestrator, subagents, task):
    """
    PARL = Parallel Agent RL
    Orchestrator decides how to decompose task into parallel subtasks.
    Subagents are frozen (from intermediate training checkpoints).
    """

    # Orchestrator generates decomposition plan
    plan = orchestrator.generate(task)  # Uses create_subagent, assign_task tools

    # Execute subtasks in parallel
    results = parallel_execute(plan, subagents)

    # Compute composite reward
    r_perf = task_performance_reward(results, task.ground_truth)
    r_parallel = parallelism_reward(plan)    # Penalize serial execution
    r_finish = completion_reward(plan)        # Penalize unfinished subtasks

    total_reward = r_perf + λ1 * r_parallel + λ2 * r_finish

    # λ1, λ2 are annealed to 0 during training
    # Initially: force parallel structure
    # Eventually: only task performance matters

    # Only orchestrator is updated (subagents frozen)
    update_policy(orchestrator, total_reward)
```

**Why freeze subagents?**
- Joint training of orchestrator + subagents is unstable (non-stationary environment)
- Frozen subagents = deterministic "tools" for the orchestrator
- Subagents from intermediate checkpoints have diverse capabilities
- Orchestrator learns to match subtasks to appropriate agent capabilities

**Why auxiliary rewards annealed to zero?**
- `r_parallel` prevents "serial collapse" (orchestrator assigns all tasks to one agent sequentially)
- `r_finish` prevents "spurious parallelism" (spawn many agents but don't use results)
- Once the orchestrator learns the structure, only performance matters
- Keeping auxiliary rewards would constrain the orchestrator's flexibility for complex tasks

---

# STEP 4 — TRAINING PIPELINE (🔁 TRAINING PIPELINE AGENT)

## 4.1 End-to-End Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      KIMI TRAINING PIPELINE                        │
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐  │
│  │ Pre-train │───→│ SFT      │───→│ Long-CoT │───→│ RL           │  │
│  │ 15.5T tok │    │ ~2M ex   │    │ SFT      │    │ Online PMD   │  │
│  │ MuonClip  │    │ 2 stages │    │ warmup   │    │              │  │
│  └──────────┘    └──────────┘    └──────────┘    │  ┌─────────┐ │  │
│                                                   │  │ Sample  │ │  │
│  K2 additions:                                    │  │ K resp  │ │  │
│  ┌──────────┐                                     │  └────┬────┘ │  │
│  │ Agentic  │──→ (tool specs + agents + trajs)    │       ↓      │  │
│  │ Data Syn │                                     │  ┌─────────┐ │  │
│  └──────────┘                                     │  │ Reward  │ │  │
│                                                   │  │ Compute │ │  │
│  K2.5 additions:                                  │  └────┬────┘ │  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    │       ↓      │  │
│  │ Zero-Vis │───→│ Visual   │───→│ Joint    │    │  ┌─────────┐ │  │
│  │ SFT      │    │ RL       │    │ Multi RL │    │  │ Policy  │ │  │
│  └──────────┘    └──────────┘    └──────────┘    │  │ Update  │ │  │
│                                                   │  └────┬────┘ │  │
│                        ┌──────────┐               │       ↓      │  │
│                        │ PARL     │               │  ┌─────────┐ │  │
│                        │ (Swarm)  │               │  │ Update  │ │  │
│                        └──────────┘               │  │ θ_ref   │ │  │
│                                                   │  └─────────┘ │  │
│                        ┌──────────┐               └──────────────┘  │
│                        │ Toggle   │                                  │
│                        │ (Effic.) │                                  │
│                        └──────────┘                                  │
└─────────────────────────────────────────────────────────────────────┘
```

## 4.2 Detailed Data Flow: One RL Iteration

```
┌──────────────┐
│ Prompt Pool   │  Curriculum: easy→hard over iterations
│ (Math/Code/   │  Prioritized: weight ∝ (1 - pass_rate)
│  General/     │
│  Agentic)     │
└──────┬───────┘
       │ Sample B prompts
       ↓
┌──────────────┐
│ Generation    │  For each prompt x:
│ (Inference)   │    Sample K=4 responses from π_θ
│               │    Temperature = 1.0, top_p = 0.95
│               │    Max tokens = 96K (K2.5 reasoning)
└──────┬───────┘
       │ B×K responses
       ↓
┌──────────────┐
│ Reward        │  Domain-specific:
│ Computation   │  ├─ Math: CoT-RM (98.5% accuracy)
│               │  ├─ Code: CYaRon sandbox (50 tests/problem)
│               │  ├─ Verifiable: Rule-based binary
│               │  ├─ Instruction: Code interp + LLM judge + hack-check
│               │  ├─ Faithfulness: FACTS sentence-level judge
│               │  └─ Safety: Human + adversarial prompts
└──────┬───────┘
       │ B×K rewards ∈ R
       ↓
┌──────────────┐
│ Advantage     │  For each prompt x:
│ Computation   │    b(x) = mean(rewards[x])
│               │    A(x,k) = r(x,k) - b(x) + λ_len * len_penalty(x,k)
└──────┬───────┘
       │ B×K advantages
       ↓
┌──────────────┐
│ Policy Loss   │  L_policy = -Σ A(x,k) * log π_θ(y_k|x)
│ Computation   │  L_reg = (τ/2) * Σ (log π_θ - log π_ref)²
│               │  L_ptx = λ_ptx * CE(π_θ, curated_text) [K2 only]
│               │  L_total = L_policy + L_reg + L_ptx
└──────┬───────┘
       │ Scalar loss
       ↓
┌──────────────┐
│ Gradient      │  loss.backward()
│ + Optimizer   │  Muon step (Newton-Schulz + QK-Clip)
│               │  optimizer.zero_grad()
└──────┬───────┘
       │ Updated θ
       ↓
┌──────────────┐
│ Reference     │  θ_ref ← θ  (update every iteration)
│ Update        │  optimizer.reset() [K1.5: full reset each iteration]
└──────────────┘
```

## 4.3 Data Structures

```python
# Prompt buffer
class PromptPool:
    prompts: List[str]           # All available prompts
    difficulty: List[float]      # [0,1] difficulty score
    success_rates: Dict[str, float]  # Running pass@K rate per prompt
    domain: List[str]            # "math", "code", "instruction", etc.

# RL batch
class RLBatch:
    prompts: Tensor[B]           # Tokenized prompts
    responses: Tensor[B, K, T]   # K samples per prompt, padded to T
    response_lengths: Tensor[B, K]  # Actual length of each response
    rewards: Tensor[B, K]        # Scalar reward per response
    log_probs: Tensor[B, K, T]   # Per-token log P under π_θ
    ref_log_probs: Tensor[B, K, T]  # Per-token log P under π_ref
    advantages: Tensor[B, K]     # Computed advantages

# For agentic tasks (K2+)
class AgenticTrace:
    prompt: str
    actions: List[ToolCall]      # Sequence of tool invocations
    observations: List[str]      # Tool execution results
    response: str                # Final answer
    tool_success: List[bool]     # Per-tool success
    total_tokens: int            # Budget tracking
```

## 4.4 Batch Shapes and Timing

### K1.5 (Text RL)
```
Prompts per iteration:  B = 64-256 (varies by domain)
Samples per prompt:     K = 4
Max response tokens:    T = 32768 (long-CoT) or 131072 (128K context)
Total tokens generated: B × K × T_avg ≈ 256 × 4 × 4096 ≈ 4M tokens/iteration
```

### K2 (Agentic RL)
```
Code sandbox instances: 10,000+ concurrent (K8s)
Tool execution time:    Variable (ms to minutes per tool call)
Trajectory length:      15-100 steps for orchestrator
Reward computation:     Async (sandbox results may arrive out of order)
```

### K2.5 (Multimodal + Agent Swarm)
```
Image tokens:           Up to 64K per image
Video tokens:           Up to 3.2M pixels (4 frames × 800K pixels)
Agent Swarm:            Orchestrator + N frozen subagents
Subagent steps:         50-100 per subtask
Total pipeline time:    Dominated by generation (inference), not training
```

## 4.5 Stability Mechanisms

### Loss Spike Prevention
1. **QK-Clip** (MuonClip): Per-head attention logit capping, τ=100
2. **Optimizer reset**: Clear momentum/velocity at start of each RL iteration
3. **L2 log-ratio regularization**: Prevents large policy shifts
4. **Gradient clipping**: Standard 1.0 max norm (implied, standard practice)

### Mode Collapse Prevention
1. **Temperature = 1.0 throughout RL** (no premature exploitation)
2. **Curriculum sampling**: Maintain diverse difficulty levels
3. **Prioritized sampling**: Under-represented problems get more weight
4. **PTX loss** (K2): Auxiliary CE loss on curated text prevents forgetting

### Length Explosion Prevention
1. **Length penalty** (K1.5): Reward bonus for concise correct responses
2. **Budget control** (K2): Hard truncation + penalty for exceeding token budget
3. **Toggle** (K2.5): Alternating budget-limited/standard phases
4. **Partial rollouts** (K1.5 long-context): Fixed output budget with replay buffer
5. **Repeat detection**: Early termination + penalty for repetitive outputs

## 4.6 Bottleneck Analysis

```
┌──────────────────────────────────────────────────┐
│           TIME BREAKDOWN (Estimated)              │
│                                                    │
│  Generation (inference):  60-70% of wall time     │
│  ├─ Long-CoT generation dominates                 │
│  ├─ Tool execution (agentic tasks) adds latency   │
│  └─ K=4 samples = 4× inference cost               │
│                                                    │
│  Reward computation:      10-20%                   │
│  ├─ Code sandbox: seconds per test suite           │
│  ├─ CoT-RM: forward pass of reward model           │
│  └─ Rule-based: negligible                         │
│                                                    │
│  Training (gradient):     10-15%                   │
│  ├─ Forward pass (policy + ref)                    │
│  ├─ Backward pass                                  │
│  └─ Muon step (NS5 + QK-Clip)                     │
│                                                    │
│  Weight transfer:         <5%                      │
│  ├─ RDMA via Mooncake: <1min training→inference    │
│  └─ ~10s inference→training                        │
│                                                    │
│  BOTTLENECK: Generation is 60-70% of time          │
│  → Key optimization: reduce K, reduce T_max,       │
│    parallelize generation across GPUs              │
└──────────────────────────────────────────────────┘
```

---

# STEP 5 — INFRASTRUCTURE MAPPING (🏗 INFRASTRUCTURE AGENT)

## 5.1 Hardware Architecture

### K2 Training Cluster

```
┌─────────────────────────────────────────────────────┐
│                    K2 CLUSTER                        │
│                                                      │
│  GPU: NVIDIA H800                                    │
│  Interconnect: 8×400 Gbps RoCE per node              │
│  RAM: 2 TB per node                                  │
│  Min allocation: 32 nodes (256 GPUs)                 │
│                                                      │
│  Model-Parallel Group: 256 GPUs                      │
│  ├─ 16-way Pipeline Parallelism (PP)                 │
│  └─ 16-way Expert Parallelism (EP)                   │
│                                                      │
│  Data Parallelism: ZeRO-1 across model-parallel      │
│  groups                                               │
└─────────────────────────────────────────────────────┘
```

### Memory Budget per GPU

```
Total GPU memory: 80 GB (H800)

Parameters:
  - 1.04T total / 256 GPUs (PP×EP) ≈ 4B params/GPU
  - BF16: 4B × 2 bytes = 8 GB

Optimizer states (Muon):
  - Momentum: 8 GB
  - NS5 workspace: ~2 GB
  - Total optimizer: ~10 GB

Activations:
  - Selective recomputation reduces peak by ~40%
  - FP8-E4M3 storage for MoE activations: ~2× savings
  - Estimated: ~8 GB with recomputation

Gradients:
  - FP32 accumulation: 4B × 4 bytes = 16 GB
  - But distributed across ZeRO-1

CPU offloading:
  - Overlapped copy engine hides latency
  - Offloads optimizer states and gradients as needed

TOTAL: ~30 GB/GPU (confirmed in paper)
Headroom: 50 GB for dynamic allocation, KV cache, etc.
```

## 5.2 Parallelism Strategy

### Pipeline Parallelism (PP=16)

```
61 layers distributed across 16 pipeline stages
≈ 3.8 layers per stage (with virtual stages for load balancing)

Interleaved 1F1B schedule:
  - Each micro-batch is split across stages
  - Forward and backward passes are interleaved
  - Virtual stages reduce pipeline bubble from O(p-1)/O(m+p-1) to O(p-1)/O(m×c+p-1)
  where p=pipeline stages, m=micro-batches, c=chunks per micro-batch

Pipeline bubble: <5% with sufficient micro-batches
```

### Expert Parallelism (EP=16)

```
384 total experts / 16 EP = 24 experts per GPU

All-to-all communication for expert dispatch:
  - Each token's top-8 experts may reside on different GPUs
  - Communication volume: batch_size × top_k × hidden_dim × 2 (dispatch + combine)

Why EP=16 (not 32 or 64)?
  - 64 attention heads ÷ 16 = 4 heads per EP group
  - Ensures full computation-communication overlap
  - Larger EP = more communication = less overlap
  - K2 explicitly chose 64 heads (not 128) to enable EP=16
```

### Communication Pattern

```
Within a pipeline stage:
  All-to-all for expert dispatch/combine (EP dimension)

Between pipeline stages:
  Point-to-point for activation transfer (PP dimension)

Across data-parallel groups:
  All-reduce for gradient synchronization (ZeRO-1)

Total bandwidth per GPU:
  8 × 400 Gbps = 3.2 Tbps bidirectional
  Per-step communication: ~100 GB estimated
  Communication time: ~250 ms
  Computation time: ~750 ms
  Overlap efficiency: ~90%+
```

## 5.3 RL Infrastructure

### Colocated Training + Inference

```
┌──────────────────────────────────────────┐
│           SINGLE GPU NODE                 │
│                                           │
│  ┌────────────┐    ┌────────────┐        │
│  │  Megatron   │    │   vLLM     │        │
│  │  (Training) │    │ (Inference)│        │
│  │  Container  │    │ Container  │        │
│  └──────┬─────┘    └──────┬─────┘        │
│         │                  │              │
│         │  K8s Sidecar     │              │
│         │  orchestration   │              │
│         ↓                  ↓              │
│  ┌─────────────────────────────────┐     │
│  │         GPU Memory               │     │
│  │  Shared model weights (RDMA)     │     │
│  └─────────────────────────────────┘     │
└──────────────────────────────────────────┘

Weight Transfer:
  Training → Inference: <1 minute via Mooncake (RDMA)
  Inference → Training: ~10 seconds

Full 1T parameter update: <30 seconds
```

### Code Sandbox Infrastructure

```
┌──────────────────────────────────────────┐
│           CODE EXECUTION                  │
│                                           │
│  Custom crun runtime:                     │
│  - 120 containers/sec (vs 27/sec Docker) │
│  - 4.4× throughput improvement            │
│                                           │
│  K8s orchestration:                       │
│  - 10,000+ concurrent sandbox instances   │
│  - Isolated execution per code sample     │
│  - CYaRon: 50 auto-generated tests/prob  │
│                                           │
│  Latency: <100ms container start          │
│  Per-test execution: <5 seconds           │
│  Total per problem: ~250 seconds          │
│  (50 tests × 5 seconds)                   │
└──────────────────────────────────────────┘
```

## 5.4 Scaling Analysis

### Pre-training Throughput

```
Total tokens:       15.5T
Estimated time:     ~2-3 months (based on similar-scale training runs)
Throughput:         ~2-3T tokens/month
Tokens/GPU/second:  ~15,000 (estimated from batch size / step time)
MFU:                ~45-55% (typical for MoE with EP communication overhead)
```

### RL Throughput

```
Generation bottleneck:
  - K=4 samples × T_avg tokens per iteration
  - Inference throughput with vLLM: ~5000 tokens/sec/GPU
  - For 1T model on 256 GPUs: ~1.28M tokens/sec

Training step:
  - Full parameter update: <30 seconds
  - This is remarkably fast for a 1T model

Iteration time:
  - Generate: ~30-60 seconds (depending on T_avg)
  - Reward: ~10-30 seconds (including sandbox)
  - Train: ~30 seconds
  - Total: ~70-120 seconds per iteration
```

### Cost Estimates

```
Hardware: 256× H800 GPUs minimum (32 nodes)

Pre-training (15.5T tokens):
  - ~2-3 months × 256 GPUs
  - At $2/GPU-hr: ~$700K-$1M

RL training:
  - Same cluster (colocated)
  - Duration: weeks to months depending on convergence
  - Additional cost: $200K-$500K estimated

Total: $1-2M estimated (comparable to DeepSeek V3's reported ~$5.5M)
```

## 5.5 NanoSeek Infrastructure Mapping

```
NanoSeek (1B active, single A6000):

Pre-training:
  - 22B tokens
  - Single GPU (48 GB VRAM)
  - Estimated: 3-7 days at ~100K tokens/sec
  - No parallelism needed

RL (GRPO):
  - Same GPU, sequential generation + training
  - K=4 samples, much shorter context
  - No separate inference engine needed (model fits in memory)
  - Simple Python sandbox for code reward

Key differences from K2:
  - No EP/PP needed (single GPU)
  - No weight transfer overhead
  - No container orchestration
  - Bottleneck: still generation (~60% of time)
```

---

# STEP 6 — EMPIRICAL RESULTS (📊 EVIDENCE & VALIDATION AGENT)

## 6.1 Claim → Evidence Mapping

### Claim 1: "RL without value networks outperforms PPO-style approaches"

**Evidence (K1.5, Section 3):**
- K1.5 reports results on AIME 2024, MATH, LiveCodeBench
- Comparison: K1.5 RL vs "standard approaches" (PPO, MCTS, PRM)
- K1.5 achieves AIME 2024: 69.6% (vs prior SOTA ~60%)

**Strength: MODERATE**
- No direct ablation: K1.5 with value network vs without
- Claim is argued theoretically (value functions penalize exploration) but not experimentally verified on same model
- Other differences (Muon optimizer, curriculum, length penalty) confound the comparison

### Claim 2: "QK-Clip enables zero loss spikes at 1T scale"

**Evidence (K2, Section 2.3):**
- K2 trained 15.5T tokens with zero loss spikes (stated as fact)
- QK-Clip trigger rate: 12.7% of heads in first 70K steps, then zero
- No comparison: K2 without QK-Clip (too expensive to ablate at 1T scale)

**Strength: STRONG (empirical) / WEAK (causal)**
- The empirical result (zero spikes) is verifiable from training logs
- But the CAUSAL claim (QK-Clip caused this) is not isolated
  - Maybe the learning rate schedule prevents spikes anyway
  - Maybe Muon with proper initialization is already stable
  - The self-deactivating property (triggers drop to 0) suggests it may be a safety net that rarely activates

### Claim 3: "Sparsity scaling law: more experts at fixed FLOPs consistently improves loss"

**Evidence (K2, Section 2.2):**
- Ablation at smaller scale with sparsity {8, 16, 24, 32, 48}
- At sparsity 48: 1.69× FLOPs reduction vs sparsity 8 for same validation loss
- Graph shows monotonic improvement (Figure in paper)

**Strength: STRONG**
- This is a proper ablation with controlled variables
- Fixed activated parameters = fixed FLOPs budget
- Multiple sparsity levels tested
- Consistent trend

**Caveat:** Ablation done at sub-1T scale. May not hold at all scales (diminishing returns possible).

### Claim 4: "Data rephrasing > multi-epoch training"

**Evidence (K2, Section 2.4):**
- SimpleQA benchmark: 10 rephrasings × 1 epoch = 28.94 vs raw × 10 epochs = 23.76
- +5.18 improvement from rephrasing

**Strength: MODERATE**
- Single benchmark (SimpleQA) — might not generalize
- Only knowledge data tested, not code or math
- No ablation on rephrasing method (chunk-wise vs whole-doc vs different LLMs)
- Cost of rephrasing (LLM inference) not reported

### Claim 5: "Zero-Vision SFT activates visual capabilities"

**Evidence (K2.5, Section 3.1):**
- Using ONLY text SFT data, visual benchmark performance emerges
- Adding human-designed visual trajectories HURTS generalization
- Works because joint pre-training already established vision-text alignment

**Strength: STRONG (surprising and well-evidenced)**
- Counter-intuitive result is more convincing than expected results
- Comparison: text-only SFT vs text+visual SFT (the latter is worse)
- Multiple visual benchmarks tested

### Claim 6: "Cross-modal transfer: Visual RL improves text performance"

**Evidence (K2.5, Section 3.2):**
- +1.7 MMLU-Pro, +2.1 GPQA-Diamond from visual RL
- Mechanism: shared representations in the MoE backbone

**Strength: MODERATE**
- Numbers are specific and testable
- But MMLU-Pro and GPQA improvements could be within noise
- No error bars or confidence intervals reported
- Correlation ≠ causation: other training changes between checkpoints

### Claim 7: "Agent Swarm achieves 3-4.5× speedup"

**Evidence (K2.5, Section 3.4):**
- BrowseComp: 78.4% (SOTA)
- Speedup: 3-4.5× over single-agent

**Strength: MODERATE**
- Speedup is wall-clock time, not FLOPs
- Depends heavily on task parallelizability
- Benchmark may be biased toward tasks that benefit from parallelism
- No analysis of when swarm FAILS (tasks that are inherently sequential)

### Claim 8: "Toggle reduces tokens 25-30% with negligible performance impact"

**Evidence (K2.5, Section 3.5):**
- 25-30% token reduction stated
- "Negligible performance impact" — but no numbers given for the impact

**Strength: WEAK-MODERATE**
- Token reduction is specific and measurable
- But "negligible" is undefined — is it 0.1%? 1%? 3%?
- No ablation comparing Toggle vs no Toggle on same benchmarks with error bars

## 6.2 Reproducibility Assessment

### What CAN Be Reproduced

1. **QK-Clip mechanism**: Fully specified, implementable from description
2. **Length penalty**: Exact formula given
3. **Toggle algorithm**: Exact algorithm with hyperparameters
4. **Sparsity scaling law**: Ablation methodology is clear

### What CANNOT Be Reproduced

1. **Training data**: 15.5T tokens, proprietary data mix
2. **Data rephrasing pipeline**: LLM used for rephrasing not specified
3. **CoT-RM**: 98.5% accuracy reward model — architecture and training data not shared
4. **Agentic data synthesis**: Tool specs, agent prompts, trajectory generation details insufficient
5. **Self-critique rubrics**: "Core", "prescriptive", "human-annotated" — examples not given
6. **PARL reward coefficients**: λ₁, λ₂ values and annealing schedule not specified
7. **Full training hyperparameters**: Many RL-specific HPs (KL coefficient τ, length penalty weight, PTX loss weight) not reported

### Reproducibility Score: 3/10

The **mechanisms** are reproducible, but the **full system** is not. This is typical for industry papers — the ideas are transferable but exact replication requires proprietary components.

## 6.3 Ablation Interpretation

### Well-Designed Ablations
1. **Sparsity scaling** (K2): Controlled variable (activated params), multiple levels
2. **Attention heads** (K2): Clear cost-benefit analysis (83% FLOP increase for 0.5-1.2% gain)
3. **Zero-Vision SFT** (K2.5): Direct comparison of text-only vs text+visual SFT

### Missing Ablations
1. **Value network vs no value network**: Theoretical argument only, no experimental comparison
2. **L2 log-ratio vs KL**: No comparison of the two regularization strategies
3. **Optimizer reset frequency**: Does resetting every iteration vs every N iterations matter?
4. **K value**: Is K=4 optimal? K=2? K=8?
5. **Length penalty weight**: Sensitivity to λ_len not reported
6. **Toggle hyperparameters**: No sensitivity analysis for ρ, λ, m

---

# STEP 7 — CRITICAL ANALYSIS (🔍 CRITIC AGENT)

## 7.1 Hidden Assumptions

### Assumption 1: Mean Baseline is Sufficient
**The claim**: Simple mean reward is an adequate baseline, no value function needed.

**The assumption**: The variance reduction from a learned value function does not outweigh the exploration penalty it creates.

**When this breaks**:
- **Very long sequences** (>100K tokens): The advantage signal becomes increasingly noisy with outcome-only reward. At T=100K, each token gets the same advantage — no credit assignment at all.
- **Sparse rewards**: If only 1/K responses is correct (math problems with low pass@4), the baseline is ≈0 and advantages are ≈ ±r. This provides very little gradient signal.
- **Multi-step agentic tasks**: A single wrong tool call at step 3 of 50 renders the entire trajectory useless, but the mean baseline can't identify which step was wrong.

### Assumption 2: Online Mirror Descent Converges
**The assumption**: Updating the reference policy every iteration produces stable convergence.

**When this breaks**:
- If the policy makes a large improvement in one iteration, the new reference is far from the old one. The next iteration's KL penalty is measured against a potentially poor reference.
- No formal convergence guarantee is provided (unlike PPO's clipped objective).
- The optimizer reset mitigates this (no stale momentum), but doesn't guarantee convergence.

### Assumption 3: Curriculum Sampling Helps
**The assumption**: Starting with easy problems and progressing to hard ones improves final performance.

**When this breaks**:
- If easy and hard problems require fundamentally different strategies, curriculum may cause the model to over-index on easy-problem strategies that don't transfer.
- The paper doesn't show curriculum vs random sampling ablation.

### Assumption 4: Self-Critique Is Calibrated
**The assumption**: The model can accurately evaluate its own outputs across all domains.

**When this breaks**:
- **Bias amplification**: If the model has systematic biases, its self-critique will share those biases
- **Dunning-Kruger in LLMs**: Models may be overconfident on topics where they perform poorly
- **Closed-loop instability**: Self-critique reward → policy update → new self-critique → could diverge
- The closed-loop refinement on verifiable tasks mitigates this, but only for verifiable domains

## 7.2 Failure Modes

### Failure Mode 1: Reward Hacking
**Scenario**: The model learns patterns that maximize reward without achieving the intended behavior.

**Specific risks**:
- **CoT-RM hacking**: The reward model has 98.5% accuracy — what about the 1.5% failure rate? Over millions of RL samples, the model will find and exploit these failure modes.
- **Length penalty gaming**: Model produces extremely terse correct answers that lack explanation, optimizing for length reward over quality.
- **Self-critique collusion**: Model learns to generate outputs that its self-critique rates highly, even if they're not actually good.

**Mitigation in K2**: Hybrid verification (code interpreter + LLM judge + hack-check). The "hack-check" specifically looks for outputs that game the reward. But this is an arms race.

### Failure Mode 2: Distribution Shift During RL
**Scenario**: The RL policy drifts far from the pre-trained distribution, losing general capabilities.

**Specific risks**:
- **Math mode**: Model becomes very good at math but forgets how to write prose
- **Tool overuse**: Agentic model calls tools even when direct response is better
- **Verbosity collapse**: Without proper length penalties, model generates increasingly long responses

**Mitigation in K2**: PTX loss on curated samples. But the balance between RL objective and PTX loss is delicate — too much PTX = no RL progress, too little = catastrophic forgetting.

### Failure Mode 3: Agent Swarm Coordination Failure
**Scenario**: The orchestrator creates plans that look parallel but don't actually decompose the task well.

**Specific risks**:
- **Over-decomposition**: Breaking a simple task into many subtasks, each of which lacks context
- **Dependency blindness**: Assigning tasks that depend on each other to parallel execution
- **Communication overhead**: Subagent results need to be integrated — the orchestrator may struggle with synthesis

**Mitigation in K2.5**: Auxiliary rewards (r_parallel, r_finish). But these are annealed to zero, so eventually the orchestrator is only optimized for task performance — may still exhibit failure modes at test time.

### Failure Mode 4: Toggle Destroys Long-Reasoning Capability
**Scenario**: Budget-limited phases teach the model to truncate reasoning, which generalizes to non-budget phases.

**Specific risks**:
- Model internalizes "shorter is always better"
- Complex problems that genuinely need long reasoning get truncated answers
- The alternating schedule may not be long enough for the model to maintain both strategies

**Partial mitigation**: Toggle only activates when accuracy > 7/8, so it doesn't constrain learning on hard problems. But once a problem becomes "easy" (high accuracy), the model may lose its long-reasoning capability for that problem class.

## 7.3 What's Missing from the Papers

1. **Failure rates in production**: How often does the agentic system fail in real deployment?
2. **Reward model training details**: Architecture, data, training procedure for the CoT-RM
3. **Actual KL coefficient values**: τ is a critical hyperparameter but never specified
4. **RL training duration**: How many iterations? How many total tokens generated during RL?
5. **Compute cost breakdown**: Pre-training vs RL vs evaluation costs
6. **Error analysis**: What types of problems does the model still fail on after RL?
7. **Ablation on K (samples per prompt)**: Is K=4 optimal? What's the variance reduction curve?
8. **Long-term stability of self-critique**: Does the closed-loop eventually diverge?

## 7.4 Comparison with Alternative Approaches

### vs PPO (OpenAI style)
| Aspect | K1.5/K2/K2.5 | PPO |
|--------|-------------|-----|
| Value network | None | Yes (critic) |
| Reference policy | Moving | Fixed (SFT) |
| Credit assignment | Outcome-only | GAE (per-token) |
| Advantage | Yes | Higher with GAE |
| Exploration | Natural (no value-based pruning) | Constrained by value function |
| Variance | Higher | Lower (with value baseline) |
| Compute | Lower (no value model) | Higher (2 forward passes) |
| Stability | L2 log-ratio + QK-Clip | Clip ratio + early stopping |

**When K1.5 approach wins**: Long-CoT reasoning, exploration-heavy tasks
**When PPO wins**: Short-response tasks, reward-dense environments, sample efficiency matters

### vs GRPO (DeepSeek style)
| Aspect | K1.5/K2/K2.5 | GRPO |
|--------|-------------|------|
| Reference policy | Moving | Fixed (SFT) |
| Regularization | L2 log-ratio | KL penalty |
| Optimizer | Muon | Adam |
| Length control | Explicit penalty + Toggle | Implicit through KL |
| Agentic | Yes (K2+) | No |
| Multimodal | Yes (K2.5) | No |

**Key difference**: Moving reference is more aggressive (allows larger policy changes per iteration) but requires optimizer reset for stability.

### vs Process Reward Models (PRM)
| Aspect | K1.5 | PRM-based |
|--------|------|-----------|
| Credit assignment | Outcome-only | Per-step |
| Training data | Easy (binary outcome) | Hard (requires per-step labels) |
| Exploration | Preserved | Constrained by per-step rewards |
| Scaling | Simple | Requires reward model per domain |
| Accuracy | Depends on pass@K | Depends on PRM quality |

**K1.5's argument**: PRMs require expensive per-step annotations and may constrain exploration. Outcome-based RL is simpler and scales better with compute.

**Counter-argument**: PRMs provide much richer gradient signal, enabling faster convergence. K1.5's success may be due to massive compute rather than algorithmic superiority.

## 7.5 Open Research Questions

1. **Can outcome-based RL scale to even longer contexts (1M+ tokens)?**
   - At some point, the variance of outcome-only reward must become prohibitive
   - Is there a crossover point where value functions become necessary?

2. **Is the moving reference policy provably convergent?**
   - No formal convergence analysis provided
   - Empirically works, but could fail on different problem distributions

3. **How does self-critique quality degrade over RL iterations?**
   - The critic is updated by the closed loop, but does it stay calibrated?
   - Could lead to systematic bias accumulation

4. **What is the optimal sparsity for a given model size?**
   - K2 shows sparsity 48 is good at 1T scale
   - Does the optimal sparsity change with scale? (Critical for NanoSeek at 4.75B total)

5. **Can Toggle be combined with Long2Short?**
   - K1.5 has Long2Short, K2.5 has Toggle
   - Are they complementary or redundant?

6. **Is Zero-Vision SFT specific to MoE architectures?**
   - Works because shared representations in the backbone
   - Would it work for dense models with separate vision/text components?

7. **Agent Swarm: optimal number of subagents?**
   - K2.5 uses intermediate checkpoints as subagents
   - Is there a principled way to choose how many and which checkpoints?

---

# STEP 8 — NANOSEEK APPLICABILITY ANALYSIS

## 8.1 What to Adopt (Ordered by Priority and Feasibility)

### Tier 1: Implement Now (Low effort, high value)

**1. QK-Clip** ★★★★★
- **Effort**: ~50 lines of code
- **Value**: Insurance against attention logit explosion
- **Implementation**: After each optimizer step, check max logit per head, rescale if > τ
- **NanoSeek-specific**: τ may need to be different at 1B scale (try τ=50 and τ=100)

**2. Length Penalty in GRPO** ★★★★
- **Effort**: ~20 lines of code
- **Value**: Prevents verbose reasoning chains
- **Implementation**: Exact formula from K1.5 (Section 2.3 above)

**3. Curriculum + Prioritized Sampling** ★★★★
- **Effort**: ~100 lines of code
- **Value**: Better sample efficiency during RL
- **Implementation**: Track per-prompt pass rates, sample proportional to (1 - pass_rate)

### Tier 2: Implement for Phase 5 (Moderate effort, high value)

**4. Toggle (Token-Efficient RL)** ★★★★
- **Effort**: ~150 lines of code
- **Value**: 25-30% token reduction during RL
- **Implementation**: Alternating budget phases, exact algorithm in Section 3.3

**5. Long2Short RL** ★★★
- **Effort**: Separate RL phase + length penalty
- **Value**: Transfer long-CoT capabilities to shorter responses
- **Implementation**: First train with long context, then RL with reduced max_len

**6. Self-Critique Reward** ★★★
- **Effort**: ~300 lines + SFT data curation
- **Value**: Enables RL on non-verifiable tasks (instruction following, etc.)
- **Implementation**: Bootstrap during SFT, use as reward during RL

### Tier 3: Consider for Future (High effort, uncertain value at 1B scale)

**7. Data Rephrasing** ★★
- **Value at NanoSeek scale**: Uncertain — 22B tokens may not benefit as much as 15.5T
- **Cost**: Requires LLM inference for rephrasing
- **Consider**: Only if ClimbMix data quality is a bottleneck

**8. Agent Swarm (PARL)** ★
- **Value at 1B scale**: Very limited — model too small for reliable multi-agent coordination
- **Consider**: Only after NanoSeek achieves strong single-agent tool use

## 8.2 What NOT to Adopt

1. **384 experts**: NanoSeek's 64 experts at sparsity 8 is appropriate for 4.75B total params. Going to higher sparsity requires proportionally more total params.

2. **EP/PP parallelism**: Single GPU training — not needed.

3. **Colocated training+inference**: At 1B scale, the model fits in memory for both generation and training — no need for separate containers.

4. **MuonClip full NS5**: NanoSeek already uses MuonAdamW. Only add QK-Clip, not the full MuonClip with different NS iterations.

---

# APPENDIX A: Complete Hyperparameter Reference

## K1.5 RL Hyperparameters (Known)
| Parameter | Value | Source |
|-----------|-------|--------|
| Context (final) | 128K | Paper Section 3 |
| Sampling K | 4 (implied) | Paper Section 3 |
| Optimizer | Muon | Paper Section 3 |
| Length penalty | Exact formula given | Paper Section 3.2 |
| Reward (math) | CoT-RM, 98.5% accuracy | Paper Section 3.3 |
| Reward (code) | CYaRon, 50 tests/problem | Paper Section 3.3 |
| Curriculum | Easy → hard | Paper Section 3.4 |
| Prioritized sampling | weight ∝ (1 - pass_rate) | Paper Section 3.4 |

## K1.5 RL Hyperparameters (UNKNOWN)
| Parameter | Status |
|-----------|--------|
| τ (KL coefficient) | Not reported |
| λ_len (length penalty weight) | Not reported |
| K (samples per prompt) | Not explicitly stated |
| B (batch size during RL) | Not reported |
| Total RL iterations | Not reported |
| Total RL compute | Not reported |

## K2 RL Hyperparameters (Known)
| Parameter | Value | Source |
|-----------|-------|--------|
| QK-Clip τ | 100 | Paper Section 2.3 |
| Weight decay | 0.1 | Paper Section 2.3 |
| SFT optimizer | Muon | Paper Section 3.1 |
| Budget control | Per-sample max + truncation | Paper Section 3.2 |
| PTX loss | Auxiliary CE on curated text | Paper Section 3.2 |
| Temperature | High → low (decay schedule) | Paper Section 3.2 |
| Sandbox | 10K+ concurrent K8s instances | Paper Section 3.3 |

## K2.5 RL Hyperparameters (Known)
| Parameter | Value | Source |
|-----------|-------|--------|
| RL temperature | 1.0 | Paper Section 3 |
| RL top-p | 0.95 | Paper Section 3 |
| Max completion | 96K tokens | Paper Section 3 |
| Toggle λ (accuracy threshold) | 7/8 | Paper Section 3.5 |
| Toggle m (phase period) | 2 iterations | Paper Section 3.5 |
| Toggle ρ (budget percentile) | 90% | Paper Section 3.5 |
| Agent Swarm steps (orchestrator) | 15-100 | Paper Section 3.4 |
| Agent Swarm steps (subagents) | 50-100 | Paper Section 3.4 |

---

# APPENDIX B: Equation Summary

## Core RL Objective
```
max_θ  E_{y~π_θ} [r(x,y)]  -  τ · KL(π_θ || π_{θ_i})
```

## Policy Gradient (REINFORCE)
```
∇_θ J = E_{y~π_θ} [(r(x,y) - b(x)) · Σ_t ∇_θ log π_θ(y_t|x,y_{<t})]
```

## L2 Log-Ratio Regularization
```
L_reg = (τ/2) · E_{y~π_θ} [Σ_t (log π_θ(y_t|·) - log π_{θ_i}(y_t|·))²]
```

## Length Penalty
```
len_reward(i) = { 0.5 - (len(i)-min_len)/(max_len-min_len)        if correct
                { min(0, 0.5 - (len(i)-min_len)/(max_len-min_len)) if incorrect
```

## QK-Clip
```
S_max^h = (1/√d) · max_{X} max_{i,j} Q_i^h · K_j^{hT}
γ_h = min(1, τ / S_max^h)
W_qc^h ← √γ_h · W_qc^h,  W_kc^h ← √γ_h · W_kc^h,  W_qr^h ← γ_h · W_qr^h
```

## Toggle Budget
```
budget = P_ρ({len(y) : y correct})    [ρ-th percentile]
active  iff  mean_accuracy > λ  AND  (iteration // m) % 2 == 1
```

## PARL Reward
```
r = r_perf + λ₁·r_parallel + λ₂·r_finish
λ₁, λ₂ → 0  during training
```

## Muon Update (Newton-Schulz 5 iterations)
```
G ← G / ||G||_F
for i in 1..5:
    X ← aX + bX·X^T·X + cX·X^T·X·X^T·X    [coefficients from Bjorck-Bowie]
W ← W - η · (sqrt(max(n,m)) · 0.2) · X - λ·W
```
