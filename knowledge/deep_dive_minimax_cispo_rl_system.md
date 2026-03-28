# Deep Dive: MiniMax CISPO RL System — First-Principles Reconstruction
## Multi-Agent Research Analysis: Math → Algorithm → Pipeline → Infrastructure → Evidence → Critique
### Compiled: March 2026 | Primary sources: arXiv:2506.13585, arXiv:2501.08313, arXiv:2510.13786, arXiv:2508.07629, arXiv:2510.06062, official blogs, Swift docs

---

## Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [Mathematical Foundation (Math Agent)](#2-mathematical-foundation)
3. [Algorithm Construction (Algorithm Agent)](#3-algorithm-construction)
4. [Training Pipeline (Pipeline Agent)](#4-training-pipeline)
5. [Infrastructure Mapping (Infrastructure Agent)](#5-infrastructure-mapping)
6. [Empirical Results (Evidence Agent)](#6-empirical-results)
7. [Critical Analysis (Critic Agent)](#7-critical-analysis)
8. [Multi-Turn Extension: Agentic CISPO](#8-multi-turn-agentic-cispo)
9. [Forge Framework Deep Dive](#9-forge-framework)
10. [Self-Evolving Training Loop](#10-self-evolving-training-loop)
11. [NanoSeek Implementation Guide](#11-nanoseek-implementation-guide)
12. [Open Research Questions](#12-open-research-questions)

---

## 1. Problem Definition

### 1.1 What Is Being Solved

**Objective**: Train a pre-trained LLM (MiniMax-Text-01, 456B MoE) to produce long-form
reasoning chains that maximize verifiable task rewards, without destroying the model's
general capabilities or destabilizing its MoE routing.

**Environment**: The "environment" is the text generation process itself:
- **State**: The prompt + generated tokens so far: `s_t = (q, o_{<t})`
- **Action**: The next token: `a_t = o_t ∈ V` (vocabulary of 200K tokens)
- **Policy**: The LLM's token distribution: `π_θ(a_t | s_t)`
- **Episode**: A complete generation from prompt to EOS
- **Reward**: Sparse, at episode end: `R(q, o) ∈ {0, 1}` for verifiable tasks (math, code)

**Source**: arXiv:2506.13585, Section 3 (MiniMax-M1 RL training)

### 1.2 The Objective Function

Maximize expected reward over the prompt distribution:

```
J(θ) = E_{q~D} E_{o~π_θ(·|q)} [R(q, o)]
```

Subject to:
- KL constraint: `D_KL(π_θ || π_ref) ≤ δ` (stay close to pre-trained model)
- Stability: MoE routing must not collapse (H_load maintained)
- Capability: General capabilities must not degrade

**Source**: Standard RLHF objective (Ziegler et al., 2019; Ouyang et al., 2022)

### 1.3 Why This Is Hard for LLMs

1. **Sparse reward**: Only signal at the end of a 40K-80K token sequence
2. **Enormous action space**: |V| = 200,064 tokens per step
3. **Credit assignment**: Which of the 40K tokens mattered?
4. **Rare critical tokens**: Reasoning pivots ("However," "Wait," "Recheck") appear at
   frequency ~0.01% but determine correctness. Standard RL suppresses their gradients.
5. **MoE instability**: RL policy updates can destabilize expert routing, causing
   catastrophic performance collapse
6. **Distribution shift**: Off-policy training with importance sampling at sequence level
   introduces high-variance gradient estimates

---

## 2. Mathematical Foundation

### 2.1 Policy Gradient Theorem (Starting Point)

**FACT** [Sutton et al., 2000; Williams, 1992]:

For any differentiable policy `π_θ`, the gradient of the expected reward is:

```
∇_θ J(θ) = E_{o~π_θ} [R(q, o) · ∇_θ log π_θ(o | q)]
```

Expanding the sequence-level log-probability into tokens:

```
log π_θ(o | q) = Σ_{t=1}^{T} log π_θ(o_t | q, o_{<t})
```

Therefore:

```
∇_θ J(θ) = E_{o~π_θ} [R(q, o) · Σ_t ∇_θ log π_θ(o_t | q, o_{<t})]
```

**MECHANISM**: Each token contributes a gradient direction (the score function
`∇ log π_θ`), weighted by the episode reward. Tokens that the policy makes more
likely get larger gradients. The reward tells us whether to reinforce or suppress
the entire trajectory.

**FAILURE MODE**: Variance is proportional to sequence length T and vocabulary size |V|.
For T = 40,000 and |V| = 200,064, the raw REINFORCE estimator is essentially useless
without variance reduction.

### 2.2 Baseline Subtraction (Variance Reduction)

**FACT** [Williams, 1992]:

Subtracting a state-dependent baseline `b(s)` from the reward does not change the
expected gradient but reduces variance:

```
∇_θ J(θ) = E_{o~π_θ} [(R(q, o) - b(q)) · Σ_t ∇_θ log π_θ(o_t | q, o_{<t})]
```

where `A(q, o) = R(q, o) - b(q)` is the **advantage**.

**GRPO's baseline** [Shao et al., 2024]: Use the group mean:

Given G completions `{o_1, ..., o_G}` per prompt q:

```
Â_i = (R_i - μ_G) / (σ_G + ε)

where:
  μ_G = (1/G) Σ_{j=1}^G R_j     (group mean)
  σ_G = std({R_1, ..., R_G})      (group std)
  ε = 1e-8                         (numerical stability)
```

**MECHANISM**: No separate value network needed. The group itself provides the baseline.
Positive advantage = "better than average in this group." This is the key insight that
makes GRPO/CISPO feasible without a critic.

**Why no value network** [independently validated by both MiniMax and Kimi]:
Value networks in long-CoT reasoning RL **punish correct intermediate reasoning steps**
that temporarily look wrong. A value network trained on final outcomes assigns low
value to exploratory reasoning tokens ("Let me reconsider..."), suppressing the
very behavior that leads to correct answers. Both MiniMax and Kimi discovered this
independently and abandoned value networks.

**Source**: arXiv:2506.13585 Section 3.1; arXiv:2501.12599 Section 3.2

### 2.3 Importance Sampling (Off-Policy Correction)

**FACT** [Owen, 2013]:

When generating completions from an old policy `π_{θ_old}` but training `π_θ`:

```
∇_θ J(θ) = E_{o~π_old} [r(θ) · Â · ∇_θ log π_θ(o_t | ...)]
```

where the importance sampling (IS) ratio is:

```
r_{i,t}(θ) = π_θ(o_t | q, o_{<t}) / π_{θ_old}(o_t | q, o_{<t})
```

**MECHANISM**: The IS ratio corrects for the distribution mismatch between the sampling
policy (old) and the policy being optimized (current). When `r > 1`, the current policy
assigns higher probability to this token than the old policy did. When `r < 1`, lower.

**FAILURE MODE**: IS ratios can be very large for rare tokens, causing extreme gradient
variance. A token that had probability 1e-5 under `π_old` and 1e-3 under `π_θ` has
ratio r = 100. This single token dominates the gradient estimate.

**Tensor shapes** (for a single training step):

```
prompt:        [B, S]                  B prompts, S tokens each
completions:   [B×G, S+T]              B×G sequences, T completion tokens
log_probs:     [B×G, T]                per-token log-probs
IS_ratios:     [B×G, T]                per-token importance weights
advantages:    [B×G]                   per-sequence advantage (scalar per completion)
```

### 2.4 PPO's Clipped Surrogate (The Problem CISPO Solves)

**FACT** [Schulman et al., 2017]:

PPO clips the surrogate objective to create a trust region:

```
L_PPO(θ) = E_t [min(r_t · Â_t, clip(r_t, 1-ε, 1+ε) · Â_t)]
```

The gradient is:

```
∇_θ L_PPO = E_t [M_t · r_t · Â_t · ∇_θ log π_θ]
```

where `M_t ∈ {0, 1}` is an **implicit mask**:

```
M_t = 0   if  Â_t > 0  AND  r_t > 1 + ε     (would increase already-too-high prob)
M_t = 0   if  Â_t < 0  AND  r_t < 1 - ε     (would decrease already-too-low prob)
M_t = 1   otherwise
```

**The critical problem for reasoning**: When `r_t` is clipped AND `Â_t` is small (which
it is for rare tokens that shift the reasoning direction), the gradient is **exactly zero**.

**NUMERIC EXAMPLE**:

Consider the token "However" at position t=15,234 in a 40K sequence:

```
π_old("However" | ...) = 0.003     (rare token)
π_θ("However" | ...)   = 0.008     (current policy slightly more likely)
r_t = 0.008 / 0.003 = 2.67         (exceeds 1+ε = 1.2 for PPO ε=0.2)

Â_t = 0.5                           (positive advantage, but moderate)

PPO gradient: 0 (masked!)           The token is outside the trust region.
The gradient for "However" is killed. PPO will never further increase its probability.
```

Now consider a common token "the" at position t=15,235:

```
π_old("the" | ...) = 0.15
π_θ("the" | ...)   = 0.16
r_t = 0.16 / 0.15 = 1.067          (within trust region)

Â_t = 0.5                           (same advantage)

PPO gradient: 1.067 × 0.5 × ∇log π_θ("the") ≠ 0   (non-zero gradient!)
```

**Conclusion**: PPO systematically suppresses rare reasoning tokens while reinforcing
common tokens. Over many training steps, this erodes the model's ability to produce
critical reasoning pivots.

**Source**: arXiv:2506.13585 Section 3.1; ScaleRL arXiv:2510.13786 Section 3

### 2.5 CISPO: The Derivation

**FACT** [MiniMax-M1, arXiv:2506.13585, Equation 4]:

CISPO modifies the objective to use **detached** (stop-gradient) clipped IS weights:

```
J_CISPO(θ) = E_{q~D, {o_i}~π_old} [
    (1/T_total) · Σ_{i=1}^G Σ_{t=1}^{|o_i|}
        sg(r̂_{i,t}(θ)) · Â_i · log π_θ(o_{i,t} | q, o_{i,<t})
]
```

where:

```
T_total = Σ_{i=1}^G |o_i|                                    (total tokens across group)

r_{i,t}(θ) = π_θ(o_{i,t} | ...) / π_{θ_old}(o_{i,t} | ...)  (per-token IS ratio)

r̂_{i,t}(θ) = clip(r_{i,t}(θ), 1 - ε_low, 1 + ε_high)       (clipped IS weight)

sg(·) = stop_gradient(·) = .detach()                           (KEY OPERATION)

Â_i = (R_i - μ_G) / (σ_G + ε)                                (group-relative advantage)
```

**DERIVATION of the gradient**:

```
∇_θ J_CISPO = (1/T_total) · Σ_i Σ_t [
    sg(r̂_{i,t}) · Â_i · ∇_θ log π_θ(o_{i,t} | ...)
  + ∇_θ[sg(r̂_{i,t})] · Â_i · log π_θ(o_{i,t} | ...)    ← THIS TERM IS ZERO
]
```

The second term vanishes because `sg()` blocks gradient flow. Therefore:

```
∇_θ J_CISPO = (1/T_total) · Σ_i Σ_t  sg(r̂_{i,t}) · Â_i · ∇_θ log π_θ(o_{i,t} | ...)
```

**MECHANISM**: Every token gets a gradient of the form `c · Â · ∇ log π_θ`, where
`c = sg(r̂_{i,t})` is a **constant scalar** (from the perspective of backpropagation).
This scalar can be larger or smaller (bounded by the clip), but it is **never zero**.

**Comparison of gradient coefficients per token**:

| Condition | CISPO coefficient | PPO coefficient |
|-----------|-------------------|-----------------|
| r_t < 1-ε_low AND Â < 0 | 1 - ε_low (constant, non-zero) | 0 (masked) |
| r_t > 1+ε_high AND Â > 0 | 1 + ε_high (constant, non-zero) | 0 (masked) |
| r_t < 1-ε_low AND Â > 0 | 1 - ε_low (may over-suppress) | r_t (unclipped, allows decrease) |
| r_t > 1+ε_high AND Â < 0 | 1 + ε_high (may over-reinforce decrease) | r_t (unclipped) |
| 1-ε_low ≤ r_t ≤ 1+ε_high | r_t (within clip bounds, identical) | r_t (identical) |

**Source**: arXiv:2506.13585 Equation 4-7; Klear-Reasoner arXiv:2508.07629 Table 2

### 2.6 Unified Framework: CISPO and PPO as Special Cases

**FACT** [arXiv:2506.13585, Equations 6-7]:

Both PPO and CISPO are instances of a unified objective with a per-token mask M_{i,t}:

```
J_unified(θ) = (1/T) · Σ_i Σ_t  sg(r̂_{i,t}) · Â_i · log π_θ(o_{i,t} | ...) · M_{i,t}
```

PPO mask:
```
M_{i,t}^PPO = 0   if (Â_i > 0 AND r_{i,t} > 1 + ε_high)
M_{i,t}^PPO = 0   if (Â_i < 0 AND r_{i,t} < 1 - ε_low)
M_{i,t}^PPO = 1   otherwise
```

CISPO mask:
```
M_{i,t}^CISPO = 1   always (no masking)
```

**DERIVED REASONING**: CISPO is PPO with the mask permanently set to 1 and the trust
region enforced via weight clipping instead of gradient masking. This is a bias-variance
tradeoff: CISPO introduces bias (the detached weights don't participate in the gradient
computation) but eliminates the zero-gradient problem that causes PPO to fail on rare tokens.

### 2.7 Variance Analysis: Why ε_high = 5.0 Is Safe

**DERIVED REASONING** (not explicitly in the paper, reconstructed from mechanism):

In PPO, ε controls the trust region size. Setting ε = 5.0 in PPO would allow the policy
to change by 500% per update — catastrophically unstable.

In CISPO, ε_high controls only the **maximum reweighting factor** for the REINFORCE
gradient. Since the weight is detached:

```
Gradient magnitude ∝ sg(r̂) · |Â| · |∇ log π_θ|
```

The maximum amplification factor is bounded by `1 + ε_high`. For ε_high = 5.0, the
maximum amplification is 6×. This means tokens that the current policy finds 6× more
likely than the old policy get at most 6× the gradient of tokens at ratio r=1.

**Why this doesn't cause instability**: The actual policy update size is still controlled by:
1. The learning rate (1e-6 for MiniMax-M1)
2. The Adam optimizer's adaptive scaling
3. Gradient clipping (max_grad_norm)
4. The KL penalty (kl_beta)

The detached weight only reweights **which tokens get more or less gradient** within
a single update step. It does not control the total step size.

**EMPIRICAL SUPPORT** [ScaleRL, arXiv:2510.13786]:
CISPO performance is "almost identical across a wide range of ε_high values." In contrast,
DAPO is "notoriously sensitive" — ε_high of 0.26 vs 0.27 produces asymptotic accuracy
of 0.530 vs 0.480, a catastrophic 10% relative difference.

### 2.8 Adam Epsilon: Why 1e-15 Not 1e-8

**FACT** [arXiv:2506.13585, Section 4.2]:

During RL training of MiniMax-M1 (456B MoE), gradient magnitudes span **1e-18 to 1e-5**.

Adam's update rule:

```
m_t = β_1 · m_{t-1} + (1 - β_1) · g_t                  (first moment)
v_t = β_2 · v_{t-1} + (1 - β_2) · g_t²                 (second moment)
θ_t = θ_{t-1} - lr · m̂_t / (√v̂_t + ε)                 (parameter update)
```

**NUMERIC EXAMPLE** (the failure mode):

Consider a gradient `g = 1e-16` (typical for a rare reasoning token in a deep layer):

```
With ε = 1e-8 (standard):
  √v̂ ≈ 1e-16  (tracking the small gradient)
  denominator = 1e-16 + 1e-8 = 1e-8  (ε dominates!)
  effective_lr = lr / 1e-8             (SAME for all small-gradient params)

With ε = 1e-15:
  √v̂ ≈ 1e-16
  denominator = 1e-16 + 1e-15 = 1.1e-15  (v̂ still contributes!)
  effective_lr = lr / 1.1e-15              (DIFFERENT from larger-gradient params)
```

**MECHANISM**: Standard Adam ε = 1e-8 flattens the adaptive learning rate for all
parameters with gradients below ~1e-8, effectively converting Adam into SGD with
a fixed learning rate for the majority of parameters during RL. Setting ε = 1e-15
preserves Adam's per-parameter adaptivity across the full gradient range.

**Also changed**: β_2 = 0.95 (not 0.999). With the wide gradient range and weak
autocorrelation in RL gradients, a faster-decaying second moment tracks current
gradient magnitude more accurately.

**Source**: arXiv:2506.13585 Section 4.2

### 2.9 FP32 LM Head: The Probability Correlation Fix

**FACT** [arXiv:2506.13585, Section 4.2]:

**Discovery**: MiniMax tracked Pearson correlation between token probabilities in
training mode vs inference mode. Found correlation of ~0.9x (exact value not reported).

**Root cause**: The LM prediction head (the final linear layer projecting from
hidden_dim to vocabulary) operates on high-magnitude activations. In BF16, the
limited mantissa (7 bits) introduces significant quantization error for large logit
values, causing different rounding patterns between training (with gradient computation
overhead) and inference (clean forward pass).

**Fix**: Cast the LM head to FP32. Correlation improved to ~0.99x.

**Why this matters**: The IS ratio `r_t = π_θ(o_t) / π_old(o_t)` requires accurate
probabilities in both numerator and denominator. With BF16 LM head:

```
π_θ^BF16("However") = 0.0028     (true: 0.0031)
π_old^BF16("However") = 0.0033   (true: 0.0030)

r_t^BF16 = 0.0028 / 0.0033 = 0.85   (direction: DECREASE probability)
r_t^FP32 = 0.0031 / 0.0030 = 1.03   (direction: INCREASE probability)
```

A 3-bit error in the mantissa reverses the sign of the IS ratio for rare tokens.
Over thousands of training steps, this noise prevents sustained reward improvement.

**Source**: arXiv:2506.13585 Section 4.2

---

## 3. Algorithm Construction

### 3.1 CISPO Step-by-Step (Single-Turn Reasoning)

**From math → algorithm**: Translating the CISPO objective into an executable procedure.

```
ALGORITHM: CISPO (Single-Turn)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INPUT:
  π_θ: current policy (456B MoE model)
  D: prompt dataset (math, logic, code, SWE)
  G: group size (16 for MiniMax-M1)
  K: gradient steps per generation (16)
  ε_high: IS clip upper bound (5.0)
  lr: learning rate (2e-6)
  β_1, β_2: Adam momenta (0.9, 0.95)
  ε_adam: Adam epsilon (1e-15)

PROCEDURE:
  Initialize θ_old ← θ          # snapshot policy
  Initialize optimizer: AdamW(θ, lr=lr, betas=(β_1, β_2), eps=ε_adam)
  Cast LM head to FP32           # CRITICAL: prevents probability correlation loss

  FOR each training iteration:

    ╔═══ GENERATION PHASE (no gradients) ═══╗
    ║                                        ║
    ║  1. Sample batch of B prompts: {q_1, ..., q_B} from D              ║
    ║                                                                     ║
    ║  2. For each prompt q_b, generate G completions:                   ║
    ║     {o_b,1, ..., o_b,G} ~ π_θ_old(· | q_b)                        ║
    ║     (temperature sampling, max_len = 40K, later 80K)               ║
    ║                                                                     ║
    ║  3. Check for repetition: if 3000 consecutive tokens               ║
    ║     each have π_θ > 0.99, TRUNCATE and assign R=0                  ║
    ║                                                                     ║
    ║  4. Score completions: R_i = reward_fn(q, o_i)                     ║
    ║     - Math: exact match against ground truth                       ║
    ║     - Code: test suite pass/fail                                   ║
    ║     - Logic: programmatic verification                             ║
    ║     - General: GenRM 5-grade scale                                 ║
    ║                                                                     ║
    ║  5. Compute log-probs under old policy:                            ║
    ║     old_logp_{i,t} = log π_θ_old(o_{i,t} | q, o_{i,<t})           ║
    ║                                                                     ║
    ╚════════════════════════════════════════╝

    ╔═══ ADVANTAGE COMPUTATION ═══╗
    ║                              ║
    ║  6. For each prompt group b:                                       ║
    ║     μ_b = mean({R_{b,1}, ..., R_{b,G}})                           ║
    ║     σ_b = std({R_{b,1}, ..., R_{b,G}})                            ║
    ║     Â_{b,i} = (R_{b,i} - μ_b) / (σ_b + 1e-8)                    ║
    ║                                                                     ║
    ╚══════════════════════════════╝

    ╔═══ GRADIENT PHASE (K steps) ═══╗
    ║                                 ║
    ║  FOR k = 1, ..., K:                                                ║
    ║                                                                     ║
    ║    7. Forward pass (with gradients):                               ║
    ║       cur_logp_{i,t} = log π_θ(o_{i,t} | q, o_{i,<t})            ║
    ║                                                                     ║
    ║    8. Compute IS ratios (per-token):                               ║
    ║       log_ratio_{i,t} = cur_logp_{i,t} - old_logp_{i,t}           ║
    ║       r_{i,t} = exp(log_ratio_{i,t})                              ║
    ║                                                                     ║
    ║    9. Clip and detach:                                             ║
    ║       r̂_{i,t} = clamp(r_{i,t}, max=1+ε_high).detach()            ║
    ║                                                                     ║
    ║   10. Compute per-token loss:                                      ║
    ║       loss_{i,t} = -r̂_{i,t} · Â_i · cur_logp_{i,t}              ║
    ║                                                                     ║
    ║   11. Normalize by total tokens:                                   ║
    ║       L = Σ_{i,t} loss_{i,t} / T_total                            ║
    ║       where T_total = Σ_i |o_i|                                    ║
    ║                                                                     ║
    ║   12. Backward + optimize:                                         ║
    ║       L.backward()                                                 ║
    ║       clip_grad_norm_(θ, max_norm)                                 ║
    ║       optimizer.step()                                             ║
    ║       optimizer.zero_grad()                                        ║
    ║                                                                     ║
    ╚═════════════════════════════════╝

    13. Update snapshot: θ_old ← θ

OUTPUT: Fine-tuned policy π_θ with improved reasoning
```

### 3.2 Implementation (PyTorch)

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CISPO Core Loss (production-ready, ~30 lines)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cispo_loss(
    cur_logps: torch.Tensor,       # [B*G, T] current policy log-probs
    old_logps: torch.Tensor,       # [B*G, T] old policy log-probs (detached)
    advantages: torch.Tensor,      # [B*G] group-relative advantages
    completion_mask: torch.Tensor,  # [B*G, T] mask for valid completion tokens
    epsilon_high: float = 5.0,     # IS clip upper bound
) -> torch.Tensor:
    """
    CISPO loss function.

    Key difference from PPO/GRPO: importance sampling weights are
    DETACHED from the computation graph. Gradients flow only through
    cur_logps (the log π_θ term).

    Returns:
        Scalar loss to be minimized.
    """
    # Per-token importance sampling ratio
    log_ratio = cur_logps - old_logps                    # [B*G, T]
    importance_weights = torch.exp(log_ratio)             # [B*G, T]

    # Clip and DETACH — the entire innovation
    clamped = torch.clamp(
        importance_weights,
        min=0.0,                # ε_low effectively disabled
        max=1.0 + epsilon_high  # = 6.0 for ε_high=5.0
    ).detach()                  # ← THIS IS THE KEY LINE

    # Per-token loss: weighted REINFORCE
    # advantages is [B*G], broadcast to [B*G, T]
    per_token_loss = -clamped * advantages.unsqueeze(1) * cur_logps

    # Apply completion mask and normalize by total tokens
    masked_loss = per_token_loss * completion_mask
    total_tokens = completion_mask.sum()

    return masked_loss.sum() / (total_tokens + 1e-8)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# For comparison: PPO/GRPO loss (what CISPO replaces)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def grpo_loss(
    cur_logps: torch.Tensor,       # [B*G, T]
    old_logps: torch.Tensor,       # [B*G, T]
    advantages: torch.Tensor,      # [B*G]
    completion_mask: torch.Tensor,  # [B*G, T]
    epsilon: float = 0.2,
) -> torch.Tensor:
    """Standard PPO/GRPO loss — kills gradients for rare tokens."""
    log_ratio = cur_logps - old_logps
    ratio = torch.exp(log_ratio)  # NOT detached — participates in gradient

    adv = advantages.unsqueeze(1)  # [B*G, 1]

    # Two branches of min
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * adv

    # min selects the lower bound → creates implicit zero-gradient mask
    per_token_loss = -torch.min(surr1, surr2)  # MASKS some tokens

    masked_loss = per_token_loss * completion_mask
    total_tokens = completion_mask.sum()

    return masked_loss.sum() / (total_tokens + 1e-8)
```

### 3.3 Execution Trace (Tiny Numeric Example)

Consider a group of G=2 completions for prompt "What is 2+3?":

```
Completion 1: "Let me think... 2+3=5"     R_1 = 1.0 (correct)
Completion 2: "The answer is 6"            R_2 = 0.0 (wrong)

μ = 0.5, σ = 0.5
Â_1 = (1.0 - 0.5) / 0.5 = +1.0
Â_2 = (0.0 - 0.5) / 0.5 = -1.0
```

For token "think" in Completion 1 at position t=3:

```
π_old("think") = 0.02
π_θ("think")   = 0.05
r_t = 0.05/0.02 = 2.5

PPO (ε=0.2):
  r_t = 2.5 > 1.2, Â > 0 → gradient is ZERO (masked)

CISPO (ε_high=5.0):
  r̂_t = clamp(2.5, max=6.0) = 2.5 (within bounds)
  r̂_t.detach() = 2.5 (constant)
  loss_t = -2.5 × 1.0 × log(0.05) = -2.5 × 1.0 × (-3.0) = 7.5
  gradient = 2.5 × 1.0 × ∇log π_θ("think")  ← NON-ZERO!
```

For token "the" in Completion 2 at position t=1:

```
π_old("the") = 0.15
π_θ("the")   = 0.14
r_t = 0.14/0.15 = 0.933

Both PPO and CISPO:
  Within clip bounds → identical behavior
  loss_t = -0.933 × (-1.0) × log(0.14) = 0.933 × 1.0 × 1.97 = 1.84
  gradient = 0.933 × (-1.0) × ∇log π_θ("the")  (suppress this token)
```

### 3.4 Comparison to Baseline Algorithms

| Algorithm | Gradient formula | Token masking | ε range | Value network |
|-----------|-----------------|---------------|---------|---------------|
| **REINFORCE** | `R · ∇ log π_θ` | None | N/A | No |
| **PPO** | `min(r·A, clip(r)·A) · ∇ log π_θ` | Yes (zero-gradient) | 0.1-0.3 | Yes (standard) |
| **GRPO** | Same as PPO, group-relative advantage | Yes (zero-gradient) | 0.2 | No (group baseline) |
| **DAPO** | PPO + entropy bonus + token-level norm | Yes (some) | 0.2-0.3 | No |
| **CISPO** | `sg(clip(r)) · A · ∇ log π_θ` | **None** (all tokens get gradient) | 0.3-5.0 | No |

**Source**: arXiv:2506.13585 Section 3; ScaleRL arXiv:2510.13786 Section 3

---

## 4. Training Pipeline

### 4.1 End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CISPO TRAINING PIPELINE                       │
│                                                                       │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐                │
│  │  Prompt   │───▶│  Generation   │───▶│   Reward    │                │
│  │  Dataset  │    │  (π_θ_old)   │    │  Scoring    │                │
│  │  ~130K    │    │  G=16/prompt │    │  Rule-based │                │
│  └──────────┘    └──────────────┘    └─────────────┘                │
│       │                │                     │                        │
│       │                ▼                     ▼                        │
│       │         ┌──────────────┐    ┌─────────────┐                 │
│       │         │  Trajectories │    │  Advantages  │                │
│       │         │  [B×G, T]    │    │  [B×G]      │                 │
│       │         └──────────────┘    └─────────────┘                 │
│       │                │                     │                        │
│       │                ▼                     ▼                        │
│       │         ┌────────────────────────────────────┐               │
│       │         │      GRADIENT LOOP (K=16 steps)    │               │
│       │         │                                     │              │
│       │         │  For k = 1..K:                     │               │
│       │         │    cur_logp = forward(π_θ, traj)   │               │
│       │         │    IS_ratio = exp(cur_logp - old_logp)│            │
│       │         │    weight = clamp(IS_ratio).detach()│               │
│       │         │    loss = -weight × adv × cur_logp │               │
│       │         │    loss.backward()                  │               │
│       │         │    clip_grad_norm + optimizer.step  │               │
│       │         └────────────────────────────────────┘               │
│       │                │                                              │
│       │                ▼                                              │
│       │         ┌──────────────┐                                     │
│       │         │  θ_old ← θ   │ (snapshot update)                   │
│       │         └──────────────┘                                     │
│       │                │                                              │
│       ▼                ▼                                              │
│    [REPEAT: next batch of prompts]                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Reward System (Detailed)

**FACT** [arXiv:2506.13585, Section 3.2]:

| Domain | Data scale | Verification method | Pass-rate filter |
|--------|-----------|---------------------|-----------------|
| Math | ~50K curated | Exact match vs ground truth | 0 < pass@10 < 0.9 |
| Logic | ~53K synthesized (41 SynLogic tasks) | Programmatic verification | 0 < pass@10 < 0.9 |
| Competitive programming | ~30K from online judges | LLM-generated test suites | 0 < pass@10 < 0.9 |
| Software engineering | Several thousand | Sandbox test pass/fail | 0 < pass@10 < 0.9 |
| General | ~25K | GenRM (5-grade scale + pairwise) | N/A |

**Pass-rate filtering mechanism**:

For each problem p, compute pass@10 using 10 independent generations from `π_θ_old`:

```
pass@10(p) = 1 - C(n-c, 10) / C(n, 10)    where c = # correct out of n=10

If pass@10 = 0:    Problem too hard → exclude (wasted compute, no gradient signal)
If pass@10 ≥ 0.9:  Problem too easy → exclude (no discriminative power, all Â ≈ 0)
```

**MECHANISM**: Optimal difficulty lies in the "zone of proximal development" where
the model succeeds ~10-90% of the time. Problems here maximize the variance of
advantages, giving the strongest gradient signal.

**GenRM (model-based rewards)**: Five-grade scale with pairwise comparison for open-ended tasks.
Online monitoring detects length-bias reward hacking; triggers immediate recalibration.

**Source**: arXiv:2506.13585, Section 3.2

### 4.3 Curriculum RL

**FACT** [arXiv:2506.13585, Section 3.3]:

Training proceeds in phases with progressive domain mixing:

```
Phase 1 (early): 100% verifiable tasks (math, logic, code)
  - Rule-based rewards only
  - Establishes strong reasoning habits

Phase 2 (mid): 70% verifiable + 30% general domain
  - Introduces model-based rewards (GenRM)
  - Cross-domain generalization begins

Phase 3 (late): 50% verifiable + 50% general domain
  - Full mixed training
  - Prevents catastrophic forgetting of general capabilities
```

**MECHANISM**: Starting with only verifiable rewards prevents the model from gaming
model-based rewards early on (when it's learning to reason). Adding general domain
gradually prevents catastrophic forgetting of conversational abilities.

### 4.4 Repetition Detection

**FACT** [arXiv:2506.13585, Section 3.4]:

```
IF 3000 consecutive tokens each have max_prob > 0.99:
    TRUNCATE generation
    ASSIGN reward = 0
    MARK trajectory as "degenerate"
```

**MECHANISM**: During RL, the policy can enter degenerate repetition loops where it
generates the same token (or short pattern) thousands of times. This wastes compute
during rollouts and provides no useful gradient signal. Early truncation saves compute
and the R=0 penalty discourages the behavior.

### 4.5 Output Length Extension (40K → 80K)

**FACT** [arXiv:2506.13585, Section 3.4]:

The transition from 40K to 80K maximum output length required three stabilization fixes:

1. **More aggressive repetition detection**: The 3000-token threshold was tuned for
   80K sequences where repetition loops are more likely
2. **Combined normalization**: Both sample-level AND token-level normalization used
   together (token-level alone caused instability at 80K)
3. **Reduced gradient clipping + reduced ε_high**: Tighter trust region at longer
   sequences to prevent pattern collapse

**MECHANISM**: At 80K tokens, the IS ratio product across the full sequence can be
astronomically large. Tighter clipping prevents the most extreme reweighting while
combined normalization ensures no single long sequence dominates the gradient.

### 4.6 Pipeline Timing and Bottlenecks

```
┌─────────────────────────────────────────────────────────┐
│ Phase          │ Time fraction │ Bottleneck              │
├────────────────┼───────────────┼─────────────────────────┤
│ Generation     │ ~65-75%       │ Autoregressive decoding │
│ Reward scoring │ ~5-10%        │ Test execution (code)   │
│ Forward pass   │ ~10-15%       │ Memory (456B model)     │
│ Backward pass  │ ~10-15%       │ Memory + communication  │
│ Optimization   │ ~2-5%         │ Negligible              │
└─────────────────────────────────────────────────────────┘
```

**DERIVED REASONING**: Generation dominates because 40K-80K tokens are generated
autoregressively. This is why MTP-based speculative decoding (Section 9.4) provides
the largest throughput improvement. The hybrid lightning attention architecture gives
MiniMax a structural advantage: at 80K tokens, inference is ~4× faster than standard
softmax attention (§2.8 of the summary).

---

## 5. Infrastructure Mapping

### 5.1 Hardware Layout

**FACT** [arXiv:2506.13585, Section 4]:

```
┌─────────────────────────────────────────────────────────────────┐
│                     MiniMax-M1 RL Training                       │
│                                                                   │
│  GPU Cluster: 512 × H800 (80GB HBM3)                            │
│  Duration: 3 weeks                                                │
│  Cost: ~$534,700                                                  │
│                                                                   │
│  ┌──────────────────┐     ┌──────────────────┐                   │
│  │  Generation Pool  │     │  Training Pool   │                   │
│  │  (inference)      │     │  (gradient)      │                   │
│  │                   │     │                   │                   │
│  │  Separated prefill│     │  Standard DP/EP   │                  │
│  │  /decode workers  │     │  parallelism      │                  │
│  │                   │     │                   │                   │
│  │  MTP speculative  │     │  FP32 LM head    │                   │
│  │  decoding         │     │  (critical!)     │                   │
│  │                   │     │                   │                   │
│  │  Global L3 KV     │     │  AdamW optimizer  │                  │
│  │  cache pool       │     │  (ε=1e-15, β2=0.95)│                │
│  └──────────────────┘     └──────────────────┘                   │
│           │                         ▲                              │
│           │    trajectories         │    gradients                 │
│           ▼                         │                              │
│  ┌──────────────────────────────────────────┐                    │
│  │          Data Pool (Middleware)            │                   │
│  │  Windowed FIFO Scheduler (W=0.3N)        │                    │
│  │  Prefix Tree Merger (40× speedup)        │                    │
│  └──────────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Memory Analysis

**DERIVED REASONING** (from model architecture):

```
MiniMax-Text-01 / M1: 456B total params, 45.9B active per token

Per-GPU memory (512 H800s, 80GB HBM each):
  Total cluster memory: 512 × 80GB = 40.96 TB
  Model parameters (BF16): 456B × 2 bytes = 912 GB
  Optimizer states (FP32, AdamW): 456B × 8 bytes = 3.65 TB (m + v + param)
  KV cache per token: depends on hybrid attention structure
  Activation memory: varies with batch size and sequence length

Parallelism strategy:
  Expert Parallelism (EP): distribute 32 experts across EP groups
  Expert Tensor Parallel (ETP): decouple MoE from non-MoE parallelism
  EP-ETP overlap: 50% communication reduction for MoE layers
  Separated prefill/decode: independent parallelism per phase
```

### 5.3 Throughput vs Latency

**FACT** [arXiv:2501.08313, Section 5; arXiv:2506.13585, Section 4]:

```
Inference speed: 100 TPS (with lightning attention variant)
Standard speed: ~50 TPS

At 80K output generation:
  Time per completion: 80,000 / 100 = 800 seconds ≈ 13.3 minutes
  Group of G=16: could be parallelized across inference workers

Per training iteration (B prompts, G=16 per prompt):
  Generation: B × 16 × 800s / (# inference workers)
  Forward pass: ~T × (model size / throughput)
  Backward pass: ~2-3× forward pass

Efficiency at long context:
  MiniMax at 100K tokens: ~25% FLOPs of DeepSeek-R1
  Reason: Lightning attention O(n) for 87.5% of layers vs O(n²)
  First-token latency: 4-5s (vs ~60s for standard attention on 100K input)
```

### 5.4 Cost Breakdown

**FACT** [arXiv:2506.13585]:

```
512 H800 GPUs × 3 weeks × 24 hours = 258,048 GPU-hours
Cost: ~$534,700
Per-GPU-hour cost: ~$2.07 (likely reserved/bulk pricing)

Breakdown (estimated):
  Generation: ~65% × $534K = ~$347K
  Training: ~30% × $534K = ~$160K
  Overhead: ~5% × $534K = ~$27K
```

---

## 6. Empirical Results

### 6.1 CISPO vs Baselines

**FACT** [arXiv:2506.13585, Figure 6 — AIME 2024, Qwen2.5-32B-base]:

| Algorithm | Accuracy at 10K steps | Steps to reach 72% |
|-----------|----------------------|---------------------|
| **CISPO** | ~76% | ~5,000 |
| **DAPO** | ~72% | ~10,000 |
| **GRPO** | ~68% | >10,000 |

**MECHANISM**: CISPO's 2× convergence advantage comes from preserving gradients for
rare reasoning tokens. These tokens are exactly what drive correct mathematical
reasoning — "Let me verify," "Actually, wait," "I need to reconsider."

### 6.2 ScaleRL Validation

**FACT** [ScaleRL, arXiv:2510.13786]:

- CISPO "substantially outperforms DAPO" in asymptotic pass-rate
- Compute efficiency: CISPO B=2.01 vs DAPO B=1.77 (higher is better)
- CISPO shows "prolonged near-linear reward increase" (no plateau)
- **Robustness**: CISPO performance is "almost identical across a wide range of ε_high"
- DAPO is "notoriously sensitive" to ε_high (0.26 vs 0.27 → 10% accuracy difference)
- CISPO is "marginally better than GSPO later in training" but more stable

### 6.3 MiniMax-M1 Final Performance

**FACT** [arXiv:2506.13585, Table 1]:

| Benchmark | MiniMax-M1 | DeepSeek-R1 | o3-mini-low |
|-----------|-----------|-------------|-------------|
| AIME 2024 | 86.7% | 79.8% | 83.6% |
| MATH-500 | 97.4% | 97.3% | 97.0% |
| LiveCodeBench v5 | 62.3% | 65.9% | 53.8% |
| SWE-bench Verified | 56.0% | 49.2% | — |

**Efficiency**: At 100K token generation, M1 consumes **~25% of FLOPs** vs R1
due to near-linear attention scaling.

### 6.4 Claim → Evidence Mapping

| Claim | Evidence | Strength | Notes |
|-------|----------|----------|-------|
| "CISPO preserves rare token gradients" | Theoretical derivation (§2.5) + gradient coefficient table | **Strong** | Mathematical proof, no empirical measurement of per-token gradients shown |
| "2× convergence speedup vs DAPO" | Figure 6 of M1 paper on AIME 2024 (Qwen2.5-32B) | **Moderate** | Single benchmark, single base model. ScaleRL confirms on different setup |
| "ε_high robust across wide range" | ScaleRL ablation | **Strong** | Multiple values tested, compared to DAPO sensitivity |
| "Adam ε=1e-15 critical" | Described in Section 4.2, no ablation shown | **Weak** | Only claim, no ablation data comparing 1e-8 vs 1e-15 |
| "FP32 LM head improves correlation to 0.99x" | Described in Section 4.2, metric named | **Moderate** | Correlation metric tracked but exact numbers not reported |
| "Repetition detection saves compute" | Described, no quantitative savings reported | **Weak** | Reasonable mechanism but no data on frequency |
| "Curriculum prevents forgetting" | Stated, no ablation | **Weak** | Standard practice, plausible but unverified for CISPO specifically |

### 6.5 Reproducibility Assessment

| Component | Reproducible? | Blocking factor |
|-----------|---------------|-----------------|
| CISPO loss function | **Yes** | ~15 lines of code, well-specified |
| Adam ε=1e-15 | **Yes** | 1-line config change |
| FP32 LM head | **Yes** | 1-line cast |
| Pass-rate filtering | **Yes** | Standard pass@k computation |
| GenRM | **No** | Training data/architecture not disclosed |
| Repetition detection | **Mostly** | Threshold (3000 tokens, p>0.99) specified |
| Curriculum schedule | **Partially** | Ratios specified but transition criteria not |
| Full training recipe | **No** | Hyperparameters partially specified, data mix unclear |

**Source**: Assessment based on information availability in arXiv:2506.13585

---

## 7. Critical Analysis

### 7.1 Hidden Assumptions

1. **Token-level advantage assignment**: CISPO assigns the same advantage Â_i to ALL
   tokens in a completion. This is a strong assumption — "However" at position 100
   and "the" at position 50,000 get the same advantage signal. The IS weight
   differentiates them somewhat, but the fundamental credit assignment is coarse.

2. **Group-relative baseline**: The group baseline assumes G samples adequately
   represent the reward distribution for each prompt. With G=16 and binary rewards,
   the baseline is noisy. For a prompt with pass@16 = 1/16 = 6.25%, only 1 completion
   is correct; the advantage for that completion is `(1 - 1/16) / std ≈ +3.5σ`,
   while the 15 wrong completions each get `(0 - 1/16) / std ≈ -0.2σ`. The gradient
   is dominated by the single correct completion.

3. **IS weight stability**: With ε_high = 5.0, the maximum ratio is 6.0. After K=16
   gradient steps, the cumulative drift from the old policy can be much larger. The
   per-step IS ratios are computed against `θ_old` (frozen), not against the previous
   gradient step. By step K=16, `π_θ` may be far from `π_θ_old`, making the IS
   correction increasingly inaccurate.

4. **FP32 LM head is sufficient**: The claim that casting only the LM head to FP32
   fixes the probability correlation issue assumes the problem is localized to the
   final projection. If numerical errors accumulate through the transformer backbone,
   the fix would be incomplete.

### 7.2 Failure Modes

1. **Reward hacking through length**: If longer responses correlate with higher
   scores (common in GenRM), the policy will learn to produce verbose outputs.
   MiniMax monitors for this and recalibrates, but the detection mechanism is reactive.

2. **Expert routing collapse during RL**: RL policy updates change the distribution
   of hidden states, which can shift expert routing patterns. If several experts
   become underutilized, the model loses capacity. MiniMax does NOT freeze routers
   (unlike DeepSeek V3.2's "Keep Routing" technique 3). The interaction between
   CISPO's IS weight reweighting and MoE routing stability is uncharacterized.

3. **Degenerate repetition not fully prevented**: The 3000-token threshold is
   arbitrary. Subtler repetition patterns (e.g., 50-token loops repeated 60 times)
   would not trigger detection but still waste compute.

4. **Catastrophic forgetting of general capabilities**: Curriculum mixing mitigates
   but may not prevent loss of specific capabilities not represented in the training
   mix. No capability-specific evaluation during training is disclosed.

5. **Distribution shift at long context**: At 80K output tokens, the IS ratio
   product across the full sequence can be very large even with per-token clipping.
   The paper acknowledges requiring "decreased gradient clipping" and "reduced ε_high"
   for stability — suggesting the basic CISPO formulation needed adaptation.

### 7.3 What Distribution Shift Kills This

**HYPOTHESIS**: CISPO's IS correction assumes that `π_θ_old` and `π_θ` are "close enough"
that the IS ratios are informative. This breaks when:

1. **Rapid policy shift** (many gradient steps on same batch): By K=16, the policy
   may have moved significantly from `θ_old`. The IS ratios computed against `θ_old`
   become stale. This is the standard on-policy drift problem.

2. **Reward distribution shift**: If the prompt difficulty distribution changes (e.g.,
   curriculum transitions), the advantage baseline `μ_G` may be poorly calibrated for
   the new distribution.

3. **Long-horizon generation**: At 80K tokens, even small per-token distribution shifts
   compound multiplicatively. A token-level IS ratio of 1.1 per token × 80K tokens
   gives a sequence-level ratio of `1.1^{80000} ≈ ∞`. CISPO handles this via
   per-token (not sequence-level) IS ratios, but the compounding effect still makes
   the gradient estimate noisier at longer contexts.

### 7.4 Comparison to Alternatives

| Alternative | Advantage over CISPO | Disadvantage vs CISPO |
|-------------|---------------------|----------------------|
| **Kimi's squared-loss MD** | Principled derivation from optimization theory; implicit KL regularization via squared penalty | May over-penalize large policy shifts; less robust to ε tuning (not tested) |
| **GPPO** (Klear-Reasoner) | Claims superior AIME2024 (90.5% vs ~76% at same scale); PPO-derived (principled trust region) | More complex implementation; not validated at MiniMax's scale |
| **DAPO** | Better entropy preservation via bonus term | 2× slower convergence; extremely sensitive to ε_high |
| **Pure REINFORCE** | Simplest, unbiased | Catastrophically high variance at 40K+ tokens |
| **PPO with value network** | Better credit assignment (per-token advantages) | Value network penalizes exploratory reasoning; 2× memory |

### 7.5 What Is Missing from the Paper

1. **Per-token gradient analysis**: No empirical measurement showing that "rare reasoning
   tokens" actually receive zero gradient under PPO and non-zero under CISPO. The claim
   is theoretical.

2. **Ablation of ε_high**: The MiniMax paper does not show an ablation of ε_high values.
   ScaleRL provides this, but on a different model/scale.

3. **Ablation of Adam ε**: No comparison of 1e-8 vs 1e-15 is shown.

4. **FP32 LM head ablation**: No before/after reward curves with and without the fix.

5. **MoE routing dynamics during RL**: No H_load or I_spec plots during CISPO training.
   We don't know if expert routing remained stable.

6. **Token-level analysis of IS weights**: No histogram of IS weights showing the
   distribution during training. We don't know how many tokens are near the clip boundary.

---

## 8. Multi-Turn Agentic CISPO

### 8.1 The Multi-Turn Challenge

**FACT** [MiniMax M2.1 blog; Forge blog]:

Agent trajectories differ from single-turn reasoning in critical ways:

```
Single-turn:  [prompt] → [completion] → reward
  - 1 LLM call per trajectory
  - All tokens from same generation
  - IS ratio well-defined

Multi-turn:   [prompt] → [think] → [tool_call] → [tool_result] → [think] → ... → reward
  - N LLM calls per trajectory (N varies per trajectory!)
  - Interleaved LLM tokens and environment tokens
  - IS ratio only defined for LLM-generated tokens
  - Trajectories have different lengths and structures
```

### 8.2 Multiple Importance Sampling (MIS)

**FACT** [MiniMax M2.1 blog — "several major techniques"]:

Standard CISPO normalizes by total tokens: `T_total = Σ_i |o_i|`. For multi-turn
trajectories with different numbers of LLM calls, this creates a bias:

```
Trajectory A: 3 LLM calls, 5000 tokens total → contributes 5000 gradient terms
Trajectory B: 1 LLM call, 2000 tokens total → contributes 2000 gradient terms

Without MIS: Trajectory A dominates the gradient (2.5× more terms)
```

**DERIVED REASONING**: MIS normalizes IS weights across trajectories with different
numbers of generation steps. A plausible formulation (exact formula not published):

```
Turn-level IS weight:
  w_k = exp((1/|y_k|) · Σ_t log(π_θ(y_{k,t}) / π_old(y_{k,t})))

Trajectory-level IS weight:
  W_traj = Π_k w_k  (product of turn-level weights)

Normalized:
  W̃_traj = W_traj / Σ_j W_traj_j  (self-normalized IS)
```

This prevents long trajectories (many turns) from having systematically larger IS
weights than short trajectories.

**OPEN QUESTION**: The exact MIS formula is not published. The above is a plausible
reconstruction based on standard multi-importance-sampling theory and the SORL paper
(arXiv:2511.20718).

### 8.3 Process Rewards (Dense Feedback)

**FACT** [Forge blog]:

```
R_total = r_task + r_process + r_speed + r_format

r_task:     Binary pass/fail from test execution (sparse, end of trajectory)
r_process:  Dense penalties for intermediate behaviors:
            - Language mixing (e.g., Chinese in English context): penalty
            - Tool invocation errors (malformed API calls): penalty
            - Format violations (missing required fields): penalty
r_speed:    Completion time relative to baseline:
            - Rewards finishing faster than baseline
            - Accounts for parallelism utilization
r_format:   Structural compliance (proper tool calling format)
```

**Reward-to-go formulation** (reduces variance at long horizon):

```
Â_{i,t} = Σ_{p=t}^T (r_p^speed + r_p^perf) - B_i

where B_i is a baseline for variance reduction
```

**Claimed impact**: ~45% reduction in context reasoning drift, ~60% reduction in
gradient variance at 200K token context.

### 8.4 Context Rot and the Context Management Action

**FACT** [MiniMax M2.1 blog]:

**Problem**: In multi-turn agent interactions, the context grows monotonically with
each tool call and response. By turn 20+, the model attends to massive amounts of
irrelevant historical context, reducing the effective signal-to-noise ratio of
attention ("attention dilution" or "context rot").

**Solution**: Context management is made an **explicit RL action**. The agent can:
- Summarize previous context into a compact representation
- Discard irrelevant tool call results
- Compress intermediate reasoning into conclusions

This is learned through RL — the agent discovers that context management actions
lead to better task completion rates, so they are reinforced.

**MECHANISM**: Without this, there's a training-inference mismatch. During training,
full context is available. During inference with context limits, the model must
manage context but was never trained to do so. By making context management an
explicit action during training, the model learns this skill.

---

## 9. Forge Framework

### 9.1 Architecture

**FACT** [Forge blog, official]:

```
┌─────────────────────┐  ┌────────────────────────────┐  ┌────────────────────┐
│    AGENT SIDE        │  │    MIDDLEWARE               │  │  TRAINING/INFERENCE │
│                      │  │                             │  │                     │
│  agent_reprocess()   │  │  Gateway Server             │  │  LLM Engine         │
│  agent_run()         │──▶  (standardized comm)        │──▶  (token generation) │
│  agent_postprocess() │  │                             │  │                     │
│  calculate_reward()  │◀──  Data Pool                  │◀──  Train Engine       │
│                      │  │  (distributed async storage)│  │  (CISPO updates)    │
│                      │  │                             │  │                     │
│  100K+ scaffolds     │  │  Windowed FIFO Scheduler    │  │  FP32 LM head      │
│  White-box + black-box│ │  Prefix Tree Merger         │  │  AdamW (ε=1e-15)   │
└─────────────────────┘  └────────────────────────────┘  └────────────────────┘
```

### 9.2 Windowed FIFO Scheduling (Detailed)

**FACT** [Forge blog]:

**Problem formalization**:
- N agent rollouts queued for RL training
- Completion times: t_0, t_1, ..., t_{N-1} (random, heavy-tailed distribution)
- Order of submission: 0, 1, ..., N-1

**Three scheduling strategies and their failure modes**:

```
Strict FIFO (sync):
  Process in order: 0, 1, 2, ...
  If task 0 takes 2 hours while tasks 1-99 take 5 minutes each:
    → 99 completed tasks idle for 2 hours
    → GPU utilization: ~1%
  FAILURE: Head-of-Line blocking ("Straggler Effect")

Greedy (async):
  Process whatever completes first
  If easy tasks complete faster than hard tasks:
    → Training initially sees only easy tasks
    → Then sees clustered hard tasks
    → Non-stationary training distribution
  FAILURE: Data distribution shift → gradient oscillation

Windowed FIFO (Forge):
  Window size W = 0.3N (30% of queue)
  Active window: [task_i, task_{i+W-1}]

  Rules:
  1. WITHIN window: consume any completed task (greedy within window)
  2. AT BOUNDARY: STRICT BLOCK — cannot fetch task_{i+W} even if complete
  3. SLIDE: window advances only as head task consumed
```

**NUMERIC EXAMPLE**:

```
N = 10 tasks, W = 3 (window = 30%)
Completion times: t0=60s, t1=5s, t2=8s, t3=3s, t4=120s, t5=4s, ...

Initial window: [T0, T1, T2]
  T1 completes first (5s) → consume T1
  T2 completes (8s) → consume T2
  T0 completes (60s) → consume T0, window slides

New window: [T3, T4, T5]
  T3 completes (3s) → consume T3
  T5 completes (4s) → consume T5
  T4 (120s) → BLOCKS window progression until T4 completes
  BUT T6, T7, T8 (if complete) cannot be consumed (outside window)

This ensures hard tasks within the window ARE eventually consumed,
preventing distribution shift, while allowing local reordering
for throughput.
```

**MECHANISM**: Windowed FIFO bounds the maximum staleness to window size W.
The distribution of tasks seen by the trainer is within W positions of the
true FIFO order. This is a formal tradeoff between sync (W=1) and fully
async (W=N).

### 9.3 Prefix Tree Merging (40× Speedup)

**FACT** [Forge blog]:

**Problem**: Multi-turn agent trajectories share extensive common prefixes:

```
Group of G=16 rollouts for same prompt:
  All share: system_prompt (2K tokens) + user_query (500 tokens) + tool_defs (3K tokens)
  First turns may diverge at different points

Naive approach: 16 separate forward passes, each processing the 5.5K common prefix
Wasted compute: 15 × 5.5K × (FLOPs per token) = 82.5K redundant token computations
```

**Solution**: Merge into a prefix tree:

```
                      [system_prompt + user_query + tool_defs]  ← computed ONCE
                                        |
                    ┌───────────────────┼───────────────────┐
                    |                   |                     |
            [tool_call_A]        [tool_call_B]          [tool_call_C]
                |                   |                     |
            ┌───┴───┐          ┌───┴───┐              [resp_C1]
        [resp_A1] [resp_A2]  [resp_B1] [resp_B2]
```

**Implementation using Magi Attention**:
- Causal attention masks ensure each path only attends to its own prefix
- The tree structure is "flattened" into a single batch with heterogeneous masks
- Magi Attention kernels handle arbitrary overlapping mask patterns efficiently
- Post-forward: tree deconstructed, loss computed per-trajectory normally

**Mathematical equivalence**: Guaranteed by causal masking — each token sees exactly
the same context as it would in an independent forward pass. The only difference is
computational (shared prefix computation), not mathematical.

**Performance**: ~40× speedup in multi-turn RL training. Also reduces memory by
avoiding redundant KV cache storage for shared prefixes.

### 9.4 MTP-Based Speculative Decoding During RL

**FACT** [Forge blog; arXiv:2501.08313]:

**Problem**: Standard speculative decoding uses a static draft model. During RL,
the policy changes continuously, causing the draft model's distribution to drift
from the target. Acceptance rates degrade, negating the speedup.

**Solution**: Use the model's own MTP (Multi-Token Prediction) heads as the draft.
These heads are already part of the architecture and predict future tokens.

```
Main model: predicts token t
MTP head 1: predicts token t+1 (given hidden state at t)
MTP head 2: predicts token t+2 (given hidden state at t)

Speculative decoding:
1. Main model generates token t
2. MTP heads speculatively predict t+1, t+2
3. Main model verifies in parallel
4. Accept correct speculative tokens (skip autoregressive steps)
```

**Top-K KL loss** (keeps MTP heads aligned with evolving RL policy):

```
L_MTP = KL_topK(p_MTP || p_main)
```

The MTP heads are fine-tuned alongside the main model during RL training.
This prevents acceptance rate degradation that plagues static draft models.

**OPEN QUESTION**: The exact Top-K KL formula is not published. "Top-K" likely
means computing KL only over the top-K most probable tokens (to avoid the long
tail where both distributions are near-zero).

### 9.5 Global L3 KV Cache Pool

**FACT** [Forge blog]:

```
Cache hierarchy:
  L1: Local HBM (active tokens, on-GPU)
  L2: Local DRAM/NVMe (recently used, host memory)
  L3: Global distributed store (across all workers)

Organization: DFS (depth-first search) tree matching conversation structure
  - Same prompts generate tree of conversations
  - KV cache for shared prefixes stored once in L3
  - Cost-aware scheduler routes requests to maximize cache hits

Scheduling optimization:
  For each inference request, score = α × queue_delay + β × cache_migration_cost
  Route to worker minimizing score (balance load vs cache locality)
```

### 9.6 Scale Numbers

**FACT** [Forge blog; M2.5 blog]:

| Metric | Value |
|--------|-------|
| Daily throughput | Millions of samples |
| Max context length | 200K tokens |
| Distinct scaffolds trained | >100,000 |
| Concurrent environments | >200,000 (M2.5) |
| Prefix tree speedup | ~40× |
| Context drift reduction | ~45% |
| Gradient variance reduction (200K) | ~60% |
| SWE-bench token usage | 3.52M tokens/task average |

---

## 10. Self-Evolving Training Loop

### 10.1 The Innovation

**FACT** [MiniMax M2.7 blog, March 2026]:

M2.7 is the first model that "deeply participates in its own evolution."

```
┌────────────────────────────────────────────────────────────┐
│              SELF-EVOLVING TRAINING LOOP                    │
│                                                              │
│  ┌─────────────────┐                                        │
│  │ 1. ANALYZE       │  Read failure trajectories from last  │
│  │    FAILURES      │  RL training run. Identify patterns.  │
│  └────────┬────────┘                                        │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │ 2. PLAN          │  Propose specific changes to          │
│  │    CHANGES       │  scaffold code, prompts, or params.   │
│  └────────┬────────┘                                        │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │ 3. MODIFY        │  Edit own agent harness code.         │
│  │    SCAFFOLD      │  (The model writes code that          │
│  └────────┬────────┘   changes its own behavior.)           │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │ 4. EVALUATE      │  Run evaluations on internal          │
│  │                  │  benchmarks with new scaffold.         │
│  └────────┬────────┘                                        │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │ 5. COMPARE       │  Metric analysis: new vs old.         │
│  │    RESULTS       │                                        │
│  └────────┬────────┘                                        │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │ 6. DECIDE        │  If improved: KEEP changes.           │
│  │    KEEP/REVERT   │  If not: REVERT to previous.          │
│  └────────┬────────┘                                        │
│           │                                                  │
│           └──────────▶ REPEAT (100+ rounds autonomously)    │
└────────────────────────────────────────────────────────────┘
```

### 10.2 What the Model Discovered (Autonomously)

**FACT** [M2.7 blog]:

| Discovery | Description | Category |
|-----------|-------------|----------|
| Systematic parameter search | Temperature, frequency penalty, presence penalty tuning | Hyperparameter |
| Workflow guideline refinement | Auto-searching for same bug patterns in other files | Strategy |
| Loop detection and prevention | Identifying and breaking infinite agent loops | Robustness |
| Context management improvements | Better context summarization strategies | Efficiency |
| Failure pattern recognition | Categorizing failure modes and pre-empting them | Generalization |

### 10.3 Scope of Automation

**FACT** [M2.7 blog]:

M2.7 handles **30-50% of the RL research workflow** autonomously:

```
Automated:
  ✓ Log reading & debugging
  ✓ Metric analysis & visualization
  ✓ Code fixes & merge requests
  ✓ Smoke testing
  ✓ Synthetic data generation
  ✓ Training environment optimization
  ✓ Literature review assistance
  ✓ Experiment specification tracking
  ✓ Data pipeline management

NOT automated:
  ✗ Algorithm design (CISPO, reward shaping)
  ✗ Architecture decisions (MoE topology, attention design)
  ✗ Data curation strategy
  ✗ Safety alignment design
  ✗ Scaling decisions
```

### 10.4 Harness Skills

**FACT** [M2.7 blog]:

The model builds **reusable instruction sets** (~2,000 tokens each):
- 40+ complex skills constructed autonomously
- 97% compliance rate
- Persistent memory store across iterations
- Skills include: debugging strategies, testing protocols, optimization recipes

### 10.5 Critical Assessment

**DERIVED REASONING**:

The self-evolving loop modifies the **scaffold** (the agent harness code that
orchestrates tool use), NOT the model weights. The model itself is updated
through normal CISPO RL. The innovation is that the model acts as an RL researcher
— it reads training logs, identifies problems, and modifies its own infrastructure.

**Risk**: Compounding errors. If the model makes a bad scaffold change that
produces higher rewards on the evaluation set but degrades robustness, it will
keep the change. Without human review, these errors can accumulate.

**Mitigation**: The keep/revert decision uses metric-based comparison, not the
model's subjective judgment. If metrics degrade, the change is reverted.

---

## 11. NanoSeek Implementation Guide

### 11.1 CISPO for NanoSeek's Existing GRPO Trainer

The existing `grpo_trainer.py` uses PPO-style clipping (lines 387-389). Converting
to CISPO requires modifying the `train_step` method:

**Changes required**:

```python
# CURRENT (grpo_trainer.py, lines 387-392):
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages
policy_loss = -torch.min(surr1, surr2)
policy_loss = (policy_loss * on_policy_mask.float()).sum() / n_on_policy

# CISPO REPLACEMENT:
clamped_weights = torch.clamp(ratio, max=1.0 + cfg.epsilon_high).detach()
policy_loss = -(clamped_weights * advantages * cur_seq_lp)
policy_loss = policy_loss.sum() / (B * G)  # token-level normalization preferred
```

**Additional changes**:
1. **GRPOConfig**: Add `epsilon_high: float = 5.0`, change optimizer eps to `1e-15`
2. **Optimizer**: Change `eps=1e-15` in AdamW initialization (line 238)
3. **LM head**: Add `model.lm_head.float()` before RL training (cast to FP32)
4. **Token-level computation**: Current implementation sums log-probs to sequence level
   (lines 362-364). CISPO works better at token level. Convert to per-token IS ratios.

**Estimated effort**: ~2 hours for full conversion. The `.detach()` is 1 line.
The FP32 LM head is 1 line. The Adam epsilon is 1 line. The structural change
from sequence-level to token-level IS ratios is the main work.

### 11.2 Critical NanoSeek-Specific Considerations

1. **MoE routing during RL**: NanoSeek's GRPO already freezes routers (technique 3).
   CISPO may or may not need this — MiniMax doesn't freeze routers. Test both.

2. **Scale**: At 1.08B active (vs MiniMax's 45.9B), gradient magnitudes will be
   different. The Adam ε = 1e-15 may need adjustment — profile gradient magnitudes
   first.

3. **I_spec monitoring**: Track expert specialization (I_spec) during CISPO training.
   If CISPO preserves routing better than GRPO, this is a novel finding about
   MoE × RL interaction.

4. **FP32 LM head**: NanoSeek's model is much smaller. BF16 precision may be
   sufficient at 1B scale. Profile probability correlation before/after to verify.

### 11.3 Ablation Design for NanoSeek Phase 5

```
Ablation A: GRPO (current, baseline)
Ablation B: CISPO (ε_high=5.0, ε_adam=1e-15, FP32 LM head)
Ablation C: CISPO without FP32 LM head (test if needed at 1B scale)
Ablation D: CISPO with router freeze vs without (test interaction)
Ablation E: Kimi squared-loss MD (for comparison)

Metrics per ablation:
  - Reward curve (GSM8K accuracy over training steps)
  - Convergence speed (steps to 50% of max reward)
  - I_spec stability (expert specialization before/during/after RL)
  - H_load stability (load balance before/during/after RL)
  - MTP acceptance rate (before/after RL)
  - Per-token gradient magnitude histogram (verify rare token preservation)
```

---

## 12. Open Research Questions

### 12.1 Fundamental

1. **Does CISPO actually preserve rare token gradients in practice?** The theoretical
   argument is sound, but no empirical per-token gradient analysis has been published.
   NanoSeek could measure this by logging gradient magnitudes for tokens categorized
   by frequency.

2. **What is the optimal ε_high as a function of model scale?** ScaleRL tested on
   specific models. Does the optimal value change at 1B vs 7B vs 70B vs 456B?

3. **Is the FP32 LM head needed at small scale (<10B)?** The probability correlation
   issue may be scale-dependent (larger logit magnitudes at larger scale).

4. **How does CISPO interact with MoE routing stability?** Neither MiniMax nor any
   follow-up paper has characterized the effect of CISPO's IS weight reweighting on
   expert routing patterns. This is directly measurable in NanoSeek.

### 12.2 Algorithmic

5. **CISPO vs squared-loss MD (Kimi) at matched compute**: No head-to-head comparison
   exists. Implementing both in NanoSeek would be the first controlled comparison.

6. **Token-level vs sequence-level advantages**: CISPO uses sequence-level advantages
   (same Â for all tokens). Would process rewards (token-level advantages) help?

7. **Adaptive ε_high during training**: Should ε_high decrease as the policy improves
   (tighter trust region later)? MiniMax reduces it for 80K extension but not
   systematically.

### 12.3 Infrastructure

8. **Prefix tree merging for NanoSeek**: At NanoSeek's scale, is the overhead of
   tree construction worth the speedup? Probably not for single-turn, but yes for
   multi-turn agent RL.

9. **Windowed FIFO scheduling overhead**: At small scale (few GPUs), is the scheduling
   complexity justified? Only matters for async multi-GPU RL.

### 12.4 Self-Evolving

10. **Can self-evolving training work at small scale?** MiniMax uses it at 230B.
    At 1B, the model may not be capable enough to analyze its own training logs
    and propose meaningful changes. This is an open frontier question.

11. **Safety of self-modifying systems**: What prevents the self-evolving loop from
    discovering reward hacking strategies that look good on metrics but degrade
    real-world performance? The keep/revert mechanism only checks metrics, not
    safety properties.

---

## Appendix A: Complete Source List

### Primary Sources (Peer-Reviewed / Official)

| Source | Content | Citation |
|--------|---------|----------|
| MiniMax-01 | Foundation model architecture, lightning attention, MoE | arXiv:2501.08313 |
| MiniMax-M1 | CISPO algorithm, RL training, curriculum, repetition detection | arXiv:2506.13585 |
| ScaleRL | CISPO ablations, ε_high robustness, convergence comparison | arXiv:2510.13786 |
| Klear-Reasoner | CISPO gradient analysis, GPPO comparison | arXiv:2508.07629 |
| ASPO | Asymmetric IS, unified framework | arXiv:2510.06062 |
| M2.7 blog | Self-evolving training, agent teams | minimax.io |
| Forge blog | Windowed FIFO, prefix tree, MIS, process rewards | minimax.io |
| M2.1 blog | Multi-scaffold training, context rot, agentic CISPO | minimax.io |
| M2.5 blog | Production benchmarks, scale numbers | minimax.io |
| CISPO docs | Implementation pseudocode, hyperparameters | swift.readthedocs.io |
| Lightning Attn-2 | Linear attention algorithm | arXiv:2401.04658 |

### Secondary Sources (Analysis / Commentary)

| Source | Content |
|--------|---------|
| RLHF Book (Nathan Lambert) | Policy gradient context for CISPO |
| SORL paper (arXiv:2511.20718) | Turn-level IS weights for multi-turn RL |
| TRL documentation (HuggingFace) | CISPO implementation in trl library |
| EmergentMind topic page | CISPO paper aggregation |
| HuggingFace MiniMax-01 deep dive | Architecture analysis |

---

## Appendix B: Notation Reference

| Symbol | Meaning |
|--------|---------|
| θ | Policy model parameters |
| π_θ(o_t \| q, o_{<t}) | Token-level policy (LLM output distribution) |
| π_{θ_old} | Snapshot policy (frozen for IS ratio computation) |
| π_ref | Reference policy (frozen for KL computation) |
| r_{i,t}(θ) | Importance sampling ratio for token t of completion i |
| r̂_{i,t}(θ) | Clipped IS ratio |
| sg(·) | Stop-gradient / .detach() operator |
| Â_i | Group-relative advantage for completion i |
| R_i | Reward for completion i |
| G | Group size (number of completions per prompt) |
| K | Number of gradient steps per generation |
| T | Total tokens across all completions |
| ε_high | IS clip upper bound (5.0 for MiniMax-M1) |
| ε_low | IS clip lower bound (effectively disabled) |
| ε_adam | Adam optimizer epsilon (1e-15) |
| β_1, β_2 | Adam momentum parameters (0.9, 0.95) |
| D | Prompt dataset |
| B | Batch size (number of prompts) |
| V | Vocabulary size (200,064 for MiniMax) |
| H_load | Expert load-balance entropy |
| I_spec | Expert specialization mutual information |
