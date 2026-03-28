# Rigorous Mathematical Analysis of RL Algorithms for LLM Post-Training
## Agent 4: RL Math Scientist — First-Principles Derivations and Recommendations
### Compiled: March 2026

---

# A. Policy Optimization Methods — Mathematical Analysis

## Notation and Setup

Throughout this document, we use the following notation:

- `pi_theta(y|x)` = policy parameterized by theta, generating response y to prompt x
- `pi_ref(y|x)` = reference policy (frozen or moving)
- `pi_old(y|x)` = behavior policy that generated the rollout
- `r(x, y)` = reward for response y to prompt x
- `G` = group size (number of completions per prompt)
- `|y_i|` = number of tokens in completion i
- `r_t(theta) = pi_theta(y_t|x, y_{<t}) / pi_old(y_t|x, y_{<t})` = per-token importance sampling ratio
- `A_i` = advantage estimate for completion i
- `epsilon` = clipping parameter

---

## A.1 PPO (Proximal Policy Optimization) — Baseline

### Objective Function

```
L_PPO(theta) = E_t[ min( r_t(theta) * A_t^GAE, clip(r_t(theta), 1-eps, 1+eps) * A_t^GAE ) ]
             - beta * D_KL(pi_theta || pi_ref)
             + c_1 * L_VF(phi)    # value function loss
```

Where the GAE advantage is:
```
A_t^GAE = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}
delta_t = r_t + gamma * V_phi(s_{t+1}) - V_phi(s_t)
```

### Why This Objective

PPO solves the problem of TRPO's computational cost. TRPO requires computing the Fisher information matrix for the natural gradient step. PPO replaces this with a first-order surrogate:

The clipped objective creates a pessimistic lower bound on the true objective. When the ratio r_t deviates beyond [1-eps, 1+eps], the gradient is zeroed, preventing the policy from moving too far in a single update.

**Derivation**: The trust region constraint `D_KL(pi_theta || pi_old) <= delta` is relaxed into a clipping penalty. The min() operator selects the more conservative of the clipped and unclipped objectives.

### Gradient Expression

```
nabla_theta L_PPO = E_t[
  A_t^GAE * nabla_theta log pi_theta(a_t|s_t)
    * I(r_t in [1-eps, 1+eps] OR (r_t > 1+eps AND A_t < 0) OR (r_t < 1-eps AND A_t > 0))
]
```

The indicator function shows that gradients are ZERO when:
- r_t > 1+eps AND A_t > 0 (policy already moved too much in the good direction)
- r_t < 1-eps AND A_t < 0 (policy already moved too much away from bad actions)

### Variance Analysis

The GAE advantage estimator has variance:
```
Var(A_t^GAE) = sum_{l=0}^{T-t} (gamma*lambda)^{2l} * Var(delta_{t+l})
             + cross terms from V_phi estimation error
```

The value network V_phi reduces variance compared to Monte Carlo returns but introduces bias when V_phi is inaccurate. The bias-variance tradeoff is controlled by lambda:
- lambda = 0: A_t = delta_t (low variance, high bias — depends entirely on V_phi accuracy)
- lambda = 1: A_t = sum of TD errors = Monte Carlo return - V_phi(s_t) (low bias, high variance)

### Failure Modes

1. **Critic instability**: V_phi can diverge, especially for long-CoT where intermediate values are poorly defined. Both Kimi and MiniMax independently abandoned value networks for this reason.

2. **Rare token masking**: When r_t exceeds the clip boundary for a positive advantage, the gradient is exactly zero. For rare reasoning tokens (prob ~0.001-0.01), even small parameter updates push r_t outside [0.8, 1.2], causing these tokens to receive zero gradient signal. This is the "Matthew Effect" — common tokens get trained, rare tokens get stuck.

3. **Memory cost**: Requires 4 models simultaneously: policy, reference, value network, reward model. At 7B scale, this is ~28B parameters in memory.

4. **Hyperparameter sensitivity**: Performance critically depends on epsilon, beta, c_1, lambda, gamma — 5+ interacting hyperparameters.

### Suitability for Small MoE Models (1-7B)

**Poor**. The 4-model memory overhead is prohibitive. The value network is fundamentally flawed for long-CoT reasoning. The symmetric clipping creates the rare-token masking problem that is especially severe in MoE architectures where expert routing changes create additional probability shifts.

---

## A.2 GRPO (Group Relative Policy Optimization) — DeepSeek

### Objective Function

```
J_GRPO(theta) = E_q[ (1/G) sum_{i=1}^G (1/|y_i|) sum_t
  min( r_{i,t}(theta) * A_i, clip(r_{i,t}(theta), 1-eps, 1+eps) * A_i )
  - beta * D_KL(pi_theta || pi_ref) ]
```

Where the advantage is group-relative:
```
A_i = (R_i - mu_G) / (sigma_G + epsilon)
mu_G = (1/G) sum_{j=1}^G R_j
sigma_G = sqrt( (1/G) sum_{j=1}^G (R_j - mu_G)^2 )
```

### Why This Objective

GRPO solves two problems:
1. **Eliminates the value network** — replaces GAE with group-relative advantage estimation
2. **Reduces memory by 50%** — only 2 models (policy + reference)

The key insight: given G completions for the same prompt, the relative reward within the group provides a natural baseline. If completion i scores above the group mean, its advantage is positive; below, negative.

**Mathematical justification**: The group mean is an unbiased estimator of V(x) = E_{y~pi}[R(x,y)] when G is large. The normalization by sigma_G ensures the advantage has unit variance, stabilizing gradient magnitudes across different prompts.

### Gradient Expression

```
nabla_theta J_GRPO = E_q[ (1/G) sum_i (1/|y_i|) sum_t
  A_i * nabla_theta log pi_theta(y_{i,t}|x, y_{i,<t})
  * I(r_{i,t} in clip region OR pessimistic case)
  - beta * nabla_theta D_KL ]
```

### Variance Analysis

The group-relative advantage has variance:
```
Var(A_i) = Var(R_i) * (1 - 1/G) / Var(R)_empirical
```

**Critical bias issue**: Per-prompt normalization introduces bias. Consider:
- Prompt A: all rewards = {0, 0, 0, 0} → A_i = 0 for all (zero gradient)
- Prompt B: all rewards = {1, 1, 1, 1} → A_i = 0 for all (zero gradient)
- Prompt C: rewards = {0, 0, 1, 1} → A_i = {-1, -1, +1, +1} (non-zero gradient)

Only prompts with MIXED rewards contribute to learning. This is a form of selection bias — the effective training distribution is filtered to "medium difficulty" prompts. This is not necessarily bad (it focuses learning), but it IS biased relative to the true policy gradient.

The bias can be quantified:
```
Bias = E_pi[R(x,y)] - E_q[mu_G(x)]  (vanishes as G -> infinity, but G is typically 4-32)
```

For G=4, the bias is O(1/sqrt(G)) = O(0.5), which is substantial.

### Failure Modes

1. **Symmetric clipping caps exploration** (Matthew Effect): Same as PPO — rare tokens with r_t > 1+eps get zero gradient when A > 0.

2. **All-same-reward prompts**: When all G completions get the same reward, sigma_G ≈ 0 and A_i ≈ 0. These prompts contribute zero gradient but consume compute. DAPO's dynamic sampling directly addresses this.

3. **MoE routing instability**: Token-level IS ratios r_{i,t} depend on expert routing. When routing changes between pi_old and pi_theta (which happens for ~10% of experts per gradient step in a 30B MoE), the ratio captures both the policy change AND the routing change, corrupting the gradient signal. GSPO was designed specifically to solve this.

4. **Length bias**: The (1/|y_i|) normalization means longer completions have each token contribute less. But all tokens share the same advantage A_i. This creates a bias toward shorter completions that happen to be correct.

### Suitability for Small MoE Models

**Moderate**. Memory-efficient (2 models), conceptually simple, but the MoE routing instability is a fundamental flaw. At 1-7B scale, the instability may be tolerable (fewer experts = less routing volatility), but it limits scaling.

---

## A.3 CISPO (Clipped Importance Sampling Policy Optimization) — MiniMax

### Objective Function

```
J_CISPO(theta) = (1/T_total) sum_i sum_t
  sg( clamp(r_{i,t}(theta), max=1+eps_high) ) * A_i * log pi_theta(y_{i,t}|x, y_{i,<t})
```

Where `sg(·)` denotes stop-gradient (`.detach()` in PyTorch).

### Why This Objective

CISPO solves the **rare-token gradient masking** problem of PPO/GRPO. The key observation:

In PPO/GRPO, the gradient of the clipped surrogate is:
```
nabla_theta L_PPO = A * nabla_theta [clip(r_t, 1-eps, 1+eps) * r_t]
```

When r_t is outside the clip range, nabla_theta r_t = nabla_theta log pi_theta * r_t, but the clip function kills this entirely. The gradient is literally zero.

CISPO's insight: **separate the importance weight from the gradient computation**:
```
nabla_theta J_CISPO = (1/T) sum_i sum_t sg(r_hat_{i,t}) * A_i * nabla_theta log pi_theta(y_{i,t})
```

The clipped weight `sg(r_hat)` is a CONSTANT scalar multiplier. The gradient flows exclusively through `log pi_theta`. Every token gets a gradient proportional to:
```
gradient ∝ sg(r_hat) * A_i * (1/pi_theta(y_t))
```

This is a weighted REINFORCE gradient where the weights are fixed importance corrections.

### Gradient Expression

```
nabla_theta J_CISPO = (1/T) sum_i sum_t
  clamp(r_{i,t}, max=1+eps_high)|_{detached} * A_i * nabla_theta log pi_theta(y_{i,t})
```

Note: no lower clipping bound. This means:
- When r_t > 1+eps_high: weight is capped at (1+eps_high), gradient still flows
- When r_t < 1: weight can go to zero, but gradient still flows through log pi_theta
- When r_t << 1: weight approaches zero, effectively reducing gradient magnitude for tokens the policy has moved away from

### Variance Analysis

```
Var(nabla J_CISPO) = (1/T^2) sum_t Var(sg(r_hat_t) * A * nabla log pi(y_t))
                   = (1/T^2) sum_t E[sg(r_hat_t)^2 * A^2 * ||nabla log pi(y_t)||^2] - (...)
```

The key variance factor is `sg(r_hat_t)^2`. Without clamping, this is `r_t^2 = (pi_theta/pi_old)^2`, which can explode for rare tokens. The upper clamp at `(1+eps_high)^2` bounds this variance.

However, there is no LOWER clamp, so tokens where pi_theta << pi_old get vanishing weights. This does not zero the gradient (since gradient flows through log pi), but it reduces the effective learning signal for these tokens.

### Failure Modes

1. **Training-inference mismatch accumulation**: Over K gradient steps per generation, pi_theta drifts from pi_old. The IS weights become increasingly stale. MiniMax uses K=16 steps, which is aggressive — the mismatch can cause performance collapse.

2. **Entropy collapse**: DISPO paper (arXiv:2602.00983) shows CISPO causes entropy collapse because the detached weights create an exploration-distillation imbalance. The upper clamp reduces the amplification of low-probability tokens (which would increase entropy), while no lower clamp allows high-probability tokens to be amplified freely (which decreases entropy).

3. **FP32 requirement at LM head**: BF16 quantization errors in logits can flip the sign of IS ratios for rare tokens. This is a practical engineering constraint, not a mathematical one, but it increases memory cost.

4. **MoE incompatibility**: Still uses token-level IS ratios → same routing instability as GRPO.

### Suitability for Small MoE Models

**Moderate**. The guaranteed non-zero gradients are valuable for small models where every gradient signal matters. The FP32 LM head requirement is manageable at 1-7B scale. However, MoE routing instability remains. The Adam eps=1e-15 trick is important and transferable.

---

## A.4 DISPO (Decoupled Importance Sampling Policy Optimization)

### Objective Function

```
J_DISPO(theta) = (1/T_total) sum_i sum_t
  sg( r^d_{i,t}(theta) ) * A_{i,t} * log pi_theta(y_{i,t}|x, y_{i,<t})
```

Where the decoupled clipping has FOUR regimes:
```
r^d_{i,t} = {
  clamp(r_t, 1-eps+_low, 1+eps+_high)    if A_{i,t} > 0   (correct responses)
  clamp(r_t, 1-eps-_low, 1+eps-_high)    if A_{i,t} < 0   (incorrect responses)
}
```

### Why This Objective (Derivation from First Principles)

DISPO recognizes that the policy update has four qualitatively different regimes:

| Regime | Advantage | Ratio Direction | Effect | Desired Behavior |
|--------|-----------|-----------------|--------|-----------------|
| 1 | A > 0, r > 1 | Amplifying correct | Entropy increase (exploration) | Allow with eps+_high |
| 2 | A > 0, r < 1 | Suppressing correct | Entropy decrease (distillation) | Limit with eps+_low |
| 3 | A < 0, r > 1 | Amplifying incorrect | Prevents repetition collapse | Allow with eps-_high |
| 4 | A < 0, r < 1 | Suppressing incorrect | Prevents length reduction | Limit with eps-_low |

The key insight: PPO/GRPO/CISPO all use the same clipping parameters regardless of regime. But optimal exploration requires different treatment:
- Regime 1 (explore correct): high eps+_high to allow exploration
- Regime 2 (distill correct): moderate eps+_low to prevent over-distillation
- Regime 3 (avoid incorrect): moderate eps-_high to prevent repetition
- Regime 4 (forget incorrect): low eps-_low to prevent excessive forgetting

### Gradient Expression

```
nabla J_DISPO = (1/T) sum_i sum_t sg(r^d_{i,t}) * A_{i,t} * nabla log pi_theta(y_{i,t})
```

Same structure as CISPO but with 4-way decoupled weights.

### Variance Analysis

The 4-way clipping provides finer variance control:
```
Var(nabla J_DISPO) <= (1/T^2) sum_t max(eps+_high, eps-_high)^2 * max(A)^2 * max(||nabla log pi||^2)
```

The additional hyperparameters (4 clip bounds vs 1-2) allow tighter variance control at the cost of hyperparameter tuning.

### Failure Modes

1. **Hyperparameter complexity**: 4 clip bounds require careful tuning. The search space is 4-dimensional vs 1-2 for simpler methods.

2. **Token-level IS ratios**: Still fundamentally token-level → MoE routing instability remains.

3. **Advantage sign sensitivity**: The regime selection depends on sign(A), which can be noisy near A≈0. Misclassification of regime for borderline tokens can cause inconsistent gradients.

### Suitability for Small MoE Models

**Moderate-to-Good** for dense models, **Poor** for MoE due to token-level IS. The 4-regime control is most beneficial when the model has sufficient capacity to benefit from fine-grained gradient control — may be overkill for 1B models but valuable for 7B.

---

## A.5 STAPO (Spurious-Token-Aware Policy Optimization)

### Objective Function

```
J_STAPO(theta) = (1/sum I^S2T) sum_i sum_t
  I^S2T_{i,t} * min( rho_{i,t} * A_i, clip(rho_{i,t}, 1-eps_low, 1+eps_high) * A_i )
```

Where the S2T (Silencing Spurious Tokens) mask is:
```
I^S2T_{i,t} = 0    if A_i > 0 AND pi(y_{i,t}) < tau_p AND H_t < tau_h
            = 1    otherwise

tau_p = 0.002  (probability threshold)
tau_h = P_80(H)  (80th percentile of entropy within mini-batch)
```

### Why This Objective

STAPO addresses a specific pathology of token-level policy gradients. Consider the gradient for a single token:
```
nabla log pi_theta(y_t) ∝ 1/pi_theta(y_t)
```

For a token with pi(y_t) = 0.001 and positive advantage A > 0:
- Gradient magnitude: 1/0.001 = 1000
- Compare to a token with pi = 0.1: gradient = 10

The rare token gets 100x amplification. If this token is genuinely part of the correct reasoning, this is desirable. But ~0.01% of tokens are "spurious": they appear in correct completions by chance, have low probability, low entropy (the model is not uncertain about them — they are rare but confident predictions), and contribute nothing to reasoning quality.

These spurious tokens create disproportionately large gradients that dominate the update, pushing the policy toward amplifying meaningless tokens.

**S2T identifies spurious tokens** via three conditions:
1. A_i > 0 (token is in a correct completion — spurious tokens in incorrect completions have negative advantage and should be suppressed anyway)
2. pi(y_t) < tau_p (low probability — high gradient magnitude)
3. H_t < tau_h (low entropy — model is confident, not exploring)

Condition 3 is critical: genuinely important rare tokens tend to have HIGH entropy (the model is uncertain and exploring). Spurious tokens have LOW entropy (the model is confidently wrong about this token's importance).

### Gradient Expression

```
nabla J_STAPO = (1/sum I^S2T) sum_{masked tokens}
  A_i * nabla log pi_theta(y_{i,t}) * I(in clip region)
```

The gradient is identical to DAPO/GRPO within the non-masked set. The mask simply removes the highest-variance gradient contributions.

### Variance Analysis

The variance reduction from S2T masking is:
```
Var_reduction = sum_{spurious tokens} sg(r_hat)^2 * A^2 * (1/pi(y_t))^2
```

Since spurious tokens have pi(y_t) < 0.002, the per-token variance contribution is > (1/0.002)^2 = 250,000. Removing ~0.01% of tokens can reduce total gradient variance by 10-50% (empirical estimate from the paper).

### Failure Modes

1. **Threshold sensitivity**: tau_p and tau_h are magic numbers. The 80th percentile entropy threshold is batch-dependent and could be unstable across training.

2. **False positives**: Genuinely important rare tokens with low entropy will be masked. This is the exploration-exploitation tradeoff embedded in the mask design.

3. **Interaction with MoE**: Token-level masking + token-level IS ratios = compounded MoE instability.

### Suitability for Small MoE Models

**Moderate**. S2T masking is valuable at any scale since spurious token amplification affects all models. However, the thresholds may need recalibration for smaller models with different probability distributions. MoE instability remains unaddressed.

---

## A.6 GSPO (Group Sequence Policy Optimization) — Qwen3

### Objective Function

```
J_GSPO(theta) = (1/G) sum_i (1/|y_i|) sum_t
  min( s_i(theta) * A_i, clip(s_i(theta), 1-eps, 1+eps) * A_i )
```

Where the SEQUENCE-LEVEL importance ratio is:
```
s_i(theta) = exp( (1/|y_i|) sum_t log(pi_theta(y_{i,t}|...) / pi_old(y_{i,t}|...)) )
           = ( prod_t pi_theta(y_{i,t}) / pi_old(y_{i,t}) )^{1/|y_i|}
```

This is the length-normalized geometric mean of per-token IS ratios.

### Why This Objective (First-Principles Derivation)

**The fundamental problem with token-level IS ratios in MoE**:

In a MoE model, the probability of a token is:
```
pi(y_t|x) = sum_e g_e(x) * f_e(y_t|x)
```
where g_e is the gating probability and f_e is the expert output.

When parameters change from theta_old to theta:
```
r_t = pi_theta(y_t) / pi_old(y_t) = sum_e g_e^new * f_e^new / (sum_e g_e^old * f_e^old)
```

Even if the expert outputs f_e barely change, the gating probabilities g_e can shift dramatically (expert routing changes for ~10% of experts per update in a 30B MoE). This creates large r_t values that reflect routing changes, NOT policy improvement.

**GSPO's solution**: aggregate to sequence level:
```
s_i = exp( (1/|y_i|) sum_t log r_t )
    = geometric_mean(r_1, r_2, ..., r_T)
```

The geometric mean has a crucial property: individual outlier r_t values (from routing changes) are dampened by the averaging. If 10% of tokens have r_t = 5 (routing change) and 90% have r_t ≈ 1 (no change), then:
- Token-level: 10% of tokens get 5x amplified gradients
- Sequence-level: s_i = exp(0.9*0 + 0.1*ln(5)) = exp(0.161) = 1.17 → modest amplification

The length normalization (1/|y_i|) prevents long sequences from having extreme products.

### Gradient Expression

```
nabla J_GSPO = (1/G) sum_i A_i * nabla s_i(theta) / s_i(theta) * (clip-dependent terms)
             = (1/G) sum_i A_i * (1/|y_i|) sum_t nabla log pi_theta(y_{i,t}) * (clip terms)
```

**Key property**: ALL tokens in sequence i share the same weight s_i. The gradient for each token is:
```
nabla_{theta} log pi_theta(y_{i,t}) * A_i * (s_i if in clip region)
```

This is mathematically equivalent to REINFORCE with a reweighted advantage, where the reweighting is the sequence-level IS correction.

### Variance Analysis

```
Var(nabla J_GSPO) = (1/G^2) sum_i A_i^2 * Var(s_i * nabla log pi sum)
```

The variance of s_i is dramatically lower than the variance of individual r_t:
```
Var(s_i) = Var(exp(mean(log r_t))) ≈ s_i^2 * Var(mean(log r_t))
         = s_i^2 * Var(log r_t) / |y_i|
```

The 1/|y_i| factor means sequence-level variance DECREASES with sequence length, whereas token-level variance is proportional to the number of tokens. This is a fundamental advantage for long-CoT.

### Failure Modes

1. **Coarse credit assignment**: All tokens share the same s_i weight. The policy cannot differentially update important vs unimportant tokens within a sequence. However, the gradient through log pi_theta(y_t) still differentiates tokens by their current probability.

2. **Sequence-level clipping may be too conservative**: If the average ratio is within clip range but individual tokens have extreme ratios, the clipping does not trigger. This could allow some instability to leak through.

3. **GSPO-token variant**: Qwen team also explored keeping token-level loss but with sequence-level clipping, partially addressing the credit assignment concern.

### Suitability for Small MoE Models

**Excellent**. This is the mathematically principled solution for MoE + RL:
- Eliminates routing instability by construction
- No Routing Replay needed (saves compute)
- Variance decreases with sequence length (ideal for CoT)
- Minimal hyperparameters (just eps)
- Used in production for Qwen3 (proven at scale with MoE)

For NanoSeek (64 experts, top-8): GSPO is the recommended primary algorithm.

---

## A.7 Online Mirror Descent / Squared-Loss Surrogate — Kimi K1.5/K2.5

### Objective Function

The RL objective is formulated as online mirror descent with relative entropy regularization:

```
theta_{i+1} = argmax_theta  E_{y~pi_theta}[r(x,y)] - tau * KL(pi_theta || pi_{theta_i})
```

Where pi_{theta_i} is the PREVIOUS ITERATION's policy (not a frozen reference).

The practical loss function uses a squared-loss surrogate:
```
L(theta) = E[ (r(x,y,y*) - tau * log Z - tau * log(pi_theta(y,z|x) / pi_{theta_i}(y,z|x)))^2 ]
```

Which yields the gradient:
```
nabla L ∝ nabla log pi_theta(y_j|x) * (r(x,y_j,y*) - r_bar) - (tau/2) * nabla (log(pi_theta/pi_{theta_i}))^2
```

In practice, Kimi uses:
```
policy_loss = -advantages * log pi_theta(y|x)
reg_loss = (tau/2) * sum_t (log pi_theta(y_t) - log pi_{theta_i}(y_t))^2   # L2 on log-ratio
total_loss = policy_loss + reg_loss
```

### Why This Objective (Mathematical Derivation)

**Mirror descent background**: Mirror descent generalizes gradient descent by using a Bregman divergence instead of Euclidean distance for the proximal step:

```
theta_{i+1} = argmin_theta [ -<nabla J(theta_i), theta> + (1/eta) * D_psi(theta, theta_i) ]
```

When D_psi is the KL divergence in policy space, this becomes the natural policy gradient. The online version updates the reference at each iteration.

**Why L2 on log-ratio instead of KL**:

KL divergence is asymmetric:
```
KL(pi_theta || pi_ref) = E_{pi_theta}[ log(pi_theta/pi_ref) ]    (forward KL)
KL(pi_ref || pi_theta) = E_{pi_ref}[ log(pi_ref/pi_theta) ]      (reverse KL)
```

The L2 regularizer on log-ratio:
```
(1/2) * sum_t (log pi_theta(y_t) - log pi_ref(y_t))^2
```

This is SYMMETRIC and equal to neither forward nor reverse KL. Taylor expanding around pi_theta ≈ pi_ref:
```
(1/2)(log(pi_theta/pi_ref))^2 ≈ (1/2)(pi_theta/pi_ref - 1)^2  (for small deviations)
```

This approximates both KL(pi_theta||pi_ref) ≈ (1/2)E[(pi_theta/pi_ref - 1)^2] for small deviations. But for LARGE deviations, the L2 on log-ratio penalizes MORE strongly than either KL direction, providing a stronger constraint against catastrophic policy shifts.

### Gradient Expression

```
nabla L = -E[ A_j * nabla log pi_theta(y_j|x) ] + tau * sum_t (log(pi_theta/pi_{theta_i})) * nabla log pi_theta(y_t)
```

The first term is standard REINFORCE. The second term is a per-token penalty that pushes pi_theta back toward pi_{theta_i} in proportion to how far it has drifted.

### Variance Analysis

The variance of the gradient has two components:
1. REINFORCE variance: Var(A_j * nabla log pi) — standard policy gradient variance
2. Regularization variance: Var(tau * log-ratio * nabla log pi) — typically low since log-ratio is deterministic given the token

The squared-loss formulation has been noted to closely mirror SPPO/GPO in structure. The practical variance is comparable to GRPO since both use group baselines.

### Failure Modes

1. **Squared loss over-penalization**: The quadratic penalty grows faster than linear for large deviations, which may prematurely constrain exploration. This is by design (safety) but at the cost of slower capability improvement.

2. **Reference policy staleness with online updates**: If iterations are long, the reference pi_{theta_i} may drift significantly from the actual behavior distribution, creating a mismatch between the regularization target and the actual constraint.

3. **Tau tuning**: The temperature tau controls the exploration-exploitation tradeoff. Too high → over-constrained, too low → instability. The optimal tau is not known and likely task-dependent.

### Suitability for Small MoE Models

**Good**. The L2 on log-ratio provides strong stability guarantees, which is valuable for small models where training is more sensitive. No value network needed. The symmetric regularization is particularly well-suited for MoE since it constrains deviations in both directions. However, the method has not been specifically designed for MoE, and token-level regularization still interacts with routing changes.

---

## A.8 DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization)

### Objective Function

```
J_DAPO(theta) = E[ (1/sum|y_i|) sum_i sum_t
  min( r_{i,t} * A_i, clip(r_{i,t}, 1-eps_low, 1+eps_high) * A_i ) ]
```

With four key innovations over GRPO:

**1. Asymmetric Clipping (Clip-Higher)**: eps_low=0.2, eps_high=0.28
```
clip(r_t, 0.8, 1.28)   vs GRPO's   clip(r_t, 0.8, 1.2)
```

The higher upper bound allows MORE increase in probability for positive-advantage tokens, promoting exploration.

**2. Dynamic Sampling**: Filter prompts where all G completions have the same reward:
```
Keep prompt iff: 0 < |{y_i : correct}| < G
```

**3. Token-Level Loss Normalization**: Divide by total tokens, not by number of samples:
```
(1/sum|y_i|) instead of (1/G)(1/|y_i|)
```

This means longer sequences contribute proportionally more to the loss.

**4. Overlong Reward Shaping**: Soft length penalty:
```
R_shaped = R * max(0, 1 - (len - L_max) / L_penalty)
```

### Why These Modifications

**Clip-Higher**: Entropy dynamics analysis (arXiv:2505.22617) shows that:
- clip-low (penalizing probability decrease) INCREASES entropy (more exploration)
- clip-high (penalizing probability increase) DECREASES entropy (less exploration)
- In standard symmetric clipping, clip-high dominates → net entropy decrease → collapse

By raising eps_high (relaxing the upper clip), DAPO reduces the entropy-decreasing effect.

**Dynamic Sampling**: From the variance analysis of GRPO, prompts with sigma_G ≈ 0 contribute near-zero gradient but full compute cost. Filtering these improves compute efficiency by 20-40%.

### Failure Modes

1. **Dynamic sampling reduces effective batch size**: Filtering all-correct/all-incorrect prompts can remove 30-50% of prompts, reducing the effective batch size and increasing variance.

2. **Clip-Higher is insufficient**: Despite raising eps_high, clip-high still dominates over training, causing net entropy decrease. The fundamental problem of clipping-based entropy control remains.

3. **Token-level IS ratios**: Same MoE instability as GRPO.

### Suitability for Small MoE Models

**Moderate**. The innovations are valuable improvements over GRPO but do not solve the fundamental MoE problem. For a dense 1-7B model, DAPO is a strong choice. For MoE, prefer GSPO.

---

## A.9 REINFORCE++

### Objective Function

```
L_REINFORCE++(theta) = -E[ A_global(y) * log pi_theta(y|x) ]
```

Where:
```
A_global(y) = (R(y) - mu_batch) / (sigma_batch + eps)
```

The advantage is normalized across the ENTIRE global batch, not per-prompt.

### Why This Objective

REINFORCE++ identifies that per-prompt normalization (as in GRPO) is a BIASED estimator:

```
E[A_GRPO] = E[(R_i - mu_G) / sigma_G] ≠ 0   (biased when G is finite)
E[A_REINFORCE++] = E[(R_i - mu_batch) / sigma_batch] → 0   (bias vanishes as batch → infinity)
```

The global normalization provides an approximately unbiased advantage estimate when the batch is large enough.

### Gradient Expression

```
nabla L = -E[ (R(y) - mu_batch) / sigma_batch * nabla log pi_theta(y|x) ]
```

This is the classic REINFORCE gradient with a learned baseline (batch mean) and variance normalization.

### Variance Analysis

```
Var(nabla L) = Var(A_global) * E[||nabla log pi||^2]
             = (1 - 1/N_batch) * E[||nabla log pi||^2]  (after normalization)
```

For large batches (N > 100), the variance is well-controlled. But for small batches, the normalization is noisy.

### Suitability for Small MoE Models

**Good**. Very simple, memory-efficient (1-2 models), no IS ratios → no MoE routing issues. The lack of IS ratios means it is a purely on-policy method with no trust-region mechanism, which can cause instability. Adding PPO-style clipping (as REINFORCE++ does) partially addresses this.

---

## A.10 RLOO (REINFORCE Leave-One-Out)

### Objective Function

```
L_RLOO(theta) = -E[ (R_i - (1/(K-1)) sum_{j≠i} R_j) * log pi_theta(y_i|x) ]
```

The baseline for completion i is the average reward of ALL OTHER completions for the same prompt.

### Why This Objective

Leave-one-out baselines have lower variance than the full-mean baseline:

```
Var(R_i - b_LOO) = Var(R_i) * (1 - 1/(K-1))     (approximately)
Var(R_i - b_mean) = Var(R_i) * (1 - 1/K)          (for GRPO-style mean)
```

The LOO baseline also avoids the "self-correlation" bias: in GRPO, R_i appears in its own baseline computation, which creates a bias term of order O(1/G).

### Suitability for Small MoE Models

**Good**. Same benefits as REINFORCE++ (no IS ratios = MoE safe) with better variance reduction. The cost is K completions per prompt (same as GRPO).

---

## A.11 New 2026 Algorithms

### OAPL (Optimal Advantage-based Policy Optimization with Lagged Inference Policy)

**Origin**: Databricks/Mosaic AI, arXiv:2602.19362

```
L_OAPL(theta) = E[ (log(pi_theta/pi_ref) - A*(x,y)/tau)^2 ]
```

Where A*(x,y) is the OPTIMAL advantage (derived from closed-form solution of KL-regularized RL).

**Key insight**: The optimal policy satisfies pi*(y|x) ∝ pi_ref(y|x) * exp(A*(x,y)/tau). Taking logs:
```
log(pi*(y|x)/pi_ref(y|x)) = A*(x,y)/tau - log Z(x)
```

OAPL regresses the log-ratio directly to the scaled optimal advantage. This is a squared regression loss, not a policy gradient.

**MoE suitability**: Excellent — no IS ratios, no clipping, regression-based loss is inherently stable.

### A*-PO (Optimal Advantage Regression)

**Origin**: Harvard Kempner Institute, arXiv:2505.20686

Separates value estimation (offline, using any data) from policy learning (online). The policy update regresses on estimated optimal advantages rather than computing policy-specific advantages.

**Key property**: 2x faster than GRPO/PPO with comparable performance, smallest KL divergence from reference.

### EPO (Entropy-regularized Policy Optimization)

**Origin**: arXiv:2509.22576

Adds an entropy smoothing regularizer:
```
L_EPO = L_base + alpha * max(0, |H(pi_theta) - H_avg| - delta)^2
```

This bounds policy entropy within a moving average, preventing both entropy collapse and entropy explosion.

### AEPO (Agentic Entropy-Balanced Policy Optimization)

**Origin**: arXiv:2510.14545

Extends EPO to multi-turn agentic settings with:
- Dynamic entropy-balanced rollouts with pre-monitoring
- Branch penalty for consecutive high-entropy tool-call steps
- Stop-gradient on high-entropy clipping terms

---

# B. Reward Design Analysis

## B.1 Outcome-Based Binary Rewards (Correct/Incorrect)

### Mathematical Properties

```
R(x, y) ∈ {0, 1}
```

**Signal strength**: Maximum for discriminating correct vs incorrect, but zero information about WHY something is wrong or HOW CLOSE to correct.

**Variance**: For a problem with success rate p:
```
Var(R) = p(1-p)
```

Maximum variance at p=0.5, zero at p=0 or p=1. This naturally creates a "zone of proximal development" — problems at p≈0.5 provide the strongest learning signal.

**Sample efficiency**: For binary rewards with group-relative advantages:
```
Effective samples per prompt = G * p * (1-p)
```

Only the CONTRAST between correct and incorrect completions provides signal. If all G are correct or all incorrect, signal is zero.

**Failure modes**:
- Cannot distinguish "almost correct" from "completely wrong"
- Cannot guide partial credit for intermediate reasoning steps
- Requires verifiable tasks (math, code with test cases)

## B.2 Process Reward Models (PRM)

### Mathematical Properties

```
R_PRM(x, y) = sum_{s=1}^S w_s * PRM(x, y_{1:s})
```

Where S is the number of reasoning steps and w_s are step weights (often uniform).

**Signal strength**: Dense supervision at each step → much stronger gradient signal per trajectory.

**Variance**:
```
Var(R_PRM) = sum_s w_s^2 * Var(PRM_s) + cross-covariance terms
```

Lower variance than ORM because each step provides independent signal, reducing the credit assignment problem.

**Sample efficiency**: O(S) times more efficient than ORM per trajectory, where S is the number of reasoning steps.

**Failure modes**:
- Requires step-level annotation (expensive)
- PRM can be gamed — model learns to produce "PRM-pleasing" intermediate steps that do not improve final correctness
- Implicit PRM (training as ORM, using as PRM) avoids annotation cost but introduces approximation error
- Step boundary detection is non-trivial for free-form CoT

## B.3 Outcome Reward Models (ORM)

```
R_ORM(x, y) = RM_phi(x, y) ∈ R
```

A learned scalar reward from a trained reward model.

**Signal strength**: Continuous, provides gradient signal even for approximately-correct answers. But the signal is only as good as RM_phi.

**Variance**: Depends on RM accuracy. For a well-calibrated RM:
```
Var(R_ORM) ≈ Var(R_true) + Var(RM_noise)
```

**Sample efficiency**: Better than binary (continuous signal) but worse than PRM (no step-level credit).

**Failure modes**: Reward hacking — the model exploits RM weaknesses (length bias, sycophancy, sophistication bias). The reward-hacking risk grows super-linearly with RL training duration.

## B.4 Format/Style Rewards

```
R_format(x, y) = sum_k lambda_k * f_k(y)
```

Where f_k are format checkers (e.g., has_boxed_answer, within_length_limit, uses_step_markers).

**Signal strength**: Weak for capability improvement, but strong for compliance. Format rewards shape output structure without improving reasoning.

**Failure modes**: Over-optimizing format rewards degrades reasoning quality. The model learns to produce well-formatted wrong answers.

## B.5 Verifier-Based Rewards (Math/Code)

```
R_verify(x, y) = Verify(extract(y), ground_truth(x))   ∈ {0, 1}
```

**Signal strength**: Maximum reliability — no reward model errors, no hacking possible (for correctly implemented verifiers).

**Practical considerations**:
- Math: SymPy-based verification has false negatives (equivalent expressions not recognized). JustRL's rule-based verifier (no SymPy) avoids this.
- Code: Test-based verification is reliable but requires test suites.
- Difficulty filtering: Only train on problems where 0 < pass@G < 1 (following MiniMax's 10-90% filter).

## B.6 Multi-Objective Reward Composition

```
R_total(x, y) = sum_k alpha_k * R_k(x, y)
```

**Mathematical challenge**: Different reward sources have different scales, distributions, and noise levels. Linear combination with fixed weights creates a moving target as each component's distribution shifts during training.

**Best practice**: Kimi's approach of annealing auxiliary rewards (lambda → 0) is mathematically sound — it uses auxiliary rewards as training wheels during early RL, then removes them to prevent reward hacking.

---

# C. Regularization and Stability Analysis

## C.1 Forward KL Divergence Penalty

```
D_KL(pi_theta || pi_ref) = E_{pi_theta}[ log(pi_theta / pi_ref) ]
```

**What it prevents**: Mode-seeking behavior. pi_theta tries to cover all modes of pi_ref.

**Gradient**:
```
nabla D_KL = E_{pi_theta}[ (1 + log(pi_theta/pi_ref)) * nabla log pi_theta ]
```

**Cost**: Forces pi_theta to maintain coverage over ALL of pi_ref's probability mass, including low-quality regions. This prevents the policy from concentrating on high-reward modes.

**Estimation issues**: Requires samples from pi_theta, which is the current policy. Estimator variance depends on the support overlap between pi_theta and pi_ref.

## C.2 Reverse KL Divergence Penalty

```
D_KL(pi_ref || pi_theta) = E_{pi_ref}[ log(pi_ref / pi_theta) ]
```

**What it prevents**: Mode-dropping. pi_theta avoids dropping any mode of pi_ref.

**Cost**: Requires samples from pi_ref, which is computationally expensive if pi_ref is frozen.

**Key distinction from forward KL**: Forward KL is zero-forcing (pi_theta = 0 wherever pi_ref > 0 is penalized infinity), while reverse KL is zero-avoiding (pi_ref = 0 wherever pi_theta > 0 is penalized).

## C.3 L2 on Log-Ratio (Kimi's Approach)

```
R_L2 = (tau/2) * sum_t (log pi_theta(y_t) - log pi_ref(y_t))^2
```

**What it prevents**: Large deviations in EITHER direction, symmetrically.

**Gradient**:
```
nabla R_L2 = tau * sum_t (log(pi_theta/pi_ref)) * nabla log pi_theta
```

**Comparison to KL**:
- For small deviations (|log ratio| < 1): L2 ≈ KL (Taylor approximation)
- For large deviations (|log ratio| >> 1): L2 grows as O(log-ratio^2) vs KL's O(log-ratio), providing STRONGER penalization

**Cost**: More conservative than KL — prevents exploration of distant policies. This is the explicit tradeoff: stability over exploration speed.

**Key advantage**: Symmetric, so it penalizes both "moving toward" and "moving away from" the reference equally. This is valuable in MoE where routing changes can create bidirectional probability shifts.

## C.4 Hard Clipping (PPO-Style)

```
clip(r_t, 1-eps, 1+eps)
```

**What it prevents**: The policy from moving too far from the behavior policy in a single update.

**Cost**:
- Creates zero gradients when r_t is outside the clip range → rare token masking
- Symmetric clipping constrains both exploration (A > 0) and distillation (A < 0) equally
- The clip boundary is a hard threshold — no gradual transition

**When this breaks**: When the policy has already moved significantly (multi-step updates), the clip boundaries become meaningless relative to the actual policy distance.

## C.5 Soft Clipping (CISPO-Style with .detach())

```
weight = clamp(r_t, max=1+eps).detach()
loss = weight * A * log pi_theta
```

**What it prevents**: Extreme IS weight amplification while preserving gradient flow.

**Cost**: No LOWER clipping — tokens where pi_theta << pi_old get vanishing weights but non-zero gradients. This creates an asymmetry that biases toward high-probability tokens.

**When this breaks**: When the detached weights become very stale (many gradient steps per generation). The IS correction becomes inaccurate, and the "soft" weights may amplify incorrect signals.

## C.6 Entropy Bonus

```
L_entropy = -alpha * H(pi_theta) = alpha * E_{pi_theta}[log pi_theta]
```

**What it prevents**: Entropy collapse (premature convergence to a narrow policy).

**Gradient**:
```
nabla L_entropy = alpha * (1 + log pi_theta) * nabla log pi_theta
```

**Cost**: Uniform entropy bonus treats all tokens equally, encouraging exploration even for tokens that should be deterministic (e.g., mathematical operators, function names). This can slow convergence.

**Better alternatives**: Adaptive entropy (EPO/AEPO), S2T masking (STAPO), asymmetric clipping (DAPO).

## C.7 Reference Policy Management

**Frozen reference** (PPO, GRPO):
- KL penalty anchors to the SFT model
- Advantage: stable reference, prevents reward hacking
- Disadvantage: constrains the policy to stay near SFT, limiting improvement

**Moving reference** (Kimi):
- Reference updates every iteration: pi_ref ← pi_theta
- Advantage: allows continuous policy improvement without KL constraint accumulation
- Disadvantage: if an iteration goes wrong, the reference also moves wrong — no safety net

**Practical hybrid** (most implementations):
- Update reference every K iterations
- Provides a compromise between stability and progress

## C.8 Sequence-Level vs Token-Level IS Ratios

**Token-level** (PPO, GRPO, DAPO, CISPO, DISPO, STAPO):
```
r_t = pi_theta(y_t|...) / pi_old(y_t|...)
```
- Pro: Fine-grained credit assignment
- Con: MoE routing instability, extreme values for rare tokens

**Sequence-level** (GSPO):
```
s_i = geometric_mean(r_1, ..., r_T)
```
- Pro: MoE stable, dampened outliers, variance decreases with length
- Con: Coarse credit assignment (all tokens share same weight)

**Mathematical relationship**: The sequence-level ratio is the geometric mean of token-level ratios:
```
s_i = exp(mean(log r_t))
```

This is a NATURAL compression: the law of large numbers ensures that for long sequences, s_i concentrates around its expectation, providing a low-variance estimator.

---

# D. Credit Assignment Analysis

## D.1 Token-Level Advantages

```
A_t = Q(s_t, a_t) - V(s_t)
```

Requires a value network to estimate Q and V. As discussed, value networks fail for long-CoT because:
1. Intermediate "bad-looking" states may lead to correct answers (exploratory detours)
2. Value estimation for states with combinatorial action spaces (full vocabulary) is extremely noisy
3. The value function would need to be nearly as powerful as the policy itself

## D.2 Sequence-Level Advantages

```
A_i = f(R_i, {R_j}_{j=1}^G)
```

Where f is the advantage function (GRPO: z-score, RLOO: leave-one-out, REINFORCE++: global normalization).

**Tradeoff**: Sequence-level advantages provide no within-sequence credit differentiation. Token t1 that made the critical insight gets the same advantage as token t2 that wrote "the" at position 50,000. However, the gradient through log pi_theta(y_t) naturally provides some differentiation — low-probability tokens get larger gradients.

## D.3 Group-Relative Baselines (GRPO)

```
A_i = (R_i - mu_G) / sigma_G
```

**Bias analysis**: The z-score normalization is biased for finite G:
```
E[A_i] = E[(R_i - mu_G) / sigma_G] ≠ (E[R_i] - E[R]) / std(R)
```

The bias arises because:
1. mu_G includes R_i (self-correlation): bias of O(1/G)
2. sigma_G is a sample standard deviation with G-1 DOF: bias of O(1/G)
3. Division by sigma_G introduces Jensen's inequality bias: E[1/sigma] ≥ 1/E[sigma]

For G=4 (typical for small models), these biases are non-negligible. RLOO's leave-one-out baseline reduces bias 1 but not 2 or 3.

## D.4 Monte Carlo Returns

```
R_i = sum_t gamma^t * r(x, y_{1:t})  ≈ R(x, y)   (for gamma=1, single terminal reward)
```

For verifiable tasks, the Monte Carlo return equals the final binary reward. No discounting is needed because there is no intermediate reward structure.

**Variance**: MC returns have the highest variance among advantage estimation methods but zero bias. The variance reduction from group baselines or leave-one-out estimators is critical.

## D.5 Why No Value Network for Long-CoT

**Mathematical argument**: For a value network to be useful, it must satisfy:
```
V(s_t) ≈ E_{y_{t:T} ~ pi}[ R(x, y) | y_{1:t} ]
```

For long-CoT reasoning (T = 10,000+ tokens), the conditional expectation E[R | y_{1:t}] is:
1. **Non-monotonic**: A correct partial solution at step 500 may be abandoned for a better approach at step 1000. V(s_500) → V(s_1000) is not monotonically increasing.
2. **Highly discontinuous**: A single insight token can change R from 0 to 1.
3. **Unpredictable**: The space of possible continuations at each step is enormous (vocabulary size^remaining_tokens).

Training V to capture this function requires the value network to solve the same problem as the policy — predicting whether a partial solution will lead to a correct answer. This is essentially training a second LLM.

**Empirical confirmation**: Both Kimi (K1.5) and MiniMax (M1) independently abandoned value networks and found improved training stability. GLM-5 never attempted value-based methods.

---

# E. Recommended Algorithm for Small MoE Model (1-7B)

## E.1 Policy Optimization: GSPO (Primary) + GRPO/JustRL (Baseline)

### Mathematical Justification

For a small MoE model with E experts and top-k routing:

**Problem**: Token-level IS ratios in MoE models contain routing noise:
```
r_t^MoE = (sum_e g_e^new * f_e^new(y_t)) / (sum_e g_e^old * f_e^old(y_t))
```

The variance of r_t^MoE includes a routing variance component:
```
Var(r_t^MoE) = Var(r_t^dense) + Var(routing_noise) + 2*Cov(...)
```

For NanoSeek (64 experts, top-8), empirical data from GSPO paper suggests ~10% of experts change routing per update in a 30B MoE. At 1-7B scale with fewer layers, this may be 5-15%, still significant.

**GSPO eliminates this**: Sequence-level geometric mean dampens routing outliers by 1/sqrt(T), making the routing noise negligible for T > 100 tokens.

### Practical Recommendation

1. **Start with JustRL baseline** (vanilla GRPO + binary rewards): Establishes a clean baseline at minimal complexity. JustRL shows that at 1.5B, simple GRPO + scale can match sophisticated pipelines.

2. **Switch to GSPO for production training**: Once the baseline is established, GSPO provides:
   - MoE stability (mathematically eliminates routing instability)
   - Better scaling with compute (continuous improvement, no plateau)
   - Simpler infrastructure (no Routing Replay needed)

3. **Monitor with DAPO techniques**: Use dynamic sampling and overlong reward shaping as plug-in improvements regardless of base algorithm.

## E.2 Reward Design: Verifier-Based Binary Rewards (Primary) + GenRM (Secondary)

### Mathematical Justification

At small scale, sample efficiency matters most. Binary verifiable rewards have:
- **Zero reward model error** (no hacking possible)
- **Maximum discriminative signal** for fixed sample budget
- **Variance**: p(1-p), maximized at p=0.5

The difficulty filtering (0.1 < pass@G < 0.9, following MiniMax's approach) ensures:
```
Effective_signal = G * p * (1-p) * I(0.1 < p < 0.9)
```

This focuses compute on problems where the model can learn, not on trivially easy or impossibly hard problems.

**For open-ended tasks**: Use GenRM (generative reward model) over classification-based RM. GenRM's CoT reasoning about quality (as in Kimi's 98.5% accuracy self-critique) provides more reliable signals than scalar regression RM.

## E.3 Regularization: L2 on Log-Ratio + Asymmetric Clipping

### Mathematical Justification

For small models, training stability is more important than exploration speed. The L2 on log-ratio (Kimi's approach) provides:

1. **Symmetric constraint**: Penalizes both directions of policy drift equally
2. **Stronger-than-KL for large deviations**: Prevents catastrophic policy shifts
3. **Compatible with MoE**: Token-level regularization is applied BEFORE aggregation, so routing changes do not corrupt the regularization signal

Combined with GSPO's sequence-level optimization:
```
L_total = J_GSPO(theta) + (tau/2) * (1/|y|) sum_t (log pi_theta(y_t) - log pi_ref(y_t))^2
```

The asymmetric clipping (eps_low=0.2, eps_high=0.28) in the GSPO objective provides additional exploration encouragement without destabilizing.

## E.4 Summary Decision Matrix

| Component | Recommendation | Justification |
|-----------|---------------|---------------|
| **Policy optimization** | GSPO | MoE-native, sequence-level IS eliminates routing instability |
| **Baseline** | JustRL (vanilla GRPO) | Proven at 1.5B, establishes clean reference |
| **Reward (verifiable)** | Binary + difficulty filter (10-90%) | Maximum reliability, zero hacking, focused signal |
| **Reward (open-ended)** | GenRM / self-critique | Higher accuracy than scalar RM |
| **Regularization** | L2 on log-ratio (tau ~ 0.01-0.1) | Symmetric, strong constraint, MoE compatible |
| **Clipping** | Asymmetric (0.2 / 0.28) | Proven in DAPO and GLM-5 |
| **Value network** | None | Fundamentally broken for long-CoT |
| **Group size** | G = 8-16 | Balance variance reduction vs compute |
| **Dynamic sampling** | Yes | DAPO's filtering of all-same-reward prompts |
| **Reference policy** | Moving (every K iterations) | Allows continuous improvement |
| **Optimizer** | AdamW, eps=1e-15, beta2=0.95 | MiniMax's RL-specific optimizer settings |
| **Precision** | FP32 at LM head | Prevents IS ratio sign errors |

## E.5 Phased Implementation Plan

```
Phase 1: JustRL Baseline
  - Vanilla GRPO + binary rewards + rule-based verifier
  - G=8, standard clipping (eps=0.2)
  - Fixed hyperparameters, 2000+ steps
  - Establishes baseline performance

Phase 2: GSPO Upgrade
  - Switch to sequence-level IS ratios
  - Add L2 log-ratio regularization
  - Add asymmetric clipping
  - Add dynamic sampling
  - Compare to Phase 1 baseline

Phase 3: Reward Expansion
  - Add code verification via test suites
  - Add GenRM for open-ended tasks (50/50 mix)
  - Difficulty-filtered curriculum

Phase 4: Advanced Techniques (if needed)
  - STAPO's S2T masking (if spurious token amplification detected)
  - Toggle algorithm for token efficiency (if budget-constrained)
  - Multi-agent PARL (if agentic capabilities needed)
```

---

# Appendix: Algorithm Comparison Summary

## Gradient Variance Ranking (Lower = Better)

```
GSPO (seq-level, 1/T dampening) < RLOO (LOO baseline) ≈ REINFORCE++ (global norm)
  < GRPO (group relative) < DAPO (group + dynamic) < CISPO (detached weights)
    < DISPO (4-regime) < STAPO (masked tokens) < PPO (full GAE with critic noise)
```

## MoE Compatibility Ranking

```
GSPO (designed for MoE) >> OAPL (no IS ratios) > REINFORCE++ (no IS)
  > RLOO (no IS) >> Kimi L2 (token-level but symmetric)
    >> GRPO ≈ DAPO ≈ STAPO (token-level IS) > CISPO ≈ DISPO (detached token IS)
      >> PPO (4 models + token IS)
```

## Simplicity Ranking (Simpler = Better for 1B)

```
JustRL (vanilla GRPO) ≈ REINFORCE++ > RLOO > GSPO > DAPO
  > Kimi L2 > CISPO > DISPO > STAPO >> PPO
```

## Recommended Priority for NanoSeek (1B MoE, 64 experts, top-8)

```
1. GSPO (sequence-level IS, MoE-native)       ← PRIMARY
2. JustRL (simplicity baseline)                 ← BASELINE
3. DAPO (proven improvements, fallback)         ← FALLBACK
4. Kimi L2 (strong regularization, optional)    ← REGULARIZATION ADD-ON
5. OAPL (off-policy, future exploration)        ← FUTURE
```

---

## Sources

### Primary Algorithm Papers
- [GSPO (Qwen3)](https://arxiv.org/abs/2507.18071)
- [DAPO (ByteDance)](https://arxiv.org/abs/2503.14476)
- [CISPO / MiniMax-M1](https://arxiv.org/abs/2506.13585)
- [DISPO](https://arxiv.org/html/2602.00983)
- [STAPO](https://arxiv.org/abs/2602.15620)
- [REINFORCE++](https://arxiv.org/abs/2501.03262)
- [Kimi K1.5](https://arxiv.org/abs/2501.12599)
- [GRPO / DeepSeekMath](https://arxiv.org/abs/2402.03300)
- [JustRL (ICLR 2026)](https://iclr-blogposts.github.io/2026/blog/2026/justrl/)
- [OAPL - Off-Policy RL for LLMs](https://arxiv.org/abs/2602.19362)
- [A*-PO](https://arxiv.org/abs/2505.20686)
- [EPO](https://arxiv.org/abs/2509.22576)
- [AEPO](https://huggingface.co/papers/2510.14545)

### Analysis and Surveys
- [GSPO Blog (Qwen)](https://qwenlm.github.io/blog/gspo/)
- [State of RL for LLM Reasoning (Raschka)](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training)
- [Entropy Mechanism of RL for Reasoning LLMs](https://arxiv.org/abs/2505.22617)
- [Stabilizing RL with LLMs](https://arxiv.org/abs/2512.01374)
- [Scaling RL for MoE](https://arxiv.org/abs/2512.07710)
- [DAPO Implementation (verl)](https://verl.readthedocs.io/en/latest/algo/dapo.html)
- [Process Reward Models](https://www.stephendiehl.com/posts/process_reward/)
- [Reward Models (RLHF Book)](https://rlhfbook.com/c/07-reward-models.html)
