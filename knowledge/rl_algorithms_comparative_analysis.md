# Reinforcement Learning Algorithms for LLM Post-Training
## Comparative Analysis Report (as of March 2026)

---

## 1. ALGORITHM CATALOG

### 1.1 PPO (Proximal Policy Optimization)
**Origin**: Schulman et al. 2017; adapted for RLHF by Ouyang et al. (InstructGPT) 2022

**Loss Function**:
```
L_PPO(θ) = E[ min( r_t(θ) * A_t,  clip(r_t(θ), 1-ε, 1+ε) * A_t ) ]

where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
```

Plus KL penalty term: `- β * D_KL(π_θ || π_ref)`

**Key Properties**:
- Requires 4 models in memory: policy, reference policy, critic (value function), reward model
- Actor-critic architecture with learned value baseline
- Token-level importance sampling with symmetric clipping
- KL penalty prevents reward hacking

**Empirical Results**: Used to train InstructGPT, ChatGPT, early Claude models. Standard for RLHF from 2022-2024.

**Known Failure Modes**:
- High memory overhead (4x model copies)
- Critic training instability — value network can diverge
- Clipping mechanism is structurally ill-suited for large vocabularies (over-penalizes low-probability tokens, under-constrains high-probability tokens)
- Prone to reward hacking via length bias and sycophancy
- Complex implementation with many hyperparameters

**Used By**: InstructGPT, ChatGPT (early), Claude (early versions), Llama 2

---

### 1.2 GRPO (Group Relative Policy Optimization)
**Origin**: Shao et al., DeepSeekMath (Feb 2024)

**Loss Function**:
```
J_GRPO(θ) = E[ (1/G) Σ_i (1/|o_i|) Σ_t ( min(r_{i,t}(θ) * A_i, clip(r_{i,t}(θ), 1-ε, 1+ε) * A_i)
             - β * D_KL(π_θ || π_ref) ) ]

where:
  r_{i,t}(θ) = π_θ(o_{i,t} | q, o_{i,<t}) / π_θ_old(o_{i,t} | q, o_{i,<t})
  A_i = (r_i - mean({r_1,...,r_G})) / std({r_1,...,r_G})
```

**Key Innovations vs PPO**:
- Removes critic/value network entirely — critic-free
- Group-based advantage estimation: samples G responses per prompt, uses relative reward as advantage
- Only 2 models needed: policy + reference (or just policy with no KL)
- Feasible on single GPU

**Empirical Results**: Powers DeepSeek-R1. DeepSeek-R1-Zero-Qwen-32B achieves 47 on AIME 2024.

**Known Failure Modes**:
- Symmetric clipping caps good tokens too early ("Matthew Effect")
- Prompts where all samples get same reward contribute zero gradient (sampling inefficiency)
- Token-level gradients diluted in long sequences
- Severe instability in MoE architectures due to expert activation volatility
- GRPO's primary benefit comes from discarding all-correct/all-incorrect prompts, not from reward normalization per se

**Used By**: DeepSeek-R1, DeepSeek-R1-Zero, DeepSeek-Math

---

### 1.3 REINFORCE++
**Origin**: Hu (Jan 2025)

**Loss Function**:
```
L_REINFORCE++(θ) = -E[ A_global * log π_θ(a_t | s_t) ]

where A_global is normalized across the entire global batch (not per-prompt)
```

Two variants:
- REINFORCE++: global advantage normalization, no group sampling
- REINFORCE++ with baseline: group sampling variant for reasoning tasks

**Key Innovations vs GRPO**:
- Global advantage normalization (across entire batch, not per-prompt groups)
- Effectively unbiased estimator (bias vanishes as batch size increases)
- Per-prompt normalization in GRPO is biased and prone to overfitting

**Empirical Results**: Outperforms PPO in complex agentic settings. Superior stability in both general RLHF and reasoning domains.

**Known Failure Modes**:
- Requires large batch sizes for the global normalization to be effective
- Simpler but may converge slower than methods with richer advantage estimation

**Used By**: Open-source community adoption; documented in Unsloth, OpenRLHF frameworks

---

### 1.4 RLOO (REINFORCE Leave-One-Out)
**Origin**: Adapted from classical statistics for LLM training

**Loss Function**:
```
L_RLOO(θ) = -E[ (r_i - (1/(K-1)) Σ_{j≠i} r_j) * log π_θ(o_i | q) ]
```
Baseline = average reward of all OTHER samples for the same prompt (leave-one-out).

**Key Innovations vs GRPO**:
- More sample-efficient than GRPO
- Leave-one-out baseline provides lower-variance advantage estimates
- Statistically better baseline than GRPO's mean/std normalization

**Empirical Results**: Significantly more sample-efficient than GRPO for fine-tuning deep research agents. Consistently performs well across evaluation metrics.

**Known Failure Modes**:
- Still requires multiple samples per prompt
- Similar MoE instability issues as other token-level methods

---

### 1.5 DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization)
**Origin**: ByteDance Seed + Tsinghua AIR (Mar 2025)

**Loss Function**:
```
J_DAPO(θ) = E[ (1/Σ|o_i|) Σ_i Σ_t min( r_{i,t}(θ) * A_i,
             clip(r_{i,t}(θ), 1-ε_low, 1+ε_high) * A_i ) ]

subject to: 0 < |{o_i | correct}| < G  (dynamic sampling constraint)
```

Default: ε_low = 0.2, ε_high = 0.28

**Key Innovations vs GRPO** (4 techniques):
1. **Clip-Higher**: Asymmetric clipping — raises upper bound to allow exploration of low-probability tokens
2. **Dynamic Sampling**: Filters prompts with accuracy = 0 or 1 (all correct/all wrong), keeping only mixed batches
3. **Token-Level Loss**: Normalizes by total tokens (not sample count), so long sequences have proportional influence
4. **Overlong Reward Shaping**: Soft length penalty that linearly decreases reward beyond max length thresholds

**Empirical Results**: AIME 2024 = 50/30 (pass@32, Qwen2.5-32B) — 50% fewer training steps than DeepSeek-R1-Zero baseline (47).

**Known Failure Modes**:
- Still token-level IS ratios — problematic for MoE
- Clip-Higher helps entropy but clip-high still dominates, causing net entropy reduction over training
- Dynamic sampling reduces effective batch size

**Used By**: Open-source RL training (verl framework), widely adopted baseline

---

### 1.6 GSPO (Group Sequence Policy Optimization)
**Origin**: Qwen team, Alibaba (Jul 2025)

**Loss Function**:
```
J_GSPO(θ) = E[ (1/G) Σ_i (1/|o_i|) Σ_t min( s_i(θ) * A_i, clip(s_i(θ), 1-ε, 1+ε) * A_i ) ]

where s_i(θ) = exp( (1/|o_i|) Σ_t log(π_θ(o_{i,t}) / π_θ_old(o_{i,t})) )
```

The key: `s_i(θ)` is a SEQUENCE-LEVEL importance ratio (length-normalized geometric mean of token ratios).

**Key Innovations vs GRPO/DAPO**:
- Sequence-level importance ratio replaces per-token ratios
- All tokens in a sequence share the same weight — reduces variance
- Completely eliminates dependency on Routing Replay for MoE models
- Length normalization in log-space prevents extreme values

**Empirical Results**: Better performance under same training cost vs GRPO on Qwen3-30B. Continuous improvement with increased compute on AIME'24, LiveCodeBench, CodeForces.

**Known Failure Modes**:
- Sequence-level ratio may miss fine-grained token-level credit assignment
- Potential for sequence-level clipping to be too coarse

**Used By**: Qwen3 models (official training algorithm)

---

### 1.7 CISPO (Clipped Importance Sampling Policy Optimization)
**Origin**: MiniMax (Jun 2025, MiniMax-M1 paper)

**Loss Function**:
```
L_CISPO(θ) = -E[ detach(min(r_t(θ), ε_high)) * A_t * log π_θ(a_t | s_t) ]
```

Implementation:
```python
importance_weights = torch.exp(per_token_logps - old_per_token_logps)
clamped_ratios = torch.clamp(importance_weights, max=epsilon_high).detach()
per_token_loss = -clamped_ratios * advantages * per_token_logps
```

**Key Innovations vs GRPO**:
- Clips importance weights (not the surrogate objective)
- Detach operation: clipped weights are fixed coefficients, not part of gradient computation
- Gradients flow exclusively through log-probability term — all tokens get gradient updates
- Combines truncated IS with vanilla policy gradient
- 2x speedup vs DAPO on same benchmark

**Empirical Results**: AIME 2024 = 55.42% (Qwen3-14B-Base). Matches DAPO with 50% fewer steps.

**Known Failure Modes**:
- Training-inference mismatch increases over training, can culminate in performance collapse
- Does not clip gradient of certain tokens, causing instability
- Requires FP32 at LLM head to mitigate numerical issues
- No lower clip bound — cannot suppress incorrect response amplification as effectively

**Used By**: MiniMax-M1

---

### 1.8 DISPO (Decoupled Importance Sampling Policy Optimization)
**Origin**: Feb 2026

**Loss Function**:
```
J_DISPO(θ) = E[ (1/Σ|o_i|) Σ_i Σ_t sg(r^d_{i,t}(θ)) * A_{i,t} * log π_θ(o_{i,t} | q, o_{i,<t}) ]

where r^d_{i,t}(θ) =
  clip(r_{i,t}(θ), 1-ε+_low, 1+ε+_high)   if A_{i,t} > 0  (correct responses)
  clip(r_{i,t}(θ), 1-ε-_low, 1+ε-_high)   if A_{i,t} < 0  (incorrect responses)
```

**Key Innovations vs CISPO/DAPO**:
- Four controllable policy update regimes:
  1. Correct + ratio>1: amplified positive updates (entropy increase, exploration)
  2. Correct + ratio<1: suppressed positive updates (entropy decrease, distillation)
  3. Incorrect + ratio>1: amplified negative updates (prevents repetition collapse)
  4. Incorrect + ratio<1: suppressed negative updates (prevents excessive length reduction)
- Independent control of exploration vs distillation per regime
- Stop-gradient on clipped weights (like CISPO) but with 4-way decoupling

**Empirical Results** (Qwen3-14B-Base):

| Method   | AIME'24 | AIME'25 | AMC'23 | MATH-500 |
|----------|---------|---------|--------|----------|
| DAPO     | 50.21%  | 38.96%  | 87.66% | 91.89%   |
| CISPO    | 55.42%  | 40.83%  | 89.84% | 93.15%   |
| **DISPO**| **61.04%** | **45.83%** | **92.03%** | **94.61%** |

+10.83pp over DAPO, +5.62pp over CISPO on AIME'24.

**Known Failure Modes**:
- More hyperparameters (4 clip bounds) requiring careful tuning
- Still token-level — MoE stability not addressed

**Used By**: Research; not yet adopted by major model releases

---

### 1.9 STAPO (Spurious-Token-Aware Policy Optimization)
**Origin**: Feb 2026

**Loss Function**:
```
J_STAPO(θ) = E[ (1/Σ I^S2T) Σ I^S2T * min(ρ_{i,t}(θ) * A_i, clip(ρ_{i,t}(θ), 1-ε_low, 1+ε_high) * A_i) ]

where I^S2T_{i,t} = 0  if  A_i > 0 AND π(y_{i,t}) < τ_p AND H_t < τ_h
                   = 1  otherwise

τ_p = 0.002 (probability threshold)
τ_h = 80th percentile of entropy within mini-batch
```

**Key Innovation**: S2T (Silencing Spurious Tokens) mechanism
- ~0.01% of tokens are "spurious": low probability, low entropy, positive advantage
- These tokens inherit full sequence reward but contribute nothing to reasoning
- Produces disproportionately amplified gradients (gradient ∝ 1/probability)
- S2T masks these tokens' gradient contributions

**Empirical Results** (training-aligned evaluation):

| Model | AIME24 | AIME25 | AMC23 | MATH500 | Avg |
|-------|--------|--------|-------|---------|-----|
| 1.7B  | 17.40% | 15.42% | 55.94% | 73.55% | 38.18% |
| 8B    | 33.44% | 28.65% | 79.92% | 90.40% | 58.76% |
| 14B   | 46.98% | 35.21% | 87.11% | 92.45% | 64.38% |

Average +7.13% over GRPO, +3.69% over GRPO+entropy baselines.

**Known Failure Modes**:
- Threshold sensitivity (τ_p, τ_h)
- Narrower performance gaps under restricted decoding (temperature 0.7, top-p 0.9)
- Adds computational cost for token-level analysis

---

### 1.10 Kimi k1.5 (Online Mirror Descent with Squared Loss)
**Origin**: Moonshot AI (Jan 2025)

**Loss Function**:
```
L(θ) = E[ (r(x,y,y*) - τ log Z - τ log(π_θ(y,z|x) / π_θi(y,z|x)))² ]

Gradient:
∇_θ L ∝ ∇_θ log π_θ(y_j,z_j|x) * (r(x,y_j,y*) - r̄) - (τ/2) * ∇_θ(log(π_θ/π_θi))²
```

**Key Innovations**:
- Formulates RL as online mirror descent with relative entropy regularization
- Squared loss between log-probability ratio and scaled reward (not standard PG)
- Curriculum sampling: progressive difficulty increase
- Prioritized sampling: inversely proportional to success rate (1 - s_i)
- Length penalty: λ = 0.5 - (len - min_len)/(max_len - min_len)

**Empirical Results**: AIME 2024 = 77.5% (long-CoT), 60.8% (short-CoT). MATH-500 = 96.2%. Matches OpenAI o1.

**Known Failure Modes**:
- Squared loss formulation closely mirrors SPPO/GPO — novelty debated
- Requires careful curriculum design
- Off-policy correction via L2 regularization may be insufficient for very long training

**Used By**: Kimi k1.5, Kimi K2 (refined version)

---

### 1.11 TOPR (Tapered Off-Policy REINFORCE)
**Origin**: Mar 2025

**Key Innovation**: Asymmetric, tapered importance sampling for off-policy training
- Fully offline — can work without on-policy rollouts
- Handles positive and negative examples in unified framework
- No explicit KL penalty needed for stable behavior
- REINFORCE baseline reinterpreted as dataset balancing mechanism

**Empirical Results**: Matches 70B model performance with 8B models through dataset curation. Continuous improvement even when policy diverges substantially from data distribution.

**Used By**: Research; Stanford CS224R projects

---

### 1.12 JustRL (Simple GRPO Baseline)
**Origin**: Tsinghua (Dec 2025, ICLR 2026 Blogpost Track)

**Approach**: Standard GRPO + binary outcome rewards + rule-based verifier (no SymPy)
- Single-stage training, no curriculum, no progressive lengthening
- Fixed hyperparameters throughout training
- 4,000+ steps of stable improvement without intervention

**Empirical Results**:
- JustRL-DeepSeek-1.5B: 54.87% avg across 9 math benchmarks
- Outperforms ProRL-V2 (53.08%) despite its 9-stage pipeline
- Uses 2x less compute than sophisticated approaches

**Key Insight**: Adequate scale with simple methods can outperform sophisticated techniques. Simplicity as a feature, not a limitation.

---

### 1.13 A*-PO (Optimal Advantage Regression)
**Origin**: Harvard Kempner Institute, 2025

**Key Innovation**: Separates value estimation (offline) and policy learning (online)
- Regresses on optimal advantages rather than estimating policy-specific ones
- Eliminates need for critics AND multiple generations per prompt
- 2x faster than GRPO/PPO with comparable performance
- Smallest KL divergence from reference policy

---

## 2. COMPARISON TABLE

| Property | PPO | GRPO | REINFORCE++ | DAPO | GSPO | CISPO | DISPO | STAPO | Kimi k1.5 |
|----------|-----|------|-------------|------|------|-------|-------|-------|-----------|
| **Critic/Value Network** | Yes | No | No | No | No | No | No | No | No |
| **Models in Memory** | 4 | 2 | 1-2 | 2 | 2 | 2 | 2 | 2 | 2 |
| **IS Ratio Level** | Token | Token | N/A (no IS) | Token | **Sequence** | Token | Token | Token | N/A |
| **Clipping** | Symmetric | Symmetric | None | **Asymmetric** | Symmetric | Upper only | **4-way decoupled** | Asymmetric | N/A (squared loss) |
| **KL Penalty** | Yes | Yes | Optional | **No** | No | No | No | No | Implicit (L2) |
| **Advantage Estimation** | GAE (critic) | Group relative | Global batch | Group relative | Group relative | Group relative | Group relative | Group relative | Reward - baseline |
| **MoE Compatible** | Partial | Poor | OK | Poor | **Excellent** | Partial | Poor | Poor | Unknown |
| **Entropy Control** | Via KL | Via KL | Global norm | Clip-Higher | Clip-Higher | None explicit | 4-regime | **S2T masking** | Via τ |
| **Loss Granularity** | Token | Token | Sequence | **Token (weighted)** | **Sequence** | Token | Token | Token (masked) | Sequence |
| **Dynamic Sampling** | No | No | No | **Yes** | Yes | No | No | No | **Yes (curriculum)** |
| **Spurious Token Handling** | No | No | No | No | No | No | No | **Yes** | No |
| **Complexity** | High | Low | Very Low | Medium | Low | Low | Medium | Medium | Medium |
| **AIME'24 (best reported)** | — | 47* | — | 50* | — | 55.42** | **61.04**** | 46.98** | **77.5*** |

\* Qwen2.5-32B base  \*\* Qwen3-14B base  \*\*\* Kimi k1.5 proprietary (not directly comparable)

---

## 3. FAILURE MODE TAXONOMY

### 3.1 Entropy Collapse
**What**: Policy entropy drops sharply early in training; probability mass concentrates on limited token subset.
**Why**: Clip-high mechanism dominates under standard parameters, causing net entropy reduction. Token-level optimization amplifies high-confidence predictions.
**Who suffers**: GRPO, PPO, any method without explicit entropy control.
**Mitigations**:
- DAPO's Clip-Higher (asymmetric clipping)
- Clip-Cov / KL-Cov (covariance-based clipping)
- AEPO (entropy-balanced optimization)
- Entropy regularization with adaptive coefficient

### 3.2 Reward Hacking
**What**: Model exploits flaws in reward function without genuine task completion.
**Manifestations**: Length bias, sycophancy, sophistication bias, LLM-as-grader exploitation.
**Severity**: Can generalize to emergent misalignment (sabotage, alignment faking).
**Mitigations**: KL penalty, reward upper bounds, PAR (Preference As Reward), inoculation prompting (reduces misalignment 75-90%).

### 3.3 Spurious Token Amplification
**What**: ~0.01% of tokens (low probability, low entropy, positive advantage) inherit full sequence reward and produce disproportionate gradients.
**Why**: Gradient magnitude inversely correlates with token probability (∝ 1/π).
**Who suffers**: All token-level methods (GRPO, DAPO, CISPO, DISPO).
**Mitigation**: STAPO's S2T masking mechanism.

### 3.4 MoE Routing Instability
**What**: Expert routing changes between π_θ and π_θ_old, invalidating token-level importance sampling ratios.
**Why**: Dynamic routing creates structural probability shifts unrelated to optimization signal.
**Who suffers**: GRPO, DAPO, CISPO, DISPO — all token-level IS methods.
**Mitigation**: GSPO (sequence-level IS eliminates per-token routing dependency).

### 3.5 Training-Inference Mismatch
**What**: Policy diverges from rollout distribution during training, causing collapse.
**Who suffers**: CISPO (identified explicitly), any off-policy method without correction.
**Mitigation**: DISPO's 4-regime control, periodic reference model updates, on-policy rollouts.

### 3.6 Length Bias / Verbosity Collapse
**What**: Model learns to produce longer responses to maximize reward.
**Why**: Both PPO and GRPO inadvertently favor longer responses due to mathematical biases.
**Mitigations**: DAPO's overlong reward shaping, Kimi k1.5's length penalty, per-token normalization.

### 3.7 Sampling Inefficiency
**What**: Prompts where all G samples get identical rewards contribute zero gradient.
**Who suffers**: GRPO, RLOO (any group-based method).
**Mitigation**: DAPO's dynamic sampling (filter all-correct / all-incorrect prompts).

### 3.8 KL Estimator Errors
**What**: Prevailing KL regularization implementations provide incorrect gradients for stated objectives.
**Why**: KL divergence is intractable to compute exactly; various estimators have subtle bugs.
**Scope**: Widespread — affects most PPO/GRPO implementations with KL penalties.
**Mitigation**: RPG framework (2025) provides correct forward and reverse KL gradient derivations.

---

## 4. EVOLUTION TREE

```
REINFORCE (Williams, 1992)
│
├─── Actor-Critic Methods
│    └─── PPO (Schulman, 2017)
│         ├─── PPO for RLHF (Ouyang/InstructGPT, 2022)
│         │    └─── DPPO (Feb 2026) — replaces clipping with divergence constraint
│         │
│         └─── [Remove Critic] ─────────────────────────────────┐
│                                                                │
├─── Critic-Free REINFORCE Variants ◄───────────────────────────┘
│    ├─── ReMax (greedy baseline)
│    ├─── RLOO (leave-one-out baseline)
│    ├─── REINFORCE++ (global advantage normalization, Jan 2025)
│    └─── TOPR (tapered off-policy, Mar 2025)
│
├─── Group-Based Methods
│    └─── GRPO (DeepSeek, Feb 2024) — group relative advantage
│         │
│         ├─── DAPO (ByteDance, Mar 2025)
│         │    ├── + Clip-Higher (asymmetric clipping)
│         │    ├── + Dynamic Sampling
│         │    ├── + Token-Level Loss normalization
│         │    └── + Overlong Reward Shaping
│         │
│         ├─── GSPO (Qwen/Alibaba, Jul 2025)
│         │    └── Sequence-level IS ratio (solves MoE instability)
│         │
│         ├─── CISPO (MiniMax, Jun 2025)
│         │    ├── Clips IS weights (not surrogate objective)
│         │    └── Stop-gradient on clipped weights
│         │    │
│         │    └─── DISPO (Feb 2026)
│         │         └── 4-way decoupled clipping per regime
│         │
│         ├─── STAPO (Feb 2026)
│         │    └── + S2T spurious token masking
│         │
│         ├─── JustRL (Dec 2025) — proves vanilla GRPO + scale suffices
│         │
│         └─── TR-GRPO (Token-Regulated GRPO)
│              └── Token-level weights correlated with predicted probability
│
├─── Regression-Based Methods
│    ├─── REBEL (regressing relative rewards)
│    ├─── A*-PO (optimal advantage regression, 2025)
│    └─── Kimi k1.5 (online mirror descent + squared loss, Jan 2025)
│         └─── Kimi K2 (refined, Jul 2025)
│
└─── Entropy-Focused Methods
     ├─── EPO (entropy-regularized, Sep 2025)
     └─── AEPO (entropy-balanced, Oct 2025)
```

**Major Evolutionary Pressures**:
1. **Remove critic** (PPO → GRPO/REINFORCE++): Memory efficiency
2. **Fix clipping** (GRPO → DAPO → DISPO): Entropy management
3. **Token → Sequence** (GRPO → GSPO): MoE compatibility
4. **Handle edge cases** (GRPO → STAPO): Spurious token stability
5. **Simplify** (all → JustRL): Scale > sophistication

---

## 5. OPEN PROBLEMS

### 5.1 Does RL Actually Teach New Reasoning?
Current RLVR does not elicit fundamentally new reasoning patterns — it primarily amplifies capabilities already present in the base model. RL-trained models excel at pass@1 but are consistently outperformed by base models at pass@256, demonstrating that RLVR narrows exploration rather than expanding it. This is perhaps the most fundamental open question.

### 5.2 Reward Model Limitations
- Static reward models ignore task difficulty, limiting optimization efficiency
- Imbalanced safety datasets overrepresent common hazards, neglect long-tail threats
- Binary 0/1 verifiable rewards are effective but fundamentally limited in scope
- Process reward models (PRMs) promising but expensive to annotate
- Relationship between reward design and hallucination/distribution shift poorly understood

### 5.3 Scaling RL Compute
How to scale RL training effectively remains unclear. JustRL shows simple methods scale well, but the optimal compute allocation between pretraining, SFT, and RL is not established. Improved paradigms such as continual scaling and multi-turn agent-environment interaction may be needed.

### 5.4 MoE + RL Stability
Only GSPO has directly addressed MoE routing instability. As MoE becomes standard architecture, all token-level IS methods face fundamental challenges. No principled theory exists for how expert routing dynamics interact with policy gradient optimization.

### 5.5 Entropy Management
No consensus on optimal entropy trajectory during training. Current approaches (Clip-Higher, entropy regularization, S2T) are ad-hoc. A principled framework connecting entropy dynamics to final model capability is missing.

### 5.6 Credit Assignment in Long-CoT
Sequence-level rewards create credit assignment challenges for long chain-of-thought. Token-level rewards provide better signals but introduce variance. The optimal granularity for reward/optimization (token vs step vs sequence) remains debated.

### 5.7 Off-Policy vs On-Policy
TOPR shows off-policy can work well; most methods assume on-policy. The optimal balance between sample efficiency (off-policy) and stability (on-policy) for LLM RL is not established.

### 5.8 Generalization of RL Gains
RL improvements on math/code benchmarks do not reliably transfer to other domains. Whether RL-trained reasoning generalizes to novel problem types is unclear.

### 5.9 Multi-Turn / Agentic RL
Extending RL beyond single-turn generation to multi-turn agent interactions (tool use, environment feedback) introduces compounding errors, sparse delayed rewards, and exploration challenges in exponentially larger action spaces.

### 5.10 Theoretical Foundations
The field lacks rigorous theory for:
- Why critic-free methods work as well as they do
- Optimal clipping strategies (token vs sequence, symmetric vs asymmetric)
- How different KL estimators affect convergence guarantees
- Whether the REINFORCE / GRPO / PPO family is fundamentally optimal or if entirely different algorithmic paradigms could dominate

---

## 6. RECOMMENDATIONS FOR NanoSeek

Given NanoSeek's architecture (MoE with 64 experts, top-8 routing):

1. **Primary Algorithm**: GSPO — designed for MoE, eliminates Routing Replay, sequence-level IS
2. **Fallback**: DAPO — well-understood, open-source, strong baselines
3. **Simplicity Baseline**: JustRL (vanilla GRPO + binary rewards) — proves scale > tricks at 1.5B
4. **Avoid**: CISPO (instability at scale), vanilla PPO (memory overhead)
5. **Monitor**: DISPO for future adoption once MoE-specific variant emerges
6. **Key Insight**: At NanoSeek's scale (~1B active), JustRL's finding that simple methods + adequate compute outperform sophisticated pipelines is directly relevant

---

## Sources

### Algorithm Papers
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- [DAPO](https://arxiv.org/abs/2503.14476) | [GitHub](https://github.com/BytedTsinghua-SIA/DAPO)
- [GSPO](https://arxiv.org/abs/2507.18071) | [Blog](https://qwenlm.github.io/blog/gspo/)
- [CISPO / MiniMax-M1](https://arxiv.org/abs/2506.13585)
- [DISPO](https://arxiv.org/html/2602.00983)
- [STAPO](https://arxiv.org/abs/2602.15620)
- [REINFORCE++](https://arxiv.org/abs/2501.03262)
- [Kimi k1.5](https://arxiv.org/abs/2501.12599)
- [Kimi K2](https://arxiv.org/abs/2507.20534)
- [JustRL](https://arxiv.org/abs/2512.16649) | [ICLR 2026 Blog](https://iclr-blogposts.github.io/2026/blog/2026/justrl/)
- [TOPR](https://arxiv.org/abs/2503.14286)
- [A*-PO](https://openreview.net/forum?id=T1V8BJO0iG)

### Analysis & Surveys
- [From GRPO to DAPO and GSPO (HuggingFace)](https://huggingface.co/blog/NormalUhr/grpo-to-dapo-and-gspo)
- [State of RL for LLM Reasoning (Raschka)](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training)
- [From REINFORCE to Dr. GRPO](https://lancelqf.github.io/note/llm_post_training/)
- [RLHF Book - Policy Gradients](https://rlhfbook.com/c/06-policy-gradients)
- [From PPO to GRPO to DAPO (SoftmaxData)](https://softmaxdata.com/blog/from-ppo-to-grpo-to-dapo-understanding-rl-for-llms-and-every-training-parameter-explained/)

### Failure Modes & Stability
- [Entropy Mechanism of RL for Reasoning LLMs](https://arxiv.org/abs/2505.22617)
- [KL Regularization Comedy of Estimators](https://arxiv.org/abs/2512.21852)
- [Rethinking Trust Region in LLM RL](https://arxiv.org/abs/2602.04879)
- [Clip-Low Increases Entropy, Clip-High Decreases](https://openreview.net/forum?id=2ZflH67Uof)
- [Reward Hacking (Lil'Log)](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)
- [Natural Emergent Misalignment from Reward Hacking (Anthropic)](https://assets.anthropic.com/m/74342f2c96095771/original/Natural-emergent-misalignment-from-reward-hacking-paper.pdf)
- [KL-Regularized RL Designed to Mode Collapse](https://arxiv.org/html/2510.20817v1)
- [RPG: KL-Regularized Policy Gradient Design](https://arxiv.org/abs/2505.17508)

### Open Problems
- [Does RL Really Incentivize Reasoning Beyond Base Model?](https://arxiv.org/abs/2504.13837)
- [Limit of RLVR](https://limit-of-rlvr.github.io/)
- [Reward Modeling Challenges Survey](https://arxiv.org/html/2602.09305v1)
- [Awesome RL for Large Reasoning Models](https://github.com/TsinghuaC3I/Awesome-RL-for-LRMs)
