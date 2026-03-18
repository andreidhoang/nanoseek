# Post-Training Techniques: State of the Art (2025-2026)

## Comprehensive Research Summary for NanoSeek

---

## 1. Beyond RLHF/DPO/GRPO: The Cutting Edge

### 1.1 GRPO Variants and Successors

**GRPO (Group Relative Policy Optimization)** remains the dominant RL algorithm for reasoning post-training. It eliminates the critic model by sampling 8-64 responses per prompt and computing advantages as `(reward_i - mean) / std`. Theoretical analysis shows GRPO's policy gradient is a U-statistic, making it asymptotically equivalent to an oracle algorithm.

**DAPO (Dynamic Attention Policy Optimization)** — ByteDance/Tsinghua, 2025:
- Tackles instabilities in long chain-of-thought training
- Key innovations: Clip-Higher (prevents entropy collapse), Dynamic Sampling (filters zero-variance prompts), Token-level Policy Gradient Loss (handles varying response lengths), Overlong Reward Shaping
- Removes KL divergence term entirely for RL training
- Result: Trained Qwen2.5-32B to 50 points on AIME 2024, outperforming DeepSeek-R1-Zero with **50% fewer training steps**

**Dr.GRPO** — Removes response length normalization and question-level standard deviation from advantage computation, resulting in more efficient training and fewer unnecessarily long answers.

**DeepSeek V3.2 GRPO Modifications:**
- Domain-specific KL weights (zero KL for math, tuned per domain)
- Unbiased KL estimation: reweights KL term with importance ratio to fix systematic gradient errors
- Masks highly off-policy negative trajectories
- Freezes MoE routing paths from sampling to training
- Preserves top-p/top-k truncation masks during sampling and applies them during training

### 1.2 RLVR (Reinforcement Learning with Verifiable Rewards)

The most consequential shift in 2025: moving from human preference labels to **verifiable rewards** for reasoning tasks.

- Uses programmatic verification (math correctness, code execution) instead of learned reward models
- DeepSeek-R1 demonstrated pure RLVR produces **emergent reasoning capabilities** (self-reflection, dynamic strategy adaptation) without human-labeled reasoning traces
- Expanding beyond math/code into chemistry, biology, and other verifiable domains
- **RISE**: Adds self-verification training within the same RL process

### 1.3 Self-Play Preference Optimization (SPPO)

- Two-player constant-sum game for LLM alignment
- Iterative policy updates provably approximate Nash equilibrium
- Using only 60k prompts + PairRM (0.4B), fine-tuned Mistral-7B to **28.53% length-controlled win rate against GPT-4-Turbo** on AlpacaEval 2.0
- No external supervision from GPT-4 needed
- Outperforms iterative DPO/IPO on MT-Bench, Arena-Hard, Open LLM Leaderboard

### 1.4 Reference-Free Preference Optimization

**SimPO** (Simple Preference Optimization):
- No reference model needed — uses average log probability as implicit reward
- Outperforms DPO by **6.4 points on AlpacaEval 2**, **7.5 points on Arena-Hard**
- Top model (Gemma-2-9B-it) achieves 72.4% LC win rate on AlpacaEval 2

**ORPO** (Odds-Ratio Preference Optimization):
- Monolithic — merges SFT and preference optimization into single objective
- Eliminates need for separate alignment phase

**KTO** (Kahneman-Tversky Optimization):
- Works with binary feedback (thumbs up/down) instead of preference pairs
- Models human asymmetry: pain of bad answer > pleasure of good one
- Based on prospect theory

### 1.5 Iterative/Online DPO Advances

- **DICE (Bootstrapping with Implicit Rewards)**: >8% LC win rate increase on AlpacaEval 2
- **APO (Accelerated Preference Optimization)**: Uses Nesterov's momentum for faster convergence
- **COPO (Count-based Exploration PO)**: Balances exploration with preference optimization

### 1.6 Constitutional AI Evolution (Anthropic, Jan 2026)

- Shifted from rule-based to **reason-based** alignment — explains logic behind principles
- 4-tier priority hierarchy: safety > ethics > compliance > helpfulness
- First major AI company to formally acknowledge possibility of AI consciousness/moral status
- Claude itself uses constitution to generate synthetic training data for self-alignment
- Two-phase training: supervised learning (self-critique + revision) then RL (RLAIF)

### 1.7 The Modern Post-Training Stack (2026 Consensus)

```
SFT (instruction following) → Preference Optimization (DPO/SimPO/KTO) → RLVR (GRPO/DAPO for reasoning)
```

---

## 2. Reward Model Innovations

### 2.1 Process Reward Models (PRM) — Latest Results

**R-PRM (Reasoning-Driven PRM)**:
- Surpasses baselines by **11.9 F1 on ProcessBench**, **8.5 F1 on PRMBench**
- Achieves **>8.5 point accuracy improvements** across six challenging math datasets

**ThinkPRM (Process Reward Models That Think)**:
- Outperforms LLM-as-a-Judge and discriminative verifiers
- Uses only **1% of process labels** in PRM800K
- Beats baselines on ProcessBench, MATH-500, AIME '24

**GM-PRM (Generative Multimodal PRM)**:
- State-of-the-art on multimodal math benchmarks
- Remarkable data efficiency: only **20K training samples** needed

### 2.2 ORM vs PRM

- ORMs provide single final-answer feedback → poor credit assignment for multi-step reasoning
- PRMs provide per-step feedback → better for math, code, and complex reasoning
- Key finding: Monte Carlo estimation-based data synthesis for PRMs yields **inferior** performance vs. LLM-as-a-judge and human annotation
- Hierarchical models like **PathFinder-PRM** decompose errors into math vs. consistency dimensions

### 2.3 Generative Reward Models (GenRM)

- Iterative algorithm training LLM on self-generated reasoning traces for preference labels
- Hybrid RLHF + RLAIF: synthetic labels matching human preference judgments
- Advantages over discriminative RMs: chain-of-thought reasoning, test-time compute via majority voting
- **P-GenRM**: First personalized GenRM with test-time user-based scaling — derives adaptive personas and rubrics

### 2.4 Self-Rewarding and Reward-Free Approaches

**Self-Rewarding Language Models**: Reward model continually improves alongside the policy, avoiding frozen-RM bottleneck

**Latent Reward Discovery**: Any next-token trained LLM implicitly encodes a latent reward function equivalent to offline inverse RL — no additional preference learning needed

**Reward Reasoning Models (RRMs)**: Execute deliberate reasoning (chain-of-thought) before generating rewards, leveraging test-time compute for complex queries

**Self-Verification Approaches**:
- **Intuitor**: Matches/exceeds RLVR models; LiveCodeBench pass@1: 0.153 vs GRPO 0.085
- **LaSeR**: Self-verification F1 improves from 49.2% to 79.6% (Qwen2.5-7B)
- **ReVeal**: Dense per-turn self-verification; Pass@1 of 42.4% at 19 turns

### 2.5 Automated Reward Design

**LEARN-Opt**: Orchestrates LLM agents to autonomously generate, execute, and refine reward code — no human reward engineering needed

---

## 3. Scaling RL Post-Training

### 3.1 Compute Scaling

- **OpenAI o3**: Used **10x more training compute** than o1 for RL
- OpenAI observed that large-scale RL exhibits the same "more compute = better performance" trend as pretraining
- "The Art of Scaling Reinforcement Learning Compute for LLMs" (Khatri & Madaan, 2025): proposes methods to extrapolate RL learning curves over compute scales

### 3.2 SFT vs RL: Key Finding

**"SFT Memorizes, RL Generalizes"** (ICML 2025):
- SFT tends to **memorize** training data, struggles OOD
- RL (especially with outcome-based reward) **generalizes** across rule-based textual and visual variants
- **Critical**: SFT remains essential — it stabilizes output format, enabling RL to achieve its gains
- RL improves underlying recognition capabilities

### 3.3 Multi-Stage RL Pipelines

**DeepSeek-R1 (4 Stages)**:
1. Cold-Start SFT: Thousands of structured CoT examples
2. Reasoning-Oriented RL: GRPO with language consistency + accuracy rewards
3. Rejection Sampling + SFT: Generate many samples, keep correct/readable ones using DeepSeek-V3 as judge
4. Diverse RL: Rule-based rewards for math; LLM feedback for other tasks

**Qwen3 (4 Stages)**:
1. Long CoT cold start
2. Reasoning-based RL (GRPO on verifiable problems)
3. Thinking mode fusion
4. General RL (instruction following, agent abilities, preference alignment)

**Llama 4 (3 Stages)**:
1. Lightweight SFT (removed >50% easy data using Llama-as-judge)
2. Continuous online RL with adaptive data filtering (medium-to-hard difficulty)
3. Lightweight DPO
- Key insight: SFT and DPO can **over-constrain** the model, restricting RL exploration

**Nemotron 340B**: 2 SFT rounds + 4 alignment rounds using RPO (reward model-weighted DPO)

**Llama 3.1**: 6 rounds of preference training

### 3.4 Practical Ratios and Numbers

- **Qwen3-4B-Instruct**: 26 steps SFT (batch 34) + 120 steps RL (group 16, batch 512) → mean reward 0.7
- **Nemotron 3 Super**: 7M samples from 40M corpus for SFT
- **SmolLM3**: ~100k examples totaling 76.1M tokens for post-training
- **Meta**: Spent **$10-20M+** on preference annotations for final models
- Modern approach: "100% of alignment budgets on preference data rather than instruction demonstrations"
- No purely sequential SFT→RL scheme can preserve optimality of both objectives — they are irreversibly coupled

### 3.5 Curriculum Learning in RL Post-Training

**Prompt Curriculum Learning (PCL)**: Selects intermediate-difficulty prompts (model has ~50% success chance) — most sample-efficient for convergence

**E2H Reasoner (Easy-to-Hard)**: Schedules tasks from easy to hard; significantly improves small LLMs (1.5B-3B) that otherwise struggle with vanilla RL

**Mixed findings**: Some research shows random sampling performs competitively with curriculum — optimal schedule varies across datasets/models

### 3.6 Infrastructure at Scale

**Meta (Llama 4)**: Fully asynchronous online RL framework → **~10x training efficiency improvement** over previous generations

**RLFactory**: 6.8x throughput improvement for Qwen3-4B

**NeMo Gym**: Generated 1.2 million rollouts across 21 environment configurations

---

## 4. Synthetic Data for Post-Training

### 4.1 Self-Play and Self-Improvement Loops

**SPICE (Self-Play In Corpus Environments)** — Meta AI, 2025:
- Challenger mines documents from corpus to generate tasks; Reasoner solves them
- Corpus grounding prevents hallucination amplification and model collapse
- Results: **+8.9% mathematical reasoning**, **+9.8% general reasoning**
- Key insight: self-improvement requires external grounding for sustained improvement

**SPIN**: Models distinguish own outputs from human-written responses

### 4.2 Rejection Sampling at Scale

- Llama 3/4: Core loop of rejection sampling → SFT → DPO
- DeepSeek-R1 Stage 3: Generate many samples, keep correct/readable ones via generative reward model
- **RAFT**: Rejection sampling using only correct outputs matches or surpasses complex RL on math reasoning
- **Reinforce-Rej**: Filters out both flawed AND flawless prompts — challenges belief that negative samples are essential

### 4.3 Synthetic Data Limits

- Pure synthetic data is NOT superior to CommonCrawl
- Mixing synthetic + CommonCrawl substantially improves over either alone
- OpenAI reportedly training next-gen on **50 trillion tokens** of largely synthetic data
- Model collapse risk: without external grounding, models plateau or collapse

### 4.4 Distillation Pipelines

**On-Policy Distillation (OPD)**: Student generates trajectories, teacher grades each token — better than training on teacher-generated data

**Off-Policy Distillation**: Train student on teacher-generated trajectories (simpler but less effective)

**Key finding from Qwen3**: On-policy distillation outperforms RL on math/programming with **one-tenth GPU computation time** — for smaller models, distillation > RL

**Extending to RL**: Non-trivial because RL continually shifts the rollout distribution; methods like "Reinforcement-aware Knowledge Distillation" address this

**Multi-Teacher**: TinyLLM, TwT explore knowledge purification across multiple teachers

### 4.5 Evol-Instruct and Instruction Evolution

- Microsoft's Evol-Instruct: Rewrites seed examples to be more complex, domain-specific, or step-by-step
- WizardCoder: Adapts Evol-Instruct for code instruction complexity
- Red Hat's InstructLab (with IBM Research): Automated instruction-tuning expansion
- **DataForge** (Hermes 4): Graph-based synthetic data generator — nodes implement struct→struct mapping via DAGs

---

## 5. MoE-Specific Post-Training

### 5.1 Expert Specialization Under Post-Training

**Problem**: Standard auxiliary load balancing loss leads to expert overlap and overly uniform routing → hinders specialization, degrades performance during post-training.

**Solution** (ICLR 2026):
1. **Orthogonality loss**: Encourages experts to process distinct token types
2. **Variance loss**: Encourages more discriminative routing decisions

### 5.2 Router Behavior Under RL Gradients

**DeepSeek V3/V3.2 approach**: During RL training, record which MoE experts were chosen during rollouts, then **force the same routing pattern** during training so gradients update exactly the experts that produced the sampled answers. This prevents expert routing from "whipsawing" between frameworks.

### 5.3 Load Balancing Approaches

**Loss-based**: Auxiliary loss `alpha * sum(f_i * P_i)` where f_i is token fraction and P_i is routing probability

**Auxiliary-loss-free (DeepSeek V3)**: Bias vector `b_i <- b_i + gamma * sign(n_bar - n_i)` — no gradient-based load balancing needed, allows more natural specialization

**SMEBU**: Sequence-wise MoE Balancing with Uniformity — momentum buffers with tanh-applied violation metrics

**Similarity-preserving objective** (mid-2025): Stabilizes expert selection for related inputs while avoiding expert collapse

### 5.4 DeepSeek V3 MoE Post-Training Specifics

- 256 fine-grained experts + shared experts
- GRPO for RL (no critic model)
- Auxiliary-loss-free load balancing via bias terms
- Shared experts learn common knowledge; routed experts specialize
- RL routing frozen from sampling to training

### 5.5 Qwen3 MoE Post-Training

- 128 experts, 8 active per token, no shared experts
- Fine-grained expert segmentation + global batch load balancing loss
- 4-stage pipeline: CoT cold start → reasoning RL → thinking mode fusion → general RL
- MoE models achieve similar performance to dense models using **10% active parameters**
- For smaller models: distillation outperforms RL

### 5.6 Llama 4 MoE Post-Training

- 128 routed experts + 1 shared expert; 1 routed expert active per token
- Alternating dense and MoE layers
- Lightweight SFT → online RL → lightweight DPO
- Fully asynchronous RL framework for 2T parameter model

### 5.7 Kimi K2 MoE

- 384 total experts, 8 active per token (highest sparsity)
- Decreased attention heads from 128 to 64 → **45% decrease in inference FLOPs** with 0.5-1.2% validation loss trade-off
- **MuonClip**: Clips attention projection weights when max logits exceed threshold
- PTX loss (pre-training cross-entropy) during RL to prevent catastrophic forgetting

---

## 6. Multimodal Post-Training

### 6.1 Current State

- Vision/audio RL is becoming standard for frontier models but remains less mature than text-only RL
- **Unified "omni-modal" models** (ICLR 2026 focus): SAM 3, Mamba-3 aligning vision, audio, video into single latent space
- **GM-PRM**: Generative multimodal process reward model for multimodal math reasoning

### 6.2 Multimodal RL Techniques

- **Dense token-level rewards** for unified multimodal LLMs using GRPO framework
- **ARM-Thinker**: Turns reward models into agents that call tools, crop images, fetch pages to justify scores
- **retrainZero**: Extends RL from post-training into pretraining stage

### 6.3 Specific Models

- **Gemma 3**: Post-training optimized using distillation + RL + model merging
- **Phi-4 Multimodal** (Microsoft): Unified vision/audio/text processing
- **Llama 4**: Natively multimodal with early fusion architecture
- **VL-Cogito**: Progressive curriculum RL for vision-language models

### 6.4 Implications for Text-Only Models

- Techniques developed for multimodal (dense rewards, progressive curriculum) transfer back to text-only
- Multimodal verification (image-grounded rewards) provides richer training signals
- Tool-use RL (ARM-Thinker pattern) applicable to text-only agentic models

---

## 7. Practical Implications for NanoSeek (3B-7B MoE)

### Recommended Post-Training Pipeline

Based on frontier lab convergence:

```
Stage 1: Cold-Start SFT
  - Curated CoT examples (thousands)
  - Focus on output format stability
  - Light — don't over-constrain

Stage 2: Reasoning RL (GRPO or DAPO)
  - Verifiable rewards for math/code
  - Domain-specific KL weights (zero for math)
  - Consider Dr.GRPO modifications for efficiency
  - Freeze MoE routing paths from sampling to training

Stage 3: Rejection Sampling + SFT
  - Generate many samples from Stage 2 checkpoint
  - Filter with correctness + readability criteria
  - Use stronger model as judge if available

Stage 4: General Alignment
  - DPO/SimPO for preference alignment (lightweight)
  - Rule-based rewards where possible
  - LLM-as-judge for open-ended tasks
```

### Key Decisions for MoE

1. **Load balancing**: Consider auxiliary-loss-free approach (DeepSeek-style bias terms) for better expert specialization
2. **RL routing**: Freeze expert routing from sampling to training during GRPO
3. **Orthogonality loss**: Add to encourage distinct expert specialization during post-training
4. **Distillation vs RL**: For smaller MoE models, on-policy distillation may beat RL (Qwen3 finding) with 10x less compute

### Critical Numbers to Remember

| Metric | Value | Source |
|--------|-------|--------|
| DAPO vs R1-Zero | 50% fewer steps, same AIME score | ByteDance 2025 |
| SimPO vs DPO | +6.4 pts AlpacaEval 2 | Princeton 2024 |
| SPICE self-play | +8.9% math, +9.8% general | Meta AI 2025 |
| Distillation vs RL | 10x less compute for small models | Qwen3 2025 |
| On-policy distill | Better than teacher-generated data | Multiple 2025 |
| ThinkPRM | Beats baselines with 1% labels | 2025 |
| Kimi K2 sparsity | 45% inference FLOP reduction | Moonshot 2025 |
| Meta RL infra | 10x efficiency improvement | Llama 4, 2025 |
| OpenAI o3 RL | 10x more compute than o1 | OpenAI 2025 |

---

## Sources

### Papers and Technical Reports
- [SPPO: Self-Play Preference Optimization](https://arxiv.org/abs/2405.00675)
- [SimPO: Simple Preference Optimization](https://arxiv.org/abs/2405.14734)
- [ORPO: Monolithic Preference Optimization](https://arxiv.org/abs/2403.07691)
- [DeepSeek-R1 Technical Report](https://arxiv.org/pdf/2501.12948)
- [DeepSeek-V3.2 Technical Report](https://arxiv.org/html/2512.02556v1)
- [SFT Memorizes, RL Generalizes](https://arxiv.org/abs/2501.17161)
- [R-PRM: Reasoning-Driven Process Reward Modeling](https://arxiv.org/abs/2503.21295)
- [ThinkPRM: Process Reward Models That Think](https://openreview.net/forum?id=V727xqBYIW)
- [GM-PRM: Generative Multimodal PRM](https://arxiv.org/abs/2508.04088v1)
- [Generative Reward Models](https://arxiv.org/abs/2410.12832)
- [Generative Verifiers](https://arxiv.org/abs/2408.15240)
- [SPICE: Self-Play In Corpus Environments](https://arxiv.org/abs/2510.24684)
- [Prompt Curriculum Learning for LLM Post-Training](https://arxiv.org/abs/2510.01135)
- [E2H: Curriculum RL from Easy to Hard](https://arxiv.org/abs/2506.06632)
- [Advancing Expert Specialization for Better MoE](https://openreview.net/forum?id=iydmH9boLb)
- [Lessons of Developing PRMs in Mathematical Reasoning](https://arxiv.org/abs/2501.07301)
- [Survey of Process Reward Models](https://arxiv.org/pdf/2510.08049)
- [REBEL and Direct Nash Optimization](https://arxiv.org/abs/2405.00675)
- [Bootstrapping with Implicit Rewards (DICE)](https://proceedings.iclr.cc/paper_files/paper/2025/file/8c4de96b9169aa869cc102afe31055e8-Paper-Conference.pdf)
- [Accelerated Preference Optimization](https://openreview.net/forum?id=TROUDY6Wg4)
- [Reward Reasoning Models](https://openreview.net/forum?id=V8Kbz7l2cr)
- [MOSAIC](https://llm-stats.com/blog/research/post-training-techniques-2026)

### Blog Posts and Industry Analysis
- [Post-Training in 2026: GRPO, DAPO, RLVR & Beyond](https://llm-stats.com/blog/research/post-training-techniques-2026)
- [Frontier Model Training Methodologies (Jan 2026)](https://djdumpling.github.io/2026/01/31/frontier_training.html)
- [A Recipe for Frontier Model Post-Training — Nathan Lambert](https://www.interconnects.ai/p/frontier-model-post-training)
- [The State of Post-Training 2025 — Nathan Lambert](https://www.interconnects.ai/p/the-state-of-post-training-2025)
- [How to Scale RL — Nathan Lambert](https://www.interconnects.ai/p/the-new-rl-scaling-laws)
- [The State of LLM Reasoning Model Training — Raschka](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training)
- [Technical Tour of DeepSeek V3 to V3.2 — Raschka](https://magazine.sebastianraschka.com/p/technical-deepseek)
- [State of LLMs 2025 — Raschka](https://magazine.sebastianraschka.com/p/state-of-llms-2025)
- [Llama 4: Challenges of Frontier LLM — Wolfe](https://cameronrwolfe.substack.com/p/llama-4)
- [Reward Models — Wolfe](https://cameronrwolfe.substack.com/p/reward-models)
- [OpenAI o3 Over-optimization — Lambert](https://www.interconnects.ai/p/openais-o3-over-optimization-is-back)

### Official Announcements
- [Anthropic: Claude's New Constitution (Jan 2026)](https://www.anthropic.com/news/claude-new-constitution)
- [Qwen3 Blog Post](https://qwenlm.github.io/blog/qwen3/)
- [Llama 4 Blog — Meta AI](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- [OpenAI o3 and o4-mini](https://openai.com/index/introducing-o3-and-o4-mini/)
- [OpenAI: Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/)
- [Complete Guide to DeepSeek Models](https://www.bentoml.com/blog/the-complete-guide-to-deepseek-models-from-v3-to-r1-and-beyond)
- [MoE Architectures 2024-2025 Literature Review](https://www.rohan-paul.com/p/mixture-of-experts-moe-architectures)
