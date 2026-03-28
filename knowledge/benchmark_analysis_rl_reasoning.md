# Benchmark Analysis for RL Reasoning Training

**Agent 7: Evaluation and Benchmark Scientist**
**Date: 2026-03-24**

---

## A. Benchmark Taxonomy for RL Relevance

### Tier 1: Directly RL-Relevant (Verifiable, Reasoning-Heavy)

These benchmarks have **binary-verifiable answers**, making them ideal as RL reward signals.

| Benchmark | What It Measures | RL Relevance | Notes |
|-----------|-----------------|--------------|-------|
| **AIME 2024/2025** | Competition math (30 problems) | **Critical** | Gold standard for reasoning. High variance at small k due to few problems. Evaluate with Avg@32 for stability. |
| **MATH-500** | 500 competition math problems (Hendrycks) | **Critical** | Most reliable math benchmark. Large enough for statistical significance. Primary RL progress metric. |
| **GSM8K** | Grade school math (8.5K problems) | **High** | Good for early training signal. Risk: saturates quickly at 1.5B+ scale with RL. Contamination concerns. |
| **LiveCodeBench** | Code generation (fresh problems) | **High** | Low contamination risk (rolling updates). Tests reasoning transfer to code. |
| **HumanEval / HumanEval+** | Function-level code synthesis | **Moderate-High** | Verifiable via test execution. But only 164 problems; noisy signal. |
| **MBPP+** | More basic Python programming | **Moderate** | Complementary to HumanEval. |
| **Codeforces** | Competitive programming rating | **High** | Excellent reasoning transfer test. Hard to game. |
| **SWE-bench Verified** | Real GitHub issue resolution | **Moderate** | Tests practical coding, less pure reasoning. Expensive to evaluate. |

### Tier 2: Reasoning Indicators (Useful But Not Directly Verifiable)

These require expert judgment or multiple-choice format, making them harder to use as direct RL rewards.

| Benchmark | What It Measures | RL Relevance | Notes |
|-----------|-----------------|--------------|-------|
| **GPQA Diamond** | PhD-level science questions | **High** | 198 expert-written questions. Tests deep scientific reasoning. Hard to game. Best "ceiling" indicator. |
| **ARC-Challenge** | Abstract reasoning (science) | **Moderate** | Tests systematic reasoning. Some saturation at larger scales. |
| **BBH (Big Bench Hard)** | 23 hard reasoning tasks | **Moderate** | Diverse reasoning. Good for detecting broad capability regression. |
| **MMLU-Pro** | Hard multi-domain knowledge + reasoning | **Moderate** | Improvement over MMLU. 10 answer choices reduces guessing. |

### Tier 3: General Capability (Important But Less RL-Relevant)

| Benchmark | What It Measures | RL Relevance | Notes |
|-----------|-----------------|--------------|-------|
| **MMLU** | Multi-domain knowledge (57 subjects) | **Low** | Widely saturated. Significant contamination evidence. Not reasoning-heavy. |
| **HellaSwag** | Commonsense NLI | **Very Low** | Saturated above 3B parameters. |
| **WinoGrande** | Coreference resolution | **Very Low** | Saturated. |
| **IFEval** | Instruction following | **Moderate** | Important for usability, but not reasoning. Track to detect RL-induced regression. |
| **MT-Bench / Arena-Hard** | Conversation quality | **Low** | Subjective. Important post-RL to check fluency hasn't degraded. |

### Tier 4: Potentially Misleading for RL

| Benchmark | Why It's Misleading |
|-----------|-------------------|
| **GSM8K** (at scale) | Saturates quickly with RL training. 90%+ scores achievable at 1.5B with distillation, creating illusion of deep reasoning. A 2023 study showed accuracy drops of **up to 13%** when tested on similar but unseen benchmarks, indicating systematic overfitting. |
| **MMLU** | Widely saturated. In 2022, 540B parameters needed for 60%; by 2024, 3.8B (Phi-3-mini) achieved same. Evidence of "data leakage" across many models. Score differences above 80% are often statistically meaningless. |
| **HellaSwag** | Fully saturated at small scales. Near-perfect scores even at 3B. No discriminative value. |
| **WinoGrande** | Same saturation problem. |
| **TruthfulQA** | Highly gameable through output formatting. Doesn't correlate with actual reasoning improvement. |
| **Benchmarks with <50 questions** (e.g., AIME alone) | High variance makes single-run scores unreliable. Must use Avg@32 or similar. |

**Key insight**: RL-trained models show significant performance drops on **newer versions** of benchmarks (e.g., AIME 2025 vs AIME 2024), suggesting overfitting to problem distributions rather than genuine reasoning gains. Always evaluate on held-out or temporally-separated test sets.

---

## B. Candidate Model Benchmark Comparison

### Small Models (1B-3B)

| Model | MATH-500 | GSM8K | HumanEval | MMLU | GPQA-D | ARC-C | AIME'24 | Notes |
|-------|----------|-------|-----------|------|--------|-------|---------|-------|
| **Qwen2.5-1.5B-Instruct** | ~55.2* | 73.2 | 61.6 | 58.4 | -- | -- | -- | Strong math/code for size |
| **Qwen3-1.7B-Base** | -- | -- | -- | 75.7 | -- | -- | -- | Matches Qwen2.5-3B on most tasks |
| **Qwen3-4B (Thinking)** | -- | -- | -- | -- | 65.8 | -- | 73.8 | Outstanding reasoning w/ thinking mode |
| **Gemma 3 1B** | -- | 62.8 | 41.5 | 38.8 | -- | -- | -- | Limited reasoning at 1B |
| **Gemma 3 4B** | -- | -- | 71.3 | 58.1 | -- | -- | -- | Good code capability |
| **Phi-4-mini (3.8B)** | 92.5** | 88.6 | 74.4 | 67.3 | -- | 83.7 | -- | **MATH-500 score is for reasoning variant** |
| **SmolLM2-1.7B** | -- | 51.6 | -- | -- | -- | 57.1 | -- | Lower tier; 11T token training |
| **DeepSeek-R1-Distill-Qwen-1.5B** | **83.9** | -- | -- | -- | 33.8 | -- | **28.9** | Reasoning distilled from R1 |
| **Llama 3.2 1B** | -- | -- | -- | -- | -- | -- | -- | Limited data available |
| **Llama 3.2 3B** | -- | 77.7 | -- | -- | -- | 78.6 | -- | Decent reasoning baseline |

*MATH (not MATH-500). **Phi-4-mini-flash-reasoning variant.

### Medium Models (7B-14B)

| Model | MATH-500 | GSM8K | HumanEval | MMLU | GPQA-D | ARC-C | AIME'24 | Notes |
|-------|----------|-------|-----------|------|--------|-------|---------|-------|
| **Qwen2.5-7B-Instruct** | 75.5* | -- | 84.8 | 74.2 | -- | -- | -- | Strong all-around |
| **Qwen2.5-14B-Instruct** | -- | -- | -- | 79.7 | -- | -- | -- | Best open 14B general |
| **Qwen3-8B-Base** | 55.6* | -- | -- | 79.7 | 32.8 | -- | -- | Excellent base for RL |
| **Llama 3.1 8B-Instruct** | -- | 84.5 | 62.1 | 68.4 | 35.6 | -- | -- | Solid baseline |
| **Mistral 7B v0.3** | 3.0** | 34.5 | -- | 63.5 | -- | -- | -- | Weak math; **not suitable for math RL** |
| **DeepSeek-R1-Distill-Qwen-7B** | **92.8** | -- | -- | -- | 49.1 | -- | **55.5** | Gold standard reasoning 7B |
| **DeepSeek-R1-Distill-Qwen-14B** | **93.9** | -- | -- | -- | 59.1 | -- | **69.7** | Best open reasoning 14B |
| **InternLM2.5-7B-Chat** | -- | -- | -- | 72.8 | -- | -- | -- | Strong multilingual |
| **GLM-Z1-9B** | -- | -- | -- | -- | -- | -- | -- | "Top-ranked" for size per Zhipu AI |

*MATH (not MATH-500). **MATH Level 5 only.

### Key Takeaways from Comparison

1. **DeepSeek-R1 distilled models dominate reasoning benchmarks** at every size (1.5B, 7B, 14B), demonstrating that distillation from a strong reasoning teacher is currently the most effective approach.
2. **Qwen3 with thinking mode at 4B achieves 73.8 on AIME 2024**, which is extraordinary for the size and rivals much larger models.
3. **Phi-4-mini (3.8B)** excels at structured benchmarks (ARC-C: 83.7, GSM8K: 88.6) but the MATH-500 score of 92.5 is for the reasoning-enhanced variant.
4. **Mistral 7B is a poor base for math RL** (MATH Level 5: 3.0, GSM8K: 34.5). Base capability matters enormously.
5. **Gemma 3 1B** has limited headroom for RL reasoning improvement (MMLU: 38.8, HumanEval: 41.5).

---

## C. RL Training Success Stories on Small Models

### 1. DeepSeek R1-Zero (671B MoE base) and Distillation to 1.5B

**What happened**:
- R1-Zero: Pure RL (GRPO) on DeepSeek-V3 base without any SFT. AIME 2024 pass@1 went from 15.6% to 71.0%; majority voting reached 86.7%.
- Observed "aha moments": self-reflection, verification, strategy switching emerged naturally.
- **Problems**: Endless repetition, poor readability, language mixing.
- Solution: Added cold-start SFT data before RL to create DeepSeek-R1.

**Distillation to 1.5B**:
- Generated 800K training samples from R1 teacher.
- DeepSeek-R1-Distill-Qwen-1.5B: MATH-500: 83.9%, AIME 2024: 28.9%.
- Outperforms GPT-4o and Claude-3.5-Sonnet on math benchmarks at 1.5B.

**Key lesson**: Distillation from a strong teacher produces better small-model reasoning than RL directly on small models. "Reasoning patterns discovered through RL on small models" underperform distilled patterns.

Source: [DeepSeek-R1 paper](https://arxiv.org/abs/2501.12948), [HuggingFace model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)

### 2. Sky-T1-32B (Budget Reasoning Model)

**Base model**: Qwen2.5-32B-Instruct
**Method**: SFT on 17K curated reasoning examples (generated by QwQ-32B-Preview, rewritten by GPT-4o-mini)
**Cost**: <$450, 19 hours on 8x H100
**Results**: Matches o1-preview on Math500 and AIME 2024. Outperforms o1 on harder tasks.
**Key lesson**: Quality data curation matters more than RL sophistication at this stage. SFT alone on well-curated reasoning traces can match RL results.

Source: [Sky-T1 blog](https://novasky-ai.github.io/posts/sky-t1/)

### 3. TinyZero (UC Berkeley, 1.5B)

**Base model**: Qwen2.5-1.5B (and 0.5B)
**Method**: Pure RL using veRL framework on Countdown game
**Cost**: ~$30
**Results**:
- From 1.5B, models learn to search, self-verify, and revise solutions.
- **Qwen2.5-0.5B fails to learn reasoning** through RL.
- Critical finding: **1.5B is the minimum viable size** for RL-induced reasoning emergence.

**Key lesson**: Below 1.5B, pure RL fails to produce reasoning behaviors. The 0.5B-to-1.5B boundary appears to be a critical threshold.

Source: [TinyZero GitHub](https://github.com/Jiayi-Pan/TinyZero)

### 4. JustRL (Tsinghua, arXiv:2512.16649)

**Base models**: DeepSeek-R1-Distill-Qwen-1.5B, OpenMath-Nemotron-1.5B
**Method**: Single-stage GRPO, fixed hyperparameters, binary rule-based rewards (no symbolic math libraries)
**Results**:
- JustRL-DeepSeek-1.5B: **54.87% average** across 9 math benchmarks (beats ProRL-V2's 53.08%)
- JustRL-Nemotron-1.5B: **64.32% average** (beats QuestA's 63.81%)
- Uses **2x less compute** than more sophisticated approaches
- Smooth, monotonic improvement over 4,380 steps. No oscillations or collapses.
- **Same hyperparameters transfer across both base models without tuning.**

**Key lessons**:
1. Simple RL recipes work. Complex interventions (curriculum, annealing, multi-stage) may be unnecessary.
2. Base model quality matters enormously: OpenMath-Nemotron-1.5B (64.3%) vs DeepSeek-R1-Distill-1.5B (54.9%) with identical training.
3. Binary rewards from lightweight verifiers are sufficient.

Source: [JustRL paper](https://arxiv.org/abs/2512.16649), [JustRL GitHub](https://github.com/thunlp/JustRL)

### 5. SimpleRL (HKUST)

**Base models**: Qwen2.5 (0.5B to 32B), Qwen2.5-Math-7B
**Method**: Pure RL with 8K MATH examples
**Results**:
- 10-20+ absolute point gains across model sizes.
- Qwen2.5-1.5B: +8.4 absolute points on math reasoning.
- For 0.5B/1.5B models: simpler prompts required (just "step-by-step"); complex prompts hurt.
- 7B and 14B: ~100 steps of training, 15 hours on 2x8 H100.

**Key lesson**: Weaker models need simpler prompt formats. Complex CoT templates that work for 7B+ may actively hurt 1.5B models.

Source: [SimpleRL GitHub](https://github.com/hkust-nlp/simpleRL-reason)

### 6. Open-Reasoner-Zero

**Base models**: Qwen2.5 (0.5B to 32B)
**Method**: Vanilla PPO with GAE (lambda=1, gamma=1), rule-based rewards, **no KL regularization**
**Results**:
- ORZ-32B: AIME 2024: 48.1, MATH-500: 92.2, GPQA-D: 55.5
- Requires only **1/10 of training steps** compared to DeepSeek-R1-Zero pipeline.
- Consistent improvements from 0.5B to 32B, demonstrating scalability.
- ORZ-R1-Distill-Qwen-14B surpasses DeepSeek-R1-Distill-Qwen-32B.

**Key lesson**: No KL regularization needed. Simple PPO scales predictably with model size.

Source: [Open-Reasoner-Zero](https://arxiv.org/abs/2503.24290), [GitHub](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero)

### 7. DAPO (ByteDance)

**Base model**: Qwen2.5-32B (also tested 1.5B/7B)
**Method**: GRPO with 4 key modifications (Decoupled clip, Adaptive KL, Dynamic sampling, Overlong filtering)
**Results**:
- Qwen2.5-32B: 50 on AIME 2024 (beats R1-Zero's 47)
- Qwen2.5-Math-1.5B + DAPO: **MATH-500: 45.4%**
- Qwen2.5-Math-7B + DAPO: **MATH-500: 58.6%**
- Vanilla GRPO only reaches 30% on Qwen2.5-32B; DAPO's fixes are essential.

**Key lessons**:
1. Naive GRPO suffers entropy collapse, reward noise, training instability.
2. DAPO's entropy-preserving tricks (decoupled clipping) are critical for long-CoT RL.
3. On-policy training is essential to prevent entropy collapse.

Source: [DAPO paper](https://arxiv.org/abs/2503.14476)

### 8. Kimi K1.5 (Moonshot AI)

**Method**: RL with 128K context window, long-CoT training
**Results**:
- AIME: 77.5, MATH-500: 96.2, Codeforces: 94th percentile
- Long-to-short transfer: short-CoT model achieves AIME: 60.8, MATH-500: 94.6
- Outperforms GPT-4o and Claude 3.5 Sonnet by up to 550%.

**Key lesson**: Context length scaling in RL is a key ingredient. Longer reasoning chains enable better RL training outcomes.

Source: [Kimi K1.5 paper](https://arxiv.org/abs/2501.12599)

---

## D. Benchmark-to-RL Transfer Analysis

### D1. Which base model scores best predict RL training success?

**Most predictive** (in order):
1. **pass@k at high k (e.g., pass@256)**: The strongest predictor. If the base model can solve a problem at all (even rarely), RL can boost the frequency. RL narrows the sampling distribution toward correct solutions that already exist in the base model's output space.
2. **MATH-500 base score**: Direct indicator of mathematical reasoning capacity that RL will amplify.
3. **HumanEval/LiveCodeBench base score**: Code capability correlates with structured reasoning ability.
4. **GPQA Diamond**: Tests deep reasoning; high scores indicate reasoning headroom.

**Least predictive**:
- MMLU (knowledge, not reasoning)
- HellaSwag, WinoGrande (saturated, trivial)
- GSM8K at instruct level (saturated; doesn't discriminate)

### D2. Higher base MATH score -> better RL improvement?

**Evidence says: Yes, but with nuance.**

| Base Model | Base MATH | Post-RL MATH | Improvement | Source |
|------------|-----------|--------------|-------------|--------|
| Qwen2.5-Math-1.5B | ~35 | 45.4 (DAPO) | +10.4 | DAPO paper |
| Qwen2.5-Math-7B | ~50 | 58.6 (DAPO) | +8.6 | DAPO paper |
| OpenMath-Nemotron-1.5B | ~50 | 64.3 (JustRL) | +14.3 | JustRL paper |
| DeepSeek-R1-Distill-1.5B | ~55 | 54.9 (JustRL) | ~0 | JustRL paper |

**Critical finding from JustRL**: The OpenMath-Nemotron-1.5B backbone (stronger base) achieved **64.3% vs 54.9%** with identical RL training. The base model quality is the dominant factor. However, the DeepSeek-R1-Distill backbone showed minimal improvement, suggesting **already-distilled reasoning models have less headroom for RL improvement**.

**Diminishing returns pattern**: RL primarily surfaces reasoning patterns already latent in the base model (per the "Limit of RLVR" research). Models with higher pass@256 but lower pass@1 have the most RL headroom.

### D3. Does base instruction-following quality matter?

**Mixed evidence:**
- SimpleRL found that for **0.5B/1.5B models, simpler prompts work better**. Complex instruction formats hurt.
- For 7B+ models, instruction-tuned bases can provide a better starting point for RL.
- **However**: RL on base models (not instruct) is the emerging standard (DeepSeek R1-Zero, Open-Reasoner-Zero).
- Instruct models may have **constrained output distributions** that limit RL exploration.

**Recommendation**: For NanoSeek at 1B, use the **base model** for RL, not an instruct variant. Use simpler prompt formats.

### D4. Does base code ability help reasoning RL?

**Yes, strongly correlated.**
- Models with higher HumanEval/LiveCodeBench scores tend to produce better structured reasoning.
- Code training teaches systematic decomposition, variable tracking, and step-by-step execution -- all beneficial for math reasoning.
- Kimi K1.5 showed that math and code RL mutually reinforce each other.

**Implication for NanoSeek**: The FIM training in NanoSeek's pretraining should provide a code-reasoning advantage for subsequent RL.

### D5. Minimum base capability for RL to work?

**The evidence points to a clear threshold around 1.5B parameters:**

| Size | RL Outcome | Evidence |
|------|-----------|----------|
| **0.5B** | **Fails**. Cannot learn reasoning through RL. | TinyZero, SimpleRL |
| **1.5B** | **Works**. Self-verification, search, revision emerge. | TinyZero, JustRL, SimpleRL, DAPO |
| **7B** | **Works well**. Substantial gains. | SimpleRL, DAPO, Open-Reasoner-Zero |
| **14B+** | **Works excellently**. Can rival much larger models. | ORZ-14B beats R1-Distill-32B |

**For NanoSeek at 1.08B active parameters**: This is **below the 1.5B threshold** observed in the literature. However:
- NanoSeek has 4.75B total parameters (MoE), which may provide more latent capacity.
- The 1.5B threshold was measured on dense models. MoE routing may enable more efficient use of capacity.
- **Recommendation**: Plan for the possibility that pure RL may underperform expectations. Have distillation as a fallback strategy.

---

## E. Evaluation Framework Recommendation

### E1. Benchmarks to Track During RL Training

**Primary (evaluate every checkpoint)**:
| Benchmark | Why | Metric |
|-----------|-----|--------|
| MATH-500 | Main reasoning target | pass@1 (greedy) |
| GSM8K | Easy math sanity check | pass@1 |
| HumanEval+ | Code reasoning transfer | pass@1 |

**Secondary (evaluate every 5th checkpoint)**:
| Benchmark | Why | Metric |
|-----------|-----|--------|
| GPQA Diamond | Deep reasoning ceiling | pass@1 |
| ARC-Challenge | Reasoning breadth | accuracy |
| AIME 2024 | Competition-level math | Avg@32 |
| LiveCodeBench | Fresh code problems | pass@1 |

**Regression monitors (evaluate every 10th checkpoint)**:
| Benchmark | Why | Metric |
|-----------|-----|--------|
| IFEval | Instruction following regression | accuracy |
| MMLU-Pro (subset) | Knowledge retention | accuracy |
| Perplexity on held-out text | Language model quality | bits-per-byte |

### E2. Evaluation Frequency

| Training Phase | Eval Frequency | Which Benchmarks |
|---------------|----------------|------------------|
| Early RL (steps 0-500) | Every 50 steps | MATH-500, GSM8K only (fast) |
| Active learning (steps 500-3000) | Every 200 steps | Primary suite |
| Maturation (steps 3000+) | Every 500 steps | Full suite including AIME Avg@32 |
| Final checkpoint | Once | Complete evaluation on all tiers |

**For AIME (30 problems)**: Always run 8 repetitions and report average. Single-run scores are too noisy.

### E3. Pass/Fail Criteria at Each Stage

**Stage 1: Initial RL Training (first 500 steps)**
- PASS: MATH-500 improves by >= 3 absolute points over base model
- PASS: GSM8K doesn't degrade by more than 2 points
- FAIL: Entropy collapse (monitor policy entropy; should not drop below 50% of initial)
- FAIL: Response length explosion (>4x base model average length without accuracy gain)

**Stage 2: Active RL Training (500-3000 steps)**
- PASS: MATH-500 improves monotonically (small fluctuations OK, no sustained regression)
- PASS: HumanEval doesn't degrade by more than 5 points
- PASS: At least 2/3 of primary benchmarks show improvement
- FAIL: Any benchmark drops >10 points from peak
- FAIL: Training loss oscillates wildly (>50% variance across 100-step windows)

**Stage 3: Late Training (3000+ steps)**
- PASS: AIME 2024 Avg@32 > base model Avg@32
- PASS: GPQA Diamond improves or holds steady
- PASS: IFEval doesn't degrade by more than 5 points
- FAIL: Performance plateaus for >1000 steps (diminishing returns; stop training)

### E4. How to Detect Reward Hacking

**Behavioral signals**:
1. **Accuracy vs. held-out benchmark divergence**: If MATH-500 improves but AIME/GPQA don't, the model may be overfitting to reward signal patterns rather than learning reasoning.
2. **Response length inflation without accuracy gain**: Model learns that longer responses get higher partial credit.
3. **Format exploitation**: Model outputs reasoning-like tokens ("Let me think step by step...") without actual reasoning content.
4. **Train/test distribution gap**: RL models trained on training split perform nearly identically to those trained on test split (per 2025 research), suggesting pattern matching rather than generalization.

**Metric-based detection**:
1. Track **pass@1 vs pass@256 ratio**. If pass@1 increases but pass@256 doesn't, RL is narrowing distribution, not expanding capability.
2. Monitor **diversity of solution strategies** (not just accuracy).
3. Compare performance on **temporally-separated benchmarks** (e.g., AIME 2024 vs AIME 2025). Large gaps indicate overfitting.
4. Use **MATH-Beyond benchmark** (specifically designed to test RL generalization beyond training distribution).

**Mitigation**:
- Use diverse reward signals (rule-based verifiers, not reward models).
- Maintain on-policy training (DAPO finding: off-policy causes entropy collapse).
- Include diverse problem types in RL training data, not just math.

### E5. How to Detect Capability Regression

**Monitor these continuously**:
1. **Perplexity on held-out text**: If BPB increases significantly, general language modeling is degrading.
2. **IFEval score**: Instruction following is often the first casualty of reasoning RL.
3. **Output coherence**: Sample random generations and check for language mixing, repetition, or formatting degradation (all observed in R1-Zero).
4. **Code benchmark stability**: If HumanEval drops while MATH improves, reasoning gains may be domain-specific rather than general.

**Prevention strategies**:
- Use KL regularization (if not using DAPO/ORZ approach that omits it).
- Include diverse evaluation prompts (not just math) in periodic checks.
- Set hard stop-loss thresholds: if any Tier 2+ benchmark drops >10 points, investigate before continuing.

---

## F. Consolidated Recommendations for NanoSeek

### F1. Base Model Evaluation Priority

Before starting RL on NanoSeek-1B, establish baseline scores on:
1. MATH-500 (primary)
2. GSM8K (secondary)
3. HumanEval+ (code transfer)
4. GPQA Diamond (reasoning ceiling)
5. Perplexity on held-out text (regression baseline)

### F2. RL Algorithm Selection

Based on success stories:
- **GRPO with DAPO-style fixes** is the recommended starting point (proven at 1.5B scale, open-source).
- JustRL's simplicity is appealing, but uses binary rewards which may be limiting.
- **Avoid**: Complex multi-stage pipelines until simple RL works.
- **Critical**: Use on-policy training to prevent entropy collapse.

### F3. Risk Assessment for 1.08B Active Parameters

- NanoSeek is **below the 1.5B dense threshold** where RL reasoning emerges.
- MoE architecture (4.75B total) may compensate, but this is **uncharted territory** for RL training.
- **Mitigation**: Prepare a distillation pipeline as primary path, with pure RL as experimental secondary.
- **Minimum viable experiment**: Train RL for 500 steps, check if MATH-500 improves by >= 3 points. If not, pivot to distillation.

### F4. Which Model to Distill From (If Needed)

Based on the benchmark analysis, the strongest teachers for small-model distillation:
1. **DeepSeek-R1** (best reasoning traces, proven at 1.5B)
2. **Qwen3-32B (Thinking)** (excellent reasoning, diverse)
3. **Kimi K1.5** (strong long-CoT, if available)

---

## Sources

- [DeepSeek-R1 paper](https://arxiv.org/abs/2501.12948) and [GitHub](https://github.com/deepseek-ai/DeepSeek-R1)
- [DeepSeek-R1-Distill-Qwen-1.5B HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- [JustRL: Scaling a 1.5B LLM with a Simple RL Recipe](https://arxiv.org/abs/2512.16649)
- [TinyZero GitHub](https://github.com/Jiayi-Pan/TinyZero)
- [SimpleRL GitHub](https://github.com/hkust-nlp/simpleRL-reason)
- [Open-Reasoner-Zero](https://arxiv.org/abs/2503.24290) and [GitHub](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero)
- [DAPO: Open-Source LLM RL at Scale](https://arxiv.org/abs/2503.14476)
- [Kimi K1.5](https://arxiv.org/abs/2501.12599)
- [Sky-T1 blog](https://novasky-ai.github.io/posts/sky-t1/)
- [Does RL Really Incentivize Reasoning Beyond Base Model?](https://arxiv.org/abs/2504.13837)
- [Limit of RLVR](https://limit-of-rlvr.github.io/)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786)
- [Phi-4-mini HuggingFace](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- [Phi-4-reasoning Technical Report](https://arxiv.org/abs/2504.21318)
- [Effective Learning for Small Reasoning Models](https://arxiv.org/abs/2506.13404)
- [RL for Reasoning in Small LLMs: What Works and What Doesn't](https://arxiv.org/abs/2503.16219)
- [MMLU-CF: Contamination-free MMLU](https://aclanthology.org/2025.acl-long.656.pdf)
- [Reward Hacking in RL (Lilian Weng)](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)
- [METR: Recent Frontier Models Are Reward Hacking](https://metr.org/blog/2025-06-05-recent-reward-hacking/)
- [Anthropic: Natural Emergent Misalignment from Reward Hacking](https://assets.anthropic.com/m/74342f2c96095771/original/Natural-emergent-misalignment-from-reward-hacking-paper.pdf)
- [MATH-Beyond: Benchmark for RL Beyond Base Model](https://arxiv.org/abs/2510.11653)
- [A Sober Look at Progress in Language Model Reasoning](https://arxiv.org/abs/2504.07086)
- [LXT: LLM Benchmarks Compared](https://www.lxt.ai/blog/llm-benchmarks/)
- [State of LLM Reasoning Model Training (Sebastian Raschka)](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training)
