# Base Model Scout Report: Best 2026 Small/Efficient Models for RL Post-Training

**Date**: March 2026
**Purpose**: Identify optimal base models for RL reasoning training (GRPO/RLVR)
**Scope**: Open-weight models, 0.5B-30B parameters, released/updated 2025-2026

---

## Executive Summary

The 2026 small model landscape has dramatically improved. Models at 4B now match 2024's 70B models on reasoning tasks. The top candidates for RL post-training are dominated by the **Qwen3** family (best breadth and proven RL track record), **Gemma 3** (strong architecture, Google distillation), and **Phi-4** variants (exceptional math reasoning from synthetic data).

**Top recommendation**: **Qwen3-8B** or **Qwen3-4B** as primary base models, with **Qwen3-30B-A3B** (MoE) as an efficiency play. These have the strongest RL training ecosystem, proven GRPO results, and Apache 2.0 licensing.

---

## Ranked Shortlist: Top 8 Candidates for RL Reasoning Training

### Tier 1: Primary Recommendations

---

### 1. Qwen3-8B (Dense)

| Attribute | Value |
|-----------|-------|
| **Parameters** | 8B (dense) |
| **Architecture** | Dense transformer, GQA |
| **Training data** | ~36T tokens |
| **Context window** | 32K (extensible) |
| **License** | Apache 2.0 |
| **Release** | April 2025 |

**Key Benchmarks (Instruct, Thinking Mode)**:
- MATH-500: ~95.2%
- AIME 2024: ~70%
- GPQA Diamond: ~65%
- HumanEval: competitive with Qwen2.5-14B
- Matches Qwen2.5-14B-Base across most benchmarks

**RL Suitability: EXCELLENT**
- Proven GRPO/RLVR results: 10-20+ absolute point gains demonstrated on Qwen2.5-7B (same architecture family)
- Sky-T1-7B (built on Qwen2.5-Math-7B) achieved near o1-mini performance with simple RL
- Massive community: Unsloth, vLLM, TRL, OpenRLHF all support Qwen3 natively
- Built-in thinking/non-thinking mode demonstrates latent reasoning capacity
- 36T token pretraining provides deep knowledge base for RL to unlock
- Qwen3 training pipeline itself used GRPO for post-training

**Strengths**: Best ecosystem support, proven RL results, strong base intelligence, Apache 2.0
**Weaknesses**: Dense = higher memory than MoE alternatives at same quality

---

### 2. Qwen3-4B (Dense)

| Attribute | Value |
|-----------|-------|
| **Parameters** | 4B (dense) |
| **Architecture** | Dense transformer, GQA |
| **Training data** | ~36T tokens |
| **Context window** | 32K |
| **License** | Apache 2.0 |
| **Release** | April 2025 |

**Key Benchmarks**:
- GSM8K: ~89% (thinking mode)
- MATH-500: ~93%
- Rivals Qwen2.5-7B-Base and in some benchmarks Qwen2.5-72B-Instruct
- HumanEval: competitive (Gemma 3 4B IT gets 71.3% for reference)

**RL Suitability: EXCELLENT**
- Same architecture as Qwen3-8B, just smaller -- all tooling works
- 4B is the sweet spot for rapid RL iteration (fits on single 24GB GPU easily)
- Research shows smaller models show LARGEST gains from fine-tuning/RL
- Qwen3-4B-Thinking-2507 variant shows ~30 point AIME jump with thinking enabled
- Extremely fast training iteration cycles

**Strengths**: Fast iteration, low compute cost, proven architecture, surprisingly strong base
**Weaknesses**: Less headroom than 8B for complex reasoning ceiling

---

### 3. Qwen3-30B-A3B (MoE)

| Attribute | Value |
|-----------|-------|
| **Parameters** | 30B total / 3B active |
| **Architecture** | MoE, GQA |
| **Training data** | ~36T tokens |
| **Context window** | 32K |
| **License** | Apache 2.0 |
| **Release** | April 2025 |

**Key Benchmarks**:
- AIME 2025: 70.9% baseline, 85.1% with UloRL tuning
- Outcompetes QwQ-32B (10x more active params) on several benchmarks
- Matches Qwen2.5 dense models at 10% of active parameters

**RL Suitability: VERY GOOD**
- 3B active parameters = fast inference for RL sample generation
- 30B total capacity = much larger knowledge base than dense 3B
- MoE routing stability is a concern for RL (reward hacking through routing)
- Fewer proven RL results on MoE models vs dense
- UloRL research already shows strong RL gains on this exact model

**Strengths**: Best efficiency ratio, large capacity with small compute, proven RL gains
**Weaknesses**: MoE routing stability during RL is less studied, more complex deployment

---

### Tier 2: Strong Alternatives

---

### 4. Gemma 3 4B (Dense)

| Attribute | Value |
|-----------|-------|
| **Parameters** | 4B (dense) |
| **Architecture** | Dense, 5:1 local/global attention, SigLIP vision encoder |
| **Training data** | 4T tokens |
| **Context window** | 128K (via RoPE rescaling) |
| **License** | Gemma license (permissive, some restrictions) |
| **Release** | March 2025 |

**Key Benchmarks**:
- GSM8K: 89.2%
- HumanEval: 71.3%
- Competitive with Gemma 2-27B-IT on instruction following
- MMLU-Pro: strong but below Qwen3.5-9B

**RL Suitability: GOOD**
- Google's distillation-based post-training is sophisticated (SFT + RLHF + code execution feedback)
- 128K context is valuable for long CoT reasoning
- Novel 5:1 local/global attention reduces KV cache by ~75% -- great for RL rollouts
- Multimodal capability (vision) could enable multimodal RL
- Less community RL tooling compared to Qwen family
- 4T tokens pretraining (vs Qwen3's 36T) may limit base knowledge

**Strengths**: 128K context, efficient KV cache, multimodal, Google quality
**Weaknesses**: Less RL ecosystem support, Gemma license vs Apache 2.0, less pretraining data

---

### 5. Phi-4-mini-reasoning (3.8B)

| Attribute | Value |
|-----------|-------|
| **Parameters** | 3.8B (dense) |
| **Architecture** | Dense transformer |
| **Training data** | Synthetic-heavy (undisclosed total) |
| **Context window** | 16K |
| **License** | MIT |
| **Release** | April-May 2025 |

**Key Benchmarks**:
- AIME 2024: 57.5%
- MATH-500: 94.6%
- GPQA Diamond: 52%
- Outperforms DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Llama-8B
- Comparable to o1-mini on math benchmarks

**RL Suitability: GOOD (with caveats)**
- Already RL-trained (Phi-4-reasoning-plus uses outcome-based RL)
- Exceptional math reasoning at 3.8B -- highest math ceiling in this size class
- MIT license is maximally permissive
- Synthetic data training means potential distribution gaps on real-world text
- Less community tooling than Qwen
- 16K context is limiting for long CoT
- Starting from an already-RL'd model may have diminishing returns

**Strengths**: Best-in-class math at <4B, MIT license, proven RL training
**Weaknesses**: Short context, synthetic data biases, already RL'd (less headroom?)

---

### 6. DeepSeek-R1-Distill-Qwen-7B

| Attribute | Value |
|-----------|-------|
| **Parameters** | 7B (dense, Qwen2.5-7B base) |
| **Architecture** | Dense transformer (Qwen2.5 architecture) |
| **Training data** | Qwen2.5 pretraining + 800K R1 distillation samples |
| **Context window** | 32K |
| **License** | MIT |
| **Release** | January 2025 |

**Key Benchmarks**:
- MATH-500: 92.8%
- GPQA Diamond: 49.1%
- AIME 2024: 55.5%
- Surpasses QwQ-32B-Preview on several benchmarks

**RL Suitability: GOOD (specialized)**
- Already contains R1 reasoning patterns via distillation
- Sky-T1-mini was built on top of this exact model with simple RL
- DeepSeek's own research shows distillation > RL on base models
- However, further RL on top of distilled models has shown gains
- MIT license
- Strong starting point but may have reasoning pattern ceiling from distillation

**Strengths**: Pre-loaded with R1 reasoning, MIT license, proven RL-on-top results
**Weaknesses**: Distillation may cap RL ceiling, older Qwen2.5 base (vs Qwen3)

---

### 7. Qwen3.5-9B (Dense)

| Attribute | Value |
|-----------|-------|
| **Parameters** | 9B (dense) |
| **Architecture** | Dense transformer, latest Qwen3.5 architecture |
| **Training data** | Undisclosed (expected >36T) |
| **Context window** | 32K+ |
| **License** | Apache 2.0 |
| **Release** | March 2026 |

**Key Benchmarks**:
- MMLU-Pro: 82.5
- GPQA Diamond: 81.7
- Beats GPT-OSS-120B on MMLU-Pro, GPQA Diamond, IFEval
- MMMU-Pro (vision): 70.1 vs GPT-5-Nano's 57.2
- First sub-10B model competitive with 100B+ models

**RL Suitability: VERY GOOD**
- Newest architecture (March 2026) with latest training innovations
- Highest base intelligence in sub-10B class = best RL starting point
- Same Qwen ecosystem for tooling
- Vision capabilities enable multimodal RL
- Very new -- less community RL experimentation yet

**Strengths**: Highest raw intelligence sub-10B, newest architecture, Apache 2.0
**Weaknesses**: Very new (less proven for RL), 9B requires more compute than 4B

---

### 8. OLMo 3 7B (Dense)

| Attribute | Value |
|-----------|-------|
| **Parameters** | 7B (dense) |
| **Architecture** | Dense transformer |
| **Training data** | Fully open (Dolma-based) |
| **Context window** | 32K |
| **License** | Apache 2.0 (fully open: data + code + weights + logs) |
| **Release** | 2025-2026 |

**Key Benchmarks**:
- MMLU: mid-80s
- Competitive with Gemma 3 27B on some benchmarks
- OLMo 3-Think variant shows strong reasoning

**RL Suitability: GOOD (unique value)**
- ONLY fully open model: training data, code, intermediate checkpoints ALL released
- Ai2's post-training pipeline (SFT + DPO + RLVR) is fully documented
- Dolci post-training data suite is open
- Best choice for reproducible RL research
- Lower raw benchmark scores than Qwen3-8B at same size
- RLVR results documented: consistent improvements on GSM8K and MATH

**Strengths**: Full openness (data+code+checkpoints), documented RL pipeline, reproducibility
**Weaknesses**: Lower raw benchmarks than Qwen3/Gemma at same size

---

## Comparison Table

| Model | Params (Active) | Arch | License | MATH-500 | AIME'24 | GPQA-D | HumanEval | Context | RL Proven? | RL Score |
|-------|----------------|------|---------|----------|---------|--------|-----------|---------|------------|----------|
| **Qwen3-8B** | 8B | Dense | Apache 2.0 | ~95% | ~70% | ~65% | Strong | 32K | Yes (GRPO) | 9.5/10 |
| **Qwen3-4B** | 4B | Dense | Apache 2.0 | ~93% | ~55% | ~55% | Good | 32K | Yes (GRPO) | 9.0/10 |
| **Qwen3-30B-A3B** | 3B/30B | MoE | Apache 2.0 | ~94% | ~71% | ~60% | Good | 32K | Yes (UloRL) | 8.5/10 |
| **Gemma 3 4B** | 4B | Dense | Gemma | ~88% | ~45% | ~50% | 71.3% | 128K | Partial | 7.5/10 |
| **Phi-4-mini** | 3.8B | Dense | MIT | 94.6% | 57.5% | 52% | Good | 16K | Yes (ORM) | 7.5/10 |
| **DS-R1-Distill-7B** | 7B | Dense | MIT | 92.8% | 55.5% | 49.1% | Good | 32K | Yes (distill+RL) | 7.0/10 |
| **Qwen3.5-9B** | 9B | Dense | Apache 2.0 | ~96% | ~75% | 81.7% | Strong | 32K | Not yet | 8.0/10 |
| **OLMo 3 7B** | 7B | Dense | Apache 2.0 | ~90% | ~50% | ~55% | Good | 32K | Yes (RLVR) | 7.0/10 |

*Note: Some benchmark numbers are approximate, sourced from multiple evaluation settings. "RL Score" is a composite suitability rating.*

---

## Honorable Mentions (Not in Top 8)

### Llama 3.2 3B / Llama 4 Scout (17B active, MoE)
- Llama 3.2 3B: GSM8K 77.7%, ARC-C 78.6%. Too weak for serious RL reasoning.
- Llama 4 Scout: 17B active / 109B total MoE. Strong but very large. Better suited as a teacher/reward model than RL target.

### SmolLM2 1.7B
- Trained on 11T tokens, strong for its size. Too small for reasoning RL (ceiling too low).

### Mistral Small 3.2 (24B)
- MMLU 84.78%, HumanEval+ 92.9%. Excellent model but 24B is expensive for RL iteration.

### Gemma 3n E4B
- 8B params, 4B active memory footprint. Optimized for mobile, not RL training. Novel architecture makes RL tooling harder.

### GLM-4.7 (32B active / 400B total)
- AIME 2025: 95.7%, MATH: 97.1%. Exceptional but massive. Better as evaluation reference or reward model.

### MiniMax-M2 (10B active / 230B total)
- Strong coding/agentic model. MoE with 10B active. Less RL research community around it.

### InternLM3-8B
- Trained on only 4T tokens but competitive with Llama 3.1-8B. Less community RL support than Qwen.

### Qwen3-1.7B / Qwen3-0.6B
- Useful for ultra-efficient experiments. Qwen3-1.7B matches Qwen2.5-3B. Good for RL method prototyping on single GPU.

---

## RL Training Ecosystem Assessment

### Framework Support Matrix

| Framework | Qwen3 | Gemma 3 | Phi-4 | Llama | OLMo |
|-----------|-------|---------|-------|-------|------|
| **Unsloth** | Full | Full | Partial | Full | Partial |
| **TRL (HF)** | Full | Full | Full | Full | Full |
| **OpenRLHF** | Full | Partial | Partial | Full | Partial |
| **vLLM** | Full | Full | Full | Full | Full |
| **SGLang** | Full | Full | Partial | Full | Partial |
| **DeepSpeed** | Full | Full | Full | Full | Full |

### Proven RL Training Results on Small Models (2025-2026)

1. **Qwen2.5-Math-7B + GRPO** -> Sky-T1-7B: +10.4% AIME24, +33.2% MATH500 (UC Berkeley)
2. **Qwen2.5-7B/14B/32B + RLVR**: 10-20+ absolute point gains (multiple research groups)
3. **DeepSeek-R1-Distill-Qwen-7B + RL** -> Sky-T1-mini: near o1-mini math performance
4. **Llama-3.2-1B + GRPO**: Reasoning model with 16GB VRAM (Unsloth demo)
5. **Phi-4-reasoning + outcome-based RL** -> Phi-4-reasoning-plus (Microsoft)
6. **OLMo 2/3 7B + RLVR**: Consistent GSM8K/MATH improvements (Ai2)
7. **Qwen3-30B-A3B + UloRL**: AIME25 70.9% -> 85.1% (surpassing Qwen3-235B-A22B)

---

## Final Recommendations

### For NanoSeek RL Post-Training

Given that NanoSeek is a 1.08B active / 4.75B total MoE model, the most relevant base models for comparison and RL technique transfer are:

1. **Primary RL target**: Use **Qwen3-4B** or **Qwen3-8B** as the base for RL experiments
   - Best ecosystem, most proven results, Apache 2.0
   - 4B for rapid iteration, 8B for final quality

2. **MoE reference**: Study **Qwen3-30B-A3B** RL results closely
   - Same MoE paradigm as NanoSeek
   - UloRL results are directly relevant to MoE RL training

3. **Technique transfer**: Study **Phi-4-mini-reasoning** training pipeline
   - Best math reasoning at <4B parameters
   - Outcome-based RL approach is directly applicable

4. **Reproducibility reference**: Use **OLMo 3 7B** documentation
   - Fully open training pipeline including RL
   - Best resource for understanding what works and why

5. **For prototyping RL methods cheaply**: **Qwen3-1.7B** or **Qwen3-0.6B**
   - Fit on single consumer GPU
   - Same architecture, results transfer upward

---

## Sources

- [Qwen3 Technical Report](https://arxiv.org/html/2505.09388v1)
- [Qwen3.5 Small Models](https://www.marktechpost.com/2026/03/02/alibaba-just-released-qwen-3-5-small-models/)
- [Qwen3 Blog](https://qwenlm.github.io/blog/qwen3/)
- [Gemma 3 Technical Report](https://arxiv.org/html/2503.19786v1)
- [Gemma 3 DeepMind](https://deepmind.google/models/gemma/gemma-3/)
- [Gemma 3n](https://deepmind.google/models/gemma/gemma-3n/)
- [Phi-4-reasoning Technical Report](https://www.microsoft.com/en-us/research/publication/phi-4-reasoning-technical-report/)
- [Phi-4-mini-reasoning HuggingFace](https://huggingface.co/microsoft/Phi-4-mini-reasoning)
- [DeepSeek-R1 GitHub](https://github.com/deepseek-ai/DeepSeek-R1)
- [DeepSeek-R1 Paper](https://arxiv.org/html/2501.12948v1)
- [OLMo 3 Blog](https://allenai.org/blog/olmo3)
- [OLMo 2 Paper](https://arxiv.org/abs/2501.00656)
- [Sky-T1 Project](https://novasky-ai.github.io/posts/sky-t1/)
- [Sky-T1-7B RL Training](https://novasky-ai.github.io/posts/sky-t1-7B/)
- [Llama 4 Meta Blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- [Mistral Small 3](https://mistral.ai/news/mistral-small-3)
- [SmolLM2 HuggingFace](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B)
- [MiniMax-M2 GitHub](https://github.com/MiniMax-AI/MiniMax-M2)
- [GLM-4.7](https://medium.com/@leucopsis/a-technical-analysis-of-glm-4-7-db7fcc54210a)
- [GRPO++ Tricks](https://cameronrwolfe.substack.com/p/grpo-tricks)
- [Small LM Leaderboard](https://awesomeagents.ai/leaderboards/small-language-model-leaderboard/)
- [Best Small LLMs Under 10B 2026](https://www.siliconflow.com/articles/en/best-small-LLMs-under-10B-parameters)
- [State of RL for LLM Reasoning](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training)
- [GRPO Base Model RL](https://www.interconnects.ai/p/papers-im-reading-base-model-rl-grpo)
- [Qwen 2.5 RL Training](https://github.com/hkust-nlp/simpleRL-reason)
