# GLM-5 RL Pipeline — Comprehensive Research Report
## Reverse-Engineered from Official Technical Report + Open-Source Code

**Paper:** "GLM-5: From Vibe Coding to Agentic Engineering" (arXiv:2602.15763, February 2026)
**Model:** 744B MoE total / 40B active, 28.5T pre-training tokens, 200K context
**Developer:** Zhipu AI (Z.AI) / THUDM, Tsinghua University
**Hardware:** ~100,000 Huawei Ascend 910B (zero NVIDIA GPUs), MindSpore framework
**License:** MIT (fully open weights on HuggingFace: zai-org/GLM-5)
**RL Framework:** Slime (open-source: github.com/THUDM/slime)
**Analysis date:** 2026-03-24

---

# 1. ARCHITECTURE

## 1.1 Model Configuration [VERIFIED — config.json on HuggingFace]

| Parameter | Value |
|-----------|-------|
| model_type | `glm_moe_dsa` |
| hidden_size | 6144 |
| num_hidden_layers | 78 |
| num_attention_heads | 64 |
| num_key_value_heads | 64 |
| head_dim | 64 |
| qk_head_dim | 256 |
| qk_nope_head_dim | 192 |
| qk_rope_head_dim | 64 |
| v_head_dim | 256 |
| vocab_size | 154,880 |
| max_position_embeddings | 202,752 |
| intermediate_size (dense FFN) | 12,288 |
| moe_intermediate_size (expert) | 2,048 |
| n_routed_experts | 256 |
| n_shared_experts | 1 |
| num_experts_per_tok | 8 |
| routed_scaling_factor | 2.5 |
| topk_method | `noaux_tc` |
| scoring_func | sigmoid |
| norm_topk_prob | true |
| rope_theta | 1,000,000 |
| rope_interleave | true |
| q_lora_rank | 2,048 |
| kv_lora_rank | 512 |
| first_k_dense_replace | 3 (first 3 layers are dense) |
| num_nextn_predict_layers | 1 (MTP) |
| index_head_dim | 128 |
| index_n_heads | 32 |
| index_topk | 2,048 (DSA token selection) |
| rms_norm_eps | 1e-05 |
| hidden_act | silu |
| Total parameters | ~744B |
| Active parameters | ~40B |

## 1.2 Key Architectural Innovations [VERIFIED — arxiv paper]

**Multi-Latent Attention (MLA) Adaptation:**
- Based on DeepSeek-V3 MLA but modified with "Muon Split" technique
- Matrix orthogonalization applied independently to attention heads for differential scaling
- Head dimension increased to 256 (from 192) while decreasing head count by 1/3
- Maintains training computation parity

**DeepSeek Sparse Attention (DSA):**
- Dynamic token importance-based selection replacing O(L^2) dense attention
- Reduces attention computation 1.5-2x for long sequences
- Two-stage adaptation: 1000-step warmup (14 sequences of 202,752 tokens) + 20B-token sparse training
- Uses deterministic `torch.topk` operator (required for RL stability)
- DSA indexer is FROZEN during RL training

**Multi-Token Prediction (MTP):**
- 1 MTP layer (parameter sharing across 3 MTP layers vs DeepSeek-V3's single layer)
- Acceptance length: 2.76 (vs 2.55 for DeepSeek-V3.2) with 4 speculative steps

## 1.3 Comparison with Predecessors [VERIFIED]

| Model | Total Params | Active Params | Pre-train Tokens | Layers |
|-------|-------------|---------------|-----------------|--------|
| GLM-4.5 | 355B | 32B | 23T | — |
| GLM-5 | 744B | 40B | 28.5T | 78 (reduced to minimize EP overhead) |

---

# 2. PRE-TRAINING [VERIFIED — arxiv paper]

## 2.1 Data Scale
- Base model: 27T tokens
- Full training budget (incl. mid-training): 28.5T tokens

## 2.2 Data Composition
- **Web data:** Refined DCLM classifier + World Knowledge classifier for long-tail knowledge
- **Code data:** 28% increase in unique tokens vs GLM-4.5; dedicated classifiers for low-resource languages (Scala, Swift, Lua, etc.)
- **Math & Science:** Multiple sources (webpages, books, papers); strict filtering excluding synthetic/AI-generated/template content

## 2.3 Mid-Training Stages
- **32K stage:** 1T tokens
- **128K stage:** 500B tokens
- **200K stage:** 50B tokens
- Software engineering subset: ~160B unique tokens from 10M issue-PR pairs
- Long-context synthetic data using interleaved packing, NextLong and EntropyLong techniques

## 2.4 Infrastructure [VERIFIED — multiple sources]
- ~100,000 Huawei Ascend 910B processors
- MindSpore framework (Huawei's open-source DL platform)
- Full-stack adaptation across 7 Chinese chip platforms: Huawei Ascend, Moore Threads, Hygon, Cambricon, Kunlunxin, MetaX, Enflame
- Interleaved pipeline parallelism with flexible MTP placement
- Pipeline ZeRO2 gradient sharding across data-parallel ranks
- Context-parallel groups with dynamic size allocation for long-sequence training
- INT4 quantization-aware training in SFT stage

---

# 3. POST-TRAINING PIPELINE [VERIFIED — arxiv paper]

The pipeline follows a **progressive alignment strategy** with 5 stages:

```
SFT → Reasoning RL → Agentic RL → General RL → Cross-Stage Distillation
```

## 3.1 Stage 1: Multi-Task Supervised Fine-Tuning (SFT)

**Key changes vs GLM-4.5:**
- Significantly expanded Agent and Coding data scale
- Extended max context to 202,752 tokens
- INT4 quantization-aware training with bitwise-identical training/inference kernels

**Three Thinking Modes introduced:**
1. **Interleaved Thinking:** Reasoning before every response/tool call
2. **Preserved Thinking:** Multi-turn retention of reasoning
3. **Turn-level Thinking:** Per-turn control

**Three Data Categories:**
1. General Chat
2. Reasoning
3. Coding & Agent

**Data Processing:**
- Difficulty-based filtering for math problems
- Rejection sampling for logical reasoning
- Expert RL and rejection sampling for coding trajectories
- Erroneous segments masked in loss function

## 3.2 Stage 2: Reasoning RL [VERIFIED — arxiv paper, Equations 1]

**Domains:** Mathematics, science, code, competitive programming

**Difficulty Filtering:** Problems solvable by stronger teachers (GPT-5.2 xhigh, Gemini 3 Pro) but challenging for GLM-4.7

**Data Sources:** Codeforces, TACO, SYNTHETIC-2-RL for competitive programming

**Algorithm: GRPO + IcePop**

The loss builds on Group Relative Policy Optimization with the IcePop stabilization technique:

```
Loss = -E[min(r_i,t * A_i, clip(r_i,t, 1-eps_low, 1+eps_high) * A_i)]

where:
  r_i,t = pi^train(token) / pi^infer(token)    # importance ratio
  A_i = (R_i - mean(R)) / std(R)                # group-normalized advantage
  eps_low = 0.2                                  # asymmetric lower clip
  eps_high = 0.28                                # asymmetric upper clip (larger!)
```

**IcePop Mismatch Suppression:**
```
rho_i,t = pi^train / pi^infer   # training-inference mismatch ratio

If rho_i,t outside [1/beta, beta] where beta=2:
  → token gradient is suppressed (zeroed out)
```

This addresses MoE instability where training and inference routing diverge, causing compounding probability ratio distortion across token sequences.

**Key insight:** Asymmetric clipping (eps_low=0.2 < eps_high=0.28) allows the policy to increase probability of good actions more aggressively than it decreases probability of bad actions. This is a departure from standard PPO symmetric clipping.

## 3.3 Stage 3: Agentic RL [VERIFIED — arxiv paper, Equations 3-5]

**Fully asynchronous architecture** separating inference and training engines.

**Infrastructure:**
- Multi-Task Rollout Orchestrator managing 10,000+ verifiable environments
- TITO (Token-In-Token-Out) gateway preserving exact tokenization
- 1,000+ concurrent rollouts

**Algorithm: Token-level importance sampling with double-sided calibration**

```
r_t(theta) = exp(log pi_theta - log pi_rollout)   # token-level importance ratio

f(x) = {
  x    if x in [1-eps_l, 1+eps_h]
  0    otherwise                      # hard zeroing outside bounds
}

Loss = E_x[1/K * sum(f(r_t) * (r(x,y_i) - r_bar(x)))]
```

- Only model-generated tokens are optimized; environment feedback tokens are ignored
- Off-policy sample dropping when version staleness w'-w_0 > tau
- Environment failure filtering to reduce noisy training signals

**Environment Coverage (9 languages):**
- Python, Java, Go, C, C++, JavaScript, TypeScript, PHP, Ruby
- RepoLaunch framework with automated dependency analysis and test command generation
- 10,000+ verifiable environments

**Terminal Environments:**
- Docker-based Harbor format tasks
- Docker construction accuracy >90%
- Three-phase synthesis: draft generation, concrete implementation, iterative optimization

**Search Tasks:**
- Web Knowledge Graph (WKG) from 2M+ high-information pages
- Multi-hop QA generation from entity neighborhoods
- Three-stage difficulty filtering

## 3.4 Stage 4: General RL [VERIFIED — arxiv paper]

**Multi-dimensional optimization:**
1. Foundational correctness
2. Emotional intelligence
3. Task-specific quality

**Reward System (Hybrid):**
- Rule-based rewards
- Outcome Reward Models (ORMs)
- Generative Reward Models (GRMs)
- Human-in-the-loop: expert human responses as stylistic anchors

**Key motivation:** Purely model-generated optimization converges toward verbose, formulaic "model-like" patterns. Human-authored responses serve as qualitative anchors for natural, human-aligned patterns.

## 3.5 Stage 5: On-Policy Cross-Stage Distillation [VERIFIED — arxiv paper, Equation 2]

**Problem solved:** Sequential RL stages cause catastrophic forgetting of earlier capabilities.

**Algorithm:**
```
A_i,t = sg[log(pi_teacher^infer / pi_train)]   # advantage from teacher

where:
  sg = stop-gradient operator
  pi_teacher = final checkpoints from Reasoning RL AND General RL
  Group size = 1
  Batch size = 1024
  Training prompts sampled from corresponding RL training sets
```

**Mechanism:**
- Teachers are the final checkpoints from ALL earlier RL stages
- On-policy distillation rapidly recovers skills from SFT, Reasoning RL, and General RL
- Advantage is computed as log-probability ratio between teacher and student

---

# 4. SLIME RL FRAMEWORK [VERIFIED — GitHub THUDM/slime + arxiv]

## 4.1 Architecture
Three interconnected modules:
1. **Training Module (Megatron):** Reads from Data Buffer, syncs parameters to rollout after training
2. **Rollout Module (SGLang + Router):** Generates training data and reward signals
3. **Data Buffer:** Bridge managing prompts, custom data, generation methods

## 4.2 Two Operating Modes
- **Synchronous (GPU co-located):** Inference and training share same GPUs
- **Asynchronous (GPU decoupled):** Separate GPU clusters for each engine

## 4.3 Supported Algorithms
- GRPO
- DAPO
- GSPO
- (PPO-style variants via the above)

## 4.4 Supported Models
- GLM series (GLM-5, GLM-4.7, GLM-4.6, GLM-4.5)
- Qwen3 series and Qwen2.5
- DeepSeek V3 series (V3, V3.1, R1)
- Llama 3

## 4.5 APRIL Integration (Active Partial Rollouts in RL)
- Addresses long-tail generation bottleneck (>90% of RL training time)
- Over-provisions rollout requests
- Terminates once target number of responses reached
- Recycles incomplete responses for continuation in future steps
- **Performance:** Up to 44% rollout throughput improvement, up to 8% higher final accuracy
- Works across GRPO, DAPO, GSPO algorithms
- Separate paper: arXiv:2509.18521

## 4.6 Key Optimizations
- **FP8 inference** for tail-latency reduction
- **Multi-Token Prediction (MTP)** during rollout
- **Prefill-Decode disaggregation**
- **Multi-node distributed KV-cache** with DP-attention (prevents copying)
- **Heartbeat-driven** fault monitoring with automatic deregistration

---

# 5. BENCHMARK RESULTS

## 5.1 Official Benchmarks [VERIFIED — arxiv paper, Artificial Analysis]

| Benchmark | GLM-5 Score | GLM-4.7 Score | Notes |
|-----------|------------|---------------|-------|
| AIME 2025 | 84.0 | 95.7 | GLM-5 trades off AIME for broader capability |
| GPQA Diamond | 86.0 | 85.7 | Slight improvement |
| LiveCodeBench | 52.0 | 84.9 | Significant trade-off |
| SWE-bench Verified | 77.8 | — | Strong agentic coding |
| Chatbot Arena | 1451 (#1) | 1445 | Highest rated model |
| AA Intelligence Index v4 | 50 | 42 | +19% improvement |
| AA Agentic Index | 63 (#3 overall, #1 open) | — | Behind Claude Opus 4.6, GPT-5.2 xhigh |
| GDPval-AA ELO | 1412 | — | #3 overall |
| tau2-Bench Telecom | 98% | — | With reasoning |
| AA-Omniscience | -1 | -36 | +35 point improvement |
| Vending-Bench 2 | $4,432 (#1 open) | — | Final account balance metric |

**Note on AIME/LiveCodeBench discrepancy:** Some sources report GLM-5 at 91.67% AIME and 81.87% LiveCodeBench, likely from different evaluation methodologies or model variants (reasoning vs non-reasoning mode).

## 5.2 Long-Context Performance (DSA Ablation) [VERIFIED]

| Benchmark | MLA (dense) | DSA (sparse) |
|-----------|------------|--------------|
| MQ-NIAH-128k | 100% | 100% |
| MV-NIAH-128k | 95.5% | 97.0% |
| SQuAD-128k | 79.7% | 86.0% |
| HotpotQA-128k | 66.3% | 63.0% |

## 5.3 Search Agent (BrowseComp) [VERIFIED]

| Configuration | Score |
|--------------|-------|
| Without keep-recent-k | 55.3% |
| With keep-recent-k (k=5) | 62.0% |
| + Hierarchical context mgmt | 75.9% |

## 5.4 Overall Positioning
- ~20% improvement over GLM-4.7 on average across 8 agentic benchmarks
- Comparable to Claude Opus 4.5 and GPT-5.2 (xhigh)
- Surpasses Gemini 3 Pro

---

# 6. GLM LINEAGE (GLM-4 to GLM-5) [VERIFIED — arxiv:2406.12793 + multiple sources]

## 6.1 ChatGLM Family Evolution

| Generation | Key Features |
|-----------|-------------|
| GLM-130B (2022) | Bidirectional prefix LM, 130B dense |
| ChatGLM-6B (2023) | First chat model, in-house alignment data |
| ChatGLM2/3 (2023-2024) | Improved alignment, multi-turn |
| GLM-4 (2024) | 10T+ tokens pre-training, multi-stage RLHF, All Tools integration |
| GLM-4.5 (2025) | 355B MoE (32B active), 23T tokens, Slime v1, Interleaved Thinking, expert/unified training |
| GLM-4.6 (2025) | Updated reasoning, AA Intelligence Index 56 |
| GLM-4.7 (2025-2026) | Enhanced thinking modes (Preserved + Turn-level), strong AIME/LiveCodeBench |
| GLM-5 (Feb 2026) | 744B MoE (40B active), 28.5T tokens, async RL, DSA, APRIL, IcePop |

## 6.2 Key Changes from GLM-4.5 to GLM-5 [VERIFIED]

| Aspect | GLM-4.5 | GLM-5 |
|--------|---------|-------|
| Parameters | 355B (32B active) | 744B (40B active) |
| Pre-training | 23T tokens | 28.5T tokens |
| Experts | Unknown | 256 routed + 1 shared, top-8 |
| Attention | MLA | MLA + DSA (sparse) |
| RL mode | Sync + async (Slime v1) | Fully async (Slime v2) + APRIL |
| RL stages | Expert + Unified training | SFT → Reasoning RL → Agentic RL → General RL → Distillation |
| Thinking | Interleaved only | Interleaved + Preserved + Turn-level |
| Context | 128K | 200K |
| Hardware | Not disclosed | 100K Huawei Ascend 910B |

## 6.3 GLM-4 Alignment Approach [VERIFIED — arxiv:2406.12793]
- Multi-stage post-training: SFT + RLHF + safety alignment
- Combined rule-based, human (RLHF), and model-based (RLAIF) feedback
- Self-Contrast: feedback-free alignment strategy (novel technique)
- Training data: combination of in-house annotation + proprietary third-party data

---

# 7. UNCERTAINTY MAP

## What is KNOWN (high confidence):
- [VERIFIED] Full architecture config (config.json on HuggingFace)
- [VERIFIED] 5-stage post-training pipeline (SFT → 3 RL stages → Distillation)
- [VERIFIED] GRPO + IcePop algorithm with asymmetric clipping (eps_low=0.2, eps_high=0.28)
- [VERIFIED] Agentic RL: token-level importance sampling with double-sided calibration
- [VERIFIED] Cross-stage distillation formula (log-ratio advantage)
- [VERIFIED] Slime framework architecture and APRIL optimization
- [VERIFIED] DSA integration with frozen indexer during RL
- [VERIFIED] 100K Huawei Ascend 910B training
- [VERIFIED] 28.5T pre-training tokens
- [VERIFIED] Benchmark numbers from official paper and Artificial Analysis

## What is PARTIALLY KNOWN (medium confidence):
- [INFERRED - STRONG] Exact reward model architectures for General RL (paper mentions ORM + GRM hybrid but doesn't detail model sizes)
- [INFERRED - STRONG] Training compute budget (100K Ascend 910B chips, but total FLOPs/training time not disclosed)
- [INFERRED - STRONG] Exact SFT data composition ratios across General/Reasoning/Coding
- [INFERRED - STRONG] Number of RL training steps per stage

## What is UNKNOWN:
- [UNKNOWN] Exact reward model parameter counts and training data
- [UNKNOWN] Total training duration in GPU-hours
- [UNKNOWN] Exact batch sizes for each RL stage (only distillation batch=1024 disclosed)
- [UNKNOWN] Learning rates for RL stages
- [UNKNOWN] Number of rollout groups (G) for GRPO
- [UNKNOWN] Exact ratio of human vs AI feedback in General RL
- [UNKNOWN] Specific Generative Reward Model architecture
- [UNKNOWN] Pre-training hyperparameters (lr schedule, warmup, etc.)
- [UNKNOWN] Cost of training (estimated >$100M based on 100K chips)

---

# 8. KEY INSIGHTS FOR NANOSEEK

## 8.1 What NanoSeek Can Adopt
1. **IcePop for MoE RL stability:** The training-inference mismatch suppression is critical for MoE models. NanoSeek should implement the rho ratio check with beta=2.
2. **Asymmetric clipping:** eps_low=0.2, eps_high=0.28 allows more aggressive improvement on good actions.
3. **Cross-stage distillation:** Using log-probability ratio as advantage is elegant and avoids needing separate value models.
4. **DSA indexer freezing during RL:** Any sparse attention mechanism should be frozen during post-training.
5. **GRPO over PPO:** GLM-5 uses GRPO throughout, not PPO. No separate value/critic network needed.

## 8.2 Scale-Down Considerations
- GLM-5's 256 experts → NanoSeek's 64 experts: IcePop may be less critical at smaller scale but still valuable
- APRIL throughput optimization: relevant even at small scale for reducing RL training time
- Cross-stage distillation with group_size=1, batch_size=1024: can scale down proportionally
- 5-stage pipeline may be overkill at 1B scale; consider merging Reasoning+General RL

---

# SOURCES

- [GLM-5 Technical Report (arXiv:2602.15763)](https://arxiv.org/html/2602.15763v1)
- [GLM-5 on HuggingFace](https://huggingface.co/zai-org/GLM-5)
- [Slime RL Framework (GitHub)](https://github.com/THUDM/slime)
- [APRIL Paper (arXiv:2509.18521)](https://arxiv.org/html/2509.18521v1)
- [ChatGLM Family Paper (arXiv:2406.12793)](https://arxiv.org/html/2406.12793v1)
- [Artificial Analysis: GLM-5](https://artificialanalysis.ai/models/glm-5)
- [GLM-5 vs Chinese Frontier Models (Maniac)](https://www.maniac.ai/blog/chinese-frontier-models-compared-glm5-minimax-kimi-qwen)
- [Slime on AMD ROCm Blog](https://rocm.blogs.amd.com/artificial-intelligence/slime/README.html)
- [GLM-5 NxCode Guide](https://www.nxcode.io/resources/news/glm-5-open-source-744b-model-complete-guide-2026)
- [Artificial Analysis Tweet on GLM-5](https://x.com/ArtificialAnlys/status/2021678229418066004)
