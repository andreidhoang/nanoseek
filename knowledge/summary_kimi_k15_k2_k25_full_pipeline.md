# Kimi Model Family: Complete Architecture & Training Pipeline Analysis
## K1.5 → Kimi-VL → K2 → K2.5 (Jan 2025 — Feb 2026)

**Papers analyzed:**
- [Kimi K1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599) (Jan 2025)
- [Kimi-VL Technical Report](https://arxiv.org/abs/2504.07491) (Apr 2025)
- [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534) (Jul 2025)
- [Kimi K2.5: Visual Agentic Intelligence](https://arxiv.org/abs/2602.02276) (Feb 2026)

---

## 1. Evolution Timeline

```
K1.5 (Jan 2025)     K2 (Jul 2025)           K2.5 (Feb 2026)
  RL scaling    →   1T MoE + MuonClip   →   Multimodal + Agent Swarm
  Long2Short         384 experts              Joint vision-text
  128K context       15.5T tokens             15T mixed tokens
  Multimodal RL      Agentic tool-use         Zero-Vision SFT
                                              Token-efficient RL (Toggle)
        Kimi-VL (Apr 2025)
        MoonViT encoder
        16B MoE (2.8B active)
        Native resolution ViT
```

---

## 2. Architecture: Kimi K2 (The Base LLM)

### 2.1 Model Dimensions

| Parameter | Kimi K2 | DeepSeek V3 | NanoSeek (ours) |
|-----------|---------|-------------|-----------------|
| Total params | **1.04T** | 671B | 4.75B |
| Active params | **32.6B** | 37B | 1.08B |
| Layers | 61 | 61 | 16 |
| Hidden dim | 7168 | 7168 | variable |
| Attention | MLA | MLA | MLA |
| Attention heads | **64** | 128 | — |
| Total experts | **384** | 256 | 64 |
| Active experts | 8 | 8 | 8 |
| Shared experts | 1 | 1 | 2 |
| Dense layers | **1** | 3 | — |
| Sparsity ratio | **48** | 33 | 8 |
| Expert hidden dim | 2048 | — | — |
| Expert grouping | **No** | Yes | — |
| Context (pretrain) | 4096 | 4096 | 4096 |
| Context (final) | 128K (YaRN) | 128K | — |

### 2.2 Key Architecture Decisions (with ablation evidence)

**Why 384 experts (sparsity 48)?**
- Sparsity scaling law: under fixed activated params (constant FLOPs), increasing total experts consistently lowers training and validation loss
- At sparsity 48, FLOPs are reduced **1.69x** vs sparsity 8 for the same validation loss
- Trade-off: more experts = more total params but same compute cost per token

**Why 64 attention heads (not 128)?**
- Doubling heads only gives 0.5-1.2% validation loss improvement
- But causes **83% increase in inference FLOPs** at 128K context
- Critical for agentic applications requiring long-context inference
- Also enables smaller EP=16, ensuring full computation-communication overlap

**Why no expert grouping?**
- Simplified routing vs DeepSeek V3's grouped routing
- No ablation provided, but consistent with removing unnecessary complexity

**Why 1 dense layer (not 3)?**
- Further reduction from DSV3, not discussed in detail

---

## 3. MuonClip Optimizer (Major Innovation)

### 3.1 Why Muon Causes Instability

Muon's `msign` operation produces weight updates where **ALL singular values are equal** (full effective rank). Unlike Adam which has a skewed spectrum with low effective rank. This higher effective rank means:
- Higher probability for singular-vector pairs in weights to align with update directions
- Causes additive singular value growth
- Since attention logits involve the bilinear form `W_q * W_k^T`, the spectral norm is **squared**, compounding singular-value increases
- Result: attention logit explosion → training divergence

### 3.2 QK-Clip Mechanism

For each attention head h, compute max logit:
```
S_max^h = (1/√d) * max_{X in batch} max_{i,j} Q_i^h · K_j^h_T
```

When `S_max^h > τ` (threshold τ=100 for K2):
1. Compute scaling factor: `γ_h = min(1, τ / S_max^h)`
2. Rescale **only the affected head's** Q/K projection weights:
   - `W_qc^h` (compressed query): scaled by `√γ_h`
   - `W_kc^h` (compressed key): scaled by `√γ_h`
   - `W_qr^h` (rotary query): scaled by `γ_h`
   - `W_kr^h` (shared rotary key): **untouched** (avoid cross-head effects)

**Properties:**
- Per-head clipping (minimal intervention)
- Does NOT alter forward/backward computation — only post-update weight rescaling
- Self-deactivating: 12.7% of heads triggered in first 70K steps; zero after that
- K2 trained 15.5T tokens with **ZERO loss spikes**

### 3.3 Full MuonClip Step
```
1. Standard Muon step:
   - Momentum update
   - Newton-Schulz orthogonalization (5 iterations)
   - Scale by sqrt(max(n,m)) * 0.2 for Adam RMS matching
   - Apply with weight decay λ=0.1

2. QK-Clip (per-head):
   - Compute S_max^h over current batch
   - If S_max^h > τ: rescale Q/K weights as above
```

### 3.4 Relevance to NanoSeek
NanoSeek uses MuonAdamW (Muon for matrix params, AdamW for vectors). The QK-Clip mechanism is directly applicable if we observe attention logit explosion. At our 1B scale, this may be less critical, but the insight about Muon's full-rank updates causing spectral norm growth is important for understanding training dynamics.

---

## 4. Pre-Training Pipeline

### 4.1 K2 Pre-Training (Text Only)

**15.5T tokens** across 3 phases:

| Phase | Tokens | Context | LR | Batch |
|-------|--------|---------|-----|-------|
| Main (constant LR) | 10T | 4096 | 2e-4 (after 500-step warmup) | 67M tokens |
| Main (cosine decay) | 5.5T | 4096 | 2e-4 → 2e-5 | 67M tokens |
| Annealing | 400B | 4096 | 2e-5 → 7e-6 | 67M tokens |
| Long-context | 60B | 32K→128K | — | — |

**Data domains:** Web Text, Code, Mathematics, Knowledge

**Key innovation — Data Rephrasing:**
- Knowledge data: LLM rephrasings in varied styles (inspired by WRAP)
  - Chunk-wise: split long docs into chunks, rephrase individually, stitch back
  - Fidelity verification: semantic alignment checks
  - Result: 10 rephrasings × 1 epoch > raw × 10 epochs on SimpleQA (28.94 vs 23.76)
  - Each corpus rephrased at most twice in practice
- Math data: Rewrite into "learning-note" style + translate from other languages

### 4.2 K2.5 Continual Pre-Training (Multimodal)

Starting from a near-end K2 checkpoint, **15T additional mixed visual+text tokens:**

| Stage | Tokens | Context | What's Trained |
|-------|--------|---------|----------------|
| ViT Training | 1T | 4096 | MoonViT-3D only |
| Joint Pre-training | 15T | 4096 | ViT + LLM |
| Long-context Mid-training | 500B→200B | 32K→262K | ViT + LLM |

**Early fusion finding:** Vision from 0% of training with 10:90 vision:text ratio outperforms late-stage injection across ALL benchmarks. Late fusion causes "dip-and-recover" pattern for text performance.

---

## 5. Vision Architecture: MoonViT → MoonViT-3D

### 5.1 MoonViT (Kimi-VL)
- **~400M params**, initialized from SigLIP-SO-400M
- **Native resolution**: no sub-image splitting (unlike LLaVA-OneVision)
- **NaViT packing**: images divided into patches, flattened, concatenated into 1D sequences
- **Dual positional encoding**: interpolated absolute positional embeddings + 2D RoPE
- **Training**: SigLIP contrastive loss + cross-entropy caption loss (CoCa-style), λ=2
- **2T tokens** ViT training + 0.1T alignment to LLM
- **Connector**: 2-layer MLP with pixel shuffle (2×2 spatial downsampling, 4× channel expansion)

### 5.2 MoonViT-3D (K2.5)
- Extended for **video**: up to 4 consecutive frames as spatiotemporal volume
- 2D patches from frames jointly flattened and packed into 1D sequence
- **Lightweight temporal pooling** before MLP projector: 4× temporal compression
- Fully shared parameters between image and video (no separate video modules)
- Can handle up to **3.2M pixels** per image in thinking variant
- Training: cross-entropy loss only (dropped contrastive loss from Kimi-VL)

### 5.3 Decoupled Encoder Process (DEP) — Infrastructure Innovation
- Three-stage training step decouples vision encoder from main backbone
- Achieves **90% multimodal training efficiency** relative to text-only
- Allows reuse of text-only parallelism strategies for the LLM

---

## 6. Post-Training Pipeline

### 6.1 K1.5 Pipeline (Foundation)

```
Pretraining → Vanilla SFT → Long-CoT SFT → RL
```

**Vanilla SFT:** ~1M text + ~1M vision examples, 2 stages:
- Stage 1: 32K context, LR 2e-5→2e-6, 1 epoch
- Stage 2: 128K context, LR 1e-5→1e-6, 1 epoch

**Long-CoT SFT:** Small high-quality warmup dataset with planning/evaluation/reflection/exploration reasoning paths

**RL — Online Policy Mirror Descent:**
```
max_θ E[(y,z)~π_θ [r(x,y,y*)] - τ · KL(π_θ(x) || π_θ_i(x))]
```

Key design choices:
- **No value network** — conventional credit assignment penalizes exploratory reasoning
- Off-policy gradient with L2 regularization on log-ratio
- Baseline = simple mean of sampled rewards
- Optimizer reset at start of each iteration

**Length Penalty (anti-overthinking):**
```
len_reward(i) = 0.5 - (len(i) - min_len) / (max_len - min_len)  [correct]
               min(0, above)                                       [incorrect]
```

**Reward Models:**
- Math: Chain-of-Thought RM achieves **98.5% accuracy** (vs 84.4% for classic RM)
- Code: Auto-generated test cases (CYaRon), 50 tests per problem
- Verifiable: Rule-based binary 0/1

**Sampling strategies:**
- Curriculum sampling: start easy, progress to hard
- Prioritized sampling: sample proportional to (1 - success_rate)

### 6.2 K2 Post-Training (Agentic Focus)

```
SFT → Joint RL (verifiable + self-critique)
```

**SFT:**
- Muon optimizer (Muon-pretrained checkpoint performs best with Muon fine-tuning)
- Maximize prompt diversity + ensure high response quality
- Self-critic capability bootstrapped during SFT

**RL Reward Domains:**
1. **Math/STEM/Logic**: Diverse QA pairs, moderate difficulty via pass@k filtering
2. **Complex Instruction Following**: Hybrid verification (code interpreter + LLM judge + hack-check)
3. **Faithfulness**: Sentence-level judge model (FACTS Grounding framework)
4. **Coding/SWE**: Competition problems + real GitHub PRs; sandbox on K8s with 10K+ concurrent instances
5. **Safety**: Human-curated + automated adversarial prompt evolution

**Self-Critique Rubric Reward (non-verifiable tasks):**
- Model evaluates own outputs via pairwise comparisons
- Three rubric types: core, prescriptive, human-annotated
- Closed-loop refinement: on-policy rollouts from verifiable prompts continuously update critic

**RL Algorithm:** Same K1.5 base + additions:
- **Budget Control**: Per-sample max token budget; exceed = truncation + penalty
- **PTX Loss**: Auxiliary loss on curated samples to prevent forgetting
- **Temperature Decay**: High temp initially (exploration) → low temp (exploitation)

**Agentic Data Synthesis (3 stages):**
1. Tool spec generation: 3,000+ real MCP tools + 20,000+ synthetic tools
2. Agent/task generation: thousands of agents with different system prompts
3. Trajectory generation: sophisticated tool simulator with state + stochasticity, LLM judge filters

### 6.3 K2.5 Post-Training (Vision + Agentic)

```
Zero-Vision SFT → Visual RL → Joint Multimodal RL → PARL (Agent Swarm) → Toggle (Token-Efficient RL)
```

**Zero-Vision SFT (Novel):**
- Uses **only text SFT data** to activate visual capabilities
- Image manipulations proxied through IPython operations
- Adding human-designed visual trajectories during SFT actually **hurts** generalization
- Works because joint pre-training already established vision-text alignment

**Visual RL:**
- Outcome-based RL on visual grounding, chart/document understanding, vision STEM
- **Cross-modal transfer**: Visual RL improves textual performance (+1.7 MMLU-Pro, +2.1 GPQA-Diamond)

**Joint Multimodal RL:**
- Organized by **ability** (knowledge, reasoning, coding, agentic) not modality
- Domain experts jointly learn from text + multimodal queries
- Generative Reward Model (GRM) across heterogeneous traces

**Agent Swarm (PARL):**
- Trainable orchestrator + frozen subagents (from intermediate checkpoints)
- Two tools: `create_subagent`, `assign_task`
- PARL reward: `r = λ₁·r_parallel + λ₂·r_finish + r_perf`
  - `r_parallel`: prevents "serial collapse"
  - `r_finish`: prevents "spurious parallelism"
  - λ₁, λ₂ annealed to zero during training
- 3x-4.5x faster execution than single-agent

**Token-Efficient RL (Toggle):**
- Alternates between budget-limited and standard phases every m iterations
- Budget: ρ-th percentile of token lengths among correct responses (ρ=90%)
- Only enforced when mean accuracy > λ (λ=7/8)
- Result: **25-30% fewer tokens** with negligible performance impact

---

## 7. Long-Context Scaling

### 7.1 K1.5 Approach
- **Partial Rollouts**: Fixed output token budget caps trajectory length
- Unfinished portions saved to replay buffer, continued next iteration
- Repeat detection with early termination and penalties
- Context: 4096 → 32768 → 131072
- Key finding: **context length is a key dimension for RL scaling**

### 7.2 K2 Approach
- Annealing on 400B tokens at 4K context
- Then 60B tokens at 32K context, extended to 128K via **YaRN**

### 7.3 K2.5 Approach
- 500B tokens at 32K context → 200B tokens at 262K context via **YaRN**
- Supports 256K context for the multimodal model

### 7.4 Long2Short Methods (K1.5)
Four approaches to transfer long-CoT to short-CoT models:
1. **Model Merging**: Weight averaging (no training)
2. **Shortest Rejection Sampling**: n=8 samples, select shortest correct
3. **DPO**: Shortest correct=positive, longer=negative
4. **Long2Short RL** (best): Separate RL phase with length penalty + reduced max rollout

---

## 8. Training Infrastructure

### 8.1 Hardware
- NVIDIA H800 cluster (K2, K2.5)
- 8×400 Gbps RoCE interconnects per node
- 2 TB RAM per node

### 8.2 Parallelism (K2)
- **16-way Pipeline Parallelism (PP)** with virtual stages + interleaved 1F1B
- **16-way Expert Parallelism (EP)**
- **ZeRO-1 Data Parallelism**
- Model-parallel group: 256 GPUs (16 PP × 16 EP)
- Node count must be multiple of 32

### 8.3 Memory Optimization
- BF16 params + FP32 gradient accumulation
- ~30 GB/GPU for all states
- Selective recomputation: LayerNorm, SwiGLU, MLA up-projections, MoE down-projections
- FP8-E4M3 storage (not compute) for MoE up-projection and SwiGLU inputs
- CPU offloading with overlapped copy engine

### 8.4 RL Infrastructure
- Colocated training + inference on same workers
- Full 1T model parameter update: **<30 seconds**
- Megatron (training) + vLLM (inference) in separate containers via K8s Sidecar
- Weight transfer via Mooncake (RDMA): <1 min training→inference, ~10s inference→training
- Code sandbox: custom `crun` runtime, 120 containers/sec (vs 27/sec Docker)

---

## 9. Key Hyperparameters Summary

### K2 Pre-Training
| Parameter | Value |
|-----------|-------|
| Total tokens | 15.5T |
| Constant LR phase | 10T tokens at LR=2e-4 |
| Cosine decay phase | 5.5T tokens, 2e-4→2e-5 |
| Annealing | 400B tokens, 2e-5→7e-6 |
| Long-context | 60B tokens, 32K→128K |
| Warmup | 500 steps |
| Batch size | 67M tokens |
| Weight decay | 0.1 |
| QK-Clip τ | 100 |
| Optimizer | MuonClip |

### K2.5 Post-Training
| Parameter | Value |
|-----------|-------|
| RL temperature | 1.0 |
| RL top-p | 0.95 |
| Max completion (reasoning) | 96K tokens |
| Image/video max tokens | 64K |
| Video frames (short) | 128 (896 res) |
| Video frames (long) | 2048 (448 res) |
| Toggle λ (accuracy threshold) | 7/8 |
| Toggle m (phase period) | 2 iterations |
| Toggle ρ (budget percentile) | 90% |
| Agent Swarm steps (orchestrator) | 15-100 |
| Agent Swarm steps (subagents) | 50-100 |

### Kimi-VL
| Parameter | Value |
|-----------|-------|
| LLM total params | 16B |
| LLM active params | 2.8B |
| ViT params | ~400M |
| ViT training tokens | 2T + 0.1T |
| Joint pre-training tokens | 1.4T |
| Cooldown tokens | 0.6T |
| Long-context tokens | 0.3T |
| CoCa loss λ | 2 |
| RoPE base (initial→final) | 50,000→800,000 |
| SFT LR | 2e-5→2e-6, then 1e-5→1e-6 |
| Pixel shuffle | 2×2 spatial |

---

## 10. Benchmark Results (Best Numbers)

### K2 (Text-Only LLM)
| Benchmark | K2 Instruct | Best Competitor |
|-----------|-------------|-----------------|
| LiveCodeBench v6 | **53.7** | SOTA across all |
| AIME 2024 | **69.6** | DSV3: 59.4 |
| AIME 2025 | **49.5** | SOTA |
| GPQA-Diamond | **75.1** | SOTA |
| Tau2-Bench (agentic) | **66.1** | DSV3: 48.8 |
| IFEval | **89.8** | SOTA |
| FACTS Grounding | **88.5** | SOTA |
| LMSYS Arena | **#1 open-source** | #5 overall |

### K2.5 (Multimodal + Agentic)
| Benchmark | K2.5 | Notes |
|-----------|------|-------|
| AIME 2025 | **96.1%** | vs GPT-5.2: 100% |
| GPQA-Diamond | **87.6%** | |
| MathVista mini | **90.1%** | SOTA |
| OCRBench | **92.3%** | SOTA |
| SWE-Bench Verified | **76.8%** | |
| BrowseComp (Swarm) | **78.4%** | SOTA |
| HLE-Full (with tools) | **50.2%** | SOTA |
| LongVideoBench | **79.8%** | SOTA |
| OSWorld-Verified | **63.3%** | vs Claude: 66.3% |

---

## 11. Connections to NanoSeek

### 11.1 Architecture Lessons

**Sparsity scaling law** (K2): NanoSeek uses 64 experts with top-8 (sparsity=8). K2 shows that at fixed activated params, increasing total experts to sparsity=48 consistently improves loss. This suggests NanoSeek could benefit from more experts if memory allows (e.g., 128 experts with smaller per-expert dim).

**Attention head count** (K2): K2 deliberately cut heads from 128→64 for inference efficiency at long contexts. For NanoSeek at 1B scale, this tradeoff is less relevant but the principle of profiling inference cost per architectural choice is valuable.

**MLA confirmed**: Both K2 and NanoSeek use MLA. K2 at 1T scale validates MLA as the attention mechanism for MoE models.

### 11.2 Optimizer Lessons

**MuonClip**: NanoSeek already uses MuonAdamW. The QK-Clip mechanism is a cheap insurance policy:
- Compute S_max per head after each optimizer step
- If above threshold τ, rescale Q/K weights
- Self-deactivating (adds zero overhead once training stabilizes)
- Consider implementing for NanoSeek's training stability

**Muon for ALL stages**: K2 uses Muon for pre-training, SFT, AND RL. This validates the choice of Muon as a universal optimizer, not just for pre-training.

### 11.3 Training Pipeline Lessons

**Data rephrasing > multi-epoch**: K2 shows that rephrasing knowledge data yields better results than repeating the same data multiple epochs. For NanoSeek's 22B token budget on ClimbMix, this suggests considering LLM-rephrased variants of high-value data.

**Early fusion for multimodal**: If NanoSeek ever adds vision, K2.5 strongly recommends mixing vision tokens from the start (10:90 ratio) rather than adding them later.

**Long2Short RL**: K1.5's technique of using long-CoT RL to improve short-CoT models is directly applicable to NanoSeek's GRPO pipeline. Train with long context first, then do a separate RL phase with length penalty.

### 11.4 RL Pipeline Lessons

**No value network**: K1.5's finding that value functions penalize exploratory reasoning is important for NanoSeek's GRPO setup. Simple mean reward baseline works better.

**Self-Critique as reward**: K2's approach of bootstrapping self-critic capability during SFT, then using it as reward signal for non-verifiable tasks, is an elegant solution to the reward model bottleneck.

**Token-efficient RL (Toggle)**: K2.5's alternating budget-limited/unlimited phases reduces output tokens by 25-30% with negligible performance impact. Directly applicable to NanoSeek's post-training.

**Budget Control**: K2's per-sample max token budget with truncation + penalty is simpler than Toggle and could be tried first.

### 11.5 Infrastructure Lessons

**Colocated training+inference for RL**: K2 runs Megatron and vLLM on the same GPUs, switching between them. For NanoSeek on a single A6000, a simpler version of this (generate with the model, then train on generated data) is already the natural approach.

**Code sandbox**: K1.5's crun-based sandbox (120 containers/sec) enables fast code execution for reward computation. For NanoSeek's code RL, consider lightweight sandboxing.

### 11.6 What NanoSeek Could Adopt (Ordered by Priority)

1. **QK-Clip** — trivial to implement, cheap insurance against attention logit explosion
2. **Data rephrasing** — use an LLM to rephrase high-value ClimbMix data for better token utility
3. **Long2Short RL** — train with long CoT, then compress to short CoT via separate RL phase
4. **Toggle (token-efficient RL)** — alternating budget-limited/unlimited phases during GRPO
5. **Self-Critique reward** — bootstrap during SFT, use as RL reward for non-verifiable tasks
6. **Curriculum + prioritized sampling** — sample harder problems more during RL
7. **Length penalty** — anti-overthinking mechanism for GRPO training

---

## 12. Cross-Paper Architecture Comparison

```
                K1.5          Kimi-VL        K2             K2.5
                (Jan 2025)    (Apr 2025)     (Jul 2025)     (Feb 2026)
─────────────────────────────────────────────────────────────────────
LLM Arch        Transformer   MoE (16B)      MoE (1.04T)    MoE (1.04T)
                (undisclosed) 2.8B active    32B active     32B active
Attention       —             MLA            MLA            MLA
Experts         —             Moonlight      384            384
Active experts  —             —              8              8
Vision          Yes           MoonViT 400M   No             MoonViT-3D 400M
Optimizer       —             Enhanced Muon  MuonClip       MuonClip
Pre-train data  —             5.2T text      15.5T text     15T mixed
                              +4.4T multi
Context         128K          128K           128K           256K
RL              Policy Mirror Same           Same + Budget  Same + Toggle
                Descent                      + PTX + TempD  + PARL
Agentic         No            Yes (OS agent) Yes (tools)    Yes (Swarm)
Long2Short      Yes           No             No             No (Toggle)
```

---

## 13. Key Takeaways

1. **MuonClip is the breakthrough optimizer** — enables stable training at 1T scale with zero loss spikes. The QK-Clip mechanism is the key innovation, and it's cheap + self-deactivating.

2. **Sparsity scaling law** — more experts at fixed compute is consistently better. K2's 384 experts (sparsity 48) vs DeepSeek V3's 256 (sparsity 33) demonstrates this.

3. **Data quality > data quantity** — K2's rephrasing pipeline and K1.5's careful RL prompt curation show that thoughtful data engineering beats raw scale.

4. **RL without value networks** — K1.5 proved that simple policy mirror descent with mean reward baseline beats approaches using value functions, MCTS, or process reward models for long-CoT reasoning.

5. **Zero-Vision SFT** — K2.5's most surprising finding: text-only SFT activates visual capabilities when the base model has strong vision-text alignment from pre-training. Human-designed visual trajectories actually hurt.

6. **Cross-modal transfer** — Visual RL improves text performance (+2.1 on GPQA-Diamond). Modalities enhance each other when trained jointly.

7. **Agent Swarm via RL** — Training an orchestrator to decompose tasks into parallel sub-agents, with carefully designed auxiliary rewards to prevent serial collapse and spurious parallelism.

8. **Token efficiency matters** — K2.5's Toggle reduces output tokens 25-30% with negligible performance impact. K2's budget control + truncation is even simpler.
