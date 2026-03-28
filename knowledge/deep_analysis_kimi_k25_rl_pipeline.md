# Kimi K1.5 / K2 / K2.5 — Complete RL Pipeline Reconstruction

**Date**: 2026-03-24
**Sources**: K1.5 (arXiv 2501.12599), K2 (arXiv 2507.20534), K2.5 (arXiv 2602.02276)
**Confidence Legend**: [VERIFIED] = from paper, [INFERRED-STRONG] = strongly implied, [INFERRED-WEAK] = industry guess, [UNKNOWN] = not disclosed

---

## 1. FACTS — Verified Claims with Sources

### Model Architecture (K2 / K2.5 share the same backbone)
| Fact | Value | Confidence |
|------|-------|------------|
| Total parameters | 1.04T | [VERIFIED] K2 paper |
| Activated parameters | 32B per token | [VERIFIED] K2 paper |
| Layers | 61 (1 dense + 60 MoE) | [VERIFIED] K2 paper |
| Experts per MoE layer | 384 routed + 1 shared | [VERIFIED] K2 paper |
| Top-k experts | 8 | [VERIFIED] K2 paper |
| Activation ratio | 3.2% of total params | [VERIFIED] K2 paper |
| Context window | 256K (extended via YaRN) | [VERIFIED] K2 paper |
| Pre-training tokens | 15.5T | [VERIFIED] K2 paper |
| Attention mechanism | Multi-head Latent Attention (MLA) | [VERIFIED] K2 paper |
| Vision encoder (K2.5) | MoonViT-3D (NaViT-based, native res) | [VERIFIED] K2.5 paper |

### Key People & Organization
- **Organization**: Moonshot AI (Beijing)
- **Papers**: "Kimi Team" — no individual first author

---

## 2. RL ALGORITHM — Mathematical Details

### 2.1 K1.5: Online Mirror Descent with Squared Loss

[VERIFIED] The core RL algorithm, introduced in K1.5 and reused in K2/K2.5:

**Objective** — maximize reward with relative entropy regularization:
```
max_θ  E_{(y,z)~π_θ} [r(x,y,y*)] - τ · KL(π_θ(x) || π_{θ_i}(x))
```

**Closed-form optimal policy**:
```
π*(y,z|x) = π_{θ_i}(y,z|x) · exp(r(x,y,y*) / τ) / Z
```

**Practical surrogate loss** (squared loss form):
```
L(θ) = E[ (r(x,y,y*) - τ·log Z - τ·log(π_θ / π_{θ_i}))² ]
```

**Practical gradient** (Equation 3 from K1.5):
```
∇L = (1/K) Σ_j [ ∇log π_θ · (r_j - r̄) - (τ/2) · ∇(log(π_θ / π_{θ_i}))² ]
```

Key design choices:
- [VERIFIED] r̄ = mean reward baseline across K samples per prompt
- [VERIFIED] No value function, no MCTS, no process reward model
- [VERIFIED] Negative gradients on incorrect responses are critical (ablation in K1.5 Fig 10)
- [VERIFIED] Curriculum sampling: problems sampled proportional to (1 - success_rate)

### 2.2 K2: Squared Advantage Loss (Evolution of K1.5)

[VERIFIED] K2 adopts K1.5's algorithm with refinements:

```
L_RL(θ) = E_{x~D} [ (1/K) Σ_{i=1}^{K} (r(x,y_i) - r̄(x) - τ·log(π_θ(y_i|x) / π_old(y_i|x)))² ]
```

Where r̄(x) = (1/K) Σ r(x,y_i) — mean reward across K rollouts per prompt.

Additions over K1.5:
- [VERIFIED] Budget control: per-sample max token budget by task type, penalty for exceeding
- [VERIFIED] PTX loss: auxiliary loss on curated high-quality samples to prevent forgetting
- [VERIFIED] Temperature decay: exploration (high T) → exploitation (low T)

### 2.3 K2.5: Token-Level Clipping Extension

[VERIFIED] K2.5 extends with token-level clipping:

```
L_RL(θ) = E[ (1/N) ΣΣ Clip(π_θ/π_old, α, β) · (r(x,y) - r̄(x)) - τ·(log(π_θ/π_old))² ]
```

- α, β, τ > 0 control clipping bounds and divergence magnitude
- [VERIFIED] Token-level gradient masking: zeros gradients for log-ratios outside [α, β]
- [VERIFIED] Prevents off-policy drift in long-horizon multi-step tool-use reasoning

### 2.4 Length Penalty (K1.5)

[VERIFIED] Applied after initial warmup:
```
len_reward(i) = { λ  if r=1 (correct)
                { min(0, λ)  if r=0 (incorrect) }

where λ = 0.5 - (len(i) - min_len) / (max_len - min_len)
```

---

## 3. TRAINING PIPELINE — Full Multi-Stage Flow

### Stage Overview (K2 → K2.5 progression)

```
K2 Pipeline:
  Pre-training (15.5T tokens, MuonClip)
    → SFT (instruction tuning + agentic data synthesis)
      → Joint RL (coding + math + tool-use + instruction + factuality + safety)

K2.5 Pipeline (extends K2):
  K2 base model
    → Vision pre-training (MoonViT-3D, 3 stages)
      → Zero-Vision SFT (text-only SFT activates visual reasoning)
        → Joint Text-Vision RL (unified environment)
          → PARL for Agent Swarm (orchestrator RL)
            → Toggle (budget-controlled efficiency RL)
```

### 3.1 Pre-Training (K2)

[VERIFIED]
- 15.5T tokens, zero training spikes
- Optimizer: MuonClip (Muon + QK-clip)
- LR: 2e-4 constant (first 10T) → cosine decay to 2e-5 (next 5.5T)
- Global batch size: 67M tokens
- Context: 4,096 tokens during pre-training
- Weight decay: 0.1

### 3.2 SFT Stage (K2)

[VERIFIED] Agentic data synthesis pipeline:
1. Domain evolution → hierarchical application domains
2. Tool generation → ~23,000 synthetic + 3,000+ real MCP tools
3. Agent diversification → varied system prompts + tool combinations
4. Rubric-based task generation → success criteria specification
5. Multi-turn trajectory generation → user simulation with LLM personas
6. Quality filtering → only high-quality trajectories retained

[VERIFIED] K2.5 SFT:
- Synthesizes responses from K2, K2 Thinking, and proprietary expert models
- Specialized domain pipelines with human annotation + prompt engineering + multi-stage verification
- **Zero-Vision SFT innovation**: Text-only SFT activates visual reasoning without visual trajectories

### 3.3 Joint RL Stage (K2)

[VERIFIED] Unified RL across all domains simultaneously:
- Coding: Kubernetes sandbox, 10,000+ concurrent instances, unit test verification
- Math/STEM: QA pairs with difficulty selection, verifiable correctness
- Tool-use: Realistic simulator + real execution environments
- Instruction following: Hybrid rule verification (code interpreter + LLM-as-judge)
- Factuality: Sentence-level judge models for unsupported claims
- Safety: Attack-target-judge pipeline with rubric-based evaluation

[VERIFIED] Muon optimizer used for RL fine-tuning (same optimizer as pre-training produces best results)

### 3.4 Visual RL (K2.5)

[VERIFIED] Task-specific verifiable rewards:
- Grounding/localization: F1 with soft IoU matching
- Segmentation: Rasterized polygon IoU vs ground-truth
- OCR: Normalized edit distance (character-level)
- Counting: Absolute difference penalty
- Visual puzzles: LLM verifier (K2 as judge)

[VERIFIED] Cross-modal transfer discovery:
- Visual RL improves text performance:
  - MMLU-Pro: 84.7% → 86.4% (+1.7%)
  - GPQA-Diamond: 84.3% → 86.4% (+2.1%)
  - LongBench v2: 56.7% → 58.9% (+2.2%)

### 3.5 PARL — Parallel Agent RL (K2.5)

[VERIFIED] Architecture:
- Trainable orchestrator agent
- Frozen sub-agents from intermediate policy checkpoints
- Sub-agent trajectories excluded from optimization objective
- Only orchestrator parameters updated via RL

[VERIFIED] PARL reward function:
```
r_PARL(x,y) = λ₁·r_parallel + λ₂·r_finish + r_perf(x,y)
```
- r_parallel: prevents "serial collapse" (defaulting to single-agent)
- r_finish: sub-agent completion rate (prevents "spurious parallelism")
- r_perf: task-level outcome evaluation
- λ₁, λ₂ annealed to zero during training

[VERIFIED] Critical steps metric (measures parallel efficiency):
```
CriticalSteps = Σ_t (S_main^(t) + max_i S_sub,i^(t))
```

[VERIFIED] Results:
- 3x-4.5x wall-clock speedup vs single-agent
- BrowseComp: 78.4% vs 60.6% single-agent (+17.8%)
- WideSearch: 79.0% vs 72.7% (+6.3%)

### 3.6 Toggle — Budget-Controlled RL (K2.5)

[VERIFIED] Alternating optimization for token efficiency:
```
Phase 0 (Budget-Limited): Enforce token budget IF mean_accuracy > λ
Phase 1 (Standard Scaling): Unconstrained generation for inference-time scaling
```

- Budget estimated from ρ-percentile of correct-response token lengths
- Fixed at training start
- [VERIFIED] Results: 25-30% token reduction with negligible performance loss

### 3.7 Long2Short Transfer (K1.5)

[VERIFIED] Four complementary methods:
1. **Model merging**: Average weights between long-CoT and short-CoT models
2. **Shortest rejection sampling**: Sample n=8, select shortest correct response for SFT
3. **DPO**: Shortest correct as positive, 1.5x-length responses as negatives
4. **Long2short RL**: Separate RL phase with aggressive length penalties + reduced max rollout

---

## 4. REWARD FUNCTIONS — Detailed Breakdown

### 4.1 Verifiable Rewards (RLVR) — K2

[VERIFIED] Binary reward system (1 for correct, 0 for incorrect):
- **Math/STEM**: Answer matching against gold labels
- **Code**: Unit test execution in Kubernetes sandbox
- **Instruction following**: Code interpreter verification for deterministic constraints + LLM-as-judge for nuanced constraints
- **Faithfulness**: Sentence-level judge model detecting unsupported factual claims

### 4.2 Self-Critique Rubric Reward — K2

[VERIFIED] For subjective tasks (creative writing, chat):
- Model performs pairwise comparisons on its own outputs
- Three rubric categories:
  1. **Core rubrics**: fundamental values/identity (clarity, conversational fluency, objective interaction)
  2. **Prescriptive rubrics**: anti-reward-hacking constraints (no initial praise, no justification)
  3. **Human-annotated rubrics**: task-specific guidance from data team

[VERIFIED] Critic closed-loop refinement:
- On-policy rollouts from verifiable tasks continuously update the critic
- Distills objective performance signals from RLVR into subjective evaluation

### 4.3 Generative Reward Models (GRMs) — K2.5

[VERIFIED] Fine-grained evaluation aligned with Kimi values:
- Helpfulness, readiness, relevance, detail level, artifact quality
- Multiple alternative rubrics prevent reward hacking
- Applied to chat, coding, search, and artifact-generation agents

### 4.4 Visual Rewards — K2.5

[VERIFIED] See Section 3.4 above for task-specific visual reward functions.

### 4.5 Chain-of-Thought Reward Model — K1.5

[VERIFIED] CoT reward model achieves 98.5% accuracy vs 84.4% for classic RM (K1.5 paper).

---

## 5. INFRASTRUCTURE — Training Systems

### 5.1 GPU Cluster (K2 Pre-training)

[VERIFIED]
- NVIDIA H800 GPUs
- Nodes: 2TB RAM + 8 GPUs each (NVLink/NVSwitch intra-node)
- Inter-node: 8x 400 Gbps RoCE
- Flexible scaling: trainable on any multiple of 32 nodes

### 5.2 Parallelism Strategy (K2)

[VERIFIED]
- 16-way Pipeline Parallelism (PP) with virtual stages
- 16-way Expert Parallelism (EP)
- ZeRO-1 data parallelism
- Model-parallel group: 256 GPUs → ~30GB per GPU for params/gradients/optimizer

### 5.3 Memory Optimizations (K2)

[VERIFIED]
- EP communication overlapped with interleaved 1F1B scheduling
- Selective recomputation: LayerNorm, SwiGLU, MLA up-projections, MoE down-projections
- FP8-E4M3 storage (1x128 tiles) for MoE input activations with FP32 scales
- CPU offload of remaining activations via copy engine, overlapped during 1F1B

### 5.4 RL Infrastructure (K2)

[VERIFIED] Colocated architecture:
- Training + inference engines on same workers
- One engine releases GPU resources when other is active
- Muon optimizer used for RL (same as pre-training)
- Partial rollout: long-tail tasks pause/resume across iterations

[VERIFIED] K1.5 RL infra:
- Megatron for training, vLLM for inference
- Kubernetes-based hybrid deployment (~1 min train→inference transition)
- Custom container runtime (crun): 0.04s startup vs 0.12s Docker, 120 containers/sec vs 27

### 5.5 LoRA RL at Scale

[VERIFIED] (from Macaron AI / Mind Lab, applied to K2):
- LoRA-based RL achieves same quality with 10% GPU footprint
- <0.5% of parameters updated via low-rank matrices
- Hybrid tensor/pipeline/expert/sequence parallelism with LoRA sharding
- Merged into NVIDIA NeMo Megatron-Bridge and ByteDance VERL

### 5.6 Vision Training Infrastructure (K2.5)

[VERIFIED] Decoupled Encoder Process (DEP):
- Vision encoder replicated across GPUs
- Balanced forward loading
- Recompute backward pass
- 90% training efficiency vs text-only

### 5.7 MuonClip Optimizer

[VERIFIED] Standard Muon update:
```
M_t = μ·M_{t-1} + G_t
O_t = Newton-Schulz(M_t) · √max(n,m) · 0.2
W_t = W_{t-1} - η·(O_t + λ·W_{t-1})
```

[VERIFIED] QK-Clip mechanism:
```
S_max^h = (1/√d) · max_{X∈B} max_{i,j} Q_i^h · (K_j^h)^T    (max attention logit per head)
γ_h = min(1, τ / S_max^h)                                       (per-head scaling factor)
```

For MLA:
- Head-specific components (qC, kC): scaled by √γ_h
- Head-specific rotary (qR): scaled by γ_h
- Shared rotary (kR): unchanged
- Threshold τ = 100

[VERIFIED] Result: 15.5T tokens trained without a single training spike.

---

## 6. BENCHMARK RESULTS — Exact Numbers

### K2.5 (Thinking Mode)

| Benchmark | Score | Source |
|-----------|-------|--------|
| AIME 2025 | 96.1 | [VERIFIED] K2.5 paper |
| HMMT Feb 2025 | 95.4 | [VERIFIED] K2.5 paper |
| IMO-AnswerBench | 81.8 | [VERIFIED] K2.5 paper |
| GPQA-Diamond | 87.6 | [VERIFIED] K2.5 paper |
| MATH-500 | 98.0 | [VERIFIED] K2.5 paper |
| LiveCodeBench v6 | 85.0 | [VERIFIED] K2.5 paper |
| SWE-Bench Verified | 76.8 | [VERIFIED] K2.5 paper |
| Terminal Bench 2.0 | 50.8 | [VERIFIED] K2.5 paper |
| MMLU | 92.0 | [VERIFIED] K2.5 paper |
| IFEval | 94.0 | [VERIFIED] K2.5 paper |
| HumanEval | 99.0 | [VERIFIED] K2.5 paper |
| MMMU-Pro | 78.5 | [VERIFIED] K2.5 paper |
| OCRBench | 92.3 | [VERIFIED] K2.5 paper |
| LVBench | 75.9 | [VERIFIED] K2.5 paper |
| OSWorld-Verified | 63.3 | [VERIFIED] K2.5 paper |
| WebArena | 58.9 | [VERIFIED] K2.5 paper |

### K2 (Non-Thinking Mode)

| Benchmark | Score | Source |
|-----------|-------|--------|
| AIME 2024 | 69.6 | [VERIFIED] K2 paper |
| AIME 2025 | 49.5 | [VERIFIED] K2 paper |
| GPQA-Diamond | 75.1 | [VERIFIED] K2 paper |
| MMLU | 89.5 | [VERIFIED] K2 paper |
| MMLU-Redux | 92.7 | [VERIFIED] K2 paper |
| IFEval | 89.8 | [VERIFIED] K2 paper |
| LiveCodeBench v6 | 53.7 | [VERIFIED] K2 paper |
| SWE-Bench Verified (single) | 65.8 | [VERIFIED] K2 paper |
| SWE-Bench Verified (multi) | 71.6 | [VERIFIED] K2 paper |
| SWE-Bench Multilingual | 47.3 | [VERIFIED] K2 paper |
| Tau2-Bench | 66.1 | [VERIFIED] K2 paper |
| ACEBench | 76.5 | [VERIFIED] K2 paper |

### K1.5

| Benchmark | Score | Source |
|-----------|-------|--------|
| AIME 2024 (long-CoT) | 77.5 | [VERIFIED] K1.5 paper |
| MATH-500 (long-CoT) | 96.2 | [VERIFIED] K1.5 paper |
| Codeforces | 94th percentile | [VERIFIED] K1.5 paper |
| MathVista | 74.9 | [VERIFIED] K1.5 paper |
| AIME 2024 (short-CoT) | 60.8 | [VERIFIED] K1.5 paper |
| MATH-500 (short-CoT) | 94.6 | [VERIFIED] K1.5 paper |
| LiveCodeBench (short) | 47.3 | [VERIFIED] K1.5 paper |

### Agent Swarm Results (K2.5)

| Benchmark | Single-Agent | Agent Swarm | Delta |
|-----------|-------------|-------------|-------|
| BrowseComp | 60.6% | 78.4% | +17.8% |
| WideSearch | 72.7% | 79.0% | +6.3% |
| In-house Swarm Bench | 41.6% | 58.3% | +16.7% |

---

## 7. UNCERTAINTY MAP — Known vs Unknown

### Well-Documented (High Confidence)
- [VERIFIED] RL algorithm formula (squared loss mirror descent)
- [VERIFIED] MuonClip optimizer (full formula + QK-clip mechanism)
- [VERIFIED] PARL architecture and reward function
- [VERIFIED] Toggle algorithm concept
- [VERIFIED] Reward function types (RLVR, self-critique, GRM)
- [VERIFIED] Training parallelism strategy
- [VERIFIED] Benchmark numbers
- [VERIFIED] Vision training (MoonViT-3D, DEP, cross-modal transfer)

### Partially Disclosed
- [INFERRED-STRONG] Exact GPU count for K2 training — described as "any multiple of 32 nodes" with model-parallel group of 256 GPUs, but total cluster size not stated
- [INFERRED-STRONG] RL training duration / compute cost — not disclosed explicitly
- [INFERRED-STRONG] Exact PTX loss formulation — described conceptually ("auxiliary loss") but exact weighting/formula not given
- [INFERRED-STRONG] Number of RL training iterations — not disclosed
- [INFERRED-STRONG] K in rollout sampling (samples per prompt) — not specified for K2/K2.5

### Unknown / Not Disclosed
- [UNKNOWN] Total GPU hours for RL training
- [UNKNOWN] Exact training data composition for RL (prompt counts per domain)
- [UNKNOWN] τ (temperature / KL coefficient) value used in RL
- [UNKNOWN] Exact LoRA rank used for K2 RL fine-tuning
- [UNKNOWN] α, β clipping bounds in K2.5 token-level clipping
- [UNKNOWN] Toggle hyperparameters (ρ-percentile, λ threshold)
- [UNKNOWN] Number of PARL training iterations
- [UNKNOWN] Sub-agent checkpoint selection strategy (which intermediate checkpoints)
- [UNKNOWN] Exact reward model architecture / size
- [UNKNOWN] Cost of full K2.5 post-training pipeline
- [UNKNOWN] How many human annotators for rubric creation
- [UNKNOWN] Exact GRM training data and procedure

---

## 8. FAILURE MODES / KNOWN ISSUES

### Documented in Papers

1. **Serial collapse** (K2.5): Multi-agent systems default to single-agent serial execution even with agents available. Addressed via r_parallel reward component in PARL.

2. **Fake/spurious parallelism** (K2.5): Agents spawn sub-agents without doing actual parallel work. Addressed via r_finish reward component.

3. **Length overfitting** (K1.5/K2.5): Budget constraints cause models to fail at higher compute scales. Addressed via Toggle algorithm's alternating optimization.

4. **Attention logit explosion** (K2): During training, attention logits can exceed 1000, causing numerical instability. Addressed via QK-clip in MuonClip (τ=100 cap).

5. **Visual SFT hurts generalization** (K2.5): "Adding human-designed visual trajectories at this stage hurts generalization." Solution: zero-vision SFT (text-only SFT activates visual reasoning).

6. **Late vision injection underperforms** (K2.5): Late-stage heavy vision injection (50:50 ratio) underperforms early constant mixing (10:90 ratio). Early fusion is better.

7. **Reward hacking in subjective domains** (K2): Self-critique without prescriptive rubrics leads to reward hacking. Addressed via multi-rubric evaluation.

8. **RL prompt difficulty calibration** (K2): "The RL prompt-set should be neither too easy nor too hard, both of which may produce little signal."

9. **Catastrophic forgetting during RL** (K2): Joint RL can degrade high-quality pre-training knowledge. Addressed via PTX auxiliary loss on curated samples.

---

## 9. KEY ARCHITECTURAL INSIGHTS FOR NANOSEEK

### What's Transferable at Nano Scale

1. **Squared loss mirror descent** (from K1.5) — simpler than PPO, no value function needed, no MCTS. The formula is straightforward to implement. Key: use mean reward baseline, apply negative gradients on failures.

2. **Curriculum sampling** — proportional to (1 - success_rate). This is scale-agnostic.

3. **Length penalty** — the λ = 0.5 - (len - min_len)/(max_len - min_len) formula works at any scale.

4. **PTX anti-forgetting** — mix curated SFT data into RL objective. Simple auxiliary loss.

5. **Temperature decay** — high T early (exploration) → low T late (exploitation).

### What Requires Scale

1. **PARL / Agent Swarm** — requires frozen sub-agent copies, only meaningful at frontier scale.
2. **384 experts** — K2 specific; NanoSeek uses 64 experts.
3. **Kubernetes sandbox with 10K concurrent instances** — infrastructure heavy.
4. **MuonClip QK-clip** — mainly needed for trillion-parameter stability.

---

## 10. EVOLUTION ACROSS K1.5 → K2 → K2.5

| Aspect | K1.5 | K2 | K2.5 |
|--------|------|-----|------|
| RL algo | Squared loss mirror descent | Same + budget control + PTX | Same + token-level clipping |
| Modality | Text + vision (joint) | Text (primary) | Text + vision (joint) |
| Agents | None | Tool-use RL | PARL + Agent Swarm |
| Rewards | Verifiable + CoT RM | RLVR + self-critique rubric | + visual rewards + GRM |
| Efficiency | Long2Short transfer | Budget per task | Toggle algorithm |
| Optimizer | Not specified | MuonClip | MuonClip (inherited) |
| Scale | Not disclosed | 1.04T params, 15.5T tokens | Same backbone |
| Key innovation | Simplistic RL (no MCTS/PRM) | Agentic data synthesis | PARL, cross-modal RL transfer |
