# NVIDIA Open-Source AI Stack: High-Impact Contribution Analysis

## Principal Engineer Perspective | March 2026

---

## 1. What Frontier Labs Actually Use from NVIDIA's Stack

### DeepSeek: Minimal NVIDIA Software, Maximum NVIDIA Hardware

**What they use:**
- H800 GPUs (export-restricted H100 variant)
- NCCL for basic collective operations (as a starting point)
- cuBLAS for GEMM operations (via `torch._scaled_mm`)

**What they built custom:**
- **FlashMLA**: Custom CUDA kernels for MLA inference (660 TFlops on H800, 3000 GB/s memory bandwidth). Open-sourced Feb 2025.
- **DeepEP**: Custom expert-parallel all-to-all communication library using PTX (assembly-level) instructions. Bypasses CUDA entirely for critical paths to control register allocation and warp-level scheduling.
- **DualPipe**: Custom pipeline parallelism algorithm overlapping computation and communication.
- **FP8 training framework**: Custom MoE-aware mixed precision (gate routers protected). Did NOT use TransformerEngine.
- **Custom MoE dispatch kernels**: Optimized token routing with L2 cache reduction via PTX auto-tuned chunk sizes.

**Key insight**: DeepSeek treats NVIDIA hardware as raw silicon and writes nearly everything above the driver level themselves. They bypass CUDA for performance-critical paths. This is the strongest signal of where NVIDIA's software stack falls short.

### Meta: PyTorch-Native, Diverging from Megatron

**What they use:**
- H100 GPUs (massive clusters, 7.38M GPU-hours for Llama 4)
- PyTorch (they own it)
- torchtitan (their own training platform, NOT Megatron)
- FSDP2 for distributed training
- torch.compile for kernel fusion

**What they built custom:**
- Custom MoE parallelization for Llama 4 (first MoE architecture for Meta)
- Fully asynchronous online RL training framework (10x efficiency gain over previous gen)
- Custom pipeline parallelism and data loading optimizations

**Key insight**: Meta moved AWAY from Megatron-LM toward torchtitan, a PyTorch-native solution. This validates the hypothesis that Megatron's complexity is a barrier. torchtitan now supports DeepSeek-V3, Llama 4, and Qwen 3.

### Google DeepMind: Entirely Separate Ecosystem

**What they use:**
- TPUs (v5e, v6e, v7) -- zero NVIDIA GPUs for frontier training
- JAX + XLA compiler (co-designed with TPU hardware)
- Pathways framework for multi-pod orchestration

**Key insight**: Google is vertically integrated (hardware + compiler + framework). They are NOT a customer of NVIDIA's training software stack. TransformerEngine has JAX support, but Google doesn't use it. This market is unreachable.

### Chinese Labs (Kimi/Moonshot, MiniMax, GLM/Zhipu)

**Kimi/Moonshot:**
- **Uses** Megatron-LM as training framework foundation (confirmed for K2.5)
- **Built custom**: Muon optimizer at scale, MoonViT3d vision tower, Mooncake (disaggregated KV-cache serving, Best Paper FAST 2025)
- Deployed on H800/H100 GPUs

**MiniMax:**
- **Built custom**: Forge (agent-native RL framework), CISPO algorithm (their own RL method, clips importance weights not token updates)
- **Built custom**: Tree-structured merging strategy (40x training speedup)
- Runs on B200/H100 GPUs

**Zhipu/GLM:**
- **GLM-5 trained entirely on Huawei Ascend 910B** -- zero NVIDIA hardware
- Uses MindSpore framework instead of PyTorch
- 100,000 Ascend chip cluster

**Key insight**: Chinese labs are bifurcating. Some (Moonshot) still use Megatron-LM as a foundation. Others (Zhipu) have fully migrated to domestic hardware. MiniMax builds everything custom above the GPU level.

---

## 2. Known Gaps in Megatron-Core for 2026 Architectures

### Gap 1: MLA (Multi-head Latent Attention) -- PARTIALLY ADDRESSED

**Status**: Megatron-Core has basic MLA support (MLATransformerConfig), and absorbed MLA optimization (PR #3044) is in progress.

**Remaining gaps:**
- FlashMLA-level kernel performance not matched (DeepSeek achieves 660 TFlops; Megatron's TE-based path has shape mismatching bugs reported in Issue #1412)
- MLA lora ranks (275, 90, 440, 143) not divisible by 16, causing FP8 incompatibility -- MLA projections must stay BF16 (your NanoSeek finding is universal)
- No absorbed MLA + expert parallelism co-optimization (DeepSeek decouples MLA and MoE into micro-batches for overlap)
- Training-time MLA is functional but inference-optimized absorbed MLA for production is still catching up

### Gap 2: MoE Expert Parallelism Efficiency -- ACTIVE DEVELOPMENT

**Status**: EP all-to-all consumes 30-40% of training time without optimization. Megatron has 1F1B A2A overlap, but:

**Remaining gaps:**
- No equivalent to DeepEP's PTX-level all-to-all kernels (fundamental performance ceiling)
- MoE Parallel Folding (heterogeneous parallelism for attention vs MoE layers) is new but untested at DeepSeek V3 scale
- Expert Parallelism in Megatron-FSDP is MISSING (Issue #1781) -- major regression, blocks FSDP-based MoE training
- Fine-grained MoE models (64+ experts like DeepSeek) stress the all-to-all more than coarse (8-expert Mixtral)

### Gap 3: MTP (Multi-Token Prediction) -- RECENTLY ADDED

**Status**: MTP is now in Megatron-Core with standalone pipeline stages for VPP balance.

**Remaining gaps:**
- MTP + MoE + MLA combined configuration is complex and under-documented
- MTP lambda scheduling (0.3 -> 0.1 at 60% like DeepSeek) not natively configurable
- Speculative decoding integration for inference after MTP training is separate concern

### Gap 4: Mixture of Depths / Conditional Computation -- NOT SUPPORTED

**Status**: No support in Megatron-Core or TransformerEngine. Not on any roadmap.

**Assessment**: This is a research direction, not yet adopted by frontier labs in production. Lower priority than MLA/MoE/MTP gaps.

### Gap 5: Native GRPO/RL Training Integration -- SEPARATE ECOSYSTEM

**Status**: NeMo-RL exists as a separate package (not part of Megatron-Core). Supports GRPO, DAPO, PPO.

**Remaining gaps:**
- LoRA on Megatron-Core backend "coming soon" (only DTensor backend supports it today)
- Megatron-Core training + generation performance for RL is still being optimized
- No equivalent to MiniMax's CISPO or their 40x-speedup tree-structured merging
- veRL (Volcengine) has deeper Megatron integration for MoE RL (DeepSeek-671B, Qwen3-235B) -- NeMo-RL is behind here

---

## 3. TransformerEngine Gaps

### Gap 1: FP4 Training -- EARLY/BROKEN

- FP4 training support requested April 2025 (Issue #1701)
- JAX has NVFP4 training recipe, PyTorch is behind
- `--fp4-param-gather` reported broken (Issue #2352 -- NVFP4Tensors not supported in `replace_raw_data`)
- Megatron-LM references FP4 code that doesn't exist in released branches
- **Impact**: Blackwell GPUs (B200) have FP4 tensor cores. Without working FP4 training, Blackwell's compute advantage is inference-only.

### Gap 2: Custom Attention Pattern Support

- Standard MHA/GQA/MLA supported
- FlashAttention-2 and FlashAttention-3 integrated
- Sliding Window Attention (SWA) added
- **Missing**: DeepSeek Sparse Attention (DSA) patterns, variable-length grouped queries, hybrid attention (dense + sparse in same model)
- **Missing**: RoPE precision issues (Issue #1245 -- lower precision RoPE leads to training instability)

### Gap 3: MoE-Aware Mixed Precision

- TransformerEngine applies FP8 uniformly to all Linear layers
- **Missing**: Built-in awareness that gate routers MUST stay in higher precision (BF16/FP32)
- **Missing**: Per-layer precision policies (e.g., shared experts in FP8, gate in BF16, MLA projections in BF16 due to non-16-divisible dimensions)
- DeepSeek and your NanoSeek both solved this manually. It should be a first-class feature.

---

## 4. Most Valuable Open GitHub Issues

### Megatron-LM (NVIDIA/Megatron-LM)

| Issue | Title | Impact | Why It Matters |
|-------|-------|--------|----------------|
| #1729 | MoE Roadmap | Critical | Master tracking for all MoE features. DeepSeek-V3, Qwen3, Blackwell |
| #1781 | EP in Megatron-FSDP | High | Blocks FSDP-based MoE training. Major regression |
| #1878 | Intra-doc causal masking | High | Better loss, long context, massive FLOPS reduction. Critical for SFT |
| #1739 | gpt-oss implementation | Medium | YaRN RoPE, attention sinks, custom activations |
| #1535 | VPP > 1 throughput | Medium | all_gather latency kills VPP performance |
| #1874 | Profiling | Medium | Community needs better profiling/debugging tools |

### TransformerEngine (NVIDIA/TransformerEngine)

| Issue | Title | Impact | Why It Matters |
|-------|-------|--------|----------------|
| #1701 | FP4 Training | Critical | Blackwell's key advantage is FP4 compute. Training support is broken/incomplete |
| #2352 | FP4 param gather broken | High | Blocks FP4 training with FSDP |
| #2171 | NVFP4 training with Megatron | High | Integration between TE and Megatron for FP4 |
| #2565 | NVFP4 inference vs training perf gap | Medium | 50 vs 35 PFLOPS on Rubin -- why? |
| #1245 | RoPE precision instability | High | Lower precision RoPE breaks training. Affects all long-context models |

---

## 5. 80/20 Analysis: Which 20% of Contributions Deliver 80% of Value

### Tier 1: MAXIMUM IMPACT (do these first)

#### 1. MoE-Aware Mixed Precision Framework for TransformerEngine
**Why**: Every lab building MoE models (DeepSeek, Meta, Moonshot, MiniMax, Qwen) manually protects gate routers from FP8. This should be a first-class TE feature with per-layer precision policies.
- Gate routers: always BF16/FP32
- MLA projections with non-16-divisible dims: BF16
- Shared experts: FP8
- Everything else: configurable
**Effort**: ~2 weeks. **Impact**: Every MoE training run globally.

#### 2. Expert Parallelism in Megatron-FSDP (Issue #1781)
**Why**: FSDP is where the PyTorch community is going (Meta chose it over Megatron's custom distributed). Blocking EP in FSDP blocks the entire FSDP-MoE training path.
**Effort**: ~4-6 weeks. **Impact**: Unlocks FSDP-based MoE for the entire PyTorch ecosystem.

#### 3. FP4 Training Pipeline Fix (Issues #1701, #2352, #2171)
**Why**: Blackwell B200 GPUs are shipping. FP4 tensor cores are the hardware differentiator. If FP4 training doesn't work, customers have no reason to upgrade from H100 for training workloads.
**Effort**: ~4-8 weeks across TE + Megatron. **Impact**: Justifies Blackwell for training (not just inference).

### Tier 2: HIGH IMPACT

#### 4. Intra-Document Causal Masking (Issue #1878)
**Why**: Proven to improve loss AND long-context capability. Reduces FLOPS. Critical for SFT post-training (merge conversations in one sequence).
**Effort**: ~2-3 weeks. **Impact**: Every SFT training run.

#### 5. DeepEP-Quality Expert Parallel Communication Kernels
**Why**: DeepSeek's DeepEP achieves fundamentally better all-to-all performance via PTX. Megatron's EP communication is 30-40% of training time. Even a 2x improvement saves millions in compute.
**Effort**: ~8-12 weeks (kernel engineering). **Impact**: 15-20% training speedup for all MoE models.

#### 6. RoPE Precision Fix in TransformerEngine (Issue #1245)
**Why**: Training instability from low-precision RoPE affects every long-context model. Simple fix, huge reliability impact.
**Effort**: ~1 week. **Impact**: Training stability for all long-context workloads.

### Tier 3: STRATEGIC VALUE

#### 7. Absorbed MLA + EP Co-Optimization
**Why**: DeepSeek's key innovation is decoupling MLA and MoE into micro-batches for compute/communication overlap. Megatron has both MLA and EP but not the co-optimization.
**Effort**: ~6-8 weeks. **Impact**: Matches DeepSeek's training efficiency for MLA+MoE architectures.

#### 8. RL Training Integration (veRL/NeMo-RL + Megatron-Core)
**Why**: Post-training is where model differentiation happens. veRL already supports Megatron backend for DeepSeek-671B and Qwen3-235B. NeMo-RL's Megatron support is newer and less battle-tested.
**Effort**: Ongoing. **Impact**: Enables GRPO/DAPO at scale on Megatron.

#### 9. MTP Lambda Scheduling as First-Class Config
**Why**: Every DeepSeek-style model needs MTP lambda decay (0.3 -> 0.1 at 60%). Currently requires manual implementation.
**Effort**: ~1 week. **Impact**: Quality-of-life for all MTP training.

---

## Summary: The Landscape Map

```
                    BUILDS EVERYTHING CUSTOM          USES NVIDIA STACK
                    ←─────────────────────────────────────────────→

  DeepSeek ████████████████░░  (FlashMLA, DeepEP, DualPipe, FP8 -- all custom)
  Meta     ████████████░░░░░░  (torchtitan, FSDP2, custom RL -- diverging from Megatron)
  Google   ████████████████████ (TPU + JAX + XLA -- completely separate)
  Zhipu    ████████████████████ (Huawei Ascend + MindSpore -- left NVIDIA entirely)
  MiniMax  ██████████████░░░░  (Forge RL framework, CISPO, custom dispatch)
  Moonshot ████████░░░░░░░░░░  (Megatron-LM base + custom optimizer + Mooncake)
  Qwen     ██████░░░░░░░░░░░░  (Megatron-LM + PAI-Megatron fork)

  ░ = NVIDIA stack    █ = Custom
```

**The strategic conclusion**: The labs that push the frontier build custom. NVIDIA's stack serves the "long tail" of thousands of smaller teams who cannot afford custom kernel engineering. The highest-impact contributions make the gap between "use Megatron out of the box" and "DeepSeek-level custom" smaller. The three Tier 1 items (MoE-aware precision, EP in FSDP, FP4 training) would close approximately 30-40% of that gap for the broader community.

---

## Sources

- [DeepSeek FlashMLA GitHub](https://github.com/deepseek-ai/FlashMLA)
- [DeepEP GitHub](https://github.com/deepseek-ai/DeepEP)
- [DeepSeek PTX bypass analysis](https://www.tomshardware.com/tech-industry/artificial-intelligence/deepseeks-ai-breakthrough-bypasses-industry-standard-cuda-uses-assembly-like-ptx-programming-instead)
- [Megatron-Core MoE Roadmap (Issue #1729)](https://github.com/NVIDIA/Megatron-LM/issues/1729)
- [EP in Megatron-FSDP (Issue #1781)](https://github.com/nvidia/megatron-lm/issues/1781)
- [Intra-doc causal masking (Issue #1878)](https://github.com/nvidia/megatron-lm/issues/1878)
- [FP4 Training (TE Issue #1701)](https://github.com/NVIDIA/TransformerEngine/issues/1701)
- [FP4 param gather broken (TE Issue #2352)](https://github.com/NVIDIA/TransformerEngine/issues/2352)
- [RoPE precision instability (TE Issue #1245)](https://github.com/NVIDIA/TransformerEngine/issues/1245)
- [MoE Parallel Folding paper](https://arxiv.org/abs/2504.14960)
- [torchtitan GitHub](https://github.com/pytorch/torchtitan)
- [NeMo-RL GitHub](https://github.com/NVIDIA-NeMo/RL)
- [NeMo-RL Megatron-Core blog](https://developer.nvidia.com/blog/reinforcement-learning-with-nvidia-nemo-rl-megatron-core-support-for-optimized-training-throughput/)
- [veRL GitHub](https://github.com/verl-project/verl)
- [Megatron-LM DeepSeek V3 guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/deepseek_v3.html)
- [TransformerEngine GitHub](https://github.com/NVIDIA/TransformerEngine)
- [GLM-5 Huawei Ascend training](https://www.essamamdani.com/blog/kimi-2-5-vs-minimax-2-5-vs-glm-5-chinese-ai-2026)
- [MiniMax M2.5 technical details](https://www.minimax.io/news/minimax-m25)
- [Kimi K2 technical deep dive](https://intuitionlabs.ai/articles/kimi-k2-technical-deep-dive)
- [NVIDIA GTC 2026 open source strategy](https://explore.n1n.ai/blog/nvidia-gtc-2026-open-source-strategy-2026-03-21)
- [Llama 4 release](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- [Google TPU architecture](https://intuitionlabs.ai/articles/google-tpu-architecture-gemini-3)
- [Chinese open-source LLMs overview](https://intuitionlabs.ai/articles/chinese-open-source-llms-2025)
- [Architectural choices in China's AI ecosystem](https://huggingface.co/blog/huggingface/one-year-since-the-deepseek-moment-blog-2)
- [Post-training 2026 overview](https://llm-stats.com/blog/research/post-training-techniques-2026)
