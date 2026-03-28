# NVIDIA GPU Architecture Deep Research for LLM Training (March 2026)

## Executive Summary

NVIDIA's GPU landscape in 2026 spans three generations relevant to LLM training:
- **Blackwell (B200/GB200)**: Shipping since late 2024, now mainstream. FP8 native, FP4 introduced.
- **Blackwell Ultra (B300/GB300)**: GA March 2026. 288GB HBM3e, 1.5x faster than B200, FP4 training mature.
- **Vera Rubin**: In production Q1 2026, partner availability H2 2026. HBM4, NVLink 6, 3nm.

For NanoSeek specifically: **B200 on RunPod at $4.99/hr** is the sweet spot for Phase 3-4. FP8 training (already implemented in `fp8.py`) maps directly to Blackwell tensor cores. FP4 is not yet practical for custom training loops but worth tracking.

---

## 1. Blackwell Architecture (B200, GB200, B100)

### 1.1 Core Specifications — B200

| Spec | Value |
|------|-------|
| Transistors | 208 billion (dual-die) |
| Process | TSMC 4NP |
| SMs | 148 |
| CUDA Cores | 18,944 (128 per SM) |
| Tensor Cores | 5th generation |
| HBM3e | 192 GB (8 stacks) |
| Memory Bandwidth | 8 TB/s |
| Chip-to-Chip Interconnect | 10 TB/s (between two dies) |
| TDP | ~1000W |

### 1.2 Tensor Core Performance (per GPU)

| Precision | Dense TFLOPS | Sparse TFLOPS | vs H100 |
|-----------|-------------|---------------|---------|
| FP4 (NVFP4) | 9,000 | 18,000 | N/A (new) |
| FP8 (E4M3/E5M2) | 4,500 | 9,000 | ~2.3x |
| BF16/FP16 | 2,250 | 4,500 | ~2.3x |
| TF32 | 2,250 | 4,500 | ~2.8x |
| FP64 | 40 | N/A | ~1.3x |

**Key insight for NanoSeek**: The FP8 path in `fp8.py` using `torch._scaled_mm` maps to these 4,500 TFLOPS dense. The 2.3x over H100 means ablation runs that took 10 hours on H100 should take ~4.3 hours on B200.

### 1.3 5th-Gen Tensor Cores — New Capabilities

- **FP4 (E2M1 NVFP4)**: Values up to +/-6, grouped in blocks of 16 sharing an FP8 (E4M3) scale factor, plus per-tensor FP32 scale. This is "micro-tensor scaling" (MXFP4-like).
- **FP6 support**: New intermediate format, not yet widely adopted in training.
- **FP8 improvements**: Native support for both E4M3 (forward) and E5M2 (backward) — exactly what NanoSeek's `fp8.py` implements.
- **2:4 Structured Sparsity**: Hardware acceleration doubles throughput when 2 of every 4 weights are zero. Available for FP4/FP8/BF16.
- **Attention acceleration**: Blackwell Ultra (B300) adds 2x attention-layer acceleration hardware vs B200.

### 1.4 GB200 Superchip

The GB200 pairs one Grace CPU + two B200 GPUs:
- 384 GB combined HBM3e (192 GB per GPU)
- 16 TB/s combined memory bandwidth
- NVLink-C2C coherent CPU-GPU interconnect at 900 GB/s

### 1.5 GB200 NVL72 Rack-Scale System

| Spec | Value |
|------|-------|
| GPUs | 72 x B200 |
| CPUs | 36 x Grace |
| Total HBM3e | Up to 30 TB |
| Total Memory BW | Up to 576 TB/s |
| FP4 Compute | 72 PFLOPS |
| FP8 Compute | 36 PFLOPS |
| NVLink BW (total) | 130 TB/s |
| Network Fabric | NVLink 5.0 all-to-all |

### 1.6 NVLink 5.0

| Spec | Value |
|------|-------|
| Per-GPU Bandwidth | 1,800 GB/s bidirectional |
| Links per GPU | 18 x 100 GB/s |
| NVSwitch Ports | 72 per switch chip |
| NVSwitch Bandwidth | 14.4 TB/s non-blocking |
| Topology | All-to-all via NVSwitch fabric |

**MoE relevance**: The all-to-all topology is critical for expert parallelism. With 1,800 GB/s per GPU, the all-to-all dispatch/combine for MoE routing is no longer the bottleneck it was on H100 (900 GB/s NVLink 4.0).

---

## 2. Blackwell Ultra (B300/GB300) — GA March 2026

### 2.1 B300 Core Specs

| Spec | B300 | B200 | Improvement |
|------|------|------|-------------|
| HBM3e | 288 GB | 192 GB | +50% |
| Memory BW | 8 TB/s | 8 TB/s | Same |
| FP4 Dense | 14 PFLOPS | 9 PFLOPS | +55% |
| FP8 Dense | ~7 PFLOPS | 4.5 PFLOPS | +55% |
| Attention Accel | 2x vs B200 | Baseline | 2x |
| Die Architecture | Same dual-die | Same dual-die | Refreshed silicon |

### 2.2 Key Architectural Improvements

1. **2x Attention Acceleration**: Dedicated hardware for scaled dot-product attention, reducing the compute bottleneck for long-context workloads.
2. **50% More Memory**: 288 GB enables loading larger models without tensor parallelism. A 70B model in BF16 fits in a single B300.
3. **Micro-Tensor Scaling (MX) Matured**: FP4 training is production-ready with Transformer Engine v2 on B300, using block-wise FP8 scales over FP4 blocks of 16.
4. **Same NVLink 5.0**: Compatible with existing NVL72 fabric.

### 2.3 GB300 NVL72

| Spec | Value |
|------|-------|
| FP4 Compute | 1.1 ExaFLOPS dense |
| GPUs | 72 x B300 |
| Total Memory | ~20.7 TB |
| Performance vs GB200 NVL72 | 1.5x |

### 2.4 Transformer Engine v2

The second-generation Transformer Engine on Blackwell Ultra introduces:

- **NVFP4 native training support**: E2M1 format with micro-tensor scaling (block size 16, FP8 E4M3 per-block scale, FP32 per-tensor scale).
- **Random Hadamard Transforms**: 16x16 Hadamard matrices applied before quantization to smooth outlier distributions, making tensors more Gaussian-like for better FP4 representation.
- **Stochastic Rounding for Gradients**: Probabilistic rounding to avoid quantization bias in backward pass.
- **MoE-aware precision**: Gate routers and attention scores maintained at higher precision (critical — aligns with NanoSeek's `_is_fp8_eligible` design).
- **SwiGLU-aware handling**: Activation layers with heterogeneous value distributions get special treatment.

---

## 3. Vera Rubin Platform (Production Q1 2026, Partner Availability H2 2026)

### 3.1 Rubin GPU Specs

| Spec | Rubin | B300 | Improvement |
|------|-------|------|-------------|
| Process | TSMC 3nm | TSMC 4NP | New node |
| Memory | 288 GB HBM4 | 288 GB HBM3e | New standard |
| Memory BW | 13-22.2 TB/s | 8 TB/s | 1.6-2.8x |
| FP4 Inference | 50 PFLOPS | 14 PFLOPS | 3.6x |
| FP4 Training | 35 PFLOPS | ~14 PFLOPS | 2.5x |
| NVLink | 6.0 (3,600 GB/s) | 5.0 (1,800 GB/s) | 2x |
| CPU-GPU C2C | 1,800 GB/s | 900 GB/s | 2x |
| Architecture | 2 compute tiles + 2 I/O dies | Dual-die | New packaging |

### 3.2 NVLink 6.0

| Spec | NVLink 6 | NVLink 5 |
|------|----------|----------|
| Per-GPU BW | 3,600 GB/s | 1,800 GB/s |
| Switch BW | 14.4 TB/s | 14.4 TB/s |
| In-Network Compute | SHARP FP8 @ 14.4 TFLOPS | Limited |
| All-to-All for MoE | 2x throughput vs NVLink 5 | Baseline |
| NVL72 Total BW | 260 TB/s | 130 TB/s |

**Critical for MoE**: NVLink 6 integrates SHARP in-network computing directly in the switch fabric. Each switch tray delivers 14.4 TFLOPS of FP8 in-network compute, meaning all-reduce, reduce-scatter, and all-gather can execute partially within the switch, reducing GPU synchronization overhead. This is a game-changer for expert-parallel MoE.

### 3.3 Rubin NVL72 Rack

| Spec | Value |
|------|-------|
| FP4 Dense | 3.6 ExaFLOPS |
| Scale-Up BW | 260 TB/s |
| GPUs | 72 x Rubin |
| CPUs | 36 x Vera |

### 3.4 Rubin Ultra (2027 Roadmap)

- 4 compute chiplets (vs 2 in Rubin)
- 1 TB HBM4E per GPU
- ~32 TB/s memory bandwidth
- Approximately 2x Rubin performance

---

## 4. Software Stack Evolution

### 4.1 CUDA and PyTorch

**PyTorch 2.7+ (current)**:
- Native Blackwell support with pre-built CUDA 12.8 wheels
- Ships Triton 3.3 with Blackwell codegen
- `torch.compile` is the reference optimization path: Dynamo captures Python bytecode -> Inductor lowers FX graph -> Triton compiles to PTX

**torch.compile state (Aug 2025 onwards)**:
- Production-ready for training on H100/Blackwell
- Automatic kernel fusion, memory planning, and operator scheduling
- Custom Triton kernels can be used within `torch.compile` graphs
- The Inductor-Triton pipeline is the de facto standard for NVIDIA hardware optimization

**Triton on Blackwell**:
- Native Blackwell PTX generation
- FP8 tensor core instructions accessible from Triton
- Foundation for FP4 support being built out
- Day-0 support for new hardware features

### 4.2 cuDNN 9+

- **Fused SDPA (Scaled Dot-Product Attention)**: Up to 2x faster than PyTorch eager in BF16, up to 3x faster in FP8
- **Score modification**: `set_score_mod` and `set_score_mod_bprop` allow custom attention score transformations (useful for ALiBi, relative position bias)
- **Block masking**: `set_block_mask` for efficient causal/sparse attention patterns
- **Mixed precision matmuls**: Native support for mixed input precision
- **Blackwell-optimized kernels**: Attention kernels tuned for 5th-gen tensor cores

### 4.3 NCCL Evolution

| Version | Key Features |
|---------|-------------|
| 2.24 | RAS subsystem for crash diagnosis, User Buffer registration for NVSwitch/IB SHARP, NIC Fusion |
| 2.25 | Blackwell platform support (compatibility release) |
| 2.26 | PAT optimizations (parallel tree computation), implicit launch ordering, GPU kernel profiler, QoS support |
| 2.27 | Fast inference mode, resilient training primitives |
| 2.28 | Device API with 3 modes: LSA (NVLink/PCIe), Multimem (NVLink SHARP), GIN (network RDMA) |

**NCCL EP (Expert Parallelism)**: Purpose-built MoE communication library on top of NCCL Device API:
- Unified `ncclEpDispatch` and `ncclEpCombine` primitives
- Low-Latency (LL) mode: Direct all-to-all RDMA+NVLink for small batches (1-128 tokens)
- High-Throughput (HT) mode: Optimized for training and prefill
- Double-buffered communication for overlapping dispatch/combine phases

**Relevance to NanoSeek**: When scaling to multi-node expert parallelism, NCCL EP provides exactly the primitives needed for MoE all-to-all token routing.

### 4.4 Transformer Engine Library

Current version: 2.12.0
- FP8 and FP4 support via Python API
- Integrated with PyTorch, JAX, and PaddlePaddle
- Automatic mixed-precision management
- Per-tensor and per-block scaling strategies
- MoE-specific optimizations in roadmap (DeepSeek-V3, Qwen3 support)

### 4.5 Megatron-LM

- Comprehensive MoE roadmap (Q3-Q4 2025): DeepSeek-V3 architecture support, advanced parallelism, FP8 optimizations
- Dynamic Context Parallelism: Up to 1.48x speedup for variable-length sequence training
- Blackwell performance enhancements integrated

---

## 5. Hardware-Software Co-Design Trends

### 5.1 FP4 Training — How It Changes the Game

**NVIDIA's NVFP4 Paper Results (March 2026)**:
- Trained LLMs end-to-end in NVFP4 with near-FP8 accuracy
- MMLU-Pro 5-shot: NVFP4 62.58% vs FP8 62.62% (negligible gap)
- Validation loss gap stays under 1% throughout training, rising to ~1.5% only near LR decay
- Throughput gains: Up to 1.59x over BF16
- **Exception**: Coding benchmarks show slight degradation in NVFP4
- Requires selective BF16 layers for convergence stability (gate routers, embeddings — same philosophy as NanoSeek's `_is_fp8_eligible`)

**Key techniques for FP4 training**:
1. **Micro-tensor scaling**: Blocks of 16 values share an FP8 scale factor
2. **Random Hadamard transforms**: 16x16 Hadamard applied pre-quantization to smooth outliers
3. **Stochastic rounding**: For gradient quantization to avoid bias
4. **Selective precision**: Embeddings, gate routers, and normalization layers stay BF16/FP32

**Implication for NanoSeek**: The FP4 path is architecturally very similar to the FP8 path already in `fp8.py`. The same `_is_fp8_eligible` logic applies. When B300 becomes available on RunPod, upgrading from FP8 to FP4 would roughly double training throughput with minimal accuracy loss.

### 5.2 Structured Sparsity (2:4)

Blackwell supports hardware-accelerated 2:4 structured sparsity:
- 2 of every 4 consecutive weights must be zero
- Doubles effective throughput at each precision level
- FP4 + sparsity = 18,000 TFLOPS (vs 9,000 dense)
- FP8 + sparsity = 9,000 TFLOPS (vs 4,500 dense)

**Current state**: Sparsity-aware training requires specialized pruning schedules. Not yet mainstream for pre-training but increasingly used for fine-tuning and inference. Research on training with sparsity from scratch (e.g., SR-STE, STEP) is active but not production-ready for LLM pre-training.

### 5.3 In-Network Computing for MoE

The evolution of MoE communication on NVIDIA platforms:

| Generation | Capability | Impact |
|-----------|-----------|--------|
| Hopper (NVLink 4) | 900 GB/s per GPU, basic all-to-all | MoE feasible but communication-bound |
| Blackwell (NVLink 5) | 1,800 GB/s, NVSwitch all-to-all | MoE practical at scale |
| Vera Rubin (NVLink 6) | 3,600 GB/s, SHARP in-switch FP8 compute | MoE communication partially offloaded to network |

NCCL EP's `ncclEpDispatch`/`ncclEpCombine` primitives abstract this hardware evolution, providing a stable API for MoE communication regardless of the underlying interconnect generation.

### 5.4 Memory-Centric Computing Trends

| Trend | Current (Blackwell) | Next (Rubin) | Impact |
|-------|---------------------|--------------|--------|
| HBM Generation | HBM3e | HBM4 | 2x interface width |
| Capacity per GPU | 192-288 GB | 288 GB (1 TB Ultra) | Larger models per GPU |
| Bandwidth | 8 TB/s | 13-22 TB/s | Memory wall pushed back |
| CPU-GPU Coherence | NVLink-C2C 900 GB/s | NVLink-C2C 1,800 GB/s | Unified memory practical |
| Compute-to-BW Ratio | ~1,125 FP8 FLOPS/byte | ~1,600 FP4 FLOPS/byte | Increasingly compute-bound |

The trend is clear: memory bandwidth is growing slower than compute, making memory-efficient techniques (MLA's 23x KV compression, FP8/FP4 training, activation checkpointing) increasingly important.

---

## 6. Cloud Availability and Pricing (March 2026)

### 6.1 Current GPU Cloud Pricing

| GPU | RunPod | Lambda | CoreWeave | Notes |
|-----|--------|--------|-----------|-------|
| B200 (192GB) | $4.99/hr | $4.99/hr | ~$5.50/hr | 180GB usable VRAM on RunPod |
| H200 (141GB) | $3.49/hr | - | $3.89/hr | Best value for memory-bound workloads |
| H100 SXM (80GB) | $2.49/hr | $2.49/hr | $2.06/hr | Still most cost-effective for training |
| H100 PCIe (80GB) | $1.99/hr | - | - | Lower NVLink bandwidth |
| A100 SXM (80GB) | $1.19/hr | - | - | No FP8 tensor cores |
| A100 PCIe (80GB) | $0.99/hr | - | - | Budget option, no FP8 |

### 6.2 Cost Analysis for NanoSeek Training

**Ablation scale (~410M active, 8.2B tokens)**:

| GPU | Est. Hours | Cost/GPU | GPUs Needed | Total Cost |
|-----|-----------|----------|-------------|------------|
| B200 | ~4-5 hrs | $4.99/hr | 1 | ~$20-25 |
| H100 SXM | ~8-10 hrs | $2.49/hr | 1 | ~$20-25 |
| A100 SXM | ~15-20 hrs | $1.19/hr | 1 | ~$18-24 |

**1B graduation run (~1.08B active, 22B tokens)**:

| GPU | Est. Hours | Cost/GPU | GPUs Needed | Total Cost |
|-----|-----------|----------|-------------|------------|
| B200 + FP8 | ~8-10 hrs | $4.99/hr | 4-8 | ~$160-400 |
| H100 SXM + FP8 | ~12-14 hrs | $2.49/hr | 8 | ~$240-280 |

### 6.3 Recommendations for NanoSeek

1. **Phase 3 Gate 1 + HP Search**: Use H100 SXM on RunPod ($2.49/hr). FP8 works on H100 (Hopper has FP8 tensor cores). Best cost-effectiveness for iterative experimentation.

2. **Phase 4 Graduation Run**: Consider B200 ($4.99/hr) with `--fp8`. The 2.3x speedup over H100 at ~2x the price means roughly similar total cost but faster wall-clock time.

3. **Do NOT use A100/A6000 with `--fp8`**: No FP8 tensor cores on Ampere. The `fp8.py` code correctly detects this and falls back to BF16.

4. **B300 when available on cloud**: Monitor RunPod/Lambda for B300 availability. The 288GB HBM3e + 1.5x compute would allow larger batch sizes and potentially FP4 training.

---

## 7. Implications for NanoSeek Architecture

### 7.1 What's Already Aligned

- **FP8 framework** (`fp8.py`): Directly maps to Blackwell/Hopper tensor cores. E4M3 forward / E5M2 backward is the standard NVIDIA approach.
- **MoE-aware FP8 exclusions**: Gate router protection in `_is_fp8_eligible` matches NVIDIA's own Transformer Engine recommendations.
- **MLA compression**: The 23x KV compression becomes even more valuable as compute-to-bandwidth ratio increases.
- **torch._scaled_mm**: Uses cuBLAS native path, will automatically benefit from Blackwell optimizations.

### 7.2 Future Optimization Opportunities

1. **FP4 training path**: When B300 is available on cloud, extend `fp8.py` to support NVFP4. Architecture: same `_is_fp8_eligible` logic, add Hadamard transform pre-quantization, stochastic rounding for gradients.

2. **cuDNN fused attention**: Replace manual attention implementation with cuDNN SDPA for up to 3x speedup in FP8. The `set_score_mod` API supports custom attention patterns.

3. **NCCL EP for expert parallelism**: When scaling beyond single-node, use `ncclEpDispatch`/`ncclEpCombine` instead of manual all-to-all for MoE routing.

4. **Structured sparsity**: Consider 2:4 pruning for inference/fine-tuning after pre-training. Not recommended during pre-training.

5. **Triton custom kernels**: Write MoE dispatch kernel in Triton for optimal Blackwell performance, usable within `torch.compile`.

---

## Sources

- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [NVIDIA B200 Specs — Jarvis Labs](https://jarvislabs.ai/ai-faqs/nvidia-b200-specs)
- [B200 Technical Analysis — ServerSimply](https://www.serversimply.com/blog/technical-analysis-of-the-blackwell-b200)
- [GB200 NVL72 — NVIDIA](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)
- [Inside NVIDIA Blackwell Ultra — NVIDIA Developer Blog](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)
- [NVIDIA B300 Blackwell Ultra — Spheron](https://www.spheron.network/blog/nvidia-b300-blackwell-ultra-guide/)
- [B300 Announcement — Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/nvidia-announces-blackwell-ultra-b300-1-5x-faster-than-b200-with-288gb-hbm3e-and-15-pflops-dense-fp4)
- [NVIDIA Vera Rubin Platform — NVIDIA Developer Blog](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/)
- [Vera Rubin NVL72 — NVIDIA](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/)
- [Vera Rubin Specs — Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/nvidias-vera-rubin-platform-in-depth-inside-nvidias-most-complex-ai-and-hpc-platform-to-date)
- [Transformer Engine 2.12.0 — NVIDIA](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- [Transformer Engine GitHub — NVIDIA](https://github.com/NVIDIA/TransformerEngine)
- [Per-Tensor and Per-Block FP8 Scaling — NVIDIA Blog](https://developer.nvidia.com/blog/per-tensor-and-per-block-scaling-strategies-for-effective-fp8-training/)
- [Pretraining LLMs with NVFP4 — arXiv](https://arxiv.org/html/2509.25149v1)
- [NVFP4 Low-Precision Training — NVIDIA Blog](https://developer.nvidia.com/blog/using-nvfp4-low-precision-model-training-for-higher-throughput-without-losing-accuracy)
- [NVFP4 Training Efficiency — Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/nvidia-details-efficiency-of-the-nvfp4-format-for-llm-training-new-paper-reveals-how-nvfp4-offers-benefits-over-fp8-and-bf16)
- [cuDNN 9 Attention — NVIDIA Blog](https://developer.nvidia.com/blog/accelerating-transformers-with-nvidia-cudnn-9/)
- [cuDNN Attention Frontend](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html)
- [NCCL 2.24 — NVIDIA Blog](https://developer.nvidia.com/blog/networking-reliability-and-observability-at-scale-with-nccl-2-24/)
- [NCCL 2.26 — NVIDIA Blog](https://developer.nvidia.com/blog/improved-performance-and-monitoring-capabilities-with-nvidia-collective-communications-library-2-26)
- [NCCL EP Expert Parallelism — arXiv](https://arxiv.org/abs/2603.13606)
- [Doubling All2All with NCCL 2.12 — NVIDIA Blog](https://developer.nvidia.com/blog/doubling-all2all-performance-with-nvidia-collective-communication-library-2-12/)
- [NVIDIA SHARP In-Network Computing — NVIDIA Blog](https://developer.nvidia.com/blog/advancing-performance-with-nvidia-sharp-in-network-computing/)
- [Triton on Blackwell — NVIDIA Blog](https://developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell-boosts-ai-performance-and-programmability/)
- [PyTorch 2.7 Release](https://pytorch.org/blog/pytorch-2-7/)
- [torch.compile State Aug 2025 — ezyang's blog](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/)
- [Blackwell MLPerf Training Results — NVIDIA Blog](https://developer.nvidia.com/blog/nvidia-blackwell-architecture-sweeps-mlperf-training-v5-1-benchmarks/)
- [RunPod B200 Pricing](https://www.runpod.io/gpu-models/b200)
- [RunPod GPU Pricing](https://www.runpod.io/gpu-pricing)
- [GPU Cloud Pricing Comparison 2026 — Spheron](https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/)
- [Lambda AI Pricing](https://lambda.ai/pricing)
- [CoreWeave Pricing](https://www.coreweave.com/pricing)
- [B200 Cloud Pricing Comparison](https://getdeploying.com/gpus/nvidia-b200)
- [NVLink Evolution — FiberMall](https://www.fibermall.com/blog/nvidia-nvlink-and-nvswitch-evolution.htm)
- [Rubin Full Production — Introl](https://introl.com/blog/nvidia-rubin-full-production-ces-2026-ai-infrastructure)
