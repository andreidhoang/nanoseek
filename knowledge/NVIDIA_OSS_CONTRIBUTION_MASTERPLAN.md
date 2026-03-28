# NVIDIA Open-Source Contribution Masterplan
## A Principal AI Engineer's Strategic Playbook for 2026 and Beyond

**Author**: NanoSeek Research Lab
**Date**: 2026-03-27
**Status**: Strategic Plan — Ready for Execution
**Perspective**: Top-tier Principal AI Engineer at NVIDIA contributing to frontier AI

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [The Landscape: NVIDIA's Open-Source AI Stack 2026](#2-the-landscape)
3. [GPU Hardware Reality: Blackwell → Vera Rubin](#3-gpu-hardware-reality)
4. [What Frontier Labs Actually Use (and Don't)](#4-what-frontier-labs-actually-use)
5. [The 80/20 Analysis: Maximum Leverage Open-Source Projects](#5-the-8020-analysis)
6. [Tier-1 Contributions: The 20% That Delivers 80% Value](#6-tier-1-contributions)
7. [Tier-2 Contributions: High-Impact Gap Fillers](#7-tier-2-contributions)
8. [Tier-3 Contributions: Ecosystem Multipliers](#8-tier-3-contributions)
9. [NanoSeek → NVIDIA Upstream Technical Mapping](#9-nanoseek-nvidia-upstream-mapping)
10. [Execution Roadmap: 12-Month Plan](#10-execution-roadmap)
11. [The Frontier AI Thesis: Where LLM Development Goes 2026+](#11-frontier-ai-thesis)
12. [Appendix: GPU Specs, GitHub Issues, References](#12-appendix)

---

## 1. Executive Summary

### The Core Insight

NVIDIA's software stack serves the **"long tail"** of thousands of teams who cannot do custom kernel engineering. **Frontier labs have outgrown it.** DeepSeek builds custom PTX. Meta uses torchtitan. Google is on TPU/JAX entirely. The highest-impact contributions **close the gap** between "use Megatron out of the box" and "DeepSeek-level custom" — making frontier techniques accessible to the entire community.

### The Three Laws of High-Impact NVIDIA OSS Contribution

1. **Hardware-software co-design wins**: Contributions that unlock new GPU capabilities (FP4, B200 tensor cores, NVLink 5) have compounding returns
2. **Frontier technique democratization**: Taking what DeepSeek/Meta build in-house and making it work in Megatron/NeMo for everyone
3. **The training-inference bridge**: Techniques that improve both training efficiency AND inference speed (MLA, MTP, MoE) are 2x leverage

### Top 3 Highest-Impact Contributions (The 80/20)

| Rank | Contribution | Target Repo | Impact | Effort |
|------|-------------|-------------|--------|--------|
| **1** | MoE-Aware Mixed Precision Policies | TransformerEngine | Unblocks FP8/FP4 for all MoE models | ~2 weeks |
| **2** | Expert Parallelism in FSDP (Issue #1781) | Megatron-Core | Unblocks PyTorch-native MoE training | ~4-6 weeks |
| **3** | FP4 Training Pipeline Fix | TransformerEngine (#1701, #2352) | Justifies B200 for training (not just inference) | ~4-8 weeks |

---

## 2. The Landscape: NVIDIA's Open-Source AI Stack 2026

### The Full Stack (Ranked by Strategic Importance)

```
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                             │
│  NeMo-Guardrails (5.8k★) │ Safety, PII, jailbreak prevention   │
├─────────────────────────────────────────────────────────────────┤
│                    INFERENCE LAYER                               │
│  TensorRT-LLM (13.2k★) │ Serving, quantization, speculative    │
│  AutoDeploy (Beta) │ PyTorch → optimized inference graphs       │
├─────────────────────────────────────────────────────────────────┤
│                    POST-TRAINING LAYER                           │
│  NeMo-RL (NEW) │ GRPO, DAPO, Ray backbone                      │
│  NeMo-Aligner (DEPRECATED May 2025) │ PPO, DPO legacy          │
├─────────────────────────────────────────────────────────────────┤
│                    TRAINING LAYER                                │
│  Megatron-Core (15.5k★) │ 5D parallelism, MLA+MoE+MTP         │
│  NeMo Automodel (NEW) │ HF-native, FSDP2, DTensor              │
│  Megatron-Bridge │ HF ↔ Megatron checkpoint conversion          │
├─────────────────────────────────────────────────────────────────┤
│                    PRECISION LAYER                               │
│  TransformerEngine (3k★) │ FP8/FP4, MXFP8, delayed/block scale │
├─────────────────────────────────────────────────────────────────┤
│                    DATA LAYER                                    │
│  NeMo-Curator (243★) │ GPU-accelerated curation, Ray migration  │
├─────────────────────────────────────────────────────────────────┤
│                    KERNEL LAYER                                  │
│  CUTLASS 4.0 (9.5k★) │ CuTe DSL, FP8/FP4 GEMMs, grouped GEMM │
│  FlashAttention-3 │ Hopper-optimized, FP8 attention             │
│  DeepGEMM (external) │ DeepSeek's FP8 GEMM, converging w/ CUTLASS│
└─────────────────────────────────────────────────────────────────┘
```

### Project Health Matrix

| Project | Stars | Activity | Community | Contribution Barrier |
|---------|-------|----------|-----------|---------------------|
| **Megatron-Core** | 15.5k | Very High | Large, active | High (complex codebase) |
| **TensorRT-LLM** | 13.2k | Very High | Large | Very High (C++/CUDA) |
| **CUTLASS** | 9.5k | High | Specialized | Very High (kernel dev) |
| **NeMo-Guardrails** | 5.8k | High | Growing | Low (Python) |
| **TransformerEngine** | 3k | High | Moderate | Medium-High (CUDA + Python) |
| **NeMo-RL** | New | Active | Small | Medium (Python + Ray) |
| **NeMo-Curator** | 243 | Active | Small | Medium (Ray + RAPIDS) |
| **NeMo Automodel** | New | Active | Very Small | Medium (PyTorch DTensor) |

---

## 3. GPU Hardware Reality: Blackwell → Vera Rubin

### Current Generation: Blackwell (B200, Available Now)

| Spec | H100 SXM | B200 | Ratio |
|------|----------|------|-------|
| FP8 Tensor TFLOPS | 1,979 | 4,500 | **2.3x** |
| FP4 Tensor TFLOPS | N/A | 9,000 | **New** |
| HBM Capacity | 80 GB HBM3 | 192 GB HBM3e | **2.4x** |
| Memory BW | 3.35 TB/s | 8 TB/s | **2.4x** |
| NVLink BW | 900 GB/s (NVL4) | 1,800 GB/s (NVL5) | **2x** |
| TDP | 700W | 1000W | 1.4x |
| RunPod $/hr | $2.49 | $4.99 | 2x |

**Key Insight**: B200's 2.3x FP8 throughput at 2x cost = **15% better cost-efficiency** than H100 for compute-bound workloads. The 2.4x memory enables fitting larger models without model parallelism.

### Blackwell Ultra (B300, GA March 2026)

| Spec | B200 | B300 (Blackwell Ultra) |
|------|------|----------------------|
| HBM Capacity | 192 GB HBM3e | **288 GB HBM3e** (+50%) |
| Compute | Baseline | **~1.5x** over B200 |
| Attention Acceleration | 1x | **2x** (hardware attention layer) |
| FP4 Training | Experimental | **Production-ready** (Transformer Engine v2) |
| Micro-tensor scaling | Per-tensor | **Blocks of 16, FP8 per-block scale** |

**Critical**: B300's **2x attention acceleration in hardware** + FP4 production readiness fundamentally changes the training cost equation. MLA + FP4 on B300 would be the optimal combination.

### Next Generation: Vera Rubin (Partner Availability H2 2026)

| Spec | B200 | Vera Rubin |
|------|------|-----------|
| Process | 4nm | **3nm** |
| HBM | HBM3e 8 TB/s | **HBM4 13-22 TB/s** (1.6-2.75x) |
| NVLink | NVL5 1,800 GB/s | **NVL6 3,600 GB/s** (2x) |
| In-Network Compute | No | **SHARP FP8 in switch fabric** |

**Game-Changer for MoE**: Vera Rubin's **SHARP in-network FP8 compute** means the NVSwitch itself can perform FP8 reductions during all-to-all expert dispatch. This nearly eliminates the communication overhead of expert parallelism — the #1 scaling bottleneck for MoE models.

### What This Means for Contributions

1. **FP4 training support** is the highest-urgency gap — B300 ships with FP4 tensor cores but TransformerEngine's FP4 training pipeline is broken (Issues #1701, #2352, #2171)
2. **MoE dispatch optimization** gets 2x more important with Vera Rubin's in-network compute — NCCL EP's `ncclEpDispatch`/`ncclEpCombine` needs framework integration
3. **MLA kernels** benefit from B300's 2x attention hardware acceleration — custom MLA CUDA kernels should target Blackwell Ultra instruction set
4. **Memory capacity** (288GB B300, HBM4 Vera Rubin) makes larger expert counts feasible — 128 or 256 experts become practical

---

## 4. What Frontier Labs Actually Use (and Don't)

### The Uncomfortable Truth

| Lab | Training Stack | NVIDIA SW Used | Custom Components |
|----|---------------|---------------|-------------------|
| **DeepSeek** | Custom everything | Raw CUDA/PTX | FlashMLA, DeepEP (PTX assembly), DualPipe, DeepGEMM, custom FP8 |
| **Meta** | torchtitan + FSDP2 | Minimal | Own parallelism, own optimizer, own data pipeline |
| **Google** | TPU + JAX + XLA | **None** | Entire stack is non-NVIDIA |
| **Qwen (Alibaba)** | PAI-Megatron fork | Fork only | Heavy modifications to Megatron |
| **Kimi (Moonshot)** | Megatron-LM | **Yes, closest** | Custom RL pipeline |
| **MiniMax** | Custom | Minimal | CISPO algorithm, custom pipeline |
| **GLM (Zhipu)** | **Huawei Ascend + MindSpore** | **None** | Left NVIDIA entirely |

**The Pattern**: Only Moonshot/Kimi and smaller labs use NVIDIA's stack as-is. Every frontier lab that trains >100B models has forked or replaced it. **The gap between "stock Megatron" and "frontier custom" is the contribution opportunity.**

### Why Labs Leave (Root Causes)

1. **Performance ceiling**: Megatron's abstractions add 10-20% overhead vs. custom PTX
2. **Architecture lag**: New techniques (MLA, MTP, auxiliary-loss-free balancing) take 6-12 months to appear in Megatron
3. **Rigidity**: Megatron's model builder pattern makes novel architectures hard to implement
4. **FP8/FP4 maturity**: TransformerEngine's delayed scaling doesn't match DeepSeek's dynamic block-wise approach
5. **MoE dispatch**: All-to-all patterns in Megatron are slower than DeepSeek's custom EP (which bypasses CUDA to use PTX directly)

### The Strategic Implication

A principal engineer's contributions should target **closing these specific gaps**. Every gap closed = one fewer reason for the next lab to fork Megatron.

---

## 5. The 80/20 Analysis: Maximum Leverage Open-Source Projects

### Tier 1: Core Training (80% of Value)

| Project | Why It's Core | 80/20 Area |
|---------|--------------|------------|
| **Megatron-Core** | The training framework everyone either uses or forks | MoE parallelism + FSDP integration |
| **TransformerEngine** | The precision layer everything depends on | FP4 training + MoE-aware policies |
| **CUTLASS** | The kernel layer everything compiles to | Grouped GEMM for MoE + FP4 kernels |

### Tier 2: Post-Training + Inference (15% of Value)

| Project | Why It Matters | 80/20 Area |
|---------|---------------|------------|
| **NeMo-RL** | RL alignment is the new differentiator | GRPO + MoE stabilization |
| **TensorRT-LLM** | Deployment is where models create value | MLA inference optimization |

### Tier 3: Data + Safety (5% of Value)

| Project | Why It Matters | 80/20 Area |
|---------|---------------|------------|
| **NeMo-Curator** | Data quality = model quality | Semantic dedup at scale |
| **NeMo-Guardrails** | Deployment requirement | MoE-aware safety |

### The Core 3 Projects (Where 80% of Effort Should Go)

```
 MEGATRON-CORE          TRANSFORMER-ENGINE          CUTLASS
 ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
 │ EP in FSDP   │       │ FP4 Training │       │ Grouped GEMM │
 │ MoE routing  │  ───► │ MoE-aware    │  ───► │ MLA kernels  │
 │ MTP pipeline │       │ Eval escape  │       │ FP4 GEMMs    │
 │ Data pipeline│       │ Block scaling│       │ CuTe DSL     │
 └──────────────┘       └──────────────┘       └──────────────┘
```

---

## 6. Tier-1 Contributions: The 20% That Delivers 80% Value

### Contribution #1: MoE-Aware Mixed Precision Policies for TransformerEngine

**The Problem**: Every MoE lab independently discovers that gate routers must be protected from FP8 quantization. Sigmoid logits are extremely sensitive to quantization noise — a 0.01 perturbation can flip expert selection for borderline tokens. There is no first-class mechanism in TransformerEngine to express "these layers stay BF16, these get FP8, and here's why."

**NanoSeek Discovery That Proves This**: MLA lora ranks (275, 90, 440, 143) are not divisible by 16 → MLA projections must stay BF16. Gate router must stay BF16. Embeddings must stay BF16. This is universal across all MoE models.

**What to Build**:
```python
# Proposed API for TransformerEngine
class MoEPrecisionPolicy:
    """Architecture-aware FP8/FP4 conversion policy."""

    def __init__(self):
        self.rules = []

    def protect_gate_routers(self):
        """Gate routers always BF16 — routing precision is sacred."""
        self.rules.append(LayerRule(
            match=lambda m: 'gate' in m.name or 'router' in m.name,
            precision='bf16',
            reason='routing_precision'
        ))

    def protect_misaligned_dims(self, alignment=16):
        """Layers with dims not divisible by alignment stay BF16."""
        self.rules.append(DimAlignmentRule(alignment=alignment))

    def protect_embeddings(self):
        """Embedding layers always BF16 — token representation precision."""
        self.rules.append(LayerRule(
            match=lambda m: isinstance(m, nn.Embedding),
            precision='bf16'
        ))

    def eval_escape_hatch(self):
        """Context manager to temporarily revert all FP8 → BF16 for eval."""
        # NanoSeek's disable_fp8() pattern
```

**Impact**: Every team training an MoE model with FP8 would use this. Mixtral, DeepSeek, Qwen-MoE, DBRX, Snowflake Arctic — all benefit.

**Effort**: ~2 weeks. Low risk. Clear value proposition.

**GitHub Issues**: Related to TransformerEngine architecture-awareness gaps.

---

### Contribution #2: Expert Parallelism in Megatron-FSDP (Issue #1781)

**The Problem**: PyTorch FSDP2 is where the ecosystem is heading (Meta uses it, torchtitan uses it, NeMo Automodel uses it). Megatron-Core now supports FSDP via `MegatronFSDP`. But **Expert Parallelism (EP) is completely missing from the FSDP path**. This means you cannot train MoE models using FSDP — only using the legacy `DistributedDataParallel` + `ColumnParallelLinear` path.

**Why This Matters**: Without EP in FSDP, teams must choose between "modern PyTorch-native training" and "MoE support." This is a false choice that blocks MoE adoption.

**What to Build**:
```
MegatronFSDP + Expert Parallelism Design:

1. Expert Sharding Strategy:
   - Each expert's parameters handled by a separate FSDP unit
   - DTensor-based sharding across EP dimension
   - All-to-all token dispatch before/after expert FSDP units

2. Communication Pattern:
   - FSDP all-gather for parameter materialization (per-expert)
   - EP all-to-all for token routing (across expert groups)
   - These two communication patterns must be overlapped

3. Integration with MoE Parallel Folding:
   - Decouple attention FSDP config from expert FSDP config
   - Allow different sharding strategies per component
```

**Impact**: Unblocks the entire FSDP-MoE training path. Every team moving to FSDP2 (which is everyone following PyTorch's direction) needs this.

**Effort**: ~4-6 weeks. Medium risk. Requires deep understanding of FSDP2 internals + MoE dispatch patterns.

**GitHub Issue**: [NVIDIA/Megatron-LM#1781](https://github.com/NVIDIA/Megatron-LM/issues/1781)

---

### Contribution #3: Fix FP4 Training Pipeline (TransformerEngine)

**The Problem**: Blackwell B200 GPUs have 9,000 TFLOPS FP4 tensor cores (2x over FP8). B300 (March 2026) makes FP4 production-ready with micro-tensor scaling. But TransformerEngine's FP4 **training** pipeline is broken/incomplete:

- Issue #1701: FP4 training support is experimental, not production-ready
- Issue #2352: FP4 parameter gathering broken in distributed training
- Issue #2171: Related FP4 integration issues
- NVFP4 works for **inference** (TensorRT-LLM) but not reliably for **training**

**Why This Matters**: Without working FP4 training, there's **no reason to upgrade from H100 to B200/B300 for training**. The 2x theoretical speedup from FP4 is stranded.

**NVIDIA's Own Results** (from their FP4 training paper):
- NVFP4 matches FP8 within ~1% on validation loss
- 62.58% vs 62.62% on MMLU-Pro (negligible degradation)
- Up to **1.59x throughput** over BF16
- Uses same gate-router/embedding exclusion philosophy as NanoSeek's `_is_fp8_eligible`

**What to Build**:
```
FP4 Training Pipeline:

1. NVFP4 Linear Layer:
   - E2M1 format (2 exponent, 1 mantissa bits)
   - Block-wise scaling (1 FP8 scale per 16 values)
   - Forward: FP4 weights × FP8 activations → BF16 accumulator
   - Backward: FP8 gradients (E5M2) with dynamic scaling

2. Mixed Precision Recipe:
   - Master weights: FP32
   - Forward weights: NVFP4
   - Activations: FP8 (E4M3)
   - Gradients: FP8 (E5M2) or BF16
   - Optimizer states: FP32 (critical for convergence)

3. Distributed Training:
   - FP4 all-gather for parameter distribution
   - BF16 reduce-scatter for gradients
   - Proper handling of scaling factors in communication
```

**Impact**: Justifies the entire Blackwell upgrade path for training. NVIDIA's hardware team ships FP4 tensor cores; the software team needs to make them usable.

**Effort**: ~4-8 weeks. Higher risk (numerical stability research needed). Highest strategic value to NVIDIA.

**GitHub Issues**: [TransformerEngine#1701](https://github.com/NVIDIA/TransformerEngine/issues/1701), [#2352](https://github.com/NVIDIA/TransformerEngine/issues/2352)

---

## 7. Tier-2 Contributions: High-Impact Gap Fillers

### Contribution #4: GRPO + MoE RL Stabilization for NeMo-RL

**The Problem**: NeMo-RL has GRPO and DAPO but lacks MoE-specific stabilization. When you run RL on an MoE model, three things break:

1. **Routing collapse**: Reward signal overwhelms the gate router, collapsing to 2-3 experts
2. **Off-policy divergence**: GRPO's group sampling creates stale data that's more harmful with MoE's discrete routing
3. **KL explosion**: Standard KL estimator (`E[log(pi/pi_ref)]`) has high variance with MoE's sparse activations

**NanoSeek's Proven Solutions** (from `grpo_trainer.py`):
- **Keep Routing**: Freeze all gate/router parameters during RL → prevents routing collapse
- **Off-policy masking**: Skip sequences where `|rho - 1| > 0.3` → prevents stale data corruption
- **Unbiased KL estimator**: `E[pi/pi_ref - 1 - log(pi/pi_ref)]` → always non-negative, lower variance
- **Group-relative advantage normalization**: Per-prompt advantage normalization

**Impact**: As MoE models become standard (Mixtral, DeepSeek, Qwen-MoE), every RL alignment pipeline needs MoE-aware stabilization. This makes NeMo-RL the first production framework with proper MoE RL support.

**Effort**: ~3-4 weeks. Low risk. Clear implementation path.

---

### Contribution #5: MLA Native Support Enhancement in Megatron-Core

**The Problem**: Megatron-Core added MLA support (claimed "native"), but the current implementation may lack:
- Proper absorb-mode inference path (the key efficiency win)
- Custom CUDA kernels for decomposed attention (`q_nope @ W_UK @ c_kv^T + q_pe @ k_pe^T`)
- Integration with FlashAttention for the training path
- KV cache format adaptation (`(c_kv, k_pe_rotated)` instead of full K/V)

**NanoSeek's Implementation** (model.py lines 184-502):
- Dual-mode attention: "naive" training path (expand to full K/V, use SDPA) and "absorb" inference path (operate on compressed latents)
- Weight absorption via zero-copy `.view()` — no parameter fusion needed
- 23x KV cache compression
- Explicit decoupled RoPE factorization

**What to Contribute**:
- Validate and benchmark Megatron-Core's MLA vs. NanoSeek's implementation
- Contribute absorb-mode optimizations if missing
- Write CUTLASS kernel for the fused absorb-attention pattern
- Document MLA + FP8 interaction (dimension alignment constraints)

**Effort**: ~4-6 weeks. Medium risk. Requires kernel development.

---

### Contribution #6: Auxiliary-Loss-Free MoE Routing for Megatron-Core

**Current State**: Megatron-Core claims support for DeepSeek-V3-style bias-based routing, but the implementation may be incomplete. NanoSeek's implementation includes:

- **Sigmoid scoring** (not softmax) — fundamentally different routing math
- **Group-based top-k** — 8 groups of 8, hierarchical selection
- **Non-gradient bias buffer** — decoupled from optimization
- **gamma_freeze_ratio = 0.95** — bias updates freeze at 95% of training

**What to Contribute**:
- Validate completeness of Megatron's implementation
- Add group-based routing if missing (critical for communication efficiency)
- Implement gamma_freeze scheduling mechanism
- Benchmark against standard auxiliary loss (load balance vs. model quality)

**Effort**: ~2-3 weeks. Low risk.

---

### Contribution #7: MuonAdamW + Polar Express Optimizer

**What NanoSeek Implements** (optim.py, 556 lines):
- Hybrid Muon (2D matrix params) + AdamW (embeddings/scalars)
- Polar Express orthogonalization (5-iter polynomial, from Amsel et al. 2505.16932)
- NorMuon variance reduction (per-neuron adaptive LR)
- Cautious weight decay (only where `g * theta >= 0`)
- `@torch.compile(dynamic=False, fullgraph=True)` fused kernels
- Distributed version with ZeRO-2 style sharding

**Why It Matters**: Muon has shown 1.5-2x convergence speedup over AdamW in recent experiments. Polar Express eliminates the expensive Newton-Schulz iteration. This is bleeding-edge optimizer research, production-ready.

**Target**: NeMo/Megatron optimizer registry. Makes Muon accessible to everyone.

**Effort**: ~3-4 weeks. Medium risk (needs validation at scale).

---

## 8. Tier-3 Contributions: Ecosystem Multipliers

### Contribution #8: BOS-Aligned Best-Fit Packing + FIM for Megatron Data Pipeline

NanoSeek's dataloader implements best-fit document packing (1000-doc search buffer, crop shortest when nothing fits) achieving 100% utilization with FIM (10% PSM) deterministically seeded for checkpoint resumability.

**Effort**: ~1-2 weeks. Low risk. Low barrier.

### Contribution #9: Selective MoE Gradient Checkpointing

NanoSeek only checkpoints MoE layers (memory-heavy), skips dense layers and last 2 layers. More nuanced than Megatron's uniform checkpointing.

**Effort**: ~1 week. Low risk.

### Contribution #10: MTP Training-Inference Bridge for TensorRT-LLM

NanoSeek's MTP modules can be used for speculative decoding at inference (~1.4x throughput). Contributing the training-to-inference conversion path ensures MTP models trained in Megatron can serve in TensorRT-LLM.

**Effort**: ~3-4 weeks. Medium risk.

---

## 9. NanoSeek → NVIDIA Upstream Technical Mapping

### Component-by-Component Mapping

| NanoSeek Component | Lines | Target NVIDIA Repo | Target Location | Novelty |
|-------------------|-------|-------------------|-----------------|---------|
| MLA (model.py:184-502) | 318 | Megatron-Core | `megatron/core/transformer/attention.py` | High — dual-mode absorb |
| MoE Gate (model.py:505-675) | 170 | Megatron-Core | `megatron/core/transformer/moe/` | High — sigmoid+group+bias |
| MoE Layer (model.py:676-891) | 215 | Megatron-Core | `megatron/core/transformer/moe/` | Medium — batched bmm |
| MTP (model.py:893-1218) | 325 | Megatron-Core + TRT-LLM | `megatron/core/models/` | High — concat fusion |
| FP8 (fp8.py) | 454 | TransformerEngine | `transformer_engine/pytorch/` | Medium — MoE-aware |
| MuonAdamW (optim.py) | 556 | Megatron-Core/NeMo | `megatron/core/optimizer/` | High — Polar Express |
| GRPO (grpo_trainer.py) | 483 | NeMo-RL | `nemo_rl/algorithms/` | High — MoE stabilization |
| Dataloader (dataloader.py) | ~400 | Megatron-Core | `megatron/core/datasets/` | Low — best-fit packing |

### Integration Complexity Matrix

```
                    Low Complexity          High Complexity
                    ┌─────────────────────────────────────┐
High Impact         │ MoE-Aware FP8     │ EP in FSDP      │
                    │ GRPO + MoE Stab   │ FP4 Training    │
                    │ Aux-Loss-Free Rte │ MLA Kernels     │
                    ├─────────────────────────────────────┤
Medium Impact       │ BOS Packing       │ MuonAdamW       │
                    │ Selective Ckpt    │ MTP Pipeline    │
                    │                   │ MTP→TRT Bridge  │
                    └─────────────────────────────────────┘
```

---

## 10. Execution Roadmap: 12-Month Plan

### Phase 1: Quick Wins + Credibility (Weeks 1-4)

**Goal**: Ship 2-3 PRs that establish credibility in the community

| Week | Action | Target |
|------|--------|--------|
| 1-2 | **MoE-Aware Precision Policies** — port NanoSeek's `_is_fp8_eligible` logic to TransformerEngine | TransformerEngine PR |
| 2-3 | **Eval Escape Hatch** — `disable_fp8()` context manager for TransformerEngine | TransformerEngine PR |
| 3-4 | **Selective MoE Gradient Checkpointing** — contribute to Megatron-Core | Megatron-Core PR |

**Deliverables**: 3 merged PRs, established relationships with maintainers.

### Phase 2: Core Infrastructure (Weeks 5-14)

**Goal**: Ship the two highest-impact contributions

| Week | Action | Target |
|------|--------|--------|
| 5-10 | **Expert Parallelism in FSDP** — design doc → implementation → testing | Megatron-Core PR (Issue #1781) |
| 8-14 | **FP4 Training Pipeline Fix** — debug existing issues → implement clean pipeline | TransformerEngine PRs (#1701, #2352) |
| 10-12 | **Auxiliary-Loss-Free Routing Validation** — validate and extend Megatron's implementation | Megatron-Core PR |

**Deliverables**: EP in FSDP merged, FP4 training working on B200/B300, bias-based routing validated.

### Phase 3: Post-Training + Optimization (Weeks 15-24)

**Goal**: Make NVIDIA's stack competitive for MoE RL training

| Week | Action | Target |
|------|--------|--------|
| 15-18 | **GRPO + MoE Stabilization** — port NanoSeek's 4 stabilization techniques to NeMo-RL | NeMo-RL PR |
| 19-22 | **MuonAdamW** — port optimizer with distributed support | Megatron-Core/NeMo PR |
| 22-24 | **BOS-aligned Packing + FIM** — contribute data pipeline improvements | Megatron-Core PR |

**Deliverables**: NeMo-RL has MoE-aware GRPO, Muon optimizer available, data pipeline improved.

### Phase 4: Frontier Techniques (Weeks 25-40)

**Goal**: Close the gap with DeepSeek-level custom implementations

| Week | Action | Target |
|------|--------|--------|
| 25-30 | **MLA CUTLASS Kernel** — absorb-mode fused attention using CuTe DSL | CUTLASS contribution |
| 30-36 | **MTP Training-Inference Bridge** — Megatron MTP → TensorRT-LLM speculative decoding | TRT-LLM + Megatron PR |
| 36-40 | **NCCL EP Integration** — integrate `ncclEpDispatch`/`ncclEpCombine` into Megatron | Megatron-Core PR |

**Deliverables**: Custom MLA kernel, MTP deployment path, next-gen MoE communication.

### Phase 5: Vera Rubin Readiness (Weeks 40-52)

**Goal**: Prepare the stack for next-gen hardware

| Week | Action | Target |
|------|--------|--------|
| 40-44 | **In-Network MoE Dispatch** — leverage SHARP FP8 compute in NVSwitch for EP | NCCL/Megatron |
| 44-48 | **HBM4 Memory Optimization** — adaptive tiling for 13-22 TB/s bandwidth | CUTLASS/Megatron |
| 48-52 | **256-Expert Support** — scale MoE to 256 experts with Vera Rubin's memory | Megatron-Core |

---

## 11. The Frontier AI Thesis: Where LLM Development Goes 2026+

### The Five Mega-Trends

#### 1. MoE Becomes Default Architecture
- DeepSeek V3, Qwen-MoE, Mixtral, DBRX, Snowflake Arctic — all MoE
- By end 2026, >80% of frontier models will be MoE
- **Implication**: Everything in the NVIDIA stack must be MoE-first, not MoE-compatible

#### 2. Training Precision Drops to FP4
- FP16 → BF16 → FP8 → FP4 trajectory is clear
- B300 makes FP4 production-ready with micro-tensor scaling
- **Implication**: TransformerEngine must have rock-solid FP4 training by H2 2026

#### 3. RL Post-Training Becomes the Differentiator
- Pre-training is commoditized (same data, same architectures)
- RL (GRPO, DAPO, CISPO) is where model quality diverges
- Multi-stage pipelines: distill → SFT → RL → rejection sampling → thinking fusion → DPO
- **Implication**: NeMo-RL must support the full pipeline, not just single-stage GRPO

#### 4. Training-Inference Co-Design
- MTP trains once, benefits both training (auxiliary loss) and inference (speculative decoding)
- MLA compresses KV cache for both memory-efficient training and fast inference
- **Implication**: Architecture choices must optimize the training+inference total cost, not just training

#### 5. Hardware-Aware Architecture Design
- Vera Rubin's in-network compute changes optimal MoE topology
- FP4 tensor cores change optimal layer width (must be aligned to hardware tile sizes)
- HBM4 bandwidth changes optimal batch size and sequence length
- **Implication**: Framework code must be parameterized by hardware capabilities, not hardcoded

### What a Principal AI Engineer Should Focus On

```
2026 Q1-Q2: Close the FP4 + FSDP MoE gap (Contributions #1-3)
2026 Q3:    Make RL alignment MoE-native (Contribution #4)
2026 Q4:    Prepare for Vera Rubin (Phase 5)
2027 Q1:    In-network MoE dispatch + 256-expert scaling
2027 Q2+:   Next-generation architecture primitives (Mixture of Depths, etc.)
```

---

## 12. Appendix

### A. Key GitHub Issues to Track

| Issue | Repo | Description | Status |
|-------|------|-------------|--------|
| [#1729](https://github.com/NVIDIA/Megatron-LM/issues/1729) | Megatron-LM | MoE Roadmap | Active, comprehensive |
| [#1781](https://github.com/NVIDIA/Megatron-LM/issues/1781) | Megatron-LM | EP in FSDP | Open, high priority |
| [#1878](https://github.com/NVIDIA/Megatron-LM/issues/1878) | Megatron-LM | Intra-doc masking for packing | Open |
| [#1701](https://github.com/NVIDIA/TransformerEngine/issues/1701) | TransformerEngine | FP4 training support | Open, critical |
| [#2352](https://github.com/NVIDIA/TransformerEngine/issues/2352) | TransformerEngine | FP4 param gathering | Open |
| [#2171](https://github.com/NVIDIA/TransformerEngine/issues/2171) | TransformerEngine | FP4 integration | Open |
| [#1245](https://github.com/NVIDIA/TransformerEngine/issues/1245) | TransformerEngine | RoPE precision handling | Open |

### B. GPU Cost Comparison for NanoSeek Training

| GPU | $/hr (RunPod) | FP8 TFLOPS | $/TFLOP-hr | Best For |
|-----|--------------|------------|-----------|----------|
| A6000 | $0.76 | N/A (no FP8) | N/A | Gate 1 smoke tests |
| H100 SXM | $2.49 | 1,979 | $0.00126 | HP search, ablation |
| B200 | $4.99 | 4,500 | $0.00111 | 1B graduation with `--fp8` |
| B300 | TBD | ~6,750 est | ~$0.001 est | FP4 training when available |

### C. The NVIDIA Open-Source Contribution Checklist

Before submitting any PR:

- [ ] Read the project's CONTRIBUTING.md
- [ ] Match existing code style (NVIDIA uses specific C++/Python conventions)
- [ ] Include comprehensive tests (NVIDIA repos have high test coverage requirements)
- [ ] Write benchmarks showing performance impact
- [ ] Document hardware requirements and compatibility
- [ ] Test on at least 2 GPU generations (e.g., H100 + A100 or H100 + B200)
- [ ] Include design doc for non-trivial changes (Megatron-Core requires RFCs)
- [ ] Sign NVIDIA CLA (Contributor License Agreement)
- [ ] Tag relevant maintainers in PR description

### D. Reference Architecture: NanoSeek's Proven Patterns

These patterns from NanoSeek have been validated through 124 tests and can serve as reference implementations for upstream contributions:

1. **MoE-Aware FP8** (`fp8.py:_is_fp8_eligible`): Gate router + embedding protection, dimension alignment check
2. **Dual-Mode MLA** (`model.py:MLAttention`): Train (expand K/V) vs. Absorb (compressed latent) modes
3. **Sigmoid Group Routing** (`model.py:Gate`): Non-competitive scoring + hierarchical group selection
4. **Bias-Based Balancing** (`model.py:Gate.update_bias`): Non-gradient buffer for load steering
5. **GRPO MoE Stabilization** (`grpo_trainer.py`): Keep Routing + off-policy masking + unbiased KL
6. **Polar Express** (`optim.py`): 5-iter polynomial orthogonalization replacing Newton-Schulz
7. **BOS Best-Fit Packing** (`dataloader.py`): 1000-doc search buffer with FIM deterministic seeding

### E. Sources

- [NVIDIA/Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM) — 15.5k stars
- [Megatron-Core MoE Roadmap (Issue #1729)](https://github.com/NVIDIA/Megatron-LM/issues/1729)
- [Megatron-Core MoE Docs](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)
- [Megatron-Core MLA Docs](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/multi_latent_attention.html)
- [Megatron-Core MTP Docs](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/multi_token_prediction.html)
- [DeepSeek-V3 GB200 Optimization Guide](https://github.com/NVIDIA/Megatron-LM/blob/dev/docs/discussions/deepseek-v3-gb200-optimization)
- [NVIDIA/TransformerEngine GitHub](https://github.com/NVIDIA/TransformerEngine) — 3k stars
- [TransformerEngine FP8/FP4 Primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- [Per-Tensor and Per-Block FP8 Scaling Blog](https://developer.nvidia.com/blog/per-tensor-and-per-block-scaling-strategies-for-effective-fp8-training/)
- [NVIDIA-NeMo/RL GitHub](https://github.com/NVIDIA-NeMo/RL) — NeMo-RL v0.5.0
- [NeMo-RL GRPO Blog](https://developer.nvidia.com/blog/reinforcement-learning-with-nvidia-nemo-rl-reproducing-a-deepscaler-recipe-using-grpo/)
- [NVIDIA/TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM) — 13.2k stars
- [NVIDIA/cutlass GitHub](https://github.com/NVIDIA/cutlass) — 9.5k stars, CUTLASS 4.0 with CuTe DSL
- [DeepSeek FlashMLA](https://github.com/deepseek-ai/FlashMLA)
- [DeepSeek DeepEP](https://github.com/deepseek-ai/DeepEP) — PTX-level expert parallelism
- [DeepSeek DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) — FP8 GEMM library
- [NVIDIA-NeMo/Curator GitHub](https://github.com/NVIDIA-NeMo/Curator) — Ray migration
- [NVIDIA-NeMo/Guardrails GitHub](https://github.com/NVIDIA-NeMo/Guardrails) — 5.8k stars
- [NVIDIA-NeMo/Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) — HF ↔ Megatron
- [NVIDIA-NeMo/Automodel](https://github.com/NVIDIA-NeMo/Automodel) — FSDP2-native training
- [MoE Parallel Folding Paper](https://arxiv.org/abs/2504.14960)
- [Amsel et al. — Polar Express](https://arxiv.org/abs/2505.16932) — Muon orthogonalization
- [NVIDIA NCCL EP](https://docs.nvidia.com/deeplearning/nccl/) — Expert parallelism communication primitives

---

*This document represents a strategic analysis based on the state of NVIDIA's open-source ecosystem as of March 2026. Hardware specs and software capabilities are subject to rapid change. Verify current state before executing.*
