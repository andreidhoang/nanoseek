# The #1 Project: Fused MoE Kernel Suite with Blockwise FP8 Scaling

## For AI Performance Engineers Targeting Top 1% Frontier Lab Roles ($800K-$1.5M+ TC)

**Author**: Senior AI Performance Engineering Research Plan
**Date**: 2026-03-28
**Codebase**: NanoSeek (DeepSeek V3.2 reimplementation) + Nanochat (GPT-2 speedrun harness)
**Target**: Staff/Principal Performance Engineer at OpenAI, Anthropic, NVIDIA, DeepMind, Meta, xAI

---

## Why THIS Project Is #1

### The Evidence-Based Reasoning

Every career-defining contribution in AI performance engineering follows the same pattern:

| Project | Who | What They Solved | Speedup | Outcome |
|---------|-----|-----------------|---------|---------|
| FlashAttention | Tri Dao | Attention memory O(N^2) → O(N) | 2-4x | Co-founded Together AI, 11K+ GitHub stars |
| PagedAttention | Woosuk Kwon | KV cache waste 60-80% → <4% | 2-4x concurrent | vLLM project lead, Berkeley PhD |
| DeepEP | DeepSeek team | MoE all-to-all communication | 30%+ | Open-sourced, adopted by LMSYS |
| Liger-Kernel | LinkedIn | Fused training ops | 20% throughput, 60% memory | 2K+ stars, production at LinkedIn |

**The pattern**: Solve a universal bottleneck with deep hardware understanding, achieve measurable large speedups (not 5% — 2x+), open-source with production quality, accompany with benchmarks/paper.

### Why MoE Kernels Are The #1 Opportunity Right Now

**Fact 1**: MoE is THE architecture of 2025-2026. DeepSeek V3 (671B), Mixtral 8x22B, Qwen MoE, Grok-1, DBRX — all MoE. Every frontier lab is either training or planning MoE models.

**Fact 2**: Expert dispatch is 45-55% of MoE training step time. This is the single largest bottleneck in the fastest-growing model architecture.

**Fact 3**: FP8 blockwise scaling (DeepSeek V3's approach) has no high-quality open-source Triton implementation. TransformerEngine does tensorwise. DeepSeek's code is custom CUTLASS. There's a gap.

**Fact 4**: The PyTorch team just published (2025) their grouped GEMM kernel for MoE — but it's a single kernel, not a complete fused pipeline. SGLang and vLLM have Triton MoE kernels but not with blockwise FP8.

**Fact 5**: NanoSeek already has the perfect testbed — 64 routed + 2 shared experts, top-8 routing, sigmoid gate, SwiGLU, sequential dispatch (the bottleneck to replace).

### The 80/20: What Matters Most

```
┌─────────────────────────────────────────────────────────────┐
│  MoE Training Step Breakdown (measured at frontier labs)     │
│                                                             │
│  Expert FFN dispatch + compute:  45-55%  ← YOUR TARGET     │
│  Attention (MLA/MHA):            15-25%  (FlashAttention)   │
│  All-to-all communication:       10-20%  (DeepEP)           │
│  Embedding + loss:                5-10%  (Liger-Kernel)     │
│  MTP heads:                       3-5%                      │
│  Other (norm, optim, etc):        5-10%                     │
└─────────────────────────────────────────────────────────────┘

The 80/20: Expert dispatch + FP8 matmul = 80% of the value
```

---

## Project Definition

### Title
**NanoFuse: Fused MoE Dispatch-Compute-Combine Kernels with Blockwise FP8 Scaling**

### One-Sentence Summary
Custom Triton kernels that fuse token routing, expert computation (SwiGLU FFN), and output combining into a single GPU launch with blockwise FP8 scaling — validated end-to-end in NanoSeek's 64-expert MoE training pipeline.

### Deliverables
1. **Fused MoE Triton kernel** (dispatch + SwiGLU FFN + combine in one launch)
2. **Blockwise FP8 scaling** (128x128 weight blocks, 1x128 activation vectors)
3. **Profiling-driven benchmarks** (roofline analysis, MFU measurement, A/B comparison)
4. **End-to-end validation** in NanoSeek training (loss curves match BF16 baseline)
5. **Technical blog post** with profiling evidence and architectural decisions
6. **Optional paper** (MLSys/ISCA workshop submission)

### Success Criteria
| Metric | Target | Measurement |
|--------|--------|-------------|
| Expert dispatch speedup vs sequential | ≥2x | Wall-clock time per training step |
| FP8 throughput vs BF16 fused | ≥1.3x | Tokens/second at batch saturation |
| Memory reduction | ≥40% peak activation memory | `torch.cuda.max_memory_allocated()` |
| Loss parity | <0.5% BPB deviation from BF16 | `ema_val/bpb` after 1000 steps |
| MFU on H100 | ≥55% for fused MoE layer | Profiler FLOPS / theoretical peak |

---

## Technical Architecture

### Current State (NanoSeek Baseline)

```python
# Current: Sequential expert dispatch in model.py MoEDispatch
# This is the bottleneck — 64 sequential expert forward passes

for expert_idx in range(self.num_experts):
    mask = (expert_assignments == expert_idx)
    if mask.any():
        expert_input = x[mask]
        expert_output = self.experts[expert_idx](expert_input)  # SwiGLU FFN
        output[mask] += expert_output * weights[mask]
```

**Why this is slow**: 64 sequential CUDA kernel launches. Each expert processes a small batch (tokens/64 on average). GPU utilization is terrible — small matmuls on a device designed for massive parallelism.

### Target Architecture: NanoFuse

```
┌──────────────────────────────────────────────────────────────┐
│                    NanoFuse Kernel Pipeline                    │
│                                                               │
│  Input: [B*T, D] tokens + [B*T, E] routing weights           │
│                                                               │
│  Stage 1: Token Dispatch (Align & Sort)                       │
│  ┌─────────────────────────────────────────┐                  │
│  │ Sort tokens by expert assignment         │                  │
│  │ Build expert_offsets[] and token_map[]    │                  │
│  │ Compute per-expert batch sizes            │                  │
│  └─────────────────────────────────────────┘                  │
│                    ↓                                          │
│  Stage 2: Grouped GEMM (Fused SwiGLU FFN + FP8)              │
│  ┌─────────────────────────────────────────┐                  │
│  │ For all experts simultaneously:          │                  │
│  │   gate = x @ W_gate_fp8  (blockwise)     │                  │
│  │   up   = x @ W_up_fp8   (blockwise)      │                  │
│  │   h    = SiLU(gate) * up  (fused)        │                  │
│  │   out  = h @ W_down_fp8  (blockwise)     │                  │
│  │                                           │                  │
│  │ Key: All 64 experts in ONE kernel launch  │                  │
│  │ Key: FP8 blockwise scaling per 128x128    │                  │
│  └─────────────────────────────────────────┘                  │
│                    ↓                                          │
│  Stage 3: Token Combine (Weighted Sum + Scatter)              │
│  ┌─────────────────────────────────────────┐                  │
│  │ Multiply by routing weights               │                  │
│  │ Scatter-add back to original positions     │                  │
│  │ Add shared expert output                   │                  │
│  └─────────────────────────────────────────┘                  │
│                                                               │
│  Output: [B*T, D] mixed expert outputs                        │
└──────────────────────────────────────────────────────────────┘
```

### Blockwise FP8 Scaling Detail

```
Standard Tensorwise (current NanoSeek fp8.py):
┌─────────────────────┐
│ One scale factor     │  ← Entire tensor shares one scale
│ for entire W matrix  │  ← Outliers dominate, precision loss
└─────────────────────┘

Blockwise (DeepSeek V3 approach, our target):
┌────┬────┬────┬────┐
│s_00│s_01│s_02│s_03│  ← Each 128×128 block has its own scale
├────┼────┼────┼────┤  ← Better dynamic range utilization
│s_10│s_11│s_12│s_13│  ← Outliers contained within blocks
├────┼────┼────┼────┤
│s_20│s_21│s_22│s_23│  ← Fused into GEMM mainloop
├────┼────┼────┼────┤     (not post-hoc rescaling)
│s_30│s_31│s_32│s_33│
└────┴────┴────┴────┘

Activation scaling:
[token_0: scale_0] [token_1: scale_1] ... ← 1×128 vectors
```

---

## How a Senior Engineer Actually Executes This

**Total timeline**: 4 weeks (not 8). Less planning, more iterating.
**Core loop**: Profile → Hypothesis → Kernel → Nsight → Fix → Verify → Next.
**Time split**: 20% quick Python timing, 50% Nsight Compute, 30% writing/testing kernels.

---

## Execution Plan: 4 Weeks

### Week 1: Profile → Grouped GEMM (Days 1-5)

**Day 1 morning**: Profile baseline. 30 minutes, not 3 days.

```bash
# Profile 50 steps, open trace in Perfetto, find the bottleneck
python -m nanoseek.scripts.pre_train \
    --run profile-baseline --scale ablation --seed 42 \
    --num-iterations 50 --eval-every 50 --save-every -1 --device-batch-size 2
```

**Day 1 afternoon**: If MoE > 30% of step → write grouped GEMM. If not → pivot.

**Days 2-3**: Implement align & sort dispatch + `torch._grouped_mm` wrapper.
- Start with Option A (`torch._grouped_mm`) — zero dependencies, 30 lines
- Unit test: fused output must match sequential within BF16 tolerance
- `triton.testing.do_bench` comparison: grouped vs batched BMM

**Days 4-5**: Nsight Compute on the grouped GEMM kernel.
- `ncu --set full -o grouped_gemm python benchmark_kernel.py`
- Check: compute throughput %, memory throughput %, warp stalls
- Sweep tile sizes: BLOCK_M=[64,128,256], BLOCK_K=[32,64,128]
- Fix bottleneck (bank conflicts? register spill? wrong tile size?)
- Re-profile, iterate until ≥1.5x speedup over batched BMM

**Decision Gate**: ≥1.5x grouped over batched. If not, investigate memory-bandwidth bottleneck.

---

### Week 2: Fused SwiGLU + Blockwise FP8 (Days 6-10)

**Days 6-7**: Fuse SwiGLU into pipeline.
- Write Triton fused_swiglu kernel (forward: 20 lines, backward with recomputation: 30 lines)
- Integrate into grouped GEMM pipeline: gate GEMM → fused SwiGLU → down GEMM
- Memory benchmark: expect 60-75% activation memory reduction
- Nsight Compute: verify SwiGLU kernel is bandwidth-bound (should be ~800+ GB/s on 4090)

**Days 8-10**: Blockwise FP8 quantization — the crown jewel.
- Implement blockwise quantize kernel (128×128 blocks, ~40 lines Triton)
- Implement blockwise GEMM with fused in-loop rescaling (~80 lines Triton)
- Key: rescale INSIDE the K-loop accumulator, not post-hoc
- Precision test: blockwise must have ≥2x lower relative error than tensorwise
- Note: actual FP8 GEMM benchmarks need H100 (Week 3). On RTX 4090, validate correctness only.

**Decision Gate**: Blockwise quant error < tensorwise error. Fused SwiGLU saves ≥40% memory.

---

### Week 3: Training Integration + Validation (Days 11-15)

**Days 11-12**: Wire everything into NanoSeek.
- Custom `autograd.Function` for backward pass (E4M3 forward, E5M2 backward, FP32 accum)
- Add `--fused-moe` flag to pre_train.py
- `torch.autograd.gradcheck` for gradient correctness
- Fused combine kernel (scatter-add + shared expert in one launch)

**Days 13-14**: Training validation — the test that matters.
```bash
# Baseline: 500 steps, seed 42
python -m nanoseek.scripts.pre_train --run baseline --scale ablation --seed 42 \
    --num-iterations 500 --eval-every 100 --save-every -1 --device-batch-size 2

# NanoFuse: same everything except --fused-moe
python -m nanoseek.scripts.pre_train --run nanofuse --scale ablation --seed 42 \
    --num-iterations 500 --eval-every 100 --save-every -1 --device-batch-size 2 --fused-moe
```
- Compare: `ema_val/bpb` within 0.5%, `train/h_load` identical, `train/grad_norm` matches
- If any metric diverges > 1%: kernel has a bug. Fix before proceeding.

**Day 15**: Before/after profiling.
- Re-run torch.profiler with NanoFuse enabled
- Generate comparison table: step_time, MFU, memory, tokens/sec
- Nsight Compute on final kernel versions
- Quick NanoChat comparison: 500 steps at matched active params, compare BPB vs wall-clock

**Decision Gate**: Loss curves match. Step speedup ≥1.3x. Memory reduction ≥30%.

---

### Week 4: Polish + Ship (Days 16-20)

**Days 16-17**: Multi-GPU smoke test + edge cases.
- 2-GPU DDP test (if available): `torchrun --nproc_per_node=2`
- Edge cases: empty experts (0 tokens), extreme imbalance, batch_size=1
- Performance regression test script (runs automatically, fails if ≥5% slower)

**Days 18-19**: H100 validation (rent 1x H100, ~$30).
- Validate FP8 tensor core path actually works
- Benchmark: blockwise FP8 grouped GEMM vs BF16 grouped GEMM
- Expected: 1.3-1.5x additional speedup from FP8 on H100

**Day 20**: Ship.
- Blog post with before/after numbers (not theory — measured)
- Clean test suite (8 test files, all passing)
- PR with benchmark results in the commit message

```
Total: 20 working days. Not 45.
The difference: less planning, more Nsight Compute, faster iteration.
```

---

## File Structure

```
nanoseek/nanoseek/kernels/
├── __init__.py                    # Public API
├── nanofuse.py                    # NanoFuseMoE layer (drop-in replacement)
├── moe_dispatch.py                # Align & Sort dispatch kernel
├── grouped_gemm.py                # Grouped GEMM wrapper (torch._grouped_mm + Triton)
├── blockwise_fp8.py               # Blockwise FP8 quantization kernels
├── fp8_grouped_gemm.py            # FP8 blockwise GEMM kernel
├── fused_swiglu.py                # Fused SwiGLU forward + backward
├── combine.py                     # Fused scatter-add combine kernel
└── benchmarks/
    ├── profile_baseline.py        # Week 0 profiling script
    ├── benchmark_grouped.py       # Week 1 grouped vs sequential
    ├── benchmark_fp8.py           # Week 3 tensorwise vs blockwise
    ├── benchmark_e2e.py           # Week 7 comprehensive benchmarks
    └── roofline.py                # Roofline analysis tools
```

---

## Risk Analysis & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Blockwise FP8 GEMM has numerical issues | Medium | High | Extensive precision tests at every step. Fallback to tensorwise. |
| torch._grouped_mm not available in PyTorch version | Low | Medium | Implement pure Triton fallback. |
| Triton compiler bug on Hopper FP8 | Medium | High | Pin Triton version. Test on A100 (BF16 path) first. |
| MoE dispatch is NOT the bottleneck | Low | Critical | Week 0 profiling gates the entire project. Decision gate: abort if <30%. |
| FSDP2 incompatibility with custom autograd | Low | Medium | Test DDP first, then FSDP2. Keep sequential fallback. |
| Training divergence with blockwise FP8 | Medium | High | Compare loss curves every 100 steps. Auto-fallback to BF16 on NaN. |

---

## Why This Gets You Hired

### Direct Skill Alignment with Top 1% Roles

| Skill Demonstrated | OpenAI Posting | Anthropic Posting | NVIDIA Posting |
|-------------------|----------------|-------------------|----------------|
| Custom Triton/CUDA kernels | "Write high-performance CUDA/Triton kernels" | "Deep systems engineering" | "CUDA engineer" |
| FP8 training at scale | "Low-precision formats (FP8, FP4)" | "Inference optimization" | "TensorRT mixed precision" |
| Profiling-driven optimization | "Profiling end-to-end training runs" | "Maximize compute efficiency" | "Performance analysis" |
| MoE architecture understanding | "New model architectures scale efficiently" | "LLM serving optimization" | "MoE inference" |
| Distributed training | "Sharding models across thousands of GPUs" | "Fleet-wide orchestration" | "Multi-GPU scaling" |
| Open-source contribution | Community evidence of impact | Same | Same |

### Portfolio Impact Score

```
This project demonstrates:
✅ Custom GPU kernel development (Triton, not just PyTorch)
✅ Low-level performance optimization (roofline, MFU, memory)
✅ FP8 numerics understanding (blockwise vs tensorwise, E4M3/E5M2)
✅ MoE architecture expertise (routing, dispatch, expert parallelism)
✅ End-to-end training validation (not just kernel benchmarks)
✅ Distributed training compatibility (FSDP2, DDP)
✅ Production-quality code (tests, docs, clean API)
✅ Communication ability (blog post, benchmarks, optional paper)

Combined: This is exactly the portfolio that FlashAttention, PagedAttention,
and DeepEP built for their creators. Solve a universal bottleneck,
demonstrate deep hardware understanding, validate end-to-end.
```

---

## Compensation Context

For context on what completing this project positions you for:

| Level | Company | Base | Total Comp | Notes |
|-------|---------|------|-----------|-------|
| L5 (Staff) | OpenAI | $340K | $1,060K | Equity-dominated ($723K stock) |
| Lead SWE | Anthropic | $350-485K | $570-900K+ | Inference team |
| L7 (Principal) | Google DeepMind | $350K+ | $996K | RSU-dominated |
| E6 (Staff) | Meta | $280-350K | $790K | PyTorch core team |
| IC7 (Principal) | NVIDIA | $300K+ | $1,040K | TensorRT/CUDA team |
| Staff | xAI | $250K+ | $600-970K | Aggressive equity |

**The supply-demand imbalance**: Perhaps a few hundred people worldwide can write custom CUDA kernels, debug NCCL collectives across 10K GPUs, and have intuition for how architectural choices affect hardware utilization. This project puts you in that pool.

---

## Quick Start

```bash
# Prerequisites: NVIDIA GPU (A6000+ for dev, H100 for FP8 validation)
# PyTorch 2.4+ with Triton

cd /workspace/nanoseek

# Week 0: Profile baseline
python -m nanoseek.scripts.pre_train \
    --run profile-baseline --scale ablation --seed 42 \
    --num-iterations 50 --eval-every 50 --save-every -1

# After implementing kernels (Week 4+):
python -m nanoseek.scripts.pre_train \
    --run validation-nanofuse --scale ablation --seed 42 \
    --num-iterations 500 --eval-every 100 --save-every -1 \
    --fused-moe --fp8

# Compare loss curves
python -m nanoseek.scripts.analyze_ablations \
    --runs validation-baseline validation-nanofuse \
    --metrics ema_val/bpb train/h_load eval/i_spec_mean
```

---

## References

### Papers
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) (Dao et al., NeurIPS 2024)
- [FlashInfer: Efficient and Customizable Attention Engine](https://arxiv.org/pdf/2501.01005) (MLSys 2025 Best Paper)
- [DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437v1) — FP8 blockwise scaling at 2048 H800 scale
- [MegaBlocks: Efficient Sparse Training with MoE](https://arxiv.org/abs/2211.15841) — Dropless MoE via block-sparse GEMM
- [Liger-Kernel: Efficient Triton Kernels for LLM Training](https://arxiv.org/pdf/2410.10989) — Fused RMSNorm, SwiGLU, CrossEntropy
- [Scaling Llama 3 Training with Efficient Parallelism](https://dl.acm.org/doi/10.1145/3695053.3731410) (ISCA 2025)
- [FP4 All the Way: Fully Quantized Training of LLMs](https://arxiv.org/pdf/2505.19115) (2025)
- [NVIDIA FP8 Scaling Strategies](https://developer.nvidia.com/blog/per-tensor-and-per-block-scaling-strategies-for-effective-fp8-training/)

### Open-Source
- [PyTorch Grouped GEMM for MoE](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/)
- [DeepEP: Expert Parallel Communication](https://github.com/deepseek-ai/DeepEP)
- [SGLang Fused MoE Design](https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang)
- [vLLM Fused MoE Kernel](https://docs.vllm.ai/en/latest/design/moe_kernel_features/)

### Career Research
- [OpenAI Training Performance Engineer](https://openai.com/careers/training-performance-engineer-san-francisco/)
- [Anthropic Staff SWE Inference](https://job-boards.greenhouse.io/anthropic/jobs/4951696008)
- [How to Get Hired at Frontier Labs 2026](https://www.sundeepteki.org/advice/how-to-get-hired-at-openai-anthropic-and-google-deepmind-in-2026)
- [AI Engineer Compensation 2026](https://www.axiomrecruit.com/resources/industry-insights/ai-engineer-compensation-2026--what-the-world-is-paying/)
