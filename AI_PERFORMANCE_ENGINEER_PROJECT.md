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

## Execution Plan: 8 Weeks

### Week 0: Profiling & Measurement (Days 1-3)

**Objective**: Establish ground truth baseline. Measure, don't guess.

**Tasks**:

1. **Profile NanoSeek ablation training** (existing sequential MoE dispatch)
   ```bash
   # Run 50 training steps with PyTorch profiler
   python -m nanoseek.scripts.pre_train \
       --run profile-baseline --scale ablation --seed 42 \
       --num-iterations 50 --eval-every 50 --save-every -1 \
       --device-batch-size 4
   ```

2. **Generate roofline analysis** for each component:
   ```python
   # Key measurements to capture:
   # 1. Per-expert GEMM time (gate_proj, up_proj, down_proj)
   # 2. Token dispatch overhead (scatter/gather)
   # 3. Routing computation time
   # 4. Memory bandwidth utilization per operation
   # 5. SM occupancy during expert computation

   # Tool: torch.profiler with trace export
   with torch.profiler.profile(
       activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
       schedule=torch.profiler.schedule(wait=5, warmup=5, active=10),
       on_trace_ready=torch.profiler.tensorboard_trace_handler('./profile_baseline'),
       record_shapes=True,
       profile_memory=True,
       with_stack=True,
       with_flops=True,
   ) as prof:
       for step in range(20):
           train_step()
           prof.step()
   ```

3. **Compute theoretical peak** and current MFU:
   ```
   H100 SXM theoretical: 989 TFLOPS (BF16), 1,979 TFLOPS (FP8)
   A6000 theoretical: 155 TFLOPS (BF16)

   MFU = measured_TFLOPS / theoretical_TFLOPS
   Target: identify how far below theoretical we are
   ```

4. **Document baseline** in structured format:
   ```
   Baseline Report:
   - Step time: ___ms
   - Expert dispatch time: ___ms (___% of step)
   - Per-expert GEMM time: ___ms
   - Token routing time: ___ms
   - Memory peak: ___GB
   - MFU: ___%
   - Bottleneck classification: compute-bound | memory-bound | launch-bound
   ```

**Deliverable**: `nanoseek/benchmarks/baseline_profile.md` with Chrome trace, roofline chart, and bottleneck classification.

**Decision Gate**: If expert dispatch is <30% of step time, pivot to attention optimization instead. Proceed only if dispatch is the dominant bottleneck.

---

### Week 1: Grouped GEMM Foundation (Days 4-7)

**Objective**: Replace sequential expert loop with a single grouped GEMM call.

**Tasks**:

1. **Implement Align & Sort dispatch** (Triton kernel):
   ```python
   # nanoseek/nanoseek/kernels/moe_dispatch.py

   @triton.jit
   def align_and_sort_kernel(
       # Input: expert_assignments [B*T], routing_weights [B*T, top_k]
       # Output: sorted_token_ids, expert_offsets, expert_sizes
       expert_assignments_ptr,
       sorted_ids_ptr,
       expert_offsets_ptr,
       num_tokens: tl.constexpr,
       num_experts: tl.constexpr,
   ):
       """
       Sort tokens by expert assignment for grouped GEMM.

       Algorithm:
       1. Histogram: count tokens per expert (atomic add)
       2. Prefix sum: compute expert_offsets = cumsum(expert_counts)
       3. Scatter: place each token_id at its expert's offset position

       This converts irregular routing into contiguous expert batches,
       enabling grouped GEMM instead of 64 sequential launches.
       """
       # Implementation follows SGLang's Align & Sort pattern
       pass
   ```

2. **Implement grouped GEMM wrapper**:
   ```python
   # Option A: torch._grouped_mm (PyTorch native, simplest)
   # Option B: Triton grouped GEMM (more control, portable)
   # Option C: CUTLASS grouped GEMM (maximum performance)

   # Start with Option A, profile, upgrade if needed

   def fused_moe_forward(
       hidden_states: torch.Tensor,      # [num_tokens, d_model]
       expert_assignments: torch.Tensor,  # [num_tokens, top_k]
       routing_weights: torch.Tensor,     # [num_tokens, top_k]
       gate_weights: list[torch.Tensor],  # [num_experts] x [inter_dim, d_model]
       up_weights: list[torch.Tensor],    # [num_experts] x [inter_dim, d_model]
       down_weights: list[torch.Tensor],  # [num_experts] x [d_model, inter_dim]
   ) -> torch.Tensor:
       """
       Fused MoE: dispatch + grouped SwiGLU FFN + combine.

       Instead of 64 sequential expert calls:
       1. Sort tokens by expert (align_and_sort)
       2. Stack all expert weights → single grouped GEMM
       3. Apply SwiGLU activation (fused)
       4. Second grouped GEMM (down projection)
       5. Scatter-add with routing weights
       """
       # Sort tokens
       sorted_ids, expert_offsets = align_and_sort(expert_assignments)
       sorted_x = hidden_states[sorted_ids]

       # Stack weights for grouped GEMM
       # Gate: [num_experts, inter_dim, d_model]
       # Up:   [num_experts, inter_dim, d_model]
       # Down: [num_experts, d_model, inter_dim]
       W_gate = torch.stack(gate_weights)
       W_up = torch.stack(up_weights)
       W_down = torch.stack(down_weights)

       # Expert sizes for grouped GEMM
       expert_sizes = expert_offsets[1:] - expert_offsets[:-1]

       # Grouped GEMM: gate projection
       gate_out = torch._grouped_mm(sorted_x, W_gate.transpose(-1,-2),
                                      offs=expert_offsets)

       # Grouped GEMM: up projection
       up_out = torch._grouped_mm(sorted_x, W_up.transpose(-1,-2),
                                    offs=expert_offsets)

       # Fused SwiGLU
       h = F.silu(gate_out) * up_out

       # Grouped GEMM: down projection
       expert_out = torch._grouped_mm(h, W_down.transpose(-1,-2),
                                        offs=expert_offsets)

       # Combine: weighted scatter-add
       output = torch.zeros_like(hidden_states)
       for k in range(top_k):
           token_ids = expert_assignments[:, k]
           weights_k = routing_weights[:, k].unsqueeze(-1)
           output.scatter_add_(0, sorted_ids.unsqueeze(-1).expand_as(expert_out),
                               expert_out * weights_k)

       return output
   ```

3. **Unit test**: Verify fused output matches sequential output within BF16 tolerance.
   ```python
   def test_fused_matches_sequential():
       """Fused MoE must match sequential within 1e-5 relative error."""
       # Run same input through both paths
       seq_output = sequential_moe(x, assignments, weights, experts)
       fused_output = fused_moe_forward(x, assignments, weights, ...)
       torch.testing.assert_close(seq_output, fused_output, rtol=1e-5, atol=1e-5)
   ```

4. **Profile grouped vs sequential**:
   ```
   Expected results (based on literature):
   - Sequential 64 experts: ~45ms per MoE layer (A6000)
   - Grouped GEMM: ~15-25ms per MoE layer
   - Speedup: 1.8-3x on MoE dispatch alone
   ```

**Deliverable**: `nanoseek/nanoseek/kernels/moe_dispatch.py` + `nanoseek/nanoseek/kernels/grouped_gemm.py` with passing tests and benchmark comparison.

**Decision Gate**: Grouped GEMM must be ≥1.5x faster than sequential. If not, investigate whether the bottleneck is memory-bandwidth (need fusion) rather than launch overhead.

---

### Week 2: SwiGLU Fusion & Activation Optimization (Days 8-11)

**Objective**: Fuse SwiGLU activation into the GEMM pipeline to eliminate intermediate tensor materialization.

**Tasks**:

1. **Fused SwiGLU Triton kernel**:
   ```python
   @triton.jit
   def fused_swiglu_kernel(
       gate_ptr, up_ptr, output_ptr,
       n_elements: tl.constexpr,
       BLOCK_SIZE: tl.constexpr,
   ):
       """
       Fused SiLU(gate) * up — eliminates materializing gate and up separately.

       Memory savings: 2 * [tokens, inter_dim] intermediate tensors eliminated.
       For NanoSeek ablation (inter_dim=3440): saves ~26MB per MoE layer.
       For NanoSeek 1B (inter_dim=5504): saves ~42MB per MoE layer.

       This is exactly what Liger-Kernel does, but fused into our MoE pipeline.
       """
       pid = tl.program_id(0)
       offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
       mask = offsets < n_elements

       gate = tl.load(gate_ptr + offsets, mask=mask)
       up = tl.load(up_ptr + offsets, mask=mask)

       # SiLU(gate) * up — fused, no intermediate materialization
       silu_gate = gate * tl.sigmoid(gate)
       result = silu_gate * up

       tl.store(output_ptr + offsets, result, mask=mask)
   ```

2. **Backward pass with recomputation** (Liger-Kernel strategy):
   ```python
   @triton.jit
   def fused_swiglu_backward_kernel(
       grad_output_ptr, gate_ptr, up_ptr,
       grad_gate_ptr, grad_up_ptr,
       n_elements: tl.constexpr,
       BLOCK_SIZE: tl.constexpr,
   ):
       """
       Backward: recompute SiLU(gate) instead of saving it.

       Trade: ~0.5% extra compute for 1.6x memory reduction.
       This is the proven Liger-Kernel approach.

       grad_gate = grad_output * up * (sigmoid(gate) + gate * sigmoid(gate) * (1 - sigmoid(gate)))
       grad_up = grad_output * SiLU(gate)
       """
       pass
   ```

3. **Integration into grouped GEMM pipeline**:
   - Gate GEMM → fused SwiGLU → Down GEMM (3 stages, 2 GEMMs + 1 fused activation)
   - Eliminate intermediate `gate_out` and `up_out` tensor allocations

4. **Memory benchmark**:
   ```
   Before (sequential + materialized):
   - gate_out: [tokens_per_expert, inter_dim] x 64 experts
   - up_out:   [tokens_per_expert, inter_dim] x 64 experts
   - Total intermediate: ~1.6GB (ablation), ~4.2GB (1B)

   After (fused, recompute in backward):
   - Only output tensor materialized
   - Total intermediate: ~0.4GB (ablation), ~1.0GB (1B)
   - Savings: 60-75%
   ```

**Deliverable**: `nanoseek/nanoseek/kernels/fused_swiglu.py` with forward + backward kernels, integrated into MoE pipeline.

---

### Week 3: Blockwise FP8 Scaling (Days 12-18)

**Objective**: Implement DeepSeek V3's blockwise FP8 scaling strategy in Triton — the frontier approach that has no high-quality open-source implementation.

**This is the crown jewel of the project.**

**Tasks**:

1. **Understand the math** (first principles):
   ```
   Tensorwise scaling (current fp8.py):
     scale = max(|W|) / E4M3_MAX
     W_fp8 = quantize(W / scale)
     Output = (A_fp8 @ W_fp8) * (scale_A * scale_W)

   Problem: Single scale for entire tensor.
   If max element is 1000 but 99% of elements are <1,
   those small elements lose ALL precision in FP8.

   Blockwise scaling (DeepSeek V3):
     For each 128×128 block of W:
       scale_ij = max(|W[i:i+128, j:j+128]|) / E4M3_MAX
       W_fp8[i:i+128, j:j+128] = quantize(W[i:i+128, j:j+128] / scale_ij)

     For each 1×128 vector of activations A:
       scale_k = max(|A[k, 0:128]|) / E4M3_MAX
       A_fp8[k, 0:128] = quantize(A[k, 0:128] / scale_k)

     Output[i,j] = sum over blocks(A_fp8_block @ W_fp8_block * scale_A_k * scale_W_ij)

   Key: Rescaling MUST happen inside the GEMM mainloop,
   not as a post-hoc multiply. This is what makes it hard.
   ```

2. **Implement blockwise quantization** (Triton):
   ```python
   @triton.jit
   def blockwise_fp8_quantize_kernel(
       input_ptr, output_ptr, scales_ptr,
       M, N,
       BLOCK_M: tl.constexpr = 128,
       BLOCK_N: tl.constexpr = 128,
   ):
       """
       Quantize a matrix to FP8 E4M3 with per-block scaling.

       Each 128×128 block gets its own scale factor.
       Scale = max(|block|) / 448.0 (E4M3 max representable value)

       The scale grid has shape [M//128, N//128].
       """
       block_m = tl.program_id(0)
       block_n = tl.program_id(1)

       # Load 128×128 block
       offsets_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
       offsets_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
       mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)

       block = tl.load(input_ptr + offsets_m[:, None] * N + offsets_n[None, :], mask=mask)

       # Compute block-level scale
       abs_max = tl.max(tl.abs(block))
       scale = abs_max / 448.0  # E4M3 max
       scale = tl.where(scale > 0, scale, 1.0)  # Avoid division by zero

       # Quantize
       quantized = (block / scale).to(tl.float8e4m3fn)

       # Store quantized values and scale
       tl.store(output_ptr + offsets_m[:, None] * N + offsets_n[None, :],
                quantized, mask=mask)
       tl.store(scales_ptr + block_m * (N // BLOCK_N) + block_n, scale)
   ```

3. **Implement blockwise GEMM with fused rescaling** (the hard part):
   ```python
   @triton.jit
   def blockwise_fp8_grouped_gemm_kernel(
       # A: [M, K] in FP8 with row scales [M, K//128]
       # B: [K, N] in FP8 with block scales [K//128, N//128]
       # C: [M, N] output in BF16
       A_ptr, B_ptr, C_ptr,
       A_scales_ptr, B_scales_ptr,
       M, N, K,
       BLOCK_M: tl.constexpr = 128,
       BLOCK_N: tl.constexpr = 128,
       BLOCK_K: tl.constexpr = 128,
   ):
       """
       Grouped GEMM with blockwise FP8 scaling.

       The critical insight from DeepSeek V3:
       - Accumulate in FP32 (not FP16) to prevent precision loss
       - Rescale per block INSIDE the K-loop, not after
       - Each (BLOCK_M, BLOCK_N) output tile accumulates
         contributions from K//BLOCK_K inner blocks,
         each with its own pair of scales

       C[m, n] = sum_k ( A_fp8[m, k] * scale_A[m, k//128]
                       * B_fp8[k, n] * scale_B[k//128, n//128] )
       """
       pid_m = tl.program_id(0)
       pid_n = tl.program_id(1)

       # Accumulator in FP32
       acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

       for k_block in range(0, K // BLOCK_K):
           # Load A block [BLOCK_M, BLOCK_K] in FP8
           a = tl.load(...)  # A[pid_m*BM : (pid_m+1)*BM, k_block*BK : (k_block+1)*BK]

           # Load B block [BLOCK_K, BLOCK_N] in FP8
           b = tl.load(...)  # B[k_block*BK : (k_block+1)*BK, pid_n*BN : (pid_n+1)*BN]

           # Load scales for this block pair
           scale_a = tl.load(A_scales_ptr + ...)  # [BLOCK_M] row scales
           scale_b = tl.load(B_scales_ptr + ...)  # Scalar block scale

           # FP8 matmul with fused rescaling
           # WGMMA on Hopper: native FP8 tensor core operation
           block_result = tl.dot(a, b)  # FP8 dot product → FP32 accumulator

           # Rescale in-loop (not post-hoc!)
           acc += block_result * (scale_a[:, None] * scale_b)

       # Store final result in BF16
       c = acc.to(tl.bfloat16)
       tl.store(C_ptr + ..., c)
   ```

4. **Integrate with grouped MoE**:
   - All 64 expert weight matrices pre-quantized to FP8 with block scales
   - Activations quantized on-the-fly with row scales (1x128)
   - Gate router remains BF16 (routing precision is sacred — NanoSeek rule)
   - MLA projections remain BF16 (lora ranks 275/90/440/143 not 128-aligned)
   - Shared experts remain BF16 (only 2, overhead negligible)

5. **Precision validation**:
   ```python
   def test_blockwise_fp8_precision():
       """Blockwise FP8 must have lower quantization error than tensorwise."""
       W = torch.randn(2048, 5504, dtype=torch.bfloat16).cuda()

       # Tensorwise (current)
       tw_scale = W.abs().max() / 448.0
       tw_quantized = (W / tw_scale).to(torch.float8_e4m3fn)
       tw_error = (tw_quantized.float() * tw_scale - W.float()).norm() / W.float().norm()

       # Blockwise (new)
       bw_quantized, bw_scales = blockwise_quantize(W, block_size=128)
       bw_reconstructed = blockwise_dequantize(bw_quantized, bw_scales)
       bw_error = (bw_reconstructed - W.float()).norm() / W.float().norm()

       # Blockwise MUST have lower error
       assert bw_error < tw_error, f"Blockwise {bw_error:.6f} >= Tensorwise {tw_error:.6f}"
       # Expected: blockwise ~2-5x lower relative error
   ```

**Deliverable**: `nanoseek/nanoseek/kernels/blockwise_fp8.py` + `nanoseek/nanoseek/kernels/fp8_grouped_gemm.py` with complete forward pass, precision tests, and benchmark comparison against tensorwise.

---

### Week 4: Backward Pass & Training Integration (Days 19-25)

**Objective**: Complete the backward pass for blockwise FP8 grouped GEMM and integrate into NanoSeek's training loop.

**Tasks**:

1. **Backward pass implementation**:
   ```
   Forward:  Y = X_fp8 @ W_fp8  (with blockwise scales)

   Backward (3 GEMMs):
   dX = dY @ W^T           ← Uses E5M2 for dY (wider range for gradients)
   dW = X^T @ dY            ← Gradient accumulation in FP32
   (no activation grad for W since W is a parameter)

   Key decisions:
   - Forward: E4M3 (higher precision, 4 exponent bits, 3 mantissa)
   - Backward: E5M2 (wider range, 5 exponent bits, 2 mantissa)
   - Accumulation: Always FP32 (never FP16, this kills training)
   - use_fast_accum: True for forward, False for backward
     (matches NanoSeek's existing fp8.py convention)
   ```

2. **Custom autograd.Function**:
   ```python
   class BlockwiseFP8GroupedMoE(torch.autograd.Function):
       @staticmethod
       def forward(ctx, hidden_states, expert_weights, routing_info):
           # Quantize activations to FP8 E4M3 with row scales
           x_fp8, x_scales = blockwise_quantize_activations(hidden_states)

           # Expert weights already quantized (done once at init/periodically)
           # Run fused dispatch + grouped GEMM + combine
           output = fused_moe_fp8_forward(x_fp8, x_scales,
                                           expert_weights, routing_info)

           ctx.save_for_backward(hidden_states, expert_weights, routing_info)
           return output

       @staticmethod
       def backward(ctx, grad_output):
           hidden_states, expert_weights, routing_info = ctx.saved_tensors

           # Quantize grad_output to FP8 E5M2 (wider range for gradients)
           grad_fp8, grad_scales = blockwise_quantize_e5m2(grad_output)

           # dX = grad @ W^T (grouped GEMM, FP8)
           grad_input = fused_moe_fp8_backward_input(grad_fp8, grad_scales,
                                                      expert_weights, routing_info)

           # dW = X^T @ grad (grouped GEMM, FP8, accumulate in FP32)
           grad_weights = fused_moe_fp8_backward_weight(hidden_states,
                                                         grad_fp8, grad_scales,
                                                         routing_info)

           return grad_input, grad_weights, None
   ```

3. **Integration into NanoSeek's pre_train.py**:
   ```python
   # In pre_train.py, add --fused-moe flag
   parser.add_argument('--fused-moe', action='store_true',
                       help='Use fused MoE kernels with blockwise FP8')

   # In model initialization
   if args.fused_moe:
       from nanoseek.kernels import NanoFuseMoE
       # Replace sequential MoEDispatch with NanoFuseMoE
       for layer in model.layers:
           layer.moe = NanoFuseMoE.from_sequential(layer.moe)
   ```

4. **Training validation** (critical):
   ```bash
   # Run 500 steps with both paths, compare loss curves
   # BF16 sequential baseline
   python -m nanoseek.scripts.pre_train \
       --run validation-baseline --scale ablation --seed 42 \
       --num-iterations 500 --eval-every 100 --save-every -1

   # Fused MoE with blockwise FP8
   python -m nanoseek.scripts.pre_train \
       --run validation-fused-fp8 --scale ablation --seed 42 \
       --num-iterations 500 --eval-every 100 --save-every -1 \
       --fused-moe --fp8

   # Compare: ema_val/bpb must be within 0.5% at step 500
   ```

5. **Gradient correctness test**:
   ```python
   def test_gradient_correctness():
       """torch.autograd.gradcheck on fused MoE layer."""
       # Use float64 for numerical gradient checking
       model_fp64 = make_small_moe(dtype=torch.float64)
       x = torch.randn(32, 256, dtype=torch.float64, requires_grad=True)
       torch.autograd.gradcheck(model_fp64, x, eps=1e-6, atol=1e-4, rtol=1e-3)
   ```

**Deliverable**: Complete forward + backward pass integrated into NanoSeek training, with loss curve comparison showing <0.5% BPB deviation.

---

### Week 5: Combine Kernel & Shared Expert Fusion (Days 26-30)

**Objective**: Optimize the output combine stage and integrate shared expert path.

**Tasks**:

1. **Fused scatter-add combine kernel**:
   ```python
   @triton.jit
   def fused_combine_kernel(
       expert_output_ptr,    # [total_selected_tokens, d_model]
       routing_weights_ptr,  # [num_tokens, top_k]
       token_map_ptr,        # Maps sorted position → original position
       output_ptr,           # [num_tokens, d_model]
       shared_expert_ptr,    # [num_tokens, d_model] (shared expert output)
       num_tokens, d_model, top_k,
       BLOCK_D: tl.constexpr = 128,
   ):
       """
       Fused: routing_weight * expert_output → scatter_add + shared_expert_add

       Eliminates 3 separate kernel launches:
       1. Multiply by routing weights
       2. Scatter-add to original positions
       3. Add shared expert output

       All in one kernel = one global memory read/write cycle.
       """
       pass
   ```

2. **Shared expert optimization**:
   ```
   NanoSeek has 2 shared experts (always active for all tokens).
   These are dense FFNs, not routed.

   Optimization: overlap shared expert compute with routed expert dispatch.
   - Launch shared expert on a separate CUDA stream
   - While grouped GEMM runs 64 routed experts on stream 0,
     2 shared experts run on stream 1
   - Combine outputs after both complete

   Expected gain: ~5-8% (shared experts are small relative to routed)
   ```

3. **End-to-end fusion benchmark**:
   ```
   Before (sequential):
     Route → 64x Expert → Combine → Shared → Add
     Total: ~45ms per MoE layer (A6000)

   After (NanoFuse):
     Route → Align&Sort → GroupedGEMM_FP8 → FusedCombine+Shared
     Total: ~12-18ms per MoE layer (expected)
     Speedup: 2.5-3.8x on MoE layer
     Overall step speedup: 1.3-1.8x (MoE is 45-55% of step)
   ```

**Deliverable**: Complete NanoFuse pipeline with all stages fused, benchmark report.

---

### Week 6: Multi-GPU & FSDP2 Integration (Days 31-35)

**Objective**: Ensure fused kernels work correctly with distributed training.

**Tasks**:

1. **FSDP2 compatibility testing**:
   ```python
   # NanoSeek targets 8xH100 for 1B graduation run
   # FSDP2 shards parameters across GPUs
   # Our fused MoE must handle:
   # - Sharded expert weights (W_gate, W_up, W_down per expert)
   # - All-gather before grouped GEMM
   # - Reduce-scatter after combine

   # Test: 2-GPU DDP smoke test
   torchrun --nproc_per_node=2 -m nanoseek.scripts.pre_train \
       --run fsdp-test --scale ablation --seed 42 \
       --num-iterations 20 --fused-moe --fp8
   ```

2. **Communication overlap**:
   ```
   For 8xH100 (NanoSeek 1B):
   - FSDP2 all-gather for next layer overlaps with current layer compute
   - Expert weights are the largest parameters (64 experts × 3 projections)
   - Pre-quantize weights to FP8 before all-gather → 2x communication reduction

   Implementation:
   - Register forward hooks on MoE layers
   - In pre-forward hook: initiate all-gather for next layer's expert weights
   - Quantize to FP8 on the source rank before sending (save bandwidth)
   ```

3. **Gradient synchronization**:
   ```python
   # Expert gradients after backward:
   # - Each rank has full gradients for its shard
   # - Reduce-scatter distributes gradient shards
   # - FP8 gradient compression for reduce-scatter (E5M2)
   #   → 2x reduction in backward communication

   # This is exactly what ZeRO++ qgZ does
   ```

**Deliverable**: Verified 2-GPU and 8-GPU training with fused MoE kernels, communication overlap benchmarks.

---

### Week 7: Comprehensive Benchmarking & Profiling (Days 36-40)

**Objective**: Generate publication-quality benchmarks and profiling evidence.

**Tasks**:

1. **Benchmark matrix**:

   | Configuration | Hardware | Scale | Steps | Metrics |
   |---------------|----------|-------|-------|---------|
   | Sequential BF16 | A6000 | ablation | 500 | step_time, MFU, memory, bpb |
   | Grouped BF16 | A6000 | ablation | 500 | step_time, MFU, memory, bpb |
   | Grouped FP8 tensorwise | A6000* | ablation | 500 | step_time, MFU, memory, bpb |
   | NanoFuse FP8 blockwise | H100 | ablation | 500 | step_time, MFU, memory, bpb |
   | NanoFuse FP8 blockwise | H100 | 1B | 200 | step_time, MFU, memory, bpb |
   | Sequential BF16 | H100 | 1B | 200 | step_time, MFU, memory, bpb |

   *Note: FP8 on A6000 is simulated (no FP8 tensor cores), measures only compute

2. **Roofline analysis** for each kernel:
   ```
   For each kernel (align_sort, grouped_gemm_fp8, fused_swiglu, combine):
   - Arithmetic intensity (FLOPS / bytes transferred)
   - Achieved bandwidth (bytes/s vs HBM peak)
   - Achieved compute (TFLOPS vs SM peak)
   - Classification: compute-bound, memory-bound, or latency-bound
   ```

3. **Scaling analysis**:
   ```
   How does NanoFuse scale with:
   - Number of experts (8, 16, 32, 64, 128)
   - Expert batch size (tokens_per_expert: 16, 64, 256, 1024)
   - Model dimension (1280, 2048, 4096)
   - Top-k (2, 4, 8, 16)

   This answers: "When should you use NanoFuse vs sequential?"
   Answer: NanoFuse wins when experts > 16 AND tokens_per_expert > 32
   ```

4. **Loss curve comparison** (final validation):
   ```
   Plot side-by-side:
   - ema_val/bpb over 1000 steps (ablation scale)
   - train/h_load (expert balance)
   - eval/i_spec_mean (expert specialization)
   - train/grad_norm

   All must match within noise floor.
   ```

**Deliverable**: `nanoseek/benchmarks/nanofuse_report.md` with tables, charts, roofline plots, and scaling analysis.

---

### Week 8: Documentation, Blog Post & Release (Days 41-45)

**Objective**: Package for open-source release and career impact.

**Tasks**:

1. **Technical blog post** (2000-3000 words):
   ```markdown
   # NanoFuse: Fused MoE Kernels with Blockwise FP8 Scaling

   ## The Problem
   - MoE dispatch is 45-55% of training step time
   - Sequential expert loops waste GPU parallelism
   - Tensorwise FP8 loses precision on outlier-containing weights

   ## The Solution
   - Align & Sort token dispatch (one kernel launch for all 64 experts)
   - Grouped GEMM with fused SwiGLU (3 stages → 1 fused pipeline)
   - Blockwise FP8 scaling (128×128 blocks, DeepSeek V3's approach)

   ## Results
   - 2.5-3.8x speedup on MoE dispatch
   - 1.3-1.8x overall training step speedup
   - 40%+ memory reduction
   - <0.5% BPB deviation from BF16 baseline
   - Validated end-to-end in NanoSeek (64 experts, top-8, 4.75B params)

   ## How It Works
   [Detailed technical explanation with diagrams]

   ## Benchmarks
   [Tables and charts from Week 7]

   ## Code
   [Link to open-source repository]
   ```

2. **Clean API documentation**:
   ```python
   # nanoseek/nanoseek/kernels/__init__.py

   from .nanofuse import NanoFuseMoE
   from .blockwise_fp8 import blockwise_quantize, blockwise_dequantize
   from .fused_swiglu import FusedSwiGLU

   __all__ = ['NanoFuseMoE', 'blockwise_quantize', 'FusedSwiGLU']

   # Usage:
   # from nanoseek.kernels import NanoFuseMoE
   # moe_layer = NanoFuseMoE(num_experts=64, d_model=2048, inter_dim=5504, top_k=8)
   # output = moe_layer(hidden_states, routing_weights, expert_assignments)
   ```

3. **Test suite** (comprehensive):
   ```
   tests/test_kernels/
   ├── test_align_and_sort.py       # Dispatch correctness
   ├── test_grouped_gemm.py         # GEMM correctness vs torch.mm
   ├── test_blockwise_fp8.py        # Quantization precision
   ├── test_fused_swiglu.py         # Activation correctness
   ├── test_nanofuse_e2e.py         # End-to-end output match
   ├── test_nanofuse_backward.py    # Gradient correctness
   ├── test_nanofuse_fsdp.py        # Distributed compatibility
   └── benchmark_nanofuse.py        # Performance regression tests
   ```

4. **Optional: Workshop paper draft** (MLSys/ISCA/EuroSys):
   ```
   Title: "NanoFuse: Fused MoE Dispatch with Blockwise FP8 Scaling
           for Efficient Mixture-of-Experts Training"

   Key contributions:
   1. First open-source Triton implementation of blockwise FP8 GEMM
   2. Fused dispatch-compute-combine pipeline for MoE
   3. End-to-end validation at 4.75B parameter scale
   4. Comprehensive scaling analysis across expert count and batch size
   ```

**Deliverable**: Blog post, documentation, test suite, optional paper draft.

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
