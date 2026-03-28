# NanoSeek Ablation — Baseline Profile Report
## RTX 5090 (Blackwell, 32GB GDDR7)
### Date: 2026-03-28

---

## Hardware

```
GPU:            NVIDIA GeForce RTX 5090
VRAM:           32,607 MiB (~32 GB)
Peak compute:   209.5 TFLOPS (BF16 Tensor Core)
Peak bandwidth: 1,792 GB/s (GDDR7)
Ridge point:    116.9 FLOP/Byte
Driver:         575.64.03
CUDA:           12.9
```

## Model

```
Scale:          Ablation
Active params:  440,174,525  (~440M)
Total params:   1,988,462,525  (~1.95B)
Expansion:      4.5x
Hidden:         1280
Layers:         16 (2 dense + 14 MoE)
Experts:        64 routed (top-8) + 2 shared
MoE inter:      480 per expert
Shared inter:   960
```

## Test Setup

```
Batch size:     2
Seq length:     512
Tokens/step:    1,024
Autocast:       BF16
Grad ckpt:      OFF
Optimizer:      NONE (OOM — see Memory section)
torch.compile:  OFF
```

---

## Measured Results

### Step Timing (CUDA events, 20 runs)

| Metric | Value |
|--------|-------|
| Forward | 47.47 +/- 0.11 ms |
| Backward | 143.04 ms |
| **Fwd + Bwd** | **190.51 +/- 0.76 ms** |
| Bwd/Fwd ratio | 3.01x |
| Tokens/sec | 5,375 |
| Achieved | 14.2 TFLOPS |
| MFU | 6.8% |

MFU is low because B=2 S=512 (1,024 tokens/step) starves the GPU.
Real training uses gradient accumulation to hit ~524K tokens/step.

### Memory

```
Parameters (FP32):          7.8 GB
Peak (forward only):        9.55 GB
Peak (fwd + bwd):          17.15 GB
Available:                 32.0 GB
Remaining for optimizer:   14.85 GB
AdamW needs:               15.6 GB (2 FP32 buffers × 1.95B params)
                           ──────
                           OOM with optimizer at B>=2
```

### MoE Kernel Benchmarks (isolated, production shapes)

```
D=1280, E=64, K=8, inter=480
N=2048 tokens (B=4, S=512), 256 tokens/expert avg

Component                       ms      %
─────────────────────────────────────────
Expert BMM (batched):         0.531  49.0%    124.9 TFLOPS, AI=183
Dispatch (sort+expand):       0.232  21.4%
Gate routing:                 0.117  10.8%
Shared expert:                0.116  10.7%
Scatter combine:              0.090   8.3%
─────────────────────────────────────────
MoE total (per layer):        1.085
x14 MoE layers:              15.19   32% of forward
```

Expert BMM is compute-bound (AI=183 > ridge=117). Good.
Dispatch at 21% of MoE time is the overhead target.

---

## Kernel Profile (torch.profiler, 5 fwd+bwd steps)

Self CUDA time total: 756.6 ms (~151 ms/step)

### Top Kernels by Self CUDA Time

| Kernel | Self CUDA | % | Calls | Category |
|--------|-----------|---|-------|----------|
| `aten::copy_` | 249.1 ms | **32.9%** | 23,465 | CastLinear FP32->BF16 casts |
| `aten::cat` (CatArrayBatchedCopy) | 147.6 ms | **19.5%** | 1,195 | torch.stack/cat expert weights |
| `elementwise_kernel` (mul) | 124.8 ms | **16.5%** | 14,700 | SwiGLU, weight scale, residuals |
| `aten::bmm` | 119.4 ms | **15.8%** | 860 | Expert batched matmul |
| `vectorized_elementwise` | 77.4 ms | **10.2%** | 4,485 | Various elementwise (add, cast) |
| `aten::mm` | 76.2 ms | **10.1%** | 2,895 | MLA projections, lm_head |
| `unrolled_elementwise` | 43.5 ms | 5.7% | 2,340 | SiLU backward, grad elementwise |
| `fmha_cutlassB_bf16` (backward) | 32.8 ms | **4.3%** | 85 | FlashAttention backward |
| `aten::mul` | 30.0 ms | 4.0% | 6,390 | Weight application, scaling |
| `cutlass_80_tensorop_bf16` | 22.0 ms | 2.9% | 140 | MLA projection GEMMs |
| `aten::index` | 14.5 ms | 1.9% | 915 | Expert unpad (gather) |
| `_index_put_impl_` | 13.0 ms | 1.7% | 365 | Expert pad (scatter) |
| `DeviceRadixSortOnesweep` | 8.5 ms | **1.1%** | 1,570 | Expert dispatch argsort |

### Categorized Breakdown

```
Category                         CUDA ms/step    %
────────────────────────────────────────────────────
Weight copying (copy_ + cat)        79.3       52.5%
  aten::copy_ (CastLinear casts)    49.8       33.0%
  aten::cat (expert weight stack)   29.5       19.5%

Elementwise ops                     55.1       36.5%
  SwiGLU (silu + mul)               24.9       16.5%
  Vectorized elementwise            15.5       10.3%
  Unrolled elementwise               8.7        5.8%
  Mul (weight application)           6.0        4.0%

Expert GEMMs (bmm)                  23.9       15.8%

MLA GEMMs (mm)                      15.2       10.1%

FlashAttention (fwd + bwd)           6.6        4.4%

Expert dispatch (radix sort)          1.7        1.1%

Other (index, scatter, norms)         5.5        3.6%
────────────────────────────────────────────────────
TOTAL                              ~151.0      100%
```

---

## Key Findings

### 1. Weight copying is 52% of step time

`aten::copy_` (33%) + `aten::cat` (19.5%) = **52.5%** of CUDA time.

Two sources:
- **CastLinear**: stores FP32 master weights, casts to BF16 every forward.
  192 Linear layers in experts alone, each doing `weight.to(dtype=x.dtype)`.
- **torch.stack**: `_batched_expert_forward` stacks 64 expert weights into
  contiguous `[E, inter, D]` tensors every forward pass. 3 stacks per call.

This is the dominant bottleneck. Not compute. Not memory bandwidth. Copies.

### 2. Expert GEMMs are only 16%, not 45-55%

The TIER1 guide predicted expert compute at 45-55%. Actual: **15.8%**.
The batched BMM path (2 kernel launches) is already efficient.
The overhead around it (stacking weights, elementwise SwiGLU) dominates.

### 3. Dispatch is negligible (1.1%)

The guide worried dispatch must be <5%. Actual: **1.1%**.
`argsort(stable=True)` + `bincount` + `cumsum` is fast. No optimization needed.

### 4. FlashAttention is small (4.4%)

MLA's 23x KV compression makes attention matrices small.
No custom attention kernel needed for training.

### 5. Bwd/Fwd = 3.01x (expected 2-3x)

Slightly high. The extra cost is activation recomputation from CastLinear
(FP32 weights cast again during backward) and SwiGLU backward elementwise ops.

---

## Optimization Priority

| # | Fix | Expected Savings | Effort | Notes |
|---|-----|-----------------|--------|-------|
| P0 | Pre-stack expert weights at init | ~30 ms/step (19%) | Low | Store `[E, inter, D]` params, eliminate per-forward torch.stack |
| P1 | torch.compile or Liger fused kernels | ~20-30 ms/step (15-20%) | Low | Fuses SwiGLU, RMSNorm, casts, residuals into fewer kernels |
| P2 | Store weights in BF16 (optimizer keeps FP32 copy) | ~50 ms/step (33%) | Medium | Eliminates CastLinear copy_ overhead entirely |
| P3 | Gradient checkpointing (enable optimizer) | Memory only | Low | Needed to fit AdamW states in 32GB |
| P4 | Grouped GEMM (torch._grouped_mm) | ~5-10 ms/step (4-7%) | Medium | Replaces bmm, eliminates padding waste |

P0 + P1 alone: ~50 ms savings -> step from 190 ms to ~140 ms -> **26% speedup**.

P2 (if feasible with optimizer): another ~50 ms -> step ~90 ms -> **53% total speedup**.

---

## Decision Gate

```
Original question:  "Expert compute+dispatch+combine > 30%?"
Original answer:    "If yes -> write grouped GEMM kernel"

Actual measurement: Expert ops = 16% of step time.
                    Weight copy = 52% of step time.

DECISION: PIVOT.
  Do NOT write grouped GEMM first.
  Fix weight copying (P0+P2), then fuse elementwise (P1).
  Re-profile after. If expert GEMMs then dominate, write grouped GEMM.
```

---

## Next Steps

1. Implement P0: pre-stack expert weights in `MoE.__init__()`.
2. Implement P1: try `torch.compile(model)` and measure.
3. Re-profile to see new breakdown.
4. Enable gradient checkpointing, measure full training step with optimizer.
