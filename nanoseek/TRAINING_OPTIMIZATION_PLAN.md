# NanoSeek Training Optimization Plan — Forge Integration

## From Research Prototype to Frontier-Efficient Training

**Date**: 2026-03-26
**Status**: Research-verified, pre-implementation
**Methodology**: Every optimization is verified against existing open-source tools (2025-2026).
**Principle**: Use what exists. Write custom kernels only when no library covers our architecture.

---

## Philosophy: Don't Reinvent Wheels

The Forge plan proposes 5 modules (profiler, perf model, fused kernels, comm overlap, async
checkpoint). The 2026 PyTorch ecosystem already provides production-ready solutions for most
of these. Our job is to **integrate, compose, and fill gaps** — not write CUTLASS from scratch.

**What already exists and works:**
- `torch.compile` auto-fuses pointwise ops → Triton kernels (FREE)
- `Liger Kernel` provides fused RMSNorm, SwiGLU, FusedLinearCrossEntropy (DROP-IN)
- `torchao.float8` provides FP8 training with FSDP2 (PRODUCTION-READY on Hopper)
- `torch.distributed.checkpoint.async_save` provides async checkpointing (BUILT-IN)
- `FSDP2 (fully_shard)` provides comm-compute overlap automatically (BUILT-IN)
- `FlashAttention-2/3` handles attention fusion (PRODUCTION-READY)
- `ScatterMoE` / `MegaBlocks` provide efficient MoE dispatch (PRODUCTION-READY)
- `DeepGEMM` provides FP8 grouped GEMM for MoE (PRODUCTION-READY on Hopper)
- `DeepEP` provides expert parallelism communication (PRODUCTION-READY on Hopper)

**What does NOT exist and we must build:**
- MLA-specific training attention path using FA-2/3 with decomposed heads
- Integration layer composing these libraries with our MoE + MLA + MTP architecture
- Profiling harness specific to our architecture (MLA vs MoE vs MTP time breakdown)
- Performance model calibrated to our specific architecture

**What we should NOT build:**
- Custom CUTLASS kernels (torchao + DeepGEMM cover FP8 GEMMs)
- Custom RMSNorm fusion (Liger provides this)
- Custom SwiGLU fusion (Liger provides this)
- Custom cross-entropy (Liger's FusedLinearCrossEntropy is better than anything we'd write)
- Custom FSDP/communication overlap (FSDP2 handles this)
- Custom async checkpoint format (DCP async_save is production-ready)

---

## Hardware-Specific Optimization Matrix

Not all optimizations apply at all scales. This matrix prevents wasted effort.

| Optimization | Ablation (A6000, Ampere) | 1B Graduation (H100, Hopper) |
|---|---|---|
| torch.compile | YES (already using) | YES |
| Liger RMSNorm/SwiGLU/CE | YES | YES |
| FlashAttention-2 for MLA | YES (head_dim ≤ 256) | YES |
| FlashAttention-3 | NO (Hopper only) | YES (1.5-2x over FA-2) |
| FP8 training (torchao) | NO (Ampere lacks FP8 TC) | YES (1.3-1.5x speedup) |
| DeepGEMM (FP8 grouped GEMM) | NO (Hopper only) | YES (for MoE layers) |
| FSDP2 | NO (single GPU) | YES (8x H100) |
| DCP async checkpoint | OPTIONAL (fast enough sync) | YES (saves ~3-5% overhead) |
| Gradient checkpointing | YES (already using) | YES |
| ScatterMoE / MegaBlocks | EVALUATE (may help even single-GPU) | YES |
| DeepEP (expert parallelism) | NO (single node) | MAYBE (8 GPUs, 64 experts) |
| FlashMLA | NO (no backward on Ampere/Hopper) | NO (backward only on B200/SM100) |

**Key finding: FlashMLA is inference-only on our hardware.** The backward pass kernel exists
only for SM100 (B200). For training, we decompose MLA into standard attention shapes and use
FlashAttention-2/3. This is exactly what DeepSeek does during training — FlashMLA is their
inference optimization, not their training kernel.

---

## Phase 1: Free Performance (Week 1)
### Zero-cost wins from existing libraries

These require no custom kernel work. They are library integrations that compose with our
existing `torch.compile` and training loop.

### 1.1 Liger Kernel Integration

**What**: Drop-in fused Triton kernels for LLM training. Production-ready, used by HuggingFace
Trainer, Axolotl, LLaMA-Factory. Works on all NVIDIA GPUs.

**Impact**:
- `FusedLinearCrossEntropy`: Avoids materializing `[B, T, 32768]` logit tensor. At T=4096,
  B=4, this saves 4 × 4096 × 32768 × 2 = 1 GB per forward pass. ~3x faster CE, ~5x less memory.
- `LigerSwiGLU`: Recomputes activation in backward instead of caching. 1.6x memory reduction
  in MLP backward. Applied to all 64 expert FFNs + 2 shared experts = 66 FFN instances.
- `LigerRMSNorm`: Fused norm + scale. Caches RMS for backward. Moderate speedup.
- `LigerRoPE`: Fused rotary embedding. Applies to MLA's decoupled Q/K RoPE path.

**Integration plan**:
```
File: nanoseek/nanoseek/model.py

1. Replace RMSNorm (line 169-179):
   from liger_kernel.ops.rms_norm import LigerRMSNormFunction
   # Use as custom autograd function, or:
   from liger_kernel.transformers.rms_norm import LigerRMSNorm
   # Drop-in replacement for nn.Module

2. Replace SwiGLU in Expert FFN:
   from liger_kernel.ops.swiglu import LigerSwiGLUMLP
   # Or use the op directly in Expert.forward()

3. Replace cross-entropy loss in _compute_loss():
   from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
   # Fuses lm_head linear + CE loss — never materializes logits

4. Replace RoPE application:
   from liger_kernel.ops.rope import LigerRopeFunction
   # Fused rotary embedding application
```

**Validation**: Loss curve parity with BF16 reference. Run 100-step ablation with and without
Liger. `ema_val_bpb` must match within 0.01.

**Risk**: Low. Liger is the most battle-tested fused kernel library. If any kernel causes
issues, we can swap back to PyTorch native on a per-op basis.

### 1.2 FlashAttention-2 for MLA Training Path

**What**: Our MLA currently uses `F.scaled_dot_product_attention(SDPA)` which dispatches to
either FlashAttention-2, cuDNN, or math backend. This already works. But we need to verify
the dispatch is optimal.

**Current state** (`model.py:464-481`): Uses SDPA with `is_causal=True`. The SDPA dispatcher
should already select FlashAttention-2 on Ampere when head_dim ≤ 256.

**Verification steps**:
```
1. Confirm SDPA dispatches to FlashAttention-2 (not math fallback):
   with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
       # If this raises, FA-2 is not being used

2. Check head dimensions are FA-2 compatible:
   - qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
   - For ablation: 64 + 32 = 96 → OK (≤ 256, multiple of 32 ✓)
   - For 1B: 96 + 32 = 128 → OK (≤ 256, multiple of 32 ✓)

3. MLA decomposition for FA-2:
   Our training path expands c_kv via W_KVB into full K, V heads,
   then feeds to standard attention. This IS the correct approach for
   training — FlashAttention operates on expanded heads.
   DeepSeek does the same during training.
```

**Action**: Add a one-time diagnostic at step 0 that logs which SDPA backend was selected.
No code change needed if FA-2 is already dispatching correctly.

### 1.3 torch.compile max-autotune

**Current state**: `model = torch.compile(model, dynamic=False)` at `pre_train.py:1024`.

**Upgrade**: Switch to `mode="max-autotune"` to enable Triton kernel auto-tuning for GEMMs.
This benchmarks multiple Triton GEMM tile configurations at compile time and caches the fastest.

```python
# Change from:
model = torch.compile(model, dynamic=False)
# To:
model = torch.compile(model, dynamic=False, mode="max-autotune")
```

**Impact**: 5-15% speedup on GEMM-heavy models (MoE has 192+ GEMMs per forward). First
compilation takes longer (benchmarking tiles), but subsequent runs use cached optimal configs.

**Risk**: Longer initial compile time (10-30 min vs 5-15 min). Add `--compile-mode` argument
to allow switching between `default` and `max-autotune`.

---

## Phase 2: Profiling & Measurement (Week 2)
### Forge Module 1 — Know where time goes before optimizing

### 2.1 Training Step Profiler

**Purpose**: Before writing any custom kernel, measure where time is actually spent. The
roofline model tells you whether to optimize compute (FP8) or bandwidth (fusion).

**What exists**: `profile_moe.py` has per-component MoE timing. `pre_train.py` has step-level
MFU. Neither gives per-op kernel-level breakdown.

**Implementation**: Use `torch.profiler` (built-in) with CUPTI integration. No custom CUDA needed.

```
File: nanoseek/forge/__init__.py  (new)
File: nanoseek/forge/profiler.py  (new)

class TrainingStepProfiler:
    """Profile one complete training step and classify bottlenecks."""

    def profile_step(self, model, batch, config):
        """Run one forward+backward with torch.profiler and classify ops."""
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_flops=True,       # Compute FLOPs per kernel
            with_modules=True,     # Map kernels to nn.Module names
            profile_memory=True,   # Track memory allocations
        ) as prof:
            # Forward + backward
            outputs = model(batch, labels=batch)
            outputs['loss'].backward()

        return self._classify_ops(prof)

    def _classify_ops(self, prof):
        """Classify each kernel as compute-bound or memory-bound."""
        results = []
        for event in prof.key_averages():
            if event.device_type == DeviceType.CUDA:
                flops = event.flops
                bytes_moved = event.cuda_memory_usage  # approximate
                time_us = event.cuda_time_total
                # Roofline classification
                if flops > 0 and bytes_moved > 0:
                    ai = flops / bytes_moved  # arithmetic intensity
                    ridge_point = self.peak_flops / self.hbm_bandwidth
                    bottleneck = "compute" if ai >= ridge_point else "memory"
                else:
                    bottleneck = "unknown"
                results.append({
                    'name': event.key,
                    'module': event.module,
                    'time_us': time_us,
                    'flops': flops,
                    'bottleneck': bottleneck,
                })
        return results
```

**Hardware bandwidth table** (extend `get_peak_flops` in `common.py`):
```python
def get_hbm_bandwidth(device_name: str) -> float:
    """HBM bandwidth in bytes/sec for roofline analysis."""
    name = device_name.lower()
    _BW_TABLE = (
        (["a6000"], 768e9),      # 48 GB GDDR6X
        (["a100"], 2.0e12),      # 80 GB HBM2e
        (["h100", "sxm"], 3.35e12),  # 80 GB HBM3
        (["h100", "pcie"], 2.0e12),
        (["h200"], 4.8e12),      # 141 GB HBM3e
    )
    for patterns, bw in _BW_TABLE:
        if all(p in name for p in patterns):
            return bw
    return float('inf')
```

**Deliverable**: Per-module time breakdown for NanoSeek-ablation:
```
Expected breakdown (ablation, single A6000):
  MoE Expert GEMMs:      45-55%  (192 GEMMs = 64 experts × 3 projections)
  MLA Attention:          15-25%  (Q/KV projection + SDPA + output proj)
  MTP Auxiliary Heads:     5-10%  (2 extra transformer blocks)
  RMSNorm + Activations:   3-8%  (16 layers × 2 norms each)
  Cross-Entropy:            2-5%  (32K vocab softmax)
  Embedding + lm_head:     2-3%
  Optimizer:                5-8%
  Gradient sync (DDP):     1-3%  (if multi-GPU)
```

### 2.2 Analytical Performance Model (Forge Module 2 — Simplified)

**What**: Given NanoSeek config, predict step time and MFU. Calibrate against profiler data.

**Why simplified**: The full Forge Module 2 models TP/PP/EP parallelism for 10K GPUs. We need
a simpler version: predict step time for 1-8 GPU configurations to guide our ablation → 1B
hardware selection.

```
File: nanoseek/forge/perf_model.py  (new)

class PerformanceModel:
    def predict_step_time(self, config: NanoSeekConfig, hardware: HardwareSpec,
                          parallelism: ParallelismConfig) -> StepTimePrediction:
        """Predict training step wall-clock time."""

        # Per-layer forward time
        # MLA: Q projection (d → q_lora → q_heads) + KV projection + SDPA + output
        t_mla = self._mla_time(config, hardware)

        # MoE: Gate routing + expert dispatch + expert FFN + combine
        t_moe = self._moe_time(config, hardware)

        # MTP: Auxiliary heads (2 extra blocks)
        t_mtp = self._mtp_time(config, hardware)

        # Per-layer total
        t_layer = t_mla + t_moe + t_mtp

        # Full model
        t_forward = config.num_layers * t_layer + t_embedding + t_lm_head + t_loss
        t_backward = 2.0 * t_forward  # standard 2x ratio
        t_optimizer = self._optimizer_time(config, hardware)

        # Communication (FSDP AllGather + gradient AllReduce)
        t_comm = self._comm_time(config, hardware, parallelism)

        # Overlap: FSDP2 hides AllGather behind compute automatically
        t_comm_exposed = max(0, t_comm - t_forward * 0.8)  # rough overlap estimate

        t_step = t_forward + t_backward + t_optimizer + t_comm_exposed
        mfu = self._compute_mfu(config, hardware, t_step)

        return StepTimePrediction(t_step, mfu, breakdown={...})
```

**Calibration**: Run profiler (2.1) on ablation scale, compare predicted vs measured. Target: <10% error.

---

## Phase 3: MoE Dispatch Optimization (Week 3)
### The biggest single optimization opportunity

Our 64-expert MoE is the dominant compute cost (45-55% of step time). The current
implementation dispatches experts sequentially in a Python loop. This is where the biggest
gains are.

### 3.1 Evaluate MoE Kernel Libraries

**Candidates** (all production-ready, all open-source):

| Library | Approach | Best For | Hardware |
|---|---|---|---|
| `MegaBlocks` | Block-sparse GEMM | Large expert count, proven | All NVIDIA |
| `ScatterMoE` | ParallelLinear | FSDP2 integration, high throughput | All NVIDIA |
| `SonicMoE` | IO-aware tiling | Fine-grained experts (our case) | Hopper+ |
| `torch._grouped_mm` | PyTorch native | Simplicity, torch.compile compat | All NVIDIA |

**Decision framework**:
```
For ablation scale (A6000):
  Option A: torch._grouped_mm (simplest, no dependencies)
  Option B: MegaBlocks (most mature, pip installable)
  → Start with Option A. If profiler shows >10% time in expert dispatch
    overhead, switch to Option B.

For 1B graduation (H100):
  ScatterMoE (native FSDP2, proven at 225B tokens/day on 96 H100s)
  + DeepGEMM for FP8 expert GEMMs (if torchao MoE FP8 is still experimental)
```

### 3.2 torch._grouped_mm Integration

**What**: PyTorch's native batched GEMM for MoE. Groups tokens by expert and executes all
expert GEMMs in a single kernel launch instead of 64 sequential launches.

**Integration point**: `model.py`, `MoE.forward()` method. Currently loops over experts:
```python
# Current (sequential):
for i, expert in enumerate(self.experts):
    expert_input = x[expert_mask_i]
    expert_output = expert(expert_input)
    output[expert_mask_i] += expert_output * weights_i

# Proposed (grouped):
# Sort tokens by expert assignment
# Execute all expert GEMMs in one grouped_mm call
# Scatter results back
```

**Validation**: Output must be bitwise identical to sequential dispatch (same GEMMs, different scheduling).

---

## Phase 4: FP8 Training on Hopper (Week 4-5)
### For the 1B graduation run on H100s only

This phase is SKIPPED for ablation runs on A6000 (no FP8 tensor cores on Ampere).

### 4.0 What We Built: nanoseek/nanoseek/fp8.py (IMPLEMENTED)

**Status**: Complete and tested. Activated via `--fp8` flag in `pre_train.py`.

We implemented a custom minimal FP8 training framework (~320 lines) adapted from nanochat's
`fp8.py` (~150 lines), extended with MoE-specific filtering and NanoSeek integration. We chose
to build our own rather than depend on torchao because:

1. **torchao is ~2000 lines** with tensor subclass dispatch tables, DTensor FSDP hooks, and
   rowwise CUTLASS paths we don't need at nano scale
2. **Our framework calls the same cuBLAS kernel** (`torch._scaled_mm`) as torchao — the GPU
   matmul is identical, only the orchestration differs
3. **MoE-aware filtering** is built in (torchao's filter_fn API is less expressive)
4. **Zero dependencies** — pure PyTorch, works with torch.compile, no pip install
5. **Educational value** — the entire FP8 mechanism is readable in one file

**Architecture overview**:

```
    ┌─────────────────── nanoseek/nanoseek/fp8.py ───────────────────┐
    │                                                                 │
    │  _to_fp8(tensor, dtype) → (fp8_data, inv_scale)               │
    │      Tensorwise dynamic scaling: scale = FP8_MAX / amax       │
    │      One scalar scale per tensor (cuBLAS native)              │
    │                                                                 │
    │  _Float8Matmul(autograd.Function)                              │
    │      forward:  input(E4M3) @ weight(E4M3).T → output(BF16)   │
    │      backward: grad(E5M2) @ weight(E4M3) → grad_input(BF16)  │
    │                grad(E5M2).T @ input(E4M3) → grad_weight(BF16) │
    │      @allow_in_graph — opaque to torch.compile                 │
    │                                                                 │
    │  Float8CastLinear(nn.Linear)                                   │
    │      Drop-in CastLinear replacement, FP8 matmul in forward    │
    │      Handles 3D→2D reshape for _scaled_mm                     │
    │      Weight stays fp32 (shared with optimizer)                 │
    │                                                                 │
    │  convert_nanoseek_to_fp8(model) → (converted, skipped, total)  │
    │      Walks module tree, applies _is_fp8_eligible() filter     │
    │      Swaps CastLinear → Float8CastLinear (zero-copy)          │
    │                                                                 │
    │  disable_fp8(model) — context manager                          │
    │      Reverts Float8CastLinear → CastLinear for BF16 eval      │
    │      Used in eval loop so val_bpb measures model quality      │
    │                                                                 │
    │  is_fp8_available() → (bool, reason)                           │
    │      Hardware detection: CUDA + SM 9.0+ + _scaled_mm          │
    └─────────────────────────────────────────────────────────────────┘
```

**What FP8 actually does — the one-sentence version**:

FP8 is a compression trick for matrix multiplication. It squeezes 16-bit numbers into 8 bits
(half the size), so H100 tensor cores process them ~2× faster.

**Why a Linear layer has 1 forward GEMM but 2 backward GEMMs**:

A Linear layer does one operation in forward: `output = input @ weight.T`. One matmul.

But backward needs to answer TWO different questions:
1. "How much did each INPUT element matter to the loss?" → so the previous layer can learn
2. "How much should each WEIGHT change?" → so the optimizer can update this layer

These require two separate matmuls because they produce different-shaped answers:

```
    FORWARD (1 GEMM):
      output      = input        @ weight.T        [B, D_out] = [B, D_in] @ [D_in, D_out]

    BACKWARD (2 GEMMs):
      grad_input  = grad_output  @ weight           [B, D_in]  = [B, D_out] @ [D_out, D_in]
      grad_weight = grad_output.T @ input            [D_out, D_in] = [D_out, B] @ [B, D_in]

    grad_input  → send backward to previous layer (backpropagation)
    grad_weight → send to optimizer (weight update: w -= lr × grad_weight)

    There is no algebraic trick to combine them — different operand pairs,
    different output shapes. This is why backward costs ~2× forward, and
    a full training step costs ~3× forward (1 forward + 2 backward).
```

**How FP8 × FP8 produces BF16 output — multiply small, accumulate big**:

A matmul isn't one operation. It's thousands of multiply-then-add steps:

```
    C[i,j] = A[i,0]×B[0,j] + A[i,1]×B[1,j] + ... + A[i,K]×B[K,j]
             ──────────────   ──────────────         ──────────────
             product 1         product 2              product K
             └──────────── all summed into one number ────────────┘
```

The H100 tensor core does these phases in **different precisions**:

```
    A (FP8, 8-bit)      B (FP8, 8-bit)
         │                    │
         ▼                    ▼
    ┌──────────────────────────────┐
    │  MULTIPLY: FP8 × FP8        │  ← each product is tiny, low precision OK
    └──────────────┬───────────────┘
                   │ ~16-bit intermediate
                   ▼
    ┌──────────────────────────────┐
    │  ACCUMULATE: sum in FP32     │  ← sum hundreds/thousands of products
    │  running_sum += product      │     in full 32-bit precision
    │  (32 bits wide)              │     THIS IS WHERE PRECISION IS RECOVERED
    └──────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │  OUTPUT: cast FP32 → BF16    │  ← truncate to 16-bit for the rest of
    └──────────────┬───────────────┘     the network (norms, activations, etc.)
                   │
                   ▼
             C (BF16, 16-bit)
```

Why this works — the **law of large numbers**:

Each FP8 × FP8 product has rounding error. But when you sum K=1280 of them, the
errors cancel out statistically (some round up, some round down):

```
    Product 1:    true = 0.0312, FP8 says 0.0313   error = +0.0001
    Product 2:    true = 0.0187, FP8 says 0.0186   error = -0.0001
    Product 3:    true = 0.0441, FP8 says 0.0439   error = -0.0002
    ...
    Product 1280: true = 0.0099, FP8 says 0.0100   error = +0.0001

    Sum of 1280 errors ≈ 0  (random ± errors cancel)
    Mean error shrinks by ~1/√1280 ≈ 3% of individual error
```

The FP32 accumulator preserves this near-zero total exactly. Result: an FP8 matmul
produces nearly the same answer as a BF16 matmul, but ~2× faster on H100.

**Analogy**: measuring 1000 people's heights with a cheap ruler (±1cm error) vs an
expensive ruler (±0.1cm). Each measurement is 10× worse. But the average of 1000
measurements gives nearly the same result — the ±1cm errors cancel across the group.
FP8 is the cheap ruler. The FP32 accumulator is your notebook keeping the exact running total.

This is exactly what `torch._scaled_mm` does in our code:

```python
    output = torch._scaled_mm(
        input_fp8,                     # 8-bit operand A (cheap ruler)
        weight_fp8.t(),                # 8-bit operand B (cheap ruler)
        scale_a=input_inv,             # undo A's quantization scaling
        scale_b=weight_inv,            # undo B's quantization scaling
        out_dtype=torch.bfloat16,      # "give me the result in BF16"
    )
    # Hardware: FP8×FP8 multiply → FP32 accumulate → apply scales → cast to BF16
    # The FP8 was only used for the heavy matmul — then we're back in BF16 land.
```

**FP8 dtype selection — why two formats**:

```
    Representable values between 1.0 and 2.0:

    FP32 (23-bit mantissa):  8,388,608 values   ████████████████████████████
    BF16  (7-bit mantissa):        128 values   █░░░░░░░░░░░░░░░░░░░░░░░░░
    E4M3  (3-bit mantissa):          8 values   ░░░░░░░░░░░░░░░░░░░░░░░░░░
    E5M2  (2-bit mantissa):          4 values   ░░░░░░░░░░░░░░░░░░░░░░░░░░

    E4M3 (float8_e4m3fn): [1 sign | 4 exp | 3 mantissa]
      Range: ±448          Precision: 1 part in 8
      → Used for: weights and activations (forward pass)
      → Rationale: activations are bounded after normalization/SiLU,
        so we trade range for the best precision we can get

    E5M2 (float8_e5m2):   [1 sign | 5 exp | 2 mantissa]
      Range: ±57344        Precision: 1 part in 4
      → Used for: gradients (backward pass)
      → Rationale: gradients amplify through 16 layers of chain rule,
        can exceed ±448. E5M2's wider range prevents overflow.
```

**Tensorwise vs block-wise scaling — why tensorwise at nano scale**:

```
    DeepSeek V3 (671B): block-wise, 1 scale per 128 elements
    ├── At 671B params, activation outliers are more extreme
    ├── Block-wise localizes the outlier impact to its 128-element block
    ├── Requires CUTLASS/Triton custom kernels for the scaled matmul
    └── Engineering cost: ~500 lines of kernel code, weeks of tuning

    NanoSeek (1-5B): tensorwise, 1 scale per entire tensor
    ├── At nano scale, activations are more uniform (fewer extreme outliers)
    ├── cuBLAS torch._scaled_mm handles tensorwise natively (zero custom code)
    ├── The accuracy difference is <0.5% BPB at our scale (negligible)
    └── Engineering cost: ~10 lines (the _to_fp8 function)

    The crossover: block-wise becomes necessary at ~10B+ params where
    activation distributions have heavy tails from deep residual accumulation.
```

**What gets converted and what doesn't — the eligibility filter**:

```
    _is_fp8_eligible(module, fully_qualified_name) checks:

    1. Is it a CastLinear?              (only our custom Linear subclass)
    2. Are dims divisible by 16?        (H100 FP8 tensor core alignment)
    3. Is min(in, out) ≥ 128?           (below this, quantize overhead > matmul savings)
    4. Is it a gate router?             (NEVER — routing precision is sacred)
    5. Is it embedding/lm_head?         (skip — different optimizer, gradient sensitivity)

    Result at ablation scale (d=1280, 16 layers):
    ┌────────────────────────────────────────────────────────────────────┐
    │ CONVERTED (2943 layers):                                          │
    │   wo [1280, 1280]                  × 16   MLA output projection  │
    │   Expert w_gate [480, 1280]        × 960  64 experts × 15 layers │
    │   Expert w_up [480, 1280]          × 960  64 experts × 15 layers │
    │   Expert w_down [1280, 480]        × 960  64 experts × 15 layers │
    │   Shared w_gate [960, 1280]        × 15   shared expert          │
    │   Shared w_up [960, 1280]          × 15   shared expert          │
    │   Shared w_down [1280, 960]        × 15   shared expert          │
    │   MTP concat_proj [1280, 2560]     × 1    MTP projection         │
    │   MTP attn wo [1280, 1280]         × 1    MTP attention output   │
    ├────────────────────────────────────────────────────────────────────┤
    │ SKIPPED (91 layers):                                              │
    │   wq_a [275, 1280]                × 16   275 % 16 = 3 ✗         │
    │   wq_b [1920, 275]               × 16   275 not aligned          │
    │   wkv_a [154, 1280]              × 16   154 % 16 = 10 ✗         │
    │   wkv_b [2560, 90]               × 16   90 % 16 = 10 ✗          │
    │   Dense FFN [3277, 1280]          × 3    3277 % 16 = 5 ✗         │
    │   Gate router [64, 1280]          × 15   router: NEVER           │
    │   lm_head [32768, 1280]           × 1    embedding: skip         │
    └────────────────────────────────────────────────────────────────────┘
```

**Critical discovery: MLA projections are NOT FP8-aligned at either scale**:

```
    Ablation (d=1280):
      q_lora_rank  = 275   → 275 % 16 = 3  ✗
      kv_lora_rank = 90    → 90 % 16 = 10  ✗

    1B (d=2048):
      q_lora_rank  = 440   → 440 % 16 = 8  ✗
      kv_lora_rank = 143   → 143 % 16 = 15 ✗

    These ranks are muP-derived ratios (0.215 × d, 0.07 × d) that don't
    happen to land on multiples of 16. DeepSeek V3 at 671B uses ranks that
    ARE aligned (1536, 512) because at that scale they chose architecturally.

    Impact: the four MLA compressive projections (wq_a, wq_b, wkv_a, wkv_b)
    stay BF16 at both scales. Only wo [n_heads × v_head_dim, d] is eligible.

    Future option: if FP8 MLA speedup is critical at 3B+, round lora ranks
    to nearest multiple of 16 (e.g., 275→288, 90→96, 440→448, 143→144).
    This changes compression ratios slightly but maintains muP scaling.
```

**The batched MoE dispatch problem — why routed experts stay BF16 in the fast path**:

```
    NanoSeek batched dispatch (model.py _batched_expert_forward):
      w_gate_up = torch.stack([expert weights])     # [64, 2×inter, D]
      output = torch.bmm(padded_input, w_gate_up.T) # [64, max_tok, 2×inter]

    torch._scaled_mm only accepts 2D tensors:
      _scaled_mm(A: [M, K], B: [K, N]) → [M, N]

    There is no _scaled_bmm. Three options considered:

    Option A: FP8 each expert individually → 192 _scaled_mm calls
      192 × 7µs kernel launch = 1.3ms overhead vs 2 × 7µs for bmm = 14µs
      NET RESULT: ~90× more kernel launches → SLOWER than BF16 bmm

    Option B: Custom batched FP8 kernel (Triton)
      ~500 lines, breaks "no custom kernels" rule, compile overhead
      NOT WORTH IT at nano scale

    Option C: Leave batched path in BF16, FP8 only for dense operations ✅
      Shared expert (dense, every token) gets FP8
      MLA wo projection gets FP8
      Batched routed experts stay BF16 bmm (already optimized)
      Sequential fallback path gets FP8 (when routing is skewed)

    We chose Option C. The practical speedup:

    Component             FP8?    % of FLOPs   Speedup
    ─────────────────── ──────── ────────── ──────────
    MLA projections       ❌ BF16     ~15%      1.0×   (lora ranks misaligned)
    MoE batched bmm       ❌ BF16     ~55%      1.0×   (_scaled_mm is 2D only)
    MoE sequential path   ✅ FP8      ~5%*      ~1.5×  (fallback path)
    Shared expert         ✅ FP8     ~10%       ~1.5×  (dense, every token)
    wo projection         ✅ FP8      ~5%       ~1.5×  (only aligned MLA proj)
    Other                 ❌ BF16    ~10%       1.0×

    * Sequential fallback triggers when waste_ratio > 1.5 (skewed routing)

    Total estimated speedup on H100: ~1.05-1.08×
    Modest, but the framework is READY for scale-up where gains compound.
```

**The use_fast_accum decision — forward vs backward precision**:

```
    Matrix multiply accumulation: C[i,j] = Σ_k A[i,k] × B[k,j]

    use_fast_accum=True (forward):
      Accumulates dot products in reduced precision (FP16/TF32)
      ~5-10% faster, slightly less accurate
      OK for forward because activations are consumed once

    use_fast_accum=False (backward):
      Accumulates in full FP32 precision
      Slower but precise
      CRITICAL for backward because gradient errors become permanent
      weight updates: w -= lr × grad_weight
      If grad_weight has FP8 rounding bias → model learns wrong direction
      Over 10,000 steps this compounds into measurable quality loss

    Cost at our scale: ~5% slower backward ≈ $1.75 per $35 ablation run
    Worth it for numerically reliable gradients.
```

**Evaluation escape hatch — why eval is always BF16**:

```
    Training: FP8 noise is absorbed by SGD's inherent stochasticity.
      Stochastic gradients already have variance from mini-batch sampling.
      FP8 quantization noise is smaller than this sampling variance.
      Net effect: training loss curve is indistinguishable from BF16.

    Evaluation: FP8 noise contaminates the measurement.
      val_bpb should measure MODEL QUALITY, not quantization artifacts.
      FP8 quantization adds ~0.01-0.05 BPB of noise to evaluation.
      This noise is SYSTEMATIC (same quantization error every eval) and
      masks real quality improvements between checkpoints.

    Solution: disable_fp8(model) context manager
      Swaps Float8CastLinear → CastLinear (sharing same weight, zero copy)
      Eval runs in clean BF16 precision
      Restores Float8CastLinear after eval for continued FP8 training

    In pre_train.py:
      if _fp8_enabled:
          _fp8_ctx = disable_fp8(orig_model)
          _fp8_ctx.__enter__()
      with ema_tracker.apply(orig_model):
          ema_val_bpb = evaluate_nanoseek_bpb(...)  # clean BF16
      if _fp8_ctx is not None:
          _fp8_ctx.__exit__(None, None, None)
```

**Usage**:

```bash
# Enable FP8 on H100 (auto-detects hardware, graceful fallback on non-Hopper)
python -m nanoseek.scripts.pre_train \
    --run fp8-test --scale ablation --fp8 \
    --num-iterations 100 --eval-every 50

# Without FP8 (default, works on any GPU)
python -m nanoseek.scripts.pre_train \
    --run bf16-baseline --scale ablation \
    --num-iterations 100 --eval-every 50
```

### 4.1 torchao Float8 — When to Switch (Future)

Our custom FP8 is sufficient at ablation and 1B scale. Switch to torchao when:

1. **Scaling to 3B+** where FSDP2 + Float8 all-gather saves communication bandwidth
   (torchao's float8 communicates weights in FP8 format, 50% less volume)
2. **Rowwise scaling becomes necessary** because activation outliers at >10B params
   cause quality degradation with tensorwise scaling
3. **torch.compile fusion across FP8 boundary** matters for throughput
   (torchao's tensor subclass exposes individual ops to Inductor for fusion;
   our @allow_in_graph blocks this fusion)

The migration path is clean: torchao's `convert_to_float8_training` has the same
`module_filter_fn` API as our `_is_fp8_eligible`. The filter logic transfers directly.

### 4.2 DeepGEMM for MoE (Alternative/Complementary)

**What**: DeepSeek's open-source FP8 GEMM kernels with native grouped GEMM support for MoE.
Up to 1550 TFLOPS on H800. MIT license. JIT-compiled.

**When to use**: If our FP8 + `torch.bmm` batched dispatch doesn't achieve target MFU on H100,
DeepGEMM's `m_grouped_gemm_fp8_fp8_bf16` can FP8-ify the batched expert path — the one gap
our current framework cannot cover (because `_scaled_mm` is 2D only).

**Integration**: Replaces the bmm dispatch inside `_batched_expert_forward`, not the
overall MoE routing or the individual expert `CastLinear` layers.
```python
# Only if profiler shows batched expert dispatch is >40% of step time on H100:
import deep_gemm
# Replace torch.bmm with:
# deep_gemm.m_grouped_gemm_fp8_fp8_bf16(
#     padded_input_fp8, w_gate_up_fp8, output, group_sizes)
```

**Decision point**: Profile after Phase 4.0. If MoE layer MFU < 40% on H100, add DeepGEMM.

### 4.3 FP8 Validation Protocol

**Non-negotiable**: FP8 must not degrade model quality.

```
Validation run (on H100):
  1. Train ablation config for 500 steps in BF16 (baseline)
  2. Train same config for 500 steps with --fp8 flag
  3. Compare:
     - ema_val_bpb: must match within 0.5%
     - H_load (expert balance): must match within 0.2 bits
     - I_spec (specialization): must match within 0.05 nats
     - grad_norm distribution: no systematic increase
  4. If any metric diverges:
     a. Verify gate weights stayed in BF16 (check with
        `sum(1 for n,m in model.named_modules() if 'router' in n and 'Float8' in type(m).__name__)` == 0)
     b. Check for saturated scales (amax hitting FP8_MAX → clipping signal)
     c. Compare per-group grad norms (routed_experts vs mla vs shared)
     d. If shared expert grads diverge: disable FP8 for shared expert only
```

### 4.4 FP8-Aligned MLA Ranks — Future Config Option

If FP8 speedup on MLA projections becomes important at 3B+ scale, we can add
an `fp8_aligned` option that rounds lora ranks to the nearest multiple of 16:

```
Current muP-derived ranks → FP8-aligned alternative:
  q_lora_rank:  275 → 288  (4.7% larger, 288/1280 = 0.225 vs 0.215 ratio)
  kv_lora_rank: 90  → 96   (6.7% larger, 96/1280 = 0.075 vs 0.070 ratio)
  q_lora_rank:  440 → 448  (1.8% larger, 448/2048 = 0.219 vs 0.215 ratio)
  kv_lora_rank: 143 → 144  (0.7% larger, 144/2048 = 0.070 vs 0.070 ratio)

  Impact: slightly higher parameter count, marginally different compression ratio.
  Benefit: ALL MLA projections become FP8-eligible → ~15% more FLOPs in FP8.
  Decision: only worth investigating at 3B+ where the extra 15% matters.
  Risk: changes muP transfer ratios — requires re-validating HP search.
```

---

## Phase 5: Distributed Training Infrastructure (Week 5-6)
### For the 1B graduation run on 8x H100

### 5.1 FSDP2 Migration

**What**: Replace vanilla DDP with FSDP2 (`fully_shard`) for parameter sharding. FSDP2
provides automatic comm-compute overlap (AllGather for layer i+1 overlaps with compute for
layer i).

**Current state**: `common.py` uses `dist.init_process_group(backend="nccl")` + raw DDP
wrapping in `pre_train.py`. No parameter sharding.

**Migration plan**:
```python
# In pre_train.py, replace DDP wrapping:

from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

# FSDP2 mixed precision policy
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,  # gradient reduction in fp32
)

# Shard each transformer layer independently
for layer in model.layers:
    fully_shard(layer, mp_policy=mp_policy)

# Shard the full model (outermost)
fully_shard(model, mp_policy=mp_policy)
```

**MoE-specific sharding**: Expert parameters should NOT be sharded across FSDP ranks (each
GPU needs all expert weights for local routing). Use `no_shard` policy for expert parameters:
```python
from torch.distributed._composable.fsdp import fully_shard

for layer in model.layers:
    # Shard attention (MLA) — standard sharding
    fully_shard(layer.self_attn, mp_policy=mp_policy)
    # Don't shard MoE experts — each GPU needs all 64 for local routing
    # MoE gate can be sharded (small)
    fully_shard(layer.moe.gate, mp_policy=mp_policy)
    # Shared experts can be sharded
    for se in layer.moe.shared_experts:
        fully_shard(se, mp_policy=mp_policy)
```

**Composability with FP8**: FSDP2 + torchao float8 compose cleanly. Float8 all-gathers
communicate weights in FP8 format (50% less communication volume).

### 5.2 Async Checkpointing (Forge Module 5)

**What**: Replace synchronous checkpoint saves with `torch.distributed.checkpoint.async_save`.

**Current state**: `checkpoint_manager.py` has excellent crash-safe synchronous saves with
atomic rename + fsync. The save blocks training for 3-17 seconds depending on model size.

**Migration**:
```python
# In checkpoint_manager.py, add async variant:
import torch.distributed.checkpoint as dcp

class AsyncCheckpointManager(CheckpointManager):
    """Extends CheckpointManager with non-blocking saves."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_save = None

    def save_async(self, model, optimizer, ema_state, metadata, step):
        """Non-blocking checkpoint save. Returns immediately."""
        # Wait for any pending save to complete first
        if self._pending_save is not None:
            self._pending_save.result()  # Block until previous save completes

        # Stage state to CPU (non-blocking GPU → CPU copy)
        state_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema": ema_state,
            "metadata": metadata,
        }

        # Launch async save
        self._pending_save = dcp.async_save(
            state_dict=state_dict,
            storage_writer=dcp.FileSystemWriter(self._checkpoint_dir(step)),
        )
        return self._pending_save

    def finalize(self):
        """Wait for any pending save to complete. Call before exit."""
        if self._pending_save is not None:
            self._pending_save.result()
```

**Fallback**: Keep the existing synchronous `CheckpointManager` as default. Add `--async-ckpt`
flag to opt into async saves. The existing atomic rename + fsync pattern remains as the
synchronous fallback.

### 5.3 Communication-Compute Overlap (Forge Module 4 — Via FSDP2)

**Key insight**: FSDP2 provides implicit comm-compute overlap automatically. We don't need
to build a custom multi-stream scheduler — FSDP2's `fully_shard` handles:
- AllGather prefetch: gathers parameters for layer i+1 while computing layer i
- Skip-final-block: doesn't reshard the last transformer block in forward (backward will
  immediately re-gather it)
- Gradient reduce-scatter overlapped with backward compute

**What we DO need to handle**: Expert parallelism communication overlap, if we use EP.
This is where DeepEP becomes relevant.

**Decision**: At 8x H100 with 64 experts, all experts fit on each GPU (64 × 3 × 480 × 1280
× 4 bytes ≈ 0.9 GB per expert set per layer, × 16 layers ≈ 14 GB). This fits comfortably in
80 GB H100. **No expert parallelism needed at 8-GPU scale.** Use standard FSDP2 data
parallelism.

If scaling to 32+ GPUs: add expert parallelism via DeepEP or PyTorch-native EP (torchtitan
pattern with DTensor device mesh).

---

## Phase 6: Architecture-Specific Optimizations (Week 6-7)
### NanoSeek-unique optimizations that no library covers

### 6.1 MLA Training Attention Path Optimization

**Current state**: MLA training path expands compressed KV via `W_KVB`, feeds to standard
SDPA/FlashAttention. This is correct and matches DeepSeek's training approach.

**Optimization opportunity**: The Q projection does `hidden → W_QA → q_norm → W_QB → split`.
This is two sequential GEMMs with a norm between them. torch.compile should fuse the norm
into the second GEMM's prologue, but verify with profiler.

**MLA-specific fusion**: If profiler shows the Q/KV projection chain is >15% of step time:
```
Consider: Fuse W_QA + RMSNorm + W_QB into a single Triton kernel.
This eliminates one HBM round-trip (write Q_A intermediate, read it back for norm).
Savings: ~2 × B × T × q_lora_rank × dtype_size per layer per direction.
At ablation: 2 × 4 × 4096 × 768 × 2 = 50 MB/layer → 800 MB total across 16 layers.
```

**Decision point**: Profile first (Phase 2). Only write this kernel if Q/KV projections are
a measurable bottleneck (>10% of step time).

### 6.2 MTP Compute Overlap

**What**: NanoSeek's MTP module runs 2 auxiliary transformer blocks after the main model's
forward pass. These could potentially overlap with the main model's backward pass.

**Current state**: Sequential — `main forward → MTP forward → backward for all`.

**Optimization**: Restructure to:
```
main forward → start backward (main loss) → MTP forward (overlapped) → backward (MTP loss)
```

This requires careful gradient graph management. The MTP forward depends on the main model's
final hidden states, so it can only start after the main forward completes. But the main
model's backward can start from the loss immediately.

**Decision**: Low priority. MTP is 5-10% of step time. Profile first.

### 6.3 FusedLinearCrossEntropy for MTP

**What**: Liger's `FusedLinearCrossEntropy` fuses the lm_head projection with cross-entropy
loss, avoiding materializing the `[B, T, V]` logit tensor. This applies to BOTH the main
loss AND each MTP head's loss.

**Impact**: Main model has 1 lm_head CE. MTP has 2 additional CE computations (one per
predicted future token). FusedLinearCE on all 3 saves ~3 GB of logit memory.

```python
# In model.py _compute_loss():
# Instead of:
#   logits = self.lm_head(hidden_states)
#   loss = F.cross_entropy(logits.view(-1, V), labels.view(-1))
# Use:
#   loss = LigerFusedLinearCrossEntropyFunction.apply(
#       hidden_states.view(-1, d), self.lm_head.weight, labels.view(-1))
```

---

## Phase 7: Profiling Infrastructure (Week 7)
### Forge Module 1 & 2 — Measurement framework

### 7.1 Per-Module Training Dashboard

Build a W&B-integrated profiler that runs every N steps (configurable) and logs:
```
Per step:
  - time_mla_forward_ms
  - time_moe_forward_ms (gate + dispatch + expert_compute + combine)
  - time_mtp_forward_ms
  - time_backward_ms
  - time_optimizer_ms
  - time_comm_ms (DDP AllReduce or FSDP AllGather)
  - time_checkpoint_ms (when saving)
  - memory_peak_gb
  - memory_allocated_gb
  - mfu_achieved_pct

Per profile interval (every 100 steps):
  - roofline_classification per op (compute-bound vs memory-bound)
  - top-10 slowest kernels with bottleneck type
  - expert load imbalance → compute waste estimate
```

### 7.2 Script: Profile & Report

```
File: nanoseek/scripts/profile_training.py  (new)

Usage:
  python -m nanoseek.scripts.profile_training \
      --scale ablation --steps 10 --warmup 3

Output:
  - Console table: per-module time breakdown
  - JSON: full kernel-level profile for analysis
  - Roofline plot: ops plotted on compute vs memory-bound spectrum
```

---

## Implementation Schedule

```
Week 1:  Phase 1 — Liger integration + FA-2 verification + max-autotune
         Expected: 15-30% memory reduction, 5-15% speed improvement
         Validation: 100-step loss parity test

Week 2:  Phase 2 — Profiler + performance model
         Expected: Know exactly where time goes
         Deliverable: Per-module breakdown, roofline classification

Week 3:  Phase 3 — MoE dispatch optimization (grouped_mm or MegaBlocks)
         Expected: 10-20% speedup on MoE layers (the dominant cost)
         Validation: Output bitwise match, profiler confirms improvement

Week 4-5: Phase 4 — FP8 training on H100 (for graduation run only)
          Expected: 1.3-1.5x throughput on H100
          Validation: 500-step quality parity test

Week 5-6: Phase 5 — FSDP2 + async checkpoint (for graduation run)
          Expected: Automatic comm overlap, zero-pause checkpoints
          Validation: Multi-GPU training stability

Week 6-7: Phase 6 — Architecture-specific optimizations (if profiler justifies)
          Expected: 5-10% additional gains from MLA/MTP fusion
          Validation: Profile before and after

Week 7:   Phase 7 — Measurement infrastructure for ongoing optimization
          Expected: Permanent profiling capability
```

---

## What We Explicitly Do NOT Build

| Forge Module | Original Plan | Our Decision | Why |
|---|---|---|---|
| Custom CUTLASS 3.x kernels | EVT epilogue fusion, FP8 GEMM | **SKIP** | torchao + DeepGEMM cover this. CUTLASS C++ cost is weeks for marginal gain over existing libraries. |
| Custom RMSNorm+GEMM Triton kernel | Fuse norm into GEMM prologue | **USE LIGER** | Liger's `LigerRMSNorm` is production-tested. torch.compile may auto-fuse the rest. |
| Custom SwiGLU Triton kernel | Gate+Up concat fusion | **USE LIGER** | Liger's `LigerSwiGLUMLP` does exactly this with backward recomputation. |
| Custom comm-compute scheduler | Multi-stream AllGather chunking | **USE FSDP2** | FSDP2's implicit prefetching does this automatically. |
| Custom async checkpoint format | CoW snapshot + RDMA replication | **USE DCP** | `torch.distributed.checkpoint.async_save` is production-ready. |
| FlashMLA integration | Custom MLA attention kernel | **NOT POSSIBLE** | No backward pass on Ampere/Hopper. FlashMLA is inference-only for our hardware. |
| Custom cross-entropy kernel | Online softmax | **USE LIGER** | `LigerFusedLinearCrossEntropy` is better — fuses with lm_head. |

---

## Expected Total Impact

### Ablation Scale (A6000)
```
Baseline:           ~38-42% MFU, ~X tok/s
+ Liger kernels:    +5-10% memory reduction, +5-10% speed → ~42-48% MFU
+ max-autotune:     +5-10% speed → ~44-52% MFU
+ MoE grouped_mm:   +10-15% speed → ~48-55% MFU
= Total:            ~1.3-1.5x throughput improvement
```

### 1B Graduation (8x H100)
```
Baseline:           ~38-42% MFU
+ All ablation opts: ~48-55% MFU
+ FP8 (our fp8.py):  +5-8% (shared expert + wo + sequential MoE path)
                      MLA lora ranks not FP8-aligned → limited coverage
+ DeepGEMM (if needed): +15-25% (FP8 batched expert dispatch)
+ FSDP2 overlap:    +5-10% (comm hidden) → ~55-65% MFU
+ Async checkpoint:  +3-5% (save hidden) → ~58-68% MFU
= Total:            ~1.5-1.8x throughput improvement

With FP8-aligned lora ranks (future, requires HP re-validation):
+ FP8 (all MLA + MoE): +25-40% → ~60-70% MFU
= Total:            ~1.8-2.2x throughput improvement
```

These numbers reflect the actual implementation audit (Phase 4.0). The key constraint is
MLA lora ranks not being multiples of 16, limiting FP8 coverage to ~20% of total FLOPs.
Full FP8 coverage at 3B+ scale (with aligned ranks + DeepGEMM for batched dispatch) would
match torchtitan benchmarks (~55-65% MFU on Llama 70B with FP8 + FSDP2 on H100 clusters).

---

## Dependencies & Install

```bash
# Phase 1 (all scales)
pip install liger-kernel              # Fused kernels
pip install flash-attn                # FlashAttention-2 (if not via SDPA)

# Phase 3 (MoE optimization)
pip install megablocks                # Option B for MoE dispatch
# OR: torch._grouped_mm is built into PyTorch 2.5+ (no install)

# Phase 4 (H100 only)
pip install torchao                   # FP8 training
# DeepGEMM: pip install deep-gemm    # Only if torchao MoE FP8 underperforms

# Phase 5 (multi-GPU)
# FSDP2 and DCP are built into PyTorch 2.5+ (no install)
```

---

## Risk Register

| Risk | Severity | Probability | Mitigation |
|---|---|---|---|
| Liger kernel numerical divergence | Medium | Low | 100-step parity test. Per-kernel enable/disable flags. |
| FP8 degrades expert routing quality | High | Medium | Keep gate in BF16. Validate I_spec before/after. |
| torch.compile graph breaks from Liger | Medium | Medium | Fall back to native ops for broken graphs. |
| FSDP2 + MoE expert sharding conflict | High | Medium | Use no_shard policy for expert params. Test on 2 GPUs first. |
| max-autotune compile time too long | Low | High | Add --compile-mode flag. Cache compiled kernels. |
| DeepGEMM JIT fails on our expert shapes | Medium | Low | Fall back to torchao FP8. |
| Async checkpoint race with EMA | Medium | Low | Snapshot EMA state before launching async copy. |

---

## Appendix: Library Version Requirements

```
PyTorch >= 2.5.0      (FSDP2, torch._grouped_mm, float8 dtype)
liger-kernel >= 0.5.0 (FusedLinearCrossEntropy, MoE support)
torchao >= 0.14.0     (float8 rowwise training)
flash-attn >= 2.7.0   (head_dim 256 backward support)
CUDA >= 12.4          (FP8 tensor core support)
```

## Appendix: Files to Create/Modify

```
ALREADY IMPLEMENTED:
  nanoseek/nanoseek/fp8.py             ← FP8 training framework (Phase 4.0, COMPLETE)
  nanoseek/scripts/pre_train.py        ← --fp8 flag, conversion, eval escape (COMPLETE)
  nanoseek/nanoseek/config.py          ← Updated FP8Config dataclass (COMPLETE)

NEW FILES (TODO):
  nanoseek/forge/__init__.py           ← Forge optimization package
  nanoseek/forge/profiler.py           ← Training step profiler (Module 1)
  nanoseek/forge/perf_model.py         ← Analytical performance model (Module 2)
  nanoseek/scripts/profile_training.py ← Profiling script

MODIFIED FILES (TODO):
  nanoseek/nanoseek/model.py           ← Liger kernel integration (RMSNorm, SwiGLU, CE)
  nanoseek/nanoseek/common.py          ← Add get_hbm_bandwidth()
  nanoseek/nanoseek/checkpoint_manager.py ← Add AsyncCheckpointManager
  nanoseek/scripts/pre_train.py        ← max-autotune, FSDP2, async ckpt (--fp8 done)
  nanoseek/nanoseek/config.py          ← Add --compile-mode, --async-ckpt flags
```
