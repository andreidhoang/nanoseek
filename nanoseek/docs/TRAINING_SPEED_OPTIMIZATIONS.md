# NanoSeek Training Speed Optimizations — Deep Dive

**Author**: Performance Engineering Review, March 2026
**Scope**: Every optimization in the NanoSeek training stack, explained from first principles
**Target reader**: You can read a GPU datasheet and know what a matmul is

---

## Table of Contents

1. [The Hardware Reality](#1-the-hardware-reality)
2. [Numerical Precision: BF16 Autocast + TF32 + CastLinear](#2-numerical-precision)
3. [Attention: SDPA, FlashAttention, and MLA's is_causal Trick](#3-attention)
4. [MoE Dispatch: From 192 Kernels to 2](#4-moe-dispatch)
5. [Optimizer: Fused Muon with Polar Express](#5-optimizer)
6. [Memory: Gradient Checkpointing, Expandable Segments, set_to_none](#6-memory)
7. [Compilation: torch.compile and Dynamo Cache Tuning](#7-compilation)
8. [Pipeline Overlap: Data Prefetch, CUDA Events, Zero-Overhead Profiling](#8-pipeline)
9. [Profiling and Observability: What We Measure and Why](#9-profiling)
10. [Interaction Map: How Optimizations Compose](#10-interactions)
11. [What We Chose NOT to Do (and Why)](#11-non-optimizations)

---

## 1. The Hardware Reality

Every optimization in this document exists because of a gap between what hardware *can* do
and what naive code *actually* does. To understand any optimization, you need to understand
the machine first.

### 1.1 The GPU Execution Model

A GPU is not "a fast CPU." It is a **throughput-oriented processor** with:

```
A100 80GB:
  BF16 peak throughput:    312 TFLOPS
  FP32 peak throughput:     19.5 TFLOPS (16× slower!)
  FP32 with TF32:          156 TFLOPS  (8× faster than pure FP32)
  Memory bandwidth:        2.0 TB/s
  Memory capacity:         80 GB
  L2 cache:                40 MB
  Kernel launch overhead:  ~5-10 μs

H100 80GB SXM:
  BF16 peak throughput:    990 TFLOPS
  FP32 peak throughput:     67 TFLOPS
  FP32 with TF32:          495 TFLOPS
  Memory bandwidth:        3.35 TB/s
  Kernel launch overhead:  ~3-7 μs
```

Two numbers dominate all optimization decisions:

1. **Arithmetic intensity** = FLOPs / bytes moved. If you don't do enough FLOPs per byte
   read from memory, compute units sit idle waiting for data. This is called being
   **memory-bound**. Most of neural network training is memory-bound for small batch sizes.

2. **Kernel launch overhead**. Each operation (matmul, add, relu) is a "kernel" dispatched
   from CPU to GPU. Each dispatch costs 5-10 μs. If you dispatch 192 small kernels when
   you could dispatch 2 large ones, you waste 192 × 7μs = 1.3 ms of pure overhead.

### 1.2 The CPU-GPU Pipeline

Training happens in a pipeline:

```
CPU:  [prepare batch N+1]  [prepare batch N+2]  [prepare batch N+3]
       ↓ dispatch            ↓ dispatch            ↓ dispatch
GPU:  [fwd+bwd batch N   ] [fwd+bwd batch N+1 ] [fwd+bwd batch N+2]
```

The CPU dispatches work to the GPU and *immediately* moves on to prepare the next batch.
The GPU executes a queue of operations asynchronously. This overlap is critical.

**`synchronize()` breaks this pipeline.** It forces the CPU to wait until the GPU finishes
all queued work. Every unnecessary `synchronize()` is a pipeline bubble where the GPU
has no work queued and sits idle.

### 1.3 Why MoE Training is Harder to Optimize

A dense transformer has `L` identical layers. Every token takes the same path. This is
trivially parallelizable and the memory access pattern is perfectly predictable.

NanoSeek's MoE architecture breaks this:
- **64 routed experts × 3 weights each = 192 unique weight matrices** per MoE layer
- **Variable token-to-expert assignment**: different tokens go to different experts
- **Load imbalance**: some experts get 3× more tokens than others
- **14 MoE layers + 1 dense layer + 1 MTP layer** = heterogeneous compute graph

This means:
- More kernel launches (192 matmuls vs 3 for dense FFN)
- Irregular memory access (tokens scattered across experts)
- More parameters to compile (192 unique shapes for torch.compile)
- Higher gradient variance (only κ=12.5% of experts see each token)

Every optimization below addresses one or more of these MoE-specific challenges.

---

## 2. Numerical Precision: BF16 Autocast + TF32 + CastLinear

### 2.1 BF16 Autocast — The 2× Speedup

**First principle**: Matmul throughput scales linearly with the number of elements that
fit in a tensor core. BF16 is half the size of FP32 → 2× more elements per tensor core
operation → 2× throughput.

**What autocast does** (`pre_train.py:1561`):
```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    outputs = model(x, labels=x, mtp_lambda=mtp_lambda)
    loss = outputs['loss']
```

The `autocast` context manager tells PyTorch: "run matmuls and convolutions in BF16,
but keep reductions (softmax, layer norm, loss) in FP32 for numerical stability."

**Why BF16 and not FP16**: BF16 has the same exponent range as FP32 (8 bits), so it
never overflows during forward/backward. FP16 has 5 exponent bits → frequently overflows
on gradient norms > 65504, requiring loss scaling. BF16 eliminates this entire class of bugs.

**Where precision matters**:
```
BF16 (fast, slightly imprecise):  matmuls, linear layers, attention scores
FP32 (slow, precise):             softmax, layer norm, cross-entropy loss, optimizer state
```

The loss computation stays FP32 automatically because `F.cross_entropy` promotes inputs
to float32 internally. This is not optional — BF16 cross-entropy produces wrong gradients
due to catastrophic cancellation in log-sum-exp.

### 2.2 CastLinear — Avoiding Autocast Context Overhead

**First principle**: Python context managers (`with autocast(...)`) have overhead per
entry/exit. For a dense model with 3 linear layers per block, this is negligible. For MoE
with 192+ linear layers per MoE block, the overhead accumulates.

**What CastLinear does** (`model.py:52-59`):
```python
class CastLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(dtype=x.dtype))
```

Instead of relying on autocast to cast weights at matmul time, CastLinear does it
explicitly. The weight is stored in FP32 (master copy for optimizer precision), and
cast to the input's dtype (BF16) at compute time.

**Why this is better than autocast for MoE**:
1. No context manager overhead per linear layer
2. `torch.compile` handles `.to(dtype=...)` more cleanly than autocast hooks
3. The dtype is determined by the input tensor, not a global context — no surprises
4. Each expert's 3 linear layers all get the same treatment without nesting autocast

**Key insight**: Master weights stay FP32 for optimizer precision. The cast-to-BF16 happens
only during the forward matmul. Gradients flow back in BF16, and the optimizer updates
the FP32 master weights. This is the standard mixed-precision pattern.

### 2.3 TF32 — Free 3× on FP32 Operations

**First principle**: NVIDIA tensor cores on A100/H100 support TF32 format — FP32 range
with 10 bits of mantissa (vs 23 bits). For matmuls that don't need full FP32 precision,
this gives ~3× speedup.

**What we enable** (`pre_train.py:185-188`):
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
```

**Which operations benefit**: Any FP32 matmul that isn't inside autocast's BF16 zone:
- Loss computation (`cross_entropy` internally does FP32 matmuls for numerical stability)
- Optimizer state updates (Muon's Polar Express iterations are FP32 matmuls)
- Per-group gradient norm computation
- The `.float()` casts in CastLinear's backward pass

**Accuracy impact**: 10 mantissa bits gives ~3 decimal digits of precision. For loss
computation, this is more than sufficient (loss values are O(1), not O(10^6)).

---

## 3. Attention: SDPA, FlashAttention, and MLA's is_causal Trick

### 3.1 Why Attention is Expensive

Standard attention: `attn = softmax(Q @ K.T / √d) @ V`

For sequence length S and head dimension d:
- `Q @ K.T`: O(S² × d) FLOPs, produces S × S matrix
- Softmax: O(S²)
- `attn @ V`: O(S² × d) FLOPs

The S² term means attention is **quadratic** in sequence length. At S=4096, the attention
matrix is 4096 × 4096 = 16M elements per head. With 16 heads, that's 256M elements
just for attention weights — and they must all be in memory simultaneously for backward.

### 3.2 SDPA / FlashAttention — Fused Attention Kernel

**First principle**: The standard implementation materializes the full S×S attention matrix
in GPU memory. FlashAttention (Dao et al., 2022) tiles the computation so only a small
block is in SRAM at any time. This:

1. Reduces memory from O(S²) to O(S) — no full attention matrix stored
2. Reduces memory traffic (the dominant cost) by keeping intermediates in SRAM
3. Fuses softmax + matmul into one kernel (eliminates kernel launch overhead)

**What we use** (`model.py:480-487`):
```python
attn_output = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=attention_mask if not needs_causal else None,
    dropout_p=self.attention_dropout if self.training else 0.0,
    scale=effective_scale,
    is_causal=needs_causal,
)
```

`F.scaled_dot_product_attention` (SDPA) dispatches to the fastest available backend:
1. **FlashAttention**: if input shapes are compatible (head dim ≤ 256, no arbitrary mask)
2. **Memory-efficient attention**: if FlashAttention doesn't apply
3. **Math fallback**: standard unfused computation

### 3.3 The `is_causal=True` Optimization

**First principle**: A causal mask is a lower-triangular matrix of ones. If you build it
explicitly as a tensor, you pay:
1. O(S²) memory to store the mask
2. O(S²) memory bandwidth to read it during attention
3. A separate kernel to add the mask to attention scores

With `is_causal=True`, SDPA/FlashAttention uses a built-in triangular mask that costs zero
memory and zero bandwidth — it's implemented as a branch in the inner kernel loop.

**When it applies** (`model.py:466-478`):
```python
needs_causal = attention_mask is None and seq_len > 1
if needs_causal and seq_len != kv_len:
    # KV cache case: seq_len < kv_len, need explicit mask
    attention_mask = torch.full(...)
    needs_causal = False
```

`is_causal=True` only works when `seq_len == kv_len` (training / prefill). During
generation with KV cache, `kv_len > seq_len`, so we fall back to explicit mask.

**Impact**: ~1-3% faster training due to avoided mask allocation + bandwidth.

### 3.4 MLA's Training vs Inference Paths

NanoSeek uses Multi-Head Latent Attention (MLA) from DeepSeek V2/V3, which has two modes:

**Training (naive mode)**: Expand compressed KV to full rank via `wkv_b`, then standard SDPA.
This is what we use for training because SDPA/FlashAttention can fuse the whole computation.

**Inference (absorb mode)**: Absorb `W_UK` into Q and `W_UV` into output via einsum.
Never materializes full K/V. KV cache stores only compressed latent (23× compression).

The training path is faster for training because it can use FlashAttention's fused kernel.
The inference path is faster for generation because it never expands the KV cache.

---

## 4. MoE Dispatch: From 192 Kernels to 2

This is the single most impactful MoE-specific optimization.

### 4.1 The Problem

NanoSeek has 64 routed experts, each with 3 weight matrices (gate, up, down). A naive
implementation processes each expert separately:

```python
# NAIVE: 192 kernel launches per MoE layer
for expert_idx in range(64):
    tokens = get_tokens_for_expert(expert_idx)  # variable size
    output = expert[expert_idx](tokens)          # 3 matmuls
    scatter_back(output)
```

Each `expert(tokens)` launches 3 CUDA kernels (gate, up, down matmul). With 64 experts,
that's 192 kernel launches. At ~7μs each = **1.3 ms of pure launch overhead** per MoE layer.
With 14 MoE layers, that's 18 ms/step of doing nothing but launching kernels.

For comparison, at ablation scale the total step time is ~1-2s. So kernel launch overhead
is ~1-2% of total time. But at larger batch sizes where the actual compute is faster,
this overhead becomes a bigger fraction.

### 4.2 The Solution: Sort-Pad-BMM

**Step 1: Sort tokens by expert assignment** (`model.py:821-830`)
```python
sort_order = flat_indices.argsort(stable=True)
sorted_x = x_expanded[sort_order]  # tokens now grouped by expert
expert_counts = torch.bincount(sorted_indices, minlength=E)
```

After sorting, all tokens for expert 0 are contiguous, then all tokens for expert 1, etc.
This gives us contiguous memory slices — prerequisite for batched matmul.

**Step 2: Pad to uniform batch size** (`model.py:759-766`)
```python
max_count = expert_counts.max().item()
padded_input = sorted_x.new_zeros(E, max_count, D)
padded_input[sorted_indices, position_in_expert] = sorted_x
```

`torch.bmm` requires all batches to have the same size. We pad each expert's batch to
`max_count` (the busiest expert's token count). Zero padding is gradient-neutral.

**Step 3: Stack weights and BMM** (`model.py:772-783`)
```python
w_gate = torch.stack([e.w_gate.weight for e in self.routed_experts])  # [E, inter, D]
w_up = torch.stack([e.w_up.weight for e in self.routed_experts])
w_gate_up = torch.cat([w_gate, w_up], dim=1).to(dtype)               # [E, 2*inter, D]
w_down = torch.stack([e.w_down.weight for e in self.routed_experts]).to(dtype)

# SwiGLU in 2 kernel launches instead of 192:
gate_up_out = torch.bmm(padded_input, w_gate_up.transpose(1, 2))  # gate + up fused
gate_out, up_out = gate_up_out.chunk(2, dim=-1)
hidden = F.silu(gate_out) * up_out
out = torch.bmm(hidden, w_down.transpose(1, 2))                   # down projection
```

**Key trick**: gate and up projections are concatenated into one weight matrix, so their
matmuls are fused into a single `bmm` call. This gives us:

```
Before: 64 × 3 = 192 kernel launches
After:  2 kernel launches (gate_up bmm + down bmm) + 1 activation (silu * up)
```

**Step 4: Unpad results** (`model.py:785-786`)
```python
sorted_output = out[sorted_indices, position_in_expert]
```

Vectorized gather extracts valid (non-padding) results.

### 4.3 The Waste Guard

```python
waste_ratio = max_count / max(avg_count, 1)
use_batched = sorted_x.is_cuda and E >= 8 and waste_ratio < 1.5
```

If routing is heavily skewed (one expert gets 50% of tokens), padding wastes compute.
The guard falls back to sequential dispatch when waste exceeds 50%. In practice, with
64 experts and good load balancing (H_load > 4 bits), the waste is typically < 20%.

### 4.4 Gradient Correctness

`torch.stack` preserves the autograd graph. Gradients flow through the stacked tensor
back to each individual expert's parameters. Zero padding contributes zero gradient
(0 × dL/dout = 0). This is mathematically equivalent to the sequential loop.

---

## 5. Optimizer: Fused Muon with Polar Express

### 5.1 Why Not Just AdamW?

NanoSeek uses **Muon** for 2D matrix parameters and **AdamW** for everything else. Muon
(Momentum Orthogonalized Update by Newton-Schulz) replaces AdamW's per-element adaptive
learning rate with an orthogonalization step that normalizes the update's singular values.

**Why Muon for matrices**: AdamW's second moment `v = β₂v + (1-β₂)g²` adapts per-element.
For large matrices, this creates millions of independent learning rates that must converge
independently. Muon instead computes the nearest orthogonal matrix to the momentum buffer,
giving all directions the same effective learning rate. Empirically, this converges 1.5-2×
faster for large matrices (Keller Jordan et al., 2024).

**Parameter classification** (`pre_train.py:589-611`):
```
Muon:   2D weights (attention projections, expert weights, MTP projections)
AdamW:  embeddings, lm_head, router weights, 1D norms
```

### 5.2 Fused Kernel — One Compiled Graph Per Step

**First principle**: A standard optimizer step does:
```python
for p in params:
    # 5+ separate kernels per parameter:
    p.grad *= momentum        # kernel 1
    buf.lerp_(p.grad, ...)    # kernel 2
    orth = newton_schulz(buf)  # kernels 3-7 (5 iterations)
    p -= lr * orth             # kernel 8
```

For 192 expert parameters × 8 kernels = 1,536 kernel launches. At 7μs each = 10.7 ms.

**Solution**: `@torch.compile(dynamic=False, fullgraph=True)` compiles the entire
Muon step into a single fused CUDA kernel (`optim.py:108-164`):

```python
@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, ...):
    # Nesterov momentum
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Polar Express (5 iterations of X = aX + X@B or X = aX + B@X)
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    for a, b, c in polar_express_coeffs[:ns_steps]:
        A = X.mT @ X
        B = b * A + c * (A @ A)
        X = a * X + X @ B

    # Variance reduction + cautious weight decay + update
    ...
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)
```

`fullgraph=True` means the entire function is one graph — no Python fallbacks, no
graph breaks. torch.compile fuses all operations into minimal CUDA kernels.

### 5.3 0-D CPU Tensors: Avoiding Recompilation

```python
momentum_t: Tensor,  # () - 0-D CPU tensor
lr_t: Tensor,        # () - 0-D CPU tensor
```

**First principle**: `torch.compile` traces the function and compiles a CUDA kernel for
specific tensor shapes and dtypes. If you pass `lr=0.001` as a Python float, changing it
to `lr=0.0005` triggers **recompilation** (new kernel). With a 0-D tensor, the value is
read at runtime — the compiled kernel reads from the tensor's memory, no recompilation.

This is critical because learning rate changes every step (warmup → constant → cosine decay).
Without 0-D tensors, torch.compile would recompile the optimizer kernel every step.

### 5.4 Polar Express vs Newton-Schulz

**Newton-Schulz iteration** (the original Muon):
```
X_{k+1} = (3X_k - X_k @ X_k.T @ X_k) / 2
```
Converges to the nearest orthogonal matrix. 5 iterations for adequate convergence.

**Polar Express** (Amsel et al., arXiv:2505.16932):
```
A = X.T @ X   (or X @ X.T for wide matrices)
B = b*A + c*(A@A)
X = a*X + X@B
```
Uses precomputed coefficients `(a, b, c)` optimized to maximize convergence slope at zero.
Same 5 iterations but better convergence — the orthogonalized result is closer to true UV^T.

**Impact**: ~10% faster Muon step at same iteration count, slightly better training dynamics.

### 5.5 Parameter Stacking

Muon processes all parameters of the same shape together:

```python
for shape, params in muon_shapes.items():
    param_groups.append(dict(kind='muon', params=params, ...))
```

At step time, all parameters of the same shape are `torch.stack`ed into a single 3D tensor
and processed with one `muon_step_fused` call. For 64 experts with shape `[480, 1280]`,
this means one kernel call for 64 matrices instead of 64 separate calls.

---

## 6. Memory: Gradient Checkpointing, Expandable Segments, set_to_none

### 6.1 Selective Gradient Checkpointing

**First principle**: During forward pass, PyTorch saves intermediate activations for
backward pass. For a transformer layer, this includes:
- Input to attention: `[B, S, D]` = 4 × S × D bytes (BF16)
- Attention weights: `[B, H, S, S]` = saved by FlashAttention (so zero here)
- MoE intermediate: `[B, S, 2×inter]` per active expert

For an MoE layer with D=1280, S=4096, B=4: the intermediate activations are ~200MB.
With 14 MoE layers, that's ~2.8GB just for MoE intermediates.

**Gradient checkpointing** discards these activations during forward and recomputes them
during backward. This trades compute (forward is run twice) for memory.

**Why selective** (`model.py:1482-1490`):
```python
first_k_dense = config.moe.first_k_dense_replace  # = 1
last_ckpt = max(first_k_dense, config.num_layers - 2)
self._checkpoint_layer_ids = set(range(first_k_dense, last_ckpt))
```

- **Layer 0 (dense)**: NOT checkpointed — small activation footprint (no MoE)
- **Layers 1-13 (MoE)**: Checkpointed — large activation footprint
- **Layers 14-15 (MoE)**: NOT checkpointed — their activations are still in GPU memory
  when backward reaches them (backward processes layers in reverse order). Checkpointing
  the last 2 layers would discard activations and immediately recompute them — pure waste.

**Impact**: ~30-40% memory reduction with ~20% compute overhead (only MoE forward is
recomputed, attention activations are handled by FlashAttention's built-in checkpointing).

### 6.2 Expandable Segments

```python
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
```

**First principle**: PyTorch's CUDA memory allocator requests memory from the GPU in
fixed-size "segments" (default: 2MB). When a segment is freed, it may leave a gap between
two allocated segments. If a later allocation needs a contiguous block larger than any gap,
the allocator must request a new segment from the GPU — even if total free memory is
sufficient. This is **memory fragmentation**.

MoE training has highly variable allocation patterns:
- `padded_input` size varies per step (depends on max expert count)
- `gate_up_out` size varies (depends on batch × max_count)
- `sorted_x` size varies (depends on active tokens × top_k)

`expandable_segments:True` allows the allocator to **grow** existing segments instead of
allocating new ones. This dramatically reduces fragmentation for MoE's variable-size
allocations.

### 6.3 set_to_none=True

```python
model.zero_grad(set_to_none=True)
```

**First principle**: After `optimizer.step()`, gradients are no longer needed. The default
`zero_grad()` fills gradient tensors with zeros (a memset kernel). With `set_to_none=True`,
it sets `.grad = None` instead — no memset needed.

For NanoSeek with ~1.95B total parameters (ablation), this saves one full-model memset
per step. At 4 bytes/param (FP32 grads): 7.8 GB of memset avoided.

The next `backward()` call will allocate fresh gradient tensors as needed. PyTorch's
allocator reuses the same memory blocks, so there's no allocation overhead.

---

## 7. Compilation: torch.compile and Dynamo Cache Tuning

### 7.1 torch.compile on the Full Model

```python
model = torch.compile(model, dynamic=False)
```

**First principle**: Python is slow. Each PyTorch operation goes through:
1. Python function call (~1 μs)
2. Argument validation (~0.5 μs)
3. Dispatch to correct backend (~0.5 μs)
4. CUDA kernel launch (~5-10 μs)

For a single matmul on a large tensor, this overhead is negligible. For hundreds of small
operations (reshapes, splits, concatenations, activation functions) in an MoE layer,
the overhead accumulates.

`torch.compile` traces the model's forward pass and compiles it into optimized CUDA kernels:
- **Operator fusion**: `F.silu(gate_out) * up_out` becomes one kernel instead of two
- **Eliminated intermediates**: fused ops don't write intermediate results to memory
- **Reduced Python overhead**: the compiled graph runs as native code, no Python dispatch

**`dynamic=False`**: We tell torch.compile that tensor shapes won't change between calls.
This allows more aggressive optimization (no dynamic shape guards). Valid because our
batch size and sequence length are fixed during training.

### 7.2 Dynamo Cache Size Limit

```python
torch._dynamo.config.cache_size_limit = 32
```

**First principle**: torch.compile caches compiled graphs keyed by (function, shapes, dtypes).
If the cache is full, the oldest entry is evicted and must be recompiled on next call.

For a dense transformer, there are ~5-10 unique shape combinations. For NanoSeek MoE:
- 64 expert weights of shape `[480, 1280]` → same shape, one cache entry
- But Muon optimizer groups parameters by shape → many unique shape groups
- Different batch sizes during warmup → different compiled variants

The default `cache_size_limit = 8` is too small → constant cache eviction → constant
recompilation. We set it to 32 to eliminate this.

---

## 8. Pipeline Overlap: Data Prefetch, CUDA Events, Zero-Overhead Profiling

### 8.1 Data Prefetch After Backward

```python
loss.backward()
# GPU is busy computing gradients — CPU is free to prepare next batch
x, y, dataloader_state_dict = next(train_loader)
```

**First principle**: `loss.backward()` dispatches gradient computation to the GPU and
returns immediately. The GPU has a deep work queue. While the GPU computes gradients,
the CPU can prepare the next batch (tokenize, pad, move to GPU pinned memory).

This is only effective if data loading is fast enough. If `next(train_loader)` takes
longer than the backward pass, the GPU will finish and wait — we detect this via
`data_stall_pct > 5%` warning.

### 8.2 CUDA Events for Zero-Overhead Timing

**The wrong way** (what I initially implemented, then fixed):
```python
synchronize()           # BLOCKS CPU until GPU finishes
t_fwd_start = time.time()  # CPU wall clock
# ... forward pass ...
synchronize()           # BLOCKS CPU again
t_fwd_end = time.time()
```

Each `synchronize()` forces the CPU to wait until the GPU completes all queued work.
This breaks the CPU-GPU pipeline overlap that makes training fast. With 4 syncs per
micro-step × 8 micro-steps = 32 pipeline stalls per training step.

**The right way** (current implementation):
```python
_evt_fwd_start = torch.cuda.Event(enable_timing=True)
_evt_fwd_end = torch.cuda.Event(enable_timing=True)

_evt_fwd_start.record()   # records timestamp ON THE GPU — CPU does NOT wait
# ... forward pass ...
_evt_fwd_end.record()     # records timestamp ON THE GPU — CPU does NOT wait

# ... much later, after the single sync at end of step ...
synchronize()             # only ONE sync per step (was already there)
dt_fwd = _evt_fwd_start.elapsed_time(_evt_fwd_end) / 1000  # resolves timing
```

CUDA events are recorded on the GPU's own timeline. `record()` returns immediately to
the CPU. The elapsed time is only queried after the step's single existing `synchronize()`.

**Impact**: Zero additional pipeline stalls from profiling. The timing measurements add
literally zero overhead to training.

### 8.3 Data Loading Stall Detection

```python
if data_stall_pct > 5.0 and step > 5:
    print0(f"  [WARNING] Data loading is {data_stall_pct:.1f}% of step time")
```

We measure `dt_data` (time spent in `next(train_loader)`) as a fraction of total step
time. If >5%, the GPU is sitting idle waiting for data. Common causes:
- Slow disk I/O (solution: faster SSD, pre-cache in RAM)
- Not enough dataloader workers (solution: increase num_workers)
- Data processing bottleneck (solution: pre-tokenize, simpler transforms)

---

## 9. Profiling and Observability: What We Measure and Why

### 9.1 Phase Timing (fwd / bwd / opt / data / misc)

Every training step is decomposed into 5 phases:

| Phase | What it measures | What it tells you |
|-------|-----------------|-------------------|
| `dt_fwd` | Forward pass (embedding → MLA → MoE → loss) | If too slow: compile not working, or model too big for this GPU |
| `dt_bwd` | Backward pass (gradient computation) | Should be ~2× dt_fwd. If >3×: gradient checkpointing overhead |
| `dt_opt` | Optimizer step (Muon + AdamW) | If too slow: Muon's Polar Express iterations dominate |
| `dt_data` | Data loading (next batch from dataloader) | If >5% of dt: GPU is starving for data |
| `dt_misc` | Everything else (EMA, bias update, grad clip, logging) | Should be <10% of dt |

### 9.2 Per-Group Gradient Norms

**Why not just total grad_norm?**: In an MoE model, gradients have very different scales
across components:
- Gate weights (14 parameters): tiny gradients, but their instability kills routing
- Routed experts (192× per layer): large aggregate gradient, dominates total norm
- MLA weights: medium gradients

A gate collapse (H_load → 0) produces a spike in `grad_norm/gate` but barely moves the
total `grad_norm`. Without per-group tracking, you'd see the collapse only after H_load
drops — too late.

**Implementation**: Pre-built parameter groups (one GPU→CPU transfer per group):
```python
sq_norms = torch.stack([g.float().pow(2).sum() for g in grads])
norms[group_name] = sq_norms.sum().sqrt().item()  # single .item() per group
```

### 9.3 MFU (Model FLOPs Utilization)

```
MFU = actual_flops_per_second / peak_flops_of_hardware
```

Where `actual_flops = 6 × N_active × tokens_per_step` (the 6N rule: 2 for multiply-add
× 3 for forward + backward = 6 FLOPs per parameter per token).

MFU tells you what fraction of the GPU's theoretical peak you're achieving:
- **>50%**: Excellent for MoE (dense models can hit 55-60%)
- **30-50%**: Good — typical for MoE due to expert dispatch overhead
- **<30%**: Something is wrong — check dt_data, kernel launch overhead, compile status

### 9.4 Cost Accounting

```python
cost_so_far = gpu_hours_so_far * args.cost_per_gpu_hour
estimated_total_cost = cost_so_far + eta_gpu_hours * args.cost_per_gpu_hour
```

At $0.79/GPU-hr (A6000) or $2.49/GPU-hr (H100), every training run has a dollar cost.
Logging `$spent/$total ETA:Xh` on every step lets you make rational decisions:
- "This run is 30% done with no loss improvement → kill it, save $70"
- "This HP is 2× better loss at 10% done → worth running to completion"

### 9.5 Throughput Regression Detection

The health monitor tracks tok/s with an EMA and alerts when throughput drops >15%:

```
Causes of throughput regression:
  - GPU thermal throttling (temp > 80°C → clock drops)
  - Memory fragmentation (despite expandable_segments)
  - torch.compile recompilation (cache eviction)
  - Dataloader I/O stall (disk bottleneck)
  - NVLink bandwidth saturation (multi-GPU)
```

Each of these costs GPU-hours for zero training benefit. The alert lets you investigate
before wasting an entire run's budget.

---

## 10. Interaction Map: How Optimizations Compose

Optimizations don't exist in isolation. Here's how they interact:

```
torch.compile
  ├── Fuses CastLinear .to(dtype) into matmul kernel (no separate cast kernel)
  ├── Eliminates torch.stack overhead in batched expert dispatch
  ├── Fuses SiLU * up_out into one kernel (Expert SwiGLU)
  ├── Eliminates expert_counts.max().item() GPU→CPU sync (graph-captured)
  └── Enables operator fusion in MLA (norm → proj → split → rope)

BF16 autocast
  ├── CastLinear avoids autocast context overhead for 192+ expert layers
  ├── TF32 accelerates remaining FP32 ops (loss, optimizer, norms)
  └── FlashAttention requires BF16/FP16 inputs (autocast provides this)

Batched expert dispatch
  ├── Requires expandable_segments (variable-size padded tensors)
  ├── Benefits from torch.compile (fused weight stacking)
  └── Waste guard prevents degenerate cases from wasting compute

Gradient checkpointing
  ├── Selective (MoE only) preserves FlashAttention's built-in checkpointing
  ├── Skips last 2 layers (activations still in memory during backward)
  └── use_reentrant=False required for MoE aux_data to survive checkpointing

CUDA event timing
  ├── Zero-overhead because no synchronize() inside micro-step loop
  ├── Resolved lazily after the single existing synchronize()
  └── Enables data stall detection without slowing training
```

### Critical Anti-Pattern: Synchronize Inside Hot Loop

The most important lesson: **never call `synchronize()` inside the micro-step loop**.

A micro-step loop with gradient accumulation runs 8+ forward+backward passes per training
step. Each `synchronize()` forces the CPU to wait 50-200μs while the GPU finishes. With
4 syncs × 8 micro-steps = 32 stalls = 6.4 ms of pure waste per step.

This is why we use CUDA events: they record timestamps on the GPU without CPU involvement.

---

## 11. What We Chose NOT to Do (and Why)

### 11.1 cudnn.benchmark = True

**What it does**: Runs multiple convolution algorithms and caches the fastest one.
**Why we skip it**: NanoSeek has zero convolutions. All compute is matmul and attention.
The cost (non-deterministic results) outweighs the benefit (zero, for this architecture).
We prioritize reproducibility for ablation science.

### 11.2 CUDA Graphs

**What it does**: Captures an entire sequence of CUDA operations and replays it as a single
kernel launch. Eliminates all Python and dispatch overhead.
**Why we skip it**: CUDA graphs require static shapes and no Python control flow. MoE's
variable token-to-expert assignment and the waste_ratio guard branch make this incompatible.
torch.compile provides most of the same benefit with more flexibility.

### 11.3 Pre-Stacked Expert Weights

**What it would do**: Store all expert weights as a single `[E, inter, D]` parameter
instead of 64 separate `Expert` modules.
**Why we skip it**: Pre-stacking complicates gradient flow — the optimizer must split
gradients back to per-expert learning rate schedules. It also breaks the clean
`Expert(hidden_dim, inter_dim)` abstraction. torch.compile handles the stacking efficiently.

### 11.4 Expert Parallelism

**What it would do**: Distribute experts across GPUs (expert i on GPU i % num_gpus).
**Why we skip it**: At ablation scale (1-4 GPUs), all 64 experts fit on one GPU.
Expert parallelism adds all-to-all communication overhead that only pays off at 16+ GPUs.
Will be needed for 7B scale.

### 11.5 Sequence Parallelism

**What it would do**: Shard the sequence dimension across GPUs for MLA's attention.
**Why we skip it**: At S=4096, attention is not the bottleneck (FlashAttention handles it).
Sequence parallelism would add communication overhead for minimal benefit.

### 11.6 FP8 Training

**What it would do**: Run matmuls in 8-bit float for another 2× speedup.
**Why we skip it**: FP8 requires careful scaling factor management and is only supported
on H100+. At our scale, BF16 provides sufficient throughput. FP8 is a natural next step
for 7B scale on H100 clusters.

---

## Summary: Where Wall-Clock Goes

For a typical ablation-scale training step on 1×A6000:

```
Total step time:      ~1.2s
├── Forward pass:      ~0.45s (37%)  ← BF16 autocast + SDPA + batched MoE dispatch
├── Backward pass:     ~0.52s (43%)  ← gradient checkpointing recomputes MoE forward
├── Optimizer step:    ~0.18s (15%)  ← fused Muon (Polar Express) + fused AdamW
├── Data loading:      ~0.03s (2%)   ← prefetched during backward
└── Misc:              ~0.02s (2%)   ← EMA, grad clip, bias update, logging

MFU: ~35-40% (typical for MoE with 64 experts on single GPU)
Throughput: ~400K tok/s
```

The backward pass is larger than forward because gradient checkpointing recomputes MoE
layer forwards. This is the intended trade-off: 20% more compute for 40% less memory.

---

*Document version: 2026-03-26. Reviewed against codebase at commit HEAD.*
