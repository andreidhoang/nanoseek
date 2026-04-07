# NanoSeek Performance Optimization Plan
## From First Principles to Maximum Training Speed
### Target: 2-3x throughput improvement, 50-65% cost reduction

---

## Table of Contents

1. [First Principles: Where Time Goes](#1-first-principles-where-time-goes)
2. [Day 1 Playbook: Profile Before Optimize](#2-day-1-playbook-profile-before-optimize)
3. [Tier 1: Zero-Risk Quick Wins (Day 1-2)](#3-tier-1-zero-risk-quick-wins)
4. [Tier 2: Medium-Effort High-Impact (Day 2-3)](#4-tier-2-medium-effort-high-impact)
5. [Tier 3: Advanced Optimizations (Day 4+)](#5-tier-3-advanced-optimizations)
6. [Codebase Audit: Specific Issues Found](#6-codebase-audit-specific-issues-found)
7. [Expected Results & MFU Targets](#7-expected-results--mfu-targets)
8. [Cost-Optimal Rental Strategy](#8-cost-optimal-rental-strategy)

---

## 1. First Principles: Where Time Goes

### The Training Step Equation

```
step_time = T_forward + T_backward + T_optimizer + T_communication + T_data + T_overhead
```

For a MoE model on 8xH100, the typical breakdown:

| Component          | % of Step | Bottleneck Type      | Optimization Lever               |
|--------------------|-----------|----------------------|----------------------------------|
| MoE Expert FFNs    | 35-45%    | Memory bandwidth     | Fused kernels, FP8, compile      |
| MLA Attention      | 15-25%    | Compute (if Flash)   | FlashAttention, compile          |
| Gradient AllReduce | 8-15%     | Communication        | Overlap, compression, no_sync    |
| Optimizer (Muon)   | 5-10%     | Compute              | Overlap with backward            |
| MoE Dispatch       | 5-10%     | Kernel launches      | Fused dispatch, eliminate syncs  |
| Data Loading       | 1-5%      | CPU / IO             | Prefetch, pin memory             |
| Python Overhead    | 2-5%      | CPU                  | torch.compile                    |

### Why MoE Expert FFNs Are Memory-Bound (Critical Insight)

Each expert processes a small batch: with B=16, S=4096, top-8 routing, 64 experts:
```
tokens_per_expert = (16 * 4096 * 8) / 64 = 8,192 on average
```

Expert FFN arithmetic intensity at D=2048, inter=768:
```
FLOPs = 6 * 8192 * 2048 * 768 = 77.3 GFLOP
Bytes = 6 * 2048 * 768 * 2 + 4 * 8192 * 2048 * 2 = 18.9 MB + 134 MB = 153 MB
AI = 77.3e9 / 153e6 = ~505 FLOP/byte
```

H100 SXM ridge point = 989 TFLOPS / 3.35 TB/s = ~295 FLOP/byte.

At 8192 tokens per expert, the expert GEMMs are **compute-bound** (AI > ridge point).
But the batched dispatch (padding, stacking, scatter/gather) is **memory-bound**.
The overhead AROUND the compute is the bottleneck, not the compute itself.

### Why MFU for MoE Is Lower Than Dense

| Factor                       | MFU Loss  | Explanation                                    |
|------------------------------|-----------|------------------------------------------------|
| Expert dispatch overhead     | 5-10%     | Sort, pad, scatter, gather, weight stacking    |
| GPU-CPU sync points          | 2-5%      | `.item()` calls in MoE routing                 |
| torch.compile graph breaks   | 5-15%     | MoE conditional logic prevents kernel fusion   |
| Communication (DDP)          | 5-10%     | 4.75B param gradient AllReduce                 |
| Python overhead              | 2-5%      | Interpreter, autograd bookkeeping              |
| **Total MFU loss**           | **19-45%**| **Target: push from 25-35% to 40-50% MFU**    |

---

## 2. Day 1 Playbook: Profile Before Optimize

**Rule: Measure first, optimize second. Never optimize based on assumptions.**

### Step 0: Hardware Verification (2 min)

```bash
# Verify GPU type, count, memory
nvidia-smi

# Verify NVLink topology (critical for DDP performance)
nvidia-smi topo -m
# Expected: all 8 GPUs connected via NVLink (NV12 or NV18)
# Red flag: any GPU connected via PCIe only

# Check CUDA version
nvcc --version
# Need: CUDA 12.1+ for H100 optimizations

# Check PyTorch version
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name())"
```

### Step 1: Smoke Test — Model Loads and Runs (5 min)

```bash
cd nanoseek

# Single GPU sanity check (no DDP complexity)
python -m nanoseek.scripts.pre_train \
    --run smoke --scale ablation --seed 42 \
    --num-iterations 10 --eval-every -1 --save-every -1 \
    --device-batch-size 4

# Check output for:
# - No OOM errors
# - Loss is finite and decreasing
# - tok/s and mfu% reported
# - dt: X.XXs (step time)
```

### Step 2: Baseline MFU Measurement (10 min)

```bash
# Single GPU baseline (no communication overhead)
python -m nanoseek.scripts.pre_train \
    --run baseline-1gpu --scale ablation --seed 42 \
    --num-iterations 30 --eval-every -1 --save-every -1 \
    --device-batch-size 8

# Record from steps 15-30 (after warmup/compile):
#   - step_time (dt)
#   - mfu%
#   - tok/s
#   - memory_allocated_gb

# Multi-GPU baseline
torchrun --nproc_per_node=8 -m nanoseek.scripts.pre_train \
    --run baseline-8gpu --scale ablation --seed 42 \
    --num-iterations 30 --eval-every -1 --save-every -1 \
    --device-batch-size 8

# Record same metrics. Compare:
#   scaling_efficiency = (tok/s at 8GPU) / (8 * tok/s at 1GPU)
#   Expected: 85-95% with NVLink
#   Red flag: <80% means communication bottleneck
```

### Step 3: Check Which SDPA Backend Is Active (5 min)

**This is CRITICAL for MLA.** MLA has Q/K dim=192 and V dim=128.
FlashAttention-2 requires all head dims equal, so it may NOT dispatch.

```python
# Add this diagnostic to pre_train.py temporarily (or run standalone):
import torch
import torch.nn.functional as F

# Simulate MLA dimensions
B, H, S = 2, 16, 4096
q = torch.randn(B, H, S, 192, dtype=torch.bfloat16, device='cuda')  # QK dim
k = torch.randn(B, H, S, 192, dtype=torch.bfloat16, device='cuda')
v = torch.randn(B, H, S, 128, dtype=torch.bfloat16, device='cuda')  # V dim (different!)

# Test each backend individually:
backends = [
    ("flash",         dict(enable_flash=True, enable_math=False, enable_mem_efficient=False)),
    ("mem_efficient",  dict(enable_flash=False, enable_math=False, enable_mem_efficient=True)),
    ("math",          dict(enable_flash=False, enable_math=True, enable_mem_efficient=False)),
]

for name, flags in backends:
    try:
        with torch.backends.cuda.sdp_kernel(**flags):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        print(f"  {name}: WORKS (output shape: {out.shape})")
    except RuntimeError as e:
        print(f"  {name}: FAILED ({e})")

# If flash FAILS: MLA attention is falling back to O(S^2) math backend
# This is the #1 optimization to fix (see Tier 1, Item 1)
```

### Step 4: Nsight Systems Profile (15 min)

```bash
# Profile 3 training steps (skip warmup)
# Adjust --delay based on observed step time (15 steps * step_time)
nsys profile \
    --trace=cuda,nvtx,cublas \
    --cuda-memory-usage=true \
    --gpu-metrics-device=all \
    --stats=true \
    --force-overwrite=true \
    --output=nanoseek_baseline \
    python -m nanoseek.scripts.pre_train \
        --run profile --scale ablation --seed 42 \
        --num-iterations 25 --eval-every -1 --save-every -1 \
        --device-batch-size 4

# Get text summary (no GUI needed)
nsys stats nanoseek_baseline.nsys-rep

# Kernel time breakdown (which CUDA kernels dominate?)
nsys stats --report cuda_gpu_kern_sum nanoseek_baseline.nsys-rep

# Look for:
# 1. GEMM kernels (should be >50% of GPU time)
# 2. nccl* kernels (AllReduce — should be <15%)
# 3. Gaps between kernels (idle GPU = Python overhead)
# 4. Memory copy kernels (H2D transfers)
```

### Step 5: PyTorch Profiler Trace (10 min)

Add this temporarily to `pre_train.py` after line 1397 (inside the training loop):

```python
# === TEMPORARY: PyTorch profiler for step 20 ===
if step == 20 and master_process:
    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            outputs = model(x, labels=x, mtp_lambda=mtp_lambda)
        loss = outputs['loss'] / current_accum
        loss.backward()
    
    # Print top 20 ops by GPU time
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    prof.export_chrome_trace("nanoseek_step20.json")
    # Open in chrome://tracing or ui.perfetto.dev
# === END TEMPORARY ===
```

### Step 6: Memory Budget Check (5 min)

```python
# Add after model init in pre_train.py:
if master_process:
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"After model init: allocated={alloc:.2f}GB, reserved={reserved:.2f}GB")
    print(f"Headroom: {80.0 - reserved:.2f}GB (for activations + gradients)")
```

**Expected memory budget (ablation scale, single GPU):**
```
Model params (fp32 master):  1.95B * 4 bytes  =  7.8 GB
Optimizer states:            ~2x params        = ~15.6 GB
EMA (CPU):                   1.95B * 4 bytes   =  7.8 GB (on CPU, free)
Gradients (bf16):            1.95B * 2 bytes   =  3.9 GB
Activations:                 ~4-8 GB (with gradient checkpointing)
─────────────────────────────────────────────────────────────
Total estimate:              ~31-35 GB per GPU
H100 capacity:               80 GB
Headroom:                    ~45-49 GB (plenty for batch size increase)
```

### Decision Point: What Did the Profile Tell You?

After Steps 1-6, you know:
1. **Current MFU** — the number to beat
2. **Which SDPA backend** — if not Flash, that's the #1 fix
3. **Top 3 kernel bottlenecks** — where to focus optimization
4. **Communication overhead** — is AllReduce a problem?
5. **Memory headroom** — can you increase batch size?
6. **Data loading time** — is `dt_data` significant?

Now proceed to the optimization tiers below, **in order of measured impact**.

---

## 3. Tier 1: Zero-Risk Quick Wins (Day 1-2)

These changes have zero quality risk and can be implemented quickly.

### T1.1: Fix SDPA/FlashAttention Dispatch for MLA

**Problem:** MLA creates Q/K with head_dim=192 and V with head_dim=128. If FlashAttention
doesn't support mismatched dims, it falls back to the O(S^2) math backend. At S=4096
with 16 heads, this means materializing a 4096x4096 attention matrix per head.

**Diagnosis (from Step 3):** If flash backend FAILED, apply this fix.

**Fix — Pad V to match Q/K head dim:**

In `model.py`, MLA naive training path (line ~473):
```python
# BEFORE:
attn_output = F.scaled_dot_product_attention(
    q, k, v,  # q,k: [B,H,S,192], v: [B,H,S,128]
    attn_mask=..., scale=effective_scale, is_causal=needs_causal,
)

# AFTER: Pad V to 192 to enable FlashAttention dispatch
v_padded = F.pad(v, (0, q.shape[-1] - v.shape[-1]))  # [B,H,S,192]
attn_output = F.scaled_dot_product_attention(
    q, k, v_padded,
    attn_mask=..., scale=effective_scale, is_causal=needs_causal,
)
attn_output = attn_output[..., :self.v_head_dim]  # Slice back to 128
```

**Impact:** 2-4x faster attention, 10-20x less attention memory. ~15-25% total speedup.
**Risk:** Zero — mathematically equivalent (padding is zeros, sliced away after).
**Time:** 30 minutes.
**Verify:** Re-run the SDPA backend test from Step 3 — flash should now WORK.

**Note:** If flash-attn >= 2.7 is installed and supports heterogeneous head dims natively,
this padding is unnecessary. Check first, pad only if needed.

### T1.2: Eliminate GPU-CPU Sync Points in Hot Path

**Problem:** Multiple `.item()` calls in the MoE forward path and training loop create
GPU pipeline bubbles. Each `.item()` forces the GPU to drain its work queue.

**Fix A — MoE dispatch sync (model.py line 831):**
```python
# BEFORE:
max_count = expert_counts.max().item()  # GPU-CPU SYNC!
waste_ratio = max_count / max(avg_count, 1)
use_batched = sorted_x.is_cuda and E >= 8 and waste_ratio < 1.5

# AFTER: Use fixed capacity factor (no sync, deterministic shapes)
capacity = int(math.ceil((N * K / E) * 1.25))  # static at graph time
# Always use batched on GPU (remove conditional entirely)
if sorted_x.is_cuda and E >= 8:
    # Clamp expert counts to capacity to bound padding
    expert_counts_clamped = expert_counts.clamp(max=capacity)
    sorted_output = self._batched_expert_forward(
        sorted_x, sorted_indices, expert_counts_clamped, expert_boundaries,
    )
else:
    # CPU sequential fallback (testing only)
    ...
```

Also in `_batched_expert_forward` line 748:
```python
# BEFORE:
max_count = expert_counts.max().item()  # SECOND sync point!

# AFTER: Already clamped by caller, use capacity directly
max_count = expert_counts.max()  # stays on GPU
# Or pass capacity as an argument from the caller
```

**Fix B — Training loop sync (pre_train.py):**
```python
# Move synchronize() to log-only steps (line 1396):
# BEFORE: synchronize() called EVERY step
# AFTER:
if step % config.log_every_steps == 0:
    synchronize()
t0 = time.time()

# Line 1512-1513: defer .item() to log steps
# BEFORE:
train_loss_f = train_loss.item()  # SYNC every step!
synchronize()

# AFTER:
synchronize()
t1 = time.time()
dt = t1 - t0
if step % config.log_every_steps == 0 or not math.isfinite(train_loss.item()):
    train_loss_f = train_loss.item()
else:
    train_loss_f = float('nan')  # placeholder, only used in logging
```

**Impact:** 3-8% total speedup (eliminates 2-4 sync points per step, each ~50-200us).
**Risk:** Zero. The sync is only needed for accurate timing/logging.
**Time:** 1 hour.

### T1.3: Defer Per-Group Gradient Norms to Log Steps

**Problem:** `compute_per_group_grad_norms()` iterates all ~200+ parameters every step
(line 1466), but the results are only logged every `log_every_steps`.

```python
# BEFORE (pre_train.py line 1466):
per_group_gn = compute_per_group_grad_norms()  # every step

# AFTER: Only compute on log steps
if step % config.log_every_steps == 0:
    per_group_gn = compute_per_group_grad_norms()
else:
    per_group_gn = {}
```

**Impact:** 2-5% speedup (eliminates ~200 small CUDA kernels per non-log step).
**Risk:** Zero — the norms are diagnostic only, don't affect training.
**Time:** 5 minutes.

### T1.4: Pin Memory + Async Data Transfer

```python
# In dataloader.py, when creating batch tensors:
# BEFORE:
x = torch.tensor(batch_tokens, dtype=torch.long)

# AFTER:
x = torch.tensor(batch_tokens, dtype=torch.long).pin_memory()
# Then in the training loop, use non_blocking=True:
x = x.to(device, non_blocking=True)
```

**Impact:** 0-5% (depends on whether data loading is a bottleneck — check `dt_data` from logs).
**Risk:** Zero.
**Time:** 15 minutes.

### T1.5: RMSNorm Redundant Float Cast

```python
# model.py line 167-169:
# BEFORE:
variance = x.float().pow(2).mean(-1, keepdim=True)
rms = torch.rsqrt(variance + self.eps)
return (x.float() * rms).to(x.dtype) * self.weight

# AFTER: Single float() conversion
x_float = x.float()
rms = torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
return (x_float * rms).to(x.dtype) * self.weight
```

**Impact:** 2-3% (called 34+ times per forward pass; eliminates one redundant copy).
**Risk:** Zero.
**Time:** 5 minutes.

### T1.6: Gradient Compression Hook for DDP

```python
# In pre_train.py, after DDP or optimizer setup:
# Only if using DDP (manual allreduce in DistMuonAdamW may not use this):
if ddp and hasattr(model, 'register_comm_hook'):
    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
    model.register_comm_hook(state=None, hook=default_hooks.bf16_compress_hook)
```

**Note:** NanoSeek uses `DistMuonAdamW` with manual gradient reduction, not standard DDP.
This hook applies IF the code switches to DDP wrapping. If using manual allreduce,
apply bf16 compression to the allreduce call in `DistMuonAdamW` instead.

**Impact:** 3-5% if communication-bound (halves allreduce data volume).
**Risk:** Very low — gradients are already bf16 in practice.
**Time:** 10 minutes.

---

## 4. Tier 2: Medium-Effort High-Impact (Day 2-3)

### T2.1: Fix torch.compile Graph Breaks in MoE

**Problem:** The MoE forward has 3 sources of graph breaks that prevent torch.compile
from fusing kernels across the MoE layer:

1. `expert_counts.max().item()` — GPU-CPU sync (fixed in T1.2)
2. `waste_ratio < 1.5` conditional — data-dependent Python branch
3. `expert_counts.tolist()` in sequential fallback

After T1.2 (always use batched + fixed capacity), issues 2 and 3 are eliminated.
The remaining concern is `torch.bincount` which has dynamic output consideration.

**Verification:**
```python
# Set environment variable to see graph breaks:
# TORCH_LOGS="+dynamo" python -m nanoseek.scripts.pre_train --run dummy ...
# Or in code:
import torch._dynamo
torch._dynamo.config.log_level = logging.DEBUG
```

**Impact:** 10-20% throughput (kernel fusion across MoE + attention layers).
**Risk:** Low — torch.compile is well-tested on H100.
**Time:** 1 day (most time spent debugging remaining graph breaks).

### T2.2: Pre-Stack Expert Weights (Eliminate Per-Forward torch.stack)

**Problem:** `_batched_expert_forward` calls `torch.stack([e.w_gate.weight for e in ...])` 
every forward pass (model.py lines 765-768). With 64 experts and 3 weight matrices each,
this creates 3 large stacked tensors EVERY forward call, 15 times per step (15 MoE layers),
TWICE with gradient checkpointing = 90 stack operations per step.

For 1B scale: each stack is [64, 768, 2048] * 2 bytes = 192 MB. Total: 90 * 192 MB = 17 GB
of transient memory allocation per step (though the allocator recycles the same buffers).

**Fix — Cache stacked weights, update after optimizer step:**

```python
class MoE(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # ... existing expert init ...
        self._stacked_weights_dirty = True  # flag for lazy recompute
        self._cached_w_gate_up = None
        self._cached_w_down = None
    
    def _update_stacked_weights(self):
        """Re-stack after optimizer step. Called once, not per-forward."""
        with torch.no_grad():
            w_gate = torch.stack([e.w_gate.weight for e in self.routed_experts])
            w_up = torch.stack([e.w_up.weight for e in self.routed_experts])
            self._cached_w_gate_up = torch.cat([w_gate, w_up], dim=1)
            self._cached_w_down = torch.stack([e.w_down.weight for e in self.routed_experts])
        self._stacked_weights_dirty = False
    
    def _batched_expert_forward(self, sorted_x, sorted_indices, expert_counts, expert_boundaries):
        # Use cached weights (requires autograd through individual expert params)
        # For training: we still need torch.stack for autograd, but we can avoid
        # the redundant stack from gradient checkpointing recomputation
        if not self.training:
            if self._stacked_weights_dirty:
                self._update_stacked_weights()
            w_gate_up = self._cached_w_gate_up.to(sorted_x.dtype)
            w_down = self._cached_w_down.to(sorted_x.dtype)
        else:
            # Training: must go through autograd graph
            w_gate = torch.stack([e.w_gate.weight for e in self.routed_experts])
            w_up = torch.stack([e.w_up.weight for e in self.routed_experts])
            w_gate_up = torch.cat([w_gate, w_up], dim=1).to(sorted_x.dtype)
            w_down = torch.stack([e.w_down.weight for e in self.routed_experts]).to(sorted_x.dtype)
        # ... rest of batched forward ...
```

**Better fix — Fused expert parameters (requires more refactoring):**

Store expert weights as single [E, inter, D] parameter tensors instead of
nn.ModuleList of Expert modules. Each expert becomes a slice view.
This eliminates torch.stack entirely and enables native bmm without copying.

This is a larger refactor but the RIGHT long-term solution. Gradient flows through
the fused parameter directly — no autograd overhead from stack.

**Impact:** 5-15% MoE forward time, ~5-10% total throughput.
**Risk:** Medium — must ensure gradients flow correctly and optimizer param groups match.
**Time:** 1-2 days.

### T2.3: NVTX Markers for Production Profiling

Add NVTX annotations to identify bottlenecks in future profiling sessions:

```python
# model.py — add at top:
from torch.profiler import record_function

# In NanoSeekDecoderLayer.forward():
def forward(self, hidden_states, freqs_cis, attention_mask=None, kv_cache=None):
    with record_function(f"Layer_{self.layer_idx}"):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        with record_function("MLA"):
            hidden_states, kv = self.self_attn(hidden_states, freqs_cis, attention_mask, kv_cache)
        
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        if self.use_moe:
            with record_function("MoE"):
                hidden_states, aux_data = self.ffn(hidden_states)
        else:
            with record_function("DenseFFN"):
                hidden_states = self.ffn(hidden_states)
                aux_data = ...
        
        hidden_states = residual + hidden_states
    return hidden_states, kv, aux_data

# In MoE.forward() — annotate subphases:
with record_function("MoE::gate"):
    weights, indices, aux_loss, metadata = self.gate(x)

with record_function("MoE::dispatch"):
    # sorting, expanding, boundary computation
    ...

with record_function("MoE::expert_compute"):
    sorted_output = self._batched_expert_forward(...)

with record_function("MoE::combine"):
    # weighted sum + scatter back
    ...

with record_function("MoE::shared_expert"):
    shared_output = self.shared_expert(x)
```

`record_function` has near-zero overhead when no profiler is active (checks a global flag).
These markers make every future nsys/PyTorch profiler session immediately informative.

**Impact:** No direct speedup — enables targeted optimization.
**Risk:** Zero.
**Time:** 1 hour.

### T2.4: EMA Update Optimization

**Problem:** `EMATracker.step()` iterates all named parameters individually, doing
per-tensor CPU transfers. `apply()` does 4 full parameter traversals for eval.

**Fix for `step()`:**
```python
def step(self, model):
    """Batch EMA update: single pass, vectorized lerp."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in self.shadow:
                # Do the lerp on GPU (fast), then copy to CPU (one transfer)
                shadow_gpu = self.shadow[name].to(param.device)
                shadow_gpu.lerp_(param.detach(), 1 - self.decay)
                self.shadow[name] = shadow_gpu.to(self.device)  # back to CPU
```

**Fix for `apply()` — avoid double copy:**
```python
@contextmanager
def apply(self, model):
    """Apply EMA weights for evaluation. Minimize copies."""
    backup = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in self.shadow:
                backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].to(param.device))
    try:
        yield
    finally:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in backup:
                    param.data.copy_(backup[name])
```

Or better: use `torch._foreach_copy_` for batched parameter operations.

**Impact:** 20-40% faster EMA operations (called every `ema_every` steps + every eval).
**Risk:** Low.
**Time:** 2 hours.

---

## 5. Tier 3: Advanced Optimizations (Day 4+)

These require more effort and testing. Only pursue after Tier 1-2 are validated.

### T3.1: FP8 for Expert FFNs (After torch.compile Validated)

**Prerequisites:**
- torch.compile working cleanly (no graph breaks in MoE)
- CLAUDE.md rule: "Do NOT use FP8 until torch.compile is validated"

H100 FP8 tensor cores: 1979 TFLOPS vs 989 TFLOPS BF16 = 2x theoretical speedup.

**Strategy — FP8 for expert FFNs only (most FLOPs, least risk):**
```python
# Port nanochat's fp8.py pattern:
# Convert only expert weight matmuls to FP8, keep:
#   - Router in FP32 (routing precision matters for load balance)
#   - Attention in BF16 (MLA has small matmuls where FP8 overhead dominates)
#   - Norms in FP32 (numerical stability)
```

**Expected:** 1.5-2x throughput on expert FFN compute (90% of total FLOPs).
**Risk:** Medium. Monitor H_load and val_bpb closely for first 500 steps.
If H_load drops >0.5 from baseline, revert.
**Time:** 2-3 days.

### T3.2: Backward-Optimizer Overlap (Muon)

**Idea:** Start optimizer step for layer N while backward is still computing layer N-1.
The Newton-Schulz iterations in Muon can overlap with backward computation.

```python
# Register post-accumulate-grad hook on each parameter:
def create_optimizer_hook(param, optimizer, group):
    def hook(grad):
        with torch.no_grad():
            # Apply Muon or AdamW update immediately
            # This runs concurrently with backward of next layer
            ...
    return hook

for group in optimizer.param_groups:
    for param in group['params']:
        param.register_post_accumulate_grad_hook(
            create_optimizer_hook(param, optimizer, group)
        )
```

**Caveat:** Requires careful integration with gradient clipping (must clip BEFORE
optimizer step, but hooks fire per-parameter). May need to use per-parameter clipping
instead of global norm clipping, or delay the hook to fire after all grads are accumulated.

**Expected:** 5-8% speedup (hides Muon's Newton-Schulz cost behind backward compute).
**Risk:** Medium — interaction with gradient accumulation and clipping is tricky.
**Time:** 2-3 days.

### T3.3: Megablocks / ScatterMoE Integration

**Megablocks** (block-sparse MoE) eliminates all padding waste by using block-sparse
matrix multiplication. Tokens are dispatched to experts without padding — each expert
processes exactly its assigned tokens.

```bash
pip install megablocks
```

```python
# Replace NanoSeek's MoE class with Megablocks' dMoE:
from megablocks.layers.dmoe import dMoE
# Requires adapting expert weight shapes and routing logic
```

**Alternative: ScatterMoE** — lighter-weight, uses torch.scatter/gather for efficient
dispatch without padding. Easier to integrate with existing code.

**Expected:** 15-30% MoE layer speedup (eliminates padding waste + fewer kernel launches).
**Risk:** Medium — new dependency, potential torch.compile conflicts.
**Time:** 3-5 days.

### T3.4: torch.compile Mode Tuning

```python
# Default:
model = torch.compile(model, dynamic=False)

# Better for H100:
model = torch.compile(model, dynamic=False, mode="max-autotune")
# max-autotune: tries more kernel variants, picks fastest for H100 Tensor Cores
# First step takes longer to compile, but steady-state is faster

# If max-autotune causes issues:
model = torch.compile(model, dynamic=False, mode="reduce-overhead")
# reduce-overhead: uses CUDA graphs where possible, reduces kernel launch overhead
```

Test all three modes and compare steady-state MFU. Expected ranking:
`max-autotune >= reduce-overhead > default`.

**Impact:** 5-15% additional over default torch.compile.
**Risk:** Low (well-tested on H100).
**Time:** 30 minutes to test, 1 day if debugging needed.

### T3.5: Progressive Batch Size Scaling

Start with smaller batch, increase during training. This saves compute in early steps
(where learning is per-token-efficient regardless of batch size).

```python
# In pre_train.py, modify get_batch_warmup_accum or add batch schedule:
def get_batch_schedule(step, total_steps, target_batch):
    """Progressive batch size: 25% → 50% → 100% of target."""
    if step < total_steps * 0.05:
        return max(1, target_batch // 4)
    elif step < total_steps * 0.15:
        return max(1, target_batch // 2)
    else:
        return target_batch
```

NanoSeek already has `get_batch_warmup_accum` that does 1/5 warmup. This could be
made more aggressive (start at 1/4, spend more steps at low batch) following
Smith et al. (2018) "Don't Decay the Learning Rate, Increase the Batch Size".

**Expected:** 10-20% total compute savings for equivalent final loss.
**Risk:** Low — mathematically equivalent to LR decay.
**Time:** 2 hours.

---

## 6. Codebase Audit: Specific Issues Found

### Critical Performance Issues (by file)

#### model.py

| Line | Issue | Severity | Fix |
|------|-------|----------|-----|
| 748, 831 | `.item()` GPU-CPU sync in MoE hot path | HIGH | Fixed capacity factor (T1.2) |
| 765-768 | `torch.stack` 90x/step for expert weights | HIGH | Cache or fuse params (T2.2) |
| 834-838 | Data-dependent conditional → graph break | HIGH | Always batched on GPU (T1.2) |
| 167-169 | RMSNorm double `.float()` cast | MEDIUM | Single conversion (T1.5) |
| 144 | RoPE creates unnecessary intermediates | LOW | torch.compile handles this |
| 48-49 | CastLinear dtype cast every call | LOW | Intentional, compile fuses |

#### pre_train.py

| Line | Issue | Severity | Fix |
|------|-------|----------|-----|
| 1396, 1513 | `synchronize()` every step | MEDIUM | Log-step only (T1.2) |
| 1466 | Per-group grad norms every step | MEDIUM | Log-step only (T1.3) |
| 1512 | `train_loss.item()` every step | MEDIUM | Defer to log steps (T1.2) |
| 1103 | Whole-model compile (graph breaks) | HIGH | Fix graph breaks first (T2.1) |
| 790-797 | EMA individual param copies | MEDIUM | Batched update (T2.4) |

#### optim.py

| Line | Issue | Severity | Fix |
|------|-------|----------|-----|
| 274-275 | `torch.stack(params)` every step | MEDIUM | Pre-allocate buffer (T2.2) |

#### dataloader.py

| Line | Issue | Severity | Fix |
|------|-------|----------|-----|
| 249 | No pin_memory for H2D | LOW | Add pin_memory (T1.4) |
| 193-198 | O(N) best-fit scan per document | LOW | Only if dt_data is bottleneck |

---

## 7. Expected Results & MFU Targets

### Before vs After Optimization

| Scenario | Expected MFU | tok/s (8xH100, ablation) | Step Time |
|----------|-------------|--------------------------|-----------|
| **Baseline (no optimization)** | 20-30% | ~50-80K | ~1.5-2.5s |
| **After Tier 1 (quick wins)** | 30-40% | ~80-120K | ~1.0-1.5s |
| **After Tier 2 (compile + cache)** | 35-45% | ~100-150K | ~0.8-1.2s |
| **After Tier 3 (FP8 + overlap)** | 40-55% | ~130-200K | ~0.5-0.8s |

### MFU Reference Points

| System | Architecture | Hardware | MFU | Source |
|--------|-------------|----------|-----|--------|
| nanochat GPT-2 | Dense 124M | 1xH100 | 55-65% | Karpathy benchmarks |
| NanoSeek target | MoE 4.75B | 8xH100 | **35-45%** | This plan |
| DeepSeek V3 | MoE 671B | 2048xH800 | ~60% (active) | DeepSeek paper |
| World-class MoE | MoE general | H100 | 40-50% | Megablocks benchmarks |

**35-45% MFU is realistic and good for a minimal MoE codebase on 8 GPUs.**
Above 45% requires FP8 + Megablocks + custom kernels.
Below 30% means something is broken — revisit the profile.

### Time Savings at Scale

| Training Run | Tokens | Baseline Time | After Tier 1+2 | After All |
|-------------|--------|--------------|----------------|-----------|
| HP search (6 runs) | 1.5B | ~3 hrs | ~2 hrs | ~1.5 hrs |
| Full ablation | 8.2B | ~12 hrs | ~8 hrs | ~5 hrs |
| Full 1B | 22B | ~36 hrs | ~24 hrs | ~15 hrs |

---

## 8. Cost-Optimal Rental Strategy

### Provider Comparison (H100 SXM 80GB, 8 GPU node)

| Provider | $/GPU/hr | 8-GPU $/hr | Preemption Risk | Best For |
|----------|----------|------------|-----------------|----------|
| Lambda Labs | $2.49 | $19.92 | None (reserved) | Reliability, no babysitting |
| RunPod | $2.19-3.29 | $17.52-26.32 | Low (community) | Good balance |
| Vast.ai | $1.80-2.50 | $14.40-20.00 | Low-Medium | Budget runs |
| GCP Spot (A3) | ~$1.08 | ~$8.64 | HIGH (1-6 hrs) | Cost-optimal with checkpointing |
| AWS Spot (p5) | ~$1.24 | ~$9.92 | HIGH | Same as GCP |

### Recommended Strategy

**For HP search + ablation (~20 hours):**
- RunPod or Vast.ai: $350-400 total, low hassle
- GCP spot: ~$170 total, but need aggressive checkpointing (every 10 min)

**For full 1B training (~24-36 hours):**
- Lambda: $480-720 total, zero risk
- RunPod: $420-630 total, minimal risk
- GCP spot: $210-310 total, needs checkpoint strategy

### Checkpointing for Spot Instances

If using spot/preemptible instances:
```bash
# Save every 10 minutes wall-clock (not every N steps)
# NanoSeek checkpoint size: ~20GB (model + optimizer + EMA)
# NVMe write speed: ~3 GB/s → 7s per checkpoint
# Max lost compute on preemption: 10 min = $1.44 at GCP spot rates

# Add to pre_train.py or launch script:
--save-every 200  # ~200 steps ≈ 10 min at 3s/step
```

### Total Budget Estimate

| Phase | GPU-Hours | Lambda ($2.49) | RunPod ($2.19) | GCP Spot ($1.08) |
|-------|-----------|----------------|----------------|------------------|
| Profiling + optimization | 8 | $20 | $18 | $9 |
| HP search (6 runs) | 16 | $40 | $35 | $17 |
| Full ablation | 64-96 | $160-240 | $140-210 | $69-104 |
| Full 1B | 192-288 | $478-717 | $421-631 | $207-311 |
| **Total** | **280-408** | **$698-1,017** | **$614-894** | **$302-441** |

With performance optimizations (1.5-2x speedup), actual GPU-hours drop proportionally:
- **Optimized total: ~150-250 GPU-hours = $375-625 (Lambda) or $160-270 (GCP spot)**

---

## Appendix A: Microbenchmark Script

Save as `nanoseek/scripts/benchmark.py`:

```python
#!/usr/bin/env python3
"""NanoSeek microbenchmarks — time individual components on a single GPU."""

import torch
import time
import argparse
from nanoseek.nanoseek.config import get_config
from nanoseek.nanoseek.model import NanoSeekModel


def bench(fn, warmup=5, repeat=50, label=""):
    """Benchmark a callable with warmup and median timing."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    
    times.sort()
    med = times[len(times) // 2]
    p10 = times[int(len(times) * 0.1)]
    p90 = times[int(len(times) * 0.9)]
    print(f"  {label:45s}  median={med:8.3f}ms  p10={p10:.3f}  p90={p90:.3f}ms")
    return med


def run(scale="ablation", bs=4, seq=4096):
    config = get_config(scale)
    device = torch.device("cuda")
    
    with torch.device("meta"):
        model = NanoSeekModel(config)
    model.to_empty(device=device)
    model.init_weights()
    model.eval()
    
    D = config.hidden_size
    x = torch.randn(bs, seq, D, dtype=torch.bfloat16, device=device)
    ids = torch.randint(0, config.vocab_size, (bs, seq), device=device)
    freqs = model.freqs_cis[:seq].to(device)
    
    print(f"\n=== {scale} | B={bs} S={seq} D={D} ===\n")
    
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        # Individual layers
        layer0 = model.layers[0]
        bench(lambda: layer0.self_attn(layer0.input_layernorm(x), freqs),
              label="MLA attention (1 layer)")
        
        if hasattr(model.layers[2], 'ffn'):
            moe_layer = model.layers[2]
            bench(lambda: moe_layer.ffn(moe_layer.post_attention_layernorm(x)),
                  label="MoE FFN (1 layer, 64e top-8)")
        
        # Full forward
        bench(lambda: model(ids, labels=ids, mtp_lambda=0.1),
              label="Full forward + loss")
    
    # Memory
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"\n  Peak GPU memory: {peak:.2f} GB / 80 GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", default="ablation")
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--seq", type=int, default=4096)
    args = parser.parse_args()
    run(args.scale, args.bs, args.seq)
```

---

## Appendix B: Multi-GPU Scaling Test

Save as `nanoseek/runs/scaling_test.sh`:

```bash
#!/bin/bash
# Test DDP scaling efficiency: 1, 2, 4, 8 GPUs
set -euo pipefail

SCALE="${1:-ablation}"
DBS="${2:-4}"
STEPS=30

echo "=== NanoSeek Scaling Test: $SCALE, DBS=$DBS ==="
echo ""

for NGPU in 1 2 4 8; do
    echo "--- $NGPU GPU(s) ---"
    if [ $NGPU -eq 1 ]; then
        python -m nanoseek.scripts.pre_train \
            --run "scale-${NGPU}gpu" --scale "$SCALE" --seed 42 \
            --num-iterations $STEPS --eval-every -1 --save-every -1 \
            --device-batch-size "$DBS" 2>&1 | grep -E "step 0(2[0-9]|30)" | tail -5
    else
        torchrun --nproc_per_node=$NGPU -m nanoseek.scripts.pre_train \
            --run "scale-${NGPU}gpu" --scale "$SCALE" --seed 42 \
            --num-iterations $STEPS --eval-every -1 --save-every -1 \
            --device-batch-size "$DBS" 2>&1 | grep -E "step 0(2[0-9]|30)" | tail -5
    fi
    echo ""
done
```

---

## Appendix C: Optimization Checklist

Print this and check off as you go:

```
DAY 1: Profile
[ ] nvidia-smi + topo check
[ ] Smoke test (10 steps, single GPU)
[ ] Baseline MFU (30 steps, 1 GPU + 8 GPU)
[ ] SDPA backend check (is FlashAttention active?)
[ ] nsys profile (text summary)
[ ] PyTorch profiler trace (step 20)
[ ] Memory budget check
[ ] Record baseline numbers: MFU=___%, tok/s=_____, step_time=____s

DAY 1-2: Tier 1 Quick Wins
[ ] T1.1: Fix FlashAttention for MLA (if needed)
[ ] T1.2: Eliminate .item() GPU-CPU syncs
[ ] T1.3: Defer grad norms to log steps
[ ] T1.4: Pin memory + async transfer
[ ] T1.5: RMSNorm single float cast
[ ] Measure again: MFU=___%, tok/s=_____, step_time=____s

DAY 2-3: Tier 2
[ ] T2.1: Fix torch.compile graph breaks
[ ] T2.2: Cache expert weight stacks
[ ] T2.3: Add NVTX markers
[ ] T2.4: EMA update optimization
[ ] T3.4: Test compile mode="max-autotune"
[ ] Measure again: MFU=___%, tok/s=_____, step_time=____s

DAY 4+: Tier 3 (only if needed)
[ ] T3.1: FP8 for expert FFNs
[ ] T3.2: Backward-optimizer overlap
[ ] T3.3: Megablocks integration
[ ] T3.5: Progressive batch size
[ ] Final measurement: MFU=___%, tok/s=_____, step_time=____s
```

---

## Key References

- **DeepSeek V3 Technical Report** — FP8, DualPipe, sigmoid routing, MFU ~60%
- **FlashAttention-2/3** (Dao et al.) — Fused attention kernels for H100
- **Megablocks** (Gale et al. 2023) — Block-sparse MoE, eliminates padding waste
- **nanochat/modded-nanogpt** — Reference MFU benchmarks for dense H100 training
- **Smith et al. 2018** — Progressive batch size scaling = implicit LR decay
- **PyTorch torch.compile docs** — Graph breaks, Inductor backend, CUDA graphs
