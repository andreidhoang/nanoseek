# Tier 1 Benchmarking & Profiling Guide
## RTX 4090 (24GB) — Weeks 0-4: Kernel Development & Validation

**Premise**: Every optimization must be measured, not guessed. A senior performance engineer at a frontier lab never writes a kernel without profiling the baseline first, never ships a kernel without A/B benchmarking, and never claims a speedup without roofline justification.

**Hardware baseline** (RTX 4090):
```
Compute:   82.6 TFLOPS (BF16 Tensor Core)
           165.2 TFLOPS (FP16 Tensor Core with sparsity)
           330.3 TFLOPS (INT8 Tensor Core)
Bandwidth: 1,008 GB/s (GDDR6X)
VRAM:      24 GB
SMs:       128
L2 Cache:  72 MB
SRAM/SM:   228 KB (shared memory + L1)
Arch:      Ada Lovelace (SM 8.9)
No FP8 tensor cores (limited FP8 via software emulation — not useful)
```

**Ridge point**: 82.6 TFLOPS / 1.008 TB/s = **81.9 FLOP/Byte**
- Operations with arithmetic intensity < 82 → memory-bound
- Operations with arithmetic intensity > 82 → compute-bound

---

## The Profiling Hierarchy: Top-Down, Always

**Rule**: Profile top-down. Stop when you find the bottleneck. Never start at the kernel level — you might optimize the wrong kernel.

```
Level 0:  Full training step (forward + backward + optimizer)
          → "How fast is my training?"
          → Answers: tokens/sec, step_time, MFU
          → Time: 5 min to profile

Level 1:  Forward vs Backward vs Optimizer (3 segments)
          → "Which phase dominates?"
          → Expected: backward ≈ 2-3x forward, optimizer ≈ 5-15%
          → Time: 5 min

Level 2:  Per-layer breakdown (16 layers)
          → "Are all layers equal, or is one a bottleneck?"
          → MoE layers are NOT equal — routing patterns differ across layers
          → Time: 15 min

Level 3:  Per-component within a layer
          → "Within one layer, what's the bottleneck?"
          → THIS is where you find: expert compute = 50%, MLA = 20%
          → Time: 30 min

Level 4:  Individual kernel (Nsight Compute)
          → "Is this specific GEMM compute-bound or memory-bound?"
          → Roofline analysis, bandwidth measurement, bank conflicts
          → Time: 1-2 hours
```

### Why Per-Layer Profiling Matters for MoE (Not Dense)

```
Dense model: every layer is ~identical (same ops, same shapes)
  → Profiling 1 layer tells you about all 16

MoE model: layers can DIFFER because:
  1. Routing patterns evolve across layers
     (early layers: broad routing, late layers: specialized)
  2. Load imbalance varies per layer
     (some layers have dead experts, others have balanced routing)
  3. Padding waste differs per layer
     (waste_ratio could be 1.1 in layer 0, 1.8 in layer 15)

So for MoE: you MUST profile multiple layers, not just one.
```

**Per-layer profiling reveals hidden bottlenecks:**

```
┌─────────┬──────────┬────────────┬────────────┬──────────────┐
│ Layer   │ Total ms │ MoE ms (%) │ MLA ms (%) │ waste_ratio  │
├─────────┼──────────┼────────────┼────────────┼──────────────┤
│ Layer 0 │ 2.31     │ 1.18 (51%) │ 0.42 (18%) │ 1.45         │
│ Layer 1 │ 2.28     │ 1.15 (50%) │ 0.41 (18%) │ 1.38         │
│ ...     │ ...      │ ...        │ ...        │ ...          │
│ Layer 8 │ 2.42     │ 1.28 (53%) │ 0.43 (18%) │ 1.62 ← worst│
│ ...     │ ...      │ ...        │ ...        │ ...          │
│ Layer 15│ 2.19     │ 1.08 (49%) │ 0.40 (18%) │ 1.22         │
└─────────┴──────────┴────────────┴────────────┴──────────────┘

Insight: Layer 8 is 10% slower than average because waste_ratio=1.62.
         The gate router sends 62% more tokens to its most popular expert
         than the average expert gets. Grouped GEMM eliminates this waste.
```

### How to Profile Each Level

**Level 0 — Full Step** (CUDA events):
```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for step in range(warmup + measured):
    start.record()
    loss = model(x, targets=y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    end.record()
    torch.cuda.synchronize()
    if step >= warmup:
        step_times.append(start.elapsed_time(end))
```

**Level 1 — Three-Phase Split**:
```python
fwd_start.record()
loss = model(x, targets=y)
fwd_end.record()

bwd_start.record()
loss.backward()
bwd_end.record()

opt_start.record()
optimizer.step()
opt_end.record()

torch.cuda.synchronize()
# Result: forward=15ms, backward=38ms, optimizer=4ms
# Backward dominates (67%) → optimize forward FIRST (backward scales with it)
```

**Level 2 — Per-Layer Hooks**:
```python
layer_times = {}
for i, layer in enumerate(model.layers):
    def make_hooks(idx):
        def pre_hook(module, input):
            module._start = torch.cuda.Event(enable_timing=True)
            module._start.record()
        def post_hook(module, input, output):
            module._end = torch.cuda.Event(enable_timing=True)
            module._end.record()
        return pre_hook, post_hook
    pre, post = make_hooks(i)
    layer.register_forward_pre_hook(pre)
    layer.register_forward_hook(post)

# After forward + synchronize:
for i, layer in enumerate(model.layers):
    layer_times[i] = layer._start.elapsed_time(layer._end)
```

**Level 3 — Per-Component Within a Layer** (see Stage 0.2 below for full implementation).

**Level 4 — Individual Kernel via NVTX + Nsight Compute**:
```python
import torch.cuda.nvtx as nvtx

# Add NVTX markers around specific operations
def expert_forward_with_markers(self, padded_input):
    nvtx.range_push("expert_bmm_gate_up")
    gate = torch.bmm(padded_input, stacked_gate_weights)
    up = torch.bmm(padded_input, stacked_up_weights)
    nvtx.range_pop()

    nvtx.range_push("expert_swiglu")
    h = F.silu(gate) * up
    nvtx.range_pop()

    nvtx.range_push("expert_bmm_down")
    out = torch.bmm(h, stacked_down_weights)
    nvtx.range_pop()
    return out
```
```bash
# Profile ONLY the marked kernel (not the entire model):
ncu --nvtx --nvtx-include "expert_bmm_gate_up" \
    --set full -o expert_gate_profile \
    python -m nanoseek.benchmarks.profile_kernel
```

---

## Stage 0: Baseline Profiling (Days 1-3)

### Objective
Answer ONE question: **Where does NanoSeek spend its time?**

Before writing a single line of kernel code, you must know the exact breakdown. This gates the entire project — if MoE dispatch is <30% of step time, the NanoFuse project pivots.

### 0.1 Training Step Profiling

**What to measure**: Wall-clock time per component of a single training step.

```python
# File: nanoseek/benchmarks/profile_baseline.py

import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

def profile_training(model, dataloader, num_steps=30):
    """
    Profile NanoSeek training with proper warmup.

    Critical rules:
    1. skip_first=5: ignore CUDA context init + JIT compilation
    2. warmup=5: let frequency scaling stabilize
    3. active=10: collect 10 steady-state steps
    4. record_shapes=True: needed for FLOP estimation
    5. with_flops=True: auto-estimate matmul/conv FLOPs
    6. profile_memory=True: track peak allocation
    7. with_stack=False: avoid overhead (add only for targeted investigation)
    """
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(skip_first=5, wait=2, warmup=3, active=10, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        with_stack=False,  # <-- False for baseline, True for targeted debugging
        on_trace_ready=tensorboard_trace_handler('./benchmarks/traces/baseline'),
    ) as prof:
        for step in range(num_steps):
            loss = train_step(model, dataloader)
            prof.step()

    # Export Chrome trace for Perfetto visualization
    prof.export_chrome_trace('./benchmarks/traces/baseline_chrome.json')

    # Print top-20 CUDA kernels by self time
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=20,
        header="TOP 20 CUDA KERNELS BY SELF TIME"
    ))

    # Print top-20 by CUDA memory
    print(prof.key_averages().table(
        sort_by="self_cuda_memory_usage",
        row_limit=20,
        header="TOP 20 BY MEMORY ALLOCATION"
    ))

    return prof
```

**Run command** (ablation scale, batch_size=2 for 24GB):
```bash
cd /workspace/nanoseek
python -m nanoseek.benchmarks.profile_baseline \
    --scale ablation --device-batch-size 2 --num-steps 30
```

### 0.2 Component-Level Timing

**What to measure**: Exact milliseconds for each model component per step.

```python
# File: nanoseek/benchmarks/component_timing.py

import torch

class ComponentTimer:
    """
    Measures per-component time using CUDA events.

    Why CUDA events instead of time.time():
    - CUDA ops are ASYNCHRONOUS. time.time() measures kernel LAUNCH, not EXECUTION.
    - CUDA events record timestamps on the GPU timeline.
    - start.elapsed_time(end) gives wall-clock GPU execution time in ms.

    Why torch.cuda.synchronize() at the end:
    - Forces all pending CUDA ops to complete before reading times.
    - Without this, event.elapsed_time() could return 0 or garbage.
    """

    def __init__(self):
        self.events = {}
        self.times = {}

    def mark(self, name: str, position: str = 'start'):
        """Record a CUDA event at this point in the execution."""
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.events[f"{name}_{position}"] = event

    def compute_times(self):
        """After synchronize(), compute elapsed times between event pairs."""
        torch.cuda.synchronize()
        for key in self.events:
            if key.endswith('_start'):
                name = key[:-6]
                end_key = f"{name}_end"
                if end_key in self.events:
                    self.times[name] = self.events[key].elapsed_time(self.events[end_key])
        return self.times


def time_nanoseek_components(model, batch, num_runs=20, warmup=5):
    """
    Time each component of a NanoSeek forward+backward pass.

    Components to isolate:
    1. Embedding lookup
    2. Per-layer:
       a. RMSNorm (pre-attention)
       b. MLA attention (Q/K/V projection + SDPA + output projection)
       c. RMSNorm (pre-MoE)
       d. Gate routing (sigmoid + topk + bias)
       e. Expert dispatch (sort + pad/scatter)
       f. Expert compute (SwiGLU FFN × 64 experts)
       g. Expert combine (weighted sum + scatter-add)
       h. Shared expert compute
    3. Final norm + LM head
    4. Loss computation
    5. Backward pass (total)
    6. Optimizer step

    Target output format:
    ┌──────────────────────────┬────────┬───────┐
    │ Component                │  ms    │   %   │
    ├──────────────────────────┼────────┼───────┤
    │ Embedding                │  0.3   │  0.2  │
    │ MLA Attention (all lyr)  │ 12.4   │ 18.2  │
    │ Gate Routing (all lyr)   │  1.2   │  1.8  │
    │ Expert Dispatch          │  3.1   │  4.5  │
    │ Expert Compute (64 exp)  │ 28.7   │ 42.1  │  ← THE TARGET
    │ Expert Combine           │  2.8   │  4.1  │
    │ Shared Expert            │  4.2   │  6.2  │
    │ RMSNorm (all)            │  1.5   │  2.2  │
    │ MTP Head                 │  3.8   │  5.6  │
    │ LM Head + Loss           │  2.1   │  3.1  │
    │ Other                    │  8.2   │ 12.0  │
    ├──────────────────────────┼────────┼───────┤
    │ TOTAL FORWARD            │ 68.3   │       │
    │ TOTAL BACKWARD           │ 142.1  │       │
    │ TOTAL STEP               │ 210.4  │       │
    └──────────────────────────┴────────┴───────┘
    """
    results = []

    for run in range(warmup + num_runs):
        timer = ComponentTimer()
        # ... instrument model forward pass with timer.mark() calls ...
        times = timer.compute_times()
        if run >= warmup:
            results.append(times)

    # Average across runs
    avg = {k: sum(r[k] for r in results) / num_runs for k in results[0]}
    total = sum(avg.values())
    for k, v in sorted(avg.items(), key=lambda x: -x[1]):
        print(f"  {k:30s}  {v:8.2f} ms  ({100*v/total:5.1f}%)")
    return avg
```

### 0.2b Per-Layer Variation Analysis (MoE-Specific)

**What to measure**: Do all 16 layers have the same performance profile, or do some layers bottleneck?

```python
# File: nanoseek/benchmarks/per_layer_profile.py

def profile_per_layer_variation(model, batch, num_runs=30, warmup=5):
    """
    Profile every layer independently to detect per-layer bottlenecks.

    Why this matters for MoE but NOT for dense models:
    ─────────────────────────────────────────────────
    Dense: Layer 0 ≈ Layer 1 ≈ ... ≈ Layer 15 (identical structure, same shapes)
           Profiling 1 layer = profiling all layers.

    MoE:   Layer 0 ≠ Layer 8 ≠ Layer 15 because:
           - Routing decisions differ (early=broad, late=specialized)
           - Expert load balance varies (waste_ratio: 1.1 to 1.8)
           - Dead expert count may differ across layers
           - Padding waste directly impacts batched GEMM time

    What to capture per layer:
      1. Total forward time (ms)
      2. MoE sub-time: gate + dispatch + compute + combine + shared
      3. MLA sub-time: projections + attention + output
      4. waste_ratio = max(expert_counts) / mean(expert_counts)
      5. Gini coefficient of expert token counts
      6. Number of dead experts (0 tokens assigned)

    Output format:
    ┌────────┬────────┬──────────┬──────────┬─────────────┬───────┬──────┐
    │ Layer  │ Tot ms │ MoE ms   │ MLA ms   │ waste_ratio │ Gini  │ Dead │
    ├────────┼────────┼──────────┼──────────┼─────────────┼───────┼──────┤
    │ 0      │ 2.31   │ 1.18     │ 0.42     │ 1.45        │ 0.08  │ 0    │
    │ 1      │ 2.28   │ 1.15     │ 0.41     │ 1.38        │ 0.07  │ 0    │
    │ ...    │ ...    │ ...      │ ...      │ ...         │ ...   │ ...  │
    │ 15     │ 2.19   │ 1.08     │ 0.40     │ 1.22        │ 0.05  │ 0    │
    ├────────┼────────┼──────────┼──────────┼─────────────┼───────┼──────┤
    │ StdDev │ 0.07   │ 0.06     │ 0.01     │ 0.13        │ 0.01  │ 0    │
    │ Max/Min│ 1.10x  │ 1.19x    │ 1.08x    │             │       │      │
    └────────┴────────┴──────────┴──────────┴─────────────┴───────┴──────┘

    Decision criteria:
    - If Max/Min ratio > 1.3x for MoE time: per-layer routing is unbalanced
      → Consider per-layer bias tuning or layer-specific optimization
    - If any layer has dead experts > 2: routing collapse in that layer
      → Check gate bias update (gamma parameter)
    - If StdDev of waste_ratio > 0.2: grouped GEMM gains vary by layer
      → NanoFuse benefit is layer-dependent, report per-layer speedup
    """
    layer_data = {i: [] for i in range(len(model.layers))}
    routing_data = {i: [] for i in range(len(model.layers))}

    # Install hooks on every layer
    hooks = []
    for i, layer in enumerate(model.layers):
        def make_timing_hooks(idx):
            def pre_hook(module, input):
                module._prof_start = torch.cuda.Event(enable_timing=True)
                module._prof_start.record()
            def post_hook(module, input, output):
                module._prof_end = torch.cuda.Event(enable_timing=True)
                module._prof_end.record()
            return pre_hook, post_hook
        pre, post = make_timing_hooks(i)
        hooks.append(layer.register_forward_pre_hook(pre))
        hooks.append(layer.register_forward_hook(post))

    # Install routing hooks on every gate
    for i, layer in enumerate(model.layers):
        def make_routing_hook(idx):
            def hook(module, input, output):
                weights, indices = output[0], output[1]
                flat = indices.view(-1)
                counts = torch.bincount(flat, minlength=64).float()
                routing_data[idx].append({
                    'waste_ratio': (counts.max() / counts.mean()).item(),
                    'gini': _compute_gini(counts).item(),
                    'dead_experts': (counts == 0).sum().item(),
                    'counts': counts.detach().cpu(),
                })
            return hook
        hooks.append(layer.moe.gate.register_forward_hook(make_routing_hook(i)))

    # Run forward passes
    for run in range(warmup + num_runs):
        with torch.no_grad():
            _ = model(batch['input_ids'])
        torch.cuda.synchronize()

        if run >= warmup:
            for i, layer in enumerate(model.layers):
                t = layer._prof_start.elapsed_time(layer._prof_end)
                layer_data[i].append(t)

    # Cleanup hooks
    for h in hooks:
        h.remove()

    # Report
    print(f"{'Layer':>6} {'Median ms':>10} {'MoE Waste':>10} {'Gini':>6} {'Dead':>5}")
    for i in range(len(model.layers)):
        median_t = sorted(layer_data[i])[len(layer_data[i])//2]
        avg_waste = sum(d['waste_ratio'] for d in routing_data[i]) / len(routing_data[i])
        avg_gini = sum(d['gini'] for d in routing_data[i]) / len(routing_data[i])
        max_dead = max(d['dead_experts'] for d in routing_data[i])
        print(f"{i:6d} {median_t:10.3f} {avg_waste:10.3f} {avg_gini:6.3f} {max_dead:5d}")

    return layer_data, routing_data


def _compute_gini(counts):
    """Gini coefficient: 0=perfectly balanced, 1=all tokens to one expert."""
    sorted_counts = counts.sort().values
    n = len(sorted_counts)
    cumsum = sorted_counts.cumsum(0)
    return (2 * (torch.arange(1, n+1, device=counts.device) * sorted_counts).sum()
            / (n * sorted_counts.sum()) - (n + 1) / n)
```

### 0.3 Memory Profiling

**What to measure**: Where does 24GB go? Can we fit ablation training?

```python
# File: nanoseek/benchmarks/memory_profile.py

def profile_memory(model, batch):
    """
    Memory budget breakdown for NanoSeek ablation on RTX 4090.

    Expected breakdown (ablation scale, BF16, batch_size=2, seq_len=4096):

    Parameters:
      Total params (ablation): ~1.95B
      BF16 storage: 1.95B × 2 bytes = ~3.9 GB
      FP32 master weights (CastLinear): 1.95B × 4 bytes = ~7.8 GB
      NOTE: CastLinear stores FP32, casts to BF16 in forward()

    Optimizer state (MuonAdamW):
      AdamW params (embeddings, scalars): ~67M × (4+4+4) = ~0.8 GB
      Muon params (2D weights): ~1.88B × (4+4+4) = ~22.5 GB
      PROBLEM: This alone exceeds 24GB

    Activations per layer (batch=2, seq=4096, hidden=1280):
      Input: 2 × 4096 × 1280 × 2 = 20 MB
      MLA intermediates: ~60 MB
      MoE intermediates: ~200 MB (64 experts × small batches)
      Per layer: ~280 MB × 16 layers = ~4.5 GB

    TOTAL ESTIMATE: 7.8 + 22.5 + 4.5 = ~34.8 GB
    RTX 4090: 24 GB → DOES NOT FIT without optimization

    Solutions:
    1. gradient_checkpointing: saves ~3.5 GB activations
    2. Reduce batch_size to 1: saves ~2.25 GB activations
    3. CPU offload optimizer states: saves ~22.5 GB
    4. Use BF16 optimizer states (Muon): saves ~7.5 GB
    5. Mixed precision (some FP32, mostly BF16): check what config does

    Action: Profile actual usage, then decide strategy.
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # 1. Model parameters only
    model.cuda()
    param_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"Parameters only: {param_mem:.2f} GB")

    # 2. Forward pass
    torch.cuda.reset_peak_memory_stats()
    output = model(batch['input_ids'].cuda())
    fwd_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"After forward: {fwd_mem:.2f} GB (+{fwd_mem - param_mem:.2f} GB activations)")

    # 3. Backward pass
    torch.cuda.reset_peak_memory_stats()
    output.loss.backward()
    bwd_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"After backward: {bwd_mem:.2f} GB")

    # 4. Optimizer step
    # ... create optimizer, step, measure peak ...

    # 5. Snapshot for detailed visualization
    torch.cuda.memory._record_memory_history(max_entries=100000)
    output = model(batch['input_ids'].cuda())
    output.loss.backward()
    torch.cuda.memory._dump_snapshot("benchmarks/memory_snapshot.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)
    print("Memory snapshot saved. View at pytorch.org/memory_viz")
```

### 0.4 MFU Calculation

**What to measure**: How efficiently are we using the RTX 4090?

```python
# File: nanoseek/benchmarks/mfu.py

def calculate_mfu(model_config, step_time_ms, batch_size, seq_len, device='rtx4090'):
    """
    Calculate Model FLOPs Utilization for NanoSeek.

    MFU = (model_flops_per_step / step_time_seconds) / hardware_peak_tflops

    For MoE, we count ACTIVE parameter FLOPs only (not total):
    - Dense components: embedding, attention, norms, LM head
    - MoE: only top-k experts per token (8 out of 64) + shared experts
    - This matches DeepSeek V3's reporting methodology

    RTX 4090 BF16 Tensor Core peak: 82.6 TFLOPS
    """
    HARDWARE_PEAKS = {
        'rtx4090': 82.6e12,   # BF16 Tensor Core
        'rtx3090': 35.6e12,   # BF16 Tensor Core
        'a6000':   38.7e12,   # BF16 Tensor Core (77.4 with sparsity)
        'h100':    989.4e12,  # BF16 Tensor Core
    }

    # FLOPs per token (forward pass only)
    # Factor of 6 = 2 (fwd matmul) + 4 (bwd: 2 grad_input + 2 grad_weight)
    #
    # Active parameters for MoE:
    #   N_active = dense_params + (K/E * routed_expert_params) + shared_expert_params
    #
    # Ablation: N_active ≈ 410M
    # 1B:       N_active ≈ 1.08B
    N_active = model_config.n_active_params

    flops_per_token = 6 * N_active  # Standard approximation

    # Attention FLOPs (not in parameter count):
    # 2 * n_layers * seq_len * n_heads * head_dim * 2 (QK^T + Attn×V)
    # For MLA: head_dim = qk_nope + qk_rope = 128 + 64 = 192
    attn_flops_per_token = (
        2 * model_config.num_layers * seq_len *
        model_config.num_heads * (model_config.qk_nope_head_dim + model_config.qk_rope_head_dim) * 2
    )

    total_flops_per_step = (flops_per_token + attn_flops_per_token) * batch_size * seq_len
    step_time_sec = step_time_ms / 1000.0
    achieved_tflops = total_flops_per_step / step_time_sec / 1e12

    peak = HARDWARE_PEAKS[device]
    mfu = total_flops_per_step / step_time_sec / peak

    tokens_per_sec = batch_size * seq_len / step_time_sec

    print(f"  Step time:        {step_time_ms:.1f} ms")
    print(f"  Tokens/sec:       {tokens_per_sec:.0f}")
    print(f"  Achieved:         {achieved_tflops:.1f} TFLOPS")
    print(f"  Hardware peak:    {peak/1e12:.1f} TFLOPS ({device})")
    print(f"  MFU:              {100*mfu:.1f}%")

    return {
        'mfu': mfu,
        'achieved_tflops': achieved_tflops,
        'tokens_per_sec': tokens_per_sec,
        'step_time_ms': step_time_ms,
    }
```

### 0.5 Decision Gate

**After Stage 0, you must have this table filled:**

```
┌──────────────────────────────────────────────────────────────┐
│  BASELINE PROFILE REPORT — NanoSeek Ablation on RTX 4090    │
│                                                              │
│  Step time:            ___ ms                                │
│  MFU:                  ___ %                                 │
│  Tokens/sec:           ___                                   │
│  Peak memory:          ___ / 24 GB                           │
│                                                              │
│  COMPONENT BREAKDOWN (forward + backward):                   │
│  ┌────────────────────────┬──────────┬───────┐               │
│  │ Expert compute (64 FF) │  ___ ms  │ ___ % │               │
│  │ Expert dispatch        │  ___ ms  │ ___ % │               │
│  │ Expert combine         │  ___ ms  │ ___ % │               │
│  │ MLA attention          │  ___ ms  │ ___ % │               │
│  │ Gate routing           │  ___ ms  │ ___ % │               │
│  │ Shared experts         │  ___ ms  │ ___ % │               │
│  │ RMSNorm               │  ___ ms  │ ___ % │               │
│  │ MTP head              │  ___ ms  │ ___ % │               │
│  │ LM head + loss        │  ___ ms  │ ___ % │               │
│  │ Optimizer             │  ___ ms  │ ___ % │               │
│  └────────────────────────┴──────────┴───────┘               │
│                                                              │
│  DECISION:                                                   │
│  If Expert compute+dispatch+combine > 30% → PROCEED          │
│  If Expert compute+dispatch+combine < 30% → PIVOT            │
│                                                              │
│  EXPECTED (based on literature):                             │
│  Expert ops: 45-55% of step time                             │
│  MLA attention: 15-25%                                       │
│  Everything else: 20-30%                                     │
└──────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Flash Attention Implementation & Benchmarking (Days 4-10)

### Objective
Implement Flash Attention in Triton from scratch. Benchmark against PyTorch SDPA. Learn the GPU programming fundamentals that transfer to MoE kernels.

### 1.1 Naive Attention Baseline

**What to build**: Standard attention in PyTorch (the thing Flash Attention replaces).

**What to measure**: Time and memory as sequence length grows.

```python
# File: nanoseek/benchmarks/attention_benchmarks.py

import triton
import triton.testing

def naive_attention(Q, K, V, causal=True):
    """Standard attention: materialize full N×N matrix."""
    scale = Q.shape[-1] ** -0.5
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, N, N]
    if causal:
        mask = torch.triu(torch.ones(N, N, device=Q.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, V)

# Benchmark across sequence lengths
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['seq_len'],
        x_vals=[512, 1024, 2048, 4096, 8192],
        line_arg='provider',
        line_vals=['naive', 'sdpa', 'triton_fa'],
        line_names=['Naive PyTorch', 'torch SDPA', 'Our Triton FA'],
        ylabel='ms',
        plot_name='attention-latency-vs-seqlen',
    )
)
def bench_attention(seq_len, provider):
    B, H, D = 2, 16, 64  # Match NanoSeek MLA dimensions
    Q = torch.randn(B, H, seq_len, D, device='cuda', dtype=torch.bfloat16)
    K = torch.randn(B, H, seq_len, D, device='cuda', dtype=torch.bfloat16)
    V = torch.randn(B, H, seq_len, D, device='cuda', dtype=torch.bfloat16)

    if provider == 'naive':
        fn = lambda: naive_attention(Q, K, V)
    elif provider == 'sdpa':
        fn = lambda: torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
    elif provider == 'triton_fa':
        fn = lambda: triton_flash_attention(Q, K, V)  # Your implementation

    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return ms
```

**What to compare**:

| Metric | Naive | SDPA | Your Triton FA | Target |
|--------|-------|------|----------------|--------|
| Time (ms) at seq=4096 | ___ | ___ | ___ | ≤1.2x SDPA |
| Memory (MB) at seq=4096 | ___ | ___ | ___ | ≤1.5x SDPA |
| Time scaling O(?) | O(N^2) | O(N) | O(N) | O(N) |
| Bandwidth (GB/s) | ___ | ___ | ___ | >500 GB/s |

### 1.2 Flash Attention Forward Pass in Triton

**What to build**: Tiled attention with online softmax.

**Critical benchmarks during development**:

```python
def validate_flash_attention_correctness(B=2, H=16, N=2048, D=64):
    """
    Correctness test: your FA must match naive attention.

    Tolerance: rtol=1e-2, atol=1e-2 for BF16
    (BF16 has ~3 decimal digits of precision)

    Why not tighter tolerance?
    - Online softmax accumulates in different order than full softmax
    - BF16 reductions are non-associative (a+b+c ≠ c+b+a in floating point)
    - The mathematically identical algorithm produces bitwise-different results
    - 1e-2 relative error is the standard for BF16 Flash Attention validation
    """
    Q = torch.randn(B, H, N, D, device='cuda', dtype=torch.bfloat16)
    K = torch.randn(B, H, N, D, device='cuda', dtype=torch.bfloat16)
    V = torch.randn(B, H, N, D, device='cuda', dtype=torch.bfloat16)

    ref = naive_attention(Q, K, V, causal=True)
    our = triton_flash_attention(Q, K, V, causal=True)

    max_err = (ref - our).abs().max().item()
    rel_err = ((ref - our).abs() / (ref.abs() + 1e-6)).max().item()

    print(f"  Max absolute error: {max_err:.6f}")
    print(f"  Max relative error: {rel_err:.6f}")
    assert rel_err < 0.02, f"Relative error {rel_err} exceeds 2% threshold"
```

**Kernel-level metrics to capture**:

```python
def profile_fa_kernel():
    """
    Profile individual Flash Attention Triton kernel.

    Metrics to extract:
    1. Achieved bandwidth: total_bytes_moved / time
       - Q read: B*H*N*D * 2 bytes (BF16)
       - K read: B*H*N*D * 2 bytes (BF16, read multiple times via tiling)
       - V read: B*H*N*D * 2 bytes
       - O write: B*H*N*D * 2 bytes
       - Theoretical minimum: 4 * B*H*N*D * 2 bytes (one read each + one write)
       - Actual: depends on SRAM tile size and reuse

    2. Achieved compute: estimated_flops / time
       - Forward attention FLOPs: 2 * B * H * N^2 * D (QK^T) + 2 * B * H * N^2 * D (Attn×V)
       - Causal: multiply by 0.5 (lower triangle only)
       - Total: 2 * B * H * N^2 * D (causal)

    3. Arithmetic intensity: FLOPs / bytes_moved
       - If < 82 (RTX 4090 ridge point) → memory-bound → optimize data reuse
       - If > 82 → compute-bound → optimize tile size for SM utilization

    4. SM occupancy via Nsight Compute
    """
    B, H, N, D = 2, 16, 4096, 64

    # Time the kernel
    ms = triton.testing.do_bench(
        lambda: triton_flash_attention(Q, K, V, causal=True),
        warmup=25, rep=100, return_mode='median'
    )

    # Calculate achieved metrics
    flops = 2 * B * H * N * N * D  # causal ≈ half
    bytes_moved = 4 * B * H * N * D * 2  # minimum bytes (BF16)

    achieved_tflops = flops / (ms * 1e-3) / 1e12
    achieved_bw = bytes_moved / (ms * 1e-3) / 1e9

    print(f"  Kernel time: {ms:.3f} ms")
    print(f"  Achieved: {achieved_tflops:.1f} TFLOPS")
    print(f"  Bandwidth: {achieved_bw:.1f} GB/s (peak: 1008 GB/s)")
    print(f"  Arithmetic intensity: {flops/bytes_moved:.1f} FLOP/Byte")
    print(f"  Memory efficiency: {100 * achieved_bw / 1008:.1f}%")
    print(f"  Compute efficiency: {100 * achieved_tflops / 82.6:.1f}%")

    # Classification
    if achieved_bw / 1008 > achieved_tflops / 82.6:
        print("  Classification: COMPUTE-BOUND (good for FA)")
    else:
        print("  Classification: MEMORY-BOUND (need better tiling)")
```

### 1.3 Flash Attention Backward Pass

**What to benchmark**: Backward pass is 2.5-3x more expensive than forward. Must verify.

```python
def benchmark_fa_forward_backward():
    """
    FA backward has 5 GEMMs vs forward's 2:
      Forward:  QK^T, Attn×V
      Backward: dAttn×V^T (→ dScores), dScores^T×Q (→ dK), dScores×K (→ dQ),
                Attn^T×dO (→ dV), plus recomputation of Attn from Q,K

    Expected ratio: backward/forward ≈ 2.5-3.0x
    If ratio is >4x, backward implementation has a bug or excessive recomputation.
    """
    Q = torch.randn(2, 16, 4096, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    K = torch.randn_like(Q, requires_grad=True)
    V = torch.randn_like(Q, requires_grad=True)

    fwd_ms = triton.testing.do_bench(
        lambda: triton_flash_attention(Q, K, V, causal=True),
        warmup=10, rep=50
    )

    def fwd_bwd():
        out = triton_flash_attention(Q, K, V, causal=True)
        out.sum().backward()

    total_ms = triton.testing.do_bench(fwd_bwd, warmup=10, rep=50)
    bwd_ms = total_ms - fwd_ms

    print(f"  Forward:  {fwd_ms:.2f} ms")
    print(f"  Backward: {bwd_ms:.2f} ms")
    print(f"  Ratio:    {bwd_ms/fwd_ms:.2f}x (expected: 2.5-3.0x)")
```

### 1.4 Compare Against NanoSeek's MLA

**What to measure**: How does your FA compare to the actual MLA attention path?

```python
def benchmark_mla_vs_standard_attention():
    """
    NanoSeek's MLA has DIFFERENT characteristics than standard MHA:

    Standard MHA:
      - KV cache per token: 2 × n_heads × head_dim = 2 × 16 × 128 = 4096 values
      - Full Q, K, V projections

    NanoSeek MLA:
      - KV cache per token: kv_lora_rank + qk_rope_head_dim = 143 + 64 = 207 values
      - 23× KV compression
      - But: extra projections (wq_a, wq_b, wkv_a, wkv_b) add compute

    Benchmark both to understand the compute-memory tradeoff.
    """
    # Standard MHA attention time at different seq lengths
    # vs MLA attention time
    # vs your Triton FA implementation

    # Expected: MLA is faster for long sequences (less memory traffic)
    # but slower for short sequences (extra projection overhead)
```

---

## Stage 2: Grouped GEMM Development & Benchmarking (Days 11-18)

### Objective
Replace sequential 64-expert loop with grouped GEMM. Measure the speedup rigorously.

### 2.1 Sequential Expert Baseline

**What to measure**: The exact cost of NanoSeek's current batched expert path.

```python
# File: nanoseek/benchmarks/moe_benchmarks.py

def benchmark_current_moe_dispatch(model, batch_size=2, seq_len=4096):
    """
    Profile NanoSeek's existing MoE dispatch.

    NanoSeek has TWO paths (model.py lines 727-863):

    1. BATCHED path (default on GPU, E≥8, waste_ratio<1.5):
       - Sorts tokens by expert assignment
       - Pads to max_count per expert
       - Uses torch.bmm for batched matrix multiply
       - 2 kernel launches for all 192 matmuls (3 projections × 64 experts)

    2. SEQUENTIAL path (fallback):
       - Loops over 64 experts
       - Each expert: 3 matmuls (gate, up, down)
       - 192 sequential CUDA kernel launches

    We must benchmark the BATCHED path since that's what runs on GPU.

    Key dimensions (ablation scale):
      hidden_dim = 1280
      inter_dim = 480 (per routed expert)
      shared_inter_dim = 960 (2 × 480)
      num_experts = 64
      top_k = 8
      tokens per step = batch_size × seq_len = 2 × 4096 = 8192
      selected tokens = tokens × top_k = 8192 × 8 = 65536
      tokens_per_expert (avg) = 65536 / 64 = 1024

    GEMMs in batched path:
      gate: bmm([64, max_count, 1280], [64, 1280, 480]) → [64, max_count, 480]
      up:   bmm([64, max_count, 1280], [64, 1280, 480]) → [64, max_count, 480]
      down: bmm([64, max_count, 480], [64, 480, 1280])  → [64, max_count, 1280]
    """
    # Isolate MoE layer from a single decoder layer
    moe_layer = model.layers[0].moe  # MoEDispatch instance

    # Create representative input
    hidden_states = torch.randn(batch_size, seq_len, model.config.hidden_size,
                                device='cuda', dtype=torch.bfloat16)

    # Warm up
    for _ in range(5):
        _ = moe_layer(hidden_states.view(-1, model.config.hidden_size))

    # Time forward
    fwd_ms = triton.testing.do_bench(
        lambda: moe_layer(hidden_states.view(-1, model.config.hidden_size)),
        warmup=10, rep=50
    )

    # Time forward + backward
    def fwd_bwd():
        out = moe_layer(hidden_states.view(-1, model.config.hidden_size))
        out.sum().backward()

    total_ms = triton.testing.do_bench(fwd_bwd, warmup=10, rep=50)

    print(f"\n  MoE Layer Timing (batched path, ablation scale):")
    print(f"  Forward:  {fwd_ms:.2f} ms")
    print(f"  Backward: {total_ms - fwd_ms:.2f} ms")
    print(f"  Total:    {total_ms:.2f} ms")

    # Break down within the forward
    # Time each sub-component separately
    return fwd_ms, total_ms
```

### 2.2 Grouped GEMM Kernel

**What to build**: Replace batched expert BMM with `torch._grouped_mm` or Triton grouped GEMM.

**What to benchmark at each step**:

```python
def benchmark_grouped_vs_batched_gemm():
    """
    A/B comparison: NanoSeek batched BMM vs grouped GEMM.

    Test matrix (sweep these dimensions):

    | num_experts | tokens_per_expert | hidden_dim | inter_dim | Provider       |
    |-------------|-------------------|------------|-----------|----------------|
    | 8           | 128               | 1280       | 480       | bmm            |
    | 8           | 128               | 1280       | 480       | grouped_mm     |
    | 8           | 128               | 1280       | 480       | triton_grouped |
    | 16          | 256               | 1280       | 480       | bmm            |
    | 16          | 256               | 1280       | 480       | grouped_mm     |
    | 32          | 512               | 1280       | 480       | bmm            |
    | 32          | 512               | 1280       | 480       | grouped_mm     |
    | 64          | 1024              | 1280       | 480       | bmm            |
    | 64          | 1024              | 1280       | 480       | grouped_mm     |
    | 64          | 1024              | 2048       | 768       | bmm            |
    | 64          | 1024              | 2048       | 768       | grouped_mm     |

    Why sweep num_experts:
    - grouped_mm advantage grows with expert count
    - At E=8 (Mixtral), bmm might win (large per-expert batches)
    - At E=64 (DeepSeek/NanoSeek), grouped_mm should dominate

    Why sweep tokens_per_expert:
    - Imbalanced routing → some experts get very few tokens
    - BMM pads to max_count → waste
    - grouped_mm handles variable sizes natively → no waste

    Report format:
    ┌───────┬──────┬────────┬───────────┬──────────┬──────────┐
    │ E     │ T/E  │ D      │ BMM (ms)  │ Grp (ms) │ Speedup  │
    ├───────┼──────┼────────┼───────────┼──────────┼──────────┤
    │ 64    │ 1024 │ 1280   │ 12.4      │ 5.1      │ 2.43x    │
    │ 64    │ 1024 │ 2048   │ 28.7      │ 11.2     │ 2.56x    │
    └───────┴──────┴────────┴───────────┴──────────┴──────────┘
    """
    results = []

    for E in [8, 16, 32, 64]:
        for tpe in [128, 256, 512, 1024]:
            for D, inter in [(1280, 480), (2048, 768)]:
                # Batched BMM (current NanoSeek)
                x = torch.randn(E, tpe, D, device='cuda', dtype=torch.bfloat16)
                W = torch.randn(E, D, inter, device='cuda', dtype=torch.bfloat16)

                bmm_ms = triton.testing.do_bench(
                    lambda: torch.bmm(x, W), warmup=25, rep=100
                )

                # Grouped GEMM (torch._grouped_mm)
                # Flatten and provide offsets
                x_flat = x.view(E * tpe, D)
                offsets = torch.arange(0, (E+1) * tpe, tpe, device='cuda')

                try:
                    grp_ms = triton.testing.do_bench(
                        lambda: torch._grouped_mm(x_flat, W, offs=offsets),
                        warmup=25, rep=100
                    )
                except:
                    grp_ms = float('inf')  # API might not be available

                speedup = bmm_ms / grp_ms if grp_ms > 0 else 0
                results.append((E, tpe, D, bmm_ms, grp_ms, speedup))

    # Print table
    print(f"{'E':>5} {'T/E':>6} {'D':>6} {'BMM(ms)':>10} {'Grp(ms)':>10} {'Speedup':>8}")
    for r in results:
        print(f"{r[0]:5d} {r[1]:6d} {r[2]:6d} {r[3]:10.2f} {r[4]:10.2f} {r[5]:8.2f}x")
```

### 2.3 Align & Sort Dispatch Kernel

**What to benchmark**: Token sorting overhead — must be negligible relative to GEMM savings.

```python
def benchmark_align_and_sort():
    """
    The dispatch kernel sorts tokens by expert assignment.

    This is overhead — it must be fast enough that the grouped GEMM
    savings exceed the sorting cost.

    Rule of thumb: dispatch should be <5% of total MoE time.

    Compare against:
    1. torch.argsort (baseline)
    2. torch.bincount + scatter (histogram approach)
    3. Custom Triton radix sort (if needed)

    Dimensions:
      num_tokens = 8192 (batch=2 × seq=4096)
      top_k = 8
      flat_tokens = 65536
      num_experts = 64
    """
    for N in [8192, 32768, 65536, 131072]:
        expert_ids = torch.randint(0, 64, (N,), device='cuda')

        # Method 1: torch.argsort
        argsort_ms = triton.testing.do_bench(
            lambda: expert_ids.argsort(stable=True),
            warmup=25, rep=100
        )

        # Method 2: bincount + scatter
        def bincount_scatter():
            counts = torch.bincount(expert_ids, minlength=64)
            offsets = torch.cumsum(counts, dim=0)
            # ... scatter tokens to sorted positions
            return offsets

        scatter_ms = triton.testing.do_bench(bincount_scatter, warmup=25, rep=100)

        print(f"  N={N:>7d}: argsort={argsort_ms:.3f}ms, scatter={scatter_ms:.3f}ms")
```

### 2.4 End-to-End Fused MoE Comparison

**The critical benchmark**: Fused MoE layer vs original MoE layer.

```python
def benchmark_fused_vs_original_moe():
    """
    End-to-end comparison: NanoFuse MoE layer vs NanoSeek original.

    This is the benchmark that matters. Individual kernel benchmarks
    can be misleading because:
    1. Kernel launch overhead compounds differently in pipelines
    2. Memory allocation patterns differ
    3. GPU caching effects vary with execution order

    Methodology:
    1. Both implementations receive IDENTICAL inputs
    2. Both produce IDENTICAL outputs (verified by correctness test)
    3. Timed with CUDA events, not wall-clock
    4. 5 warmup + 50 timed iterations, report median
    5. Separate forward-only and forward+backward measurements

    Report format:
    ┌──────────────────────────┬──────────┬──────────┬──────────┐
    │ Metric                   │ Original │ NanoFuse │ Speedup  │
    ├──────────────────────────┼──────────┼──────────┼──────────┤
    │ Forward (ms)             │          │          │          │
    │ Backward (ms)            │          │          │          │
    │ Total (ms)               │          │          │          │
    │ Peak memory (MB)         │          │          │          │
    │ Tokens/sec               │          │          │          │
    │ TFLOPS (expert compute)  │          │          │          │
    └──────────────────────────┴──────────┴──────────┴──────────┘
    """
    pass  # Implement after building NanoFuse
```

---

## Stage 3: Fused SwiGLU & Activation Optimization (Days 19-22)

### Objective
Fuse SwiGLU activation into the MoE pipeline. Benchmark memory savings.

### 3.1 SwiGLU Kernel Comparison

```python
def benchmark_swiglu_implementations():
    """
    Compare SwiGLU implementations:

    1. Standard (3 ops):
       gate_out = x @ W_gate          # GEMM
       up_out = x @ W_up              # GEMM
       h = F.silu(gate_out) * up_out  # elementwise (2 kernel launches)

    2. Fused Triton (1 op):
       h = fused_swiglu(gate_out, up_out)  # 1 kernel launch

    3. Liger-Kernel reference (if installed):
       h = LigerSiLUMulFunction.apply(gate_out, up_out)

    Memory comparison:
    - Standard: materializes gate_out AND up_out AND silu_gate
      → 3 × [tokens, inter_dim] intermediate tensors
    - Fused: materializes ONLY h
      → 1 × [tokens, inter_dim] intermediate tensor
    - With backward recomputation: saves silu_gate entirely
      → 1.6x memory reduction (Liger-Kernel verified number)

    The key insight: this is a MEMORY optimization, not a compute optimization.
    SwiGLU is elementwise → memory-bound → bandwidth is the metric.
    """
    for N in [1024, 4096, 16384, 65536]:
        for D in [480, 768, 1536]:  # ablation inter, 1B inter, shared inter
            gate = torch.randn(N, D, device='cuda', dtype=torch.bfloat16)
            up = torch.randn(N, D, device='cuda', dtype=torch.bfloat16)

            # Standard
            def standard():
                return F.silu(gate) * up

            std_ms = triton.testing.do_bench(standard, warmup=25, rep=100)

            # Fused Triton
            def fused():
                return fused_swiglu_triton(gate, up)

            fused_ms = triton.testing.do_bench(fused, warmup=25, rep=100)

            # Bandwidth calculation
            # Input: 2 × N × D × 2 bytes (gate + up, BF16)
            # Output: 1 × N × D × 2 bytes
            total_bytes = 3 * N * D * 2
            std_bw = total_bytes / (std_ms * 1e-3) / 1e9
            fused_bw = total_bytes / (fused_ms * 1e-3) / 1e9

            print(f"  N={N:>5d} D={D:>4d}: "
                  f"std={std_ms:.3f}ms ({std_bw:.0f} GB/s), "
                  f"fused={fused_ms:.3f}ms ({fused_bw:.0f} GB/s), "
                  f"speedup={std_ms/fused_ms:.2f}x, "
                  f"bw_efficiency={100*fused_bw/1008:.0f}%")
```

### 3.2 Memory Savings Measurement

```python
def measure_swiglu_memory_savings():
    """
    Measure ACTUAL memory savings from fused SwiGLU.

    Method: torch.cuda.max_memory_allocated() before vs after.

    Expected (ablation scale, tokens=8192, inter_dim=480):
      Standard: 3 × 8192 × 480 × 2 bytes = 23.6 MB per expert
                × 64 experts = 1.51 GB (but batched, so [64, max_count, 480])
      Fused:    1 × 8192 × 480 × 2 bytes = 7.9 MB per expert
                × 64 experts = 0.50 GB
      Savings:  ~1.0 GB per layer × 16 layers = ~16 GB total

    But wait — with gradient checkpointing, activations are recomputed.
    So the savings stack with checkpointing (save the recomputed intermediate too).
    """
    pass
```

---

## Stage 4: Blockwise FP8 Quantization Development (Days 23-28)

### Objective
Implement blockwise FP8 quantization in Triton. Benchmark precision vs tensorwise.

**NOTE**: RTX 4090 has NO FP8 tensor cores. We can still:
1. Develop and test the quantization logic
2. Measure quantization error (precision comparison)
3. Profile the quantization kernel itself (overhead measurement)
4. Validate correctness by dequantizing and comparing to BF16

FP8 GEMM benchmarks require H100 (Stage 5 / Tier 2).

### 4.1 Quantization Precision Comparison

```python
def benchmark_fp8_quantization_precision():
    """
    Compare quantization error: tensorwise vs blockwise.

    This is the PRECISION benchmark — no GPU-specific hardware needed.

    For each NanoSeek weight matrix:
    1. Quantize with tensorwise scaling (current fp8.py approach)
    2. Quantize with blockwise 128×128 scaling (DeepSeek V3 approach)
    3. Dequantize both
    4. Compare reconstruction error vs original BF16

    Metrics:
    - Max absolute error
    - Mean absolute error
    - Relative error (Frobenius norm)
    - Per-channel error distribution (histogram)
    - Outlier analysis: % of values with >1% relative error

    Why blockwise should win:
    - Tensorwise: ONE scale for entire matrix
      If max(|W|) = 100 but 99% of values < 0.1,
      those small values get just 1-2 bits of precision in FP8
    - Blockwise: ONE scale per 128×128 block
      Each block's dynamic range is utilized fully
      Small values in blocks without outliers get full precision

    Expected results:
    - Blockwise relative error: 2-5x lower than tensorwise
    - Improvement is largest for matrices with outlier rows/columns
    """
    # Test on actual NanoSeek weights (or random with similar statistics)
    weight_shapes = {
        'expert_gate':     (480, 1280),    # ablation: routed expert gate proj
        'expert_up':       (480, 1280),    # ablation: routed expert up proj
        'expert_down':     (1280, 480),    # ablation: routed expert down proj
        'shared_gate':     (960, 1280),    # ablation: shared expert
        'mla_wq_a':        (275, 1280),    # ablation: MLA Q compression
        'mla_wkv_a':       (207, 1280),    # ablation: MLA KV compression (kv_lora+rope)
        'mla_wo':          (1280, 1280),   # ablation: MLA output (n_heads*v_head_dim)
        '1b_expert_gate':  (768, 2048),    # 1B: routed expert
        '1b_expert_down':  (2048, 768),    # 1B: routed expert
        '1b_mla_wo':       (2048, 2048),   # 1B: MLA output
    }

    for name, (out_dim, in_dim) in weight_shapes.items():
        W = torch.randn(out_dim, in_dim, dtype=torch.bfloat16, device='cuda')
        # Add realistic outliers (1% of values are 10x larger)
        outlier_mask = torch.rand_like(W) < 0.01
        W[outlier_mask] *= 10.0

        # Tensorwise
        tw_scale = W.float().abs().max() / 448.0
        tw_q = (W.float() / tw_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        tw_recon = tw_q.float() * tw_scale
        tw_err = (tw_recon - W.float()).norm() / W.float().norm()

        # Blockwise 128×128
        bw_err = blockwise_quantize_and_measure_error(W, block_size=128)

        # FP8 eligibility check (NanoSeek rules)
        eligible = (min(out_dim, in_dim) >= 128 and
                   out_dim % 16 == 0 and in_dim % 16 == 0)

        print(f"  {name:20s} [{out_dim:>4d}×{in_dim:>4d}] "
              f"tw_err={tw_err:.6f} bw_err={bw_err:.6f} "
              f"improvement={tw_err/bw_err:.2f}x "
              f"fp8_eligible={eligible}")
```

### 4.2 Quantization Kernel Overhead

```python
def benchmark_quantization_overhead():
    """
    Measure the TIME COST of quantization itself.

    Blockwise quantization requires:
    1. Tiling the matrix into 128×128 blocks
    2. Computing max(|block|) for each block
    3. Dividing by scale and casting to FP8

    This is pure overhead — it adds time that the GEMM speedup must exceed.

    Rule: quantization overhead must be <10% of GEMM time.
    If quantization takes 1ms and GEMM takes 5ms, total = 6ms.
    Original BF16 GEMM must take >6ms for FP8 to be worthwhile.

    For weight quantization: done ONCE at model init (amortized to zero).
    For activation quantization: done EVERY forward pass (must be fast).
    """
    for M, K in [(8192, 1280), (8192, 2048), (65536, 1280), (65536, 2048)]:
        x = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)

        # Tensorwise (current fp8.py: ~3 ops)
        def tensorwise():
            amax = x.float().abs().max()
            scale = 448.0 / amax.clamp(min=1e-12)
            return (x.float() * scale).clamp(-448, 448).to(torch.float8_e4m3fn), scale.reciprocal()

        tw_ms = triton.testing.do_bench(tensorwise, warmup=25, rep=100)

        # Blockwise (our Triton kernel)
        def blockwise():
            return blockwise_fp8_quantize(x, block_size=128)

        bw_ms = triton.testing.do_bench(blockwise, warmup=25, rep=100)

        # Reference: how long does a BF16 GEMM take at this size?
        W = torch.randn(K, 480, device='cuda', dtype=torch.bfloat16)  # expert weight
        gemm_ms = triton.testing.do_bench(lambda: x @ W, warmup=25, rep=100)

        print(f"  [{M:>5d}×{K:>4d}]: "
              f"tw_quant={tw_ms:.3f}ms, bw_quant={bw_ms:.3f}ms, "
              f"gemm_bf16={gemm_ms:.3f}ms, "
              f"overhead={100*bw_ms/gemm_ms:.1f}%")
```

---

## Stage 5: End-to-End Training Validation (Days 29-35)

### Objective
Validate that NanoFuse produces identical training dynamics to the baseline.

### 5.1 Loss Curve Comparison

```python
def run_training_comparison(num_steps=500):
    """
    THE definitive validation: do loss curves match?

    Run two identical training runs, differing ONLY in MoE implementation:
    1. Baseline: NanoSeek sequential/batched MoE
    2. NanoFuse: Fused grouped GEMM MoE

    Same seed, same data, same everything else.

    What to compare (per step):
    - train/loss: must match within noise (< 0.1% relative difference)
    - ema_val/bpb: must match within 0.5% after 500 steps
    - train/h_load: expert balance must be identical
    - eval/i_spec_mean: expert specialization must be identical
    - train/grad_norm: gradient statistics must match
    - train/mtp_loss: MTP head must be unaffected

    If ANY metric diverges > 1%: the fused kernel has a bug.
    Gradient correctness issues often manifest as:
    - Slowly increasing gap in train/loss (accumulating error)
    - Different expert specialization patterns (routing affected)
    - Grad norm divergence (numerical instability)

    Report format:
    ┌───────────────────────────┬──────────┬──────────┬──────────┐
    │ Metric @ step 500         │ Baseline │ NanoFuse │ Δ (%)    │
    ├───────────────────────────┼──────────┼──────────┼──────────┤
    │ ema_val/bpb               │ 4.231    │ 4.228    │ -0.07%   │
    │ train/h_load              │ 5.82     │ 5.81     │ -0.17%   │
    │ eval/i_spec_mean          │ 0.142    │ 0.143    │ +0.70%   │
    │ train/grad_norm           │ 0.847    │ 0.849    │ +0.24%   │
    │ train/mtp_loss            │ 6.123    │ 6.119    │ -0.07%   │
    │ step_time (ms)            │ 210      │ 135      │ -35.7%   │ ← THE WIN
    │ tokens/sec                │ 39024    │ 60741    │ +55.6%   │
    │ peak_memory (GB)          │ 18.2     │ 12.4     │ -31.9%   │
    └───────────────────────────┴──────────┴──────────┴──────────┘
    """
    # Run baseline
    # python -m nanoseek.scripts.pre_train \
    #     --run bench-baseline --scale ablation --seed 42 \
    #     --num-iterations 500 --eval-every 100 --save-every -1 \
    #     --device-batch-size 2

    # Run NanoFuse
    # python -m nanoseek.scripts.pre_train \
    #     --run bench-nanofuse --scale ablation --seed 42 \
    #     --num-iterations 500 --eval-every 100 --save-every -1 \
    #     --device-batch-size 2 --fused-moe
    pass
```

### 5.2 Throughput Metrics

```python
def compute_throughput_metrics(step_times_ms, batch_size, seq_len, model_config):
    """
    Compute all throughput metrics that a frontier lab would report.

    Metrics:
    1. Tokens/second = batch_size × seq_len / step_time
    2. Samples/second = batch_size / step_time
    3. MFU = model_flops / (step_time × hardware_peak)
    4. HFU = total_flops_including_recomputation / (step_time × hardware_peak)
    5. Time to train (extrapolated) = total_tokens / tokens_per_second
    6. Cost to train = time_to_train × GPU_cost_per_hour

    For NanoSeek ablation on RTX 4090:
      total_tokens = 8.2B
      GPU cost = ~$0.40/hr (cloud) or $0/hr (owned)

    Report:
      Baseline: _____ tokens/sec → _____ hours → $___
      NanoFuse: _____ tokens/sec → _____ hours → $___
      Savings:  _____ hours, $___
    """
    median_ms = sorted(step_times_ms)[len(step_times_ms) // 2]
    tokens_per_sec = batch_size * seq_len / (median_ms / 1000)

    total_tokens = model_config.total_tokens  # 8.2B for ablation
    hours_to_train = total_tokens / tokens_per_sec / 3600

    print(f"  Median step time: {median_ms:.1f} ms")
    print(f"  Tokens/sec:       {tokens_per_sec:.0f}")
    print(f"  Hours to train:   {hours_to_train:.1f}")
    print(f"  Cost @ $0.40/hr:  ${hours_to_train * 0.40:.0f}")
```

### 5.3 Roofline Analysis for Each Kernel

```python
def generate_roofline_data():
    """
    Generate data points for roofline plot.

    Each kernel gets ONE point on the roofline:
    - X-axis: Arithmetic Intensity (FLOP/Byte)
    - Y-axis: Achieved GFLOP/s

    RTX 4090 roofline:
    - Memory bandwidth ceiling: 1008 GB/s
    - Compute ceiling: 82.6 TFLOPS (BF16)
    - Ridge point: 82.6 / 1.008 = 81.9 FLOP/Byte

    Kernels to plot:
    ┌──────────────────────────────────────────────────────────────┐
    │                    RTX 4090 Roofline                         │
    │                                                              │
    │  82.6T ─────────────────────────────────────── compute ceil  │
    │         /                                                    │
    │  GFLOP/s /     ★ grouped_gemm                               │
    │       /      ★ align_sort                                    │
    │      /   ★ fused_swiglu                                      │
    │     /  ★ rmsnorm                                             │
    │    / ★ combine                                               │
    │   /                                                          │
    │  ── 1008 GB/s bandwidth ceil ──────────                      │
    │                                                              │
    │  1      10     82    100    1000                              │
    │         Arithmetic Intensity (FLOP/Byte)                     │
    └──────────────────────────────────────────────────────────────┘

    Expected classification:
    - grouped_gemm: COMPUTE-BOUND (high AI from large matmuls)
    - align_sort: MEMORY-BOUND (scatter/gather, low AI)
    - fused_swiglu: MEMORY-BOUND (elementwise, AI ≈ 1)
    - rmsnorm: MEMORY-BOUND (reduction + elementwise, AI ≈ 2)
    - combine: MEMORY-BOUND (scatter-add, AI ≈ 1)
    """
    kernels = {
        'grouped_gemm': {
            'flops': lambda E, T, D, I: 2 * E * T * D * I,  # matmul
            'bytes': lambda E, T, D, I: (E*T*D + E*D*I + E*T*I) * 2,  # BF16
        },
        'fused_swiglu': {
            'flops': lambda N, D: 5 * N * D,  # silu (4 ops) + mul
            'bytes': lambda N, D: 3 * N * D * 2,  # 2 inputs + 1 output
        },
        'rmsnorm': {
            'flops': lambda N, D: 5 * N * D,  # square, sum, rsqrt, mul, mul
            'bytes': lambda N, D: 3 * N * D * 2,  # input + weight + output
        },
        'align_sort': {
            'flops': lambda N: N * 10,  # ~10 ops per element (comparison, swap)
            'bytes': lambda N: N * 8,  # read int64 + write int64
        },
        'combine_scatter': {
            'flops': lambda N, D: 2 * N * D,  # multiply + add
            'bytes': lambda N, D: 3 * N * D * 2,  # expert_out + weights + output
        },
    }

    # Compute AI and achieved GFLOP/s for each kernel
    # Plot using matplotlib
    return kernels
```

---

## Appendix A: Nsight Compute Commands for RTX 4090

```bash
# Profile a specific training step (isolate with NVTX)
ncu --set full \
    --target-processes all \
    --launch-skip 100 --launch-count 10 \
    -o nanofuse_profile \
    python -m nanoseek.benchmarks.profile_kernel --kernel moe_forward

# Extract key metrics for a Triton kernel
ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum \
    python -m nanoseek.benchmarks.profile_kernel

# Compare two implementations
ncu --set full -o baseline python -m nanoseek.benchmarks.run_moe_baseline
ncu --set full -o nanofuse python -m nanoseek.benchmarks.run_moe_nanofuse
# Open both .ncu-rep files in Nsight Compute GUI for side-by-side comparison

# Check for bank conflicts
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
    python -m nanoseek.benchmarks.profile_kernel
```

---

## Appendix B: Benchmark Execution Checklist

### Before Every Benchmark Session
```
□ Close all other GPU processes (nvidia-smi → no other processes)
□ Set GPU to persistence mode: sudo nvidia-smi -pm 1
□ Lock GPU clocks: sudo nvidia-smi -lgc 2520 (RTX 4090 max)
□ Lock memory clocks: sudo nvidia-smi -lmc 10501
□ Verify: nvidia-smi -q -d CLOCK shows locked frequencies
□ Set CUDA_VISIBLE_DEVICES=0
□ Disable torch.compile caching: TORCHINDUCTOR_CACHE_DIR=/tmp/triton_$$
```

### During Benchmarks
```
□ Always warmup (≥5 iterations, ≥25ms for triton.testing.do_bench)
□ Use CUDA events for timing (not time.time())
□ Call torch.cuda.synchronize() before reading results
□ Report median, not mean (outliers from GC, OS interrupts)
□ Record GPU temperature (nvidia-smi -q -d TEMPERATURE)
□ If temp > 83°C, wait for cooldown before next benchmark
□ Run each benchmark ≥3 times, report median of medians
```

### After Benchmarks
```
□ Unlock GPU clocks: sudo nvidia-smi -rgc
□ Unlock memory clocks: sudo nvidia-smi -rmc
□ Save all raw data (not just summary statistics)
□ Commit benchmark scripts + results to git
□ Update baseline_profile.md with new measurements
```

---

## Appendix C: Common Benchmarking Mistakes

| Mistake | Why It's Wrong | How to Fix |
|---------|---------------|------------|
| Using `time.time()` | Measures kernel launch, not execution | Use `torch.cuda.Event` |
| No warmup | First iteration includes JIT, cache warmup | Skip ≥5 iterations |
| Reporting mean | Outliers from GC/OS skew results | Report median |
| Unlocked GPU clocks | Frequency throttling varies between runs | Lock clocks |
| Benchmarking compiled code first run | torch.compile takes minutes to compile | Separate compile from benchmark |
| Comparing across temperatures | GPU throttles at 83°C | Monitor temp, wait for cooldown |
| Not clearing gradient | Grad accumulation changes memory pressure | Set `grad_to_none=True` |
| Profiling with `with_stack=True` | 10-50% overhead from stack tracing | Only for targeted debugging |
| Missing `torch.cuda.synchronize()` | Event timing returns garbage | Always sync before reading |
| Benchmarking under `torch.no_grad()` only | Backward pass has different characteristics | Benchmark both fwd and fwd+bwd |

---

## Appendix D: File Structure

```
nanoseek/benchmarks/
├── README.md                          # This guide (condensed)
├── profile_baseline.py                # Stage 0: PyTorch profiler
├── component_timing.py                # Stage 0: CUDA event timing
├── memory_profile.py                  # Stage 0: Memory breakdown
├── mfu.py                             # Stage 0: MFU calculation
├── attention_benchmarks.py            # Stage 1: FA implementations
├── moe_benchmarks.py                  # Stage 2: Grouped GEMM
├── swiglu_benchmarks.py               # Stage 3: Fused activation
├── fp8_precision_benchmarks.py        # Stage 4: Quantization error
├── fp8_overhead_benchmarks.py         # Stage 4: Quant kernel time
├── training_comparison.py             # Stage 5: Loss curve A/B
├── roofline.py                        # Roofline plot generation
├── traces/                            # Chrome traces from profiler
│   ├── baseline/                      # Stage 0 traces
│   ├── nanofuse/                      # Post-optimization traces
│   └── *.json                         # Chrome trace exports
├── results/                           # Benchmark results (CSV/JSON)
│   ├── baseline_profile.md            # Stage 0 report
│   ├── grouped_gemm_comparison.csv    # Stage 2 data
│   ├── swiglu_comparison.csv          # Stage 3 data
│   ├── fp8_precision.csv              # Stage 4 data
│   └── training_comparison.csv        # Stage 5 data
└── plots/                             # Generated charts
    ├── roofline_rtx4090.png
    ├── attention_scaling.png
    ├── moe_speedup.png
    └── loss_curves_comparison.png
```

---

## Summary: What Gets Measured at Each Stage

| Stage | Days | Key Question | Key Metric | Pass Criterion |
|-------|------|-------------|------------|----------------|
| **0** | 1-3 | Where does time go? | % per component | MoE > 30% |
| **1** | 4-10 | Can you write GPU kernels? | FA vs SDPA time | ≤ 1.2x SDPA |
| **2** | 11-18 | Does grouped GEMM beat batched? | Speedup ratio | ≥ 1.5x |
| **3** | 19-22 | Does fusion save memory? | Peak MB reduction | ≥ 40% |
| **4** | 23-28 | Does blockwise beat tensorwise? | Relative error | ≥ 2x lower |
| **5** | 29-35 | Does training still converge? | BPB deviation | < 0.5% |
