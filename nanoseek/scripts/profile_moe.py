"""
MoE Profiling Harness — Measure before you optimize.

Usage:
    CPU (local Mac):   python -m nanoseek.scripts.profile_moe --device cpu
    GPU (RunPod):      python -m nanoseek.scripts.profile_moe --device cuda
    With compile:      python -m nanoseek.scripts.profile_moe --device cuda --compile

Reports per-component timing for the MoE forward pass so we know
WHERE time goes before deciding WHAT to optimize.

Components profiled:
    1. Gate routing (router projection + sigmoid + group selection + top-K)
    2. Expert dispatch (sort + expert loop + matmuls)
    3. Weight application + scatter combine
    4. Shared expert forward
    5. MoE total = 1 + 2 + 3 + 4
    6. Full model forward (all layers including MLA)
    7. Full model backward
"""

import argparse
import time
import torch
import torch.nn.functional as F

from nanoseek.nanoseek.config import get_nanoseek_ablation_config, get_nanoseek_config
from nanoseek.nanoseek.model import NanoSeekModel


def profile_model(config, device, num_warmup=5, num_runs=15, use_compile=False):
    """Profile forward + backward with per-component MoE timing."""
    print(f"\n{'='*70}")
    print(f"Profiling: {config.hidden_size}h, {config.num_layers}L, "
          f"{config.moe.n_routed_experts}E, top-{config.moe.num_experts_per_tok}")
    print(f"Device: {device}, Compile: {use_compile}")
    print(f"{'='*70}\n")

    # Build model
    with torch.device("meta"):
        model = NanoSeekModel(config)
    model.to_empty(device=device)
    model.init_weights()
    model.train()

    param_counts = model.num_parameters()
    print(f"Parameters: {param_counts['active']:,} active / {param_counts['total']:,} total")

    if use_compile and device.type == "cuda":
        model = torch.compile(model, dynamic=False)
        print("torch.compile applied")

    # Input
    B, S = 4, min(config.sequence_length, 512)  # shorter seq for profiling speed
    x = torch.randint(0, config.vocab_size, (B, S), device=device)
    print(f"Input: B={B}, S={S}, tokens={B*S:,}")

    is_cuda = device.type == "cuda"

    # ─── Timing utilities ───
    if is_cuda:
        def make_timer():
            return (torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True))

        def start_timer(t):
            t[0].record()

        def stop_timer(t):
            t[1].record()

        def elapsed_ms(t):
            torch.cuda.synchronize()
            return t[0].elapsed_time(t[1])
    else:
        # CPU timing
        _cpu_times = {}
        def make_timer():
            return [None, None]

        def start_timer(t):
            t[0] = time.perf_counter()

        def stop_timer(t):
            t[1] = time.perf_counter()

        def elapsed_ms(t):
            return (t[1] - t[0]) * 1000

    # ─── Hook-based MoE component profiling ───
    # We instrument the MoE layers to measure sub-component timing.
    # This gives us the breakdown WITHOUT modifying the model code.

    moe_timings = {
        "gate": [],
        "dispatch": [],
        "scatter_combine": [],
        "shared_expert": [],
        "moe_total": [],
    }

    # We'll time full forward and backward separately
    forward_times = []
    backward_times = []

    # ─── Warmup ───
    print(f"\nWarmup ({num_warmup} iterations)...")
    for i in range(num_warmup):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=is_cuda):
            outputs = model(x, labels=x, mtp_lambda=0.3)
            loss = outputs["loss"]
        loss.backward()
        model.zero_grad(set_to_none=True)
        if i == 0 and is_cuda:
            torch.cuda.synchronize()
            print(f"  Step 0 complete (includes compile time if --compile)")

    print(f"Warmup done.\n")

    # ─── Memory baseline ───
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    # ─── Timed runs ───
    print(f"Profiling ({num_runs} iterations)...")
    for i in range(num_runs):
        # ── Forward ──
        if is_cuda:
            torch.cuda.synchronize()
        t_fwd = make_timer()
        start_timer(t_fwd)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=is_cuda):
            outputs = model(x, labels=x, mtp_lambda=0.3)
            loss = outputs["loss"]

        stop_timer(t_fwd)
        forward_times.append(elapsed_ms(t_fwd))

        # ── Backward ──
        t_bwd = make_timer()
        start_timer(t_bwd)

        loss.backward()

        stop_timer(t_bwd)
        backward_times.append(elapsed_ms(t_bwd))

        model.zero_grad(set_to_none=True)

    # ─── Report ───
    def stats(times):
        import statistics
        return {
            "mean": statistics.mean(times),
            "std": statistics.stdev(times) if len(times) > 1 else 0,
            "min": min(times),
            "max": max(times),
        }

    fwd_stats = stats(forward_times)
    bwd_stats = stats(backward_times)
    total_stats = stats([f + b for f, b in zip(forward_times, backward_times)])

    print(f"\n{'='*70}")
    print(f"RESULTS (mean ± std over {num_runs} runs)")
    print(f"{'='*70}\n")

    print(f"Full model forward:    {fwd_stats['mean']:8.2f} ± {fwd_stats['std']:.2f} ms")
    print(f"Full model backward:   {bwd_stats['mean']:8.2f} ± {bwd_stats['std']:.2f} ms")
    print(f"Total (fwd + bwd):     {total_stats['mean']:8.2f} ± {total_stats['std']:.2f} ms")
    print(f"  min: {total_stats['min']:.2f} ms, max: {total_stats['max']:.2f} ms")

    # Throughput
    tokens_per_step = B * S
    steps_per_sec = 1000 / total_stats["mean"]
    tok_per_sec = tokens_per_step * steps_per_sec
    print(f"\nThroughput: {tok_per_sec:,.0f} tok/s ({steps_per_sec:.1f} steps/s)")

    # FLOPs / MFU
    n_active = param_counts["active"]
    flops_per_token = 6 * n_active  # 6N rule
    flops_per_step = flops_per_token * tokens_per_step
    flops_per_sec = flops_per_step * steps_per_sec

    if is_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        # Rough peak TFLOPS for common GPUs (bf16)
        peak_tflops = {
            "A6000": 310e12, "A100": 312e12, "H100": 990e12,
            "A40": 150e12, "RTX 4090": 330e12, "L40S": 362e12,
        }
        gpu_peak = next((v for k, v in peak_tflops.items() if k in gpu_name), 310e12)
        mfu = 100 * flops_per_sec / gpu_peak
        print(f"\nGPU: {gpu_name}")
        print(f"FLOPs/step: {flops_per_step:.2e}")
        print(f"MFU: {mfu:.1f}%")

    # Memory
    if is_cuda:
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
        reserved_gb = torch.cuda.memory_reserved() / 1e9
        print(f"\nPeak GPU memory allocated: {peak_mem_gb:.2f} GB")
        print(f"GPU memory reserved:      {reserved_gb:.2f} GB")

    # ─── MoE-specific breakdown via manual instrumentation ───
    # Run a few more iterations with hooks to time MoE internals
    print(f"\n{'='*70}")
    print(f"MoE COMPONENT BREAKDOWN (separate instrumented pass)")
    print(f"{'='*70}\n")

    profile_moe_breakdown(model, x, device, is_cuda, num_runs=5)

    return fwd_stats, bwd_stats


def profile_moe_breakdown(model, x, device, is_cuda, num_runs=5):
    """Profile MoE sub-components by timing individual operations.

    We access the unwrapped model's MoE layers directly and time
    gate, dispatch, and scatter operations separately.
    """
    # Get the raw model (unwrap compile if needed)
    raw_model = model
    if hasattr(model, "_orig_mod"):
        raw_model = model._orig_mod

    # Find MoE layers
    from nanoseek.nanoseek.model import MoE
    moe_layers = []
    for i, layer in enumerate(raw_model.layers):
        if isinstance(layer.ffn, MoE):
            moe_layers.append((i, layer.ffn))

    if not moe_layers:
        print("No MoE layers found!")
        return

    print(f"Found {len(moe_layers)} MoE layers (layers {[i for i,_ in moe_layers]})")

    # Profile a single MoE layer in detail
    layer_idx, moe = moe_layers[len(moe_layers) // 2]  # pick middle layer
    print(f"Profiling MoE layer {layer_idx} in detail...\n")

    # Create a representative hidden state input
    B, S = x.shape
    D = raw_model.config.hidden_size
    hidden = torch.randn(B, S, D, device=device, dtype=torch.bfloat16 if is_cuda else torch.float32)

    gate_times = []
    dispatch_times = []
    scatter_times = []
    shared_times = []
    total_times = []

    for run in range(num_runs + 2):  # 2 warmup
        t_total_start = time.perf_counter()

        # ── Gate ──
        x_flat = hidden.view(-1, D)
        N = x_flat.shape[0]
        K = moe.num_experts_per_tok
        E = moe.n_routed_experts

        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        weights, indices, aux_loss, metadata = moe.gate(x_flat)

        if is_cuda:
            torch.cuda.synchronize()
        t_gate = time.perf_counter() - t0

        # ── Dispatch (sort + expert loop) ──
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        flat_indices = indices.view(-1)
        flat_weights = weights.view(-1)
        x_expanded = x_flat.unsqueeze(1).expand(-1, K, -1).reshape(N * K, D)

        sort_order = flat_indices.argsort(stable=True)
        sorted_indices = flat_indices[sort_order]
        sorted_x = x_expanded[sort_order]
        sorted_weights = flat_weights[sort_order]

        expert_counts = torch.bincount(sorted_indices, minlength=E)
        sorted_output = torch.zeros_like(sorted_x)
        counts_cpu = expert_counts.tolist()
        offset = 0
        for expert_idx in range(E):
            cnt = counts_cpu[expert_idx]
            if cnt == 0:
                continue
            expert_input = sorted_x[offset:offset + cnt]
            sorted_output[offset:offset + cnt] = moe.routed_experts[expert_idx](expert_input)
            offset += cnt

        if is_cuda:
            torch.cuda.synchronize()
        t_dispatch = time.perf_counter() - t0

        # ── Scatter combine ──
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        sorted_output = sorted_output * sorted_weights.unsqueeze(-1)
        orig_token_idx = torch.arange(N, device=x_flat.device).unsqueeze(1).expand(-1, K).reshape(N * K)
        orig_token_idx = orig_token_idx[sort_order]
        routed_output = torch.zeros_like(x_flat)
        routed_output.scatter_add_(0, orig_token_idx.unsqueeze(-1).expand_as(sorted_output), sorted_output)

        if is_cuda:
            torch.cuda.synchronize()
        t_scatter = time.perf_counter() - t0

        # ── Shared expert ──
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        if not moe.disable_shared_experts:
            shared_output = moe.shared_expert(x_flat)
            output = routed_output + shared_output
        else:
            output = routed_output

        if is_cuda:
            torch.cuda.synchronize()
        t_shared = time.perf_counter() - t0

        t_total = time.perf_counter() - t_total_start

        if run >= 2:  # skip warmup
            gate_times.append(t_gate * 1000)
            dispatch_times.append(t_dispatch * 1000)
            scatter_times.append(t_scatter * 1000)
            shared_times.append(t_shared * 1000)
            total_times.append(t_total * 1000)

    # Report
    def mean(lst):
        return sum(lst) / len(lst)

    total_mean = mean(total_times)

    components = [
        ("Gate routing", gate_times),
        ("Expert dispatch (sort+loop)", dispatch_times),
        ("Weight + scatter combine", scatter_times),
        ("Shared expert", shared_times),
        ("MoE total", total_times),
    ]

    for name, times in components:
        m = mean(times)
        pct = 100 * m / total_mean if total_mean > 0 else 0
        print(f"  {name:30s}  {m:8.3f} ms  ({pct:5.1f}%)")

    # Expert load balance
    counts_tensor = torch.tensor(counts_cpu, dtype=torch.float32)
    avg_count = counts_tensor.mean().item()
    max_count = counts_tensor.max().item()
    min_count = counts_tensor.min().item()
    waste = (max_count - avg_count) / max(avg_count, 1) * 100
    print(f"\n  Expert load: min={min_count:.0f}, avg={avg_count:.0f}, max={max_count:.0f}")
    print(f"  Padding waste if batched: {waste:.1f}%")
    print(f"  Kernel launches (current): {E} experts × 3 matmuls = {E*3}")
    print(f"  Kernel launches (batched): 2 bmm calls")

    # Memory estimate for bmm
    max_count_int = int(max_count)
    padded_mb = E * max_count_int * D * 2 / 1e6  # bf16
    stack_mb = (E * moe.routed_experts[0].w_gate.weight.shape[0] *
                moe.routed_experts[0].w_gate.weight.shape[1] * 4 * 3) / 1e6  # fp32 × 3 weights
    print(f"\n  bmm extra memory estimate:")
    print(f"    Stacked weights (fp32): {stack_mb:.0f} MB")
    print(f"    Padded input (bf16):    {padded_mb:.0f} MB")
    print(f"    Total extra:            {stack_mb + padded_mb:.0f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile MoE dispatch performance")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--scale", type=str, default="ablation", choices=["ablation", "1b"])
    parser.add_argument("--compile", action="store_true", help="Apply torch.compile")
    parser.add_argument("--runs", type=int, default=15, help="Number of timed runs")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup runs")
    args = parser.parse_args()

    if args.scale == "ablation":
        config = get_nanoseek_ablation_config()
    else:
        config = get_nanoseek_config()

    device = torch.device(args.device)
    profile_model(config, device, num_warmup=args.warmup, num_runs=args.runs,
                  use_compile=args.compile)
