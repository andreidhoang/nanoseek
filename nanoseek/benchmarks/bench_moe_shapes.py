"""Isolated MoE kernel benchmarks at production tensor shapes.

This is how a senior perf engineer profiles MoE dispatch:
  1. Create tensors at the EXACT shapes the kernel sees in production
  2. Time the kernel directly — no model, no autograd, no overhead
  3. Sweep across shapes (ablation × 1b) to see scaling behavior
  4. Report roofline metrics (FLOPS, bandwidth, arithmetic intensity)

No model instantiation needed. Runs in seconds even on CPU.

Usage:
    python -m nanoseek.benchmarks.bench_moe_shapes
    python -m nanoseek.benchmarks.bench_moe_shapes --device cuda
    python -m nanoseek.benchmarks.bench_moe_shapes --device cuda --sweep
"""

import argparse
import time

import torch
import torch.nn.functional as F

from nanoseek.benchmarks._utils import SHAPE_CONFIGS, Timer, stats


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _bench(fn, device, warmup=5, runs=20):
    """Time a function, return list of elapsed_ms."""
    for _ in range(warmup):
        fn()
    _sync(device)

    times = []
    for _ in range(runs):
        t = Timer(device)
        _sync(device)
        t.start()
        fn()
        t.stop()
        times.append(t.elapsed_ms())
    return times


# ─── Individual kernel benchmarks ──────────────────────────────────────────

def bench_gate_routing(shapes, device, dtype, runs=20):
    """Gate: linear projection + sigmoid + group selection + top-k.

    This is the router — must be fast and precise (never FP8).
    Shapes: [N, D] → proj [D, E] → scores [N, E] → topk → [N, K]
    """
    N, D, E, K = shapes["N"], shapes["D"], shapes["E"], shapes["K"]
    n_group, topk_group = shapes["n_group"], shapes["topk_group"]

    x = torch.randn(N, D, device=device, dtype=dtype)
    W = torch.randn(E, D, device=device, dtype=dtype)  # router weight

    def gate_fn():
        # Linear + sigmoid
        logits = F.linear(x, W)          # [N, E]
        scores = logits.float().sigmoid()  # [N, E] in fp32

        # Group selection (simplified — real gate does top-2 within groups)
        scores_grouped = scores.view(N, n_group, -1)         # [N, G, E/G]
        group_scores = scores_grouped.topk(2, dim=-1).values.sum(dim=-1)  # [N, G]
        top_groups = group_scores.topk(topk_group, dim=-1).indices        # [N, topk_group]

        # Top-K selection
        topk_weights, topk_indices = scores.topk(K, dim=-1)  # [N, K]
        return topk_weights, topk_indices

    times = _bench(gate_fn, device, runs=runs)
    return times


def bench_expert_dispatch_sort(shapes, device, dtype, runs=20):
    """Dispatch: flatten + argsort + expand + bincount.

    This is pure overhead — the grouped GEMM savings must exceed this cost.
    """
    N, D, E, K = shapes["N"], shapes["D"], shapes["E"], shapes["K"]

    indices = torch.randint(0, E, (N, K), device=device)
    weights = torch.rand(N, K, device=device, dtype=dtype)
    x = torch.randn(N, D, device=device, dtype=dtype)

    def dispatch_fn():
        flat_indices = indices.view(-1)
        flat_weights = weights.view(-1)
        x_expanded = x.unsqueeze(1).expand(-1, K, -1).reshape(N * K, D)

        sort_order = flat_indices.argsort(stable=True)
        sorted_indices = flat_indices[sort_order]
        sorted_x = x_expanded[sort_order]
        sorted_weights = flat_weights[sort_order]

        expert_counts = torch.bincount(sorted_indices, minlength=E)
        expert_boundaries = torch.zeros(E + 1, dtype=torch.long, device=device)
        expert_boundaries[1:] = expert_counts.cumsum(0)
        return sorted_x, sorted_indices, sorted_weights, expert_counts, expert_boundaries

    times = _bench(dispatch_fn, device, runs=runs)
    return times


def bench_expert_bmm(shapes, device, dtype, runs=20):
    """Expert compute via batched matmul (the production GPU path).

    This is the single biggest compute cost: 2 bmm calls replacing 192 GEMMs.
    Shapes: bmm([E, max_count, D], [E, D, 2*inter]) and bmm([E, max_count, inter], [E, inter, D])
    """
    E, D = shapes["E"], shapes["D"]
    inter = shapes["moe_inter"]
    tpe = shapes["tokens_per_expert"]  # avg tokens per expert

    # Simulate balanced routing (max_count ≈ avg_count)
    max_count = int(tpe * 1.1)  # 10% padding waste (realistic)

    padded_input = torch.randn(E, max_count, D, device=device, dtype=dtype)
    w_gate_up = torch.randn(E, 2 * inter, D, device=device, dtype=dtype)
    w_down = torch.randn(E, D, inter, device=device, dtype=dtype)

    def bmm_fn():
        # Fused gate+up projection
        gate_up = torch.bmm(padded_input, w_gate_up.transpose(1, 2))  # [E, max, 2*inter]
        gate_out, up_out = gate_up.chunk(2, dim=-1)
        h = F.silu(gate_out) * up_out                                 # [E, max, inter]
        out = torch.bmm(h, w_down.transpose(1, 2))                    # [E, max, D]
        return out

    times = _bench(bmm_fn, device, runs=runs)

    # Compute roofline metrics
    m = stats(times)["mean"]
    # FLOPs: 2 bmm calls
    flops_gate_up = 2 * E * max_count * D * (2 * inter)  # bmm1
    flops_down = 2 * E * max_count * inter * D            # bmm2
    flops_silu = 5 * E * max_count * inter                # silu + mul (elementwise)
    total_flops = flops_gate_up + flops_down + flops_silu

    # Bytes: inputs + weights + outputs (bf16 = 2 bytes)
    bytes_moved = (E * max_count * D + E * 2 * inter * D +        # bmm1 inputs
                   E * max_count * inter +                          # silu input
                   E * max_count * inter + E * inter * D +          # bmm2 inputs
                   E * max_count * D) * 2                           # output, all bf16

    return times, {
        "flops": total_flops,
        "bytes": bytes_moved,
        "ai": total_flops / bytes_moved if bytes_moved > 0 else 0,
        "tflops": total_flops / (m / 1000) / 1e12 if m > 0 else 0,
        "bw_gbs": bytes_moved / (m / 1000) / 1e9 if m > 0 else 0,
    }


def bench_expert_sequential(shapes, device, dtype, runs=20):
    """Expert compute via sequential loop (the CPU / fallback path).

    64 experts × 3 matmuls = 192 kernel launches.
    """
    E, D = shapes["E"], shapes["D"]
    inter = shapes["moe_inter"]
    tpe = shapes["tokens_per_expert"]

    # Per-expert weights (separate tensors, like nn.ModuleList)
    w_gates = [torch.randn(inter, D, device=device, dtype=dtype) for _ in range(E)]
    w_ups = [torch.randn(inter, D, device=device, dtype=dtype) for _ in range(E)]
    w_downs = [torch.randn(D, inter, device=device, dtype=dtype) for _ in range(E)]

    # Simulate sorted tokens with variable counts
    counts = [tpe] * E  # balanced for fair comparison
    x_per_expert = [torch.randn(c, D, device=device, dtype=dtype) for c in counts]

    def sequential_fn():
        outputs = []
        for i in range(E):
            xi = x_per_expert[i]
            gate = F.linear(xi, w_gates[i])   # [tpe, inter]
            up = F.linear(xi, w_ups[i])       # [tpe, inter]
            h = F.silu(gate) * up             # [tpe, inter]
            out = F.linear(h, w_downs[i])     # [tpe, D]
            outputs.append(out)
        return outputs

    times = _bench(sequential_fn, device, runs=runs)
    return times


def bench_scatter_combine(shapes, device, dtype, runs=20):
    """Scatter combine: weight application + scatter_add back to token order.

    Must be fast — this is pure memory bandwidth.
    """
    N, D, E, K = shapes["N"], shapes["D"], shapes["E"], shapes["K"]
    NK = N * K

    sorted_output = torch.randn(NK, D, device=device, dtype=dtype)
    sorted_weights = torch.rand(NK, device=device, dtype=dtype)
    orig_token_idx = torch.randint(0, N, (NK,), device=device)

    def combine_fn():
        weighted = sorted_output * sorted_weights.unsqueeze(-1)
        result = torch.zeros(N, D, device=device, dtype=dtype)
        result.scatter_add_(0, orig_token_idx.unsqueeze(-1).expand_as(weighted), weighted)
        return result

    times = _bench(combine_fn, device, runs=runs)
    return times


def bench_shared_expert(shapes, device, dtype, runs=20):
    """Shared expert: single SwiGLU FFN on ALL tokens."""
    N, D = shapes["N"], shapes["D"]
    inter = shapes["shared_inter"]

    x = torch.randn(N, D, device=device, dtype=dtype)
    w_gate = torch.randn(inter, D, device=device, dtype=dtype)
    w_up = torch.randn(inter, D, device=device, dtype=dtype)
    w_down = torch.randn(D, inter, device=device, dtype=dtype)

    def shared_fn():
        gate = F.linear(x, w_gate)
        up = F.linear(x, w_up)
        h = F.silu(gate) * up
        return F.linear(h, w_down)

    times = _bench(shared_fn, device, runs=runs)
    return times


# ─── Main ───────────────────────────────────────────────────────────────────

def run_single_scale(scale_name, shapes, device, dtype, runs):
    """Run all MoE kernel benchmarks for one shape config."""
    N = shapes["N"]
    E = shapes["E"]
    D = shapes["D"]
    inter = shapes["moe_inter"]
    tpe = shapes["tokens_per_expert"]

    print(f"\n{'='*72}")
    print(f"  MoE KERNEL BENCHMARKS — {scale_name.upper()}")
    print(f"  D={D}, E={E}, K={shapes['K']}, inter={inter}")
    print(f"  N={N} tokens (B={shapes['B']}×S={shapes['S']}), "
          f"NK={shapes['NK']} pairs, {tpe} tokens/expert avg")
    print(f"  Device: {device}, Dtype: {dtype}")
    print(f"{'='*72}\n")

    results = {}

    # Gate routing
    t = bench_gate_routing(shapes, device, dtype, runs)
    s = stats(t)
    results["gate"] = s["mean"]
    print(f"  Gate routing:          {s['mean']:8.3f} +/- {s['std']:.3f} ms")

    # Dispatch sort
    t = bench_expert_dispatch_sort(shapes, device, dtype, runs)
    s = stats(t)
    results["dispatch"] = s["mean"]
    print(f"  Dispatch (sort+expand): {s['mean']:8.3f} +/- {s['std']:.3f} ms")

    # Expert BMM (GPU production path)
    t, metrics = bench_expert_bmm(shapes, device, dtype, runs)
    s = stats(t)
    results["expert_bmm"] = s["mean"]
    print(f"  Expert BMM (batched):  {s['mean']:8.3f} +/- {s['std']:.3f} ms")
    print(f"    {metrics['tflops']:.1f} TFLOPS, {metrics['bw_gbs']:.0f} GB/s, "
          f"AI={metrics['ai']:.0f} FLOP/Byte")

    # Expert sequential (CPU / fallback path)
    t = bench_expert_sequential(shapes, device, dtype, runs)
    s = stats(t)
    results["expert_seq"] = s["mean"]
    print(f"  Expert sequential:     {s['mean']:8.3f} +/- {s['std']:.3f} ms")

    # BMM speedup
    if results["expert_seq"] > 0:
        speedup = results["expert_seq"] / results["expert_bmm"]
        print(f"    BMM speedup vs sequential: {speedup:.1f}x")

    # Scatter combine
    t = bench_scatter_combine(shapes, device, dtype, runs)
    s = stats(t)
    results["combine"] = s["mean"]
    print(f"  Scatter combine:       {s['mean']:8.3f} +/- {s['std']:.3f} ms")

    # Shared expert
    t = bench_shared_expert(shapes, device, dtype, runs)
    s = stats(t)
    results["shared"] = s["mean"]
    print(f"  Shared expert:         {s['mean']:8.3f} +/- {s['std']:.3f} ms")

    # Total and breakdown
    total = sum(results.values()) - results["expert_seq"]  # use bmm path
    print(f"\n  {'Component':<25s} {'ms':>8s} {'%':>6s}")
    print(f"  {'-'*41}")
    for name, ms in results.items():
        if name == "expert_seq":
            continue  # skip sequential (it's the alternative, not additive)
        pct = 100 * ms / total if total > 0 else 0
        print(f"  {name:<25s} {ms:8.3f} {pct:5.1f}%")
    print(f"  {'-'*41}")
    print(f"  {'TOTAL (bmm path)':<25s} {total:8.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Isolated MoE kernel benchmarks at production tensor shapes")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--scale", type=str, default=None, choices=["ablation", "1b"],
                        help="Run single scale (default: both)")
    parser.add_argument("--sweep", action="store_true",
                        help="Also sweep batch sizes and waste ratios")
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    scales = [args.scale] if args.scale else ["ablation", "1b"]

    all_results = {}
    for scale in scales:
        shapes = SHAPE_CONFIGS[scale]
        results = run_single_scale(scale, shapes, device, dtype, args.runs)
        all_results[scale] = results

    # Cross-scale comparison
    if len(all_results) > 1:
        print(f"\n{'='*72}")
        print(f"  CROSS-SCALE COMPARISON")
        print(f"{'='*72}\n")
        print(f"  {'Kernel':<25s} ", end="")
        for s in scales:
            print(f"{'  ' + s + ' (ms)':>14s}", end="")
        if len(scales) == 2:
            print(f"  {'ratio':>8s}", end="")
        print()
        print(f"  {'-'*60}")

        for kernel in ["gate", "dispatch", "expert_bmm", "combine", "shared"]:
            print(f"  {kernel:<25s} ", end="")
            vals = []
            for s in scales:
                v = all_results[s].get(kernel, 0)
                vals.append(v)
                print(f"  {v:12.3f}", end="")
            if len(vals) == 2 and vals[0] > 0:
                print(f"  {vals[1]/vals[0]:7.2f}x", end="")
            print()

    # Waste ratio sweep
    if args.sweep and device.type == "cuda":
        print(f"\n{'='*72}")
        print(f"  WASTE RATIO SWEEP (ablation shapes, varying load imbalance)")
        print(f"{'='*72}\n")

        shapes = SHAPE_CONFIGS["ablation"]
        E, D, inter = shapes["E"], shapes["D"], shapes["moe_inter"]
        tpe = shapes["tokens_per_expert"]

        print(f"  {'waste_ratio':>12s} {'max_count':>10s} {'bmm (ms)':>10s} {'pad MB':>8s}")
        print(f"  {'-'*44}")

        for waste in [1.0, 1.1, 1.2, 1.5, 2.0, 3.0]:
            max_count = int(tpe * waste)
            padded = torch.randn(E, max_count, D, device=device, dtype=dtype)
            w = torch.randn(E, 2 * inter, D, device=device, dtype=dtype)

            t = _bench(lambda: torch.bmm(padded, w.transpose(1, 2)), device, runs=10)
            pad_mb = E * max_count * D * 2 / 1e6

            print(f"  {waste:12.1f} {max_count:10d} {stats(t)['mean']:10.3f} {pad_mb:7.0f}")


if __name__ == "__main__":
    main()
