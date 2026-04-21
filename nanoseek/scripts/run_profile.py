#!/usr/bin/env python3
"""NanoSeek profiling toolkit — one-command profiling for training steps.

Three modes:
  1. torch-profile: Profile N forward+backward steps with torch.profiler
     → Chrome trace + top-K kernel summary + memory timeline
  2. nsys-wrap: Generate and print the nsys command (user runs it)
  3. memory-snapshot: Capture peak memory breakdown by layer

Usage:
    # Quick profile: 3 steps, print top 20 kernels
    python -m nanoseek.scripts.profile --scale ablation --bs 4 --seq 4096

    # Full profile with chrome trace export
    python -m nanoseek.scripts.profile --scale ablation --bs 4 --seq 4096 --trace

    # Memory breakdown by phase
    python -m nanoseek.scripts.profile --scale ablation --bs 4 --seq 4096 --memory

    # Generate nsys command (copy-paste to run)
    python -m nanoseek.scripts.profile --nsys-cmd --scale ablation

    # Compare two runs (A/B)
    python -m nanoseek.scripts.profile --compare run_a.json run_b.json
"""

import os
import json
import time
import argparse

import torch
import torch.nn.functional as F


def profile_training_steps(scale, bs, seq, n_steps=3, export_trace=False, output_dir="runs/profiles"):
    """Profile N training steps with torch.profiler.

    Returns structured summary: top kernels, memory, MFU estimate.
    """
    if not torch.cuda.is_available():
        print("CUDA not available — profiling requires GPU")
        return None

    from torch.profiler import profile, ProfilerActivity

    from nanoseek.nanoseek.config import get_config
    from nanoseek.nanoseek.model import NanoSeekModel

    config = get_config(scale)
    device = torch.device("cuda")

    print(f"\n{'='*60}")
    print(f"  NanoSeek Profiler — {scale} | B={bs} S={seq}")
    print(f"{'='*60}\n")

    # Build model
    print("Building model...")
    with torch.device("meta"):
        model = NanoSeekModel(config)
    model.to_empty(device=device)
    model.init_weights()
    model.train()

    params = model.num_parameters()
    n_active = params['active']
    print(f"  Active: {n_active:,}  Total: {params['total']:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ids = torch.randint(0, config.vocab_size, (bs, seq), device=device)

    # Warmup (outside profiler)
    print(f"Warming up ({max(3, n_steps)} steps)...")
    for _ in range(max(3, n_steps)):
        with torch.autocast("cuda", torch.bfloat16):
            outputs = model(ids, labels=ids, mtp_lambda=0.1)
        outputs['loss'].backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    # Profile
    print(f"Profiling {n_steps} steps...\n")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        step_times = []
        for i in range(n_steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.autocast("cuda", torch.bfloat16):
                outputs = model(ids, labels=ids, mtp_lambda=0.1)
            outputs['loss'].backward()
            optimizer.step()
            optimizer.zero_grad()

            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)

    # ─── Results ───
    med_time = sorted(step_times)[len(step_times) // 2]
    tok_per_sec = bs * seq / med_time
    flops_per_sec = 6 * n_active * bs * seq / med_time
    peak_flops = 989e12  # H100 SXM BF16
    gpu_name = torch.cuda.get_device_name(device)
    if "A100" in gpu_name:
        peak_flops = 312e12
    elif "4090" in gpu_name:
        peak_flops = 165e12
    mfu = 100 * flops_per_sec / peak_flops

    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
    reserved_mem = torch.cuda.memory_reserved(device) / 1e9

    # Print kernel summary
    print("─" * 80)
    print(f"  TOP 20 CUDA KERNELS (by GPU time)")
    print("─" * 80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # Print operator summary grouped by input shape
    print("\n" + "─" * 80)
    print(f"  TOP 10 OPERATORS (by GPU time, grouped)")
    print("─" * 80)
    print(prof.key_averages(group_by_input_shape=True).table(
        sort_by="cuda_time_total", row_limit=10
    ))

    # Compute CUDA idle time (gaps between kernels)
    events = prof.key_averages()
    total_cuda_time_us = sum(e.cuda_time_total for e in events if e.cuda_time_total > 0)
    total_cpu_time_us = sum(e.cpu_time_total for e in events if e.cpu_time_total > 0)

    # Summary
    print("\n" + "=" * 60)
    print(f"  PROFILING SUMMARY")
    print("=" * 60)
    print(f"  GPU:              {gpu_name}")
    print(f"  Scale:            {scale} (active={n_active:,})")
    print(f"  Batch:            B={bs} S={seq} ({bs*seq:,} tokens)")
    print(f"  Steps profiled:   {n_steps}")
    print(f"  Median step time: {med_time*1000:.1f} ms")
    print(f"  Throughput:       {tok_per_sec:,.0f} tok/s")
    print(f"  MFU:              {mfu:.1f}%")
    print(f"  Peak memory:      {peak_mem:.2f} GB allocated / {reserved_mem:.2f} GB reserved")
    print(f"  Fragmentation:    {reserved_mem - peak_mem:.2f} GB")
    print(f"  Total CUDA time:  {total_cuda_time_us/1000:.1f} ms")
    print(f"  Total CPU time:   {total_cpu_time_us/1000:.1f} ms")
    print()

    # Export chrome trace
    if export_trace:
        os.makedirs(output_dir, exist_ok=True)
        trace_path = os.path.join(output_dir, f"profile_{scale}_b{bs}_s{seq}.json")
        prof.export_chrome_trace(trace_path)
        print(f"  Chrome trace: {trace_path}")
        print(f"  Open in: chrome://tracing or ui.perfetto.dev")

    # Export structured JSON summary
    summary = {
        "gpu": gpu_name,
        "scale": scale,
        "batch_size": bs,
        "seq_len": seq,
        "n_active_params": n_active,
        "median_step_ms": round(med_time * 1000, 1),
        "tok_per_sec": round(tok_per_sec),
        "mfu_pct": round(mfu, 1),
        "peak_mem_gb": round(peak_mem, 2),
        "reserved_mem_gb": round(reserved_mem, 2),
        "step_times_ms": [round(t * 1000, 1) for t in step_times],
        "top_kernels": [
            {
                "name": e.key,
                "cuda_time_ms": round(e.cuda_time_total / 1000, 2),
                "cpu_time_ms": round(e.cpu_time_total / 1000, 2),
                "calls": e.count,
                "flops": e.flops,
            }
            for e in sorted(events, key=lambda e: e.cuda_time_total, reverse=True)[:20]
            if e.cuda_time_total > 0
        ],
    }

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"summary_{scale}_b{bs}_s{seq}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  JSON summary: {json_path}\n")

    return summary


def profile_memory_phases(scale, bs, seq):
    """Profile memory allocation at each phase of a training step.

    Reports: after model init, after forward, after backward, after optimizer.
    """
    if not torch.cuda.is_available():
        print("CUDA not available — memory profiling requires GPU")
        return

    from nanoseek.nanoseek.config import get_config
    from nanoseek.nanoseek.model import NanoSeekModel

    config = get_config(scale)
    device = torch.device("cuda")

    print(f"\n{'='*60}")
    print(f"  Memory Phase Profiler — {scale} | B={bs} S={seq}")
    print(f"{'='*60}\n")

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    # Phase 0: Model init
    with torch.device("meta"):
        model = NanoSeekModel(config)
    model.to_empty(device=device)
    model.init_weights()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    mem_model = torch.cuda.memory_allocated(device) / 1e9
    print(f"  After model init:     {mem_model:.2f} GB")

    ids = torch.randint(0, config.vocab_size, (bs, seq), device=device)

    # Warmup to trigger optimizer state allocation
    with torch.autocast("cuda", torch.bfloat16):
        outputs = model(ids, labels=ids, mtp_lambda=0.1)
    outputs['loss'].backward()
    optimizer.step()
    optimizer.zero_grad()

    # Now measure each phase cleanly
    torch.cuda.reset_peak_memory_stats(device)
    mem_baseline = torch.cuda.memory_allocated(device) / 1e9
    print(f"  Baseline (warm):      {mem_baseline:.2f} GB")

    # Forward
    with torch.autocast("cuda", torch.bfloat16):
        outputs = model(ids, labels=ids, mtp_lambda=0.1)
    mem_after_fwd = torch.cuda.memory_allocated(device) / 1e9
    peak_after_fwd = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"  After forward:        {mem_after_fwd:.2f} GB  (peak: {peak_after_fwd:.2f} GB)  [+{mem_after_fwd - mem_baseline:.2f} activations]")

    # Backward
    outputs['loss'].backward()
    mem_after_bwd = torch.cuda.memory_allocated(device) / 1e9
    peak_after_bwd = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"  After backward:       {mem_after_bwd:.2f} GB  (peak: {peak_after_bwd:.2f} GB)  [+{mem_after_bwd - mem_after_fwd:.2f} gradients]")

    # Optimizer
    optimizer.step()
    optimizer.zero_grad()
    mem_after_optim = torch.cuda.memory_allocated(device) / 1e9
    peak_after_optim = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"  After optimizer:      {mem_after_optim:.2f} GB  (peak: {peak_after_optim:.2f} GB)")

    reserved = torch.cuda.memory_reserved(device) / 1e9
    print(f"\n  Reserved:             {reserved:.2f} GB")
    print(f"  Fragmentation:        {reserved - mem_after_optim:.2f} GB")
    print(f"  Overall peak:         {peak_after_optim:.2f} GB")

    # Headroom estimate
    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem = 80.0  # default H100
    if "A100" in gpu_name:
        gpu_mem = 80.0
    elif "4090" in gpu_name:
        gpu_mem = 24.0
    elif "3090" in gpu_name:
        gpu_mem = 24.0
    print(f"  Headroom:             {gpu_mem - reserved:.1f} GB (of {gpu_mem:.0f} GB {gpu_name})")

    # Memory breakdown
    print(f"\n  {'Component':<25s}  {'GB':>8s}  {'% of peak':>10s}")
    print(f"  {'─'*25}  {'─'*8}  {'─'*10}")

    components = [
        ("Model params", mem_model),
        ("Optimizer states", mem_baseline - mem_model),
        ("Activations (fwd)", mem_after_fwd - mem_baseline),
        ("Gradients (bwd)", mem_after_bwd - mem_after_fwd),
        ("Optimizer overhead", max(0, peak_after_optim - mem_after_bwd)),
    ]
    total = peak_after_optim
    for name, gb in components:
        pct = 100 * gb / total if total > 0 else 0
        print(f"  {name:<25s}  {gb:8.2f}  {pct:9.1f}%")
    print()


def generate_nsys_command(scale, bs, seq, n_steps=25, output_name="nanoseek_profile"):
    """Generate the nsys profiling command for copy-paste execution."""
    print(f"\n{'='*60}")
    print(f"  Nsight Systems Command Generator")
    print(f"{'='*60}\n")

    cmd = f"""nsys profile \\
    --trace=cuda,nvtx,cublas \\
    --cuda-memory-usage=true \\
    --gpu-metrics-device=all \\
    --stats=true \\
    --force-overwrite=true \\
    --output={output_name} \\
    python -m nanoseek.scripts.pre_train \\
        --run profile --scale {scale} --seed 42 \\
        --num-iterations {n_steps} --eval-every -1 --save-every -1 \\
        --device-batch-size {bs} --no-compile"""

    print(f"  Run this command:\n")
    print(f"  {cmd}\n")
    print(f"  Then analyze with:")
    print(f"    nsys stats {output_name}.nsys-rep")
    print(f"    nsys stats --report cuda_gpu_kern_sum {output_name}.nsys-rep")
    print()


def generate_ncu_command(scale, bs, seq, kernel_filter=None):
    """Generate Nsight Compute command for kernel-level hardware profiling.

    Nsight Compute is the DEEPEST profiling level:
    - Chrome trace (torch.profiler): WHAT is slow (PyTorch ops)
    - Nsight Systems (nsys):         WHY is it slow (kernel gaps, idle GPU)
    - Nsight Compute (ncu):          HOW to fix it (occupancy, memory stalls, tensor cores)

    ncu replays each kernel ~100x to collect hardware counters, so it's very slow.
    Always filter to specific kernels — never profile an entire training step.
    """
    print(f"\n{'='*60}")
    print(f"  Nsight Compute Command Generator")
    print(f"{'='*60}\n")

    # Default: profile GEMM kernels (the ones that matter for MoE experts + MLA)
    if kernel_filter is None:
        kernel_filter = "gemm|cutlass"

    # Step 1: Identify target kernels with nsys first
    print("  STEP 1: Find the kernel to profile (run nsys first)")
    print("  ─────────────────────────────────────────────────────")
    print(f"    nsys stats --report cuda_gpu_kern_sum nanoseek_profile.nsys-rep")
    print(f"    # Pick the slowest kernel name from the output")
    print()

    # Step 2: Profile that specific kernel
    print("  STEP 2: Profile the target kernel with ncu")
    print("  ─────────────────────────────────────────────────────")

    cmd_roofline = f"""ncu \\
    --set roofline \\
    --kernel-name-base demangled \\
    --kernel-name regex:"{kernel_filter}" \\
    --launch-skip 10 --launch-count 3 \\
    --export ncu_roofline \\
    python -m nanoseek.scripts.benchmark --scale {scale} --bs {bs} --seq {seq}"""

    cmd_full = f"""ncu \\
    --set full \\
    --kernel-name-base demangled \\
    --kernel-name regex:"{kernel_filter}" \\
    --launch-skip 10 --launch-count 1 \\
    --export ncu_full \\
    python -m nanoseek.scripts.benchmark --scale {scale} --bs {bs} --seq {seq}"""

    print(f"\n  Roofline analysis (fast, answers: compute-bound or memory-bound?):\n")
    print(f"    {cmd_roofline}\n")
    print(f"  Full analysis (slow, answers: occupancy, warp stalls, tensor cores):\n")
    print(f"    {cmd_full}\n")

    # Step 3: Read results
    print("  STEP 3: Analyze results")
    print("  ─────────────────────────────────────────────────────")
    print("    ncu --import ncu_roofline.ncu-rep    # GUI (if available)")
    print("    ncu --import ncu_full.ncu-rep --csv   # CLI summary")
    print()

    # Decision guide
    print("  DECISION GUIDE: What ncu results mean for NanoSeek")
    print("  ─────────────────────────────────────────────────────")
    print("  ┌─────────────────────────────────┬────────────────────────────────────┐")
    print("  │ ncu finding                     │ Action                             │")
    print("  ├─────────────────────────────────┼────────────────────────────────────┤")
    print("  │ Low occupancy (<50%)            │ Reduce registers (--maxrregcount)  │")
    print("  │                                 │ or increase block size             │")
    print("  │ Memory-bound (below roofline)   │ Fuse kernels, reduce dispatch      │")
    print("  │                                 │ overhead, use FP8 to halve bytes   │")
    print("  │ Compute-bound (at roofline)     │ Already optimal for this kernel —  │")
    print("  │                                 │ reduce FLOP count or use FP8       │")
    print("  │ No tensor core usage            │ Check dtype (must be bf16/fp16)    │")
    print("  │                                 │ and shapes (must be multiples of 8)│")
    print("  │ L2 cache thrashing              │ Reduce working set or tile better  │")
    print("  │ Warp stall: memory dependency   │ Prefetch, increase parallelism     │")
    print("  └─────────────────────────────────┴────────────────────────────────────┘")
    print()


def compare_profiles(path_a, path_b):
    """Compare two JSON profile summaries (A/B test)."""
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)

    print(f"\n{'='*60}")
    print(f"  Profile Comparison: A vs B")
    print(f"{'='*60}\n")

    print(f"  {'Metric':<25s}  {'A':>12s}  {'B':>12s}  {'Delta':>10s}  {'Change':>8s}")
    print(f"  {'─'*25}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*8}")

    metrics = [
        ("Step time (ms)", "median_step_ms", False),   # lower is better
        ("Throughput (tok/s)", "tok_per_sec", True),    # higher is better
        ("MFU (%)", "mfu_pct", True),                   # higher is better
        ("Peak memory (GB)", "peak_mem_gb", False),     # lower is better
    ]

    for label, key, higher_is_better in metrics:
        va, vb = a.get(key, 0), b.get(key, 0)
        delta = vb - va
        if va > 0:
            pct = 100 * delta / va
            direction = "+" if delta > 0 else ""
            # Color hint: good vs bad
            if (higher_is_better and delta > 0) or (not higher_is_better and delta < 0):
                marker = " <<< BETTER"
            elif abs(pct) < 2:
                marker = ""
            else:
                marker = " !!! WORSE"
        else:
            pct = 0
            direction = ""
            marker = ""

        print(f"  {label:<25s}  {va:12.1f}  {vb:12.1f}  {direction}{delta:9.1f}  {direction}{pct:.1f}%{marker}")

    print(f"\n  A: {path_a}")
    print(f"  B: {path_b}")

    # Compare top kernels if both have them
    if "top_kernels" in a and "top_kernels" in b:
        print(f"\n  Top kernel changes (by CUDA time):")
        a_kernels = {k['name']: k['cuda_time_ms'] for k in a['top_kernels'][:10]}
        b_kernels = {k['name']: k['cuda_time_ms'] for k in b['top_kernels'][:10]}
        all_names = list(dict.fromkeys(list(a_kernels.keys()) + list(b_kernels.keys())))

        for name in all_names[:10]:
            ta = a_kernels.get(name, 0)
            tb = b_kernels.get(name, 0)
            delta = tb - ta
            if ta > 0:
                pct = 100 * delta / ta
                print(f"    {name[:50]:<50s}  {ta:8.2f} → {tb:8.2f} ms  ({pct:+.1f}%)")
            else:
                print(f"    {name[:50]:<50s}  {'N/A':>8s} → {tb:8.2f} ms  (new)")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NanoSeek profiling toolkit")
    parser.add_argument("--scale", default="ablation", choices=["anchor", "ablation", "1b"])
    parser.add_argument("--bs", type=int, default=4, help="Batch size")
    parser.add_argument("--seq", type=int, default=4096, help="Sequence length")
    parser.add_argument("--steps", type=int, default=3, help="Steps to profile")
    parser.add_argument("--trace", action="store_true", help="Export chrome trace")
    parser.add_argument("--memory", action="store_true", help="Memory phase breakdown")
    parser.add_argument("--nsys-cmd", action="store_true", help="Print nsys command")
    parser.add_argument("--ncu-cmd", action="store_true",
                        help="Print Nsight Compute command for kernel-level hardware profiling")
    parser.add_argument("--kernel-filter", type=str, default=None,
                        help="Regex filter for ncu kernel names (default: 'gemm|cutlass')")
    parser.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                        help="Compare two profile JSON summaries")
    parser.add_argument("--output-dir", default="runs/profiles", help="Output directory")
    parser.add_argument("--all", action="store_true", help="Run all profiling modes")
    args = parser.parse_args()

    if args.compare:
        compare_profiles(args.compare[0], args.compare[1])
    elif args.nsys_cmd or args.all:
        generate_nsys_command(args.scale, args.bs, args.seq)

    if args.ncu_cmd or args.all:
        generate_ncu_command(args.scale, args.bs, args.seq, args.kernel_filter)

    if args.memory or args.all:
        profile_memory_phases(args.scale, args.bs, args.seq)

    if not args.compare and not args.nsys_cmd and not args.ncu_cmd and not args.memory or args.all:
        profile_training_steps(
            args.scale, args.bs, args.seq,
            n_steps=args.steps,
            export_trace=args.trace or args.all,
            output_dir=args.output_dir,
        )
