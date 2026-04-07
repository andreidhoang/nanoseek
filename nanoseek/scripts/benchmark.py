#!/usr/bin/env python3
"""NanoSeek microbenchmarks — time individual components on a single GPU.

Usage:
    python -m nanoseek.scripts.benchmark --scale ablation --bs 4 --seq 4096
    python -m nanoseek.scripts.benchmark --scale 1b --bs 16 --seq 4096
    python -m nanoseek.scripts.benchmark --sdpa-check  # verify FlashAttention dispatch
"""

import torch
import torch.nn.functional as F
import time
import argparse


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
    n = len(times)
    med = times[n // 2]
    p10 = times[int(n * 0.1)]
    p90 = times[int(n * 0.9)]
    print(f"  {label:45s}  median={med:8.3f}ms  p10={p10:.3f}  p90={p90:.3f}ms")
    return med


def check_sdpa_backends():
    """Check which SDPA backends work with MLA's Q/K=192, V=128 dimensions."""
    if not torch.cuda.is_available():
        print("CUDA not available — skipping SDPA check")
        return

    device = torch.device("cuda")
    B, H, S = 2, 16, 512  # small seq for quick check
    q = torch.randn(B, H, S, 192, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, H, S, 192, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, H, S, 128, dtype=torch.bfloat16, device=device)

    print("\n=== SDPA Backend Check (MLA: Q/K=192, V=128) ===\n")

    backends = [
        ("flash",        dict(enable_flash=True,  enable_math=False, enable_mem_efficient=False)),
        ("mem_efficient", dict(enable_flash=False, enable_math=False, enable_mem_efficient=True)),
        ("math",         dict(enable_flash=False, enable_math=True,  enable_mem_efficient=False)),
    ]

    for name, flags in backends:
        try:
            with torch.backends.cuda.sdp_kernel(**flags):
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            print(f"  {name:15s}: WORKS  (output shape: {tuple(out.shape)})")
        except RuntimeError as e:
            err_msg = str(e)[:80]
            print(f"  {name:15s}: FAILED ({err_msg})")

    # Also test with equal dims (padded V) as fallback
    v_padded = F.pad(v, (0, 64))  # pad V from 128 to 192
    print(f"\n  With V padded to 192:")
    for name, flags in backends:
        try:
            with torch.backends.cuda.sdp_kernel(**flags):
                out = F.scaled_dot_product_attention(q, k, v_padded, is_causal=True)
            print(f"    {name:15s}: WORKS")
        except RuntimeError as e:
            err_msg = str(e)[:80]
            print(f"    {name:15s}: FAILED ({err_msg})")

    print()


def run_microbenchmarks(scale="ablation", bs=4, seq=4096):
    """Benchmark individual model components."""
    if not torch.cuda.is_available():
        print("CUDA not available — skipping microbenchmarks")
        return

    from nanoseek.nanoseek.config import get_config
    from nanoseek.nanoseek.model import NanoSeekModel

    config = get_config(scale)
    device = torch.device("cuda")

    print(f"\nBuilding {scale} model...")
    with torch.device("meta"):
        model = NanoSeekModel(config)
    model.to_empty(device=device)
    model.init_weights()
    model.eval()

    D = config.hidden_size
    x = torch.randn(bs, seq, D, dtype=torch.bfloat16, device=device)
    ids = torch.randint(0, config.vocab_size, (bs, seq), device=device)
    freqs = model.freqs_cis[:seq].to(device)

    # Count params
    params = model.num_parameters()
    print(f"\n=== Microbenchmarks: {scale} | B={bs} S={seq} D={D} ===")
    print(f"    Active: {params['active']:,}  Total: {params['total']:,}")
    print()

    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        # --- Individual layer components ---
        layer0 = model.layers[0]

        bench(lambda: layer0.input_layernorm(x),
              label="RMSNorm (1 call)")

        bench(lambda: layer0.self_attn(layer0.input_layernorm(x), freqs),
              label="MLA attention (1 layer)")

        # Dense FFN (layer 0 or 1)
        if not layer0.use_moe:
            bench(lambda: layer0.ffn(layer0.post_attention_layernorm(x)),
                  label="Dense FFN (1 layer)")

        # MoE FFN (layer 2+)
        for i in range(config.num_layers):
            if model.layers[i].use_moe:
                moe_layer = model.layers[i]
                bench(lambda: moe_layer.ffn(moe_layer.post_attention_layernorm(x)),
                      label=f"MoE FFN (layer {i}, 64e top-8)")

                # Gate routing only
                moe = moe_layer.ffn
                x_flat = x.view(-1, D)
                bench(lambda: moe.gate(x_flat),
                      label="  MoE gate only")

                # Shared expert only
                bench(lambda: moe.shared_expert(x_flat),
                      label="  Shared expert only")
                break  # only benchmark one MoE layer

        # --- Full model ---
        print()
        bench(lambda: model(ids),
              label="Full forward (no loss)", repeat=20)

        bench(lambda: model(ids, labels=ids, mtp_lambda=0.1),
              label="Full forward + loss", repeat=20)

    # Memory report
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    print(f"\n  Peak allocated: {peak:.2f} GB")
    print(f"  Peak reserved:  {reserved:.2f} GB")
    print(f"  Headroom:       {80.0 - reserved:.1f} GB (of 80 GB H100)")

    # MFU estimate for full forward+loss
    n_active = params['active']
    fwd_time = bench(lambda: model(ids, labels=ids, mtp_lambda=0.1),
                     label="Full forward + loss (for MFU)", repeat=20)
    tokens = bs * seq
    # Forward only = 2*N FLOPs/token (no backward)
    flops_fwd = 2 * n_active * tokens
    flops_per_sec = flops_fwd / (fwd_time / 1000)
    peak_flops = 989e12  # H100 SXM BF16
    mfu_fwd = 100 * flops_per_sec / peak_flops
    print(f"\n  Forward-only MFU: {mfu_fwd:.1f}% (of 989 TFLOPS BF16)")
    print(f"  Training MFU estimate: ~{mfu_fwd * 0.7:.1f}% (forward is ~1/3 of fwd+bwd+optim)")


def run_batch_sweep(scale="ablation", seq=4096):
    """Find max batch size and optimal throughput."""
    if not torch.cuda.is_available():
        print("CUDA not available — skipping batch sweep")
        return

    from nanoseek.nanoseek.config import get_config
    from nanoseek.nanoseek.model import NanoSeekModel

    config = get_config(scale)
    device = torch.device("cuda")

    print(f"\n=== Batch Size Sweep: {scale} S={seq} ===\n")
    print(f"  {'BS':>4s}  {'step_ms':>10s}  {'tok/s':>10s}  {'mem_GB':>8s}  {'MFU%':>6s}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*6}")

    with torch.device("meta"):
        model = NanoSeekModel(config)
    model.to_empty(device=device)
    model.init_weights()
    model.train()

    n_active = model.num_parameters()['active']
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for bs in [1, 2, 4, 8, 16, 32, 64]:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            ids = torch.randint(0, config.vocab_size, (bs, seq), device=device)

            # Warmup
            for _ in range(3):
                with torch.autocast("cuda", torch.bfloat16):
                    outputs = model(ids, labels=ids, mtp_lambda=0.1)
                outputs['loss'].backward()
                optimizer.step()
                optimizer.zero_grad()

            torch.cuda.synchronize()

            # Measure
            times = []
            for _ in range(10):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.autocast("cuda", torch.bfloat16):
                    outputs = model(ids, labels=ids, mtp_lambda=0.1)
                outputs['loss'].backward()
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)

            med_s = sorted(times)[len(times) // 2]
            tok_per_s = bs * seq / med_s
            peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
            flops_per_s = 6 * n_active * bs * seq / med_s
            mfu = 100 * flops_per_s / 989e12

            print(f"  {bs:4d}  {med_s*1000:10.1f}  {tok_per_s:10,.0f}  {peak_gb:8.2f}  {mfu:6.1f}")

        except torch.cuda.OutOfMemoryError:
            print(f"  {bs:4d}  {'OOM':>10s}")
            torch.cuda.empty_cache()
            # Re-create model for next iteration
            del model, optimizer
            with torch.device("meta"):
                model = NanoSeekModel(config)
            model.to_empty(device=device)
            model.init_weights()
            model.train()
            n_active = model.num_parameters()['active']
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NanoSeek performance benchmarks")
    parser.add_argument("--scale", default="ablation", choices=["anchor", "ablation", "1b"])
    parser.add_argument("--bs", type=int, default=4, help="Batch size for microbenchmarks")
    parser.add_argument("--seq", type=int, default=4096, help="Sequence length")
    parser.add_argument("--sdpa-check", action="store_true", help="Check SDPA backend dispatch")
    parser.add_argument("--batch-sweep", action="store_true", help="Sweep batch sizes for max throughput")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    args = parser.parse_args()

    if args.sdpa_check or args.all:
        check_sdpa_backends()

    if args.batch_sweep or args.all:
        run_batch_sweep(args.scale, args.seq)

    if not args.sdpa_check and not args.batch_sweep or args.all:
        run_microbenchmarks(args.scale, args.bs, args.seq)
