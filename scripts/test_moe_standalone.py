#!/usr/bin/env python3
"""
Standalone MoE Test Script - Following Tutorial 04

Tests the token-centric dispatch implementation from model/model.py
without requiring pytest or complex fixtures.

Usage:
    cd nanoseek
    python scripts/test_moe_standalone.py           # Run all tests
    python scripts/test_moe_standalone.py benchmark # Run benchmarks only
    python scripts/test_moe_standalone.py quick     # Quick validation

Reference: tutorials/moe/04_Production_Implementation.md
"""

import sys
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model import (
    MoE,
    Gate,
    SwiGLUFFN,
    Expert,
    token_centric_dispatch,
)
from model.config import get_nanoseek_config


# =============================================================================
# NAIVE IMPLEMENTATION (Reference for Correctness)
# =============================================================================

def naive_moe_dispatch(
    x: torch.Tensor,           # [N, D]
    indices: torch.Tensor,     # [N, K]
    weights: torch.Tensor,     # [N, K]
    experts: nn.ModuleList,    # E experts
) -> torch.Tensor:
    """
    Naive O(K×E×N) dispatch - slow but obviously correct.

    From Tutorial 02: "The Dispatch Problem"
    This is what we're optimizing against.
    """
    N, D = x.shape
    K = indices.shape[1]
    E = len(experts)
    output = torch.zeros(N, D, device=x.device, dtype=x.dtype)

    for k in range(K):
        for e in range(E):
            mask = indices[:, k] == e
            if mask.any():
                expert_out = experts[e](x[mask])
                output[mask] += weights[mask, k:k+1] * expert_out

    return output


# =============================================================================
# TEST 1: DISPATCH EQUIVALENCE (Most Critical Test)
# =============================================================================

def test_dispatch_equivalence():
    """
    Test: Token-centric dispatch == Naive dispatch

    This is the most important test from Tutorial 04.
    If this passes, the optimization is mathematically correct.
    """
    print("\n" + "=" * 60)
    print("TEST 1: Dispatch Equivalence (Tutorial 04 Core Test)")
    print("=" * 60)

    torch.manual_seed(42)

    # Configuration
    N, D, E, K = 128, 256, 8, 2
    print(f"Config: N={N} tokens, D={D} dim, E={E} experts, K={K} active")

    # Create inputs
    x = torch.randn(N, D)
    indices = torch.randint(0, E, (N, K))
    weights = torch.rand(N, K)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

    # Use SwiGLUFFN experts (same as MoE uses)
    experts = nn.ModuleList([SwiGLUFFN(D, D * 2) for _ in range(E)])

    # Run both implementations
    print("\nRunning naive dispatch...")
    naive_out = naive_moe_dispatch(x, indices, weights, experts)

    print("Running token-centric dispatch...")
    fast_out = token_centric_dispatch(x, indices, weights, experts)

    # Compare
    max_diff = (naive_out - fast_out).abs().max().item()
    mean_diff = (naive_out - fast_out).abs().mean().item()

    print(f"\nResults:")
    print(f"  Max difference:  {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    print(f"  Tolerance:       1e-5")

    if max_diff < 1e-5:
        print("\n✓ PASS: Token-centric dispatch matches naive implementation!")
        return True
    else:
        print(f"\n✗ FAIL: Outputs differ by {max_diff:.2e}")
        return False


# =============================================================================
# TEST 2: FULL MOE FORWARD PASS
# =============================================================================

def test_moe_forward_pass():
    """Test complete MoE module forward pass."""
    print("\n" + "=" * 60)
    print("TEST 2: MoE Forward Pass")
    print("=" * 60)

    torch.manual_seed(42)

    # NanoSeek-like configuration
    hidden_size = 256
    moe_inter_dim = 512
    n_routed_experts = 8
    n_activated_experts = 2
    n_shared_experts = 1
    batch_size, seq_len = 4, 64

    print(f"Config:")
    print(f"  hidden_size: {hidden_size}")
    print(f"  n_routed_experts: {n_routed_experts}")
    print(f"  n_activated_experts: {n_activated_experts}")
    print(f"  n_shared_experts: {n_shared_experts}")

    moe = MoE(
        dim=hidden_size,
        moe_inter_dim=moe_inter_dim,
        n_routed_experts=n_routed_experts,
        n_activated_experts=n_activated_experts,
        n_shared_experts=n_shared_experts,
        score_func="softmax",
        route_scale=1.0,
    )

    x = torch.randn(batch_size, seq_len, hidden_size)
    print(f"\nInput shape: {x.shape}")

    output, aux_data = moe(x)

    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.4f}")
    print(f"Output std: {output.std().item():.4f}")

    # Verify
    checks = [
        ("Shape matches", output.shape == x.shape),
        ("No NaN", not torch.isnan(output).any()),
        ("No Inf", not torch.isinf(output).any()),
    ]

    all_pass = True
    for name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
        all_pass = all_pass and passed

    if all_pass:
        print("\n✓ PASS: MoE forward pass successful!")
    return all_pass


# =============================================================================
# TEST 3: EXPERT LOAD TRACKING
# =============================================================================

def test_expert_load_tracking():
    """Test that expert load is tracked correctly."""
    print("\n" + "=" * 60)
    print("TEST 3: Expert Load Tracking")
    print("=" * 60)

    torch.manual_seed(42)

    hidden_size = 128
    n_routed_experts = 8
    n_activated_experts = 2
    batch_size, seq_len = 8, 64

    moe = MoE(
        dim=hidden_size,
        moe_inter_dim=256,
        n_routed_experts=n_routed_experts,
        n_activated_experts=n_activated_experts,
        n_shared_experts=0,
    )
    moe.train()

    x = torch.randn(batch_size, seq_len, hidden_size)
    output, aux_data = moe(x)

    expert_load = aux_data.get("expert_load")
    if expert_load is None:
        expert_load = moe.gate.expert_load

    total_tokens = batch_size * seq_len
    expected_assignments = total_tokens * n_activated_experts

    print(f"Total tokens: {total_tokens}")
    print(f"Experts per token: {n_activated_experts}")
    print(f"Expected assignments: {expected_assignments}")
    print(f"Actual assignments: {expert_load.sum().item():.0f}")
    print(f"\nLoad per expert: {expert_load.tolist()}")

    # Check load sums correctly
    load_diff = abs(expert_load.sum().item() - expected_assignments)
    passed = load_diff < 1

    if passed:
        print("\n✓ PASS: Expert load tracking correct!")
    else:
        print(f"\n✗ FAIL: Load mismatch by {load_diff}")
    return passed


# =============================================================================
# TEST 4: LOAD BALANCE BIAS UPDATE
# =============================================================================

def test_load_balance_bias():
    """Test auxiliary-loss-free load balancing."""
    print("\n" + "=" * 60)
    print("TEST 4: Load Balance Bias Update (DeepSeek Innovation)")
    print("=" * 60)

    torch.manual_seed(42)

    moe = MoE(
        dim=128,
        moe_inter_dim=256,
        n_routed_experts=8,
        n_activated_experts=2,
        n_shared_experts=0,
        score_func="softmax",
    )
    moe.train()

    initial_bias = moe.gate.expert_bias.clone()
    print(f"Initial bias range: [{initial_bias.min():.4f}, {initial_bias.max():.4f}]")

    x = torch.randn(16, 64, 128)

    print("\nRunning 10 forward passes with bias updates...")
    for step in range(10):
        output, aux_data = moe(x)
        moe.update_load_balance_bias(gamma=0.01)

        if step in [0, 4, 9]:
            load = moe.gate.expert_load
            bias = moe.gate.expert_bias
            print(f"  Step {step}: load_std={load.std().item():.2f}, "
                  f"bias_range=[{bias.min():.4f}, {bias.max():.4f}]")

    final_bias = moe.gate.expert_bias
    bias_changed = not torch.allclose(initial_bias, final_bias, atol=1e-6)

    print(f"\nBias changed: {bias_changed}")

    if bias_changed:
        print("✓ PASS: Load balance bias updates correctly!")
    else:
        print("✗ FAIL: Bias should change after updates")
    return bias_changed


# =============================================================================
# TEST 5: GRADIENT FLOW
# =============================================================================

def test_gradient_flow():
    """Test gradients flow through MoE."""
    print("\n" + "=" * 60)
    print("TEST 5: Gradient Flow")
    print("=" * 60)

    torch.manual_seed(42)

    moe = MoE(
        dim=128,
        moe_inter_dim=256,
        n_routed_experts=4,
        n_activated_experts=2,
        n_shared_experts=1,
    )
    moe.train()

    x = torch.randn(4, 32, 128, requires_grad=True)
    output, aux_data = moe(x)

    loss = output.mean()
    loss.backward()

    checks = []

    # Check input gradient
    input_has_grad = x.grad is not None and x.grad.abs().mean() > 0
    checks.append(("Input gradient", input_has_grad))

    # Check gate gradient
    gate_has_grad = moe.gate.weight.grad is not None and moe.gate.weight.grad.abs().mean() > 0
    checks.append(("Gate gradient", gate_has_grad))

    # Check at least some experts have gradients
    experts_with_grad = sum(
        1 for e in moe.experts
        if e.gate_proj.weight.grad is not None and e.gate_proj.weight.grad.abs().mean() > 0
    )
    checks.append(("Expert gradients", experts_with_grad > 0))

    # Check shared expert gradient
    if moe.shared_experts:
        shared_has_grad = (
            moe.shared_experts[0].gate_proj.weight.grad is not None and
            moe.shared_experts[0].gate_proj.weight.grad.abs().mean() > 0
        )
        checks.append(("Shared expert gradient", shared_has_grad))

    all_pass = True
    for name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
        all_pass = all_pass and passed

    if all_pass:
        print("\n✓ PASS: Gradients flow correctly!")
    return all_pass


# =============================================================================
# TEST 6: EMPTY EXPERT BATCHES
# =============================================================================

def test_empty_expert_batches():
    """Test handling of experts with no assigned tokens."""
    print("\n" + "=" * 60)
    print("TEST 6: Empty Expert Batches")
    print("=" * 60)

    torch.manual_seed(42)

    # Small batch, many experts → some get no tokens
    N, D, E, K = 4, 64, 16, 2

    x = torch.randn(N, D)
    # Force specific assignments to leave most experts empty
    indices = torch.tensor([[0, 1], [0, 1], [2, 3], [2, 3]])
    weights = torch.ones(N, K) * 0.5

    experts = nn.ModuleList([SwiGLUFFN(D, D * 2) for _ in range(E)])

    print(f"Tokens: {N}, Experts: {E}")
    print(f"Forcing tokens to experts 0-3 only")

    output = token_centric_dispatch(x, indices, weights, experts)

    passed = (
        output.shape == x.shape and
        not torch.isnan(output).any() and
        not torch.isinf(output).any()
    )

    if passed:
        print(f"\n✓ PASS: Empty expert batches handled correctly!")
    else:
        print(f"\n✗ FAIL: Issue with empty expert handling")
    return passed


# =============================================================================
# TEST 7: NANOSEEK CONFIG INTEGRATION
# =============================================================================

def test_nanoseek_config():
    """Test MoE with NanoSeek production config."""
    print("\n" + "=" * 60)
    print("TEST 7: NanoSeek Config Integration")
    print("=" * 60)

    config = get_nanoseek_config()

    print(f"NanoSeek MoE Configuration:")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  n_routed_experts: {config.moe.n_routed_experts}")
    print(f"  num_experts_per_tok: {config.moe.num_experts_per_tok}")
    print(f"  n_shared_experts: {config.moe.n_shared_experts}")
    print(f"  moe_intermediate_size: {config.moe.moe_intermediate_size}")
    print(f"  scoring_func: {config.moe.scoring_func}")

    moe = MoE(
        dim=config.hidden_size,
        moe_inter_dim=config.moe.moe_intermediate_size,
        n_routed_experts=config.moe.n_routed_experts,
        n_activated_experts=config.moe.num_experts_per_tok,
        n_shared_experts=config.moe.n_shared_experts,
        n_expert_groups=config.moe.n_group,
        n_limited_groups=config.moe.topk_group,
        score_func=config.moe.scoring_func,
        route_scale=config.moe.routed_scaling_factor,
    )

    # Count parameters
    total_params = sum(p.numel() for p in moe.parameters())
    print(f"\nMoE total parameters: {total_params:,}")

    # Forward pass
    x = torch.randn(2, 64, config.hidden_size)
    output, aux_data = moe(x)

    passed = output.shape == x.shape
    print(f"Forward pass shape: {output.shape}")

    if passed:
        print("\n✓ PASS: NanoSeek config integration successful!")
    return passed


# =============================================================================
# BENCHMARK: PERFORMANCE
# =============================================================================

def benchmark_dispatch():
    """Benchmark token-centric vs naive dispatch (from Tutorial 04)."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Dispatch Performance")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    configs = [
        ("Small",  256,  128, 8,  2),
        ("Medium", 1024, 256, 16, 4),
        ("Large",  4096, 512, 32, 8),
    ]

    for name, N, D, E, K in configs:
        print(f"{name}: N={N}, D={D}, E={E}, K={K}")

        x = torch.randn(N, D, device=device)
        indices = torch.randint(0, E, (N, K), device=device)
        weights = torch.rand(N, K, device=device)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        experts = nn.ModuleList([
            nn.Linear(D, D, bias=False).to(device) for _ in range(E)
        ])

        # Warmup
        for _ in range(5):
            _ = token_centric_dispatch(x, indices, weights, experts)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark token-centric
        num_runs = 20
        start = time.perf_counter()
        for _ in range(num_runs):
            _ = token_centric_dispatch(x, indices, weights, experts)
        if device.type == "cuda":
            torch.cuda.synchronize()
        fast_time = (time.perf_counter() - start) / num_runs * 1000

        # Benchmark naive (only for smaller configs)
        if N <= 1024:
            for _ in range(3):
                _ = naive_moe_dispatch(x, indices, weights, experts)
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(num_runs // 2):
                _ = naive_moe_dispatch(x, indices, weights, experts)
            if device.type == "cuda":
                torch.cuda.synchronize()
            naive_time = (time.perf_counter() - start) / (num_runs // 2) * 1000

            speedup = naive_time / fast_time
            print(f"  Naive: {naive_time:.3f}ms | Token-centric: {fast_time:.3f}ms | Speedup: {speedup:.1f}×")
        else:
            throughput = N / fast_time * 1000
            print(f"  Token-centric: {fast_time:.3f}ms | Throughput: {throughput:,.0f} tok/s")

        print()

    print("✓ Benchmark complete!")
    return True


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests():
    """Run all MoE tests."""
    print()
    print("=" * 60)
    print("  NanoSeek MoE Test Suite")
    print("  Reference: tutorials/moe/04_Production_Implementation.md")
    print("=" * 60)

    tests = [
        ("Dispatch Equivalence", test_dispatch_equivalence),
        ("MoE Forward Pass", test_moe_forward_pass),
        ("Expert Load Tracking", test_expert_load_tracking),
        ("Load Balance Bias", test_load_balance_bias),
        ("Gradient Flow", test_gradient_flow),
        ("Empty Expert Batches", test_empty_expert_batches),
        ("NanoSeek Config", test_nanoseek_config),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ {name} ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed_count = sum(1 for _, p in results if p)
    total = len(results)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed_count}/{total} tests passed")

    # Run benchmark if all tests pass
    if passed_count == total:
        print("\n" + "-" * 60)
        benchmark_dispatch()

    return passed_count == total


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "benchmark":
            benchmark_dispatch()
        elif cmd == "equivalence":
            test_dispatch_equivalence()
        elif cmd == "quick":
            print("Quick validation...")
            test_dispatch_equivalence()
            test_moe_forward_pass()
            test_gradient_flow()
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python test_moe_standalone.py [benchmark|equivalence|quick]")
            sys.exit(1)
    else:
        success = run_all_tests()
        sys.exit(0 if success else 1)
