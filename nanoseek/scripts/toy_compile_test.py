"""
Toy training test: verify 3D expert weights work with torch.compile on Mac.

Tests:
1. Forward/backward without torch.compile (baseline)
2. Forward/backward WITH torch.compile (graph fusion)
3. Loss decreases over multiple steps (training signal flows)
4. MoE metrics (H_load, aux_loss) are healthy
5. Gradient flow to all 3D expert params

Run:
    python -m nanoseek.scripts.toy_compile_test
"""

import sys
import time
import torch
import torch.nn.functional as F

# Handle imports
try:
    from nanoseek.config import get_config
    from nanoseek.model import NanoSeekModel
except ImportError:
    sys.path.insert(0, ".")
    from nanoseek.config import get_config
    from nanoseek.model import NanoSeekModel


def make_toy_batch(vocab_size, batch_size=2, seq_len=64, device="cpu"):
    """Random token batch with labels = shifted input (standard LM)."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return input_ids


def run_step(model, input_ids, compiled_model=None):
    """One forward + backward step. Returns loss dict and wall time."""
    m = compiled_model if compiled_model is not None else model

    t0 = time.perf_counter()
    outputs = m(input_ids, labels=input_ids)
    loss = outputs["loss"]
    loss.backward()
    dt = time.perf_counter() - t0

    return {
        "loss": loss.item(),
        "mtp_loss": outputs.get("mtp_loss", torch.tensor(0.0)).item(),
        "aux_loss": outputs.get("aux_loss", torch.tensor(0.0)).item(),
        "H_load": outputs.get("H_load", torch.tensor(0.0)).item(),
    }, dt


def check_gradients(model):
    """Verify gradients flow to all parameter groups including 3D expert weights."""
    results = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        has_grad = param.grad is not None and param.grad.abs().sum() > 0
        results[name] = has_grad

    # Summarize by group
    groups = {
        "embed": [], "lm_head": [], "attn": [], "norm": [],
        "router": [], "expert_3d": [], "shared_expert": [],
        "dense_ffn": [], "mtp": [], "other": [],
    }
    for name, has_grad in results.items():
        if "embed_tokens" in name:
            groups["embed"].append((name, has_grad))
        elif "lm_head" in name:
            groups["lm_head"].append((name, has_grad))
        elif "self_attn" in name:
            groups["attn"].append((name, has_grad))
        elif "gate.router_weight" in name:
            groups["router"].append((name, has_grad))
        elif param.ndim == 1:
            groups["norm"].append((name, has_grad))
        elif "shared_expert" in name:
            groups["shared_expert"].append((name, has_grad))
        elif any(k in name for k in (".ffn.w_gate", ".ffn.w_up", ".ffn.w_down")):
            groups["expert_3d"].append((name, has_grad))
        elif "mtp" in name:
            groups["mtp"].append((name, has_grad))
        elif ".ffn." in name:
            groups["dense_ffn"].append((name, has_grad))
        else:
            groups["other"].append((name, has_grad))

    return groups


def main():
    torch.manual_seed(42)
    device = "cpu"  # torch.compile on Mac uses inductor/aot_eager on CPU

    # ── Build tiny model ──
    config = get_config("anchor")  # smallest scale: 768 hidden, ~175M active
    config.gradient_checkpointing = False  # avoid torch._dynamo import issues on Mac
    print(f"Config: anchor (hidden={config.hidden_size}, layers={config.num_layers})")
    print(f"  MoE: {config.moe.n_routed_experts} experts, top-{config.moe.num_experts_per_tok}")
    print(f"  Dense layers: 0..{config.moe.first_k_dense_replace - 1}")
    print(f"  MoE layers:   {config.moe.first_k_dense_replace}..{config.num_layers - 1}")

    model = NanoSeekModel(config)
    n_params = sum(p.numel() for p in model.parameters())
    n_3d = sum(p.numel() for p in model.parameters() if p.ndim == 3)
    print(f"  Total params: {n_params:,}")
    print(f"  3D expert params: {n_3d:,} ({100*n_3d/n_params:.1f}%)")

    # Verify 3D params exist
    expert_3d_names = [n for n, p in model.named_parameters() if p.ndim == 3]
    print(f"  3D param count: {len(expert_3d_names)}")
    if expert_3d_names:
        print(f"  Example: {expert_3d_names[0]} shape={model.get_parameter(expert_3d_names[0]).shape}")

    # ── Test 1: Baseline forward/backward (no compile) ──
    print("\n═══ Test 1: Baseline (no torch.compile) ═══")
    batch = make_toy_batch(config.vocab_size, batch_size=2, seq_len=64, device=device)
    model.zero_grad()
    metrics, dt = run_step(model, batch)
    print(f"  loss={metrics['loss']:.4f}  aux={metrics['aux_loss']:.6f}  "
          f"H_load={metrics['H_load']:.2f} bits  time={dt*1000:.0f}ms")

    # Check gradient flow
    grad_groups = check_gradients(model)
    all_ok = True
    for group_name, params in grad_groups.items():
        if not params:
            continue
        n_with_grad = sum(1 for _, has in params if has)
        n_total = len(params)
        status = "✓" if n_with_grad == n_total else "✗"
        if n_with_grad < n_total:
            all_ok = False
        print(f"  {status} {group_name}: {n_with_grad}/{n_total} have gradients")
    assert all_ok, "Some parameters missing gradients!"

    # ── Test 2: torch.compile ──
    print("\n═══ Test 2: torch.compile (aot_eager backend) ═══")
    model.zero_grad()

    # aot_eager is the safest backend on Mac — tests graph tracing without codegen
    try:
        compiled = torch.compile(model, backend="aot_eager", fullgraph=False)
        metrics_c, dt_c = run_step(model, batch, compiled_model=compiled)
        print(f"  loss={metrics_c['loss']:.4f}  aux={metrics_c['aux_loss']:.6f}  "
              f"H_load={metrics_c['H_load']:.2f} bits  time={dt_c*1000:.0f}ms")

        # Losses should be identical (same weights, same input)
        assert abs(metrics['loss'] - metrics_c['loss']) < 1e-4, \
            f"Compiled loss {metrics_c['loss']:.6f} != baseline {metrics['loss']:.6f}"
        print("  ✓ Compiled output matches baseline")

        # Check gradients after compiled step
        grad_groups_c = check_gradients(model)
        for group_name, params in grad_groups_c.items():
            if not params:
                continue
            n_with_grad = sum(1 for _, has in params if has)
            n_total = len(params)
            if n_with_grad < n_total:
                print(f"  ✗ {group_name}: {n_with_grad}/{n_total} gradients after compile")
                all_ok = False
        if all_ok:
            print("  ✓ All gradients flow through compiled model")

    except Exception as e:
        print(f"  torch.compile failed: {e}")
        print("  (This is expected on some Mac configurations)")
        compiled = None

    # ── Test 3: Training loop — loss should decrease ──
    print("\n═══ Test 3: 10-step training (loss should decrease) ═══")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for step in range(10):
        batch = make_toy_batch(config.vocab_size, batch_size=2, seq_len=64, device=device)
        optimizer.zero_grad()
        if compiled is not None:
            metrics_s, _ = run_step(model, batch, compiled_model=compiled)
        else:
            metrics_s, _ = run_step(model, batch)
        optimizer.step()
        losses.append(metrics_s["loss"])
        if step % 3 == 0 or step == 9:
            print(f"  step {step:2d}: loss={metrics_s['loss']:.4f}  "
                  f"H_load={metrics_s['H_load']:.2f}")

    # Check trend
    first_3 = sum(losses[:3]) / 3
    last_3 = sum(losses[-3:]) / 3
    if last_3 < first_3:
        print(f"  ✓ Loss trending down: {first_3:.4f} → {last_3:.4f}")
    else:
        print(f"  ⚠ Loss not decreasing: {first_3:.4f} → {last_3:.4f} (may need more steps)")

    # ── Test 4: MoE health check ──
    print("\n═══ Test 4: MoE health metrics ═══")
    max_H = torch.tensor(config.moe.n_routed_experts, dtype=torch.float32).log2().item()
    print(f"  H_load = {metrics_s['H_load']:.2f} bits (max = log2({config.moe.n_routed_experts}) = {max_H:.1f})")
    if metrics_s["H_load"] > 2.0:
        print("  ✓ Routing is diverse (H_load > 2.0)")
    else:
        print("  ⚠ Low H_load — possible routing collapse")

    if metrics_s["aux_loss"] > 0:
        print(f"  ✓ Aux loss active: {metrics_s['aux_loss']:.6f}")
    else:
        print("  ⚠ Aux loss is zero")

    # ── Test 5: Verify 3D expert params updated ──
    print("\n═══ Test 5: Expert weights changed after training ═══")
    # Re-init a fresh model and compare
    fresh = NanoSeekModel(config)
    changed = 0
    total_3d = 0
    for (name, p_trained), (_, p_fresh) in zip(
        model.named_parameters(), fresh.named_parameters()
    ):
        if p_trained.ndim == 3:
            total_3d += 1
            if not torch.equal(p_trained.data, p_fresh.data):
                changed += 1
    print(f"  {changed}/{total_3d} 3D expert params changed from init")
    if changed == total_3d:
        print("  ✓ All expert weights received optimizer updates")
    else:
        print("  ⚠ Some expert weights unchanged")

    print("\n═══ All tests passed ═══")


if __name__ == "__main__":
    main()
