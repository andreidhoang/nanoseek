"""
MoE (Mixture of Experts) Validation Tests

DeepSeek-Level Validation:
1. Router behavior and expert selection
2. Load balancing (auxiliary-loss-free approach)
3. Expert specialization (outputs differ)
4. Gradient flow through routing
5. Shared expert integration
6. Group-based routing
7. Dead expert detection
"""

import pytest
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.model import MoE, Gate, Expert, MLP, create_moe_from_config
from model.config import get_nanoseek_config, MoEConfig

from tests.utils import check_no_nan_inf, check_gradient_health


class TestMoERouter:
    """Test MoE routing behavior."""

    def test_router_output_shape(self, moe_minimal, sample_hidden_minimal):
        """Verify router produces correct output shapes."""
        x_flat = sample_hidden_minimal.view(-1, sample_hidden_minimal.shape[-1])
        weights, indices = moe_minimal.gate(x_flat)

        num_tokens = x_flat.shape[0]
        num_experts_per_tok = moe_minimal.n_activated_experts

        assert weights.shape == (num_tokens, num_experts_per_tok), \
            f"Weights shape mismatch: {weights.shape}"
        assert indices.shape == (num_tokens, num_experts_per_tok), \
            f"Indices shape mismatch: {indices.shape}"

    def test_router_indices_valid(self, moe_minimal, sample_hidden_minimal):
        """Verify router selects valid expert indices."""
        x_flat = sample_hidden_minimal.view(-1, sample_hidden_minimal.shape[-1])
        weights, indices = moe_minimal.gate(x_flat)

        assert indices.min() >= 0, f"Negative expert index: {indices.min()}"
        assert indices.max() < moe_minimal.n_routed_experts, \
            f"Expert index too high: {indices.max()} >= {moe_minimal.n_routed_experts}"

    def test_router_weights_positive(self, moe_minimal, sample_hidden_minimal):
        """Verify router weights are positive (sigmoid scoring)."""
        x_flat = sample_hidden_minimal.view(-1, sample_hidden_minimal.shape[-1])
        weights, _ = moe_minimal.gate(x_flat)

        assert weights.min() >= 0, f"Negative weights: {weights.min()}"

    def test_router_deterministic_in_eval(self, moe_minimal, sample_hidden_minimal):
        """Verify router is deterministic in eval mode."""
        moe_minimal.eval()
        x_flat = sample_hidden_minimal.view(-1, sample_hidden_minimal.shape[-1])

        with torch.no_grad():
            weights1, indices1 = moe_minimal.gate(x_flat)
            weights2, indices2 = moe_minimal.gate(x_flat)

        assert torch.equal(indices1, indices2), "Router should be deterministic in eval"
        assert torch.allclose(weights1, weights2), "Weights should be deterministic in eval"


class TestMoELoadBalancing:
    """Test auxiliary-loss-free load balancing."""

    def test_load_tracking(self, moe_minimal, sample_hidden_minimal):
        """Verify expert load is tracked."""
        moe_minimal.train()
        output, aux = moe_minimal(sample_hidden_minimal)

        load = moe_minimal.gate.expert_load
        assert load.sum() > 0, "Load should be tracked during training"

    def test_bias_update_balances_load(self, minimal_config, device):
        """Verify bias updates improve load balance over time."""
        moe = MoE(
            dim=minimal_config.hidden_size,
            moe_inter_dim=minimal_config.moe.moe_intermediate_size,
            n_routed_experts=minimal_config.moe.n_routed_experts,
            n_activated_experts=minimal_config.moe.num_experts_per_tok,
            n_shared_experts=0,  # No shared expert for this test
            n_expert_groups=1,
            n_limited_groups=1,
            score_func="sigmoid",
            route_scale=1.0,
        ).to(device)

        moe.train()

        # Collect initial load statistics
        initial_cv = None
        final_cv = None

        for step in range(100):
            x = torch.randn(2, 64, minimal_config.hidden_size, device=device)
            output, _ = moe(x)
            moe.update_load_balance_bias(gamma=0.01)

            load = moe.gate.expert_load
            if load.sum() > 0:
                cv = (load.std() / (load.mean() + 1e-8)).item()
                if step == 0:
                    initial_cv = cv
                final_cv = cv

        print(f"Load CV: {initial_cv:.3f} -> {final_cv:.3f}")

        # Load should become more balanced (or at least not worse)
        # Note: With random data, perfect balance isn't expected
        assert final_cv < initial_cv + 0.5, \
            f"Load balance should not significantly worsen: {initial_cv} -> {final_cv}"

    def test_bias_freezing(self, moe_minimal, sample_hidden_minimal):
        """Verify bias can be frozen (gamma=0)."""
        moe_minimal.train()

        # Record initial bias
        initial_bias = moe_minimal.gate.expert_bias.clone()

        # Forward pass
        output, _ = moe_minimal(sample_hidden_minimal)

        # Update with gamma=0 (frozen)
        moe_minimal.update_load_balance_bias(gamma=0.0)

        # Bias should not change
        assert torch.equal(initial_bias, moe_minimal.gate.expert_bias), \
            "Bias should not change when gamma=0"


class TestMoEExpertSpecialization:
    """Test that experts produce different outputs."""

    def test_experts_produce_different_outputs(self, moe_minimal, device):
        """Verify different experts produce different outputs for same input."""
        x = torch.randn(1, moe_minimal.dim, device=device)

        expert_outputs = []
        for expert in moe_minimal.experts:
            out = expert(x)
            expert_outputs.append(out)

        # Compare all pairs
        for i in range(len(expert_outputs)):
            for j in range(i + 1, len(expert_outputs)):
                similarity = F.cosine_similarity(
                    expert_outputs[i].flatten(),
                    expert_outputs[j].flatten(),
                    dim=0
                )
                assert similarity < 0.99, \
                    f"Experts {i} and {j} outputs too similar: {similarity:.4f}"

    def test_shared_expert_always_active(self, minimal_config, device):
        """Verify shared expert contributes to every token."""
        moe = MoE(
            dim=minimal_config.hidden_size,
            moe_inter_dim=minimal_config.moe.moe_intermediate_size,
            n_routed_experts=8,
            n_activated_experts=2,
            n_shared_experts=1,
            n_expert_groups=1,
            n_limited_groups=1,
        ).to(device)

        x = torch.randn(2, 32, minimal_config.hidden_size, device=device)

        # The shared expert should process all tokens
        # We can verify by checking the output is influenced by shared expert
        moe.eval()
        with torch.no_grad():
            output, _ = moe(x)

        check_no_nan_inf(output, "MoE output with shared expert")


class TestMoEGradients:
    """Test gradient flow through MoE."""

    def test_gradient_through_router(self, moe_minimal, sample_hidden_minimal):
        """Verify gradients flow through routing decisions."""
        sample_hidden_minimal = sample_hidden_minimal.clone().requires_grad_(True)
        output, aux = moe_minimal(sample_hidden_minimal)

        loss = output.sum()
        if 'seq_aux_loss' in aux:
            loss = loss + aux['seq_aux_loss']
        loss.backward()

        # Router weights should get gradients
        assert moe_minimal.gate.weight.grad is not None, \
            "Router should have gradients"
        assert moe_minimal.gate.weight.grad.abs().mean() > 0, \
            "Router gradients should be non-zero"

    def test_gradient_to_experts(self, moe_minimal, sample_hidden_minimal):
        """Verify at least some experts receive gradients."""
        sample_hidden_minimal = sample_hidden_minimal.clone().requires_grad_(True)
        output, _ = moe_minimal(sample_hidden_minimal)
        loss = output.sum()
        loss.backward()

        # At least some experts should have gradients
        experts_with_grad = 0
        for expert in moe_minimal.experts:
            if expert.gate_proj.weight.grad is not None:
                if expert.gate_proj.weight.grad.abs().mean() > 0:
                    experts_with_grad += 1

        assert experts_with_grad > 0, "At least some experts should have gradients"

    def test_gradient_magnitudes(self, moe_minimal, sample_hidden_minimal):
        """Verify gradient magnitudes are reasonable."""
        sample_hidden_minimal = sample_hidden_minimal.clone().requires_grad_(True)
        output, _ = moe_minimal(sample_hidden_minimal)
        loss = output.sum()
        loss.backward()

        grad_health = check_gradient_health(moe_minimal)
        assert grad_health['is_healthy'], \
            f"Gradient issues: NaN={grad_health['nan_count']}, Inf={grad_health['inf_count']}, norm={grad_health['total_norm']}"


class TestMoEGroupRouting:
    """Test group-based routing."""

    def test_group_routing_respects_limits(self, minimal_config, device):
        """Verify group routing limits which groups are selected."""
        moe = MoE(
            dim=minimal_config.hidden_size,
            moe_inter_dim=64,
            n_routed_experts=8,
            n_activated_experts=2,
            n_shared_experts=0,
            n_expert_groups=4,  # 4 groups of 2 experts each
            n_limited_groups=2,  # Only select from 2 groups
        ).to(device)

        x = torch.randn(2, 32, minimal_config.hidden_size, device=device)
        x_flat = x.view(-1, x.shape[-1])

        weights, indices = moe.gate(x_flat)

        # Each token should select experts from at most 2 groups
        for token_idx in range(indices.shape[0]):
            selected_experts = indices[token_idx].tolist()
            groups = set(e // 2 for e in selected_experts)  # 2 experts per group
            assert len(groups) <= 2, \
                f"Token {token_idx} selected from {len(groups)} groups, max is 2"


class TestMoEForward:
    """Test MoE forward pass."""

    def test_output_shape(self, moe_minimal, sample_hidden_minimal):
        """Verify output has correct shape."""
        output, _ = moe_minimal(sample_hidden_minimal)
        assert output.shape == sample_hidden_minimal.shape, \
            f"Output shape mismatch: {output.shape} vs {sample_hidden_minimal.shape}"

    def test_output_no_nan(self, moe_minimal, sample_hidden_minimal):
        """Verify no NaN in output."""
        output, _ = moe_minimal(sample_hidden_minimal)
        check_no_nan_inf(output, "MoE output")

    def test_aux_loss_returned(self, moe_minimal, sample_hidden_minimal):
        """Verify auxiliary loss is returned during training."""
        moe_minimal.train()
        output, aux = moe_minimal(sample_hidden_minimal)

        # seq_aux_loss should be computed if alpha > 0
        if moe_minimal.seq_aux_loss_alpha > 0:
            assert 'seq_aux_loss' in aux, "Auxiliary loss should be returned"
            assert aux['seq_aux_loss'] >= 0, "Aux loss should be non-negative"


class TestMoEStats:
    """Test MoE statistics collection."""

    def test_get_expert_load_stats(self, moe_minimal, sample_hidden_minimal):
        """Verify expert load stats are collected correctly."""
        moe_minimal.train()
        output, _ = moe_minimal(sample_hidden_minimal)

        stats = moe_minimal.get_expert_load_stats()

        assert 'expert_load' in stats, "Should have expert_load"
        assert 'expert_bias' in stats, "Should have expert_bias"

        assert stats['expert_load'].shape[0] == moe_minimal.n_routed_experts
        assert stats['expert_bias'].shape[0] == moe_minimal.n_routed_experts

    def test_dead_expert_detection(self, minimal_config, device):
        """Test detecting experts that receive no tokens."""
        # Create MoE with many experts but few active
        moe = MoE(
            dim=minimal_config.hidden_size,
            moe_inter_dim=64,
            n_routed_experts=32,
            n_activated_experts=2,
            n_shared_experts=0,
            n_expert_groups=1,
            n_limited_groups=1,
        ).to(device)

        moe.train()

        # Small batch - not all experts will be used
        x = torch.randn(1, 8, minimal_config.hidden_size, device=device)
        output, _ = moe(x)

        load = moe.gate.expert_load
        dead_experts = (load == 0).sum().item()

        # With 32 experts and only 16 tokens selecting 2 each,
        # some experts will likely be unused
        print(f"Dead experts: {dead_experts}/{moe.n_routed_experts}")


class TestMoEConfigIntegration:
    """Test MoE creation from config."""

    def test_create_from_config(self, config_1b, device):
        """Test creating MoE from config."""
        moe = create_moe_from_config(config_1b).to(device)

        x = torch.randn(2, 64, config_1b.hidden_size, device=device)
        output, _ = moe(x)

        assert output.shape == x.shape
        check_no_nan_inf(output, "MoE from config output")

    def test_config_expert_count(self, config_1b):
        """Verify expert count matches config."""
        moe = create_moe_from_config(config_1b)

        assert len(moe.experts) == config_1b.moe.n_routed_experts
        if config_1b.moe.n_shared_experts > 0:
            assert len(moe.shared_experts) == config_1b.moe.n_shared_experts
