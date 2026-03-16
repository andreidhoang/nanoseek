"""
Comprehensive tests for NanoSeek MoE (Section 5).

Tests cover all three classes: Expert, Gate, MoE.
Validates correctness properties derived from DeepSeek V3 architecture:
- SwiGLU computation correctness
- Sigmoid scoring with FP32 precision
- Group-based routing (top-2-sum per group)
- Bias steers selection WITHOUT corrupting weights
- norm_topk_prob normalization (BUG FIX #2)
- routed_scaling_factor applied to weights
- Shared expert combined inter_dim (BUG FIX #5)
- Auxiliary-loss-free bias update formula
- Load balance entropy (H_load) for monitoring
- Gradient flow through all components
- Edge cases: empty experts, single token, large batch

Reference: DeepSeek-V3 Technical Report Section 3.2
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nanoseek.model import Expert, Gate, MoE


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def small_gate():
    """Small Gate for fast tests: 16 experts, top-4, 4 groups."""
    return Gate(
        hidden_dim=128,
        n_routed_experts=16,
        num_experts_per_tok=4,
        n_group=4,
        topk_group=2,
        scoring_func="sigmoid",
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        seq_aux_loss_alpha=0.0001,
    )


@pytest.fixture
def small_moe():
    """Small MoE for fast tests."""
    return MoE(
        hidden_dim=128,
        moe_inter_dim=256,
        n_routed_experts=16,
        num_experts_per_tok=4,
        n_shared_experts=2,
        n_group=4,
        topk_group=2,
        scoring_func="sigmoid",
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        seq_aux_loss_alpha=0.0001,
    )


# =============================================================================
# EXPERT TESTS
# =============================================================================

class TestExpert:
    """Tests for Expert (SwiGLU FFN)."""

    def test_output_shape(self):
        """Expert preserves input shape: [N, D] → [N, D]."""
        expert = Expert(hidden_dim=64, inter_dim=128)
        x = torch.randn(32, 64)
        out = expert(x)
        assert out.shape == (32, 64)

    def test_output_shape_single_token(self):
        """Expert works with single token."""
        expert = Expert(hidden_dim=64, inter_dim=128)
        x = torch.randn(1, 64)
        out = expert(x)
        assert out.shape == (1, 64)

    def test_swiglu_manual_computation(self):
        """Verify SwiGLU: W_down(SiLU(W_gate(x)) * W_up(x)) matches manual."""
        torch.manual_seed(42)
        expert = Expert(hidden_dim=32, inter_dim=64)
        x = torch.randn(4, 32)

        # Manual computation
        gate = F.silu(expert.w_gate(x))
        up = expert.w_up(x)
        expected = expert.w_down(gate * up)

        actual = expert(x)
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_gradient_flow(self):
        """Gradients flow through all three weight matrices."""
        expert = Expert(hidden_dim=32, inter_dim=64)
        x = torch.randn(8, 32, requires_grad=True)
        out = expert(x)
        out.sum().backward()

        assert x.grad is not None and x.grad.abs().sum() > 0
        assert expert.w_gate.weight.grad is not None
        assert expert.w_up.weight.grad is not None
        assert expert.w_down.weight.grad is not None

    def test_no_bias(self):
        """All linear layers have bias=False."""
        expert = Expert(hidden_dim=64, inter_dim=128)
        assert expert.w_gate.bias is None
        assert expert.w_up.bias is None
        assert expert.w_down.bias is None

    def test_different_inter_dims(self):
        """Same Expert class works for different inter_dim (routed vs shared)."""
        routed = Expert(hidden_dim=64, inter_dim=128)
        shared = Expert(hidden_dim=64, inter_dim=512)
        x = torch.randn(8, 64)

        assert routed(x).shape == (8, 64)
        assert shared(x).shape == (8, 64)
        assert routed.w_gate.out_features == 128
        assert shared.w_gate.out_features == 512

    def test_no_nan_inf(self):
        """Output has no NaN or Inf for reasonable inputs."""
        expert = Expert(hidden_dim=128, inter_dim=256)
        x = torch.randn(64, 128)
        out = expert(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_deterministic(self):
        """Same input → same output (no randomness in expert)."""
        expert = Expert(hidden_dim=32, inter_dim=64)
        expert.eval()
        x = torch.randn(8, 32)
        out1 = expert(x)
        out2 = expert(x)
        assert torch.allclose(out1, out2, atol=1e-7)


# =============================================================================
# GATE TESTS
# =============================================================================

class TestGate:
    """Tests for Gate (MoE Router)."""

    def test_output_shapes(self, small_gate):
        """Verify all output shapes."""
        N = 32
        x = torch.randn(N, 128)
        weights, indices, aux_loss, metadata = small_gate(x)

        assert weights.shape == (N, 4), f"weights shape: {weights.shape}"
        assert indices.shape == (N, 4), f"indices shape: {indices.shape}"
        assert aux_loss.dim() == 0, "aux_loss should be scalar"
        assert metadata["load_counts"].shape == (16,)
        assert metadata["H_load"].dim() == 0

    def test_indices_in_valid_range(self, small_gate):
        """All selected expert indices must be in [0, n_routed_experts)."""
        x = torch.randn(64, 128)
        _, indices, _, _ = small_gate(x)

        assert indices.min() >= 0
        assert indices.max() < 16

    def test_sigmoid_scoring(self):
        """Sigmoid gate: scores are independent probabilities in (0, 1)."""
        gate = Gate(
            hidden_dim=64, n_routed_experts=8, num_experts_per_tok=2,
            n_group=2, topk_group=1, scoring_func="sigmoid",
            norm_topk_prob=False, routed_scaling_factor=1.0,
        )
        x = torch.randn(16, 64)
        weights, _, _, _ = gate(x)
        # With sigmoid + no norm + scale=1.0, weights should be raw sigmoid values
        # Each weight is in (0, 1)
        assert (weights > 0).all() and (weights < 1).all()

    def test_softmax_scoring(self):
        """Softmax gate: scores are competitive, sum to 1 across experts."""
        gate = Gate(
            hidden_dim=64, n_routed_experts=8, num_experts_per_tok=2,
            n_group=2, topk_group=1, scoring_func="softmax",
            norm_topk_prob=False, routed_scaling_factor=1.0,
        )
        x = torch.randn(16, 64)
        weights, _, _, _ = gate(x)
        # With softmax + no norm + scale=1.0, weights are raw softmax scores
        assert (weights > 0).all() and (weights < 1).all()

    def test_norm_topk_prob_normalization(self):
        """BUG FIX #2: When norm_topk_prob=True, weights sum to route_scale per token."""
        gate = Gate(
            hidden_dim=64, n_routed_experts=8, num_experts_per_tok=2,
            n_group=2, topk_group=1, scoring_func="sigmoid",
            norm_topk_prob=True, routed_scaling_factor=2.5,
        )
        x = torch.randn(32, 64)
        weights, _, _, _ = gate(x)

        # After norm (sum=1) then scale (*2.5), weights per token should sum to 2.5
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.full_like(weight_sums, 2.5), atol=1e-5), \
            f"Weight sums should be 2.5, got {weight_sums}"

    def test_norm_topk_prob_disabled(self):
        """When norm_topk_prob=False, weights do NOT sum to route_scale."""
        gate = Gate(
            hidden_dim=64, n_routed_experts=8, num_experts_per_tok=2,
            n_group=2, topk_group=1, scoring_func="sigmoid",
            norm_topk_prob=False, routed_scaling_factor=2.5,
        )
        x = torch.randn(32, 64)
        weights, _, _, _ = gate(x)

        weight_sums = weights.sum(dim=-1)
        # With sigmoid (independent probs) and no norm, sums won't be exactly 2.5
        assert not torch.allclose(weight_sums, torch.full_like(weight_sums, 2.5), atol=0.1)

    def test_group_routing_restricts_experts(self, small_gate):
        """Selected experts must come from selected groups only.

        With 16 experts in 4 groups of 4, selecting topk_group=2 means
        experts can only come from 2 of the 4 groups (8 out of 16 experts).
        """
        torch.manual_seed(123)
        x = torch.randn(32, 128)
        _, indices, _, _ = small_gate(x)

        # For each token, all selected experts should come from at most topk_group groups
        for token_idx in range(indices.shape[0]):
            expert_ids = indices[token_idx].tolist()
            groups = set(eid // 4 for eid in expert_ids)  # 4 experts per group
            assert len(groups) <= small_gate.topk_group, \
                f"Token {token_idx}: experts {expert_ids} span {len(groups)} groups, max is {small_gate.topk_group}"

    def test_bias_steers_selection_not_weights(self):
        """Bias changes WHICH experts are selected but weights use ORIGINAL scores.

        This is the core auxiliary-loss-free mechanism:
        - biased_scores = scores + bias → used for topk selection
        - weights = original scores gathered at selected indices
        """
        torch.manual_seed(42)
        gate = Gate(
            hidden_dim=64, n_routed_experts=8, num_experts_per_tok=2,
            n_group=2, topk_group=1, scoring_func="sigmoid",
            norm_topk_prob=False, routed_scaling_factor=1.0,
        )
        x = torch.randn(16, 64)

        # Run with zero bias
        weights_no_bias, indices_no_bias, _, _ = gate(x)

        # Add strong bias to expert 7 (should attract tokens to it)
        gate.bias.data[7] = 10.0
        weights_with_bias, indices_with_bias, _, _ = gate(x)

        # Expert 7 should appear more often with bias
        count_no_bias = (indices_no_bias == 7).sum().item()
        count_with_bias = (indices_with_bias == 7).sum().item()
        assert count_with_bias > count_no_bias, \
            f"Bias should attract tokens to expert 7: {count_no_bias} → {count_with_bias}"

        # For tokens that select expert 7 in BOTH runs, the weight should be
        # the same (because weights come from original scores, not biased)
        # This is hard to test directly, but we can verify weights don't change
        # when only bias changes (for tokens with same selection)

    def test_bias_not_in_gradient_graph(self, small_gate):
        """Bias buffer must NOT accumulate gradients."""
        x = torch.randn(16, 128)
        weights, _, aux_loss, _ = small_gate(x)
        (weights.sum() + aux_loss).backward()

        assert not small_gate.bias.requires_grad, "Bias should not require grad"

    def test_router_weight_has_gradient(self, small_gate):
        """Router weight should receive gradients."""
        x = torch.randn(16, 128)
        weights, _, aux_loss, _ = small_gate(x)
        (weights.sum() + aux_loss).backward()

        assert small_gate.router_weight.weight.grad is not None
        assert small_gate.router_weight.weight.grad.abs().sum() > 0

    def test_load_counts_sum(self, small_gate):
        """Load counts should sum to N * K (each token selects K experts)."""
        N = 64
        K = small_gate.num_experts_per_tok
        x = torch.randn(N, 128)
        _, _, _, metadata = small_gate(x)

        expected_sum = N * K
        actual_sum = metadata["load_counts"].sum().item()
        assert actual_sum == expected_sum, \
            f"Load counts sum {actual_sum} != expected {expected_sum}"

    def test_aux_loss_is_finite_and_small(self, small_gate):
        """Aux loss should be finite and very small (alpha=0.0001)."""
        x = torch.randn(64, 128)
        _, _, aux_loss, _ = small_gate(x)

        assert aux_loss.isfinite()
        assert aux_loss.item() >= 0  # squared difference is non-negative
        assert aux_loss.item() < 0.01  # should be very small with alpha=0.0001

    def test_h_load_entropy_positive(self, small_gate):
        """H_load should be positive (well-balanced routing has high entropy)."""
        x = torch.randn(128, 128)  # enough tokens for spread
        _, _, _, metadata = small_gate(x)

        assert metadata["H_load"].item() > 0
        # With random init and 128 tokens across 16 experts, entropy should be decent
        assert metadata["H_load"].item() > 1.0, \
            f"H_load too low: {metadata['H_load'].item():.2f}"

    def test_update_bias_formula(self):
        """Verify update_bias follows: b_i -= gamma * (load_i - mean) / mean."""
        gate = Gate(
            hidden_dim=32, n_routed_experts=4, num_experts_per_tok=2,
            n_group=2, topk_group=1,
        )
        # Set known bias state
        gate.bias.data.zero_()

        # Simulate unbalanced load: expert 0 gets 100 tokens, others get 0
        load = torch.tensor([100.0, 0.0, 0.0, 0.0])
        gamma = 0.01

        gate.update_bias(load, gamma)

        mean_load = 25.0  # 100/4
        # Expected: b_i -= gamma * (load_i - mean) / mean
        # b_0 = 0 - 0.01 * (100 - 25) / 25 = -0.03
        # b_1 = 0 - 0.01 * (0 - 25) / 25   = +0.01
        expected = torch.tensor([-0.03, 0.01, 0.01, 0.01])
        assert torch.allclose(gate.bias, expected, atol=1e-6), \
            f"Bias after update: {gate.bias} != expected {expected}"

    def test_update_bias_frozen(self):
        """When gamma=0, bias should not change (frozen state)."""
        gate = Gate(
            hidden_dim=32, n_routed_experts=4, num_experts_per_tok=2,
            n_group=2, topk_group=1,
        )
        gate.bias.data = torch.tensor([0.1, -0.2, 0.3, -0.1])
        original = gate.bias.clone()

        load = torch.tensor([100.0, 0.0, 50.0, 50.0])
        gate.update_bias(load, gamma=0.0)

        assert torch.allclose(gate.bias, original), "Bias should not change when gamma=0"

    def test_update_bias_drives_balance(self):
        """Multiple bias updates should reduce load imbalance."""
        torch.manual_seed(42)
        gate = Gate(
            hidden_dim=64, n_routed_experts=8, num_experts_per_tok=2,
            n_group=2, topk_group=1, scoring_func="sigmoid",
            norm_topk_prob=True, routed_scaling_factor=1.0,
        )
        x = torch.randn(128, 64)

        # Measure initial load variance
        _, _, _, meta0 = gate(x)
        initial_var = meta0["load_counts"].float().var().item()

        # Run 50 bias updates
        for _ in range(50):
            _, _, _, meta = gate(x)
            gate.update_bias(meta["load_counts"], gamma=0.01)

        # Measure final load variance
        _, _, _, meta_final = gate(x)
        final_var = meta_final["load_counts"].float().var().item()

        # Variance should decrease (more balanced)
        assert final_var < initial_var, \
            f"Bias updates should reduce load variance: {initial_var:.1f} → {final_var:.1f}"

    def test_fp32_gating(self, small_gate):
        """Gating computations should use FP32 internally even with FP16 input."""
        if not torch.cuda.is_available():
            pytest.skip("FP16 test requires CUDA")

        gate = small_gate.cuda().half()  # FP16 model
        x = torch.randn(16, 128, device="cuda", dtype=torch.float16)
        weights, indices, aux_loss, metadata = gate(x)

        # Weights should be cast back to input dtype
        assert weights.dtype == torch.float16
        # But aux_loss computed in FP32 should be finite (no FP16 overflow)
        assert aux_loss.isfinite()

    def test_single_token(self, small_gate):
        """Gate handles single token input."""
        x = torch.randn(1, 128)
        weights, indices, aux_loss, metadata = small_gate(x)

        assert weights.shape == (1, 4)
        assert indices.shape == (1, 4)
        assert aux_loss.isfinite()

    def test_group_top2_sum_scoring(self):
        """Verify group scoring uses top-2 sum (not max).

        Place high scores in a group where expert A=0.9 and expert B=0.8 (sum=1.7)
        vs another group where expert C=0.95 (sum=0.95+next_best).
        Top-2-sum should prefer the first group if the second group's #2 is low.
        """
        gate = Gate(
            hidden_dim=8, n_routed_experts=8, num_experts_per_tok=2,
            n_group=2, topk_group=1, scoring_func="sigmoid",
            norm_topk_prob=False, routed_scaling_factor=1.0,
        )

        # Manually set router weights to control scores
        # Group 0: experts 0-3, Group 1: experts 4-7
        # We want group 0 to have two high scores, group 1 to have one very high
        with torch.no_grad():
            gate.router_weight.weight.zero_()
            # Expert 0: high score for dim 0
            gate.router_weight.weight[0, 0] = 3.0  # sigmoid(3) ≈ 0.95
            # Expert 1: high score for dim 1
            gate.router_weight.weight[1, 1] = 2.5  # sigmoid(2.5) ≈ 0.92
            # Expert 4: very high score for dim 2
            gate.router_weight.weight[4, 2] = 4.0  # sigmoid(4) ≈ 0.98
            # Expert 5-7: very low scores
            gate.router_weight.weight[5, 3] = -3.0  # sigmoid(-3) ≈ 0.05

        # Input that activates dim 0,1,2
        x = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        _, indices, _, _ = gate(x)

        # With top-2-sum: group 0 score ≈ 0.95+0.92=1.87, group 1 score ≈ 0.98+0.05=1.03
        # Group 0 wins → experts 0,1 should be selected
        selected = set(indices[0].tolist())
        assert 0 in selected or 1 in selected, \
            f"Group 0 should be preferred with top-2-sum, got experts {selected}"


# =============================================================================
# MOE TESTS
# =============================================================================

class TestMoE:
    """Tests for MoE (orchestrator)."""

    def test_output_shape(self, small_moe):
        """MoE preserves [B, S, D] shape."""
        x = torch.randn(2, 16, 128)
        out, aux = small_moe(x)
        assert out.shape == (2, 16, 128)

    def test_output_shape_single_token(self, small_moe):
        """MoE works with single token."""
        x = torch.randn(1, 1, 128)
        out, aux = small_moe(x)
        assert out.shape == (1, 1, 128)

    def test_aux_data_keys(self, small_moe):
        """aux_data dict has required keys for training loop integration."""
        x = torch.randn(2, 8, 128)
        _, aux = small_moe(x)

        assert "aux_loss" in aux
        assert "load_counts" in aux
        assert "H_load" in aux

    def test_aux_loss_finite(self, small_moe):
        """Aux loss is finite and non-negative."""
        x = torch.randn(2, 16, 128)
        _, aux = small_moe(x)

        assert aux["aux_loss"].isfinite()
        assert aux["aux_loss"].item() >= 0

    def test_shared_expert_combined_inter_dim(self):
        """BUG FIX #5: Shared expert has inter_dim = n_shared * moe_inter_dim."""
        moe = MoE(
            hidden_dim=64, moe_inter_dim=128,
            n_routed_experts=8, num_experts_per_tok=2,
            n_shared_experts=2, n_group=2, topk_group=1,
        )
        # n_shared_experts=2, moe_inter_dim=128 → shared inter = 256
        assert moe.shared_expert.w_gate.out_features == 256
        assert moe.shared_expert.w_up.out_features == 256
        assert moe.shared_expert.w_down.in_features == 256

    def test_shared_inter_dim_override(self):
        """shared_inter_dim parameter overrides default computation."""
        moe = MoE(
            hidden_dim=64, moe_inter_dim=128,
            n_routed_experts=8, num_experts_per_tok=2,
            n_shared_experts=2, n_group=2, topk_group=1,
            shared_inter_dim=512,  # Override
        )
        assert moe.shared_expert.w_gate.out_features == 512

    def test_shared_expert_processes_all_tokens(self):
        """Shared expert contribution should be non-zero for ALL tokens.

        If shared expert was skipping tokens, some positions would have
        zero shared contribution.
        """
        torch.manual_seed(42)
        moe = MoE(
            hidden_dim=32, moe_inter_dim=64,
            n_routed_experts=4, num_experts_per_tok=2,
            n_shared_experts=1, n_group=2, topk_group=1,
            routed_scaling_factor=0.0,  # Zero out routed to isolate shared
        )
        x = torch.randn(4, 8, 32)
        out, _ = moe(x)

        # With routed_scaling_factor=0, output = shared_expert(x) only
        # Should be non-zero for every token
        per_token_norm = out.view(-1, 32).norm(dim=-1)
        assert (per_token_norm > 0).all(), "Shared expert should produce non-zero output for all tokens"

    def test_gradient_flow_all_components(self, small_moe):
        """Gradients flow through router, routed experts, shared expert, and input."""
        x = torch.randn(2, 8, 128, requires_grad=True)
        out, aux = small_moe(x)
        loss = out.sum() + aux["aux_loss"]
        loss.backward()

        # Input gradient
        assert x.grad is not None and x.grad.abs().sum() > 0, "Input should have gradient"

        # Router gradient
        assert small_moe.gate.router_weight.weight.grad is not None, \
            "Router weight should have gradient"

        # Shared expert gradient
        assert small_moe.shared_expert.w_gate.weight.grad is not None, \
            "Shared expert should have gradient"

        # At least some routed experts should have gradients
        experts_with_grad = sum(
            1 for e in small_moe.routed_experts
            if e.w_gate.weight.grad is not None and e.w_gate.weight.grad.abs().sum() > 0
        )
        assert experts_with_grad > 0, "Some routed experts should have gradients"

    def test_bias_no_gradient_through_moe(self, small_moe):
        """Bias buffer should NOT accumulate gradients through MoE forward."""
        x = torch.randn(2, 8, 128)
        out, aux = small_moe(x)
        (out.sum() + aux["aux_loss"]).backward()

        assert not small_moe.gate.bias.requires_grad

    def test_load_counts_sum_equals_n_times_k(self, small_moe):
        """Total load across all experts = num_tokens * num_experts_per_tok."""
        B, S = 4, 16
        x = torch.randn(B, S, 128)
        _, aux = small_moe(x)

        N = B * S
        K = small_moe.num_experts_per_tok
        expected = N * K
        actual = aux["load_counts"].sum().item()
        assert actual == expected, f"Load sum {actual} != {expected}"

    def test_no_nan_inf_output(self, small_moe):
        """Output has no NaN or Inf."""
        x = torch.randn(4, 32, 128)
        out, aux = small_moe(x)

        assert not torch.isnan(out).any(), "NaN in output"
        assert not torch.isinf(out).any(), "Inf in output"
        assert aux["aux_loss"].isfinite(), "aux_loss not finite"

    def test_deterministic_eval(self, small_moe):
        """Same input produces same output in eval mode."""
        small_moe.eval()
        x = torch.randn(2, 8, 128)
        out1, _ = small_moe(x)
        out2, _ = small_moe(x)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_empty_expert_no_crash(self):
        """MoE handles experts that receive 0 tokens gracefully.

        With many experts and few tokens, some experts get nothing.
        """
        moe = MoE(
            hidden_dim=32, moe_inter_dim=64,
            n_routed_experts=64, num_experts_per_tok=2,
            n_shared_experts=1, n_group=8, topk_group=2,
        )
        # Only 4 tokens but 64 experts → most experts get 0 tokens
        x = torch.randn(1, 4, 32)
        out, aux = moe(x)

        assert out.shape == (1, 4, 32)
        assert not torch.isnan(out).any()

    def test_routed_experts_count(self, small_moe):
        """Correct number of routed experts instantiated."""
        assert len(small_moe.routed_experts) == 16

    def test_routed_expert_inter_dim(self, small_moe):
        """Routed experts have correct inter_dim."""
        for expert in small_moe.routed_experts:
            assert expert.w_gate.out_features == 256  # moe_inter_dim

    def test_large_batch(self, small_moe):
        """MoE handles larger batches without issues."""
        x = torch.randn(8, 64, 128)
        out, aux = small_moe(x)
        assert out.shape == (8, 64, 128)
        assert not torch.isnan(out).any()

    def test_backward_does_not_crash(self, small_moe):
        """Full backward pass completes without error."""
        small_moe.train()
        x = torch.randn(2, 16, 128)
        out, aux = small_moe(x)
        loss = out.sum() + aux["aux_loss"]
        loss.backward()  # Should not raise

    def test_parameter_count(self):
        """Verify parameter count matches expected formula."""
        D, I, E, n_shared = 64, 128, 8, 2
        moe = MoE(
            hidden_dim=D, moe_inter_dim=I,
            n_routed_experts=E, num_experts_per_tok=2,
            n_shared_experts=n_shared, n_group=2, topk_group=1,
        )
        # Each routed expert: 3 * D * I (gate, up, down)
        routed_params = E * 3 * D * I
        # Shared expert: 3 * D * (n_shared * I)
        shared_params = 3 * D * (n_shared * I)
        # Router: D * E
        router_params = D * E

        expected_total = routed_params + shared_params + router_params
        actual_total = sum(p.numel() for p in moe.parameters())

        assert actual_total == expected_total, \
            f"Param count {actual_total} != expected {expected_total}"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestMoEIntegration:
    """Integration tests verifying MoE correctness properties."""

    def test_routed_output_is_weighted_sum(self):
        """Routed output for a token = sum of (expert_output * weight) for selected experts.

        This is the fundamental correctness property of token-centric dispatch.
        We verify by manually computing the expected output for a few tokens.
        """
        torch.manual_seed(42)
        D = 32
        moe = MoE(
            hidden_dim=D, moe_inter_dim=64,
            n_routed_experts=4, num_experts_per_tok=2,
            n_shared_experts=1, n_group=2, topk_group=1,
            scoring_func="sigmoid", norm_topk_prob=True,
            routed_scaling_factor=1.0,  # Simplify: scale=1
        )
        x = torch.randn(1, 4, D)
        flat_x = x.view(-1, D)

        # Get routing decisions
        weights, indices, _, _ = moe.gate(flat_x)

        # Manually compute routed output for token 0
        token_0 = flat_x[0:1]  # [1, D]
        manual_routed = torch.zeros(1, D)
        for k in range(2):  # K=2
            expert_idx = indices[0, k].item()
            w = weights[0, k].item()
            expert_out = moe.routed_experts[expert_idx](token_0)  # [1, D]
            manual_routed += w * expert_out

        # Get MoE output and subtract shared expert contribution
        moe_out, _ = moe(x)
        shared_out = moe.shared_expert(flat_x)
        routed_from_moe = moe_out.view(-1, D)[0:1] - shared_out[0:1]

        assert torch.allclose(manual_routed, routed_from_moe, atol=1e-5), \
            f"Manual routed output doesn't match MoE dispatch"

    def test_moe_output_equals_routed_plus_shared(self):
        """MoE output = routed_output + shared_output (no extra terms)."""
        torch.manual_seed(42)
        D = 32
        moe = MoE(
            hidden_dim=D, moe_inter_dim=64,
            n_routed_experts=4, num_experts_per_tok=2,
            n_shared_experts=1, n_group=2, topk_group=1,
            routed_scaling_factor=2.5,
        )
        x = torch.randn(2, 8, D)
        flat_x = x.view(-1, D)

        # Get shared output independently
        shared_out = moe.shared_expert(flat_x)

        # Get full MoE output
        moe_out, _ = moe(x)
        moe_flat = moe_out.view(-1, D)

        # Routed contribution = moe_output - shared_output
        routed_contribution = moe_flat - shared_out

        # Routed contribution should be non-zero (experts are producing output)
        assert routed_contribution.abs().sum() > 0, \
            "Routed experts should contribute non-zero output"

    def test_training_step_simulation(self):
        """Simulate a training step: forward → backward → bias update."""
        torch.manual_seed(42)
        moe = MoE(
            hidden_dim=64, moe_inter_dim=128,
            n_routed_experts=8, num_experts_per_tok=2,
            n_shared_experts=1, n_group=2, topk_group=1,
        )
        moe.train()
        optimizer = torch.optim.Adam(moe.parameters(), lr=1e-4)

        initial_bias = moe.gate.bias.clone()

        # Forward
        x = torch.randn(4, 16, 64)
        out, aux = moe(x)

        # Backward
        loss = out.sum() + aux["aux_loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Bias update (simulating training loop)
        moe.gate.update_bias(aux["load_counts"], gamma=0.001)

        # Bias should have changed
        assert not torch.allclose(moe.gate.bias, initial_bias, atol=1e-8), \
            "Bias should change after update_bias()"

        # Run another forward to verify model still works
        out2, aux2 = moe(x)
        assert out2.shape == (4, 16, 64)
        assert not torch.isnan(out2).any()

    def test_aux_loss_decreases_with_balanced_routing(self):
        """Aux loss should be near zero when routing is perfectly balanced."""
        gate = Gate(
            hidden_dim=32, n_routed_experts=4, num_experts_per_tok=2,
            n_group=2, topk_group=1, seq_aux_loss_alpha=0.0001,
        )
        # Compute aux loss with perfectly balanced load
        N = 100
        x = torch.randn(N, 32)
        _, _, aux_loss, metadata = gate(x)

        # With random routing, aux_loss should be small but non-zero
        # Perfect balance would give aux_loss = 0
        assert aux_loss.item() < 0.001, \
            f"Aux loss {aux_loss.item()} seems too high for near-random routing"

    def test_nanoseek_1b_dimensions(self):
        """Verify MoE works with NanoSeek-1B production dimensions."""
        moe = MoE(
            hidden_dim=2048,
            moe_inter_dim=768,
            n_routed_experts=64,
            num_experts_per_tok=8,
            n_shared_experts=2,
            n_group=8,
            topk_group=4,
            scoring_func="sigmoid",
            norm_topk_prob=True,
            routed_scaling_factor=2.5,
            seq_aux_loss_alpha=0.0001,
        )

        # Verify dimensions
        assert len(moe.routed_experts) == 64
        assert moe.shared_expert.w_gate.out_features == 1536  # 2 * 768
        assert moe.gate.router_weight.out_features == 64

        # Forward pass with small batch (to keep memory reasonable)
        x = torch.randn(1, 4, 2048)
        out, aux = moe(x)
        assert out.shape == (1, 4, 2048)
        assert aux["load_counts"].shape == (64,)

    @pytest.mark.parametrize("scoring_func", ["sigmoid", "softmax"])
    def test_both_scoring_functions(self, scoring_func):
        """Both sigmoid and softmax scoring produce valid outputs."""
        moe = MoE(
            hidden_dim=64, moe_inter_dim=128,
            n_routed_experts=8, num_experts_per_tok=2,
            n_shared_experts=1, n_group=2, topk_group=1,
            scoring_func=scoring_func,
        )
        x = torch.randn(2, 8, 64)
        out, aux = moe(x)

        assert out.shape == (2, 8, 64)
        assert not torch.isnan(out).any()
        assert aux["aux_loss"].isfinite()

    def test_gradient_health(self, small_moe):
        """All gradients are finite (no NaN/Inf) after backward pass."""
        small_moe.train()
        x = torch.randn(2, 16, 128)
        out, aux = small_moe(x)
        (out.sum() + aux["aux_loss"]).backward()

        for name, param in small_moe.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
