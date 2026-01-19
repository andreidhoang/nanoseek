"""
Numerical Stability Validation Tests

DeepSeek-Level Validation:
1. NaN/Inf detection in forward pass
2. Gradient explosion/vanishing detection
3. Mixed precision stability
4. Edge case handling
"""

import pytest
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.model import NanoSeekModel, MultiHeadLatentAttention, MoE
from model.config import get_nanoseek_config

from tests.utils import check_no_nan_inf, check_gradient_health


class TestNaNInfDetection:
    """Test NaN and Inf detection in forward pass."""

    def test_no_nan_in_forward(self, model_minimal, sample_batch_minimal):
        """Verify no NaN values in forward pass."""
        outputs = model_minimal(
            sample_batch_minimal['input_ids'],
            labels=sample_batch_minimal['labels']
        )

        # Check all outputs
        check_no_nan_inf(outputs['logits'], "logits")
        check_no_nan_inf(outputs['loss'], "loss")
        check_no_nan_inf(outputs['main_loss'], "main_loss")
        check_no_nan_inf(outputs['hidden_states'], "hidden_states")

    def test_no_nan_with_extreme_inputs(self, model_minimal, minimal_config, device):
        """Verify no NaN with extreme input values."""
        model_minimal.eval()

        # Test with edge case token IDs
        edge_cases = [
            torch.zeros(2, 32, dtype=torch.long, device=device),  # All zeros
            torch.full((2, 32), minimal_config.vocab_size - 1, dtype=torch.long, device=device),  # Max token
        ]

        for i, x in enumerate(edge_cases):
            with torch.no_grad():
                outputs = model_minimal(x)
            check_no_nan_inf(outputs['logits'], f"edge_case_{i}_logits")

    def test_no_nan_after_many_iterations(self, model_minimal, sample_batch_minimal):
        """Verify no NaN accumulation over many forward passes."""
        model_minimal.train()

        for i in range(10):
            outputs = model_minimal(
                sample_batch_minimal['input_ids'],
                labels=sample_batch_minimal['labels']
            )
            check_no_nan_inf(outputs['loss'], f"iteration_{i}_loss")

    def test_attention_softmax_stability(self, mla_minimal, device):
        """Verify attention softmax doesn't produce NaN/Inf."""
        # Large values that might cause softmax overflow
        x_large = torch.randn(2, 64, mla_minimal.hidden_size, device=device) * 100

        output, _ = mla_minimal(x_large)
        check_no_nan_inf(output, "MLA output with large inputs")

        # Small values that might cause underflow
        x_small = torch.randn(2, 64, mla_minimal.hidden_size, device=device) * 0.001

        output, _ = mla_minimal(x_small)
        check_no_nan_inf(output, "MLA output with small inputs")


class TestGradientStability:
    """Test gradient stability during training."""

    def test_no_gradient_explosion(self, model_minimal, sample_batch_minimal):
        """Verify gradients don't explode."""
        outputs = model_minimal(
            sample_batch_minimal['input_ids'],
            labels=sample_batch_minimal['labels']
        )
        outputs['loss'].backward()

        grad_health = check_gradient_health(model_minimal)

        # Check for explosion
        assert grad_health['total_norm'] < 1000, \
            f"Gradient explosion detected: norm={grad_health['total_norm']}"
        assert grad_health['max_grad'] < 100, \
            f"Individual gradient explosion: max={grad_health['max_grad']}"

    def test_no_gradient_vanishing(self, model_minimal, sample_batch_minimal):
        """Verify gradients don't vanish completely."""
        outputs = model_minimal(
            sample_batch_minimal['input_ids'],
            labels=sample_batch_minimal['labels']
        )
        outputs['loss'].backward()

        grad_health = check_gradient_health(model_minimal)

        # Check for vanishing (at least some parameters should have non-trivial gradients)
        assert grad_health['total_norm'] > 1e-8, \
            f"Gradient vanishing detected: norm={grad_health['total_norm']}"

    def test_gradient_clipping_effectiveness(self, model_minimal, sample_batch_minimal):
        """Verify gradient clipping works correctly."""
        outputs = model_minimal(
            sample_batch_minimal['input_ids'],
            labels=sample_batch_minimal['labels']
        )
        outputs['loss'].backward()

        # Apply gradient clipping
        max_norm = 1.0
        total_norm = torch.nn.utils.clip_grad_norm_(model_minimal.parameters(), max_norm)

        # After clipping, norm should be <= max_norm (with some tolerance)
        clipped_norm = 0.0
        for p in model_minimal.parameters():
            if p.grad is not None:
                clipped_norm += p.grad.norm().item() ** 2
        clipped_norm = clipped_norm ** 0.5

        # Clipped norm should be reasonable
        assert clipped_norm < max_norm * len(list(model_minimal.parameters())) + 1.0

    def test_gradient_flow_to_embeddings(self, model_minimal, sample_batch_minimal):
        """Verify gradients reach embedding layer."""
        outputs = model_minimal(
            sample_batch_minimal['input_ids'],
            labels=sample_batch_minimal['labels']
        )
        outputs['loss'].backward()

        # Embedding should have gradients
        assert model_minimal.embed_tokens.weight.grad is not None, \
            "Embedding should have gradients"

        grad_norm = model_minimal.embed_tokens.weight.grad.norm().item()
        assert grad_norm > 1e-10, \
            f"Embedding gradient too small: {grad_norm}"


class TestMixedPrecisionStability:
    """Test stability with mixed precision training."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for bfloat16")
    def test_bfloat16_forward_stability(self, minimal_config, device):
        """Verify bfloat16 forward pass is stable."""
        model = NanoSeekModel(minimal_config).to(device).to(torch.bfloat16)

        x = torch.randint(0, minimal_config.vocab_size, (2, 64), device=device)
        labels = torch.randint(0, minimal_config.vocab_size, (2, 64), device=device)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(x, labels=labels)

        assert not torch.isnan(outputs['loss']), "NaN in bfloat16 loss"
        assert not torch.isinf(outputs['loss']), "Inf in bfloat16 loss"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for bfloat16")
    def test_bfloat16_backward_stability(self, minimal_config, device):
        """Verify bfloat16 backward pass is stable."""
        model = NanoSeekModel(minimal_config).to(device).to(torch.bfloat16)

        x = torch.randint(0, minimal_config.vocab_size, (2, 64), device=device)
        labels = torch.randint(0, minimal_config.vocab_size, (2, 64), device=device)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(x, labels=labels)
            outputs['loss'].backward()

        # Check gradients
        nan_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_count += 1

        assert nan_count == 0, f"NaN gradients in {nan_count} parameters"

    def test_float32_precision_baseline(self, model_minimal, sample_batch_minimal):
        """Establish float32 baseline for comparison."""
        model_minimal.float()

        outputs = model_minimal(
            sample_batch_minimal['input_ids'],
            labels=sample_batch_minimal['labels']
        )

        check_no_nan_inf(outputs['loss'], "float32 loss")

        outputs['loss'].backward()
        grad_health = check_gradient_health(model_minimal)

        assert grad_health['nan_count'] == 0, "NaN in float32 gradients"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_token_sequence(self, model_minimal, minimal_config, device):
        """Test with single token sequence."""
        x = torch.randint(0, minimal_config.vocab_size, (1, 1), device=device)

        outputs = model_minimal(x)
        check_no_nan_inf(outputs['logits'], "single_token_logits")

    def test_very_long_sequence(self, minimal_config, device):
        """Test with sequence at max length."""
        # Use a smaller model for this test
        config = minimal_config
        model = NanoSeekModel(config).to(device)
        model.eval()

        max_len = min(config.max_position_embeddings, 256)  # Limit for test speed
        x = torch.randint(0, config.vocab_size, (1, max_len), device=device)

        with torch.no_grad():
            outputs = model(x)

        check_no_nan_inf(outputs['logits'], "long_sequence_logits")

    def test_batch_size_one(self, model_minimal, minimal_config, device):
        """Test with batch size 1."""
        x = torch.randint(0, minimal_config.vocab_size, (1, 32), device=device)
        labels = torch.randint(0, minimal_config.vocab_size, (1, 32), device=device)

        outputs = model_minimal(x, labels=labels)

        check_no_nan_inf(outputs['loss'], "batch_1_loss")

        outputs['loss'].backward()
        grad_health = check_gradient_health(model_minimal)
        assert grad_health['nan_count'] == 0

    def test_all_same_tokens(self, model_minimal, minimal_config, device):
        """Test with sequence of identical tokens."""
        x = torch.full((2, 32), 100, dtype=torch.long, device=device)

        outputs = model_minimal(x)
        check_no_nan_inf(outputs['logits'], "same_token_logits")

    def test_sequential_token_ids(self, model_minimal, device):
        """Test with sequential token IDs (0, 1, 2, ...)."""
        seq_len = 64
        x = torch.arange(seq_len, device=device).unsqueeze(0).expand(2, -1)

        outputs = model_minimal(x)
        check_no_nan_inf(outputs['logits'], "sequential_tokens_logits")


class TestMoENumericalStability:
    """Test MoE-specific numerical stability."""

    def test_moe_router_no_nan(self, moe_minimal, sample_hidden_minimal):
        """Verify MoE router doesn't produce NaN."""
        x_flat = sample_hidden_minimal.view(-1, sample_hidden_minimal.shape[-1])

        weights, indices = moe_minimal.gate(x_flat)

        check_no_nan_inf(weights, "router_weights")

    def test_moe_expert_isolation(self, moe_minimal, device):
        """Verify individual experts don't produce NaN."""
        x = torch.randn(10, moe_minimal.dim, device=device)

        for i, expert in enumerate(moe_minimal.experts):
            output = expert(x)
            check_no_nan_inf(output, f"expert_{i}_output")

    def test_moe_with_extreme_routing(self, moe_minimal, device):
        """Test MoE with inputs that might cause extreme routing."""
        # Very large input magnitudes
        x_large = torch.randn(2, 32, moe_minimal.dim, device=device) * 100
        output, _ = moe_minimal(x_large)
        check_no_nan_inf(output, "moe_large_input")

        # Very small input magnitudes
        x_small = torch.randn(2, 32, moe_minimal.dim, device=device) * 0.001
        output, _ = moe_minimal(x_small)
        check_no_nan_inf(output, "moe_small_input")


class TestLossComponentStability:
    """Test stability of individual loss components."""

    def test_cross_entropy_stability(self, model_minimal, sample_batch_minimal, minimal_config):
        """Verify cross-entropy loss is stable."""
        outputs = model_minimal(
            sample_batch_minimal['input_ids'],
            labels=sample_batch_minimal['labels']
        )

        # Main loss should be positive and finite
        main_loss = outputs['main_loss']
        assert main_loss > 0, f"Main loss should be positive: {main_loss}"
        assert main_loss < 100, f"Main loss unreasonably high: {main_loss}"
        check_no_nan_inf(main_loss, "main_loss")

    def test_mtp_loss_stability(self, model_minimal, sample_batch_minimal):
        """Verify MTP loss is stable."""
        if model_minimal.mtp is None:
            pytest.skip("Model has no MTP")

        outputs = model_minimal(
            sample_batch_minimal['input_ids'],
            labels=sample_batch_minimal['labels']
        )

        if 'mtp_loss' in outputs:
            mtp_loss = outputs['mtp_loss']
            check_no_nan_inf(mtp_loss, "mtp_loss")
            assert mtp_loss >= 0, f"MTP loss should be non-negative: {mtp_loss}"

    def test_aux_loss_stability(self, model_minimal, sample_batch_minimal):
        """Verify auxiliary loss is stable."""
        outputs = model_minimal(
            sample_batch_minimal['input_ids'],
            labels=sample_batch_minimal['labels']
        )

        if 'aux_loss' in outputs:
            aux_loss = outputs['aux_loss']
            check_no_nan_inf(aux_loss, "aux_loss")
            assert aux_loss >= 0, f"Aux loss should be non-negative: {aux_loss}"

    def test_combined_loss_stability(self, model_minimal, sample_batch_minimal):
        """Verify combined loss is stable."""
        outputs = model_minimal(
            sample_batch_minimal['input_ids'],
            labels=sample_batch_minimal['labels']
        )

        total_loss = outputs['loss']
        main_loss = outputs['main_loss']

        # Total loss should be >= main loss
        assert total_loss >= main_loss - 1e-6, \
            f"Total loss {total_loss} < main loss {main_loss}"

        check_no_nan_inf(total_loss, "total_loss")
