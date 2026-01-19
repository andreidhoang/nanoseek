"""
Integration Tests - Full NanoSeek Model

DeepSeek-Level Validation:
1. Complete forward pass
2. Complete backward pass
3. Loss computation (all components)
4. Memory efficiency verification
5. Parameter count validation
6. Generation tests
7. Checkpoint save/load
8. Multi-phase training simulation
"""

import pytest
import torch
import torch.nn.functional as F
import tempfile
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.model import NanoSeekModel
from model.config import get_nanoseek_config

from tests.utils import (
    check_no_nan_inf,
    check_gradient_health,
    compute_kv_cache_size,
    compute_mha_cache_size,
)


class TestFullForward:
    """Test complete forward pass."""

    def test_forward_outputs(self, model_minimal, sample_batch_minimal):
        """Verify forward returns all expected outputs."""
        outputs = model_minimal(
            sample_batch_minimal['input_ids'],
            labels=sample_batch_minimal['labels']
        )

        assert 'logits' in outputs, "Should return logits"
        assert 'loss' in outputs, "Should return loss"
        assert 'main_loss' in outputs, "Should return main_loss"
        assert 'hidden_states' in outputs, "Should return hidden_states"

    def test_logits_shape(self, model_minimal, sample_batch_minimal, minimal_config):
        """Verify logits shape."""
        outputs = model_minimal(sample_batch_minimal['input_ids'])

        batch, seq = sample_batch_minimal['input_ids'].shape
        expected_shape = (batch, seq, minimal_config.vocab_size)

        assert outputs['logits'].shape == expected_shape, \
            f"Logits shape: {outputs['logits'].shape} vs {expected_shape}"

    def test_no_nan_in_outputs(self, model_minimal, sample_batch_minimal):
        """Verify no NaN in any output."""
        outputs = model_minimal(
            sample_batch_minimal['input_ids'],
            labels=sample_batch_minimal['labels']
        )

        check_no_nan_inf(outputs['logits'], "logits")
        check_no_nan_inf(outputs['loss'], "loss")
        check_no_nan_inf(outputs['hidden_states'], "hidden_states")

    def test_loss_reasonable_range(self, model_minimal, sample_batch_minimal, minimal_config):
        """Verify loss is in reasonable range for random init."""
        outputs = model_minimal(
            sample_batch_minimal['input_ids'],
            labels=sample_batch_minimal['labels']
        )

        loss = outputs['loss'].item()

        # Random init should have loss ~ ln(vocab_size)
        expected_random_loss = torch.log(torch.tensor(float(minimal_config.vocab_size))).item()

        # Should be within reasonable range (0.5x to 2x expected)
        assert 0.5 * expected_random_loss < loss < 2 * expected_random_loss, \
            f"Loss {loss} outside expected range for vocab_size={minimal_config.vocab_size}"


class TestFullBackward:
    """Test complete backward pass."""

    def test_backward_completes(self, model_minimal, sample_batch_minimal):
        """Verify backward pass completes without error."""
        outputs = model_minimal(
            sample_batch_minimal['input_ids'],
            labels=sample_batch_minimal['labels']
        )

        outputs['loss'].backward()

        # Should complete without error
        assert True

    def test_all_parameters_have_gradients(self, model_minimal, sample_batch_minimal):
        """Verify all trainable parameters receive gradients."""
        outputs = model_minimal(
            sample_batch_minimal['input_ids'],
            labels=sample_batch_minimal['labels']
        )
        outputs['loss'].backward()

        params_without_grad = []
        for name, param in model_minimal.named_parameters():
            if param.requires_grad and param.grad is None:
                params_without_grad.append(name)

        # Some parameters might not get gradients (e.g., unused experts)
        # But core parameters should have gradients
        core_params = ['embed_tokens', 'lm_head', 'norm']
        for core in core_params:
            core_missing = [p for p in params_without_grad if core in p]
            assert len(core_missing) == 0, f"Core param missing gradient: {core_missing}"

    def test_gradient_health(self, model_minimal, sample_batch_minimal):
        """Verify gradient magnitudes are healthy."""
        outputs = model_minimal(
            sample_batch_minimal['input_ids'],
            labels=sample_batch_minimal['labels']
        )
        outputs['loss'].backward()

        grad_health = check_gradient_health(model_minimal)

        assert grad_health['nan_count'] == 0, f"NaN gradients: {grad_health['nan_count']}"
        assert grad_health['inf_count'] == 0, f"Inf gradients: {grad_health['inf_count']}"
        assert grad_health['total_norm'] < 1000, f"Gradient norm too high: {grad_health['total_norm']}"


class TestLossComponents:
    """Test all loss components."""

    def test_main_loss(self, model_minimal, sample_batch_minimal):
        """Verify main cross-entropy loss."""
        outputs = model_minimal(
            sample_batch_minimal['input_ids'],
            labels=sample_batch_minimal['labels']
        )

        assert outputs['main_loss'] >= 0
        assert outputs['main_loss'] <= outputs['loss']

    def test_mtp_loss(self, model_minimal, sample_batch_minimal):
        """Verify MTP loss is included."""
        if model_minimal.mtp is None:
            pytest.skip("Model has no MTP")

        outputs = model_minimal(
            sample_batch_minimal['input_ids'],
            labels=sample_batch_minimal['labels']
        )

        assert 'mtp_loss' in outputs, "MTP loss should be returned"
        assert outputs['mtp_loss'] >= 0

    def test_aux_loss(self, model_minimal, sample_batch_minimal):
        """Verify auxiliary loss from MoE."""
        outputs = model_minimal(
            sample_batch_minimal['input_ids'],
            labels=sample_batch_minimal['labels']
        )

        # Aux loss might be 0 if alpha is 0
        if 'aux_loss' in outputs:
            assert outputs['aux_loss'] >= 0


class TestMemoryEfficiency:
    """Test memory efficiency claims."""

    def test_kv_cache_compression(self, model_minimal, sample_batch_minimal, minimal_config):
        """Verify KV cache is compressed."""
        model_minimal.eval()

        with torch.no_grad():
            outputs = model_minimal(
                sample_batch_minimal['input_ids'],
                use_cache=True
            )

        cache = outputs['past_key_values']
        cache_bytes = compute_kv_cache_size(cache)

        batch, seq = sample_batch_minimal['input_ids'].shape
        mha_bytes = compute_mha_cache_size(
            num_layers=minimal_config.num_layers,
            seq_len=seq,
            num_heads=minimal_config.num_heads,
            head_dim=minimal_config.head_dim,
            batch_size=batch
        )

        compression = mha_bytes / cache_bytes
        print(f"KV cache compression: {compression:.1f}x")
        print(f"MLA cache: {cache_bytes / 1024:.1f} KB")
        print(f"MHA equivalent: {mha_bytes / 1024:.1f} KB")

        assert compression > 10, f"Expected >10x compression, got {compression:.1f}x"


class TestParameterCount:
    """Test parameter counting."""

    def test_actual_vs_estimated_total(self, model_minimal, minimal_config):
        """Verify actual params match estimate."""
        actual = sum(p.numel() for p in model_minimal.parameters())
        estimated = minimal_config.estimated_total_params

        ratio = actual / estimated
        print(f"Total params: actual={actual:,}, estimated={estimated:,}, ratio={ratio:.2f}")

        assert 0.8 < ratio < 1.2, f"Param count mismatch: ratio={ratio:.2f}"

    def test_trainable_params(self, model_minimal):
        """Verify all params are trainable by default."""
        total = sum(p.numel() for p in model_minimal.parameters())
        trainable = sum(p.numel() for p in model_minimal.parameters() if p.requires_grad)

        assert trainable == total, "All params should be trainable by default"

    @pytest.mark.slow
    def test_700m_param_count(self, config_700m, device):
        """Test 700M configuration param count."""
        model = NanoSeekModel(config_700m).to(device)

        total = sum(p.numel() for p in model.parameters())
        active = config_700m.estimated_active_params

        print(f"NanoSeek-700M: total={total:,}, active={active:,}")

        # Active should be in reasonable range for the 700M-1B class
        assert 500_000_000 < active < 1_500_000_000, f"Active params: {active:,}"
        # Total should be 3-6B (5x expansion ratio)
        assert 2_000_000_000 < total < 7_000_000_000, f"Total params: {total:,}"


class TestGeneration:
    """Test text generation."""

    def test_generate_basic(self, model_minimal, minimal_config, device):
        """Test basic generation."""
        model_minimal.eval()

        prompt = torch.randint(0, minimal_config.vocab_size, (1, 10), device=device)

        with torch.no_grad():
            generated = model_minimal.generate(prompt, max_new_tokens=5)

        assert generated.shape == (1, 15), f"Generated shape: {generated.shape}"
        assert generated[:, :10].equal(prompt), "Prompt should be preserved"

    def test_generate_simple_streaming(self, model_minimal, minimal_config, device):
        """Test streaming generation."""
        model_minimal.eval()

        prompt = list(torch.randint(0, minimal_config.vocab_size, (10,)).tolist())
        tokens = []

        for token in model_minimal.generate_simple(prompt, max_tokens=5, temperature=0.8):
            tokens.append(token)

        assert len(tokens) == 5, f"Generated {len(tokens)} tokens"

    def test_generate_cached_streaming(self, model_minimal, minimal_config, device):
        """Test cached streaming generation."""
        model_minimal.eval()

        prompt = list(torch.randint(0, minimal_config.vocab_size, (10,)).tolist())
        tokens = []

        for token in model_minimal.generate_cached(prompt, max_tokens=5, temperature=0.8):
            tokens.append(token)

        assert len(tokens) == 5


class TestCheckpoint:
    """Test checkpoint save/load."""

    def test_save_load_state_dict(self, model_minimal, sample_batch_minimal, device):
        """Test saving and loading state dict."""
        # Get initial output
        with torch.no_grad():
            outputs1 = model_minimal(sample_batch_minimal['input_ids'])

        # Save state
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save({
                'model_state_dict': model_minimal.state_dict(),
                'tokens_processed': model_minimal.tokens_processed.item(),
            }, f.name)
            checkpoint_path = f.name

        try:
            # Create new model and load
            new_config = model_minimal.config
            new_model = NanoSeekModel(new_config).to(device)

            checkpoint = torch.load(checkpoint_path)
            new_model.load_state_dict(checkpoint['model_state_dict'])

            # Verify same output
            with torch.no_grad():
                outputs2 = new_model(sample_batch_minimal['input_ids'])

            assert torch.allclose(outputs1['logits'], outputs2['logits']), \
                "Loaded model should produce same output"

        finally:
            os.unlink(checkpoint_path)


class TestTrainingSimulation:
    """Simulate training loop."""

    def test_training_step(self, model_minimal, sample_batch_minimal, device):
        """Test single training step."""
        model_minimal.train()

        optimizer = torch.optim.AdamW(model_minimal.parameters(), lr=1e-4)

        outputs = model_minimal(
            sample_batch_minimal['input_ids'],
            labels=sample_batch_minimal['labels']
        )

        loss = outputs['loss']
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model_minimal.parameters(), 1.0)

        optimizer.step()
        optimizer.zero_grad()

        # Should complete without error
        assert True

    def test_moe_bias_update(self, model_minimal, sample_batch_minimal):
        """Test MoE load balance bias update."""
        model_minimal.train()

        # Forward pass
        outputs = model_minimal(
            sample_batch_minimal['input_ids'],
            labels=sample_batch_minimal['labels']
        )

        # Update bias
        model_minimal.update_load_balance_bias()

        # Get stats
        stats = model_minimal.get_expert_load_stats()

        # Should have stats for MoE layers
        assert len(stats) > 0, "Should have MoE layer stats"

    def test_tokens_processed_tracking(self, model_minimal, sample_batch_minimal):
        """Test token counter."""
        initial = model_minimal.tokens_processed.item()

        batch, seq = sample_batch_minimal['input_ids'].shape
        model_minimal.update_tokens_processed(batch * seq)

        assert model_minimal.tokens_processed.item() == initial + batch * seq


class TestGradientCheckpointing:
    """Test gradient checkpointing."""

    def test_gradient_checkpointing_forward(self, model_minimal, sample_batch_minimal):
        """Test forward with gradient checkpointing."""
        model_minimal.gradient_checkpointing = True
        model_minimal.train()

        outputs = model_minimal(
            sample_batch_minimal['input_ids'],
            labels=sample_batch_minimal['labels']
        )

        check_no_nan_inf(outputs['loss'], "loss with checkpointing")

    def test_gradient_checkpointing_backward(self, model_minimal, sample_batch_minimal):
        """Test backward with gradient checkpointing."""
        model_minimal.gradient_checkpointing = True
        model_minimal.train()

        outputs = model_minimal(
            sample_batch_minimal['input_ids'],
            labels=sample_batch_minimal['labels']
        )

        outputs['loss'].backward()

        grad_health = check_gradient_health(model_minimal)
        assert grad_health['nan_count'] == 0
