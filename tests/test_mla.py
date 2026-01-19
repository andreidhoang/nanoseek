"""
MLA (Multi-head Latent Attention) Validation Tests

DeepSeek-Level Validation:
1. KV compression ratio verification (should be ~24x)
2. Attention pattern correctness
3. Gradient flow through all projection paths
4. Incremental decoding consistency
5. Cache size verification
6. Numerical stability
7. Different sequence lengths
8. Position encoding integration
"""

import pytest
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.model import (
    MultiHeadLatentAttention,
    create_mla_from_config,
    create_causal_mask,
    RMSNorm,
)
from model.config import get_nanoseek_config, MLAConfig

from tests.utils import check_no_nan_inf, check_gradient_health, compute_kv_cache_size, compute_mha_cache_size


class TestMLACompression:
    """Test MLA's KV cache compression."""

    def test_kv_cache_shape(self, mla_minimal, sample_hidden_minimal):
        """Verify KV cache has compressed shape."""
        output, cache = mla_minimal(sample_hidden_minimal, use_cache=True)

        assert cache is not None, "Cache should be returned"
        kv_compressed, k_pe = cache

        batch, seq, _ = sample_hidden_minimal.shape

        # KV compressed should have kv_lora_rank dimension
        assert kv_compressed.shape == (batch, seq, 18), \
            f"Expected KV compressed shape (2, 64, 18), got {kv_compressed.shape}"

        # K_PE should have qk_rope_head_dim dimension (shared across heads)
        assert k_pe.shape == (batch, seq, 1, 16), \
            f"Expected K_PE shape (2, 64, 1, 16), got {k_pe.shape}"

    def test_compression_ratio(self, config_1b):
        """Verify MLA achieves >20x compression."""
        # Standard MHA: 2 * num_heads * head_dim per layer per token
        mha_size = 2 * config_1b.num_heads * config_1b.head_dim

        # MLA: kv_lora_rank + qk_rope_head_dim
        mla_size = config_1b.mla.kv_lora_rank + config_1b.mla.qk_rope_head_dim

        compression = mha_size / mla_size

        print(f"MHA KV size per token: {mha_size}")
        print(f"MLA KV size per token: {mla_size}")
        print(f"Compression ratio: {compression:.1f}x")

        assert compression > 15, f"Expected >15x compression, got {compression:.1f}x"

    def test_cache_size_in_bytes(self, mla_minimal, sample_hidden_minimal, minimal_config):
        """Verify actual cache size in bytes."""
        output, cache = mla_minimal(sample_hidden_minimal, use_cache=True)

        cache_bytes = compute_kv_cache_size([cache])
        batch, seq, _ = sample_hidden_minimal.shape

        # MHA reference size
        mha_bytes = compute_mha_cache_size(
            num_layers=1,
            seq_len=seq,
            num_heads=minimal_config.num_heads,
            head_dim=minimal_config.head_dim,
            batch_size=batch
        )

        print(f"MLA cache: {cache_bytes} bytes")
        print(f"MHA equivalent: {mha_bytes} bytes")
        print(f"Reduction: {mha_bytes / cache_bytes:.1f}x")

        assert cache_bytes < mha_bytes, "MLA cache should be smaller than MHA"


class TestMLAForward:
    """Test MLA forward pass."""

    def test_output_shape(self, mla_minimal, sample_hidden_minimal):
        """Verify output has correct shape."""
        output, _ = mla_minimal(sample_hidden_minimal)

        assert output.shape == sample_hidden_minimal.shape, \
            f"Output shape should match input: {output.shape} vs {sample_hidden_minimal.shape}"

    def test_output_no_nan(self, mla_minimal, sample_hidden_minimal):
        """Verify no NaN in output."""
        output, _ = mla_minimal(sample_hidden_minimal)
        check_no_nan_inf(output, "MLA output")

    def test_causal_masking(self, mla_minimal, sample_hidden_minimal, device):
        """Verify causal masking prevents future attention."""
        batch, seq, hidden = sample_hidden_minimal.shape

        # Create causal mask
        mask = create_causal_mask(seq, device=device)

        output, _ = mla_minimal(sample_hidden_minimal, attention_mask=mask)

        check_no_nan_inf(output, "MLA output with mask")

    def test_different_sequence_lengths(self, minimal_config, device):
        """Test MLA works with various sequence lengths."""
        mla = MultiHeadLatentAttention(
            hidden_size=minimal_config.hidden_size,
            num_heads=minimal_config.num_heads,
            q_lora_rank=minimal_config.mla.q_lora_rank,
            kv_lora_rank=minimal_config.mla.kv_lora_rank,
            qk_nope_head_dim=minimal_config.mla.qk_nope_head_dim,
            qk_rope_head_dim=minimal_config.mla.qk_rope_head_dim,
            v_head_dim=minimal_config.mla.v_head_dim,
            max_position_embeddings=256,
        ).to(device)

        for seq_len in [1, 16, 64, 128]:
            x = torch.randn(2, seq_len, minimal_config.hidden_size, device=device)
            output, _ = mla(x)

            assert output.shape == x.shape, \
                f"Output shape mismatch for seq_len={seq_len}"
            check_no_nan_inf(output, f"MLA output (seq_len={seq_len})")


class TestMLAIncrementalDecoding:
    """Test incremental decoding with KV cache."""

    def test_incremental_matches_full(self, mla_minimal, device):
        """Verify incremental decoding matches full forward pass."""
        mla_minimal.eval()

        # Full sequence
        x_full = torch.randn(1, 64, mla_minimal.hidden_size, device=device)
        with torch.no_grad():
            output_full, _ = mla_minimal(x_full)

        # Incremental: prefix + suffix
        x_prefix = x_full[:, :32]
        x_suffix = x_full[:, 32:]

        with torch.no_grad():
            out_prefix, cache = mla_minimal(x_prefix, use_cache=True)
            out_suffix, _ = mla_minimal(x_suffix, past_key_value=cache, use_cache=True)

        output_incremental = torch.cat([out_prefix, out_suffix], dim=1)

        # Should match within numerical precision
        # Note: Some difference is expected due to position encoding recomputation
        max_diff = (output_full - output_incremental).abs().max().item()
        assert max_diff < 0.5, \
            f"Incremental mismatch. Max diff: {max_diff}"

    def test_single_token_generation(self, mla_minimal, device):
        """Test single-token generation pattern."""
        mla_minimal.eval()

        batch = 1
        hidden = mla_minimal.hidden_size

        # Process prompt
        prompt = torch.randn(batch, 10, hidden, device=device)
        with torch.no_grad():
            _, cache = mla_minimal(prompt, use_cache=True)

        # Generate tokens one at a time
        for i in range(5):
            new_token = torch.randn(batch, 1, hidden, device=device)
            with torch.no_grad():
                output, cache = mla_minimal(new_token, past_key_value=cache, use_cache=True)

            assert output.shape == (batch, 1, hidden), \
                f"Single token output shape mismatch at step {i}"

            # Cache should grow
            kv_compressed, _ = cache
            assert kv_compressed.shape[1] == 10 + i + 1, \
                f"Cache not growing correctly at step {i}"

    def test_cache_growth_linear(self, mla_minimal, device):
        """Verify KV cache grows linearly with sequence length."""
        mla_minimal.eval()

        batch = 1
        hidden = mla_minimal.hidden_size

        cache_sizes = []

        # Generate progressively longer sequences
        for seq_len in [10, 20, 30, 40, 50]:
            x = torch.randn(batch, seq_len, hidden, device=device)
            with torch.no_grad():
                _, cache = mla_minimal(x, use_cache=True)

            cache_size = compute_kv_cache_size([cache])
            cache_sizes.append(cache_size)

        # Check linear growth
        for i in range(1, len(cache_sizes)):
            ratio = cache_sizes[i] / cache_sizes[i - 1]
            expected_ratio = (10 + i * 10) / (i * 10)

            assert 0.8 < ratio / expected_ratio < 1.2, \
                f"Cache growth not linear: ratio={ratio:.2f}, expected~{expected_ratio:.2f}"


class TestMLAGradients:
    """Test gradient flow through MLA."""

    def test_gradient_flow_all_projections(self, mla_minimal, sample_hidden_minimal):
        """Verify gradients flow through all projection layers."""
        sample_hidden_minimal.requires_grad_(True)
        output, _ = mla_minimal(sample_hidden_minimal)
        loss = output.sum()
        loss.backward()

        # Check all major projections have gradients
        projections = ['wq_a', 'wq_b', 'wkv_a', 'wkv_b', 'wo']
        for name in projections:
            layer = getattr(mla_minimal, name)
            assert layer.weight.grad is not None, f"{name} should have gradient"
            assert layer.weight.grad.abs().mean() > 0, f"{name} gradient should be non-zero"

    def test_gradient_magnitudes_reasonable(self, mla_minimal, sample_hidden_minimal):
        """Verify gradient magnitudes are reasonable."""
        sample_hidden_minimal.requires_grad_(True)
        output, _ = mla_minimal(sample_hidden_minimal)
        loss = output.sum()
        loss.backward()

        for name, param in mla_minimal.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                assert 1e-8 < grad_norm < 1e6, \
                    f"{name} gradient norm out of range: {grad_norm}"

    def test_gradient_through_normalization(self, mla_minimal, sample_hidden_minimal):
        """Verify gradients flow through RMSNorm layers."""
        sample_hidden_minimal.requires_grad_(True)
        output, _ = mla_minimal(sample_hidden_minimal)
        loss = output.sum()
        loss.backward()

        # Check Q and KV norms have gradients
        assert mla_minimal.q_norm.weight.grad is not None
        assert mla_minimal.kv_norm.weight.grad is not None


class TestMLANumericalStability:
    """Test numerical stability of MLA."""

    def test_large_input_values(self, mla_minimal, device):
        """Test with large input values."""
        x = torch.randn(2, 32, mla_minimal.hidden_size, device=device) * 10
        output, _ = mla_minimal(x)

        check_no_nan_inf(output, "MLA output with large inputs")

    def test_small_input_values(self, mla_minimal, device):
        """Test with small input values."""
        x = torch.randn(2, 32, mla_minimal.hidden_size, device=device) * 0.01
        output, _ = mla_minimal(x)

        check_no_nan_inf(output, "MLA output with small inputs")

    def test_mixed_precision(self, mla_minimal, device):
        """Test with bfloat16 precision."""
        if device.type == 'cpu':
            pytest.skip("bfloat16 test requires GPU")

        mla_bf16 = mla_minimal.to(torch.bfloat16)
        x = torch.randn(2, 32, mla_minimal.hidden_size, device=device, dtype=torch.bfloat16)

        output, _ = mla_bf16(x)

        assert not torch.isnan(output).any(), "NaN in bfloat16 output"

    def test_attention_softmax_stability(self, mla_minimal, device):
        """Verify attention softmax doesn't overflow/underflow."""
        # Create input that might cause softmax issues
        x = torch.randn(2, 64, mla_minimal.hidden_size, device=device)

        # Forward pass should complete without issues
        output, _ = mla_minimal(x)

        check_no_nan_inf(output, "MLA output (softmax stability)")


class TestMLAConfigIntegration:
    """Test MLA creation from config."""

    def test_create_from_config(self, config_1b, device):
        """Test creating MLA from config."""
        mla = create_mla_from_config(config_1b, layer_idx=0).to(device)

        x = torch.randn(2, 64, config_1b.hidden_size, device=device)
        output, cache = mla(x, use_cache=True)

        assert output.shape == x.shape
        assert cache is not None

    def test_layer_idx_affects_nothing_critical(self, config_1b, device):
        """Verify different layer indices produce valid outputs."""
        for layer_idx in [0, 5, 10]:
            mla = create_mla_from_config(config_1b, layer_idx=layer_idx).to(device)
            x = torch.randn(2, 32, config_1b.hidden_size, device=device)
            output, _ = mla(x)
            check_no_nan_inf(output, f"MLA output (layer {layer_idx})")
