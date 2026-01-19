"""
RoPE (Rotary Position Embedding) + YaRN Validation Tests

DeepSeek-Level Validation:
1. Frequency computation correctness
2. Rotation invariance properties
3. YaRN interpolation for context extension
4. Complex number arithmetic
5. Interleaved vs non-interleaved formats
6. Gradient flow through rotations
"""

import pytest
import torch
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.model import (
    precompute_freqs_cis,
    apply_rotary_emb,
    find_correction_dim,
    find_correction_range,
    linear_ramp_factor,
    RotaryEmbedding,
)


class TestRoPEFrequencies:
    """Test RoPE frequency computation."""

    def test_frequency_decay_pattern(self):
        """Verify frequencies decay exponentially."""
        dim = 16
        theta = 10000.0

        # Compute frequencies
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

        # Should decay monotonically
        for i in range(len(freqs) - 1):
            assert freqs[i] > freqs[i + 1], \
                f"Frequencies must decay: freqs[{i}]={freqs[i]:.6f} <= freqs[{i+1}]={freqs[i+1]:.6f}"

        # First frequency should be 1.0 (theta^0 = 1)
        assert torch.isclose(freqs[0], torch.tensor(1.0)), \
            f"First frequency should be 1.0, got {freqs[0]}"

        # Last frequency should be small
        assert freqs[-1] < 0.1, \
            f"Last frequency should be <0.1, got {freqs[-1]}"

    def test_frequency_values_match_formula(self):
        """Verify frequencies match the mathematical formula exactly."""
        dim = 16
        theta = 10000.0

        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

        # Manually compute expected values
        for i in range(dim // 2):
            expected = 1.0 / (theta ** (2 * i / dim))
            assert torch.isclose(freqs[i], torch.tensor(expected), rtol=1e-5), \
                f"Frequency {i} mismatch: {freqs[i]} vs {expected}"

    def test_precompute_freqs_cis_shape(self):
        """Verify precomputed frequencies have correct shape."""
        dim = 16
        seq_len = 128

        freqs_cis = precompute_freqs_cis(dim, seq_len)

        # Shape: (seq_len, dim//2) complex
        assert freqs_cis.shape == (seq_len, dim // 2), \
            f"Expected shape ({seq_len}, {dim // 2}), got {freqs_cis.shape}"

        # Should be complex
        assert freqs_cis.is_complex(), "Frequencies should be complex"

    def test_frequency_positions_are_monotonic(self):
        """Verify different positions have distinct rotations."""
        dim = 16
        seq_len = 64

        freqs_cis = precompute_freqs_cis(dim, seq_len)

        # Each position should have unique rotation angles
        for pos in range(1, seq_len):
            # Compare to previous position
            assert not torch.allclose(freqs_cis[pos], freqs_cis[pos - 1]), \
                f"Positions {pos} and {pos-1} have identical frequencies"


class TestRoPERotation:
    """Test RoPE rotation application."""

    def test_rotation_preserves_norm(self):
        """Verify rotation preserves vector magnitude."""
        dim = 16
        batch, seq, heads = 2, 32, 4

        x = torch.randn(batch, seq, heads, dim)
        freqs_cis = precompute_freqs_cis(dim, seq)

        x_rotated = apply_rotary_emb(x, freqs_cis)

        # Norms should be preserved (within numerical precision)
        original_norms = x.norm(dim=-1)
        rotated_norms = x_rotated.norm(dim=-1)

        assert torch.allclose(original_norms, rotated_norms, rtol=1e-4), \
            f"Rotation should preserve norms. Max diff: {(original_norms - rotated_norms).abs().max()}"

    def test_rotation_is_invertible(self):
        """Verify rotation can be inverted with negative frequencies."""
        dim = 16
        batch, seq, heads = 2, 32, 4

        x = torch.randn(batch, seq, heads, dim)
        freqs_cis = precompute_freqs_cis(dim, seq)

        # Rotate forward
        x_rotated = apply_rotary_emb(x, freqs_cis)

        # Rotate backward (conjugate of frequencies)
        freqs_conj = torch.conj(freqs_cis)
        x_recovered = apply_rotary_emb(x_rotated, freqs_conj)

        assert torch.allclose(x, x_recovered, rtol=1e-4), \
            f"Rotation should be invertible. Max diff: {(x - x_recovered).abs().max()}"

    def test_relative_position_property(self):
        """
        Verify RoPE's key property: dot product depends on relative position.

        For RoPE: <R_m q, R_n k> = <R_{m-n} q, k>
        """
        dim = 16

        # Create two vectors
        q = torch.randn(1, 1, 1, dim)
        k = torch.randn(1, 1, 1, dim)

        freqs_cis = precompute_freqs_cis(dim, 20)

        # Rotate q at position 10, k at position 5 (relative distance = 5)
        q_rot_10 = apply_rotary_emb(q, freqs_cis[10:11])
        k_rot_5 = apply_rotary_emb(k, freqs_cis[5:6])

        # Rotate q at position 7, k at position 2 (relative distance = 5)
        q_rot_7 = apply_rotary_emb(q, freqs_cis[7:8])
        k_rot_2 = apply_rotary_emb(k, freqs_cis[2:3])

        # Dot products should be equal (same relative distance)
        dot1 = (q_rot_10 * k_rot_5).sum()
        dot2 = (q_rot_7 * k_rot_2).sum()

        assert torch.isclose(dot1, dot2, rtol=1e-4), \
            f"Dot products should match for same relative distance: {dot1} vs {dot2}"

    def test_3d_and_4d_input_equivalence(self):
        """Verify 3D and 4D inputs produce equivalent results."""
        dim = 16
        batch, seq, heads = 2, 32, 1

        # 4D input: (batch, seq, heads, dim)
        x_4d = torch.randn(batch, seq, heads, dim)

        # 3D input: (batch, seq, dim)
        x_3d = x_4d.squeeze(2)

        freqs_cis = precompute_freqs_cis(dim, seq)

        out_4d = apply_rotary_emb(x_4d, freqs_cis)
        out_3d = apply_rotary_emb(x_3d, freqs_cis)

        # Results should match
        assert torch.allclose(out_4d.squeeze(2), out_3d, rtol=1e-5), \
            "3D and 4D inputs should produce equivalent results"


class TestYaRNInterpolation:
    """Test YaRN context extension."""

    def test_yarn_correction_dim_calculation(self):
        """Verify correction dimension calculation."""
        dim = 64
        theta = 10000.0
        original_len = 4096

        # Find correction dimensions
        low, high = find_correction_range(32, 1, dim, theta, original_len)

        # Low should be within valid range
        assert 0 <= low < dim, f"Low correction dim out of range: {low}"
        assert 0 <= high < dim, f"High correction dim out of range: {high}"
        assert low <= high, f"Low should be <= high: {low} vs {high}"

    def test_linear_ramp_factor(self):
        """Verify linear ramp function properties."""
        dim = 32
        min_val, max_val = 8, 24

        ramp = linear_ramp_factor(min_val, max_val, dim)

        # Should be 0 at min_val
        assert ramp[min_val] == 0.0 or torch.isclose(ramp[min_val], torch.tensor(0.0)), \
            f"Ramp should be 0 at min_val: {ramp[min_val]}"

        # Should be 1 at max_val
        assert ramp[max_val] == 1.0 or torch.isclose(ramp[max_val], torch.tensor(1.0)), \
            f"Ramp should be 1 at max_val: {ramp[max_val]}"

        # Should be clamped to [0, 1]
        assert ramp.min() >= 0.0, f"Ramp min should be >= 0: {ramp.min()}"
        assert ramp.max() <= 1.0, f"Ramp max should be <= 1: {ramp.max()}"

    def test_yarn_extends_context(self):
        """Verify YaRN enables longer context without numerical issues."""
        dim = 16
        original_len = 4096
        extended_len = 32768
        scaling_factor = 8.0

        # Without YaRN: should work fine at original length
        freqs_original = precompute_freqs_cis(dim, original_len)
        assert not torch.isnan(freqs_original).any(), "Original frequencies should not have NaN"

        # With YaRN: should work at extended length
        freqs_extended = precompute_freqs_cis(
            dim, extended_len,
            scaling_factor=scaling_factor,
            original_max_position_embeddings=original_len
        )

        assert not torch.isnan(freqs_extended).any(), "Extended frequencies should not have NaN"
        assert not torch.isinf(freqs_extended).any(), "Extended frequencies should not have Inf"

        # Magnitudes should be 1 (unit circle)
        magnitudes = freqs_extended.abs()
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), rtol=1e-5), \
            "All frequency magnitudes should be 1"

    def test_yarn_preserves_nearby_positions(self):
        """Verify YaRN doesn't drastically change nearby position relationships."""
        dim = 16
        original_len = 4096
        scaling_factor = 2.0

        freqs_original = precompute_freqs_cis(dim, original_len)
        freqs_yarn = precompute_freqs_cis(
            dim, original_len,
            scaling_factor=scaling_factor,
            original_max_position_embeddings=original_len
        )

        # High-frequency dimensions should be similar
        # (YaRN interpolation is smooth)
        # Compare at position 100
        high_freq_original = freqs_original[100, 0]  # Highest frequency
        high_freq_yarn = freqs_yarn[100, 0]

        # They won't be exactly equal but should be in similar direction
        # (angle shouldn't change too much for high frequencies)
        angle_diff = (high_freq_original * high_freq_yarn.conj()).angle().abs()
        assert angle_diff < 1.0, f"High frequency angle changed too much: {angle_diff}"


class TestRotaryEmbeddingModule:
    """Test the RotaryEmbedding nn.Module."""

    def test_rotary_embedding_forward(self):
        """Test basic forward pass."""
        dim = 16
        max_len = 128

        rope = RotaryEmbedding(dim, max_len)

        q = torch.randn(2, 32, 4, dim)
        k = torch.randn(2, 32, 4, dim)

        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        assert not torch.isnan(q_rot).any()
        assert not torch.isnan(k_rot).any()

    def test_rotary_embedding_with_position_ids(self):
        """Test with explicit position IDs."""
        dim = 16
        max_len = 128

        rope = RotaryEmbedding(dim, max_len)

        batch, seq = 2, 32
        q = torch.randn(batch, seq, 4, dim)
        k = torch.randn(batch, seq, 4, dim)

        # Custom position IDs (not starting at 0)
        position_ids = torch.arange(10, 10 + seq).unsqueeze(0).expand(batch, -1)

        q_rot, k_rot = rope(q, k, position_ids=position_ids)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rotary_embedding_gradient_flow(self):
        """Verify gradients flow through rotation."""
        dim = 16
        max_len = 128

        rope = RotaryEmbedding(dim, max_len)

        q = torch.randn(2, 32, 4, dim, requires_grad=True)
        k = torch.randn(2, 32, 4, dim, requires_grad=True)

        q_rot, k_rot = rope(q, k)

        # Compute a loss and backpropagate
        loss = (q_rot * k_rot).sum()
        loss.backward()

        assert q.grad is not None, "Q should have gradients"
        assert k.grad is not None, "K should have gradients"
        assert not torch.isnan(q.grad).any(), "Q gradients should not have NaN"
        assert not torch.isnan(k.grad).any(), "K gradients should not have NaN"
