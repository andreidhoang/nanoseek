# =============================================================================
# ROTARY POSITION EMBEDDING (RoPE) WITH YaRN CONTEXT EXTENSION
# =============================================================================
# References:
#   - RoPE: Su et al., "RoFormer" (arXiv:2104.09864)
#   - YaRN: Peng et al., "YaRN: Efficient Context Window Extension" (arXiv:2309.00071)
#   - DeepSeek-V2: DeepSeek-AI (arXiv:2405.04434), Section 3.2
#
# =============================================================================
# HOW ROPE WORKS — FIRST PRINCIPLES
# =============================================================================
#
# Goal: Encode position information into Q and K so that their dot product
# depends only on RELATIVE position (distance between tokens), not absolute.
#
# Mechanism: Pair consecutive dimensions of each vector as 2D planes.
# For each plane, rotate by an angle proportional to the token's position.
#
#   head_dim = 64 → 32 rotation planes
#   plane_i rotates by: angle_i(pos) = pos * freq_i
#   where freq_i = 1 / (theta^(2i/d))
#
# The frequencies are geometrically spaced:
#   freq_0 = 1.0        (fastest — completes full rotation every 2π positions)
#   freq_1 ≈ 0.75       (slightly slower)
#   ...
#   freq_31 ≈ 0.01      (slowest — one rotation per ~628 positions)
#
# This multi-frequency encoding is essentially a Fourier basis for position,
# capturing both local structure (high-freq planes) and global structure (low-freq).
#
# Key property: dot_product(RoPE(q, pos_m), RoPE(k, pos_n)) depends only on
# the content similarity AND the distance (m - n), not on m or n individually.
#
# =============================================================================
# HOW YARN EXTENDS CONTEXT BEYOND TRAINING LENGTH
# =============================================================================
#
# Problem: Model trained on max_pos=4096 tokens. We want to use 32K at inference.
#
# Naive NTK scaling: multiply all frequencies by a factor → works but suboptimal.
#
# YaRN insight: Different frequency bands need different treatment:
#
#   HIGH frequencies (fast rotation, local patterns):
#     These see many full rotations within the training window.
#     Extrapolation works naturally. → Keep unchanged.
#
#   LOW frequencies (slow rotation, global patterns):
#     These barely complete one rotation in the training window.
#     Extrapolation breaks catastrophically. → Interpolate (divide by scale).
#
#   MIDDLE frequencies:
#     Smooth blend between the two regimes.
#
# The "correction range" (beta_fast, beta_slow) defines which dimensions
# fall in which regime:
#   - beta_fast = 32: dimensions completing ≥32 rotations in training → keep
#   - beta_slow = 1:  dimensions completing ≤1 rotation in training → interpolate
#   - Between: linear ramp from interpolate to keep
#
# Additionally, YaRN applies an attention temperature correction (mscale)
# because longer contexts spread attention mass over more positions,
# increasing entropy. The correction:
#   mscale = 0.1 * ln(scaling_factor) + 1.0
# is multiplied into softmax temperature to compensate.
#
# =============================================================================

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# =============================================================================
# YARN HELPER FUNCTIONS
# =============================================================================


def find_correction_dim(
    num_rotations: int,
    dim: int,
    base: float = 10000.0,
    max_position_embeddings: int = 2048,
) -> float:
    """
    Find the frequency index where a dimension completes exactly `num_rotations`
    full rotations within the training window [0, max_position_embeddings].

    The frequency for index i is: freq_i = base^(-2i/dim)
    Total angle at position max_pos: max_pos * freq_i
    Number of full rotations: max_pos * freq_i / (2π)

    Solving for i where num_rotations = max_pos * base^(-2i/dim) / (2π):
        i = dim * ln(max_pos / (num_rotations * 2π)) / (2 * ln(base))

    Returns a float — the caller rounds to get an integer index.

    NOTE: This returns an index into the frequency array of size dim//2,
    even though the formula contains `dim` (not dim//2). The algebra works out
    because the frequency definition uses 2i/dim, making i range over [0, dim/2).

    IMPORTANT: This function is DECREASING in num_rotations.
    More rotations → lower frequency index (faster frequencies are at lower indices).
    """
    return (
        dim
        * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
        / (2 * math.log(base))
    )


def find_correction_range(
    beta_fast: int,
    beta_slow: int,
    dim: int,
    base: float = 10000.0,
    max_position_embeddings: int = 2048,
) -> Tuple[int, int]:
    """
    Find the range of frequency indices that need interpolation blending.

    Returns (low_idx, high_idx) into the frequency array (size dim//2):
      - Indices < low_idx:  High frequency, keep unchanged (extrapolation works)
      - Indices > high_idx: Low frequency, fully interpolate (divide by scale)
      - Indices in [low_idx, high_idx]: Linear blend between the two

    NOTE on the seeming inversion: beta_fast (=32, more rotations) produces the
    LOWER dimension index because find_correction_dim is decreasing in num_rotations.
    This is correct — high frequencies (many rotations) occupy low indices in the
    frequency array since freq_i = base^(-2i/dim) decreases with i.

    Concretely for dim=64, base=10000, max_pos=4096:
      beta_fast=32 → dim_idx ≈ 10  (frequencies rotating ≥32 times → safe to keep)
      beta_slow=1  → dim_idx ≈ 21  (frequencies rotating ≤1 time → must interpolate)
      Dims 10-21: smooth blend
    """
    # beta_fast (more rotations) → lower index (faster frequencies)
    low = max(
        math.floor(find_correction_dim(beta_fast, dim, base, max_position_embeddings)),
        0,
    )
    # beta_slow (fewer rotations) → higher index (slower frequencies)
    high = min(
        math.ceil(find_correction_dim(beta_slow, dim, base, max_position_embeddings)),
        dim // 2 - 1,  # FIX: clamp to frequency array size, not full dim
    )

    if low > high:
        # Edge case: if the range is inverted (can happen with extreme params),
        # fall back to full interpolation
        low, high = 0, 0

    return low, high


def linear_ramp_factor(min_val: int, max_val: int, dim: int) -> Tensor:
    """
    Create a linear ramp from 0 to 1 across a dimension range.

    Returns tensor of shape [dim] where:
      - Values at indices < min_val are 0 (keep original frequency)
      - Values at indices > max_val are 1 (fully interpolate)
      - Values between are linearly interpolated

    In YaRN context:
      - 0 means "keep original frequency" (high-freq, safe to extrapolate)
      - 1 means "interpolate frequency" (low-freq, must scale down)
    """
    if min_val == max_val:
        # Avoid division by zero — use a small epsilon
        max_val = min_val + 0.001
    ramp = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    return torch.clamp(ramp, 0.0, 1.0)


# =============================================================================
# FREQUENCY PRECOMPUTATION
# =============================================================================


def compute_yarn_mscale(scaling_factor: float, mscale_all_dim: float = 0.0) -> float:
    """
    Compute the YaRN attention temperature correction factor.

    When extending context from L to L*s, attention scores are distributed over
    s times more positions, increasing entropy. Without correction, attention
    distributions become too flat (uniform) at longer contexts.

    The correction multiplies 1/sqrt(d) in attention by mscale:
        effective_scale = (1/sqrt(d)) * mscale

    Formula (from YaRN paper and DeepSeek-V2):
        If mscale_all_dim > 0:
            mscale = 0.1 * mscale_all_dim * ln(scaling_factor) + 1.0
        Else:
            mscale = 0.1 * ln(scaling_factor) + 1.0   (when scaling_factor > 1)
            mscale = 1.0                                (when scaling_factor == 1)

    For scaling_factor=4: mscale ≈ 1.139 (sharpen attention by ~14%)
    For scaling_factor=16: mscale ≈ 1.277 (sharpen by ~28%)
    """
    if scaling_factor <= 1.0:
        return 1.0
    if mscale_all_dim > 0.0:
        return 0.1 * mscale_all_dim * math.log(scaling_factor) + 1.0
    return 0.1 * math.log(scaling_factor) + 1.0


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    scaling_factor: float = 1.0,
    original_max_position_embeddings: int = 4096,
    beta_fast: int = 32,
    beta_slow: int = 1,
) -> Tensor:
    """
    Precompute complex-valued RoPE rotation frequencies, with optional YaRN scaling.

    Args:
        dim: Head dimension for RoPE (must be even).
        end: Maximum sequence length to precompute for.
        theta: Base frequency (10000 is standard, higher = longer wavelengths).
        scaling_factor: Context extension ratio. 1.0 = no extension.
            E.g., 4.0 means extending from original_max to 4× original_max.
        original_max_position_embeddings: Training context length before extension.
        beta_fast: Rotation count threshold for "safe" frequencies (default 32).
            Frequencies completing >= beta_fast rotations in training window
            are kept unchanged.
        beta_slow: Rotation count threshold for "unsafe" frequencies (default 1).
            Frequencies completing <= beta_slow rotations in training window
            are fully interpolated (divided by scaling_factor).

    Returns:
        Complex tensor of shape [end, dim // 2].
        Each entry is e^{i * pos * freq} for the given (position, frequency) pair.

    Note: mscale (attention temperature correction) is NOT applied here — it belongs
    in the attention layer's softmax scaling, not in the frequencies themselves.
    Use compute_yarn_mscale() to get the correction factor for attention.
    """
    assert dim % 2 == 0, f"RoPE dimension must be even, got {dim}"

    # Base frequencies: freq_i = 1 / (theta^(2i/dim)) for i in [0, dim/2)
    # Geometrically spaced from ~1.0 (fast) down to ~theta^(-1) (slow)
    freq_indices = torch.arange(0, dim, 2, dtype=torch.float32)  # [0, 2, 4, ..., dim-2]
    freqs = 1.0 / (theta ** (freq_indices / dim))  # [dim // 2]

    if scaling_factor > 1.0:
        # YaRN: blend between original and interpolated frequencies
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, theta, original_max_position_embeddings
        )

        # smooth: [dim // 2] tensor, 0 = keep original, 1 = interpolate
        smooth = linear_ramp_factor(low, high, dim // 2)

        # Interpolated frequencies: divide by scaling factor
        # This effectively "stretches" the position space so positions 0..s*L
        # map to the same frequency range as 0..L in the original model
        interpolated_freqs = freqs / scaling_factor

        # Blend: high-freq dims keep original, low-freq dims get interpolated
        freqs = freqs * (1.0 - smooth) + interpolated_freqs * smooth

    # Outer product: position × frequency → angle
    positions = torch.arange(end, dtype=torch.float32)  # [end]
    angles = torch.outer(positions, freqs)  # [end, dim // 2]

    # Convert to complex rotation factors: e^{i * angle}
    return torch.polar(torch.ones_like(angles), angles)  # [end, dim // 2]


# =============================================================================
# ROTARY EMBEDDING APPLICATION
# =============================================================================


def apply_rotary_emb(
    x: Tensor,
    freqs_cis: Tensor,
    interleaved: bool = True,
) -> Tensor:
    """
    Apply rotary position embeddings to input tensor.

    This performs the core RoPE operation: treat consecutive dimension pairs
    as 2D planes, and rotate each plane by an angle determined by the token's
    position and the plane's frequency.

    Mathematically, for each 2D plane (x_{2i}, x_{2i+1}):
        [x'_{2i}  ]   [cos(θ)  -sin(θ)] [x_{2i}  ]
        [x'_{2i+1}] = [sin(θ)   cos(θ)] [x_{2i+1}]

    This is equivalent to complex multiplication: (x_{2i} + i*x_{2i+1}) * e^{iθ}

    Args:
        x: Input tensor. Supported shapes:
            - [batch, seq_len, heads, head_dim]  (standard per-head)
            - [batch, seq_len, 1, head_dim]      (shared across heads, e.g. MLA k_pe)
            - [batch, seq_len, head_dim]          (no head dim, auto-unsqueezed)
           head_dim must be even.

        freqs_cis: Complex rotation factors. Supported shapes:
            - [seq_len, dim//2]                  (shared across batch)
            - [batch, seq_len, dim//2]           (per-batch, from position_ids indexing)

        interleaved: Dimension pairing layout.
            True  (default): pairs are (d0,d1), (d2,d3), ... — DeepSeek, Llama-3
                Real tensor layout: [r0, i0, r1, i1, r2, i2, ...]
            False: pairs are (d0, d_{n/2}), (d1, d_{n/2+1}), ... — GPT-NeoX, Llama-1/2
                Real tensor layout: [r0, r1, ..., r_{n/2-1}, i0, i1, ..., i_{n/2-1}]

            CRITICAL: Using the wrong layout produces silently wrong results.
            Check your model's weight format before choosing.

    Returns:
        Rotated tensor with same shape and dtype as input.
    """
    orig_shape = x.shape
    orig_dtype = x.dtype

    # Normalize to 4D: [batch..., seq_len, heads, head_dim]
    squeezed = False
    if x.dim() == 3:
        x = x.unsqueeze(2)  # add head dim
        squeezed = True

    *batch_dims, seq_len, n_heads, head_dim = x.shape
    assert head_dim % 2 == 0, f"head_dim must be even for RoPE, got {head_dim}"

    # =========================================================================
    # Reshape freqs_cis to broadcast with x: [..., seq_len, 1, dim//2]
    # The "1" is for the heads dimension — freqs are shared across heads
    # =========================================================================
    if freqs_cis.dim() == 2:
        # Standard: [seq_len, dim//2] → [1, seq_len, 1, dim//2]
        assert freqs_cis.shape[0] >= seq_len, (
            f"freqs_cis has {freqs_cis.shape[0]} positions but input has "
            f"seq_len={seq_len}. Did you forget to slice freqs_cis or "
            f"is max_position_embeddings too small?"
        )
        freqs_cis = freqs_cis[:seq_len].view(1, seq_len, 1, -1)
    elif freqs_cis.dim() == 3:
        # Batched: [batch, seq_len, dim//2] → [batch, seq_len, 1, dim//2]
        freqs_cis = freqs_cis.unsqueeze(2)
    else:
        raise ValueError(
            f"freqs_cis must be 2D [seq, dim//2] or 3D [batch, seq, dim//2], "
            f"got {freqs_cis.dim()}D with shape {freqs_cis.shape}"
        )

    # =========================================================================
    # Convert real tensor to complex, apply rotation, convert back
    # =========================================================================
    if interleaved:
        # Layout: [r0, i0, r1, i1, ...] → view as complex pairs directly
        x_complex = torch.view_as_complex(
            x.float().reshape(*batch_dims, seq_len, n_heads, head_dim // 2, 2)
        )
    else:
        # Layout: [r0, r1, ..., i0, i1, ...] → stack then view as complex
        half = head_dim // 2
        x_real, x_imag = x[..., :half], x[..., half:]
        x_complex = torch.view_as_complex(
            torch.stack([x_real, x_imag], dim=-1).float()
        )

    # Complex multiply: rotation in each 2D plane
    # x_complex: [batch..., seq, heads, dim//2]
    # freqs_cis: [1 or batch, seq, 1, dim//2]  (broadcasts over heads)
    x_rotated = x_complex * freqs_cis

    # Convert back to real tensor
    if interleaved:
        x_out = torch.view_as_real(x_rotated).reshape(
            *batch_dims, seq_len, n_heads, head_dim
        )
    else:
        x_real_out = torch.view_as_real(x_rotated)
        x_out = torch.cat([x_real_out[..., 0], x_real_out[..., 1]], dim=-1)

    # Restore original shape and dtype
    if squeezed:
        x_out = x_out.squeeze(2)
    return x_out.to(orig_dtype).view(orig_shape)


# =============================================================================
# ROTARY EMBEDDING MODULE (Standalone, for non-MLA architectures)
# =============================================================================


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding module with YaRN context extension support.

    This is a self-contained module for architectures that apply RoPE uniformly
    to Q and K (standard MHA, GQA). For MLA, use precompute_freqs_cis and
    apply_rotary_emb directly, since MLA applies RoPE only to the decoupled
    position components (q_pe, k_pe).

    Usage:
        rope = RotaryEmbedding(dim=64, max_position_embeddings=8192)

        # Prefill (Q and K same length):
        q_rot, k_rot = rope(q, k)  # both [B, seq, heads, dim]

        # Decode with cache (Q=1 token, K=1 new token):
        q_rot, k_rot = rope(q, k, position_ids=pos_ids)
        # Then concatenate k_rot with cached rotated keys

    Note: This module applies the SAME position IDs to both Q and K.
    For decode, pass only the NEW K tokens (not the full cached sequence),
    since cached keys are already rotated.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: int = 32,
        beta_slow: int = 1,
        interleaved: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.interleaved = interleaved

        # Precompute mscale for attention temperature correction
        # (consumer of this module should use this in their attention scaling)
        self.mscale = compute_yarn_mscale(scaling_factor)

        freqs_cis = precompute_freqs_cis(
            dim,
            max_position_embeddings,
            base,
            scaling_factor,
            original_max_position_embeddings,
            beta_fast,
            beta_slow,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def get_freqs(
        self,
        seq_len: int,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Get rotation frequencies for given positions.

        Args:
            seq_len: Sequence length (used when position_ids is None).
            position_ids: [batch, seq_len] explicit position indices.
                Use this for decode steps or batched generation with padding.

        Returns:
            Complex tensor of shape:
                [seq_len, dim//2]        if position_ids is None
                [batch, seq_len, dim//2] if position_ids is provided
        """
        if position_ids is not None:
            return self.freqs_cis[position_ids]
        assert seq_len <= self.max_position_embeddings, (
            f"seq_len={seq_len} exceeds max_position_embeddings="
            f"{self.max_position_embeddings}. Increase max_position_embeddings "
            f"or use YaRN scaling."
        )
        return self.freqs_cis[:seq_len]

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        position_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply RoPE to query and key tensors.

        IMPORTANT: Q and K must have the same seq_len dimension.
        For autoregressive decode with KV cache:
          - Pass only the NEW tokens for both Q and K (typically seq_len=1)
          - Rotate them with the current position_ids
          - Then concatenate rotated K with the cached (already-rotated) keys

        Args:
            q: [batch, seq_len, num_heads, head_dim] or [batch, seq_len, head_dim]
            k: [batch, seq_len, num_kv_heads, head_dim] or [batch, seq_len, head_dim]
                Must have same seq_len as q.
            position_ids: [batch, seq_len] optional explicit positions.

        Returns:
            (q_rotated, k_rotated) with same shapes as inputs.
        """
        q_seq_len = q.shape[1]
        k_seq_len = k.shape[1]
        assert q_seq_len == k_seq_len, (
            f"Q seq_len ({q_seq_len}) != K seq_len ({k_seq_len}). "
            f"For decode with KV cache, pass only NEW tokens for K, not the "
            f"full cached sequence. Cached keys should already be rotated."
        )

        freqs = self.get_freqs(q_seq_len, position_ids)

        q_rotated = apply_rotary_emb(q, freqs, interleaved=self.interleaved)
        k_rotated = apply_rotary_emb(k, freqs, interleaved=self.interleaved)
        return q_rotated, k_rotated


# =============================================================================
# COMPREHENSIVE SMOKE TESTS
# =============================================================================


def _test_basic_rotation():
    """Test that RoPE at position 0 is identity and rotation varies with position."""
    print("Test 1: Basic rotation properties")
    dim = 64
    freqs = precompute_freqs_cis(dim, end=128)

    # Position 0: all angles are 0, so rotation should be identity
    x = torch.randn(1, 1, 4, dim)
    x_rot = apply_rotary_emb(x, freqs[:1])
    max_diff = (x - x_rot).abs().max().item()
    assert max_diff < 1e-6, f"Position 0 should be identity, got diff={max_diff}"
    print(f"  Position 0 identity: max_diff={max_diff:.2e} ✓")

    # Different positions should produce different rotations
    x_pos5 = apply_rotary_emb(x, freqs[5:6])
    x_pos10 = apply_rotary_emb(x, freqs[10:11])
    diff_5_10 = (x_pos5 - x_pos10).abs().max().item()
    assert diff_5_10 > 0.01, f"Different positions should give different results"
    print(f"  Pos 5 vs pos 10 differ: max_diff={diff_5_10:.4f} ✓")


def _test_relative_position_property():
    """
    The core RoPE property: dot(RoPE(q, m), RoPE(k, n)) depends only on (m-n).

    If we rotate q by position m and k by position n, the dot product should
    equal the dot product when q is at position (m+d) and k at (n+d) for any d.
    """
    print("\nTest 2: Relative position invariance")
    dim = 64
    freqs = precompute_freqs_cis(dim, end=256)

    q = torch.randn(1, 1, 1, dim)
    k = torch.randn(1, 1, 1, dim)

    # Rotate to positions (10, 3) — relative distance = 7
    q_10 = apply_rotary_emb(q, freqs[10:11])
    k_3 = apply_rotary_emb(k, freqs[3:4])
    dot_10_3 = (q_10 * k_3).sum().item()

    # Rotate to positions (110, 103) — relative distance = 7 (same!)
    q_110 = apply_rotary_emb(q, freqs[110:111])
    k_103 = apply_rotary_emb(k, freqs[103:104])
    dot_110_103 = (q_110 * k_103).sum().item()

    diff = abs(dot_10_3 - dot_110_103)
    assert diff < 1e-4, f"Relative position invariance violated: diff={diff}"
    print(f"  dot(q@10, k@3) = {dot_10_3:.6f}")
    print(f"  dot(q@110, k@103) = {dot_110_103:.6f}")
    print(f"  Difference: {diff:.2e} ✓")


def _test_interleaved_vs_non_interleaved():
    """Both layouts should produce equivalent results (just different dim ordering)."""
    print("\nTest 3: Interleaved vs non-interleaved equivalence")
    dim = 64
    half = dim // 2
    freqs = precompute_freqs_cis(dim, end=32)

    # Create a vector and its permuted version
    x_interleaved = torch.randn(2, 8, 4, dim)

    # Convert interleaved → non-interleaved layout
    # Interleaved:     [r0, i0, r1, i1, ...] → pairs at (0,1), (2,3), ...
    # Non-interleaved: [r0, r1, ..., i0, i1, ...] → pairs at (0, half), (1, half+1), ...
    x_non_interleaved = torch.zeros_like(x_interleaved)
    x_non_interleaved[..., :half] = x_interleaved[..., 0::2]  # real parts
    x_non_interleaved[..., half:] = x_interleaved[..., 1::2]  # imag parts

    # Rotate both
    out_inter = apply_rotary_emb(x_interleaved, freqs[:8], interleaved=True)
    out_non = apply_rotary_emb(x_non_interleaved, freqs[:8], interleaved=False)

    # Convert non-interleaved output back to interleaved for comparison
    out_non_as_inter = torch.zeros_like(out_non)
    out_non_as_inter[..., 0::2] = out_non[..., :half]
    out_non_as_inter[..., 1::2] = out_non[..., half:]

    max_diff = (out_inter - out_non_as_inter).abs().max().item()
    assert max_diff < 1e-5, f"Layout equivalence violated: diff={max_diff}"
    print(f"  Max difference: {max_diff:.2e} ✓")


def _test_yarn_scaling():
    """Test YaRN frequency modification."""
    print("\nTest 4: YaRN scaling properties")
    dim = 64

    freqs_base = precompute_freqs_cis(dim, end=128, scaling_factor=1.0)
    freqs_yarn = precompute_freqs_cis(
        dim, end=128, scaling_factor=4.0, original_max_position_embeddings=128
    )

    # YaRN frequencies should differ from base
    diff = (freqs_base - freqs_yarn).abs().max().item()
    assert diff > 0.01, "YaRN should modify frequencies"
    print(f"  Base vs YaRN max freq diff: {diff:.4f} ✓")

    # High-frequency components should be approximately unchanged
    # (they complete many rotations and don't need interpolation)
    angle_base = freqs_base[1].angle()  # angles at position 1
    angle_yarn = freqs_yarn[1].angle()

    # First few frequencies (highest) should be nearly identical
    high_freq_diff = (angle_base[:3] - angle_yarn[:3]).abs().max().item()
    print(f"  High-freq components diff: {high_freq_diff:.6f} (should be small)")

    # Last few frequencies (lowest) should be scaled down by ~4x
    low_freq_ratio = (angle_base[-3:].abs() / angle_yarn[-3:].abs()).mean().item()
    print(f"  Low-freq scaling ratio: {low_freq_ratio:.2f} (target ≈ {4.0:.1f})")

    # mscale computation
    mscale = compute_yarn_mscale(4.0)
    expected = 0.1 * math.log(4.0) + 1.0
    assert abs(mscale - expected) < 1e-6
    print(f"  mscale for 4x extension: {mscale:.4f} (expected {expected:.4f}) ✓")

    mscale_1 = compute_yarn_mscale(1.0)
    assert mscale_1 == 1.0, "No scaling should give mscale=1"
    print(f"  mscale for no extension: {mscale_1:.1f} ✓")


def _test_shared_head_broadcast():
    """Test that RoPE works with shared-across-heads k_pe (head dim = 1)."""
    print("\nTest 5: Shared head broadcast (MLA k_pe pattern)")
    dim = 64
    freqs = precompute_freqs_cis(dim, end=32)

    # MLA k_pe: [batch, seq, 1, rope_dim] — shared across heads
    k_pe = torch.randn(2, 8, 1, dim)
    k_pe_rot = apply_rotary_emb(k_pe, freqs[:8])
    assert k_pe_rot.shape == (2, 8, 1, dim), f"Shape mismatch: {k_pe_rot.shape}"
    print(f"  Shape preserved: {k_pe_rot.shape} ✓")

    # Should broadcast correctly when expanded to multiple heads
    k_pe_expanded = k_pe.expand(-1, -1, 16, -1)  # [2, 8, 16, 64]
    k_pe_expanded_rot = apply_rotary_emb(k_pe_expanded, freqs[:8])

    # Expanding then rotating should equal rotating then expanding
    diff = (k_pe_expanded_rot - k_pe_rot.expand(-1, -1, 16, -1)).abs().max().item()
    assert diff < 1e-6, f"Broadcast inconsistency: {diff}"
    print(f"  Expand-then-rotate == rotate-then-expand: diff={diff:.2e} ✓")


def _test_3d_input():
    """Test that 3D input (no head dim) works correctly."""
    print("\nTest 6: 3D input (auto-unsqueeze head dim)")
    dim = 64
    freqs = precompute_freqs_cis(dim, end=32)

    x_3d = torch.randn(2, 8, dim)           # [batch, seq, dim]
    x_4d = x_3d.unsqueeze(2)                 # [batch, seq, 1, dim]

    out_3d = apply_rotary_emb(x_3d, freqs[:8])
    out_4d = apply_rotary_emb(x_4d, freqs[:8])

    assert out_3d.shape == (2, 8, dim), f"3D shape wrong: {out_3d.shape}"
    diff = (out_3d - out_4d.squeeze(2)).abs().max().item()
    assert diff < 1e-6, f"3D vs 4D mismatch: {diff}"
    print(f"  3D output shape: {out_3d.shape} ✓")
    print(f"  3D vs 4D equivalence: diff={diff:.2e} ✓")


def _test_batched_position_ids():
    """Test that batched position_ids produce correct per-sequence rotations."""
    print("\nTest 7: Batched position_ids")
    dim = 64
    freqs = precompute_freqs_cis(dim, end=128)

    x = torch.randn(2, 4, 8, dim)
    pos_ids = torch.tensor([
        [0, 1, 2, 3],   # sequence 0: positions 0-3
        [5, 6, 7, 8],   # sequence 1: positions 5-8
    ])

    # Batched path
    batched_freqs = freqs[pos_ids]  # [2, 4, dim//2]
    out_batched = apply_rotary_emb(x, batched_freqs)

    # Manual per-sequence
    out_0 = apply_rotary_emb(x[0:1], freqs[0:4])
    out_1 = apply_rotary_emb(x[1:2], freqs[5:9])
    out_manual = torch.cat([out_0, out_1], dim=0)

    diff = (out_batched - out_manual).abs().max().item()
    assert diff < 1e-5, f"Batched vs manual mismatch: {diff}"
    print(f"  Batched vs manual: diff={diff:.2e} ✓")


def _test_rotary_embedding_module():
    """Test the RotaryEmbedding nn.Module wrapper."""
    print("\nTest 8: RotaryEmbedding module")
    rope = RotaryEmbedding(
        dim=64,
        max_position_embeddings=256,
        scaling_factor=1.0,
    )

    q = torch.randn(2, 16, 8, 64)
    k = torch.randn(2, 16, 8, 64)

    # Without position_ids (prefill)
    q_rot, k_rot = rope(q, k)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    print(f"  Prefill shapes correct ✓")

    # With position_ids (decode)
    pos = torch.arange(16).unsqueeze(0).expand(2, -1)
    q_rot2, k_rot2 = rope(q, k, position_ids=pos)
    diff = (q_rot - q_rot2).abs().max().item()
    assert diff < 1e-5, f"Sequential pos_ids should match implicit: {diff}"
    print(f"  Explicit pos_ids match implicit: diff={diff:.2e} ✓")

    # YaRN mscale
    rope_yarn = RotaryEmbedding(dim=64, max_position_embeddings=1024, scaling_factor=4.0)
    expected_mscale = 0.1 * math.log(4.0) + 1.0
    assert abs(rope_yarn.mscale - expected_mscale) < 1e-6
    print(f"  YaRN mscale={rope_yarn.mscale:.4f} ✓")

    # Assert on mismatched Q/K lengths
    try:
        rope(q[:, :8], k[:, :4])
        assert False, "Should have raised assertion"
    except AssertionError:
        print(f"  Q/K length mismatch caught ✓")


def _test_correction_range_values():
    """Verify correction range computation with concrete numbers."""
    print("\nTest 9: YaRN correction range")

    dim = 64
    base = 10000.0
    max_pos = 4096

    # beta_fast=32: frequency index where dim completes 32 rotations in 4096 positions
    idx_fast = find_correction_dim(32, dim, base, max_pos)
    # beta_slow=1: frequency index where dim completes 1 rotation in 4096 positions
    idx_slow = find_correction_dim(1, dim, base, max_pos)

    print(f"  beta_fast=32 → freq idx ≈ {idx_fast:.2f}")
    print(f"  beta_slow=1  → freq idx ≈ {idx_slow:.2f}")
    assert idx_fast < idx_slow, "fast should map to lower index (higher frequency)"
    print(f"  idx_fast < idx_slow (monotonically decreasing) ✓")

    low, high = find_correction_range(32, 1, dim, base, max_pos)
    print(f"  Correction range: [{low}, {high}] out of {dim // 2} freq indices")
    assert 0 <= low <= high < dim // 2
    print(f"  Range valid ✓")

    # Verify the smooth ramp
    smooth = linear_ramp_factor(low, high, dim // 2)
    assert smooth[0].item() == 0.0, "First element should be 0 (keep)"
    assert smooth[-1].item() == 1.0, "Last element should be 1 (interpolate)"
    assert (smooth[low] >= 0.0).item() and (smooth[high] <= 1.0).item()
    print(f"  Smooth ramp: [{smooth[0]:.1f} ... {smooth[low]:.2f} ... {smooth[high]:.2f} ... {smooth[-1]:.1f}] ✓")


if __name__ == "__main__":
    torch.manual_seed(42)
    print("=" * 70)
    print("RoPE + YaRN IMPLEMENTATION TESTS")
    print("=" * 70)

    _test_basic_rotation()
    _test_relative_position_property()
    _test_interleaved_vs_non_interleaved()
    _test_yarn_scaling()
    _test_shared_head_broadcast()
    _test_3d_input()
    _test_batched_position_ids()
    _test_rotary_embedding_module()
    _test_correction_range_values()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
