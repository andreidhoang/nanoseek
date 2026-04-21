"""
Standalone MLA tests — works with current model.py (MLA-only implementation).

Tests:
1. Forward shape correctness (naive mode)
2. No NaN/Inf in output
3. Gradient flow through all projection paths
4. KV cache shape and incremental decoding consistency
5. Absorb mode matches naive mode (mathematical equivalence)
6. Causal masking prevents future attention leakage
7. mscale squared behavior (DeepSeek V3 spec)
8. Various sequence lengths
9. Single-token decode with cache
10. Position IDs support
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add nanoseek/nanoseek/ to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent / "nanoseek"))

from model import (
    MultiHeadLatentAttention,
    RMSNorm,
    apply_rotary_emb,
    precompute_freqs_cis,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────────

DEVICE = torch.device("cpu")

# Minimal MLA config matching DeepSeek structure but small enough for CPU tests
MLA_KWARGS = dict(
    hidden_size=256,
    num_heads=4,
    q_lora_rank=55,
    kv_lora_rank=18,
    qk_nope_head_dim=48,
    qk_rope_head_dim=16,
    v_head_dim=48,
    max_position_embeddings=256,
    rope_theta=10000.0,
    rope_scaling_factor=1.0,
    original_max_position_embeddings=256,
    mscale=1.0,
    attention_dropout=0.0,
    layer_idx=0,
)

BATCH, SEQ = 2, 32


@pytest.fixture
def mla():
    torch.manual_seed(42)
    m = MultiHeadLatentAttention(**MLA_KWARGS).to(DEVICE)
    m.eval()
    return m


@pytest.fixture
def hidden(mla):
    torch.manual_seed(7)
    return torch.randn(BATCH, SEQ, mla.hidden_size, device=DEVICE)


# ─── 1. Forward Shape ─────────────────────────────────────────────────────────

class TestForwardShape:

    def test_output_shape_matches_input(self, mla, hidden):
        out, _ = mla(hidden)
        assert out.shape == hidden.shape

    def test_output_shape_with_cache(self, mla, hidden):
        out, cache = mla(hidden, use_cache=True)
        assert out.shape == hidden.shape
        assert cache is not None

    @pytest.mark.parametrize("seq_len", [1, 4, 16, 64, 128])
    def test_various_seq_lengths(self, mla, seq_len):
        x = torch.randn(1, seq_len, mla.hidden_size, device=DEVICE)
        out, _ = mla(x)
        assert out.shape == x.shape


# ─── 2. No NaN/Inf ────────────────────────────────────────────────────────────

class TestNumericalHealth:

    def test_no_nan_inf(self, mla, hidden):
        out, _ = mla(hidden)
        assert not torch.isnan(out).any(), "NaN in output"
        assert not torch.isinf(out).any(), "Inf in output"

    def test_large_inputs(self, mla):
        x = torch.randn(1, 16, mla.hidden_size) * 10.0
        out, _ = mla(x)
        assert not torch.isnan(out).any()

    def test_small_inputs(self, mla):
        x = torch.randn(1, 16, mla.hidden_size) * 0.001
        out, _ = mla(x)
        assert not torch.isnan(out).any()


# ─── 3. Gradient Flow ─────────────────────────────────────────────────────────

class TestGradientFlow:

    def test_all_projections_get_gradients(self, hidden):
        torch.manual_seed(42)
        mla = MultiHeadLatentAttention(**MLA_KWARGS).to(DEVICE)
        mla.train()

        hidden_copy = hidden.clone().requires_grad_(True)
        out, _ = mla(hidden_copy)
        out.sum().backward()

        for name in ["wq_a", "wq_b", "wkv_a", "wkv_b", "wo"]:
            layer = getattr(mla, name)
            assert layer.weight.grad is not None, f"{name} missing gradient"
            assert layer.weight.grad.abs().mean() > 0, f"{name} has zero gradient"

    def test_norm_layers_get_gradients(self, hidden):
        torch.manual_seed(42)
        mla = MultiHeadLatentAttention(**MLA_KWARGS).to(DEVICE)
        mla.train()

        out, _ = mla(hidden.clone().requires_grad_(True))
        out.sum().backward()

        assert mla.q_norm.weight.grad is not None, "q_norm missing gradient"
        assert mla.kv_norm.weight.grad is not None, "kv_norm missing gradient"

    def test_gradient_magnitudes_reasonable(self, hidden):
        torch.manual_seed(42)
        mla = MultiHeadLatentAttention(**MLA_KWARGS).to(DEVICE)
        mla.train()

        out, _ = mla(hidden.clone().requires_grad_(True))
        out.sum().backward()

        for name, param in mla.named_parameters():
            if param.grad is not None:
                norm = param.grad.norm().item()
                assert norm < 1e6, f"{name} gradient exploded: {norm}"
                assert norm > 1e-10, f"{name} gradient vanished: {norm}"


# ─── 4. KV Cache ──────────────────────────────────────────────────────────────

class TestKVCache:

    def test_cache_shape(self, mla, hidden):
        _, cache = mla(hidden, use_cache=True)
        c_kv, k_pe = cache

        assert c_kv.shape == (BATCH, SEQ, mla.kv_lora_rank)
        assert k_pe.shape == (BATCH, SEQ, 1, mla.qk_rope_head_dim)

    def test_cache_grows_with_tokens(self, mla):
        """Single-token decode should grow cache by 1 each step."""
        prompt = torch.randn(1, 10, mla.hidden_size, device=DEVICE)
        _, cache = mla(prompt, use_cache=True)
        assert cache[0].shape[1] == 10

        for step in range(5):
            tok = torch.randn(1, 1, mla.hidden_size, device=DEVICE)
            _, cache = mla(tok, past_key_value=cache, use_cache=True)
            assert cache[0].shape[1] == 11 + step, f"Cache length wrong at step {step}"

    def test_incremental_matches_full(self, mla):
        """Prefill + incremental decode must match full forward."""
        torch.manual_seed(99)
        x = torch.randn(1, 20, mla.hidden_size, device=DEVICE)

        # Full forward
        with torch.no_grad():
            out_full, _ = mla(x)

        # Split: prefix(12) + suffix(8)
        with torch.no_grad():
            out_pre, cache = mla(x[:, :12], use_cache=True)
            out_suf, _ = mla(x[:, 12:], past_key_value=cache, use_cache=True)

        out_inc = torch.cat([out_pre, out_suf], dim=1)
        max_diff = (out_full - out_inc).abs().max().item()
        assert max_diff < 1e-4, f"Incremental vs full mismatch: {max_diff}"


# ─── 5. Absorb Mode Equivalence ───────────────────────────────────────────────

class TestAbsorbMode:

    def test_absorb_matches_naive(self, hidden):
        """Core correctness test: absorb mode must produce same output as naive."""
        torch.manual_seed(42)
        mla = MultiHeadLatentAttention(**MLA_KWARGS).to(DEVICE)
        mla.eval()

        with torch.no_grad():
            # Naive (training) mode
            mla.absorb = False
            out_naive, cache_naive = mla(hidden, use_cache=True)

            # Absorb (inference) mode
            mla.absorb = True
            out_absorb, cache_absorb = mla(hidden, use_cache=True)

        # Cache format should be identical (both store c_kv + k_pe)
        assert cache_naive[0].shape == cache_absorb[0].shape, "c_kv cache shape mismatch"
        assert cache_naive[1].shape == cache_absorb[1].shape, "k_pe cache shape mismatch"

        # Cache content should match
        ckv_diff = (cache_naive[0] - cache_absorb[0]).abs().max().item()
        assert ckv_diff < 1e-5, f"c_kv cache differs: {ckv_diff}"

        # Output should match within numerical tolerance
        max_diff = (out_naive - out_absorb).abs().max().item()
        assert max_diff < 1e-3, f"Absorb vs naive output mismatch: {max_diff}"

    def test_absorb_incremental_decode(self):
        """Absorb mode with KV cache for autoregressive generation."""
        torch.manual_seed(42)
        mla = MultiHeadLatentAttention(**MLA_KWARGS).to(DEVICE)
        mla.eval()
        mla.absorb = True

        prompt = torch.randn(1, 8, mla.hidden_size, device=DEVICE)
        with torch.no_grad():
            _, cache = mla(prompt, use_cache=True)

        # Generate 5 tokens
        for i in range(5):
            tok = torch.randn(1, 1, mla.hidden_size, device=DEVICE)
            with torch.no_grad():
                out, cache = mla(tok, past_key_value=cache, use_cache=True)
            assert out.shape == (1, 1, mla.hidden_size)
            assert not torch.isnan(out).any(), f"NaN at decode step {i}"
            assert cache[0].shape[1] == 9 + i

    def test_absorb_matches_naive_with_cache(self):
        """Absorb and naive must match even during incremental decode."""
        torch.manual_seed(42)
        mla = MultiHeadLatentAttention(**MLA_KWARGS).to(DEVICE)
        mla.eval()

        prompt = torch.randn(1, 8, mla.hidden_size, device=DEVICE)
        new_tok = torch.randn(1, 1, mla.hidden_size, device=DEVICE)

        with torch.no_grad():
            # Naive path
            mla.absorb = False
            _, cache_n = mla(prompt, use_cache=True)
            out_n, _ = mla(new_tok, past_key_value=cache_n, use_cache=True)

            # Absorb path
            mla.absorb = True
            _, cache_a = mla(prompt, use_cache=True)
            out_a, _ = mla(new_tok, past_key_value=cache_a, use_cache=True)

        max_diff = (out_n - out_a).abs().max().item()
        assert max_diff < 1e-3, f"Absorb vs naive decode mismatch: {max_diff}"

    def test_absorb_matches_naive_with_cached_chunk(self):
        """Absorb and naive must match for cached multi-token suffixes, not just 1-token decode."""
        torch.manual_seed(42)
        mla = MultiHeadLatentAttention(**MLA_KWARGS).to(DEVICE)
        mla.eval()

        prompt = torch.randn(1, 8, mla.hidden_size, device=DEVICE)
        suffix = torch.randn(1, 5, mla.hidden_size, device=DEVICE)

        with torch.no_grad():
            mla.absorb = False
            _, cache_n = mla(prompt, use_cache=True)
            out_n, _ = mla(suffix, past_key_value=cache_n, use_cache=True)

            mla.absorb = True
            _, cache_a = mla(prompt, use_cache=True)
            out_a, _ = mla(suffix, past_key_value=cache_a, use_cache=True)

        max_diff = (out_n - out_a).abs().max().item()
        assert max_diff < 1e-3, f"Absorb vs naive cached chunk mismatch: {max_diff}"


# ─── 6. Causal Masking ────────────────────────────────────────────────────────

class TestCausalMasking:

    def test_future_tokens_dont_affect_past(self, mla):
        """Changing future tokens should not change output for past positions."""
        torch.manual_seed(42)
        x1 = torch.randn(1, 16, mla.hidden_size, device=DEVICE)
        x2 = x1.clone()
        x2[:, 8:] = torch.randn(1, 8, mla.hidden_size, device=DEVICE)  # change future

        with torch.no_grad():
            out1, _ = mla(x1)
            out2, _ = mla(x2)

        # First 8 positions should be identical (causal = no future leakage)
        diff = (out1[:, :8] - out2[:, :8]).abs().max().item()
        assert diff < 1e-5, f"Future tokens leaked into past: {diff}"

    def test_explicit_mask_works(self, mla, hidden):
        """Explicit attention mask should produce valid output."""
        mask = torch.tril(torch.ones(SEQ, SEQ, device=DEVICE)).unsqueeze(0).unsqueeze(0)
        out, _ = mla(hidden, attention_mask=mask)
        assert not torch.isnan(out).any()

    def test_single_token_no_mask_needed(self, mla):
        """seq_len=1 with cache should work without any mask."""
        prompt = torch.randn(1, 10, mla.hidden_size, device=DEVICE)
        with torch.no_grad():
            _, cache = mla(prompt, use_cache=True)
            out, _ = mla(
                torch.randn(1, 1, mla.hidden_size, device=DEVICE),
                past_key_value=cache,
            )
        assert not torch.isnan(out).any()


# ─── 7. mscale Squared ────────────────────────────────────────────────────────

class TestMscale:

    def test_mscale_squared_effect(self):
        """mscale > 1 should change output via squared scaling."""
        torch.manual_seed(42)
        x = torch.randn(1, 8, 256, device=DEVICE)

        mla_1 = MultiHeadLatentAttention(**{**MLA_KWARGS, "mscale": 1.0}).to(DEVICE)
        mla_2 = MultiHeadLatentAttention(**{**MLA_KWARGS, "mscale": 1.5}).to(DEVICE)

        # Copy weights so only mscale differs
        mla_2.load_state_dict(mla_1.state_dict())

        with torch.no_grad():
            out1, _ = mla_1(x)
            out2, _ = mla_2(x)

        # Outputs should differ (mscale=1.5 → scale factor 1.5² = 2.25× different)
        diff = (out1 - out2).abs().max().item()
        assert diff > 1e-4, f"mscale had no effect: diff={diff}"

    def test_mscale_1_is_noop(self):
        """mscale=1.0 should give same scale as no mscale (1²=1)."""
        import math
        mla = MultiHeadLatentAttention(**MLA_KWARGS)
        expected = 1.0 / math.sqrt(mla.qk_head_dim)  # base_scale * 1.0 * 1.0
        actual = mla.base_scale * mla.mscale * mla.mscale
        assert abs(expected - actual) < 1e-10


# ─── 8. Position IDs ──────────────────────────────────────────────────────────

class TestPositionIDs:

    def test_explicit_position_ids(self, mla):
        x = torch.randn(1, 8, mla.hidden_size, device=DEVICE)
        pos = torch.arange(8, device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            out_auto, _ = mla(x)
            out_explicit, _ = mla(x, position_ids=pos)

        # Same positions → same output
        diff = (out_auto - out_explicit).abs().max().item()
        assert diff < 1e-5, f"Explicit pos_ids differ from auto: {diff}"

    def test_different_positions_give_different_output(self, mla):
        """Verify RoPE makes position_ids actually affect q_pe/k_pe.

        Note: MLA output difference is small because nope_dim >> rope_dim (48 vs 16),
        so the position-independent component dominates. We test RoPE directly instead.
        """
        freqs_0 = mla._get_freqs(torch.tensor([[0, 1, 2, 3]]), 0, 4)
        freqs_100 = mla._get_freqs(torch.tensor([[100, 101, 102, 103]]), 0, 4)

        # RoPE frequencies must differ for different absolute positions
        assert not torch.allclose(freqs_0, freqs_100, atol=1e-6), \
            "RoPE freqs should differ for positions 0-3 vs 100-103"

        # Also verify via apply_rotary_emb that rotation actually changes values
        x = torch.randn(1, 4, 1, mla.qk_rope_head_dim)
        rot_0 = apply_rotary_emb(x, freqs_0)
        rot_100 = apply_rotary_emb(x, freqs_100)
        diff = (rot_0 - rot_100).abs().max().item()
        assert diff > 0.01, f"RoPE should rotate differently at different positions, diff={diff}"


# ─── 9. Compression Ratio ─────────────────────────────────────────────────────

class TestCompression:

    def test_kv_cache_compression(self, mla, hidden):
        """MLA KV cache should be much smaller than MHA equivalent."""
        _, cache = mla(hidden, use_cache=True)
        c_kv, k_pe = cache

        # MLA cache per token: kv_lora_rank + qk_rope_head_dim
        mla_per_token = mla.kv_lora_rank + mla.qk_rope_head_dim

        # MHA cache per token: 2 * num_heads * head_dim (K + V)
        # Using qk_head_dim for K and v_head_dim for V
        mha_per_token = mla.num_heads * (mla.qk_head_dim + mla.v_head_dim)

        ratio = mha_per_token / mla_per_token
        assert ratio > 5, f"Expected >5x compression, got {ratio:.1f}x"


# ─── 10. RoPE Functions ───────────────────────────────────────────────────────

class TestRoPE:

    def test_precompute_shape(self):
        freqs = precompute_freqs_cis(dim=16, end=128)
        assert freqs.shape == (128, 8)  # dim//2 complex values

    def test_apply_rotary_preserves_shape(self):
        x = torch.randn(1, 8, 4, 16)  # [B, S, heads, rope_dim]
        freqs = precompute_freqs_cis(dim=16, end=8)
        out = apply_rotary_emb(x, freqs)
        assert out.shape == x.shape

    def test_apply_rotary_shared_head(self):
        """k_pe with head=1 should work (broadcast)."""
        x = torch.randn(1, 8, 1, 16)  # shared across heads
        freqs = precompute_freqs_cis(dim=16, end=8)
        out = apply_rotary_emb(x, freqs)
        assert out.shape == x.shape

    def test_rotary_changes_values(self):
        x = torch.randn(1, 8, 1, 16)
        freqs = precompute_freqs_cis(dim=16, end=8)
        out = apply_rotary_emb(x, freqs)
        # Position 0 with freq at t=0 is exp(0)=1, so no rotation
        # Other positions should differ
        assert not torch.allclose(x[:, 1:], out[:, 1:], atol=1e-5)

    def test_yarn_scaling(self):
        """YaRN with scaling_factor > 1 should produce different freqs."""
        f1 = precompute_freqs_cis(dim=16, end=64, scaling_factor=1.0)
        f2 = precompute_freqs_cis(dim=16, end=64, scaling_factor=4.0)
        assert not torch.allclose(f1, f2)


# ─── 11. RMSNorm ──────────────────────────────────────────────────────────────

class TestRMSNorm:

    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalizes(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64) * 100
        out = norm(x)
        rms = out.float().pow(2).mean(-1).sqrt()
        # After RMSNorm with weight=1, RMS should be ~1
        assert rms.mean().item() < 5.0  # reasonable range


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
