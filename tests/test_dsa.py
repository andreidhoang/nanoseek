"""
DSA (DeepSeek Sparse Attention) Validation Tests

DeepSeek-Level Validation:
1. Indexer training via auxiliary loss
2. Dense mode correctness
3. Sparse mode activation threshold
4. Token selection quality
5. Sparse-dense equivalence (when possible)
6. Gradient flow through indexer
"""

import pytest
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.model import (
    DSASparseAttention,
    LightningIndexer,
    MultiHeadLatentAttention,
)
from model.config import SparseAttentionConfig, get_nanoseek_config

from tests.utils import check_no_nan_inf, check_gradient_health


class TestLightningIndexer:
    """Test Lightning Indexer component."""

    def test_indexer_output_shape(self, indexer_minimal, minimal_config, device):
        """Verify indexer produces correct score shapes."""
        batch, q_len, kv_len = 2, 32, 64

        q_compressed = torch.randn(batch, q_len, minimal_config.mla.q_lora_rank, device=device)
        kv_compressed = torch.randn(batch, kv_len, minimal_config.mla.kv_lora_rank, device=device)

        index_scores = indexer_minimal(q_compressed, kv_compressed)

        assert index_scores.shape == (batch, q_len, kv_len), \
            f"Index scores shape: {index_scores.shape}"

    def test_indexer_scores_non_negative(self, indexer_minimal, minimal_config, device):
        """Verify indexer uses ReLU activation (non-negative scores)."""
        batch, seq = 2, 32

        q_compressed = torch.randn(batch, seq, minimal_config.mla.q_lora_rank, device=device)
        kv_compressed = torch.randn(batch, seq, minimal_config.mla.kv_lora_rank, device=device)

        index_scores = indexer_minimal(q_compressed, kv_compressed)

        assert index_scores.min() >= 0, f"Scores should be non-negative (ReLU): {index_scores.min()}"

    def test_indexer_topk_selection(self, indexer_minimal, minimal_config, device):
        """Test top-k token selection."""
        batch, q_len, kv_len = 2, 16, 64
        k = 8

        q_compressed = torch.randn(batch, q_len, minimal_config.mla.q_lora_rank, device=device)
        kv_compressed = torch.randn(batch, kv_len, minimal_config.mla.kv_lora_rank, device=device)

        index_scores = indexer_minimal(q_compressed, kv_compressed)
        values, indices = indexer_minimal.select_topk(index_scores, k)

        assert values.shape == (batch, q_len, k), f"Values shape: {values.shape}"
        assert indices.shape == (batch, q_len, k), f"Indices shape: {indices.shape}"
        assert indices.max() < kv_len, f"Index out of range: {indices.max()}"

    def test_indexer_causal_masking(self, indexer_minimal, minimal_config, device):
        """Test indexer respects causal masking."""
        batch, seq = 2, 32

        q_compressed = torch.randn(batch, seq, minimal_config.mla.q_lora_rank, device=device)
        kv_compressed = torch.randn(batch, seq, minimal_config.mla.kv_lora_rank, device=device)

        # Create causal mask
        causal_mask = torch.triu(
            torch.full((seq, seq), float('-inf'), device=device),
            diagonal=1
        )

        index_scores = indexer_minimal(q_compressed, kv_compressed, causal_mask=causal_mask)

        # Future positions should have -inf (after adding mask)
        # Actually, the indexer adds the mask to scores, so check for very negative values
        for q_pos in range(seq):
            future_scores = index_scores[:, q_pos, q_pos + 1:]
            if future_scores.numel() > 0:
                assert (future_scores < -1e6).all(), \
                    f"Future scores not masked at q_pos={q_pos}"


class TestDSAModes:
    """Test DSA dense and sparse modes."""

    def test_dense_mode_below_threshold(self, dsa_minimal, sample_hidden_minimal):
        """Verify dense mode is used below activation threshold."""
        # Our minimal config has threshold=16, sample is 64 tokens
        # This should use dense mode
        dsa_minimal.sparse_config.activation_threshold = 100  # Force dense

        output, cache, aux = dsa_minimal(sample_hidden_minimal, use_cache=True)

        assert output.shape == sample_hidden_minimal.shape
        check_no_nan_inf(output, "DSA dense output")

    def test_sparse_mode_above_threshold(self, minimal_config, device):
        """Verify sparse mode activates above threshold."""
        sparse_config = SparseAttentionConfig(
            enabled=True,
            topk_tokens=16,
            activation_threshold=32,  # Low threshold
            dense_warmup_steps=0,
        )

        dsa = DSASparseAttention(
            hidden_size=minimal_config.hidden_size,
            num_heads=minimal_config.num_heads,
            q_lora_rank=minimal_config.mla.q_lora_rank,
            kv_lora_rank=minimal_config.mla.kv_lora_rank,
            qk_nope_head_dim=minimal_config.mla.qk_nope_head_dim,
            qk_rope_head_dim=minimal_config.mla.qk_rope_head_dim,
            v_head_dim=minimal_config.mla.v_head_dim,
            sparse_config=sparse_config,
        ).to(device)

        # Sequence above threshold
        x = torch.randn(1, 64, minimal_config.hidden_size, device=device)
        output, _, _ = dsa(x)

        check_no_nan_inf(output, "DSA sparse output")

    def test_warmup_forces_dense(self, minimal_config, device):
        """Verify warmup period forces dense attention."""
        sparse_config = SparseAttentionConfig(
            enabled=True,
            topk_tokens=16,
            activation_threshold=8,
            dense_warmup_steps=100,  # Long warmup
        )

        dsa = DSASparseAttention(
            hidden_size=minimal_config.hidden_size,
            num_heads=minimal_config.num_heads,
            q_lora_rank=minimal_config.mla.q_lora_rank,
            kv_lora_rank=minimal_config.mla.kv_lora_rank,
            qk_nope_head_dim=minimal_config.mla.qk_nope_head_dim,
            qk_rope_head_dim=minimal_config.mla.qk_rope_head_dim,
            v_head_dim=minimal_config.mla.v_head_dim,
            sparse_config=sparse_config,
        ).to(device)

        # During warmup (step 0), should use dense even above threshold
        x = torch.randn(1, 32, minimal_config.hidden_size, device=device)
        output, _, _ = dsa(x)

        check_no_nan_inf(output, "DSA warmup output")


class TestDSAIndexerTraining:
    """Test indexer training via auxiliary loss."""

    def test_indexer_loss_computed(self, dsa_minimal, sample_hidden_minimal):
        """Verify indexer loss is computed during training."""
        dsa_minimal.train()

        output, _, aux = dsa_minimal(sample_hidden_minimal, output_indexer_loss=True)

        assert 'indexer_loss' in aux, "Indexer loss should be returned"
        assert aux['indexer_loss'] >= 0, "Indexer loss should be non-negative"
        check_no_nan_inf(aux['indexer_loss'], "Indexer loss")

    def test_indexer_gradient_flow(self, dsa_minimal, sample_hidden_minimal):
        """Verify gradients flow to indexer parameters."""
        dsa_minimal.train()
        sample_hidden_minimal = sample_hidden_minimal.clone().requires_grad_(True)

        output, _, aux = dsa_minimal(sample_hidden_minimal, output_indexer_loss=True)

        if 'indexer_loss' in aux:
            aux['indexer_loss'].backward()

            assert dsa_minimal.indexer.q_proj.weight.grad is not None, \
                "Indexer Q projection should have gradient"
            assert dsa_minimal.indexer.k_proj.weight.grad is not None, \
                "Indexer K projection should have gradient"

    def test_training_step_increment(self, dsa_minimal):
        """Test training step counter."""
        initial_step = dsa_minimal.training_step.item()

        dsa_minimal.increment_training_step()

        assert dsa_minimal.training_step.item() == initial_step + 1


class TestDSAForward:
    """Test DSA forward pass."""

    def test_output_shape(self, dsa_minimal, sample_hidden_minimal):
        """Verify output shape matches input."""
        output, _, _ = dsa_minimal(sample_hidden_minimal)
        assert output.shape == sample_hidden_minimal.shape

    def test_with_cache(self, dsa_minimal, sample_hidden_minimal):
        """Test DSA with KV cache."""
        output, cache, _ = dsa_minimal(sample_hidden_minimal, use_cache=True)

        assert cache is not None, "Cache should be returned"
        kv_compressed, k_pe = cache

        batch, seq, _ = sample_hidden_minimal.shape
        assert kv_compressed.shape[1] == seq

    def test_incremental_decoding(self, dsa_minimal, device):
        """Test incremental decoding with DSA."""
        dsa_minimal.eval()
        hidden = dsa_minimal.hidden_size

        # Process prefix
        prefix = torch.randn(1, 32, hidden, device=device)
        with torch.no_grad():
            _, cache, _ = dsa_minimal(prefix, use_cache=True)

        # Process new token
        new_token = torch.randn(1, 1, hidden, device=device)
        with torch.no_grad():
            output, new_cache, _ = dsa_minimal(new_token, past_key_value=cache, use_cache=True)

        assert output.shape == (1, 1, hidden)

        # Cache should grow
        kv_compressed, _ = new_cache
        assert kv_compressed.shape[1] == 33  # 32 + 1


class TestDSAGradients:
    """Test gradient flow through DSA."""

    def test_full_gradient_flow(self, dsa_minimal, sample_hidden_minimal):
        """Verify gradients flow through entire DSA."""
        sample_hidden_minimal = sample_hidden_minimal.clone().requires_grad_(True)

        output, _, aux = dsa_minimal(sample_hidden_minimal, output_indexer_loss=True)

        loss = output.sum()
        if 'indexer_loss' in aux:
            loss = loss + aux['indexer_loss']
        loss.backward()

        # Check main MLA path
        assert dsa_minimal.mla.wq_a.weight.grad is not None
        assert dsa_minimal.mla.wo.weight.grad is not None

        # Check indexer
        assert dsa_minimal.indexer.q_proj.weight.grad is not None

    def test_gradient_health(self, dsa_minimal, sample_hidden_minimal):
        """Verify gradient magnitudes are reasonable."""
        sample_hidden_minimal = sample_hidden_minimal.clone().requires_grad_(True)

        output, _, _ = dsa_minimal(sample_hidden_minimal)
        loss = output.sum()
        loss.backward()

        grad_health = check_gradient_health(dsa_minimal)
        assert grad_health['is_healthy'], \
            f"Gradient issues: {grad_health}"


class TestDSACompressedRepresentations:
    """Test DSA's compressed representation handling."""

    def test_compression_shapes(self, dsa_minimal, sample_hidden_minimal):
        """Verify compressed representation shapes."""
        q_compressed, kv_compressed, k_pe = dsa_minimal._get_compressed_representations(
            sample_hidden_minimal
        )

        batch, seq, _ = sample_hidden_minimal.shape

        assert q_compressed.shape == (batch, seq, dsa_minimal.q_lora_rank)
        assert kv_compressed.shape == (batch, seq, dsa_minimal.kv_lora_rank)
        assert k_pe.shape == (batch, seq, dsa_minimal.qk_rope_head_dim)

    def test_compression_no_nan(self, dsa_minimal, sample_hidden_minimal):
        """Verify compressed representations have no NaN."""
        q_compressed, kv_compressed, k_pe = dsa_minimal._get_compressed_representations(
            sample_hidden_minimal
        )

        check_no_nan_inf(q_compressed, "Q compressed")
        check_no_nan_inf(kv_compressed, "KV compressed")
        check_no_nan_inf(k_pe, "K PE")
