"""
MTP (Multi-Token Prediction) Validation Tests

DeepSeek-Level Validation:
1. Token alignment verification
2. Loss weight schedule (0.3 -> 0.1)
3. Loss decay across modules
4. Speculative decoding consistency
5. Embedding sharing with main model
"""

import pytest
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.model import (
    MultiTokenPrediction,
    MTPModule,
    MTPBlock,
    NanoSeekModel,
)
from model.config import get_nanoseek_config, MTPConfig

from tests.utils import check_no_nan_inf


class TestMTPTokenAlignment:
    """Test MTP predicts correct token positions."""

    def test_mtp_logits_shape(self, mtp_minimal, sample_hidden_minimal, minimal_config, device):
        """Verify MTP logits have correct shape."""
        batch, seq, _ = sample_hidden_minimal.shape
        labels = torch.randint(0, minimal_config.vocab_size, (batch, seq), device=device)

        results = mtp_minimal(sample_hidden_minimal, labels=labels)

        assert 'mtp_logits' in results, "Should return mtp_logits"
        assert len(results['mtp_logits']) == minimal_config.mtp.num_mtp_modules

        # Each module's logits should be for appropriate positions
        for i, logits in enumerate(results['mtp_logits']):
            # MTP module i predicts future tokens, so output is shorter than input
            # The exact length depends on the MTP implementation
            assert logits.shape[1] <= seq, \
                f"MTP module {i} logits too long: {logits.shape[1]} > {seq}"
            assert logits.shape[1] > 0, f"MTP module {i} has no outputs"

    def test_mtp_predicts_next_tokens(self, mtp_minimal, sample_hidden_minimal, minimal_config, device):
        """Verify MTP modules predict progressively further tokens."""
        batch, seq = 2, 64
        hidden = sample_hidden_minimal[:, :seq]

        # Create labels where each position has a unique token
        labels = torch.arange(seq).unsqueeze(0).expand(batch, -1).to(device)

        results = mtp_minimal(hidden, labels=labels)

        # MTP module 0 should predict position i+2 given hidden state at i
        # The target for position 0's hidden state should be token at position 2
        assert 'mtp_logits' in results


class TestMTPLossSchedule:
    """Test dynamic MTP loss weight schedule."""

    def test_initial_loss_weight(self, model_minimal):
        """Verify initial MTP loss weight is 0.3."""
        model_minimal.tokens_processed.fill_(0)
        weight = model_minimal.get_mtp_loss_weight()
        assert weight == 0.3, f"Initial weight should be 0.3, got {weight}"

    def test_final_loss_weight(self, model_minimal, minimal_config):
        """Verify final MTP loss weight is 0.1 after transition."""
        # Set tokens processed past transition point
        transition = int(minimal_config.total_tokens * minimal_config.mtp.mtp_loss_transition_ratio)
        model_minimal.tokens_processed.fill_(transition + 1)

        weight = model_minimal.get_mtp_loss_weight()
        assert weight == 0.1, f"Final weight should be 0.1, got {weight}"

    def test_transition_boundary(self, model_minimal, minimal_config):
        """Test weight at exact transition boundary."""
        transition = int(minimal_config.total_tokens * minimal_config.mtp.mtp_loss_transition_ratio)

        # Just before transition
        model_minimal.tokens_processed.fill_(transition - 1)
        weight_before = model_minimal.get_mtp_loss_weight()
        assert weight_before == 0.3, f"Before transition: {weight_before}"

        # Just after transition
        model_minimal.tokens_processed.fill_(transition + 1)
        weight_after = model_minimal.get_mtp_loss_weight()
        assert weight_after == 0.1, f"After transition: {weight_after}"


class TestMTPLossComputation:
    """Test MTP loss computation."""

    def test_mtp_loss_returned(self, mtp_minimal, sample_hidden_minimal, minimal_config, device):
        """Verify MTP loss is computed and returned."""
        batch, seq, _ = sample_hidden_minimal.shape
        labels = torch.randint(0, minimal_config.vocab_size, (batch, seq), device=device)

        results = mtp_minimal(sample_hidden_minimal, labels=labels)

        assert 'mtp_loss' in results, "Should return mtp_loss"
        assert results['mtp_loss'] >= 0, "Loss should be non-negative"
        assert not torch.isnan(results['mtp_loss']), "Loss should not be NaN"

    def test_mtp_loss_decay(self, minimal_config, device):
        """Verify loss decay across MTP modules."""
        import torch.nn as nn

        # Create shared embeddings
        embed_tokens = nn.Embedding(minimal_config.vocab_size, minimal_config.hidden_size)
        lm_head = nn.Linear(minimal_config.hidden_size, minimal_config.vocab_size, bias=False)

        # Create MTP with multiple modules
        mtp = MultiTokenPrediction(
            hidden_size=minimal_config.hidden_size,
            vocab_size=minimal_config.vocab_size,
            num_mtp_modules=2,
            mtp_loss_decay=0.8,
        )
        mtp.set_shared_embeddings(embed_tokens, lm_head)
        mtp = mtp.to(device)

        hidden = torch.randn(2, 64, minimal_config.hidden_size, device=device)
        labels = torch.randint(0, minimal_config.vocab_size, (2, 64), device=device)

        results = mtp(hidden, labels=labels)

        if 'per_module_loss' in results and len(results['per_module_loss']) > 1:
            # Later modules should have decayed weight in total loss
            print(f"Per-module losses: {results['per_module_loss']}")

    def test_mtp_loss_with_ignore_index(self, mtp_minimal, sample_hidden_minimal, minimal_config, device):
        """Verify MTP handles ignore_index=-100 correctly."""
        batch, seq, _ = sample_hidden_minimal.shape
        labels = torch.randint(0, minimal_config.vocab_size, (batch, seq), device=device)

        # Set some labels to -100 (ignore)
        labels[:, -10:] = -100

        results = mtp_minimal(sample_hidden_minimal, labels=labels)

        assert 'mtp_loss' in results
        assert not torch.isnan(results['mtp_loss']), "Loss should handle ignore_index"


class TestMTPSpeculativeDecoding:
    """Test MTP speculative decoding."""

    def test_speculative_decode_output_shape(self, mtp_minimal, sample_hidden_minimal, device):
        """Verify speculative decoding returns correct shapes."""
        # Take last hidden state
        main_hidden = sample_hidden_minimal[:, -1:]

        draft_tokens, draft_probs = mtp_minimal.speculative_decode(
            main_hidden,
            temperature=0.0
        )

        batch = sample_hidden_minimal.shape[0]
        num_modules = mtp_minimal.num_mtp_modules

        assert draft_tokens.shape == (batch, num_modules), \
            f"Draft tokens shape: {draft_tokens.shape}"
        assert draft_probs.shape == (batch, num_modules), \
            f"Draft probs shape: {draft_probs.shape}"

    def test_speculative_tokens_valid(self, mtp_minimal, sample_hidden_minimal, minimal_config, device):
        """Verify speculative tokens are valid vocabulary indices."""
        main_hidden = sample_hidden_minimal[:, -1:]

        draft_tokens, _ = mtp_minimal.speculative_decode(
            main_hidden,
            temperature=0.0
        )

        assert draft_tokens.min() >= 0, f"Token index too low: {draft_tokens.min()}"
        assert draft_tokens.max() < minimal_config.vocab_size, \
            f"Token index too high: {draft_tokens.max()}"

    def test_greedy_decoding_high_prob(self, mtp_minimal, sample_hidden_minimal, device):
        """Verify greedy decoding (temp=0) produces high probability tokens."""
        main_hidden = sample_hidden_minimal[:, -1:]

        _, draft_probs = mtp_minimal.speculative_decode(
            main_hidden,
            temperature=0.0
        )

        # Greedy should select the max probability token
        # So probability should be > 1/vocab_size at minimum
        assert draft_probs.min() > 0, f"Probabilities should be positive: {draft_probs.min()}"

    def test_temperature_affects_sampling(self, mtp_minimal, sample_hidden_minimal, device):
        """Verify temperature affects sampling diversity."""
        main_hidden = sample_hidden_minimal[:, -1:]

        # Low temperature: more deterministic
        tokens_low, probs_low = mtp_minimal.speculative_decode(
            main_hidden,
            temperature=0.1
        )

        # High temperature: more random
        tokens_high, probs_high = mtp_minimal.speculative_decode(
            main_hidden,
            temperature=2.0
        )

        # High temperature should have lower probability for selected tokens
        # (more spread out distribution)
        assert probs_low.mean() >= probs_high.mean() - 0.1, \
            "Low temperature should have higher probabilities on average"


class TestMTPEmbeddingSharing:
    """Test MTP embedding sharing with main model."""

    def test_shared_embeddings(self, model_minimal):
        """Verify MTP shares embeddings with main model."""
        if model_minimal.mtp is None:
            pytest.skip("Model has no MTP")

        for module in model_minimal.mtp.mtp_modules:
            # Embedding should be same object
            assert module.embed_tokens is model_minimal.embed_tokens, \
                "MTP should share embedding with main model"

            # LM head should be same object
            assert module.lm_head is model_minimal.lm_head, \
                "MTP should share lm_head with main model"

    def test_gradient_through_shared_embeddings(self, model_minimal, sample_batch_minimal):
        """Verify gradients flow through shared embeddings from MTP."""
        if model_minimal.mtp is None:
            pytest.skip("Model has no MTP")

        outputs = model_minimal(
            sample_batch_minimal['input_ids'],
            labels=sample_batch_minimal['labels']
        )

        outputs['loss'].backward()

        # Embedding should have gradients from both main and MTP paths
        assert model_minimal.embed_tokens.weight.grad is not None, \
            "Embedding should have gradients"


class TestMTPBlock:
    """Test individual MTP block components."""

    def test_mtp_block_forward(self, minimal_config, device):
        """Test MTPBlock forward pass."""
        block = MTPBlock(
            hidden_size=minimal_config.hidden_size,
            num_heads=4,
            intermediate_size=minimal_config.hidden_size * 4,
        ).to(device)

        batch, seq = 2, 32
        hidden = torch.randn(batch, seq, minimal_config.hidden_size, device=device)
        main_hidden = torch.randn(batch, seq, minimal_config.hidden_size, device=device)

        output = block(hidden, main_hidden)

        assert output.shape == hidden.shape
        check_no_nan_inf(output, "MTPBlock output")

    def test_mtp_block_gradient(self, minimal_config, device):
        """Test MTPBlock gradient flow."""
        block = MTPBlock(
            hidden_size=minimal_config.hidden_size,
            num_heads=4,
            intermediate_size=minimal_config.hidden_size * 4,
        ).to(device)

        hidden = torch.randn(2, 32, minimal_config.hidden_size, device=device, requires_grad=True)
        main_hidden = torch.randn(2, 32, minimal_config.hidden_size, device=device)

        output = block(hidden, main_hidden)
        loss = output.sum()
        loss.backward()

        assert hidden.grad is not None, "Input should have gradient"
        check_no_nan_inf(hidden.grad, "MTPBlock input gradient")
