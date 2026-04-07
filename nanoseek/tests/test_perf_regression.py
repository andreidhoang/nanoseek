"""
Performance regression tests for NanoSeek.

These tests run on CPU and catch Python-level performance regressions:
- Accidental .item() calls in the forward hot path
- Unnecessary GPU-CPU sync points
- Debug prints left in model code

They do NOT measure absolute GPU throughput — use scripts/benchmark.py for that.
They DO catch regressions that silently kill throughput with no test failure.

Usage:
    pytest tests/test_perf_regression.py -v
    pytest tests/test_perf_regression.py -v -k "sync_points"
"""

import time
import warnings
import unittest.mock as mock

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "nanoseek"))

from config import NanoSeekConfig
from model import NanoSeekModel, MoE


# ---------------------------------------------------------------------------
# Session-scoped fixtures (build model ONCE, reuse across all tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def perf_config():
    """Small config for perf tests — fast to construct, exercises all code paths."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return NanoSeekConfig(
            hidden_size=256,
            num_layers=2,
            num_heads=4,
            intermediate_size=512,
            vocab_size=1000,
            max_position_embeddings=128,
            total_tokens=1_000_000,
            global_batch_size=4,
            sequence_length=64,
            _qk_nope_head_dim=48,
            _qk_rope_head_dim=16,
            _v_head_dim=48,
            _q_lora_rank_override=55,
            _kv_lora_rank_override=18,
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_shared_experts=1,
            _moe_intermediate_size_override=128,
            n_group=2,
            topk_group=1,
            first_k_dense_replace=1,
            num_mtp_modules=1,
        )


@pytest.fixture(scope="module")
def perf_model(perf_config):
    """Model on CPU for perf regression testing. Built once per module."""
    model = NanoSeekModel(perf_config)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

class ItemCallTracker:
    """Monkey-patches torch.Tensor.item to count calls during a scope."""

    def __init__(self):
        self.call_count = 0
        self._original_item = torch.Tensor.item

    def __enter__(self):
        tracker = self

        def tracked_item(tensor_self):
            tracker.call_count += 1
            return tracker._original_item(tensor_self)

        torch.Tensor.item = tracked_item
        return self

    def __exit__(self, *args):
        torch.Tensor.item = self._original_item


# Known .item() calls in MoE dispatch (model.py lines 748, 831).
# Documented in PERFORMANCE_OPTIMIZATION.md T1.2 for removal.
# Once fixed, change to 0 to prevent regressions.
KNOWN_ITEM_CALLS = 1  # 1 per MoE layer (lines 748+831 share a code path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSyncPoints:
    """Verify no NEW GPU-CPU sync points in the forward path."""

    def test_forward_no_new_item_calls(self, perf_model, perf_config):
        """Forward+loss must not introduce NEW .item() calls beyond known ones.

        Known sync points (to fix via PERFORMANCE_OPTIMIZATION.md T1.2):
        - model.py:748 — expert_counts.max().item() in _batched_expert_forward
        - model.py:831 — expert_counts.max().item() in dispatch routing decision
        """
        B, S = 2, perf_config.sequence_length
        ids = torch.randint(0, perf_config.vocab_size, (B, S))

        with torch.no_grad(), ItemCallTracker() as tracker:
            _ = perf_model(ids, labels=ids, mtp_lambda=0.1)

        assert tracker.call_count <= KNOWN_ITEM_CALLS, (
            f"Forward+loss called .item() {tracker.call_count} time(s), "
            f"but only {KNOWN_ITEM_CALLS} are known. "
            f"A new .item() was introduced — find and remove it."
        )

    def test_no_prints_during_forward(self, perf_model, perf_config):
        """Model forward must not call print() — it forces string formatting + I/O."""
        B, S = 2, perf_config.sequence_length
        ids = torch.randint(0, perf_config.vocab_size, (B, S))

        with mock.patch("builtins.print") as mock_print:
            with torch.no_grad():
                _ = perf_model(ids, labels=ids, mtp_lambda=0.1)

        assert mock_print.call_count == 0, (
            f"Forward pass called print() {mock_print.call_count} time(s). "
            f"Remove debug prints from model.py hot path."
        )


class TestMoEDispatchPerf:
    """Verify MoE dispatch path is sync-free (isolated from full model)."""

    def test_moe_forward_no_new_item_calls(self, perf_config):
        """MoE forward must not introduce NEW .item() calls for routing."""
        moe = MoE(
            hidden_dim=perf_config.hidden_size,
            moe_inter_dim=perf_config.moe.moe_intermediate_size,
            n_routed_experts=perf_config.moe.n_routed_experts,
            num_experts_per_tok=perf_config.moe.num_experts_per_tok,
            n_shared_experts=perf_config.moe.n_shared_experts,
            n_group=perf_config.moe.n_group,
            topk_group=perf_config.moe.topk_group,
            scoring_func=perf_config.moe.scoring_func,
            routed_scaling_factor=perf_config.moe.routed_scaling_factor,
            seq_aux_loss_alpha=perf_config.moe.seq_aux_loss_alpha,
        )
        moe.eval()

        B, S, D = 2, perf_config.sequence_length, perf_config.hidden_size
        x = torch.randn(B, S, D)

        with torch.no_grad(), ItemCallTracker() as tracker:
            _ = moe(x)

        assert tracker.call_count <= KNOWN_ITEM_CALLS, (
            f"MoE forward called .item() {tracker.call_count} time(s), "
            f"but only {KNOWN_ITEM_CALLS} are known. "
            f"The dispatch path must stay on-device for torch.compile compatibility."
        )
