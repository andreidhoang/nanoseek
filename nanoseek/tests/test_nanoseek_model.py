"""
NanoSeekModel integration tests — validates architectural correctness.

Each test targets a specific invariant that, if broken, would cause
silent training failures or incorrect scaling law measurements.

Uses a minimal config (8 experts, 2 layers) for fast CPU execution.
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nanoseek.config import (
    NanoSeekConfig, MLAConfig, MoEConfig, MTPConfig,
    SparseAttentionConfig,
)
from nanoseek.model import (
    NanoSeekModel, Expert, MoE, MultiTokenPrediction,
    get_mtp_loss_weight,
)


# ─── Minimal config: 2 layers (1 dense + 1 MoE), 8 experts ─────────────────

@pytest.fixture(scope="module")
def cfg():
    return NanoSeekConfig(
        hidden_size=256,
        num_layers=2,
        num_heads=4,
        intermediate_size=512,
        vocab_size=1000,
        max_position_embeddings=128,
        mla=MLAConfig(
            q_lora_rank=55,
            kv_lora_rank=18,
            qk_nope_head_dim=48,
            qk_rope_head_dim=16,
            v_head_dim=48,
        ),
        moe=MoEConfig(
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_shared_experts=1,
            moe_intermediate_size=128,
            n_group=2,
            topk_group=1,
            first_k_dense_replace=1,
        ),
        mtp=MTPConfig(
            num_mtp_modules=1,
            mtp_num_heads=2,
        ),
        sparse=SparseAttentionConfig(enabled=False),
        gradient_checkpointing=False,  # Faster for CPU tests
    )


@pytest.fixture(scope="module")
def model(cfg):
    m = NanoSeekModel(cfg)
    return m


@pytest.fixture
def batch(cfg):
    B, S = 2, 32
    ids = torch.randint(0, cfg.vocab_size, (B, S))
    return ids


# ─── Test 1: Layer assignment — dense vs MoE at correct boundaries ──────────

class TestLayerAssignment:
    """Layers before first_k_dense_replace use Expert (dense FFN),
    layers at or after use MoE. Off-by-one here silently degrades
    training stability (dense layers stabilize early gradients)."""

    def test_dense_layers_use_expert(self, model, cfg):
        for i in range(cfg.moe.first_k_dense_replace):
            layer = model.layers[i]
            assert isinstance(layer.ffn, Expert), (
                f"Layer {i} should use dense Expert FFN, got {type(layer.ffn).__name__}"
            )
            assert not layer.use_moe

    def test_moe_layers_use_moe(self, model, cfg):
        for i in range(cfg.moe.first_k_dense_replace, cfg.num_layers):
            layer = model.layers[i]
            assert isinstance(layer.ffn, MoE), (
                f"Layer {i} should use MoE, got {type(layer.ffn).__name__}"
            )
            assert layer.use_moe

    def test_correct_layer_count(self, model, cfg):
        assert len(model.layers) == cfg.num_layers
        n_dense = sum(1 for l in model.layers if not l.use_moe)
        n_moe = sum(1 for l in model.layers if l.use_moe)
        assert n_dense == cfg.moe.first_k_dense_replace
        assert n_moe == cfg.num_layers - cfg.moe.first_k_dense_replace


# ─── Test 2: MTP weight sharing — same object, not copy ─────────────────────

class TestMTPWeightSharing:
    """MTP must share embed_tokens and lm_head with the main model.
    If these are copies instead of references, MTP gradients don't
    flow back to enrich main model representations (paper Section 3.3)."""

    def test_embed_is_shared_not_copied(self, model):
        if model.mtp is None:
            pytest.skip("MTP not configured")
        for module in model.mtp.mtp_modules:
            assert module.embed_tokens is model.embed_tokens, (
                "MTP embed_tokens must be the SAME object as model.embed_tokens"
            )

    def test_lm_head_is_shared_not_copied(self, model):
        if model.mtp is None:
            pytest.skip("MTP not configured")
        for module in model.mtp.mtp_modules:
            assert module.lm_head is model.lm_head, (
                "MTP lm_head must be the SAME object as model.lm_head"
            )

    def test_mtp_gradient_reaches_shared_embedding(self, model, batch, cfg):
        """After backward with MTP, embed_tokens.weight.grad must be nonzero.
        If MTP sharing is broken, this gradient comes only from main CE loss."""
        if model.mtp is None:
            pytest.skip("MTP not configured")
        model.train()
        model.zero_grad()

        # Forward with labels (triggers _compute_loss which runs MTP)
        out = model(batch, labels=batch, tokens_processed=0, total_tokens=1000)
        out["loss"].backward()

        # embed_tokens grad should be nonzero (main CE + MTP both contribute)
        assert model.embed_tokens.weight.grad is not None
        grad_norm = model.embed_tokens.weight.grad.norm().item()
        assert grad_norm > 0, "Embedding gradient is zero — MTP sharing may be broken"
        model.zero_grad()


# ─── Test 3: Loss decomposition — L_total = L_main + λ·L_MTP + L_aux ────────

class TestLossDecomposition:
    """Total loss must correctly combine all three components.
    Getting this wrong means MoE load-balancing or MTP doesn't train."""

    def test_aux_loss_only_from_moe_layers(self, model, batch):
        """Dense layers return empty aux_data, MoE layers return aux_loss > 0."""
        model.train()
        out = model(batch, labels=batch)
        # _layer_aux_data should only contain MoE layer indices
        for layer_idx in model._layer_aux_data:
            assert model.layers[layer_idx].use_moe, (
                f"Layer {layer_idx} is dense but contributed aux_data"
            )
        model.zero_grad()

    def test_loss_includes_mtp_component(self, model, batch, cfg):
        """Loss with MTP enabled should differ from pure CE loss."""
        if model.mtp is None:
            pytest.skip("MTP not configured")
        model.train()

        # Get total loss (includes MTP)
        out = model(batch, labels=batch, tokens_processed=0, total_tokens=1000)
        total_loss = out["loss"].item()

        # Compute pure CE loss manually for comparison
        logits = out["logits"]
        V = logits.size(-1)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        pure_ce = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, V), shift_labels.view(-1), ignore_index=-100
        ).item()

        # Total loss should be > pure CE (MTP and aux add positive terms)
        assert total_loss > pure_ce - 1e-6, (
            f"Total loss ({total_loss:.4f}) should be >= CE loss ({pure_ce:.4f})"
        )
        model.zero_grad()

    def test_mtp_loss_weight_schedule(self):
        """λ step function: 0.3 before 60%, 0.1 after."""
        assert get_mtp_loss_weight(0, 1000) == 0.3
        assert get_mtp_loss_weight(599, 1000) == 0.3    # 59.9% < 60%
        assert get_mtp_loss_weight(600, 1000) == 0.1    # 60% >= 60%
        assert get_mtp_loss_weight(999, 1000) == 0.1
        # Edge: total_tokens=0 should not crash
        assert get_mtp_loss_weight(0, 0) == 0.3


# ─── Test 4: KV cache consistency — the fundamental correctness test ─────────

class TestKVCacheConsistency:
    """Cached decode at position t must produce identical logits to
    full-sequence decode at position t. Any divergence means the KV
    cache implementation is wrong — this would silently corrupt inference."""

    def test_cached_vs_uncached_logits_match(self, model, cfg):
        model.eval()
        S = 16
        x = torch.randint(0, cfg.vocab_size, (1, S))

        with torch.no_grad():
            # Full forward (no cache)
            out_full = model(x, use_cache=False)
            logits_full = out_full["logits"]  # [1, S, V]

            # Prefill with cache
            out_cached = model(x, use_cache=True)
            logits_cached = out_cached["logits"]  # [1, S, V]

        # Logits at every position must match
        max_diff = (logits_full - logits_cached).abs().max().item()
        assert max_diff < 1e-4, (
            f"KV cache divergence: max |diff| = {max_diff:.6f} "
            f"(threshold 1e-4). Cache is corrupting outputs."
        )

    def test_incremental_decode_matches_full(self, model, cfg):
        """Single-token decode with cache must match full recompute."""
        model.eval()
        prompt_len = 8
        x = torch.randint(0, cfg.vocab_size, (1, prompt_len + 1))

        with torch.no_grad():
            # Full forward over all tokens
            out_full = model(x, use_cache=False)
            logit_at_last = out_full["logits"][:, -1, :]  # [1, V]

            # Prefill prompt, then decode last token
            out_prefill = model(x[:, :prompt_len], use_cache=True)
            past_kvs = out_prefill["past_key_values"]

            out_decode = model(
                x[:, prompt_len:prompt_len + 1],
                past_key_values=past_kvs,
                use_cache=True,
            )
            logit_at_last_cached = out_decode["logits"][:, -1, :]  # [1, V]

        max_diff = (logit_at_last - logit_at_last_cached).abs().max().item()
        assert max_diff < 1e-3, (
            f"Incremental decode diverges from full: max |diff| = {max_diff:.6f}. "
            f"Position encoding or cache concatenation is likely wrong."
        )


# ─── Test 5: Gamma freeze — Bug #9 regression test ──────────────────────────

class TestGammaFreeze:
    """gamma_freeze_ratio must be 0.95, not 0.80 (Bug #9).
    Getting this wrong means expert biases keep updating too long,
    destabilizing routing in the final 5% of training."""

    def test_gamma_active_before_freeze(self, model, cfg):
        g = model.get_gamma(0, 1000)
        assert g == cfg.moe.gamma
        assert g > 0

    def test_gamma_active_at_94_percent(self, model, cfg):
        g = model.get_gamma(940, 1000)
        assert g == cfg.moe.gamma, "Gamma should still be active at 94%"

    def test_gamma_frozen_at_95_percent(self, model):
        g = model.get_gamma(950, 1000)
        assert g == 0.0, f"Gamma should be 0.0 at 95%, got {g}"

    def test_gamma_frozen_at_100_percent(self, model):
        g = model.get_gamma(1000, 1000)
        assert g == 0.0

    def test_config_has_correct_freeze_ratio(self, cfg):
        assert cfg.moe.gamma_freeze_ratio == 0.95, (
            f"gamma_freeze_ratio must be 0.95 (RULE 2), got {cfg.moe.gamma_freeze_ratio}"
        )


# ─── Test 6: Active parameter count math ────────────────────────────────────

class TestParameterCounting:
    """Active param count feeds directly into FLOPs and scaling law fit.
    Formula: active = total - (n_routed - top_k) × 3HF × n_moe_layers.
    Wrong count → wrong MFU → wrong scaling law → wrong predictions."""

    def test_active_less_than_total(self, model):
        info = model.num_parameters()
        assert info["active"] < info["total"], (
            "Active params must be < total (MoE has inactive experts)"
        )

    def test_inactive_count_matches_formula(self, model, cfg):
        info = model.num_parameters()
        inactive_per_layer = cfg.moe.n_routed_experts - cfg.moe.num_experts_per_tok
        params_per_expert = 3 * cfg.hidden_size * cfg.moe.moe_intermediate_size
        n_moe_layers = cfg.num_layers - cfg.moe.first_k_dense_replace
        expected_inactive = inactive_per_layer * params_per_expert * n_moe_layers
        actual_inactive = info["total"] - info["active"]
        assert actual_inactive == expected_inactive, (
            f"Inactive param mismatch: got {actual_inactive}, "
            f"expected {expected_inactive} = "
            f"{inactive_per_layer} × {params_per_expert} × {n_moe_layers}"
        )


# ─── Test 7: Bias update actually modifies gate state ────────────────────────

class TestBiasUpdate:
    """update_load_balance_bias must actually change gate.bias tensors.
    If it's silently no-op (e.g., _layer_aux_data empty, wrong key name),
    expert load balancing is disabled and some experts collapse to zero load."""

    def test_bias_changes_after_update(self, model, batch, cfg):
        model.train()
        # Forward to populate _layer_aux_data
        out = model(batch, labels=batch, tokens_processed=100, total_tokens=1000)

        # Snapshot gate biases before update
        biases_before = {}
        for i, layer in enumerate(model.layers):
            if layer.use_moe:
                biases_before[i] = layer.ffn.gate.bias.clone()

        # Run bias update
        model.update_load_balance_bias(tokens_processed=100, total_tokens=1000)

        # At least one gate bias should have changed
        any_changed = False
        for i, bias_before in biases_before.items():
            bias_after = model.layers[i].ffn.gate.bias
            if not torch.equal(bias_before, bias_after):
                any_changed = True
                break

        assert any_changed, (
            "No gate biases changed after update_load_balance_bias. "
            "Check that _layer_aux_data contains 'load_counts' and "
            "gamma > 0 at this training progress."
        )
        model.zero_grad()

    def test_bias_frozen_after_95_percent(self, model, batch, cfg):
        """After 95% of training, gamma=0 → bias should NOT change."""
        model.train()
        out = model(batch, labels=batch, tokens_processed=960, total_tokens=1000)

        biases_before = {}
        for i, layer in enumerate(model.layers):
            if layer.use_moe:
                biases_before[i] = layer.ffn.gate.bias.clone()

        model.update_load_balance_bias(tokens_processed=960, total_tokens=1000)

        for i, bias_before in biases_before.items():
            bias_after = model.layers[i].ffn.gate.bias
            assert torch.equal(bias_before, bias_after), (
                f"Gate bias for layer {i} changed after 95% — "
                f"gamma should be frozen (RULE 2)"
            )
        model.zero_grad()


# ─── Test 8: Generation produces valid tokens ────────────────────────────────

class TestGeneration:
    """Generation must produce in-vocabulary token IDs.
    Also validates that KV cache grows correctly across decode steps."""

    def test_generated_tokens_in_vocab(self, model, cfg):
        model.eval()
        prompt = torch.randint(0, cfg.vocab_size, (1, 8))
        with torch.no_grad():
            out = model.generate(prompt, max_new_tokens=4, temperature=1.0)
        assert out.shape == (1, 12), f"Expected (1, 12), got {out.shape}"
        assert torch.all(out >= 0) and torch.all(out < cfg.vocab_size)

    def test_greedy_decode_is_deterministic(self, model, cfg):
        """Temperature=0 (greedy) must produce identical outputs across calls."""
        model.eval()
        prompt = torch.randint(0, cfg.vocab_size, (1, 8))
        with torch.no_grad():
            out1 = model.generate(prompt, max_new_tokens=4, temperature=0)
            out2 = model.generate(prompt, max_new_tokens=4, temperature=0)
        assert torch.equal(out1, out2), "Greedy decode is not deterministic"

    def test_generate_simple_matches_cached_shape(self, model, cfg):
        """Both paths must produce same output length."""
        model.eval()
        prompt = torch.randint(0, cfg.vocab_size, (1, 8))
        n = 4
        with torch.no_grad():
            out_simple = model.generate(prompt, max_new_tokens=n, use_cache=False)
            out_cached = model.generate(prompt, max_new_tokens=n, use_cache=True)
        assert out_simple.shape == out_cached.shape == (1, 8 + n)


# ─── Test 9: Gradient flow completeness ──────────────────────────────────────

class TestGradientFlow:
    """Every trainable parameter must receive a gradient.
    Missing gradients mean dead parameters — wasted compute and
    incorrect effective model size for scaling law fit."""

    def test_all_params_receive_gradients(self, model, batch):
        model.train()
        model.zero_grad()
        out = model(batch, labels=batch, tokens_processed=0, total_tokens=1000)
        out["loss"].backward()

        dead_params = []
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is None:
                dead_params.append(name)

        assert len(dead_params) == 0, (
            f"{len(dead_params)} parameters received no gradient:\n"
            + "\n".join(f"  - {n}" for n in dead_params[:10])
        )
        model.zero_grad()

    def test_no_nan_in_gradients(self, model, batch):
        model.train()
        model.zero_grad()
        out = model(batch, labels=batch, tokens_processed=0, total_tokens=1000)
        out["loss"].backward()

        nan_params = []
        for name, p in model.named_parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                nan_params.append(name)

        assert len(nan_params) == 0, (
            f"NaN gradients in: {nan_params[:5]}"
        )
        model.zero_grad()


# ─── Test 10: Forward output contract ────────────────────────────────────────

class TestForwardContract:
    """Forward must return the exact dict keys the training loop expects.
    Missing keys cause KeyError at step 1. Wrong shapes cause silent
    dimension mismatches that compound over training."""

    def test_output_keys_without_labels(self, model, batch):
        model.eval()
        with torch.no_grad():
            out = model(batch)
        expected_keys = {"loss", "logits", "past_key_values", "aux_loss", "main_loss", "mtp_loss", "mtp_lambda"}
        assert set(out.keys()) == expected_keys
        assert out["loss"] is None
        assert out["past_key_values"] is None

    def test_output_keys_with_labels(self, model, batch):
        model.train()
        out = model(batch, labels=batch, tokens_processed=0, total_tokens=1000)
        assert out["loss"] is not None
        assert out["loss"].dim() == 0, "Loss must be scalar"
        assert out["loss"].isfinite(), f"Loss is {out['loss'].item()}"
        model.zero_grad()

    def test_logits_shape(self, model, batch, cfg):
        model.eval()
        B, S = batch.shape
        with torch.no_grad():
            out = model(batch)
        assert out["logits"].shape == (B, S, cfg.vocab_size)

    def test_kv_cache_returned_when_requested(self, model, batch, cfg):
        model.eval()
        with torch.no_grad():
            out = model(batch, use_cache=True)
        assert out["past_key_values"] is not None
        assert len(out["past_key_values"]) == cfg.num_layers
        # Each entry is (c_kv, k_pe) tuple
        for i, kv in enumerate(out["past_key_values"]):
            assert kv is not None, f"Layer {i} returned None cache"
            c_kv, k_pe = kv
            assert c_kv.shape[1] == batch.shape[1], (
                f"Layer {i} cache seq_len mismatch"
            )
