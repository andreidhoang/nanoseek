"""
Training correctness tests for NanoSeek.

These tests verify the complete training pipeline works correctly:
1. seq_aux_loss_alpha=0 produces zero aux loss
2. Forward → backward → optimizer step → loss decreases
3. Checkpoint save → load produces identical logits
4. Muon optimizer step is numerically healthy

These fill the critical test coverage gaps identified during the codebase audit.
No existing tests cover the optimizer, checkpointing, or end-to-end training steps.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nanoseek.model import NanoSeekModel, Gate, MoE
from nanoseek.config import NanoSeekConfig
from nanoseek.checkpoint_manager import save_checkpoint, load_checkpoint


# =============================================================================
# Shared fixtures
# =============================================================================

def _make_tiny_config(**overrides):
    """Create a minimal config for fast tests."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kwargs = dict(
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
            num_mtp_modules=0,  # Disable MTP for speed
            gradient_checkpointing=False,
        )
        kwargs.update(overrides)
        return NanoSeekConfig(**kwargs)


@pytest.fixture
def tiny_config():
    return _make_tiny_config()


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# Test 1: seq_aux_loss_alpha=0 produces zero aux loss
# =============================================================================

class TestSeqAuxLossAlpha:
    """Verify the ablation knob seq_aux_loss_alpha works correctly."""

    def test_alpha_zero_produces_zero_aux_loss(self):
        """When alpha=0, aux_loss must be exactly 0.0 — ablation validity depends on this."""
        gate = Gate(
            hidden_dim=128,
            n_routed_experts=16,
            num_experts_per_tok=4,
            n_group=4,
            topk_group=2,
            seq_aux_loss_alpha=0.0,
        )
        x = torch.randn(4, 8, 128)  # [B, S, D]
        flat_x = x.view(-1, 128)
        weights, indices, aux_loss, metadata = gate(flat_x)
        assert aux_loss.item() == 0.0, f"Expected aux_loss=0.0 with alpha=0, got {aux_loss.item()}"

    def test_alpha_nonzero_produces_nonzero_aux_loss(self):
        """When alpha>0, aux_loss should be positive (load is never perfectly uniform)."""
        gate = Gate(
            hidden_dim=128,
            n_routed_experts=16,
            num_experts_per_tok=4,
            n_group=4,
            topk_group=2,
            seq_aux_loss_alpha=0.01,
        )
        x = torch.randn(4, 8, 128)
        flat_x = x.view(-1, 128)
        weights, indices, aux_loss, metadata = gate(flat_x)
        assert aux_loss.item() > 0.0, f"Expected positive aux_loss with alpha=0.01, got {aux_loss.item()}"

    def test_alpha_scales_linearly(self):
        """Doubling alpha should approximately double aux_loss (same input)."""
        torch.manual_seed(42)
        x = torch.randn(8, 16, 128)
        flat_x = x.view(-1, 128)

        gate_a = Gate(hidden_dim=128, n_routed_experts=16, num_experts_per_tok=4,
                      n_group=4, topk_group=2, seq_aux_loss_alpha=0.001)
        gate_b = Gate(hidden_dim=128, n_routed_experts=16, num_experts_per_tok=4,
                      n_group=4, topk_group=2, seq_aux_loss_alpha=0.002)
        # Share weights so routing decisions are identical
        gate_b.load_state_dict(gate_a.state_dict())

        _, _, loss_a, _ = gate_a(flat_x)
        _, _, loss_b, _ = gate_b(flat_x)

        ratio = loss_b.item() / max(loss_a.item(), 1e-12)
        assert 1.9 < ratio < 2.1, f"Expected ~2x ratio, got {ratio:.3f}"

    def test_alpha_zero_in_full_model(self, tiny_config, device):
        """End-to-end: alpha=0 in full NanoSeekModel produces zero avg aux_loss."""
        config = _make_tiny_config(seq_aux_loss_alpha=0.0)
        model = NanoSeekModel(config).to(device)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (2, 32), device=device)
        labels = torch.randint(0, config.vocab_size, (2, 32), device=device)

        with torch.no_grad():
            out = model(input_ids, labels=labels)

        assert out["aux_loss"].item() == 0.0, (
            f"Expected aux_loss=0.0 in full model with alpha=0, got {out['aux_loss'].item()}"
        )


# =============================================================================
# Test 2: End-to-end training step — loss decreases
# =============================================================================

class TestTrainingStep:
    """Verify the complete training pipeline: forward → backward → step → loss drops."""

    def test_loss_decreases_over_steps(self, tiny_config, device):
        """Train for 5 steps, loss must decrease. Uses plain Adam for simplicity."""
        torch.manual_seed(42)
        model = NanoSeekModel(tiny_config).to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 32), device=device)
        labels = input_ids.clone()  # Teacher forcing — should be learnable

        losses = []
        for step in range(8):
            optimizer.zero_grad()
            out = model(input_ids, labels=labels)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}\n"
            f"All losses: {[f'{l:.4f}' for l in losses]}"
        )

    def test_bias_update_after_training_step(self, tiny_config, device):
        """Verify update_load_balance_bias uses _layer_aux_data from forward pass."""
        torch.manual_seed(42)
        model = NanoSeekModel(tiny_config).to(device)
        model.train()

        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 32), device=device)
        labels = input_ids.clone()

        # Forward pass populates _layer_aux_data
        out = model(input_ids, labels=labels)
        out["loss"].backward()

        # _layer_aux_data should have entries for MoE layers
        assert len(model._layer_aux_data) > 0, "No _layer_aux_data after forward pass"

        # Capture bias before update
        moe_layer = None
        for layer in model.layers:
            if hasattr(layer.ffn, 'gate'):
                moe_layer = layer.ffn
                break
        assert moe_layer is not None, "No MoE layer found"
        bias_before = moe_layer.gate.bias.clone()

        # Update bias
        model.update_load_balance_bias(tokens_processed=1000, total_tokens=tiny_config.total_tokens)

        bias_after = moe_layer.gate.bias
        # Bias should change (load is never perfectly uniform with random init)
        assert not torch.allclose(bias_before, bias_after, atol=1e-10), (
            "Bias did not change after update_load_balance_bias"
        )

    def test_gradient_health_after_backward(self, tiny_config, device):
        """Verify no NaN/Inf gradients after backward through MoE."""
        torch.manual_seed(42)
        model = NanoSeekModel(tiny_config).to(device)
        model.train()

        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 32), device=device)
        labels = input_ids.clone()

        out = model(input_ids, labels=labels)
        out["loss"].backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

    def test_aux_loss_in_total_loss(self, device):
        """Verify aux_loss contributes to total_loss when alpha > 0."""
        config_with_aux = _make_tiny_config(seq_aux_loss_alpha=0.01)
        config_without_aux = _make_tiny_config(seq_aux_loss_alpha=0.0)

        torch.manual_seed(42)
        model_with = NanoSeekModel(config_with_aux).to(device)
        torch.manual_seed(42)
        model_without = NanoSeekModel(config_without_aux).to(device)

        input_ids = torch.randint(0, config_with_aux.vocab_size, (2, 32), device=device)
        labels = input_ids.clone()

        out_with = model_with(input_ids, labels=labels)
        out_without = model_without(input_ids, labels=labels)

        # Total loss with aux should be >= total loss without
        # (aux loss is always non-negative)
        assert out_with["loss"].item() >= out_without["loss"].item() - 1e-6, (
            f"Loss with aux ({out_with['loss'].item():.6f}) < "
            f"loss without ({out_without['loss'].item():.6f})"
        )


# =============================================================================
# Test 3: Checkpoint round-trip — save → load → identical logits
# =============================================================================

class TestCheckpointRoundTrip:
    """Verify checkpoint save/load preserves model state exactly."""

    def test_save_load_identical_logits(self, tiny_config, device):
        """Save model, load into fresh model, verify identical logits."""
        torch.manual_seed(42)
        model1 = NanoSeekModel(tiny_config).to(device)
        model1.eval()

        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 32), device=device)

        with torch.no_grad():
            logits1 = model1(input_ids)["logits"]

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(
                checkpoint_dir=tmpdir,
                step=0,
                model_state=model1.state_dict(),
                optimizer_state=None,
                metadata={"step": 0, "tokens_processed": 0},
            )

            # Load into fresh model
            model_state, metadata, loaded_step = load_checkpoint(tmpdir, device=device)

            model2 = NanoSeekModel(tiny_config).to(device)
            model2.load_state_dict(model_state)
            model2.eval()

            with torch.no_grad():
                logits2 = model2(input_ids)["logits"]

        assert torch.allclose(logits1, logits2, atol=1e-5), (
            f"Logits differ after checkpoint round-trip. "
            f"Max diff: {(logits1 - logits2).abs().max().item():.2e}"
        )

    def test_save_load_metadata_preserved(self, tiny_config, device):
        """Verify metadata survives save/load."""
        model = NanoSeekModel(tiny_config).to(device)

        metadata = {
            "step": 42,
            "tokens_processed": 123456,
            "config": {"hidden_size": 256},
            "val_loss": 3.14,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(
                checkpoint_dir=tmpdir,
                step=42,
                model_state=model.state_dict(),
                optimizer_state=None,
                metadata=metadata,
            )

            _, loaded_metadata, loaded_step = load_checkpoint(tmpdir, device=device)

        assert loaded_step == 42
        assert loaded_metadata["step"] == 42
        assert loaded_metadata["tokens_processed"] == 123456
        assert loaded_metadata["val_loss"] == 3.14

    def test_latest_json_points_to_latest(self, tiny_config, device):
        """Save multiple checkpoints, latest.json should point to the last one."""
        model = NanoSeekModel(tiny_config).to(device)

        with tempfile.TemporaryDirectory() as tmpdir:
            for step in [10, 20, 30]:
                save_checkpoint(
                    checkpoint_dir=tmpdir,
                    step=step,
                    model_state=model.state_dict(),
                    optimizer_state=None,
                    metadata={"step": step},
                )

            _, metadata, loaded_step = load_checkpoint(tmpdir, device=device)

        assert loaded_step == 30, f"Expected latest step=30, got {loaded_step}"

    def test_atomic_write_creates_no_tmp_files(self, tiny_config, device):
        """After successful save, no .tmp files should remain."""
        model = NanoSeekModel(tiny_config).to(device)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(
                checkpoint_dir=tmpdir,
                step=0,
                model_state=model.state_dict(),
                optimizer_state=None,
                metadata={"step": 0},
            )

            tmp_files = list(Path(tmpdir).glob("*.tmp"))
            assert len(tmp_files) == 0, f"Leftover .tmp files: {tmp_files}"


# =============================================================================
# Test 4: Muon optimizer smoke test
# =============================================================================

class TestMuonSmoke:
    """Verify MuonAdamW optimizer doesn't produce NaN/Inf and updates params."""

    def test_muon_single_step(self, tiny_config, device):
        """Single Muon step: params change, no NaN/Inf."""
        from nanoseek.optim import MuonAdamW

        model = NanoSeekModel(tiny_config).to(device)
        model.train()

        # Build param groups matching pre_train.py pattern
        muon_params = {}
        adamw_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim >= 2 and 'embed' not in name and 'lm_head' not in name and 'gate.linear' not in name and 'norm' not in name:
                key = p.shape
                if key not in muon_params:
                    muon_params[key] = []
                muon_params[key].append(p)
            else:
                adamw_params.append(p)

        param_groups = []
        if adamw_params:
            param_groups.append(dict(
                kind='adamw', params=adamw_params,
                lr=1e-3, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0,
            ))
        for shape, params in muon_params.items():
            param_groups.append(dict(
                kind='muon', params=params,
                lr=0.01, momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=0.01,
            ))

        optimizer = MuonAdamW(param_groups)

        # Snapshot params before step
        params_before = {n: p.clone() for n, p in model.named_parameters() if p.requires_grad}

        # Forward + backward
        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 32), device=device)
        out = model(input_ids, labels=input_ids)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        # Verify: params changed, no NaN/Inf
        any_changed = False
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            assert not torch.isnan(p).any(), f"NaN in param {name} after Muon step"
            assert not torch.isinf(p).any(), f"Inf in param {name} after Muon step"
            if not torch.allclose(p, params_before[name], atol=1e-10):
                any_changed = True

        assert any_changed, "No parameters changed after optimizer step"

    def test_muon_loss_decreases(self, tiny_config, device):
        """Train with MuonAdamW for several steps, loss should decrease."""
        from nanoseek.optim import MuonAdamW

        torch.manual_seed(42)
        model = NanoSeekModel(tiny_config).to(device)
        model.train()

        # Simple grouping: everything into AdamW (tests the AdamW path)
        # Muon path tested separately above
        param_groups = [dict(
            kind='adamw',
            params=[p for p in model.parameters() if p.requires_grad],
            lr=1e-3, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0,
        )]
        optimizer = MuonAdamW(param_groups)

        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 32), device=device)

        losses = []
        for _ in range(8):
            optimizer.zero_grad()
            out = model(input_ids, labels=input_ids)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(out["loss"].item())

        assert losses[-1] < losses[0], (
            f"Loss didn't decrease with MuonAdamW: {losses[0]:.4f} → {losses[-1]:.4f}"
        )


# =============================================================================
# Test 5: Bias update affects routing (functional test)
# =============================================================================

class TestBiasAffectsRouting:
    """Verify that modified bias actually changes expert selection."""

    def test_large_bias_attracts_tokens(self):
        """Setting one expert's bias very high should attract most tokens."""
        gate = Gate(
            hidden_dim=128,
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_group=2,
            topk_group=1,
            seq_aux_loss_alpha=0.0001,
        )
        torch.manual_seed(42)
        x = torch.randn(16, 128)  # 16 tokens

        # Baseline: count tokens routed to expert 0
        _, indices_before, _, _ = gate(x)
        count_before = (indices_before == 0).sum().item()

        # Set large bias for expert 0 — should attract more tokens
        gate.bias.data[0] = 10.0

        _, indices_after, _, _ = gate(x)
        count_after = (indices_after == 0).sum().item()

        assert count_after > count_before, (
            f"Bias=10 for expert 0 didn't attract more tokens: "
            f"before={count_before}, after={count_after}"
        )

    def test_negative_bias_repels_tokens(self):
        """Setting one expert's bias very negative should repel tokens."""
        gate = Gate(
            hidden_dim=128,
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_group=2,
            topk_group=1,
            seq_aux_loss_alpha=0.0001,
        )
        torch.manual_seed(42)
        x = torch.randn(32, 128)

        _, indices_before, _, _ = gate(x)
        count_before = (indices_before == 0).sum().item()

        gate.bias.data[0] = -10.0

        _, indices_after, _, _ = gate(x)
        count_after = (indices_after == 0).sum().item()

        assert count_after < count_before, (
            f"Bias=-10 for expert 0 didn't repel tokens: "
            f"before={count_before}, after={count_after}"
        )
