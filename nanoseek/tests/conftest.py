"""
NanoSeek Test Configuration and Shared Fixtures

DeepSeek-Level Validation Infrastructure
Provides reusable fixtures for testing all model components.
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add nanoseek/nanoseek/ to path so we can import config, model directly
sys.path.insert(0, str(Path(__file__).parent.parent / "nanoseek"))

from config import NanoSeekConfig, get_config
from model import (
    NanoSeekModel,
    MultiHeadLatentAttention,
    MoE,
    Gate,
    Expert,
    MultiTokenPrediction,
    MTPModule,
    MTPTransformerBlock,
    NanoSeekDecoderLayer,
    RMSNorm,
    CastLinear,
    precompute_freqs_cis,
    apply_rotary_emb,
)


# =============================================================================
# Device Configuration
# =============================================================================

def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture(scope="session")
def device():
    """Session-scoped device fixture."""
    return get_device()


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def config_1b():
    """NanoSeek-1B configuration - main config (1.08B active / 4.75B total)."""
    return get_config("1b")


@pytest.fixture
def minimal_config():
    """Minimal configuration for ultra-fast unit tests.

    Uses non-standard dims (256 hidden, head_dim=64) for speed.
    The warning about non-standard head_dim is expected.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return NanoSeekConfig(
            hidden_size=256,
            num_layers=2,
            num_heads=4,          # 256/4 = 64 head_dim (non-standard, OK for tests)
            intermediate_size=512,
            vocab_size=1000,
            max_position_embeddings=128,
            total_tokens=1_000_000,
            global_batch_size=4,
            sequence_length=64,
            # MLA overrides for small test
            _qk_nope_head_dim=48,
            _qk_rope_head_dim=16,
            _v_head_dim=48,
            _q_lora_rank_override=55,
            _kv_lora_rank_override=18,
            # MoE overrides for small test
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_shared_experts=1,
            _moe_intermediate_size_override=128,
            n_group=2,
            topk_group=1,
            first_k_dense_replace=1,
            num_mtp_modules=1,
        )


# =============================================================================
# Model Fixtures
# =============================================================================

@pytest.fixture
def model_minimal(minimal_config, device):
    """Minimal model for fast unit tests."""
    model = NanoSeekModel(minimal_config)
    return model.to(device)


@pytest.fixture
def model_1b(config_1b, device):
    """NanoSeek-1B model - main model (1.08B active / 4.75B total)."""
    model = NanoSeekModel(config_1b)
    return model.to(device)


# =============================================================================
# Component Fixtures
# =============================================================================

@pytest.fixture
def mla_minimal(minimal_config, device):
    """Minimal MLA for unit tests."""
    mla = MultiHeadLatentAttention(
        hidden_size=minimal_config.hidden_size,
        num_heads=minimal_config.num_heads,
        q_lora_rank=minimal_config.mla.q_lora_rank,
        kv_lora_rank=minimal_config.mla.kv_lora_rank,
        qk_nope_head_dim=minimal_config.mla.qk_nope_head_dim,
        qk_rope_head_dim=minimal_config.mla.qk_rope_head_dim,
        v_head_dim=minimal_config.mla.v_head_dim,
        max_position_embeddings=minimal_config.max_position_embeddings,
    )
    return mla.to(device)


@pytest.fixture
def moe_minimal(minimal_config, device):
    """Minimal MoE for unit tests."""
    moe = MoE(
        hidden_dim=minimal_config.hidden_size,
        moe_inter_dim=minimal_config.moe.moe_intermediate_size,
        n_routed_experts=minimal_config.moe.n_routed_experts,
        num_experts_per_tok=minimal_config.moe.num_experts_per_tok,
        n_shared_experts=minimal_config.moe.n_shared_experts,
        n_group=minimal_config.moe.n_group,
        topk_group=minimal_config.moe.topk_group,
        scoring_func=minimal_config.moe.scoring_func,
        routed_scaling_factor=minimal_config.moe.routed_scaling_factor,
        seq_aux_loss_alpha=minimal_config.moe.seq_aux_loss_alpha,
    )
    return moe.to(device)


@pytest.fixture
def mtp_minimal(minimal_config, device):
    """Minimal MTP for unit tests."""
    # Create embedding and lm_head for MTP to share (on device)
    embed_tokens = nn.Embedding(minimal_config.vocab_size, minimal_config.hidden_size).to(device)
    lm_head = nn.Linear(minimal_config.hidden_size, minimal_config.vocab_size, bias=False).to(device)

    mtp = MultiTokenPrediction(
        hidden_size=minimal_config.hidden_size,
        vocab_size=minimal_config.vocab_size,
        num_mtp_modules=minimal_config.mtp.num_mtp_modules,
        mtp_num_heads=minimal_config.mtp.mtp_num_heads,
        mtp_loss_weight=minimal_config.mtp.mtp_loss_weight_initial,
        mtp_loss_decay=minimal_config.mtp.mtp_loss_decay,
    ).to(device)

    # Set shared embeddings (like the model does)
    mtp.set_shared_embeddings(embed_tokens, lm_head)

    return mtp


# DSA and Indexer fixtures omitted — Phase 2 (Sections 8-9 not yet implemented)


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_batch_minimal(minimal_config, device):
    """Sample batch for minimal config."""
    batch_size, seq_len = 2, 64
    return {
        'input_ids': torch.randint(0, minimal_config.vocab_size, (batch_size, seq_len), device=device),
        'labels': torch.randint(0, minimal_config.vocab_size, (batch_size, seq_len), device=device),
        'attention_mask': torch.ones(batch_size, seq_len, dtype=torch.bool, device=device),
    }


@pytest.fixture
def sample_batch_1b(config_1b, device):
    """Sample batch for 1B config."""
    batch_size, seq_len = 2, 128
    return {
        'input_ids': torch.randint(0, config_1b.vocab_size, (batch_size, seq_len), device=device),
        'labels': torch.randint(0, config_1b.vocab_size, (batch_size, seq_len), device=device),
        'attention_mask': torch.ones(batch_size, seq_len, dtype=torch.bool, device=device),
    }


@pytest.fixture
def sample_hidden_minimal(minimal_config, device):
    """Sample hidden states for minimal config."""
    batch_size, seq_len = 2, 64
    return torch.randn(batch_size, seq_len, minimal_config.hidden_size, device=device)


@pytest.fixture
def sample_hidden_1b(config_1b, device):
    """Sample hidden states for 1B config."""
    batch_size, seq_len = 2, 128
    return torch.randn(batch_size, seq_len, config_1b.hidden_size, device=device)


# =============================================================================
# Utility Functions for Tests
# =============================================================================

def check_no_nan_inf(tensor: torch.Tensor, name: str = "tensor"):
    """Check tensor has no NaN or Inf values."""
    assert not torch.isnan(tensor).any(), f"NaN detected in {name}"
    assert not torch.isinf(tensor).any(), f"Inf detected in {name}"


def check_gradient_health(model: nn.Module, max_norm: float = 1000.0):
    """Check model gradients are healthy."""
    total_norm = 0.0
    nan_count = 0
    inf_count = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                nan_count += 1
            if torch.isinf(param.grad).any():
                inf_count += 1
            total_norm += param.grad.norm().item() ** 2

    total_norm = total_norm ** 0.5

    return {
        'total_norm': total_norm,
        'nan_count': nan_count,
        'inf_count': inf_count,
        'is_healthy': nan_count == 0 and inf_count == 0 and total_norm < max_norm,
    }


def compute_kv_cache_size(cache, element_size: int = 2):
    """Compute KV cache size in bytes."""
    total_bytes = 0
    for layer_cache in cache:
        if layer_cache is not None:
            kv_compressed, k_pe = layer_cache
            total_bytes += kv_compressed.numel() * element_size
            total_bytes += k_pe.numel() * element_size
    return total_bytes


def compute_mha_cache_size(num_layers: int, seq_len: int, num_heads: int,
                           head_dim: int, batch_size: int = 1, element_size: int = 2):
    """Compute theoretical MHA cache size for comparison."""
    # MHA: 2 (K+V) * layers * batch * seq * heads * head_dim * element_size
    return 2 * num_layers * batch_size * seq_len * num_heads * head_dim * element_size


# =============================================================================
# pytest Markers
# =============================================================================

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "numerical: marks numerical stability tests")
