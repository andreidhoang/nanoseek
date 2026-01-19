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

# Add model directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import (
    NanoSeekConfig,
    MLAConfig,
    MoEConfig,
    MTPConfig,
    SparseAttentionConfig,
    get_nanoseek_config,
)
from model.model import (
    NanoSeekModel,
    MultiHeadLatentAttention,
    MoE,
    Gate,
    Expert,
    MLP,
    MultiTokenPrediction,
    MTPModule,
    LightningIndexer,
    DSASparseAttention,
    RMSNorm,
    RotaryEmbedding,
    precompute_freqs_cis,
    apply_rotary_emb,
    create_causal_mask,
    create_mla_from_config,
    create_moe_from_config,
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
def config_700m():
    """NanoSeek-700M configuration - main config."""
    return get_nanoseek_config()


@pytest.fixture
def minimal_config():
    """Minimal configuration for ultra-fast unit tests."""
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
        sparse=SparseAttentionConfig(
            enabled=False,
            topk_tokens=64,
            activation_threshold=32,
        ),
        total_tokens=1_000_000,
        global_batch_size=4,
        sequence_length=64,
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
def model_700m(config_700m, device):
    """NanoSeek-700M model - main model."""
    model = NanoSeekModel(config_700m)
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
        dim=minimal_config.hidden_size,
        moe_inter_dim=minimal_config.moe.moe_intermediate_size,
        n_routed_experts=minimal_config.moe.n_routed_experts,
        n_activated_experts=minimal_config.moe.num_experts_per_tok,
        n_shared_experts=minimal_config.moe.n_shared_experts,
        n_expert_groups=minimal_config.moe.n_group,
        n_limited_groups=minimal_config.moe.topk_group,
        score_func=minimal_config.moe.scoring_func,
        route_scale=minimal_config.moe.routed_scaling_factor,
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


@pytest.fixture
def dsa_minimal(minimal_config, device):
    """Minimal DSA for unit tests."""
    sparse_config = SparseAttentionConfig(
        enabled=True,
        topk_tokens=32,
        activation_threshold=16,
        indexer_num_heads=2,
        indexer_head_dim=32,
    )
    dsa = DSASparseAttention(
        hidden_size=minimal_config.hidden_size,
        num_heads=minimal_config.num_heads,
        q_lora_rank=minimal_config.mla.q_lora_rank,
        kv_lora_rank=minimal_config.mla.kv_lora_rank,
        qk_nope_head_dim=minimal_config.mla.qk_nope_head_dim,
        qk_rope_head_dim=minimal_config.mla.qk_rope_head_dim,
        v_head_dim=minimal_config.mla.v_head_dim,
        max_position_embeddings=minimal_config.max_position_embeddings,
        sparse_config=sparse_config,
    )
    return dsa.to(device)


@pytest.fixture
def indexer_minimal(minimal_config, device):
    """Minimal Lightning Indexer for unit tests."""
    indexer = LightningIndexer(
        q_lora_rank=minimal_config.mla.q_lora_rank,
        kv_lora_rank=minimal_config.mla.kv_lora_rank,
        num_heads=4,
        head_dim=32,
    )
    return indexer.to(device)


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
def sample_batch_700m(config_700m, device):
    """Sample batch for 700M config."""
    batch_size, seq_len = 2, 128
    return {
        'input_ids': torch.randint(0, config_700m.vocab_size, (batch_size, seq_len), device=device),
        'labels': torch.randint(0, config_700m.vocab_size, (batch_size, seq_len), device=device),
        'attention_mask': torch.ones(batch_size, seq_len, dtype=torch.bool, device=device),
    }


@pytest.fixture
def sample_hidden_minimal(minimal_config, device):
    """Sample hidden states for minimal config."""
    batch_size, seq_len = 2, 64
    return torch.randn(batch_size, seq_len, minimal_config.hidden_size, device=device)


@pytest.fixture
def sample_hidden_700m(config_700m, device):
    """Sample hidden states for 700M config."""
    batch_size, seq_len = 2, 128
    return torch.randn(batch_size, seq_len, config_700m.hidden_size, device=device)


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
