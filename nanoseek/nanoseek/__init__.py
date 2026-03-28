"""
NanoSeek: Educational DeepSeek V3.2 Implementation

A nano-scale implementation of DeepSeek V3.2 architecture featuring:
- MLA (Multi-head Latent Attention): ~23x KV cache compression
- MoE (Mixture of Experts): 5x parameter capacity with sparse activation
- MTP (Multi-Token Prediction): 1.4x inference speedup
- DSA (DeepSeek Sparse Attention): O(L²) → O(Lk) complexity reduction
- YaRN RoPE: Extended context length support

Reference: DeepSeek-V3 Technical Report (2024)
"""

__version__ = "0.1.0"

from .config import (
    NanoSeekConfig,
    MLAConfig,
    MoEConfig,
    MTPConfig,
    SparseAttentionConfig,
    YaRNConfig,
    TrainingPhaseConfig,
    get_nanoseek_config,
    get_nanoseek_ablation_config,
    get_nanoseek_anchor_config,
    get_training_phases,
)

from .model import (
    NanoSeekModel,
    MultiHeadLatentAttention,
    MoE,
    MultiTokenPrediction,
    create_nanoseek,
)

__all__ = [
    "NanoSeekConfig",
    "MLAConfig", 
    "MoEConfig",
    "MTPConfig",
    "SparseAttentionConfig",
    "YaRNConfig",
    "TrainingPhaseConfig",
    "get_nanoseek_config",
    "get_nanoseek_ablation_config",
    "get_nanoseek_anchor_config",
    "get_training_phases",
    "NanoSeekModel",
    "MultiHeadLatentAttention",
    "MoE",
    "MultiTokenPrediction",
    "create_nanoseek",
]
