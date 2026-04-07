"""
NanoSeek: Educational DeepSeek V3.2 Implementation

A nano-scale implementation of DeepSeek V3.2 architecture featuring:
- MLA (Multi-head Latent Attention): ~23x KV cache compression
- MoE (Mixture of Experts): 5x parameter capacity with sparse activation
- MTP (Multi-Token Prediction): 1.4x inference speedup

Reference: DeepSeek-V3 Technical Report (2024)
"""

__version__ = "0.1.0"

from .config import (
    NanoSeekConfig,
    get_config,
    get_optimizer_lrs,
    compute_mup_lr,
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
    "get_config",
    "get_optimizer_lrs",
    "compute_mup_lr",
    "NanoSeekModel",
    "MultiHeadLatentAttention",
    "MoE",
    "MultiTokenPrediction",
    "create_nanoseek",
]
