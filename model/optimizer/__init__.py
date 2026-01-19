"""
NanoSeek Optimizers - Custom optimizer implementations matching nanochat.

This module provides optimizers optimized for transformer training:
- Muon: SGD with momentum + Newton-Schulz orthogonalization for 2D params
- DistMuon: Distributed version with ZeRO-2 style gradient sharding
- DistAdamW: Distributed AdamW with ZeRO-2 style optimizer state sharding

Usage:
    from model.optimizer import Muon, DistMuon, DistAdamW

    # Single GPU
    muon = Muon(matrix_params, lr=0.02, momentum=0.95)
    adamw = torch.optim.AdamW(embed_params, lr=0.2)

    # Distributed (torchrun)
    muon = DistMuon(matrix_params, lr=0.02, momentum=0.95)
    adamw = DistAdamW(adamw_param_groups, lr=0.2)
"""

from .muon import Muon, DistMuon, zeropower_via_newtonschulz5
from .adamw import DistAdamW

__all__ = [
    'Muon',
    'DistMuon',
    'DistAdamW',
    'zeropower_via_newtonschulz5',
]
