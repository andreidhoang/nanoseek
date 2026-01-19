"""
Test Utility Functions

Shared utility functions for NanoSeek test suite.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional


def check_no_nan_inf(tensor: torch.Tensor, name: str = "tensor"):
    """Check tensor has no NaN or Inf values."""
    assert not torch.isnan(tensor).any(), f"NaN detected in {name}"
    assert not torch.isinf(tensor).any(), f"Inf detected in {name}"


def check_gradient_health(model: nn.Module, max_norm: float = 10000.0) -> Dict[str, Any]:
    """Check model gradients are healthy."""
    total_norm = 0.0
    nan_count = 0
    inf_count = 0
    max_grad = 0.0

    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                nan_count += 1
            if torch.isinf(param.grad).any():
                inf_count += 1
            grad_norm = param.grad.norm().item()
            total_norm += grad_norm ** 2
            max_grad = max(max_grad, param.grad.abs().max().item())

    total_norm = total_norm ** 0.5

    return {
        'total_norm': total_norm,
        'max_grad': max_grad,
        'nan_count': nan_count,
        'inf_count': inf_count,
        'is_healthy': nan_count == 0 and inf_count == 0 and total_norm < max_norm,
    }


def compute_kv_cache_size(cache: List, element_size: int = 2) -> int:
    """Compute KV cache size in bytes."""
    total_bytes = 0
    for layer_cache in cache:
        if layer_cache is not None:
            kv_compressed, k_pe = layer_cache
            total_bytes += kv_compressed.numel() * element_size
            total_bytes += k_pe.numel() * element_size
    return total_bytes


def compute_mha_cache_size(
    num_layers: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    batch_size: int = 1,
    element_size: int = 2,
) -> int:
    """Compute theoretical MHA cache size for comparison."""
    # MHA: 2 (K+V) * layers * batch * seq * heads * head_dim * element_size
    return 2 * num_layers * batch_size * seq_len * num_heads * head_dim * element_size
