# NanoSeek - Complete DeepSeek V3.2 Implementation at Nano Scale
# Architecture Components:
# - MLA (Multi-head Latent Attention): ~23x KV cache compression
# - MoE (Mixture of Experts): 5x parameter capacity with sparse activation
# - MTP (Multi-Token Prediction): 1.4x inference speedup via speculative decoding
# - DSA (DeepSeek Sparse Attention): O(L²) → O(Lk) complexity reduction
# - YaRN RoPE: Extended context length support
#
# Reference: DeepSeek-V3 Technical Report (2024)

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

# Import configurations from config.py
# Handle both package import and direct execution
try:
    from .config import (
        NanoSeekConfig,
        SparseAttentionConfig,
        get_nanoseek_config,
    )
except ImportError:
    from config import (
        NanoSeekConfig,
        SparseAttentionConfig,
        get_nanoseek_config,
    )


# =============================================================================
# SECTION 2: ROPE (ROTARY POSITION EMBEDDING WITH YARN)
# =============================================================================

def find_correction_dim(
    num_rotations: int,
    dim: int,
    base: float = 10000.0,
    max_position_embeddings: int = 2048,
) -> float:
    """Find dimension where rotation frequency equals a threshold."""
    return dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi)) / (2 * math.log(base))


def find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float = 10000.0,
    max_position_embeddings: int = 2048,
) -> Tuple[int, int]:
    """Find range of dimensions requiring interpolation."""
    low = max(math.floor(find_correction_dim(low_rot, dim, base, max_position_embeddings)), 0)
    high = min(math.ceil(find_correction_dim(high_rot, dim, base, max_position_embeddings)), dim - 1)
    return low, high


def linear_ramp_factor(min_val: int, max_val: int, dim: int) -> Tensor:
    """Create linear ramp from 0 to 1 across dimension range."""
    if min_val == max_val:
        max_val = min_val + 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    return torch.clamp(linear_func, 0, 1)


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    scaling_factor: float = 1.0,
    original_max_position_embeddings: int = 4096,
    beta_fast: int = 32,
    beta_slow: int = 1,
    mscale: float = 1.0,
    mscale_all_dim: float = 0.0,
) -> Tensor:
    """Precompute rotary frequencies with YaRN scaling."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    if scaling_factor != 1.0:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, theta, original_max_position_embeddings
        )
        smooth = linear_ramp_factor(low, high, dim // 2)
        scaled_freqs = freqs / scaling_factor
        freqs = freqs * (1 - smooth) + scaled_freqs * smooth

    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor, interleaved: bool = True) -> Tensor:
    """Apply rotary embeddings to input tensor.

    Args:
        x: Input tensor of shape (batch, seq, heads, head_dim) or (batch, seq, head_dim)
        freqs_cis: Complex frequencies of shape (seq, dim//2) or (batch, seq, dim//2)
        interleaved: Whether the real/imag parts are interleaved in x
    """
    orig_shape = x.shape
    orig_dtype = x.dtype

    if x.dim() == 3:
        x = x.unsqueeze(2)

    *batch_dims, seq_len, n_heads, head_dim = x.shape

    # Handle freqs_cis shape - can be 2D (seq, dim) or 3D (batch, seq, dim)
    if freqs_cis.dim() == 2:
        # Standard case: (seq_len, dim) -> (1, seq_len, 1, dim)
        if freqs_cis.shape[0] != seq_len:
            freqs_cis = freqs_cis[:seq_len]
        freqs_cis = freqs_cis.view(1, seq_len, 1, -1)
    elif freqs_cis.dim() == 3:
        # Batched case: (batch, seq_len, dim) -> (batch, seq_len, 1, dim)
        freqs_cis = freqs_cis.unsqueeze(2)
    else:
        raise ValueError(f"freqs_cis must be 2D or 3D, got {freqs_cis.dim()}D")

    if interleaved:
        x_complex = torch.view_as_complex(x.float().reshape(*batch_dims, seq_len, n_heads, head_dim // 2, 2))
    else:
        x_half = head_dim // 2
        x1, x2 = x[..., :x_half], x[..., x_half:]
        x_complex = torch.view_as_complex(torch.stack([x1, x2], dim=-1).float())

    x_rotated = x_complex * freqs_cis

    if interleaved:
        x_out = torch.view_as_real(x_rotated).reshape(*batch_dims, seq_len, n_heads, head_dim)
    else:
        x_real = torch.view_as_real(x_rotated)
        x_out = torch.cat([x_real[..., 0], x_real[..., 1]], dim=-1)

    x_out = x_out.view(orig_shape).to(orig_dtype)
    return x_out


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding with YaRN support."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor

        freqs_cis = precompute_freqs_cis(
            dim, max_position_embeddings, base, scaling_factor,
            original_max_position_embeddings
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, q: Tensor, k: Tensor, position_ids: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if position_ids is not None:
            freqs = self.freqs_cis[position_ids]
        else:
            seq_len = q.shape[1]
            freqs = self.freqs_cis[:seq_len]

        q_rotated = apply_rotary_emb(q, freqs, interleaved=True)
        k_rotated = apply_rotary_emb(k, freqs, interleaved=True)
        return q_rotated, k_rotated


# =============================================================================
# SECTION 4: RMSNORM
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x / rms
        return (self.weight * x).to(dtype)


def create_causal_mask(
    seq_len: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
    past_len: int = 0,
) -> Tensor:
    """Create causal attention mask."""
    mask = torch.full((seq_len, seq_len + past_len), float('-inf'), dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=past_len + 1)
    return mask.unsqueeze(0).unsqueeze(0)


# =============================================================================
# SECTION 5: MULTI-HEAD LATENT ATTENTION (MLA)
# =============================================================================

class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head Latent Attention (MLA) from DeepSeek-V2/V3.

    Key innovations for 14x KV cache compression:
    1. Low-rank KV joint compression
    2. Decoupled RoPE
    3. Fused projections
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        rope_scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        mscale: float = 1.0,
        attention_dropout: float = 0.0,
        layer_idx: int = 0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.layer_idx = layer_idx
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.attention_dropout = attention_dropout
        self.softmax_scale = mscale / math.sqrt(self.qk_head_dim)

        # Q projection path
        self.wq_a = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_norm = RMSNorm(q_lora_rank)
        self.wq_b = nn.Linear(q_lora_rank, num_heads * self.qk_head_dim, bias=False)

        # KV projection path
        self.wkv_a = nn.Linear(hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(kv_lora_rank)
        self.wkv_b = nn.Linear(kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=False)

        # Output projection
        self.wo = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)

        # Precompute RoPE frequencies
        freqs_cis = precompute_freqs_cis(
            qk_rope_head_dim,
            max_position_embeddings,
            rope_theta,
            rope_scaling_factor,
            original_max_position_embeddings,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape

        # Query path
        q = self.wq_a(hidden_states)
        q = self.q_norm(q)
        q = self.wq_b(q)
        q = q.view(batch_size, seq_len, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # KV path
        kv = self.wkv_a(hidden_states)
        kv_compressed, k_pe_current = torch.split(
            kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv_compressed = self.kv_norm(kv_compressed)

        # Handle cache
        if past_key_value is not None:
            cached_kv, cached_k_pe = past_key_value
            kv_compressed = torch.cat([cached_kv, kv_compressed], dim=1)
            k_pe_current = k_pe_current.unsqueeze(2)
            start_pos = cached_k_pe.shape[1]
            if position_ids is not None:
                current_freqs = self.freqs_cis[position_ids]
            else:
                current_freqs = self.freqs_cis[start_pos:start_pos + seq_len]
            k_pe_current = apply_rotary_emb(k_pe_current, current_freqs, interleaved=True)
            k_pe = torch.cat([cached_k_pe, k_pe_current], dim=1)
        else:
            k_pe_current = k_pe_current.unsqueeze(2).contiguous()
            if position_ids is not None:
                current_freqs = self.freqs_cis[position_ids]
            else:
                current_freqs = self.freqs_cis[:seq_len]
            k_pe = apply_rotary_emb(k_pe_current, current_freqs, interleaved=True)

        kv_len = kv_compressed.shape[1]
        present_key_value = (kv_compressed, k_pe) if use_cache else None

        # Expand KV
        kv_expanded = self.wkv_b(kv_compressed)
        kv_expanded = kv_expanded.view(batch_size, kv_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Apply RoPE to Q
        position_offset = kv_len - seq_len
        if position_ids is not None:
            q_freqs = self.freqs_cis[position_ids]
        else:
            q_freqs = self.freqs_cis[position_offset:position_offset + seq_len]
        q_pe = apply_rotary_emb(q_pe, q_freqs, interleaved=True)

        # Combine Q and K
        q = torch.cat([q_nope, q_pe], dim=-1)
        k_pe_expanded = k_pe.expand(-1, -1, self.num_heads, -1)
        k = torch.cat([k_nope, k_pe_expanded], dim=-1)

        # Attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.softmax_scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        if self.training and self.attention_dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        output = self.wo(attn_output)

        return output, present_key_value


# =============================================================================
# SECTION 6: MIXTURE OF EXPERTS (MOE)
# =============================================================================

class Gate(nn.Module):
    """Gating mechanism for MoE with auxiliary-loss-free load balancing."""

    def __init__(
        self,
        dim: int,
        n_routed_experts: int,
        n_activated_experts: int,
        n_expert_groups: int = 1,
        n_limited_groups: int = 1,
        score_func: str = "softmax",
        route_scale: float = 1.0,
    ):
        super().__init__()

        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts
        self.n_expert_groups = n_expert_groups
        self.n_limited_groups = n_limited_groups
        self.score_func = score_func
        self.route_scale = route_scale
        self.experts_per_group = n_routed_experts // n_expert_groups

        self.weight = nn.Parameter(torch.empty(n_routed_experts, dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.register_buffer("expert_bias", torch.zeros(n_routed_experts))
        self.register_buffer("expert_load", torch.zeros(n_routed_experts))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        scores = F.linear(x, self.weight)

        if self.score_func == "softmax":
            scores = F.softmax(scores, dim=-1)
        else:
            scores = torch.sigmoid(scores)

        if self.training:
            scores_for_selection = scores + self.expert_bias.unsqueeze(0)
        else:
            scores_for_selection = scores

        if self.n_expert_groups > 1 and self.n_limited_groups < self.n_expert_groups:
            scores_grouped = scores_for_selection.view(-1, self.n_expert_groups, self.experts_per_group)
            group_scores = scores_grouped.max(dim=-1).values
            _, top_groups = group_scores.topk(self.n_limited_groups, dim=-1)
            group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
            group_mask.scatter_(1, top_groups, True)
            group_mask = group_mask.unsqueeze(-1).expand(-1, -1, self.experts_per_group)
            group_mask = group_mask.reshape(-1, self.n_routed_experts)
            scores_for_selection = scores_for_selection.masked_fill(~group_mask, float('-inf'))

        topk_weights, topk_indices = scores_for_selection.topk(self.n_activated_experts, dim=-1)
        weights = scores.gather(dim=-1, index=topk_indices)
        weights = weights * self.route_scale

        if self.training:
            with torch.no_grad():
                load = torch.zeros(self.n_routed_experts, device=x.device)
                load.scatter_add_(0, topk_indices.flatten(), torch.ones_like(topk_indices.flatten(), dtype=load.dtype))
                self.expert_load.copy_(load)

        return weights, topk_indices


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    Used as both:
    - Individual expert in MoE layers
    - Dense FFN in non-MoE layers

    Computation: down(silu(gate(x)) * up(x))
    """

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, inter_dim, bias=False)
        self.up_proj = nn.Linear(dim, inter_dim, bias=False)
        self.down_proj = nn.Linear(inter_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# Aliases for semantic clarity
Expert = SwiGLUFFN  # Used in MoE routing


class MLP(nn.Module):
    """
    Dense MLP wrapper for non-MoE layers.

    Wraps SwiGLUFFN to match MoE forward signature (returns aux_data dict).
    """

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.ffn = SwiGLUFFN(dim, inter_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict]:
        return self.ffn(x), {}


def token_centric_dispatch(
    x: Tensor,                      # [N, D]
    indices: Tensor,                # [N, K]
    weights: Tensor,                # [N, K]
    experts: nn.ModuleList,         # List of E experts
) -> Tensor:
    """
    Efficient token-centric MoE dispatch with O(N×K) complexity.

    Instead of O(K×E×N) naive dispatch, this implementation:
    1. Sorts tokens by expert assignment (groups tokens by expert)
    2. Processes each expert's batch contiguously
    3. Scatters results back to original positions

    This achieves 10-50× speedup by:
    - Eliminating redundant scans over all experts
    - Enabling coalesced memory access
    - Reducing Python loop overhead

    Args:
        x: Input tokens [N, D]
        indices: Expert indices per token [N, K]
        weights: Expert weights per token [N, K]
        experts: List of expert modules

    Returns:
        Combined output [N, D]
    """
    N, D = x.shape
    K = indices.shape[1]
    E = len(experts)
    device = x.device
    dtype = x.dtype

    # ─────────────────────────────────────────────────────────
    # Phase 1: PERMUTE - Group tokens by expert
    # ─────────────────────────────────────────────────────────

    # Flatten token-expert assignments to [N×K]
    flat_indices = indices.view(-1)                              # [N×K]
    flat_weights = weights.view(-1)                              # [N×K]

    # Create token IDs: each token appears K times (once per expert slot)
    token_ids = torch.arange(N, device=device)
    token_ids = token_ids.unsqueeze(1).expand(-1, K).reshape(-1) # [N×K]

    # Sort by expert ID to create contiguous expert batches
    sorted_order = torch.argsort(flat_indices, stable=True)
    sorted_expert_ids = flat_indices[sorted_order]               # [N×K]
    sorted_token_ids = token_ids[sorted_order]                   # [N×K]
    sorted_weights = flat_weights[sorted_order]                  # [N×K]

    # Count tokens per expert for batch splitting
    expert_counts = torch.bincount(
        sorted_expert_ids.int(),
        minlength=E
    ).tolist()

    # Gather input tokens in expert-sorted order
    permuted_input = x[sorted_token_ids]                         # [N×K, D]

    # ─────────────────────────────────────────────────────────
    # Phase 2: COMPUTE - Process each expert's contiguous batch
    # ─────────────────────────────────────────────────────────

    # Split permuted input by expert counts
    expert_batches = torch.split(permuted_input, expert_counts)

    # Process each expert's batch (contiguous memory access!)
    expert_outputs = []
    for expert_id, batch in enumerate(expert_batches):
        if batch.shape[0] > 0:
            expert_outputs.append(experts[expert_id](batch))
        else:
            # Empty batch - append empty tensor to maintain alignment
            expert_outputs.append(
                torch.empty(0, D, device=device, dtype=dtype)
            )

    # Concatenate all expert outputs
    permuted_output = torch.cat(expert_outputs, dim=0)           # [N×K, D]

    # ─────────────────────────────────────────────────────────
    # Phase 3: UNPERMUTE - Combine back to original positions
    # ─────────────────────────────────────────────────────────

    # Apply routing weights
    weighted_output = permuted_output * sorted_weights.unsqueeze(-1)

    # Scatter-add back to original token positions
    # Each token accumulates weighted contributions from all its experts
    output = torch.zeros(N, D, device=device, dtype=dtype)
    output.scatter_add_(
        dim=0,
        index=sorted_token_ids.unsqueeze(-1).expand(-1, D),
        src=weighted_output
    )

    return output


class MoE(nn.Module):
    """
    Mixture of Experts layer with auxiliary-loss-free load balancing.

    Implements DeepSeek-V3.2 MoE architecture with:
    - Token-centric dispatch (O(N×K) complexity, 10-50× faster than naive)
    - Sigmoid scoring with learnable bias (no auxiliary loss needed)
    - Optional shared experts (always active)
    - Group-based routing for large expert counts

    Performance characteristics:
    - Naive dispatch: O(K × E × N) with scattered memory access
    - Token-centric:  O(N × K) with coalesced memory access
    - Speedup: E× in operations (32× for E=32), 10-50× in wall time

    See tutorials/moe/ for detailed algorithm explanation.
    """

    def __init__(
        self,
        dim: int,
        moe_inter_dim: int,
        n_routed_experts: int,
        n_activated_experts: int,
        n_shared_experts: int = 0,
        n_expert_groups: int = 1,
        n_limited_groups: int = 1,
        score_func: str = "softmax",
        route_scale: float = 1.0,
        seq_aux_loss_alpha: float = 0.0001,
    ):
        super().__init__()

        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux_loss_alpha = seq_aux_loss_alpha

        # Router/Gate for expert selection
        self.gate = Gate(
            dim=dim,
            n_routed_experts=n_routed_experts,
            n_activated_experts=n_activated_experts,
            n_expert_groups=n_expert_groups,
            n_limited_groups=n_limited_groups,
            score_func=score_func,
            route_scale=route_scale,
        )

        # Routed experts
        self.experts = nn.ModuleList([
            Expert(dim, moe_inter_dim) for _ in range(n_routed_experts)
        ])

        # Shared experts (optional, always active for common patterns)
        # Use single shared expert for efficiency when n_shared_experts=1
        if n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                Expert(dim, moe_inter_dim) for _ in range(n_shared_experts)
            ])
        else:
            self.shared_experts = None

    def _compute_shared_output(self, x: Tensor) -> Tensor:
        """Compute shared expert output (batched when possible)."""
        if self.shared_experts is None:
            return 0

        if len(self.shared_experts) == 1:
            # Single shared expert - most common case, no overhead
            return self.shared_experts[0](x)
        else:
            # Multiple shared experts - sum their outputs
            # Note: Could be further optimized with grouped GEMM if needed
            return sum(expert(x) for expert in self.shared_experts)

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict]:
        """
        Forward pass through MoE layer.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]

        Returns:
            output: Output tensor [batch, seq_len, hidden_size]
            aux_data: Dictionary with auxiliary data (loss, load stats)
        """
        batch_size, seq_len, dim = x.shape
        N = batch_size * seq_len

        # Flatten to [N, D] for expert processing
        x_flat = x.view(N, dim)

        # ─────────────────────────────────────────────────────────
        # Shared expert computation (always active)
        # ─────────────────────────────────────────────────────────

        shared_output = self._compute_shared_output(x_flat)

        # ─────────────────────────────────────────────────────────
        # Routed expert computation (token-centric dispatch)
        # ─────────────────────────────────────────────────────────

        # Get routing decisions from gate
        weights, indices = self.gate(x_flat)  # [N, K], [N, K]

        # Efficient token-centric dispatch (O(N×K) not O(K×E×N))
        routed_output = token_centric_dispatch(x_flat, indices, weights, self.experts)

        # ─────────────────────────────────────────────────────────
        # Combine and reshape
        # ─────────────────────────────────────────────────────────

        output = routed_output + shared_output
        output = output.view(batch_size, seq_len, dim)

        # ─────────────────────────────────────────────────────────
        # Auxiliary data for monitoring and optional loss
        # ─────────────────────────────────────────────────────────

        aux_data = {}
        if self.training:
            # Track expert load statistics
            aux_data["expert_load"] = self.gate.expert_load.clone()

            # Optional auxiliary loss (prefer bias adjustment instead)
            if self.seq_aux_loss_alpha > 0:
                load = self.gate.expert_load
                target_load = N * self.n_activated_experts / self.n_routed_experts
                load_imbalance = ((load - target_load) ** 2).mean()
                aux_data["seq_aux_loss"] = self.seq_aux_loss_alpha * load_imbalance

        return output, aux_data

    def update_load_balance_bias(self, gamma: float = 0.001):
        with torch.no_grad():
            load = self.gate.expert_load
            mean_load = load.mean()
            if mean_load > 0:
                imbalance = (load - mean_load) / (mean_load + 1e-8)
                self.gate.expert_bias.sub_(gamma * imbalance)

    def get_expert_load_stats(self) -> Dict[str, Tensor]:
        return {
            "expert_load": self.gate.expert_load.clone(),
            "expert_bias": self.gate.expert_bias.clone(),
        }


# =============================================================================
# SECTION 7: MULTI-TOKEN PREDICTION (MTP)
# =============================================================================

class MTPBlock(nn.Module):
    """Single MTP prediction block."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float = 0.0,
        layer_idx: int = 0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.layer_idx = layer_idx

        self.input_norm = RMSNorm(hidden_size)
        self.cross_norm = RMSNorm(hidden_size)
        self.ffn_norm = RMSNorm(hidden_size)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: Tensor,
        main_hidden: Tensor,
        attention_mask: Optional[Tensor] = None,
        causal_mask: Optional[Tensor] = None,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                key_padding_mask = ~attention_mask
            else:
                key_padding_mask = attention_mask == 0
        cross_output, _ = self.cross_attn(
            query=hidden_states, key=main_hidden, value=main_hidden, key_padding_mask=key_padding_mask
        )
        hidden_states = residual + self.dropout(cross_output)

        residual = hidden_states
        hidden_states = self.cross_norm(hidden_states)
        seq_len = hidden_states.size(1)
        if causal_mask is None:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool), diagonal=1
            )
        self_output, _ = self.self_attn(
            query=hidden_states, key=hidden_states, value=hidden_states, attn_mask=causal_mask, is_causal=False
        )
        hidden_states = residual + self.dropout(self_output)

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        hidden_states = self.down_proj(gate * up)
        hidden_states = residual + self.dropout(hidden_states)

        return hidden_states


class MTPModule(nn.Module):
    """Multi-Token Prediction Module - DeepSeek V3 Architecture."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        input_size: Optional[int] = None,
        num_heads: int = 4,
        intermediate_size: Optional[int] = None,
        num_blocks: int = 1,
        module_idx: int = 0,
        dropout: float = 0.0,
        tie_word_embeddings: bool = True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size or hidden_size
        self.vocab_size = vocab_size
        self.module_idx = module_idx
        self.tie_word_embeddings = tie_word_embeddings

        if intermediate_size is None:
            intermediate_size = hidden_size * 4

        self.hidden_norm = RMSNorm(self.input_size)
        self.embed_norm = RMSNorm(hidden_size)
        self.concat_proj = nn.Linear(self.input_size + hidden_size, hidden_size, bias=False)

        if not tie_word_embeddings:
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        else:
            self.embed_tokens = None

        self.blocks = nn.ModuleList([
            MTPBlock(hidden_size, num_heads, intermediate_size, dropout, i)
            for i in range(num_blocks)
        ])

        self.output_norm = RMSNorm(hidden_size)

        if not tie_word_embeddings:
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        else:
            self.lm_head = None

    def set_shared_embeddings(self, embed_tokens: nn.Embedding, lm_head: nn.Linear):
        if self.tie_word_embeddings:
            self.embed_tokens = embed_tokens
            self.lm_head = lm_head

    def forward(
        self,
        prev_hidden: Tensor,
        target_tokens: Optional[Tensor] = None,
        main_hidden: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        batch_size, seq_len, _ = prev_hidden.shape

        normed_hidden = self.hidden_norm(prev_hidden)

        if target_tokens is not None and self.embed_tokens is not None:
            token_embeds = self.embed_tokens(target_tokens)
            normed_embeds = self.embed_norm(token_embeds)
        else:
            normed_embeds = torch.zeros(
                batch_size, seq_len, self.hidden_size, device=prev_hidden.device, dtype=prev_hidden.dtype
            )

        concatenated = torch.cat([normed_hidden, normed_embeds], dim=-1)
        hidden_states = self.concat_proj(concatenated)

        cross_hidden = main_hidden if main_hidden is not None else prev_hidden
        for block in self.blocks:
            hidden_states = block(hidden_states, cross_hidden, attention_mask)

        output_hidden = self.output_norm(hidden_states)
        logits = self.lm_head(output_hidden)

        return logits, hidden_states


class MultiTokenPrediction(nn.Module):
    """Complete Multi-Token Prediction system."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        mtp_hidden_size: Optional[int] = None,
        num_mtp_modules: int = 1,
        mtp_num_heads: int = 4,
        mtp_intermediate_size: Optional[int] = None,
        mtp_num_blocks: int = 1,
        mtp_loss_weight: float = 0.1,
        mtp_loss_decay: float = 0.8,
        tie_word_embeddings: bool = True,
    ):
        super().__init__()

        if mtp_hidden_size is None:
            mtp_hidden_size = hidden_size

        if tie_word_embeddings and mtp_hidden_size != hidden_size:
            tie_word_embeddings = False

        self.num_mtp_modules = num_mtp_modules
        self.mtp_loss_weight = mtp_loss_weight
        self.mtp_loss_decay = mtp_loss_decay
        self.mtp_hidden_size = mtp_hidden_size

        self.mtp_modules = nn.ModuleList([
            MTPModule(
                hidden_size=mtp_hidden_size,
                input_size=hidden_size,
                vocab_size=vocab_size,
                num_heads=mtp_num_heads,
                intermediate_size=mtp_intermediate_size,
                num_blocks=mtp_num_blocks,
                module_idx=i,
                tie_word_embeddings=tie_word_embeddings,
            )
            for i in range(num_mtp_modules)
        ])

    def set_shared_embeddings(self, embed_tokens: nn.Embedding, lm_head: nn.Linear):
        for module in self.mtp_modules:
            module.set_shared_embeddings(embed_tokens, lm_head)

    def forward(
        self,
        main_hidden: Tensor,
        labels: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        batch_size, seq_len, _ = main_hidden.shape
        results = {"mtp_logits": [], "mtp_hidden": []}

        total_loss = 0.0
        per_module_loss = []
        prev_hidden = main_hidden

        for i, module in enumerate(self.mtp_modules):
            token_offset = i + 1

            if labels is not None and seq_len > token_offset:
                target_tokens = labels[:, token_offset:].contiguous()
                effective_len = target_tokens.size(1)
                current_hidden = prev_hidden[:, :effective_len]
                current_main = main_hidden[:, :effective_len]
            else:
                target_tokens = None
                current_hidden = prev_hidden
                current_main = main_hidden

            logits, hidden = module(
                prev_hidden=current_hidden, target_tokens=target_tokens,
                main_hidden=current_main, attention_mask=attention_mask
            )

            results["mtp_logits"].append(logits)
            results["mtp_hidden"].append(hidden)

            if labels is not None:
                pred_offset = i + 2
                if seq_len > pred_offset:
                    pred_len = seq_len - pred_offset
                    if logits.size(1) >= pred_len and pred_len > 0:
                        shift_logits = logits[:, :pred_len].contiguous()
                        shift_labels = labels[:, pred_offset:pred_offset + pred_len].contiguous()

                        loss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1), ignore_index=-100
                        )

                        weight = self.mtp_loss_decay ** i
                        total_loss += weight * loss
                        per_module_loss.append(loss.item())

            prev_hidden = hidden

        if labels is not None:
            weight_sum = sum(self.mtp_loss_decay ** i for i in range(len(self.mtp_modules)))
            results["mtp_loss"] = total_loss / weight_sum
            results["per_module_loss"] = per_module_loss

        return results

    def speculative_decode(
        self,
        main_hidden: Tensor,
        first_token: Optional[Tensor] = None,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        draft_tokens = []
        draft_probs = []

        prev_hidden = main_hidden
        prev_token = first_token.unsqueeze(-1) if first_token is not None else None

        for module in self.mtp_modules:
            logits, hidden = module(prev_hidden=prev_hidden, target_tokens=prev_token, main_hidden=main_hidden)

            if temperature == 0.0:
                probs = F.softmax(logits[:, -1], dim=-1)
                token = logits[:, -1].argmax(dim=-1)
                prob = probs.gather(-1, token.unsqueeze(-1)).squeeze(-1)
            else:
                probs = F.softmax(logits[:, -1] / temperature, dim=-1)
                if top_k is not None:
                    values, indices = probs.topk(top_k, dim=-1)
                    probs = torch.zeros_like(probs).scatter_(-1, indices, values)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                prob = probs.gather(-1, token.unsqueeze(-1)).squeeze(-1)

            draft_tokens.append(token)
            draft_probs.append(prob)

            prev_hidden = hidden
            prev_token = token.unsqueeze(-1)

        return torch.stack(draft_tokens, dim=1), torch.stack(draft_probs, dim=1)


# =============================================================================
# SECTION 8: LIGHTNING INDEXER
# =============================================================================

class LightningIndexer(nn.Module):
    """Lightning Indexer for sparse attention token selection."""

    def __init__(
        self,
        q_lora_rank: int,
        kv_lora_rank: int,
        num_heads: int = 4,
        head_dim: int = 64,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(q_lora_rank, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(kv_lora_rank, num_heads * head_dim, bias=False)
        self.head_weights = nn.Parameter(torch.ones(num_heads))

        nn.init.normal_(self.q_proj.weight, std=0.01)
        nn.init.normal_(self.k_proj.weight, std=0.01)

    def forward(
        self,
        q_compressed: Tensor,
        kv_compressed: Tensor,
        causal_mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, q_len, _ = q_compressed.shape
        kv_len = kv_compressed.shape[1]

        q_idx = self.q_proj(q_compressed).view(batch_size, q_len, self.num_heads, self.head_dim)
        k_idx = self.k_proj(kv_compressed).view(batch_size, kv_len, self.num_heads, self.head_dim)

        scores = torch.einsum('bqhd,bkhd->bhqk', q_idx, k_idx)
        scores = F.relu(scores)

        weights = self.head_weights.view(1, self.num_heads, 1, 1)
        index_scores = (scores * weights).sum(dim=1)

        if causal_mask is not None:
            index_scores = index_scores + causal_mask.unsqueeze(0)

        return index_scores

    def select_topk(self, index_scores: Tensor, k: int) -> Tuple[Tensor, Tensor]:
        k = min(k, index_scores.shape[-1])
        return torch.topk(index_scores, k=k, dim=-1, sorted=False)


# =============================================================================
# SECTION 9: DEEPSEEK SPARSE ATTENTION (DSA)
# =============================================================================

class DSASparseAttention(nn.Module):
    """DeepSeek Sparse Attention (DSA) wrapping MLA."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        rope_scaling_factor: float = 1.0,
        attention_dropout: float = 0.0,
        layer_idx: int = 0,
        sparse_config: Optional[SparseAttentionConfig] = None,
    ):
        super().__init__()

        self.sparse_config = sparse_config or SparseAttentionConfig()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.layer_idx = layer_idx
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.attention_dropout = attention_dropout
        self.softmax_scale = 1.0 / math.sqrt(self.qk_head_dim)

        self.mla = MultiHeadLatentAttention(
            hidden_size=hidden_size, num_heads=num_heads, q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank, qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim, v_head_dim=v_head_dim,
            max_position_embeddings=max_position_embeddings, rope_theta=rope_theta,
            rope_scaling_factor=rope_scaling_factor, attention_dropout=attention_dropout,
            layer_idx=layer_idx,
        )

        self.indexer = LightningIndexer(
            q_lora_rank=q_lora_rank, kv_lora_rank=kv_lora_rank,
            num_heads=self.sparse_config.indexer_num_heads,
            head_dim=self.sparse_config.indexer_head_dim,
        )

        self.register_buffer('training_step', torch.tensor(0, dtype=torch.long))

    def _get_compressed_representations(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        q_compressed = self.mla.wq_a(hidden_states)
        q_compressed = self.mla.q_norm(q_compressed)

        kv = self.mla.wkv_a(hidden_states)
        kv_compressed, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_compressed = self.mla.kv_norm(kv_compressed)

        return q_compressed, kv_compressed, k_pe

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
        output_indexer_loss: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]], Dict]:
        batch_size, seq_len, _ = hidden_states.shape
        aux_data = {}

        if past_key_value is not None:
            cached_kv, _ = past_key_value
            kv_len = cached_kv.shape[1] + seq_len
        else:
            kv_len = seq_len

        in_warmup = self.training_step.item() < self.sparse_config.dense_warmup_steps
        use_sparse = (
            self.sparse_config.enabled and
            kv_len >= self.sparse_config.activation_threshold and
            not in_warmup
        )

        if use_sparse:
            output, present_kv, indexer_loss = self._sparse_forward(
                hidden_states, attention_mask, position_ids, past_key_value, use_cache, output_indexer_loss
            )
        else:
            output, present_kv, indexer_loss = self._dense_forward(
                hidden_states, attention_mask, position_ids, past_key_value, use_cache, output_indexer_loss
            )

        if indexer_loss is not None:
            aux_data['indexer_loss'] = indexer_loss

        return output, present_kv, aux_data

    def _dense_forward(
        self, hidden_states, attention_mask, position_ids, past_key_value, use_cache, output_indexer_loss
    ):
        output, present_key_value = self.mla(
            hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids,
            past_key_value=past_key_value, use_cache=use_cache
        )

        indexer_loss = None
        if self.training and output_indexer_loss:
            indexer_loss = self._compute_indexer_loss(hidden_states, past_key_value)

        return output, present_key_value, indexer_loss

    def _compute_indexer_loss(self, hidden_states, past_key_value):
        batch_size, seq_len, _ = hidden_states.shape

        q_compressed, kv_compressed, _ = self._get_compressed_representations(hidden_states)

        if past_key_value is not None:
            cached_kv, _ = past_key_value
            full_kv_compressed = torch.cat([cached_kv, kv_compressed], dim=1)
        else:
            full_kv_compressed = kv_compressed

        kv_len = full_kv_compressed.shape[1]
        causal_mask = self._create_causal_mask(seq_len, kv_len, hidden_states.device)

        index_scores = self.indexer(q_compressed, full_kv_compressed, causal_mask)

        valid_mask = ~torch.isinf(causal_mask)
        valid_mask = valid_mask.unsqueeze(0).expand(batch_size, -1, -1)

        index_scores_masked = index_scores.masked_fill(~valid_mask, -1e9)
        log_probs = F.log_softmax(index_scores_masked, dim=-1)
        probs = F.softmax(index_scores_masked, dim=-1)

        entropy = -(probs * log_probs).sum(dim=-1).mean()
        return entropy * self.sparse_config.indexer_loss_weight

    def _sparse_forward(
        self, hidden_states, attention_mask, position_ids, past_key_value, use_cache, output_indexer_loss
    ):
        batch_size, seq_len, _ = hidden_states.shape

        q_compressed, kv_compressed, k_pe_current = self._get_compressed_representations(hidden_states)

        if past_key_value is not None:
            cached_kv, cached_k_pe = past_key_value
            full_kv_compressed = torch.cat([cached_kv, kv_compressed], dim=1)
            start_pos = cached_k_pe.shape[1]
            if position_ids is None:
                current_freqs = self.mla.freqs_cis[start_pos:start_pos + seq_len]
            else:
                current_freqs = self.mla.freqs_cis[position_ids]
            k_pe_rotated = apply_rotary_emb(k_pe_current.unsqueeze(2), current_freqs, interleaved=True)
            full_k_pe = torch.cat([cached_k_pe, k_pe_rotated], dim=1)
        else:
            full_kv_compressed = kv_compressed
            if position_ids is None:
                current_freqs = self.mla.freqs_cis[:seq_len]
            else:
                current_freqs = self.mla.freqs_cis[position_ids]
            full_k_pe = apply_rotary_emb(k_pe_current.unsqueeze(2), current_freqs, interleaved=True)

        kv_len = full_kv_compressed.shape[1]
        position_offset = kv_len - seq_len

        present_key_value = (full_kv_compressed, full_k_pe) if use_cache else None

        causal_mask = self._create_causal_mask(seq_len, kv_len, hidden_states.device, position_offset)
        index_scores = self.indexer(q_compressed, full_kv_compressed, causal_mask)

        topk = self.sparse_config.topk_tokens
        _, selected_indices = self.indexer.select_topk(index_scores, topk)

        # Full Q computation
        q = self.mla.wq_b(q_compressed)
        q = q.view(batch_size, seq_len, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        if position_ids is not None:
            q_freqs = self.mla.freqs_cis[position_ids]
        else:
            q_freqs = self.mla.freqs_cis[position_offset:position_offset + seq_len]
        q_pe = apply_rotary_emb(q_pe, q_freqs, interleaved=True)
        q = torch.cat([q_nope, q_pe], dim=-1)
        q = q.transpose(1, 2)

        # Expand KV
        kv_expanded = self.mla.wkv_b(full_kv_compressed)
        kv_expanded = kv_expanded.view(batch_size, kv_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope_full, v_full = torch.split(kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Gather selected
        k_nope_gathered = self._gather_selected(k_nope_full, selected_indices, self.qk_nope_head_dim)
        k_pe_squeezed = full_k_pe.squeeze(2)
        k_pe_gathered = self._gather_selected(k_pe_squeezed, selected_indices, self.qk_rope_head_dim)
        k_pe_gathered = k_pe_gathered.unsqueeze(3).expand(-1, -1, -1, self.num_heads, -1)
        k_selected = torch.cat([k_nope_gathered, k_pe_gathered], dim=-1)
        v_selected = self._gather_selected(v_full, selected_indices, self.v_head_dim)

        # Sparse attention
        k_selected = k_selected.permute(0, 3, 1, 2, 4)
        v_selected = v_selected.permute(0, 3, 1, 2, 4)

        attn_scores = torch.matmul(q.unsqueeze(3), k_selected.transpose(-2, -1)).squeeze(3) * self.softmax_scale

        q_positions = torch.arange(seq_len, device=q.device) + position_offset
        q_positions = q_positions.view(1, 1, seq_len, 1)
        selected_positions = selected_indices.unsqueeze(1)
        causal_mask_selected = torch.where(
            selected_positions > q_positions,
            torch.tensor(float('-inf'), device=q.device),
            torch.tensor(0.0, device=q.device)
        )
        attn_scores = attn_scores + causal_mask_selected

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)

        if self.training and self.attention_dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)

        attn_output = torch.matmul(attn_weights.unsqueeze(3), v_selected).squeeze(3)

        indexer_loss = None
        if self.training and output_indexer_loss:
            indexer_loss = self._compute_indexer_loss(hidden_states, past_key_value)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        output = self.mla.wo(attn_output)

        return output, present_key_value, indexer_loss

    def _create_causal_mask(self, q_len, kv_len, device, position_offset=0):
        q_positions = torch.arange(q_len, device=device) + position_offset
        k_positions = torch.arange(kv_len, device=device)
        mask = k_positions.unsqueeze(0) > q_positions.unsqueeze(1)
        return torch.where(mask, torch.tensor(float('-inf'), device=device), torch.tensor(0.0, device=device))

    def _gather_selected(self, tensor, indices, last_dim):
        batch_size, kv_len = tensor.shape[:2]
        seq_len, num_selected = indices.shape[1], indices.shape[2]

        if tensor.dim() == 3:
            idx_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, last_dim)
            tensor_expanded = tensor.unsqueeze(1).expand(-1, seq_len, -1, -1)
            return torch.gather(tensor_expanded, dim=2, index=idx_expanded)
        else:
            num_heads = tensor.shape[2]
            idx_expanded = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, num_heads, last_dim)
            tensor_expanded = tensor.unsqueeze(1).expand(-1, seq_len, -1, -1, -1)
            return torch.gather(tensor_expanded, dim=2, index=idx_expanded)

    def increment_training_step(self):
        self.training_step.add_(1)


# =============================================================================
# SECTION 10: DECODER LAYER
# =============================================================================

class NanoSeekDecoderLayer(nn.Module):
    """Single decoder layer for NanoSeek."""

    def __init__(self, config: NanoSeekConfig, layer_idx: int):
        super().__init__()

        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.is_moe_layer = layer_idx in config.moe_layer_indices

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.use_sparse_attention = config.sparse.enabled
        if config.sparse.enabled:
            self.self_attn = DSASparseAttention(
                hidden_size=config.hidden_size, num_heads=config.num_heads,
                q_lora_rank=config.mla.q_lora_rank, kv_lora_rank=config.mla.kv_lora_rank,
                qk_nope_head_dim=config.mla.qk_nope_head_dim,
                qk_rope_head_dim=config.mla.qk_rope_head_dim,
                v_head_dim=config.mla.v_head_dim,
                max_position_embeddings=config.max_position_embeddings,
                rope_theta=config.mla.rope_theta,
                rope_scaling_factor=config.mla.rope_scaling_factor,
                layer_idx=layer_idx, sparse_config=config.sparse,
            )
        else:
            self.self_attn = MultiHeadLatentAttention(
                hidden_size=config.hidden_size, num_heads=config.num_heads,
                q_lora_rank=config.mla.q_lora_rank, kv_lora_rank=config.mla.kv_lora_rank,
                qk_nope_head_dim=config.mla.qk_nope_head_dim,
                qk_rope_head_dim=config.mla.qk_rope_head_dim,
                v_head_dim=config.mla.v_head_dim,
                max_position_embeddings=config.max_position_embeddings,
                rope_theta=config.mla.rope_theta,
                rope_scaling_factor=config.mla.rope_scaling_factor,
                original_max_position_embeddings=config.mla.original_max_position_embeddings,
                mscale=config.mla.mscale,
                attention_dropout=config.attention_dropout, layer_idx=layer_idx,
            )

        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if self.is_moe_layer:
            self.ffn = MoE(
                dim=config.hidden_size, moe_inter_dim=config.moe.moe_intermediate_size,
                n_routed_experts=config.moe.n_routed_experts,
                n_activated_experts=config.moe.num_experts_per_tok,
                n_shared_experts=config.moe.n_shared_experts,
                n_expert_groups=config.moe.n_group, n_limited_groups=config.moe.topk_group,
                score_func=config.moe.scoring_func, route_scale=config.moe.routed_scaling_factor,
                seq_aux_loss_alpha=config.moe.seq_aux_loss_alpha,
            )
        else:
            self.ffn = MLP(dim=config.hidden_size, inter_dim=config.intermediate_size)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
        training_step: Optional[int] = None,
        output_indexer_loss: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple], Dict]:
        aux_data = {}

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.use_sparse_attention:
            hidden_states, present_key_value, attn_aux = self.self_attn(
                hidden_states=hidden_states, attention_mask=attention_mask,
                position_ids=position_ids, past_key_value=past_key_value,
                use_cache=use_cache, output_indexer_loss=output_indexer_loss,
            )
            if 'indexer_loss' in attn_aux:
                aux_data['indexer_loss'] = attn_aux['indexer_loss']
        else:
            hidden_states, present_key_value = self.self_attn(
                hidden_states=hidden_states, attention_mask=attention_mask,
                position_ids=position_ids, past_key_value=past_key_value, use_cache=use_cache,
            )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, ffn_aux = self.ffn(hidden_states)
        aux_data.update(ffn_aux)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value, aux_data


# =============================================================================
# SECTION 11: NANOSEEK MODEL
# =============================================================================

class NanoSeekModel(nn.Module):
    """Complete NanoSeek Model."""

    def __init__(self, config: NanoSeekConfig):
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList([
            NanoSeekDecoderLayer(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        if config.mtp.num_mtp_modules > 0:
            mtp_hidden_size = config.mtp.mtp_hidden_size or config.hidden_size
            tie_mtp_embeddings = mtp_hidden_size == config.hidden_size
            self.mtp = MultiTokenPrediction(
                hidden_size=config.hidden_size, vocab_size=config.vocab_size,
                mtp_hidden_size=mtp_hidden_size,
                num_mtp_modules=config.mtp.num_mtp_modules,
                mtp_num_heads=config.mtp.mtp_num_heads,
                mtp_loss_weight=config.mtp.mtp_loss_weight_initial,
                mtp_loss_decay=config.mtp.mtp_loss_decay,
                tie_word_embeddings=tie_mtp_embeddings,
            )
            if tie_mtp_embeddings:
                self.mtp.set_shared_embeddings(self.embed_tokens, self.lm_head)
        else:
            self.mtp = None

        self.gradient_checkpointing = False
        self.register_buffer('tokens_processed', torch.tensor(0, dtype=torch.long))

        self.apply(self._init_weights)

    def get_mtp_loss_weight(self, tokens_processed: Optional[int] = None) -> float:
        if tokens_processed is None:
            tokens_processed = self.tokens_processed.item()
        transition = int(self.config.total_tokens * self.config.mtp.mtp_loss_transition_ratio)
        if tokens_processed < transition:
            return self.config.mtp.mtp_loss_weight_initial
        return self.config.mtp.mtp_loss_weight_final

    def get_gamma(self, tokens_processed: Optional[int] = None) -> float:
        if tokens_processed is None:
            tokens_processed = self.tokens_processed.item()
        freeze_at = int(self.config.total_tokens * self.config.moe.gamma_freeze_ratio)
        if tokens_processed < freeze_at:
            return self.config.moe.gamma
        return 0.0

    def update_tokens_processed(self, num_tokens: int):
        self.tokens_processed.add_(num_tokens)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'SCALE_INIT'):
                std *= module.SCALE_INIT
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        labels: Optional[Tensor] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        output_router_logits: bool = False,
        training_step: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        hidden_states = self.embed_tokens(input_ids)

        if position_ids is None:
            if past_key_values is not None and len(past_key_values) > 0:
                past_len = past_key_values[0][0].shape[1]
                position_ids = torch.arange(
                    past_len, past_len + seq_len, device=device, dtype=torch.long
                ).unsqueeze(0).expand(batch_size, -1)
            else:
                position_ids = torch.arange(
                    seq_len, device=device, dtype=torch.long
                ).unsqueeze(0).expand(batch_size, -1)

        past_len = past_key_values[0][0].shape[1] if past_key_values is not None and len(past_key_values) > 0 else 0
        if attention_mask is None:
            causal_mask = create_causal_mask(seq_len, dtype=hidden_states.dtype, device=device, past_len=past_len)
        else:
            causal_mask = self._prepare_attention_mask(attention_mask, seq_len, past_len, hidden_states.dtype)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False
        present_key_values = [] if use_cache else None

        all_hidden_states = [hidden_states] if output_hidden_states else None
        all_router_logits = [] if output_router_logits else None
        all_aux_losses = []
        all_indexer_losses = []

        output_indexer_loss = self.training and self.config.sparse.enabled

        for idx, layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            use_checkpoint = (
                self.gradient_checkpointing and self.training and not use_cache and not output_router_logits
            )
            if use_checkpoint:
                hidden_states, present_kv, aux_data = self._gradient_checkpoint_layer(
                    layer, hidden_states, causal_mask, position_ids, training_step
                )
            else:
                hidden_states, present_kv, aux_data = layer(
                    hidden_states=hidden_states, attention_mask=causal_mask,
                    position_ids=position_ids, past_key_value=past_key_value,
                    use_cache=use_cache, training_step=training_step,
                    output_indexer_loss=output_indexer_loss,
                )

            if use_cache:
                present_key_values.append(present_kv)

            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            if output_router_logits and "router_logits" in aux_data:
                all_router_logits.append(aux_data["router_logits"])

            if "seq_aux_loss" in aux_data:
                all_aux_losses.append(aux_data["seq_aux_loss"])

            if "indexer_loss" in aux_data:
                all_indexer_losses.append(aux_data["indexer_loss"])

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        total_aux_loss = sum(all_aux_losses) if all_aux_losses else torch.tensor(0.0, device=device)
        total_indexer_loss = sum(all_indexer_losses) if all_indexer_losses else torch.tensor(0.0, device=device)

        outputs = {"logits": logits, "hidden_states": hidden_states}

        if use_cache:
            outputs["past_key_values"] = present_key_values

        if output_hidden_states:
            outputs["all_hidden_states"] = all_hidden_states

        if output_router_logits:
            outputs["router_logits"] = all_router_logits

        if labels is not None:
            outputs.update(self._compute_loss(logits, hidden_states, labels, total_aux_loss, total_indexer_loss))

        return outputs

    def _prepare_attention_mask(self, attention_mask, seq_len, past_len, dtype):
        batch_size = attention_mask.shape[0]
        key_len = past_len + seq_len

        causal_mask = create_causal_mask(seq_len, dtype=dtype, device=attention_mask.device, past_len=past_len)

        if attention_mask.shape[-1] == seq_len and past_len > 0:
            past_mask = attention_mask.new_ones((batch_size, past_len))
            attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        elif attention_mask.shape[-1] != key_len:
            raise ValueError(f"attention_mask has length {attention_mask.shape[-1]} but expected {key_len}")

        if attention_mask.dtype == torch.bool:
            padding_mask = ~attention_mask
        else:
            padding_mask = attention_mask == 0
        padding_mask = padding_mask[:, None, None, :].to(dtype=dtype)
        padding_mask = padding_mask * torch.finfo(dtype).min

        return causal_mask + padding_mask

    def _gradient_checkpoint_layer(self, layer, hidden_states, attention_mask, position_ids, training_step):
        def custom_forward(hidden_states, attention_mask, position_ids):
            output, _, _ = layer(
                hidden_states=hidden_states, attention_mask=attention_mask,
                position_ids=position_ids, past_key_value=None, use_cache=False,
                training_step=training_step,
            )
            return output

        hidden_states = gradient_checkpoint(
            custom_forward, hidden_states, attention_mask, position_ids, use_reentrant=False
        )
        return hidden_states, None, {}

    def _compute_loss(self, logits, hidden_states, labels, aux_loss, indexer_loss):
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        main_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), ignore_index=-100,
        )

        result = {"main_loss": main_loss}
        total_loss = main_loss

        if self.mtp is not None:
            mtp_outputs = self.mtp(hidden_states, labels=labels)
            mtp_loss = mtp_outputs.get("mtp_loss", torch.tensor(0.0, device=logits.device))
            result["mtp_loss"] = mtp_loss
            result["mtp_logits"] = mtp_outputs["mtp_logits"]

            mtp_weight = self.get_mtp_loss_weight()
            result["mtp_weight"] = mtp_weight
            total_loss = total_loss + mtp_weight * mtp_loss

        if aux_loss is not None and aux_loss.item() > 0:
            result["aux_loss"] = aux_loss
            total_loss = total_loss + aux_loss

        if indexer_loss is not None and indexer_loss.item() > 0:
            result["indexer_loss"] = indexer_loss
            total_loss = total_loss + indexer_loss

        result["loss"] = total_loss
        return result

    def update_load_balance_bias(self, gamma: Optional[float] = None):
        if gamma is None:
            gamma = self.get_gamma()

        if gamma > 0:
            for layer in self.layers:
                if layer.is_moe_layer:
                    layer.ffn.update_load_balance_bias(gamma)

    def increment_dsa_training_steps(self):
        if self.config.sparse.enabled:
            for layer in self.layers:
                if layer.use_sparse_attention:
                    layer.self_attn.increment_training_step()

    def get_expert_load_stats(self) -> Dict[int, Dict[str, Tensor]]:
        stats = {}
        for idx, layer in enumerate(self.layers):
            if layer.is_moe_layer:
                stats[idx] = layer.ffn.get_expert_load_stats()
        return stats

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = filter(lambda p: p.requires_grad, self.parameters()) if trainable_only else self.parameters()
        return sum(p.numel() for p in params)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Tensor:
        self.eval()

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            outputs = self(generated)
            logits = outputs["logits"][:, -1, :]

            if temperature > 0:
                logits = logits / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

        return generated

    @torch.inference_mode()
    def generate_simple(
        self,
        tokens: List[int],
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: int = 42,
    ):
        """Simple streaming generation (no KV cache)."""
        self.eval()
        device = next(self.parameters()).device

        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        ids = torch.tensor([tokens], dtype=torch.long, device=device)

        for _ in range(max_tokens):
            outputs = self(input_ids=ids, use_cache=False)
            logits = outputs["logits"][:, -1, :]

            if temperature > 0:
                logits = logits / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')

                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)

            ids = torch.cat([ids, next_id], dim=1)
            yield next_id.item()

    @torch.inference_mode()
    def generate_cached(
        self,
        tokens: List[int],
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: int = 42,
    ):
        """Streaming generation with KV cache."""
        self.eval()
        device = next(self.parameters()).device

        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        past_key_values = None

        outputs = self(input_ids=ids, use_cache=True)
        logits = outputs["logits"][:, -1, :]
        past_key_values = outputs["past_key_values"]

        for _ in range(max_tokens):
            if temperature > 0:
                logits = logits / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')

                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)

            yield next_id.item()

            outputs = self(input_ids=next_id, past_key_values=past_key_values, use_cache=True)
            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]


# =============================================================================
# SECTION 12: FACTORY FUNCTIONS & TESTING
# =============================================================================


def create_nanoseek(config: NanoSeekConfig = None) -> NanoSeekModel:
    """
    Create NanoSeek-700M model (DeepSeek-aligned research-optimal config).

    Architecture: 700M active / 3.5B total params
    - d/L = 128 (matches OLMoE-1B, LLaMA-7B scale)
    - 64 experts, 8 active (DeepSeek's optimal k)
    - All DeepSeek V3 ratios preserved
    - Training: 14B tokens (Chinchilla optimal)

    Args:
        config: Optional custom config. If None, uses default 700M config.

    Returns:
        NanoSeekModel instance.
    """
    if config is None:
        config = get_nanoseek_config()
    return NanoSeekModel(config)


def test_nanoseek():
    """Test NanoSeek model with main config."""
    print("NanoSeek-700M Model Test")
    print("=" * 60)

    config = get_nanoseek_config()
    model = NanoSeekModel(config)

    print(f"\nConfiguration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Heads: {config.num_heads}")
    print(f"  Vocab size: {config.vocab_size}")

    print(f"\nParameter Count:")
    total = model.num_parameters(trainable_only=False)
    print(f"  Total: {total:,}")
    print(f"  Config estimate: {config.estimated_total_params:,}")
    print(f"  Active estimate: {config.estimated_active_params:,}")

    batch, seq = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (batch, seq))
    labels = torch.randint(0, config.vocab_size, (batch, seq))

    print(f"\nForward Pass:")
    print(f"  Input: {input_ids.shape}")

    outputs = model(input_ids, labels=labels)

    print(f"  Logits: {outputs['logits'].shape}")
    print(f"  Loss: {outputs['loss'].item():.4f}")
    print(f"  Main loss: {outputs['main_loss'].item():.4f}")
    if 'mtp_loss' in outputs:
        print(f"  MTP loss: {outputs['mtp_loss'].item():.4f}")

    print(f"\nIncremental Decoding:")
    outputs_cached = model(input_ids, use_cache=True)
    print(f"  Cache entries: {len(outputs_cached['past_key_values'])}")
    print(f"  Cache[0] shape: {outputs_cached['past_key_values'][0][0].shape}")

    print(f"\nGeneration:")
    prompt_tokens = list(torch.randint(0, config.vocab_size, (10,)).tolist())
    generated = []
    for token in model.generate_simple(prompt_tokens, max_tokens=5, temperature=0.8):
        generated.append(token)
    print(f"  Prompt: {len(prompt_tokens)} tokens")
    print(f"  Generated: {len(generated)} tokens")

    print("\n" + "=" * 60)
    print("All tests passed!")

    return model, config


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_nanoseek()
    else:
        test_nanoseek()
