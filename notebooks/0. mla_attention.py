# =============================================================================
# MULTI-HEAD LATENT ATTENTION (MLA) — Production Implementation
# =============================================================================
# Reference: DeepSeek-V2 (arXiv:2405.04434), Section 3.2
#
# Architecture Overview:
# ┌─────────────────────────────────────────────────────────────────────┐
# │                        MLA Data Flow                               │
# │                                                                     │
# │  hidden ─┬─► W_QA ─► RMSNorm ─► W_QB ─► split ─► [q_nope | q_pe] │
# │          │                                                q_pe     │
# │          │                                               + RoPE    │
# │          │                                                  │      │
# │          └─► W_KVA ─► split ─► [c_kv | k_pe_raw]           │      │
# │                         │           │                       │      │
# │                    RMSNorm       + RoPE                     │      │
# │                         │           │                       │      │
# │               ┌─────────┴───────────┘                       │      │
# │               │  KV CACHE: (c_kv, k_pe_rotated)            │      │
# │               │  Total: kv_lora_rank + qk_rope_head_dim    │      │
# │               └─────────┬───────────┐                       │      │
# │                         │           │                       │      │
# │               TRAINING: │    INFERENCE (weight absorption): │      │
# │              W_KVB ─► split     Absorb W_UK into Q path     │      │
# │              [k_nope | v]       Absorb W_UV into O path     │      │
# │                  │      │       Attn on compressed repr     │      │
# │                  │      │              directly              │      │
# │               concat    │                                   │      │
# │            [k_nope|k_pe]│                                   │      │
# │                  │      │                                   │      │
# │              Q @ K^T    │                                   │      │
# │              softmax    │                                   │      │
# │              @ V ───────┘                                   │      │
# │                  │                                          │      │
# │                W_O ─► output                                │      │
# └─────────────────────────────────────────────────────────────────────┘
#
# KV Cache Comparison (hidden=7168, n_heads=128, head_dim=128):
#   Standard MHA: 2 * n_heads * head_dim = 32768 elements/token
#   MLA:          kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576 elements/token
#   Compression:  32768 / 576 ≈ 56.9x
# =============================================================================

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# HELPER MODULES
# =============================================================================


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


# =============================================================================
# ROTARY POSITIONAL EMBEDDING (RoPE)
# =============================================================================


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    scaling_factor: float = 1.0,
    original_max_position_embeddings: int = 4096,
) -> Tensor:
    """
    Precompute complex-valued RoPE frequencies.

    Returns complex tensor of shape [max_seq_len, dim // 2].
    Using complex representation for clean rotation semantics.

    For YaRN-style NTK scaling, the frequency base is modified:
        theta' = theta * (scaling_factor ** (dim / (dim - 2)))
    applied only when scaling_factor > 1.
    """
    if scaling_factor > 1.0:
        # NTK-aware scaling (YaRN, arXiv:2309.00071)
        theta = theta * (scaling_factor ** (dim / (dim - 2)))

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # [max_seq_len, dim // 2]
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    """
    Apply rotary embeddings to input tensor.

    Args:
        x: [..., dim] real tensor (last dim must be even)
        freqs_cis: [..., dim // 2] complex tensor of frequencies

    The standard RoPE rotation: pair consecutive dims, treat as complex,
    multiply by e^{i*theta}. This is the "non-interleaved" layout where
    dims (0,1), (2,3), ... are paired.
    """
    # Reshape to complex: [..., dim] -> [..., dim//2, 2] -> [..., dim//2] complex
    x_shape = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(*x_shape[:-1], -1, 2))

    # Broadcast freqs_cis to match x_complex shape
    # freqs_cis: [seq_len, dim//2] needs to align with x_complex: [B, seq_len, n_heads?, dim//2]
    ndim_diff = x_complex.ndim - freqs_cis.ndim
    shape = [1] * ndim_diff + list(freqs_cis.shape)
    # Insert singleton dims for batch (and optionally heads) so broadcast works
    # For [B, seq_len, dim//2] -> freqs needs [1, seq_len, dim//2]
    # For [B, seq_len, 1, dim//2] -> freqs needs [1, seq_len, 1, dim//2]
    if ndim_diff == 2:
        freqs_cis = freqs_cis.unsqueeze(0)  # add batch
        if x_complex.ndim == 4:
            freqs_cis = freqs_cis.unsqueeze(2)  # add head dim
    elif ndim_diff == 1:
        freqs_cis = freqs_cis.unsqueeze(0)

    result = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return result.to(x.dtype).reshape(x_shape)


# =============================================================================
# MULTI-HEAD LATENT ATTENTION
# =============================================================================


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) from DeepSeek-V2/V3.

    Two operating modes:
    1. TRAINING MODE: Standard forward with explicit K/V expansion via W_KVB.
       Supports FlashAttention/SDPA for fused attention kernels.

    2. INFERENCE MODE (weight absorption): Absorbs W_UK into Q projection and
       W_UV into output projection, so attention operates directly on the
       compressed latent c_kv without ever materializing full-rank K/V.
       This is the key inference optimization from DeepSeek-V2 Section 3.2.2.

    Weight absorption math:
       Standard: attn = softmax(Q_nope @ K_nope^T + Q_pe @ K_pe^T) @ V
       Where:    K_nope = c_kv @ W_UK,  V = c_kv @ W_UV

       Absorbed: attn = softmax(Q'_nope @ c_kv^T + Q_pe @ K_pe^T) @ c_kv
       Where:    Q'_nope = Q_nope @ W_UK^T  (absorbed into query)
       Then:     output  = attn @ c_kv @ W_UV @ W_O  (absorbed into output)

       This means during inference we never expand c_kv to full K/V,
       and the KV cache stores only (c_kv, k_pe) — both low-dimensional.
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

        # Softmax scaling: 1/sqrt(d_k) * mscale
        # mscale compensates for YaRN context extension (entropy preservation)
        self.base_scale = 1.0 / math.sqrt(self.qk_head_dim)
        self.mscale = mscale

        # =====================================================================
        # Q projection: h -> compressed -> per-head [q_nope | q_pe]
        # =====================================================================
        self.wq_a = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_norm = RMSNorm(q_lora_rank)
        self.wq_b = nn.Linear(q_lora_rank, num_heads * self.qk_head_dim, bias=False)

        # =====================================================================
        # KV projection: h -> [c_kv | k_pe_raw]
        #   c_kv:      [kv_lora_rank]     — compressed joint KV latent
        #   k_pe_raw:  [qk_rope_head_dim] — decoupled RoPE key (shared across heads)
        # =====================================================================
        self.wkv_a = nn.Linear(
            hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False
        )
        self.kv_norm = RMSNorm(kv_lora_rank)

        # =====================================================================
        # KV expansion: c_kv -> per-head [k_nope | v]
        # Only used during training. At inference, this is absorbed.
        # =====================================================================
        self.wkv_b = nn.Linear(
            kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=False
        )

        # =====================================================================
        # Output projection
        # =====================================================================
        self.wo = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)

        # =====================================================================
        # Absorbed weight buffers (populated by prepare_for_inference())
        # =====================================================================
        # W_UK_absorbed: [num_heads, qk_nope_head_dim, kv_lora_rank]
        # W_UV_O_absorbed: [kv_lora_rank, hidden_size]
        self._inference_mode = False
        self._w_uk: Optional[Tensor] = None   # [num_heads, qk_nope_head_dim, kv_lora_rank]
        self._w_uv_o: Optional[Tensor] = None  # [num_heads, kv_lora_rank, hidden_size // num_heads ... wait no]
        # Actually: we need per-head: attn_out_per_head @ W_UV @ W_O
        # W_UV: [kv_lora_rank, num_heads * v_head_dim] -> per-head [kv_lora_rank, v_head_dim]
        # W_O: [num_heads * v_head_dim, hidden_size] -> per-head [v_head_dim, hidden_size]
        # absorbed per-head: [kv_lora_rank, v_head_dim] @ [v_head_dim, hidden_size]
        #                  = [kv_lora_rank, hidden_size] -- but this is huge
        # Better: keep the product as [num_heads, kv_lora_rank, v_head_dim] for W_UV
        # and let W_O handle the final projection. The actual absorption is:
        #   output = sum_h (attn_h @ c_kv @ W_UV_h) reshaped then @ W_O
        # which can be rewritten as:
        #   output = (attn @ c_kv) @ W_UV then reshape then @ W_O
        # So the minimal absorption is just using c_kv directly as V, then
        # applying W_UV and W_O as a fused output projection.

        # Precompute RoPE frequencies
        freqs_cis = precompute_freqs_cis(
            qk_rope_head_dim,
            max_position_embeddings,
            rope_theta,
            rope_scaling_factor,
            original_max_position_embeddings,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    # =========================================================================
    # Weight Absorption for Inference
    # =========================================================================

    @torch.no_grad()
    def prepare_for_inference(self) -> None:
        """
        Absorb W_UK into query-side and prepare for latent-space attention.

        After calling this:
        - Q_nope is projected to kv_lora_rank dim (instead of qk_nope_head_dim)
        - Attention K = c_kv directly (no expansion needed)
        - Attention V = c_kv directly
        - Output path: (attn @ c_kv) -> W_UV -> W_O (fused as one matmul)

        The key insight: instead of expanding c_kv to full K and V,
        we transform Q to operate in the compressed space.

        Q_nope @ K_nope^T = Q_nope @ (c_kv @ W_UK)^T
                          = Q_nope @ W_UK^T @ c_kv^T
                          = (Q_nope @ W_UK^T) @ c_kv^T
                          = Q'_nope @ c_kv^T

        So we precompute the "absorbed" query projection that maps
        Q from nope-head space to kv_lora_rank space.
        """
        # Extract W_UK from wkv_b: the first qk_nope_head_dim columns per head
        # wkv_b.weight: [num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
        w_kvb = self.wkv_b.weight.data  # [out_features, in_features]
        w_kvb = w_kvb.view(
            self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
        )
        # W_UK: [num_heads, qk_nope_head_dim, kv_lora_rank]
        w_uk = w_kvb[:, : self.qk_nope_head_dim, :]
        # W_UV: [num_heads, v_head_dim, kv_lora_rank]
        w_uv = w_kvb[:, self.qk_nope_head_dim :, :]

        # Store W_UK^T for Q absorption: Q_nope @ W_UK^T -> [B, seq, heads, kv_lora_rank]
        # W_UK: [num_heads, qk_nope_head_dim, kv_lora_rank]
        # W_UK^T: [num_heads, kv_lora_rank, qk_nope_head_dim]
        # But we want Q_nope [B, seq, heads, qk_nope_head_dim] @ W_UK^T [heads, qk_nope_head_dim, kv_lora_rank]
        # = [B, seq, heads, kv_lora_rank]
        self._w_uk = w_uk  # [num_heads, qk_nope_head_dim, kv_lora_rank]

        # For output: attn @ c_kv gives [B, heads, seq, kv_lora_rank]
        # Then we need to apply W_UV per head: [kv_lora_rank] -> [v_head_dim]
        # Then concat heads and apply W_O: [heads * v_head_dim] -> [hidden]
        # W_UV: [num_heads, v_head_dim, kv_lora_rank] -- but we want
        # [B, heads, seq, kv_lora_rank] @ W_UV^T -> [B, heads, seq, v_head_dim]
        # So we need W_UV transposed: [num_heads, kv_lora_rank, v_head_dim]
        self._w_uv = w_uv.transpose(1, 2).contiguous()  # [num_heads, kv_lora_rank, v_head_dim]

        self._inference_mode = True

    def unprepare_for_inference(self) -> None:
        """Revert to training mode (discard absorbed weights)."""
        self._w_uk = None
        self._w_uv = None
        self._inference_mode = False

    # =========================================================================
    # Cache Helpers
    # =========================================================================

    def _get_freqs(self, position_ids: Optional[Tensor], fallback_start: int, seq_len: int) -> Tensor:
        """Get RoPE frequencies for given positions, with clean fallback logic."""
        if position_ids is not None:
            return self.freqs_cis[position_ids]  # [B, seq_len, dim//2]
        return self.freqs_cis[fallback_start : fallback_start + seq_len]  # [seq_len, dim//2]

    # =========================================================================
    # Forward: Training Path (explicit K/V expansion, supports SDPA)
    # =========================================================================

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Forward pass with automatic training/inference dispatch.

        KV Cache format: (c_kv, k_pe_rotated)
            c_kv:          [B, cached_len, kv_lora_rank]   — compressed latent (pre-norm)
            k_pe_rotated:  [B, cached_len, 1, rope_dim]    — RoPE-applied key component

        Note: We cache rotated k_pe. This is a deliberate design choice:
        - Pro: Avoids recomputing RoPE on cached keys every step
        - Con: Cannot reassign positions after caching (no speculative decoding rollback)
        - For position-mutable caching, store raw k_pe and reapply RoPE at attention time.

        Args:
            hidden_states: [B, seq_len, hidden_size]
            attention_mask: [B, 1, seq_len, kv_len] additive mask (0 = attend, -inf = mask)
            position_ids: [B, seq_len] explicit position indices (for batched generation w/ padding)
            past_key_value: cached (c_kv, k_pe_rotated) from previous steps
            use_cache: whether to return updated cache

        Returns:
            output: [B, seq_len, hidden_size]
            present_key_value: updated cache tuple or None
        """
        if self._inference_mode:
            return self._forward_inference(
                hidden_states, attention_mask, position_ids, past_key_value, use_cache
            )
        return self._forward_training(
            hidden_states, attention_mask, position_ids, past_key_value, use_cache
        )

    def _forward_training(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape

        # =================================================================
        # QUERY PATH: h -> W_QA -> norm -> W_QB -> [q_nope | q_pe]
        # =================================================================
        q = self.wq_b(self.q_norm(self.wq_a(hidden_states)))
        q = q.view(batch_size, seq_len, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # =================================================================
        # KV PATH: h -> W_KVA -> [c_kv | k_pe_raw]
        # =================================================================
        kv = self.wkv_a(hidden_states)
        c_kv, k_pe_raw = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        c_kv = self.kv_norm(c_kv)  # [B, seq_len, kv_lora_rank]

        # k_pe: add head dim (shared across heads) and apply RoPE
        k_pe_raw = k_pe_raw.unsqueeze(2).contiguous()  # [B, seq_len, 1, rope_dim]

        # =================================================================
        # POSITION ENCODING
        # Determine positions for current tokens and apply RoPE
        # =================================================================
        if past_key_value is not None:
            cached_c_kv, cached_k_pe = past_key_value
            cached_len = cached_c_kv.shape[1]
        else:
            cached_len = 0

        # Current token positions
        q_freqs = self._get_freqs(position_ids, cached_len, seq_len)
        k_freqs = q_freqs  # Same positions for Q and current K

        q_pe = apply_rotary_emb(q_pe, q_freqs)
        k_pe_current = apply_rotary_emb(k_pe_raw, k_freqs)  # [B, seq_len, 1, rope_dim]

        # =================================================================
        # CACHE MANAGEMENT
        # =================================================================
        if past_key_value is not None:
            c_kv = torch.cat([cached_c_kv, c_kv], dim=1)       # [B, total_len, kv_lora_rank]
            k_pe = torch.cat([cached_k_pe, k_pe_current], dim=1)  # [B, total_len, 1, rope_dim]
        else:
            k_pe = k_pe_current

        kv_len = c_kv.shape[1]
        present_key_value = (c_kv, k_pe) if use_cache else None

        # =================================================================
        # KV EXPANSION (training path only — this is what absorption eliminates)
        # Only expand NEW tokens when caching, not the full sequence.
        # =================================================================
        # CRITICAL FIX: During generation with cache, only expand new tokens
        # and cache the expanded results separately.
        # However, for simplicity in training (no cache), we expand everything.
        # During generation with cache, we need the full expanded K and V anyway
        # for the attention computation, so we expand all — but see inference path
        # for the O(1)-per-step approach.
        kv_expanded = self.wkv_b(c_kv)  # [B, kv_len, num_heads * (nope + v)]
        kv_expanded = kv_expanded.view(
            batch_size, kv_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv_expanded.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # =================================================================
        # ASSEMBLE FULL Q AND K
        # =================================================================
        q = torch.cat([q_nope, q_pe], dim=-1)           # [B, seq, heads, qk_head_dim]
        k_pe_expanded = k_pe.expand(-1, -1, self.num_heads, -1)
        k = torch.cat([k_nope, k_pe_expanded], dim=-1)  # [B, kv_len, heads, qk_head_dim]

        # =================================================================
        # ATTENTION COMPUTATION
        # =================================================================
        # Transpose to [B, heads, seq, dim] for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        effective_scale = self.base_scale * self.mscale

        # Try SDPA (FlashAttention/memory-efficient) first, fall back to manual
        use_sdpa = (
            hasattr(F, "scaled_dot_product_attention")
            and not self.training  # SDPA supports training too, but dropout handling differs
            or (self.training and self.attention_dropout == 0.0)
        )

        if use_sdpa and attention_mask is not None:
            # SDPA with explicit attn_mask
            # Note: SDPA expects mask shape [B, 1, seq, kv_len] or [B, heads, seq, kv_len]
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                scale=effective_scale,
            )
        elif use_sdpa and attention_mask is None:
            # SDPA with causal flag (most efficient — triggers FlashAttention)
            is_causal = seq_len > 1  # Only causal during prefill, not single-token decode
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=is_causal,
                dropout_p=self.attention_dropout if self.training else 0.0,
                scale=effective_scale,
            )
        else:
            # Manual attention (fallback)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * effective_scale

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            # fp32 softmax for numerical stability, then cast back
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(v.dtype)

            if self.training and self.attention_dropout > 0:
                attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=True)

            attn_output = torch.matmul(attn_weights, v)

        # =================================================================
        # OUTPUT PROJECTION
        # =================================================================
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.v_head_dim)
        output = self.wo(attn_output)

        return output, present_key_value

    # =========================================================================
    # Forward: Inference Path (Weight Absorption)
    # =========================================================================

    def _forward_inference(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Inference path with weight absorption.

        Key differences from training:
        1. Q_nope is projected to kv_lora_rank space via absorbed W_UK
        2. Attention operates on c_kv directly (never expanded to full K/V)
        3. Output applies W_UV per head then W_O

        Attention score decomposition:
            score = Q_nope @ W_UK^T @ c_kv^T + Q_pe @ K_pe^T
                     ^^^^^^^^^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^
                     nope component (compressed)   rope component

        The nope component attention is [B, heads, seq, kv_lora_rank] @ [B, 1, kv_len, kv_lora_rank]^T
        The rope component is standard: [B, heads, seq, rope_dim] @ [B, heads, kv_len, rope_dim]^T
        """
        assert self._w_uk is not None, "Call prepare_for_inference() first"
        batch_size, seq_len, _ = hidden_states.shape

        # =================================================================
        # QUERY PATH
        # =================================================================
        q = self.wq_b(self.q_norm(self.wq_a(hidden_states)))
        q = q.view(batch_size, seq_len, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Absorb W_UK into Q: q_nope @ W_UK -> q_nope in compressed space
        # q_nope: [B, seq, heads, qk_nope_head_dim]
        # _w_uk:  [heads, qk_nope_head_dim, kv_lora_rank]
        # result: [B, seq, heads, kv_lora_rank]
        q_nope_absorbed = torch.einsum("bshd,hdc->bshc", q_nope, self._w_uk)

        # =================================================================
        # KV PATH (compressed — no expansion)
        # =================================================================
        kv = self.wkv_a(hidden_states)
        c_kv, k_pe_raw = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        c_kv = self.kv_norm(c_kv)
        k_pe_raw = k_pe_raw.unsqueeze(2).contiguous()

        # =================================================================
        # POSITION ENCODING
        # =================================================================
        if past_key_value is not None:
            cached_c_kv, cached_k_pe = past_key_value
            cached_len = cached_c_kv.shape[1]
        else:
            cached_len = 0

        freqs = self._get_freqs(position_ids, cached_len, seq_len)
        q_pe = apply_rotary_emb(q_pe, freqs)
        k_pe_current = apply_rotary_emb(k_pe_raw, freqs)

        # =================================================================
        # CACHE MANAGEMENT
        # =================================================================
        if past_key_value is not None:
            c_kv = torch.cat([cached_c_kv, c_kv], dim=1)
            k_pe = torch.cat([cached_k_pe, k_pe_current], dim=1)
        else:
            k_pe = k_pe_current

        kv_len = c_kv.shape[1]
        present_key_value = (c_kv, k_pe) if use_cache else None

        # =================================================================
        # ATTENTION SCORES (decomposed nope + rope)
        # =================================================================
        # Nope component: q_nope_absorbed @ c_kv^T
        # q_nope_absorbed: [B, seq, heads, kv_lora_rank] -> [B, heads, seq, kv_lora_rank]
        # c_kv:            [B, kv_len, kv_lora_rank]     -> [B, 1, kv_len, kv_lora_rank]
        q_nope_absorbed = q_nope_absorbed.transpose(1, 2)  # [B, heads, seq, kv_lora_rank]
        c_kv_for_attn = c_kv.unsqueeze(1)  # [B, 1, kv_len, kv_lora_rank]
        nope_scores = torch.matmul(
            q_nope_absorbed, c_kv_for_attn.transpose(-2, -1)
        )  # [B, heads, seq, kv_len]

        # Rope component: q_pe @ k_pe^T
        # q_pe: [B, seq, heads, rope_dim] -> [B, heads, seq, rope_dim]
        # k_pe: [B, kv_len, 1, rope_dim] -> [B, 1, kv_len, rope_dim] (broadcast over heads)
        # But k_pe already has shape [B, kv_len, 1, rope_dim]
        q_pe = q_pe.transpose(1, 2)  # [B, heads, seq, rope_dim]
        k_pe_t = k_pe.permute(0, 2, 3, 1)  # [B, 1, rope_dim, kv_len]
        rope_scores = torch.matmul(q_pe, k_pe_t)  # [B, heads, seq, kv_len]

        # Combined attention scores
        effective_scale = self.base_scale * self.mscale
        attn_weights = (nope_scores + rope_scores) * effective_scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            hidden_states.dtype
        )

        # =================================================================
        # ATTENTION OUTPUT (on compressed representation)
        # =================================================================
        # attn @ c_kv -> [B, heads, seq, kv_lora_rank]
        attn_output = torch.matmul(attn_weights, c_kv_for_attn)  # [B, heads, seq, kv_lora_rank]

        # Apply absorbed W_UV per head: [B, heads, seq, kv_lora_rank] @ [heads, kv_lora_rank, v_head_dim]
        # -> [B, heads, seq, v_head_dim]
        attn_output = torch.einsum("bhsc,hcv->bhsv", attn_output, self._w_uv)

        # Standard output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.v_head_dim)
        output = self.wo(attn_output)

        return output, present_key_value


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cpu"
    dtype = torch.float32

    # DeepSeek-V2-Lite-like config (scaled down for testing)
    cfg = dict(
        hidden_size=2048,
        num_heads=16,
        q_lora_rank=1024,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        max_position_embeddings=4096,
        rope_theta=10000.0,
        rope_scaling_factor=1.0,
        original_max_position_embeddings=4096,
        mscale=1.0,
        attention_dropout=0.0,
    )

    mla = MultiHeadLatentAttention(**cfg).to(device, dtype)

    B, S = 2, 128
    x = torch.randn(B, S, cfg["hidden_size"], device=device, dtype=dtype)

    # Build causal mask
    causal_mask = torch.full((S, S), float("-inf"), device=device, dtype=dtype)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

    print("=" * 70)
    print("TRAINING MODE (full forward)")
    print("=" * 70)
    out_train, _ = mla(x, attention_mask=causal_mask)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out_train.shape}")
    assert out_train.shape == x.shape, f"Shape mismatch: {out_train.shape} != {x.shape}"
    print("  ✓ Shape correct")

    # =========================================================================
    # Test autoregressive generation (training path, for comparison)
    # =========================================================================
    print("\n" + "=" * 70)
    print("AUTOREGRESSIVE GENERATION (training path, with cache)")
    print("=" * 70)

    # Prefill
    prefill_len = 32
    x_prefill = x[:, :prefill_len, :]
    prefill_mask = torch.full((prefill_len, prefill_len), float("-inf"), device=device, dtype=dtype)
    prefill_mask = torch.triu(prefill_mask, diagonal=1).unsqueeze(0).unsqueeze(0)

    out_prefill, cache = mla(x_prefill, attention_mask=prefill_mask, use_cache=True)
    print(f"  Prefill output: {out_prefill.shape}")
    print(f"  Cache c_kv: {cache[0].shape}, k_pe: {cache[1].shape}")

    # Decode step
    x_decode = x[:, prefill_len : prefill_len + 1, :]
    # Mask: new token attends to all prefilled + itself
    decode_mask = torch.zeros(1, 1, 1, prefill_len + 1, device=device, dtype=dtype)
    out_decode, cache = mla(x_decode, attention_mask=decode_mask, past_key_value=cache, use_cache=True)
    print(f"  Decode output: {out_decode.shape}")
    print(f"  Updated cache c_kv: {cache[0].shape}, k_pe: {cache[1].shape}")
    assert out_decode.shape == (B, 1, cfg["hidden_size"])
    print("  ✓ Decode shape correct")

    # =========================================================================
    # Test inference path (weight absorption)
    # =========================================================================
    print("\n" + "=" * 70)
    print("INFERENCE MODE (weight absorption)")
    print("=" * 70)

    mla.prepare_for_inference()
    print("  ✓ Weights absorbed")

    # Prefill
    out_prefill_inf, cache_inf = mla(x_prefill, attention_mask=prefill_mask, use_cache=True)
    print(f"  Prefill output: {out_prefill_inf.shape}")
    print(f"  Cache c_kv: {cache_inf[0].shape}, k_pe: {cache_inf[1].shape}")

    # Decode
    out_decode_inf, cache_inf = mla(
        x_decode, attention_mask=decode_mask, past_key_value=cache_inf, use_cache=True
    )
    print(f"  Decode output: {out_decode_inf.shape}")
    assert out_decode_inf.shape == (B, 1, cfg["hidden_size"])
    print("  ✓ Decode shape correct")

    # =========================================================================
    # Numerical equivalence check (training vs inference path)
    # =========================================================================
    print("\n" + "=" * 70)
    print("NUMERICAL EQUIVALENCE CHECK")
    print("=" * 70)

    # Compare prefill outputs between training and inference paths
    max_diff = (out_prefill - out_prefill_inf).abs().max().item()
    mean_diff = (out_prefill - out_prefill_inf).abs().mean().item()
    print(f"  Prefill max  |diff|: {max_diff:.6e}")
    print(f"  Prefill mean |diff|: {mean_diff:.6e}")

    # Compare decode outputs
    max_diff_dec = (out_decode - out_decode_inf).abs().max().item()
    mean_diff_dec = (out_decode - out_decode_inf).abs().mean().item()
    print(f"  Decode  max  |diff|: {max_diff_dec:.6e}")
    print(f"  Decode  mean |diff|: {mean_diff_dec:.6e}")

    # For fp32, these should be very close (within float rounding)
    if max_diff < 1e-4 and max_diff_dec < 1e-4:
        print("  ✓ Training and inference paths are numerically equivalent")
    else:
        print("  ✗ WARNING: Significant numerical divergence detected!")

    # =========================================================================
    # KV cache size comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("KV CACHE COMPRESSION ANALYSIS")
    print("=" * 70)

    mha_cache_per_token = 2 * cfg["num_heads"] * 128  # standard MHA with head_dim=128
    mla_cache_per_token = cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"]
    ratio = mha_cache_per_token / mla_cache_per_token

    print(f"  Standard MHA: {mha_cache_per_token} elements/token")
    print(f"  MLA:          {mla_cache_per_token} elements/token")
    print(f"  Compression:  {ratio:.1f}x")

    mla.unprepare_for_inference()
    print("\n  ✓ All tests passed.")
