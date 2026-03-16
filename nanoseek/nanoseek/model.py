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
        YaRNConfig,
        SparseAttentionConfig,
        get_nanoseek_config,
    )
except ImportError:
    from config import (
        NanoSeekConfig,
        YaRNConfig,
        SparseAttentionConfig,
        get_nanoseek_config,
    )

# =============================================================================
# SECTION 1.5: CASTLINEAR (DTYPE-CASTING LINEAR)
# =============================================================================
# Adapted from nanochat's Linear class. Master weights stay fp32 for optimizer
# precision (Adam second moments need mantissa bits). The cast to activation
# dtype (bf16) happens in the matmul call — no autocast overhead.
#
# Why this matters for MoE: 64 experts × 3 weights = 192 Linear layers in experts
# alone. Autocast context manager overhead per-layer adds up. Explicit cast also
# makes torch.compile happier (fewer dtype-related graph breaks).
#
# Reference: nanochat/gpt.py (Linear class)

class CastLinear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.

    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings).
    """
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(dtype=x.dtype))


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
) -> Tensor:
    """Precompute rotary frequencies with YaRN scaling.

    Matches DeepSeek V3 official implementation:
    - Base freqs: 1/(theta^(2i/dim)) for i in [0, dim/2)
    - YaRN: smooth interpolation between scaled and unscaled freqs
      using correction range from beta_fast/beta_slow

    Args:
        dim: RoPE dimension (qk_rope_head_dim)
        end: Maximum sequence length to precompute
        theta: RoPE base frequency
        scaling_factor: YaRN scaling factor (1.0 = no scaling)
        original_max_position_embeddings: Original context length before YaRN
        beta_fast: High rotation threshold for YaRN correction range
        beta_slow: Low rotation threshold for YaRN correction range

    Returns:
        Complex tensor of shape [end, dim//2] containing e^(i*freq*t)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    if scaling_factor != 1.0:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, theta, original_max_position_embeddings
        )
        # DeepSeek V3: smooth=1 means "keep original freq", smooth=0 means "scale down"
        # 1 - linear_ramp gives: low dims (high freq) → smooth=1 (keep), high dims (low freq) → smooth=0 (scale)
        smooth = 1.0 - linear_ramp_factor(low, high, dim // 2)
        freqs = (freqs / scaling_factor) * (1 - smooth) + freqs * smooth

    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    """Apply rotary position embeddings via complex multiplication.

    Matches DeepSeek V3 official implementation exactly.
    Uses interleaved layout: last dim contains [re0, im0, re1, im1, ...] pairs.

    Args:
        x: Input tensor [B, S, heads, rope_dim] or [B, S, 1, rope_dim] (shared k_pe)
        freqs_cis: Complex frequencies [S, rope_dim//2] or [B, S, rope_dim//2]

    Returns:
        Rotated tensor, same shape and dtype as x.
    """
    dtype = x.dtype
    # x: [..., head_dim] → [..., head_dim//2, 2] → complex [..., head_dim//2]
    # .contiguous() needed when x comes from split/unsqueeze (non-contiguous strides)
    # .clone() is needed (not just .contiguous()) because split() can produce
    # tensors with odd storage offsets (e.g., kv_lora_rank=143). view_as_complex
    # requires storage_offset divisible by 2, which .contiguous() doesn't fix.
    x_complex = torch.view_as_complex(x.float().clone().view(*x.shape[:-1], -1, 2))
    # freqs_cis: reshape for broadcasting over batch and heads
    # 2D [S, dim//2] → [1, S, 1, dim//2]  (no position_ids, single batch)
    # 3D [B, S, dim//2] → [B, S, 1, dim//2]  (explicit position_ids, batched)
    if freqs_cis.dim() == 2:
        freqs_cis = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))
    else:
        freqs_cis = freqs_cis.unsqueeze(2)  # [B, S, 1, dim//2]
    # Complex multiply applies rotation, then flatten back to real pairs
    x_out = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return x_out.to(dtype)

# =============================================================================
# SECTION 3: RMSNORM
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
# SECTION 4: MULTI-HEAD LATENT ATTENTION (MLA)
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
        # Store base scale and mscale separately so mscale can be adjusted
        # at inference time for YaRN context extension without reinitializing.
        # DeepSeek V3 applies mscale SQUARED: softmax_scale *= mscale * mscale
        # (see DeepSeek-V3/inference/model.py and HF yarn_get_mscale with mscale_all_dim).
        # When mscale=1.0 this is a no-op; matters for YaRN extended context.
        self.base_scale = 1.0 / math.sqrt(self.qk_head_dim)
        self.mscale = mscale

        # =====================================================================
        # Q projection: h -> compressed -> per-head [q_nope | q_pe] 
        # =====================================================================
        self.wq_a = CastLinear(hidden_size, q_lora_rank, bias=False)
        self.q_norm = RMSNorm(q_lora_rank)
        self.wq_b = CastLinear(q_lora_rank, num_heads * self.qk_head_dim, bias=False)

        # KV projection path
        #   c_kv:      [kv_lora_rank]     — compressed joint KV latent
        #   k_pe_raw:  [qk_rope_head_dim] — decoupled RoPE key (shared across heads)
        self.wkv_a = CastLinear(hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(kv_lora_rank)
        # =====================================================================
        # KV expansion: c_kv -> per-head [k_nope | v]
        # Only used during training. At inference, this is absorbed.
        # =====================================================================
        self.wkv_b = CastLinear(kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=False)

        # Output projection
        self.wo = CastLinear(num_heads * v_head_dim, hidden_size, bias=False)

        # =====================================================================
        # Absorption mode flag (matches DeepSeek's attn_impl: "naive"|"absorb")
        # "absorb" = inference path: W_UK absorbed into Q, attention on c_kv directly
        # "naive"  = training path: explicit K/V expansion via wkv_b
        # =====================================================================
        self.absorb = False

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
    # Cache Helpers
    # =========================================================================

    def _get_freqs(self, position_ids: Optional[Tensor], fallback_start: int, seq_len: int) -> Tensor:
        """Get RoPE frequencies for given positions, with clean fallback logic."""
        if position_ids is not None:
            return self.freqs_cis[position_ids]  # [B, seq_len, dim//2]
        return self.freqs_cis[fallback_start : fallback_start + seq_len]  # [seq_len, dim//2]
    
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Unified forward pass matching DeepSeek V3 inference/model.py pattern.

        Two modes controlled by self.absorb (like DeepSeek's attn_impl="naive"|"absorb"):
        - absorb=False (training): Expand c_kv via wkv_b to full K/V, standard attention.
        - absorb=True  (inference): Absorb W_UK into Q, attention on c_kv directly.
          No pre-extraction needed — wkv_b.weight is reshaped via zero-copy .view() each call.

        Weight absorption math (absorb=True):
            score = (Q_nope @ W_UK) @ c_kv^T + Q_pe @ K_pe^T
            output = softmax(score) @ c_kv @ W_UV

        Why NOT pre-fuse W_UK into W_Q or W_UV into W_O (from DeepSeek GitHub #848):
        1. FP8: wkv_b is stored in FP8, dequantized on-the-fly. Pre-fusing creates
           huge matrices (q_lora_rank × kv_lora_rank × n_heads) harder to quantize.
        2. Tensor parallelism: wkv_b is ColumnParallelLinear, already head-sharded.
           Pre-fusing would require cross-GPU communication.
        3. Memory: separate einsum reuses wkv_b for both K-absorption and V-absorption.

        KV Cache format: (c_kv, k_pe_rotated)
            c_kv:          [B, cached_len, kv_lora_rank]   — compressed latent (post-norm)
            k_pe_rotated:  [B, cached_len, 1, rope_dim]    — RoPE-applied key component

        Args:
            hidden_states: [B, seq_len, hidden_size]
            attention_mask: [B, 1, seq_len, kv_len] additive mask (0 = attend, -inf = mask)
            position_ids: [B, seq_len] explicit position indices
            past_key_value: cached (c_kv, k_pe_rotated) from previous steps
            use_cache: whether to return updated cache

        Returns:
            output: [B, seq_len, hidden_size]
            present_key_value: updated cache tuple or None
        """
        batch_size, seq_len, _ = hidden_states.shape

        # =================================================================
        # QUERY PATH (shared between both modes)
        # =================================================================
        q = self.wq_b(self.q_norm(self.wq_a(hidden_states)))  # [B, S, n_heads * qk_head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # =================================================================
        # KV PATH (shared between both modes)
        # =================================================================
        kv = self.wkv_a(hidden_states)  # [B, S, kv_lora_rank + qk_rope_head_dim]
        c_kv, k_pe_raw = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        c_kv = self.kv_norm(c_kv)
        k_pe_raw = k_pe_raw.unsqueeze(2)  # [B, S, 1, rope_dim]

        # =================================================================
        # ROPE + CACHE (shared between both modes)
        # =================================================================
        if past_key_value is not None:
            cached_c_kv, cached_k_pe = past_key_value
            cached_len = cached_c_kv.shape[1]
        else:
            cached_len = 0

        freqs = self._get_freqs(position_ids, cached_len, seq_len)
        q_pe = apply_rotary_emb(q_pe, freqs)
        k_pe_current = apply_rotary_emb(k_pe_raw, freqs)

        if past_key_value is not None:
            c_kv = torch.cat([cached_c_kv, c_kv], dim=1)
            k_pe = torch.cat([cached_k_pe, k_pe_current], dim=1)
        else:
            k_pe = k_pe_current

        kv_len = c_kv.shape[1]
        present_key_value = (c_kv, k_pe) if use_cache else None
        # DeepSeek V3: mscale applied twice (squared) per official inference code
        effective_scale = self.base_scale * self.mscale * self.mscale

        if self.absorb:
            # =============================================================
            # ABSORB MODE (DeepSeek production inference pattern)
            # Zero-copy view of wkv_b every call — no pre-extraction needed.
            # =============================================================
            # wkv_b.weight: [n_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
            # Reshape to per-head view (zero-copy):
            wkv_b = self.wkv_b.weight.view(
                self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
            )

            # Step 1: Absorb W_UK into Q at runtime
            # q_nope: [B, S, heads, qk_nope_head_dim]
            # W_UK:   [heads, qk_nope_head_dim, kv_lora_rank]
            # result: [B, S, heads, kv_lora_rank]
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim, :])

            # Step 2: Decomposed attention scores (nope + rope)
            # Nope: q_nope @ c_kv^T -> [B, heads, S, kv_len]
            nope_scores = torch.einsum("bshc,btc->bsht", q_nope, c_kv)
            # Rope: q_pe @ k_pe^T -> [B, heads, S, kv_len]
            # k_pe: [B, kv_len, 1, rope_dim] -> squeeze head dim for einsum
            rope_scores = torch.einsum("bshr,btr->bsht", q_pe, k_pe.squeeze(2))

            attn_weights = (nope_scores + rope_scores) * effective_scale

            if attention_mask is not None:
                # attention_mask: [B, 1, S, kv_len], scores: [B, S, heads, kv_len]
                # Transpose mask to match: [B, S, 1, kv_len] or just permute scores
                attn_weights = attn_weights.permute(0, 2, 1, 3)  # [B, heads, S, kv_len]
                attn_weights = attn_weights + attention_mask
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
                attn_weights = attn_weights.permute(0, 2, 1, 3)  # [B, S, heads, kv_len]
            else:
                # Build causal mask for prefill (seq_len > 1)
                if seq_len > 1:
                    causal = torch.full(
                        (seq_len, kv_len), float("-inf"), device=q.device, dtype=q.dtype
                    )
                    causal = causal.triu_(diagonal=kv_len - seq_len + 1)
                    # causal: [S, kv_len] -> [1, S, 1, kv_len] for broadcast
                    attn_weights = attn_weights + causal.unsqueeze(0).unsqueeze(2)

                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(hidden_states.dtype)

            # Step 3: Output through compressed path
            # attn @ c_kv -> [B, S, heads, kv_lora_rank]
            attn_output = torch.einsum("bsht,btc->bshc", attn_weights, c_kv)
            # Apply W_UV per head: [B, S, heads, kv_lora_rank] @ [heads, v_head_dim, kv_lora_rank]^T
            # -> [B, S, heads, v_head_dim]
            # DeepSeek einsum pattern: no transpose needed, einsum handles it
            attn_output = torch.einsum("bshc,hdc->bshd", attn_output, wkv_b[:, -self.v_head_dim:, :])

            # Output projection
            attn_output = attn_output.contiguous().view(batch_size, seq_len, self.num_heads * self.v_head_dim)
            output = self.wo(attn_output)

        else:
            # =============================================================
            # NAIVE MODE (training path — explicit K/V expansion via wkv_b)
            # Supports SDPA / FlashAttention for fused kernels.
            # =============================================================
            kv_expanded = self.wkv_b(c_kv)  # [B, kv_len, n_heads * (qk_nope + v)]
            kv_expanded = kv_expanded.view(
                batch_size, kv_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            k_nope, v = kv_expanded.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            # Assemble full Q and K
            q = torch.cat([q_nope, q_pe], dim=-1)
            k_pe_expanded = k_pe.expand(-1, -1, self.num_heads, -1)
            k = torch.cat([k_nope, k_pe_expanded], dim=-1)

            # Transpose to [B, heads, seq, dim]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Build causal mask when no explicit mask provided and seq_len > 1
            if attention_mask is None and seq_len > 1:
                attention_mask = torch.full(
                    (seq_len, kv_len), float("-inf"), device=q.device, dtype=q.dtype
                )
                attention_mask = attention_mask.triu_(diagonal=kv_len - seq_len + 1)
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

            # Use SDPA when available (FlashAttention / memory-efficient kernels)
            # SDPA natively supports dropout via dropout_p parameter
            use_sdpa = hasattr(F, "scaled_dot_product_attention")

            if use_sdpa:
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attention_mask,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    scale=effective_scale,
                )
            else:
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) * effective_scale
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(v.dtype)
                if self.training and self.attention_dropout > 0:
                    attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=True)
                attn_output = torch.matmul(attn_weights, v)

            # Output projection
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.v_head_dim)
            output = self.wo(attn_output)

        return output, present_key_value


# =============================================================================
# SECTION 5: MIXTURE OF EXPERTS (MOE)
# =============================================================================
# DeepSeek V3 MoE: sigmoid scoring, group-based routing, auxiliary-loss-free
# bias balancing, shared expert (single MLP with combined inter_dim).
# Reference: DeepSeek-V3 Technical Report Section 3.2


class Expert(nn.Module):
    """SwiGLU FFN expert.

    Used for both routed experts and the shared expert (with different inter_dim).
    SwiGLU: output = W_down(SiLU(W_gate(x)) * W_up(x))
    """

    def __init__(self, hidden_dim: int, inter_dim: int):
        super().__init__()
        self.w_gate = CastLinear(hidden_dim, inter_dim, bias=False)
        self.w_up = CastLinear(hidden_dim, inter_dim, bias=False)
        self.w_down = CastLinear(inter_dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # x: [N_tokens, D] → [N_tokens, D]
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Gate(nn.Module):
    """MoE Router with DeepSeek V3 auxiliary-loss-free load balancing.

    Key design (from DeepSeek V3 official code):
    1. Sigmoid scoring — independent expert probabilities (not competitive softmax)
    2. Group-based routing — top-2-sum per group, select topk_group groups
    3. Bias for selection, original scores for weights — bias steers without corrupting gradients
    4. FP32 for all gating computations — numerical stability
    5. norm_topk_prob — normalize selected weights to sum to 1 (BUG FIX #2)
    6. routed_scaling_factor applied to weights after normalization
    """

    def __init__(
        self,
        hidden_dim: int,
        n_routed_experts: int,
        num_experts_per_tok: int,
        n_group: int,
        topk_group: int,
        scoring_func: str = "sigmoid",
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 2.5,
        seq_aux_loss_alpha: float = 0.0001,
    ):
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.experts_per_group = n_routed_experts // n_group
        self.scoring_func = scoring_func
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.seq_aux_loss_alpha = seq_aux_loss_alpha

        # Router projection: hidden_dim → n_routed_experts
        self.router_weight = CastLinear(hidden_dim, n_routed_experts, bias=False)

        # Dynamic bias for auxiliary-loss-free load balancing (NOT a parameter)
        self.register_buffer("bias", torch.zeros(n_routed_experts))

    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        """Route tokens to experts.

        Args:
            hidden_states: [N, D] (pre-flattened by MoE, N = B*S)

        Returns:
            weights:  [N, K] — routing weights (scaled, normalized original scores)
            indices:  [N, K] — selected expert indices
            aux_loss: scalar — sequence-level auxiliary loss
            metadata: dict with 'load_counts' [E] and 'H_load' scalar
        """
        N = hidden_states.shape[0]
        E = self.n_routed_experts
        K = self.num_experts_per_tok
        G = self.n_group
        EPG = self.experts_per_group
        dtype = hidden_states.dtype
        device = hidden_states.device

        # STEP 1: Raw scores in FP32
        logits = self.router_weight(hidden_states)  # [N, E]
        if self.scoring_func == "sigmoid":
            scores = torch.sigmoid(logits).float()  # [N, E] FP32
        else:
            scores = F.softmax(logits, dim=-1).float()  # [N, E] FP32

        # STEP 2: Group-based routing
        # Score each group by sum of its top-2 expert scores (DeepSeek V3 official pattern)
        scores_grouped = scores.view(N, G, EPG)  # [N, G, EPG]
        group_top2, _ = scores_grouped.topk(2, dim=-1)  # [N, G, 2]
        group_scores = group_top2.sum(dim=-1)  # [N, G]
        _, top_groups = group_scores.topk(self.topk_group, dim=-1)  # [N, topk_group]

        # Build expert mask from selected groups
        group_mask = torch.zeros(N, G, device=device, dtype=scores.dtype)
        group_mask.scatter_(1, top_groups, 1.0)  # [N, G]
        expert_mask = group_mask.unsqueeze(-1).expand(-1, -1, EPG).reshape(N, E)  # [N, E]

        # STEP 3: Biased selection (bias NOT in gradient graph)
        biased = scores + self.bias.unsqueeze(0)  # [N, E]
        biased = biased.masked_fill(expert_mask == 0, float("-inf"))

        # STEP 4: Top-K on biased scores → get indices
        _, indices = biased.topk(K, dim=-1)  # [N, K]

        # STEP 5: Gather ORIGINAL scores for weights (not biased)
        weights = scores.gather(1, indices)  # [N, K]

        # STEP 6: BUG FIX #2 — normalize top-k probabilities
        if self.norm_topk_prob:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)

        # STEP 7: Apply routed_scaling_factor (DeepSeek V3 pattern: scale weights, not output)
        weights = weights * self.routed_scaling_factor

        # STEP 8: Compute auxiliary loss and metadata
        one_hot = torch.zeros(N, E, device=device, dtype=scores.dtype)
        one_hot.scatter_(1, indices, 1.0)
        load_counts = one_hot.sum(dim=0)  # [E]

        # Sequence-level aux loss: L = alpha * mean((f_i - 1/E)^2)
        f = load_counts / max(N, 1)
        target = 1.0 / E
        aux_loss = self.seq_aux_loss_alpha * ((f - target) ** 2).mean()

        # Load balance entropy for monitoring (H_load → higher = more balanced)
        f_safe = f + 1e-10
        H_load = -(f_safe * f_safe.log()).sum()

        metadata = {
            "load_counts": load_counts.detach(),
            "H_load": H_load.detach(),
        }

        # Cast weights back to input dtype
        weights = weights.to(dtype)

        return weights, indices, aux_loss, metadata

    def update_bias(self, load_counts: Tensor, gamma: float) -> None:
        """Update dynamic bias for load balancing (called by training loop).

        Args:
            load_counts: [E] — token counts per expert from forward pass
            gamma: bias update rate (0.0 if frozen)
        """
        if gamma <= 0:
            return
        mean_load = load_counts.float().mean()
        if mean_load > 0:
            self.bias -= gamma * (load_counts.float() - mean_load) / mean_load


class MoE(nn.Module):
    """Mixture of Experts with DeepSeek V3 architecture.

    Components:
    - Gate (router): sigmoid scoring + group routing + bias balancing
    - Routed experts: 64 SwiGLU FFNs, top-8 selected per token
    - Shared expert: 1 SwiGLU MLP with combined inter_dim (n_shared * moe_inter_dim)
      Applied to ALL tokens unconditionally (captures common patterns)

    Token-centric dispatch: iterate over experts, process each expert's batch.
    """

    def __init__(
        self,
        hidden_dim: int,
        moe_inter_dim: int,
        n_routed_experts: int = 64,
        num_experts_per_tok: int = 8,
        n_shared_experts: int = 2,
        n_group: int = 8,
        topk_group: int = 4,
        scoring_func: str = "sigmoid",
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 2.5,
        seq_aux_loss_alpha: float = 0.0001,
        shared_inter_dim: Optional[int] = None,
    ):
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok

        # Gate (router)
        self.gate = Gate(
            hidden_dim=hidden_dim,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            n_group=n_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=routed_scaling_factor,
            seq_aux_loss_alpha=seq_aux_loss_alpha,
        )

        # Routed experts (sparse, selected per token)
        self.routed_experts = nn.ModuleList(
            [Expert(hidden_dim, moe_inter_dim) for _ in range(n_routed_experts)]
        )

        # Shared expert (dense, always active) — single MLP with combined inter_dim
        # DeepSeek V3 pattern: n_shared_experts is a multiplier, not a count of modules
        effective_shared_dim = shared_inter_dim or (n_shared_experts * moe_inter_dim)
        self.shared_expert = Expert(hidden_dim, effective_shared_dim)

    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Process tokens through routed + shared experts.

        Args:
            hidden_states: [B, S, D]

        Returns:
            output:   [B, S, D]
            aux_data: dict with 'aux_loss', 'load_counts', 'H_load'
        """
        orig_shape = hidden_states.shape  # [B, S, D]
        hidden_dim = orig_shape[-1]
        x = hidden_states.view(-1, hidden_dim)  # [N, D] where N = B*S

        # Route tokens
        weights, indices, aux_loss, metadata = self.gate(x)
        # weights: [N, K], indices: [N, K]

        # Token-centric dispatch for routed experts 
        # Experts are processed in contiguous batches, avoiding scattered memory access
        routed_output = torch.zeros_like(x)  # [N, D]
        for expert_idx in range(self.n_routed_experts):
            # Find tokens assigned to this expert across any of the K slots
            mask = (indices == expert_idx)  # [N, K] bool
            token_mask = mask.any(dim=-1)  # [N] bool

            if not token_mask.any():
                continue  # Skip experts with 0 assigned tokens

            # Process this expert's batch
            expert_input = x[token_mask]  # [n_tokens, D]
            expert_out = self.routed_experts[expert_idx](expert_input)  # [n_tokens, D]

            # Sum weights across all slots that selected this expert per token
            # (handles rare case where same expert selected in multiple slots)
            expert_weights = (weights * mask.float()).sum(dim=-1)  # [N]
            active_weights = expert_weights[token_mask]  # [n_tokens]

            routed_output[token_mask] += expert_out * active_weights.unsqueeze(-1)

        # Shared expert — processes ALL tokens
        shared_output = self.shared_expert(x)  # [N, D]

        # Combine routed + shared
        output = routed_output + shared_output  # [N, D]
        output = output.view(*orig_shape)  # [B, S, D]

        aux_data = {
            "aux_loss": aux_loss,
            "load_counts": metadata["load_counts"],
            "H_load": metadata["H_load"],
        }

        return output, aux_data


# =============================================================================
# SECTION 6: MULTI-TOKEN PREDICTION (MTP)
# =============================================================================
# DeepSeek V3 MTP (Section 3.3): concatenation + linear projection + standard
# transformer block. NOT cross-attention (Bug Fix #1).
# Formula: h'_i^k = M_k [ RMSNorm(Emb(t_{i+k})) ; RMSNorm(h_i^{k-1}) ]
# Reference: DeepSeek-V3 paper Eq. 21, SGLang deepseek_nextn.py


def get_mtp_loss_weight(
    tokens_processed: int,
    total_tokens: int,
    initial_weight: float = 0.3,
    final_weight: float = 0.1,
    transition_ratio: float = 0.60,
) -> float:
    """DeepSeek V3 MTP loss schedule: λ=0.3 for first 60%, then λ=0.1.

    Step function matching DeepSeek V3 (0.3 for 10T tokens, 0.1 for 4.8T).
    NanoSeek transition at 60% of total training tokens (config-driven).

    Args:
        tokens_processed: tokens seen so far.
        total_tokens: total training token budget.
        initial_weight: λ before transition (default 0.3).
        final_weight: λ after transition (default 0.1).
        transition_ratio: fraction of training at which to switch (default 0.60).

    Returns:
        Current MTP loss weight λ.
    """
    progress = tokens_processed / max(total_tokens, 1)
    return initial_weight if progress < transition_ratio else final_weight


class MTPTransformerBlock(nn.Module):
    """Standard pre-norm transformer block for MTP modules.

    Bug Fix #1: This replaces the old cross-attention MTPBlock.
    DeepSeek V3 MTP uses a standard transformer block (self-attention + FFN),
    NOT cross-attention. Reuses MLA for attention and Expert (SwiGLU) for FFN.

    Architecture:
        x → RMSNorm → MLA (self-attention) → residual → RMSNorm → SwiGLU FFN → residual
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
        intermediate_size: int,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        rope_scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        mscale: float = 1.0,
        rms_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.input_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.attn = MultiHeadLatentAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling_factor=rope_scaling_factor,
            original_max_position_embeddings=original_max_position_embeddings,
            mscale=mscale,
            attention_dropout=attention_dropout,
            layer_idx=0,
        )
        self.post_attn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        # Dense SwiGLU FFN — MTP is lightweight, no MoE routing needed
        self.ffn = Expert(hidden_size, intermediate_size)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: [B, S, H] input hidden states.
            attention_mask: [B, 1, S, kv_len] additive mask or None.

        Returns:
            [B, S, H] output hidden states.
        """
        # Pre-norm residual attention
        h = x + self.attn(self.input_norm(x), attention_mask=attention_mask)[0]
        # Pre-norm residual FFN
        return h + self.ffn(self.post_attn_norm(h))


class MTPModule(nn.Module):
    """One MTP prediction module at depth k.

    Implements DeepSeek V3 MTP (Section 3.3, Eq. 21):
        h'_i^k = M_k [ RMSNorm(Emb(t_{i+k})) ; RMSNorm(h_i^{k-1}) ]
    followed by a standard transformer block and shared LM head.

    Shared weights (set via set_shared_embeddings):
        - embed_tokens: shared with main model embedding table
        - lm_head: shared with main model output projection

    Concatenation order (from SGLang reference): embedding FIRST, hidden SECOND.
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
        intermediate_size: int,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        rope_scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        mscale: float = 1.0,
        rms_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        module_idx: int = 0,
    ):
        super().__init__()
        self.module_idx = module_idx
        self.vocab_size = None  # Set by set_shared_embeddings or parent
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.embed_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        # M_k: Linear projection [2H → H] (the concatenation fusion)
        self.concat_proj = CastLinear(2 * hidden_size, hidden_size, bias=False)
        self.transformer = MTPTransformerBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling_factor=rope_scaling_factor,
            original_max_position_embeddings=original_max_position_embeddings,
            mscale=mscale,
            rms_norm_eps=rms_norm_eps,
            attention_dropout=attention_dropout,
        )
        self.output_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        # Shared with main model — set via set_shared_embeddings()
        self.embed_tokens: Optional[nn.Embedding] = None
        self.lm_head: Optional[nn.Linear] = None

    def set_shared_embeddings(
        self, embed_tokens: nn.Embedding, lm_head: nn.Linear
    ) -> None:
        """Wire shared weights from main model. References, not copies."""
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head

    def forward(
        self,
        prev_hidden: Tensor,
        target_tokens: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            prev_hidden: [B, S', H] hidden states from previous depth or main model.
            target_tokens: [B, S'] token IDs for embedding lookup (teacher forcing).
            attention_mask: [B, 1, S', S'] additive mask or None (MLA creates causal).

        Returns:
            logits: [B, S', V] prediction logits.
            h: [B, S', H] hidden states (passed to next MTP module in chain).
        """
        assert self.embed_tokens is not None, "Call set_shared_embeddings() first"
        # SGLang order: embedding FIRST, hidden SECOND
        t_emb = self.embed_norm(self.embed_tokens(target_tokens))  # [B, S', H]
        h_norm = self.hidden_norm(prev_hidden)  # [B, S', H]
        h = self.concat_proj(torch.cat([t_emb, h_norm], dim=-1))  # [B, S', 2H] → [B, S', H]
        h = self.transformer(h, attention_mask)  # [B, S', H]
        logits = self.lm_head(self.output_norm(h))  # [B, S', padded_V]
        # Slice off vocab padding (lm_head uses padded_vocab_size for matmul alignment)
        if self.vocab_size is not None:
            logits = logits[..., :self.vocab_size]
        return logits, h


class MultiTokenPrediction(nn.Module):
    """Container for D MTP modules with sequential chaining and loss computation.

    DeepSeek V3 uses D=1 (one extra module predicting token at offset +2).
    Modules are chained sequentially: module k's hidden state feeds module k+1.
    Each module has its own distinct transformer block and projection (not shared).
    Only embed_tokens and lm_head are shared with the main model.

    Token shifting for module k (0-indexed):
        embed_tokens  = input_ids[:, k+1 : L-(k+1)]   tokens for embedding
        target_labels = input_ids[:, k+2 : L-k]        prediction targets
        curr_hidden   = prev_hidden[:, :S', :]          aligned hidden states
    """

    def __init__(
        self,
        num_modules: int,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        intermediate_size: int,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        rope_scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        mscale: float = 1.0,
        rms_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        loss_decay: float = 0.8,
    ):
        super().__init__()
        self.num_modules = num_modules
        self.loss_decay = loss_decay
        self.mtp_modules = nn.ModuleList([
            MTPModule(
                hidden_size=hidden_size,
                num_heads=num_heads,
                q_lora_rank=q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                intermediate_size=intermediate_size,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                rope_scaling_factor=rope_scaling_factor,
                original_max_position_embeddings=original_max_position_embeddings,
                mscale=mscale,
                rms_norm_eps=rms_norm_eps,
                attention_dropout=attention_dropout,
                module_idx=k,
            )
            for k in range(num_modules)
        ])

    def set_shared_embeddings(
        self, embed_tokens: nn.Embedding, lm_head: nn.Linear
    ) -> None:
        """Wire shared embeddings to all MTP modules."""
        for module in self.mtp_modules:
            module.set_shared_embeddings(embed_tokens, lm_head)

    def forward(
        self,
        main_hidden: Tensor,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[List[Tensor], Tensor]:
        """
        Args:
            main_hidden: [B, L, H] hidden states from main model's last decoder layer.
            input_ids: [B, L] full input token sequence.
            attention_mask: unused — MLA creates its own causal mask internally.

        Returns:
            all_logits: list of [B, S'_k, V] logit tensors, one per module.
            total_loss: scalar MTP loss (averaged over modules, weighted by loss_decay).
        """
        L = input_ids.size(1)
        prev_hidden = main_hidden
        all_logits: List[Tensor] = []
        total_loss = torch.tensor(0.0, device=main_hidden.device, dtype=main_hidden.dtype)

        for k, module in enumerate(self.mtp_modules):
            # Token shifting: module k predicts token at offset k+2
            embed_offset = k + 1
            target_offset = k + 2
            if target_offset >= L:
                break  # Not enough tokens for this module

            # Aligned slices — all have length L - (k+2)
            embed_tokens = input_ids[:, embed_offset : L - (k + 1)]  # [B, S']
            target_labels = input_ids[:, target_offset : L - k]  # [B, S']
            S_prime = embed_tokens.size(1)
            if S_prime == 0:
                break
            curr_hidden = prev_hidden[:, :S_prime, :]  # [B, S', H]

            # Forward through MTP module (mask=None → MLA creates causal internally)
            logits_k, h_k = module(curr_hidden, embed_tokens)
            # logits_k: [B, S', V]    h_k: [B, S', H]

            # Cross-entropy loss for this module's predictions
            loss_k = F.cross_entropy(
                logits_k.reshape(-1, logits_k.size(-1)),
                target_labels[:, :S_prime].reshape(-1),
                ignore_index=-100,
            )

            # Weight by decay factor (module 0 = 1.0, module 1 = 0.8, ...)
            total_loss = total_loss + (self.loss_decay ** k) * loss_k
            all_logits.append(logits_k)
            prev_hidden = h_k  # Chain to next module

        # Average over modules
        if self.num_modules > 0:
            total_loss = total_loss / self.num_modules

        return all_logits, total_loss

# =============================================================================
# DECODER LAYER
# =============================================================================
# Standard pre-norm transformer decoder layer following DeepSeek V3 architecture.
# Wires together: RMSNorm + MLA (or DSA) + dense FFN (Expert) or MoE.

class NanoSeekDecoderLayer(nn.Module):
    """Single decoder layer: pre-norm attention + pre-norm FFN with residual connections.

    Architecture:
        x → input_layernorm → self_attn → + residual
          → post_attention_layernorm → ffn → + residual → output

    The attention module is MLA (Multi-head Latent Attention). DSA (DeepSeek Sparse
    Attention) support will be added when Section 9 is implemented.

    The FFN module is either:
        - Dense SwiGLU (Expert class) for early layers (layer_idx < first_k_dense_replace)
        - MoE for later layers (layer_idx >= first_k_dense_replace)
    """

    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        num_heads: int,
        # MLA parameters
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        rope_scaling_factor: float,
        original_max_position_embeddings: int,
        mscale: float,
        attention_dropout: float,
        rms_norm_eps: float,
        # FFN parameters — exactly one of these will be used
        # Dense FFN (for early layers)
        intermediate_size: int = 5243,
        # MoE (for later layers)
        use_moe: bool = False,
        n_routed_experts: int = 64,
        n_shared_experts: int = 2,
        num_experts_per_tok: int = 8,
        moe_intermediate_size: int = 768,
        shared_inter_dim: Optional[int] = None,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 2.5,
        n_group: int = 4,
        topk_group: int = 2,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_moe = use_moe

        # Pre-attention norm
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Attention — MLA for now, DSA added later (Section 9)
        self.self_attn = MultiHeadLatentAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling_factor=rope_scaling_factor,
            original_max_position_embeddings=original_max_position_embeddings,
            mscale=mscale,
            attention_dropout=attention_dropout,
            layer_idx=layer_idx,
        )

        # Post-attention norm
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # FFN — dense or MoE based on layer position
        if use_moe:
            self.ffn = MoE(
                hidden_dim=hidden_size,
                moe_inter_dim=moe_intermediate_size,
                n_routed_experts=n_routed_experts,
                num_experts_per_tok=num_experts_per_tok,
                n_shared_experts=n_shared_experts,
                n_group=n_group,
                topk_group=topk_group,
                norm_topk_prob=norm_topk_prob,
                routed_scaling_factor=routed_scaling_factor,
                shared_inter_dim=shared_inter_dim,
            )
        else:
            self.ffn = Expert(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]], Dict[str, Tensor]]:
        """
        Forward pass for a single decoder layer.

        Args:
            hidden_states: [B, S, H] input tensor
            attention_mask: Optional attention mask
            position_ids: Optional position indices for RoPE
            past_key_value: Optional KV cache from previous step
            use_cache: Whether to return updated KV cache

        Returns:
            hidden_states: [B, S, H] output tensor
            present_key_value: Updated KV cache (None if use_cache=False)
            aux_data: Dict with MoE routing metrics (empty for dense layers)
        """
        # ---- Attention block (pre-norm + residual) ----
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, present_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_output

        # ---- FFN block (pre-norm + residual) ----
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.use_moe:
            # MoE returns (output, aux_data_dict)
            ffn_output, aux_data = self.ffn(hidden_states)
        else:
            # Dense FFN returns just the output tensor
            ffn_output = self.ffn(hidden_states)
            aux_data = {}

        hidden_states = residual + ffn_output

        return hidden_states, present_key_value, aux_data


# =============================================================================
# NANOSEEK MODEL
# =============================================================================

class NanoSeekModel(nn.Module):
    """Complete NanoSeek language model.

    Architecture:
        embed_tokens [V, H]
        → 16 DecoderLayers (2 dense + 14 MoE)
        → RMSNorm
        → lm_head [H, V]
        + MTP auxiliary prediction head (training only)

    Training loss:
        L_total = L_main + λ(step) · L_MTP + L_aux
        where λ = 0.3 for first 60% of training, 0.1 after
        and L_aux = averaged MoE load-balancing loss across layers
    """

    def __init__(self, config: NanoSeekConfig):
        super().__init__()
        self.config = config
        self.logit_softcap = config.logit_softcap

        # Pad vocab to multiple of 64 for matmul alignment (nanochat pattern).
        # 32768 is already aligned, but this guards against special token additions.
        pad_to = 64
        self.padded_vocab_size = ((config.vocab_size + pad_to - 1) // pad_to) * pad_to
        self.embed_tokens = nn.Embedding(self.padded_vocab_size, config.hidden_size)
        # ---- Decoder layers ----
        # Layers 0..first_k_dense_replace-1: dense SwiGLU FFN
        # Layers first_k_dense_replace..num_layers-1: MoE
        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            use_moe = (i >= config.moe.first_k_dense_replace)
            self.layers.append(NanoSeekDecoderLayer(
                layer_idx=i,
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                # MLA parameters
                q_lora_rank=config.mla.q_lora_rank,
                kv_lora_rank=config.mla.kv_lora_rank,
                qk_nope_head_dim=config.mla.qk_nope_head_dim,
                qk_rope_head_dim=config.mla.qk_rope_head_dim,
                v_head_dim=config.mla.v_head_dim,
                max_position_embeddings=config.max_position_embeddings,
                rope_theta=config.mla.rope_theta,
                rope_scaling_factor=config.mla.rope_scaling_factor,
                original_max_position_embeddings=config.mla.original_max_position_embeddings,
                mscale=config.mla.mscale,
                attention_dropout=config.attention_dropout,
                rms_norm_eps=config.rms_norm_eps,
                # Dense FFN
                intermediate_size=config.intermediate_size,
                # MoE (only used when use_moe=True)
                use_moe=use_moe,
                n_routed_experts=config.moe.n_routed_experts,
                n_shared_experts=config.moe.n_shared_experts,
                num_experts_per_tok=config.moe.num_experts_per_tok,
                moe_intermediate_size=config.moe.moe_intermediate_size,
                shared_inter_dim=config.moe.shared_inter_dim,
                norm_topk_prob=config.moe.norm_topk_prob,
                routed_scaling_factor=config.moe.routed_scaling_factor,
                n_group=config.moe.n_group,
                topk_group=config.moe.topk_group,
            ))

        # ---- Final norm + LM head ----
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = CastLinear(config.hidden_size, self.padded_vocab_size, bias=False)

        # ---- MTP (Multi-Token Prediction) ----
        self.mtp = None
        if config.mtp.num_mtp_modules > 0:
            self.mtp = MultiTokenPrediction(
                num_modules=config.mtp.num_mtp_modules,
                hidden_size=config.hidden_size,
                num_heads=config.mtp.mtp_num_heads,
                q_lora_rank=config.mla.q_lora_rank,
                kv_lora_rank=config.mla.kv_lora_rank,
                qk_nope_head_dim=config.mla.qk_nope_head_dim,
                qk_rope_head_dim=config.mla.qk_rope_head_dim,
                v_head_dim=config.mla.v_head_dim,
                intermediate_size=config.intermediate_size,
                max_position_embeddings=config.max_position_embeddings,
                rope_theta=config.mla.rope_theta,
                rope_scaling_factor=config.mla.rope_scaling_factor,
                original_max_position_embeddings=config.mla.original_max_position_embeddings,
                mscale=config.mla.mscale,
                rms_norm_eps=config.rms_norm_eps,
                attention_dropout=config.attention_dropout,
                loss_decay=config.mtp.mtp_loss_decay,
            )
            self.mtp.set_shared_embeddings(self.embed_tokens, self.lm_head)
            # Pass real vocab_size so MTP slices off padding from lm_head output
            for mtp_mod in self.mtp.mtp_modules:
                mtp_mod.vocab_size = config.vocab_size

        # ---- State for load-balance bias updates ----
        self._layer_aux_data: Dict[int, Dict[str, Tensor]] = {}

        # ---- Gradient checkpointing ----
        self.gradient_checkpointing = config.gradient_checkpointing

        # ---- Initialize weights ----
        self.init_weights()

    def init_weights(self) -> None:
        """Public API for weight initialization.

        Called explicitly after to_empty(device=...) in pre_train.py,
        and implicitly during __init__().
        """
        self._init_weights()

    def _init_weights(self) -> None:
        """Width-dependent weight initialization with zero-init output projections.

        Linear layers: N(0, 1/√hidden_size) — variance-preserving (GPT-3/nanochat).
        Embeddings: N(0, 1/√hidden_size) — same scale for consistent activation norms.
        Output projections (wo, w_down, concat_proj): zero-init so each block starts
        as identity, critical for MoE where random routing + random projections = chaos.
        """
        hidden_size = self.config.hidden_size
        std = 1.0 / (hidden_size ** 0.5)  # Variance-preserving: Var(out) ≈ Var(in)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)

        # Zero-init output projections: each block is identity at init
        # wo (attention output), w_down (FFN/expert output), concat_proj (MTP fusion)
        for layer in self.layers:
            # Attention output projection
            torch.nn.init.zeros_(layer.self_attn.wo.weight)
            # Dense FFN layer (first_k_dense layers)
            if isinstance(layer.ffn, Expert):
                torch.nn.init.zeros_(layer.ffn.w_down.weight)
            # MoE layer: zero-init all routed + shared expert w_down
            elif isinstance(layer.ffn, MoE):
                for expert in layer.ffn.routed_experts:
                    torch.nn.init.zeros_(expert.w_down.weight)
                torch.nn.init.zeros_(layer.ffn.shared_expert.w_down.weight)
        # MTP concat_proj zero-init
        if self.mtp is not None:
            for mtp_mod in self.mtp.mtp_modules:
                torch.nn.init.zeros_(mtp_mod.concat_proj.weight)

    # -----------------------------------------------------------------
    # Forward Pass
    # -----------------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
        labels: Optional[Tensor] = None,
        tokens_processed: int = 0,
        total_tokens: int = 1,
    ) -> Dict[str, Tensor]:
        """Complete forward pass.

        Args:
            input_ids: [B, S] token IDs
            attention_mask: unused — MLA creates causal mask internally
            position_ids: [B, S] position indices (auto-computed if None)
            past_key_values: list of (c_kv, k_pe) tuples per layer for KV cache
            use_cache: whether to return updated KV cache
            labels: [B, S] target token IDs for loss computation
            tokens_processed: current token count (for MTP/gamma schedule)
            total_tokens: total training tokens (for MTP/gamma schedule)

        Returns:
            dict with keys:
                loss:            scalar total loss (None if labels not provided)
                logits:          [B, S, V] logit predictions
                past_key_values: list of KV caches (None if use_cache=False)
                aux_loss:        scalar averaged MoE load-balancing loss
        """
        B, S = input_ids.shape

        # 1. Token embedding + post-embedding norm
        hidden_states = self.embed_tokens(input_ids)  # [B, S, H]
        # Parameterless RMSNorm: stabilize activation scale entering layer 0
        # Without this, embedding magnitude varies with vocab distribution
        hidden_states = F.rms_norm(hidden_states, (hidden_states.size(-1),))

        # 2. Position IDs — only auto-compute for cached generation.
        # When past_key_values is None, leave position_ids=None so MLA's
        # internal fallback path handles it (avoids batched freqs shape issue).
        if position_ids is None and past_key_values is not None:
            past_len = past_key_values[0][0].size(1) if past_key_values[0] is not None else 0
            position_ids = torch.arange(
                past_len, past_len + S,
                device=input_ids.device, dtype=torch.long,
            ).unsqueeze(0).expand(B, -1)

        # 3. Decoder layers
        present_key_values: Optional[List] = [] if use_cache else None
        total_aux_loss = torch.tensor(
            0.0, device=hidden_states.device, dtype=hidden_states.dtype
        )
        n_aux_layers = 0
        self._layer_aux_data = {}  # Reset for this forward pass

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                # use_reentrant=False: required for MoE aux_data to survive checkpointing
                hidden_states, present_kv, aux_data = gradient_checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_kv,
                    use_cache,
                    use_reentrant=False,
                )
            else:
                hidden_states, present_kv, aux_data = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                )

            if use_cache and present_key_values is not None:
                present_key_values.append(present_kv)

            # Accumulate MoE auxiliary losses
            if "aux_loss" in aux_data:
                total_aux_loss = total_aux_loss + aux_data["aux_loss"]
                n_aux_layers += 1
                self._layer_aux_data[i] = aux_data

        # 4. Final norm
        hidden_states = self.norm(hidden_states)  # [B, S, H]

        # 5. LM head + logit softcap
        logits = self.lm_head(hidden_states)  # [B, S, padded_V]
        logits = logits[..., :self.config.vocab_size]  # Slice off padding rows
        if self.logit_softcap > 0.0:
            # Gemma 2 technique: tanh squash to [-cap, cap] in fp32
            # Prevents logit explosion from MoE expert specialization
            logits = logits.float()
            logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
            logits = logits.to(hidden_states.dtype)

        # 6. Loss computation
        loss = None
        main_loss = None
        mtp_loss = None
        mtp_lambda = 0.0
        if labels is not None:
            loss_dict = self._compute_loss(
                logits, labels, hidden_states, input_ids,
                total_aux_loss, n_aux_layers,
                tokens_processed, total_tokens,
            )
            loss = loss_dict["total_loss"]
            main_loss = loss_dict["main_loss"]
            mtp_loss = loss_dict["mtp_loss"]
            mtp_lambda = loss_dict["mtp_lambda"]

        # 7. Average aux loss for reporting
        avg_aux_loss = (
            total_aux_loss / n_aux_layers if n_aux_layers > 0
            else total_aux_loss
        )

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": present_key_values,
            "aux_loss": avg_aux_loss,
            "main_loss": main_loss,
            "mtp_loss": mtp_loss,
            "mtp_lambda": mtp_lambda,
        }

    def _compute_loss(
        self,
        logits: Tensor,
        labels: Tensor,
        hidden_states: Tensor,
        input_ids: Tensor,
        total_aux_loss: Tensor,
        n_aux_layers: int,
        tokens_processed: int,
        total_tokens: int,
    ) -> Dict[str, Tensor]:
        """Compute total training loss with component breakdown.

        L_total = L_main + λ · L_MTP + L_aux

        Args:
            logits: [B, S, V] — model predictions
            labels: [B, S] — target tokens
            hidden_states: [B, S, H] — post-norm hidden states (for MTP)
            input_ids: [B, S] — original input (for MTP token shifting)
            total_aux_loss: accumulated MoE load-balancing loss
            n_aux_layers: number of MoE layers that contributed
            tokens_processed: for MTP loss weight schedule
            total_tokens: for MTP loss weight schedule

        Returns:
            dict with total_loss, main_loss, mtp_loss, mtp_lambda
        """
        V = logits.size(-1)

        # Main cross-entropy loss (shift by 1: position i predicts token i+1)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        main_loss = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        total_loss = main_loss
        mtp_loss = torch.zeros(1, device=logits.device)
        mtp_lambda = 0.0

        # MTP auxiliary loss (training only)
        if self.training and self.mtp is not None:
            _, mtp_loss = self.mtp(hidden_states, input_ids)
            mtp_lambda = get_mtp_loss_weight(
                tokens_processed,
                total_tokens,
                self.config.mtp.mtp_loss_weight_initial,
                self.config.mtp.mtp_loss_weight_final,
                self.config.mtp.mtp_loss_transition_ratio,
            )
            total_loss = total_loss + mtp_lambda * mtp_loss

        # MoE load-balancing auxiliary loss
        if n_aux_layers > 0:
            avg_aux = total_aux_loss / n_aux_layers
            total_loss = total_loss + avg_aux

        return {
            "total_loss": total_loss,
            "main_loss": main_loss,
            "mtp_loss": mtp_loss,
            "mtp_lambda": mtp_lambda,
        }

    # -----------------------------------------------------------------
    # Training Helper Methods
    # -----------------------------------------------------------------

    def get_gamma(self, tokens_processed: int, total_tokens: int) -> float:
        """Dynamic bias update rate for MoE load balancing.

        Freezes at 95% of training.
        # V3 paper spec: 14.3T/14.8T = 96.6%, use 0.95 as conservative approximation
        """
        progress = tokens_processed / max(total_tokens, 1)
        if progress >= self.config.moe.gamma_freeze_ratio:
            return 0.0
        return self.config.moe.gamma

    def update_load_balance_bias(
        self, tokens_processed: int, total_tokens: int
    ) -> None:
        """Update expert routing biases for load balancing.

        Called by training loop after each optimizer step.
        Uses load_counts cached from the most recent forward pass.
        Formula: b_i -= gamma * (load_i - mean_load) / mean_load
        """
        gamma = self.get_gamma(tokens_processed, total_tokens)
        if gamma <= 0:
            return
        for layer_idx, aux_data in self._layer_aux_data.items():
            if "load_counts" in aux_data:
                self.layers[layer_idx].ffn.gate.update_bias(
                    aux_data["load_counts"], gamma
                )

    def num_parameters(self) -> Dict[str, int]:
        """Return active and total parameter counts.

        Active params exclude inactive routed experts (64 - 8 = 56 per MoE layer).
        """
        total = sum(p.numel() for p in self.parameters())

        # Inactive routed experts per MoE layer
        inactive_per_layer = (
            self.config.moe.n_routed_experts - self.config.moe.num_experts_per_tok
        )
        # Each routed expert: 3 linear layers (gate, up, down) in SwiGLU
        params_per_expert = (
            3 * self.config.hidden_size * self.config.moe.moe_intermediate_size
        )
        n_moe_layers = (
            self.config.num_layers - self.config.moe.first_k_dense_replace
        )
        inactive_params = inactive_per_layer * params_per_expert * n_moe_layers

        active = total - inactive_params
        return {"active": active, "total": total}

    def get_expert_load_stats(self) -> Dict[str, Tensor]:
        """Aggregate expert load statistics across all MoE layers.

        Returns:
            entropy: mean H_load across MoE layers (higher = more balanced)
            load_per_expert: [E] mean token counts per expert
        """
        all_H = []
        all_counts = []
        for aux_data in self._layer_aux_data.values():
            if "H_load" in aux_data:
                all_H.append(aux_data["H_load"])
            if "load_counts" in aux_data:
                all_counts.append(aux_data["load_counts"])

        if not all_H:
            n_experts = self.config.moe.n_routed_experts
            return {
                "entropy": torch.tensor(0.0),
                "load_per_expert": torch.zeros(n_experts),
            }

        return {
            "entropy": torch.stack(all_H).mean(),
            "load_per_expert": torch.stack(all_counts).float().mean(dim=0),
        }

    # -----------------------------------------------------------------
    # Generation Methods (test/debug only — production uses Engine)
    # -----------------------------------------------------------------

    def _sample(
        self, logits: Tensor, temperature: float, top_p: float
    ) -> Tensor:
        """Sample next token with temperature scaling and nucleus (top-p) sampling.

        Args:
            logits: [B, V] unnormalized logit scores for next token
            temperature: scaling factor (0 = greedy argmax)
            top_p: cumulative probability threshold for nucleus sampling

        Returns:
            [B] sampled token IDs
        """
        if temperature <= 0:
            return logits.argmax(dim=-1)

        logits = logits / temperature

        # Sort and compute cumulative probabilities
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        # Mask tokens beyond top_p threshold
        # Keep the first token that crosses the threshold (subtract its own prob)
        sorted_mask = (cumulative_probs - probs) >= top_p
        sorted_logits[sorted_mask] = float("-inf")

        # Sample from filtered distribution
        filtered_probs = F.softmax(sorted_logits, dim=-1)
        sampled_idx = torch.multinomial(filtered_probs, num_samples=1)  # [B, 1]

        # Map back to original vocab indices
        next_tokens = sorted_indices.gather(-1, sampled_idx).squeeze(-1)  # [B]
        return next_tokens

    @torch.no_grad()
    def generate_simple(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.95,
    ) -> Tensor:
        """Auto-regressive generation WITHOUT KV cache.

        Re-processes entire sequence at each step. Slow but simple.
        Use for testing/debugging only.

        Args:
            prompt_ids: [1, S] prompt token IDs
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (0 = greedy)
            top_p: nucleus sampling threshold

        Returns:
            [1, S + max_new_tokens] full sequence including prompt
        """
        generated = prompt_ids.clone()

        for _ in range(max_new_tokens):
            outputs = self.forward(generated, use_cache=False)
            next_logits = outputs["logits"][:, -1, :]  # [1, V]
            next_token = self._sample(next_logits, temperature, top_p)
            generated = torch.cat(
                [generated, next_token.unsqueeze(-1)], dim=-1
            )

        return generated

    @torch.no_grad()
    def generate_cached(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.95,
    ) -> Tensor:
        """Auto-regressive generation WITH KV cache.

        Prefills the full prompt, then decodes one token at a time using
        cached key-value pairs. Much faster than generate_simple().

        Args:
            prompt_ids: [1, S] prompt token IDs
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (0 = greedy)
            top_p: nucleus sampling threshold

        Returns:
            [1, S + max_new_tokens] full sequence including prompt
        """
        # Prefill: process entire prompt, build initial KV cache
        outputs = self.forward(prompt_ids, use_cache=True)
        past_kvs = outputs["past_key_values"]
        next_logits = outputs["logits"][:, -1, :]  # [1, V]
        next_token = self._sample(next_logits, temperature, top_p)
        generated = [next_token]

        # Decode: one token at a time with KV cache
        for _ in range(max_new_tokens - 1):
            outputs = self.forward(
                next_token.unsqueeze(-1),  # [1, 1]
                past_key_values=past_kvs,
                use_cache=True,
            )
            past_kvs = outputs["past_key_values"]
            next_logits = outputs["logits"][:, -1, :]
            next_token = self._sample(next_logits, temperature, top_p)
            generated.append(next_token)

        # Concatenate prompt + generated tokens
        generated_ids = torch.stack(generated, dim=-1)  # [1, max_new_tokens]
        return torch.cat([prompt_ids, generated_ids], dim=-1)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.95,
        use_cache: bool = True,
    ) -> Tensor:
        """Standard generation interface.

        Args:
            input_ids: [B, S] prompt token IDs (B=1 for generation)
            max_new_tokens: tokens to generate
            temperature: sampling temperature
            top_p: nucleus sampling threshold
            use_cache: use KV cache (True = fast, False = simple)

        Returns:
            [B, S + max_new_tokens] generated sequence
        """
        if use_cache:
            return self.generate_cached(
                input_ids, max_new_tokens, temperature, top_p
            )
        return self.generate_simple(
            input_ids, max_new_tokens, temperature, top_p
        )


# =============================================================================
# SECTION 12: FACTORY FUNCTIONS AND TESTS
# =============================================================================


def create_nanoseek(config: Optional[NanoSeekConfig] = None) -> NanoSeekModel:
    """Create a NanoSeekModel instance.

    Args:
        config: NanoSeekConfig (uses default 1B config if None)

    Returns:
        Initialized NanoSeekModel
    """
    if config is None:
        config = get_nanoseek_config()
    return NanoSeekModel(config)


def test_nanoseek() -> None:
    """Comprehensive smoke test for the complete NanoSeek model.

    Tests:
        1. Forward pass produces correct shapes and finite loss
        2. Gradient flow reaches all parameters
        3. MoE auxiliary loss is non-zero
        4. KV-cached generation produces valid token sequences
        5. Parameter counts (active vs total)
        6. MTP contributes to total training loss
        7. Load balance bias update runs without error
        8. Gamma schedule: active early, frozen at 95%
    """
    print("=" * 60)
    print("NanoSeek Model — Comprehensive Smoke Test")
    print("=" * 60)

    config = get_nanoseek_config()
    model = create_nanoseek(config)

    B, S, V = 2, 64, config.vocab_size
    input_ids = torch.randint(0, V, (B, S))
    labels = torch.randint(0, V, (B, S))

    # ---- Test 1: Forward pass shapes + finite loss ----
    model.train()
    outputs = model(input_ids, labels=labels, tokens_processed=0, total_tokens=1000)
    assert outputs["logits"].shape == (B, S, V), (
        f"Wrong logits shape: {outputs['logits'].shape}"
    )
    assert outputs["loss"] is not None and outputs["loss"].isfinite(), (
        f"Loss is not finite: {outputs['loss']}"
    )
    print(
        f"  Test 1: Forward pass — logits {outputs['logits'].shape}, "
        f"loss={outputs['loss'].item():.4f}"
    )

    # ---- Test 2: Gradient flow ----
    outputs["loss"].backward()
    n_with_grad = sum(
        1 for p in model.parameters() if p.requires_grad and p.grad is not None
    )
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  Test 2: Gradient flow — {n_with_grad}/{n_trainable} params have gradients")
    model.zero_grad()

    # ---- Test 3: MoE auxiliary loss > 0 ----
    aux = outputs["aux_loss"]
    print(f"  Test 3: Aux loss = {aux.item():.6f} (MoE load-balancing)")

    # ---- Test 4: KV cache generation ----
    model.eval()
    with torch.no_grad():
        prompt = input_ids[:1, :16]  # [1, 16]
        generated = model.generate(prompt, max_new_tokens=8, temperature=1.0)
    expected_len = 16 + 8
    assert generated.shape == (1, expected_len), (
        f"Wrong generation shape: {generated.shape}, expected (1, {expected_len})"
    )
    assert torch.all((generated >= 0) & (generated < V)), "Invalid token IDs"
    print(f"  Test 4: Generation — {generated.shape} (16 prompt + 8 new)")

    # ---- Test 5: Parameter counts ----
    param_info = model.num_parameters()
    print(
        f"  Test 5: Parameters — "
        f"active={param_info['active'] / 1e6:.1f}M, "
        f"total={param_info['total'] / 1e6:.1f}M, "
        f"ratio={param_info['active'] / param_info['total']:.1%}"
    )

    # ---- Test 6: MTP contributes to loss ----
    model.train()
    out_mtp = model(input_ids, labels=labels, tokens_processed=0, total_tokens=1000)
    print(f"  Test 6: Total loss (includes MTP + aux) = {out_mtp['loss'].item():.4f}")
    model.zero_grad()

    # ---- Test 7: Load balance bias update ----
    _ = model(input_ids, labels=labels, tokens_processed=100, total_tokens=1000)
    model.update_load_balance_bias(tokens_processed=100, total_tokens=1000)
    print("  Test 7: Load balance bias update — no crash")
    model.zero_grad()

    # ---- Test 8: Gamma schedule ----
    gamma_early = model.get_gamma(0, 1000)
    gamma_mid = model.get_gamma(500, 1000)
    gamma_late = model.get_gamma(960, 1000)
    assert gamma_early == config.moe.gamma, (
        f"Early gamma should be {config.moe.gamma}, got {gamma_early}"
    )
    assert gamma_mid == config.moe.gamma, (
        f"Mid gamma should be {config.moe.gamma}, got {gamma_mid}"
    )
    assert gamma_late == 0.0, (
        f"Late gamma (>95%) should be 0.0, got {gamma_late}"
    )
    print(
        f"  Test 8: Gamma schedule — "
        f"early={gamma_early}, mid={gamma_mid}, late={gamma_late} "
        f"(frozen at 95%)"
    )

    print()
    print("=" * 60)
    print("ALL 8 TESTS PASSED")
    print("=" * 60)


# =============================================================================
# SECTION 8: LIGHTNING INDEXER (Phase 2)
# =============================================================================


# =============================================================================
# SECTION 9: DEEPSEEK SPARSE ATTENTION (Phase 2)
# =============================================================================


if __name__ == "__main__":
    test_nanoseek()

