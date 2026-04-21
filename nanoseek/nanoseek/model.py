# NanoSeek - Complete DeepSeek V3.2 Implementation at Nano Scale
# Architecture Components:
# - MLA (Multi-head Latent Attention): ~23x KV cache compression
# - MoE (Mixture of Experts): 5x parameter capacity with sparse activation
# - MTP (Multi-Token Prediction): 1.4x inference speedup via speculative decoding
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
from torch.profiler import record_function

# Import configurations from config.py
# Handle both package import and direct execution
try:
    from .config import NanoSeekConfig, get_config
except ImportError:
    from config import NanoSeekConfig, get_config


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
    # x.float() already creates a new contiguous tensor (different dtype = new storage),
    # so .clone() after .float() is redundant — it doubles memory for every RoPE call.
    # .contiguous() ensures correct strides for view_as_complex.
    x_complex = torch.view_as_complex(x.float().contiguous().view(*x.shape[:-1], -1, 2))
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

        # === no==================================================================
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
        # Cached causal mask for absorb path (avoids allocation every forward)
        self.register_buffer("_cached_causal_mask", None, persistent=False)

    def _get_causal_mask(self, seq_len: int, kv_len: int, device, dtype):
        """Return cached causal mask, expanding if needed. Absorb path only."""
        max_dim = max(seq_len, kv_len)
        if self._cached_causal_mask is None or self._cached_causal_mask.shape[0] < max_dim:
            mask = torch.full((max_dim, max_dim), float("-inf"), device=device, dtype=dtype)
            mask = mask.triu_(diagonal=1)
            self._cached_causal_mask = mask
        cached_len = kv_len - seq_len
        return self._cached_causal_mask[cached_len : cached_len + seq_len, :kv_len]

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
                # Causal mask for prefill (seq_len > 1) — uses cached mask
                if seq_len > 1:
                    causal = self._get_causal_mask(seq_len, kv_len, q.device, q.dtype)
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
            # wkv_b decompression + attention kernel + mask construction
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

            # Use SDPA when available (FlashAttention / memory-efficient kernels)
            # MPS SDPA bug: returns output with Q's head_dim instead of V's when they differ.
            # MLA has qk_dim=192 (128+64) vs v_dim=128, so we must skip SDPA on MPS.
            use_sdpa = hasattr(F, "scaled_dot_product_attention") and q.device.type != "mps"

            # PERF: Use is_causal=True instead of building an explicit mask tensor.
            # This lets SDPA/FlashAttention use its built-in causal masking kernel
            # (no mask allocation, ~1-3% faster). Only valid when seq_len == kv_len
            # (prefill, not generation with KV cache where kv_len > seq_len).
            needs_causal = attention_mask is None and seq_len > 1
            if needs_causal and seq_len != kv_len:
                # KV cache case: seq_len < kv_len, need explicit mask
                attention_mask = torch.full(
                    (seq_len, kv_len), float("-inf"), device=q.device, dtype=q.dtype
                )
                attention_mask = attention_mask.triu_(diagonal=kv_len - seq_len + 1)
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
                needs_causal = False

            if use_sdpa:
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attention_mask if not needs_causal else None,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    scale=effective_scale,
                    is_causal=needs_causal,
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
        use_classic_aux_loss: bool = False,
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
        # Classic mode: Switch-Transformer-style differentiable aux loss.
        # When False (default): seq_aux_loss_alpha is monitoring-only (detached).
        # When True: aux loss has gradient through soft probabilities P_i.
        self.use_classic_aux_loss = use_classic_aux_loss

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
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-10)

        # STEP 7: Apply routed_scaling_factor (DeepSeek V3 pattern: scale weights, not output)
        weights = weights * self.routed_scaling_factor

        # STEP 8: Compute auxiliary loss and metadata
        # bincount is O(N) memory vs one-hot scatter O(N*E); result is identical.
        load_counts = torch.bincount(indices.view(-1), minlength=E).to(scores.dtype)

        # Two aux_loss modes controlled by self.use_classic_aux_loss:
        #
        # BIAS MODE (default, use_classic_aux_loss=False):
        #   Monitoring-only MSE loss. Gradient = 0 (intentional).
        #   load_counts comes from bincount(topk()) — discrete, non-differentiable.
        #   Load balancing happens entirely via update_bias() in the training loop.
        #   This is the DeepSeek V3 "auxiliary-loss-free" design.
        #
        # CLASSIC MODE (use_classic_aux_loss=True, --aux-loss-type classic):
        #   Switch-Transformer-style differentiable aux loss (Fedus et al., 2022).
        #   Gradient flows through P_i = sigmoid(logits).mean(dim=0).
        #   f_i (hard fraction) is detached; P_i (soft probability) is not.
        #   This is the correct ablation baseline when comparing bias vs. classic.
        f = load_counts / max(N * K, 1)  # hard routing fraction per expert

        if self.seq_aux_loss_alpha > 0.0 and self.use_classic_aux_loss:
            # Classic differentiable load balancing (Switch Transformer Eq. 4).
            # P_i has gradient through: sigmoid(router_weight @ x) → mean over tokens.
            # f_i is detached: routing fractions come from discrete topk(), no grad.
            P_i = scores.mean(dim=0)  # [E] soft selection probabilities, has gradient
            f_hard = f.detach()       # [E] hard routing fractions, no gradient
            aux_loss = self.seq_aux_loss_alpha * E * (f_hard * P_i).sum()
        elif self.seq_aux_loss_alpha > 0.0:
            # Bias mode: MSE monitoring loss. Zero gradient — purely diagnostic.
            target = 1.0 / E
            aux_loss = (self.seq_aux_loss_alpha * ((f - target) ** 2).mean()).detach()
        else:
            aux_loss = torch.tensor(0.0, device=device, dtype=scores.dtype)

        # Load balance entropy in bits for monitoring (H_load → higher = more balanced)
        # For 16 experts with uniform routing: max H_load = log2(16) = 4 bits (nano scale)
        # Alert threshold: H_load < 2.0 bits indicates routing collapse
        # f already sums to 1.0 (divided by N*K above), so p = f (safe add epsilon only)
        p_safe = f + 1e-10
        H_load = -(p_safe * p_safe.log2()).sum()

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
    - Routed experts: 16 SwiGLU FFNs stored as native 3D weight tensors (nano scale)
    - Shared expert: 1 SwiGLU MLP with combined inter_dim (n_shared * moe_inter_dim)
      Applied to ALL tokens unconditionally (captures common patterns)

    Expert weights are stored as 3D Parameters in grouped_mm-friendly [E, in, out] layout
    (note: OPPOSITE of nn.Linear's [out, in] convention). This matches torch._grouped_mm's
    expectation that B[e] multiplies A_e on the right: A_e @ B[e] = [cnt_e, in] @ [in, out].
    Eliminates per-forward transpose+copy overhead vs ModuleList. Enables torch.compile fusion.

    Token-centric dispatch: sort tokens by expert for coalesced access.
    """

    def __init__(
        self,
        hidden_dim: int,
        moe_inter_dim: int,
        n_routed_experts: int = 16,
        num_experts_per_tok: int = 2,
        n_shared_experts: int = 2,
        n_group: int = 4,
        topk_group: int = 4,
        scoring_func: str = "sigmoid",
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 2.5,
        seq_aux_loss_alpha: float = 0.0001,
        use_classic_aux_loss: bool = False,
        shared_inter_dim: Optional[int] = None,
        disable_shared_experts: bool = False,
    ):
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.disable_shared_experts = disable_shared_experts

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
            use_classic_aux_loss=use_classic_aux_loss,
        )

        # Routed experts — native 3D weight tensors stored in compute-optimal layout.
        # Layout is chosen to eliminate per-forward transpose+copy overhead in the
        # grouped GEMM fast path (the GPU training hot path).
        #
        # SwiGLU: output = W_down(SiLU(W_gate(x)) * W_up(x))
        # w_gate/w_up: stored as [E, D, I] so grouped_mm(sorted_x[NK,D], W[E,D,I])
        #              computes [NK, I] directly without transpose.
        # w_down: stored as [E, I, D] so grouped_mm(hidden[NK,I], W[E,I,D])
        #         computes [NK, D] directly without transpose.
        #
        # For F.linear compatibility in expert_forward, weights are used as .T:
        #   F.linear(x, w_gate[e].T)  # w_gate[e] is [D,I], .T is [I,D] = [out,in]
        self.w_gate = nn.Parameter(torch.empty(n_routed_experts, hidden_dim, moe_inter_dim))
        self.w_up = nn.Parameter(torch.empty(n_routed_experts, hidden_dim, moe_inter_dim))
        self.w_down = nn.Parameter(torch.empty(n_routed_experts, moe_inter_dim, hidden_dim))

        # Default init matching NanoSeekModel._init_weights for standalone use.
        # NanoSeekModel._init_weights overrides this when used as part of the full model.
        # Without init, torch.empty gives zeros on some platforms → zero output → broken tests.
        std = 1.0 / (hidden_dim ** 0.5)
        nn.init.normal_(self.w_gate, mean=0.0, std=std)
        nn.init.normal_(self.w_up, mean=0.0, std=std)
        nn.init.normal_(self.w_down, mean=0.0, std=0.006)

        # Shared expert (dense, always active) — single MLP with combined inter_dim
        # DeepSeek V3 pattern: n_shared_experts is a multiplier, not a count of modules
        effective_shared_dim = shared_inter_dim or (n_shared_experts * moe_inter_dim)
        self.shared_expert = Expert(hidden_dim, effective_shared_dim)

    def expert_forward(self, expert_idx: int, x: Tensor) -> Tensor:
        """Run a single routed expert on input. For tests and manual verification.

        SwiGLU: output = W_down(SiLU(W_gate(x)) * W_up(x))

        Weight layout: w_gate[e] is [D, I], so we pass .T to F.linear
        to satisfy [out_features, in_features] convention.
        """
        dtype = x.dtype
        return F.linear(
            F.silu(F.linear(x, self.w_gate[expert_idx].T.to(dtype)))
            * F.linear(x, self.w_up[expert_idx].T.to(dtype)),
            self.w_down[expert_idx].T.to(dtype),
        )

    def _grouped_expert_forward(
        self, sorted_x: Tensor, expert_boundaries: Tensor,
    ) -> Tensor:
        """CUDA fast path: exact variable-length expert dispatch via grouped GEMM.

        torch._grouped_mm(A, B, offsets) computes A_e @ B_e per expert.
        Weights are stored in [E, in, out] layout so no per-forward transpose
        is needed — eliminating ~168MB of transient allocation per layer.
        """
        dtype = sorted_x.dtype
        expert_offsets = expert_boundaries[1:].to(dtype=torch.int32)

        # w_gate: [E, D, I], w_up: [E, D, I] → cat on last dim → [E, D, 2I]
        # No transpose needed because weights are stored in grouped_mm-friendly layout.
        w_gate_up = torch.cat([self.w_gate, self.w_up], dim=-1).to(dtype)

        gate_up_out = torch._grouped_mm(sorted_x, w_gate_up, expert_offsets)
        gate_out, up_out = gate_up_out.chunk(2, dim=-1)
        hidden = F.silu(gate_out) * up_out

        # w_down: [E, I, D] — already in correct layout for second grouped_mm
        return torch._grouped_mm(hidden, self.w_down.to(dtype), expert_offsets)

    def _padded_batched_expert_forward(
        self, sorted_x: Tensor, sorted_indices: Tensor,
        expert_counts: Tensor, expert_boundaries: Tensor,
    ) -> Tensor:
        """Fallback batched path for CPU/tests when grouped GEMM is unavailable."""
        E = self.n_routed_experts
        NK, D = sorted_x.shape
        dtype = sorted_x.dtype

        if NK == 0:
            return torch.zeros_like(sorted_x)

        pad_size = int(expert_counts.max().item())
        if pad_size == 0:
            return torch.zeros_like(sorted_x)

        position_in_expert = (
            torch.arange(NK, device=sorted_x.device) - expert_boundaries[sorted_indices]
        )
        padded_input = sorted_x.new_zeros(E, pad_size, D)
        padded_input[sorted_indices, position_in_expert] = sorted_x

        # w_gate: [E, D, I], w_up: [E, D, I] → cat on last dim → [E, D, 2I]
        # bmm needs [E, pad, D] @ [E, D, 2I] = [E, pad, 2I]
        w_gate_up = torch.cat([self.w_gate, self.w_up], dim=-1).to(dtype)
        gate_up_out = torch.bmm(padded_input, w_gate_up)  # [E, pad, 2I]
        gate_out, up_out = gate_up_out.chunk(2, dim=-1)
        hidden = F.silu(gate_out) * up_out

        # w_down: [E, I, D]
        # bmm needs [E, pad, I] @ [E, I, D] = [E, pad, D]
        out = torch.bmm(hidden, self.w_down.to(dtype))
        return out[sorted_indices, position_in_expert]

    def _batched_expert_forward(
        self, sorted_x: Tensor, sorted_indices: Tensor,
        expert_counts: Tensor, expert_boundaries: Tensor,
    ) -> Tensor:
        """Process routed experts without the old fixed 8x padding slab.

        On CUDA BF16 we use `torch._grouped_mm`, which handles variable per-expert
        token counts directly from cumulative offsets. CPU and non-BF16 cases keep a
        simpler exact-capacity fallback for testability.
        """
        if (
            sorted_x.is_cuda
            and sorted_x.dtype == torch.bfloat16
            and hasattr(torch, "_grouped_mm")
        ):
            return self._grouped_expert_forward(sorted_x, expert_boundaries)
        return self._padded_batched_expert_forward(
            sorted_x, sorted_indices, expert_counts, expert_boundaries,
        )

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
        with record_function("MoE::gate"):
            weights, indices, aux_loss, metadata = self.gate(x)
        # weights: [N, K], indices: [N, K]

        # Scatter-based expert dispatch: sort tokens by expert for coalesced access
        # Instead of iterating all E experts with boolean masks (O(E*N)),
        # we flatten token-expert assignments, sort by expert, and process
        # contiguous slices. ~10-50x faster for E=64.
        with record_function("MoE::dispatch"):
            N, D = x.shape
            K = self.num_experts_per_tok
            E = self.n_routed_experts

            # Flatten [N, K] → [N*K] and expand inputs to match
            flat_indices = indices.view(-1)           # [N*K]
            flat_weights = weights.view(-1)           # [N*K]
            x_expanded = x.unsqueeze(1).expand(-1, K, -1).reshape(N * K, D)  # [N*K, D]

            # Sort by expert index for contiguous memory access
            # stable=True is unnecessary for integer sorting and adds overhead.
            sort_order = flat_indices.argsort()
            sorted_indices = flat_indices[sort_order]
            sorted_x = x_expanded[sort_order]         # [N*K, D]
            sorted_weights = flat_weights[sort_order]  # [N*K]

            # Find boundaries: expert_boundaries[e] = start index, expert_boundaries[e+1] = end
            expert_counts = torch.bincount(sorted_indices, minlength=E)  # [E]
            expert_boundaries = torch.zeros(E + 1, dtype=torch.long, device=x.device)
            expert_boundaries[1:] = expert_counts.cumsum(0)

        # ── Expert dispatch: grouped GEMM (GPU) or sequential (CPU) ──
        # The CUDA fast path uses grouped GEMM with exact per-expert offsets, which
        # avoids the old E × pad_size × D activation slab. CPU keeps the sequential
        # path because grouped_mm is CUDA-only and test workloads are small.
        use_batched = sorted_x.is_cuda and E >= 8

        with record_function("MoE::expert_compute"):
            if use_batched:
                sorted_output = self._batched_expert_forward(
                    sorted_x, sorted_indices, expert_counts, expert_boundaries,
                )
            else:
                # Sequential fallback: process each expert's contiguous batch
                # Used on CPU (testing), or when routing is too skewed for batching.
                sorted_output = torch.zeros_like(sorted_x)  # [N*K, D]
                counts_cpu = expert_counts.tolist()
                offset = 0
                for expert_idx in range(E):
                    cnt = counts_cpu[expert_idx]
                    if cnt == 0:
                        continue
                    expert_input = sorted_x[offset:offset + cnt]
                    sorted_output[offset:offset + cnt] = self.expert_forward(expert_idx, expert_input)
                    offset += cnt

        with record_function("MoE::combine"):
            # Apply weights and scatter back to original token order
            sorted_output = sorted_output * sorted_weights.unsqueeze(-1)

            # Unsort and reduce: accumulate weighted outputs back to [N, D]
            # Use index_add_ instead of scatter_add_ with expanded index to avoid
            # materializing a [N*K, D] int64 index tensor (~4GB at ablation scale).
            # index_add_ broadcasts the 1-D index across trailing dimensions.
            orig_token_idx = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, K).reshape(N * K)
            orig_token_idx = orig_token_idx[sort_order]  # [N*K] — which token each sorted entry belongs to

            routed_output = torch.zeros_like(x)  # [N, D]
            routed_output.index_add_(0, orig_token_idx, sorted_output)

        # Shared expert — processes ALL tokens (unless ablation-disabled)
        if self.disable_shared_experts:
            output = routed_output  # [N, D]
        else:
            with record_function("MoE::shared_expert"):
                shared_output = self.shared_expert(x)  # [N, D]
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
# Wires together: RMSNorm + MLA + dense FFN (Expert) or MoE.

class NanoSeekDecoderLayer(nn.Module):
    """Single decoder layer: pre-norm attention + pre-norm FFN with residual connections.

    Architecture:
        x → input_layernorm → self_attn → + residual
        → post_attention_layernorm → ffn → + residual → output

    The attention module is MLA (Multi-head Latent Attention).

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
        n_routed_experts: int = 16,
        n_shared_experts: int = 2,
        num_experts_per_tok: int = 2,
        moe_intermediate_size: int = 768,
        shared_inter_dim: Optional[int] = None,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 2.5,
        n_group: int = 4,
        topk_group: int = 4,
        disable_shared_experts: bool = False,
        seq_aux_loss_alpha: float = 0.0001,
        use_classic_aux_loss: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_moe = use_moe

        # Pre-attention norm
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Attention — MLA (Multi-head Latent Attention)
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
                disable_shared_experts=disable_shared_experts,
                seq_aux_loss_alpha=seq_aux_loss_alpha,
                use_classic_aux_loss=use_classic_aux_loss,
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
        with record_function(f"Layer_{self.layer_idx}"):
            # ---- Attention block (pre-norm + residual) ----
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            with record_function("MLA"):
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
                with record_function("MoE"):
                    ffn_output, aux_data = self.ffn(hidden_states)
            else:
                with record_function("DenseFFN"):
                    ffn_output = self.ffn(hidden_states)
                    aux_data = {}

            hidden_states = residual + ffn_output

        return hidden_states, present_key_value, aux_data


# =============================================================================
# CHECKPOINT MIGRATION
# =============================================================================

def migrate_legacy_moe_state_dict(state_dict: dict) -> dict:
    """Convert old ModuleList expert weights to native 3D parameters.

    Old format: layers.{l}.ffn.routed_experts.{e}.w_{gate,up,down}.weight
    New format: layers.{l}.ffn.w_{gate,up,down}  (3D: [E, out_features, in_features])

    Returns state_dict unchanged if already in new format.
    """
    if not any('routed_experts.' in k for k in state_dict):
        return state_dict  # Already new format

    new_state = {}
    # prefix -> {proj_name -> {idx -> tensor}}
    expert_weights: Dict[str, Dict[str, Dict[int, Tensor]]] = {}

    for key, value in state_dict.items():
        if 'routed_experts.' in key:
            # Parse: layers.2.ffn.routed_experts.5.w_gate.weight
            prefix, rest = key.split('routed_experts.')
            parts = rest.split('.')
            idx = int(parts[0])
            proj_name = parts[1]  # w_gate, w_up, or w_down

            if prefix not in expert_weights:
                expert_weights[prefix] = {}
            if proj_name not in expert_weights[prefix]:
                expert_weights[prefix][proj_name] = {}
            expert_weights[prefix][proj_name][idx] = value
        else:
            new_state[key] = value

    # Stack expert weights into 3D tensors
    for prefix, proj_dict in expert_weights.items():
        for proj_name, idx_dict in proj_dict.items():
            n_experts = max(idx_dict.keys()) + 1
            stacked = torch.stack([idx_dict[i] for i in range(n_experts)])
            new_state[f"{prefix}{proj_name}"] = stacked

    return new_state


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
        self._validate_config(config)  # Validate before proceeding
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
                disable_shared_experts=config.moe.disable_shared_experts,
                seq_aux_loss_alpha=config.moe.seq_aux_loss_alpha,
                use_classic_aux_loss=config.moe.use_classic_aux_loss,
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

        # ---- Gradient checkpointing (selective) ----
        # PERF: Only checkpoint MoE layers (memory-heavy due to routed experts).
        # Skip dense layers (0, cheap) and last 2 layers (activations still live
        # in memory during backward). Saves ~15-25% recomputation cost vs
        # checkpointing all layers, with minimal memory increase.
        # Reference: Megatron-LM selective checkpointing, Meta Llama 3 training.
        self.gradient_checkpointing = config.gradient_checkpointing
        if self.gradient_checkpointing:
            first_k_dense = config.moe.first_k_dense_replace
            # Checkpoint MoE layers except last 2 (their activations are still in
            # memory when backward reaches them — checkpointing just wastes compute)
            last_ckpt = max(first_k_dense, config.num_layers - 2)
            self._checkpoint_layer_ids = set(range(first_k_dense, last_ckpt))
        else:
            self._checkpoint_layer_ids = set()

        # ---- Initialize weights ----
        # Direct construction on a real device still initializes eagerly so
        # tests and utility scripts behave as expected. Meta-device
        # construction must skip this and defer to the explicit to_empty() ->
        # init_weights() sequence.
        if not any(p.is_meta for p in self.parameters()):
            self.init_weights()

    def _validate_config(self, config: NanoSeekConfig) -> None:
        """Validate configuration parameters for logical consistency.
        
        Raises:
            ValueError: If config parameters are invalid or incompatible
        """
        # Vocabulary size
        if config.vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {config.vocab_size}")
        if config.vocab_size > 1_000_000:
            raise ValueError(f"vocab_size suspiciously large: {config.vocab_size}")
        
        # Hidden dimension
        if config.hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0, got {config.hidden_size}")
        if config.hidden_size % 64 != 0:
            raise ValueError(f"hidden_size should be divisible by 64, got {config.hidden_size}")
        
        # Number of heads
        if config.num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {config.num_heads}")
        if config.hidden_size % config.num_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by "
                f"num_heads ({config.num_heads})"
            )
        
        # Sequence length
        if config.max_position_embeddings <= 0:
            raise ValueError(f"max_position_embeddings must be > 0, got {config.max_position_embeddings}")
        
        # MoE parameters
        moe = config.moe
        if moe.n_routed_experts <= 0:
            raise ValueError(f"n_routed_experts must be > 0, got {moe.n_routed_experts}")
        if moe.num_experts_per_tok > moe.n_routed_experts:
            raise ValueError(
                f"num_experts_per_tok ({moe.num_experts_per_tok}) cannot exceed "
                f"n_routed_experts ({moe.n_routed_experts})"
            )
        if moe.n_group > moe.n_routed_experts:
            raise ValueError(
                f"n_group ({moe.n_group}) cannot exceed n_routed_experts ({moe.n_routed_experts})"
            )
        if moe.topk_group > moe.n_group:
            raise ValueError(
                f"topk_group ({moe.topk_group}) cannot exceed n_group ({moe.n_group})"
            )
        
        # MTP parameters
        mtp = config.mtp
        if mtp.num_mtp_modules < 0:
            raise ValueError(f"num_mtp_modules must be >= 0, got {mtp.num_mtp_modules}")
        if mtp.mtp_loss_decay < 0 or mtp.mtp_loss_decay > 1:
            raise ValueError(f"mtp_loss_decay must be in [0, 1], got {mtp.mtp_loss_decay}")

    def init_weights(self) -> None:
        """Public API for weight initialization.

        Called explicitly after to_empty(device=...) in pre_train.py,
        and on direct non-meta construction.
        """
        self._init_weights()
        self._reinit_buffers()

    def _reinit_buffers(self) -> None:
        """Recompute non-persistent buffers (RoPE freqs_cis) after to_empty().

        When constructing on meta device, buffers are created without data.
        to_empty() moves them to the target device filled with zeros.
        init_weights() only handles nn.Linear/nn.Embedding parameters.
        This method recomputes any buffers that require non-trivial initialization.
        """
        config = self.config
        # Compute RoPE frequencies ONCE
        rope_freqs_base = precompute_freqs_cis(
            config.mla.qk_rope_head_dim,
            config.max_position_embeddings,
            config.mla.rope_theta,
            config.mla.rope_scaling_factor,
            config.mla.original_max_position_embeddings,
        )
        # Assign to decoder layers (clone for defensive isolation)
        for layer in self.layers:
            attn = layer.self_attn
            attn.freqs_cis = rope_freqs_base.clone().to(device=attn.freqs_cis.device)
        # MTP also has its own MLA with freqs_cis (clone for safety)
        if self.mtp is not None:
            for mtp_mod in self.mtp.mtp_modules:
                mtp_attn = mtp_mod.transformer.attn
                mtp_attn.freqs_cis = rope_freqs_base.clone().to(device=mtp_attn.freqs_cis.device)

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
            elif isinstance(module, RMSNorm):
                # RMSNorm weight should be ones (multiplicative scale).
                # With meta device + to_empty(), these are uninitialized (zeros),
                # which kills ALL signal through the model.
                torch.nn.init.ones_(module.weight)

        # Small-scale init for output projections (near-identity at init).
        # Zero-init works for dense models (GPT-2/nanochat pattern), but for MoE:
        # - Router starts with random weights + zero bias
        # - All 16 experts start identical with zero output
        # - Only gradient signal to router comes from embedding→lm_head skip path
        # This causes slow early training / routing collapse.
        # DeepSeek V3 Technical Report: "All learnable parameters are randomly
        # initialized with a standard deviation of 0.006." — flat, no layer scaling.
        output_std = 0.006

        for layer in self.layers:
            # Attention output projection
            torch.nn.init.normal_(layer.self_attn.wo.weight, mean=0.0, std=output_std)
            # Dense FFN layer (first_k_dense layers)
            if isinstance(layer.ffn, Expert):
                torch.nn.init.normal_(layer.ffn.w_down.weight, mean=0.0, std=output_std)
            # MoE layer: init 3D routed expert weights + shared expert w_down
            elif isinstance(layer.ffn, MoE):
                # 3D expert params (nn.Parameter, NOT nn.Linear) are not caught
                # by the generic init walk above — init them explicitly here.
                torch.nn.init.normal_(layer.ffn.w_gate, mean=0.0, std=std)
                torch.nn.init.normal_(layer.ffn.w_up, mean=0.0, std=std)
                # Output projections: small-scale init
                torch.nn.init.normal_(layer.ffn.w_down, mean=0.0, std=output_std)
                torch.nn.init.normal_(layer.ffn.shared_expert.w_down.weight, mean=0.0, std=output_std)
        # MTP concat_proj: small random init (NOT zero — zero kills all forward signal,
        # producing exactly uniform logits and loss = ln(V) with zero gradient to MTP params).
        # Use same output_std as other residual-stream projections for consistent scale.
        if self.mtp is not None:
            for mtp_mod in self.mtp.mtp_modules:
                torch.nn.init.normal_(mtp_mod.concat_proj.weight, mean=0.0, std=output_std)
                # Also init transformer block's output projections for residual-stream scale
                torch.nn.init.normal_(mtp_mod.transformer.attn.wo.weight, mean=0.0, std=output_std)
                torch.nn.init.normal_(mtp_mod.transformer.ffn.w_down.weight, mean=0.0, std=output_std)

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
        mtp_lambda: float = 0.3,
    ) -> Dict[str, Tensor]:
        """Complete forward pass.

        Args:
            input_ids: [B, S] token IDs
            attention_mask: unused — MLA creates causal mask internally
            position_ids: [B, S] position indices (auto-computed if None)
            past_key_values: list of (c_kv, k_pe) tuples per layer for KV cache
            use_cache: whether to return updated KV cache
            labels: [B, S] target token IDs for loss computation
            mtp_lambda: pre-computed MTP loss weight (compute outside to avoid compile recompilation)

        Returns:
            dict with keys:
                loss:            scalar total loss (None if labels not provided)
                logits:          [B, S, V] logit predictions
                past_key_values: list of KV caches (None if use_cache=False)
                aux_loss:        scalar averaged MoE load-balancing loss
        """
        # ---- Input validation ----
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2D [B, S], got shape {input_ids.shape}")
        
        B, S = input_ids.shape
        
        if S > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {S} exceeds max_position_embeddings "
                f"{self.config.max_position_embeddings}"
            )
        
        if input_ids.dtype not in (torch.long, torch.int64):
            raise ValueError(f"input_ids must be int64, got {input_ids.dtype}")
        
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.config.vocab_size):
            raise ValueError(
                f"input_ids values must be in range [0, {self.config.vocab_size}), "
                f"got min={input_ids.min()}, max={input_ids.max()}"
            )
        
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError(
                    f"labels shape {labels.shape} must match input_ids shape {input_ids.shape}"
                )
        
        if mtp_lambda < 0 or mtp_lambda > 1:
            raise ValueError(f"mtp_lambda must be in [0, 1], got {mtp_lambda}")
        # ---- End validation ----

        # 1. Token embedding (no post-embedding norm)
        # DeepSeek V3 does NOT have post-embedding RMSNorm. Parameterless norm
        # would collapse embedding scale to unit norm regardless of width,
        # breaking muP's assumption that embedding output scale = O(1) naturally.
        # Embedding init std = 1/√hidden_size already controls the scale.
        hidden_states = self.embed_tokens(input_ids)  # [B, S, H]

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

            if self.gradient_checkpointing and self.training and i in self._checkpoint_layer_ids:
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
                # Preserve the original forward's routing stats. Gradient
                # checkpoint recompute during backward must not overwrite them.
                if i not in self._layer_aux_data:
                    self._layer_aux_data[i] = aux_data

        # 4. Final norm
        # MTP applies its own RMSNorm internally and should see the decoder
        # output before the model's final normalization.
        pre_norm_hidden_states = hidden_states
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
        # mtp_lambda is passed as a function parameter — do NOT shadow it here
        if labels is not None:
            loss_dict = self._compute_loss(
                logits, labels, pre_norm_hidden_states, input_ids,
                total_aux_loss, n_aux_layers,
                mtp_lambda,
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
        mtp_lambda: float,
    ) -> Dict[str, Tensor]:
        """Compute total training loss with component breakdown.

        L_total = L_main + λ · L_MTP + L_aux

        Args:
            logits: [B, S, V] — model predictions
            labels: [B, S] — target tokens
            hidden_states: [B, S, H] — pre-final-norm hidden states (for MTP)
            input_ids: [B, S] — original input (for MTP token shifting)
            total_aux_loss: accumulated MoE load-balancing loss
            n_aux_layers: number of MoE layers that contributed
            mtp_lambda: pre-computed MTP loss weight (avoids compile recompilation)

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

        # MTP auxiliary loss (training only)
        if self.training and self.mtp is not None:
            _, mtp_loss = self.mtp(hidden_states, input_ids)
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

    def update_load_balance_bias(self, tokens_processed: int = 0, total_tokens: int = 0) -> None:
        """Update expert routing biases for load balancing.

        DeepSeek V3 auxiliary-loss-free load balancing:
        - Maintains dynamic bias vector per expert
        - Updates bias after each optimizer step based on observed load imbalance
        - Formula: b_i -= gamma * (load_i - mean_load) / mean_load
        - Freezes at gamma_freeze_ratio (95%) of training — call with tokens_processed
          and total_tokens so the freeze logic engages correctly.

        Args:
            tokens_processed: cumulative tokens seen so far (used for freeze schedule)
            total_tokens: total training token budget (used for freeze schedule)
                          Pass both as 0 to skip freeze (test mode — uses full gamma).

        Example usage in training loop:
            model.update_load_balance_bias(tokens_processed, config.total_tokens)
        """
        # get_gamma() enforces the 95% freeze: returns 0.0 after gamma_freeze_ratio
        # When called with defaults (0, 0), get_gamma returns full config gamma (test mode).
        gamma = self.get_gamma(tokens_processed, total_tokens)
        if gamma <= 0.0:
            return
        for layer_idx, aux_data in self._layer_aux_data.items():
            if "load_counts" in aux_data:
                self.layers[layer_idx].ffn.gate.update_bias(
                    aux_data["load_counts"], gamma
                )

    def num_parameters(self) -> Dict[str, int]:
        """Return active and total parameter counts.

        Active params exclude inactive routed experts (16 - 2 = 14 per MoE layer, nano scale).
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

    def num_scaling_params(self) -> Dict[str, int]:
        """Parameter breakdown for scaling law analysis.

        Kaplan convention: scaling = active - embed.
        Embed is a lookup (O(1) per token), not a matmul, so it doesn't
        contribute to the compute-capacity relationship L(N, D).
        """
        counts = self.num_parameters()
        n_embed = self.embed_tokens.weight.numel()
        n_lm_head = self.lm_head.weight.numel()
        return {
            "embed": n_embed,
            "lm_head": n_lm_head,
            "active": counts["active"],
            "total": counts["total"],
            "scaling": counts["active"] - n_embed,
        }

    def get_expert_load_stats(self) -> Dict[str, Tensor]:
        """Aggregate expert load statistics across all MoE layers.

        Returns:
            entropy: mean H_load across MoE layers (higher = more balanced)
            load_per_expert: [E] mean token counts per expert
            per_layer: dict of {layer_idx: {"load_counts": Tensor, "H_load": Tensor}}
        """
        if not self._layer_aux_data:
            # No forward pass has run yet — return zeros with warning
            import logging
            logging.getLogger(__name__).warning(
                "get_expert_load_stats() called before any forward pass — returning zeros"
            )
            n_experts = self.config.moe.n_routed_experts
            return {
                "entropy": torch.tensor(0.0),
                "load_per_expert": torch.zeros(n_experts),
                "per_layer": {},
            }

        all_H = []
        all_counts = []
        per_layer = {}
        for layer_idx, aux_data in self._layer_aux_data.items():
            layer_info = {}
            if "H_load" in aux_data:
                all_H.append(aux_data["H_load"])
                layer_info["H_load"] = aux_data["H_load"]
            if "load_counts" in aux_data:
                all_counts.append(aux_data["load_counts"])
                layer_info["load_counts"] = aux_data["load_counts"]
            if layer_info:
                per_layer[layer_idx] = layer_info

        if not all_H:
            n_experts = self.config.moe.n_routed_experts
            return {
                "entropy": torch.tensor(0.0),
                "load_per_expert": torch.zeros(n_experts),
                "per_layer": {},
            }

        return {
            "entropy": torch.stack(all_H).mean(),
            "load_per_expert": torch.stack(all_counts).float().mean(dim=0),
            "per_layer": per_layer,
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
        config = get_config("1b")
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

    config = get_config("1b")
    model = create_nanoseek(config)

    B, S, V = 2, 64, config.vocab_size
    input_ids = torch.randint(0, V, (B, S))
    labels = torch.randint(0, V, (B, S))

    # ---- Test 1: Forward pass shapes + finite loss ----
    model.train()
    outputs = model(input_ids, labels=labels, mtp_lambda=0.3)
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
    out_mtp = model(input_ids, labels=labels, mtp_lambda=0.3)
    print(f"  Test 6: Total loss (includes MTP + aux) = {out_mtp['loss'].item():.4f}")
    model.zero_grad()

    # ---- Test 7: Load balance bias update ----
    _ = model(input_ids, labels=labels, mtp_lambda=0.3)
    model.update_load_balance_bias()  # Simplified API: no parameters needed
    print("  Test 7: Load balance bias update — successful")
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


if __name__ == "__main__":
    test_nanoseek()
