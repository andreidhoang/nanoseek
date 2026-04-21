# NanoSeek Configuration — One Flat Dataclass
#
# Design: nanochat-style single dataclass. Sub-configs (MLA, MoE, MTP) are
# exposed as @property namespace accessors for model.py backward compat.
#
# Three scales: anchor (768h), ablation (1280h), 1B (2048h)
# Reference: DeepSeek-V3 (arXiv:2412.19437), DeepSeek-V3.2 (arXiv:2512.02556)

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional
import torch


# ============================================================================
# DeepSeek Family Constants (Architecture Invariants)
# ============================================================================
# Absolute constants across DeepSeek V2-Lite (2048h), V2 (5120h), V3 (7168h).

DEEPSEEK_QK_NOPE_HEAD_DIM = 128   # Non-positional query/key dimension
DEEPSEEK_QK_ROPE_HEAD_DIM = 64    # RoPE dimension (shared across heads)
DEEPSEEK_V_HEAD_DIM = 128         # Value dimension

# MLA compression ratios (relative to hidden_size)
DEEPSEEK_Q_LORA_RATIO = 0.215     # q_lora_rank / hidden_size
DEEPSEEK_KV_LORA_RATIO = 0.070    # kv_lora_rank / hidden_size

# FFN ratios
DEEPSEEK_DENSE_FFN_RATIO = 2.56   # intermediate_size / hidden_size for dense layers

# MoE intermediate size ratio — NANO SCALE ADJUSTMENT
#
# Conservation law: N_experts × MOE_INTER_RATIO = constant (preserves total params + active FLOPs)
#   DeepSeek V3 original (7168h, 671B total): 64 × 0.375 = 24
#   NanoSeek nano scale  (768-2048h, ≤5B total): 16 × 1.5 = 24  ← same product
#
# WHY we change from 64→16 experts with 0.375→1.5 inter ratio at nano scale:
#
# 1. Expert capacity: V3's 0.375 ratio gives 29-token FFN width at 7168h. At 768h that's
#    only 288 tokens — too narrow for experts to learn distinct specializations.
#    With ratio=1.5: 1152 at 768h (matches dense sublayer), 7.4M params per expert at 1280h.
#
# 2. Statistical efficiency: 64 experts needs ~16K tokens/step to see each expert ~250 times.
#    16 experts needs only ~4K tokens/step. Our seq_len=4096, batch=128 → 512K tokens/step.
#    Both are fine, but 16 experts have 4× better gradient signal per expert.
#
# 3. Routing collapse risk: With 64 experts top-8 at nano scale, routing entropy collapses
#    faster (less capacity per expert → experts can't differentiate → all load on winners).
#    16 experts top-2 is sparser (12.5% = same) but each expert has 4× more capacity to
#    learn a niche before becoming overloaded.
#
# 4. sparsity ratio: 2/16 = 8/64 = 12.5% — IDENTICAL. Active FLOPs unchanged.
#
# Reference: Krajewski et al. 2024 (MoE granularity scaling) shows expert capacity per param
# should scale ~linearly with model size; keeping V3's ratio at nano scale under-provisions experts.
DEEPSEEK_MOE_INTER_RATIO = 1.5    # Nano-scale adjusted (V3 original was 0.375 at 7168h)
DEEPSEEK_V3_MOE_INTER_RATIO = 0.375  # V3 original — kept for reference / future 7B+ scales

# MoE topology constants (promoted from MoEConfig)
# Nano-scale: 16 experts top-2 (was 64/8 at V3 7168h scale)
# Group routing: N_GROUP=4, TOPK_GROUP=4 → all groups always selected → pure top-k routing.
# This disables EP-style group constraint which is only beneficial with expert parallelism
# across ≥8 GPUs. At nano scale (1-8 GPUs), pure top-k is simpler and more stable.
N_ROUTED_EXPERTS = 16              # Nano scale (V3: 64)
NUM_EXPERTS_PER_TOK = 2            # Active experts per token — top-k (V3: 8, ratio same: 2/16=8/64=12.5%)
N_SHARED_EXPERTS = 2               # Always-active shared experts (unchanged)
N_GROUP = 4                        # Expert groups (V3: 8; TOPK_GROUP=N_GROUP → pure top-k, no EP constraint)
TOPK_GROUP = 4                     # All groups selected (V3: 4; =N_GROUP here disables group gating)
ROUTED_SCALING_FACTOR = 2.5        # DeepSeek V3 empirical value — kept (scale-invariant normalization factor)

# Training constants
DEEPSEEK_GAMMA = 0.001             # Load balancing bias update rate
DEEPSEEK_GAMMA_FREEZE_RATIO = 0.95 # 14.3T/14.8T ~ 0.966 -> 0.95 conservative
DEEPSEEK_BETA2 = 0.95              # Adam beta2 (not 0.999)
DEEPSEEK_GRAD_CLIP = 1.0           # Global gradient clipping

# MTP constants (promoted from MTPConfig)
DEEPSEEK_MTP_LAMBDA_INITIAL = 0.3     # Early training MTP weight
DEEPSEEK_MTP_LAMBDA_FINAL = 0.1       # Late training MTP weight
DEEPSEEK_MTP_TRANSITION_RATIO = 0.60  # Switch at 60% of training
DEEPSEEK_MTP_LOSS_DECAY = 0.8         # Weight decay for deeper predictions


# ============================================================================
# Main Config
# ============================================================================

@dataclass
class NanoSeekConfig:
    """Complete NanoSeek configuration — one flat dataclass.

    Three validated scales:
    - Anchor (768h):    ~175M active / ~730M total,   2.1B tokens,  muP anchor
    - Ablation (1280h): ~410M active / ~1.95B total,  8.2B tokens,  experiments
    - 1B (2048h):       ~1.08B active / ~4.75B total, 22B tokens,   graduation
    """

    # === Required ===
    hidden_size: int                  # 768, 1280, or 2048
    num_heads: int                    # hidden_size / 128

    # === Scale metadata ===
    scale_name: str = "custom"
    total_tokens: int = 0

    # === Core architecture ===
    vocab_size: int = 32768
    num_layers: int = 16
    max_position_embeddings: int = 4096
    use_bias: bool = False
    rms_norm_eps: float = 1e-6
    hidden_act: str = "swiglu"
    tie_word_embeddings: bool = False
    gradient_checkpointing: bool = True
    attention_dropout: float = 0.0
    logit_softcap: float = 30.0       # Gemma 2 style

    # === MLA head dims (defaults = DeepSeek V3, overridable for tests) ===
    _qk_nope_head_dim: int = DEEPSEEK_QK_NOPE_HEAD_DIM
    _qk_rope_head_dim: int = DEEPSEEK_QK_ROPE_HEAD_DIM
    _v_head_dim: int = DEEPSEEK_V_HEAD_DIM
    _q_lora_rank_override: Optional[int] = None
    _kv_lora_rank_override: Optional[int] = None

    # === MoE topology (defaults = DeepSeek V3, overridable for tests) ===
    n_routed_experts: int = N_ROUTED_EXPERTS
    num_experts_per_tok: int = NUM_EXPERTS_PER_TOK
    n_shared_experts: int = N_SHARED_EXPERTS
    n_group: int = N_GROUP
    topk_group: int = TOPK_GROUP
    _moe_intermediate_size_override: Optional[int] = None  # Set to bypass ratio calc

    # === Ablation knobs (overridable per-run) ===
    first_k_dense_replace: int = 1    # Layer 0 uses dense FFN
    disable_shared_experts: bool = False
    num_mtp_modules: int = 1
    seq_aux_loss_alpha: float = 0.0001
    # When True: aux loss has gradient (Switch-Transformer style, --aux-loss-type classic).
    # When False (default): aux loss is monitoring-only, zero gradient (bias mode).
    use_classic_aux_loss: bool = False
    routed_scaling_factor: float = ROUTED_SCALING_FACTOR
    scoring_func: str = "sigmoid"
    norm_topk_prob: bool = True

    # === RoPE (no YaRN for Phase 1) ===
    rope_theta: float = 10000.0

    # === Training ===
    sequence_length: int = 4096
    global_batch_size: int = 128
    learning_rate: float = 3e-4
    lr_min: float = 3e-5
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = DEEPSEEK_BETA2
    max_grad_norm: float = DEEPSEEK_GRAD_CLIP
    warmup_steps: int = 1000
    constant_phase_ratio: float = 0.70
    cosine_decay_end_ratio: float = 0.95
    dtype: str = "bfloat16"
    log_every_steps: int = 10
    eval_every_steps: int = 500
    checkpoint_every_steps: int = 1000

    # === Computed (set in __post_init__) ===
    intermediate_size: Optional[int] = None

    def __post_init__(self):
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        head_dim = self.hidden_size // self.num_heads
        if head_dim != 128:
            import warnings
            warnings.warn(
                f"Non-standard head_dim={head_dim} (DeepSeek uses 128). "
                f"OK for tests but production should use head_dim=128."
            )
        if self.intermediate_size is None:
            self.intermediate_size = int(self.hidden_size * DEEPSEEK_DENSE_FFN_RATIO)

    # ------------------------------------------------------------------
    # Derived dimensions
    # ------------------------------------------------------------------

    @property
    def q_lora_rank(self) -> int:
        if self._q_lora_rank_override is not None:
            return self._q_lora_rank_override
        return int(self.hidden_size * DEEPSEEK_Q_LORA_RATIO)

    @property
    def kv_lora_rank(self) -> int:
        if self._kv_lora_rank_override is not None:
            return self._kv_lora_rank_override
        return int(self.hidden_size * DEEPSEEK_KV_LORA_RATIO)

    @property
    def moe_intermediate_size(self) -> int:
        if self._moe_intermediate_size_override is not None:
            return self._moe_intermediate_size_override
        v = int(self.hidden_size * DEEPSEEK_MOE_INTER_RATIO)
        return (v // 32) * 32

    @property
    def tokens_per_step(self) -> int:
        return self.global_batch_size * self.sequence_length

    @property
    def total_steps(self) -> int:
        return self.total_tokens // self.tokens_per_step if self.tokens_per_step > 0 else 0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def moe_layer_indices(self) -> List[int]:
        return list(range(self.first_k_dense_replace, self.num_layers))

    @property
    def dense_layer_indices(self) -> List[int]:
        return list(range(self.first_k_dense_replace))

    def get_dtype(self) -> torch.dtype:
        return {"float32": torch.float32, "float16": torch.float16,
                "bfloat16": torch.bfloat16}.get(self.dtype, torch.bfloat16)

    # ------------------------------------------------------------------
    # Namespace accessors — backward compat with model.py
    # (model.py does config.mla.q_lora_rank, config.moe.n_routed_experts, etc.)
    # ------------------------------------------------------------------

    @property
    def mla(self):
        """Namespace for MLA params."""
        return SimpleNamespace(
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self._qk_nope_head_dim,
            qk_rope_head_dim=self._qk_rope_head_dim,
            v_head_dim=self._v_head_dim,
            rope_theta=self.rope_theta,
            rope_scaling_factor=1.0,
            original_max_position_embeddings=self.max_position_embeddings,
            mscale=1.0,
            q_head_dim=self._qk_nope_head_dim + self._qk_rope_head_dim,
            kv_cache_dim_per_layer=self.kv_lora_rank + self._qk_rope_head_dim,
        )

    @property
    def moe(self):
        """Namespace for MoE params."""
        return SimpleNamespace(
            n_routed_experts=self.n_routed_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            n_shared_experts=self.n_shared_experts,
            moe_intermediate_size=self.moe_intermediate_size,
            n_group=self.n_group,
            topk_group=self.topk_group,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            norm_topk_prob=self.norm_topk_prob,
            gamma=DEEPSEEK_GAMMA,
            gamma_freeze_ratio=DEEPSEEK_GAMMA_FREEZE_RATIO,
            seq_aux_loss_alpha=self.seq_aux_loss_alpha,
            use_classic_aux_loss=self.use_classic_aux_loss,
            first_k_dense_replace=self.first_k_dense_replace,
            disable_shared_experts=self.disable_shared_experts,
            experts_per_group=self.n_routed_experts // self.n_group,
            activation_ratio=self.num_experts_per_tok / self.n_routed_experts,
            shared_inter_dim=self.n_shared_experts * self.moe_intermediate_size,
        )

    @property
    def mtp(self):
        """Namespace for MTP params."""
        return SimpleNamespace(
            num_mtp_modules=self.num_mtp_modules,
            mtp_loss_weight_initial=DEEPSEEK_MTP_LAMBDA_INITIAL,
            mtp_loss_weight_final=DEEPSEEK_MTP_LAMBDA_FINAL,
            mtp_loss_transition_ratio=DEEPSEEK_MTP_TRANSITION_RATIO,
            mtp_loss_decay=DEEPSEEK_MTP_LOSS_DECAY,
            mtp_num_heads=self.num_heads,
            mtp_hidden_size=self.hidden_size,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        return {
            "scale_name": self.scale_name,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "vocab_size": self.vocab_size,
            "total_tokens": self.total_tokens,
            "sequence_length": self.sequence_length,
            "global_batch_size": self.global_batch_size,
            "learning_rate": self.learning_rate,
            "intermediate_size": self.intermediate_size,
            "mla": {
                "q_lora_rank": self.mla.q_lora_rank,
                "kv_lora_rank": self.mla.kv_lora_rank,
                "qk_nope_head_dim": self.mla.qk_nope_head_dim,
                "qk_rope_head_dim": self.mla.qk_rope_head_dim,
                "v_head_dim": self.mla.v_head_dim,
            },
            "moe": {
                "n_routed_experts": self.moe.n_routed_experts,
                "num_experts_per_tok": self.moe.num_experts_per_tok,
                "n_shared_experts": self.moe.n_shared_experts,
                "moe_intermediate_size": self.moe.moe_intermediate_size,
                "gamma": self.moe.gamma,
                "gamma_freeze_ratio": self.moe.gamma_freeze_ratio,
            },
        }


# ============================================================================
# Config Registry
# ============================================================================

CONFIG_REGISTRY = {
    "anchor": lambda: NanoSeekConfig(
        scale_name="anchor",
        hidden_size=768,
        num_heads=6,
        total_tokens=2_097_152_000,
        global_batch_size=64,
        warmup_steps=200,
    ),
    "ablation": lambda: NanoSeekConfig(
        scale_name="ablation",
        hidden_size=1280,
        num_heads=10,
        total_tokens=8_200_000_000,
        global_batch_size=128,
        warmup_steps=500,
        eval_every_steps=250,
        checkpoint_every_steps=500,
    ),
    "1b": lambda: NanoSeekConfig(
        scale_name="1b",
        hidden_size=2048,
        num_heads=16,
        total_tokens=22_000_000_000,
        global_batch_size=128,
        warmup_steps=1000,
        eval_every_steps=500,
        checkpoint_every_steps=2000,
    ),
}


NANOSEEK_ASPECT_RATIO = 80  # d16 × 80 = 1280 (ablation scale)


def create_from_depth(depth: int, aspect_ratio: int = NANOSEEK_ASPECT_RATIO) -> NanoSeekConfig:
    """Create config from depth alone. Width = ceil(depth × aspect_ratio / 128) × 128.

    d16 with aspect_ratio=80 reproduces the ablation config exactly (H=1280, 10 heads).
    Training tokens and batch size are NOT set here — pre_train.py auto-computes them
    from scaling params via Power Lines paper and target_param_data_ratio.

    Depth ladder (aspect_ratio=80):
        d12: H=1024, 8 heads, ~239M active / ~965M total
        d14: H=1152, 9 heads, ~327M active / ~1.41B total
        d16: H=1280, 10 heads, ~437M active / ~1.99B total  (PRIMARY)
        d18: H=1536, 12 heads, ~671M active / ~3.20B total
        d20: H=1664, 13 heads, ~851M active / ~4.17B total
        d24: H=1920, 15 heads, ~1.30B active / ~6.65B total
    """
    base_dim = depth * aspect_ratio
    hidden_size = ((base_dim + 127) // 128) * 128  # round UP to 128 multiple
    num_heads = hidden_size // 128
    warmup_steps = max(200, depth * 30)

    return NanoSeekConfig(
        scale_name=f"d{depth}",
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=depth,
        total_tokens=0,          # auto-computed by pre_train.py
        global_batch_size=0,     # 0 signals "auto-compute"
        warmup_steps=warmup_steps,
    )


def get_config(scale: str) -> NanoSeekConfig:
    """Get config by scale name: 'anchor', 'ablation', '1b', or 'd16' style."""
    if scale in CONFIG_REGISTRY:
        return CONFIG_REGISTRY[scale]()
    if scale.startswith("d") and scale[1:].isdigit():
        return create_from_depth(int(scale[1:]))
    raise ValueError(f"Unknown scale: {scale}. Choose from {list(CONFIG_REGISTRY.keys())} or 'd<N>'")


# ============================================================================
# muP Learning Rate Scaling
# ============================================================================

def compute_mup_lr(
    base_lr: float,
    base_hidden: int,
    target_hidden: int,
    base_batch: int = 64,
    target_batch: int = 128,
) -> float:
    """muP scaling: LR_scaled = LR_base * sqrt(B/B_ref) * (w_ref / w)."""
    width_factor = base_hidden / target_hidden
    batch_factor = (target_batch / base_batch) ** 0.5
    return base_lr * width_factor * batch_factor


def get_optimizer_lrs(
    config: NanoSeekConfig,
    matrix_lr_base: float,
    embedding_lr_base: float,
) -> Dict[str, float]:
    """Get all optimizer learning rates for a config.

    Returns dict with keys: matrix, embedding, lm_head, router, norm
    """
    anchor_hidden = 768
    anchor_batch = 64

    matrix_lr = compute_mup_lr(
        matrix_lr_base, anchor_hidden, config.hidden_size,
        anchor_batch, config.global_batch_size,
    )
    batch_factor = (config.global_batch_size / anchor_batch) ** 0.5
    embedding_lr = embedding_lr_base * batch_factor
    lm_head_lr = embedding_lr * 0.1

    return {
        "matrix": matrix_lr,
        "embedding": embedding_lr,
        "lm_head": lm_head_lr,
        "router": 3e-4,
        "norm": 3e-4,
    }


# ============================================================================
# CLI Demo
# ============================================================================

if __name__ == "__main__":
    print("NanoSeek Configuration Summary")
    print("=" * 60)

    for scale_name in ["anchor", "ablation", "1b"]:
        config = get_config(scale_name)

        print(f"\n{scale_name.upper()} Scale:")
        print(f"  Hidden: {config.hidden_size}, Heads: {config.num_heads}, Layers: {config.num_layers}")
        print(f"  MLA: q_lora={config.mla.q_lora_rank}, kv_lora={config.mla.kv_lora_rank}")
        print(f"  MoE: {config.moe.n_routed_experts} experts, top-{config.moe.num_experts_per_tok}, inter={config.moe.moe_intermediate_size}")
        print(f"  Tokens: {config.total_tokens/1e9:.1f}B, Steps: ~{config.total_steps:,}")

        standard_kv = 2 * config.num_heads * config.head_dim
        mla_kv = config.mla.kv_cache_dim_per_layer
        print(f"  KV compression: {standard_kv} -> {mla_kv} dims ({standard_kv/mla_kv:.1f}x)")

    print("\n" + "=" * 60)
    print("muP LR Scaling Example (base_matrix_lr=0.02, base_embedding_lr=0.3):")

    for scale_name in ["anchor", "ablation", "1b"]:
        config = get_config(scale_name)
        lrs = get_optimizer_lrs(config, matrix_lr_base=0.02, embedding_lr_base=0.3)
        print(f"\n  {scale_name}:")
        print(f"    matrix: {lrs['matrix']:.5f}, embedding: {lrs['embedding']:.4f}")
        print(f"    lm_head: {lrs['lm_head']:.5f}, router: {lrs['router']:.4f}")
