# NanoSeek Configuration - Nano-scale DeepSeek V3.2 Implementation
#
# This module defines the complete configuration hierarchy for NanoSeek,
# preserving all architectural innovations from DeepSeek V3.2 at nano scale.
#
# PRIMARY CONFIG: NanoSeek-1B
# =============================
# - Active parameters: ~1.13B (embeddings + MLA + shared + 8 routed experts)
# - Total parameters: ~4.9B (all 64 routed experts)
# - Expansion ratio: ~4.3× (total/active)
# - Training: 22B tokens (~20× active params, Chinchilla optimal)
# - Hardware: 8×H100, ~14 hours, ~$275-350

from dataclasses import dataclass, field
from typing import List, Optional, Literal


# ============================================================================
# Multi-Phase Training Configuration (DeepSeek Methodology)
# ============================================================================
#
# NanoSeek follows DeepSeek's proven training pipeline:
#
#   Phase 1: Dense MLA Pre-training (80% of tokens)
#   ├── Context: 4096 tokens
#   ├── Attention: Dense (full attention)
#   ├── DSA: Disabled (indexer trains via auxiliary loss)
#   ├── YaRN: Disabled (native positions)
#   └── Goal: Learn strong representations
#
#   Phase 2: Sparse DSA Fine-tuning (20% of tokens)
#   ├── Context: 8192 tokens (2x Phase 1)
#   ├── Attention: Sparse (DSA active)
#   ├── DSA: Enabled (indexer selects top-k)
#   ├── YaRN: Enabled (interpolate to 8K)
#   └── Goal: Long-context adaptation
#
# This "train short, extend long" approach maximizes gradient updates
# while enabling efficient long-context inference.
#
# ============================================================================


@dataclass
class TrainingPhaseConfig:
    """
    Configuration for a single training phase.

    NanoSeek uses multi-phase training following DeepSeek methodology:
    - Phase 1: Dense attention at 4K context (base training)
    - Phase 2: Sparse attention at 8K context (context extension)

    Each phase can have different context lengths, batch sizes,
    learning rates, and attention modes.
    """
    # Phase identification
    name: str = "phase1_dense"

    # Context configuration
    sequence_length: int = 4096
    global_batch_size: int = 128

    # Token budget for this phase (as fraction of total)
    token_fraction: float = 0.8  # 80% of total tokens

    # Learning rate (can reduce for later phases)
    learning_rate: float = 3e-4
    lr_min: float = 3e-5

    # DSA configuration for this phase
    dsa_enabled: bool = False
    dsa_activation_threshold: int = 4096

    # YaRN configuration for this phase
    yarn_enabled: bool = False

    # Warmup (only for first phase typically)
    warmup_steps: int = 1000


# Default training phases (can be customized via get_training_phases())
# These are baseline values; actual phases are generated from config
PHASE1_DENSE = TrainingPhaseConfig(
    name="phase1_dense_mla",
    sequence_length=4096,
    global_batch_size=128,
    token_fraction=0.8,           # 80% of tokens
    learning_rate=3e-4,
    lr_min=3e-5,
    dsa_enabled=False,            # Dense attention
    dsa_activation_threshold=4096,
    yarn_enabled=False,           # Native 4K positions
    warmup_steps=1000,
)

PHASE2_SPARSE = TrainingPhaseConfig(
    name="phase2_sparse_dsa",
    sequence_length=8192,         # 2x Phase 1 context
    global_batch_size=64,         # Reduced (2x context = 2x memory)
    token_fraction=0.2,           # 20% of tokens
    learning_rate=1e-4,           # Lower LR for fine-tuning
    lr_min=1e-5,
    dsa_enabled=True,             # Sparse attention active
    dsa_activation_threshold=4096,# Sparse for seq > 4K
    yarn_enabled=True,            # Extend positions via YaRN
    warmup_steps=100,             # Brief warmup for phase transition
)


# ============================================================================
# DeepSeek V3.2 Production Configuration Additions
# ============================================================================

@dataclass
class YaRNConfig:
    """
    YaRN (Yet another RoPE extensioN) configuration for long context.

    DeepSeek V3.2 uses YaRN with proper correction factors for context
    extension beyond original training length (4096 → 128K+).

    NanoSeek Strategy (following DeepSeek):
    - Train at 4K context (optimal for 1B model, $100 budget)
    - YaRN extends to 16K-32K at inference time
    - DSA activates for extended context, making it compute-efficient

    The key insight: Train short, infer long.
    - Training: More gradient updates, fits natural document lengths
    - Inference: YaRN + DSA enables efficient long context

    Reference: DeepSeek-V3 Technical Report
    """
    # Original training sequence length
    original_seq_len: int = 4096           # NanoSeek trains at 4K

    # Scaling factor for extended context (inference-time)
    # factor=8 enables 32K inference from 4K training
    rope_factor: float = 8.0               # 4K → 32K extension capability

    # Correction factors (from official V3.2 implementation)
    # These control the frequency interpolation ramp
    beta_fast: int = 32  # Fast correction dimension
    beta_slow: int = 1   # Slow correction dimension

    # Attention scaling for extended context
    # Compensates for attention score distribution changes at longer context
    mscale: float = 1.0

    # Enable YaRN
    # OFF during training (native 4K positions)
    # ON at inference for context extension beyond 4K
    enabled: bool = False                  # Enable at inference time
    
    @property
    def rope_scaling_factor_computed(self) -> float:
        """
        Compute RoPE scaling factor based on enabled status.
        
        When YaRN is enabled, uses rope_factor for context extension.
        When disabled, uses 1.0 (native positions).
        
        This property should be used to set MLAConfig.rope_scaling_factor
        during model initialization.
        """
        return self.rope_factor if self.enabled else 1.0


@dataclass
class FP8Config:
    """
    FP8 (8-bit floating point) training and inference configuration.

    Requires NVIDIA H100/H200 with Transformer Engine for native FP8.
    Provides ~2x memory reduction and ~1.5x speed improvement.

    Reference: DeepSeek-V3 uses FP8 for both weights and activations
    """
    # Enable FP8 (requires H100/H200)
    enabled: bool = False

    # Computation format
    dtype: Literal["bf16", "fp8"] = "bf16"

    # Block size for block-based quantization
    block_size: int = 128

    # Scale format for quantization
    scale_fmt: Optional[str] = None

    # Use delayed scaling for better training stability
    use_delayed_scaling: bool = True


@dataclass
class ParallelConfig:
    """
    Distributed parallelism configuration.

    DeepSeek V3.2 supports:
    - Tensor Parallelism (TP): Split attention/FFN across GPUs
    - Pipeline Parallelism (PP): Split layers across GPUs
    - Expert Parallelism (EP): Distribute MoE experts
    - Data Parallelism (DP): Replicate model for different batches

    Reference: DeepSeek-V3.2-Exp/inference/model.py:92-269
    """
    # World size (total GPUs)
    world_size: int = 1

    # Tensor parallelism (split within layers)
    tensor_parallel_size: int = 1

    # Pipeline parallelism (split across layers)
    pipeline_parallel_size: int = 1

    # Expert parallelism (distribute experts)
    expert_parallel_size: int = 1

    # Data parallelism (replicate model)
    # Computed: world_size / (tp * pp * ep)

    @property
    def data_parallel_size(self) -> int:
        """Compute data parallel size from other parallelism."""
        return self.world_size // (
            self.tensor_parallel_size *
            self.pipeline_parallel_size *
            self.expert_parallel_size
        )


@dataclass
class MLAConfig:
    """
    Multi-head Latent Attention configuration.

    MLA compresses KV cache by projecting to a low-rank latent space.
    This achieves ~23x KV cache reduction compared to standard MHA.

    Key insight: The RoPE component is SHARED across all heads in the
    compressed space, enabling massive memory savings during inference.

    DeepSeek V3 Ratios (scale-independent):
    - q_lora_rank / hidden_size ≈ 0.21
    - kv_lora_rank / hidden_size ≈ 0.07

    Reference: DeepSeek-V2 Technical Report (Section 3.1)
    """
    # Query compression (q_lora_rank / hidden_size ≈ 0.21)
    # For hidden=2048: 0.21 × 2048 = 430
    q_lora_rank: int = 430

    # KV compression (kv_lora_rank / hidden_size ≈ 0.07)
    # For hidden=2048: 0.07 × 2048 = 143
    # This is the key parameter for KV cache reduction
    kv_lora_rank: int = 143

    # Head dimensions — FIXED CONSTANTS across all model sizes (DeepSeek family invariant)
    # V2-Lite (2048h), V2 (5120h), V3 (7168h) all use identical 128/64/128
    # These are NOT ratios of hidden_size — they are absolute architectural constants
    qk_nope_head_dim: int = 128  # Non-positional component (DeepSeek V2/V2-Lite/V3: 128)
    qk_rope_head_dim: int = 64   # RoPE component, SHARED across heads (DeepSeek: 64)
    v_head_dim: int = 128        # Value dimension (DeepSeek V2/V2-Lite/V3: 128)

    # RoPE configuration
    rope_theta: float = 10000.0
    rope_scaling_type: Optional[str] = None  # "yarn" for context extension
    rope_scaling_factor: float = 1.0         # Set from YaRNConfig.rope_scaling_factor_computed

    # YaRN context extension (matches official V3.2)
    # Reference: DeepSeek-V3.2-Exp/inference/model.py:535-537
    original_max_position_embeddings: int = 4096  # Training context length
    mscale: float = 1.0  # Magnitude scaling for attention after YaRN (set from YaRNConfig)

    @property
    def q_head_dim(self) -> int:
        """Total query head dimension = nope + rope."""
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def kv_cache_dim_per_layer(self) -> int:
        """
        KV cache dimension per layer per token.

        Standard MHA: 2 * num_heads * head_dim
        MLA: kv_lora_rank + qk_rope_head_dim (RoPE is shared!)

        Example for NanoSeek-1B (hidden=2048, num_heads=16):
        - Standard: 2 * 16 * 128 = 4096
        - MLA: 143 + 64 = 207
        - Compression: ~20x
        """
        return self.kv_lora_rank + self.qk_rope_head_dim


@dataclass
class MoEConfig:
    """
    Mixture of Experts configuration with DeepSeek V3 load balancing.

    DeepSeek V3 innovations:
    1. Sigmoid scoring (not softmax) - better gradient flow
    2. Group-based routing - reduces communication overhead
    3. Auxiliary-loss-free TOKEN-level balancing via dynamic bias
    4. Complementary SEQUENCE-level auxiliary loss (α=0.0001)
    5. Shared expert - captures common patterns

    Reference: DeepSeek-V3 Technical Report (Section 3.2)
    """
    # Expert configuration (aligned with 1B config)
    # 64 experts, 8 active = 12.5% activation (DeepSeek standard)
    n_routed_experts: int = 64           # Total number of routed experts
    num_experts_per_tok: int = 8         # Active experts per token (DeepSeek's optimal k)
    n_shared_experts: int = 2            # Always-active shared experts

    # Per-expert FFN dimension - sized for fine-grained experts
    # For hidden=2048: 768/2048 = 0.375 granularity
    moe_intermediate_size: int = 768

    # Shared expert FFN dimension (BUG FIX #5)
    # DeepSeek V3 uses 1 shared MLP with inter_dim = n_shared_experts * moe_inter_dim
    # Default None → computed as n_shared_experts * moe_intermediate_size
    shared_inter_dim: Optional[int] = None

    # Group-based routing (reduces communication)
    n_group: int = 8                     # Number of expert groups (64/8 = 8 per group)
    topk_group: int = 4                  # Route to half the groups

    # Scoring function
    scoring_func: Literal["sigmoid", "softmax"] = "sigmoid"
    routed_scaling_factor: float = 2.5   # DeepSeek V3 empirical value (NOT sqrt(K)=2.83)
    
    # Top-k probability normalization (CRITICAL BUG FIX)
    # Must be applied in Gate.forward() after top-k selection:
    # if norm_topk_prob: weights /= weights.sum(dim=-1, keepdim=True)
    # This ensures router outputs sum to 1.0 for proper expert weighting
    norm_topk_prob: bool = True          # Normalize top-k probabilities

    # ================================================================
    # DeepSeek V3 Load Balancing (Hybrid Approach)
    # ================================================================
    # 1. Token-level: Bias-based (aux-loss-free!)
    gamma: float = 0.001                 # Bias update rate
    # Freeze bias at this fraction of total training
    # DeepSeek V3: 14.3T / 14.8T ≈ 0.966 → use 0.95 as conservative lower bound
    # Rationale: Bias should be active nearly the whole training run (V3 paper confirms)
    gamma_freeze_ratio: float = 0.95     # Freeze bias at 95% of training

    # 2. Sequence-level: Small auxiliary loss (V3 addition)
    seq_aux_loss_alpha: float = 0.0001   # Very small! (V3 default)

    # Ablation flags
    disable_shared_experts: bool = False  # Ablation: zero out shared expert output

    # Which layers use MoE (layers before this use dense FFN)
    first_k_dense_replace: int = 1       # Layer 0 uses dense FFN (V2-Lite precedent at this scale)

    @property
    def experts_per_group(self) -> int:
        """Number of experts per group."""
        return self.n_routed_experts // self.n_group


@dataclass
class MTPConfig:
    """
    Multi-Token Prediction configuration - DeepSeek V3 Architecture.

    V3 MTP uses concatenation + projection (not addition!):
    h'ᵢᵏ = Mₖ [RMSNorm(hᵢᵏ⁻¹) ; RMSNorm(Emb(tᵢ₊ₖ))]

    Key innovations:
    1. Concatenation-based fusion (not additive)
    2. Dynamic loss weight schedule (λ=0.3 → 0.1)
    3. Shared embeddings with main model

    Reference: DeepSeek-V3 Technical Report (Section 3.3)

    NanoSeek Adjustments:
    - Transition at 60% of training (ratio-based for scale independence)
    - Same weight schedule (0.3 → 0.1)
    """
    # Number of MTP modules (each predicts one additional token)
    num_mtp_modules: int = 1             # Predict next 2 tokens total

    # ================================================================
    # DeepSeek V3 Loss Schedule (Ratio-Based for Scale Independence)
    # ================================================================
    # V3 original: λ = 0.3 for first ~67% of training, then 0.1
    # (10T transition / 14.8T total ≈ 0.67)
    # Using ratio makes this config portable across any training budget
    mtp_loss_weight_initial: float = 0.3   # V3: λ=0.3 early training
    mtp_loss_weight_final: float = 0.1     # V3: λ=0.1 late training
    mtp_loss_transition_ratio: float = 0.60  # Switch at 60% of training

    # Weight decay for deeper predictions (module 1 < module 0)
    mtp_loss_decay: float = 0.8          # Weight decay for further predictions

    # MTP module architecture (lightweight transformer block)
    mtp_hidden_size: Optional[int] = None  # Default: same as model hidden
    mtp_num_heads: int = 16              # Same as main model (V3: MTP uses full transformer block)

    # Speculative decoding configuration
    speculative_draft_tokens: int = 2    # Draft tokens per speculation
    speculative_temperature: float = 0.0  # Greedy for speculation

    @property
    def total_predictions(self) -> int:
        """Total number of tokens predicted (1 main + N MTP)."""
        return 1 + self.num_mtp_modules

    @property
    def mtp_loss_weight(self) -> float:
        """
        Default MTP loss weight (for backward compatibility).

        Note: The actual weight should be computed dynamically using
        model.get_mtp_loss_weight() based on tokens_processed.
        This property returns the initial weight for display purposes.
        """
        return self.mtp_loss_weight_initial


@dataclass
class SparseAttentionConfig:
    """
    DeepSeek Sparse Attention (DSA) configuration.

    DSA reduces attention complexity from O(L²) to O(Lk) where k << L.

    Key innovations from DeepSeek V3.2:
    1. Lightning Indexer: Multi-head ReLU scoring for token selection
    2. Token-level selection: Direct top-k tokens (not block-based)
    3. Indexer training: KL-divergence alignment with main attention

    NanoSeek Training Strategy (following DeepSeek methodology):
    - Train at 4K with DENSE attention (indexer learns via aux loss)
    - DSA activates only at INFERENCE for context > 4K
    - YaRN extends to 16K-32K at inference, DSA makes it efficient

    This approach maximizes gradient updates during training while
    enabling efficient long-context inference.

    Reference: DeepSeek-V3 Technical Report (Section on Sparse Attention)
    https://arxiv.org/html/2512.02556v1
    """
    # Enable/disable sparse attention
    # OFF during training (dense attention learns better patterns)
    # ON at inference for long context (>4K) efficiency
    enabled: bool = False                  # Train dense, infer sparse

    # ========================================================================
    # Token Selection (DSA uses token-level, not block-level)
    # ========================================================================
    topk_tokens: int = 2048                # Tokens to select per query (V3 default)

    # Activation control - DSA kicks in above this threshold
    # Set to training length so DSA only activates for extended inference
    activation_threshold: int = 4096       # At inference: >4K uses sparse

    # ========================================================================
    # Lightning Indexer (Multi-head ReLU architecture)
    # Trains via auxiliary loss even when DSA is disabled
    # ========================================================================
    indexer_num_heads: int = 4             # Number of indexer heads (H^I)
    indexer_head_dim: int = 64             # Dimension per indexer head (d^I)

    # ========================================================================
    # Training Configuration
    # ========================================================================
    # Indexer trains via KL-divergence with dense attention throughout
    # No "warm-up" needed - we train dense and use indexer at inference
    dense_warmup_steps: int = 0            # Always dense during training
    indexer_loss_weight: float = 0.01      # Small weight, auxiliary objective

    # ========================================================================
    # Sliding window for inference (guarantees local context)
    # ========================================================================
    use_sliding_window: bool = True        # Always attend to recent tokens
    sliding_window_size: int = 512         # Local context window

    @property
    def sparse_tokens_per_query(self) -> int:
        """Total tokens attended per query in sparse mode."""
        if self.use_sliding_window:
            return self.topk_tokens + self.sliding_window_size
        return self.topk_tokens


@dataclass
class NanoSeekConfig:
    """
    Complete NanoSeek-1B configuration following empirical scaling laws.

    SCALING LAW COMPLIANCE:
    =======================
    This configuration follows OLMoE/LLaMA depth-width scaling:
    - d/L = 2048/16 = 128 (matches OLMoE-1B, LLaMA-7B scale)
    - Chinchilla optimal: D = 20 × N (22B tokens for 1.08B active)

    DeepSeek V3 architectural innovations:
    - MLA: ~23x KV cache compression
    - MoE: ~4.4x parameter capacity (4.75B total)
    - MTP: 1.4x inference speedup
    - Aux-loss-free: Better training stability

    PARAMETER SUMMARY:
    ==================
    - Active: ~1.08B (embeddings + MLA + shared + 8 routed experts)
    - Total: ~4.87B (all 64 routed experts)
    - Expansion: ~4.5×

    Training target: 22B tokens on 8×H100, ~14 hours, ~$300 (Chinchilla optimal)
    """

    # ========================================================================
    # Core Architecture (OLMoE-aligned d/L = 128)
    # ========================================================================
    # Depth-Width Ratio: d/L = 2048/16 = 128
    # This follows OLMoE-1B and scales well for MoE architectures
    vocab_size: int = 32768              # 32K vocab — reduces embedding tax (see RESEARCH_ENGINEER.md §15)
    hidden_size: int = 2048              # d/L = 2048/16 = 128
    num_layers: int = 16                 # Optimal depth for 700M-1B active
    num_heads: int = 16                  # head_dim = 2048/16 = 128 (standard)

    # Dense FFN (for first_k_dense_replace layers)
    intermediate_size: int = 5243        # 2.56 × 2048 (DeepSeek ratio)

    # Activation and normalization
    hidden_act: str = "swiglu"           # SwiGLU (gate * up)
    rms_norm_eps: float = 1e-6
    use_bias: bool = False               # No bias in linear layers

    # Position encoding
    # Train at 4K, extend to 32K via YaRN at inference
    max_position_embeddings: int = 4096
    tie_word_embeddings: bool = False    # Separate input/output embeddings

    # Dropout
    attention_dropout: float = 0.0       # Attention dropout probability

    # Logit softcap — tanh squash to [-cap, cap] before CE loss
    # Prevents logit explosion from MoE expert specialization (Gemma 2 technique)
    logit_softcap: float = 30.0

    # ========================================================================
    # Component Configurations
    # ========================================================================
    mla: MLAConfig = field(default_factory=MLAConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    mtp: MTPConfig = field(default_factory=MTPConfig)
    sparse: SparseAttentionConfig = field(default_factory=SparseAttentionConfig)

    # ========================================================================
    # DeepSeek V3.2 Production Configurations
    # ========================================================================
    yarn: YaRNConfig = field(default_factory=YaRNConfig)
    fp8: FP8Config = field(default_factory=FP8Config)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)

    # ========================================================================
    # Training Configuration
    # ========================================================================
    # Batch and sequence
    # 4K context: optimal balance for MoE training
    global_batch_size: int = 128         # Standard batch size
    sequence_length: int = 4096          # 4K context, extend via YaRN at inference
    # tokens_per_step = 128 * 4096 = 524,288 (512K)

    # Optimizer (AdamW, DeepSeek style)
    learning_rate: float = 3e-4          # lr_max
    lr_min: float = 3e-5                 # 10% of max
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # LR Schedule (DeepSeek style: warmup → constant → cosine decay)
    warmup_steps: int = 1000
    constant_phase_ratio: float = 0.70   # 70% at max LR
    cosine_decay_end_ratio: float = 0.95 # Decay from 70% to 95%
    # Final 5% at lr_min

    # Training duration (Chinchilla optimal: 20 × 1.08B active params)
    total_tokens: int = 22_000_000_000   # 22B tokens (Chinchilla optimal)

    # Precision
    dtype: str = "bfloat16"              # Primary dtype
    use_fp8: bool = False                # FP8 optional for nano scale

    # Checkpointing and logging
    gradient_checkpointing: bool = True
    checkpoint_every_steps: int = 1000
    log_every_steps: int = 10
    eval_every_steps: int = 500

    # ========================================================================
    # Distributed Configuration (legacy - use parallel config instead)
    # ========================================================================
    # NOTE: These are kept for backward compatibility but parallel config is preferred
    # For 8xH100, NanoSeek-1B fits on single GPU with DP=8
    data_parallel_size: int = 8          # Use parallel.world_size instead
    # pipeline_parallel_size and expert_parallel_size moved to parallel config

    # ========================================================================
    # Derived Properties
    # ========================================================================
    @property
    def tokens_per_step(self) -> int:
        """Total tokens processed per optimization step."""
        return self.global_batch_size * self.sequence_length

    @property
    def total_steps(self) -> int:
        """Total training steps."""
        return self.total_tokens // self.tokens_per_step

    @property
    def warmup_tokens(self) -> int:
        """Tokens during warmup phase."""
        return self.warmup_steps * self.tokens_per_step

    @property
    def head_dim(self) -> int:
        """Per-head dimension for standard MHA comparison."""
        return self.hidden_size // self.num_heads

    @property
    def moe_layer_indices(self) -> List[int]:
        """Layer indices that use MoE (vs dense FFN)."""
        return list(range(self.moe.first_k_dense_replace, self.num_layers))

    @property
    def dense_layer_indices(self) -> List[int]:
        """Layer indices that use dense FFN."""
        return list(range(self.moe.first_k_dense_replace))

    @property
    def estimated_total_params(self) -> int:
        """
        Estimated total parameters (including all MoE experts).

        Breakdown for NanoSeek-1B (hidden=2048, layers=16):
        - Embeddings: 32768 × 2048 × 2 = 134M
        - MLA (16 layers): 16 × ~5M = 80M
        - Dense FFN (2 layers): 2 × 32M = 64M
        - Shared experts (14 MoE layers × 2): 14 × 2 × 4.7M = 132M
        - Routed experts (14 MoE layers × 64): 14 × 64 × 4.7M = 4.2B
        - Router + norms: ~10M
        Total: ~4.62B (134M less than 65K-vocab estimate due to 32K vocab)
        """
        # Embeddings (input + output if not tied)
        embed_params = self.vocab_size * self.hidden_size
        if not self.tie_word_embeddings:
            embed_params *= 2

        # Per dense layer
        dense_attn_params = self._estimate_mla_params()
        dense_ffn_params = 3 * self.hidden_size * self.intermediate_size  # SwiGLU
        dense_layer_params = dense_attn_params + dense_ffn_params
        total_dense = len(self.dense_layer_indices) * dense_layer_params

        # Per MoE layer
        moe_attn_params = dense_attn_params
        expert_params = 3 * self.hidden_size * self.moe.moe_intermediate_size
        shared_expert_params = expert_params * self.moe.n_shared_experts
        routed_experts_params = expert_params * self.moe.n_routed_experts
        router_params = self.hidden_size * self.moe.n_routed_experts
        moe_layer_params = moe_attn_params + shared_expert_params + routed_experts_params + router_params
        total_moe = len(self.moe_layer_indices) * moe_layer_params

        # Layer norms (input + post-attention per layer + final)
        norm_params = (self.num_layers * 2 + 1) * self.hidden_size

        # MTP
        mtp_params = self._estimate_mtp_params()

        return int(embed_params + total_dense + total_moe + norm_params + mtp_params)

    @property
    def estimated_active_params(self) -> int:
        """
        Estimated active parameters per forward pass.

        For NanoSeek-1B: ~0.94B active (embeddings + MLA + shared + k routed)

        Breakdown:
        - Embeddings: 134M (32K vocab × 2048 × 2)
        - MLA (16 layers): 80M
        - Dense FFN (2 layers): 64M
        - Shared experts (14 layers × 2): 132M
        - Active routed (14 layers × 8): 14 × 8 × 4.7M = 526M
        - Norms: ~2M
        Total: ~0.94B (accounting for shared weights with MTP)
        """
        # Embeddings (input + output)
        embed_params = self.vocab_size * self.hidden_size
        if not self.tie_word_embeddings:
            embed_params *= 2

        # All attention layers (MLA always active)
        attn_params = self.num_layers * self._estimate_mla_params()

        # Dense FFN layers
        dense_ffn_params = 3 * self.hidden_size * self.intermediate_size
        total_dense_ffn = len(self.dense_layer_indices) * dense_ffn_params

        # MoE layers (only active experts count)
        expert_params = 3 * self.hidden_size * self.moe.moe_intermediate_size
        active_routed = self.moe.num_experts_per_tok * expert_params
        shared = self.moe.n_shared_experts * expert_params
        router_params = self.hidden_size * self.moe.n_routed_experts  # Router is always used
        total_moe_active = len(self.moe_layer_indices) * (active_routed + shared + router_params)

        # Layer norms
        norm_params = (self.num_layers * 2 + 1) * self.hidden_size

        return int(embed_params + attn_params + total_dense_ffn + total_moe_active + norm_params)

    def _estimate_mla_params(self) -> int:
        """Estimate parameters per MLA layer."""
        # Down projections
        kv_down = self.hidden_size * self.mla.kv_lora_rank
        q_down = self.hidden_size * self.mla.q_lora_rank

        # Up projections
        k_up = self.mla.kv_lora_rank * (self.num_heads * self.mla.qk_nope_head_dim)
        v_up = self.mla.kv_lora_rank * (self.num_heads * self.mla.v_head_dim)

        # RoPE query projection (per head)
        q_rope = self.hidden_size * (self.num_heads * self.mla.qk_rope_head_dim)

        # RoPE key projection (SHARED across heads - key MLA innovation!)
        k_rope = self.hidden_size * self.mla.qk_rope_head_dim

        # Q up projection (nope components only)
        q_up = self.mla.q_lora_rank * (self.num_heads * self.mla.qk_nope_head_dim)

        # Output projection
        o_proj = self.num_heads * self.mla.v_head_dim * self.hidden_size

        # Layer norms (q_norm + kv_norm)
        norms = self.mla.q_lora_rank + self.mla.kv_lora_rank

        return kv_down + q_down + k_up + v_up + q_rope + k_rope + q_up + o_proj + norms

    def _estimate_mtp_params(self) -> int:
        """
        Estimate parameters for MTP modules.

        MTP module structure (Bug Fix #1 — no cross-attention):
        - hidden_norm, embed_norm, output_norm: RMSNorm(hidden_size)
        - concat_proj: Linear(2 * hidden_size, hidden_size, bias=False)
        - MTPTransformerBlock:
            - input_norm, post_attn_norm: RMSNorm(hidden_size)
            - attn: MLA with mtp_num_heads (same as main model per V3 paper)
            - ffn: Expert (SwiGLU dense FFN, same intermediate_size as dense layers)
        - embed_tokens, lm_head: shared with main model (not counted)
        """
        if self.mtp.num_mtp_modules == 0:
            return 0

        mtp_hidden = self.mtp.mtp_hidden_size or self.hidden_size
        mtp_heads = self.mtp.mtp_num_heads
        qk_head_dim = self.mla.qk_nope_head_dim + self.mla.qk_rope_head_dim

        per_module = (
            # 5 RMSNorm: hidden_norm, embed_norm, input_norm, post_attn_norm, output_norm
            5 * mtp_hidden +
            # concat_proj: Linear(2H → H)
            2 * mtp_hidden * mtp_hidden +
            # MLA: wq_a + q_norm + wq_b + wkv_a + kv_norm + wkv_b + wo
            mtp_hidden * self.mla.q_lora_rank +                                     # wq_a
            self.mla.q_lora_rank +                                                   # q_norm
            self.mla.q_lora_rank * mtp_heads * qk_head_dim +                        # wq_b
            mtp_hidden * (self.mla.kv_lora_rank + self.mla.qk_rope_head_dim) +      # wkv_a
            self.mla.kv_lora_rank +                                                  # kv_norm
            self.mla.kv_lora_rank * mtp_heads * (self.mla.qk_nope_head_dim + self.mla.v_head_dim) +  # wkv_b
            mtp_heads * self.mla.v_head_dim * mtp_hidden +                           # wo
            # SwiGLU FFN: 3 linear layers (same intermediate_size as dense layers)
            3 * mtp_hidden * self.intermediate_size
        )
        # LM head and embed_tokens are shared with main model, not counted
        return self.mtp.num_mtp_modules * per_module

    def validate(self) -> None:
        """Validate configuration consistency."""
        # Basic checks
        assert self.hidden_size % self.num_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"

        head_dim = self.hidden_size // self.num_heads

        # MoE checks
        assert self.moe.n_routed_experts % self.moe.n_group == 0, \
            f"n_routed_experts ({self.moe.n_routed_experts}) must be divisible by n_group ({self.moe.n_group})"
        assert self.moe.topk_group <= self.moe.n_group, \
            f"topk_group ({self.moe.topk_group}) must be <= n_group ({self.moe.n_group})"
        assert self.moe.num_experts_per_tok <= self.moe.n_routed_experts, \
            f"num_experts_per_tok ({self.moe.num_experts_per_tok}) must be <= n_routed_experts"
        assert self.moe.first_k_dense_replace <= self.num_layers, \
            f"first_k_dense_replace ({self.moe.first_k_dense_replace}) must be <= num_layers"

        # MLA checks
        assert self.mla.qk_rope_head_dim % 2 == 0, \
            f"qk_rope_head_dim ({self.mla.qk_rope_head_dim}) must be even for RoPE"

        # Validate MLA head dimensions are positive and even for RoPE
        # Note: qk_nope/qk_rope/v are model-independent constants (128/64/128)
        # following the DeepSeek family invariant, so they may exceed head_dim
        mla_q_head_dim = self.mla.qk_nope_head_dim + self.mla.qk_rope_head_dim
        assert mla_q_head_dim > 0, "MLA q_head_dim must be positive"
        assert self.mla.v_head_dim > 0, "MLA v_head_dim must be positive"

        # Validate LoRA ranks are reasonable fractions of hidden_size
        assert self.mla.q_lora_rank <= self.hidden_size, \
            f"q_lora_rank ({self.mla.q_lora_rank}) should be <= hidden_size ({self.hidden_size})"
        assert self.mla.kv_lora_rank <= self.hidden_size, \
            f"kv_lora_rank ({self.mla.kv_lora_rank}) should be <= hidden_size ({self.hidden_size})"

        # Training checks
        assert self.constant_phase_ratio < self.cosine_decay_end_ratio, \
            "constant_phase must end before cosine_decay_end"

        # ================================================================
        # Ratio-based hyperparameter validation (scale-independent!)
        # These must be between 0 and 1 to work with any training budget
        # ================================================================
        assert 0.0 < self.moe.gamma_freeze_ratio <= 1.0, \
            f"gamma_freeze_ratio ({self.moe.gamma_freeze_ratio}) must be in (0, 1]"
        assert 0.0 < self.mtp.mtp_loss_transition_ratio <= 1.0, \
            f"mtp_loss_transition_ratio ({self.mtp.mtp_loss_transition_ratio}) must be in (0, 1]"
        assert self.total_tokens > 0, \
            f"total_tokens ({self.total_tokens}) must be positive"

    def get_dtype(self):
        """Get torch dtype from string."""
        import torch
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.dtype, torch.bfloat16)

    def wire_yarn_config(self):
        """
        Wire YaRN configuration to MLA settings.
        
        This method should be called after initialization to ensure
        rope_scaling_factor and mscale are properly set from YaRNConfig.
        """
        # Wire rope scaling factor
        object.__setattr__(self.mla, 'rope_scaling_factor', self.yarn.rope_scaling_factor_computed)
        
        # Wire mscale
        object.__setattr__(self.mla, 'mscale', self.yarn.mscale)
        
        # Set rope scaling type if YaRN is enabled
        if self.yarn.enabled:
            object.__setattr__(self.mla, 'rope_scaling_type', 'yarn')

    def __post_init__(self):
        """Run validation after initialization."""
        # Set default MTP hidden size if not specified
        if self.mtp.mtp_hidden_size is None:
            object.__setattr__(self.mtp, 'mtp_hidden_size', self.hidden_size)
        
        # Wire YaRN configuration
        self.wire_yarn_config()

        self.validate()


# ============================================================================
# Pre-defined Configurations
# ============================================================================
#
# SCALING LAW COMPLIANCE
# ======================
# All configurations follow empirically validated scaling laws:
#
# 1. Chinchilla Optimal: D = 20 × N (tokens = 20× active parameters)
# 2. Depth-Width Ratio: d/L follows GPT-2 series interpolation
#    - 124M: d/L = 64 (GPT-2-Small)
#    - 355M: d/L = 42.7 (GPT-2-Medium)
#    - 561M: d/L = 36.6 (interpolated)
#    - 774M: d/L = 35.6 (GPT-2-Large)
#    - 1.5B: d/L = 33.3 (GPT-2-XL)
#
# 3. DeepSeek V3 Ratios (preserved across all scales):
#    - q_lora_rank / hidden = 0.215
#    - kv_lora_rank / hidden = 0.070
#    - intermediate_size / hidden = 2.56 (SwiGLU FFN)
#    - qk_nope : qk_rope : v = 3 : 1 : 3 (head dimensions)
#
# 4. MoE Activation Ratio: ~20-25% of total experts active per token
#    (Apple research suggests 20-40% optimal for training stability)
#
# ============================================================================


def get_nanoseek_config() -> NanoSeekConfig:
    """
    NanoSeek-1B: Research MoE configuration derived from first principles.

    DESIGN PRINCIPLES (what governs each choice):
    ==============================================
    1. N_active ≈ 1B from budget constraint + Chinchilla (Hoffmann 2022): 20:1 token ratio
    2. Depth=16, Width=2048 from Allen-Zhu depth analysis + OLMoE-1B precedent (d/L=128)
    3. Granularity G≈29 from Krajewski et al. (ICML 2024): G=16-32 optimal at ~1e20 FLOPs
    4. MLA ratios from DeepSeek V3 paper (q_lora/h=0.215, kv_lora/h=0.070)
    5. Dense FFN ratio 2.56× from DeepSeek V3 SwiGLU sizing
    6. Aux-loss-free load balancing from DeepSeek V3 (avoids loss distortion)

    WHAT'S FROM DEEPSEEK V3 vs WHAT'S NOT:
    =======================================
    From DeepSeek V3: MLA ratios, SwiGLU 2.56× dense FFN, sigmoid routing,
      grouped routing (n_group=8, topk_group=4), aux-loss-free γ=0.001,
      top-8 activation, first-2-dense pattern, MTP, DSA concept.
    NOT from DeepSeek V3:
      - Expert count: V3 uses 256 experts, we use 64 (OLMoE precedent at 1B scale)
      - κ (sparsity): V3 has κ=8/256=3.1%, we have κ=8/64=12.5% (4× denser)
      - moe_inter ratio: V3 has 2048/7168=0.286, we have 768/2048=0.375
      - Shared experts: V3 uses 1, we use 2 (DeepSeekMoE small-scale precedent)
      - Expansion: V3 has 18.1×, we have 4.4× (consequence of 64E at 1B scale)

    MOE GRANULARITY DERIVATION (forward, from Krajewski et al.):
    ============================================================
    G = N_active / (top_k × expert_params)
      = 1.08B / (8 × 4.72M) = 28.6
    Krajewski optimal range at our compute: G = 16-32 ✓

    Given G≈29 and top_k=8:
      expert_params = 1.08B / (8 × 29) ≈ 4.66M
      moe_inter = expert_params / (3 × hidden) = 4.66M / (3 × 2048) = 758
      → rounded to 768 for hardware alignment

    NOTE: moe_inter=768 gives ratio 0.375. This is NOT "DeepSeek's ratio" —
    DeepSeek V3 uses 0.286. Our ratio is a consequence of targeting G≈29 at
    (E=64, top_k=8). Both ratios are valid but serve different granularity targets.

    WHY 64 EXPERTS (not 256 like DeepSeek V3):
    ===========================================
    At 1B active with 256 experts, each expert would be ~1.2M params.
    At anchor scale (480 hidden), that drops to ~65K params — too small for
    meaningful specialization in HP search. 64 experts is the practical choice
    at 1B scale, validated by OLMoE-1B (same E=64, top-8 config).
    DeepSeek's "more, finer-grained" philosophy is correct in principle but
    requires larger base hidden size to keep per-expert capacity viable.

    WHY 2 SHARED EXPERTS (not 1 like DeepSeek V3):
    ===============================================
    DeepSeek V3 uses 1 shared expert at 671B total. DeepSeekMoE (2401.06066)
    used 2 at smaller scale (16B). At our 4.75B total, 2 shared experts provide
    more common-knowledge capacity relative to model size. This is a small-scale
    hypothesis, not a validated principle — could be ablated.

    PARAMETER COUNT (Verified):
    ===========================
    - Embeddings:      32768 × 2048 × 2 = 134M (always active)
    - MLA (16 layers): 16 × ~4.3M = 69M (always active)
    - Dense FFN (2):   2 × 32M = 64M (always active)
    - Shared experts:  14 × 2 × 4.7M = 132M (always active)
    - Routed experts:  14 × 64 × 4.7M = 4.2B total, 8 active = 526M
    - Router + norms:  ~18M

    TOTALS:
    - Active: 268 + 69 + 64 + 132 + 526 + 18 = ~1.08B ✓
    - Total:  268 + 69 + 64 + 4200 + 132 + 18 = ~4.75B ✓
    - Expansion: 4.75B / 1.08B = 4.4×

    MEMORY ANALYSIS (8×H100):
    =========================
    - Model (BF16): 4.75B × 2 = 9.5 GB
    - Optimizer (FP32): 4.75B × 12 = 57 GB (Adam states)
    - Gradients: 9.5 GB
    - Activations: ~15 GB
    - Total/GPU with DDP: ~12 GB (sharded optimizer)
    ✓ DDP works comfortably - NO FSDP required!

    TRAINING (Chinchilla Optimal):
    ==============================
    - Tokens: 22B (20× active params — Hoffmann 2022)
    - Compute: ~112 H100-hrs
    - Cost: ~$275-350
    - Time: ~14 hours on 8×H100

    REFERENCE COMPARISONS:
    ======================
    | Model          | Active | Total  | d/L | E   | top_k | κ     | Expansion |
    |----------------|--------|--------|-----|-----|-------|-------|-----------|
    | DeepSeek-V3    | 37B    | 671B   | —   | 256 | 8     | 3.1%  | 18.1×     |
    | OLMoE-1B-7B    | 1.3B   | 6.9B   | 128 | 64  | 8     | 12.5% | 5.3×      |
    | NanoSeek-1B    | 1.08B  | 4.75B  | 128 | 64  | 8     | 12.5% | 4.4×      |
    | DeepSeek-V2-L  | 2.4B   | 16B    | 117 | —   | —     | —     | 6.7×      |

    KEY INSIGHT: NanoSeek follows OLMoE's scale-class design (E=64, κ=12.5%),
    NOT DeepSeek V3's fine-grained design (E=256, κ=3.1%). The MLA, routing
    mechanism, load balancing, and training techniques come from DeepSeek V3.
    The MoE sizing comes from Krajewski's granularity theory + OLMoE precedent.
    """
    return NanoSeekConfig(
        # ====================================================================
        # Core Architecture (OLMoE-aligned d/L = 128)
        # ====================================================================
        vocab_size=32768,             # 32K — embedding tax 14.2% (was 24.8% at 65K)
        hidden_size=2048,             # d/L = 2048/16 = 128 (OLMoE, LLaMA scale)
        num_layers=16,                # Allen-Zhu: depth=reasoning hops; OLMoE-1B precedent
        num_heads=16,                 # head_dim = 2048/16 = 128 (standard)

        # Dense FFN (for first_k_dense_replace layers)
        intermediate_size=5243,       # 2.56 × 2048 (DeepSeek ratio)

        # ====================================================================
        # MLA Configuration (DeepSeek V3 ratios at 2048 scale)
        # ====================================================================
        mla=MLAConfig(
            q_lora_rank=440,           # 0.215 × 2048 = 440 (muP ratio must match anchor/500M)
            kv_lora_rank=143,          # 0.07 × 2048 = 143 ✓
            qk_nope_head_dim=128,      # DeepSeek family constant (V2-Lite/V2/V3 all use 128)
            qk_rope_head_dim=64,       # DeepSeek family constant (all use 64)
            v_head_dim=128,            # DeepSeek family constant (all use 128)
            rope_theta=10000.0,
            original_max_position_embeddings=4096,
        ),

        # ====================================================================
        # MoE Configuration (Krajewski G≈29 + OLMoE precedent)
        # ====================================================================
        # 64 experts, 8 active → κ=12.5% (OLMoE-1B design, NOT DeepSeek V3's κ=3.1%)
        # G = N_active/(top_k × expert_params) = 1.08B/(8×4.72M) ≈ 29 (Krajewski optimal: 16-32)
        moe=MoEConfig(
            n_routed_experts=64,         # OLMoE-1B precedent; Krajewski G≈29 at this E
            num_experts_per_tok=8,       # Same k as DeepSeek V3, but κ=12.5% not 3.1%
            n_shared_experts=2,          # DeepSeekMoE small-scale (V3 uses 1 at 671B)
            moe_intermediate_size=768,   # From G≈29: 1.08B/(8×29)=4.66M → 4.66M/(3×2048)=758→768

            # Routing configuration
            n_group=8,                   # 8 experts per group (64/8)
            topk_group=4,                # Route to half the groups
            scoring_func="sigmoid",      # DeepSeek V3 innovation
            routed_scaling_factor=2.5,   # DeepSeek V3 empirical value (NOT sqrt(K)=2.83)
            norm_topk_prob=True,

            # Load balancing (DeepSeek V3 aux-loss-free)
            gamma=0.001,
            gamma_freeze_ratio=0.95,     # V3: 14.3T/14.8T ≈ 0.966 → 0.95 conservative
            seq_aux_loss_alpha=0.0001,   # Very small sequence-level aux loss

            # First layer uses dense FFN (V2-Lite precedent at comparable scale)
            first_k_dense_replace=1,
        ),

        # ====================================================================
        # MTP Configuration
        # ====================================================================
        mtp=MTPConfig(
            num_mtp_modules=1,
            mtp_num_heads=16,            # Same as main model (V3 paper)
            mtp_loss_weight_initial=0.3,
            mtp_loss_weight_final=0.1,
            mtp_loss_transition_ratio=0.60,
        ),

        # ====================================================================
        # Sparse Attention Configuration
        # ====================================================================
        sparse=SparseAttentionConfig(
            enabled=False,               # Train dense, infer sparse
            topk_tokens=2048,
            activation_threshold=4096,
            indexer_num_heads=4,
            indexer_head_dim=64,
        ),

        # ====================================================================
        # Training Configuration (Chinchilla optimal: 22B = 20× 1.08B)
        # ====================================================================
        total_tokens=22_000_000_000,     # 22B tokens (Chinchilla optimal)
        global_batch_size=128,           # Standard batch size
        sequence_length=4096,            # 4K context

        # Optimizer (DeepSeek style)
        learning_rate=3e-4,
        lr_min=3e-5,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,

        # LR Schedule (warmup → constant → cosine decay)
        warmup_steps=1000,
        constant_phase_ratio=0.70,
        cosine_decay_end_ratio=0.95,

        # Precision
        dtype="bfloat16",
        use_fp8=False,

        # Checkpointing
        gradient_checkpointing=True,
        checkpoint_every_steps=1000,
        log_every_steps=10,
        eval_every_steps=500,

        # Distributed (DDP sufficient - no FSDP needed!)
        data_parallel_size=8,
    )


# ============================================================================
# NanoSeek-500M Configuration (muP intermediate, G≈30)
# ============================================================================

def get_nanoseek_500m_config() -> NanoSeekConfig:
    """
    NanoSeek-500M: muP-aligned intermediate config for HP transfer validation.

    DESIGN PHILOSOPHY:
    ==================
    This configuration is the INTERMEDIATE VALIDATION point in Option B's
    3-point muP transfer path: anchor (55M) → 500M → 1B.
    Architecture ratios MUST match NanoSeek-1B exactly for muP transfer:
      - num_layers = 16 (same depth as 1B — muP transfers across width, not depth)
      - κ = top_k/n_experts = 8/64 = 12.5% (constant sparsity ratio)
      - moe_inter/hidden = 0.375 (constant expert shape ratio)
      - n_shared_experts = 2 (constant shared expert count)

    PARAMETER COUNT (muP-aligned):
    ==============================
    Component breakdown:
    - Embeddings:      32768 × 1280 × 2 = 84M (always active)
    - MLA (16 layers): 16 × ~2.0M = 32M (always active)
    - Dense FFN (2):   2 × 12.6M = 25M (always active)
    - Shared experts:  14 × 2 × 1.47M = 41M (always active)
    - Routed active:   14 × 8 × 1.47M = 165M (8 of 64 experts)
    - Router + norms:  ~10M

    TOTALS:
    - Active: 168 + 32 + 25 + 41 + 165 + 10 = ~441M
    - Total:  168 + 32 + 25 + 41 + (64×1.47M×14) + 10 = ~1.59B + 276M = ~2.1B
    - Expansion: ~2.1B / 0.44B = ~4.8× (matches 1B's 4.4× within tolerance)

    EXPERT SHAPE (muP-consistent, Krajewski G-preserving):
    =====================================================
    - moe_intermediate_size = floor(0.375 × 1280) = 480
      (ratio 0.375 derived from 1B's G≈29 target, held constant for muP)
    - Expert params = 3 × 1280 × 480 = 1.84M per expert
    - G = 441M / (8 × 1.84M) ≈ 30 (within Krajewski optimal 16-32 ✓)
    - κ = 8/64 = 12.5% (constant across all configs for muP transfer)

    TRAINING:
    =========
    - Tokens: 8.8B (Chinchilla optimal: 20 × 441M)
    - Purpose: Validate that muP-transferred HPs from anchor produce good training
    """
    return NanoSeekConfig(
        # ====================================================================
        # Core Architecture (scaled from 1B)
        # ====================================================================
        vocab_size=32768,             # Same vocabulary (32K across all scales)
        hidden_size=1280,             # Scaled down from 2048
        num_layers=16,                # MUST match 1B for muP transfer (depth constant)
        num_heads=10,                 # head_dim = 1280/10 = 128 (standard)

        # Dense FFN (for first_k_dense_replace layers)
        intermediate_size=3277,       # 2.56 × 1280 (DeepSeek ratio preserved)

        # ====================================================================
        # MLA Configuration (DeepSeek V3 ratios at 1280 scale)
        # ====================================================================
        mla=MLAConfig(
            q_lora_rank=275,           # 0.215 × 1280 = 275 ✓
            kv_lora_rank=90,           # 0.070 × 1280 = 90 ✓
            qk_nope_head_dim=128,      # DeepSeek family constant (V2-Lite/V2/V3 all use 128)
            qk_rope_head_dim=64,       # DeepSeek family constant (all use 64)
            v_head_dim=128,            # DeepSeek family constant (all use 128)
            rope_theta=10000.0,
            original_max_position_embeddings=4096,
        ),

        # ====================================================================
        # MoE Configuration (muP-aligned, κ=12.5%)
        # ====================================================================
        # 64 experts, 8 active = 12.5% activation ratio (κ constant with 1B)
        # Expert shape: moe_inter = 0.375 × hidden = 480 (consistent ratio)
        moe=MoEConfig(
            n_routed_experts=64,         # 64 experts (κ=8/64=12.5%)
            num_experts_per_tok=8,       # 8 active (12.5% ratio preserved)
            n_shared_experts=2,          # Must match 1B (2 shared experts)
            moe_intermediate_size=480,   # 0.375 × 1280 = 480 (muP-consistent ratio)

            # Routing configuration (8 groups of 8 experts)
            n_group=8,                   # 8 experts per group (64/8)
            topk_group=4,                # Route to half the groups
            scoring_func="sigmoid",      # DeepSeek V3 innovation
            routed_scaling_factor=2.5,   # DeepSeek V3 empirical value (NOT sqrt(K)=2.83)
            norm_topk_prob=True,

            # Load balancing (DeepSeek V3 aux-loss-free)
            gamma=0.001,
            gamma_freeze_ratio=0.95,     # V3: 14.3T/14.8T ≈ 0.966 → 0.95 conservative
            seq_aux_loss_alpha=0.0001,

            # First layer uses dense FFN (V2-Lite precedent at comparable scale)
            first_k_dense_replace=1,
        ),

        # ====================================================================
        # MTP Configuration (same as 1B, works at any scale)
        # ====================================================================
        mtp=MTPConfig(
            num_mtp_modules=1,
            mtp_num_heads=10,            # Same as main model (V3 paper)
            mtp_loss_weight_initial=0.3,
            mtp_loss_weight_final=0.1,
            mtp_loss_transition_ratio=0.60,
        ),

        # ====================================================================
        # Sparse Attention Configuration (same strategy)
        # ====================================================================
        sparse=SparseAttentionConfig(
            enabled=False,               # Train dense, infer sparse
            topk_tokens=1024,            # Scaled for smaller context
            activation_threshold=4096,
            indexer_num_heads=4,
            indexer_head_dim=48,         # Scaled from 64
        ),

        # ====================================================================
        # Training Configuration (Chinchilla optimal: 8.8B ≈ 20 × 441M active)
        # ====================================================================
        total_tokens=8_800_000_000,      # 8.8B tokens (Chinchilla for ~441M active)
        global_batch_size=128,           # Same batch size
        sequence_length=4096,            # Same 4K context

        # Optimizer (same hyperparameters - scale-independent)
        learning_rate=3e-4,
        lr_min=3e-5,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,

        # LR Schedule (same ratios)
        warmup_steps=500,                # Scaled from 1000 (fewer total steps)
        constant_phase_ratio=0.70,
        cosine_decay_end_ratio=0.95,

        # Precision
        dtype="bfloat16",
        use_fp8=False,

        # Checkpointing
        gradient_checkpointing=True,
        checkpoint_every_steps=500,      # More frequent for shorter training
        log_every_steps=10,
        eval_every_steps=250,

        # Distributed
        data_parallel_size=8,
    )


def get_nanoseek_anchor_config() -> NanoSeekConfig:
    """
    NanoSeek muP Anchor: Small-scale config for hyperparameter search.

    DESIGN PHILOSOPHY:
    ==================
    This is the ANCHOR model in Option B's muP transfer path.
    All architecture ratios MUST match NanoSeek-1B exactly:
      - num_layers = 16 (depth constant — muP transfers across width only)
      - κ = top_k/n_experts = 8/64 = 12.5% (constant sparsity ratio)
      - moe_inter/hidden = 0.375 (constant expert shape ratio)
      - n_shared_experts = 2 (constant shared expert count)

    muP TRANSFER THEORY (Tensor Programs V + μP-MoE + Complete(d)P):
    ================================================================
    - Width is the ONLY free variable (480 → 1280 → 2048)
    - Hidden weight LRs scale as 1/fan_in across widths
    - Expert weights are "hidden weights" (LR ∝ 1/width)
    - Router weights are "output weights" (LR constant)
    - σ_attn, σ_expert (constant-scale multipliers) tuned here, transfer directly

    PARAMETER COUNT:
    ================
    - Embeddings:      32768 × 480 × 2 = 31M (always active)
    - MLA (16 layers): 16 × ~0.28M = 4.5M (always active)
    - Dense FFN (2):   2 × 1.77M = 3.5M (always active)
    - Shared experts:  14 × 2 × 0.26M = 7.3M (always active)
    - Routed active:   14 × 8 × 0.26M = 29M (8 of 64 experts)
    - Router + norms:  ~2M

    TOTALS:
    - Active: 31 + 4.5 + 3.5 + 7.3 + 29 + 2 ≈ 77M (32K vocab halves embedding tax)
    - Total:  31 + 4.5 + 3.5 + 7.3 + (64×0.26M×14) + 2 ≈ 282M
    - Expansion: ~282M / 77M ≈ 3.7× (improved from 2.9× with 65K vocab)

    NOTE: Anchor quality doesn't need to be high — it only needs to find
    good HP ratios that transfer to larger scales via muP rules.
    """
    return NanoSeekConfig(
        # ====================================================================
        # Core Architecture (muP anchor — width=480, depth=16 matching 1B)
        # ====================================================================
        vocab_size=32768,             # Same vocabulary (32K across all scales)
        hidden_size=480,              # muP anchor width (transfers to 1280→2048)
        num_layers=16,                # MUST match 1B (depth constant in muP)
        num_heads=6,                  # head_dim = 480/6 = 80

        # Dense FFN (for first_k_dense_replace layers)
        intermediate_size=1229,       # 2.56 × 480 (DeepSeek ratio preserved)

        # ====================================================================
        # MLA Configuration (DeepSeek V3 ratios at 480 scale)
        # ====================================================================
        mla=MLAConfig(
            q_lora_rank=103,           # 0.215 × 480 = 103
            kv_lora_rank=34,           # 0.070 × 480 = 34
            qk_nope_head_dim=128,      # DeepSeek family constant (V2-Lite/V2/V3 all use 128)
            qk_rope_head_dim=64,       # DeepSeek family constant (all use 64)
            v_head_dim=128,            # DeepSeek family constant (all use 128)
            rope_theta=10000.0,
            original_max_position_embeddings=4096,
        ),

        # ====================================================================
        # MoE Configuration (muP-aligned, κ=12.5%)
        # ====================================================================
        moe=MoEConfig(
            n_routed_experts=64,         # 64 experts (κ=8/64=12.5%)
            num_experts_per_tok=8,       # 8 active (12.5% ratio preserved)
            n_shared_experts=2,          # Must match 1B (2 shared experts)
            moe_intermediate_size=180,   # floor(0.375 × 480) = 180

            # Routing configuration (8 groups of 8 experts)
            n_group=8,
            topk_group=4,
            scoring_func="sigmoid",
            routed_scaling_factor=2.5,   # DeepSeek V3 empirical value (NOT sqrt(K)=2.83)
            norm_topk_prob=True,

            # Load balancing
            gamma=0.001,
            gamma_freeze_ratio=0.95,     # RULE 2: always 0.95
            seq_aux_loss_alpha=0.0001,

            # First layer uses dense FFN (V2-Lite precedent)
            first_k_dense_replace=1,
        ),

        # ====================================================================
        # MTP Configuration (same as 1B)
        # ====================================================================
        mtp=MTPConfig(
            num_mtp_modules=1,
            mtp_num_heads=6,             # Match num_heads for anchor
            mtp_loss_weight_initial=0.3,
            mtp_loss_weight_final=0.1,
            mtp_loss_transition_ratio=0.60,
        ),

        # ====================================================================
        # Sparse Attention Configuration
        # ====================================================================
        sparse=SparseAttentionConfig(
            enabled=False,
            topk_tokens=512,
            activation_threshold=4096,
            indexer_num_heads=2,
            indexer_head_dim=32,
        ),

        # ====================================================================
        # Training Configuration (short runs for HP grid search)
        # ====================================================================
        total_tokens=1_100_000_000,     # ~1.1B tokens (20 × 55M non-embed active)
        global_batch_size=64,           # Smaller for anchor scale
        sequence_length=4096,

        # Optimizer (BASE values — muP grid search tunes these)
        learning_rate=3e-4,
        lr_min=3e-5,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,

        # LR Schedule
        warmup_steps=200,
        constant_phase_ratio=0.70,
        cosine_decay_end_ratio=0.95,

        # Precision
        dtype="bfloat16",
        use_fp8=False,

        # Checkpointing
        gradient_checkpointing=True,
        checkpoint_every_steps=500,
        log_every_steps=10,
        eval_every_steps=250,

        # Distributed (anchor can run on fewer GPUs)
        data_parallel_size=4,
    )


# ============================================================================
# Training Phase Helpers
# ============================================================================

def get_training_phases(config: NanoSeekConfig) -> List[TrainingPhaseConfig]:
    """
    Get the training phases for NanoSeek, derived from the config.

    Returns a list of TrainingPhaseConfig objects that define the
    multi-phase training pipeline.

    Phase 1: Dense MLA (80% tokens, base context from config)
    Phase 2: Sparse DSA (20% tokens, 2x context with YaRN)

    The phases are generated based on the config's sequence_length,
    global_batch_size, and learning rate settings.
    """
    phase1 = TrainingPhaseConfig(
        name="phase1_dense_mla",
        sequence_length=config.sequence_length,
        global_batch_size=config.global_batch_size,
        token_fraction=0.8,
        learning_rate=config.learning_rate,
        lr_min=config.lr_min,
        dsa_enabled=False,
        dsa_activation_threshold=config.sequence_length,
        yarn_enabled=False,
        warmup_steps=config.warmup_steps,
    )

    phase2 = TrainingPhaseConfig(
        name="phase2_sparse_dsa",
        sequence_length=config.sequence_length * 2,  # 2x context extension
        global_batch_size=config.global_batch_size // 2,  # Reduce for memory
        token_fraction=0.2,
        learning_rate=config.learning_rate / 3,  # Lower LR for fine-tuning
        lr_min=config.lr_min / 3,
        dsa_enabled=True,
        dsa_activation_threshold=config.sequence_length,
        yarn_enabled=True,
        warmup_steps=100,  # Brief warmup for phase transition
    )

    return [phase1, phase2]


def apply_phase_config(
    config: NanoSeekConfig,
    phase: TrainingPhaseConfig,
) -> NanoSeekConfig:
    """
    Apply a training phase configuration to a NanoSeekConfig.

    This updates the config in-place to match the phase settings.
    Used when transitioning between training phases.

    Args:
        config: Base NanoSeekConfig
        phase: TrainingPhaseConfig to apply

    Returns:
        Updated NanoSeekConfig (same object, modified)
    """
    # Update context and batch
    config.sequence_length = phase.sequence_length
    config.global_batch_size = phase.global_batch_size

    # Update learning rate
    config.learning_rate = phase.learning_rate
    config.lr_min = phase.lr_min
    config.warmup_steps = phase.warmup_steps

    # Update DSA
    config.sparse.enabled = phase.dsa_enabled
    config.sparse.activation_threshold = phase.dsa_activation_threshold

    # Update YaRN
    config.yarn.enabled = phase.yarn_enabled

    # Update max positions for the new context length
    config.max_position_embeddings = phase.sequence_length

    return config


def get_phase_tokens(config: NanoSeekConfig, phase: TrainingPhaseConfig) -> int:
    """Get the number of tokens to train for this phase."""
    return int(config.total_tokens * phase.token_fraction)


def get_phase_steps(config: NanoSeekConfig, phase: TrainingPhaseConfig) -> int:
    """Get the number of steps to train for this phase."""
    tokens = get_phase_tokens(config, phase)
    tokens_per_step = phase.global_batch_size * phase.sequence_length
    return tokens // tokens_per_step


def print_training_pipeline(config: NanoSeekConfig) -> None:
    """Print a summary of the training pipeline."""
    print("\n" + "=" * 70)
    print("NanoSeek Multi-Phase Training Pipeline")
    print("=" * 70)
    print("""
    Following DeepSeek methodology: Train short, extend long.

    ┌─────────────────────────────────────────────────────────────────┐
    │  Phase 1: Dense MLA                                             │
    │  ───────────────────                                            │
    │  • Context: 4096 tokens                                         │
    │  • Attention: Dense (full attention)                            │
    │  • DSA: OFF (but indexer trains via KL-divergence aux loss)     │
    │  • YaRN: OFF (native positions)                                 │
    │  • Goal: Learn strong representations with max gradient updates │
    │                                                                 │
    │                            ↓                                    │
    │                     (checkpoint)                                │
    │                            ↓                                    │
    │                                                                 │
    │  Phase 2: Sparse DSA                                            │
    │  ────────────────────                                           │
    │  • Context: 8192 tokens (2x Phase 1)                            │
    │  • Attention: Sparse (top-k selection via indexer)              │
    │  • DSA: ON (indexer active, selecting important tokens)         │
    │  • YaRN: ON (interpolate RoPE to 8K positions)                  │
    │  • Goal: Adapt to long context efficiently                      │
    │                                                                 │
    │                            ↓                                    │
    │                     (final model)                               │
    │                            ↓                                    │
    │                                                                 │
    │  Inference: YaRN + DSA enables 32K context                      │
    │  ──────────────────────────────────────                         │
    │  • YaRN extends to 32K (8x training length)                     │
    │  • DSA keeps attention efficient (top-2K of 32K)                │
    │  • MLA keeps KV cache small (~23x compression)                  │
    └─────────────────────────────────────────────────────────────────┘
    """)

    phases = get_training_phases(config)
    total_steps = 0

    for i, phase in enumerate(phases, 1):
        tokens = get_phase_tokens(config, phase)
        steps = get_phase_steps(config, phase)
        total_steps += steps

        print(f"Phase {i}: {phase.name}")
        print(f"  ├── Context:    {phase.sequence_length:,} tokens")
        print(f"  ├── Batch:      {phase.global_batch_size}")
        print(f"  ├── Tokens:     {tokens:,} ({tokens/1e9:.1f}B, {phase.token_fraction*100:.0f}%)")
        print(f"  ├── Steps:      {steps:,}")
        print(f"  ├── LR:         {phase.learning_rate} → {phase.lr_min}")
        print(f"  ├── DSA:        {'✓ Enabled' if phase.dsa_enabled else '✗ Disabled'}")
        print(f"  └── YaRN:       {'✓ Enabled' if phase.yarn_enabled else '✗ Disabled'}")
        print()

    print(f"Total training: {config.total_tokens/1e9:.1f}B tokens, ~{total_steps:,} steps")
    print("=" * 70)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Demonstrate configuration
    config = get_nanoseek_config()

    # Calculate KV compression ratio
    standard_kv = 2 * config.num_heads * config.head_dim
    mla_kv = config.mla.kv_cache_dim_per_layer
    compression_ratio = standard_kv / mla_kv

    print("NanoSeek-1B (DeepSeek-Aligned) Configuration Summary")
    print("=" * 60)
    print(f"\nArchitecture:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Heads: {config.num_heads}")
    print(f"  Head dim: {config.head_dim}")
    print(f"  Vocab: {config.vocab_size:,}")

    print(f"\nMLA ({compression_ratio:.0f}x KV cache compression):")
    print(f"  Q LoRA rank: {config.mla.q_lora_rank} ({config.mla.q_lora_rank/config.hidden_size:.2%} of hidden)")
    print(f"  KV LoRA rank: {config.mla.kv_lora_rank} ({config.mla.kv_lora_rank/config.hidden_size:.2%} of hidden)")
    print(f"  KV cache per layer: {mla_kv} dims")
    print(f"  vs Standard MHA: {standard_kv} dims")
    print(f"  Compression: {compression_ratio:.1f}x")

    print(f"\nMoE (5x parameter capacity):")
    print(f"  Routed experts: {config.moe.n_routed_experts}")
    print(f"  Active per token: {config.moe.num_experts_per_tok}")
    print(f"  Shared experts: {config.moe.n_shared_experts}")
    print(f"  Groups: {config.moe.n_group}")
    print(f"  Dense layers: {config.dense_layer_indices}")
    print(f"  MoE layers: {config.moe_layer_indices}")
    print(f"  Scoring: {config.moe.scoring_func} (aux-loss-free)")

    print(f"\nMTP (1.4x inference speedup):")
    print(f"  MTP modules: {config.mtp.num_mtp_modules}")
    print(f"  Total predictions: {config.mtp.total_predictions}")
    print(f"  Loss weight: {config.mtp.mtp_loss_weight_initial} → {config.mtp.mtp_loss_weight_final}")

    print(f"\nParameters:")
    print(f"  Total: {config.estimated_total_params:,} (~{config.estimated_total_params/1e9:.2f}B)")
    print(f"  Active: {config.estimated_active_params:,} (~{config.estimated_active_params/1e6:.0f}M)")

    print(f"\nTraining:")
    print(f"  Tokens: {config.total_tokens:,} ({config.total_tokens/1e9:.1f}B)")
    print(f"  Steps: {config.total_steps:,}")
    print(f"  Tokens/step: {config.tokens_per_step:,}")
    print(f"  LR: {config.learning_rate} -> {config.lr_min}")

    # Show training phases
    print(f"\n" + "=" * 60)
    print("Multi-Phase Training Pipeline")
    print("=" * 60)

    phases = get_training_phases(config)
    for i, phase in enumerate(phases, 1):
        tokens = int(config.total_tokens * phase.token_fraction)
        steps = tokens // (phase.global_batch_size * phase.sequence_length)
        print(f"\nPhase {i}: {phase.name}")
        print(f"  Context: {phase.sequence_length}")
        print(f"  Batch size: {phase.global_batch_size}")
        print(f"  Tokens: {tokens/1e9:.1f}B ({phase.token_fraction*100:.0f}%)")
        print(f"  Steps: ~{steps:,}")
        print(f"  LR: {phase.learning_rate} -> {phase.lr_min}")
        print(f"  DSA: {'Enabled' if phase.dsa_enabled else 'Disabled (indexer trains via aux loss)'}")
        print(f"  YaRN: {'Enabled' if phase.yarn_enabled else 'Disabled (native positions)'}")
