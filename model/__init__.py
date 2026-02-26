# Copyright (c) 2024 DeepSeek AI. All rights reserved.
# NanoSeek Model Package
#
# This package implements the NanoSeek architecture, a nano-scale version
# of DeepSeek V3.2 with all production-grade components:
# - MLA (Multi-head Latent Attention)
# - MoE (Mixture of Experts with aux-loss-free balancing)
# - MTP (Multi-Token Prediction)
# - DSA (DeepSeek Sparse Attention with Lightning Indexer)
#
# Single-file reference implementation following top-tier open-source models
# (nanoGPT, LLaMA, GPT-2). Core architecture in model.py.

# =============================================================================
# Configuration (from config.py)
# =============================================================================
from .config import (
    NanoSeekConfig,
    MLAConfig,
    MoEConfig,
    MTPConfig,
    SparseAttentionConfig,
    get_nanoseek_config,
)

# =============================================================================
# Core Model (from model.py)
# =============================================================================
from .model import (
    # RoPE
    RotaryEmbedding,
    precompute_freqs_cis,
    apply_rotary_emb,
    # Normalization
    RMSNorm,
    # MLA
    MultiHeadLatentAttention,
    create_causal_mask,
    # MoE
    MoE,
    MLP,
    Gate,
    Expert,
    # MTP
    MTPBlock,
    MTPModule,
    MultiTokenPrediction,
    # Sparse Attention
    LightningIndexer,
    DSASparseAttention,
    # Model
    NanoSeekDecoderLayer,
    NanoSeekModel,
    # Factory functions
    create_nanoseek,
    create_mla_from_config as _create_mla_from_config_core,
    create_moe_from_config as _create_moe_from_config_core,
    # Test
    test_nanoseek,
)

# =============================================================================
# Training Configuration (from config.py)
# =============================================================================
try:
    from .config import (
        # V3.2 Production configs
        YaRNConfig,
        FP8Config,
        ParallelConfig,
        # Multi-phase training (DeepSeek methodology)
        TrainingPhaseConfig,
        PHASE1_DENSE,
        PHASE2_SPARSE,
        get_training_phases,
        apply_phase_config,
        get_phase_tokens,
        get_phase_steps,
        print_training_pipeline,
        # Scaled configurations for experiments
        get_nanoseek_500m_config,
    )
    TRAINING_CONFIG_AVAILABLE = True
except ImportError:
    TRAINING_CONFIG_AVAILABLE = False

# =============================================================================
# Helper Functions
# =============================================================================

def create_mla_from_config(
    config: NanoSeekConfig,
    layer_idx: int = 0,
) -> MultiHeadLatentAttention:
    """Create MLA from NanoSeek config."""
    return _create_mla_from_config_core(config, layer_idx=layer_idx)


def create_moe_from_config(config: NanoSeekConfig) -> MoE:
    """Create MoE from NanoSeek config."""
    return _create_moe_from_config_core(config)

# =============================================================================
# Optional: Advanced Indexer (for training)
# =============================================================================
try:
    from .indexer import (
        IndexerTrainingWrapper,
        compute_indexer_loss,
        rotate_activation,
    )
    INDEXER_TRAINING_AVAILABLE = True
except ImportError:
    INDEXER_TRAINING_AVAILABLE = False

# =============================================================================
# Optional: Parallel (for distributed training)
# =============================================================================
try:
    from .parallel import (
        ParallelEmbedding,
        ColumnParallelLinear,
        RowParallelLinear,
        ExpertParallelLinear,
        initialize_model_parallel,
        get_tensor_parallel_world_size,
        get_tensor_parallel_rank,
        get_expert_parallel_world_size,
        get_data_parallel_world_size,
    )
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False

# =============================================================================
# Optional: FP8 (for H100/H200)
# =============================================================================
try:
    from .fp8 import (
        FP8Linear,
        FP8KVCache,
        FP8TrainingContext,
        is_fp8_available,
        get_fp8_recipe,
        quantize_to_fp8_block,
        dequantize_from_fp8_block,
    )
    FP8_AVAILABLE = True
except ImportError:
    FP8_AVAILABLE = False

# =============================================================================
# Exports
# =============================================================================
__all__ = [
    # =========================================================================
    # Core Configuration
    # =========================================================================
    'NanoSeekConfig',
    'MLAConfig',
    'MoEConfig',
    'MTPConfig',
    'SparseAttentionConfig',
    'get_nanoseek_config',

    # =========================================================================
    # Model Architecture
    # =========================================================================
    'NanoSeekModel',
    'NanoSeekDecoderLayer',
    'create_nanoseek',
    'test_nanoseek',

    # =========================================================================
    # MLA (Multi-head Latent Attention)
    # =========================================================================
    'MultiHeadLatentAttention',
    'RMSNorm',
    'create_causal_mask',
    'create_mla_from_config',

    # =========================================================================
    # MoE (Mixture of Experts)
    # =========================================================================
    'MoE',
    'MLP',
    'Gate',
    'Expert',
    'create_moe_from_config',

    # =========================================================================
    # MTP (Multi-Token Prediction)
    # =========================================================================
    'MTPBlock',
    'MTPModule',
    'MultiTokenPrediction',

    # =========================================================================
    # DSA (DeepSeek Sparse Attention)
    # =========================================================================
    'DSASparseAttention',
    'LightningIndexer',

    # =========================================================================
    # RoPE (Rotary Position Embedding)
    # =========================================================================
    'RotaryEmbedding',
    'precompute_freqs_cis',
    'apply_rotary_emb',
]

# Add training config exports if available
if TRAINING_CONFIG_AVAILABLE:
    __all__.extend([
        'YaRNConfig',
        'FP8Config',
        'ParallelConfig',
        'TrainingPhaseConfig',
        'PHASE1_DENSE',
        'PHASE2_SPARSE',
        'get_training_phases',
        'apply_phase_config',
        'get_phase_tokens',
        'get_phase_steps',
        'print_training_pipeline',
        # Scaled configurations
        'get_nanoseek_500m_config',
    ])

# Add optional exports if available
if INDEXER_TRAINING_AVAILABLE:
    __all__.extend([
        'IndexerTrainingWrapper',
        'compute_indexer_loss',
        'rotate_activation',
    ])

if PARALLEL_AVAILABLE:
    __all__.extend([
        'ParallelEmbedding',
        'ColumnParallelLinear',
        'RowParallelLinear',
        'ExpertParallelLinear',
        'initialize_model_parallel',
        'get_tensor_parallel_world_size',
        'get_tensor_parallel_rank',
        'get_expert_parallel_world_size',
        'get_data_parallel_world_size',
    ])

if FP8_AVAILABLE:
    __all__.extend([
        'FP8Linear',
        'FP8KVCache',
        'FP8TrainingContext',
        'is_fp8_available',
        'get_fp8_recipe',
        'quantize_to_fp8_block',
        'dequantize_from_fp8_block',
    ])
