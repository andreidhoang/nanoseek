#!/usr/bin/env python3
"""
===============================================================================
NanoSeek Pre-training Script
===============================================================================

Train NanoSeek model with all DeepSeek V3 innovations:
- MLA (Multi-head Latent Attention): 24x KV cache compression
- MoE (Mixture of Experts): 5x parameter capacity with aux-loss-free balancing
- MTP (Multi-Token Prediction): 1.4x inference speedup via speculative decoding
- DSA (Differentiable Sparse Attention): O(L²) → O(L*k) for long context

Run as single GPU:
    python scripts/pre-train.py

Run distributed (8 GPUs):
    torchrun --nproc_per_node=8 scripts/pre-train.py

Quick test on CPU/MacBook:
    python scripts/pre-train.py --model_size=125m --max_seq_len=512 \\
        --device_batch_size=1 --total_batch_size=512 --num_iterations=100

Multi-Phase DSA Training (DeepSeek V3 methodology):
    python scripts/pre-train.py --enable_multiphase=true
    # Phase 1: Dense MLA (80% tokens, 4K context) - indexer trains via aux loss
    # Phase 2: Sparse DSA (20% tokens, 8K context) - DSA enabled, YaRN enabled

===============================================================================
Architecture Comparison (vs nanochat d20):
===============================================================================

                    nanochat d20          NanoSeek-1B
                    ────────────          ───────────
Active Params       561M                  1.08B
Total Params        561M                  4.75B (4.4x via MoE)
KV Cache/Layer      2560 dims             175 dims (23x smaller)
Attention           Standard MHA          MLA (low-rank)
FFN                 Dense                 MoE (4/32 active)
Predictions         1 token               2 tokens (MTP)
Load Balancing      N/A                   Aux-loss-free bias

===============================================================================
Training Features:
===============================================================================

1. FLOP-based Training Horizon
   - target_flops: Train to specific compute budget
   - target_param_data_ratio: Chinchilla-optimal (20x params)
   - num_iterations: Explicit step count

2. MFU (Model FLOPs Utilization) Tracking
   - Measures actual vs theoretical H100 throughput
   - Critical for hardware efficiency analysis

3. MoE Aux-Loss-Free Load Balancing (DeepSeek V3)
   - Dynamic bias adjustment instead of auxiliary loss
   - gamma schedule: 0.001 → 0 (freeze at 80% training)
   - Sequence-level aux loss: α = 0.0001 (very small)

4. MTP Dynamic Loss Schedule (DeepSeek V3)
   - λ = 0.3 for first 60% of training
   - λ = 0.1 for remaining 40%
   - Ratio-based for scale independence

5. Optimizer Setup
   - Muon: For attention/FFN matrix parameters (momentum-based)
   - AdamW: For embeddings/unembeddings
   - Decoupled learning rates for different parameter groups

6. Multi-Phase DSA Training (DeepSeek V3)
   - Phase 1: Dense MLA at 4K context (80% tokens)
     * DSA disabled, indexer trains via auxiliary loss
     * Lightning Indexer learns token importance patterns
   - Phase 2: Sparse DSA at 8K context (20% tokens)
     * DSA enabled, top-k token selection active
     * YaRN enabled for context extension
   - Enables 32K+ context at inference time

===============================================================================
"""

import os
# Prevent CUDA memory fragmentation with expandable segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import time
import math
import json
from pathlib import Path
from contextlib import nullcontext
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Add project root to path for imports
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# NanoSeek Model Imports
# ============================================================================
from model.config import (
    NanoSeekConfig,
    get_nanoseek_config,
    # Multi-phase DSA training configuration
    TrainingPhaseConfig,
    PHASE1_DENSE, PHASE2_SPARSE,
    get_training_phases,
    apply_phase_config,
    get_phase_tokens,
    get_phase_steps,
    print_training_pipeline,
)
from model.model import NanoSeekModel

# Custom Optimizers (from model.optimizer - matching nanochat implementation)
try:
    from model.optimizer.muon import Muon, DistMuon
    from model.optimizer.adamw import DistAdamW
    CUSTOM_OPTIMIZERS_AVAILABLE = True
except ImportError:
    CUSTOM_OPTIMIZERS_AVAILABLE = False
    print("Warning: Custom optimizers not available, using PyTorch defaults")

# Tokenizer (for sample text generation)
try:
    from scripts.tokenizer import get_tokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    try:
        from tokenizer import get_tokenizer
        TOKENIZER_AVAILABLE = True
    except ImportError:
        TOKENIZER_AVAILABLE = False
        print("Warning: Tokenizer not available, sample text generation disabled")

# Utils (for consistent checkpoint paths)
try:
    from scripts.utils import get_base_dir
except ImportError:
    try:
        from utils import get_base_dir
    except ImportError:
        # Fallback implementation
        def get_base_dir():
            return str(PROJECT_ROOT / "data")

# Streaming Dataloader (for real training data)
try:
    from scripts.dataloader import (
        tokenizing_distributed_data_loader_with_state,
        tokenizing_distributed_data_loader,
    )
    DATALOADER_AVAILABLE = True
except ImportError:
    try:
        from dataloader import (
            tokenizing_distributed_data_loader_with_state,
            tokenizing_distributed_data_loader,
        )
        DATALOADER_AVAILABLE = True
    except ImportError:
        DATALOADER_AVAILABLE = False
        print("Warning: Real dataloader not available, using dummy data")

# Checkpoint Manager (nanochat-style with per-rank optimizer state)
from model.eval.checkpoint_manager import (
    save_checkpoint as _save_checkpoint,
    load_checkpoint as _load_checkpoint,
    find_last_step,
)

# ============================================================================
# Configuration Dataclass
# ============================================================================

@dataclass
class TrainingConfig:
    """
    Training configuration for NanoSeek pre-training.

    This uses the nanochat-style single-file approach where all hyperparameters
    are explicit module-level variables for easy inspection and modification.

    Key Design Decisions:
    ---------------------
    1. FLOP-based horizons: Train to compute budget, not arbitrary steps
    2. Chinchilla ratios: Default 20x param:data for compute-optimal
    3. Separate LRs: Embeddings need different LR than matrix params
    4. Dynamic schedules: MTP weight and gamma follow V3 schedules
    """
    # ========================================================================
    # Run Identification
    # ========================================================================
    run_name: str = "nanoseek"           # W&B run name ("dummy" = no logging)
    model_tag: str = ""                   # Checkpoint directory name override

    # ========================================================================
    # Model Selection
    # ========================================================================
    model_size: str = "1b"                # Default to 1.08B active / 4.75B total (DeepSeek-aligned)              

    # ========================================================================
    # Hardware Configuration
    # ========================================================================
    device_type: str = ""                 # "cuda"|"cpu"|"mps" (empty=autodetect)
    compile_model: bool = True            # Use torch.compile for speedup

    # ========================================================================
    # Training Horizon (only ONE will be used, in order of precedence)
    # ========================================================================
    num_iterations: int = -1              # Explicit steps (-1 = disable)
    target_flops: float = -1.0            # Target FLOPs (-1 = disable)
    target_param_data_ratio: float = 20.0 # Chinchilla ratio (-1 = disable)

    # ========================================================================
    # Batch Configuration
    # ========================================================================
    device_batch_size: int = 8            # Per-device batch (tune to not OOM)
    total_batch_size: int = 524288        # Total tokens per step (512K)
    max_seq_len: int = 4096               # Context length

    # ========================================================================
    # Optimizer: Decoupled Learning Rates
    # ========================================================================
    # Embeddings/unembeddings use AdamW with these LRs
    embedding_lr: float = 0.2             # Input embedding LR
    unembedding_lr: float = 0.004         # Output projection LR
    embed_weight_decay: float = 0.0       # Weight decay for embed params

    # Matrix parameters use Muon (or AdamW if unavailable)
    matrix_lr: float = 0.02               # Attention/FFN matrix LR
    matrix_weight_decay: float = 0.0      # Weight decay for matrix params

    # AdamW betas (for embedding optimizer)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    # ========================================================================
    # Learning Rate Schedule
    # ========================================================================
    warmup_ratio: float = 0.0             # Warmup as fraction of total steps
    warmdown_ratio: float = 0.2           # Cosine decay as fraction
    final_lr_frac: float = 0.0            # Final LR = initial * this

    # ========================================================================
    # Gradient Configuration
    # ========================================================================
    grad_clip: float = 1.0                # Gradient norm clipping (0=disabled)

    # ========================================================================
    # MoE Load Balancing (DeepSeek V3 Aux-Loss-Free)
    # ========================================================================
    # These override model config if set (useful for experiments)
    gamma_override: Optional[float] = None          # Bias update rate
    gamma_freeze_ratio_override: Optional[float] = None  # When to freeze

    # ========================================================================
    # MTP Loss Schedule (DeepSeek V3)
    # ========================================================================
    # These override model config if set
    mtp_loss_initial_override: Optional[float] = None    # Early training weight
    mtp_loss_final_override: Optional[float] = None      # Late training weight
    mtp_transition_ratio_override: Optional[float] = None # When to switch

    # ========================================================================
    # Multi-Phase DSA Training (DeepSeek V3 Methodology)
    # ========================================================================
    # Phase 1: Dense MLA (80% tokens, 4K context) - indexer trains via aux loss
    # Phase 2: Sparse DSA (20% tokens, 8K context) - DSA enabled, YaRN enabled
    enable_multiphase: bool = False           # Enable multi-phase training
    phase1_token_fraction: float = 0.8        # Fraction of tokens for Phase 1
    phase2_context_length: int = 8192         # Context length for Phase 2
    indexer_loss_weight: float = 0.01         # Weight for indexer auxiliary loss

    # ========================================================================
    # Evaluation
    # ========================================================================
    eval_every: int = 250                 # Steps between validation
    eval_tokens: int = 10_485_760         # Tokens to evaluate (10M)
    sample_every: int = 2000              # Steps between generation samples

    # ========================================================================
    # Checkpointing
    # ========================================================================
    save_every: int = -1                  # Steps between saves (-1=only at end)
    resume_from_step: int = -1            # Resume from this step (-1=fresh)

    # ========================================================================
    # Logging
    # ========================================================================
    log_every: int = 10                   # Steps between log prints
    use_wandb: bool = False               # Enable W&B logging
    wandb_project: str = "nanoseek"       # W&B project name

    # ========================================================================
    # Note on Parallelism Strategy
    # ========================================================================
    # NanoSeek-1B (4.75B total / 1.08B active) memory per GPU with DDP:
    #   - Model weights (bf16):    ~9.5 GB
    #   - Gradients (bf16):        ~9.5 GB
    #   - AdamW states (fp32):    ~57 GB
    #   - Activations:            ~15 GB
    #   - Total:                  ~91 GB (requires H100 80GB with optimizer sharding)
    #
    # DDP is sufficient and preferred for this model size.
    # FSDP would add unnecessary communication overhead.


# ============================================================================
# Utility Functions
# ============================================================================

def print0(*args, **kwargs):
    """Print only on rank 0 in distributed training."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def autodetect_device_type() -> str:
    """
    Autodetect the best available device.

    Priority: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compute_init(device_type: str):
    """
    Initialize compute environment and distributed training.

    Returns:
        ddp: Whether DDP is enabled
        ddp_rank: Global rank (0 for non-distributed)
        ddp_local_rank: Local rank on this node
        ddp_world_size: Total number of processes
        device: torch.device for this process
    """
    # Check if running under torchrun
    ddp = int(os.environ.get('RANK', -1)) != -1

    if ddp:
        # Distributed training
        dist.init_process_group(backend='nccl')
        ddp_rank = dist.get_rank()
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = dist.get_world_size()
        device = torch.device(f'{device_type}:{ddp_local_rank}')
        torch.cuda.set_device(device)
    else:
        # Single GPU/CPU
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        device = torch.device(device_type)

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device


def compute_cleanup():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_model_config(model_size: str, max_seq_len: int) -> NanoSeekConfig:
    """
    Get NanoSeekConfig for the specified model size.

    Args:
        model_size: "1b" - NanoSeek-1B (1.08B active / 4.75B total, DeepSeek-aligned)
        max_seq_len: Maximum sequence length (overrides config default)

    Returns:
        NanoSeekConfig with sequence_length set to max_seq_len
    """
    if model_size != "1b":
        raise ValueError(f"Unknown model size: {model_size}. Use '1b' for NanoSeek-1B.")

    config = get_nanoseek_config()

    # Override sequence length
    config.sequence_length = max_seq_len
    config.max_position_embeddings = max_seq_len

    return config


def estimate_flops_per_token(config: NanoSeekConfig) -> int:
    """
    Estimate FLOPs per token for forward + backward pass.

    For MoE models, we count ACTIVE parameters only since that's what
    determines actual compute cost. This matches the nanochat comparison.

    Forward pass FLOPs ≈ 2 * active_params (matmul is 2 FLOPs per param)
    Backward pass FLOPs ≈ 4 * active_params (gradient + weight update)
    Total: 6 * active_params per token

    Note: This is an approximation. Actual FLOPs depend on:
    - Attention: O(seq_len²) for full attention
    - MLA: Reduced due to low-rank projections
    - MoE: Only active experts counted
    """
    active_params = config.estimated_active_params
    return 6 * active_params


def setup_optimizers(
    model: NanoSeekModel,
    config: TrainingConfig,
    ddp: bool = False,
) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
    """
    Setup optimizers with decoupled learning rates.

    Architecture-aware parameter grouping (following nanochat pattern):
    - Embedding params: AdamW with embedding_lr
    - Unembedding params: AdamW with unembedding_lr
    - Matrix params: Muon with matrix_lr (or AdamW fallback)

    For distributed training (ddp=True):
    - Uses DistAdamW (ZeRO-2 style sharded optimizer states)
    - Uses DistMuon (distributed gradient reduction)

    For single-GPU training (ddp=False):
    - Uses standard AdamW (or fused AdamW on CUDA)
    - Uses Muon (single-GPU Newton-Schulz)

    Args:
        model: NanoSeekModel (unwrapped, before DDP)
        config: TrainingConfig with LR settings
        ddp: Whether running in distributed mode

    Returns:
        Tuple of (adamw_optimizer, muon_optimizer or None)
    """
    # =========================================================================
    # Phase 1: Classify all parameters into their optimizer groups
    # =========================================================================
    embed_params = []       # Input embeddings → AdamW
    unembed_params = []     # Output projection (lm_head) → AdamW
    muon_params = []        # 2D matrices → Muon (if available)
    adamw_matrix_params = []  # Non-2D matrices or fallback → AdamW

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'embed_tokens' in name:
            embed_params.append(param)
        elif 'lm_head' in name:
            unembed_params.append(param)
        elif CUSTOM_OPTIMIZERS_AVAILABLE and param.ndim == 2:
            # 2D params can use Muon (if custom optimizers available)
            muon_params.append(param)
        else:
            # Non-2D params or fallback when custom optimizers unavailable
            adamw_matrix_params.append(param)

    # =========================================================================
    # Phase 2: Log parameter counts
    # =========================================================================
    print0(f"Parameter groups:")
    print0(f"  Embedding params: {sum(p.numel() for p in embed_params):,}")
    print0(f"  Unembedding params: {sum(p.numel() for p in unembed_params):,}")
    if muon_params:
        print0(f"  Muon-eligible 2D params: {sum(p.numel() for p in muon_params):,}")
    if adamw_matrix_params:
        print0(f"  AdamW matrix params (non-2D or fallback): {sum(p.numel() for p in adamw_matrix_params):,}")

    # =========================================================================
    # Phase 3: Build AdamW param groups (all at once)
    # =========================================================================
    adamw_param_groups = [
        {
            'params': embed_params,
            'lr': config.embedding_lr,
            'initial_lr': config.embedding_lr,
            'weight_decay': config.embed_weight_decay,
        },
        {
            'params': unembed_params,
            'lr': config.unembedding_lr,
            'initial_lr': config.unembedding_lr,
            'weight_decay': config.embed_weight_decay,
        },
    ]

    # Add non-2D matrix params to AdamW if any exist
    if adamw_matrix_params:
        adamw_param_groups.append({
            'params': adamw_matrix_params,
            'lr': config.matrix_lr,
            'initial_lr': config.matrix_lr,
            'weight_decay': config.matrix_weight_decay,
        })

    # =========================================================================
    # Phase 4: Create AdamW optimizer (ONCE)
    # =========================================================================
    if ddp and CUSTOM_OPTIMIZERS_AVAILABLE:
        print0("  Using DistAdamW (ZeRO-2 style)")
        adamw_optimizer = DistAdamW(
            adamw_param_groups,
            lr=config.embedding_lr,  # Default, overridden per-group
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.embed_weight_decay,
        )
    else:
        print0("  Using torch.optim.AdamW")
        adamw_optimizer = torch.optim.AdamW(
            adamw_param_groups,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            fused=torch.cuda.is_available(),
        )

    # Ensure initial_lr is set for scheduler
    for group in adamw_optimizer.param_groups:
        if 'initial_lr' not in group:
            group['initial_lr'] = group['lr']

    # =========================================================================
    # Phase 5: Create Muon optimizer (ONCE, if applicable)
    # =========================================================================
    muon_optimizer = None
    if muon_params:
        if ddp:
            print0("  Using DistMuon (distributed Newton-Schulz)")
            muon_optimizer = DistMuon(
                muon_params,
                lr=config.matrix_lr,
                momentum=0.95,
            )
        else:
            print0("  Using Muon (Newton-Schulz orthogonalization)")
            muon_optimizer = Muon(
                muon_params,
                lr=config.matrix_lr,
                momentum=0.95,
            )
        # Set initial_lr for scheduler
        for group in muon_optimizer.param_groups:
            group['initial_lr'] = config.matrix_lr

    return adamw_optimizer, muon_optimizer


# ============================================================================
# Learning Rate and Momentum Schedulers
# ============================================================================

def get_lr_multiplier(step: int, num_iterations: int, config: TrainingConfig) -> float:
    """
    Compute learning rate multiplier for the current step.

    Schedule: warmup → constant → cosine decay

        LR
        ^
        |     ┌─────────────────┐
        |    /                   ╲
        |   /                     ╲
        |  /                       ╲____
        | /
        └──────────────────────────────────> step
          warmup    constant      decay

    This matches the DeepSeek/nanochat LR schedule.
    """
    warmup_iters = round(config.warmup_ratio * num_iterations)
    warmdown_iters = round(config.warmdown_ratio * num_iterations)

    if step < warmup_iters:
        # Linear warmup
        return (step + 1) / warmup_iters
    elif step <= num_iterations - warmdown_iters:
        # Constant phase
        return 1.0
    else:
        # Cosine decay to final_lr_frac
        progress = (num_iterations - step) / warmdown_iters
        return progress * 1.0 + (1 - progress) * config.final_lr_frac


def get_muon_momentum(step: int) -> float:
    """
    Compute Muon momentum for the current step.

    Schedule: 0.85 → 0.95 over first 300 steps, then constant 0.95

    This warmup prevents early training instability from high momentum.
    """
    frac = min(step / 300, 1.0)
    return (1 - frac) * 0.85 + frac * 0.95


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_loss(
    model: NanoSeekModel,
    dataloader,
    eval_steps: int,
    device: torch.device,
    autocast_ctx,
) -> Dict[str, float]:
    """
    Evaluate model on validation data.

    Returns loss, main_loss, mtp_loss, and perplexity.
    Unlike nanochat's BPB, we use standard cross-entropy loss
    which is tokenizer-dependent but simpler to compute.
    """
    model.eval()

    total_loss = 0.0
    total_main_loss = 0.0
    total_mtp_loss = 0.0
    total_tokens = 0

    for i, (x, y) in enumerate(dataloader):
        if i >= eval_steps:
            break

        x = x.to(device)
        y = y.to(device)

        with autocast_ctx:
            outputs = model(input_ids=x, labels=y)

        batch_tokens = y.numel()
        total_loss += outputs['loss'].item() * batch_tokens
        total_main_loss += outputs['main_loss'].item() * batch_tokens
        if 'mtp_loss' in outputs:
            total_mtp_loss += outputs['mtp_loss'].item() * batch_tokens
        total_tokens += batch_tokens

    model.train()

    avg_loss = total_loss / total_tokens
    avg_main_loss = total_main_loss / total_tokens
    avg_mtp_loss = total_mtp_loss / total_tokens

    return {
        'val_loss': avg_loss,
        'val_main_loss': avg_main_loss,
        'val_mtp_loss': avg_mtp_loss,
        'val_perplexity': math.exp(avg_loss) if avg_loss < 20 else float('inf'),
    }


# ============================================================================
# Sample Text Generation (for monitoring training progress)
# ============================================================================

@torch.no_grad()
def generate_sample_text(
    model: NanoSeekModel,
    tokenizer,
    device: torch.device,
    autocast_ctx,
    prompt: str = "The quick brown fox",
    max_new_tokens: int = 64,
    temperature: float = 0.8,
) -> str:
    """
    Generate sample text from the model to monitor training progress.

    Uses greedy/sampling decoding without MTP speculative for simplicity.

    Args:
        model: NanoSeekModel (can be DDP-wrapped)
        tokenizer: Tokenizer with encode/decode methods
        device: torch.device
        autocast_ctx: Mixed precision context
        prompt: Text prompt to continue
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (1.0 = neutral)

    Returns:
        Generated text string
    """
    model.eval()

    # Handle DDP-wrapped model
    raw_model = model.module if hasattr(model, 'module') else model

    # Encode prompt
    input_ids = tokenizer.encode(prompt, allowed_special="all")
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Generate tokens autoregressively
    generated_ids = input_ids.clone()

    for _ in range(max_new_tokens):
        with autocast_ctx:
            outputs = raw_model(input_ids=generated_ids)
            logits = outputs['logits']

        # Get next token logits (last position)
        next_logits = logits[:, -1, :] / temperature

        # Sample from distribution
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append to sequence
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        # Check for EOS (if tokenizer has it)
        if hasattr(tokenizer, 'eos_id') and next_token.item() == tokenizer.eos_id:
            break

    model.train()

    # Decode and return
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    return generated_text


# ============================================================================
# Checkpoint Management (using nanochat-style checkpoint_manager)
# ============================================================================

def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    model: NanoSeekModel,
    optimizers: List[torch.optim.Optimizer],
    model_config: NanoSeekConfig,
    training_config: TrainingConfig,
    loop_state: Dict[str, Any],
    dataloader_state: Optional[Dict] = None,
    rank: int = 0,
):
    """
    Save training checkpoint using nanochat-style checkpoint_manager.

    Checkpoint structure (per-rank optimizer state for distributed):
        checkpoint_dir/
            model_{step:06d}.pt           # Model state dict (rank 0 only)
            optim_{step:06d}_rank{r}.pt   # Optimizer state (per-rank)
            meta_{step:06d}.json          # Training metadata (rank 0 only)
    """
    # Prepare model state dict
    model_data = model.state_dict()

    # Prepare optimizer states (list of state dicts)
    optimizer_data = [opt.state_dict() for opt in optimizers if opt is not None]

    # Prepare metadata
    meta_data = {
        "step": step,
        "model_config": asdict(model_config) if hasattr(model_config, '__dataclass_fields__') else model_config,
        "training_config": asdict(training_config),
        "loop_state": loop_state,
        "dataloader_state": dataloader_state,
    }

    # Use checkpoint_manager's save function (handles per-rank optimizer state)
    _save_checkpoint(
        checkpoint_dir=str(checkpoint_dir),
        step=step,
        model_data=model_data,
        optimizer_data=optimizer_data,
        meta_data=meta_data,
        rank=rank,
    )

    print0(f"Saved checkpoint to {checkpoint_dir} (step {step:06d})")


def load_checkpoint(
    checkpoint_dir: Path,
    step: int,
    model: NanoSeekModel,
    optimizers: List[torch.optim.Optimizer],
    device: torch.device,
    rank: int = 0,
) -> Tuple[Dict[str, Any], Optional[Dict]]:
    """
    Load training checkpoint using nanochat-style checkpoint_manager.

    Returns:
        loop_state: Training loop state (losses, etc.)
        dataloader_state: Dataloader resumption state
    """
    # Use checkpoint_manager's load function (handles per-rank optimizer state)
    model_data, optimizer_data, meta_data = _load_checkpoint(
        checkpoint_dir=str(checkpoint_dir),
        step=step,
        device=device,
        load_optimizer=True,
        rank=rank,
    )

    # Handle torch.compile prefix in state dict keys
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

    # Load model state
    model.load_state_dict(model_data)
    del model_data  # Free memory (nanochat pattern)

    # Load optimizer states
    if optimizer_data is not None:
        active_optimizers = [opt for opt in optimizers if opt is not None]
        for opt, state in zip(active_optimizers, optimizer_data):
            opt.load_state_dict(state)
        del optimizer_data  # Free memory (nanochat pattern)

    return meta_data.get("loop_state", {}), meta_data.get("dataloader_state")


# ============================================================================
# Simple Data Loading (placeholder - replace with your dataloader)
# ============================================================================

def create_dummy_dataloader(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    num_batches: int = 1000,
):
    """
    Create a dummy dataloader for testing.

    Replace this with your actual data loading code (e.g., FineWeb-Edu).
    """
    for _ in range(num_batches):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        yield x, y


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    """
    Main training entry point.

    Training Loop Structure:
    ========================

    For each step:
    1. Evaluation (periodic)
       - Compute validation loss
       - Generate text samples
       - Log to W&B

    2. Checkpointing (periodic)
       - Save model, optimizer, state

    3. Forward/Backward Pass
       - Gradient accumulation over micro-batches
       - Mixed precision (BF16)

    4. Optimizer Step
       - Gradient clipping
       - LR schedule update
       - Muon momentum update

    5. MoE Load Balance Update
       - Aux-loss-free bias adjustment
       - Follows V3 schedule (freeze at 80%)

    6. MTP Token Counter Update
       - For dynamic loss weight schedule

    7. Logging
       - Loss, MTP loss, aux loss
       - Gradient norm
       - Tokens/sec, MFU
    """

    # ========================================================================
    # Configuration
    # ========================================================================
    config = TrainingConfig()

    # CLI overrides via exec (nanochat-style)
    # Example: python pre-train.py --model_size=125m --num_iterations=1000
    config_keys = [k for k in TrainingConfig.__dataclass_fields__.keys()]
    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            key_value = arg[2:].split('=', 1)
            if len(key_value) == 2:
                key, value = key_value
                if key in config_keys:
                    field_type = type(getattr(config, key))
                    if field_type == bool:
                        setattr(config, key, value.lower() in ('true', '1', 'yes'))
                    elif field_type == int:
                        setattr(config, key, int(value))
                    elif field_type == float:
                        setattr(config, key, float(value))
                    else:
                        setattr(config, key, value)

    # ========================================================================
    # Compute Initialization
    # ========================================================================
    device_type = autodetect_device_type() if config.device_type == "" else config.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0

    # Mixed precision context
    if device_type == "cuda":
        autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        # Enable TF32 for faster matmuls on Ampere+
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        autocast_ctx = nullcontext()

    # Synchronization functions (no-op on non-CUDA)
    synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
    get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

    # ========================================================================
    # W&B Initialization
    # ========================================================================
    use_wandb = config.use_wandb and config.run_name != "dummy" and master_process
    if use_wandb:
        import wandb
        wandb.init(
            project=config.wandb_project,
            name=config.run_name,
            config=asdict(config),
        )

    # ========================================================================
    # Print Banner
    # ========================================================================
    print0("=" * 80)
    print0("NanoSeek Pre-training")
    print0("=" * 80)
    print0(f"Device: {device_type} (world_size={ddp_world_size})")
    print0(f"Model: NanoSeek-{config.model_size}")
    print0(f"Sequence length: {config.max_seq_len}")

    # ========================================================================
    # Model Configuration
    # ========================================================================
    model_config = get_model_config(config.model_size, config.max_seq_len)

    # Apply config overrides
    if config.gamma_override is not None:
        model_config.moe.gamma = config.gamma_override
    if config.gamma_freeze_ratio_override is not None:
        model_config.moe.gamma_freeze_ratio = config.gamma_freeze_ratio_override
    if config.mtp_loss_initial_override is not None:
        model_config.mtp.mtp_loss_weight_initial = config.mtp_loss_initial_override
    if config.mtp_loss_final_override is not None:
        model_config.mtp.mtp_loss_weight_final = config.mtp_loss_final_override
    if config.mtp_transition_ratio_override is not None:
        model_config.mtp.mtp_loss_transition_ratio = config.mtp_transition_ratio_override

    # ========================================================================
    # Batch Size Calculation
    # ========================================================================
    tokens_per_fwdbwd = config.device_batch_size * config.max_seq_len
    world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size

    assert config.total_batch_size % world_tokens_per_fwdbwd == 0, \
        f"total_batch_size ({config.total_batch_size}) must be divisible by " \
        f"world_tokens_per_fwdbwd ({world_tokens_per_fwdbwd})"

    grad_accum_steps = config.total_batch_size // world_tokens_per_fwdbwd

    print0(f"\nBatch configuration:")
    print0(f"  Tokens/micro-batch/rank: {config.device_batch_size} x {config.max_seq_len} = {tokens_per_fwdbwd:,}")
    print0(f"  Tokens/micro-batch (all ranks): {world_tokens_per_fwdbwd:,}")
    print0(f"  Total batch size: {config.total_batch_size:,}")
    print0(f"  Gradient accumulation steps: {grad_accum_steps}")

    # ========================================================================
    # Model Initialization
    # ========================================================================
    print0(f"\nInitializing model...")

    # Use meta device for memory-efficient initialization
    with torch.device("meta"):
        model = NanoSeekModel(model_config)

    # Materialize on target device
    model.to_empty(device=device)

    # Initialize weights
    model.apply(model._init_weights)

    # Keep reference to uncompiled model (for checkpointing)
    orig_model = model

    # Compile for speedup (CUDA only)
    if config.compile_model and device_type == "cuda":
        print0("Compiling model with torch.compile...")
        # fullgraph=False for MoE dynamic routing compatibility
        model = torch.compile(model, mode='reduce-overhead', fullgraph=False, dynamic=True)

    # Parameter counts
    num_params = sum(p.numel() for p in model.parameters())
    active_params = model_config.estimated_active_params
    total_params = model_config.estimated_total_params

    print0(f"\nModel parameters:")
    print0(f"  Total (all experts): {total_params:,} ({total_params/1e9:.2f}B)")
    print0(f"  Active (per forward): {active_params:,} ({active_params/1e6:.0f}M)")
    print0(f"  Trainable: {num_params:,}")

    # FLOP estimation (based on ACTIVE params for fair comparison)
    num_flops_per_token = estimate_flops_per_token(model_config)
    print0(f"  FLOPs/token (active): {num_flops_per_token:e}")

    # ========================================================================
    # Training Horizon Calculation
    # ========================================================================
    print0(f"\nTraining horizon:")

    if config.num_iterations > 0:
        num_iterations = config.num_iterations
        print0(f"  Using explicit num_iterations: {num_iterations:,}")
    elif config.target_flops > 0:
        num_iterations = round(config.target_flops / (num_flops_per_token * config.total_batch_size))
        print0(f"  From target_flops ({config.target_flops:e}): {num_iterations:,}")
    elif config.target_param_data_ratio > 0:
        # Chinchilla-optimal: tokens = ratio * active_params
        target_tokens = config.target_param_data_ratio * active_params
        num_iterations = int(target_tokens // config.total_batch_size)
        print0(f"  From Chinchilla ratio ({config.target_param_data_ratio}x): {num_iterations:,}")
    else:
        raise ValueError("Must specify num_iterations, target_flops, or target_param_data_ratio")

    total_tokens = config.total_batch_size * num_iterations
    print0(f"  Total tokens: {total_tokens:,} ({total_tokens/1e9:.2f}B)")
    print0(f"  Tokens:Params ratio: {total_tokens / active_params:.2f}x")
    print0(f"  Total FLOPs: {num_flops_per_token * total_tokens:e}")

    # Update model config with computed total_tokens (for ratio-based schedules)
    model_config.total_tokens = total_tokens

    # ========================================================================
    # Multi-Phase Training Configuration (DeepSeek V3 DSA Methodology)
    # ========================================================================
    # Phase 1: Dense MLA (80% tokens, 4K context) - indexer trains via aux loss
    # Phase 2: Sparse DSA (20% tokens, 8K context) - DSA enabled, YaRN enabled

    current_phase = 0  # 0 = Phase 1 (dense), 1 = Phase 2 (sparse)
    phase_step = 0     # Steps within current phase

    if config.enable_multiphase:
        print0(f"\n{'='*80}")
        print0("Multi-Phase DSA Training Enabled")
        print0(f"{'='*80}")

        # Customize phases based on config
        phase1 = TrainingPhaseConfig(
            name="phase1_dense_mla",
            sequence_length=config.max_seq_len,
            global_batch_size=config.total_batch_size // config.max_seq_len,
            token_fraction=config.phase1_token_fraction,
            learning_rate=config.matrix_lr,
            lr_min=config.matrix_lr * config.final_lr_frac,
            dsa_enabled=False,            # Dense attention
            dsa_activation_threshold=config.max_seq_len,
            yarn_enabled=False,
            warmup_steps=int(config.warmup_ratio * num_iterations * config.phase1_token_fraction),
        )

        phase2 = TrainingPhaseConfig(
            name="phase2_sparse_dsa",
            sequence_length=config.phase2_context_length,
            global_batch_size=config.total_batch_size // config.phase2_context_length,
            token_fraction=1.0 - config.phase1_token_fraction,
            learning_rate=config.matrix_lr * 0.33,  # Lower LR for fine-tuning
            lr_min=config.matrix_lr * 0.1 * config.final_lr_frac,
            dsa_enabled=True,
            dsa_activation_threshold=config.max_seq_len,  # Sparse for seq > 4K
            yarn_enabled=True,
            warmup_steps=100,
        )

        training_phases = [phase1, phase2]

        # Calculate steps per phase
        phase1_tokens = int(total_tokens * phase1.token_fraction)
        phase2_tokens = int(total_tokens * phase2.token_fraction)
        phase1_steps = phase1_tokens // config.total_batch_size
        phase2_steps = phase2_tokens // (config.total_batch_size // 2)  # Reduced batch for 2x context

        print0(f"\nPhase 1: Dense MLA Training")
        print0(f"  Context:     {phase1.sequence_length:,} tokens")
        print0(f"  Tokens:      {phase1_tokens:,} ({phase1_tokens/1e9:.2f}B, {phase1.token_fraction*100:.0f}%)")
        print0(f"  Steps:       ~{phase1_steps:,}")
        print0(f"  DSA:         Disabled (indexer trains via aux loss)")
        print0(f"  YaRN:        Disabled")

        print0(f"\nPhase 2: Sparse DSA Training")
        print0(f"  Context:     {phase2.sequence_length:,} tokens")
        print0(f"  Tokens:      {phase2_tokens:,} ({phase2_tokens/1e9:.2f}B, {phase2.token_fraction*100:.0f}%)")
        print0(f"  Steps:       ~{phase2_steps:,}")
        print0(f"  DSA:         Enabled (Lightning Indexer active)")
        print0(f"  YaRN:        Enabled (context extension)")

        # Apply Phase 1 config initially
        model_config = apply_phase_config(model_config, phase1)
        print0(f"\nStarting with Phase 1 configuration...")
    else:
        training_phases = None
        phase1_steps = num_iterations  # Single phase = all iterations

    # ========================================================================
    # Optimizer Setup
    # ========================================================================
    print0(f"\nSetting up optimizers...")
    adamw_optimizer, muon_optimizer = setup_optimizers(orig_model, config, ddp=ddp)
    optimizers = [adamw_optimizer, muon_optimizer] if muon_optimizer else [adamw_optimizer]

    # ========================================================================
    # Distributed Data Parallel (DDP)
    # ========================================================================
    # DDP is sufficient for NanoSeek-1B (4.75B total params):
    #   - Each GPU holds full model replica (~52GB with optimizer states)
    #   - H100 80GB has ample headroom
    #   - No sharding overhead = faster training
    if ddp:
        print0(f"\nWrapping model with DDP...")
        model = DDP(model, device_ids=[ddp_local_rank])

    # ========================================================================
    # Data Loading
    # ========================================================================
    print0(f"\nSetting up data loading...")

    # Track dataloader state for checkpointing
    dataloader_resume_state = None

    if DATALOADER_AVAILABLE:
        print0("  Using streaming dataloader from parquet files")

        def build_train_loader(resume_state=None):
            """Build streaming train dataloader with optional resume state."""
            return tokenizing_distributed_data_loader_with_state(
                B=config.device_batch_size,
                T=config.max_seq_len,
                split="train",
                tokenizer_threads=4,
                tokenizer_batch_size=128,
                device=device,
                resume_state_dict=resume_state,
            )

        def build_val_loader():
            """Build streaming validation dataloader (no state tracking needed)."""
            return tokenizing_distributed_data_loader(
                B=config.device_batch_size,
                T=config.max_seq_len,
                split="val",
                tokenizer_threads=4,
                tokenizer_batch_size=128,
                device=device,
            )

        train_loader = build_train_loader(resume_state=dataloader_resume_state)

        # Prefetch first batch (with state_dict)
        x, y, dataloader_resume_state = next(train_loader)
    else:
        print0("  WARNING: Using dummy data. Real dataloader not available.")

        def build_train_loader(resume_state=None):
            return create_dummy_dataloader(
                batch_size=config.device_batch_size,
                seq_len=config.max_seq_len,
                vocab_size=model_config.vocab_size,
                device=device,
                num_batches=num_iterations * grad_accum_steps * 2,
            )

        def build_val_loader():
            return create_dummy_dataloader(
                batch_size=config.device_batch_size,
                seq_len=config.max_seq_len,
                vocab_size=model_config.vocab_size,
                device=device,
                num_batches=100,
            )

        train_loader = build_train_loader()
        train_iter = iter(train_loader)

        # Prefetch first batch
        x, y = next(train_iter)

    # ========================================================================
    # Checkpoint Directory
    # ========================================================================
    # Use consistent path with downstream scripts (mid_train, chat_sft, etc.)
    # Format: data/base_checkpoints/d{layers}/ (e.g., d18 for 350M model)
    base_dir = Path(get_base_dir())
    model_tag = config.model_tag or f"d{model_config.num_layers}"
    checkpoint_dir = base_dir / "base_checkpoints" / model_tag
    if master_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print0(f"Checkpoint directory: {checkpoint_dir}")

    # ========================================================================
    # Tokenizer (for sample text generation - nanochat pattern)
    # ========================================================================
    tokenizer = None
    if TOKENIZER_AVAILABLE:
        try:
            tokenizer = get_tokenizer()
            print0(f"Loaded tokenizer with vocab size: {tokenizer.get_vocab_size()}")
        except Exception as e:
            print0(f"Warning: Could not load tokenizer: {e}")
            tokenizer = None

    # ========================================================================
    # Resume from Checkpoint
    # ========================================================================
    resuming = config.resume_from_step > 0

    if resuming:
        print0(f"\nResuming from step {config.resume_from_step}...")
        loop_state, saved_dataloader_state = load_checkpoint(
            checkpoint_dir, config.resume_from_step, orig_model, optimizers, device,
            rank=ddp_rank,
        )
        step = loop_state["step"]
        min_val_loss = loop_state["min_val_loss"]
        smooth_train_loss = loop_state["smooth_train_loss"]
        total_training_time = loop_state["total_training_time"]
        tokens_processed = loop_state.get("tokens_processed", step * config.total_batch_size)
        orig_model.tokens_processed.fill_(tokens_processed)

        # Restore multi-phase state if available
        if config.enable_multiphase:
            current_phase = loop_state.get("current_phase", 0)
            phase_step = loop_state.get("phase_step", 0)
            if current_phase == 1:
                # Already in Phase 2, apply Phase 2 config
                print0(f"Resuming in Phase 2 (Sparse DSA)")
                model_config = apply_phase_config(model_config, training_phases[1])
                config.max_seq_len = training_phases[1].sequence_length
            else:
                print0(f"Resuming in Phase 1 (Dense MLA), phase_step={phase_step}")

        # Restore dataloader state if available
        if saved_dataloader_state is not None:
            if DATALOADER_AVAILABLE:
                # Rebuild train loader with resume state
                dataloader_resume_state = saved_dataloader_state
                train_loader = build_train_loader(resume_state=dataloader_resume_state)
                x, y, dataloader_resume_state = next(train_loader)
                print0(f"Restored dataloader state: pq_idx={saved_dataloader_state.get('pq_idx')}, rg_idx={saved_dataloader_state.get('rg_idx')}")
            else:
                # Dummy dataloader doesn't support state
                print0("Warning: Dataloader state found but using dummy dataloader")
    else:
        step = 0
        min_val_loss = float("inf")
        smooth_train_loss = 0.0
        total_training_time = 0.0

    # ========================================================================
    # Training Loop
    # ========================================================================
    print0(f"\n{'='*80}")
    print0("Starting training...")
    print0(f"{'='*80}\n")

    model.train()

    while True:
        last_step = step == num_iterations
        flops_so_far = num_flops_per_token * config.total_batch_size * step

        # ====================================================================
        # Evaluation (periodic)
        # ====================================================================
        if last_step or step % config.eval_every == 0:
            eval_steps = config.eval_tokens // (config.device_batch_size * config.max_seq_len * ddp_world_size)
            val_loader = build_val_loader()

            with autocast_ctx:
                metrics = evaluate_loss(model, val_loader, eval_steps, device, autocast_ctx)

            print0(f"Step {step:05d} | Val loss: {metrics['val_loss']:.4f} | "
                   f"Val PPL: {metrics['val_perplexity']:.2f}")

            if metrics['val_loss'] < min_val_loss:
                min_val_loss = metrics['val_loss']

            if use_wandb:
                wandb.log({
                    "step": step,
                    "total_flops": flops_so_far,
                    "val/loss": metrics['val_loss'],
                    "val/main_loss": metrics['val_main_loss'],
                    "val/mtp_loss": metrics['val_mtp_loss'],
                    "val/perplexity": metrics['val_perplexity'],
                })

        # ====================================================================
        # Sample Text Generation (periodic - nanochat pattern)
        # ====================================================================
        if master_process and tokenizer is not None and step % config.sample_every == 0 and step > 0:
            sample_prompts = [
                "The quick brown fox",
                "Once upon a time",
                "In the year 2050",
            ]
            print0(f"\n--- Sample generations at step {step} ---")
            for prompt in sample_prompts:
                try:
                    generated = generate_sample_text(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        autocast_ctx=autocast_ctx,
                        prompt=prompt,
                        max_new_tokens=64,
                        temperature=0.8,
                    )
                    # Truncate for display
                    display_text = generated[:200] + "..." if len(generated) > 200 else generated
                    print0(f"Prompt: '{prompt}'\n  → {display_text}")
                except Exception as e:
                    print0(f"Warning: Generation failed for prompt '{prompt}': {e}")
            print0("--- End sample generations ---\n")

        # ====================================================================
        # Checkpointing (periodic)
        # ====================================================================
        should_save = (
            last_step or
            (config.save_every > 0 and step > 0 and step != config.resume_from_step
             and step % config.save_every == 0)
        )

        if should_save:
            loop_state = {
                "step": step,
                "min_val_loss": min_val_loss,
                "smooth_train_loss": smooth_train_loss,
                "total_training_time": total_training_time,
                "tokens_processed": orig_model.tokens_processed.item(),
                # Multi-phase state
                "current_phase": current_phase,
                "phase_step": phase_step,
            }
            # Save dataloader state for approximate resume
            # For streaming dataloader, we track pq_idx and rg_idx
            save_checkpoint(
                checkpoint_dir, step, orig_model, optimizers,
                model_config, config, loop_state,
                dataloader_resume_state if DATALOADER_AVAILABLE else None,
                ddp_rank
            )

        # ====================================================================
        # Termination Check
        # ====================================================================
        if last_step:
            break

        # ====================================================================
        # Phase Transition Check (Multi-Phase DSA Training)
        # ====================================================================
        if config.enable_multiphase and current_phase == 0 and phase_step >= phase1_steps:
            print0(f"\n{'='*80}")
            print0("Phase Transition: Dense MLA → Sparse DSA")
            print0(f"{'='*80}")

            # Save Phase 1 checkpoint
            phase1_checkpoint = {
                "step": step,
                "phase": 0,
                "min_val_loss": min_val_loss,
                "smooth_train_loss": smooth_train_loss,
                "total_training_time": total_training_time,
                "tokens_processed": orig_model.tokens_processed.item(),
            }
            if master_process:
                save_checkpoint(
                    checkpoint_dir / "phase1_final", step, orig_model, optimizers,
                    model_config, config, phase1_checkpoint,
                    dataloader_resume_state if DATALOADER_AVAILABLE else None,
                    ddp_rank
                )
                print0(f"Saved Phase 1 final checkpoint")

            # Transition to Phase 2
            current_phase = 1
            phase_step = 0
            phase2 = training_phases[1]

            # Apply Phase 2 config
            model_config = apply_phase_config(model_config, phase2)

            # Update RoPE frequencies for YaRN context extension
            if phase2.yarn_enabled:
                rope_scaling_factor = phase2.sequence_length / phase1.sequence_length
                print0(f"Updating RoPE for YaRN context extension: factor={rope_scaling_factor:.2f}")
                orig_model.update_rope_for_context_extension(
                    new_max_position_embeddings=phase2.sequence_length,
                    rope_scaling_factor=rope_scaling_factor,
                    original_max_position_embeddings=phase1.sequence_length,
                )

            # Update sequence length for dataloader
            print0(f"Updating context length: {phase1.sequence_length} → {phase2.sequence_length}")
            config.max_seq_len = phase2.sequence_length

            # Rebuild dataloader with new sequence length
            if DATALOADER_AVAILABLE:
                print0("Rebuilding dataloader for Phase 2...")
                train_loader = build_train_loader(resume_state=None)  # Fresh start
                x, y, dataloader_resume_state = next(train_loader)
            else:
                train_iter = iter(build_train_loader())
                x, y = next(train_iter)

            # Update grad accumulation for potentially different batch size
            tokens_per_fwdbwd = config.device_batch_size * config.max_seq_len
            world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
            grad_accum_steps = config.total_batch_size // world_tokens_per_fwdbwd

            # Reset LR schedule for Phase 2 (with brief warmup)
            for opt in optimizers:
                if opt is not None:
                    for group in opt.param_groups:
                        group['initial_lr'] = group['initial_lr'] * 0.33  # Lower LR for Phase 2

            print0(f"\nPhase 2 Active:")
            print0(f"  Context:     {config.max_seq_len:,} tokens")
            print0(f"  DSA:         Enabled")
            print0(f"  YaRN:        Enabled")
            print0(f"  Grad accum:  {grad_accum_steps}")
            print0(f"{'='*80}\n")

        # ====================================================================
        # Training Step
        # ====================================================================
        synchronize()
        t0 = time.time()

        # Gradient accumulation
        indexer_loss_total = 0.0
        for micro_step in range(grad_accum_steps):
            with autocast_ctx:
                outputs = model(input_ids=x, labels=y)
                loss = outputs['loss']

                # Phase 1: Add indexer auxiliary loss (trains indexer via KL-divergence)
                # The indexer learns to predict which tokens will get high attention
                # by aligning with actual attention patterns, even when DSA is disabled
                if config.enable_multiphase and current_phase == 0:
                    if 'indexer_loss' in outputs:
                        indexer_aux = outputs['indexer_loss'] * config.indexer_loss_weight
                        loss = loss + indexer_aux
                        indexer_loss_total += indexer_aux.detach().item()

            # Track loss for logging (last micro-batch)
            train_loss = loss.detach()

            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps
            loss.backward()

            # Prefetch next batch
            if DATALOADER_AVAILABLE:
                # Streaming dataloader yields (inputs, targets, state_dict)
                x, y, dataloader_resume_state = next(train_loader)
            else:
                # Dummy dataloader
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(build_train_loader())
                    x, y = next(train_iter)

        # Gradient clipping
        grad_norm = 0.0
        if config.grad_clip > 0.0:
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                orig_model.parameters(), config.grad_clip
            )
            grad_norm = grad_norm_tensor.item()

        # Learning rate update
        lrm = get_lr_multiplier(step, num_iterations, config)
        for opt in optimizers:
            if opt is not None:
                for group in opt.param_groups:
                    group['lr'] = group['initial_lr'] * lrm

        # Muon momentum update
        if muon_optimizer is not None:
            muon_momentum = get_muon_momentum(step)
            for group in muon_optimizer.param_groups:
                group['momentum'] = muon_momentum

        # Optimizer step
        for opt in optimizers:
            if opt is not None:
                opt.step()

        # Zero gradients
        model.zero_grad(set_to_none=True)

        # ====================================================================
        # MoE Load Balance Bias Update (DeepSeek V3)
        # ====================================================================
        # Uses ratio-based schedule: gamma = 0.001 until 80% of training, then 0
        gamma = orig_model.get_gamma()
        orig_model.update_load_balance_bias(gamma)

        # ====================================================================
        # MTP Token Counter Update
        # ====================================================================
        # For dynamic MTP loss weight schedule (V3)
        orig_model.update_tokens_processed(config.total_batch_size)

        # ====================================================================
        # DSA Training Step Update (Multi-Phase Training)
        # ====================================================================
        # Tracks warm-up phase for DSA sparse attention activation
        if config.enable_multiphase:
            orig_model.increment_dsa_training_steps()

        synchronize()
        t1 = time.time()
        dt = t1 - t0

        # ====================================================================
        # Logging
        # ====================================================================
        if step > 10:
            total_training_time += dt

        # EMA smoothing for loss
        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))

        # Throughput metrics
        tok_per_sec = int(config.total_batch_size / dt)
        flops_per_sec = num_flops_per_token * config.total_batch_size / dt

        # MFU (Model FLOPs Utilization) - theoretical max for H100 SXM BF16
        promised_flops_per_sec_h100 = 989e12 * ddp_world_size
        mfu = 100 * flops_per_sec / promised_flops_per_sec_h100

        # Get loss components
        main_loss = outputs.get('main_loss', train_loss).item()
        mtp_loss = outputs.get('mtp_loss', torch.tensor(0.0)).item()
        mtp_weight = outputs.get('mtp_weight', 0.0)
        aux_loss = outputs.get('aux_loss', torch.tensor(0.0))
        aux_loss = aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss

        # Print log
        if step % config.log_every == 0:
            pct_done = 100 * step / num_iterations

            # Add phase indicator for multi-phase training
            phase_str = ""
            if config.enable_multiphase:
                phase_name = "Dense" if current_phase == 0 else "Sparse"
                phase_str = f"[P{current_phase+1}:{phase_name}] "

            log_parts = [
                f"{phase_str}step {step:05d}/{num_iterations:05d} ({pct_done:.1f}%)",
                f"loss: {debiased_smooth_loss:.4f}",
                f"main: {main_loss:.4f}",
                f"mtp: {mtp_loss:.4f} (λ={mtp_weight:.2f})",
            ]

            # Add indexer loss for Phase 1
            if config.enable_multiphase and current_phase == 0 and indexer_loss_total > 0:
                log_parts.append(f"idx: {indexer_loss_total:.4f}")

            if config.grad_clip > 0:
                log_parts.append(f"grad: {grad_norm:.4f}")

            log_parts.extend([
                f"lrm: {lrm:.2f}",
                f"γ: {gamma:.4f}",
                f"tok/s: {tok_per_sec:,}",
                f"mfu: {mfu:.1f}%",
                f"dt: {dt*1000:.0f}ms",
            ])

            print0(" | ".join(log_parts))

            # W&B logging
            if use_wandb and step % 100 == 0:
                log_dict = {
                    "step": step,
                    "total_flops": flops_so_far,
                    "total_training_time": total_training_time,
                    "train/loss": debiased_smooth_loss,
                    "train/main_loss": main_loss,
                    "train/mtp_loss": mtp_loss,
                    "train/mtp_weight": mtp_weight,
                    "train/aux_loss": aux_loss,
                    "train/gamma": gamma,
                    "train/grad_norm": grad_norm,
                    "train/lrm": lrm,
                    "train/tok_per_sec": tok_per_sec,
                    "train/mfu": mfu,
                    "train/dt_ms": dt * 1000,
                }

                # Add multi-phase training metrics
                if config.enable_multiphase:
                    log_dict["train/phase"] = current_phase
                    log_dict["train/phase_step"] = phase_step
                    if current_phase == 0:
                        log_dict["train/indexer_loss"] = indexer_loss_total
                    log_dict["train/dsa_enabled"] = current_phase == 1

                wandb.log(log_dict)

        step += 1
        phase_step += 1  # Track steps within current phase

    # ========================================================================
    # Training Complete
    # ========================================================================
    print0(f"\n{'='*80}")
    print0("Training complete!")
    print0(f"{'='*80}")
    print0(f"Total steps: {step}")
    print0(f"Total tokens: {step * config.total_batch_size:,}")
    print0(f"Total training time: {total_training_time/60:.2f}m")
    print0(f"Best validation loss: {min_val_loss:.4f}")
    print0(f"Peak memory: {get_max_memory() / 1e9:.2f}GB")

    # Multi-phase training summary
    if config.enable_multiphase:
        print0(f"\nMulti-Phase Training Summary:")
        print0(f"  Phase 1 (Dense MLA): {phase1_steps:,} steps at {phase1.sequence_length} context")
        if current_phase == 1:
            print0(f"  Phase 2 (Sparse DSA): {phase_step:,} steps at {phase2.sequence_length} context")
            print0(f"  DSA Status: Enabled (Lightning Indexer active)")
            print0(f"  YaRN Status: Enabled (context extension ready)")
        else:
            print0(f"  Note: Training ended in Phase 1 (did not reach Phase 2)")

    if use_wandb:
        wandb.finish()

    compute_cleanup()


if __name__ == "__main__":
    main()
