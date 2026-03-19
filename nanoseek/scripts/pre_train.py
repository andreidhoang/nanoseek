"""
NanoSeek Stage 1 Pre-Training Script.

Usage:
    Single GPU:  python -m nanoseek.scripts.pre_train
    Multi-GPU:   torchrun --nproc_per_node=8 -m nanoseek.scripts.pre_train

Adapts nanochat's base_train.py for NanoSeek's MoE + MLA + MTP architecture.
Key differences from nanochat:
    1. Model returns dict, not scalar loss
    2. muP 1/width scaling (not 1/√width)
    3. Cosine decay (not linear warmdown)
    4. Gradient clipping (MoE gradient variance)
    5. EMA tracking for all evaluation (RULE 1)
    6. MoE load-balance bias update after each step
    7. Batch size warmup (1/5 → 1× over first 10%)
    8. Phase transition (4K→8K at 80% of training)
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import time
import math
import argparse
from dataclasses import asdict
from contextlib import contextmanager

import wandb
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_

from nanoseek.nanoseek.config import (
    NanoSeekConfig,
    get_nanoseek_config,
    get_nanoseek_500m_config,
    get_nanoseek_anchor_config,
    get_training_phases,
    apply_phase_config,
)
from nanoseek.nanoseek.model import NanoSeekModel

# Nanochat imports - change later
from nanoseek.nanoseek.optim import MuonAdamW, DistMuonAdamW
from nanoseek.nanoseek.common import (
    compute_init, compute_cleanup, print0,
    DummyWandb, autodetect_device_type, get_peak_flops,
    COMPUTE_DTYPE, is_ddp_initialized,
)
from nanoseek.nanoseek.tokenizer import get_tokenizer, get_token_bytes
from nanoseek.nanoseek.dataloader import (
    tokenizing_distributed_data_loader_bos_bestfit,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from nanoseek.nanoseek.checkpoint_manager import (
    save_checkpoint, load_checkpoint, load_optimizer_checkpoint,
    load_ema_checkpoint, CheckpointManager,
)

# Eval modules (RULE 7: H_load + I_spec; RULE 9: MTP acceptance rate)
from nanoseek.eval.information_metrics import compute_i_spec
from nanoseek.eval.moe_diagnostics import compute_mtp_acceptance_rate, compute_dead_experts
from nanoseek.eval.domain_bpb import compute_domain_bpb


parser = argparse.ArgumentParser(description="NanoSeek Stage 1 Pre-Training")

# Run configuration
parser.add_argument("--run", type=str, default="dummy",
                    help="wandb run name ('dummy' disables logging)")
parser.add_argument("--device-type", type=str, default="",
                    help="cuda|cpu|mps (empty = autodetect)")

# Model scale selection (muP transfer path)
parser.add_argument("--scale", type=str, default="1b",
                    choices=["anchor", "500m", "1b"],
                    help="which config to use: anchor(480h), 500m(1280h), 1b(2048h)")
# muP anchor reference (DO NOT CHANGE unless retuning anchor)
parser.add_argument("--mup-ref-width", type=int, default=480,
                    help="anchor hidden_size for muP scaling (must match anchor config)")
parser.add_argument("--mup-ref-batch-tokens", type=int, default=262144,
                    help="anchor batch size in tokens (64 × 4096)")

# muP base learning rates (tuned at anchor scale)
parser.add_argument("--matrix-lr", type=float, default=0.02,
                    help="base Muon LR for hidden weights (tuned at anchor)")
parser.add_argument("--embedding-lr", type=float, default=0.3,
                    help="base AdamW LR for embeddings (tuned at anchor)")
parser.add_argument("--unembedding-lr", type=float, default=0.008,
                    help="base AdamW LR for lm_head (tuned at anchor)")
parser.add_argument("--router-lr", type=float, default=3e-4,
                    help="AdamW LR for router weights (CONSTANT across scales)")
parser.add_argument("--norm-lr", type=float, default=3e-4,
                    help="AdamW LR for norm parameters (CONSTANT across scales)")

# Weight decay
parser.add_argument("--weight-decay", type=float, default=0.1,
                    help="base weight decay for Muon groups")

# Training iterations and batch
parser.add_argument("--num-iterations", type=int, default=-1,
                    help="override total iterations (-1 = compute from total_tokens)")
parser.add_argument("--device-batch-size", type=int, default=4,
                    help="micro-batch size per GPU (sequences)")

# Evaluation
parser.add_argument("--eval-every", type=int, default=250,
                    help="evaluate ema_val_bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=10_000_000,
                    help="number of tokens for validation evaluation")
parser.add_argument("--save-every", type=int, default=1000,
                    help="save checkpoint every N steps (-1 = only at end)")

# EMA configuration
parser.add_argument("--ema-decay", type=float, default=0.9999,
                    help="EMA decay rate (Polyak averaging)")
parser.add_argument("--ema-every", type=int, default=10,
                    help="update EMA every N steps")

# Resume
parser.add_argument("--resume-from-step", type=int, default=-1,
                help="resume from checkpoint at this step")

# Reproducibility
parser.add_argument("--seed", type=int, default=42,
                    help="random seed for reproducibility")

# ─── Ablation flags (Phase 3 stability & architecture experiments) ───
parser.add_argument("--no-seq-aux", action="store_true",
                    help="Ablation: disable sequence-level auxiliary loss (set alpha=0)")
parser.add_argument("--no-grad-clip", action="store_true",
                    help="Ablation: disable gradient clipping (set max_norm=inf)")
parser.add_argument("--aux-loss-type", type=str, default="bias",
                    choices=["bias", "classic"],
                    help="Ablation: 'bias' = aux-loss-free balancing (V3), "
                         "'classic' = traditional aux loss (alpha=0.01)")
parser.add_argument("--no-mtp", action="store_true",
                    help="Ablation: disable MTP (set lambda=0 throughout training)")
parser.add_argument("--no-shared-experts", action="store_true",
                    help="Ablation: remove shared expert contribution (zero out)")
parser.add_argument("--inject-bad-batch", type=int, default=-1,
                    help="Ablation: multiply gradient by 10x at this step (-1 = disabled)")
parser.add_argument("--no-compile", action="store_true",
                    help="Skip torch.compile (useful for debugging or incompatible GPUs)")

# ─── Architecture override flags (Phase 3 architecture experiments) ───
parser.add_argument("--num-experts", type=int, default=-1,
                    help="Override n_routed_experts (e.g. 16 for fewer-experts ablation)")
parser.add_argument("--top-k", type=int, default=-1,
                    help="Override num_experts_per_tok (e.g. 2 for fewer-experts ablation)")
parser.add_argument("--n-group", type=int, default=-1,
                    help="Override n_group for routing (e.g. 4 for 16-expert config)")
parser.add_argument("--topk-group", type=int, default=-1,
                    help="Override topk_group for routing")

# ─── Scaling law sweep overrides ───
parser.add_argument("--total-tokens", type=int, default=-1,
                    help="Override total training tokens (-1 = use config default)")
parser.add_argument("--hidden-size", type=int, default=-1,
                    help="Override hidden_size (for scaling sweep configs)")

# ─── Config from YAML ───
parser.add_argument("--config-yaml", type=str, default="",
                    help="Path to YAML config file (overrides individual CLI flags)")

args = parser.parse_args()
user_config = vars(args).copy()  # for logging

# -----------------------------------------------------------------------------
# Seed management for reproducibility
# Without this, every run produces different results — can't reproduce ablations.
import random
import numpy as np
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # for multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
# Compute init and wandb logging
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
# ─── GPU info ───
if device_type == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_name)
    print0(f"GPU: {gpu_name} | Peak BF16 FLOPS: {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')

# ─── W&B with ablation metadata ───
# Ablation taxonomy: each run is tagged with its ablation group + specific variant
# so that W&B filtering works: "show me all stability runs", "compare A vs C", etc.
def _classify_ablation(run_name, args):
    """Derive W&B group, tags, and notes from run name and ablation flags."""
    tags = [f"scale:{args.scale}"]
    group = f"{args.scale}"  # default: group by scale
    notes = ""

    # Detect ablation group from run name prefix
    # HP search: hp-r1-* (Round 1 coarse), hp-r2-* (Round 2 fine),
    #            hp-seed-* (Round 3 multi-seed), hp-wd-* (Round 4 weight decay),
    #            hp-anchor-* (legacy Round 1 naming)
    if run_name.startswith("hp-"):
        tags.append("ablation:hp-transfer")
        group = f"hp-{args.scale}"
        if run_name.startswith("hp-r1-"):
            tags.append("hp-round:1")
            notes = "HP grid Round 1: coarse grid search"
        elif run_name.startswith("hp-r2-"):
            tags.append("hp-round:2")
            notes = "HP grid Round 2: fine grid around R1 winner"
        elif run_name.startswith("hp-seed-"):
            tags.append("hp-round:3")
            tags.append(f"seed:{args.seed}")
            notes = "HP grid Round 3: multi-seed validation"
        elif run_name.startswith("hp-wd-"):
            tags.append("hp-round:4")
            tags.append(f"weight_decay:{args.weight_decay}")
            notes = "HP grid Round 4: weight decay sensitivity"
        elif run_name.startswith("hp-anchor-"):
            tags.append("hp-round:1")
            tags.append("legacy-naming")
            notes = "HP grid Round 1: coarse grid search (legacy naming)"
        else:
            notes = "muP hyperparameter transfer validation"
    elif run_name.startswith("stab-"):
        tags.append("ablation:stability")
        group = f"stability-{args.scale}"
        notes = "Training stability ablation (DeepSeek V3.2 techniques)"
    elif run_name.startswith("arch-"):
        tags.append("ablation:architecture")
        group = f"architecture-{args.scale}"
        notes = "Architecture sensitivity ablation"
    elif run_name.startswith("gate1"):
        tags.append("gate:smoke-test")
        group = f"gate1-{args.scale}"
        notes = "Gate 1 smoke test (100 steps)"

    # Tag specific ablation flags
    if args.no_seq_aux:
        tags.append("variant:no-seq-aux")
    if args.no_grad_clip:
        tags.append("variant:no-grad-clip")
    if args.aux_loss_type == "classic":
        tags.append("variant:classic-aux-loss")
    if args.no_mtp:
        tags.append("variant:no-mtp")
    if args.no_shared_experts:
        tags.append("variant:no-shared-experts")
    if args.inject_bad_batch >= 0:
        tags.append(f"variant:bad-batch-{args.inject_bad_batch}")
    if args.num_experts > 0:
        tags.append(f"variant:experts-{args.num_experts}")
    if args.top_k > 0:
        tags.append(f"variant:topk-{args.top_k}")
    if args.config_yaml:
        tags.append(f"config:{os.path.basename(args.config_yaml)}")

    return group, tags, notes

ablation_group, ablation_tags, ablation_notes = _classify_ablation(args.run, args)

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
    project="nanoseek",
    name=args.run,
    group=ablation_group,
    tags=ablation_tags,
    notes=ablation_notes,
    config=vars(args),  # CLI args (pre-override)
)
# ─── Select config from muP transfer path ───
config_map = {
    "anchor": get_nanoseek_anchor_config,
    "500m":   get_nanoseek_500m_config,
    "1b":     get_nanoseek_config,
}
config = config_map[args.scale]()

print0(f"Model scale: {args.scale}")
print0(f"  hidden_size:   {config.hidden_size}")
print0(f"  num_layers:    {config.num_layers}")
print0(f"  n_experts:     {config.moe.n_routed_experts}")
print0(f"  top_k:         {config.moe.num_experts_per_tok}")
print0(f"  moe_inter:     {config.moe.moe_intermediate_size}")
print0(f"  vocab_size:    {config.vocab_size}")

# ─── Config validation (catch CLAUDE.md rule violations before training) ───
def validate_config(cfg):
    """Sanity-check config against CLAUDE.md critical rules."""
    errors = []
    if cfg.moe.gamma_freeze_ratio != 0.95:
        errors.append(f"RULE 2: gamma_freeze_ratio={cfg.moe.gamma_freeze_ratio}, must be 0.95")
    if cfg.adam_beta2 != 0.95:
        errors.append(f"adam_beta2={cfg.adam_beta2}, expected 0.95 (DeepSeek style)")
    if cfg.max_grad_norm != 1.0 and not args.no_grad_clip:
        errors.append(f"max_grad_norm={cfg.max_grad_norm}, expected 1.0")
    if errors:
        for e in errors:
            print0(f"  CONFIG ERROR: {e}")
        raise ValueError(f"Config validation failed: {len(errors)} error(s)")
    print0("  Config validation: PASSED")

# ─── Apply ablation overrides BEFORE validation ───
if args.no_seq_aux:
    config.moe.seq_aux_loss_alpha = 0.0
    print0("ABLATION: seq_aux disabled (alpha=0)")

if args.no_grad_clip:
    config.max_grad_norm = float('inf')
    print0("ABLATION: gradient clipping disabled (max_norm=inf)")

if args.aux_loss_type == "classic":
    # Classic aux loss: higher alpha, no bias-based balancing
    config.moe.seq_aux_loss_alpha = 0.01
    print0("ABLATION: classic aux loss (alpha=0.01, bias updates will be skipped)")

if args.no_mtp:
    config.mtp.mtp_loss_weight_initial = 0.0
    config.mtp.mtp_loss_weight_final = 0.0
    print0("ABLATION: MTP disabled (lambda=0 throughout training)")

if args.no_shared_experts:
    config.moe.disable_shared_experts = True
    print0("ABLATION: shared experts disabled (output zeroed)")

if args.inject_bad_batch >= 0:
    print0(f"ABLATION: bad batch injection at step {args.inject_bad_batch} (10x gradient)")

# ─── Architecture overrides (must come before model build) ───
if args.num_experts > 0:
    old_e = config.moe.n_routed_experts
    config.moe.n_routed_experts = args.num_experts
    print0(f"OVERRIDE: n_routed_experts {old_e} → {args.num_experts}")

if args.top_k > 0:
    old_k = config.moe.num_experts_per_tok
    config.moe.num_experts_per_tok = args.top_k
    print0(f"OVERRIDE: num_experts_per_tok {old_k} → {args.top_k}")

if args.n_group > 0:
    old_g = config.moe.n_group
    config.moe.n_group = args.n_group
    print0(f"OVERRIDE: n_group {old_g} → {args.n_group}")

if args.topk_group > 0:
    old_tg = config.moe.topk_group
    config.moe.topk_group = args.topk_group
    print0(f"OVERRIDE: topk_group {old_tg} → {args.topk_group}")

if args.total_tokens > 0:
    old_t = config.total_tokens
    config.total_tokens = args.total_tokens
    print0(f"OVERRIDE: total_tokens {old_t:,} → {args.total_tokens:,}")

if args.hidden_size > 0:
    old_h = config.hidden_size
    config.hidden_size = args.hidden_size
    print0(f"OVERRIDE: hidden_size {old_h} → {args.hidden_size}")

# ─── YAML config override (loads all overrides from a single file) ───
if args.config_yaml:
    import yaml
    with open(args.config_yaml, 'r') as f:
        yaml_cfg = yaml.safe_load(f)
    print0(f"Loading config overrides from {args.config_yaml}")
    # Apply top-level overrides
    for key in ['total_tokens', 'hidden_size', 'num_layers', 'sequence_length',
                'vocab_size', 'max_grad_norm']:
        if key in yaml_cfg:
            old_val = getattr(config, key)
            setattr(config, key, yaml_cfg[key])
            print0(f"  YAML: {key} {old_val} → {yaml_cfg[key]}")
    # Apply MoE overrides
    if 'moe' in yaml_cfg:
        for key, val in yaml_cfg['moe'].items():
            if hasattr(config.moe, key):
                old_val = getattr(config.moe, key)
                setattr(config.moe, key, val)
                print0(f"  YAML: moe.{key} {old_val} → {val}")
    # Apply MTP overrides
    if 'mtp' in yaml_cfg:
        for key, val in yaml_cfg['mtp'].items():
            if hasattr(config.mtp, key):
                old_val = getattr(config.mtp, key)
                setattr(config.mtp, key, val)
                print0(f"  YAML: mtp.{key} {old_val} → {val}")
    # Apply MLA overrides
    if 'mla' in yaml_cfg:
        for key, val in yaml_cfg['mla'].items():
            if hasattr(config.mla, key):
                old_val = getattr(config.mla, key)
                setattr(config.mla, key, val)
                print0(f"  YAML: mla.{key} {old_val} → {val}")
    # Apply training overrides (LR, etc.)
    if 'training' in yaml_cfg:
        for key, val in yaml_cfg['training'].items():
            if key in vars(args):
                old_val = getattr(args, key.replace('-', '_'))
                setattr(args, key.replace('-', '_'), val)
                print0(f"  YAML: args.{key} {old_val} → {val}")

validate_config(config)

# ─── Log effective config to W&B (post-ablation overrides) ───
# This captures the ACTUAL values used, not just CLI args.
# Critical for reproducing: "what was seq_aux_loss_alpha for run stab-C?"
if not use_dummy_wandb:
    wandb_run.config.update({
        "effective/seq_aux_loss_alpha": config.moe.seq_aux_loss_alpha,
        "effective/max_grad_norm": config.max_grad_norm,
        "effective/mtp_loss_weight_initial": config.mtp.mtp_loss_weight_initial,
        "effective/mtp_loss_weight_final": config.mtp.mtp_loss_weight_final,
        "effective/disable_shared_experts": config.moe.disable_shared_experts,
        "effective/gamma_freeze_ratio": config.moe.gamma_freeze_ratio,
        "effective/aux_loss_type": args.aux_loss_type,
        "effective/inject_bad_batch": args.inject_bad_batch,
        "effective/n_routed_experts": config.moe.n_routed_experts,
        "effective/num_experts_per_tok": config.moe.num_experts_per_tok,
        "effective/n_group": config.moe.n_group,
        "effective/topk_group": config.moe.topk_group,
        "effective/total_tokens": config.total_tokens,
        "effective/hidden_size": config.hidden_size,
    }, allow_val_change=True)

# ─── Build model ───
print0("Building model on meta device...")
with torch.device("meta"):
    model = NanoSeekModel(config)

# Move to device and initialize
model.to_empty(device=device)
model.init_weights()

# ─── Parameter counts ───
param_counts = model.num_parameters()
n_active = param_counts['active']
n_total = param_counts['total']
print0(f"Parameters: {n_active:,} active / {n_total:,} total "
    f"(expansion: {n_total/n_active:.1f}×)")

# Critical for muP: all subsequent scaling uses config.hidden_size,
# NOT parameter counts. The parameter count tells us Chinchilla-optimal tokens.
# The hidden_size tells us muP LR scaling. These are independent axes

# ─── FLOPs per token (for MFU calculation) ───
# Law 3 from RESEARCH_ENGINEER.md: FLOPs = 6 × N_active × D
# Per token: flops_per_token = 6 × N_active
# The "6" comes from: 2 (multiply-add) × 3 (forward + backward = 3×forward)
#   Forward pass:  each weight does 1 multiply + 1 add = 2 FLOPs per param per token
#   Backward pass: 2× forward (grad w.r.t. activations + grad w.r.t. weights)
#   Total:         2 + 4 = 6 FLOPs per param per token
#
num_flops_per_token = 6 * n_active
print0(f"FLOPs per token: {num_flops_per_token:.2e}")
print0(f"Total training FLOPs: {num_flops_per_token * config.total_tokens:.2e}")

# ═══════════════════════════════════════════════════════════════════
# muP Hyperparameter Transfer (Tensor Programs V + Complete(d)P)
# ═══════════════════════════════════════════════════════════════════
#
# We have two scaling factors that compose multiplicatively:
#
# Factor 1: √(B/B_ref)  — from Complete(d)P (batch size scaling)
#   Larger batch → cleaner gradient → can take bigger step
#   η ∝ √B because gradient noise ∝ 1/√B
#
# Factor 2: w_ref/w     — from Tensor Programs V (width scaling)
#   Wider network → each weight contributes less to activation update
#   η ∝ 1/width to keep ||Δh|| = Θ(1) across widths
#
# Combined for hidden weights:  η = η_ref × √(B/B_ref) × (w_ref/w)
# For input/output weights:     η = η_ref × √(B/B_ref)
# For scale-independent params: η = η_ref (constant)
# ═══════════════════════════════════════════════════════════════════

w = config.hidden_size          # current model width
w_ref = args.mup_ref_width      # anchor width (480)

# Batch size in tokens
total_batch_tokens = config.global_batch_size * config.sequence_length
B_ref = args.mup_ref_batch_tokens  # anchor batch: 64 × 4096 = 262144

# Factor 1: √(B/B_ref) — Complete(d)P batch scaling
batch_lr_scale = math.sqrt(total_batch_tokens / B_ref)

# Factor 2: w_ref/w — Tensor Programs V width scaling
width_lr_scale = w_ref / w

print0(f"muP scaling factors:")
print0(f"  w_ref={w_ref}, w={w}")
print0(f"  B_ref={B_ref:,}, B={total_batch_tokens:,}")
print0(f"  √(B/B_ref) = {batch_lr_scale:.4f}")
print0(f"  w_ref/w    = {width_lr_scale:.4f}")
print0(f"  combined (hidden weights) = {batch_lr_scale * width_lr_scale:.4f}")

# ═══════════════════════════════════════════════════════════════════
# Weight Decay Scaling — T_epoch framework (arXiv:2405.13698)
# ═══════════════════════════════════════════════════════════════════
#
# Central idea: T_epoch = B / (η × λ × D) should be constant.
#
# We already scaled η by √(B/B_ref). To keep T_epoch constant:
#   λ = λ_ref × √(B/B_ref) × (D_ref / D)
#
# Where D is total training tokens.
#
# Derivation:
#   T_epoch = B / (η × λ × D)
#   At reference: T_ref = B_ref / (η_ref × λ_ref × D_ref)
#   At target:    T_tgt = B / (η_ref × √(B/B_ref) × λ × D)
#
#   Set T_tgt = T_ref:
#   B / (η_ref × √(B/B_ref) × λ × D) = B_ref / (η_ref × λ_ref × D_ref)
#
#   Solve for λ:
#   λ = λ_ref × (B × D_ref) / (B_ref × D) × (1/√(B/B_ref))
#     = λ_ref × √(B/B_ref) × (D_ref/D)
# ═══════════════════════════════════════════════════════════════════

D_ref = get_nanoseek_anchor_config().total_tokens  # anchor training tokens
D = config.total_tokens                            # current training tokens

weight_decay_scaled = args.weight_decay * batch_lr_scale * (D_ref / D)
print0(f"Weight decay: {args.weight_decay} → {weight_decay_scaled:.6f} "
        f"(T_epoch scaling: √B={batch_lr_scale:.4f} × D_ref/D={D_ref/D:.4f})")


# ═══════════════════════════════════════════════════════════════════
# Optimizer Construction — MoE-aware parameter groups
# ═══════════════════════════════════════════════════════════════════
#
# NanoSeekModel has NO setup_optimizer() method.
# We build param groups here because MoE requires careful classification:
#
#   Muon groups:  2D weight matrices EXCEPT embed, lm_head, gate.weight
#                 Grouped by shape for efficient stacking in optimizer
#                 LR: base × √(B/B_ref) × (w_ref/w)
#
#   AdamW groups:
#     embed:      LR: base × √(B/B_ref)         (input weight — no 1/w)
#     lm_head:    LR: base × √(B/B_ref)         (output weight — no 1/w)
#     router:     LR: constant                   (μP-MoE output weight)
#     norms:      LR: constant, no WD            (1D parameters)
# ═══════════════════════════════════════════════════════════════════

def setup_optimizer(model, config, args, batch_lr_scale, width_lr_scale,
                    weight_decay_scaled, ddp):
    """Build MoE-aware MuonAdamW optimizer with muP scaling.

    Parameter classification visual:
    NanoSeekModel
    ├── embed_tokens.weight          → AdamW (embedding)
    ├── layers[0..15]
    │   ├── self_attn
    │   │   ├── wq_a.weight          → Muon (2D hidden weight)
    │   │   ├── wq_b.weight          → Muon
    │   │   ├── wkv_a.weight         → Muon
    │   │   ├── wkv_b.weight         → Muon
    │   │   └── wo.weight            → Muon
    │   ├── input_layernorm.weight   → AdamW (norm, 1D)
    │   ├── post_attention_layernorm → AdamW (norm, 1D)
    |   ├── ffn (Dense, layers 0-1)
    |   │   ├── gate_proj.weight     → Muon (2D hidden weight)
    |   │   ├── up_proj.weight       → Muon
    |   │   └── down_proj.weight     → Muon
    |   ├── ffn (MoE, layers 2-15)
    |   │   ├── gate.weight          → AdamW (ROUTER — constant LR)
    |   │   ├── shared_experts
    |   │   │   ├── gate_proj.weight → Muon (2D hidden weight)
    |   │   │   ├── up_proj.weight   → Muon
    |   │   │   └── down_proj.weight → Muon
    |   │   └── experts[0..63]
    |   │       ├── gate_proj.weight → Muon (2D hidden weight)
    |   │       ├── up_proj.weight   → Muon
    |   │       └── down_proj.weight → Muon
    |   └── mtp (if exists)
    |       ├── proj.weight              → Muon (2D hidden weight)
    |       └── ...                      → Muon
    """
    # Classify every parameter
    embedding_params = []
    lm_head_params = []
    router_params = []
    norm_params = []
    muon_shapes = {}  # shape → [params] for Muon stacking

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'embed_tokens' in name:
            embedding_params.append(param)
        elif 'lm_head' in name:
            lm_head_params.append(param)
        elif 'gate.router_weight' in name:
            # Match MoE router (gate.router_weight.weight) but NOT SwiGLU gate_proj.weight
            router_params.append(param)
        elif param.ndim == 1:
            norm_params.append(param)
        elif param.ndim == 2:
            muon_shapes.setdefault(param.shape, []).append(param)
        else:
            assert False, f"Unknown parameter: {name} with shape {param.shape}"

    # ─── Compute scaled LRs ───
    # Hidden weight LR: base × √(B/B_ref) × (w_ref/w)
    hidden_lr_scale = batch_lr_scale * width_lr_scale

    # Input/output weight LR: base × √(B/B_ref) only
    boundary_lr_scale = batch_lr_scale

    # Build param_groups list
    param_groups = []

    # AdamW groups
    if embedding_params:
        param_groups.append(dict(
            kind='adamw',
            params=embedding_params,
            lr=args.embedding_lr * boundary_lr_scale,
            betas=(0.9, 0.95),
            eps=1e-10,
            weight_decay=0.001,
        ))

    if lm_head_params:
        param_groups.append(dict(
            kind='adamw',
            params=lm_head_params,
            lr=args.unembedding_lr * boundary_lr_scale,
            betas=(0.9, 0.95),
            eps=1e-10,
            weight_decay=0.01,
        ))

    if router_params:
        param_groups.append(dict(
            kind='adamw',
            params=router_params,
            lr=args.router_lr,  # CONSTANT — no scaling
            betas=(0.9, 0.95),
            eps=1e-10,
            weight_decay=0.0,   # no WD for router
        ))

    if norm_params:
        param_groups.append(dict(
            kind='adamw',
            params=norm_params,
            lr=args.norm_lr,    # CONSTANT — no scaling
            betas=(0.9, 0.95),
            eps=1e-10,
            weight_decay=0.0,   # no WD for norms
        ))

    # Muon groups (one per unique shape, for stacking)
    for shape, params in muon_shapes.items():
        param_groups.append(dict(
            kind='muon',
            params=params,
            lr=args.matrix_lr * hidden_lr_scale,
            momentum=0.95,
            ns_steps=5,
            beta2=0.9,
            weight_decay=weight_decay_scaled,
        ))

    # ─── Sanity check: every parameter accounted for ───
    n_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_grouped = sum(p.numel() for g in param_groups for p in g['params'])
    assert n_total_params == n_grouped, \
        f"Parameter mismatch: model has {n_total_params} but optimizer has {n_grouped}"
    print0(f"Optimizer sanity check passed: all {n_total_params:,} parameters accounted for")

    # Create the optimizer
    Factory = DistMuonAdamW if ddp else MuonAdamW
    optimizer = Factory(param_groups)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]

    # ─── Log optimizer ───
    n_muon = sum(len(g['params']) for g in param_groups if g['kind'] == 'muon')
    n_adamw = sum(len(g['params']) for g in param_groups if g['kind'] == 'adamw')
    print0(f"Optimizer: {Factory.__name__}")
    print0(f"  Muon params:  {n_muon} (in {sum(1 for g in param_groups if g['kind']=='muon')} shape groups)")
    print0(f"  AdamW params: {n_adamw} (embed={len(embedding_params)}, "
            f"lm_head={len(lm_head_params)}, router={len(router_params)}, "
            f"norm={len(norm_params)})")
    print0(f"  Hidden weight LR:  {args.matrix_lr * hidden_lr_scale:.6f} "
            f"(base={args.matrix_lr} × {hidden_lr_scale:.4f})")
    print0(f"  Embedding LR:      {args.embedding_lr * boundary_lr_scale:.6f}")
    print0(f"  LM head LR:        {args.unembedding_lr * boundary_lr_scale:.6f}")
    print0(f"  Router LR:         {args.router_lr} (constant)")
    print0(f"  Weight decay:      {weight_decay_scaled:.6f}")
    return optimizer


# ═══════════════════════════════════════════════════════════════════
# LR Schedule — warmup → constant → cosine decay
# ═══════════════════════════════════════════════════════════════════

def get_lr_multiplier(step, warmup_steps, constant_steps, decay_steps, lr_min_ratio):
    """Returns multiplier in [lr_min_ratio, 1.0] for the current step.
    Args:
        step: current training step
        warmup_steps: number of linear warmup steps
        constant_steps: number of constant LR steps
        decay_steps: number of cosine decay steps
        lr_min_ratio: minimum LR as fraction of peak (e.g., 0.1 for lr_min/lr)
    """
    if step < warmup_steps:
        # Linear warmup: 0 -> 1.0
        return (step + 1) / warmup_steps
    elif step <= warmup_steps + constant_steps:
        # Constant LR
        return 1.0
    else:
        # Cosine decay: 1.0 → lr_min_ratio
        progress = (step - warmup_steps - constant_steps) / max(decay_steps, 1)
        progress = min(progress, 1.0)  # clamp to [0, 1]
        return lr_min_ratio + 0.5 * (1.0 - lr_min_ratio) * (1.0 + math.cos(math.pi * progress))

# ─── Muon momentum warmup (from nanochat, works well) ───
def get_muon_momentum(step):
    """Warms momentum from 0.85 → 0.97 over first 400 steps.

    Why warm up momentum? At step 0, momentum buffer is empty.
    High momentum (0.97) with empty buffer = noisy.
    Start low (0.85), increase as buffer fills with signal.
    """
    frac = min(step / 400, 1.0)
    return (1 - frac) * 0.85 + frac * 0.97

# ═══════════════════════════════════════════════════════════════════
# Batch Size Warmup — ramp from 1/5 → 1× of target
# ═══════════════════════════════════════════════════════════════════
#
# DeepSeek V3: ramps over first 3% of training (aggressive)
# NanoSeek:    ramps over first 10% of training (conservative)
#
# Mechanism: we DON'T change the data loader batch size.
# Instead, we reduce grad_accum_steps and scale loss accordingly.
#
# Example for 1B:
#   target_batch = 128 × 4096 = 524,288 tokens
#   device_batch = 16 × 4096  = 65,536 tokens per GPU
#   world_batch  = 65,536 × 8 = 524,288 tokens per fwdbwd
#   target_accum = 524,288 / 524,288 = 1  (no accum needed for 8 GPUs)
#
# For anchor (4 GPUs):
#   target_batch = 64 × 4096  = 262,144 tokens
#   device_batch = 16 × 4096  = 65,536 tokens per GPU
#   world_batch  = 65,536 × 4 = 262,144 tokens per fwdbwd
#   target_accum = 262,144 / 262,144 = 1  (no accum needed)
#
# But if device_batch_size is smaller (e.g., 8 for OOM):
#   world_batch  = 8 × 4096 × 8 = 262,144
#   target_accum = 524,288 / 262,144 = 2  (need 2 accumulation steps)
#   warmup_accum = ceil(2 / 5) = 1        (start with 1 accum step)
# ═══════════════════════════════════════════════════════════════════

def get_batch_warmup_accum(step, target_accum, total_steps, warmup_fraction=0.10):
    """Get current grad accumulation steps during batch warmup.
    Ramps from ceil(target_accum/5) to target_accum over first
    warmup_fraction of training.
    Returns:
        current_accum: int, number of gradient accumulation steps this step
    """
    warmup_end = int(total_steps * warmup_fraction)

    if step >= warmup_end or target_accum <= 1:
        return target_accum

    min_accum = max(1, math.ceil(target_accum / 5))

    # Linear ramp from min_accum to target_accum
    progress = step / max(warmup_end, 1)
    current = min_accum + (target_accum - min_accum) * progress
    return max(min_accum, round(current))

# ═══════════════════════════════════════════════════════════════════
# EMA Weight Tracker — CPU-side Polyak averaging
# ═══════════════════════════════════════════════════════════════════
#
# RULE 1: ALL evaluation uses EMA weights, never raw weights.
# RULE 3: Scaling law fit uses ema_val_bpb, not val_bpb.
#
# EMA formula: θ_ema = α × θ_ema + (1 - α) × θ_model
#   α = 0.9999 (decay rate)
#   Update every 10 steps (not every step — too expensive for 4.75B params)
#
# Why CPU-side?
#   EMA weights are 4.75B × 2 bytes = 9.5 GB in bf16.
#   If we keep them on GPU: doubles our parameter memory from 9.5→19 GB.
#   On CPU: free (RAM is cheap). Trade-off: slower eval (need to copy to GPU).
#   For eval every 250 steps, the copy cost is negligible.
#
# Why α=0.9999?
#   The "window" of EMA ≈ 1/(1-α) = 10,000 updates.
#   With 1 update per 10 steps (ema_every=10): effective window ≈ 100,000 steps.
#   At anchor scale (~4,200 steps), this means EMA is very slow — initial weights
#   still dominate. At 1B scale (~5,400 steps), same issue. This is intentional:
#   gentle averaging avoids tracking SGD noise while capturing the trajectory.
# ═══════════════════════════════════════════════════════════════════
class EMATracker:
    """CPU-side EMA weight tracker for all evaluation.

    Uses Karras-style decay warmup: effective_decay = min(decay, 1 - 1/(1+count)).
    This makes early updates more impactful so EMA tracks the model during short
    runs (smoke tests), while converging to the target decay for long runs.
    Without this, decay=0.9999 means shadow weights barely move for the first
    ~10,000 updates (99.9% initial weights after 10 updates).
    """
    def __init__(self, model, decay=0.9999, device="cpu"):
        self.decay = decay
        self.device = device
        self.update_count = 0
        # Deep copy all parameters to CPU
        self.shadow = {
            name: param.detach().clone().to(device)
            for name, param in model.named_parameters()
        }

    @torch.no_grad()
    def step(self, model):
        """Update EMA weights from model with Karras decay warmup."""
        self.update_count += 1
        # Karras warmup: ramp decay from 0 → target over first 1/(1-decay) updates
        effective_decay = min(self.decay, 1.0 - 1.0 / (1.0 + self.update_count))
        for name, param in model.named_parameters():
            # lerp_: shadow = decay * shadow + (1 - decay) * param
            self.shadow[name].lerp_(param.detach().to(self.device), 1 - effective_decay)

    @contextmanager
    def apply(self, model):
        """Context manager to temporarily swap model weights with EMA weights."""
        original_weights = {}
        for name, param in model.named_parameters():
            original_weights[name] = param.detach().clone()
            param.data.copy_(self.shadow[name].to(param.device))
        try:
            yield
        finally:
            for name, param in model.named_parameters():
                param.data.copy_(original_weights[name])

    def state_dict(self):
        """for checkpointing"""
        d = {k: v.clone() for k, v in self.shadow.items()}
        d['__ema_update_count__'] = self.update_count
        return d

    def load_state_dict(self, state_dict):
        """for loading checkpoints"""
        for k, v in state_dict.items():
            if k == '__ema_update_count__':
                self.update_count = v
            else:
                self.shadow[k].copy_(v)

    def __repr__(self):
        return f"EMATracker(decay={self.decay}, device={self.device})"


# ═══════════════════════════════════════════════════════════════════
# NanoSeek-specific BPB evaluation
# ═══════════════════════════════════════════════════════════════════
#
# nanochat's evaluate_bpb() calls model(x, y, loss_reduction='none')
# which is the GPT interface: positional args, returns a loss tensor.
#
# NanoSeek's forward is: model(input_ids, labels=...) -> Dict.
# These are INCOMPATIBLE interfaces.
#
# Additionally, nanochat's dataloader returns PRE-SHIFTED targets:
#   x = row_buffer[:, :-1]  (inputs)
#   y = row_buffer[:, 1:]   (shifted targets)
#
# NanoSeek's _compute_loss does its OWN shifting (HuggingFace convention).
# So we must NOT pass y as labels (double-shift bug).
#
# For bpb evaluation, we forward without labels to get logits,
# then compute per-token cross-entropy against the dataloader's
# pre-shifted targets (y) directly — bypassing the model's shift.
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_nanoseek_bpb(model, batches, steps, token_bytes):
    """Compute bits-per-byte for NanoSeek model.

    Uses logits from model forward (no labels) and computes
    per-token CE against the dataloader's pre-shifted targets.
    This avoids the double-shift bug that would occur if we
    passed the pre-shifted y as labels to _compute_loss.

    Args:
        model: NanoSeekModel (uncompiled, with EMA weights applied)
        batches: val_loader yielding (x, y) tuples
        steps: number of eval steps
        token_bytes: [vocab_size] tensor of byte lengths per token
    """
    device = next(model.parameters()).device
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        # Forward WITHOUT labels to get logits only (no loss computation)
        outputs = model(x)
        logits = outputs['logits']  # [B, T, V]
        # Compute per-token CE against pre-shifted targets directly
        # logits[i] predicts position i+1's token, y[i] IS position i+1's token
        loss_per_token = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=-1,
            reduction='none',
        )  # [B*T]
        y_flat = y.view(-1)
        if (y_flat.int() < 0).any():
            valid = y_flat >= 0
            y_safe = torch.where(valid, y_flat, torch.zeros_like(y_flat))
            num_bytes = torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y_flat, dtype=token_bytes.dtype)
            )
            total_nats += (loss_per_token * (num_bytes > 0)).sum()
            total_bytes += num_bytes.sum()
        else:
            num_bytes = token_bytes[y_flat]
            total_nats += (loss_per_token * (num_bytes > 0)).sum()
            total_bytes += num_bytes.sum()

    # Reduce across ranks for distributed evaluation
    if is_ddp_initialized():
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)

    # Convert nats to bits, normalize by bytes
    bpb = total_nats.item() / math.log(2) / max(total_bytes.item(), 1)
    return bpb


# ═══════════════════════════════════════════════════════════════════
# Training Loop Setup
# ═══════════════════════════════════════════════════════════════════

# ─── GC management (from nanochat — reduces GC pauses during training) ───
gc.collect()
gc.freeze()
gc.disable()

# ─── Tokenizer and data ───
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
print0(f"Vocab size: {tokenizer.get_vocab_size():,}")

# ─── Number of training iterations ───
total_batch_tokens = config.global_batch_size * config.sequence_length
if args.num_iterations > 0:
    num_iterations = args.num_iterations
else:
    num_iterations = config.total_tokens // total_batch_tokens
print0(f"Total iterations: {num_iterations:,}")
print0(f"Total tokens: {total_batch_tokens * num_iterations:,}")

# ─── LR schedule phase boundaries ───
# Config says: warmup → constant until 70% → cosine decay 70%→95% → lr_min
# constant_phase_ratio = 0.70 means "constant phase ENDS at the 70% mark"
# cosine_decay_end_ratio = 0.95 means "cosine ends at 95%, then lr_min"
warmup_steps = config.warmup_steps
constant_end_step = int(config.constant_phase_ratio * num_iterations)
decay_end_step = int(config.cosine_decay_end_ratio * num_iterations)
constant_steps = max(0, constant_end_step - warmup_steps)
decay_steps = max(0, decay_end_step - constant_end_step)
lr_min_ratio = config.lr_min / config.learning_rate  # 3e-5/3e-4 = 0.1

print0(f"LR schedule: warmup={warmup_steps} (0→{warmup_steps}), "
        f"constant={constant_steps} ({warmup_steps}→{constant_end_step}), "
        f"decay={decay_steps} ({constant_end_step}→{decay_end_step}), "
        f"lr_min ({decay_end_step}→{num_iterations}), ratio={lr_min_ratio:.2f}")

# ─── Gradient accumulation ───
tokens_per_microbatch = args.device_batch_size * config.sequence_length
world_tokens_per_fwdbwd = tokens_per_microbatch * ddp_world_size
assert total_batch_tokens % world_tokens_per_fwdbwd == 0, \
    f"total_batch_tokens ({total_batch_tokens}) must be divisible by " \
    f"world_tokens_per_fwdbwd ({world_tokens_per_fwdbwd})"
target_grad_accum = total_batch_tokens // world_tokens_per_fwdbwd

print0(f"Gradient accumulation: {target_grad_accum} steps "
        f"({args.device_batch_size}×{config.sequence_length}×{ddp_world_size} "
        f"= {world_tokens_per_fwdbwd:,} tokens/fwdbwd)")

optimizer = setup_optimizer(
    model, config, args, batch_lr_scale,
    width_lr_scale, weight_decay_scaled, ddp
)

orig_model = model

# ─── torch.compile with timing ───
# torch.compile is lazy — actual kernel compilation happens on first forward pass.
# Expected compile times by scale:
#   anchor (~55M):  1-5 min (first run), <30s (cached)
#   500M:           2-8 min (first run), <30s (cached)
#   1B:             3-10 min (first run), <30s (cached)
# If compile exceeds 15 min, something is wrong (PyTorch/CUDA version mismatch,
# incompatible GPU architecture, or infinite recompilation loop).
COMPILE_TIMEOUT_MINUTES = 15
_compile_start_time = time.time()
if args.no_compile:
    print0("torch.compile SKIPPED (--no-compile flag set)")
else:
    model = torch.compile(model, dynamic=False)
    print0(f"torch.compile registered (lazy — actual compilation on first forward pass)")
    print0(f"  Compile timeout: {COMPILE_TIMEOUT_MINUTES} min. If step 0 takes longer, check:")
    print0(f"  1. PyTorch version supports your GPU (run: python -c \"import torch; torch.zeros(1).cuda()\")")
    print0(f"  2. CUDA toolkit version matches PyTorch build")
    print0(f"  3. Try --no-compile flag or TORCH_COMPILE_DISABLE=1 to skip compilation")

# ─── EMA tracker ───
ema_tracker = EMATracker(orig_model, decay=args.ema_decay)
print0(f"EMA tracker initialized (decay={args.ema_decay}, update every {args.ema_every} steps)")

# ─── Checkpoint manager (atomic writes + disk cleanup) ───
# Include run name in checkpoint dir so ablation runs don't overwrite each other.
# e.g., checkpoints/nanoseek_anchor/stab-A-baseline/
checkpoint_dir = os.path.join("checkpoints", f"nanoseek_{args.scale}", args.run)
ckpt_manager = CheckpointManager(
    checkpoint_dir=checkpoint_dir,
    save_every=args.save_every,
    keep_last_n=3,           # ~228 GB max for 1B (3 × 76 GB) — fits 200 GB volume
    save_optimizer=True,     # MUST save optimizer for correct resume
)
print0(f"CheckpointManager: dir={checkpoint_dir}, save_every={args.save_every}, keep_last_n=3")

# ─── Graceful shutdown signal handler ───
# RunPod sends SIGTERM 30s before preemption. Without this handler,
# we lose everything since the last checkpoint (potentially hours of work).
import signal

_shutdown_requested = False

def _shutdown_handler(signum, frame):
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    print0(f"\n⚠️  Received {sig_name} — will save emergency checkpoint after current step")
    _shutdown_requested = True

signal.signal(signal.SIGTERM, _shutdown_handler)
signal.signal(signal.SIGINT, _shutdown_handler)

# ─── Resume from checkpoint ───
resume_step = 0
resume_dataloader_state = None
resume_loop_state = {}

if args.resume_from_step >= 0:
    resume_step_arg = args.resume_from_step if args.resume_from_step > 0 else None  # 0 = latest
    print0(f"Resuming from checkpoint (step={resume_step_arg or 'latest'})...")

    # Load model weights
    model_state, metadata, loaded_step = load_checkpoint(checkpoint_dir, step=resume_step_arg, device=device)
    orig_model.load_state_dict(model_state)
    print0(f"  Model state loaded from step {loaded_step}")

    # Load optimizer state
    opt_state = load_optimizer_checkpoint(checkpoint_dir, loaded_step, device=device)
    if opt_state is not None:
        optimizer.load_state_dict(opt_state)
        print0(f"  Optimizer state loaded")
    else:
        print0(f"  WARNING: No optimizer checkpoint found — optimizer reset to initial state")

    # Load EMA state
    try:
        ema_state = load_ema_checkpoint(checkpoint_dir, loaded_step, device=torch.device("cpu"))
        ema_tracker.load_state_dict(ema_state)
        print0(f"  EMA state loaded")
    except FileNotFoundError:
        print0(f"  WARNING: No EMA checkpoint found — EMA re-initialized from model weights")

    # Restore training loop state from metadata
    resume_step = loaded_step
    resume_dataloader_state = metadata.get("dataloader_state_dict", None)
    resume_loop_state = metadata.get("loop_state", {})
    print0(f"  Resuming from step {resume_step}, tokens={metadata.get('tokens_processed', 'unknown')}")

    # Re-compile after loading weights
    _compile_start_time = time.time()
    if not args.no_compile:
        model = torch.compile(orig_model, dynamic=False)

# ─── Data loaders ───
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, args.device_batch_size, config.sequence_length,
    split="train", device=device, resume_state_dict=resume_dataloader_state,
    fim_rate=0.10,  # RULE 6: 10% PSM FIM from token 1
)
# Eval uses smaller batch to avoid OOM from materialized [B,T,V] logits
eval_batch_size = min(args.device_batch_size, 2)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, eval_batch_size, config.sequence_length,
    split="val", device=device,
)

# ─── Kick off first batch ───
x, y, dataloader_state_dict = next(train_loader)
print0(f"First batch prefetched: {x.shape} inputs, {y.shape} targets")

# ═══════════════════════════════════════════════════════════════════
# Training Health Monitor — Automated Tripwires
# ═══════════════════════════════════════════════════════════════════
#
# What top-tier labs (DeepSeek, OLMo, Meta) learned:
#   1. Gradient norm is the #1 leading indicator (OLMo: blowup at GN>0.4)
#   2. Loss spike frequency matters more than individual spikes
#   3. Expert collapse is silent — monitor H_load continuously
#   4. ZClip (adaptive grad norm clipping) > fixed threshold
#
# These tripwires run every step with ~0 overhead (just comparisons).
# ═══════════════════════════════════════════════════════════════════

class TrainingHealthMonitor:
    """Automated early-warning system for training instability.

    Tracks 3 signals that catch 80% of problems (from frontier lab postmortems):
    1. Gradient norm: EMA + z-score spike detection (ZClip-inspired)
    2. Loss: rolling average + spike ratio detection
    3. H_load: expert collapse detection

    Reference: OLMo 2 (GN threshold), ZClip (z-score), DeepSeek V3 (zero rollbacks)
    """
    def __init__(self, warmup_steps=0):
        # Gradient norm tracking (ZClip-inspired adaptive detection)
        self.grad_norm_ema = 0.0
        self.grad_norm_var_ema = 0.0
        self.grad_norm_ema_beta = 0.99
        self.grad_norm_initialized = False

        # Loss tracking
        self.loss_ema = 0.0
        self.loss_ema_beta = 0.95
        self.loss_initialized = False

        # Spike counting (OLMo insight: frequency increase = real danger)
        self.grad_spikes_last_100 = []
        self.loss_spikes_last_100 = []

        # Consecutive NaN counter
        self.nan_count = 0

        # Grace period: during LR warmup, gradient norms naturally grow monotonically.
        # Spike detection during this period generates only false positives.
        self.warmup_steps = warmup_steps

    def update(self, step, grad_norm_val, loss_val, h_load_val):
        """Check all tripwires. Returns list of alerts (empty = healthy)."""
        alerts = []

        # ─── NaN/Inf detection (IMMEDIATE — always active, even during warmup) ───
        gn = grad_norm_val.item() if hasattr(grad_norm_val, 'item') else float(grad_norm_val)
        lv = loss_val.item() if hasattr(loss_val, 'item') else float(loss_val)
        hl = h_load_val.item() if hasattr(h_load_val, 'item') else float(h_load_val)

        if math.isnan(gn) or math.isinf(gn) or math.isnan(lv) or math.isinf(lv):
            self.nan_count += 1
            alerts.append(("CRITICAL", f"NaN/Inf detected at step {step} "
                          f"(grad_norm={gn}, loss={lv}). Count: {self.nan_count}"))
            return alerts
        self.nan_count = 0

        in_warmup = step < self.warmup_steps

        # ─── Gradient norm: EMA + z-score (ZClip-inspired) ───
        # Always update EMA (even during warmup) so it's calibrated when warmup ends
        if not self.grad_norm_initialized:
            self.grad_norm_ema = gn
            self.grad_norm_var_ema = 0.0
            self.grad_norm_initialized = True
        else:
            beta = self.grad_norm_ema_beta
            self.grad_norm_var_ema = beta * self.grad_norm_var_ema + (1 - beta) * (gn - self.grad_norm_ema) ** 2
            self.grad_norm_ema = beta * self.grad_norm_ema + (1 - beta) * gn

            # Skip spike detection during warmup (gradients grow monotonically with LR)
            if not in_warmup:
                # Z-score spike detection
                std = math.sqrt(self.grad_norm_var_ema + 1e-8)
                z_score = (gn - self.grad_norm_ema) / std
                if z_score > 4.0:
                    alerts.append(("WARNING", f"Gradient norm spike: {gn:.4f} "
                                  f"(z={z_score:.1f}, ema={self.grad_norm_ema:.4f})"))
                    self.grad_spikes_last_100.append(step)

        # ─── Loss spike detection ───
        if not self.loss_initialized:
            self.loss_ema = lv
            self.loss_initialized = True
        else:
            self.loss_ema = self.loss_ema_beta * self.loss_ema + (1 - self.loss_ema_beta) * lv

            if not in_warmup:
                ratio = lv / max(self.loss_ema, 1e-8)
                if ratio > 2.0:
                    alerts.append(("WARNING", f"Loss spike: {lv:.4f} "
                                  f"({ratio:.1f}x rolling avg {self.loss_ema:.4f})"))
                    self.loss_spikes_last_100.append(step)
                if ratio > 3.0:
                    alerts.append(("CRITICAL", f"Severe loss spike: {lv:.4f} "
                                  f"({ratio:.1f}x avg). Consider checkpoint restore."))

        # ─── Expert collapse detection (always active) ───
        if hl < 2.0:
            alerts.append(("CRITICAL", f"Expert collapse: H_load={hl:.2f} bits "
                          f"(threshold: 2.0). Routing is degenerate."))
        elif hl < 4.0 and step > self.warmup_steps:
            alerts.append(("WARNING", f"H_load declining: {hl:.2f} bits "
                          f"(healthy > 4.0 at init)"))

        # ─── Spike frequency (OLMo insight) — only after warmup ───
        if not in_warmup:
            self.grad_spikes_last_100 = [s for s in self.grad_spikes_last_100 if step - s < 100]
            self.loss_spikes_last_100 = [s for s in self.loss_spikes_last_100 if step - s < 100]

            if len(self.grad_spikes_last_100) >= 5:
                alerts.append(("WARNING", f"Spike frequency increasing: "
                              f"{len(self.grad_spikes_last_100)} grad spikes in last 100 steps"))
            if len(self.loss_spikes_last_100) >= 3:
                alerts.append(("WARNING", f"Loss spike frequency: "
                              f"{len(self.loss_spikes_last_100)} spikes in last 100 steps"))

        return alerts


health_monitor = TrainingHealthMonitor(warmup_steps=warmup_steps)

# ─── Training loop state ───
step = resume_step
tokens_processed = resume_loop_state.get("tokens_processed", 0) if resume_step > 0 else 0
smooth_train_loss = resume_loop_state.get("smooth_train_loss", 0.0)
ema_val_bpb = None  # Set by first eval; used in checkpoint metadata
min_ema_val_bpb = resume_loop_state.get("min_ema_val_bpb", None)
total_training_time = resume_loop_state.get("total_training_time", 0.0)
t0 = time.time()

while True:
    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end

    # ─────────────────────────────────────────────────────────────
    # EVALUATE (RULE 1: ALL eval uses EMA weights)
    # Ensure EMA is up-to-date before evaluation (even if not on ema_every boundary)
    # ─────────────────────────────────────────────────────────────
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        if step > 0:
            ema_tracker.step(orig_model)
        model.eval()
        eval_steps = args.eval_tokens // (eval_batch_size * config.sequence_length * ddp_world_size)
        # Fresh val_loader for BPB eval — each metric gets its own loader
        # to avoid consuming data that subsequent metrics need.
        val_loader_bpb = build_val_loader()
        # Swap EMA weights onto orig_model, evaluate, then restore
        with ema_tracker.apply(orig_model):
            ema_val_bpb = evaluate_nanoseek_bpb(orig_model, val_loader_bpb, eval_steps, token_bytes)

            # ─── Milestone eval: I_spec, domain BPB, dead experts (RULE 7) ───
            progress = step / max(num_iterations, 1)
            is_milestone = any(abs(progress - p) < (0.5 / num_iterations + 1e-9) for p in [0.20, 0.50, 0.80, 1.00])
            milestone_metrics = {}

            if is_milestone or last_step:
                print0(f"  Milestone eval at {progress:.1%}...")
                try:
                    val_loader_ispec = build_val_loader()
                    i_spec_result = compute_i_spec(orig_model, val_loader_ispec, device)
                    milestone_metrics["eval/i_spec_mean"] = i_spec_result['i_spec_mean']
                    for i, v in enumerate(i_spec_result.get('i_spec_per_layer', [])):
                        milestone_metrics[f"eval/i_spec_layer_{i}"] = v
                    print0(f"  I_spec: {i_spec_result['i_spec_mean']:.4f}")
                except Exception as e:
                    print0(f"  I_spec failed: {e}")

                try:
                    val_loader_dead = build_val_loader()
                    dead_result = compute_dead_experts(orig_model, val_loader_dead, device)
                    milestone_metrics["eval/dead_expert_count"] = dead_result['total_dead_count']
                    if dead_result['total_dead_count'] > 0:
                        print0(f"  WARNING: {dead_result['total_dead_count']} dead experts detected!")
                    else:
                        print0(f"  No dead experts")
                except Exception as e:
                    print0(f"  Dead expert check failed: {e}")

                try:
                    domain_bpb = compute_domain_bpb(orig_model, tokenizer, device)
                    for domain, bpb in domain_bpb.items():
                        milestone_metrics[f"eval/domain_bpb/{domain}"] = bpb
                    print0(f"  Domain BPB: { {k: f'{v:.3f}' for k, v in domain_bpb.items()} }")
                except Exception as e:
                    print0(f"  Domain BPB failed: {e}")

            # ─── MTP acceptance rate every 2000 steps (RULE 9) ───
            mtp_metrics = {}
            if step > 0 and step % 2000 == 0:
                try:
                    val_loader_mtp = build_val_loader()
                    mtp_result = compute_mtp_acceptance_rate(orig_model, val_loader_mtp, device)
                    mtp_metrics["eval/mtp_acceptance_rate"] = mtp_result['acceptance_rate']
                    for pos, rate in mtp_result.get('per_position_rates', {}).items():
                        mtp_metrics[f"eval/mtp_pos_{pos}"] = rate
                    print0(f"  MTP acceptance: {mtp_result['acceptance_rate']:.1%}")
                except Exception as e:
                    print0(f"  MTP acceptance failed: {e}")

        # model.train() OUTSIDE the EMA context manager
        model.train()

        print0(f"Step {step:05d} | EMA Validation bpb: {ema_val_bpb:.4f}")
        if min_ema_val_bpb is None or ema_val_bpb < min_ema_val_bpb:
            min_ema_val_bpb = ema_val_bpb

        flops_so_far = num_flops_per_token * tokens_processed
        # Log expert load histogram at eval frequency (not every step — 64 values)
        load_stats = orig_model.get_expert_load_stats()
        load_per_expert = load_stats['load_per_expert']  # [E]
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "ema_val/bpb": ema_val_bpb,
            "moe/load_per_expert": wandb.Histogram(load_per_expert.cpu().numpy()),
            **milestone_metrics,
            **mtp_metrics,
        })

    # ─────────────────────────────────────────────────────────────
    # CHECKPOINT SAVING (atomic writes + disk cleanup + EMA bundled)
    # ─────────────────────────────────────────────────────────────
    should_save = ckpt_manager.should_save(step, last_step) or _shutdown_requested
    if should_save:
        # DDP barrier: ensure all ranks are done with training step
        # before rank 0 starts I/O (prevents gradient sync stalls)
        if ddp:
            dist.barrier()

        ckpt_metadata = {
            "step": step,
            "tokens_processed": tokens_processed,
            "ema_val_bpb": ema_val_bpb,
            "config": asdict(config),
            "dataloader_state_dict": dataloader_state_dict,
            "loop_state": {
                "tokens_processed": tokens_processed,
                "min_ema_val_bpb": min_ema_val_bpb,
                "smooth_train_loss": smooth_train_loss,
                "total_training_time": total_training_time,
            },
        }

        # Save model + optimizer + EMA atomically (all or nothing per file)
        # EMA is bundled into save_checkpoint so it can't be missing after crash
        saved_path = ckpt_manager.save(
            step=step,
            model_state=orig_model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            metadata=ckpt_metadata,
            ema_state=ema_tracker.state_dict(),
            rank=ddp_rank,
        )
        if saved_path:
            print0(f"Checkpoint saved: {saved_path} (step {step})")

        # If this was a shutdown request, exit cleanly after saving
        if _shutdown_requested:
            print0(f"Emergency checkpoint saved at step {step}. Exiting gracefully.")
            wandb_run.finish()
            gc.enable()
            gc.collect()
            compute_cleanup()
            import sys
            sys.exit(0)

    # ─── Termination ───
    if last_step:
        break

    # ─────────────────────────────────────────────────────────────
    # SINGLE TRAINING STEP
    # ─────────────────────────────────────────────────────────────
    current_accum = get_batch_warmup_accum(
        step, target_grad_accum, num_iterations
    )
    current_batch_tokens = world_tokens_per_fwdbwd * current_accum

    synchronize()
    t0 = time.time()

    # ─── Micro-step accumulation loop ───
    # Each micro-step does forward+backward, ACCUMULATING gradients.
    # Optimizer step happens ONCE after all micro-steps complete.
    #
    # IMPORTANT: We pass labels=x (NOT labels=y) because:
    #   - The dataloader returns y = pre-shifted targets (row[:, 1:])
    #   - NanoSeek's _compute_loss does its OWN shift (HuggingFace convention):
    #       shift_logits = logits[:, :-1]
    #       shift_labels = labels[:, 1:]
    #   - If we passed labels=y, the model would shift AGAIN → double-shift,
    #     training on wrong targets (off by one position)
    #   - With labels=x, the model's internal shift produces:
    #       shift_labels = x[:, 1:] = [t1, t2, ..., t_{T-1}]
    #     which correctly matches logits at positions 0 to T-2.
    #   - We lose 1 prediction per sequence (last token) — 0.024% at T=4096.
    train_loss_accum = 0.0
    main_loss_accum = 0.0
    mtp_loss_accum = 0.0
    aux_loss_accum = 0.0
    for micro_step in range(current_accum):
        # autocast: run forward pass in bf16 for 2x memory savings + 2x faster matmuls
        # Without this, model runs in fp32 and CastLinear does nothing useful.
        # Loss computation stays in fp32 (cross_entropy promotes automatically).
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=(device_type == "cuda")):
            from nanoseek.nanoseek.model import get_mtp_loss_weight
            mtp_lambda = get_mtp_loss_weight(
                tokens_processed, config.total_tokens,
                config.mtp.mtp_loss_weight_initial,
                config.mtp.mtp_loss_weight_final,
                config.mtp.mtp_loss_transition_ratio,
            )
            outputs = model(
                x,
                labels=x,
                mtp_lambda=mtp_lambda,
            )
            loss = outputs['loss']
        train_loss_accum += loss.detach()
        main_loss_accum += outputs['main_loss'].detach()
        mtp_loss_accum += outputs['mtp_loss'].detach()
        aux_loss_accum += outputs['aux_loss'].detach()

        # Scale loss for gradient accumulation
        # Each .backward() ADDS to .grad -> divide by accum count
        loss = loss / current_accum
        loss.backward()

        # Prefetch next batch while the GPU is busy with backward
        x, y, dataloader_state_dict = next(train_loader)

    # Average losses across all micro-steps for accurate logging
    train_loss = train_loss_accum / current_accum
    main_loss_avg = main_loss_accum / current_accum
    mtp_loss_avg = mtp_loss_accum / current_accum
    aux_loss_avg = aux_loss_accum / current_accum

    # ═══════════════════════════════════════════════════════════
    # EVERYTHING BELOW IS OUTSIDE THE MICRO-STEP LOOP
    # One optimizer step per training step, not per micro-step
    # ═══════════════════════════════════════════════════════════

    # ─── Bad batch injection (ablation stab-F) ───
    # Simulate a gradient spike by scaling all gradients 10×
    if args.inject_bad_batch >= 0 and step == args.inject_bad_batch:
        print0(f"ABLATION: Injecting 10x gradient spike at step {step}")
        for p in orig_model.parameters():
            if p.grad is not None:
                p.grad.mul_(10.0)

    # ─── Gradient clipping (Difference #3) ───
    # MoE gradient variance is 8× higher than dense (κ=12.5%)
    # Without clipping: P(NaN within 1000 steps) ≈ 1 for bf16
    clip_norm = float('inf') if args.no_grad_clip else config.max_grad_norm
    grad_norm = clip_grad_norm_(orig_model.parameters(), max_norm=clip_norm)

    # Update learning rate and momentum
    lrm = get_lr_multiplier(step, warmup_steps, constant_steps, decay_steps, lr_min_ratio)
    muon_momentum = get_muon_momentum(step)

    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum

    # Step the optimizer
    optimizer.step()
    model.zero_grad(set_to_none=True)

    # ─── MoE load-balance bias update (Difference #6) ───
    # This is the aux-loss-free balancing mechanism from DeepSeek V3.
    # After each gradient step, adjust expert biases to redistribute load.
    # Formula: b_i -= gamma × (load_i - mean) / mean
    # Gamma freezes at 0 after 95% of training (RULE 2).
    # Skip bias updates when using classic aux loss (no bias-based balancing)
    if args.aux_loss_type != "classic":
        orig_model.update_load_balance_bias(tokens_processed, config.total_tokens)

    # --- EMA update (Difference #4) ---
    if step % args.ema_every == 0:
        ema_tracker.step(orig_model)

    tokens_processed += current_batch_tokens

    # ─── Timing ───
    train_loss_f = train_loss.item()
    synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # ─── MoE health metrics ───
    load_stats = orig_model.get_expert_load_stats()
    H_load = load_stats['entropy'].item()
    gamma = orig_model.get_gamma(tokens_processed, config.total_tokens)

    # ─── Health monitor (automated tripwires) ───
    alerts = health_monitor.update(step, grad_norm, train_loss_f, H_load)
    for severity, msg in alerts:
        print0(f"  [{severity}] {msg}")
    if alerts and not use_dummy_wandb:
        n_critical = sum(1 for s, _ in alerts if s == "CRITICAL")
        n_warning = sum(1 for s, _ in alerts if s == "WARNING")
        wandb_run.log({
            "health/critical_alerts": n_critical,
            "health/warning_alerts": n_warning,
            "step": step,
        })

    # ─── Logging ───
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))

    tok_per_sec = int(current_batch_tokens / dt)
    flops_per_sec = num_flops_per_token * current_batch_tokens / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)

    pct_done = 100 * step / num_iterations
    print0(
        f"step {step:05d}/{num_iterations} ({pct_done:.1f}%) | "
        f"loss: {debiased_loss:.4f} (main:{main_loss_avg.item():.4f} mtp:{mtp_loss_avg.item():.4f} aux:{aux_loss_avg.item():.6f}) | "
        f"H_load: {H_load:.2f} | "
        f"lr×: {lrm:.3f} | γ: {gamma:.4f} | "
        f"grad: {grad_norm:.2f} | "
        f"dt: {dt:.2f}s | "
        f"mfu: {mfu:.1f}% | "
        f"tok/s: {tok_per_sec:,}"
    )

    # ─── Compile time check (step 0 includes torch.compile kernel generation) ───
    if step == 0 and '_compile_start_time' in dir():
        compile_elapsed = (time.time() - _compile_start_time) / 60
        if compile_elapsed > COMPILE_TIMEOUT_MINUTES:
            print0(f"  [CRITICAL] torch.compile took {compile_elapsed:.1f} min "
                   f"(limit: {COMPILE_TIMEOUT_MINUTES} min). This usually means:")
            print0(f"    - PyTorch doesn't support your GPU architecture")
            print0(f"    - CUDA version mismatch")
            print0(f"    - Try: TORCH_COMPILE_DISABLE=1 or upgrade PyTorch")
        elif compile_elapsed > 5:
            print0(f"  [INFO] torch.compile took {compile_elapsed:.1f} min "
                   f"(normal: 1-5 min first run, <30s cached)")
        else:
            print0(f"  [INFO] torch.compile: {compile_elapsed:.1f} min (OK)")

    if step % config.log_every_steps == 0:
        # ─── Collect per-group LRs (verify muP scaling is correct) ───
        group_lrs = {}
        for group in optimizer.param_groups:
            kind = group['kind']
            # Only log first group of each kind (they share the same LR)
            key = f"lr/{kind}"
            if key not in group_lrs:
                group_lrs[key] = group['lr']

        # ─── GPU memory (detect OOM risk before it crashes) ───
        mem_stats = {}
        if device_type == "cuda":
            mem_stats["gpu/memory_allocated_gb"] = torch.cuda.memory_allocated(device) / 1e9
            mem_stats["gpu/memory_reserved_gb"] = torch.cuda.memory_reserved(device) / 1e9

        wandb_run.log({
            "step": step,
            "tokens_processed": tokens_processed,
            # ─── Loss breakdown (detect MTP vs LM issues separately) ───
            "train/loss": debiased_loss,
            "train/loss_raw": train_loss_f,
            "train/main_loss": main_loss_avg.item(),
            "train/mtp_loss": mtp_loss_avg.item(),
            "train/mtp_lambda": outputs['mtp_lambda'],
            "train/aux_loss": aux_loss_avg.item(),
            # ─── LR & optimization ───
            "train/lr_multiplier": lrm,
            "train/grad_norm": grad_norm.item(),
            "train/step_time_s": dt,
            # ─── MoE health ───
            "train/H_load": H_load,
            "train/gamma": gamma,
            # ─── Throughput ───
            "train/mfu": mfu,
            "train/tok_per_sec": tok_per_sec,
            "train/batch_tokens": current_batch_tokens,
            # ─── FIM (RULE 6) ───
            "train/fim_fraction": dataloader_state_dict.get("fim_fraction", 0.0),
            # ─── Health monitor (tripwire internals) ───
            "health/grad_norm_ema": health_monitor.grad_norm_ema,
            "health/loss_ema": health_monitor.loss_ema,
            "health/grad_spikes_last_100": len(health_monitor.grad_spikes_last_100),
            "health/loss_spikes_last_100": len(health_monitor.loss_spikes_last_100),
            # ─── Per-group LRs + memory ───
            **group_lrs,
            **mem_stats,
        })

    step += 1

    # ─── Periodic GC (from nanochat — prevents memory accumulation) ───
    if step % 5000 == 0:
        gc.collect()

# ═══════════════════════════════════════════════════════════════════
# End of training — cleanup
# ═══════════════════════════════════════════════════════════════════
print0(f"\nTraining complete!")
print0(f"  Total steps: {step}")
print0(f"  Total tokens: {tokens_processed:,}")
print0(f"  Total time: {total_training_time:.1f}s")
print0(f"  Best EMA val bpb: {min_ema_val_bpb:.4f}" if min_ema_val_bpb is not None else "  Best EMA val bpb: N/A (eval disabled)")
print0(f"  Final smooth loss: {smooth_train_loss:.4f}")

wandb_run.finish()

# Re-enable GC before cleanup
gc.enable()
gc.collect()
compute_cleanup()
