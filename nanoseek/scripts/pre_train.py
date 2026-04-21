"""
NanoSeek Stage 1 Pre-Training Script.

Usage:
    Single GPU:  python -m nanoseek.scripts.pre_train
    Multi-GPU:   torchrun --nproc_per_node=8 -m nanoseek.scripts.pre_train

CPU/Macbook smoke test:
    python -m nanoseek.scripts.pre_train --scale ablation --num-iterations 20 \
        --device-batch-size 1 --eval-tokens=512 --eval-every=-1
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import time
import math
import argparse
import signal
import random
from dataclasses import asdict
from contextlib import contextmanager

import numpy as np
import wandb
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_

from nanoseek.nanoseek.config import NanoSeekConfig, get_config, create_from_depth
from nanoseek.nanoseek.model import NanoSeekModel, get_mtp_loss_weight
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

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="NanoSeek Stage 1 Pre-Training")

# Run
parser.add_argument("--run", type=str, default="dummy",
                    help="wandb run name ('dummy' disables logging)")
parser.add_argument("--device-type", type=str, default="",
                    help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--seed", type=int, default=42)

# Model scale
parser.add_argument("--scale", type=str, default="ablation",
                    help="config scale: anchor, ablation, 1b, or d<N>")
parser.add_argument("--depth", type=int, default=-1,
                    help="model depth (overrides --scale)")
parser.add_argument("--aspect-ratio", type=int, default=80,
                    help="width = ceil(depth × ratio / 128) × 128")

# muP reference
parser.add_argument("--mup-ref-width", type=int, default=1280,
                    help="reference hidden_size for muP scaling")
parser.add_argument("--mup-ref-batch-tokens", type=int, default=524288,
                    help="reference batch size in tokens")

# Learning rates
parser.add_argument("--matrix-lr", type=float, default=0.02,
                    help="base Muon LR for hidden weights")
parser.add_argument("--embedding-lr", type=float, default=0.3,
                    help="base AdamW LR for embeddings")
parser.add_argument("--unembedding-lr", type=float, default=0.008,
                    help="base AdamW LR for lm_head")
parser.add_argument("--router-lr", type=float, default=3e-4,
                    help="AdamW LR for router (CONSTANT across scales)")
parser.add_argument("--norm-lr", type=float, default=3e-4,
                    help="AdamW LR for norms (CONSTANT across scales)")
parser.add_argument("--weight-decay", type=float, default=0.1)

# Training
parser.add_argument("--num-iterations", type=int, default=-1)
parser.add_argument("--device-batch-size", type=int, default=4,
                    help="micro-batch size per GPU (sequences)")
parser.add_argument("--total-batch-size", type=int, default=-1,
                    help="total batch in tokens (-1 = auto)")
parser.add_argument("--target-flops", type=float, default=-1.0)
parser.add_argument("--target-param-data-ratio", type=float, default=-1.0)
parser.add_argument("--total-tokens", type=int, default=-1)
parser.add_argument("--hidden-size", type=int, default=-1)

# Evaluation
parser.add_argument("--eval-every", type=int, default=250,
                    help="evaluate ema_val_bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=10_000_000)
parser.add_argument("--save-every", type=int, default=1000)

# EMA
parser.add_argument("--ema-decay", type=float, default=0.9999)
parser.add_argument("--ema-every", type=int, default=10,
                    help="update EMA every N steps")

# Resume
parser.add_argument("--resume-from-step", type=int, default=-1)

# MoE ablation flags
parser.add_argument("--no-seq-aux", action="store_true",
                    help="Ablation: disable auxiliary loss (alpha=0)")
parser.add_argument("--aux-loss-type", type=str, default="bias",
                    choices=["bias", "classic"],
                    help="Ablation: 'bias' (V3 aux-loss-free) or 'classic' "
                         "(Switch-Transformer differentiable load-balance loss)")
parser.add_argument("--aux-loss-alpha", type=float, default=0.01,
                    help="Aux loss alpha when --aux-loss-type=classic")
parser.add_argument("--kv-lora-ratio", type=float, default=-1.0,
                    help="Override KV LoRA compression ratio (default: 0.070 from DeepSeek V3). "
                         "Higher = less compression, better small-scale quality. "
                         "Ablation ladder: 0.07 (V3 default), 0.15, 0.25, 0.50.")
parser.add_argument("--no-mtp", action="store_true",
                    help="Ablation: disable MTP")
parser.add_argument("--no-shared-experts", action="store_true",
                    help="Ablation: remove shared experts")
parser.add_argument("--no-compile", action="store_true",
                    help="Skip torch.compile")

# Architecture overrides
parser.add_argument("--num-experts", type=int, default=-1)
parser.add_argument("--top-k", type=int, default=-1)
parser.add_argument("--n-group", type=int, default=-1)
parser.add_argument("--topk-group", type=int, default=-1)
parser.add_argument("--routed-scaling-factor", type=float, default=-1.0)

args = parser.parse_args()
user_config = vars(args).copy()

# -----------------------------------------------------------------------------
# Seed + compute init
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
if device_type == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_name)
    print0(f"GPU: {gpu_name} | Peak BF16 FLOPS: {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
    project="nanoseek", name=args.run, config=vars(args),
)

# -----------------------------------------------------------------------------
# Config
if args.depth > 0:
    config = create_from_depth(args.depth, args.aspect_ratio)
else:
    config = get_config(args.scale)

print0(f"Config ({config.scale_name}): hidden={config.hidden_size}, "
       f"heads={config.num_heads}, layers={config.num_layers}, "
       f"experts={config.n_routed_experts}, top_k={config.num_experts_per_tok}")

# Apply ablation overrides (write to flat fields, NOT config.moe.X)
if args.no_seq_aux:
    config.seq_aux_loss_alpha = 0.0
if args.aux_loss_type == "classic":
    # Classic mode: Switch-Transformer differentiable aux loss.
    # BOTH fields required: alpha controls the loss magnitude,
    # use_classic_aux_loss=True makes the gradient non-zero.
    # Without setting use_classic_aux_loss, aux_loss is always detached
    # regardless of alpha — the ablation would silently do nothing.
    config.seq_aux_loss_alpha = args.aux_loss_alpha
    config.use_classic_aux_loss = True
if args.no_mtp:
    config.num_mtp_modules = 0
if args.no_shared_experts:
    config.disable_shared_experts = True

# KV LoRA ratio override (ablation: compare 0.07 vs 0.15 vs 0.25)
if args.kv_lora_ratio > 0:
    config._kv_lora_rank_override = int(config.hidden_size * args.kv_lora_ratio)
    print0(f"KV LoRA ratio override: {args.kv_lora_ratio:.3f} → "
           f"kv_lora_rank={config._kv_lora_rank_override} "
           f"(default would be {int(config.hidden_size * 0.070)})")

# Architecture overrides
for val, attr in [
    (args.num_experts, "n_routed_experts"),
    (args.top_k, "num_experts_per_tok"),
    (args.n_group, "n_group"),
    (args.topk_group, "topk_group"),
    (args.total_tokens, "total_tokens"),
    (args.hidden_size, "hidden_size"),
]:
    if val > 0:
        setattr(config, attr, val)
if args.routed_scaling_factor > 0:
    config.routed_scaling_factor = args.routed_scaling_factor

# Derive heads from hidden_size after overrides
if args.hidden_size > 0:
    assert config.hidden_size % 128 == 0, \
        f"hidden_size={config.hidden_size} must be multiple of 128"
    config.num_heads = config.hidden_size // 128
    config.intermediate_size = int(config.hidden_size * 2.56)

# Validate
assert config.hidden_size % config.num_heads == 0
assert 0 < config.num_experts_per_tok <= config.n_routed_experts
assert config.n_routed_experts % config.n_group == 0
assert config.n_routed_experts // config.n_group >= 2
assert config.topk_group <= config.n_group

if not use_dummy_wandb:
    wandb_run.config.update(asdict(config), allow_val_change=True)

# -----------------------------------------------------------------------------
# Build model
with torch.device("meta"):
    model = NanoSeekModel(config)
model.to_empty(device=device)
model.init_weights()

param_counts = model.num_parameters()
n_active = param_counts['active']
n_total = param_counts['total']
num_flops_per_token = 6 * n_active
scaling_counts = model.num_scaling_params()
n_scaling = scaling_counts['scaling']

print0(f"Parameters: {n_active:,} active / {n_total:,} total "
       f"({n_total/n_active:.1f}× expansion)")
print0(f"FLOPs/token: {num_flops_per_token:.2e} | Scaling params: {n_scaling:,}")

# MLA KV compression summary — helps verify --kv-lora-ratio ablations
kv_cache_dim = config.mla.kv_lora_rank + config.mla.qk_rope_head_dim
full_kv_dim = config.num_heads * (config.mla.qk_nope_head_dim + config.mla.v_head_dim + config.mla.qk_rope_head_dim)
print0(f"MLA KV cache: {full_kv_dim} → {kv_cache_dim} dims/token/layer "
       f"({full_kv_dim/kv_cache_dim:.1f}× compression | "
       f"kv_lora_rank={config.mla.kv_lora_rank})")

# -----------------------------------------------------------------------------
# Training horizon + batch size
D_REF = 8_200_000_000
B_REF = 524_288

# Training tokens
if args.target_param_data_ratio > 0:
    config.total_tokens = int(args.target_param_data_ratio * n_scaling)
elif args.total_tokens > 0:
    config.total_tokens = args.total_tokens
elif config.total_tokens <= 0 and args.target_flops <= 0 and args.num_iterations <= 0:
    raise ValueError("No training horizon. Use --target-param-data-ratio, --target-flops, "
                     "--num-iterations, --total-tokens, or a named --scale")
elif config.total_tokens <= 0:
    config.total_tokens = int(20 * n_scaling)

# Batch size (Power Lines: B_opt ∝ D^0.383)
if args.total_batch_size > 0:
    config.global_batch_size = args.total_batch_size // config.sequence_length
elif config.global_batch_size <= 0:
    batch_ratio = config.total_tokens / D_REF
    predicted = B_REF * batch_ratio ** 0.383
    auto_batch = 2 ** round(math.log2(max(predicted, config.sequence_length)))
    config.global_batch_size = auto_batch // config.sequence_length
    print0(f"Auto batch: {auto_batch:,} tokens ({config.global_batch_size} seqs)")

total_batch_tokens = config.global_batch_size * config.sequence_length

# Iterations
if args.num_iterations > 0:
    num_iterations = args.num_iterations
elif args.target_flops > 0:
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_tokens))
else:
    num_iterations = config.total_tokens // total_batch_tokens
config.total_tokens = num_iterations * total_batch_tokens

chinchilla_ratio = config.total_tokens / n_active
print0(f"Training: {num_iterations:,} iters × {total_batch_tokens:,} tok/step "
       f"= {config.total_tokens:,} tokens | Tokens:Active = {chinchilla_ratio:.1f}x")

# Chinchilla compute-optimality check (Hoffmann et al., 2022: optimal ≈ 20x for dense).
# For MoE, we use N_active as the Chinchilla N. This convention follows DeepSeek V3.
# Note: anchor at 12x is intentionally undertrained (smoke test only, not quality target).
if chinchilla_ratio < 15.0 and args.num_iterations < 0:
    print0(f"[WARNING] Chinchilla: {chinchilla_ratio:.1f}x tokens/active_param < 15x. "
           f"Model is undertrained — results will be below architecture potential. "
           f"This is expected for anchor (smoke test). Not valid for quality comparisons.")

# -----------------------------------------------------------------------------
# muP scaling + weight decay
batch_lr_scale = math.sqrt(total_batch_tokens / args.mup_ref_batch_tokens)
width_lr_scale = args.mup_ref_width / config.hidden_size

D_ref_config = get_config("ablation").total_tokens
weight_decay_scaled = args.weight_decay * batch_lr_scale * (D_ref_config / config.total_tokens)

print0(f"muP: √(B/B_ref)={batch_lr_scale:.4f}, w_ref/w={width_lr_scale:.4f}, "
       f"WD={args.weight_decay}→{weight_decay_scaled:.6f}")

# -----------------------------------------------------------------------------
# Optimizer (MoE-aware parameter groups)
def build_optimizer(model):
    embed_p, lm_head_p, router_p, norm_p = [], [], [], []
    muon_shapes = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'embed_tokens' in name:
            embed_p.append(param)
        elif 'lm_head' in name:
            lm_head_p.append(param)
        elif 'gate.router_weight' in name:
            router_p.append(param)
        elif param.ndim == 1:
            norm_p.append(param)
        elif param.ndim >= 2:
            muon_shapes.setdefault(param.shape, []).append(param)

    hidden_lr = args.matrix_lr * batch_lr_scale * width_lr_scale
    boundary_lr = args.embedding_lr * batch_lr_scale
    groups = []
    if embed_p:
        groups.append(dict(kind='adamw', params=embed_p, lr=boundary_lr,
                           betas=(0.9, 0.95), eps=1e-10, weight_decay=0.001))
    if lm_head_p:
        groups.append(dict(kind='adamw', params=lm_head_p,
                           lr=args.unembedding_lr * batch_lr_scale,
                           betas=(0.9, 0.95), eps=1e-10, weight_decay=0.01))
    if router_p:
        groups.append(dict(kind='adamw', params=router_p, lr=args.router_lr,
                           betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
    if norm_p:
        groups.append(dict(kind='adamw', params=norm_p, lr=args.norm_lr,
                           betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
    for shape, params in muon_shapes.items():
        groups.append(dict(kind='muon', params=params, lr=hidden_lr,
                           momentum=0.95, ns_steps=5, beta2=0.9,
                           weight_decay=weight_decay_scaled))

    n_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_grouped = sum(p.numel() for g in groups for p in g['params'])
    assert n_model == n_grouped, f"Param mismatch: {n_model} vs {n_grouped}"

    Factory = DistMuonAdamW if ddp else MuonAdamW
    optimizer = Factory(groups)
    for g in optimizer.param_groups:
        g["initial_lr"] = g["lr"]
    return optimizer

optimizer = build_optimizer(model)

# -----------------------------------------------------------------------------
# LR schedule (warmup → constant → cosine decay)
warmup_steps = config.warmup_steps
constant_end = int(config.constant_phase_ratio * num_iterations)
decay_end = int(config.cosine_decay_end_ratio * num_iterations)
lr_min_ratio = config.lr_min / config.learning_rate

def get_lr_multiplier(step):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    elif step <= constant_end:
        return 1.0
    elif step <= decay_end:
        progress = (step - constant_end) / max(decay_end - constant_end, 1)
        return lr_min_ratio + 0.5 * (1.0 - lr_min_ratio) * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return lr_min_ratio

def get_muon_momentum(step):
    frac = min(step / 400, 1.0)
    return (1 - frac) * 0.85 + frac * 0.97

def get_batch_warmup_accum(step, target_accum, warmup_frac=0.10):
    warmup_end = int(num_iterations * warmup_frac)
    if step >= warmup_end or target_accum <= 1:
        return target_accum
    min_a = max(1, math.ceil(target_accum / 5))
    return max(min_a, round(min_a + (target_accum - min_a) * step / max(warmup_end, 1)))

# -----------------------------------------------------------------------------
# Distributed gradient clipping
def distributed_clip_grad_norm_(parameters, max_norm):
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return torch.tensor(0.0)
    local_sq = torch.zeros(1, device=parameters[0].device)
    for p in parameters:
        local_sq += p.grad.data.float().pow(2).sum()
    dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
    global_norm = (local_sq / dist.get_world_size()).sqrt().item()
    if max_norm < float('inf') and global_norm > max_norm:
        for p in parameters:
            p.grad.data.mul_(max_norm / (global_norm + 1e-6))
    return torch.tensor(global_norm)

# -----------------------------------------------------------------------------
# EMA tracker (CPU-side, Karras decay warmup)
class EMATracker:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.count = 0
        self.shadow = {n: p.detach().clone().to("cpu") for n, p in model.named_parameters()}

    @torch.no_grad()
    def reset_from_model(self, model):
        self.shadow = {n: p.detach().clone().to("cpu") for n, p in model.named_parameters()}
        self.count = 0

    @torch.no_grad()
    def step(self, model):
        self.count += 1
        d = min(self.decay, 1.0 - 1.0 / (1.0 + self.count))
        for n, p in model.named_parameters():
            self.shadow[n].lerp_(p.detach().to("cpu"), 1 - d)

    @contextmanager
    def apply(self, model):
        orig = {n: p.detach().clone() for n, p in model.named_parameters()}
        for n, p in model.named_parameters():
            p.data.copy_(self.shadow[n].to(p.device))
        try:
            yield
        finally:
            for n, p in model.named_parameters():
                p.data.copy_(orig[n])

    def state_dict(self):
        d = {k: v.clone() for k, v in self.shadow.items()}
        d['__ema_update_count__'] = self.count
        return d

    def load_state_dict(self, sd):
        for k, v in sd.items():
            if k == '__ema_update_count__':
                self.count = v
            elif k in self.shadow:
                self.shadow[k].copy_(v)

# -----------------------------------------------------------------------------
# BPB evaluation
@torch.no_grad()
def evaluate_bpb(model, val_loader, steps, token_bytes):
    dev = next(model.parameters()).device
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=dev)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=dev)
    it = iter(val_loader)
    for _ in range(steps):
        x, y = next(it)
        logits = model(x)['logits']
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                               ignore_index=-1, reduction='none')
        y_flat = y.view(-1)
        valid = y_flat >= 0
        y_safe = torch.where(valid, y_flat, torch.zeros_like(y_flat))
        num_bytes = torch.where(valid, token_bytes[y_safe],
                                torch.zeros_like(y_flat, dtype=token_bytes.dtype))
        total_nats += (loss * (num_bytes > 0)).sum()
        total_bytes += num_bytes.sum()
    if is_ddp_initialized():
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    return total_nats.item() / math.log(2) / max(total_bytes.item(), 1)

# -----------------------------------------------------------------------------
# Compile, EMA, checkpoints
orig_model = model
if not args.no_compile:
    model = torch.compile(model, dynamic=False)

ema_tracker = EMATracker(orig_model, decay=args.ema_decay)

checkpoint_dir = os.path.join("checkpoints", f"nanoseek_{config.scale_name}", args.run)
ckpt_manager = CheckpointManager(checkpoint_dir, save_every=args.save_every,
                                  keep_last_n=3, save_optimizer=True)

# Graceful shutdown on SIGTERM/SIGINT
_shutdown = False
def _handler(sig, frame):
    global _shutdown
    _shutdown = True
signal.signal(signal.SIGTERM, _handler)
signal.signal(signal.SIGINT, _handler)

# -----------------------------------------------------------------------------
# Resume from checkpoint
resume_step = 0
resume_dl_state = None
resume_loop = {}

if args.resume_from_step >= 0:
    step_arg = args.resume_from_step if args.resume_from_step > 0 else None
    model_state, metadata, loaded_step = load_checkpoint(checkpoint_dir, step=step_arg, device=device)
    orig_model.load_state_dict(model_state)
    print0(f"Resumed model from step {loaded_step}")

    opt_state = load_optimizer_checkpoint(checkpoint_dir, loaded_step, device=device)
    if opt_state:
        optimizer.load_state_dict(opt_state)
        print0("  Optimizer loaded")
    try:
        ema_state = load_ema_checkpoint(checkpoint_dir, loaded_step, device=torch.device("cpu"))
        ema_tracker.load_state_dict(ema_state)
        print0("  EMA loaded")
    except FileNotFoundError:
        ema_tracker.reset_from_model(orig_model)
        print0("  EMA re-initialized (no checkpoint found)")

    resume_step = loaded_step
    resume_dl_state = metadata.get("dataloader_state_dict")
    resume_loop = metadata.get("loop_state", {})
    if not args.no_compile:
        model = torch.compile(orig_model, dynamic=False)

# -----------------------------------------------------------------------------
# Data loaders
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
print0(f"Vocab size: {tokenizer.get_vocab_size():,}")

train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, args.device_batch_size, config.sequence_length,
    split="train", device=device, resume_state_dict=resume_dl_state, fim_rate=0.10,
)
eval_bs = min(args.device_batch_size, 2)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, eval_bs, config.sequence_length, split="val", device=device,
)

# Gradient accumulation
tokens_per_micro = args.device_batch_size * config.sequence_length
world_tokens = tokens_per_micro * ddp_world_size
assert total_batch_tokens % world_tokens == 0, \
    f"total_batch_tokens ({total_batch_tokens}) must divide world_tokens ({world_tokens})"
target_accum = total_batch_tokens // world_tokens
print0(f"Grad accum: {target_accum} steps ({world_tokens:,} tok/fwdbwd, "
       f"{total_batch_tokens:,} tok/step)")

# Prefetch first batch
x, y, dl_state = next(train_loader)

# Per-group grad norm tracking (catch MoE gate collapse)
_grad_groups = {}
for name, param in orig_model.named_parameters():
    if not param.requires_grad:
        continue
    if 'embed_tokens' in name:
        key = "embed"
    elif 'lm_head' in name:
        key = "lm_head"
    elif 'gate.router_weight' in name:
        key = "gate"
    elif 'self_attn' in name:
        key = "attn"
    elif 'shared_expert' in name:
        key = "shared"
    elif param.ndim == 3:
        key = "experts"
    elif 'mtp' in name:
        key = "mtp"
    elif param.ndim == 1:
        key = "norm"
    else:
        key = "attn"
    _grad_groups.setdefault(key, []).append(param)
_grad_groups = {k: v for k, v in _grad_groups.items() if v}

# GC management
gc.collect()
gc.freeze()
gc.disable()

# -----------------------------------------------------------------------------
# Training loop
step = resume_step
tokens_processed = resume_loop.get("tokens_processed", 0) if resume_step > 0 else 0
smooth_loss = resume_loop.get("smooth_train_loss", 0.0)
ema_val_bpb = None
min_ema_val_bpb = resume_loop.get("min_ema_val_bpb")
total_time = resume_loop.get("total_training_time", 0.0)
nan_count = 0

print0(f"LR schedule: warmup={warmup_steps}, constant→{constant_end}, "
       f"cosine→{decay_end}, lr_min_ratio={lr_min_ratio:.2f}")

while True:
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * tokens_processed

    # --- Evaluate (RULE 1: always use EMA weights) ---
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        if step > 0:
            ema_tracker.step(orig_model)
        model.eval()
        eval_steps = args.eval_tokens // (eval_bs * config.sequence_length * ddp_world_size)
        with ema_tracker.apply(orig_model):
            ema_val_bpb = evaluate_bpb(orig_model, build_val_loader(), eval_steps, token_bytes)
            load_stats = orig_model.get_expert_load_stats()
        model.train()

        if min_ema_val_bpb is None or ema_val_bpb < min_ema_val_bpb:
            min_ema_val_bpb = ema_val_bpb
        H_load = load_stats['entropy'].item()
        max_H_load = math.log2(max(config.n_routed_experts, 1))  # 4.0 bits for 16 experts (nano scale)
        print0(f"Step {step:05d} | EMA val bpb: {ema_val_bpb:.4f} | "
               f"H_load: {H_load:.2f}/{max_H_load:.1f} bits")

        # Routing collapse early warning.
        # H_load < 2.0 bits = fewer than 4 experts receiving meaningful traffic.
        # At step 0 this is expected (routing not yet learned). By step 500 it should
        # be > 3.0 bits. If it stays collapsed, check: (1) GAMMA scale,
        # (2) router init, (3) gradient clipping catching routing spikes.
        if step > 0 and H_load < 2.0:
            print0(f"[ROUTING ALERT] H_load={H_load:.2f} < 2.0 bits at step {step}. "
                   f"Routing likely collapsed. "
                   f"Action: check train/gamma, grad_norm/gate in wandb. "
                   f"Consider --aux-loss-type classic as recovery.")

        wandb_run.log({
            "step": step, "total_training_flops": flops_so_far,
            "total_training_time": total_time,
            "ema_val/bpb": ema_val_bpb, "train/H_load": H_load,
            "moe/load_per_expert": wandb.Histogram(load_stats['load_per_expert'].cpu().numpy()),
        })

    # --- Checkpoint ---
    should_save = ckpt_manager.should_save(step, last_step) or _shutdown
    if should_save:
        if ddp:
            dist.barrier()
        ckpt_manager.save(
            step=step, model_state=orig_model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            metadata={
                "step": step, "tokens_processed": tokens_processed,
                "ema_val_bpb": ema_val_bpb, "config": asdict(config),
                "dataloader_state_dict": dl_state,
                "loop_state": {
                    "tokens_processed": tokens_processed,
                    "min_ema_val_bpb": min_ema_val_bpb,
                    "smooth_train_loss": smooth_loss,
                    "total_training_time": total_time,
                },
            },
            ema_state=ema_tracker.state_dict(), rank=ddp_rank,
        )
        if ddp:
            dist.barrier()
        if _shutdown:
            print0(f"Emergency checkpoint saved at step {step}. Exiting.")
            wandb_run.finish()
            gc.enable()
            gc.collect()
            compute_cleanup()
            import sys
            sys.exit(0)

    if last_step:
        break

    # --- Single training step ---
    cur_accum = get_batch_warmup_accum(step, target_accum)
    cur_batch = world_tokens * cur_accum

    synchronize()
    t0 = time.time()
    loss_accum = main_accum = mtp_accum = aux_accum = 0.0

    for micro in range(cur_accum):
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                            enabled=(device_type == "cuda")):
            mtp_lambda = get_mtp_loss_weight(
                tokens_processed, config.total_tokens,
                config.mtp.mtp_loss_weight_initial, config.mtp.mtp_loss_weight_final,
                config.mtp.mtp_loss_transition_ratio,
            )
            out = model(x, labels=x, mtp_lambda=mtp_lambda)
            loss = out['loss']
        loss_accum += loss.detach()
        main_accum += out['main_loss'].detach()
        mtp_accum += out['mtp_loss'].detach()
        aux_accum += out['aux_loss'].detach()
        (loss / cur_accum).backward()
        x, y, dl_state = next(train_loader)

    train_loss = loss_accum / cur_accum

    # Gradient clipping
    clip_norm = config.max_grad_norm
    if ddp:
        grad_norm = distributed_clip_grad_norm_(orig_model.parameters(), max_norm=clip_norm)
    else:
        grad_norm = clip_grad_norm_(orig_model.parameters(), max_norm=clip_norm)

    # LR + momentum schedule
    lrm = get_lr_multiplier(step)
    momentum = get_muon_momentum(step)
    for g in optimizer.param_groups:
        g["lr"] = g["initial_lr"] * lrm
        if g['kind'] == 'muon':
            g["momentum"] = momentum

    # Per-group grad norms — MUST be computed before zero_grad, after clipping
    per_group_gn = {}
    if step % config.log_every_steps == 0:
        for gname, params in _grad_groups.items():
            grads = [p.grad.data for p in params if p.grad is not None]
            if grads:
                per_group_gn[gname] = torch.stack(
                    [g.float().pow(2).sum() for g in grads]).sum().sqrt().item()

    optimizer.step()
    model.zero_grad(set_to_none=True)

    # MoE load-balance bias update (aux-loss-free, DeepSeek V3 style)
    if args.aux_loss_type != "classic":
        orig_model.update_load_balance_bias(tokens_processed, config.total_tokens)

    # EMA update
    if step % args.ema_every == 0:
        ema_tracker.step(orig_model)

    tokens_processed += cur_batch
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    if step > 10:
        total_time += dt

    # MoE health
    train_loss_f = train_loss.item()
    gamma = orig_model.get_gamma(tokens_processed, config.total_tokens)
    load_stats = orig_model.get_expert_load_stats()
    H_load = load_stats['entropy'].item()
    tok_per_sec = int(cur_batch / dt) if dt > 0 else 0
    mfu = 100 * num_flops_per_token * cur_batch / dt / (gpu_peak_flops * ddp_world_size)

    # NaN check
    if not math.isfinite(train_loss_f):
        nan_count += 1
        print0(f"[CRITICAL] NaN/Inf loss at step {step} ({nan_count}/3)")
        if nan_count >= 3:
            wandb_run.finish()
            gc.enable()
            gc.collect()
            compute_cleanup()
            import sys
            sys.exit(1)
    else:
        nan_count = 0

    # EMA-smoothed loss for logging
    ema_beta = 0.9
    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * train_loss_f
    debiased = smooth_loss / (1 - ema_beta ** (step + 1))

    pct = 100 * step / num_iterations
    print0(
        f"step {step:05d}/{num_iterations} ({pct:.1f}%) | "
        f"loss: {debiased:.4f} (main:{main_accum.item()/cur_accum:.4f} "
        f"mtp:{mtp_accum.item()/cur_accum:.4f} aux:{aux_accum.item()/cur_accum:.6f}) | "
        f"H_load: {H_load:.2f} | lr×: {lrm:.3f} | γ: {gamma:.4f} | "
        f"grad: {grad_norm:.2f} | dt: {dt:.2f}s | mfu: {mfu:.1f}% | tok/s: {tok_per_sec:,}"
    )

    if step % config.log_every_steps == 0:
        log_data = {
            "step": step, "tokens_processed": tokens_processed,
            "train/loss": debiased, "train/loss_raw": train_loss_f,
            "train/main_loss": main_accum.item() / cur_accum,
            "train/mtp_loss": mtp_accum.item() / cur_accum,
            "train/aux_loss": aux_accum.item() / cur_accum,
            "train/lr_multiplier": lrm, "train/grad_norm": grad_norm.item(),
            "train/H_load": H_load, "train/gamma": gamma,
            "train/mfu": mfu, "train/tok_per_sec": tok_per_sec,
            "train/step_time_s": dt, "train/batch_tokens": cur_batch,
        }
        if device_type == "cuda":
            log_data["gpu/memory_allocated_gb"] = torch.cuda.memory_allocated(device) / 1e9
        log_data.update({f"grad_norm/{k}": v for k, v in per_group_gn.items()})
        wandb_run.log(log_data)

    step += 1

    if step % 5000 == 0:
        gc.collect()

# -----------------------------------------------------------------------------
# Cleanup
print0(f"\nTraining complete!")
print0(f"  Steps: {step} | Tokens: {tokens_processed:,} | Time: {total_time:.1f}s")
if min_ema_val_bpb is not None:
    print0(f"  Best EMA val bpb: {min_ema_val_bpb:.4f}")

wandb_run.finish()
gc.enable()
gc.collect()
compute_cleanup()
