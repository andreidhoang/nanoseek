"""
Unified evaluation script for NanoSeek base models.

Supports evaluation modes (comma-separated):
  --eval core       : CORE metric (accuracy on ICL tasks)
  --eval bpb        : Bits per byte on train/val splits
  --eval sample     : Generate samples from the model
  --eval domain_bpb : Per-domain BPB (code, math, science, web, books)
  --eval i_spec     : Expert specialization mutual information
  --eval mtp_accept : MTP acceptance rate (test-time scaling signal)
  --eval moe_diag   : Enhanced MoE diagnostics (dead experts, Gini)

Default: --eval core,bpb,sample

Key features:
  - EMA weight evaluation (RULE 1: ALL eval uses EMA weights)
  - NanoSeek-specific BPB (handles dict-returning model interface)
  - MoE health metrics (expert load balance, aux loss)
  - Multi-token prediction evaluation

Examples:

    # Evaluate a NanoSeek model (1B scale) using 8 GPUs
    torchrun --nproc_per_node=8 -m nanoseek.scripts.base_eval --scale 1b --step 5000

    # Evaluate from a specific checkpoint with EMA weights
    torchrun --nproc_per_node=8 -m nanoseek.scripts.base_eval \
        --checkpoint-dir checkpoints/nanoseek_1b --step 10000 --use-ema

    # Quick evaluation using a single GPU (subset of data)
    python -m nanoseek.scripts.base_eval --scale anchor --device-batch-size=16 \
        --max-per-task=100 --split-tokens=524288 --eval bpb,sample

    # Evaluate only BPB on validation split
    python -m nanoseek.scripts.base_eval --scale 500m --eval bpb --split-tokens=1048576
"""

import os
import sys
import csv
import time
import math
import json
import yaml
import shutil
import random
import zipfile
import tempfile
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist

# NanoSeek imports
from nanoseek.nanoseek.config import (
    NanoSeekConfig,
    get_nanoseek_config,
    get_nanoseek_500m_config,
    get_nanoseek_anchor_config,
)
from nanoseek.nanoseek.model import NanoSeekModel
from nanoseek.nanoseek.common import (
    compute_init, compute_cleanup, print0, get_base_dir,
    autodetect_device_type, download_file_with_lock, is_ddp_initialized,
)
from nanoseek.nanoseek.engine import Engine

# Nanochat imports (shared infrastructure)
from nanochat.tokenizer import HuggingFaceTokenizer, get_tokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.core_eval import evaluate_task


# -----------------------------------------------------------------------------
# Model Loading with EMA Support
# -----------------------------------------------------------------------------

def load_nanoseek_model(
    scale: str,
    device: torch.device,
    checkpoint_dir: Optional[str] = None,
    step: Optional[int] = None,
    use_ema: bool = False,
) -> Tuple[NanoSeekModel, NanoSeekConfig, HuggingFaceTokenizer, Optional[Dict]]:
    """
    Load a NanoSeek model with optional EMA weights.
    
    Args:
        scale: Model scale ('anchor', '500m', '1b')
        device: Device to load model on
        checkpoint_dir: Directory containing checkpoints (default: checkpoints/nanoseek_{scale})
        step: Specific step to load (default: latest)
        use_ema: Whether to load EMA weights instead of raw weights
        
    Returns:
        model: Loaded NanoSeekModel (eval mode, on device)
        config: NanoSeekConfig used
        tokenizer: HuggingFaceTokenizer
        ema_state: EMA state dict if use_ema=True, else None
    """
    # Get config for scale
    config_map = {
        "anchor": get_nanoseek_anchor_config,
        "500m": get_nanoseek_500m_config,
        "1b": get_nanoseek_config,
    }
    if scale not in config_map:
        raise ValueError(f"Unknown scale: {scale}. Choose from {list(config_map.keys())}")
    
    config = config_map[scale]()
    print0(f"Loading NanoSeek-{scale} config:")
    print0(f"  Hidden size: {config.hidden_size}")
    print0(f"  Layers: {config.num_layers}")
    print0(f"  Active params: ~{config.estimated_active_params/1e6:.0f}M")
    print0(f"  Total params: ~{config.estimated_total_params/1e6:.0f}M")
    
    # Build model
    print0("Building model...")
    with torch.device("meta"):
        model = NanoSeekModel(config)
    
    model.to_empty(device=device)
    model.init_weights()
    
    # Determine checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join("checkpoints", f"nanoseek_{scale}")
    
    # Load checkpoint if exists
    ema_state = None
    checkpoint_path = None
    
    if os.path.exists(checkpoint_dir):
        # Find checkpoint files
        model_files = list(Path(checkpoint_dir).glob("model_*.pt"))
        
        if model_files:
            if step is not None:
                # Load specific step
                checkpoint_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
                if not os.path.exists(checkpoint_path):
                    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            else:
                # Load latest
                latest = max(model_files, key=lambda p: int(p.stem.split('_')[1]))
                checkpoint_path = str(latest)
                step = int(latest.stem.split('_')[1])
            
            print0(f"Loading checkpoint: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            
            # Load EMA if requested
            if use_ema:
                ema_path = os.path.join(checkpoint_dir, f"ema_{step:06d}.pt")
                if os.path.exists(ema_path):
                    print0(f"Loading EMA weights: {ema_path}")
                    ema_state = torch.load(ema_path, map_location=device, weights_only=True)
                    # Apply EMA weights to model
                    for name, param in model.named_parameters():
                        if name in ema_state:
                            param.data.copy_(ema_state[name])
                else:
                    print0(f"Warning: EMA checkpoint not found at {ema_path}, using raw weights")
        else:
            print0(f"Warning: No checkpoints found in {checkpoint_dir}, using fresh initialization")
    else:
        print0(f"Warning: Checkpoint directory not found: {checkpoint_dir}")
        if step is not None:
            raise FileNotFoundError(f"Cannot load step {step}: directory doesn't exist")
    
    # Get tokenizer
    tokenizer = get_tokenizer()
    
    # Set to eval mode
    model.eval()
    
    return model, config, tokenizer, ema_state


# -----------------------------------------------------------------------------
# HuggingFace Model Wrapper (for comparing NanoSeek against HF models)
# -----------------------------------------------------------------------------

class HFModelWrapper:
    """Lightweight wrapper to give HuggingFace models a NanoSeek-compatible interface."""
    
    def __init__(self, model, max_seq_len: int = 4096):
        self.model = model
        self.max_seq_len = max_seq_len
        self.config = model.config if hasattr(model, 'config') else None
    
    def __call__(self, input_ids, labels=None, **kwargs):
        """Forward pass compatible with NanoSeek's interface."""
        outputs = self.model(input_ids, labels=labels)
        # Return dict like NanoSeek
        return {
            "loss": outputs.loss if hasattr(outputs, 'loss') else None,
            "logits": outputs.logits if hasattr(outputs, 'logits') else outputs,
            "past_key_values": getattr(outputs, 'past_key_values', None),
            "aux_loss": torch.tensor(0.0),  # No MoE for HF models
            "main_loss": outputs.loss if hasattr(outputs, 'loss') else None,
            "mtp_loss": torch.tensor(0.0),
            "mtp_lambda": 0.0,
        }
    
    def get_device(self):
        return next(self.model.parameters()).device


def load_hf_model(hf_path: str, device: torch.device):
    """Load a HuggingFace model with NanoSeek-compatible interface."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print0(f"Loading HuggingFace model from: {hf_path}")
    model = AutoModelForCausalLM.from_pretrained(hf_path)
    model.to(device)
    model.eval()
    
    # Determine max sequence length
    max_seq_len = getattr(model.config, 'max_position_embeddings', 2048)
    if "gpt2" in hf_path.lower():
        max_seq_len = 1024
    
    wrapped_model = HFModelWrapper(model, max_seq_len=max_seq_len)
    tokenizer = HuggingFaceTokenizer.from_pretrained(hf_path)
    
    return wrapped_model, tokenizer, max_seq_len


# -----------------------------------------------------------------------------
# BPB Evaluation (NanoSeek-specific)
# -----------------------------------------------------------------------------

@torch.no_grad()
def evaluate_nanoseek_bpb(
    model: NanoSeekModel,
    batches,
    steps: int,
    token_bytes: torch.Tensor,
) -> float:
    """
    Compute bits-per-byte for NanoSeek model.
    
    Uses logits from model forward (no labels) and computes
    per-token CE against the dataloader's pre-shifted targets.
    This avoids the double-shift bug.
    
    Args:
        model: NanoSeekModel (eval mode)
        batches: DataLoader yielding (x, y) tuples
        steps: Number of eval steps
        token_bytes: [vocab_size] tensor of byte lengths per token
        
    Returns:
        bpb: Bits per byte (lower is better)
    """
    device = next(model.parameters()).device
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
    
    batch_iter = iter(batches)
    
    for _ in range(steps):
        x, y = next(batch_iter)
        
        # Forward WITHOUT labels to get logits only
        outputs = model(x, use_cache=False)
        logits = outputs['logits']  # [B, T, V]
        
        # Compute per-token CE against pre-shifted targets
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


# -----------------------------------------------------------------------------
# CORE Evaluation (In-Context Learning Benchmark)
# -----------------------------------------------------------------------------

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


def place_eval_bundle(file_path: str):
    """Unzip eval_bundle.zip and place it in the base directory."""
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    
    print0(f"Placed eval_bundle directory at {eval_bundle_dir}")


class ModelWrapperForCore:
    """
    Wrapper to adapt NanoSeek's dict-returning interface for CORE evaluation.
    nanochat's evaluate_task expects model(input_ids) -> logits or loss.
    """
    
    def __init__(self, model: NanoSeekModel):
        self.model = model
        self.config = model.config
    
    def __call__(self, input_ids, targets=None, loss_reduction='mean'):
        """
        Compatible with nanochat's evaluate_task interface.
        
        Args:
            input_ids: [B, S] token IDs
            targets: [B, S] target token IDs (optional)
            loss_reduction: 'mean', 'sum', or 'none'
            
        Returns:
            If targets is None: logits [B, S, V]
            If targets is not None: scalar loss
        """
        outputs = self.model(input_ids, use_cache=False)
        logits = outputs['logits']
        
        if targets is None:
            return logits
        
        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction
        )
        return loss
    
    def get_device(self):
        return next(self.model.parameters()).device


def evaluate_core(
    model: NanoSeekModel,
    tokenizer: HuggingFaceTokenizer,
    device: torch.device,
    max_per_task: int = -1,
) -> Dict:
    """
    Evaluate a NanoSeek model on the CORE benchmark.
    
    Args:
        model: NanoSeekModel (eval mode)
        tokenizer: Tokenizer for the model
        device: Device to run evaluation on
        max_per_task: Max examples per task (-1 = all)
        
    Returns:
        dict with 'results', 'centered_results', and 'core_metric'
    """
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    
    # Download eval bundle if needed
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(
            EVAL_BUNDLE_URL,
            "eval_bundle.zip",
            postprocess_fn=place_eval_bundle
        )
    
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        eval_config = yaml.safe_load(f)
    tasks = eval_config['icl_tasks']
    
    # Load random baseline values
    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row['Eval Task']
            random_baseline = row['Random baseline']
            random_baselines[task_name] = float(random_baseline)
    
    # Wrap model for CORE evaluation compatibility
    wrapped_model = ModelWrapperForCore(model)
    
    # Evaluate each task
    results = {}
    centered_results = {}
    
    for task in tasks:
        start_time = time.time()
        label = task['label']
        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }
        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end='')
        
        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]
        
        # Shuffle for consistent subsampling
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]
        
        accuracy = evaluate_task(wrapped_model, tokenizer, data, device, task_meta)
        results[label] = accuracy
        
        random_baseline = random_baselines[label]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        
        elapsed = time.time() - start_time
        print0(f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {elapsed:.2f}s")
    
    core_metric = sum(centered_results.values()) / len(centered_results)
    
    return {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric
    }


# -----------------------------------------------------------------------------
# Sample Generation
# -----------------------------------------------------------------------------

def generate_samples(
    model: NanoSeekModel,
    tokenizer: HuggingFaceTokenizer,
    device: torch.device,
    prompts: Optional[list] = None,
) -> Dict[str, list]:
    """
    Generate samples from the model for qualitative evaluation.
    
    Args:
        model: NanoSeekModel (eval mode)
        tokenizer: Tokenizer
        device: Device
        prompts: List of prompt strings (uses defaults if None)
        
    Returns:
        dict with 'conditioned' and 'unconditioned' samples
    """
    if prompts is None:
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
            "The Pythagorean theorem states that",
        ]
    
    engine = Engine(model, tokenizer)
    samples = []
    unconditioned_samples = []
    
    print0("\n" + "="*80)
    print0("Model Samples")
    print0("="*80)
    
    # Conditioned samples
    print0("\nConditioned samples:")
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, bos=True, eos=False)
        sample_tokens, _ = engine.generate_batch(
            tokens,
            num_samples=1,
            max_tokens=32,
            temperature=0.0,  # Greedy for consistency
            seed=42,
        )
        sample_str = tokenizer.decode(sample_tokens[0])
        print0("-" * 80)
        print0(sample_str)
        samples.append(sample_str)
    
    # Unconditioned samples
    print0("\nUnconditioned samples:")
    tokens = tokenizer.encode("", bos=True, eos=False)
    uncond_tokens, _ = engine.generate_batch(
        tokens,
        num_samples=4,
        max_tokens=128,
        temperature=1.0,
        top_k=50,
        seed=42,
    )
    for sample in uncond_tokens:
        sample_str = tokenizer.decode(sample)
        print0("-" * 80)
        print0(sample_str)
        unconditioned_samples.append(sample_str)
    
    return {
        "conditioned": samples,
        "unconditioned": unconditioned_samples,
    }


# -----------------------------------------------------------------------------
# MoE Health Metrics
# -----------------------------------------------------------------------------

def evaluate_moe_health(model: NanoSeekModel, batch_x: torch.Tensor) -> Dict[str, float]:
    """
    Evaluate MoE routing health metrics.
    
    Args:
        model: NanoSeekModel
        batch_x: A batch of input IDs to run through the model
        
    Returns:
        dict with MoE health metrics
    """
    model.eval()
    with torch.no_grad():
        outputs = model(batch_x, use_cache=False)
    
    load_stats = model.get_expert_load_stats()
    
    metrics = {
        "aux_loss": outputs.get("aux_loss", torch.tensor(0.0)).item(),
        "load_entropy": load_stats.get("entropy", torch.tensor(0.0)).item(),
    }
    
    # Compute load balance coefficient of variation
    load_per_expert = load_stats.get("load_per_expert", torch.zeros(model.config.moe.n_routed_experts))
    if load_per_expert.sum() > 0:
        load_mean = load_per_expert.mean().item()
        load_std = load_per_expert.std().item()
        metrics["load_cv"] = load_std / load_mean if load_mean > 0 else 0.0
    else:
        metrics["load_cv"] = 0.0
    
    return metrics


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NanoSeek base model evaluation")
    
    # Evaluation modes
    parser.add_argument(
        '--eval', type=str, default='core,bpb,sample',
        help='Comma-separated evaluations: core,bpb,sample (default: all)'
    )
    
    # Model source (NanoSeek or HuggingFace)
    parser.add_argument(
        '--scale', type=str, default='1b',
        choices=['anchor', '500m', '1b'],
        help='NanoSeek model scale to evaluate'
    )
    parser.add_argument(
        '--hf-path', type=str, default=None,
        help='HuggingFace model path (e.g., openai-community/gpt2-xl)'
    )
    
    # Checkpoint loading
    parser.add_argument(
        '--checkpoint-dir', type=str, default=None,
        help='Checkpoint directory (default: checkpoints/nanoseek_{scale})'
    )
    parser.add_argument(
        '--step', type=int, default=None,
        help='Model step to load (default = latest)'
    )
    parser.add_argument(
        '--use-ema', action='store_true',
        help='Use EMA weights for evaluation (RULE 1)'
    )
    
    # BPB evaluation settings
    parser.add_argument(
        '--device-batch-size', type=int, default=16,
        help='Per-device batch size for BPB evaluation'
    )
    parser.add_argument(
        '--split-tokens', type=int, default=40*524288,
        help='Number of tokens to evaluate per split for BPB'
    )
    
    # CORE evaluation settings
    parser.add_argument(
        '--max-per-task', type=int, default=-1,
        help='Max examples per CORE task (-1 = all)'
    )
    
    # Device settings
    parser.add_argument(
        '--device-type', type=str, default='',
        help='cuda|cpu|mps (empty = autodetect)'
    )
    
    args = parser.parse_args()
    
    # Parse evaluation modes
    eval_modes = set(mode.strip() for mode in args.eval.split(','))
    valid_modes = {'core', 'bpb', 'sample', 'domain_bpb', 'i_spec', 'mtp_accept', 'moe_diag'}
    invalid = eval_modes - valid_modes
    if invalid:
        parser.error(f"Invalid eval modes: {invalid}. Valid: {valid_modes}")
    
    # Distributed / device setup
    device_type = autodetect_device_type() if args.device_type == '' else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    
    # Load model
    is_hf_model = args.hf_path is not None
    
    if is_hf_model:
        model, tokenizer, sequence_len = load_hf_model(args.hf_path, device)
        config = None  # HF models don't have NanoSeek config
        model_name = args.hf_path
        model_slug = args.hf_path.replace("/", "-")
        use_ema = False
    else:
        model, config, tokenizer, ema_state = load_nanoseek_model(
            scale=args.scale,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            step=args.step,
            use_ema=args.use_ema,
        )
        sequence_len = config.sequence_length if config else 4096
        model_name = f"NanoSeek-{args.scale}"
        if args.step is not None:
            model_name += f" (step {args.step})"
            model_slug = f"nanoseek_{args.scale}_{args.step:06d}"
        else:
            model_slug = f"nanoseek_{args.scale}_latest"
        if args.use_ema:
            model_name += " [EMA]"
            model_slug += "_ema"
    
    print0(f"\nEvaluating model: {model_name}")
    print0(f"Evaluation modes: {', '.join(sorted(eval_modes))}")
    
    # Results storage
    core_results = None
    bpb_results = {}
    samples = None
    moe_health = None
    
    # --- Sample Generation ---
    if 'sample' in eval_modes and not is_hf_model:
        print0("\n" + "="*80)
        samples = generate_samples(model, tokenizer, device)
    elif 'sample' in eval_modes and is_hf_model:
        print0("\nSkipping sampling for HuggingFace models (not yet supported)")
    
    # --- BPB Evaluation ---
    if 'bpb' in eval_modes:
        print0("\n" + "="*80)
        print0("BPB Evaluation")
        print0("="*80)
        
        # Get token bytes for BPB calculation
        token_bytes = get_token_bytes(device=device)
        
        # Adjust tokens per step
        tokens_per_step = args.device_batch_size * sequence_len * ddp_world_size
        if args.split_tokens % tokens_per_step != 0:
            args.split_tokens = (args.split_tokens // tokens_per_step) * tokens_per_step
            print0(f"Adjusted split_tokens to {args.split_tokens} (divisible by {tokens_per_step})")
        
        steps = args.split_tokens // tokens_per_step
        
        for split_name in ["train", "val"]:
            loader = tokenizing_distributed_data_loader_bos_bestfit(
                tokenizer,
                args.device_batch_size,
                sequence_len,
                split_name,
                device=device,
            )
            
            if is_hf_model:
                # HF model BPB evaluation
                # TODO: Implement HF-compatible BPB evaluation
                print0(f"{split_name} BPB: N/A (HF model)")
            else:
                # NanoSeek BPB evaluation
                bpb = evaluate_nanoseek_bpb(model, loader, steps, token_bytes)
                bpb_results[split_name] = bpb
                print0(f"{split_name} BPB: {bpb:.6f}")
        
        # MoE health check (run on a small batch from val)
        if not is_hf_model and config and config.moe.n_routed_experts > 0:
            try:
                # Get a small batch for MoE health check
                val_loader = tokenizing_distributed_data_loader_bos_bestfit(
                    tokenizer, 4, sequence_len, "val", device=device
                )
                batch_x, _ = next(iter(val_loader))
                moe_health = evaluate_moe_health(model, batch_x)
                print0(f"\nMoE Health:")
                print0(f"  Aux loss: {moe_health['aux_loss']:.6f}")
                print0(f"  Load entropy: {moe_health['load_entropy']:.4f}")
                print0(f"  Load CV: {moe_health['load_cv']:.4f}")
            except Exception as e:
                print0(f"Warning: Could not compute MoE health: {e}")
    
    # --- CORE Evaluation ---
    if 'core' in eval_modes:
        print0("\n" + "="*80)
        print0("CORE Evaluation")
        print0("="*80)
        
        if is_hf_model:
            # HF model CORE evaluation
            from nanochat.checkpoint_manager import load_model as load_hf_for_core
            # Note: This would need a wrapper similar to nanochat's
            print0("Warning: CORE evaluation for HF models requires additional setup")
        else:
            core_results = evaluate_core(
                model,
                tokenizer,
                device,
                max_per_task=args.max_per_task,
            )
            
            # Write CSV output
            if ddp_rank == 0:
                base_dir = get_base_dir()
                output_dir = os.path.join(base_dir, "nanoseek_eval")
                os.makedirs(output_dir, exist_ok=True)
                output_csv_path = os.path.join(output_dir, f"{model_slug}.csv")
                
                with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
                    f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
                    for label in core_results["results"]:
                        acc = core_results["results"][label]
                        centered = core_results["centered_results"][label]
                        f.write(f"{label:<35}, {acc:<10.6f}, {centered:<10.6f}\n")
                    f.write(f"{'CORE':<35}, {'':<10}, {core_results['core_metric']:<10.6f}\n")
                
                print0(f"\nResults written to: {output_csv_path}")
                print0(f"CORE metric: {core_results['core_metric']:.4f}")
    

    # --- Domain BPB Evaluation ---
    if 'domain_bpb' in eval_modes and not is_hf_model:
        print0("
" + "="*80)
        print0("Domain BPB Evaluation")
        print0("="*80)
        try:
            from nanoseek.eval.domain_bpb import compute_domain_bpb
            domain_bpb_results = compute_domain_bpb(model, tokenizer, device)
            print0("Per-domain BPB:")
            for domain, bpb_val in domain_bpb_results.items():
                print0(f"  {domain}: {bpb_val:.4f}")
        except Exception as e:
            print0(f"Warning: Domain BPB evaluation failed: {e}")
    
    # --- I_spec Evaluation ---
    if 'i_spec' in eval_modes and not is_hf_model:
        print0("
" + "="*80)
        print0("I_spec (Expert Specialization MI) Evaluation")
        print0("="*80)
        try:
            from nanoseek.eval.information_metrics import compute_i_spec
            val_loader = tokenizing_distributed_data_loader_bos_bestfit(
                tokenizer, 4, sequence_len, "val", device=device
            )
            i_spec_results = compute_i_spec(
                model, val_loader, device,
                n_experts=config.moe.n_routed_experts if config else 64,
            )
            print0(f"I_spec mean: {i_spec_results['i_spec_mean']:.4f} (healthy: 0.3-0.7)")
            print0(f"H(expert): {i_spec_results['h_expert']:.4f}")
        except Exception as e:
            print0(f"Warning: I_spec evaluation failed: {e}")
    
    # --- MTP Acceptance Rate ---
    if 'mtp_accept' in eval_modes and not is_hf_model:
        print0("
" + "="*80)
        print0("MTP Acceptance Rate Evaluation")
        print0("="*80)
        try:
            from nanoseek.eval.moe_diagnostics import compute_mtp_acceptance_rate
            val_loader = tokenizing_distributed_data_loader_bos_bestfit(
                tokenizer, 4, sequence_len, "val", device=device
            )
            mtp_results = compute_mtp_acceptance_rate(model, val_loader, device)
            print0(f"MTP acceptance rate: {mtp_results['acceptance_rate']:.2%} (target: >75%)")
            if mtp_results.get('per_position_rates'):
                for pos, rate in mtp_results['per_position_rates'].items():
                    print0(f"  Position {pos}: {rate:.2%}")
        except Exception as e:
            print0(f"Warning: MTP acceptance evaluation failed: {e}")
    
    # --- Enhanced MoE Diagnostics ---
    if 'moe_diag' in eval_modes and not is_hf_model:
        print0("
" + "="*80)
        print0("Enhanced MoE Diagnostics")
        print0("="*80)
        try:
            from nanoseek.eval.moe_diagnostics import compute_dead_experts, get_expert_load_heatmap
            val_loader = tokenizing_distributed_data_loader_bos_bestfit(
                tokenizer, 4, sequence_len, "val", device=device
            )
            dead_results = compute_dead_experts(model, val_loader, device)
            print0(f"Dead experts (total): {dead_results['total_dead_count']}")
            for layer_idx, gini in dead_results.get('utilization_gini_per_layer', {}).items():
                print0(f"  Layer {layer_idx} Gini: {gini:.4f}")
        except Exception as e:
            print0(f"Warning: Enhanced MoE diagnostics failed: {e}")
    
    # --- Summary ---
    print0("\n" + "="*80)
    print0("Evaluation Summary")
    print0("="*80)
    print0(f"Model: {model_name}")
    
    if bpb_results:
        print0(f"\nBPB Results:")
        for split, bpb in bpb_results.items():
            print0(f"  {split}: {bpb:.6f}")
    
    if moe_health:
        print0(f"\nMoE Health:")
        print0(f"  Aux loss: {moe_health['aux_loss']:.6f}")
        print0(f"  Load entropy: {moe_health['load_entropy']:.4f}")
        print0(f"  Load CV: {moe_health['load_cv']:.4f}")
    
    if core_results:
        print0(f"\nCORE Metric: {core_results['core_metric']:.4f}")
    
    if samples:
        print0(f"\nGenerated {len(samples['conditioned'])} conditioned samples")
        print0(f"Generated {len(samples['unconditioned'])} unconditioned samples")
    
    # Cleanup
    compute_cleanup()


if __name__ == "__main__":
    main()
