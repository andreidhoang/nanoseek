"""
Evaluate base model loss/BPB and sample from model.

Loads a checkpoint, and:
- Evaluates the BPB (bits per byte) on train/val splits
- Samples from the model to verify generation quality

Example runs:
    # Single GPU
    python -m scripts.base_loss

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=8 -m scripts.base_loss

    # Custom configuration
    python -m scripts.base_loss --device-batch-size 16 --split-tokens 10485760
"""

import os
import argparse
from contextlib import nullcontext

import torch

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils import (
    get_dist_info,
    compute_init,
    compute_cleanup,
    autodetect_device_type,
    print0,
)
from scripts.dataloader import tokenizing_distributed_data_loader
from model.eval.loss_eval import evaluate_bpb, get_token_bytes
from model.eval.checkpoint_manager import load_model
from inference.engine import Engine


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate base model loss/BPB')
    parser.add_argument('--source', type=str, default='base',
                        choices=['base', 'mid'],
                        help='Model source for local checkpoints')
    parser.add_argument('--model-tag', type=str, default=None,
                        help='Model tag to load')
    parser.add_argument('--step', type=int, default=None,
                        help='Checkpoint step to load')
    parser.add_argument('--device-batch-size', type=int, default=32,
                        help='Batch size per device')
    parser.add_argument('--split-tokens', type=int, default=20*524288,
                        help='Number of tokens to evaluate per split')
    parser.add_argument('--device-type', type=str, default='',
                        choices=['cuda', 'cpu', 'mps', ''],
                        help='Device type (empty = autodetect)')
    args = parser.parse_args()

    # ==========================================================================
    # Initialization
    # ==========================================================================

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )

    # ==========================================================================
    # Load model and tokenizer
    # ==========================================================================

    model, tokenizer, meta = load_model(
        args.source,
        device=device,
        phase="eval",
        model_tag=args.model_tag,
        step=args.step
    )

    sequence_len = meta["model_config"].get("max_seq_len", 2048)
    print0(f"Loaded model: {args.source} (step {meta['step']})")
    print0(f"Sequence length: {sequence_len}")

    # ==========================================================================
    # Evaluate loss on each split
    # ==========================================================================

    tokens_per_step = args.device_batch_size * sequence_len * ddp_world_size
    assert args.split_tokens % tokens_per_step == 0, \
        f"split_tokens ({args.split_tokens}) must be divisible by tokens_per_step ({tokens_per_step})"
    steps = args.split_tokens // tokens_per_step

    token_bytes = get_token_bytes(tokenizer, device=device)
    bpb_results = {}

    for split_name in ["train", "val"]:
        print0(f"Evaluating {split_name} split ({steps} steps)...")

        loader = tokenizing_distributed_data_loader(
            args.device_batch_size,
            sequence_len,
            split_name,
            device=device
        )

        with autocast_ctx:
            bpb = evaluate_bpb(model, loader, steps, token_bytes)

        print0(f"{split_name} BPB: {bpb:.4f}")
        bpb_results[split_name] = bpb

    # ==========================================================================
    # Sample from model (rank 0 only)
    # ==========================================================================

    samples = []
    if ddp_rank == 0:
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]

        engine = Engine(model, tokenizer)
        print0("\nSampling from model:")
        print0("-" * 60)

        for prompt in prompts:
            # Tokenize with BOS
            tokens = tokenizer(prompt, prepend=tokenizer.get_bos_token_id())
            if isinstance(tokens[0], list):
                tokens = tokens[0]

            with autocast_ctx:
                result_tokens, _ = engine.generate_batch(
                    tokens,
                    num_samples=1,
                    max_tokens=16,
                    temperature=0  # Greedy for reproducibility
                )

            sample_str = tokenizer.decode(result_tokens[0])
            print0(sample_str)
            samples.append(sample_str)

        print0("-" * 60)

    # ==========================================================================
    # Log to report
    # ==========================================================================

    if ddp_rank == 0:
        from model.eval.report import get_report
        get_report().log(section="Base model loss", data=[
            {
                "train bpb": bpb_results["train"],
                "val bpb": bpb_results["val"],
            },
            {f"sample {i}": sample for i, sample in enumerate(samples)},
        ])
        print0(f"\nResults logged to report.")

    # ==========================================================================
    # Cleanup
    # ==========================================================================

    compute_cleanup()


if __name__ == "__main__":
    main()
