"""
CLI entry point for GRPO post-training.

Usage:
    python -m nanoseek.training_ops.run_grpo \
        --model-path checkpoints/nanoseek-1b-sft/ \
        --dataset gsm8k \
        --reward-type math \
        --group-size 4 \
        --kl-beta 0.01 \
        --clip-eps 0.2 \
        --lr 1e-6 \
        --steps 1000 \
        --eval-every 100 \
        --freeze-router
"""

import argparse
import logging
import os
import sys

import torch

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="NanoSeek GRPO Post-Training")

    # Model
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to SFT checkpoint (.pt file or directory)")

    # Data
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "math"],
                        help="Training dataset")
    parser.add_argument("--num-prompts", type=int, default=2000,
                        help="Number of training prompts")
    parser.add_argument("--max-prompt-len", type=int, default=256,
                        help="Maximum prompt token length")

    # GRPO
    parser.add_argument("--reward-type", type=str, default="math",
                        choices=["math", "code"],
                        help="Reward function type")
    parser.add_argument("--group-size", type=int, default=4,
                        help="Completions per prompt (G)")
    parser.add_argument("--kl-beta", type=float, default=0.01,
                        help="KL penalty coefficient")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="PPO clipping epsilon")
    parser.add_argument("--off-policy-delta", type=float, default=0.3,
                        help="Off-policy masking threshold")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for generation")
    parser.add_argument("--max-gen-len", type=int, default=512,
                        help="Maximum generation length")

    # Optimization
    parser.add_argument("--lr", type=float, default=1e-6,
                        help="Learning rate")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Total training steps")
    parser.add_argument("--warmup-steps", type=int, default=20,
                        help="LR warmup steps")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Prompts per training step")
    parser.add_argument("--gradient-accumulation", type=int, default=1,
                        help="Gradient accumulation steps")

    # MoE stabilization
    parser.add_argument("--freeze-router", action="store_true", default=True,
                        help="Freeze router params during RL (Keep Routing)")
    parser.add_argument("--no-freeze-router", action="store_false", dest="freeze_router",
                        help="Don't freeze router params")

    # Eval & checkpointing
    parser.add_argument("--eval-every", type=int, default=100,
                        help="Evaluate and checkpoint every N steps")
    parser.add_argument("--save-dir", type=str, default="checkpoints/nanoseek_grpo",
                        help="Checkpoint output directory")

    # SFT warmup
    parser.add_argument("--sft-warmup", action="store_true",
                        help="Run SFT warmup before GRPO")
    parser.add_argument("--sft-epochs", type=int, default=2,
                        help="SFT warmup epochs")
    parser.add_argument("--sft-lr", type=float, default=1e-5,
                        help="SFT warmup learning rate")

    # Device
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto, cpu, cuda, mps)")

    return parser.parse_args()


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _load_gsm8k_prompts(tokenizer, num_prompts: int, max_len: int):
    """Load GSM8K prompts and ground truths."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="train")
        data = [{"question": ex["question"], "answer": ex["answer"]} for ex in ds]
    except ImportError:
        logger.warning("HuggingFace datasets unavailable, using synthetic prompts")
        data = []
        for i in range(100):
            a, b = i + 1, i + 5
            data.append({
                "question": f"What is {a} + {b}?",
                "answer": f"{a} + {b} = {a + b}\n#### {a + b}",
            })

    data = data[:num_prompts]

    prompts = []
    ground_truths = []
    for item in data:
        prompt_text = f"Question: {item['question']}\n\nAnswer: "
        tokens = tokenizer.encode(prompt_text)
        if len(tokens) <= max_len:
            prompts.append(tokens)
            # Extract final answer from GSM8K format
            parts = item['answer'].split('####')
            gt = parts[-1].strip() if len(parts) >= 2 else item['answer'].strip()
            ground_truths.append(gt)

    logger.info(f"Loaded {len(prompts)} prompts (max_len={max_len})")
    return prompts, ground_truths


def _batch_prompts(prompts: list[list[int]], batch_size: int, device: torch.device):
    """Yield batches of padded prompt tensors."""
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        if len(batch) < batch_size:
            continue  # skip incomplete final batch
        max_len = max(len(p) for p in batch)
        padded = torch.zeros(len(batch), max_len, dtype=torch.long)
        for j, p in enumerate(batch):
            padded[j, :len(p)] = torch.tensor(p, dtype=torch.long)
        yield padded.to(device)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    args = parse_args()
    device = _resolve_device(args.device)
    logger.info(f"Device: {device}")

    # Add parent dirs to path for imports
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    # Load model
    from nanoseek.model import create_nanoseek
    from nanoseek.config import get_nanoseek_config

    config = get_nanoseek_config()
    model = create_nanoseek(config)

    # Load checkpoint
    if os.path.isfile(args.model_path):
        state_dict = torch.load(args.model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded model from {args.model_path}")
    elif os.path.isdir(args.model_path):
        ckpt_path = os.path.join(args.model_path, "sft_final.pt")
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict)
            logger.info(f"Loaded model from {ckpt_path}")
        else:
            logger.warning(f"No checkpoint found at {args.model_path}, using random init")

    model = model.to(device)

    # Tokenizer (use the model's tokenizer or a simple one)
    try:
        from nanoseek.common import get_tokenizer
        tokenizer = get_tokenizer()
    except (ImportError, AttributeError):
        logger.info("Using tiktoken cl100k_base tokenizer")
        import tiktoken
        tokenizer = tiktoken.get_encoding("cl100k_base")

    # Optional SFT warmup
    if args.sft_warmup:
        from nanoseek.training_ops.sft_warmup import run_sft_warmup, SFTConfig
        sft_config = SFTConfig(
            lr=args.sft_lr,
            epochs=args.sft_epochs,
            save_dir=os.path.join(args.save_dir, "sft"),
        )
        logger.info("Running SFT warmup before GRPO...")
        model = run_sft_warmup(model, tokenizer, device, sft_config)

    # Load prompts
    prompts, ground_truths = _load_gsm8k_prompts(
        tokenizer, args.num_prompts, args.max_prompt_len
    )

    # Build GRPO config
    from nanoseek.training_ops.grpo_trainer import GRPOConfig, GRPOTrainer

    grpo_config = GRPOConfig(
        group_size=args.group_size,
        max_gen_len=args.max_gen_len,
        temperature=args.temperature,
        clip_eps=args.clip_eps,
        kl_beta=args.kl_beta,
        off_policy_delta=args.off_policy_delta,
        lr=args.lr,
        total_steps=args.steps,
        warmup_steps=args.warmup_steps,
        gradient_accumulation=args.gradient_accumulation,
        freeze_router=args.freeze_router,
        reward_type=args.reward_type,
        eval_every=args.eval_every,
        save_dir=args.save_dir,
        prompts_per_step=args.batch_size,
    )

    trainer = GRPOTrainer(model, tokenizer, device, grpo_config)

    # Build prompt iterator that cycles through data
    def prompt_iterator():
        """Yield (prompt_batch, ground_truth_batch) cycling through data."""
        idx = 0
        while True:
            batch_prompts = []
            batch_gts = []
            for _ in range(args.batch_size):
                batch_prompts.append(prompts[idx % len(prompts)])
                batch_gts.append(ground_truths[idx % len(ground_truths)])
                idx += 1

            # Pad to same length
            max_len = max(len(p) for p in batch_prompts)
            padded = torch.zeros(len(batch_prompts), max_len, dtype=torch.long)
            for j, p in enumerate(batch_prompts):
                padded[j, :len(p)] = torch.tensor(p, dtype=torch.long)

            yield padded.to(device), batch_gts

    # Train
    trainer.train(prompt_iterator(), ground_truths)

    logger.info("GRPO training complete!")


if __name__ == "__main__":
    main()
