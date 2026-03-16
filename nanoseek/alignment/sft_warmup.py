"""
Cold-start SFT warmup on GSM8K before GRPO.

Purpose: Teach the model the <think>...</think> output format before RL,
preventing reward hacking on format during early RL steps.

Process:
1. Download/load GSM8K train split
2. Reformat into chat format with reasoning traces
3. Standard cross-entropy fine-tuning (2-3 epochs, lr=1e-5)
4. Save SFT checkpoint as GRPO starting point
"""

import os
import re
import json
import math
import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    """Configuration for SFT warmup."""
    lr: float = 1e-5
    epochs: int = 2
    batch_size: int = 4
    max_length: int = 1024
    warmup_steps: int = 50
    weight_decay: float = 0.01
    gradient_accumulation: int = 4
    num_examples: int = 2000
    save_dir: str = "checkpoints/nanoseek_sft"


def format_gsm8k_example(question: str, answer: str) -> str:
    """
    Reformat a GSM8K example into chat format with <think> traces.

    Input answer format: "Step 1. ...\nStep 2. ...\n#### 42"
    Output format: "<think>Step 1. ...\nStep 2. ...</think>\n\nThe answer is \\boxed{42}"
    """
    # Split answer into reasoning and final answer
    parts = answer.split('####')
    if len(parts) == 2:
        reasoning = parts[0].strip()
        final_answer = parts[1].strip()
    else:
        reasoning = answer.strip()
        final_answer = ""

    formatted = f"Question: {question}\n\n<think>\n{reasoning}\n</think>\n\nThe answer is \\boxed{{{final_answer}}}"
    return formatted


class GSM8KSFTDataset(Dataset):
    """Dataset of formatted GSM8K examples for SFT."""

    def __init__(self, tokenizer, max_length: int = 1024, num_examples: int = 2000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        # Load GSM8K from HuggingFace datasets or local file
        data = self._load_gsm8k(num_examples)

        for item in data:
            formatted = format_gsm8k_example(item['question'], item['answer'])
            tokens = tokenizer.encode(formatted)
            if len(tokens) <= max_length:
                self.examples.append(tokens)

        logger.info(f"Loaded {len(self.examples)} GSM8K SFT examples (max_length={max_length})")

    def _load_gsm8k(self, num_examples: int) -> list[dict]:
        """Load GSM8K dataset."""
        try:
            from datasets import load_dataset
            ds = load_dataset("openai/gsm8k", "main", split="train")
            data = [{"question": ex["question"], "answer": ex["answer"]} for ex in ds]
        except ImportError:
            logger.warning("HuggingFace datasets not available, using synthetic examples")
            data = self._synthetic_examples()

        return data[:num_examples]

    def _synthetic_examples(self) -> list[dict]:
        """Fallback synthetic math examples if GSM8K unavailable."""
        examples = []
        for i in range(100):
            a, b = i + 1, i + 5
            q = f"What is {a} + {b}?"
            ans = f"We need to add {a} and {b}.\n{a} + {b} = {a + b}\n#### {a + b}"
            examples.append({"question": q, "answer": ans})
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]
        # Pad to max_length
        padded = tokens[:self.max_length]
        return torch.tensor(padded, dtype=torch.long)


def collate_sft(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate function that pads and creates input/target pairs."""
    max_len = max(t.size(0) for t in batch)
    input_ids = torch.full((len(batch), max_len), -1, dtype=torch.long)
    for i, t in enumerate(batch):
        input_ids[i, :t.size(0)] = t

    # Input: tokens[:-1], Target: tokens[1:]
    x = input_ids[:, :-1]
    y = input_ids[:, 1:]
    return x, y


@torch.no_grad()
def evaluate_sft(model, dataloader, device, max_steps=50):
    """Quick SFT loss evaluation."""
    model.eval()
    total_loss = 0.0
    n_steps = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-1)
        total_loss += loss.item()
        n_steps += 1
        if n_steps >= max_steps:
            break
    model.train()
    return total_loss / max(n_steps, 1)


def run_sft_warmup(model, tokenizer, device, config: SFTConfig | None = None):
    """
    Run cold-start SFT warmup.

    Args:
        model: NanoSeekModel
        tokenizer: Tokenizer with encode() method
        device: Compute device
        config: SFT configuration

    Returns:
        model: Fine-tuned model
    """
    if config is None:
        config = SFTConfig()

    logger.info(f"Starting SFT warmup: {config.epochs} epochs, lr={config.lr}")

    # Create dataset
    dataset = GSM8KSFTDataset(tokenizer, config.max_length, config.num_examples)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size,
        shuffle=True, collate_fn=collate_sft, drop_last=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr,
        weight_decay=config.weight_decay, betas=(0.9, 0.95),
    )

    # Cosine LR schedule
    total_steps = len(dataloader) * config.epochs // config.gradient_accumulation
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / max(config.warmup_steps, 1)
        progress = (step - config.warmup_steps) / max(total_steps - config.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    step = 0
    accum_loss = 0.0

    for epoch in range(config.epochs):
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=-1,
            )
            loss = loss / config.gradient_accumulation
            loss.backward()
            accum_loss += loss.item()

            if (batch_idx + 1) % config.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % 50 == 0:
                    logger.info(
                        f"SFT step {step}/{total_steps} | "
                        f"loss: {accum_loss:.4f} | "
                        f"lr: {scheduler.get_last_lr()[0]:.2e}"
                    )
                accum_loss = 0.0

        # End of epoch eval
        eval_loss = evaluate_sft(model, dataloader, device)
        logger.info(f"Epoch {epoch + 1}/{config.epochs} | eval loss: {eval_loss:.4f}")

    # Save checkpoint
    os.makedirs(config.save_dir, exist_ok=True)
    save_path = os.path.join(config.save_dir, "sft_final.pt")
    torch.save(model.state_dict(), save_path)
    logger.info(f"SFT checkpoint saved to {save_path}")

    return model
