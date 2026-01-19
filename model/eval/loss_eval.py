"""
Loss evaluation utilities for base model evaluation.

Provides bits-per-byte (BPB) evaluation which is a tokenization-independent
metric for comparing language models. Unlike mean loss, BPB normalizes
by the number of bytes in the target tokens, making it comparable across
different tokenizers and vocabulary sizes.

Example:
    from model.eval.loss_eval import evaluate_bpb, get_token_bytes

    token_bytes = get_token_bytes(tokenizer, device)
    bpb = evaluate_bpb(model, data_loader, steps=100, token_bytes=token_bytes)
    print(f"Validation BPB: {bpb:.4f}")
"""
import math
import torch
import torch.distributed as dist
from typing import Iterator, Tuple


@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    """
    Calculate bits-per-byte (BPB) metric for a language model.

    BPB is a tokenization-independent metric that normalizes loss by the
    number of bytes represented by target tokens. This allows fair comparison
    across models with different tokenizers/vocabularies.

    How it works:
    1) All "normal" tokens are normalized by the length of the token in bytes
    2) No special tokens (e.g. <|bos|>) are included in the metric - they are masked out.
    3) No actively masked tokens (using ignore_index of e.g. -1) are included in the metric.

    Args:
        model: NanoSeek model with forward returning dict with "loss" key
        batches: Iterator yielding (input_ids, targets) tuples
        steps: Number of batches to evaluate
        token_bytes: 1D tensor of shape (vocab_size,), indicating the number of bytes for each token id, or 0 if the token is to not be counted (e.g. special tokens)

    Returns:
        Bits per byte (lower is better)
    """
    device = next(model.parameters()).device
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)

    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)

        # Forward pass with per-token loss (loss_reduction='none')
        outputs = model(input_ids=x, targets=y, loss_reduction='none')
        loss2d = outputs["loss"]  # (B, T) per-token loss
        loss2d = loss2d.view(-1)  # Flatten
        y = y.view(-1)  # Flatten targets

        # Handle ignore_index tokens (targets < 0)
        # Note: MPS doesn't have kernel for < 0 on int64, only int32
        if (y.int() < 0).any():
            # Complex path: some tokens are masked
            valid = y >= 0
            y_safe = torch.where(valid, y, torch.zeros_like(y))

            # Map valid targets to byte length; ignored targets → 0 bytes
            num_bytes2d = torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y, dtype=token_bytes.dtype)
            )
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
        else:
            # Fast path: no ignored targets, safe to index directly
            num_bytes2d = token_bytes[y]
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()

    # Sum reduce across all ranks
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)

    # Calculate BPB
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()

    if total_bytes == 0:
        return float('inf')

    # Convert nats to bits: bits = nats / ln(2)
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb


def get_token_bytes(tokenizer, device: torch.device) -> torch.Tensor:
    """
    Get byte lengths for all tokens in the vocabulary.

    Returns a tensor of shape (vocab_size,) where each element is the
    number of bytes for that token ID. Special tokens have 0 bytes.

    Args:
        tokenizer: Tokenizer with decode() and get_vocab_size() methods
        device: Device to place the tensor on

    Returns:
        Tensor of byte lengths per token
    """
    vocab_size = tokenizer.get_vocab_size()
    token_bytes = torch.zeros(vocab_size, dtype=torch.int64, device=device)

    for token_id in range(vocab_size):
        try:
            # Decode the token and get its UTF-8 byte length
            token_str = tokenizer.decode([token_id])
            token_bytes[token_id] = len(token_str.encode('utf-8'))
        except Exception:
            # Special tokens or invalid tokens get 0 bytes
            token_bytes[token_id] = 0

    return token_bytes


@torch.no_grad()
def evaluate_loss(model, batches, steps) -> float:
    """
    Simple mean loss evaluation (cross-entropy).

    Args:
        model: NanoSeek model
        batches: Iterator yielding (input_ids, targets) tuples
        steps: Number of batches to evaluate

    Returns:
        Mean cross-entropy loss
    """
    device = next(model.parameters()).device
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_tokens = torch.tensor(0, dtype=torch.int64, device=device)

    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)

        outputs = model(input_ids=x, targets=y)
        loss = outputs["loss"]

        # Count valid tokens (not masked)
        valid_tokens = (y >= 0).sum()

        total_loss += loss * valid_tokens
        total_tokens += valid_tokens

    # Sum reduce across ranks
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

    if total_tokens.item() == 0:
        return float('inf')

    return (total_loss / total_tokens).item()
