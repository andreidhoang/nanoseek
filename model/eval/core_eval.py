"""
DCLM CORE benchmark evaluation utilities.

Implements the CORE metric from the DCLM paper (https://arxiv.org/abs/2406.11794)
for evaluating base language models on a suite of in-context learning tasks.

The CORE metric includes:
- Multiple choice tasks (e.g., HellaSwag, WinoGrande)
- Schema tasks (e.g., Copa, StoryCloze)
- Language modeling tasks (e.g., LAMBADA)

Example:
    from model.eval.core_eval import evaluate_task

    accuracy = evaluate_task(model, tokenizer, data, device, task_meta)
    print(f"Task accuracy: {accuracy:.4f}")
"""

import random
from jinja2 import Template
import torch
import torch.distributed as dist
from typing import List, Dict, Any, Optional


# =============================================================================
# Prompt rendering utilities
# =============================================================================

def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """
    Render complete prompts for a multiple choice question.

    Args:
        item: Dict with 'query', 'choices', 'gold' keys
        continuation_delimiter: String separating context from answer
        fewshot_examples: Optional list of few-shot examples

    Returns:
        List of prompts, one for each choice
    """
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(choice=choice, **context) for choice in item['choices']]
    return prompts


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """
    Render complete prompts for a schema question.

    Args:
        item: Dict with 'context_options', 'continuation', 'gold' keys
        continuation_delimiter: String separating context from answer
        fewshot_examples: Optional list of few-shot examples

    Returns:
        List of prompts, one for each context option
    """
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(context=context_option, **context)
               for context_option in item['context_options']]
    return prompts


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """
    Render complete prompt for a language modeling task.

    Args:
        item: Dict with 'context', 'continuation' keys
        continuation_delimiter: String separating context from answer
        fewshot_examples: Optional list of few-shot examples

    Returns:
        List of two prompts: [without_continuation, with_continuation]
    """
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    # Return two prompts: without and with the continuation
    prompt_without = template.render(include_continuation=False, **context)
    prompt_with = template.render(include_continuation=True, **context)
    # Strip to avoid trailing whitespace issues
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]


# =============================================================================
# Token sequence utilities
# =============================================================================

def find_common_length(token_sequences, direction='left'):
    """
    Find the length of the common prefix or suffix across token sequences.

    Args:
        token_sequences: List of token ID lists
        direction: 'left' for prefix, 'right' for suffix

    Returns:
        Length of common prefix/suffix
    """
    min_len = min(len(seq) for seq in token_sequences)
    indices = {
        'left': range(min_len),
        'right': range(-1, -min_len-1, -1)
    }[direction]

    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens, pad_token_id):
    """
    Stack token sequences into a padded tensor.

    Args:
        tokens: List of token ID lists
        pad_token_id: Token ID to use for padding

    Returns:
        Tensor of shape (batch_size, max_seq_len)
    """
    bsz, seq_len = len(tokens), max(len(x) for x in tokens)
    input_ids = torch.full((bsz, seq_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(tokens):
        input_ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return input_ids


def batch_sequences_mc(tokenizer, prompts):
    """Batch sequences for multiple choice task (common prefix)."""
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    answer_start_idx = find_common_length(tokens, direction='left')
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(x) for x in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts):
    """Batch sequences for schema task (common suffix)."""
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    suffix_length = find_common_length(tokens, direction='right')
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts):
    """Batch sequences for language modeling task."""
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    assert start_idx < end_idx, "prompt without should be prefix of prompt with"
    assert tokens_without == tokens_with[:start_idx], "prompt without should be prefix of prompt with"
    # Batch size of 1 for LM tasks
    return [tokens_with], [start_idx], [end_idx]


# =============================================================================
# Model forward and evaluation
# =============================================================================

@torch.no_grad()
def forward_model(model, input_ids):
    """
    Forward pass through model, returning losses and predictions.

    Args:
        model: NanoSeek model
        input_ids: Tensor of shape (batch_size, seq_len)

    Returns:
        losses: Tensor of shape (batch_size, seq_len) with NaN in last column
        predictions: Tensor of shape (batch_size, seq_len) with argmax predictions
    """
    batch_size, seq_len = input_ids.size()
    outputs = model(input_ids=input_ids)
    logits = outputs["logits"]

    # Roll tensor left to get autoregressive targets
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)

    # Calculate cross entropy at all positions
    losses = torch.nn.functional.cross_entropy(
        logits.view(batch_size * seq_len, -1),
        target_ids.view(batch_size * seq_len),
        reduction='none'
    ).view(batch_size, seq_len)

    # Last column has no target, set to NaN
    losses[:, -1] = float('nan')

    # Get argmax predictions
    predictions = logits.argmax(dim=-1)

    return losses, predictions


@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta):
    """
    Evaluate a single example.

    Args:
        idx: Example index in data
        model: NanoSeek model
        tokenizer: Tokenizer instance
        data: List of examples
        device: torch.device
        task_meta: Dict with task configuration

    Returns:
        True if correct, False otherwise
    """
    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']

    # Sample few-shot examples (excluding current item)
    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, min(num_fewshot, len(available_indices)))
        fewshot_examples = [data[i] for i in fewshot_indices]

    # Render prompts and batch sequences based on task type
    if task_type == 'multiple_choice':
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_type == 'schema':
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_type == 'language_modeling':
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # Handle max sequence length truncation
    if hasattr(model, 'max_seq_len') and model.max_seq_len is not None:
        max_tokens = model.max_seq_len
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_tokens:
                num_to_crop = len(t) - max_tokens
                new_tokens.append(t[-max_tokens:])
                new_start_idxs.append(max(0, s - num_to_crop))
                new_end_idxs.append(e - num_to_crop)
            else:
                new_tokens.append(t)
                new_start_idxs.append(s)
                new_end_idxs.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

    # Stack sequences into batch
    pad_token_id = tokenizer.get_bos_token_id()  # Use BOS as pad token
    input_ids = stack_sequences(tokens, pad_token_id)
    input_ids = input_ids.to(device)

    # Forward model
    losses, predictions = forward_model(model, input_ids)

    # Evaluate based on task type
    if task_type == 'language_modeling':
        # LM task: check if predictions match targets
        si, ei = start_idxs[0], end_idxs[0]
        predicted_tokens = predictions[0, si-1:ei-1]
        actual_tokens = input_ids[0, si:ei]
        is_correct = torch.all(predicted_tokens == actual_tokens).item()
    elif task_type in ['multiple_choice', 'schema']:
        # MC/Schema: find option with lowest average loss
        mean_losses = [losses[i, si-1:ei-1].mean().item()
                       for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))]
        pred_idx = mean_losses.index(min(mean_losses))
        is_correct = pred_idx == item['gold']
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    return is_correct


def evaluate_task(model, tokenizer, data, device, task_meta):
    """
    Evaluate a task across all examples with DDP support.

    Args:
        model: NanoSeek model
        tokenizer: Tokenizer instance
        data: List of examples
        device: torch.device
        task_meta: Dict with task configuration

    Returns:
        Mean accuracy across all examples
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    correct = torch.zeros(len(data), dtype=torch.float32, device=device)

    # Stride examples across ranks
    for idx in range(rank, len(data), world_size):
        is_correct = evaluate_example(idx, model, tokenizer, data, device, task_meta)
        correct[idx] = float(is_correct)

    # Sync results across ranks
    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)

    # Compute mean accuracy
    mean_correct = correct.mean().item()
    return mean_correct
