"""
Chat-style evaluation script for NanoSeek using nanochat's Task system.

Supports evaluation on:
  - ARC-Easy / ARC-Challenge (science QA)
  - MMLU (57 subjects)
  - GSM8K (grade school math)
  - HumanEval (code generation)
  - Custom JSON tasks

Unlike base_eval.py (which uses few-shot ICL from eval_bundle),
this script uses conversation-style evaluation with the Task class.

Examples:

    # Evaluate on ARC-Challenge
    python -m nanoseek.scripts.chat_eval --scale 1b --step 5000 --task arc --subset ARC-Challenge

    # Evaluate on MMLU (all test subjects)
    python -m nanoseek.scripts.chat_eval --scale 1b --task mmlu --subset all --split test

    # Evaluate on GSM8K
    torchrun --nproc_per_node=8 -m nanoseek.scripts.chat_eval --scale 1b --task gsm8k

    # Multi-task evaluation
    python -m nanoseek.scripts.chat_eval --scale 1b --task arc,mmlu,gsm8k

    # With custom checkpoint
    python -m nanoseek.scripts.chat_eval \
        --checkpoint-dir checkpoints/nanoseek_1b \
        --step 10000 --use-ema --task arc --subset ARC-Challenge
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm

# NanoSeek imports
from nanoseek.nanoseek.config import (
    NanoSeekConfig,
    get_nanoseek_config,
    get_nanoseek_500m_config,
    get_nanoseek_anchor_config,
)
from nanoseek.nanoseek.model import NanoSeekModel
from nanoseek.nanoseek.common import (
    compute_init, compute_cleanup, print0,
    autodetect_device_type, is_ddp_initialized,
)

# Nanochat imports
from nanochat.tokenizer import get_tokenizer
from nanochat.tasks.common import Task, TaskMixture
from nanochat.tasks.arc import ARC
from nanochat.tasks.mmlu import MMLU
from nanochat.tasks.gsm8k import GSM8K
from nanochat.tasks.humaneval import HumanEval
from nanochat.tasks.customjson import CustomJSON


# -----------------------------------------------------------------------------
# Model Loading (same as base_eval.py)
# -----------------------------------------------------------------------------

def load_nanoseek_model(
    scale: str,
    device: torch.device,
    checkpoint_dir: Optional[str] = None,
    step: Optional[int] = None,
    use_ema: bool = False,
) -> Tuple[NanoSeekModel, NanoSeekConfig, Any, Optional[Dict]]:
    """Load a NanoSeek model with optional EMA weights."""
    config_map = {
        "anchor": get_nanoseek_anchor_config,
        "500m": get_nanoseek_500m_config,
        "1b": get_nanoseek_config,
    }
    if scale not in config_map:
        raise ValueError(f"Unknown scale: {scale}")
    
    config = config_map[scale]()
    print0(f"Loading NanoSeek-{scale} config:")
    print0(f"  Hidden size: {config.hidden_size}")
    print0(f"  Layers: {config.num_layers}")
    
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
    if os.path.exists(checkpoint_dir):
        model_files = list(Path(checkpoint_dir).glob("model_*.pt"))
        
        if model_files:
            if step is not None:
                checkpoint_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
            else:
                latest = max(model_files, key=lambda p: int(p.stem.split('_')[1]))
                checkpoint_path = str(latest)
                step = int(latest.stem.split('_')[1])
            
            if os.path.exists(checkpoint_path):
                print0(f"Loading checkpoint: {checkpoint_path}")
                state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
                model.load_state_dict(state_dict)
                
                if use_ema:
                    ema_path = os.path.join(checkpoint_dir, f"ema_{step:06d}.pt")
                    if os.path.exists(ema_path):
                        print0(f"Loading EMA weights: {ema_path}")
                        ema_state = torch.load(ema_path, map_location=device, weights_only=True)
                        for name, param in model.named_parameters():
                            if name in ema_state:
                                param.data.copy_(ema_state[name])
    
    tokenizer = get_tokenizer()
    model.eval()
    
    return model, config, tokenizer, ema_state


# -----------------------------------------------------------------------------
# Task Loading
# -----------------------------------------------------------------------------

TASK_REGISTRY = {
    "arc": ARC,
    "mmlu": MMLU,
    "gsm8k": GSM8K,
    "humaneval": HumanEval,
    "custom": CustomJSON,
}


def load_task(task_name: str, subset: Optional[str] = None, split: str = "test", **kwargs) -> Task:
    """
    Load a task by name.
    
    Args:
        task_name: One of arc, mmlu, gsm8k, humaneval, custom
        subset: Task subset (e.g., "ARC-Challenge", "all")
        split: Dataset split (train/validation/test/dev)
        **kwargs: Additional arguments for task constructor
        
    Returns:
        Task instance
    """
    task_name = task_name.lower()
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_REGISTRY.keys())}")
    
    task_class = TASK_REGISTRY[task_name]
    
    # Handle task-specific defaults
    if task_name == "arc":
        subset = subset or "ARC-Challenge"
        return task_class(subset=subset, split=split, **kwargs)
    
    elif task_name == "mmlu":
        subset = subset or "all"
        return task_class(subset=subset, split=split, **kwargs)
    
    elif task_name == "gsm8k":
        return task_class(split=split, **kwargs)
    
    elif task_name == "humaneval":
        return task_class(split=split, **kwargs)
    
    elif task_name == "custom":
        if "path" not in kwargs:
            raise ValueError("Custom task requires 'path' argument")
        return task_class(split=split, **kwargs)
    
    else:
        return task_class(split=split, **kwargs)


# -----------------------------------------------------------------------------
# Chat Evaluation
# -----------------------------------------------------------------------------

@torch.no_grad()
def evaluate_chat_task(
    model: NanoSeekModel,
    tokenizer: Any,
    task: Task,
    device: torch.device,
    max_samples: int = -1,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """
    Evaluate a model on a chat-style task.
    
    Args:
        model: NanoSeek model (eval mode)
        tokenizer: Tokenizer
        task: Task instance (ARC, MMLU, etc.)
        device: Device to run on
        max_samples: Max samples to evaluate (-1 = all)
        temperature: Sampling temperature (0 = greedy)
        max_new_tokens: Max tokens to generate
        
    Returns:
        Dict with accuracy and per-sample results
    """
    model.eval()
    
    # Get number of examples
    num_examples = len(task)
    if max_samples > 0:
        num_examples = min(num_examples, max_samples)
    
    print0(f"Evaluating on {num_examples} examples...")
    
    correct = 0
    total = 0
    results = []
    
    # Progress bar on rank 0
    iterator = range(num_examples)
    if int(os.environ.get('RANK', 0)) == 0:
        iterator = tqdm(iterator, desc=f"Evaluating {task.__class__.__name__}")
    
    for idx in iterator:
        # Get conversation
        conversation = task[idx]
        messages = conversation["messages"]
        
        # Format as chat prompt
        # For base models, we format as: "User: {question}\nAssistant: {answer}"
        prompt = format_chat_prompt(messages[:-1])  # All except last (assistant response)
        expected = messages[-1]["content"]  # Expected response
        
        # Tokenize
        input_ids = tokenizer.encode(prompt, bos=True, eos=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        # Generate
        generated_ids = generate_greedy(model, input_tensor, max_new_tokens=max_new_tokens)
        
        # Decode
        generated_text = tokenizer.decode(generated_ids[0].tolist())
        
        # Extract answer (task-specific)
        predicted = extract_answer(task, generated_text, conversation)
        
        # Evaluate
        is_correct = task.evaluate(conversation, predicted)
        if is_correct:
            correct += 1
        total += 1
        
        # Store result
        results.append({
            "idx": idx,
            "prompt": prompt,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
        })
        
        # Print some examples
        if idx < 3 and int(os.environ.get('RANK', 0)) == 0:
            print0(f"\n--- Example {idx} ---")
            print0(f"Prompt: {prompt[:200]}...")
            print0(f"Expected: {expected}")
            print0(f"Predicted: {predicted}")
            print0(f"Correct: {is_correct}")
    
    # Aggregate results across ranks if using DDP
    if is_ddp_initialized():
        correct_tensor = torch.tensor(correct, dtype=torch.int32, device=device)
        total_tensor = torch.tensor(total, dtype=torch.int32, device=device)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        correct = correct_tensor.item()
        total = total_tensor.item()
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


def format_chat_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Format messages into a prompt string.
    
    Args:
        messages: List of {"role": "user"|"assistant", "content": str}
        
    Returns:
        Formatted prompt string
    """
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    parts.append("Assistant:")  # Trigger assistant response
    return "\n\n".join(parts)


@torch.no_grad()
def generate_greedy(
    model: NanoSeekModel,
    input_ids: torch.Tensor,
    max_new_tokens: int = 512,
    stop_tokens: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Greedy generation for evaluation.
    
    Args:
        model: NanoSeek model
        input_ids: [B, S] input token IDs
        max_new_tokens: Max tokens to generate
        stop_tokens: List of token IDs to stop at
        
    Returns:
        [B, S+T] generated token IDs
    """
    if stop_tokens is None:
        stop_tokens = []
    
    generated = input_ids.clone()
    
    for _ in range(max_new_tokens):
        outputs = model(generated, use_cache=True)
        logits = outputs["logits"]
        
        # Get next token (greedy)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        
        # Append
        generated = torch.cat([generated, next_token], dim=-1)
        
        # Check for stop tokens
        if next_token.item() in stop_tokens:
            break
    
    return generated


def extract_answer(task: Task, generated_text: str, conversation: Dict) -> str:
    """
    Extract answer from generated text based on task type.
    
    Args:
        task: Task instance
        generated_text: Full generated text
        conversation: Original conversation
        
    Returns:
        Extracted answer
    """
    # Get the part after "Assistant:"
    if "Assistant:" in generated_text:
        answer_part = generated_text.split("Assistant:")[-1].strip()
    else:
        answer_part = generated_text.strip()
    
    # Task-specific extraction
    task_name = task.__class__.__name__
    
    if task_name == "ARC" or task_name == "MMLU":
        # Multiple choice: extract first letter A/B/C/D
        letters = conversation.get("letters", ["A", "B", "C", "D"])
        for char in answer_part.upper():
            if char in letters:
                return char
        return answer_part[:1].upper()  # Fallback to first char
    
    elif task_name == "GSM8K":
        # Math: extract last number
        import re
        numbers = re.findall(r'-?\d+\.?\d*', answer_part)
        if numbers:
            return numbers[-1]
        return answer_part
    
    elif task_name == "HumanEval":
        # Code: return full generated code
        return answer_part
    
    else:
        # Default: return first line
        return answer_part.split('\n')[0].strip()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NanoSeek chat-style evaluation")
    
    # Task selection
    parser.add_argument(
        '--task', type=str, required=True,
        help='Comma-separated task names: arc,mmlu,gsm8k,humaneval,custom'
    )
    parser.add_argument(
        '--subset', type=str, default=None,
        help='Task subset (e.g., ARC-Challenge, ARC-Easy, all)'
    )
    parser.add_argument(
        '--split', type=str, default='test',
        help='Dataset split (train/validation/test/dev)'
    )
    parser.add_argument(
        '--max-samples', type=int, default=-1,
        help='Max samples per task (-1 = all)'
    )
    
    # Model settings
    parser.add_argument(
        '--scale', type=str, default='1b',
        choices=['anchor', '500m', '1b'],
        help='NanoSeek model scale'
    )
    parser.add_argument(
        '--checkpoint-dir', type=str, default=None,
        help='Checkpoint directory'
    )
    parser.add_argument(
        '--step', type=int, default=None,
        help='Model step to load'
    )
    parser.add_argument(
        '--use-ema', action='store_true',
        help='Use EMA weights'
    )
    
    # Generation settings
    parser.add_argument(
        '--temperature', type=float, default=0.0,
        help='Sampling temperature (0 = greedy)'
    )
    parser.add_argument(
        '--max-new-tokens', type=int, default=512,
        help='Max tokens to generate'
    )
    
    # Output
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSON file for results'
    )
    
    # Device
    parser.add_argument(
        '--device-type', type=str, default='',
        help='cuda|cpu|mps (empty = autodetect)'
    )
    
    args = parser.parse_args()
    
    # Parse tasks
    task_names = [t.strip() for t in args.task.split(',')]
    
    # Distributed setup
    device_type = autodetect_device_type() if args.device_type == '' else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    
    # Load model
    model, config, tokenizer, ema_state = load_nanoseek_model(
        scale=args.scale,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        step=args.step,
        use_ema=args.use_ema,
    )
    
    model_name = f"NanoSeek-{args.scale}"
    if args.step is not None:
        model_name += f" (step {args.step})"
    if args.use_ema:
        model_name += " [EMA]"
    
    print0(f"\nEvaluating model: {model_name}")
    print0(f"Tasks: {', '.join(task_names)}")
    print0(f"Split: {args.split}")
    
    # Evaluate each task
    all_results = {}
    
    for task_name in task_names:
        print0(f"\n{'='*60}")
        print0(f"Task: {task_name.upper()}")
        print0(f"{'='*60}")
        
        try:
            # Load task
            task = load_task(task_name, subset=args.subset, split=args.split)
            print0(f"Loaded {len(task)} examples")
            
            # Evaluate
            results = evaluate_chat_task(
                model=model,
                tokenizer=tokenizer,
                task=task,
                device=device,
                max_samples=args.max_samples,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )
            
            all_results[task_name] = results
            
            print0(f"\nResults for {task_name}:")
            print0(f"  Accuracy: {results['accuracy']:.4f}")
            print0(f"  Correct: {results['correct']}/{results['total']}")
            
        except Exception as e:
            print0(f"Error evaluating {task_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print0(f"\n{'='*60}")
    print0("SUMMARY")
    print0(f"{'='*60}")
    
    for task_name, results in all_results.items():
        print0(f"{task_name:15s}: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    
    # Save results
    if args.output and ddp_rank == 0:
        output_data = {
            "model": model_name,
            "tasks": all_results,
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print0(f"\nResults saved to: {args.output}")
    
    compute_cleanup()


if __name__ == "__main__":
    main()
