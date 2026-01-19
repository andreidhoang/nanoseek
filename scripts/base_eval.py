"""
Evaluate the DCLM CORE metric for a base language model.

The CORE metric evaluates base models on a suite of in-context learning tasks
including multiple choice, schema, and language modeling tasks.

Example runs:
    # Single GPU
    python -m scripts.base_eval

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=8 -m scripts.base_eval

    # Evaluate HuggingFace model
    python -m scripts.base_eval --hf-path openai-community/gpt2

    # Quick test with fewer examples
    python -m scripts.base_eval --max-per-task 100
"""

import os
import csv
import time
import json
import yaml
import shutil
import random
import zipfile
import tempfile
import argparse
import urllib.request
import sys
from contextlib import nullcontext
from filelock import FileLock

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.tokenizer import get_tokenizer
from model.eval.checkpoint_manager import load_model, get_base_dir
from model.eval.core_eval import evaluate_task


# =============================================================================
# Utilities
# =============================================================================

def print0(*args, **kwargs):
    """Print only on rank 0."""
    if int(os.environ.get('RANK', 0)) == 0:
        print(*args, **kwargs)


def autodetect_device_type() -> str:
    """Detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dist_info():
    """Get distributed training info."""
    if int(os.environ.get('RANK', -1)) != -1:
        return (
            True,
            int(os.environ['RANK']),
            int(os.environ['LOCAL_RANK']),
            int(os.environ['WORLD_SIZE'])
        )
    return False, 0, 0, 1


def compute_init(device_type: str = "cuda"):
    """Initialize compute environment."""
    import torch.distributed as dist

    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)
        torch.set_float32_matmul_precision("high")

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type)

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device


def compute_cleanup():
    """Clean up distributed environment."""
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


def download_file_with_lock(url, filename, postprocess_fn=None):
    """Download a file with lock to prevent concurrent downloads."""
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path

    with FileLock(lock_path):
        # Recheck after acquiring lock
        if os.path.exists(file_path):
            return file_path

        print0(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read()

        with open(file_path, 'wb') as f:
            f.write(content)
        print0(f"Downloaded to {file_path}")

        if postprocess_fn is not None:
            postprocess_fn(file_path)

    return file_path


# =============================================================================
# Eval bundle management
# =============================================================================

# ~162MB of data needed to evaluate the CORE metric
EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


def place_eval_bundle(file_path):
    """Unzip eval bundle to base directory."""
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)

    print0(f"Placed eval_bundle directory at {eval_bundle_dir}")


def evaluate_model(model, tokenizer, device, max_per_task=-1):
    """
    Evaluate a base model on the CORE benchmark.

    Args:
        model: NanoSeek model
        tokenizer: Tokenizer instance
        device: torch.device
        max_per_task: Max examples per task for testing (-1 = all)

    Returns:
        Dict with results, centered_results, and core_metric
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

    # Load config and task metadata
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']

    # Load random baseline values
    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row['Eval Task']
            random_baseline = row['Random baseline']
            random_baselines[task_name] = float(random_baseline)

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

        # Load task data
        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        # Shuffle data for consistent subsampling
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        # Evaluate task
        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)

        results[label] = accuracy
        random_baseline = random_baselines[label]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result

        end_time = time.time()
        print0(f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {end_time - start_time:.2f}s")

    # Calculate CORE metric
    core_metric = sum(centered_results.values()) / len(centered_results)

    return {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric
    }


# =============================================================================
# HuggingFace model support
# =============================================================================

class ModelWrapper:
    """Lightweight wrapper for HuggingFace models."""

    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids):
        outputs = self.model(input_ids)
        return {"logits": outputs.logits}

    def parameters(self):
        return self.model.parameters()


class HuggingFaceTokenizer:
    """Wrapper to make HF tokenizer compatible with our API."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, path):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(path)
        return cls(tokenizer)

    def __call__(self, texts, prepend=None):
        if isinstance(texts, str):
            texts = [texts]
        results = []
        for text in texts:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if prepend is not None:
                ids = [prepend] + ids
            results.append(ids)
        return results if len(results) > 1 else results[0]

    def encode(self, text, add_special_tokens=False):
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def get_bos_token_id(self):
        return self.tokenizer.bos_token_id or self.tokenizer.eos_token_id

    def get_vocab_size(self):
        return len(self.tokenizer)


def load_hf_model(hf_path: str, device):
    """Load a HuggingFace model."""
    from transformers import AutoModelForCausalLM

    print0(f"Loading model from: {hf_path}")
    model = AutoModelForCausalLM.from_pretrained(hf_path)
    model.to(device)
    model.eval()

    # Set max sequence length for GPT-2
    max_seq_len = 1024 if "gpt2" in hf_path.lower() else None
    model = ModelWrapper(model, max_seq_len=max_seq_len)

    tokenizer = HuggingFaceTokenizer.from_pretrained(hf_path)

    return model, tokenizer


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate base model on CORE benchmark")
    parser.add_argument('--hf-path', type=str, default=None,
                        help='HuggingFace model path to evaluate')
    parser.add_argument('--source', type=str, default='base',
                        choices=['base', 'mid'],
                        help='Model source for local checkpoints')
    parser.add_argument('--model-tag', type=str, default=None,
                        help='Model tag to load')
    parser.add_argument('--step', type=int, default=None,
                        help='Checkpoint step to load')
    parser.add_argument('--max-per-task', type=int, default=-1,
                        help='Max examples per task (-1 = all)')
    parser.add_argument('--device-type', type=str, default='',
                        choices=['cuda', 'cpu', 'mps', ''],
                        help='Device type (empty = autodetect)')
    args = parser.parse_args()

    # Compute setup
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )

    # Load model
    if args.hf_path is not None:
        model, tokenizer = load_hf_model(args.hf_path, device)
        model_name = args.hf_path
        model_slug = args.hf_path.replace("/", "-")
    else:
        model, tokenizer, meta = load_model(
            args.source,
            device=device,
            phase="eval",
            model_tag=args.model_tag,
            step=args.step
        )
        model_name = f"{args.source}_model (step {meta['step']})"
        model_slug = f"{args.source}_model_{meta['step']:06d}"

    # Evaluate
    with autocast_ctx:
        out = evaluate_model(model, tokenizer, device, max_per_task=args.max_per_task)

    # Write results
    if ddp_rank == 0:
        base_dir = get_base_dir()
        output_csv_path = os.path.join(base_dir, "base_eval", f"{model_slug}.csv")
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

        results = out["results"]
        centered_results = out["centered_results"]
        core_metric = out["core_metric"]

        with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
            f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
            for label in results:
                f.write(f"{label:<35}, {results[label]:<10.6f}, {centered_results[label]:<10.6f}\n")
            f.write(f"{'CORE':<35}, {'':<10}, {core_metric:<10.6f}\n")

        # Print results
        print0("=" * 80)
        print0(f"Model: {model_name}")
        print0("=" * 80)
        with open(output_csv_path, 'r', encoding='utf-8') as f:
            print0(f.read())

        print0(f"\nCORE metric: {core_metric:.4f}")
        print0(f"Results saved to: {output_csv_path}")

    compute_cleanup()


if __name__ == "__main__":
    main()
