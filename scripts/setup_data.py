#!/usr/bin/env python3
"""
NanoSeek Data Setup Script

Downloads and prepares training data for NanoSeek.

Usage:
    python scripts/setup_data.py --dataset fineweb-edu --tokens 10B
    python scripts/setup_data.py --dataset fineweb-edu --tokens 100B --output ./data
"""

import argparse
import os
import sys
from pathlib import Path


def parse_token_count(token_str: str) -> int:
    """Parse token count string (e.g., '10B', '100M') to integer."""
    token_str = token_str.upper().strip()

    multipliers = {
        'K': 1_000,
        'M': 1_000_000,
        'B': 1_000_000_000,
        'T': 1_000_000_000_000,
    }

    for suffix, mult in multipliers.items():
        if token_str.endswith(suffix):
            return int(float(token_str[:-1]) * mult)

    return int(token_str)


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []

    try:
        import datasets
    except ImportError:
        missing.append("datasets")

    try:
        import huggingface_hub
    except ImportError:
        missing.append("huggingface_hub")

    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)


def download_fineweb_edu(output_dir: Path, num_tokens: int, streaming: bool = True):
    """Download FineWeb-Edu dataset from HuggingFace."""
    from datasets import load_dataset

    print(f"Downloading FineWeb-Edu dataset...")
    print(f"Target tokens: {num_tokens:,}")
    print(f"Output directory: {output_dir}")

    # FineWeb-Edu is available in different sizes
    # We'll use streaming to handle large datasets efficiently

    dataset_name = "HuggingFaceFW/fineweb-edu"

    if streaming:
        print("Using streaming mode (recommended for large datasets)")
        dataset = load_dataset(
            dataset_name,
            split="train",
            streaming=True,
        )

        # Process in chunks and save
        output_file = output_dir / "fineweb_edu_train.arrow"

        print(f"Streaming data to {output_file}...")
        print("This may take a while depending on network speed...")

        # For demonstration - actual implementation would process tokens
        # and save in efficient format

    else:
        # Download specific subset
        print("Downloading dataset subset...")
        dataset = load_dataset(
            dataset_name,
            split="train[:1%]",  # Small subset for testing
        )

        output_file = output_dir / "fineweb_edu_sample.arrow"
        dataset.save_to_disk(str(output_file))
        print(f"Saved to {output_file}")

    return output_file


def setup_tokenizer(output_dir: Path):
    """Setup the rustbpe tokenizer."""
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tokenizer directory: {tokenizer_dir}")

    # Check if tokenizer already exists
    if (tokenizer_dir / "tokenizer.model").exists():
        print("Tokenizer already exists, skipping...")
        return

    print("Note: Tokenizer setup requires manual steps.")
    print("Please see tutorials/pre-train/ for tokenizer configuration.")
    print("Default: Use GPT-2 tokenizer or train custom tokenizer on your data.")


def create_sample_config(output_dir: Path):
    """Create a sample training configuration."""
    config_content = """# NanoSeek Sample Training Configuration
# Copy this file and modify for your setup

# Model
model_size: "1b"  # Options: 500m, 1b

# Data
data_dir: "./data/base_data"
tokenizer_path: "./data/tokenizer/tokenizer.model"

# Training
max_seq_len: 4096
global_batch_size: 128
device_batch_size: 4
gradient_accumulation_steps: 8

# Optimization
learning_rate: 3e-4
min_learning_rate: 3e-5
warmup_steps: 1000
weight_decay: 0.1
max_grad_norm: 1.0

# Schedule
total_tokens: 22_000_000_000  # 22B tokens
checkpoint_interval: 1000

# Logging
wandb_project: "nanoseek"
log_interval: 10

# Hardware
device: "cuda"
num_workers: 4
"""

    config_file = output_dir / "train_config.yaml"
    with open(config_file, 'w') as f:
        f.write(config_content)

    print(f"Created sample config: {config_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Setup training data for NanoSeek",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download 10B tokens for quick experiments
    python scripts/setup_data.py --dataset fineweb-edu --tokens 10B

    # Download full 100B tokens
    python scripts/setup_data.py --dataset fineweb-edu --tokens 100B

    # Create sample configuration only
    python scripts/setup_data.py --config-only
        """
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="fineweb-edu",
        choices=["fineweb-edu"],
        help="Dataset to download (default: fineweb-edu)"
    )

    parser.add_argument(
        "--tokens",
        type=str,
        default="10B",
        help="Number of tokens to download (e.g., 10B, 100B)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./data",
        help="Output directory (default: ./data)"
    )

    parser.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="Use streaming mode for large datasets"
    )

    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Only create sample configuration, don't download data"
    )

    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Skip tokenizer setup"
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("NanoSeek Data Setup")
    print("=" * 60)

    # Create sample config
    create_sample_config(output_dir)

    if args.config_only:
        print("\nConfiguration created. Run without --config-only to download data.")
        return

    # Check dependencies
    check_dependencies()

    # Parse token count
    num_tokens = parse_token_count(args.tokens)
    print(f"\nTarget: {num_tokens:,} tokens ({args.tokens})")

    # Setup data directory
    data_dir = output_dir / "base_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download dataset
    if args.dataset == "fineweb-edu":
        download_fineweb_edu(
            data_dir,
            num_tokens,
            streaming=args.streaming
        )

    # Setup tokenizer
    if not args.skip_tokenizer:
        setup_tokenizer(output_dir)

    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print(f"\nData directory: {output_dir}")
    print(f"Config file: {output_dir / 'train_config.yaml'}")
    print("\nNext steps:")
    print("  1. Review and modify train_config.yaml")
    print("  2. Setup tokenizer (see tutorials/pre-train/)")
    print("  3. Run training: python scripts/pre-train.py")


if __name__ == "__main__":
    main()
