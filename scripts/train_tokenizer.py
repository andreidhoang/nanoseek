"""
Train a custom BPE tokenizer for NanoSeek.

This script trains a 65,536 vocab tokenizer on FineWeb-Edu data
to match nanochat's tokenizer for fair comparison.

Usage:
    # First, download some data shards
    python dataset.py -n 5  # Download 5 shards (~470MB)

    # Then train the tokenizer
    python train_tokenizer.py --vocab-size 65536 --num-shards 5

    # Quick test with smaller vocab
    python train_tokenizer.py --vocab-size 8192 --num-shards 1
"""

import os
import argparse
import time
from tqdm import tqdm

# Handle imports for both package and direct execution
try:
    from .utils import get_base_dir
    from .dataset import list_parquet_files, parquets_iter_batched
    from .tokenizer import (
        RustBPETokenizer,
        HuggingFaceTokenizer,
        SPECIAL_TOKENS,
        HAS_RUSTBPE,
        HAS_HF_TOKENIZERS,
    )
except ImportError:
    from utils import get_base_dir
    from dataset import list_parquet_files, parquets_iter_batched
    from tokenizer import (
        RustBPETokenizer,
        HuggingFaceTokenizer,
        SPECIAL_TOKENS,
        HAS_RUSTBPE,
        HAS_HF_TOKENIZERS,
    )


def text_iterator_from_parquets(num_shards=None, max_docs=None, progress=True):
    """
    Iterate over documents from parquet files.

    Args:
        num_shards: Maximum number of shards to use (None = all available)
        max_docs: Maximum documents to yield (None = all)
        progress: Show progress bar

    Yields:
        Individual document strings
    """
    parquet_paths = list_parquet_files()

    if len(parquet_paths) == 0:
        raise RuntimeError(
            "No parquet files found! Please download data first:\n"
            "  python dataset.py -n 5  # Download 5 shards"
        )

    if num_shards is not None:
        parquet_paths = parquet_paths[:num_shards]

    print(f"Training tokenizer on {len(parquet_paths)} shards...")

    import pyarrow.parquet as pq

    doc_count = 0

    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        shard_name = os.path.basename(filepath)

        # Estimate total rows for progress bar
        total_rows = sum(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups))

        pbar = tqdm(
            total=total_rows,
            desc=f"Processing {shard_name}",
            disable=not progress
        )

        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()

            for text in texts:
                yield text
                doc_count += 1
                pbar.update(1)

                if max_docs is not None and doc_count >= max_docs:
                    pbar.close()
                    print(f"Reached max_docs limit: {max_docs}")
                    return

        pbar.close()

    print(f"Total documents processed: {doc_count:,}")


def train_tokenizer(
    vocab_size: int = 65536,
    num_shards: int = None,
    max_docs: int = None,
    backend: str = "auto",
    save_dir: str = None,
):
    """
    Train a BPE tokenizer on FineWeb-Edu data.

    Args:
        vocab_size: Target vocabulary size (default: 65536 to match nanochat)
        num_shards: Number of parquet shards to use (None = all)
        max_docs: Maximum documents to train on (None = all)
        backend: "rustbpe", "huggingface", or "auto"
        save_dir: Directory to save tokenizer (default: ~/.cache/nanoseek/tokenizer/)

    Returns:
        Trained tokenizer instance
    """
    # Determine save directory
    if save_dir is None:
        base_dir = get_base_dir()
        save_dir = os.path.join(base_dir, "tokenizer")
    os.makedirs(save_dir, exist_ok=True)

    # Select backend
    if backend == "auto":
        if HAS_RUSTBPE:
            backend = "rustbpe"
            print("Using RustBPE backend (fast)")
        elif HAS_HF_TOKENIZERS:
            backend = "huggingface"
            print("Using HuggingFace tokenizers backend")
        else:
            raise ImportError(
                "No tokenizer backend available!\n"
                "Install one of:\n"
                "  pip install rustbpe tiktoken  # Recommended\n"
                "  pip install tokenizers        # Alternative"
            )

    print(f"\n{'='*60}")
    print(f"Training {vocab_size:,} vocab tokenizer")
    print(f"{'='*60}")
    print(f"Backend: {backend}")
    print(f"Special tokens: {len(SPECIAL_TOKENS)}")
    print(f"  {SPECIAL_TOKENS}")
    print(f"Save directory: {save_dir}")
    print()

    # Create text iterator
    text_iter = text_iterator_from_parquets(
        num_shards=num_shards,
        max_docs=max_docs,
        progress=True,
    )

    # Train tokenizer
    start_time = time.time()

    if backend == "rustbpe":
        if not HAS_RUSTBPE:
            raise ImportError("rustbpe not installed: pip install rustbpe tiktoken")
        tokenizer = RustBPETokenizer.train_from_iterator(text_iter, vocab_size)

    elif backend == "huggingface":
        if not HAS_HF_TOKENIZERS:
            raise ImportError("tokenizers not installed: pip install tokenizers")
        tokenizer = HuggingFaceTokenizer.train_from_iterator(text_iter, vocab_size)

    else:
        raise ValueError(f"Unknown backend: {backend}")

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Training complete in {elapsed:.1f}s")
    print(f"{'='*60}")

    # Save tokenizer
    tokenizer.save(save_dir)

    # Verify
    print(f"\nVerifying trained tokenizer...")
    actual_vocab = tokenizer.get_vocab_size()
    print(f"  Vocab size: {actual_vocab:,}")
    print(f"  BOS token ID: {tokenizer.get_bos_token_id()}")

    # Test encode/decode
    test_text = "Hello, world! This is NanoSeek training data."
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    print(f"\n  Test encoding:")
    print(f"    Original: {test_text}")
    print(f"    Tokens:   {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
    print(f"    Decoded:  {decoded}")

    # Test special tokens
    print(f"\n  Special tokens:")
    for st in SPECIAL_TOKENS[:4]:  # Show first 4
        try:
            st_id = tokenizer.encode_special(st)
            print(f"    {st}: {st_id}")
        except Exception as e:
            print(f"    {st}: (not found)")

    print(f"\n✅ Tokenizer saved to: {save_dir}")
    print(f"   The training pipeline will now use this tokenizer automatically.")

    return tokenizer


def verify_tokenizer():
    """Verify the trained tokenizer can be loaded and used."""
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")

    print(f"\nVerifying tokenizer in: {tokenizer_dir}")

    # Check which file exists
    pkl_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
    json_path = os.path.join(tokenizer_dir, "tokenizer.json")

    if os.path.exists(pkl_path):
        print(f"  Found: tokenizer.pkl (RustBPE format)")
        tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
    elif os.path.exists(json_path):
        print(f"  Found: tokenizer.json (HuggingFace format)")
        tokenizer = HuggingFaceTokenizer.from_directory(tokenizer_dir)
    else:
        print(f"  ❌ No tokenizer found!")
        print(f"     Run: python train_tokenizer.py --vocab-size 65536")
        return None

    print(f"  Vocab size: {tokenizer.get_vocab_size():,}")
    print(f"  BOS token: {tokenizer.get_bos_token_id()}")

    # Quick encode test
    test = "The quick brown fox jumps over the lazy dog."
    tokens = tokenizer.encode(test)
    decoded = tokenizer.decode(tokens)

    print(f"\n  Encode test:")
    print(f"    Input:   {test}")
    print(f"    Tokens:  {len(tokens)} tokens")
    print(f"    Decoded: {decoded}")

    # Check roundtrip
    if decoded == test:
        print(f"  ✅ Roundtrip: PASS")
    else:
        print(f"  ⚠️  Roundtrip: MISMATCH (may be whitespace normalization)")

    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer for NanoSeek",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train full 65K tokenizer on 5 shards (recommended for fair comparison)
  python train_tokenizer.py --vocab-size 65536 --num-shards 5

  # Quick test with small vocab
  python train_tokenizer.py --vocab-size 8192 --num-shards 1 --max-docs 100000

  # Verify existing tokenizer
  python train_tokenizer.py --verify
        """
    )

    parser.add_argument(
        "--vocab-size", "-v",
        type=int,
        default=65536,
        help="Vocabulary size (default: 65536 to match nanochat)"
    )
    parser.add_argument(
        "--num-shards", "-n",
        type=int,
        default=None,
        help="Number of parquet shards to use (default: all available)"
    )
    parser.add_argument(
        "--max-docs", "-d",
        type=int,
        default=None,
        help="Maximum documents to train on (default: all)"
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["auto", "rustbpe", "huggingface"],
        default="auto",
        help="Tokenizer backend (default: auto)"
    )
    parser.add_argument(
        "--save-dir", "-o",
        type=str,
        default=None,
        help="Output directory (default: ~/.cache/nanoseek/tokenizer/)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing tokenizer instead of training"
    )

    args = parser.parse_args()

    if args.verify:
        verify_tokenizer()
    else:
        train_tokenizer(
            vocab_size=args.vocab_size,
            num_shards=args.num_shards,
            max_docs=args.max_docs,
            backend=args.backend,
            save_dir=args.save_dir,
        )
