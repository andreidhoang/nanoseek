"""
Train a tokenizer using rustbpe BPE library.
Adapted from nanochat/scripts/tok_train.py for NanoSeek.

Usage: python -m nanoseek.scripts.tok_train --max-chars=2000000000
"""
import os
import time
import argparse
import torch
from nanoseek.nanoseek.tokenizer import RustBPETokenizer, _default_tokenizer_dir, SPECIAL_TOKENS
from nanoseek.nanoseek.dataset import parquets_iter_batched

parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
parser.add_argument('--max-chars', type=int, default=2_000_000_000,
                    help='Maximum characters to train on (default: 2B)')
parser.add_argument('--doc-cap', type=int, default=10_000,
                    help='Maximum characters per document (default: 10,000)')
parser.add_argument('--vocab-size', type=int, default=32768,
                    help='Vocabulary size (default: 32768 = 2^15)')
args = parser.parse_args()
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")

# Text iterator
def text_iterator():
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc
            if len(doc_text) > args.doc_cap:
                doc_text = doc_text[:args.doc_cap]
            nchars += len(doc_text)
            yield doc_text
            if nchars > args.max_chars:
                return

# Train
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iterator(), args.vocab_size)
t1 = time.time()
print(f"Training time: {t1 - t0:.2f}s")

# Save
tokenizer_dir = _default_tokenizer_dir()
tokenizer.save(tokenizer_dir)

# Sanity check
test_text = "Hello world! Numbers: 123. Unicode: 你好世界"
assert tokenizer.decode(tokenizer.encode(test_text)) == test_text
print("Sanity check passed")

# Cache token_bytes for BPB evaluation
vocab_size = tokenizer.get_vocab_size()
special_ids = set()
for name in SPECIAL_TOKENS:
    special_ids.add(tokenizer.encode_special(name))
token_bytes = []
for token_id in range(vocab_size):
    if token_id in special_ids:
        token_bytes.append(0)
    else:
        token_str = tokenizer.decode([token_id])
        token_bytes.append(len(token_str.encode("utf-8")))
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")
