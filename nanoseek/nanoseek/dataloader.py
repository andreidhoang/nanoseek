"""
Distributed dataloaders for pretraining.

BOS-aligned bestfit:
   - Every row starts with BOS token
   - Documents packed using best-fit algorithm to minimize cropping
   - When no document fits remaining space, crops a document to fill exactly
   - 100% utilization (no padding), ~35% tokens cropped at T=2048

Compared to the original tokenizing_distributed_data_loader:
BOS-aligned loses ~35% of tokens to cropping, but ensures that
there are fewer "confusing" tokens in the train/val batches as every token can
now attend back to the BOS token and sees the full context of the document.

Fallback to the original if you have very limited data AND long documents:
https://github.com/karpathy/nanochat/blob/3c3a3d7/nanochat/dataloader.py#L78-L117
"""

import random
import threading
from queue import Queue

import torch
import pyarrow.parquet as pq

from nanoseek.nanoseek.common import get_dist_info
from nanoseek.nanoseek.dataset import list_parquet_files


def fim_transform(tokens, fim_rate, fim_tokens, rng):
    """Apply FIM PSM transform with probability fim_rate (RULE 6).

    PSM format: BOS [fim_prefix] prefix [fim_suffix] [fim_middle] middle
    Reference: Bavarian et al. 2022 "Efficient Training of LMs to Fill in the Middle"
    """
    if rng.random() >= fim_rate or len(tokens) < 4:
        return tokens, False
    # Skip BOS token — split only content
    bos = tokens[0]
    content = tokens[1:]
    split = rng.randint(1, max(1, len(content) - 1))
    prefix, middle = content[:split], content[split:]
    # PSM: BOS [prefix_tok] prefix [suffix_tok] [middle_tok] middle
    transformed = (
        [bos, fim_tokens['prefix']] + prefix
        + [fim_tokens['suffix'], fim_tokens['middle']] + middle
    )
    return transformed, True

def _document_batches(split, resume_state_dict, tokenizer_batch_size):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.

    Handles DDP sharding and approximate resume. Each yield is (text_batch, (pq_idx, rg_idx, epoch))
    where text_batch is a list of document strings, indices track position for resumption,
    and epoch counts how many times we've cycled through the dataset (starts at 1).
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    warn_on_legacy = ddp_rank == 0 and split == "train" # rank 0 on train split will warn on legacy
    parquet_paths = list_parquet_files(warn_on_legacy=warn_on_legacy)
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:  # iterate infinitely (multi-epoch)
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            # Start from resume point if resuming on same file, otherwise from DDP rank
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1  # advance by 1 so we don't repeat data after resuming
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None  # only do this once
            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000,
    fim_rate=0.0,
):
    """
    BOS-aligned dataloader with Best-Fit Cropping.

    Reduces token waste compared to simple greedy cropping by searching a buffer
    for documents that fit well, while maintaining 100% utilization (no padding).

    Algorithm for each row:
    1. From buffered docs, pick the LARGEST doc that fits entirely
    2. Repeat until no doc fits
    3. When nothing fits, crop a doc to fill remaining space exactly

    Key properties:
    - Every row starts with BOS
    - 100% utilization (no padding, every token is trained on)
    - Approximately 35% of all tokens are discarded due to cropping
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    # FIM setup (RULE 6: 10% PSM from token 1, train split only)
    use_fim = fim_rate > 0 and split == "train"
    fim_tokens = tokenizer.get_fim_tokens() if use_fim else None
    fim_rng = random.Random(42)  # deterministic FIM decisions
    # Restore FIM RNG state from checkpoint (if available) so that
    # FIM decisions after resume match what they would have been
    # in a continuous run. Without this, the RNG resets to seed 42
    # on every resume, breaking ablation reproducibility.
    if resume_state_dict is not None and "fim_rng_state" in resume_state_dict:
        fim_rng.setstate(tuple(
            tuple(x) if isinstance(x, list) else x
            for x in resume_state_dict["fim_rng_state"]
        ))
    fim_count = 0  # running count of FIM-transformed docs
    total_count = 0  # running count of all docs

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch, fim_count, total_count
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            total_count += 1
            if use_fim:
                tokens, was_fim = fim_transform(tokens, fim_rate, fim_tokens, fim_rng)
                if was_fim:
                    fim_count += 1
            doc_buffer.append(tokens)

    # Pre-allocate buffers once: layout is [inputs (B*T) | targets (B*T)]
    # This gives us contiguous views and a single HtoD transfer
    use_cuda = device == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long) # for building rows without creating Python lists
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device) # on-device buffer
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    # ─── Double-buffered prefetch (PERF: overlap CPU packing + GPU training) ───
    # Without this, the GPU idles while the CPU packs the next batch (and vice versa).
    # With double buffering, a background thread packs batch N+1 into cpu_buf_B
    # while the GPU trains on batch N (already copied from cpu_buf_A).
    # Python GIL releases during tokenizer.encode() (Rust FFI) and torch.tensor(),
    # so the background thread genuinely runs in parallel with the main thread.
    cpu_buf_A = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    cpu_buf_B = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    prefetch_queue = Queue(maxsize=2)

    def _pack_one_batch(cpu_buffer):
        """Pack a full batch into the given CPU buffer. Runs in background thread."""
        nonlocal pq_idx, rg_idx, epoch, fim_count, total_count
        cpu_inputs = cpu_buffer[:B * T].view(B, T)
        cpu_targets = cpu_buffer[B * T:].view(B, T)

        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                # Ensure buffer has documents
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                    pos += doc_len
                else:
                    # No doc fits - crop shortest in buffer to fill remaining and minimize waste
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        # Copy to pinned CPU buffer
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        # Save FIM RNG state so it can be restored on resume.
        fim_rng_state = list(fim_rng.getstate()) if use_fim else None
        if fim_rng_state is not None:
            fim_rng_state[1] = list(fim_rng_state[1])
        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch,
                      "fim_fraction": fim_count / max(total_count, 1),
                      "fim_rng_state": fim_rng_state}

        return cpu_buffer, state_dict

    def _prefetch_worker():
        """Background thread: alternately packs batches into cpu_buf_A and cpu_buf_B."""
        batch_num = 0
        try:
            while True:
                buf = cpu_buf_A if (batch_num % 2 == 0) else cpu_buf_B
                cpu_buffer, state_dict = _pack_one_batch(buf)
                prefetch_queue.put((cpu_buffer, state_dict))
                batch_num += 1
        except Exception as e:
            # Propagate exception to main thread via queue
            prefetch_queue.put(e)

    worker = threading.Thread(target=_prefetch_worker, daemon=True)
    worker.start()

    while True:
        result = prefetch_queue.get()
        if isinstance(result, Exception):
            raise result
        cpu_buffer, state_dict = result

        # Single HtoD copy into persistent GPU buffer and yield
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets

"""
  ──────────────────────────────────────────────────────────────────────────────────────────────────────
  🖼️ VISUAL: Data Flow Architecture

  ┌─────────────────────────────────────────────────────────────┐
  │                    PARQUET SHARDS (HuggingFace)              │
  │  ┌─────────┐ ┌─────────┐ ┌─────────┐       ┌─────────┐     │
  │  │shard_000│ │shard_001│ │shard_002│  ...  │shard_542│     │
  │  │(train)  │ │(train)  │ │(train)  │       │(val)    │     │
  │  └────┬────┘ └────┬────┘ └────┬────┘       └────┬────┘     │
  └───────┼───────────┼───────────┼─────────────────┼───────────┘
          │           │           │                 │
          ▼           ▼           ▼                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │              PYARROW ROW GROUPS (per parquet)                │
  │  Each parquet → multiple row groups (~10k docs each)         │
  │  DDP sharding: rank r processes row_groups r, r+W, r+2W...  │
  └─────────────────────────────────────────────────────────────┘
          │
          ▼
  ┌─────────────────────────────────────────────────────────────┐
  │           DOCUMENT BATCHES (list of text strings)            │
  │  ["The cat sat...", "In 1492, Columbus...", ...]            │
  │  Tokenized with BOS prepended: [[1, 45, 67], [1, 23, ...]]  │
  └─────────────────────────────────────────────────────────────┘
          │
          ▼
  ┌─────────────────────────────────────────────────────────────┐
  │              BEST-FIT PACKING ALGORITHM                      │
  │                                                              │
  │  Buffer: [doc0, doc1, doc2, doc3, doc4, ...]  (1000 docs)   │
  │                                                              │
  │  For each row in batch (B rows):                             │
  │    remaining = T + 1                                         │
  │    while remaining > 0:                                      │
  │      best = argmax(len(doc)) where len(doc) <= remaining     │
  │      if best exists:                                         │
  │        place doc, remaining -= len(doc)                      │
  │      else:                                                   │
  │        crop shortest doc to fill remaining exactly           │
  │        remaining = 0                                         │
  └─────────────────────────────────────────────────────────────┘
          │
          ▼
  ┌─────────────────────────────────────────────────────────────┐
  │              MEMORY LAYOUT (Pre-allocated)                   │
  │                                                              │
  │  row_buffer: [B, T+1]  (CPU, building rows)                  │
  │  cpu_buffer:  [2*B*T]  (pinned memory, staging)              │
  │    ├── cpu_inputs:  [B, T]  (first half)                    │
  │    └── cpu_targets: [B, T]  (second half)                   │
  │  gpu_buffer: [2*B*T] on CUDA (device memory)                │
  │    ├── inputs:  [B, T]                                      │
  │    └── targets: [B, T]                                      │
  │                                                              │
  │  Single HtoD copy: gpu_buffer.copy_(cpu_buffer, non_blocking)│
  └─────────────────────────────────────────────────────────────┘

  ──────────────────────────────────────────────────────────────────────────────────────────────────────
"""