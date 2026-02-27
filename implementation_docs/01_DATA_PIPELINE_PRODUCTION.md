# 01 — Production Data Pipeline for NanoSeek Training

## Principal Engineer's Implementation Guide — February 2026

**Author**: Distinguished AI Researcher & Principal Engineer, Foundation Models Division
**Scope**: Complete production data pipeline — from raw parquet on disk to GPU tensors
**Prerequisite**: Read `00_MASTER_PLAN.md` for context on the full production stack

---

## 1. Problem Statement

The existing `scripts/dataloader.py` streams from parquet, tokenizes on-the-fly, and yields
batches. It works for prototyping but has critical gaps for 22B-token training on 8×H100:

| Gap | Impact | Severity |
|-----|--------|----------|
| No prefetch pipeline | GPU stalls waiting for CPU tokenization | **CRITICAL** |
| Approximate resume only | Repeated/skipped documents after checkpoint restore | HIGH |
| No data-quality monitoring | Silent data corruption goes undetected | MEDIUM |
| No integrity verification | Corrupt parquet files crash training mid-run | HIGH |
| No throughput instrumentation | Cannot diagnose data pipeline bottlenecks | MEDIUM |

**Target**: At ~300ms/step for NanoSeek-1B on H100, the pipeline needs <100ms batch latency.
13.6M tokens/sec aggregate (1.7M/GPU) with zero GPU stalls and exact checkpoint resume.

---

## 2. First Principles Analysis

### Streaming Over Random Access

LLM pretraining processes each document once per epoch. Random access requires pre-tokenizing
all 22B tokens (doubling storage) and managing a ~170GB shuffle index. Streaming reads parquet
row groups sequentially, tokenizes on CPU, and pushes to GPU. State is just
`(parquet_idx, row_group_idx, token_buffer_offset)`.

### Token Buffer vs Document Packing

Cross-document contamination within sequences is standard practice (GPT-2/3, LLaMA, DeepSeek
all concatenate with BOS delimiters). Token buffer approach: zero padding waste, simpler code,
perfect GPU utilization. We keep it.

### Exact vs Approximate Resume

The current `(pq_idx, rg_idx)` state skips documents and loses the token buffer on resume.
**Cost of exact resume**: ~50 additional lines to serialize the buffer and track offsets.
The benefit — deterministic training with identical loss curves — is worth it.

---

## 3. Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 Production Data Pipeline (per GPU rank)                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Parquet Shards (~94MB each, pre-shuffled FineWeb-Edu)                 │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────────────────────┐                                       │
│  │  Row Group Reader           │  DDP: rank r reads rg[r::world_size]  │
│  │  (PyArrow, sequential I/O)  │  from each parquet file               │
│  └──────────┬──────────────────┘                                       │
│             │ List[str] (~1024 docs per row group)                      │
│             ▼                                                           │
│  ┌─────────────────────────────┐                                       │
│  │  Tokenizer Thread Pool      │  tiktoken encode_ordinary_batch       │
│  │  (N=4, batch_size=128 docs) │  with BOS prepend                     │
│  └──────────┬──────────────────┘                                       │
│             │ flattened token IDs                                       │
│             ▼                                                           │
│  ┌─────────────────────────────┐                                       │
│  │  Circular Token Buffer      │  deque, capacity ~4×B×T tokens        │
│  │  (append right, pop left)   │  smooths row-group/batch boundary     │
│  └──────────┬──────────────────┘                                       │
│             │ B*T+1 tokens                                             │
│             ▼                                                           │
│  ┌─────────────────────────────┐                                       │
│  │  Batch Formation            │  pin_memory=True for async transfer   │
│  │  inputs=scratch[:-1]        │  .to(device, non_blocking=True)       │
│  │  targets=scratch[1:]        │                                       │
│  └──────────┬──────────────────┘                                       │
│             │                                                           │
│             ▼                                                           │
│  ┌─────────────────────────────┐                                       │
│  │  Prefetch Queue (depth=2)   │  Background thread produces,          │
│  │  (double-buffered)          │  main thread consumes                 │
│  └──────────┬──────────────────┘                                       │
│             ▼                                                           │
│        Training Loop                                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

The background producer thread stays 1–2 batches ahead. With `maxsize=2`, it blocks
if GPU consumes slower than CPU produces (the normal case — GPU compute dominates).

---

## 4. Production Code

### 4a. Enhanced DataLoader (`scripts/dataloader.py` replacement)

```python
"""
Production Streaming Distributed DataLoader for NanoSeek Training.

Enhancements over baseline:
- Background prefetch thread with configurable queue depth
- Exact checkpoint resume via full state serialization
- DDP-aware row-group sharding with deterministic assignment
- CUDA pinned memory with async GPU transfer
- Token throughput instrumentation
- Corrupted parquet file detection and skip
"""

import os
import time
import logging
import threading
from collections import deque
from dataclasses import dataclass, field, asdict
from queue import Queue, Empty
from typing import Optional, Dict, Any, List, Tuple, Iterator

import torch
import pyarrow.parquet as pq

try:
    from .utils import get_dist_info
    from .dataset import list_parquet_files
    from .tokenizer import get_tokenizer
except ImportError:
    from utils import get_dist_info
    from dataset import list_parquet_files
    from tokenizer import get_tokenizer

logger = logging.getLogger(__name__)


@dataclass
class DataLoaderState:
    """Complete state for exact checkpoint resume."""
    pq_idx: int = 0
    rg_idx: int = 0
    doc_batch_offset: int = 0
    token_buffer: List[int] = field(default_factory=list)
    tokens_yielded: int = 0
    epoch: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataLoaderState":
        return cls(**d)


class ThroughputTracker:
    """Tracks token throughput with exponential moving average."""

    def __init__(self, window_size: int = 50):
        self._times: deque = deque(maxlen=window_size)
        self._counts: deque = deque(maxlen=window_size)
        self._total_tokens: int = 0
        self._start_time: float = time.monotonic()

    def record(self, num_tokens: int) -> None:
        self._times.append(time.monotonic())
        self._counts.append(num_tokens)
        self._total_tokens += num_tokens

    @property
    def tokens_per_sec(self) -> float:
        if len(self._times) < 2:
            return 0.0
        dt = self._times[-1] - self._times[0]
        return sum(self._counts) / max(dt, 1e-9)

    @property
    def total_tokens(self) -> int:
        return self._total_tokens


def _read_row_group_safe(filepath: str, rg_idx: int) -> Optional[List[str]]:
    """Read a single row group, returning None on corruption."""
    try:
        pf = pq.ParquetFile(filepath)
        if rg_idx >= pf.num_row_groups:
            return None
        rg = pf.read_row_group(rg_idx)
        return rg.column("text").to_pylist()
    except Exception as e:
        logger.warning("Corrupt row group %d in %s: %s — skipping", rg_idx, filepath, e)
        return None


def _document_stream(
    split: str, ddp_rank: int, ddp_world_size: int, state: DataLoaderState,
) -> Iterator[Tuple[List[str], int, int, int]]:
    """
    Infinite iterator over document batches with DDP sharding.
    Yields (doc_batch, pq_idx, rg_idx, doc_batch_offset).
    """
    parquet_paths = list_parquet_files()
    if not parquet_paths:
        raise RuntimeError("No parquet files found. Run `python scripts/dataset.py` first.")

    paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    epoch = state.epoch
    resume_pq, resume_rg = state.pq_idx, state.rg_idx
    resume_doc_offset = state.doc_batch_offset
    is_first_pass = True

    while True:
        start_pq = resume_pq if is_first_pass else 0
        for pq_idx in range(start_pq, len(paths)):
            try:
                pf = pq.ParquetFile(paths[pq_idx])
            except Exception as e:
                logger.warning("Cannot open %s: %s — skipping", paths[pq_idx], e)
                continue

            start_rg = resume_rg if (is_first_pass and pq_idx == resume_pq) else ddp_rank
            rg_idx = start_rg
            while rg_idx < pf.num_row_groups:
                docs = _read_row_group_safe(paths[pq_idx], rg_idx)
                if docs is not None:
                    if is_first_pass and pq_idx == resume_pq and rg_idx == resume_rg:
                        docs = docs[resume_doc_offset:]
                        is_first_pass = False
                        yield docs, pq_idx, rg_idx, resume_doc_offset
                    else:
                        is_first_pass = False
                        yield docs, pq_idx, rg_idx, 0
                rg_idx += ddp_world_size

        epoch += 1
        state.epoch = epoch
        resume_pq, resume_rg, resume_doc_offset = 0, ddp_rank, 0
        is_first_pass = False


def _producer_loop(
    queue: Queue, stop_event: threading.Event,
    split: str, B: int, T: int,
    tokenizer_threads: int, tokenizer_batch_size: int,
    device: str, state: DataLoaderState,
    ddp_rank: int, ddp_world_size: int, throughput: ThroughputTracker,
) -> None:
    """Background thread: reads, tokenizes, forms batches, enqueues."""
    needed_tokens = B * T + 1
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    token_buffer = deque(state.token_buffer)
    state.token_buffer = []
    doc_stream = _document_stream(split, ddp_rank, ddp_world_size, state)
    use_cuda = device == "cuda"

    for doc_batch, pq_idx, rg_idx, doc_offset in doc_stream:
        if stop_event.is_set():
            return
        for sub_start in range(0, len(doc_batch), tokenizer_batch_size):
            if stop_event.is_set():
                return
            sub_batch = doc_batch[sub_start : sub_start + tokenizer_batch_size]
            token_lists = tokenizer.encode(
                sub_batch, prepend=bos_token, num_threads=tokenizer_threads
            )
            for tokens in token_lists:
                token_buffer.extend(tokens)

            while len(token_buffer) >= needed_tokens:
                batch_tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
                scratch = torch.tensor(batch_tokens, dtype=torch.long, pin_memory=use_cuda)
                inputs_cpu = scratch[:-1].view(B, T)
                targets_cpu = scratch[1:].view(B, T)

                batch_state = DataLoaderState(
                    pq_idx=pq_idx, rg_idx=rg_idx,
                    doc_batch_offset=doc_offset + sub_start + tokenizer_batch_size,
                    token_buffer=list(token_buffer),
                    tokens_yielded=state.tokens_yielded + needed_tokens - 1,
                    epoch=state.epoch,
                )
                throughput.record(needed_tokens - 1)
                try:
                    queue.put((inputs_cpu, targets_cpu, batch_state), timeout=5.0)
                except Exception:
                    if stop_event.is_set():
                        return
                    queue.put((inputs_cpu, targets_cpu, batch_state))


class ProductionDataLoader:
    """
    Production streaming dataloader with prefetch and exact resume.

    Usage:
        loader = ProductionDataLoader(B=8, T=4096, split="train", device="cuda")
        for inputs, targets, state_dict in loader:
            loss = model(inputs, targets)
            ...  # save state_dict to checkpoint for exact resume
    """

    def __init__(
        self, B: int, T: int, split: str,
        tokenizer_threads: int = 4, tokenizer_batch_size: int = 128,
        device: str = "cuda", prefetch_depth: int = 2,
        resume_state_dict: Optional[Dict[str, Any]] = None,
    ):
        self.B, self.T, self.split, self.device = B, T, split, device
        _, ddp_rank, _, ddp_world_size = get_dist_info()
        self.ddp_rank, self.ddp_world_size = ddp_rank, ddp_world_size

        self._state = (DataLoaderState.from_dict(resume_state_dict)
                       if resume_state_dict else DataLoaderState(rg_idx=ddp_rank))
        self._queue: Queue = Queue(maxsize=prefetch_depth)
        self._stop_event = threading.Event()
        self.throughput = ThroughputTracker()

        self._thread = threading.Thread(
            target=_producer_loop,
            args=(self._queue, self._stop_event, split, B, T,
                  tokenizer_threads, tokenizer_batch_size, device,
                  self._state, ddp_rank, ddp_world_size, self.throughput),
            daemon=True, name=f"dataloader-{split}-rank{ddp_rank}",
        )
        self._thread.start()

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        while True:
            try:
                inputs_cpu, targets_cpu, batch_state = self._queue.get(timeout=30.0)
                break
            except Empty:
                if not self._thread.is_alive():
                    raise StopIteration("Producer thread died")
                logger.warning("Dataloader queue empty for 30s — producer may be slow")
        use_cuda = self.device == "cuda"
        inputs = inputs_cpu.to(device=self.device, non_blocking=use_cuda)
        targets = targets_cpu.to(device=self.device, non_blocking=use_cuda)
        return inputs, targets, batch_state.to_dict()

    def shutdown(self) -> None:
        self._stop_event.set()
        try:
            while not self._queue.empty():
                self._queue.get_nowait()
        except Empty:
            pass
        self._thread.join(timeout=5.0)

    def __del__(self):
        self.shutdown()


# Drop-in compatibility wrappers (same API as original dataloader.py)
def tokenizing_distributed_data_loader_with_state(
    B, T, split, tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
):
    """Drop-in replacement: yields (inputs, targets, state_dict) tuples."""
    loader = ProductionDataLoader(
        B=B, T=T, split=split, tokenizer_threads=tokenizer_threads,
        tokenizer_batch_size=tokenizer_batch_size, device=device,
        resume_state_dict=resume_state_dict,
    )
    yield from loader


def tokenizing_distributed_data_loader(*args, **kwargs):
    """Compatibility wrapper that strips the state_dict."""
    for inputs, targets, _ in tokenizing_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets
```

### 4b. Enhanced Dataset Preparation (`scripts/setup_data.py` enhancement)

```python
"""
Production data preparation: integrity verification, statistics, split creation.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from multiprocessing import Pool

import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def verify_parquet_integrity(filepath: str) -> Tuple[bool, str]:
    """Verify parquet file is readable with expected 'text' column."""
    try:
        pf = pq.ParquetFile(filepath)
        if "text" not in pf.schema_arrow.names:
            return False, f"Missing 'text' column. Found: {pf.schema_arrow.names}"
        if pf.num_row_groups == 0:
            return False, "0 row groups"
        # Spot-check first and last row groups
        for rg_idx in [0, pf.num_row_groups - 1]:
            texts = pf.read_row_group(rg_idx).column("text").to_pylist()
            if not texts or not isinstance(texts[0], str):
                return False, f"Row group {rg_idx}: empty or wrong type"
        return True, f"OK: {pf.num_row_groups} row groups"
    except Exception as e:
        return False, f"Error: {e}"


def compute_shard_statistics(filepath: str) -> Dict:
    """Compute document/token statistics for a single parquet shard."""
    try:
        pf = pq.ParquetFile(filepath)
        total_docs, total_chars = 0, 0
        min_len, max_len = float("inf"), 0
        for rg_idx in range(pf.num_row_groups):
            for text in pf.read_row_group(rg_idx).column("text").to_pylist():
                n = len(text)
                total_docs += 1
                total_chars += n
                min_len, max_len = min(min_len, n), max(max_len, n)
        return {
            "filepath": filepath, "num_row_groups": pf.num_row_groups,
            "total_docs": total_docs, "total_chars": total_chars,
            "est_tokens": total_chars // 4,
            "min_doc_chars": min_len if min_len != float("inf") else 0,
            "max_doc_chars": max_len,
            "file_size_mb": os.path.getsize(filepath) / (1024 * 1024),
        }
    except Exception as e:
        return {"filepath": filepath, "error": str(e)}


def validate_dataset(data_dir: str) -> bool:
    """Validate all parquet files. Returns True if all pass."""
    try:
        from dataset import list_parquet_files
    except ImportError:
        from scripts.dataset import list_parquet_files
    paths = list_parquet_files(data_dir)
    if not paths:
        logger.error("No parquet files in %s", data_dir)
        return False
    all_valid = True
    for path in paths:
        valid, msg = verify_parquet_integrity(path)
        logger.info("  %s %s: %s", "✓" if valid else "✗", os.path.basename(path), msg)
        if not valid:
            all_valid = False
    return all_valid


def report_dataset_statistics(data_dir: str, num_workers: int = 4) -> Dict:
    """Compute and report aggregate statistics across all shards."""
    try:
        from dataset import list_parquet_files
    except ImportError:
        from scripts.dataset import list_parquet_files
    paths = list_parquet_files(data_dir)
    if not paths:
        return {}
    with Pool(processes=num_workers) as pool:
        stats_list = pool.map(compute_shard_statistics, paths)
    total_docs = total_chars = total_tokens = total_rg = 0
    total_mb = 0.0
    for s in stats_list:
        if "error" not in s:
            total_docs += s["total_docs"]
            total_chars += s["total_chars"]
            total_tokens += s["est_tokens"]
            total_rg += s["num_row_groups"]
            total_mb += s["file_size_mb"]
    logger.info("Dataset: %d shards, %d row groups, %s docs, ~%.1fB tokens, %.1f GB",
                len(paths), total_rg, f"{total_docs:,}", total_tokens / 1e9, total_mb / 1024)
    return {"num_shards": len(paths), "total_docs": total_docs,
            "est_total_tokens": total_tokens, "total_size_mb": round(total_mb, 1)}
```

### 4c. Data Quality Monitoring (`scripts/data_utils.py`)

```python
"""
Data quality monitoring for NanoSeek training.

Tracks token throughput, batch latency, data staleness, and token distribution.
"""

import time
import logging
import hashlib
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class BatchStatistics:
    """Statistics for a single training batch."""
    batch_idx: int
    timestamp: float
    num_tokens: int
    batch_latency_ms: float
    unique_token_ratio: float


class DataQualityMonitor:
    """
    Monitors data pipeline health during training.

    Tracks token throughput (rolling window), batch formation latency,
    data staleness (repeated n-gram sequences), and logs periodic summaries.
    """

    def __init__(
        self, vocab_size: int, window_size: int = 100,
        staleness_ngram_size: int = 64, staleness_history: int = 1000,
        log_interval: int = 100,
    ):
        self.vocab_size = vocab_size
        self.log_interval = log_interval
        self._batch_times: deque = deque(maxlen=window_size)
        self._batch_tokens: deque = deque(maxlen=window_size)
        self._latencies_ms: deque = deque(maxlen=window_size)
        self._total_tokens = 0
        self._total_batches = 0
        self._start_time = time.monotonic()
        self._ngram_size = staleness_ngram_size
        self._seen_hashes: deque = deque(maxlen=staleness_history)
        self._duplicate_count = 0

    def record_batch(
        self, inputs: torch.Tensor, batch_latency_ms: float,
    ) -> Optional[BatchStatistics]:
        """Record a batch. Returns BatchStatistics on logging steps."""
        now = time.monotonic()
        num_tokens = inputs.numel()
        self._batch_times.append(now)
        self._batch_tokens.append(num_tokens)
        self._latencies_ms.append(batch_latency_ms)
        self._total_tokens += num_tokens
        self._total_batches += 1

        # Staleness: hash first sequence
        first_seq = inputs[0].cpu()
        if len(first_seq) >= self._ngram_size:
            h = hashlib.md5(first_seq[: self._ngram_size].numpy().tobytes()).hexdigest()
            if h in self._seen_hashes:
                self._duplicate_count += 1
            self._seen_hashes.append(h)

        unique_ratio = float(inputs.unique().numel()) / max(num_tokens, 1)
        stats = BatchStatistics(
            batch_idx=self._total_batches, timestamp=now,
            num_tokens=num_tokens, batch_latency_ms=batch_latency_ms,
            unique_token_ratio=unique_ratio,
        )

        if self._total_batches % self.log_interval == 0:
            self._log_summary()
            return stats
        return None

    def _log_summary(self) -> None:
        elapsed = time.monotonic() - self._start_time
        avg_tp = self._total_tokens / max(elapsed, 1e-6)
        rolling_tp = 0.0
        if len(self._batch_times) >= 2:
            dt = self._batch_times[-1] - self._batch_times[0]
            rolling_tp = sum(self._batch_tokens) / max(dt, 1e-6)
        avg_lat = sum(self._latencies_ms) / max(len(self._latencies_ms), 1)
        max_lat = max(self._latencies_ms) if self._latencies_ms else 0.0
        logger.info(
            "[DataPipeline] batch=%d | tok/s=%.0f (avg=%.0f) | "
            "lat=%.1fms (max=%.1fms) | dupes=%d | total=%.2fB",
            self._total_batches, rolling_tp, avg_tp,
            avg_lat, max_lat, self._duplicate_count, self._total_tokens / 1e9,
        )

    def get_metrics(self) -> Dict[str, float]:
        """Return current metrics dict (for W&B logging)."""
        elapsed = time.monotonic() - self._start_time
        rolling_tp = 0.0
        if len(self._batch_times) >= 2:
            dt = self._batch_times[-1] - self._batch_times[0]
            rolling_tp = sum(self._batch_tokens) / max(dt, 1e-6)
        return {
            "data/tokens_per_sec": rolling_tp,
            "data/avg_tokens_per_sec": self._total_tokens / max(elapsed, 1e-6),
            "data/batch_latency_ms": sum(self._latencies_ms) / max(len(self._latencies_ms), 1),
            "data/duplicate_sequences": self._duplicate_count,
            "data/total_tokens": self._total_tokens,
        }
```

---

## 5. File Placement

```
scripts/
├── dataloader.py     # Enhanced streaming dataloader (Section 4a)
│                     # Drop-in replacement — same API, new internals
├── setup_data.py     # Enhanced data preparation (Section 4b)
│                     # Added: verify_parquet_integrity, validate_dataset,
│                     # compute_shard_statistics, report_dataset_statistics
├── data_utils.py     # NEW: Data quality monitoring (Section 4c)
│                     # DataQualityMonitor, BatchStatistics
├── dataset.py        # Unchanged — parquet file listing and download
├── tokenizer.py      # Unchanged — tokenizer backends
└── utils.py          # Unchanged — DDP utilities
```

The enhanced `dataloader.py` exports the same two functions with identical signatures.
`pre-train.py` continues to work without modification.

---

## 6. Integration Guide

### Swapping into pre-train.py

**Zero changes required.** The existing import:

```python
from scripts.dataloader import (
    tokenizing_distributed_data_loader_with_state,
    tokenizing_distributed_data_loader,
)
```

works as-is. The returned `state_dict` now contains full token buffer state for exact
resume instead of just `(pq_idx, rg_idx)`.

### Adding Data Quality Monitoring

```python
from scripts.data_utils import DataQualityMonitor

# Initialize once (after model config is loaded)
monitor = DataQualityMonitor(vocab_size=model_config.vocab_size, log_interval=100)

# In training loop, after receiving batch:
t_batch = time.time()
x, y, dataloader_state = next(train_loader)
batch_latency_ms = (time.time() - t_batch) * 1000

stats = monitor.record_batch(x, batch_latency_ms)
if use_wandb and stats is not None:
    wandb.log(monitor.get_metrics())
```

### Dataset Validation Pre-Flight

```python
from scripts.setup_data import validate_dataset
from scripts.dataset import DATA_DIR

if master_process and not validate_dataset(DATA_DIR):
    print("ERROR: Dataset validation failed. Fix corrupt files before training.")
    sys.exit(1)
```

### Config Parameters to Add

```python
# In TrainingConfig dataclass:
tokenizer_threads: int = 4        # Threads for tiktoken batch encoding
tokenizer_batch_size: int = 128   # Documents per tokenizer call
prefetch_depth: int = 2           # Batches to prefetch (2 = double buffer)
```

---

## 7. Verification

### Unit Tests for Exact Resume

```python
"""tests/test_dataloader_resume.py"""
import torch
from scripts.dataloader import ProductionDataLoader, DataLoaderState

class TestExactResume:
    def test_resume_produces_same_tokens(self):
        """Run N batches, save state, resume, verify next M batches match."""
        B, T, N, M = 2, 128, 5, 3
        loader1 = ProductionDataLoader(B=B, T=T, split="train", device="cpu")
        all_batches, state_at_n = [], None
        for i, (x, y, state) in enumerate(loader1):
            all_batches.append((x.clone(), y.clone()))
            if i == N - 1:
                state_at_n = state
            if i >= N + M - 1:
                break
        loader1.shutdown()

        loader2 = ProductionDataLoader(B=B, T=T, split="train", device="cpu",
                                        resume_state_dict=state_at_n)
        for i, (x, y, _) in enumerate(loader2):
            orig_x, orig_y = all_batches[N + i]
            assert torch.equal(orig_x, x), f"Input mismatch at batch {i}"
            assert torch.equal(orig_y, y), f"Target mismatch at batch {i}"
            if i >= M - 1:
                break
        loader2.shutdown()

    def test_state_roundtrip(self):
        state = DataLoaderState(pq_idx=3, rg_idx=7, doc_batch_offset=42,
                                token_buffer=[1, 2, 3], tokens_yielded=100000, epoch=1)
        assert state == DataLoaderState.from_dict(state.to_dict())
```

### Throughput Benchmark

```python
"""benchmarks/bench_dataloader.py"""
import time
from scripts.dataloader import ProductionDataLoader

def benchmark_throughput(B=8, T=4096, num_batches=100, device="cpu"):
    loader = ProductionDataLoader(B=B, T=T, split="train", device=device)
    for i, _ in enumerate(loader):  # warmup
        if i >= 5: break
    start = time.monotonic()
    for i, _ in enumerate(loader):
        if i >= num_batches: break
    elapsed = time.monotonic() - start
    loader.shutdown()
    tok_per_sec = (num_batches * B * T) / elapsed
    print(f"Throughput: {tok_per_sec:,.0f} tok/s | {elapsed/num_batches*1000:.1f} ms/batch")

if __name__ == "__main__":
    benchmark_throughput()
```

### DDP Consistency Test

```python
"""tests/test_dataloader_ddp.py — run with: torchrun --nproc_per_node=2"""
import torch.distributed as dist
from scripts.dataloader import ProductionDataLoader

def test_ddp_no_overlap():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    loader = ProductionDataLoader(B=2, T=128, split="train", device="cpu")
    hashes = set()
    for i, (x, _, _) in enumerate(loader):
        hashes.add(hash(x.numpy().tobytes()))
        if i >= 20: break
    loader.shutdown()
    all_h = [None] * world
    dist.all_gather_object(all_h, hashes)
    if rank == 0:
        for i in range(world):
            for j in range(i+1, world):
                assert not (all_h[i] & all_h[j]), f"Rank {i} and {j} share batches"
        print("DDP consistency: PASSED")
    dist.destroy_process_group()

if __name__ == "__main__":
    test_ddp_no_overlap()
```

---

## 8. Performance Targets

| Metric | Target | How to Measure |
|--------|--------|---------------|
| Token throughput (per GPU) | >500K tok/sec | `bench_dataloader.py` with CUDA |
| Aggregate (8×H100) | >4M tok/sec | DDP benchmark |
| Batch latency | <100ms | `DataQualityMonitor.batch_latency_ms` |
| GPU utilization | >95% | `nvidia-smi` during training |
| Resume accuracy | Exact (0 token diff) | `test_resume_produces_same_tokens` |
| Prefetch queue occupancy | >1.0 avg | `queue.qsize()` |
| Corrupt file handling | Skip + log, no crash | Inject corrupt file test |

**Throughput budget** (per batch, B=8, T=4096):

| Stage | Latency |
|-------|---------|
| Parquet row group read | ~5ms |
| Tokenization (4 threads) | ~15ms |
| Buffer + tensor creation | ~2ms |
| Pin + async transfer | ~4ms |
| **Total** | **~26ms** (well under 300ms step) |

With `prefetch_depth=2`, the next batch is always ready. Bottleneck is GPU compute, not data.

---

## 9. Gotchas & Edge Cases

### 1. Parquet Row Group Sizes vs Batch Sizes

Each shard has ~1024 row groups of ~2000 docs. The token buffer smooths boundaries, but the
checkpoint must capture buffer contents for exact resume. At B=8, T=4096, worst case buffer
is ~132K tokens (~1MB serialized) — negligible vs model checkpoint (~9.5GB).

### 2. Tokenizer Memory in Long-Running Workers

tiktoken's Rust thread pool can cause gradual memory growth in 12+ hour runs if `List[str]`
batches aren't GC'd promptly. The producer thread processes one row group at a time and lets
sub-batches go out of scope. Monitor RSS if runs exceed 24 hours.

### 3. DDP Rank 0 Bottleneck for Validation

The validation split uses only the last parquet file. If it has fewer row groups than
`world_size`, some ranks get zero data. For validation, this is fine — ranks wrap to the
next epoch. For production, have rank 0 compute val loss and broadcast.

### 4. Token Buffer Drift Across Ranks

Each rank's buffer may drift in size due to different document lengths. This doesn't affect
correctness — each rank serializes its own `DataLoaderState` independently.

### 5. Queue Starvation on Slow Storage

NFS/EBS latency spikes >5s can drain the prefetch queue. Set `prefetch_depth=3+` on shared
storage. The 30s timeout in `__next__` logs a warning rather than crashing. Prefer local
NVMe where parquet shards are pre-staged.

### 6. Thread-Pool Non-Determinism

tiktoken's `encode_ordinary_batch` with `num_threads>1` is thread-scheduled. Output list
order is correct, but for bit-exact reproducibility across runs, pin `tokenizer_threads=1`.

### 7. Phase 2 Buffer Size

At T=8192 (Phase 2), worst-case buffer is ~262K tokens (~2MB as Python list, ~6MB
JSON-serialized). Still trivial vs model checkpoints.

---

*"The best data pipeline is invisible — the GPU never waits, the engineer never worries,
and the tokens flow like water."*

— Principal Engineer's Note, Foundation Models Division, 2026
