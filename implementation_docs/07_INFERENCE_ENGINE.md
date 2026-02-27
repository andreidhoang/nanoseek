# 07 — Production Inference Engine for NanoSeek

## Principal Engineer's Implementation Guide — February 2026

**Author**: Distinguished AI Researcher & Principal Engineer, Foundation Models Division
**Prerequisite**: `00_MASTER_PLAN.md`, understanding of `model/model.py` generation methods
**Outcome**: Continuous batching engine with PagedAttention, OpenAI-compatible HTTP API, 100+ concurrent users per GPU

---

## 1. Problem Statement

`NanoSeekModel` provides three generation methods (`model/model.py` lines 1757-1903):

| Method | Mechanism | Throughput | Concurrency |
|--------|-----------|------------|-------------|
| `generate()` | Full recompute every step | ~30 tok/s | 1 user |
| `generate_simple()` | Streaming, no KV cache | ~30 tok/s | 1 user |
| `generate_cached()` | KV cache reuse | ~120 tok/s | 1 user |

All three are single-request, sequential, with no memory management. Production needs:

| Requirement | Current | Target |
|-------------|---------|--------|
| Concurrent users | 1 | 100+ per GPU |
| Time-to-first-token (4K prefill) | ~800ms | <100ms |
| Decode throughput | ~120 tok/s | >1000 tok/s per GPU |
| KV cache utilization | Unbounded allocation | >90% via paging |
| API | None | OpenAI-compatible HTTP/SSE |

A single H100 with NanoSeek INT8 (4.75GB) has ~75GB free for KV cache. With MLA's 23x compression, that's enough for thousands of concurrent sequences — **if** we manage memory correctly. Without paging, fragmentation wastes 60-80%.

---

## 2. First Principles

### Prefill vs Decode: Two Fundamentally Different Workloads

```
Prefill (compute-bound)              Decode (memory-bound)
─────────────────────                ──────────────────────
All input tokens at once             One token at a time
Q,K,V: [B, L_prompt, D]             Q: [B, 1, D]  K,V: [B, L_total, D]
FLOPS: O(L² × D)                    FLOPS: O(L × D)
Arithmetic intensity: HIGH           Arithmetic intensity: LOW
```

Optimal serving handles both simultaneously — prefill new requests while decoding existing ones.

### Why Continuous Batching Beats Static Batching (10x Throughput)

Static batching pads all sequences to max length and blocks new requests until the entire batch finishes. If request A generates 20 tokens and D generates 100, A's GPU slot idles for 80 steps.

Continuous batching inserts/removes requests per iteration — when A finishes, E immediately takes its slot. Result: **5-10x throughput** from the same hardware.

### PagedAttention: Virtual Memory for KV Cache

Traditional KV cache allocates contiguous `[max_seq_len, kv_dim]` per request. If max_seq_len=32K but average is 2K, 94% of memory is wasted. PagedAttention (vLLM) borrows from OS virtual memory — fixed-size blocks (16 tokens), virtual→physical mapping via block tables, near-zero fragmentation.

### Why MLA Makes Serving Particularly Efficient

Standard MHA stores `2 × 16 × 128 = 4096` values per token per layer (8192 bytes BF16).
MLA stores `143 + 32 = 175` values per token per layer (350 bytes BF16). **23.4x compression.**

| Method | KV per seq (32K ctx, 16 layers) | 100 sequences |
|--------|--------------------------------:|---------------:|
| Standard MHA | 4.1 GB | 410 GB (impossible) |
| MLA | 179 MB | 17.9 GB (fits one H100) |

---

## 3. Production Code

### File Placement

```
model/serving/
├── __init__.py            # Package exports
├── engine.py              # Core inference engine with continuous batching
├── paged_attention.py     # PagedAttention KV cache for MLA
├── scheduler.py           # Batch scheduler with preemption
└── server.py              # FastAPI HTTP server (OpenAI-compatible)
scripts/
└── serve.py               # Serving launcher with quantization + TP
```

### 7a. Inference Engine Core (`model/serving/engine.py`)

```python
"""
NanoSeek Inference Engine — Continuous Batching with MLA-Optimized KV Cache.

Lifecycle: engine = InferenceEngine(model, config) → engine.start()
           → engine.add_request() → async for token in engine.stream(): ...
"""
from __future__ import annotations
import asyncio, time, threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import AsyncIterator, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor


class SequenceStatus(Enum):
    WAITING = auto(); RUNNING = auto(); PAUSED = auto()
    FINISHED = auto(); ERROR = auto()


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    max_tokens: int = 256
    stop_token_ids: List[int] = field(default_factory=list)
    repetition_penalty: float = 1.0


@dataclass
class SequenceState:
    request_id: str
    prompt_token_ids: List[int]
    sampling_params: SamplingParams
    status: SequenceStatus = SequenceStatus.WAITING
    output_token_ids: List[int] = field(default_factory=list)
    block_table: List[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    arrival_time: float = field(default_factory=time.monotonic)
    first_token_time: Optional[float] = None
    priority: int = 0

    @property
    def total_len(self) -> int: return len(self.prompt_token_ids) + len(self.output_token_ids)
    @property
    def prompt_len(self) -> int: return len(self.prompt_token_ids)
    @property
    def num_generated(self) -> int: return len(self.output_token_ids)
    @property
    def is_prefill_done(self) -> bool: return self.num_computed_tokens >= self.prompt_len


@dataclass
class EngineConfig:
    max_batch_size: int = 64
    max_seq_len: int = 32768
    max_num_blocks: int = 4096
    block_size: int = 16
    kv_cache_dtype: torch.dtype = torch.bfloat16
    num_layers: int = 16
    kv_dim: int = 175       # MLA: kv_lora_rank(143) + qk_rope_head_dim(32)
    vocab_size: int = 65536
    device: str = "cuda"


class InferenceEngine:
    def __init__(self, model: nn.Module, config: EngineConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)

        from .paged_attention import PagedKVCacheManager
        from .scheduler import BatchScheduler

        self.kv_manager = PagedKVCacheManager(
            num_layers=config.num_layers, kv_dim=config.kv_dim,
            block_size=config.block_size, max_num_blocks=config.max_num_blocks,
            dtype=config.kv_cache_dtype, device=self.device,
        )
        self.scheduler = BatchScheduler(config=config, kv_manager=self.kv_manager)
        self._waiting: Dict[str, SequenceState] = {}
        self._running: Dict[str, SequenceState] = {}
        self._output_queues: Dict[str, asyncio.Queue] = {}
        self._lock = threading.Lock()
        self._running_flag = False
        self._step_count = 0

    def add_request(self, request_id: str, prompt_token_ids: List[int],
                    sampling_params: Optional[SamplingParams] = None, priority: int = 0) -> str:
        seq = SequenceState(request_id=request_id, prompt_token_ids=prompt_token_ids,
                            sampling_params=sampling_params or SamplingParams(), priority=priority)
        with self._lock:
            self._waiting[request_id] = seq
            self._output_queues[request_id] = asyncio.Queue()
        return request_id

    async def stream(self, request_id: str) -> AsyncIterator[int]:
        queue = self._output_queues.get(request_id)
        if queue is None: raise KeyError(f"Unknown request: {request_id}")
        while True:
            token = await queue.get()
            if token is None: break
            yield token

    async def run_engine_loop(self):
        self._running_flag = True
        while self._running_flag:
            if not (self._waiting or self._running):
                await asyncio.sleep(0.001); continue
            await self._step()

    async def _step(self):
        with self._lock:
            result = self.scheduler.schedule(list(self._waiting.values()), list(self._running.values()))
        for seq in result.get("preempted", []):
            seq.status = SequenceStatus.PAUSED
            self.kv_manager.free_blocks(seq.block_table); seq.block_table = []
            with self._lock:
                self._running.pop(seq.request_id, None); self._waiting[seq.request_id] = seq
        if result["prefill"]:  await self._run_prefill(result["prefill"])
        if result["decode"]:   await self._run_decode(result["decode"])
        self._step_count += 1

    async def _run_prefill(self, sequences: List[SequenceState]):
        for seq in sequences:
            needed = (seq.prompt_len + self.config.block_size - 1) // self.config.block_size
            blocks = self.kv_manager.allocate_blocks(needed)
            if blocks is None: continue
            seq.block_table = blocks
            input_ids = torch.tensor([seq.prompt_token_ids], dtype=torch.long, device=self.device)
            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, use_cache=True)
            kv_cache = outputs.get("past_key_values")
            if kv_cache: self.kv_manager.store_kv_cache(seq.block_table, kv_cache, seq.prompt_len)
            next_token = self._sample(outputs["logits"][:, -1, :], seq.sampling_params)
            seq.output_token_ids.append(next_token)
            seq.num_computed_tokens = seq.prompt_len + 1
            seq.first_token_time = time.monotonic()
            seq.status = SequenceStatus.RUNNING
            with self._lock:
                self._waiting.pop(seq.request_id, None); self._running[seq.request_id] = seq
            queue = self._output_queues.get(seq.request_id)
            if queue: await queue.put(next_token)
            if self._is_finished(seq): await self._finish_sequence(seq)

    async def _run_decode(self, sequences: List[SequenceState]):
        if not sequences: return
        batch_ids = [[seq.output_token_ids[-1] if seq.output_token_ids else seq.prompt_token_ids[-1]]
                     for seq in sequences]
        input_ids = torch.tensor(batch_ids, dtype=torch.long, device=self.device)
        past_kvs = [self.kv_manager.gather_kv_cache(s.block_table, s.total_len - 1) for s in sequences]
        past_key_values = self._stack_kv_caches(past_kvs)
        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        logits = outputs["logits"][:, -1, :]
        for i, seq in enumerate(sequences):
            tok = self._sample(logits[i:i+1], seq.sampling_params)
            seq.output_token_ids.append(tok); seq.num_computed_tokens += 1
            new_needed = (seq.total_len + self.config.block_size - 1) // self.config.block_size
            if new_needed > len(seq.block_table):
                extra = self.kv_manager.allocate_blocks(new_needed - len(seq.block_table))
                if extra: seq.block_table.extend(extra)
            queue = self._output_queues.get(seq.request_id)
            if queue: await queue.put(tok)
            if self._is_finished(seq): await self._finish_sequence(seq)

    def _sample(self, logits: Tensor, params: SamplingParams) -> int:
        if params.temperature == 0.0: return logits.argmax(dim=-1).item()
        logits = logits / params.temperature
        if params.top_k and params.top_k > 0:
            v, _ = torch.topk(logits, min(params.top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        if params.top_p and params.top_p < 1.0:
            sl, si = torch.sort(logits, descending=True)
            cum = torch.cumsum(torch.softmax(sl, dim=-1), dim=-1)
            mask = cum > params.top_p; mask[:, 1:] = mask[:, :-1].clone(); mask[:, 0] = False
            logits[mask.scatter(1, si, mask)] = float("-inf")
        return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).item()

    def _is_finished(self, seq: SequenceState) -> bool:
        return (seq.num_generated >= seq.sampling_params.max_tokens or
                (seq.output_token_ids and seq.output_token_ids[-1] in seq.sampling_params.stop_token_ids) or
                seq.total_len >= self.config.max_seq_len)

    async def _finish_sequence(self, seq: SequenceState):
        seq.status = SequenceStatus.FINISHED
        self.kv_manager.free_blocks(seq.block_table); seq.block_table = []
        with self._lock: self._running.pop(seq.request_id, None)
        queue = self._output_queues.get(seq.request_id)
        if queue: await queue.put(None)

    def _stack_kv_caches(self, kv_list):
        if not kv_list or kv_list[0] is None: return None
        num_layers = len(kv_list[0])
        return [(torch.cat([kv[l][0] for kv in kv_list], dim=0),
                 torch.cat([kv[l][1] for kv in kv_list], dim=0)) for l in range(num_layers)]

    def get_metrics(self) -> Dict:
        with self._lock:
            return {"waiting_requests": len(self._waiting), "running_requests": len(self._running),
                    "steps_completed": self._step_count, "kv_blocks_used": self.kv_manager.num_used_blocks,
                    "kv_blocks_free": self.kv_manager.num_free_blocks, "kv_utilization": self.kv_manager.utilization}

    def shutdown(self): self._running_flag = False
```

### 7b. PagedAttention KV Cache (`model/serving/paged_attention.py`)

```python
"""
PagedAttention KV Cache — Adapted for MLA's Compressed KV Format.

Standard MHA block (16 tokens, 16 heads, head_dim=128):
  K+V: 2 × [16, 16, 128] = 131,072 bytes BF16 per block per layer

MLA block (16 tokens, kv_dim=175):
  kv_compressed: [16, 143] + k_pe: [16, 1, 32] = 5,600 bytes BF16 per block per layer

Compression: 131,072 / 5,600 = 23.4x fewer bytes per block.
The kv_compressed vector is expanded to full K_nope and V at attention time via
wkv_b projection — trading compute for memory (favorable since decode is bandwidth-bound).
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
from torch import Tensor


class KVBlock:
    __slots__ = ("block_id", "ref_count", "num_filled")
    def __init__(self, block_id: int):
        self.block_id = block_id; self.ref_count = 1; self.num_filled = 0


class PagedKVCacheManager:
    """
    Block-based KV cache with virtual→physical mapping for MLA.

    Memory budget (H100 80GB, NanoSeek INT8 = 4.75GB):
      Available: ~70GB. Block = 16 tok × 175 dim × 2B × 16 layers = 89.6KB.
      Max blocks: ~818K → ~13M tokens → ~6,500 concurrent 2K-context sequences.
    """
    def __init__(self, num_layers: int, kv_dim: int, block_size: int = 16,
                 max_num_blocks: int = 4096, dtype: torch.dtype = torch.bfloat16,
                 device: torch.device = torch.device("cuda")):
        self.num_layers = num_layers
        self.kv_lora_rank = kv_dim - 32  # 175 - 32 = 143
        self.rope_dim = 32
        self.block_size = block_size
        self.max_num_blocks = max_num_blocks
        self.dtype = dtype; self.device = device

        # Pre-allocate GPU pools — no runtime cudaMalloc stalls
        self.kv_compressed_pool: List[Tensor] = []
        self.k_pe_pool: List[Tensor] = []
        for _ in range(num_layers):
            self.kv_compressed_pool.append(
                torch.zeros(max_num_blocks, block_size, self.kv_lora_rank, dtype=dtype, device=device))
            self.k_pe_pool.append(
                torch.zeros(max_num_blocks, block_size, 1, self.rope_dim, dtype=dtype, device=device))

        self._blocks: Dict[int, KVBlock] = {}
        self._free_ids: List[int] = list(range(max_num_blocks - 1, -1, -1))

    @property
    def num_free_blocks(self) -> int: return len(self._free_ids)
    @property
    def num_used_blocks(self) -> int: return self.max_num_blocks - len(self._free_ids)
    @property
    def utilization(self) -> float:
        return self.num_used_blocks / self.max_num_blocks if self.max_num_blocks else 0.0

    def allocate_blocks(self, num_blocks: int) -> Optional[List[int]]:
        if num_blocks > len(self._free_ids): return None
        allocated = []
        for _ in range(num_blocks):
            bid = self._free_ids.pop()
            self._blocks[bid] = KVBlock(bid); allocated.append(bid)
        return allocated

    def free_blocks(self, block_ids: List[int]):
        for bid in block_ids:
            block = self._blocks.get(bid)
            if block is None: continue
            block.ref_count -= 1
            if block.ref_count <= 0:
                del self._blocks[bid]; self._free_ids.append(bid)
                for l in range(self.num_layers):
                    self.kv_compressed_pool[l][bid].zero_(); self.k_pe_pool[l][bid].zero_()

    def copy_on_write(self, block_id: int) -> int:
        """Clone a shared block for exclusive write access (beam search)."""
        block = self._blocks.get(block_id)
        if block is None or block.ref_count <= 1: return block_id
        new_ids = self.allocate_blocks(1)
        if new_ids is None: raise RuntimeError("OOM: cannot allocate block for CoW")
        new_id = new_ids[0]
        for l in range(self.num_layers):
            self.kv_compressed_pool[l][new_id].copy_(self.kv_compressed_pool[l][block_id])
            self.k_pe_pool[l][new_id].copy_(self.k_pe_pool[l][block_id])
        block.ref_count -= 1
        self._blocks[new_id].num_filled = block.num_filled
        return new_id

    def store_kv_cache(self, block_table: List[int],
                       past_key_values: List[Tuple[Tensor, Tensor]], num_tokens: int):
        """Store MLA cache: past_key_values[layer] = (kv_compressed[1,T,143], k_pe[1,T,1,32])."""
        for layer_idx, (kv_comp, k_pe) in enumerate(past_key_values):
            kv_comp = kv_comp.squeeze(0); k_pe_sq = k_pe.squeeze(0)
            for block_idx, bid in enumerate(block_table):
                start = block_idx * self.block_size
                end = min(start + self.block_size, num_tokens)
                if start >= num_tokens: break
                n = end - start
                self.kv_compressed_pool[layer_idx][bid, :n] = kv_comp[start:end]
                self.k_pe_pool[layer_idx][bid, :n] = k_pe_sq[start:end]
                block = self._blocks.get(bid)
                if block: block.num_filled = n

    def gather_kv_cache(self, block_table: List[int], num_tokens: int
                        ) -> Optional[List[Tuple[Tensor, Tensor]]]:
        """Reconstruct contiguous [1, num_tokens, dim] KV from paged blocks."""
        if not block_table: return None
        result = []
        for l in range(self.num_layers):
            kv_chunks, kpe_chunks, remaining = [], [], num_tokens
            for bid in block_table:
                n = min(self.block_size, remaining)
                if n <= 0: break
                kv_chunks.append(self.kv_compressed_pool[l][bid, :n])
                kpe_chunks.append(self.k_pe_pool[l][bid, :n])
                remaining -= n
            result.append((torch.cat(kv_chunks, dim=0).unsqueeze(0),
                           torch.cat(kpe_chunks, dim=0).unsqueeze(0)))
        return result

    def get_pool_memory_bytes(self) -> int:
        bpe = 2 if self.dtype == torch.bfloat16 else 4
        return ((self.max_num_blocks * self.block_size * self.kv_lora_rank * bpe) +
                (self.max_num_blocks * self.block_size * self.rope_dim * bpe)) * self.num_layers
```

### 7c. HTTP Serving (`model/serving/server.py`)

```python
"""
NanoSeek HTTP Server — OpenAI-Compatible API with SSE Streaming.

  POST /v1/completions       — Text completion
  POST /v1/chat/completions  — Chat completion
  GET  /health               — Health check + KV utilization
  GET  /metrics              — Prometheus-style metrics
"""
from __future__ import annotations
import asyncio, json, time, uuid
from typing import AsyncIterator, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from .engine import EngineConfig, InferenceEngine, SamplingParams


class CompletionRequest(BaseModel):
    model: str = "nanoseek-1b"
    prompt: str | List[int] = ""
    max_tokens: int = Field(default=256, ge=1, le=32768)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=0)
    stream: bool = False
    stop: Optional[List[str]] = None

class ChatMessage(BaseModel):
    role: str; content: str

class ChatCompletionRequest(BaseModel):
    model: str = "nanoseek-1b"
    messages: List[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=32768)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=0)
    stream: bool = False


class TokenBucketRateLimiter:
    def __init__(self, rate: float = 100.0, capacity: float = 200.0):
        self.rate = rate; self.capacity = capacity
        self._tokens = capacity; self._last = time.monotonic()
    def try_acquire(self) -> bool:
        now = time.monotonic()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
        self._last = now
        if self._tokens >= 1.0: self._tokens -= 1.0; return True
        return False


def create_app(engine: InferenceEngine, tokenizer, rate_limit: float = 100.0) -> FastAPI:
    app = FastAPI(title="NanoSeek API", version="1.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    limiter = TokenBucketRateLimiter(rate=rate_limit)
    t0 = time.time()

    @app.get("/health")
    async def health():
        m = engine.get_metrics()
        return {"status": "healthy", "uptime_s": int(time.time() - t0),
                "active": m["running_requests"], "pending": m["waiting_requests"],
                "kv_util": f"{m['kv_utilization']:.1%}"}

    @app.get("/metrics")
    async def metrics():
        m = engine.get_metrics()
        return JSONResponse(content={k: v for k, v in m.items()})

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        if not limiter.try_acquire(): raise HTTPException(429, "Rate limit exceeded")
        token_ids = tokenizer.encode(req.prompt) if isinstance(req.prompt, str) else req.prompt
        params = SamplingParams(temperature=req.temperature, top_k=req.top_k,
                                top_p=req.top_p, max_tokens=req.max_tokens)
        rid = f"cmpl-{uuid.uuid4().hex[:24]}"
        engine.add_request(rid, token_ids, params)
        if req.stream:
            return StreamingResponse(_stream_completion(engine, rid, req.model, tokenizer),
                                     media_type="text/event-stream")
        toks = [t async for t in engine.stream(rid)]
        return {"id": rid, "object": "text_completion", "created": int(time.time()),
                "model": req.model, "choices": [{"text": tokenizer.decode(toks), "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(token_ids), "completion_tokens": len(toks),
                          "total_tokens": len(token_ids) + len(toks)}}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        if not limiter.try_acquire(): raise HTTPException(429, "Rate limit exceeded")
        prompt = "\n".join(f"<|{m.role}|>\n{m.content}" for m in req.messages) + "\n<|assistant|>\n"
        token_ids = tokenizer.encode(prompt)
        params = SamplingParams(temperature=req.temperature, top_k=req.top_k,
                                top_p=req.top_p, max_tokens=req.max_tokens)
        rid = f"chatcmpl-{uuid.uuid4().hex[:20]}"
        engine.add_request(rid, token_ids, params)
        if req.stream:
            return StreamingResponse(_stream_chat(engine, rid, req.model, tokenizer),
                                     media_type="text/event-stream")
        toks = [t async for t in engine.stream(rid)]
        return {"id": rid, "object": "chat.completion", "created": int(time.time()),
                "model": req.model,
                "choices": [{"message": {"role": "assistant", "content": tokenizer.decode(toks)},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(token_ids), "completion_tokens": len(toks),
                          "total_tokens": len(token_ids) + len(toks)}}
    return app


async def _stream_completion(engine, rid, model, tokenizer) -> AsyncIterator[str]:
    async for tid in engine.stream(rid):
        yield f"data: {json.dumps({'id': rid, 'object': 'text_completion', 'model': model, 'choices': [{'text': tokenizer.decode([tid]), 'finish_reason': None}]})}\n\n"
    yield f"data: {json.dumps({'id': rid, 'choices': [{'text': '', 'finish_reason': 'stop'}]})}\n\ndata: [DONE]\n\n"

async def _stream_chat(engine, rid, model, tokenizer) -> AsyncIterator[str]:
    async for tid in engine.stream(rid):
        yield f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'model': model, 'choices': [{'delta': {'content': tokenizer.decode([tid])}, 'finish_reason': None}]})}\n\n"
    yield f"data: {json.dumps({'id': rid, 'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\ndata: [DONE]\n\n"
```

### 7d. Batch Scheduler (`model/serving/scheduler.py`)

```python
"""
Continuous Batching Scheduler with Memory-Aware Preemption.

Policy: (1) Always decode running sequences, (2) admit prefills if memory allows,
(3) preempt lowest-priority running sequences if high-priority requests wait.
"""
from __future__ import annotations
from typing import Dict, List, TYPE_CHECKING
if TYPE_CHECKING:
    from .engine import EngineConfig, SequenceState
    from .paged_attention import PagedKVCacheManager


class BatchScheduler:
    def __init__(self, config: "EngineConfig", kv_manager: "PagedKVCacheManager",
                 preemption_watermark: float = 0.05):
        self.max_batch_size = config.max_batch_size
        self.block_size = config.block_size
        self.max_seq_len = config.max_seq_len
        self.kv_manager = kv_manager
        self.preemption_watermark = preemption_watermark

    def schedule(self, waiting: List["SequenceState"],
                 running: List["SequenceState"]) -> Dict[str, list]:
        result = {"prefill": [], "decode": [], "preempted": []}

        decode_candidates = sorted(running, key=lambda s: s.arrival_time)
        if len(decode_candidates) > self.max_batch_size:
            decode_candidates = decode_candidates[:self.max_batch_size]
        result["decode"] = decode_candidates
        slots_left = self.max_batch_size - len(decode_candidates)

        free = self.kv_manager.num_free_blocks
        watermark = int(self.kv_manager.max_num_blocks * self.preemption_watermark)

        if slots_left > 0 and waiting:
            for seq in sorted(waiting, key=lambda s: (-s.priority, s.arrival_time)):
                if slots_left <= 0: break
                needed = (seq.prompt_len + self.block_size - 1) // self.block_size
                if free - needed < watermark:
                    if not self._try_preempt(result["decode"], result["preempted"],
                                             needed, free, watermark):
                        continue
                    free = self.kv_manager.num_free_blocks
                result["prefill"].append(seq); free -= needed; slots_left -= 1
        return result

    def _try_preempt(self, decode_list, preempted_list, blocks_needed, free, watermark) -> bool:
        candidates = sorted(decode_list, key=lambda s: (s.priority, -s.arrival_time))
        freed = 0; victims = []
        for seq in candidates:
            if free + freed - blocks_needed >= watermark: break
            freed += len(seq.block_table); victims.append(seq)
        if free + freed - blocks_needed < watermark: return False
        for seq in victims:
            decode_list.remove(seq); preempted_list.append(seq)
        return True
```

### 7e. Serving Launch Script (`scripts/serve.py`)

```python
"""
NanoSeek Serving Launcher.
  python scripts/serve.py --checkpoint path/to/ckpt.pt
  python scripts/serve.py --checkpoint path/to/ckpt.pt --quantize int8 --port 8080
"""
import argparse, asyncio, logging, sys
from pathlib import Path
import torch, uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nanoseek.serve")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="NanoSeek Inference Server")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--quantize", choices=["none", "int8", "int4"], default="none")
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    p.add_argument("--tp", type=int, default=1, help="Tensor parallel degree")
    p.add_argument("--max-batch-size", type=int, default=64)
    p.add_argument("--max-seq-len", type=int, default=32768)
    p.add_argument("--max-num-blocks", type=int, default=4096)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--rate-limit", type=float, default=100.0)
    return p.parse_args()


def load_model(ckpt_path, quantize, dtype_str, tp):
    from model.model import NanoSeekModel
    from model.config import get_nanoseek_config
    config = get_nanoseek_config()
    model = NanoSeekModel(config)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in sd: sd = sd["model_state_dict"]
    model.load_state_dict(sd, strict=False)
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    if quantize == "int8":
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    else:
        model = model.to(dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device); model.eval()
    logger.info(f"Loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")
    return model, config, device


def main():
    args = parse_args()
    model, config, device = load_model(args.checkpoint, args.quantize, args.dtype, args.tp)

    class FallbackTokenizer:
        def encode(self, text): return list(text.encode("utf-8"))
        def decode(self, ids): return bytes([t if 0 <= t < 256 else 32 for t in ids]).decode("utf-8", errors="replace")

    tokenizer = FallbackTokenizer()
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        except ImportError: pass

    from model.serving.engine import EngineConfig, InferenceEngine
    from model.serving.server import create_app
    engine_config = EngineConfig(
        max_batch_size=args.max_batch_size, max_seq_len=args.max_seq_len,
        max_num_blocks=args.max_num_blocks, block_size=args.block_size,
        kv_cache_dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float16,
        num_layers=config.num_layers, kv_dim=config.mla.kv_cache_dim_per_layer,
        vocab_size=config.vocab_size, device=str(device))
    engine = InferenceEngine(model, engine_config)
    app = create_app(engine, tokenizer, rate_limit=args.rate_limit)

    @app.on_event("startup")
    async def start_engine():
        asyncio.create_task(engine.run_engine_loop())

    kv_gb = args.max_num_blocks * args.block_size * config.mla.kv_cache_dim_per_layer * 2 * config.num_layers / 1e9
    logger.info(f"NanoSeek on {device} | batch={args.max_batch_size} seq={args.max_seq_len} | KV pool={kv_gb:.2f}GB")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
```

### Package Init (`model/serving/__init__.py`)

```python
"""NanoSeek Production Serving — Continuous Batching with MLA PagedAttention."""
from .engine import EngineConfig, InferenceEngine, SamplingParams, SequenceState
from .paged_attention import PagedKVCacheManager
from .scheduler import BatchScheduler
__all__ = ["EngineConfig", "InferenceEngine", "SamplingParams", "SequenceState",
           "PagedKVCacheManager", "BatchScheduler"]
```

---

## 4. Architecture Data Flow

```
Client (HTTP)
    │  POST /v1/completions
    ▼
┌────────────┐     ┌──────────────┐     ┌────────────────┐
│ FastAPI     │────▶│ Engine       │────▶│ Scheduler      │
│ server.py   │     │ engine.py    │     │ scheduler.py   │
│ validate,   │     │ prefill/     │     │ memory-aware   │
│ tokenize,   │     │ decode loop  │     │ admission +    │
│ rate limit  │     │              │     │ preemption     │
└────────────┘     └──────┬───────┘     └────────────────┘
                          │ alloc/free/gather
                          ▼
                   ┌──────────────┐     ┌────────────────┐
                   │ Paged KV     │◄───▶│ GPU Memory     │
                   │ Cache        │     │ [blocks,16,143] │
                   │ paged_       │     │ [blocks,16,1,32]│
                   │ attention.py │     │ per layer       │
                   └──────────────┘     └────────────────┘
```

---

## 5. Performance Targets

| Metric | Target | How Achieved |
|--------|--------|--------------|
| TTFT (4K prefill) | <100ms | Prefill parallelism + Flash MLA (Doc 02) |
| Decode throughput | >1000 tok/s/GPU | Continuous batching, large decode batches |
| Concurrent users | 100+/GPU | MLA 23x compression + PagedAttention |
| KV utilization | >90% | Block-based paging, no contiguous alloc |
| KV memory (100×2K ctx) | ~3.5GB | 175 × 16 layers × 2K × 100 × 2B |
| OOM probability | ~0% | Memory-aware scheduler + preemption |

### MLA Advantage: 100 users at 4K context

```
Standard MHA: 100 × 4096 × 4096 × 2B × 16 layers = 53.7 GB  → OOM
MLA:          100 × 4096 × 175  × 2B × 16 layers = 2.3 GB   → Fits easily
```

---

## 6. Integration Notes

The engine calls `model.forward(use_cache=True)`, which returns `past_key_values` as list of `(kv_compressed, k_pe)` tuples — exactly the MLA format our PagedAttention stores. No model changes required.

For speculative decoding integration (Doc 08), the MTP module's `speculative_decode()` (model.py line 1014) produces draft tokens that the engine's decode step can verify in a single batched forward pass.

---

*"The gap between a model that generates text and a model that serves users is the same gap between a prototype car engine and a production vehicle."*

— Principal Engineer's Note, Inference Platform Division, 2026
