# 13 — Production Integration Guide: Wiring Docs 01–10 into the NanoSeek Codebase

## Principal Engineer's Integration Blueprint — February 2026

**Author**: Distinguished AI Researcher & Principal Engineer, Foundation Models Division
**Scope**: Exact file-level, line-level integration instructions for every production component
**Prerequisite**: Docs 01–10 implemented as standalone modules; this doc wires them into the live codebase

---

## 1. Integration Overview

### 1a. What Each Implementation Doc Produces

| Doc | Title | Artifact Produced | Primary Target File(s) |
|-----|-------|-------------------|----------------------|
| 01 | Data Pipeline Production | `scripts/dataloader_v2.py` — prefetching, async tokenization, packing | `scripts/pre-train.py` lines 156–169 (imports), 1158–1213 (loader init) |
| 02 | Flash MLA Attention Kernel | `model/kernels/flash_mla.py` — Triton fused Q×K, softmax, ×V | `model/model.py` lines 352–367 (`MultiHeadLatentAttention.forward`) |
| 03 | Fused MoE Expert Parallelism | `model/kernels/fused_moe.py` — Triton grouped GEMM dispatch | `model/model.py` lines 548–565 (`token_centric_dispatch` expert loop) |
| 04 | Distributed Training (FSDP) | `model/distributed/fsdp_utils.py` — FSDP wrapping, expert sharding | `scripts/pre-train.py` lines 1002–1019 (model init), 1146–1148 (DDP) |
| 05 | FP8 Mixed Precision | `model/fp8.py` — FP8Linear, block quantization | `model/config.py` (FP8Config at line 156), `model/model.py` (linear layers) |
| 06 | Multi-Phase Orchestration | Upgrades to `scripts/pre-train.py` — robust phase transitions | `scripts/pre-train.py` lines 1068–1131 (phase logic) |
| 07 | Inference Engine | `model/serving/engine.py` — continuous batching, KV cache mgmt | New `model/serving/` directory |
| 08 | Speculative Decoding (MTP) | `model/serving/speculative.py` — draft/verify pipeline | `model/model.py` lines 1014–1049 (`speculative_decode`), `model/serving/` |
| 09 | Post-Training (SFT/DPO) | `scripts/sft.py`, `scripts/dpo.py`, `model/lora.py` | New scripts + new module |
| 10 | Evaluation Benchmarks | `model/eval/benchmarks/`, `scripts/evaluate.py` | `model/eval/` directory extension |

### 1b. Dependency Order (Critical Path)

Integration MUST follow this order due to import and runtime dependencies:

```
Phase 1: Flash MLA (Doc 02) ─────────────────────────────────────── standalone kernel
Phase 2: Fused MoE (Doc 03) ─────────────────────────────────────── standalone kernel
Phase 3: Data Pipeline (Doc 01) ─────────────────────────────────── no model deps
Phase 4: FSDP (Doc 04) ──────── depends on model init path ──────── after Phase 1+2
Phase 5: FP8 (Doc 05) ───────── depends on FSDP wrapping ────────── after Phase 4
Phase 6: Inference (Docs 07+08) ─ depends on kernels ─────────────── after Phase 1+2
Phase 7: Post-Training (Doc 09) ─ depends on FSDP + data pipeline ── after Phase 3+4
Phase 8: Evaluation (Doc 10) ──── depends on inference engine ─────── after Phase 6
```

### 1c. Estimated Integration Effort

| Phase | Effort | Risk | Reason |
|-------|--------|------|--------|
| 1. Flash MLA | 2 days | Medium | Triton kernel correctness; numerical parity with `torch.matmul` path |
| 2. Fused MoE | 3 days | High | Grouped GEMM is perf-sensitive; expert count boundary conditions |
| 3. Data Pipeline | 1 day | Low | Drop-in replacement; same yield signature |
| 4. FSDP | 3 days | High | State dict key renaming (`_orig_mod.`); MoE sharding policy |
| 5. FP8 | 2 days | Medium | Requires H100; loss scaling edge cases |
| 6. Inference | 3 days | Medium | Continuous batching correctness; memory management |
| 7. Post-Training | 2 days | Low | Standard SFT/DPO; LoRA is well-understood |
| 8. Evaluation | 1 day | Low | Orchestration; no model changes |

**Total: ~17 engineer-days for complete integration.**

---

## 2. Phase 1: Flash MLA Integration (from Doc 02)

### 2a. Create kernel directory structure

```bash
mkdir -p model/kernels
touch model/kernels/__init__.py
```

`model/kernels/__init__.py` contents:

```python
"""NanoSeek production CUDA/Triton kernels."""
try:
    from .flash_mla import flash_mla_attention, flash_mla_available
except ImportError:
    flash_mla_available = lambda: False

try:
    from .fused_moe import fused_moe_forward, fused_moe_available
except ImportError:
    fused_moe_available = lambda: False
```

### 2b. Create `model/kernels/flash_mla.py`

This file is produced by Doc 02. The key export is `flash_mla_attention(q, k, v, causal_mask, softmax_scale) -> Tensor`.

### 2c. Modify `MultiHeadLatentAttention.forward()` — `model/model.py` lines 352–370

This is the core integration point. The attention computation currently uses standard `torch.matmul`:

**BEFORE** (`model/model.py` lines 352–370):
```python
        # Attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.softmax_scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        if self.training and self.attention_dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        output = self.wo(attn_output)
```

**AFTER** (replace lines 352–370):
```python
        # Attention
        q = q.transpose(1, 2)  # [B, H, S, D]
        k = k.transpose(1, 2)  # [B, H, KV, D]
        v = v.transpose(1, 2)  # [B, H, KV, Dv]

        # Try Flash MLA kernel for fused attention (2-3x speedup on H100)
        _use_flash_mla = False
        if not self.training or self.attention_dropout == 0:
            try:
                from model.kernels.flash_mla import flash_mla_attention, flash_mla_available
                _use_flash_mla = flash_mla_available()
            except ImportError:
                pass

        if _use_flash_mla:
            # Fused kernel: Q×K scaling, causal masking, softmax, ×V in one pass
            # Eliminates O(S²) materialized attention matrix
            attn_output = flash_mla_attention(
                q, k, v,
                softmax_scale=self.softmax_scale,
                causal=attention_mask is None,  # use built-in causal when no explicit mask
            )
        else:
            # Fallback: standard PyTorch attention (always correct, used for CPU/testing)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.softmax_scale

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

            if self.training and self.attention_dropout > 0:
                attn_weights = F.dropout(attn_weights, p=self.attention_dropout)

            attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        output = self.wo(attn_output)
```

**Why this is correct:**
1. The kernel produces identical output to the `matmul → softmax → matmul` path (verified by Doc 02's numerical tests)
2. The `flash_mla_available()` guard ensures we only use the kernel on supported hardware (Triton + CUDA)
3. The fallback preserves 100% backward compatibility — CPU tests, MPS, and non-Triton environments are unaffected
4. Dropout is only applied during training; Flash MLA handles causal masking internally

**How to test:**
```bash
# Verify no regression on existing tests:
pytest tests/test_mla.py -v

# Verify numerical parity (new test from Doc 02):
pytest tests/test_kernels.py::test_flash_mla_numerical_parity -v

# Benchmark speedup:
python -m model.kernels.flash_mla --benchmark
```

### 2d. Also update `DSASparseAttention._sparse_forward()` — `model/model.py` lines 1317–1338

The sparse attention path in DSA has its own matmul-based attention at lines 1321–1338. This should also benefit from kernel acceleration, but the gather-based indexing makes direct Flash MLA integration non-trivial. **Recommendation**: Keep the sparse path using standard attention initially; optimize in a follow-up after Phase 1 is stable.

---

## 3. Phase 2: Fused MoE Integration (from Doc 03)

### 3a. Create `model/kernels/fused_moe.py`

This file is produced by Doc 03. The key export is:
```python
def fused_moe_forward(
    x: Tensor,           # [N, D] - flattened tokens
    indices: Tensor,      # [N, K] - expert assignments
    weights: Tensor,      # [N, K] - routing weights
    expert_weights: List[Tuple[Tensor, Tensor, Tensor]],  # (gate, up, down) per expert
) -> Tensor:             # [N, D] - output
```

### 3b. Modify `token_centric_dispatch()` — `model/model.py` lines 485–583

The current Python implementation loops over experts sequentially at lines 554–562:

**BEFORE** (`model/model.py` lines 548–583):
```python
    # Split permuted input by expert counts
    expert_batches = torch.split(permuted_input, expert_counts)

    # Process each expert's batch (contiguous memory access!)
    expert_outputs = []
    for expert_id, batch in enumerate(expert_batches):
        if batch.shape[0] > 0:
            expert_outputs.append(experts[expert_id](batch))
        else:
            # Empty batch - append empty tensor to maintain alignment
            expert_outputs.append(
                torch.empty(0, D, device=device, dtype=dtype)
            )

    # Concatenate all expert outputs
    permuted_output = torch.cat(expert_outputs, dim=0)           # [N×K, D]

    # ─────────────────────────────────────────────────────────
    # Phase 3: UNPERMUTE - Combine back to original positions
    # ─────────────────────────────────────────────────────────

    # Apply routing weights
    weighted_output = permuted_output * sorted_weights.unsqueeze(-1)

    # Scatter-add back to original token positions
    # Each token accumulates weighted contributions from all its experts
    output = torch.zeros(N, D, device=device, dtype=dtype)
    output.scatter_add_(
        dim=0,
        index=sorted_token_ids.unsqueeze(-1).expand(-1, D),
        src=weighted_output
    )

    return output
```

**AFTER** (replace the entire `token_centric_dispatch` function body after the Phase 1 PERMUTE section, starting at line 548):
```python
    # ─────────────────────────────────────────────────────────
    # Phase 2: COMPUTE - Fused or sequential expert execution
    # ─────────────────────────────────────────────────────────

    _use_fused = False
    try:
        from model.kernels.fused_moe import fused_moe_forward, fused_moe_available
        _use_fused = fused_moe_available() and permuted_input.is_cuda
    except ImportError:
        pass

    if _use_fused:
        # Fused Triton grouped GEMM: all experts in one kernel launch
        # Eliminates Python loop overhead and enables coalesced memory access
        # across expert boundaries. Speedup: 3-8x over sequential dispatch.
        expert_weight_tuples = [
            (expert.gate_proj.weight, expert.up_proj.weight, expert.down_proj.weight)
            for expert in experts
        ]
        permuted_output = fused_moe_forward(
            permuted_input, sorted_expert_ids, expert_counts, expert_weight_tuples
        )
    else:
        # Fallback: sequential Python dispatch (CPU/testing compatible)
        expert_batches = torch.split(permuted_input, expert_counts)
        expert_outputs = []
        for expert_id, batch in enumerate(expert_batches):
            if batch.shape[0] > 0:
                expert_outputs.append(experts[expert_id](batch))
            else:
                expert_outputs.append(
                    torch.empty(0, D, device=device, dtype=dtype)
                )
        permuted_output = torch.cat(expert_outputs, dim=0)

    # ─────────────────────────────────────────────────────────
    # Phase 3: UNPERMUTE - Combine back to original positions
    # ─────────────────────────────────────────────────────────

    # Apply routing weights
    weighted_output = permuted_output * sorted_weights.unsqueeze(-1)

    # Scatter-add back to original token positions
    output = torch.zeros(N, D, device=device, dtype=dtype)
    output.scatter_add_(
        dim=0,
        index=sorted_token_ids.unsqueeze(-1).expand(-1, D),
        src=weighted_output
    )

    return output
```

**Why this is correct:**
1. The fused kernel receives the SAME sorted/permuted inputs and produces the SAME output tensor shape
2. Expert weight extraction uses the existing `SwiGLUFFN` attribute names (`gate_proj`, `up_proj`, `down_proj`) defined at lines 458–460
3. The `fused_moe_available()` guard checks for Triton + CUDA availability
4. Fallback preserves the exact original Python loop for CPU/MPS/testing

**How to test:**
```bash
# Existing MoE tests (no regression):
pytest tests/test_moe.py tests/test_moe_standalone.py -v

# Numerical parity test (new, from Doc 03):
pytest tests/test_kernels.py::test_fused_moe_parity -v

# Benchmark:
python -m model.kernels.fused_moe --benchmark --num_experts=64 --active=8
```

---

## 4. Phase 3: Production Data Pipeline (from Doc 01)

### 4a. Create `scripts/dataloader_v2.py`

Doc 01 produces a production data pipeline with:
- Async prefetching (overlaps I/O with compute)
- Multi-worker tokenization
- Document packing (eliminates padding waste)
- Proper distributed sharding with deterministic resume

### 4b. Modify `scripts/pre-train.py` import section — lines 155–170

**BEFORE** (`scripts/pre-train.py` lines 155–170):
```python
# Streaming Dataloader (for real training data)
try:
    from scripts.dataloader import (
        tokenizing_distributed_data_loader_with_state,
        tokenizing_distributed_data_loader,
    )
    DATALOADER_AVAILABLE = True
except ImportError:
    try:
        from dataloader import (
            tokenizing_distributed_data_loader_with_state,
            tokenizing_distributed_data_loader,
        )
        DATALOADER_AVAILABLE = True
    except ImportError:
        DATALOADER_AVAILABLE = False
        print("Warning: Real dataloader not available, using dummy data")
```

**AFTER**:
```python
# Production Dataloader (async prefetch + packing)
try:
    from scripts.dataloader_v2 import (
        create_train_dataloader,
        create_val_dataloader,
        DataloaderConfig,
    )
    DATALOADER_V2_AVAILABLE = True
except ImportError:
    DATALOADER_V2_AVAILABLE = False

# Fallback: Original streaming dataloader
if not DATALOADER_V2_AVAILABLE:
    try:
        from scripts.dataloader import (
            tokenizing_distributed_data_loader_with_state,
            tokenizing_distributed_data_loader,
        )
        DATALOADER_AVAILABLE = True
    except ImportError:
        try:
            from dataloader import (
                tokenizing_distributed_data_loader_with_state,
                tokenizing_distributed_data_loader,
            )
            DATALOADER_AVAILABLE = True
        except ImportError:
            DATALOADER_AVAILABLE = False
            print("Warning: Real dataloader not available, using dummy data")
else:
    DATALOADER_AVAILABLE = True  # V2 subsumes V1
```

### 4c. Modify `scripts/pre-train.py` data loading initialization — lines 1158–1213

**BEFORE** (`scripts/pre-train.py` lines 1158–1213):
```python
    if DATALOADER_AVAILABLE:
        print0("  Using streaming dataloader from parquet files")

        def build_train_loader(resume_state=None):
            """Build streaming train dataloader with optional resume state."""
            return tokenizing_distributed_data_loader_with_state(
                B=config.device_batch_size,
                T=config.max_seq_len,
                split="train",
                tokenizer_threads=4,
                tokenizer_batch_size=128,
                device=device,
                resume_state_dict=resume_state,
            )

        def build_val_loader():
            """Build streaming validation dataloader (no state tracking needed)."""
            return tokenizing_distributed_data_loader(
                B=config.device_batch_size,
                T=config.max_seq_len,
                split="val",
                tokenizer_threads=4,
                tokenizer_batch_size=128,
                device=device,
            )

        train_loader = build_train_loader(resume_state=dataloader_resume_state)

        # Prefetch first batch (with state_dict)
        x, y, dataloader_resume_state = next(train_loader)
```

**AFTER**:
```python
    if DATALOADER_V2_AVAILABLE:
        print0("  Using production dataloader v2 (async prefetch + packing)")

        dl_config = DataloaderConfig(
            batch_size=config.device_batch_size,
            seq_len=config.max_seq_len,
            num_workers=4,
            prefetch_factor=2,
            pin_memory=(device_type == "cuda"),
            pack_documents=True,
        )

        def build_train_loader(resume_state=None):
            return create_train_dataloader(
                dl_config, device=device, resume_state=resume_state
            )

        def build_val_loader():
            return create_val_dataloader(dl_config, device=device)

        train_loader = build_train_loader(resume_state=dataloader_resume_state)
        x, y, dataloader_resume_state = next(train_loader)

    elif DATALOADER_AVAILABLE:
        print0("  Using streaming dataloader from parquet files")

        def build_train_loader(resume_state=None):
            return tokenizing_distributed_data_loader_with_state(
                B=config.device_batch_size,
                T=config.max_seq_len,
                split="train",
                tokenizer_threads=4,
                tokenizer_batch_size=128,
                device=device,
                resume_state_dict=resume_state,
            )

        def build_val_loader():
            return tokenizing_distributed_data_loader(
                B=config.device_batch_size,
                T=config.max_seq_len,
                split="val",
                tokenizer_threads=4,
                tokenizer_batch_size=128,
                device=device,
            )

        train_loader = build_train_loader(resume_state=dataloader_resume_state)
        x, y, dataloader_resume_state = next(train_loader)
```

**Why this is correct:**
1. The new dataloader yields the same `(inputs, targets, state_dict)` triple — the training loop at lines 1471–1503 is unchanged
2. The fallback chain preserves backward compatibility: V2 → V1 → dummy
3. `DataloaderConfig` is a standalone dataclass; no model dependencies

**How to test:**
```bash
# Unit test the new dataloader:
python -c "from scripts.dataloader_v2 import create_train_dataloader; print('Import OK')"

# Run a short training loop with the new pipeline:
python scripts/pre-train.py --num_iterations=10 --device_batch_size=1 --max_seq_len=512 --total_batch_size=512
```

---

## 5. Phase 4: FSDP Integration (from Doc 04)

### 5a. Create `model/distributed/` directory

```bash
mkdir -p model/distributed
touch model/distributed/__init__.py
```

Doc 04 produces `model/distributed/fsdp_utils.py` with:
- `wrap_model_fsdp(model, config) -> FSDP` — applies FSDP wrapping with MoE-aware sharding
- `get_fsdp_policy(config) -> MixedPrecision` — BF16/FP32 mixed precision policy
- `save_fsdp_checkpoint(model, path)` / `load_fsdp_checkpoint(model, path)` — full/sharded state dict conversion

### 5b. Modify `scripts/pre-train.py` model initialization — lines 1002–1019

**BEFORE** (`scripts/pre-train.py` lines 1002–1019):
```python
    # Use meta device for memory-efficient initialization
    with torch.device("meta"):
        model = NanoSeekModel(model_config)

    # Materialize on target device
    model.to_empty(device=device)

    # Initialize weights
    model.apply(model._init_weights)

    # Keep reference to uncompiled model (for checkpointing)
    orig_model = model

    # Compile for speedup (CUDA only)
    if config.compile_model and device_type == "cuda":
        print0("Compiling model with torch.compile...")
        # fullgraph=False for MoE dynamic routing compatibility
        model = torch.compile(model, mode='reduce-overhead', fullgraph=False, dynamic=True)
```

**AFTER**:
```python
    # Use meta device for memory-efficient initialization
    with torch.device("meta"):
        model = NanoSeekModel(model_config)

    # Materialize on target device
    model.to_empty(device=device)

    # Initialize weights
    model.apply(model._init_weights)

    # Keep reference to uncompiled model (for checkpointing)
    orig_model = model

    # FSDP wrapping (for large models or expert parallelism)
    use_fsdp = ddp and ddp_world_size > 1 and getattr(config, 'use_fsdp', False)
    if use_fsdp:
        try:
            from model.distributed.fsdp_utils import wrap_model_fsdp, get_fsdp_policy
            print0("Wrapping model with FSDP (MoE-aware sharding)...")
            model = wrap_model_fsdp(model, model_config, device)
            orig_model = model  # FSDP model IS the model for checkpointing
        except ImportError:
            print0("Warning: FSDP utils not available, falling back to DDP")
            use_fsdp = False

    # Compile for speedup (CUDA only, after FSDP wrapping)
    if config.compile_model and device_type == "cuda":
        print0("Compiling model with torch.compile...")
        model = torch.compile(model, mode='reduce-overhead', fullgraph=False, dynamic=True)
```

### 5c. Modify DDP wrapping — lines 1146–1148

**BEFORE** (`scripts/pre-train.py` lines 1146–1148):
```python
    if ddp:
        print0(f"\nWrapping model with DDP...")
        model = DDP(model, device_ids=[ddp_local_rank])
```

**AFTER**:
```python
    if ddp and not use_fsdp:
        print0(f"\nWrapping model with DDP...")
        model = DDP(model, device_ids=[ddp_local_rank])
    elif use_fsdp:
        print0(f"\nUsing FSDP (already wrapped, skipping DDP)")
```

### 5d. Modify gradient accumulation for FSDP `no_sync` — lines 1471–1491

**BEFORE** (`scripts/pre-train.py` lines 1471–1491):
```python
        for micro_step in range(grad_accum_steps):
            with autocast_ctx:
                outputs = model(input_ids=x, labels=y)
                loss = outputs['loss']
                ...
            train_loss = loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()
            ...
```

**AFTER**:
```python
        for micro_step in range(grad_accum_steps):
            # FSDP no_sync: skip gradient all-reduce on intermediate micro-steps
            is_last_micro = (micro_step == grad_accum_steps - 1)
            sync_ctx = nullcontext() if (is_last_micro or not ddp) else model.no_sync()

            with sync_ctx:
                with autocast_ctx:
                    outputs = model(input_ids=x, labels=y)
                    loss = outputs['loss']
                    ...
                train_loss = loss.detach()
                loss = loss / grad_accum_steps
                loss.backward()
            ...
```

**Why this is correct:**
1. `model.no_sync()` is available on both DDP and FSDP — it defers gradient all-reduce until the final micro-step
2. Without `no_sync`, each micro-step triggers an all-reduce — wasting `(grad_accum_steps - 1)` communications per step
3. The `nullcontext()` on the last step ensures the final all-reduce happens normally
4. For single-GPU (`not ddp`), `nullcontext()` is a no-op — no behavior change

### 5e. Modify checkpoint save/load for FSDP state dicts

**Key pitfall**: FSDP wraps parameters with `FlatParameter`, changing state dict keys. When loading a DDP checkpoint into FSDP (or vice versa), you must convert:

```python
# In load_checkpoint() — scripts/pre-train.py line 815
# Already handles torch.compile prefix:
model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

# ALSO strip FSDP prefix if present:
model_data = {k.removeprefix("_fsdp_wrapped_module."): v for k, v in model_data.items()}
```

**How to test:**
```bash
# Single-GPU (should be unchanged):
python scripts/pre-train.py --num_iterations=5 --device_batch_size=1 --max_seq_len=512 --total_batch_size=512

# Multi-GPU FSDP (requires 2+ GPUs):
torchrun --nproc_per_node=2 scripts/pre-train.py --use_fsdp=true --num_iterations=5 --device_batch_size=1 --max_seq_len=512 --total_batch_size=512

# Verify checkpoint round-trip:
# 1. Save with FSDP
# 2. Load without FSDP (should work after key stripping)
```

---

## 6. Phase 5: FP8 Integration (from Doc 05)

### 6a. Activate FP8 config in `model/config.py`

The `FP8Config` dataclass already exists at line 156 of `model/config.py`:

```python
@dataclass
class FP8Config:
    enabled: bool = False
    dtype: Literal["bf16", "fp8"] = "bf16"
    block_size: int = 128
    scale_fmt: Optional[str] = None
    use_delayed_scaling: bool = True
```

And `NanoSeekConfig` already includes it at line 527:
```python
    fp8: FP8Config = field(default_factory=FP8Config)
```

**No config changes needed** — just set `config.fp8.enabled = True` at runtime.

### 6b. Create `model/fp8.py` — FP8Linear wrapper

Doc 05 produces `model/fp8.py` with `FP8Linear` (a drop-in replacement for `nn.Linear` that quantizes weights and activations to FP8). The `model/__init__.py` already has the import guard at lines 138–150:

```python
try:
    from .fp8 import (
        FP8Linear,
        FP8KVCache,
        FP8TrainingContext,
        ...
    )
    FP8_AVAILABLE = True
except ImportError:
    FP8_AVAILABLE = False
```

### 6c. Add FP8 linear replacement in model initialization

Add `enable_fp8()` method to `NanoSeekModel` after `_init_weights` (line 1554 of `model/model.py`). This method iterates `self.named_modules()`, replaces each `nn.Linear` (except `embed_tokens` and `lm_head`) with `FP8Linear.from_linear(module)`, and returns the count of replaced layers. It guards on `is_fp8_available()` from `model/fp8.py` to fail cleanly on non-H100 hardware.

### 6d. Wire FP8 activation in `scripts/pre-train.py`

**ADD** after model initialization (after line 1019 of `scripts/pre-train.py`):
```python
    # FP8 activation (requires H100/H200)
    if model_config.fp8.enabled:
        try:
            num_replaced = orig_model.enable_fp8()
            print0(f"  FP8 enabled: replaced {num_replaced} Linear layers with FP8Linear")
        except Exception as e:
            print0(f"  FP8 requested but not available: {e}")
            model_config.fp8.enabled = False
```

**Why this is correct:**
1. FP8 replacement happens before `torch.compile` and DDP/FSDP wrapping, ensuring the computation graph includes FP8 ops
2. Embedding and lm_head are excluded because they need full precision for stable convergence
3. `FP8Linear.from_linear()` copies existing weights, converting to FP8 format
4. Runtime guard prevents crashes on non-H100 hardware

**How to test:**
```bash
# Verify FP8 module imports (on H100):
python -c "from model.fp8 import FP8Linear, is_fp8_available; print(is_fp8_available())"

# Training with FP8:
python scripts/pre-train.py --use_fp8=true --num_iterations=10

# Numerical comparison (FP8 vs BF16 should have <0.5% loss difference):
python scripts/pre-train.py --use_fp8=false --num_iterations=100 --run_name=bf16_baseline
python scripts/pre-train.py --use_fp8=true --num_iterations=100 --run_name=fp8_test
```

---

## 7. Phase 6: Inference Engine (from Docs 07, 08)

### 7a. Create `model/serving/` directory

```bash
mkdir -p model/serving
touch model/serving/__init__.py
```

Doc 07 produces:
- `model/serving/engine.py` — continuous batching engine with KV cache management
- `model/serving/kv_cache.py` — paged KV cache pool

Doc 08 produces:
- `model/serving/speculative.py` — MTP-based speculative decoding pipeline

`model/serving/__init__.py` contents:
```python
"""NanoSeek production serving infrastructure."""
from .engine import InferenceEngine, EngineConfig
from .speculative import SpeculativeDecoder, SpeculativeConfig
from .kv_cache import KVCachePool, PagedKVCache

__all__ = [
    'InferenceEngine', 'EngineConfig',
    'SpeculativeDecoder', 'SpeculativeConfig',
    'KVCachePool', 'PagedKVCache',
]
```

### 7b. Wire speculative decoding into existing `MultiTokenPrediction.speculative_decode()`

The existing `speculative_decode` method at `model/model.py` lines 1014–1049 already implements the draft phase. The production speculative decoder from Doc 08 wraps this with:
1. Batch-level draft/verify orchestration
2. Token tree pruning
3. Acceptance rate tracking

**Key integration point**: `model/serving/speculative.py` calls `model.mtp.speculative_decode()` (line 1014) for the draft phase and `model.forward()` for verification. No changes to `model/model.py` are needed — the serving layer composes existing methods.

### 7c. Create `scripts/serve.py` launch script

A thin CLI that: (1) loads a checkpoint via `build_model()` from `model/eval/checkpoint_manager.py` line 71, (2) creates an `InferenceEngine` with `EngineConfig`, and (3) calls `engine.serve(port=args.port)`. Accepts `--checkpoint_dir`, `--port`, `--max_batch_size`, `--max_seq_len`, `--use_speculative` flags.

**How to test:**
```bash
# Start server:
python scripts/serve.py --checkpoint_dir=data/base_checkpoints/d16

# Test with curl:
curl -X POST http://localhost:8080/generate -d '{"prompt": "Hello world", "max_tokens": 50}'

# Benchmark throughput:
python -m model.serving.engine --benchmark --checkpoint_dir=data/base_checkpoints/d16
```

---

## 8. Phase 7: Post-Training Pipeline (from Doc 09)

### 8a. Create `scripts/sft.py` — Supervised Fine-Tuning

Doc 09 produces a standard SFT script that:
1. Loads a pre-trained checkpoint via `build_model()` from `model/eval/checkpoint_manager.py` line 71
2. Applies chat template formatting
3. Trains with cross-entropy on assistant responses only (masking user tokens with `ignore_index=-100`)
4. Uses the same optimizer setup as `pre-train.py` (Muon + AdamW)

The key integration with the existing codebase is the model loading path:

```python
# In scripts/sft.py — uses existing checkpoint infrastructure
from model.eval.checkpoint_manager import build_model, save_checkpoint

model, tokenizer, meta = build_model(checkpoint_dir, step, device, phase="train")
# model is a NanoSeekModel instance, ready for fine-tuning
```

### 8b. Create `scripts/dpo.py` — Direct Preference Optimization

DPO requires paired (chosen, rejected) completions. The script:
1. Loads model via same `build_model()` path
2. Computes log-probabilities for both completions using `model.forward()` (line 1562)
3. Applies the DPO loss: `loss = -log_sigmoid(β * (log_π_chosen - log_π_rejected - log_π_ref_chosen + log_π_ref_rejected))`

### 8c. Create `model/lora.py` — LoRA adapter module

**Integration points** in `model/model.py`:

The LoRA module wraps specific `nn.Linear` layers. For NanoSeek, the targets are:
- MLA projections: `wq_a` (line 265), `wq_b` (line 267), `wkv_a` (line 270), `wkv_b` (line 272), `wo` (line 275)
- Expert projections: `gate_proj` (line 458), `up_proj` (line 459), `down_proj` (line 460)

**ADD** `model/lora.py`:
```python
"""LoRA (Low-Rank Adaptation) for NanoSeek fine-tuning."""
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_out + lora_out


def apply_lora(model, target_modules=None, rank=16, alpha=32.0):
    """Apply LoRA to specified modules in a NanoSeekModel."""
    if target_modules is None:
        target_modules = ['wq_a', 'wq_b', 'wkv_a', 'wkv_b', 'wo']

    replaced = 0
    for name, module in model.named_modules():
        attr_name = name.rsplit('.', 1)[-1] if '.' in name else name
        if attr_name in target_modules and isinstance(module, nn.Linear):
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            parent = dict(model.named_modules())[parent_name] if parent_name else model
            setattr(parent, attr_name, LoRALinear(module, rank=rank, alpha=alpha))
            replaced += 1

    # Freeze all non-LoRA parameters
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad_(False)

    return replaced
```

**How to test:**
```bash
# LoRA application test (use minimal 2-layer config):
python -c "from model.lora import apply_lora; print('LoRA module importable')"
```

---

## 9. Phase 8: Evaluation Suite (from Doc 10)

### 9a. Create `model/eval/benchmarks/` directory

```bash
mkdir -p model/eval/benchmarks
touch model/eval/benchmarks/__init__.py
```

Doc 10 produces evaluation harnesses for:
- MMLU (knowledge)
- HellaSwag (commonsense)
- GSM8K (math)
- HumanEval (code)
- LAMBADA (language modeling)

These use the existing `model/eval/core_eval.py` (line 1) infrastructure for prompt rendering and scoring.

### 9b. Create `scripts/evaluate.py`

**Key integration**: The evaluation script uses `load_model()` from `model/eval/checkpoint_manager.py` line 206:

```python
# In scripts/evaluate.py
from model.eval.checkpoint_manager import load_model

model, tokenizer, meta = load_model("base", device=device, phase="eval")
# Now run benchmarks...
```

This leverages the existing checkpoint directory convention:
- `data/base_checkpoints/` — pre-trained models
- `data/chatsft_checkpoints/` — SFT models
- `data/chatrl_checkpoints/` — RLHF/DPO models

### 9c. Wire into existing `model/eval/` infrastructure

The existing `model/eval/core_eval.py` already has `evaluate_task()` (line 15 reference in docstring). The benchmark suite calls this per-task, aggregating results:

```python
# In model/eval/benchmarks/mmlu.py
from model.eval.core_eval import evaluate_task

def evaluate_mmlu(model, tokenizer, device, num_fewshot=5):
    results = {}
    for subject in MMLU_SUBJECTS:
        data = load_mmlu_data(subject, num_fewshot)
        task_meta = {"type": "multiple_choice", "choices": 4}
        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)
        results[subject] = accuracy
    return results
```

The `model/eval/loss_eval.py` BPB evaluation (line 23: `evaluate_bpb`) is used for perplexity benchmarks.

**How to test:**
```bash
# Run evaluation suite:
python scripts/evaluate.py --source=base --benchmarks=hellaswag,lambada

# Quick smoke test (no GPU needed):
python -c "from model.eval.benchmarks import AVAILABLE_BENCHMARKS; print(AVAILABLE_BENCHMARKS)"
```

---

## 10. Integration Verification Checklist

For **EACH** integration phase, verify:

### Phase 1: Flash MLA ✓
- [ ] `pytest tests/test_mla.py -v` — all existing tests pass
- [ ] `pytest tests/test_kernels.py::test_flash_mla_numerical_parity -v` — max abs error < 1e-3
- [ ] `pytest tests/test_numerical.py -v` — no numerical stability regressions
- [ ] Memory: attention peak memory reduced by ~50% at seq_len=4096

### Phase 2: Fused MoE ✓
- [ ] `pytest tests/test_moe.py tests/test_moe_standalone.py -v` — all pass
- [ ] `pytest tests/test_kernels.py::test_fused_moe_parity -v` — max abs error < 1e-3
- [ ] `pytest tests/test_integration.py -v` — full model forward/backward still works
- [ ] Throughput: MoE dispatch 3-8x faster (measure with `--benchmark` flag)

### Phase 3: Data Pipeline ✓
- [ ] `python scripts/pre-train.py --num_iterations=10` — trains without error
- [ ] Dataloader yields correct shapes: `(B, T)` for inputs and targets
- [ ] Resume state round-trip: save state, reload, verify no data repetition
- [ ] GPU utilization stays >85% during training (no I/O stalls)

### Phase 4: FSDP ✓
- [ ] Single-GPU training unchanged: `python scripts/pre-train.py --num_iterations=5`
- [ ] Multi-GPU FSDP: `torchrun --nproc_per_node=2 scripts/pre-train.py --use_fsdp=true --num_iterations=5`
- [ ] Checkpoint save/load with FSDP prefix stripping works
- [ ] `_orig_mod.` and `_fsdp_wrapped_module.` prefixes handled in `load_checkpoint` (line 815)

### Phase 5: FP8 ✓
- [ ] `python -c "from model.fp8 import is_fp8_available; print(is_fp8_available())"` returns True on H100
- [ ] Training with `--use_fp8=true` runs without NaN/Inf
- [ ] Loss within 0.5% of BF16 baseline after 100 steps
- [ ] Memory reduction: ~30-40% lower peak memory

### Phase 6: Inference Engine ✓
- [ ] `python scripts/serve.py --checkpoint_dir=... --port=8080` starts successfully
- [ ] Generate endpoint returns coherent text
- [ ] Speculative decoding acceptance rate >60% (with MTP draft model)
- [ ] Throughput: >1000 tokens/sec on single H100

### Phase 7: Post-Training ✓
- [ ] `python scripts/sft.py --num_iterations=10` — SFT trains without error
- [ ] `python scripts/dpo.py --num_iterations=10` — DPO trains without error
- [ ] LoRA: `apply_lora(model)` — trainable params <5% of total
- [ ] SFT checkpoint loadable by inference engine

### Phase 8: Evaluation ✓
- [ ] `python scripts/evaluate.py --benchmarks=hellaswag` runs end-to-end
- [ ] Results are reproducible (seeded evaluation)
- [ ] Report generation works: `model/eval/report.py`

### Cross-cutting ✓
- [ ] `pytest tests/ -v -m "not slow"` — ALL existing tests pass after full integration
- [ ] `pytest tests/ -v` — including slow tests
- [ ] No import errors from any entry point
- [ ] `model/__init__.py` exports still work (optional imports don't crash on missing modules)

---

## 11. Common Integration Pitfalls

### 11a. `torch.compile` interaction with custom kernels

**Problem**: `torch.compile` may try to trace through Triton kernels, causing compilation failures or incorrect code generation.

**Solution**: Mark custom kernels as `torch.compiler.disable`-d or use `torch.compiler.is_compiling()` guards:

```python
# In model/kernels/flash_mla.py
@torch.compiler.disable
def flash_mla_attention(q, k, v, softmax_scale, causal):
    # Triton kernel call here — opaque to torch.compile
    ...
```

**Affected files**:
- `model/model.py` line 1019: `torch.compile(model, ...)` — uses `fullgraph=False` and `dynamic=True` which already allows graph breaks at custom kernel boundaries
- `model/kernels/flash_mla.py` — must be marked non-compilable
- `model/kernels/fused_moe.py` — must be marked non-compilable

### 11b. FSDP state dict key naming (`_orig_mod.` prefix)

**Problem**: `torch.compile` wraps the model, prepending `_orig_mod.` to all state dict keys. FSDP adds `_fsdp_wrapped_module.`. When loading a checkpoint saved under one configuration into another, keys don't match.

**Solution** (already partially implemented at `scripts/pre-train.py` line 815):

```python
# Complete key normalization chain:
def normalize_state_dict_keys(state_dict):
    """Strip all framework wrapper prefixes from state dict keys."""
    normalized = {}
    for k, v in state_dict.items():
        clean_key = k
        clean_key = clean_key.removeprefix("_orig_mod.")
        clean_key = clean_key.removeprefix("_fsdp_wrapped_module.")
        clean_key = clean_key.removeprefix("module.")  # DDP prefix
        normalized[clean_key] = v
    return normalized
```

**Affected files**:
- `scripts/pre-train.py` line 815: `load_checkpoint()` — add FSDP prefix stripping
- `model/eval/checkpoint_manager.py` line 100: `build_model()` — already handles `_orig_mod.`; add FSDP

### 11c. Gradient accumulation with mixed custom/standard ops

**Problem**: Custom Triton kernels (Flash MLA, Fused MoE) have custom backward implementations. When mixed with standard PyTorch autograd ops in the same graph, gradient accumulation can produce incorrect results if the custom backward doesn't handle accumulated gradients.

**Solution**:
- Verify that custom kernel backward functions use `+=` (accumulate) not `=` (overwrite) for gradient tensors
- The `loss = loss / grad_accum_steps` scaling at `scripts/pre-train.py` line 1489 ensures gradients are correctly scaled regardless of kernel implementation

**Test**: Compare gradients with `grad_accum=1` vs `grad_accum=2` (same total loss), verify max diff < 1e-5 per parameter.

### 11d. Triton autotuning cache invalidation

**Problem**: Triton caches auto-tuned kernel configurations in `~/.triton/cache/`. When you change kernel parameters (e.g., `BLOCK_SIZE`, `num_warps`), stale cache entries can cause incorrect behavior or crashes.

**Solution**:
```bash
# Clear Triton cache when changing kernel configs:
rm -rf ~/.triton/cache/

# Or set a unique cache directory per experiment:
export TRITON_CACHE_DIR=/tmp/triton_cache_$(date +%s)
```

### 11e. MoE expert count must match between kernel and model

**Problem**: The fused MoE kernel (Doc 03) allocates memory based on `n_routed_experts`. If the model config's `n_routed_experts` (64, at `model/config.py` line 298) doesn't match the kernel's expectation, you get silent corruption.

**Solution**: The fused kernel reads expert count from the `experts` ModuleList length at runtime:
```python
E = len(experts)  # model/model.py line 515
```
The kernel must accept `E` as a runtime parameter, not a compile-time constant.

### 11f. KV cache format compatibility between training and inference

**Problem**: During training, `MultiHeadLatentAttention.forward()` (line 287) stores KV cache as `(kv_compressed, k_pe)` — shapes `[B, S, kv_lora_rank]` and `[B, S, 1, qk_rope_head_dim]`. The inference engine (Doc 07) must use the same format.

**Solution**: The inference engine's `KVCachePool` must allocate per-layer caches matching:
```python
kv_compressed_shape = (max_batch, max_seq, config.mla.kv_lora_rank)  # 143 dims
k_pe_shape = (max_batch, max_seq, 1, config.mla.qk_rope_head_dim)   # 32 dims
# Total per token per layer: 143 + 32 = 175 (the ~23x compression)
```

This is already consistent with the `present_key_value` tuple returned at `model/model.py` line 332:
```python
present_key_value = (kv_compressed, k_pe) if use_cache else None
```

### 11g. MTP module weight sharing across training and inference

**Problem**: MTP modules share embeddings with the main model (`model/model.py` lines 1516–1517):
```python
if tie_mtp_embeddings:
    self.mtp.set_shared_embeddings(self.embed_tokens, self.lm_head)
```

When saving/loading checkpoints with FSDP, the shared weight references can become disconnected.

**Solution**: After loading a checkpoint, explicitly re-tie:
```python
# After model.load_state_dict():
if model.mtp is not None and model.config.tie_word_embeddings:
    model.mtp.set_shared_embeddings(model.embed_tokens, model.lm_head)
```

Add this after `model/eval/checkpoint_manager.py` line 112:
```python
    model.load_state_dict(model_data, strict=True, assign=True)

    # Re-tie MTP shared embeddings (may be disconnected after state dict load)
    if hasattr(model, 'mtp') and model.mtp is not None:
        if model.config.tie_word_embeddings or (
            model.config.mtp.mtp_hidden_size == model.config.hidden_size
        ):
            model.mtp.set_shared_embeddings(model.embed_tokens, model.lm_head)
```

---

## 12. File-Level Change Summary

### Files Modified

| File | Lines Changed | Description |
|------|---------------|-------------|
| `model/model.py` | 352–370 | Flash MLA attention dispatch |
| `model/model.py` | 548–583 | Fused MoE dispatch |
| `model/model.py` | after 1554 | `enable_fp8()` method |
| `scripts/pre-train.py` | 155–170 | Dataloader import chain |
| `scripts/pre-train.py` | 1002–1019 | FSDP model init |
| `scripts/pre-train.py` | 1146–1148 | DDP/FSDP branching |
| `scripts/pre-train.py` | 1158–1213 | Dataloader init |
| `scripts/pre-train.py` | 1471–1491 | FSDP `no_sync` |
| `scripts/pre-train.py` | after 1019 | FP8 activation |
| `model/eval/checkpoint_manager.py` | 100, 112 | Key stripping, MTP re-tie |

### Files Created

| File | Source Doc | Description |
|------|-----------|-------------|
| `model/kernels/__init__.py` | — | Kernel package init |
| `model/kernels/flash_mla.py` | Doc 02 | Triton flash MLA kernel |
| `model/kernels/fused_moe.py` | Doc 03 | Triton fused MoE dispatch |
| `scripts/dataloader_v2.py` | Doc 01 | Production data pipeline |
| `model/distributed/__init__.py` | — | Distributed package init |
| `model/distributed/fsdp_utils.py` | Doc 04 | FSDP wrapping utilities |
| `model/fp8.py` | Doc 05 | FP8 linear & quantization |
| `model/serving/__init__.py` | — | Serving package init |
| `model/serving/engine.py` | Doc 07 | Inference engine |
| `model/serving/kv_cache.py` | Doc 07 | Paged KV cache |
| `model/serving/speculative.py` | Doc 08 | Speculative decoding |
| `scripts/sft.py` | Doc 09 | Supervised fine-tuning |
| `scripts/dpo.py` | Doc 09 | DPO alignment |
| `model/lora.py` | Doc 09 | LoRA adapter |
| `model/eval/benchmarks/__init__.py` | Doc 10 | Benchmark suite |
| `scripts/evaluate.py` | Doc 10 | Evaluation runner |
| `scripts/serve.py` | Docs 07/08 | Inference server |

### Files Unchanged

All test files (`tests/*.py`), `model/config.py`, `model/optimizer/*.py`, `scripts/scheduler.py`, `scripts/tokenizer.py`, `scripts/dataset.py`, `scripts/utils.py` — these require **zero** modifications for integration. The entire integration is additive.

---

## 13. End-to-End Integration Smoke Test

After all phases are integrated, run this validation sequence:

```bash
#!/bin/bash
set -e
echo "=== NanoSeek Production Integration Smoke Test ==="

# 1. Import verification
python -c "from model.model import NanoSeekModel; from model.config import get_nanoseek_config; print('OK')"

# 2. Kernel availability
python -c "
try:
    from model.kernels import flash_mla_available, fused_moe_available
    print(f'Flash MLA: {flash_mla_available()}, Fused MoE: {fused_moe_available()}')
except ImportError:
    print('Kernels not installed (expected pre-integration)')
"

# 3. Existing test suite (MUST pass)
pytest tests/ -v -m "not slow" --tb=short

# 4. Forward/backward + KV cache + checkpoint round-trip
pytest tests/test_integration.py tests/test_mla.py tests/test_moe.py -v --tb=short

echo "=== All smoke tests passed ==="
```

---

## 14. Recommended Integration Order for a Single Engineer

**Week 1**: Phases 1–2 (Kernels)
- Day 1–2: Flash MLA kernel + integration tests
- Day 3–5: Fused MoE kernel + integration tests + benchmarks

**Week 2**: Phases 3–5 (Training Infrastructure)
- Day 1: Data pipeline V2
- Day 2–3: FSDP integration + checkpoint compatibility
- Day 4–5: FP8 integration (if H100 available)

**Week 3**: Phases 6–8 (Inference + Post-Training + Eval)
- Day 1–2: Inference engine + speculative decoding
- Day 3: Post-training scripts (SFT, DPO, LoRA)
- Day 4: Evaluation suite
- Day 5: End-to-end integration testing + documentation

**Week 4**: Hardening — performance benchmarking, edge case testing (OOM, checkpoint corruption), CI/CD pipeline setup.
