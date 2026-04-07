# NanoSeek Training Failure Analysis

## Date: 2026-04-07

This document catalogs every bug, error, and failure mode discovered during
the Gate 1 smoke test debugging session. Issues are ordered by severity.

---

## CRITICAL: Bugs That Crash Training

### 1. Stale Cached Weight References After `to_empty()` (FIXED)

**File:** `model.py:716-722`
**Symptom:** `RuntimeError: Tensor on device cuda:0 is not on the expected device meta!`

The `MoE.__init__()` cached expert weight references in Python lists:
```python
self._expert_gate_weights = [e.w_gate.weight for e in self.routed_experts]
self._expert_up_weights   = [e.w_up.weight for e in self.routed_experts]
self._expert_down_weights = [e.w_down.weight for e in self.routed_experts]
```

When model is built on meta device (`with torch.device("meta")`), these lists
store references to **meta tensors**. After `to_empty(device="cuda")`, the
`nn.Parameter` objects get new CUDA storage, but these cached lists still hold
the old meta tensor references. `torch.stack(self._expert_gate_weights)` then
produces a meta tensor, causing device mismatch with CUDA activations.

**Fix:** Removed cached lists. Use inline access in forward:
```python
torch.stack([e.w_gate.weight for e in self.routed_experts])
```

**Root cause:** The `to_empty()` + `init_weights()` pattern replaces parameter
`.data` storage but does NOT update plain Python references to old tensors.
Never cache `nn.Parameter.weight` references during `__init__` when using
meta device construction.

---

### 2. `.item()` Causes torch.compile Infinite Recompilation (FIXED)

**File:** `model.py:749, 834`
**Symptom:** `torch._dynamo hit config.accumulated_cache_size_limit (256)`
followed by training falling back to uncompiled (slow) mode.

Two `.item()` calls in the MoE forward path:
```python
# Line 749 - inside _batched_expert_forward
max_count = expert_counts.max().item()
padded_input = sorted_x.new_zeros(E, max_count, D)  # dynamic shape!

# Line 834 - dispatch decision
max_count = expert_counts.max().item()
waste_ratio = max_count / max(avg_count, 1)
use_batched = waste_ratio < 1.5  # Python conditional on GPU value
```

With `torch.compile(dynamic=False)`, every unique `max_count` value creates a
different tensor shape → triggers recompilation. Since `max_count` varies every
forward pass (different routing decisions), this causes 256 recompilations
before torch.compile gives up entirely.

**Attempted fixes and why they failed:**

1. **`capture_scalar_outputs=True`** — Told torch.compile to handle `.item()` as
   dynamic shapes. Result: still recompiles on every unique value.

2. **Static pad_size = 4x average** — `pad_size = (NK+E-1)//E * 4 = 8192`.
   Result: `CUDA index out of bounds` because during actual training with
   gradient checkpointing, some experts received >8192 tokens (max observed: 8601).

3. **Static pad_size = 8x average** — `pad_size = (NK+E-1)//E * 8 = 16384`.
   Result: Works for anchor scale but risks OOM at ablation/1B scale.

**Current fix:** Using 8x average padding. This is a compile-time constant
(NK and E are fixed with `dynamic=False`), so torch.compile caches a single
compiled graph.

**Memory impact of 8x padding:**
| Scale    | NK       | avg/expert | pad_size | padded_input size | Status |
|----------|----------|------------|----------|-------------------|--------|
| Anchor   | 131,072  | 2,048      | 16,384   | 1.2 GB (bf16)     | OK     |
| Ablation | 4.2M     | 65,536     | 524,288  | **86 GB (bf16)**  | OOM!   |
| 1B       | 4.2M     | 65,536     | 524,288  | **138 GB (bf16)** | OOM!   |

**TODO:** For ablation/1B scales, must either:
- Use smaller pad multiplier (2x) + runtime OOB check with fallback to sequential
- Or revert to `.item()` with `torch.compile(dynamic=True)`
- Or implement a proper fused MoE kernel that doesn't need padding

---

### 3. All Ablation Override Flags Were Silently Broken (FIXED)

**File:** `pre_train.py:267-304`
**Symptom:** `--no-seq-aux`, `--no-mtp`, `--no-shared-experts`, `--num-experts`,
`--top-k`, `--n-group`, `--topk-group` flags all did **nothing**.

`config.moe` and `config.mtp` are `@property` methods that return a **new
`SimpleNamespace`** on every access. Writing `config.moe.X = Y` creates a
temporary object, sets the attribute on it, and immediately discards it.

```python
# BROKEN: modifies a temporary SimpleNamespace, NOT the config
config.moe.seq_aux_loss_alpha = 0.0

# CORRECT: writes to the flat dataclass field
config.seq_aux_loss_alpha = 0.0
```

**Fix:** Changed all ablation overrides to write to flat config fields.
Changed `--no-mtp` to set `config.num_mtp_modules = 0` instead of trying
to set MTP loss weights (which are constants in the property, not config fields).

---

## HIGH: Bugs That Cause Incorrect Training

### 4. Gradient Checkpointing + Mutable aux_data Overwrite

**File:** `model.py:1650-1677`
**Symptom:** Load balance bias updates use stale/wrong expert load counts.

With `use_reentrant=False` gradient checkpointing, the forward pass is
re-executed during backward. Each MoE layer stores `aux_data` (expert load
counts, H_load) in `self._layer_aux_data[i]` during forward:

```python
# Line 1677 - stores aux_data during forward
if aux_data:
    self._layer_aux_data[i] = aux_data
```

When gradient checkpointing re-executes the forward during backward, the
routing decisions produce different `load_counts` (because the RNG state
and floating-point order may differ). These new values overwrite
`self._layer_aux_data[i]`, so the bias update at line 1810 uses the
recomputed values, not the original forward pass values.

**Impact:** The load balance bias update uses incorrect expert load counts,
potentially causing routing instability. This is subtle and won't crash
training, but it corrupts the aux-loss-free balancing mechanism.

**Proposed fix:** Save aux_data from the original forward pass and don't
let checkpoint recompute overwrite it:
```python
if aux_data and i not in self._layer_aux_data:
    self._layer_aux_data[i] = aux_data
```

---

### 5. Mutable `_cached_causal_mask` Buffer During Forward

**File:** `model.py:283`
**Symptom:** torch.compile graph breaks / recompilations if sequence length varies.

```python
def _get_causal_mask(self, seq_len, kv_len, device, dtype):
    max_dim = max(seq_len, kv_len)
    if self._cached_causal_mask is None or self._cached_causal_mask.shape[0] < max_dim:
        mask = torch.full((max_dim, max_dim), float("-inf"), device=device, dtype=dtype)
        mask = mask.triu_(diagonal=1)
        self._cached_causal_mask = mask  # MUTATES BUFFER DURING FORWARD
    return self._cached_causal_mask[:seq_len, :kv_len]
```

This buffer is registered with `register_buffer()` but gets reassigned during
forward pass. This violates torch.compile's assumption that buffer shapes are
static. If sequence length ever increases, the buffer is reallocated, causing
a new graph trace.

**Note:** With the current dataloader (fixed T=4096) and `dynamic=False`,
this doesn't trigger in practice. But it's a latent bug.

**Proposed fix:** Pre-allocate at `max_position_embeddings` during `__init__`:
```python
mask = torch.full((max_pos, max_pos), float("-inf"))
mask.triu_(diagonal=1)
self.register_buffer("_cached_causal_mask", mask, persistent=False)
```

---

### 6. Routing Bias Buffer Mutation Between Forward and Checkpoint Recompute

**File:** `model.py:564, 662`  
**File:** `pre_train.py:1577`
**Symptom:** Gradient checkpointing recompute uses different bias values.

The routing bias (`self.bias`) is updated after each optimizer step:
```python
# pre_train.py:1577
orig_model.update_load_balance_bias(tokens_processed, config.total_tokens)
```

The bias update happens AFTER backward but BEFORE the next forward. This
ordering is correct. However, if gradient checkpointing re-executes a
forward pass during backward of a LATER layer, the bias from the current
step is used, not the bias from when the original forward was computed.

In practice, the bias update only happens once per full training step
(after all micro-steps complete), so within a single forward-backward pass,
the bias doesn't change. This is safe as currently implemented.

**Status:** Not a bug with current training loop ordering. Document as
invariant: bias update MUST happen after all backward passes complete.

---

## MEDIUM: Performance Issues

### 7. RoPE Recomputed 17x Per `_reinit_buffers()` Call (FIXED)

**File:** `model.py:1505-1533`
**Symptom:** Wasted CPU time during initialization.

`precompute_freqs_cis()` was called identically for each of 16 decoder
layers + MTP modules. All calls produced the same result.

**Fix:** Compute once, share across all layers:
```python
rope_freqs = precompute_freqs_cis(...)
for layer in self.layers:
    layer.self_attn.freqs_cis = rope_freqs.to(device=...)
```

---

### 8. Complex Number Operations Force torch.compile Fallback

**File:** `model.py:125, 145, 154`
**Symptom:** `Torchinductor does not support code generation for complex operators`

RoPE uses `torch.polar()`, `torch.view_as_complex()`, and complex
multiplication. TorchInductor cannot generate fused CUDA kernels for these,
falling back to eager execution for the entire RoPE subgraph.

**Impact:** RoPE is a small fraction of total compute (~2-3% of FLOPs),
so the performance impact is minor. But it prevents torch.compile from
fusing RoPE with adjacent operations (layer norm, attention).

**Proposed fix:** Rewrite RoPE using explicit sin/cos rotation:
```python
cos = freqs_cis.real  # [S, dim//2]
sin = freqs_cis.imag  # [S, dim//2]
x1, x2 = x[..., ::2], x[..., 1::2]
x_out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
```

---

### 9. `print0` Not Flushing Output (FIXED)

**File:** `common.py:117-120`
**Symptom:** When output is redirected to a file (background tasks), stdout
is fully buffered. Training appears "stuck" because print output sits in
a buffer for minutes before being flushed.

**Fix:** Added `flush=True` to `print0`:
```python
def print0(s="", **kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, flush=True, **kwargs)
```

---

### 10. Step 0 Takes 5+ Minutes (Inherent, Not a Bug)

**File:** `pre_train.py:1466-1492`
**Symptom:** First training step takes 300+ seconds vs ~15s for subsequent steps.

Causes (cumulative):
1. **torch.compile graph tracing** — first forward triggers compilation
   of the entire model graph. With 64 experts, MoE routing creates complex
   branching that takes ~30-60s to trace.
2. **cuBLAS autotuning** — first matmul calls trigger GEMM algorithm selection.
3. **CUDA caching allocator** — first allocations trigger pool sizing.
4. **Gradient checkpointing recompile** — backward triggers a second
   compilation for the recomputed forward graph.

**Not a bug.** Steps 1+ run at steady-state speed. Profile at steps 20-22
(per CLAUDE.md guidance) for representative measurements.

---

## LOW: Design Risks (Not Current Bugs)

### 11. Config Property Write-Through Has No Enforcement

**File:** `config.py:213-264`

The `@property` accessors for `config.moe`, `config.mtp`, `config.mla`
return new `SimpleNamespace` objects. Any write like `config.moe.X = Y`
silently fails. There's no runtime error, no warning, no linter check.

A future developer will inevitably write `config.moe.some_field = value`
and spend hours debugging why the change has no effect.

**Proposed fix:** Make the properties return frozen namespaces or add
`__setattr__` validation on NanoSeekConfig.

---

### 12. `to_dict()` Creates Redundant SimpleNamespace Objects

**File:** `config.py:270-297`

`config.to_dict()` accesses `self.moe.n_routed_experts`, `self.mla.q_lora_rank`,
etc. — each creating a new SimpleNamespace. Wastes memory and CPU but
functionally correct (reads work fine, only writes are broken).

---

### 13. `validate_config` Checks a Constant

**File:** `pre_train.py:254`
```python
if cfg.moe.gamma_freeze_ratio != 0.95:
```

`gamma_freeze_ratio` is not a config field — it's a constant
(`DEEPSEEK_GAMMA_FREEZE_RATIO = 0.95`) hardcoded in the property.
This validation always passes. It's dead code.

---

## Summary of Changes Made

| Fix | File | Lines | Status |
|-----|------|-------|--------|
| Flush print0 | common.py | 120 | DONE |
| Fix ablation overrides | pre_train.py | 267-304 | DONE |
| RoPE compute once | model.py | 1505-1533 | DONE |
| Remove stale weight cache | model.py | 716-722, 778-781 | DONE |
| Static pad_size (8x) | model.py | 762 | DONE (anchor only) |
| Remove .item() dispatch | model.py | 834-840 | DONE |
| Add timing instrumentation | pre_train.py | 332-338, 997, 1130, 1218 | DONE |

## Open Items

1. **pad_size scaling for ablation/1B** — 8x will OOM. Need per-scale strategy.
2. **aux_data overwrite during checkpoint recompute** — needs guard.
3. **Complex RoPE → real RoPE** — performance optimization, not blocking.
4. **Config property enforcement** — add `__setattr__` guard.
5. **Remove timing instrumentation** — clean up before production runs.
