# NanoSeek Training Failure Analysis — Part 2

## Date: 2026-04-07
## Complements: TRAINING_FAILURE_ANALYSIS.md (Issues 1-13)

This document covers bugs and failure modes discovered during the MoE
architecture review that are **NOT** in the original TRAINING_FAILURE_ANALYSIS.md.
Issues are ordered by severity.

---

## CRITICAL: Bugs That Silently Corrupt Training

### 14. `seq_aux_loss_alpha` Never Reached the Gate — All Ablation Flags Broken (FIXED)

**File:** `nanoseek/nanoseek/model.py:1260-1321` (NanoSeekDecoderLayer)
**File:** `nanoseek/nanoseek/model.py:683-714` (MoE)
**Symptom:** `--no-seq-aux` ablation flag does nothing. Gate always uses default alpha=0.0001.

**Root cause:** Two breaks in the parameter threading chain:

1. `NanoSeekDecoderLayer.__init__` accepted no `seq_aux_loss_alpha` parameter, so it
   never passed the config value to `MoE()`.
2. Even though `MoE.__init__` accepted `seq_aux_loss_alpha`, it was only receiving
   the default value because the caller never passed it.

The Gate constructor has its own default:
```python
class Gate(nn.Module):
    def __init__(self, ..., seq_aux_loss_alpha: float = 0.0001):
```

So even when a user set `config.seq_aux_loss_alpha = 0.0` via `--no-seq-aux`,
the value never propagated past `NanoSeekDecoderLayer`. The Gate always computed
aux loss with alpha=0.0001.

**Impact:** Every ablation run that tested `--no-seq-aux` was **invalid** — the
aux loss was never actually disabled. Any conclusions drawn from comparing
`--no-seq-aux` runs to baselines are wrong.

**Fix applied:** Added `seq_aux_loss_alpha` parameter to `NanoSeekDecoderLayer.__init__`
and threaded it through to `MoE()`. Full chain now:
```
config.moe.seq_aux_loss_alpha (from config.seq_aux_loss_alpha)
  → NanoSeekDecoderLayer(seq_aux_loss_alpha=...)     [model.py:1442]
    → MoE(seq_aux_loss_alpha=...)                    [model.py:1315]
      → Gate(seq_aux_loss_alpha=...)                 [model.py:708]
        → self.seq_aux_loss_alpha used in forward    [model.py:630]
```

**Verification:**
```python
moe = MoE(..., seq_aux_loss_alpha=0.0)
assert moe.gate.seq_aux_loss_alpha == 0.0  # Now passes
x = torch.randn(2, 4, 128)
_, aux = moe(x)
assert aux['aux_loss'].item() == 0.0       # Now passes
```

**Lesson:** Always write an end-to-end test that verifies config values reach
the component that uses them. Unit tests on individual classes miss threading bugs.

---

### 15. `routed_scaling_factor=2.5` Copied From Wrong Architecture Scale

**File:** `nanoseek/nanoseek/config.py:38`
**File:** `nanoseek/nanoseek/model.py:619-620`
**Status:** UNFIXED — needs HP tuning
**Symptom:** Shared expert contribution suppressed relative to routed experts.

**What happens:**
```python
# Gate.forward, lines 616-620:
weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)  # sum to 1.0
weights = weights * self.routed_scaling_factor                    # multiply by 2.5
```

After normalization, weights sum to 1.0. After scaling by 2.5, the routed expert
branch contributes 2.5x the hidden state norm. The shared expert contributes 1.0x:
```python
# MoE.forward, line 880:
output = routed_output + shared_output  # 2.5x + 1.0x = 3.5x total
```

**Why 2.5 is wrong for NanoSeek:**

DeepSeek V3 uses **256 routed experts, top-8** (activation ratio = 3.125%).
NanoSeek uses **64 routed experts, top-8** (activation ratio = 12.5% — **4x higher**).

The scaling factor compensates for the fraction of expert capacity that's active.
With 4x more activation ratio, the routed branch is already contributing
proportionally more per token. Using the same 2.5 factor over-amplifies it.

**Impact:**
- Shared expert gradient signal is relatively weaker (1/2.5 = 0.4x of routed)
- Shared expert may learn more slowly, failing to capture common patterns
- Training instability from amplified routing noise early in training
- The severity depends on the model — may be fine, may cause measurable quality loss

**Recommendation:** Add `routed_scaling_factor` to the HP search grid:
```bash
for rsf in 1.0 1.5 2.0 2.5; do
  python -m nanoseek.scripts.pre_train \
      --run "hp-rsf${rsf}" --scale ablation \
      --routed-scaling-factor $rsf ...
done
```

Currently `ROUTED_SCALING_FACTOR = 2.5` is a hardcoded constant in config.py (line 38).
It should be made tunable per run.

---

## HIGH: Bugs That Cause Incorrect Training Results

### 16. Gradient Checkpointing Overwrites `_layer_aux_data` During Backward

**File:** `nanoseek/nanoseek/model.py:1645-1677`
**Status:** UNFIXED
**Symptom:** Load balance bias updates use wrong expert load counts.

This was documented as Issue #4 in Part 1 but has a deeper implication that
wasn't fully analyzed. Here is the complete failure mode:

**Sequence of events:**
1. Forward pass: Layer 5 (MoE) runs, stores `_layer_aux_data[5] = {load_counts: [3,5,2,8,...], H_load: 5.2}`
2. Forward pass continues through layers 6-15
3. Backward pass begins at layer 15, works back toward layer 0
4. Backward reaches layer 5, which is gradient-checkpointed
5. Gradient checkpoint **re-executes** layer 5's forward
6. Re-execution produces **different** routing decisions (floating-point non-determinism from different memory layout during recompute)
7. `_layer_aux_data[5]` is **overwritten** with recomputed load counts
8. After backward completes, `update_load_balance_bias()` reads the recomputed (wrong) load counts

**Why the routing differs on recompute:**
- Gate uses `torch.topk` which is not deterministic for ties
- Floating-point reduction order may differ between forward and recompute
- Even with `torch.backends.cudnn.deterministic=True`, PyTorch does not guarantee
  identical results for recomputed operations with different memory layouts

**Fix (not yet applied):**
```python
# model.py, line 1674-1677, change:
if "aux_loss" in aux_data:
    total_aux_loss = total_aux_loss + aux_data["aux_loss"]
    n_aux_layers += 1
    self._layer_aux_data[i] = aux_data

# to:
if "aux_loss" in aux_data:
    total_aux_loss = total_aux_loss + aux_data["aux_loss"]
    n_aux_layers += 1
    if i not in self._layer_aux_data:  # Guard: keep original forward's data
        self._layer_aux_data[i] = aux_data
```

**Impact:** Bias updates push experts toward balance based on wrong load counts.
Over thousands of steps, this creates a random walk in bias space instead of
systematic balancing. The effect is subtle — H_load may appear reasonable but
expert specialization (I_spec) may be lower than expected.

---

### 17. MTP Receives Post-Norm Hidden States — Potential Scale Mismatch

**File:** `nanoseek/nanoseek/model.py:1679-1766`
**Status:** UNFIXED — needs verification against DeepSeek V3 paper
**Symptom:** MTP module may receive differently-scaled hidden states than intended.

**What happens:**
```python
# Line 1679: hidden_states is post-final-norm
hidden_states = self.norm(hidden_states)  # RMSNorm applied

# Line 1698-1699: passed to _compute_loss, which passes to MTP
loss_dict = self._compute_loss(logits, labels, hidden_states, input_ids, ...)

# Line 1766: MTP receives post-norm hidden_states
_, mtp_loss = self.mtp(hidden_states, input_ids)
```

The MTP module's `forward()` then applies its own `hidden_norm` (line 1091):
```python
h_norm = self.hidden_norm(prev_hidden)  # Second RMSNorm
```

This means hidden states are normalized **twice**: once by the model's final norm,
once by MTP's hidden_norm. Double normalization collapses the magnitude to ~1.0
regardless of the actual hidden state variance.

**DeepSeek V3 paper (Eq. 21):**
```
h'_i^k = M_k [ RMSNorm(Emb(t_{i+k})) ; RMSNorm(h_i^{k-1}) ]
```

Where `h_i^{k-1}` for k=0 is the **last decoder layer's output** (pre-final-norm).
The paper applies RMSNorm inside MTP, so the input should be the raw decoder
output, NOT the post-final-norm version.

**Proposed fix:**
```python
# In NanoSeekModel.forward(), save pre-norm hidden states for MTP:
pre_norm_hidden = hidden_states           # Before final norm
hidden_states = self.norm(hidden_states)  # For logits
...
# In _compute_loss, pass pre_norm_hidden to MTP instead:
_, mtp_loss = self.mtp(pre_norm_hidden, input_ids)
```

**Impact:** With double normalization, MTP's hidden_norm is a no-op (already
normalized). This wastes parameters and may reduce MTP's ability to use hidden
state magnitude as a signal. Severity depends on whether magnitude carries
useful information — likely MEDIUM impact on MTP acceptance rate.

---

## MEDIUM: Performance and Correctness Risks

### 18. Static 8x Padding Will OOM at Ablation/1B Scale

**File:** `nanoseek/nanoseek/model.py:762`
**Status:** UNFIXED — works at anchor scale only
**Symptom:** `CUDA out of memory` when running ablation or 1B training.

**Current code:**
```python
pad_size = (NK + E - 1) // E * 8
```

This fixes the torch.compile recompilation issue (Issue #2 in Part 1) by making
`pad_size` a compile-time constant. But the memory cost scales quadratically:

| Scale    | NK (B×S×K) | avg/expert | pad_size (8x) | padded_input (bf16) |
|----------|------------|------------|---------------|---------------------|
| Anchor   | 131,072    | 2,048      | 16,384        | 1.2 GB              |
| Ablation | 4,194,304  | 65,536     | 524,288       | 86 GB               |
| 1B       | 4,194,304  | 65,536     | 524,288       | 138 GB              |

**Root cause:** NK = batch_size × seq_len × top_k. At ablation scale with
batch_size=16, seq_len=4096, top_k=8: NK = 16 × 4096 × 8 = 524,288.
Then pad_size = (524288 + 63) // 64 × 8 = 65,536. `padded_input` shape is
[64, 65536, 1280] × 2 bytes = 10.5 GB per MoE layer. With 14 MoE layers
and activations, this blows past 80 GB GPU memory.

**Possible fixes (in order of preference):**
1. **Per-scale pad multiplier**: 8x for anchor, 2x for ablation, 1.5x for 1B
2. **torch.compile(dynamic=True)**: Accept recompilation overhead, remove padding
3. **Fused MoE kernel**: Use Megablocks/ScatterMoE-style kernels that don't need padding
4. **Fallback to sequential**: Detect OOM risk, use sequential dispatch

---

### 19. `validate_config` Checks a Constant — Dead Code

**File:** `nanoseek/scripts/pre_train.py:254`
**Status:** UNFIXED
**Symptom:** Validation always passes, gives false confidence.

```python
def validate_config(cfg):
    errors = []
    if cfg.moe.gamma_freeze_ratio != 0.95:
        errors.append(f"RULE 2: gamma_freeze_ratio={cfg.moe.gamma_freeze_ratio}, must be 0.95")
```

`cfg.moe` is a `@property` that returns `SimpleNamespace(gamma_freeze_ratio=DEEPSEEK_GAMMA_FREEZE_RATIO)`.
`DEEPSEEK_GAMMA_FREEZE_RATIO` is a module-level constant `= 0.95`. This check
always passes. It cannot catch anything.

Similarly, `cfg.adam_beta2` and `cfg.max_grad_norm` are dataclass defaults that
can only differ if explicitly overridden, and there's no CLI flag to override them.

**Impact:** LOW — No incorrect behavior, but the validation creates false confidence
that rules are being enforced when they're actually just checking constants.

**Fix:** Either make these values configurable (so validation has something to catch)
or remove the checks and rely on the constants being correct.

---

### 20. Config Property Write-Through Silently Fails

**File:** `nanoseek/nanoseek/config.py:213-264`
**Status:** UNFIXED — affects any future code that writes to `config.moe.*`
**Symptom:** Config changes silently discarded, ablation runs use wrong values.

The `@property` methods `config.moe`, `config.mla`, `config.mtp` return **new**
`SimpleNamespace` objects on every access. Any write is discarded:

```python
config.moe.n_routed_experts = 32  # SILENTLY DOES NOTHING
print(config.moe.n_routed_experts)  # Still 64
```

This was already discovered for Issue #3 in Part 1 (ablation override flags),
but the ROOT CAUSE was not fixed — only the symptoms were patched by writing
to flat config fields instead. Any future developer who writes `config.moe.X = Y`
will hit the same silent failure.

**Fix (not yet applied):** Add `__setattr__` enforcement:
```python
class _FrozenNamespace(SimpleNamespace):
    def __setattr__(self, name, value):
        raise AttributeError(
            f"Cannot set {name} on config.moe — write to config.{name} instead"
        )

@property
def moe(self):
    return _FrozenNamespace(n_routed_experts=self.n_routed_experts, ...)
```

---

### 21. Complex-Number RoPE Prevents torch.compile Fusion

**File:** `nanoseek/nanoseek/model.py:118, 131, 151, 160`
**Status:** UNFIXED — performance issue
**Symptom:** TorchInductor cannot generate fused CUDA kernels for `torch.polar()`,
`torch.view_as_complex()`, or complex multiplication. Entire RoPE subgraph falls
back to eager execution.

**Current implementation:**
```python
# precompute_freqs_cis:
freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # Complex tensor

# apply_rotary_emb:
x_complex = torch.view_as_complex(x.float().contiguous().view(..., -1, 2))
x_out = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
```

**torch.compile output:** `Torchinductor does not support code generation for complex operators`

**Performance impact:** RoPE is ~2-3% of total FLOPs, so the direct impact is small.
But it prevents fusion with adjacent operations (layer norm before attention,
attention score computation), which could save kernel launch overhead.

**Fix (not yet applied):** Rewrite using explicit sin/cos rotation:
```python
def apply_rotary_emb(x, freqs_cos, freqs_sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * freqs_cos - x2 * freqs_sin,
                        x1 * freqs_sin + x2 * freqs_cos], dim=-1).flatten(-2)
```

This requires changing `precompute_freqs_cis` to return `(cos, sin)` instead of
a complex tensor. All 16 MLA layers + MTP MLA share the same freqs, so the
precomputation cost is negligible.

---

### 22. `cudnn.benchmark=False` + `deterministic=True` — Ongoing 5-10% Speed Tax

**File:** `nanoseek/scripts/pre_train.py:205-206`
**Status:** INTENTIONAL — accepted tradeoff for reproducibility
**Symptom:** Every training step is 5-10% slower than necessary.

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

`benchmark=False` disables cuDNN kernel autotuning. Every convolution-like op
uses a default (often suboptimal) algorithm. `deterministic=True` further
restricts to deterministic algorithms, excluding faster non-deterministic options.

**Why it's set this way:** Reproducible ablation results. Two runs with the same
seed must produce identical loss curves to compare hyperparameters.

**When to change:** After HP search is complete and you're running the final
1B graduation run, switch to:
```python
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
```

This saves ~$15-35 on the $350 graduation run.

---

## LOW: Latent Risks (Not Currently Triggered)

### 23. Causal Mask Buffer Mutation During Forward (Absorb Path Only)

**File:** `nanoseek/nanoseek/model.py:283-290`
**Status:** UNFIXED — only affects inference (absorb=True), not training
**Symptom:** torch.compile recompilation if sequence length changes during inference.

```python
def _get_causal_mask(self, seq_len, kv_len, device, dtype):
    if self._cached_causal_mask is None or self._cached_causal_mask.shape[0] < max_dim:
        mask = torch.full((max_dim, max_dim), ...)
        self._cached_causal_mask = mask  # Mutates buffer shape during forward
    return self._cached_causal_mask[:seq_len, :kv_len]
```

This is ONLY used in the absorb path (inference). Training uses the naive path
with SDPA, which handles causal masking internally via `is_causal=True`.

**Impact:** None during training. During inference with variable-length prompts,
this could cause recompilation if the model is compiled with `dynamic=False`.

---

### 24. Best-Fit Document Packing Uses O(buffer_size) Linear Scan

**File:** `nanoseek/nanoseek/dataloader.py:191-210`
**Status:** UNFIXED — acceptable for current scale
**Symptom:** CPU packing thread takes longer than necessary.

```python
for i, doc in enumerate(doc_buffer):  # O(1000) per position
    doc_len = len(doc)
    if doc_len <= remaining and doc_len > best_len:
        best_idx = i
        best_len = doc_len
```

Per-batch cost: O(B × positions_per_row × buffer_size) ≈ O(4 × 20 × 1000) = 80K ops.
This runs in a background thread while the GPU trains, so it's fully overlapped.

**Why it's acceptable:** The background thread finishes packing batch N+1 well
before the GPU finishes training on batch N. The linear scan costs ~0.5ms per
batch vs ~15ms GPU step time. No stall.

**When it matters:** At very small batch sizes (device_batch_size=1) or very
large buffer sizes, the CPU thread could become the bottleneck. Monitor by
checking if `prefetch_queue` is ever empty when the main thread reads from it.

**Fix (if needed):** Replace with `bisect.insort` on a sorted list:
```python
import bisect
sorted_buffer = []  # Sorted by document length
bisect.insort(sorted_buffer, (len(doc), doc))
# Find largest fitting: bisect_right for remaining capacity → O(log N)
```

---

## Summary of All New Issues

| # | Issue | File | Severity | Status |
|---|-------|------|----------|--------|
| 14 | `seq_aux_loss_alpha` never reached Gate | model.py:1260-1321 | CRITICAL | **FIXED** |
| 15 | `routed_scaling_factor=2.5` from wrong arch | config.py:38 | CRITICAL | UNFIXED (HP tuning) |
| 16 | Gradient checkpoint overwrites aux_data | model.py:1677 | HIGH | UNFIXED |
| 17 | MTP receives post-norm hidden (double norm) | model.py:1679-1766 | HIGH | UNFIXED |
| 18 | Static 8x padding OOMs at ablation/1B | model.py:762 | MEDIUM | UNFIXED |
| 19 | `validate_config` checks constants | pre_train.py:254 | MEDIUM | UNFIXED |
| 20 | Config property write-through fails silently | config.py:213-264 | MEDIUM | UNFIXED |
| 21 | Complex RoPE prevents compile fusion | model.py:118-160 | MEDIUM | UNFIXED |
| 22 | cudnn deterministic mode speed tax | pre_train.py:205-206 | LOW | INTENTIONAL |
| 23 | Causal mask buffer mutation (inference only) | model.py:283 | LOW | UNFIXED |
| 24 | Best-fit packing O(N) scan | dataloader.py:191 | LOW | ACCEPTABLE |

## Combined Priority Order (Part 1 + Part 2)

**Fix before ANY training run:**
1. Issue #14: seq_aux_loss_alpha threading (**DONE**)
2. Issue #16: aux_data overwrite guard (1-line fix)
3. Issue #17: MTP post-norm → pre-norm (3-line fix)

**Fix before ablation/1B runs:**
4. Issue #18: Static padding OOM (per-scale multiplier)
5. Issue #15: Tune `routed_scaling_factor` (add to HP grid)

**Fix for performance (after correctness is validated):**
6. Issue #21: Real-valued RoPE (sin/cos rewrite)
7. Issue #22: Disable deterministic mode for graduation run

**Fix for maintainability (no urgency):**
8. Issue #20: Frozen config namespaces
9. Issue #19: Remove dead validation
10. Issue #24: Sorted buffer for packing (only if CPU-bound)
