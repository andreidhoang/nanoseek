# NanoSeek Training Bugs Postmortem
## Root Cause Analysis — gate1-smoke v1-v4 (March 18-19, 2026)

All smoke tests v1-v4 were training with critically broken initialization.
This document records every root cause to prevent recurrence.

---

## BUG 1: RMSNorm Weights Initialized to Zero (CRITICAL)

**Symptom**: Model produces loss = ln(V) = 10.3972 at init (perfectly uniform logits).
Main loss decreased slowly over training but from a much worse starting point than intended.

**Root Cause**: When building on meta device (`with torch.device("meta")`),
`to_empty(device)` fills all tensors with zeros. `init_weights()` only handled
`nn.Linear` and `nn.Embedding` — it did NOT reinitialize `RMSNorm.weight`.

RMSNorm formula: `output = (x * rsqrt(mean(x^2) + eps)) * weight`

With `weight = 0`, every RMSNorm layer outputs ALL ZEROS, killing all signal
through the entire model. The model degenerates to `lm_head(zeros)` = uniform logits.

**Why training still showed loss decrease**: RMSNorm weights are `nn.Parameter`,
so the optimizer updates them. After a few steps, they drift from zero and signal
partially returns — but from a catastrophically bad starting point.

**Fix**: Added `RMSNorm` handling to `_init_weights()`:
```python
elif isinstance(module, RMSNorm):
    torch.nn.init.ones_(module.weight)
```

**File**: `nanoseek/model.py:_init_weights()`

---

## BUG 2: RoPE freqs_cis Buffers Initialized to Zero (CRITICAL)

**Symptom**: No positional encoding — every token position looks identical to the model.
Attention patterns become position-independent.

**Root Cause**: `freqs_cis` is a non-persistent buffer (`register_buffer(..., persistent=False)`)
containing precomputed complex exponentials (cos/sin) for RoPE. When built on meta device,
the buffer is created without data. `to_empty()` fills it with zeros. `init_weights()` only
handles parameters, not buffers.

Correct values: complex tensor with real/imag parts in [-1, 1].
Actual values after to_empty: all zeros.

**Impact**: Affects ALL attention layers (16 main + 1 MTP = 17 MLA instances).
Without position encoding, the model cannot learn token ordering — a fundamental
requirement for language modeling.

**Fix**: Added `_reinit_buffers()` method that recomputes `freqs_cis` for all MLA
instances (both main model and MTP):
```python
def _reinit_buffers(self):
    for layer in self.layers:
        layer.self_attn.freqs_cis = precompute_freqs_cis(...).to(device)
    if self.mtp:
        for mtp_mod in self.mtp.mtp_modules:
            mtp_mod.transformer.attn.freqs_cis = precompute_freqs_cis(...).to(device)
```

**File**: `nanoseek/model.py:_reinit_buffers()`

---

## BUG 3: MTP concat_proj Zero-Initialized (HIGH)

**Symptom**: MTP loss frozen at exactly ln(V) = 10.3972 for all training steps.
Zero batch-to-batch variance.

**Root Cause**: `_init_weights()` explicitly set `concat_proj.weight` to zeros:
```python
torch.nn.init.zeros_(mtp_mod.concat_proj.weight)
```
The comment said "MTP heads should start as identity" — but zero weight is NOT
identity, it's the zero function. All MTP hidden states become zero, producing
uniform logits through lm_head.

**Why it was hidden**: Bug 1 (zero RMSNorm) already made MTP output zero, so
the concat_proj zero-init was not independently observable until Bug 1 was fixed.

**Fix**: Changed to small random init matching other output projections:
```python
torch.nn.init.normal_(mtp_mod.concat_proj.weight, mean=0.0, std=output_std)
torch.nn.init.normal_(mtp_mod.transformer.attn.wo.weight, mean=0.0, std=output_std)
torch.nn.init.normal_(mtp_mod.transformer.ffn.w_down.weight, mean=0.0, std=output_std)
```

**File**: `nanoseek/model.py:_init_weights()`

---

## BUG 4: EMA Decay Too Aggressive for Short Runs (MEDIUM)

**Symptom**: EMA val BPB stuck at 3.1725 for all eval checkpoints (step 0, 50, 100).

**Root Cause**: EMA decay = 0.9999. Each update blends with weight (1-0.9999) = 0.0001.
With updates every 10 steps, 100 training steps = 10 EMA updates.
After 10 updates: shadow weights = 99.9% initial weights + 0.1% current weights.
BPB barely changes because eval weights are essentially frozen.

**Fix**: Karras-style decay warmup: `effective_decay = min(decay, 1 - 1/(1 + count))`.
First update uses decay=0.5 (50% new weights), ramping to target 0.9999 over time.
After 10 updates with warmup: shadow = 91% of the way to current (vs 0.1% without).

**File**: `scripts/pre_train.py:EMATracker.step()`

---

## BUG 5: Dead Expert Check Tensor-to-Numpy Crash (MEDIUM)

**Symptom**: `Dead expert check failed: can't convert cuda:0 device type tensor to numpy`

**Root Cause**: Two issues:
1. `get_expert_load_stats()` didn't return `per_layer` key. `compute_dead_experts()`
   expected it, fell through to aggregate path.
2. GPU tensors passed through computation chain without `.cpu()` before `.numpy()`.

**Fix**:
- Added `per_layer` dict to `get_expert_load_stats()` return value
- Added `.detach().float().cpu()` throughout `compute_dead_experts()`
- Added `.float()` before `.cpu().numpy()` in gini computation (bf16 not numpy-compatible)

**Files**: `nanoseek/model.py:get_expert_load_stats()`, `eval/moe_diagnostics.py`

---

## BUG 6: I_spec Always Zero (sklearn Missing + NumPy Compat) (MEDIUM)

**Symptom**: I_spec = 0.0 at every eval. Log: "sklearn not available, using simplified I_spec"

**Root Cause**:
1. scikit-learn not installed — falls back to simplified version that returns 0
2. When sklearn IS available: `.numpy()` called on bfloat16 tensors (not supported)
3. `np.sum(generator)` deprecated in NumPy 2.x

**Fix**:
- Installed scikit-learn
- Added `.float()` before `.numpy()` in `compute_i_spec()` lines 127-128
- Replaced `np.sum(p * np.log(p) for p in probs if p > 0)` with vectorized
  `np.sum(nonzero * np.log(nonzero + 1e-10))`

**File**: `eval/information_metrics.py`

---

## BUG 7: Gradient Spike False Alarms During Warmup (LOW)

**Symptom**: 18 gradient spike warnings in first 19 steps, plus 34 "blowup zone"
warnings from step 67 onward. Logs completely dominated by false alarm noise.

**Root Cause**:
1. No warmup grace period in `TrainingHealthMonitor`. During LR warmup, gradient
   norms grow monotonically (LR ramps 0→peak). Z-score detector flags every step
   because EMA hasn't caught up to the monotonic trend.
2. Absolute threshold of 0.5 hardcoded from OLMo (7B scale). At 85M anchor scale,
   gradient norms are naturally higher.

**Fix**:
- Added `warmup_steps` parameter to `TrainingHealthMonitor`
- Skip spike detection during warmup (EMA still updated for calibration)
- Removed hardcoded 0.5 absolute threshold (rely on z-score which is relative)
- Expert collapse checks still active during warmup (safety)

**File**: `scripts/pre_train.py:TrainingHealthMonitor`

---

## Impact Assessment

| Bug | v1-v3 Runs | v4 Run | v5 Run |
|-----|-----------|--------|--------|
| 1. RMSNorm zeros | BROKEN | BROKEN | FIXED |
| 2. RoPE zeros | BROKEN | BROKEN | FIXED |
| 3. MTP zero init | BROKEN | FIXED (but masked by #1) | FIXED |
| 4. EMA too slow | BROKEN | FIXED | FIXED |
| 5. Dead expert crash | BROKEN | FIXED | FIXED |
| 6. I_spec always 0 | BROKEN | FIXED | FIXED |
| 7. False alarm spam | BROKEN | FIXED | FIXED |

### v5 vs v4 Metrics (first 37 steps)

| Metric | v4 (Bugs 1+2 active) | v5 (all fixed) |
|--------|---------------------|----------------|
| Main loss step 0 | 10.3972 (= random) | 10.93 (non-random) |
| Main loss step 37 | 10.38 | **6.35** |
| MTP loss step 0 | 10.3972 (frozen) | 10.93 |
| MTP loss step 37 | 10.3972 (FROZEN) | **6.53** |
| H_load | 3.00 (stuck) | **5.72** (healthy) |
| Grad spike warnings | 0 (fixed in v4) | 0 |

---

## Prevention Rules

1. **Any `to_empty()` call MUST be followed by full reinitialization** — not just
   parameters but also buffers. Add a `_reinit_buffers()` method to any model
   that uses `register_buffer()`.

2. **Never zero-init a projection that feeds a normalization layer** — the norm
   will produce NaN or degenerate output. Use small random init (std=0.006) instead.

3. **Test the meta device → to_empty → init_weights path separately** — don't
   assume direct construction and meta-device construction produce the same model.
   Add a test: `assert meta_model_loss != ln(V)`.

4. **EMA trackers need warmup** for short runs — Karras-style `min(decay, 1-1/(1+n))`
   ensures shadow weights track the model from the first update.

5. **Always `.cpu().float()` before `.numpy()`** — GPU tensors and bfloat16 tensors
   cannot be converted to numpy directly.

6. **Spike detectors need warmup grace periods** — monotonic trends during LR warmup
   are not spikes. Only alert after warmup completes.

7. **List ALL non-parameter tensors (buffers) and verify their initialization** —
   `model.named_buffers()` shows everything. Each buffer should have a documented
   initialization strategy.
