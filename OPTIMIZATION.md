# NanoSeek Architecture Ablation: Pure MLA (+DSA) vs KDA+MLA Hybrid

## Context

Two competing paradigms for efficient long-context attention at scale:

**DeepSeek path**: Pre-train pure MLA (4K) → YaRN extend (128K) → DSA post-training (learned sparse selection, O(Lk))
**Kimi path**: Pre-train KDA+MLA hybrid (3:1) → native long-context via O(d^2) recurrent state, no DSA needed

NanoSeek currently follows the DeepSeek path but has only Phase 1 built (pure MLA). DSA (Sections 8-9) are empty placeholders. We need to determine which path is better for our 1B training.

**Why two phases**: DSA is a post-training technique (not active during pre-training). Phase A compares pre-training quality at 4K (fair without DSA). Phase B implements DSA and compares the complete long-context systems.

---

## Phase A: Pre-Training Quality Ablation (4K context)

### Goal
Determine if KDA+MLA matches pure MLA on pre-training quality AND is compatible with our MoE + MTP.

### Two Runs

| Run | Config | Notes |
|-----|--------|-------|
| **A0** | MLA(16) — current NanoSeek | Baseline (reuse HP search anchor run) |
| **H1** | KDA(12) + MLA(4) — Kimi-style | Hybrid challenger |

Both at anchor scale (480h, ~55M active), 1.1B tokens, same seed/data/optimizer/schedule.

**H1 layer assignment** (every 4th layer = full MLA):
```
Layer:  0    1    2    3    4    5    6    7    8    9    10   11   12   13   14   15
Attn:  KDA  KDA  KDA  MLA  KDA  KDA  KDA  MLA  KDA  KDA  KDA  MLA  KDA  KDA  KDA  MLA
FFN:   DNS  DNS  MoE  MoE  MoE  MoE  MoE  MoE  MoE  MoE  MoE  MoE  MoE  MoE  MoE  MoE
```
(DNS = dense FFN, MoE = mixture of experts — unchanged from current config)

### What to Measure

1. **ema_val_bpb** — within 0.02 of A0 = viable, >0.03 worse = reject
2. **I_spec dynamics** at steps [100, 500, 1000, final] — most novel measurement: how does expert specialization change under recurrence?
   - Per-layer I_spec: compare KDA-fed MoE layers vs MLA-fed MoE layers WITHIN H1
3. **MTP loss + acceptance rate** — does MTP converge when prev_hidden comes from KDA?
4. **Domain BPB** (code, math, science, web, books) — code/math regression >0.05 = reject (MiniMax failure mode)
5. **H_load per layer** — routing entropy under KDA vs MLA

### Decision Gate (Phase A)

```
H1.bpb > A0 + 0.03           → REJECT KDA. Stay pure MLA. Skip Phase B.
H1.MTP_loss doesn't decrease  → REJECT KDA. MTP incompatible.
H1.code/math_bpb > A0 + 0.05 → REJECT KDA. Reasoning hurt.
H1.I_spec < 0.85 * A0.I_spec → INVESTIGATE routing. Don't proceed blindly.
ALL PASS                      → KDA+MLA viable. Proceed to Phase B.
```

---

## Phase B: Complete System Comparison (8K+ context)

### Goal
Compare the full long-context stacks: MLA+DSA vs KDA+MLA. This is the real architecture decision.

### Prerequisites
- Phase A passes (KDA quality is viable)
- DSA Sections 8-9 implemented (Lightning Indexer + Sparse Attention)

### DSA Implementation (Sections 8-9)

**Section 8: Lightning Indexer** (~150 lines in model.py)

```python
class LightningIndexer(nn.Module):
    """
    Lightweight scoring module: learns which tokens to attend to.
    Score: I_{t,s} = SUM_j w_{t,j} * ReLU(q_{t,j} . k_s)

    Architecture: Multi-head ReLU with learned per-head weights.
    """
    def __init__(self, config):
        self.num_heads = config.sparse.indexer_num_heads    # 4 at anchor
        self.head_dim = config.sparse.indexer_head_dim      # 64 at anchor
        self.topk = config.sparse.topk_tokens               # 2048

        # Indexer projections (separate from MLA)
        self.q_proj = CastLinear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = CastLinear(hidden_size, num_heads * head_dim, bias=False)
        # Per-head weights (query-dependent modulation)
        self.head_weights = CastLinear(hidden_size, num_heads, bias=False)

    def forward(self, hidden_states):
        # Compute indexer Q, K
        q = self.q_proj(hidden_states)  # [B, S, H_I * d_I]
        k = self.k_proj(hidden_states)  # [B, S, H_I * d_I]
        w = self.head_weights(hidden_states)  # [B, S, H_I]

        # Score: sum of ReLU(q.k) weighted by head weights
        scores = compute_indexer_scores(q, k, w)  # [B, S, S] (causal)

        # Select top-k positions per query
        _, indices = scores.topk(self.topk, dim=-1)  # [B, S, topk]
        return indices, scores
```

**Indexer Loss** (KL-divergence, NOT entropy — Correction 1 from paper analysis):
```python
def compute_indexer_loss(self, indexer_scores, mla_attn_probs):
    """
    L_I = D_KL(p_{t,:} || Softmax(I_{t,:}))
    p = actual MLA attention probs (DETACHED)
    I = indexer scores
    """
    return F.kl_div(
        F.log_softmax(indexer_scores, dim=-1),
        mla_attn_probs.detach(),  # CRITICAL: detach from main model
        reduction='batchmean'
    )
```

**Section 9: DSA Sparse Attention** (~200 lines in model.py)

```python
class DSASparseAttention(nn.Module):
    """
    Wraps MLA with sparse token selection via Lightning Indexer.
    MQA mode: gather COMPRESSED c_kv first, expand via wkv_b AFTER.
    """
    def __init__(self, mla, indexer, sparse_config):
        self.mla = mla
        self.indexer = indexer
        self.sparse_config = sparse_config

    def forward(self, hidden_states, ...):
        if not self.sparse_config.enabled or seq_len <= self.sparse_config.activation_threshold:
            # Dense mode (Phase 1 training or short sequences)
            attn_out, attn_probs = self.mla(hidden_states, ..., return_attn_probs=True)
            indexer_loss = self.indexer.compute_loss(hidden_states, attn_probs)
            return attn_out, indexer_loss

        # Sparse mode (Phase 2 / inference)
        indices, _ = self.indexer(hidden_states)  # [B, S, topk]

        # CRITICAL: gather COMPRESSED c_kv, then expand (Correction 3)
        selected_c_kv = gather(c_kv_cache, indices)  # [B, S, topk, kv_lora_rank]
        # Expand AFTER gathering
        selected_kv = selected_c_kv @ self.mla.wkv_b.weight  # Per-head KV

        # Run attention only over selected tokens
        attn_out = sparse_mla_attention(q, selected_kv, indices)
        return attn_out, indexer_loss
```

**Two-Stage LR** (in pre_train.py):
- Stage 1 (indexer warmup): main model FROZEN, indexer LR=1e-3, 1000 steps
- Stage 2 (sparse training): all unfrozen, LR=7.3e-6 (scaled for NanoSeek), indexer DETACHED

### Phase B Runs

| Run | Config | Context | Notes |
|-----|--------|---------|-------|
| **B0** | MLA(16) + DSA | 8K (YaRN) | DeepSeek complete stack |
| **B1** | KDA(12) + MLA(4) | 8K (native) | Kimi stack (no DSA needed) |

Both start from the best Phase A checkpoints, extended to 8K via:
- B0: YaRN scaling → DSA warmup (1000 steps) → sparse fine-tuning
- B1: Direct 8K training (KDA state handles it natively)

### Phase B Measurements

1. **ema_val_bpb at 8K** — quality at long context
2. **Retrieval accuracy** — needle-in-a-haystack at 4K, 6K, 8K positions
3. **Inference throughput** — tokens/sec at 8K context: MLA+DSA vs KDA+MLA
4. **KV cache memory** — MLA+DSA cache (c_kv + indexer cache) vs KDA fixed state + MLA cache
5. **I_spec at 8K** — does expert specialization change at longer context?

---

## Implementation Plan — Phase A

### Step 1: KDA Layer (~2 days)

**File: `nanoseek/nanoseek/nanoseek/model.py`** — new class after MLA (line ~488)

```python
class KimiDeltaAttention(nn.Module):
    """
    Kimi Delta Attention — GatedDeltaNet with channel-wise alpha.
    State: S_t = (I - beta*kk^T) * Diag(alpha) * S_{t-1} + beta*k*v^T
    """
```

**Projections** (full rank QKV, not low-rank like MLA):
- `q_proj, k_proj, v_proj`: CastLinear(hidden_size, num_heads * head_dim)
- `o_proj`: CastLinear(num_heads * head_dim, hidden_size)
- Short conv (kernel=4, depthwise) on Q, K, V
- Channel-wise alpha: low-rank gate `sigmoid(W_up * Swish(W_down * x))`
  - `alpha_down`: hidden_size → head_dim (bottleneck)
  - `alpha_up`: head_dim → num_heads * head_dim
  - `A_log`: nn.Parameter(log(Uniform(1, 16)))
- Beta: scalar per head `sigmoid(Linear(hidden_size → num_heads))`
- Output gate: `o_proj(RMSNorm(o) * silu(gate_proj(x)))`

**Forward**: Uses `fla.ops.gated_delta_rule.chunk_gated_delta_rule` for chunkwise parallel training via WY decomposition.

**Init** (in `_init_weights`):
- QKV: N(0, 1/sqrt(hidden_size))
- o_proj: N(0, 0.006) — matches MLA's wo
- A_log: log(Uniform(1, 16)) — Kimi spec
- Alpha gate: xavier_uniform
- Short conv: default kaiming_uniform

### Step 2: Hybrid Decoder Layer (~0.5 day)

**File: `nanoseek/nanoseek/nanoseek/model.py`** — modify `NanoSeekDecoderLayer.__init__` (~line 1153)

```python
if config.attention_type == "kda_mla" and layer_idx not in config.full_attention_layers:
    self.self_attn = KimiDeltaAttention(config)
else:
    self.self_attn = MultiHeadLatentAttention(config)
```

### Step 3: Config Extension (~0.5 day)

**File: `nanoseek/nanoseek/nanoseek/config.py`**

```python
attention_type: Literal["mla", "kda_mla"] = "mla"
full_attention_interval: int = 4
full_attention_layers: Optional[List[int]] = None  # Auto: [3,7,11,15]
kda_head_dim: int = 128
kda_short_conv_kernel: int = 4
kda_chunk_size: int = 64
```

### Step 4: Extended Eval (~0.5 day)

**File: `nanoseek/nanoseek/nanoseek/eval/information_metrics.py`**

Per-layer-type I_spec: split KDA-fed vs MLA-fed MoE layers.

### Step 5: Tests (~0.5 day)

**File: `nanoseek/nanoseek/tests/test_kda.py`** (new)

- `TestKDALayer`: forward shape, causal masking, alpha is channel-wise, short conv is causal
- `TestHybridModel`: layer assignment, forward pass, meta device init, MTP with hybrid

### Step 6: Run (~3 hrs on A6000)

```bash
python -m nanoseek.scripts.pre_train \
    --run "arch-kda-mla" --scale anchor --seed 42 \
    --attention-type kda_mla \
    --num-iterations 1000 --eval-every 200 --save-every 500 \
    --device-batch-size 4
```

---

## Implementation Plan — Phase B (after Phase A passes)

### Step 7: Lightning Indexer — Section 8 (~2 days)

**File: `nanoseek/nanoseek/nanoseek/model.py`** — Section 8 placeholder (line ~2072)

- `LightningIndexer` class: multi-head ReLU scoring, KL-div loss
- Own KV cache (separate from MLA's c_kv)
- Config already exists: `SparseAttentionConfig` in config.py (indexer_num_heads=4, head_dim=64, topk=2048)

### Step 8: DSA Sparse Attention — Section 9 (~2 days)

**File: `nanoseek/nanoseek/nanoseek/model.py`** — Section 9 placeholder (line ~2077)

- `DSASparseAttention` wrapper: dense mode (indexer trains via KL), sparse mode (top-k selection)
- MQA gather: compressed c_kv first, expand via wkv_b after (Correction 3)
- Sliding window: always attend to recent 512 tokens + top-k selected

### Step 9: Two-Stage Training (~1 day)

**File: `nanoseek/nanoseek/scripts/pre_train.py`**

- Indexer warmup stage: freeze main model, indexer LR=1e-3, 1000 steps
- Sparse training: unfreeze all, indexer detached, phase transition logic

### Step 10: DSA Tests (~0.5 day)

**File: `nanoseek/nanoseek/tests/test_dsa.py`** (new)

- Indexer forward/loss, sparse attention gather order, MQA mode, phase transitions

### Step 11: Phase B Runs (~1 day)

- B0: MLA+DSA at 8K (YaRN + DSA warmup + sparse fine-tuning)
- B1: KDA+MLA at 8K (direct 8K extension)
- Compare: quality, retrieval, throughput, memory, I_spec

---

## Timeline

| Phase | Days | Output |
|-------|------|--------|
| **A: KDA impl + ablation** | 5 | KDA quality verdict, I_spec dynamics, MTP compatibility |
| **B: DSA impl + comparison** | 6 | Complete system comparison: MLA+DSA vs KDA+MLA at 8K |
| **Total** | ~11 | Architecture decision for 1B training |

Phase A is independent — if KDA fails Phase A gates, skip Phase B entirely and stay with pure MLA.

---

## Dependencies

```bash
pip install flash-linear-attention  # Phase A: KDA Triton kernels
# Phase B: no extra deps (DSA uses native PyTorch)
```

---

## Files Modified

### Phase A
| File | Change |
|------|--------|
| `nanoseek/nanoseek/nanoseek/model.py` | Add `KimiDeltaAttention`, modify `NanoSeekDecoderLayer` |
| `nanoseek/nanoseek/nanoseek/config.py` | Add hybrid attention config fields |
| `nanoseek/nanoseek/nanoseek/eval/information_metrics.py` | Per-layer-type I_spec |
| `nanoseek/nanoseek/tests/test_kda.py` | New tests |
| `requirements.txt` / `pyproject.toml` | Add `flash-linear-attention` |

### Phase B
| File | Change |
|------|--------|
| `nanoseek/nanoseek/nanoseek/model.py` | Implement Sections 8-9 (Indexer + DSA) |
| `nanoseek/nanoseek/scripts/pre_train.py` | Two-stage LR, phase transitions |
| `nanoseek/nanoseek/tests/test_dsa.py` | New tests |

## Files Read-Only (reference)

| File | Why |
|------|-----|
| `model.py:184-488` | MLA — pattern for attention layers |
| `model.py:517-654` | Gate/Router — routing input understanding |
| `model.py:827-992` | MTP — hidden state chaining |
| `model.py:1420-1473` | _init_weights — init patterns |
| `model.py:2072-2078` | Section 8-9 placeholders |
| `config.py:425-488` | SparseAttentionConfig — DSA config already defined |
| `docs/PAPER_ANALYSIS_V3_V32.md` | Corrections 1-3 for DSA implementation |
| `docs/TRAINING_BUGS_POSTMORTEM.md` | Init pitfalls |

---

## Verification

### Phase A
1. `pytest tests/test_kda.py -v` — KDA + hybrid tests pass
2. `pytest tests/ -v` — all 120+ existing tests pass
3. `TestMetaDeviceInit` passes for hybrid config
4. H1 runs 10 steps without NaN
5. Apply Phase A decision gate → proceed or reject

### Phase B
1. `pytest tests/test_dsa.py -v` — indexer + DSA tests pass
2. Indexer loss decreases (KL-div converges)
3. Sparse attention output matches dense within tolerance at short context
4. B0 and B1 complete without divergence
5. Compare full metrics → final architecture recommendation


# NanoSeek Low-Level MoE Kernel Optimization Plan

## Context

Previous round implemented high-level optimizations (`.item()` fix, causal mask cache, selective checkpointing, dataloader double-buffering — all done, 119 tests passing). This plan addresses **low-level MoE dispatch optimizations** — but starts with profiling to avoid optimizing blind.

## Engineering Review: What the Numbers Actually Say

The original plan claimed "2-4x speedup on MoE dispatch". Quantitative analysis shows this is **scale-dependent and overstated for 1B**:

| Scale | Seq compute | Launch overhead | Seq total | bmm total | Speedup | Extra memory |
|-------|-------------|----------------|-----------|-----------|---------|-------------|
| **Anchor** (D=480) | 219µs | 1,344µs | 1,563µs | 244µs | **6.4x** | 347MB |
| **1B** (D=2048) | 4,998µs | 1,344µs | 6,342µs | 5,261µs | **1.2x** | 6.0GB |

**Why the difference:** At anchor scale, kernel launch overhead (192 launches × 7µs = 1,344µs) dominates compute (219µs). Batching eliminates launches → huge win. At 1B scale, compute dominates launches → modest win, but 170x more peak memory.

**Key insight the `.item()` fix already captured:** The `.item()` fix (previous round) already eliminated the CUDA pipeline drain that was the largest stall. The remaining 192 kernel launches have ~7µs overhead each (vs ~50µs per `.item()` sync). Diminishing returns.

### Correctness Analysis: Does bmm change training?

**No.** Batched dispatch computes the identical function:
- Same weights (stacked from same parameters, `torch.stack` preserves autograd)
- Same matmul operations (bmm = batch of individual matmuls)
- Padding with zeros is gradient-neutral (0 × dL/dout = 0)
- Only difference: float non-associativity (~1e-6), expected and harmless

### Memory Risk at 1B Scale

The padded input `[64, max_count, 2048]` alone is **2.3GB in bf16**. With intermediates, peak is ~6GB extra per MoE layer. With gradient checkpointing recomputing forward in backward, these tensors are created **twice**. On H100 (80GB) this may be fine, but it could push OOM depending on batch size.

**Decision: Profile first, then decide whether bmm is worth it at each scale.**

---

## Optimization 1: Batched Expert Dispatch via Stacked Weights + `torch.bmm`
**File:** [model.py:671-810](nanoseek/nanoseek/model.py#L671-L810)
**Impact (corrected):** ~6x at anchor (launch-bound), ~1.2x at 1B (compute-bound)
**Requires:** Pure PyTorch (no Triton needed)
**Confidence:** 95% at anchor, 70% at 1B (memory pressure may negate gains)

### Problem
Current dispatch (lines 777-783) processes each expert sequentially:
```python
for expert_idx in range(64):  # 64 separate kernel launch groups
    sorted_output[offset:offset+cnt] = self.routed_experts[expert_idx](expert_input)
    # Each expert does 3 matmuls = 192 total kernel launches per MoE layer
```
With 14 MoE layers: **2,688 kernel launches per forward pass**. Each launch has ~5-10µs overhead and underutilizes GPU SMs (small per-expert batch sizes).

### Fix: Stack weights + `torch.bmm` (3 calls instead of 192)

**Key design constraint:** Keep `nn.ModuleList` for parameter names, checkpointing, and `_init_weights()` compatibility. Build stacked weight views in forward.

**A) `_stack_expert_weights()` method on MoE:**
```python
def _stack_expert_weights(self):
    # torch.stack preserves autograd — gradients flow back to individual params
    w_gate = torch.stack([e.w_gate.weight for e in self.routed_experts])  # [E, inter, D]
    w_up   = torch.stack([e.w_up.weight   for e in self.routed_experts])  # [E, inter, D]
    w_down = torch.stack([e.w_down.weight for e in self.routed_experts])  # [E, D, inter]
    return w_gate, w_up, w_down
```

**B) Concatenate gate+up into single matmul (2 bmm calls instead of 3):**
Since `w_gate` and `w_up` have identical shapes `[E, inter, D]`, concatenate to `[E, 2*inter, D]`:
```python
w_gate_up = torch.cat([wg, wu], dim=1)              # [E, 2*inter, D]
gate_up = torch.bmm(padded_input, w_gate_up.transpose(1,2))  # [E, max_count, 2*inter]
gate_out, up_out = gate_up.chunk(2, dim=-1)
hidden = F.silu(gate_out) * up_out                   # [E, max_count, inter]
out = torch.bmm(hidden, wd.transpose(1,2))           # [E, max_count, D]
```
Result: **2 bmm calls** instead of 192 individual matmuls.

**C) Pad tokens into `[E, max_count, D]` tensor:**
```python
# Use sorted_indices (already computed) for vectorized pad/unpad:
position_in_expert = _cumcount(sorted_indices)  # position of each token within its expert's batch

padded_input = sorted_x.new_zeros(E, max_count, D)
padded_input[sorted_indices, position_in_expert] = sorted_x  # scatter in

# ... bmm operations ...

sorted_output = out[sorted_indices, position_in_expert]        # gather out
```
`_cumcount` computes each token's index within its expert group using `torch.cumsum` on a one-hot mask — fully vectorized, no Python loops.

**D) Dispatch decision (batched on CUDA, sequential on CPU):**
```python
use_batched = sorted_x.is_cuda and E >= 8
if use_batched:
    sorted_output = self._batched_expert_forward(sorted_x, sorted_indices, expert_counts)
else:
    # Existing sequential fallback for CPU / testing
    ...
```

**E) Padding ceiling for torch.compile:**
Pad `max_count` to `ceil(N*K/E * 1.5)` (fixed ceiling) so that `torch.compile(dynamic=False)` doesn't recompile when load balance shifts slightly between steps.

### Why `torch.stack` preserves gradients
`torch.stack` is a differentiable op in PyTorch's autograd. On `loss.backward()`, gradients flow through the stacked tensor back to each `expert.w_gate.weight` parameter. This is the standard pattern used by Megablocks and ScatterMoE.

### Waste and memory estimate (corrected)

**Compute waste** (with good load balancing):
- Average tokens/expert: `N*K/E`
- Max tokens/expert: ~5% above average
- Padding waste: ~5% extra FLOPs — acceptable

**Memory overhead** (the real concern):
| Scale | Stacked weights (fp32) | Padded input (bf16) | Intermediates | Total extra |
|-------|----------------------|-------------------|-------------|------------|
| Anchor | 66 MB | 132 MB | 149 MB | **347 MB** ✓ |
| 1B | 1,208 MB | 2,255 MB | 2,500 MB | **~6 GB** ⚠️ |

At 1B scale, the 6GB extra is significant. With gradient checkpointing, stacks are recomputed in backward, so peak memory sees this twice. **Must verify with profiler before committing.**

**Net gain (corrected):**
- Anchor: 6.4x dispatch speedup, 347MB extra → **clear win**
- 1B: 1.2x dispatch speedup, 6GB extra → **marginal, profile first**

---

## Optimization 2: Fused Weighted Scatter-Add (Triton kernel, GPU-only)
**File:** New `triton_kernels.py` + [model.py:786-794](nanoseek/nanoseek/model.py#L786-L794)
**Impact:** 5-10% on MoE forward (memory-bound savings)
**Requires:** Triton (available on RunPod CUDA PyTorch, fallback to PyTorch on CPU)
**Confidence:** 85%

### Problem
After expert dispatch, two memory-bound operations happen sequentially:
```python
sorted_output *= sorted_weights.unsqueeze(-1)                    # [N*K, D] read+write
routed_output.scatter_add_(0, idx.expand_as(sorted_output), sorted_output)  # [N*K, D] read + [N, D] atomic write
```
The multiply materializes a full `[N*K, D]` intermediate (e.g., 512MB at bf16), then scatter_add reads it again.

### Fix: Single Triton kernel
```python
@triton.jit
def _weighted_scatter_add_kernel(sorted_out_ptr, weights_ptr, idx_ptr, output_ptr, NK, D, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    w = tl.load(weights_ptr + row)
    target = tl.load(idx_ptr + row)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    vals = tl.load(sorted_out_ptr + row * D + cols, mask=mask)
    tl.atomic_add(output_ptr + target * D + cols, vals * w, mask=mask)
```

Wrap with `torch.autograd.Function` (forward only in Triton, backward via PyTorch unfused — gather is already fast).

**Fallback:** Two-line PyTorch version when Triton unavailable.

### Risk: Atomic add contention
With K=8, each output position accumulates exactly 8 values. This is minimal contention — atomics perform well at this level.

---

## Optimization 3: Triton Fused SwiGLU (GPU-only, optional)
**File:** `triton_kernels.py`
**Impact:** Additional 5-10% on MoE forward beyond Opt 1
**Requires:** Triton
**Confidence:** 80%

### Problem
After the gate+up bmm, the pointwise `F.silu(gate_out) * up_out` materializes an `[E, max_count, inter]` intermediate.

### Fix: Triton kernel fusing silu + element-wise multiply
```python
@triton.jit
def _swiglu_fwd_kernel(gate_ptr, up_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    g = tl.load(gate_ptr + offs, mask=mask)
    u = tl.load(up_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, g * tl.sigmoid(g.to(tl.float32)).to(g.dtype) * u, mask=mask)
```

**Lower priority** than Opt 1 and 2 because:
- `torch.compile` may already fuse `silu + multiply` automatically
- The concatenated gate+up approach (Opt 1B) already reduces memory traffic
- Pure pointwise — moderate savings

---

## NOT Implementing (and why)

| Technique | Why Skip |
|-----------|----------|
| **Custom CUDA C++ kernels** | Triton covers all our needs, simpler to maintain |
| **Expert parallelism / all-to-all** | Single node, 64 experts fit in memory |
| **Block-sparse matrix formats** | Experts aren't internally sparse — each expert's SwiGLU is dense |
| **Topology-aware routing** | Single node with NVLink, no inter-node communication |
| **Custom gating kernel** | Gating is <1% of compute, not worth the complexity |
| **Expert replication** | Load balance bias handles imbalance without replicating |

---

## Implementation Order (Profile → Implement → Measure)

### Step 0: Profiling Harness (DO THIS FIRST)

Create `scripts/profile_moe.py` — a standalone script that:

1. **Builds anchor-scale model** on GPU (or CPU with `--cpu` flag)
2. **Runs 20 forward+backward passes** (first 5 warmup, last 15 timed)
3. **Reports per-component timing:**
   - Gate routing (router projection + sigmoid + group selection + top-K)
   - Expert dispatch (sort + pad + matmul loop)
   - Weight application + scatter combine
   - Shared expert forward
   - Total MoE forward
   - Total model forward
4. **Reports memory:** `torch.cuda.max_memory_allocated()` and `torch.cuda.memory_reserved()`
5. **Uses `torch.cuda.Event` for accurate GPU timing** (not wall-clock, which includes CPU overhead)

```python
# Timing pattern (correct for async CUDA):
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
# ... operation ...
end.record()
torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)
```

This gives us a **baseline** before any optimization, and a **repeatable benchmark** after each change.

**Key metrics to track:**
- `moe_dispatch_ms`: time in expert dispatch loop (sort→expert loop→scatter)
- `moe_total_ms`: total MoE forward (gate + dispatch + shared)
- `model_forward_ms`: full model forward
- `model_backward_ms`: full model backward
- `peak_memory_gb`: GPU peak allocated memory
- `mfu_percent`: MFU calculated from FLOPs / time / peak_flops

### Step 1: Batched Expert Dispatch (pure PyTorch)

Only proceed after profiling confirms dispatch is the bottleneck.

**Implementation (as described in Opt 1 above) with these corrections:**

A. Add `_cumcount()` — vectorized position-in-expert:
```python
def _cumcount(sorted_indices, boundaries):
    """For each token, compute its position within its expert's batch."""
    # sorted_indices[i] = which expert token i is assigned to
    # boundaries[e] = start index of expert e in sorted array
    # position = i - boundaries[sorted_indices[i]]
    return torch.arange(len(sorted_indices), device=sorted_indices.device) - boundaries[sorted_indices]
```

B. Add waste threshold: **skip bmm if max_count / avg_count > 1.5** (>50% wasted compute)
```python
waste_ratio = max_count / max(avg_count, 1)
use_batched = sorted_x.is_cuda and E >= 8 and waste_ratio < 1.5
```

C. Add scale-aware dispatch: at 1B scale where compute >> launch overhead, the sequential path with `.tolist()` may be faster than bmm (due to 6GB extra memory pressure). **Let profiling decide.**

D. Write equivalence tests (output + gradient match within atol=1e-5)

E. Re-run profiler to measure actual improvement

### Step 2: Triton Fused Kernels (GPU-only, only if profiling justifies)

Only implement after Step 1 profiling shows scatter_add or SwiGLU are significant bottlenecks (>5% of MoE time).

### Step 3: Skip unless profiling shows need

---

## Safety Checks (Things That Must Not Break)

| Check | Why | How to Verify |
|-------|-----|---------------|
| **Gradient flow** | `torch.stack` must preserve autograd graph | Test: compare `param.grad` between sequential and batched paths |
| **Padding neutrality** | Zeros in padded positions must not affect loss or gradients | Test: verify padded positions produce zero output and zero gradient |
| **Checkpoint compat** | Parameter names must not change | Test: `model.state_dict().keys()` identical before/after |
| **torch.compile** | No new graph breaks introduced | Test: `TORCH_LOGS=graph_breaks python -c "..."` shows no breaks |
| **Determinism** | Same seed → same loss trajectory | Test: 50 steps with fixed seed, compare loss within 0.1% |
| **Memory budget** | Peak memory must fit GPU | Test: `torch.cuda.max_memory_allocated()` before/after |

## Files to Modify

| File | Changes |
|------|---------|
| [profile_moe.py](nanoseek/scripts/profile_moe.py) | **NEW**: Profiling harness with per-component timing |
| [model.py](nanoseek/nanoseek/model.py) | MoE class: add `_cumcount`, `_stack_expert_weights`, `_batched_expert_forward`, dispatch decision with waste threshold |
| [test_moe.py](nanoseek/tests/test_moe.py) | Add batched vs sequential equivalence test, gradient test |
| [triton_kernels.py](nanoseek/nanoseek/triton_kernels.py) | **NEW** (Step 2 only): weighted_scatter_add, optional SwiGLU |

## Key Existing Code to Reuse

- `Expert` class (model.py:513-528): weight shapes, CastLinear dtype cast pattern
- `MoE.forward()` sort+dispatch (model.py:744-794): keep sort, replace dispatch loop
- `expert_counts` / `expert_boundaries` (model.py:764-767): reuse for `_cumcount`
- `get_nanoseek_anchor_config()` (config.py): for profiling at anchor scale

## Verification Protocol

1. **Baseline profile**: Run `profile_moe.py` on current code → save numbers
2. **After Step 1**: Run `profile_moe.py` again → compare
3. **Correctness**: 119 existing tests pass + new equivalence tests
4. **Loss trajectory**: 50 training steps must match within 0.1% (same seed)
5. **Memory**: Peak memory increase < 1GB at anchor scale, < 8GB at 1B scale
6. **Decision gate**: If Step 1 gives <10% improvement on MoE dispatch at target scale, skip Steps 2-3
