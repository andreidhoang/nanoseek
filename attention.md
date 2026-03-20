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
