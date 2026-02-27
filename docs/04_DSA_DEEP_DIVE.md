# DeepSeek Sparse Attention (DSA) — First Principles Deep Dive

> **Perspective:** Senior Research Scientist, Frontier Efficiency Group  
> **Codebase:** NanoSeek — educational DeepSeek V3.2 implementation  
> **Date:** February 2026  
> **Prerequisites:** Familiarity with transformer attention, MLA (see NanoSeek docs), basic PyTorch

---

## Quick Context — O(L²) → O(Lk) Complexity Reduction

Standard self-attention computes a full score matrix between every query and every key. For sequence length L, that is L² dot products per head per layer. At L = 32,768 this is over one billion operations *per head*. When your model has 16+ heads across 16+ layers, the cost becomes the dominant bottleneck — not in FLOPs alone, but in memory bandwidth for materializing and reading the attention matrix.

DeepSeek Sparse Attention (DSA) replaces O(L²) with O(Lk) where k is a fixed budget of tokens each query attends to (default k = 2,048). The key insight: most attention mass concentrates on a small subset of keys. If you can *predict* which keys matter before computing full attention, you only need to compute attention over those k tokens.

**The concrete numbers for NanoSeek:**

| Metric | Dense (MLA) | Sparse (DSA) | Ratio |
|--------|-------------|--------------|-------|
| Attention ops per query | L | k = 2,048 | L/2048 |
| At L = 4,096 | 4,096 | 4,096 (dense) | 1× (below threshold) |
| At L = 8,192 | 8,192 | 2,048 | 4× reduction |
| At L = 32,768 | 32,768 | 2,048 | 16× reduction |

DSA achieves this through a lightweight **Lightning Indexer** — a small multi-head network that scores every (query, key) pair in compressed space, then selects the top-k. The indexer operates on MLA's *already-compressed* representations (dimensions 430 for Q, 143 for KV), making the scoring step itself very cheap.

**What DSA is NOT:**
- Not a sliding window (it selects *globally*, not just nearby tokens)
- Not random sparse attention (selection is learned, not stochastic)
- Not block-sparse attention (operates at individual token granularity)
- Not approximate attention (once tokens are selected, exact attention is computed)

---

## 🔴 STAGE 1: INPUTS

### What enters the DSA layer

Every decoder layer in NanoSeek receives hidden states from the previous layer (or embedding layer for layer 0). At the DSA entry point, we have:

```
hidden_states: Tensor[batch_size, seq_len, hidden_size]
               Tensor[B, L, 2048]
```

In NanoSeek's default configuration:
- `hidden_size` = 2,048
- `seq_len` = 4,096 (Phase 1 training) or 8,192 (Phase 2 / inference)
- `batch_size` = variable (typically 1–128)

The DSA module also receives:
- `attention_mask`: optional causal mask `[1, 1, L, L]` or `None` (auto-generated)
- `position_ids`: optional position indices for RoPE `[B, L]`
- `past_key_value`: optional KV cache tuple for incremental decoding
- `output_indexer_loss`: bool flag — whether to compute the indexer training loss

### The compressed representations — DSA's secret weapon

Before DSA does *anything* sparse, it extracts compressed representations via MLA's down-projection layers. This is critical: the indexer never sees the full 2,048-dimensional hidden states. It works entirely in compressed space.

```
┌──────────────────────────────────────────────────────────────┐
│  hidden_states [B, L, 2048]                                  │
│         │                          │                         │
│         ▼                          ▼                         │
│  ┌──────────────┐          ┌───────────────────┐             │
│  │ wq_a (Linear)│          │ wkv_a (Linear)    │             │
│  │ 2048 → 430   │          │ 2048 → 143 + 32   │             │
│  └──────┬───────┘          └─────────┬─────────┘             │
│         │                            │                       │
│         ▼                      ┌─────┴─────┐                 │
│  ┌──────────────┐              │   split    │                │
│  │  q_norm      │              ▼            ▼                │
│  │  (RMSNorm)   │     kv_compressed   k_pe (RoPE)           │
│  └──────┬───────┘      [B,L,143]      [B,L,32]              │
│         ▼                                                    │
│  q_compressed                                                │
│  [B, L, 430]                                                 │
└──────────────────────────────────────────────────────────────┘
```

From `model/model.py` — the actual implementation:

```python
def _get_compressed_representations(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    q_compressed = self.mla.wq_a(hidden_states)       # [B, L, 430]
    q_compressed = self.mla.q_norm(q_compressed)

    kv = self.mla.wkv_a(hidden_states)                 # [B, L, 175]
    kv_compressed, k_pe = torch.split(
        kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )                                                   # [B,L,143], [B,L,32]
    kv_compressed = self.mla.kv_norm(kv_compressed)

    return q_compressed, kv_compressed, k_pe
```

These three tensors — `q_compressed`, `kv_compressed`, and `k_pe` — are the inputs to both the Lightning Indexer and the subsequent attention computation.

### Why compression-first matters

The standard approach to sparse attention would be: compute full Q and K, score them, select top-k, then compute attention on the selection. This is wasteful — you expand to full dimension just to select, then expand again.

DSA inverts this: score in *compressed* space (430 + 143 dims), select top-k, *then* expand only the selected tokens to full dimension. The indexer's scoring cost is proportional to `430 × 143`, not `(16 × 96) × (16 × 96)` — roughly 10× cheaper.

---

## 🔴 STAGE 2: MODEL ARCHITECTURE

### 2.1 Why sparse attention? The quadratic wall

Self-attention's O(L²) complexity creates a hard wall:

```
Sequence Length    Attention Ops (per head)    Memory for Scores
L = 1,024         1,048,576                   4 MB  (fp32)
L = 4,096         16,777,216                  64 MB
L = 8,192         67,108,864                  256 MB
L = 32,768        1,073,741,824               4 GB
L = 131,072       17,179,869,184              64 GB  ← DeepSeek V3 target
```

For NanoSeek with 16 heads and 16 layers, multiply the per-head numbers by 256. At L = 32K, that's ~275 billion attention operations and ~1 TB of score memory *per forward pass*.

MLA already solves the KV *cache* problem (23× compression), but it does NOT reduce the attention *computation*. You still compute the full L² score matrix — you just reconstruct K and V from compressed representations first. DSA addresses the complementary problem: reducing the score matrix itself.

### 2.2 Lightning Indexer — how it learns which tokens matter

The Lightning Indexer is a small, specialized network that predicts attention importance scores without computing full attention. Here is its complete architecture:

```python
class LightningIndexer(nn.Module):
    def __init__(self, q_lora_rank, kv_lora_rank, num_heads=4, head_dim=64):
        self.q_proj = nn.Linear(q_lora_rank, num_heads * head_dim, bias=False)
        # Linear(430 → 4×64 = 256)

        self.k_proj = nn.Linear(kv_lora_rank, num_heads * head_dim, bias=False)
        # Linear(143 → 4×64 = 256)

        self.head_weights = nn.Parameter(torch.ones(num_heads))
        # Learnable [4] — per-head importance weights
```

**Parameter count:** 430 × 256 + 143 × 256 + 4 = **146,692 parameters** per layer. This is tiny compared to the MLA it serves (~4.3M parameters per layer). The indexer is less than 3.5% of the attention layer's parameters.

**Forward pass — step by step:**

```
q_compressed [B, L_q, 430]    kv_compressed [B, L_kv, 143]
        │                              │
        ▼                              ▼
   q_proj (430→256)              k_proj (143→256)
        │                              │
        ▼                              ▼
   reshape to                    reshape to
   [B, L_q, 4, 64]              [B, L_kv, 4, 64]
        │                              │
        └──────────┬───────────────────┘
                   ▼
         einsum('bqhd,bkhd->bhqk')
         ← Multi-head dot products
                   │
                   ▼
              ReLU activation
         ← Zero out negative scores
                   │
                   ▼
         Weighted sum across heads
         (head_weights: [4] → [1,4,1,1])
                   │
                   ▼
            index_scores [B, L_q, L_kv]
```

From `model/model.py`:

```python
def forward(self, q_compressed, kv_compressed, causal_mask=None):
    batch_size, q_len, _ = q_compressed.shape
    kv_len = kv_compressed.shape[1]

    q_idx = self.q_proj(q_compressed).view(batch_size, q_len, self.num_heads, self.head_dim)
    k_idx = self.k_proj(kv_compressed).view(batch_size, kv_len, self.num_heads, self.head_dim)

    scores = torch.einsum('bqhd,bkhd->bhqk', q_idx, k_idx)
    scores = F.relu(scores)                                      # Non-negative!

    weights = self.head_weights.view(1, self.num_heads, 1, 1)
    index_scores = (scores * weights).sum(dim=1)                 # [B, q_len, kv_len]

    if causal_mask is not None:
        index_scores = index_scores + causal_mask.unsqueeze(0)   # Enforce causality

    return index_scores
```

**Why ReLU, not softmax?** Softmax would normalize scores to sum to 1, making all tokens compete. ReLU allows scores to be independently large or zero. This is crucial: the indexer needs to say "these 2,048 tokens are important" without forcing a distribution. Tokens that are irrelevant get exactly zero. Tokens that matter get positive scores proportional to their importance.

**Why multiple indexer heads?** Different heads can learn to detect different types of importance:
- Head 1 might detect syntactic proximity (nearby tokens)
- Head 2 might detect semantic relevance (topically similar tokens)
- Head 3 might detect structural markers (punctuation, section boundaries)
- Head 4 might detect repeated entities (coreference)

The learnable `head_weights` let the model decide the relative importance of each detection pattern. During training, these weights are updated to align with actual attention patterns in the main MLA.

**Top-k selection:**

```python
def select_topk(self, index_scores, k):
    k = min(k, index_scores.shape[-1])
    return torch.topk(index_scores, k=k, dim=-1, sorted=False)
    # Returns: values [B, L_q, k], indices [B, L_q, k]
```

`sorted=False` is a deliberate optimization — we don't need the selected tokens in order, just their identities. This makes `topk` faster on GPU.

### 2.3 Dense vs Sparse mode decision

DSA is a *wrapper* around MLA, not a replacement. Every DSA layer contains a complete MLA instance and decides at runtime whether to use dense or sparse attention:

```python
use_sparse = (
    self.sparse_config.enabled and           # DSA turned on?
    kv_len >= self.sparse_config.activation_threshold and   # Long enough sequence?
    not in_warmup                            # Past warmup period?
)
```

The three conditions:

1. **`enabled`**: Master switch. `False` during Phase 1 training, `True` during Phase 2 and inference.
2. **`activation_threshold`** (default 4,096): Sparse attention only activates for sequences longer than this. For short sequences, dense attention is cheaper than the indexer overhead.
3. **`in_warmup`**: During the first `dense_warmup_steps` steps after enabling DSA, force dense mode to let the indexer stabilize.

```
┌─────────────────────────────────────────────────────────┐
│                  DSA Decision Logic                      │
│                                                         │
│  if not enabled:                                        │
│      → Dense (standard MLA)                             │
│                                                         │
│  elif kv_len < activation_threshold (4096):             │
│      → Dense (overhead not worth it)                    │
│                                                         │
│  elif training_step < dense_warmup_steps:               │
│      → Dense (indexer still calibrating)                │
│                                                         │
│  else:                                                  │
│      → Sparse (indexer selects top-k tokens)            │
│                                                         │
│  ALWAYS: if training and output_indexer_loss:            │
│      compute indexer entropy loss (even in dense mode!) │
└─────────────────────────────────────────────────────────┘
```

This decision tree is simple but has a subtle and powerful property: **the indexer loss is computed regardless of which mode is active**. This is the key to the "train dense, infer sparse" strategy.

### 2.4 The "train dense, infer sparse" strategy (multi-phase training)

NanoSeek follows DeepSeek's two-phase training methodology:

```
╔═══════════════════════════════════════════════════════════════════╗
║                    TRAINING TIMELINE                              ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Phase 1: Dense MLA Pre-training (80% of tokens = 17.6B)         ║
║  ─────────────────────────────────────────────────────            ║
║  • Context: 4,096 tokens                                         ║
║  • Attention: DENSE (full L² computation)                        ║
║  • DSA enabled: NO (sparse_config.enabled = False)               ║
║  • Indexer: TRAINS via auxiliary entropy loss                     ║
║  • YaRN: OFF                                                     ║
║  • LR: 3e-4 → 3e-5                                              ║
║                                                                   ║
║  What's happening: The main model learns strong representations.  ║
║  The indexer watches dense attention and learns to predict which  ║
║  tokens would have received high attention scores.                ║
║                                                                   ║
║                          ──── transition ────                     ║
║                                                                   ║
║  Phase 2: Sparse DSA Fine-tuning (20% of tokens = 4.4B)          ║
║  ────────────────────────────────────────────────────             ║
║  • Context: 8,192 tokens (2× Phase 1)                           ║
║  • Attention: SPARSE (indexer selects top-2048 of 8192)          ║
║  • DSA enabled: YES (sparse_config.enabled = True)               ║
║  • Indexer: active selection + continued training                 ║
║  • YaRN: ON (extend RoPE from 4K → 8K)                          ║
║  • LR: 1e-4 → 1e-5 (reduced for fine-tuning)                    ║
║                                                                   ║
║  What's happening: The model adapts to sparse attention.          ║
║  The indexer is now "live" — its selections directly affect       ║
║  which tokens participate in attention.                            ║
║                                                                   ║
╠═══════════════════════════════════════════════════════════════════╣
║  Inference: YaRN + DSA for efficient long context                 ║
║  • YaRN extends to 32K (8× training length)                      ║
║  • DSA selects top-2048 out of up to 32K tokens                  ║
║  • MLA compresses KV cache by 23×                                ║
║  • Result: 16× attention reduction + 23× cache reduction         ║
╚═══════════════════════════════════════════════════════════════════╝
```

**Why not just train sparse from the start?**

Three reasons, from most to least important:

1. **Gradient quality.** Dense attention provides gradients to *every* (query, key) pair. Sparse attention only provides gradients through the selected pairs. Early in training, the indexer doesn't know which tokens matter, so its selections would be near-random, giving the model a severely impoverished gradient signal.

2. **Indexer cold start.** The indexer needs to learn what "important" looks like. It learns by comparing its predictions to actual dense attention patterns. Without dense attention as a teacher signal, the indexer has no supervision.

3. **Training efficiency.** Phase 1 uses 4K context with dense attention, processing 4× more gradient updates per token-hour than 8K context with sparse attention. The shorter context is simply more FLOP-efficient for learning representations.

### 2.5 Sparse forward pass — complete walkthrough with ASCII diagrams

When DSA activates sparse mode, here is the complete forward pass. I'll trace a concrete example:

**Setup:** B=1, L=8192, hidden=2048, num_heads=16, topk=2048

```
Step 1: Extract compressed representations
═══════════════════════════════════════════

  hidden_states [1, 8192, 2048]
         │
    _get_compressed_representations()
         │
         ├──→ q_compressed  [1, 8192, 430]    (via wq_a + q_norm)
         ├──→ kv_compressed [1, 8192, 143]    (via wkv_a + kv_norm)
         └──→ k_pe          [1, 8192, 32]     (RoPE positional component)
```

```
Step 2: Apply RoPE to positional key component
═══════════════════════════════════════════════

  k_pe [1, 8192, 32]
    │
    ▼
  unsqueeze(2) → [1, 8192, 1, 32]
    │
    ▼
  apply_rotary_emb(k_pe, freqs_cis[:8192])
    │
    ▼
  full_k_pe [1, 8192, 1, 32]    ← Position-encoded, shared across all heads
```

```
Step 3: Lightning Indexer scores all (query, key) pairs
═══════════════════════════════════════════════════════

  q_compressed [1, 8192, 430]    kv_compressed [1, 8192, 143]
        │                                │
        ▼                                ▼
  ┌─────────────────────────────────────────────────────────┐
  │              Lightning Indexer                           │
  │                                                         │
  │  q_idx = q_proj(q_compressed)   [1, 8192, 4, 64]       │
  │  k_idx = k_proj(kv_compressed)  [1, 8192, 4, 64]       │
  │                                                         │
  │  scores = einsum('bqhd,bkhd->bhqk', q_idx, k_idx)      │
  │           [1, 4, 8192, 8192]                            │
  │                                                         │
  │  scores = ReLU(scores)                                  │
  │                                                         │
  │  index_scores = weighted_sum_over_heads(scores)         │
  │                 [1, 8192, 8192]                         │
  │                                                         │
  │  + causal_mask (future positions → -inf)                │
  │                                                         │
  └─────────────────────────────┬───────────────────────────┘
                                │
                                ▼

  NOTE: The indexer computes 8192 × 8192 = 67M scores,
  but in COMPRESSED space (64-dim heads, 4 heads).
  Cost: 4 × 8192 × 8192 × 64 = 17.2B MACs
  vs full attention: 16 × 8192 × 8192 × 96 = 103B MACs
  The indexer is ~6× cheaper than full attention.
```

```
Step 4: Select top-k tokens per query position
═══════════════════════════════════════════════

  index_scores [1, 8192, 8192]
        │
        ▼
  torch.topk(index_scores, k=2048, dim=-1, sorted=False)
        │
        ├──→ selected_values  [1, 8192, 2048]
        └──→ selected_indices [1, 8192, 2048]
                                    │
                                    ▼
              Each of the 8192 query positions now knows
              WHICH 2048 key positions to attend to.

              Example for query position 5000:
              selected_indices[0, 5000] = [0, 3, 17, 42, ..., 4998, 4999]
              (2048 indices from the range [0, 5000])
              ← only past/present positions (causal mask enforced)
```

```
Step 5: Full Q computation (expand from compressed)
════════════════════════════════════════════════════

  q_compressed [1, 8192, 430]
        │
        ▼
  wq_b(q_compressed)  ← Linear(430 → 16 × 96 = 1536)
        │
        ▼
  reshape [1, 8192, 16, 96]
        │
        ├──→ q_nope [1, 8192, 16, 64]   (non-positional component)
        └──→ q_pe   [1, 8192, 16, 32]   (positional component)
                │
                ▼
        apply_rotary_emb(q_pe, freqs_cis)
                │
                ▼
        q = cat([q_nope, q_pe], dim=-1)  [1, 8192, 16, 96]
        q = q.transpose(1,2)             [1, 16, 8192, 96]
```

```
Step 6: Full KV expansion + gather selected tokens
═══════════════════════════════════════════════════

  kv_compressed [1, 8192, 143]
        │
        ▼
  wkv_b(kv_compressed)  ← Linear(143 → 16 × (64+64) = 2048)
        │
        ▼
  reshape [1, 8192, 16, 128]
        │
        ├──→ k_nope_full [1, 8192, 16, 64]   (all 8192 positions)
        └──→ v_full      [1, 8192, 16, 64]   (all 8192 positions)

  NOW: gather only the selected 2048 positions for each query:

  k_nope_gathered = gather(k_nope_full, selected_indices)
                    [1, 8192, 2048, 16, 64]
                     B   Q     K    H   D

  k_pe_gathered   = gather(full_k_pe, selected_indices)
                    [1, 8192, 2048, 1, 32] → expand → [1, 8192, 2048, 16, 32]

  k_selected = cat([k_nope_gathered, k_pe_gathered], dim=-1)
               [1, 8192, 2048, 16, 96]

  v_selected = gather(v_full, selected_indices)
               [1, 8192, 2048, 16, 64]
```

```
Step 7: Sparse attention — only attend to selected tokens!
══════════════════════════════════════════════════════════

  q          [1, 16, 8192, 96]       ← full queries (all positions)
  k_selected [1, 16, 8192, 2048, 96] ← selected keys (per query position)
  v_selected [1, 16, 8192, 2048, 64] ← selected values (per query position)

  Attention scores:
  attn_scores = (q @ k_selected^T) * softmax_scale
              = [1, 16, 8192, 1, 96] @ [1, 16, 8192, 96, 2048]
              = [1, 16, 8192, 2048]

  + causal mask (enforce selected_positions <= query_position)

  attn_weights = softmax(attn_scores, dim=-1)
                 [1, 16, 8192, 2048]

  attn_output  = attn_weights @ v_selected
               = [1, 16, 8192, 2048] @ [1, 16, 8192, 2048, 64]
               = [1, 16, 8192, 64]
```

```
Step 8: Output projection
═════════════════════════

  attn_output [1, 16, 8192, 64]
        │
        ▼
  transpose(1,2) → [1, 8192, 16, 64]
        │
        ▼
  reshape → [1, 8192, 1024]    (16 × 64 = 1024)
        │
        ▼
  wo(attn_output) → [1, 8192, 2048]   ← Linear(1024 → 2048)
        │
        ▼
  output [1, 8192, 2048]   ← Same shape as input!
```

**Total sparse attention cost vs dense:**

| Operation | Dense | Sparse | Savings |
|-----------|-------|--------|---------|
| Score computation | 16 × 8192² × 96 = 103B MACs | 16 × 8192 × 2048 × 96 = 25.8B MACs | 4× |
| Indexer overhead | 0 | 4 × 8192² × 64 = 17.2B MACs | — |
| Score memory | 16 × 8192² × 4B = 4 GB | 16 × 8192 × 2048 × 4B = 1 GB | 4× |
| **Net** | **103B MACs** | **43B MACs** | **2.4×** |

At L = 32K the savings are much more dramatic (16× for scores alone), which is the intended operating regime.

### 2.6 How the indexer trains via entropy loss during dense phase

During Phase 1, DSA runs dense attention but still computes an auxiliary loss for the indexer. This is the clever bootstrapping mechanism:

```python
def _compute_indexer_loss(self, hidden_states, past_key_value):
    # Step 1: Get compressed representations
    q_compressed, kv_compressed, _ = self._get_compressed_representations(hidden_states)

    # Step 2: Build causal mask
    causal_mask = self._create_causal_mask(seq_len, kv_len, device, position_offset)

    # Step 3: Run indexer
    index_scores = self.indexer(q_compressed, full_kv_compressed, causal_mask)

    # Step 4: Compute entropy loss
    valid_mask = ~torch.isinf(causal_mask)
    index_scores_masked = index_scores.masked_fill(~valid_mask, -1e9)

    log_probs = F.log_softmax(index_scores_masked, dim=-1)
    probs = F.softmax(index_scores_masked, dim=-1)

    entropy = -(probs * log_probs).sum(dim=-1).mean()

    return entropy * self.sparse_config.indexer_loss_weight  # weight = 0.01
```

**What this entropy loss does:**

The entropy of the indexer's score distribution measures how "spread out" or "concentrated" the scores are.

- **High entropy** = uniform distribution → indexer thinks all tokens are equally important → bad (doesn't discriminate)
- **Low entropy** = peaked distribution → indexer concentrates on a few tokens → potentially too aggressive
- **Medium entropy** = the indexer has learned meaningful preferences → good

The loss pushes the indexer toward a distribution that *has structure* (not uniform) but *isn't collapsed* (not all mass on one token). The `indexer_loss_weight = 0.01` keeps this auxiliary objective small relative to the main language modeling loss.

**Critical detail:** The entropy is computed over the *softmax* of index scores, not the raw ReLU scores. This normalizes the scores into a probability distribution for entropy computation, even though the actual selection mechanism uses raw scores with `topk`.

**Training signal flow during Phase 1:**

```
┌────────────────────────────────────────────────────────────────┐
│  Phase 1: Dense Forward Pass                                   │
│                                                                │
│  hidden_states ──→ MLA (dense attention) ──→ output            │
│       │                                        │               │
│       │              main_loss = CE(output, labels)             │
│       │                                                        │
│       └──→ compressed reps ──→ Indexer ──→ entropy_loss         │
│                                    │                           │
│                                    │    indexer_loss = 0.01 ×  │
│                                    │    entropy(softmax(scores))│
│                                    │                           │
│                                    ▼                           │
│                          total_loss = main_loss                │
│                                    + mtp_loss × λ             │
│                                    + indexer_loss              │
│                                    + aux_moe_loss              │
│                                                                │
│  Gradients flow to:                                            │
│  • Main model (via main_loss + mtp_loss)                       │
│  • Indexer (via indexer_loss only — detached from main path)    │
│  • MoE router (via aux_moe_loss)                               │
└────────────────────────────────────────────────────────────────┘
```

**Why entropy and not KL-divergence with dense attention?**

The NanoSeek implementation uses entropy-based training. The config docstring mentions "KL-divergence alignment with main attention" as the original DeepSeek approach. The entropy-based approach is simpler and avoids the need to compute and store the full dense attention distribution as a target. In practice, maximizing the informativeness of the indexer's distribution (entropy regularization) achieves similar results to direct KL alignment, especially when the indexer has sufficient capacity.

### 2.7 The gather mechanism — how selected tokens are assembled

The `_gather_selected` method is the workhorse that assembles the sparse key/value tensors. It handles both 3D tensors (no head dimension, e.g., k_pe before expansion) and 4D tensors (with head dimension, e.g., k_nope after MLA expansion):

```python
def _gather_selected(self, tensor, indices, last_dim):
    batch_size, kv_len = tensor.shape[:2]
    seq_len, num_selected = indices.shape[1], indices.shape[2]

    if tensor.dim() == 3:
        # tensor: [B, kv_len, D], indices: [B, seq_len, k]
        idx_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, last_dim)
        tensor_expanded = tensor.unsqueeze(1).expand(-1, seq_len, -1, -1)
        return torch.gather(tensor_expanded, dim=2, index=idx_expanded)
        # Result: [B, seq_len, k, D]
    else:
        # tensor: [B, kv_len, num_heads, D], indices: [B, seq_len, k]
        num_heads = tensor.shape[2]
        idx_expanded = indices.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, -1, num_heads, last_dim
        )
        tensor_expanded = tensor.unsqueeze(1).expand(-1, seq_len, -1, -1, -1)
        return torch.gather(tensor_expanded, dim=2, index=idx_expanded)
        # Result: [B, seq_len, k, num_heads, D]
```

**Memory consideration:** The `tensor.unsqueeze(1).expand(-1, seq_len, -1, ...)` step creates a view (no copy) that logically replicates the KV tensor along the query dimension. The `torch.gather` then materializes only the selected elements. This is memory-efficient on GPU because `expand` doesn't allocate.

### 2.8 Causal masking in sparse attention

Sparse attention requires a different causal mask than dense attention. In dense attention, we use a triangular mask of shape `[L, L]`. In sparse attention, each query position has its own set of selected key positions, so we need a per-query mask:

```python
# In _sparse_forward:
q_positions = torch.arange(seq_len, device=q.device) + position_offset
q_positions = q_positions.view(1, 1, seq_len, 1)           # [1, 1, L, 1]
selected_positions = selected_indices.unsqueeze(1)           # [B, 1, L, k]

causal_mask_selected = torch.where(
    selected_positions > q_positions,                        # future token?
    torch.tensor(float('-inf'), device=q.device),            # mask it
    torch.tensor(0.0, device=q.device)                       # keep it
)
attn_scores = attn_scores + causal_mask_selected
```

This is necessary because `torch.topk` might select future positions if the indexer's causal mask didn't fully suppress them (edge case with very negative but not `-inf` scores from the ReLU output). The explicit causal mask in the attention computation is the safety net.

---

## 🔴 STAGE 3: GROUND TRUTH

### What DSA learns from

DSA's ground truth is implicit — there are no explicit "these tokens should be selected" labels. Instead, the indexer learns from two signals:

#### Signal 1: Entropy loss (auxiliary objective)

The entropy-based loss encourages the indexer to produce *structured* predictions — not uniform (all tokens equal) and not collapsed (all mass on one token). The "correct" behavior emerges indirectly: if the indexer assigns high scores to tokens that the main attention would have attended to, the model's main loss decreases. If the indexer misses important tokens, the main loss increases, and backpropagation adjusts the model to compensate (or in Phase 2, directly adjusts the indexer through the sparse attention path).

#### Signal 2: End-to-end gradients (Phase 2)

In Phase 2, the indexer's selections directly determine which tokens participate in attention. The main loss backpropagates through:

```
main_loss → output → wo → attn_output → attn_weights → attn_scores
                                                            │
                                            (depends on selected K, V)
                                                            │
                                            (selected by indexer's topk)
```

`torch.topk` is not differentiable with respect to the selection indices, but it IS differentiable with respect to the *values* of selected elements. This means:
- The indexer does NOT receive gradients through the selection operation itself
- But the KV values that were selected DO receive gradients
- The indexer still receives gradients through its separate entropy loss
- And the MLA weights adjust to work well with whatever the indexer selects

This creates a co-adaptation dynamic: the MLA learns to encode important information in tokens the indexer favors, and the indexer learns to favor tokens that help the MLA produce good outputs.

### Training data flow — what tokens teach the indexer

Consider a concrete example. Input: "The capital of France is Paris. The Eiffel Tower..."

During Phase 1 (dense), the full attention pattern might show:
- "Tower" strongly attends to "Eiffel" (bigram association)
- "Tower" attends to "Paris" (semantic context)
- "Tower" weakly attends to "The", "of", etc.
- "Tower" barely attends to tokens beyond 500 positions back

The indexer observes this through its entropy loss and learns that:
- Nearby tokens are usually important (high indexer score)
- Semantically related distant tokens are important (moderate score)
- Generic function words at distance are unimportant (zero after ReLU)

During Phase 2 (sparse), the indexer's top-2048 selection for "Tower" at position 5000 would include:
- Positions 4998–5000 (immediate context — always included due to recency)
- Position ~30 (where "Paris" appeared — semantic relevance)
- Various other content-bearing positions scattered through the context

### Labels and loss targets

The NanoSeek model computes total loss as:

```python
total_loss = main_loss + mtp_weight * mtp_loss + aux_moe_loss + indexer_loss
```

Where:
- `main_loss`: Cross-entropy between predicted next token and actual next token
- `mtp_loss`: Cross-entropy for multi-token prediction heads (weight 0.3 → 0.1)
- `aux_moe_loss`: Sequence-level expert load balancing (weight 0.0001)
- `indexer_loss`: Entropy regularization for the Lightning Indexer (weight 0.01)

The `labels` tensor is `[B, L]` containing token IDs, with `-100` for positions to ignore (standard PyTorch cross-entropy convention).

---

## 🔴 STAGE 4: LOSS FUNCTION — Indexer Entropy Loss + Main Loss

### 4.1 The complete loss landscape

NanoSeek's total loss during training is a weighted sum of four components:

```
L_total = L_main + λ_mtp × L_mtp + α × L_moe_aux + β × L_indexer

Where:
  L_main     = CE(shift_logits, shift_labels)         main next-token prediction
  L_mtp      = Σᵢ (0.8^i × CE(mtp_logits_i, labels)) multi-token prediction
  L_moe_aux  = 0.0001 × mean((load - target_load)²)   expert load balance
  L_indexer  = 0.01 × H(softmax(index_scores))         indexer entropy

  λ_mtp = 0.3 (first 60% of training) → 0.1 (remaining 40%)
  α = 0.0001 (constant)
  β = 0.01 (constant)
```

### 4.2 Indexer entropy loss — deep analysis

The indexer loss is the most novel component. Let's decompose it mathematically:

Given index_scores `s[q, k]` for query position q and key position k (after causal masking), compute:

```
p[q, k] = softmax(s[q, :])[k]                  # probability distribution over keys
H[q]    = -Σ_k p[q,k] × log(p[q,k])           # entropy for query q
L_indexer = β × mean_q(H[q])                   # average entropy × weight
```

**What happens at extremes:**

| Indexer state | Entropy | Loss contribution | Gradient effect |
|---------------|---------|-------------------|-----------------|
| Uniform scores (all equal) | H = log(L) ≈ 8.3 for L=4096 | High (0.083) | Push toward differentiation |
| Concentrated (1 token dominates) | H ≈ 0 | Low (≈0) | Little effect |
| Well-spread (top-k clearly above rest) | H ≈ 5-7 | Medium (0.05-0.07) | Stable equilibrium |

**The entropy gradient:**

```
∂H/∂s[q,k] = p[q,k] × (H[q] + log(p[q,k]))
```

This gradient:
- Increases scores of tokens with probability *below* average (log p < -H → positive gradient)
- Decreases scores of tokens with probability *above* average (log p > -H → negative gradient)
- Pushes toward a more uniform distribution IF scores are already concentrated
- Has NO effect on tokens already at zero (ReLU output), which is the desired behavior

**Why entropy works as a proxy for attention quality:**

A good indexer should:
1. Assign high scores to a *subset* of tokens (not all)
2. Not collapse all mass onto a single token
3. Have a "soft" ranking that can be refined through training

Entropy regularization achieves (1) by penalizing uniform distributions, (2) by being low when concentrated (so no incentive to further concentrate), and (3) by keeping the softmax temperature reasonable. The key insight is that the *main loss* handles the actual alignment — the entropy loss just keeps the indexer's distribution well-behaved enough to be useful.

### 4.3 Interaction between losses during training

The four losses interact through shared parameters:

```
┌─────────────────────────────────────────────────────────────┐
│                   Parameter Groups                           │
│                                                             │
│  Embedding weights ←── L_main, L_mtp                        │
│                                                             │
│  MLA weights       ←── L_main (dense path in Phase 1)       │
│                        L_main (sparse path in Phase 2)       │
│                                                             │
│  MoE experts       ←── L_main, L_mtp                        │
│  MoE router        ←── L_main, L_mtp, L_moe_aux             │
│                                                             │
│  Indexer weights   ←── L_indexer ONLY                        │
│  (q_proj, k_proj,     (no gradient from main loss in        │
│   head_weights)        Phase 1; indirect in Phase 2)         │
│                                                             │
│  MLA down-projections ←── L_main AND L_indexer              │
│  (wq_a, wkv_a)         (compressed reps are shared!)       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The critical shared parameters are `wq_a` and `wkv_a` — MLA's down-projection layers. Both the main attention path and the indexer consume their outputs. This means the indexer loss subtly influences how the compressed representations are formed, which in turn affects the main attention quality. The `indexer_loss_weight = 0.01` keeps this influence small enough not to destabilize main training, but present enough for the indexer to shape the representations.

### 4.4 Loss magnitude analysis

At initialization (random weights), approximate loss magnitudes:

```
L_main     ≈ log(65536) ≈ 11.1     (random prediction over 65K vocab)
L_mtp      ≈ 11.1                    (same, for MTP head)
L_moe_aux  ≈ 0.0001 × variance     ≈ 0.001
L_indexer  ≈ 0.01 × log(4096)      ≈ 0.083

L_total ≈ 11.1 + 0.3 × 11.1 + 0.001 + 0.083 ≈ 14.5
```

The indexer loss is about 0.6% of total loss at initialization. As training progresses and L_main drops to ~3-4, the indexer loss proportion increases to ~2-3% of total. This is by design — as the model becomes more capable, the indexer's relative influence grows, preparing it for Phase 2 where its selections become consequential.

---

## 🔴 STAGE 5: OUTPUTS

### 5.1 What DSA produces

The DSA module returns a triple:

```python
def forward(self, hidden_states, ...) -> Tuple[Tensor, Optional[Tuple], Dict]:
    # ...
    return output, present_kv, aux_data
```

1. **`output`** `[B, L, 2048]` — The attention output, same shape as input. This feeds into the residual connection and subsequent FFN/MoE layer.

2. **`present_kv`** `(kv_compressed, full_k_pe)` — KV cache for incremental decoding:
   - `kv_compressed` `[B, L, 143]` — Compressed KV representations
   - `full_k_pe` `[B, L, 1, 32]` — Position-encoded RoPE component
   - Total cache per layer per token: 143 + 32 = **175 values** (vs. 4,096 for standard MHA)

3. **`aux_data`** `Dict` — Auxiliary outputs:
   - `indexer_loss` (float Tensor): The entropy regularization loss, present when `output_indexer_loss=True` and the model is training

### 5.2 Output during incremental (cached) decoding

During autoregressive generation, DSA processes one new token at a time with the KV cache:

```
Step 1 (prefill): Process full prompt
  input: [1, L_prompt, 2048]
  output: [1, L_prompt, 2048], cache=(kv_compressed[1,L_prompt,143], k_pe[1,L_prompt,1,32])

Step 2+ (decode): Process one token at a time
  input: [1, 1, 2048]
  past_key_value: (kv_compressed[1, L_prev, 143], k_pe[1, L_prev, 1, 32])

  Decision: is L_prev + 1 >= activation_threshold?
    If yes → sparse mode (indexer selects top-k from L_prev + 1 positions)
    If no  → dense mode (attend to all L_prev + 1 positions)

  output: [1, 1, 2048]
  new_cache: (kv_compressed[1, L_prev+1, 143], k_pe[1, L_prev+1, 1, 32])
```

**Key insight for inference:** During decode, each new token's query only needs scores against the existing cache. With the indexer, we score the new query against all cached positions in compressed space (430 × 143 dimensional), select top-k, then expand only those k positions to compute attention. For a 32K context, this means scoring 32K candidates but only computing full attention with 2,048 of them.

### 5.3 Concrete numerical walkthrough — sparse token selection

Let's trace through a specific example to make the abstract concrete.

**Setup:**

```python
config = SparseAttentionConfig(
    enabled=True,
    topk_tokens=2048,
    activation_threshold=4096,
    indexer_num_heads=4,
    indexer_head_dim=64,
    indexer_loss_weight=0.01,
)
```

**Input:** A sequence of 8,192 tokens (Phase 2 training or inference).

**Step-by-step shapes:**

```
hidden_states:     [2, 8192, 2048]   ← batch=2, seq=8192, hidden=2048

After compression:
  q_compressed:    [2, 8192, 430]    ← 430 = q_lora_rank
  kv_compressed:   [2, 8192, 143]    ← 143 = kv_lora_rank
  k_pe:            [2, 8192, 32]     ← 32  = qk_rope_head_dim

Indexer intermediate:
  q_idx:           [2, 8192, 4, 64]  ← 4 heads × 64 dim
  k_idx:           [2, 8192, 4, 64]

  raw_scores:      [2, 4, 8192, 8192] ← 4 heads × (q_len × kv_len)
  after ReLU:      [2, 4, 8192, 8192] ← zeros where negative
  after weight+sum:[2, 8192, 8192]    ← weighted combination across heads
  after causal:    [2, 8192, 8192]    ← upper triangle = -inf

After topk selection (k=2048):
  selected_indices:[2, 8192, 2048]    ← which 2048 out of 8192 keys

Full Q expansion:
  q:               [2, 16, 8192, 96]  ← 16 heads × (64 nope + 32 rope)

Full KV expansion (all positions, then gather):
  k_nope_full:     [2, 8192, 16, 64]
  v_full:          [2, 8192, 16, 64]
  k_nope_gathered: [2, 8192, 2048, 16, 64]  ← only selected positions
  v_gathered:      [2, 8192, 2048, 16, 64]

Sparse attention:
  attn_scores:     [2, 16, 8192, 2048]  ← Q @ K_selected^T (not 8192!)
  attn_weights:    [2, 16, 8192, 2048]  ← softmax
  attn_output:     [2, 16, 8192, 64]    ← weighted sum of V_selected

Final:
  output:          [2, 8192, 2048]       ← after wo projection
```

**Memory comparison at this sequence length:**

| Component | Dense | Sparse | Ratio |
|-----------|-------|--------|-------|
| Attention score matrix | 2×16×8192²×4B = 8 GB | 2×16×8192×2048×4B = 2 GB | 4× |
| Gathered KV | 0 (direct) | 2×8192×2048×16×64×4B = 128 GB | overhead |
| Net peak memory | ~8 GB | ~2.5 GB | 3.2× savings |

Note: The gathered KV looks expensive in raw numbers, but it's computed via `expand` (view, no copy) + `gather` (selective copy), and only the 2048-selected positions are materialized per query.

### 5.4 How DSA output flows through the rest of the model

After DSA, the output re-enters the standard transformer residual stream:

```python
# In NanoSeekDecoderLayer.forward():
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)

# DSA or MLA attention
hidden_states, present_key_value, attn_aux = self.self_attn(
    hidden_states=hidden_states, ...
)

hidden_states = residual + hidden_states         # ← Residual connection

residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states, ffn_aux = self.ffn(hidden_states)  # ← MoE or dense FFN
hidden_states = residual + hidden_states         # ← Residual connection
```

The DSA output is functionally identical to MLA's output from the downstream perspective. The FFN/MoE layer doesn't know (or care) whether the attention was dense or sparse. This encapsulation is clean — you can toggle DSA on/off without affecting any other component.

### 5.5 DSA's effect on model quality

The indexer's selection quality directly affects what information each query token can access:

**Good selection** (trained indexer):
- Nearby tokens are always included (local context)
- Semantically relevant distant tokens are included (long-range dependencies)
- Uninformative tokens are excluded (padding, generic function words at distance)
- Result: Minimal quality degradation vs dense attention

**Poor selection** (untrained or random indexer):
- Random subset of tokens selected
- Important long-range dependencies missed
- Result: Significant quality degradation, especially for tasks requiring long context

This is why the two-phase training is essential: Phase 1 builds a good indexer before Phase 2 relies on it.

---

## Common Misconceptions

### Misconception 1: "DSA is just another sliding window attention"

**Wrong.** Sliding window attention attends to a fixed window of nearby tokens (e.g., positions [i-W, i]). DSA's Lightning Indexer selects tokens *globally* — a query at position 8,000 might select tokens at positions 3, 42, 1,500, and 7,999. The selection is *content-dependent*, not position-dependent. The indexer learns to detect semantic relevance, not just proximity.

NanoSeek does have a `sliding_window_size` config option (512 tokens), but this is a *safety net* that can be combined with DSA's top-k selection, not a replacement for it. In the current implementation, the main selection mechanism is pure top-k without explicit sliding window enforcement.

### Misconception 2: "You can't backpropagate through top-k"

**Partially true, but not a problem.** `torch.topk` is differentiable with respect to the *values* of selected elements, but not with respect to which elements are selected. The indexer doesn't receive gradients through the selection operation itself. But this is by design — the indexer learns through its own entropy loss, not through the main attention path.

The MLA weights DO receive gradients through the sparse attention computation — they just only get gradients from the selected tokens, not all tokens. This is sufficient because the indexer is selecting the *most important* tokens, which carry the most informative gradients.

### Misconception 3: "DSA needs special training data or curriculum"

**Wrong.** DSA trains on the same data as the base model. The only "curriculum" is the two-phase schedule: dense first, then sparse. The indexer learns from the natural attention patterns that emerge during pre-training — no special curation needed.

### Misconception 4: "The indexer computation is almost as expensive as full attention"

**Sometimes close, but asymptotically better.** At L = 4,096, the indexer's cost (4 heads × 64 dim × L²) is comparable to the savings from sparse attention (16 heads × 96 dim × L×k). This is why DSA only activates *above* the threshold. At L = 32K, the indexer's 4×64=256-dim computation over L² is dwarfed by the savings from reducing 16×96-dim attention from L² to L×k.

### Misconception 5: "DSA replaces MLA"

**Wrong.** DSA *wraps* MLA. Every DSA layer contains a complete MLA instance. DSA uses MLA's compression layers (`wq_a`, `wkv_a`, `wq_b`, `wkv_b`, `wo`) and its RoPE implementation. The indexer is an *addition* to MLA, not a replacement. When DSA is disabled, the layer behaves exactly as standard MLA.

---

## Production Gotchas

### 1. Memory spikes during gather operations

The `_gather_selected` method expands tensors along the query dimension before gathering. For large batch sizes and long sequences, this can create temporary memory spikes:

```python
# This expand creates a view (cheap), but gather materializes the result
tensor_expanded = tensor.unsqueeze(1).expand(-1, seq_len, -1, -1, -1)
result = torch.gather(tensor_expanded, dim=2, index=idx_expanded)
```

**Mitigation:** Process queries in chunks when memory is tight. The `seq_len` dimension can be split without affecting correctness (each query's selection is independent).

### 2. Indexer score underflow with very long sequences

At very long sequences (>32K), the ReLU activation can cause most scores to be exactly zero, making `topk` select from a mostly-zero pool. This doesn't break correctness (selected tokens still get proper causal masking and softmax attention), but it may degrade quality because the selection becomes less discriminative.

**Mitigation:** Consider temperature scaling on the indexer scores, or increasing `indexer_head_dim` for longer-context deployments.

### 3. KV cache growth is NOT reduced by DSA

DSA reduces *attention computation*, not KV cache size. The cache still stores all `kv_compressed` and `k_pe` values for every position. MLA already handles cache compression (23×). DSA handles compute reduction. They are complementary, not overlapping.

### 4. Causality enforcement is double-checked

The implementation enforces causality in both the indexer (via `causal_mask` parameter) and the attention computation (via `causal_mask_selected`). This redundancy is intentional — a bug in either mask alone could lead to information leakage from future tokens, which is catastrophic for autoregressive generation.

### 5. The training_step buffer must be incremented externally

```python
# In model.py:
def increment_dsa_training_steps(self):
    if self.config.sparse.enabled:
        for layer in self.layers:
            if layer.use_sparse_attention:
                layer.self_attn.increment_training_step()
```

The training loop must call `model.increment_dsa_training_steps()` each step. Forgetting this means the warmup counter never advances and sparse mode never activates (if warmup > 0). In the default config, `dense_warmup_steps=0`, so this only matters if you add a warmup period.

### 6. Phase transition requires config modification

Switching from Phase 1 to Phase 2 requires updating the config:

```python
from model.config import apply_phase_config, get_training_phases

phases = get_training_phases(config)
config = apply_phase_config(config, phases[1])  # Switch to Phase 2
```

This updates `sparse.enabled`, `sequence_length`, `global_batch_size`, learning rate, and YaRN settings. The model itself doesn't need to be reconstructed — the DSA layers read from `self.sparse_config` at each forward pass.

---

## 2026 Best Practices

### 1. Always benchmark the break-even point

Before deploying DSA, measure where the indexer overhead crosses the attention savings for your specific hardware. The theoretical crossover is at `activation_threshold`, but GPU memory bandwidth and kernel efficiency shift the real crossover. Profile both modes at your target sequence length.

### 2. Monitor indexer entropy during training

Plot `indexer_loss / indexer_loss_weight` to get raw entropy. If it's near `log(L)` after many steps, the indexer isn't learning. If it's near 0, it's collapsing. Healthy values are typically in the range [3, 7] for L ≈ 4K–8K.

### 3. Validate sparse vs dense equivalence on held-out data

After Phase 2, run inference on a validation set with both sparse and dense attention. The perplexity gap should be < 0.5% for a well-trained indexer. If it's > 2%, the indexer needs more Phase 2 training or a higher `topk_tokens`.

### 4. Consider topk_tokens as a quality-speed tradeoff knob

The default `topk_tokens=2048` was chosen for DeepSeek V3's scale. For NanoSeek at 8K context, this selects 25% of tokens. You can:
- Increase to 3072 (37.5%) for higher quality at modest speed cost
- Decrease to 1024 (12.5%) for faster inference with some quality loss
- Use adaptive k based on sequence length: `k = min(2048, L // 2)`

### 5. Profile indexer head count for your use case

4 indexer heads with 64-dim each is the default. For code (high locality, structured syntax), 2 heads might suffice. For long-document QA (complex long-range dependencies), 8 heads might help. The head count trades off indexer accuracy vs indexer compute cost.

### 6. Integration testing is critical

The NanoSeek test suite (`tests/test_dsa.py`) validates:
- Indexer output shapes and non-negativity (ReLU)
- Dense/sparse mode switching at threshold
- Warmup period enforcement
- Indexer loss computation and gradient flow
- Incremental decoding with cache growth
- Compressed representation shapes and numerics

Run the full DSA test suite whenever modifying any DSA-adjacent code:

```bash
pytest tests/test_dsa.py -v
```

### 7. The "train dense, infer sparse" philosophy extends beyond DSA

The principle — train with full information, deploy with efficiency approximations — applies broadly:
- Train with dense attention, deploy with sparse (DSA)
- Train with full precision, deploy with quantization (FP8)
- Train with short context, deploy with extended context (YaRN)
- Train with all experts, deploy with fewer active experts

NanoSeek's multi-phase training is a concrete instantiation of this philosophy. The 80/20 split (80% dense, 20% sparse) is empirically derived from DeepSeek's production experience.

---

## Appendix A: NanoSeek DSA Configuration Reference

```python
@dataclass
class SparseAttentionConfig:
    enabled: bool = False                  # Master switch
    topk_tokens: int = 2048               # Budget per query
    activation_threshold: int = 4096       # Min seq length for sparse
    indexer_num_heads: int = 4             # Indexer multi-head count
    indexer_head_dim: int = 64             # Dimension per indexer head
    dense_warmup_steps: int = 0            # Steps of forced dense after enable
    indexer_loss_weight: float = 0.01      # Entropy loss scale
    use_sliding_window: bool = True        # Sliding window safety net
    sliding_window_size: int = 512         # Sliding window width
```

## Appendix B: Key Code Pointers

| Component | File | Line(s) | Description |
|-----------|------|---------|-------------|
| `LightningIndexer` | `model/model.py` | 1056–1102 | Indexer architecture |
| `DSASparseAttention` | `model/model.py` | 1109–1372 | Full DSA wrapper |
| `_sparse_forward` | `model/model.py` | 1255–1348 | Sparse attention path |
| `_dense_forward` | `model/model.py` | 1213–1225 | Dense attention fallback |
| `_compute_indexer_loss` | `model/model.py` | 1227–1253 | Entropy loss computation |
| `_gather_selected` | `model/model.py` | 1356–1368 | Token gather helper |
| `SparseAttentionConfig` | `model/config.py` | 398–461 | Configuration dataclass |
| `TrainingPhaseConfig` | `model/config.py` | 45–78 | Multi-phase training |
| `PHASE2_SPARSE` | `model/config.py` | 96–107 | Phase 2 defaults |
| DSA tests | `tests/test_dsa.py` | all | Validation suite |

## Appendix C: Full Sparse Attention Data Flow (One Diagram)

```
                        hidden_states [B, L, 2048]
                                │
                ┌───────────────┼───────────────┐
                │               │               │
                ▼               ▼               ▼
            wq_a            wkv_a           (decision)
          [2048→430]     [2048→175]        enabled? threshold?
                │               │               │
                ▼               │               ▼
          q_norm(RMS)     ┌────┴────┐     use_sparse?
                │         │         │         │
                ▼         ▼         ▼    ┌────┴────┐
         q_compressed  kv_comp   k_pe    │         │
          [B,L,430]   [B,L,143] [B,L,32] │         │
                │         │       │      YES       NO
                │    kv_norm(RMS) │       │         │
                │         │       │       │    MLA dense
                │         │   RoPE(k_pe)  │    (standard)
                │         │       │       │         │
                ▼         ▼       ▼       │         ▼
           ┌─────────────────────────┐    │   output, cache
           │   Lightning Indexer     │    │
           │                         │    │
           │ q_proj → [B,L,4,64]     │    │
           │ k_proj → [B,L,4,64]     │    │
           │ einsum → [B,4,L,L]      │    │
           │ ReLU                    │    │
           │ weighted sum → [B,L,L]  │    │
           │ + causal mask           │    │
           └────────────┬────────────┘    │
                        │                 │
                        ▼                 │
                 topk(scores, k=2048)     │
                        │                 │
                 selected_indices         │
                 [B, L, 2048]             │
                        │                 │
           ┌────────────┤                 │
           │            │                 │
           ▼            ▼                 │
     wq_b(q_comp)  wkv_b(kv_comp)        │
     [430→1536]    [143→2048]             │
           │            │                 │
           ▼            ▼                 │
     split+RoPE   split k_nope, v         │
           │            │                 │
           ▼            ▼                 │
     Q [B,16,L,96]  gather(selected)      │
                        │                 │
                 K_sel [B,16,L,k,96]      │
                 V_sel [B,16,L,k,64]      │
                        │                 │
                        ▼                 │
                  Q @ K_sel^T             │
                  [B,16,L,k]              │
                  + causal mask           │
                  softmax                 │
                  @ V_sel                 │
                  [B,16,L,64]             │
                        │                 │
                        ▼                 │
                     wo proj              │
                  [B,L,2048]              │
                        │                 │
                        ▼                 │
                     output ◄─────────────┘
                  [B, L, 2048]
```

---

*This document reflects the NanoSeek implementation as of February 2026. For the latest code, see `model/model.py` sections 8–9 and `model/config.py` `SparseAttentionConfig`.*
