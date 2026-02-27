# Multi-Head Latent Attention (MLA) — First Principles Deep Dive

> **Document status**: Production reference for NanoSeek (DeepSeek V3.2 at nano scale)
> **Source of truth**: `model/model.py` — `MultiHeadLatentAttention` class
> **Audience**: Senior ML researchers who already understand standard multi-head attention

---

## Quick Context

### The Problem: KV Cache Explosion

Standard multi-head attention (MHA) stores **separate K and V vectors for every head, for every token, for every layer**. At inference time, this KV cache dominates memory:

```
Standard MHA KV cache per token per layer:
  = 2 (K + V) × num_heads × head_dim
  = 2 × 16 × 128
  = 4,096 values

At 32K context, 16 layers, bf16:
  = 4,096 × 32,768 × 16 × 2 bytes
  = 4.29 GB   ← just the cache
```

This is the fundamental bottleneck for long-context inference. Doubling context length doubles KV cache. Doubling model width quadruples it. Every production LLM deployment team in 2026 hits this wall.

### MLA's Solution: Compress, Then Reconstruct

Multi-Head Latent Attention compresses the KV representation into a **low-rank latent vector** (143 dims instead of 4,096), stores only that at inference, and reconstructs the full multi-head K and V on the fly during attention computation.

```
MLA KV cache per token per layer:
  = kv_lora_rank + qk_rope_head_dim
  = 143 + 32
  = 175 values

Compression ratio:  4,096 / 175 ≈ 23.4×
```

### 2026 Status

MLA is the **de facto standard** for production-scale inference-efficient transformers. DeepSeek V2/V3/V3.2 proved it at frontier scale. The key insight — that you can decouple positional information from the compressed latent and still get excellent quality — has been validated at scales from 1B to 671B parameters. Every serious new architecture in 2026 either uses MLA or a close variant (e.g., GQA+quantized KV is the poor man's alternative, but MLA is strictly better when you can afford the up-projection compute).

---

## 🔴 STAGE 1: INPUTS

### What Enters the MLA Module

The MLA module receives hidden states from the preceding layer norm in each decoder layer. These are dense floating-point tensors representing the contextual embedding of each token at the current layer depth.

```
Input: hidden_states
  Shape:  [B, L, 2048]
  Dtype:  bf16 (bfloat16) during training
  Range:  Roughly N(0, 1) after RMSNorm, but can drift

Where:
  B = batch size       (e.g., 2)
  L = sequence length   (e.g., 512)
  2048 = hidden_size    (model width)
```

### Concrete Example

For a training batch processing 2 sequences of 512 tokens each:

```
hidden_states.shape = [2, 512, 2048]
hidden_states.dtype = torch.bfloat16
hidden_states.numel() = 2 × 512 × 2048 = 2,097,152 values
hidden_states.nbytes = 2,097,152 × 2 = 4,194,304 bytes ≈ 4 MB
```

Each of the 2,048 dimensions represents a learned feature direction. By the time we reach MLA in a decoder layer, these hidden states have already been:
1. Embedded from token IDs via `nn.Embedding(65536, 2048)`
2. Processed through all preceding decoder layers (each: LayerNorm → MLA → Residual → LayerNorm → FFN/MoE → Residual)
3. Normalized by the current layer's `input_layernorm` (RMSNorm)

### Preprocessing: None Beyond LayerNorm

The MLA module receives **already-normalized** hidden states. The `NanoSeekDecoderLayer.forward()` applies `self.input_layernorm(hidden_states)` before calling `self.self_attn(...)`. This is the Pre-LN Transformer pattern:

```python
# In NanoSeekDecoderLayer.forward():
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)   # RMSNorm
hidden_states, present_key_value = self.self_attn(     # MLA receives normalized input
    hidden_states=hidden_states, ...
)
hidden_states = residual + hidden_states               # Residual connection
```

### Optional Inputs

The MLA module also accepts:
- `attention_mask`: Causal mask `[1, 1, L, L+past]`, float tensor with 0 and -inf
- `position_ids`: `[B, L]` — explicit position indices (used for KV cache offset)
- `past_key_value`: Tuple of `(kv_compressed [B, past_L, 143], k_pe [B, past_L, 1, 32])` from previous decoding steps
- `use_cache`: Boolean — whether to return the KV cache for incremental decoding

### 💡 Feynman Explainer: Inputs

*Think of each token's hidden state as a 2,048-dimensional "description card" that summarizes everything the model knows about that token so far. MLA's job is to let every token look at every other token's description card and selectively copy relevant information — but it does this through a compressed "summary" rather than the full card.*

---

## 🔴 STAGE 2: MODEL ARCHITECTURE

This is the core section. We will dissect every operation, every matrix, every dimension.

### High-Level: Why MLA Over MHA/GQA/MQA

| Method | KV Cache per Token | Quality | Compute Overhead |
|--------|-------------------|---------|-----------------|
| **MHA** | `2 × H × d` = 4,096 | Baseline | Baseline |
| **MQA** | `2 × d` = 256 | Degraded (shared heads) | Lower |
| **GQA-4** | `2 × 4 × d` = 1,024 | Slight degradation | Lower |
| **MLA** | `c_kv + d_rope` = 175 | **Same or better** | Higher (up-projection) |

Where H=16 heads, d=128 head_dim, c_kv=143 kv_lora_rank, d_rope=32.

The fundamental insight: MQA and GQA reduce cache by **sharing heads** (fewer distinct K/V representations). MLA reduces cache by **compressing all heads into a shared latent** and reconstructing per-head representations on the fly. This is strictly more powerful because:
1. The compression is **learned** — the model discovers what information to keep
2. Each head still gets its **own** K and V after reconstruction
3. The compression bottleneck acts as a useful **information bottleneck regularizer**

### The Two Paths: Query and Key-Value

MLA has two parallel projection paths. Here is the complete data flow:

```
                         hidden_states [B, L, 2048]
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
              QUERY PATH                       KV PATH
                    │                               │
            ┌───────┴───────┐               ┌───────┴───────┐
            │   wq_a        │               │   wkv_a       │
            │ [2048 → 430]  │               │ [2048 → 175]  │
            └───────┬───────┘               └───────┬───────┘
                    │                               │
            [B, L, 430]                     [B, L, 175]
                    │                               │
            ┌───────┴───────┐               ┌───────┴───────────┐
            │   q_norm      │               │     split         │
            │ RMSNorm(430)  │               │  [143]    [32]    │
            └───────┬───────┘               └────┬────────┬─────┘
                    │                            │        │
            [B, L, 430]                  kv_compressed   k_pe
                    │                    [B, L, 143]  [B, L, 32]
            ┌───────┴───────┐                │        │
            │   wq_b        │         ┌──────┴──────┐ │
            │ [430 → 1536]  │         │  kv_norm    │ │
            └───────┬───────┘         │ RMSNorm(143)│ │
                    │                 └──────┬──────┘ │
            [B, L, 1536]                     │        │
                    │                 [B, L, 143]     │
            ┌───────┴───────┐                │        │
            │   reshape     │         ┌──────┴──────┐ │
            │ → [B,L,16,96] │         │   wkv_b     │ │
            └───────┬───────┘         │ [143→2048]  │ │
                    │                 └──────┬──────┘ │
            ┌───────┴───────┐                │        │
            │    split      │         [B, L, 2048]    │
            │ [64]    [32]  │                │        │
            └──┬────────┬───┘         ┌──────┴──────┐ │
               │        │            │   reshape    │ │
            q_nope    q_pe           │→[B,L,16,128] │ │
          [B,L,16,64] [B,L,16,32]   └──────┬──────┘ │
               │        │                  │        │
               │   ┌────┴────┐      ┌──────┴──────┐ │
               │   │  RoPE   │      │   split     │ │
               │   │ apply   │      │ [64]  [64]  │ │
               │   └────┬────┘      └──┬──────┬───┘ │
               │        │             │      │     │
               │   q_pe_rot       k_nope     v     │
               │  [B,L,16,32]   [B,L,16,64] [B,L,16,64]
               │        │             │      │     │
               │        │             │      │  ┌──┴────┐
               │        │             │      │  │ RoPE  │
               │        │             │      │  │ apply │
               │        │             │      │  └──┬────┘
               │        │             │      │     │
               │        │             │      │  k_pe_rot
               │        │             │      │  [B,L,1,32]
               │        │             │      │     │
               │        │             │      │  expand to
               │        │             │      │  [B,L,16,32]
               │        │             │      │     │
            ┌──┴────────┴──┐       ┌──┴──────┴─────┘
            │   concat     │       │   concat
            │ [64]+[32]=96 │       │ [64]+[32]=96
            └──────┬───────┘       └──────┬───────┘
                   │                      │
              Q [B,L,16,96]          K [B,L,16,96]       V [B,L,16,64]
                   │                      │                    │
                   └──────────┬───────────┘                    │
                              │                                │
                    ┌─────────┴─────────┐                      │
                    │  Scaled Dot-Product │                     │
                    │  Attention          │◄────────────────────┘
                    │  Q K^T / scale      │
                    │  + causal mask      │
                    │  softmax (fp32)     │
                    │  × V                │
                    └─────────┬───────────┘
                              │
                    [B, 16, L, 64]  (attn_output per head)
                              │
                    ┌─────────┴─────────┐
                    │  reshape + wo      │
                    │  [16×64=1024→2048] │
                    └─────────┬─────────┘
                              │
                    output [B, L, 2048]
```

### Component-by-Component Deep Dive

#### 2.1 Query Down-Projection: `wq_a`

```python
self.wq_a = nn.Linear(hidden_size, q_lora_rank, bias=False)
# Shape: [2048, 430] — 880,640 parameters
```

This compresses the 2,048-dim hidden state into a 430-dim query latent. The ratio `q_lora_rank / hidden_size = 430 / 2048 ≈ 0.21` is preserved from DeepSeek V3 (where it's `1536 / 7168 ≈ 0.21`).

**Math:**
```
q_compressed = hidden_states × W_q_a^T

hidden_states:  [B, L, 2048]
W_q_a:          [430, 2048]
q_compressed:   [B, L, 430]
```

**Why 0.21×?** This is an empirically-determined sweet spot. Too small (e.g., 0.05×) and you lose query expressiveness — the model can't distinguish fine-grained attention patterns. Too large (e.g., 0.5×) and you get diminishing returns while wasting compute. DeepSeek found 0.21× through ablation studies and NanoSeek preserves this ratio exactly.

#### 2.2 Query Normalization: `q_norm`

```python
self.q_norm = RMSNorm(q_lora_rank)  # RMSNorm(430)
```

Applies Root Mean Square normalization to the compressed query:

```
q_normalized = q_compressed / RMS(q_compressed) × γ

where RMS(x) = sqrt(mean(x²) + ε),  ε = 1e-6
γ is a learned scale parameter [430]
```

**Why normalize here?** The down-projection can produce values with wildly varying magnitudes across the latent dimensions. Normalizing before the up-projection ensures stable gradient flow and prevents certain latent dimensions from dominating. This is a critical stability trick — removing this norm causes training divergence in practice.

#### 2.3 Query Up-Projection: `wq_b`

```python
self.wq_b = nn.Linear(q_lora_rank, num_heads * qk_head_dim, bias=False)
# Shape: [430, 16 × 96] = [430, 1536] — 660,480 parameters
```

Expands from the compressed latent back to per-head query vectors:

```
q = q_normalized × W_q_b^T

q_normalized:  [B, L, 430]
W_q_b:         [1536, 430]
q:             [B, L, 1536]

Reshaped to:   [B, L, 16, 96]    (16 heads, 96 = qk_head_dim)
```

Note: `qk_head_dim = qk_nope_head_dim + qk_rope_head_dim = 64 + 32 = 96`

This is the "bottleneck" architecture: 2048 → 430 → 1536. The 430-dim bottleneck forces the model to learn a compressed representation of what each query "wants to attend to."

#### 2.4 Query Split: nope + rope

```python
q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
# q_nope: [B, L, 16, 64]  — content-based query component
# q_pe:   [B, L, 16, 32]  — position-based query component
```

Each head's 96-dim query vector is split into:
- **nope** (No Position Encoding): 64 dims for content-based matching
- **pe** (Position Encoding): 32 dims that will receive RoPE

**This split is the architectural crux of MLA.** Standard transformers apply RoPE to the entire Q and K. MLA only applies RoPE to a subset (32 of 96 dims), keeping the majority (64 dims) position-free. This decoupling enables the KV compression trick (explained below).

#### 2.5 KV Down-Projection: `wkv_a`

```python
self.wkv_a = nn.Linear(hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False)
# Shape: [2048, 143 + 32] = [2048, 175] — 358,400 parameters
```

This single projection produces **both** the compressed KV latent and the RoPE key component:

```
kv = hidden_states × W_kv_a^T

hidden_states:  [B, L, 2048]
W_kv_a:         [175, 2048]
kv:             [B, L, 175]
```

Then immediately split:
```python
kv_compressed, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
# kv_compressed: [B, L, 143]  — the latent (THIS IS WHAT GETS CACHED)
# k_pe:          [B, L, 32]   — RoPE component (ALSO CACHED, shared across heads)
```

**Why fused?** Combining the KV compression and RoPE key extraction into a single linear projection is more parameter-efficient than two separate projections. It also ensures the RoPE component is "aware of" the compressed content during training.

#### 2.6 KV Normalization: `kv_norm`

```python
self.kv_norm = RMSNorm(kv_lora_rank)  # RMSNorm(143)
```

Applied **only** to the compressed KV part (not the RoPE component):

```python
kv_compressed = self.kv_norm(kv_compressed)
# Shape stays [B, L, 143], but values are normalized
```

The `k_pe` component is not normalized here because it will receive RoPE rotations, which expect unnormalized inputs. Normalizing before RoPE would distort the learned frequency patterns.

#### 2.7 KV Up-Projection: `wkv_b`

```python
self.wkv_b = nn.Linear(kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=False)
# Shape: [143, 16 × (64 + 64)] = [143, 2048] — 292,864 parameters
```

Reconstructs full per-head K (nope part) and V from the compressed latent:

```
kv_expanded = kv_compressed × W_kv_b^T

kv_compressed:  [B, L, 143]
W_kv_b:         [2048, 143]
kv_expanded:    [B, L, 2048]

Reshaped to:    [B, L, 16, 128]   (16 heads, 128 = 64 + 64)
```

Then split:
```python
k_nope, v = torch.split(kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
# k_nope: [B, L, 16, 64]  — per-head content key
# v:      [B, L, 16, 64]  — per-head value
```

**The KV bottleneck is 143 dims serving 16 heads.** That's ~8.9 dims per head of shared information. This is aggressive compression — the model must learn to encode all cross-head KV information into just 143 dimensions. The fact that this works (and works well) is the empirical surprise of MLA.

#### 2.8 RoPE Application

RoPE (Rotary Position Embedding) encodes absolute position as rotation in complex space, giving the model relative position awareness through the dot product.

```python
# For queries: RoPE applied to q_pe component
q_pe = apply_rotary_emb(q_pe, q_freqs, interleaved=True)
# q_pe: [B, L, 16, 32] → [B, L, 16, 32]  (same shape, rotated)

# For keys: RoPE applied to k_pe component
k_pe = apply_rotary_emb(k_pe_current, current_freqs, interleaved=True)
# k_pe: [B, L, 1, 32] → [B, L, 1, 32]  (same shape, rotated)
```

**Critical design decision: k_pe is [B, L, 1, 32] — shared across all 16 heads.**

In standard MHA, RoPE is applied per-head. In MLA, the positional key component is shared across all heads. This is possible because:
1. Position information is the same for all heads at a given token
2. The content-based component `k_nope` is already per-head (reconstructed from the latent)
3. Sharing saves `(16 - 1) × 32 = 480` values per token in the KV cache

The shared `k_pe` is broadcast to all heads during attention:
```python
k_pe_expanded = k_pe.expand(-1, -1, self.num_heads, -1)
# [B, L, 1, 32] → [B, L, 16, 32]
```

#### 2.9 Assembly and Attention

Final Q and K are assembled by concatenating the content and positional components:

```python
q = torch.cat([q_nope, q_pe], dim=-1)   # [B, L, 16, 64+32] = [B, L, 16, 96]
k = torch.cat([k_nope, k_pe_expanded], dim=-1)  # [B, L, 16, 64+32] = [B, L, 16, 96]
```

Transpose for attention computation:
```python
q = q.transpose(1, 2)  # [B, 16, L, 96]
k = k.transpose(1, 2)  # [B, 16, L, 96]
v = v.transpose(1, 2)  # [B, 16, L, 64]
```

Scaled dot-product attention:
```python
attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.softmax_scale
# q:           [B, 16, L, 96]
# k^T:         [B, 16, 96, L]
# attn_weights: [B, 16, L, L]

# Scale factor
softmax_scale = mscale / sqrt(qk_head_dim) = 1.0 / sqrt(96) ≈ 0.10206
```

Causal masking and softmax:
```python
attn_weights = attn_weights + attention_mask  # Add -inf for future positions
attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
# ↑ CRITICAL: softmax computed in fp32 even when rest is bf16
```

Value aggregation:
```python
attn_output = torch.matmul(attn_weights, v)
# attn_weights: [B, 16, L, L]
# v:            [B, 16, L, 64]
# attn_output:  [B, 16, L, 64]
```

#### 2.10 Output Projection: `wo`

```python
self.wo = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)
# Shape: [16 × 64, 2048] = [1024, 2048] — 2,097,152 parameters
```

```python
attn_output = attn_output.transpose(1, 2).contiguous()  # [B, L, 16, 64]
attn_output = attn_output.view(batch_size, seq_len, -1)  # [B, L, 1024]
output = self.wo(attn_output)                              # [B, L, 2048]
```

Note: the output dimension is `num_heads × v_head_dim = 16 × 64 = 1024`, **not** `num_heads × head_dim = 16 × 128 = 2048`. This is because MLA uses a smaller `v_head_dim` (64) than the standard `head_dim` (128). The output projection maps from this 1024-dim concatenation back to the 2048-dim hidden space.

### The KEY INSIGHT: What Gets Cached

At inference time, the KV cache stores **only the compressed representation**:

```
┌──────────────────────────────────────────────────┐
│              KV Cache Per Token                    │
│                                                    │
│  kv_compressed: [143 values]  (the latent)        │
│  k_pe:          [32 values]   (RoPE component)    │
│                                                    │
│  Total: 175 values per token per layer             │
│                                                    │
│  vs Standard MHA:                                  │
│  K: [16 heads × 128 dim] = 2,048 values           │
│  V: [16 heads × 128 dim] = 2,048 values           │
│  Total: 4,096 values per token per layer           │
│                                                    │
│  Compression: 4,096 / 175 = 23.4×                 │
└──────────────────────────────────────────────────┘
```

During each decoding step, the model:
1. Retrieves `kv_compressed` and `k_pe` from cache
2. Applies `wkv_b` to reconstruct full `k_nope` and `v` for ALL cached tokens
3. Uses the reconstructed K and V for attention computation

This trades **memory** (23× less cache) for **compute** (extra matrix multiply to reconstruct). At long context lengths, this is overwhelmingly worth it because:
- Memory bandwidth is the bottleneck in autoregressive decoding
- The reconstruction matmul (`wkv_b`: [143 → 2048]) is small relative to the attention computation itself
- Fitting more context in the same memory budget is a strict capability improvement

### Full Math Summary

Let `h ∈ ℝ^{B×L×d}` be the input hidden states where `d = 2048`.

**Query path:**
```
c_q = h · W_q_a^T                         ∈ ℝ^{B×L×r_q}        r_q = 430
ĉ_q = RMSNorm(c_q)                        ∈ ℝ^{B×L×r_q}
q_full = ĉ_q · W_q_b^T                    ∈ ℝ^{B×L×(H·d_qk)}  H=16, d_qk=96
[q_nope, q_pe] = split(q_full, [64, 32])  per head
q_pe = RoPE(q_pe)
q = [q_nope ∥ q_pe]                        ∈ ℝ^{B×L×H×d_qk}
```

**KV path:**
```
[c_kv, k_pe_raw] = (h · W_kv_a^T) split at [143, 32]
ĉ_kv = RMSNorm(c_kv)                       ∈ ℝ^{B×L×r_kv}     r_kv = 143
k_pe = RoPE(k_pe_raw)                       ∈ ℝ^{B×L×1×d_rope} d_rope = 32
kv = ĉ_kv · W_kv_b^T                       ∈ ℝ^{B×L×H×(d_nope+d_v)}
[k_nope, v] = split(kv, [64, 64])          per head
k = [k_nope ∥ expand(k_pe, H)]             ∈ ℝ^{B×L×H×d_qk}
```

**Attention:**
```
A = softmax( (q · k^T) / (mscale / √d_qk) + M )   ∈ ℝ^{B×H×L×L}
o = A · v                                            ∈ ℝ^{B×H×L×d_v}
output = reshape(o) · W_o^T                          ∈ ℝ^{B×L×d}
```

**Cache (inference only):**
```
cache = (ĉ_kv ∈ ℝ^{B×L×143}, k_pe ∈ ℝ^{B×L×1×32})
```

### Design Decisions: Why These Choices?

#### Why Low-Rank Compression?

The key observation: in standard MHA, most of the information in K and V is **redundant across heads**. Heads tend to learn correlated representations (especially deeper in the network). By forcing all heads to reconstruct from a shared 143-dim latent, MLA exploits this redundancy. The model learns to store only the "essential information" in the latent, and each head's up-projection learns to extract its own view.

This is formally analogous to a **shared-encoder, per-head-decoder** architecture for KV representations.

#### Why Decouple RoPE?

RoPE modifies the Q and K vectors in a position-dependent way. If you applied RoPE to the full K vector and then tried to compress it, the compressed representation would be **position-specific** — you'd need to store a different compressed vector for each position, defeating the purpose.

By decoupling RoPE into a separate `k_pe` component:
1. The compressed `kv_compressed` is **position-independent** — it only encodes content
2. The `k_pe` component carries position information separately
3. Both are cached, but the position part is tiny (32 dims) and shared across heads

This is the single most important design decision in MLA. Without it, the compression doesn't work.

#### Why Shared RoPE Across Heads?

Position is a property of the **token**, not the **head**. Token 42 is at position 42 regardless of which attention head is looking at it. So sharing a single 32-dim RoPE component across all 16 heads is information-theoretically sound and saves `15 × 32 = 480` values per token in the cache.

The per-head differentiation comes from `k_nope` (reconstructed from the latent), not from position encoding.

#### Why qk_nope:qk_rope:v = 64:32:64 (2:1:2)?

This ratio allocates:
- **Most capacity to content matching** (64 nope dims for both Q and K)
- **Minimal but sufficient capacity to position** (32 rope dims)
- **Equal value capacity** (64 v dims, matching content key dims)

The 2:1:2 ratio was found empirically by DeepSeek. The intuition: position is a low-dimensional signal (you mainly need to know "how far away is this token?"), while content matching requires more dimensions to distinguish between semantically different tokens.

### Data Flow Visualization With Tensor Shapes

Complete shape trace for `B=2, L=512`:

```
INPUT
  hidden_states                              [2, 512, 2048]

QUERY PATH
  wq_a(hidden_states)                        [2, 512, 430]
  q_norm(...)                                [2, 512, 430]
  wq_b(...)                                  [2, 512, 1536]
  reshape → q                                [2, 512, 16, 96]
  split → q_nope                             [2, 512, 16, 64]
           q_pe                              [2, 512, 16, 32]
  RoPE(q_pe) → q_pe                          [2, 512, 16, 32]
  cat → q                                    [2, 512, 16, 96]
  transpose → q                              [2, 16, 512, 96]

KV PATH
  wkv_a(hidden_states)                       [2, 512, 175]
  split → kv_compressed                      [2, 512, 143]
           k_pe_raw                          [2, 512, 32]
  kv_norm(kv_compressed)                     [2, 512, 143]
  k_pe_raw.unsqueeze(2)                      [2, 512, 1, 32]
  RoPE(k_pe) → k_pe                          [2, 512, 1, 32]
  wkv_b(kv_compressed)                       [2, 512, 2048]
  reshape                                    [2, 512, 16, 128]
  split → k_nope                             [2, 512, 16, 64]
           v                                 [2, 512, 16, 64]
  expand(k_pe) → k_pe_expanded               [2, 512, 16, 32]
  cat → k                                    [2, 512, 16, 96]
  transpose → k                              [2, 16, 512, 96]
  transpose → v                              [2, 16, 512, 64]

ATTENTION
  q @ k^T × scale                            [2, 16, 512, 512]
  + causal_mask                              [2, 16, 512, 512]
  softmax (fp32)                             [2, 16, 512, 512]
  cast back to bf16                          [2, 16, 512, 512]
  @ v                                        [2, 16, 512, 64]

OUTPUT
  transpose                                  [2, 512, 16, 64]
  reshape                                    [2, 512, 1024]
  wo(...)                                    [2, 512, 2048]

CACHE (if use_cache=True)
  kv_compressed                              [2, 512, 143]
  k_pe                                       [2, 512, 1, 32]
```

### Parameter Count Breakdown

```
Component          | Shape           | Parameters  | % of MLA
─────────────────────────────────────────────────────────────
wq_a               | [430, 2048]     |   880,640   | 20.3%
q_norm             | [430]           |       430   |  0.0%
wq_b               | [1536, 430]     |   660,480   | 15.2%
wkv_a              | [175, 2048]     |   358,400   |  8.3%
kv_norm            | [143]           |       143   |  0.0%
wkv_b              | [2048, 143]     |   292,864   |  6.8%
wo                 | [2048, 1024]    | 2,097,152   | 48.3%
freqs_cis (buffer) | [4096, 16]      |     (buf)   |   —
─────────────────────────────────────────────────────────────
TOTAL              |                 | 4,290,109   | 100%
Per layer: ~4.3M parameters
16 layers: ~68.6M parameters (total MLA)
```

The output projection `wo` dominates at 48.3% of MLA parameters. This is typical — the "fan-out" from compressed attention back to full hidden size is the most expensive part.

### Modern Optimizations (2026)

#### Flash Attention Compatibility

The current NanoSeek implementation uses **explicit attention weight materialization** (`torch.matmul(q, k.T)` → softmax → `torch.matmul(weights, v)`). This is the "textbook" implementation for clarity.

In production 2026 deployments, you would replace this with Flash Attention 2/3, which fuses the entire attention computation into a single GPU kernel, avoiding materializing the `[B, H, L, L]` attention matrix. MLA is compatible with Flash Attention — you just pass the assembled Q, K, V tensors. The only consideration is that Q/K have `d_qk=96` and V has `d_v=64`, which are different (Flash Attention handles this via padding or split-K kernels).

The canonical incantation would be:
```python
# With flash_attn library (production path)
from flash_attn import flash_attn_func
attn_output = flash_attn_func(q, k, v, softmax_scale=self.softmax_scale, causal=True)
```

#### Mixed Precision Strategy

The NanoSeek MLA follows the standard mixed-precision pattern:
- **bf16**: All matrix multiplications (wq_a, wq_b, wkv_a, wkv_b, wo, attention)
- **fp32**: Softmax computation (critical for numerical stability)
- **fp32**: RMSNorm internal computation (accumulated in fp32, cast back to bf16)

```python
# From the actual code:
attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
#                                              ↑ fp32 softmax    ↑ cast back
```

This is non-negotiable for training stability. bf16 softmax causes gradient underflow at scale.

### 💡 Feynman Explainer: The Airport Analogy

*Imagine an airport (the transformer) processing passengers (tokens). In standard MHA, every gate agent (attention head) keeps a complete dossier on every passenger — name, destination, seat preference, frequent flyer status, meal choice. 16 gate agents × full dossier = massive filing cabinet (KV cache).*

*MLA says: "Wait — most of that information overlaps between agents. Let's have a central desk create a compressed summary card (143 dimensions) for each passenger, plus a boarding pass with their gate number (32 dimensions for position). Each agent can reconstruct the information they need from the summary card."*

*The summary card is tiny compared to 16 full dossiers (175 vs 4,096 values). But any agent can still figure out everything they need by "reading" the card through their own lens (the up-projection wkv_b). The boarding pass (RoPE component) is universal — every agent can see where the passenger is sitting.*

*The trade-off: agents do a tiny bit more work to "read" the summary card (the reconstruction matmul). But the filing cabinet (KV cache) is 23× smaller, so you can handle 23× more passengers (longer context) in the same space.*

---

## 🔴 STAGE 3: GROUND TRUTH

### What the Model is Learning To Predict

NanoSeek (like all autoregressive language models) is trained with **next-token prediction**. The ground truth for each position is simply the next token in the sequence.

```
Input tokens:   [The, cat, sat, on,  the, mat]
Labels:         [cat, sat, on,  the, mat, <eos>]
```

In practice, the labels are the **same tensor as input_ids, shifted by one position**:

```python
# From NanoSeekModel._compute_loss():
shift_logits = logits[:, :-1, :].contiguous()   # Predictions for positions 0..L-2
shift_labels = labels[:, 1:].contiguous()        # Targets: tokens at positions 1..L-1
```

### Concrete Example

```
Batch element 0:
  input_ids:  [1042, 553, 12004, 87, 3001, 449, 22156, ...]
  labels:     [1042, 553, 12004, 87, 3001, 449, 22156, ...]  (same tensor)

  At position 0: model sees token 1042, must predict token 553
  At position 1: model sees tokens [1042, 553], must predict 12004
  At position 2: model sees tokens [1042, 553, 12004], must predict 87
  ...

  shift_logits covers positions 0 through L-2
  shift_labels covers positions 1 through L-1
```

MLA's output feeds into the full model pipeline: `MLA output → residual → LayerNorm → FFN/MoE → ... → final LayerNorm → lm_head → logits`. The ground truth operates on the **final logits**, not directly on MLA's output. But MLA's quality directly determines how well the model can attend to relevant context for each prediction.

### MTP Ground Truth

NanoSeek also uses Multi-Token Prediction, which creates **additional shifted targets**:

```
Main head:   predict token at position t+1
MTP head 0:  predict token at position t+2
```

MTP loss is computed separately and added to the main loss with a weight schedule (0.3 → 0.1). This doesn't change MLA's architecture — MTP operates on the hidden states that MLA produces.

---

## 🔴 STAGE 4: LOSS FUNCTION

### Cross-Entropy on Logits

The loss function is standard cross-entropy between the model's predicted probability distribution over the vocabulary and the one-hot ground truth:

```python
main_loss = F.cross_entropy(
    shift_logits.view(-1, shift_logits.size(-1)),  # [B×(L-1), 65536]
    shift_labels.view(-1),                          # [B×(L-1)]
    ignore_index=-100,
)
```

**Mathematically:**

```
L = -1/(N) Σᵢ log( softmax(zᵢ)[yᵢ] )

Where:
  zᵢ ∈ ℝ^V  = logit vector for position i (V = 65536 = vocab_size)
  yᵢ ∈ {0, ..., V-1}  = target token ID
  N = number of non-ignored positions
```

### How Gradients Flow Back Through MLA

The gradient path from loss to MLA parameters traverses the entire model in reverse. Here is the chain rule decomposition for one MLA weight, say `wq_a`:

```
∂L/∂W_q_a = ∂L/∂logits × ∂logits/∂h_final × ∂h_final/∂h_MLA_out ×
             ∂h_MLA_out/∂attn_out × ∂attn_out/∂q × ∂q/∂q_compressed ×
             ∂q_compressed/∂W_q_a

Where each factor corresponds to:
  ∂L/∂logits           — from cross-entropy
  ∂logits/∂h_final     — through lm_head linear
  ∂h_final/∂h_MLA_out  — through subsequent layers + residual stream
  ∂h_MLA_out/∂attn_out — through wo projection
  ∂attn_out/∂q         — through attention (softmax × V)
  ∂q/∂q_compressed     — through wq_b projection
  ∂q_compressed/∂W_q_a — the down-projection itself
```

The **critical gradient bottleneck** in MLA is the low-rank compression. Gradients must flow through the 430-dim (query) and 143-dim (KV) bottlenecks. This is why the RMSNorm layers at these bottlenecks are essential — they prevent gradient vanishing/exploding through the narrow latent space.

### Gradient Flow Through Attention

The softmax in attention creates a gradient distribution where:
- Tokens that received high attention weights get strong gradients
- Tokens that received near-zero attention get near-zero gradients

This is computed in **fp32** regardless of model precision:

```python
attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
```

The `dtype=torch.float32` ensures that:
1. The softmax computation doesn't suffer from bf16 precision loss
2. Very small attention weights (e.g., 1e-7) are preserved rather than rounded to zero
3. The gradient `∂softmax/∂z` is numerically stable (softmax gradients involve products of probabilities, which can underflow in bf16)

### Total Loss Composition

```python
total_loss = main_loss + mtp_weight × mtp_loss + aux_loss + indexer_loss

Where:
  main_loss:    cross-entropy on next-token prediction
  mtp_loss:     cross-entropy on t+2 token prediction (weight: 0.3 → 0.1)
  aux_loss:     MoE sequence-level load balancing (α = 0.0001)
  indexer_loss: DSA Lightning Indexer training signal (when DSA enabled)
```

MLA is affected by all of these through the gradient chain, but the dominant signal is always `main_loss`.

---

## 🔴 STAGE 5: OUTPUTS

### MLA Module Output

The MLA module returns:

```python
return output, present_key_value

# output:             [B, L, 2048]  — attention output (same shape as input)
# present_key_value:  Tuple of (kv_compressed, k_pe) or None
#   kv_compressed:    [B, L, 143]
#   k_pe:             [B, L, 1, 32]
```

### From MLA to Final Logits

The MLA output goes through the rest of the decoder layer and model:

```
MLA output [B, L, 2048]
  → + residual                           [B, L, 2048]
  → post_attention_layernorm (RMSNorm)   [B, L, 2048]
  → FFN / MoE                            [B, L, 2048]
  → + residual                           [B, L, 2048]
  → (repeat for remaining layers)
  → final norm (RMSNorm)                 [B, L, 2048]
  → lm_head (Linear)                    [B, L, 65536]
```

The final logits tensor `[B, L, 65536]` contains unnormalized log-probabilities over the entire vocabulary for each position in the sequence.

### Inference Output

During autoregressive generation:
1. Only the **last position's logits** matter: `logits[:, -1, :]` → `[B, 65536]`
2. Apply temperature scaling, top-k/top-p filtering
3. Sample or argmax to get the next token
4. Feed the new token back, using the KV cache from previous steps

The KV cache grows by exactly **175 values per token per layer** at each generation step:
```
Step 0 (prompt, L=512):  cache = 175 × 512 × 16 layers = 1,433,600 values
Step 1 (1 new token):    cache = 175 × 513 × 16 layers = 1,436,400 values
Step 100 (100 new):      cache = 175 × 612 × 16 layers = 1,713,600 values
```

---

## Full Example Walkthrough

One complete forward pass through MLA with `B=2, L=512, hidden_size=2048`.

### Step 0: Input

```
hidden_states: [2, 512, 2048] (bf16, ~4 MB)
```

### Step 1: Query Down-Projection

```
q = wq_a(hidden_states)
  [2, 512, 2048] × [430, 2048]^T → [2, 512, 430]
  FLOPs: 2 × 2 × 512 × 2048 × 430 = 1,804,861,440 ≈ 1.8 GFLOPs
```

### Step 2: Query Normalization

```
q = q_norm(q)
  RMSNorm on last dim (430): compute RMS, divide, scale
  [2, 512, 430] → [2, 512, 430]
  FLOPs: negligible relative to matmuls
```

### Step 3: Query Up-Projection

```
q = wq_b(q)
  [2, 512, 430] × [1536, 430]^T → [2, 512, 1536]
  FLOPs: 2 × 2 × 512 × 430 × 1536 = 1,351,188,480 ≈ 1.4 GFLOPs
```

### Step 4: Query Reshape and Split

```
q = q.view(2, 512, 16, 96)
q_nope = q[:, :, :, :64]           → [2, 512, 16, 64]
q_pe   = q[:, :, :, 64:]           → [2, 512, 16, 32]
```

### Step 5: KV Down-Projection

```
kv = wkv_a(hidden_states)
  [2, 512, 2048] × [175, 2048]^T → [2, 512, 175]
  FLOPs: 2 × 2 × 512 × 2048 × 175 = 734,003,200 ≈ 0.7 GFLOPs
```

### Step 6: KV Split and Normalize

```
kv_compressed = kv[:, :, :143]      → [2, 512, 143]
k_pe_raw      = kv[:, :, 143:]      → [2, 512, 32]
kv_compressed = kv_norm(kv_compressed) → [2, 512, 143]
```

### Step 7: RoPE

```
k_pe = k_pe_raw.unsqueeze(2)        → [2, 512, 1, 32]
k_pe = apply_rotary_emb(k_pe, freqs) → [2, 512, 1, 32]
q_pe = apply_rotary_emb(q_pe, freqs) → [2, 512, 16, 32]
```

### Step 8: KV Up-Projection

```
kv_expanded = wkv_b(kv_compressed)
  [2, 512, 143] × [2048, 143]^T → [2, 512, 2048]
  FLOPs: 2 × 2 × 512 × 143 × 2048 = 599,785,472 ≈ 0.6 GFLOPs

kv_expanded = kv_expanded.view(2, 512, 16, 128)
k_nope = kv_expanded[:, :, :, :64]  → [2, 512, 16, 64]
v      = kv_expanded[:, :, :, 64:]  → [2, 512, 16, 64]
```

### Step 9: Assemble Q, K

```
q = cat([q_nope, q_pe], dim=-1)      → [2, 512, 16, 96]
k_pe_exp = k_pe.expand(-1, -1, 16, -1) → [2, 512, 16, 32]
k = cat([k_nope, k_pe_exp], dim=-1)  → [2, 512, 16, 96]
```

### Step 10: Attention

```
q = q.transpose(1, 2)                → [2, 16, 512, 96]
k = k.transpose(1, 2)                → [2, 16, 512, 96]
v = v.transpose(1, 2)                → [2, 16, 512, 64]

scores = q @ k^T × 0.10206           → [2, 16, 512, 512]
  FLOPs: 2 × 2 × 16 × 512 × 96 × 512 = 1,610,612,736 ≈ 1.6 GFLOPs

scores = scores + causal_mask
weights = softmax(scores, fp32)       → [2, 16, 512, 512]
weights = weights.to(bf16)

attn_out = weights @ v                → [2, 16, 512, 64]
  FLOPs: 2 × 2 × 16 × 512 × 512 × 64 = 1,073,741,824 ≈ 1.1 GFLOPs
```

### Step 11: Output Projection

```
attn_out = attn_out.transpose(1, 2)   → [2, 512, 16, 64]
attn_out = attn_out.reshape(2, 512, 1024)
output = wo(attn_out)                  → [2, 512, 2048]
  FLOPs: 2 × 2 × 512 × 1024 × 2048 = 4,294,967,296 ≈ 4.3 GFLOPs
```

### Total FLOPs for One MLA Layer

```
wq_a:       1.8 GFLOPs
wq_b:       1.4 GFLOPs
wkv_a:      0.7 GFLOPs
wkv_b:      0.6 GFLOPs
QK^T:       1.6 GFLOPs
AV:         1.1 GFLOPs
wo:         4.3 GFLOPs
───────────────────────
Total:     ~11.5 GFLOPs per layer
16 layers: ~184 GFLOPs per forward pass (attention only)
```

### KV Cache at End of Forward Pass

```
kv_compressed: [2, 512, 143]  → 146,432 values × 2 bytes = 292,864 bytes
k_pe:          [2, 512, 1, 32] → 32,768 values × 2 bytes = 65,536 bytes
────────────────────────────────────────────────────────────────────────
Per layer:                                                   358,400 bytes
16 layers:                                                 5,734,400 bytes ≈ 5.5 MB

vs Standard MHA:
K: [2, 512, 16, 128] → 2,097,152 values × 2 bytes = 4,194,304 bytes
V: [2, 512, 16, 128] → 2,097,152 values × 2 bytes = 4,194,304 bytes
Per layer:                                            8,388,608 bytes
16 layers:                                          134,217,728 bytes ≈ 128 MB

Compression: 128 MB / 5.5 MB ≈ 23.4×
```

---

## Common Misconceptions

### "MLA is just GQA with fewer groups"

**Wrong.** GQA shares full-dimensional K/V heads across groups of query heads. Each group still stores its own `head_dim`-dimensional K and V. MLA compresses **all heads** into a single latent and reconstructs — this is a fundamentally different operation (learned compression vs. hard parameter sharing). MLA can express richer cross-head interactions because the up-projection `wkv_b` can produce any linear function of the shared latent for each head.

### "MLA loses quality because of compression"

**Mostly wrong.** Empirically, MLA matches or slightly exceeds standard MHA quality at the same parameter count. The information bottleneck acts as a regularizer — it forces the model to learn more efficient KV representations. The slight compute overhead (reconstruction matmul) is irrelevant for quality. The only scenario where MLA might lag is when the `kv_lora_rank` is set too aggressively low (e.g., < 0.03× hidden), but the standard 0.07× ratio has been extensively validated.

### "You need to store the reconstructed K and V for backprop"

**Partly correct.** During training, you need to store intermediate activations for backward pass, so the memory savings of MLA apply primarily to **inference KV cache**. During training, the memory profile is dominated by activations and optimizer states anyway. The training benefit of MLA is architectural (better representations via bottleneck) rather than memory-based.

### "The RoPE component is wasted cache space"

**Wrong.** Without the separate RoPE component, MLA would be unable to distinguish positions. You'd need to either apply RoPE to the compressed latent (which would make each position's latent different, eliminating compression benefits for reconstruction) or abandon positional encoding entirely. The 32 dims for `k_pe` is a tiny price (18% of total cache) for full positional awareness.

### "MLA can't work with Flash Attention"

**Wrong.** MLA produces standard Q, K, V tensors (just with non-standard dimensions). Flash Attention works perfectly — you just pass `q [B, L, H, 96]`, `k [B, L, H, 96]`, `v [B, L, H, 64]`. The only nuance is that Q/K and V have different head dimensions, which modern Flash Attention implementations handle natively.

---

## Production Gotchas

### 1. Softmax Scale Factor

The softmax scale in NanoSeek is:
```python
self.softmax_scale = mscale / math.sqrt(self.qk_head_dim)
# = 1.0 / sqrt(96) ≈ 0.10206
```

Note this uses `qk_head_dim = 96` (nope + rope), **not** the standard `head_dim = 128`. If you copy MLA code and accidentally use the wrong dimension for scaling, attention scores will be systematically too large or too small, causing either peaked attention (poor diversity) or flat attention (poor selectivity).

### 2. RMSNorm Before Up-Projection is Non-Negotiable

Removing `q_norm` or `kv_norm` appears to work initially but causes training instability after ~1000 steps. The norms ensure the up-projection input has consistent magnitude regardless of how the down-projection scales. Without them, gradient variance through the bottleneck becomes unbounded.

### 3. k_pe Dimension in Cache is [B, L, 1, 32], not [B, L, 32]

The extra dimension (the "1" representing a single shared head) is critical for correct broadcasting during attention. If you squeeze it away for "cleanliness" in your cache implementation, the RoPE application and head expansion will fail silently (wrong broadcasts) or loudly (shape mismatches). The NanoSeek code maintains this carefully:

```python
k_pe_current = k_pe_current.unsqueeze(2)  # [B, L, 32] → [B, L, 1, 32]
```

### 4. Position Offset During Incremental Decoding

When generating token-by-token with KV cache, the position offset for RoPE must account for all previously cached tokens:

```python
position_offset = kv_len - seq_len  # kv_len includes cached tokens
q_freqs = self.freqs_cis[position_offset:position_offset + seq_len]
```

Getting this wrong is one of the most common bugs in KV-cached attention implementations. With MLA, the bug manifests subtly: outputs look reasonable for short generations but diverge for longer sequences because Q and K RoPE rotations are misaligned.

### 5. Cache Concatenation Order

When concatenating new KV with cached KV, the order matters:
```python
kv_compressed = torch.cat([cached_kv, kv_compressed], dim=1)   # old, then new
k_pe = torch.cat([cached_k_pe, k_pe_current], dim=1)          # same order
```

Reversing this order produces valid-looking outputs that are completely wrong because causal masking assumes chronological ordering.

### 6. wkv_b Output Dimension

The up-projection `wkv_b` maps to `num_heads × (qk_nope_head_dim + v_head_dim)`, **not** `num_heads × head_dim`. In NanoSeek: `16 × (64 + 64) = 2048`, which coincidentally equals `hidden_size`. But this is a coincidence — changing `v_head_dim` would change the `wkv_b` output dimension but not `hidden_size`.

---

## 2026 Best Practices

### Architecture

1. **Preserve DeepSeek ratios**: `q_lora_rank / hidden ≈ 0.21`, `kv_lora_rank / hidden ≈ 0.07`. These have been validated across scales from 1B to 671B. Do not "optimize" them without extensive ablation.

2. **Always decouple RoPE**: The nope/rope split is not optional — it's what makes the compression work. The specific ratio (2:1 nope:rope) can be adjusted but the principle is load-bearing.

3. **Use separate q_lora_rank and kv_lora_rank**: Query needs more capacity than KV for the latent (0.21× vs 0.07×). This asymmetry is important — the query must represent "what to look for" which is more complex than "what information to store."

### Training

4. **fp32 softmax, bf16 everything else**: This is the minimum viable mixed-precision strategy. Do not attempt fp16 softmax at scale — it will fail during training even if it passes unit tests.

5. **Gradient checkpointing interacts with MLA**: When using activation checkpointing, the recomputed MLA forward pass must produce identical results. This requires deterministic RoPE frequency lookups and consistent cache handling. NanoSeek handles this correctly by disabling caching during gradient-checkpointed layers.

6. **Monitor attention entropy per head**: If any head collapses to near-zero entropy (always attending to the same token), the corresponding dimensions in the KV latent are being wasted. This is more likely with MLA than standard MHA because the shared bottleneck can make some heads redundant.

### Inference

7. **Batch the reconstruction matmul**: When decoding, the `wkv_b` reconstruction over the full cached sequence is the compute bottleneck. Batch this across layers where possible, and consider quantizing `wkv_b` weights for inference (the reconstruction is less precision-sensitive than the attention itself).

8. **Profile reconstruction vs. attention**: For very long contexts (>16K), the `wkv_b` reconstruction cost grows linearly with context length. At some point, it may be worth caching partially-reconstructed K/V for hot tokens. This is an active area of systems research in 2026.

9. **KV cache quantization stacks with MLA**: You can quantize the 143-dim compressed latent to int8 or int4 for even more cache savings. The compression + quantization stack gives ~50-100× reduction over standard MHA with fp16 cache.

---

## When to Use MLA vs Not Use MLA

### Use MLA When:

- **Long-context inference is a primary use case**: MLA's 23× cache reduction is transformative for 32K+ context serving
- **Memory-bound inference**: When GPU memory (not compute) limits your batch size or context length
- **Training quality matters as much as inference efficiency**: MLA doesn't sacrifice quality for cache savings
- **You need multi-head expressiveness**: Unlike MQA/GQA, MLA preserves per-head diversity

### Consider Alternatives When:

- **Ultra-low latency single-token generation**: The reconstruction matmul adds a small fixed cost per step. For applications where every microsecond of latency matters (e.g., real-time speech), standard MHA with smaller models might be faster.
- **Very short context only**: If your use case never exceeds 512 tokens, KV cache is not a bottleneck and MLA's complexity isn't justified.
- **Hardware without efficient matmul**: MLA relies on fast small matrix multiplies for reconstruction. On hardware without efficient GEMM (e.g., some edge devices), the overhead may not be worth it.
- **Training-only models (never deployed for inference)**: If you're training embeddings or fine-tuning and never serving the model, MLA's inference benefits are irrelevant.

### The Decision Matrix

```
┌─────────────────────┬───────────────┬──────────────┬──────────────┐
│ Scenario            │ MHA           │ GQA          │ MLA          │
├─────────────────────┼───────────────┼──────────────┼──────────────┤
│ Short context       │ ✓ Simple      │ ✓ OK         │ ○ Overkill   │
│ Long context (32K+) │ ✗ OOM         │ ○ Marginal   │ ✓ Best       │
│ High throughput     │ ✗ Memory-bound│ ○ Better     │ ✓ Best       │
│ Max quality         │ ✓ Baseline    │ ○ Slight loss│ ✓ Equal+     │
│ Simple impl         │ ✓ Trivial     │ ✓ Easy       │ ✗ Complex    │
│ Edge deployment     │ ○ Depends     │ ✓ Good       │ ○ Depends    │
└─────────────────────┴───────────────┴──────────────┴──────────────┘

✓ = recommended, ○ = acceptable, ✗ = avoid
```

---

## Appendix A: Complete Weight Shapes

For reference, every learnable parameter in one MLA layer (NanoSeek-1B config):

```
wq_a.weight:    [430, 2048]     torch.bfloat16    880,640 params
q_norm.weight:  [430]           torch.bfloat16        430 params
wq_b.weight:    [1536, 430]     torch.bfloat16    660,480 params
wkv_a.weight:   [175, 2048]     torch.bfloat16    358,400 params
kv_norm.weight: [143]           torch.bfloat16        143 params
wkv_b.weight:   [2048, 143]     torch.bfloat16    292,864 params
wo.weight:      [2048, 1024]    torch.bfloat16  2,097,152 params
──────────────────────────────────────────────────────────────
Total:                                          4,290,109 params
```

## Appendix B: Comparison of Cache Formats

```
Method       │ Cache per token per layer │ 16 layers, 4K ctx, B=1, bf16 │
─────────────┼──────────────────────────┼──────────────────────────────┤
MHA          │ 2 × 16 × 128 = 4,096    │ 4,096 × 4,096 × 16 × 2 = 512 MB   │
GQA (4 grp)  │ 2 × 4 × 128 = 1,024     │ 1,024 × 4,096 × 16 × 2 = 128 MB   │
MQA          │ 2 × 1 × 128 = 256       │   256 × 4,096 × 16 × 2 =  32 MB   │
MLA          │ 143 + 32 = 175           │   175 × 4,096 × 16 × 2 =  22 MB   │
─────────────┼──────────────────────────┼──────────────────────────────┤
MLA vs MHA   │ 23.4× compression        │ 23.4× less memory                  │
MLA vs GQA-4 │ 5.9× compression         │ 5.9× less memory                   │
MLA vs MQA   │ 1.5× compression         │ 1.5× less memory (but MLA > MQA quality) │
```

## Appendix C: NanoSeek MLA vs DeepSeek V3 MLA

```
Parameter            │ NanoSeek-1B    │ DeepSeek V3      │ Ratio Preserved?
─────────────────────┼────────────────┼──────────────────┼─────────────────
hidden_size          │ 2,048          │ 7,168            │ —
num_heads            │ 16             │ 128              │ —
q_lora_rank          │ 430            │ 1,536            │ 0.21× ✓
kv_lora_rank         │ 143            │ 512              │ 0.07× ✓
qk_nope_head_dim     │ 64             │ 128              │ —
qk_rope_head_dim     │ 32             │ 64               │ —
v_head_dim           │ 64             │ 128              │ —
qk_head_dim          │ 96             │ 192              │ —
Cache per token      │ 175            │ 576              │ —
Compression vs MHA   │ 23.4×          │ 22.2×            │ ~matched ✓
```

The critical ratios (q_lora/hidden = 0.21, kv_lora/hidden = 0.07, compression ~23×) are preserved across scales. This means insights from training NanoSeek transfer directly to understanding production DeepSeek behavior.
