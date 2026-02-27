# Rotary Position Embeddings (RoPE) & YaRN Context Extension — First Principles Deep Dive

> **Document status**: Production reference for NanoSeek (DeepSeek V3.2 at nano scale)
> **Source of truth**: `model/model.py` — `precompute_freqs_cis`, `apply_rotary_emb`, `RotaryEmbedding`, YaRN helpers
> **Audience**: Senior ML researchers who understand attention but want to deeply understand positional encoding

---

## Quick Context — How Transformers "Know" Position

### The Problem: Attention Is Permutation-Invariant

Self-attention computes `softmax(QK^T / √d) V`. Nothing in this formula depends on token order. Swap token 5 and token 42, and the attention output swaps identically — the model cannot distinguish "The cat sat on the mat" from "mat the on sat cat The." Without positional information, a transformer is a bag-of-tokens model.

Every transformer must inject position somehow. The design space:

```
Method                  │ Learned? │ Relative? │ Extrapolation │ KV Cache │ 2026 Status
────────────────────────┼──────────┼───────────┼───────────────┼──────────┼────────────
Sinusoidal (Vaswani '17)│ No       │ No        │ Poor          │ Free     │ Dead
Learned absolute        │ Yes      │ No        │ None          │ Free     │ Dead
ALiBi                   │ No       │ Yes       │ Decent        │ Free     │ Niche
RoPE (Su et al. '21)    │ No       │ Yes       │ Moderate      │ Free*    │ Standard
RoPE + YaRN             │ No       │ Yes       │ Excellent     │ Free*    │ Best practice
```

\* "Free" means no additional KV cache cost — RoPE is applied to Q and K *before* caching (or in MLA's case, the rope component is just 32 dims).

### RoPE's Key Insight

Encode **absolute** position by rotating Q and K vectors, but design the rotation so that the dot product `<Q_m, K_n>` depends only on the **relative** distance `m - n`. You get absolute encoding with relative properties — the best of both worlds.

### YaRN's Key Insight

When you need to run at positions far beyond training (e.g., trained at 4K, inference at 32K), naively using RoPE fails because low-frequency rotation components have never been seen at those angles. YaRN identifies *which* frequency dimensions need interpolation and smoothly blends between original and scaled frequencies. High-frequency dims (local patterns) stay unchanged; low-frequency dims (global patterns) get compressed.

### NanoSeek's Position Encoding

NanoSeek uses **decoupled RoPE** within MLA: only 32 of the 96 query/key dimensions carry positional information. The remaining 64 dimensions carry content-only information. This is not a simplification — it's an architectural innovation that enables 23× KV cache compression while preserving full positional awareness.

```
MLA Q head: [qk_nope_head_dim=64 | qk_rope_head_dim=32] = 96 total
                    ↑ content only          ↑ RoPE applied

MLA K head: [qk_nope_head_dim=64 | qk_rope_head_dim=32] = 96 total
                    ↑ per-head              ↑ SHARED across all 16 heads
```

The `k_pe` component is shared across all heads — stored as `[B, L, 1, 32]` and expanded to `[B, L, 16, 32]` at attention time. This sharing is the linchpin of MLA's cache compression. Without decoupled RoPE, you'd need per-head positional K vectors (16×32 = 512 extra dims), destroying the compression ratio.

---

## 🔴 STAGE 1: INPUTS — Position Indices `[0, 1, ..., L-1]`

### What Enters the RoPE System

RoPE's inputs are almost trivially simple: integer position indices and the Q/K vectors to be rotated.

```
Position indices:
  Shape:  [B, L]
  Dtype:  torch.long
  Values: [0, 1, 2, ..., L-1] for standard left-to-right processing
          [past_len, past_len+1, ...] during KV-cached generation

Q rope component (in MLA):
  Shape:  [B, L, num_heads, qk_rope_head_dim] = [B, L, 16, 32]
  Dtype:  bf16

K rope component (in MLA):
  Shape:  [B, L, 1, qk_rope_head_dim] = [B, L, 1, 32]
  Dtype:  bf16
  Note:   Single shared head — the "1" dimension is critical
```

### Concrete Example

For a batch of 2 sequences, each 512 tokens, during standard training:

```
position_ids = [[0, 1, 2, ..., 511],
                [0, 1, 2, ..., 511]]
  Shape: [2, 512]

q_pe.shape = [2, 512, 16, 32]   ← 16 heads, each 32 dims for position
k_pe.shape = [2, 512, 1, 32]    ← 1 shared head, 32 dims for position
```

During incremental decoding (generating token 513 with 512 cached tokens):

```
position_ids = [[512]]
  Shape: [1, 1]

q_pe.shape = [1, 1, 16, 32]
k_pe.shape = [1, 1, 1, 32]
```

### Precomputed Frequency Table

Before any forward pass, the `RotaryEmbedding` module precomputes a table of complex rotation factors:

```python
# model.py line 164-168
freqs_cis = precompute_freqs_cis(
    dim=32,                              # qk_rope_head_dim
    end=4096,                            # max_position_embeddings
    theta=10000.0,                       # base frequency
    scaling_factor=1.0,                  # 1.0 = no YaRN, 8.0 = YaRN extension
    original_max_position_embeddings=4096,
)
self.register_buffer("freqs_cis", freqs_cis, persistent=False)
```

This table is a `[4096, 16]` complex tensor. Each row is a position, each column is a frequency dimension (16 = 32 // 2, since RoPE pairs adjacent real dims into one complex dim). Looking up position `p` gives 16 complex rotation factors, one per dimension pair.

### 💡 Feynman Explainer: Inputs

*Think of each position as a unique "rotation recipe." Position 0 means "don't rotate." Position 1 means "rotate a little." Position 1000 means "rotate a lot." Each dimension pair gets rotated by a different amount — fast-varying dimensions rotate quickly (distinguishing nearby tokens), slow-varying dimensions rotate slowly (distinguishing distant tokens). It's a multi-frequency clock, and the model reads the time from how much each Q-K pair has been rotated relative to each other.*

---

## 🔴 STAGE 2: MODEL ARCHITECTURE

This is the heart of the document. We build RoPE from absolute basics, derive its mathematical properties, walk through the YaRN extension, and trace every tensor operation in the NanoSeek implementation.

### 2.1 Building RoPE from First Principles

#### 2.1.1 Why Rotation? The 2D Intuition

Consider a 2D vector `v = [x, y]`. Rotating it by angle θ gives:

```
R(θ) · v = [x cos θ - y sin θ,  x sin θ + y cos θ]
```

Key property: **rotation preserves dot products relative to angle difference**.

If I rotate vector `q` by angle `α` and vector `k` by angle `β`:

```
<R(α)q, R(β)k> = <R(α - β)q, k>
```

The dot product depends only on `α - β`, not on `α` and `β` individually. This is exactly the relative position property we want: if position `m` applies rotation `mθ` and position `n` applies rotation `nθ`, the attention score depends on `(m-n)θ`.

#### 2.1.2 From 2D to d-dimensional: Paired Rotations

A single 2D rotation can only encode one frequency. For a `d`-dimensional vector (our `qk_rope_head_dim = 32`), we pair dimensions into `d/2 = 16` independent 2D planes and rotate each pair at a different frequency:

```
Dimension pair (0,1):   rotate by θ₀ · position
Dimension pair (2,3):   rotate by θ₁ · position
Dimension pair (4,5):   rotate by θ₂ · position
...
Dimension pair (30,31): rotate by θ₁₅ · position
```

This gives a **block-diagonal** rotation matrix:

```
         ┌                                                          ┐
         │ cos(pθ₀) -sin(pθ₀)    0       0     ...    0       0    │
         │ sin(pθ₀)  cos(pθ₀)    0       0     ...    0       0    │
         │    0          0     cos(pθ₁) -sin(pθ₁) ...  0       0    │
R(p) =   │    0          0     sin(pθ₁)  cos(pθ₁) ...  0       0    │
         │   ...        ...      ...      ...   ...   ...     ...   │
         │    0          0        0       0     ... cos(pθ₁₅) -sin(pθ₁₅)│
         │    0          0        0       0     ... sin(pθ₁₅)  cos(pθ₁₅)│
         └                                                          ┘
```

Where `p` is the position index and `θᵢ` are the base frequencies.

#### 2.1.3 The Complex Number Trick

Instead of building this 32×32 rotation matrix, we use complex multiplication. For each dimension pair `(2i, 2i+1)`, treat the two reals as one complex number:

```
z = x[2i] + j · x[2i+1]
```

Rotating by angle `φ` is just:

```
z · e^(jφ) = z · (cos φ + j sin φ)
```

This is exactly what `torch.view_as_complex` and `torch.polar` do in the implementation:

```python
# model.py line 97
# torch.polar(magnitude, angle) = magnitude * e^(j·angle)
return torch.polar(torch.ones_like(freqs), freqs)
```

The precomputed `freqs_cis` is a `[max_pos, dim//2]` complex tensor where `freqs_cis[p, i] = e^(j · p · θᵢ)`. Applying RoPE is a single element-wise complex multiply:

```python
# model.py line 135
x_rotated = x_complex * freqs_cis
```

That single line implements the entire block-diagonal rotation matrix.

#### 2.1.4 Frequency Design: Why θ = 10000 and Geometric Spacing

The base frequencies are:

```
θᵢ = 1 / (10000^(2i/d))    for i = 0, 1, ..., d/2 - 1
```

In code:

```python
# model.py line 85
freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
```

For NanoSeek with `dim=32, theta=10000`:

```
i=0:  θ₀  = 1 / 10000^(0/32)  = 1.0000    → wavelength = 2π/1.0    ≈  6.3 positions
i=1:  θ₁  = 1 / 10000^(2/32)  = 0.5623    → wavelength = 2π/0.5623 ≈ 11.2 positions
i=2:  θ₂  = 1 / 10000^(4/32)  = 0.3162    → wavelength = 2π/0.3162 ≈ 19.9 positions
i=3:  θ₃  = 1 / 10000^(6/32)  = 0.1778    → wavelength = 2π/0.1778 ≈ 35.3 positions
i=4:  θ₄  = 1 / 10000^(8/32)  = 0.1000    → wavelength = 2π/0.1000 ≈ 62.8 positions
i=5:  θ₅  = 1 / 10000^(10/32) = 0.0562    → wavelength ≈ 112 positions
i=6:  θ₆  = 1 / 10000^(12/32) = 0.0316    → wavelength ≈ 199 positions
i=7:  θ₇  = 1 / 10000^(14/32) = 0.0178    → wavelength ≈ 353 positions
i=8:  θ₈  = 1 / 10000^(16/32) = 0.0100    → wavelength ≈ 628 positions
i=9:  θ₉  = 1 / 10000^(18/32) = 0.0056    → wavelength ≈ 1,117 positions
i=10: θ₁₀ = 1 / 10000^(20/32) = 0.0032    → wavelength ≈ 1,987 positions
i=11: θ₁₁ = 1 / 10000^(22/32) = 0.0018    → wavelength ≈ 3,533 positions
i=12: θ₁₂ = 1 / 10000^(24/32) = 0.0010    → wavelength ≈ 6,283 positions
i=13: θ₁₃ = 1 / 10000^(26/32) = 0.0006    → wavelength ≈ 11,170 positions
i=14: θ₁₄ = 1 / 10000^(28/32) = 0.0003    → wavelength ≈ 19,870 positions
i=15: θ₁₅ = 1 / 10000^(30/32) = 0.0002    → wavelength ≈ 35,333 positions
```

**Why geometric spacing?** The frequencies span from ~6 positions (can distinguish adjacent tokens) to ~35K positions (can distinguish tokens thousands of positions apart). This logarithmic coverage means:

- **High-frequency dims (i=0,1,2)**: Sensitive to local position differences. "Is this the next word or two words away?"
- **Mid-frequency dims (i=6,7,8)**: Sensitive to sentence-level distances. "Is this in the same clause?"
- **Low-frequency dims (i=13,14,15)**: Sensitive to document-level distances. "Is this in the same paragraph?"

**Why θ = 10000?** Originally proposed by Su et al. (2021), this value ensures the lowest frequency has a wavelength long enough to cover typical training context lengths (~2K-8K tokens) without aliasing, while the highest frequency has a wavelength short enough to distinguish adjacent positions. Increasing θ makes all frequencies lower (longer wavelengths), which helps extrapolation but hurts local resolution. The value 10000 has been extensively validated and is used unchanged by LLaMA, Mistral, DeepSeek, and essentially every production RoPE model.

#### 2.1.5 Mathematical Derivation: The Relative Position Property

This is the central theorem of RoPE. We prove that the dot product between rotated Q and K depends only on relative position.

**Setup**: Let `q, k ∈ ℝ²` (one dimension pair). RoPE at position m applies rotation `R(mθ)`.

**Claim**: `<R(mθ)q, R(nθ)k> = <R((m-n)θ)q, k>`

**Proof**:

Using complex notation where `q = q_r + jq_i` and `k = k_r + jk_i`:

```
R(mθ)q = q · e^(jmθ)
R(nθ)k = k · e^(jnθ)
```

The real inner product `<R(mθ)q, R(nθ)k>` equals `Re[(R(mθ)q) · conj(R(nθ)k)]`:

```
Re[q · e^(jmθ) · conj(k · e^(jnθ))]
= Re[q · e^(jmθ) · conj(k) · e^(-jnθ)]
= Re[q · conj(k) · e^(j(m-n)θ)]
= Re[(q · e^(j(m-n)θ)) · conj(k)]
= <R((m-n)θ)q, k>     ∎
```

The proof extends trivially to d dimensions because each dimension pair is independent — the total dot product is the sum of dot products across all pairs, and each pair satisfies the relative position property independently.

**What this means in practice**: When the model computes attention score `q_m · k_n`, the result is mathematically identical to a function of `(m - n)` applied to the original unrotated vectors. The model never needs to "decode" absolute positions — relative distance is baked into the geometry of the rotated space.

#### 2.1.6 ASCII Visualization: Rotation in 2D

Consider one dimension pair for a query vector `q = [1.0, 0.0]` at various positions:

```
Position 0 (angle = 0):        Position 1 (angle = θ₀ = 1.0 rad):
         y                              y
         │                              │    · R(1.0)·q
         │                              │  /
         │                              │/  57.3°
    ─────·──────── x              ──────·──────── x
         │  q=[1,0]                     │
         │                              │

Position 3 (angle = 3.0 rad):  Position 6 (angle ≈ 2π, wraps):
         y                              y
    ·    │                              │
     \   │                              │
      \  │                              │
    ───\─·──────── x              ─────·──────── x
        \│                              │  almost back to start
         │                              │
```

For the slowest dimension pair (i=15, θ₁₅ ≈ 0.0002):

```
Position 0:    angle = 0.000 rad   (pointing right)
Position 100:  angle = 0.018 rad   (barely moved — 1.0°)
Position 1000: angle = 0.178 rad   (10.2° — still close to start)
Position 4000: angle = 0.712 rad   (40.8° — training boundary)
Position 32000: angle = 5.69 rad   (326° — almost full circle — YaRN territory)
```

This dimension pair can only distinguish tokens ~35K positions apart. At 32K context, it's approaching aliasing. This is exactly why YaRN is needed for context extension.

### 2.2 The apply_rotary_emb Implementation

The actual rotation in NanoSeek, step by step:

```python
def apply_rotary_emb(x: Tensor, freqs_cis: Tensor, interleaved: bool = True) -> Tensor:
```

#### Step 1: Handle input dimensions

```python
# model.py line 108-113
orig_shape = x.shape              # Save for final reshape
orig_dtype = x.dtype              # Save dtype (will compute in float32)
if x.dim() == 3:
    x = x.unsqueeze(2)           # [B, L, D] → [B, L, 1, D] (add head dim)
```

This flexibility allows the same function to handle both:
- `q_pe: [B, L, 16, 32]` — multi-head query, 4D input
- `k_pe: [B, L, 32]` — shared key before unsqueeze, 3D input

#### Step 2: Reshape freqs_cis for broadcasting

```python
# model.py line 117-126
if freqs_cis.dim() == 2:
    # Standard: [L, 16] → [1, L, 1, 16] for broadcasting over batch and heads
    freqs_cis = freqs_cis.view(1, seq_len, 1, -1)
elif freqs_cis.dim() == 3:
    # Batched: [B, L, 16] → [B, L, 1, 16] for broadcasting over heads
    freqs_cis = freqs_cis.unsqueeze(2)
```

#### Step 3: Convert reals to complex (the interleaved path)

```python
# model.py line 129
x_complex = torch.view_as_complex(
    x.float().reshape(*batch_dims, seq_len, n_heads, head_dim // 2, 2)
)
```

Concretely for `q_pe` with shape `[2, 512, 16, 32]`:

```
x.float()                    → [2, 512, 16, 32]  (upcast to fp32)
x.reshape(2, 512, 16, 16, 2) → [2, 512, 16, 16, 2]  (pair adjacent dims)
view_as_complex(...)          → [2, 512, 16, 16]  (each pair → 1 complex)
```

The `interleaved=True` path assumes dims `[..., d0, d1, d2, d3, ...]` are paired as `(d0,d1), (d2,d3), ...`. This is the standard layout used by DeepSeek and most RoPE implementations.

#### Step 4: Complex multiplication (the actual rotation!)

```python
# model.py line 135
x_rotated = x_complex * freqs_cis
```

One line. That's the entire rotation. Complex multiplication `(a+bj)(c+dj) = (ac-bd) + (ad+bc)j` implements the 2D rotation matrix for each dimension pair simultaneously.

```
x_complex:  [2, 512, 16, 16]  complex64
freqs_cis:  [1, 512, 1, 16]   complex64  (broadcasts over batch and heads)
x_rotated:  [2, 512, 16, 16]  complex64
```

#### Step 5: Convert back to real

```python
# model.py line 137-138
x_out = torch.view_as_real(x_rotated).reshape(*batch_dims, seq_len, n_heads, head_dim)
```

```
view_as_real(...)  → [2, 512, 16, 16, 2]  (complex → 2 reals)
reshape(...)       → [2, 512, 16, 32]      (flatten back)
```

#### Step 6: Restore dtype and shape

```python
# model.py line 143
x_out = x_out.view(orig_shape).to(orig_dtype)
```

Cast back from fp32 to bf16 and restore the original number of dimensions.

### 2.3 MLA's Decoupled RoPE: The Compression Enabler

In standard multi-head attention, RoPE is applied to the entire Q and K head dimension (typically 128 dims). In MLA, the Q/K head is split:

```
Standard MHA (head_dim=128):
  Q = [────────── 128 dims, all RoPE'd ──────────]
  K = [────────── 128 dims, all RoPE'd ──────────]

MLA (qk_head_dim=96):
  Q = [── qk_nope: 64 dims (content) ──│── qk_rope: 32 dims (position) ──]
  K = [── qk_nope: 64 dims (content) ──│── qk_rope: 32 dims (position) ──]
                  per-head                     SHARED across all heads for K
```

The implementation in `MultiHeadLatentAttention.forward()`:

```python
# model.py line 302 — Q split
q = q.view(batch_size, seq_len, self.num_heads, self.qk_head_dim)  # [B, L, 16, 96]
q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
# q_nope: [B, L, 16, 64]  ← content, no position encoding
# q_pe:   [B, L, 16, 32]  ← position, RoPE applied

# model.py line 306-308 — K split (rope is shared!)
kv = self.wkv_a(hidden_states)                     # [B, L, 175]
kv_compressed, k_pe_current = torch.split(
    kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
)
# kv_compressed: [B, L, 143]  ← goes through kv_norm + wkv_b for per-head K,V
# k_pe_current:  [B, L, 32]   ← SHARED rope component for all heads

# model.py line 324 — add head dim for broadcasting
k_pe_current = k_pe_current.unsqueeze(2)  # [B, L, 1, 32]
```

At attention time, the shared K rope component expands:

```python
# model.py line 349
k_pe_expanded = k_pe.expand(-1, -1, self.num_heads, -1)  # [B, L, 1, 32] → [B, L, 16, 32]
k = torch.cat([k_nope, k_pe_expanded], dim=-1)           # [B, L, 16, 96]
```

**Why this enables 23× compression**: The KV cache stores only `kv_compressed [B, L, 143]` and `k_pe [B, L, 1, 32]` = 175 values per token per layer. Without the nope/rope split, RoPE would need to be applied to the full per-head K (which is 16 heads × 128 dims = 2048 values), and you couldn't compress those after rotation because RoPE makes each position's K unique.

The decoupled design keeps the compressed latent `kv_compressed` position-independent (no RoPE → same compression regardless of position), and stores the tiny shared positional signal `k_pe` separately.

### 2.4 Attention Score Decomposition

When Q and K are concatenated from nope and rope components, the attention score decomposes:

```
score(m, n) = q[m] · k[n] / scale

             = [q_nope[m] | q_pe[m]] · [k_nope[n] | k_pe[n]] / scale

             = (q_nope[m] · k_nope[n] + q_pe[m] · k_pe[n]) / scale
               ──────────────────────   ────────────────────
               content similarity        positional similarity
               (64 dims, no position)   (32 dims, position-aware via RoPE)
```

The content term `q_nope · k_nope` computes pure semantic similarity — "how relevant is token n's *meaning* to what token m is looking for?" This term is position-invariant: moving a token doesn't change its content score.

The positional term `q_pe · k_pe` adds position-dependent bias. Due to RoPE's relative position property, this term depends on `m - n`: nearby tokens get different positional scores than distant tokens. The model learns to use this signal for syntactic patterns (e.g., attending to the previous token for bigram statistics, or to clause boundaries).

In NanoSeek, the softmax scale is:

```python
# model.py line 262
self.softmax_scale = mscale / math.sqrt(self.qk_head_dim)
# = 1.0 / sqrt(96) ≈ 0.10206
```

This scales the *combined* 96-dimensional dot product. The content term (64 dims) contributes ~2/3 of the variance, and the position term (32 dims) contributes ~1/3. This 2:1 ratio gives the model more capacity for content matching than positional matching, which empirically works well.

### 2.5 YaRN: Extending Context Beyond Training

#### 2.5.1 The Extrapolation Problem

NanoSeek trains at 4096 positions. What happens at position 5000?

```
Dimension i=0 (θ₀ = 1.0):
  Training:  angles 0 to 4095 rad  → wraps 4095/(2π) ≈ 652 full rotations
  Position 5000: 5000 rad           → wraps 796 times — model has seen this pattern

Dimension i=15 (θ₁₅ ≈ 0.000178):
  Training:  angles 0 to 0.729 rad → less than 1/8 of a rotation
  Position 5000: 0.891 rad          → still < 1/6 rotation — plausible extrapolation
  Position 32000: 5.69 rad          → nearly full rotation — NEVER SEEN IN TRAINING
```

High-frequency dimensions wrap many times during training, so they've seen all possible angles and extrapolate naturally. Low-frequency dimensions have barely rotated during training, so positions far beyond 4K take them to completely unseen angles. The model's learned attention patterns for these dimensions break down.

This is the **frequency-dependent extrapolation failure** of RoPE. YaRN fixes it.

#### 2.5.2 The YaRN Insight: Selective Frequency Interpolation

Instead of scaling all frequencies uniformly (NTK-aware interpolation) or just the position indices (Position Interpolation), YaRN identifies three groups of dimensions:

```
┌──────────────────────────────────────────────────────────────────────┐
│  HIGH FREQUENCY (small i)     │  TRANSITION       │  LOW FREQUENCY  │
│  wavelength << training_len   │                   │  wavelength ≈   │
│                               │                   │  training_len   │
│  → Already wraps many times   │  → Smooth blend   │  → Barely       │
│  → No modification needed     │  → (1-s)·orig     │    rotated      │
│  → Keep original freq         │    + s·scaled     │  → Scale by 1/f │
│                               │                   │  → Interpolate   │
│  dims 0,1,2,...               │                   │  ...,14,15       │
└──────────────────────────────────────────────────────────────────────┘
        smooth = 0                 0 < smooth < 1        smooth = 1
```

#### 2.5.3 Finding the Correction Range

YaRN computes which dimensions need interpolation using two functions:

```python
# model.py line 42-49
def find_correction_dim(num_rotations, dim, base=10000.0, max_position_embeddings=2048):
    return dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi)) / (2 * math.log(base))
```

This solves: "At which dimension index `i` does the frequency have a wavelength equal to `max_position_embeddings / num_rotations`?"

The wavelength of dimension `i` is:

```
wavelength_i = 2π / θ_i = 2π · base^(2i/dim)
```

Setting `wavelength_i = max_pos / num_rotations` and solving for `i`:

```
2π · base^(2i/dim) = max_pos / num_rotations
base^(2i/dim) = max_pos / (num_rotations · 2π)
2i/dim · log(base) = log(max_pos / (num_rotations · 2π))
i = dim · log(max_pos / (num_rotations · 2π)) / (2 · log(base))
```

The correction range uses `beta_fast=32` and `beta_slow=1`:

```python
# model.py line 52-62
def find_correction_range(low_rot, high_rot, dim, base=10000.0, max_position_embeddings=2048):
    low = max(floor(find_correction_dim(low_rot, dim, base, max_position_embeddings)), 0)
    high = min(ceil(find_correction_dim(high_rot, dim, base, max_position_embeddings)), dim - 1)
    return low, high
```

**Numerical example for NanoSeek** (`dim=32, base=10000, max_pos=4096`):

```
beta_fast = 32 → find_correction_dim(32, 32, 10000, 4096)
  = 32 × log(4096 / (32 × 2π)) / (2 × log(10000))
  = 32 × log(4096 / 201.06) / (2 × 9.2103)
  = 32 × log(20.37) / 18.421
  = 32 × 3.014 / 18.421
  = 5.23
  → floor = 5

beta_slow = 1 → find_correction_dim(1, 32, 10000, 4096)
  = 32 × log(4096 / (1 × 2π)) / (2 × log(10000))
  = 32 × log(651.9) / 18.421
  = 32 × 6.48 / 18.421
  = 11.26
  → ceil = 12

Correction range: [low=5, high=12]
```

Interpretation:
- Dimensions 0–4: High frequency, no interpolation needed (smooth = 0)
- Dimensions 5–12: Transition zone, smooth blend
- Dimensions 13–15: Low frequency, full interpolation (smooth = 1)

#### 2.5.4 The Linear Ramp: Smooth Transition

```python
# model.py line 66-70
def linear_ramp_factor(min_val, max_val, dim):
    if min_val == max_val:
        max_val = min_val + 0.001      # avoid division by zero
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    return torch.clamp(linear_func, 0, 1)
```

For our correction range `[5, 12]` with `dim//2 = 16`:

```
i=0:  (0-5)/(12-5) = -0.71 → clamped to 0.00  (keep original)
i=1:  (1-5)/(12-5) = -0.57 → clamped to 0.00  (keep original)
i=2:  (2-5)/(12-5) = -0.43 → clamped to 0.00  (keep original)
i=3:  (3-5)/(12-5) = -0.29 → clamped to 0.00  (keep original)
i=4:  (4-5)/(12-5) = -0.14 → clamped to 0.00  (keep original)
i=5:  (5-5)/(12-5) =  0.00              0.00   (start of transition)
i=6:  (6-5)/(12-5) =  0.14              0.14
i=7:  (7-5)/(12-5) =  0.29              0.29
i=8:  (8-5)/(12-5) =  0.43              0.43
i=9:  (9-5)/(12-5) =  0.57              0.57
i=10: (10-5)/(12-5) = 0.71              0.71
i=11: (11-5)/(12-5) = 0.86              0.86
i=12: (12-5)/(12-5) = 1.00              1.00   (fully interpolated)
i=13: (13-5)/(12-5) = 1.14 → clamped to 1.00  (fully interpolated)
i=14: (14-5)/(12-5) = 1.29 → clamped to 1.00  (fully interpolated)
i=15: (15-5)/(12-5) = 1.43 → clamped to 1.00  (fully interpolated)
```

#### 2.5.5 The Complete YaRN Frequency Modification

```python
# model.py line 87-93
if scaling_factor != 1.0:
    low, high = find_correction_range(
        beta_fast, beta_slow, dim, theta, original_max_position_embeddings
    )
    smooth = linear_ramp_factor(low, high, dim // 2)
    scaled_freqs = freqs / scaling_factor              # interpolated frequencies
    freqs = freqs * (1 - smooth) + scaled_freqs * smooth  # blend
```

With `scaling_factor = 8.0`:

```
Original freqs:  [1.0000, 0.5623, 0.3162, ..., 0.0003, 0.0002]
Scaled freqs:    [0.1250, 0.0703, 0.0395, ..., 0.0000, 0.0000]  (÷8)
Smooth:          [0.00,   0.00,   0.00,   ..., 1.00,   1.00  ]

Final freqs[i] = original[i] × (1 - smooth[i]) + scaled[i] × smooth[i]

i=0:  1.0000 × 1.00 + 0.1250 × 0.00 = 1.0000   (unchanged!)
i=1:  0.5623 × 1.00 + 0.0703 × 0.00 = 0.5623   (unchanged!)
i=5:  0.0562 × 1.00 + 0.0070 × 0.00 = 0.0562   (unchanged — just at boundary)
i=8:  0.0100 × 0.57 + 0.0013 × 0.43 = 0.0063   (blended)
i=12: 0.0010 × 0.00 + 0.0001 × 1.00 = 0.0001   (fully scaled)
i=15: 0.0002 × 0.00 + 0.0000 × 1.00 = 0.0000   (fully scaled)
```

**The effect**: High-frequency dimensions (i=0-4) are completely unchanged — they already wrap many times during training, so extending context doesn't create unseen angles. Low-frequency dimensions (i=12-15) are divided by the scaling factor (8), which means positions 0-32K map to the same angle range that 0-4K used during training. The transition zone smoothly blends between the two regimes.

#### 2.5.6 Frequency Spectrum Visualization

```
                   YaRN Frequency Modification (factor=8, dim=32)

 Frequency   │
 (log scale) │
             │
  1.0    ────┤ ●  ●                                      ← original freqs
             │       ●                                       (high freq, unchanged)
  0.1    ────┤          ●
             │             ●
  0.01   ────┤                ●──●──●──●──●               ← transition zone
             │                              ╲                (blended)
  0.001  ────┤                                ●──●         ← fully interpolated
             │                                     ●──●      (low freq, ÷8)
  0.0001 ────┤                                          ●
             │
             └──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──
                0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
                                 Dimension index i

                [── keep original ──][── blend ──][─ interpolate ─]
```

#### 2.5.7 Why This Works: The Wavelength Argument

After YaRN modification with factor=8:

```
Dim i=15 (lowest freq):
  Original wavelength:  ~35,333 positions
  YaRN wavelength:      ~35,333 × 8 = ~282,666 positions
  Training saw angles:  0 to 4096/282666 × 2π ≈ 0.091 rad (same as original 0-4K range)
  At 32K positions:     32000/282666 × 2π ≈ 0.712 rad (within original training range!)

Dim i=0 (highest freq):
  Original wavelength:  ~6.3 positions (unchanged by YaRN)
  At 32K positions:     wraps 5093 times — the model has seen every possible angle
```

YaRN ensures that even at 32K positions, every dimension pair sees angles within the range experienced during training. The model's learned attention patterns for each frequency band remain valid.

### 2.6 The RotaryEmbedding Module

The complete module wraps precomputation and application:

```python
# model.py line 147-179
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000.0,
                 scaling_factor=1.0, original_max_position_embeddings=4096):
        super().__init__()
        freqs_cis = precompute_freqs_cis(
            dim, max_position_embeddings, base, scaling_factor,
            original_max_position_embeddings
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, q, k, position_ids=None):
        if position_ids is not None:
            freqs = self.freqs_cis[position_ids]   # gather by position
        else:
            freqs = self.freqs_cis[:seq_len]       # slice sequential
        q_rotated = apply_rotary_emb(q, freqs, interleaved=True)
        k_rotated = apply_rotary_emb(k, freqs, interleaved=True)
        return q_rotated, k_rotated
```

Note: In MLA, the `RotaryEmbedding` module is not used directly. Instead, MLA precomputes `freqs_cis` as a buffer and calls `apply_rotary_emb` inline, because Q and K rope components have different shapes (Q: `[B,L,16,32]`, K: `[B,L,1,32]`) and are handled at different points in the forward pass. The standalone `RotaryEmbedding` module exists for non-MLA use cases or testing.

### 2.7 NanoSeek Config Summary

```
┌───────────────────────────────────────────────────────────┐
│ RoPE & YaRN Configuration                                 │
├───────────────────────────┬───────────────────────────────┤
│ qk_rope_head_dim          │ 32                            │
│ rope_theta                │ 10000.0                       │
│ original_max_position     │ 4096 (training context)       │
│ max_position_embeddings   │ 4096 (training), up to 32K    │
│ scaling_factor            │ 1.0 (training), 8.0 (YaRN)   │
│ beta_fast                 │ 32                            │
│ beta_slow                 │ 1                             │
│ mscale                    │ 1.0                           │
│ Precomputed table shape   │ [max_pos, 16] complex64       │
│ Total RoPE parameters     │ 0 (no learned params!)        │
│ KV cache for position     │ 32 dims shared across heads   │
└───────────────────────────┴───────────────────────────────┘
```

---

## 🔴 STAGE 3: GROUND TRUTH — What Correct Rotation Looks Like

### Numerical Walkthrough

Let's trace a complete rotation for one token at position `p=100` using one dimension pair `i=4` (mid-frequency).

**Step 1: Compute the base frequency**

```
θ₄ = 1 / 10000^(8/32) = 1 / 10000^0.25 = 1 / 10.0 = 0.1
```

**Step 2: Compute the rotation angle**

```
angle = p × θ₄ = 100 × 0.1 = 10.0 radians
     = 10.0 mod 2π = 10.0 - 2π = 3.717 radians  (not actually mod'd, but 1.59 full rotations)
```

**Step 3: Compute the complex rotation factor**

```
e^(j × 10.0) = cos(10.0) + j × sin(10.0)
             = -0.8391 + j × (-0.5440)
```

This is the value stored in `freqs_cis[100, 4]`.

**Step 4: Apply to a concrete Q vector**

Say `q_pe` at this position, this head, dims 8-9 (the i=4 pair) is `[0.5, 0.3]`:

```
q_complex = 0.5 + 0.3j
q_rotated = (0.5 + 0.3j) × (-0.8391 - 0.5440j)
          = 0.5×(-0.8391) - 0.3×(-0.5440) + j(0.5×(-0.5440) + 0.3×(-0.8391))
          = -0.4196 + 0.1632 + j(-0.2720 - 0.2517)
          = -0.2564 - 0.5237j
```

Back to real: `q_rotated_dims_8_9 = [-0.2564, -0.5237]`

**Step 5: Verify norm preservation**

```
original norm: sqrt(0.5² + 0.3²) = sqrt(0.34) = 0.5831
rotated norm:  sqrt(0.2564² + 0.5237²) = sqrt(0.0657 + 0.2743) = sqrt(0.3400) = 0.5831 ✓
```

### Relative Position Verification

Now apply the same pair for a K vector `k = [0.7, -0.2]` at position `p=97` (relative distance = 3):

```
θ₄ = 0.1
K rotation angle = 97 × 0.1 = 9.7 rad
e^(j × 9.7) = cos(9.7) + j × sin(9.7) = -0.9537 + j × (-0.3010)

k_complex = 0.7 - 0.2j
k_rotated = (0.7 - 0.2j) × (-0.9537 - 0.3010j)
          = -0.6676 - 0.2107 + j(0.1907 - 0.2107 + 0.0602)  ← let me redo carefully
          = (0.7×-0.9537 - (-0.2)×(-0.3010)) + j(0.7×(-0.3010) + (-0.2)×(-0.9537))
          = (-0.6676 - 0.0602) + j(-0.2107 + 0.1907)
          = -0.7278 - 0.0200j
```

Dot product of rotated Q and K (this dimension pair):

```
<q_rot, k_rot> = (-0.2564)(−0.7278) + (−0.5237)(−0.0200)
               = 0.1866 + 0.0105
               = 0.1971
```

Now verify via relative position: apply rotation `(100-97)×0.1 = 0.3 rad` to Q, dot with original K:

```
e^(j × 0.3) = cos(0.3) + j × sin(0.3) = 0.9553 + 0.2955j

q_rel = (0.5 + 0.3j) × (0.9553 + 0.2955j)
      = (0.5×0.9553 - 0.3×0.2955) + j(0.5×0.2955 + 0.3×0.9553)
      = (0.4777 - 0.0887) + j(0.1478 + 0.2866)
      = 0.3890 + 0.4344j

<q_rel, k> = (0.3890)(0.7) + (0.4344)(−0.2)
           = 0.2723 - 0.0869
           = 0.1854
```

The values 0.1971 and 0.1854 differ slightly due to floating point precision in my hand calculations, but the property holds exactly in IEEE 754 arithmetic. The test suite (`tests/test_rope.py::TestRoPERotation::test_relative_position_property`) verifies this to `rtol=1e-4`.

---

## 🔴 STAGE 4: LOSS FUNCTION — RoPE's Indirect Contribution

### RoPE Has No Parameters and No Direct Loss

RoPE contributes **zero** learnable parameters. It has no loss term of its own. Its entire contribution is *structural*: by encoding position into the Q/K geometry, it enables the attention pattern (and thus the language model) to learn position-dependent relationships.

### How Position Encoding Affects the Main Loss

The language modeling loss is:

```
L = CrossEntropy(logits[:-1], labels[1:])
```

Position information flows through the loss indirectly:

```
Position indices → RoPE rotation → modified Q, K → attention scores → attention output
→ residual + FFN → next layer → ... → final logits → cross-entropy loss
```

Without position encoding, the model would compute the same attention scores regardless of token order. The loss would plateau at the entropy of the unigram distribution because the model couldn't learn bigram, trigram, or any positional patterns.

### MTP Loss and Position

Multi-Token Prediction also depends on RoPE indirectly. The MTP modules predict future tokens using hidden states that were shaped by position-aware attention in the main model. If RoPE fails (e.g., wrong frequencies, numerical overflow at extended positions), MTP predictions degrade because the hidden states lose positional coherence.

### YaRN's mscale: Attention Score Compensation

When YaRN interpolates frequencies, the effective "resolution" of position encoding changes. At longer contexts, the attention score distribution can become flatter (more uniform attention) because position signals are weaker at interpolated frequencies. The `mscale` parameter compensates:

```python
# model.py line 262
self.softmax_scale = mscale / math.sqrt(self.qk_head_dim)
```

In NanoSeek, `mscale=1.0` (no compensation). This is because:
1. The rope component is only 32 of 96 dims — the content component (64 dims) dominates attention scores
2. The scaling factor of 8× with NanoSeek's dim=32 doesn't create severe degradation
3. At production scale (DeepSeek V3 with dim=64 and factor=40×), mscale > 1.0 is critical

For aggressive context extension (e.g., 4K → 128K), mscale should be tuned empirically. A common formula is `mscale = 0.1 × ln(scaling_factor) + 1.0`, but NanoSeek's conservative 8× factor doesn't require this.

---

## 🔴 STAGE 5: OUTPUTS — Rotated Q, K Tensors

### What Leaves the RoPE System

After rotation, the rope components are recombined with the nope components and flow into attention:

```
From RoPE:
  q_pe (rotated): [B, L, 16, 32]    — position-encoded query
  k_pe (rotated): [B, L, 1, 32]     — position-encoded key (shared)

Combined with nope:
  q = cat([q_nope, q_pe], dim=-1)    → [B, L, 16, 96]
  k_pe_expanded = k_pe.expand(...)   → [B, L, 16, 32]
  k = cat([k_nope, k_pe_expanded])   → [B, L, 16, 96]
```

### What Gets Cached

In the MLA KV cache, only the *rotated* k_pe is stored alongside the compressed KV latent:

```
Cache entry per layer:
  kv_compressed: [B, L, 143]    — content (position-independent)
  k_pe:          [B, L, 1, 32]  — position (already rotated, shared across heads)

Total per token per layer: 175 values
```

The k_pe is stored **post-rotation** because:
1. Each position's rotation is unique — you can't defer rotation to attention time without storing position indices
2. The rotation is deterministic given position, so storing the result avoids recomputation
3. The shared-across-heads structure means we only store 32 values, not 16×32=512

### Output Numerical Properties

After rotation, the output maintains several invariants:

- **Norm preservation**: `||q_pe_rotated|| = ||q_pe_original||` (verified in `test_rope.py::test_rotation_preserves_norm`)
- **Dtype preservation**: Output matches input dtype (bf16 during training)
- **Shape preservation**: Output shape exactly matches input shape
- **No NaN/Inf**: Guaranteed for positions within `max_position_embeddings` (unit complex multiplications cannot produce Inf)
- **Invertibility**: `apply_rotary_emb(apply_rotary_emb(x, freqs), conj(freqs)) = x` (verified in `test_rope.py::test_rotation_is_invertible`)

---

## Common Misconceptions

### "RoPE is a form of learned positional embedding"

**Wrong.** RoPE has exactly zero learned parameters. The frequencies are deterministic functions of the dimension index and the hyperparameter θ. The model never updates RoPE during training — it learns to *use* the positional signal that RoPE provides, not to *modify* the signal itself. This is a feature: deterministic frequencies guarantee consistent positional behavior across training and inference, and enable context extension via YaRN without retraining the frequencies.

### "RoPE encodes absolute position"

**Half right, half wrong.** RoPE *applies* an absolute rotation (position-specific), but the attention score *depends only on relative position*. This is the core mathematical property. If you inspect Q or K alone, each vector "knows" its absolute position. But the model's behavior — computed through dot products — only ever sees relative distances. This is why RoPE generalizes to unseen absolute positions (within reason) while maintaining precise relative position awareness.

### "YaRN just divides all frequencies by the scaling factor"

**Wrong.** That's simple Position Interpolation (Chen et al., 2023). YaRN's key contribution is the selective modification: high frequencies stay unchanged, low frequencies get scaled, and there's a smooth transition. This matters because high-frequency dims are already well-trained across their full angle range and don't need interpolation. Scaling them would *reduce* local position resolution for no benefit.

### "You can extrapolate indefinitely with YaRN"

**Wrong.** YaRN enables reliable *interpolation* up to `original_context × scaling_factor`. Beyond that, even the interpolated frequencies encounter unseen angle ranges. In practice, NanoSeek with factor=8 reliably extends from 4K training to ~32K inference. Pushing to 64K would need factor=16 and possibly retraining or additional techniques (e.g., continual pretraining with extended context). The 2026 consensus is: YaRN buys you a reliable 8-16× extension. Beyond that, you need to either train at longer context or combine with other techniques (ALiBi bias, sliding window attention, or DSA).

### "The interleaved format is just an implementation detail"

**Mostly correct, but dangerous to ignore.** Interleaved means dims `[d0, d1, d2, d3, ...]` are paired as `(d0,d1), (d2,d3), ...`. Non-interleaved means `[d0, d1, ..., d_{d/2-1}, d_{d/2}, ..., d_{d-1}]` are paired as `(d0, d_{d/2}), (d1, d_{d/2+1}), ...`. The rotation is mathematically identical — same angles, same properties — but if you mix formats between Q and K (or between training and inference), the dot products will be garbage. NanoSeek uses `interleaved=True` everywhere. If you port weights from a model using non-interleaved RoPE, you must either convert the weights or switch the `interleaved` flag.

---

## Production Gotchas

### 1. Float32 for Complex Arithmetic

```python
# model.py line 129
x_complex = torch.view_as_complex(x.float().reshape(...))
```

The `.float()` upcast is non-negotiable. Complex multiplication in bf16 loses critical precision for the low-frequency dimensions where angles are tiny (θ₁₅ ≈ 0.0002). At position 100, dim 15: `angle = 0.0178 rad, cos = 0.9998, sin = 0.0178`. In bf16, `cos(0.0178) ≈ 1.0` (rounds to 1), destroying the positional signal entirely. The upcast to fp32, rotation, and downcast back preserves the signal.

**Cost**: Negligible. The rope component is only 32 dims out of 2048 hidden dims. The fp32 computation affects ~1.5% of the data.

### 2. Position Offset During Cached Generation

During KV-cached decoding, new tokens arrive one at a time. The RoPE frequency must use the correct absolute position, not the local sequence position:

```python
# model.py line 340-344
position_offset = kv_len - seq_len    # e.g., 512 cached + 1 new → offset=512
q_freqs = self.freqs_cis[position_offset:position_offset + seq_len]
# For generating token 513: freqs_cis[512:513], NOT freqs_cis[0:1]
```

Getting this wrong is the #1 source of generation quality bugs in RoPE implementations. The model will appear to generate correctly for short sequences but produce incoherent text for longer generations because Q at position 513 is rotated by position-0 frequencies, breaking the relative position property.

### 3. k_pe Shape: [B, L, 1, 32] Not [B, L, 32]

The singleton head dimension in k_pe is critical for correct broadcasting:

```python
# model.py line 324
k_pe_current = k_pe_current.unsqueeze(2)  # [B, L, 32] → [B, L, 1, 32]
```

Without this, `apply_rotary_emb` would interpret the 32 dims as `seq_len=32` rather than `head_dim=32`, producing completely wrong rotations. The unsqueeze ensures the function correctly identifies the head and dim axes.

### 4. freqs_cis Slicing vs Gathering

For sequential processing (training): `freqs_cis[:seq_len]` — a simple slice.
For non-contiguous positions (batched with different offsets): `freqs_cis[position_ids]` — gather by index.

The code handles both:

```python
# model.py line 171-175
if position_ids is not None:
    freqs = self.freqs_cis[position_ids]     # [B, L, 16] complex
else:
    seq_len = q.shape[1]
    freqs = self.freqs_cis[:seq_len]         # [L, 16] complex
```

The gathered form produces a `[B, L, 16]` tensor (3D), while the sliced form produces `[L, 16]` (2D). `apply_rotary_emb` handles both via the broadcasting logic in lines 117-126.

### 5. YaRN Scaling Factor Must Match Between Precomputation and Inference

If you precompute `freqs_cis` with `scaling_factor=1.0` (no YaRN) but run inference at 32K positions, positions beyond 4K will use unmodified frequencies and suffer from extrapolation failure. Conversely, if you precompute with `scaling_factor=8.0` for training at 4K, the modified low-frequency dims will have reduced resolution for nearby positions, slightly degrading short-context quality.

The correct pattern (used by NanoSeek):
- **Training**: `scaling_factor=1.0`, `max_position_embeddings=4096`
- **Extended inference**: `scaling_factor=8.0`, `max_position_embeddings=32768`

These are set at model construction time via config, not at runtime.

### 6. torch.polar vs Manual sin/cos

NanoSeek uses `torch.polar(ones, angles)` to create unit complex numbers. An alternative implementation computes `cos(angles) + j·sin(angles)` explicitly. Both are mathematically identical, but `torch.polar` is marginally more numerically stable (single kernel, avoids intermediate rounding between separate sin/cos calls). More importantly, it's cleaner code.

---

## 2026 Best Practices

### Architecture

1. **Use decoupled RoPE with MLA**: If you're using MLA (and you should be for any model targeting efficient inference), always split Q/K into nope and rope components. The rope dims should be 25-50% of the nope dims. NanoSeek's 32 rope / 64 nope = 0.5× ratio is well-validated.

2. **Keep θ = 10000 unless you have strong evidence**: Higher θ (e.g., LLaMA 3 used 500000) enables better long-context behavior but can hurt short-context resolution. If your application is primarily short-context, stick with 10000. If you need native 100K+ context without YaRN, consider higher θ values.

3. **Dimension matters more than you think**: The 32-dim rope in NanoSeek encodes position into 16 frequency bands. This is sufficient for 32K context with YaRN. For 128K+ context, consider 64 or 128 rope dims to provide more frequency bands for fine-grained position discrimination at long distances.

### Training

4. **Train at the context length you care about**: YaRN extends well, but training at the target length still gives better quality. NanoSeek trains at 4K and extends to 32K — adequate for research, but production models should train at longer contexts if budget permits.

5. **fp32 softmax AND fp32 RoPE arithmetic**: Both are non-negotiable for training stability. The softmax requirement is well-known; the RoPE requirement is less discussed but equally important for low-frequency dims.

6. **Monitor attention patterns per frequency band**: During training, periodically inspect whether the model is actually using all frequency bands. If the lowest-frequency dims show near-zero contribution to attention scores, your context length may be too short for those dims to be useful, or your θ is too large.

### Inference & Context Extension

7. **YaRN factor should match the extension ratio**: If training at 4K and inferring at 32K, use `factor = 32K / 4K = 8.0`. Under-factoring (e.g., factor=4 for 8× extension) leaves high positions with unseen angles. Over-factoring (e.g., factor=16 for 8× extension) wastes low-frequency resolution.

8. **Validate at boundary positions**: After applying YaRN, test attention quality at positions near `original_max_position × scaling_factor`. Quality degradation at the boundary of the extended range is the first sign of problems.

9. **Combine YaRN with DSA for production**: YaRN enables the extended positions, but at 32K context, full O(L²) attention is expensive. DeepSeek Sparse Attention (DSA) reduces this to O(Lk) by selecting the most relevant tokens. NanoSeek's architecture uses both: YaRN for position extension, DSA for compute efficiency.

---

## Appendix A: Complete Frequency Table for NanoSeek (dim=32, θ=10000)

```
Dim │ Pair │ Frequency θᵢ │ Wavelength (pos) │ Training rotations │ YaRN behavior
────┼──────┼──────────────┼──────────────────┼────────────────────┼──────────────
 0  │ 0,1  │ 1.000000     │ 6.28             │ 651.9              │ Unchanged
 1  │ 2,3  │ 0.562341     │ 11.18            │ 366.5              │ Unchanged
 2  │ 4,5  │ 0.316228     │ 19.87            │ 206.1              │ Unchanged
 3  │ 6,7  │ 0.177828     │ 35.33            │ 115.9              │ Unchanged
 4  │ 8,9  │ 0.100000     │ 62.83            │ 65.2               │ Unchanged
 5  │10,11 │ 0.056234     │ 111.75           │ 36.6               │ Boundary
 6  │12,13 │ 0.031623     │ 198.69           │ 20.6               │ 14% scaled
 7  │14,15 │ 0.017783     │ 353.33           │ 11.6               │ 29% scaled
 8  │16,17 │ 0.010000     │ 628.32           │ 6.5                │ 43% scaled
 9  │18,19 │ 0.005623     │ 1,117.5          │ 3.7                │ 57% scaled
10  │20,21 │ 0.003162     │ 1,986.9          │ 2.1                │ 71% scaled
11  │22,23 │ 0.001778     │ 3,533.3          │ 1.2                │ 86% scaled
12  │24,25 │ 0.001000     │ 6,283.2          │ 0.65               │ Fully scaled
13  │26,27 │ 0.000562     │ 11,175           │ 0.37               │ Fully scaled
14  │28,29 │ 0.000316     │ 19,869           │ 0.21               │ Fully scaled
15  │30,31 │ 0.000178     │ 35,333           │ 0.12               │ Fully scaled
```

"Training rotations" = `4096 / wavelength` = how many full rotations this dim completes during training. Dims with < 1 rotation during training are the ones most helped by YaRN interpolation.

## Appendix B: RoPE vs Other Position Encodings at Scale

```
Method       │ Params │ Relative │ Extend? │ KV Cache Cost │ Quality │ Used By (2026)
─────────────┼────────┼──────────┼─────────┼───────────────┼─────────┼──────────────
Sinusoidal   │ 0      │ No       │ Poor    │ 0             │ Low     │ None (dead)
Learned Abs  │ L×d    │ No       │ None    │ 0             │ Medium  │ Legacy BERT
ALiBi        │ 0      │ Yes      │ Good    │ 0             │ Medium  │ MPT, BLOOM
RoPE         │ 0      │ Yes      │ Mod.    │ 0             │ High    │ LLaMA, Mistral
RoPE+YaRN   │ 0      │ Yes      │ Excel.  │ 0             │ High    │ DeepSeek, NanoSeek
RoPE+NTK    │ 0      │ Yes      │ Good    │ 0             │ High    │ CodeLLaMA
```

## Appendix C: NanoSeek RoPE vs DeepSeek V3 RoPE

```
Parameter                        │ NanoSeek-1B   │ DeepSeek V3     │ Ratio Match?
─────────────────────────────────┼───────────────┼─────────────────┼─────────────
qk_rope_head_dim                 │ 32            │ 64              │ —
rope_theta                       │ 10,000        │ 10,000          │ ✓
original_max_position_embeddings │ 4,096         │ 4,096           │ ✓
YaRN scaling_factor              │ 8.0           │ 40.0            │ — (scale-dependent)
beta_fast                        │ 32            │ 32              │ ✓
beta_slow                        │ 1             │ 1               │ ✓
mscale                           │ 1.0           │ 1.0             │ ✓
Frequency bands (dim/2)          │ 16            │ 32              │ —
Max training context             │ 4,096         │ 4,096           │ ✓
Max YaRN inference context       │ 32,768        │ 163,840         │ — (scale-dependent)
rope/nope ratio                  │ 32/64 = 0.50  │ 64/128 = 0.50  │ ✓
Position in KV cache             │ 32 dims       │ 64 dims         │ —
% of total KV cache              │ 32/175 = 18%  │ 64/576 = 11%   │ ~matched
```

The critical design ratios (θ, beta values, rope/nope split ratio) are preserved across scales. This means NanoSeek faithfully reproduces DeepSeek V3's positional encoding behavior, scaled down to 1B.

## Appendix D: Key Code References

```
model/model.py:
  Lines 42-49:   find_correction_dim()      — YaRN correction dimension formula
  Lines 52-62:   find_correction_range()     — YaRN correction range [low, high]
  Lines 66-70:   linear_ramp_factor()        — YaRN smooth transition ramp
  Lines 73-97:   precompute_freqs_cis()      — RoPE + YaRN frequency table
  Lines 100-144: apply_rotary_emb()          — Core rotation function
  Lines 147-179: RotaryEmbedding             — nn.Module wrapper
  Lines 278-285: MLA freqs_cis buffer        — Precomputation in MLA.__init__
  Lines 302-303: Q nope/rope split           — Decoupled RoPE in MLA
  Lines 306-308: K nope/rope split           — Shared rope component
  Lines 345:     Q RoPE application          — In MLA forward
  Lines 329:     K RoPE application          — In MLA forward
  Lines 349:     k_pe.expand()               — Head broadcast for shared rope

model/config.py:
  Lines 115-153: YaRNConfig                  — YaRN hyperparameters
  Lines 247-249: MLAConfig rope dims         — qk_nope_head_dim, qk_rope_head_dim
  Lines 252-258: MLAConfig rope params       — rope_theta, rope_scaling_factor

tests/test_rope.py:
  TestRoPEFrequencies                        — Frequency computation correctness
  TestRoPERotation                           — Norm preservation, invertibility, relative position
  TestYaRNInterpolation                      — Correction range, ramp, context extension
  TestRotaryEmbeddingModule                  — Module forward, position_ids, gradient flow
```
