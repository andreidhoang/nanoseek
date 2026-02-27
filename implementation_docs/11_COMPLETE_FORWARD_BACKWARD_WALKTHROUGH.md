# 11. Complete Forward + Backward Pass Walkthrough

## Engineer's Thinking Process

> "I'm about to trace a single training iteration through the entire NanoSeek-1B model.
> My input is a batch of 2 sequences, each 512 tokens long. I need to track every
> tensor shape at every operation, understand every computation, and see exactly where
> gradients flow back. This is a Phase 1 dense-attention training step — DSA is
> disabled, YaRN is off, and we're operating at the native 4096 max context with a
> 512-token window.
>
> I'll start from `input_ids`, flow through embedding, 16 decoder layers (2 dense +
> 14 MoE), final norm, the LM head, loss computation including MTP and auxiliary
> losses, and then trace the backward pass through the entire graph.
>
> Every shape, every weight matrix, every computation — nothing is skipped."

---

## Section 1: The Setup

### 1.1 Configuration Snapshot

All values come from `get_nanoseek_config()` in `model/config.py` (line 847):

```
hidden_size       = 2048       num_heads        = 16
num_layers        = 16         vocab_size       = 65536
q_lora_rank       = 430        kv_lora_rank     = 143
qk_nope_head_dim  = 64         qk_rope_head_dim = 32
v_head_dim        = 64         intermediate_size = 5243
moe_intermediate  = 768        n_routed_experts = 64
num_experts_per_tok = 8        n_shared_experts = 2
n_group           = 8          topk_group       = 4
first_k_dense     = 2          num_mtp_modules  = 1
rms_norm_eps      = 1e-6       dtype            = bfloat16
scoring_func      = "sigmoid"  routed_scaling   = 2.5
```

### 1.2 Concrete Inputs

```python
batch_size = 2
seq_len    = 512
device     = "cuda"
dtype      = torch.bfloat16

# Token IDs — two sequences of 512 tokens each
input_ids = torch.tensor([
    [   1, 4523, 8192, 331, ...,  7720],  # sequence 0 (512 tokens)
    [   1, 9102, 2048, 512, ..., 33001],  # sequence 1 (512 tokens)
], dtype=torch.long, device=device)                   # shape: [2, 512]

# Labels — the teacher signal (same shape, shifted internally during loss)
labels = torch.tensor([
    [4523, 8192, 331, 1029, ...,    2],  # sequence 0 labels
    [9102, 2048, 512, 7843, ...,    2],  # sequence 1 labels
], dtype=torch.long, device=device)                   # shape: [2, 512]
```

### 1.3 Model Instantiation

From `NanoSeekModel.__init__()` (line 1483 in `model/model.py`):

```python
config = get_nanoseek_config()
model = NanoSeekModel(config)   # calls create_nanoseek() at line 1910
model = model.to(device=device, dtype=dtype)
model.train()
```

The constructor builds:
- `self.embed_tokens`: `nn.Embedding(65536, 2048)` — line 1491
- `self.layers`: `nn.ModuleList` of 16 `NanoSeekDecoderLayer` — line 1493
- `self.norm`: `RMSNorm(2048)` — line 1498
- `self.lm_head`: `nn.Linear(2048, 65536, bias=False)` — line 1499
- `self.mtp`: `MultiTokenPrediction(...)` — line 1507
- Shared embeddings set: `self.mtp.set_shared_embeddings(embed_tokens, lm_head)` — line 1517

---

## Section 2: Embedding Layer

**Entry point**: `NanoSeekModel.forward()`, line 1577.

```python
hidden_states = self.embed_tokens(input_ids)
```

### 2.1 The Computation

```
embed_tokens.weight: [65536, 2048]  (Parameter)
input_ids:           [2, 512]       (Long tensor)
─────────────────────────────────────────────────
hidden_states:       [2, 512, 2048] (bf16 tensor)
```

Each of the 1024 tokens (2 × 512) indexes into the 65536-row embedding table,
pulling out a 2048-dimensional vector. This is a pure lookup — no multiplication.

### 2.2 Parameter Count

```
embed_tokens: 65,536 × 2,048 = 134,217,728 parameters
Memory (bf16): 134,217,728 × 2 bytes = 268,435,456 bytes ≈ 256 MB

lm_head (separate, not tied — tie_word_embeddings=False):
              65,536 × 2,048 = 134,217,728 parameters  ≈ 256 MB

Combined embedding layers: 268,435,456 parameters ≈ 512 MB
```

### 2.3 Output

```
hidden_states: [2, 512, 2048]  dtype=bf16
```

Each element is a 2048-d vector. At initialization (std=0.02), typical values
are in the range [-0.06, +0.06].

---

## Section 3: Position Setup

**Lines 1579–1594** in `NanoSeekModel.forward()`.

### 3.1 Position IDs

```python
# Since past_key_values is None (first forward pass):
position_ids = torch.arange(512, device=device, dtype=torch.long)
position_ids = position_ids.unsqueeze(0).expand(2, -1)
# Result: [2, 512] = [[0, 1, 2, ..., 511],
#                      [0, 1, 2, ..., 511]]
```

### 3.2 Causal Mask

Created by `create_causal_mask()` (line 202):

```python
causal_mask = create_causal_mask(512, dtype=bf16, device=device, past_len=0)
```

```
Shape: [1, 1, 512, 512]

          k=0    k=1    k=2    ...   k=511
q=0   [  0.0,  -inf,  -inf,  ...   -inf  ]
q=1   [  0.0,   0.0,  -inf,  ...   -inf  ]
q=2   [  0.0,   0.0,   0.0,  ...   -inf  ]
 ...
q=511 [  0.0,   0.0,   0.0,  ...    0.0  ]
```

This is a standard lower-triangular causal mask. The `-inf` values ensure that
after softmax, future positions get zero attention weight. The mask is broadcast
across batch (dim 0) and heads (dim 1).

### 3.3 Gradient Checkpointing & Cache

Lines 1596–1598:
```python
# gradient_checkpointing defaults to False in __init__ (line 1521)
# use_cache=False (default), so present_key_values = None
```

---

## Section 4: Layer 0 — Dense Layer (MLA + MLP)

**Class**: `NanoSeekDecoderLayer` at line 1378.

Layer 0 is a **dense layer** (index 0 < `first_k_dense_replace=2`), meaning:
- Attention: `MultiHeadLatentAttention` (not DSA, since `sparse.enabled=False`)
- FFN: `MLP` wrapping `SwiGLUFFN` (not MoE)

### 4a. Input LayerNorm

**Line 1449**: `hidden_states = self.input_layernorm(hidden_states)`

`RMSNorm` class at line 186:

```python
def forward(self, x: Tensor) -> Tensor:      # x: [2, 512, 2048]
    dtype = x.dtype                            # bf16
    x = x.float()                              # upcast to fp32 for stability
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    #     x.pow(2):  [2, 512, 2048]
    #     .mean(-1): [2, 512, 1]     (mean over 2048 dims)
    #     + 1e-6:    [2, 512, 1]
    #     sqrt:      [2, 512, 1]
    x = x / rms                                # [2, 512, 2048] / [2, 512, 1]
    return (self.weight * x).to(dtype)         # weight: [2048], broadcast multiply
```

**Concrete example** for one token (random init values ~N(0, 0.02)):

```
Input vector (2048 dims): [0.012, -0.018, 0.005, 0.031, ...]
RMS = sqrt(mean(x²) + 1e-6) ≈ sqrt(0.0004 + 1e-6) ≈ 0.020
Normalized: [0.012/0.020, -0.018/0.020, ...] = [0.6, -0.9, ...]
Scaled: weight * normalized (weight initialized to 1.0)
```

```
Input:  [2, 512, 2048]  →  Output: [2, 512, 2048]
Params: weight [2048] = 2,048 parameters
```

### 4b. MLA Attention — The Critical Path

**Class**: `MultiHeadLatentAttention` at line 218.
**Forward**: line 287.

This is the heart of DeepSeek's KV cache compression innovation. We trace
every single operation.

#### Step 1: Q Projection Path (lines 298–302)

```python
# (a) Down-project to compressed latent space
q = self.wq_a(hidden_states)           # wq_a: Linear(2048 → 430, no bias)
#   hidden_states: [2, 512, 2048]
#   wq_a.weight:   [430, 2048]
#   Output q:      [2, 512, 430]

# (b) Normalize in compressed space
q = self.q_norm(q)                     # RMSNorm(430)
#   q: [2, 512, 430] → [2, 512, 430]

# (c) Up-project to multi-head Q
q = self.wq_b(q)                       # wq_b: Linear(430 → 1536, no bias)
#   q: [2, 512, 430] → [2, 512, 1536]
#   where 1536 = num_heads × qk_head_dim = 16 × 96

# (d) Reshape to per-head format
q = q.view(batch_size, seq_len, self.num_heads, self.qk_head_dim)
#   q: [2, 512, 1536] → [2, 512, 16, 96]

# (e) Split into non-positional and positional components
q_nope, q_pe = torch.split(q, [64, 32], dim=-1)
#   q_nope: [2, 512, 16, 64]   (content-based query)
#   q_pe:   [2, 512, 16, 32]   (position-based query)
```

**Why the two-stage projection?** The bottleneck `2048 → 430 → 1536` is a
low-rank factorization. Instead of a single `2048 → 1536` matrix (3.1M params),
we use `2048 → 430` then `430 → 1536` (0.88M + 0.66M = 1.54M params) — a 2×
parameter reduction with minimal quality loss, thanks to the RMSNorm in between
that stabilizes the compressed representation.

#### Step 2: KV Projection Path (lines 305–309)

```python
# (a) Down-project to joint compressed KV
kv = self.wkv_a(hidden_states)         # wkv_a: Linear(2048 → 175, no bias)
#   hidden_states: [2, 512, 2048]
#   wkv_a.weight:  [175, 2048]
#   Output kv:     [2, 512, 175]
#   where 175 = kv_lora_rank + qk_rope_head_dim = 143 + 32

# (b) Split compressed KV and positional key
kv_compressed, k_pe_current = torch.split(kv, [143, 32], dim=-1)
#   kv_compressed: [2, 512, 143]   (will expand to K_nope + V)
#   k_pe_current:  [2, 512, 32]    (positional key component, SHARED across heads!)

# (c) Normalize compressed KV
kv_compressed = self.kv_norm(kv_compressed)    # RMSNorm(143)
#   kv_compressed: [2, 512, 143] → [2, 512, 143]
```

**Key insight**: The KV cache only needs to store `kv_compressed [143]` and
`k_pe [32]` = **175 values per token per layer**. Standard MHA stores
`2 × 16 × 128 = 4096` values. Compression ratio: `4096 / 175 ≈ 23.4×`.

#### Step 3: KV Expansion (lines 335–337)

```python
# Since past_key_value is None, we skip cache concatenation.
# (lines 312-329 handle cache; we take the else branch at line 323)

# Add sequence dim for RoPE compatibility
k_pe_current = k_pe_current.unsqueeze(2).contiguous()
#   k_pe_current: [2, 512, 32] → [2, 512, 1, 32]

# Expand compressed KV to full multi-head K and V
kv_expanded = self.wkv_b(kv_compressed)   # wkv_b: Linear(143 → 2048, no bias)
#   kv_compressed: [2, 512, 143]
#   wkv_b.weight:  [2048, 143]
#   kv_expanded:   [2, 512, 2048]
#   where 2048 = num_heads × (qk_nope_head_dim + v_head_dim) = 16 × (64 + 64)

# Reshape to per-head
kv_expanded = kv_expanded.view(batch_size, kv_len, self.num_heads,
                                self.qk_nope_head_dim + self.v_head_dim)
#   kv_expanded: [2, 512, 2048] → [2, 512, 16, 128]

# Split into key (non-positional) and value
k_nope, v = torch.split(kv_expanded, [64, 64], dim=-1)
#   k_nope: [2, 512, 16, 64]   (content-based key)
#   v:      [2, 512, 16, 64]   (value)
```

#### Step 4: RoPE Application (lines 325–345)

```python
# RoPE frequencies precomputed in __init__ (line 278):
# freqs_cis = precompute_freqs_cis(32, 4096, 10000.0, ...)
# freqs_cis shape: [4096, 16]  (complex64; 16 = qk_rope_head_dim/2 = 32/2)

# Get frequencies for current positions
current_freqs = self.freqs_cis[:seq_len]    # [512, 16] (complex)

# Apply RoPE to key positional embedding
k_pe = apply_rotary_emb(k_pe_current, current_freqs, interleaved=True)
#   k_pe_current: [2, 512, 1, 32]
#   → view as complex: [2, 512, 1, 16] (complex64)
#   → multiply by freqs_cis: [2, 512, 1, 16]
#   → view as real: [2, 512, 1, 32]
#   k_pe: [2, 512, 1, 32]

# Apply RoPE to query positional embedding (line 345)
q_freqs = self.freqs_cis[:seq_len]          # [512, 16]
q_pe = apply_rotary_emb(q_pe, q_freqs, interleaved=True)
#   q_pe: [2, 512, 16, 32] → [2, 512, 16, 32]
```

**How RoPE works** (`apply_rotary_emb`, line 100): The 32-dim vector is viewed
as 16 complex numbers. Each is multiplied by `e^(i·θ·pos)` where θ depends on
the dimension index. This encodes absolute position in a way that relative
position emerges naturally in the dot product.

#### Step 5: Assemble Full Q and K (lines 348–350)

```python
# Concatenate non-positional and positional components
q = torch.cat([q_nope, q_pe], dim=-1)
#   q_nope: [2, 512, 16, 64]  +  q_pe: [2, 512, 16, 32]
#   q: [2, 512, 16, 96]       (qk_head_dim = 64 + 32 = 96)

# Expand k_pe across all heads (it's SHARED — key MLA innovation!)
k_pe_expanded = k_pe.expand(-1, -1, self.num_heads, -1)
#   k_pe: [2, 512, 1, 32] → k_pe_expanded: [2, 512, 16, 32]

k = torch.cat([k_nope, k_pe_expanded], dim=-1)
#   k_nope: [2, 512, 16, 64]  +  k_pe_expanded: [2, 512, 16, 32]
#   k: [2, 512, 16, 96]
```

#### Step 6: Attention Computation (lines 352–367)

```python
# Transpose to [batch, heads, seq, dim] for batched matmul
q = q.transpose(1, 2)          # [2, 16, 512, 96]
k = k.transpose(1, 2)          # [2, 16, 512, 96]
v = v.transpose(1, 2)          # [2, 16, 512, 64]

# Scaled dot-product attention
# softmax_scale = mscale / sqrt(qk_head_dim) = 1.0 / sqrt(96) ≈ 0.10206
attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.softmax_scale
#   q: [2, 16, 512, 96]  @  k^T: [2, 16, 96, 512]
#   → attn_weights: [2, 16, 512, 512]
#   × 0.10206
#   → attn_weights: [2, 16, 512, 512]

# Apply causal mask
attn_weights = attn_weights + attention_mask
#   attention_mask: [1, 1, 512, 512] (broadcast)
#   Positions above diagonal become -inf

# Softmax in fp32 for numerical stability, then cast back
attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
#   attn_weights: [2, 16, 512, 512] (fp32 → bf16)
#   Each row sums to 1.0; future positions are 0.0

# Weighted sum of values
attn_output = torch.matmul(attn_weights, v)
#   attn_weights: [2, 16, 512, 512]  @  v: [2, 16, 512, 64]
#   → attn_output: [2, 16, 512, 64]
```

**FLOPs for attention**:
- Q@K^T: `2 × 16 × 512 × 96 × 512 = 805M` multiply-adds
- Attn@V: `2 × 16 × 512 × 512 × 64 = 537M` multiply-adds
- Total attention FLOPs per layer: ~1.34 GFLOPs

#### Step 7: Output Projection (lines 368–371)

```python
# Transpose back and reshape
attn_output = attn_output.transpose(1, 2).contiguous()
#   [2, 16, 512, 64] → [2, 512, 16, 64]
attn_output = attn_output.view(batch_size, seq_len, -1)
#   [2, 512, 16, 64] → [2, 512, 1024]   (16 × 64 = 1024)

# Project back to hidden dimension
output = self.wo(attn_output)          # wo: Linear(1024 → 2048, no bias)
#   [2, 512, 1024] → [2, 512, 2048]
```

#### MLA Parameter Summary for One Layer

| Weight     | Shape          | Parameters   |
|------------|----------------|-------------|
| `wq_a`    | [430, 2048]    | 880,640     |
| `q_norm`  | [430]          | 430         |
| `wq_b`    | [1536, 430]    | 660,480     |
| `wkv_a`   | [175, 2048]    | 358,400     |
| `kv_norm` | [143]          | 143         |
| `wkv_b`   | [2048, 143]    | 292,864     |
| `wo`      | [2048, 1024]   | 2,097,152   |
| **Total** |                | **4,290,109** |

### 4c. First Residual Connection

**Line 1465**: `hidden_states = residual + hidden_states`

```
residual:      [2, 512, 2048]  (saved before layernorm at line 1448)
attn_output:   [2, 512, 2048]  (from MLA)
────────────────────────────────
hidden_states: [2, 512, 2048]  (element-wise addition)
```

### 4d. Post-Attention LayerNorm + Dense MLP

**Lines 1467–1469**:

```python
residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)  # RMSNorm(2048)
hidden_states, ffn_aux = self.ffn(hidden_states)              # MLP → SwiGLUFFN
```

The `MLP` class (line 470) wraps `SwiGLUFFN` (line 445):

```python
# Inside SwiGLUFFN.forward() (line 462):
def forward(self, x: Tensor) -> Tensor:
    return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

Traced step by step:

```python
x = hidden_states                      # [2, 512, 2048]

gate = self.gate_proj(x)              # Linear(2048 → 5243, no bias)
#   [2, 512, 2048] → [2, 512, 5243]

up = self.up_proj(x)                  # Linear(2048 → 5243, no bias)
#   [2, 512, 2048] → [2, 512, 5243]

gate = F.silu(gate)                   # SiLU(x) = x × σ(x)
#   [2, 512, 5243] → [2, 512, 5243]

hidden = gate * up                    # element-wise
#   [2, 512, 5243]

output = self.down_proj(hidden)       # Linear(5243 → 2048, no bias)
#   [2, 512, 5243] → [2, 512, 2048]
```

**Dense MLP Parameters**:

| Weight       | Shape          | Parameters   |
|-------------|----------------|-------------|
| `gate_proj` | [5243, 2048]   | 10,737,664  |
| `up_proj`   | [5243, 2048]   | 10,737,664  |
| `down_proj` | [2048, 5243]   | 10,737,664  |
| **Total**   |                | **32,212,992** |

`MLP.forward()` returns `(output, {})` — empty aux_data dict since this is a dense layer.

### 4e. Second Residual Connection

**Line 1471**: `hidden_states = residual + hidden_states`

```
residual:      [2, 512, 2048]
mlp_output:    [2, 512, 2048]
────────────────────────────────
hidden_states: [2, 512, 2048]
```

### 4f. Layer 0 Complete Output

```
Input:  [2, 512, 2048]   →   Output: [2, 512, 2048]
Total params: 4,290,109 (MLA) + 4,096 (norms) + 32,212,992 (MLP) = 36,507,197
```

---

## Section 5: Layer 1 — Second Dense Layer

Layer 1 is structurally identical to Layer 0 (index 1 < `first_k_dense_replace=2`):
- Same MLA architecture with identical weight shapes
- Same dense MLP with SwiGLU (2048 → 5243 → 2048)
- Same residual connections and layer norms

The only difference is the learned weight values. The tensor shapes at every
step are identical to Section 4.

```
Layer 1 params: 36,507,197 (same as Layer 0)
Cumulative params after 2 dense layers: 73,014,394
```

---

## Section 6: Layer 2 — First MoE Layer (The Other Critical Path)

**Class**: `NanoSeekDecoderLayer` at line 1378, with `is_moe_layer=True`.

Layer 2 is the first MoE layer (index 2 >= `first_k_dense_replace=2`). The
attention path is identical to layers 0–1, but the FFN is now a full MoE with
64 routed experts + 2 shared experts.

### 6a. MLA Attention (Same as Layer 0)

Identical architecture and shapes as Section 4b. The MLA parameters are
independent weights per layer, but the computation is the same:

```
Input:  [2, 512, 2048]
→ input_layernorm → MLA → residual add
Output: [2, 512, 2048]
```

### 6b. Post-Attention LayerNorm

```python
residual = hidden_states                                    # [2, 512, 2048]
hidden_states = self.post_attention_layernorm(hidden_states) # [2, 512, 2048]
```

### 6c. MoE Forward — The Other Critical Path

**Class**: `MoE` at line 586, `forward()` at line 663.

#### Step 1: Flatten Input

```python
batch_size, seq_len, dim = x.shape     # 2, 512, 2048
N = batch_size * seq_len               # 1024
x_flat = x.view(N, dim)               # [1024, 2048]
```

All 1024 tokens are now independent rows. The MoE routes each token
independently to its own set of experts.

#### Step 2: Shared Expert Computation (line 684)

```python
shared_output = self._compute_shared_output(x_flat)
```

With `n_shared_experts=2`, this calls `_compute_shared_output` (line 650):

```python
# Two shared experts — sum their outputs (line 661)
shared_out = expert_0(x_flat) + expert_1(x_flat)
```

Each shared expert is a `SwiGLUFFN(dim=2048, inter_dim=768)`:

```python
# For each shared expert:
gate = gate_proj(x_flat)   # Linear(2048 → 768): [1024, 2048] → [1024, 768]
up   = up_proj(x_flat)     # Linear(2048 → 768): [1024, 2048] → [1024, 768]
out  = down_proj(F.silu(gate) * up)  # Linear(768 → 2048): [1024, 768] → [1024, 2048]
```

```
shared_output: [1024, 2048]  (sum of 2 expert outputs)
```

**Shared expert params (per expert)**: 3 × 2048 × 768 = 4,718,592
**Total shared (2 experts)**: 9,437,184

#### Step 3: Gate / Router (line 691)

```python
weights, indices = self.gate(x_flat)   # Gate.forward() at line 409
```

Inside `Gate.forward()` (line 409):

```python
# (a) Compute raw scores
scores = F.linear(x_flat, self.weight)
#   x_flat:      [1024, 2048]
#   gate.weight: [64, 2048]     (Parameter)
#   scores:      [1024, 64]

# (b) Apply sigmoid scoring (DeepSeek V3 innovation)
scores = torch.sigmoid(scores)
#   scores: [1024, 64]  — each value in (0, 1)

# (c) Add bias for load balancing during training (line 418)
scores_for_selection = scores + self.expert_bias.unsqueeze(0)
#   expert_bias: [64]  (buffer, initialized to zeros)
#   scores_for_selection: [1024, 64]

# (d) Group-based routing (lines 422-430)
#   n_expert_groups=8, n_limited_groups(topk_group)=4, experts_per_group=8
scores_grouped = scores_for_selection.view(-1, 8, 8)    # [1024, 8, 8]
group_scores = scores_grouped.max(dim=-1).values         # [1024, 8]
_, top_groups = group_scores.topk(4, dim=-1)             # [1024, 4]

# Create group mask — only keep scores from top-4 groups
group_mask = torch.zeros_like(group_scores, dtype=torch.bool)  # [1024, 8]
group_mask.scatter_(1, top_groups, True)                        # 4 True per row
group_mask = group_mask.unsqueeze(-1).expand(-1, -1, 8)        # [1024, 8, 8]
group_mask = group_mask.reshape(-1, 64)                         # [1024, 64]
scores_for_selection = scores_for_selection.masked_fill(~group_mask, float('-inf'))

# (e) Select top-8 experts from the allowed groups (line 432)
topk_weights, topk_indices = scores_for_selection.topk(8, dim=-1)
#   topk_weights: [1024, 8]
#   topk_indices: [1024, 8]

# (f) Get original (unbiased) scores for the selected experts
weights = scores.gather(dim=-1, index=topk_indices)
#   weights: [1024, 8]  — sigmoid scores, not bias-adjusted

# (g) Scale by route_scale
weights = weights * self.route_scale    # × 2.5
#   weights: [1024, 8]
```

**Concrete example — Token 42 routing**:

```
Token 42 input: [2048-dim vector]

Sigmoid scores (all 64 experts):
  Expert  0: 0.52   Expert  1: 0.48   Expert  2: 0.61   ...
  Expert  7: 0.73   Expert  8: 0.44   ...
  Expert 12: 0.69   Expert 18: 0.67   ...
  Expert 25: 0.58   Expert 31: 0.55   ...
  Expert 48: 0.64   Expert 55: 0.71   ...

Group scores (max per group of 8):
  Group 0 (experts 0-7):   0.73    ← selected
  Group 1 (experts 8-15):  0.69    ← selected
  Group 2 (experts 16-23): 0.67    ← selected
  Group 3 (experts 24-31): 0.58
  Group 4 (experts 32-39): 0.51
  Group 5 (experts 40-47): 0.47
  Group 6 (experts 48-55): 0.71    ← selected
  Group 7 (experts 56-63): 0.43

Top-4 groups: [0, 1, 2, 6]
Experts in allowed groups: 0-23, 48-55 (32 of 64 candidates)

Top-8 from allowed: [7, 55, 12, 18, 2, 48, 25→MASKED, ...]
  → Actual: [7, 55, 12, 18, 2, 48, 3, 15]

Weights (sigmoid × 2.5):
  [0.73×2.5, 0.71×2.5, 0.69×2.5, 0.67×2.5, 0.61×2.5, 0.64×2.5, 0.57×2.5, 0.53×2.5]
= [1.825,    1.775,    1.725,    1.675,    1.525,    1.600,    1.425,    1.325]
```

**Gate params**: weight [64, 2048] = 131,072 parameters

#### Step 4: Token-Centric Dispatch (line 694)

```python
routed_output = token_centric_dispatch(x_flat, indices, weights, self.experts)
```

`token_centric_dispatch` at line 485 — the most performance-critical function:

```python
N, D = x_flat.shape    # 1024, 2048
K = indices.shape[1]   # 8
E = len(experts)       # 64

# ─── Phase 1: PERMUTE (lines 524-544) ───

flat_indices = indices.view(-1)                    # [8192]  (1024×8)
flat_weights = weights.view(-1)                    # [8192]

token_ids = torch.arange(1024).unsqueeze(1).expand(-1, 8).reshape(-1)  # [8192]
# token_ids = [0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, ..., 1023,...]

# Sort by expert ID to create contiguous batches
sorted_order = torch.argsort(flat_indices, stable=True)    # [8192]
sorted_expert_ids = flat_indices[sorted_order]              # [8192] (sorted)
sorted_token_ids  = token_ids[sorted_order]                 # [8192]
sorted_weights    = flat_weights[sorted_order]              # [8192]

# Count tokens per expert
expert_counts = torch.bincount(sorted_expert_ids.int(), minlength=64)
# expert_counts: [64]  — roughly 128 tokens each (8192/64)
# e.g., [131, 125, 128, 134, 127, 130, ...]

# Gather input in expert-sorted order
permuted_input = x_flat[sorted_token_ids]           # [8192, 2048]

# ─── Phase 2: COMPUTE (lines 551-565) ───

expert_batches = torch.split(permuted_input, expert_counts.tolist())
# 64 tensors, e.g., expert_batches[0]: [131, 2048], expert_batches[1]: [125, 2048], ...

expert_outputs = []
for expert_id, batch in enumerate(expert_batches):
    if batch.shape[0] > 0:
        expert_outputs.append(experts[expert_id](batch))
        # Each expert is SwiGLUFFN(2048, 768):
        #   batch: [~128, 2048] → [~128, 2048]

permuted_output = torch.cat(expert_outputs, dim=0)  # [8192, 2048]

# ─── Phase 3: UNPERMUTE (lines 572-582) ───

# Weight the outputs
weighted_output = permuted_output * sorted_weights.unsqueeze(-1)
#   [8192, 2048] × [8192, 1] → [8192, 2048]

# Scatter-add back to original token positions
output = torch.zeros(1024, 2048, device=device, dtype=dtype)
output.scatter_add_(
    dim=0,
    index=sorted_token_ids.unsqueeze(-1).expand(-1, 2048),  # [8192, 2048]
    src=weighted_output                                       # [8192, 2048]
)
# output: [1024, 2048]
# Each token accumulates weighted contributions from its 8 experts
```

#### Step 5: Combine Shared + Routed (line 700)

```python
output = routed_output + shared_output    # [1024, 2048] + [1024, 2048]
output = output.view(batch_size, seq_len, dim)  # [2, 512, 2048]
```

#### Step 6: Auxiliary Loss (lines 708–717)

During training, the MoE computes a sequence-level auxiliary loss:

```python
# Track expert load (line 710-711)
aux_data["expert_load"] = self.gate.expert_load.clone()    # [64]

# Sequence aux loss (lines 713-717)
load = self.gate.expert_load               # [64] — counts per expert
target_load = 1024 * 8 / 64               # = 128.0 (perfect balance)
load_imbalance = ((load - 128.0) ** 2).mean()   # scalar
aux_data["seq_aux_loss"] = 0.0001 * load_imbalance   # seq_aux_loss_alpha
```

#### MoE Parameter Summary for One Layer

| Component          | Shape/Count       | Parameters     |
|--------------------|-------------------|----------------|
| Gate weight        | [64, 2048]        | 131,072        |
| 64 routed experts  | 64 × SwiGLU(2048,768) | 64 × 4,718,592 = 301,989,888 |
| 2 shared experts   | 2 × SwiGLU(2048,768)  | 2 × 4,718,592 = 9,437,184 |
| **Total MoE**      |                   | **311,558,144** |

### 6d. MoE Residual Connection

```python
hidden_states = residual + hidden_states   # [2, 512, 2048]
```

### 6e. Layer 2 Complete

```
Total Layer 2 params:
  MLA:           4,290,109
  Layer norms:       4,096
  MoE:         311,558,144
  ─────────────────────────
  Total:       315,852,349
```

---

## Section 7: Layers 3–15 (Remaining MoE Layers)

Layers 3 through 15 are all MoE layers, structurally identical to Layer 2:
- MLA attention (same shapes, independent weights)
- MoE FFN with 64 routed + 2 shared experts

Each layer processes:
```
Input: [2, 512, 2048] → Output: [2, 512, 2048]
Params per layer: 315,852,349
```

There are 13 more MoE layers (layers 3–15), for a total of 14 MoE layers.

### Cumulative Parameter Count After All 16 Layers

```
Embeddings (input):     134,217,728
Embeddings (output):    134,217,728
2 dense layers:       2 × 36,507,197 =    73,014,394
14 MoE layers:       14 × 315,852,349 = 4,421,932,886
Final norm:                                     2,048
──────────────────────────────────────────────────────
Subtotal (pre-MTP):              4,763,384,784
```

### Shape Trace Summary for All 16 Layers

```
                    hidden_states shape
                    ─────────────────────
After embed_tokens: [2, 512, 2048]
After Layer  0:     [2, 512, 2048]   (Dense: MLA + MLP)
After Layer  1:     [2, 512, 2048]   (Dense: MLA + MLP)
After Layer  2:     [2, 512, 2048]   (MoE:  MLA + 64E+2S)
After Layer  3:     [2, 512, 2048]   (MoE:  MLA + 64E+2S)
  ...
After Layer 15:     [2, 512, 2048]   (MoE:  MLA + 64E+2S)
```

The hidden dimension **never changes** through all 16 layers — it's always
[batch, seq, 2048]. The depth of computation and richness of representation
increases, but the tensor shape is invariant.

---

## Section 8: Final Norm + LM Head

**Lines 1640–1641** of `NanoSeekModel.forward()`:

```python
hidden_states = self.norm(hidden_states)    # RMSNorm(2048)
#   [2, 512, 2048] → [2, 512, 2048]

logits = self.lm_head(hidden_states)        # Linear(2048 → 65536, no bias)
#   [2, 512, 2048] → [2, 512, 65536]
```

### Output Dictionary Construction

Lines 1646–1660 build the output dict:

```python
outputs = {
    "logits":        # [2, 512, 65536]  — raw predictions for every position
    "hidden_states": # [2, 512, 2048]   — last-layer hidden states
}
```

Since `labels` is provided, `_compute_loss()` is called at line 1658.

---

## Section 9: Loss Computation

**Method**: `_compute_loss()` at line 1697.

### 9a. Main Cross-Entropy Loss

```python
# Shift logits and labels for next-token prediction
shift_logits = logits[:, :-1, :].contiguous()
#   logits: [2, 512, 65536] → shift_logits: [2, 511, 65536]
#   (remove last position — no label for it)

shift_labels = labels[:, 1:].contiguous()
#   labels: [2, 512] → shift_labels: [2, 511]
#   (remove first position — it's predicted by the embedding)

# Cross-entropy loss
main_loss = F.cross_entropy(
    shift_logits.view(-1, 65536),    # [1022, 65536]
    shift_labels.view(-1),            # [1022]
    ignore_index=-100,
)
# → scalar tensor
```

**What this computes**: For each of the 1022 token positions (2 × 511), compute
`-log(softmax(logits)[correct_token])`, then average. At initialization with
random weights, `main_loss ≈ ln(65536) ≈ 11.09`.

### 9b. MTP Loss

**Lines 1709–1717**:

```python
if self.mtp is not None:
    mtp_outputs = self.mtp(hidden_states, labels=labels)
```

This calls `MultiTokenPrediction.forward()` at line 954.

#### MTP Forward Pass Trace

```python
# In MultiTokenPrediction.forward() (line 954):
main_hidden = hidden_states    # [2, 512, 2048]  (from last layer)
prev_hidden = main_hidden      # [2, 512, 2048]

# For module 0 (the only MTP module, num_mtp_modules=1):
token_offset = 1

# Get target tokens for embedding lookup
target_tokens = labels[:, 1:].contiguous()    # [2, 511]
effective_len = 511
current_hidden = prev_hidden[:, :511]          # [2, 511, 2048]
current_main = main_hidden[:, :511]            # [2, 511, 2048]
```

Now `MTPModule.forward()` at line 868 is called:

```python
# (a) Normalize previous hidden states
normed_hidden = self.hidden_norm(prev_hidden)    # RMSNorm(2048)
#   [2, 511, 2048] → [2, 511, 2048]

# (b) Embed target tokens and normalize
safe_target_tokens = torch.where(target_tokens < 0, 0, target_tokens)
token_embeds = self.embed_tokens(safe_target_tokens)  # shared embedding
#   [2, 511] → [2, 511, 2048]
normed_embeds = self.embed_norm(token_embeds)    # RMSNorm(2048)
#   [2, 511, 2048] → [2, 511, 2048]

# (c) Concatenate and project (DeepSeek V3: concat, NOT add!)
concatenated = torch.cat([normed_hidden, normed_embeds], dim=-1)
#   [2, 511, 2048] + [2, 511, 2048] → [2, 511, 4096]
hidden_states = self.concat_proj(concatenated)   # Linear(4096 → 2048)
#   [2, 511, 4096] → [2, 511, 2048]

# (d) Process through MTPBlock (line 898-899)
for block in self.blocks:    # 1 block
    hidden_states = block(hidden_states, cross_hidden, attention_mask)
```

Inside `MTPBlock.forward()` at line 774:

```python
# Cross-attention with main hidden states
residual = hidden_states                         # [2, 511, 2048]
hidden_states = self.input_norm(hidden_states)   # RMSNorm
cross_output, _ = self.cross_attn(
    query=hidden_states,     # [2, 511, 2048]
    key=main_hidden,         # [2, 511, 2048]
    value=main_hidden        # [2, 511, 2048]
)
# nn.MultiheadAttention with 8 heads, head_dim=256
# cross_output: [2, 511, 2048]
hidden_states = residual + cross_output

# Self-attention with causal mask
residual = hidden_states
hidden_states = self.cross_norm(hidden_states)
causal_mask = torch.triu(ones(511, 511), diagonal=1)  # bool upper-tri
self_output, _ = self.self_attn(
    query=hidden_states, key=hidden_states, value=hidden_states,
    attn_mask=causal_mask
)
# self_output: [2, 511, 2048]
hidden_states = residual + self_output

# SwiGLU FFN
residual = hidden_states
hidden_states = self.ffn_norm(hidden_states)
gate = F.silu(self.gate_proj(hidden_states))   # Linear(2048 → 8192)
up = self.up_proj(hidden_states)                # Linear(2048 → 8192)
hidden_states = self.down_proj(gate * up)       # Linear(8192 → 2048)
hidden_states = residual + hidden_states
# Output: [2, 511, 2048]
```

Back in `MTPModule.forward()`:

```python
# (e) Final norm and LM head
output_hidden = self.output_norm(hidden_states)    # [2, 511, 2048]
logits = self.lm_head(output_hidden)               # shared lm_head
#   [2, 511, 2048] → [2, 511, 65536]
```

#### MTP Loss Computation

Back in `MultiTokenPrediction.forward()` (lines 988–1003):

```python
# Predict position i+2 tokens
pred_offset = 2    # module_idx(0) + 2
pred_len = 512 - 2 = 510

shift_logits = logits[:, :510].contiguous()           # [2, 510, 65536]
shift_labels = labels[:, 2:512].contiguous()           # [2, 510]

mtp_loss_single = F.cross_entropy(
    shift_logits.view(-1, 65536),    # [1020, 65536]
    shift_labels.view(-1),            # [1020]
    ignore_index=-100,
)

# Weight by decay (line 1001): weight = 0.8^0 = 1.0
total_loss = 1.0 * mtp_loss_single

# Normalize (line 1008-1009): weight_sum = 1.0
results["mtp_loss"] = total_loss / 1.0
```

### 9c. Auxiliary Losses

Collected during the forward pass through all 14 MoE layers:

```python
# From NanoSeekModel.forward() lines 1634-1638:
# all_aux_losses — list of 14 scalar tensors (one per MoE layer)
# all_indexer_losses — empty list (DSA disabled in Phase 1)

total_aux_loss = sum(all_aux_losses)         # scalar
total_indexer_loss = sum(all_indexer_losses)  # 0.0 (DSA disabled)
```

Each MoE layer's aux loss is:
```
α × mean((load_i - target_load)²)
= 0.0001 × mean((load_i - 128)²)
```

Typical value at random init: `~0.0001 × 500 = 0.05` per layer.
Total across 14 layers: `~0.7`.

### 9d. Total Loss Assembly

**Lines 1707–1727**:

```python
total_loss = main_loss                                    # ~11.09

# MTP contribution
mtp_weight = self.get_mtp_loss_weight()                   # 0.3 (initial weight)
total_loss = total_loss + 0.3 * mtp_loss                  # +0.3 × ~11.09 = +3.33

# Auxiliary loss from MoE load balancing
total_loss = total_loss + aux_loss                        # +~0.7

# Indexer loss (0 when DSA disabled)
total_loss = total_loss + indexer_loss                    # +0.0

# TOTAL ≈ 11.09 + 3.33 + 0.7 + 0.0 ≈ 15.12
```

**Return dict** (assembled across lines 1706–1728):
```python
{
    "loss":       total_loss,     # scalar ~15.12
    "main_loss":  main_loss,      # scalar ~11.09
    "mtp_loss":   mtp_loss,       # scalar ~11.09
    "mtp_weight": 0.3,            # float
    "mtp_logits": [...],          # list of [2, 511, 65536]
    "aux_loss":   aux_loss,       # scalar ~0.7
    "logits":     logits,         # [2, 512, 65536]
    "hidden_states": hidden_states, # [2, 512, 2048]
}
```

---

## Section 10: Backward Pass

```python
outputs["loss"].backward()
```

The backward pass flows through the computational graph in reverse. PyTorch's
autograd engine traverses every operation and computes gradients via the chain
rule. Here we trace the major gradient flow paths.

### 10.1 Loss Gradients → LM Head

```
d_loss/d_total_loss = 1.0

# Main loss path:
d_loss/d_shift_logits → d_cross_entropy
#   grad shape: [1022, 65536] (same as shift_logits.view(-1, 65536))
#   For each position: grad = softmax(logits) - one_hot(label)
#   This is the beauty of CE + softmax: gradient is simply (prediction - target)

# Unshift: grad flows to logits[:, :-1, :]
d_logits: [2, 512, 65536]  (last position gets zero grad from main loss)

# Through lm_head: Linear(2048 → 65536)
d_hidden_final = d_logits @ lm_head.weight    # [2, 512, 65536] @ [65536, 2048]
#   d_hidden_final: [2, 512, 2048]
d_lm_head_weight = d_logits.T @ hidden_states # [65536, 2048]
```

### 10.2 MTP Loss Gradients

```
# MTP loss path (weight = 0.3):
d_loss/d_mtp_loss = 0.3

# Flows back through MTP module:
d_mtp_logits: [2, 511, 65536]
→ d_mtp_lm_head (shared with main lm_head — gradients accumulate!)
→ d_mtp_output_norm
→ d_MTPBlock:
    → d_ffn_norm + d_gate_proj + d_up_proj + d_down_proj
    → d_self_attn (Q, K, V, O projections)
    → d_cross_attn (Q=mtp, K/V=main_hidden) → d_main_hidden
    → d_input_norm
→ d_concat_proj: [2, 511, 4096] split into:
    → d_normed_hidden: [2, 511, 2048] → d_hidden_norm → d_prev_hidden
    → d_normed_embeds: [2, 511, 2048] → d_embed_norm → d_embed_tokens

# Critical: MTP gradients flow BACK to the main hidden states!
# d_main_hidden from MTP adds to d_hidden_final from the main loss.
```

### 10.3 Through Final Norm

```python
# d_hidden accumulates gradients from:
#   1. Main loss through lm_head
#   2. MTP loss through cross-attention
d_hidden_states: [2, 512, 2048]

# Through RMSNorm (self.norm):
# d_x = d_out * weight / rms - d_out * weight * x * (x·d_out·weight) / (rms³ * dim)
# Shape preserved: [2, 512, 2048]
```

### 10.4 Through Layers in Reverse (Layer 15 → Layer 0)

For each layer, working backwards:

```
Layer i (reverse order):
─────────────────────────────────────────────────

d_hidden: [2, 512, 2048]  (from layer above, or from norm)

# ═══ Second Residual (line 1471) ═══
# hidden = residual + ffn_output
# d_residual_2 = d_hidden
# d_ffn_output = d_hidden

# ═══ FFN Backward ═══
# Through post_attention_layernorm:
d_ffn_input: [2, 512, 2048]

# If MoE layer (layers 2-15):
#   Gradients flow to ALL 8 active experts for each token
#   + both shared experts (always active)
#   + the router/gate weights
#   d_gate_weight: [64, 2048]
#   d_expert_j for each active j: d_gate_proj, d_up_proj, d_down_proj
#   d_shared_expert_k: d_gate_proj, d_up_proj, d_down_proj

# If Dense layer (layers 0-1):
#   d_gate_proj: [5243, 2048]
#   d_up_proj:   [5243, 2048]
#   d_down_proj: [2048, 5243]

# ═══ First Residual (line 1465) ═══
# hidden = residual + attn_output
# d_residual_1 = d_residual_2 + d_ffn_backward_to_input
# d_attn_output = d_residual_2 + d_ffn_backward_to_input

# ═══ MLA Attention Backward ═══
# Through input_layernorm:
d_normed: [2, 512, 2048]

# Through wo: Linear(1024 → 2048)
d_attn_flat: [2, 512, 1024]
d_wo_weight: [2048, 1024]

# Reshape to [2, 16, 512, 64]
# Through attn_weights @ v:
d_attn_weights: [2, 16, 512, 512]
d_v:            [2, 16, 512, 64]

# Through softmax:
# d_score_ij = attn_ij × (d_attn_ij - Σ_k attn_ik × d_attn_ik)
d_scores: [2, 16, 512, 512]

# Through Q@K^T:
d_q: [2, 16, 512, 96] = d_scores @ k × scale
d_k: [2, 16, 512, 96] = d_scores.T @ q × scale

# Split q grad into nope + pe:
d_q_nope: [2, 512, 16, 64]
d_q_pe:   [2, 512, 16, 32]

# Through RoPE (rotation is orthogonal → inverse is conjugate):
d_q_pe_pre_rope: [2, 512, 16, 32]

# Through wq_b:
d_q_compressed: [2, 512, 430]
d_wq_b_weight: [1536, 430]

# Through q_norm:
d_q_pre_norm: [2, 512, 430]

# Through wq_a:
d_hidden_from_q: [2, 512, 2048]
d_wq_a_weight: [430, 2048]

# Similarly for K path:
d_k_nope → d_wkv_b → d_kv_compressed → d_kv_norm → d_wkv_a → d_hidden_from_kv
d_k_pe → d_rope → d_wkv_a (the pe component)

# Total gradient to layer input:
d_layer_input = d_residual_1 + d_hidden_from_q + d_hidden_from_kv
```

### 10.5 MoE-Specific Backward Details

For MoE layers, the backward through `token_centric_dispatch` reverses the
permute-compute-unpermute pattern:

```
# Forward: scatter_add_(src=weighted_output, index=sorted_token_ids)
# Backward: gather from d_output using sorted_token_ids

d_weighted_output = d_output[sorted_token_ids]    # [8192, 2048]

# Un-weight:
d_permuted_output = d_weighted_output * sorted_weights.unsqueeze(-1)
d_sorted_weights += (d_weighted_output * permuted_output).sum(-1)

# Split back into per-expert batches and backward through each expert
for expert_id, d_batch in enumerate(d_expert_batches):
    # Backward through SwiGLUFFN:
    # d_gate_proj, d_up_proj, d_down_proj for this expert
    pass

# Reverse permutation to get d_x_flat
d_x_flat.scatter_add_(0, sorted_token_ids.unsqueeze(-1).expand(-1, D), d_input_from_experts)

# Router/gate gradients:
# Through scores.gather → sigmoid → F.linear
d_gate_weight: [64, 2048]  (accumulated from all 1024 tokens)
```

**Key insight**: Even though only 8 of 64 experts are active per token, ALL 8
receive gradients. The router also receives gradients for its decisions, enabling
it to learn better routing over time.

### 10.6 Through Embeddings

```
# After all 16 layers backward:
d_embed_output: [2, 512, 2048]

# Through embed_tokens (lookup backward = scatter_add):
# For each (batch, position), add d_embed_output[b, p, :] to
# d_embed_weight[input_ids[b, p], :]
d_embed_weight: [65536, 2048]
# Only ~1024 of 65536 rows receive non-zero gradients (the tokens in the batch)
```

### 10.7 Gradient Summary

After `loss.backward()`, every parameter in the model has a `.grad` tensor:

| Component | Gradient Shapes | Count |
|-----------|----------------|-------|
| embed_tokens.weight | [65536, 2048] | 134M |
| lm_head.weight | [65536, 2048] | 134M |
| 16× MLA (wq_a, wq_b, wkv_a, wkv_b, wo, norms) | various | 16 × 4.29M = 68.6M |
| 16× layer norms | [2048] each | 16 × 4K = 65K |
| 2× dense MLP | 3×[5243, 2048] etc. | 2 × 32.2M = 64.4M |
| 14× MoE gate | [64, 2048] | 14 × 131K = 1.8M |
| 14× 64 routed experts | 3×[768, 2048] each | 14 × 64 × 4.72M = 4.23B |
| 14× 2 shared experts | 3×[768, 2048] each | 14 × 2 × 4.72M = 132M |
| MTP module | various | ~92M |
| Final norm | [2048] | 2K |

---

## Section 11: Parameter Count Verification

### 11.1 Detailed Weight Matrix Inventory

#### Embedding Layers

| Weight | Shape | Params |
|--------|-------|--------|
| `embed_tokens.weight` | [65536, 2048] | 134,217,728 |
| `lm_head.weight` | [65536, 2048] | 134,217,728 |
| **Subtotal** | | **268,435,456** |

#### Per Dense Layer (layers 0–1) — MLA + MLP + Norms

| Weight | Shape | Params |
|--------|-------|--------|
| `input_layernorm.weight` | [2048] | 2,048 |
| `self_attn.wq_a.weight` | [430, 2048] | 880,640 |
| `self_attn.q_norm.weight` | [430] | 430 |
| `self_attn.wq_b.weight` | [1536, 430] | 660,480 |
| `self_attn.wkv_a.weight` | [175, 2048] | 358,400 |
| `self_attn.kv_norm.weight` | [143] | 143 |
| `self_attn.wkv_b.weight` | [2048, 143] | 292,864 |
| `self_attn.wo.weight` | [2048, 1024] | 2,097,152 |
| `post_attention_layernorm.weight` | [2048] | 2,048 |
| `ffn.ffn.gate_proj.weight` | [5243, 2048] | 10,737,664 |
| `ffn.ffn.up_proj.weight` | [5243, 2048] | 10,737,664 |
| `ffn.ffn.down_proj.weight` | [2048, 5243] | 10,737,664 |
| **Per dense layer** | | **36,507,197** |
| **× 2 layers** | | **73,014,394** |

#### Per MoE Layer (layers 2–15) — MLA + MoE + Norms

| Weight | Shape | Params |
|--------|-------|--------|
| `input_layernorm.weight` | [2048] | 2,048 |
| MLA (same as dense layer) | — | 4,290,109 |
| `post_attention_layernorm.weight` | [2048] | 2,048 |
| `ffn.gate.weight` | [64, 2048] | 131,072 |
| 64× `ffn.experts[i].gate_proj.weight` | [768, 2048] | 64 × 1,572,864 |
| 64× `ffn.experts[i].up_proj.weight` | [768, 2048] | 64 × 1,572,864 |
| 64× `ffn.experts[i].down_proj.weight` | [2048, 768] | 64 × 1,572,864 |
| 2× `ffn.shared_experts[i].gate_proj.weight` | [768, 2048] | 2 × 1,572,864 |
| 2× `ffn.shared_experts[i].up_proj.weight` | [768, 2048] | 2 × 1,572,864 |
| 2× `ffn.shared_experts[i].down_proj.weight` | [2048, 768] | 2 × 1,572,864 |
| **Per MoE layer** | | **315,852,349** |
| **× 14 layers** | | **4,421,932,886** |

#### Final Norm

| Weight | Shape | Params |
|--------|-------|--------|
| `norm.weight` | [2048] | 2,048 |

#### MTP Module (1 module)

| Weight | Shape | Params |
|--------|-------|--------|
| `mtp.mtp_modules[0].hidden_norm.weight` | [2048] | 2,048 |
| `mtp.mtp_modules[0].embed_norm.weight` | [2048] | 2,048 |
| `mtp.mtp_modules[0].concat_proj.weight` | [2048, 4096] | 8,388,608 |
| `mtp.mtp_modules[0].blocks[0].input_norm.weight` | [2048] | 2,048 |
| `mtp.mtp_modules[0].blocks[0].cross_attn.in_proj_weight` | [6144, 2048] | 12,582,912 |
| `mtp.mtp_modules[0].blocks[0].cross_attn.in_proj_bias` | [6144] | 6,144 |
| `mtp.mtp_modules[0].blocks[0].cross_attn.out_proj.weight` | [2048, 2048] | 4,194,304 |
| `mtp.mtp_modules[0].blocks[0].cross_attn.out_proj.bias` | [2048] | 2,048 |
| `mtp.mtp_modules[0].blocks[0].cross_norm.weight` | [2048] | 2,048 |
| `mtp.mtp_modules[0].blocks[0].self_attn.in_proj_weight` | [6144, 2048] | 12,582,912 |
| `mtp.mtp_modules[0].blocks[0].self_attn.in_proj_bias` | [6144] | 6,144 |
| `mtp.mtp_modules[0].blocks[0].self_attn.out_proj.weight` | [2048, 2048] | 4,194,304 |
| `mtp.mtp_modules[0].blocks[0].self_attn.out_proj.bias` | [2048] | 2,048 |
| `mtp.mtp_modules[0].blocks[0].ffn_norm.weight` | [2048] | 2,048 |
| `mtp.mtp_modules[0].blocks[0].gate_proj.weight` | [8192, 2048] | 16,777,216 |
| `mtp.mtp_modules[0].blocks[0].up_proj.weight` | [8192, 2048] | 16,777,216 |
| `mtp.mtp_modules[0].blocks[0].down_proj.weight` | [2048, 8192] | 16,777,216 |
| `mtp.mtp_modules[0].output_norm.weight` | [2048] | 2,048 |
| **MTP subtotal** | | **92,303,360** |
| (embed_tokens, lm_head shared — 0 extra) | | |

#### Grand Total

```
Embeddings:           268,435,456
2 Dense layers:        73,014,394
14 MoE layers:      4,421,932,886
Final norm:                 2,048
MTP module:            92,303,360
───────────────────────────────────
TOTAL:             4,855,688,144  ≈ 4.86B parameters
```

**Config estimate** (`estimated_total_params`): The config's estimate method
(line 609) computes a slightly different number because it uses simplified
counting (doesn't include MTP nn.MultiheadAttention biases, etc.). The actual
`model.num_parameters()` count is the ground truth.

#### Active Parameters per Forward Pass

```
Embeddings:           268,435,456   (always active)
2 Dense layers:        73,014,394   (always active)
14 MoE layers MLA:    14 × 4,294,205 = 60,118,870  (always active)
14 MoE norms:         14 × 4,096 = 57,344
14× gate:             14 × 131,072 = 1,834,496
14× 2 shared experts: 14 × 9,437,184 = 132,120,576
14× 8 routed experts: 14 × 8 × 4,718,592 = 528,481,536
Final norm:           2,048
MTP module:           92,303,360
──────────────────────────────────
ACTIVE:               ~1,156,368,080 ≈ 1.16B
```

---

## Section 12: Memory Analysis

All analysis assumes bf16 (2 bytes per parameter) unless noted.

### 12.1 Model Parameters

```
Total parameters: 4,855,688,144
Memory (bf16):    4,855,688,144 × 2 = 9,711,376,288 bytes ≈ 9.05 GB
```

### 12.2 Gradients

```
Same shape as parameters (one gradient per parameter):
Memory (bf16): ≈ 9.05 GB
```

### 12.3 Optimizer States (AdamW)

AdamW stores 2 states per parameter in fp32:
- First moment (m): mean of gradients
- Second moment (v): mean of squared gradients

```
Per parameter: 2 × 4 bytes (fp32) = 8 bytes
Total: 4,855,688,144 × 8 = 38,845,505,152 bytes ≈ 36.2 GB
```

With 8×H100 DDP (data parallel), optimizer states are NOT sharded by default.
Each GPU holds the full optimizer state. With ZeRO-1 or FSDP, this can be
sharded 8 ways: `36.2 / 8 ≈ 4.5 GB/GPU`.

### 12.4 Activations (Without Gradient Checkpointing)

For batch_size=2, seq_len=512:

```
Per-layer activations (stored for backward):
─────────────────────────────────────────────
  Input to layernorm:       2 × 512 × 2048 × 2 = 4.0 MB
  MLA intermediates:
    q after wq_a:           2 × 512 × 430 × 2  = 0.8 MB
    q after wq_b:           2 × 512 × 1536 × 2 = 3.0 MB
    kv after wkv_a:         2 × 512 × 175 × 2  = 0.3 MB
    kv after wkv_b:         2 × 512 × 2048 × 2 = 4.0 MB
    attention weights:      2 × 16 × 512² × 2  = 16.8 MB  ← largest!
    attn output:            2 × 16 × 512 × 64 × 2 = 2.0 MB
  FFN (MoE layer):
    shared expert acts:     2 × (2 × 1024 × 768 × 2) = 6.0 MB
    routed expert acts:     varies, ~8192 × 768 × 2 × 2 = 25.2 MB
    gate scores:            1024 × 64 × 2 = 0.1 MB
    permutation indices:    ~8192 × 8 bytes = 0.1 MB

  Per MoE layer total:    ~62 MB
  Per Dense layer total:  ~45 MB

Total 16-layer activations:
  2 × 45 + 14 × 62 = 90 + 868 = 958 MB ≈ ~1.0 GB

Plus MTP activations:     ~50 MB
Plus logits:              2 × 512 × 65536 × 2 = 128 MB
──────────────────────────────────────
Total activations:        ~1.2 GB
```

### 12.5 With Gradient Checkpointing

When `gradient_checkpointing=True` (config default, line 1009):

Activations are recomputed during backward. Only layer inputs are stored:

```
Per layer: just the input hidden_states = 2 × 512 × 2048 × 2 = 4.0 MB
16 layers: 64 MB

But attention weights are recomputed (2× compute, ~0.5× memory):
Total with checkpointing: ~200 MB (vs 1.2 GB without)
```

**Note**: The implementation at line 1683 (`_gradient_checkpoint_layer`) uses
`torch.utils.checkpoint.checkpoint` with `use_reentrant=False`, which is the
modern PyTorch approach. However, it drops `aux_data` (returns empty dict at
line 1695), meaning auxiliary losses are NOT computed under gradient
checkpointing.

### 12.6 KV Cache (Inference)

Per token per layer, the MLA KV cache stores:
```
kv_compressed: [143]  +  k_pe: [1, 32]  →  175 values
Memory per token per layer (bf16): 175 × 2 = 350 bytes

Standard MHA for comparison:
  2 × 16 × 128 = 4096 values → 8,192 bytes
  MLA compression: 8192 / 350 = 23.4×
```

For 4096-token context, 16 layers:
```
MLA:      16 × 4096 × 350 = 22.9 MB
Standard: 16 × 4096 × 8192 = 536.9 MB
Savings:  514 MB per batch element
```

### 12.7 Memory Summary Table

| Component | Size (bf16/fp32) | With 8-way DDP |
|-----------|-----------------|----------------|
| Model parameters | 9.05 GB | 9.05 GB each |
| Gradients | 9.05 GB | 9.05 GB each |
| Optimizer (AdamW fp32) | 36.2 GB | 36.2 GB each |
| Activations (no ckpt, bs=2×512) | ~1.2 GB | ~1.2 GB each |
| Activations (with ckpt, bs=2×512) | ~0.2 GB | ~0.2 GB each |
| **Total per GPU (no ckpt)** | **~55.5 GB** | |
| **Total per GPU (with ckpt)** | **~54.5 GB** | |

With ZeRO-1 (optimizer sharding):
```
Per GPU: 9.05 + 9.05 + (36.2/8) + 1.2 = ~23.8 GB  ← fits in H100 80GB easily
```

---

## Section 13: Complete Data Flow Diagram

```
input_ids [2, 512]
    │
    ▼
embed_tokens ──────────────────────────────────────────── [2, 512, 2048]
    │
    ├─── position_ids [2, 512]
    ├─── causal_mask [1, 1, 512, 512]
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 0 (Dense)                                                    │
│                                                                     │
│  residual ← hidden_states                                           │
│  hidden = input_layernorm(hidden_states)              [2,512,2048]  │
│                                                                     │
│  ┌── MLA ─────────────────────────────────────────────────────────┐ │
│  │  Q: hidden → wq_a[→430] → q_norm → wq_b[→1536]               │ │
│  │     → reshape [2,512,16,96] → split nope[64]+pe[32]           │ │
│  │  KV: hidden → wkv_a[→175] → split kv[143]+k_pe[32]           │ │
│  │      kv → kv_norm → wkv_b[→2048] → reshape → split k[64]+v[64]│ │
│  │  RoPE: q_pe, k_pe × freqs_cis                                 │ │
│  │  Q=[nope;pe] K=[nope;pe_expanded]  → Q@K^T/√96 + mask         │ │
│  │  → softmax(fp32) → @V → wo[1024→2048]                         │ │
│  └────────────────────────────────────────────── [2,512,2048] ────┘ │
│                                                                     │
│  hidden_states = residual + attn_output                             │
│  residual ← hidden_states                                           │
│  hidden = post_attention_layernorm(hidden_states)                   │
│                                                                     │
│  ┌── Dense MLP (SwiGLU) ─────────────────────────────────────────┐ │
│  │  gate = silu(gate_proj(hidden))     [2048→5243]               │ │
│  │  up   = up_proj(hidden)             [2048→5243]               │ │
│  │  out  = down_proj(gate × up)        [5243→2048]               │ │
│  └────────────────────────────────────────────── [2,512,2048] ────┘ │
│                                                                     │
│  hidden_states = residual + mlp_output                              │
└─────────────────────────────────────────────────── [2,512,2048] ────┘
    │
    ▼
┌─ Layer 1 (Dense) ──── same structure ──── [2,512,2048] ─────────────┐
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 2 (First MoE Layer)                                          │
│                                                                     │
│  (MLA identical to Layer 0)                                         │
│                                                                     │
│  ┌── MoE ────────────────────────────────────────────────────────┐  │
│  │  x_flat = hidden.view(1024, 2048)                             │  │
│  │                                                               │  │
│  │  Shared: expert_0(x) + expert_1(x)         [1024, 2048]      │  │
│  │                                                               │  │
│  │  Gate: sigmoid(x @ W^T)                     [1024, 64]        │  │
│  │    + bias → group routing (top-4 of 8 groups)                 │  │
│  │    → top-8 experts per token                                  │  │
│  │                                                               │  │
│  │  Dispatch: argsort → permute → per-expert SwiGLU → unpermute  │  │
│  │    → weighted scatter_add                   [1024, 2048]      │  │
│  │                                                               │  │
│  │  output = routed + shared                   [1024, 2048]      │  │
│  │  → view                                     [2, 512, 2048]    │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  hidden_states = residual + moe_output                              │
└─────────────────────────────────────────────────── [2,512,2048] ────┘
    │
    ▼
┌─ Layers 3-15 (MoE) ─── same structure ─── [2,512,2048] ────────────┐
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
norm (RMSNorm) ────────────────────────────────────── [2, 512, 2048]
    │
    ▼
lm_head (Linear 2048→65536) ──────────────────────── [2, 512, 65536]
    │                                                      │
    │                                                      ▼
    │                                               shift + CE loss
    │                                                → main_loss (scalar)
    ▼
┌── MTP Module ───────────────────────────────────────────────────────┐
│  hidden_norm(hidden_states) ∥ embed_norm(embed(labels))            │
│  → concat [2,511,4096] → proj [2,511,2048]                         │
│  → MTPBlock (cross_attn + self_attn + SwiGLU FFN)                  │
│  → output_norm → lm_head → shift + CE → mtp_loss (scalar)          │
└─────────────────────────────────────────────────────────────────────┘

total_loss = main_loss + 0.3 × mtp_loss + Σ(aux_loss) + Σ(indexer_loss)
    │
    ▼
loss.backward()
    │
    ▼
(gradients flow through entire graph in reverse)
```

---

## Section 14: Verification Code

```python
"""
Verification script — run this to confirm all tensor shapes described
in this document. Uses a reduced config for fast execution.

Usage:
    cd /workspace
    python -c "exec(open('implementation_docs/verify_shapes.py').read())"
"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/workspace')

from model.config import NanoSeekConfig, MLAConfig, MoEConfig, MTPConfig, SparseAttentionConfig
from model.model import (
    NanoSeekModel, MultiHeadLatentAttention, MoE, Gate,
    RMSNorm, create_causal_mask, SwiGLUFFN, MLP,
    token_centric_dispatch, precompute_freqs_cis, apply_rotary_emb,
)

# ─── Full-size config (no reduction — verify real shapes) ───
config = NanoSeekConfig(
    vocab_size=65536,
    hidden_size=2048,
    num_layers=16,
    num_heads=16,
    intermediate_size=5243,
    mla=MLAConfig(
        q_lora_rank=430, kv_lora_rank=143,
        qk_nope_head_dim=64, qk_rope_head_dim=32, v_head_dim=64,
    ),
    moe=MoEConfig(
        n_routed_experts=64, num_experts_per_tok=8, n_shared_experts=2,
        moe_intermediate_size=768, n_group=8, topk_group=4,
        scoring_func="sigmoid", routed_scaling_factor=2.5,
        first_k_dense_replace=2, seq_aux_loss_alpha=0.0001,
    ),
    mtp=MTPConfig(num_mtp_modules=1, mtp_num_heads=8),
    sparse=SparseAttentionConfig(enabled=False),
    total_tokens=22_000_000_000,
)

batch_size, seq_len = 2, 512
device = "cpu"

print("=" * 70)
print("NanoSeek Shape Verification")
print("=" * 70)

# ─── Section 2: Embedding ───
embed = nn.Embedding(65536, 2048)
input_ids = torch.randint(0, 65536, (batch_size, seq_len))
hidden = embed(input_ids)
assert hidden.shape == (2, 512, 2048), f"Embedding: {hidden.shape}"
print(f"✓ Embedding:     {hidden.shape}")

# ─── Section 3: Causal Mask ───
mask = create_causal_mask(512, dtype=torch.float32)
assert mask.shape == (1, 1, 512, 512), f"Mask: {mask.shape}"
print(f"✓ Causal mask:   {mask.shape}")

# ─── Section 4: MLA shapes ───
mla = MultiHeadLatentAttention(
    hidden_size=2048, num_heads=16,
    q_lora_rank=430, kv_lora_rank=143,
    qk_nope_head_dim=64, qk_rope_head_dim=32, v_head_dim=64,
    max_position_embeddings=4096,
)
# Trace internal shapes
x = hidden.clone()
q = mla.wq_a(x)
assert q.shape == (2, 512, 430), f"wq_a: {q.shape}"
print(f"✓ wq_a output:   {q.shape}")

q = mla.q_norm(q)
q = mla.wq_b(q)
assert q.shape == (2, 512, 1536), f"wq_b: {q.shape}"
print(f"✓ wq_b output:   {q.shape}")

q = q.view(2, 512, 16, 96)
q_nope, q_pe = torch.split(q, [64, 32], dim=-1)
assert q_nope.shape == (2, 512, 16, 64), f"q_nope: {q_nope.shape}"
assert q_pe.shape == (2, 512, 16, 32), f"q_pe: {q_pe.shape}"
print(f"✓ q_nope:        {q_nope.shape}")
print(f"✓ q_pe:          {q_pe.shape}")

kv = mla.wkv_a(x)
assert kv.shape == (2, 512, 175), f"wkv_a: {kv.shape}"
print(f"✓ wkv_a output:  {kv.shape}")

kv_comp, k_pe = torch.split(kv, [143, 32], dim=-1)
assert kv_comp.shape == (2, 512, 143)
assert k_pe.shape == (2, 512, 32)
print(f"✓ kv_compressed: {kv_comp.shape}")
print(f"✓ k_pe_current:  {k_pe.shape}")

kv_comp = mla.kv_norm(kv_comp)
kv_exp = mla.wkv_b(kv_comp)
assert kv_exp.shape == (2, 512, 2048), f"wkv_b: {kv_exp.shape}"
print(f"✓ wkv_b output:  {kv_exp.shape}")

kv_exp = kv_exp.view(2, 512, 16, 128)
k_nope, v = torch.split(kv_exp, [64, 64], dim=-1)
assert k_nope.shape == (2, 512, 16, 64)
assert v.shape == (2, 512, 16, 64)
print(f"✓ k_nope:        {k_nope.shape}")
print(f"✓ v:             {v.shape}")

# Full MLA forward
out, _ = mla(x, attention_mask=mask.to(x.dtype))
assert out.shape == (2, 512, 2048), f"MLA output: {out.shape}"
print(f"✓ MLA output:    {out.shape}")

# ─── Section 4d: Dense MLP ───
mlp = MLP(dim=2048, inter_dim=5243)
mlp_out, aux = mlp(x)
assert mlp_out.shape == (2, 512, 2048), f"MLP: {mlp_out.shape}"
print(f"✓ Dense MLP:     {mlp_out.shape}")

# ─── Section 6c: MoE ───
moe = MoE(
    dim=2048, moe_inter_dim=768,
    n_routed_experts=64, n_activated_experts=8, n_shared_experts=2,
    n_expert_groups=8, n_limited_groups=4,
    score_func="sigmoid", route_scale=2.5, seq_aux_loss_alpha=0.0001,
)
moe.train()
moe_out, moe_aux = moe(x)
assert moe_out.shape == (2, 512, 2048), f"MoE: {moe_out.shape}"
print(f"✓ MoE output:    {moe_out.shape}")

# Gate shapes
x_flat = x.view(1024, 2048)
scores = torch.sigmoid(torch.nn.functional.linear(x_flat, moe.gate.weight))
assert scores.shape == (1024, 64), f"Gate scores: {scores.shape}"
print(f"✓ Gate scores:   {scores.shape}")

# ─── Section 8: LM Head ───
lm_head = nn.Linear(2048, 65536, bias=False)
logits = lm_head(x)
assert logits.shape == (2, 512, 65536), f"Logits: {logits.shape}"
print(f"✓ Logits:        {logits.shape}")

# ─── Section 9: Loss shapes ───
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = input_ids[:, 1:].contiguous()
assert shift_logits.shape == (2, 511, 65536)
assert shift_labels.shape == (2, 511)
print(f"✓ Shift logits:  {shift_logits.shape}")
print(f"✓ Shift labels:  {shift_labels.shape}")

# ─── Parameter count ───
print(f"\n{'='*70}")
print("Parameter Counts")
print(f"{'='*70}")
print(f"  embed_tokens:  {65536 * 2048:>15,}")
print(f"  lm_head:       {65536 * 2048:>15,}")
print(f"  MLA per layer: {sum(p.numel() for p in mla.parameters()):>15,}")
print(f"  Dense MLP:     {sum(p.numel() for p in mlp.parameters()):>15,}")
print(f"  MoE layer:     {sum(p.numel() for p in moe.parameters()):>15,}")

total_est = (
    2 * 65536 * 2048 +                    # embeddings
    16 * sum(p.numel() for p in mla.parameters()) +  # MLA
    16 * 2 * 2048 +                        # layernorms
    2 * sum(p.numel() for p in mlp.parameters()) +   # dense MLP
    14 * sum(p.numel() for p in moe.parameters()) +  # MoE
    2048                                    # final norm
)
print(f"  Estimated total (pre-MTP): {total_est:>12,}")

print(f"\n{'='*70}")
print("All shape assertions passed!")
print(f"{'='*70}")
```

---

## Section 15: Key Architectural Insights

### 15.1 Why MLA Works

The core insight is **information bottleneck theory**: not all heads need
independent K/V representations. By compressing to a shared 143-dim latent and
then expanding, MLA forces the model to learn a compact, shared representation
that all heads can use. The RoPE component (32 dims) is kept separate because
positional information must be preserved exactly.

**During inference**, only the 175-dim compressed representation is cached.
The expansion through `wkv_b` happens on-the-fly, which trades compute for
memory — exactly the right trade-off for autoregressive generation where memory
bandwidth is the bottleneck.

### 15.2 Why Sigmoid + Bias (Not Softmax)

Softmax routing forces expert scores to sum to 1, creating competition between
experts. This leads to unstable training because a small change in one expert's
score affects all others.

Sigmoid scoring (line 415) produces independent scores in (0, 1). The bias
adjustment (line 418) provides load balancing without an auxiliary loss that
distorts the main training objective. The bias is updated after each step by
`update_load_balance_bias()` (line 721):

```python
imbalance = (load - mean_load) / (mean_load + 1e-8)
self.gate.expert_bias.sub_(gamma * imbalance)
# Overloaded experts get negative bias → less likely to be selected
# Underloaded experts get positive bias → more likely to be selected
```

### 15.3 Token-Centric Dispatch Efficiency

The naive approach processes each expert separately, scanning all tokens:
```
for expert in experts:       # O(E)
    for token in tokens:     # O(N)
        if expert in token.assignments:
            process(token, expert)
# Total: O(E × N) = O(64 × 1024) = 65,536 iterations
```

Token-centric dispatch (line 485) sorts once and processes contiguously:
```
sort tokens by expert    # O(N×K × log(N×K))
for expert in experts:   # O(E)
    process(batch[expert])   # O(count_e) — contiguous memory!
# Total: O(N×K × log(N×K)) ≈ O(N×K × 13) for realistic sizes
# Plus: coalesced GPU memory access = massive bandwidth improvement
```

### 15.4 The MTP Training Signal

MTP provides an auxiliary training signal that forces hidden states to be
**predictive of multiple future tokens**, not just the immediate next one.
This has two benefits:

1. **Richer hidden representations**: The model must encode enough information
   to predict token i+1 (main loss) AND token i+2 (MTP). This is a stronger
   constraint that leads to better representations.

2. **Speculative decoding at inference**: The MTP module can draft 1 extra
   token in parallel with the main model, then verify it in a single forward
   pass. Hit rate of ~60-70% gives ~1.4× throughput improvement.

### 15.5 Residual Stream Architecture

Every layer follows the same pattern:
```
hidden = hidden + attention(layernorm(hidden))
hidden = hidden + ffn(layernorm(hidden))
```

This creates a **residual stream** where information flows directly from input
to output. Each layer adds its contribution but never destructively overwrites.
The gradient flows directly through the residual connections (gradient of
addition is 1), preventing vanishing gradients even in deep networks.

---

## Section 16: Training Step Timeline

Putting it all together, here is the complete timeline of a single training
step with our batch_size=2, seq_len=512 input:

```
Time →

1. FORWARD PASS (left to right through the model)
   ├─ Embedding lookup:                    ~0.1 ms
   ├─ Layer 0 (Dense):
   │   ├─ input_layernorm:                 ~0.05 ms
   │   ├─ MLA attention:                   ~0.5 ms
   │   ├─ residual add:                    ~0.01 ms
   │   ├─ post_attention_layernorm:        ~0.05 ms
   │   ├─ Dense MLP (SwiGLU):             ~0.3 ms
   │   └─ residual add:                    ~0.01 ms
   ├─ Layer 1 (Dense):                     ~0.9 ms (same as layer 0)
   ├─ Layers 2-15 (MoE × 14):
   │   ├─ MLA attention (same):            ~0.5 ms each
   │   ├─ MoE routing + dispatch:          ~2.0 ms each
   │   └─ Per layer total:                 ~3.0 ms each × 14 = ~42 ms
   ├─ Final norm:                          ~0.05 ms
   ├─ LM head:                             ~1.0 ms
   └─ Loss computation:                    ~0.5 ms
       ├─ Main CE loss:                    ~0.3 ms
       ├─ MTP forward + loss:              ~2.0 ms
       └─ Auxiliary loss aggregation:      ~0.1 ms

   Total forward: ~48 ms

2. BACKWARD PASS (right to left, ~2× forward)
   ├─ Loss gradients:                      ~0.5 ms
   ├─ MTP backward:                        ~4.0 ms
   ├─ LM head backward:                   ~2.0 ms
   ├─ Final norm backward:                ~0.1 ms
   ├─ Layers 15-2 (MoE × 14):
   │   ├─ MoE backward:                   ~4.0 ms each
   │   ├─ MLA backward:                   ~1.0 ms each
   │   └─ Per layer:                       ~5.5 ms × 14 = ~77 ms
   ├─ Layers 1-0 (Dense × 2):
   │   └─ Per layer:                       ~1.8 ms × 2 = ~3.6 ms
   └─ Embedding backward:                 ~0.2 ms

   Total backward: ~87 ms

3. OPTIMIZER STEP (AdamW)
   ├─ Gradient clipping (max_grad_norm=1.0): ~1.0 ms
   ├─ Adam update (all params):              ~5.0 ms
   └─ Load balance bias update:              ~0.1 ms

   Total optimizer: ~6 ms

TOTAL STEP: ~141 ms (for batch_size=2, seq_len=512 on single GPU)
```

**Note**: These timings are approximate and assume a single H100 GPU. Actual
timings depend on GPU utilization, memory bandwidth, kernel fusion, and other
factors. The MoE dispatch is the dominant cost due to the scatter/gather
operations and the sequential expert computation in the Python loop (a fused
CUDA kernel would be significantly faster).

---

## Section 17: Gradient Flow Visualization

```
loss
 │
 ├──── d_main_loss ──────────────────────────────────────────────────┐
 │      │                                                            │
 │      ▼                                                            │
 │   d_shift_logits [2,511,65536]                                    │
 │      │                                                            │
 │      ▼                                                            │
 │   lm_head.weight ◄── grad [65536,2048]                           │
 │      │                                                            │
 │      ▼ d_hidden [2,512,2048]                                      │
 │      │                                                            │
 │   norm.weight ◄── grad [2048]                                     │
 │      │                                                            │
 ├──── d_mtp_loss × 0.3 ────────────────┐                           │
 │                                       │                           │
 │   MTP grads ─► concat_proj, blocks,   │                           │
 │                embed_tokens, lm_head  │                           │
 │                       │               │                           │
 │                       ▼ d_main_hidden │                           │
 │                       │               │                           │
 │   ◄───────────────────┘               │                           │
 │                                       │                           │
 ├──── d_aux_loss (14 MoE layers) ──────── d_gate.weight per layer  │
 │                                                                   │
 │                                                                   │
 ▼                                                                   │
Layer 15 ◄── d_hidden + d_mtp_hidden ────────────────────────────────┘
 │
 ├─ d_residual_2 ──────────────────────────┐
 │                                          │
 ├─ MoE backward:                           │
 │   ├─ d_shared_experts (2 experts)        │
 │   ├─ d_routed_experts (8 per token)      │
 │   ├─ d_gate.weight                       │
 │   └─ d_ffn_input                         │
 │       │                                  │
 │       ▼                                  │
 │   d_post_attn_norm                       │
 │       │                                  │
 ├─ d_residual_1 ─── + ────────────────────┘
 │                    │
 ├─ MLA backward:     │
 │   ├─ d_wo          │
 │   ├─ d_attn_weights│
 │   ├─ d_v ──► d_wkv_b ──► d_kv_norm ──► d_wkv_a
 │   ├─ d_q ──► d_wq_b ──► d_q_norm  ──► d_wq_a
 │   └─ d_layer_input
 │       │
 │       ▼
 │   d_input_layernorm
 │       │
 ▼       │
Layer 14 ◄──── d_hidden (accumulated)
 │
 ... (repeat for all layers)
 │
 ▼
Layer 0
 │
 ▼
embed_tokens.weight ◄── grad [65536, 2048]
  (sparse: only 1024 rows of 65536 receive non-zero gradients)
```

---

## Section 18: Numerical Precision Considerations

### 18.1 Mixed Precision Strategy

The model operates in bf16 by default, with strategic fp32 upcasts:

| Operation | Precision | Reason |
|-----------|-----------|--------|
| Linear projections | bf16 | Speed; bf16 matmul on tensor cores |
| RMSNorm internals | fp32 | Line 196: `x = x.float()` — stability |
| Softmax in attention | fp32 | Line 362: `dtype=torch.float32` — exp overflow |
| RoPE (complex mul) | fp32 | Line 129: `.float()` — rotation precision |
| Loss (cross entropy) | fp32 | PyTorch default for CE |
| Gradient accumulation | bf16 | Matches parameter dtype |
| Optimizer states | fp32 | Adam moments need high precision |

### 18.2 Numerical Risks

1. **Attention scores** before masking can be large positive values. The
   `-inf` mask values overflow in fp16 but are handled correctly in bf16
   (bf16 represents ±inf natively).

2. **Expert routing scores** after sigmoid are in (0, 1). The `route_scale=2.5`
   multiplier amplifies them but they remain numerically stable.

3. **RMSNorm** divides by the RMS value, which can be very small for
   near-zero inputs. The `eps=1e-6` prevents division by zero.

---

*This document traces the complete forward and backward pass through NanoSeek-1B,
covering every tensor shape, every computation, and every gradient path. The
architecture implements DeepSeek V3's innovations (MLA, MoE with sigmoid routing
and bias-based load balancing, MTP, and DSA) at a research-feasible 1B-active /
4.86B-total parameter scale.*

*Source code: `model/model.py` (2038 lines), `model/config.py` (1409 lines).*
