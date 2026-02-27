# Multi-Token Prediction (MTP) — First Principles Deep Dive

> **Perspective**: Senior research scientist, large-scale language model team.
> **Codebase**: NanoSeek — a nano-scale educational replica of DeepSeek V3.2.
> **Audience**: Engineers and researchers who can read PyTorch and want to understand *why* every design choice exists, not just *what* the code does.

---

## Quick Context — solving the "one token at a time" bottleneck

Standard autoregressive language models predict a single next token per forward pass.
That constraint bites twice:

1. **Training signal density.** A sequence of length $L$ produces $L-1$ gradient
   updates, all sharing the same backbone activations. The last-layer LM head sees
   each hidden state only once. Every bit of supervision that could come from
   predicting $t{+}2, t{+}3, \ldots$ is thrown away.

2. **Inference latency.** Decoding is memory-bandwidth–bound. Each token requires a
   full KV-cache read, yet modern accelerators can compute far more FLOPs than the
   bandwidth wall allows. The GPU sits idle waiting for memory.

Multi-Token Prediction (MTP) addresses both problems with a single architectural
addition:

| Problem | MTP mechanism | DeepSeek V3 result |
|---------|--------------|-------------------|
| Sparse training signal | Extra cross-entropy losses on future tokens | Measurable perplexity improvement, especially on coding/math |
| Slow inference | Draft tokens from MTP modules enable speculative decoding | ~1.8× TPS (tokens per second) with `num_mtp_modules=1` |

The key insight from the DeepSeek V3 paper — and the thing NanoSeek faithfully
reproduces — is that these two benefits share *all* their parameters with the main
model (shared embeddings, shared LM head), so the incremental training cost is
small and the inference overhead for draft generation is near zero.

The rest of this document walks through the NanoSeek MTP implementation from
first principles, using the mandatory five-stage framework: **Inputs →
Architecture → Ground Truth → Loss → Outputs**.

---

## 🔴 STAGE 1: INPUTS — `main_hidden [B, L, 2048]` + labels for training

### 1.1 What flows into MTP

MTP does **not** see raw tokens directly. It receives two tensors from the main
model:

```
main_hidden : Tensor[B, L, hidden_size]   # final hidden states from the backbone
labels      : Tensor[B, L]                # token IDs (same labels used for main CE loss)
```

In NanoSeek these arrive inside `NanoSeekModel._compute_loss`:

```python
# model/model.py, inside _compute_loss
if self.mtp is not None:
    mtp_outputs = self.mtp(hidden_states, labels=labels)
```

`hidden_states` is the output of `self.norm(hidden_states)` — the post-LayerNorm
representation *after* all 16 decoder layers but *before* the LM head projection.
This is the richest representation the backbone can produce.

### 1.2 Why post-norm hidden states?

Using pre-LM-head hidden states (rather than, say, an intermediate layer) is
deliberate:

- These states already encode the full context of the sequence.
- The main model's `lm_head` is a linear map from `hidden_size → vocab_size`;
  MTP modules share that same head, so they must operate in the same
  representation space.
- Gradient signals from MTP flow back through the *entire* backbone, not just
  the top layers. This is what makes MTP an effective auxiliary training
  objective — it enriches the gradient for all parameters.

### 1.3 Shape walkthrough (NanoSeek-1B defaults)

| Tensor | Shape | Notes |
|--------|-------|-------|
| `main_hidden` | `[B, L, 2048]` | `hidden_size=2048` from config |
| `labels` | `[B, L]` | Same labels passed to main CE loss |
| `embed_tokens.weight` | `[65536, 2048]` | Shared embedding matrix |
| `lm_head.weight` | `[65536, 2048]` | Shared output projection |

At inference time, `labels` is `None` and MTP switches to speculative decoding
mode (Stage 5).

### 1.4 The embedding lookup for target tokens

Each MTP module needs the embedding of the *target* token it is trying to
predict. During training this is available via teacher forcing from `labels`:

```python
# model/model.py – MTPModule.forward
safe_target_tokens = torch.where(
    target_tokens < 0,
    torch.zeros_like(target_tokens),
    target_tokens,
)
token_embeds = self.embed_tokens(safe_target_tokens)
```

The `torch.where` guard handles the `ignore_index=-100` convention:
positions with `-100` in labels get their embedding replaced with the
embedding of token 0 (a safe index). The downstream CE loss will ignore
these positions via `ignore_index=-100`, so the garbage embedding does
not affect training. This is a standard but easy-to-miss detail — get it
wrong and you get an `IndexError` from `nn.Embedding`.

---

## 🔴 STAGE 2: MODEL ARCHITECTURE — the concatenation insight and MTP pipeline

This is the heart of the design. We will cover:

1. The concatenation insight (V3 vs V2 additive approach)
2. MTPModule internals
3. MTPBlock architecture with cross-attention
4. How it differs from simply adding a second LM head
5. Speculative decoding pipeline at inference
6. Full data-flow diagram

### 2.1 The concatenation insight — V3's key innovation over V2

DeepSeek V2 fused hidden states and token embeddings by **addition**:

```
h'ₖ = RMSNorm(prev_hidden) + RMSNorm(Emb(target_token))
```

DeepSeek V3 (and NanoSeek) uses **concatenation + linear projection**:

```
h'ₖ = Linear([RMSNorm(prev_hidden) ; RMSNorm(Emb(target_token))])
```

In code (`MTPModule.forward`):

```python
normed_hidden = self.hidden_norm(prev_hidden)       # RMSNorm
normed_embeds = self.embed_norm(token_embeds)        # RMSNorm

concatenated = torch.cat([normed_hidden, normed_embeds], dim=-1)
hidden_states = self.concat_proj(concatenated)       # Linear(2*H → H)
```

**Why concatenation beats addition:**

| Property | Addition | Concatenation + proj |
|----------|----------|---------------------|
| Information preservation | Lossy — vectors must share the same subspace | Lossless — the projection *learns* how to fuse |
| Capacity | Fixed; identical to input dim | Learnable; `concat_proj` has `2H × H` parameters |
| Gradient path | Shared gradient to both branches identically | Independent gradient contributions; `concat_proj` can weight them differently |
| Training stability | Sensitive to relative magnitudes | RMSNorm on each branch + learned projection isolates scales |

The intuition: addition forces the model to represent "previous context" and
"next token identity" in the *same* vector space dimension-for-dimension.
Concatenation gives the model a full `hidden_size` worth of extra dimensions
to keep the two signals distinct until the projection decides how to merge
them. At NanoSeek's `hidden_size=2048`, `concat_proj` is a `4096 → 2048`
matrix — 8.4M learnable parameters per MTP module dedicated entirely to
learning the optimal fusion strategy.

This is V3's critical innovation for MTP quality. It is a small parameter cost
for a meaningful improvement in how well the MTP module can condition on both
the context representation and the upcoming token identity.

### 2.2 MTPModule — the complete per-module architecture

Each `MTPModule` is a self-contained prediction unit. Here is the full
parameter inventory:

```python
class MTPModule(nn.Module):
    hidden_norm   : RMSNorm(input_size)              # normalize incoming hidden
    embed_norm    : RMSNorm(hidden_size)              # normalize token embedding
    concat_proj   : Linear(input_size + hidden_size → hidden_size)  # fusion
    blocks        : ModuleList[MTPBlock × num_blocks] # transformer block(s)
    output_norm   : RMSNorm(hidden_size)              # final normalization
    lm_head       : Linear(hidden_size → vocab_size)  # SHARED with main model
    embed_tokens  : Embedding(vocab_size, hidden_size) # SHARED with main model
```

Forward pass in detail:

```
prev_hidden [B, L, H]    target_tokens [B, L]
        │                         │
        ▼                         ▼
   hidden_norm              embed_tokens (shared)
        │                         │
        │                    embed_norm
        │                         │
        └────── cat(dim=-1) ──────┘
                    │
                    ▼
              [B, L, 2H]
                    │
               concat_proj
                    │
                    ▼
              [B, L, H]
                    │
          ┌─────────┤
          │   MTPBlock (cross_attn → self_attn → SwiGLU FFN)
          │         │
          │    (repeat for num_blocks, default=1)
          │         │
          └─────────┤
                    │
               output_norm
                    │
                    ▼
              [B, L, H]
                    │
                lm_head (shared)
                    │
                    ▼
           [B, L, vocab_size]  ← logits for this module's target position
```

### 2.3 MTPBlock — cross-attention to main hidden states

The `MTPBlock` is a single transformer-style block with three sub-layers:

```python
class MTPBlock(nn.Module):
    # Sub-layer 1: Cross-attention (query=MTP hidden, key/value=main hidden)
    input_norm  : RMSNorm
    cross_attn  : nn.MultiheadAttention

    # Sub-layer 2: Causal self-attention
    cross_norm  : RMSNorm
    self_attn   : nn.MultiheadAttention

    # Sub-layer 3: SwiGLU FFN
    ffn_norm    : RMSNorm
    gate_proj   : Linear(H → 4H)
    up_proj     : Linear(H → 4H)
    down_proj   : Linear(4H → H)
```

The forward pass:

```python
def forward(self, hidden_states, main_hidden, attention_mask, causal_mask):
    # 1. Cross-attention: "What did the main model see?"
    residual = hidden_states
    hidden_states = self.input_norm(hidden_states)
    cross_output, _ = self.cross_attn(
        query=hidden_states, key=main_hidden, value=main_hidden
    )
    hidden_states = residual + cross_output

    # 2. Causal self-attention: "What do I know from my own context?"
    residual = hidden_states
    hidden_states = self.cross_norm(hidden_states)
    self_output, _ = self.self_attn(
        query=hidden_states, key=hidden_states, value=hidden_states,
        attn_mask=causal_mask   # upper-triangular mask
    )
    hidden_states = residual + self_output

    # 3. SwiGLU FFN: "Process the combined information"
    residual = hidden_states
    hidden_states = self.ffn_norm(hidden_states)
    gate = F.silu(self.gate_proj(hidden_states))
    up = self.up_proj(hidden_states)
    hidden_states = self.down_proj(gate * up)
    hidden_states = residual + hidden_states

    return hidden_states
```

**Why cross-attention?** The MTP hidden states live in a separate "stream"
from the main model. Cross-attention lets them selectively attend to the
main model's representations at every position. Without cross-attention,
the MTP module would only know about the main model through the initial
concatenation — a single linear mix. Cross-attention gives it *query-dependent*
access: for each position, the MTP module can decide which parts of the main
model's representation are most relevant for predicting the next-next token.

**Why causal self-attention?** Even though MTP modules are lightweight (typically
`num_blocks=1`), the self-attention allows information to flow between positions
within the MTP stream. This is critical for maintaining autoregressive
consistency — position $i$ must not peek at information from position $i{+}1$.

**Why SwiGLU?** Consistency with the main model's FFN architecture. The gated
structure (`silu(gate) * up`) provides better gradient flow than vanilla ReLU
FFN, and the same activation function means the MTP module operates in a
similar nonlinear regime as the backbone.

### 2.4 How MTP differs from "just add another LM head"

A naive approach to multi-token prediction would be:

```python
# WRONG: naive approach
lm_head_t1 = Linear(H, V)  # predict t+1 (already exists)
lm_head_t2 = Linear(H, V)  # predict t+2 (just add this!)
```

This fails badly because:

1. **The hidden state at position $i$ is optimized to predict token $i{+}1$.**
   Predicting $i{+}2$ from the same hidden state requires the model to
   simultaneously encode information for both targets — a conflicting objective
   that degrades the representation for the primary task.

2. **No access to the identity of token $i{+}1$.** Predicting $i{+}2$
   *without knowing* what token $i{+}1$ is requires marginalizing over all
   possible $i{+}1$ tokens — an exponentially hard task.

MTP solves both problems:

- The **concatenation with `Emb(t_{i+1})`** provides the identity of the
  intervening token directly (teacher-forced during training).
- The **separate MTPBlock** transforms the hidden state specifically for the
  $t{+}2$ prediction task, so the backbone's representations are not
  distorted.
- The **cross-attention** allows the MTP module to selectively re-read the
  backbone's representations rather than relying on a single fixed vector.

### 2.5 The MultiTokenPrediction orchestrator

`MultiTokenPrediction` is the container that manages one or more `MTPModule`
instances and chains them sequentially:

```python
class MultiTokenPrediction(nn.Module):
    mtp_modules : ModuleList[MTPModule × num_mtp_modules]
    mtp_loss_weight : float   # λ in total_loss = main_loss + λ * mtp_loss
    mtp_loss_decay  : float   # γ for weighting deeper modules
```

The chaining logic is the clever part. For `num_mtp_modules=1`:

- Module 0 receives `main_hidden` as `prev_hidden`, and `labels[:, 1:]` as
  `target_tokens` (the token at $t{+}1$, which is the one the main model
  predicts).
- Module 0 produces logits that are evaluated against `labels[:, 2:]` — the
  token at $t{+}2$.

For `num_mtp_modules=2`, the chain extends:

- Module 0 as above, producing `hidden_0`.
- Module 1 receives `hidden_0` as `prev_hidden`, `labels[:, 2:]` as
  `target_tokens`, and produces logits evaluated against `labels[:, 3:]`.

This is implemented in `MultiTokenPrediction.forward`:

```python
prev_hidden = main_hidden
for i, module in enumerate(self.mtp_modules):
    token_offset = i + 1
    target_tokens = labels[:, token_offset:]
    current_hidden = prev_hidden[:, :effective_len]
    current_main = main_hidden[:, :effective_len]

    logits, hidden = module(
        prev_hidden=current_hidden,
        target_tokens=target_tokens,
        main_hidden=current_main,
    )

    # Loss computed against labels[:, i+2 : i+2+pred_len]
    pred_offset = i + 2
    shift_logits = logits[:, :pred_len]
    shift_labels = labels[:, pred_offset : pred_offset + pred_len]
    loss = F.cross_entropy(shift_logits.view(-1, V), shift_labels.view(-1))

    prev_hidden = hidden   # chain to next module
```

**Key detail:** each module's `prev_hidden` comes from the *output* of the
previous module, but `main_hidden` always references the *backbone's* hidden
states. This means every module has cross-attention access to the same
backbone representations, even though the module-local hidden states evolve
through the chain.

### 2.6 Speculative decoding at inference

At inference time, MTP modules switch roles: instead of computing losses,
they generate **draft tokens** for speculative decoding.

```python
def speculative_decode(self, main_hidden, first_token=None, temperature=0.0):
    draft_tokens = []
    draft_probs  = []
    prev_hidden  = main_hidden
    prev_token   = first_token.unsqueeze(-1) if first_token is not None else None

    for module in self.mtp_modules:
        logits, hidden = module(
            prev_hidden=prev_hidden,
            target_tokens=prev_token,
            main_hidden=main_hidden,
        )

        if temperature == 0.0:
            # Greedy: pick argmax
            probs = F.softmax(logits[:, -1], dim=-1)
            token = logits[:, -1].argmax(dim=-1)
        else:
            # Sampling with optional top-k
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)
            token = torch.multinomial(probs, num_samples=1).squeeze(-1)

        prob = probs.gather(-1, token.unsqueeze(-1)).squeeze(-1)
        draft_tokens.append(token)
        draft_probs.append(prob)

        prev_hidden = hidden
        prev_token = token.unsqueeze(-1)

    return torch.stack(draft_tokens, dim=1), torch.stack(draft_probs, dim=1)
```

The speculative decoding protocol:

1. Main model produces `main_hidden` and its own next-token prediction ($t{+}1$).
2. MTP module 0 takes `main_hidden` + embedding of the predicted $t{+}1$ →
   produces draft of $t{+}2$.
3. (If more modules) Module 1 takes module 0's hidden + embedding of draft
   $t{+}2$ → produces draft of $t{+}3$.
4. The main model verifies all draft tokens in a single forward pass (they
   can be batch-verified because the main model is autoregressive — it just
   needs to extend the KV cache).
5. Accepted tokens are committed; rejected tokens cause a rollback.

With `num_mtp_modules=1`, each speculation proposes 1 draft token. The
acceptance rate determines the effective speedup. DeepSeek V3 reports ~1.8×
TPS improvement in production — this means roughly 40-50% of draft tokens
are accepted on average.

### 2.7 Full data-flow diagram (ASCII)

```
╔═══════════════════════════════════════════════════════════════════════╗
║                        NANOSEEK MODEL                                ║
║                                                                       ║
║  input_ids [B, L]                                                     ║
║      │                                                                ║
║      ▼                                                                ║
║  embed_tokens ─────────────────────────────────────────┐              ║
║      │                                                 │ (shared)     ║
║      ▼                                                 │              ║
║  ┌─────────────────────────────────┐                   │              ║
║  │   Decoder Layers × 16          │                   │              ║
║  │   (MLA + MoE/Dense FFN)        │                   │              ║
║  └─────────────────────────────────┘                   │              ║
║      │                                                 │              ║
║      ▼                                                 │              ║
║  self.norm(hidden_states)                              │              ║
║      │                                                 │              ║
║      ├──────────────────────────┐                      │              ║
║      │                          │                      │              ║
║      ▼                          ▼                      │              ║
║  lm_head ──► logits[t+1]    MTP System                │              ║
║  (shared)    (main loss)        │                      │              ║
║      │                          │                      │              ║
║      │         ┌────────────────┼──────────────────────┘              ║
║      │         │                │                                     ║
║      │         ▼                ▼                                     ║
║      │    ┌─────────────────────────────────────┐                    ║
║      │    │         MTP MODULE 0                │                    ║
║      │    │                                     │                    ║
║      │    │  prev_hidden ──► hidden_norm         │                    ║
║      │    │                      │               │                    ║
║      │    │  labels[:,1:] ──► embed_tokens(shared)│                   ║
║      │    │                  ──► embed_norm       │                    ║
║      │    │                      │               │                    ║
║      │    │         cat([normed_hidden,           │                    ║
║      │    │               normed_embeds], dim=-1) │                    ║
║      │    │                      │               │                    ║
║      │    │                concat_proj            │                    ║
║      │    │              (4096 → 2048)            │                    ║
║      │    │                      │               │                    ║
║      │    │                 MTPBlock              │                    ║
║      │    │          ┌───────────┤               │                    ║
║      │    │          │  cross_attn(q=mtp,        │                    ║
║      │    │          │    k=main_hidden,          │                    ║
║      │    │          │    v=main_hidden)          │                    ║
║      │    │          │       │                   │                    ║
║      │    │          │  self_attn(causal)         │                    ║
║      │    │          │       │                   │                    ║
║      │    │          │  SwiGLU FFN                │                    ║
║      │    │          └───────────┤               │                    ║
║      │    │                      │               │                    ║
║      │    │                output_norm            │                    ║
║      │    │                      │               │                    ║
║      │    │                lm_head (shared)       │                    ║
║      │    │                      │               │                    ║
║      │    │                      ▼               │                    ║
║      │    │             logits[t+2]               │                    ║
║      │    │             (mtp_loss_0)              │                    ║
║      │    │                                     │                    ║
║      │    │  hidden_0 ──► [chain to module 1     │                    ║
║      │    │                 if num_modules > 1]  │                    ║
║      │    └─────────────────────────────────────┘                    ║
║      │                                                                ║
║      ▼                                                                ║
║  total_loss = main_loss + λ × mtp_loss                               ║
╚═══════════════════════════════════════════════════════════════════════╝
```

---

## 🔴 STAGE 3: GROUND TRUTH — shifted labels (`t+2`, `t+3`, etc.)

### 3.1 The shifting arithmetic

Standard next-token prediction uses:

```
logits at position i  →  target = labels[i+1]
```

MTP module $k$ (0-indexed) predicts further ahead:

```
logits at position i  →  target = labels[i + k + 2]
```

This is because:
- The main model predicts $t{+}1$ (offset 1)
- MTP module 0 predicts $t{+}2$ (offset 2)
- MTP module 1 predicts $t{+}3$ (offset 3)
- ...in general, module $k$ predicts $t{+}k{+}2$

In `MultiTokenPrediction.forward`:

```python
for i, module in enumerate(self.mtp_modules):
    token_offset = i + 1      # target_tokens = labels shifted by this
    pred_offset  = i + 2      # CE target = labels shifted by this

    target_tokens = labels[:, token_offset:]       # feed Emb(t_{i+1})
    shift_logits  = logits[:, :pred_len]
    shift_labels  = labels[:, pred_offset : pred_offset + pred_len]
```

### 3.2 Why the sequence gets shorter

Each additional prediction offset reduces the usable sequence length by 1:

| Component | Logits positions | Target positions | Usable length |
|-----------|-----------------|-----------------|---------------|
| Main model | `[0, L-2]` | `labels[1, L-1]` | `L - 1` |
| MTP module 0 | `[0, L-3]` | `labels[2, L-1]` | `L - 2` |
| MTP module 1 | `[0, L-4]` | `labels[3, L-1]` | `L - 3` |

With `L=4096` and `num_mtp_modules=1`, the MTP module sees 4094 positions
vs the main model's 4095. This is a negligible reduction (0.02%) but must
be handled correctly in the slicing logic. The code does this with:

```python
effective_len = target_tokens.size(1)
current_hidden = prev_hidden[:, :effective_len]
```

### 3.3 The `target_tokens` vs `shift_labels` distinction

This is a common source of confusion, so let's be explicit:

- **`target_tokens`** = the tokens whose *embeddings* are fed into the MTP
  module as input. For module 0, this is `labels[:, 1:]` — the token at
  $t{+}1$. This is the token the *main model* predicts, and the MTP module
  uses its embedding to condition its own prediction.

- **`shift_labels`** = the tokens the MTP module's *output logits* are
  evaluated against. For module 0, this is `labels[:, 2:]` — the token at
  $t{+}2$. This is the MTP module's actual prediction target.

In other words: `target_tokens` is **input** (teacher forcing), `shift_labels`
is **ground truth** (loss target). They are offset by exactly 1 position
from each other.

### 3.4 Handling padding and ignore indices

The `ignore_index=-100` convention propagates naturally:

1. In the embedding lookup, `-100` tokens are replaced with token 0
   (safe index) via `torch.where`.
2. In the CE loss, `ignore_index=-100` causes those positions to be
   excluded from the loss and gradient computation.
3. This means padded sequences "just work" — pad tokens labeled `-100`
   are ignored in both the embedding input and the loss target.

---

## 🔴 STAGE 4: LOSS FUNCTION — dynamic weighting, decay, and gradient flow

### 4.1 The complete loss equation

NanoSeek's total training loss is:

```
total_loss = main_loss + λ(t) × mtp_loss + aux_losses
```

Where:

- `main_loss` = cross-entropy of main LM head logits vs `labels[:, 1:]`
- `mtp_loss` = weighted combination of per-module CE losses
- `aux_losses` = MoE sequence-level auxiliary loss + DSA indexer loss (outside MTP scope)

The MTP loss itself is a weighted average across modules:

```
                   Σᵢ (γⁱ × CE_loss_module_i)
    mtp_loss  =  ─────────────────────────────
                       Σᵢ γⁱ
```

Where $\gamma$ = `mtp_loss_decay` = 0.8.

In code:

```python
for i, module in enumerate(self.mtp_modules):
    loss = F.cross_entropy(shift_logits.view(-1, V), shift_labels.view(-1),
                           ignore_index=-100)
    weight = self.mtp_loss_decay ** i     # γ⁰=1.0, γ¹=0.8, γ²=0.64, ...
    total_loss += weight * loss

weight_sum = sum(self.mtp_loss_decay ** i for i in range(num_modules))
results["mtp_loss"] = total_loss / weight_sum
```

### 4.2 The decay factor across modules

With `mtp_loss_decay=0.8` and `num_mtp_modules=1`:

| Module | Predicts | Raw weight | Normalized weight |
|--------|----------|-----------|-------------------|
| 0 | $t{+}2$ | $0.8^0 = 1.0$ | $1.0 / 1.0 = 1.0$ |

With hypothetical `num_mtp_modules=3`:

| Module | Predicts | Raw weight | Normalized weight |
|--------|----------|-----------|-------------------|
| 0 | $t{+}2$ | $0.8^0 = 1.0$ | $1.0 / 2.44 = 0.41$ |
| 1 | $t{+}3$ | $0.8^1 = 0.8$ | $0.8 / 2.44 = 0.33$ |
| 2 | $t{+}4$ | $0.8^2 = 0.64$ | $0.64 / 2.44 = 0.26$ |

**Why decay?** Predicting further-ahead tokens is inherently harder and
noisier. Without decay, the loss from module 2 (predicting $t{+}4$) would
dominate gradients during early training when all modules have similar
loss magnitudes, pulling the backbone toward optimizing for long-range
prediction at the expense of the primary $t{+}1$ task. Decay ensures
the near-term predictions receive stronger gradient signal.

### 4.3 Dynamic weight schedule: λ = 0.3 → 0.1

The MTP loss weight $\lambda$ follows a two-phase schedule:

```
λ(t) = 0.3    if tokens_processed < 0.60 × total_tokens
λ(t) = 0.1    if tokens_processed ≥ 0.60 × total_tokens
```

In code (`NanoSeekModel.get_mtp_loss_weight`):

```python
def get_mtp_loss_weight(self, tokens_processed=None):
    if tokens_processed is None:
        tokens_processed = self.tokens_processed.item()
    transition = int(self.config.total_tokens *
                     self.config.mtp.mtp_loss_transition_ratio)
    if tokens_processed < transition:
        return self.config.mtp.mtp_loss_weight_initial    # 0.3
    return self.config.mtp.mtp_loss_weight_final          # 0.1
```

And the final combination in `_compute_loss`:

```python
mtp_weight = self.get_mtp_loss_weight()
total_loss = main_loss + mtp_weight * mtp_loss
```

**Why this schedule?**

The rationale comes directly from DeepSeek V3's training observations:

1. **Early training (λ=0.3, first 60%)**: The backbone is still forming
   basic representations. The stronger MTP signal forces it to build
   representations that encode not just the immediate next token but also
   longer-range structure. This is like a regularizer — it prevents the
   backbone from taking shortcuts that only work for $t{+}1$ prediction.

2. **Late training (λ=0.1, final 40%)**: The backbone has converged on
   strong representations. Now the primary $t{+}1$ task should dominate
   fine-tuning. Reducing λ prevents the MTP auxiliary task from
   interfering with the final optimization of the main objective.

3. **Why not a smooth decay?** A step schedule is simpler to implement,
   debug, and reproduce. DeepSeek V3 experimented with smooth schedules
   and found the step schedule performed comparably with less
   implementation complexity. The exact transition ratio (0.60 for
   NanoSeek, ~0.67 for V3) is not critical — what matters is that the
   transition happens after the backbone has learned good representations
   but before the final fine-tuning phase.

### 4.4 How gradients flow through shared embeddings

This is the most subtle and important aspect of MTP's training dynamics.

The shared embedding matrix `embed_tokens.weight` receives gradients from
**three** sources:

```
                    ┌─── Main model input embedding
                    │    (forward pass: tokens → hidden states)
                    │
embed_tokens.weight ├─── MTP target embedding
                    │    (forward pass: target tokens → normed_embeds → concat)
                    │
                    └─── LM head (tied weights, if tie_word_embeddings=True)
                         (backward from both main CE loss and MTP CE loss)
```

And the `lm_head.weight` (shared via `tie_word_embeddings` when hidden sizes
match) receives gradients from:

```
                 ┌─── Main model CE loss (logits → main_loss)
lm_head.weight ──┤
                 └─── MTP module CE loss (logits → mtp_loss)
                      (one gradient contribution per MTP module)
```

**The gradient scaling concern:**

With `num_mtp_modules=1` and `mtp_loss_weight=0.3`, the `lm_head` receives
1.3× the gradient it would without MTP (1.0 from main + 0.3 from MTP).
This is generally fine because:

- The effective learning rate for `lm_head` is slightly higher, which is
  offset by the fact that MTP gradients point in a consistent direction
  (predicting the next-next token uses similar vocabulary distributions).
- The `max_grad_norm=1.0` gradient clipping in the optimizer caps the
  total gradient magnitude regardless of the number of loss sources.

However, if you scale to `num_mtp_modules=4` with `mtp_loss_weight=0.3`,
the `lm_head` gradient could be up to 2.2× the base level. This is why
the decay factor and the λ reduction are both important — they bound the
total MTP gradient contribution.

**Production gotcha — shared embedding gradient scaling:**

When `tie_word_embeddings=False` (NanoSeek's default for the 1B config),
`embed_tokens` and `lm_head` are separate parameter tensors, so each
accumulates gradients independently. The `embed_tokens` tensor still gets
double-duty gradients (from main input embedding + MTP target embedding),
but this is a smaller effect since the MTP embedding gradient is scaled
by the concatenation projection.

When `tie_word_embeddings=True` (common for MTP modules where
`mtp_hidden_size == hidden_size`), the gradients from all sources
accumulate on a single tensor. Monitor the gradient norm of this
tensor during training — if it's significantly larger than other
parameters' gradient norms, consider reducing `mtp_loss_weight` or
adding a separate learning rate group for shared embeddings.

### 4.5 Per-module loss tracking

The implementation tracks per-module losses for monitoring:

```python
per_module_loss = []
for i, module in enumerate(self.mtp_modules):
    ...
    per_module_loss.append(loss.item())

results["per_module_loss"] = per_module_loss
```

In a healthy training run, you should observe:

- **Module 0 loss** decreasing steadily, tracking slightly above the main loss
  (predicting $t{+}2$ is harder than $t{+}1$).
- **Module 1 loss** (if present) consistently higher than module 0, but still
  decreasing.
- **Convergence gap** between modules roughly proportional to the prediction
  offset — module 1's loss plateau is higher than module 0's.

If module losses *increase* while the main loss decreases, the MTP weight
schedule is likely wrong (too high λ too late) or the MTP modules are
underpowered (too few heads, too small intermediate size).

---

## 🔴 STAGE 5: OUTPUTS — what MTP produces

### 5.1 Training outputs

During training, `MultiTokenPrediction.forward` returns a dictionary:

```python
{
    "mtp_logits":     List[Tensor[B, L', V]],   # per-module logits
    "mtp_hidden":     List[Tensor[B, L', H]],   # per-module hidden states
    "mtp_loss":       Tensor[],                   # scalar, weighted combination
    "per_module_loss": List[float],               # raw CE per module
}
```

These are consumed by `NanoSeekModel._compute_loss`:

```python
mtp_loss = mtp_outputs.get("mtp_loss", 0.0)
mtp_weight = self.get_mtp_loss_weight()
total_loss = main_loss + mtp_weight * mtp_loss
```

### 5.2 Inference outputs (speculative decoding)

During inference, `speculative_decode` returns:

```python
draft_tokens : Tensor[B, num_mtp_modules]   # proposed token IDs
draft_probs  : Tensor[B, num_mtp_modules]   # probability of each draft
```

With `num_mtp_modules=1`, this means exactly 1 draft token per speculation
step. The verification protocol:

1. Main model predicts token $A$ at position $t{+}1$.
2. MTP proposes token $B$ as draft for position $t{+}2$.
3. Main model runs a forward pass with both $A$ and $B$ appended to the
   KV cache, producing logits for positions $t{+}2$ and $t{+}3$.
4. If main model agrees with $B$ at position $t{+}2$ → accept both tokens,
   advance by 2.
5. If main model disagrees → accept only $A$, discard $B$, advance by 1.

The `draft_probs` tensor is used for rejection sampling when
`temperature > 0`: the acceptance criterion is
$\min(1, p_{\text{main}}(B) / p_{\text{draft}}(B))$, ensuring the final
distribution matches the main model exactly.

### 5.3 Output shape walkthrough

For `B=2, L=4096, hidden_size=2048, vocab_size=65536, num_mtp_modules=1`:

| Output | Shape | Description |
|--------|-------|-------------|
| `mtp_logits[0]` | `[2, 4094, 65536]` | Module 0 logits over vocabulary |
| `mtp_hidden[0]` | `[2, 4094, 2048]` | Module 0 hidden states |
| `mtp_loss` | `[]` (scalar) | Weighted average of module losses |
| `per_module_loss` | `[float]` (length 1) | Raw CE for module 0 |

For speculative decoding with `B=1`:

| Output | Shape | Description |
|--------|-------|-------------|
| `draft_tokens` | `[1, 1]` | One draft token per batch element |
| `draft_probs` | `[1, 1]` | Probability of the draft token |

---

## Common Misconceptions

### Misconception 1: "MTP = parallel decoding"

**Wrong.** MTP modules predict tokens *sequentially*, not in parallel. Module 0
must complete before module 1 can start (because module 1 needs module 0's
hidden state as input). The speedup comes from speculative decoding — proposing
multiple tokens that the main model verifies in a single batched forward pass.

The main model *does* verify in parallel (it processes the draft sequence in
one forward pass), but the MTP modules themselves are sequential. This is a
fundamental difference from approaches like Medusa (which predicts all future
tokens from the same hidden state independently).

### Misconception 2: "MTP is just beam search with extra heads"

**Wrong.** Beam search explores multiple *hypotheses* for the same token
position. MTP predicts different *positions* — each module is responsible
for exactly one future position. There is no search over alternatives; each
module produces a single greedy/sampled prediction.

### Misconception 3: "MTP adds significant training cost"

**Partially wrong.** The MTP modules share the expensive components
(embeddings, LM head) with the main model. The incremental cost is:

- `concat_proj`: $2H \times H = 2 \times 2048^2 = 8.4\text{M}$ params
- `MTPBlock`: $\sim 4 \times H \times H \times 3 = 50\text{M}$ params (cross-attn + self-attn + FFN)
- Norms: negligible

Total: ~58M parameters per module, or about 5% of the 1.08B active params.
The compute cost is proportional, so MTP adds roughly 5-10% to training
FLOP per step. This is a small price for the dual benefit of better training
signal and inference speedup.

### Misconception 4: "MTP modules can be dropped at inference to save memory"

**Correct, but with a caveat.** If you don't need speculative decoding, you
can remove MTP modules entirely at inference. The main model's predictions
are unchanged. However, if you trained with `mtp_loss_weight=0.3`, the
backbone has been optimized with MTP gradients throughout training — the
representations are *different* from what you'd get without MTP. You cannot
remove MTP from training and expect the same model quality; you can only
remove it from inference.

### Misconception 5: "More MTP modules = better"

**Wrong in practice.** DeepSeek V3 uses `num_mtp_modules=1` (predicting
$t{+}2$ only). Adding more modules has diminishing returns:

- Each additional module's loss is noisier (predicting further ahead is
  inherently harder).
- The sequential dependency means more inference latency for draft
  generation.
- The gradient contribution is geometrically decayed (0.8× per module),
  so deep modules barely affect training.

The sweet spot from DeepSeek's experiments: 1 module. It provides the
training signal benefit without meaningful inference overhead.

---

## Production Gotchas

### Gotcha 1: Shared embedding gradient scaling

As discussed in Stage 4, `embed_tokens.weight` accumulates gradients from
multiple sources. In practice, monitor:

```python
# During training
embed_grad_norm = model.embed_tokens.weight.grad.norm().item()
other_grad_norm = sum(p.grad.norm().item()**2
                      for n, p in model.named_parameters()
                      if 'embed' not in n and p.grad is not None) ** 0.5
ratio = embed_grad_norm / other_grad_norm
# If ratio > 3.0, consider reducing mtp_loss_weight
```

### Gotcha 2: Sequence length interaction with MTP offsets

With `num_mtp_modules=1`, you lose 2 positions from your effective sequence
length (1 for main model shift, 1 for MTP shift). At `seq_len=4096` this
is negligible. But if you train with `seq_len=64` (e.g., for debugging) and
`num_mtp_modules=4`, you lose 5 positions — nearly 8% of your sequence.
Your effective batch size in tokens is reduced accordingly. Always verify
that `seq_len > num_mtp_modules + 2` in your config.

### Gotcha 3: The `ignore_index` guard must come before embedding lookup

The `torch.where` guard in `MTPModule.forward`:

```python
safe_target_tokens = torch.where(
    target_tokens < 0,
    torch.zeros_like(target_tokens),
    target_tokens,
)
```

This *must* happen before `self.embed_tokens(safe_target_tokens)`. If you
accidentally pass `-100` to `nn.Embedding`, you get an `IndexError`. The
replacement with `0` is harmless because the downstream CE loss uses
`ignore_index=-100` to exclude those positions.

### Gotcha 4: MTP module hidden state dimensionality

If `mtp_hidden_size != hidden_size`, the code disables weight tying:

```python
if tie_word_embeddings and mtp_hidden_size != hidden_size:
    tie_word_embeddings = False
```

This creates separate `embed_tokens` and `lm_head` for the MTP module,
significantly increasing parameter count. In NanoSeek, the default config
sets `mtp_hidden_size=None` which resolves to `hidden_size`, keeping
weight tying active. If you customize `mtp_hidden_size`, be aware of
this.

### Gotcha 5: Cross-attention mask propagation

The MTP block's cross-attention uses `key_padding_mask` derived from the
attention mask:

```python
if attention_mask is not None:
    if attention_mask.dtype == torch.bool:
        key_padding_mask = ~attention_mask
    else:
        key_padding_mask = attention_mask == 0
```

This means padded positions in the main model's hidden states are correctly
masked in the MTP cross-attention. But note: this is a *padding* mask, not
a *causal* mask. Cross-attention from MTP to the main model is intentionally
non-causal — the MTP module at position $i$ can attend to main model hidden
states at *all* positions (including $i{+}1$, $i{+}2$, etc.) because the
main model has already processed those positions in a causal manner. The
causal constraint is enforced only in the MTP self-attention.

### Gotcha 6: Speculative decoding requires `eval()` mode

The `speculative_decode` method should only be called in inference mode. If
called during training (with `model.training=True`), dropout layers in the
MTP block will be active, introducing noise into the draft tokens and
reducing acceptance rates. Always call `model.eval()` before speculative
decoding.

---

## 2026 Best Practices

### Best Practice 1: Start with `num_mtp_modules=1`

DeepSeek V3's production config uses 1 module. Start there. Only add more
if you have empirical evidence (via controlled ablation) that additional
modules improve your specific task distribution.

### Best Practice 2: Use the dynamic weight schedule

The λ=0.3→0.1 schedule is well-tested. If you're training a different model
size, keep the ratio (0.60 transition point) but consider adjusting the
absolute weights:

- Smaller models (< 500M active): try λ=0.2→0.05 (smaller models are more
  sensitive to auxiliary task interference).
- Larger models (> 10B active): λ=0.3→0.1 works well; larger models have
  more capacity to handle the multi-task objective.

### Best Practice 3: Monitor per-module loss curves

Add per-module loss logging to your training loop. The module 0 loss should
track ~0.2-0.5 nats above the main loss. If the gap is larger, the MTP
module may be underpowered. If the gap is near zero, the MTP module may
be too powerful and should have its capacity reduced.

### Best Practice 4: Leverage MTP for curriculum detection

MTP loss is a natural measure of local predictability. Sequences where
`mtp_loss >> main_loss` contain tokens that are individually predictable
but sequentially surprising — a signal that the model struggles with
long-range dependencies. Use this signal for curriculum learning: upweight
these sequences in later training phases.

### Best Practice 5: Profile speculative decoding acceptance rates

Track `draft_acceptance_rate = accepted_drafts / total_drafts` during
inference. If acceptance rate < 30%, speculative decoding provides minimal
speedup and may not be worth the extra memory for MTP module parameters.
Common causes of low acceptance:

- Distribution mismatch between training and inference (domain shift).
- Temperature too high (high-entropy distributions are harder to match).
- Model undertrained (MTP modules haven't converged).

### Best Practice 6: Shared embeddings are non-negotiable

Always share `embed_tokens` and `lm_head` between the main model and MTP
modules (when dimensions match). The benefits:

- 65536 × 2048 × 2 = 268M parameters saved per module.
- Consistent vocabulary representation across all prediction heads.
- Gradient signal from MTP enriches the shared embedding quality.

The only exception is when `mtp_hidden_size != hidden_size`, which NanoSeek
handles by automatically disabling weight tying.

### Best Practice 7: Concatenation, not addition

If implementing MTP from scratch, use the V3 concatenation approach:

```python
# ✅ V3 approach (NanoSeek)
concat = torch.cat([norm(hidden), norm(embed)], dim=-1)
fused = linear_proj(concat)

# ❌ V2 approach (deprecated)
fused = norm(hidden) + norm(embed)
```

The linear projection gives the model a learnable fusion strategy. The
parameter cost (8.4M for `hidden_size=2048`) is negligible compared to
the quality improvement.

---

## Summary: MTP in One Diagram

```
   TRAINING                                    INFERENCE
   ════════                                    ═════════

   main_hidden ──┐                             main_hidden ──┐
                 │                                            │
   Emb(t+1) ────┤                             Emb(predicted  │
   (teacher      │                              t+1) ────────┤
    forced)      │                                            │
                 ▼                                            ▼
         ┌──────────────┐                           ┌──────────────┐
         │  RMSNorm     │                           │  RMSNorm     │
         │  RMSNorm     │                           │  RMSNorm     │
         │  cat + proj  │                           │  cat + proj  │
         │  MTPBlock    │                           │  MTPBlock    │
         │  RMSNorm     │                           │  RMSNorm     │
         │  lm_head     │                           │  lm_head     │
         └──────┬───────┘                           └──────┬───────┘
                │                                          │
                ▼                                          ▼
         logits for t+2                              draft token t+2
                │                                    + probability
                ▼                                          │
         CE(logits, t+2)                                   ▼
                │                                   Main model verifies
                ▼                                   accept / reject
         mtp_loss × λ(t)
                │
                ▼
         total_loss = main_loss + λ × mtp_loss
```

---

## Appendix A: Parameter Count Breakdown for MTP

For NanoSeek-1B defaults (`hidden_size=2048`, `mtp_num_heads=8`,
`intermediate_size=8192` i.e., `4 × 2048`):

| Component | Parameters | Formula |
|-----------|-----------|---------|
| `hidden_norm` | 2,048 | `hidden_size` (RMSNorm weight) |
| `embed_norm` | 2,048 | `hidden_size` |
| `concat_proj` | 8,390,656 | `(2 × hidden_size) × hidden_size` |
| `cross_attn` (Q,K,V,O) | 16,781,312 | `4 × hidden_size²` |
| `self_attn` (Q,K,V,O) | 16,781,312 | `4 × hidden_size²` |
| `gate_proj` | 16,777,216 | `hidden_size × 4 × hidden_size` |
| `up_proj` | 16,777,216 | `hidden_size × 4 × hidden_size` |
| `down_proj` | 16,777,216 | `4 × hidden_size × hidden_size` |
| `output_norm` | 2,048 | `hidden_size` |
| Block norms (×3) | 6,144 | `3 × hidden_size` |
| `embed_tokens` | **shared** | 0 (tied) |
| `lm_head` | **shared** | 0 (tied) |
| **Total per module** | **~92M** | |

With `num_mtp_modules=1`, this is ~92M incremental parameters on top of
the ~1.08B active base — about 8.5% overhead.

## Appendix B: Config Reference

From `model/config.py` → `MTPConfig`:

```python
@dataclass
class MTPConfig:
    num_mtp_modules: int = 1
    mtp_loss_weight_initial: float = 0.3
    mtp_loss_weight_final: float = 0.1
    mtp_loss_transition_ratio: float = 0.60
    mtp_loss_decay: float = 0.8
    mtp_hidden_size: Optional[int] = None   # defaults to hidden_size
    mtp_num_heads: int = 8
    speculative_draft_tokens: int = 2
    speculative_temperature: float = 0.0
```

## Appendix C: Test Coverage

MTP is validated by 16 tests in `tests/test_mtp.py`:

| Test class | Tests | What it validates |
|-----------|-------|------------------|
| `TestMTPTokenAlignment` | 2 | Logit shapes, correct prediction positions |
| `TestMTPLossSchedule` | 3 | λ=0.3 initial, λ=0.1 final, transition boundary |
| `TestMTPLossComputation` | 3 | Loss returned, decay across modules, ignore_index handling |
| `TestMTPSpeculativeDecoding` | 4 | Output shapes, valid tokens, greedy probs, temperature effects |
| `TestMTPEmbeddingSharing` | 2 | Shared embed/lm_head objects, gradient flow through shared weights |
| `TestMTPBlock` | 2 | Forward pass shapes, gradient health |

Run with: `pytest tests/test_mtp.py -v`

---

*Document version: 2026-02-27. Based on NanoSeek commit on branch
`cursor/development-environment-setup-c0da`. All code references point to
`model/model.py` (lines 737–1050) and `model/config.py` (`MTPConfig` dataclass).*
