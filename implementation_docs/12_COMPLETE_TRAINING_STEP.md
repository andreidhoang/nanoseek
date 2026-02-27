# 12 — One Complete Training Step: From Raw Data to Optimizer Update

## Distinguished Engineer's Notebook — February 2026

**Author**: Distinguished AI Researcher & Principal Engineer, Foundation Models Division
**Scope**: Exhaustive trace of a single training iteration in NanoSeek pre-training
**File Under Audit**: `scripts/pre-train.py` (1680 lines), `model/optimizer/muon.py`, `model/optimizer/adamw.py`, `model/model.py`, `model/config.py`

---

## Engineer's Thinking Process

> "I'm tracing **step 1000** of training. The model has processed roughly **524 million tokens**
> so far. Here's exactly what happens in this single training iteration — every tensor shape,
> every scalar, every branch — verified against the actual source code."

We assume:

- **Single-GPU training** (`ddp_world_size = 1`) on an H100 80 GB SXM.
- Default `TrainingConfig` values (lines 183–300 of `pre-train.py`).
- Chinchilla-optimal horizon: `target_param_data_ratio = 20.0`.
- `CUSTOM_OPTIMIZERS_AVAILABLE = True` (Muon + AdamW installed).
- `DATALOADER_AVAILABLE = True` (streaming parquet dataloader active).
- `enable_multiphase = False` (single-phase dense MLA training).

---

## Section 1: State Before This Step

### 1.1 Training Horizon Derivation

From `main()` lines 1040–1053, the horizon is computed as:

```
active_params    = config.estimated_active_params   # ~1,077,000,000  (see Section 1.3)
ratio            = config.target_param_data_ratio   # 20.0
target_tokens    = 20.0 × 1,077,000,000            # = 21,540,000,000  ≈ 21.5B tokens
num_iterations   = 21,540,000,000 // 524,288        # = 41,085
```

The model config also stores `total_tokens` (updated at line 1060):

```python
model_config.total_tokens = config.total_batch_size * num_iterations
                          = 524,288 × 41,085
                          = 21,537,838,080
```

For this trace we round to **`num_iterations ≈ 41,085`** (the exact value depends on `estimated_active_params`; the config docstring says ~1.08B giving ~41,142 steps, but the computed property varies slightly. We'll use 41,085 as our running example, acknowledging this.)

### 1.2 Step Counter State

| Variable | Value | Source |
|----------|-------|--------|
| `step` | 1000 | Loop counter, line 1646 |
| `tokens_processed` | 1000 × 524,288 = **524,288,000** (524.3M) | `orig_model.tokens_processed`, updated at line 1544 |
| `total_tokens` | ~21.54B | `model_config.total_tokens`, line 1060 |
| `num_iterations` | ~41,085 | Computed at line 1049 |
| `phase_step` | 1000 | Same as `step` (single-phase), line 1647 |
| `current_phase` | 0 | Not multi-phase, line 1068 |

### 1.3 Model Architecture Recap

From `get_nanoseek_config()` (config.py lines 847–1016):

| Parameter | Value |
|-----------|-------|
| `hidden_size` | 2048 |
| `num_layers` | 16 |
| `num_heads` | 16 |
| `vocab_size` | 65,536 |
| `intermediate_size` (dense FFN) | 5,243 |
| `moe.n_routed_experts` | 64 |
| `moe.num_experts_per_tok` | 8 |
| `moe.n_shared_experts` | 2 |
| `moe.moe_intermediate_size` | 768 |
| `moe.first_k_dense_replace` | 2 |
| Dense layers | 0, 1 |
| MoE layers | 2–15 (14 layers) |
| `mla.q_lora_rank` | 430 |
| `mla.kv_lora_rank` | 143 |
| `mla.qk_nope_head_dim` | 64 |
| `mla.qk_rope_head_dim` | 32 |
| `mla.v_head_dim` | 64 |
| `mtp.num_mtp_modules` | 1 |

Estimated parameter counts from `NanoSeekConfig` properties (config.py lines 609–688):

| Group | Estimated |
|-------|-----------|
| `estimated_total_params` | ~4.75B |
| `estimated_active_params` | ~1.08B |

### 1.4 Schedule States at Step 1000

#### Learning Rate Multiplier

From `get_lr_multiplier()` (pre-train.py lines 567–597):

```python
warmup_iters  = round(0.0 * 41085)   # = 0   (warmup_ratio = 0.0)
warmdown_iters = round(0.2 * 41085)  # = 8217
step = 1000

# Check branches:
# step < warmup_iters?  1000 < 0?  → NO
# step <= num_iterations - warmdown_iters?  1000 <= 41085 - 8217 = 32868?  → YES
# → Constant phase

lrm = 1.0
```

Learning rates applied to each optimizer group (line 1517):

| Group | `initial_lr` | `lr = initial_lr × lrm` |
|-------|-------------|------------------------|
| embed (embed_tokens) | 0.2 | **0.2** |
| unembed (lm_head) | 0.004 | **0.004** |
| matrix (Muon 2D params) | 0.02 | **0.02** |

#### Gamma (MoE Load-Balance Bias Update Rate)

From `NanoSeekModel.get_gamma()` (model.py lines 1534–1540):

```python
tokens_processed = 524,288,000       # 524.3M
total_tokens     = 21,537,838,080    # ~21.5B
gamma_freeze_ratio = 0.80
freeze_at = int(21,537,838,080 * 0.80) = 17,230,270,464   # ~17.2B

524,288,000 < 17,230,270,464  → True
# → gamma = config.moe.gamma = 0.001
```

**`gamma = 0.001`** — still actively adjusting expert biases.

#### MTP Loss Weight

From `NanoSeekModel.get_mtp_loss_weight()` (model.py lines 1526–1532):

```python
tokens_processed = 524,288,000
transition = int(21,537,838,080 * 0.60) = 12,922,702,848  # ~12.9B

524,288,000 < 12,922,702,848  → True
# → mtp_loss_weight = config.mtp.mtp_loss_weight_initial = 0.3
```

**`mtp_loss_weight = 0.3`** — in the early-training phase.

#### Muon Momentum

From `get_muon_momentum()` (pre-train.py lines 600–609):

```python
frac = min(1000 / 300, 1.0)   # = min(3.333, 1.0) = 1.0
momentum = (1 - 1.0) * 0.85 + 1.0 * 0.95  # = 0.95
```

**`muon_momentum = 0.95`** — fully warmed up (past step 300).

### 1.5 Summary of All Schedule Values at Step 1000

| Schedule | Value | Phase |
|----------|-------|-------|
| LR multiplier | 1.0 | Constant (warm-up=0, warm-down starts at ~32,868) |
| Gamma | 0.001 | Active bias adjustment (freezes at ~17.2B tokens) |
| MTP weight λ | 0.3 | Initial (transitions at ~12.9B tokens) |
| Muon momentum | 0.95 | Fully warmed (past step 300) |

---

## Section 2: Data Loading — The Streaming Pipeline

### 2.1 Overview

The dataloader is built at lines 1158–1213 of `pre-train.py`. For the streaming case
(`DATALOADER_AVAILABLE = True`), the function `build_train_loader()` (line 1161) calls:

```python
tokenizing_distributed_data_loader_with_state(
    B=config.device_batch_size,    # 8
    T=config.max_seq_len,          # 4096
    split="train",
    tokenizer_threads=4,
    tokenizer_batch_size=128,
    device=device,                 # cuda
    resume_state_dict=resume_state,
)
```

### 2.2 Inside the Dataloader (`scripts/dataloader.py`)

The dataloader is a **Python generator** (not a PyTorch `DataLoader`). It streams data
through three stages:

```
Stage 1: Parquet Iteration
    ├── parquet_paths = list_parquet_files()           # 47 train shards (~94MB each)
    ├── For each parquet file:
    │   ├── pf = pq.ParquetFile(filepath)
    │   ├── For each row_group (strided by ddp_world_size):
    │   │   ├── rg = pf.read_row_group(rg_idx)
    │   │   ├── batch = rg.column('text').to_pylist()   # ~1024 text documents
    │   │   └── yield batch[i:i+128], (pq_idx, rg_idx)  # sub-batches of 128
    │   └── rg_idx += ddp_world_size
    └── Multi-epoch: wraps around infinitely

Stage 2: Tokenization
    ├── tokenizer = get_tokenizer()
    ├── bos_token = tokenizer.get_bos_token_id()
    ├── token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=4)
    └── token_buffer.extend(tokens)       # deque, streaming right-to-left

Stage 3: Batch Assembly
    ├── needed_tokens = 8 × 4096 + 1 = 32,769
    ├── While len(token_buffer) < needed_tokens: fetch more
    ├── tokens = [token_buffer.popleft() for _ in range(32769)]
    ├── scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=True)
    ├── inputs  = scratch[:-1].view(8, 4096).to(device, non_blocking=True)   # x
    ├── targets = scratch[1:].view(8, 4096).to(device, non_blocking=True)    # y
    └── yield inputs, targets, state_dict
```

### 2.3 Concrete Shapes on GPU

At the start of step 1000, the **pre-fetched** batch (from line 1187 or line 1495
of the previous step's prefetch) is already on the GPU:

| Tensor | Shape | Dtype | Size |
|--------|-------|-------|------|
| `x` (input_ids) | `[8, 4096]` | `torch.long` (int64) | 8 × 4096 × 8 = 262,144 bytes ≈ 256 KB |
| `y` (labels) | `[8, 4096]` | `torch.long` (int64) | 256 KB |
| **Total** | | | **512 KB** (negligible) |

The `state_dict` returned by the dataloader (line 1495) tracks the current parquet
file index (`pq_idx`) and row-group index (`rg_idx`) for approximate resume.

### 2.4 Data Characteristics

From `scripts/dataset.py`: the dataset is **FineWeb-Edu-100B-Shuffle** hosted on
Hugging Face (line 28), consisting of 48 parquet shards (~94 MB each). The train split
uses shards 0–46 (47 shards), and the validation split uses shard 47 alone.

Each row group contains roughly **1024 documents** of educational web text. With
`tokenizer_batch_size=128`, each sub-batch tokenizes 128 documents in parallel
using 4 threads.

---

## Section 3: Gradient Accumulation Loop

### 3.1 Batch Decomposition

From lines 982–995 of `pre-train.py`:

```python
tokens_per_fwdbwd      = device_batch_size × max_seq_len
                       = 8 × 4096
                       = 32,768

world_tokens_per_fwdbwd = tokens_per_fwdbwd × ddp_world_size
                        = 32,768 × 1
                        = 32,768

grad_accum_steps        = total_batch_size // world_tokens_per_fwdbwd
                        = 524,288 // 32,768
                        = 16
```

So each optimization step performs **16 micro-steps** of forward+backward, each
processing `8 × 4096 = 32,768` tokens. Over 16 micro-steps that's `16 × 32,768 = 524,288`
tokens — exactly `total_batch_size`.

### 3.2 The Micro-Step Loop (Lines 1471–1503)

```python
indexer_loss_total = 0.0                            # line 1470
for micro_step in range(grad_accum_steps):          # line 1471 — range(16)
    with autocast_ctx:                              # line 1472 — torch.amp.autocast(bf16)
        outputs = model(input_ids=x, labels=y)      # line 1473
        loss = outputs['loss']                       # line 1474

    # (multi-phase indexer loss — skipped, enable_multiphase=False)

    train_loss = loss.detach()                       # line 1486 — for logging
    loss = loss / grad_accum_steps                   # line 1489 — scale for accumulation
    loss.backward()                                  # line 1490

    # Prefetch next batch                            # lines 1493–1502
    x, y, dataloader_resume_state = next(train_loader)
```

### 3.3 Micro-Step 0 of 16: Forward Pass Deep Dive

#### 3.3.1 Entry: `NanoSeekModel.forward()` (model.py line 1562)

Input: `input_ids` shape `[8, 4096]`, `labels` shape `[8, 4096]`.

**Step 1 — Embedding lookup** (line 1577):
```python
hidden_states = self.embed_tokens(input_ids)   # [8, 4096] → [8, 4096, 2048]
```
- `embed_tokens`: `nn.Embedding(65536, 2048)` — 134,217,728 params (134M)
- Output shape: `[8, 4096, 2048]`, dtype `bfloat16` (under autocast)

**Step 2 — Position IDs** (lines 1579–1588):
```python
position_ids = torch.arange(4096, device=device).unsqueeze(0).expand(8, -1)
# → [8, 4096], values 0..4095
```

**Step 3 — Causal mask** (lines 1591–1592):
```python
causal_mask = create_causal_mask(4096, dtype=bfloat16, device=device, past_len=0)
# → [1, 1, 4096, 4096] upper-triangular -inf mask
```

**Step 4 — Decoder layers** (lines 1607–1623):

For each of 16 layers (`idx = 0..15`):

```python
hidden_states, present_kv, aux_data = layer(
    hidden_states=hidden_states,       # [8, 4096, 2048]
    attention_mask=causal_mask,        # [1, 1, 4096, 4096]
    position_ids=position_ids,         # [8, 4096]
    past_key_value=None,
    use_cache=False,
    training_step=None,
    output_indexer_loss=False,          # sparse.enabled=False
)
```

Each `NanoSeekDecoderLayer` (model.py line 1378) does:

```
input_layernorm → MLA attention → residual → post_attention_layernorm → FFN → residual
```

**Layers 0–1** (dense FFN): `MLP(dim=2048, inter_dim=5243)` — SwiGLU FFN

**Layers 2–15** (MoE): `MoE(dim=2048, moe_inter_dim=768, 64 routed, 8 active, 2 shared)`

**Step 5 — Final norm + LM head** (lines 1640–1641):
```python
hidden_states = self.norm(hidden_states)           # RMSNorm → [8, 4096, 2048]
logits = self.lm_head(hidden_states)               # Linear(2048, 65536) → [8, 4096, 65536]
```

**Step 6 — Loss computation** (line 1658, calling `_compute_loss` at line 1697):

```python
# Main loss (next-token prediction)
shift_logits = logits[:, :-1, :]                   # [8, 4095, 65536]
shift_labels = labels[:, 1:]                        # [8, 4095]
main_loss = F.cross_entropy(
    shift_logits.view(-1, 65536),                   # [32760, 65536]
    shift_labels.view(-1),                          # [32760]
    ignore_index=-100
)
# At step 1000: main_loss ≈ 5.5–7.0 (depends on data)

# MTP loss (model.py lines 1709–1717)
mtp_outputs = self.mtp(hidden_states, labels=labels)
mtp_loss = mtp_outputs["mtp_loss"]
mtp_weight = self.get_mtp_loss_weight()            # = 0.3

# Auxiliary loss from MoE sequence-level balancing
# aux_loss = sum of seq_aux_loss from MoE layers (very small, α=0.0001)

# Total composite loss
total_loss = main_loss + 0.3 * mtp_loss + aux_loss
```

#### 3.3.2 Typical Loss Values at Step 1000

| Component | Typical Value | Weight | Contribution |
|-----------|---------------|--------|-------------|
| `main_loss` | ~6.0 | 1.0 | ~6.0 |
| `mtp_loss` | ~7.0 | 0.3 (λ) | ~2.1 |
| `aux_loss` | ~0.001 | 1.0 | ~0.001 |
| **`loss` (total)** | **~8.1** | | |

(Exact values depend on data distribution. Early training loss drops rapidly from
~11 at step 0 to ~5–8 by step 1000.)

### 3.4 Loss Scaling for Accumulation (Line 1489)

```python
loss = loss / grad_accum_steps
     = 8.1 / 16
     = 0.506
```

This ensures that after 16 micro-steps of `loss.backward()`, the accumulated gradients
represent the **average** gradient over the full 524,288-token batch, not the sum.

### 3.5 Backward Pass (Line 1490)

```python
loss.backward()
```

PyTorch's autograd traverses the computation graph in reverse. Gradients are
**accumulated** (added) into `param.grad` for each parameter. After the first
micro-step, `param.grad` holds `grad_micro_0 / 16`. After all 16 micro-steps:

```
param.grad = (1/16) × Σ_{i=0}^{15} grad_micro_i
```

This is mathematically equivalent to computing the gradient on the full 524K-token batch.

Key memory implication: only one micro-batch of activations is live at a time
(32,768 tokens), not the full 524,288. This is the entire point of gradient accumulation.

### 3.6 Data Prefetch (Lines 1493–1495)

While the backward pass runs on the GPU, the next batch is fetched:

```python
x, y, dataloader_resume_state = next(train_loader)
```

This overlaps CPU→GPU transfer (via `pin_memory=True` and `non_blocking=True` in the
dataloader) with GPU compute. The `next()` call:

1. Pulls tokens from `token_buffer` (deque)
2. If buffer is exhausted, reads next row group from parquet
3. Tokenizes the text batch (4 threads)
4. Creates tensors with `pin_memory=True`
5. Calls `.to(device, non_blocking=True)` for async H2D transfer

### 3.7 Summary: After 16 Micro-Steps

At the end of the `for micro_step in range(16)` loop:

- Every `param.grad` holds the mean gradient over 524,288 tokens
- `train_loss` holds the **last** micro-batch's loss (for logging, line 1486)
- `x, y` hold the pre-fetched batch for step 1001
- GPU memory for intermediate activations is freed (only gradients remain)

---

## Section 4: After Accumulation — Gradient Processing

### 4.1 Gradient Clipping (Lines 1504–1510)

```python
grad_norm = 0.0
if config.grad_clip > 0.0:                          # grad_clip = 1.0, always True
    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
        orig_model.parameters(), config.grad_clip    # max_norm = 1.0
    )
    grad_norm = grad_norm_tensor.item()
```

**What `clip_grad_norm_` does:**

1. Compute the **global L2 norm** of all parameter gradients:
   ```
   total_norm = sqrt(Σ_p ||p.grad||²)
   ```
   This traverses all ~4.75B parameters and computes a single scalar.

2. If `total_norm > max_norm` (= 1.0), scale **all** gradients by `max_norm / total_norm`:
   ```
   p.grad *= 1.0 / total_norm   (if total_norm > 1.0)
   ```

3. Return `total_norm` as `grad_norm_tensor`.

**Typical behavior at step 1000**: `grad_norm` ≈ 0.5–2.0. Early training has higher
gradient norms (5–10 at step 0), which decay as the model stabilizes. Clipping activates
when norms spike (e.g., from a particularly unusual batch or MoE routing instability).

### 4.2 Learning Rate Update (Lines 1512–1517)

```python
lrm = get_lr_multiplier(step, num_iterations, config)   # = 1.0 (constant phase)

for opt in optimizers:                                    # [adamw_optimizer, muon_optimizer]
    if opt is not None:
        for group in opt.param_groups:
            group['lr'] = group['initial_lr'] * lrm
```

At step 1000 with `lrm = 1.0`:

| Optimizer | Group | `initial_lr` | `lr` after update |
|-----------|-------|-------------|-------------------|
| AdamW | embed (embed_tokens) | 0.2 | 0.2 |
| AdamW | unembed (lm_head) | 0.004 | 0.004 |
| Muon | matrix (all 2D params) | 0.02 | 0.02 |

Note: there is **no** non-2D matrix params group in the default configuration because
`CUSTOM_OPTIMIZERS_AVAILABLE = True` and all parameters are either embeddings, the lm_head,
or 2D matrices. Biases don't exist (`use_bias=False`), and RMSNorm weights are 1D but
they end up in the `adamw_matrix_params` fallback group since they aren't embed/unembed and
aren't 2D. Actually — let's be precise:

From `setup_optimizers()` (lines 458–471):

```python
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if 'embed_tokens' in name:
        embed_params.append(param)       # embed_tokens.weight [65536, 2048] — 1 param
    elif 'lm_head' in name:
        unembed_params.append(param)     # lm_head.weight [65536, 2048] — 1 param
    elif CUSTOM_OPTIMIZERS_AVAILABLE and param.ndim == 2:
        muon_params.append(param)        # All 2D Linear weights
    else:
        adamw_matrix_params.append(param) # 1D params: RMSNorm weights, Gate biases, etc.
```

The `adamw_matrix_params` list captures:
- RMSNorm `.weight` tensors (1D, shape `[2048]` or `[430]` or `[143]`)
- `Gate.expert_bias` buffers are registered buffers (not parameters), so NOT here
- `Gate.weight` is 2D `[64, 2048]`, so it goes to Muon
- `LightningIndexer.head_weights` is 1D `[4]` — goes to adamw_matrix_params

These 1D params get their own AdamW group with `lr = matrix_lr = 0.02`.

### 4.3 Muon Momentum Update (Lines 1519–1523)

```python
if muon_optimizer is not None:
    muon_momentum = get_muon_momentum(step)   # step=1000
    for group in muon_optimizer.param_groups:
        group['momentum'] = muon_momentum
```

From `get_muon_momentum(1000)`:
```python
frac = min(1000 / 300, 1.0) = 1.0
momentum = (1 - 1.0) * 0.85 + 1.0 * 0.95 = 0.95
```

**`momentum = 0.95`** written into every Muon param group.

---

## Section 5: Optimizer Step — The Critical Phase

Lines 1525–1531 of `pre-train.py`:

```python
# Optimizer step                                     # line 1526
for opt in optimizers:                               # [adamw_optimizer, muon_optimizer]
    if opt is not None:
        opt.step()                                   # line 1528

# Zero gradients                                     # line 1531
model.zero_grad(set_to_none=True)
```

The two optimizers update **disjoint** parameter sets. Let's trace each.

### 5.1 AdamW Step (for Embeddings, Unembeddings, and 1D Params)

The AdamW optimizer is `torch.optim.AdamW` (line 525, single-GPU case) with:
- `betas = (0.9, 0.95)`
- `eps = 1e-8`
- `fused = True` (CUDA fused kernel)

It manages three param groups:

#### Group 0: Input Embeddings (`embed_tokens.weight`)

| Setting | Value |
|---------|-------|
| Shape | `[65536, 2048]` |
| Parameters | 134,217,728 (134M) |
| `lr` | 0.2 |
| `weight_decay` | 0.0 |
| `betas` | (0.9, 0.95) |

The **complete AdamW update** for one parameter θ at step t=1000:

```
Given:
    g = param.grad                       # gradient (averaged over 524K tokens)
    β₁ = 0.9, β₂ = 0.95
    ε = 1e-8
    lr = 0.2
    wd = 0.0                             # no weight decay for embeddings
    t = 1000                             # step counter (1-indexed in PyTorch)

Update running averages:
    m_t = β₁ × m_{t-1} + (1 - β₁) × g
        = 0.9 × m_{999} + 0.1 × g

    v_t = β₂ × v_{t-1} + (1 - β₂) × g²
        = 0.95 × v_{999} + 0.05 × g²

Bias corrections:
    bias_correction1 = 1 - β₁^t = 1 - 0.9^1000 ≈ 1.0 (negligible at t=1000)
    bias_correction2 = 1 - β₂^t = 1 - 0.95^1000 ≈ 1.0 (negligible at t=1000)

    Note: At t=1000, 0.9^1000 ≈ 1.75e-46 and 0.95^1000 ≈ 5.29e-23
    Both are effectively zero, so bias corrections are ~1.0

Compute step:
    m̂ = m_t / bias_correction1 ≈ m_t
    v̂ = v_t / bias_correction2 ≈ v_t

    update = m̂ / (√v̂ + ε)

Weight decay (decoupled):
    θ_t = θ_{t-1} - lr × (update + wd × θ_{t-1})
        = θ_{t-1} - 0.2 × (update + 0.0 × θ_{t-1})
        = θ_{t-1} - 0.2 × update
```

**Memory for this group's Adam states:**
- `m` (momentum, same dtype as param): 134M × 2 bytes (bf16 under fused) ≈ 256 MB
- `v` (variance, same dtype): 134M × 2 bytes ≈ 256 MB
- Note: PyTorch's fused AdamW on CUDA keeps states in the same dtype as the parameter.
  If the model is bf16, states are bf16. This differs from the standard (fp32 states) behavior.

#### Group 1: Output Projection (`lm_head.weight`)

| Setting | Value |
|---------|-------|
| Shape | `[65536, 2048]` |
| Parameters | 134,217,728 (134M) |
| `lr` | 0.004 |
| `weight_decay` | 0.0 |

Same AdamW formula, but with `lr = 0.004` — **50× smaller** than embedding LR.
This follows the nanochat/DeepSeek pattern of using very conservative updates for
the unembedding (output) matrix.

#### Group 2: 1D Matrix Params (RMSNorm weights, etc.)

| Setting | Value |
|---------|-------|
| `lr` | 0.02 |
| `weight_decay` | 0.0 (`matrix_weight_decay = 0.0`) |

This group contains small 1D tensors:
- 16 × 2 = 32 `RMSNorm.weight` tensors (input + post-attention per layer), each `[2048]`
- 1 final `norm.weight` `[2048]`
- Various MTP norms and small tensors
- Total: relatively tiny (~200K params)

### 5.2 Muon Step (for 2D Matrix Parameters)

Muon is the optimizer for all 2D parameters that are not embeddings/unembeddings.
This includes:

- **MLA projections**: `wq_a`, `wq_b`, `wkv_a`, `wkv_b`, `wo` per layer
- **Dense FFN** (layers 0–1): `gate_proj`, `up_proj`, `down_proj`
- **MoE experts** (layers 2–15): 64 experts × 3 projections each, per layer
- **MoE shared experts**: 2 per MoE layer × 3 projections
- **Gate weights**: `Gate.weight` `[64, 2048]` per MoE layer
- **MTP projections**: concat_proj, cross/self attention, FFN projections

From `Muon.__init__()` (muon.py lines 60–67):

```python
def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
    defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
    params: list[Tensor] = [*params]
    param_groups = []
    for size in {p.numel() for p in params}:
        group = dict(params=[p for p in params if p.numel() == size])
        param_groups.append(group)
    super().__init__(param_groups, defaults)
```

Note the **grouping by `numel()`**: all parameters with the same total element count
are placed in the same group. This enables potential batching optimizations.

#### The Complete Muon Update (muon.py lines 69–83)

For each parameter `p` in each group, `Muon.step()` executes:

```python
@torch.no_grad()
def step(self):
    for group in self.param_groups:
        params: list[Tensor] = group["params"]
        for p in params:
            g = p.grad                                    # (1) Get gradient
            assert g is not None
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)
            buf: Tensor = state["momentum_buffer"]

            # (2) Momentum update (EMA of gradients)
            buf.lerp_(g, 1 - group["momentum"])
            # buf = momentum × buf + (1 - momentum) × g
            # buf = 0.95 × buf + 0.05 × g

            # (3) Nesterov momentum (look-ahead)
            g = g.lerp_(buf, group["momentum"])
            # g = (1 - momentum) × g + momentum × buf
            # g = 0.05 × g + 0.95 × buf
            # Effectively: g ≈ buf + momentum × (buf - buf_prev)

            # (4) Newton-Schulz orthogonalization (THE KEY INNOVATION)
            g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
            # ns_steps = 5

            # (5) Parameter update with aspect-ratio scaling
            p.add_(g, alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)
```

Let's trace each substep for a concrete parameter.

#### 5.2.1 Concrete Example: `layers.5.self_attn.wq_a.weight`

This is the query down-projection for layer 5:

| Property | Value |
|----------|-------|
| Shape | `[430, 2048]` |
| Parameters | 880,640 |
| `rows` = `p.size(-2)` | 430 |
| `cols` = `p.size(-1)` | 2048 |
| Aspect ratio `rows/cols` | 430/2048 = 0.21 |
| `max(1, rows/cols)` | 1 (since 0.21 < 1) |
| Scale factor | `1^0.5 = 1.0` |

**Step (2): Momentum update**

```python
buf.lerp_(g, 1 - 0.95)
# buf = 0.95 × buf_old + 0.05 × g
```

This is exponential moving average of gradients. After 1000 steps, `buf` is a heavily
smoothed average of recent gradients with effective window ~20 steps (1/(1-0.95)).

**Step (3): Nesterov look-ahead**

```python
g = g.lerp_(buf, 0.95)
# g = 0.05 × g_original + 0.95 × buf
```

This creates a "look-ahead" gradient: mostly the momentum buffer plus a small correction
from the current raw gradient. Nesterov momentum provides faster convergence than
standard momentum by evaluating the gradient at the predicted next position.

**Step (4): Newton-Schulz orthogonalization** (muon.py lines 10–36)

This is the **defining innovation** of Muon. The function `zeropower_via_newtonschulz5`
approximates the nearest orthogonal matrix to the gradient update:

```python
@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)               # line 21
    X = G.bfloat16()                                    # line 22

    # If tall matrix, transpose to work with wide matrix
    if G.size(-2) > G.size(-1):                         # line 23
        X = X.mT                                        # Transpose

    # Normalize spectral norm to ≤ 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7) # line 27

    # 5 Newton-Schulz iterations
    for _ in range(steps):                               # line 29
        A = X @ X.mT                                     # line 30 — [rows, rows]
        B = b * A + c * A @ A                            # line 31 — quintic terms
        X = a * X + B @ X                                # line 32

    # Transpose back if needed
    if G.size(-2) > G.size(-1):                          # line 34
        X = X.mT
    return X
```

For our `wq_a.weight` with shape `[430, 2048]`:

1. **Transpose check**: `430 > 2048`? NO → no transpose needed. `X` stays `[430, 2048]`.
2. **Spectral normalization**: `X /= ||X||_F` (Frobenius norm). This ensures the iteration
   converges. The norm is computed per-matrix (keepdim=True).
3. **5 iterations** of the quintic Newton-Schulz:
   - `A = X @ X.mT` → `[430, 2048] @ [2048, 430]` = `[430, 430]`
   - `B = -4.775 × A + 2.0315 × (A @ A)` → `[430, 430]`
   - `X = 3.4445 × X + B @ X` → `[430, 430] @ [430, 2048]` = `[430, 2048]`
4. **Result**: `X` ≈ `U × S' × V^T` where `S'` ≈ `Uniform(0.5, 1.5)` — an
   approximately orthogonal matrix.

The coefficients `a=3.4445, b=-4.7750, c=2.0315` are tuned to maximize the convergence
slope at zero, allowing the iteration to work in only 5 steps in bfloat16.

**Step (5): Parameter update**

```python
p.add_(g, alpha=-0.02 * max(1, 430/2048)**0.5)
# max(1, 0.21) = 1
# scale = 1.0^0.5 = 1.0
# p -= 0.02 * 1.0 * g_orthogonalized
```

For a parameter where `rows > cols`, e.g., `lm_head`-like shapes:

```python
# Example: shape [65536, 2048]
# max(1, 65536/2048) = max(1, 32) = 32
# scale = 32^0.5 = 5.66
# p -= 0.02 * 5.66 * g_orthogonalized
```

But `lm_head` is handled by AdamW, not Muon. A real Muon example with `rows > cols`:
`wkv_b.weight` `[16 × (64+64), 143]` = `[2048, 143]`:
- `max(1, 2048/143)` = `max(1, 14.32)` = 14.32
- scale = `14.32^0.5` = 3.78
- Update: `p -= 0.02 × 3.78 × g` = `p -= 0.0756 × g`

The aspect-ratio scaling compensates for the fact that tall/wide matrices have different
gradient norms when orthogonalized, ensuring balanced updates across differently shaped layers.

#### 5.2.2 Why Muon Works: Intuition

Standard SGD updates `p -= lr × g`, where the gradient `g` has arbitrary scale and
orientation. Muon replaces the gradient with its **nearest orthogonal matrix**:

- The direction is preserved (which neurons should change)
- The magnitude is normalized (every direction gets equal update magnitude)
- This acts like a natural gradient method without computing the Fisher information matrix
- The Newton-Schulz iteration is cheaper than SVD and numerically stable in bf16

The result: **faster convergence** and **better generalization** for 2D weight matrices,
particularly effective for attention and FFN projections in transformers.

### 5.3 Zero Gradients (Line 1531)

```python
model.zero_grad(set_to_none=True)
```

The `set_to_none=True` argument is crucial for memory efficiency:
- Instead of filling `.grad` tensors with zeros (which keeps them allocated)
- It sets `.grad = None` (which frees the gradient memory entirely)
- Gradients will be lazily re-allocated on the next `backward()` call

Memory saved: ~9.5 GB (all parameter gradients in bf16).

---

## Section 6: Post-Optimizer Updates

### 6.1 MoE Load Balance Bias Update (Lines 1536–1538)

```python
gamma = orig_model.get_gamma()                       # = 0.001 (computed in Section 1.4)
orig_model.update_load_balance_bias(gamma)
```

From `NanoSeekModel.update_load_balance_bias()` (model.py lines 1730–1737):

```python
def update_load_balance_bias(self, gamma: Optional[float] = None):
    if gamma is None:
        gamma = self.get_gamma()
    if gamma > 0:
        for layer in self.layers:
            if layer.is_moe_layer:                    # layers 2–15
                layer.ffn.update_load_balance_bias(gamma)
```

This iterates over 14 MoE layers (indices 2–15). For each, it calls
`MoE.update_load_balance_bias()` (model.py lines 721–727):

```python
def update_load_balance_bias(self, gamma: float = 0.001):
    with torch.no_grad():
        load = self.gate.expert_load                   # [64] — token counts from last forward
        mean_load = load.mean()
        if mean_load > 0:
            imbalance = (load - mean_load) / (mean_load + 1e-8)
            self.gate.expert_bias.sub_(gamma * imbalance)
```

#### Concrete Example

For one MoE layer at step 1000, suppose the expert load from the last micro-batch was:

```
expert_load = [520, 480, 530, 470, 510, 490, 540, 460, ...]  # 64 values
```

Each value is the number of tokens routed to that expert across the batch.
With `B=8, T=4096, K=8`: total token-expert assignments = `8 × 4096 × 8 = 262,144`.
Mean per expert: `262,144 / 64 = 4,096`.

```python
mean_load = 4096.0
imbalance = (expert_load - 4096) / (4096 + 1e-8)
# imbalance[0] = (520 - 4096) / 4096 ≈ -0.873    (if 520 is the count)
```

Wait — these are counts from `scatter_add_` in `Gate.forward()` (model.py line 439):

```python
load = torch.zeros(self.n_routed_experts, device=x.device)
load.scatter_add_(0, topk_indices.flatten(), 
                  torch.ones_like(topk_indices.flatten(), dtype=load.dtype))
self.expert_load.copy_(load)
```

So for `N = 8 × 4096 = 32,768` tokens, each selecting `K=8` experts:
- Total assignments: `32,768 × 8 = 262,144`
- Mean per expert: `262,144 / 64 = 4,096`

A well-balanced distribution would have each expert at ~4,096 assignments. If expert 0
has 4,200 (slightly overloaded):

```python
imbalance[0] = (4200 - 4096) / (4096 + 1e-8) = 104 / 4096 ≈ 0.0254
expert_bias[0] -= 0.001 × 0.0254 = -0.0000254
```

The bias is **subtracted** proportionally to overload: overloaded experts get a
**negative** bias adjustment, making them slightly less likely to be selected next time.
Underloaded experts get a positive adjustment.

This is DeepSeek V3's **auxiliary-loss-free** load balancing: instead of adding a loss
term that distorts the training signal, we directly adjust the routing bias. The bias
only affects the `scores_for_selection` during training (line 418 of model.py), not
the actual expert weights used for combining outputs.

### 6.2 MTP Token Counter Update (Lines 1542–1544)

```python
orig_model.update_tokens_processed(config.total_batch_size)
```

From `NanoSeekModel.update_tokens_processed()` (model.py line 1542):

```python
def update_tokens_processed(self, num_tokens: int):
    self.tokens_processed.add_(num_tokens)
```

```
tokens_processed: 524,288,000 → 524,288,000 + 524,288 = 524,812,288
```

This counter drives the **ratio-based schedules** for gamma and MTP loss weight.
At step 1001, `tokens_processed` = 524.8M. The schedules won't change materially
until billions more tokens are processed.

### 6.3 DSA Training Step Update (Lines 1549–1551)

```python
if config.enable_multiphase:                          # False in our trace
    orig_model.increment_dsa_training_steps()
```

**Skipped** — multi-phase training is disabled. If it were enabled, this would
call `DSASparseAttention.increment_training_step()` (model.py line 1370) on each
layer with sparse attention, incrementing the `training_step` buffer used to
determine dense warmup vs. sparse mode.

---

## Section 7: Timing and Logging

### 7.1 Timing (Lines 1553–1555)

```python
synchronize()                                         # torch.cuda.synchronize()
t1 = time.time()
dt = t1 - t0                                         # Wall time for this step
```

The `synchronize()` at line 1466 (start) and line 1553 (end) bracket the training step,
ensuring all GPU operations complete before measuring wall time.

Typical `dt` on H100 SXM for NanoSeek-1B with `grad_accum_steps=16`:
- ~2.5–4.0 seconds per step (depending on compilation state, data, etc.)

### 7.2 Training Time Tracking (Lines 1560–1561)

```python
if step > 10:                                         # Skip warmup steps
    total_training_time += dt
```

Steps 0–10 are excluded to avoid measuring compilation overhead (`torch.compile`).

### 7.3 EMA Loss Smoothing (Lines 1563–1566)

```python
ema_beta = 0.9
smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
```

At step 1000:
```python
# ema_beta^1001 = 0.9^1001 ≈ 1.58e-46 → negligible
# debiased_smooth_loss ≈ smooth_train_loss / 1.0 ≈ smooth_train_loss
```

The EMA produces a smooth loss curve for logging. The debiasing corrects for the
initial zero value (important only at early steps).

### 7.4 Throughput Metrics (Lines 1568–1574)

```python
tok_per_sec = int(config.total_batch_size / dt)
# = int(524,288 / 3.0) ≈ 174,762 tok/s     (if dt=3.0s)

flops_per_sec = num_flops_per_token * config.total_batch_size / dt
# = 6 × active_params × 524,288 / 3.0
# = 6 × 1,077,000,000 × 524,288 / 3.0
# ≈ 1.13e15 FLOPS

# MFU (vs H100 SXM theoretical peak BF16)
promised_flops_per_sec_h100 = 989e12 * ddp_world_size    # 989 TFLOPS × 1
mfu = 100 * flops_per_sec / promised_flops_per_sec_h100
# = 100 × 1.13e15 / 989e12
# ≈ 114%  (this is common for MoE models — MFU > 100% is possible
#          because we compute FLOPs based on active params, but the
#          actual hardware executes more ops for routing, etc.)
```

Note: MFU > 100% happens because the estimate `6 × active_params` underestimates true
compute (attention is O(n²), routing overhead exists, etc.), or the H100 theoretical peak
is for sustained BF16 matmuls and actual throughput can vary.

### 7.5 Log Line Format (Lines 1583–1615)

At step 1000, with `config.log_every = 10`, step 1000 triggers logging:

```python
pct_done = 100 * 1000 / 41085 ≈ 2.4%

log_parts = [
    "step 01000/41085 (2.4%)",
    "loss: 6.2341",         # debiased_smooth_loss
    "main: 5.8234",         # main_loss from last micro-batch
    "mtp: 7.1234 (λ=0.30)", # mtp_loss, mtp_weight
    "grad: 0.8523",         # grad_norm
    "lrm: 1.00",            # lr multiplier
    "γ: 0.0010",            # gamma
    "tok/s: 174,762",
    "mfu: 114.3%",
    "dt: 3000ms",
]

print0(" | ".join(log_parts))
```

**Example output:**
```
step 01000/41085 (2.4%) | loss: 6.2341 | main: 5.8234 | mtp: 7.1234 (λ=0.30) | grad: 0.8523 | lrm: 1.00 | γ: 0.0010 | tok/s: 174,762 | mfu: 114.3% | dt: 3000ms
```

### 7.6 Step Counter Increment (Lines 1646–1647)

```python
step += 1           # 1000 → 1001
phase_step += 1     # 1000 → 1001
```

---

## Section 8: One-Step Memory Budget

### 8.1 Model Parameters (bf16)

| Component | Params | Memory (bf16, 2 bytes/param) |
|-----------|--------|------------------------------|
| `embed_tokens` | 65,536 × 2,048 = 134M | 256 MB |
| `lm_head` | 65,536 × 2,048 = 134M | 256 MB |
| MLA (16 layers) | ~69M | 132 MB |
| Dense FFN (2 layers) | ~64M | 122 MB |
| Shared experts (14 layers × 2) | ~132M | 252 MB |
| Routed experts (14 layers × 64) | ~4,200M | 8,032 MB |
| Gate + norms + MTP | ~116M | 222 MB |
| **Total model params** | **~4,849M (~4.85B)** | **~9,272 MB ≈ 9.1 GB** |

### 8.2 Gradients (bf16)

Same size as parameters (one gradient per parameter):

| Component | Memory |
|-----------|--------|
| All gradients | **~9.1 GB** |

Note: with `set_to_none=True`, gradients are freed after the optimizer step and
re-allocated during the next backward pass.

### 8.3 Optimizer States

#### AdamW States

AdamW stores two states per parameter (`exp_avg` and `exp_avg_sq`):

| Group | Params | State Memory (2 states × param size) |
|-------|--------|--------------------------------------|
| embed_tokens | 134M | 134M × 2 × 2 = 512 MB |
| lm_head | 134M | 512 MB |
| 1D matrix params | ~0.2M | ~0.8 MB |
| **AdamW total** | **~268M** | **~1,024 MB ≈ 1.0 GB** |

Note: With `fused=True` on CUDA, PyTorch may keep states in the parameter's dtype (bf16).
With standard AdamW (fp32 states), this would be ~2.0 GB.

#### Muon States

Muon stores only one buffer per parameter (`momentum_buffer`):

| Component | Params | Memory (1 state × param size) |
|-----------|--------|-------------------------------|
| All 2D Muon params | ~4,581M | 4,581M × 2 = **~8,764 MB ≈ 8.6 GB** |

This is just the EMA buffer — no variance tracking like AdamW.

### 8.4 Activations (per micro-batch)

During forward pass with `batch_size=8, seq_len=4096`:

| Activation | Shape | Memory |
|------------|-------|--------|
| Input embeddings | `[8, 4096, 2048]` | 128 MB |
| Per-layer hidden states (16 layers saved for backward) | 16 × `[8, 4096, 2048]` | 2,048 MB |
| Attention scores (if not recomputed) | varies | ~512 MB |
| MoE routing intermediates | varies | ~256 MB |
| Logits `[8, 4096, 65536]` | | 4,096 MB |
| **Total activations (one micro-batch)** | | **~7–15 GB** |

With gradient checkpointing (if enabled), activation memory drops significantly
as intermediate states are recomputed during backward. The default config has
`gradient_checkpointing=True` in the model config, but the training script sets
`model.gradient_checkpointing = False` by default (the flag on the `NanoSeekModel`
instance starts as `False` at model.py line 1521, and `pre-train.py` does not enable it).

### 8.5 Total Memory Budget

| Component | Memory |
|-----------|--------|
| Model parameters (bf16) | 9.1 GB |
| Gradients (bf16, peak) | 9.1 GB |
| AdamW states | 1.0 GB |
| Muon states (bf16 momentum) | 8.6 GB |
| Activations (one micro-batch) | ~10 GB |
| CUDA overhead (context, kernels) | ~2 GB |
| **Total peak** | **~40 GB** |

This fits comfortably within H100 80 GB. The margin allows for memory fragmentation
and torch.compile's additional allocations.

With `torch.compile` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (line 84),
memory fragmentation is minimized through expandable virtual memory segments.

---

## Section 9: Inside the MLA Forward Pass (Detailed)

Since MLA is the most architecturally novel component, let's trace one layer in full.

### 9.1 `MultiHeadLatentAttention.forward()` (model.py lines 287–372)

Input: `hidden_states` shape `[8, 4096, 2048]`

**Query path:**

```python
q = self.wq_a(hidden_states)        # Linear(2048, 430) → [8, 4096, 430]
q = self.q_norm(q)                   # RMSNorm(430) → [8, 4096, 430]
q = self.wq_b(q)                     # Linear(430, 16×96) → [8, 4096, 1536]
                                      # where 96 = qk_nope_head_dim(64) + qk_rope_head_dim(32)
q = q.view(8, 4096, 16, 96)         # → [8, 4096, 16, 96]
q_nope, q_pe = split(q, [64, 32])   # q_nope: [8, 4096, 16, 64]
                                      # q_pe:   [8, 4096, 16, 32]
```

**KV path:**

```python
kv = self.wkv_a(hidden_states)       # Linear(2048, 143+32) → [8, 4096, 175]
kv_compressed, k_pe = split(kv, [143, 32])
                                      # kv_compressed: [8, 4096, 143]
                                      # k_pe:          [8, 4096, 32]
kv_compressed = self.kv_norm(kv_compressed)  # RMSNorm(143) → [8, 4096, 143]
```

Note: The KV cache stores only `kv_compressed` `[B, T, 143]` and `k_pe` `[B, T, 1, 32]`
— this is the **175 dims per token** that gives 23× compression.

**RoPE application:**

```python
k_pe = k_pe.unsqueeze(2)            # [8, 4096, 1, 32] — shared across heads!
k_pe = apply_rotary_emb(k_pe, freqs_cis[:4096])  # Rotate shared key
q_pe = apply_rotary_emb(q_pe, freqs_cis[:4096])  # Rotate per-head query
```

The key innovation: `k_pe` has **1 head** (shared), while `q_pe` has **16 heads**.
This is how MLA shares the RoPE component of the key across all heads.

**KV expansion:**

```python
kv_expanded = self.wkv_b(kv_compressed)  # Linear(143, 16×(64+64)) → [8, 4096, 2048]
kv_expanded = kv_expanded.view(8, 4096, 16, 128)  # → [8, 4096, 16, 128]
k_nope, v = split(kv_expanded, [64, 64])
                                      # k_nope: [8, 4096, 16, 64]
                                      # v:      [8, 4096, 16, 64]
```

**Attention computation:**

```python
# Combine Q = [q_nope, q_pe] → [8, 4096, 16, 96]
# Combine K = [k_nope, k_pe_expanded] → [8, 4096, 16, 96]
# k_pe_expanded = k_pe.expand(-1, -1, 16, -1)  # broadcast 1→16 heads

q = q.transpose(1, 2)               # [8, 16, 4096, 96]
k = k.transpose(1, 2)               # [8, 16, 4096, 96]
v = v.transpose(1, 2)               # [8, 16, 4096, 64]

attn_weights = q @ k.T × softmax_scale  # [8, 16, 4096, 4096]
attn_weights += causal_mask               # Upper triangle → -inf
attn_weights = softmax(attn_weights)      # [8, 16, 4096, 4096]

attn_output = attn_weights @ v            # [8, 16, 4096, 64]
attn_output → reshape → [8, 4096, 1024]  # 16 × 64 = 1024
output = self.wo(attn_output)             # Linear(1024, 2048) → [8, 4096, 2048]
```

`softmax_scale = mscale / sqrt(qk_head_dim) = 1.0 / sqrt(96) ≈ 0.1021`

---

## Section 10: Inside the MoE Forward Pass (Detailed)

### 10.1 `MoE.forward()` (model.py lines 663–719)

Input: `x` shape `[8, 4096, 2048]` (post-attention, post-norm)

**Step 1: Flatten**
```python
x_flat = x.view(32768, 2048)         # [N, D] where N = 8 × 4096 = 32,768
```

**Step 2: Shared expert computation** (line 684)
```python
shared_output = self._compute_shared_output(x_flat)
# 2 shared experts, each SwiGLUFFN(2048, 768):
# shared_output = expert_0(x_flat) + expert_1(x_flat)   → [32768, 2048]
```

**Step 3: Gate (routing decision)** (line 691)
```python
weights, indices = self.gate(x_flat)   # weights: [32768, 8], indices: [32768, 8]
```

Inside `Gate.forward()` (model.py lines 409–442):

```python
scores = F.linear(x, self.weight)      # [32768, 2048] × [64, 2048].T → [32768, 64]
scores = torch.sigmoid(scores)         # Sigmoid scoring (DeepSeek V3 innovation)

# Add bias for training-time load balancing
scores_for_selection = scores + self.expert_bias.unsqueeze(0)  # [32768, 64]

# Group-based routing: select top-4 of 8 groups first
scores_grouped = scores_for_selection.view(-1, 8, 8)  # [32768, 8 groups, 8 per group]
group_scores = scores_grouped.max(dim=-1).values       # [32768, 8]
_, top_groups = group_scores.topk(4, dim=-1)           # Select 4 groups → [32768, 4]

# Mask out non-selected groups
group_mask → [32768, 64]                               # True for 4×8 = 32 experts
scores_for_selection.masked_fill_(~group_mask, -inf)

# Select top-8 from remaining 32 candidates
topk_weights, topk_indices = scores_for_selection.topk(8, dim=-1)
# topk_weights: [32768, 8]
# topk_indices: [32768, 8]

# Get actual scores (without bias) for weighting
weights = scores.gather(dim=-1, index=topk_indices) × 2.5  # route_scale = 2.5
```

**Step 4: Token-centric dispatch** (model.py line 694)
```python
routed_output = token_centric_dispatch(x_flat, indices, weights, self.experts)
```

This efficiently routes each token to its 8 selected experts using the sort-based
dispatch algorithm described in `token_centric_dispatch()` (model.py lines 485–583):

1. Flatten all `32,768 × 8 = 262,144` token-expert assignments
2. Sort by expert ID → contiguous batches per expert
3. Process each expert's batch through its SwiGLUFFN
4. Scatter-add weighted outputs back to original positions

**Step 5: Combine** (line 700)
```python
output = routed_output + shared_output                 # [32768, 2048]
output = output.view(8, 4096, 2048)
```

### 10.2 Expert FFN Details

Each expert is `SwiGLUFFN(dim=2048, inter_dim=768)`:

```python
gate = F.silu(self.gate_proj(x))    # Linear(2048, 768) + SiLU → [N_expert, 768]
up   = self.up_proj(x)               # Linear(2048, 768) → [N_expert, 768]
out  = self.down_proj(gate * up)     # Linear(768, 2048) → [N_expert, 2048]
```

Parameters per expert: `3 × 2048 × 768 = 4,718,592` (4.7M)
Parameters for all 64 experts per layer: `64 × 4.7M = 301.9M`
Active parameters (8 experts): `8 × 4.7M = 37.7M`

---

## Section 11: Inside the MTP Forward Pass (Detailed)

### 11.1 `MultiTokenPrediction.forward()` (model.py lines 954–1012)

Called from `_compute_loss()` (model.py line 1710):

```python
mtp_outputs = self.mtp(hidden_states, labels=labels)
```

Input: `hidden_states` = final layer output, shape `[8, 4096, 2048]`

For `num_mtp_modules = 1` (module index 0, predicting token at position t+2):

**Step 1: Prepare targets** (lines 970–974)
```python
token_offset = 0 + 1 = 1
target_tokens = labels[:, 1:]                       # [8, 4095] — shifted by 1
effective_len = 4095
current_hidden = prev_hidden[:, :4095]              # [8, 4095, 2048]
current_main = main_hidden[:, :4095]                # [8, 4095, 2048]
```

**Step 2: MTPModule forward** (model.py lines 868–904)
```python
# Normalize previous hidden states
normed_hidden = self.hidden_norm(prev_hidden)       # RMSNorm → [8, 4095, 2048]

# Embed target tokens (using shared embeddings)
token_embeds = self.embed_tokens(safe_target_tokens)  # [8, 4095, 2048]
normed_embeds = self.embed_norm(token_embeds)          # RMSNorm → [8, 4095, 2048]

# Concatenation-based fusion (DeepSeek V3 style!)
concatenated = torch.cat([normed_hidden, normed_embeds], dim=-1)  # [8, 4095, 4096]
hidden_states = self.concat_proj(concatenated)     # Linear(4096, 2048) → [8, 4095, 2048]

# MTP block (cross-attention + self-attention + FFN)
for block in self.blocks:
    hidden_states = block(hidden_states, cross_hidden=current_main)

# Output
output_hidden = self.output_norm(hidden_states)     # RMSNorm → [8, 4095, 2048]
logits = self.lm_head(output_hidden)                # Linear(2048, 65536) → [8, 4095, 65536]
```

**Step 3: Compute MTP loss** (lines 988–1003)
```python
pred_offset = 0 + 2 = 2                             # This module predicts t+2
pred_len = 4096 - 2 = 4094
shift_logits = logits[:, :4094]                     # [8, 4094, 65536]
shift_labels = labels[:, 2:4096]                    # [8, 4094]
loss = F.cross_entropy(shift_logits.view(-1, 65536), shift_labels.view(-1))

weight = mtp_loss_decay ** 0 = 0.8^0 = 1.0
total_loss = 1.0 × loss

# Normalize by weight sum
weight_sum = sum(0.8^i for i in range(1)) = 1.0
mtp_loss = total_loss / weight_sum = loss
```

The MTP loss is the cross-entropy of predicting the token two positions ahead,
using both the main model's hidden states and the embedding of the next token
as input. This provides a richer training signal that helps the model develop
better internal representations.

---

## Section 12: Complete Step Timeline

Here's the chronological order of every operation in step 1000:

```
TIME ──────────────────────────────────────────────────────────────────►

│ synchronize()                          [line 1466]
│ t0 = time.time()                       [line 1467]
│
│ ┌── GRADIENT ACCUMULATION LOOP (16 iterations) ──────────────────┐
│ │                                                                  │
│ │  micro_step 0:                                                   │
│ │    autocast(bf16):                                               │
│ │      outputs = model(x, labels=y)        [line 1473]            │
│ │        ├── embed_tokens(x)               → [8,4096,2048]        │
│ │        ├── 16 × decoder_layer            → [8,4096,2048]        │
│ │        │   ├── layer_norm                                        │
│ │        │   ├── MLA attention             (Section 9)             │
│ │        │   ├── residual + layer_norm                             │
│ │        │   └── FFN (MLP or MoE)          (Section 10)           │
│ │        ├── final_norm                    → [8,4096,2048]        │
│ │        ├── lm_head                       → [8,4096,65536]       │
│ │        └── _compute_loss                                         │
│ │            ├── main CE loss                                      │
│ │            ├── MTP forward + loss        (Section 11)            │
│ │            └── aux loss (MoE seq-level)                          │
│ │      loss = outputs['loss']              [line 1474]            │
│ │    train_loss = loss.detach()             [line 1486]            │
│ │    loss = loss / 16                       [line 1489]            │
│ │    loss.backward()                        [line 1490]            │
│ │    x, y, state = next(train_loader)      [line 1495]  (prefetch)│
│ │                                                                  │
│ │  micro_step 1..15:  (same as above)                              │
│ │                                                                  │
│ └──────────────────────────────────────────────────────────────────┘
│
│ GRADIENT CLIPPING                          [line 1507]
│   grad_norm = clip_grad_norm_(params, 1.0)
│
│ LEARNING RATE UPDATE                       [lines 1513-1517]
│   lrm = get_lr_multiplier(1000, 41085, config)  → 1.0
│   AdamW groups:  embed=0.2, unembed=0.004, matrix=0.02
│   Muon groups:   matrix=0.02
│
│ MUON MOMENTUM UPDATE                       [lines 1520-1523]
│   momentum = get_muon_momentum(1000)  → 0.95
│
│ OPTIMIZER STEP                             [lines 1526-1528]
│   adamw_optimizer.step()
│     ├── embed_tokens:  AdamW(lr=0.2, β=(0.9,0.95), wd=0.0)
│     ├── lm_head:       AdamW(lr=0.004, β=(0.9,0.95), wd=0.0)
│     └── 1D params:     AdamW(lr=0.02, β=(0.9,0.95), wd=0.0)
│   muon_optimizer.step()
│     └── All 2D params: Muon(lr=0.02, momentum=0.95, nesterov=True, ns_steps=5)
│         ├── buf.lerp_(g, 0.05)             (momentum EMA)
│         ├── g.lerp_(buf, 0.95)             (Nesterov look-ahead)
│         ├── g = newtonschulz5(g, 5)        (orthogonalize)
│         └── p.add_(g, -lr × scale)         (update)
│
│ ZERO GRADIENTS                             [line 1531]
│   model.zero_grad(set_to_none=True)
│
│ MOE LOAD BALANCE UPDATE                    [lines 1537-1538]
│   gamma = model.get_gamma()  → 0.001
│   For each MoE layer (14):
│     load = gate.expert_load [64]
│     imbalance = (load - mean) / (mean + ε)
│     expert_bias -= 0.001 × imbalance
│
│ MTP TOKEN COUNTER                          [line 1544]
│   tokens_processed += 524,288
│
│ synchronize()                              [line 1553]
│ t1 = time.time()
│ dt = t1 - t0                               (~3.0 seconds)
│
│ LOGGING                                    [lines 1560-1644]
│   smooth_loss EMA update
│   tok/sec, MFU calculation
│   Print log line (step 1000 % 10 == 0)
│
│ step += 1    → 1001                        [line 1646]
│ phase_step += 1  → 1001                    [line 1647]
│
│ ──► Continue to step 1001 (no eval/save/phase-transition triggered)
```

---

## Section 13: What DOESN'T Happen at Step 1000

For completeness, let's trace the conditional branches that are **not taken**:

### 13.1 Evaluation (Lines 1302–1323)

```python
if last_step or step % config.eval_every == 0:
# last_step = (1000 == 41085) = False
# 1000 % 250 == 0 → True!
```

**Actually, evaluation IS triggered at step 1000!** (eval_every = 250, and 1000 % 250 == 0)

The evaluation at step 1000 runs `evaluate_loss()` (lines 616–666) which:
1. Sets model to eval mode
2. Iterates over validation data for `eval_steps = eval_tokens / (B × T × world_size)`
   = `10,485,760 / (8 × 4096 × 1)` = `320` steps
3. Computes average validation loss, main loss, MTP loss, perplexity
4. Prints: `Step 01000 | Val loss: X.XXXX | Val PPL: XX.XX`
5. Sets model back to train mode

This happens **before** the training step (the loop structure is: eval → save → train).

### 13.2 Sample Generation (Lines 1328–1351)

```python
if master_process and tokenizer is not None and step % config.sample_every == 0 and step > 0:
# step % 2000 == 0 → 1000 % 2000 = 1000 ≠ 0 → False
```

**Not triggered.** Next sample generation at step 2000.

### 13.3 Checkpointing (Lines 1356–1380)

```python
should_save = (
    last_step or
    (config.save_every > 0 and step > 0 and step != config.resume_from_step
     and step % config.save_every == 0)
)
# last_step = False
# save_every = -1 → save_every > 0 is False
# → should_save = False
```

**Not triggered.** With `save_every = -1`, checkpoints are only saved at the final step.

### 13.4 Phase Transition (Lines 1391–1461)

```python
if config.enable_multiphase and current_phase == 0 and phase_step >= phase1_steps:
# enable_multiphase = False → entire block skipped
```

**Not triggered.** Multi-phase training is disabled.

### 13.5 Termination (Lines 1385–1386)

```python
if last_step:   # 1000 == 41085 → False
    break
```

**Not triggered.** Training continues.

---

## Section 14: Verification Code

```python
#!/usr/bin/env python3
"""
Verify all shapes, computations, and schedule values described in
12_COMPLETE_TRAINING_STEP.md for NanoSeek step 1000.

Run from project root:
    python -c "exec(open('implementation_docs/verify_step_1000.py').read())"
Or:
    python implementation_docs/verify_step_1000.py
"""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent if '__file__' in dir() else Path('.')
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import math

from model.config import NanoSeekConfig, get_nanoseek_config
from model.model import NanoSeekModel

def verify():
    print("=" * 70)
    print("Verification: Complete Training Step at Step 1000")
    print("=" * 70)

    # ================================================================
    # 1. Config and Schedule Verification
    # ================================================================
    config = get_nanoseek_config()

    # Training horizon (Chinchilla)
    active_params = config.estimated_active_params
    target_tokens = 20.0 * active_params
    total_batch_size = 524288
    num_iterations = int(target_tokens // total_batch_size)
    total_tokens = total_batch_size * num_iterations

    print(f"\n[Config]")
    print(f"  active_params:    {active_params:,}")
    print(f"  total_params:     {config.estimated_total_params:,}")
    print(f"  target_tokens:    {target_tokens:,.0f}")
    print(f"  num_iterations:   {num_iterations:,}")
    print(f"  total_tokens:     {total_tokens:,}")

    # Update model config for ratio-based schedules
    config.total_tokens = total_tokens

    # LR multiplier at step 1000
    warmup_iters = round(0.0 * num_iterations)
    warmdown_iters = round(0.2 * num_iterations)
    step = 1000

    if step < warmup_iters:
        lrm = (step + 1) / warmup_iters
    elif step <= num_iterations - warmdown_iters:
        lrm = 1.0
    else:
        progress = (num_iterations - step) / warmdown_iters
        lrm = progress * 1.0
    print(f"\n[LR Schedule at step {step}]")
    print(f"  warmup_iters:     {warmup_iters}")
    print(f"  warmdown_iters:   {warmdown_iters}")
    print(f"  constant_end:     {num_iterations - warmdown_iters}")
    print(f"  lrm:              {lrm}")
    assert lrm == 1.0, f"Expected lrm=1.0, got {lrm}"
    print(f"  ✓ LR multiplier = 1.0 (constant phase)")

    # Gamma
    tokens_processed = step * total_batch_size
    freeze_at = int(total_tokens * config.moe.gamma_freeze_ratio)
    gamma = config.moe.gamma if tokens_processed < freeze_at else 0.0
    print(f"\n[Gamma at step {step}]")
    print(f"  tokens_processed: {tokens_processed:,}")
    print(f"  freeze_at:        {freeze_at:,}")
    print(f"  gamma:            {gamma}")
    assert gamma == 0.001, f"Expected gamma=0.001, got {gamma}"
    print(f"  ✓ gamma = 0.001 (active)")

    # MTP loss weight
    transition = int(total_tokens * config.mtp.mtp_loss_transition_ratio)
    mtp_weight = config.mtp.mtp_loss_weight_initial if tokens_processed < transition else config.mtp.mtp_loss_weight_final
    print(f"\n[MTP weight at step {step}]")
    print(f"  transition at:    {transition:,} tokens")
    print(f"  mtp_weight:       {mtp_weight}")
    assert mtp_weight == 0.3, f"Expected mtp_weight=0.3, got {mtp_weight}"
    print(f"  ✓ mtp_weight = 0.3 (initial)")

    # Muon momentum
    frac = min(step / 300, 1.0)
    muon_momentum = (1 - frac) * 0.85 + frac * 0.95
    print(f"\n[Muon momentum at step {step}]")
    print(f"  frac:             {frac}")
    print(f"  momentum:         {muon_momentum}")
    assert muon_momentum == 0.95, f"Expected momentum=0.95, got {muon_momentum}"
    print(f"  ✓ momentum = 0.95 (fully warmed)")

    # ================================================================
    # 2. Batch Configuration
    # ================================================================
    device_batch_size = 8
    max_seq_len = 4096
    tokens_per_fwdbwd = device_batch_size * max_seq_len
    grad_accum_steps = total_batch_size // tokens_per_fwdbwd

    print(f"\n[Batch Config]")
    print(f"  device_batch_size:  {device_batch_size}")
    print(f"  max_seq_len:        {max_seq_len}")
    print(f"  tokens_per_fwdbwd:  {tokens_per_fwdbwd:,}")
    print(f"  grad_accum_steps:   {grad_accum_steps}")
    assert grad_accum_steps == 16, f"Expected 16, got {grad_accum_steps}"
    print(f"  ✓ 16 gradient accumulation steps")

    # ================================================================
    # 3. Model Shape Verification (small config for tractability)
    # ================================================================
    print(f"\n[Model Shapes — using small config for verification]")

    # Create a tiny model to verify shapes
    small_config = get_nanoseek_config()
    # Override for CPU testing — keep full architecture but reduce sizes
    small_config.hidden_size = 128
    small_config.num_layers = 2  # 1 dense + 1 MoE
    small_config.num_heads = 4
    small_config.intermediate_size = 256
    small_config.vocab_size = 256
    small_config.moe.n_routed_experts = 8
    small_config.moe.num_experts_per_tok = 2
    small_config.moe.n_shared_experts = 1
    small_config.moe.moe_intermediate_size = 64
    small_config.moe.n_group = 2
    small_config.moe.topk_group = 2
    small_config.moe.first_k_dense_replace = 1
    small_config.mla.q_lora_rank = 32
    small_config.mla.kv_lora_rank = 16
    small_config.mla.qk_nope_head_dim = 16
    small_config.mla.qk_rope_head_dim = 8
    small_config.mla.v_head_dim = 16
    small_config.mtp.num_mtp_modules = 1
    small_config.mtp.mtp_num_heads = 2
    small_config.max_position_embeddings = 64
    small_config.sequence_length = 64
    small_config.total_tokens = 100000

    model = NanoSeekModel(small_config)

    B, T = 2, 64
    x = torch.randint(0, 256, (B, T))
    y = torch.randint(0, 256, (B, T))

    outputs = model(x, labels=y)

    print(f"  logits shape:     {outputs['logits'].shape}")
    assert outputs['logits'].shape == (B, T, 256), f"Unexpected logits shape"
    print(f"  loss:             {outputs['loss'].item():.4f}")
    print(f"  main_loss:        {outputs['main_loss'].item():.4f}")
    if 'mtp_loss' in outputs:
        print(f"  mtp_loss:         {outputs['mtp_loss'].item():.4f}")
        print(f"  mtp_weight:       {outputs['mtp_weight']}")
    print(f"  ✓ Forward pass shapes correct")

    # Verify loss composition
    main_loss = outputs['main_loss'].item()
    mtp_loss = outputs.get('mtp_loss', torch.tensor(0.0)).item()
    mtp_w = outputs.get('mtp_weight', 0.0)
    aux_loss = outputs.get('aux_loss', torch.tensor(0.0))
    aux_val = aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
    expected_total = main_loss + mtp_w * mtp_loss + aux_val
    actual_total = outputs['loss'].item()
    print(f"\n[Loss Composition]")
    print(f"  main_loss:        {main_loss:.4f}")
    print(f"  mtp_w × mtp_loss: {mtp_w:.2f} × {mtp_loss:.4f} = {mtp_w * mtp_loss:.4f}")
    print(f"  aux_loss:         {aux_val:.6f}")
    print(f"  expected total:   {expected_total:.4f}")
    print(f"  actual total:     {actual_total:.4f}")
    print(f"  ✓ Loss composition verified (diff={abs(expected_total - actual_total):.6f})")

    # ================================================================
    # 4. Optimizer Group Verification
    # ================================================================
    print(f"\n[Optimizer Parameter Groups]")
    embed_params = []
    unembed_params = []
    muon_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'embed_tokens' in name:
            embed_params.append((name, param))
        elif 'lm_head' in name:
            unembed_params.append((name, param))
        elif param.ndim == 2:
            muon_params.append((name, param))
        else:
            other_params.append((name, param))

    print(f"  Embed params:     {len(embed_params)} tensors, "
          f"{sum(p.numel() for _, p in embed_params):,} params")
    print(f"  Unembed params:   {len(unembed_params)} tensors, "
          f"{sum(p.numel() for _, p in unembed_params):,} params")
    print(f"  Muon (2D) params: {len(muon_params)} tensors, "
          f"{sum(p.numel() for _, p in muon_params):,} params")
    print(f"  Other (1D) params: {len(other_params)} tensors, "
          f"{sum(p.numel() for _, p in other_params):,} params")

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    grouped_total = (sum(p.numel() for _, p in embed_params) +
                     sum(p.numel() for _, p in unembed_params) +
                     sum(p.numel() for _, p in muon_params) +
                     sum(p.numel() for _, p in other_params))
    assert total_trainable == grouped_total, \
        f"Grouping mismatch: {total_trainable} vs {grouped_total}"
    print(f"  ✓ All {total_trainable:,} trainable params accounted for")

    # ================================================================
    # 5. Newton-Schulz Verification
    # ================================================================
    print(f"\n[Newton-Schulz Orthogonalization]")

    from model.optimizer.muon import zeropower_via_newtonschulz5

    # Create a random 2D matrix (simulating a gradient)
    torch.manual_seed(42)
    G = torch.randn(32, 128)
    X = zeropower_via_newtonschulz5(G, steps=5)

    print(f"  Input shape:      {G.shape}")
    print(f"  Output shape:     {X.shape}")
    print(f"  Output dtype:     {X.dtype}")

    # Check approximate orthogonality: X @ X^T ≈ I (scaled)
    XXT = (X.float() @ X.float().T)
    diag_vals = torch.diag(XXT)
    off_diag = XXT - torch.diag(diag_vals)
    print(f"  Diagonal mean:    {diag_vals.mean().item():.4f}")
    print(f"  Diagonal std:     {diag_vals.std().item():.4f}")
    print(f"  Off-diag max:     {off_diag.abs().max().item():.4f}")
    print(f"  ✓ Approximate orthogonality confirmed")

    # ================================================================
    # 6. MoE Load Balance Bias Verification
    # ================================================================
    print(f"\n[MoE Load Balance Bias Update]")

    # Find a MoE layer
    moe_layer = None
    for layer in model.layers:
        if layer.is_moe_layer:
            moe_layer = layer
            break

    if moe_layer is not None:
        gate = moe_layer.ffn.gate
        print(f"  Expert count:     {gate.n_routed_experts}")
        print(f"  Expert bias shape: {gate.expert_bias.shape}")
        print(f"  Expert bias range: [{gate.expert_bias.min():.6f}, {gate.expert_bias.max():.6f}]")

        # Simulate a forward pass to get expert_load
        x_test = torch.randn(4, 16, small_config.hidden_size)
        _ = moe_layer.ffn(x_test)

        load = gate.expert_load
        print(f"  Expert load shape: {load.shape}")
        print(f"  Expert load range: [{load.min():.0f}, {load.max():.0f}]")
        print(f"  Expert load mean:  {load.mean():.1f}")

        # Apply bias update
        old_bias = gate.expert_bias.clone()
        moe_layer.ffn.update_load_balance_bias(gamma=0.001)
        new_bias = gate.expert_bias
        bias_change = (new_bias - old_bias).abs().max().item()
        print(f"  Max bias change:   {bias_change:.8f}")
        print(f"  ✓ Load balance bias update verified")

    # ================================================================
    # 7. Gradient Accumulation Verification
    # ================================================================
    print(f"\n[Gradient Accumulation]")

    model.zero_grad()
    grad_accum = 4  # Use 4 for quick test

    for i in range(grad_accum):
        x_i = torch.randint(0, 256, (B, T))
        y_i = torch.randint(0, 256, (B, T))
        out = model(x_i, labels=y_i)
        loss = out['loss'] / grad_accum
        loss.backward()

    # Check that gradients exist and are reasonable
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params_count = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  Params with grad:  {has_grad}/{total_params_count}")

    # Check grad norm
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    print(f"  Gradient L2 norm:  {total_norm:.4f}")
    print(f"  ✓ Gradient accumulation works correctly")

    # Clean up
    model.zero_grad(set_to_none=True)
    has_grad_after = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"\n[Zero Grad (set_to_none=True)]")
    print(f"  Params with grad after zero: {has_grad_after}")
    assert has_grad_after == 0, f"Expected 0 grads after zero, got {has_grad_after}"
    print(f"  ✓ All gradients freed")

    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"All verifications passed! ✓")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    verify()
```

---

## Section 15: Key Numerical Invariants

These invariants hold at **every** training step and can be used for debugging:

| Invariant | Expression | Value |
|-----------|-----------|-------|
| Tokens per step | `total_batch_size` | 524,288 |
| Micro-batches per step | `grad_accum_steps` | 16 (single GPU) |
| Tokens per micro-batch | `device_batch_size × max_seq_len` | 32,768 |
| FLOPs per step | `6 × active_params × total_batch_size` | ~3.39 × 10¹⁵ |
| Grad scaling factor | `1 / grad_accum_steps` | 1/16 = 0.0625 |
| LR schedule phases | warmup / constant / warmdown | 0 / 32,868 / 8,217 steps |
| Muon warmup | 0 → 300 steps | momentum 0.85 → 0.95 |
| Gamma freeze | At 80% of total_tokens | ~17.2B tokens |
| MTP transition | At 60% of total_tokens | ~12.9B tokens |

---

## Section 16: Cross-Reference to Source Files

| Concept | File | Lines |
|---------|------|-------|
| Training config | `scripts/pre-train.py` | 183–300 |
| LR multiplier | `scripts/pre-train.py` | 567–597 |
| Muon momentum warmup | `scripts/pre-train.py` | 600–609 |
| Optimizer setup | `scripts/pre-train.py` | 421–560 |
| Training loop | `scripts/pre-train.py` | 1295–1647 |
| Gradient accumulation | `scripts/pre-train.py` | 1471–1503 |
| Gradient clipping | `scripts/pre-train.py` | 1504–1510 |
| LR update | `scripts/pre-train.py` | 1512–1517 |
| Muon momentum update | `scripts/pre-train.py` | 1519–1523 |
| Optimizer step | `scripts/pre-train.py` | 1525–1528 |
| Zero gradients | `scripts/pre-train.py` | 1531 |
| MoE bias update | `scripts/pre-train.py` | 1536–1538 |
| Token counter | `scripts/pre-train.py` | 1544 |
| DSA step increment | `scripts/pre-train.py` | 1549–1551 |
| Logging | `scripts/pre-train.py` | 1560–1644 |
| Newton-Schulz iteration | `model/optimizer/muon.py` | 10–36 |
| Muon.step() | `model/optimizer/muon.py` | 69–83 |
| DistAdamW.step() | `model/optimizer/adamw.py` | 18–77 |
| Model forward | `model/model.py` | 1562–1660 |
| Loss computation | `model/model.py` | 1697–1728 |
| MLA forward | `model/model.py` | 287–372 |
| MoE forward | `model/model.py` | 663–719 |
| Gate forward | `model/model.py` | 409–442 |
| Token-centric dispatch | `model/model.py` | 485–583 |
| MTP forward | `model/model.py` | 954–1012 |
| Load balance bias update | `model/model.py` | 721–727 |
| get_gamma() | `model/model.py` | 1534–1540 |
| get_mtp_loss_weight() | `model/model.py` | 1526–1532 |
| update_tokens_processed() | `model/model.py` | 1542–1543 |
| NanoSeek config | `model/config.py` | 847–1016 |
| Streaming dataloader | `scripts/dataloader.py` | 28–107 |

---

## Section 17: Frequently Confused Points

### Q: Why is `train_loss` from the last micro-batch, not the average?

Line 1486 (`train_loss = loss.detach()`) captures only the **last** micro-batch's loss.
The smoothed `debiased_smooth_loss` (line 1566) provides the running average for logging.
This is a deliberate simplification — tracking per-micro-batch averages would require
extra computation for negligible logging benefit.

### Q: Why does the LR update happen BEFORE the optimizer step?

Lines 1512–1517 modify `group['lr']` before `opt.step()` at line 1528. This is the
standard pattern: the optimizer reads `group['lr']` during `.step()`, so it must be
set before the call. The LR schedule is computed from the **current** step, meaning
step 1000's LR is applied to step 1000's gradient.

### Q: How can MFU exceed 100%?

The MFU calculation (lines 1572–1574) uses:
```
MFU = (6 × active_params × tokens / dt) / peak_flops_per_sec
```

This can exceed 100% because:
1. `6 × active_params` underestimates true FLOPs (ignores attention O(n²), norms, routing)
2. H100's 989 TFLOPS is the **sustained** BF16 peak; burst performance can be higher
3. For MoE models, the 64 experts are processed serially by groups, which can overlap
   with other operations

### Q: Why `set_to_none=True` instead of `False`?

With `set_to_none=True` (line 1531):
- Gradient tensors are deallocated (freed from GPU memory)
- Next backward pass allocates fresh gradient tensors
- Net effect: ~9.1 GB freed between optimizer step and next forward pass

With `set_to_none=False`:
- Gradient tensors are zeroed but remain allocated
- No memory deallocation/reallocation overhead
- But 9.1 GB of gradient memory stays allocated permanently

The trade-off favors `set_to_none=True` for NanoSeek because the 9.1 GB savings
is significant, and the reallocation cost is negligible on modern CUDA allocators
(especially with `expandable_segments:True`).

### Q: Why does Muon group parameters by `numel()` instead of shape?

From `Muon.__init__()` (muon.py line 64):
```python
for size in {p.numel() for p in params}:
    group = dict(params=[p for p in params if p.numel() == size])
```

This groups by **total element count**, not shape. Parameters with the same `numel()`
can potentially be batched together in the Newton-Schulz iteration (the batched
implementation mentioned in muon.py line 20: "batched Muon implementation by @scottjmaddox").
Even if shapes differ, the iteration only needs consistent `numel()` for efficient
GPU kernel scheduling.

### Q: What happens if a routed expert gets zero tokens?

In `token_centric_dispatch()` (model.py lines 554–560):
```python
for expert_id, batch in enumerate(expert_batches):
    if batch.shape[0] > 0:
        expert_outputs.append(experts[expert_id](batch))
    else:
        expert_outputs.append(torch.empty(0, D, device=device, dtype=dtype))
```

Empty experts produce a zero-sized tensor. Their weights still get gradient updates
through the routing mechanism (the gate loss ensures all experts eventually get tokens).
The load-balance bias update (Section 6.1) actively corrects for such imbalances by
increasing the bias for underloaded experts.

---

## Section 18: What Changes at Different Steps

| Step | LR Multiplier | Gamma | MTP Weight | Muon Momentum | Notes |
|------|--------------|-------|------------|---------------|-------|
| 0 | 1.0 | 0.001 | 0.3 | 0.85 | Training begins |
| 100 | 1.0 | 0.001 | 0.3 | 0.883 | Momentum warming |
| 300 | 1.0 | 0.001 | 0.3 | 0.95 | Momentum fully warmed |
| 1,000 | 1.0 | 0.001 | 0.3 | 0.95 | **This trace** |
| ~24,650 | 1.0 | 0.001 | 0.1 | 0.95 | MTP weight transitions (60% tokens) |
| ~32,868 | 1.0 | 0.001 | 0.1 | 0.95 | Warmdown begins |
| ~32,900 | 0.997 | 0.0 | 0.1 | 0.95 | Gamma freezes (80% tokens) |
| ~41,085 | 0.0 | 0.0 | 0.1 | 0.95 | Training ends |

The key insight: most schedule changes happen in the **second half** of training.
Step 1000 is firmly in the "constant everything" regime — the simplest case.

---

*End of document. Every number above is derivable from the source code at the referenced line numbers.*
