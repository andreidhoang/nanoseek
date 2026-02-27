# The Complete DeepSeek V3.2 Training Pipeline — First Principles Deep Dive

> **Document status**: Production reference for NanoSeek pre-training
> **Source of truth**: `scripts/pre-train.py`, `scripts/scheduler.py`, `model/optimizer/muon.py`, `model/optimizer/adamw.py`, `model/model.py`, `model/config.py`
> **Audience**: Senior ML researchers who have trained transformers at scale and need to understand every optimizer step, every schedule transition, and every distributed synchronization point in the NanoSeek training pipeline.

---

## Quick Context

**What this document covers.** The complete end-to-end training pipeline for NanoSeek-1B, a nano-scale faithful reimplementation of DeepSeek V3.2. We trace one token from raw text on disk through tokenization, batching, mixed-precision forward pass, composite loss computation, gradient accumulation, dual-optimizer update (Muon + AdamW), dynamic schedule adjustments, and checkpoint serialization. Every hyperparameter is cited from the actual codebase. No hand-waving.

**Why training pipelines are hard.** A model architecture is a function. A training pipeline is a *system*: optimizers with internal state, learning rate schedules with phase transitions, load-balancing controllers running outside the gradient graph, loss weights that change mid-run, distributed communication patterns that interleave with computation, and checkpoint formats that must allow exact resumption across all of these. Getting any single component wrong doesn't cause a crash — it causes a 14-hour training run that produces a worse model than you expected, and you won't know why until you've audited everything.

**What makes this pipeline distinctive.**

| Component | Standard Practice | NanoSeek / DeepSeek V3.2 |
|---|---|---|
| Optimizer for matrices | AdamW everywhere | **Muon** (SGD + Newton-Schulz orthogonalization) |
| Optimizer for embeddings | Same AdamW | **Separate AdamW** with decoupled LR (0.2 vs 0.02) |
| LR schedule | Cosine from step 0 | **4-phase**: warmup → constant (70%) → cosine decay → floor |
| Load balancing | Auxiliary loss in gradient | **Bias adjustment outside gradient** (aux-loss-free) |
| MTP loss weight | Constant λ | **Dynamic**: λ=0.3 → λ=0.1 at 60% of training |
| Training phases | Single phase | **Multi-phase**: Dense MLA → Sparse DSA with context extension |
| Distributed optimizer | Standard DDP AllReduce | **ZeRO-2 style** sharded states (DistMuon, DistAdamW) |

**Scale reference.** NanoSeek-1B has 1.08B active parameters (4.75B total with all 64 MoE experts). Chinchilla-optimal training: 22B tokens, ~42,000 steps at 512K tokens/step, 8×H100, ~14 hours, ~$300.

---

## 🔴 STAGE 1: INPUTS — Raw Text → Tokenized Sequences → Batched Tensors

### 1.1 Data Source: FineWeb-Edu in Parquet

NanoSeek streams training data from parquet files using PyArrow. The data is FineWeb-Edu — curated web text filtered for educational content.

```
Parquet file on disk
  → PyArrow reads row groups (each ~1024 rows of text)
  → Each row is a document (variable-length string)
  → Documents are batched for tokenization (128 rows at a time)
```

The streaming approach means we never load the entire dataset into memory. This is essential for multi-billion-token training budgets.

### 1.2 Tokenization: On-the-Fly with Multi-Threading

From `scripts/dataloader.py`, the `tokenizing_distributed_data_loader_with_state` function:

```python
tokenizer = get_tokenizer()
bos_token = tokenizer.get_bos_token_id()
token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
```

Each document gets a BOS (beginning-of-sequence) token prepended. Tokenization is parallelized across `tokenizer_threads=4` threads. The tokenizer has a vocabulary of 65,536 tokens.

**Why on-the-fly?** Pre-tokenizing 22B tokens would produce ~44GB of int32 token IDs. Streaming tokenization trades CPU cycles for storage and startup time. With 4 threads and batch size 128, tokenization throughput exceeds training consumption rate on H100s.

### 1.3 Token Buffer: Accumulate and Reshape

The dataloader uses a `deque`-based token buffer that accumulates tokens from multiple documents:

```python
needed_tokens = B * T + 1  # +1 for target at last position
token_buffer = deque()

while len(token_buffer) < needed_tokens:
    doc_batch, (pq_idx, rg_idx) = next(batches)
    token_lists = tokenizer.encode(doc_batch, ...)
    for tokens in token_lists:
        token_buffer.extend(tokens)

tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=True)
inputs = scratch[:-1].view(B, T).to(device, non_blocking=True)
targets = scratch[1:].view(B, T).to(device, non_blocking=True)
```

The `+1` is critical: we need `B*T` input tokens and `B*T` target tokens, offset by one position. The `scratch[:-1]` / `scratch[1:]` split creates the standard autoregressive (input, target) pairs.

**Shape at this point:**

```
inputs:  [B, T]     = [8, 4096]  = [device_batch_size, max_seq_len]
targets: [B, T]     = [8, 4096]
dtype: torch.long (int64)
device: cuda:local_rank (after async transfer)
```

### 1.4 DDP-Aware Data Partitioning

Each rank reads different row groups from the parquet files:

```python
rg_idx = ddp_rank  # Start at rank's offset
while rg_idx < pf.num_row_groups:
    rg = pf.read_row_group(rg_idx)
    ...
    rg_idx += ddp_world_size  # Stride by world size
```

With 8 GPUs, rank 0 reads row groups 0, 8, 16, ...; rank 1 reads 1, 9, 17, ...; etc. This ensures no data overlap between ranks without explicit coordination.

### 1.5 Approximate Resume via State Tracking

The dataloader returns a `state_dict` with every batch:

```python
state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx}
yield inputs, targets, state_dict
```

On resume, this state is passed back to skip to approximately the right position. The resumption is *approximate* — we advance one full `ddp_world_size` block past the saved position to guarantee no data repetition, at the cost of potentially skipping a few documents.

**Why approximate?** Exact resumption would require tracking the precise byte offset within a row group's tokenized output, plus the exact state of the token buffer deque. The engineering complexity is not worth the ~0.001% data waste.

### 1.6 Gradient Accumulation: From Micro-batch to Global Batch

The per-device batch of `[8, 4096]` is a *micro-batch*. The full training step operates on a much larger global batch:

```python
tokens_per_fwdbwd = device_batch_size * max_seq_len        # 8 × 4096 = 32,768
world_tokens_per_fwdbwd = tokens_per_fwdbwd * world_size   # 32,768 × 8 = 262,144
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd  # 524,288 / 262,144 = 2
```

With 8 GPUs and device_batch_size=8, we need 2 gradient accumulation steps to reach the 512K tokens/step global batch size.

---

## 🔴 STAGE 2: ARCHITECTURE — The Training Loop as a System

This is the heart of the document. Stage 2 covers every component that makes the training loop a system, not just a forward-backward-update cycle.

### 2.1 Optimizer Architecture: The Muon + AdamW Hybrid

NanoSeek uses two fundamentally different optimizers for different parameter types. This is not a minor implementation detail — it is a core architectural decision that affects convergence, stability, and final model quality.

**Parameter classification** (from `setup_optimizers` in `scripts/pre-train.py`):

```python
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if 'embed_tokens' in name:
        embed_params.append(param)        # → AdamW, lr=0.2
    elif 'lm_head' in name:
        unembed_params.append(param)      # → AdamW, lr=0.004
    elif param.ndim == 2:
        muon_params.append(param)         # → Muon, lr=0.02
    else:
        adamw_matrix_params.append(param) # → AdamW, lr=0.02
```

The `param.ndim == 2` check is the key discriminator. Every weight matrix in attention projections and FFN layers is 2D and goes to Muon. Everything else — embeddings (2D but explicitly routed), layer norm weights (1D), biases (1D), scalar parameters (0D) — goes to AdamW.

**Why different optimizers?** Muon's Newton-Schulz orthogonalization is designed for matrices where the update should preserve approximate orthogonality. This is natural for attention projections (which learn rotations in representation space) and FFN weight matrices. Embeddings are lookup tables, not linear transformations — orthogonalizing their updates makes no geometric sense.

**Why different learning rates?** The embedding matrix (`embed_tokens`) maps discrete token IDs to continuous vectors. It needs aggressive updates early in training to establish good representations (lr=0.2). The unembedding matrix (`lm_head`) projects from hidden space to vocabulary logits — it's the final classifier and needs more conservative updates (lr=0.004). Matrix parameters use lr=0.02, which is 100× larger than typical AdamW learning rates because Muon's orthogonalization step normalizes the update magnitude.

### 2.2 Muon: Newton-Schulz Orthogonalization from First Principles

Muon (MomentUm Orthogonalized by Newton-schulz) is the optimizer that makes NanoSeek's training distinctive. Let's derive it from first principles.

**The core idea.** Standard SGD with momentum computes an update direction, then steps in that direction. Muon adds one more operation: before stepping, it *orthogonalizes* the update matrix. Specifically, it replaces the update `G` with its closest orthogonal matrix `UV^T` (where `USV^T = G` is the SVD).

**Why orthogonalize?** In a linear layer `y = Wx`, the weight matrix `W` acts as a linear transformation. The "quality" of this transformation depends on its singular value spectrum. If all singular values are similar (the matrix is well-conditioned), the layer preserves information equally across all directions. Orthogonal updates encourage the weight matrix to maintain a healthy singular value spectrum throughout training.

**The Newton-Schulz iteration.** Computing the exact SVD is expensive (`O(min(m,n)²·max(m,n))`). Newton-Schulz iteration approximates `UV^T` iteratively and can be implemented entirely with matrix multiplications — which GPUs are optimized for.

The standard Newton-Schulz iteration for computing `X = UV^T` from `G = USV^T`:

```
X₀ = G / ||G||
Xₖ₊₁ = Xₖ · (3I - Xₖᵀ Xₖ) / 2
```

This converges to `UV^T` when the singular values of `X₀` are in `(0, √3)`.

**The quintic variant (NanoSeek implementation).** NanoSeek uses a *quintic* (5th-order) iteration from `zeropower_via_newtonschulz5` in `model/optimizer/muon.py`:

```python
a, b, c = (3.4445, -4.7750, 2.0315)
X = G.bfloat16()
if G.size(-2) > G.size(-1):
    X = X.mT  # Work with the transpose if tall

X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)  # Normalize spectral norm to ~1

for _ in range(steps):  # steps = 5
    A = X @ X.mT
    B = b * A + c * A @ A
    X = a * X + B @ X
```

**Deriving the coefficients.** The quintic polynomial `f(s) = a·s + b·s³ + c·s⁵` is chosen to maximize the slope at zero — i.e., the convergence rate for small singular values. The coefficients `a=3.4445, b=-4.7750, c=2.0315` satisfy:

1. `f(1) = 1`: Fixed point at the target (singular value 1)
2. `f'(0)` is maximized: Fastest convergence for small singular values
3. `a + b + c = 1`: Consistency constraint

After 5 iterations, the result is *not* exactly `UV^T` but rather `US'V^T` where `S'` has diagonal entries approximately uniform in `[0.5, 1.5]`. Empirically, this slightly noisy orthogonalization works just as well as exact orthogonalization — the key property is that extreme singular values are compressed, not that they're all exactly 1.

**Why the transpose for tall matrices?** If `G` is `m × n` with `m > n`, working with `G^T` (which is `n × m`, wider than tall) makes the iteration numerically better behaved and reduces the number of operations (the `A = X @ X^T` product is `n × n` instead of `m × m`).

**The complete Muon step.** Here is the full single-GPU update (from `Muon.step()`):

```python
# 1. SGD momentum accumulation
buf.lerp_(g, 1 - momentum)           # buf = momentum * buf + (1 - momentum) * g

# 2. Nesterov look-ahead
g = g.lerp_(buf, momentum)           # g = (1 - momentum) * g + momentum * buf
                                      # = g + momentum * (buf - g)

# 3. Newton-Schulz orthogonalization (5 iterations)
g = zeropower_via_newtonschulz5(g, steps=5)

# 4. Aspect-ratio-scaled step
p.add_(g, alpha=-lr * sqrt(max(1, rows/cols)))
```

**The aspect ratio scaling** `max(1, rows/cols)^0.5` ensures that tall matrices (more rows than columns) get proportionally larger updates. This compensates for the fact that the orthogonalized update has unit spectral norm regardless of aspect ratio, but the gradient magnitude naturally scales with the number of rows.

### 2.3 Muon Momentum Warmup

Muon's momentum starts at 0.85 and warms up to 0.95 over the first 300 steps:

```python
def get_muon_momentum(step: int) -> float:
    frac = min(step / 300, 1.0)
    return (1 - frac) * 0.85 + frac * 0.95
```

**Why?** High momentum (0.95) means the optimizer retains 95% of the previous update direction, making it slow to change course. Early in training, the loss landscape is essentially random — the model hasn't learned anything yet, and gradient directions are noisy. Starting with lower momentum (0.85) allows faster initial adaptation, then increasing to 0.95 provides the smoothing and acceleration benefits once the optimizer has found a productive direction.

### 2.4 DistMuon: Distributed Muon with ZeRO-2 Style Communication

In distributed training, `DistMuon` performs the same computation but with an optimized communication pattern:

```
Step 1: reduce_scatter(AVG)
  - Each rank averages gradients across all ranks
  - Output: each rank gets its "owned" slice of averaged gradients
  - Block-cyclic assignment: rank r owns params r, r+W, r+2W, ...

Step 2: Owner computes Muon update
  - Only the owning rank runs momentum → nesterov → orthogonalize → step
  - Other ranks skip the computation for this parameter

Step 3: all_gather
  - Owner broadcasts updated parameter to all ranks
```

This is ZeRO-2 style: optimizer states (momentum buffers) are sharded across ranks. Each rank only maintains momentum buffers for its owned parameters. For NanoSeek with 8 GPUs, each rank stores 1/8 of the Muon momentum buffers.

**Communication pattern in detail** (from `DistMuon.step()`):

```python
for base_i in range(0, len(params), world_size):
    owner_idx = base_i + rank
    rs_input = [p.grad for p in params[base_i:base_i + world_size]]
    rs_output = params[owner_idx].grad
    dist.reduce_scatter(rs_output, rs_input, op=dist.ReduceOp.AVG, async_op=True)
```

Parameters are processed in groups of `world_size`. Within each group, rank `r` owns the `r`-th parameter. The `reduce_scatter` averages all ranks' gradients and places each rank's owned slice into the corresponding output buffer. All operations are async for overlap with computation.

### 2.5 AdamW: The Embedding Optimizer

For embeddings and non-2D parameters, NanoSeek uses standard AdamW with non-default betas:

```python
betas = (0.9, 0.95)  # NOT the default (0.9, 0.999)!
```

**Why beta2=0.95 instead of 0.999?** The second moment `v_t = beta2 * v_{t-1} + (1-beta2) * g_t²` tracks the exponential moving average of squared gradients. With beta2=0.999, this average has an effective window of ~1000 steps — the optimizer "remembers" gradient magnitudes from 1000 steps ago. With beta2=0.95, the window shrinks to ~20 steps. This makes the optimizer more responsive to changing gradient distributions, which is important during the rapid phase transitions in DeepSeek-style training (constant → decay, MTP weight changes, gamma freeze).

**DistAdamW: ZeRO-2 for AdamW** (from `model/optimizer/adamw.py`):

```python
# Each parameter's first dimension is sharded across ranks
rank_size = grad.shape[0] // world_size
grad_slice = torch.empty_like(grad[:rank_size])
dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG)

# Compute Adam update on owned slice only
p_slice = p[rank * rank_size:(rank + 1) * rank_size]
exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
# ... bias correction and step ...

# Reconstruct full parameter
dist.all_gather_into_tensor(p, p_slice)
```

**Memory savings.** Adam maintains two state tensors per parameter (first moment `m`, second moment `v`), each the same size as the parameter itself. For 4.75B total parameters in fp32: `4.75B × 4 × 2 = 38GB`. With ZeRO-2 across 8 GPUs, each rank stores only ~4.75GB of optimizer state.

### 2.6 LR Schedule: The DeepSeek 4-Phase Design

NanoSeek implements two LR schedules. The pre-training script uses a simpler 3-phase approach via `get_lr_multiplier`, while `scripts/scheduler.py` provides the full DeepSeek 4-phase schedule via `DeepSeekLRScheduler`.

#### 2.6.1 The Training Script's Schedule (get_lr_multiplier)

```python
def get_lr_multiplier(step, num_iterations, config):
    warmup_iters = round(config.warmup_ratio * num_iterations)
    warmdown_iters = round(config.warmdown_ratio * num_iterations)  # 0.2

    if step < warmup_iters:
        return (step + 1) / warmup_iters           # Linear warmup: 0 → 1
    elif step <= num_iterations - warmdown_iters:
        return 1.0                                   # Constant at max
    else:
        progress = (num_iterations - step) / warmdown_iters
        return progress * 1.0 + (1 - progress) * config.final_lr_frac  # Linear decay
```

Applied to all optimizer groups via multiplier:

```python
lrm = get_lr_multiplier(step, num_iterations, config)
for opt in optimizers:
    for group in opt.param_groups:
        group['lr'] = group['initial_lr'] * lrm
```

This means all parameter groups share the same *multiplier* but different *base learning rates*. At `lrm=1.0`: embeddings get lr=0.2, unembeddings get lr=0.004, matrix params get lr=0.02.

#### 2.6.2 The DeepSeek Scheduler (DeepSeekLRScheduler)

The `DeepSeekLRScheduler` in `scripts/scheduler.py` implements the full 4-phase schedule:

```
Phase 1: Warmup (0 to warmup_steps)
  lr = lr_max × (step / warmup_steps)
  Linear ramp from 0 to lr_max

Phase 2: Constant (warmup_steps to 70% of total)
  lr = lr_max
  Extended plateau for stable gradient signal

Phase 3: Cosine Decay (70% to 95% of total)
  progress = (step - constant_end) / (decay_end - constant_end)
  lr = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × progress))
  Smooth transition from lr_max to lr_min

Phase 4: Floor (95% to 100% of total)
  lr = lr_min
  Final stabilization at minimum LR
```

**Visualized for NanoSeek-1B (42,000 steps):**

```
LR
 ↑
 lr_max ─────────────────────────────┐
 |      /                             \
 |     /                               \
 |    /                                 \________
 |   /                                    lr_min
 └────────────────────────────────────────────────→ step
    0    warmup   ~29,400          ~39,900  42,000
         (2.3%)   (70%)            (95%)   (100%)
```

**Why 70% constant?** DeepSeek's empirical finding: models benefit from extended exposure to the maximum learning rate. The constant phase allows the optimizer to explore broadly before the cosine decay narrows the search. Standard cosine schedules start decaying immediately after warmup, which can prematurely constrain exploration.

**Why 5% floor?** The final constant-at-minimum phase ensures the model doesn't continue oscillating during the last few thousand steps. It's a stabilization buffer. With lr_min = 3e-5 (10% of lr_max = 3e-4), the model still updates but very conservatively.

### 2.7 Dynamic Schedules: Gamma and MTP Weight

#### 2.7.1 Load Balancing Gamma Schedule

The MoE load balancing bias update rate `gamma` follows a simple schedule:

```python
def get_gamma(self, tokens_processed=None):
    freeze_at = int(self.config.total_tokens * self.config.moe.gamma_freeze_ratio)
    # gamma_freeze_ratio = 0.80
    if tokens_processed < freeze_at:
        return self.config.moe.gamma  # 0.001
    return 0.0  # Frozen
```

**The bias update itself** (from `MoE.update_load_balance_bias`):

```python
load = self.gate.expert_load        # [n_routed_experts] = [64]
mean_load = load.mean()
imbalance = (load - mean_load) / (mean_load + 1e-8)
self.gate.expert_bias.sub_(gamma * imbalance)
```

Translation: If expert `i` is receiving more tokens than average, decrease its bias (making it less likely to be selected). If receiving fewer, increase its bias. The update magnitude is proportional to the relative imbalance, scaled by gamma.

**Why freeze at 80%?** In the first 80% of training, the bias actively balances expert utilization. After 80%, freezing the bias allows the router to settle into its final routing pattern without interference. DeepSeek V3 froze at 97% (14.3T/14.8T tokens), but NanoSeek uses 80% as a safety margin — with fewer total tokens, allowing more time for post-freeze adaptation is prudent.

**Crucially, this update runs outside the gradient computation.** The bias is modified with `torch.no_grad()` and `.sub_()` — it's a control loop, not a learned parameter. This is the "auxiliary-loss-free" innovation: no gradient flows through the load balancing mechanism, so it cannot interfere with the language modeling objective.

#### 2.7.2 MTP Loss Weight Schedule

```python
def get_mtp_loss_weight(self, tokens_processed=None):
    transition = int(self.config.total_tokens * self.config.mtp.mtp_loss_transition_ratio)
    # mtp_loss_transition_ratio = 0.60
    if tokens_processed < transition:
        return self.config.mtp.mtp_loss_weight_initial  # 0.3
    return self.config.mtp.mtp_loss_weight_final          # 0.1
```

**Why decrease from 0.3 to 0.1?** Early in training, the MTP auxiliary objective provides useful gradient signal that helps the model learn robust representations (predicting the next-next token requires understanding deeper structure). As training matures, the primary next-token prediction objective should dominate — reducing MTP weight prevents the auxiliary objective from distorting the final model's next-token prediction quality.

**Why step function, not gradual decay?** DeepSeek V3's paper shows that a discrete transition works as well as a smooth one and is simpler to implement and reason about. The 60% transition point was empirically chosen.

### 2.8 Multi-Phase Training Pipeline

NanoSeek implements DeepSeek's two-phase training methodology, enabled with `--enable_multiphase=true`.

#### Phase 1: Dense MLA Pre-training (80% of tokens, ~4K context)

```
Configuration:
  sequence_length = 4096
  DSA: disabled (dense full attention)
  YaRN: disabled (native positions)
  Learning rate: full (matrix_lr = 0.02)
  Indexer: trains via auxiliary KL-divergence loss
```

During Phase 1, the Lightning Indexer component of DSA trains passively. It observes actual attention patterns (computed by dense attention) and learns to predict which tokens receive high attention scores. This training happens via an auxiliary loss weighted by `indexer_loss_weight=0.01`:

```python
if config.enable_multiphase and current_phase == 0:
    if 'indexer_loss' in outputs:
        indexer_aux = outputs['indexer_loss'] * config.indexer_loss_weight
        loss = loss + indexer_aux
```

#### Phase 2: Sparse DSA Fine-tuning (20% of tokens, ~8K context)

```
Configuration:
  sequence_length = 8192 (2x Phase 1)
  DSA: enabled (indexer selects top-k tokens)
  YaRN: enabled (interpolate RoPE to 8K positions)
  Learning rate: 0.33x Phase 1
  Batch size: halved (2x context → 2x memory → 0.5x batch)
```

The phase transition performs several critical operations:

1. **Checkpoint Phase 1**: Save a `phase1_final` checkpoint for analysis/rollback.
2. **Apply Phase 2 config**: Enable DSA and YaRN in the model config.
3. **Update RoPE frequencies**: Recompute rotary embedding frequencies for extended context.
4. **Rebuild dataloader**: New sequence length requires new batching.
5. **Update gradient accumulation**: `grad_accum_steps` changes because batch size changes.
6. **Reduce LR**: Multiply all `initial_lr` values by 0.33.

```python
rope_scaling_factor = phase2.sequence_length / phase1.sequence_length  # 8192/4096 = 2.0
orig_model.update_rope_for_context_extension(
    new_max_position_embeddings=phase2.sequence_length,
    rope_scaling_factor=rope_scaling_factor,
    original_max_position_embeddings=phase1.sequence_length,
)
```

**Why multi-phase?** Training at 4K context for 80% of tokens maximizes gradient updates per compute dollar (shorter sequences = more steps per token = more optimizer updates). The 20% Phase 2 at 8K context teaches the model to use the extended context efficiently, while the pre-trained indexer ensures DSA selects the right tokens.

### 2.9 Distributed Training: DDP with ZeRO-2 Optimizers

NanoSeek uses DDP (DistributedDataParallel), not FSDP. The model fits on a single H100 80GB.

**Why DDP over FSDP?**

```
NanoSeek-1B memory per GPU with DDP:
  Model weights (bf16):    ~9.5 GB
  Gradients (bf16):        ~9.5 GB
  Optimizer states (fp32): ~57 GB → ~7.1 GB with ZeRO-2 (÷8 ranks)
  Activations:             ~15 GB
  Total:                   ~41 GB (fits comfortably on H100 80GB)
```

FSDP would add unnecessary communication overhead: each forward/backward pass would require all-gather/reduce-scatter for every layer's parameters. DDP only communicates gradients once per step.

The key insight: the ZeRO-2 optimization is handled by the custom optimizers (DistMuon, DistAdamW), not by the DDP wrapper. DDP handles gradient averaging via AllReduce; the optimizers handle state sharding via reduce-scatter/all-gather during the optimizer step.

```python
# DDP wrapping (after compile, before training)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
```

### 2.10 Mixed Precision Strategy

```python
# Enable TF32 for faster matmuls on Ampere+
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Autocast context for forward/backward
autocast_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
```

**BF16, not FP16.** BFloat16 has the same exponent range as FP32 (8 bits) with reduced mantissa precision (7 bits vs 23). This means it can represent the same range of values — no loss scaling needed. FP16's limited range (5-bit exponent) requires loss scaling to prevent underflow/overflow, adding complexity and potential instability.

**TF32 for matmuls.** TensorFloat-32 uses 19-bit mantissa for internal accumulation in matrix multiplications. It's essentially "free" performance — same API as FP32, 8× faster on H100, with negligible accuracy impact for training.

**Where precision matters:**

| Operation | Precision | Why |
|---|---|---|
| Forward pass matmuls | BF16 (via autocast) | Speed |
| Backward pass matmuls | BF16 (via autocast) | Speed |
| Loss computation | FP32 (cross_entropy internal) | Numerical accuracy |
| Optimizer states (Adam) | FP32 | Accumulation precision |
| Newton-Schulz iteration | BF16 (explicit cast) | GPU-friendly, stable |
| Gradient norm computation | FP32 (clip_grad_norm default) | Overflow prevention |
| Muon momentum buffer | Same as param (BF16) | Memory efficiency |

### 2.11 Model Initialization and Compilation

```python
# Meta-device initialization (zero memory until materialized)
with torch.device("meta"):
    model = NanoSeekModel(model_config)

# Materialize on target device
model.to_empty(device=device)

# Initialize weights (Gaussian, std=0.02)
model.apply(model._init_weights)

# Compile (CUDA only)
model = torch.compile(model, mode='reduce-overhead', fullgraph=False, dynamic=True)
```

**Meta device initialization.** `torch.device("meta")` creates parameter tensors with shape and dtype but no actual storage. This allows constructing the full model graph (4.75B parameters) without allocating ~9.5GB of memory. `to_empty(device)` then allocates actual storage on the target device.

**torch.compile settings:**

- `mode='reduce-overhead'`: Aggressive optimization including CUDA graph capture. Trades longer compilation for faster steady-state execution.
- `fullgraph=False`: Required because MoE routing is dynamic (different experts active for different tokens), which breaks full graph capture.
- `dynamic=True`: Allows variable batch sizes, important for the last micro-batch in an epoch or during evaluation.

---

## 🔴 STAGE 3: GROUND TRUTH — What the Model Must Learn

### 3.1 The Autoregressive Objective

The ground truth is the next token. Given input tokens `[t₀, t₁, ..., t_{T-1}]`, the model must predict `[t₁, t₂, ..., t_T]`. This is implemented as:

```python
shift_logits = logits[:, :-1, :]     # [B, T-1, V] - predictions for positions 0 to T-2
shift_labels = labels[:, 1:]          # [B, T-1]    - ground truth at positions 1 to T-1
```

The shift ensures that the prediction at position `i` is compared against the token at position `i+1`. The model never sees the token it's trying to predict (causal masking enforces this in attention).

### 3.2 MTP Ground Truth

The MTP module predicts token `t_{i+2}` given the main model's hidden state at position `i` and the embedding of token `t_{i+1}`. Its ground truth is the same token sequence, but shifted by an additional position:

```
Main model:  predict t₁ from t₀,  predict t₂ from [t₀,t₁],  ...
MTP module:  predict t₂ from t₀,  predict t₃ from [t₀,t₁],  ...
```

### 3.3 Data Quality as Ground Truth

NanoSeek trains on FineWeb-Edu, which is filtered web text optimized for educational content. The quality of the training data *is* the ground truth — the model learns to predict what comes next in high-quality web text. This is why data curation matters enormously: garbage in, garbage out, regardless of how sophisticated your optimizer is.

**Chinchilla optimal ratio.** For 1.08B active parameters, the optimal training budget is approximately 22B tokens (20× active params). This is the compute-optimal frontier where additional tokens provide diminishing returns relative to the cost of processing them. Training beyond this point is not harmful but becomes increasingly inefficient.

---

## 🔴 STAGE 4: LOSS FUNCTION — Composite Loss with All Components

### 4.1 The Complete Loss Computation

From `NanoSeekModel._compute_loss`:

```python
def _compute_loss(self, logits, hidden_states, labels, aux_loss, indexer_loss):
    # 1. Main autoregressive loss
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    main_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1), ignore_index=-100,
    )

    total_loss = main_loss

    # 2. MTP loss (multi-token prediction)
    if self.mtp is not None:
        mtp_outputs = self.mtp(hidden_states, labels=labels)
        mtp_loss = mtp_outputs.get("mtp_loss", torch.tensor(0.0))
        mtp_weight = self.get_mtp_loss_weight()
        total_loss = total_loss + mtp_weight * mtp_loss

    # 3. Sequence-level auxiliary loss (MoE load balancing)
    if aux_loss is not None and aux_loss.item() > 0:
        total_loss = total_loss + aux_loss

    # 4. Indexer loss (DSA training)
    if indexer_loss is not None and indexer_loss.item() > 0:
        total_loss = total_loss + indexer_loss

    return {"loss": total_loss, "main_loss": main_loss, "mtp_loss": mtp_loss, ...}
```

### 4.2 Loss Component Breakdown

```
total_loss = main_loss + λ × mtp_loss + aux_loss + indexer_loss
```

**Component 1: Main Loss (cross-entropy)**

```
main_loss = -Σᵢ log P(tᵢ₊₁ | t₁...tᵢ)
```

Standard autoregressive cross-entropy over vocabulary of 65,536 tokens. This is the primary training signal.

**Component 2: MTP Loss (weighted cross-entropy)**

```
mtp_loss = -Σᵢ log P_mtp(tᵢ₊₂ | t₁...tᵢ)
λ = 0.3  (steps 0 to 60% of total)
λ = 0.1  (steps 60% to 100% of total)
```

The MTP module uses its own cross-entropy loss, weighted by the dynamic `λ`. At the transition point (60% of training), the weight drops discretely from 0.3 to 0.1.

**Component 3: Sequence-Level Auxiliary Loss (MoE)**

```python
load = self.gate.expert_load                          # [64]
target_load = N * n_activated / n_routed              # Expected tokens per expert
load_imbalance = ((load - target_load) ** 2).mean()   # MSE
aux_loss = seq_aux_loss_alpha * load_imbalance        # α = 0.0001
```

This is an extremely small loss (`α=0.0001`) that provides gentle gradient-based pressure toward balanced routing. It complements the bias-based load balancing (which operates outside the gradient).

**Component 4: Indexer Loss (DSA)**

```
indexer_loss = indexer_loss_weight × KL(indexer_scores || attention_scores)
```

Only active during Phase 1 of multi-phase training (`indexer_loss_weight=0.01`). Trains the Lightning Indexer to predict which tokens will receive high attention scores, preparing it for Phase 2 when DSA activates.

### 4.3 Gradient Accumulation and Loss Scaling

```python
for micro_step in range(grad_accum_steps):
    with autocast_ctx:
        outputs = model(input_ids=x, labels=y)
        loss = outputs['loss']

    loss = loss / grad_accum_steps  # Scale for accumulation
    loss.backward()
```

**The division by `grad_accum_steps` is critical.** Without it, accumulated gradients would be `grad_accum_steps` times larger than expected, effectively multiplying the learning rate by the accumulation factor. This is the single most common gradient accumulation bug.

### 4.4 Common Misconceptions About Composite Loss

**Misconception: "The auxiliary loss hurts main task performance."**
At `α=0.0001`, the auxiliary loss gradient is approximately 10,000× smaller than the main loss gradient. It provides a negligible perturbation to the optimization trajectory. The real load balancing happens through the bias mechanism outside the gradient.

**Misconception: "MTP loss should be constant throughout training."**
DeepSeek V3 demonstrated that reducing MTP weight later in training improves final next-token prediction quality. The MTP objective is an *auxiliary* training signal, not the end goal. Keeping it high throughout training would over-optimize for multi-token prediction at the expense of the primary objective.

**Misconception: "Cross-entropy reduction means the model is learning."**
A 1% reduction in cross-entropy from 3.00 to 2.97 is meaningful (represents learning real language patterns). A 1% reduction from 0.50 to 0.495 late in training might just be memorization. Always monitor validation loss alongside training loss.

---

## 🔴 STAGE 5: OUTPUTS — Trained Model Checkpoints

### 5.1 Checkpoint Format (nanochat-style)

```
checkpoint_dir/
  model_{step:06d}.pt           # Full model state dict (rank 0 only)
  optim_{step:06d}_rank{r}.pt   # Per-rank optimizer state
  meta_{step:06d}.json          # Training metadata
```

**Model file** (`model_000042.pt`): Contains the complete model state dict including all 4.75B parameters. Saved only on rank 0 since DDP ensures all ranks have identical model parameters. Size: ~9.5GB (bf16).

**Optimizer files** (`optim_000042_rank0.pt` through `optim_000042_rank7.pt`): Per-rank because DistMuon and DistAdamW shard optimizer states. Each rank saves only its owned slice. For Muon, this includes momentum buffers; for AdamW, first and second moment estimates plus step counters.

**Metadata file** (`meta_000042.json`): JSON containing:

```json
{
  "step": 42000,
  "model_config": { ... },
  "training_config": { ... },
  "loop_state": {
    "step": 42000,
    "min_val_loss": 2.847,
    "smooth_train_loss": 2.891,
    "total_training_time": 49820.5,
    "tokens_processed": 22020096000,
    "current_phase": 1,
    "phase_step": 8400
  },
  "dataloader_state": {
    "pq_idx": 147,
    "rg_idx": 2384
  }
}
```

### 5.2 Checkpoint Loading and Resume

```python
# Handle torch.compile prefix in state dict keys
model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
model.load_state_dict(model_data)
```

The `_orig_mod.` prefix stripping is essential: `torch.compile` wraps the model in an `_orig_mod` attribute, so saved state dict keys have this prefix. Loading into an uncompiled model requires removing it.

**Resume sequence:**

1. Load model state dict (rank 0 saves, all ranks load).
2. Load per-rank optimizer states (each rank loads its own file).
3. Restore loop state (step, losses, timing).
4. Restore multi-phase state (current_phase, phase_step).
5. Rebuild dataloader with saved `(pq_idx, rg_idx)` state.
6. Continue training from saved step.

### 5.3 FLOP Estimation and MFU Tracking

```python
# FLOPs per token (standard approximation)
num_flops_per_token = 6 * active_params  # 6 × 1.08B ≈ 6.48 GFLOPs/token

# MFU against H100 SXM BF16 peak
promised_flops = 989e12 * ddp_world_size  # 989 TFLOPS × 8 GPUs = 7.912 PFLOPS
flops_per_sec = num_flops_per_token * total_batch_size / dt
mfu = 100 * flops_per_sec / promised_flops
```

**The 6× factor:** 2× for forward pass (each parameter participates in one multiply-accumulate = 2 FLOPs) + 4× for backward pass (gradient w.r.t. activations + gradient w.r.t. weights, each requiring the same 2× as forward). This approximation ignores attention complexity (O(L²) not captured by parameter count), activation recomputation overhead, and communication costs.

**Typical MFU for NanoSeek:** 30-45% on H100 SXM. Below the 50%+ achieved by dense models because MoE routing overhead, expert load imbalance, and the Newton-Schulz iterations in Muon all consume FLOPS that don't directly contribute to the matmuls counted in the numerator.

---

## Full Walkthrough: One Complete Training Step

Let's trace step 21,000 (exactly 50% through training) in detail.

### Step 21,000 State

```
tokens_processed = 21,000 × 524,288 = 11,010,048,000 (~11B tokens)
MTP weight λ = 0.3 (before 60% transition at 12.6B tokens)
gamma = 0.001 (before 80% freeze at 17.6B tokens)
LR multiplier = 1.0 (in constant phase, before 80% warmdown start)
Muon momentum = 0.95 (warmed up long ago at step 300)
Phase: 1 (Dense MLA) in multi-phase mode
```

### Micro-step 1 of 2

1. **Load micro-batch**: `(x, y)` of shape `[8, 4096]` from streaming dataloader.
2. **Forward pass** under `autocast(bf16)`:
   - Embed tokens: `[8, 4096] → [8, 4096, 2048]`
   - 16 transformer layers (2 dense FFN, 14 MoE):
     - Each layer: RMSNorm → MLA → Residual → RMSNorm → FFN/MoE → Residual
     - MoE layers: Route each token to 8 of 64 experts + 2 shared experts
     - Collect aux_loss from each MoE layer (α=0.0001 × load_imbalance²)
     - Collect indexer_loss from sparse attention layers (if multi-phase enabled)
   - Final RMSNorm → lm_head: `[8, 4096, 2048] → [8, 4096, 65536]`
3. **Compute composite loss**:
   - `main_loss`: cross-entropy on shifted logits/labels
   - `mtp_loss`: MTP module forward + cross-entropy
   - `total_loss = main_loss + 0.3 × mtp_loss + aux_loss + indexer_loss`
4. **Scale and backward**: `(total_loss / 2).backward()` (÷2 for 2 grad accum steps)
5. **Prefetch next micro-batch** (overlapped with backward via async data transfer)

### Micro-step 2 of 2

Same as micro-step 1, gradients accumulate into the same `.grad` tensors.

### Optimizer Update

6. **Gradient clipping**: `clip_grad_norm_(parameters, max_norm=1.0)` — computes global L2 norm across all parameters, scales down if exceeding 1.0.

7. **LR update**: Compute `lrm = get_lr_multiplier(21000, 42000, config) = 1.0` (constant phase). Apply to all optimizer groups.

8. **Muon momentum update**: `get_muon_momentum(21000) = 0.95` (converged long ago).

9. **Muon step** (for all 2D matrix parameters):
   - DistMuon: `reduce_scatter(AVG)` to average gradients
   - Owner rank: `buf.lerp_(g, 0.05)` → Nesterov → `zeropower_via_newtonschulz5(g, 5)` → `p.add_(g, -0.02 * sqrt_ratio)`
   - `all_gather` to broadcast updated parameters

10. **AdamW step** (for embeddings + non-2D):
    - DistAdamW: `reduce_scatter_tensor(AVG)` on gradient
    - Owner slice: Adam update with `β₁=0.9, β₂=0.95`
    - `all_gather_into_tensor` to reconstruct full parameter

11. **Zero gradients**: `model.zero_grad(set_to_none=True)` (set_to_none saves memory vs zeroing)

### Post-Update Schedules

12. **Load balance bias update**: For each MoE layer: `bias[i] -= 0.001 × (load[i] - mean_load) / mean_load`

13. **Token counter update**: `tokens_processed += 524,288`

14. **DSA step counter increment** (if multi-phase enabled)

### Logging (every 10 steps)

15. **Compute metrics**:
    - `tok/s = 524,288 / dt`
    - `mfu = (6 × 1.08B × 524,288 / dt) / (989e12 × 8) × 100`
    - EMA smoothed loss: `smooth = 0.9 × smooth + 0.1 × loss`

---

## Common Misconceptions

### "Muon is just Adam with extra steps"

No. Muon and Adam are fundamentally different. Adam adapts per-parameter learning rates based on gradient history (second moment). Muon replaces the update direction entirely — the Newton-Schulz orthogonalization can rotate the update by up to 90° relative to the gradient. They operate on different geometric principles: Adam follows a scaled gradient; Muon follows the nearest orthogonal matrix to the momentum-accumulated gradient.

### "DDP AllReduce duplicates the ZeRO-2 reduce-scatter"

DDP's AllReduce averages gradients across ranks. DistMuon's reduce-scatter *also* averages gradients but delivers only a shard to each rank. These are the same communication (averaging) but with different output distributions. When using DistMuon/DistAdamW, DDP's gradient AllReduce is redundant for those parameters. In practice, DDP still handles synchronization and the distributed backward hook; the custom optimizers overlay their own communication pattern on top.

### "You need FSDP for 4.75B total parameters"

No. Total parameter count is misleading for MoE models. The memory bottleneck is optimizer states (12 bytes per parameter for Adam in fp32). With ZeRO-2 across 8 GPUs, each rank stores optimizer states for ~594M parameters (4.75B / 8), which is ~7.1GB — well within H100 80GB capacity even after accounting for model weights, gradients, and activations.

### "Gradient accumulation is just a loop"

Almost, but the loss scaling (`loss / grad_accum_steps`) is non-trivial. Without it, the effective learning rate scales with the accumulation factor. Also, gradient clipping must happen *after* all micro-steps complete — clipping per-micro-step would clip the partial gradient, which has a different norm than the full accumulated gradient.

### "The MTP module doubles the compute cost"

No. The MTP module is a lightweight prediction head that processes the *already-computed* hidden states. It adds ~10-15% compute overhead, not 100%. The main model's 16 transformer layers dominate compute; the MTP module is a single block with cross-attention and a small FFN.

---

## Production Gotchas

### Gotcha 1: Gradient Accumulation + DDP Sync

In standard DDP, gradient AllReduce happens during `.backward()` automatically. With gradient accumulation, you want AllReduce only on the *last* micro-step, not every micro-step. PyTorch provides `model.no_sync()` context manager for this. NanoSeek's current implementation does AllReduce every micro-step — this is correctness-preserving (averaged gradients accumulate correctly) but wastes communication bandwidth. For production training at scale, wrap non-final micro-steps in `model.no_sync()`.

### Gotcha 2: torch.compile + MoE Dynamic Routing

MoE routing is data-dependent: different tokens go to different experts. This means the computation graph changes every forward pass. `fullgraph=False` is required because the router creates dynamic control flow (selecting different experts). `dynamic=True` allows variable tensor sizes from the router. Without these flags, compilation will fail or produce incorrect results.

### Gotcha 3: Checkpoint _orig_mod Prefix

`torch.compile` wraps the model, prepending `_orig_mod.` to all state dict keys. If you save a compiled model's state dict and load it into an uncompiled model (or vice versa), key names won't match. The `.removeprefix("_orig_mod.")` call in `load_checkpoint` handles this, but it's easy to forget when building custom loading code.

### Gotcha 4: Per-Rank Optimizer Files in DistMuon/DistAdamW

Each rank saves its own optimizer state file because the ZeRO-2 sharding means each rank holds different state. Resuming on a different number of GPUs requires re-sharding the optimizer states — which NanoSeek does not currently support. If you trained on 8 GPUs, you must resume on 8 GPUs.

### Gotcha 5: Approximate Dataloader Resume

The dataloader resumes from `(pq_idx, rg_idx)` by advancing one full `ddp_world_size` block past the saved position. This means after resume, the first few row groups may be skipped. For 22B token training runs, this is negligible. For short debugging runs (<1000 steps), be aware that resumed runs see slightly different data than continuous runs.

### Gotcha 6: EMA Loss Debiasing

The training log shows EMA-smoothed loss with debiasing:

```python
smooth_train_loss = 0.9 * smooth_train_loss + (1 - 0.9) * train_loss.item()
debiased = smooth_train_loss / (1 - 0.9 ** (step + 1))
```

The debiasing correction is critical for early steps. At step 0, without debiasing, the EMA would be 0.1× the true loss (since it's initialized to 0). The denominator `(1 - 0.9^1) = 0.1` corrects this. By step 50, the correction is negligible (`1 - 0.9^50 ≈ 0.995`).

### Gotcha 7: The set_to_none=True Memory Optimization

```python
model.zero_grad(set_to_none=True)
```

`set_to_none=True` deallocates gradient tensors instead of zeroing them. This saves memory (no need to maintain zero tensors) and can be faster (no memset kernel). However, it means `param.grad is None` rather than `param.grad == 0` — code that checks `.grad is not None` to determine if a parameter received gradients will behave differently.

---

## 2026 Best Practices Reflected in NanoSeek

### Practice 1: Decouple Optimizer by Parameter Role

The industry has converged on the insight that different parameter types benefit from different optimization strategies. NanoSeek's Muon-for-matrices + AdamW-for-embeddings pattern reflects this. In 2026, hybrid optimizers are standard for frontier models — the debate is about *which* combination, not *whether* to combine.

### Practice 2: Extended Constant Phase in LR Schedule

The DeepSeek-style 70% constant phase is now widely adopted. Pure cosine schedules waste the middle 50% of training on a declining LR that's still too high for fine detail but too low for exploration. The constant phase lets the model explore at maximum learning rate for longer, then the cosine decay handles fine convergence.

### Practice 3: Bias-Based Load Balancing

Auxiliary-loss-free balancing via bias adjustment is the 2026 consensus for MoE training. The approach is simpler, more stable, and demonstrably better than auxiliary losses. The only tunable is gamma (update rate), and the freeze schedule provides robustness — even a poorly-tuned gamma becomes harmless once frozen.

### Practice 4: Ratio-Based Schedule Parameters

NanoSeek uses ratios (`gamma_freeze_ratio=0.80`, `mtp_loss_transition_ratio=0.60`, `warmdown_ratio=0.20`) instead of absolute step counts. This makes configurations portable across different training budgets — double your tokens, and all schedules automatically scale.

### Practice 5: Multi-Phase Context Extension

The "train short, extend long" paradigm is standard for cost-effective long-context models. Training at 4K context for 80% of compute maximizes gradient updates per dollar. Phase 2 at 2× context teaches context utilization. YaRN enables further extension at inference with no additional training.

### Practice 6: Compute-Optimal Data Ratio

Chinchilla scaling (20× active params) remains the reference point. NanoSeek targets exactly this: 22B tokens for 1.08B active parameters. Going significantly beyond this (e.g., Llama 3's 200× ratio) requires careful data curation to avoid quality degradation from over-exposure to web text.

### Practice 7: MFU as the North Star Metric

MFU (Model FLOPs Utilization) against H100 peak BF16 throughput (989 TFLOPS) is the standard measure of training efficiency. Every percent of MFU improvement directly translates to wall-clock speedup and cost savings. NanoSeek tracks MFU every log step — if it drops below 30%, something is wrong (memory pressure causing swapping, communication bottleneck, or load imbalance).

---

## Appendix: Key Hyperparameter Reference

| Parameter | Value | Source |
|---|---|---|
| `matrix_lr` (Muon) | 0.02 | `TrainingConfig.matrix_lr` |
| `embedding_lr` (AdamW) | 0.2 | `TrainingConfig.embedding_lr` |
| `unembedding_lr` (AdamW) | 0.004 | `TrainingConfig.unembedding_lr` |
| `adam_beta1` | 0.9 | `TrainingConfig.adam_beta1` |
| `adam_beta2` | 0.95 | `TrainingConfig.adam_beta2` |
| `muon_momentum` | 0.85 → 0.95 (300 steps) | `get_muon_momentum()` |
| `muon_ns_steps` | 5 | `Muon.__init__` default |
| `ns_coefficients` | a=3.4445, b=-4.7750, c=2.0315 | `zeropower_via_newtonschulz5` |
| `warmup_ratio` | 0.0 (pre-train) | `TrainingConfig.warmup_ratio` |
| `warmdown_ratio` | 0.2 | `TrainingConfig.warmdown_ratio` |
| `constant_phase_ratio` | 0.70 | `NanoSeekConfig.constant_phase_ratio` |
| `cosine_decay_end_ratio` | 0.95 | `NanoSeekConfig.cosine_decay_end_ratio` |
| `grad_clip` | 1.0 | `TrainingConfig.grad_clip` |
| `gamma` | 0.001 | `MoEConfig.gamma` |
| `gamma_freeze_ratio` | 0.80 | `MoEConfig.gamma_freeze_ratio` |
| `seq_aux_loss_alpha` | 0.0001 | `MoEConfig.seq_aux_loss_alpha` |
| `mtp_loss_weight_initial` | 0.3 | `MTPConfig.mtp_loss_weight_initial` |
| `mtp_loss_weight_final` | 0.1 | `MTPConfig.mtp_loss_weight_final` |
| `mtp_loss_transition_ratio` | 0.60 | `MTPConfig.mtp_loss_transition_ratio` |
| `indexer_loss_weight` | 0.01 | `SparseAttentionConfig.indexer_loss_weight` |
| `total_batch_size` | 524,288 (512K tokens) | `TrainingConfig.total_batch_size` |
| `device_batch_size` | 8 | `TrainingConfig.device_batch_size` |
| `max_seq_len` | 4096 (Phase 1), 8192 (Phase 2) | `TrainingConfig.max_seq_len` |
| `vocab_size` | 65,536 | `NanoSeekConfig.vocab_size` |
| `active_params` | ~1.08B | `NanoSeekConfig.estimated_active_params` |
| `total_params` | ~4.75B | `NanoSeekConfig.estimated_total_params` |
| `total_tokens` | 22B (Chinchilla 20×) | `NanoSeekConfig.total_tokens` |
| `H100 BF16 peak` | 989 TFLOPS | `pre-train.py` line 1573 |
