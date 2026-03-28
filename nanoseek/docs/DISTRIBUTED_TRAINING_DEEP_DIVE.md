# NanoSeek Distributed Training: A First-Principles Deep Dive

**Author perspective**: Written as a lead senior AI researcher/engineer at a frontier lab would explain to a teammate ramping up on the codebase. Every line of code is dissected from first principles.

**Tác giả**: Viết theo góc nhìn của lead senior AI researcher tại lab AI hàng đầu, giải thích cho đồng nghiệp mới. Mỗi dòng code được phân tích từ nguyên lý gốc.

---

## Table of Contents

1. [The Big Picture: Why NOT DDP?](#1-the-big-picture)
2. [Chapter 1: Process Initialization](#2-process-initialization)
3. [Chapter 2: Model Construction — No Wrapping](#3-model-construction)
4. [Chapter 3: The Training Loop — Forward & Backward](#4-the-training-loop)
5. [Chapter 4: DistMuonAdamW — The Heart of Distribution](#5-distmuonadamw)
6. [Chapter 5: Phase 1 — Launch Async Reduces](#6-phase-1)
7. [Chapter 6: Phase 2 — Compute on Shards](#7-phase-2)
8. [Chapter 7: Phase 3 — Gather & Reassemble](#8-phase-3)
9. [Chapter 8: Distributed Gradient Clipping](#9-distributed-grad-clip)
10. [The Complete Data Flow Diagram](#10-complete-flow)

---

## 1. The Big Picture: Why NOT DDP?

### The Standard Approach (PyTorch DDP)

Most distributed training uses `DistributedDataParallel` (DDP):

```
┌──────────────────────────────────────────────────────────┐
│  Standard DDP Pipeline                                    │
│                                                           │
│  forward() → loss.backward()                              │
│       ↓                                                   │
│  DDP hooks fire: all_reduce(gradients, AVG)    ← HERE     │
│       ↓                                                   │
│  optimizer.step() ← sees averaged gradients               │
│       ↓                                                   │
│  model.zero_grad()                                        │
└──────────────────────────────────────────────────────────┘
```

DDP wraps your model in `DistributedDataParallel`, which:
1. Registers backward hooks on every parameter
2. During `loss.backward()`, hooks fire to all-reduce gradients across ranks
3. By the time `.backward()` returns, all ranks have the same averaged gradients
4. The optimizer sees identical gradients → produces identical parameters

**Vấn đề với DDP cho Muon**: DDP all-reduce gradients ngay trong backward. Nhưng Muon optimizer cần xử lý gradient theo cách đặc biệt — nó stack nhiều parameter cùng shape lại, rồi chạy orthogonalization trên cả stack. DDP không biết về cấu trúc này, nên nó sẽ all-reduce từng gradient riêng lẻ → lãng phí bandwidth.

### NanoSeek's Approach: Optimizer-Internal Communication

```
┌──────────────────────────────────────────────────────────┐
│  NanoSeek Pipeline (NO DDP)                               │
│                                                           │
│  forward() → loss.backward()                              │
│       ↓                                                   │
│  gradients are LOCAL to each rank    ← no sync here!      │
│       ↓                                                   │
│  optimizer.step()                    ← sync happens HERE   │
│    Phase 1: reduce_scatter(grads)                         │
│    Phase 2: compute update on shard                       │
│    Phase 3: all_gather(updated params)                    │
│       ↓                                                   │
│  model.zero_grad()                                        │
└──────────────────────────────────────────────────────────┘
```

**Why this is better for MoE + Muon:**

1. **Muon batching**: Muon groups all same-shape params (e.g., 192 expert weight matrices of shape `[480, 1280]`) into one stacked tensor `[192, 480, 1280]`. One `reduce_scatter` on this stack is far cheaper than 192 individual `all_reduce` calls (communication startup cost dominates for small tensors).

2. **ZeRO-2 sharding**: Each rank only computes the optimizer step on 1/N of the parameters, so optimizer state memory is divided by N. For AdamW, this saves 2× the parameter memory in `exp_avg` and `exp_avg_sq`.

3. **Async pipeline**: Communication and computation overlap — while rank 0 is computing its Muon update, the `all_gather` from the previous group is still running on the NCCL stream.

**Tại sao thiết kế này tốt hơn DDP?** Vì Muon optimizer cần nhóm các gradient có cùng shape lại thành 1 tensor lớn (ví dụ: 192 expert weights `[480, 1280]` → `[192, 480, 1280]`). Một lần `reduce_scatter` trên tensor lớn nhanh hơn nhiều so với 192 lần `all_reduce` riêng lẻ. Ngoài ra, mỗi GPU chỉ cần tính optimizer step cho 1/N tham số → tiết kiệm bộ nhớ optimizer state.

---

## 2. Process Initialization

**File**: `nanoseek/nanoseek/common.py`

### How `torchrun` Sets Up the Environment

When you launch with:
```bash
torchrun --nproc_per_node=8 -m nanoseek.scripts.pre_train
```

`torchrun` spawns 8 processes and sets these environment variables in each:

```
Process 0: RANK=0, LOCAL_RANK=0, WORLD_SIZE=8
Process 1: RANK=1, LOCAL_RANK=1, WORLD_SIZE=8
...
Process 7: RANK=7, LOCAL_RANK=7, WORLD_SIZE=8
```

**`torchrun` tự động spawn 8 process và set biến môi trường cho từng process.** `RANK` = ID duy nhất toàn cục. `LOCAL_RANK` = ID trên node hiện tại (quan trọng khi chạy multi-node). `WORLD_SIZE` = tổng số process.

### Detection: `is_ddp_requested()` and `get_dist_info()`

```python
# common.py lines 136-160

def is_ddp_requested() -> bool:
    """
    True if launched by torchrun (env present), even before init.
    Used to decide whether we *should* initialize a PG.
    """
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
```

**First principles**: This function does NOT check if distributed training is *running* — it checks if it was *requested*. The distinction matters: we haven't called `init_process_group()` yet. We're just reading environment variables that `torchrun` set before our Python process even started.

**Nguyên lý**: Hàm này KHÔNG kiểm tra distributed training có đang chạy không — nó kiểm tra có được YÊU CẦU không. Sự khác biệt quan trọng: chúng ta chưa gọi `init_process_group()`. Chúng ta chỉ đọc biến môi trường mà `torchrun` đã set trước khi Python process bắt đầu.

```python
def get_dist_info():
    if is_ddp_requested():
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1
```

**Data flow example** (8 GPUs on 1 node):

```
┌─────────────────────────────────────────────────┐
│  Machine with 8 GPUs                             │
│                                                   │
│  GPU 0: RANK=0, LOCAL_RANK=0  ─┐                 │
│  GPU 1: RANK=1, LOCAL_RANK=1   │                 │
│  GPU 2: RANK=2, LOCAL_RANK=2   │  WORLD_SIZE=8   │
│  GPU 3: RANK=3, LOCAL_RANK=3   │                 │
│  GPU 4: RANK=4, LOCAL_RANK=4   │                 │
│  GPU 5: RANK=5, LOCAL_RANK=5   │                 │
│  GPU 6: RANK=6, LOCAL_RANK=6   │                 │
│  GPU 7: RANK=7, LOCAL_RANK=7  ─┘                 │
└─────────────────────────────────────────────────┘
```

For multi-node (2 nodes × 4 GPUs):

```
┌─────────────────────────┐  ┌─────────────────────────┐
│  Node 0                  │  │  Node 1                  │
│  GPU 0: R=0, LR=0       │  │  GPU 0: R=4, LR=0       │
│  GPU 1: R=1, LR=1       │  │  GPU 1: R=5, LR=1       │
│  GPU 2: R=2, LR=2       │  │  GPU 2: R=6, LR=2       │
│  GPU 3: R=3, LR=3       │  │  GPU 3: R=7, LR=3       │
└─────────────────────────┘  └─────────────────────────┘
WORLD_SIZE = 8 for all processes
```

`RANK` = global unique ID. `LOCAL_RANK` = which GPU on this specific machine. They differ only in multi-node.

### Initialization: `compute_init()`

```python
# common.py lines 186-215

def compute_init(device_type="cuda", seed: int = 42):
    """Basic initialization that we keep doing over and over, so make common."""

    assert device_type in ["cuda", "mps", "cpu"]
    if device_type == "cuda":
        assert torch.cuda.is_available()

    # Reproducibility — full seed control across all RNG sources
    set_reproducibility_seed(seed)
```

**Why seed first?** Seeds must be set BEFORE any random operations (model init, dropout, etc.). All ranks use the same seed so model initialization produces identical weights.

**Tại sao set seed trước?** Seed phải được set TRƯỚC bất kỳ thao tác random nào. Tất cả các rank dùng cùng seed → model init ra weight giống hệt nhau trên mọi GPU.

```python
    # Precision
    if device_type == "cuda":
        torch.set_float32_matmul_precision("high")  # TF32 for remaining FP32 ops
```

**First principles**: `"high"` enables TF32 (Tensor Float 32) for any matmul that stays in FP32. TF32 uses 10-bit mantissa instead of 23-bit, giving ~3× speedup on Ampere/Hopper GPUs for FP32 operations. Since our main compute is already in BF16 (via autocast), this mainly affects the loss computation, optimizer math, and gradient norms.

```python
    # Distributed setup
    is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if is_ddp_requested and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)   # ← (A)
        torch.cuda.set_device(device)                     # ← (B)
        dist.init_process_group(backend="nccl", device_id=device)  # ← (C)
        dist.barrier()                                    # ← (D)
    else:
        device = torch.device(device_type)
```

Let me break down each line:

**(A)** `device = torch.device("cuda", ddp_local_rank)`

Creates a device handle for this rank's specific GPU. On a machine with 8 GPUs:
- Rank 0 → `torch.device("cuda", 0)` → GPU 0
- Rank 3 → `torch.device("cuda", 3)` → GPU 3
- Rank 7 → `torch.device("cuda", 7)` → GPU 7

**Tại sao dùng `LOCAL_RANK` chứ không phải `RANK`?** Vì `RANK` là ID toàn cục (có thể >7 trên multi-node), nhưng mỗi machine chỉ có N GPU (thường 8). `LOCAL_RANK` đảm bảo rank 0 trên node 1 dùng GPU 0 của node 1, không phải GPU 4 (không tồn tại).

**(B)** `torch.cuda.set_device(device)`

Sets the default CUDA device for this process. After this call:
- `torch.zeros(5).cuda()` goes to the correct GPU automatically
- `torch.tensor(..., device="cuda")` goes to the correct GPU
- Without this, all processes would fight over GPU 0

**(C)** `dist.init_process_group(backend="nccl", device_id=device)`

This is where the magic happens. It:

1. **Creates a TCP/shared-memory rendezvous** between all processes (using env vars `MASTER_ADDR`, `MASTER_PORT` set by `torchrun`)
2. **Establishes NCCL communicators** — direct GPU-to-GPU communication channels using NVLink (intra-node) or InfiniBand (inter-node)
3. **Assigns this process its rank** in the communication group

```
                    NCCL Communicator
    ┌──────┐  NVLink  ┌──────┐  NVLink  ┌──────┐
    │ GPU 0 │◄───────►│ GPU 1 │◄───────►│ GPU 2 │ ...
    └──────┘          └──────┘          └──────┘
       ↕ PCIe            ↕ PCIe            ↕ PCIe
    ┌──────┐          ┌──────┐          ┌──────┐
    │ CPU 0 │          │ CPU 1 │          │ CPU 2 │
    └──────┘          └──────┘          └──────┘
```

**Why NCCL and not Gloo?** NCCL (NVIDIA Collective Communications Library) uses GPU-direct communication: data goes GPU→NVLink→GPU without touching the CPU or host memory. Gloo routes through CPU memory. For the tensor sizes in NanoSeek (model weights are ~4GB), NCCL is 10-50× faster.

**Tại sao NCCL mà không phải Gloo?** NCCL truyền data trực tiếp GPU→NVLink→GPU mà không qua CPU. Gloo phải đi qua CPU memory. Với tensor lớn như weight model (~4GB), NCCL nhanh hơn 10-50 lần.

**(D)** `dist.barrier()`

All 8 processes block here until the slowest one arrives. This ensures:
- All NCCL communicators are fully initialized
- No process starts model construction before communication is ready
- If any process fails to init, we hang here (visible) instead of crashing later (mysterious)

```
Time →
Rank 0: ─── init_pg() ──── WAIT ──── barrier() ──── proceed ───►
Rank 1: ─── init_pg() ───────────── barrier() ──── proceed ───►
Rank 2: ─── init_pg() ──────── barrier() ───── proceed ───────►
                                      ↑
                              All ranks proceed
                              only after ALL arrive
```

### Cleanup: `compute_cleanup()`

```python
def compute_cleanup():
    """Companion function to compute_init, to clean things up before script exit"""
    if is_ddp_initialized():
        dist.destroy_process_group()
```

**`is_ddp_initialized()`** checks `dist.is_available() and dist.is_initialized()` — unlike `is_ddp_requested()` which only checks env vars. `destroy_process_group()` tears down NCCL communicators and frees GPU-side resources.

---

## 3. Model Construction — No DDP Wrapping

**File**: `nanoseek/scripts/pre_train.py`, lines 446-452, 694-696, 1020, 1100-1104

### Build on Meta Device, Materialize, Initialize

```python
# pre_train.py lines 446-452

print0("Building model on meta device...")
with torch.device("meta"):            # ← (A)
    model = NanoSeekModel(config)

model.to_empty(device=device)          # ← (B)
model.init_weights()                   # ← (C)
```

**(A)** `with torch.device("meta"):`

**First principles**: The "meta" device is a virtual device that records tensor shapes and dtypes WITHOUT allocating memory. A 2048×2048 weight matrix on meta device takes 0 bytes. This lets us:
1. Build the full model graph (all 4.75B parameters for 1B scale)
2. Inspect architecture, count parameters, validate shapes
3. WITHOUT needing 20+ GB of RAM to hold the tensors

**`torch.device("meta")` là thiết bị ảo** — ghi lại shape và dtype nhưng KHÔNG cấp phát bộ nhớ. Matrix 2048×2048 trên meta device chiếm 0 byte. Cho phép xây dựng graph model đầy đủ (4.75B tham số) mà không cần 20GB RAM.

**(B)** `model.to_empty(device=device)`

Materializes the model: allocates real GPU memory for all tensors but fills them with garbage (uninitialized memory). This is faster than `model.to(device)` which would also copy values — but we're about to overwrite everything with init_weights() anyway.

```
Meta device:    [shape=(2048,2048), dtype=bf16, data=∅]
                         ↓ to_empty(device="cuda:3")
GPU 3:          [shape=(2048,2048), dtype=bf16, data=garbage]
                         ↓ init_weights()
GPU 3:          [shape=(2048,2048), dtype=bf16, data=properly initialized]
```

**(C)** `model.init_weights()`

Since all ranks use the same seed (set in `compute_init()`), and PyTorch's random number generators are deterministic given a seed, all 8 GPUs produce IDENTICAL initial weights. This is critical — if they started with different weights, the gradient averaging would be meaningless.

**Vì tất cả rank dùng cùng seed**, PyTorch RNG sinh ra weight khởi tạo giống hệt nhau trên mọi GPU. Điều này quan trọng — nếu weight ban đầu khác nhau, việc average gradient sẽ vô nghĩa.

### Optimizer Selection: DDP or Single-GPU

```python
# pre_train.py lines 694-696

Factory = DistMuonAdamW if ddp else MuonAdamW
optimizer = Factory(param_groups)
```

This is the single point where the training script branches:
- **Single GPU**: `MuonAdamW` — no communication, straightforward optimizer step
- **Multi GPU**: `DistMuonAdamW` — all communication logic embedded in the optimizer

### Keeping a Reference to the Uncompiled Model

```python
# pre_train.py line 1020

orig_model = model
```

**Why?** When we call `torch.compile(model)` below, the returned object is a `_TorchCompileWrapper` — it works for forward/backward but NOT for:
- `state_dict()` / `load_state_dict()` (checkpointing)
- `named_parameters()` (optimizer setup, gradient clipping)
- `update_load_balance_bias()` (MoE-specific)

`orig_model` keeps a pointer to the raw `NanoSeekModel` for these operations.

**Tại sao giữ `orig_model`?** `torch.compile()` trả về wrapper object — chỉ dùng được cho forward/backward. Để lưu checkpoint, setup optimizer, clip gradient... chúng ta cần model gốc chưa compile.

### torch.compile — NOT DDP

```python
# pre_train.py lines 1100-1104

if args.no_compile:
    print0("torch.compile SKIPPED")
else:
    model = torch.compile(model, dynamic=False)
```

**Critical design choice**: We compile the raw model, NOT a DDP-wrapped model. In standard PyTorch distributed training, you'd do `model = DDP(model)` then optionally compile. Here, there is no DDP wrapper at all. The model is just:

```
raw NanoSeekModel → torch.compile → compiled model
```

`dynamic=False` tells the compiler that input shapes won't change. This enables aggressive kernel fusion (30-50% speedup) but means changing batch size or sequence length requires recompilation.

```
┌─────────────────────────────────────────────────────┐
│  Standard DDP stack:                                 │
│  NanoSeekModel → DDP(model) → compile(DDP(model))   │
│                                                      │
│  NanoSeek stack:                                     │
│  NanoSeekModel → compile(model)    ← NO DDP layer   │
│                                                      │
│  All communication lives in DistMuonAdamW.step()     │
└─────────────────────────────────────────────────────┘
```

---

## 4. The Training Loop — Forward & Backward

**File**: `nanoseek/scripts/pre_train.py`, lines 1543-1670

This is where data flows through the model. Understanding this is essential because the gradients produced here are what the distributed optimizer operates on.

### Gradient Accumulation Loop

```python
# pre_train.py lines 1558-1607

train_loss_accum = 0.0
main_loss_accum = 0.0
mtp_loss_accum = 0.0
aux_loss_accum = 0.0

for micro_step in range(current_accum):      # ← (A)
```

**(A)** `current_accum` = number of micro-steps per training step

**First principles of gradient accumulation**: We want a large effective batch (e.g., 524,288 tokens) but can only fit a small micro-batch per GPU (e.g., 4 × 4096 = 16,384 tokens). With 8 GPUs:

```
Tokens per micro-batch per GPU:  4 × 4096 = 16,384
Tokens per micro-batch (world):  16,384 × 8 = 131,072
Target total batch:              524,288
Grad accum steps:                524,288 / 131,072 = 4

Each GPU does 4 forward+backward passes, accumulating gradients.
Then ONE optimizer step averages and applies.
```

**Gradient accumulation** = tích lũy gradient qua nhiều micro-step trước khi update weight. Mỗi `.backward()` CỘNG gradient vào `.grad` (không ghi đè). Sau `current_accum` lần, gradient tích lũy ≈ gradient của batch lớn gấp `current_accum` lần.

```python
    # Forward pass with BF16 autocast
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                        enabled=(device_type == "cuda")):
        mtp_lambda = get_mtp_loss_weight(...)
        outputs = model(
            x,
            labels=x,           # ← model does internal shift
            mtp_lambda=mtp_lambda,
        )
        loss = outputs['loss']  # scalar tensor on GPU
```

**Data flow through the forward pass**:

```
Input:  x = [B=4, S=4096]  (token IDs, dtype=int64)
              ↓
         ┌─────────────┐
         │  Embedding    │  x → [4, 4096, 2048] (hidden states, bf16)
         ├─────────────┤
         │  Layer 0     │  MLA attention + Dense FFN
         │  Layer 1     │  MLA attention + Dense FFN
         │  Layer 2-15  │  MLA attention + MoE FFN (64 experts, top-8)
         ├─────────────┤
         │  RMSNorm     │  [4, 4096, 2048] → [4, 4096, 2048]
         │  LM Head     │  [4, 4096, 2048] → [4, 4096, 32768] (logits)
         │  MTP Heads   │  Multi-Token Prediction (speculative decoding heads)
         └─────────────┘
              ↓
Output: {'loss': scalar, 'main_loss': scalar, 'mtp_loss': scalar,
         'aux_loss': scalar, 'logits': [4, 4096, 32768]}
```

```python
    # Detach component losses for logging (doesn't add to compute graph)
    train_loss_accum += loss.detach()
    main_loss_accum += outputs['main_loss'].detach()
    mtp_loss_accum += outputs['mtp_loss'].detach()
    aux_loss_accum += outputs['aux_loss'].detach()

    # Scale loss for gradient accumulation
    loss = loss / current_accum   # ← (B)
    loss.backward()               # ← (C)
```

**(B)** `loss = loss / current_accum`

**First principles**: `.backward()` ADDS to `.grad`, it doesn't replace. If we do 4 micro-steps, we get `grad = g₁ + g₂ + g₃ + g₄`. We want the AVERAGE `grad = (g₁ + g₂ + g₃ + g₄) / 4`. Dividing the loss by 4 means each backward produces `gₖ/4`, and the sum is `(g₁ + g₂ + g₃ + g₄) / 4`. Correct.

**Nguyên lý**: `.backward()` CỘNG vào `.grad`, không thay thế. Nếu chạy 4 micro-step, ta được `grad = g₁ + g₂ + g₃ + g₄`. Nhưng ta muốn trung bình `(g₁+g₂+g₃+g₄)/4`. Chia loss cho 4 trước khi backward → mỗi backward sinh ra `gₖ/4`, tổng = trung bình.

**(C)** `loss.backward()`

This is where PyTorch's autograd engine traces backward through the entire model:

```
┌────────────────────────────────────────────────────────────┐
│  Backward Pass (automatic differentiation)                  │
│                                                             │
│  loss (scalar)                                              │
│    ↓ ∂loss/∂logits                                          │
│  LM Head W [2048, 32768]  → .grad += ∂loss/∂W_lm           │
│    ↓                                                        │
│  Layer 15 (MoE):                                            │
│    MLA attention params → .grad += ...                      │
│    Gate router weights  → .grad += ...                      │
│    Expert 0..63 weights → .grad += ...  (only top-8 get     │
│    Shared expert weights → .grad += ... nonzero gradients)  │
│    ↓                                                        │
│  Layer 14 ... Layer 2 (same as 15)                          │
│    ↓                                                        │
│  Layer 1, 0 (Dense FFN instead of MoE)                      │
│    ↓                                                        │
│  Embedding [32768, 2048] → .grad += ∂loss/∂W_embed          │
└────────────────────────────────────────────────────────────┘
```

**Critical for MoE**: With top-8 routing out of 64 experts, only 8 experts receive gradients per token. The other 56 experts have `.grad = None` (or zero if they received zero tokens in the batch). This is why the optimizer must handle `None` gradients gracefully.

**Quan trọng cho MoE**: Với top-8 routing từ 64 experts, chỉ 8 experts nhận gradient cho mỗi token. 56 experts còn lại có `.grad = None`. Optimizer phải xử lý được `None` gradient.

```python
    # Prefetch next batch while GPU is busy with backward
    x, y, dataloader_state_dict = next(train_loader)
```

**Overlap insight**: While the GPU is still computing backward, the CPU fetches the next batch. This hides data loading latency behind GPU compute — a classic pipeline optimization.

### After the Micro-Step Loop: Optimizer Step

```python
# pre_train.py lines 1636-1656

# Gradient clipping (distributed-aware for multi-GPU)
clip_norm = float('inf') if args.no_grad_clip else config.max_grad_norm
if ddp:
    grad_norm = distributed_clip_grad_norm_(orig_model.parameters(), max_norm=clip_norm)
else:
    grad_norm = clip_grad_norm_(orig_model.parameters(), max_norm=clip_norm)

# Update learning rate
for group in optimizer.param_groups:
    group["lr"] = group["initial_lr"] * lrm

# THE CRITICAL CALL — all distributed communication happens inside:
optimizer.step()                    # ← All 3 phases happen here!
model.zero_grad(set_to_none=True)   # ← Free gradient memory
```

**`optimizer.step()`** is where ALL distributed communication happens. For single-GPU `MuonAdamW`, it's just local computation. For `DistMuonAdamW`, it's the entire 3-phase async pipeline described next.

**`model.zero_grad(set_to_none=True)`**: `set_to_none=True` doesn't zero the gradient tensors — it sets `.grad = None`, freeing the memory entirely. This is faster than memset and reduces peak memory. The next backward will allocate fresh gradient tensors.

---

## 5. DistMuonAdamW — The Heart of Distribution

**File**: `nanoseek/nanoseek/optim.py`, lines 314-556

### The 0-D CPU Tensor Trick

```python
# optim.py lines 372-384

def __init__(self, param_groups: list[dict]):
    super().__init__(param_groups, defaults={})
    # 0-D CPU tensors to avoid torch.compile recompilation when values change
    self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
    self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
    # ... (10 total tensors)
```

**First principles of torch.compile cache**: `torch.compile` traces the computation graph once and JIT-compiles it to optimized CUDA kernels. If an input changes **type or device**, the compiled graph is invalidated and must be recompiled (slow — seconds). But if a tensor's **value** changes while its type/shape/device stay the same, the compiled graph reuses the cached kernel.

Problem: LR changes every step (cosine decay). If we pass `lr=0.001` as a Python float, torch.compile sees a new constant and recompiles. Solution: pass `lr_t` as a 0-D CPU tensor — torch.compile sees "same tensor object, same shape `()`, same dtype `float32`" → reuses the compiled kernel. We just `.fill_()` the tensor with the new value.

**Nguyên lý torch.compile cache**: `torch.compile` biên dịch graph 1 lần thành CUDA kernel tối ưu. Nếu input thay đổi type/device → phải recompile (chậm). Nhưng nếu chỉ thay đổi giá trị trong tensor cùng shape/dtype/device → dùng lại kernel đã compile. Do đó ta dùng 0-D tensor và `.fill_()` giá trị mới mỗi step thay vì truyền Python float.

```
Step 100: lr_t.fill_(0.001)  → compiled kernel runs with lr=0.001
Step 101: lr_t.fill_(0.00099) → SAME compiled kernel, just different value
Step 102: lr_t.fill_(0.00098) → SAME compiled kernel

vs. without 0-D tensors:
Step 100: adamw_step(lr=0.001)   → compile (3 seconds)
Step 101: adamw_step(lr=0.00099) → recompile (3 seconds)  ← DISASTER
Step 102: adamw_step(lr=0.00098) → recompile (3 seconds)
```

### The 3-Phase `.step()` Method

```python
# optim.py lines 530-556

@torch.no_grad()
def step(self):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Phase 1: launch all async reduce ops
    reduce_infos: list[dict] = []
    for group in self.param_groups:
        if group['kind'] == 'adamw':
            reduce_infos.append(self._reduce_adamw(group, world_size))
        elif group['kind'] == 'muon':
            reduce_infos.append(self._reduce_muon(group, world_size))

    # Phase 2: wait for reduces, compute updates, launch gathers
    gather_list: list[dict] = []
    for group, info in zip(self.param_groups, reduce_infos):
        if group['kind'] == 'adamw':
            self._compute_adamw(group, info, gather_list, rank, world_size)
        elif group['kind'] == 'muon':
            self._compute_muon(group, info, gather_list, rank)

    # Phase 3: wait for gathers, copy back
    self._finish_gathers(gather_list)
```

**Pipeline visualization** (time flows left to right):

```
              Phase 1           Phase 2                Phase 3
              (launch)          (compute)              (finish)
         ┌──────────────┬─────────────────────┬───────────────┐
NCCL:    │ reduce_scatter│ all_gather(group A)  │ wait & copy   │
stream   │ (all groups)  │ ↗ overlap            │               │
         ├──────────────┼─────────────────────┤               │
Compute  │              │ wait(group A)         │               │
stream   │   (idle)     │ compute(group A)      │   (idle)      │
         │              │ wait(group B)         │               │
         │              │ compute(group B) ...  │               │
         └──────────────┴─────────────────────┴───────────────┘

Key insight: While GPU computes group B's update,
group A's all_gather runs on the NCCL stream simultaneously.
```

**Nhìn tổng quan pipeline**: Trong khi GPU tính optimizer update cho group B, thì `all_gather` của group A đang chạy song song trên NCCL stream. Communication và computation overlap → tổng thời gian ≈ max(compute, comm), không phải sum.

---

## 6. Phase 1 — Launch Async Reduces

### AdamW Reduce: Two Strategies Based on Size

```python
# optim.py lines 386-404

def _reduce_adamw(self, group: dict, world_size: int) -> dict:
    """Launch async reduce ops for AdamW group."""
    param_infos = {}
    for p in group['params']:
        grad = p.grad
        if p.numel() < 1024:                     # ← (A)
            # Small params: all_reduce
            future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
            param_infos[p] = dict(future=future, grad_slice=grad, is_small=True)
        else:
            # Large params: reduce_scatter
            assert grad.shape[0] % world_size == 0
            rank_size = grad.shape[0] // world_size
            grad_slice = torch.empty_like(grad[:rank_size])    # ← (B)
            future = dist.reduce_scatter_tensor(               # ← (C)
                grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True
            )
            param_infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False)
    return dict(param_infos=param_infos)
```

**(A)** Size threshold: 1024 elements

**Why two strategies?** NCCL communication has a fixed startup cost (~5-20μs) per operation. For small tensors (<1024 elements = 2-4 KB), the startup cost dominates and `reduce_scatter` + `all_gather` (two operations) is slower than a single `all_reduce`. For large tensors, `reduce_scatter` + `all_gather` enables ZeRO-2 memory savings (optimizer state sharded by N).

**Tại sao 2 chiến lược?** NCCL có overhead cố định ~5-20μs cho mỗi operation. Tensor nhỏ (<1024 phần tử = 2-4 KB) → overhead cố định > thời gian truyền data → `all_reduce` (1 op) nhanh hơn `reduce_scatter` + `all_gather` (2 op). Tensor lớn → `reduce_scatter` + `all_gather` tiết kiệm bộ nhớ optimizer state.

**`all_reduce` vs `reduce_scatter` — the fundamental difference:**

```
all_reduce(AVG):  Every rank ends up with the SAME averaged tensor
─────────────────────────────────────────────────────────────
Before:  Rank0=[a₀,b₀,c₀,d₀]  Rank1=[a₁,b₁,c₁,d₁]
After:   Rank0=[ā, b̄, c̄, d̄]   Rank1=[ā, b̄, c̄, d̄]     (ā = avg(a₀,a₁))

   Total data moved: N elements × N ranks = O(N²) bandwidth
   But NCCL uses ring/tree → O(N) effective


reduce_scatter(AVG):  Each rank gets 1/N of the averaged tensor
─────────────────────────────────────────────────────────────
Before:  Rank0=[a₀,b₀,c₀,d₀]  Rank1=[a₁,b₁,c₁,d₁]
After:   Rank0=[ā, b̄]          Rank1=[c̄, d̄]

   Each rank gets a DIFFERENT slice of the averaged result.
   Total output = N/ranks elements per rank → 1/N memory
```

**(B)** `grad_slice = torch.empty_like(grad[:rank_size])`

Pre-allocates the output buffer for `reduce_scatter`. Each rank will receive `rank_size = shape[0] / world_size` rows of the averaged gradient.

**Concrete example** with `embed_tokens` weight `[32768, 2048]` on 8 GPUs:

```
rank_size = 32768 / 8 = 4096

Rank 0: grad_slice = empty [4096, 2048]  ← will hold avg of rows 0-4095
Rank 1: grad_slice = empty [4096, 2048]  ← will hold avg of rows 4096-8191
...
Rank 7: grad_slice = empty [4096, 2048]  ← will hold avg of rows 28672-32767
```

**(C)** `dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True)`

- **Input**: `grad` — the FULL local gradient `[32768, 2048]`
- **Output**: `grad_slice` — this rank's portion of the averaged gradient `[4096, 2048]`
- **`async_op=True`**: Returns immediately, NCCL runs in background
- Returns a `Future` that we can `.wait()` on later

```
Rank 0 grad [32768, 2048]──┐
Rank 1 grad [32768, 2048]──┤  NCCL
Rank 2 grad [32768, 2048]──┤  reduce_scatter(AVG)
...                         │
Rank 7 grad [32768, 2048]──┘
         ↓
Rank 0 gets avg rows [0:4096]      = [4096, 2048]
Rank 1 gets avg rows [4096:8192]   = [4096, 2048]
...
Rank 7 gets avg rows [28672:32768] = [4096, 2048]
```

### Muon Reduce: Stack, Pad, Scatter

```python
# optim.py lines 406-424

def _reduce_muon(self, group: dict, world_size: int) -> dict:
    """Launch async reduce op for Muon group."""
    params = group['params']
    chunk_size = (len(params) + world_size - 1) // world_size    # ← (A)
    padded_num_params = chunk_size * world_size                   # ← (B)
    p = params[0]
    shape, device, dtype = p.shape, p.device, p.dtype

    # Stack grads and zero-pad to padded_num_params
    grad_stack = torch.stack([p.grad for p in params])            # ← (C)
    stacked_grads = torch.empty(padded_num_params, *shape, dtype=dtype, device=device)
    stacked_grads[:len(params)].copy_(grad_stack)                 # ← (D)
    if len(params) < padded_num_params:
        stacked_grads[len(params):].zero_()                       # ← (E)

    # Reduce_scatter to get this rank's chunk
    grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
    future = dist.reduce_scatter_tensor(                          # ← (F)
        grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True
    ).get_future()
    return dict(future=future, grad_chunk=grad_chunk,
                stacked_grads=stacked_grads, chunk_size=chunk_size)
```

**Concrete example**: 192 expert `w_gate` parameters, each `[480, 1280]`, on 8 GPUs:

**(A)** `chunk_size = ceil(192 / 8) = 24` — each rank "owns" 24 expert weights

**(B)** `padded_num_params = 24 × 8 = 192` (happens to divide evenly here)

If we had 195 params: `chunk_size = ceil(195/8) = 25`, `padded = 25 × 8 = 200`. The last 5 slots would be zero-padded.

**(C)** `torch.stack([p.grad for p in params])` → `[192, 480, 1280]`

All 192 gradients stacked into a single contiguous tensor. This is key to efficiency — one NCCL operation on this large tensor is vastly cheaper than 192 small operations.

**Tất cả 192 gradient được stack thành 1 tensor liên tục** `[192, 480, 1280]`. Một thao tác NCCL trên tensor lớn này rẻ hơn nhiều so với 192 thao tác nhỏ.

**(D-E)** Copy into padded buffer and zero remaining slots

```
stacked_grads [192, 480, 1280]:
┌────────────────────────────────┐
│ grad of expert 0   [480, 1280] │  ← slot 0
│ grad of expert 1   [480, 1280] │  ← slot 1
│ ...                             │
│ grad of expert 191 [480, 1280] │  ← slot 191
│ (zeros if padded)              │  ← slots 192-199 if needed
└────────────────────────────────┘
```

**(F)** `reduce_scatter_tensor(grad_chunk, stacked_grads, AVG, async=True)`

```
Before reduce_scatter (each rank has stacked_grads [192, 480, 1280]):

Rank 0: [g₀⁰, g₁⁰, ..., g₁₉₁⁰]   (192 grads from rank 0's data)
Rank 1: [g₀¹, g₁¹, ..., g₁₉₁¹]   (192 grads from rank 1's data)
...
Rank 7: [g₀⁷, g₁⁷, ..., g₁₉₁⁷]   (192 grads from rank 7's data)

After reduce_scatter (each rank gets chunk_size=24 averaged grads):

Rank 0: grad_chunk = [ḡ₀, ḡ₁, ..., ḡ₂₃]      experts 0-23
Rank 1: grad_chunk = [ḡ₂₄, ḡ₂₅, ..., ḡ₄₇]    experts 24-47
Rank 2: grad_chunk = [ḡ₄₈, ḡ₄₉, ..., ḡ₇₁]    experts 48-71
...
Rank 7: grad_chunk = [ḡ₁₆₈, ḡ₁₆₉, ..., ḡ₁₉₁] experts 168-191

where ḡᵢ = avg(gᵢ⁰, gᵢ¹, ..., gᵢ⁷) = averaged gradient for expert i
```

**Mỗi rank chỉ nhận 24 averaged gradient** (thay vì 192) → mỗi rank chỉ cần tính optimizer step cho 24 expert → optimizer state (momentum, variance) chỉ cần lưu cho 24 expert → tiết kiệm 8× bộ nhớ.

---

## 7. Phase 2 — Compute on Local Shards

### AdamW Compute: Shard or Replicate

```python
# optim.py lines 426-470

def _compute_adamw(self, group, info, gather_list, rank, world_size):
    param_infos = info['param_infos']
    for p in group['params']:
        pinfo = param_infos[p]
        pinfo['future'].wait()          # ← (A) Block until reduce completes
        grad_slice = pinfo['grad_slice']
        state = self.state[p]

        if pinfo['is_small']:
            p_slice = p                  # ← (B) REPLICATED: operate on full param
        else:
            rank_size = p.shape[0] // world_size
            p_slice = p[rank * rank_size:(rank + 1) * rank_size]  # ← (C) VIEW!
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p_slice)      # ← (D)
                state['exp_avg_sq'] = torch.zeros_like(p_slice)
        state['step'] += 1

        # ... fill 0-D tensors ...

        adamw_step_fused(
            p_slice, grad_slice, state['exp_avg'], state['exp_avg_sq'],
            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
        )
        if not pinfo['is_small']:
            future = dist.all_gather_into_tensor(    # ← (E)
                p, p_slice, async_op=True
            ).get_future()
            gather_list.append(dict(future=future, params=None))
```

**(A)** `pinfo['future'].wait()` — blocks until NCCL `reduce_scatter` completes. The gradient data is now valid in `grad_slice`.

**(B)** Small params: All ranks have the SAME averaged gradient (from `all_reduce`). Each rank updates the full parameter identically. Optimizer state (`exp_avg`, `exp_avg_sq`) is replicated — wasteful but these params are tiny (norm weights, biases).

**(C)** `p_slice = p[rank * rank_size:(rank + 1) * rank_size]` — this is a **VIEW**, not a copy!

**First principles of PyTorch views**: `p[start:end]` returns a tensor that shares the SAME underlying memory as `p`. Modifying `p_slice` modifies the corresponding rows of `p` in-place. This is critical for the all_gather that follows — it reads from `p` directly.

**`p_slice` là VIEW, không phải copy!** `p[start:end]` trả về tensor chia sẻ CÙNG bộ nhớ với `p`. Thay đổi `p_slice` sẽ thay đổi các hàng tương ứng trong `p`. Điều này quan trọng vì `all_gather` sẽ đọc trực tiếp từ `p`.

```
p = [32768, 2048] (on GPU)
     ┌─────────────────┐
     │ rows 0-4095     │ ← Rank 0's VIEW (p_slice)
     ├─────────────────┤
     │ rows 4096-8191  │ ← Rank 1's VIEW
     ├─────────────────┤
     │ ...              │
     ├─────────────────┤
     │ rows 28672-32767│ ← Rank 7's VIEW
     └─────────────────┘

Each rank's adamw_step_fused modifies its VIEW in-place →
modifies the corresponding rows of the FULL parameter p.
```

**(D)** Optimizer state is only allocated for the slice, not the full parameter:
```
Full param: [32768, 2048] = 134M floats
Rank's slice: [4096, 2048] = 16.8M floats → 8× less memory

exp_avg:    [4096, 2048]  (not [32768, 2048])
exp_avg_sq: [4096, 2048]  (not [32768, 2048])
→ ZeRO-2: optimizer state sharded across ranks
```

**(E)** After updating its slice, launch `all_gather` to broadcast updated slices to all ranks:

```
Before all_gather:
  Rank 0's p: [updated rows 0-4095    | stale rows 4096-32767]
  Rank 1's p: [stale rows 0-4095      | updated rows 4096-8191 | stale ...]
  ...

After all_gather:
  All ranks' p: [updated rows 0-4095 | updated rows 4096-8191 | ... | updated rows 28672-32767]
  ← Every rank has the complete, fully updated parameter!
```

### Muon Compute: Stack, Orthogonalize, Scatter

```python
# optim.py lines 472-520

def _compute_muon(self, group, info, gather_list, rank):
    info['future'].wait()                    # ← Wait for reduce_scatter
    params = group['params']
    chunk_size = info['chunk_size']          # 24 (for 192 experts / 8 GPUs)
    grad_chunk = info['grad_chunk']          # [24, 480, 1280] — this rank's averaged grads

    # How many params does this rank own?
    start_idx = rank * chunk_size            # Rank 3 → start at expert 72
    num_owned = min(chunk_size, max(0, len(params) - start_idx))  # usually 24
```

**Mỗi rank "sở hữu" 24 expert.** Rank 0 sở hữu expert 0-23, Rank 1 sở hữu expert 24-47, v.v. Mỗi rank CHỈ tính Muon update cho experts nó sở hữu.

```python
    # Get or create group-level state — SHARDED (only chunk_size, not all 192)
    state = self.state[p]
    if "momentum_buffer" not in state:
        state["momentum_buffer"] = torch.zeros(chunk_size, *shape, ...)
        # [24, 480, 1280] — NOT [192, 480, 1280]!
    if "second_momentum_buffer" not in state:
        state_shape = (chunk_size, shape[-2], 1)  # [24, 480, 1] — factored
        state["second_momentum_buffer"] = torch.zeros(state_shape, ...)
```

**Memory savings**: Each rank stores momentum for 24 experts instead of 192 → 8× memory reduction. For NanoSeek with 64 experts × 3 weights each = 192 Muon params of shape `[480, 1280]`:

```
Full momentum buffer:    192 × 480 × 1280 × 4 bytes = 471 MB
Per-rank (8 GPUs):        24 × 480 × 1280 × 4 bytes =  59 MB  ← 8× savings
```

```python
    # Build output buffer for all_gather
    updated_params = torch.empty(chunk_size, *shape, ...)  # [24, 480, 1280]

    if num_owned > 0:
        owned_params = [params[start_idx + i] for i in range(num_owned)]
        stacked_owned = torch.stack(owned_params)  # [24, 480, 1280]

        # Fill 0-D tensors and run fused kernel
        muon_step_fused(
            grad_chunk[:num_owned],                           # averaged grads
            stacked_owned,                                     # current params
            state["momentum_buffer"][:num_owned],             # momentum
            state["second_momentum_buffer"][:num_owned],      # variance
            self._muon_momentum_t, self._muon_lr_t,
            self._muon_wd_t, self._muon_beta2_t,
            group["ns_steps"], red_dim,
        )
        updated_params[:num_owned].copy_(stacked_owned)  # copy updated values
```

**What `muon_step_fused` does** (the Muon algorithm in 4 stages):

```
Input:  grad_chunk [24, 480, 1280]    (averaged gradients for 24 experts)
        stacked_owned [24, 480, 1280] (current parameter values)

Stage 1: Nesterov Momentum
    m ← 0.97·m + 0.03·g           (exponential moving average)
    g ← 0.03·g + 0.97·m           (look-ahead gradient)

Stage 2: Polar Express Orthogonalization (5 iterations)
    X ← g / ||g||                  (normalize)
    for i in 1..5:
        A ← X^T X                 (or X X^T for wide matrices)
        B ← b·A + c·A²
        X ← a·X + X·B             (converges to polar factor ≈ UV^T)
    g ← X                          (now has uniform singular values)

Stage 3: NorMuon Variance Reduction
    v ← EMA of per-neuron variance
    scale ← 1/√v × (global_norm / scaled_norm)
    g ← g × scale                  (normalize per-neuron update magnitudes)

Stage 4: Cautious Update
    mask ← (g × θ ≥ 0)            (update only in alignment direction)
    θ ← θ - lr·g - lr·wd·θ·mask   (parameter update with cautious decay)

Output: stacked_owned [24, 480, 1280] is modified IN-PLACE
```

**Polar Express lý giải**: SVD phân tích ma trận G = UΣV^T. Muon muốn dùng UV^T (bỏ singular values Σ) để mọi hướng update có cùng magnitude. Newton-Schulz/Polar Express tìm xấp xỉ UV^T qua 5 iterations mà không cần tính SVD thật (SVD tốn O(n³), Polar Express tốn O(n² × 5 iterations)).

```python
    # Reuse stacked_grads buffer for all_gather output (saves memory)
    stacked_params = info["stacked_grads"]    # ← (F) Buffer reuse!
    future = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True)
    gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))
```

**(F)** **Buffer reuse**: `stacked_grads` was the input to `reduce_scatter`. After `reduce_scatter` completes, that buffer is no longer needed. We reuse it as the output buffer for `all_gather`, saving `192 × 480 × 1280 × 2 bytes = 236 MB` of GPU memory.

**Tái sử dụng buffer**: `stacked_grads` là input của `reduce_scatter`. Sau khi `reduce_scatter` xong, buffer này không còn cần. Ta tái sử dụng nó làm output buffer cho `all_gather`, tiết kiệm ~236 MB bộ nhớ GPU.

---

## 8. Phase 3 — Gather & Reassemble

```python
# optim.py lines 522-528

def _finish_gathers(self, gather_list: list) -> None:
    """Wait for all gathers and copy Muon params back."""
    for info in gather_list:
        info["future"].wait()               # ← (A)
        if info["params"] is not None:      # ← (B) Muon only
            # Copy from stacked buffer back to individual params
            torch._foreach_copy_(           # ← (C)
                info["params"],
                list(info["stacked_params"][:len(info["params"])].unbind(0))
            )
```

**(A)** Wait for each `all_gather` to complete. At this point:
- For AdamW large params: `p` already has all updated rows (all_gather wrote into `p` directly via the view trick)
- For Muon params: `stacked_params` has all 192 updated expert weights

**(B)** `info["params"] is not None` distinguishes Muon entries (which need copy-back) from AdamW entries (which don't — the all_gather wrote into `p` directly).

**(C)** `torch._foreach_copy_` — the batch copy operation:

```
stacked_params [192, 480, 1280]:
┌──────────────────────┐
│ updated expert 0     │ ← from Rank 0's computation
│ updated expert 1     │ ← from Rank 0's computation
│ ...                   │
│ updated expert 23    │ ← from Rank 0's computation
│ updated expert 24    │ ← from Rank 1's computation
│ ...                   │
│ updated expert 191   │ ← from Rank 7's computation
└──────────────────────┘
           ↓ unbind(0) + foreach_copy_
┌──────┐ ┌──────┐ ┌──────┐       ┌────────┐
│exp 0 │ │exp 1 │ │exp 2 │ ...   │exp 191 │
└──────┘ └──────┘ └──────┘       └────────┘
   ↓         ↓         ↓              ↓
model.layers[2].ffn.routed_experts[0].w_gate.weight
model.layers[2].ffn.routed_experts[0].w_up.weight
model.layers[2].ffn.routed_experts[0].w_down.weight
...
```

`unbind(0)` splits `[192, 480, 1280]` into 192 tensors of `[480, 1280]`. `torch._foreach_copy_` copies each tensor back to the original parameter in the model. This uses CUDA memcpy internally — faster than a Python loop.

**`unbind(0)` tách [192, 480, 1280] thành 192 tensor [480, 1280]. `foreach_copy_` copy mỗi tensor về parameter gốc trong model.** Dùng CUDA memcpy nội bộ → nhanh hơn vòng lặp Python.

---

## 9. Distributed Gradient Clipping

**File**: `nanoseek/scripts/pre_train.py`

### The Problem

```python
# BEFORE (broken for multi-GPU):
grad_norm = clip_grad_norm_(orig_model.parameters(), max_norm=1.0)
```

Each rank clips based on LOCAL gradient norms (un-averaged). Rank 0 might clip by 0.5×, Rank 1 by 0.8×. After the optimizer averages, the effective gradient is inconsistently scaled.

**Vấn đề**: Mỗi rank clip dựa trên norm gradient LOCAL (chưa average). Rank 0 có thể clip 0.5×, Rank 1 clip 0.8×. Sau khi optimizer average, gradient hiệu dụng bị scale không nhất quán.

### The Fix

```python
def distributed_clip_grad_norm_(parameters, max_norm):
    """Gradient clipping consistent across distributed ranks."""
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)

    # Step 1: Local squared norm
    local_norm_sq = torch.zeros(1, device=parameters[0].device)
    for p in parameters:
        local_norm_sq += p.grad.data.float().pow(2).sum()

    # Step 2: All-reduce to get GLOBAL sum of squared norms
    dist.all_reduce(local_norm_sq, op=dist.ReduceOp.SUM)

    # Step 3: RMS norm ≥ true averaged norm (Cauchy-Schwarz)
    world_size = dist.get_world_size()
    global_norm = (local_norm_sq / world_size).sqrt().item()

    # Step 4: Same clip factor on ALL ranks
    if max_norm < float('inf') and global_norm > max_norm:
        clip_coef = max_norm / (global_norm + 1e-6)
        for p in parameters:
            p.grad.data.mul_(clip_coef)

    return torch.tensor(global_norm)
```

**Cauchy-Schwarz guarantee**: Let gᵢ be rank i's gradient. The true averaged norm is `||mean(gᵢ)||`. We compute `sqrt(mean(||gᵢ||²))`, which by Cauchy-Schwarz is ≥ `||mean(gᵢ)||`. So our clip is **conservative** — we may clip slightly more than needed, but we NEVER under-clip.

```
Rank 0: ||g₀|| = 2.0  →  ||g₀||² = 4.0
Rank 1: ||g₁|| = 1.5  →  ||g₁||² = 2.25

all_reduce(SUM): total_sq = 4.0 + 2.25 = 6.25
global_norm = sqrt(6.25 / 2) = sqrt(3.125) = 1.77

True averaged norm = ||(g₀+g₁)/2|| ≤ 1.77  (by Cauchy-Schwarz)

With max_norm=1.0: clip_coef = 1.0 / 1.77 = 0.565
Both ranks multiply their gradients by 0.565 → consistent!
```

---

## 10. The Complete Data Flow Diagram

```
╔══════════════════════════════════════════════════════════════════════════╗
║                    NANOSEEK DISTRIBUTED TRAINING                         ║
║                    (One Complete Training Step)                           ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  ┌─────────────────────────────────────────────────────────────────┐     ║
║  │  EACH RANK INDEPENDENTLY (no communication)                     │     ║
║  │                                                                 │     ║
║  │  for micro_step in range(grad_accum):                           │     ║
║  │      x, y = next(dataloader)      # different data per rank     │     ║
║  │      with autocast(bf16):                                       │     ║
║  │          outputs = model(x)       # same model, different data  │     ║
║  │          loss = outputs['loss']                                 │     ║
║  │      (loss / grad_accum).backward()  # gradients ACCUMULATE     │     ║
║  │                                                                 │     ║
║  │  Result: each rank has LOCAL accumulated gradients               │     ║
║  │          (not yet averaged across ranks)                         │     ║
║  └─────────────────────────────────────────────────────────────────┘     ║
║                              ↓                                           ║
║  ┌─────────────────────────────────────────────────────────────────┐     ║
║  │  DISTRIBUTED GRADIENT CLIPPING (one all_reduce)                  │     ║
║  │                                                                  │     ║
║  │  Each rank: local_sq = sum(g².sum() for all params)              │     ║
║  │  all_reduce(local_sq, SUM)  → global_sq on all ranks             │     ║
║  │  global_norm = sqrt(global_sq / world_size)                      │     ║
║  │  if global_norm > max_norm:                                      │     ║
║  │      all_grads *= max_norm / global_norm   # same factor!        │     ║
║  └─────────────────────────────────────────────────────────────────┘     ║
║                              ↓                                           ║
║  ┌─────────────────────────────────────────────────────────────────┐     ║
║  │  DistMuonAdamW.step()                                           │     ║
║  │                                                                 │     ║
║  │  ┌─── PHASE 1: Launch Async Reduces ──────────────────────┐     │     ║
║  │  │  AdamW small: all_reduce(grad, AVG, async)              │     │     ║
║  │  │  AdamW large: reduce_scatter(grad→shard, AVG, async)    │     │     ║
║  │  │  Muon:        stack 192 grads → reduce_scatter(async)   │     │     ║
║  │  │  → All ops running on NCCL stream in background         │     │     ║
║  │  └────────────────────────────────────────────────────────┘     │     ║
║  │                              ↓                                  │     ║
║  │  ┌─── PHASE 2: Compute Updates + Launch Gathers ──────────┐     │     ║
║  │  │  For each group:                                        │     │     ║
║  │  │    future.wait()  ← block until reduce completes        │     │     ║
║  │  │    AdamW: adamw_step_fused(p_slice, grad_slice, ...)    │     │     ║
║  │  │           → all_gather(p, p_slice, async)               │     │     ║
║  │  │    Muon:  muon_step_fused(grad_chunk, owned_params,...) │     │     ║
║  │  │           → all_gather(stacked_params, updated, async)  │     │     ║
║  │  └────────────────────────────────────────────────────────┘     │     ║
║  │                              ↓                                  │     ║
║  │  ┌─── PHASE 3: Finish Gathers ───────────────────────────┐     │     ║
║  │  │  For each gather:                                       │     │     ║
║  │  │    future.wait()  ← block until gather completes        │     │     ║
║  │  │    Muon: foreach_copy_(params, stacked_params.unbind()) │     │     ║
║  │  │    AdamW: (already in-place via view trick)             │     │     ║
║  │  └────────────────────────────────────────────────────────┘     │     ║
║  └─────────────────────────────────────────────────────────────────┘     ║
║                              ↓                                           ║
║  ┌─────────────────────────────────────────────────────────────────┐     ║
║  │  POST-STEP (each rank independently)                            │     ║
║  │                                                                 │     ║
║  │  model.zero_grad(set_to_none=True)   # free gradient memory     │     ║
║  │  update_load_balance_bias()          # MoE expert rebalancing   │     ║
║  │  ema_tracker.step(model)             # EMA weight update        │     ║
║  │                                                                 │     ║
║  │  All ranks now have IDENTICAL parameters (invariant maintained)  │     ║
║  └─────────────────────────────────────────────────────────────────┘     ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### The Fundamental Invariant

**At the start and end of every training step, ALL ranks have IDENTICAL model parameters.**

This invariant is maintained because:
1. All ranks start with the same seed → same initial weights
2. Gradients are averaged across ranks (reduce_scatter + all_gather)
3. All ranks apply the same optimizer algorithm with the same hyperparameters
4. All ranks use the same clip factor (distributed_clip_grad_norm_)
5. All ranks call the same MoE bias update function with the same inputs

If this invariant is ever broken, training silently diverges — different ranks produce different gradients for different models, and averaging them is meaningless.

**Bất biến cơ bản: Mọi rank có CÙNG model parameters tại đầu và cuối mỗi training step.** Nếu bất biến này bị phá vỡ, training âm thầm diverge — các rank khác nhau sinh gradient cho model khác nhau, và average chúng vô nghĩa. Bất biến được duy trì vì: cùng seed → cùng weight ban đầu; gradient được average; cùng optimizer; cùng clip factor; cùng MoE bias update.

---

## Summary: Why This Design?

| Design Choice | Why |
|---|---|
| No DDP wrapper | Muon needs to stack same-shape params; DDP would all-reduce them individually |
| Optimizer-internal comm | Enables reduce_scatter (ZeRO-2 sharding) + Muon-aware batching |
| 3-phase async pipeline | Overlaps communication with computation |
| NCCL backend | GPU-direct communication via NVLink (10-50× faster than CPU-routed) |
| 0-D CPU tensors | Prevents torch.compile recompilation on LR/momentum changes |
| Buffer reuse | stacked_grads → all_gather output, saves ~236 MB |
| View-based sharding | p_slice is a view of p, so all_gather writes to p directly |
| Conservative grad clip | Cauchy-Schwarz upper bound ensures all ranks clip identically |

**Tổng kết**: Thiết kế này KHÔNG dùng DDP vì Muon cần stack gradient cùng shape. Thay vào đó, tất cả communication nằm trong optimizer, cho phép: (1) reduce_scatter → tiết kiệm bộ nhớ 8×, (2) batching → 1 NCCL op thay vì 192, (3) async pipeline → overlap compute/comm. Kết quả: training speed gần tuyến tính với số GPU, với overhead communication minimal.
