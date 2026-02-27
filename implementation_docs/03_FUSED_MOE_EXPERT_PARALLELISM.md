# Fused MoE Kernels & Expert Parallelism — Production Implementation

**Author**: Distinguished AI Researcher & Principal Engineer, Foundation Models Division
**Date**: February 2026
**Scope**: Fused Triton grouped GEMM for MoE dispatch; expert parallelism for multi-GPU scaling
**Prerequisites**: `model/model.py` MoE classes, `docs/02_MOE_DEEP_DIVE.md`

---

## 1. Problem Statement

NanoSeek's `token_centric_dispatch()` works correctly but has three performance bottlenecks:

**Python for-loop over 64 experts** (`model/model.py` line 555): Each of 64 iterations launches 3 CUDA kernels (gate_proj, up_proj, down_proj), synchronizes, returns to Python. That's 192 kernel launches with ~5-15μs Python overhead each = 1-3ms pure overhead per MoE layer. With 14 MoE layers: 14-42ms per forward pass — comparable to actual GEMM time at small batch.

**Non-fused memory operations**: The permute→compute→unpermute pipeline materializes multiple intermediate tensors in HBM. `argsort`, `gather`, `split`, `cat`, element-wise weight multiply, and `scatter_add_` are each separate memory-bound kernels with near-zero arithmetic intensity.

**No communication overlap**: Expert parallelism requires all-to-all dispatch across GPUs. Without overlap, communication latency blocks computation entirely.

**Target**: 3-5× single-GPU speedup; near-linear scaling across 8 GPUs for MoE layers.

---

## 2. First Principles Analysis

**Token-centric vs expert-centric dispatch**: Token-centric sorts tokens by expert, creating contiguous batches with coalesced memory access. One argsort replaces E separate gathers. NanoSeek already uses this; the fused kernel preserves it.

**Grouped GEMM**: All 64 experts have identical shapes (2048→768→2048 SwiGLU). Stack all expert weights into `[E, D, I]` tensors and launch a single kernel where each thread block handles one tile of one expert's GEMM. Replaces 192 kernel launches with 2 (gate+up, then silu+down).

**Expert parallelism**: With 64 experts on 8 GPUs, each GPU owns 8 experts. Tokens route via NCCL `all_to_all_single`. On H100 NVLink (900 GB/s), transferring ~2 GB takes ~2.2ms — overlappable with shared expert computation.

**Communication-computation overlap**: Shared experts (2 always-active) process ALL tokens independently of routed experts. Launch shared computation concurrently with the dispatch all-to-all to hide communication latency.

**Why Triton**: Portable (NVIDIA/AMD/Intel), rapid iteration (~150 lines vs ~2000 CUDA), autotuning, 85-95% of hand-tuned CUDA for these problem sizes.

---

## 3. Architecture & Data Flow

### Fused MoE Kernel Pipeline

```
Input: x [N, D=2048], indices [N, K=8], weights [N, K=8]

PERMUTE:  argsort(flat_indices) → sorted_token_ids, expert_offsets [E+1]

GROUPED GEMM (single launch, all experts in parallel):
  Per expert e, tokens [off[e]:off[e+1]]:
    gate_out = tokens @ W_gate[e]   [M_e, 2048] × [2048, 768] → [M_e, 768]
    up_out   = tokens @ W_up[e]     [M_e, 2048] × [2048, 768] → [M_e, 768]
    hidden   = SiLU(gate_out) * up_out                          [M_e, 768]
    out_e    = hidden @ W_down[e]   [M_e, 768] × [768, 2048]  → [M_e, 2048]

UNPERMUTE + WEIGHT (fused into down-proj epilogue):
  output[orig_token_id[p]] += weight[p] * out_e[p]   (atomic add)

Output: combined [N, D=2048]
```

### Expert Parallelism Communication

```
8 GPUs, 64 experts (8 local per GPU). Per GPU batch: N_local tokens.

1. All-to-All Dispatch: send tokens for remote experts to owning GPUs
   GPU 0 → GPU 1: tokens for experts 8-15 (~N×K/E tokens, ~512 KB each)
2. Local Expert Compute: fused GEMM on received tokens
3. All-to-All Combine: reverse dispatch, scatter-add at original positions
```

### Memory Layout

```
Expert weights (stacked): W_gate [64, 2048, 768] = 192 MB BF16
                          W_up   [64, 2048, 768] = 192 MB
                          W_down [64, 768, 2048]  = 192 MB
                          Total per MoE layer: 576 MB (14 layers = 8 GB)

Sorted token buffer: [N×K, D] — contiguous per-expert batches
  Expert 0: [M_0, 2048] | Expert 1: [M_1, 2048] | ... | Expert 63: [M_63, 2048]
  sum(M_e) = N×K; E[M_e] = N×K/E (if balanced)
```

---

## 4. Production Code

### 4a. Fused MoE Triton Kernel — `model/kernels/fused_moe.py`

```python
"""
Fused MoE Triton Kernel — Grouped GEMM with SwiGLU.
Replaces Python for-loop with single kernel launch processing ALL experts.
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=2, num_warps=8),
    ],
    key=["N", "K"],
)
@triton.jit
def _fused_gate_up_kernel(
    Input_ptr, W_gate_ptr, W_up_ptr,           # inputs
    Gate_out_ptr, Up_out_ptr,                   # outputs
    Expert_offsets_ptr, Sorted_ids_ptr,         # routing metadata
    N: tl.constexpr, K: tl.constexpr,           # I=inter_dim, D=hidden_dim
    stride_input_t, stride_input_d,
    stride_w_e, stride_w_d, stride_w_i,
    stride_out_t, stride_out_i,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Grouped GEMM: gate + up projection for all experts simultaneously.
    Grid: (ceil(max_M_e/BLOCK_M) * ceil(I/BLOCK_N), E).
    Each program handles one tile of one expert's GEMM."""

    expert_id = tl.program_id(1)
    tile_id = tl.program_id(0)

    expert_start = tl.load(Expert_offsets_ptr + expert_id)
    expert_end = tl.load(Expert_offsets_ptr + expert_id + 1)
    num_tokens_e = expert_end - expert_start

    num_n_tiles = tl.cdiv(N, BLOCK_N)
    tile_m = tile_id // num_n_tiles
    tile_n = tile_id % num_n_tiles
    row_start = tile_m * BLOCK_M
    col_start = tile_n * BLOCK_N

    if row_start >= num_tokens_e:
        return

    # Row/column index ranges for this tile
    rows = row_start + tl.arange(0, BLOCK_M)
    row_mask = rows < num_tokens_e
    global_rows = expert_start + rows
    cols = col_start + tl.arange(0, BLOCK_N)
    col_mask = cols < N

    # FP32 accumulators for numerical stability
    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Tiled matmul over hidden dimension (K=D=2048)
    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_idx < K

        # Load input tile [BLOCK_M, BLOCK_K]
        inp = tl.load(
            Input_ptr + global_rows[:, None] * stride_input_t + k_idx[None, :] * stride_input_d,
            mask=row_mask[:, None] & k_mask[None, :], other=0.0,
        )
        # Load W_gate tile [BLOCK_K, BLOCK_N]
        wg = tl.load(
            W_gate_ptr + expert_id * stride_w_e + k_idx[:, None] * stride_w_d + cols[None, :] * stride_w_i,
            mask=k_mask[:, None] & col_mask[None, :], other=0.0,
        )
        # Load W_up tile [BLOCK_K, BLOCK_N]
        wu = tl.load(
            W_up_ptr + expert_id * stride_w_e + k_idx[:, None] * stride_w_d + cols[None, :] * stride_w_i,
            mask=k_mask[:, None] & col_mask[None, :], other=0.0,
        )
        acc_gate += tl.dot(inp, wg)
        acc_up += tl.dot(inp, wu)

    # Store results in BF16
    mask_2d = row_mask[:, None] & col_mask[None, :]
    tl.store(Gate_out_ptr + global_rows[:, None] * stride_out_t + cols[None, :] * stride_out_i,
             acc_gate.to(tl.bfloat16), mask=mask_2d)
    tl.store(Up_out_ptr + global_rows[:, None] * stride_out_t + cols[None, :] * stride_out_i,
             acc_up.to(tl.bfloat16), mask=mask_2d)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=8),
    ],
    key=["N", "K"],
)
@triton.jit
def _fused_silu_down_kernel(
    Gate_out_ptr, Up_out_ptr, W_down_ptr,       # inputs
    Expert_offsets_ptr, Sorted_ids_ptr, Sorted_weights_ptr,
    Output_ptr,                                  # output [N_tokens, D]
    N: tl.constexpr, K: tl.constexpr,            # D=hidden_dim, I=inter_dim
    stride_inter_t, stride_inter_i,
    stride_wd_e, stride_wd_i, stride_wd_d,
    stride_out_t, stride_out_d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Fused SiLU(gate)*up → down_proj → weight → atomic scatter-add.
    SiLU is fused into the GEMM prologue to avoid materializing the
    full intermediate hidden buffer in HBM."""

    expert_id = tl.program_id(1)
    tile_id = tl.program_id(0)

    expert_start = tl.load(Expert_offsets_ptr + expert_id)
    expert_end = tl.load(Expert_offsets_ptr + expert_id + 1)
    num_tokens_e = expert_end - expert_start

    num_n_tiles = tl.cdiv(N, BLOCK_N)
    tile_m = tile_id // num_n_tiles
    tile_n = tile_id % num_n_tiles
    row_start = tile_m * BLOCK_M
    col_start = tile_n * BLOCK_N

    if row_start >= num_tokens_e:
        return

    rows = row_start + tl.arange(0, BLOCK_M)
    row_mask = rows < num_tokens_e
    global_rows = expert_start + rows
    cols = col_start + tl.arange(0, BLOCK_N)
    col_mask = cols < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_idx < K
        mask_mk = row_mask[:, None] & k_mask[None, :]

        # Load gate output and apply SiLU: x * sigmoid(x)
        gate = tl.load(
            Gate_out_ptr + global_rows[:, None] * stride_inter_t + k_idx[None, :] * stride_inter_i,
            mask=mask_mk, other=0.0,
        ).to(tl.float32)
        gate_act = gate * tl.sigmoid(gate)

        # Load up output, compute SwiGLU
        up = tl.load(
            Up_out_ptr + global_rows[:, None] * stride_inter_t + k_idx[None, :] * stride_inter_i,
            mask=mask_mk, other=0.0,
        ).to(tl.float32)
        hidden = gate_act * up

        # Load W_down tile
        wd = tl.load(
            W_down_ptr + expert_id * stride_wd_e + k_idx[:, None] * stride_wd_i + cols[None, :] * stride_wd_d,
            mask=k_mask[:, None] & col_mask[None, :], other=0.0,
        )
        acc += tl.dot(hidden.to(tl.bfloat16), wd)

    # Apply routing weights and atomic scatter-add to original positions
    orig_ids = tl.load(Sorted_ids_ptr + global_rows, mask=row_mask, other=0)
    wts = tl.load(Sorted_weights_ptr + global_rows, mask=row_mask, other=0.0)
    weighted = acc.to(tl.float32) * wts[:, None]

    tl.atomic_add(
        Output_ptr + orig_ids[:, None] * stride_out_t + cols[None, :] * stride_out_d,
        weighted.to(tl.bfloat16),
        mask=row_mask[:, None] & col_mask[None, :],
    )


def fused_moe_forward(
    x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor,
    w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Fused MoE forward: replaces token_centric_dispatch + expert for-loop."""
    N, D = x.shape
    K = indices.shape[1]
    E = num_experts
    I = w_gate.shape[2]
    device = x.device

    # Phase 1: Sort tokens by expert
    flat_indices = indices.view(-1)
    flat_weights = weights.view(-1).float()
    token_ids = torch.arange(N, device=device).unsqueeze(1).expand(-1, K).reshape(-1)

    sorted_order = torch.argsort(flat_indices, stable=True)
    sorted_token_ids = token_ids[sorted_order]
    sorted_weights = flat_weights[sorted_order]

    expert_counts = torch.bincount(flat_indices[sorted_order].int(), minlength=E)
    expert_offsets = torch.zeros(E + 1, dtype=torch.int64, device=device)
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

    sorted_input = x[sorted_token_ids]
    total = sorted_order.shape[0]

    # Phase 2: Fused gate + up projection
    gate_out = torch.empty(total, I, dtype=torch.bfloat16, device=device)
    up_out = torch.empty(total, I, dtype=torch.bfloat16, device=device)

    max_m = expert_counts.max().item()
    grid = (triton.cdiv(max_m, 64) * triton.cdiv(I, 64), E)

    _fused_gate_up_kernel[grid](
        sorted_input, w_gate, w_up, gate_out, up_out,
        expert_offsets, sorted_token_ids,
        N=I, K=D,
        stride_input_t=sorted_input.stride(0), stride_input_d=sorted_input.stride(1),
        stride_w_e=w_gate.stride(0), stride_w_d=w_gate.stride(1), stride_w_i=w_gate.stride(2),
        stride_out_t=gate_out.stride(0), stride_out_i=gate_out.stride(1),
    )

    # Phase 3: Fused SiLU + down projection + weighted scatter
    output = torch.zeros(N, D, dtype=torch.bfloat16, device=device)
    grid_down = (triton.cdiv(max_m, 64) * triton.cdiv(D, 64), E)

    _fused_silu_down_kernel[grid_down](
        gate_out, up_out, w_down,
        expert_offsets, sorted_token_ids, sorted_weights, output,
        N=D, K=I,
        stride_inter_t=gate_out.stride(0), stride_inter_i=gate_out.stride(1),
        stride_wd_e=w_down.stride(0), stride_wd_i=w_down.stride(1), stride_wd_d=w_down.stride(2),
        stride_out_t=output.stride(0), stride_out_d=output.stride(1),
    )
    return output


class FusedMoELayer(nn.Module):
    """Drop-in replacement for Expert ModuleList — stacks weights for grouped GEMM."""

    def __init__(self, num_experts: int, dim: int, inter_dim: int, dtype=torch.bfloat16):
        super().__init__()
        self.num_experts = num_experts
        self.w_gate = nn.Parameter(torch.empty(num_experts, dim, inter_dim, dtype=dtype))
        self.w_up = nn.Parameter(torch.empty(num_experts, dim, inter_dim, dtype=dtype))
        self.w_down = nn.Parameter(torch.empty(num_experts, inter_dim, dim, dtype=dtype))
        for w in [self.w_gate, self.w_up, self.w_down]:
            nn.init.kaiming_uniform_(w.data.float())
            w.data = w.data.to(dtype)

    @classmethod
    def from_expert_list(cls, experts: nn.ModuleList, dtype=torch.bfloat16):
        """Convert existing Expert ModuleList to fused stacked representation."""
        E = len(experts)
        dim = experts[0].gate_proj.in_features
        inter_dim = experts[0].gate_proj.out_features
        layer = cls(E, dim, inter_dim, dtype=dtype)
        with torch.no_grad():
            for e, exp in enumerate(experts):
                layer.w_gate.data[e] = exp.gate_proj.weight.T.to(dtype)
                layer.w_up.data[e] = exp.up_proj.weight.T.to(dtype)
                layer.w_down.data[e] = exp.down_proj.weight.T.to(dtype)
        return layer

    def forward(self, x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return fused_moe_forward(x, indices, weights, self.w_gate, self.w_up, self.w_down, self.num_experts)
```

### 4b. Expert Parallelism Module — `model/expert_parallel.py`

```python
"""
Expert Parallelism: distribute 64 experts across GPUs with all-to-all routing.
Each GPU owns experts [rank*E_local : (rank+1)*E_local] and runs them on
tokens received from all GPUs via NCCL all_to_all_single.
"""
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, Optional, Tuple


class ExpertParallelConfig:
    def __init__(self, num_experts=64, num_experts_per_tok=8, ep_size=1, capacity_factor=1.25):
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.ep_size = ep_size
        self.experts_per_rank = num_experts // ep_size
        self.capacity_factor = capacity_factor

    def compute_capacity(self, num_tokens: int) -> int:
        balanced = (num_tokens * self.num_experts_per_tok) // self.num_experts
        return int(balanced * self.capacity_factor)


class AllToAllDispatcher:
    """Handles all-to-all token dispatch and combine for expert parallelism."""

    def __init__(self, config: ExpertParallelConfig, group=None):
        self.config = config
        self.group = group

    def dispatch(self, x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor):
        """Route tokens to expert-owning GPUs via all-to-all.

        Args:
            x: [N, D] local tokens
            indices: [N, K] expert indices
            weights: [N, K] routing weights

        Returns: (recv_tokens, recv_local_indices, recv_weights, metadata)
        Concrete shapes (8 GPUs, N=65536, K=8):
          - send per pair: ~65536 tokens, ~512KB
          - recv total: ~65536 tokens (balanced)
        """
        N, D = x.shape
        K = indices.shape[1]
        ep_size = self.config.ep_size
        epr = self.config.experts_per_rank
        rank = dist.get_rank(self.group)
        device = x.device

        flat_indices = indices.view(-1)
        flat_weights = weights.view(-1)
        flat_tokens = torch.arange(N, device=device).unsqueeze(1).expand(-1, K).reshape(-1)
        dest_rank = flat_indices // epr

        # Sort by destination rank for contiguous all-to-all buffers
        sort_idx = torch.argsort(dest_rank, stable=True)
        sorted_tokens = x[flat_tokens[sort_idx]]
        sorted_indices = flat_indices[sort_idx]
        sorted_weights = flat_weights[sort_idx]
        sorted_orig = flat_tokens[sort_idx]

        # Exchange counts
        send_counts = torch.stack([(dest_rank[sort_idx] == r).sum() for r in range(ep_size)])
        all_counts = [torch.zeros_like(send_counts) for _ in range(ep_size)]
        dist.all_gather(all_counts, send_counts, group=self.group)
        recv_counts = torch.stack(all_counts)[:, rank]

        send_splits = send_counts.tolist()
        recv_splits = recv_counts.tolist()

        # All-to-all data exchange
        recv_tok = torch.empty(sum(recv_splits), D, dtype=x.dtype, device=device)
        recv_idx = torch.empty(sum(recv_splits), dtype=indices.dtype, device=device)
        recv_wt = torch.empty(sum(recv_splits), dtype=weights.dtype, device=device)

        for buf_out, buf_in in [(recv_tok, sorted_tokens), (recv_idx, sorted_indices), (recv_wt, sorted_weights)]:
            dist.all_to_all_single(buf_out, buf_in, recv_splits, send_splits, group=self.group)

        metadata = dict(send_splits=send_splits, recv_splits=recv_splits,
                        sort_idx=sort_idx, sorted_orig=sorted_orig, N=N, D=D)
        return recv_tok, recv_idx % epr, recv_wt, metadata

    def combine(self, expert_output: torch.Tensor, metadata: dict) -> torch.Tensor:
        """Reverse all-to-all: return expert results to originating GPUs."""
        N, D = metadata["N"], metadata["D"]
        device = expert_output.device

        recv = torch.empty(sum(metadata["send_splits"]), D, dtype=expert_output.dtype, device=device)
        dist.all_to_all_single(recv, expert_output, metadata["send_splits"], metadata["recv_splits"], group=self.group)

        output = torch.zeros(N, D, dtype=expert_output.dtype, device=device)
        output.scatter_add_(0, metadata["sorted_orig"].unsqueeze(-1).expand(-1, D), recv)
        return output


class ExpertParallelMoE(nn.Module):
    """MoE with expert parallelism. Each GPU holds shared experts (replicated)
    + a shard of routed experts. Shared expert compute overlaps with all-to-all."""

    def __init__(self, moe_layer, ep_config: ExpertParallelConfig, process_group=None):
        super().__init__()
        self.gate = moe_layer.gate
        self.shared_experts = moe_layer.shared_experts
        self.n_routed_experts = moe_layer.n_routed_experts
        self.n_activated_experts = moe_layer.n_activated_experts
        self.ep_config = ep_config
        self.dispatcher = AllToAllDispatcher(ep_config, group=process_group)

        rank = dist.get_rank(process_group) if process_group else 0
        start = rank * ep_config.experts_per_rank
        self.local_experts = nn.ModuleList(list(moe_layer.experts[start:start + ep_config.experts_per_rank]))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        B, S, D = x.shape
        x_flat = x.view(B * S, D)

        # Shared experts: launch on separate stream to overlap with all-to-all
        shared_output = None
        if self.shared_experts is not None:
            shared_stream = torch.cuda.Stream()
            with torch.cuda.stream(shared_stream):
                shared_output = sum(e(x_flat) for e in self.shared_experts) if len(self.shared_experts) > 1 else self.shared_experts[0](x_flat)

        weights, indices = self.gate(x_flat)

        # All-to-all dispatch → local compute → all-to-all combine
        recv_tok, recv_idx, recv_wt, meta = self.dispatcher.dispatch(x_flat, indices, weights)

        # Apply capacity constraint: drop lowest-weight tokens if overloaded
        capacity = self.ep_config.compute_capacity(B * S)
        for local_e in range(self.ep_config.experts_per_rank):
            mask = recv_idx == local_e
            if mask.sum() > capacity:
                positions = mask.nonzero(as_tuple=True)[0]
                _, order = recv_wt[positions].sort()
                drop = positions[order[:mask.sum() - capacity]]
                recv_tok, recv_idx, recv_wt = [t[~torch.isin(torch.arange(len(t), device=t.device), drop)] for t in [recv_tok, recv_idx, recv_wt]]

        from model.model import token_centric_dispatch
        local_out = token_centric_dispatch(recv_tok, recv_idx.unsqueeze(-1), recv_wt.unsqueeze(-1), self.local_experts)
        routed = self.dispatcher.combine(local_out, meta)

        if shared_output is not None:
            torch.cuda.current_stream().wait_stream(shared_stream)
            routed = routed + shared_output

        return routed.view(B, S, D), {"expert_load": self.gate.expert_load.clone()}
```

### 4c. Enhanced Gate — `model/model.py` Modifications

```python
# Add to Gate class: pre-compute metadata for fused kernel path
def forward_with_fused_metadata(self, x: torch.Tensor) -> dict:
    """Gate forward returning pre-sorted metadata for the fused kernel."""
    weights, indices = self.forward(x)
    N, K, E = x.shape[0], self.n_activated_experts, self.n_routed_experts

    flat = indices.view(-1)
    token_ids = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, K).reshape(-1)
    order = torch.argsort(flat, stable=True)

    counts = torch.bincount(flat[order].int(), minlength=E)
    offsets = torch.zeros(E + 1, dtype=torch.int64, device=x.device)
    offsets[1:] = counts.cumsum(0)

    return dict(weights=weights, indices=indices, sorted_order=order,
                expert_offsets=offsets, sorted_token_ids=token_ids[order],
                sorted_weights=weights.view(-1)[order], expert_counts=counts)


# JIT-compiled fused group routing (avoids intermediate allocations)
@torch.jit.script
def fused_group_topk(
    scores: torch.Tensor, n_groups: int, n_limited_groups: int,
    n_activated: int, experts_per_group: int, expert_bias: torch.Tensor, training: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused group scoring + selection + top-k in single JIT pass.
    NanoSeek: 8 groups × 8 experts/group, select 4 groups, then top-8."""
    N, E = scores.shape
    sel = scores + expert_bias.unsqueeze(0) if training else scores
    grouped = sel.view(N, n_groups, experts_per_group)
    _, top_g = grouped.max(-1).values.topk(n_limited_groups, dim=-1)
    mask = torch.zeros(N, n_groups, dtype=torch.bool, device=scores.device)
    mask.scatter_(1, top_g, True)
    expert_mask = mask.unsqueeze(-1).expand(-1, -1, experts_per_group).reshape(N, E)
    sel = sel.masked_fill(~expert_mask, float("-inf"))
    _, topk_idx = sel.topk(n_activated, dim=-1)
    return scores.gather(-1, topk_idx), topk_idx
```

### 4d. Load Balancing Monitor

```python
"""Real-time expert load monitor: tracks utilization, detects dead experts."""
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from collections import deque


class ExpertLoadMonitor:
    def __init__(self, num_experts=64, window_size=100, dead_threshold=0.01, imbalance_threshold=0.20):
        self.num_experts = num_experts
        self.window_size = window_size
        self.dead_threshold = dead_threshold
        self.imbalance_threshold = imbalance_threshold
        self.load_history: Dict[int, deque] = {}
        self.bias_history: Dict[int, deque] = {}
        self.step = 0

    def update(self, expert_stats: Dict[int, Dict[str, torch.Tensor]], step: int) -> List[dict]:
        self.step = step
        alerts = []
        for layer, stats in expert_stats.items():
            load = stats["expert_load"].float().cpu()
            bias = stats["expert_bias"].float().cpu()
            self.load_history.setdefault(layer, deque(maxlen=self.window_size)).append(load)
            self.bias_history.setdefault(layer, deque(maxlen=self.window_size)).append(bias)

            if len(self.load_history[layer]) >= 10:
                avg = torch.stack(list(self.load_history[layer])).mean(0)
                mean = avg.mean()
                if mean > 0:
                    # Dead expert detection
                    dead = (avg / mean < self.dead_threshold).nonzero(as_tuple=True)[0].tolist()
                    if dead:
                        alerts.append(dict(type="dead_expert", layer=layer, step=step, ids=dead))
                    # Imbalance detection
                    cv = (avg.std() / mean).item()
                    if cv > self.imbalance_threshold:
                        alerts.append(dict(type="imbalance", layer=layer, step=step, cv=cv))
        return alerts

    def get_summary(self) -> dict:
        summary = {"step": self.step, "layers": {}}
        for layer, hist in self.load_history.items():
            if not hist:
                continue
            avg = torch.stack(list(hist)).mean(0)
            m = avg.mean().item()
            summary["layers"][layer] = dict(
                mean=m, std=avg.std().item(), cv=avg.std().item() / m if m > 0 else float("inf"),
                min=avg.min().item(), max=avg.max().item(),
                dead_count=int((avg < avg.mean() * self.dead_threshold).sum()),
            )
        return summary

    @staticmethod
    def recover_dead_experts(model, layer_idx: int, dead_ids: List[int]):
        """Reinitialize dead experts from the most-loaded donor (ST-MoE approach)."""
        for layer in model.layers:
            if not hasattr(layer, "ffn") or layer.layer_idx != layer_idx:
                continue
            moe = layer.ffn
            load = moe.gate.expert_load.clone()
            load[dead_ids] = -1
            donor_id = load.argmax().item()
            donor = moe.experts[donor_id]
            for did in dead_ids:
                with torch.no_grad():
                    for (_, pd), (_, ps) in zip(moe.experts[did].named_parameters(), donor.named_parameters()):
                        pd.copy_(ps + torch.randn_like(ps) * 0.01)
                    moe.gate.expert_bias[did] = moe.gate.expert_bias.max() + 0.1
            break
```

---

## 5. File Placement

```
model/
├── kernels/
│   ├── __init__.py            # Exports FusedMoELayer, fused_moe_forward
│   └── fused_moe.py           # Fused MoE Triton kernel (§4a)
├── expert_parallel.py         # Expert parallelism (§4b)
├── load_monitor.py            # Load balancing monitor (§4d)
└── model.py                   # Gate.forward_with_fused_metadata (§4c)
                               # MoE gains use_fused=True option
```

Integration in `MoE.__init__`: add `use_fused: bool = False` flag. When True, call `FusedMoELayer.from_expert_list(self.experts)` and route through `self.fused_experts(x_flat, indices, weights)` in forward instead of `token_centric_dispatch`.

---

## 6. Verification

**Correctness** — fused output matches Python dispatch within BF16 tolerance:
```python
# Compare fused_moe_forward vs token_centric_dispatch on same input/weights
# Expected: max_abs_diff < 1e-2, cosine_similarity > 0.9999
# The fused kernel accumulates in FP32 and stores BF16; ordering differences
# in the reduction cause minor numerical divergence.
```

**Load balancing** — expert variance converges after bias adjustment:
```python
# After 1000 steps with gamma=0.001:
# Coefficient of variation (std/mean) of expert load per layer < 0.20
# No dead experts (all experts > 1% of expected load)
```

**Expert parallelism** — correct cross-GPU token routing:
```python
# Deterministic routing: token i → expert (i % 64). After dispatch+combine,
# verify each token received output from its assigned expert.
# Run: torchrun --nproc_per_node=8 verify_ep.py
```

---

## 7. Performance Targets

| Metric | Baseline (Python) | Fused Target | Source of speedup |
|--------|-------------------|-------------|-------------------|
| MoE layer (N=4096) | ~12ms | ~3ms (4×) | Kernel launch elimination |
| MoE layer (N=65536) | ~85ms | ~20ms (4.25×) | + Memory fusion |
| 14 MoE layers total | ~170ms | ~42ms (4×) | Combined |
| Expert parallel 8-GPU | N/A | 0.85× linear | 15% comm overhead |
| All-to-all latency (H100 NVLink) | N/A | ~2ms | 2GB @ 900 GB/s |

Speedup breakdown: kernel launch overhead elimination (~1.5×) × memory traffic reduction (~1.5×) × better SM utilization (~1.5×) ≈ 3.4× conservative.

---

## 8. Gotchas

### Expert Capacity Factor

When an expert receives more tokens than `(N×K/E) × capacity_factor`, excess tokens are dropped (lowest-probability first). Without this cap, a popular expert can cause OOM or create latency long-tails. With K=8, losing 1 expert costs ~12.5% of that token's representation quality — tolerable while bias adjustment corrects the imbalance over ~100-500 steps.

### Gradient Flow Through Top-K

`topk` selects discrete indices (non-differentiable). Gradients flow through the continuous weights via `gather` on the score tensor. Unselected experts get zero gradient for that token. DeepSeek V3 does NOT use a straight-through estimator — the bias-based balancing is sufficient and avoids noisy gradient artifacts.

### Load Imbalance During Early Training

Before bias stabilizes (~500-2000 steps), expert loads can be 75× skewed. Sigmoid scoring (unlike softmax) doesn't normalize across experts, so initial weight variance causes a few experts to dominate. Gamma=0.001 requires ~1000 steps for significant correction. **Implication for fused kernels**: tile sizes must handle highly variable expert batch sizes — use conservative BLOCK_M (64 or 128).

### Memory Spikes During All-to-All

Peak temporary memory per GPU: `2 × N × K × D × sizeof(BF16)`. For N=65536, K=8, D=2048: ~4 GB. Mitigations: chunked all-to-all (8K token chunks), pre-allocated reusable buffers, and capacity factor limiting max recv size.

### Numerical Differences: Fused vs Reference

FP32 accumulation with BF16 stores causes operation-ordering differences. Expected: max abs diff < 1e-2, cosine sim > 0.9999. These do not affect training convergence — the key invariant is identical routing decisions (same expert selection), with only intermediate arithmetic differing.

---

*"Fusing the MoE dispatch removes 191 boundaries where the GPU sits idle. The grouped GEMM turns 64 tiny problems into one large problem — exactly what GPUs are designed to solve."*

— Principal Engineer's Note, Foundation Models Division, 2026
