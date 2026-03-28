# MLA Production Inference Strategy — Verified Research Plan

## From NanoSeek Reference Implementation to DeepSeek-Scale Deployment

**Author**: NanoSeek Engineering
**Date**: 2026-03-12
**Status**: Research-verified plan, pre-implementation
**Target**: DeepSeek-V3 scale (671B params, 128 heads, 128K context, H800/H100)
**Our Scale**: NanoSeek-1B (1.08B active, 32 heads, 4K→8K context)

---

## Methodology

Every claim in this document was verified against primary sources:
- DeepSeek-V3 official inference code (`deepseek-ai/DeepSeek-V3/inference/model.py`)
- DeepSeek-V3 technical report (arXiv:2412.19437)
- ISCA 2025 hardware paper (arXiv:2505.09343)
- DeepSeek Open Source Week Day 6 infrastructure disclosure
- SGLang source code (`sgl-project/sglang`)
- FlashMLA repository (`deepseek-ai/FlashMLA`)
- AMD ROCm AITER blog posts

Verdicts: TRUE / PARTIALLY TRUE / FALSE — with primary source citations.

---

## Table of Contents

1. [Strategy 1: Weight Absorption](#strategy-1-weight-absorption)
2. [Strategy 2: KV Cache Architecture](#strategy-2-kv-cache-architecture)
3. [Strategy 3: Fused CUDA Kernels](#strategy-3-fused-cuda-kernels)
4. [Strategy 4: FP8 Quantized KV Cache](#strategy-4-fp8-quantized-kv-cache)
5. [Strategy 5: Speculative Decoding](#strategy-5-speculative-decoding)
6. [Strategy 6: Prefill-Decode Disaggregation](#strategy-6-prefill-decode-disaggregation)
7. [Strategy 7: Softmax Scale with YaRN mscale](#strategy-7-softmax-scale-with-yarn-mscale)
8. [Strategy 8: Dual Micro-Batch Overlap](#strategy-8-dual-micro-batch-overlap)
9. [Strategy 9: On-Disk KV Cache](#strategy-9-on-disk-kv-cache)
10. [Gap Analysis: Our Code vs DeepSeek](#gap-analysis)
11. [Implementation Priority](#implementation-priority)

---

## Strategy 1: Weight Absorption

### Verified Claims

| # | Claim | Verdict | Source |
|---|-------|---------|--------|
| 1.1 | `attn_impl: Literal["naive", "absorb"] = "absorb"` is the default | **TRUE** | `inference/model.py` line 15 |
| 1.2 | They do NOT pre-fuse W_UK into W_Q or W_UV into W_O | **TRUE** | `inference/model.py`, MLA.forward() absorb branch |
| 1.3 | Runtime absorption via einsum patterns | **TRUE** | Exact einsum strings match: `"bshd,hdc->bshc"`, `"bshc,btc->bsht"`, `"bshr,btr->bsht"` |
| 1.4 | Uses `n_local_heads` not `n_heads` (TP-aware) | **TRUE** (our doc said `n_heads`) | `inference/model.py` uses `self.n_local_heads` |
| 1.5 | `wkv_b` is `ColumnParallelLinear` | **TRUE** | Explicit in code |
| 1.6 | `weight_dequant` conditional on `self.wkv_b.scale` | **TRUE** | `wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(...)` |
| 1.7 | GitHub issue #848 confirms absorption pattern | **PARTIALLY TRUE** | Issue describes the pattern but was auto-closed with no official response |

### What DeepSeek Actually Does (Verified)

```
# Step 1: Conditional FP8 dequantization
wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(
    self.wkv_b.weight, self.wkv_b.scale, block_size
)

# Step 2: Reshape to per-head view (zero-copy .view())
wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
#                   ^^^^^^^^^^^^^^^^ NOT n_heads — accounts for TP sharding

# Step 3: Absorb W_UK into query at runtime
q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])

# Step 4: Two-component attention score on compressed representations
scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
          torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale

# Step 5: Attention output through compressed path
x = torch.einsum("bsht,btc->bshc", attn_weights, self.kv_cache[:bsz, :end_pos])
x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])  # W_UV absorption
```

### Why NOT Pre-Fuse (Verified Reasoning)

1. **FP8 quantization** (TRUE): DeepSeek stores weights in FP8 E4M3 with per-block scales (block_size=128). Pre-fusing W_UK into W_Q would create a matrix of size `[n_heads, q_lora_rank, kv_lora_rank]` = `[128, 1536, 512]` = 100M elements per layer — harder to quantize cleanly and larger than keeping them separate.

2. **Tensor parallelism** (TRUE): `wkv_b` is `ColumnParallelLinear`, sharded along the head dimension across GPUs. Each GPU holds `n_local_heads = n_heads / tp_size` heads worth of weights. Pre-fusing would require gathering weights across GPUs or duplicating storage.

3. **Memory reuse** (TRUE): The same `wkv_b` tensor serves dual purpose — first half for K-absorption (W_UK), second half for V-absorption (W_UV). Pre-fusing would require separate weight matrices, doubling parameter memory for this layer.

### Our Implementation Status

**What we have** (in `model.py:440-487`):
- `prepare_for_inference()` extracts `_w_uk` and `_w_uv` from `wkv_b.weight`
- Mathematically identical to DeepSeek's runtime approach
- Uses `.view()` for the reshape (zero-copy, correct)

**Gaps to close**:
1. Our `_forward_inference()` is a stub (`pass`) — needs full implementation
2. We pre-extract weights in `prepare_for_inference()`, while DeepSeek reshapes at each forward call. Both are correct, but we should match DeepSeek's pattern for FP8 compatibility (dequant must happen at runtime)
3. We use `num_heads` everywhere — need to add `n_local_heads` concept for TP support
4. No FP8 weight dequantization path

### Detailed Implementation Plan

**Step 1.1: Implement `_forward_inference()` (HIGH PRIORITY)**

The inference forward must implement exactly this flow:
```
Input: hidden_states [B, S, hidden_size]

1. Q projection: h → W_QA → q_norm → W_QB → split [q_nope | q_pe]
   q_nope: [B, S, n_heads, qk_nope_head_dim]
   q_pe:   [B, S, n_heads, qk_rope_head_dim]

2. KV projection: h → W_KVA → split [c_kv | k_pe_raw]
   c_kv:     [B, S, kv_lora_rank]  (apply kv_norm)
   k_pe_raw: [B, S, qk_rope_head_dim]

3. Apply RoPE to q_pe and k_pe_raw

4. Cache update: write c_kv and k_pe into static buffers at start_pos

5. Weight absorption (runtime, NOT pre-extracted):
   wkv_b = self.wkv_b.weight.view(n_heads, -1, kv_lora_rank)
   q_nope_absorbed = einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :qk_nope_head_dim])

6. Two-component attention:
   nope_scores = einsum("bshc,btc->bsht", q_nope_absorbed, kv_cache[:bsz, :end_pos])
   rope_scores = einsum("bshr,btr->bsht", q_pe, pe_cache[:bsz, :end_pos])
   scores = (nope_scores + rope_scores) * softmax_scale

7. Softmax + attention output:
   attn = softmax(scores, dim=-1)
   x = einsum("bsht,btc->bshc", attn, kv_cache[:bsz, :end_pos])

8. V absorption + output projection:
   x = einsum("bshc,hdc->bshd", x, wkv_b[:, -v_head_dim:])
   output = wo(x.reshape(B, S, n_heads * v_head_dim))
```

**Decision: Runtime vs Pre-extracted absorption**

For NanoSeek-1B (no FP8, no TP), pre-extracting in `prepare_for_inference()` is fine and equivalent. For production scale, runtime reshaping is necessary because:
- FP8 dequant must happen per-forward-call (weights stored quantized)
- TP-aware `n_local_heads` changes based on world size

**Recommendation**: Implement both paths. Use pre-extracted for NanoSeek-1B testing, add runtime path with FP8 support as a production mode.

**Step 1.2: Add TP-aware head count**

Add to config:
```python
tp_size: int = 1  # Tensor parallelism degree
```

In MLA:
```python
self.n_local_heads = num_heads // tp_size
```

Use `n_local_heads` in all inference-path einsum operations.

**Step 1.3: FP8 weight dequant path (PHASE 2)**

Follow DeepSeek's `kernel.py` pattern:
- Block-wise quantization with `block_size=128`
- Per-block FP32 scale factors
- `weight_dequant(weight, scale, block_size)` → BF16 tensor
- Conditional: only when `scale is not None`

This is Phase 2 work — not needed for correctness, only for production efficiency.

---

## Strategy 2: KV Cache Architecture

### Verified Claims

| # | Claim | Verdict | Source |
|---|-------|---------|--------|
| 2.1 | DeepSeek uses `register_buffer("kv_cache", zeros(max_batch, max_seq, kv_lora_rank))` | **TRUE** | `inference/model.py`, with `persistent=False` |
| 2.2 | Separate `pe_cache` for RoPE keys | **TRUE** | `register_buffer("pe_cache", zeros(max_batch, max_seq, qk_rope_head_dim))` |
| 2.3 | Position tracking via `start_pos` integer | **TRUE** | Forward takes `start_pos: int`, writes at `[start_pos:end_pos]` |
| 2.4 | Memory calc: 8×16384×(512+64)×2 = 150 MB/layer | **TRUE** | Arithmetic correct for their config |
| 2.5 | MLA pages ~56× smaller than MHA pages | **TRUE** | 18 KB vs 1 MB per page verified (see calculations below) |
| 2.6 | SGLang uses `--enable-dp-attention` for DeepSeek | **TRUE** | Real flag, documented in SGLang DeepSeek docs |

### Page Size Calculations (Independently Verified)

**MLA page (DeepSeek-V3 config)**:
- `kv_lora_rank = 512`, `qk_rope_head_dim = 64`
- Per token: (512 + 64) × 2 bytes (BF16) = 1,152 bytes
- 16-token page: 16 × 1,152 = **18,432 bytes = 18 KB per page per layer**

**Equivalent MHA page (same model)**:
- `num_attention_heads = 128`, `qk_nope_head_dim = 128`, `v_head_dim = 128`
- Per token: 2 (K+V) × 128 (heads) × 128 (head_dim) × 2 bytes = 65,536 bytes
- 16-token page: 16 × 65,536 = **1,048,576 bytes = 1 MB per page per layer**

**Compression ratio**: 1,048,576 / 18,432 = **56.9×**

**NanoSeek-1B page (our config)**:
- `kv_lora_rank = 143`, `qk_rope_head_dim = 32`
- Per token: (143 + 32) × 2 = 350 bytes
- 16-token page: 16 × 350 = **5,600 bytes = 5.5 KB per page per layer**
- Equivalent MHA: 2 × 32 (heads) × 64 (head_dim) × 2 = 8,192 bytes/token
- MHA page: 16 × 8,192 = 131 KB
- Our compression: **23.4×**

### Why DP Attention for MLA (Verified)

Standard TP splits KV cache along the head dimension. But MLA's `kv_cache` has shape `[B, T, kv_lora_rank]` — **no head dimension**. The compressed latent is shared across all heads. So:
- TP splitting along heads → each GPU duplicates the full `kv_cache` (wasteful)
- DP attention → each GPU holds full `kv_cache` for a **subset of sequences**
- After MLA computation, an all-gather synchronizes hidden states across GPUs
- Then MoE proceeds with expert parallelism

SGLang documents this at `docs/references/deepseek.md` and recommends `--enable-dp-attention --tp 16 --dp 2` for multi-node deployment.

### Our Implementation Status

**What we have**:
- Dynamic `torch.cat` cache in `_forward_training()` (lines 351-355)
- Cache format: `(c_kv, k_pe)` tuple
- Works but has all the problems of dynamic allocation

**Gaps to close**:
1. No static pre-allocated buffers
2. No `start_pos` / `end_pos` integer tracking
3. No `max_batch_size` / `max_seq_len` in config
4. No page table interface

### Detailed Implementation Plan

**Step 2.1: Add cache config parameters**

```python
@dataclass
class InferenceConfig:
    max_batch_size: int = 8       # Max concurrent sequences
    max_seq_len: int = 8192       # Max sequence length (Phase 2 context)
    cache_dtype: str = "bfloat16" # Cache precision
```

**Step 2.2: Static buffer allocation in MLA.__init__**

Following DeepSeek's exact pattern:
```python
# Only allocate during inference setup, not during training
def allocate_cache(self, max_batch_size: int, max_seq_len: int):
    self.register_buffer("kv_cache",
        torch.zeros(max_batch_size, max_seq_len, self.kv_lora_rank),
        persistent=False)
    self.register_buffer("pe_cache",
        torch.zeros(max_batch_size, max_seq_len, self.qk_rope_head_dim),
        persistent=False)
```

**Step 2.3: Forward signature change**

Training path keeps `past_key_value` tuple for gradient computation.
Inference path uses `start_pos: int` for static buffer indexing:

```python
def _forward_inference(self, hidden_states, start_pos: int, ...):
    bsz, seq_len, _ = hidden_states.shape
    end_pos = start_pos + seq_len

    # Write to static buffer (no allocation)
    self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
    self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)

    # Read full valid range
    scores = einsum("bshc,btc->bsht", q_absorbed, self.kv_cache[:bsz, :end_pos])
```

**Step 2.4: Memory budget calculation for NanoSeek-1B**

```
Per layer: max_batch × max_seq × (kv_lora_rank + rope_dim) × dtype_size
         = 8 × 8192 × (143 + 32) × 2 bytes
         = 8 × 8192 × 350
         = 22.9 MB per layer

16 layers: 22.9 × 16 = 367 MB total cache
```

This is trivially small — confirms MLA's cache efficiency advantage even at nano scale.

**Decision: Keep both cache modes**

- **Training**: Keep `torch.cat` tuple cache. Required for gradient flow through cached values. DeepSeek also uses different cache strategies for training vs inference.
- **Inference**: Use static buffers with `start_pos` tracking. Match DeepSeek's exact pattern.

---

## Strategy 3: Fused CUDA Kernels

### Verified Claims

| # | Claim | Verdict | Source |
|---|-------|---------|--------|
| 3.1 | DeepSeek reference uses PyTorch einsum, not custom kernels | **TRUE** | `inference/model.py` uses `torch.einsum` |
| 3.2 | AITER MLA kernel 17× speedup | **PARTIALLY TRUE** | 17× is kernel-level vs unoptimized baseline; practical speedup is 1.5-2× |
| 3.3 | FlashAttention-MLA exists | **TRUE** | DeepSeek open-sourced `FlashMLA` (github.com/deepseek-ai/FlashMLA) |
| 3.4 | SGLang has custom Triton kernels for MLA | **TRUE** | PR #905, v0.3 showed 7× throughput improvement |
| 3.5 | Decode is memory-bound (GEMV) | **TRUE** | 128K × 512 × 2 = 131 MB read vs 16.8 GFLOP compute — clearly memory-bound |
| 3.6 | Prefill uses FlashAttention (GEMM) | **TRUE** | Standard tiled approach works for prefill |

### Corrected AITER Performance Claims

The strategy document claims "17× speedup" for AITER. The actual numbers from AMD's ROCm blog:
- **Kernel-level** vs unoptimized baseline: up to 17× (technically correct but misleading)
- **MLA decode kernel** across batch sizes: up to **1.47×** speedup
- **MLA decode kernel** across context lengths: up to **2×** speedup
- **End-to-end** with SGLang integration: 52% prefill latency reduction, 47% decode latency reduction

For our planning purposes, expect **1.5-2× practical speedup** from fused kernels, not 17×.

### Memory Access Analysis (Verified)

Without fusion (our current einsum approach), the absorbed attention does:
```
Pass 1: q_nope_absorbed @ kv_cache^T  → read kv_cache [B, T, 512]     (131 MB at 128K)
Pass 2: q_pe @ pe_cache^T             → read pe_cache  [B, T, 64]     (16 MB at 128K)
Pass 3: softmax(scores) @ kv_cache    → read kv_cache AGAIN [B, T, 512] (131 MB at 128K)
                                                            Total: 278 MB per layer
```

With fusion (single-pass):
```
Load each kv_cache+pe_cache row ONCE:
  - Compute nope_score and rope_score contributions
  - Accumulate online softmax (FlashAttention-style)
  - Accumulate V output (kv_cache IS the value)
                                                            Total: 147 MB per layer (1.9× reduction)
```

### FlashMLA Details (Verified)

DeepSeek's FlashMLA (`github.com/deepseek-ai/FlashMLA`):
- Designed for Hopper GPUs (H800/H100)
- Handles MLA's larger effective head dimension (576 vs standard 128)
- Operates on compressed latent, no full KV decompression during decode
- Reports up to 660 TFLOPS on H800 SXM5
- Used in SGLang as one of multiple MLA attention backends

SGLang now supports these MLA backends:
- FlashAttention3 (with MLA extension)
- FlashInfer
- FlashMLA (DeepSeek's kernel)
- CutlassMLA
- TRTLLM MLA (Blackwell-optimized)
- Triton (earliest MLA implementation)

### Implementation Plan for NanoSeek

**Phase 3.1: torch.compile (Week 3)**

Before writing custom kernels, get free speedup:
```python
@torch.compile(mode="reduce-overhead")
def _forward_inference(self, ...):
    ...
```

Expected: 2-3× speedup on the einsum path with zero code changes. The Triton backend will auto-fuse sequential einsum operations where possible.

**Phase 3.2: Triton MLA decode kernel (Week 4)**

For decode (seq_len=1), write a Triton kernel that:
1. Takes `q_nope_absorbed [B, 1, H, kv_lora_rank]`, `q_pe [B, 1, H, rope_dim]`
2. Takes `kv_cache [B, T, kv_lora_rank]`, `pe_cache [B, T, rope_dim]`
3. In a single pass over T positions:
   - Load `kv_cache[t]` and `pe_cache[t]` into shared memory
   - Compute `nope_score[t] = q_nope_absorbed @ kv_cache[t]`
   - Compute `rope_score[t] = q_pe @ pe_cache[t]`
   - Combined score = `(nope_score + rope_score) * scale`
   - Online softmax: update running max, running sum, running weighted sum
4. Output: `attn_output [B, H, kv_lora_rank]` (weighted sum of kv_cache)
5. Apply W_UV absorption outside kernel (separate matmul)

**Phase 3.3: FlashAttention for MLA prefill (Week 5-6)**

Two options:
- **Option A**: Use standard FlashAttention on the "naive" expanded K/V (what DeepSeek's naive mode does). During prefill, the expansion cost is amortized over S tokens, so this is efficient.
- **Option B**: Modified FlashAttention that accepts two Q/K components and tiles over `kv_lora_rank` instead of `head_dim`.

**Recommendation**: Use Option A for prefill (simple, proven, FA-compatible), custom Triton for decode (where absorption is essential).

---

## Strategy 4: FP8 Quantized KV Cache

### Verified Claims

| # | Claim | Verdict | Source |
|---|-------|---------|--------|
| 4.1 | DeepSeek "proposes" FP8 mixed precision training | **TRUE** (paper says "propose", not "design") | arXiv:2412.19437, Section 3.3 |
| 4.2 | Block-wise quantization with block_size=128 | **TRUE** | `kernel.py` in DeepSeek-V3 repo |
| 4.3 | E4M3 for weights, E5M2 for gradients | **TRUE** | Paper Section 3.3 |
| 4.4 | Reference code caches in BF16 | **TRUE** | `inference/model.py` |
| 4.5 | SGLang/vLLM support FP8 KV cache for MLA | **TRUE** | SGLang docs, LMSYS blog |
| 4.6 | 3.8× prefill speedup from FP8 on GB200 | **PARTIALLY TRUE** | 3.8× is from combined optimizations (FP8 + NVFP4 + EP + PD disagg), not FP8 alone. FP8 attention alone ≈ 1.8× |

### Why MLA's KV Cache Is Quantization-Friendly (Verified Reasoning)

The strategy document's reasoning is sound:
1. **Learned bottleneck**: c_kv is a trained 512-d compression. The model learned to represent information in this space, producing smoother activation distributions than raw hidden states. This is verified by the fact that DeepSeek trains in FP8 and the compressed representations survive the low precision.

2. **Low dimensionality**: Quantization error is spread across 512 values, not 32K+. Per-token error is proportionally lower.

3. **Block-wise overhead is negligible**: 512 / 128 = 4 blocks → 4 scale factors × 4 bytes = 16 bytes overhead per token. This is 3% of the 512-byte FP8 payload.

### Corrected Compression Numbers

The strategy document's compression calculations:
```
BF16: 576 × 2 = 1,152 bytes/token/layer → 57× vs MHA     ✓ CORRECT
FP8:  576 × 1 + 16 = 592 bytes/token/layer → 110× vs MHA  ✓ CORRECT
INT4: 576/2 + 32 = 320 bytes/token/layer → 203× vs MHA     ✓ CORRECT (arithmetic)
```

But INT4 quality impact is unverified — no published results on INT4 MLA KV cache. FP8 is the safe bet given DeepSeek trains natively in FP8.

### Implementation Plan

**Phase 4.1: INT8 per-token quantization (Week 5)**

Simplest approach, minimal quality loss:
```python
def quantize_kv_int8(c_kv):
    scale = c_kv.abs().amax(dim=-1, keepdim=True) / 127.0
    c_kv_int8 = (c_kv / scale).round().clamp(-128, 127).to(torch.int8)
    return c_kv_int8, scale.to(torch.float16)

def dequantize_kv_int8(c_kv_int8, scale):
    return c_kv_int8.float() * scale
```

**Phase 4.2: FP8 E4M3 with block-wise scales (Week 6)**

Follow DeepSeek's `kernel.py`:
```python
def act_quant(x, block_size=128):
    """Activation quantization to FP8 E4M3"""
    shape = x.shape
    x = x.view(-1, block_size)
    scale = x.abs().amax(dim=-1, keepdim=True) / 448.0  # E4M3 max
    x_fp8 = (x / scale).to(torch.float8_e4m3fn)
    return x_fp8.view(shape), scale.view(shape[:-1] + (-1,))
```

**Phase 4.3: Fused dequant-attention kernel (Week 7)**

The key optimization: dequantize during matmul, not before. This avoids materializing the full BF16 kv_cache:
```
# Instead of: dequant(kv_cache_fp8) → BF16 buffer → matmul
# Do:         matmul with FP8 inputs → dequant fused into accumulation
```

This requires FP8 tensor core support (H100/H800 native, H20 via software).

---

## Strategy 5: Speculative Decoding

### Verified Claims

| # | Claim | Verdict | Source |
|---|-------|---------|--------|
| 5.1 | DeepSeek-V3 has built-in MTP for speculative decoding | **TRUE** | Paper Section 2.3 |
| 5.2 | MTP acceptance rate 85-90% | **PARTIALLY TRUE** | ISCA paper (arXiv:2505.09343) says **80-90%**, not 85-90% |
| 5.3 | 1.8× TPS increase | **TRUE** | ISCA paper Section 2.3.3 |
| 5.4 | SGLang NEXTN flags | **TRUE** | All 4 flags verified in SGLang docs, draft model on HuggingFace |
| 5.5 | Rollback = just change end_pos integer | **TRUE** | Follows from static buffer architecture |
| 5.6 | Draft model shares embeddings + output head | **TRUE** | MTP architecture in paper |

### Corrected MTP Stats

- Acceptance rate: **80-90%** for predicting the second subsequent token (not 85-90%)
- Source: arXiv:2505.09343 Section 2.3.3, not the V3 technical report itself
- The V3 report only says MTP "can also be used for speculative decoding"
- SGLang benchmarks show up to **3× throughput** improvement on H20 (batch 1: 17→52 t/s)

### Cache Rollback Design (Verified)

DeepSeek's static buffer approach makes rollback trivial:
```
# After verification finds token at position P was wrong:
# No memory operation needed — just update the integer
end_pos = P  # Rejected positions [P, P+1, ...] are still in buffer but ignored
# Next forward call writes at start_pos = P, overwriting rejected tokens
```

With our current `torch.cat` approach, rollback requires:
```
c_kv = c_kv[:, :P, :]      # Slice — allocates new tensor
k_pe = k_pe[:, :P, :, :]   # Slice — allocates new tensor
```

This works but is less clean. Static buffers are a prerequisite for efficient speculative decoding.

### Implementation Plan

**Step 5.1: Static buffer cache (prerequisite — from Strategy 2)**

Must complete Strategy 2 before speculative decoding.

**Step 5.2: end_pos tracking**

```python
class MLACache:
    def __init__(self, max_batch, max_seq, kv_lora_rank, rope_dim, device):
        self.kv_cache = torch.zeros(max_batch, max_seq, kv_lora_rank, device=device)
        self.pe_cache = torch.zeros(max_batch, max_seq, rope_dim, device=device)
        self.end_pos = 0  # Current valid end position

    def rollback(self, new_end_pos: int):
        """O(1) rollback — just update the integer"""
        assert new_end_pos <= self.end_pos
        self.end_pos = new_end_pos
```

**Step 5.3: Verify-and-rollback API**

```python
def verify_and_accept(self, draft_tokens, draft_positions, verifier_logits):
    """
    Compare draft model predictions with verifier (main model) predictions.
    Accept the longest prefix that matches.
    Roll back cache to the last accepted position.

    Returns: number of accepted tokens, the correct next token
    """
    # Standard speculative decoding acceptance logic
    # (rejection sampling or argmax comparison)
    ...
    self.cache.rollback(accepted_end_pos)
    return n_accepted, correct_next_token
```

**Step 5.4: MTP integration (LATER — Phase 3+)**

NanoSeek's MTP module serves as the draft model. Key architectural requirements:
- MTP shares embeddings and output head with main model (no separate model)
- MTP predicts next 1-2 tokens (conservative, matches DeepSeek's `--speculative-num-steps 2`)
- Cache is local (same GPU), so transfer cost is zero

---

## Strategy 6: Prefill-Decode Disaggregation

### Verified Claims

| # | Claim | Verdict | Source |
|---|-------|---------|--------|
| 6.1 | DeepSeek uses PD disaggregation in production | **TRUE** | Open Source Week Day 6 |
| 6.2 | Prefill: EP32, DP32, 4 nodes, 9 routed + 1 shared per GPU | **TRUE** | Exact match from Day 6 disclosure |
| 6.3 | Decode: EP144, DP144, 18 nodes, 2 routed + 1 shared per GPU | **TRUE** | Exact match from Day 6 disclosure |
| 6.4 | KV transfer: 140 MB for 2K tokens, 61 layers (MLA) vs 8 GB (MHA) | **TRUE** | Arithmetic verified |
| 6.5 | 400 Gbps InfiniBand: 2.8 ms transfer time | **TRUE** | 140 MB / 50 GB/s = 2.8 ms |
| 6.6 | SGLang uses NIXL or Mooncake | **TRUE** | Both integrated, documented in SGLang PD disaggregation docs |

### Production Infrastructure Summary (Verified from Day 6)

```
DeepSeek V3/R1 Production (24-hour snapshot, Feb 27-28, 2025):
├── Total nodes: 278 peak, 226.75 average
├── Hardware: H800 GPUs (8 per node)
├── Model: 256 routed experts + 1 shared, 8 activated per token
│
├── Prefill Pool:
│   ├── 4 nodes per deployment unit (32 GPUs)
│   ├── EP32 (expert parallelism across all 32 GPUs)
│   ├── DP32 for MLA/shared experts
│   ├── 9 routed experts per GPU + 1 shared (256/32 = 8, +1 redundant)
│   └── Dual micro-batch overlap for compute-communication pipelining
│
├── Decode Pool:
│   ├── 18 nodes per deployment unit (144 GPUs)
│   ├── EP144 (expert parallelism across all 144 GPUs)
│   ├── DP144 for MLA/shared experts
│   ├── 2 routed experts per GPU + 1 shared (256/144 ≈ 1.78, round up)
│   └── 32 redundant experts per deployment unit
│
└── Key Metrics:
    ├── Total input: 608B tokens/day
    ├── On-disk KV cache hit: 342B tokens (56.3%)
    ├── Total output: 168B tokens/day
    └── Claimed profit margin: 545%
```

### Why MLA Enables PD Disaggregation

The key insight: MLA's compressed KV cache makes the prefill→decode transfer **57× cheaper**:

```
                    MLA              MHA
Per token/layer:    1,152 bytes      65,536 bytes
2K tokens, 61 L:    140 MB           8 GB
Transfer at IB:     2.8 ms           160 ms
```

For standard MHA, the 160 ms transfer time is comparable to or exceeds the decode step time, making PD disaggregation impractical. For MLA, the 2.8 ms transfer is negligible.

### Relevance to NanoSeek

**Low priority for NanoSeek-1B**. PD disaggregation is a serving infrastructure concern, not a model architecture concern. Our model code needs:
1. A clean API that separates prefill and decode paths (already natural with the two-mode forward)
2. KV cache serialization (just `torch.save` / `torch.load` on the static buffers)
3. The DP attention concept (relevant if we serve on multi-GPU)

---

## Strategy 7: Softmax Scale with YaRN mscale

### Verified Claims

| # | Claim | Verdict | Source |
|---|-------|---------|--------|
| 7.1 | `mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0` | **TRUE** | Exact match in `inference/model.py` |
| 7.2 | `softmax_scale = softmax_scale * mscale * mscale` (SQUARED) | **TRUE** | Exact match |
| 7.3 | Only applies when `max_seq_len > original_seq_len` | **TRUE** | Conditional in code |
| 7.4 | Config: `rope_factor=40, mscale=1.0, original_seq_len=4096` | **TRUE** | Default values in ModelArgs, not overridden by config_671B.json |
| 7.5 | At 128K: mscale = 1.369, softmax_scale = 0.1353 | **TRUE** | `0.1 * 1.0 * ln(40) + 1.0 = 0.1 * 3.689 + 1.0 = 1.369`; `(1/√192) * 1.369² = 0.0722 * 1.874 = 0.1353` |

### Critical Bug in Our Implementation

**Current code** (model.py:222-226):
```python
self.base_scale = 1.0 / math.sqrt(self.qk_head_dim)
self.mscale = mscale
self.softmax_scale = self.base_scale * self.mscale  # ← WRONG: multiplied ONCE
```

**Also at line 387**:
```python
effective_scale = self.base_scale * self.mscale  # ← WRONG: multiplied ONCE
```

**DeepSeek's code**:
```python
self.softmax_scale = self.qk_head_dim ** -0.5
if args.max_seq_len > args.original_seq_len:
    mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
    self.softmax_scale = self.softmax_scale * mscale * mscale  # ← CORRECT: SQUARED
```

### Why Squared?

The mscale compensates for attention score distribution changes at extended context. Conceptually:
```
score = (Q * mscale) @ (K * mscale)^T / sqrt(d)
      = Q @ K^T * mscale² / sqrt(d)
```

Both Q and K are affected by the extended context, so the scale correction applies to both sides of the dot product, hence squared.

### Additional Issue: Our mscale Formula

Our config wires `self.yarn.mscale` directly as the mscale value. But DeepSeek computes mscale at runtime from config parameters:
```python
mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
```

We need to verify our `YaRNConfig.mscale` matches this formula. If `YaRNConfig.mscale = 1.0` (the base parameter), then we need to compute the effective mscale using the formula above.

### Implementation Plan

**Step 7.1: Fix mscale application (HIGHEST PRIORITY — correctness bug)**

In `__init__`:
```python
self.base_scale = 1.0 / math.sqrt(self.qk_head_dim)
self.mscale_param = mscale  # Store the raw config parameter

# Compute effective mscale following DeepSeek's formula
# Only apply when actually extending beyond training context
self._compute_softmax_scale(max_position_embeddings, original_max_position_embeddings, rope_scaling_factor)
```

New method:
```python
def _compute_softmax_scale(self, max_seq_len, original_seq_len, rope_factor):
    self.softmax_scale = self.base_scale
    if max_seq_len > original_seq_len and rope_factor > 1.0:
        mscale = 0.1 * self.mscale_param * math.log(rope_factor) + 1.0
        self.softmax_scale = self.base_scale * mscale * mscale
    # If within training window, mscale = 1.0 (no adjustment)
```

**Step 7.2: Verify NanoSeek-1B values**

For Phase 1 (4K context, no YaRN):
- `max_seq_len = 4096`, `original_seq_len = 4096`
- Condition: `4096 > 4096` is FALSE → no mscale → `softmax_scale = 1/√96 = 0.1021`

For Phase 2 (8K context, YaRN enabled):
- `rope_factor` for 2× extension... need to check our config
- If `rope_factor = 2.0`: `mscale = 0.1 * 1.0 * ln(2) + 1.0 = 1.069`
- `softmax_scale = 0.1021 * 1.069² = 0.1021 * 1.143 = 0.1167`
- Small adjustment — but must be squared, not linear

---

## Strategy 8: Dual Micro-Batch Overlap

### Verified Claims

| # | Claim | Verdict | Source |
|---|-------|---------|--------|
| 8.1 | Paper is real: arXiv:2505.09343 (ISCA 2025) | **TRUE** | Published May 14, 2025, presented at ISCA |
| 8.2 | Pipeline description with MLA/MoE decoupling | **TRUE** | Section 2.3.1 |
| 8.3 | Hides all-to-all communication behind computation | **TRUE** | Alternating μ-batches overlap compute and comm |

### Pipeline Details (Verified)

```
Time →
μ-batch A: [MLA compute] [MoE dispatch all2all] [MoE compute] [MoE combine all2all]
μ-batch B:               [MLA compute]          [MoE dispatch] [MoE compute]          ...
```

Key insight: While GPU computes MLA attention for μ-batch B (reading kv_cache from HBM), the NIC simultaneously sends expert routing tokens for μ-batch A's MoE dispatch (network bandwidth). HBM bandwidth and NIC bandwidth are independent resources, so both are utilized simultaneously.

### Relevance to NanoSeek

**Not applicable for model implementation**. This is a serving framework concern:
- Requires multi-node expert parallelism (NanoSeek-1B trains on single node)
- Requires all-to-all communication for MoE dispatch (NanoSeek uses local MoE)
- Would only be relevant if we built a production serving system

**Documented for reference only** — useful context for understanding DeepSeek's production architecture.

---

## Strategy 9: On-Disk KV Cache

### Verified Claims

| # | Claim | Verdict | Source |
|---|-------|---------|--------|
| 9.1 | 608B input tokens, 342B (56.3%) hit on-disk cache | **TRUE** | DeepSeek's official X/Twitter post, Feb 27-28, 2025 |
| 9.2 | MLA: 140 MB on disk for 2K tokens, 61 layers | **TRUE** | 2000 × 61 × 1152 = 140.5 MB |
| 9.3 | MHA: 8 GB on disk for same | **TRUE** | 2000 × 61 × 65536 = 7.99 GB |
| 9.4 | SSD read: MLA = 20 ms, MHA = 1.14 s | **TRUE** | At 7 GB/s: 140 MB / 7 GB/s = 20 ms |

### Economic Impact

56.3% cache hit rate means more than half of input tokens skip prefill entirely. This is enabled by:
1. Common system prompts across requests (cached and reused)
2. SGLang's prefix-cache aware routing (routes requests to GPUs that already have the prefix cached)
3. MLA's 57× smaller cache makes disk storage practical (10 GB SSD holds ~71K prompts worth of system prompt cache vs ~1.25K for MHA)

### Relevance to NanoSeek

**Low priority for NanoSeek-1B training/eval**. On-disk caching is a serving optimization. However, the concept is useful for:
- Caching evaluation prompts during RL post-training (avoid re-prefilling same prompts)
- Benchmark evaluation where same prefix is used across many completions

---

## Gap Analysis

### Our Code vs DeepSeek's Inference Code

| Component | DeepSeek Reference | Our Code | Status |
|-----------|-------------------|----------|--------|
| Absorption mode default | `"absorb"` | Training mode only | **INCOMPLETE** — `_forward_inference()` is stub |
| Runtime absorption | einsum with conditional FP8 dequant | Pre-extracted in `prepare_for_inference()` | **WORKS** but different pattern |
| KV cache | Static buffers, `start_pos` integer | Dynamic `torch.cat`, tuple cache | **NEEDS REWRITE** for inference |
| mscale | Squared, conditional on context extension | Applied once (not squared) | **BUG** — fix immediately |
| mscale formula | `0.1 * mscale * log(factor) + 1.0` | Raw config value passed through | **NEEDS VERIFICATION** |
| TP-aware heads | `n_local_heads = n_heads / tp_size` | `num_heads` (no TP support) | **MISSING** |
| FP8 weight dequant | Conditional `weight_dequant()` | Not implemented | **PHASE 2** |
| RoPE cache | Pre-rotated k_pe in static buffer | Pre-rotated k_pe in tuple | **CORRECT** approach |
| Causal mask | Not used (single-token decode) | Created when needed | **CORRECT** |
| SDPA/FlashAttention | Not used in reference (einsum) | Used in training path | **CORRECT** for training |

### Critical Bugs to Fix (Ordered by Priority)

1. **mscale squared** — correctness bug affecting all context-extended inference
2. **`_forward_inference()` stub** — blocking all inference testing
3. **mscale formula** — verify our config computes effective mscale correctly

### Missing Features (Ordered by Priority)

1. Static KV cache buffers — prerequisite for 4+ other features
2. Runtime weight absorption (matching DeepSeek's exact pattern)
3. torch.compile integration
4. FP8 weight/cache support
5. TP-aware head count
6. Speculative decoding rollback API
7. Triton MLA decode kernel

---

## Implementation Priority

### Phase 1: Core Correctness (Week 1)

**Goal**: Match DeepSeek's exact behavior, all arithmetic correct.

| # | Task | Priority | Prerequisite | Estimated Effort |
|---|------|----------|-------------|-----------------|
| 1 | Fix mscale to be squared + match formula | **P0** | None | 1 hour |
| 2 | Implement `_forward_inference()` with absorbed attention | **P0** | None | 4 hours |
| 3 | Switch inference cache to static pre-allocated buffers | **P0** | Task 2 | 4 hours |
| 4 | Add `start_pos` integer tracking | **P0** | Task 3 | 1 hour |
| 5 | Numerical equivalence test: training path == inference path | **P0** | Tasks 2-4 | 2 hours |
| 6 | Add `InferenceConfig` to config.py | **P1** | None | 1 hour |

**Verification gate**: For identical inputs, training-path output and inference-path output must match within `atol=1e-5, rtol=1e-4`.

### Phase 2: Production Efficiency (Weeks 2-3)

| # | Task | Priority | Prerequisite | Estimated Effort |
|---|------|----------|-------------|-----------------|
| 7 | `torch.compile` on inference path | **P1** | Phase 1 complete | 2 hours |
| 8 | FP8 weight dequant (match `kernel.py`) | **P2** | Phase 1 complete | 1 day |
| 9 | FP8 KV cache (INT8 first, then FP8 E4M3) | **P2** | Phase 1 complete | 2 days |
| 10 | TP-aware `n_local_heads` | **P2** | Phase 1 complete | 4 hours |
| 11 | Benchmark: tokens/sec on single GPU | **P1** | Tasks 7+ | 4 hours |

### Phase 3: Serving Features (Weeks 3-4)

| # | Task | Priority | Prerequisite | Estimated Effort |
|---|------|----------|-------------|-----------------|
| 12 | Speculative decoding rollback API | **P2** | Phase 1 (static cache) | 1 day |
| 13 | MTP as draft model integration | **P2** | Task 12 + MTP module | 2 days |
| 14 | KV cache serialization for PD transfer | **P3** | Phase 1 | 4 hours |
| 15 | Page table interface design | **P3** | Phase 1 | 1 day |

### Phase 4: Custom Kernels (Weeks 4-6)

| # | Task | Priority | Prerequisite | Estimated Effort |
|---|------|----------|-------------|-----------------|
| 16 | Triton MLA decode kernel | **P2** | Phase 1 + benchmarks showing bottleneck | 3 days |
| 17 | Fused dequant-attention kernel | **P3** | Tasks 9, 16 | 3 days |
| 18 | FlashAttention MLA prefill adaptation | **P3** | Phase 1 | 3 days |

### Phase 5: Scale-out (Month 2+)

| # | Task | Priority | Prerequisite | Estimated Effort |
|---|------|----------|-------------|-----------------|
| 19 | ColumnParallelLinear / RowParallelLinear | **P3** | All above | 2 days |
| 20 | DP attention mode | **P3** | Task 19 | 2 days |
| 21 | On-disk KV cache with prefix routing | **P4** | Phase 3 | 3 days |

---

## Appendix A: Numerical Verification Checklist

Before any production deployment, verify these numerical properties:

```
□ mscale = 1.0 when max_seq_len <= original_seq_len (no context extension)
□ mscale = 0.1 * config_mscale * log(rope_factor) + 1.0 when extending
□ softmax_scale = (1/sqrt(qk_head_dim)) * mscale² (SQUARED)
□ Training forward == Inference forward (within float tolerance)
□ Absorbed Q @ c_kv^T == Q_nope @ K_nope^T (mathematical equivalence)
□ Absorbed attn @ c_kv @ W_UV == attn @ V (mathematical equivalence)
□ Static buffer cache produces identical outputs to torch.cat cache
□ Speculative decoding rollback produces identical state to non-speculative generation
□ FP8 quantized cache produces acceptable quality (perplexity within 0.5% of BF16)
```

## Appendix B: Key DeepSeek-V3 Configuration Values (Verified)

```python
# From deepseek-ai/DeepSeek-V3/inference/configs/config_671B.json
# and inference/model.py ModelArgs defaults

hidden_size = 7168          # d_model
num_layers = 61             # transformer blocks
num_attention_heads = 128   # n_heads
kv_lora_rank = 512          # compressed KV dimension
q_lora_rank = 1536          # compressed Q dimension
qk_nope_head_dim = 128      # non-RoPE component per head
qk_rope_head_dim = 64       # RoPE component per head (SHARED)
v_head_dim = 128            # value dimension per head

# RoPE / YaRN
rope_theta = 10000.0
rope_factor = 40.0          # YaRN scaling factor
mscale = 1.0                # Base mscale parameter
original_seq_len = 4096     # Training context length

# MoE
num_experts = 256           # routed experts
num_experts_per_tok = 8     # top-k selection
num_shared_experts = 1      # always-active expert

# Derived
qk_head_dim = 128 + 64 = 192  # total query/key dimension
kv_cache_per_token = 512 + 64 = 576  # total cached per token
```

## Appendix C: Source References

1. **DeepSeek-V3 inference code**: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
2. **DeepSeek-V3 technical report**: arXiv:2412.19437
3. **ISCA hardware paper**: arXiv:2505.09343 — "Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures"
4. **DeepSeek Open Source Week Day 6**: https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md
5. **FlashMLA**: https://github.com/deepseek-ai/FlashMLA
6. **SGLang DeepSeek docs**: https://docs.sglang.io/basic_usage/deepseek_v3.html
7. **SGLang PD disaggregation**: https://docs.sglang.ai/advanced_features/pd_disaggregation.html
8. **SGLang PR #905 (Triton MLA)**: https://github.com/sgl-project/sglang/pull/905
9. **SGLang PR #1970 (DP MLA)**: https://github.com/sgl-project/sglang/pull/1970
10. **SGLang PR #3582 (NEXTN speculative)**: https://github.com/sgl-project/sglang/pull/3582
11. **AITER MLA blog (AMD)**: https://rocm.blogs.amd.com/software-tools-optimization/aiter-mla/README.html
12. **LMSYS GB200 Part II**: https://lmsys.org/blog/2025-09-25-gb200-part-2/
13. **DeepSeek production stats (X/Twitter)**: https://x.com/deepseek_ai/status/1895688300574462431
14. **Mooncake (FAST 2025)**: https://github.com/kvcache-ai/Mooncake
15. **NVIDIA NIXL**: https://developer.nvidia.com/blog/enhancing-distributed-inference-performance-with-the-nvidia-inference-transfer-library/


