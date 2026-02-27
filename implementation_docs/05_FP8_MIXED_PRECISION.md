# FP8 Mixed Precision Training & Inference Quantization for NanoSeek

## Principal Engineer's Implementation Guide — February 2026

**Author**: Distinguished AI Researcher & Principal Engineer, Foundation Models Division  
**Scope**: FP8 training on H100/H200, dynamic loss scaling, INT4/INT8 inference quantization  
**Prerequisites**: `model/config.py` (FP8Config dataclass), `model/model.py` (BF16 training baseline)

---

## 1. Problem Statement

NanoSeek trains in BF16 today. Every parameter, activation, and gradient occupies 2 bytes. On H100, the FP8 Tensor Cores sit completely idle — half the silicon is dark.

**Current Memory Budget (BF16, per GPU with FSDP FULL_SHARD on 8×H100):**

| Component | BF16 Size | FP8 Target | Savings |
|-----------|-----------|------------|---------|
| Model weights | 4.75B × 2B = 9.5 GB | 4.75B × 1B = 4.75 GB | **2×** |
| Activations (est.) | ~15 GB | ~7.5 GB | **2×** |
| KV cache (inference) | MLA 175 dims × 2B | 175 dims × 1B | **2×** |
| Gradients | 9.5 GB | 9.5 GB (E5M2) | ~1× |
| Optimizer states | 57 GB (FP32, sharded) | 57 GB (unchanged) | — |

**H100 Tensor Core Throughput:**

| Precision | TFLOPS (dense) | Relative |
|-----------|---------------:|:--------:|
| BF16 | 1,979 | 1× (current) |
| **FP8** | **3,958** | **2× (target)** |

The theoretical 2× throughput gain is reduced to 1.5-2× in practice because training is not 100% compute-bound — memory bandwidth, communication, and Python overhead consume ~25-40% of wall time. DeepSeek V3 reports 1.4× measured speedup on their 2048-GPU cluster; at NanoSeek's 8-GPU scale the ratio is better (less communication overhead).

**For NanoSeek specifically:**
- Model weights: 9.5 GB → 4.75 GB, fits on single GPU with room for larger batches
- Training: ~14h → ~8-10h on 8×H100, saving ~$100-150 per run
- Inference: FP8 weights + FP8 KV cache = 4× memory reduction vs BF16
- Post-training INT4: 8× reduction from BF16, enables single-GPU serving

---

## 2. First Principles: FP8 Format & Why It Works

### E4M3 vs E5M2: The Fundamental Tradeoff

```
E4M3 (4-bit exponent, 3-bit mantissa):
┌───┬────────┬─────────┐
│ S │  EEEE  │   MMM   │     Range: ±448,  Precision: 1/8 ULP
└───┴────────┴─────────┘     Use: Forward pass (weights + activations)

E5M2 (5-bit exponent, 2-bit mantissa):
┌───┬─────────┬────────┐
│ S │  EEEEE  │   MM   │     Range: ±57344, Precision: 1/4 ULP
└───┴─────────┴────────┘     Use: Backward pass (gradients)
```

**Why two formats?** Gradients span a wider dynamic range than weights/activations. E5M2's extra exponent bit gives 16× more range at the cost of halving precision — acceptable for gradients where the SGD noise floor already limits useful precision to ~8 bits.

**Why FP8 works for transformers specifically:**
1. **Outlier-friendly activations**: Post-LayerNorm activations follow ~log-normal distributions. Block-wise scaling handles heavy tails.
2. **Concentrated weight distributions**: Trained transformer weights cluster around zero (σ ≈ 0.02). E4M3's precision near zero is sufficient.
3. **MoE amplifies the benefit**: 64 expert weight matrices dominate NanoSeek's memory. FP8 halves this from 4.2 GB to 2.1 GB.
4. **MLA's low-rank representations**: The 143-dim KV compressed space has inherently lower effective precision requirements.

### Scaling Strategies

FP8's limited range (±448 for E4M3) requires scaling tensors into the representable range:

```
Per-tensor:   scale = amax(tensor) / FP8_MAX
Per-channel:  scale[c] = amax(tensor[:, c]) / FP8_MAX
Block-wise:   scale[b] = amax(tensor[b*128:(b+1)*128]) / FP8_MAX   ← NanoSeek
```

**NanoSeek uses block-wise scaling with block_size=128** (matching `FP8Config.block_size`), consistent with DeepSeek V3's approach.

### Delayed Scaling

Computing exact `amax` requires a full tensor reduction — expensive on GPU. **Delayed scaling** uses the `amax` from the *previous* iteration:

```
Step t:  scale_t = amax_history[t-1] / FP8_MAX
         tensor_fp8 = quantize(tensor, scale_t)
         amax_history[t] = amax(tensor)      # For step t+1
```

This works because tensor statistics change slowly between iterations (LR ≈ 3e-4). Controlled by `FP8Config.use_delayed_scaling = True`.

---

## 3. Production Code

### File Placement

```
model/kernels/
├── fp8_linear.py          # 5a: FP8 Linear layer with delayed scaling
├── fp8_moe.py             # 5b: FP8 MoE integration
├── quantization.py        # 5c/5e: FP8 KV cache + INT4/INT8 quantization
model/serving/
└── quantize.py            # 5e: Inference quantization pipeline
```

---

### 5a. FP8 Linear Layer (`model/kernels/fp8_linear.py`)

```python
"""
FP8 Linear Layer — H100/H200 Tensor Core acceleration.
Forward: E4M3 GEMM. Backward: E5M2 GEMM. Delayed scaling with amax history.
Falls back to BF16 when FP8 not available.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    HAS_TE = True
except ImportError:
    HAS_TE = False

try:
    E4M3 = torch.float8_e4m3fn
    E5M2 = torch.float8_e5m2
    HAS_FP8_DTYPE = True
except AttributeError:
    HAS_FP8_DTYPE = False


class AmaxTracker:
    """Rolling amax history for delayed FP8 scaling."""
    def __init__(self, history_len: int = 16, fp8_max: float = 448.0):
        self.history_len = history_len
        self.fp8_max = fp8_max
        self.amax_history = torch.zeros(history_len)
        self._step = 0

    def update(self, tensor: Tensor) -> None:
        with torch.no_grad():
            idx = self._step % self.history_len
            self.amax_history[idx] = tensor.abs().max().float().cpu()
            self._step += 1

    def get_scale(self) -> Tensor:
        if self._step == 0:
            return torch.tensor(1.0)
        amax = self.amax_history[:min(self._step, self.history_len)].max()
        return (self.fp8_max / amax.clamp(min=1e-12)).float()

    def to(self, device): self.amax_history = self.amax_history.to(device); return self


def quantize_to_fp8(tensor, scale, fp8_dtype=None, block_size=128):
    """Block-wise FP8 quantization. Returns (quantized, per_block_scales)."""
    if fp8_dtype is None:
        fp8_dtype = E4M3 if HAS_FP8_DTYPE else torch.bfloat16
    fp8_max = torch.finfo(fp8_dtype).max if HAS_FP8_DTYPE else 448.0

    original_shape = tensor.shape
    flat = tensor.reshape(-1)
    pad_len = (block_size - flat.numel() % block_size) % block_size
    if pad_len > 0:
        flat = F.pad(flat, (0, pad_len))

    blocks = flat.view(-1, block_size)
    block_scales = fp8_max / blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    quantized = (blocks * block_scales).clamp(-fp8_max, fp8_max)
    quantized = quantized.to(fp8_dtype if HAS_FP8_DTYPE else torch.bfloat16)
    return quantized.view(-1)[:tensor.numel()].view(original_shape), block_scales.squeeze(-1)


class FP8Linear(nn.Module):
    """
    Drop-in nn.Linear replacement with FP8 Tensor Core acceleration.
    Uses Transformer Engine on H100, PyTorch native FP8 as fallback,
    or plain BF16 matmul on non-FP8 hardware.
    """
    def __init__(self, in_features, out_features, bias=False, block_size=128,
                 use_delayed_scaling=True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.block_size = block_size
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        self.input_amax = AmaxTracker(fp8_max=448.0)
        self.weight_amax = AmaxTracker(fp8_max=448.0)
        self._use_te = HAS_TE
        if HAS_TE:
            self._te_linear = te.Linear(in_features, out_features, bias=bias is not False)
            self._te_linear.weight = self.weight

    def forward(self, x):
        if self._use_te and self.training:
            return self._forward_te(x)
        elif HAS_FP8_DTYPE:
            return self._forward_fp8_native(x)
        return F.linear(x, self.weight, self.bias)

    def _forward_te(self, x):
        recipe = DelayedScaling(margin=0, interval=1, fp8_format=Format.HYBRID,
                                amax_history_len=16, amax_compute_algo="max")
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            return self._te_linear(x)

    def _forward_fp8_native(self, x):
        in_scale = self.input_amax.get_scale().to(x.device)
        w_scale = self.weight_amax.get_scale().to(x.device)
        x_fp8 = (x.float() * in_scale).clamp(-448, 448).to(E4M3)
        w_fp8 = (self.weight.float() * w_scale).clamp(-448, 448).to(E4M3)
        out = torch._scaled_mm(
            x_fp8.reshape(-1, x.shape[-1]), w_fp8.t(),
            scale_a=torch.tensor(1.0/in_scale.item(), device=x.device),
            scale_b=torch.tensor(1.0/w_scale.item(), device=x.device),
            out_dtype=torch.bfloat16,
        ).view(*x.shape[:-1], self.out_features)
        if self.training:
            self.input_amax.update(x)
            self.weight_amax.update(self.weight.data)
        return out + self.bias if self.bias is not None else out

    @classmethod
    def from_linear(cls, linear, block_size=128):
        fp8 = cls(linear.in_features, linear.out_features,
                   bias=linear.bias is not None, block_size=block_size)
        fp8.weight = linear.weight
        if linear.bias is not None: fp8.bias = linear.bias
        return fp8


class DynamicLossScaler:
    """
    Auto loss scaling for FP8 training. Detects gradient overflow → halves scale.
    Sustained non-overflow → doubles scale. Integrates with gradient accumulation.
    """
    def __init__(self, init_scale=2**16, scale_factor=2.0, scale_window=2000,
                 min_scale=1.0, max_scale=2**24):
        self.scale = init_scale
        self.scale_factor, self.scale_window = scale_factor, scale_window
        self.min_scale, self.max_scale = min_scale, max_scale
        self._non_overflow_steps = 0
        self._overflow_count = self._total_steps = 0

    def scale_loss(self, loss): return loss * self.scale

    def unscale_grads(self, optimizer):
        """Unscale grads, return True if valid (no inf/nan)."""
        inv_scale, found_inf = 1.0 / self.scale, False
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.data.mul_(inv_scale)
                    if torch.isinf(p.grad).any() or torch.isnan(p.grad).any():
                        found_inf = True
        self._total_steps += 1
        if found_inf:
            self._overflow_count += 1; self._non_overflow_steps = 0
            self.scale = max(self.scale / self.scale_factor, self.min_scale)
            for g in optimizer.param_groups:
                for p in g["params"]:
                    if p.grad is not None: p.grad.data.zero_()
            return False
        self._non_overflow_steps += 1
        if self._non_overflow_steps >= self.scale_window:
            self.scale = min(self.scale * self.scale_factor, self.max_scale)
            self._non_overflow_steps = 0
        return True

    def state_dict(self):
        return {"scale": self.scale, "non_overflow_steps": self._non_overflow_steps,
                "overflow_count": self._overflow_count, "total_steps": self._total_steps}

    def load_state_dict(self, s):
        self.scale, self._non_overflow_steps = s["scale"], s["non_overflow_steps"]
        self._overflow_count, self._total_steps = s["overflow_count"], s["total_steps"]
```

---

### 5b. FP8 MoE Integration (`model/kernels/fp8_moe.py`)

```python
"""
FP8 MoE for NanoSeek — FP8 expert weights & grouped GEMM.

Precision hierarchy:
  Router/Gate: FP32 (routing decisions must be precise)
  RMSNorm:     FP32 (normalization statistics)
  Experts:     FP8  (E4M3 — the big memory win, 64 experts)
  Accumulate:  FP32 (internal to Tensor Core)
  Output:      BF16 (after dequantize)

Memory: 14 MoE layers × 64 experts → BF16: 8.4 GB, FP8: 4.2 GB (saves 4.2 GB)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple

try:
    from .fp8_linear import FP8Linear, HAS_FP8_DTYPE
except ImportError:
    from fp8_linear import FP8Linear, HAS_FP8_DTYPE


class FP8SwiGLUExpert(nn.Module):
    """Single SwiGLU expert with FP8 projections."""
    def __init__(self, dim, inter_dim, block_size=128):
        super().__init__()
        self.gate_proj = FP8Linear(dim, inter_dim, block_size=block_size)
        self.up_proj = FP8Linear(dim, inter_dim, block_size=block_size)
        self.down_proj = FP8Linear(inter_dim, dim, block_size=block_size)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    @classmethod
    def from_swiglu(cls, expert, block_size=128):
        fp8 = cls(expert.gate_proj.in_features, expert.gate_proj.out_features, block_size)
        fp8.gate_proj.weight, fp8.up_proj.weight = expert.gate_proj.weight, expert.up_proj.weight
        fp8.down_proj.weight = expert.down_proj.weight
        return fp8


class FP8MoE(nn.Module):
    """
    FP8-accelerated Mixture of Experts. Gate stays FP32, experts use FP8.
    Drop-in replacement for model.model.MoE.
    """
    def __init__(self, dim, moe_inter_dim, n_routed_experts, n_activated_experts,
                 n_shared_experts=0, n_expert_groups=1, n_limited_groups=1,
                 score_func="sigmoid", route_scale=1.0, seq_aux_loss_alpha=0.0001,
                 block_size=128):
        super().__init__()
        self.dim, self.n_routed_experts = dim, n_routed_experts
        self.n_activated_experts, self.n_shared_experts = n_activated_experts, n_shared_experts
        self.seq_aux_loss_alpha = seq_aux_loss_alpha

        from model.model import Gate
        self.gate = Gate(dim, n_routed_experts, n_activated_experts, n_expert_groups,
                         n_limited_groups, score_func, route_scale)
        self.experts = nn.ModuleList([
            FP8SwiGLUExpert(dim, moe_inter_dim, block_size) for _ in range(n_routed_experts)])
        self.shared_experts = nn.ModuleList([
            FP8SwiGLUExpert(dim, moe_inter_dim, block_size) for _ in range(n_shared_experts)
        ]) if n_shared_experts > 0 else None

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict]:
        B, S, D = x.shape
        x_flat = x.view(B * S, D)

        shared_out = 0
        if self.shared_experts:
            shared_out = sum(e(x_flat) for e in self.shared_experts) if len(
                self.shared_experts) > 1 else self.shared_experts[0](x_flat)

        with torch.cuda.amp.autocast(enabled=False):
            weights, indices = self.gate(x_flat.float())

        from model.model import token_centric_dispatch
        routed = token_centric_dispatch(x_flat, indices, weights, self.experts)
        output = (routed + shared_out).view(B, S, D)

        aux = {}
        if self.training and self.seq_aux_loss_alpha > 0:
            load = self.gate.expert_load
            target = B * S * self.n_activated_experts / self.n_routed_experts
            aux["seq_aux_loss"] = self.seq_aux_loss_alpha * ((load - target)**2).mean()
            aux["expert_load"] = load.clone()
        return output, aux

    def update_load_balance_bias(self, gamma=0.001):
        with torch.no_grad():
            load = self.gate.expert_load
            mean = load.mean()
            if mean > 0:
                self.gate.expert_bias.sub_(gamma * (load - mean) / (mean + 1e-8))

    @classmethod
    def from_moe(cls, moe, block_size=128):
        """Convert standard MoE → FP8MoE, preserving weights."""
        fp8 = cls(moe.dim, moe.experts[0].gate_proj.out_features, moe.n_routed_experts,
                  moe.n_activated_experts, moe.n_shared_experts,
                  seq_aux_loss_alpha=moe.seq_aux_loss_alpha, block_size=block_size)
        fp8.gate = moe.gate
        for i, e in enumerate(moe.experts):
            fp8.experts[i] = FP8SwiGLUExpert.from_swiglu(e, block_size)
        if moe.shared_experts:
            for i, e in enumerate(moe.shared_experts):
                fp8.shared_experts[i] = FP8SwiGLUExpert.from_swiglu(e, block_size)
        return fp8
```

---

### 5c. FP8 KV Cache for Inference (`model/kernels/quantization.py`)

```python
"""
FP8 KV cache + INT4/INT8 weight quantization for NanoSeek inference.

FP8 KV cache on top of MLA's 23× compression:
  Standard MHA:  4096 dims × 2B = 8192 B/token/layer
  MLA:            175 dims × 2B =  350 B/token/layer (23×)
  MLA + FP8:      175 dims × 1B =  175 B/token/layer (47×)
"""
import math, torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional, Tuple

HAS_FP8 = hasattr(torch, "float8_e4m3fn")
E4M3 = torch.float8_e4m3fn if HAS_FP8 else None


class FP8KVCache:
    """
    FP8 KV cache for MLA inference with per-position scaling.
    Stores kv_compressed in FP8, k_pe in BF16 (already small at 32 dims).
    Dequantizes on-the-fly during attention.
    """
    def __init__(self, num_layers, max_seq_len, kv_lora_rank, rope_dim,
                 num_heads=16, device=None):
        self.num_layers, self.max_seq_len = num_layers, max_seq_len
        self.kv_lora_rank, self.rope_dim = kv_lora_rank, rope_dim
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = E4M3 if HAS_FP8 else torch.bfloat16

        self._seq_len = [0] * num_layers
        self.kv_cache = [torch.zeros(1, max_seq_len, kv_lora_rank, dtype=dtype, device=dev)
                         for _ in range(num_layers)]
        self.kpe_cache = [torch.zeros(1, max_seq_len, 1, rope_dim, dtype=torch.bfloat16, device=dev)
                          for _ in range(num_layers)]
        self.kv_scales = [torch.ones(1, max_seq_len, 1, device=dev) for _ in range(num_layers)]

    def update(self, layer_idx, kv_compressed, k_pe):
        """Append new entries, return full dequantized (kv_bf16, kpe_bf16)."""
        B, L, _ = kv_compressed.shape
        start, end = self._seq_len[layer_idx], self._seq_len[layer_idx] + L

        amax = kv_compressed.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        fp8_max = torch.finfo(E4M3).max if HAS_FP8 else 448.0
        scale = fp8_max / amax
        quantized = (kv_compressed * scale).clamp(-fp8_max, fp8_max)
        quantized = quantized.to(E4M3 if HAS_FP8 else torch.bfloat16)

        self.kv_cache[layer_idx][:B, start:end] = quantized
        self.kv_scales[layer_idx][:B, start:end] = scale
        self.kpe_cache[layer_idx][:B, start:end] = k_pe
        self._seq_len[layer_idx] = end

        cached = self.kv_cache[layer_idx][:B, :end]
        scales = self.kv_scales[layer_idx][:B, :end]
        return (cached.float() / scales).to(torch.bfloat16), self.kpe_cache[layer_idx][:B, :end]

    def reset(self):
        for i in range(self.num_layers):
            self._seq_len[i] = 0; self.kv_cache[i].zero_(); self.kv_scales[i].fill_(1.0)


class QuantizedLinear(nn.Module):
    """
    INT4/INT8 quantized linear for inference. Group-wise scaling with AWQ-style
    activation-aware quantization support.
    """
    def __init__(self, in_features, out_features, bits=4, group_size=128, bias=False):
        super().__init__()
        assert bits in (4, 8)
        self.in_features, self.out_features = in_features, out_features
        self.bits, self.group_size = bits, group_size
        self.n_groups = math.ceil(in_features / group_size)

        packed_in = math.ceil(in_features / 2) if bits == 4 else in_features
        self.register_buffer("weight_packed", torch.zeros(out_features, packed_in, dtype=torch.int8))
        self.register_buffer("scales", torch.ones(out_features, self.n_groups, dtype=torch.float16))
        self.register_buffer("zeros", torch.zeros(out_features, self.n_groups, dtype=torch.float16))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        return F.linear(x, self._dequantize().to(x.dtype), self.bias)

    def _dequantize(self):
        if self.bits == 4:
            high = (self.weight_packed >> 4) & 0x0F
            low = self.weight_packed & 0x0F
            w = torch.cat([low, high], dim=-1)[:, :self.in_features].float() - 8
        else:
            w = self.weight_packed.float()
        out = torch.zeros(self.out_features, self.in_features, device=w.device, dtype=torch.float16)
        for g in range(self.n_groups):
            s, e = g * self.group_size, min((g+1) * self.group_size, self.in_features)
            out[:, s:e] = (w[:, s:e] - self.zeros[:, g:g+1]) * self.scales[:, g:g+1]
        return out

    @classmethod
    def from_linear(cls, linear, bits=4, group_size=128, act_scales=None):
        """Quantize nn.Linear to INT4/INT8 with optional AWQ activation awareness."""
        q = cls(linear.in_features, linear.out_features, bits, group_size,
                bias=linear.bias is not None)
        weight = linear.weight.data.float()
        if act_scales is not None:
            threshold = act_scales.to(weight.device).float().quantile(0.99)
            boost = (act_scales.float() / threshold).clamp(min=1.0)
            weight = weight * boost.unsqueeze(0)

        qmin, qmax = (0, 15) if bits == 4 else (-128, 127)
        for g in range(q.n_groups):
            s, e = g * group_size, min((g+1) * group_size, linear.in_features)
            gw = weight[:, s:e]
            scale = ((gw.amax(dim=-1, keepdim=True) - gw.amin(dim=-1, keepdim=True))
                     / (qmax - qmin)).clamp(min=1e-8)
            zero = qmin - gw.amin(dim=-1, keepdim=True) / scale
            q.scales[:, g], q.zeros[:, g] = scale.squeeze().half(), zero.squeeze().half()
            quantized = (gw / scale + zero).round().clamp(qmin, qmax).to(torch.int8)
            if bits == 4:
                ps = s // 2
                q.weight_packed[:, ps:ps+(e-s)//2] = ((quantized[:, 1::2] << 4)
                                                       | (quantized[:, 0::2] & 0x0F)).to(torch.int8)
            else:
                q.weight_packed[:, s:e] = quantized
        if linear.bias is not None: q.bias = nn.Parameter(linear.bias.data.clone())
        return q
```

---

### 5e. Inference Quantization Pipeline (`model/serving/quantize.py`)

```python
"""
Post-training quantization pipeline for NanoSeek.
BF16 (9.5 GB) → INT8 (4.75 GB, 2×) → INT4 (2.4 GB, 4×).
"""
import gc, torch, torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional
from model.kernels.quantization import QuantizedLinear


def collect_activation_scales(model, calibration_data, device=None):
    """Record per-channel activation magnitudes across calibration set for AWQ."""
    device = device or next(model.parameters()).device
    scales, hooks = {}, []
    def make_hook(name):
        def hook(mod, inp, out):
            x = inp[0].view(-1, inp[0].shape[-1]) if inp[0].dim() == 3 else inp[0]
            s = x.abs().mean(dim=0)
            scales[name] = torch.max(scales[name], s) if name in scales else s
        return hook
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear): hooks.append(m.register_forward_hook(make_hook(n)))
    model.eval()
    with torch.no_grad():
        for b in calibration_data: model(b.to(device))
    for h in hooks: h.remove()
    return scales


def quantize_model(model, bits=4, group_size=128, calibration_data=None,
                   skip_modules=None):
    """
    Quantize all eligible Linear layers. Skips embeddings, norms, gates.
    Uses AWQ-style activation-aware quantization when calibration data provided.
    """
    skip = skip_modules or ["embed_tokens", "lm_head", "gate.weight", "norm"]
    act_scales = collect_activation_scales(model, calibration_data) if calibration_data else None
    count = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear) or any(s in name for s in skip): continue
        parent_name, child = ".".join(name.split(".")[:-1]), name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, QuantizedLinear.from_linear(
            mod, bits, group_size, act_scales.get(name) if act_scales else None))
        count += 1
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"Quantized {count} layers to INT{bits}")
    return model
```

---

## 4. Model Conversion Utilities

```python
def convert_to_fp8_training(model, config):
    """Replace MoE layers with FP8 variants for training."""
    from model.kernels.fp8_moe import FP8MoE
    for layer in model.layers:
        if layer.is_moe_layer:
            layer.ffn = FP8MoE.from_moe(layer.ffn, block_size=config.fp8.block_size)
    return model

def convert_to_int4_inference(model, calibration_data=None):
    """Full INT4 quantization for serving."""
    from model.serving.quantize import quantize_model
    return quantize_model(model, bits=4, group_size=128, calibration_data=calibration_data)
```

**Training loop integration:**

```python
from model.kernels.fp8_linear import DynamicLossScaler
scaler = DynamicLossScaler()

for step, batch in enumerate(dataloader):
    loss = model(batch["input_ids"], labels=batch["labels"])["loss"]
    scaler.scale_loss(loss).backward()
    if (step + 1) % grad_accum == 0:
        if scaler.unscale_grads(optimizer):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        optimizer.zero_grad()
```

---

## 5. Performance Targets

### Training (BF16 → FP8)

| Metric | BF16 Baseline | FP8 Target | Improvement |
|--------|:------------:|:----------:|:-----------:|
| Wall time (22B tokens, 8×H100) | ~14h | ~8-10h | **1.5×** |
| Weight memory | 9.5 GB | 4.75 GB | **2×** |
| Activation memory | ~15 GB | ~7.5 GB | **2×** |
| Peak GPU memory | ~70 GB | ~45 GB | **1.6×** |
| Training cost | ~$300 | ~$175-200 | **1.5×** |

### Inference (BF16 → INT4)

| Metric | BF16 | FP8 | INT8 | INT4 |
|--------|:----:|:---:|:----:|:----:|
| Model size | 9.5 GB | 4.75 GB | 4.75 GB | 2.4 GB |
| KV cache (32K ctx) | 175 MB | 88 MB | — | — |
| Throughput relative | 1× | 1.5× | 1.5× | 2× |
| Single 4090/3090 | Yes | Yes | Yes | **Yes** |

### Quality

| Metric | FP8 Training | INT8 Inference | INT4 Inference |
|--------|:------------:|:--------------:|:--------------:|
| Perplexity degradation | **<0.2%** | **<0.3%** | **<0.5%** |
| MMLU accuracy drift | ±0.3% | -0.5% | -1.0% |

---

## 6. Gotchas & Edge Cases

1. **First-step delayed scaling**: No amax history at step 0. Safe because weights are ~N(0, 0.02) — scale=1.0 covers the range.

2. **MoE load imbalance × quantization**: Experts receiving 10× more tokens have different activation distributions. Block-wise scaling (128) handles this; per-tensor scaling would fail.

3. **Gradient accumulation + loss scaling**: `DynamicLossScaler.unscale_grads()` must only be called after the full accumulation window, not per micro-batch.

4. **Checkpoint compatibility**: FP8 state dicts include amax trackers. Loading BF16 checkpoints into FP8 models requires re-initialization (one calibration step suffices).

5. **Gate/Router must stay FP32**: Quantizing routing weights causes catastrophic expert collapse — all tokens route to 2-3 experts.

6. **RMSNorm must stay FP32**: FP32 norm compute cost is <0.1% of total FLOPS. Quantizing it breaks stability.

7. **AWQ calibration diversity**: Use 128 samples from the training distribution. Domain-specific sets improve targeted quality but hurt generalization.

8. **MoE experts are heterogeneous**: Different experts learn different features. Per-expert quantization (not shared scales across experts) is critical for INT4 quality.

---

*"FP8 is not about tolerating lower precision — it's about recognizing that transformers never needed 16-bit precision for 90% of their compute in the first place."*

— Principal Engineer's Note, Foundation Models Division, 2026
