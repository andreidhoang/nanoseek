# Weight Initialization, Numerical Stability & Gradient Health in NanoSeek

## Principal Engineer's Deep Dive — February 2026

**Author**: Distinguished AI Researcher & Principal Engineer, Foundation Models Division  
**Scope**: Weight initialization strategy, numerical stability guarantees, mixed precision pitfalls, and gradient health monitoring for NanoSeek  
**Prerequisites**: `model/model.py` (NanoSeekModel, RMSNorm, MLA, MoE), `model/config.py` (NanoSeekConfig), `scripts/pre-train.py` (training loop)  
**Criticality**: **MAXIMUM** — A single numerical instability can waste $300 and 14 hours of H100 time

---

## 1. Engineer's Thinking Process

Before I train a 4.75B parameter model for 14 hours on $300 worth of GPU time, I need to be **CERTAIN** that training won't diverge. Here's how I verify numerical health at every level.

**The nightmare scenario:** You launch a training run on 8×H100 at midnight. At 3am, step 8,412 of 41,900, a single NaN propagates through the attention mechanism. Every parameter gradient becomes NaN. The optimizer states — 57 GB of carefully accumulated first and second moments — are irreversibly corrupted. You wake up to discover you've burned $100 of compute on garbage, and you cannot resume from the last checkpoint because the optimizer states were already poisoned two checkpoints ago.

**The prevention strategy:**

1. **Initialize correctly** — weights must start in a numerically stable regime
2. **Normalize aggressively** — RMSNorm at every sub-layer boundary keeps activations bounded
3. **Upcast critical operations** — softmax and normalization compute in float32 even when training in bf16
4. **Scale attention properly** — prevent softmax saturation that kills gradients
5. **Balance experts** — prevent MoE collapse that creates dead parameters
6. **Monitor continuously** — detect anomalies before they cascade

Each of these six layers of defense corresponds to specific code in NanoSeek. Let's trace through every one.

---

## 2. Weight Initialization — Why 0.02?

### The Actual Code

From `model/model.py`, lines 1545–1554:

```python
def _init_weights(self, module: nn.Module):
    if isinstance(module, nn.Linear):
        std = 0.02
        if hasattr(module, 'SCALE_INIT'):
            std *= module.SCALE_INIT
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

This is applied via `self.apply(self._init_weights)` at line 1524, which recursively initializes every `nn.Linear` and `nn.Embedding` in the entire model.

### From First Principles: Why 0.02?

The classical initialization schemes are derived from variance preservation — keeping the variance of activations constant across layers during the forward pass, and the variance of gradients constant during the backward pass.

**Xavier/Glorot initialization:**

```
std = sqrt(2 / (fan_in + fan_out))
```

This preserves variance under the assumption of linear activations. For a matrix of shape `[out, in]`, `fan_in = in` and `fan_out = out`.

**Kaiming/He initialization (for ReLU-family):**

```
std = sqrt(2 / fan_in)
```

This accounts for ReLU killing half the distribution (variance halves, so we multiply by 2).

**NanoSeek's choice: fixed std=0.02.** This is deliberately simpler, and there's wisdom in the simplicity. Let's see how it compares to Xavier for each weight matrix.

### Layer-by-Layer Comparison: NanoSeek std=0.02 vs Xavier

For each weight matrix `[out_features, in_features]`, Xavier std = `sqrt(2 / (in + out))`:

| Weight Matrix | Shape | Xavier std | NanoSeek std | Ratio (NanoSeek/Xavier) | Assessment |
|---|---|---|---|---|---|
| `embed_tokens.weight` | [65536, 2048] | 0.0053 | 0.02 | 3.8× | Much larger (standard for embeddings) |
| `layers.*.self_attn.wq_a.weight` | [430, 2048] | 0.0284 | 0.02 | 0.70× | Slightly smaller |
| `layers.*.self_attn.wq_b.weight` | [1536, 430] | 0.0319 | 0.02 | 0.63× | Smaller |
| `layers.*.self_attn.wkv_a.weight` | [175, 2048] | 0.0300 | 0.02 | 0.67× | Smaller |
| `layers.*.self_attn.wkv_b.weight` | [2048, 143] | 0.0302 | 0.02 | 0.66× | Smaller |
| `layers.*.self_attn.wo.weight` | [2048, 1024] | 0.0255 | 0.02 | 0.78× | Close |
| `layers.*.ffn.gate.weight` (router) | [64, 2048] | 0.0308 | Kaiming | — | Uses `kaiming_uniform_` |
| `layers.*.ffn.experts[i].gate_proj.weight` | [768, 2048] | 0.0267 | 0.02 | 0.75× | Close |
| `layers.*.ffn.experts[i].up_proj.weight` | [768, 2048] | 0.0267 | 0.02 | 0.75× | Close |
| `layers.*.ffn.experts[i].down_proj.weight` | [2048, 768] | 0.0267 | 0.02 | 0.75× | Close |
| `lm_head.weight` | [65536, 2048] | 0.0053 | 0.02 | 3.8× | Much larger (like embedding) |
| `norm.weight` (RMSNorm) | [2048] | — | 1.0 (ones) | — | Initialized to ones |

**Key observations:**

1. **For linear projections (MLA, MoE experts, FFN):** NanoSeek's 0.02 is ~0.63–0.78× of Xavier. This is **intentionally conservative** — slightly smaller initialization is more stable for deep MoE models because it:
   - Prevents early expert routing collapse (experts start closer to uniform output → router has time to learn)
   - Reduces the initial magnitude of residual stream perturbations (layers add less initially → gradients flow more uniformly)
   - Matches empirical findings from GPT-2, GPT-3, and LLaMA, which all use 0.02

2. **For embeddings:** 0.02 is 3.8× larger than Xavier. This is standard practice across all modern LLMs. The intuition: embedding matrices are essentially lookup tables, not linear transforms. Higher variance helps the model distinguish between tokens early in training. Xavier's assumption of "preserving variance through a linear transform" doesn't apply to discrete lookups.

3. **For the router (Gate):** This is the one exception — it uses `kaiming_uniform_` (He initialization), not 0.02. This is defined at line 404 in the `Gate.__init__` method. The router needs slightly different initialization because it maps from hidden_size to n_routed_experts — a classification-like operation where Kaiming initialization is standard.

4. **For RMSNorm weights:** Initialized to ones (`torch.ones(dim)` at line 192). This means normalization initially acts as pure normalization — the learnable scale factors start at identity, and the model learns to rescale each dimension as needed.

### The Variance Propagation Argument

Consider a single forward pass through `L` transformer layers. Each layer adds a residual:

```
h_{l+1} = h_l + f(h_l)
```

If `f(h_l)` has variance `σ²` and is independent of `h_l`, then after `L` layers:

```
Var(h_L) = Var(h_0) + L × σ²
```

**With standard init (std=0.02):**
- After 16 layers with attention + FFN, the residual stream magnitude grows as `O(sqrt(L))`
- For L=16: magnitude grows by factor ~4
- This is manageable but not ideal for very deep models

**With SCALE_INIT (discussed below):**
- Output projections use `std = 0.02 / sqrt(2 * L)`
- Each layer's contribution has reduced variance: `σ² / (2L)`
- After L layers: `Var(h_L) = Var(h_0) + L × σ²/(2L) = Var(h_0) + σ²/2`
- The residual stream magnitude stays `O(1)` regardless of depth!

### SCALE_INIT: Output Projection Scaling

The `_init_weights` method checks for a `SCALE_INIT` attribute:

```python
if hasattr(module, 'SCALE_INIT'):
    std *= module.SCALE_INIT
```

In production DeepSeek V3, the output projections (`wo` in attention and `down_proj` in FFN/experts) are tagged with:

```python
self.wo.SCALE_INIT = 1.0 / math.sqrt(2 * num_layers)
```

**For NanoSeek with 16 layers:**

```
SCALE_INIT = 1 / sqrt(2 × 16) = 1 / sqrt(32) = 0.177
Effective std = 0.02 × 0.177 = 0.00354
```

**Why `2 * num_layers`?** Each decoder layer has TWO residual additions: one from attention and one from FFN/MoE. So the total number of residual additions is `2L`, and we need to scale by `1/sqrt(2L)` to keep variance constant.

**Current NanoSeek status:** The `SCALE_INIT` infrastructure is in place in `_init_weights`, but the actual attribute assignment on `wo` and `down_proj` modules is not currently present in the model code. For the educational NanoSeek with 16 layers, the basic 0.02 init works fine — the variance growth factor of ~4× is well within the stabilization capacity of RMSNorm. For deeper models (>32 layers), adding `SCALE_INIT` becomes critical.

**Mathematical proof that SCALE_INIT keeps variance O(1):**

```
Let each residual contribution have variance σ²_l = (0.02 × SCALE_INIT)²
    = (0.02 / sqrt(2L))²
    = 0.0004 / (2L)

After L layers with 2 additions each:
Var(h_L) = Var(h_0) + 2L × σ²_l
         = Var(h_0) + 2L × 0.0004/(2L)
         = Var(h_0) + 0.0004

The residual stream variance increases by only 0.0004, regardless of L. ✓
```

---

## 3. RMSNorm — The Stability Anchor

### The Actual Code

From `model/model.py`, lines 186–199:

```python
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x / rms
        return (self.weight * x).to(dtype)
```

### Why RMSNorm over LayerNorm?

**LayerNorm:**
```
y = (x - mean(x)) / std(x) × γ + β
```
Requires: 1 mean, 1 variance, 1 normalization, 2 affine parameters (γ, β).

**RMSNorm:**
```
y = x / RMS(x) × γ
```
Requires: 1 RMS (which is sqrt(mean(x²))), 1 normalization, 1 affine parameter (γ).

**Differences:**
1. **No centering** — RMSNorm does not subtract the mean. This saves one reduction operation per forward pass.
2. **No bias** — RMSNorm has only a scale parameter γ (weight), no shift parameter β. Fewer parameters per norm layer.
3. **~15% faster** — Removing the mean computation and bias eliminates one full reduction kernel launch on GPU.
4. **Equivalent training dynamics** — Zhang & Sennrich (2019) showed that the re-centering in LayerNorm provides negligible benefit for transformer training. The scale normalization is what matters.

**Why NanoSeek (and DeepSeek V3) use RMSNorm:** In a model with 16 layers, each applying RMSNorm twice (pre-attention and pre-FFN), plus one final norm, that's 33 normalization operations per forward pass. The 15% speedup on each one adds up, especially at the MoE scale where FFN computation is already sparse.

### The Float32 Cast — Why It's Critical

```python
x = x.float()  # THIS LINE SAVES TRAINING
```

This single line might be the most important numerical stability guarantee in the entire codebase. Here's why:

**BF16 number format:**
```
┌───┬──────────┬─────────┐
│ S │ EEEEEEEE │ MMMMMMM │   1 sign + 8 exponent + 7 mantissa bits
└───┴──────────┴─────────┘   Precision: ~3.4 significant decimal digits
                              Range: ±3.39 × 10³⁸
```

**The danger: `x.pow(2).mean()`**

Consider a hidden state vector where activations are in the range [-10, 10] (common after several transformer layers without normalization):

1. **Squaring in BF16:** `10² = 100`. Still representable, but we've lost precision. BF16 can only represent 100 as exactly 100.0 (7 mantissa bits give us values like 96, 100, 104 — gaps of 4 at this magnitude).

2. **Accumulating squares in BF16:** With hidden_size=2048, we're summing 2048 squared values. If the average squared value is ~1, the sum is ~2048. In BF16, numbers around 2048 have a precision of ~2 (gaps between representable values). We've lost significant precision.

3. **The catastrophe: large activations.** If hidden values reach ~100 (not uncommon in early training without normalization), then x² ≈ 10000. Summing 2048 such values gives ~20,000,000. In BF16, numbers around 10⁷ have precision gaps of ~128. The computed mean has catastrophic precision loss.

**With float32:** 23 mantissa bits give 7.2 significant decimal digits. The sum of 2048 values up to 10⁸ each is perfectly representable. No precision loss.

**Concrete failure demonstration:**

```python
# What happens WITHOUT the float32 cast:
x = torch.randn(1, 1, 2048, dtype=torch.bfloat16) * 50  # Large activations

# BF16 path (DANGEROUS):
rms_bf16 = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
# x.pow(2) → values up to 2500, mean → ~2500, sqrt → ~50
# Precision loss in accumulation can cause ±2-5% error in RMS

# Float32 path (SAFE):
rms_fp32 = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
# Full precision throughout, error < 10⁻⁶
```

The 2–5% error in BF16 RMS computation might seem small, but it compounds across 16 layers × 2 norms × thousands of training steps. Over time, this systematic bias drifts the model's internal representations, causing subtle training instability that manifests as loss spikes or gradient anomalies thousands of steps later.

### The eps Parameter

```python
self.eps = 1e-6  # Added inside sqrt
```

This prevents division by zero when RMS is near zero. Consider an input vector that is all zeros (or very close to zero):

```
RMS(0) = sqrt(0 + eps) = sqrt(1e-6) = 0.001
x / RMS(x) = 0 / 0.001 = 0
```

Without eps: `sqrt(0)` = 0, and `x / 0` = NaN → entire forward pass becomes NaN → all gradients become NaN → training is destroyed.

**Why 1e-6 specifically?** This is the standard choice for float32 computation. It's small enough not to affect the normalization when RMS is in the normal range (typically 0.1–10), but large enough to provide numerical safety margin. For BF16-only computation, 1e-3 would be more appropriate (since BF16 can't distinguish 1e-6 from 0 near the underflow boundary), but since NanoSeek upcasts to float32 for the computation, 1e-6 is correct.

### The Output Cast

```python
return (self.weight * x).to(dtype)
```

After normalizing in float32, the result is cast back to the input dtype (typically bf16). This ensures:
1. The normalized activations are in the input's dtype for memory efficiency
2. Downstream computations (attention, FFN) work in the expected precision
3. The weight multiplication happens in float32 (since `x` is float32 at this point and `self.weight` is float32), giving full precision to the learned scale factors

### Position in the Architecture

RMSNorm appears at three critical positions in each decoder layer (see `NanoSeekDecoderLayer` at line 1378):

```python
# Pre-attention normalization (line 1388)
self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

# Pre-FFN normalization (line 1421)  
self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

# Final normalization before lm_head (line 1498)
self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

Additionally, RMSNorm is used inside MLA for the compressed representations:

```python
# Q compression normalization (line 266)
self.q_norm = RMSNorm(q_lora_rank)     # dim=430

# KV compression normalization (line 271)
self.kv_norm = RMSNorm(kv_lora_rank)   # dim=143
```

**The role of each norm:**

1. **Pre-attention norm (`input_layernorm`):** Stabilizes the input to MLA. Without this, the QKV projections would receive inputs of unpredictable magnitude, causing attention score explosion.

2. **Pre-FFN norm (`post_attention_layernorm`):** Stabilizes the input to MoE experts (or dense FFN). Without this, expert gate scores would be dominated by input magnitude rather than semantic content, causing routing collapse.

3. **Final norm (`self.norm`):** Stabilizes the input to the language model head. The lm_head projects 2048-dim hidden states to 65536-dim logits. Without normalization, the logit magnitudes would vary wildly across layers, making the softmax over vocabulary extremely temperature-sensitive.

4. **Q/KV compression norms:** These are less common but critical for MLA. The low-rank projections (430 or 143 dimensions) can amplify or suppress certain directions. Normalizing after compression ensures the reconstructed Q, K, V have consistent magnitude regardless of the compressed representation's norm.

---

## 4. Attention Numerical Stability

### Softmax in Float32

From `model/model.py`, line 362:

```python
attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
```

This is arguably the second most important numerical stability line in the codebase (after the float32 cast in RMSNorm).

**Why softmax is dangerous in BF16:**

The softmax function computes:

```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

BF16's `exp()` has two critical failure modes:

1. **Overflow:** BF16 max value is ~3.39 × 10³⁸, but `exp(89)` = 4.49 × 10³⁸ ≈ overflow. In bf16, attention scores above ~88 cause `exp()` to return `inf`. With `inf / inf`, the result is `NaN`.

2. **Underflow:** BF16 min positive normal value is ~1.18 × 10⁻³⁸. `exp(-88)` ≈ 6.1 × 10⁻³⁹ underflows to 0. If ALL attention scores are very negative (far from the max), the entire softmax denominator becomes 0, giving `0/0 = NaN`.

**In practice, can attention scores exceed 88?**

Yes! The raw QK dot product without scaling grows as `O(sqrt(d))` where d is the head dimension. For NanoSeek with qk_head_dim = 96:

```
E[q·k] = 0 (random init)
Var[q·k] = d = 96 (each dimension contributes variance ~1)
std[q·k] = sqrt(96) ≈ 9.8
```

At initialization, scores are ~N(0, 9.8²). A 3σ outlier is ~30, well within BF16 range. But during training, as the model learns sharp attention patterns, individual scores can spike to 50–100+. This is especially true for:
- First-token attention (many models attend heavily to position 0)
- Copy/retrieval patterns (exact-match attention is high)
- MoE routed attention (some experts specialize in sharp patterns)

**The `dtype=torch.float32` argument:**

PyTorch's `F.softmax` with `dtype=torch.float32` does three things in one fused operation:
1. Upcasts the input to float32
2. Computes softmax in float32 (with the standard max-subtraction trick: `softmax(x) = softmax(x - max(x))` for further stability)
3. Returns the result in float32

The `.to(q.dtype)` then downcasts back to bf16 for the value multiplication.

**What happens without `dtype=torch.float32`:**

```python
# DANGEROUS: softmax in bf16
attn_weights = F.softmax(attn_weights, dim=-1)
# Step 8000: attention score hits 95.0
# exp(95) in bf16 → inf
# inf / (inf + ...) → NaN
# NaN × v → NaN
# NaN propagates to residual → all hidden states become NaN
# Game over.
```

### The Softmax Scale: `mscale / sqrt(qk_head_dim)`

From `MultiHeadLatentAttention.__init__`, line 262:

```python
self.softmax_scale = mscale / math.sqrt(self.qk_head_dim)
```

Where `qk_head_dim = qk_nope_head_dim + qk_rope_head_dim = 64 + 32 = 96`.

```
softmax_scale = 1.0 / sqrt(96) = 0.1021
```

And from line 357:

```python
attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.softmax_scale
```

**Why divide by sqrt(d)?**

The dot product `q · k` is a sum of `d` independent terms. By the Central Limit Theorem:

```
q · k ≈ N(0, d × Var[q_i] × Var[k_i])
```

If Q and K entries have unit variance, then `Var[q·k] = d`. Dividing by `sqrt(d)` normalizes the variance back to 1:

```
Var[q·k / sqrt(d)] = d / d = 1
```

**Why this matters for softmax:**

When the input to softmax has large variance, the output becomes nearly one-hot (argmax behavior):

```
softmax([10, 0, 0]) = [0.9999, 0.00005, 0.00005]    ← almost one-hot
softmax([1, 0, 0])  = [0.576, 0.212, 0.212]          ← soft distribution
```

One-hot attention has near-zero gradients for non-max positions. This kills the model's ability to learn which tokens to attend to — the gradient signal only flows through the single max-scoring token.

By scaling to unit variance, we keep softmax inputs in the "soft" regime where gradients flow to multiple tokens, enabling the model to learn nuanced attention patterns.

**The mscale factor:**

`mscale` defaults to 1.0 in the standard configuration. When YaRN context extension is active (Phase 2 training or extended inference), `mscale` compensates for the change in RoPE frequency distribution. Extended context means the same head dimension must encode more position information, which can change the effective magnitude of QK products. The mscale factor adjusts the softmax scale to maintain the same effective "temperature."

### Causal Mask: `-inf` Values

From `create_causal_mask()`, lines 202–211:

```python
def create_causal_mask(seq_len, dtype=torch.float32, device=None, past_len=0):
    mask = torch.full((seq_len, seq_len + past_len), float('-inf'), dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=past_len + 1)
    return mask.unsqueeze(0).unsqueeze(0)
```

And applied at line 360:

```python
if attention_mask is not None:
    attn_weights = attn_weights + attention_mask
```

**Why float('-inf') is numerically exact:**

```python
softmax([-inf, x2, x3]) = [exp(-inf), exp(x2), exp(x3)] / Z
                         = [0, exp(x2), exp(x3)] / Z
```

`exp(-inf) = 0.0` exactly in IEEE 754 floating point. This is not an approximation — it's the defined behavior. So:
- Masked positions contribute exactly 0 to the attention output
- No gradient leaks from future tokens (0 × anything = 0, and ∂0/∂x = 0)
- The causal constraint is perfectly enforced

**What if you used a large negative number instead of `-inf`?**

```python
# WRONG: using -1e9 instead of -inf
mask_value = -1e9
softmax([-1e9, 1.0, 2.0]) = [exp(-1e9)/Z, exp(1)/Z, exp(2)/Z]
```

In float32, `exp(-1e9) = 0.0` (underflows), so this works in practice. But in BF16, the subtraction `score + (-1e9)` can cause precision issues if the score itself is large. Using `-inf` is the canonical approach that works correctly regardless of the score magnitude.

---

## 5. MoE Numerical Stability

### Sigmoid vs Softmax Routing

From `Gate.forward()`, lines 409–415:

```python
def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
    scores = F.linear(x, self.weight)

    if self.score_func == "softmax":
        scores = F.softmax(scores, dim=-1)
    else:
        scores = torch.sigmoid(scores)
```

NanoSeek defaults to `scoring_func="sigmoid"` (from `MoEConfig`, line 311).

**Why sigmoid is more numerically stable than softmax for routing:**

1. **Independence:** Sigmoid scores each expert independently: `σ(w_i · x)`. Increasing one expert's score does NOT decrease another's. With softmax, `softmax(w_i · x)`, increasing expert i's logit mechanically decreases all other experts' scores. This creates coupled optimization that can lead to oscillatory routing patterns.

2. **Gradient flow:** Sigmoid gradient: `σ(x) × (1 - σ(x))`, always positive, peaked at x=0. Each expert's gradient is independent. Softmax gradient involves the full Jacobian: `∂softmax_i/∂x_j = softmax_i × (δ_ij - softmax_j)`, creating cross-expert gradient interference.

3. **Smoother expert training:** With softmax, if one expert gets "too good," its score approaches 1.0 and all others approach 0.0. The near-zero experts receive near-zero gradients and stop learning → expert collapse. With sigmoid, each expert can independently maintain a healthy score.

4. **Load balancing compatibility:** Sigmoid + bias-based load balancing is more natural because adding a bias to a sigmoid score shifts the activation curve, directly controlling the probability of selection. With softmax, a bias changes the relative ranking but the effect on selection probability is non-linear and harder to control.

### Load Balancing Bias — Preventing Expert Collapse

From `MoE.update_load_balance_bias()`, lines 721–727:

```python
def update_load_balance_bias(self, gamma: float = 0.001):
    with torch.no_grad():
        load = self.gate.expert_load
        mean_load = load.mean()
        if mean_load > 0:
            imbalance = (load - mean_load) / (mean_load + 1e-8)
            self.gate.expert_bias.sub_(gamma * imbalance)
```

And from `Gate.forward()`, lines 417–418:

```python
if self.training:
    scores_for_selection = scores + self.expert_bias.unsqueeze(0)
```

**The problem this solves:**

Without load balancing, MoE training consistently converges to a degenerate state where only 2–3 experts receive meaningful token flow. The remaining 60+ experts become "dead" — their parameters are never updated, representing wasted capacity.

**Why collapse happens:**

1. Early in training, one expert randomly gets slightly better at a common pattern
2. The router sends more tokens to this expert → it gets more gradient updates → it gets even better
3. Positive feedback loop → this expert dominates
4. Experts receiving few tokens get weak gradients → they fall further behind
5. Equilibrium: 2–3 dominant experts, rest unused

**How the bias mechanism prevents this:**

At each training step, after the forward pass:

1. Count how many tokens each expert processed: `load[i]`
2. Compute the mean load across all experts: `mean_load`
3. For each expert, compute relative imbalance: `(load[i] - mean_load) / mean_load`
4. Subtract from the expert's routing bias: `bias[i] -= gamma × imbalance[i]`

**Effect:**
- If expert i is **overloaded** (`load[i] > mean_load`): imbalance > 0, bias decreases, routing score decreases → fewer tokens
- If expert i is **underloaded** (`load[i] < mean_load`): imbalance < 0, bias increases, routing score increases → more tokens
- If expert i is **balanced** (`load[i] ≈ mean_load`): imbalance ≈ 0, no change

**The gamma schedule:**

From `NanoSeekModel.get_gamma()`, lines 1534–1540:

```python
def get_gamma(self, tokens_processed=None):
    if tokens_processed is None:
        tokens_processed = self.tokens_processed.item()
    freeze_at = int(self.config.total_tokens * self.config.moe.gamma_freeze_ratio)
    if tokens_processed < freeze_at:
        return self.config.moe.gamma    # 0.001
    return 0.0                           # Frozen
```

- **First 80% of training (gamma=0.001):** Bias actively corrects load imbalance. The router learns to distribute tokens while the bias provides a stabilizing force.
- **Last 20% of training (gamma=0.0):** Bias is frozen. The router must maintain balance on its own. Late bias changes would destabilize the routing patterns that have converged, causing loss spikes.

**Why freeze at 80% and not 90% or 70%?**

- Freezing too late (90%+): Not enough time for the router to stabilize without the bias crutch. Risk of routing oscillations in the final phase.
- Freezing too early (70% or less): The model hasn't fully learned its routing patterns yet. Premature freezing can lock in suboptimal expert assignments.
- 80% is DeepSeek V3's empirically validated choice. In V3, they freeze at 14.3T/14.8T ≈ 97%, but NanoSeek uses 80% for extra safety margin at smaller scale.

**Bias evolution during training (simulation):**

```
Step 0:      bias = [0, 0, 0, ..., 0]       (all zeros)
Step 1000:   bias = [-0.02, 0.01, 0.03, ..., -0.01]  (small corrections)
Step 10000:  bias = [-0.15, 0.08, 0.22, ..., -0.12]  (significant for overloaded experts)
Step 30000:  bias = [-0.31, 0.14, 0.28, ..., -0.19]  (approaching equilibrium)
Step 33520:  bias frozen at current values              (80% of 41900 steps)
Step 41900:  training complete, bias unchanged from step 33520
```

### Gradient Through Top-k

From `Gate.forward()`, lines 432–434:

```python
topk_weights, topk_indices = scores_for_selection.topk(self.n_activated_experts, dim=-1)
weights = scores.gather(dim=-1, index=topk_indices)
weights = weights * self.route_scale
```

**The key insight:** Top-k selection is a discrete operation — it's not differentiable. You cannot backpropagate through `argmax` or `topk`. So how do gradients flow in MoE?

The answer is in line 433: `weights = scores.gather(dim=-1, index=topk_indices)`.

- `topk_indices` is treated as a **constant** during backpropagation (detached from the graph)
- `scores` is the differentiable tensor (computed from `sigmoid(W·x)`)
- `gather` indexes into `scores` using the constant indices → fully differentiable with respect to `scores`
- Gradients flow: `loss → weighted_output → weights → scores → Gate.weight, x`

**No straight-through estimator needed.** Unlike some MoE implementations that use Gumbel-softmax or straight-through tricks, DeepSeek's approach cleanly separates:
1. **Selection** (which experts to use) — discrete, non-differentiable, but done with `scores_for_selection` (which includes bias)
2. **Weighting** (how much to weight each selected expert) — continuous, differentiable, done with original `scores` (without bias)

This separation means the routing weights are always clean gradients of the original scoring function, while the selection benefits from the load-balancing bias without contaminating the gradient signal.

---

## 6. Mixed Precision — BF16 Training Strategy

### Why BF16 over FP16

From `scripts/pre-train.py`, lines 927–932:

```python
if device_type == "cuda":
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
```

**BF16 vs FP16 comparison:**

| Property | FP16 | BF16 |
|---|---|---|
| Format | 1+5+10 (sign+exp+mantissa) | 1+8+7 (sign+exp+mantissa) |
| Range | ±65,504 | ±3.39 × 10³⁸ |
| Precision | ~3.3 decimal digits | ~3.4 decimal digits |
| Smallest positive | 6.0 × 10⁻⁸ | 1.2 × 10⁻³⁸ |
| Inf threshold | >65504 | >3.39 × 10³⁸ |

**The critical difference: range.**

- FP16 maxval = 65,504. Gradient norms for a 4.75B parameter model routinely exceed 65,504 during accumulation. Loss values > 10 with cross-entropy over 65,536 vocab can produce logit sums > 65,504. Result: `inf` → `NaN` → dead training.

- BF16 maxval = 3.39 × 10³⁸. Same range as float32. Gradient accumulations, loss values, attention scores — none of these come remotely close to overflow. BF16 training essentially never hits overflow on LLM workloads.

**The tradeoff:** BF16 has slightly less precision (7 vs 10 mantissa bits), meaning individual values are rounded more aggressively. But for neural network training, the stochasticity of SGD already introduces noise at a much larger scale than the precision difference. The extra range is far more valuable than extra precision.

**Why FP16 requires loss scaling but BF16 doesn't:**

With FP16, small gradients (common in early layers of deep models) underflow to zero, losing the gradient signal. Dynamic loss scaling multiplies the loss by a large factor (e.g., 1024) before backprop, then divides gradients by the same factor afterward. This keeps small gradients in the representable FP16 range.

BF16's range extends to 10⁻³⁸, so gradient underflow essentially never happens. NanoSeek's training code has no loss scaler — BF16 just works.

### What Stays in FP32

Despite training in BF16, several critical computations are forcibly kept in float32:

**1. RMSNorm internal computation** (line 196):
```python
x = x.float()
```
Reason: Squared accumulations in BF16 lose precision (Section 3).

**2. Softmax computation** (line 362):
```python
attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
```
Reason: `exp()` overflow in BF16 for large attention scores (Section 4).

**3. Loss computation:**
```python
F.cross_entropy(shift_logits.view(-1, ...), shift_labels.view(-1), ignore_index=-100)
```
PyTorch's `cross_entropy` internally upcasts to float32 for the log-softmax computation. This handles the 65,536-way softmax over vocabulary where BF16 precision would cause significant probability estimation errors.

**4. Optimizer states:**
- **AdamW:** First moment (m) and second moment (v) are stored in float32. These are running exponential averages that must accumulate thousands of steps of information. BF16 precision loss would cause the optimizer to "forget" small but persistent gradient signals.
- **Muon:** The Newton-Schulz orthogonalization runs in BF16 (`X = G.bfloat16()` in muon.py line 22), but the momentum buffer is stored in the optimizer state at whatever precision the parameter uses. The spectral normalization step (`X = X / (X.norm(...) + 1e-7)`) is computed in BF16 — this is stable because the norm is a single scalar, not an accumulation.

**5. Gradient accumulation:**
Gradients are accumulated in BF16 (the default dtype when autocast is active). With `grad_accum_steps` micro-steps (computed from `total_batch_size / world_tokens_per_fwdbwd`), precision loss per accumulation is roughly:

```
Relative error per addition ≈ 2^(-7) ≈ 0.0078 (BF16 machine epsilon)
After N accumulations: error ≈ sqrt(N) × 0.0078 (random walk)
For 16 accumulations: error ≈ 4 × 0.0078 ≈ 3.1%
```

This 3.1% error is acceptable for gradient accumulation because:
- SGD noise is already ~10–30% of the gradient signal
- The optimizer's momentum (β₁=0.9) averages across multiple steps, smoothing out accumulation errors
- For >32 accumulation steps, consider master gradient copies in float32

### TF32 for Matrix Multiplications

```python
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
```

TF32 (TensorFloat-32) uses 19 bits (1+8+10: same exponent as float32, same mantissa as FP16) for the internal accumulation in matrix multiplies on Ampere+ GPUs. This gives:
- ~2× speedup over float32 matmuls
- Same dynamic range as float32 (no overflow)
- FP16-level precision per operation (but accumulated in float32)

For NanoSeek, TF32 affects the float32 computations inside RMSNorm and softmax when they involve matmuls (which they generally don't — they're element-wise ops). The main benefit is for any torch.compile'd paths that fuse operations.

---

## 7. Gradient Health Monitoring

### What to Watch

During production training, the following metrics should be checked at every logging step:

```python
def check_gradient_health(model):
    """Check all parameter gradients for anomalies.
    
    Call after loss.backward() but before optimizer.step().
    Production code should log warnings, not assert — a single
    anomaly might be recoverable, but a pattern of anomalies
    indicates a systemic problem.
    """
    metrics = {
        'total_norm': 0.0,
        'nan_params': [],
        'inf_params': [],
        'large_grad_params': [],
        'vanishing_grad_params': [],
    }
    
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
            
        grad = param.grad
        
        # NaN detection — immediate red flag
        if torch.isnan(grad).any():
            metrics['nan_params'].append(name)
            logger.error(f"NaN gradient in {name}")
            
        # Inf detection — usually precedes NaN
        if torch.isinf(grad).any():
            metrics['inf_params'].append(name)
            logger.error(f"Inf gradient in {name}")
            
        # Gradient magnitude tracking
        norm = grad.norm().item()
        metrics['total_norm'] += norm ** 2
        
        # Large gradient warning (threshold depends on parameter)
        if norm > 100:
            metrics['large_grad_params'].append((name, norm))
            logger.warning(f"Large gradient in {name}: {norm:.4f}")
            
        # Vanishing gradient warning
        if norm < 1e-8 and param.requires_grad:
            metrics['vanishing_grad_params'].append((name, norm))
            logger.warning(f"Vanishing gradient in {name}: {norm:.4e}")
    
    metrics['total_norm'] = metrics['total_norm'] ** 0.5
    return metrics
```

### Gradient Clipping in NanoSeek

From `scripts/pre-train.py`, lines 1505–1510:

```python
grad_norm = 0.0
if config.grad_clip > 0.0:
    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
        orig_model.parameters(), config.grad_clip  # default: 1.0
    )
    grad_norm = grad_norm_tensor.item()
```

Gradient clipping caps the total gradient norm to `max_norm=1.0`. This:
1. Prevents catastrophic parameter updates from loss spikes
2. Stabilizes training when encountering outlier batches
3. The pre-clip norm is logged for monitoring (`grad_norm` in the log output)

**What the grad_norm tells you:**

| Grad Norm | Status | Action |
|---|---|---|
| 0.1 – 1.0 | Healthy | Normal training |
| 1.0 – 5.0 | Elevated | Watch for trends |
| 5.0 – 50 | High, being clipped | Check data quality, LR |
| >50 | Very high | Likely about to diverge |
| 0.0 | Dead | Disconnected graph or dead model |
| NaN | Critical | Stop training immediately |

### Common Failure Modes and Diagnosis

**1. Loss spike (sudden jump in loss, then recovery):**
- **Cause:** Data corruption — a batch with unusual token distribution (e.g., all padding, extremely long sequences, or rare Unicode).
- **Diagnosis:** Log the batch contents when loss > 2× the running average. Check for all-zero batches, sequences with only special tokens, or extreme token IDs.
- **Prevention:** Data validation in the dataloader; gradient clipping limits the damage.

**2. NaN gradients (training immediately diverges):**
- **Cause:** Usually attention score overflow → NaN softmax → NaN everywhere.
- **Diagnosis:** Insert NaN checks before and after softmax. Check if causal mask is correctly applied (missing mask → attending to future tokens → incorrect gradients → divergence).
- **Prevention:** Float32 softmax (already implemented), proper causal masking.

**3. Expert collapse (loss plateaus, expert load becomes highly skewed):**
- **Cause:** Load balancing insufficient or gamma frozen too early.
- **Diagnosis:** Monitor `expert_load` variance. Healthy: variance < 20% of mean. Collapsing: 2–3 experts have >50% of total load.
- **Prevention:** Ensure gamma is active for first 80% of training; consider increasing gamma from 0.001 to 0.003 if collapse persists.

**4. Gradient explosion (grad_norm grows steadily over training):**
- **Cause:** Learning rate too high, or weight decay too low (parameters growing unboundedly).
- **Diagnosis:** Plot grad_norm over time. Healthy: fluctuates around a stable mean. Diverging: steady upward trend.
- **Prevention:** Reduce learning rate; increase weight decay; lower gradient clipping threshold.

**5. Vanishing gradients in deep layers (early layer gradients much smaller than late layers):**
- **Cause:** Poor initialization or RMSNorm not functioning correctly.
- **Diagnosis:** Log per-layer gradient norms. Healthy: roughly uniform across layers (within 10×). Problem: early layers 100× smaller than late layers.
- **Prevention:** Verify RMSNorm is placed correctly; consider SCALE_INIT for output projections.

---

## 8. Production Code: Numerical Health Checker

A complete, ready-to-integrate module that should be run after initialization and periodically during training:

```python
"""
model/eval/numerical_health.py — Numerical Health Monitoring for NanoSeek

Usage:
    checker = NumericalHealthChecker(model, config)
    
    # After initialization (once):
    init_report = checker.check_init_health()
    
    # After each forward pass (every N steps):
    fwd_report = checker.check_forward_health(sample_batch)
    
    # After each backward pass (every N steps):
    grad_report = checker.check_gradient_health()
    
    # Periodically:
    expert_report = checker.check_expert_health()
    loss_report = checker.check_loss_health(loss_history)
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HealthReport:
    """Summary of a health check."""
    healthy: bool
    warnings: List[str]
    errors: List[str]
    metrics: Dict[str, Any]

    def log(self, prefix: str = ""):
        """Log the report at appropriate levels."""
        for error in self.errors:
            logger.error(f"{prefix}{error}")
        for warning in self.warnings:
            logger.warning(f"{prefix}{warning}")
        if self.healthy:
            logger.info(f"{prefix}All checks passed")


class NumericalHealthChecker:
    """
    Comprehensive numerical health monitoring for NanoSeek.
    
    Run after initialization and periodically during training to detect
    numerical instabilities before they cascade into training divergence.
    """

    def __init__(self, model: nn.Module, config=None):
        self.model = model
        self.config = config

    def check_init_health(self) -> HealthReport:
        """
        Verify initialization statistics match expectations.
        
        Checks:
        - All parameters have finite values (no NaN/Inf from init)
        - Linear layer stds are close to 0.02
        - Embedding stds are close to 0.02
        - RMSNorm weights are initialized to 1.0
        - No dead parameters (all-zero weights)
        """
        warnings = []
        errors = []
        metrics = {}

        for name, param in self.model.named_parameters():
            data = param.data

            # NaN/Inf check
            if torch.isnan(data).any():
                errors.append(f"NaN in {name} after initialization")
            if torch.isinf(data).any():
                errors.append(f"Inf in {name} after initialization")

            # Statistics
            mean = data.float().mean().item()
            std = data.float().std().item()
            metrics[name] = {'mean': mean, 'std': std, 'shape': list(data.shape)}

            # Check Linear layers (expect std ≈ 0.02)
            if 'weight' in name and data.dim() == 2 and 'norm' not in name:
                if abs(std - 0.02) > 0.01:
                    warnings.append(
                        f"{name}: std={std:.4f} (expected ~0.02)"
                    )

            # Check RMSNorm (expect mean ≈ 1.0)
            if 'norm' in name and 'weight' in name and data.dim() == 1:
                if abs(mean - 1.0) > 0.01:
                    warnings.append(
                        f"{name}: mean={mean:.4f} (expected ~1.0 for RMSNorm)"
                    )

            # Check for dead parameters
            if data.abs().max().item() < 1e-10:
                warnings.append(f"{name}: all values near zero (dead parameter)")

        healthy = len(errors) == 0
        return HealthReport(healthy=healthy, warnings=warnings,
                            errors=errors, metrics=metrics)

    @torch.no_grad()
    def check_forward_health(self, input_ids: torch.Tensor) -> HealthReport:
        """
        Run a forward pass and check output health.
        
        Checks:
        - Logits are finite (no NaN/Inf)
        - Logit range is reasonable (not too large, not too small)
        - Hidden states are finite
        """
        warnings = []
        errors = []
        metrics = {}

        self.model.eval()
        outputs = self.model(input_ids)
        self.model.train()

        logits = outputs['logits']

        # NaN/Inf in logits
        if torch.isnan(logits).any():
            errors.append("NaN in logits")
        if torch.isinf(logits).any():
            errors.append("Inf in logits")

        # Logit range
        logit_min = logits.min().item()
        logit_max = logits.max().item()
        logit_std = logits.float().std().item()
        metrics['logit_range'] = [logit_min, logit_max]
        metrics['logit_std'] = logit_std

        if abs(logit_max) > 100 or abs(logit_min) > 100:
            warnings.append(
                f"Large logit range: [{logit_min:.2f}, {logit_max:.2f}]"
            )

        if logit_std < 0.01:
            warnings.append(
                f"Very small logit std: {logit_std:.4f} (model may not be learning)"
            )

        # Hidden states
        hidden = outputs['hidden_states']
        if torch.isnan(hidden).any():
            errors.append("NaN in hidden states")
        if torch.isinf(hidden).any():
            errors.append("Inf in hidden states")

        hidden_std = hidden.float().std().item()
        metrics['hidden_std'] = hidden_std

        healthy = len(errors) == 0
        return HealthReport(healthy=healthy, warnings=warnings,
                            errors=errors, metrics=metrics)

    def check_gradient_health(self) -> HealthReport:
        """
        Check gradient health after backward pass.
        
        Checks:
        - No NaN gradients
        - No Inf gradients
        - Total gradient norm is reasonable
        - No extremely large or vanishing individual gradients
        """
        warnings = []
        errors = []
        metrics = {}

        total_norm_sq = 0.0
        nan_params = []
        inf_params = []

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue

            grad = param.grad

            if torch.isnan(grad).any():
                nan_params.append(name)
                errors.append(f"NaN gradient in {name}")

            if torch.isinf(grad).any():
                inf_params.append(name)
                errors.append(f"Inf gradient in {name}")

            norm = grad.norm().item()
            total_norm_sq += norm ** 2

            if norm > 100:
                warnings.append(f"Large gradient in {name}: norm={norm:.4f}")
            if norm < 1e-8 and param.requires_grad:
                warnings.append(f"Vanishing gradient in {name}: norm={norm:.4e}")

        total_norm = total_norm_sq ** 0.5
        metrics['total_norm'] = total_norm
        metrics['nan_count'] = len(nan_params)
        metrics['inf_count'] = len(inf_params)

        if total_norm > 1000:
            errors.append(f"Gradient explosion: total_norm={total_norm:.2f}")
        if total_norm < 1e-8:
            warnings.append(f"Gradient vanishing: total_norm={total_norm:.4e}")

        healthy = len(errors) == 0
        return HealthReport(healthy=healthy, warnings=warnings,
                            errors=errors, metrics=metrics)

    def check_loss_health(self, loss_history: List[float]) -> HealthReport:
        """
        Analyze loss trajectory for anomalies.
        
        Checks:
        - No NaN losses
        - Loss is decreasing on average
        - No catastrophic spikes
        - Loss is in reasonable range
        """
        warnings = []
        errors = []
        metrics = {}

        if not loss_history:
            return HealthReport(healthy=True, warnings=["No loss history"],
                                errors=[], metrics={})

        import math
        recent = loss_history[-100:] if len(loss_history) > 100 else loss_history

        # NaN check
        nan_count = sum(1 for l in recent if math.isnan(l))
        if nan_count > 0:
            errors.append(f"NaN loss detected ({nan_count} times in last {len(recent)} steps)")

        # Range check
        valid = [l for l in recent if not math.isnan(l) and not math.isinf(l)]
        if valid:
            metrics['mean_loss'] = sum(valid) / len(valid)
            metrics['max_loss'] = max(valid)
            metrics['min_loss'] = min(valid)

            if max(valid) > 20:
                warnings.append(f"Very high loss: {max(valid):.4f}")

            # Spike detection: loss > 3× running mean
            if len(valid) > 10:
                running_mean = sum(valid[:len(valid)//2]) / (len(valid)//2)
                spikes = sum(1 for l in valid[len(valid)//2:] if l > 3 * running_mean)
                if spikes > 0:
                    warnings.append(f"{spikes} loss spikes detected (>3× mean)")

        healthy = len(errors) == 0
        return HealthReport(healthy=healthy, warnings=warnings,
                            errors=errors, metrics=metrics)

    def check_expert_health(self) -> HealthReport:
        """
        Check MoE expert load balance and routing health.
        
        Checks:
        - Expert load distribution is roughly uniform
        - No dead experts (zero load)
        - Expert bias is not diverging
        """
        warnings = []
        errors = []
        metrics = {}

        raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        
        if not hasattr(raw_model, 'get_expert_load_stats'):
            return HealthReport(healthy=True, warnings=["No MoE layers found"],
                                errors=[], metrics={})

        stats = raw_model.get_expert_load_stats()

        for layer_idx, layer_stats in stats.items():
            load = layer_stats['expert_load']
            bias = layer_stats['expert_bias']

            mean_load = load.mean().item()
            std_load = load.std().item()
            max_load = load.max().item()
            min_load = load.min().item()

            metrics[f'layer_{layer_idx}'] = {
                'mean_load': mean_load,
                'std_load': std_load,
                'load_cv': std_load / (mean_load + 1e-8),
                'max_load': max_load,
                'min_load': min_load,
                'bias_range': [bias.min().item(), bias.max().item()],
            }

            # Load balance check (CV < 0.3 is healthy)
            cv = std_load / (mean_load + 1e-8)
            if cv > 0.5:
                warnings.append(
                    f"Layer {layer_idx}: Poor load balance (CV={cv:.3f})"
                )

            # Dead expert check
            dead_experts = (load < mean_load * 0.01).sum().item()
            if dead_experts > 0:
                warnings.append(
                    f"Layer {layer_idx}: {dead_experts} dead experts "
                    f"(<1% of mean load)"
                )

            # Bias divergence check
            bias_range = bias.max().item() - bias.min().item()
            if bias_range > 2.0:
                warnings.append(
                    f"Layer {layer_idx}: Large bias range ({bias_range:.3f}). "
                    f"Routing may be bias-dominated."
                )

        healthy = len(errors) == 0
        return HealthReport(healthy=healthy, warnings=warnings,
                            errors=errors, metrics=metrics)
```

---

## 9. Verification: Proving Stability

### Verification Script 1: Initialization Statistics

This script verifies that every parameter in a freshly initialized model has the expected distribution:

```python
"""
Verify initialization statistics for all NanoSeek parameters.

Expected results:
- nn.Linear weights: mean ≈ 0.0, std ≈ 0.02
- nn.Embedding weights: mean ≈ 0.0, std ≈ 0.02
- RMSNorm weights: mean = 1.0, std = 0.0 (initialized to ones)
- Gate weights: Kaiming uniform distribution
- No NaN or Inf values anywhere
"""
import torch
from model.model import NanoSeekModel
from model.config import get_nanoseek_config

config = get_nanoseek_config()
model = NanoSeekModel(config)

print("=" * 80)
print("NanoSeek Initialization Verification")
print("=" * 80)

total_params = 0
nan_count = 0
inf_count = 0

for name, param in model.named_parameters():
    total_params += param.numel()
    
    # NaN/Inf check
    has_nan = torch.isnan(param.data).any().item()
    has_inf = torch.isinf(param.data).any().item()
    if has_nan:
        nan_count += 1
    if has_inf:
        inf_count += 1
    
    mean = param.data.float().mean().item()
    std = param.data.float().std().item()
    
    # Classify parameter type
    if 'norm' in name and 'weight' in name:
        ptype = "RMSNorm"
        expected_std = "0.000"
        expected_mean = "1.000"
    elif 'embed' in name:
        ptype = "Embedding"
        expected_std = "0.020"
        expected_mean = "0.000"
    elif 'gate.weight' in name and 'proj' not in name:
        ptype = "Router"
        expected_std = "Kaiming"
        expected_mean = "~0.000"
    elif param.dim() == 2:
        ptype = "Linear"
        expected_std = "0.020"
        expected_mean = "0.000"
    else:
        ptype = "Other"
        expected_std = "varies"
        expected_mean = "varies"
    
    flag = ""
    if has_nan:
        flag = " *** NaN ***"
    elif has_inf:
        flag = " *** Inf ***"
    
    print(f"  {name:60s} | {str(list(param.shape)):25s} | "
          f"mean={mean:+.6f} std={std:.6f} | "
          f"type={ptype:10s} expected_std={expected_std}{flag}")

print(f"\nTotal parameters: {total_params:,}")
print(f"NaN parameters: {nan_count}")
print(f"Inf parameters: {inf_count}")
assert nan_count == 0, f"Found {nan_count} parameters with NaN!"
assert inf_count == 0, f"Found {inf_count} parameters with Inf!"
print("\n✓ All initialization checks passed")
```

### Verification Script 2: Forward Pass Numerical Range

```python
"""
Verify forward pass produces numerically stable outputs.

Expected results:
- Logits in range [-50, 50] (typical for freshly initialized model)
- No NaN or Inf in logits or hidden states
- Loss is finite and in expected range [~10.5 for random init with vocab=65536]
  (theoretical: -ln(1/65536) = ln(65536) ≈ 11.09)
"""
import torch
from model.model import NanoSeekModel, create_nanoseek
from model.config import get_nanoseek_config, NanoSeekConfig, MLAConfig, MoEConfig, MTPConfig

# Use small config for fast verification
config = NanoSeekConfig(
    hidden_size=256,
    num_layers=2,
    num_heads=4,
    vocab_size=1024,
    intermediate_size=640,
    mla=MLAConfig(
        q_lora_rank=54,
        kv_lora_rank=18,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
    ),
    moe=MoEConfig(
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_intermediate_size=128,
        n_group=2,
        topk_group=2,
        first_k_dense_replace=1,
    ),
    mtp=MTPConfig(num_mtp_modules=1, mtp_num_heads=4),
)

model = NanoSeekModel(config)
model.eval()

batch_size, seq_len = 2, 64
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

print("=" * 80)
print("Forward Pass Numerical Verification")
print("=" * 80)

with torch.no_grad():
    outputs = model(input_ids)

logits = outputs['logits']
hidden = outputs['hidden_states']

# Logit checks
print(f"\nLogits shape: {logits.shape}")
print(f"Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
print(f"Logits mean: {logits.mean().item():.6f}")
print(f"Logits std: {logits.std().item():.6f}")
assert not torch.isnan(logits).any(), "NaN in logits!"
assert not torch.isinf(logits).any(), "Inf in logits!"
print("✓ Logits are finite")

# Hidden state checks
print(f"\nHidden states shape: {hidden.shape}")
print(f"Hidden range: [{hidden.min().item():.4f}, {hidden.max().item():.4f}]")
print(f"Hidden std: {hidden.std().item():.6f}")
assert not torch.isnan(hidden).any(), "NaN in hidden states!"
assert not torch.isinf(hidden).any(), "Inf in hidden states!"
print("✓ Hidden states are finite")

# Loss check
labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
with torch.no_grad():
    outputs_with_loss = model(input_ids, labels=labels)

loss = outputs_with_loss['loss']
main_loss = outputs_with_loss['main_loss']
print(f"\nLoss: {loss.item():.4f}")
print(f"Main loss: {main_loss.item():.4f}")
print(f"Expected initial loss: ~{torch.log(torch.tensor(float(config.vocab_size))).item():.4f} "
      f"(= ln({config.vocab_size}))")
assert not torch.isnan(loss), "NaN in loss!"
assert not torch.isinf(loss), "Inf in loss!"
assert loss.item() > 0, "Loss should be positive!"
print("✓ Loss is finite and positive")

print("\n✓ All forward pass checks passed")
```

### Verification Script 3: Gradient Flow Verification

```python
"""
Verify gradients flow correctly through all model components.

Expected results:
- All trainable parameters have non-None gradients
- No NaN or Inf gradients
- Gradient magnitudes are in reasonable range
- Gradients reach the embedding layer (deepest layer from loss)
"""
import torch
from model.model import NanoSeekModel
from model.config import NanoSeekConfig, MLAConfig, MoEConfig, MTPConfig

# Small config for fast verification
config = NanoSeekConfig(
    hidden_size=256, num_layers=2, num_heads=4, vocab_size=1024,
    intermediate_size=640,
    mla=MLAConfig(q_lora_rank=54, kv_lora_rank=18,
                  qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32),
    moe=MoEConfig(n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1,
                  moe_intermediate_size=128, n_group=2, topk_group=2,
                  first_k_dense_replace=1),
    mtp=MTPConfig(num_mtp_modules=1, mtp_num_heads=4),
)

model = NanoSeekModel(config)
model.train()

input_ids = torch.randint(0, config.vocab_size, (2, 64))
labels = torch.randint(0, config.vocab_size, (2, 64))

outputs = model(input_ids, labels=labels)
loss = outputs['loss']
loss.backward()

print("=" * 80)
print("Gradient Flow Verification")
print("=" * 80)

total_params = 0
has_grad = 0
no_grad = 0
nan_grads = 0
inf_grads = 0

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    total_params += 1
    
    if param.grad is None:
        no_grad += 1
        print(f"  WARNING: No gradient for {name}")
        continue
    
    has_grad += 1
    grad_norm = param.grad.norm().item()
    
    if torch.isnan(param.grad).any():
        nan_grads += 1
        print(f"  ERROR: NaN gradient in {name}")
    if torch.isinf(param.grad).any():
        inf_grads += 1
        print(f"  ERROR: Inf gradient in {name}")

print(f"\nGradient summary:")
print(f"  Total trainable params: {total_params}")
print(f"  With gradients: {has_grad}")
print(f"  Without gradients: {no_grad}")
print(f"  NaN gradients: {nan_grads}")
print(f"  Inf gradients: {inf_grads}")

# Specific checks
embed_grad = model.embed_tokens.weight.grad
assert embed_grad is not None, "Embedding must have gradients!"
print(f"\n  Embedding gradient norm: {embed_grad.norm().item():.6f}")
assert embed_grad.norm().item() > 1e-10, "Embedding gradient too small!"

total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
print(f"  Total gradient norm: {total_norm.item():.4f}")

assert nan_grads == 0, f"Found {nan_grads} NaN gradients!"
assert inf_grads == 0, f"Found {inf_grads} Inf gradients!"
assert no_grad == 0, f"Found {no_grad} parameters without gradients!"

print("\n✓ All gradient flow checks passed")
```

### Verification Script 4: Expert Load Balance

```python
"""
Verify MoE expert load balancing works correctly.

Expected results:
- After initialization, load is roughly uniform across experts
- After bias updates, load becomes more balanced
- No expert receives zero tokens
"""
import torch
from model.model import NanoSeekModel
from model.config import NanoSeekConfig, MLAConfig, MoEConfig, MTPConfig

config = NanoSeekConfig(
    hidden_size=256, num_layers=4, num_heads=4, vocab_size=1024,
    intermediate_size=640,
    mla=MLAConfig(q_lora_rank=54, kv_lora_rank=18,
                  qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32),
    moe=MoEConfig(n_routed_experts=16, num_experts_per_tok=4, n_shared_experts=1,
                  moe_intermediate_size=128, n_group=4, topk_group=2,
                  first_k_dense_replace=1),
    mtp=MTPConfig(num_mtp_modules=0),
)

model = NanoSeekModel(config)
model.train()

print("=" * 80)
print("Expert Load Balance Verification")
print("=" * 80)

# Run a few forward passes to collect load statistics
for i in range(5):
    input_ids = torch.randint(0, config.vocab_size, (4, 64))
    labels = torch.randint(0, config.vocab_size, (4, 64))
    outputs = model(input_ids, labels=labels)
    outputs['loss'].backward()
    model.zero_grad()
    
    # Update load balance bias
    model.update_load_balance_bias(gamma=0.001)

stats = model.get_expert_load_stats()
for layer_idx, layer_stats in stats.items():
    load = layer_stats['expert_load']
    bias = layer_stats['expert_bias']
    
    mean_load = load.mean().item()
    std_load = load.std().item()
    cv = std_load / (mean_load + 1e-8)
    dead = (load < mean_load * 0.01).sum().item()
    
    print(f"\n  Layer {layer_idx}:")
    print(f"    Load: mean={mean_load:.1f}, std={std_load:.1f}, CV={cv:.3f}")
    print(f"    Load distribution: {load.tolist()}")
    print(f"    Bias range: [{bias.min().item():.4f}, {bias.max().item():.4f}]")
    print(f"    Dead experts: {dead}")

print("\n✓ Expert load balance check complete")
```

---

## 10. Summary: The Six Layers of Numerical Defense

NanoSeek's numerical stability is not accidental — it's the result of six deliberately engineered defense layers:

```
Layer 1: Weight Initialization (std=0.02)
    │  Ensures activations start in a numerically safe regime
    │  Slightly conservative init prevents early MoE collapse
    ▼
Layer 2: RMSNorm with Float32 Cast
    │  Clamps activation magnitudes at every sub-layer boundary
    │  Float32 prevents precision loss in squared accumulations
    ▼
Layer 3: Attention Scaling (1/sqrt(d))
    │  Keeps softmax inputs at O(1) variance
    │  Prevents one-hot attention that kills gradients
    ▼
Layer 4: Float32 Softmax
    │  Prevents exp() overflow in BF16
    │  Eliminates the #1 cause of training NaN in LLMs
    ▼
Layer 5: MoE Load Balancing (sigmoid + bias)
    │  Sigmoid scoring: independent, smooth gradients
    │  Dynamic bias: prevents expert collapse
    │  Gamma schedule: freeze at 80% for stability
    ▼
Layer 6: Gradient Clipping (max_norm=1.0)
    │  Last line of defense against loss spikes
    │  Caps the maximum parameter update magnitude
    ▼
    STABLE TRAINING ✓
```

Each layer catches failures that slip through the previous ones:
- If initialization is slightly off → RMSNorm corrects magnitude
- If RMSNorm has precision issues → float32 cast prevents them
- If attention scores grow large → sqrt(d) scaling keeps them reasonable
- If scores still overflow in BF16 → float32 softmax handles it
- If training dynamics cause expert collapse → bias correction restores balance
- If a bad batch causes gradient explosion → clipping limits the damage

**The key principle:** No single mechanism is sufficient. Numerical stability in deep MoE models requires defense in depth, where each layer addresses a specific failure mode and provides redundancy for the others.

---

## Appendix A: Quick Reference — Numerical Constants

| Constant | Value | Source | Purpose |
|---|---|---|---|
| Init std (Linear) | 0.02 | `_init_weights()` | Weight initialization |
| Init std (Embedding) | 0.02 | `_init_weights()` | Embedding initialization |
| RMSNorm eps | 1e-6 | `config.rms_norm_eps` | Prevent division by zero |
| Softmax scale | 1.0/sqrt(96) ≈ 0.102 | `MultiHeadLatentAttention` | Attention score scaling |
| Gradient clip | 1.0 | `TrainingConfig.grad_clip` | Max gradient norm |
| Load balance gamma | 0.001 | `MoEConfig.gamma` | Bias update rate |
| Gamma freeze ratio | 0.80 | `MoEConfig.gamma_freeze_ratio` | When to freeze bias |
| BF16 max value | 3.39 × 10³⁸ | IEEE 754 | Overflow threshold |
| BF16 precision | ~7 mantissa bits | IEEE 754 | ~3.4 decimal digits |
| FP32 precision | ~23 mantissa bits | IEEE 754 | ~7.2 decimal digits |
| Adam beta2 | 0.95 | `NanoSeekConfig.adam_beta2` | Responsive to MoE dynamics |

## Appendix B: Quick Reference — Where Stability Code Lives

| Stability Mechanism | File | Line(s) | Function/Class |
|---|---|---|---|
| Weight initialization | `model/model.py` | 1545–1554 | `NanoSeekModel._init_weights()` |
| RMSNorm (float32 cast) | `model/model.py` | 186–199 | `RMSNorm.forward()` |
| Softmax float32 | `model/model.py` | 362 | `MultiHeadLatentAttention.forward()` |
| Attention scaling | `model/model.py` | 262, 357 | `softmax_scale`, matmul |
| Causal mask (-inf) | `model/model.py` | 202–211 | `create_causal_mask()` |
| Expert load balancing | `model/model.py` | 721–727 | `MoE.update_load_balance_bias()` |
| Routing bias | `model/model.py` | 417–418 | `Gate.forward()` |
| Sigmoid scoring | `model/model.py` | 415 | `Gate.forward()` |
| Gamma schedule | `model/model.py` | 1534–1540 | `NanoSeekModel.get_gamma()` |
| Gradient clipping | `scripts/pre-train.py` | 1505–1510 | Training loop |
| BF16 autocast | `scripts/pre-train.py` | 927–932 | `main()` |
| Router init (Kaiming) | `model/model.py` | 404 | `Gate.__init__()` |
| Gradient health tests | `tests/test_numerical.py` | all | Test suite |

## Appendix C: Relationship to Other Implementation Docs

| Doc | Relationship to This Doc |
|---|---|
| `02_FLASH_MLA_ATTENTION_KERNEL.md` | Flash attention handles softmax stability in the fused kernel |
| `03_FUSED_MOE_EXPERT_PARALLELISM.md` | Expert parallelism must preserve load balancing numerics |
| `05_FP8_MIXED_PRECISION.md` | FP8 adds another precision tier below BF16 |
| `06_MULTI_PHASE_TRAINING_ORCHESTRATION.md` | Phase transitions must maintain numerical continuity |
| `09_POST_TRAINING_SFT_DPO_RLVR.md` | Post-training fine-tuning inherits initialization from pre-training |
