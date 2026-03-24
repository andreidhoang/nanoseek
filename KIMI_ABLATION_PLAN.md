# NanoSeek × Kimi Architecture Ablation Plan
## Implementing Kimi Linear (KDA) & K2 Innovations for Pre-Training Ablation
### Senior Research Engineer Plan — March 2026

---

## 0. Research Thesis

**Central question**: At nano scale (1B active), which architectural innovations from
Kimi Linear and Kimi K2 transfer, and do they compound?

We test three independent axes of innovation:
1. **Attention**: KDA+MLA hybrid (Kimi Linear) vs pure MLA (DeepSeek V3)
2. **MoE sparsity**: Higher sparsity (K2-style 128 experts) vs current (64 experts)
3. **Optimizer**: MuonClip (QK-Clip) vs MuonAdamW

**Why this matters**: No one has tested these at nano scale. Kimi Linear was validated
at 3B active (48B total), K2 at 32B active (1T total). If these advantages transfer
down to 1B, it changes the optimal architecture for efficient models.

**Expected outcome**: A Pareto frontier of quality vs inference cost, with 4-6 data
points at anchor scale, validated at 500M, and the winner trained at 1B.

---

## 1. The Ablation Matrix

### 1.1 Independent Variables

| Axis | Baseline (NanoSeek) | Variant A (Kimi Linear) | Variant B (Kimi K2) |
|------|--------------------|-----------------------|---------------------|
| Attention | 16× MLA | 12× KDA + 4× MLA (3:1) | 16× MLA |
| MoE | 64 experts, top-8, grouped | 64 experts, top-8, grouped | 128 experts, top-8, no grouping |
| Optimizer | MuonAdamW | MuonAdamW | MuonClip (QK-Clip) |
| Positional | RoPE everywhere | NoPE on MLA layers, KDA handles position | RoPE everywhere |
| Shared experts | 2 | 2 | 1 |

### 1.2 Experimental Runs (Anchor Scale)

We DON'T test all combinations (2³ = 8). Instead, we use a **fractional factorial design**
that maximizes information per GPU-hour:

```
Run 0: NanoSeek baseline         [MLA-16, E64-grouped, MuonAdamW, RoPE]
Run 1: +KDA hybrid only          [KDA12+MLA4, E64-grouped, MuonAdamW, NoPE-MLA]
Run 2: +High sparsity only       [MLA-16, E128-no-group, MuonAdamW, RoPE]
Run 3: +QK-Clip only             [MLA-16, E64-grouped, MuonClip, RoPE]
Run 4: KDA + high sparsity       [KDA12+MLA4, E128-no-group, MuonAdamW, NoPE-MLA]
Run 5: Full Kimi (all three)     [KDA12+MLA4, E128-no-group, MuonClip, NoPE-MLA]
```

**Why this design**:
- Runs 0-3: isolate each variable's marginal effect
- Run 4: test the KDA×sparsity interaction (most likely to compound)
- Run 5: full combination (if all three help, this should be best)
- If Run 1 dominates, KDA is the key innovation → focus there
- If Run 2 dominates, sparsity scaling law holds at nano scale → publish

### 1.3 Compute Budget

| Scale | Params (active) | Tokens | Time/run (A6000) | Runs | Total |
|-------|-----------------|--------|-------------------|------|-------|
| Anchor | ~55M | ~1.1B | ~4-6 hrs | 6 | ~30 hrs |
| 500M validation | ~440M | ~4.4B | ~14 hrs | 2-3 | ~35 hrs |
| 1B full | ~1.08B | ~22B | 8×H100 ~14 hrs | 1-2 | ~28 hrs |

Total anchor ablation: **~30 GPU-hours** (fits in 2 days on single A6000).
Plus 500M validation of top 2-3 variants: **~35 GPU-hours** (2 more days).

---

## 2. Architecture Specifications

### 2.1 NanoKDA — Kimi Linear at Nano Scale

**Layer layout** (16 layers, 3:1 ratio):
```
Layer  0: Dense FFN  + KDA attention    ← dense FFN (first_k_dense_replace=1)
Layer  1: MoE        + KDA attention
Layer  2: MoE        + KDA attention
Layer  3: MoE        + MLA attention    ← full attention every 4th layer
Layer  4: MoE        + KDA attention
Layer  5: MoE        + KDA attention
Layer  6: MoE        + KDA attention
Layer  7: MoE        + MLA attention
Layer  8: MoE        + KDA attention
Layer  9: MoE        + KDA attention
Layer 10: MoE        + KDA attention
Layer 11: MoE        + MLA attention
Layer 12: MoE        + KDA attention
Layer 13: MoE        + KDA attention
Layer 14: MoE        + KDA attention
Layer 15: MoE        + MLA attention    ← last layer is MLA (global context)
```

Result: **12 KDA layers + 4 MLA layers** (75% linear, 25% full attention).

**KDA layer specification** (from Kimi Linear paper, Sec 4):
```python
class KDAAttention(nn.Module):
    """Kimi Delta Attention — channel-wise gated delta rule linear attention."""

    # Core recurrence:
    # S_t = (I - β_t k_t k_t^T) Diag(α_t) S_{t-1} + β_t k_t v_t^T
    # o_t = S_t^T q_t

    def __init__(self, hidden_size, num_heads, head_dim=128):
        # Input projections (Eq. from paper Section 4.1)
        self.w_qk = CastLinear(hidden_size, num_heads * head_dim * 2)  # q,k jointly
        self.w_v  = CastLinear(hidden_size, num_heads * head_dim)
        self.w_o  = CastLinear(num_heads * head_dim, hidden_size)

        # Short convolution (kernel=4, causal, per-head)
        self.short_conv_qk = nn.Conv1d(num_heads * head_dim * 2, ..., kernel_size=4, padding=3, groups=...)
        self.short_conv_v  = nn.Conv1d(num_heads * head_dim, ..., kernel_size=4, padding=3, groups=...)

        # Forget gate α ∈ [0,1]^{d_k} — low-rank parameterization
        self.alpha_down = CastLinear(hidden_size, head_dim)       # d → d_k
        self.alpha_up   = CastLinear(head_dim, num_heads * head_dim)  # d_k → n_heads*d_k

        # Beta gate β ∈ [0,1] — scalar per head
        self.beta_proj = CastLinear(hidden_size, num_heads)

        # Output gate — Sigmoid (NOT Swish! Paper ablation shows Sigmoid >> Swish)
        self.gate_down = CastLinear(hidden_size, head_dim)
        self.gate_up   = CastLinear(head_dim, num_heads * head_dim)

        # Head-wise RMSNorm before output gating
        self.head_norm = nn.GroupNorm(num_heads, num_heads * head_dim, affine=True)

    def forward(self, x, ...):
        # 1. Project q, k, v
        qk = self.short_conv_qk(self.w_qk(x))  # ShortConv → Swish → L2Norm
        q, k = qk.chunk(2, dim=-1)
        q, k = l2_norm(F.silu(q)), l2_norm(F.silu(k))

        v = F.silu(self.short_conv_v(self.w_v(x)))

        # 2. Compute gates
        alpha = torch.sigmoid(self.alpha_up(self.alpha_down(x)))  # [B, S, n_heads*d_k]
        beta  = torch.sigmoid(self.beta_proj(x))                   # [B, S, n_heads]

        # 3. Run KDA kernel (chunkwise parallel or recurrent)
        o = kda_kernel(q, k, v, alpha, beta, chunk_size=64)

        # 4. Output gating
        gate = torch.sigmoid(self.gate_up(self.gate_down(x)))
        o = self.w_o(gate * self.head_norm(o))
        return o
```

**MLA layers in hybrid mode: NoPE (No Position Encoding)**
- Remove RoPE from the 4 MLA layers entirely
- KDA layers handle ALL positional information via channel-wise gating
- MLA layers become pure content-based attention
- This simplifies long-context extension (no YaRN needed for MLA layers)
- Paper evidence: NoPE + KDA >> RoPE + KDA on long-context benchmarks

**Parameter count matching**:
KDA layer params ≈ MLA layer params (both ≈5M at anchor scale), because:
- KDA saves: no q_lora_rank/kv_lora_rank compression projections
- KDA adds: ShortConv, alpha gate, beta gate, output gate
- Net: roughly comparable. Fine-tune head_dim to match exactly.

### 2.2 NanoK2 — High-Sparsity MoE at Nano Scale

**Key changes from NanoSeek baseline**:

```yaml
# NanoSeek baseline
n_routed_experts: 64
moe_intermediate_size: 768     # per-expert FFN dim
n_group: 8, topk_group: 4     # grouped routing
n_shared_experts: 2
num_experts_per_tok: 8

# NanoK2 variant
n_routed_experts: 128          # 2× more experts
moe_intermediate_size: 384     # halved FFN dim to keep N_active constant!
n_group: 1, topk_group: 1     # NO grouping (K2 style)
n_shared_experts: 1            # 1 shared expert (K2 style)
num_experts_per_tok: 8         # same top-8
```

**Parameter budget verification**:
```
Baseline:  active = 8 × (3 × 2048 × 768) = 8 × 4.72M = 37.7M per MoE layer
NanoK2:    active = 8 × (3 × 2048 × 384) = 8 × 2.36M = 18.9M per MoE layer
           + need to compensate: increase num_experts_per_tok to 16? Or keep 8 and accept fewer active params?
```

**IMPORTANT**: To keep N_active constant, we have two options:
1. **Option A (prefer)**: 128 experts, top-8, inter_dim=768 → N_total doubles, N_active same
2. **Option B**: 128 experts, top-16, inter_dim=384 → N_total same, N_active same, more experts active

K2 used Option A (sparsity=48, same active params). We follow: **128 experts, top-8, inter_dim=768**.
This doubles N_total but keeps N_active and FLOPs identical. Tests the pure sparsity scaling law.

**Corrected NanoK2 config**:
```yaml
n_routed_experts: 128          # 2× more experts
moe_intermediate_size: 768     # SAME per-expert dim (not halved)
n_group: 1, topk_group: 1     # no grouping
n_shared_experts: 1
num_experts_per_tok: 8         # same top-8
# N_active: identical to baseline
# N_total: ~2× baseline (~9.5B vs ~4.75B)
# Sparsity: 16 (vs 8 in baseline)
```

Memory impact: 2× total params means 2× memory for expert weights.
At anchor scale (~55M active), total ≈ 240M → 480M. Still fits A6000 (48GB) easily.

### 2.3 MuonClip — QK-Clip for Muon

Add QK-Clip to existing MuonAdamW optimizer:

```python
class MuonClip:
    """MuonAdamW + QK-Clip for attention stability."""

    def __init__(self, ..., qk_clip_tau=100.0):
        self.tau = qk_clip_tau

    def step(self):
        # 1. Normal Muon step (Newton-Schulz + scale + weight decay)
        super().step()

        # 2. QK-Clip: post-step weight rescaling
        if self.tau > 0:
            self._qk_clip()

    def _qk_clip(self):
        """Per-head QK-Clip from Kimi K2 paper.

        For each attention head h:
          S_max^h = (1/√d) max_{batch} max_{i,j} Q_i^h · K_j^h_T
          If S_max^h > τ:
            γ_h = min(1, τ / S_max^h)
            W_qc^h *= √γ_h    (compressed query)
            W_kc^h *= √γ_h    (compressed key)
            W_qr^h *= γ_h     (rotary query, squared effect)
            W_kr: untouched    (shared across heads)
        """
        for layer in self.model.layers:
            attn = layer.self_attn
            if not isinstance(attn, MultiHeadLatentAttention):
                continue

            # Get current batch's max attention logits per head
            # This requires hooking into the forward pass or computing S_max
            # from the weight matrices directly (cheaper approximation)
            self._clip_head_weights(attn)
```

**Implementation note**: K2 computes S_max from the actual batch during forward pass.
For simplicity at anchor scale, we can approximate using weight spectral norms.
The full batch-based version is needed at larger scales.

---

## 3. Implementation Plan

### Phase 1: KDA Implementation (Priority — Highest Impact)

**Step 1.1: KDA Kernel** (3-5 days)

The KDA chunkwise algorithm requires an efficient CUDA kernel. Options:

| Option | Effort | Speed | Compatibility |
|--------|--------|-------|---------------|
| `flash-linear-attention` library | 1 day setup | Fast (optimized Triton) | Requires Triton 2.2+ |
| Pure PyTorch (reference) | 2 days | Slow (~5× slower) | Universal |
| Custom Triton kernel | 5 days | Fast | Requires Triton knowledge |

**Recommendation**: Start with `flash-linear-attention` (FLA) library which already
has the delta rule + channel-wise gating kernel. Kimi Linear's open-source code
uses this. Fall back to pure PyTorch reference for correctness testing.

```bash
pip install flash-linear-attention  # or build from source
```

FLA provides: `fla.ops.delta_rule.chunk_delta_rule` for the chunkwise KDA kernel.
The key is to adapt Kimi Linear's parameterization (ShortConv, L2Norm, Sigmoid gate)
on top of FLA's kernel.

**Step 1.2: KDA Module Implementation** (2 days)

New file: `nanoseek/nanoseek/kda.py`
```python
# Components needed:
# 1. ShortConv1d — causal convolution with kernel=4
# 2. KDAAttention — full KDA layer with gates
# 3. Integration into NanoSeekDecoderLayer (attention_type='mla' or 'kda')
```

**Step 1.3: Config Extension** (0.5 days)

```python
@dataclass
class KDAConfig:
    """Kimi Delta Attention configuration."""
    enabled: bool = False
    kda_ratio: int = 3            # N KDA layers per 1 MLA layer (3:1 default)
    head_dim: int = 128           # d_k = d_v = 128
    conv_kernel: int = 4          # ShortConv kernel size
    alpha_rank: int = 128         # Low-rank gate dimension
    chunk_size: int = 64          # Chunkwise parallelism chunk size
    use_nope_for_mla: bool = True # Remove RoPE from MLA layers in hybrid mode
```

**Step 1.4: Layer Assignment Logic** (0.5 days)

Modify `NanoSeekModel.__init__()`:
```python
for i in range(config.num_layers):
    if config.kda.enabled:
        use_kda = (i % (config.kda.kda_ratio + 1)) != config.kda.kda_ratio
        # e.g., for 3:1 → layers 0,1,2 = KDA, layer 3 = MLA, ...
    else:
        use_kda = False

    if use_kda:
        attention = KDAAttention(...)
    else:
        attention = MultiHeadLatentAttention(..., use_rope=not config.kda.use_nope_for_mla)
```

**Step 1.5: Unit Tests** (1 day)

```python
# tests/test_kda.py
class TestKDA:
    def test_kda_output_shape(self):
        """KDA output matches MLA output shape."""

    def test_kda_recurrent_equals_parallel(self):
        """Recurrent and chunkwise modes produce same output."""

    def test_kda_causal_mask(self):
        """KDA respects causality (future tokens don't leak)."""

    def test_kda_state_update(self):
        """KDA state S updates correctly across chunks."""

    def test_hybrid_layer_assignment(self):
        """3:1 ratio produces correct KDA/MLA assignment."""

    def test_nope_mla_no_rope(self):
        """MLA layers in hybrid mode have no RoPE."""

    def test_kda_gradient_flow(self):
        """Gradients flow through KDA → gates → projections."""

    def test_param_count_comparable(self):
        """KDA hybrid has comparable param count to pure MLA."""
```

### Phase 2: High-Sparsity MoE (1 day)

**Step 2.1: Config Changes**

Already supported by existing `MoEConfig`. Just need new config preset:
```python
def get_nanok2_anchor_config():
    """NanoK2 anchor config: 128 experts, no grouping."""
    config = get_nanoseek_config("anchor")
    config.moe.n_routed_experts = 128
    config.moe.n_group = 1          # no grouping
    config.moe.topk_group = 1
    config.moe.n_shared_experts = 1
    # moe_intermediate_size stays 768 (same per-expert dim)
    return config
```

**Step 2.2: Routing Adaptation**

The Gate class already supports n_group=1 (degenerates to global top-k).
Need to verify:
- `router_weight` shape: [hidden_dim, 128] (just changes E)
- `bias` buffer shape: [128]
- Group routing with n_group=1: `scores_grouped.view(N, 1, 128)` → topk(2, dim=-1) → selects top-2 globally
  - Actually with n_group=1, topk_group=1, the group step is a no-op. Simplify.

**Step 2.3: Memory Verification**

At anchor scale (hidden=480):
- Baseline 64 experts: 64 × 3 × 480 × 768 × 4 bytes ≈ 282MB per MoE layer
  - Wait, anchor hidden is 480, so intermediate = 0.375 × 480 = 180
  - 64 × 3 × 480 × 180 = 64 × 259K = 16.6M params × 4B = 66MB per MoE layer
- NanoK2 128 experts: 128 × 3 × 480 × 180 = 33.2M params × 4B = 133MB per MoE layer
- Total model: 15 MoE layers × (133 - 66) = +1GB. Still fits A6000 easily.

### Phase 3: MuonClip (QK-Clip) (1 day)

**Step 3.1: Extend optim.py**

Add QK-Clip as a post-step hook:

```python
def qk_clip_step(model, tau=100.0):
    """Post-optimizer QK-Clip for Muon stability.

    Called after each optimizer step. Rescales Q/K weights
    per attention head when max logit exceeds threshold.
    """
    for layer in model.layers:
        attn = layer.self_attn
        if not hasattr(attn, 'wq_b'):  # Skip KDA layers
            continue

        # Approximate S_max from weight norms (cheaper than batch computation)
        # Full method: track S_max during forward pass
        with torch.no_grad():
            # wq_b: [q_lora_rank, num_heads * qk_head_dim]
            # wkv_b: [kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim)]
            q_weight = attn.wq_b.weight  # [num_heads * qk_head_dim, q_lora_rank]
            kv_weight = attn.wkv_b.weight  # [num_heads * (nope + v), kv_lora_rank]

            nh = attn.num_heads
            qk_dim = attn.qk_head_dim

            # Reshape to per-head
            q_heads = q_weight.view(nh, qk_dim, -1)  # [H, qk_dim, rank]
            # K nope is first qk_nope_head_dim dims of each head's KV block
            nope = attn.qk_nope_head_dim
            kv_per_head = nope + attn.v_head_dim
            kv_heads = kv_weight.view(nh, kv_per_head, -1)
            k_heads = kv_heads[:, :nope, :]  # [H, nope, rank]

            for h in range(nh):
                # Approximate max logit via spectral norm product
                q_norm = torch.linalg.matrix_norm(q_heads[h], ord=2)
                k_norm = torch.linalg.matrix_norm(k_heads[h], ord=2)
                s_max_approx = q_norm * k_norm / math.sqrt(qk_dim)

                if s_max_approx > tau:
                    gamma = tau / s_max_approx
                    sqrt_gamma = gamma.sqrt()
                    # Scale compressed Q/K projections
                    q_heads[h].mul_(sqrt_gamma)
                    k_heads[h].mul_(sqrt_gamma)
                    # Note: RoPE Q projection (wq_b rope part) gets gamma (not sqrt)
                    # For simplicity at anchor scale, apply sqrt to all

            # Write back
            q_weight.data.copy_(q_heads.view_as(q_weight))
            kv_weight.data.copy_(kv_heads.view_as(kv_weight))  # K part modified, V untouched? No, need to rebuild.
            # Actually, kv_weight has K_nope and V interleaved per head.
            # Need careful indexing. See detailed implementation below.
```

**Note**: The full MLA-compatible QK-Clip requires careful handling of the compressed
projection structure. The key insight is that in MLA, Q and K are expanded from
low-rank latents, so we clip the expansion weights (wq_b, wkv_b) per head.

### Phase 4: Anchor Ablation Runs (2-3 days)

**Step 4.1: Training Script Extension**

Add `--arch` flag to `pre_train.py`:
```bash
# Run 0: NanoSeek baseline
python -m nanoseek.scripts.pre_train --run ablation-baseline --scale anchor --arch nanoseek

# Run 1: +KDA hybrid
python -m nanoseek.scripts.pre_train --run ablation-kda --scale anchor --arch nanokda

# Run 2: +High sparsity
python -m nanoseek.scripts.pre_train --run ablation-sparsity --scale anchor --arch nanok2

# Run 3: +QK-Clip
python -m nanoseek.scripts.pre_train --run ablation-qkclip --scale anchor --arch nanoseek --qk-clip

# Run 4: KDA + high sparsity
python -m nanoseek.scripts.pre_train --run ablation-kda-sparse --scale anchor --arch nanokda-k2

# Run 5: Full combination
python -m nanoseek.scripts.pre_train --run ablation-full-kimi --scale anchor --arch nanokda-k2 --qk-clip
```

**Step 4.2: Metrics to Track**

For each run, log to W&B:

```yaml
Primary:
  - ema_val_bpb         # Main quality metric (RULE 3)
  - train_loss          # Training loss curve
  - wall_clock_time     # Total training time

MoE Health (RULE 7):
  - H_load              # Load balance entropy (>2 bits)
  - I_spec              # Expert specialization MI
  - dead_expert_count   # Number of never-selected experts
  - expert_gini         # Gini coefficient of load

Attention Stability:
  - max_attn_logit      # Max attention logit (for QK-Clip analysis)
  - kda_state_norm      # KDA state matrix Frobenius norm (track growth)
  - qk_clip_trigger_pct # Percentage of heads triggering QK-Clip

Efficiency:
  - tokens_per_second   # Throughput
  - peak_memory_gb      # GPU memory usage
  - mfu                 # Model FLOPs utilization

Inference Proxy:
  - mtp_acceptance_rate # MTP speculative acceptance
  - kv_cache_size_mb    # KV cache memory at seq_len=4096
```

**Step 4.3: Analysis Protocol**

After all 6 runs complete:

1. **Rank by ema_val_bpb**: Which architecture gives lowest validation loss?
2. **Plot scaling curves**: loss vs tokens for all 6 runs on same axes
3. **Compute marginal effects**:
   - KDA effect = Run1 - Run0 (and Run4 - Run2)
   - Sparsity effect = Run2 - Run0 (and Run4 - Run1)
   - QK-Clip effect = Run3 - Run0
   - Interaction: (Run4 - Run0) vs (Run1 - Run0) + (Run2 - Run0)
4. **Check MoE health**: Any runs with H_load < 2 or dead experts?
5. **Check stability**: Any attention logit spikes? QK-Clip triggers?
6. **Efficiency comparison**: tokens/sec and memory across architectures

### Phase 5: 500M Validation (2 days)

Take top 2-3 architectures from anchor ablation. Scale to 500M config:
- hidden=1280, layers=16, ~441M active
- muP transfer: verify scaling rules hold across architectures
- 4.4B tokens each run
- **Key question**: Does the ranking from anchor scale hold at 500M?

### Phase 6: 1B Training (1 day setup + 14 hrs GPU)

Train the winning architecture at full NanoSeek-1B scale (22B tokens, 8×H100).

---

## 4. Detailed KDA Math for Implementation

### 4.1 Forward Pass (Chunkwise Parallel)

For a chunk of C tokens [t_start, t_start+C):

**Intra-chunk (within chunk, causal):**
```
O_intra = Tril((Γ·Q)(K/Γ)^T) · (U - W·S_prev)
```

where:
- Γ = cumulative product of diag(α) (channel-wise decay factors)
- W, U = WY-packed auxiliary vectors (from UT transform)
- S_prev = state from previous chunk

**Inter-chunk (state readout):**
```
O_inter = (Γ·Q) · S_prev
```

**Total output:**
```
O = O_inter + O_intra
```

**State update:**
```
S_new = Diag(γ_C) · S_prev + (Γ·K)^T · (U - W·S_prev)
```

where γ_C = product of all α within the chunk.

### 4.2 FLOPs Analysis at NanoSeek Scale

For sequence length T=4096, head dim d=128, chunk size C=64:

```
KDA per head:  6T·d² + 3T·C·d + T·C² = 6·4096·16384 + 3·4096·64·128 + 4096·4096
             = 402M + 100M + 16.8M = 519M FLOPs

MLA per head:  2T²·d = 2·4096²·128 = 4,295M FLOPs

Ratio: KDA/MLA ≈ 0.12× (8.3× fewer FLOPs per head!)
```

At 4K context, KDA is dramatically cheaper. The savings compound:
- 12 KDA layers × 0.12× + 4 MLA layers × 1.0× = 1.44 + 4.0 = 5.44 "MLA-equivalents"
- vs 16 MLA layers = 16.0 "MLA-equivalents"
- **Attention FLOPs reduction: 66%**

However, total FLOPs include MoE FFN (which dominates at this scale), so overall
training speedup is more modest: **~10-15% faster** per step.

### 4.3 KV Cache Analysis

| Model | KV Cache per Layer per Token | 16 Layers @ 4K |
|-------|------------------------------|-----------------|
| NanoSeek (pure MLA) | kv_lora_rank + rope_dim = 143 + 64 = 207 | 16 × 207 × 4096 × 2B = 27MB |
| NanoKDA (hybrid) | 4 MLA layers: 207 each, 12 KDA layers: d_k × d_v = 128×128 = 16K (state only) | 4×207×4096×2B + 12×16K×2B = 7MB + 0.4MB = 7.4MB |

**KV cache reduction: 3.6×** (from 27MB to 7.4MB at 4K context).
At 128K context, the advantage grows to ~10×+ because KDA state is O(1) per token.

---

## 5. Risk Analysis

### 5.1 High Risk: KDA Kernel Availability

**Risk**: `flash-linear-attention` may not support our exact KDA variant, or may
have compatibility issues with our training setup (PyTorch version, CUDA version).

**Mitigation**:
1. First validate with pure PyTorch reference implementation (slow but correct)
2. If FLA works, switch for speed
3. If FLA fails, use the Kimi Linear open-source Triton kernel directly

### 5.2 Medium Risk: KDA at Nano Scale

**Risk**: KDA's advantage may not hold at 55M active params (anchor scale).
The Kimi Linear paper only tested at 3B active.

**Mitigation**: This is exactly what we're testing. If KDA fails at anchor,
we learn something valuable (scale-dependent benefit). Run pure-PyTorch
reference at anchor first to get the learning signal quickly.

### 5.3 Medium Risk: 128-Expert Memory

**Risk**: 128 experts at 500M/1B scale may cause memory pressure on A6000.

**Mitigation**: At 1B scale, 128 experts × 3 × 2048 × 768 × 4B = 1.2GB per MoE layer
× 15 layers = 18GB just for expert weights (vs 9GB for 64 experts). Still fits
A6000 (48GB) but tight with activations. May need gradient checkpointing tuning.

### 5.4 Low Risk: QK-Clip at Nano Scale

**Risk**: At 1B scale, Muon may not cause logit explosion (K2 saw it at 1T).

**Mitigation**: This is cheap to test. If QK-Clip never triggers (like K2's self-
deactivation after 70K steps), we confirm it's unnecessary at nano scale.
Still worth having as insurance.

### 5.5 Low Risk: NoPE Breaks Short Context

**Risk**: Removing RoPE from MLA layers could hurt 4K quality.

**Mitigation**: KDA's channel-wise gating provides richer positional information
than RoPE (data-dependent vs fixed). The paper shows NoPE+KDA > RoPE+KDA.
If it fails at nano scale, we can add RoPE back to MLA layers (Run 1 variant).

---

## 6. Expected Outcomes & Decision Framework

### 6.1 Scenario Analysis

**Scenario A: KDA dominates** (Run 1 >> Run 0)
→ Action: Focus on KDA optimization, validate at 500M, train NanoKDA-1B
→ Paper angle: "Hybrid linear attention scales down to 1B active params"

**Scenario B: High sparsity dominates** (Run 2 >> Run 0)
→ Action: Push sparsity further (256 experts?), validate at 500M
→ Paper angle: "Sparsity scaling law holds at nano scale"

**Scenario C: Both compound** (Run 4 >> Run 1, Run 2 individually)
→ Action: Train NanoKDA-K2-1B with both innovations
→ Paper angle: "KDA + high-sparsity MoE: compounding architectural advances"

**Scenario D: QK-Clip is the key** (Run 3 >> Run 0, others marginal)
→ Action: QK-Clip was masking training instability. Investigate baseline training.
→ Unlikely but informative.

**Scenario E: Nothing helps** (all runs ≈ Run 0)
→ Action: These innovations are scale-dependent, only help at ≥3B active.
→ Paper angle: "Negative result: Kimi innovations require scale" (still publishable)

### 6.2 Go/No-Go Criteria

| Criterion | Threshold | Action if Fail |
|-----------|-----------|----------------|
| KDA hybrid val_bpb improvement | > 0.02 nats vs baseline | Skip KDA, focus on sparsity |
| High-sparsity val_bpb improvement | > 0.01 nats vs baseline | Keep 64 experts |
| QK-Clip stability improvement | Any spike prevented OR no spikes in any run | Drop QK-Clip |
| MoE health (all runs) | H_load > 2.0, dead_experts < 5% | Debug routing before proceeding |
| Training speed regression | < 20% slower than baseline | Accept if quality gain > 5% |

---

## 7. Timeline

```
Week 1 (Days 1-5):
  Day 1-2: KDA module implementation (kda.py) + pure PyTorch reference
  Day 3:   FLA kernel integration + correctness tests
  Day 4:   Config extension + layer assignment logic + unit tests
  Day 5:   High-sparsity MoE config + QK-Clip implementation

Week 2 (Days 6-10):
  Day 6:   Integration testing (all 6 configs build + forward pass)
  Day 7-8: Anchor ablation runs 0-5 (parallel where possible)
  Day 9:   Analysis + decision on top variants
  Day 10:  500M validation runs (top 2-3 architectures)

Week 3 (Days 11-14):
  Day 11-12: 500M runs complete + analysis
  Day 13:    1B training setup + launch on multi-GPU
  Day 14:    Results + report
```

---

## 8. File Changes Summary

### New Files
```
nanoseek/nanoseek/kda.py              ← KDA attention module
nanoseek/tests/test_kda.py            ← KDA unit tests
nanoseek/tests/test_hybrid_model.py   ← Hybrid model integration tests
docs/KIMI_ABLATION_PLAN.md            ← THIS FILE
```

### Modified Files
```
nanoseek/nanoseek/config.py           ← Add KDAConfig, NanoK2 config presets
nanoseek/nanoseek/model.py            ← Layer assignment logic for KDA/MLA hybrid
nanoseek/nanoseek/optim.py            ← Add qk_clip_step() function
nanoseek/scripts/pre_train.py         ← --arch flag, QK-Clip integration
```

### Dependencies
```
flash-linear-attention               ← KDA kernel (optional, for speed)
triton >= 2.2                        ← Required by FLA
```

---

## 9. Connection to NanoSeek Roadmap

This ablation plan fits into the existing NanoSeek roadmap as follows:

```
Phase 2 (COMPLETE) → Phase 3A: HP Grid Search (current plan)
                   → Phase 3B: Architecture Ablation (THIS PLAN)  ← NEW
                   → Phase 4: 1B Training (best architecture from 3A + 3B)
```

Phase 3B runs in parallel with or immediately after Phase 3A. The HP grid search
(Phase 3A) finds optimal learning rates for the baseline architecture. The architecture
ablation (Phase 3B) finds the optimal architecture. The winner of Phase 3B inherits
the best HPs from Phase 3A (with minor re-tuning if architecture changes significantly).

**Key integration point**: Both Phase 3A and 3B use the same anchor config (16L, 480h)
and the same evaluation protocol (ema_val_bpb, H_load, I_spec). Results are directly
comparable.

---

## 10. What Would a Top 1% Researcher Focus On?

### The Non-Obvious Insights

1. **KDA × RL interaction is the real prize.** Kimi Linear showed that KDA scales
   BETTER under RL than MLA. If we confirm this at nano scale, it means NanoKDA
   will benefit more from our GRPO pipeline (Phase 5). This is the highest-leverage
   finding because it affects post-training, not just pre-training.

2. **Sparsity scaling law at nano scale is publishable.** K2 showed the law at 1T scale.
   Demonstrating it at 1B (with 2-4 data points on the sparsity axis) would be a
   genuine contribution to the MoE literature.

3. **The NoPE finding is underappreciated.** If KDA eliminates the need for RoPE on
   MLA layers, it removes YaRN complexity entirely. Long-context extension becomes
   trivial. This is an engineering win beyond quality improvement.

4. **QK-Clip is cheap insurance, not a research question.** At nano scale it almost
   certainly won't trigger. Include it for completeness but don't expect surprises.

5. **The interaction between KDA and expert specialization (I_spec) is uncharted.**
   Does linear attention change how experts specialize? KDA compresses context
   differently than MLA — this could affect routing patterns. Track I_spec carefully
   in KDA vs MLA runs.

### What NOT to Waste Time On

- Don't ablate KDA ratio at anchor scale (3:1 is well-validated, test other ratios only if 3:1 fails)
- Don't implement MoonViT / vision (K2.5's multimodal is out of scope for pre-training ablation)
- Don't implement Agent Swarm (post-training concern, not pre-training)
- Don't over-optimize the QK-Clip implementation (spectral norm approximation is fine)
- Don't ablate shared expert count (1 vs 2 is a minor effect compared to KDA and sparsity)

### The Meta-Lesson

The biggest insight from studying all 4 Kimi papers is: **the attention mechanism
and the MoE architecture are independently improvable.** Kimi Linear improved
attention (KDA). K2 improved MoE (sparsity). K2.5 showed they compose well.
Our ablation tests whether this composability holds at nano scale — and if so,
NanoSeek gets both improvements for free.
