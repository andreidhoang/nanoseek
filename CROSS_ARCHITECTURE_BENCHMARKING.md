# Cross-Architecture Benchmarking: NanoChat (Dense) vs NanoSeek (MoE)
## How a Senior AI Performance Engineer at a Frontier Lab Would Execute This

**Date**: 2026-03-28
**Codebases**: NanoChat (dense GPT, ~124M-1.5B) | NanoSeek (MoE DeepSeek V3.2, 410M active / 1.95B total)
**Hardware**: RTX 4090 (24GB) primary, H100 for FP8 validation

---

## Why This Comparison Matters

You have two architectures in the same workspace:
- **NanoChat**: Dense transformer. GQA attention, ReLU² MLP, sliding window. Optimized for speed (1.80hr GPT-2 speedrun on 8xH100).
- **NanoSeek**: Sparse MoE transformer. MLA attention (23x KV compression), 64-expert SwiGLU MoE (top-8), MTP heads. Optimized for research.

A senior performance engineer would ask: **At equal compute budget, which architecture produces better loss? And where does each waste hardware resources?**

This is exactly what DeepSeek, Google, and Apple do when deciding between dense and MoE for their next frontier model.

---

## Architecture Comparison Map

```
                    NanoChat (Dense)              NanoSeek (MoE)
                    ────────────────              ──────────────
Attention:          GQA (n_head=n_kv_head)        MLA (23× KV compression)
                    gpt.py:65-126                 model.py:184-502
                    head_dim = n_embd/n_head      qk_nope=128, qk_rope=64, v=128
                    QK RMSNorm + 1.15 scale       QK RMSNorm
                    Sliding window (SSSL)         Full context (4096)

FFN:                Dense ReLU² MLP               64-expert SwiGLU MoE + 2 shared
                    gpt.py:129-139                model.py:513-890
                    4×d_model expansion           per-expert inter_dim
                    1 gate + 1 proj               64×(gate+up+down) + 2×shared

Activation:         ReLU² (sparse)                SiLU (gate) × up (SwiGLU)
                    F.relu(x).square()            F.silu(w_gate(x)) * w_up(x)

Precision:          FP32 master → BF16 compute    FP32 master → BF16 compute
                    gpt.py:45-50 (Linear)         model.py:43-60 (CastLinear)
                    FP8 via fp8.py                FP8 via fp8.py (MoE-aware)

Optimizer:          MuonAdamW                     MuonAdamW
                    optim.py (identical impl)     optim.py (identical impl)
                    Polar Express (5 iter)        Polar Express (5 iter)

Position:           RoPE                          RoPE + YaRN (8K extension)
Normalization:      RMSNorm (function)            RMSNorm (module)
Residual:           λ_resid × x + λ_x0 × x0      Standard residual
Training:           torch.compile(dynamic=False)  torch.compile (planned)
```

---

## Phase 1: Establish Ground Truth Baselines (Days 1-4)

### 1.1 NanoChat Baseline Profile

**What to measure**: Where does NanoChat spend its time?

```python
# File: benchmarks/nanochat_profile.py
"""
Profile NanoChat at multiple depths to understand scaling behavior.

NanoChat uses a single `--depth` dial that auto-computes all hyperparameters:
  d4:  4 layers,  ~15M params,  tiny (CPU testing)
  d12: 12 layers, ~124M params, GPT-2 small equivalent
  d20: 20 layers, ~768M params, GPT-2 large equivalent
  d24: 24 layers, ~1.2B params, production speedrun config
  d26: 26 layers, ~1.5B params, current leaderboard config

For fair comparison with NanoSeek ablation (410M active):
  Choose d16-d18 (~400M params) to match active parameter count
"""

def profile_nanochat(depth, batch_size, seq_len=2048, num_steps=30):
    """
    Component timing for NanoChat.

    Components to isolate:
    1. Token embedding + norm           (gpt.py:410-412)
    2. Per-layer residual scaling       (gpt.py:415, λ_resid × x + λ_x0 × x0)
    3. Value embedding lookup           (gpt.py:416, conditional)
    4. Attention:
       a. Q/K/V projections             (gpt.py:87-89, 3 Linear calls)
       b. RoPE application              (gpt.py:99)
       c. QK RMSNorm                    (gpt.py:100)
       d. Flash Attention / SDPA        (gpt.py:108-118)
       e. Output projection             (gpt.py:125)
    5. MLP:
       a. Up projection (d → 4d)        (gpt.py:136)
       b. ReLU² activation              (gpt.py:137)
       c. Down projection (4d → d)      (gpt.py:138)
    6. Final norm                        (gpt.py:418)
    7. LM head (d → vocab)              (gpt.py:422, LARGEST matmul)
    8. Softcap (tanh)                   (gpt.py:425)
    9. Cross-entropy loss               (gpt.py:432)
    10. Backward pass (total)
    11. Optimizer step (MuonAdamW)

    Expected breakdown (d20, RTX 4090, BF16):
    ┌──────────────────────────┬────────┬───────┐
    │ Component                │  ms    │   %   │
    ├──────────────────────────┼────────┼───────┤
    │ QKV Projections          │  3.2   │ 12.1  │
    │ Flash Attn / SDPA        │  5.8   │ 21.9  │
    │ Attn Output Proj         │  1.1   │  4.2  │
    │ MLP Up (d→4d)            │  4.2   │ 15.9  │
    │ ReLU²                    │  0.3   │  1.1  │
    │ MLP Down (4d→d)          │  4.2   │ 15.9  │
    │ LM Head                  │  3.1   │ 11.7  │
    │ RMSNorm (all)            │  0.6   │  2.3  │
    │ Embedding + residual     │  0.4   │  1.5  │
    │ Softcap + loss           │  0.5   │  1.9  │
    │ Other                    │  3.1   │ 11.7  │
    ├──────────────────────────┼────────┼───────┤
    │ TOTAL FORWARD            │ 26.5   │       │
    └──────────────────────────┴────────┴───────┘

    Key difference from NanoSeek:
    NanoChat: Attention ~38% + MLP ~33% + LM head ~12%
    NanoSeek: MoE ~50% + MLA ~20% + MTP ~6%

    In NanoChat, attention and MLP are roughly equal.
    In NanoSeek, MoE dominates because 64 experts are expensive.
    """
    pass
```

### 1.2 NanoSeek Baseline Profile

```python
def profile_nanoseek(scale='ablation', batch_size=2, seq_len=4096, num_steps=30):
    """
    Component timing for NanoSeek.
    (Already detailed in TIER1_BENCHMARKING_GUIDE.md Stage 0)

    Key dimensions (ablation):
      hidden_dim = 1280, num_layers = 16
      MLA: q_lora=275, kv_lora=143, qk_nope=128, qk_rope=64, v=128
      MoE: 64 experts × inter_dim=480, 2 shared × inter_dim=960
      MTP: 1 module

    Key dimensions (1B):
      hidden_dim = 2048, num_layers = 16
      MoE: 64 experts × inter_dim=768, 2 shared × inter_dim=1536
    """
    pass
```

### 1.3 Side-by-Side Baseline Report

**Target output**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  BASELINE COMPARISON: NanoChat (Dense) vs NanoSeek (MoE)                    │
│  Hardware: RTX 4090 (24GB), BF16                                            │
│                                                                             │
│  Configuration:                                                             │
│  ┌──────────────────────┬─────────────────────┬────────────────────────┐     │
│  │                      │ NanoChat d~16        │ NanoSeek ablation     │     │
│  ├──────────────────────┼─────────────────────┼────────────────────────┤     │
│  │ Active params        │ ~410M (dense)        │ ~410M (8/64 experts)  │     │
│  │ Total params         │ ~410M               │ ~1.95B                │     │
│  │ Attention            │ GQA                  │ MLA                   │     │
│  │ FFN                  │ Dense ReLU²          │ 64-expert SwiGLU MoE  │     │
│  │ Sequence length      │ 2048                 │ 4096                  │     │
│  │ Batch size           │ TBD                  │ 2                     │     │
│  │ Vocab                │ 32768                │ 32768                 │     │
│  └──────────────────────┴─────────────────────┴────────────────────────┘     │
│                                                                             │
│  Step-Level Metrics:                                                        │
│  ┌──────────────────────┬─────────────────────┬────────────────────────┐     │
│  │ Metric               │ NanoChat             │ NanoSeek              │     │
│  ├──────────────────────┼─────────────────────┼────────────────────────┤     │
│  │ Step time (ms)       │ ___                  │ ___                   │     │
│  │ Tokens/sec           │ ___                  │ ___                   │     │
│  │ MFU (%)              │ ___                  │ ___                   │     │
│  │ Peak memory (GB)     │ ___                  │ ___                   │     │
│  │ Params/memory ratio  │ ___                  │ ___                   │     │
│  └──────────────────────┴─────────────────────┴────────────────────────┘     │
│                                                                             │
│  Component Breakdown (% of forward pass):                                   │
│  ┌──────────────────────┬─────────────────────┬────────────────────────┐     │
│  │ Component            │ NanoChat             │ NanoSeek              │     │
│  ├──────────────────────┼─────────────────────┼────────────────────────┤     │
│  │ Attention            │ ~38%                 │ ~20%                  │     │
│  │ FFN / MoE            │ ~33%                 │ ~50%                  │     │
│  │ LM Head              │ ~12%                 │ ~8%                   │     │
│  │ Routing              │ 0%                   │ ~3%                   │     │
│  │ Dispatch/Combine     │ 0%                   │ ~8%                   │     │
│  │ MTP                  │ 0%                   │ ~6%                   │     │
│  │ Other                │ ~17%                 │ ~5%                   │     │
│  └──────────────────────┴─────────────────────┴────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 2: Fair Cross-Architecture Comparison (Days 5-10)

### 2.1 The Three Comparison Axes

A frontier lab never compares on a single axis. You need all three:

```
Axis 1: Loss vs Training FLOPs (compute-normalized)
  → "At equal compute, which architecture learns more?"
  → X-axis: cumulative 6 × N_active × tokens_seen
  → Y-axis: val BPB on identical eval set
  → This favors MoE (ignores dispatch overhead)

Axis 2: Loss vs Wall-Clock Time (time-normalized)
  → "At equal wall-clock training time, which is better?"
  → X-axis: elapsed seconds on same hardware
  → Y-axis: val BPB
  → This is the fairest comparison (captures all overhead)
  → This is what Du et al. (2024) recommends

Axis 3: Loss vs Training Dollars (cost-normalized)
  → "At equal budget, which gives better results?"
  → X-axis: cumulative GPU-hours × $/hr
  → Y-axis: val BPB
  → This is the business-relevant metric
```

### 2.2 Matching Configurations

```python
def create_matched_configs():
    """
    Create fair comparison configurations.

    CRITICAL: You must match on ONE dimension and measure the others.
    You CANNOT match on all dimensions simultaneously.

    Match 1: Equal Active Parameters (~410M)
    ─────────────────────────────────────────
    NanoChat:  d_model=1280, n_layer=16, n_head=16
               MLP: 1280 → 5120 → 1280 (ReLU²)
               ~410M total = ~410M active

    NanoSeek:  d_model=1280, n_layer=16, n_head=16 (ablation config)
               MoE: 64 experts × (1280 → 480 → 1280), top-8
               ~410M active, ~1.95B total

    What differs: total memory, step time, tokens/sec

    Match 2: Equal Total Parameters (~1.95B)
    ─────────────────────────────────────────
    NanoChat:  d_model=2048, n_layer=24 (roughly d24 config)
               ~1.5-2B total

    NanoSeek:  ablation scale
               ~1.95B total, ~410M active

    What differs: active compute per token (NanoChat 5× more active FLOPs)

    Match 3: Equal Step Time (most fair)
    ─────────────────────────────────────
    Find NanoChat depth where step_time ≈ NanoSeek ablation step_time
    This automatically adjusts for all overhead.
    """
    configs = {
        'nanochat_410m': {
            'type': 'dense',
            'depth': 16,  # approximate, needs tuning
            'n_embd': 1280,
            'n_head': 16,
            'n_kv_head': 16,
            'seq_len': 2048,
            'active_params': '~410M',
            'total_params': '~410M',
        },
        'nanoseek_ablation': {
            'type': 'moe',
            'n_layer': 16,
            'hidden_size': 1280,
            'n_head': 16,
            'num_experts': 64,
            'top_k': 8,
            'expert_inter': 480,
            'seq_len': 4096,
            'active_params': '~410M',
            'total_params': '~1.95B',
        },
    }
    return configs
```

### 2.3 BPB Evaluation (Architecture-Independent Loss)

```python
def compute_bpb(model, eval_data, tokenizer):
    """
    Bits-per-Byte: the architecture-independent loss metric.

    BPB normalizes for tokenizer differences:
      BPB = CE_loss × (num_tokens / num_bytes) / ln(2)

    Why not just compare CE loss?
    - NanoChat and NanoSeek may have different tokenizers
    - A tokenizer with larger vocab has lower CE (more bits per token)
    - BPB normalizes this away

    Both use vocab=32768, so BPB difference is mainly from
    compression ratio (tokens per byte) which depends on
    the tokenizer's merge rules.

    Eval set MUST be identical for both models.
    Use a held-out subset of ClimbMix (both projects use ClimbMix).
    """
    total_ce_loss = 0.0
    total_tokens = 0
    total_bytes = 0

    for batch in eval_data:
        text = batch['text']
        tokens = tokenizer.encode(text)
        text_bytes = len(text.encode('utf-8'))

        with torch.no_grad():
            loss = model(tokens, targets=tokens[1:])

        total_ce_loss += loss.item() * len(tokens)
        total_tokens += len(tokens)
        total_bytes += text_bytes

    avg_ce = total_ce_loss / total_tokens
    tokens_per_byte = total_tokens / total_bytes
    bpb = avg_ce * tokens_per_byte / math.log(2)

    return bpb
```

### 2.4 Training Curves on Identical Data

```bash
# Run NanoChat at ~410M params on ClimbMix
cd /workspace/nanoseek/nanochat
python -m scripts.base_train \
    --depth 16 --batch-size 2 --seq-len 2048 \
    --num-iterations 500 --eval-every 100 \
    --run bench-nanochat-410m --seed 42

# Run NanoSeek ablation on ClimbMix
cd /workspace/nanoseek/nanoseek
python -m nanoseek.scripts.pre_train \
    --run bench-nanoseek-ablation --scale ablation --seed 42 \
    --num-iterations 500 --eval-every 100 --save-every -1 \
    --device-batch-size 2
```

**Plot all three axes**:

```python
def plot_training_curves(nanochat_log, nanoseek_log):
    """
    Generate the three comparison plots.

    Plot 1: BPB vs Training FLOPs
    ──────────────────────────────
    X: cumulative FLOPs = steps × batch × seq × flops_per_token
       NanoChat: flops_per_token = 6 × 410M + attn_flops
       NanoSeek: flops_per_token = 6 × 410M + mla_attn_flops
    Y: eval BPB (both on same eval set)

    Interpretation:
    - If NanoSeek curve is BELOW NanoChat → MoE learns more per FLOP
    - Expected: MoE should be 10-30% more FLOP-efficient at this scale
      (based on FLAME-MoE and Apple scaling law results)

    Plot 2: BPB vs Wall-Clock Seconds
    ──────────────────────────────────
    X: cumulative wall-clock seconds
    Y: eval BPB

    Interpretation:
    - If NanoSeek curve is BELOW despite slower step time →
      MoE's FLOP efficiency overcomes its overhead
    - If NanoChat is BELOW → MoE overhead negates efficiency
    - This is the most important plot for architecture selection

    Plot 3: BPB vs Training Cost ($)
    ─────────────────────────────────
    X: cumulative GPU-hours × $0.40/hr (RTX 4090 rate)
    Y: eval BPB

    Same as wall-clock on single GPU, but important when
    scaling to multi-GPU (MoE needs more memory, potentially
    more GPUs, so cost diverges from time)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: FLOPs
    axes[0].plot(nanochat_flops, nanochat_bpb, label='NanoChat (Dense)')
    axes[0].plot(nanoseek_flops, nanoseek_bpb, label='NanoSeek (MoE)')
    axes[0].set_xlabel('Training FLOPs (6 × N_active × tokens)')
    axes[0].set_ylabel('Eval BPB')
    axes[0].set_title('Compute Efficiency')

    # Plot 2: Wall-clock
    axes[1].plot(nanochat_seconds, nanochat_bpb, label='NanoChat (Dense)')
    axes[1].plot(nanoseek_seconds, nanoseek_bpb, label='NanoSeek (MoE)')
    axes[1].set_xlabel('Wall-Clock Seconds')
    axes[1].set_ylabel('Eval BPB')
    axes[1].set_title('Time Efficiency (Most Fair)')

    # Plot 3: Cost
    axes[2].plot(nanochat_cost, nanochat_bpb, label='NanoChat (Dense)')
    axes[2].plot(nanoseek_cost, nanoseek_bpb, label='NanoSeek (MoE)')
    axes[2].set_xlabel('Training Cost ($)')
    axes[2].set_ylabel('Eval BPB')
    axes[2].set_title('Cost Efficiency')
```

---

## Phase 3: Component-Level Performance Analysis (Days 11-18)

### 3.1 Attention: MLA vs GQA

```python
def benchmark_mla_vs_gqa():
    """
    Compare NanoSeek's MLA against NanoChat's GQA.

    NanoChat GQA (gpt.py:65-126):
      Q: [B, T, n_head, head_dim]     → [2, 2048, 16, 80]
      K: [B, T, n_kv_head, head_dim]  → [2, 2048, 16, 80]  (n_kv_head=n_head for full MHA)
      V: [B, T, n_kv_head, head_dim]  → [2, 2048, 16, 80]
      KV cache per token: 2 × n_kv_head × head_dim = 2 × 16 × 80 = 2560 values

    NanoSeek MLA (model.py:184-502):
      Q: compressed via wq_a (2048→275) then expanded via wq_b (275→16×192)
      KV: compressed via wkv_a (2048→207) → stored as latent
      KV cache per token: kv_lora_rank + qk_rope_dim = 143 + 64 = 207 values
      Compression: 2560 / 207 = 12.4× (ablation), more at larger head_dim

    What to measure:

    1. Projection FLOP cost:
       GQA:  2 × B × T × d × (n_head + 2×n_kv_head) × head_dim
       MLA:  2 × B × T × (d×q_lora + q_lora×n_head×qk_dim + d×(kv_lora+rope) + kv_lora×n_head×(nope+v))
       MLA has MORE projection FLOPs (extra compression/decompression)

    2. Attention compute:
       GQA:  2 × B × n_head × T² × head_dim (standard)
       MLA:  2 × B × n_head × T² × (qk_nope + qk_rope) (effectively head_dim=192 for MLA)
       MLA has LARGER effective head dim (192 vs 80) → more attention compute

    3. KV cache memory:
       GQA:  B × T × 2 × n_kv_head × head_dim × 2 bytes
       MLA:  B × T × (kv_lora_rank + qk_rope_dim) × 2 bytes
       MLA is 12× smaller

    4. Memory bandwidth during inference:
       GQA:  Must read full KV cache each decode step
       MLA:  Reads compressed latent, decompresses on-the-fly
       At long context, MLA's bandwidth savings dominate

    Test matrix:
    ┌──────────┬──────────┬──────────┬──────────┬──────────┐
    │ Seq Len  │ GQA (ms) │ MLA (ms) │ GQA mem  │ MLA mem  │
    ├──────────┼──────────┼──────────┼──────────┼──────────┤
    │ 512      │          │          │          │          │
    │ 1024     │          │          │          │          │
    │ 2048     │          │          │          │          │
    │ 4096     │          │          │          │          │
    │ 8192     │          │          │          │          │
    └──────────┴──────────┴──────────┴──────────┴──────────┘

    Expected crossover: MLA slower at short context (extra projections),
    faster at long context (memory bandwidth savings).
    """
    pass
```

### 3.2 FFN: Dense ReLU² vs MoE SwiGLU

```python
def benchmark_dense_vs_moe_ffn():
    """
    Compare NanoChat's dense FFN against NanoSeek's MoE FFN.

    NanoChat Dense FFN (gpt.py:129-139):
      Up:   [B×T, d] @ [d, 4d]    → [B×T, 4d]     (1 matmul)
      Act:  ReLU²(x)                                  (elementwise)
      Down: [B×T, 4d] @ [4d, d]   → [B×T, d]      (1 matmul)
      Total: 2 GEMMs + 1 activation
      FLOPs: 2 × B×T × d × 4d × 2 = 16 × B×T × d²

    NanoSeek MoE FFN (model.py:513-890):
      Route: sigmoid + topk + group selection          (small overhead)
      Dispatch: sort tokens by expert                   (indexing overhead)
      Per expert (× 8 active out of 64):
        Gate: [tokens_e, d] @ [d, inter]               (1 matmul)
        Up:   [tokens_e, d] @ [d, inter]               (1 matmul)
        Act:  SiLU(gate) × up                          (elementwise)
        Down: [tokens_e, inter] @ [inter, d]           (1 matmul)
      Combine: weighted scatter-add                     (indexing overhead)
      Shared: 2× dense FFN (always active)             (2 more GEMMs)
      Total: 8×3 + 2×2 = 28 GEMMs + routing + dispatch + combine
      FLOPs: 8 × 2 × B×T × d × inter × 2 + shared
           = 8 × 4 × B×T × d × inter + 2 × 4 × B×T × d × shared_inter

    Active FLOP comparison (ablation, d=1280):
      NanoChat: 16 × B×T × 1280² = 26.2M FLOPs/token
      NanoSeek: 8×4×B×T×1280×480 + 2×4×B×T×1280×960
             = 19.7M + 9.8M = 29.5M FLOPs/token
      MoE is ~13% more FLOPs (but has 5× more total parameters)

    Test matrix:
    ┌────────────────────┬──────────┬──────────┬──────────┬──────────┐
    │ Metric             │ Dense    │ MoE batched │ MoE fused │ Δ     │
    ├────────────────────┼──────────┼──────────┼──────────┼──────────┤
    │ Forward (ms)       │          │          │          │          │
    │ Backward (ms)      │          │          │          │          │
    │ Peak memory (MB)   │          │          │          │          │
    │ TFLOPS achieved    │          │          │          │          │
    │ MFU (%)            │          │          │          │          │
    │ FLOPs/token        │          │          │          │          │
    └────────────────────┴──────────┴──────────┴──────────┴──────────┘

    Key questions:
    1. How much overhead does routing+dispatch+combine add?
       overhead_pct = (moe_time - equivalent_dense_time) / equivalent_dense_time

    2. Does batched MoE (current) underutilize the GPU?
       Compare MoE MFU vs Dense MFU

    3. Where is the crossover?
       At what expert count does MoE overhead exceed benefit?
    """
    pass
```

### 3.3 Activation Functions: ReLU² vs SwiGLU

```python
def benchmark_activation_functions():
    """
    ReLU² (NanoChat) vs SwiGLU (NanoSeek) performance comparison.

    ReLU² (gpt.py:137):
      F.relu(x).square()
      - 2 ops: ReLU clamp + square
      - Produces sparse activations (~50% zero after ReLU)
      - Sparse activations = less memory bandwidth in backward
      - Used by Nemotron-4 340B for sparsity advantages

    SwiGLU (model.py:526-528):
      F.silu(self.w_gate(x)) * self.w_up(x)
      - 4 ops: sigmoid, multiply, multiply with up
      - Requires 2 projections (gate + up) vs 1 for ReLU²
      - Produces dense activations (no sparsity)
      - Used by LLaMA, DeepSeek, Mistral

    Performance comparison:
    ┌──────────────────────────┬──────────┬──────────┬──────────┐
    │ Metric                   │ ReLU²    │ SwiGLU   │ Ratio    │
    ├──────────────────────────┼──────────┼──────────┼──────────┤
    │ Forward (ms)             │          │          │          │
    │ Backward (ms)            │          │          │          │
    │ Activation memory (MB)   │          │          │          │
    │ Sparsity (% zero)        │ ~50%     │ ~0%      │          │
    │ FLOPs                    │ 2N       │ 5N       │ 2.5x     │
    │ Bandwidth (GB/s)         │          │          │          │
    └──────────────────────────┴──────────┴──────────┴──────────┘

    Key insight: ReLU² is faster per op AND sparser.
    But SwiGLU produces better loss at equal FLOPs (consistently shown in lit).
    The question is: does SwiGLU's quality advantage justify its 2.5x higher cost?

    For the NanoFuse project: if you fuse SwiGLU (as planned), the gap shrinks.
    Fused SwiGLU with backward recomputation = ~1.2x ReLU² cost (not 2.5x).
    """
    for N in [8192, 32768, 65536]:
        for D in [480, 768, 1280, 2048]:
            x = torch.randn(N, D, device='cuda', dtype=torch.bfloat16)

            # ReLU²
            relu2_ms = triton.testing.do_bench(lambda: F.relu(x).square(), warmup=25, rep=100)

            # SwiGLU (requires 2x input)
            gate = torch.randn(N, D, device='cuda', dtype=torch.bfloat16)
            up = torch.randn(N, D, device='cuda', dtype=torch.bfloat16)
            swiglu_ms = triton.testing.do_bench(lambda: F.silu(gate) * up, warmup=25, rep=100)

            # Bandwidth
            relu2_bytes = 2 * N * D * 2  # read + write, BF16
            swiglu_bytes = 3 * N * D * 2  # 2 reads + 1 write
            relu2_bw = relu2_bytes / (relu2_ms * 1e-3) / 1e9
            swiglu_bw = swiglu_bytes / (swiglu_ms * 1e-3) / 1e9

            print(f"  [{N:>5d}×{D:>4d}] ReLU²={relu2_ms:.3f}ms ({relu2_bw:.0f}GB/s) "
                  f"SwiGLU={swiglu_ms:.3f}ms ({swiglu_bw:.0f}GB/s) "
                  f"ratio={swiglu_ms/relu2_ms:.2f}x")
```

### 3.4 Optimizer Comparison

```python
def benchmark_optimizer_step():
    """
    Both NanoChat and NanoSeek use identical MuonAdamW.
    But the parameter distribution differs:

    NanoChat:
      AdamW params: embedding (~25M) + lm_head (~25M) + scalars (~320)
      Muon params: attention projections + MLP weights (~360M for d20)
      All params are DENSE matrices

    NanoSeek:
      AdamW params: embedding (~67M) + scalars
      Muon params: MLA projections + expert weights (~1.88B total)
      Expert weights: 64 × 3 = 192 matrices of varying shapes
      Muon stacking: 192 matrices stacked into single tensor for batched update

    The key difference: NanoSeek stacks 192 expert matrices.
    This SHOULD be efficient (single fused kernel on large tensor).
    But if expert shapes vary, padding waste could reduce efficiency.

    Measure:
    ┌──────────────────────────┬──────────┬──────────┐
    │ Metric                   │ NanoChat │ NanoSeek │
    ├──────────────────────────┼──────────┼──────────┤
    │ Optimizer step time (ms) │          │          │
    │ # Muon groups            │          │          │
    │ Stacked tensor sizes     │          │          │
    │ Polar Express time (ms)  │          │          │
    │ AdamW time (ms)          │          │          │
    │ Optimizer memory (GB)    │          │          │
    └──────────────────────────┴──────────┴──────────┘
    """
    pass
```

---

## Phase 4: MoE Overhead Decomposition (Days 19-24)

### 4.1 Isolate MoE Overhead

```python
def measure_moe_overhead():
    """
    THE critical measurement: What is the TOTAL overhead of MoE
    compared to a dense FFN with equal active parameters?

    Method:
    1. Create a dense FFN with N_active FLOPs matching NanoSeek's MoE
    2. Time both on identical input
    3. Difference = MoE overhead

    Dense equivalent for NanoSeek ablation:
      Active FLOPs = 8 × 2 × 1280 × 480 = 9.83M per token (routed)
                   + 2 × 2 × 1280 × 960 = 4.92M per token (shared)
                   = 14.75M FLOPs/token

      Dense equivalent: d_model=1280, inter_dim = 14.75M / (4 × 1280) = 2884
      So: Linear(1280, 2884) + activation + Linear(2884, 1280)

    Overhead components:
    ┌──────────────────────────┬──────────┬──────────────────────────────┐
    │ Component                │ Time (ms)│ What it measures              │
    ├──────────────────────────┼──────────┼──────────────────────────────┤
    │ Gate routing             │          │ sigmoid + group_topk + bias  │
    │ Token dispatch (sort)    │          │ argsort + scatter            │
    │ Padding waste            │          │ max_count vs mean_count      │
    │ Small-batch inefficiency │          │ 64 small GEMMs vs 1 large   │
    │ Token combine            │          │ weighted gather + scatter-add│
    │ Shared expert            │          │ 2 extra dense FFN passes     │
    │ Bias update              │          │ EMA load balance update      │
    ├──────────────────────────┼──────────┼──────────────────────────────┤
    │ TOTAL MoE OVERHEAD       │          │ MoE_time - dense_equiv_time  │
    │ OVERHEAD %               │          │ overhead / dense_equiv × 100 │
    └──────────────────────────┴──────────┴──────────────────────────────┘

    Expected: 30-60% overhead at ablation scale, single GPU.
    Literature: 15-25% overhead at production scale with fused kernels.
    Our target: reduce overhead to <20% via NanoFuse.
    """
    B, T, D = 2, 4096, 1280
    tokens = B * T  # 8192

    # Dense equivalent
    dense_ffn = nn.Sequential(
        nn.Linear(1280, 2884, bias=False),
        nn.SiLU(),
        nn.Linear(2884, 1280, bias=False),
    ).cuda().bfloat16()

    x = torch.randn(tokens, D, device='cuda', dtype=torch.bfloat16)

    dense_ms = triton.testing.do_bench(lambda: dense_ffn(x), warmup=25, rep=100)

    # NanoSeek MoE layer
    moe_layer = model.layers[0].moe
    moe_ms = triton.testing.do_bench(lambda: moe_layer(x), warmup=25, rep=100)

    overhead_ms = moe_ms - dense_ms
    overhead_pct = 100 * overhead_ms / dense_ms

    print(f"  Dense equivalent: {dense_ms:.2f} ms")
    print(f"  MoE layer:        {moe_ms:.2f} ms")
    print(f"  Overhead:         {overhead_ms:.2f} ms ({overhead_pct:.1f}%)")
```

### 4.2 Padding Waste Analysis

```python
def analyze_padding_waste():
    """
    NanoSeek's batched MoE path pads to max_count per expert.

    If routing is balanced: each expert gets ~1024 tokens (65536/64)
    If routing is imbalanced: some get 2000, others get 100
    Padding to max_count wastes compute on zeros.

    waste_ratio = max_count / mean_count
    If waste_ratio > 1.5, NanoSeek falls back to sequential path.

    Measure actual waste_ratio during training:
    1. Hook into Gate.forward() to capture expert_counts
    2. Compute waste_ratio = max(expert_counts) / mean(expert_counts)
    3. Track over 100 steps

    Expected:
    - Early training: waste_ratio ~1.3-1.8 (routing not yet learned)
    - Late training: waste_ratio ~1.1-1.3 (balanced routing)
    - With bias update: waste_ratio should decrease over time

    If waste > 30%: grouped GEMM (NanoFuse) saves this entirely
    because grouped GEMM handles variable expert sizes natively.
    """
    waste_ratios = []
    gini_coefficients = []

    def routing_hook(module, input, output):
        weights, indices = output[:2]
        flat_indices = indices.view(-1)
        counts = torch.bincount(flat_indices, minlength=64).float()
        waste_ratio = counts.max() / counts.mean()
        gini = compute_gini(counts)
        waste_ratios.append(waste_ratio.item())
        gini_coefficients.append(gini.item())

    model.layers[0].moe.gate.register_forward_hook(routing_hook)

    for step in range(100):
        train_step()

    print(f"  Waste ratio: mean={np.mean(waste_ratios):.3f}, "
          f"max={np.max(waste_ratios):.3f}, min={np.min(waste_ratios):.3f}")
    print(f"  Gini coefficient: mean={np.mean(gini_coefficients):.3f}")
    print(f"  Compute wasted by padding: {100*(np.mean(waste_ratios)-1):.1f}%")
```

### 4.3 Expert Utilization Heat Map

```python
def generate_expert_utilization_heatmap():
    """
    Visualize which experts are active across training.

    This is a NanoSeek-only diagnostic, but it directly informs
    performance optimization:
    - Dead experts (0 tokens) = wasted parameters + memory
    - Overloaded experts = bottleneck in batched dispatch
    - Balanced utilization = best for grouped GEMM

    Generate: [num_steps × num_experts] heatmap
    X-axis: training step
    Y-axis: expert index (0-63)
    Color: token count (log scale)

    Also track:
    - H_load (balance entropy) over time → should stay > 4 bits
    - I_spec (specialization MI) over time → should increase
    - Dead expert count over time → should be 0
    """
    pass
```

---

## Phase 5: FP8 Comparison (Days 25-28)

### 5.1 FP8 Eligibility Comparison

```python
def compare_fp8_eligibility():
    """
    Both codebases have FP8 implementations. Compare what gets converted.

    NanoChat FP8 (fp8.py:243-266):
      Filter: dims % 16 == 0 AND min_dim >= 128
      Applies to: ALL Linear layers meeting size requirements
      No special exclusions (no gate router, no MoE)

    NanoSeek FP8 (fp8.py:290-331):
      Filter: dims % 16 == 0 AND min_dim >= 128
      Exclusions:
        - Gate router (routing precision sacred)
        - Embeddings / LM head
        - Anything with "gate" and "router" in FQN
      Additional: MLA lora ranks (275, 90, 440, 143) mostly NOT 16-aligned
        → MLA projections may stay BF16

    Comparison:
    ┌────────────────────────┬──────────┬──────────┐
    │ Layer Type             │ NanoChat │ NanoSeek │
    ├────────────────────────┼──────────┼──────────┤
    │ Attention Q/K/V proj   │ FP8      │ Partial  │
    │ Attention output proj  │ FP8      │ FP8      │
    │ FFN up projection      │ FP8      │ FP8      │
    │ FFN down projection    │ FP8      │ FP8      │
    │ Gate router            │ N/A      │ BF16     │
    │ MLA lora projections   │ N/A      │ Depends  │
    │ Embeddings             │ BF16     │ BF16     │
    │ LM Head                │ BF16     │ BF16     │
    ├────────────────────────┼──────────┼──────────┤
    │ % params FP8-eligible  │ ~85%     │ ~60%     │
    │ Expected speedup       │ 1.3-1.5x │ 1.1-1.3x│
    └────────────────────────┴──────────┴──────────┘

    Key insight: NanoSeek gets LESS benefit from FP8 because:
    1. Gate router must stay BF16 (routing precision)
    2. MLA lora ranks are not 16-aligned (hardware constraint)
    3. Batched expert path uses torch.bmm (not torch._scaled_mm)

    This is where blockwise FP8 + grouped GEMM (NanoFuse) helps:
    grouped GEMM can use torch._scaled_mm directly (unlike bmm).
    """
    pass
```

### 5.2 FP8 Quantization Error Comparison

```python
def compare_fp8_quantization_error():
    """
    Compare quantization error across both architectures' weights.

    Hypothesis: MoE expert weights have DIFFERENT distribution than dense FFN weights.
    - Dense FFN: each weight matrix sees ALL tokens → broad distribution
    - MoE expert: each weight sees ~1/8 of tokens → potentially narrower distribution
    - Narrower distribution → better FP8 utilization → lower quantization error

    Test: quantize actual weights from both models, compare error.

    Also compare: do expert weights have more outliers than dense weights?
    (Outliers are the enemy of tensorwise FP8 scaling)
    """
    for name, param in nanochat_model.named_parameters():
        if 'weight' in name and param.dim() == 2:
            tw_err = tensorwise_quantization_error(param)
            bw_err = blockwise_quantization_error(param, block_size=128)
            outlier_pct = (param.abs() > 3 * param.std()).float().mean()
            print(f"  NanoChat {name:40s}: tw={tw_err:.6f} bw={bw_err:.6f} outliers={100*outlier_pct:.2f}%")

    for name, param in nanoseek_model.named_parameters():
        if 'weight' in name and param.dim() == 2:
            tw_err = tensorwise_quantization_error(param)
            bw_err = blockwise_quantization_error(param, block_size=128)
            outlier_pct = (param.abs() > 3 * param.std()).float().mean()
            print(f"  NanoSeek {name:40s}: tw={tw_err:.6f} bw={bw_err:.6f} outliers={100*outlier_pct:.2f}%")
```

---

## Phase 6: Inference Comparison (Days 29-32)

### 6.1 Prefill vs Decode Profiling

```python
def benchmark_inference():
    """
    Compare inference characteristics.

    NanoChat inference (engine.py:169-274):
      Prefill: single forward pass, builds KV cache
      Decode: token-by-token with KV cache
      KV cache: [n_layers, B, T, n_kv_head, head_dim]
      Memory per token: 2 × n_kv_head × head_dim × 2 bytes × n_layers

    NanoSeek inference (MLA absorbed mode, model.py:388-440):
      Prefill: forward with wkv_b expansion, builds COMPRESSED KV cache
      Decode: einsum-based weight absorption, no explicit K/V expansion
      KV cache: [n_layers, B, T, kv_lora_rank + qk_rope_dim]
      Memory per token: (kv_lora_rank + qk_rope_dim) × 2 bytes × n_layers

    Comparison:
    ┌────────────────────────┬──────────────────────┬──────────────────────┐
    │ Metric                 │ NanoChat             │ NanoSeek             │
    ├────────────────────────┼──────────────────────┼──────────────────────┤
    │ KV cache per token     │ 2×16×80×2 = 5120 B  │ (143+64)×2 = 414 B  │
    │ KV compression ratio   │ 1x                   │ 12.4x               │
    │ Prefill time (ms)      │                      │                      │
    │ Decode time/token (ms) │                      │                      │
    │ Tokens/sec (decode)    │                      │                      │
    │ Max context @ 24GB     │ ~___K tokens         │ ~___K tokens         │
    ├────────────────────────┼──────────────────────┼──────────────────────┤
    │ MoE decode overhead    │ N/A                  │ Route+dispatch/token │
    │ MTP speculative decode │ N/A                  │ Acceptance rate: _%  │
    └────────────────────────┴──────────────────────┴──────────────────────┘

    Key insight for inference:
    NanoSeek's MLA gives massive KV cache savings (12x).
    But MoE inference is expensive: ALL 1.95B params must be in memory,
    even though only 410M are active per token.

    For the RTX 4090 (24GB):
    NanoChat 410M: params=0.82GB + KV(4K context)=0.08GB = easy fit
    NanoSeek ablation: params=3.9GB + KV(4K context)=0.01GB = fits
    NanoSeek 1B: params=9.5GB + KV(4K context)=0.02GB = fits but tight

    At production scale (70B+ MoE), KV cache savings from MLA are crucial
    because expert weights consume most of GPU memory.
    """
    # Prefill benchmark
    for seq_len in [128, 512, 1024, 2048, 4096]:
        tokens = torch.randint(0, 32768, (1, seq_len), device='cuda')

        # NanoChat prefill
        nanochat_ms = triton.testing.do_bench(
            lambda: nanochat_model(tokens),
            warmup=5, rep=20
        )

        # NanoSeek prefill
        nanoseek_ms = triton.testing.do_bench(
            lambda: nanoseek_model(tokens),
            warmup=5, rep=20
        )

        print(f"  Prefill seq={seq_len}: NanoChat={nanochat_ms:.2f}ms, "
              f"NanoSeek={nanoseek_ms:.2f}ms, ratio={nanoseek_ms/nanochat_ms:.2f}x")

    # Decode benchmark (token-by-token with KV cache)
    # ...
```

---

## Phase 7: Consolidated Report (Days 33-35)

### 7.1 The Final Report Format

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              ARCHITECTURE COMPARISON: Dense vs MoE at Nano Scale            │
│              NanoChat (Dense GPT) vs NanoSeek (MoE DeepSeek V3.2)          │
│              Hardware: RTX 4090, BF16                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. EXECUTIVE SUMMARY                                                       │
│     Winner at equal FLOPs:    _____________ (by ___% BPB)                   │
│     Winner at equal time:     _____________ (by ___% BPB)                   │
│     Winner at equal cost:     _____________ (by ___% BPB)                   │
│     MoE overhead:             ___% (routing + dispatch + combine)           │
│     MLA KV savings:           ___x compression                              │
│                                                                             │
│  2. TRAINING EFFICIENCY                                                     │
│     ┌──────────────────────┬────────────┬────────────┐                      │
│     │ Metric               │ NanoChat   │ NanoSeek   │                      │
│     ├──────────────────────┼────────────┼────────────┤                      │
│     │ Tokens/sec           │            │            │                      │
│     │ MFU (%)              │            │            │                      │
│     │ BPB @ 1B tokens      │            │            │                      │
│     │ BPB @ 4B tokens      │            │            │                      │
│     │ BPB @ 8B tokens      │            │            │                      │
│     │ Peak memory (GB)     │            │            │                      │
│     │ Cost to train ($)    │            │            │                      │
│     └──────────────────────┴────────────┴────────────┘                      │
│                                                                             │
│  3. COMPONENT BREAKDOWN                                                     │
│     [Stacked bar chart: % time per component for each arch]                 │
│                                                                             │
│  4. SCALING ANALYSIS                                                        │
│     [Loss vs FLOPs curve at 3+ scales for each arch]                        │
│                                                                             │
│  5. MOE OVERHEAD DECOMPOSITION                                              │
│     [Pie chart: routing, dispatch, padding, combine, shared]                │
│                                                                             │
│  6. INFERENCE COMPARISON                                                    │
│     [Decode tokens/sec, KV cache memory, context length limits]             │
│                                                                             │
│  7. OPTIMIZATION OPPORTUNITIES                                              │
│     NanoChat: [what would speed it up most]                                 │
│     NanoSeek: [what NanoFuse project targets]                               │
│                                                                             │
│  8. RECOMMENDATIONS                                                         │
│     [When to choose dense vs MoE at this scale]                             │
│     [What changes at larger scale]                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 What the Report Should Conclude

Based on literature and architecture analysis, the expected findings:

```
At ablation scale (410M active):
─────────────────────────────────
1. NanoSeek (MoE) has SLOWER step time (MoE overhead: 30-60%)
2. NanoSeek has BETTER BPB per FLOP (10-30% more efficient)
3. At equal wall-clock: CLOSE RACE (overhead partially cancels efficiency)
4. NanoSeek uses 3-5× more memory (1.95B total vs 410M)
5. NanoSeek has 12× smaller KV cache (MLA)
6. NanoFuse would shift the balance toward NanoSeek
   by reducing MoE overhead from 30-60% to 15-25%

At 1B+ scale (expected, based on literature):
──────────────────────────────────────────────
1. MoE advantages grow with scale
2. Communication overhead dominates (multi-GPU)
3. MLA KV savings become critical for long context
4. Dense models hit memory walls earlier
5. MoE is definitively better at equal compute (1.8-3.4 accuracy points)
```

---

## Execution Checklist

```
Phase 1 (Days 1-4):
  □ Profile NanoChat at d16 (~410M) on RTX 4090
  □ Profile NanoSeek ablation on RTX 4090
  □ Generate side-by-side baseline table
  □ Verify both fit in 24GB (may need batch_size=1)

Phase 2 (Days 5-10):
  □ Match configs (equal active params)
  □ Implement BPB evaluation on shared eval set
  □ Run 500-step training for both
  □ Generate 3 comparison plots (FLOPs, time, cost)

Phase 3 (Days 11-18):
  □ MLA vs GQA attention benchmark
  □ Dense FFN vs MoE FFN benchmark
  □ ReLU² vs SwiGLU benchmark
  □ Optimizer step comparison

Phase 4 (Days 19-24):
  □ MoE overhead decomposition
  □ Padding waste analysis
  □ Expert utilization heatmap
  □ Dense-equivalent comparison

Phase 5 (Days 25-28):
  □ FP8 eligibility comparison
  □ FP8 quantization error comparison

Phase 6 (Days 29-32):
  □ Inference prefill benchmark
  □ Inference decode benchmark
  □ KV cache memory comparison
  □ Maximum context length comparison

Phase 7 (Days 33-35):
  □ Consolidated report with all tables and charts
  □ Recommendations for architecture selection
  □ Identification of optimization opportunities
```

---

## References

- Du et al., "Revisiting MoE and Dense Speed-Accuracy Comparisons" (2024) — fair comparison methodology
- Abnar et al., "Parameters vs FLOPs: Scaling Laws for Optimal Sparsity" (ICML 2025) — 6×N_active×D formula
- FLAME-MoE (CMU, 2025) — open platform with 64 experts, top-8 (same as NanoSeek)
- DeepSeek V3 Technical Report — MoE at frontier scale
- NanoChat README + dev/LOG.md — speedrun methodology and ClimbMix data
- NanoSeek CLAUDE.md — architecture rules and known bugs
