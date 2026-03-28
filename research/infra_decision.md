# RL Pipeline Infrastructure Decision (Refined)
## Senior Systems Engineer Recommendation
### Date: 2026-03-24 | Target: Qwen3.5-35B-A3B (MoE) for RL Post-Training
### Revision: v2 — Updated from NanoSeek 1B to Qwen3.5-35B-A3B based on architecture-pipeline alignment analysis

---

# 0. Why Qwen3.5-35B-A3B — Architecture-Pipeline Alignment

## Model Architecture [FACT — HuggingFace config, Qwen blog Feb 2026]

| Parameter | Value |
|-----------|-------|
| **Total parameters** | 35B |
| **Active parameters** | 3B (8.6% activation ratio) |
| **Layers** | 40 |
| **Hidden dimension** | 2,048 |
| **Vocab size** | 248,320 |
| **Context (native)** | 262,144 tokens |
| **Context (extensible)** | 1,010,000 tokens |
| **MoE: total experts** | 256 |
| **MoE: routed per token** | 8 |
| **MoE: shared experts** | 1 (always active) |
| **MoE: expert intermediate dim** | 512 |
| **Attention layout** | 10 × (3 × GatedDeltaNet-MoE + 1 × GatedAttention-MoE) |
| **Linear attention (75%)** | GatedDeltaNet: 32 V-heads, 16 QK-heads, head_dim=128 |
| **Softmax attention (25%)** | GatedAttention: 16 Q-heads, 2 KV-heads, head_dim=256, RoPE_dim=64 |
| **MTP** | Multi-Token Prediction enabled |
| **License** | Apache 2.0 |
| **Pre-training** | Part of Qwen3.5 family (training data volume not disclosed; Qwen3 was 36T tokens) |
| **Inference speed** | 179.5 TPS (Alibaba API) |

## Why This Model Is the Ideal Test Subject for the Three Pipelines

The Qwen3.5-35B-A3B is architecturally **almost identical** to the models built by all three labs we analyzed:

| Dimension | Qwen3.5-35B-A3B | MiniMax Text-01/M2.x | GLM-5 | Kimi K2 |
|-----------|-----------------|---------------------|-------|---------|
| **MoE experts** | 256 | 32 | 256 | 384 |
| **Active experts** | 8+1 shared | 2 (no shared) | 8+1 shared | 8+1 shared |
| **Attention type** | **75% linear + 25% softmax** | **87.5% lightning + 12.5% softmax** | MLA + DSA | MLA |
| **Linear attn variant** | GatedDeltaNet | Lightning Attention | N/A (DSA is sparse, not linear) | N/A |
| **Active params** | 3B | 45.9B (M1) / ~10B (M2.x) | 40B | 32.6B |
| **Total params** | 35B | 456B (M1) / ~230B (M2.x) | 744B | 1.04T |
| **Activation ratio** | 8.6% | 10% (M1) / ~4.3% (M2.x) | 5.4% | 3.1% |

**Critical architectural matches:**

1. **MoE with 256 experts** → Same as GLM-5. GSPO is critical. Routing noise at 256 experts is well-documented. Qwen team themselves designed GSPO specifically for this architecture.

2. **Hybrid linear + softmax attention** → Almost identical to MiniMax (75% linear vs 87.5% lightning). This means:
   - Rollout generation is O(n) for 75% of layers → dramatically faster than O(n²) for long sequences
   - MiniMax's CISPO stability fixes (Adam ε, FP32 LM head) apply directly
   - MiniMax's prefix tree merging insights are architecturally relevant

3. **8+1 expert routing** → Same routing pattern as GLM-5 and Kimi. IcePop, GSPO, and routing stability analyses transfer directly.

4. **GSPO is native** → Qwen3.5-397B-A17B (same architecture family, larger) uses GSPO for RL post-training [FACT — Qwen blog]. The algorithm was literally designed for this model family.

5. **Proven RL ceiling** → Qwen3-30B-A3B (predecessor) achieved 85.1% AIME25 with UloRL, surpassing Qwen3-235B-A22B (22B active). At 3B active, this model is above the 1.5B RL viability threshold.

**This is not a random model choice — it's the model whose architecture most precisely matches the three frontier pipelines we analyzed, at a scale where RL training is affordable.**

---

# 1. Engineering Feasibility Ranking (Revised for 35B MoE)

## Rank 1 (Most Feasible): MiniMax CISPO + GSPO Hybrid

**Feasibility Score: 9/10**

**Why it ranks first — now even MORE strongly than before:**

1. **Architecture match is near-perfect**: Qwen3.5-35B-A3B uses hybrid linear+softmax attention, same as MiniMax. CISPO was designed for and validated on this exact architecture class.

2. **GSPO is native**: The Qwen team designed GSPO (arXiv:2507.18071) specifically for their MoE models. Using GSPO on Qwen3.5-35B-A3B is the intended use case, not a hack.

3. **Framework support**: Unsloth, Swift (Alibaba's own), and OpenRLHF all support Qwen3.5 models with GRPO/GSPO + LoRA. No custom framework needed.

4. **LoRA RL is proven at this scale**: Kimi K2 published LoRA-based RL achieving same quality with 10% GPU footprint. At 35B total, LoRA is not optional — it's required for practical training.

5. **Stability fixes transfer directly**: Adam ε=1e-15, FP32 LM head, β2=0.95 — all discovered on MiniMax's hybrid-attention MoE. Same architecture → same fixes.

**What to adopt (in implementation order):**
```python
# 1. QLoRA setup for 35B MoE
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(
    r=64,                          # LoRA rank
    lora_alpha=128,                # scaling factor
    target_modules=[               # Apply to attention + MoE projections
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention
        "gate_proj", "up_proj", "down_proj",       # expert FFN
    ],
    lora_dropout=0.0,              # No dropout for RL (need gradient signal)
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
# Trainable params: ~200-500M out of 35B (~1.4%)

# 2. GSPO + CISPO hybrid loss (~30 lines)
def hybrid_gspo_cispo_loss(cur_logps, old_logps, rewards, mask, tau=0.05):
    # GSPO: sequence-level IS ratio (MoE-safe, designed for Qwen MoE)
    token_log_ratios = (cur_logps - old_logps) * mask
    seq_log_ratio = token_log_ratios.sum(-1) / (mask.sum(-1) + 1e-8)
    seq_ratio = torch.exp(seq_log_ratio)

    # CISPO: detach IS weights (preserve all-token gradients)
    clamped_ratio = torch.clamp(seq_ratio, max=6.0).detach()

    # Group-relative advantage (no value network)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # Policy gradient (through LoRA parameters only)
    policy_loss = -(clamped_ratio * advantages * seq_log_ratio).mean()

    # Kimi L2 regularization (symmetric KL proxy)
    reg_loss = (tau / 2) * (token_log_ratios ** 2).sum(-1).mean()

    return policy_loss + reg_loss

# 3. Stability fixes (CRITICAL — from MiniMax, validated on hybrid-attention MoE)
optimizer = AdamW(
    model.parameters(),       # Only LoRA params are trainable
    lr=5e-6,                  # Lower LR for LoRA (vs 1e-5 for full-param)
    eps=1e-15,                # NOT 1e-8 — preserves per-param adaptivity
    betas=(0.9, 0.95),        # β2=0.95 tracks RL gradient non-stationarity
)
model.lm_head = model.lm_head.float()  # FP32 LM head prevents IS ratio sign reversal
```

**Time to first working experiment**: 1-2 days (frameworks already support this model)

## Rank 2 (Medium Feasibility): Kimi Squared-Loss Mirror Descent

**Feasibility Score: 6/10** (downgraded from 7 due to 35B scale)

**Why it drops slightly:**
1. Squared-loss mirror descent + Muon is untested on hybrid linear+softmax attention. Muon's Newton-Schulz orthogonalization may behave differently for GatedDeltaNet vs standard attention parameters.
2. At 35B scale with LoRA, the squared-loss objective's interaction with low-rank updates is poorly understood. CISPO's `.detach()` on the other hand has no such interaction — gradients flow cleanly through `log π_θ`.
3. τ (KL coefficient) is still [UNKNOWN]. More hyperparameter risk at larger scale.

**Still adopt:**
- PTX auxiliary loss (anti-forgetting)
- Curriculum sampling (proportional to `1 - success_rate`)
- L2 log-ratio regularization (as secondary regularization alongside GSPO)

## Rank 3 (Hardest): GLM-5 Full Pipeline (Slime + IcePop)

**Feasibility Score: 5/10** (upgraded from 4 — GLM-5 uses 256 experts like Qwen3.5)

**Why it upgrades slightly:**
- GLM-5 uses exactly 256 experts like Qwen3.5-35B-A3B — IcePop's MoE stability insights are directly transferable
- Slime framework explicitly supports Qwen3 series [FACT — Slime README]
- APRIL optimization is more relevant at 35B (rollout cost is higher)

**Still not recommended as primary because:**
- 5-stage pipeline is still overkill (3B active doesn't have capacity for 5 separate RL stages)
- IcePop solves async mismatch — we'll use colocated or framework-managed sync
- Cross-stage distillation risk (LiveCodeBench 63% regression warning)
- Better to use Slime's GSPO support but skip the full 5-stage orchestration

**Newly relevant to adopt:**
- APRIL concept (rollout is more expensive at 35B, throughput optimization matters)
- Asymmetric clipping (ε_low=0.2, ε_high=0.28) can compose with GSPO

---

# 2. Infrastructure Risk Analysis (Revised for 35B MoE)

## GSPO + CISPO on Qwen3.5-35B-A3B (Recommended)

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **LoRA rank insufficient for RL** (policy can't express needed updates at r=64) | LOW-MEDIUM | MEDIUM (RL plateaus) | Increase rank to 128-256; or target more modules (router projections); Kimi showed LoRA RL works at scale |
| **CISPO entropy collapse** (late-stage degeneration) | MEDIUM | HIGH | Monitor entropy every step; DISPO 4-regime clipping as fallback; entropy bonus 0.01-0.05 |
| **Router freeze vs. train debate** (should LoRA touch the MoE router?) | MEDIUM | MEDIUM (routing collapse or stagnation) | Start with frozen router; ablate unfrozen later. GLM-5 freezes DSA indexer; analogous principle |
| **GatedDeltaNet gradient dynamics under RL** [UNKNOWN] | LOW-MEDIUM | MEDIUM | 75% of layers use linear attention — RL gradient flow through recurrent state is less studied. Monitor per-layer gradient norms |
| **QLoRA quantization artifacts in IS ratio** | LOW | MEDIUM (noisy IS ratios from quantized logits) | FP32 LM head mitigates the critical path; QeRL paper shows quantized RL can match full-precision |
| **Memory pressure during rollout** (35B model + KV cache at 262K context) | LOW | HIGH (OOM) | GatedDeltaNet KV cache is O(n) for 75% layers; limit context to 4-8K for math RL, 32K for agentic |
| **Adam ε=1e-15 with LoRA** (gradient magnitudes differ from full-param) | VERY LOW | MEDIUM | LoRA gradients are denser than full-param; ε=1e-15 is conservative (safe). Verify with gradient histogram |

## New Risks Not Present at 1B Scale

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Rollout cost dominance** (35B inference is 7× slower than 5B) | HIGH | HIGH (training takes weeks instead of days) | Use vLLM/SGLang with FP8 quantization for inference; GatedDeltaNet O(n) helps; batch aggressively |
| **LoRA + MoE interaction** (RO-GRPO paper: LoRA-MoE GRPO needs routing-aware rewards) | MEDIUM | MEDIUM | Monitor routing distribution; use GSPO (sequence-level) which is less sensitive to per-token routing than GRPO |
| **Framework compatibility** (hybrid GDN+GQA attention is newer than standard transformers) | LOW-MEDIUM | HIGH (inference engine bugs) | Qwen3.5 is already supported by vLLM, SGLang, Unsloth [FACT]; test rollout quality before RL |
| **Reference model memory** (need old policy log-probs → two copies during training?) | MEDIUM | HIGH (2× model memory) | Use LoRA: reference = base model (frozen); current = base + LoRA adapters. Only LoRA params differ. Single model, zero overhead |

---

# 3. Recommended Infrastructure Strategy (Revised)

## Decision: GSPO + CISPO Hybrid via LoRA on Qwen3.5-35B-A3B-Base

### Rationale — Why This Exact Configuration

| Component | Source | Why | Confidence |
|-----------|--------|-----|------------|
| **Base model** | Qwen3.5-35B-A3B-Base | Architecture matches all 3 analyzed pipelines; GSPO native; proven RL ceiling from predecessor (85.1% AIME25); Apache 2.0 | 95% |
| **Loss function** | GSPO (Qwen, arXiv:2507.18071) + CISPO `.detach()` (MiniMax) | GSPO designed for this exact model family; `.detach()` preserves all-token gradients | 90% |
| **Training method** | QLoRA (4-bit base + LoRA r=64) | 35B total params → full-param RL needs ~430 GB; QLoRA fits on single A100 80GB | 90% |
| **Optimizer** | AdamW (ε=1e-15, β2=0.95) | MiniMax-proven for hybrid-attention MoE RL | 95% |
| **Precision** | NF4 base + BF16 LoRA + FP32 LM head | Memory-optimal; FP32 LM head prevents IS ratio sign reversal | 95% |
| **Regularization** | L2 log-ratio (Kimi, τ=0.05) | Symmetric KL proxy; stronger than CISPO's implicit clip alone | 75% |
| **Anti-forgetting** | PTX auxiliary loss (Kimi) + LoRA inherent regularization | LoRA's low-rank constraint naturally prevents catastrophic forgetting; PTX adds explicit anchor | 85% |
| **Curriculum** | Sample proportional to `1 - success_rate` (Kimi) | Scale-agnostic, well-proven | 85% |
| **Stability** | Repetition detection (MiniMax, 3K tokens) | Defense against degenerate outputs | 90% |
| **Router** | Frozen during RL (GLM-5 principle) | Prevents expert collapse under noisy RL gradients | 75% |
| **Framework** | Unsloth or OpenRLHF (with vLLM inference) | Both support Qwen3.5 + LoRA + GRPO/GSPO natively | 85% |

### Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│        Qwen3.5-35B-A3B RL Training Loop (QLoRA)                    │
│        GPU Options: 1×A100 80GB (QLoRA) or 2×A100 (BF16+LoRA)     │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │ PHASE 1: ROLLOUT (Inference Mode via vLLM/SGLang)          │     │
│  │                                                            │     │
│  │  Base model: 35B params, 3B active per token               │     │
│  │  Quantization: FP8 (inference) or NF4 (if memory tight)   │     │
│  │  LoRA adapters: merged for inference (no overhead)         │     │
│  │                                                            │     │
│  │  Attention: 75% GatedDeltaNet O(n) + 25% GQA O(n²)       │     │
│  │  → Inference speed: ~150-180 TPS at short context          │     │
│  │  → KV cache: minimal (linear attention dominates)          │     │
│  │                                                            │     │
│  │  Generate G=8-16 completions per prompt                    │     │
│  │  Save: prompts, completions, per-token log_probs, rewards  │     │
│  │  Repetition detection (3K tokens > 0.99 prob → R=0)        │     │
│  │                                                            │     │
│  │  Memory: ~18GB (NF4) + KV cache ~2-4GB = ~22GB            │     │
│  │  Headroom: 58GB on A100 80GB → large batch possible        │     │
│  └────────────────────────────┬───────────────────────────────┘     │
│                               │ (mode transition ~30-60s)          │
│  ┌────────────────────────────▼───────────────────────────────┐     │
│  │ PHASE 2: TRAINING (Gradient Mode, LoRA only)               │     │
│  │                                                            │     │
│  │  Algorithm: GSPO + CISPO hybrid                            │     │
│  │                                                            │     │
│  │  Loss computation:                                         │     │
│  │    1. token_log_ratios = cur_logps - old_logps             │     │
│  │    2. seq_ratio = exp(mean(token_log_ratios per sequence)) │     │
│  │    3. clamped = clamp(seq_ratio, max=6.0).detach()         │     │
│  │    4. advantages = (R - mean(R)) / (std(R) + 1e-8)        │     │
│  │    5. policy_loss = -(clamped × adv × seq_log_ratio)       │     │
│  │    6. reg_loss = (τ/2) × mean(token_log_ratios²)          │     │
│  │    7. ptx_loss = -log_prob(curated_sft_data)               │     │
│  │    8. total = policy + reg + λ_ptx × ptx                   │     │
│  │                                                            │     │
│  │  Optimizer: AdamW (ε=1e-15, β1=0.9, β2=0.95)             │     │
│  │  Only LoRA params updated (~300-500M of 35B)               │     │
│  │  Router: FROZEN (GLM-5 principle)                          │     │
│  │                                                            │     │
│  │  Memory breakdown (QLoRA on A100 80GB):                    │     │
│  │    Base model (NF4):        ~18 GB                         │     │
│  │    LoRA adapters (BF16):    ~0.6-1.0 GB                    │     │
│  │    LoRA optimizer (FP32):   ~2.4-4.0 GB                    │     │
│  │    LoRA gradients (BF16):   ~0.6-1.0 GB                    │     │
│  │    Activations (grad ckpt): ~6-10 GB                       │     │
│  │    FP32 LM head:           ~2.0 GB                         │     │
│  │    Misc (buffers, etc):     ~2-4 GB                        │     │
│  │    ────────────────────────────────────                     │     │
│  │    TOTAL:                   ~32-40 GB                       │     │
│  │    Headroom:                40-48 GB ✓                      │     │
│  └────────────────────────────┬───────────────────────────────┘     │
│                               │                                     │
│  ┌────────────────────────────▼───────────────────────────────┐     │
│  │ MONITORING (Continuous, WandB)                              │     │
│  │                                                            │     │
│  │  Policy entropy — detect collapse (alert < 3.0)            │     │
│  │  Expert activation distribution — detect dead experts      │     │
│  │  Router Gini coefficient — detect concentration            │     │
│  │  IS weight histogram — detect drift / clamp saturation     │     │
│  │  Reward statistics (mean, std, min, max)                   │     │
│  │  Per-domain loss (math / code / general)                   │     │
│  │  LoRA weight norms — detect training instability           │     │
│  │  GatedDeltaNet vs GatedAttention gradient ratio            │     │
│  │                                                            │     │
│  │  NOVEL METRIC: Expert specialization under RL              │     │
│  │  → Does RL preserve or destroy expert routing patterns?    │     │
│  │  → First systematic study on 256-expert model with LoRA RL │     │
│  └────────────────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────────────┘
```

### Memory Budget — Three Hardware Options

#### Option A: QLoRA on Single A100 80GB (Recommended — Cheapest)

| Component | Phase | Size |
|-----------|-------|------|
| Base model (NF4 quantized) | Both | 18 GB |
| LoRA adapters (BF16, r=64) | Both | 0.8 GB |
| FP32 LM head | Both | 2.0 GB |
| LoRA optimizer states (AdamW FP32) | Train | 3.2 GB |
| LoRA gradients (BF16) | Train | 0.8 GB |
| Activations (gradient checkpointing) | Train | 8 GB |
| KV cache (inference, G=8, 4K ctx) | Inference | 3 GB |
| Prompt/completion buffer | Both | 2 GB |
| **Total (train)** | | **~35 GB** (45 GB headroom) |
| **Total (inference)** | | **~26 GB** (54 GB headroom) |

**Verdict**: Comfortable fit. Large headroom enables G=16 or longer sequences.

#### Option B: BF16 + LoRA on 2×A100 80GB (Better Quality)

| Component | Phase | Size (per GPU with FSDP) |
|-----------|-------|------|
| Base model (BF16, sharded) | Both | 35 GB/GPU |
| LoRA adapters | Both | 0.5 GB/GPU |
| FP32 LM head | Both | 1 GB/GPU |
| LoRA optimizer | Train | 2 GB/GPU |
| Activations | Train | 6 GB/GPU |
| **Total (train)** | | **~45 GB/GPU** (35 GB headroom) |

**Verdict**: No quantization artifacts. Better gradient quality. ~2× cost.

#### Option C: Full-Parameter RL on 8×H100 (Research-Grade, Expensive)

| Component | Size (total) | Per GPU (ZeRO-3) |
|-----------|-------------|-------------------|
| Weights (BF16) | 70 GB | 8.75 GB |
| Optimizer (AdamW FP32) | 280 GB | 35 GB |
| Gradients (BF16) | 70 GB | 8.75 GB |
| Activations | 64 GB | 8 GB |
| **Total** | **~484 GB** | **~61 GB/GPU** |

**Verdict**: Fits on 8×H100. Maximum quality but ~$22/hr on RunPod. Only if LoRA RL proves insufficient.

### Cost Estimates (March 2026)

| Setup | Hardware | Hourly Cost | RL Duration | Total Cost |
|-------|----------|------------|-------------|------------|
| **QLoRA (recommended)** | 1×A100 80GB (RunPod) | $1.39/hr | 200-400 hrs | **$280-$560** |
| QLoRA (budget) | 1×A6000 48GB (RunPod) | $0.58/hr | 300-600 hrs | **$174-$348** |
| BF16 + LoRA | 2×A100 80GB (RunPod) | $2.78/hr | 150-300 hrs | **$417-$834** |
| Full-param | 8×H100 SXM (RunPod) | $21.52/hr | 80-160 hrs | **$1,722-$3,443** |

**Reference**: MiniMax M1 (456B MoE, full-param RL) cost $534K on 512 H800. Qwen3.5-35B is 13× smaller total params → naively ~$41K. But with QLoRA (only ~1.4% params trained), we get another ~70× reduction → **$280-560 is realistic**.

### Batching Strategy

| Parameter | Value | Why |
|-----------|-------|-----|
| Group size (G) | 8-16 | G=8 for QLoRA (memory constrained); G=16 for BF16+LoRA |
| Gradient steps per generation (K) | 1-2 | K=1 prevents IS drift; K=2 is safe with GSPO sequence-level averaging |
| Batch size (prompts per step) | 4-8 | Limited by activation memory during training |
| Sequence length (math RL) | 4,096-8,192 | Short sequences for verifiable math rewards |
| Sequence length (agentic RL) | 16,384-32,768 | Longer for code/tool-use; GatedDeltaNet handles this efficiently |
| Gradient accumulation | 4-8 steps | Simulates larger effective batch without memory cost |
| LoRA rank (r) | 64 | Start here; increase to 128 if RL plateaus |

### Communication Strategy

**Option A (QLoRA, single GPU)**: No communication needed. Entire loop is local.

**Option B (BF16+LoRA, 2 GPUs)**: FSDP sharding across 2 GPUs on same node. NVLink bandwidth is sufficient (600+ GB/s on A100 pairs). Communication overhead: <5%.

**Option C (Full-param, 8 GPUs)**: ZeRO-3 across 8 GPUs. Standard distributed training. Communication is ~10-15% overhead at this scale.

---

# 4. Justification (First Principles, Revised)

## Why GSPO Is Non-Negotiable for This Model

**System constraint**: Qwen3.5-35B-A3B has 256 experts with top-8+1 routing. After each gradient update, the router may reassign ~10% of tokens to different experts (per RSPO paper, arXiv:2510.23027). With 256 experts, this creates massive token-level IS ratio noise.

**GSPO (sequence-level IS)** was designed by the Qwen team specifically for their MoE models (arXiv:2507.18071):
```
seq_ratio = exp( (1/|y|) × Σ_t (log π_θ(y_t) - log π_old(y_t)) )
```
The geometric mean across all tokens averages out individual routing perturbations. With 256 experts, this averaging is critical — 8 routing decisions per token × hundreds of tokens = massive variance without sequence-level aggregation.

**GRPO would fail**: Qwen's own blog states: "GRPO necessitates the Routing Replay training strategy for the normal convergence of MoE RL, while GSPO has obviated the need for this strategy." Routing Replay is expensive (replay the exact routing decisions from rollout during training). GSPO eliminates this engineering complexity entirely.

**Confidence**: 95% — this is the algorithm the model's creators designed for it.

## Why QLoRA over Full-Parameter RL

**System constraint**: 35B total params × 8 bytes (AdamW FP32 states) = 280 GB for optimizer alone. Single A100 80GB cannot fit this.

**Why LoRA works for RL**:
1. **Kimi K2 validation**: "LoRA-based RL achieves same quality with 10% GPU footprint" [FACT — K2 paper, Macaron AI / Mind Lab]
2. **Reference model is free**: With LoRA, the reference policy = base model (frozen). Current policy = base + LoRA. No need to keep two copies — the base model IS the reference. This eliminates a major memory cost.
3. **RL updates are small**: Policy gradient updates during RL are much smaller than pre-training gradient updates. A rank-64 LoRA can express the needed policy adjustments for reasoning tasks.
4. **Implicit regularization**: LoRA's low-rank constraint acts as implicit regularization against catastrophic policy shifts — it literally cannot make large changes to the base model. This complements GSPO's explicit KL constraint.

**When LoRA might fail**: If the RL task requires fundamentally reshaping the model's representations (e.g., learning a new language or modality). For math/code reasoning RL, the base model already has the representations — RL just sharpens the policy. LoRA is sufficient.

**Escape hatch**: If LoRA RL plateaus, try:
1. Increase rank: r=64 → r=128 → r=256
2. Target more modules (include router projections)
3. Move to Option B (BF16 + LoRA on 2×A100)
4. Last resort: Option C (full-param on 8×H100)

## Why Frozen Router During RL

**System constraint**: The MoE router learned expert specialization during 35B params of pre-training on potentially >36T tokens. RL training involves orders of magnitude fewer gradient steps with noisier signals.

**Risk of unfrozen router**:
- RL gradients can push the router toward a few "safe" experts that consistently produce correct answers
- This kills expert diversity → capacity collapse → the 35B model behaves like a 3B dense model
- Dead expert rate increases; I_spec (specialization MI) drops; Gini coefficient spikes

**GLM-5 precedent**: GLM-5 freezes the DSA indexer during RL. The analogous operation for MoE is freezing the router.

**When to unfreeze**: Only after validating that RL with frozen router produces meaningful improvement. Then ablate with unfrozen router and compare routing statistics.

## Why PTX + LoRA over GLM-5's Cross-Stage Distillation

**At 3B active parameters**, the model has limited capacity. Multi-stage RL (5 stages like GLM-5) risks each stage overwriting the previous one. Cross-stage distillation tries to recover degraded capabilities, but:
- Adds ~70% complexity (teacher checkpoint management, stop-gradient mechanics)
- GLM-5 still suffered LiveCodeBench 63% regression despite distillation
- At small scale, single-stage RL + PTX is more robust

**LoRA provides inherent anti-forgetting**: The base model weights never change. All RL learning is captured in the LoRA adapters. If the LoRA update hurts a capability, you can reduce the LoRA merge ratio or use multi-LoRA strategies.

---

# 5. Implementation Notes (Senior Engineer Perspective, Revised)

## What to Build First (Priority Order)

### Day 1: Environment Setup + Model Verification

```bash
# Install framework (choose one)
pip install unsloth  # Recommended for Qwen3.5 + LoRA + GRPO
# OR
pip install openrlhf  # If need more control over RL loop

# Verify model loads and generates correctly
python -c "
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    'Qwen/Qwen3.5-35B-A3B-Base',  # Base model, not instruct
    max_seq_length=4096,
    load_in_4bit=True,  # QLoRA
)
# Test generation
inputs = tokenizer('Solve: What is 2+2?', return_tensors='pt')
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
"
```

### Day 2: LoRA Setup + GSPO Loss Implementation

```python
# LoRA configuration for RL
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",    # Attention
        "gate_proj", "up_proj", "down_proj",          # Expert FFN
        # Do NOT include router/gate parameters — keep frozen
    ],
    lora_dropout=0.0,
    bias="none",
)

# GSPO + CISPO hybrid loss
def hybrid_gspo_cispo_loss(cur_logps, old_logps, rewards, mask, tau=0.05):
    token_log_ratios = (cur_logps - old_logps) * mask
    seq_log_ratio = token_log_ratios.sum(-1) / (mask.sum(-1) + 1e-8)
    seq_ratio = torch.exp(seq_log_ratio)
    clamped_ratio = torch.clamp(seq_ratio, max=6.0).detach()
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    policy_loss = -(clamped_ratio * advantages * seq_log_ratio).mean()
    reg_loss = (tau / 2) * (token_log_ratios ** 2).sum(-1).mean()
    return policy_loss + reg_loss

# Stability fixes
optimizer = AdamW(model.parameters(), lr=5e-6, eps=1e-15, betas=(0.9, 0.95))
model.lm_head = model.lm_head.float()  # FP32 LM head
```

### Day 3: Rollout Pipeline + Reward Functions
- Merge LoRA for inference (Unsloth supports fast merge/unmerge)
- Generate G=8 completions per prompt using merged model
- Binary rewards: math (exact match on GSM8K/MATH-500), code (test execution)
- Save per-token log-probs from the MERGED model (reference = base = unmerged)
- Repetition detection: 3K consecutive tokens > 0.99 prob → R=0, truncate
- Curriculum sampling: weight prompts by `1 - success_rate`

### Day 4-5: Training Loop + Monitoring + First Run
- Colocated loop: merge LoRA → generate → unmerge → train → repeat
- WandB logging: loss, entropy, reward, expert activation histogram, LoRA weight norms
- PTX auxiliary loss: mix 10-20% curated SFT data per batch
- 100-step smoke test → 1000-step validation

## What to Test Early

1. **Entropy monitoring** (highest priority): Same as before — CISPO entropy collapse is the #1 failure mode. Alert threshold: entropy < 3.0.

2. **LoRA rank sufficiency**: After 500 steps, check if reward is still improving. If plateau → increase rank to 128. If still plateau → the base model's pass@256 ceiling may be reached (RLVR limitation).

3. **Expert activation distribution**: With frozen router, experts should maintain their pre-training specialization. Log expert activation counts every 100 steps. If distribution shifts significantly, RL is indirectly affecting routing through representation changes.

4. **GatedDeltaNet vs GatedAttention gradient norms**: This is novel territory — no one has published RL training behavior for GatedDeltaNet layers. Log per-layer-type gradient norms. If GDN layers show gradient instability, consider freezing them and only training GatedAttention + MoE layers.

5. **Quantization effect on IS ratios**: NF4 quantization introduces noise in logit computation. Compare IS ratio distributions from QLoRA vs BF16 (if you have 2×A100 to compare). FP32 LM head should handle the critical path, but verify.

## What to Monitor

| Metric | Frequency | Alert Threshold | Action |
|--------|-----------|----------------|--------|
| Policy entropy | Every step | < 3.0 | Add entropy bonus (0.01-0.05); reduce lr |
| Reward mean | Every step | No improvement for 500 steps | Check LoRA rank; check FP32 LM head; check IS weights |
| Expert activation histogram | Every 100 steps | Any expert with 0 activations | Check if expert was dead pre-RL; if newly dead, router may be drifting |
| Router Gini coefficient | Every 100 steps | > 0.8 (concentrated) | Verify router is truly frozen; check indirect routing effects |
| IS weight clamp rate | Every step | > 30% at max clamp | Reduce ε_high from 6.0 to 3.0-4.0 |
| LoRA weight L2 norm | Every 100 steps | Growth > 10× from init | Reduce lr; add weight decay to LoRA |
| GatedDeltaNet gradient norm | Every 100 steps | > 10× GatedAttention gradient | GDN layers may be unstable; consider selective layer freezing |
| Loss magnitude | Every step | NaN or Inf | Reduce lr; check numerical stability; verify FP32 accumulation |
| Repetition rate | Every step | > 5% truncated | Model degenerating; check reward function and entropy |
| Memory usage | Every 100 steps | > 75 GB peak | Reduce batch size or sequence length |

## Common Pitfalls (Revised for 35B MoE + LoRA)

1. **Forgetting to freeze the router**: If LoRA target_modules accidentally includes the router/gate parameters, you'll update routing during RL. At 256 experts with noisy RL gradients, this can kill expert diversity within hundreds of steps. **Explicitly exclude router parameters from LoRA.**

2. **Using GRPO instead of GSPO**: The Qwen team themselves say GRPO requires Routing Replay for MoE convergence. GSPO eliminates this. Using GRPO on Qwen3.5-35B is fighting the architecture.

3. **Forgetting `eps=1e-15`**: Same as before. Even MORE critical with LoRA — LoRA gradients can be very small for certain modules, and default ε=1e-8 will zero them out.

4. **Not merging LoRA for inference**: During rollout, the model should have LoRA merged into base weights for correct generation. During training, keep separate for reference policy computation. Frameworks like Unsloth handle this automatically.

5. **Context length mismatch**: Qwen3.5-35B supports 262K native, but RL for math/code should use 4K-8K to keep memory manageable. Don't accidentally generate at 32K+ context — KV cache explodes for the 25% softmax layers.

6. **Ignoring the linear attention gradient question**: GatedDeltaNet layers have recurrent state that accumulates across tokens. RL gradients flowing backward through this recurrence may behave differently than softmax attention. **This is uncharted territory** — monitor closely.

---

# 6. Fast Validation Plan (Revised)

## Experiment E0: GSPO on Qwen3.5-35B-A3B-Base with QLoRA (2-4 hours, 1×A100 80GB)

**Purpose**: Verify QLoRA RL works on this exact model before committing to longer runs.

```python
# Using Unsloth or TRL with custom GSPO loss
# 200 steps on GSM8K (grade school math)
# G=8, K=1, r=64, lr=5e-6
# Binary reward: exact match answer extraction
# Expected: reward ↑ from baseline ~60-70% to ~70-80%
```

**Metrics to track**:
- Reward trend (must be upward within 200 steps)
- Entropy (must stay > 3.0)
- IS weight distribution (must not all be at clamp)
- Memory peak (must be < 40 GB for QLoRA)

**Pass criteria**: Reward improves AND entropy > 3.0 AND no OOM.
**Fail action**: Debug loss function; check LoRA target modules; verify FP32 LM head.

## Experiment E1: Extended RL on MATH-500 (48-96 hours, 1×A100 80GB)

**Purpose**: Validate full pipeline produces meaningful reasoning improvement.

```python
# 3000 steps, G=8, K=1, MATH-500 + competition math
# QLoRA r=64, lr=5e-6, AdamW (eps=1e-15, beta2=0.95)
# GSPO + CISPO hybrid loss + PTX (10% SFT data)
# Curriculum: proportional to 1-success_rate
```

| Metric | Baseline | Target (post-3000 steps) |
|--------|----------|-------------------------|
| MATH-500 pass@1 | Measure (expect ~70-80%) | +5-15% improvement |
| MATH-500 pass@8 | Measure | Should be > pass@1 (RL ceiling indicator) |
| AIME 2024 pass@1 | Measure (expect ~40-60%) | Any improvement |
| Entropy | Measure | > 3.0 throughout |
| Expert activation Gini | Measure | Within 0.1 of baseline |
| Dead expert count | Measure | No increase |
| Peak memory | — | < 40 GB (QLoRA) |

**Pass criteria**:
1. MATH-500 pass@1 improves ≥ 3% absolute
2. Entropy > 3.0 throughout
3. No expert collapse (Gini stable, no new dead experts)
4. No OOM or NaN

**Fail actions**:
- No improvement: Increase LoRA rank to 128; check base model pass@8 (RLVR ceiling)
- Entropy collapse: Add entropy bonus 0.01-0.05; try DISPO 4-regime clipping
- Expert collapse: Router leak — verify router params are truly frozen
- OOM: Reduce G to 4; enable more aggressive gradient checkpointing

## Experiment E2: GSPO vs GRPO vs CISPO Ablation (3×48 hours)

**Purpose**: Empirically validate GSPO superiority on 256-expert MoE.

```python
# Same setup as E1, three runs:
# Run A: GSPO (sequence-level IS + .detach()) — expected winner
# Run B: CISPO (token-level IS + .detach()) — MiniMax baseline
# Run C: GRPO (token-level IS, no .detach()) — vanilla baseline
# 3000 steps each, same seed, same prompts
```

**Expected result**: GSPO > CISPO > GRPO on this model. If confirmed, this validates the Qwen team's algorithm design for their own architecture.

## Experiment E3: LoRA Rank Ablation (3×48 hours, optional)

**Purpose**: Find optimal LoRA rank for RL on this model.

```python
# Run A: r=32 (minimal capacity)
# Run B: r=64 (default)
# Run C: r=128 (high capacity)
# Same GSPO loss, same data, same seed
```

**Expected result**: r=64 should be sufficient. r=128 marginal improvement. r=32 may plateau early.

---

# Summary (Revised)

| Decision | Choice | Confidence |
|----------|--------|------------|
| **Base model** | Qwen3.5-35B-A3B-Base (35B total, 3B active, 256 experts) | 95% |
| **Loss function** | GSPO + CISPO hybrid (.detach() on sequence-level IS) | 90% |
| **Training method** | QLoRA (NF4 base + BF16 LoRA adapters, r=64) | 90% |
| **Optimizer** | AdamW (ε=1e-15, β1=0.9, β2=0.95) | 95% |
| **Precision** | NF4 base + BF16 LoRA + FP32 LM head | 95% |
| **Router** | Frozen during RL (GLM-5 principle) | 75% |
| **Anti-forgetting** | PTX auxiliary loss + LoRA inherent regularization | 85% |
| **Regularization** | L2 log-ratio (Kimi, τ=0.05, tunable) | 75% |
| **Curriculum** | Proportional to 1-success_rate | 85% |
| **Group size** | G=8 (QLoRA) or G=16 (BF16+LoRA) | 70% |
| **Framework** | Unsloth (primary) or OpenRLHF (alternative) | 85% |
| **Hardware** | 1×A100 80GB for QLoRA; 2×A100 for BF16+LoRA | 90% |
| **Budget** | $280-560 (QLoRA on RunPod) | 80% |
| **First experiment** | E0: GSPO on Qwen3.5-35B-A3B, QLoRA, 200 steps (2-4 hours) | N/A |
| **Validation** | E1: MATH-500, 3000 steps (48-96 hours) | N/A |

---

## Sources

### Primary Pipeline Analysis Sources
- Kimi K1.5 (arXiv:2501.12599), K2 (arXiv:2507.20534), K2.5 (arXiv:2602.02276)
- GLM-5 (arXiv:2602.15763), Slime (github.com/THUDM/slime)
- MiniMax-01 (arXiv:2501.08313), M1 (arXiv:2506.13585), Forge blog, M2.7 blog
- GSPO (arXiv:2507.18071), CISPO (Swift docs), DISPO (arXiv:2602.00983), STAPO (arXiv:2602.15620)
- ScaleRL (arXiv:2510.13786), APRIL (arXiv:2509.18521)

### Qwen3.5-35B-A3B Sources
- [Qwen3.5-35B-A3B on HuggingFace](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- [Qwen3.5-35B-A3B-Base on HuggingFace](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base)
- [Qwen3.5-35B-A3B Specifications (apxml)](https://apxml.com/models/qwen35-35b-a3b)
- [Qwen3.5 Architecture Analysis (Medium)](https://medium.com/data-science-in-your-pocket/qwen-3-5-explained-architecture-upgrades-over-qwen-3-benchmarks-and-real-world-use-cases-af38b01e9888)
- [Qwen3.5 Fine-Tuning MoE vs Dense (Medium)](https://medium.com/@ishaafsalman/qwen3-5-fine-tuning-in-2026-moe-vs-dense-b2d17de73a9e)
- [GSPO: Scalable RL for Language Models (Qwen blog)](https://qwenlm.github.io/blog/gspo/)
- [Artificial Analysis: Qwen3.5-35B-A3B](https://artificialanalysis.ai/models/qwen3-5-35b-a3b)
- [Qwen3.5-35B-A3B on OpenRouter](https://openrouter.ai/qwen/qwen3.5-35b-a3b)

### LoRA RL Sources
- [veRL LoRA PPO Documentation](https://verl.readthedocs.io/en/latest/advance/ppo_lora.html)
- [LoRA-MoE with GRPO: RO-GRPO (OpenReview)](https://openreview.net/forum?id=rhD7ZuFAjU)
- [Unsloth RL Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide)
- [OpenRLHF GitHub](https://github.com/OpenRLHF/OpenRLHF)
- [RL with LoRA Deep Dive (kalomaze)](https://kalomaze.bearblog.dev/rl-lora-ddd/)

### Architecture-Pipeline Analysis
- NanoSeek knowledge/architecture_rl_fit_and_engineering_feasibility.md
- NanoSeek knowledge/unified_rl_pipeline_analysis.md
- NanoSeek research/infra_analysis.md
