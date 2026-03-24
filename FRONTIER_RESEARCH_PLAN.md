# NanoSeek: Frontier Pre-Training Research Plan
## A Scaling Law Study of Hybrid Attention + MoE Architectures
### Designed to Demonstrate Anthropic Pre-Training RE Capabilities

---

## Executive Summary

**One-line pitch**: First systematic comparison of KDA+MLA hybrid attention vs pure MLA
in MoE models across 4 scales (55M→1B active), with scaling law fits, MuonClip optimizer,
and RL post-training — producing falsifiable predictions and open artifacts.

**Why this matters**: No one has tested whether Kimi Linear's advantages (1.16× compute
efficiency, better RL scaling) transfer to nano scale. No one has combined KDA + high-sparsity
MoE. No one has derived joint scaling laws for hybrid attention + MoE architectures.

**Model size recommendation**: **1B active / 5-10B total is the optimal flagship scale** for a
solo researcher. Here's why:

| Scale | Solo Researcher Fit | Research Signal | Anthropic Relevance |
|-------|-------------------|-----------------|---------------------|
| 100M | Fast iteration but too small for convincing results | Weak | Low |
| **500M-1B** | **Sweet spot: meaningful results, affordable compute** | **Strong** | **High** |
| 3B+ | Too expensive for proper ablation ($2K+ per run) | Very strong | Medium (they have bigger) |

The key insight: **Anthropic hires for scientific rigor, not model size.** A beautifully
executed 4-point scaling law study at 55M→1B with 3 architecture variants is vastly more
impressive than training a single 3B model. This mirrors what their own team does internally
(Kaplan et al. used models from 768 params to 1.5B for the original scaling laws).

---

## 0. What This Project Demonstrates (Mapped to Anthropic JD)

| JD Requirement | How This Project Demonstrates It |
|---------------|--------------------------------|
| "Model architecture research" | 3 architecture variants: pure MLA, KDA+MLA hybrid, KDA+MLA+high-sparsity |
| "Algorithms, optimizers" | MuonClip (QK-Clip), MuonAdamW, weight decay analysis |
| "Data processing" | ClimbMix curation, FIM 10% PSM, quality filtering pipeline |
| "Design, run, analyze experiments" | 4-scale × 3-arch = 12 scaling law points + RL ablation |
| "Scale training infrastructure" | DDP/FSDP2 multi-GPU, gradient checkpointing, MoE expert parallelism |
| "Dev tooling" | W&B dashboards, eval harness, checkpoint management |
| "Optimizing throughput of novel attention mechanisms" | KDA Triton kernel integration, FLA library, throughput benchmarks |
| "Comparing compute efficiency of Transformer variants" | KDA+MLA vs MLA scaling laws: L(N,D) per architecture |
| "Preparing large-scale datasets" | ClimbMix 170 shards, domain BPB tracking, data quality analysis |
| "Scaling distributed training" | 8×H100 for 1B, expert parallelism for high-sparsity variant |
| "Fault tolerance" | Checkpoint resume, EMA state management, loss spike detection |
| "Interactive visualizations of model internals" | Expert specialization heatmaps (I_spec), attention pattern analysis, routing visualization |

**Sample projects from JD that directly match**:
- "Optimizing the throughput of novel attention mechanisms" → KDA kernel benchmarking
- "Comparing compute efficiency of different Transformer variants" → THE CORE OF THIS PROJECT
- "Preparing large-scale datasets for efficient model consumption" → ClimbMix pipeline
- "Scaling distributed training jobs to thousands of GPUs" → multi-GPU MoE training
- "Creating interactive visualizations of model internals" → expert routing + attention patterns

---

## 1. The Research Questions

### Primary (Architecture)
**Q1**: Does KDA+MLA hybrid attention outperform pure MLA in MoE models at 1B scale?
- Kimi Linear showed 1.16× compute efficiency at 3B active. Does this hold at 1B? At 150M?
- If yes → we have a strictly better architecture for efficient LLMs
- If no → the advantage is scale-dependent (publishable negative result)

**Q2**: Does the sparsity scaling law (more experts at fixed compute = better) hold at nano scale?
- Kimi K2 showed it at 1T. "Towards Greater Leverage" (Jul 2025) showed MoE-mini at 0.85B active matched 6.1B dense.
- We test: 64 experts (sparsity=8) vs 128 experts (sparsity=16) at matched FLOPs

### Secondary (Optimizer)
**Q3**: Does MuonClip (QK-Clip) improve training stability at 1B scale?
- At 1T scale, K2 saw logit explosion. At 1B, does Muon cause instability?
- "Muon + MLA + MoE achieves 48-52% of AdamW's compute" (2509.24406) — we validate

### Tertiary (RL Interaction)
**Q4**: Does KDA scale better under RL than MLA?
- Kimi Linear's most surprising finding. If confirmed at 1B → changes the optimal pre-training architecture
- Test via GRPO on math/code after pre-training

---

## 2. Architecture Variants

### Variant A: NanoSeek (Baseline — DeepSeek V3.2)
```
Attention: 16× MLA (full attention every layer)
MoE: 64 experts, top-8, 2 shared, grouped routing (8 groups, topk_group=4)
Position: RoPE on all layers
Optimizer: MuonAdamW
Status: ALREADY IMPLEMENTED
```

### Variant B: NanoKDA (Kimi Linear-style Hybrid)
```
Attention: 12× KDA + 4× MLA (3:1 ratio, every 4th layer = MLA)
MoE: 64 experts, top-8, 2 shared, grouped routing (unchanged from A)
Position: KDA handles position natively; MLA layers use NoPE
Optimizer: MuonAdamW
KDA details:
  - Channel-wise gated delta rule (Kimi Linear Eq. 1)
  - ShortConv(kernel=4) + Swish + L2Norm on Q,K
  - Sigmoid output gate (NOT Swish — paper ablation)
  - Low-rank alpha gate (rank = head_dim)
  - Head-wise RMSNorm before output gating
  - Chunk size = 64 for chunkwise parallel training
  - FLA library: fla.ops.kda kernel (PR #621, merged)
Status: TO IMPLEMENT
```

### Variant C: NanoKDA-S (KDA Hybrid + High-Sparsity MoE)
```
Attention: 12× KDA + 4× MLA (same as B)
MoE: 128 experts, top-8, 1 shared, NO grouped routing (K2-style)
Position: KDA handles position; MLA layers NoPE
Optimizer: MuonClip (MuonAdamW + QK-Clip, τ=100)
Status: TO IMPLEMENT (combines B + K2 MoE + QK-Clip)
```

### Why only 3 variants (not 6)?

The original KIMI_ABLATION_PLAN.md proposed 6 runs. After research, I'm cutting to 3:

1. **QK-Clip as standalone ablation is low-value** at 1B scale — K2 needed it at 1T,
   we almost certainly won't see logit explosion at 1B. Include it in Variant C for free.
2. **High-sparsity alone (without KDA) is less interesting** — the compound effect is the question.
3. **3 variants × 4 scales = 12 runs** is the maximum for a solo researcher to do rigorously.
   6 variants × 4 scales = 24 runs is unfeasible.

The design is: **A is the baseline, B tests attention innovation, C tests everything together.**
If C > B > A, both innovations compound. If B > C, high sparsity hurts. If A > B, KDA fails.

---

## 3. The Scaling Ladder

### 4 Scales for Scaling Law Fit

| Scale | Hidden | Layers | Heads | Active Params | Total Params | Tokens | Time (A6000) |
|-------|--------|--------|-------|---------------|--------------|--------|-------------|
| **Nano-55M** | 480 | 16 | 8 | ~55M | ~240M | 1.1B | ~2 hrs |
| **Nano-150M** | 768 | 16 | 12 | ~150M | ~650M | 3B | ~6 hrs |
| **Nano-500M** | 1280 | 16 | 16 | ~441M | ~2B | 8.8B | ~18 hrs |
| **NanoSeek-1B** | 2048 | 16 | 16 | ~1.08B | ~5B | 22B | ~14 hrs (8×H100) |

**Why 16 layers for all**: Fixes depth, isolates width scaling. This is standard practice
(muP paper, OLMo scaling experiments). Changing depth breaks muP transfer.

**Why add 150M**: The 55M→441M jump is 8×, too large for reliable scaling law interpolation.
Adding 150M gives us an 8× total range with ~3× steps between points.

**Token budget**: Chinchilla optimal D ≈ 20×N_active for each scale.

### Nano-150M Config (NEW)

```python
# New config to add to config.py
def get_nano150m_config():
    config = NanoSeekConfig(
        hidden_size=768,
        num_layers=16,
        num_heads=12,
        vocab_size=32768,
        intermediate_size=1966,        # 2.56 × 768
        max_position_embeddings=4096,
        sequence_length=4096,
    )
    config.mla.q_lora_rank = 161       # 0.21 × 768
    config.mla.kv_lora_rank = 54       # 0.07 × 768
    config.moe.moe_intermediate_size = 288  # 0.375 × 768
    config.total_tokens = 3_000_000_000  # 3B (20× active)
    return config
```

---

## 4. Experimental Protocol

### Phase 1: Anchor Screening (3 variants × 55M, ~6 GPU-hours)

**Goal**: Quick check that all 3 variants train without issues.

```bash
# Run all 3 at anchor scale
for arch in nanoseek nanokda nanokda-s; do
  python -m nanoseek.scripts.pre_train \
    --run "screen-${arch}" --scale anchor --arch $arch \
    --num-iterations 500 --eval-every 100 --save-every 500 --seed 42
done
```

**Pass criteria**:
- No NaN/divergence in any run
- ema_val_bpb within 0.05 nats across all 3 variants
- H_load > 2.0 bits for all runs
- KDA state norm doesn't explode

### Phase 2: Scaling Law Study (3 × 4 = 12 runs, ~120 GPU-hours)

**Goal**: Derive L(N_active, D) per architecture variant.

```bash
# For each architecture and each scale
for arch in nanoseek nanokda nanokda-s; do
  for scale in anchor nano150m nano500m; do
    python -m nanoseek.scripts.pre_train \
      --run "scaling-${arch}-${scale}" --scale $scale --arch $arch \
      --seed 42 --eval-every 200 --save-every -1
  done
done
# 1B runs on multi-GPU (top 2 variants only, after analyzing 3 smaller scales)
```

**Budget**: 12 runs total, but the 3 × 1B runs are expensive. Strategy:
- Run all 3 variants at 55M, 150M, 500M first (9 runs, ~78 GPU-hours on A6000)
- Fit preliminary scaling laws from 3 points
- Run top 2 variants at 1B (2 runs, ~28 GPU-hours on 8×H100)
- Total: ~$250 (A6000) + ~$700 (H100 cluster) = **~$950**

### Phase 3: Scaling Law Analysis

For each variant, fit the Chinchilla loss function:
```
L(N, D) = A/N^α + B/D^β + E
```

Where N = active params, D = tokens, and (A, α, B, β, E) are fit parameters.

**Key comparisons**:
1. **Compute efficiency**: At matched FLOPs, which variant achieves lowest loss?
2. **Scaling exponent**: Does α differ across architectures? (Different α means different returns to scale)
3. **Crossover point**: At what N does KDA+MLA beat pure MLA? (If the advantage grows with scale → huge finding)
4. **Sparsity bonus**: Does Variant C's extra experts (128 vs 64) shift the scaling law?

### Phase 4: RL Post-Training (top 2 variants, ~20 GPU-hours)

**Goal**: Test KDA × RL synergy claim from Kimi Linear.

```bash
# GRPO on math + code for top 2 architectures
for arch in nanoseek nanokda; do  # or nanokda-s
  python -m nanoseek.alignment.run_grpo \
    --checkpoint "scaling-${arch}-nano500m/final.pt" \
    --budget 5pct --stages reasoning --seed 42
done
```

**Measurements**:
- RL training accuracy curve (KDA vs MLA — does KDA improve faster?)
- GSM8K / HumanEval before/after RL
- Token efficiency (same accuracy in fewer tokens?)
- MTP acceptance rate change during RL

### Phase 5: Flagship Training (winner architecture, 1B, ~14 GPU-hours on 8×H100)

The culmination. Train the winning architecture at full 1B scale with 22B tokens.

---

## 5. Metrics & Monitoring (Full Stack)

### Per-Run Logging (to W&B)

```yaml
# Quality
ema_val_bpb: "PRIMARY metric. EMA weights only (RULE 1)"
train_loss: "Training loss per step"
domain_bpb: "Per-domain: code, math, science, web, books"
fim_loss: "FIM-specific loss (should decrease)"

# MoE Health
H_load: "Load balance entropy in bits (>2.0 required)"
I_spec: "Expert specialization MI (higher = more specialized)"
dead_expert_pct: "% of never-selected experts"
expert_gini: "Gini coefficient of load distribution"
routing_entropy_per_layer: "Layer-by-layer routing entropy"

# Attention Diagnostics
max_attn_logit: "Max attention logit (QK-Clip trigger detection)"
kda_state_frobenius: "KDA state matrix norm (growth = instability)"
qk_clip_trigger_rate: "% of heads triggering QK-Clip per step"

# Training Dynamics
grad_norm: "Global gradient L2 norm"
learning_rate: "Current LR (for schedule verification)"
tokens_per_second: "Training throughput"
mfu: "Model FLOPs utilization"
peak_memory_gb: "GPU memory usage"

# MTP
mtp_loss: "MTP auxiliary loss"
mtp_acceptance_rate: "Speculative acceptance rate"

# Per-Layer (for hybrid analysis)
per_layer_attn_type: "KDA or MLA label"
per_layer_grad_norm: "Gradient norm per layer"
per_layer_activation_norm: "Activation norm per layer"
```

### Novel Measurements (Not in Standard Pipelines)

1. **KDA-fed vs MLA-fed MoE I_spec**: In Variant B/C, compare expert specialization
   between layers that receive KDA output vs MLA output. Does the attention mechanism
   affect downstream routing?

2. **Attention pattern comparison**: Visualize what KDA "attends to" vs MLA on the
   same sequences. KDA has no explicit attention matrix, but we can compute the
   effective attention via the state readout S_t^T q_t.

3. **Scaling law residuals**: Plot actual loss vs predicted loss at each scale point.
   Systematic residuals indicate architectural effects not captured by the simple power law.

---

## 6. Implementation Roadmap

### Week 1: KDA Implementation + Anchor Screening

| Day | Task | Output |
|-----|------|--------|
| 1 | KDA module: core recurrence + FLA kernel integration | `nanoseek/kda.py` |
| 2 | KDA module: ShortConv, gates, output gating, NoPE for MLA | `nanoseek/kda.py` complete |
| 3 | Config extension + layer assignment + Nano-150M config | `config.py` updated |
| 4 | Unit tests + integration tests + meta device init | `tests/test_kda.py` |
| 5 | QK-Clip in optim.py + high-sparsity MoE config | `optim.py` + config presets |
| 6 | Anchor screening runs (3 × 2hrs) | W&B: screen-{a,b,c} |
| 7 | Analysis + go/no-go decision | Screen report |

### Week 2: Scaling Law Runs (55M + 150M + 500M)

| Day | Task | Output |
|-----|------|--------|
| 8-9 | Run 3 × 55M (parallel if possible) | 3 anchor-scale data points |
| 10-11 | Run 3 × 150M | 3 mid-scale data points |
| 12-14 | Run 3 × 500M (longest: ~18 hrs each) | 3 validation-scale data points |

### Week 3: Analysis + 1B Training + RL

| Day | Task | Output |
|-----|------|--------|
| 15 | Fit scaling laws, select top 2 variants | Scaling law report |
| 16-17 | Run top 2 × 1B on multi-GPU cluster | 2 flagship data points |
| 18-19 | RL post-training on top 2 × 500M | RL comparison data |
| 20-21 | Final analysis, visualizations, write-up | Complete report |

**Total**: 3 weeks, ~$950-1200 compute budget.

---

## 7. Why 1B is the Right Flagship Size

### Evidence from Literature

1. **"Towards Greater Leverage" (Jul 2025)**: MoE-mini at 0.85B active matched a 6.1B
   dense model with 7× less compute. Our 1.08B active is in this validated range.

2. **OLMoE (Allen AI)**: 1.3B active / 6.9B total. Found clear expert specialization,
   routing patterns, and scaling behavior. Fully open. Our 1.08B is comparable.

3. **Kaplan scaling laws (Anthropic)**: Demonstrated clean power-law fits from 768 params
   to 1.5B. Our 55M→1B range (20× span) is well within this validated regime.

4. **"Scaling Laws Meet Model Architecture" (ICLR 2026)**: Tested 200+ models from
   80M to 3B. Our range (55M to 1B) overlaps their validated regime.

5. **muP for MoE (Aug 2025)**: Validated HP transfer from 51M to 2B+ total params.
   Our 240M→5B total range is exactly in their validated regime.

### The Practical Calculus

```
                    55M         150M        500M        1B
Time per run:       2 hrs       6 hrs       18 hrs      14 hrs (8×H100)
Cost per run:       $4          $12         $36         $350
Ablation (3 arch):  $12         $36         $108        $700
Total (12 runs):    ────────── ~$950 ──────────
```

At 3B, the 1B equivalent would cost ~$2000 per run, making a 12-run scaling study ~$6000.
Not impossible but unnecessarily expensive when the science works at 1B.

### What Makes 1B Compelling

1. **Large enough for real phenomena**: Expert specialization, routing patterns, scaling laws,
   attention mechanisms all produce clear signal at 1B.

2. **Small enough for proper science**: Can run 12+ experiments, fit scaling laws, do ablations.
   A single 70B run teaches you almost nothing about architecture.

3. **Matches the research literature**: OLMoE (1.3B), Moonlight (3B), MoE-mini (0.85B)
   all produced publishable results at this scale.

4. **Requires real infrastructure**: Multi-GPU training, gradient checkpointing, expert
   parallelism. Shows you can build production systems.

5. **Enables RL post-training**: 1B is large enough that GRPO produces meaningful
   improvement on GSM8K/HumanEval.

---

## 8. What a Top 1% Researcher Would Focus On

### The Non-Obvious Insights from This Research Sprint

**1. KDA × RL is the real prize (not just pre-training quality)**

Kimi Linear showed KDA improves faster under RL than MLA, with the gap widening over time.
If we confirm this at 1B, it means the optimal pre-training architecture depends on your
post-training plan. This is a genuinely novel finding that would change how labs choose
attention mechanisms.

**2. Weight decay matters more than muP for HP transfer**

"Weight Decay may matter more than muP for LR Transfer" (Oct 2025, arXiv:2510.19093)
found that muP's assumptions hold only briefly at training start. For the bulk of training,
weight decay is what enables LR transfer. This changes our HP search strategy:
- **Old plan**: Tune LR at anchor, transfer via muP scaling rules
- **New plan**: Tune (LR, weight_decay) jointly at anchor, transfer both

**3. Expert granularity breaks muP, expert count doesn't**

"muP for MoE" (Aug 2025, arXiv:2508.09752) proved that changing expert count across
scales is safe for HP transfer, but changing expert granularity (inter_dim) breaks it.
This means Variant C (128 experts, same inter_dim) should transfer HPs from anchor
perfectly, while a hypothetical variant with different inter_dim would need re-tuning.

**4. The 3:1 ratio is robustly optimal**

A systematic study of 72 models at 340M and 1.3B (arXiv:2507.06457) confirmed that
3:1 and 6:1 hybrid ratios achieve near-Transformer quality. Importantly, the best
standalone linear attention variant is NOT necessarily the best in a hybrid — what
matters is complementary memory mechanisms. This validates our 3:1 KDA:MLA choice.

**5. OLMo Hybrid's hierarchy: hybrid GDN > pure GDN > transformer > hybrid Mamba2**

Allen AI's OLMo Hybrid showed that NOT all linear attention variants work equally well
in hybrids. GDN (which KDA extends) is the best. Mamba2 actually hurts in hybrid
configuration. The delta rule is the key ingredient — it enables precise memory retrieval
that complements the full attention layers.

### What NOT to Waste Time On

- **Don't ablate KDA ratio**: 3:1 is well-validated by 3 independent efforts. Test other ratios
  only if 3:1 fails at nano scale.
- **Don't implement vision/multimodal**: K2.5's multimodal is irrelevant for pre-training ablation.
- **Don't over-optimize QK-Clip**: At 1B it probably never triggers. Include it, monitor it, move on.
- **Don't try to match Claude's architecture**: Anthropic doesn't disclose it. Focus on the science.
- **Don't chase MMLU/benchmarks**: At 1B, absolute benchmark numbers are unimpressive. The VALUE
  is in the relative comparison across architectures and the scaling law predictions.

---

## 9. The Portfolio Artifacts

When complete, this project produces:

### Research Artifacts
1. **Scaling law fits**: L(N,D) for 3 architecture variants, 4 scales each
2. **Architecture comparison**: KDA+MLA vs MLA quality/efficiency tradeoff curve
3. **Sparsity scaling law verification**: 64 vs 128 experts at matched compute
4. **RL × attention interaction**: First test of KDA RL advantage at nano scale
5. **Expert specialization under hybrid attention**: I_spec dynamics in KDA-fed vs MLA-fed layers

### Engineering Artifacts
6. **KDA module**: Production-quality implementation with FLA kernel integration
7. **MuonClip optimizer**: QK-Clip implementation for Muon
8. **Scaling law fitting tool**: Automated L(N,D) fitting with confidence intervals
9. **Multi-architecture training framework**: --arch flag supporting 3+ variants
10. **W&B dashboards**: Per-experiment comparison views, scaling law plots

### Open-Source Deliverables
11. **Model weights**: EMA weights at all 4 scales for winning architecture
12. **Training configs**: All 12+ experiment configs (fully reproducible)
13. **Training logs**: Complete W&B logs for every run
14. **Analysis notebooks**: Scaling law fitting, visualization, comparison

### Paper-Quality Figures
15. **Figure 1**: Scaling curves (loss vs tokens) for 3 architectures, 4 scales
16. **Figure 2**: Compute efficiency curves (loss vs FLOPs) showing crossover
17. **Figure 3**: Expert specialization heatmaps (I_spec per layer, KDA vs MLA)
18. **Figure 4**: RL training curves (accuracy vs RL steps, KDA vs MLA)
19. **Figure 5**: Inference efficiency (throughput vs context length)

---

## 10. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| FLA KDA kernel incompatible with our setup | Low (kernel merged) | High | Pure PyTorch reference implementation as fallback |
| KDA worse than MLA at all scales | Medium | Medium | Publishable negative result; we still have pure MLA baseline |
| 128-expert memory pressure at 1B | Low | Medium | Gradient checkpointing tuning; fall back to 64 experts |
| muP transfer fails across architectures | Medium | Medium | Re-tune at each scale (more expensive but still feasible) |
| QK-Clip unnecessary at 1B | High | Low | No cost — we just observe it never triggers |
| Total budget exceeds $1500 | Low | Low | Cut 1B runs to 1 variant; scaling law still valid from 3 scales |

---

## 11. How This Compares to What Top Labs Do

| What We Do | What Top Labs Do | Gap |
|-----------|-----------------|-----|
| 4-scale scaling law (55M-1B) | 10+ scale scaling law (100M-100B+) | Scale, not methodology |
| 3 architecture variants | 5-10+ variants per study | Breadth |
| MuonClip optimizer research | Custom optimizers per architecture | Same quality |
| KDA+MLA hybrid (novel at 1B) | Novel architectures at 100B+ | Scale, not novelty |
| GRPO post-training | RLHF/Constitutional AI at scale | Scale + sophistication |
| Single researcher | Teams of 5-20 | Resources |
| ~$1000 compute | $1M+ compute | Budget |
| Open-source everything | Often proprietary | We win on openness |

**The gap is SCALE, not METHODOLOGY.** This is exactly what Anthropic wants to see:
someone who does the right experiments at affordable scale, and could do them at
larger scale with more resources.

---

## 12. Timeline Summary

```
Week 1: Implementation
  ├── KDA module + FLA integration (3 days)
  ├── Config + tests + QK-Clip (2 days)
  └── Anchor screening (1 day)

Week 2: Scaling Law Runs
  ├── 3 × {55M, 150M} = 6 runs (~48 GPU-hours)
  └── 3 × 500M = 3 runs (~54 GPU-hours)

Week 3: Analysis + Flagship + RL
  ├── Scaling law fitting + variant selection (1 day)
  ├── Top 2 × 1B on multi-GPU (2 days)
  ├── RL post-training on 500M (2 days)
  └── Final analysis + write-up (2 days)

Total: 21 days, ~$950-1200 compute
```

---

## 13. Key Papers Referenced

### Architecture
- Kimi Linear (arXiv:2510.26692) — KDA + MLA hybrid, 1.16× efficiency
- Kimi K2 (arXiv:2507.20534) — 384 experts, MuonClip, sparsity scaling
- OLMo Hybrid (Ai2, 2025) — GDN + attention, 3:1 ratio validated
- Systematic Hybrid Analysis (arXiv:2507.06457) — 72 models, optimal ratios

### Scaling Laws
- Kaplan et al. (arXiv:2001.08361) — Original scaling laws
- Joint MoE Scaling Laws (arXiv:2502.05172) — MoE-specific scaling
- Towards Greater Leverage (arXiv:2507.17702) — 300+ models, MoE-mini
- Scaling Laws Meet Architecture (arXiv:2510.18245) — Architecture-conditional scaling

### Optimizers
- Moonlight (arXiv:2502.16982) — Muon at scale with weight decay
- Practical Muon (arXiv:2505.02222) — muP + Muon HP transfer
- Muon + MLA + MoE (arXiv:2509.24406) — 48-52% compute savings

### HP Transfer
- muP (arXiv:2203.03466) — Foundational HP transfer
- muP for MoE (arXiv:2508.09752) — Expert count safe, granularity breaks transfer
- Weight Decay > muP (arXiv:2510.19093) — WD matters more for transfer

### Training Stability
- SPAM (arXiv:2501.06842) — Momentum reset on spikes
- ZClip (arXiv:2504.02507) — Adaptive gradient clipping
- Spike No More (arXiv:2312.16903) — Root causes of loss spikes

---

## 14. Final Recommendation

**Build a scaling law study, not just a big model.**

The single most impressive thing you can produce for an Anthropic Pre-Training RE
application is a clean, reproducible, multi-scale comparison of attention mechanisms
in MoE architectures — with scaling law fits, proper controls, novel measurements
(I_spec under hybrid attention, KDA×RL interaction), and open artifacts.

**The model size is 1B active / 5-10B total.** Not because it's big, but because it's
the largest scale where a solo researcher can run a proper 12-run experiment with
scaling law analysis in 3 weeks for under $1500.

**The architecture is KDA+MLA hybrid (3:1) with MoE.** This is the frontier design
pattern (Kimi Linear, OLMo Hybrid, Qwen 3.5), and testing it at nano scale with
proper scaling laws is genuinely novel work.

**The optimizer is MuonClip.** Muon + MLA + MoE gives 2× efficiency over AdamW.
QK-Clip adds stability insurance. This is the production optimizer for MoE models
in 2026.

**The science matters more than the engineering.** Anthropic hires researchers who
engineer, not engineers who sometimes research. The scaling law fits, the I_spec
analysis, the KDA×RL finding — these are what make this project stand out. The
infrastructure is table stakes.
