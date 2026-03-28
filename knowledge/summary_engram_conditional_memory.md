# Engram: Conditional Memory via Scalable Lookup (DeepSeek, Jan 2025)
## Paper: arxiv.org/abs/2601.07372

---

## Core Idea

Introduces **conditional memory** as a new sparsity axis complementary to MoE's conditional computation. Language modeling involves two sub-tasks: (1) compositional reasoning (needs dynamic computation) and (2) knowledge retrieval (static, local, stereotyped patterns). Standard transformers waste early layers **simulating retrieval through computation** — reconstructing named entities, formulaic phrases, etc. via expensive attention+FFN computation when a simple O(1) lookup would suffice.

**Engram** = modernized N-gram embedding module with:
- Tokenizer compression (23% vocab reduction via NFKC normalization + lowercasing)
- Multi-head hashing (K hash heads per N-gram order to reduce collisions)
- Context-aware gating (hidden state as Query, memory as Key/Value, sigmoid gate)
- Lightweight depthwise causal convolution (residual, SiLU activation)
- Multi-branch integration (shared V projection, branch-specific K projections for mHC)

## Key Results

### U-Shaped Sparsity Allocation Law
Given fixed total params and FLOPs, optimal split between MoE experts and Engram memory is **ρ ≈ 75-80%** for MoE, 20-25% for Engram. Pure MoE (ρ=100%) is suboptimal.

### Large-Scale Results (all 3.8B active params, 262B tokens, 30 layers)
| Model | Total Params | Experts | Engram | MMLU | BBH | HumanEval | MATH |
|-------|-------------|---------|--------|------|-----|-----------|------|
| Dense-4B | 4.1B | - | - | 48.6 | 42.8 | 26.8 | 15.2 |
| MoE-27B | 26.7B | 2+72 (top-6) | - | 57.4 | 50.9 | 37.8 | 28.3 |
| Engram-27B | 26.7B | 2+55 (top-6) | 5.7B | **60.4** | **55.9** | **40.8** | **30.7** |
| Engram-40B | 39.5B | 2+55 (top-6) | 18.5B | 60.6 | 57.5 | 38.4 | 30.6 |

### Gains are NOT just knowledge — reasoning benefits MORE
- Knowledge (MMLU +3.0, CMMLU +4.0) — expected
- Reasoning (BBH +5.0, ARC-C +3.7, DROP +3.3) — **surprising and larger**
- Code/Math (HumanEval +3.0, MATH +2.4, GSM8K +2.2) — **surprising**

### Mechanistic Explanation
- **LogitLens**: Engram accelerates prediction convergence — early layers already near-final distribution
- **CKA**: Engram layer 5 ≈ MoE layer 12 in representational alignment → **effectively deepens the network**
- Engram relieves backbone from static reconstruction → frees depth for complex reasoning

### Long Context
Engram substantially boosts long-context (Multi-Query NIAH: 84.2 → 97.0, Variable Tracking: 77.0 → 89.0). By handling local dependencies via lookup, attention capacity freed for global context.

### System Efficiency
- Deterministic addressing → prefetchable from host memory
- 100B Engram table offloaded to CPU DRAM: <3% throughput loss
- Zipfian distribution enables multi-level cache hierarchy

## Architecture Details

### Configurations
- All models: 30 layers, d=2560, MLA with 32 heads, mHC expansion=4, Muon optimizer, LR=4e-4, 4096 seq len, 262B tokens
- Engram-27B: layers [2, 15], N-gram [2,3], 8 heads, d_mem=1280, vocab_size=2.26M, LR multiplier 5x, no weight decay, Adam for embeddings

### Ablation Insights
- **Layer placement**: Layer 2 optimal for single-layer (one attention round = sufficient context for gating)
- **Split placement** [2, 15] better than single layer (early intervention + late-stage contextual gating)
- **Most important components**: multi-branch integration, context-aware gating, tokenizer compression
- **Less important**: depthwise conv (marginal), 4-grams (slightly worse under fixed budget)

### Scaling Law Experiment Setup
- Sparsity ratio P_tot/P_act ≈ 10 maintained
- C=2e20 FLOPs: P_tot≈5.7B, P_act=568M, baseline 106 experts
- C=6e20 FLOPs: P_tot≈9.9B, P_act=993M, baseline 99 experts
- Optimal ρ stable at 75-80% across both regimes

---

## Relevance to NanoSeek

### Current NanoSeek Architecture
- 1B scale: 1.08B active / 4.75B total, 16 layers, d=2048, 64 experts top-8, MLA + MoE + MTP
- Anchor: 55M active / 282M total, 16 layers, d=480, 64 experts top-8

### Critical Questions for NanoSeek

**1. Does Engram help at NanoSeek's scale (1B active)?**
The paper's scaling law experiments use P_act=568M (close to NanoSeek's anchor/500M) and P_act=993M (close to NanoSeek's 1B). The U-shaped allocation law holds at both scales. This suggests Engram WOULD help at NanoSeek's scale.

**2. How would we add Engram to NanoSeek?**
- Reduce routed experts from 64 to ~48-50 (keeping ρ≈75-80%)
- Allocate freed params to Engram tables at layers [2, ~10]
- Use N-gram [2,3], multi-head hashing, context-aware gating
- Engram params trained with Adam (not Muon), 5x LR multiplier, no weight decay
- Conv zero-init for identity mapping at start

**3. What NanoSeek already has that aligns**
- MLA attention (paper uses MLA too)
- Muon optimizer (paper uses Muon)
- Multi-phase training (paper does pre-train then long-context extension)

**4. What NanoSeek would need to add**
- Engram module (hashing, tokenizer compression, gating, conv)
- Modified optimizer setup (separate Adam for Engram embeddings)
- Integration with existing MoE routing (fewer experts)
- mHC (Manifold-Constrained Hyper-Connections) — paper's default backbone

**5. Key consideration: NanoSeek doesn't use mHC**
The paper's multi-branch integration (mHC with M=4) is a significant component. NanoSeek uses standard residual connections. The "w/o multi branch" ablation shows this is the biggest contributor. Without mHC, gains would be smaller but still positive (context-aware gating and tokenizer compression are independently beneficial).

### Potential Experiment
At anchor scale (55M active), compare:
1. Baseline: 64 experts, top-8 (current)
2. Engram: 48 experts, top-8, + Engram table at layer 2
This would cost ~6 GPU-hours and tell us if the allocation law transfers to NanoSeek's architecture.

### What NOT to do
- Don't add Engram before completing the current research plan (HP search, stability, dynamics)
- Don't implement mHC just for Engram (too much scope creep)
- Don't expect same magnitude of gains without mHC multi-branch integration
