# 05: Canon Layers Deep Dive — Architecture Design from Physics of Language Models

## Allen-Zhu, Parts 4.1 & 4.2 × NanoSeek Integration Analysis

**Paper:** [arXiv:2512.17351](https://arxiv.org/abs/2512.17351) (Part 4.1, NeurIPS 2025)
**Code:** [github.com/facebookresearch/PhysicsLM4](https://github.com/facebookresearch/PhysicsLM4) (Part 4.2)
**Models:** [HuggingFace facebook/PhysicsLM4.2](https://huggingface.co/facebook/PhysicsLM4.2__Llama-3B-Nemo-1T-lr0.003) (16 pretrained models)
**Authors:** Zeyuan Allen-Zhu (Canon Layers jointly conceived with Xiaoli Xu)
**Authority level:** Reference document — does not override PAPER_ANALYSIS_V3_V32.md or REIMPLEMENTATION_PLAN.md
**Last updated:** 2026-03-14

---

## Table of Contents

1. [The Fundamental Insight: Three Axes of Information Flow](#1-the-fundamental-insight)
2. [What Canon Layers Are (And Are Not)](#2-what-canon-layers-are)
3. [Canon-ABCD: Four Insertion Points](#3-canon-abcd-four-insertion-points)
4. [The Synthetic Pretraining Playground](#4-the-synthetic-pretraining-playground)
5. [Quantitative Results: The Hard Numbers](#5-quantitative-results)
6. [Part 4.2: Canon at Scale — Real-World Validation](#6-part-42-canon-at-scale)
7. [Why Canon Layers Work: First-Principles Analysis](#7-why-canon-layers-work)
8. [Complementary Innovations: QK-Norm and Partial RoPE](#8-complementary-innovations)
9. [NanoSeek Integration Analysis: MLA × Canon Synergy Hypothesis](#9-nanoseek-integration-analysis)
10. [NanoSeek Integration Analysis: MoE × Canon-C Interaction](#10-moe-canon-c-interaction)
11. [Implementation Specification for NanoSeek](#11-implementation-specification)
12. [Experimental Design: Canon Layer Ablation Study](#12-experimental-design)
13. [What NOT to Do: Anti-Patterns and Timing](#13-what-not-to-do)
14. [Principles Extracted](#14-principles-extracted)
15. [Open Research Questions](#15-open-research-questions)
16. [References](#16-references)

---

## 1. The Fundamental Insight: Three Axes of Information Flow

Standard transformer architectures have two information flow pathways. Allen-Zhu's
research reveals a third axis is missing — and that filling it yields dramatic gains.

```
AXIS 1: VERTICAL (depth)
├── Mechanism: Residual stream through stacked layers
├── Bandwidth: High (full hidden dimension per layer)
├── Range: Layer-to-layer (bounded by network depth)
└── What it provides: Compositional reasoning, hierarchical processing

AXIS 2: GLOBAL HORIZONTAL (attention)
├── Mechanism: Self-attention over full sequence
├── Bandwidth: Sparse, selective (softmax picks few high-weight tokens)
├── Range: Full sequence length (all positions can attend to all others)
└── What it provides: Long-range dependency resolution, fact retrieval

AXIS 3: LOCAL HORIZONTAL (MISSING in standard transformers)
├── Mechanism: ??? ← Canon layers fill this gap
├── Bandwidth: Dense (every neighboring token pair mixed)
├── Range: Short (kernel size 4 = nearest 3 tokens)
└── What it provides: Multi-hop reasoning chains, local context enrichment
```

### Why the gap matters

Consider a 5-hop reasoning chain: A → B → C → D → E, where each fact is at
a different position in the sequence.

- **Attention** can retrieve any individual fact (A, B, C, D, or E) in one step.
- But **chaining** them requires the output of one retrieval to inform the next.
- In a standard transformer, this chaining happens **vertically** — each layer
  resolves one hop. A 5-hop chain needs ≥5 layers of depth.
- **Canon layers** provide a shortcut: they mix neighboring token representations
  **before** attention, giving each token access to its neighbors' content.
  This means attention can retrieve pre-mixed representations that already
  contain multi-token context, effectively compressing multiple hops.

This is why reasoning depth improves 2-4×: the effective depth of the network
increases because each attention layer receives richer, locally-mixed inputs.

### The musical analogy

The name "Canon" comes from music: in a canon (like Pachelbel's Canon), the same
melody is played by multiple voices with a time delay. Each voice echoes its
neighbors. Canon layers do the same — each token's representation echoes
information from its neighboring tokens, creating a richer "chord" that
attention can then operate on.

---

## 2. What Canon Layers Are (And Are Not)

### What they ARE

Canon layers are **depthwise causal 1D convolutions** applied to the hidden
state or internal projections of a transformer block.

```python
# The entire Canon layer mechanism:
class ShortConvolution(nn.Module):
    def __init__(self, dim, kernel_size=4, residual=True, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            groups=dim,          # depthwise: each channel has its own kernel
            padding="same",      # preserve sequence length
            bias=bias,
        )
        self.residual = residual

    def forward(self, x):
        # x: (batch, seq_len, dim)
        h = x.transpose(1, 2)                # → (batch, dim, seq_len)
        h = self.conv(h)                      # depthwise conv1d
        h = h.transpose(1, 2)                 # → (batch, seq_len, dim)
        return h + x if self.residual else h  # optional residual
```

**Key properties:**

| Property | Value | Why |
|----------|-------|-----|
| Kernel size | 4 (default) | Covers 3 preceding tokens + current. Empirically optimal |
| Groups | dim (depthwise) | Each hidden dimension has independent 4-weight kernel. No cross-channel mixing |
| Causality | Causal padding | Token i sees only tokens ≤ i. Maintains autoregressive property |
| Parameter count | 4 × dim per Canon layer | Negligible. For dim=2048: 8,192 params per insertion point |
| Residual | Optional (default True) | Identity path preserves gradient flow |

### What they are NOT

- **NOT attention.** No query/key/value, no softmax, no O(L²) complexity.
- **NOT a replacement** for attention. They are complementary — different axis.
- **NOT learned attention patterns.** The kernel weights are fixed across positions.
- **NOT recurrence.** Processing is fully parallel (Conv1d over entire sequence).
- **NOT depth-wise mixing.** They operate horizontally (across tokens), not vertically (across layers).

### Parameter overhead analysis for NanoSeek-1B

```
NanoSeek-1B: hidden_dim = 2048, n_layers = 16

Per insertion point per layer: 4 × 2048 = 8,192 params
Full Canon-ABCD (4 points × 16 layers): 4 × 16 × 8,192 = 524,288 params

NanoSeek total params: ~4.75B
Canon overhead: 524K / 4.75B = 0.011%

This is genuinely negligible — less than a single expert's bias terms.
```

---

## 3. Canon-ABCD: Four Insertion Points

Canon layers can be placed at four strategic positions within each transformer
decoder block. Each position targets a different representation:

```
┌──────────────────────────────────────────────────────┐
│                  Decoder Block l                      │
│                                                      │
│  input x_l                                           │
│    │                                                 │
│    ▼                                                 │
│  RMSNorm (input_layernorm)                           │
│    │                                                 │
│    ▼                                                 │
│  ╔══════════════════╗                                │
│  ║   CANON-A        ║  ← Pre-attention mixing        │
│  ║   ShortConv(h)   ║     Mixes hidden states        │
│  ╚══════════════════╝     before Q/K/V projection    │
│    │                                                 │
│    ▼                                                 │
│  ┌──────────────────┐                                │
│  │  Q, K, V = proj  │                                │
│  │  ╔══════════════╗│                                │
│  │  ║  CANON-B     ║│  ← Inside-attention mixing     │
│  │  ║  ShortConv(  ║│     Mixes concatenated Q,K,V   │
│  │  ║   [Q;K;V])   ║│     before attention compute   │
│  │  ╚══════════════╝│                                │
│  │  attn_output     │                                │
│  └──────────────────┘                                │
│    │                                                 │
│    + (residual)                                      │
│    │                                                 │
│    ▼                                                 │
│  RMSNorm (post_attention_layernorm)                  │
│    │                                                 │
│    ▼                                                 │
│  ╔══════════════════╗                                │
│  ║   CANON-C        ║  ← Pre-MLP/MoE mixing         │
│  ║   ShortConv(h)   ║     Mixes hidden states        │
│  ╚══════════════════╝     before expert routing      │
│    │                                                 │
│    ▼                                                 │
│  ┌──────────────────┐                                │
│  │  gate, up = proj │                                │
│  │  ╔══════════════╗│                                │
│  │  ║  CANON-D     ║│  ← Inside-MLP mixing          │
│  │  ║  ShortConv(  ║│     Mixes gate+up projections  │
│  │  ║   [gate;up]) ║│     inside SwiGLU              │
│  │  ╚══════════════╝│                                │
│  │  mlp_output      │                                │
│  └──────────────────┘                                │
│    │                                                 │
│    + (residual)                                      │
│    │                                                 │
│    ▼                                                 │
│  output x_{l+1}                                      │
└──────────────────────────────────────────────────────┘
```

### What each position does

**Canon-A (Pre-Attention):**
- Operates on: normalized hidden states before Q/K/V projection
- Effect: Each token's query/key/value incorporates information from neighboring
  tokens. Attention is computed on locally-enriched representations.
- Intuition: "Before I decide what to attend to, let me first absorb my
  immediate context."

**Canon-B (Inside-Attention):**
- Operates on: concatenated [Q, K, V] after linear projection
- Effect: Query, key, and value vectors are locally smoothed. Attention patterns
  become influenced by local context, not just the single-token projection.
- Intuition: "My query is not just about me — it's about my local neighborhood."
- Implementation detail: concat Q,K,V → ShortConv → split back into Q,K,V

**Canon-C (Pre-MLP/MoE):**
- Operates on: normalized hidden states before MLP/MoE routing
- Effect: MLP (or MoE router) receives locally-mixed representations. In MoE,
  this means routing decisions incorporate neighboring token context.
- Intuition: "Before the router decides which expert handles me, let me
  summarize my local context."
- **This is the highest-value position for NanoSeek** (see §10).

**Canon-D (Inside-MLP):**
- Operates on: concatenated [gate, up] projections inside SwiGLU
- Effect: The gating mechanism and up-projection incorporate local context.
  Gate decisions are influenced by what nearby tokens are doing.
- Intuition: "The gate should be informed by local context, not just my
  own representation."
- Implementation detail: concat gate,up → ShortConv → split back

### Subset configurations from the paper

The paper and code test various combinations:

| Configuration | Description | Best for |
|---------------|-------------|----------|
| Canon-ABCD | All four positions | Maximum improvement, slight overhead |
| Canon-AbCD | A + inside-attn(b) + C + D | Default for linear models (GLA, Mamba2) |
| Canon-AC | Pre-attention + Pre-MLP only | Cost-effective for transformers |
| Canon-C | Pre-MLP only | Minimal intervention, MoE routing benefit |
| Canon-BD | Inside-attention + Inside-MLP | Tests whether internal mixing suffices |

**Key finding:** Even Canon-A alone (simplest, cheapest) provides significant
reasoning depth improvement. But Canon-ABCD is optimal.

---

## 4. The Synthetic Pretraining Playground

Allen-Zhu argues that real-world benchmarks (MMLU, ARC, HellaSwag) are too
noisy to isolate specific architectural capabilities. Part 4.1 introduces
five synthetic datasets, each designed to measure one specific capability
with controlled ground truth.

### The five diagnostic datasets

**Depo (Dependency Depth):**
```
What it measures: Maximum reasoning chain depth
                  "Can the model follow A→B→C→D→...→Z?"

Setup: Multi-hop dependency chains of varying length
       Each hop requires resolving a single fact
       Chain length = number of hops

What we learn: The maximum depth before accuracy drops below threshold
               Standard transformer: ~4 hops
               Transformer + Canon: ~8-16 hops (2-4× improvement)

Why it matters for NanoSeek:
  - NanoSeek has 16 layers → theoretical max reasoning depth = 16 hops
  - MLA compression may reduce effective depth (information bottleneck)
  - Canon could recover lost depth from MLA compression
```

**Brevo (Breadth of Reasoning):**
```
What it measures: Parallel dependency tracking width
                  "Can the model track 5 independent fact chains simultaneously?"

Setup: Multiple independent retrieval chains processed in parallel
       Model must maintain all chains without cross-contamination

What we learn: How many parallel tracks the model can maintain
               Canon adds ~30% more parallel tracking capacity

Why it matters for NanoSeek:
  - MoE's expert specialization naturally segments parallel tracks
  - Canon + MoE might be synergistic for breadth tasks
  - Different experts handle different tracks, Canon provides local context
```

**Capo (Knowledge Capacity / Composition):**
```
What it measures: How much knowledge the model can store and compose
                  Uses bioS (biography-simple) and bioR (biography-rich) generators

Setup: Synthetic biographies with known facts
       Test: retrieve + compose facts across biographies
       Controlled fact density and diversity

What we learn: Knowledge capacity in bits per parameter
               Allen-Zhu's 2-bit/param law (from Part 3.3) is validated here
               Canon adds ~10-15% effective knowledge capacity

Why it matters for NanoSeek:
  - 4.75B total params × 2 bits/param = ~9.5B bits theoretical capacity
  - But only 1.08B params are active per forward pass
  - MoE's knowledge capacity = f(N_total, E, κ) — how Canon interacts is unknown
```

**Mano (Knowledge Manipulation):**
```
What it measures: Ability to transform/manipulate retrieved knowledge
                  "Given fact X, can the model derive/transform to get Y?"

Setup: Knowledge manipulation chains requiring transformation steps
       Each step applies a learned transformation to retrieved facts

What we learn: Length of manipulation chains the model can handle
               Canon extends manipulation length by ~30%

Why it matters for NanoSeek:
  - Knowledge manipulation is downstream of both storage (MoE) and
    retrieval (MLA attention)
  - Canon's local mixing could help chain manipulation steps
```

**Lano (Language Structure):**
```
What it measures: Hierarchical language structure processing
                  Direct connection to Part 1's findings on parse trees

Setup: Synthetic context-free grammars with controlled nesting depth
       Tests whether model builds correct internal parse trees

What we learn: Maximum nesting depth the model handles correctly
               Canon helps especially for deeper nesting structures

Why it matters for NanoSeek:
  - Validates Part 1's depth principle at the architecture level
  - NanoSeek's 16 layers should handle depth-16 structures
  - Canon may allow deeper effective nesting within the same 16 layers
```

### Using these benchmarks for NanoSeek evaluation

**Recommendation:** Port Depo, Brevo, Capo, Mano, Lano from
`github.com/facebookresearch/PhysicsLM4` into NanoSeek's evaluation harness.

```
When to evaluate:
  - End of Phase 1 pretraining (22B tokens) — baseline capabilities
  - After each RL stage — capability changes
  - In Canon ablation study — direct comparison with/without Canon

What to measure:
  - Depo score at hop depths 1, 2, 4, 8, 16
  - Brevo score at breadth 1, 2, 4, 8
  - Capo knowledge density (bits stored per active parameter)
  - Mano manipulation chain length at 80% accuracy
  - Lano nesting depth at 80% accuracy
```

---

## 5. Quantitative Results: The Hard Numbers

### Part 4.1 — Synthetic tasks (controlled, precise measurements)

| Architecture | Metric | Without Canon | With Canon-ABCD | Improvement |
|-------------|--------|---------------|-----------------|-------------|
| Llama (RoPE) | Reasoning depth (Depo) | 4 hops | 8-16 hops | **2-4×** |
| Llama (RoPE) | Reasoning breadth (Brevo) | baseline | +30% | **+30%** |
| Llama (RoPE) | Knowledge manipulation (Mano) | baseline | +30% | **+30%** |
| Llama (RoPE) | Knowledge capacity (Capo) | baseline | +10-15% | **+10-15%** |
| Llama (NoPE) | Overall | weak baseline | **matches RoPE** | architecture rescue |
| GLA | Reasoning depth | 1 hop | 4 hops | **4×** |
| GLA | Reasoning breadth | baseline | 2× baseline | **2×** |
| GLA | Knowledge manipulation | baseline | 2× baseline | **2×** |
| GLA + Canon | vs Mamba2 | inferior | **matches or exceeds** | architecture rescue |

### Part 4.2 — Real-world pretraining (1B, 3B, 8B on Nemotron-CC)

| Model | Scale | Tokens | MMLU Improvement (Canon vs baseline) |
|-------|-------|--------|--------------------------------------|
| LlamaCanon vs Llama | 1B | 1T | ~2% |
| LlamaCanon vs Llama | 3B | 1T | ~2% |
| LlamaCanon vs Llama | 3B | 2T | ~2% |
| LlamaCanon vs Llama | 8B | 1T | ~2% |
| All 8 controlled pairs | 1B-8B | 1T-2T | **Consistent ~2%** |

**Training details for Part 4.2 models:**
```
Sequence length:     16,384 tokens
Batch size:          1 per GPU (128 GPUs total with FSDP)
Total steps:         480,000
Learning rates:      0.002, 0.003, 0.005 (grid searched)
Dataset:             Nemotron-CC (open-source)
Eval benchmarks:     MMLU, BoolQ, PIQA, HellaSwag, RACE, ARC, Winogrande,
                     CommonsenseQA (15+ total)
Eval frequency:      Every 6,000 steps on 8 async GPUs
Checkpoint freq:     Every 3,000 steps
```

### Length generalization finding

```
Without Canon: Models trained at 4K context fail rapidly beyond 4K
With Canon:    ~50% length generalization (4K training → ~6K effective)
               No long-context fine-tuning applied — pure architecture benefit
```

### Cross-architecture comparison (with Canon as equalizer)

When all architectures are equipped with full Canon layers (fair comparison):

| Architecture Family | Knowledge Capacity | Reasoning Depth | Trade-off |
|--------------------|--------------------|-----------------|-----------|
| Transformer (Llama) | baseline | **2-4× deeper** | Best for reasoning |
| Linear (GLA + Canon) | **~40% higher** | baseline | Best for storage |
| SSM (Mamba2) | moderate | moderate | Balanced |

**Critical insight:** Linear models store MORE knowledge but reason LESS deeply.
The depth limitation comes from accumulated compression and retrieval errors
in the recurrent state, not from memory capacity. Canon helps but cannot fully
close the gap.

---

## 6. Part 4.2: Canon at Scale — Real-World Validation

Part 4.2 validates that synthetic pretraining findings transfer to real-world
language modeling. This is the bridge between controlled science (Part 4.1)
and practical engineering (NanoSeek).

### Released artifacts

```
16 base models on HuggingFace:
  facebook/PhysicsLM4.2__Llama-{1B,3B,8B}-Nemo-{1T,2T}-lr{0.002,0.003,0.005}
  facebook/PhysicsLM4.2__LlamaCanon-{1B,3B,8B}-Nemo-{1T,2T}-lr{0.002,0.003,0.005}

48 linear model checkpoints (GLA5, GDN2, Mamba2 with/without Canon)

Full training code: github.com/facebookresearch/PhysicsLM4
  Based on Meta's Lingua framework
  Apache 2.0 license (Canon code) + BSD-3-Clause (Lingua modifications)
```

### Key validation results

1. **Synthetic → Real transfer confirmed:** Trends observed on Depo/Brevo/Capo/Mano/Lano
   match MMLU/HellaSwag/ARC improvements at scale.

2. **Consistency across scales:** ~2% MMLU improvement holds at 1B, 3B, and 8B.
   No diminishing returns within tested range.

3. **Consistency across LR:** Improvement holds across learning rates 0.002-0.005.
   Canon is not sensitive to LR choice.

4. **Length generalization bonus:** LlamaCanon extends to ~150% of training context
   without fine-tuning.

5. **GLA+Canon matches GDN and outperforms Mamba2** on breadth tasks (Brevo),
   confirming synthetic findings transfer.

### Implementation detail: causal-conv1d

Part 4.2 uses `causal-conv1d` CUDA kernel for efficiency:

```python
# pip install causal-conv1d
# Required for efficient ShortConvolution at scale
# Compatible with transformers==4.47.1, 4.53.3
```

For NanoSeek at 1B scale, standard PyTorch nn.Conv1d is sufficient.
The causal-conv1d kernel becomes important at 3B+ with long sequences.

---

## 7. Why Canon Layers Work: First-Principles Analysis

### The information flow topology argument

Standard transformers create a specific computation graph:

```
Token_1  Token_2  Token_3  Token_4  Token_5
  │        │        │        │        │
  ▼        ▼        ▼        ▼        ▼
[Layer 1: attention connects all pairs, then MLP transforms each independently]
  │        │        │        │        │
  ▼        ▼        ▼        ▼        ▼
[Layer 2: attention connects all pairs, then MLP transforms each independently]
  ...
```

**The bottleneck:** Between attention layers, information flows **only through
the residual stream of individual tokens**. Token 3 at layer 2 contains the
result of layer-1's attention over all tokens, but this is a compressed
representation. If Token 3 needs to combine information from Token 1 AND
Token 5, it must do so through the single attention output — a narrow channel.

**Canon layers widen this channel:**

```
Token_1  Token_2  Token_3  Token_4  Token_5
  │        │        │        │        │
  ▼        ▼        ▼        ▼        ▼
[Canon-A: local conv mixes neighboring tokens]
  │╲      ╱│╲      ╱│╲      ╱│╲      ╱│
  │  ╲  ╱  │  ╲  ╱  │  ╲  ╱  │  ╲  ╱  │
  │    ╳    │    ╳    │    ╳    │    ╳    │
  │  ╱  ╲  │  ╱  ╲  │  ╱  ╲  │  ╱  ╲  │
  │╱      ╲│╱      ╲│╱      ╲│╱      ╲│
  ▼        ▼        ▼        ▼        ▼
[Attention: now operates on locally-enriched representations]
```

After Canon-A, Token 3's representation contains weighted information from
Tokens {1, 2, 3, 4} (kernel size 4, causal). When attention then queries
Token 3, it gets a richer signal that already encodes local structure.

### Why even random averaging works

Allen-Zhu shows that **random fixed-weight averaging of neighboring tokens**
provides most of the benefit. This suggests the value is in the **topology**
(connecting neighboring tokens), not in the **learned weights**.

The trainable 1D convolution of kernel size 4 is optimal because:
1. It learns position-dependent mixing (some channels weight token i-1 more,
   others weight token i-3 more)
2. Kernel size 4 captures the sweet spot: enough context for multi-hop chains,
   not so large as to blur token identity
3. Depthwise (groups=dim) means each hidden dimension can learn its own
   mixing pattern — some dimensions may prefer very local mixing, others
   slightly wider

### Why the benefit is architecture-agnostic

The local mixing bottleneck exists in ANY architecture that processes tokens
through sequential layers:

- **Transformers:** Information between attention layers flows only through
  individual token residual streams
- **SSMs (Mamba2):** State updates are sequential; each token sees a compressed
  version of history, losing local detail
- **Linear Attention (GLA):** Similar to SSMs — compressed state loses
  local structure
- **All architectures:** The common bottleneck is that between layer n and
  layer n+1, each token is represented as a single vector, with no explicit
  encoding of its local neighborhood

Canon layers address this universal bottleneck by injecting local context
into every representation before it enters the next processing stage.

---

## 8. Complementary Innovations: QK-Norm and Partial RoPE

### QK-Norm (Query-Key Normalization)

**What:** Apply LayerNorm to query and key vectors before computing attention scores.

```python
# Standard attention:
attn_scores = (q @ k.T) / sqrt(d_k)

# With QK-norm:
q = layer_norm(q)
k = layer_norm(k)
attn_scores = (q @ k.T) / sqrt(d_k)
```

**Why it helps:**
- Prevents attention logit explosion during training (especially at scale)
- Decouples representation magnitude from attention pattern formation
- Makes training more stable without requiring careful initialization

**Evidence:**
- Used in Part 4.2's LlamaCanon models for all scales
- Also adopted independently by Google (ViT-22B), Meta (various), and others
- Confirmed stable training benefit across 1B, 3B, 8B models

**NanoSeek relevance:**
- NanoSeek's MLA already operates in compressed space where magnitude
  instabilities could be amplified
- MoE training is stability-sensitive (expert collapse risk at H_load < 2 bits)
- QK-norm is a zero-risk stability improvement
- **Recommendation: Adopt for NanoSeek Phase 1 rewrite** (see §13 for caveats)

### Partial RoPE

**What:** Apply Rotary Position Embedding to only the first `rope_dim` dimensions
of each attention head, leaving remaining dimensions without positional encoding.

```python
# Standard RoPE: apply to all d_head dimensions
q_rope = apply_rope(q)  # all 128 dims get rotary encoding

# Partial RoPE: apply to first rope_dim dimensions only
q_rope = apply_rope(q[:, :, :rope_dim])     # first 64 dims: positional
q_nope = q[:, :, rope_dim:]                  # last 64 dims: position-free
q = torch.cat([q_rope, q_nope], dim=-1)
```

**Why it helps:**
- Some attention heads benefit from position-free matching (semantic similarity
  regardless of position)
- Others need positional information (local structure, relative ordering)
- Partial RoPE lets both coexist within the same model

**NanoSeek alignment:**
NanoSeek's MLA already implements exactly this pattern:

```
MLA head structure:
  qk_rope_head_dim = 32   → gets RoPE (positional)
  qk_nope_head_dim = 64   → no positional encoding (semantic)
  Total Q head dim = 96 = 32 + 64
  Partial RoPE ratio = 32/96 = 33%
```

This is **independently validated** by Allen-Zhu's finding. DeepSeek V3's MLA
design already incorporates the optimal partial RoPE pattern. No change needed.

---

## 9. NanoSeek Integration Analysis: MLA × Canon Synergy Hypothesis

### The bandwidth bottleneck argument

NanoSeek uses MLA with aggressive KV compression:

```
Standard MHA (Llama-style):
  KV cache per head:  d_head = 128
  KV cache per layer: n_heads × d_head = 16 × 128 = 2048 floats
  Total bandwidth:    Full hidden dimension available for KV

NanoSeek MLA:
  KV compressed to:   kv_lora_rank = 143
  Compression ratio:  2048 / 143 = 14.3× (before expansion via wkv_b)
  Effective bandwidth: 143 latent dims carry ALL KV information for ALL heads
```

**The hypothesis:** MLA's extreme compression creates a bottleneck in the
global horizontal information flow (Axis 2). Each attention head's key/value
information is derived from the same 143-dimensional compressed representation.
While this is sufficient for most retrieval patterns (MLA works well in practice),
it may lose fine-grained local structure.

**Canon layers provide a complementary channel:**
- Canon operates on the full hidden dimension (2048), not the compressed KV (143)
- Canon provides dense local mixing without going through the KV bottleneck
- The combination is potentially synergistic: MLA handles global patterns
  efficiently, Canon handles local patterns at full bandwidth

### Expected interaction by Canon position

| Canon Position | Interaction with MLA | Expected Benefit |
|----------------|---------------------|------------------|
| Canon-A (pre-attn) | Enriches input to MLA projections | MLA's Q/K projections start with locally-mixed representations. May improve attention quality in compressed space |
| Canon-B (inside-attn) | **Complex with MLA** | MLA projects to compressed space. Canon-B would operate on compressed Q/K/V. Benefit unclear — compression already mixes |
| Canon-C (pre-MoE) | Independent of MLA | Enriches post-attention representation before MoE routing. Clean separation from MLA |
| Canon-D (inside-MLP) | Independent of MLA | Enriches gate/up inside SwiGLU experts. Clean separation from MLA |

**Key insight for Canon-B in MLA:**
In standard attention, Canon-B operates on Q,K,V in head-dimension space.
In MLA, the Q projection goes through compressed q_a (q_lora_rank=440),
and KV goes through compressed kv_c (kv_lora_rank=143). Canon-B would
need to operate in this compressed space, which already has implicit mixing
from the low-rank projection. The benefit is unclear and may even be negative
(double-mixing could blur representations).

**Recommendation:** For NanoSeek, test Canon-ACD (skip B). If B is tested,
operate on the compressed space, not the expanded Q/K/V.

### Predicted NanoSeek-specific results

Based on the MLA bandwidth argument:

```
Prediction 1: Canon provides LARGER reasoning depth improvement for MLA
              than for standard MHA, because MLA's compression creates a
              larger local bandwidth deficit for Canon to fill.
              Expected: 3-5× reasoning depth improvement (vs 2-4× for MHA)

Prediction 2: Canon-C provides disproportionate benefit for NanoSeek because
              it enriches MoE routing (not just MLP input). This interaction
              doesn't exist in dense models.
              Expected: I_spec improvement (better expert specialization)

Prediction 3: Canon-B provides SMALLER benefit (or no benefit) for MLA
              compared to standard MHA, because MLA's compression already
              performs implicit cross-dimension mixing.
              Expected: Canon-ACD ≈ Canon-ABCD for MLA architectures

These are testable hypotheses for the ablation study (§12).
```

---

## 10. NanoSeek Integration Analysis: MoE × Canon-C Interaction

### Why Canon-C is highest-priority for NanoSeek

In a dense transformer, Canon-C enriches the input to an MLP — a per-token
feedforward network. In NanoSeek's MoE, Canon-C enriches the input to the
**routing decision** — the gate that selects which 8 of 64 experts process
each token.

```
WITHOUT Canon-C:
  Token representation → Router → Expert selection → Expert processing

WITH Canon-C:
  Token representation → Canon-C (local mixing) → Router → Expert selection → Expert processing
```

The router sees a richer representation that incorporates local context:

```
Example: Token "the" at position 15 in "The cat sat on the mat"
                                                     ^

Without Canon-C:
  Router sees: embedding("the") + residual from prior layers
  → May route to generic function-word expert

With Canon-C:
  Router sees: weighted_sum(embed("on"), embed("the"), embed("mat"), ...)
  → Richer context: "the" in "on the mat" context
  → May route to spatial/prepositional expert instead
  → Better expert specialization
```

### Predicted effects on MoE metrics

**H_load (Expert Load Balance Entropy):**
```
Without Canon-C: H_load depends on token-level routing decisions
With Canon-C:    Routing decisions informed by local context
                 → More discriminative routing
                 → Could go either direction:
                   - Better balance (tokens route more appropriately)
                   - Worse balance (contextual routing may cluster)
                 → Need to measure empirically
```

**I_spec (Expert Specialization Mutual Information):**
```
Without Canon-C: I_spec measures MI(expert_id; domain)
With Canon-C:    Routing has richer input → more semantic routing decisions
                 → Prediction: I_spec INCREASES
                 → Experts develop clearer domain specialization
                 → This is a POSITIVE outcome
```

**Expert collapse risk:**
```
Canon-C provides richer features to the router.
Richer features → more discriminative routing → potentially LESS collapse risk.
However, if local context causes routing to cluster on local patterns
(e.g., all tokens in a sentence go to same experts), collapse risk increases.

Mitigation: NanoSeek already uses auxiliary-loss-free balancing with
per-expert bias (gamma). This should counteract any clustering tendency.
```

### Interaction with shared experts

NanoSeek has 2 shared experts (always-active) + 8/64 routed experts per token.
Canon-C affects routed expert selection but not shared expert computation.

```
Post-attention hidden state
    │
    ▼
  Canon-C (local mixing)
    │
    ├──→ Shared Expert 1 (always active, sees Canon-C enriched input)
    ├──→ Shared Expert 2 (always active, sees Canon-C enriched input)
    │
    ├──→ Router (sigmoid scoring, sees Canon-C enriched input)
    │      │
    │      ▼
    │    Top-8 expert selection (informed by local context)
    │      │
    │      ▼
    ├──→ Routed Expert i (Canon-C enriched input)
    ├──→ Routed Expert j (Canon-C enriched input)
    │    ... (8 selected experts)
    │
    ▼
  Weighted sum of all expert outputs
```

**Observation:** Shared experts also receive Canon-C enriched input. This
could make shared experts more effective at capturing common local patterns,
potentially changing the division of labor between shared and routed experts.

---

## 11. Implementation Specification for NanoSeek

### Configuration additions to config.py

```python
@dataclass
class CanonConfig:
    """
    Canon Layer configuration.

    Canon layers are lightweight depthwise 1D convolutions that promote
    horizontal information flow across neighboring tokens.

    Reference: Allen-Zhu (2025), Physics of Language Models Part 4.1
               arXiv:2512.17351, NeurIPS 2025
    """
    # Enable/disable Canon layers
    enabled: bool = False  # Off by default; enable for ablation study

    # Which positions to insert Canon layers
    # A=pre-attention, B=inside-attention, C=pre-MLP, D=inside-MLP
    canon_set: str = "ACD"  # Skip B for MLA (see §9)

    # Convolution kernel size (default from paper: 4)
    kernel_size: int = 4

    # Whether to add residual connection
    residual: bool = True

    # Whether to include bias in Conv1d
    bias: bool = True
```

### Module implementation

```python
class ShortConvolution(nn.Module):
    """
    Canon Layer: depthwise causal 1D convolution for local token mixing.

    Computes weighted sum of neighboring tokens for each hidden dimension
    independently. Kernel size 4 means each token sees itself and 3 prior
    tokens (causal).

    Reference: Allen-Zhu (2025), "Physics of Language Models: Part 4.1,
    Architecture Design and the Magic of Canon Layers." arXiv:2512.17351
    """
    def __init__(
        self,
        dim: int,
        kernel_size: int = 4,
        residual: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            groups=dim,                     # depthwise: independent per channel
            padding=kernel_size - 1,        # left-pad for causal
            bias=bias,
        )
        self.residual = residual
        self.kernel_size = kernel_size

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            (batch, seq_len, dim) with local token mixing applied
        """
        B, L, D = x.shape
        h = x.transpose(1, 2)                          # (B, D, L)
        h = self.conv(h)[:, :, :L]                      # causal: truncate future
        h = h.transpose(1, 2)                           # (B, L, D)
        return h + x if self.residual else h
```

### Integration into DecoderLayer

```python
class NanoSeekDecoderLayer(nn.Module):
    def __init__(self, config: NanoSeekConfig, layer_idx: int):
        super().__init__()
        # ... existing components ...

        # Canon layers (optional, for ablation study)
        self.canon_config = config.canon
        if self.canon_config.enabled:
            if "A" in self.canon_config.canon_set:
                self.canon_a = ShortConvolution(
                    config.hidden_size,
                    kernel_size=self.canon_config.kernel_size,
                    residual=self.canon_config.residual,
                )
            if "C" in self.canon_config.canon_set:
                self.canon_c = ShortConvolution(
                    config.hidden_size,
                    kernel_size=self.canon_config.kernel_size,
                    residual=self.canon_config.residual,
                )
            if "D" in self.canon_config.canon_set:
                # Canon-D operates on concatenated [gate, up] inside SwiGLU
                # Dimension = 2 × intermediate_size (gate and up projections)
                canon_d_dim = 2 * self._get_intermediate_size(config, layer_idx)
                self.canon_d = ShortConvolution(
                    canon_d_dim,
                    kernel_size=self.canon_config.kernel_size,
                    residual=self.canon_config.residual,
                )

    def forward(self, x, ...):
        # Pre-attention
        h = self.input_layernorm(x)

        # Canon-A: pre-attention local mixing
        if self.canon_config.enabled and hasattr(self, 'canon_a'):
            h = self.canon_a(h)

        # MLA attention
        attn_output = self.self_attn(h, ...)
        x = x + attn_output

        # Pre-MLP/MoE
        h = self.post_attention_layernorm(x)

        # Canon-C: pre-MLP/MoE local mixing (HIGHEST VALUE for MoE routing)
        if self.canon_config.enabled and hasattr(self, 'canon_c'):
            h = self.canon_c(h)

        # MoE or dense FFN (Canon-D applied inside if enabled)
        mlp_output = self.mlp(h, ...)  # Canon-D goes inside MLP/Expert
        x = x + mlp_output

        return x
```

### Canon-D inside SwiGLU expert

```python
class Expert(nn.Module):
    def __init__(self, config, canon_d=None):
        super().__init__()
        self.gate_proj = CastLinear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.up_proj = CastLinear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.down_proj = CastLinear(config.moe_intermediate_size, config.hidden_size, bias=False)
        self.canon_d = canon_d  # Shared across experts or per-expert

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # Canon-D: mix gate+up across neighboring tokens
        if self.canon_d is not None:
            gu = torch.cat([gate, up], dim=-1)  # (B, L, 2*inter)
            gu = self.canon_d(gu)
            gate, up = gu.chunk(2, dim=-1)

        return self.down_proj(F.silu(gate) * up)
```

### QK-Norm integration into MLA

```python
# Inside MLA.forward(), after computing q and k in head space:

if self.qk_norm:
    q = F.layer_norm(q, (self.qk_head_dim,))
    k = F.layer_norm(k, (self.qk_head_dim,))

# Then proceed with attention computation
```

Where `qk_head_dim = qk_nope_head_dim + qk_rope_head_dim = 64 + 32 = 96`

---

## 12. Experimental Design: Canon Layer Ablation Study

### When to run

**NOT during Phase 1-4.** Canon layers are an enhancement, not a correction.
The DeepSeek V3.2 reimplementation must be completed and validated first.

**Run after NanoSeek-1B training completes** (post-Week 8), as an additional
research contribution.

### Experiment matrix

All experiments at **anchor scale (~55M active)** for cost efficiency.
Use muP-transferred hyperparameters from the validated anchor config.

```
Experiment 0: NanoSeek-Anchor baseline (existing, no Canon)
              → This already exists from Phase 3 HP search

Experiment 1: NanoSeek-Anchor + Canon-ABCD (all 4 positions, skip B for MLA)
              → Actually Canon-ACD since B is skipped for MLA
              → Full Canon treatment

Experiment 2: NanoSeek-Anchor + Canon-C only (pre-MoE routing)
              → Tests the MoE-specific hypothesis
              → Cheapest useful intervention

Experiment 3: NanoSeek-Anchor + Canon-AC (pre-attention + pre-MoE)
              → Tests attention + routing enrichment

Experiment 4: NanoSeek-Anchor + Canon-ACD + QK-norm
              → Tests whether QK-norm and Canon are synergistic or redundant
```

### Metrics to collect

For each experiment, log:

```
Standard training metrics:
  - ema_val_bpb (primary quality metric)
  - train_loss trajectory
  - gradient norm

MoE-specific metrics:
  - H_load (expert load balance entropy) — Canon-C effect on routing balance
  - I_spec (expert specialization MI) — Canon-C effect on semantic specialization
  - Per-expert activation frequency — routing distribution changes

Allen-Zhu diagnostic benchmarks:
  - Depo: reasoning depth at hop 1, 2, 4, 8, 16
  - Brevo: reasoning breadth at width 1, 2, 4, 8
  - Capo: knowledge capacity (bits per active parameter)
  - Mano: manipulation chain length at 80% accuracy
  - Lano: nesting depth at 80% accuracy

Standard benchmarks:
  - MMLU, ARC-E, ARC-C, HellaSwag, BoolQ
  - (Port from nanochat/core_eval.py)

MLA-specific:
  - Attention entropy (does Canon-A change attention patterns?)
  - KV cache utilization (does Canon change what MLA stores?)
```

### Hypotheses to test

```
H1: Canon-ACD improves ema_val_bpb by ≥0.02 (≥1% relative improvement)
    Basis: ~2% MMLU improvement at 1B-8B with standard MHA

H2: Canon-C alone improves I_spec by ≥0.1 nats
    Basis: richer router input → more discriminative routing → better specialization

H3: Canon provides LARGER Depo improvement for MLA than for standard MHA
    Basis: MLA's KV compression creates larger local bandwidth deficit

H4: Canon-B provides no significant additional benefit over Canon-ACD for MLA
    Basis: MLA's low-rank projection already performs implicit mixing

H5: H_load remains stable (within ±0.5 bits) with Canon-C
    Basis: auxiliary-loss-free balancing counteracts any routing clustering

H6: Canon + MoE ema_val_bpb improvement > Canon + dense ema_val_bpb improvement
    Basis: MoE routing benefits from richer inputs (multiplicative effect)
```

### Estimated compute cost

```
Anchor scale: ~55M active, ~280M total
Training: ~2B tokens (Chinchilla-optimal for 55M)
Per experiment: ~$5-10 on 1×H100

5 experiments × ~$8 average = ~$40 total
Timeline: 1-2 days (experiments can run in parallel)
```

### Novel research contribution

If results confirm the hypotheses, this produces a paper-worthy finding:

**"Canon Layers × MLA × MoE: Local Mixing Synergizes with Compressed Attention
and Expert Routing"**

Key claims:
1. Canon layers provide disproportionate benefit for MLA architectures
   (due to KV compression bandwidth deficit)
2. Canon-C before MoE routing improves expert specialization (I_spec)
   without compromising load balance (H_load)
3. The combination of Canon (local horizontal) + MLA (compressed global
   horizontal) + MoE (conditional computation) creates a uniquely efficient
   information flow topology

No prior work has tested Canon layers with MLA or MoE architectures.

---

## 13. What NOT to Do: Anti-Patterns and Timing

### Do NOT add Canon layers during Phase 1 rewrite

```
❌ WRONG: Add Canon to model.py during the Phase 1 DeepSeek V3.2 reimplementation

WHY: Canon is a Meta/FAIR innovation, not part of DeepSeek V3.2.
     Adding it now:
     1. Confounds the reimplementation (is a bug in our code or Canon's effect?)
     2. Breaks muP HP transfer (Canon adds parameters → width semantics change)
     3. Makes NanoSeek no longer a faithful DeepSeek V3.2 reproduction
     4. Prevents clean baseline comparison

✅ RIGHT: Complete Phase 1-4 as pure DeepSeek V3.2, then add Canon as ablation
```

### Do NOT add QK-norm without understanding MLA interaction

```
❌ WRONG: Blindly add QK-norm to MLA because Allen-Zhu's paper shows it helps

WHY: MLA operates in compressed Q/K space:
     - Q goes through q_lora_rank (440) then projects to per-head Q
     - K comes from kv_lora_rank (143) compressed representation
     - The Q/K vectors are already in a specific scale regime from the
       low-rank projection

     QK-norm in MLA requires careful thought:
     - Apply AFTER rope split? Or before?
     - Apply to compressed space or expanded space?
     - Does it interact with the mscale factor?

✅ RIGHT: If adding QK-norm, apply to the final Q and K vectors (after RoPE
          application, in per-head space). Test at anchor scale first.
          This matches how LlamaCanon applies QK-norm in standard attention.
```

### Do NOT conflate Canon with DSA

```
❌ WRONG: "Canon layers make DSA unnecessary" or "DSA is like Canon"

WHY: Canon and DSA solve different problems:
     - Canon: adds LOCAL horizontal information flow (kernel size 4)
     - DSA: reduces GLOBAL attention cost from O(L²) to O(Lk)
     Canon makes attention inputs richer; DSA makes attention computation cheaper.
     They are orthogonal and can coexist.

✅ RIGHT: Canon and DSA are independent optimizations. Canon-A enriches the
          input to attention (richer representations). DSA then selects which
          tokens to attend to (cheaper computation). No conflict.
```

### Do NOT use Canon to fix expert collapse

```
❌ WRONG: "If H_load drops below 2 bits, add Canon-C to fix routing"

WHY: Expert collapse is a training dynamics problem (gradient signals, bias
     terms, initialization). Canon-C can make routing more discriminative,
     but if experts have already collapsed, richer routing input won't help —
     the expert weights themselves are degenerate.

✅ RIGHT: Fix expert collapse with:
          1. gamma bias adjustment (auxiliary-loss-free, already in NanoSeek)
          2. Router initialization
          3. Expert weight reinitialization (last resort)
          Canon-C is a preventive measure (better routing from the start),
          not a corrective measure (fixing collapsed experts).
```

---

## 14. Principles Extracted

These principles are additive to the 22 principles in
`PHYSICS_OF_LANGUAGE_MODELS_ANALYSIS.md` (§14).

```
PRINCIPLE 23: Transformer information flow has three axes (vertical, global
              horizontal, local horizontal). Standard architectures are
              missing Axis 3. Canon layers fill this gap with negligible
              parameter cost (0.01% overhead).

PRINCIPLE 24: The value of local token mixing is in the TOPOLOGY (connecting
              neighbors), not the learned WEIGHTS. Even random averaging helps.
              Trainable depthwise Conv1d(kernel=4) is optimal but not required.

PRINCIPLE 25: MLA's KV compression creates a bandwidth deficit in global
              horizontal flow (Axis 2). Canon layers provide a complementary
              full-bandwidth local channel (Axis 3). The combination is
              predicted to be synergistic — MLA handles global patterns at
              reduced cost, Canon handles local patterns at full bandwidth.

PRINCIPLE 26: Canon-C (pre-MoE routing) is uniquely valuable for MoE
              architectures. It enriches routing decisions with local context,
              potentially improving expert specialization (I_spec) without
              requiring any changes to the routing algorithm itself.

PRINCIPLE 27: Synthetic diagnostic benchmarks (Depo, Brevo, Capo, Mano, Lano)
              are more precise than aggregate benchmarks (MMLU, ARC) for
              measuring specific architectural effects. Use them for ablation
              studies, even if the model won't be deployed on synthetic tasks.

PRINCIPLE 28: Architecture improvements and training improvements are
              orthogonal axes. Canon layers (architecture) can be added
              post-hoc without retraining if the goal is evaluation.
              But for maximum benefit, they should be present during training
              to influence learned representations.
```

---

## 15. Open Research Questions

These are questions that NanoSeek's Canon ablation study could answer,
contributing novel findings to the research community.

### Q1: Does MLA amplify or diminish Canon's benefit?

```
Hypothesis: MLA amplifies (due to KV bandwidth deficit)
Test: Compare Canon improvement for NanoSeek-MLA vs NanoSeek-dense-MHA baseline
      at matched active parameter count
Metric: Depo reasoning depth improvement ratio
Expected: MLA + Canon > MHA + Canon (in relative improvement)
```

### Q2: Does Canon-C improve expert specialization without hurting load balance?

```
Hypothesis: I_spec increases, H_load stays stable
Test: Canon-C only experiment, measure I_spec and H_load trajectories
Metric: I_spec (expert specialization MI), H_load (load balance entropy)
Expected: I_spec ↑ 0.1-0.3 nats, H_load within ±0.5 bits
```

### Q3: Is Canon-B unnecessary for MLA architectures?

```
Hypothesis: Canon-B provides no additional benefit over Canon-ACD for MLA
Test: Canon-ABCD vs Canon-ACD, all other factors equal
Metric: ema_val_bpb, Depo, Brevo
Expected: No significant difference → confirms MLA implicit mixing hypothesis
```

### Q4: Does Canon interact with MTP?

```
Hypothesis: Canon improves MTP acceptance rate (local mixing helps next-token prediction)
Test: Measure MTP acceptance rate with/without Canon
Metric: MTP acceptance rate at end of training
Expected: +5-10% acceptance rate improvement → better speculative decoding
Rationale: Next-token prediction inherently benefits from local context;
           Canon makes each token's representation richer for predicting its successor
```

### Q5: Does Canon change DSA's indexer behavior?

```
Hypothesis: Canon-A enriches attention input → DSA's indexer selects different top-k tokens
Test: Compare DSA indexer selections with/without Canon-A in Phase 2
Metric: Indexer selection overlap, indexer KL-div loss convergence
Expected: Indexer converges faster with Canon-A (better input representations)
```

### Q6: Does Canon transfer across scales via muP?

```
Hypothesis: Canon layer weights are width-independent (depthwise → no cross-channel interaction)
Test: Train Canon at anchor scale, check if benefit transfers to 500M and 1B
Metric: Relative ema_val_bpb improvement at each scale
Expected: Similar relative improvement across scales (Canon is width-agnostic)
Note: Canon's depthwise structure means it should transfer perfectly under muP
      because kernel weights are per-channel and muP scales per-channel operations
```

---

## 16. References

### Primary sources

1. **Part 4.1:** Allen-Zhu, Z. (2025). "Physics of Language Models: Part 4.1,
   Architecture Design and the Magic of Canon Layers."
   [arXiv:2512.17351](https://arxiv.org/abs/2512.17351). NeurIPS 2025.

2. **Part 4.2:** Allen-Zhu, Z. (2025). "Physics of Language Models: Part 4.2,
   Canon Layers at Scale where Synthetic Pretraining Resonates in Reality."
   [GitHub](https://github.com/facebookresearch/PhysicsLM4).

### Related Physics of LM papers (cross-referenced)

3. **Part 1:** Allen-Zhu, Z. & Li, Y. (2023). "Physics of Language Models:
   Part 1, Learning Hierarchical Language Structures."
   [arXiv:2305.13673](https://arxiv.org/abs/2305.13673). ICML 2024.
   → Depth determines reasoning capacity; Canon extends effective depth.

4. **Part 3.3:** Allen-Zhu, Z. & Li, Y. (2024). "Physics of Language Models:
   Part 3.3, Knowledge Capacity Scaling Laws."
   [arXiv:2404.05405](https://arxiv.org/abs/2404.05405). ICLR 2025.
   → 2-bit/param bound; Canon adds ~10-15% effective capacity.

### NanoSeek cross-references

- `PHYSICS_OF_LANGUAGE_MODELS_ANALYSIS.md` — Overview of full research program,
  §8 and §9 cover Parts 4.1 and 4.2 briefly. This document is the deep dive.
- `PAPER_ANALYSIS_V3_V32.md` — DeepSeek V3/V3.2 ground truth (higher authority).
- `REIMPLEMENTATION_PLAN.md` — Build spec. Canon is NOT part of Phase 1.
- `docs/01_MLA_DEEP_DIVE.md` — MLA theory. Canon-B interacts with MLA (§9).
- `docs/02_MOE_DEEP_DIVE.md` — MoE theory. Canon-C interacts with routing (§10).

### Implementation references

- [PhysicsLM4 Architecture (DeepWiki)](https://deepwiki.com/facebookresearch/PhysicsLM4/2-architecture-and-core-concepts) — Canon-ABCD implementation details
- [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) — CUDA-optimized 1D causal convolution
- [HuggingFace PhysicsLM4.2 models](https://huggingface.co/facebook/PhysicsLM4.2__Llama-3B-Nemo-1T-lr0.003) — Pretrained Canon models for reference

---

*Document generated: 2026-03-14*
*Authority level: Reference document (does not override PAPER_ANALYSIS_V3_V32.md or REIMPLEMENTATION_PLAN.md)*
*Purpose: Deep dive into Canon Layers architecture from Physics of Language Models, with NanoSeek-specific integration analysis and experimental design*
