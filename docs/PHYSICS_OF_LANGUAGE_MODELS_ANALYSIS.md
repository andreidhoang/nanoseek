# Physics of Language Models × NanoSeek
## Complete Analysis: First Principles, Core Ideas, and Integration with SCALING_LAB_PLAN
### March 2026 — Based on Zeyuan Allen-Zhu's Research Program (2023–2025)

---

## Table of Contents

1. [Research Program Overview](#1-research-program-overview)
2. [Part 1: Hierarchical Language Structures](#2-part-1-hierarchical-language-structures)
3. [Part 2.1: Grade-School Math — Hidden Reasoning Process](#3-part-21-hidden-reasoning-process)
4. [Part 2.2: Learning From Mistakes](#4-part-22-learning-from-mistakes)
5. [Part 3.1: Knowledge Storage and Extraction](#5-part-31-knowledge-storage-and-extraction)
6. [Part 3.2: Knowledge Manipulation](#6-part-32-knowledge-manipulation)
7. [Part 3.3: Knowledge Capacity Scaling Laws](#7-part-33-knowledge-capacity-scaling-laws)
8. [Part 4.1: Architecture Design — Canon Layers](#8-part-41-canon-layers)
9. [Part 4.2: Canon at Scale](#9-part-42-canon-at-scale)
10. [Impact Matrix: Physics of LM × NanoSeek Pillars](#10-impact-matrix)
11. [Detailed Impact by Pillar](#11-detailed-impact-by-pillar)
12. [Recommended Video Watching Order](#12-recommended-video-watching-order)
13. [5 Concrete Changes to SCALING_LAB_PLAN](#13-concrete-changes)
14. [Fundamental Principles Summary](#14-fundamental-principles)
15. [References](#15-references)

---

## 1. Research Program Overview

### What Makes This Research Unique

Zeyuan Allen-Zhu's "Physics of Language Models" program takes a fundamentally different
approach from typical LLM research. Instead of benchmarking commercial models (GPT-4,
Llama, etc.), it creates **synthetic controlled environments** to discover **universal laws**
governing how language models learn, reason, and store knowledge.

The analogy is precise: a physicist doesn't study individual bridges to understand mechanics —
they derive F=ma from idealized experiments. Allen-Zhu derives the equivalent of F=ma for
language models: laws that hold across all architectures and scales.

**Key methodological principles:**
- **Synthetic data with known ground truth** — eliminates confounds from real data
- **Controlled single-variable experiments** — isolates specific mechanisms
- **Universal laws, not model-specific observations** — results apply to GPT, Llama, Mistral, and NanoSeek equally
- **Falsifiable predictions** — each finding generates testable hypotheses

### Research Structure

The series is organized into two ICML tutorials plus NeurIPS publications:

```
Tutorial I (ICML 2024) — Foundational Components:
  Part 1:   Hierarchical Language Structures (100 min)
  Part 2.1: Grade-School Math — Hidden Reasoning Process (60 min)
  Part 2.2: Learning From Mistakes (50 min)
  Part 3.1: Knowledge Storage and Extraction (80 min, combined with 3.2/3.3)
  Part 3.2: Knowledge Manipulation
  Part 3.3: Knowledge Capacity Scaling Laws

Tutorial II — Architecture Design:
  Part 4.1: Synthetic Pretrain & Architecture Design (150 min)
  Part 4.2: Canon at Scale
```

### Authors and Venue History

- **Zeyuan Allen-Zhu** (Meta FAIR → independent) and **Yuanzhi Li** (co-PI)
- Additional co-authors per paper: Tian Ye, Zicheng Xu, Xiaoli Xu
- Publications at: ICML 2024, ICLR 2025 (multiple papers), NeurIPS 2025

---

## 2. Part 1: Hierarchical Language Structures

**Paper:** [arXiv:2305.13673](https://arxiv.org/abs/2305.13673)
**Venue:** ICML 2024

### Core Research Question

Can GPT-style autoregressive models learn and reason over hierarchical structures
(context-free grammars), and if so, what internal mechanism do they use?

### Key Findings

**Finding 1: Transformers learn to simulate dynamic programming on CFGs.**
- Hidden states precisely capture the parse tree structure of context-free grammars
- Attention patterns implement information passing that mirrors dynamic programming
- This is not an approximation — the model builds exact structural representations

**Finding 2: Autoregressive > Encoder-only for structural reasoning.**
- GPT-style models accurately learn deep CFG hierarchies
- BERT/DeBERTa (encoder-only) struggle with deep structural reasoning
- Uniform attention alone is surprisingly effective for hierarchy processing

**Finding 3: Depth is the critical dimension for hierarchical reasoning.**
- Deeper models process deeper parse trees
- Width (hidden dimension) improves representation quality but not depth of reasoning
- There is a hard minimum depth for each level of structural complexity

### First Principles Extracted

```
PRINCIPLE 1: The transformer is not a "bag of features" — it builds
             structured parse trees incrementally in hidden states.

PRINCIPLE 2: Depth determines maximum reasoning depth. A 4-layer model
             cannot represent structures requiring 8 levels of nesting.

PRINCIPLE 3: Autoregressive generation naturally aligns with incremental
             parsing — each token extends the parse tree by one step.
```

### Relevance to NanoSeek

| NanoSeek Component | Connection | Implication |
|---|---|---|
| Series A scale sweep | Depth varies from 4→16 layers | Expect discontinuous capability jumps at depth thresholds |
| MLA (latent attention) | Compressed representations still capture hierarchy | MLA's 23× compression must preserve structural information |
| Architecture co-variation | Depth-to-width ratio changes across configs | Not just a confound — it's a capability dimension |

---

## 3. Part 2.1: Grade-School Math — Hidden Reasoning Process

**Paper:** [arXiv:2407.20311](https://arxiv.org/abs/2407.20311)
**Venue:** ICLR 2025

### Core Research Questions

1. Do language models develop genuine reasoning skills or memorize templates?
2. What is the model's hidden (mental) reasoning process?
3. Do models use skills similar to humans for mathematical reasoning?
4. Do models develop reasoning skills beyond what's necessary for the training distribution?
5. What mental processes cause reasoning mistakes?
6. How large/deep must a model be for effective math reasoning?

### Key Findings

**Finding 1: Models develop genuine reasoning, not template matching.**
- Using synthetic datasets mimicking real math problems with hierarchical and
  dependency structures, models achieve high accuracy through actual computation
- Verified via V-probing: internal representations encode the problem structure,
  not surface patterns

**Finding 2: Internal planning precedes token generation.**
- V-probing reveals models **pre-compute a solution plan** in hidden states before
  generating solution tokens
- The model processes dependency chains internally, then emits tokens expressing the plan
- This planning is not visible in the output — it happens in hidden state computation

**Finding 3: Chain-of-thought externalizes an existing internal process.**
- CoT works because it provides intermediate token positions where the model can
  "write down" its internal plan, enabling deeper reasoning than a single forward pass
- Without CoT, the plan exists but cannot be fully expressed for complex problems
  that exceed the model's single-pass reasoning depth

**Finding 4: Model depth determines maximum reasoning depth.**
- There is a quantifiable minimum model depth for each level of mathematical complexity
- Shallow models literally cannot represent deep dependency chains
- This is not a training issue — it's an architectural capacity constraint

**Finding 5: Models learn transferable reasoning skills.**
- Skills learned on one problem type transfer to novel problem structures
- The model develops general dependency-tracking abilities, not problem-specific solutions

### First Principles Extracted

```
PRINCIPLE 4: Reasoning = internal planning + sequential expression.
             CoT works by providing "scratch space" for plan expression.

PRINCIPLE 5: Maximum reasoning depth is bounded by model depth.
             No amount of data or training can overcome this architectural limit.

PRINCIPLE 6: Reasoning skills are transferable — the model learns
             general dependency-tracking, not problem-specific templates.
```

### Relevance to NanoSeek

| NanoSeek Component | Connection | Implication |
|---|---|---|
| RL Stage 1 (Reasoning) | GRPO reward for math reasoning | Reward should measure genuine reasoning improvement, not just answer correctness |
| MTP (Multi-Token Prediction) | Internal planning relates to multi-step prediction | MTP acceptance rate may correlate with internal planning quality |
| Model depth (16 layers) | Minimum depth for math reasoning | 16 layers may limit achievable reasoning depth — quantify this |
| Process bonus (+0.2 for self-correction) | V-probing validates internal error detection | Self-correction is a legitimate internal capability, not a surface trick |

### V-Probing Methodology (Implementable)

V-probing is a diagnostic technique to examine what information is encoded in
hidden states at each layer. Implementation sketch:

```python
# For each layer l and position t:
# 1. Extract hidden state h_l^t during forward pass on math problems
# 2. Train a linear probe: W @ h_l^t → predicted_dependency_structure
# 3. Measure accuracy: does the hidden state encode the problem's
#    dependency graph at this layer?
# 4. Key diagnostic: at which layer does the model "know" the answer
#    before it generates the answer tokens?
```

This can be added to NanoSeek's evaluation suite to measure whether GRPO
actually improves the internal reasoning process, not just output formatting.

---

## 4. Part 2.2: Learning From Mistakes

**Paper:** [arXiv:2408.16293](https://arxiv.org/abs/2408.16293)
**Venue:** ICLR 2025

### Core Research Questions

1. Why do language models make reasoning mistakes in the first place?
2. Why don't they correct mistakes immediately during generation?
3. Can error-correction data in pretraining improve reasoning accuracy?

### Key Findings

**Finding 1: Errors arise from distribution shift, not capability limitation.**
- Even with perfect (error-free) training data, models make mistakes during generation
- Root cause: autoregressive generation creates states the model never encountered
  during training — once it generates a wrong intermediate step, subsequent steps
  are conditioned on an out-of-distribution context
- This is fundamental, not fixable by scaling or better data alone

**Finding 2: Post-generation error correction ("retry upon regret") is weak.**
- Retry mechanisms are minimally effective unless error detection is near-perfect
- The model must detect that it made an error, which requires the same reasoning
  capability that failed in the first place
- This creates a chicken-and-egg problem for inference-time correction

**Finding 3: Error-correction data in PRETRAINING dramatically helps.**
- Training data containing "wrong step → correction → continue" patterns
  teaches the model to recover from distributional shift mid-generation
- This outperforms training on the same amount of error-free data
- The model learns a general "recovery" skill, not problem-specific corrections

**Finding 4: "Fake mistake" augmentation is nearly as effective as real mistakes.**
- Simple augmentation: take a correct solution, insert a future step as a
  "fake error," then correct it and continue with the original solution
- This is cheap to generate and nearly as effective as manually crafted retry data
- Practical implication: error-correction augmentation scales with your data pipeline

### First Principles Extracted

```
PRINCIPLE 7: Model errors are fundamentally caused by distribution shift
             between training and autoregressive generation.

PRINCIPLE 8: Robustness to distribution shift must be TRAINED IN during
             pretraining, not PROMPTED IN during inference.

PRINCIPLE 9: Error-correction data in pretraining > error-correction
             via RL or inference-time retry. The order matters.
```

### Relevance to NanoSeek

| NanoSeek Component | Connection | Impact |
|---|---|---|
| 22B token training data | Error-correction augmentation | Allocate 2-5% of tokens to mistake-correction sequences |
| 10% FIM format | Same principle — train on "unusual" token orderings | FIM is a form of distribution-shift robustness training |
| RL Stage 1 process bonus | +0.2 for self-correction in CoT | Validated by this paper, but pretraining-stage correction is stronger |
| MTP training | MTP forces the model to predict multiple future tokens | MTP may naturally create "fake mistake" scenarios when prediction is wrong |

### Specific Data Augmentation Technique

From the paper, the cheapest effective augmentation for NanoSeek:

```
Original solution: Step 1 → Step 2 → Step 3 → Step 4 → Answer
Augmented:         Step 1 → Step 2 → [Step 4 as fake error] →
                   "Wait, let me reconsider." → Step 3 → Step 4 → Answer

This teaches the model:
1. Recognize when a step doesn't follow from the previous context
2. Backtrack to the correct computation
3. Continue normally after recovery
```

---

## 5. Part 3.1: Knowledge Storage and Extraction

**Paper:** [arXiv:2309.14316](https://arxiv.org/abs/2309.14316)
**Venue:** ICML 2024

### Core Research Question

Do LLMs answer knowledge questions by exposure to similar questions during training
(cheating), or by genuinely learning to extract knowledge from raw text sources?

### Key Findings

**Finding 1: Models genuinely learn to extract knowledge from text.**
- Using controlled biographical data, the study shows models can extract facts
  from text they've only seen in narrative form (never in Q&A format)
- This is genuine knowledge extraction, not pattern matching

**Finding 2: Knowledge extraction requires data DIVERSITY.**
- Strong correlation between extraction ability and diversity measures of training data
- Critical augmentation types: paraphrasing, sentence shuffling, translations
- Without augmentation: knowledge is memorized but INACCESSIBLE — 0% extraction
  accuracy even after instruction fine-tuning

**Finding 3: Knowledge is encoded in two ways internally.**
- **Entity name embeddings**: linearly encoded in the token embeddings of entity names
- **Distributed encoding**: distributed across other token embeddings in the training text
- Linear probing can reveal which storage mechanism is active

**Finding 4: The inaccessibility problem is a training problem, not a capacity problem.**
- A model that has "seen" a fact during training may completely fail to recall it
- The failure is in the extraction pathway, not storage
- More data of the SAME distribution does not fix this — diversity is the key

### First Principles Extracted

```
PRINCIPLE 10: Knowledge accessibility ≠ knowledge storage.
              A model can store information it cannot retrieve.

PRINCIPLE 11: Knowledge accessibility is a function of TRAINING DATA
              DIVERSITY, not model capacity. A bigger model with
              homogeneous data stores more but extracts less.

PRINCIPLE 12: Data augmentation (paraphrasing, shuffling, translation)
              is not just regularization — it creates the extraction
              pathways that make stored knowledge usable.
```

### Relevance to NanoSeek

| NanoSeek Component | Connection | Impact |
|---|---|---|
| 22B token dataset | Data diversity requirement | Homogeneous data = locked knowledge. Must measure diversity metrics |
| Series A scale sweep | Knowledge extraction vs model size | Larger models store more but extraction depends on data |
| Expert specialization (I_spec) | MoE routing may affect knowledge pathways | Do different experts extract different types of knowledge? |
| Evaluation harness | What to measure | Add knowledge extraction probes alongside MMLU/ARC |

### Actionable Data Diversity Metrics

Before training NanoSeek-1B, measure these on the 22B token corpus:

```python
# Diversity metrics from Part 3.1 (correlate with extraction success):

1. Paraphrase diversity: For each fact, how many syntactically different
   expressions exist in the corpus? (Target: ≥3 per fact)

2. Context diversity: How many different surrounding contexts does each
   entity appear in? (Target: ≥5 contexts per entity)

3. Template diversity: How many different sentence structures express
   the same relationship? (Target: ≥4 templates per relation type)

4. Cross-document overlap: What fraction of facts appear in multiple
   documents with different wordings? (Target: ≥30%)
```

---

## 6. Part 3.2: Knowledge Manipulation

**Paper:** [ICLR 2025 Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/file/d5494c8747276d3cdb2598e5617de89d-Paper-Conference.pdf)
**Venue:** ICLR 2025

### Core Research Question

Can models manipulate stored knowledge — retrieving parts of attributes,
combining multiple attributes, or performing operations on stored facts?

### Key Findings

**Finding 1: Knowledge is stored as atomic, indivisible units.**
- Models correctly answer "What is the birth DATE of Anya?" but fail at
  "What is the birth YEAR of Anya?"
- The attribute (birth date) is stored as one unit — sub-attribute extraction
  (year from date) requires a separate computation the model doesn't learn

**Finding 2: Multi-attribute retrieval is harder than single-attribute.**
- Retrieving one attribute works well
- Retrieving two attributes simultaneously degrades accuracy
- The degradation is not from interference but from computational complexity

**Finding 3: Knowledge granularity is determined at STORAGE time.**
- Fine-tuning cannot fix storage-level granularity limitations
- If the model learned "birth date" as atomic, no amount of post-training will
  teach it to extract "birth year" from that stored representation
- The solution is pretraining with finer-grained knowledge decomposition

### First Principles Extracted

```
PRINCIPLE 13: Knowledge granularity is fixed at storage (pretraining) time.
              Fine-tuning cannot decompose atomic knowledge units.

PRINCIPLE 14: Multi-attribute retrieval has computational overhead
              that grows with the number of attributes requested.
```

### Relevance to NanoSeek

| NanoSeek Component | Connection | Impact |
|---|---|---|
| Evaluation design | Don't test fine-grained knowledge manipulation at 1B | Save eval compute — test what's achievable |
| Data preparation | Include decomposed facts in training data | "Born: March 15, 1990. Birth year: 1990. Birth month: March." |
| RL Stage 3 (General) | Knowledge manipulation won't improve via DPO | DPO cannot fix pretraining-level knowledge granularity |

---

## 7. Part 3.3: Knowledge Capacity Scaling Laws

**Paper:** [arXiv:2404.05405](https://arxiv.org/abs/2404.05405)
**Venue:** ICLR 2025

### Core Research Question

How much factual knowledge can a language model store, and how do architecture
choices, training duration, quantization, and sparsity affect this capacity?

### Key Findings — 12 Results

**THE FUNDAMENTAL RESULT: 2 bits of knowledge per parameter.**

This is a hard limit. A 7B model stores ~14B bits of knowledge, which exceeds
English Wikipedia + textbooks combined. This holds even under int8 quantization.

**Result 1: Training duration affects realized capacity.**
- Short training: capacity underutilized
- Optimal training: approaches 2 bits/param asymptotically
- Over-training: diminishing returns on capacity per additional token

**Result 2: Architecture effects on capacity.**
- GPT-2 with RoPE matches or surpasses LLaMA/Mistral for knowledge storage
  over shorter training durations
- Reason: GatedMLP (SwiGLU, used by LLaMA/Mistral AND NanoSeek) is harder
  to train early on — it takes more steps to reach the same capacity utilization
- Long-duration training eventually favors GatedMLP

**Result 3: Quantization preserves capacity.**
- int8 quantization maintains the 2-bit/param capacity
- This means quantized models don't lose knowledge — they lose precision in
  computation but not in storage

**Result 4: MoE sparsity affects knowledge capacity.**
- Sparsity constraints (only top-k experts active) affect how efficiently
  parameters store knowledge
- The effect is measurable but moderate — consistent with the log(E)^γ
  correction term in NanoSeek's scaling law

**Result 5: Data signal-to-noise ratio matters.**
- Higher SNR data (knowledge-dense text) yields more stored knowledge per token
- Prepending domain names significantly increases capacity — models learn to
  prioritize knowledge-rich domains autonomously

**Results 6-12: Additional architecture and training dynamics findings.**
- Attention head count, FFN ratio, normalization choices all affect capacity
  but within the 2-bit/param envelope
- The 2-bit bound is fundamental; architecture affects how quickly and
  efficiently you approach it

### First Principles Extracted

```
PRINCIPLE 15: Knowledge capacity = 2 bits per parameter. This is a
              fundamental limit, not an empirical observation.

PRINCIPLE 16: Realized capacity depends on training recipe — duration,
              data quality, architecture choices determine how close
              to the 2-bit bound you get.

PRINCIPLE 17: GatedMLP (SwiGLU) is harder to train early but better
              long-term. This directly affects NanoSeek's stability
              ablations at short duration (3000 steps).

PRINCIPLE 18: MoE sparsity has a moderate, measurable effect on
              knowledge capacity — consistent with log(E)^γ correction.

PRINCIPLE 19: Data with explicit domain markers increases storage
              efficiency. Models learn to prioritize knowledge-dense domains.
```

### Relevance to NanoSeek — CRITICAL

This is the single most important paper for NanoSeek's Pillar 1.

| NanoSeek Component | Connection | Impact Level |
|---|---|---|
| **Scaling law formula** | 2-bit/param bound gives theoretical interpretation of L_irr | **CRITICAL** |
| **Series B (expert sweep)** | MoE sparsity effect on capacity = what Series B measures | **CRITICAL** |
| **fit_scaling_law.py** | Add knowledge capacity analysis alongside loss fitting | **HIGH** |
| **Stability ablations** | SwiGLU harder to train at short duration = confound | **HIGH** |
| **Data preparation** | Domain prefixing increases capacity utilization | **MEDIUM** |

### Knowledge Capacity Analysis for NanoSeek

```
NanoSeek-1B knowledge capacity calculation:

  N_active = 1.08B parameters
  Theoretical capacity = 2 × 1.08B = 2.16B bits of knowledge

  Comparison:
    English Wikipedia ≈ 4B tokens × ~2 bits/token useful info ≈ 8B bits
    But factual density is much lower — estimated ~0.5B distinct facts

  NanoSeek capacity (2.16B bits) > Wikipedia factual content (~0.5B facts × ~20 bits/fact)

  BUT: realized capacity depends on training duration and data diversity (Part 3.1)
  At 22B tokens (20× N_active), training duration is sufficient for near-optimal
  capacity utilization IF data diversity is adequate.

  Key diagnostic for fit_scaling_law.py:
    If L_irr (irreducible loss) is higher than expected:
    → Data quality/diversity bottleneck, NOT model capacity bottleneck
    → Part 3.1 diversity augmentation would help more than model scaling
```

---

## 8. Part 4.1: Architecture Design — Canon Layers

**Paper:** [arXiv:2512.17351](https://arxiv.org/abs/2512.17351)
**Venue:** NeurIPS 2025

### Core Research Question

Can we design synthetic pretraining tasks that isolate and evaluate core model
capabilities, and use them to discover new architectural components?

### Key Findings

**Finding 1: Canon layers — a new architectural primitive.**
- Canon layers are lightweight components that promote horizontal information
  flow across neighboring tokens
- They compute weighted sums of nearby token representations
- Named after the musical "canon" — each voice echoes nearby voices
- Integrate seamlessly into transformers, linear attention, or state-space models

**Finding 2: Canon layers enhance reasoning by 2× depth.**
- 2× improvement in reasoning depth
- Improved reasoning breadth (parallel dependency tracking)
- Enhanced knowledge manipulation
- These are additive improvements on top of existing architecture

**Finding 3: Canon layers rescue weak architectures.**
- NoPE (no positional encoding) + Canon layers matches RoPE performance
- GLA (Gated Linear Attention) + Canon layers: 1-hop → 4-hop reasoning depth
- Linear attention + Canon layers rivals Mamba2 and even surpasses it on some tasks
- This suggests Canon layers provide a general-purpose capability boost

**Finding 4: Synthetic pretraining playground enables precise capability measurement.**
- Five diagnostic datasets: Depo, Brevo, Capo, Mano, Lano
- Each isolates a specific capability:
  - **Depo**: Dependency tracking depth
  - **Brevo**: Breadth of simultaneous dependency tracking
  - **Capo**: Compositional capability
  - **Mano**: Knowledge manipulation ability
  - **Lano**: Language structure processing
- These are more precise than MMLU/ARC for diagnosing capabilities

### First Principles Extracted

```
PRINCIPLE 20: Local token mixing (beyond self-attention) is a separate,
              additive capability axis. Architecture design has untapped
              gains orthogonal to scaling.

PRINCIPLE 21: Synthetic diagnostic benchmarks can isolate capabilities
              that aggregate benchmarks (MMLU, ARC) conflate.

PRINCIPLE 22: Weak architectures can be rescued by targeted architectural
              additions — you don't always need to scale up.
```

### Relevance to NanoSeek

| NanoSeek Component | Connection | Impact |
|---|---|---|
| Current architecture | Canon layers are orthogonal to MLA+MoE | Future enhancement, not current priority |
| Evaluation suite | Depo/Brevo/Capo/Mano/Lano as diagnostics | **HIGH** — add these to evaluation harness |
| Architecture decisions | Canon layers could lift NanoSeek's reasoning depth | Post-v1.0 exploration |
| Scaling law sweep | Adding Canon layers now would confound sweep | Do NOT add during current experiments |

---

## 9. Part 4.2: Canon at Scale

**Paper:** Available at [GitHub: facebookresearch/PhysicsLM4](https://github.com/facebookresearch/PhysicsLM4)
**Status:** Published 2025

### Core Finding

Part 4.1's synthetic pretraining insights transfer to real-world pretraining at
academic scale. Canon layers validated on actual language modeling tasks, not just
synthetic benchmarks.

### Relevance to NanoSeek

This validates that synthetic → real transfer works, which strengthens the case
for using Allen-Zhu's diagnostic benchmarks (Depo, Brevo, etc.) on NanoSeek.
If these synthetic tasks predict real-world capabilities, they are valid
diagnostic tools for the scaling law sweep.

---

## 10. Impact Matrix: Physics of LM × NanoSeek Pillars

### Direct Impact Assessment

| Physics of LM Part | NanoSeek Pillar | Impact | Why |
|---|---|---|---|
| **Part 3.3** (Knowledge Scaling) | **Pillar 1** (Scaling Laws) | **CRITICAL** | 2-bit/param law directly validates/challenges L(N,D,E). MoE sparsity effect = what Series B measures |
| **Part 2.1** (Hidden Reasoning) | **Pillar 4** (RL Post-Training) | **HIGH** | Understanding internal planning essential for Stage 1 reward design. V-probing validates RL improvements |
| **Part 2.2** (Learning from Mistakes) | **Pillar 4** + Training Data | **HIGH** | Error-correction pretraining > RL retry. Directly actionable for data pipeline |
| **Part 3.1** (Knowledge Extraction) | Training Data Design | **HIGH** | Data diversity requirement means 22B tokens need careful curation |
| **Part 1** (Hierarchical Structures) | Architecture Understanding | **MEDIUM** | Validates depth choices in Series A. Depth thresholds explain capability jumps |
| **Part 4.1** (Canon Layers) | Evaluation (Exploratory) | **MEDIUM** | Diagnostic benchmarks (Depo/Brevo/Capo/Mano/Lano) as precision evaluation tools |
| **Part 3.2** (Knowledge Manipulation) | Evaluation Design | **LOW-MEDIUM** | Informs what 1B model CAN'T do — don't waste eval compute |

### Temporal Relevance (When Each Part Matters Most)

```
Phase 1 (Model Implementation, Week 1-2):
  → Part 1 (depth choices), Part 3.3 (architecture effects on capacity)

Phase 2 (Data Preparation, Week 2-3):
  → Part 3.1 (diversity requirement), Part 2.2 (error-correction augmentation)

Phase 3 (Scaling Law Sweep, Week 4-8):
  → Part 3.3 (knowledge capacity interpretation of scaling law)

Phase 4 (Stability, Week 8-9):
  → Part 3.3 (SwiGLU instability at short training)

Phase 5 (1B Training, Week 10-12):
  → Part 3.1 (data diversity), Part 3.2 (evaluation expectations)

Phase 6 (RL Post-Training, Week 13-14):
  → Part 2.1 (reasoning mechanism), Part 2.2 (error correction)

Phase 7 (Future Iterations):
  → Part 4.1, 4.2 (Canon layers as architecture enhancement)
```

---

## 11. Detailed Impact by Pillar

### Pillar 1: Scaling Laws

**Part 3.3 provides the theoretical foundation for your scaling law formula.**

NanoSeek's formula:
```
L(N_active, D, E) = L_irr + A/N_active^α + B_e·log(E)^γ + B_d/D^δ
```

Physics of LM interpretation of each term:

| Term | Traditional Interpretation | Allen-Zhu Interpretation |
|---|---|---|
| `L_irr` | Irreducible loss (data entropy) | **Includes knowledge inaccessibility** — data with low diversity has higher effective L_irr because stored knowledge can't be extracted (Part 3.1) |
| `A/N_active^α` | Parameter scaling | **Knowledge capacity bound**: at 2 bits/param (Part 3.3), this term captures how efficiently the model approaches the capacity limit |
| `B_e·log(E)^γ` | Expert routing correction | **Sparsity effect on knowledge capacity**: Part 3.3 shows MoE sparsity moderately affects capacity. Your Series B directly measures this |
| `B_d/D^δ` | Data scaling | **Knowledge extraction efficiency**: Part 3.1 shows this depends on data diversity, not just quantity. Homogeneous 22B tokens may give worse δ than diverse 15B tokens |

**Specific addition to `fit_scaling_law.py`:**

```python
# After fitting L(N_active, D, E), add knowledge capacity analysis:

def analyze_knowledge_capacity(fitted_params, config):
    """
    Compare fitted scaling law to Allen-Zhu's 2-bit/param bound.

    If L_irr >> theoretical_minimum:
      → Data quality bottleneck (Part 3.1: low diversity)
      → NOT model capacity bottleneck

    If α differs from dense baselines:
      → MLA may introduce scale-dependent bottleneck (testable)
    """
    theoretical_bits = 2 * config.n_active_params  # 2-bit/param bound

    # Estimate knowledge in training data
    # (requires separate analysis of data corpus)

    # Report: realized_capacity / theoretical_capacity ratio
    # This is the "knowledge utilization efficiency"
```

### Pillar 2: Training Stability

**Part 3.3's architecture finding directly affects stability ablations.**

Allen-Zhu shows GatedMLP (SwiGLU) — which NanoSeek uses — is **harder to train
over short durations** than standard MLP. Your stability ablations run for 3000 steps
at nano-150M, placing them squarely in the "short training" regime.

**Implication for Run C vs D:**
- QK-norm may appear more important than it actually is
- SwiGLU's early-training instability compounds with MoE routing instability
- This is a confound: you might attribute instability to missing QK-norm when
  the real cause is SwiGLU's training dynamics
- **Mitigation**: Document this confound in STABILITY_PLAYBOOK.md. Consider
  running one ablation with standard MLP to isolate the SwiGLU effect

### Pillar 3: Production Observability

**Part 3.1's knowledge extraction metrics could enhance dashboards.**

Add to the evaluation harness:
- Knowledge extraction accuracy on a controlled fact set
- Track across training: does extraction accuracy track loss, or diverge?
- If they diverge: data diversity issue (Part 3.1 predicts this)

### Pillar 4: RL Post-Training

**Parts 2.1 and 2.2 are essential pre-reading for RL design.**

**Part 2.1 impact on Stage 1 (Reasoning RL):**

Your current reward design:
```
Math: parse final numeric answer, compare to ground truth
Code: run generated code, check test cases pass
Process bonus: reward self-correction in CoT (+0.2)
```

Allen-Zhu insight: this only rewards the OUTPUT, not the INTERNAL PLANNING PROCESS.
The model might achieve correct answers through better surface-level pattern matching
rather than improved internal reasoning.

**Enhancement**: Add V-probing evaluation at stage boundaries:
```
Pre-RL:     measure reasoning depth via V-probing on math problems
Post-Stage1: re-measure. If depth increased → genuine reasoning improvement
             If depth unchanged but accuracy up → surface pattern improvement
Post-Stage2: same measurement for agent reasoning
Post-Stage3: verify no regression
```

**Part 2.2 impact on training data:**

Current plan: 10% FIM + 90% standard autoregressive.

Suggested enhancement: 10% FIM + 3% error-correction + 87% standard.

The 3% error-correction data should follow Part 2.2's "fake mistake" recipe:
- Take correct math/code solutions from training data
- Insert a future step as a fake error
- Add correction marker ("Wait, that's not right. Let me reconsider.")
- Continue with correct solution
- This is cheap to generate and directly trains the recovery circuit that
  Stage 1 RL tries to strengthen

---

## 12. Recommended Video Watching Order

### Matched to NanoSeek's 16-Week Build Order

#### Watch NOW (Week 0, Before Coding)

**1. Part 3.3: Knowledge Capacity Scaling Laws** — MUST WATCH FIRST

Why: You're about to implement `fit_scaling_law.py` and design sweep configs.
The 2-bit/param bound and MoE sparsity effects will change how you interpret
every sweep result. This is the theoretical foundation for Pillar 1.

Key moments to watch for:
- The 12 results on architecture/quantization/sparsity
- MoE sparsity result — validates/challenges your log(E) correction
- GatedMLP training difficulty — directly relevant to stability ablations
- Domain-name prefixing effect — cheap data improvement

Estimated viewing: ~40 min (part of 80-min Tutorial I Session 3)

---

**2. Part 1: Hierarchical Language Structures** — HIGH PRIORITY

Why: You're designing Series A with models ranging 4→16 layers. Part 1 tells
you depth is the primary driver of hierarchical reasoning. This informs whether
depth-to-width ratio changes are a feature or confound.

Key moments to watch for:
- Dynamic programming interpretation of attention
- Autoregressive vs encoder-only comparison
- Depth thresholds for structural reasoning levels

Estimated viewing: ~100 min (Tutorial I Session 1)

---

#### Watch Before Data Preparation (Week 2-3)

**3. Part 3.1: Knowledge Storage and Extraction** — HIGH PRIORITY

Why: You're preparing your 22B token dataset. Data diversity determines
whether stored knowledge is extractable. Without this, you train a model
that stores knowledge it can't use.

Key moments to watch for:
- Diversity metrics that correlate with extraction accuracy
- The "0% accuracy even after fine-tuning" result on low-diversity data
- Linear probing to reveal knowledge storage mechanisms

Estimated viewing: ~40 min (part of Tutorial I Session 3)

---

**4. Part 3.2: Knowledge Manipulation** — MEDIUM PRIORITY

Why: Before designing your evaluation harness, understand what knowledge
operations your 1B model can and cannot perform. Don't test impossible tasks.

Key moments to watch for:
- Atomic vs decomposable attribute storage
- Multi-attribute retrieval degradation

Estimated viewing: ~30 min (part of Tutorial I Session 3)

---

#### Watch Before RL Post-Training (Week 12-13)

**5. Part 2.1: Hidden Reasoning Process** — CRITICAL FOR RL

Why: You're implementing `grpo_trainer.py` Stage 1. This paper tells you
what's happening inside the model during math reasoning. Without it, you're
optimizing blind.

Key moments to watch for:
- V-probing methodology — implement this as RL diagnostic
- Minimum model depth for math reasoning — bounds your 16-layer model
- Internal planning before generation — implications for CoT rewards

Estimated viewing: ~60 min (Tutorial I Session 2)

---

**6. Part 2.2: Learning From Mistakes** — HIGH FOR RL

Why: Your Stage 1 has process bonus (+0.2 for self-correction). This paper
validates the idea but shows pretraining-stage correction is stronger.

Key moments to watch for:
- Distribution shift as root cause of errors
- "Fake mistake" augmentation technique
- Retry-upon-regret effectiveness analysis

Estimated viewing: ~50 min (Tutorial I Session 2)

---

#### Watch For Future Iterations (Post-Week 16)

**7. Part 4.1 + 4.2: Canon Layers** — FUTURE PRIORITY

Why: Powerful architectural enhancement, but adding to NanoSeek now confounds
your experiments. Watch to plan v2.0.

Key moments to watch for:
- Synthetic pretraining playground (Depo/Brevo/Capo/Mano/Lano)
  → Use these as diagnostic benchmarks NOW, even without Canon layers
- 2× reasoning depth improvement quantifies future gains
- Linear attention rescue — relevant if you explore MLA alternatives

Estimated viewing: ~150 min (Tutorial II)

---

### Summary Table

| Priority | Part | When to Watch | Duration | Maps to Week |
|---|---|---|---|---|
| 1 (MUST) | **3.3 Knowledge Scaling** | Before any coding | ~40 min | Week 0 |
| 2 (HIGH) | **1 Hierarchical Structures** | Before sweep design | ~100 min | Week 0 |
| 3 (HIGH) | **3.1 Knowledge Extraction** | Before data prep | ~40 min | Week 2 |
| 4 (MED) | **3.2 Knowledge Manipulation** | Before eval design | ~30 min | Week 2 |
| 5 (CRIT) | **2.1 Hidden Reasoning** | Before RL implementation | ~60 min | Week 12 |
| 6 (HIGH) | **2.2 Learning from Mistakes** | Before RL implementation | ~50 min | Week 12 |
| 7 (FUT) | **4.1 + 4.2 Canon Layers** | After v1.0 complete | ~150 min | Post-Week 16 |

**Total essential viewing before 1B training: ~260 min (~4.5 hours)**
**Total including RL prep: ~370 min (~6 hours)**

---

## 13. 5 Concrete Changes to SCALING_LAB_PLAN

Based on Physics of Language Models findings, these are the recommended
modifications to the existing plan, in priority order.

### Change 1: Add Knowledge Capacity Analysis to Pillar 1

**What:** In `fit_scaling_law.py`, after fitting L(N,D,E), add knowledge
capacity diagnostics based on Part 3.3's 2-bit/param bound.

**Why:** This gives your scaling law an explanatory interpretation, not just
a curve fit. No other scaling law paper has this.

**Implementation:**
```python
# In fit_scaling_law.py, after Step 5 (generate plots):

# Step 6b: Knowledge Capacity Analysis (from Allen-Zhu Part 3.3)
#
# For each sweep config:
#   theoretical_capacity = 2 * n_active_params  (bits)
#   training_tokens_per_param = D / n_active_params
#
# Plot 8b: Knowledge utilization vs training tokens/param
#   x-axis: D/N_active (log scale)
#   y-axis: estimated knowledge utilization (from loss curve shape)
#   Reference: Part 3.3 shows capacity approaches 2 bits/param
#   asymptotically with training duration
#
# Diagnostic: If L_irr from fit >> theoretical minimum
#   → Data diversity bottleneck (cite Part 3.1)
#   → Augmentation would help more than scaling
```

**Effort:** ~50 lines of code added to existing script.
**Risk:** None — additive analysis, doesn't change existing outputs.

---

### Change 2: Data Diversity Audit Before Training

**What:** Before the 22B token training run, compute diversity metrics on
the training corpus. Part 3.1 shows diversity determines knowledge accessibility.

**Why:** Without diversity, you may train a model that stores knowledge but
can't extract it. This is a cheap preventive measure.

**Implementation:**
```python
# New file: nanoseek/data_analysis/diversity_audit.py

# Metrics to compute on training corpus sample (1M tokens):
#
# 1. Paraphrase diversity: for sampled facts, count syntactically
#    different expressions. Target: ≥3 per fact.
#
# 2. Context diversity: for sampled entities, count distinct
#    surrounding contexts. Target: ≥5 per entity.
#
# 3. Domain distribution: fraction of tokens per domain.
#    If any domain < 5%: risk of domain-specific knowledge lock-out.
#
# 4. Fact repetition with variation: how often are facts restated
#    differently? Target: ≥30% of facts appear in ≥2 wordings.
#
# Output: DIVERSITY_AUDIT.md with pass/fail and recommendations
```

**Effort:** 1-2 hours to implement and run.
**Risk:** May reveal need for data augmentation, which adds prep time.

---

### Change 3: Add Error-Correction Sequences to Pretraining

**What:** Allocate 2-5% of training data to mistake-correction sequences
for math and code, following Part 2.2's "fake mistake" augmentation.

**Why:** Error correction in pretraining > error correction via RL. This
directly strengthens the capability that Stage 1 RL tries to improve.

**Implementation:**
```
Current data mix: 90% standard + 10% FIM
Proposed data mix: 87% standard + 10% FIM + 3% error-correction

Error-correction generation (add to dataset.py):
  1. Take correct math/code solutions from training data
  2. At random step k, insert step (k+2) as "fake error"
  3. Add correction marker: "Wait, that's incorrect. Let me reconsider."
  4. Continue with correct step k+1 onward

  This is a pure data augmentation — no model changes needed.
  Log: "error_correction_fraction" alongside "fim_fraction"
```

**Effort:** ~100 lines in dataset.py.
**Risk:** Low — if ineffective, only costs 3% of data budget.
**Expected gain:** Part 2.2 shows this can improve reasoning accuracy
by up to 5-10% on grade-school math.

---

### Change 4: Add V-Probing to RL Evaluation

**What:** Implement V-probing (from Part 2.1) as a diagnostic tool to
measure whether GRPO improves internal reasoning or surface patterns.

**Why:** Without this, you can't distinguish "model reasons better" from
"model pattern-matches better." V-probing is the only tool that measures
the internal planning process.

**Implementation:**
```python
# New file: nanoseek/model/eval/v_probing.py

class ReasoningDepthProbe:
    """
    V-probing: measure reasoning depth in hidden states.

    At each stage boundary (pre-RL, post-Stage1, post-Stage2, post-Stage3):
    1. Run model on math problems, extract hidden states at each layer
    2. Train linear probes: h_l^t → dependency_structure
    3. Measure: at which layer does the model "know" the answer?
    4. Report: reasoning_depth = deepest dependency layer with >80% accuracy

    If reasoning_depth increases after Stage 1 → genuine reasoning improvement
    If reasoning_depth unchanged but accuracy up → surface pattern improvement
    """
```

**Effort:** ~200 lines + probe training pipeline.
**Risk:** Requires careful probe design. May not show clear signal at 1B scale.
**Expected value:** Differentiates your RL results from everyone else's.

---

### Change 5: Use Allen-Zhu's Synthetic Benchmarks as Diagnostics

**What:** After training, evaluate NanoSeek on Part 4.1's synthetic tasks
(Depo, Brevo, Capo, Mano, Lano) to isolate specific capabilities.

**Why:** MMLU and ARC are aggregate scores that conflate many capabilities.
These synthetic benchmarks precisely measure reasoning depth, breadth,
compositional ability, and knowledge manipulation.

**Implementation:**
```python
# Add to nanoseek/fms/eval_harness/synthetic_diagnostics.py

# Port from github.com/facebookresearch/PhysicsLM4:
#   Depo: dependency tracking depth → maps to Part 1 findings
#   Brevo: reasoning breadth → measures parallel dependency tracking
#   Capo: compositional capability → measures multi-step composition
#   Mano: knowledge manipulation → maps to Part 3.2 findings
#   Lano: language structure → maps to Part 1 findings
#
# Run at:
#   - End of pretraining (baseline capabilities)
#   - After each RL stage (capability changes)
#   - Compare across sweep configs (capability vs scale)
```

**Effort:** ~4-6 hours (mostly porting from Facebook's repo).
**Risk:** None — purely additive evaluation.
**Expected value:** Precise capability diagnosis vs aggregate benchmarks.

---

## 14. Fundamental Principles Summary

The 22 principles extracted from Physics of Language Models, organized by theme:

### Architecture & Representation (Parts 1, 4.1)
```
P1:  Transformers build structured parse trees in hidden states
P2:  Depth determines maximum reasoning depth
P3:  Autoregressive generation aligns with incremental parsing
P20: Local token mixing is a separate, additive capability axis
P21: Synthetic benchmarks isolate capabilities that aggregate benchmarks conflate
P22: Weak architectures can be rescued by targeted additions
```

### Reasoning & Planning (Parts 2.1, 2.2)
```
P4:  Reasoning = internal planning + sequential expression
P5:  Maximum reasoning depth is bounded by model depth
P6:  Reasoning skills are transferable
P7:  Model errors are caused by distribution shift
P8:  Robustness must be TRAINED IN, not PROMPTED IN
P9:  Error correction in pretraining > error correction via RL
```

### Knowledge (Parts 3.1, 3.2, 3.3)
```
P10: Knowledge accessibility ≠ knowledge storage
P11: Accessibility is a function of data diversity, not capacity
P12: Data augmentation creates extraction pathways
P13: Knowledge granularity is fixed at storage (pretraining) time
P14: Multi-attribute retrieval has growing computational overhead
P15: Knowledge capacity = 2 bits per parameter (fundamental limit)
P16: Realized capacity depends on training recipe
P17: GatedMLP is harder to train early but better long-term
P18: MoE sparsity has moderate effect on knowledge capacity
P19: Domain markers increase storage efficiency
```

### The Meta-Principle

```
Language models are NOT black boxes. They have:
  - Quantifiable knowledge capacity (2 bits/param)
  - Measurable reasoning depth (bounded by layers)
  - Diagnosable knowledge accessibility (function of data diversity)
  - Observable internal planning (detectable via V-probing)
  - Trainable error recovery (via pretraining augmentation)

NanoSeek's scaling law work, combined with these Physics of LM findings,
can produce results with EXPLANATORY POWER, not just curve fitting.
```

---

## 15. References

### Papers (in recommended reading order)

1. **Part 3.3** — Allen-Zhu, Z. & Li, Y. (2024). "Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws." [arXiv:2404.05405](https://arxiv.org/abs/2404.05405). ICLR 2025.

2. **Part 1** — Allen-Zhu, Z. & Li, Y. (2023). "Physics of Language Models: Part 1, Learning Hierarchical Language Structures." [arXiv:2305.13673](https://arxiv.org/abs/2305.13673). ICML 2024.

3. **Part 3.1** — Allen-Zhu, Z. & Li, Y. (2023). "Physics of Language Models: Part 3.1, Knowledge Storage and Extraction." [arXiv:2309.14316](https://arxiv.org/abs/2309.14316). ICML 2024.

4. **Part 3.2** — Allen-Zhu, Z. & Li, Y. (2025). "Physics of Language Models: Part 3.2, Knowledge Manipulation." [ICLR 2025 Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/file/d5494c8747276d3cdb2598e5617de89d-Paper-Conference.pdf).

5. **Part 2.1** — Ye, T., Xu, Z., Li, Y. & Allen-Zhu, Z. (2024). "Physics of Language Models: Part 2.1, Grade-School Math and the Hidden Reasoning Process." [arXiv:2407.20311](https://arxiv.org/abs/2407.20311). ICLR 2025.

6. **Part 2.2** — Ye, T., Xu, Z., Li, Y. & Allen-Zhu, Z. (2024). "Physics of Language Models: Part 2.2, How to Learn From Mistakes on Grade-School Math Problems." [arXiv:2408.16293](https://arxiv.org/abs/2408.16293). ICLR 2025.

7. **Part 4.1** — Allen-Zhu, Z. (2025). "Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers." [arXiv:2512.17351](https://arxiv.org/abs/2512.17351). NeurIPS 2025.

8. **Part 4.2** — Allen-Zhu, Z. (2025). "Physics of Language Models: Part 4.2, Canon Layers at Scale where Synthetic Pretraining Resonates in Reality." [GitHub](https://github.com/facebookresearch/PhysicsLM4).

### Project Website

- [Physics of Language Models — Main Site](https://physics.allen-zhu.com/)
- [Zeyuan Allen-Zhu Publications](http://zeyuan.allen-zhu.com/publications.php)

### ICML 2024 Tutorial

- Tutorial I: Parts 1-3, synthesized as 2-hour ICML 2024 tutorial
- Tutorial II: Part 4 (architecture design), separate session

### NanoSeek Cross-References

- `SCALING_LAB_PLAN.md` — Pillar 1-4 experimental design
- `REIMPLEMENTATION_PLAN.md` — Model implementation spec
- `PAPER_ANALYSIS_V3_V32.md` — DeepSeek V3/V3.2 ground truth
- `docs/01_MLA_DEEP_DIVE.md` — MLA theory (Part 1 validates depth importance)
- `docs/02_MOE_DEEP_DIVE.md` — MoE theory (Part 3.3 validates sparsity effects)

---

*Document generated: March 2026*
*Authority level: Reference document — does not override PAPER_ANALYSIS_V3_V32.md or REIMPLEMENTATION_PLAN.md*
*Purpose: Theoretical context from Physics of LM research to inform NanoSeek experimental design*
