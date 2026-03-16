# NanoSeek: Frontier Lab Alignment Deep Dive

## From 48% → 85% — Building the Complete Research Pipeline

### Who This Document Is For

You want to be hired as a senior AI research engineer at a top-tier lab (DeepSeek, Meta FAIR,
Google DeepMind, Anthropic, OpenAI) in 2026. You don't have a PhD, a team, or years of
experience. You have ONE asset: a project that demonstrates you can think and build like
the people who train frontier models.

This document identifies exactly what NanoSeek is missing, why it matters, and how to build
each piece — with the first-principles reasoning that distinguishes a research engineer
from someone who follows tutorials.

### Scope: Tiered Priority System

Components are classified by impact on frontier alignment:

```
MUST HAVE (Phases A-C): ~$100-150, 12-14 weeks → ~80% frontier coverage
  Core MoE+MLA+MTP architecture, 5 anchor ablations, 1B training,
  data pipeline, eval framework, Canon ablation, scaling law fit.
  This alone is more than 95% of applicants show.

STRONG VALUE-ADD (Phase D core, Phase E basic): +$50-80 → ~85% frontier coverage
  GRPO post-training with MoE stabilization, basic interpretability
  (expert specialization viz, I_spec analysis), 1 Triton kernel,
  iterative DPO (1 round), test-time compute scaling curves.

TARGET-SPECIFIC (cherry-pick 2-3 based on where you apply): +$20-2000
  Anthropic interpretability → SAE + alignment probes (Phase E full)
  Anthropic safety → CAI + PRM (Phase D2-D3)
  OpenAI/DeepMind training → 3B scale + Triton kernels (Phase F1-F2)
  Meta FAIR → IsoFLOP + width/depth Pareto (Phase F1, F4)
  7B training ($2000+) → CUT unless specifically needed
```

Total budget: ~$100-300 (core) + optional extras. Timeline: 12-16 weeks.

---

## The Gap Analysis: What Frontier Labs Actually Do

### Current NanoSeek Coverage

```
STRONG (85-90%):
  ✅ Architecture: MoE + MLA + MTP + DSA (DeepSeek V3 faithful)
  ✅ HP Transfer: muP + CompleteP + μP-MoE (state-of-the-art methodology)
  ✅ Stability Engineering: Ablation matrix with controlled variables
  ✅ Optimizer: Muon + AdamW with per-group LR (frontier practice)
  ✅ Training Loop: Batch warmup, grad clip, aux-loss-free, EMA
  ✅ RL Post-Training Plan: 3-stage GRPO + DPO (matches DeepSeek R1)
  ✅ Mechanistic Interpretability Plan: SAE on MoE experts, fTRI methodology,
     feature birth/death tracking, alignment faking detection (MACHANIC_INTERPRET.md)
  ✅ Canon Layers: Allen-Zhu local horizontal information flow, 4-point ablation
     design with MoE routing interaction analysis (05_CANON_LAYERS_DEEP_DIVE.md)

MODERATE (40-60%):
  ⚠️ Training Observability (40% — H_load planned but not wired, no MFU validation)
  ⚠️ Post-Training (50% — detailed GRPO plan + MoE stabilization, but no code yet)
  ⚠️ Reproducibility Infrastructure (30% — no seed control, no deterministic mode)

WEAK OR MISSING (0-20%):
  ❌ Data Curation Pipeline (0% — uses raw ClimbMix-400B with zero filtering)
  ❌ Evaluation Framework (20% — BPB only, no downstream benchmarks)
  ❌ Scaling Law Fitting (0% — has 3 data points but no fitting code)
  ❌ Data Mixture Optimization (0% — single data source, no domain balancing)
  ❌ Inference Optimization (0% — no quantization, no KV cache optimization benchmarks)
  ❌ Test-Time Compute Scaling (mentioned in RL plan but no implementation)
  ❌ Safety & Alignment (0% — no constitutional AI, no alignment evals)
  ❌ Scale Validation (0% — no 3B-7B runs, no multi-node distributed)
  ❌ Mechanistic Interpretability Code (0% — plan exists but no SAE/probe code)
  ❌ Process Reward Model (0% — needed for reasoning RL beyond rule-based)
```

### Weighted Impact on Frontier Alignment

```
                                Weight    Current    Must-Have    With Extras
Architecture + Optimizer         12%       90%         92%          92%
HP Transfer Science              10%       85%         90%          90%
Data Curation                    15%       10%         85%          85%
Evaluation Framework             10%       20%         90%          90%
Scaling Laws                      5%        0%         85%          85%
Post-Training (GRPO/DPO)        12%       50%         85%          90%
Training Infrastructure           8%       60%         80%          90%
Mechanistic Interpretability      8%       25%*        50%          85%
Research Novelty (Canon)          8%       30%**       75%          85%
Scale Validation                  5%        0%          0%          80%
Safety & Alignment                4%        0%         30%          75%
Reproducibility + Rigor           3%       30%         90%          90%

* MACHANIC_INTERPRET.md plan exists (25%) but no code
** Canon deep dive + MoE interaction analysis planned (30%) but no runs

Weighted Current:    ~42%
Must-Have Only:      ~78-80% (Phases A-C + core D — well-executed 1B with
                     clean ablations already beats 95% of applicants)
With Strong Extras:  ~85% (+ basic interp, 1 Triton kernel, iterative DPO)
With Target-Specific: ~88-93% (cherry-pick per employer: SAE/PRM/3B/IsoFLOP)

KEY INSIGHT: Going from 80% → 85% is high ROI (~$50-80).
Going from 85% → 93% is low ROI (~$800-2000+) and risks doing
everything at surface level instead of doing the core deeply.
A clean, deep 1B beats a sloppy 7B every time.
```

---

# PART 1: DATA CURATION PIPELINE

## Why This Is the #1 Gap

Every frontier lab in 2025-2026 attributes a large share of their gains to data, not
architecture:

- **Llama 3** (Meta): "Data quality has outsized influence on model quality." 15T tokens
  with semantic deduplication + classifier filtering. Final mix: ~50% general knowledge,
  25% math/reasoning, 17% code, 8% multilingual.
- **Qwen 2.5** (Alibaba): Used Qwen2-Instruct models as data quality filters. Instance-
  level mixture optimization via ablations on proxy models. 18T tokens.
- **Qwen 3**: 36T tokens across 119 languages. Three-stage pre-training with domain-
  specific phases.
- **DeepSeek V3**: 14.8T tokens. Automatic filters for toxicity, spam, PII. PSM framework
  for code. "Refined to minimize redundancy while maintaining corpus diversity."

NanoSeek currently uses ClimbMix-400B with ZERO filtering. Every token gets equal weight.
This is equivalent to training on raw Common Crawl circa 2020 — a practice no serious lab
has done since GPT-3.

## The Physics of Why Data Quality Matters

### Information Theory Argument

A language model learns a compression function f: tokens → probability distribution. The
quality of this function depends on the SIGNAL-TO-NOISE ratio of the training data:

```
L_final = L_irreducible + L_model_capacity + L_data_noise

Where:
  L_irreducible: entropy of natural language (can't reduce)
  L_model_capacity: limited by N_active (can't change at fixed budget)
  L_data_noise: reducible by data curation
```

At our scale (1.08B active, 22B tokens), L_model_capacity dominates. But L_data_noise is
the only term we can reduce WITHOUT spending more GPU-hours. Every 0.01 BPB we save on
noise is 0.01 BPB more capacity for actual language understanding.

### Scaling Law Argument

Goyal et al. (CVPR 2024, "Scaling Laws for Data Filtering") proved:

> The optimal data filtering strategy depends on total training compute. You cannot pick
> one quality filter and use it for all compute budgets.

At our compute budget (~1e19 FLOPs), aggressive filtering is optimal because:
- We train for 22B tokens, but ClimbMix has 400B available
- We use <6% of the data → we can be VERY selective
- The "data quality scaling law" says: at low compute, filtering aggressively gives
  disproportionate gains

### Empirical Evidence

FineWeb-Edu filtered FineWeb from 15T → 1.3T tokens (removed 92%) using a quality
classifier. Models trained on FineWeb-Edu at the SAME token count consistently outperform
models trained on unfiltered FineWeb. The gain is equivalent to ~2× more training compute
on unfiltered data.

DCLM (DataComp-LM) showed: a fastText classifier trained on ~50K high-quality examples,
used to filter 240T → 3.8T tokens, produces models that match or beat Llama 2 at half the
parameters.

## What to Build: NanoSeek Data Pipeline

### Architecture

```
                    ClimbMix-400B (raw)
                          │
                    ┌─────┴─────┐
                    │  Stage 1   │  Heuristic Filters
                    │  (cheap)   │  ~5 minutes on CPU
                    └─────┬─────┘
                          │ ~300B tokens
                    ┌─────┴─────┐
                    │  Stage 2   │  Quality Classifier
                    │ (moderate) │  fastText, ~2 hours on CPU
                    └─────┬─────┘
                          │ ~80B tokens
                    ┌─────┴─────┐
                    │  Stage 3   │  Deduplication
                    │ (moderate) │  MinHash LSH, ~4 hours
                    └─────┬─────┘
                          │ ~60B tokens
                    ┌─────┴─────┐
                    │  Stage 4   │  Domain Mixture
                    │  (fast)    │  Sampling by domain weights
                    └─────┬─────┘
                          │ 22B tokens (training set)
                          │ + 500M tokens (validation set)
```

### Stage 1: Heuristic Filters (Cheap, CPU-only)

**What**: Rule-based filters that remove obviously bad documents. These are the filters
every frontier lab applies as a first pass.

```python
# nanoseek/data/heuristic_filters.py

def filter_document(doc: str) -> bool:
    """Returns True if document passes all heuristic filters."""

    # 1. Length filters
    if len(doc) < 100:           # Too short to contain useful information
        return False
    if len(doc) > 1_000_000:     # Likely machine-generated or data dump
        return False

    # 2. Character-level quality signals
    words = doc.split()
    if len(words) < 20:          # Too few words
        return False

    # Average word length (natural English: 4-6 chars)
    avg_word_len = sum(len(w) for w in words) / len(words)
    if avg_word_len > 15:        # Likely base64, URLs, or code dump
        return False

    # 3. Repetition filters (detect boilerplate/spam)
    lines = doc.strip().split('\n')
    unique_lines = set(lines)
    if len(unique_lines) / max(len(lines), 1) < 0.3:  # >70% duplicate lines
        return False

    # 4. Special character ratio
    alpha_count = sum(c.isalpha() for c in doc)
    if alpha_count / max(len(doc), 1) < 0.4:  # <40% alphabetic
        return False

    # 5. Stop word presence (English text should have common words)
    # Cheap proxy for "is this actually English prose?"
    stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but'}
    lower_words = set(w.lower() for w in words[:100])
    if len(lower_words & stop_words) < 2:
        return False

    return True
```

**Why these specific thresholds**: Each threshold comes from empirical analysis in the
FineWeb and RedPajama papers. They're designed to be CONSERVATIVE — we'd rather keep a
borderline document than lose a good one, because Stage 2 will catch quality issues.

**Expected reduction**: ~25% of documents removed (ClimbMix is already partially cleaned,
so less aggressive filtering needed than raw Common Crawl).

### Stage 2: Quality Classifier (The Key Differentiator)

**What**: A small classifier that scores each document on "educational/informational value."
This is the technique that produces the largest quality gain per dollar spent.

**The DCLM approach (recommended for our budget)**:

```
1. Collect ~50K high-quality examples (from Wikipedia + textbooks + curated sources)
2. Collect ~50K random web examples (negative class)
3. Train a fastText classifier (binary: high-quality vs random)
4. Score every document in ClimbMix
5. Keep top 20-30% (threshold tuned on proxy model loss)
```

**Why fastText and not a neural classifier**:
- fastText processes ~1M documents/second on CPU. A BERT classifier does ~1K/second.
- At 400B tokens, we have ~200M documents. FastText: ~3 minutes. BERT: ~2.5 days.
- DCLM showed fastText quality classification performs within 2% of BERT-based
  classifiers for this task. The documents that are obviously bad (spam, nonsense) are
  trivially classified. The marginal cases where BERT is better don't affect the final
  model significantly.

**Why this works from first principles**:
- The classifier learns a "quality manifold" in document feature space
- Documents close to the Wikipedia/textbook distribution are scored high
- Documents close to the spam/nonsense distribution are scored low
- The boundary between "high quality" and "low quality" is surprisingly sharp —
  most documents are either clearly educational or clearly not

```python
# nanoseek/data/quality_classifier.py

import fasttext
from pathlib import Path

def train_quality_classifier(
    positive_dir: str,   # Path to high-quality documents
    negative_dir: str,   # Path to random web documents
    output_path: str = "models/quality_classifier.bin"
) -> fasttext.FastText:
    """
    Train a fastText binary classifier for document quality.

    Positive examples: Wikipedia articles, textbook excerpts, curated educational content
    Negative examples: Random sample from ClimbMix (representative of web quality)

    The key insight (DCLM paper): you don't need millions of labeled examples.
    50K positive + 50K negative is sufficient because the quality boundary is
    high-dimensional but smooth — there's a clear manifold separating "educational
    content" from "random web text" in TF-IDF space.
    """
    # Prepare training data in fastText format
    train_file = _prepare_fasttext_format(positive_dir, negative_dir)

    model = fasttext.train_supervised(
        input=train_file,
        lr=0.5,           # Default for text classification
        epoch=5,           # Sufficient for binary classification with 100K examples
        wordNgrams=2,      # Bigrams capture phrase-level quality signals
        dim=100,           # Embedding dimension (100 is standard for fastText)
        loss='softmax'     # Binary classification
    )

    model.save_model(output_path)
    return model

def score_document(model: fasttext.FastText, doc: str) -> float:
    """Return quality score in [0, 1] for a single document."""
    # fastText predicts (__label__positive, __label__negative)
    labels, probs = model.predict(doc.replace('\n', ' ')[:10000], k=2)
    # Return probability of positive class
    for label, prob in zip(labels, probs):
        if label == '__label__positive':
            return prob
    return 0.0
```

**Threshold selection**: Don't pick an arbitrary threshold. Use a proxy model:

```
1. Train a tiny model (10M params, 500M tokens) on ClimbMix filtered at threshold 0.3
2. Train the same model on ClimbMix filtered at threshold 0.5
3. Train the same model on ClimbMix filtered at threshold 0.7
4. Compare val_bpb. Pick the threshold that minimizes val_bpb.
5. Cost: ~$3 total (3 tiny training runs)
```

This is exactly what FineWeb-Edu and DCLM did at larger scale.

### Stage 3: Deduplication

**What**: Remove near-duplicate documents that would waste training tokens on repeated
information.

**Why it matters**: Web crawls contain massive duplication. RedPajama found ~30% of Common
Crawl is near-duplicate content. Training on duplicates has two costs:
1. Wasted tokens (obvious)
2. Memorization risk — the model memorizes duplicated content instead of learning
   generalizable patterns, leading to higher test loss than training loss

**Method: MinHash LSH (Locality-Sensitive Hashing)**

```
1. For each document, compute MinHash signature (128 hash functions)
2. Use LSH to find candidate duplicate pairs (documents with Jaccard > 0.8)
3. Within each duplicate cluster, keep the longest document
4. Discard the rest
```

**Why MinHash and not exact dedup**: Exact deduplication (string matching) misses
paraphrased content, reformatted content, and content with minor edits. MinHash catches
documents that share >80% of their n-gram content, which is the threshold where
"near-duplicate" genuinely means "same information."

```python
# nanoseek/data/dedup.py

from datasketch import MinHash, MinHashLSH

def build_dedup_index(
    documents: list[str],
    threshold: float = 0.8,
    num_perm: int = 128
) -> MinHashLSH:
    """
    Build a MinHash LSH index for near-duplicate detection.

    Parameters:
      threshold: Jaccard similarity threshold (0.8 = industry standard)
      num_perm: Number of hash permutations (128 = good accuracy/speed tradeoff)

    Why 0.8 threshold: Below 0.8, documents share significant content but may
    have meaningfully different information. Above 0.8, they're essentially
    the same document with minor formatting differences.

    Why 128 permutations: Error rate ≈ 1/√num_perm ≈ 8.8%. At 128, the
    false positive rate is ~1% and false negative rate is ~2%. Increasing to
    256 halves the error but doubles memory and compute. Not worth it for our
    scale (~200M documents).
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    for i, doc in enumerate(documents):
        mh = MinHash(num_perm=num_perm)
        # Use 5-grams for MinHash (captures phrase-level similarity)
        words = doc.lower().split()
        for j in range(len(words) - 4):
            ngram = ' '.join(words[j:j+5])
            mh.update(ngram.encode('utf8'))
        lsh.insert(str(i), mh)

    return lsh
```

### Stage 4: Domain Mixture Optimization

**What**: Control the proportion of different data types (code, math, web text, books,
scientific papers) in the training set.

**Why it matters**: The optimal mixture is NOT uniform. Different domains have different
information density and different utility for downstream tasks:

```
Domain mixtures used by frontier labs (2025-2026):

Llama 3 (Meta):    50% general | 25% math/reasoning | 17% code | 8% multilingual
Qwen 2.5:          ~40% web | ~25% code | ~20% STEM | ~10% books | ~5% multilingual
DeepSeek V3:       Undisclosed ratios, but explicit FIM code (10% PSM format)
```

**For NanoSeek (22B tokens, 1B active)**, a reasonable starting mixture:

```
Domain          Tokens   Share    Why
─────────────   ──────   ─────    ───────────────────────────────────
Web (general)    8.8B    40%     Foundation knowledge, language patterns
Code             4.4B    20%     Reasoning structure, logical patterns
Math/STEM        4.4B    20%     Quantitative reasoning, problem-solving
Books/Wiki       3.3B    15%     Long-range coherence, factual knowledge
Multilingual     1.1B     5%     Basic cross-lingual capability
─────────────   ──────   ─────
Total           22.0B   100%
```

**How to validate the mixture**: Train 3 tiny proxy models (10M params) with different
mixtures for 500M tokens each. Evaluate on downstream tasks (not just BPB). The mixture
that gives the best downstream score is the winner.

This is exactly what Qwen calls "instance-level data mixture optimization via ablation on
small proxy models." The cost is ~$5 for 3 proxy runs.

```python
# nanoseek/data/mixture.py

from dataclasses import dataclass

@dataclass
class DomainMixture:
    """
    Domain sampling weights for training data mixture.

    The mixture is applied via weighted sampling during dataloader construction:
    each batch is assembled by drawing documents from each domain proportional
    to its weight. This is simpler than maintaining separate data streams and
    gives the same expected mixture over the full training run.
    """
    web: float = 0.40
    code: float = 0.20
    math_stem: float = 0.20
    books_wiki: float = 0.15
    multilingual: float = 0.05

    def __post_init__(self):
        total = self.web + self.code + self.math_stem + self.books_wiki + self.multilingual
        assert abs(total - 1.0) < 1e-6, f"Mixture weights must sum to 1.0, got {total}"

    def get_weights(self) -> dict[str, float]:
        return {
            'web': self.web,
            'code': self.code,
            'math_stem': self.math_stem,
            'books_wiki': self.books_wiki,
            'multilingual': self.multilingual
        }
```

### Multi-Stage Pre-Training (Qwen Approach)

Qwen 2.5 and Qwen 3 both use multi-stage pre-training, and it's becoming standard:

```
Stage 1 (70% of tokens): General pre-training
  - Broad web + code + books mix
  - Standard mixture weights
  - Goal: general language understanding

Stage 2 (20% of tokens): Knowledge-intensive
  - Upweight STEM, code, math
  - More curated/filtered subset
  - Goal: reasoning and domain knowledge

Stage 3 (10% of tokens): Long-context + specialization
  - This is your Phase 2 (8K context + DSA)
  - But ALSO upweight high-quality long documents
  - Goal: long-range coherence and capability
```

NanoSeek's Phase 1/Phase 2 split already captures Stage 1 + Stage 3. The missing piece
is Stage 2 — a domain-upweighting phase between general pre-training and long-context
extension.

### Data Pipeline Deliverables

```
nanoseek/data/
  ├── heuristic_filters.py       # Stage 1: rule-based filtering
  ├── quality_classifier.py      # Stage 2: fastText quality scoring
  ├── dedup.py                   # Stage 3: MinHash LSH deduplication
  ├── mixture.py                 # Stage 4: domain mixture config
  ├── domain_classifier.py       # Classify documents into domains
  ├── prepare_training_data.py   # End-to-end pipeline script
  └── README.md                  # Pipeline documentation with reproduction steps
```

### What This Demonstrates to Hiring Managers

A data curation pipeline shows you understand that:
1. Data quality matters more than architecture at fixed compute
2. Filtering is compute-dependent (you can't pick one filter for all budgets)
3. Deduplication is a scaling requirement (not just a nice-to-have)
4. Domain mixtures are a tunable hyperparameter (not a fixed choice)
5. You can build the full pipeline from raw data to training-ready tokens

This is the SINGLE MOST IMPACTFUL addition for frontier alignment. Labs hire
research engineers who understand data, not just models.

---

# PART 2: EVALUATION FRAMEWORK

## Why BPB Alone Is Insufficient

NanoSeek currently measures ONE metric: `ema_val_bpb`. This is the equivalent of measuring
a car's quality by its weight — correlated with quality but not the thing you actually care
about.

### What BPB tells you:
- The model's compression efficiency on held-out text
- Whether training is converging (loss going down)
- Relative comparison between runs at the same scale

### What BPB does NOT tell you:
- Whether the model can follow instructions
- Whether it can solve math problems
- Whether it can write code
- Whether it can reason about science
- Whether it has common-sense knowledge
- Whether MTP is actually helping (MTP improves generation but may not affect BPB)

### The BPB → Capability Gap

Two models with identical BPB can have VERY different downstream capabilities. This is
because BPB measures average cross-entropy across ALL tokens, but capabilities depend on
the model's performance on SPECIFIC token distributions (code tokens, math tokens,
reasoning chains).

A model that memorizes web boilerplate (cookie notices, navigation menus) can achieve
excellent BPB while being terrible at reasoning. Data curation reduces this risk, but
evaluation quantifies it.

## What to Build: NanoSeek Evaluation Harness

### Architecture

```
nanoseek/eval/
  ├── harness.py              # Main evaluation orchestrator
  ├── benchmarks/
  │   ├── bpb.py              # Bits-per-byte on validation set (existing)
  │   ├── arc.py              # ARC (science reasoning, few-shot)
  │   ├── mmlu.py             # MMLU (academic knowledge, 5-shot)
  │   ├── gsm8k.py            # GSM8K (grade-school math, chain-of-thought)
  │   ├── humaneval.py        # HumanEval (code generation, pass@k)
  │   ├── hellaswag.py        # HellaSwag (common sense, 10-shot)
  │   └── math500.py          # MATH-500 (competition math)
  ├── scaling_law.py          # Fit + predict scaling laws from checkpoint data
  ├── moe_diagnostics.py      # Expert-specific evaluation metrics
  └── report.py               # Generate evaluation report (markdown + W&B)
```

### Benchmark Selection Rationale

| Benchmark | What It Tests | Why Include | Difficulty for 1B |
|-----------|---------------|-------------|-------------------|
| **BPB** | Compression | Already have. Continuous metric. | N/A |
| **ARC-Easy** | Science reasoning | Standard for small models. Few-shot. | Achievable |
| **MMLU** (5-shot) | Broad knowledge | Industry standard. Compare to OLMoE. | Hard at 1B |
| **GSM8K** (CoT) | Math reasoning | Tests chain-of-thought. RL target. | Very hard |
| **HumanEval** | Code generation | Tests code capability. RL target. | Moderate |
| **HellaSwag** | Common sense | Standard for small models. | Achievable |
| **MATH-500** | Competition math | Frontier benchmark. RL ceiling test. | Very hard |

**Why these 6 + BPB (not more, not fewer)**:
- **Fewer**: Can't distinguish capability profiles. A model might ace ARC but fail GSM8K.
- **More**: Diminishing returns at 1B scale. Advanced benchmarks (GPQA, SWE-Bench)
  require capabilities that 1B models don't have. Testing on them produces noise.
- **These 6**: Cover the 4 capability axes that matter — reasoning, knowledge, code, math.
  Each has published baselines for ~1B models to compare against.

### Integration with lm-evaluation-harness

The right approach is NOT to rewrite evaluation from scratch. EleutherAI's
lm-evaluation-harness is the industry standard — used by NVIDIA, Cohere, BigScience,
and powers the HuggingFace Open LLM Leaderboard.

```python
# nanoseek/eval/harness.py

"""
NanoSeek evaluation harness.

Architecture decision: Wrap lm-evaluation-harness rather than reimplementing.

Why wrap instead of reimplement:
  1. lm-eval-harness has correct prompt templates for 60+ benchmarks
  2. Results are directly comparable to published numbers
  3. Community-validated — bugs are found and fixed by thousands of users
  4. HuggingFace leaderboard uses it — same harness = same numbers

Why wrap instead of use directly:
  1. NanoSeek uses EMA weights (not standard checkpoint format)
  2. NanoSeek has MoE-specific metrics (H_load, I_spec) that lm-eval doesn't track
  3. We want unified reporting (BPB + benchmarks + MoE diagnostics in one report)
  4. We need to evaluate at specific training checkpoints (not just final model)
"""

import json
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class EvalConfig:
    """Configuration for evaluation run."""
    checkpoint_path: str
    use_ema: bool = True                  # RULE 3: always evaluate EMA weights
    benchmarks: list[str] = field(default_factory=lambda: [
        "arc_easy",       # ARC Easy (25-shot)
        "mmlu",           # MMLU (5-shot)
        "gsm8k_cot",      # GSM8K with chain-of-thought (8-shot)
        "humaneval",       # HumanEval (pass@1, pass@10)
        "hellaswag",       # HellaSwag (10-shot)
    ])
    bpb_validation_set: str = "data/val.parquet"
    output_dir: str = "eval_results/"
    wandb_project: str = "nanoseek-eval"

    # MoE-specific diagnostics
    log_expert_routing: bool = True       # Per-benchmark expert activation patterns
    log_mtp_acceptance: bool = True       # MTP acceptance rate during generation


@dataclass
class EvalResult:
    """Structured evaluation result for one benchmark."""
    benchmark: str
    metric: str          # "accuracy", "pass@1", "bpb", etc.
    score: float
    num_samples: int
    few_shot: int
    metadata: dict = field(default_factory=dict)  # Expert routing stats, timing, etc.


def evaluate_checkpoint(config: EvalConfig) -> list[EvalResult]:
    """
    Run full evaluation suite on a NanoSeek checkpoint.

    Returns list of EvalResult objects, one per benchmark.
    Also logs to W&B if wandb_project is set.
    """
    results = []

    # 1. Load model (EMA weights if available)
    model = _load_model(config.checkpoint_path, use_ema=config.use_ema)

    # 2. BPB evaluation (custom — uses our dataloader)
    bpb_result = _evaluate_bpb(model, config.bpb_validation_set)
    results.append(bpb_result)

    # 3. Benchmark evaluation (via lm-evaluation-harness)
    for benchmark in config.benchmarks:
        result = _evaluate_benchmark(model, benchmark)
        results.append(result)

    # 4. MoE diagnostics (custom)
    if config.log_expert_routing:
        routing_stats = _evaluate_expert_routing(model, config.bpb_validation_set)
        results.append(routing_stats)

    # 5. MTP acceptance rate (custom)
    if config.log_mtp_acceptance:
        mtp_stats = _evaluate_mtp_acceptance(model)
        results.append(mtp_stats)

    # 6. Report generation
    _generate_report(results, config.output_dir)
    _log_to_wandb(results, config.wandb_project)

    return results
```

### Per-Domain BPB (The Missing Diagnostic)

Beyond aggregate BPB, measure BPB per domain:

```
Domain        BPB at 20%    BPB at 50%    BPB at 100%    Trend
──────────    ──────────    ──────────    ───────────    ─────
Web text        3.2           2.8           2.5          ↓ Normal
Code            4.1           3.4           2.9          ↓ Normal
Math            5.0           4.5           4.2          ↓ Slower (expected)
Books           3.0           2.6           2.3          ↓ Normal
Multilingual    4.8           4.2           3.8          ↓ Normal
```

This tells you:
- Whether the model is learning all domains (not just easy web text)
- Whether the data mixture is balanced (if one domain plateaus early, it's over-represented)
- Where to expect downstream benchmark performance (high code BPB → low HumanEval)

### Evaluation Schedule

```
When to evaluate (during training):

  Step 0:       BPB baseline (random model)
  Every 500:    BPB on validation set (cheap, <1 min)
  Every 2000:   BPB per domain (moderate, ~5 min)
  10%, 25%:     Full benchmark suite (expensive, ~30 min each)
  50%:          Full benchmark suite + scaling law check
  75%:          Full benchmark suite (before Phase 2 transition)
  100%:         Full benchmark suite (final evaluation)

Post Phase 2:   Full benchmark suite (measure long-context impact)
Post RL:        Full benchmark suite (measure RL impact per stage)
```

### What This Demonstrates to Hiring Managers

An evaluation framework shows you understand:
1. BPB is necessary but not sufficient for model quality assessment
2. Downstream benchmarks measure what users actually care about
3. Evaluation must be reproducible (same harness → same numbers)
4. MoE models need additional diagnostics beyond standard benchmarks
5. Evaluation informs training decisions (not just final reporting)

---

# PART 3: SCALING LAW FITTING

## Why This Matters Despite "Only" 3 Data Points

NanoSeek has 3 scale points: anchor (~55M active), nano-500M (~441M), NanoSeek-1B (1.08B).
The EXPERIMENT_REASONING.md plan uses these for muP validation but never fits a scaling law.

### What a scaling law gives you:

1. **Prediction**: Before training 1B, predict its final loss from anchor + 500M data
2. **Validation**: If 1B loss matches prediction → scaling is working correctly
3. **Diagnosis**: If 1B loss deviates from prediction → something specific is wrong
4. **Extrapolation**: Predict what a 3B or 7B model would achieve (for the paper)

### The Minimum Viable Scaling Law

With 3 points, you can fit the Chinchilla-style power law:

```
L(N) = E + A / N^α

Where:
  L = final ema_val_bpb
  N = active parameters
  E = irreducible loss (entropy of the data)
  A = scaling coefficient
  α = scaling exponent

With 3 data points (55M, 441M, 1.08B), you have 3 equations and 3 unknowns.
This is exactly determined — no uncertainty estimate, but sufficient for
prediction and validation.
```

**The key insight**: You don't need to FIT the scaling law at the start. You train all
3 models as planned, then fit the curve POST-HOC. The scaling law is a VALIDATION tool,
not a PLANNING tool (at your budget).

But the act of fitting it, interpreting it, and reporting it demonstrates scaling law
literacy — which is a core competency for research engineers.

### What to Build

```python
# nanoseek/eval/scaling_law.py

"""
Scaling law fitting for NanoSeek.

Fits L(N) = E + A * N^(-alpha) from training checkpoints across scales.

Theory (Hoffmann et al., 2022 — Chinchilla):
  For compute-optimal training (D ∝ N), the loss scales as a power law in N.
  The exponent alpha is architecture-dependent (typically 0.05-0.10 for transformers).

For MoE, DeepSeek showed:
  - MoE scaling exponent is HIGHER than dense (more efficient scaling)
  - Active parameters (not total) determine the effective scale
  - Granularity ratio (κ = top_k/n_experts) affects the scaling coefficient

We fit using N_active (not N_total) because:
  1. Active parameters determine per-token compute
  2. μP-MoE transfers HPs using N_active
  3. DeepSeek's scaling analysis uses N_active
"""

import numpy as np
from scipy.optimize import curve_fit
from dataclasses import dataclass

@dataclass
class ScalingPoint:
    """One data point on the scaling curve."""
    n_active: int           # Active parameters
    n_total: int            # Total parameters
    tokens_trained: int     # Number of training tokens
    final_bpb: float        # Final ema_val_bpb
    compute_flops: float    # Total training FLOPs

@dataclass
class ScalingLaw:
    """Fitted scaling law parameters."""
    E: float       # Irreducible loss
    A: float       # Scaling coefficient
    alpha: float   # Scaling exponent
    r_squared: float  # Fit quality (should be > 0.99 for 3 points)

def fit_scaling_law(points: list[ScalingPoint]) -> ScalingLaw:
    """
    Fit L(N) = E + A * N^(-alpha) to observed data points.

    With 3 points, this is exactly determined. With more points,
    uses least-squares fitting (more robust to noise).
    """
    N = np.array([p.n_active for p in points])
    L = np.array([p.final_bpb for p in points])

    def power_law(n, E, A, alpha):
        return E + A * np.power(n, -alpha)

    # Initial guess: E=2.0 (typical for good data), A=10.0, alpha=0.07
    popt, pcov = curve_fit(
        power_law, N, L,
        p0=[2.0, 10.0, 0.07],
        bounds=([0, 0, 0.01], [5, 1000, 0.5]),
        maxfev=10000
    )

    E, A, alpha = popt

    # R-squared
    L_pred = power_law(N, E, A, alpha)
    ss_res = np.sum((L - L_pred) ** 2)
    ss_tot = np.sum((L - np.mean(L)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    return ScalingLaw(E=E, A=A, alpha=alpha, r_squared=r_squared)

def predict_loss(law: ScalingLaw, n_active: int) -> float:
    """Predict ema_val_bpb for a model with n_active parameters."""
    return law.E + law.A * n_active ** (-law.alpha)

def validate_prediction(
    law: ScalingLaw,
    actual: ScalingPoint,
    tolerance: float = 0.05
) -> bool:
    """
    Check if actual loss matches scaling law prediction.

    tolerance=0.05 means we accept ±0.05 BPB deviation.
    This is the same threshold used in the muP coordinate check.
    """
    predicted = predict_loss(law, actual.n_active)
    deviation = abs(actual.final_bpb - predicted)
    return deviation < tolerance
```

### Two-Stage Prediction (FLOPs → Loss → Performance)

The frontier practice is to predict DOWNSTREAM performance, not just loss:

```
Stage 1: FLOPs → Loss   (power law: L = A * C^(-alpha))
Stage 2: Loss → Score    (sigmoid: S = 1 / (1 + exp(-k * (L_threshold - L))))
```

Stage 2 uses the empirical finding that "models with similar pre-training loss show
comparable task performance regardless of specific training trajectory."

With your 3 training runs, you can:
1. Fit Stage 1 from anchor + 500M + 1B losses
2. Fit Stage 2 from anchor + 500M + 1B benchmark scores
3. Predict what a 3B or 7B NanoSeek would score on each benchmark

This prediction is itself a publishable finding: "Scaling law validation for MoE+MLA
with muP HP transfer."

---

# PART 4: TRAINING OBSERVABILITY

## What's Missing from the Current Training Loop

The `pre_train.py` script logs basic metrics (loss, learning rate, gradient norm) but
is missing the MoE-specific observability that catches silent failures.

### The Silent Failure Problem

MoE models have failure modes that don't show up in aggregate loss:

1. **Expert Collapse**: 58 of 64 experts become inactive. Loss looks fine because
   the remaining 6 experts carry all the load. But the model has 9× fewer effective
   parameters than intended. H_load drops below 2 bits but loss barely changes.

2. **Routing Oscillation**: Experts rapidly swap their roles every ~100 steps.
   Loss looks stable on average but the model never develops stable expert
   specialization. I_spec stays near zero.

3. **MTP Degradation**: The MTP module stops contributing useful gradients.
   MTP loss stays low (MTP predicts easy tokens) but MTP acceptance rate
   doesn't improve. Inference speed gains are illusory.

4. **Attention Logit Growth**: Q and K norms grow slowly over training.
   Loss looks fine until softmax saturates → sudden catastrophic spike.
   By the time loss spikes, the model may be unrecoverable.

### What to Log: The Complete MoE Dashboard

```
Logged every 10 steps (cheap, <1ms overhead):
  ├── train_loss              # Total training loss
  ├── main_loss               # Next-token prediction loss
  ├── mtp_loss                # MTP auxiliary loss
  ├── grad_norm               # Global gradient norm (detect spikes)
  ├── per_group_lr            # LR for each parameter group (verify muP)
  ├── gpu_memory_allocated    # Memory usage (detect leaks)
  └── tokens_per_second       # Throughput (detect degradation)

Logged every 100 steps (moderate, ~10ms overhead):
  ├── H_load                  # Expert load-balance entropy (bits)
  │                           # = -Σ p_i * log2(p_i) where p_i = fraction of
  │                           #   tokens routed to expert i
  │                           # Healthy: > 4 bits (for 64 experts, max = 6 bits)
  │                           # Warning: < 3 bits
  │                           # Critical: < 2 bits (expert collapse)
  ├── load_per_expert         # Token count per expert (histogram)
  ├── router_entropy          # Entropy of routing probabilities (before top-k)
  ├── qk_norm_ratio           # ||Q|| / ||K|| ratio (detect attention logit growth)
  └── expert_bias_mean_std    # Aux-loss-free bias statistics

Logged every 500 steps (expensive, ~30s overhead):
  ├── ema_val_bpb             # RULE 3: the metric that matters
  ├── domain_bpb              # Per-domain validation BPB
  ├── I_spec                  # Expert specialization mutual information
  │                           # = MI(token_domain, expert_id)
  │                           # Measures whether experts develop semantic roles
  │                           # Should increase over training (experts specialize)
  ├── mtp_acceptance_rate     # Fraction of MTP predictions accepted by verifier
  ├── routing_heatmap         # (domain × expert) activation matrix
  └── MFU                     # Model FLOPs Utilization
                              # = actual_flops / theoretical_peak_flops
                              # Target: 47% for MoE on A100 (lower than dense
                              # due to routing overhead + expert imbalance)
```

### Alert System

```python
# nanoseek/monitoring/alerts.py

"""
Real-time training health alerts.

These catch silent MoE failures that don't show up in aggregate loss.
Each alert has a threshold, a severity, and a recommended action.
"""

ALERTS = {
    'expert_collapse': {
        'condition': 'H_load < 2.0',
        'severity': 'CRITICAL',
        'action': 'Stop training. Check gamma, router_lr, bias update.',
        'explanation': (
            'Expert collapse means most tokens route to a few experts. '
            'The model has far fewer effective parameters than intended. '
            'Common causes: router_lr too low relative to expert_lr '
            '(experts change faster than routing can follow), or gamma '
            'too small to counteract natural winner-take-all dynamics.'
        )
    },
    'expert_imbalance_warning': {
        'condition': 'H_load < 3.0',
        'severity': 'WARNING',
        'action': 'Monitor. If trend is downward, consider increasing gamma.',
    },
    'gradient_spike': {
        'condition': 'grad_norm > 10 * moving_avg_grad_norm',
        'severity': 'WARNING',
        'action': 'Check data batch. Grad clip should handle it.',
    },
    'attention_logit_growth': {
        'condition': 'qk_norm_ratio > 2.0 * initial_qk_norm_ratio',
        'severity': 'WARNING',
        'action': 'QK norms growing. If no QK-norm: consider adding it.',
    },
    'mtp_stagnation': {
        'condition': 'mtp_acceptance_rate unchanged for 2000 steps',
        'severity': 'INFO',
        'action': 'MTP module may not be learning. Check MTP loss trend.',
    },
    'mfu_degradation': {
        'condition': 'MFU < 0.35',
        'severity': 'WARNING',
        'action': 'Throughput below expected. Check expert load balance.',
        'explanation': (
            'Low MFU in MoE usually means severe load imbalance — some '
            'GPUs wait while others process overloaded experts. Or memory '
            'pressure causing swapping. Or communication bottleneck.'
        )
    },
}
```

---

# PART 5: POST-TRAINING ENGINEERING

## Current State

NanoSeek's MAIN_PLAN.md describes a 3-stage RL post-training pipeline:
- Stage 1: Reasoning RL (GRPO, 60% budget)
- Stage 2: Agent RL (GRPO, 25% budget)
- Stage 3: General Alignment (DPO, 15% budget)

This is a PLAN. No code exists. The gap is implementation.

## What Frontier Labs Actually Do (2025-2026)

### DeepSeek R1 Pipeline (The Gold Standard for Reasoning RL)

```
Phase 1: Cold-Start SFT
  - Small amount of curated reasoning examples (~thousands)
  - Teaches the model the FORMAT of chain-of-thought
  - NOT teaching it to reason — just teaching it to show its work

Phase 2: Reasoning RL (GRPO)
  - Prompts: math, code, logic problems with VERIFIABLE answers
  - Reward: binary (correct answer = 1, wrong = 0)
  - No reward model needed (rule-based verification)
  - The model learns to reason by TRIAL AND ERROR
  - Emergent behaviors: self-reflection, verification, backtracking

Phase 3: Rejection Sampling + SFT
  - Generate many completions for diverse prompts
  - Keep only the best completions (by reward)
  - SFT on the best completions
  - This "distills" the RL policy into clean supervised examples

Phase 4: General RL
  - Broader reward signals (helpfulness, safety, formatting)
  - May use reward model for non-verifiable tasks
  - Lower learning rate than Phase 2
```

### The Key Engineering Challenges

**1. GRPO Implementation**

GRPO is simpler than PPO but still requires careful engineering:

```python
# Pseudocode for GRPO training step

def grpo_step(policy, prompts, reward_fn, ref_policy, config):
    """
    Group Relative Policy Optimization.

    For each prompt, generate K completions from the current policy.
    Compute advantages from intra-group reward distribution.
    Update policy to maximize expected advantage while staying close to reference.

    Why GRPO over PPO:
      - No critic/value network needed (saves 50% memory)
      - Advantages estimated from group statistics (more stable)
      - DeepSeek R1 showed GRPO matches PPO on reasoning tasks
      - Simpler to implement and debug

    Why GRPO over DPO:
      - DPO requires paired preferences (hard to collect for reasoning)
      - GRPO works with scalar rewards (easy for verifiable tasks)
      - DPO can't improve beyond the preference data quality
      - GRPO can discover novel reasoning strategies via exploration
    """
    all_losses = []

    for prompt in prompts:
        # 1. Generate K completions from current policy
        completions = policy.generate(prompt, n=config.group_size)  # K=8 typical

        # 2. Compute rewards (rule-based for math/code)
        rewards = [reward_fn(prompt, c) for c in completions]

        # 3. Compute group-relative advantages
        mean_r = np.mean(rewards)
        std_r = np.std(rewards) + 1e-8
        advantages = [(r - mean_r) / std_r for r in rewards]

        # 4. Compute policy gradient with KL penalty
        for completion, advantage in zip(completions, advantages):
            log_prob = policy.log_prob(prompt, completion)
            ref_log_prob = ref_policy.log_prob(prompt, completion)
            kl = log_prob - ref_log_prob

            loss = -(advantage * log_prob - config.kl_coeff * kl)
            all_losses.append(loss)

    # 5. Backward and update
    total_loss = torch.stack(all_losses).mean()
    total_loss.backward()
    optimizer.step()
```

**2. Rule-Based Reward Functions**

```python
# nanoseek/rl/rewards.py

"""
Rule-based reward functions for GRPO.

Why rule-based (not model-based):
  1. No reward hacking — the answer is objectively correct or not
  2. No reward model training needed — saves compute and complexity
  3. DeepSeek R1 showed rule-based rewards are SUFFICIENT for
     emergent reasoning (self-reflection, verification, etc.)
  4. Model-based rewards are needed ONLY for subjective tasks
     (helpfulness, style) — not for math/code

For NanoSeek at 1B scale, rule-based rewards are the right choice
because our RL budget is limited and we want to maximize the
signal-to-noise ratio of the reward signal.
"""

def math_reward(prompt: str, completion: str, ground_truth: str) -> float:
    """
    Binary reward for math problems.
    Extracts the final numerical answer and compares to ground truth.
    """
    extracted = extract_final_answer(completion)
    if extracted is None:
        return 0.0  # No answer found — penalize
    return 1.0 if is_numerically_equal(extracted, ground_truth) else 0.0

def code_reward(prompt: str, completion: str, test_cases: list) -> float:
    """
    Execution-based reward for code generation.
    Runs the generated code against test cases in a sandbox.
    Returns fraction of test cases passed.
    """
    try:
        code = extract_code_block(completion)
        results = execute_in_sandbox(code, test_cases, timeout=10)
        return sum(results) / len(results)
    except (TimeoutError, SyntaxError, RuntimeError):
        return 0.0

def format_reward(completion: str) -> float:
    """
    Reward for following the expected output format.
    Checks for <think>...</think> and <answer>...</answer> tags.
    This teaches the model to separate reasoning from final answer.
    """
    has_think = '<think>' in completion and '</think>' in completion
    has_answer = '<answer>' in completion and '</answer>' in completion
    return 0.5 * has_think + 0.5 * has_answer
```

**3. MoE-Specific RL Stabilization (DeepSeek V3.2)**

This is NanoSeek's UNIQUE contribution — no open-source project implements all 4
MoE stabilization techniques for RL:

```
1. Unbiased KL Penalty
   Standard KL: KL(π || π_ref) computed per-token
   Problem: MoE routing changes during RL → experts see different tokens →
   per-expert KL is biased (comparing different data distributions)
   Fix: Use importance-weighted KL that corrects for routing changes

2. Off-Policy Masking
   Problem: During GRPO, some completions in the group were generated by
   an earlier version of the policy (off-policy). Their gradients are stale.
   Fix: Mask out completions where |log π(x) - log π_old(x)| > threshold

3. Keep Routing (CRITICAL for Agent RL)
   Problem: RL gradients push tokens toward "reward-maximizing" experts,
   destroying the routing distribution learned during pre-training
   Fix: Freeze router weights during RL. Only update expert FFN weights.
   This preserves expert specialization while allowing expert behavior to change.

4. Keep Sampling Mask
   Problem: Temperature sampling during RL generation creates high-variance
   gradient estimates for rare tokens
   Fix: Apply the same sampling mask across all K completions in a group,
   so the variance comes from policy differences, not sampling randomness
```

### What This Demonstrates to Hiring Managers

RL post-training implementation shows you understand:
1. The full LLM pipeline (pre-training is only half the story)
2. GRPO vs PPO vs DPO trade-offs (not just using one blindly)
3. Reward engineering (rule-based vs model-based, when each is appropriate)
4. MoE-specific challenges in RL (routing stability, KL bias)
5. You can implement from paper to code (not just calling APIs)

---

# PART 6: REPRODUCIBILITY AND RIGOR

## Why This Matters for Hiring

Frontier labs have been burned by irreproducible results. Meta's Llama 3 paper
explicitly discusses reproducibility. OpenAI's interview process tests experimental
rigor. Demonstrating reproducibility in NanoSeek signals "this person won't waste
6 months of cluster time on a bug that deterministic training would have caught."

### What to Add

**1. Seed Control**

```python
# nanoseek/common.py — add to existing

def set_deterministic_mode(seed: int = 42):
    """
    Set all random seeds for reproducible training.

    Why this matters for MoE specifically:
    - Router decisions depend on random initialization
    - Expert assignment in early training is sensitive to initial conditions
    - Without seed control, two identical runs can develop DIFFERENT
      expert specialization patterns, making ablation comparison meaningless
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic algorithms (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch 2.0+ deterministic mode
    torch.use_deterministic_algorithms(True, warn_only=True)
```

**2. Config Serialization**

Every training run should save its COMPLETE configuration as JSON, so any run
can be reproduced exactly:

```python
# nanoseek/config.py — add to existing

def save_run_config(config, output_path: str):
    """
    Save complete training configuration for reproducibility.

    Includes: model config, optimizer config, data config, hardware info,
    git commit hash, library versions, random seeds.

    This is what frontier labs do — every run is traceable back to exact code
    and configuration.
    """
    import json
    import subprocess
    import torch

    run_info = {
        'model_config': asdict(config),
        'git_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).decode().strip(),
        'git_diff_stat': subprocess.check_output(
            ['git', 'diff', '--stat']
        ).decode().strip(),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'gpu_count': torch.cuda.device_count(),
    }

    with open(output_path, 'w') as f:
        json.dump(run_info, f, indent=2, default=str)
```

**3. Experiment Tracking**

Every experiment should have a unique ID, tracked in a central registry:

```
experiments/
  ├── registry.json           # Maps experiment_id → config + results
  ├── anchor_grid_001/        # HP grid search run 1
  │   ├── config.json
  │   ├── metrics.json
  │   └── checkpoints/
  ├── anchor_grid_002/
  ├── ...
  ├── stability_A/
  ├── stability_C/
  ├── stability_D/
  ├── nano_500m/
  └── nanoseek_1b/
```

---

# PART 7: INFERENCE OPTIMIZATION

## Why Include This (Even for a Training Project)

Frontier labs evaluate research engineers on their understanding of the FULL lifecycle.
A model that's been trained but can't be efficiently served demonstrates incomplete
thinking.

### What to Build

**1. KV Cache Quantization Analysis**

MLA already provides 23× KV cache compression. But the remaining KV cache can be further
quantized for deployment:

```
Standard attention KV cache:  2 × n_layers × n_heads × d_head × seq_len × 2 bytes (bf16)
MLA compressed KV cache:      n_layers × d_compressed × seq_len × 2 bytes
Further quantized (INT8):     n_layers × d_compressed × seq_len × 1 byte

For NanoSeek-1B at 8K context:
  Standard attention: 16 × 2 × 16 × 128 × 8192 × 2 = 1.07 GB
  MLA compressed:     16 × 143 × 8192 × 2 = 37.5 MB (28× smaller)
  MLA + INT8:         16 × 143 × 8192 × 1 = 18.7 MB (57× smaller)
```

Document this analysis. It demonstrates you understand WHY MLA matters for deployment,
not just how to implement it.

**2. MTP Speculative Decoding Benchmark**

MTP enables speculative decoding at inference time. Benchmark it:

```
Metric                  Without MTP    With MTP (acceptance=70%)
──────────────────────  ───────────    ────────────────────────
Tokens/second (A100)    ~150           ~240 (1.6× speedup)
Tokens/second (4090)    ~80            ~128 (1.6× speedup)
Latency per token       ~6.7ms         ~4.2ms
Batch throughput        ~2400 t/s      ~3200 t/s
```

The MTP acceptance rate is the key metric. It should increase after RL post-training
(DeepSeek V3.2 reports this). Measuring this before and after RL is a novel finding.

---

# PART 8: TEST-TIME COMPUTE SCALING

## Why This Is the 2026 Frontier

The core finding from 2025-2026 research: "A smaller model with more inference compute
can outperform a larger model with standard inference." This makes test-time compute
the new scaling axis.

### What NanoSeek Should Measure

```
For each benchmark (GSM8K, MATH-500, HumanEval):
  1. Standard inference: 1 completion, greedy decoding
  2. Best-of-N: Generate N={4, 8, 16, 32} completions, pick best
  3. Chain-of-thought: Prompt with "Let's think step by step"
  4. Self-consistency: Generate N completions, majority vote on answer
  5. Sequential refinement: Generate → critique → refine (3 rounds)

Plot: accuracy vs total inference tokens (log scale)
```

This produces the "test-time compute scaling curve" — the signature plot of 2025-2026
AI research. For NanoSeek, this curve shows:
- How much test-time compute compensates for NanoSeek's small size
- Whether RL training shifts the curve (it should — RL teaches exploration)
- Whether MTP acceptance rate predicts test-time scaling efficiency

### The Novel Finding Opportunity

No published work measures test-time compute scaling for MoE+MTP models. NanoSeek can
be the first to answer: "Does MTP's speculative decoding interact with test-time compute
scaling?" Hypothesis: MTP makes best-of-N cheaper (faster generation per completion),
effectively giving you more test-time compute per dollar.

---

# PART 9: MECHANISTIC INTERPRETABILITY ON MoE (SAE + Expert Analysis)
# ⚡ TARGET-SPECIFIC: Best for Anthropic interpretability / DeepMind alignment roles

## Why This Is a Frontier Differentiator

MACHANIC_INTERPRET.md already lays out the research plan. Here's why it matters and
what to build — this is the intersection of interpretability + MoE + RL that no
published work has fully explored.

### The Research Gap (From MACHANIC_INTERPRET.md §7)

```
UNEXPLORED (as of March 2026):
  1. No SAE analysis of individual expert activations in MoE models
  2. No circuit-level analysis of how GRPO changes MoE computation graphs
  3. No systematic tracking of expert routing distributions DURING GRPO training
  4. No comparison of feature evolution in shared vs routed experts during RL
  5. No transcoder/CLT analysis of MoE architectures
```

NanoSeek can address gaps 1-4 directly. This is greenfield research territory.

### What to Build

**1. Expert-Level SAE Training (~$20 GPU cost)**

```python
# nanoseek/interp/expert_sae.py

"""
Train separate SAEs on individual expert activations.

Why per-expert SAEs (not one SAE on all experts):
  - Each expert learns different features (code expert ≠ math expert)
  - A single SAE on concatenated expert outputs conflates features
  - Per-expert SAEs reveal what each expert COMPUTES, not just what
    the aggregated output represents

Architecture:
  - Input: expert_output for expert_i (dim = moe_inter_dim = 768 for 1B)
  - Hidden: 32× expansion (24,576 features) — follows Anthropic's recipe
  - Sparsity: K=32 (TopK SAE, same as "How LLMs Learn" paper)
  - Training data: 1M tokens of expert outputs from a fixed eval set
  - Cost: ~$2-3 per expert × 8 experts (top-8, not all 64)
"""

from saelens import SAE, SAEConfig  # SAELens library

def train_expert_saes(model, eval_dataset, expert_indices=[0,8,16,24,32,40,48,56]):
    """Train SAEs on 8 representative experts (evenly spaced)."""
    saes = {}
    for expert_id in expert_indices:
        config = SAEConfig(
            d_in=model.config.moe_inter_dim,          # 768
            expansion_factor=32,                        # 32× = 24,576 features
            k=32,                                       # TopK sparsity
            normalize_activations="expected_l2_norm",   # Anthropic standard
        )
        # Collect activations from this expert only
        activations = collect_expert_activations(model, eval_dataset, expert_id)
        sae = SAE(config)
        sae.fit(activations)
        saes[expert_id] = sae
    return saes
```

**2. Feature Birth/Death Tracking Across GRPO**

```
Methodology (adapted from "How LLMs Learn" paper):

1. Select 6 GRPO checkpoints: pre-RL, 1%, 5%, 25%, 50%, 100%
2. At each checkpoint, train expert SAEs with IDENTICAL hyperparameters
3. For each expert, compare SAE features across checkpoints:
   - Feature BIRTH: exists at checkpoint N but not N-1 (new capability)
   - Feature DEATH: exists at checkpoint N-1 but not N (suppressed)
   - Feature DRIFT: exists in both but activation pattern changed

Metrics:
  - Cosine similarity between corresponding features across checkpoints
  - Feature activation frequency changes
  - Top-50 activation examples for qualitative assessment
  - Monosemanticity scores (do features become more/less interpretable?)

What we expect to find:
  - Shared experts: minimal feature change (broad knowledge preserved)
  - "Math" experts: feature BIRTH for self-correction, verification
  - "Code" experts: feature BIRTH for test-case reasoning
  - All experts: NO feature death for basic language features
```

**3. fTRI (Functional Token Resonance Imaging) for NanoSeek**

```python
# nanoseek/interp/ftri.py

"""
Adapted from MoTE paper (arXiv:2502.11096).

fTRI maps behavioral categories to specific experts:
  1. Sum expert activations across all prompt tokens → per-prompt activation map
  2. Classify response maps: REASONED, REFUSED, ALIGNED, CODE, MATH
  3. Average activation maps within each category
  4. Differential activation: target class - all other classes

For NanoSeek at 1B scale with 64 experts:
  - Activation map shape: (n_moe_layers × 64) = (14 × 64) = 896 dimensions
  - Much smaller than DeepSeek-R1 (58 × 256 = 14,848) → faster analysis
  - But same methodology applies

Key experiment:
  - Run fTRI BEFORE and AFTER each GRPO stage
  - Track which experts become "reasoning specialists" vs "alignment specialists"
  - Compare with H_load and I_spec metrics (ground truth for routing changes)
  - Validate that fTRI-identified experts match I_spec domain heatmap
"""
```

**4. Alignment Faking Detection (Anthropic-Style Probes)**

```
Even at 1B scale, we can demonstrate the METHODOLOGY:

1. Train linear probes on residual stream at layer 8 (~50% depth)
   - Concept: "helpful" vs "harmful" activation directions
   - Data: 500 contrast pairs (helpful response vs refusal)
   - Expected: >95% AUROC on held-out set

2. Test probe transfer across GRPO stages
   - Do "helpfulness" probes from pre-RL still work post-RL?
   - If yes: RL preserved the representation structure
   - If no: RL fundamentally changed how the model represents concepts

3. Decision timing analysis
   - At which layer does the model "decide" to reason vs refuse?
   - Does GRPO shift this decision to earlier layers? (hypothesis: yes)

This demonstrates safety-relevant interpretability skills —
a CRITICAL differentiator for Anthropic/DeepMind hiring.
```

### What This Demonstrates to Hiring Managers

Mechanistic interpretability on MoE shows:
1. You understand the frontier of interpretability research (SAEs, probes, circuits)
2. You can apply these tools to novel architectures (MoE — no prior work exists)
3. You think about safety and alignment (not just capability)
4. You can design and execute interpretability experiments (not just use pretrained SAEs)
5. You contribute to open research questions (gaps 1-4 above)

---

# PART 10: CANON × MoE ABLATION STUDY

## Why Canon Layers Matter for MoE

Allen-Zhu's "Physics of Language Models" (Parts 4.1 & 4.2) identifies THREE axes of
information flow in transformers:

```
1. Vertical:           Residual stream depth (attention → MLP → attention → ...)
2. Global horizontal:  Attention across ALL positions (causal mask)
3. Local horizontal:   Information flow between ADJACENT tokens ← THIS IS MISSING
```

Standard transformers lack axis 3. Canon Layers add it via depthwise causal 1D convolutions.
The theory predicts Canon-C (pre-MoE insertion) is most valuable for MoE because it
enriches router inputs with local context — the router sees "this token AND its neighbors"
instead of just "this token."

### The NanoSeek Canon Experiment (From 05_CANON_LAYERS_DEEP_DIVE.md)

```
5 runs at anchor scale (~$10 total):

Run 0: Baseline (no Canon layers)
Run 1: Canon-A only (pre-attention)
Run 2: Canon-C only (pre-MoE) ← predicted winner
Run 3: Canon-A + Canon-C (both)
Run 4: Canon-ABCD (all 4 insertion points)

Measurements:
  - ema_val_bpb (primary)
  - I_spec (does Canon-C increase expert specialization?)
  - H_load (does Canon affect load balance?)
  - Router entropy per layer (does local context change routing distribution?)

Canon-C hypothesis:
  Without Canon-C: router(x_t) — routes based on single token
  With Canon-C: router(conv(x_{t-k:t})) — routes based on local window
  → Router sees richer input → better expert selection → higher I_spec

Canon-C × MoE interaction (novel):
  No published work studies Conv × MoE routing interaction.
  Allen-Zhu's theory predicts it but experiments are on dense models only.
  NanoSeek would be the FIRST to measure Canon-C's effect on MoE routing.
```

### Cross-Architecture Canon Comparison (~$40)

If Canon-C helps at anchor scale, validate at 500M and 1B:

```
Width 480  (anchor):  Canon-C improves I_spec by +X nats
Width 1280 (500M):    Canon-C improves I_spec by +Y nats
Width 2048 (1B):      Canon-C improves I_spec by +Z nats

Research question: Does Canon-C's MoE routing improvement scale with width?
  - If X ≈ Y ≈ Z: constant benefit (architecture-level improvement)
  - If Z > Y > X: increasing benefit at scale (Canon-C becomes more important)
  - If Z < Y < X: diminishing benefit (wider models already capture local context)
```

---

# PART 11: SCALE VALIDATION — 3B-7B TRAINING
# ⚡ TARGET-SPECIFIC: Best for roles that specifically require scale experience
# NOTE: 7B ($2000+) has diminishing returns — 3B usually sufficient

## Why Scale Matters for the Portfolio

Training at 1B proves you can build. Training at 3B proves you can SCALE.
But a clean 1B with excellent ablations beats a sloppy 7B every time.
Nobody expects a solo researcher to train 7B — interviewers care about
methodology, not that you spent $2000.

### The Scaling Ladder

```
Scale          Active    Total     Cost      What It Proves
──────────    ───────   ──────    ──────    ──────────────────────────────
NanoSeek-1B    1.08B    4.75B     $300      Architecture + HP transfer
NanoSeek-3B    3.2B     ~14B      $800      muP transfer at 3× + FSDP2
NanoSeek-7B    7.0B     ~30B      $2000+    Multi-node + expert parallelism

Scale validation: does NanoSeek's recipe produce competitive results at each scale?
```

### NanoSeek-3B (~$800, 2-3 days on 8×A100)

```
Config: 16 layers, 3072 hidden, 64 experts, top-8
  κ = 12.5% (same), moe_inter/hidden = 0.375 (same)
  N_active ≈ 3.2B, N_total ≈ 14B
  Tokens: 64B (Chinchilla-optimal: 20× N_active)

What this validates:
  1. muP HP transfer at 6.4× width ratio (480→3072)
  2. FSDP2 sharding at a scale where DDP runs OOM
  3. Scaling law prediction from 3 prior points (55M, 441M, 1.08B)
  4. Expert routing dynamics at higher capacity (more parameters per expert)

New infrastructure needed:
  - FSDP2 integration (shard model + optimizer across 8 GPUs)
  - Gradient checkpointing tuning (selective per-layer)
  - Communication profiling (all-gather vs reduce-scatter balance)
```

### NanoSeek-7B (~$2000+, multi-node)

```
Config: 24 layers, 4096 hidden, 128 experts, top-16
  κ = 12.5% (same), moe_inter/hidden = 0.375 (same)
  N_active ≈ 7.0B, N_total ≈ 30B
  Tokens: 140B (Chinchilla-optimal)

CRITICAL CHANGE: n_experts doubles (64→128), depth increases (16→24)

What this validates:
  1. Multi-node distributed training (2+ nodes, 16+ GPUs)
  2. Expert parallelism (128 experts across 16 GPUs = 8 experts/GPU)
  3. Depth scaling (16→24 layers — muP doesn't cover this, need separate validation)
  4. At this scale, results are PUBLISHABLE and comparable to OLMoE, Mixtral 8x7B

New infrastructure needed:
  - Megatron-Core integration OR PyTorch native expert parallelism
  - Inter-node communication optimization (NCCL tuning)
  - Fault tolerance (checkpoint-and-restart for multi-day training)
  - Sequence packing for efficient GPU utilization
```

### Width vs Depth Pareto Analysis (~$10)

Before committing to 3B/7B configs, determine the optimal width/depth ratio:

```
At ~3B active, compare:
  Config A: 16 layers, 3072 hidden  (wide and shallow)
  Config B: 24 layers, 2560 hidden  (balanced)
  Config C: 32 layers, 2048 hidden  (deep and narrow)

Each run: 2000 steps, measure ema_val_bpb.

This answers: "For MoE+MLA, is depth or width more compute-efficient?"
Dense models favor depth (Llama 3). MoE models may differ because
more experts already provide capacity — depth may be less critical.
```

---

# PART 12: ADVANCED POST-TRAINING — PRM, CAI, ITERATIVE DPO
# ⚡ TARGET-SPECIFIC: PRM → Anthropic safety / OpenAI reasoning
#                     CAI → Anthropic safety specifically
#                     Iterative DPO → strong value-add (1 round recommended)

## Beyond Rule-Based Rewards: Process Reward Model

### Why Rule-Based Rewards Hit a Ceiling

Part 5 describes rule-based rewards (binary correct/wrong for math, test-case pass/fail
for code). These work for VERIFIABLE tasks. But frontier labs need:

```
Tasks where rule-based rewards FAIL:
  - Open-ended reasoning (no single correct answer)
  - Multi-step proofs (intermediate steps matter, not just conclusion)
  - Agent tasks (many valid action sequences)
  - Creative writing, summarization, instruction following
```

### Process Reward Model (PRM) — $10-20

A PRM scores each STEP of reasoning, not just the final answer:

```
Example:
  Prompt: "Solve 15 × 23"
  Step 1: "15 × 23 = 15 × 20 + 15 × 3"    → PRM score: 1.0 (correct decomposition)
  Step 2: "= 300 + 45"                       → PRM score: 1.0 (correct arithmetic)
  Step 3: "= 345"                            → PRM score: 1.0 (correct sum)

vs:
  Step 1: "15 × 23 ≈ 15 × 25"               → PRM score: 0.3 (wrong approximation)
  Step 2: "= 375"                            → PRM score: 0.0 (wrong from step 1)
```

**Why PRM matters for MoE**:
- Different experts handle different reasoning steps
- PRM signal can identify which experts contribute to WRONG steps
- Combined with fTRI: which expert activated → which step failed → which expert to fix
- This is the MoE + interpretability + RL intersection

**Implementation**:

```python
# nanoseek/rl/process_reward.py

"""
Process Reward Model for step-level reasoning supervision.

Architecture: Same as NanoSeek-1B but with a scalar value head.
Training data: GSM8K + MATH solutions with step-level annotations.

Two approaches:
1. AUTOMATED (cheaper, ~$10): Use the NanoSeek model itself to generate
   step-by-step solutions, verify each step programmatically (for math),
   label correct/incorrect steps → train PRM on these labels.

2. MONTE CARLO (more accurate, ~$20): For each reasoning step, sample
   N completions from that point → compute fraction that reach correct
   final answer → step score = completion rate. (OpenAI's original PRM method)

For NanoSeek, use approach 1 first. If PRM-guided GRPO outperforms
rule-based GRPO on MATH-500, the PRM is validated.
"""

def train_prm(base_model, math_dataset, method="automated"):
    """Train a process reward model from step-annotated solutions."""
    if method == "automated":
        # Generate solutions, verify steps programmatically
        labeled_data = auto_annotate_steps(base_model, math_dataset)
    elif method == "monte_carlo":
        # Sample completions from each step, compute completion rates
        labeled_data = monte_carlo_annotate(base_model, math_dataset, n_samples=16)

    # Add value head to base model
    prm = add_value_head(base_model, output_dim=1)
    prm.train_on(labeled_data, loss="binary_cross_entropy")
    return prm
```

### Constitutional AI Self-Critique Loop — $5

Anthropic's Constitutional AI (CAI) teaches models to evaluate and revise their own outputs:

```
The CAI loop (adapted for NanoSeek):

1. Generate: Model produces response to prompt
2. Critique: Model evaluates its own response against principles
   "Does this response contain harmful content?"
   "Is this response helpful and accurate?"
   "Does this response follow the user's instructions?"
3. Revise: Model produces improved response based on critique
4. Train: DPO on (original, revised) pairs — revised is "chosen"

Why this matters for NanoSeek specifically:
  - At 1B scale, the model is too small for a separate reward model to be reliable
  - Self-critique uses the SAME model — no reward model needed
  - It produces DPO training pairs for FREE (no human annotation)
  - For MoE: critique may route through different experts than generation
    → interpretability opportunity: which experts handle self-evaluation?

Implementation:
  1. Define 10-15 constitutional principles (from Anthropic's paper)
  2. Generate 10K prompt-response pairs
  3. Self-critique + revise each pair (2× inference cost)
  4. Train DPO on (original, revised) pairs
  Cost: ~$5 for 10K pairs at 1B scale
```

### Iterative DPO (2 Rounds) — $15

Standard DPO trains once on a fixed preference dataset. Iterative DPO improves
the preference data each round:

```
Round 1:
  1. Generate responses from current policy
  2. Score with PRM (or rule-based for math/code)
  3. Construct preference pairs (best vs worst per prompt)
  4. DPO training with β=0.1

Round 2:
  1. Generate NEW responses from Round 1 policy (better quality)
  2. Score again — higher quality pairs because policy improved
  3. DPO training with β=0.05 (lower β for fine-grained adjustment)

Why 2 rounds (not 1, not 5):
  - Round 1: large improvement from naive policy → preference-aligned
  - Round 2: modest improvement from aligned → well-calibrated
  - Round 3+: diminishing returns, risk of reward hacking
  - At 1B scale, 2 rounds is the sweet spot (DeepSeek R1 uses 2-3 rounds)
```

---

# PART 13: ENGINEERING SKILLS INVENTORY

## What Top-Tier Labs Test in Interviews

Based on OpenAI's interview guide and analysis of 2025-2026 hiring patterns:

### 1. PyTorch at Depth (Not Just High-Level APIs)

**What they test**: Custom autograd functions, memory management, CUDA kernel understanding

**What NanoSeek demonstrates**: Custom MLA attention (low-level matmuls), Muon optimizer
(Newton-Schulz iterations), MoE routing (scatter/gather operations)

**Gap**: No custom CUDA kernels. No Triton kernels. No torch.compile optimization.

**How to close (Tier 4)**: Write 2-3 custom Triton kernels:

```python
# 1. MoE routing scatter kernel (~100 lines)
#    Fused top-k selection + token permutation + expert dispatch
#    Why: The default PyTorch scatter is 3 separate kernel launches
#    (top-k, permute, scatter) with 3 GPU syncs. Fusing = ~1.5× speedup.

# 2. Fused SwiGLU activation kernel (~50 lines)
#    gate = silu(x @ W_gate) * (x @ W_up)
#    Why: 2 matmuls + silu + elementwise multiply = 4 kernel launches.
#    Fuse silu + multiply into one kernel = 1.3× speedup on the FFN.

# 3. MLA compressed KV cache kernel (~80 lines)
#    Fused latent-to-KV decompress + attention score computation
#    Why: MLA decompresses c_kv → K, V at every forward pass.
#    Fusing decompress + attention avoids materializing full K, V tensors.
```

These demonstrate kernel-level understanding of MoE + MLA computation patterns —
the exact skill that distinguishes "can implement papers" from "can optimize systems."

### 2. Distributed Training

**What they test**: FSDP vs DeepSpeed trade-offs, communication optimization, fault
tolerance

**What NanoSeek demonstrates**: DDP-aware training loop, rank-0 logging, gradient
accumulation

**Gap**: No FSDP. No expert parallelism. No pipeline parallelism. No fault tolerance.

**How to close**: Add FSDP2 support to the training loop. For a 1B model on 8×A100,
FSDP2 is the right choice (simpler than Megatron, sufficient for this scale). This
demonstrates you understand distributed training beyond "just add DDP."

```
Parallelism strategy for NanoSeek-1B (8×A100):

Option A (current): DDP only
  - Each GPU holds full model (~10GB) + optimizer states (~30GB)
  - Total memory: ~40GB per GPU (fits in 80GB A100)
  - Communication: all-reduce gradients only
  - ✅ Simple. Works for 1B.

Option B (recommended): FSDP2
  - Shard model + optimizer across 8 GPUs
  - Memory: ~5GB model + ~4GB optimizer per GPU = ~9GB
  - Leaves ~70GB for activations + KV cache
  - Enables larger batch sizes or longer sequences
  - Communication: all-gather params (forward) + reduce-scatter grads (backward)
  - ✅ More professional. Demonstrates FSDP knowledge.

Option C (NOW NEEDED for Tier 4): Megatron-Core with expert parallelism
  - Required for 3B-7B MoE models (Tier 4 scale validation)
  - Expert parallelism: distribute 64 experts across GPUs (8 experts/GPU on 8×A100)
  - Tensor parallelism: shard attention/MLP within each GPU group
  - At 7B active / ~30B total: FSDP2 alone runs OOM → need Megatron-Core
  - ✅ Demonstrates frontier-level distributed training knowledge

Option D (multi-node): FSDP2 + expert parallelism across nodes
  - For 3B+ models on 2+ nodes (16+ GPUs)
  - Inter-node communication: NCCL all-reduce over InfiniBand/RoCE
  - Expert parallelism is communication-friendly: each expert is self-contained
  - Pipeline parallelism NOT needed at 3B-7B (micro-batch overhead > benefit)
  - ✅ Multi-node experience is the #1 distinguishing skill for frontier labs
```

### 3. Experiment Design

**What they test**: Ablation methodology, statistical rigor, controlled variables

**What NanoSeek demonstrates**: Stability ablations (A/C/D), muP coordinate check,
HP grid search with selection criteria

**This is already strong.** The EXPERIMENT_REASONING.md document would impress
interviewers. The addition of data mixture ablations and evaluation benchmarks
makes it even stronger.

### 4. Paper Literacy

**What they test**: "Read this paper before the interview. Discuss its strengths,
weaknesses, and limitations."

**What NanoSeek demonstrates**: Deep analysis of DeepSeek V3, μP-MoE, CompleteP,
Allen-Zhu's Physics of Language Models. The PAPER_ANALYSIS_V3_V32.md document shows
paper-reading rigor.

**How to strengthen**: Add a LITERATURE_NOTES.md that documents your analysis of
each paper you built on. Not a summary — a critical analysis:

```markdown
## DeepSeek V3 (arXiv:2412.19437)
### What they got right
- Aux-loss-free load balancing: elegant control-theoretic approach
- MLA: 23× KV compression with minimal quality loss
### What's questionable
- They claim "no auxiliary loss" but still have a load-balance bias update
  that functions as implicit regularization — it's not truly "loss-free"
- MTP module described vaguely — the paper doesn't specify whether MTP
  uses shared or separate embedding weights
### What's missing
- No ablation on gamma (bias update rate) — how sensitive is it?
- No analysis of expert specialization patterns (H_load reported but not I_spec)
```

### 5. Mathematical Foundations

**What they test**: Derive a gradient, explain a loss function, analyze convergence

**What NanoSeek demonstrates**: muP scaling derivations, information-theoretic expert
metrics (H_load, I_spec), scaling law fitting

**This is strong if you can explain it verbally.** Practice deriving:
- Why muP says η ∝ 1/width for hidden weights (from Tensor Programs V §3)
- Why GRPO doesn't need a value function (advantage = group-relative reward)
- Why MoE has 8× gradient variance per expert (1/top_k tokens flow through each)

---

# PART 14: IMPLEMENTATION CHECKLIST — PRIORITY-TIERED

## Tier System: Must-Have → Value-Add → Target-Specific

A well-executed 1B MoE with MLA+MTP+GRPO, clean ablations, and solid evals
is already more than 95% of applicants show. Don't do everything at surface
level when you can do the core deeply.

```
═══════════════════════════════════════════════════════════════════════
▓▓▓ MUST HAVE — PHASE A: FOUNDATIONS (Week 1-3, ~$0) ▓▓▓
  Build infrastructure that all subsequent work depends on.
  Without this, nothing else matters.
═══════════════════════════════════════════════════════════════════════

A1. Model rewrite (Week 1-2, $0):
  □ config.py → RMSNorm → RoPE → MLA → Gate → MoE → MTP → DSA
  □ Canon layer module (ShortConvolution, 4 insertion points)
  □ All 145+ unit tests passing
  □ speculative_eval.py (MTP acceptance rate harness)

A2. Training infrastructure (Week 3, $0):
  □ ema_tracker.py (CPU-side, decay=0.9999)
  □ expert_specialization.py (H_load + I_spec logging)
  □ dataset.py (FIM 10% PSM format)
  □ pre_train.py (Muon+AdamW, batch warmup, EMA, muP scaling)
  □ Reproducibility: seed control, config serialization, experiment registry
  □ FSDP2 integration for training loop

A3. Data pipeline ($5-10, 1-2 days):
  □ heuristic_filters.py (length, repetition, special chars)
  □ quality_classifier.py (fastText on 50K pos/neg examples)
  □ dedup.py (MinHash LSH, threshold=0.8, 128 permutations)
  □ domain_classifier.py (web/code/math/books/multilingual)
  □ mixture.py (domain sampling weights, proxy model validation)
  □ Quality threshold tuning via 3 tiny proxy models (~$3)

A4. Evaluation framework ($0, 1-2 days):
  □ lm-evaluation-harness integration (wrap, don't reimplement)
  □ 7 benchmarks: BPB + ARC + MMLU + GSM8K + HumanEval + HellaSwag + MATH-500
  □ Per-domain BPB logging (web/code/math/books/multilingual)
  □ MoE diagnostic metrics (H_load, I_spec, router entropy per layer)
  □ Alert system for silent MoE failures

═══════════════════════════════════════════════════════════════════════
▓▓▓ MUST HAVE — PHASE B: ANCHOR EXPERIMENTS (Week 4-5, ~$50) ▓▓▓
  Without ablations, you have no research story.
═══════════════════════════════════════════════════════════════════════

B1. Anchor HP grid search (~$40, 1-2 days):
  □ 15-20 random samples from 3^4 grid (matrix_lr × embed_lr × unembed_lr × wd)
  □ Selection: lowest ema_val_bpb, filtered by H_load > 2 bits + no spikes
  □ Coordinate check: 2 widths (480, 960) for 500 steps ($5)

B2. Stability ablations (~$6, same day):
  □ Run A: traditional aux loss, no QK-norm
  □ Run C: aux-loss-free, no QK-norm
  □ Run D: aux-loss-free, QK-norm ON (= default)
  □ Bad batch injection at step 1500, recovery measurement
  □ A vs C: I_spec comparison → load balancing decision
  □ C vs D: spike recovery comparison → QK-norm decision

B3. Canon × MoE ablation (~$10, same day):
  □ Run 0: baseline (no Canon)
  □ Run 1: Canon-A only (pre-attention)
  □ Run 2: Canon-C only (pre-MoE) ← predicted winner
  □ Run 3: Canon-A + Canon-C
  □ Run 4: Canon-ABCD (all insertion points)
  □ Measure: ema_val_bpb, I_spec, router entropy per layer

═══════════════════════════════════════════════════════════════════════
▓▓▓ MUST HAVE — PHASE C: VALIDATION + 1B TRAINING (Week 6-9, ~$330) ▓▓▓
  Proves scaling behavior, not just small-scale tricks.
═══════════════════════════════════════════════════════════════════════

C1. nano-500M validation (~$30, 1-2 days):
  □ Auto-scaled HPs from anchor (muP: √B + 1/width + T_epoch)
  □ Full Chinchilla-optimal training (8.8B tokens)
  □ Pass Gate 2: converged, reasonable BPB, H_load > 2, MTP improving
  □ Canon-C validated at 500M scale (if winner at anchor)

C2. NanoSeek-1B training (~$300, 3-5 days):
  □ Auto-scaled HPs, 22B tokens (Phase 1: 4K, Phase 2: 8K+DSA)
  □ Full evaluation suite at 10%, 25%, 50%, 75%, 100%
  □ Phase 2 transition at 80%: DSA + YaRN + indexer warmup
  □ Pass Gate E: BPB within 2% of scaling prediction, MTP > 75%

C3. Scaling law fit (post-hoc, $0):
  □ L(N) = E + A/N^α from 3 points (55M, 441M, 1.08B)
  □ Two-stage prediction: loss → downstream performance

═══════════════════════════════════════════════════════════════════════
  ⬆ EVERYTHING ABOVE IS MUST-HAVE (~$100-150, 12-14 weeks)
  At this point you have ~80% frontier coverage.
  This is already more competitive than 95% of applicants.
═══════════════════════════════════════════════════════════════════════


═══════════════════════════════════════════════════════════════════════
▒▒▒ STRONG VALUE-ADD — PHASE D: POST-TRAINING (Week 10-13, ~$80) ▒▒▒
  Standard post-training — expected knowledge for research engineers.
  D1 is must-have. D2-D4 are strong but optional.
═══════════════════════════════════════════════════════════════════════

D1. GRPO implementation + training (~$80) ← MUST HAVE:
  □ GRPO trainer with group-relative advantages (K=8)
  □ Rule-based rewards: math (binary), code (pass@k), format (tag check)
  □ 4 V3.2 MoE stabilization techniques (unbiased KL, off-policy mask,
    Keep Routing, Keep Sampling Mask)
  □ 3-stage pipeline: Reasoning (60%) → Agent (25%) → General DPO (15%)
  □ 3 compute budgets: 2%, 5%, 10% of pre-training FLOPs
  □ Staging ablation: single-stage vs three-stage at Budget 2

D2. Iterative DPO (~$15) ← STRONG VALUE-ADD:
  □ Round 1: DPO with β=0.1 on rule-based preference pairs
  □ Measure: capability preservation (MMLU within 1% of pre-RL)
  □ (Round 2 optional — diminishing returns at 1B scale)

D3. Test-time compute scaling ($0, inference only) ← STRONG VALUE-ADD:
  □ Plot accuracy vs inference tokens (256/512/1024/2048)
  □ 5 strategies: greedy, best-of-N, CoT, self-consistency, sequential refinement
  □ Compare pre-RL vs post-RL curves
  □ MTP acceptance rate vs problem difficulty correlation

D4. Basic interpretability ($0, analysis only) ← STRONG VALUE-ADD:
  □ Expert specialization visualization (routing heatmaps per domain)
  □ I_spec tracking across training
  □ Qualitative expert analysis: what does each top-used expert compute?
  □ 1 custom Triton kernel (MoE routing scatter — ~100 lines)

═══════════════════════════════════════════════════════════════════════
  ⬆ WITH D1-D4: ~85% frontier coverage (~$180-230 total)
  High ROI. GRPO + basic interp + 1 Triton kernel is the sweet spot.
═══════════════════════════════════════════════════════════════════════


═══════════════════════════════════════════════════════════════════════
░░░ TARGET-SPECIFIC — CHERRY-PICK 2-3 BASED ON EMPLOYER ░░░
  Each item below is valuable but NOT all are needed.
  Pick based on where you're applying.
═══════════════════════════════════════════════════════════════════════

OPTION E: MECHANISTIC INTERPRETABILITY (2-3 weeks, ~$25)
  → Best for: Anthropic interpretability, DeepMind alignment
  □ Expert-level SAE training on 8 experts (per-expert, 32× expansion)
  □ Feature birth/death tracking across 6 GRPO checkpoints
  □ fTRI behavioral mapping (pre-RL vs post-RL)
  □ Alignment probes at layer 8 (helpful/harmful concepts)
  □ Probe transfer across GRPO stages

OPTION F1: PROCESS REWARD MODEL (~$15)
  → Best for: Anthropic safety, OpenAI reasoning
  □ Auto-annotate math solutions with step-level labels
  □ Train PRM (value head on NanoSeek-1B backbone)
  □ PRM-guided GRPO on MATH-500 (compare vs rule-based GRPO)

OPTION F2: CONSTITUTIONAL AI (~$5)
  → Best for: Anthropic safety specifically
  □ Define 10-15 constitutional principles
  □ Generate 10K prompt-response pairs
  □ Self-critique + revision loop → DPO on (original, revised) pairs
  □ Honest caveat: a toy CAI at 1B doesn't prove much depth

OPTION F3: WIDTH VS DEPTH PARETO + IsoFLOP (~$20)
  → Best for: Meta FAIR, Google DeepMind research
  □ 3 configs at ~3B active: wide/shallow, balanced, deep/narrow
  □ 2000 steps each, compare ema_val_bpb
  □ IsoFLOP: vary N_active and D at fixed compute
  □ MoE-specific: does κ affect the optimal D/N ratio?

OPTION F4: NanoSeek-3B (~$800, 2-3 days)
  → Best for: roles that specifically ask for scale experience
  □ Config from Pareto winner, 64B tokens
  □ FSDP2 sharding across 8 GPUs
  □ Validate scaling law prediction from 3 prior points
  □ Full benchmark suite comparison to published 3B models

OPTION F5: NanoSeek-7B (~$2000+, multi-node) ← PROBABLY CUT THIS
  → Only if: you have money to burn AND applying to infra-heavy roles
  → Why cut: $2000 buys almost nothing the 1B→3B scaling curve
    doesn't already show. A clean 1B with excellent ablations
    beats a sloppy 7B every time. Interviewers care about
    methodology, not that you spent $2000.
  □ 24 layers, 4096 hidden, 128 experts, top-16
  □ Multi-node distributed (2+ nodes, 16+ GPUs)
  □ Expert parallelism + tensor parallelism

OPTION F6: ADDITIONAL TRITON KERNELS (~$0, just engineering time)
  → Best for: systems-heavy roles (NVIDIA, xAI, OpenAI infra)
  □ Fused SwiGLU activation kernel (~50 lines)
  □ MLA compressed KV cache kernel (~80 lines)

═══════════════════════════════════════════════════════════════════════
PHASE G: REPORTS + PACKAGING (Week 14-16, $0)
  Applies regardless of which options above were chosen.
═══════════════════════════════════════════════════════════════════════

  □ HP_TRANSFER_REPORT.md (muP validated for MoE+MLA across 3+ scales)
  □ STABILITY_PLAYBOOK.md (aux-loss-free + QK-norm + Canon-C recommendations)
  □ RL_SCALING_REPORT.md (3-stage GRPO + test-time scaling)
  □ Model weights on HuggingFace (1B EMA checkpoint)
  □ W&B dashboards archive
  □ Full reproducibility package (configs, seeds, data pipeline scripts)
  □ + Any reports from target-specific options chosen above
```

## Frontier Alignment Score — Realistic Assessment

```
                                Weight    Before    Must-Have    +Extras
Architecture + Optimizer         12%       90%         92%        92%
HP Transfer Science              10%       85%         90%        90%
Data Curation                    15%       10%         85%        85%
Evaluation Framework             10%       20%         90%        90%
Scaling Laws                      5%        0%         85%        85%
Post-Training (GRPO/DPO)        12%       50%         85%        90%
Training Infrastructure           8%       60%         80%        90%
Mechanistic Interpretability      8%       25%         50%        85%
Research Novelty (Canon)          8%       30%         75%        85%
Scale Validation                  5%        0%          0%        80%
Safety & Alignment                4%        0%         30%        75%
Reproducibility + Rigor           3%       30%         90%        90%

Weighted Before:     ~42%
Must-Have (A-C+D1):  ~78-80%  ← already beats 95% of applicants
+Strong Value-Add:   ~85%     ← high ROI sweet spot ($180-230 total)
+Target-Specific:    ~88-93%  ← diminishing returns, pick wisely
```

The 80% → 85% jump is ~$50-80 of high-ROI work.
The 85% → 93% jump is ~$800-2000+ of low-ROI work.
Do the core deeply. Cherry-pick extras per employer.

---

# PART 15: WHAT DISTINGUISHES "RESEARCH ENGINEER" FROM "ML ENGINEER"

## The Core Difference

An ML engineer takes a known recipe and makes it work in production.
A research engineer designs the recipe from first principles.

### ML Engineer Portfolio:
- "I fine-tuned Llama 3 on our customer data using LoRA"
- "I deployed a RAG pipeline with 99.9% uptime"
- "I optimized inference latency from 200ms to 50ms"

### Research Engineer Portfolio (NanoSeek — Must-Have Core):
- "I trained a 1B-active MoE model from scratch that reproduces DeepSeek V3's
  architecture — MLA, sparse MoE with aux-loss-free routing, MTP, DSA, Canon layers.
  I validated muP HP transfer across 3 scale points and showed aux-loss-free load
  balancing outperforms traditional aux loss at small scale."
- "I built a data curation pipeline with quality classification, deduplication, and
  domain mixture optimization. I ran 5 controlled ablations at anchor scale including
  the first measurement of Canon-C × MoE routing interaction."
- "I implemented GRPO post-training with 4 MoE-specific stabilization techniques
  from DeepSeek V3.2, with test-time compute scaling curves showing how RL shifts
  the inference efficiency frontier."

### What NanoSeek Demonstrates (Core)

1. **Full pipeline ownership**: Data → architecture → training → evaluation → RL
2. **First-principles reasoning**: Every decision justified from theory (muP,
   information theory, scaling laws, Allen-Zhu's Physics of Language Models)
3. **Experimental rigor**: Controlled ablations, scaling law fitting,
   reproducibility infrastructure
4. **Novel findings** (at least 3 publishable results from core alone):
   - muP validated for MoE+MLA+Muon at 3 scale points
   - Canon-C × MoE routing interaction (first measurement)
   - MTP acceptance rate as test-time scaling signal
   - Stability ablations (aux-loss-free vs traditional) at small scale

### What Target-Specific Additions Demonstrate (If Chosen)

5. **Interpretability depth** (Option E): SAE feature evolution in MoE experts
   during GRPO — greenfield research territory
6. **Safety consciousness** (Options F1-F2): PRM step-level rewards, CAI
   self-critique methodology
7. **Scale experience** (Option F4): FSDP2 sharding, scaling law validation
   at 3B
8. **Systems depth** (Options D4, F6): Custom Triton kernels for MoE dispatch

---

## Closing: The Interview Pitch

### Must-Have Core (what you can always say):

> "I built NanoSeek, a 1B-active MoE model that reproduces DeepSeek V3's
> architecture from scratch — MLA, sparse MoE with aux-loss-free routing,
> multi-token prediction, dynamic sparse attention, and Canon layers. I validated
> hyperparameter transfer across 3 scale points using muP theory, built a data
> curation pipeline with quality classification and deduplication, and ran 5
> controlled ablations including the first measurement of Canon-C's effect on
> MoE expert routing.
>
> For post-training, I implemented GRPO with 4 MoE-specific stabilization
> techniques from DeepSeek V3.2, measured test-time compute scaling curves,
> and showed that MTP acceptance rate predicts inference efficiency. Everything
> is open-source with full experiment documentation and reproducibility
> infrastructure."

### Then add 1-2 sentences per employer:

> **+ Anthropic**: "I also applied SAE mechanistic interpretability to individual
> MoE experts, tracking feature birth and death during GRPO training, with
> alignment probes showing how the model's concept representations shift."
>
> **+ OpenAI/DeepMind**: "I also validated the architecture at 3B scale with
> FSDP2 sharding and wrote custom Triton kernels for fused MoE routing dispatch."
>
> **+ Meta FAIR**: "I also ran IsoFLOP analysis and width-vs-depth Pareto
> experiments to find the compute-optimal configuration for MoE+MLA."

That covers architecture, data, training, RL, and 3+ novel contributions.
At ~80-85% frontier alignment, that's what gets you in the door.
The interview itself tests whether you can explain WHY, not just WHAT.
