# NanoSeek Post-Training Pipeline
## 9-Stage, 2-Phase Research-Grade Post-Training for MoE Language Models
### March 2026 — Grounded in DeepSeek R1, Qwen3.5, Agentic RL Frontier, and 2025-2026 RL Literature

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Research Landscape & Motivation](#2-research-landscape--motivation)
3. [Pipeline Architecture Overview](#3-pipeline-architecture-overview)
4. [Stage 0 — Teacher Distillation SFT](#4-stage-0--teacher-distillation-sft)
5. [Stage 1 — Extended Reasoning SFT](#5-stage-1--extended-reasoning-sft)
6. [Stage 2 — Reasoning RLVR (GSPO/DAPO)](#6-stage-2--reasoning-rlvr-gspo-dapo)
7. [Stage 3 — Rejection Sampling + SFT Refinement](#7-stage-3--rejection-sampling--sft-refinement)
8. [Stage 4 — Thinking Mode Fusion](#8-stage-4--thinking-mode-fusion)
9. [Stage 5 — General Alignment (DPO/SimPO)](#9-stage-5--general-alignment-dposimpo)
10. [Cross-Stage Distillation (Phase 1)](#10-cross-stage-distillation-phase-1)
11. [Phase 2 — Agentic RL Extension](#11-phase-2--agentic-rl-extension)
12. [GSPO vs GRPO vs DAPO — Algorithm Selection](#12-gspo-vs-grpo-vs-dapo--algorithm-selection)
13. [MoE-Specific Post-Training Considerations](#13-moe-specific-post-training-considerations)
14. [Monitoring & Quality Gates](#14-monitoring--quality-gates)
15. [Data Recipes & Sources](#15-data-recipes--sources)
16. [Budget & Compute Estimates](#16-budget--compute-estimates)
17. [Implementation Roadmap](#17-implementation-roadmap)
18. [Updated Rule 8](#18-updated-rule-8)
19. [References](#19-references)
- [Appendix A: GSPO vs GRPO — Mathematical Comparison](#appendix-a-gspo-vs-grpo--mathematical-comparison)
- [Appendix B: Qwen3.5 Architecture Summary](#appendix-b-qwen35-architecture-summary-for-context)
- [Appendix C: Canonical 2026 Post-Training Pipeline](#appendix-c-canonical-2026-post-training-pipeline-industry-consensus)
- [Appendix D: Agentic RL Frameworks Comparison](#appendix-d-agentic-rl-frameworks-comparison)
- [Appendix E: Credit Assignment Methods Comparison](#appendix-e-credit-assignment-methods-comparison)

---

## 1. Executive Summary

This document defines the complete post-training pipeline for NanoSeek (1.08B active / 4.87B
total parameters, DeepSeek V3.2 architecture) after pre-training on 22B tokens. The pipeline
synthesizes lessons from frontier-lab post-training systems and the emerging agentic RL frontier:

- **DeepSeek R1** (Jan 2025): 4-stage pipeline (cold-start SFT → reasoning RL → rejection
  sampling → universal RL), rule-based verifiable rewards, GRPO
- **Qwen3** (May 2025): 4-stage pipeline (long-CoT cold-start → reasoning RL → thinking
  mode fusion → general RL), hybrid thinking/non-thinking modes
- **Qwen3.5** (Feb 2026): GSPO algorithm (sequence-level RL for MoE stability), 512
  fine-grained experts, 20K parallel agentic RL environments
- **Agentic RL frontier** (2025-2026): Kimi-Researcher (26.9% HLE), DeepSWE (42.2%
  SWE-bench from pure RL), Open-AgentRL (4B > 32B), ASearcher (ICLR 2026, 100+ tool calls)

**Two-release strategy**: The pipeline produces two ship targets:
1. **NanoSeek-Reason** (Phase 1, Stages 0-5): Reasoning + alignment, ships first
2. **NanoSeek-Agent** (Phase 2, Stages 6-8): Agentic tool use, ships from Phase 1 checkpoint

**NeurIPS 2025 insight** (OpenReview: 4OsgYD7em5): "Does RL Really Incentivize Reasoning
Beyond the Base Model?" found that RLVR redistributes probability mass toward latent reasoning
paths already present in the base model — it does NOT create genuinely new reasoning capabilities.
Distillation is where new capabilities enter. This validates Stage 0 as even more critical than
originally framed: it is the **capability creation** stage, while Stage 2 is **capability amplification**.

**Key departures from original RULE 8** (3-stage pipeline):

| Original Plan | Revised Plan | Why |
|---------------|-------------|-----|
| Single SFT warmup (2 epochs) | Teacher distill + extended SFT (5 + 10 epochs) | Extended SFT stabilizes downstream RL; distillation > direct RL for small models |
| GRPO (1000 steps) | GSPO/DAPO (50-100 steps) | GRPO token-level ratios are unstable for MoE; small models saturate at 50-100 steps |
| No rejection sampling | Rejection sampling from RL checkpoint | DeepSeek R1's most impactful innovation — filters RL artifacts |
| No thinking mode control | Thinking mode fusion | Qwen3's dual-mode capability (think vs direct) is strictly more useful |
| 3 stages | 9 stages in 2 phases + cross-stage distillation | Each stage has a distinct purpose; Phase 2 extends to agentic capabilities |
| No agentic RL | Phase 2 agentic extension (Stages 6-8) | Frontier shift from single-turn reasoning to multi-turn tool-using agency |

**Expected outcomes at NanoSeek-1B scale**:

*Phase 1 (NanoSeek-Reason)*:
- GSM8K: 40-55% (vs ~20% base), based on DeepSeek distillation results at 1.5B
- HumanEval: 30-45% (vs ~15% base)
- Dual-mode operation: `<think>` for complex problems, direct for simple ones
- MTP acceptance rate usable as test-time scaling signal

*Phase 2 (NanoSeek-Agent)*:
- Simple tool use: calculator, code execution, search, retrieval
- 3-10 turn agentic trajectories with ReAct format
- MoE expert specialization under agentic RL (novel research territory)

---

## 2. Research Landscape & Motivation

### 2.1 The Qwen3.5 Breakthrough: Why GRPO Fails for MoE

Qwen3.5's team (Feb 2026) discovered a fundamental instability in applying GRPO to MoE models.
The core problem:

```
GRPO computes importance sampling ratios at the TOKEN level:
  ρ_t = π_θ(a_t | s_t) / π_old(a_t | s_t)

In MoE, each token routes through different experts. After one gradient step,
~10% of expert assignments change. This means:
  - The same token, same context → different experts → different logits
  - Token-level ρ_t fluctuates wildly between iterations
  - Clipping at ρ ∈ [0.8, 1.2] catches ~90% of updates as "off-policy"
  - Training either doesn't converge or converges to a bad optimum
```

Qwen3.5 developed **GSPO (Group Sequence Policy Optimization)** to fix this:

```
GSPO computes importance sampling ratios at the SEQUENCE level:
  ρ_seq = exp(Σ_t log π_θ(a_t | s_t) - Σ_t log π_old(a_t | s_t))

Sequence-level ratios are stable because:
  - Individual token routing changes average out over the full sequence
  - The sequence-level probability is robust to expert activation volatility
  - Clipping operates on a stable signal → meaningful gradient updates
```

**Source**: Qwen GSPO paper (arXiv:2507.18071), Qwen3.5 model card (HuggingFace)

### 2.2 Small Model RL Saturation

DeepSeek R1 and subsequent reproduction attempts (Open-R1, Phi-4-Reasoning) consistently show:

```
Small models (1B-7B active) during RL:
  Steps 0-50:   Rapid improvement (accuracy rises 10-30%)
  Steps 50-100: Plateau (diminishing returns)
  Steps 100+:   Degradation (reward hacking, format collapse, expert drift)
```

**Empirical evidence**:
- DeepSeek: "Distillation outperformed applying RL directly to small models" (R1 paper §5)
- Phi-4-Reasoning: Only 90 GRPO steps boosted AIME by 10%+, using 72K math problems
- DAPO paper: Qwen2.5-1.5B peaks at 50-100 RL steps across all benchmarks
- Statistical instability: AIME24 scores vary by several % with seed changes alone at 1.5B

**Implication**: NanoSeek's original `total_steps=1000` is 10-20× too many for a 1B-active model.

### 2.3 Extended SFT Stabilizes RL

Recent research (late 2025) shows:

```
AIME 2024 accuracy vs SFT epochs (before RL):
  Epoch 1:  baseline
  Epoch 3:  +5% accuracy, but RL diverges 40% of the time
  Epoch 5:  +8% accuracy, RL diverges 15% of the time
  Epoch 10: +12% accuracy, RL diverges <5% of the time (plateau)
```

Short SFT (1-3 epochs) actually **increases solution length** and **destabilizes RL**.
Extended SFT creates a more robust starting point with consistent formatting and reasoning
patterns.

**Implication**: NanoSeek's original `SFTConfig.epochs=2` should be 8-10.

### 2.4 The DeepSeek R1 Pipeline: Gold Standard

DeepSeek R1's 4-stage pipeline remains the most validated post-training recipe:

```
Stage 1 — Cold-Start SFT:
  - "Thousands" of curated long CoT examples
  - Sources: few-shot prompting, R1-Zero outputs, human annotation
  - Format: |special_token|<reasoning>|special_token|<summary>

Stage 2 — Reasoning RL (GRPO):
  - Rule-based rewards ONLY (no neural reward models)
  - Accuracy: boxed-answer for math, compiler for code
  - Format: enforce <think>/<think> tags
  - Temperature 0.6, top-p 0.95, max 32K tokens

Stage 3 — Rejection Sampling + SFT:
  - Sample from RL checkpoint, retain correct outputs
  - Filter: remove mixed-language CoT, long paragraphs, code in reasoning
  - Dataset: ~600K reasoning + ~200K general = 800K total
  - RETRAIN FROM BASE (not continue from RL checkpoint)
  - Train 2 epochs

Stage 4 — Universal RL (Alignment):
  - RL across all task types
  - Reasoning: rule-based rewards
  - General: neural preference models
  - Helpfulness reward on summary only; harmlessness on full response
```

### 2.5 The Qwen3 Innovation: Thinking Mode Fusion

Qwen3's post-training introduced dynamic mode switching:

```
Stage 3 — Thinking Mode Fusion:
  - Fine-tune on combination of long CoT data + standard instruction data
  - Model learns TWO operational modes in a single model:
    - enable_thinking=True:  uses <think>...</think> for complex problems
    - enable_thinking=False: responds directly for simple queries
  - Dynamic switching via /think and /no_think tags
  - Performance scales smoothly with allocated reasoning compute budget
```

This is a practical and elegant solution. A single model replaces what would otherwise
require two separate models (reasoning + chat).

### 2.6 NeurIPS 2025 — RL as Capability Redistribution, Not Creation

**Paper**: "Does RL Really Incentivize Reasoning Beyond the Base Model?"
(NeurIPS 2025 Best Paper Runner-Up, OpenReview forum: 4OsgYD7em5)

**Core finding**: RLVR (RL with Verifiable Rewards) does NOT teach new reasoning patterns.
Instead, it **redistributes probability mass** — amplifying latent reasoning paths that already
exist in the base model's distribution but have low probability under the default sampling.

```
Base model:    P(correct reasoning path) = 0.02 (exists but rare)
After RLVR:   P(correct reasoning path) = 0.35 (amplified, not created)

Evidence:      All reasoning patterns found post-RL are detectable in base model
               via best-of-N sampling with large N
```

**Counterpoint**: 1-Shot RLVR (arXiv: 2505.00981) showed a single RL step can jump from
36% → 73.6% on MATH500, suggesting the redistribution effect is dramatic even if it
doesn't create fundamentally new capabilities.

**Implication for NanoSeek**: Stage 0 (Teacher Distillation) is the **capability creation**
stage — it imports reasoning patterns the base model doesn't have. Stage 2 (RLVR) is the
**capability amplification** stage — it makes those imported patterns reliable. This ordering
is not just convenient; it's theoretically necessary.

### 2.7 The Agentic RL Frontier

The frontier has shifted from single-turn reasoning to multi-turn tool-using agency.
Five systems demonstrate this capability stack:

| System | Scale | Method | Result | Key Innovation |
|--------|-------|--------|--------|----------------|
| DeepSWE (Cognition Labs, 2025) | 32B | Pure RL (no SFT) | 42.2% SWE-bench Verified | Tool masking, environment reward |
| Kimi-Researcher (Moonshot AI, 2025) | — | Multi-turn RL | 26.9% HLE (SOTA) | Long-horizon credit assignment |
| ASearcher (ICLR 2026) | 7B-72B | Agentic search RL | 100+ tool calls per query | Progressive horizon expansion |
| Open-AgentRL (2025) | 4B | Structured data + reward design | 4B > 32B on agent tasks | Small model agentic RL |
| OpenAI Deep Research (2025) | — | Multi-step web research | Production deployed | End-to-end agentic pipeline |

**Capability stack** — each layer requires the previous one:

```
Layer 3: Multi-step Planning
  ├── Plan across 3-10+ turns
  ├── Revise strategy based on observations
  └── Requires: Layer 2

Layer 2: Tool Use
  ├── Generate well-formed tool calls
  ├── Parse and integrate observations
  └── Requires: Layer 1

Layer 1: Reasoning
  ├── Chain-of-thought in <think> blocks
  ├── Verify intermediate steps
  └── Foundation (Phase 1 provides this)
```

**Key insight**: Reasoning is a prerequisite, not the endgame. The value proposition of
NanoSeek shifts from "small model that reasons" to "small model that reasons AND acts."

**RL2F** (Google, Feb 2026): Self-improving RL where the model generates its own language
feedback, then trains on it. Relevant for future NanoSeek iterations beyond Phase 2.

### 2.8 Small Model Agentic RL

Can 1B-4B models learn agentic behavior through RL? Recent evidence says yes, with caveats:

**Planner-R1** (arXiv: 2505.15966): Dense process rewards are 3.5× more compute-efficient
than sparse outcome rewards at 8B scale. For small models, per-turn credit is critical because
trajectory-level rewards dilute across turns — the model can't determine which action mattered.

```
Sparse reward (trajectory-level):   Model completes 8-turn task, gets reward 1.0
  → Each turn gets signal 1.0/8 ≈ 0.125
  → At 1B scale, this is too dilute to learn from

Dense reward (per-turn):            Model gets partial credit at each turn
  → Turn 3 (correct tool call): +0.3
  → Turn 5 (wrong observation parse): -0.1
  → 3.5× more compute-efficient (Planner-R1 finding)
```

**Open-AgentRL** (arXiv: 2503.11696): A 4B model outperforms 32B models on agent tasks
through structured training data and careful reward design. Key insight: **data quality and
reward shaping matter more than model scale** for agentic capabilities.

**ScalingInter-RL** (arXiv: 2504.12680): Progressive horizon expansion prevents reward
hacking and policy collapse in multi-turn RL:
- Start with 3-turn trajectories (easy credit assignment)
- Expand to 5 turns after convergence
- Expand to 10 turns for final training
- Each expansion re-initializes from the best checkpoint of the previous horizon

**Credit assignment is the critical bottleneck at 1B scale**. With 64 routed experts and
8-expert-per-token routing, the gradient signal must traverse both the policy network AND
the routing network. Dense per-turn rewards (Planner-R1) or progressive horizons
(ScalingInter-RL) are essential — pure trajectory-level RL will not converge.

---

## 3. Pipeline Architecture Overview

```
Pre-trained NanoSeek-1B (EMA weights, 22B tokens, ema_val_bpb converged)
    │
    │  Checkpoint: pretrain_ema_final.pt
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 0: Teacher Distillation SFT                      │
│  ─────────────────────────────────────────────────────── │
│  Input:   50K CoT samples from strong teacher models     │
│  Method:  Standard cross-entropy SFT                     │
│  Epochs:  5                                              │
│  LR:      2e-5 (cosine decay)                           │
│  Purpose: Transfer reasoning patterns from larger models │
│  Router:  UNFROZEN (adapts to reasoning distribution)    │
│  Output:  checkpoint/stage0_distill.pt                   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1: Extended Reasoning SFT                         │
│  ─────────────────────────────────────────────────────── │
│  Input:   10K diverse reasoning examples                 │
│  Method:  Cross-entropy SFT with <think> format          │
│  Epochs:  8-10 (extended for RL stability)               │
│  LR:      5e-6 (cosine decay)                           │
│  Purpose: Solidify <think> format + reasoning patterns   │
│  Router:  UNFROZEN                                       │
│  Output:  checkpoint/stage1_reasoning_sft.pt             │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2: Reasoning RLVR (GSPO or DAPO)                 │
│  ─────────────────────────────────────────────────────── │
│  Input:   Math/code/logic prompts with verifiable answers │
│  Method:  GSPO (sequence-level RL) or DAPO (no KL)      │
│  Steps:   50-100 (small model saturation point)          │
│  LR:      1e-6                                           │
│  Rewards: Rule-based only (math, code, format)           │
│  Router:  FROZEN (Keep Routing — Technique 3)            │
│  MoE:     All 4 V3.2 stabilization techniques active     │
│  Output:  checkpoint/stage2_rlvr.pt                      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 3: Rejection Sampling + SFT Refinement            │
│  ─────────────────────────────────────────────────────── │
│  Input:   Stage 2 RL model generates 8 completions/prompt │
│  Filter:  Keep correct answers, remove artifacts          │
│  Base:    RETRAIN FROM PRETRAINED BASE (not RL checkpoint)│
│  Data:    ~5K filtered reasoning + ~2K general = 7K      │
│  Epochs:  2                                              │
│  LR:      1e-5                                           │
│  Purpose: Reasoning quality without RL artifacts          │
│  Router:  UNFROZEN                                       │
│  Output:  checkpoint/stage3_rejection_sft.pt             │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 4: Thinking Mode Fusion                           │
│  ─────────────────────────────────────────────────────── │
│  Input:   60% thinking data (with <think>) +             │
│           40% non-thinking data (direct answers)         │
│  Method:  Mixed SFT                                      │
│  Epochs:  3                                              │
│  LR:      5e-6 (low — preserve reasoning capability)     │
│  Purpose: Single model supports both think/no-think modes │
│  Router:  UNFROZEN                                       │
│  Output:  checkpoint/stage4_thinking_fusion.pt           │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 5: General Alignment (DPO or SimPO)               │
│  ─────────────────────────────────────────────────────── │
│  Input:   2K-5K preference pairs (helpful, safe, format) │
│  Method:  DPO (primary) or SimPO (if memory-constrained) │
│  Steps:   200                                            │
│  LR:      5e-7                                           │
│  Beta:    0.1 (DPO) or implicit (SimPO)                  │
│  Router:  FROZEN (Keep Routing)                          │
│  Output:  checkpoint/stage5_aligned.pt                   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 1 Cross-Stage Distillation                        │
│  ─────────────────────────────────────────────────────── │
│  Input:   Mixed data from Stages 0-5                     │
│  Method:  SFT on combined dataset                        │
│  Steps:   500                                            │
│  Purpose: Prevent catastrophic forgetting across stages  │
│  Output:  checkpoint/nanoseek_reason.pt                  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
═══════════ Phase 1 Complete: NanoSeek-Reason ships here ═══════════
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 6: Tool Format SFT                                │
│  ─────────────────────────────────────────────────────── │
│  Input:   2K ReAct examples (4 tool types)               │
│  Method:  Cross-entropy SFT on ReAct format              │
│  Epochs:  3                                              │
│  LR:      5e-6                                           │
│  Base:    Phase 1 cross-distill checkpoint                │
│  Router:  UNFROZEN (adapts to tool-use distribution)     │
│  Output:  checkpoint/stage6_tool_sft.pt                  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 7: Agentic RL                                     │
│  ─────────────────────────────────────────────────────── │
│  Input:   Agent task prompts (math calc, code debug, QA) │
│  Method:  GRPO + token masking (mask observation tokens) │
│  Steps:   50 (progressive: 3→5→10 turn horizons)        │
│  LR:      1e-6                                           │
│  Rewards: Outcome + format + efficiency + dense progress │
│  Router:  FROZEN (Keep Routing — Technique 3)            │
│  MoE:     All 4 V3.2 stabilization techniques active     │
│  Output:  checkpoint/stage7_agentic_rl.pt                │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 8: Agent Rejection Sampling                       │
│  ─────────────────────────────────────────────────────── │
│  Input:   Stage 7 model generates 8 trajectories/task    │
│  Filter:  Task completion + format + trajectory efficiency│
│  Base:    Phase 1 cross-distill checkpoint (NOT Stage 7) │
│  Epochs:  2                                              │
│  Router:  UNFROZEN                                       │
│  Output:  checkpoint/stage8_agent_rejection.pt           │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 2 Cross-Stage Distillation                        │
│  ─────────────────────────────────────────────────────── │
│  Input:   Mixed data from Stages 0-8                     │
│  Method:  SFT on combined dataset                        │
│  Steps:   300                                            │
│  Purpose: Consolidate agentic + reasoning capabilities   │
│  Output:  checkpoint/nanoseek_agent.pt                   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
        NanoSeek-Agent (Phase 2 Release)
```

---

## 4. Stage 0 — Teacher Distillation SFT

### 4.1 Motivation

DeepSeek R1 conclusively demonstrated that **distillation from a strong teacher consistently
outperforms direct RL** on small models (1.5B-14B):

```
DeepSeek-R1-Distill-Qwen-1.5B:
  - Trained on 800K CoT samples from R1-671B
  - AIME 2024: 28.3% (vs o1-preview 44.6%, R1-Zero 71.0%)
  - At $42 compute + 7K RL examples → surpassed o1-preview
  - Distillation alone captured ~70% of teacher performance
```

At 1B active parameters, the model cannot discover complex reasoning strategies through RL
alone — the search space is too large for the model's capacity. But it CAN learn to imitate
reasoning patterns demonstrated by a stronger model.

**Key insight from CoT distillation research**: "The overall structure of long CoT matters
more than individual step content." With even ~200 high-quality CoT samples (Merge-of-Thought),
small models can acquire reasoning capability that transfers to unseen problems.

### 4.2 Data Recipe

| Domain | Count | Source | Format |
|--------|-------|--------|--------|
| Math reasoning | 30,000 | GSM8K, MATH, competition math via R1 outputs | `<think>step-by-step</think>\n\n\\boxed{answer}` |
| Code reasoning | 10,000 | HumanEval, MBPP, LeetCode via R1/Qwen3 outputs | `<think>approach + plan</think>\n\n```python\ncode\n```\n` |
| Logic/reasoning | 5,000 | ARC-Challenge, BBH, LogiQA via teacher outputs | `<think>logical chain</think>\n\nanswer` |
| General instruction | 5,000 | Diverse instruction-following via teacher outputs | `<think>reasoning</think>\n\nresponse` |
| **Total** | **50,000** | | |

**Data sources (all open, zero API cost)**:
- DeepSeek-R1 distillation dataset (800K available on HuggingFace, subsample 50K)
- Open-R1 dataset (HuggingFace community reproduction)
- Qwen3-32B generated CoT (run locally or use existing community outputs)
- NuminaMath-CoT (high-quality math CoT dataset)
- CodeContests (competitive programming with solutions)

### 4.3 Data Quality Filters

```python
# Mandatory filters before including any sample:

def filter_distill_sample(sample: dict) -> bool:
    """Filter criteria for teacher distillation data."""

    # 1. Correctness: Final answer must be verifiable and correct
    if not verify_answer(sample['response'], sample['ground_truth']):
        return False

    # 2. Length bounds: 200-4000 tokens (at 1B scale, cap CoT length)
    token_count = len(tokenizer.encode(sample['response']))
    if token_count < 200 or token_count > 4000:
        return False

    # 3. Language consistency: >95% English (no mixed-language CoT)
    if english_fraction(sample['response']) < 0.95:
        return False

    # 4. Format compliance: Must have <think>...</think> structure
    if '<think>' not in sample['response'] or '</think>' not in sample['response']:
        return False

    # 5. No degenerate patterns: repetition, lorem ipsum, etc.
    if has_repetition(sample['response'], threshold=3):
        return False

    # 6. Step structure: CoT should have discernible reasoning steps
    think_content = extract_think_content(sample['response'])
    if count_reasoning_steps(think_content) < 2:
        return False

    return True
```

### 4.4 Training Configuration

```python
@dataclass
class DistillSFTConfig:
    """Stage 0: Teacher Distillation SFT."""
    lr: float = 2e-5                 # Higher than later SFT (fresh learning)
    epochs: int = 5                   # 5 passes over 50K samples
    batch_size: int = 2               # Small batch for long sequences
    max_length: int = 2048            # Cap at 2K tokens (1B memory constraint)
    warmup_steps: int = 100           # 100 steps warmup
    weight_decay: float = 0.01        # Standard AdamW decay
    gradient_accumulation: int = 8    # Effective batch = 16
    num_examples: int = 50_000        # 50K curated samples
    save_dir: str = "checkpoints/nanoseek_stage0_distill"

    # Optimizer
    betas: tuple = (0.9, 0.95)       # Match pre-training optimizer

    # Monitoring
    eval_every: int = 500             # Eval every 500 steps
    log_every: int = 50               # Log metrics every 50 steps

    # Quality gates
    max_val_loss_increase: float = 0.05  # Early stop if val loss rises 2 epochs
```

### 4.5 Training Protocol

```
1. Load pre-trained NanoSeek-1B (EMA weights — RULE 1)
2. Initialize optimizer (AdamW, lr=2e-5, betas=(0.9, 0.95))
3. Router: UNFROZEN (must adapt to reasoning data distribution)
4. γ load balancing: ACTIVE at 0.001 (router needs to rebalance for new distribution)
5. Train for 5 epochs on 50K samples
6. Monitor:
   a. val_loss every 500 steps (early stop if rises for 2 consecutive epochs)
   b. H_load every eval step (must stay > 2 bits — RULE 7)
   c. I_spec every eval step (should increase as experts specialize on reasoning domains)
   d. Format compliance: sample 100 generations, check <think> structure
   e. GSM8K accuracy (sample 200 problems, greedy decode) at each epoch end
7. Save best checkpoint by val_loss
```

### 4.6 Expected Outcomes

| Metric | Before Stage 0 | After Stage 0 | Basis |
|--------|----------------|---------------|-------|
| GSM8K (greedy) | ~15-20% | ~30-40% | DeepSeek distill-1.5B gets 28% with 800K samples |
| Format compliance | ~0% | >90% | Model learns `<think>` structure |
| H_load | >2 bits | >2 bits | Must maintain |
| I_spec | baseline | +0.1-0.3 nats | Experts start specializing on math/code/logic |
| MTP acceptance | 50% (init) | 60-70% | Model becomes more predictable in reasoning |

---

## 5. Stage 1 — Extended Reasoning SFT

### 5.1 Motivation

Stage 0 teaches reasoning patterns via imitation. Stage 1 **solidifies** these patterns
with extended training on diverse, high-quality reasoning data. The critical insight:

> Short SFT (1-3 epochs) creates a fragile foundation. RL applied to a fragile foundation
> causes format collapse, reward hacking, and expert routing instability. Extended SFT
> (8-10 epochs) creates a robust foundation where RL can safely explore.

### 5.2 Data Recipe

| Domain | Count | Source | Difficulty |
|--------|-------|--------|------------|
| GSM8K | 3,000 | Original train split, reformatted | Easy-Medium |
| MATH | 3,000 | Level 3-5 problems | Medium-Hard |
| HumanEval/MBPP | 1,500 | Code with reasoning traces | Medium |
| ARC-Challenge | 1,000 | Science reasoning | Medium |
| BBH (selected) | 1,000 | Logical deduction, date understanding | Hard |
| Custom synthetic | 500 | Generated edge cases for format diversity | Mixed |
| **Total** | **10,000** | | |

**Key difference from Stage 0**: Stage 0 uses teacher outputs. Stage 1 uses **ground-truth
reformatted data** with shorter, more diverse reasoning traces. This prevents the model from
overfitting to the teacher's specific reasoning style.

### 5.3 Format Specification

```
Question: {question}

<think>
{step-by-step reasoning}
</think>

The answer is \boxed{answer}
```

For code:
```
Question: {problem_description}

<think>
{approach planning and edge case analysis}
</think>

```python
{solution_code}
```

```

### 5.4 Training Configuration

```python
@dataclass
class ExtendedReasoningSFTConfig:
    """Stage 1: Extended Reasoning SFT."""
    lr: float = 5e-6                  # Lower than Stage 0 (refinement, not learning)
    epochs: int = 10                   # Extended — literature shows plateau at epoch 10
    batch_size: int = 4                # Moderate batch
    max_length: int = 2048             # Consistent with Stage 0
    warmup_steps: int = 50             # Short warmup (already initialized from Stage 0)
    weight_decay: float = 0.01
    gradient_accumulation: int = 4     # Effective batch = 16
    num_examples: int = 10_000
    save_dir: str = "checkpoints/nanoseek_stage1_reasoning_sft"

    # Early stopping
    patience: int = 2                  # Stop if val_loss increases 2 consecutive epochs
    min_epochs: int = 6                # But always train at least 6 epochs
```

### 5.5 Training Protocol

```
1. Load Stage 0 checkpoint (stage0_distill.pt)
2. Router: UNFROZEN (continue adapting)
3. γ load balancing: ACTIVE at 0.001
4. Train for 8-10 epochs on 10K samples (stop at patience or 10 epochs)
5. Monitor (same as Stage 0 plus):
   a. Per-domain accuracy: math, code, logic separately
   b. CoT length distribution: should stabilize, not grow unboundedly
   c. Step-count in reasoning: track avg number of reasoning steps
6. Save best checkpoint by val_loss AND per-domain accuracy composite
```

### 5.6 Why Not Merge Stages 0 and 1?

They serve different purposes:

| | Stage 0 (Distillation) | Stage 1 (Extended SFT) |
|---|---|---|
| **Data** | Teacher CoT (imitation) | Ground-truth reformatted (diverse) |
| **LR** | 2e-5 (learning new patterns) | 5e-6 (solidifying) |
| **Goal** | Acquire reasoning capability | Stabilize for downstream RL |
| **Risk if skipped** | Model can't reason | RL destabilizes format |

Merging them would require balancing two conflicting objectives (learning vs stabilizing)
with a single LR schedule. Keeping them separate is cleaner and more debuggable.

---

## 6. Stage 2 — Reasoning RLVR (GSPO/DAPO)

### 6.1 Algorithm Selection

This is the most critical architectural decision in the pipeline. Three candidates:

#### Option A: GSPO (Group Sequence Policy Optimization) — RECOMMENDED

**Provenance**: Developed by Qwen team specifically for MoE RL stability (arXiv:2507.18071).
Production-validated at Qwen3.5-397B-A17B (512 experts, 10 routed per token).

**Core mechanism**:
```python
def gspo_loss(
    cur_log_probs: torch.Tensor,    # [B*G, T] per-token log-probs
    old_log_probs: torch.Tensor,    # [B*G, T] per-token log-probs (frozen)
    advantages: torch.Tensor,        # [B*G] group-relative advantages
    clip_eps: float = 0.2,
    prompt_len: int = 0,
) -> torch.Tensor:
    """
    GSPO: sequence-level clipping for MoE stability.

    Key insight: token-level importance ratios are unstable in MoE because
    ~10% of expert assignments change per gradient step. Sequence-level
    ratios average out this volatility.
    """
    # Sequence-level log-probs (sum over completion tokens)
    cur_seq_lp = cur_log_probs[:, prompt_len:].sum(dim=-1)   # [B*G]
    old_seq_lp = old_log_probs[:, prompt_len:].sum(dim=-1)   # [B*G]

    # Sequence-level importance ratio
    seq_log_ratio = cur_seq_lp - old_seq_lp
    seq_ratio = seq_log_ratio.exp()  # ρ_seq = π_θ(τ) / π_old(τ)

    # PPO clipped objective at SEQUENCE level
    surr1 = seq_ratio * advantages
    surr2 = torch.clamp(seq_ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = -torch.min(surr1, surr2).mean()

    return loss
```

**Why GSPO for NanoSeek**:
1. NanoSeek has 64 routed experts (substantial MoE) → token-level instability is real
2. Your existing `grpo_trainer.py` already computes `cur_seq_lp` and `old_seq_lp` → minimal code change
3. Portfolio differentiator: "Implemented GSPO for nano-scale MoE" is a strong signal
4. Eliminates need for DeepSeek's "Routing Replay" workaround

#### Option B: DAPO (Decoupled Clip and Dynamic Sampling) — FALLBACK

**Provenance**: ByteDance, validated on Qwen2.5-32B and Qwen2.5-1.5B.

Four modifications to GRPO:
1. **Clip-Higher**: upper clip = 0.28, lower = 0.2 (asymmetric prevents entropy collapse)
2. **Token-level loss**: average across ALL tokens (not per-response mean)
3. **Dynamic sampling**: oversample then filter flat-reward batches
4. **Overlong handling**: mask truncated samples + soft length penalty

**DAPO removes KL penalty entirely** → no reference model needed → 50% memory savings.

**When to use DAPO instead of GSPO**:
- If GSPO doesn't converge at NanoSeek's 1B scale (unlikely but possible)
- If memory is critically constrained (DAPO needs no reference model)
- As an ablation comparison (GSPO vs DAPO for nano-MoE)

#### Option C: Standard GRPO — NOT RECOMMENDED

The current `grpo_trainer.py` implements standard GRPO. While functional, it has the
token-level instability problem for MoE. The 4 V3.2 stabilization techniques mitigate
but don't eliminate this. Use as a baseline for comparison only.

### 6.2 Reward Design

**Principle**: At 1B scale, use **rule-based verifiable rewards only**. Neural reward models
are susceptible to reward hacking at small scale (confirmed by DeepSeek R1 §4.1).

```python
# Reward functions (extend existing rewards.py)

def reasoning_reward(
    response: str,
    ground_truth: str,
    reward_type: str = 'math',
    test_cases: list[dict] | None = None,
) -> float:
    """
    Combined reward for Stage 2 RLVR.

    Components:
      - Accuracy:  0.0 or 1.0 (binary — correct or not)
      - Format:    0.0-0.3 (has <think>, proper structure, step indicators)
      - Language:   0.0-0.1 (>95% English in CoT)
      - Length:    -0.1 to 0.0 (penalty for excessive length, following Dr. GRPO)

    Total range: -0.1 to 1.4
    """
    score = 0.0

    # Accuracy (binary, highest weight)
    if reward_type == 'math':
        if math_reward(response, ground_truth) >= 1.0:
            score += 1.0
    elif reward_type == 'code' and test_cases:
        score += code_reward(response, test_cases)

    # Format compliance
    score += format_reward(response)  # 0.0-0.3 (existing function)

    # Language consistency
    think_content = extract_think_content(response)
    if think_content:
        eng_frac = english_fraction(think_content)
        score += 0.1 * min(eng_frac / 0.95, 1.0)

    # Length penalty (Dr. GRPO insight: remove length bias)
    total_tokens = len(response.split())
    if total_tokens > 1000:  # Excessive for 1B model
        score -= 0.1 * min((total_tokens - 1000) / 1000, 1.0)

    return score
```

### 6.3 Training Configuration

```python
@dataclass
class RLVRConfig:
    """Stage 2: Reasoning RLVR (GSPO or DAPO)."""

    # Algorithm
    algorithm: str = 'gspo'           # 'gspo', 'dapo', or 'grpo' (baseline)

    # Generation
    group_size: int = 8               # G completions per prompt (4-8 for small models)
    max_gen_len: int = 1024           # Max completion length
    temperature: float = 0.7          # Sampling temperature

    # GSPO/GRPO specific
    clip_eps: float = 0.2             # PPO clip (GSPO: sequence-level)
    clip_eps_upper: float = 0.28      # DAPO only: asymmetric upper clip
    kl_beta: float = 0.01             # KL coefficient (0 for DAPO)

    # Off-policy masking (V3.2 Technique 2)
    off_policy_delta: float = 0.3     # Mask if |ρ - 1| > delta

    # Optimization
    lr: float = 1e-6                  # Low LR for RL stability
    weight_decay: float = 0.01
    warmup_steps: int = 10            # Very short warmup
    total_steps: int = 75             # 50-100 range (small model saturation)
    gradient_accumulation: int = 1
    max_grad_norm: float = 1.0

    # MoE stabilization (all 4 V3.2 techniques)
    freeze_router: bool = True        # Technique 3: Keep Routing
    use_unbiased_kl: bool = True      # Technique 1: Unbiased KL estimator
    off_policy_masking: bool = True   # Technique 2: Off-policy masking
    keep_sampling_mask: bool = True   # Technique 4: Keep Sampling Mask

    # Load balancing during RL
    gamma: float = 0.0                # DISABLE γ bias update during RL
                                      # (router is frozen, bias update is interference)

    # Prompt configuration
    prompts_per_step: int = 8         # B prompts per step
    reward_type: str = 'math'         # Start with math, extend to code+logic

    # Eval
    eval_every: int = 10              # Eval every 10 steps (frequent for short run)
    save_dir: str = "checkpoints/nanoseek_stage2_rlvr"

    # Domain allocation (within total_steps)
    math_steps: int = 40              # 40 steps on math
    code_steps: int = 20              # 20 steps on code
    logic_steps: int = 10             # 10 steps on logic
    # Total: 70 steps (remaining 5 as buffer)
```

### 6.4 Step Budget Allocation

```
Step  0-40:  Math RLVR (GSM8K + MATH prompts, boxed-answer verification)
Step 40-60:  Code RLVR (HumanEval + MBPP prompts, execution-based verification)
Step 60-70:  Logic RLVR (ARC-Challenge, BBH prompts, exact-match verification)
Step 70-75:  Mixed domain (randomly sample from all three)
```

### 6.5 Monitoring Protocol

```
Every step:
  - reward_mean, reward_std
  - policy_loss, kl_loss (if using KL)
  - on_policy_fraction (should be >50%, if <30% → instability)
  - sequence-level ratio mean and std (GSPO diagnostic)

Every 10 steps:
  - H_load across all layers (MUST stay > 2 bits — RULE 7)
  - I_spec (expert specialization MI)
  - GSM8K accuracy (sample 100 problems)
  - HumanEval pass@1 (sample 50 problems)
  - MTP acceptance rate
  - Average CoT length (should not grow unboundedly)

Stop conditions:
  - H_load drops below 1.5 bits → expert collapse, stop immediately
  - reward_mean plateaus for 20 consecutive steps → saturation reached
  - on_policy_fraction < 10% for 5 consecutive steps → instability
  - val_loss diverges (increases by >0.5 from minimum) → overfitting/hacking
```

### 6.6 GSPO vs GRPO Ablation

Run both algorithms at the same budget (75 steps) and compare:

| Metric | GSPO | GRPO (baseline) |
|--------|------|-----------------|
| GSM8K accuracy | ? | ? |
| H_load stability (std across training) | ? | ? |
| on_policy_fraction mean | Expected: >60% | Expected: 30-50% |
| Expert activation churn (% routing changes/step) | Expected: <3% | Expected: ~10% |
| Convergence speed (steps to peak accuracy) | Expected: faster | Expected: slower |

This ablation is a **publishable finding** if GSPO shows clear advantages at nano-MoE scale.

---

## 7. Stage 3 — Rejection Sampling + SFT Refinement

### 7.1 Motivation

This is DeepSeek R1's most underrated innovation. After RL, the model has acquired improved
reasoning but also picked up artifacts:
- Reward hacking patterns (format gaming, length exploitation)
- Mixed-language CoT fragments
- Code blocks appearing in reasoning sections
- Repetitive reasoning patterns

**Rejection sampling extracts the signal (improved reasoning) and discards the noise (artifacts)**
by retraining the pre-trained base on the RL model's best outputs.

### 7.2 Rejection Sampling Process

```python
def rejection_sample(
    rl_model: nn.Module,           # Stage 2 checkpoint
    prompts: list[str],            # Diverse evaluation prompts
    ground_truths: list[str],      # Verifiable answers
    n_samples: int = 8,            # Completions per prompt
    temperature: float = 0.7,
    max_length: int = 2048,
) -> list[dict]:
    """
    Generate multiple completions per prompt, keep only the best.

    Filtering criteria:
    1. Correctness: final answer matches ground truth
    2. Language: >95% English in CoT
    3. Format: proper <think>...</think> structure
    4. Length: 200-2000 tokens (not too short, not too long)
    5. Quality: no repetition, no code in reasoning, clear step structure
    """
    accepted = []

    for prompt, gt in zip(prompts, ground_truths):
        candidates = generate_n(rl_model, prompt, n=n_samples, temp=temperature)
        best = None
        best_score = -1

        for candidate in candidates:
            # Correctness check (mandatory)
            if not is_correct(candidate, gt):
                continue

            # Quality score
            score = 0.0
            score += format_quality(candidate)       # 0.0-0.3
            score += language_consistency(candidate)  # 0.0-0.1
            score += step_clarity(candidate)          # 0.0-0.2
            score -= length_penalty(candidate)        # 0.0-0.1

            if score > best_score:
                best = candidate
                best_score = score

        if best is not None:
            accepted.append({
                'prompt': prompt,
                'response': best,
                'ground_truth': gt,
                'quality_score': best_score,
            })

    return accepted
```

### 7.3 Data Composition

```
Rejection-sampled reasoning data:
  - 2,000 math (from GSM8K + MATH prompts)
  - 1,500 code (from HumanEval + MBPP prompts)
  - 1,000 logic (from ARC + BBH prompts)
  - 500 mixed (from diverse reasoning prompts)
  Subtotal: 5,000 reasoning examples

General-purpose data (reused from pre-training / instruction data):
  - 800 writing (creative writing, summarization)
  - 500 factual QA (general knowledge)
  - 300 translation (short passages)
  - 200 self-cognition (identity, capabilities, limitations)
  - 200 safety (refusal patterns for harmful requests)
  Subtotal: 2,000 general examples

Total: 7,000 examples
```

### 7.4 Critical Design Choice: Retrain from Base

```
                          ┌── RL model (Stage 2) has artifacts
                          │
                          ▼
  Pre-trained base ──► [SFT on RL model's BEST outputs] ──► Stage 3 checkpoint
        ▲                                                        │
        │                                                        │
        └── Uses PRETRAINED weights, not RL weights ─────────────┘

WHY: The RL checkpoint has good reasoning BUT also reward hacking artifacts,
     expert routing perturbations, and format noise. By SFT-ing the pristine
     pre-trained base on only the correct, high-quality RL outputs, we get:
     - Reasoning quality (from the RL model's best outputs)
     - Clean expert routing (from the pre-trained base's stable routing)
     - No reward hacking artifacts

ALTERNATIVE: Continue from RL checkpoint. This is simpler but preserves artifacts.
             DeepSeek explicitly chose to retrain from base.
```

### 7.5 Training Configuration

```python
@dataclass
class RejectionSFTConfig:
    """Stage 3: Rejection Sampling + SFT Refinement."""
    lr: float = 1e-5                  # Standard SFT rate
    epochs: int = 2                    # Short — this is refinement
    batch_size: int = 4
    max_length: int = 2048
    warmup_steps: int = 30
    weight_decay: float = 0.01
    gradient_accumulation: int = 4     # Effective batch = 16
    num_reasoning: int = 5_000         # Rejection-sampled reasoning examples
    num_general: int = 2_000           # General-purpose examples
    save_dir: str = "checkpoints/nanoseek_stage3_rejection_sft"

    # Rejection sampling parameters
    n_samples_per_prompt: int = 8      # Generate 8 completions per prompt
    min_quality_score: float = 0.3     # Minimum quality threshold
    sampling_temperature: float = 0.7
```

---

## 8. Stage 4 — Thinking Mode Fusion

### 8.1 Motivation

A model that always produces `<think>` blocks is suboptimal:
- Simple questions ("What is 2+2?") don't need reasoning traces
- Users want fast responses for factual queries
- Token budget is wasted on trivial reasoning

Qwen3's solution: train the model to support **both modes** in a single model, controlled
by the user at inference time.

### 8.2 Data Recipe

```
Thinking data (60%):
  - 3,000 examples from Stage 3 rejection samples (complex reasoning)
  - 2,000 examples from Stage 1 reasoning data (diverse domains)
  - 1,000 new complex examples requiring multi-step reasoning
  Subtotal: 6,000 examples with <think>...</think> traces

Non-thinking data (40%):
  - 2,000 factual QA (direct answers, no reasoning)
  - 1,000 simple instructions (formatting, translation, summarization)
  - 500 chitchat / conversational (greetings, opinions, recommendations)
  - 500 simple math/code (one-step problems with direct answers)
  Subtotal: 4,000 examples WITHOUT <think> traces

Mode-switching examples (included in above counts):
  - 500 multi-turn conversations that switch between think/no-think modes
  - Use /think and /no_think control tokens
  - Model must follow the MOST RECENT instruction

Total: 10,000 examples
```

### 8.3 Format Examples

**Thinking mode** (complex question):
```
User: Find all prime numbers between 100 and 120.

<think>
I need to check each number in [101, 120] for primality.
A number n is prime if it has no divisors other than 1 and n.
For numbers up to 120, I only need to check divisors up to √120 ≈ 10.9,
so I check divisibility by 2, 3, 5, 7.

101: not ÷2, not ÷3, not ÷5, not ÷7 → prime
103: not ÷2, not ÷3, not ÷5, not ÷7 → prime
107: not ÷2, not ÷3, not ÷5, not ÷7 → prime
109: not ÷2, not ÷3, not ÷5, not ÷7 → prime
113: not ÷2, not ÷3, not ÷5, not ÷7 → prime
</think>

The prime numbers between 100 and 120 are: **101, 103, 107, 109, 113**.
```

**Non-thinking mode** (simple question):
```
User: What is the capital of France?

The capital of France is Paris.
```

**Mode switching** (mid-conversation):
```
User: /no_think What is Python?
Assistant: Python is a high-level, interpreted programming language known for its
readability and versatility.

User: /think Now explain how Python's GIL works and its implications for concurrency.
Assistant: <think>
The GIL (Global Interpreter Lock) is a mutex in CPython that...
[detailed reasoning]
</think>

The GIL is a mutex in CPython that allows only one thread to execute Python
bytecode at a time...
```

### 8.4 Training Configuration

```python
@dataclass
class ThinkingFusionConfig:
    """Stage 4: Thinking Mode Fusion."""
    lr: float = 5e-6                  # Low LR — preserve reasoning capability
    epochs: int = 3                    # Short — fusion, not learning
    batch_size: int = 4
    max_length: int = 2048
    warmup_steps: int = 30
    weight_decay: float = 0.01
    gradient_accumulation: int = 4     # Effective batch = 16
    num_thinking: int = 6_000          # Examples with <think> traces
    num_non_thinking: int = 4_000      # Examples without <think>
    save_dir: str = "checkpoints/nanoseek_stage4_thinking_fusion"

    # Data mixing
    thinking_ratio: float = 0.6        # 60% thinking, 40% non-thinking
    shuffle: bool = True               # Interleave thinking and non-thinking
```

### 8.5 Inference-Time Mode Control

```python
# At inference time, control thinking mode via system prompt or tags:

THINK_SYSTEM = "You are a helpful assistant. When solving complex problems, \
use <think>...</think> to reason step by step before giving your answer."

NO_THINK_SYSTEM = "You are a helpful assistant. Respond directly and concisely."

# Or via inline tags:
# User: /think <complex question>
# User: /no_think <simple question>
```

### 8.6 Validation

After Stage 4, verify both modes work:

| Test | Thinking Mode | Non-Thinking Mode |
|------|---------------|-------------------|
| GSM8K accuracy | Should match Stage 3 (±2%) | Lower (expected, no reasoning) |
| Response latency | Higher (reasoning tokens) | Lower (direct) |
| Format compliance | >90% have `<think>` | <5% have `<think>` |
| Simple QA accuracy | High | Should match or exceed thinking |
| H_load | >2 bits | >2 bits |

---

## 9. Stage 5 — General Alignment (DPO/SimPO)

### 9.1 Algorithm Selection

| Criterion | DPO | SimPO | KTO |
|-----------|-----|-------|-----|
| Reference model needed | Yes (memory) | No | No |
| Data format | Preference pairs | Preference pairs | Binary (thumbs up/down) |
| Performance | Proven | +6.4 AlpacaEval over DPO | Weaker signal |
| Memory overhead | High (2× model) | Low (1× model) | Low (1× model) |
| Implementation complexity | Low | Low | Low |
| Track record at small scale | Strong | Moderate | Limited |

**Primary**: DPO (proven, well-understood).
**Fallback**: SimPO if memory-constrained (NanoSeek-1B is ~5B total params, ref model
doubles this to ~10B — may be tight on single GPU).

### 9.2 DPO Implementation

```python
def dpo_loss(
    policy_chosen_logps: torch.Tensor,     # [B] log P(chosen | policy)
    policy_rejected_logps: torch.Tensor,   # [B] log P(rejected | policy)
    ref_chosen_logps: torch.Tensor,        # [B] log P(chosen | reference)
    ref_rejected_logps: torch.Tensor,      # [B] log P(rejected | reference)
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Direct Preference Optimization loss.

    L_DPO = -E[log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x)
                          - log π_θ(y_l|x)/π_ref(y_l|x)))]
    """
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    return loss
```

### 9.3 Preference Data

```
Helpfulness pairs (1,500):
  - Chosen: detailed, accurate, well-structured response
  - Rejected: incomplete, incorrect, or poorly structured response
  - Sources: generate pairs from Stage 4 model, human-annotate preference

Safety pairs (500):
  - Chosen: appropriate refusal or safe response
  - Rejected: harmful, biased, or privacy-violating response
  - Sources: standard safety datasets (Anthropic HH, BeaverTails)

Format pairs (500):
  - Chosen: follows instructions precisely (format, length, style)
  - Rejected: ignores formatting requirements
  - Sources: generate pairs with varying instruction adherence

Thinking mode pairs (500):
  - Chosen: appropriate mode for question complexity
  - Rejected: uses <think> for trivial questions, or skips for complex ones
  - Sources: curate from Stage 4 model outputs

Total: 3,000 preference pairs
```

### 9.4 Training Configuration

```python
@dataclass
class AlignmentConfig:
    """Stage 5: General Alignment (DPO/SimPO)."""
    algorithm: str = 'dpo'            # 'dpo' or 'simpo'
    lr: float = 5e-7                  # Very low LR (alignment, not learning)
    beta: float = 0.1                  # DPO temperature parameter
    total_steps: int = 200             # Short alignment phase
    batch_size: int = 4
    max_length: int = 2048
    warmup_steps: int = 10
    weight_decay: float = 0.01
    gradient_accumulation: int = 2     # Effective batch = 8
    max_grad_norm: float = 1.0
    save_dir: str = "checkpoints/nanoseek_stage5_aligned"

    # MoE stabilization
    freeze_router: bool = True         # Keep Routing during RL-like optimization
    gamma: float = 0.0                 # Disable load balance updates

    # SimPO specific (if algorithm='simpo')
    simpo_gamma: float = 0.5           # Reward margin
    simpo_beta: float = 2.5            # Inverse temperature

    # Eval
    eval_every: int = 50
```

---

## 10. Cross-Stage Distillation (Phase 1)

### 10.1 Purpose

After 6 stages, the model may have partially forgotten capabilities acquired in earlier
stages (catastrophic forgetting). Phase 1 cross-stage distillation addresses this by training on
a **balanced mix of data from Stages 0-5**.

> **Note**: Phase 2 has its own 300-step cross-stage distillation covering data from all 9 stages
> (Stages 0-8). See [§11.4](#114-stage-8--agent-rejection-sampling) for Phase 2 distillation details.

### 10.2 Data Mix

```
Cross-stage distillation dataset (balanced):

From Stage 0: 500 teacher distillation examples (hard math/code)
From Stage 1: 500 extended reasoning examples (diverse domains)
From Stage 2: 200 best RL outputs (highest reward, verified correct)
From Stage 3: 500 rejection-sampled examples (filtered quality)
From Stage 4: 300 thinking + 200 non-thinking examples
From Stage 5: 300 preferred responses from DPO pairs

Total: 2,500 examples
```

### 10.3 Training Configuration

```python
@dataclass
class CrossDistillConfig:
    """Cross-Stage Distillation."""
    lr: float = 3e-6                  # Moderate LR
    total_steps: int = 500             # Short consolidation phase
    batch_size: int = 4
    max_length: int = 2048
    warmup_steps: int = 20
    weight_decay: float = 0.01
    gradient_accumulation: int = 4     # Effective batch = 16
    save_dir: str = "checkpoints/nanoseek_reason"
```

### 10.4 Validation

After cross-stage distillation, verify NO capability regression:

| Capability | Measurement | Acceptable Regression |
|-----------|-------------|----------------------|
| Math reasoning | GSM8K accuracy | ≤ 2% drop from Stage 2 peak |
| Code generation | HumanEval pass@1 | ≤ 3% drop from Stage 2 peak |
| Format compliance | <think> structure check | ≤ 5% drop from Stage 1 |
| Thinking mode | Mode-appropriate response rate | ≤ 5% drop from Stage 4 |
| Safety | Refusal rate on harmful prompts | ≤ 2% drop from Stage 5 |
| Expert health | H_load | Must be > 2 bits |
| Specialization | I_spec | Within ±0.1 nats of Stage 1 |

---

## 11. Phase 2 — Agentic RL Extension

### 11.1 Motivation and Research Grounding

The frontier has shifted from single-turn reasoning to multi-turn tool-using agency. Phase 2
extends NanoSeek from a reasoning model to an agentic model capable of interacting with
external tools across multiple turns.

| System | Scale | Result | Key Innovation |
|--------|-------|--------|----------------|
| DeepSWE | 32B | 42.2% SWE-bench Verified | Pure RL, no SFT, tool masking |
| Kimi-Researcher | — | 26.9% HLE (SOTA) | Long-horizon credit assignment |
| ASearcher | 7B-72B | 100+ tool calls/query | Progressive horizon expansion |
| Open-AgentRL | 4B | 4B > 32B on agent tasks | Structured data + reward design |
| OpenAI Deep Research | — | Production deployed | End-to-end agentic pipeline |

**Two-release strategy**: Phase 1 (NanoSeek-Reason) ships first with Stages 0-5. Phase 2
(NanoSeek-Agent) starts from the Phase 1 cross-stage distillation checkpoint and adds
Stages 6-8 plus a 300-step Phase 2 cross-stage distillation.

**Why start from cross-distill checkpoint** (not Stage 5): The cross-distill checkpoint has
consolidated capabilities from all 6 stages with no catastrophic forgetting. Starting agentic
training from a balanced checkpoint prevents the new distribution from overwriting alignment.

### 11.2 Stage 6 — Tool Format SFT

**Purpose**: Teach the ReAct action/observation protocol. The model must learn to generate
well-formed tool calls and parse observation tokens before RL can optimize tool-use behavior.

**Data**: 2,000 ReAct examples across 4 tool types:

| Tool Type | Count | Example Task |
|-----------|-------|--------------|
| `calculator` | 600 | Multi-step arithmetic requiring tool calls |
| `search` | 500 | Factual questions requiring retrieval |
| `code_exec` | 500 | Code debugging with execution feedback |
| `retrieve` | 400 | Multi-hop QA requiring document lookup |

**Format example**:

```
<think>
I need to solve 47 × 83 + 291. Let me use the calculator.
</think>
<action>calculator: 47 * 83</action>
<observation>3901</observation>
<think>
Now I need to add 291 to 3901.
</think>
<action>calculator: 3901 + 291</action>
<observation>4192</observation>
<answer>4192</answer>
```

**Training configuration**:

```python
@dataclass
class ToolFormatSFTConfig:
    """Stage 6: Tool Format SFT — teach ReAct protocol."""
    base_checkpoint: str = "checkpoints/nanoseek_reason.pt"  # Phase 1 cross-distill
    lr: float = 5e-6                    # Same as Stage 1 (SFT regime)
    epochs: int = 3
    batch_size: int = 4
    max_length: int = 2048              # ReAct trajectories are longer
    warmup_steps: int = 20
    weight_decay: float = 0.01
    gradient_accumulation: int = 4      # Effective batch = 16
    router_frozen: bool = False         # UNFROZEN — adapts to tool-use distribution
    gamma_load_balance: float = 0.001   # Active (SFT stage)
    save_dir: str = "checkpoints/stage6_tool_sft"
```

**Router**: UNFROZEN — the tool-use distribution (alternating think/action/observation tokens)
is significantly different from pure reasoning. The router must adapt to route action-generation
tokens through potentially different expert pathways.

### 11.3 Stage 7 — Agentic RL

**Algorithm**: GRPO with token masking. GRPO (not GSPO) is used here because agentic
trajectories contain observation tokens generated by the environment, not the model. Token
masking excludes these observation tokens from the gradient — only model-generated tokens
(think, action, answer) receive RL gradient signal. This makes sequence-level ratio computation
(GSPO's advantage) less critical, since the masked token positions already reduce MoE routing
noise in the gradient.

**Progressive horizon expansion** (ScalingInter-RL pattern):

```
Phase 7a (steps 1-15):   3-turn trajectories
  → Easy credit assignment, model learns basic tool calling
  → Converge to >60% task completion before expanding

Phase 7b (steps 16-30):  5-turn trajectories
  → Multi-step reasoning with tool interaction
  → Re-initialize from best Phase 7a checkpoint

Phase 7c (steps 31-50):  10-turn trajectories
  → Complex multi-hop tasks requiring planning
  → Re-initialize from best Phase 7b checkpoint
```

**Dense process rewards** (Planner-R1 finding — critical for 1B models):

Rather than waiting until trajectory completion to assign a single reward, each turn receives
partial credit. See §11.5 for credit assignment strategy and §11.6 for reward design.

**Training configuration**:

```python
@dataclass
class AgenticRLConfig:
    """Stage 7: Agentic RL — GRPO with token masking."""
    base_checkpoint: str = "checkpoints/stage6_tool_sft.pt"
    algorithm: str = "grpo"             # GRPO (not GSPO — see rationale above)
    total_steps: int = 50               # Progressive: 15 + 15 + 20
    lr: float = 1e-6                    # Same as Stage 2
    group_size: int = 4                 # Smaller group (trajectories are expensive)
    max_turns: int = 10                 # Maximum turns per trajectory
    progressive_horizons: list = (3, 5, 10)  # Turn limits per phase
    horizon_steps: list = (15, 15, 20)  # Steps per horizon phase
    token_masking: bool = True          # Mask observation tokens from gradient
    kl_coeff: float = 0.01             # Low KL — allow policy exploration
    clip_epsilon: float = 0.2
    router_frozen: bool = True          # FROZEN (Keep Routing — Technique 3)
    gamma_load_balance: float = 0.0     # Disabled during RL
    save_dir: str = "checkpoints/stage7_agentic_rl"
```

**Router**: FROZEN (Keep Routing). Same rationale as Stage 2 — RL gradients destabilize
MoE routing. All 4 V3.2 stabilization techniques active.

**γ**: 0.0 (disabled during RL). Same rationale as Stages 2 and 5.

### 11.4 Stage 8 — Agent Rejection Sampling

**Purpose**: Extract high-quality agentic trajectories from the RL-trained model, then
retrain from the Phase 1 cross-distill checkpoint (NOT the Stage 7 RL checkpoint).

**Process**:
1. Sample 8 trajectories per task from the Stage 7 checkpoint
2. Filter by: task completion (binary) + format compliance (ReAct structure) +
   trajectory efficiency (prefer shorter successful trajectories)
3. Keep ~1,000 high-quality trajectories
4. SFT from Phase 1 cross-distill checkpoint on these trajectories for 2 epochs

**Why retrain from cross-distill** (same rationale as Stage 3): The RL checkpoint has RL
artifacts (reward hacking patterns, distribution shift). By using the RL model only as a
**data generator** and retraining from a clean SFT checkpoint, we get agentic capability
without RL artifacts.

```python
@dataclass
class AgentRejectionConfig:
    """Stage 8: Agent Rejection Sampling."""
    rl_checkpoint: str = "checkpoints/stage7_agentic_rl.pt"    # For generation
    base_checkpoint: str = "checkpoints/nanoseek_reason.pt"    # For retraining
    samples_per_task: int = 8
    min_completion_rate: float = 0.6     # At least 60% of tasks have ≥1 success
    epochs: int = 2
    lr: float = 1e-5
    max_length: int = 4096              # Longer trajectories
    router_frozen: bool = False         # UNFROZEN (SFT stage)
    gamma_load_balance: float = 0.001
    save_dir: str = "checkpoints/stage8_agent_rejection"
```

**Phase 2 Cross-Stage Distillation**: After Stage 8, run 300 steps of cross-stage distillation
covering data from ALL stages (0-8). This produces the final `nanoseek_agent.pt` checkpoint.

```python
@dataclass
class Phase2CrossDistillConfig:
    """Phase 2 Cross-Stage Distillation (Stages 0-8)."""
    base_checkpoint: str = "checkpoints/stage8_agent_rejection.pt"
    lr: float = 3e-6
    total_steps: int = 300              # Shorter than Phase 1 (500 steps)
    batch_size: int = 4
    max_length: int = 4096
    save_dir: str = "checkpoints/nanoseek_agent"
```

### 11.5 Credit Assignment Strategy

Credit assignment is the critical bottleneck for agentic RL at 1B scale. Three approaches,
ordered by complexity:

**1. Gamma-decay reward** (Kimi-Researcher):

```python
def gamma_decay_reward(final_reward: float, num_turns: int, gamma: float = 0.9):
    """Assign decaying credit to earlier turns. Simplest baseline."""
    return [final_reward * (gamma ** (num_turns - i - 1)) for i in range(num_turns)]

# Example: 5-turn trajectory, reward=1.0, γ=0.9
# Turn rewards: [0.656, 0.729, 0.810, 0.900, 1.000]
```

Simple but imprecise — all turns get positive credit even if some were harmful.

**2. Dense progress rewards** (Planner-R1) — **RECOMMENDED for 1B**:

```python
def dense_progress_reward(trajectory, task):
    """Per-turn partial credit based on task progress. 3.5× more compute-efficient."""
    rewards = []
    for turn in trajectory:
        r = 0.0
        if turn.action_well_formed:      r += 0.1   # Format compliance
        if turn.observation_parsed:       r += 0.1   # Used observation correctly
        if turn.made_progress(task):      r += 0.2   # Moved closer to solution
        if turn.is_final and turn.correct: r += 0.5  # Task completion
        rewards.append(r)
    return rewards
```

Per-turn partial credit — the model knows exactly which actions helped. 3.5× more
compute-efficient than trajectory-level rewards at 8B scale (Planner-R1, arXiv: 2505.15966).

**3. HCAPO hindsight credit** (arXiv: 2603.08754):

Critic-based credit assignment where a separate model evaluates each turn's contribution
in hindsight. Too expensive for 1B (requires training a critic), but valuable for future
scaling to 3B-7B.

**Recommendation**: Start with gamma-decay (simplest to implement). Upgrade to dense
progress rewards if credit dilution is observed (reward signal doesn't decrease loss).
HCAPO is flagged for future work at larger scale.

### 11.6 Reward Design for Agent Tasks

```python
def agent_reward(trajectory, task, max_turns: int = 10):
    """Combined reward for agentic RL (Stage 7)."""
    # Outcome reward: binary task completion
    outcome = 1.0 if task.check_completion(trajectory) else 0.0

    # Format compliance: ReAct structure (think/action/observation/answer)
    format_score = sum(
        1.0 for turn in trajectory if turn.has_valid_react_format
    ) / len(trajectory)
    format_reward = 0.2 * format_score

    # Efficiency bonus: reward shorter successful trajectories
    if outcome > 0:
        efficiency = 0.1 * (1.0 - len(trajectory) / max_turns)
    else:
        efficiency = 0.0

    # Error penalty: malformed tool calls
    error_count = sum(1 for turn in trajectory if turn.has_malformed_tool_call)
    error_penalty = -0.1 * error_count

    return outcome + format_reward + efficiency + error_penalty
```

### 11.7 Environment Design

Three text-based environments appropriate for 1B scale:

**1. Math Calculator**: Multi-step math problems requiring tool calls.
- Task: "What is (47 × 83 + 291) / 12?"
- Tools: `calculator` (evaluates arithmetic expressions)
- Verification: Exact numerical match

**2. Code Debugger**: Buggy code + test cases, model calls execution tool.
- Task: "Fix this function so all test cases pass: [code + tests]"
- Tools: `code_exec` (runs Python code, returns stdout/stderr)
- Verification: All test cases pass

**3. Retrieval QA**: Multi-hop questions requiring search queries.
- Task: "What year was the university founded where [person] studied?"
- Tools: `search` (returns text snippets), `retrieve` (returns document)
- Verification: Exact answer match

```python
class AgentEnvironment:
    """Base interface for agent task environments."""
    def generate_task(self) -> dict:
        """Return {prompt, ground_truth, max_turns, tools_available}."""
        ...

    def execute_action(self, action: str) -> str:
        """Execute tool call, return observation string."""
        ...

    def check_completion(self, trajectory) -> bool:
        """Check if task is completed correctly."""
        ...
```

### 11.8 MoE Expert Specialization Under Agentic RL

> **Novel research territory** — no published work exists on MoE expert specialization
> under agentic RL at nano scale. The following is **hypothesis**, not established finding.

**Hypothesis**: Under agentic RL training, the MoE routing pattern will differentiate across
trajectory phases:

```
Shared experts (2, always active):
  → Absorb ReAct format tokens (<think>, <action>, <observation>, <answer>)
  → Universal across all tool types and trajectory phases

Routed experts (64, top-8 per token):
  → Phase differentiation hypothesis:
    - "Reasoning experts" activate during <think> blocks
    - "Action experts" activate during tool call generation
    - "Parsing experts" activate during observation processing
```

**Measurement**: Track I_spec (mutual information between expert and domain) separately
for each trajectory phase:
- I_spec(expert; thinking_tokens)
- I_spec(expert; action_tokens)
- I_spec(expert; observation_tokens)

If these diverge significantly (Δ > 0.2 nats), it indicates the MoE has learned to
specialize experts for different phases of agentic reasoning. This would be a portfolio
differentiator — **cite honestly as hypothesis** in any writeup.

### 11.9 Token Masking Strategy

Observation tokens are generated by the environment, not the model. Including them in the
RL gradient would train the model on signals it can't control — like training a student on
the teacher's words instead of the student's own answers.

```python
def create_token_mask(trajectory):
    """
    Binary mask: 1 for model-generated tokens, 0 for environment observations.
    In GRPO: multiply per-token log-probs by mask before sequence-level summation.
    """
    mask = []
    for turn in trajectory:
        mask.extend([1] * len(turn.think_tokens))       # Model generated
        mask.extend([1] * len(turn.action_tokens))       # Model generated
        mask.extend([0] * len(turn.observation_tokens))  # Environment generated
        mask.extend([1] * len(turn.answer_tokens))       # Model generated
    return torch.tensor(mask, dtype=torch.float32)

def masked_grpo_loss(log_probs, advantages, mask):
    """GRPO loss with token masking for agentic RL."""
    # Zero out observation token log-probs before computing sequence ratio
    masked_log_probs = log_probs * mask
    seq_log_prob = masked_log_probs.sum(dim=-1)  # Sum only model tokens
    # ... rest of GRPO loss computation
    return loss
```

**Source**: Token masking is consensus practice across DeepSWE (Cognition Labs), Open-AgentRL,
and Cognition Labs' internal work. The specific implementation varies but the principle is
universal: don't backpropagate through tokens the model didn't generate.

---

## 12. GSPO vs GRPO vs DAPO — Algorithm Selection

### 12.1 Detailed Comparison

| Criterion | GSPO | GRPO | DAPO |
|-----------|------|------|------|
| **Provenance** | Qwen3.5 (2026) | DeepSeek R1 (2025) | ByteDance (2025) |
| **MoE stability** | Designed for MoE | General-purpose | General-purpose |
| **Importance ratio level** | Sequence | Token | Token |
| **KL penalty** | Yes | Yes | No (removed) |
| **Reference model** | Yes | Yes | No |
| **Memory overhead** | Higher | Higher | Lower |
| **Clipping** | Symmetric (ε=0.2) | Symmetric (ε=0.2) | Asymmetric (0.2/0.28) |
| **Loss aggregation** | Sequence-level | Sequence-level | Token-level |
| **Dynamic sampling** | No | No | Yes (oversample + filter) |
| **Expert activation churn** | ~2-3% per step | ~10% per step | ~10% per step |
| **Research novelty** | High (GSPO at nano-MoE) | Low (well-known) | Moderate |
| **Proven at 1B scale** | No (proven at 17B) | Yes (via distillation) | Yes (at 1.5B) |

### 12.2 Theoretical Connection

Recent research ("It Takes Two: Your GRPO Is Secretly DPO", 2025) shows GRPO is formally
equivalent to a form of contrastive learning, connecting it to DPO. GSPO extends this
insight by operating the contrastive objective at the sequence level, where MoE routing
noise averages out.

```
GRPO ≈ Contrastive Learning (token-level) ≈ DPO with group normalization
GSPO ≈ Contrastive Learning (sequence-level) = more stable for MoE
DAPO ≈ GRPO - KL + asymmetric clipping + token aggregation
```

### 12.3 Recommendation

```
Primary:   GSPO — implement first, use as default
Fallback:  DAPO — if GSPO fails to converge at 1B (try after 20 steps)
Baseline:  GRPO — run as comparison for ablation study

Decision point: After 20 GSPO steps, check:
  - Is on_policy_fraction > 40%? If no → switch to DAPO
  - Is H_load stable (std < 0.3 bits)? If no → switch to DAPO
  - Is reward_mean increasing? If no → switch to DAPO

All three produce a publishable comparison for the portfolio.
```

### 12.4 Dr. GRPO Insights to Apply

Regardless of algorithm choice, apply Dr. GRPO's corrections:
1. **Remove length bias**: Don't normalize advantage by response length
2. **Remove std normalization**: Use `A_i = R_i - mean(R_group)` without dividing by std
3. These corrections apply to GSPO and DAPO as well

```python
def compute_advantages_dr_grpo(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """
    Dr. GRPO advantages: remove std normalization for clearer training signal.
    A_i = R_i - mean(R_group)   (NO division by std)
    """
    B = rewards.shape[0] // group_size
    grouped = rewards.view(B, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    advantages = grouped - mean  # No std normalization
    return advantages.view(-1)
```

---

## 13. MoE-Specific Post-Training Considerations

### 13.1 Expert Activation Volatility

The fundamental challenge of post-training MoE models:

```
Pre-training: Router learns stable expert assignments over millions of steps
              Each token type consistently routes to specific experts
              Expert specialization develops (math → experts 12,37,51; code → experts 5,28,44)

RL gradient:  Changes policy logits → changes token probabilities →
              changes routing decisions → changes which experts process which tokens →
              changes expert gradients → cascading instability

Result:       ~10% of expert assignments change per gradient step (GRPO)
              ~2-3% with GSPO or Keep Routing (V3.2 Technique 3)
```

### 13.2 Router Freeze Strategy Per Stage

| Stage | Router Status | γ (load balance) | Rationale |
|-------|--------------|-------------------|-----------|
| Stage 0 (Distill SFT) | **UNFROZEN** | 0.001 | Router must adapt to reasoning data distribution |
| Stage 1 (Extended SFT) | **UNFROZEN** | 0.001 | Continue adapting |
| Stage 2 (RLVR) | **FROZEN** | 0.0 | RL gradients would destabilize routing |
| Stage 3 (Rejection SFT) | **UNFROZEN** | 0.001 | Retraining from base; router re-adapts |
| Stage 4 (Thinking Fusion) | **UNFROZEN** | 0.001 | Mode fusion requires routing adjustment |
| Stage 5 (DPO Alignment) | **FROZEN** | 0.0 | Preference optimization is RL-like |
| Phase 1 Cross-Distill | **UNFROZEN** | 0.001 | Phase 1 routing consolidation |
| Stage 6 (Tool Format SFT) | **UNFROZEN** | 0.001 | Router adapts to tool-use distribution |
| Stage 7 (Agentic RL) | **FROZEN** | 0.0 | RL gradients would destabilize routing |
| Stage 8 (Agent Rejection SFT) | **UNFROZEN** | 0.001 | Retraining from cross-distill; re-adapts |
| Phase 2 Cross-Distill | **UNFROZEN** | 0.001 | Final routing consolidation |

### 13.3 Shared Expert Behavior During Post-Training

NanoSeek has 2 shared experts (always active, process every token). Expected behavior:

```
After pre-training:
  - Shared experts learn universal features (syntax, grammar, common patterns)
  - Routed experts specialize on domain-specific features

After Stage 0-1 (SFT):
  - Shared experts absorb <think> format (universal across all reasoning)
  - Routed experts specialize on domain reasoning (math vs code vs logic)
  - I_spec should INCREASE (more specialization)

After Stage 2 (RLVR):
  - Shared experts maintain format (router frozen)
  - Routed expert weights update within frozen routing
  - I_spec should remain STABLE (±0.1 nats)

After Stage 4 (Thinking Fusion):
  - Shared experts learn mode-switching pattern
  - Routed experts maintain domain specialization
  - I_spec may decrease slightly (mode mixing)
```

### 13.4 MTP During Post-Training

MTP (Multi-Token Prediction) acceptance rate is a novel diagnostic during post-training:

```
High acceptance rate (>80%):
  → Model is confident in its next-token predictions
  → Reasoning is predictable and structured
  → Good sign: model has learned consistent patterns

Low acceptance rate (<50%):
  → Model is uncertain about next tokens
  → Reasoning is unpredictable or chaotic
  → Warning sign: RL may be introducing noise

Expected trajectory:
  Pre-training end:    ~60-70%
  After Stage 0:       ~70-75% (more predictable reasoning patterns)
  After Stage 1:       ~75-80% (solidified patterns)
  After Stage 2:       ~70-75% (RL introduces some exploration, slight decrease OK)
  After Stage 4:       ~75-80% (fusion stabilizes patterns)
  After Stage 5:       ~75-80% (alignment doesn't change much)
```

**Test-time scaling signal**: Low MTP acceptance on a specific prompt = model is uncertain
→ allocate more inference tokens. High acceptance = model is confident → fewer tokens needed.

```python
def adaptive_inference(model, prompt, mtp_threshold=0.75):
    """Use MTP acceptance rate to control inference token budget."""
    acceptance = model.get_mtp_acceptance_rate(prompt)

    if acceptance > mtp_threshold:
        max_tokens = 256     # Confident → short response
    elif acceptance > 0.5:
        max_tokens = 512     # Moderate → medium response
    else:
        max_tokens = 1024    # Uncertain → allow extended reasoning

    return model.generate(prompt, max_tokens=max_tokens)
```

### 13.5 γ During Post-Training

The γ=0.001 bias-only load balancing was designed for pre-training (DeepSeek V3).
During RL post-training:

```
Problem: γ updates the router bias to balance load.
         But during RL, the router is FROZEN (Keep Routing).
         If γ is active, it updates the bias of a frozen router →
         the bias changes but routing decisions can't adapt →
         load balance degrades, not improves.

Solution: Set γ = 0.0 during RL stages (2, 5, and 7).
          Re-enable γ = 0.001 during SFT stages (0, 1, 3, 4, 6, 8, cross-stage)
          where the router is unfrozen and can respond to bias updates.
```

### 13.6 MoE Behavior During Agentic RL

Agentic trajectories introduce a new token distribution not seen during Phase 1:

```
Phase 1 token types:  <think> reasoning, direct answers, preferences
Phase 2 token types:  <think> reasoning, <action> tool calls,
                      <observation> environment outputs, <answer> final answers
```

**Expected behavior**:

```
After Stage 6 (Tool Format SFT):
  - Shared experts absorb ReAct format tokens (universal across all tool types)
  - Routed experts begin differentiating action-generation vs reasoning
  - I_spec should INCREASE (new specialization dimension)

After Stage 7 (Agentic RL):
  - Router FROZEN — expert assignments stable
  - Routed expert weights update within frozen routing
  - I_spec should remain STABLE (±0.1 nats)
  - MTP acceptance rate may DROP during tool-call tokens
    (tool syntax is less predictable than natural language)

After Stage 8 (Agent Rejection SFT):
  - Router UNFROZEN — retraining from cross-distill checkpoint
  - Expert patterns re-established with agentic capability
  - I_spec should recover to Stage 6 level
```

**MTP trajectory through Phase 2**:

```
After Phase 1 cross-distill:  ~75-80% (reasoning patterns consolidated)
After Stage 6:                ~70-75% (new format, slight decrease expected)
After Stage 7:                ~65-70% (RL exploration, further decrease OK)
After Stage 8:                ~70-75% (rejection sampling stabilizes)
After Phase 2 cross-distill:  ~72-78% (final consolidation)
```

---

## 14. Monitoring & Quality Gates

### 14.1 Metrics Tracked Across All Stages

| Metric | Tool | Frequency | Alert Threshold |
|--------|------|-----------|-----------------|
| H_load (expert load entropy) | eval/information_metrics.py | Every eval step | < 2.0 bits → STOP |
| I_spec (expert specialization MI) | eval/information_metrics.py | Every eval step | Δ > 0.3 nats between stages |
| MTP acceptance rate | eval/moe_diagnostics.py | Every eval step | < 40% → WARNING |
| Dead experts (< 1% traffic) | eval/moe_diagnostics.py | Every eval step | > 5 dead → WARNING |
| GSM8K accuracy | scripts/base_eval.py | Stage boundaries | Regression > 3% |
| HumanEval pass@1 | scripts/base_eval.py | Stage boundaries | Regression > 5% |
| Format compliance | custom metric | Stage boundaries | < 85% → WARNING |
| Average CoT length | logged in RLVR | Every RL step | Unbounded growth → STOP |
| on_policy_fraction | grpo_trainer.py | Every RL step | < 10% for 5 steps → STOP |
| Task completion rate | agent eval | Every RL step (Stage 7) | < 10% after 20 steps → STOP |
| Avg trajectory length | agent eval | Every RL step (Stage 7) | Unbounded growth → STOP |
| Tool call compliance | agent eval | Every RL step (Stage 7) | < 50% → WARNING |
| Token masking fraction | agent trainer | Every RL step (Stage 7) | > 70% masked → WARNING |

### 14.2 Quality Gates

#### Gate 5 (REVISED): Before Marking Post-Training Valid

```
✅ Pre-training baseline measured on EMA weights (RULE 1):
   - GSM8K (greedy), HumanEval (pass@1), MMLU (5-shot), ARC-C, BoolQ
   - These are the "before" numbers for all post-training comparisons

✅ Stage 0 (Teacher Distill):
   - GSM8K accuracy ≥ 25% (baseline + distillation gain)
   - Format compliance ≥ 80% (model uses <think> structure)
   - H_load > 2 bits throughout
   - I_spec logged and compared to pre-training baseline

✅ Stage 1 (Extended SFT):
   - GSM8K accuracy ≥ 30% (further improvement from extended SFT)
   - Format compliance ≥ 90%
   - H_load > 2 bits throughout
   - No val_loss increase for final 3 epochs (model has stabilized)

✅ Stage 2 (RLVR):
   - GSM8K accuracy improved ≥ 5% over Stage 1 (RL value-add)
   - H_load preserved (± 0.5 bits of Stage 1 value)
   - I_spec preserved (± 0.1 nats of Stage 1 value)
   - on_policy_fraction > 30% average across training
   - MTP acceptance rate measured pre/post
   - Algorithm comparison logged (GSPO vs GRPO or DAPO vs GRPO)

✅ Stage 3 (Rejection Sampling):
   - At least 60% of prompts yielded ≥ 1 correct completion
   - Quality filter accepted ≥ 40% of correct completions
   - SFT from base converged (val_loss decreased)

✅ Stage 4 (Thinking Fusion):
   - Thinking mode: GSM8K accuracy within 2% of Stage 3
   - Non-thinking mode: simple QA accuracy ≥ 80%
   - Mode-appropriate responses: ≥ 85% (think for complex, direct for simple)
   - No mode leakage (thinking in non-think mode < 5%)

✅ Stage 5 (Alignment):
   - Safety refusal rate on harmful prompts ≥ 80%
   - Helpfulness maintained (GSM8K accuracy within 2% of Stage 4)
   - H_load > 2 bits

✅ Cross-Stage Distillation:
   - No capability regression > thresholds (see §10.4 table)
   - H_load > 2 bits
   - I_spec within ±0.1 nats of Stage 1

✅ Final measurements (all on EMA weights — RULE 1):
   - GSM8K, HumanEval, MMLU, ARC-C, BoolQ (full benchmark suite)
   - H_load, I_spec, MTP acceptance rate
   - Test-time scaling curve: accuracy vs tokens at 256/512/1024/2048
   - Thinking mode vs non-thinking mode comparison
   - Written up in reports/RL_SCALING_REPORT.md

✅ Phase 2 — Stage 6 (Tool Format SFT):
   - ReAct format compliance ≥ 80% on tool-use test prompts
   - Phase 1 reasoning preserved (GSM8K within 3% of Phase 1 final)
   - H_load > 2 bits throughout

✅ Phase 2 — Stage 7 (Agentic RL):
   - Task completion rate ≥ 30% on held-out agent tasks
   - Tool call compliance ≥ 60%
   - H_load preserved (± 0.5 bits of Stage 6 value)
   - I_spec preserved (± 0.1 nats of Stage 6 value)
   - Token masking fraction between 20-60%
   - Progressive horizon: completion ≥ 60% before each expansion

✅ Phase 2 — Stage 8 (Agent Rejection Sampling):
   - At least 50% of tasks yielded ≥ 1 successful trajectory
   - SFT from cross-distill converged (val_loss decreased)

✅ Phase 2 — Cross-Stage Distillation:
   - No Phase 1 capability regression > thresholds (§10.4 table)
   - Agent task completion within 2% of Stage 8 peak
   - H_load > 2 bits
```

### 14.3 Stage Boundary Diagnostics

At every stage transition, run:

```python
def stage_boundary_diagnostics(model, stage_name: str):
    """Run and log diagnostics at each stage boundary."""
    diagnostics = {
        'stage': stage_name,
        'h_load': compute_h_load(model),           # Expert load entropy
        'i_spec': compute_i_spec(model),             # Specialization MI
        'mtp_acceptance': compute_mtp_acceptance(model),
        'dead_experts': count_dead_experts(model),
        'gsm8k_acc': eval_gsm8k(model, n=200),
        'humaneval_pass1': eval_humaneval(model, n=50),
        'format_compliance': eval_format(model, n=100),
        'avg_cot_length': measure_cot_length(model, n=100),
    }

    # Log to W&B as a stage boundary event
    wandb.log({f'stage_boundary/{k}': v for k, v in diagnostics.items()})

    # Check critical thresholds
    assert diagnostics['h_load'] > 2.0, f"H_load collapse at {stage_name}!"
    assert diagnostics['dead_experts'] < 10, f"Too many dead experts at {stage_name}!"

    return diagnostics
```

---

## 15. Data Recipes & Sources

### 15.1 Complete Data Requirements

| Stage | Data Type | Count | Source | Cost |
|-------|-----------|-------|--------|------|
| Stage 0 | Teacher CoT | 50,000 | R1 distill dataset, Open-R1, NuminaMath | $0 |
| Stage 1 | Reasoning SFT | 10,000 | GSM8K, MATH, HumanEval, ARC, BBH (reformatted) | $0 |
| Stage 2 | RL prompts | ~5,000 | GSM8K train, MATH train, HumanEval, ARC-C | $0 |
| Stage 3 | Rejection-sampled | ~5,000 | Self-generated from Stage 2 model | $0 (compute only) |
| Stage 3 | General purpose | ~2,000 | Writing, QA, safety datasets | $0 |
| Stage 4 | Thinking mix | 10,000 | Stages 1+3 reasoning + new non-thinking | $0 |
| Stage 5 | Preference pairs | 3,000 | Self-generated + Anthropic HH + BeaverTails | $0 |
| Phase 1 Cross | Mixed | 2,500 | Sampled from Stages 0-5 | $0 |
| Stage 6 | ReAct examples | 2,000 | Synthetic ReAct trajectories (4 tool types) | $0 |
| Stage 7 | Agent RL prompts | ~1,000 | Math calc, code debug, retrieval QA | $0 |
| Stage 8 | Rejection-sampled trajectories | ~1,000 | Self-generated from Stage 7 model | $0 (compute only) |
| Phase 2 Cross | Mixed | 1,500 | Sampled from Stages 0-8 | $0 |
| **Total** | | **~93,000** | | **$0 data cost** |

### 15.2 Open-Source Datasets

```
Reasoning:
  - openai/gsm8k (HuggingFace)                    — 8.5K math word problems
  - hendrycks/competition_math (MATH)              — 12.5K competition math
  - AI-MO/NuminaMath-CoT                           — 860K math with CoT
  - openai/human-eval                              — 164 code problems
  - google/mbpp                                    — 974 code problems
  - allenai/arc (Challenge)                        — 2.6K science reasoning
  - lukaemon/bbh                                   — 23 hard reasoning tasks

Distillation:
  - deepseek-ai/DeepSeek-R1-Distill-Data           — 800K CoT from R1
  - open-r1/OpenR1-SFT-Data                        — Community R1 reproduction

Alignment:
  - Anthropic/hh-rlhf                              — 170K preference pairs
  - PKU-Alignment/BeaverTails                      — 330K safety annotations
  - HuggingFaceH4/ultrafeedback_binarized          — 64K preference pairs

General:
  - tatsu-lab/alpaca                               — 52K instruction-following
  - Open-Orca/OpenOrca                             — 4.2M instruction data
```

### 15.3 Data Processing Pipeline

```
Raw datasets
    │
    ▼
[1. Download & Format]
    - Standardize format: {prompt, response, ground_truth, domain}
    - Apply <think> formatting where needed
    │
    ▼
[2. Quality Filter]
    - Correctness verification (rule-based)
    - Length bounds (200-4000 tokens)
    - Language consistency (>95% English)
    - Format compliance (<think> structure)
    - Deduplication (MinHash)
    │
    ▼
[3. Domain Balance]
    - Target distribution per stage
    - Difficulty calibration (easy/medium/hard)
    - Dedup across stages
    │
    ▼
[4. Tokenize & Pack]
    - BOS-aligned packing
    - Pad to max_length
    - Save as Parquet
```

---

## 16. Budget & Compute Estimates

### 16.1 Per-Stage Compute

| Stage | GPU Hours (8×H100) | Wall Time | Est. Cost |
|-------|---------------------|-----------|-----------|
| Stage 0: Teacher Distill SFT | 16-32 H100-hours | 2-4h | $40-80 |
| Stage 1: Extended Reasoning SFT | 32-48 H100-hours | 4-6h | $80-120 |
| Stage 2: Reasoning RLVR (75 steps) | 16-24 H100-hours | 2-3h | $40-60 |
| Stage 3: Rejection Sampling + SFT | 8-16 H100-hours | 1-2h | $20-40 |
| Stage 4: Thinking Mode Fusion | 8-16 H100-hours | 1-2h | $20-40 |
| Stage 5: General Alignment DPO | 4-8 H100-hours | 0.5-1h | $10-20 |
| Phase 1 Cross-Stage Distillation | 4-8 H100-hours | 30min | $10-20 |
| Phase 1 Evaluation & diagnostics | 8-16 H100-hours | 1-2h | $20-40 |
| **Phase 1 Subtotal** | **96-168 H100-hours** | **~12-20h** | **$240-420** |
| | | | |
| Stage 6: Tool Format SFT | 4-8 H100-hours | 30min-1h | $10-20 |
| Stage 7: Agentic RL (50 steps) | 16-24 H100-hours | 2-3h | $40-60 |
| Stage 8: Agent Rejection Sampling + SFT | 8-16 H100-hours | 1-2h | $20-40 |
| Phase 2 Cross-Stage Distillation | 2-4 H100-hours | 20min | $5-10 |
| Phase 2 Evaluation & diagnostics | 4-8 H100-hours | 30min-1h | $10-20 |
| **Phase 2 Subtotal** | **34-60 H100-hours** | **~4-7h** | **$85-150** |
| | | | |
| **Combined Total** | **130-228 H100-hours** | **~16-27h** | **$325-570** |

**Note**: Rejection sampling in Stage 3 requires generating 8× completions per prompt from
the RL model. This is the most generation-heavy stage. With KV-cache and batch generation,
it's manageable in 1-2 hours.

### 16.2 Total Project Compute (Pre-training + Post-training)

| Phase | Est. Cost |
|-------|-----------|
| Anchor HP search (15 runs) | $40-60 |
| nano-500M validation | $50-80 |
| NanoSeek-1B pre-training (22B tokens) | $275-350 |
| Post-training Phase 1 (6 stages) | $240-420 |
| Post-training Phase 2 (3 stages) | $85-150 |
| **Total** | **$690-1060** |

### 16.3 Compute-Optimal Budgeting

DeepSeek R1 and Qwen3 both spend **~10-15% of total compute** on post-training relative
to pre-training. NanoSeek's ratio:

```
Pre-training:   22B tokens × 6 × 1.08B active × 2 = ~285T FLOPs
Post-training:  ~50T FLOPs (estimated across all stages)
Ratio:          50/285 ≈ 17.5% (slightly higher than frontier, acceptable for small scale)
```

---

## 17. Implementation Roadmap

### 17.1 File Structure

```
alignment/
├── __init__.py
│
├── # ─── Stage 0 ───
├── distill_sft.py                 ← NEW: Teacher distillation SFT
│   ├── DistillSFTConfig
│   ├── DistillDataset             (loads R1 distill data, applies filters)
│   └── run_distill_sft()          (training loop)
│
├── # ─── Stage 1 ───
├── reasoning_sft.py               ← RENAME from sft_warmup.py + extend
│   ├── ExtendedReasoningSFTConfig
│   ├── ReasoningSFTDataset        (GSM8K + MATH + code + logic)
│   └── run_reasoning_sft()        (extended training loop, 8-10 epochs)
│
├── # ─── Stage 2 ───
├── gspo_trainer.py                ← NEW: GSPO implementation
│   ├── GSPOConfig
│   ├── gspo_loss()                (sequence-level clipping)
│   └── GSPOTrainer                (extends GRPOTrainer logic)
│
├── grpo_trainer.py                ← MODIFY: Add GSPO mode, reduce default steps
│   ├── GRPOConfig                 (update defaults: total_steps=75)
│   └── GRPOTrainer                (add algorithm='gspo'|'dapo'|'grpo' switch)
│
├── rewards.py                     ← EXTEND: Add language consistency reward
│   ├── math_reward()              (existing)
│   ├── code_reward()              (existing)
│   ├── format_reward()            (existing)
│   ├── language_reward()          (NEW: English fraction in CoT)
│   ├── length_penalty()           (NEW: Dr. GRPO insight)
│   └── reasoning_reward()         (NEW: combined reward for Stage 2)
│
├── # ─── Stage 3 ───
├── rejection_sampling.py          ← NEW: Sample from RL model, filter
│   ├── RejectionConfig
│   ├── rejection_sample()         (generate N completions, filter)
│   ├── quality_filter()           (correctness + format + language)
│   └── build_rejection_dataset()  (prepare SFT data from samples)
│
├── # ─── Stage 4 ───
├── thinking_fusion.py             ← NEW: Thinking mode fusion
│   ├── ThinkingFusionConfig
│   ├── ThinkingFusionDataset      (mixed thinking + non-thinking data)
│   └── run_thinking_fusion()      (SFT with mode-aware data)
│
├── # ─── Stage 5 ───
├── dpo_trainer.py                 ← NEW: DPO/SimPO alignment
│   ├── AlignmentConfig
│   ├── dpo_loss()                 (standard DPO)
│   ├── simpo_loss()               (reference-free SimPO)
│   └── DPOTrainer                 (training loop with preference pairs)
│
├── # ─── Cross-Stage ───
├── cross_distill.py               ← NEW: Cross-stage distillation
│   ├── CrossDistillConfig
│   ├── CrossDistillDataset        (balanced mix from all stages)
│   └── run_cross_distill()        (500-step consolidation)
│
├── # ─── Stage 6 ───
├── tool_format_sft.py             ← NEW: Tool Format SFT (ReAct protocol)
│   ├── ToolFormatSFTConfig
│   ├── ReActDataset               (loads ReAct examples, 4 tool types)
│   └── run_tool_format_sft()      (training loop)
│
├── # ─── Stage 7 ───
├── agentic_rl.py                  ← NEW: Agentic RL with token masking
│   ├── AgenticRLConfig
│   ├── AgentEnvironment           (base class for agent environments)
│   ├── create_token_mask()        (mask observation tokens)
│   ├── masked_grpo_loss()         (GRPO with token masking)
│   └── AgenticRLTrainer           (progressive horizon expansion)
│
├── # ─── Stage 8 ───
├── agent_rejection.py             ← NEW: Agent rejection sampling
│   ├── AgentRejectionConfig
│   ├── sample_trajectories()      (generate N trajectories/task)
│   ├── filter_trajectories()      (completion + format + efficiency)
│   └── build_agent_rejection_dataset()
│
├── # ─── Orchestration ───
├── pipeline.py                    ← NEW: End-to-end pipeline orchestrator
│   ├── PostTrainingPipeline       (runs all 9 stages in 2 phases)
│   ├── stage_boundary_diagnostics() (metrics at each transition)
│   └── run_full_pipeline()        (CLI entry point)
│
├── # ─── Existing (kept) ───
├── run_grpo.py                    ← KEEP: Update config defaults
└── sft_warmup.py                  ← DEPRECATED: replaced by reasoning_sft.py
```

### 17.2 Implementation Order

```
Priority 1 (Highest ROI — implement first):
  1. distill_sft.py              — Stage 0 is the single highest-value stage
  2. reasoning_sft.py            — Extend sft_warmup.py to 10 epochs
  3. gspo_trainer.py             — GSPO is ~30 lines delta from existing GRPO
  4. pipeline.py                 — Orchestrate stages sequentially

Priority 2 (Core functionality):
  5. rejection_sampling.py       — Depends on Stage 2 working
  6. dpo_trainer.py              — Standard DPO, well-understood
  7. thinking_fusion.py          — Depends on Stage 3 data

Priority 3 (Polish):
  8. cross_distill.py            — Simple SFT on mixed data
  9. rewards.py extensions       — language_reward, length_penalty
  10. Updated monitoring          — stage_boundary_diagnostics

Priority 4 (Phase 2 — after NanoSeek-Reason ships):
  11. tool_format_sft.py         — Stage 6, ReAct protocol SFT
  12. agentic_rl.py              — Stage 7, GRPO + token masking + progressive horizons
  13. agent_rejection.py         — Stage 8, trajectory sampling and filtering
  14. Phase 2 cross-distill      — Extend cross_distill.py for Stages 0-8 data
```

### 17.3 Testing Strategy

Each new file needs unit tests before integration (RULE 4):

```
tests/
├── test_distill_sft.py          — Data loading, format, training step
├── test_gspo.py                 — GSPO loss computation, sequence-level clipping
├── test_rejection_sampling.py   — Sampling, filtering, quality scoring
├── test_thinking_fusion.py      — Data mixing, mode detection
├── test_dpo.py                  — DPO loss, preference pair handling
├── test_cross_distill.py        — Data mixing, no capability regression
├── test_pipeline.py             — End-to-end smoke test (tiny config)
├── test_tool_format_sft.py      — ReAct data loading, format validation
├── test_agentic_rl.py           — Token masking, progressive horizons, agent reward
├── test_agent_rejection.py      — Trajectory sampling, filtering, efficiency scoring
└── test_agent_environments.py   — Environment interface, action execution, completion check
```

---

## 18. Updated Rule 8

The original RULE 8:
```
RULE 8: RL post-training uses 3-stage pipeline, not single-stage GRPO.
  Stage 1: Reasoning RL (GRPO, 60% budget)
  Stage 2: Agent RL (GRPO, 25%)
  Stage 3: General Alignment (DPO, 15%)
  Cross-stage distillation (500 steps).
```

**RULE 8 (REVISED — 9-Stage, 2-Phase)**:
```
RULE 8: Post-training uses 9-stage, 2-phase pipeline.

  Phase 1 (NanoSeek-Reason):
    Stage 0: Teacher Distillation SFT (50K samples, 5 epochs, lr=2e-5)
             — Transfer reasoning from strong teacher model
    Stage 1: Extended Reasoning SFT (10K samples, 8-10 epochs, lr=5e-6)
             — Stabilize <think> format for downstream RL
    Stage 2: Reasoning RLVR (GSPO primary / DAPO fallback, 50-100 steps, lr=1e-6)
             — Verifiable rewards only (math, code, logic)
             — All 4 V3.2 MoE stabilization techniques active
    Stage 3: Rejection Sampling + SFT refinement (from pre-trained base, 2 epochs)
             — Extract reasoning quality, discard RL artifacts
    Stage 4: Thinking Mode Fusion (60/40 thinking/direct, 3 epochs, lr=5e-6)
             — Single model supports both think and no-think modes
    Stage 5: General Alignment (DPO/SimPO, 200 steps, lr=5e-7)
             — Safety + helpfulness + format preferences
    Cross-Stage Distillation (500 steps) → NanoSeek-Reason ships here

  Phase 2 (NanoSeek-Agent):
    Stage 6: Tool Format SFT (2K ReAct examples, 3 epochs, lr=5e-6)
             — Teach ReAct action/observation protocol
    Stage 7: Agentic RL (GRPO + token masking, 50 steps, lr=1e-6)
             — Progressive horizon: 3→5→10 turns
             — Dense process rewards (Planner-R1)
             — All 4 V3.2 MoE stabilization techniques active
    Stage 8: Agent Rejection Sampling (from Phase 1 cross-distill, 2 epochs)
             — Extract high-quality trajectories, discard RL artifacts
    Cross-Stage Distillation (300 steps) → NanoSeek-Agent ships here

  Router freeze strategy:
    FROZEN in Stages 2, 5, and 7 (RL-like optimization)
    UNFROZEN in Stages 0, 1, 3, 4, 6, 8, and cross-stage (SFT)
  γ load balance:
    0.001 in SFT stages (0, 1, 3, 4, 6, 8, cross-stage)
    0.0 in RL stages (2, 5, 7)
  Token masking:
    Stage 7 masks observation tokens from gradient (environment-generated, not model)
  Monitoring:
    H_load + I_spec + MTP acceptance at EVERY stage boundary
    Stage boundary diagnostics logged to W&B
    Test-time scaling curve at final evaluation
```

---

## 19. References

### Core Papers

| Paper | arXiv | Key Contribution |
|-------|-------|-----------------|
| DeepSeek-R1 | 2501.12948 | 4-stage post-training pipeline, GRPO, rule-based rewards, rejection sampling |
| Qwen3 Technical Report | 2505.09388 | 4-stage pipeline, thinking mode fusion, hybrid think/no-think |
| GSPO (Qwen3.5) | 2507.18071 | Sequence-level RL for MoE stability, fixes GRPO token-level instability |
| DAPO | 2503.14476 | Asymmetric clipping, no KL, token-level aggregation, overlong handling |
| Dr. GRPO | 2503.20783 | Remove length bias and std normalization from GRPO |
| GRPO (DeepSeekMath) | 2402.03300 | Original GRPO algorithm, group-relative advantages |
| DPO | 2305.18290 | Direct Preference Optimization, reference-free alignment |
| SimPO | 2405.14734 | Reference-free DPO variant, +6.4 AlpacaEval over DPO |

### Architecture & Pre-training

| Paper | arXiv | Key Contribution |
|-------|-------|-----------------|
| DeepSeek-V3 | 2412.19437 | Aux-loss-free load balancing, MTP, full architecture |
| DeepSeek-V3.2 | 2512.02556 | GRPO RL, 4 MoE stabilization techniques, DSA |
| Qwen3.5 Model Card | HuggingFace | 512 experts, hybrid attention (DeltaNet + full), GSPO |
| Joint MoE Scaling | 2502.05172 | L(N_active, D, E) formula for MoE scaling laws |

### Small Model RL

| Paper | arXiv | Key Contribution |
|-------|-------|-----------------|
| Phi-4-Reasoning | Microsoft TR | 90 GRPO steps sufficient, SFT on teacher demos |
| Open-R1 | HuggingFace blog | Community R1 reproduction, distillation datasets |
| Kimi k1.5 | 2501.12599 | Long2Short reasoning compression, RL prompt curation |
| SmolLM3 | HuggingFace | 3B recipe: 11.2T pretrain + 140B midtrain + APO |

### Alignment & Safety

| Paper | arXiv | Key Contribution |
|-------|-------|-----------------|
| Constitutional AI | 2212.08073 | Critique-revision SFT + RLAIF, 2026 constitution |
| KTO | 2402.01306 | Binary feedback alignment (thumbs up/down) |
| ORPO | 2403.07691 | Combined SFT + preference in single objective |
| SPICE | 2504.08547 | Self-play grounded in external documents |

### Process & Outcome Reward Models

| Paper | arXiv | Key Contribution |
|-------|-------|-----------------|
| Let's Verify Step by Step | 2305.20050 | Process reward models for math reasoning |
| PRM Training | 2312.08935 | Step-level annotation and training methodology |
| RLVR Analysis | 2506.14245 | RLVR as search compression, not expanded capability |

### Theoretical Connections

| Paper | arXiv | Key Contribution |
|-------|-------|-----------------|
| GRPO is DPO | 2510.00977 | Formal equivalence through contrastive learning lens |
| Beyond Two-Stage | 2025 | Synergistic SFT-RL interaction |
| CoT Distillation Structure | Snorkel blog | Structure > content in CoT distillation |

### RL Theory

| Paper | arXiv / Venue | Key Contribution |
|-------|---------------|-----------------|
| Does RL Really Incentivize Reasoning Beyond the Base Model? | OpenReview: 4OsgYD7em5 (NeurIPS 2025) | RL redistributes probability mass, doesn't create new reasoning |
| 1-Shot RLVR | 2505.00981 | Single RL step: 36% → 73.6% on MATH500 |

### Agentic RL

| Paper | arXiv / Source | Key Contribution |
|-------|---------------|-----------------|
| DeepSWE | Cognition Labs (2025) | 42.2% SWE-bench from pure RL, token masking, no SFT needed |
| Kimi-Researcher | Moonshot AI (2025) | 26.9% HLE (SOTA), long-horizon credit assignment |
| ASearcher | ICLR 2026 | 100+ tool calls per query, progressive horizon expansion |
| Open-AgentRL | 2503.11696 | 4B outperforms 32B with structured data + reward design |
| OpenAI Deep Research | OpenAI (2025) | Production multi-step web research agent |
| RL2F | Google (Feb 2026) | Self-improving RL via language feedback |

### Credit Assignment

| Paper | arXiv | Key Contribution |
|-------|-------|-----------------|
| Planner-R1 | 2505.15966 | Dense process rewards 3.5× more compute-efficient at 8B |
| HCAPO | 2603.08754 | Hindsight credit assignment with critic model |
| ScalingInter-RL | 2504.12680 | Progressive horizon expansion prevents collapse |

### Agentic Frameworks

| Paper | arXiv / Source | Key Contribution |
|-------|---------------|-----------------|
| Agent-R1 | 2503.05592 | Test-time search + RL for agentic reasoning |
| AgentGym-RL | 2502.09696 | Multi-environment RL training for agent capabilities |
| OpenManus-RL | Community (2025) | Open-source agentic RL framework |
| ART | 2504.09469 | Agent Reinforcement Training with trajectory-level rewards |
| SWE-Gym | 2501.01540 | Training environment for software engineering agents |

---

## Appendix A: GSPO vs GRPO — Mathematical Comparison

### GRPO (Token-Level)

```
L_GRPO = -E_i[min(ρ_i · A_i, clip(ρ_i, 1-ε, 1+ε) · A_i)] + β · KL(π_θ || π_ref)

where:
  ρ_i = Π_t [π_θ(a_t | s_t) / π_old(a_t | s_t)]   (product over tokens)
  A_i = (R_i - μ_group) / (σ_group + ε)              (group-relative advantage)
  KL  = E[π/π_ref - 1 - log(π/π_ref)]                (unbiased estimator)

Problem for MoE:
  ρ_i involves per-token ratios. In MoE, each token routes through different
  experts. After one gradient step, ~10% of expert assignments change.
  This means π_θ(a_t | s_t) changes not because the policy changed,
  but because a DIFFERENT EXPERT is now processing token t.
  Result: ρ_i fluctuates wildly → clipping catches most updates → no learning.
```

### GSPO (Sequence-Level)

```
L_GSPO = -E_i[min(ρ_seq_i · A_i, clip(ρ_seq_i, 1-ε, 1+ε) · A_i)] + β · KL

where:
  ρ_seq_i = exp(Σ_t log π_θ(a_t | s_t) - Σ_t log π_old(a_t | s_t))
          = exp(log_prob_seq_θ - log_prob_seq_old)

  (Same formula as GRPO, but the CLIPPING operates on the sequence-level ratio)

Why stable for MoE:
  Individual token routing changes average out over the full sequence.
  If token 42 routes to expert 7 instead of expert 12, its logit changes,
  but across 512 tokens, these fluctuations cancel out.
  The sequence-level ratio is a STABLE signal for gradient computation.
```

### Empirical Comparison (Qwen3.5 Findings)

```
Metric: Fraction of tokens clipped during training

GRPO:  ~90% of update steps clip most tokens (ratio outside [0.8, 1.2])
GSPO:  ~60% of update steps clip (ratio outside [0.8, 1.2])

Despite more clipping in absolute terms, GSPO trains MORE efficiently
because each unclipped update carries a MEANINGFUL gradient signal,
while GRPO's unclipped updates are dominated by routing noise.
```

---

## Appendix B: Qwen3.5 Architecture Summary (For Context)

The latest frontier MoE architecture that informs our post-training design:

```
Qwen3.5-397B-A17B:
  Total Parameters:    397B
  Active Parameters:   17B per token (4.3% activation)
  Hidden Dimension:    4,096
  Layers:              60
  Total Experts:       512
  Routed per token:    10
  Shared Experts:      1
  Expert Intermediate: 1,024 (very fine-grained)

  Hybrid Attention: 15 × (3 × GatedDeltaNet-MoE + 1 × GatedAttention-MoE)
    - 75% layers use linear attention (O(n))
    - 25% layers use full softmax attention (O(n²))
    - Native 262K context, extendable to ~1M via YaRN

  Post-Training:
    Stage 1: Long-CoT Cold Start
    Stage 2: Reasoning RL with GSPO (NOT GRPO)
    Stage 3: Thinking Mode Fusion
    Stage 4: Massive-Scale Agentic RL (20K parallel environments)

  Key Innovation: GSPO replaces GRPO for MoE RL stability
```

NanoSeek's architecture (64 experts, top-8, 1.08B active) is much smaller but shares the
same fundamental MoE instability challenges. GSPO is equally relevant at nano scale.

---

## Appendix C: Canonical 2026 Post-Training Pipeline (Industry Consensus)

For reference, the consensus post-training pipeline across major labs as of early 2026:

```
Base Model (pretrained)
  │
  ▼
[Stage 1] SFT on CoT Data
  - 800K-1M curated reasoning examples
  - Long CoT (10K+ tokens for large models, 2K-4K for small)
  - Extended training (up to 10 epochs)
  │
  ▼
[Stage 2] RLVR (Reasoning RL)
  - GRPO/DAPO/GSPO (no critic model needed)
  - Rule-based/verifiable rewards only (math, code, logic)
  - Group size 16-64 (4-8 for small models)
  - 90-100 steps sufficient for significant gains
  │
  ▼
[Stage 3] Rejection Sampling + SFT Refinement
  - Sample from RL checkpoint, filter for quality
  - Add general-purpose data
  - Retrain from base (DeepSeek) or continue (simpler)
  │
  ▼
[Stage 4] Alignment (RLHF/DPO/CAI)
  - Neural reward models or preference optimization
  - Constitutional AI principles for safety
  - SimPO/KTO as efficient DPO alternatives
```

NanoSeek's 9-stage, 2-phase pipeline adds three innovations to this consensus:
1. **Stage 0 (Teacher Distillation)**: Critical for small models where direct RL is insufficient
2. **Stage 4 (Thinking Mode Fusion)**: From Qwen3, creating a dual-mode model
3. **Phase 2 (Agentic RL Extension)**: Post-alignment agentic capability via Stages 6-8,
   extending the reasoning model to tool-using agency

---

## Appendix D: Agentic RL Frameworks Comparison

| Framework | Scale | SFT Needed? | Reward Type | Key Innovation |
|-----------|-------|-------------|-------------|----------------|
| AgentGym-RL | 7B-70B | Yes (warmup) | Environment reward | Multi-environment training |
| Agent-R1 | 7B-72B | Yes (CoT SFT) | Outcome + search | Test-time search + RL |
| OpenManus-RL | 7B-14B | Yes (ReAct SFT) | Task completion | Open-source agentic RL |
| ART | 8B-70B | Yes (trajectory SFT) | Trajectory-level | Agent Reinforcement Training |
| DeepSWE | 32B | **No** (pure RL) | Binary completion | Token masking, no SFT needed |

**Key insight for NanoSeek**: At 1B scale, SFT warmup (Stage 6) is essential — DeepSWE's
"pure RL, no SFT" approach requires 32B+ to work. All frameworks that succeed at <8B use
SFT warmup before agentic RL.

---

## Appendix E: Credit Assignment Methods Comparison

| Method | Complexity | Signal Quality | Memory Overhead | Best For |
|--------|-----------|----------------|-----------------|----------|
| Trajectory-level | O(1) | Low (sparse) | None | Large models (>32B) |
| Gamma-decay | O(T) | Medium | None | Simple baseline, any scale |
| Dense progress | O(T·K) | High | Low (K reward checks/turn) | **Small models (1B-8B)** |
| HCAPO (hindsight) | O(T·C) | Very high | High (critic model) | Future scaling (3B-7B) |
| Progressive horizon | O(H·T) | High | Low (H horizon phases) | Multi-turn RL (any scale) |

Where T = trajectory length, K = reward components per turn, C = critic forward passes,
H = number of horizon expansion phases.

**NanoSeek recommendation**: Dense progress rewards (Planner-R1) + progressive horizon
expansion (ScalingInter-RL). Combined, these address the two critical bottlenecks at 1B:
credit dilution (dense rewards) and policy collapse (progressive horizons).

---

*Document version: 2.0 | Last updated: 2026-03-18*
*Based on research through February 2026 including Qwen3.5, DeepSeek R1, and Agentic RL frontier*
