# NanoSeek — Claude Code Context File
## Project: Research-Grade DeepSeek V3.2 at Nano Scale
### Last updated: 2026-03-15 | Phase: Implementation (model Sections 1-7 complete, training infra active)

---

## What This Project Is (2 Sentences)

NanoSeek is a from-scratch reimplementation of DeepSeek V3.2 at 1.08B active / 4.75B total
parameters — written to be correct, educational, and experimentally rigorous. The goal is not
just a working model: it is a complete scaling science lab (scaling laws + training stability +
production observability + RL post-training) that produces **falsifiable, measurable artifacts**.

---

## Current Project Phase

```
Phase 0 (COMPLETE): Planning + paper analysis
  ✅ REIMPLEMENTATION_PLAN.md — full model + training spec
  ✅ SCALING_LAB_PLAN.md — 4-pillar scaling science plan
  ✅ PAPER_ANALYSIS_V3_V32.md — 5 critical corrections, 8 new components identified
  ✅ docs/ — 30+ architecture deep-dives

Phase 1 (MOSTLY COMPLETE): model/ implementation
  ✅ Sections 1-7: RoPE, RMSNorm, MLA, Gate, MoE, MTP, DecoderLayer, NanoSeekModel
  ✅ config.py: All configs correct (gamma_freeze_ratio=0.95, beta2=0.95, etc.)
  ❌ Sections 8-9: Lightning Indexer + DSA (Phase 2 only, empty placeholders)

Phase 2 (MOSTLY COMPLETE): Training infrastructure
  ✅ EMA tracking (pre_train.py EMATracker, decay=0.9999, every 10 steps)
  ✅ Batch warmup (1/5 → 1× over first 10%)
  ✅ FLOPs = 6 × N_active
  ✅ muP scaling (√B, 1/width, T_epoch)
  ✅ FIM 10% PSM in dataloader (RULE 6)
  ✅ Eval wiring: I_spec, MTP acceptance, domain BPB, dead experts
  ✅ Data curation pipeline (heuristic filters, quality classifier, dedup, mixture)
  ✅ Evaluation framework (domain BPB, I_spec, MoE diagnostics, scaling law)
  ❌ DSA two-stage LR (Phase 2 only)
  ✅ Checkpoint resume (--resume-from-step loads model/optimizer/EMA/dataloader state)

Phase 3 (NOT STARTED): Anchor HP search + stability ablations + nano-500M validation (Option B)
Phase 4 (NOT STARTED): NanoSeek-1B training (22B tokens)
Phase 5 (IN PROGRESS): RL post-training
  ✅ Single-stage GRPO with 4 V3.2 MoE stabilization techniques
  ✅ Reward functions (math, code, format)
  ✅ SFT warmup
  ❌ 3-stage pipeline (Reasoning→Agent→General + cross-stage distillation)
```

**What to work on next**: Run Gate 1 checks. If passing, start Phase 3 (anchor HP search).
For Phase 2 DSA: implement Sections 8-9 after Phase 1 4K training completes.

---

## Authority Hierarchy (Which Files to Trust)

When there is a conflict between files, this order wins:

```
1. docs/PAPER_ANALYSIS_V3_V32.md     — ground truth from papers (highest authority)
2. CLAUDE.md (this file)             — implementation status + rules + bug table
3. docs/SCALING_LAB_PLAN.md          — experimental plan (what to measure)
4. docs/05_CANON_LAYERS_DEEP_DIVE.md — architecture theory
5. nanoseek/model.py (current)       — existing code (Sections 1-7 correct, 8-9 placeholder)
```

Note: REIMPLEMENTATION_PLAN.md and OPTION_B_PLAN.md no longer exist as separate files.
Their content has been consolidated into CLAUDE.md (this file) and the plan file.

---

## The 12 Known Bugs — Fix These Exactly, Do Not Reinvestigate

These are documented in REIMPLEMENTATION_PLAN.md. Here is the precise fix for each:

| # | Bug | Location | Status | Fix |
|---|-----|----------|--------|-----|
| 1 | MTP uses cross-attention | `model.py:MTPBlock` | ✅ FIXED | Uses linear projection M_k + standard transformer block |
| 2 | `norm_topk_prob` never applied | `model.py:Gate.forward()` | ✅ FIXED | Line 610: `weights /= weights.sum(dim=-1, keepdim=True)` |
| 3 | `_gather_selected` dim bug | `model.py:DSASparseAttention` | ❌ Phase 2 | DSA not yet implemented (Section 9 placeholder) |
| 4 | `mscale` baked into init | `model.py:MLA` | ✅ FIXED | base_scale stored separately, applied at forward time |
| 5 | Shared experts use same `moe_inter_dim` | `model.py:MoE` | ✅ FIXED | `shared_inter_dim` parameter added |
| 6 | FLOPs use N_total not N_active | `scripts/pre_train.py` | ✅ FIXED | `6 * n_active` at line 184 |
| 7 | Indexer loss uses entropy, not KL | `model.py:_compute_indexer_loss` | ❌ Phase 2 | Indexer not yet implemented (Section 8 placeholder) |
| 8 | DSA MQA gather order wrong | `model.py:DSA._sparse_forward` | ❌ Phase 2 | DSA not yet implemented (Section 9 placeholder) |
| 9 | `get_gamma()` freezes at 80% | `model.py:NanoSeekModel` | ✅ FIXED | config.py: `gamma_freeze_ratio=0.95` everywhere |
| 10 | No FIM training | `nanoseek/dataloader.py` | ✅ FIXED | `fim_transform()` with 10% PSM, `fim_fraction` logged |
| 11 | No EMA tracking | `scripts/pre_train.py` | ✅ FIXED | EMATracker class, decay=0.9999, update every 10 steps |
| 12 | DSA two-stage LR missing | `scripts/pre_train.py` | ❌ Phase 2 | Needed when DSA Sections 8-9 are implemented |

---

## Critical Rules (Never Violate Without Discussion)

```
RULE 1: EMA weights are mandatory for ALL evaluation.
        Never evaluate on raw checkpoint weights. EMA only.
        If EMA checkpoint missing: refuse evaluation, raise error.

RULE 2: gamma_freeze_ratio = 0.95 everywhere.
        The old value of 0.80 was wrong. Do not use 0.80 anywhere.
        Comment: # V3 paper spec: 14.3T/14.8T = 96.6%, use 0.95

RULE 3: All evaluation uses ema_val_bpb, not val_bpb.
        All training runs (anchor, validation, 1B) must log ema_val_bpb.

RULE 4: Every new component needs a unit test before integration.
        Build order: component → unit test passes → integrate → integration test passes.

RULE 5: Do not copy from model/model.py to implement new components.
        The current model.py has known bugs. Derive from paper equations only.
        Use REIMPLEMENTATION_PLAN.md as the spec, papers as ground truth.

RULE 6: FIM requires training from scratch. Never add FIM via fine-tuning.
        The 10% PSM format must be in dataset.py from token 1.

RULE 7: Expert routing metrics (H_load + I_spec) must be logged for ALL stability runs.
        H_load = load-balance entropy (collapse detection). I_spec = MI(expert; domain) (specialization).
        These are DIFFERENT metrics — H_load measures uniformity, I_spec measures semantic roles.
        Primary scientific output: A vs C I_spec comparison (does aux-loss-free allow more specialization?).

RULE 8: Post-training uses 9-stage, 2-phase pipeline.
        Phase 1 (NanoSeek-Reason): Stage 0 Teacher Distill → Stage 1 Extended SFT →
        Stage 2 RLVR (GSPO) → Stage 3 Rejection Sampling → Stage 4 Thinking Fusion →
        Stage 5 Alignment (DPO) → Cross-stage distill (500 steps).
        Phase 2 (NanoSeek-Agent): Stage 6 Tool Format SFT → Stage 7 Agentic RL
        (GRPO + token masking, 50 steps) → Stage 8 Agent Rejection Sampling →
        Cross-stage distill (300 steps).
        All 4 V3.2 MoE stabilization techniques active in RL stages (2, 5, 7).
        Router FROZEN in RL stages, UNFROZEN in SFT stages.
        H_load + I_spec + MTP at EVERY stage boundary.

RULE 9: MTP acceptance rate is a test-time scaling signal — measure it during RL.
        Track at each stage boundary AND as a function of inference token budget.
        Plot test-time scaling curve: accuracy vs tokens at 256/512/1024/2048.
```

---

## Quality Gates — Pass Before Proceeding

### Gate 1: Before starting any training run
```bash
# Must all pass:
python -m nanoseek.model.model                    # test_nanoseek() — all shapes, loss finite
python -m pytest nanoseek/tests/ -v               # all 145+ tests pass
# Manual check: MFU calculation matches expected range for hardware
# Manual check: Expert load entropy > 4 bits at initialization (random routing)
# Manual check: EMA checkpoint saved alongside model checkpoint at step 100
# Manual check: FIM tokens appear at ~10% rate; fim_loss logged separately
# Manual check: MTP acceptance rate ~50% at initialization (untrained MTP)
```

### Gate 2: Before proceeding from nano-500M to NanoSeek-1B (Option B)
```
✅ nano-500M converged (no divergence, no NaN)
✅ ema_val_bpb is reasonable for 500M-class MoE (compare to literature)
✅ H_load stayed > 2 bits throughout (no expert collapse)
✅ MTP acceptance rate increased over training
✅ HP transfer produced these results WITHOUT per-scale tuning
   (if manual tuning was needed, document which scaling rule broke)
✅ gamma_freeze_ratio = 0.95 in run config
✅ EMA checkpoint exists at final step
```

### Gate 4: Before marking stability ablation valid
```
✅ All 3 runs (A, C, D) completed at anchor scale (~55M active)
✅ Bad batch injected at step 1500 for each run
✅ H_load and I_spec logged for all runs (RULE 7)
✅ A vs C: I_spec (specialization MI) comparison documented
✅ C vs D: spike recovery comparison documented (with SwiGLU confound noted)
```

### Gate 5: Before marking RL post-training valid
```
✅ GSM8K and HumanEval baselines measured on pre-trained model (EMA weights)
✅ 3 compute budgets (2%, 5%, 10%) each run through full 3-stage pipeline
✅ All 4 V3.2 stabilization techniques implemented (unbiased KL, off-policy mask,
   Keep Routing, Keep Sampling Mask) — active in ALL 3 stages
✅ Staging ablation: single-stage vs three-stage at Budget 2, comparison documented
✅ H_load preserved under ALL RL stages (± 0.5 bits of pre-RL value)
✅ I_spec preserved under ALL RL stages (± 0.1 nats of pre-RL value)
✅ H_load and I_spec measured at each stage boundary (pre-RL, post-Stage1, post-Stage2, post-Stage3)
✅ MTP acceptance rate measured before and after RL at each budget
✅ Test-time scaling curve: accuracy vs inference tokens (256/512/1024/2048) plotted
✅ Routing divergence between stages documented
✅ Cross-stage distillation completed (500 steps) with capability preservation verified
```

---

## File Map (What Each File Does)

```
nanoseek/                           ← PROJECT ROOT (this directory)
│
├── CLAUDE.md                       ← THIS FILE — project context for AI/human engineers
├── AGENTS.md                       ← STRATEGIC CONTEXT — original gaps analysis
├── MACHANIC_INTERPRET.md           ← Mechanistic interpretability notes
├── MLA_PRODUCTION_VERIFIED_PLAN.md ← MLA verification plan
│
├── nanoseek/                       ← CORE PACKAGE (model, config, data, training infra)
│   ├── config.py                   ← All model configs (anchor/500M/1B), gamma=0.95, beta2=0.95
│   ├── model.py                    ← Sections 1-7 complete (MLA, Gate, MoE, MTP, DecoderLayer, NanoSeekModel)
│   │                                  Sections 8-9 (Indexer, DSA) are Phase 2 placeholders
│   ├── dataloader.py               ← BOS-aligned best-fit packing + FIM 10% PSM (RULE 6)
│   ├── dataset.py                  ← Parquet dataset listing, NANOSEEK_DATA_DIR support
│   ├── tokenizer.py                ← Tokenizer with FIM token support (get_fim_tokens)
│   ├── checkpoint_manager.py       ← Save/load model, optimizer, EMA, metadata checkpoints
│   ├── common.py                   ← DDP init, distributed utilities
│   ├── engine.py                   ← Training engine utilities
│   ├── optim.py                    ← Optimizer setup (muP scaling, AdamW)
│   ├── report.py                   ← Training report generation
│   │
│   └── data_curation/              ← DATA PIPELINE (Priority 1)
│       ├── heuristic_filters.py    ← Rule-based quality filters
│       ├── quality_classifier.py   ← FastText-based quality scoring
│       ├── dedup.py                ← MinHash + LSH deduplication
│       ├── mixture.py              ← Domain mixture optimization
│       └── run_pipeline.py         ← End-to-end curation pipeline
│
├── eval/                           ← EVALUATION FRAMEWORK (Priority 2)
│   ├── domain_bpb.py               ← Per-domain BPB (code/math/science/web/books)
│   ├── information_metrics.py      ← I_spec (expert specialization MI), H_load (balance)
│   ├── moe_diagnostics.py          ← MTP acceptance rate, dead expert detection
│   └── scaling_law.py              ← Scaling law fitting L(N_active, D, E)
│
├── alignment/                      ← RL POST-TRAINING (Priority 4)
│   ├── grpo_trainer.py             ← Single-stage GRPO with 4 V3.2 MoE stabilization techniques
│   ├── rewards.py                  ← Math, code, format reward functions
│   ├── sft_warmup.py               ← SFT warmup before RL
│   └── run_grpo.py                 ← GRPO training script
│   # TODO: dpo.py, agent_environment.py, cross_stage_distill.py, pipeline.py (3-stage, RULE 8)
│
├── scripts/                        ← TRAINING & EVAL SCRIPTS
│   ├── pre_train.py                ← Main training loop: EMA, FIM, eval wiring, checkpoint resume,
│   │                                  config validation, batch warmup, muP scaling, FLOPs=6×N_active
│   ├── base_eval.py                ← Benchmark evaluation
│   └── chat_eval.py                ← Interactive chat evaluation
│
└── tests/                          ← TEST SUITE (115 passing, 8 skipped)
    ├── conftest.py                 ← Shared fixtures (minimal/1B configs, model/MLA/MoE/MTP fixtures)
    ├── test_nanoseek_model.py      ← Full model tests (layer assignment, MTP sharing, KV cache, gamma freeze)
    ├── test_moe.py                 ← MoE unit tests (gate, routing, load balance, gradient flow)
    ├── test_mla_standalone.py      ← MLA standalone tests (compression, causality, cache)
    └── test_moe_standalone.py      ← SKIPPED — needs rewrite for current Expert API
```

---

## Anti-Patterns (Never Do These)

```
❌ DON'T eval on raw weights — always use EMA
❌ DON'T use gamma_freeze_ratio=0.80 — it's wrong, use 0.95
❌ DON'T copy from current model.py — it has known bugs
❌ DON'T add FIM via fine-tuning — must train from token 1
❌ DON'T fit scaling law from train_loss — use ema_val_bpb
❌ DON'T use entropy loss for indexer — use KL-divergence (detached)
❌ DON'T expand all T tokens then gather in DSA — gather compressed first
❌ DON'T run stability ablations without gamma_freeze_ratio=0.95
❌ DON'T skip the unit test for a component before integrating it
❌ DON'T implement MMLU/ARC/BoolQ from scratch — port from nanochat/core_eval.py
❌ DON'T mark any training run valid if ema_val_bpb is missing from W&B
❌ DON'T run GRPO without all 4 V3.2 MoE stabilization techniques
❌ DON'T run single-stage RL — use 3-stage pipeline (Reasoning→Agent→General)
❌ DON'T skip cross-stage distillation — it prevents capability forgetting
❌ DON'T use generative reward models at 1B scale — stick with verifiable rewards
❌ DON'T ignore MTP acceptance rate during RL — it's a test-time scaling signal
```

---

## Key Numbers to Memorize

```
Architecture:
  N_active = 1.08B     N_total = 4.75B     expansion = 4.4×
  n_layers = 16        n_experts = 64      top_k = 8     κ = 12.5%
  hidden_dim = 2048    moe_inter = 768     G ≈ 29 (Krajewski optimal: 16-32)
  kv_lora_rank = 143   MLA compression = 23×
  Training tokens = 22B  Context Phase 1 = 4K  Context Phase 2 = 8K
  Design lineage: MoE sizing from OLMoE + Krajewski; MLA/routing from DeepSeek V3

Training hyperparameters:
  gamma_freeze_ratio = 0.95    (NOT 0.80)
  ema_decay = 0.9999           (update every 10 steps)
  fim_rate = 0.10              (PSM format, 10% of sequences)
  batch_warmup: 1/5→1× of target batch (over first 10% of steps, V3's 5× ratio)
  beta_2 = 0.95                (NOT 0.999)
  grad_clip = 1.0

Quality targets:
  Expert load entropy H_load > 2.0 bits (alert threshold for collapse)
  MTP acceptance rate > 75% (by end of training)
  MFU target: 47%
  HP transfer validation: nano-500M trains successfully with auto-scaled HPs

Option B validation path (3 points, muP-aligned):
  muP anchor (16L, 480h, ~55M active) → nano-500M (16L, 1280h, ~441M active) → NanoSeek-1B (16L, 2048h, 1.08B active)
  All configs: 64 experts, top-8, κ=12.5%, moe_inter=0.375×hidden, 2 shared experts
  Research question: do muP-corrected scaling rules (√B + 1/width + T_epoch) transfer to MoE?
```

---

## How to Onboard (Read This First, Then Do This)

```
Step 1: Read this file (CLAUDE.md) — project context, rules, bug table, current status
Step 2: Read docs/PAPER_ANALYSIS_V3_V32.md — understand the 5 critical corrections
Step 3: Read nanoseek/config.py — understand config structure (anchor/500M/1B)
Step 4: Read nanoseek/model.py — Sections 1-7 (complete), note Section 8-9 placeholders
Step 5: Read scripts/pre_train.py — training loop, EMA, FIM, eval wiring, checkpoint resume
Step 6: Run tests: python -m pytest tests/ -v — verify 115 pass
Step 7: Check plan file for remaining work packages
```

---

## What "Done" Looks Like for This Project (Tier 4 — No Constraints)

The project is complete when all 9 of these exist simultaneously:

```
1. TRAINED MODELS: NanoSeek-1B, 3B, 7B weights (EMA) on HuggingFace
   Metric: final ema_val_bpb on held-out validation set at each scale

2. HP TRANSFER: muP scaling rules validated for MoE (4-point: 55M → 500M → 1B → 3B)
   File: reports/HP_TRANSFER_REPORT.md

3. STABILITY: Ablation matrix (A, C, D) + Canon × MoE (5 runs)
   File: reports/STABILITY_PLAYBOOK.md with recommendations

4. DATA PIPELINE: Quality classifier + dedup + domain mix + proxy model validation
   Files: nanoseek/data/*.py (heuristic_filters, quality_classifier, dedup, mixture)

5. EVALUATION: lm-evaluation-harness integration + 7 benchmarks + safety evals
   File: nanoseek/eval/harness.py + per-domain BPB + MoE diagnostics

6. RL: 3-stage GRPO + PRM + Constitutional AI + iterative DPO
   + test-time scaling curve + MTP as scaling signal
   File: reports/RL_SCALING_REPORT.md

7. INTERPRETABILITY: SAE on MoE experts + fTRI + alignment probes
   File: reports/INTERPRETABILITY_REPORT.md

8. SCALE: 3B (FSDP2) + 7B (multi-node, expert parallelism)
   File: reports/SCALING_LAW_REPORT.md (L(N) fit, predictions vs actuals)

9. OBSERVABILITY: W&B dashboards + alert system + MFU profiling
   Files: training_ops/dashboards/*.json + monitoring/alerts.py
```

---

## Quick Reference: What's Next

```
COMPLETED  Model Sections 1-7, config, dataloader (FIM), pre_train.py (EMA, eval wiring,
           checkpoint resume, config validation), data curation, eval framework,
           single-stage GRPO, reward functions, SFT warmup. Tests: 115 passing.

NEXT       Phase 3: Anchor HP search (~55M active) on GPU
           → Verify Gate 1 manual checks (H_load > 4 bits, MTP ~50%, FIM ~10%)
           → nano-500M validation → NanoSeek-1B training (22B tokens)

LATER      Phase 2 DSA: model.py Sections 8-9 (Indexer + DSA) after 4K training
           3-stage GRPO pipeline (RULE 8): dpo.py, agent_environment.py,
           cross_stage_distill.py, pipeline.py in alignment/
           Mechanistic interpretability, scaling to 3B/7B
```
