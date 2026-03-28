# NanoSeek — Project Context
## Research-Grade DeepSeek V3.2 at Nano Scale
### Last updated: 2026-03-26 | Phase: Gate 1 PASSING, FP8 framework complete, ready for GPU training

---

## What This Project Is

NanoSeek is a from-scratch reimplementation of DeepSeek V3.2 at 1.08B active / 4.75B total
parameters. It is a complete scaling science lab: scaling laws + training stability +
production observability + RL post-training.

Architecture: MLA (23x KV compression) + MoE (64 experts, top-8) + MTP (speculative decoding) + FIM (10% PSM).

---

## Current Status (March 26, 2026)

```
Phase 0 (COMPLETE): Planning + paper analysis
Phase 1 (COMPLETE): Model Sections 1-7 (MLA, Gate, MoE, MTP, DecoderLayer, NanoSeekModel)
Phase 2 (COMPLETE): Training infra (EMA, FIM, muP, MuonAdamW, eval wiring, checkpoints)
Phase 2.5 (COMPLETE): FP8 training framework + training optimization plan
Phase 3 (READY):   Gate 1 checklist PASSING. Code complete. Needs GPU pod + data.
Phase 4 (NOT STARTED): NanoSeek-1B graduation run (22B tokens)
Phase 5 (PARTIAL):  Single-stage GRPO exists. 3-stage pipeline not yet built.
```

**What just happened (March 26)**:
1. Built custom FP8 training framework (`nanoseek/nanoseek/fp8.py`, ~320 lines):
   - MoE-aware selective conversion (gate router + embeddings protected)
   - Tensorwise dynamic scaling via `torch._scaled_mm` (cuBLAS native, no custom kernels)
   - E4M3 forward / E5M2 backward, `use_fast_accum=True` forward / `False` backward
   - `disable_fp8()` eval escape hatch for clean BF16 val_bpb measurement
   - Activated via `--fp8` flag in pre_train.py (auto-detects H100+, graceful fallback)
   - Discovery: MLA lora ranks (275, 90, 440, 143) not divisible by 16 → MLA projections
     stay BF16 at both scales. FP8 benefits shared expert + wo + sequential MoE path.
2. Wrote comprehensive training optimization plan (`TRAINING_OPTIMIZATION_PLAN.md`):
   - 7-phase plan: Liger kernels → profiling → MoE dispatch → FP8 → FSDP2 → arch-specific → dashboards
   - Phase 4.0 (FP8) fully implemented with first-principles documentation
3. All 124 tests passing, 8 skipped (expected).

**What to do next**: Spin up RunPod GPU pod → download data → Gate 1 smoke test → HP search.
See `RESEARCH_PLAN.md` for the full plan.

---

## Key Files — Read In This Order

### Setup & Operations
| File | What It Contains |
|------|-----------------|
| `SETUP.md` | **START HERE for new machines.** All dependencies, install commands, data download, verification checklist, exact training commands, troubleshooting. |
| `docs/TRAINING_EXECUTION_PLAN.md` | Step-by-step training plan: Gate 1 → HP search → 1B muP probe → 1B full. GPU requirements, cost estimates, RunPod pod specs, pass/fail criteria. |
| `docs/TRAINING_BUGS_POSTMORTEM.md` | 7 bugs that broke v1-v4 training. Root causes, fixes, prevention rules. **Read before modifying init or buffer code.** |

### Architecture & Design
| File | What It Contains |
|------|-----------------|
| `nanoseek/CLAUDE.md` | Full project context: architecture rules, 12 known bugs table, 9 critical rules, quality gates, file map, anti-patterns. **The authoritative reference.** |
| `docs/PAPER_ANALYSIS_V3_V32.md` | Ground truth from DeepSeek papers. 5 critical corrections. **Highest authority** when conflicts exist. |
| `docs/SCALING_LAB_PLAN.md` | 4-pillar research plan: HP transfer, stability, data, RL. |
| `docs/PROJECT_STATE.md` | Status dashboard, decision log. |

### Training Infrastructure
| File | What It Contains |
|------|-----------------|
| `nanoseek/TRAINING_OPTIMIZATION_PLAN.md` | **7-phase optimization roadmap**: Liger kernels, profiling, MoE dispatch, FP8 (implemented), FSDP2, arch-specific, dashboards. Phase 4.0 (FP8) complete with first-principles documentation. |
| `nanoseek/nanoseek/fp8.py` | **FP8 training framework** (~320 lines): MoE-aware conversion, tensorwise scaling, E4M3/E5M2, eval escape hatch. Activated via `--fp8` flag. |
| `nanoseek/TRAINING_PLAN_PHASE3.md` | Detailed Phase 3 plan: GPU selection, ablation design, RunPod setup, cost breakdown. |
| `nanoseek/RUNPOD_AGENT.md` | RunPod-specific agent context: setup scripts, monitoring, resume commands. |

---

## Project Structure

```
/workspace/
├── CLAUDE.md                          ← THIS FILE (top-level context)
├── SETUP.md                           ← Setup guide for new machines
│
├── nanoseek/                          ← Project root
│   ├── nanoseek/                      ← Python package
│   │   ├── nanoseek/                  ← Core modules
│   │   │   ├── model.py              ← Architecture (2,050 lines, Sections 1-7)
│   │   │   ├── config.py             ← Configs: ablation (1280h, 410M), 1b (2048h, 1.08B)
│   │   │   ├── fp8.py               ← FP8 training framework (MoE-aware, ~320 lines)
│   │   │   ├── dataloader.py         ← BOS-aligned packing + FIM 10% PSM
│   │   │   ├── optim.py              ← MuonAdamW + DistMuonAdamW + Polar Express
│   │   │   ├── tokenizer.py          ← RustBPE tokenizer (vocab=32768)
│   │   │   ├── dataset.py            ← ClimbMix data download + shard management
│   │   │   ├── checkpoint_manager.py ← Save/load model + optimizer + EMA + dataloader
│   │   │   ├── common.py             ← DDP init, device detection, logging
│   │   │   ├── engine.py             ← Training engine utilities
│   │   │   ├── report.py             ← Training report generation
│   │   │   ├── data_curation/        ← Quality filters, dedup, mixture (optional)
│   │   │   └── eval/                 ← Evaluation modules
│   │   │       ├── domain_bpb.py     ← Per-domain loss (code/math/science/web/books)
│   │   │       ├── information_metrics.py ← I_spec (expert specialization MI)
│   │   │       ├── moe_diagnostics.py    ← Dead experts, MTP acceptance, Gini
│   │   │       └── scaling_law.py        ← Chinchilla scaling law fitting
│   │   │
│   │   ├── scripts/
│   │   │   ├── pre_train.py          ← MAIN TRAINING SCRIPT (1,600 lines)
│   │   │   ├── run_phase3.py         ← Phase 3 orchestration
│   │   │   ├── monitored_train.py    ← Training wrapper with status files
│   │   │   ├── analyze_ablations.py  ← Ablation analysis
│   │   │   ├── base_eval.py          ← Benchmark evaluation
│   │   │   └── chat_eval.py          ← Interactive chat evaluation
│   │   │
│   │   ├── tests/                    ← 124 tests passing
│   │   │   ├── test_nanoseek_model.py ← Full model tests + meta device init regression
│   │   │   ├── test_moe.py           ← MoE unit tests
│   │   │   ├── test_mla_standalone.py ← MLA tests
│   │   │   └── conftest.py           ← Shared fixtures
│   │   │
│   │   ├── alignment/               ← RL post-training (partial)
│   │   │   ├── grpo_trainer.py       ← Single-stage GRPO
│   │   │   ├── rewards.py           ← Math, code, format rewards
│   │   │   └── sft_warmup.py        ← SFT before RL
│   │   │
│   │   ├── CLAUDE.md                 ← Detailed project rules & architecture context
│   │   ├── TRAINING_PLAN_PHASE3.md   ← Phase 3 detailed plan
│   │   └── RUNPOD_AGENT.md           ← RunPod agent context
│   │
│   ├── docs/
│   │   ├── TRAINING_EXECUTION_PLAN.md    ← Step-by-step training guide
│   │   ├── TRAINING_BUGS_POSTMORTEM.md   ← 7 bugs found & fixed
│   │   ├── PROJECT_STATE.md              ← Status dashboard
│   │   ├── PAPER_ANALYSIS_V3_V32.md      ← Paper ground truth
│   │   └── SCALING_LAB_PLAN.md           ← Research plan
│   │
│   ├── checkpoints/                  ← Saved model weights
│   ├── logs/                         ← Training logs
│   │   ├── gate1-smoke-v5.log        ← First correct smoke test (38 steps)
│   │   └── gate1-smoke-v5b.log       ← Full 100-step run (in progress)
│   └── wandb/                        ← W&B run data
│
└── data/
    └── base_data_climbmix/           ← Training data (parquet shards)
```

---

## Critical Rules (Never Violate)

1. **EMA weights for ALL evaluation** — never eval on raw checkpoint weights
2. **gamma_freeze_ratio = 0.95** — the old 0.80 was wrong
3. **FIM from token 1** — never add FIM via fine-tuning
4. **H_load + I_spec logged for ALL runs** — primary scientific output
5. **Test meta-device init before training** — `TestMetaDeviceInit` must pass
6. **Run all 120 tests before GPU training** — `pytest tests/ -v`
7. **Working directory = `/workspace/nanoseek`** — NOT `/workspace/nanoseek/nanoseek`

---

## Training Command Quick Reference

All commands run from `/workspace/nanoseek`:

```bash
# Gate 1 smoke test (100 steps, verify everything works)
WANDB_API_KEY="<key>" python -m nanoseek.scripts.pre_train \
    --run gate1-smoke --scale ablation --seed 42 \
    --num-iterations 100 --eval-every 50 --save-every 100 --device-batch-size 4

# HP search: 6 runs at ablation scale (500 steps each, ~$42 total)
for mlr in 0.005 0.01 0.02; do
  for elr in 0.2 0.5; do
    python -m nanoseek.scripts.pre_train \
        --run "hp-abl-mlr${mlr}-elr${elr}" --scale ablation \
        --matrix-lr $mlr --embedding-lr $elr \
        --num-iterations 500 --eval-every 100 --save-every -1 --seed 42
  done
done

# Full ablation training (1-4x A6000, ~10 hrs)
python -m nanoseek.scripts.pre_train \
    --run nanoseek-ablation-v1 --scale ablation \
    --matrix-lr <best> --embedding-lr <best> \
    --eval-every 250 --save-every 1000 --device-batch-size 16 --seed 42

# NanoSeek-1B graduation run (8x H100, ~14 hrs, FP8 enabled)
torchrun --nproc_per_node=8 -m nanoseek.scripts.pre_train \
    --run nanoseek-1b-v1 --scale 1b --fp8 \
    --matrix-lr <best> --embedding-lr <best> \
    --eval-every 500 --save-every 2000 --device-batch-size 16 --seed 42

# Download training data
cd nanoseek && python -m nanoseek.nanoseek.dataset -n 170
```

---

## Key Numbers

### Ablation Scale (PRIMARY — where experiments happen)
| Parameter | Value |
|-----------|-------|
| N_active (ablation) | ~410M |
| N_total (ablation) | ~1.95B |
| d_model | 1280 |
| Layers | 16 (same as 1B) |
| Training tokens | 8.2B |
| Cost per full run | ~$35 |

### 1B Scale (graduation run)
| Parameter | Value |
|-----------|-------|
| N_active (1B) | 1.08B |
| N_total (1B) | 4.75B |
| d_model | 2048 |
| Layers | 16 |
| Training tokens | 22B |
| Cost per full run | ~$350 |

### Shared across all scales
| Parameter | Value |
|-----------|-------|
| Experts | 64 routed + 2 shared |
| Top-k | 8 |
| Vocab | 32,768 |
| Sequence length | 4,096 |
| EMA decay | 0.9999 (Karras warmup) |
| FIM rate | 10% PSM |
| MTP lambda | 0.3 → 0.1 at 60% |
| gamma_freeze | 0.95 |
| beta2 | 0.95 |
| grad_clip | 1.0 |

---

## What NOT to Do

- Do NOT eval on raw weights — always use EMA
- Do NOT use gamma_freeze_ratio=0.80 — it's wrong, use 0.95
- Do NOT zero-init any projection feeding a normalization layer
- Do NOT call `to_empty()` without `init_weights()` + `_reinit_buffers()`
- Do NOT add FIM via fine-tuning — must train from token 1
- Do NOT run from `/workspace/nanoseek/nanoseek` — run from `/workspace/nanoseek`
- Do NOT skip `pytest tests/` before GPU training
- Do NOT modify `_init_weights()` without reading `docs/TRAINING_BUGS_POSTMORTEM.md`
- Do NOT FP8-ify the gate router — routing precision is sacred (see `fp8.py:_is_fp8_eligible`)
- Do NOT use `--fp8` on Ampere GPUs (A100/A6000) — no FP8 tensor cores, would be slower
- Do NOT evaluate with FP8 enabled — `disable_fp8()` context manager ensures BF16 eval
