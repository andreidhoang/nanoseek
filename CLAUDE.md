# NanoSeek — Project Context
## DeepSeek V3.2 at Nano Scale — Simplified to First Principles
### Last updated: 2026-04-06 | Phase: Simplified, 124 tests passing, ready for GPU training

---

## What This Project Is

NanoSeek is a from-scratch reimplementation of DeepSeek V3.2 at 1.08B active / 4.75B total
parameters, following nanochat's philosophy: minimal, hackable, no features that haven't been
tested on GPU yet.

**Three innovations that matter:**
1. **MLA** (Multi-head Latent Attention) — 23x KV cache compression
2. **MoE** (Mixture of Experts) — aux-loss-free sigmoid routing, 64 experts top-8
3. **MTP** (Multi-Token Prediction) — speculative decoding during training

Everything else (FP8, RL, data curation, scaling law fitting) was stripped. Add back when needed.

---

## Current Status (April 6, 2026)

```
Phase 1 (COMPLETE): Model (MLA + MoE + MTP) — 2,157 lines
Phase 2 (COMPLETE): Training infra (EMA, FIM, muP, MuonAdamW, eval, checkpoints)
Phase 3 (READY):    Code complete. 124 tests pass. Needs GPU + data.
```

**What just happened**: Major simplification following nanochat's design principles.
- Config: 7 dataclasses → 1 flat dataclass (750 → 418 lines)
- Model: Removed KDA/DSA dead code (2,228 → 2,157 lines)
- Training: Removed bloat (2,126 → 1,569 lines)
- Deleted: FP8, KDA, data_curation, alignment, benchmarks, scaling_law_lab, 50+ markdown files
- Total: 21K → 12K Python lines, 48K → 2K markdown lines

**What to do next**: Rent GPU → download ClimbMix → Gate 1 smoke test → watch val_bpb go down.

---

## Project Structure

```
nanoseek/                              ← Project root (run commands from here)
├── nanoseek/                          ← Core package
│   ├── model.py                       ← MLA + MoE + MTP (2,157 lines)
│   ├── config.py                      ← One flat dataclass, 3 scales (418 lines)
│   ├── dataloader.py                  ← BOS-aligned packing + FIM 10% PSM
│   ├── dataset.py                     ← ClimbMix download + shard management
│   ├── optim.py                       ← MuonAdamW + DistMuonAdamW + muP
│   ├── tokenizer.py                   ← RustBPE tokenizer (vocab=32768)
│   ├── checkpoint_manager.py          ← Save/load model + optimizer + EMA
│   ├── common.py                      ← DDP init, device detection, logging
│   └── engine.py                      ← Inference with MLA KV cache
├── eval/
│   ├── domain_bpb.py                  ← Per-domain BPB (THE metric)
│   ├── information_metrics.py         ← I_spec + H_load (MoE science)
│   └── moe_diagnostics.py            ← Dead experts, MTP acceptance
├── scripts/
│   ├── pre_train.py                   ← MAIN TRAINING SCRIPT (1,569 lines)
│   ├── base_eval.py                   ← Benchmark evaluation
│   └── chat_eval.py                   ← Interactive chat evaluation
├── tests/                             ← 124 tests passing, 8 skipped
│   ├── test_nanoseek_model.py         ← Full model integration tests
│   ├── test_moe.py                    ← MoE unit tests
│   ├── test_mla_standalone.py         ← MLA tests
│   ├── test_moe_standalone.py         ← MoE standalone tests
│   └── conftest.py                    ← Shared fixtures
├── docs/
│   ├── PAPER_ANALYSIS_V3_V32.md       ← Ground truth from papers (highest authority)
│   ├── TRAINING_BUGS_POSTMORTEM.md    ← 7 bugs found & fixed
│   ├── TRAINING_EXECUTION_PLAN.md     ← Step-by-step training guide
│   ├── SCALING_LAB_PLAN.md            ← Research plan
│   └── PROJECT_STATE.md               ← Status dashboard
└── CLAUDE.md                          ← Detailed architecture rules
```

---

## Critical Rules (Never Violate)

1. **EMA weights for ALL evaluation** — never eval on raw checkpoint weights
2. **gamma_freeze_ratio = 0.95** — the old 0.80 was wrong
3. **FIM from token 1** — never add FIM via fine-tuning
4. **H_load + I_spec logged for ALL runs** — primary scientific output
5. **Run all tests before GPU training** — `pytest tests/ -v`
6. **Working directory = nanoseek/** — NOT nanoseek/nanoseek/

---

## Training Commands

```bash
# Gate 1 smoke test (100 steps)
python -m nanoseek.scripts.pre_train \
    --run gate1-smoke --scale ablation --seed 42 \
    --num-iterations 100 --eval-every 50 --save-every 100 --device-batch-size 4

# HP search: 6 runs (500 steps each)
for mlr in 0.005 0.01 0.02; do
  for elr in 0.2 0.5; do
    python -m nanoseek.scripts.pre_train \
        --run "hp-mlr${mlr}-elr${elr}" --scale ablation \
        --matrix-lr $mlr --embedding-lr $elr \
        --num-iterations 500 --eval-every 100 --save-every -1 --seed 42
  done
done

# Full ablation (8.2B tokens, ~$35)
python -m nanoseek.scripts.pre_train \
    --run nanoseek-ablation-v1 --scale ablation \
    --matrix-lr <best> --embedding-lr <best> \
    --eval-every 250 --save-every 1000 --device-batch-size 16 --seed 42

# NanoSeek-1B graduation (22B tokens, 8xH100, ~$350)
torchrun --nproc_per_node=8 -m nanoseek.scripts.pre_train \
    --run nanoseek-1b-v1 --scale 1b \
    --matrix-lr <best> --embedding-lr <best> \
    --eval-every 500 --save-every 2000 --device-batch-size 16 --seed 42

# Download training data
python -m nanoseek.nanoseek.dataset -n 170
```

---

## Key Numbers

| Parameter | Ablation | 1B |
|-----------|----------|-----|
| N_active | ~410M | 1.08B |
| N_total | ~1.95B | 4.75B |
| d_model | 1280 | 2048 |
| Layers | 16 | 16 |
| Training tokens | 8.2B | 22B |
| Cost/run | ~$35 | ~$350 |

Shared: 64 experts top-8, vocab 32K, seq 4096, EMA 0.9999, FIM 10%, MTP 0.3→0.1 at 60%, gamma_freeze 0.95, beta2 0.95, grad_clip 1.0.

---

## What NOT to Do

- Do NOT eval on raw weights — always use EMA
- Do NOT use gamma_freeze_ratio=0.80 — use 0.95
- Do NOT add FIM via fine-tuning — must train from token 1
- Do NOT zero-init any projection feeding a normalization layer
- Do NOT call `to_empty()` without `init_weights()` + `_reinit_buffers()`
- Do NOT modify `_init_weights()` without reading `docs/TRAINING_BUGS_POSTMORTEM.md`
