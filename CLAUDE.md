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


  Gate 1 Smoke Test — What Actually Happens, Step by Step

  What Gate 1 IS

  Gate 1 is 100 training steps on a single GPU. ~5 minutes, ~$0.50. Its job is to answer one question: "Does
  the code work on a real GPU?"

  This is the first moment your model touches real hardware. Everything before this — 124 passing tests, the
  full model build — was CPU. The GPU introduces an entirely different set of failure modes.

  The Command

  python -m nanoseek.scripts.pre_train \
      --run gate1-smoke --scale ablation --seed 42 \
      --num-iterations 100 --eval-every 50 --save-every 100 \
      --device-batch-size 4 \
      --profile-steps 20,21,22 --profile-memory

  The first 6 flags are Gate 1 as documented. The last 2 are what we added — they piggyback on Gate 1 to
  establish a performance baseline at zero extra cost.

  Why Steps 20, 21, 22 (Not 0, 1, 2)

  Step 0-5:    torch.compile is lazily tracing the graph
               ↓ First forward triggers compilation (~30-60s)
               ↓ Kernels are being JIT'd, nothing is representative

  Step 5-15:   Compiled graph is running, but:
               ↓ CUDA caching allocator is still sizing pools
               ↓ cuBLAS is autotuning GEMM tile sizes
               ↓ First few steps have abnormal memory/timing

  Step 15-19:  Everything has stabilized
               ↓ Memory pools are sized
               ↓ cuBLAS has selected optimal kernels
               ↓ Step time is steady-state

  Step 20-22:  ← PROFILE HERE
               ↓ Representative of actual training performance
               ↓ 3 steps to get median (filter out outliers)

  Step 23-100: Normal training continues, profiling OFF

  Profiling steps 0-5 would measure compilation overhead, not training speed. Step 20 is the earliest point
  where your measurements reflect reality.

  What --profile-steps 20,21,22 Captures

  For each of these 3 steps, torch.profiler records:

  Step 20 timeline:
  ├── Forward ─────────────────────────────────────────────
  │   ├── Layer_0 ─────────────────────────────────────────
  │   │   ├── MLA ──── [aten::linear 0.3ms] [sdpa 1.2ms] [aten::linear 0.2ms]
  │   │   └── DenseFFN [aten::linear 0.4ms] [silu 0.1ms] [aten::linear 0.3ms]
  │   ├── Layer_1 ─────────────────────────────────────────
  │   │   ├── MLA ──── [...]
  │   │   └── MoE ─────────────────────────────────────────
  │   │       ├── MoE::gate       [aten::linear 0.1ms] [topk 0.05ms]
  │   │       ├── MoE::dispatch   [argsort 0.2ms] [bincount 0.05ms]
  │   │       ├── MoE::expert_compute  [bmm 2.1ms] ← likely the biggest
  │   │       ├── MoE::combine    [scatter_add 0.3ms]
  │   │       └── MoE::shared_expert  [linear 0.4ms] [silu] [linear]
  │   ├── Layer_2 ... Layer_15 (same structure)
  │   └── lm_head [aten::linear 0.5ms]
  ├── Loss [cross_entropy 0.2ms]
  ├── Backward ────────────────────────────────────────────
  │   └── (mirrors forward in reverse, usually ~2x forward time)
  └── Optimizer [muon step, adamw step]

  This becomes a Chrome trace file (runs/gate1-smoke/profiles/step_20.json). Open it in ui.perfetto.dev and
  you see a full flame chart of where every microsecond went.

  It also prints a table to stdout:

  [PROFILER] Step 20 — Top 20 CUDA ops:
  --------------------------  ----------  ----------  ----------
  Name                        CPU total   CUDA total  # Calls
  --------------------------  ----------  ----------  ----------
  MoE::expert_compute         12.3ms      8.7ms       14        ← 14 MoE layers
  MLA                         8.1ms       6.2ms       16        ← 16 layers
  MoE::dispatch               3.2ms       1.8ms       14
  aten::cross_entropy_loss    1.1ms       0.9ms       2
  MoE::shared_expert          2.8ms       2.1ms       14
  MoE::gate                   1.4ms       0.8ms       14
  MoE::combine                1.2ms       0.7ms       14
  ...

  What --profile-memory Captures

  On every log step (every 10 steps), it records 5 numbers:

  Step 20:
    mem/after_fwd_gb:      18.4 GB   ← activations held for backward
    mem/after_bwd_gb:      22.1 GB   ← + gradients allocated
    mem/after_optim_gb:    15.2 GB   ← activations freed, optimizer states updated
    mem/reserved_gb:       28.0 GB   ← what CUDA actually reserved (includes pools)
    mem/fragmentation_gb:  12.8 GB   ← reserved - allocated = wasted pool memory

  This goes to wandb, so you get a chart like:

  Memory (GB)
  30 ┤
     │    ╭── reserved (28 GB) ──────────────────────
  25 ┤    │
     │    │   ╭── after_bwd (22 GB) ─── peak working set
  20 ┤    │   │
     │    │   │   ╭── after_fwd (18 GB)
  15 ┤    │   │   │
     │    │   │   │   ╰── after_optim (15 GB) ── steady state
  10 ┤    │   │   │
     └────┴───┴───┴─────────────────────────────────── Step
          10  20  30  40  50  60  70  80  90  100

  The 6 Numbers You Record From Gate 1

  After 100 steps, you have your baseline. Write these down (they go to wandb automatically, but also note
  them):

  1. step_time:     _____ ms     (median of steps 20-22, from profiler)
  2. MFU:           _____%       (from wandb train/mfu)
  3. tok/s:         _______      (from wandb train/tok_per_sec)
  4. peak_memory:   _____ GB     (mem/after_bwd_gb — the actual peak)
  5. SDPA backend:  flash/math   (from profiler — is FlashAttention working?)
  6. top bottleneck: ________    (from profiler table — which component?)

  What These Numbers Tell You

  MFU = 25-35% → Normal for unoptimized MoE. Proceed to Tier 1 optimizations.
  MFU < 15%** → Something is fundamentally broken. Check SDPA backend (probably falling back to math O(S^2)
  instead of FlashAttention).
  **MFU > 40% → Surprisingly good. Skip Tier 1, go straight to training.

  SDPA = math → FlashAttention isn't dispatching for MLA's mismatched Q/K=192 vs V=128. This is the T1.1 fix
  (pad V). Worth 15-25% speedup alone.
  SDPA = flash → Good. FlashAttention is working.

  Top bottleneck = MoE::expert_compute (>40%) → Expected. This is where the actual FLOPs happen.
  Top bottleneck = MoE::dispatch (>20%) → Dispatch overhead is too high. T1.2 (.item() removal) + T2.1
  (torch.compile graph breaks).
  Top bottleneck = MLA (>30%) → Attention-bound. Check SDPA backend first.

  Peak memory = 22 GB on 80 GB GPU → 58 GB headroom. Increase batch size for higher MFU.
  Peak memory = 72 GB → Tight. Don't increase batch size. Consider gradient checkpointing.

  Fragmentation > 30% of reserved → The CUDA allocator is wasting memory.
  PYTORCH_ALLOC_CONF=expandable_segments:True (already set in your code) should help.

  The Decision Tree After Gate 1

  Gate 1 passes (loss drops, H_load > 4, no OOM)
  │
  ├── MFU > 35%?
  │   ├── YES → Skip perf optimization, proceed to Phase 1 (HP search)
  │   └── NO  → Check SDPA backend
  │             ├── math backend → Apply T1.1 (pad V), re-measure
  │             └── flash backend → Check profiler table
  │                  ├── dispatch > 20% → Apply T1.2 (.item() removal)
  │                  ├── expert_compute > 50% → Normal, try batch size increase
  │                  └── idle gaps visible in nsys → Apply T1.3 (grad norm deferral ← already done)
  │
  ├── After Tier 1 fixes, re-run --profile-steps, compare with baseline
  │   └── scripts/profile.py --compare gate1_before.json gate1_after.json
  │
  └── MFU > 35% now? → Proceed to Phase 1 (HP search)

  Why This Matters For Budget

  Your pipeline costs:

  Phase 0 (Gate 1):    $0.50    ← you are here
  Phase 1 (HP search): $30
  Phase 2 (IsoFLOP):   $200
  Phase 4 (1B grad):   $350
  ────────────────────────────
  Total:               ~$580

  A 2x MFU improvement (25% → 50%) means the $350 Phase 4 run finishes in half the wall time, or
  equivalently, costs half as much if you're paying by the hour. The $0.50 Gate 1 profile that identifies a
  15% speedup from fixing SDPA dispatch saves you $50+ downstream. That's 100x ROI on the profiling
  investment.

  That's why you profile at Gate 1 — it's the cheapest moment to discover the most impactful optimizations.
