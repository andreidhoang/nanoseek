# NanoSeek Production Implementation Master Plan

## Principal Engineer's Assessment — February 2026

**Author**: Distinguished AI Researcher & Principal Engineer, Foundation Models Division
**Scope**: Complete production-readiness engineering plan for NanoSeek (DeepSeek V3.2 at nano scale)
**Status**: Educational implementation → Production training & serving

---

## Executive Summary

NanoSeek has a solid architectural foundation implementing all four DeepSeek V3.2 innovations (MLA, MoE, MTP, DSA). However, significant engineering work is required to bridge the gap from educational PyTorch code to production-grade training on real GPU clusters and serving to users. This document series provides the complete implementation blueprint.

**Current State**: All core architecture implemented in pure PyTorch. Tests passing. Forward/backward verified.

**Target State**: Production training on 8×H100, inference serving with speculative decoding, post-training alignment.

---

## Gap Analysis Summary

| Component | Current State | Production Need | Priority | Doc |
|-----------|--------------|-----------------|----------|-----|
| Attention Kernels | `torch.matmul` | Flash Attention / Triton | **CRITICAL** | `02` |
| MoE Dispatch | Python loop | Fused CUDA/Triton kernels | **CRITICAL** | `03` |
| Data Pipeline | Basic streaming | Production prefetch + sharding | **CRITICAL** | `01` |
| Distributed Training | DDP only | FSDP + Expert Parallelism | **CRITICAL** | `04` |
| Mixed Precision | Basic BF16 | FP8 + dynamic loss scaling | **IMPORTANT** | `05` |
| Training Orchestration | Basic multi-phase | Robust phase transitions + monitoring | **IMPORTANT** | `06` |
| Inference Serving | Basic `generate()` | Continuous batching + PagedAttention | **CRITICAL** | `07` |
| Speculative Decoding | MTP modules exist | Full draft-verify pipeline | **IMPORTANT** | `08` |
| Post-Training | None | SFT + DPO + RLVR | **CRITICAL** | `09` |
| Evaluation | Basic loss eval | Full benchmark suite | **IMPORTANT** | `10` |
| **Forward/Backward Walkthrough** | N/A | End-to-end tensor trace | **ESSENTIAL** | `11` |
| **Complete Training Step** | N/A | One iteration dissected | **ESSENTIAL** | `12` |
| **Integration Guide** | N/A | Exact diffs to wire in | **ESSENTIAL** | `13` |
| **Numerical Stability** | N/A | Init, precision, gradient health | **ESSENTIAL** | `14` |
| API Server | None | OpenAI-compatible FastAPI + SSE | **CRITICAL** | `15` |
| Quantization (Serving) | None | INT8/INT4 post-training quantization | **IMPORTANT** | `16` |
| Eval Gate | None | Ship/reject decision system | **CRITICAL** | `17` |
| Data Leakage Detection | None | N-gram contamination checking | **CRITICAL** | `18` |
| Perf Load Testing | None | Load generator + metrics + reports | **IMPORTANT** | `19` |
| Safety Evaluation | Basic in Doc 10 | Dedicated refusal accuracy framework | **CRITICAL** | `20` |
| Experiment Configs | Scattered argparse | YAML schema + reproducibility | **IMPORTANT** | `21` |
| Pipeline Orchestration | None | End-to-end shell scripts + DAG | **ESSENTIAL** | `22` |

---

## Reading Order

### Path 1: "I want to train this model on real GPUs"
1. `01_DATA_PIPELINE_PRODUCTION.md` — Get real data flowing
2. `04_DISTRIBUTED_TRAINING_FSDP.md` — Scale to multi-GPU
3. `02_FLASH_MLA_ATTENTION_KERNEL.md` — Make attention fast
4. `03_FUSED_MOE_EXPERT_PARALLELISM.md` — Make MoE fast
5. `05_FP8_MIXED_PRECISION.md` — Maximize GPU utilization
6. `06_MULTI_PHASE_TRAINING_ORCHESTRATION.md` — Orchestrate full training

### Path 2: "I want to serve this model to users"
1. `07_INFERENCE_ENGINE.md` — Production serving infrastructure
2. `08_SPECULATIVE_DECODING_MTP.md` — Fast generation with MTP
3. `15_OPENAI_COMPATIBLE_API_SERVER.md` — FastAPI + SSE streaming API
4. `16_QUANTIZATION_INT8_INT4_SERVING.md` — INT8/INT4 for efficient serving
5. `09_POST_TRAINING_SFT_DPO_RLVR.md` — Make it chat-capable
6. `10_EVALUATION_BENCHMARKS.md` — Measure quality
7. `19_PERFORMANCE_LOAD_TESTING_HARNESS.md` — Benchmark serving performance

### Path 3: "I want to understand the model deeply first"
1. `11_COMPLETE_FORWARD_BACKWARD_WALKTHROUGH.md` — Trace every tensor through the full model
2. `12_COMPLETE_TRAINING_STEP.md` — One complete training iteration dissected
3. `14_WEIGHT_INIT_NUMERICAL_STABILITY.md` — Why training doesn't blow up
4. Then read component docs (01-10) for production optimizations
5. `13_PRODUCTION_INTEGRATION_GUIDE.md` — How to wire everything in

### Path 4: "I want to deploy to production (FMS Lab pipeline)"
1. `21_EXPERIMENT_CONFIGS_REPRODUCIBILITY.md` — Config system for reproducible experiments
2. `18_DATA_LEAKAGE_CONTAMINATION_DETECTION.md` — Ensure eval integrity
3. `20_SAFETY_EVALUATION_FRAMEWORK.md` — Safety testing framework
4. `17_EVAL_GATE_SHIP_REJECT.md` — Ship/reject decision system
5. `22_END_TO_END_PIPELINE_ORCHESTRATION.md` — Full pipeline orchestration

### Path 5: "I want to understand everything end-to-end"
Read 00 → 11 → 14 → 01 → 02 → 03 → 04 → 05 → 06 → 12 → 07 → 08 → 15 → 16 → 09 → 10 → 20 → 18 → 17 → 19 → 21 → 22 → 13

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NanoSeek Production Stack                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │ Pre-Training │───▶│Post-Training│───▶│  Inference   │             │
│  │   Pipeline   │    │  Pipeline   │    │   Engine     │             │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘             │
│         │                   │                   │                    │
│  ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐             │
│  │ Data        │    │ SFT         │    │ Continuous   │             │
│  │ Pipeline    │    │ DPO/RLVR    │    │ Batching     │             │
│  │ (Doc 01)    │    │ (Doc 09)    │    │ (Doc 07)     │             │
│  └──────┬──────┘    └─────────────┘    └──────┬──────┘             │
│         │                                      │                    │
│  ┌──────▼──────────────────────────────────────▼──────┐             │
│  │              Core Model Engine                      │             │
│  │  ┌──────────┐ ┌──────────┐ ┌──────┐ ┌──────────┐  │             │
│  │  │Flash MLA │ │Fused MoE │ │ MTP  │ │   DSA    │  │             │
│  │  │(Doc 02)  │ │(Doc 03)  │ │Spec. │ │  Sparse  │  │             │
│  │  │          │ │          │ │Decode│ │  Attn    │  │             │
│  │  │          │ │          │ │(08)  │ │          │  │             │
│  │  └──────────┘ └──────────┘ └──────┘ └──────────┘  │             │
│  └────────────────────────┬───────────────────────────┘             │
│                           │                                         │
│  ┌────────────────────────▼───────────────────────────┐             │
│  │           Infrastructure Layer                      │             │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │             │
│  │  │ FSDP +   │ │ FP8 +    │ │ Eval +   │            │             │
│  │  │ Expert   │ │ Mixed    │ │ Monitor  │            │             │
│  │  │ Parallel │ │ Precision│ │          │            │             │
│  │  │ (Doc 04) │ │ (Doc 05) │ │ (Doc 10) │            │             │
│  │  └──────────┘ └──────────┘ └──────────┘            │             │
│  └────────────────────────────────────────────────────┘             │
│                                                                     │
│  ┌────────────────────────────────────────────────────┐             │
│  │     Training Orchestration (Doc 06)                 │             │
│  │     Phase 1 (Dense) → Phase 2 (Sparse) → Serving   │             │
│  └────────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## File/Folder Structure for Production Codebase

Each implementation doc specifies exactly where code should be placed:

```
nanoseek/
├── model/
│   ├── model.py                    # Core model (existing, enhanced)
│   ├── config.py                   # Configuration (existing, enhanced)
│   ├── kernels/                    # NEW: Custom kernels
│   │   ├── flash_mla.py           # Doc 02: Flash MLA Triton kernel
│   │   ├── fused_moe.py           # Doc 03: Fused MoE dispatch kernel
│   │   ├── fused_rope.py          # Doc 02: Fused RoPE kernel
│   │   └── quantization.py        # Doc 05: FP8 kernels
│   ├── optimizer/                  # Existing + enhanced
│   │   ├── muon.py                # Enhanced with better comm overlap
│   │   └── adamw.py               # Enhanced with ZeRO-3 option
│   ├── eval/                       # Existing + enhanced
│   │   ├── benchmarks/            # Doc 10: Benchmark suite
│   │   │   ├── mmlu.py
│   │   │   ├── humaneval.py
│   │   │   └── gsm8k.py
│   │   └── monitoring.py          # Doc 06: Training monitors
│   └── serving/                    # NEW: Inference engine
│       ├── engine.py              # Doc 07: Continuous batching engine
│       ├── paged_attention.py     # Doc 07: PagedAttention KV cache
│       ├── speculative.py         # Doc 08: Speculative decoding
│       ├── server.py              # Doc 07: FastAPI server
│       └── quantize.py            # Doc 05: Inference quantization
├── scripts/
│   ├── pre-train.py               # Enhanced with FSDP (Doc 04)
│   ├── dataloader.py              # Enhanced production pipeline (Doc 01)
│   ├── sft.py                     # NEW: Doc 09 - SFT training
│   ├── dpo.py                     # NEW: Doc 09 - DPO training
│   ├── rlvr.py                    # NEW: Doc 09 - RLVR training
│   ├── evaluate.py                # NEW: Doc 10 - Benchmark runner
│   └── serve.py                   # NEW: Doc 07 - Serving launcher
├── fms/                               # NEW: FMS Lab additions (Docs 15-22)
│   ├── serving/
│   │   ├── server.py              # Doc 15: FastAPI OpenAI-compatible server
│   │   └── quantize.py            # Doc 16: INT8/INT4 post-training quantization
│   ├── eval_harness/
│   │   ├── gate.py                # Doc 17: Ship/reject decision logic
│   │   ├── leakage.py             # Doc 18: Data leakage detection
│   │   └── safety.py              # Doc 20: Safety evaluation framework
│   ├── perf/
│   │   ├── loadgen.py             # Doc 19: Load generator
│   │   ├── metrics.py             # Doc 19: Perf metrics collection
│   │   └── report.py              # Doc 19: Report generation
│   └── configs/
│       ├── schema.py              # Doc 21: Config validation schemas
│       ├── loader.py              # Doc 21: Config loading + merging
│       ├── pretrain.yaml          # Doc 21: Pre-training config
│       ├── sft.yaml               # Doc 21: SFT config
│       ├── dpo.yaml               # Doc 21: DPO config
│       ├── serving.yaml           # Doc 21: Serving config
│       └── eval.yaml              # Doc 21: Eval config
├── scripts/
│   ├── train_sft.sh               # Doc 22: One-command SFT
│   ├── train_dpo.sh               # Doc 22: One-command DPO
│   ├── serve.sh                   # Doc 22: One-command serving
│   ├── eval_all.sh                # Doc 22: One-command full eval
│   ├── gate_check.sh              # Doc 22: One-command gate check
│   ├── run_pipeline.sh            # Doc 22: Master pipeline script
│   └── generate_report.py         # Doc 22: Final Table generation
└── tests/
    ├── test_kernels.py            # NEW: Kernel correctness tests
    ├── test_serving.py            # NEW: Serving tests
    └── benchmarks/                # NEW: Performance benchmarks
        ├── bench_attention.py
        ├── bench_moe.py
        └── bench_inference.py
```

---

## Compute Budget Analysis (NanoSeek-1B on 8×H100)

| Phase | Tokens | Compute (H100-hrs) | Wall Time | Cost |
|-------|--------|--------------------:|----------:|-----:|
| Pre-training Phase 1 (Dense, 4K ctx) | 17.6B | ~90 | ~11h | ~$225 |
| Pre-training Phase 2 (Sparse, 8K ctx) | 4.4B | ~22 | ~3h | ~$55 |
| SFT | 1B | ~5 | ~40min | ~$12 |
| DPO | 200M | ~2 | ~15min | ~$5 |
| **Total** | **~23B** | **~119** | **~15h** | **~$300** |

---

## Key Engineering Decisions

### Why Flash Attention for MLA (not just standard Flash Attention)?
MLA's decoupled Q/K structure (nope + rope components) requires a custom attention pattern. Standard Flash Attention expects uniform Q/K dimensions. We need a Triton kernel that handles the split dimensions natively.

### Why Fused MoE Kernels?
The current Python-based token-centric dispatch has three performance bottlenecks: (1) `argsort` on GPU, (2) Python loop over experts, (3) `scatter_add_` for unpermute. A fused Triton kernel eliminates all three.

### Why FSDP over DeepSpeed ZeRO?
PyTorch-native FSDP integrates cleanly with the existing codebase, supports mixed precision natively, and avoids the DeepSpeed dependency. For NanoSeek-1B, FSDP with ZeRO-2 is sufficient.

### Why DPO over RLHF for initial alignment?
DPO achieves equivalent alignment quality with 40% less compute and no reward model training. For NanoSeek's budget (~$300 total), DPO is the rational choice. RLHF can be added later for safety-critical applications.

---

## Document Conventions

Each implementation doc follows this structure:

1. **Problem Statement** — What gap this addresses, why it matters
2. **First Principles Analysis** — Why this approach, what alternatives exist
3. **Architecture & Data Flow** — ASCII diagrams with tensor shapes
4. **Production Code** — Complete, tested, ready to integrate
5. **File Placement** — Exact paths in the codebase
6. **Integration Guide** — How to wire into existing code
7. **Verification** — How to test correctness
8. **Performance Targets** — Expected speedups/memory savings
9. **Gotchas & Edge Cases** — What breaks at scale

---

*"The difference between a research prototype and a production system is not cleverness — it's the thousand small engineering decisions that compound into reliability."*

— Principal Engineer's Note, Foundation Models Division, 2026
