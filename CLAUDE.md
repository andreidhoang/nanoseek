# NanoSeek — Architecture Rules & Context
## DeepSeek V3.2 MoE at Nano Scale
### Last updated: 2026-04-06 | Phase: Simplified, ready for GPU training

---

## What This Is

NanoSeek implements DeepSeek V3.2 at nano scale: MLA + MoE + MTP.
Following nanochat's philosophy: minimal, hackable, one dial controls everything.

---

## Critical Rules (Never Violate)

1. **EMA weights for ALL evaluation** — never eval on raw checkpoint weights
2. **gamma_freeze_ratio = 0.95** — the old 0.80 was wrong (V3 paper: 14.3T/14.8T)
3. **FIM from token 1** — never add FIM via fine-tuning
4. **H_load + I_spec logged for ALL runs** — primary scientific output
5. **Gradient clipping = 1.0 always** — MoE routing creates gradient spikes, clipping is mandatory
6. **Working directory = nanoseek/** — NOT nanoseek/nanoseek/

---

## Architecture Constants (from DeepSeek V3, invariant across scales)

```python
# MLA head dimensions
QK_NOPE_HEAD_DIM = 128    # query/key non-positional
QK_ROPE_HEAD_DIM = 64     # RoPE dimension
V_HEAD_DIM = 128          # value dimension
HEAD_DIM = 128            # num_heads = hidden_size / 128

# MLA compression ratios
Q_LORA_RATIO = 0.215      # q_lora_rank / hidden_size
KV_LORA_RATIO = 0.070     # kv_lora_rank / hidden_size

# MoE topology (nano-scale adjusted — see config.py for conservation law derivation)
N_ROUTED_EXPERTS = 16      # V3 original: 64. Conservation law: 16×1.5 = 64×0.375 = 24
NUM_EXPERTS_PER_TOK = 2    # top-k. Sparsity ratio: 2/16 = 8/64 = 12.5% (identical)
N_SHARED_EXPERTS = 2
N_GROUP = 4                # = TOPK_GROUP → pure top-k routing (no EP group constraint)
TOPK_GROUP = 4
MOE_INTER_RATIO = 1.5      # Nano-scale (V3 original: 0.375 at 7168h)

# Training
GAMMA_FREEZE_RATIO = 0.95
BETA2 = 0.95
GRAD_CLIP = 1.0
```

---

## MoE-Relevant Ablations (Phase 3)

Only ablations that test MoE-emergent behavior:

| Flag | Tests | Why it matters |
|---|---|---|
| `--aux-loss-type bias\|classic` | Aux-loss-free vs traditional balancing | THE core V3 innovation |
| `--no-seq-aux` | Sequence-level aux loss | Balance quality |
| `--no-shared-experts` | Shared expert necessity | DeepSeekMoE showed catastrophic without (loss 1.808→2.414) |
| `--no-mtp` | MTP contribution | V3 innovation, must quantify |
| `--num-experts/--top-k` | Expert topology | Granularity vs efficiency tradeoff |

**Deleted (not MoE-emergent):**
- `--inject-bad-batch` — dense model stability test, MoE already has natural routing spikes
- `--no-grad-clip` — MoE REQUIRES clipping (KAPATHY finding: gradient spikes from routing)

---

## File Map

```
nanoseek/                          <- CORE PACKAGE
  model.py                         <- MLA + MoE + MTP (2,157 lines)
  config.py                        <- One flat dataclass, 3 scales (418 lines)
  dataloader.py                    <- BOS-aligned packing + FIM 10% PSM
  dataset.py                       <- ClimbMix download + shard management
  optim.py                         <- MuonAdamW + DistMuonAdamW + muP
  tokenizer.py                     <- RustBPE tokenizer (vocab=32768)
  checkpoint_manager.py            <- Save/load model + optimizer + EMA
  common.py                        <- DDP init, COMPUTE_DTYPE, logging
  engine.py                        <- Inference with MLA KV cache

eval/
  domain_bpb.py                    <- Per-domain BPB (THE metric)
  information_metrics.py           <- I_spec + H_load (MoE science)
  moe_diagnostics.py               <- Dead experts, MTP acceptance

scripts/
  pre_train.py                     <- Main training script
  base_eval.py                     <- Benchmark evaluation
  chat_eval.py                     <- Interactive chat evaluation

tests/                             <- 124 tests passing
  test_nanoseek_model.py           <- Full model integration tests
  test_moe.py                      <- MoE unit tests
  test_mla_standalone.py           <- MLA tests
  test_moe_standalone.py           <- MoE standalone tests
  conftest.py                      <- Shared fixtures

docs/
  PAPER_ANALYSIS_V3_V32.md         <- Ground truth from papers (highest authority)
  TRAINING_BUGS_POSTMORTEM.md      <- 7 bugs found & fixed
  TRAINING_EXECUTION_PLAN.md       <- Step-by-step training guide
```

---

## Key Numbers

| | Anchor | Ablation (PRIMARY) | 1B |
|---|---|---|---|
| hidden_size | 768 | 1280 | 2048 |
| num_heads | 6 | 10 | 16 |
| num_layers | 16 | 16 | 16 |
| N_active | ~175M | ~410M | ~1.08B |
| N_total | ~730M | ~1.95B | ~4.75B |
| Tokens | 2.1B | 8.2B | 22B |
| Cost/run | ~$5 | ~$35 | ~$350 |

Shared: 64 experts top-8, 2 shared, vocab 32K, seq 4096, head_dim 128,
EMA 0.9999, FIM 10% PSM, MTP 0.3→0.1 at 60%, gamma_freeze 0.95.

---

## What NOT to Do

- Do NOT eval on raw weights — always use EMA
- Do NOT use gamma_freeze_ratio=0.80 — use 0.95
- Do NOT add FIM via fine-tuning — must train from token 1
- Do NOT zero-init any projection feeding a normalization layer
- Do NOT call `to_empty()` without `init_weights()` + `_reinit_buffers()`
- Do NOT modify `_init_weights()` without reading `docs/TRAINING_BUGS_POSTMORTEM.md`
- Do NOT disable gradient clipping for MoE — routing creates gradient spikes
- Do NOT use FP8 until torch.compile is validated (KAPATHY: FP8 without compile is 4x slower)

---

## Authority Hierarchy

```
1. docs/PAPER_ANALYSIS_V3_V32.md  <- ground truth from papers
2. ../CLAUDE.md                   <- project status & training commands
3. This file                      <- architecture rules & constants
4. nanoseek/model.py              <- the code itself
```
