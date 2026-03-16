# Frontier Model Systems Lab-in-a-Box: The NanoSeek Transformation Plan

## From Research Prototype to Production-Grade Frontier Pipeline

**Author**: Principal Research Engineer, Foundation Models Division
**Date**: February 2026
**Scope**: Transform NanoSeek into a reproducible mini-frontier-lab that demonstrates training → post-training → eval → serving — the complete stack that frontier labs pay $763K–$1.44M/year to build.

---

## 0) One-Sentence Goal

Build a single runnable repository that trains a DeepSeek V3 architecture model, improves it via SFT+DPO, evaluates it with regression-gated quality+safety metrics, and serves it with measured inference optimizations — proving end-to-end frontier model systems competency.

---

## 1) System Map

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    Frontier Model Systems Lab (FMS Lab)                           │
│                    Built on NanoSeek (DeepSeek V3.2 Architecture)                 │
└──────────────────────────────────────────────────────────────────────────────────┘

 LOOP A: CAPABILITY                  GATE                    LOOP B: SYSTEMS
 (Research)                          (Ship/Reject)           (Production)

┌─────────────────┐   ┌────────────────────┐   ┌───────────────────────────────┐
│ DATA             │   │ EVAL + GATE        │   │ SERVING                       │
│                  │   │                    │   │                               │
│ FineWeb-Edu      │   │ Quality:           │   │ Engine: NanoSeek native       │
│ (pre-train)      │──▶│  MMLU, GSM8K,      │──▶│  + KV cache (MLA 23x)        │
│                  │   │  HumanEval         │   │  + MTP speculative decode     │
│ UltraChat        │   │                    │   │  + continuous batching        │
│ (SFT)            │   │ Safety:            │   │  + INT8 quantization          │
│                  │   │  TruthfulQA,       │   │                               │
│ UltraFeedback    │   │  refusal accuracy  │   │ API: OpenAI-compatible        │
│ (DPO)            │   │                    │   │  FastAPI + SSE streaming      │
│                  │   │ Systems:           │   │                               │
│ Leakage checks   │   │  tok/s, p95 lat,   │   │ Perf: loadgen + reports       │
│ n-gram dedup     │   │  VRAM, $/1M tok    │   │  A/B per optimization lever   │
└─────────────────┘   │                    │   └───────────────────────────────┘
                       │ DECISION:          │
                       │ Ship if:           │
                       │  quality ≥ +X      │
                       │  safety ≥ baseline  │
                       │  p95 ≤ budget      │
                       │  cost ≤ budget     │
                       │ ELSE: reject       │
                       └────────────────────┘

 ┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────────┐
 │ Pre-train    │───▶│ SFT          │───▶│ DPO          │───▶│ Eval + Ship     │
 │ 22B tokens   │    │ UltraChat    │    │ UltraFeedback│    │ Gate Decision   │
 │ ~14hrs 8xH100│    │ ~2hrs        │    │ ~30min       │    │                 │
 └─────────────┘    └──────────────┘    └──────────────┘    └─────────────────┘
```

---

## 2) First-Principles Constraints

### 2.1 Physical Bottlenecks

| Constraint | Value | Source | Implication |
|-----------|-------|--------|-------------|
| H100 80GB VRAM | 80 GB | Hardware spec | NanoSeek 4.75B bf16 = 9.5GB model + ~35GB optimizer → fits with DDP+sharded opt |
| H100 BF16 TFLOPS | 989 | Hardware spec | MFU target ≥ 40% → 395 TFLOPS utilized |
| H100 memory bandwidth | 3.35 TB/s | Hardware spec | Decode bottleneck: 9.5GB model / 3.35 TB/s = 2.8ms per token minimum |
| NanoSeek KV cache/token | 175 dims × 16 layers × 2B = 5.6KB | Code: kv_lora_rank=143 + rope=32 | At 32K context: 179MB KV. Standard MHA would be 4.1GB. MLA is 23× cheaper. |
| NanoSeek active params | 1.08B | Config: estimated_active_params | FLOPs/token ≈ 6 × 1.08B = 6.48 GFLOPs (fwd+bwd) |
| Chinchilla tokens | 22B = 20 × 1.08B | Config: total_tokens | ~42K steps at 512K tokens/step |

### 2.2 Failure Modes (What Kills the Project)

| Failure | Detection | Mitigation |
|---------|-----------|------------|
| MoE expert collapse | Expert load Gini > 0.5 | bias-based balancing (gamma=0.001), already implemented |
| Training divergence | Loss NaN or spike > 3σ | Gradient clipping (1.0), checkpoint every 1K steps |
| Eval leakage | n-gram overlap with train | Dedup check before SFT/DPO data used |
| DPO mode collapse | KL divergence from ref → ∞ | β=0.1 constraint, KL monitoring |
| Inference OOM | VRAM > 80GB | INT8 quantization, batch size limits |
| Tail latency | p95 > 5× p50 | Request timeout, queue backpressure |
| Safety regression | Refusal accuracy drops | Gate blocks ship if safety < baseline |

### 2.3 What Must Be Measured (The Metrics Contract)

**Quality metrics (higher is better):**
- MMLU 5-shot accuracy (target: 30-40% for 1B)
- GSM8K 5-shot CoT accuracy (target: 10-20%)
- HumanEval pass@1 (target: 10-15%)
- MT-Bench score (post-training, target: 4-5/10)

**Safety metrics (must not regress):**
- TruthfulQA MC1 accuracy (baseline → ≥ baseline after post-training)
- Refusal accuracy on harmful prompts (custom 50-item set)

**Systems metrics (lower is better except throughput):**
- Throughput: tokens/sec under load
- p50 latency, p95 latency (ms/token)
- Peak VRAM (GB)
- Cost proxy: $/1M generated tokens

---

## 3) Spec-First Design

### 3.1 Repo Structure

The FMS Lab builds ON TOP of the existing NanoSeek codebase. We add, not replace.

```
nanoseek/                              # EXISTING (unchanged)
├── model/
│   ├── model.py                       # Core architecture (2038 lines)
│   ├── config.py                      # Configuration (1409 lines)
│   ├── optimizer/                     # Muon + DistAdamW
│   └── eval/                          # Checkpoint manager + basic eval
├── scripts/
│   ├── pre-train.py                   # Pre-training (1680 lines)
│   ├── dataloader.py                  # Streaming data
│   ├── scheduler.py                   # LR schedules
│   └── tokenizer.py                   # Tokenizer with chat templates
├── tests/                             # 145 tests
│
├── fms/                               # NEW: FMS Lab additions
│   ├── __init__.py
│   ├── serving/                       # LOOP B: Production serving
│   │   ├── engine.py                  # Inference engine with continuous batching
│   │   ├── server.py                  # FastAPI OpenAI-compatible server
│   │   ├── quantize.py                # INT8/INT4 post-training quantization
│   │   └── speculative.py             # MTP speculative decoding wrapper
│   ├── post_training/                 # LOOP A: Capability improvement
│   │   ├── sft.py                     # Supervised fine-tuning script
│   │   ├── dpo.py                     # Direct preference optimization
│   │   ├── chat_data.py               # Chat data loading + formatting
│   │   └── lora.py                    # LoRA for memory-efficient fine-tuning
│   ├── eval_harness/                  # GATE: Evaluation + regression
│   │   ├── runner.py                  # Unified eval runner
│   │   ├── benchmarks.py              # MMLU, GSM8K, HumanEval, TruthfulQA
│   │   ├── safety.py                  # Safety eval (refusal accuracy)
│   │   ├── leakage.py                 # Data leakage detection
│   │   └── gate.py                    # Ship/reject decision logic
│   ├── perf/                          # Systems measurement
│   │   ├── loadgen.py                 # Load generator (steady + burst)
│   │   ├── metrics.py                 # Perf metrics collection
│   │   └── report.py                  # Perf report generation
│   └── configs/                       # Reproducible experiment configs
│       ├── pretrain.yaml              # Pre-training config
│       ├── sft.yaml                   # SFT config
│       ├── dpo.yaml                   # DPO config
│       ├── serving.yaml               # Serving config
│       └── eval.yaml                  # Eval config
│
├── scripts/                           # NEW scripts (extend existing)
│   ├── train_sft.sh                   # One-command SFT
│   ├── train_dpo.sh                   # One-command DPO
│   ├── serve.sh                       # One-command serving
│   ├── eval_all.sh                    # One-command full eval
│   └── gate_check.sh                  # One-command ship gate
│
└── reports/                           # Generated artifacts
    ├── pretrain_loss.png              # Training loss curve
    ├── expert_loads.png               # MoE routing evolution
    ├── eval_baseline.json             # Base model eval results
    ├── eval_sft.json                  # SFT model eval results
    ├── eval_dpo.json                  # DPO model eval results
    ├── perf_baseline.json             # Baseline serving perf
    ├── perf_optimized.json            # Optimized serving perf
    └── gate_decision.json             # Final ship/reject decision
```

### 3.2 Config Schema

```yaml
# fms/configs/experiment.yaml — Single source of truth for one experiment
experiment:
  name: "nanoseek-1b-fms-v1"
  seed: 42
  hardware: "8xH100"

pretrain:
  model_size: "1b"
  total_tokens: 22_000_000_000
  batch_size: 524288
  seq_len: 4096
  lr: 3e-4
  lr_min: 3e-5
  optimizer: "muon+adamw"
  checkpoint_every: 1000

sft:
  dataset: "ultrachat-200k"
  epochs: 3
  lr: 2e-5
  batch_size: 4
  seq_len: 2048
  lora_rank: 16  # 0 = full fine-tune
  lora_targets: ["wq_a", "wq_b", "wkv_a", "wkv_b", "wo"]

dpo:
  dataset: "ultrafeedback-60k"
  epochs: 1
  lr: 5e-7
  beta: 0.1
  batch_size: 2
  seq_len: 2048
  ref_model: "sft"  # Use SFT checkpoint as reference

eval:
  benchmarks: ["mmlu", "gsm8k", "humaneval", "truthfulqa"]
  shots:
    mmlu: 5
    gsm8k: 5
    humaneval: 0
    truthfulqa: 0
  safety_set: "safety/refusal_50.jsonl"
  leakage_check: true

serving:
  quantization: "int8"  # none, int8, int4
  max_batch_size: 32
  max_seq_len: 4096
  speculative_decoding: true
  port: 8000

gate:
  quality_threshold: "+2% MMLU over baseline"
  safety_threshold: "≥ baseline refusal accuracy"
  p95_latency_budget: "200ms"
  cost_budget: "$5/1M tokens"
```

### 3.3 API Contracts

**Inference API (OpenAI-compatible):**
```
POST /v1/completions
POST /v1/chat/completions
GET  /health
GET  /metrics
```

**Internal interfaces:**
```python
# fms/eval_harness/runner.py
def run_eval(model_path: str, config: EvalConfig) -> EvalResults:
    """Run all benchmarks, return structured results."""

# fms/eval_harness/gate.py
def check_gate(baseline: EvalResults, candidate: EvalResults, 
               perf_baseline: PerfResults, perf_candidate: PerfResults,
               gate_config: GateConfig) -> GateDecision:
    """Return SHIP or REJECT with reasons."""

# fms/serving/engine.py
class InferenceEngine:
    def add_request(self, prompt: str, params: SamplingParams) -> RequestHandle: ...
    def step(self) -> List[CompletionResult]: ...
```

### 3.4 Acceptance Tests

| Test | Pass Criteria | Command |
|------|---------------|---------|
| Pre-train smoke | Loss < 8.0 after 500 steps | `python scripts/pre-train.py --num_iterations=500` |
| SFT train completes | No OOM, loss decreasing | `python -m fms.post_training.sft --config fms/configs/sft.yaml` |
| DPO train completes | chosen_reward > rejected_reward | `python -m fms.post_training.dpo --config fms/configs/dpo.yaml` |
| Eval reproducible | Same results ± 0.5% across 2 runs | `bash scripts/eval_all.sh && bash scripts/eval_all.sh` |
| Serving starts | /health returns 200 | `bash scripts/serve.sh && curl localhost:8000/health` |
| Perf baseline | tok/s > 0 and p95 < 10s | `python -m fms.perf.loadgen --config fms/configs/serving.yaml` |
| Gate runs | Produces gate_decision.json | `bash scripts/gate_check.sh` |
| Leakage check | 0 n-gram overlaps | `python -m fms.eval_harness.leakage --train_data ... --eval_data ...` |

---

## 4) Build Plan as PRs

### PR 1: FMS Lab Foundation (Day 1-2)
**Goal:** Create directory structure, configs, and entry points.
**Files:** `fms/__init__.py`, `fms/configs/*.yaml`, `scripts/*.sh`
**Command:** `pytest tests/ -v -m "not slow"` (existing tests still pass)
**Metric:** Zero regressions

### PR 2: Eval Harness — Framework + MMLU (Day 3-5)
**Goal:** Build eval runner that can measure MMLU on any NanoSeek checkpoint.
**Files:** `fms/eval_harness/runner.py`, `fms/eval_harness/benchmarks.py`
**Command:** `python -m fms.eval_harness.runner --model_path <ckpt> --benchmarks mmlu`
**Metric:** Produces accuracy number (expect ~25% for random init = chance on MMLU)

### PR 3: Eval Harness — GSM8K + HumanEval + TruthfulQA (Day 5-7)
**Goal:** Complete benchmark coverage.
**Files:** extend `fms/eval_harness/benchmarks.py`, add `fms/eval_harness/safety.py`
**Command:** `python -m fms.eval_harness.runner --benchmarks all`
**Metric:** Full eval table with 4 benchmarks

### PR 4: Eval Gate + Leakage Detection (Day 7-8)
**Goal:** Implement ship/reject gate logic and data leakage checker.
**Files:** `fms/eval_harness/gate.py`, `fms/eval_harness/leakage.py`
**Command:** `python -m fms.eval_harness.gate --baseline eval_base.json --candidate eval_sft.json`
**Metric:** Produces SHIP/REJECT with structured reasoning

### PR 5: Pre-Training Smoke Test (Day 8-10)
**Goal:** Run 1B token training with real data and capture artifacts.
**Files:** Scripts and monitoring additions
**Commands:**
```bash
python scripts/setup_data.py --dataset=fineweb-edu --tokens=2B
python scripts/pre-train.py --num_iterations=2000 --device_batch_size=1 --max_seq_len=512
```
**Metric:** Loss curve from ~11 to < 7, saved checkpoint, expert load logs

### PR 6: Full Pre-Training Run (Day 10-11, 14 hours compute)
**Goal:** Train NanoSeek-1B for 22B tokens on 8×H100.
**Command:** `torchrun --nproc_per_node=8 scripts/pre-train.py --target_param_data_ratio=20.0`
**Metric:** Final loss < 4.5, checkpoint saved, full training logs

### PR 7: Baseline Eval of Trained Model (Day 12)
**Goal:** Benchmark pre-trained model.
**Command:** `python -m fms.eval_harness.runner --model_path checkpoints/pretrain_final --benchmarks all`
**Metric:** MMLU > 28%, eval table published to `reports/eval_baseline.json`

### PR 8: SFT Training (Day 12-13)
**Goal:** Fine-tune on UltraChat for instruction following.
**Files:** `fms/post_training/sft.py`, `fms/post_training/chat_data.py`
**Commands:**
```bash
python -m fms.post_training.sft \
  --model_path checkpoints/pretrain_final \
  --dataset ultrachat-200k \
  --epochs 3 --lr 2e-5 --batch_size 4
```
**Metric:** SFT loss decreasing, checkpoint saved

### PR 9: DPO Training (Day 13-14)
**Goal:** Align model with preference optimization.
**Files:** `fms/post_training/dpo.py`
**Commands:**
```bash
python -m fms.post_training.dpo \
  --model_path checkpoints/sft_final \
  --ref_model_path checkpoints/sft_final \
  --dataset ultrafeedback-60k \
  --beta 0.1 --epochs 1 --lr 5e-7
```
**Metric:** chosen_reward > rejected_reward consistently, DPO loss decreasing

### PR 10: Post-Training Eval + Gate (Day 14-15)
**Goal:** Eval SFT and DPO models, run gate.
**Commands:**
```bash
python -m fms.eval_harness.runner --model_path checkpoints/sft_final --benchmarks all
python -m fms.eval_harness.runner --model_path checkpoints/dpo_final --benchmarks all
python -m fms.eval_harness.gate --baseline reports/eval_baseline.json --candidate reports/eval_dpo.json
```
**Metric:** Quality improvement documented, safety not regressed, gate decision

### PR 11: Inference Engine — Core (Day 15-17)
**Goal:** Build continuous batching engine with MLA KV cache.
**Files:** `fms/serving/engine.py`, `fms/serving/server.py`
**Command:** `python -m fms.serving.server --model_path checkpoints/dpo_final --port 8000`
**Metric:** /health returns 200, can generate text via /v1/completions

### PR 12: MTP Speculative Decoding (Day 17-18)
**Goal:** Integrate MTP modules for speculative decoding in serving.
**Files:** `fms/serving/speculative.py`
**Command:** `python -m fms.serving.server --model_path checkpoints/dpo_final --speculative true`
**Metric:** Acceptance rate > 50%, throughput improvement measured

### PR 13: INT8 Quantization (Day 18-19)
**Goal:** Post-training weight quantization for serving.
**Files:** `fms/serving/quantize.py`
**Command:** `python -m fms.serving.quantize --model_path checkpoints/dpo_final --dtype int8`
**Metric:** VRAM reduction from 9.5GB to ~5GB, quality loss < 1% on MMLU

### PR 14: Perf Harness + Load Testing (Day 19-20)
**Goal:** Measure serving performance under realistic load.
**Files:** `fms/perf/loadgen.py`, `fms/perf/metrics.py`, `fms/perf/report.py`
**Commands:**
```bash
python -m fms.perf.loadgen --target http://localhost:8000 --concurrency 16 --duration 60
```
**Metric:** Produces `reports/perf_optimized.json` with tok/s, p50, p95, VRAM

### PR 15: Final Report + README (Day 20-21)
**Goal:** Generate the "hire-me" results table and documentation.
**Files:** `README.md` updates, `reports/` generation scripts
**Metric:** Single table with before/after numbers

---

## 5) Implementation Details

### 5.1 Why Build a Native Serving Engine (Not vLLM/TRT-LLM)

**The key decision:** vLLM now supports DeepSeek V3 MLA natively. However, for this project, building a focused native engine is the correct choice:

1. **Educational value**: Understanding inference systems at the kernel level is the skill frontier labs pay for. Using vLLM as a black box teaches you nothing about KV cache management.

2. **MLA-specific optimizations**: NanoSeek's MLA stores only 175 values per token per layer (vs 4096 for standard MHA). A custom engine exploits this 23× advantage directly. vLLM's PagedAttention was designed for standard attention; adapting it for MLA's compressed format requires understanding both systems deeply.

3. **MTP integration**: NanoSeek's MTP modules provide native draft tokens for speculative decoding. No external draft model needed. A custom engine can tightly couple the MTP forward pass with the verification step.

4. **Portfolio signal**: "I built an inference engine" is a stronger signal than "I configured vLLM."

**Architecture of the native engine:**

```
Request → Queue → Scheduler → Batch Former → Model Forward → Token Sampler → Response
                      ↑              ↓              ↓
                 Backpressure   Prefill/Decode    KV Cache
                 (reject if      split logic     Manager
                  queue full)                  (MLA-specific:
                                                175 dims/tok)
```

The engine handles two phases:
- **Prefill** (compute-bound): Process full prompt, fill KV cache
- **Decode** (memory-bound): Generate one token at a time, read KV cache

MLA makes decode uniquely efficient because reading 175 values/token instead of 4096 means 23× less memory bandwidth per attention computation.

### 5.2 Instrumentation

Every request is traced with:
```python
@dataclass
class RequestTrace:
    request_id: str
    prompt_tokens: int
    generated_tokens: int
    queue_time_ms: float       # Time waiting in queue
    prefill_time_ms: float     # Time to process prompt
    decode_time_ms: float      # Time to generate tokens
    total_time_ms: float       # End-to-end
    tokens_per_sec: float      # Generation throughput
    peak_vram_mb: float        # Peak memory during request
    speculative_accepted: int  # MTP tokens accepted
    speculative_total: int     # MTP tokens proposed
```

Load generator produces realistic workloads:
```python
class LoadGenerator:
    def __init__(self, target_url: str):
        self.prompt_lengths = [32, 128, 512, 1024, 2048]  # Distribution
        self.output_lengths = [64, 128, 256, 512]
    
    def steady_load(self, concurrency: int, duration_sec: int): ...
    def burst_load(self, peak_concurrency: int, duration_sec: int): ...
```

### 5.3 The Three "Money Levers" of Inference Optimization

#### Lever 1: MLA KV Cache Management

**Mechanism:** Standard transformers cache K and V tensors at full dimension. MLA caches only the compressed representation (143 + 32 = 175 dims per token per layer). This is already implemented in `MultiHeadLatentAttention.forward()` via the `use_cache=True` path.

**What to measure:**
- Memory per active sequence at various context lengths
- Time to reconstruct full K/V from compressed representation (wkv_b projection)

**Expected tradeoff:** 23× less memory per sequence enables 23× more concurrent sequences (or longer contexts). The cost is the wkv_b decompression step during each attention computation.

**Implementation:** Already in `model/model.py` lines 312-332. The engine must manage block allocation for the compressed format.

```
Standard MHA cache per token per layer: 2 × 16 heads × 128 dim × 2 bytes = 8,192 bytes
MLA cache per token per layer:          (143 + 32) × 2 bytes              =   350 bytes
Savings: 23.4×

At 4K context, 16 layers:
  Standard: 4096 × 16 × 8,192 = 512 MB per sequence
  MLA:      4096 × 16 × 350   =  22 MB per sequence
```

#### Lever 2: Continuous Batching

**Mechanism:** Instead of waiting for all sequences in a batch to finish, process each sequence independently. When one finishes, immediately add a new request to the batch. This eliminates idle GPU cycles caused by waiting for the longest sequence.

**What to measure:**
- Throughput (tok/s) at various concurrency levels
- p95 latency vs throughput tradeoff curve

**Expected tradeoff:** 3-5× throughput improvement over static batching at the cost of implementation complexity. p95 latency may increase slightly for individual requests under heavy load.

**Implementation:** The engine maintains an active batch and a request queue. Each step:
1. Remove completed sequences from batch
2. Add new requests from queue (up to max_batch_size)
3. Run one decode step for all active sequences
4. Return any newly completed tokens

#### Lever 3: MTP Speculative Decoding

**Mechanism:** NanoSeek's MTP module generates draft tokens using a lightweight single-block transformer. The main model verifies these drafts in parallel (one forward pass processes multiple draft positions). Accepted tokens skip expensive decode steps.

**What to measure:**
- Acceptance rate (% of MTP draft tokens accepted by main model)
- Throughput improvement vs baseline autoregressive
- Quality verification (output distribution must be IDENTICAL to non-speculative)

**Expected tradeoff:** Based on 2025 benchmarks, speculative decoding achieves 1.5-2.5× throughput. MTP's advantage over external draft models is higher acceptance rates due to shared representations. DeepSeek reports 85-90% acceptance for the second token prediction.

**Implementation (from existing code):**
```python
# model/model.py line 1014 - MultiTokenPrediction.speculative_decode()
# Already returns (draft_tokens, draft_probs)
# 
# We need to add:
# 1. Verification step: run main model on [prompt + drafts]
# 2. Acceptance: compare p_main(token) vs p_draft(token)
# 3. Accept prefix where p_main ≥ p_draft
# 4. Sample correction token from adjusted distribution
```

The mathematical guarantee: speculative decoding produces EXACTLY the same distribution as standard autoregressive decoding. No quality loss. This is proven via the acceptance-rejection sampling theorem.

### 5.4 Post-Training Loop

#### SFT Recipe

```python
# fms/post_training/sft.py — Key design decisions

# WHY conversation masking:
# We only compute loss on ASSISTANT tokens, not user/system.
# If we train on user tokens, the model learns to generate user messages,
# which is the wrong distribution for a chatbot.

# WHY LoRA at rank 16:
# Full fine-tuning of 4.75B params requires ~57GB optimizer states.
# LoRA at rank 16 adds only ~33M trainable params (0.7% of total).
# For MoE: apply LoRA to shared experts + MLA projections, NOT all 64 routed experts.
# Reason: 64 × LoRA matrices would negate the memory savings.

# WHY lr=2e-5 (10× smaller than pre-training):
# SFT should adjust behavior, not rewrite representations.
# Too high LR → catastrophic forgetting of pre-training knowledge.

# WHY 3 epochs:
# Small SFT datasets (100K-200K examples) benefit from multiple passes.
# But monitor: if eval improves in epoch 1-2 then drops in epoch 3 → stop early.
```

#### DPO Recipe

```python
# fms/post_training/dpo.py — Key design decisions

# DPO LOSS (from Rafailov et al. 2023):
# L_DPO = -E[log σ(β(log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]
# 
# WHERE:
#   y_w = chosen (preferred) response
#   y_l = rejected response  
#   π_θ = current policy (model being trained)
#   π_ref = reference policy (frozen SFT model)
#   β = temperature (0.1 default — controls how much we deviate from reference)
#   σ = sigmoid function

# WHY β=0.1:
# β too low (0.01): model changes too aggressively → mode collapse
# β too high (1.0): model barely changes → no improvement
# β=0.1 is the empirically validated sweet spot for small models (HuggingFace SmolLM3 study)

# WHY reference model = SFT checkpoint:
# DPO measures improvement RELATIVE to a reference.
# Using pre-train as reference would penalize all SFT improvements.
# Using SFT as reference isolates the preference learning signal.

# MONITORING during DPO:
# 1. chosen_reward = β × log(π_θ(y_w)/π_ref(y_w)) — should increase
# 2. rejected_reward = β × log(π_θ(y_l)/π_ref(y_l)) — should decrease
# 3. reward_margin = chosen - rejected — should increase
# 4. KL(π_θ || π_ref) — should stay bounded (< 0.1 nats)
```

### 5.5 Distributed Systems

**Pre-training:** 8×H100, DDP with DistMuon+DistAdamW (existing, verified)

**SFT:** Single GPU with LoRA (LoRA reduces trainable params to 33M → fits easily)

**DPO:** Single GPU with LoRA. Need to hold reference model in memory alongside training model.
```
Memory budget for DPO on 1 GPU:
  Training model (bf16): 9.5 GB
  Reference model (bf16): 9.5 GB  (frozen, no optimizer states)
  LoRA optimizer states:  ~0.1 GB  (only 33M params)
  Activations:           ~5 GB
  Total: ~24 GB → fits on single H100
```

**Serving:** Single GPU. Model in INT8 = 4.75 GB. KV cache for 32 concurrent sequences at 4K = 32 × 22MB = 704 MB. Total: ~6 GB.

**Checkpointing:** Use existing `model/eval/checkpoint_manager.py` which supports per-rank optimizer state and metadata.

---

## 6) Benchmarks & Reports

### The Final Table (What Goes in the README)

```
┌──────────────────────────────────────────────────────────────────────────┐
│              NanoSeek-1B FMS Lab Results                                  │
├──────────────────┬──────────────┬──────────────┬──────────────┬──────────┤
│ Metric           │ Base Model   │ + SFT        │ + DPO        │ Δ Total  │
├──────────────────┼──────────────┼──────────────┼──────────────┼──────────┤
│ MMLU (5-shot)    │ 32.1%        │ 33.5%        │ 34.2%        │ +2.1%    │
│ GSM8K (5-shot)   │ 8.3%         │ 12.1%        │ 13.7%        │ +5.4%    │
│ HumanEval (p@1)  │ 6.7%         │ 10.2%        │ 11.0%        │ +4.3%    │
│ TruthfulQA       │ 38.2%        │ 39.1%        │ 41.5%        │ +3.3%    │
│ Refusal Accuracy  │ 62%          │ 78%          │ 85%          │ +23%     │
│ MT-Bench         │ —            │ 3.8          │ 4.5          │ —        │
├──────────────────┼──────────────┼──────────────┼──────────────┼──────────┤
│ GATE DECISION    │              │              │ ✅ SHIP       │          │
└──────────────────┴──────────────┴──────────────┴──────────────┴──────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│              Serving Performance (Single H100)                            │
├──────────────────┬──────────────┬──────────────┬──────────────┬──────────┤
│ Config           │ Baseline     │ + Cont.Batch │ + Spec.Dec.  │ + INT8   │
├──────────────────┼──────────────┼──────────────┼──────────────┼──────────┤
│ Throughput (t/s) │ 45           │ 180          │ 320          │ 410      │
│ p50 latency (ms) │ 22           │ 18           │ 12           │ 10       │
│ p95 latency (ms) │ 85           │ 42           │ 28           │ 22       │
│ Peak VRAM (GB)   │ 18           │ 18           │ 19           │ 11       │
│ $/1M tokens      │ $12          │ $3           │ $1.7         │ $1.3     │
├──────────────────┼──────────────┼──────────────┼──────────────┼──────────┤
│ Spec. Accept Rate│ —            │ —            │ 72%          │ 71%      │
│ MoE Experts/tok  │ 8            │ 8            │ 8            │ 8        │
│ KV Cache/seq     │ 22 MB        │ 22 MB        │ 22 MB        │ 11 MB    │
└──────────────────┴──────────────┴──────────────┴──────────────┴──────────┘

Note: Numbers above are TARGETS. Actual numbers will be filled after experiments.
Baseline = autoregressive, batch_size=1, BF16, no speculative decoding.
```

**Generation script:** `python -m fms.perf.report --results_dir reports/ > README_table.md`

---

## 7) Risks & Mitigations

| # | Risk | Detection | Mitigation | Severity |
|---|------|-----------|------------|----------|
| 1 | MoE expert collapse during pre-training | Expert load Gini > 0.5 at any checkpoint | Already have bias-based balancing. If Gini > 0.5 at step 1000, increase gamma from 0.001 to 0.005 | HIGH |
| 2 | MTP module shape mismatch at full config | Runtime error during training | Run 100-step smoke test at full config before committing to 14hr run | HIGH |
| 3 | SFT catastrophic forgetting | MMLU drops > 2% after SFT | Use LoRA (rank 16), add 5% pre-training data replay to SFT mix | MEDIUM |
| 4 | DPO mode collapse | KL divergence > 0.5 nats from reference | Reduce β from 0.1 to 0.05, add KL penalty term | MEDIUM |
| 5 | Eval data leaks into SFT/DPO training data | n-gram overlap > 0 | Run leakage check BEFORE training; block if overlaps found | HIGH |
| 6 | Inference OOM with concurrent requests | VRAM > 80GB | Set max_batch_size dynamically based on available memory | LOW |
| 7 | Speculative decoding slower than baseline | Throughput < autoregressive baseline | Disable speculation; investigate acceptance rate | LOW |
| 8 | Pre-training compute overrun | Training takes > 20 hours | Budget 2× ($600); checkpoint every 1K steps for restart | MEDIUM |
| 9 | Quantization quality regression | MMLU drops > 2% with INT8 | Try per-channel quantization; fall back to BF16 serving | LOW |
| 10 | Training data download fails | setup_data.py errors | Mirror data to S3/GCS; support multiple download sources | LOW |

---

## 8) Teach-Back (Feynman)

### Explaining to a Smart 15-Year-Old

Imagine you want to build a restaurant. Here's what we're doing:

**Step 1: Grow the ingredients (Pre-training)**
We feed the AI 22 billion words from the internet. This takes 14 hours on 8 very expensive computers. After this, the AI understands language — it can complete sentences, but it doesn't know how to have a conversation yet. This is like growing raw ingredients on a farm.

**Step 2: Learn the recipes (SFT — Supervised Fine-Tuning)**
We show the AI 200,000 examples of good conversations: "When someone asks X, a helpful answer looks like Y." After 2 hours of studying these, the AI can follow instructions and have basic conversations. This is like a chef learning recipes.

**Step 3: Develop taste (DPO — Direct Preference Optimization)**
We show the AI pairs of answers: "This one is better than that one." After studying 60,000 such pairs, the AI learns to prefer helpful, honest, harmless answers. This is like a chef developing a refined palate.

**Step 4: Quality control (Evaluation Gate)**
Before we open the restaurant, we test the food. We have specific dishes that must taste at least as good as before (no safety regression), and the new special dishes must be better (quality improvement). If any test fails, we go back to the kitchen.

**Step 5: Open the restaurant (Serving)**
We need to serve many customers at once, not just one. So we build:
- **A waiting system** (continuous batching): instead of making one dish at a time, we cook multiple dishes simultaneously
- **A prediction system** (speculative decoding): we guess what the next step of cooking will be before we're asked, so we can work ahead
- **Efficient storage** (MLA KV cache): we remember conversations using a compressed format that takes 23× less space, so we can remember more conversations at once
- **Portion control** (INT8 quantization): we use slightly less precise measurements to cook twice as fast, with barely noticeable difference in taste

**Why this matters:**
The people who can build ALL of these steps — from growing ingredients to running the restaurant — are the most valuable engineers in AI. They understand both the science (how the AI learns) and the engineering (how to make it fast and reliable). That's why this project exists: to prove one person can do all of it.

---

## 9) Timeline Summary

```
Week 1  │ Eval harness + data pipeline verification + 1B token smoke test
Week 2  │ Full 22B token pre-training run (14 hrs compute)
Week 3  │ SFT + DPO + post-training eval + gate check
Week 4  │ Inference engine + speculative decoding + quantization
Week 5  │ Perf harness + load testing + final report
Week 6  │ README, blog post, model release, demo deployment
```

Total compute cost: ~$400
Total engineering time: ~6 weeks part-time or ~3 weeks full-time
Deliverable: A complete, reproducible frontier model systems lab with measured results.

---

## 10) The Ship Gate: One Decision That Governs Everything

Every PR, every experiment, every optimization passes through one question:

> **"Does this improve quality without regressing safety, and does it meet the systems budget?"**

If yes → SHIP.
If no → REJECT with specific failure reason.
If uncertain → Design the smallest experiment to resolve uncertainty.

This discipline — measurement-first, gate-governed, regression-protected — is what separates a frontier lab from a research notebook. It is the single most important engineering pattern in this entire project.

---

*"The purpose of computing is insight, not numbers."*
— Richard Hamming

*Applied to frontier ML: The purpose of a model systems lab is evidence, not architecture. Train the model. Measure the results. Ship what works.*
