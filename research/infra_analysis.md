# RL Pipeline Infrastructure Analysis — Kimi K2.5 × GLM-5 × MiniMax M2.7
## Senior Systems Engineer Assessment
### Date: 2026-03-24 | Analyst Perspective: Production-Grade Infrastructure

---

**Analysis Methodology**: Each pipeline evaluated from verified architectural details (papers, code, blogs). Every claim labeled:
- **[FACT]** — directly stated in official source
- **[INFERRED-STRONG]** — strongly implied by published data
- **[INFERRED-WEAK]** — reasonable inference from industry practice
- **[UNKNOWN]** — not publicly disclosed

---

# Pipeline 1: Kimi K2.5 (Moonshot AI)

## 1. FACTS

| Component | Value | Source |
|-----------|-------|--------|
| Model params | 1.04T total / 32.6B active | [FACT] K2 paper |
| MoE topology | 384 routed + 1 shared, top-8 | [FACT] K2 paper |
| Attention | MLA (23× KV compression) | [FACT] K2 paper |
| Layers | 61 (1 dense + 60 MoE) | [FACT] K2 paper |
| Pre-train tokens | 15.5T | [FACT] K2 paper |
| GPU type | NVIDIA H800 | [FACT] K2 paper |
| Parallelism | PP16 × EP16 × ZeRO-1 | [FACT] K2 paper |
| Model-parallel group | 256 GPUs (~30GB/GPU) | [FACT] K2 paper |
| Inference engine | vLLM | [FACT] K1.5 paper |
| Training engine | Megatron | [FACT] K1.5 paper |
| Checkpoint transfer | RDMA (Mooncake), <30s full param broadcast | [FACT] K2 paper |
| RL optimizer | Muon (same as pre-training) | [FACT] K2 paper |
| Transition time | <1 min (train→infer), ~10s (infer→train) | [FACT] K2 paper |
| Code sandboxes | 10K+ concurrent K8s instances | [FACT] K2 paper |
| RL GPU count | ~256+ GPUs | [INFERRED-STRONG] from "multiples of 32 nodes" |
| RL training duration | Unknown | [UNKNOWN] |
| RL training cost | Unknown | [UNKNOWN] |

## 2. SYSTEM DESIGN

### Actor-Learner Topology: Colocated Hybrid

```
                    ┌──────────────────────────────┐
                    │     Centralized Controller     │
                    └──────────┬───────────────────┘
                               │ orchestrates
                    ┌──────────▼───────────────────┐
                    │   GPU Worker Pool (≥256 GPUs)  │
                    │                                │
                    │  ┌─────────────────────────┐   │
                    │  │   MODE A: Inference      │   │
                    │  │   - vLLM engine          │   │
                    │  │   - PagedAttention        │   │
                    │  │   - MLA KV cache (~1/23)  │   │
                    │  │   - G=4 completions/prompt │   │
                    │  └──────────┬──────────────┘   │
                    │             │ swap (~10s-1min)  │
                    │  ┌──────────▼──────────────┐   │
                    │  │   MODE B: Training       │   │
                    │  │   - Megatron engine       │   │
                    │  │   - PP16 × EP16 × ZeRO-1 │   │
                    │  │   - Muon optimizer        │   │
                    │  │   - Gradient accumulation  │   │
                    │  └─────────────────────────┘   │
                    └────────────────────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
     ┌────────────┐   ┌──────────────┐   ┌──────────────┐
     │ K8s Sandbox │   │ Mooncake RDMA│   │  Prompt DB   │
     │ Pool (10K+) │   │ Checkpoint   │   │  (Curriculum) │
     │ Code exec   │   │ Engine <30s  │   │  Difficulty   │
     └────────────┘   └──────────────┘   │  Filtered     │
                                          └──────────────┘
```

**Data Flow**:
1. Controller selects prompts (curriculum sampling: proportional to `1 - success_rate`)
2. GPU pool enters inference mode → vLLM generates G=4 completions per prompt
3. Rewards computed (binary for math/code via K8s sandbox, self-critique for open-ended)
4. GPU pool transitions to training mode (~10s-60s dead time)
5. Megatron runs policy gradient update with Muon optimizer
6. Checkpoint broadcast via RDMA <30s
7. Repeat

**Synchronization model**: Fully synchronous. No train-infer overlap. One mode at a time.

## 3. BOTTLENECK ANALYSIS

| Phase | % Time | Bottleneck | Why |
|-------|--------|-----------|-----|
| Rollout generation | ~80-85% | Autoregressive decoding is sequential, memory-bound | [INFERRED-STRONG] Industry standard 85-95%, MLA helps reduce KV cache pressure |
| Mode transition | ~5-10% | GPU memory swap (release vLLM → init Megatron) | [FACT] <1 min per cycle |
| Training step | ~5-8% | Compute-bound (forward + backward + Muon NS orthogonalization) | [INFERRED-STRONG] Training is fast relative to generation |
| Reward computation | ~2-5% | K8s sandbox execution for code; instant for math | [INFERRED-STRONG] |
| Checkpoint sync | <1% | RDMA broadcast <30s is negligible per cycle | [FACT] |

**Primary bottleneck**: Rollout generation. MLA's 23× KV compression helps — enables larger batch sizes during inference — but autoregressive decoding is fundamentally sequential.

**Secondary bottleneck**: Mode transition dead time. 10-60 seconds per cycle where GPUs produce zero useful work. At 256 GPUs, even 30s dead time = 2.1 GPU-hours/cycle.

**Scaling breakpoint**: Adding more GPUs doesn't help rollout speed unless you can increase batch size (KV cache memory is the limit). EP16 parallelism helps distribute expert memory but adds communication.

## 4. MEMORY ANALYSIS

### GPU Memory Budget (per GPU, at frontier scale)

| Component | Size | Phase | Notes |
|-----------|------|-------|-------|
| Model weights (BF16) | ~30 GB (shared across PP16×EP16×ZeRO-1) | Both | [FACT] "~30GB per GPU" |
| Optimizer states (Muon) | ~4-6 GB/GPU (distributed) | Train | Muon has momentum + NS workspace |
| Gradients | ~2-4 GB/GPU (distributed) | Train | Distributed via ZeRO-1 |
| Activations | ~8-16 GB/GPU | Train | Selective recompute (LayerNorm, SwiGLU, MLA up-proj, MoE down-proj) |
| FP8 activation storage | ~4-8 GB | Train | [FACT] E4M3 1×128 tiles with FP32 scales |
| KV cache (inference) | ~2-6 GB/GPU | Inference | MLA compression: only 512-dim latent cached per layer |
| vLLM overhead | ~2-4 GB | Inference | PagedAttention metadata, scheduling |

**Peak memory (inference)**: ~36-40 GB/GPU → fits on H800 (80GB) with ~40GB headroom for batch scaling
**Peak memory (training)**: ~50-64 GB/GPU → fits on H800 with ~16-30GB headroom

**Key advantage**: MLA reduces inference memory by ~23× compared to standard MHA, enabling dramatically larger batch sizes during rollout. This directly accelerates the bottleneck phase.

**OOM risk**: LOW. Well within H800 capacity. Selective recomputation + FP8 storage keeps activations manageable.

## 5. COMMUNICATION ANALYSIS

| Communication Type | Pattern | Bandwidth | Frequency |
|-------------------|---------|-----------|-----------|
| EP all-to-all (expert dispatch) | Intra-node NVLink + inter-node RoCE | 8×400 Gbps = 3.2 Tbps per node | Every token (forward) |
| PP pipeline bubble | Interleaved 1F1B | Internal to PP group | Every microbatch |
| ZeRO-1 gradient AllReduce | Cross-node | ~400 Gbps per link | Every training step |
| Checkpoint broadcast | RDMA (Mooncake) | Pipelined, param-by-param | Every train→infer transition |
| K8s sandbox ↔ controller | HTTP/gRPC | Low bandwidth (results only) | Per code execution |

**EP communication is the critical path**. At 384 experts with EP16, each GPU hosts 24 experts. All-to-all dispatch requires sending tokens to remote GPUs for non-local experts. Overlapped with interleaved 1F1B scheduling during training. During inference, EP communication is the latency bottleneck.

**Network scaling ceiling**: At >256 GPUs, the all-to-all communication for EP starts saturating inter-node bandwidth. Going beyond requires more sophisticated EP sharding or reducing experts.

## 6. SCALING BEHAVIOR

| Scaling Axis | Behavior | Limit |
|-------------|----------|-------|
| More GPUs (horizontal) | Reduces per-GPU memory; enables larger batch | EP communication saturates at ~512+ GPUs |
| Bigger GPUs (vertical) | Larger batch during rollout; fewer GPUs needed | H800→H200 would reduce GPU count ~2× |
| More prompts per batch | Linear throughput increase until KV cache OOM | MLA helps (~8-16× more prompts than MHA) |
| Longer sequences | KV cache grows linearly (MLA: slowly) | 256K context is practical with MLA |
| More experts (EP scaling) | Each expert goes to fewer GPUs | All-to-all communication overhead grows |

**What breaks first**: EP all-to-all communication. With 384 experts and EP16, adding more GPUs requires wider EP groups, increasing cross-node all-to-all volume. The 400Gbps RoCE links become the ceiling.

**Diminishing returns**: Beyond ~512 GPUs, the mode transition dead time (10-60s) becomes a larger fraction of total time because training steps get faster with more GPUs but rollout speed is batch-limited.

## 7. ENGINEERING COMPLEXITY

| Aspect | Difficulty | Why |
|--------|-----------|-----|
| Initial setup | MEDIUM | vLLM + Megatron are mature; mode-switching is the novel part |
| Mode transition logic | HIGH | Safely releasing GPU memory from one engine and initializing another without leaks is tricky |
| Checkpoint pipelining | HIGH | RDMA broadcast of 1T params in <30s requires careful staging |
| Muon optimizer for RL | MEDIUM-HIGH | Muon is not standard; Newton-Schulz orthogonalization adds per-step overhead; interaction with RL gradients is poorly understood |
| Curriculum sampling | LOW | Standard implementation |
| K8s sandbox orchestration | HIGH | 10K concurrent containers with reliable reward signals; failure handling |
| Debugging rollout quality | MEDIUM | Colocated means you can inspect state locally; no async mismatch |

**Total estimated LoC for custom infra**: ~2,000-3,000 (mode transition + checkpoint engine + curriculum + K8s interface)

## 8. FAILURE MODES

1. **Mode transition memory leak** [INFERRED-STRONG]: If vLLM doesn't fully release GPU memory before Megatron starts, training OOMs. Need explicit CUDA memory cleanup between transitions.

2. **Muon numerical instability during RL** [INFERRED-WEAK]: Muon's Newton-Schulz iteration was tuned for pre-training gradients. RL gradients are sparser and noisier — may require re-tuning iteration count or conditioning.

3. **K8s sandbox timeout** [INFERRED-STRONG]: Code execution in K8s containers can hang. Need aggressive timeouts + reward = 0 for timeouts. At 10K concurrent containers, even 0.1% hang rate = 10 stuck containers per batch.

4. **Checkpoint broadcast failure** [INFERRED-WEAK]: RDMA failure during weight sync leaves some workers with stale weights. Need checksumming or retry logic.

5. **GPU utilization oscillation** [FACT-implied]: During transition windows, all GPUs are idle. At scale, this is significant wasted compute. No overlap between rollout and training.

## 9. UNCERTAINTY LABELS

- Mode transition time (<1 min, ~10s): [FACT]
- RDMA checkpoint <30s: [FACT]
- GPU count for RL: [INFERRED-STRONG] (~256+)
- Muon hyperparameters for RL: [UNKNOWN]
- EP communication overlap effectiveness: [INFERRED-STRONG] (described in paper)
- Total RL training cost: [UNKNOWN]
- Mode transition memory management details: [UNKNOWN]
- K8s failure handling specifics: [UNKNOWN]

---

# Pipeline 2: GLM-5 (Zhipu AI / Slime Framework)

## 1. FACTS

| Component | Value | Source |
|-----------|-------|--------|
| Model params | 744B total / 40B active | [FACT] arXiv:2602.15763 |
| MoE topology | 256 routed + 1 shared, top-8 | [FACT] HuggingFace config.json |
| Attention | MLA + DSA (Dynamic Sparse Attention) | [FACT] Paper |
| Layers | 78 | [FACT] HuggingFace config.json |
| Pre-train tokens | 28.5T | [FACT] Paper |
| Hardware | ~100,000 Huawei Ascend 910B | [FACT] Paper |
| Training engine | Megatron-based | [FACT] Paper |
| Inference engine | SGLang + custom Router | [FACT] Paper + Slime repo |
| RL framework | Slime (open-source) | [FACT] github.com/THUDM/slime |
| RL algorithm | GRPO + IcePop | [FACT] Paper Eq. 1 |
| IcePop β | 2 (suppress if ρ outside [0.5, 2.0]) | [FACT] Paper |
| Asymmetric clip | ε_low=0.2, ε_high=0.28 | [FACT] Paper |
| MTP | 1 prediction layer, accept length 2.76 | [FACT] Paper |
| DSA indexer | Frozen during RL | [FACT] Paper |
| TITO | Exact token preservation inference→training | [FACT] Paper |
| APRIL | 44% rollout throughput improvement | [FACT] arXiv:2509.18521 |
| RL stages | 5 (SFT → Reasoning RL → Agentic RL → General RL → Distillation) | [FACT] Paper |
| Operating modes | Sync (co-located) + Async (decoupled) | [FACT] Slime README |
| Distillation batch | 1024 | [FACT] Paper Eq. 2 |
| RL GPU count | [UNKNOWN] | Not disclosed |
| RL duration | [UNKNOWN] | Not disclosed |
| RL cost | [UNKNOWN] | Estimated >$100M based on 100K chips |

## 2. SYSTEM DESIGN

### Actor-Learner Topology: Fully Separated Async

```
         ┌───────────────────────────────────────────────────────┐
         │              SLIME ORCHESTRATOR (Central)              │
         └───────┬──────────┬──────────┬──────────┬─────────────┘
                 │          │          │          │
    ┌────────────▼──┐  ┌───▼──────┐  ┌▼────────┐  ┌▼──────────────┐
    │ INFERENCE      │  │ TRAINING │  │ DATA    │  │ ENVIRONMENT   │
    │ CLUSTER        │  │ CLUSTER  │  │ BUFFER  │  │ POOL          │
    │                │  │          │  │         │  │               │
    │ SGLang +Router │  │ Megatron │  │ Prompts │  │ K8s containers│
    │ FP8 inference  │  │ GRPO+    │  │ Custom  │  │ 10K+ envs     │
    │ EP64, DP64     │  │ IcePop   │  │ data    │  │ 9 languages   │
    │ MTP speculative│  │ Muon     │  │ Gen     │  │ Terminal      │
    │ Prefill-decode │  │ Split    │  │ methods │  │ Browser       │
    │ disaggregation │  │          │  │         │  │ Search        │
    └───────┬────────┘  └───┬──────┘  └────┬────┘  └───────────────┘
            │               │              │
            │  ┌────────────▼──────────┐   │
            │  │   WEIGHT SYNC (RDMA)   │   │
            │  │   Every K steps        │   │
            └──│   + optimizer reset    │───┘
               │   IcePop corrects      │
               │   staleness            │
               └────────────────────────┘
```

**Data Flow**:
1. Slime orchestrator dispatches prompts to Data Buffer (difficulty-filtered)
2. Inference cluster generates completions (G=32 for reasoning, G=1 for distillation)
3. Completions sent to Data Buffer with rollout policy log-probs
4. Environment Pool executes code, returns binary rewards
5. Data Buffer feeds (prompt, completion, reward, old_logps_infer) to Training cluster
6. Training cluster computes IcePop pop operator: ρ = π_train/π_infer
   - If ρ ∈ [0.5, 2.0]: token contributes normally
   - If ρ outside bounds: token gradient zeroed (suppressed)
7. Policy gradient update with GRPO + asymmetric clipping + IcePop
8. Every K steps: RDMA weight sync to inference cluster + optimizer reset
9. Inference cluster continues generating with slightly stale policy (corrected by IcePop)

**Synchronization model**: Asynchronous. Training and inference run concurrently on separate GPU pools. IcePop handles the resulting policy version mismatch.

## 3. BOTTLENECK ANALYSIS

| Phase | % Time | Bottleneck | Why |
|-------|--------|-----------|-----|
| Rollout generation | ~75-85% | Still autoregressive, but APRIL + MTP reduce to lower end | [FACT] APRIL: 44% throughput improvement |
| Weight sync (RDMA) | ~3-8% | Full model sync every K steps | [INFERRED-STRONG] 744B params, even with RDMA |
| Training step | ~5-10% | Overlapped with inference (async benefit) | [INFERRED-STRONG] |
| Reward computation | ~3-5% | >10K environment executions | [INFERRED-STRONG] |
| IcePop computation | <1% | Simple ratio check + masking per token | [INFERRED-STRONG] Lightweight |
| Distillation (Stage 5) | N/A | Separate stage, G=1 (no multi-sample overhead) | [FACT] |

**Primary bottleneck**: Rollout generation (reduced by APRIL + MTP).
- APRIL: Over-provisions rollout requests, terminates early → 44% throughput boost
- MTP: Speculative decoding with accept length 2.76 → ~1.5-2× inference speedup
- Combined: Rollout share drops from ~90% to ~75-80%

**Secondary bottleneck**: Weight synchronization. Every K training steps, full model weights must be synced from training → inference cluster via RDMA. For 744B params in BF16 = ~1.5 TB. Even at 400 Gbps, theoretical transfer = ~30 seconds. With pipelining and partial updates, practical = 15-45 seconds. During sync, inference may stall or use stale weights.

**Critical insight**: IcePop's token-level suppression handles staleness, but the suppression rate is unknown — if too many tokens are suppressed, effective gradient signal degrades.

## 4. MEMORY ANALYSIS

### Inference Cluster (per GPU, estimated)

| Component | Size | Notes |
|-----------|------|-------|
| Model weights (FP8) | ~0.7 TB total / ~0.04 TB active per token | FP8 storage: 744B × 1 byte ≈ 744 GB distributed across EP64 |
| Per-GPU weights | ~11-12 GB | 744 GB / 64 EP GPUs |
| KV cache (MLA compressed) | ~2-4 GB/GPU | MLA compresses, DSA further reduces |
| SGLang overhead | ~2-3 GB | RadixAttention, scheduling |
| MTP auxiliary | ~1-2 GB | Extra prediction head |
| **Total per GPU (infer)** | ~18-22 GB | Fits Ascend 910B (64 GB HBM) |

### Training Cluster (per GPU, estimated)

| Component | Size | Notes |
|-----------|------|-------|
| Model weights (BF16) | ~1.5 TB distributed | 744B × 2 bytes |
| Per-GPU weights | ~12-24 GB | Depends on PP + EP + ZeRO partitioning |
| Optimizer states (Muon) | ~20-40 GB/GPU | Muon momentum + NS workspace, distributed |
| Gradients | ~12-24 GB/GPU | Distributed |
| Activations | ~8-16 GB/GPU | Selective recompute |
| **Total per GPU (train)** | ~52-100+ GB | Tight on Ascend 910B (64 GB HBM) |

**OOM risk**: MEDIUM-HIGH for training. Ascend 910B has only 64 GB HBM (vs H100's 80 GB). With 744B params, memory is tight. Requires aggressive ZeRO + activation checkpointing + CPU offload.

**Key concern**: Ascend 910B has ~320 TFLOPS FP16, roughly 1/3 of H100's ~989 TFLOPS. This means ~3× more chip-hours for equivalent compute. The 100K chip count compensates but leaves less headroom for memory-intensive operations.

## 5. COMMUNICATION ANALYSIS

| Communication Type | Pattern | Bandwidth | Frequency |
|-------------------|---------|-----------|-----------|
| EP all-to-all (inference) | Across EP64 group | Ascend interconnect | Every token |
| EP all-to-all (training) | Across training EP group | Ascend interconnect | Every forward pass |
| Weight sync (RDMA) | Inference ← Training | Full model, every K steps | Every K gradient updates |
| Data Buffer ↔ both clusters | Prompts + completions + rewards | Network | Continuous |
| K8s environment ↔ Buffer | Reward signals | Network | Per trajectory |
| Heartbeat monitoring | Slime fault detection | Low bandwidth | Continuous |

**Weight sync is the critical communication path**. Unlike Kimi's colocated design (where weights are already in GPU memory), GLM-5 must physically transfer ~1.5 TB every K steps across network.

**Network topology matters**:
- If inference and training clusters are in the same rack: ~100-400 Gbps, sync = 30-120s
- If cross-rack: potentially minutes, IcePop must compensate for longer staleness

**Ascend interconnect**: Huawei's proprietary HCCS (Huawei Cache Coherent System). Specifications:
- Intra-node: 8 chips connected, ~400 GB/s total bandwidth [INFERRED-STRONG]
- Inter-node: Less documented than NVIDIA's NVSwitch + RoCE
- [INFERRED-WEAK] Likely lower inter-node bandwidth than NVIDIA fabric

## 6. SCALING BEHAVIOR

| Scaling Axis | Behavior | Limit |
|-------------|----------|-------|
| More inference GPUs | More throughput (EP wider, more DP groups) | EP communication; Router scheduling |
| More training GPUs | Faster gradient steps | Diminishing returns on short gradient steps |
| More environments | More concurrent reward evaluation | K8s scheduling overhead |
| APRIL aggressiveness | More over-provisioning → higher throughput | Wasted compute on discarded partial rollouts |
| Longer context | MLA + DSA help, but still linear growth | DSA indexer frozen during RL: no adaptation to RL-induced attention shifts |

**What breaks first**: Training cluster memory on Ascend 910B. 64 GB HBM is tight for 744B params. Adding more chips helps distribute memory but adds communication overhead. The Ascend interconnect (less documented than NVIDIA) is a potential ceiling for EP scaling.

**GLM-5 specific scaling advantage**: APRIL recycling. Incomplete rollouts from over-provisioning are recycled for future steps, improving sample efficiency. This means scaling more inference GPUs has better-than-linear returns.

## 7. ENGINEERING COMPLEXITY

| Aspect | Difficulty | Why |
|--------|-----------|-----|
| Slime setup | HIGH | Megatron-based, SGLang integration, Ascend-first (rough edges on NVIDIA) |
| Async orchestration | VERY HIGH | Two separate GPU clusters, Data Buffer management, staleness tracking |
| IcePop implementation | MEDIUM | Simple ratio check, but tuning β and understanding suppression rates is hard |
| TITO tokenization | MEDIUM | Exact token preservation is conceptually simple but requires careful plumbing |
| 5-stage pipeline | VERY HIGH | Each stage has different configs, reward functions, group sizes; cross-stage distillation adds a 6th moving part (teacher model management) |
| APRIL integration | HIGH | Over-provisioning logic, partial rollout recycling, scheduler modification |
| Weight sync management | HIGH | RDMA reliability, optimizer state reset, version tracking |
| Debugging async mismatch | VERY HIGH | When training diverges, is it IcePop suppressing too much? Stale weights? Router drift? Very hard to diagnose |
| Cross-stage distillation | HIGH | Managing teacher checkpoints, stop-gradient mechanics, conflicting teacher signals |
| DSA indexer management | MEDIUM | Must ensure indexer is frozen during RL; verify it doesn't degrade with RL-shifted attention patterns |

**Total estimated LoC for custom infra**: ~5,000-8,000 (Slime modifications + 5-stage orchestration + APRIL + cross-stage distillation)

## 8. FAILURE MODES

1. **IcePop over-suppression** [INFERRED-STRONG]: If β=2 is too aggressive, >50% of tokens could be suppressed in late RL stages (where train-infer divergence grows). This silently reduces effective batch size without warning. **No ablation published** on suppression rates.

2. **Weight sync corruption** [INFERRED-WEAK]: RDMA failure during 1.5 TB transfer on Ascend hardware (less battle-tested than NVIDIA InfiniBand) could leave inference cluster with partially updated weights. Slime has heartbeat monitoring but recovery specifics are [UNKNOWN].

3. **Cross-stage distillation conflict** [INFERRED-STRONG]: Stage 2 (reasoning) teacher and Stage 4 (general) teacher may give conflicting signals for ambiguous prompts. The stop-gradient advantage `sg[log(π_teacher/π_student)]` doesn't resolve this — it just averages conflicting teachers.

4. **LiveCodeBench regression** [FACT]: GLM-5 dropped from 84.9% (GLM-4.7) to 52.0% on LiveCodeBench — a 63% drop. No explanation provided. Possible cause: agentic RL (Stage 3) over-specialized for repo-level coding at expense of algorithmic coding. This is a 5-stage pipeline hazard: later stages can silently degrade earlier capabilities.

5. **Ascend hardware failures** [INFERRED-STRONG]: At 100K chips, hardware failure rate is non-trivial. Slime has heartbeat-driven deregistration, but recovery from mid-training failures at this scale is hard. Checkpointing frequency and recovery time are [UNKNOWN].

6. **DSA indexer staleness** [INFERRED-STRONG]: DSA indexer is frozen during RL. But RL changes attention patterns. If the frozen indexer selects wrong tokens for sparse attention, reasoning quality could degrade without visible loss signal.

7. **Optimizer state reset timing** [UNKNOWN]: When training cluster syncs weights to inference, optimizer states are reset. If this happens too frequently, Muon's momentum buffer never builds up. Too infrequently, IcePop must handle larger staleness gaps.

## 9. UNCERTAINTY LABELS

- Slime architecture: [FACT] (open-source)
- GRPO + IcePop: [FACT] (full equations)
- Async topology: [FACT] (described in paper)
- APRIL 44% improvement: [FACT] (separate paper)
- IcePop suppression rate: [UNKNOWN] (no ablation)
- Weight sync frequency (K): [UNKNOWN]
- Ascend interconnect bandwidth: [INFERRED-WEAK]
- Training cluster memory layout: [INFERRED-STRONG]
- Recovery from hardware failures: [UNKNOWN]
- Cross-stage distillation effectiveness: [FACT] (mentioned, but no ablation vs not doing it)

---

# Pipeline 3: MiniMax M2.7 (Forge Framework)

## 1. FACTS

| Component | Value | Source |
|-----------|-------|--------|
| M1 model params | 456B total / 45.9B active | [FACT] arXiv:2501.08313 |
| M2.x model params | ~230B total / ~10B active | [FACT] HuggingFace, API docs |
| MoE topology | 32 routed, top-2, no shared expert | [FACT] arXiv:2501.08313 |
| Attention | Hybrid: 7 lightning + 1 softmax per 8 layers | [FACT] arXiv:2501.08313 |
| Layers (Text-01) | 80 | [FACT] arXiv:2501.08313 |
| Expert hidden dim | 9,216 | [FACT] arXiv:2501.08313 |
| Context (train) | 1M tokens | [FACT] arXiv:2501.08313 |
| Context (infer) | 4M tokens | [FACT] arXiv:2501.08313 |
| RL algorithm | CISPO | [FACT] arXiv:2506.13585 |
| IS weight clip | max = 1 + ε_high, ε_high=5.0 | [FACT] CISPO docs + paper |
| .detach() on IS weights | Yes (core innovation) | [FACT] arXiv:2506.13585 |
| M1 RL GPUs | 512 H800 | [FACT] arXiv:2506.13585 |
| M1 RL duration | 3 weeks | [FACT] arXiv:2506.13585 |
| M1 RL cost | ~$534,700 | [FACT] arXiv:2506.13585 |
| Adam ε | 1e-15 | [FACT] arXiv:2506.13585 |
| Adam β1, β2 | 0.9, 0.95 | [FACT] arXiv:2506.13585 |
| FP32 LM head | Required for RL stability | [FACT] arXiv:2506.13585 |
| Repetition detection | 3000 tokens > 0.99 prob → truncate | [FACT] arXiv:2506.13585 |
| CISPO vs DAPO | 2× faster (50% steps) | [FACT] arXiv:2506.13585 |
| Thinking budgets | 40K and 80K tokens | [FACT] arXiv:2506.13585 |
| Gradient range | 1e-18 to 1e-5 | [FACT] arXiv:2506.13585 |
| Prefix tree merging | 40× speedup | [FACT] Forge blog |
| Windowed FIFO | 30% visibility window | [FACT] Forge blog |
| Scaffold types | 100K+ real-world | [FACT] Forge blog |
| Self-evolving rounds | 100+ | [FACT] M2.7 blog |
| Self-evolving improvement | 30% | [FACT] M2.7 blog |
| Inference speed | ~100 TPS | [FACT] arXiv:2501.08313 |

## 2. SYSTEM DESIGN

### Actor-Learner Topology: Separated with Middleware

```
    ┌──────────────────────────────────────────────────┐
    │           AGENT SIDE (100K+ scaffolds)            │
    │                                                    │
    │  ┌──────────┐ ┌──────────┐ ┌──────────┐          │
    │  │ Scaffold  │ │ Scaffold  │ │ Scaffold  │  ...    │
    │  │ Type A    │ │ Type B    │ │ Type C    │          │
    │  │ (ReAct)   │ │ (CodeAct) │ │ (Custom)  │          │
    │  └────┬──────┘ └────┬──────┘ └────┬──────┘          │
    │       └─────────────┼─────────────┘                  │
    │                     │ 4 interfaces per scaffold       │
    │               reprocess / run / postprocess /         │
    │               calculate_reward                        │
    └─────────────────────┼────────────────────────────────┘
                          │
              ┌───────────▼───────────────┐
              │     MIDDLEWARE LAYER        │
              │                            │
              │  ┌────────────────────┐    │
              │  │ Gateway + Router    │    │
              │  │ (API abstraction)   │    │
              │  └────────┬───────────┘    │
              │           │                │
              │  ┌────────▼───────────┐    │
              │  │ Windowed FIFO      │    │
              │  │ Scheduler          │    │
              │  │ (30% visibility)   │    │
              │  └────────┬───────────┘    │
              └───────────┼────────────────┘
                          │
          ┌───────────────┼───────────────────┐
          │               │                   │
    ┌─────▼──────┐  ┌────▼──────────┐  ┌────▼──────────┐
    │ INFERENCE   │  │ TRAINING      │  │ KV CACHE      │
    │ ENGINE      │  │ ENGINE        │  │ (Global L3)   │
    │             │  │               │  │               │
    │ Lightning   │  │ CISPO loss    │  │ DFS-structured│
    │ attention   │  │ AdamW ε=1e-15 │  │ Prefix tree   │
    │ MTP decode  │  │ FP32 LM head  │  │ merging (40×) │
    │ 100 TPS     │  │ β2=0.95       │  │               │
    │ FP8         │  │ G=16, K=16    │  │ Magi Attention│
    └─────────────┘  └───────────────┘  └───────────────┘
```

**Data Flow**:
1. Scaffolds generate agent interaction trajectories via 4-interface API
2. Middleware Gateway routes requests; Windowed FIFO scheduler prevents easy-sample dominance
3. Inference engine generates completions using lightning attention (100 TPS) + MTP
4. Prefix tree merging: shared prefixes across multi-turn trajectories → single forward pass (40× speedup)
5. Rewards computed via scaffold's `calculate_reward` (F2P/P2P for code, exact match for math)
6. Training engine runs CISPO: `detach(clamp(r, max=6.0)) × Â × log π_θ`
7. K=16 gradient steps per generation batch
8. Weight sync to inference (mechanism: [UNKNOWN])
9. M2.7: Self-evolving loop modifies scaffolds autonomously every N rounds

**Synchronization model**: Separated pools, synchronization details [UNKNOWN] but likely similar to GLM-5 (RDMA sync). The middleware adds a layer of indirection.

## 3. BOTTLENECK ANALYSIS

| Phase | % Time | Bottleneck | Why |
|-------|--------|-----------|-----|
| Rollout generation | ~65-75% | Lightning attention (87.5% O(n)) makes this faster than competitors | [INFERRED-STRONG] 100 TPS is ~2-3× faster than standard softmax |
| Middleware scheduling | ~2-5% | Windowed FIFO + gateway routing | [INFERRED-STRONG] HTTP/gRPC overhead per request |
| Prefix tree construction | ~3-5% | Building tree from multi-turn trajectories | [INFERRED-STRONG] One-time per batch, amortized |
| Training step | ~10-15% | CISPO loss + K=16 gradient steps per generation | [INFERRED-STRONG] K=16 means 16 train steps per rollout |
| Reward computation | ~5-8% | F2P/P2P test execution in sandbox | [INFERRED-STRONG] SWE-heavy |
| Weight sync | ~2-5% | [UNKNOWN mechanism] | [UNKNOWN] |

**Primary bottleneck**: Still rollout, but lightning attention reduces it more than any competitor.

**Key advantage**: Lightning attention gives O(n) complexity for 87.5% of layers → at 100K+ token trajectories (agentic RL), this is dramatically cheaper than O(n²) softmax:
- At 100K tokens: ~25% FLOPs of DeepSeek-R1 [FACT]
- At 200K tokens: ~12.5% FLOPs [INFERRED-STRONG] (O(n) vs O(n²) scaling)

**Prefix tree merging is a game-changer for multi-turn**: Multiple completions sharing the same system prompt + conversation history → single prefix in tree. 40× speedup on shared prefix portions. This is unique to MiniMax and directly addresses the multi-turn RL bottleneck.

**K=16 gradient steps per generation** [FACT]: MiniMax takes 16 gradient steps per batch of generated data, vs typical 1-4. This increases the training phase share but extracts more learning per rollout. Trade-off: higher risk of IS weight drift after 16 steps.

## 4. MEMORY ANALYSIS

### M1 (456B total, 45.9B active) — During RL Training

| Component | Size | Notes |
|-----------|------|-------|
| Model weights (BF16) | ~912 GB total | 456B × 2 bytes |
| Per-GPU weights (512 H800) | ~1.8 GB/GPU (with EP) | Highly distributed |
| Optimizer (AdamW) | ~3.6 TB total | 456B × 8 bytes (fp32 moments) |
| Per-GPU optimizer | ~7 GB/GPU | Distributed across 512 GPUs |
| Gradients | ~912 GB total / ~1.8 GB/GPU | Distributed |
| FP32 LM head | ~0.8 GB extra | Vocab=200K × hidden=6144 × 4 bytes |
| Activations | ~4-8 GB/GPU | With recompute |
| **Total per GPU** | ~16-20 GB | Comfortable on H800 80GB |

### M1 — During RL Inference

| Component | Size | Notes |
|-----------|------|-------|
| Model weights (FP8) | ~456 GB total | FP8 inference |
| Per-GPU weights | ~0.9 GB/GPU | Distributed across 512 |
| KV cache (lightning attention) | Minimal | O(n) memory for 87.5% layers |
| Softmax KV cache (12.5% layers) | ~0.5-2 GB/GPU | Only 10/80 layers need standard KV cache |
| **Total per GPU** | ~4-8 GB | Very light — room for massive batches |

### M2.x (230B total, ~10B active) — Inference Production

| Component | Size | Notes |
|-----------|------|-------|
| Model weights | ~460 GB (BF16) or ~230 GB (FP8) | Smaller model |
| KV cache at 205K context | ~4-8 GB per sequence | Lightning attention: only softmax layers accumulate |
| **Inference memory** | Very efficient at 10B active | Production-optimized for low-cost serving |

**Key memory advantage**: Lightning attention's O(n) memory scaling for 87.5% of layers means KV cache pressure is dramatically lower than competitors. At 200K tokens, MiniMax needs ~6× less KV cache memory than a standard MLA model. This enables massive batch sizes during rollout.

**OOM risk**: LOW for M1 (512 H800 for 456B is comfortable). LOW for M2.x (small model on production hardware).

## 5. COMMUNICATION ANALYSIS

| Communication Type | Pattern | Bandwidth | Frequency |
|-------------------|---------|-----------|-----------|
| Middleware ↔ Engines | HTTP/gRPC | Moderate | Per-request (scaffold interaction) |
| EP all-to-all | Standard MoE dispatch | NVLink/NVSwitch | Per token |
| Weight sync (infer←train) | [UNKNOWN mechanism] | [UNKNOWN] | [UNKNOWN frequency] |
| Scaffold ↔ Environment | HTTP to sandbox containers | Low | Per tool invocation |
| Global L3 KV cache sync | DFS-structured, local | Internal to inference cluster | Per prefill |

**Middleware is the novel communication path**. Unlike Kimi (direct controller → GPU) and GLM-5 (Slime orchestrator → GPU clusters), MiniMax adds a gateway + scheduler between scaffolds and engines. This adds latency per request but enables:
- Scaffold-agnostic routing (any scaffold type works)
- Fair scheduling (Windowed FIFO prevents easy-sample bias)
- Decoupled scaling (add scaffolds without changing engine)

**Communication overhead**: [INFERRED-STRONG] ~5-10% total from middleware. Per-request HTTP/gRPC adds ~1-5ms latency, which is negligible for 200K-token trajectories (seconds to generate) but non-trivial for short interactions.

## 6. SCALING BEHAVIOR

| Scaling Axis | Behavior | Limit |
|-------------|----------|-------|
| More inference GPUs | Linear throughput increase | Middleware scheduling becomes bottleneck |
| More scaffolds | Logarithmic benefit (diversity) | Middleware gateway load; KV cache memory |
| Longer context | Lightning attention: near-linear scaling | Softmax layers (12.5%) become bottleneck at >500K |
| More training steps (K) | More learning per rollout | IS weight drift after K>16 makes correction unreliable |
| More environments | More concurrent reward evaluation | Sandbox orchestration overhead |

**What breaks first**: Middleware scheduler. At 100K+ scaffolds with millions of daily samples, the gateway + FIFO scheduler becomes a single point of failure and throughput bottleneck. Need to shard the middleware horizontally.

**Lightning attention scaling advantage**: At 1M tokens, where other models need O(n²) compute for attention, MiniMax needs O(n) for 87.5% of layers. This is an asymptotic advantage that grows with context length. For agentic RL with 200K+ token trajectories, this is transformative.

**K=16 scaling concern**: After 16 gradient steps, the policy has changed significantly from the rollout policy. The IS ratio `r = π_θ/π_old` becomes unreliable. CISPO's `.detach()` helps (gradient isn't sensitive to the exact IS weight) but the advantage estimate Â_i is still from the original rollout.

## 7. ENGINEERING COMPLEXITY

| Aspect | Difficulty | Why |
|--------|-----------|-----|
| CISPO implementation | LOW | ~20 lines of core code |
| Adam ε + FP32 LM head fixes | TRIVIAL | One line each, but CRITICAL to know about |
| Repetition detection | LOW | Simple probability threshold check |
| Forge middleware | HIGH | Gateway + scheduler + 4-interface contract; HTTP/gRPC plumbing |
| Prefix tree merging | VERY HIGH | Custom attention kernel (Magi Attention); tree construction; post-forward deconstruction |
| Multi-scaffold integration | HIGH | Each scaffold type needs 4 interface implementations; testing across hundreds of types |
| Windowed FIFO scheduler | MEDIUM | Algorithmic complexity is moderate; tuning the 30% window is the hard part |
| Self-evolving loop (M2.7) | EXTREME | Model modifying its own training scaffolds; guardrails; evaluation; rollback; 100+ autonomous rounds |
| Global L3 KV cache | HIGH | DFS-structured cache management; prefix cache hit optimization |
| MTP speculative decoding during RL | MEDIUM-HIGH | Must maintain acceptance rate as policy evolves; Top-K KL loss for MTP head stability |

**Total estimated LoC for custom infra**: ~3,000-5,000 (Forge middleware + prefix tree + multi-scaffold + KV cache management). Self-evolving adds unbounded complexity.

## 8. FAILURE MODES

1. **CISPO entropy collapse** [FACT — STAPO arXiv:2602.15620]: CISPO's "all tokens get gradients" causes entropy collapse in late training. Performance declines as entropy drops. Root cause: spurious tokens (~0.01%) with high IS weights amplify noise. **Mitigation**: DISPO/STAPO, but [UNKNOWN] if M2.7 uses them.

2. **Middleware single point of failure** [INFERRED-STRONG]: Gateway + scheduler must handle millions of daily requests. Failure → all scaffold interactions stop. Need redundancy, but adds complexity.

3. **Scaffold diversity explosion** [INFERRED-STRONG]: 100K+ scaffolds × varying tool formats × different context management strategies = massive combinatorial testing surface. A bug in one scaffold type can produce reward signals that poison the entire training batch.

4. **K=16 IS weight drift** [INFERRED-STRONG]: After 16 gradient steps, π_θ has shifted substantially from π_old. Even with `.detach()`, the advantage estimate Â_i (computed from old policy rollouts) becomes stale. At K=16 vs typical K=1-4, this is 4-16× more staleness.

5. **FP32 LM head memory** [FACT-implied]: Keeping the LM head in FP32 (vocab=200K × hidden=6144 × 4 bytes = ~5 GB) while rest is BF16/FP8. Not a huge amount, but must be handled correctly in mixed-precision training.

6. **Self-evolving loop divergence** [INFERRED-STRONG]: M2.7's model modifies its own scaffolds. Without strong guardrails:
   - Compounding errors over 100+ rounds
   - Scaffold modifications that game evaluation but degrade real performance
   - No external grounding (unlike Kimi's verifiable-reward-calibrated self-critique)

7. **Lightning attention RL gradient dynamics** [INFERRED-WEAK]: Lightning attention's O(n) computation uses different attention mechanisms than softmax. RL gradients through lightning attention layers may behave differently than through softmax layers. This is unstudied in literature.

## 9. UNCERTAINTY LABELS

- CISPO algorithm: [FACT] (full equation + code)
- Adam ε=1e-15: [FACT] (paper + seminar)
- FP32 LM head: [FACT] (paper)
- M1 RL cost ($534K): [FACT]
- Forge architecture: [FACT] (blog with diagrams)
- Prefix tree 40×: [FACT] (blog claim, no independent verification)
- M2.7 architecture: [UNKNOWN] (proprietary, ~230B inferred)
- M2.7 RL training details: [UNKNOWN]
- Weight sync mechanism/frequency: [UNKNOWN]
- Self-evolving loop implementation: [UNKNOWN] (concept verified, code not released)
- Whether M2.7 uses DISPO/STAPO: [UNKNOWN]
- Middleware failure handling: [UNKNOWN]

---

# COMPARATIVE ANALYSIS

## Head-to-Head Scoring

| Dimension | Kimi K2.5 | GLM-5 | MiniMax M2.7 |
|-----------|----------|-------|-------------|
| **Actor-Learner Complexity** | LOW (colocated, no async) | HIGH (fully separated, async, IcePop) | MEDIUM-HIGH (separated, middleware layer) |
| **Rollout Throughput** | MEDIUM (MLA helps KV cache, but standard autoregressive) | MEDIUM-HIGH (APRIL +44%, MTP +1.5×) | HIGH (lightning attention O(n) + prefix tree 40× + MTP) |
| **Memory Pressure** | LOW (well within H800 capacity) | MEDIUM-HIGH (tight on Ascend 910B 64GB) | LOW (512 H800 for 456B; lightning attention reduces KV) |
| **Communication Cost** | LOW (colocated = no weight transfer; EP all-to-all is main cost) | HIGH (RDMA weight sync every K steps + EP on Ascend) | MEDIUM (middleware HTTP overhead + weight sync [UNKNOWN]) |
| **Scaling Difficulty** | MEDIUM (EP ceiling at ~512 GPUs; mode transition overhead) | HIGH (Ascend interconnect limits; 5-stage coordination) | MEDIUM (middleware scaling; lightning attention helps) |
| **Debuggability** | HIGH (colocated = single system, no async mismatch) | LOW (async mismatch, 5 stages, IcePop suppression rate unknown) | MEDIUM (CISPO is simple, but middleware + scaffold diversity add opacity) |
| **Cost Efficiency** | MEDIUM (GPU idle during transitions; Muon overhead) | LOW (Ascend 910B ~1/3 FLOPS of H100; more chip-hours needed) | HIGH (lightning attention = fewer FLOPs; M1 cost published: $535K for frontier model) |
| **Time to First Result** | 3-5 days (need custom mode-switching + Muon setup) | 1-2 weeks (Slime setup + 5-stage orchestration) | 1-2 days (CISPO is ~20 lines; Forge middleware optional for first experiment) |
| **MoE Stability During RL** | MEDIUM (384 experts, no published MoE-RL correction) | MEDIUM-HIGH (256 experts, IcePop handles some routing drift) | HIGH (32 experts only, fewer routing failure modes; but top-2 means more concentrated) |
| **Reproducibility** | MEDIUM (Muon hyperparams [UNKNOWN]; τ [UNKNOWN]) | LOW-MEDIUM (many [UNKNOWN] hyperparams; Ascend-specific) | HIGH (full CISPO equation + stability fixes published; ScaleRL validated) |

## Why Each Differs — Root Cause Analysis

### Actor-Learner Complexity

**Kimi is simple because**: Colocated design eliminates network weight transfer entirely. Same GPUs, same memory — just switch execution mode. This works because at RL scale (not pre-training scale), a single cluster is sufficient.

**GLM-5 is complex because**: Fully async design was driven by hardware constraints — Ascend 910B has lower per-chip FLOPS, so maximizing utilization (never idle) is critical. The async design keeps both clusters busy but introduces the train-infer mismatch problem requiring IcePop.

**MiniMax is moderately complex because**: The middleware layer exists to support 100K+ scaffold types — a diversity goal that neither Kimi nor GLM-5 prioritized. The middleware is not needed for RL per se, but for multi-scaffold agentic training.

### Rollout Throughput

**MiniMax leads because**: Lightning attention gives O(n) complexity for 87.5% of layers. At 100K+ token agentic trajectories, this is 4-25× cheaper than O(n²) softmax. Plus prefix tree merging eliminates redundant prefilling in multi-turn (40×). These are architectural advantages, not just engineering.

**GLM-5 is second because**: APRIL (44% throughput boost) and MTP speculative decoding (accept length 2.76) are engineering optimizations that don't change the fundamental O(n²) attention cost but reduce practical overhead significantly.

**Kimi is third because**: MLA reduces KV cache memory (enabling larger batches) but doesn't change attention compute complexity. Still O(n²) per attention layer.

### Memory Pressure

**GLM-5 is worst because**: Ascend 910B has 64 GB HBM vs H800's 80 GB. For a 744B model, this is tight. Training requires aggressive memory optimization (ZeRO, recompute, CPU offload) that adds engineering complexity and potential performance overhead.

**Kimi and MiniMax are comfortable because**: Both run on H800 (80 GB) with well-distributed model sharding. MiniMax's lightning attention further reduces KV cache pressure during inference.

### Cost Efficiency

**MiniMax leads because**: Lightning attention fundamentally reduces FLOPs for long sequences. At 100K tokens, ~25% FLOPs of DeepSeek-R1. Combined with M1's published cost ($534K for 3 weeks on 512 H800), this is the most cost-transparent pipeline.

**GLM-5 is worst because**: ~100K Ascend 910B chips at ~1/3 FLOPS each = ~3× more chip-hours for equivalent compute. The hardware constraint (China export controls) forces a less efficient solution.

### Debuggability

**Kimi leads because**: Colocated design means no async mismatch to debug. The model that generated rollouts is exactly the same as the model being trained (modulo the gradient update). One system, one state.

**GLM-5 is worst because**: Async design means rollout policy ≠ training policy. When training diverges, the diagnostic question is: "Is it the algorithm? IcePop parameters? Weight sync timing? Router drift? DSA indexer staleness? Cross-stage teacher conflict?" Five stages multiply the search space.

---

# NanoSeek Implications

## Scale-Down Translation (1B MoE, 4.75B total, single GPU)

At NanoSeek's scale:
- **No async needed**: Single GPU handles both inference and training
- **No EP needed**: 64 experts fit in one GPU's memory
- **No RDMA needed**: Everything is local
- **No middleware needed**: Direct training loop is sufficient
- **Colocated is the only sensible topology**

This eliminates GLM-5's async advantages and MiniMax's middleware. What remains is:
1. **The loss function** (algorithm quality)
2. **Stability fixes** (Adam ε, FP32 LM head, repetition detection)
3. **Sample efficiency** (rollout group size, gradient steps per generation)
4. **Anti-forgetting** (PTX vs curriculum vs distillation)

At this scale, MiniMax's CISPO algorithm + stability fixes offer the most directly transferable value, while GLM-5's IcePop and Kimi's mode-switching infrastructure are irrelevant.
