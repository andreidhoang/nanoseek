# Architecture-to-RL Fit Analysis & Engineering Feasibility Assessment
## Agent 5+6: Architecture-RL Compatibility + Systems/Infrastructure Engineering
### Compiled: 2026-03-24 | For NanoSeek Project

---

# PART 1: Architecture-to-RL Fit Analysis

## 1A. Dense Transformers (Llama, Gemma, Phi, Qwen-dense)

### Policy Gradient Stability: EXCELLENT
- Every parameter participates in every forward pass -- no routing stochasticity
- IS (importance sampling) ratios reflect genuine policy changes, not routing artifacts
- Standard GRPO/PPO/REINFORCE++ work out of the box without modifications
- No "routing replay" or "router shift" corrections needed
- Gradient flow is clean: all parameters receive consistent, unambiguous signal

### Memory Efficiency During Rollout Generation: MODERATE
- Full model must be in memory (no sparsity savings)
- KV cache scales linearly with sequence length and batch size
- For 7B dense: ~14GB model weights (BF16) + KV cache
- Rollout batch size limited by available GPU memory
- GQA (Llama 3, Gemma 2) reduces KV cache by 4-8x vs MHA, helping rollout throughput

### KV Cache Behavior During RL: STRAIGHTFORWARD
- Standard KV cache mechanics, well-supported by vLLM/SGLang
- No attention pattern changes between training and inference
- GQA models (Llama 3.x, Gemma 2/3) offer good memory-throughput tradeoff
- MHA models (older Llama, GPT-2 style) have larger KV cache but no compatibility issues

### Sample Efficiency: MODERATE
- All parameters are active, so each gradient step updates the full model
- No "dead parameter" waste, but also no specialization benefit
- At 1-7B scale, dense models learn reasoning patterns efficiently
- JustRL (arXiv:2512.16649) shows 1.5B dense model achieves 54.87% on 9 math benchmarks with vanilla GRPO

### Deployment Simplicity: EXCELLENT
- Mature tooling: vLLM, SGLang, TensorRT-LLM all optimized for dense models
- Every RL framework (veRL, OpenRLHF, TRL, torchforge) supports dense models natively
- No expert parallelism needed -- standard TP/PP/DP suffices
- Quantization (GPTQ, AWQ, FP8) is straightforward

### Overall RL Fit Score: 9/10
**Best for**: Teams that want maximum simplicity, fastest iteration, lowest debugging overhead. Ideal for 1-7B scale RL research.

---

## 1B. MoE Transformers (DeepSeek, Qwen-MoE, OLMoE, GLM-5)

### Routing Stability Under Policy Updates: PROBLEMATIC
- **Core issue**: After each gradient update, the router may assign tokens to different experts (~10% routing change per step observed in practice [RSPO paper, arXiv:2510.23027])
- This means the IS ratio `r_t = pi_theta / pi_old` conflates TWO sources of change:
  1. Genuine policy improvement (what we want to optimize)
  2. Routing drift (noise from expert reassignment)
- Result: inflated gradient variance, unstable training, potential divergence
- **Mitigations**: GSPO (sequence-level IS), RSPO (router-shift correction), IcePop (mismatch suppression), routing replay

### Expert Collapse / Dead Expert Risk During RL: HIGH
- RL gradients are much noisier than pre-training gradients (sparse binary rewards vs dense next-token prediction)
- Noisy gradients can push the router toward a few "safe" experts, starving others
- Dead experts during pre-training are already a known problem; RL amplifies this
- Load balancing auxiliary losses (used during pre-training) may conflict with RL objective
- Need to monitor I_spec (expert specialization MI) and Gini coefficient throughout RL
- **Mitigation**: Freeze router during early RL stages (but reduces adaptability)

### Token-Level vs Sequence-Level IS Ratio Implications: CRITICAL
- **Token-level IS (GRPO, CISPO)**: Each token's ratio independently computed. For MoE, individual token ratios are corrupted by routing changes. A token routed to Expert 3 under pi_old but Expert 7 under pi_theta has a structurally different probability, unrelated to policy improvement
- **Sequence-level IS (GSPO)**: Geometric mean across all tokens. Routing noise at individual tokens averages out across the sequence. This is why Qwen3 developed GSPO specifically for their MoE models
- **Recommendation**: For MoE models, sequence-level IS (GSPO) is strongly preferred over token-level IS

### Load Balancing Interaction with RL Gradients: COMPLEX
- Pre-training uses auxiliary load-balancing loss to prevent expert collapse
- During RL, should this loss be kept? Removed? Modified?
- If kept: competes with RL objective, potentially constraining policy improvement
- If removed: risk of expert collapse under noisy RL gradients
- DeepSeek V3 uses "aux-loss-free" dynamic bias during pre-training -- unclear how this interacts with RL
- GLM-5 freezes the DSA indexer during RL but does NOT freeze the MoE router
- **No consensus**: This is an open research question

### Memory Pressure: SIGNIFICANT
- MoE models have more total parameters even though fewer are active per token
- During RL rollout (inference): only active experts needed -- memory similar to dense model of active size
- During RL training (backprop): ALL expert parameters must be in memory for gradient computation
- NanoSeek 4.75B total / 1.08B active: training requires ~9.5GB (BF16) vs ~2.2GB if it were dense
- Expert parallelism (EP) needed at scale, adding communication overhead
- Optimizer state: 4.75B params * 8 bytes (Adam states) = ~38GB -- much larger than equivalent dense model

### GSPO vs GRPO for MoE: WHY SEQUENCE-LEVEL MATTERS

| Property | GRPO (token-level) | GSPO (sequence-level) |
|----------|-------------------|----------------------|
| IS ratio | Per-token: r_t = exp(logp_t - logp_old_t) | Per-sequence: s = exp(mean(logp - logp_old)) |
| Routing noise | Amplified per token | Averaged out across sequence |
| Clipping | Token-level: each token independently clipped | Sequence-level: one clip per completion |
| MoE stability | Requires routing replay for convergence | Converges without routing replay [VERIFIED] |
| Performance | Good for dense, unstable for MoE | Designed for MoE, validated on Qwen3-30B |

**GSPO eliminates the need for routing replay** -- a significant engineering simplification. The sequence-level geometric mean smooths out per-token routing perturbations.

### Overall RL Fit Score: 6/10
**Best for**: Teams that need MoE's inference efficiency AND have engineering capacity to handle routing stability. Requires GSPO or RSPO. Not recommended with vanilla GRPO.

---

## 1C. Hybrid/Novel Architectures

### MLA (Multi-Head Latent Attention) -- DeepSeek/NanoSeek

**RL Rollout Advantage**: MLA compresses KV cache by ~23x (DeepSeek V3) or ~12x vs GQA. During RL, rollout generation is the bottleneck (>90% of training time per APRIL paper). MLA enables:
- Larger batch sizes during rollout (more samples per GPU)
- Longer sequences without OOM
- Higher throughput for the generation phase

**RL Training Consideration**: During backprop, KV cache compression doesn't help (full forward pass computes all projections). The low-rank decomposition adds slight compute overhead during training.

**Muon Interaction**: GLM-5 found that standard Muon treats all MLA attention heads equally, constraining heads that need different update scales. Their "Muon Split" applies Newton-Schulz orthogonalization independently per head. This is relevant for RL where gradient distributions differ from pre-training.

**RL Fit**: GOOD -- the rollout speedup from compressed KV cache directly accelerates the RL bottleneck. Minor training overhead is acceptable.

### Linear Attention Variants -- Kimi Linear, Lightning (MiniMax)

**Kimi Linear (KDA + MLA hybrid)**:
- [VERIFIED] Outperforms full attention during Math RL training throughout the entire RL process
- O(n) inference complexity for linear attention layers enables much faster rollout for long sequences
- Hybrid design (linear + MLA layers) preserves expressiveness while gaining efficiency
- 3B active / 48B total parameters

**Lightning Attention (MiniMax)**:
- 7/8 layers use lightning attention (O(n)), 1/8 use softmax
- Enables 1M context training, 4M inference
- 100 TPS inference speed -- fast rollouts
- At 100K tokens: ~25% FLOPs of DeepSeek-R1

**RL Fit**: EXCELLENT for long-horizon RL tasks (agentic, multi-turn). The O(n) inference directly reduces rollout cost. Key concern: linear attention may have different gradient dynamics than softmax attention during policy updates -- less studied.

### Sliding Window Attention -- Gemma 2/3, Mistral

**Gemma 2/3**: Alternating layers -- every other layer uses sliding window (4096 tokens), intervening layers use global attention (8192 tokens)
**Mistral 7B**: Originally used sliding window for all layers; Mistral-7B-v0.2+ switched to full attention

**RL Implications**:
- Sliding window limits long-range dependency capture per layer
- For reasoning RL (math, code), models need to attend to early problem statement tokens from late reasoning tokens
- Global attention layers compensate, but at reduced capacity
- Memory savings during rollout are modest compared to MLA or linear attention

**RL Fit**: MODERATE -- works fine for short-to-medium context RL, but not ideal for long-horizon agentic RL where full context access matters.

### GQA vs MHA vs MLA for RL -- Comparative Summary

| Property | MHA | GQA | MLA |
|----------|-----|-----|-----|
| KV cache size (relative) | 1.0x | 0.12-0.25x | 0.04-0.08x |
| Rollout throughput | Baseline | 2-4x faster | 4-8x faster |
| Training overhead | Baseline | Same | Slight increase (decomposition) |
| Framework support | Universal | Universal | DeepSeek/GLM-5 only (growing) |
| RL gradient behavior | Well-understood | Same as MHA | Needs Muon Split for optimal RL |
| Expressiveness | Maximum | Slightly reduced | Comparable to MHA (per DeepSeek ablations) |

**Ranking for RL**: MLA > GQA > MHA (for rollout-dominated RL workloads)

---

## 1D. Key Architecture Properties for RL -- Ranked by Importance

| Rank | Property | Why It Matters | Weight |
|------|----------|---------------|--------|
| 1 | **Inference speed** (rollout generation) | Rollout is >90% of RL training time. Faster inference = more samples = better policy. | 25% |
| 2 | **Gradient flow quality** | Clean gradients enable stable policy improvement. MoE routing noise degrades this. | 20% |
| 3 | **Routing stability** (MoE only) | Unstable routing corrupts IS ratios, causing training instability or divergence. | 15% |
| 4 | **Parameter efficiency** (active vs total) | MoE gives more capacity per FLOP, but total memory cost is higher. | 12% |
| 5 | **Representation capacity** | More capacity = more room for policy improvement. Matters at small scale. | 10% |
| 6 | **Context window** | Long context matters for agentic RL (multi-turn, tool use). Less critical for math RL. | 8% |
| 7 | **Attention pattern flexibility** | MLA/linear attention enable different tradeoffs. Affects long-horizon tasks. | 10% |

**Key insight**: For RL, inference speed matters MORE than training efficiency because the rollout bottleneck dominates. This means architectures optimized for inference (MLA, linear attention, MoE's sparse activation) have a structural advantage in RL training throughput, even if they add training complexity.

---

# PART 2: Systems/Infrastructure Engineering Analysis

## 2A. Actor-Learner Topology

### Kimi K2.5: Colocated Hybrid
```
Centralized Controller
  ├── Same GPU workers alternate between:
  │   ├── Inference Engine (rollout generation) ← vLLM
  │   └── Training Engine (gradient updates) ← Megatron
  └── Distributed Checkpoint Engine (RDMA, <30s sync)
```
- **Communication**: Weight transfer happens via in-memory swap, not network
- **Transition time**: <1 min (train→infer), ~10s (infer→train)
- **Scaling bottleneck**: GPU utilization drops during transitions (neither training nor generating)
- **Advantage**: No network transfer for weights; simpler architecture
- **Disadvantage**: GPUs idle during mode transitions; cannot overlap rollout + training

### GLM-5: Fully Separated (Async via Slime)
```
Slime Orchestrator (centralized)
  ├── Inference Cluster (SGLang + Router)
  │   └── FP8, EP64, DP64, MTP speculative decoding
  ├── Training Cluster (Megatron-based)
  │   └── Separate GPU pool
  ├── Environment Pool (K8s containers)
  └── Data Buffer + Weight Sync (RDMA, every K steps)
```
- **Communication**: RDMA weight sync every K training steps + optimizer reset
- **Scaling bottleneck**: Weight sync bandwidth; staleness management (IcePop needed)
- **Advantage**: Maximum GPU utilization (both clusters always active)
- **Disadvantage**: Requires IcePop to handle train-infer policy mismatch; more complex

### MiniMax M2.7: Separated with Forge
```
Agent Side (scaffolds, 100K+)
  ├── Middleware (Gateway + FIFO Scheduler)
  │   └── Windowed FIFO (30% visibility window)
  └── Training/Inference Engine (separated pools)
      └── Prefix Tree Merging (40x speedup multi-turn)
```
- **Communication**: HTTP/gRPC between middleware and engines
- **Scaling bottleneck**: Middleware scheduler throughput; prefix tree memory
- **Advantage**: Scaffold-agnostic (4 interfaces: reprocess/run/postprocess/calculate_reward)
- **Disadvantage**: More middleware overhead; scaffold management complexity

### For NanoSeek (1B-7B): Recommended Topology
**Colocated (Kimi-style)** for simplicity:
- At 1B-7B scale, a single node (1-8 GPUs) can handle both training and inference
- No RDMA infrastructure needed
- Transition overhead is small relative to total training time
- No need for IcePop (no async mismatch)
- Use veRL or OpenRLHF which support colocated mode natively

---

## 2B. Rollout Infrastructure

### Generation Backends

| System | Used By | Key Feature | Throughput |
|--------|---------|-------------|------------|
| vLLM | Kimi, veRL, OpenRLHF | PagedAttention, continuous batching | High (standard) |
| SGLang | GLM-5 (Slime) | RadixAttention, prefix caching | High (long-context optimized) |
| Custom (Forge) | MiniMax | Prefix tree merging, MTP | 100 TPS (lightning attention) |

### Batch Formation and Scheduling

**Kimi**: Centralized controller dispatches prompts. G=4 completions per prompt. Difficulty-filtered prompt sampling (curriculum).

**GLM-5**: Slime orchestrator manages prompt buffer. G=32 for reasoning, G=1 for distillation. APRIL optimization: over-provision rollout requests, terminate early once target reached (44% throughput improvement).

**MiniMax**: Windowed FIFO scheduling with 30% visibility window. Prevents easy-sample dominance (samples from a window of the queue, not just FIFO head). G=16, K=16 gradient steps per generation.

### GPU Utilization During Rollout vs Training

| Phase | GPU Utilization | Bottleneck |
|-------|----------------|------------|
| Rollout (inference) | 40-60% (memory-bound, autoregressive) | KV cache memory, batch size |
| Training (backprop) | 80-95% (compute-bound) | Forward + backward pass |
| Reward computation | 10-30% (depends on verifiable vs model-based) | Environment execution for agentic |

**Key insight**: Rollout is the bottleneck because autoregressive generation is inherently sequential and memory-bound. APRIL (GLM-5) addresses this by over-provisioning and recycling incomplete rollouts.

---

## 2C. Memory and Compute Analysis (1B-7B models)

### Peak Memory During Rollout Generation

| Model Size | Weights (BF16) | KV Cache (4K ctx, batch=32, GQA) | Total |
|------------|----------------|----------------------------------|-------|
| 1B dense | 2 GB | 1.5 GB | ~4 GB |
| 1B MoE (4.75B total) | 9.5 GB | 1.5 GB | ~11 GB |
| 7B dense | 14 GB | 8 GB | ~22 GB |
| 7B MoE (30B total) | 60 GB | 8 GB | ~68 GB |

**Note**: MoE rollout can use expert offloading or EP to reduce per-GPU memory, but adds latency.

### Peak Memory During Training

| Model Size | Weights | Optimizer (Adam) | Gradients | Activations | Total |
|------------|---------|-----------------|-----------|-------------|-------|
| 1B dense | 2 GB | 8 GB | 2 GB | ~4 GB | ~16 GB |
| 1B MoE (4.75B total) | 9.5 GB | 38 GB | 9.5 GB | ~4 GB | ~61 GB |
| 7B dense | 14 GB | 56 GB | 14 GB | ~16 GB | ~100 GB |
| 7B MoE (30B total) | 60 GB | 240 GB | 60 GB | ~16 GB | ~376 GB |

**Critical**: MoE training memory scales with TOTAL params (all experts need gradients), not active params. A 4.75B MoE requires ~4x more memory than a 1B dense model for training.

### Compute Ratio: Rollout Time vs Training Time

| Model Scale | Rollout % | Training % | Reward % | Source |
|-------------|-----------|-----------|----------|--------|
| General (measured) | 85-95% | 3-10% | 2-5% | APRIL paper, multiple frameworks |
| With APRIL optimization | 75-85% | 10-15% | 5-10% | GLM-5 paper |
| With MTP speculative decoding | 70-80% | 15-20% | 5-10% | Estimated from MTP accept length |

**Conclusion**: Rollout dominates. Any architecture or technique that speeds up inference has outsized impact on total RL training time.

### Communication Overhead

| Topology | Overhead | When |
|----------|----------|------|
| Colocated (Kimi) | ~1-5% (mode transitions) | Every rollout-train cycle |
| Separated async (GLM-5) | ~5-15% (weight sync RDMA) | Every K training steps |
| Separated sync (veRL default) | ~10-20% (blocking weight transfer) | Every train step |

---

## 2D. Implementation Complexity Ranking

### Ranking by Lines of Code to Implement

| Rank | Pipeline | Estimated LoC | Why |
|------|----------|---------------|-----|
| 1 (simplest) | MiniMax CISPO | ~500-800 | Single loss function, no async, no special infra |
| 2 | Kimi Mirror Descent | ~800-1200 | Simple loss, but colocated engine switching adds complexity |
| 3 (most complex) | GLM-5 IcePop+Slime | ~2000-3000 | 5 stages, async architecture, IcePop ratio tracking, cross-stage distillation |

### Ranking by Number of Moving Parts

| Rank | Pipeline | Moving Parts | Components |
|------|----------|-------------|------------|
| 1 | CISPO (MiniMax) | 4 | Policy, reference, reward fn, optimizer |
| 2 | Kimi Mirror Descent | 6 | Policy, reference, reward fn, optimizer, checkpoint engine, controller |
| 3 | GLM-5 Full Pipeline | 10+ | Policy, reference, reward fn, optimizer, Slime orchestrator, SGLang rollout, data buffer, K8s environments, distillation teachers, IcePop monitor |

### Ranking by Debugging Difficulty

| Rank | Pipeline | Difficulty | Why |
|------|----------|-----------|-----|
| 1 (easiest) | CISPO | LOW-MEDIUM | .detach() makes gradient flow predictable; but entropy collapse is hard to diagnose |
| 2 | Kimi Mirror Descent | MEDIUM | Squared loss is well-behaved, but Muon+RL interaction is poorly understood |
| 3 | GLM-5 IcePop | HIGH | Async mismatch is hard to debug; IcePop suppression rate hard to tune; 5 stages multiply debugging surface |

### Ranking by Reproducibility

| Rank | Pipeline | Reproducibility | Why |
|------|----------|----------------|-----|
| 1 | CISPO | HIGH | Full equation + code published; ScaleRL independent validation |
| 2 | Kimi | MEDIUM | Full derivation published, but exact hyperparameters (tau, lr schedule) unknown |
| 3 | GLM-5 | LOW-MEDIUM | IcePop described but no ablation; Slime is open-source but many undisclosed hyperparameters |

### Ranking by Time to First Working Experiment

| Rank | Pipeline | Time (for 1B model) | Prerequisites |
|------|----------|-------------------|---------------|
| 1 | CISPO + veRL/OpenRLHF | 1-2 days | Install framework, write loss fn, run |
| 2 | GSPO + veRL | 1-2 days | Same as above, GSPO is ~20 lines |
| 3 | Kimi Mirror Descent (custom) | 3-5 days | Need to implement custom loss, test Muon interaction |
| 4 | GLM-5 full pipeline (Slime) | 1-2 weeks | Setup Slime, configure SGLang, implement IcePop, test async |

---

## 2E. Available Open-Source Frameworks (March 2026)

### veRL (Volcano Engine RL) -- ByteDance
- **Repository**: github.com/verl-project/verl
- **Supported algorithms**: PPO, GRPO, REINFORCE++, RLOO, PRIME, ReMax
- **Supported models**: Qwen 2.5, Llama 3.x, Gemma 2, DeepSeek-LLM (via HuggingFace Transformers)
- **Training backends**: FSDP, Megatron-LM
- **Inference backends**: vLLM, SGLang (experimental), HF Transformers
- **Scale**: Up to 70B models, hundreds of GPUs
- **Maturity**: HIGH -- production-tested at ByteDance, active development
- **MoE support**: Via HuggingFace model loading (not natively optimized for expert parallelism)
- **Known issues**: SGLang integration is experimental; large-scale MoE requires manual EP configuration
- **Code size**: ~32,325 lines (feature-rich but complex)
- **Best for**: Teams that want a batteries-included solution with Ray-based orchestration

### OpenRLHF
- **Repository**: github.com/OpenRLHF/OpenRLHF
- **Supported algorithms**: PPO, REINFORCE++, GRPO, RLOO, DAPO, TIS
- **Supported models**: Any HuggingFace model including MoE (via --aux_loss_coef flag)
- **Training backends**: DeepSpeed
- **Inference backends**: vLLM
- **Scale**: Up to 70B, distributed via Ray
- **Maturity**: HIGH -- widely used in research (ProRL V2 used it for SOTA 1.5B reasoning model)
- **MoE support**: YES (via HuggingFace + aux_loss_coef)
- **Known issues**: Less efficient than veRL at very large scale; DeepSpeed-only (no Megatron)
- **Code size**: ~8,523 lines (concise, high performance)
- **Best for**: Researchers who want simplicity and fast iteration

### TRL (HuggingFace Transformers Reinforcement Learning)
- **Repository**: github.com/huggingface/trl
- **Supported algorithms**: PPO, DPO, GRPO, SFT, ORPO, KTO
- **Supported models**: Any HuggingFace model
- **Training backends**: Accelerate, PEFT (LoRA)
- **Inference backends**: HuggingFace generate
- **Scale**: Single-node optimized (multi-node possible but not primary focus)
- **Maturity**: HIGH -- most popular for small-scale experiments
- **MoE support**: Basic (via HuggingFace model loading)
- **Known issues**: Not optimized for large-scale distributed RL; slower than veRL/OpenRLHF for production
- **Code size**: ~19,071 lines
- **Best for**: Quick prototyping, HuggingFace ecosystem integration, LoRA fine-tuning

### Slime (GLM's Framework) -- Zhipu AI
- **Repository**: github.com/THUDM/slime
- **Supported algorithms**: GRPO, DAPO, GSPO
- **Supported models**: GLM-5/4.7/4.6/4.5, Qwen3/2.5, DeepSeek V3, Llama 3
- **Training backends**: Megatron-based
- **Inference backends**: SGLang + custom router
- **Scale**: Designed for large-scale (100K+ GPUs on Ascend)
- **Maturity**: MEDIUM -- open-source but primarily designed for Zhipu's infrastructure
- **MoE support**: NATIVE (built for GLM-5's 256-expert MoE)
- **Known issues**: Ascend-first design may have rough edges on NVIDIA; less community support
- **APRIL integration**: YES (44% rollout throughput improvement)
- **Best for**: Teams with Megatron expertise who want native MoE support and async training

### torchforge (Meta/PyTorch)
- **Repository**: github.com/meta-pytorch/torchforge
- **Supported algorithms**: Custom (build-your-own via Monarch actors)
- **Supported models**: Any PyTorch model (via torchtitan for training, vLLM for inference)
- **Training backends**: torchtitan
- **Inference backends**: vLLM
- **Scale**: Designed for large-scale distributed
- **Maturity**: LOW -- EXPERIMENTAL, under active development, APIs may change
- **Known issues**: Very new, limited documentation, requires understanding of Monarch actor model
- **Best for**: Meta-internal teams and early adopters willing to deal with instability

### Framework Comparison Matrix

| Framework | Simplicity | Performance | MoE Support | Scale | Community | Best Algorithm for NanoSeek |
|-----------|-----------|-------------|-------------|-------|-----------|---------------------------|
| veRL | MEDIUM | HIGH | Basic | 70B+ | Active | GRPO (native), GSPO (custom) |
| OpenRLHF | HIGH | HIGH | Basic | 70B+ | Very active | GRPO (native), CISPO (custom) |
| TRL | VERY HIGH | MEDIUM | Basic | ~13B | Largest | GRPO (native) |
| Slime | LOW | HIGH | NATIVE | 100K+ GPU | Small | GSPO (native) |
| torchforge | LOW | HIGH (projected) | Unknown | Large | Nascent | Custom |

### Recommendation for NanoSeek (1B MoE)
**Primary**: OpenRLHF or veRL with custom GSPO loss
- Both support MoE models via HuggingFace
- Both have vLLM integration for fast rollouts
- GSPO can be implemented as a custom loss (~20 lines) on top of either framework
- OpenRLHF is simpler (8.5K LoC vs 32K); veRL is more feature-rich

**Secondary**: Slime if native MoE + async is needed later
- Better MoE support but higher complexity
- GSPO is natively supported

---

## 2F. Cost Analysis

### GPU Hours for RL Training (1B-7B models)

Estimates based on published data points and scaling:

| Model Size | Architecture | G (group) | Steps | Estimated GPU-Hours | Source/Basis |
|------------|-------------|-----------|-------|-------------------|-------------|
| 1B | Dense | 16 | 5,000 | 50-100 (1x A100) | Extrapolated from JustRL |
| 1B | MoE (4.75B total) | 16 | 5,000 | 100-200 (1x A100) | ~2x dense due to total param memory |
| 3B | Dense | 16 | 10,000 | 200-400 (2x A100) | Linear scaling |
| 7B | Dense | 16 | 10,000 | 500-1,000 (4x A100) | Extrapolated from M1 (3 weeks, 512 H800 for 45B active) |
| 7B | MoE (30B total) | 16 | 10,000 | 1,000-2,000 (8x A100) | ~2x dense |

**Reference point**: MiniMax M1 (45.9B active / 456B total) took 3 weeks on 512 H800 GPUs = ~258,000 GPU-hours. Scaling down to 7B active: ~258K * (7/45.9) * (rollout_ratio_adjustment) ~ 2,000-5,000 GPU-hours on H100.

### Recommended Hardware

| Model Size | Recommended GPU | Minimum GPUs | Why |
|------------|----------------|-------------|-----|
| 1B dense | A6000 (48GB) or A100 (40GB) | 1 | Fits in single GPU with room for KV cache |
| 1B MoE (4.75B) | A100 (80GB) | 1 | Need 80GB for optimizer states |
| 3B dense | A100 (80GB) | 1-2 | Comfortable fit with large batch |
| 7B dense | A100 (80GB) or H100 | 2-4 | Need multi-GPU for optimizer + KV cache |
| 7B MoE (30B) | H100 (80GB) | 4-8 | Total params require distributed training |

### Cloud Cost Estimates (March 2026)

| Model | GPU Config | Provider | Hourly Cost | RL Duration | Total Cost |
|-------|-----------|----------|------------|-------------|------------|
| 1B dense | 1x A100 80GB | RunPod | $1.39/hr | 50-100 hrs | $70-$140 |
| 1B MoE (NanoSeek) | 1x A100 80GB | RunPod | $1.39/hr | 100-200 hrs | $140-$280 |
| 1B MoE (NanoSeek) | 1x A6000 48GB | RunPod | $0.58/hr | 150-300 hrs | $87-$174 |
| 7B dense | 4x A100 80GB | RunPod | $5.56/hr | 125-250 hrs | $695-$1,390 |
| 7B dense | 2x H100 SXM | RunPod | $5.38/hr | 80-160 hrs | $430-$860 |
| 7B MoE (30B) | 8x A100 80GB | RunPod | $11.12/hr | 125-250 hrs | $1,390-$2,780 |
| 7B MoE (30B) | 4x H100 SXM | RunPod | $10.76/hr | 80-160 hrs | $860-$1,720 |

**NanoSeek-specific estimate**:
- RL training for 1B MoE on 1x A100 80GB: **$140-$280** (RunPod)
- RL training for 1B MoE on 1x A6000 48GB: **$87-$174** (RunPod) -- may be tight on memory, need gradient checkpointing
- Budget-optimal: Start on A6000 for algorithm development, move to A100 for final runs

---

# SYNTHESIS: Rankings and Recommendations

## Architecture Ranking for RL Post-Training

| Rank | Architecture | RL Fit | Best Use Case | Key Risk |
|------|-------------|--------|---------------|----------|
| 1 | Dense + GQA (Llama 3, Qwen-dense) | 9/10 | Maximum simplicity, fast iteration | Lower capacity per FLOP |
| 2 | Dense + MLA (if available) | 8.5/10 | Fast rollouts + simplicity | Limited framework support |
| 3 | MoE + MLA + GSPO (NanoSeek, DeepSeek) | 7/10 | Best capacity/FLOP with stable RL | Routing instability (mitigated by GSPO) |
| 4 | MoE + Linear Attention (Kimi Linear) | 7/10 | Ultra-fast rollouts for long-horizon RL | Least studied for RL |
| 5 | MoE + GQA + GSPO (Qwen3-MoE) | 6.5/10 | Proven at scale (Qwen3) | Standard MoE challenges |
| 6 | MoE + Lightning (MiniMax) | 6/10 | 1M+ context agentic RL | Proprietary, entropy collapse risk |

## Pipeline Ranking for NanoSeek

| Rank | Pipeline | Feasibility | Time to Working | Cost | Best For |
|------|----------|-------------|----------------|------|---------|
| 1 | GSPO + OpenRLHF/veRL | HIGH | 1-2 days | $140-280 | **Recommended starting point** |
| 2 | CISPO + OpenRLHF | HIGH | 1-2 days | $140-280 | Alternative if GSPO underperforms |
| 3 | Kimi Mirror Descent (custom) | MEDIUM | 3-5 days | $140-280 | If principled optimization theory matters |
| 4 | GLM-5 full pipeline (Slime) | LOW | 1-2 weeks | $280-560 | If async + MoE-native needed |

## Final Recommendations for NanoSeek

1. **Start with GSPO on OpenRLHF** -- simplest path, designed for MoE, ~20 lines of custom loss code
2. **Use A100 80GB** for RL training (single GPU sufficient for 1B MoE)
3. **Monitor I_spec during RL** -- this would be a genuinely novel contribution
4. **Implement CISPO as ablation** -- compare GSPO vs CISPO on MoE to empirically validate which preserves expert specialization better
5. **Do NOT use vanilla GRPO** for MoE -- routing instability is well-documented
6. **Do NOT build async infrastructure** at 1B scale -- colocated is sufficient and much simpler
7. **Budget**: $150-300 for the full RL training phase on cloud GPUs

---

## Sources

- [GSPO: Towards Scalable RL for Language Models (Qwen)](https://qwenlm.github.io/blog/gspo/)
- [GSPO Paper (arXiv:2507.18071)](https://arxiv.org/abs/2507.18071)
- [RSPO: Stable RL for MoE (arXiv:2510.23027)](https://arxiv.org/html/2510.23027v1)
- [Stabilizing MoE RL by Aligning Routers (arXiv:2510.11370)](https://arxiv.org/html/2510.11370v1)
- [OpenRLHF GitHub](https://github.com/OpenRLHF/OpenRLHF)
- [veRL GitHub](https://github.com/volcengine/verl)
- [Slime GitHub (THUDM)](https://github.com/THUDM/slime)
- [torchforge (Meta PyTorch)](https://github.com/meta-pytorch/torchforge)
- [Anatomy of RL Frameworks](https://www.hanifleo.com/anatomy-of-rl-frameworks/)
- [Keep the Tokens Flowing: 16 Open-Source RL Libraries (HuggingFace)](https://huggingface.co/blog/async-rl-training-landscape)
- [OpenRLHF vs veRL Deep Dive](https://langcopilot.com/posts/2025-11-06-openrlhf-vs-verl-ray-framework-deep)
- [GSPO vs GRPO for MoE Models](https://kaitchup.substack.com/p/gspo-vs-grpo-reinforcement-learning)
- [Kimi Linear: Expressive Efficient Attention (arXiv:2510.26692)](https://arxiv.org/abs/2510.26692)
- [MLA Explained (HuggingFace)](https://huggingface.co/blog/NormalUhr/mla-explanation)
- [TransMLA (arXiv:2502.07864)](https://arxiv.org/abs/2502.07864)
- [GPU Cloud Pricing Comparison 2026](https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/)
- [RunPod GPU Pricing](https://www.runpod.io/pricing)
- [H100 vs A100 Price Reality Check 2026](https://www.gpu.fund/blog/h100-vs-a100-price-reality-check-2026)
- [AI Training Costs 2026 (GPUnex)](https://www.gpunex.com/blog/ai-training-costs-2026/)
