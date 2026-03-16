# Paper Analysis: DeepSeek-V3 and DeepSeek-V3.2
## What the Papers Reveal vs. What Our Project Plans

**Papers analyzed:**
- DeepSeek-V3 Technical Report — arXiv:2412.19437 (Feb 2025)
- DeepSeek-V3.2 — arXiv:2512.02556 (Dec 2025)

**Compared against:** SCALING_LAB_PLAN.md, REIMPLEMENTATION_PLAN.md, current model/ code

---

## Executive Summary

Reading both papers against our project plan reveals **5 critical corrections** (things
we planned wrong), **8 new components** to add, and **3 scaling insights** that update
the SCALING_LAB_PLAN. Priority-ordered below.

---

## Part 1: Critical Corrections — Things We Got Wrong

These are bugs in our *plan*, not just in the current code. Fix before implementing.

---

### Correction 1 — DSA Indexer Loss: KL-Divergence, NOT Entropy

**What our plan says** (REIMPLEMENTATION_PLAN.md, Section 8):
> "Indexer trains via KL-divergence alignment with dense attention"
> (mentioned vaguely)

**What the current code does** (`model.py:_compute_indexer_loss`):
```python
entropy = -(probs * log_probs).sum(dim=-1).mean()
return entropy * self.sparse_config.indexer_loss_weight
```
This is an **entropy regularizer** — it pushes the indexer to assign attention broadly.
This is the wrong objective entirely.

**What V3.2 specifies** (Section 2.1.1, Eq. 3 — Stage 1):
```
L_I = sum_t D_KL( p_{t,:} || Softmax(I_{t,:}) )

where:
  p_{t,:}    = actual attention distribution from the main MLA (L1-normalized, summed across heads)
  I_{t,:}    = indexer scores over all key positions
  D_KL       = KL divergence, direction: main attention IS the target
```

**Stage 2 variant** (Eq. 4 — sparse training, only over selected tokens):
```
L_I = sum_t D_KL( p_{t, S_t} || Softmax(I_{t, S_t}) )

where S_t = set of top-k tokens selected by indexer for token t
```

**Why this matters:** Entropy maximization encourages the indexer to be uncertain.
KL-divergence alignment teaches the indexer to *mimic* the main attention pattern.
These are opposite objectives. Our current implementation trains the indexer wrong.

**Fix in model.py** (`_compute_indexer_loss`):
```
1. Run the main MLA forward (dense) to get attention weights p_{t,:}
2. Run the indexer to get I_{t,:}
3. L_I = F.kl_div(log_softmax(I_{t,:}), p_{t,:}.detach(), reduction='batchmean')
Note: detach() on p — indexer doesn't affect main model's gradients
```

---

### Correction 2 — DSA Two-Stage Has a Specific Learning Rate

**What our plan says:** Two-phase training conceptually correct (dense warm-up →
sparse training), but unspecified LR for the indexer-only stage.

**What V3.2 specifies** (Section 2.1.1):
- **Stage 1 (Dense Warm-up):** ~1000 steps, 2.1B tokens
  - Main model parameters: **FROZEN** (all of them)
  - Indexer parameters only: trained at **LR = 1e-3** (extremely high vs. main model)
  - Loss: KL-div between indexer and full dense attention distribution (Eq. 3)
  - Rationale: indexer needs to learn quickly from scratch; high LR is fine since only
    its small parameter set is changing

- **Stage 2 (Sparse Training):** 15,000 steps, 943.7B tokens
  - All model parameters: trained at **LR = 7.3e-6** (same as context extension phase)
  - Indexer: **detached from main model's computational graph**
    - Indexer trains on `L_I` (KL over selected tokens only)
    - Main model trains on standard LM loss
    - The two losses are separate — indexer gradient does NOT flow into main model
  - Data: same distribution as 128K long-context extension data

**What this means for NanoSeek:**
- Phase 1 (Dense): indexer trains via KL-div even during main pre-training, but at
  the main model's LR (1e-4 range) — not the high LR V3.2 uses for warm-up
- Our plan should add an explicit "indexer warm-up phase" before Phase 2 where
  the main model is frozen and indexer trains at high LR
- In Phase 2, indexer loss must be computed with `detach()` on indexer inputs —
  indexer gradient must not flow into MLA's compressed representations

---

### Correction 3 — DSA Uses MQA Mode, Not MHA Mode

**What our plan says:** DSA wraps MLA and uses standard MLA attention.

**What V3.2 specifies** (Appendix A):
- V3.1-Terminus uses **MHA mode** (Multi-Head Attention) for training and prefilling
- For **decoding**: MQA mode (Multi-Query Attention — single KV shared across all heads)
- V3.2 DSA uses **MQA mode for both training and decoding** (throughout, not just decode)

**MQA mode of MLA** means:
- The KV latent vector `c_t^{KV}` is NOT expanded per head
- Instead, one KV vector serves all query heads
- This is already the compressed form — `c_t^{KV}` of dim `d_c` rather than
  `n_h × (d_nope + d_v)` per head
- The `wkv_b` expansion (latent → per-head KV) happens **inside the attention kernel**,
  not as a separate materialized tensor

**Why this matters for sparse attention:**
In sparse mode, we gather `k` tokens' `c_t^{KV}` vectors, then expand them via `wkv_b`.
MQA mode means we do this expansion ONCE per position, not `n_h` times. This is where
the efficiency comes from — the KV tokens are still compressed when gathered.

**Fix for DSA sparse path:**
- Gather the compressed `kv_c` vectors (shape `[B, S, K, kv_lora_rank]`)
- Expand via `wkv_b` AFTER gathering (not before)
- Current code expands first then gathers — wrong order, much more expensive

---

### Correction 4 — γ Freeze Point Is 96.6%, Not 80%

**What our plan says** (config.py `MoEConfig`):
```python
gamma_freeze_ratio: float = 0.80  # Freeze bias at 80% of training
```

**What V3 specifies** (Section 2.1.2):
> γ = 0.001 for first 14.3T tokens, then 0.0 for remaining 500B tokens

`14.3T / 14.8T = 0.966` — freeze at **96.6%**, not 80%.

**Why the difference matters:** At 80%, we freeze load balancing with 20% of training
still to go. If routing has not fully stabilized by that point, the final 20% will
have increasing load imbalance — potentially hurting the final model quality.

**Recommended change:** Set `gamma_freeze_ratio = 0.95` for NanoSeek
(conservative buffer from 0.966, but much better than 0.80).

**Also:** The V3 paper notes that batch-wise load balancing achieves the same quality
as aux-loss-free, while allowing more expert specialization. Worth testing this
variant in the ablation (replace our Run A's traditional aux-loss with batch-wise).

---

### Correction 5 — MTP Depth D=1 and Sequential Chain

**What our plan says:** MTP correct in concept (standard transformer block, shared
embeddings).

**What V3 specifies** (Section 2.2):
- D = 1 (only ONE extra MTP module, predicting the token at offset +1)
- This means NanoSeek predicts 2 tokens total: main model (offset 0) + MTP (offset 1)
- The modules are **sequential** — module k's hidden state `h^k` feeds into module k+1
  (not parallel independent heads)
- Each module k has its **own distinct transformer block TRM_k** and projection M_k
  (not shared across modules)
- The **only things shared** across modules and main model: embedding Emb(·) and
  output head OutHead(·)

**Quantitative result from V3 ablation:**
- With MTP (D=1): HumanEval +9.2, GSM8K +1.7, MATH +1.2 at 228.7B param scale
- MTP second-token acceptance rate in speculative decoding: **85–90%** → **1.8× TPS**

This is important for the test suite: we should validate acceptance rate reaches
>80% on held-out text after training, not just that MTP loss is finite.

---

## Part 2: New Components to Add

Ordered by implementation priority for NanoSeek.

---

### Addition 1 — FIM (Fill-in-Middle) Training [HIGH PRIORITY]

**Source:** V3 Technical Report, Section 4.1

**What it is:**
10% of training tokens use the Fill-in-Middle format using PSM framework:
```
PSM (Prefix-Suffix-Middle):
  <|fim_prefix|> [prefix text] <|fim_suffix|> [suffix text] <|fim_middle|> [middle text]

  Input to model: prefix + suffix
  Model must predict: middle content

SPM (Suffix-Prefix-Middle) — alternative layout at smaller fraction
```

**Why it matters:** Without FIM, the model learns "complete from left to right" only.
With FIM, the model understands bidirectional context — critical for code completion
(editing in the middle of a function) and fills a genuine capability gap that is
observable in early eval benchmarks (HumanEval fill-in-middle variants).

**What to add to NanoSeek:**
- `scripts/dataset.py`: Add `apply_fim_transform(sample, rate=0.1)` function
- Apply PSM layout at 10% probability during tokenization/collation
- Add `<fim_prefix>`, `<fim_suffix>`, `<fim_middle>` special tokens to vocabulary
- In the training loop: track FIM vs non-FIM loss separately for debugging

**Impact on scaling law experiments:** FIM changes the data distribution. For the
scaling law sweep (Series A/B/C), either: (a) apply FIM consistently across all runs
for an apples-to-apples comparison, or (b) run the law without FIM first, then add FIM
for the main 1B run. Option (a) is cleaner but requires ensuring FIM tokens don't
confuse the bits-per-byte metric (they're part of the natural distribution).

---

### Addition 2 — EMA (Exponential Moving Average) Parameter Tracking [MEDIUM PRIORITY]

**Source:** V3 Technical Report, Section 4.3

**What it is:**
Maintain an exponential moving average of model weights in CPU memory:
```python
ema_weights[k] = beta_ema * ema_weights[k] + (1 - beta_ema) * model_weights[k]
```
Updated asynchronously (CPU thread, no GPU overhead). Used for evaluation checkpoints
— EMA model often has better eval metrics than the current training weights (less
noise in the weights).

**What to add:**
- `training_ops/ema_tracker.py`: lightweight EMA updater
  - `beta_ema = 0.99` (or `1 - 1/num_steps` schedule)
  - CPU copy updated every N steps asynchronously
  - `eval_with_ema()`: swap in EMA weights for evaluation, restore afterwards
- Run eval harness on EMA weights, not live training weights
- Log EMA loss separately to W&B to distinguish from training loss fluctuations

**Why this is especially valuable for MoE:** The dynamic bias terms `b_i` in load
balancing are updated every step and can fluctuate. EMA smooths over this. The EMA
model is a stable snapshot of the model's learned representations.

---

### Addition 3 — Batch Size Schedule [MEDIUM PRIORITY]

**Source:** V3 Technical Report, Section 4.2

**V3 batch size schedule:**
- Start: 3072 sequences
- End: 15360 sequences
- Transition: over first 469B tokens (~3% of total training)

**Why small batch early then large batch later:**
- Small batch early = more frequent weight updates = faster initial learning
- Large batch late = more stable gradients = better convergence near the end
- This is especially important for MoE: small batches early mean more noisy expert
  routing, but the model can "explore" more expert assignments before settling

**What to add to NanoSeek:**
- `scripts/scheduler.py`: Add `BatchSizeScheduler`
  - Token-based ramp (not step-based, for scale invariance)
  - Linear or cosine interpolation from `batch_start` to `batch_target`
  - Typically: ramp over first 3-5% of total tokens
- For our 22B token run: ramp from batch=64 to batch=128 over first 0.66B tokens

**Interaction with scaling law experiments:** Scaling sweep runs (Series A/B/C) are
short runs. Use fixed batch size for sweeps (simpler comparison). Only apply batch
ramp to the main 1B training run.

---

### Addition 4 — Expert Specialization Analysis [OBSERVABILITY]

**Source:** V3 Technical Report, Section 2.1.2 and Figure 3

**What V3 found:** With aux-loss-free load balancing, experts develop domain-specific
specialization. Examples from V3:
- GitHub code tokens → routes to experts {3, 17, 42, 61}
- DeepMind Mathematics tokens → routes to experts {7, 23, 31, 55}
- Wikipedia text → routes to experts {1, 12, 29, 44}

This specialization does NOT emerge with traditional auxiliary loss (which forces
every expert to process roughly equal amounts of every domain).

**What to add to NanoSeek:**
- `stability_engine/expert_specialization.py`:
  ```
  analyze_expert_routing(model, dataset_by_domain):
    For each domain (code, math, text, etc.):
      Run 1000 tokens through model
      Record which experts each token routes to
      Compute expert co-occurrence matrix per domain
      Return: per-domain "preferred experts" (top-5 by frequency)
  ```
- Run this analysis at checkpoints (every 2000 steps)
- Log to W&B as a heatmap: rows = domains, cols = experts, values = routing frequency
- Early sign of healthy training: experts should start differentiating by step 2000-3000

**Why this matters for ablation Run A vs C:**
If aux-loss-free (Run C/D) shows domain specialization and traditional aux-loss
(Run A) shows uniform routing across domains — this is a qualitative difference
beyond the quantitative PPL gap. It explains WHY aux-loss-free eventually wins.

---

### Addition 5 — MTP Speculative Decoding Acceptance Rate Metric [EVALUATION]

**Source:** V3 Technical Report, Section 2.2

**V3 result:** 85–90% acceptance rate for second-token → 1.8× TPS improvement.

**What to add to NanoSeek:**
- `model/eval/speculative_eval.py`:
  ```
  compute_acceptance_rate(model, eval_tokens, temperature=0.0):
    For each position i in eval_tokens:
      Draft token = MTP module's top-1 prediction at position i
      Verify: does draft_token == eval_tokens[i+1]?
      acceptance_rate = count(verified) / total
  ```
- Add to evaluation harness (run every 500 steps during training)
- Target: acceptance rate should reach >75% by end of Phase 1, >85% by end of training
- If acceptance rate < 50% at end of training: MTP is not working (debug signal)

**Connect to scaling law experiments:** Track acceptance rate for each Series A config.
Hypothesis: acceptance rate scales with training compute, not with model size alone.
If validated, acceptance rate becomes a **fast proxy metric** for model quality
(cheaper to compute than full benchmarks).

---

### Addition 6 — GRPO Post-Training Infrastructure [FUTURE — PHASE 2]

**Source:** V3.2 Technical Report, Section 3

**Four novel GRPO stabilization techniques** (all new, not in V3):

**6a. Unbiased KL Estimate** (Eq. 7):
Standard GRPO uses K3 estimator for KL(π_θ || π_ref). K3 is biased when π_θ deviates
strongly from π_ref (unbounded weights). V3.2 corrects with importance-sampling ratio:
```
D_KL_corrected = (π_θ/π_old) × (π_ref/π_θ - log(π_ref/π_θ) - 1)
```
Prevents GRPO from over-penalizing large policy updates.

**6b. Off-Policy Sequence Masking** (Eqs. 8-9):
When negative-advantage sequences have drifted far from the sampling policy
(high KL between π_old and π_θ), mask them from the gradient:
```
M_{i,t} = 0 if A_{i,t} < 0 AND KL_divergence(π_old, π_θ) > δ
```
Prevents learning from stale negative examples that are no longer "near" current policy.

**6c. Keep Routing:**
Save expert routing decisions from inference framework.
Apply the same routing during RL training step.
Prevents abrupt shifts in active parameter subspace during RL.
V3.2 calls this "crucial for RL training stability of MoE models."

**6d. Keep Sampling Mask:**
Save top-p/top-k truncation masks during rollout generation.
Apply them to π_θ during policy update step.
Ensures π_θ and π_old share the same action space → prevents language consistency failure.

**What to add to NanoSeek (later — not in initial pretraining):**
- `scripts/rl_train.py`: GRPO training loop with all 4 techniques
- Priority: implement after pretraining is validated
- "Keep Routing" is most critical — implement before "Keep Sampling Mask"

---

### Addition 7 — Node-Limited Routing [DISTRIBUTED TRAINING NOTE]

**Source:** V3 Technical Report, Section 2.1.2

**V3 approach:** Each token routed to at most M=4 nodes (to limit all-to-all
communication). Within the 4-node budget: select K_r=8 experts by summing K_r/M=2
highest-affinity experts per node.

**For NanoSeek (small scale — single or few GPUs):** This is not needed now.
BUT: if scaling to 8-GPU training (EP=8, one expert group per GPU), implement
node-limited routing. Without it, every token potentially goes to all 8 GPUs for
expert computation — 8× communication overhead.

**Add to future distributed plan (not MVP):**
- `model/parallel/expert_routing.py`: Node-limited routing implementation
- Config parameter: `max_routing_nodes: int = 1` (default, no constraint)
- Activate when `expert_parallel_size > 4`

---

### Addition 8 — Thinking Context Management [AGENT POST-TRAINING]

**Source:** V3.2 Technical Report, Section 3.2.1

**The insight:** Standard reasoning models (like R1) discard chain-of-thought at the
end of each turn. This causes redundant re-reasoning on every message in multi-turn
agentic scenarios.

V3.2's fix: In multi-turn tool-use scenarios, retain `<think>` traces as long as
only tool response messages arrive. Discard only when a new human message appears.

**Quantitative impact (Table 9):** Thinking context management adds ~4-8 points on
agentic benchmarks (τ²-Bench, MCP-Universe, MCP-Mark) at no additional compute cost.

**Add to NanoSeek (inference system, post-training):**
- `scripts/inference.py`: Context management logic
- Rule: retain thinking traces across tool_result messages, clear on human messages
- This is an inference-time change — no training required

---

## Part 3: Scaling Insights from the Papers

These update the SCALING_LAB_PLAN.md directly.

---

### Insight 1 — RL Compute Budget Scaling

**Source:** V3.2 Technical Report, Introduction

> "RL compute budget for V3.2 already exceeds 10% of pre-training cost —
> consistent performance improvements observed with extended RL training budget"

**Implication for NanoSeek:**
This is an early observation of **RL scaling laws** — performance continues improving
as RL compute scales. For NanoSeek's 22B token pre-training at ~$300 cost:
- 10% of pre-training RL budget = ~$30 of GPU compute for RL fine-tuning
- This is tractable. Include in the project timeline as Phase 2 post-training.

**What to add to SCALING_LAB_PLAN.md:** A Pillar 4 sketch:
```
Pillar 4: RL Scaling (Phase 2)
- Budget: 10-30% of pre-training compute
- Technique: GRPO with all 4 V3.2 stabilization techniques
- Metrics: acceptance rate, benchmark scores, agentic task performance
- Scale: 3-5 RL runs at different compute budgets to observe the scaling trend
```

---

### Insight 2 — Expert Specialization as a Quality Signal

**Source:** V3 Technical Report, Figure 3 and surrounding analysis

**V3's finding:** Aux-loss-free routing leads to domain specialization by experts.
Traditional auxiliary loss suppresses this specialization to force balance.
The domain specialization is CAUSALLY LINKED to better model quality (not just
a correlate) — because specialized experts develop stronger domain representations.

**Implication for our ablation design (Pillar 2):**
Our ablation (Runs A vs C vs D) should measure **expert specialization entropy**,
not just final loss. Expected finding:
- Run A (traditional aux-loss): low specialization entropy (experts serve all domains)
- Run C/D (aux-loss-free): high specialization entropy (experts prefer certain domains)
- Run F (no balancing): partial collapse — some experts see no traffic

This transforms the ablation from "which config has better loss?" to "why does
aux-loss-free produce better models?" — a much richer story.

**Add to stability_engine/experiments/ablation_matrix.py:**
- At end of each ablation run: run `analyze_expert_routing()` on 3-5 domain splits
- Include domain-expert heatmap in STABILITY_PLAYBOOK.md

---

### Insight 3 — Token Efficiency Gap Is the Key Frontier

**Source:** V3.2 Technical Report, Section 4 and Table 3

**V3.2-Speciale vs Gemini-3.0-Pro on Codeforces:**
- V3.2-Speciale: rating 2701, using **77K tokens** per problem
- Gemini-3.0-Pro: rating 2708, using **22K tokens** per problem
- Same performance, 3.5× more tokens — V3.2 explicitly identifies this as key future work

**What this means for NanoSeek's speculative decoding / MTP:**
The acceptance rate metric (Addition 5) connects to this. A model that generates
more tokens to reach the same answer is less token-efficient. Higher MTP acceptance
rate → model is more "certain" about the next token → potentially more token-efficient.

**Add to Pillar 1 scaling law analysis (`fit_scaling_law.py`):**
- Track "tokens per correct answer" on a fixed benchmark subset across the depth sweep
- Hypothesis: smaller models need more tokens to reach correct answers (lower efficiency)
- This gives a second scaling law: token efficiency as a function of model size
- May be the more practical metric for deployment (cost = tokens × serving cost)

---

## Priority-Ordered Action Items

**Implement before writing any model code:**

1. ✅ Understand DSA KL-div indexer loss (Correction 1) — affects model.py Section 9
2. ✅ Understand DSA MQA mode (Correction 3) — affects DSA gather order in Section 9
3. ✅ Update γ freeze ratio to 0.95 (Correction 4) — one-line config fix

**Implement alongside model code:**

4. Add FIM transform to dataset pipeline (Addition 1) — needed before any training
5. Fix DSA two-stage LR in pre-train.py (Correction 2) — affects Phase 1→2 transition
6. Add batch size scheduler (Addition 3) — main training run only

**Implement after model is working:**

7. Add EMA tracker (Addition 2) — improves evaluation quality
8. Add MTP acceptance rate metric (Addition 5) — key evaluation signal
9. Add expert specialization analysis (Addition 4) — transforms ablation story
10. Update ablation design to include specialization entropy (Insight 2)

**Phase 2 (post-pretraining):**

11. GRPO infrastructure with all 4 V3.2 techniques (Addition 6)
12. Thinking context management (Addition 8)
13. RL scaling mini-study (Insight 1)

---

## Updated Corrections to REIMPLEMENTATION_PLAN.md

The following sections of REIMPLEMENTATION_PLAN.md need updating based on this analysis:

| Section | Original Text | Correction |
|---------|--------------|------------|
| Section 9 (DSA) | "KL-divergence alignment with dense attention" | Specify exact formula: `D_KL(p_{t,:} \|\| Softmax(I_{t,:}))` where p is L1-normalized attention, summed across heads |
| Section 9 (DSA) | Expand KV then gather | **Gather compressed kv_c first, THEN expand via wkv_b** (MQA mode — critical for efficiency and correctness) |
| Section 9 (DSA phases) | Warm-up mentioned vaguely | Stage 1: freeze main model, train indexer at LR=1e-3. Stage 2: detach indexer from main graph |
| config.py (MoEConfig) | `gamma_freeze_ratio = 0.80` | Change to `0.95` |
| pre-train.py | Phase transition | Add explicit indexer warm-up stage (1000 steps) before Phase 2 |
| tests/ | No acceptance rate test | Add `test_mtp_acceptance_rate.py` targeting >75% on heldout tokens |
| SCALING_LAB_PLAN.md | 3 pillars | Add Pillar 4 sketch (RL scaling), add token efficiency metric to Pillar 1 |

---

## What We Did NOT Need to Change

These aspects of our plan are confirmed correct by the papers:

| Component | Our Plan | Paper Confirmation |
|-----------|----------|-------------------|
| MTP architecture (standard transformer block, NOT cross-attention) | ✅ correct | V3 Section 2.2, Eq. 21-25 |
| Shared embeddings + LM head for MTP | ✅ correct | V3 Section 2.2 |
| Gate scoring: sigmoid not softmax | ✅ correct | V3 Eq. 15 |
| Aux-loss-free with γ=0.001 as default | ✅ correct | V3 Section 2.1.2 |
| β₂=0.95, grad_clip=1.0 | ✅ correct | V3 Table 1 (hyperparameters) |
| KV cache stores (c_kv, k_rope) not expanded KV | ✅ correct | V3 Eq. 1-5 |
| MTP λ schedule: 0.3 → 0.1 at 60% of training | ✅ correct | V3: 0.3 for 10T, 0.1 for 4.8T = 67% transition |
| RoPE applied only to decoupled portion of KV | ✅ correct | V3 Eq. 3-4 |
| RMSNorm at MLA bottlenecks | ✅ correct | V3 Section 2.1.1 footnote |
| AdamW with weight_decay=0.1 | ✅ correct | V3 Table 1 |
| MoE group-based routing (n_expert_groups) | ✅ correct | V3 node-limited routing analog |
| Lightning Indexer: ReLU scoring | ✅ correct | V3.2 Eq. 1, "ReLU activation chosen for throughput" |
| topk_tokens = 2048 for DSA | ✅ correct | V3.2 Section 2.1.1 Stage 2: "k=2048 KV tokens" |
