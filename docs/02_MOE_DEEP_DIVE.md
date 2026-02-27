# Mixture of Experts (MoE) with Auxiliary-Loss-Free Load Balancing — First Principles Deep Dive

> Written from the perspective of a senior researcher who has trained MoE models at scale, for a peer who needs to understand exactly what happens to every tensor, every gradient, and every bias update. No hand-waving. All shapes verified against the NanoSeek codebase (`model/model.py`, `model/config.py`).

---

## Quick Context

**What this document covers:** The complete MoE subsystem in NanoSeek / DeepSeek V3.2 — from the moment a hidden state enters the Gate to the moment a weighted-combined expert output exits. Every shape, every design choice, every failure mode.

**What MoE gives you:** A model with ~4.75B total parameters but only ~1.08B active per forward pass. You get the representational capacity of a large model with the compute cost of a small one. In NanoSeek: 64 routed experts + 2 shared experts, 8 active per token, yielding ~4.4× expansion.

**Why this matters in 2026:** MoE is no longer exotic. DeepSeek V3 (671B total, 37B active), Mixtral, Grok, DBRX, Qwen2-MoE, and Arctic all ship MoE. The hard part is not building the router — it is keeping the experts alive during training. This document focuses on how DeepSeek V3 solved that problem and how NanoSeek implements that solution.

**Key innovation:** Auxiliary-loss-free load balancing via dynamic bias adjustment. Prior approaches (GShard, Switch Transformer, ST-MoE) all used auxiliary losses to encourage balanced routing. These losses interfere with the primary language modeling objective and require careful tuning. DeepSeek V3 eliminated this by using a simple bias term that is updated outside the gradient computation, achieving better balance with less interference. NanoSeek implements this exactly.

**Reading this document:** Follow the five stages linearly. Each builds on the previous. If you skip Stage 2, nothing after it will make sense.

---

## 🔴 STAGE 1: INPUTS — Hidden States from Attention

### What enters the MoE layer

Every decoder layer in NanoSeek follows the same pattern:

```
hidden_states → RMSNorm → Attention → Residual → RMSNorm → FFN/MoE → Residual
```

The MoE layer receives the post-attention, post-norm hidden states:

```
Input: x ∈ ℝ^[B, L, 2048]

where:
  B = batch size (e.g., 2)
  L = sequence length (e.g., 4096)
  2048 = hidden_size (d_model)
```

**Concretely**, if `B=2, L=128`:

```
x.shape = [2, 128, 2048]
```

### Which layers use MoE vs dense FFN

Not every layer is MoE. The first `first_k_dense_replace = 2` layers use a standard dense SwiGLU FFN. The remaining 14 layers (indices 2–15) use MoE.

```python
# From config.py
dense_layers = [0, 1]           # Dense SwiGLU FFN, inter_dim=5243
moe_layers   = [2, 3, ..., 15]  # MoE with 64 routed + 2 shared experts
```

**Why dense first layers?** The earliest layers primarily learn low-level token representations (subword features, positional patterns). These representations are universal — every token benefits from the same transformations. Specializing via experts this early wastes capacity. DeepSeek, Mixtral, and Qwen all use dense early layers.

### The flatten operation

MoE processes tokens independently (no sequence-level interaction within the FFN). The first thing MoE does is flatten the batch and sequence dimensions:

```python
# In MoE.forward() — model.py line 677
batch_size, seq_len, dim = x.shape    # [2, 128, 2048]
N = batch_size * seq_len              # 256
x_flat = x.view(N, dim)              # [256, 2048]
```

From this point forward, we work with `N` independent tokens. The batch/sequence structure is irrelevant until we reshape at the end.

---

## 🔴 STAGE 2: MODEL ARCHITECTURE

This is the heart of the document. We will trace a single forward pass through:

1. The Gate (router) — sigmoid scoring, bias adjustment, group routing, top-k selection
2. Token-centric dispatch — the permute-compute-unpermute algorithm
3. Expert computation — SwiGLU FFN
4. Shared experts — always-active computation
5. Output combination

### 2.1 The Gate: Routing Tokens to Experts

The Gate is defined in `model.py` (class `Gate`, line 379). It decides which experts process each token.

#### 2.1.1 Gate parameters

```python
Gate(
    dim=2048,                  # Input hidden dimension
    n_routed_experts=64,       # Total routed experts (E)
    n_activated_experts=8,     # Experts per token (K)
    n_expert_groups=8,         # Number of groups (G)
    n_limited_groups=4,        # Groups selected per token
    score_func="sigmoid",      # NOT softmax
    route_scale=2.5,           # Output scaling factor
)
```

**Learnable parameters:**
```
weight: [64, 2048]  — The routing weight matrix
expert_bias: [64]   — Non-gradient bias for load balancing (buffer, not parameter)
expert_load: [64]   — Running load count (buffer)
```

The `weight` matrix has 64 × 2048 = 131,072 parameters per MoE layer. There are 14 MoE layers, so routing adds ~1.8M parameters total. Tiny compared to the experts themselves.

#### 2.1.2 Step 1: Compute raw scores (sigmoid, NOT softmax)

```python
# model.py line 410-415
scores = F.linear(x, self.weight)  # [N, 64]  — raw logits
scores = torch.sigmoid(scores)     # [N, 64]  — probabilities in (0, 1)
```

**Concrete example** with `N=256`:
```
x_flat:   [256, 2048]
weight:   [64, 2048]
scores:   [256, 64]    — each token gets a score for each expert

Token 42, expert 7:  scores[42, 7] = σ(x[42] · w[7]) = 0.73
Token 42, expert 55: scores[42, 55] = σ(x[42] · w[55]) = 0.02
```

#### 2.1.3 Why sigmoid instead of softmax

This is a critical design choice that DeepSeek V3 introduced, breaking from the softmax tradition (GShard, Switch, Mixtral, ST-MoE).

**Softmax problem:** Scores are a normalized distribution that sums to 1. If expert A's score goes up, expert B's must go down — even if B is independently relevant. This creates a **competitive coupling** between experts that is harmful in two ways:

1. **During training:** Gradients for expert A's routing weight are entangled with expert B's score. The router cannot independently increase affinity for two experts.

2. **During inference with variable K:** Softmax probabilities are calibrated for a specific K. Changing K at inference time (e.g., reducing from 8 to 4 for speed) distorts the relative weights.

**Sigmoid advantage:** Each expert's score is computed independently. `σ(x · w_i)` depends only on expert `i`'s weight vector. Scores do not sum to 1. This means:

- Multiple experts can independently have high affinity for the same token.
- Scores are **absolute** measures of relevance, not relative.
- Gradient flow to each expert's routing weight is decoupled from other experts.

**The catch:** Sigmoid scores don't normalize to 1, so you need the `route_scale` factor to calibrate the output magnitude. NanoSeek uses `route_scale = 2.5`.

**Historical context:** Softmax routing was the standard from GShard (2020) through Mixtral (2023). DeepSeek V3 (December 2024) was the first major model to successfully switch to sigmoid at scale. By 2026, sigmoid routing is the default in new MoE architectures.

#### 2.1.4 Step 2: Add bias for selection (training only)

```python
# model.py line 417-420
if self.training:
    scores_for_selection = scores + self.expert_bias.unsqueeze(0)  # [N, 64]
else:
    scores_for_selection = scores  # At inference: no bias
```

**This is the auxiliary-loss-free mechanism at work.** The bias does NOT affect the actual routing weights used for combining expert outputs — only the selection of which experts are chosen. During training:

- `scores` — used to compute the final routing weights (gradient flows through these)
- `scores_for_selection` — used only for top-k selection (bias included)

The bias gently nudges underloaded experts into the top-k selection. An expert with a slightly lower score might get selected because its bias is positive (it has been underloaded). An overloaded expert might be excluded because its bias is negative.

**Key insight:** The bias operates on **selection**, not on **weighting**. Once an expert is selected, its actual contribution weight comes from the original unbiased score. This means:

- The language modeling gradient is uncontaminated by the load balancing mechanism.
- The router still learns genuine token-expert affinity.
- The bias is just a tie-breaker that favors underloaded experts.

#### 2.1.5 Step 3: Group-based routing

With 64 experts, naively selecting top-8 could concentrate all selections in a few "popular" groups, leaving entire groups unused. Group-based routing prevents this.

```python
# model.py line 422-430
# Configuration: n_expert_groups=8, n_limited_groups=4, experts_per_group=8

# Step 3a: Reshape scores into groups
scores_grouped = scores_for_selection.view(-1, 8, 8)  # [N, 8 groups, 8 experts/group]

# Step 3b: Score each group by its best expert
group_scores = scores_grouped.max(dim=-1).values  # [N, 8]

# Step 3c: Select top-4 groups
_, top_groups = group_scores.topk(4, dim=-1)  # [N, 4]

# Step 3d: Mask experts in non-selected groups
group_mask = torch.zeros_like(group_scores, dtype=torch.bool)  # [N, 8]
group_mask.scatter_(1, top_groups, True)                        # [N, 8]
group_mask = group_mask.unsqueeze(-1).expand(-1, -1, 8)         # [N, 8, 8]
group_mask = group_mask.reshape(-1, 64)                         # [N, 64]

scores_for_selection = scores_for_selection.masked_fill(~group_mask, float('-inf'))
```

**Concrete walkthrough for token 42:**

```
Expert groups (8 experts each):
  Group 0: experts [0,  1,  2,  3,  4,  5,  6,  7]
  Group 1: experts [8,  9,  10, 11, 12, 13, 14, 15]
  Group 2: experts [16, 17, 18, 19, 20, 21, 22, 23]
  Group 3: experts [24, 25, 26, 27, 28, 29, 30, 31]
  Group 4: experts [32, 33, 34, 35, 36, 37, 38, 39]
  Group 5: experts [40, 41, 42, 43, 44, 45, 46, 47]
  Group 6: experts [48, 49, 50, 51, 52, 53, 54, 55]
  Group 7: experts [56, 57, 58, 59, 60, 61, 62, 63]

Token 42's group scores (max expert score per group):
  Group 0: 0.73   Group 1: 0.45   Group 2: 0.61   Group 3: 0.82
  Group 4: 0.38   Group 5: 0.55   Group 6: 0.69   Group 7: 0.41

Top-4 groups selected: [3, 0, 6, 2]  (scores: 0.82, 0.73, 0.69, 0.61)

Experts available for top-8 selection:
  From Group 3: [24, 25, 26, 27, 28, 29, 30, 31]
  From Group 0: [0,  1,  2,  3,  4,  5,  6,  7]
  From Group 6: [48, 49, 50, 51, 52, 53, 54, 55]
  From Group 2: [16, 17, 18, 19, 20, 21, 22, 23]

  Groups 1, 4, 5, 7: all scores set to -inf (masked out)
```

**Why group routing?** Two reasons:

1. **Communication overhead (production):** In distributed settings with expert parallelism, each group maps to a node. Selecting 4 of 8 groups means you only communicate with 4 of 8 nodes, cutting all-to-all communication by 50%.

2. **Expert diversity:** Without group routing, the top-8 might all come from 1–2 groups. Group routing guarantees diversity — experts are selected from at least 4 different groups, encouraging broader specialization.

#### 2.1.6 Step 4: Top-K expert selection

```python
# model.py line 432-434
topk_weights, topk_indices = scores_for_selection.topk(8, dim=-1)
# topk_weights: [N, 8]  — scores including bias (used for selection only)
# topk_indices: [N, 8]  — expert IDs
```

**Continuing token 42's example:**

```
After group masking, scores_for_selection[42]:
  Expert 3:  0.72    Expert 7:  0.68    Expert 16: 0.61    Expert 22: 0.58
  Expert 24: 0.82    Expert 31: 0.71    Expert 48: 0.69    Expert 55: 0.64
  (all others: -inf because their groups were masked out)

Top-8 selection (sorted by biased score):
  topk_indices[42] = [24, 3, 31, 48, 7, 55, 16, 22]
  topk_weights[42] = [0.82, 0.72, 0.71, 0.69, 0.68, 0.64, 0.61, 0.58]
```

#### 2.1.7 Step 5: Extract actual (unbiased) weights

```python
# model.py line 433-434
weights = scores.gather(dim=-1, index=topk_indices)  # [N, 8]
weights = weights * self.route_scale                  # [N, 8] — multiply by 2.5
```

**This is crucial.** The `weights` used for combining expert outputs come from the **original unbiased scores**, not from `scores_for_selection`. The bias only affected which experts were selected.

```
Token 42's actual routing weights (from original sigmoid scores × 2.5):
  Expert 24: 0.80 × 2.5 = 2.00    Expert 3:  0.71 × 2.5 = 1.775
  Expert 31: 0.68 × 2.5 = 1.70    Expert 48: 0.67 × 2.5 = 1.675
  Expert 7:  0.66 × 2.5 = 1.65    Expert 55: 0.62 × 2.5 = 1.55
  Expert 16: 0.59 × 2.5 = 1.475   Expert 22: 0.56 × 2.5 = 1.40

  Note: These do NOT sum to 1. That's intentional with sigmoid routing.
  Total weight = 13.225 — the route_scale of 2.5 calibrates the magnitude.
```

#### 2.1.8 Step 6: Track expert load (training only)

```python
# model.py line 437-441
if self.training:
    with torch.no_grad():
        load = torch.zeros(64, device=x.device)
        load.scatter_add_(0, topk_indices.flatten(),
                         torch.ones_like(topk_indices.flatten(), dtype=load.dtype))
        self.expert_load.copy_(load)
```

This counts how many tokens were routed to each expert in this forward pass. For `N=256` tokens, each selecting `K=8` experts:

```
Total assignments: 256 × 8 = 2048
Ideal per expert: 2048 / 64 = 32

Actual load (example):
  Expert 0: 28   Expert 1: 35   Expert 2: 31   Expert 3: 45  ← overloaded
  Expert 4: 22   Expert 5: 33   Expert 6: 29   Expert 7: 38
  ...
  Expert 63: 26
```

This load information drives the bias update in Stage 4.

### 2.2 Token-Centric Dispatch: The Performance Innovation

The most important engineering contribution in the MoE implementation is the token-centric dispatch algorithm. This is what makes MoE actually fast in practice.

#### 2.2.1 The naive approach and why it fails

The naive implementation loops over experts:

```python
# NAIVE: O(K × E × N) — DO NOT DO THIS
output = torch.zeros(N, D)
for expert_id in range(64):
    mask = (indices == expert_id).any(dim=-1)  # O(N) scan per expert
    if mask.any():
        expert_input = x[mask]                  # gather
        expert_output = experts[expert_id](expert_input)
        output[mask] += expert_output * weights[mask]  # scatter
```

This has three problems:
1. **O(E × N) masking** — scanning all N tokens for each of 64 experts
2. **Scattered memory access** — boolean indexing creates non-contiguous tensors
3. **Python loop overhead** — 64 sequential Python iterations

#### 2.2.2 Token-centric dispatch: O(N×K)

NanoSeek's `token_centric_dispatch()` (model.py line 485) eliminates all three problems. Here is the algorithm decomposed into three phases:

```
Phase 1: PERMUTE   — Sort tokens by expert assignment
Phase 2: COMPUTE   — Process each expert's contiguous batch
Phase 3: UNPERMUTE — Scatter results back to original positions
```

Let me trace this with a **concrete small example**: `N=4` tokens, `K=2` experts per token, `E=3` total experts.

```
Tokens:    [t0, t1, t2, t3]
Indices:   [[1, 2],    — token 0 → experts 1, 2
            [0, 1],    — token 1 → experts 0, 1
            [2, 0],    — token 2 → experts 2, 0
            [1, 0]]    — token 3 → experts 1, 0

Weights:   [[0.6, 0.4],
            [0.7, 0.3],
            [0.5, 0.5],
            [0.8, 0.2]]
```

#### Phase 1: PERMUTE

```python
# Flatten assignments: each token appears K times
flat_indices = indices.view(-1)     # [1, 2, 0, 1, 2, 0, 1, 0]
flat_weights = weights.view(-1)     # [0.6, 0.4, 0.7, 0.3, 0.5, 0.5, 0.8, 0.2]

# Token IDs: which original token each assignment belongs to
token_ids = [0, 0, 1, 1, 2, 2, 3, 3]

# Sort by expert ID
sorted_order = argsort(flat_indices)  # [2, 5, 7, 0, 3, 6, 1, 4]
```

After sorting:

```
Position:         0    1    2    3    4    5    6    7
sorted_expert_id: 0    0    0    1    1    1    2    2
sorted_token_id:  1    2    3    0    1    3    0    2
sorted_weight:    0.7  0.5  0.2  0.6  0.3  0.8  0.4  0.5

expert_counts = [3, 3, 2]  — Expert 0 gets 3 tokens, Expert 1 gets 3, Expert 2 gets 2
```

```
ASCII visualization of the permutation:

BEFORE (token-major order):        AFTER (expert-major order):
  t0→E1, t0→E2                       t1→E0, t2→E0, t3→E0    ← Expert 0's batch
  t1→E0, t1→E1                       t0→E1, t1→E1, t3→E1    ← Expert 1's batch
  t2→E2, t2→E0                       t0→E2, t2→E2            ← Expert 2's batch
  t3→E1, t3→E0

Gather input tokens in sorted order:
  permuted_input = x[sorted_token_ids] = [x[1], x[2], x[3], x[0], x[1], x[3], x[0], x[2]]
```

#### Phase 2: COMPUTE

```python
# Split into per-expert batches
expert_batches = torch.split(permuted_input, [3, 3, 2])

# Process each expert's CONTIGUOUS batch
expert_outputs = [
    experts[0](expert_batches[0]),  # [3, D] — tokens 1, 2, 3
    experts[1](expert_batches[1]),  # [3, D] — tokens 0, 1, 3
    experts[2](expert_batches[2]),  # [2, D] — tokens 0, 2
]

# Concatenate
permuted_output = torch.cat(expert_outputs)  # [8, D]
```

**Why this is fast:** Each expert processes a contiguous memory block. No boolean masking, no scattered reads. The GPU kernel gets a single dense matrix multiply per expert.

#### Phase 3: UNPERMUTE

```python
# Apply routing weights
weighted_output = permuted_output * sorted_weights.unsqueeze(-1)  # [8, D]

# Scatter-add back to original positions
output = torch.zeros(N, D)
output.scatter_add_(0, sorted_token_ids.unsqueeze(-1).expand(-1, D), weighted_output)
```

```
scatter_add accumulates:
  output[0] += weighted_output[3] + weighted_output[6]   (E1 + E2 contributions for t0)
  output[1] += weighted_output[0] + weighted_output[4]   (E0 + E1 contributions for t1)
  output[2] += weighted_output[1] + weighted_output[7]   (E0 + E2 contributions for t2)
  output[3] += weighted_output[2] + weighted_output[5]   (E0 + E1 contributions for t3)
```

#### 2.2.3 Complexity analysis

```
Naive:          O(K × E × N) operations     — scan all N tokens for each of E experts, K times
Token-centric:  O(N × K × log(N×K)) sort    — dominated by the argsort
                + O(N × K) gather/scatter    — contiguous memory access

For NanoSeek:  N=524,288 (batch=128, seq=4096), K=8, E=64
  Naive:    8 × 64 × 524,288 = 268B operations
  Centric:  524,288 × 8 × log(4,194,304) ≈ 92M operations (sort) + 4.2M (scatter)

  Speedup: ~2900× in operations, ~10-50× in wall time
  (Wall time gap is smaller due to GPU parallelism in the naive approach)
```

The full ASCII diagram of the three-phase pipeline at NanoSeek scale:

```
                    TOKEN-CENTRIC DISPATCH PIPELINE
                    ================================

 Input: x [N=256, D=2048]     indices [N=256, K=8]     weights [N=256, K=8]
   │                              │                        │
   ▼                              ▼                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE 1: PERMUTE                                                    │
│                                                                      │
│  flat_indices = indices.view(-1)                    [N×K = 2048]     │
│  flat_weights = weights.view(-1)                    [2048]           │
│  token_ids = [0,0,...,0, 1,1,...,1, ..., 255,...,255] [2048]         │
│                    K times    K times                                 │
│                                                                      │
│  sorted_order = argsort(flat_indices, stable=True)  [2048]           │
│  permuted_input = x[sorted_token_ids]               [2048, 2048]    │
│  expert_counts = bincount(sorted_expert_ids)        [64]            │
│                                                                      │
│  Result: Tokens grouped by expert in contiguous blocks               │
│  ┌─────────┬─────────┬─────────┬───┬──────────┐                     │
│  │Expert 0 │Expert 1 │Expert 2 │...│Expert 63 │                     │
│  │ ~32 tok │ ~32 tok │ ~32 tok │   │ ~32 tok  │                     │
│  └─────────┴─────────┴─────────┴───┴──────────┘                     │
└──────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE 2: COMPUTE                                                    │
│                                                                      │
│  for each expert_id, batch in enumerate(split(permuted_input)):      │
│      if batch.shape[0] > 0:                                          │
│          output[expert_id] = experts[expert_id](batch)               │
│                                                                      │
│  Each expert: SwiGLU FFN                                             │
│    batch [~32, 2048] → gate_proj [~32, 768] → SiLU                  │
│                       → up_proj   [~32, 768]                         │
│                       → element-wise multiply [~32, 768]             │
│                       → down_proj [~32, 2048]                        │
│                                                                      │
│  permuted_output = cat(all expert outputs)          [2048, 2048]    │
└──────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE 3: UNPERMUTE                                                  │
│                                                                      │
│  weighted_output = permuted_output × sorted_weights  [2048, 2048]   │
│                                                                      │
│  output = zeros(N, D)                                [256, 2048]    │
│  output.scatter_add_(0,                                              │
│      sorted_token_ids.expand(-1, D),                                 │
│      weighted_output)                                                │
│                                                                      │
│  Each token accumulates weighted contributions from its 8 experts    │
│  output[i] = Σ_{k=1}^{8} weight[i,k] × expert[k](x[i])            │
└──────────────────────────────────────────────────────────────────────┘
   │
   ▼
 Output: routed_output [N=256, D=2048]
```

### 2.3 Expert Architecture: SwiGLU FFN

Each of the 64 routed experts (and 2 shared experts) is a SwiGLU FFN:

```python
class SwiGLUFFN(nn.Module):
    def __init__(self, dim=2048, inter_dim=768):
        self.gate_proj = nn.Linear(2048, 768, bias=False)  # [2048, 768]
        self.up_proj   = nn.Linear(2048, 768, bias=False)  # [2048, 768]
        self.down_proj = nn.Linear(768, 2048, bias=False)  # [768, 2048]

    def forward(self, x):  # x: [batch, 2048]
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

**Per-expert parameter count:**
```
gate_proj: 2048 × 768 = 1,572,864
up_proj:   2048 × 768 = 1,572,864
down_proj: 768 × 2048 = 1,572,864
Total:     4,718,592 (~4.7M) per expert
```

**Total MoE parameters per layer:**
```
64 routed experts: 64 × 4.7M = 301M
2 shared experts:  2 × 4.7M  = 9.4M
Router:            64 × 2048  = 131K
Total per layer:   ~311M
14 MoE layers:     ~4.35B
```

**Active parameters per layer per token:**
```
8 routed experts:  8 × 4.7M  = 37.7M
2 shared experts:  2 × 4.7M  = 9.4M
Router:            131K
Total active:      ~47.2M
```

#### Why SwiGLU and not ReLU/GELU?

SwiGLU (SiLU gating × up projection) consistently outperforms ReLU and GELU FFNs at the same parameter count. The key insight: the gating mechanism (SiLU applied to `gate_proj` output) creates a **multiplicative interaction** that the network can use to selectively amplify or suppress features. This is especially valuable in MoE where each expert needs to specialize efficiently.

The `inter_dim=768` is notably small (768/2048 = 0.375× the hidden size). Compare to dense FFN's `inter_dim=5243` (2.56× hidden). This fine-grained expert design is intentional — many small experts with high combinatorial diversity outperform fewer large experts (DeepSeek V2 finding).

#### Why fine-grained experts?

DeepSeek V2 introduced the fine-grained expert paradigm: instead of 8 large experts (like Mixtral), use 64 small experts with 8 active. The math:

```
Mixtral approach:  8 experts, 2 active = C(8,2) = 28 possible combinations
NanoSeek approach: 64 experts, 8 active = C(64,8) = 4,426,165,368 possible combinations

4.4 billion vs 28. The combinatorial expressiveness is astronomically higher.
```

Each token can potentially activate a unique combination of experts, enabling extreme specialization without expert duplication.

### 2.4 Shared Experts: Always-Active Computation

```python
# model.py lines 643-648
if n_shared_experts > 0:
    self.shared_experts = nn.ModuleList([
        Expert(dim, moe_inter_dim) for _ in range(n_shared_experts)
    ])
```

NanoSeek has `n_shared_experts = 2`. These experts process **every token** regardless of routing decisions.

**Why shared experts?**

1. **Common patterns:** Some transformations are universally useful (e.g., attention-to-FFN feature alignment). Making every routed expert learn these wastes capacity.

2. **Routing failure safety net:** If a token gets routed to suboptimal experts (routing is stochastic early in training), the shared experts still provide reasonable processing.

3. **Gradient stability:** Shared experts receive gradients from every token, making them well-trained quickly. They provide a stable baseline that the routed experts can specialize relative to.

**Shared/active ratio:** 2/(2+8) = 20% of active compute goes through shared experts. This matches DeepSeek V3's ratio.

### 2.5 Combining Routed and Shared Outputs

```python
# model.py lines 684-701
shared_output = self._compute_shared_output(x_flat)  # [N, 2048]
weights, indices = self.gate(x_flat)                  # [N, 8], [N, 8]
routed_output = token_centric_dispatch(x_flat, indices, weights, self.experts)  # [N, 2048]

output = routed_output + shared_output                # [N, 2048]
output = output.view(batch_size, seq_len, dim)        # [B, L, 2048]
```

The shared and routed outputs are simply summed. No gating, no learned combination weight. This is intentional simplicity — the router already controls the routed contribution via weights, and adding a learnable combination factor would be redundant.

### 2.6 Full Data Flow with Shapes (Complete Walkthrough)

Let's trace the complete forward pass with `B=2, L=128`:

```
Step 1: Input
  x: [2, 128, 2048]                         — from attention + residual + norm

Step 2: Flatten
  x_flat: [256, 2048]                       — N = 2 × 128 = 256 tokens

Step 3: Shared experts (parallel with routing)
  shared_expert_0(x_flat): [256, 2048]
  shared_expert_1(x_flat): [256, 2048]
  shared_output: [256, 2048]                — sum of both shared experts

Step 4: Gate/Router
  4a. Linear:           [256, 2048] × [2048, 64]ᵀ = [256, 64]    — raw logits
  4b. Sigmoid:          [256, 64]                                  — scores ∈ (0,1)
  4c. Add bias:         [256, 64] + [1, 64] = [256, 64]          — selection scores
  4d. Group routing:    [256, 8, 8] → top-4 groups → mask
  4e. Top-8 selection:  indices [256, 8], weights [256, 8]
  4f. Gather weights:   scores.gather(indices) × 2.5 → [256, 8]
  4g. Track load:       bincount(indices) → [64]

Step 5: Token-centric dispatch
  5a. Flatten indices:  [256×8] = [2048]
  5b. Sort by expert:   argsort → [2048]
  5c. Permute input:    x[sorted_tokens] → [2048, 2048]
  5d. Split by expert:  64 batches, ~32 tokens each
  5e. Process experts:  each [~32, 2048] → SwiGLU → [~32, 2048]
  5f. Concatenate:      [2048, 2048]
  5g. Apply weights:    × sorted_weights → [2048, 2048]
  5h. Scatter-add:      → [256, 2048]

Step 6: Combine
  output = routed_output + shared_output    — [256, 2048]

Step 7: Reshape
  output: [2, 128, 2048]                   — back to [B, L, D]

Step 8: Residual (in decoder layer, not in MoE)
  final = residual + output                — [2, 128, 2048]
```

---

## 🔴 STAGE 3: GROUND TRUTH — What "Correct" Looks Like

### 3.1 Balanced routing

In a perfectly balanced system with `N=256, K=8, E=64`:

```
Total assignments: 256 × 8 = 2048
Expected per expert: 2048 / 64 = 32
Acceptable range: ±20% → [25.6, 38.4]
```

**Metrics to monitor:**
- **Load variance:** `var(expert_load) / mean(expert_load)²` — should be < 0.05
- **Max/min ratio:** `max(load) / min(load)` — should be < 2.0
- **Dead experts:** count of experts with load < 5% of mean — should be 0

### 3.2 Expert specialization

After sufficient training, experts should develop distinct specializations. Signs of healthy specialization:

```
Expert 3:  High activation for code tokens (brackets, keywords)
Expert 17: High activation for mathematical notation
Expert 42: High activation for conversational patterns
Expert 55: High activation for named entities
```

You can verify this by examining `scores` (pre-selection) for tokens from different domains. Healthy experts show bimodal score distributions — high for their specialty, low for everything else.

### 3.3 Routing weight distribution

With sigmoid scoring and `route_scale=2.5`:

```
Typical routing weights per token (after scaling):
  Highest expert:  ~2.0-2.5  (sigmoid ~0.8-1.0 × 2.5)
  Lowest expert:   ~0.5-1.0  (sigmoid ~0.2-0.4 × 2.5)
  Sum of 8 weights: ~8-15    (varies; no normalization constraint)
```

The sum varies because sigmoid scores are independent. This is fine — the residual connection and layer norm handle magnitude calibration.

### 3.4 Concrete token-routing example (full walkthrough)

Let's trace token 42 through the complete routing pipeline with realistic numbers:

```
Token 42: hidden state x[42] ∈ ℝ^2048  (from attention output)

Step 1: Raw logits = x[42] · W_gate^T  →  [64] values
  logit[3]=1.87  logit[7]=1.54  logit[12]=0.92  logit[15]=1.31
  logit[22]=1.12 logit[24]=2.34 logit[31]=1.65  logit[48]=1.49
  logit[55]=1.21 ...  (remaining ~0.0 to 0.8)

Step 2: Sigmoid scores
  score[3]=0.866  score[7]=0.824  score[12]=0.715  score[15]=0.787
  score[22]=0.754 score[24]=0.912 score[31]=0.839  score[48]=0.816
  score[55]=0.770 ...  (remaining ~0.5 to 0.69)

Step 3: Add bias (training only)
  bias[3]=+0.002  bias[7]=-0.001  bias[12]=+0.005  bias[15]=+0.003
  bias[24]=-0.003 bias[31]=+0.001 bias[48]=+0.002  bias[55]=-0.002

  selection_score[3]=0.868  selection_score[24]=0.909  ...

Step 4: Group routing
  Group 0 (experts 0-7):   max = score[3]+bias = 0.868
  Group 1 (experts 8-15):  max = score[15]+bias = 0.790
  Group 2 (experts 16-23): max = score[22]+bias = 0.754
  Group 3 (experts 24-31): max = score[24]+bias = 0.909
  Group 4 (experts 32-39): max = 0.621
  Group 5 (experts 40-47): max = 0.583
  Group 6 (experts 48-55): max = score[48]+bias = 0.818
  Group 7 (experts 56-63): max = 0.601

  Top-4 groups: [3, 0, 6, 1]  →  available experts: [0-7, 8-15, 24-31, 48-55]

Step 5: Top-8 from available experts (by selection score)
  Selected: [24, 3, 31, 48, 7, 55, 15, 12]

Step 6: Final weights (from original scores × 2.5)
  w[24]=0.912×2.5=2.280   w[3]=0.866×2.5=2.165   w[31]=0.839×2.5=2.098
  w[48]=0.816×2.5=2.040   w[7]=0.824×2.5=2.060   w[55]=0.770×2.5=1.925
  w[15]=0.787×2.5=1.968   w[12]=0.715×2.5=1.788

Step 7: Expert computation
  Each selected expert processes x[42] through its SwiGLU:
    e_24 = Expert24(x[42])  →  [2048]
    e_3  = Expert3(x[42])   →  [2048]
    ...

Step 8: Weighted combination
  routed_out[42] = 2.280·e_24 + 2.165·e_3 + 2.098·e_31 + 2.040·e_48
                 + 2.060·e_7  + 1.925·e_55 + 1.968·e_15 + 1.788·e_12

Step 9: Add shared experts
  output[42] = routed_out[42] + shared_0(x[42]) + shared_1(x[42])
```

---

## 🔴 STAGE 4: LOSS FUNCTION — The Training Objective

### 4.1 Total loss composition

The total loss in NanoSeek is:

```
L_total = L_main + λ_mtp × L_mtp + L_seq_aux + L_bias_update (non-gradient)
```

For the MoE component specifically:

```
MoE losses:
  1. L_seq_aux = α × mean((load_i - target_load)²)    — tiny auxiliary loss
  2. Bias update: bias_i -= γ × (load_i - mean_load) / mean_load  — NOT a loss
```

### 4.2 Main cross-entropy loss (unchanged by MoE)

```python
# model.py line 1701-1705
shift_logits = logits[:, :-1, :].contiguous()    # [B, L-1, V]
shift_labels = labels[:, 1:].contiguous()        # [B, L-1]
main_loss = F.cross_entropy(
    shift_logits.view(-1, V), shift_labels.view(-1), ignore_index=-100
)
```

MoE does not change the main loss computation. It only changes the hidden states that produce the logits. This is important — the language modeling objective is pure.

### 4.3 Sequence-level auxiliary loss (tiny complement)

```python
# model.py line 712-717
if self.seq_aux_loss_alpha > 0:  # alpha = 0.0001
    load = self.gate.expert_load
    target_load = N * self.n_activated_experts / self.n_routed_experts
    # target_load = 256 * 8 / 64 = 32 (ideal per expert)
    load_imbalance = ((load - target_load) ** 2).mean()
    aux_data["seq_aux_loss"] = 0.0001 * load_imbalance
```

**This loss is intentionally tiny.** With `α = 0.0001`, even a 10× imbalance contributes:

```
load_imbalance = mean((load_i - 32)²)
If worst case: some experts at 64 (2×), some at 0:
  imbalance ≈ 1024
  aux_loss = 0.0001 × 1024 = 0.1024

Compare to main_loss ≈ 5.0 early in training
Ratio: 0.1024 / 5.0 = 2.0%  (worst case)
Typical ratio: < 0.1%
```

The sequence-level auxiliary loss serves as a soft safety net — it creates a gentle gradient signal toward balance, but it is too small to meaningfully interfere with language modeling. The heavy lifting is done by the bias update mechanism.

### 4.4 Auxiliary-Loss-Free Bias Update (THE key innovation)

This is what DeepSeek V3 is known for. Instead of adding an auxiliary loss to the gradient computation, they update a bias term **outside** the autograd graph.

```python
# model.py line 721-727 — MoE.update_load_balance_bias()
def update_load_balance_bias(self, gamma=0.001):
    with torch.no_grad():  # ← NOT part of gradient computation
        load = self.gate.expert_load          # [64] — from last forward pass
        mean_load = load.mean()               # scalar
        if mean_load > 0:
            imbalance = (load - mean_load) / (mean_load + 1e-8)  # [64]
            self.gate.expert_bias.sub_(gamma * imbalance)         # [64]
```

Let's trace this with concrete numbers:

```
After a forward pass with N=256, K=8:

expert_load:  [28, 35, 31, 45, 22, 33, 29, 38, ...]
                               ↑ overloaded    ↑ underloaded

mean_load = 32.0

imbalance = (load - 32) / 32:
  Expert 0:  (28-32)/32 = -0.125   (underloaded)
  Expert 1:  (35-32)/32 = +0.094   (overloaded)
  Expert 2:  (31-32)/32 = -0.031
  Expert 3:  (45-32)/32 = +0.406   (significantly overloaded)
  Expert 4:  (22-32)/32 = -0.313   (significantly underloaded)
  Expert 5:  (33-32)/32 = +0.031
  ...

Bias update (gamma=0.001):
  bias[0] -= 0.001 × (-0.125) → bias[0] += 0.000125  (pushed UP → more likely selected)
  bias[1] -= 0.001 × (+0.094) → bias[1] -= 0.000094  (pushed DOWN → less likely selected)
  bias[3] -= 0.001 × (+0.406) → bias[3] -= 0.000406  (pushed DOWN significantly)
  bias[4] -= 0.001 × (-0.313) → bias[4] += 0.000313  (pushed UP significantly)
```

**After thousands of steps, the biases accumulate:**

```
Step 0:     bias = [0, 0, 0, 0, ...]
Step 1000:  bias = [+0.12, -0.08, +0.03, -0.15, +0.21, ...]
Step 5000:  bias = [+0.05, -0.02, +0.01, -0.03, +0.04, ...]  (converging)
```

The biases converge because they form a **negative feedback loop**:
1. Expert overloaded → bias decreases → fewer tokens selected → load decreases
2. Expert underloaded → bias increases → more tokens selected → load increases

This is a simple control system (integral controller) that converges to balanced load without touching the gradient computation.

### 4.5 Why the bias approach is superior to auxiliary losses

| Aspect | Auxiliary Loss (GShard, Switch) | Bias Update (DeepSeek V3) |
|--------|-------------------------------|--------------------------|
| Gradient contamination | Yes — aux loss gradient flows through router weights, distorting them from optimal LM routing | No — bias is updated with `torch.no_grad()` |
| Hyperparameter sensitivity | High — α too large kills LM performance, α too small fails to balance | Low — γ=0.001 works across scales |
| Effect on router learning | Router learns to balance AND route, conflicting objectives | Router learns ONLY to route; bias handles balance |
| Convergence | Router must solve a multi-objective optimization | Bias converges independently via control theory |
| Implementation complexity | Requires differentiable load counting | Simple running statistics |

### 4.6 Gamma schedule: freeze at 80% of training

```python
# model.py line 1534-1540 — NanoSeekModel.get_gamma()
def get_gamma(self, tokens_processed=None):
    if tokens_processed is None:
        tokens_processed = self.tokens_processed.item()
    freeze_at = int(self.config.total_tokens * self.config.moe.gamma_freeze_ratio)
    # freeze_at = 22B × 0.80 = 17.6B tokens
    if tokens_processed < freeze_at:
        return 0.001
    return 0.0  # Freeze bias — stop updating
```

**Why freeze?** Two reasons:

1. **Late-training stability:** By 80% of training, the router has learned stable token-expert assignments. Continuing to adjust the bias introduces noise that can destabilize the final convergence.

2. **Router adaptation:** After the bias freezes, the router weights have the remaining 20% of training to adapt to the now-fixed bias landscape. This lets the router fully internalize the load-balanced routing pattern.

**DeepSeek V3's schedule:** They freeze at 14.3T/14.8T ≈ 96.6% of training. NanoSeek uses 80% as a safety margin — at smaller scale, you want more time for the router to adapt after freezing.

### 4.7 The training loop interaction

Here's how the bias update integrates into the training loop (called from the training script, not inside the model forward pass):

```python
# Simplified training loop pseudocode
for step, batch in enumerate(dataloader):
    # Forward pass — computes losses, records expert_load
    outputs = model(input_ids, labels=labels)
    loss = outputs["loss"]  # main + mtp + seq_aux

    # Backward pass — gradients computed, bias NOT involved
    loss.backward()

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    # Bias update — AFTER optimizer step, OUTSIDE autograd
    gamma = model.get_gamma(tokens_processed)
    model.update_load_balance_bias(gamma)

    # Update token counter
    model.update_tokens_processed(batch_size * seq_len)
```

The bias update is a post-step hook, completely decoupled from the gradient computation.

---

## 🔴 STAGE 5: OUTPUTS — What Comes Out

### 5.1 MoE layer output

```
Input:  x ∈ ℝ^[B, L, 2048]   — post-attention hidden states
Output: y ∈ ℝ^[B, L, 2048]   — MoE-processed hidden states

y = routed_output + shared_output
  where:
    routed_output[i] = Σ_{k∈top8(i)} w[i,k] × Expert_k(x[i])
    shared_output[i] = SharedExpert_0(x[i]) + SharedExpert_1(x[i])
```

### 5.2 Auxiliary data dictionary

During training, `MoE.forward()` returns auxiliary data alongside the output:

```python
# Return type: Tuple[Tensor, Dict]
output, aux_data = self.ffn(hidden_states)

# aux_data contents:
{
    "expert_load": Tensor[64],      # How many tokens each expert processed
    "seq_aux_loss": Tensor[],       # Scalar — sequence-level auxiliary loss (if alpha > 0)
}
```

### 5.3 How MoE output integrates into the decoder layer

```python
# model.py line 1448-1471 — NanoSeekDecoderLayer.forward()

# Attention block
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)
hidden_states, present_key_value = self.self_attn(hidden_states, ...)
hidden_states = residual + hidden_states

# FFN/MoE block
residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states, ffn_aux = self.ffn(hidden_states)  # ← MoE here (layers 2-15)
hidden_states = residual + hidden_states           # ← Residual connection

return hidden_states, present_key_value, ffn_aux
```

The residual connection around MoE is critical. Even if routing is suboptimal, the residual path preserves the attention output. The MoE layer's contribution is additive — it refines the representation rather than replacing it.

### 5.4 Aggregate statistics across layers

The model collects per-layer stats via `NanoSeekModel.get_expert_load_stats()` (line 1745), returning `Dict[layer_idx, {"expert_load": [64], "expert_bias": [64]}]` — one entry per MoE layer (14 total).

---

## Common Misconceptions

### Misconception 1: "MoE experts learn to be domain specialists from the start"

**Reality:** Expert specialization is an emergent property that develops gradually. Early in training, routing is essentially random. Experts develop specializations over billions of tokens as the router learns which experts respond well to which input patterns. For the first ~10% of training, expert outputs are nearly interchangeable.

### Misconception 2: "Auxiliary-loss-free means there is no load balancing loss"

**Reality:** NanoSeek still has `seq_aux_loss_alpha = 0.0001` — a tiny sequence-level auxiliary loss. "Auxiliary-loss-free" refers to the primary balancing mechanism (bias update) being outside the gradient computation. The tiny aux loss is a complementary safety net, not the primary mechanism.

### Misconception 3: "Sigmoid routing means expert weights don't sum to 1, so the output magnitude is wrong"

**Reality:** The `route_scale = 2.5` factor, combined with the residual connection and subsequent layer norm, handles magnitude calibration. The absolute values of sigmoid scores carry meaningful information — a score of 0.9 means "this expert is very relevant" regardless of other experts' scores. With softmax, a score of 0.9 only means "much more relevant than others" — the absolute scale is lost.

### Misconception 4: "More experts always means better performance"

**Reality:** Expert count has diminishing returns and must match the token budget. Each expert needs sufficient training tokens for its parameters to converge. With NanoSeek's 22B tokens:

```
Tokens per expert = (22B × 8) / 64 = 2.75B tokens per expert
Params per expert = 4.7M
Token/param ratio = 2.75B / 4.7M = 585×  ← Healthy

If we doubled to 128 experts:
Tokens per expert = (22B × 8) / 128 = 1.375B
Token/param ratio = 293×  ← Still okay but getting thin
```

### Misconception 5: "The bias update is just a hacky workaround"

**Reality:** The bias update is a principled integral controller from control theory. It is the same mechanism that thermostats use: measure the error (load imbalance), integrate over time (accumulate in bias), apply correction (shift selection scores). It provably converges to zero imbalance under mild assumptions (bounded load variance, positive γ). This is more principled than auxiliary losses, which are heuristic penalty terms with no convergence guarantee.

### Misconception 6: "Group routing is just for distributed communication"

**Reality:** Group routing serves dual purposes. Yes, it reduces all-to-all communication in expert-parallel setups. But it also enforces **diversity** — forcing experts from at least 4 different groups prevents a token from selecting 8 experts that are all slight variations of each other (which can happen without group constraints, especially early in training).

---

## Production Gotchas

### Gotcha 1: Expert Collapse

**Symptom:** One or more experts receive near-zero traffic. Their parameters stop updating (no gradients if no tokens). The bias pushes up, but the expert's stale weights make it useless when selected, creating a death spiral.

**Detection:**
```python
load_stats = model.get_expert_load_stats()
for layer_idx, stats in load_stats.items():
    load = stats["expert_load"]
    dead = (load < load.mean() * 0.05).sum().item()
    if dead > 0:
        print(f"WARNING: Layer {layer_idx} has {dead} collapsed experts")
```

**Prevention:**
- The bias mechanism naturally prevents this by increasing dead experts' selection probability
- Monitor expert load every 100 steps during early training
- If collapse persists despite bias, check that gamma is not too small

**Recovery:** Increase gamma temporarily (e.g., 0.01) for 1000 steps, or re-initialize collapsed expert weights from a healthy expert (with small noise). In extreme cases, reload from a pre-collapse checkpoint.

### Gotcha 2: Load Imbalance at Scale

**Symptom:** Some experts consistently get 2-3× the average load. This causes two problems:
1. **Compute imbalance:** The overloaded expert becomes the bottleneck (all-to-all wait)
2. **Quality degradation:** Overloaded experts see too diverse a token distribution to specialize

**Root cause:** Usually a feedback loop — a slightly better expert gets more tokens, gets better gradients, becomes even better. The bias mechanism counteracts this, but gamma=0.001 may be too slow.

**Detection:**
```python
load = model.get_expert_load_stats()[2]["expert_load"]  # Layer 2
max_ratio = load.max() / load.mean()
if max_ratio > 2.0:
    print(f"Load imbalance: max/mean = {max_ratio:.2f}")
```

### Gotcha 3: Gamma Schedule Sensitivity

**Symptom:** Performance degrades after bias freezing (at 80% of training).

**Explanation:** If the bias was doing too much work (i.e., the router hadn't learned to balance naturally), freezing the bias exposes the router's actual imbalanced preferences. The remaining 20% of training may not be enough to adapt.

**Prevention:**
- Monitor load balance metrics BOTH with and without bias during training
- If load balance is poor without bias at 70% of training, consider extending gamma_freeze_ratio
- The ideal state at freeze time: balanced BOTH with and without bias

### Gotcha 4: Batch Size Effects on Load Statistics

**Symptom:** Load balance appears perfect during validation (small batch) but is imbalanced during training (large batch).

**Explanation:** Expert load is measured per-batch. Small batches have high variance in load statistics. The bias is updated based on these noisy measurements, which can cause oscillation.

**Mitigation:** Use exponential moving average of load statistics for bias updates (not currently implemented in NanoSeek but common in production): `ema_load = 0.99 * ema_load + 0.01 * current_load`.

### Gotcha 5: Inference Without Bias

**Symptom:** Quality differences between training and inference routing.

**Explanation:** During training, the bias affects expert selection. At inference, the bias is NOT used (see `model.py` line 419-420). If training heavily relies on bias to route tokens correctly, inference routing may diverge.

**Mitigation:** This is actually the intended design. By freezing bias at 80% and training for 20% more, the router adapts to route correctly WITHOUT bias. By inference time, the router weights encode the balanced routing pattern directly.

### Gotcha 6: Memory Profiling

MoE layers are memory-hungry despite sparse activation. Per MoE layer in BF16: 64 experts × 4.7M × 2 bytes = 602 MB. Across 14 layers: ~8.7 GB (parameters alone). With AdamW optimizer states (FP32 m, v, master weights): ~52 GB total. **MoE saves compute (only 8 experts run) but NOT memory (all 64 are stored).** This is why expert parallelism exists in production.

---

## 2026 Best Practices

### Best Practice 1: Start with Proven Configurations

Do not innovate on multiple axes simultaneously. NanoSeek's configuration is derived directly from DeepSeek V3:

```
Proven and safe to keep:
  - 64 experts, 8 active (12.5% activation)
  - Sigmoid scoring with route_scale ≈ sqrt(K)
  - Gamma = 0.001, freeze at 80%
  - Group routing with topk_group = n_group / 2
  - SwiGLU expert FFN
  - Shared experts = 2

Safe to experiment with:
  - Expert intermediate dimension (768 → 512 or 1024)
  - Number of shared experts (1 vs 2 vs 4)
  - Group count / structure
  - Gamma freeze ratio (0.75 → 0.90)
```

### Best Practice 2: Monitor Expert Entropy

Expert entropy is the single most informative metric for MoE health:

```python
# Compute expert selection entropy
load = model.get_expert_load_stats()[2]["expert_load"]
probs = load / load.sum()
entropy = -(probs * torch.log(probs + 1e-10)).sum()
max_entropy = math.log(64)  # Uniform distribution

utilization = entropy / max_entropy  # Should be > 0.90
```

If utilization drops below 0.85, investigate immediately. Below 0.75 means multiple experts are collapsing.

### Best Practice 3: Use Gradient Checkpointing for MoE Layers

MoE layers have high activation memory because all 8 expert forward passes must be stored for backward:

```
Per MoE layer activations:
  Expert inputs:  N × K × D = 256 × 8 × 2048 = 4M values = 8 MB (BF16)
  Expert intermediates: 256 × 8 × 768 × 2 = 3.1M values = 6.2 MB
  Total per MoE layer: ~14 MB
  14 layers: ~200 MB
```

Gradient checkpointing recomputes these during backward, trading compute for memory. Essential for training with large batch sizes.

### Best Practice 4: Validate Load Balance Before Scaling

Before committing to a full training run:

```bash
# Run 1000 steps and check balance
python scripts/pre-train.py --num_iterations=1000
# Inspect wandb/tensorboard for:
#   expert_load_variance per layer
#   expert_entropy per layer
#   bias magnitude distribution
```

If load balance is poor after 1000 steps with gamma=0.001, something is wrong with the initialization or configuration. Do not hope it will fix itself at scale.

### Best Practice 5: Separate Router Learning Rate (Advanced)

In production settings, use a lower learning rate for the router than for expert parameters. The router's gradients are noisier (they depend on which experts were selected, which is stochastic). A lower LR prevents overshooting. Not implemented in NanoSeek, but straightforward: create separate `AdamW` param groups with `lr=1e-4` for `gate.weight` and `lr=3e-4` for expert parameters.

### Best Practice 6: Inference Optimization — Expert Caching

At inference time, most tokens route to a small subset of popular experts. The top 16 experts typically handle ~60% of assignments. This suggests a 2-tier expert placement strategy: frequently-used experts in HBM/L2, less-used experts paged from slower storage.

### Best Practice 7: The 2026 Frontier — Dynamic Expert Count

The sigmoid + bias approach already decouples expert scores from normalization. The natural next step: **dynamic K** — varying active experts per token based on difficulty (K=4 for common words, K=12 for rare entities). Straightforward with sigmoid (scores are absolute, not relative) but requires careful compute budget calibration. NanoSeek's architecture is already compatible.

---

## Appendix A: Parameter Counts (Verified)

| Component | Count | Per Layer | × 14 Layers |
|-----------|-------|-----------|-------------|
| Per routed expert (SwiGLU 2048→768→2048) | 4,718,592 | — | — |
| 64 routed experts | — | 301,989,888 | ~4.23B |
| 2 shared experts | — | 9,437,184 | ~132M |
| Router weight [64, 2048] | — | 131,072 | ~1.8M |
| **Total MoE params** | — | **~312M** | **~4.36B** |
| **Active per forward** (8 routed + 2 shared + router) | — | **~47.3M** | **~662M** |
| **Expansion ratio** (MoE subsystem) | | | **6.6×** |

## Appendix B: Configuration Reference

```python
# From model/config.py — MoEConfig
MoEConfig(
    n_routed_experts=64,           # Total routed experts
    num_experts_per_tok=8,         # Active per token
    n_shared_experts=2,            # Always-active shared experts
    moe_intermediate_size=768,     # Expert FFN hidden dim
    n_group=8,                     # Expert groups for routing
    topk_group=4,                  # Groups selected per token
    scoring_func="sigmoid",        # Scoring function
    routed_scaling_factor=2.5,     # Output scale factor
    norm_topk_prob=True,           # Normalize top-k probs
    gamma=0.001,                   # Bias update rate
    gamma_freeze_ratio=0.80,       # Freeze bias at 80% training
    seq_aux_loss_alpha=0.0001,     # Tiny sequence auxiliary loss
    first_k_dense_replace=2,       # Dense FFN for layers 0-1
)
```

## Appendix C: Key Code References

| Component | File | Line | Class/Function |
|-----------|------|------|----------------|
| Gate (router) | `model/model.py` | 379 | `Gate` |
| SwiGLU expert | `model/model.py` | 445 | `SwiGLUFFN` |
| Token-centric dispatch | `model/model.py` | 485 | `token_centric_dispatch()` |
| MoE layer | `model/model.py` | 586 | `MoE` |
| Bias update | `model/model.py` | 721 | `MoE.update_load_balance_bias()` |
| Gamma schedule | `model/model.py` | 1534 | `NanoSeekModel.get_gamma()` |
| MoE config | `model/config.py` | 282 | `MoEConfig` |
| Layer assignment | `model/config.py` | 599 | `NanoSeekConfig.moe_layer_indices` |
| Decoder integration | `model/model.py` | 1378 | `NanoSeekDecoderLayer` |
| Loss computation | `model/model.py` | 1697 | `NanoSeekModel._compute_loss()` |
| MoE tests | `tests/test_moe.py` | — | Full MoE test suite |

---

*Document version: 2026-02-27. Based on NanoSeek codebase at commit HEAD. All tensor shapes and line numbers verified against `model/model.py` and `model/config.py`.*
