# muP: Tensor Programs V — Tuning Large Neural Networks via Zero-Shot HP Transfer
## Paper: arxiv.org/abs/2203.03466 (Yang et al., Microsoft Research)

---

## Core Guarantee

muP (Maximal Update Parametrization) ensures that **optimal hyperparameters converge as width → ∞**. Tune HPs on a cheap narrow proxy → copy directly to wide target. No adjustment needed.

**Why it works**: muP ensures every layer's activations update on the same order during training, regardless of width. SP (standard param) causes logits to blow up with width while embeddings barely move.

**What transfers**: LR, momentum, Adam betas, LR schedule, init variance, parameter multipliers.
**What does NOT transfer**: dropout, weight decay (regularization depends on data size, not just model size).

---

## The Scaling Rules (Practical Form)

For Adam optimizer (most relevant to NanoSeek):

| Weight Type | Init Variance | Adam LR | Examples |
|-------------|--------------|---------|----------|
| Input (embedding) | 1/fan_in | η (constant) | Token embeddings |
| Hidden (matrix) | 1/fan_in | **η/fan_in** (shrinks) | W_q, W_k, W_v, W_o, MLP W1/W2, expert FFN |
| Output (lm_head) | 1/fan_in² | **η/fan_in** (shrinks) | LM head, output projections |

**Critical for Transformers**: Use **1/d_head attention** (not 1/√d_head). During training Q and K correlate, so QK^T scales as d (LLN), not √d (CLT).

**Base-width formulation**: Define `r = width / base_width`. At base_width, muP = SP. As width grows, hidden LRs scale as `η/r`, output init scales as `1/width²`.

---

## Practical Recipe

1. Build proxy by shrinking **width only** (keep depth, d_head ≥ 32)
2. Classify params: matrix-like (hidden), vector-like (input/output/bias/norm)
3. Tune on proxy: LR, init scale, alpha_output, alpha_attn, embedding LR
4. Copy HPs directly to target (muP scaling handles the rest)
5. **Verify**: coordinate check at 2+ widths — activations should be O(1)
6. **Verify**: wider-is-better — same HPs should give lower loss at wider model

---

## Key Experiments & Anchor Sizes

| Target | Proxy | Width Ratio | HPs Tuned | Grid Size | Result |
|--------|-------|-------------|-----------|-----------|--------|
| GPT-3 6.7B | 40M (w=256) | 16× | 6 HPs | 467 runs | Beat published GPT-3 6.7B |
| BERT-large 350M | 13M (w=256) | 16× | 6 HPs | 256 runs | Beat Megatron BERT |
| WMT 211M | 15M (w=256) | 4× | 3 HPs | ~2304 grid | Matched fairseq default |
| IWSLT 40M | 4M (w=128) | 4× | 3 HPs | ~2304 grid | Significant improvement |

**The GPT-3 recipe**: 467 proxy runs at 40M = 7% of one 6.7B pretraining cost. Found HPs that beat the original GPT-3.

---

## Limitations Critical for NanoSeek

1. **MoE NOT covered** — muP was derived for dense networks. Router weights, expert capacity, load balancing are uncharted.
2. **Depth transfer is empirical only** — init_std does NOT transfer across depth. Pre-layernorm required.
3. **d_head ≥ 32 recommended** — smaller makes HP landscape too noisy (Fig 13).
4. **FP16 can underflow** — muP picks more aggressive HPs. GPT-3 experiment needed FP32.
5. **Minimum width ~256** — empirical, not theoretical. Below this, proxy HP landscape is too noisy.
6. **Weight decay doesn't transfer** — must tune separately at target.

---

## Relevance to NanoSeek

### What muP means for our 2-scale plan (anchor d=768 → 1B d=2048)

**Width ratio**: 2048/768 = 2.67×. This is modest — GPT-3 experiment used 16×.
The transfer should be reliable at this ratio.

**What to tune at anchor**: LR (matrix_lr), embedding_lr, init_scale, alpha_output.
**What to tune at 1B directly**: weight_decay, dropout (if any).

### The MoE problem

muP's theory assumes dense layers. In NanoSeek:
- **Expert FFN weights**: These are "hidden weights" (matrix-like) → LR ∝ 1/width ✓
- **Router weights**: These project d → n_experts. As d scales, fan_in changes but fan_out (64 experts) is fixed → router is "output-like" → LR constant ✓
- **Expert selection (top-k routing)**: Discrete, non-differentiable — outside muP's scope entirely

The router + expert FFN scaling follows muP rules naturally. The ROUTING DYNAMICS (which experts get selected, load balance) are NOT covered by muP and may behave differently at different scales.

**Implication**: HP transfer should work for learning rates and init. Whether routing dynamics transfer (same I_spec trajectory, same specialization timing) is our research question — muP doesn't predict this.

### The anchor size question

Our anchor: d=768, ~175M active. The muP paper's smallest reliable proxy was d=256 (~13-40M). Our anchor at d=768 is **3× larger than their minimum**. This is very safe.

The paper used 256-467 HP search runs. We plan 12. This is 20-40× fewer samples.
But: our search space is 2D (matrix_lr × embedding_lr = 4×3 = 12), theirs was 3-6D.
With only 2 free dimensions, 12 points is sufficient for a coarse grid.

### Could we skip the anchor entirely?

The GPT-3 experiment spent 7% of pretraining cost on HP search at proxy scale.
For NanoSeek: 1B pretraining = ~$350. 7% = $24.50.
Our HP grid at anchor costs ~$120. That's 34% of pretraining cost.

**If we reduced to 6 HP runs** (3 matrix_lr × 2 embedding_lr): ~$60 = 17%.
Still workable, and twice what the muP paper suggests is sufficient.

Alternative: Run 500 steps at 1B for each HP combo. Cost per run: ~$15.
6 runs × $15 = $90. Not much more than anchor, and no transfer risk.
