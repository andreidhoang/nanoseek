# Kimi Linear: An Expressive, Efficient Attention Architecture
## Paper Summary & Deep Analysis

**Paper**: arXiv:2510.26692 (October 2025)
**Authors**: Kimi Team (Moonshot AI)
**Code**: https://github.com/MoonshotAI/Kimi-Linear
**Model**: 48B total / 3B active (MoE), 256 experts, top-8

---

## 1. Core Contribution

Kimi Linear is the **first hybrid linear attention architecture that outperforms full attention** under fair comparisons across short-context, long-context, and RL scaling regimes. It introduces **Kimi Delta Attention (KDA)**, a channel-wise gated variant of the delta rule for linear attention, combined with periodic full-attention (MLA) layers in a 3:1 ratio.

**Key results**:
- Outperforms full MLA on **all** evaluated tasks (short-context, long-context, SFT, RL)
- Up to **75% KV cache reduction** and **6x decoding throughput** at 1M context
- **~1.16x compute efficiency** vs MLA baselines in scaling law experiments
- Drop-in replacement for full attention (no modification to caching/scheduling)

---

## 2. Mathematical Foundation

### 2.1 Linear Attention as Online Learning

The paper frames all linear attention variants through an **online learning / fast-weight** lens:

**Standard Linear Attention** (Katharopoulos 2020):
```
S_t = S_{t-1} + k_t v_t^T        (state accumulation)
o_t = S_t^T q_t                   (output query)
```
This is gradient descent on the **unbounded correlation objective**:
```
L_t(S) = -<S^T k_t, v_t>
```
Problem: No forgetting mechanism, state grows unbounded, interference over long contexts.

**DeltaNet** (Schlag 2021):
Reinterprets as gradient descent on a **reconstruction loss**:
```
L_t(S) = (1/2) ||S^T k_t - v_t||^2
S_t = (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T
```
This is the **classical delta rule** — the state `S` is a learnable associative memory that self-corrects toward the mapping `k_t -> v_t`. The rank-1 update structure is equivalent to a **generalized Householder transformation**, enabling hardware-efficient chunkwise parallelization.

**Gated DeltaNet (GDN)** (Yang 2025):
Adds a **scalar forget gate** (weight decay on fast weights):
```
S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T
```
where `alpha_t in [0,1]` is a data-dependent **scalar** gate (per-head).

### 2.2 KDA: The Channel-Wise Gated Delta Rule

**KDA's core recurrence** (Eq. 1 of the paper):
```
S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
o_t = S_t^T q_t
```

**Critical difference from GDN**: `alpha_t` is a **vector** `in R^{d_k}`, not a scalar. Each feature dimension has an **independent forgetting rate**. This is the key innovation:

| Method   | Gate Type | Granularity |
|----------|-----------|-------------|
| RetNet   | Scalar, static | Per-model |
| Mamba2   | Scalar, dynamic | Per-head |
| GDN      | Scalar, dynamic | Per-head |
| GLA      | Diagonal, dynamic | Per-channel (no delta rule) |
| **KDA**  | **Diagonal, dynamic** | **Per-channel + delta rule** |

### 2.3 KDA as Learnable Position Embeddings

A profound insight: KDA serves as a **data-dependent, learnable positional encoding** that generalizes RoPE. The output can be written as:

```
o_t = sum_{i=1}^{t} (q_t^T (prod_{j=i+1}^{t} Diag(alpha_j)(I - beta_j k_j k_j^T)) k_i) v_i
```

Compare with RoPE in softmax attention:
```
s_{t,i} = q_t^T (prod_{j=i+1}^{t} R_j) k_i
```

**RoPE**: Fixed rotation matrices (block-diagonal, orthogonal, per-2-dimensional frequencies)
**KDA**: Data-dependent transition matrices (diagonal + rank-1, non-orthogonal, per-channel decay)

This means KDA **relaxes the orthogonality constraint of RoPE** and can be "potentially more powerful." This is why they use **NoPE** (No Position Encoding) for the MLA layers — KDA handles all positional information.

### 2.4 KDA as Constrained DPLR

KDA is a **constrained Diagonal-Plus-Low-Rank** (DPLR) formulation:
```
S_t = (D - a_t b_t^T) S_{t-1} + k_t v_t^T
```
where:
- `D = Diag(alpha_t)`
- `a_t = beta_t k_t`
- `b_t = k_t * alpha_t` (element-wise)

By **binding both a and b to k**, KDA achieves:
- Removes 2 secondary chunking steps (numerical stability without log-domain tricks)
- Eliminates ~3 matrix multiplications in inter-chunk computation
- **~2x kernel speed** vs general DPLR for sequences up to 64K

---

## 3. Hardware-Efficient Chunkwise Algorithm

### 3.1 WY Representation

The cumulative product of Householder-like matrices `P_{[t]}^r` is packed into a compact WY form:
```
P_{[t]}^r = Diag(gamma_{[t]}^r) - sum_{i=1}^{r} Diag(gamma_{[t]}^{i->r}) k_{[t]}^i w_{[t]}^{iT}
```

Auxiliary vector `w` via recurrence:
```
w_{[t]}^r = beta_{[t]}^r (Diag(gamma_{[t]}^r) k_{[t]}^r - sum_{i=1}^{r-1} w_{[t]}^i (k_{[t]}^{iT} Diag(gamma_{[t]}^{i->r}) k_{[t]}^r))
```

Similarly for `H_{[t]}^r` (the history term) with auxiliary vector `u`.

### 3.2 UT Transform

Applied to reduce non-matmul FLOPs (crucial for Tensor Core utilization):
```
M_{[t]} = (I + StrictTril(Diag(beta) (Gamma * K)(K/Gamma)^T))^{-1} Diag(beta)
W_{[t]} = M_{[t]} (Gamma * K)
U_{[t]} = M_{[t]} V_{[t]}
```

### 3.3 Chunk State Update
```
S_{[t+1]} = Diag(gamma_{[t]}^C) S_{[t]} + (Gamma * K)^T (U - W S_{[t]})
```

### 3.4 Output Computation (Inter + Intra chunk)
```
O_{[t]} = (Gamma * Q) S_{[t]}                                  [inter-chunk: state readout]
        + Tril((Gamma * Q)(K/Gamma)^T) (U - W S_{[t]})        [intra-chunk: within-chunk attention]
```

### 3.5 FLOPs Analysis

Per head, per sequence of length T, with chunk size C=64 and head dim d_h:
```
FLOPs_KDA = 6T d_h^2 + 3T C d_h + T C^2
FLOPs_Attn = 2T^2 d_h    (full attention)
```

KDA is **linear in T** (the dominant term is `6T d_h^2`), while full attention is **quadratic** (`2T^2 d_h`). The crossover point where KDA becomes cheaper: `T > 3 d_h` (for d_h=128, this is T > 384).

---

## 4. Neural Parameterization

### 4.1 Input Projections
```python
q, k = L2Norm(Swish(ShortConv(W_qk @ x)))   # in R^{d_k}
v = Swish(ShortConv(W_v @ x))                # in R^{d_v}
alpha = f(W_alpha_up @ W_alpha_down @ x)      # in [0,1]^{d_k}, low-rank
beta = Sigmoid(W_beta @ x)                    # in [0,1], scalar per head
```

Key design choices:
- **ShortConv** (kernel=4): Captures local token dependencies before attention
- **L2Norm** on q,k: Ensures eigenvalue stability (from DeltaNet paper)
- **Swish** activation: Applied before normalization
- **Low-rank alpha**: `W_alpha_down in R^{d_k x d}`, `W_alpha_up in R^{d x d_k}` (rank = head dimension)
- d_k = d_v = 128 for all experiments

### 4.2 Output Gating
```python
o = W_o @ (Sigmoid(W_g_up @ W_g_down @ x) * RMSNorm(KDA(q, k, v, alpha, beta)))
```

- **Sigmoid output gate** (NOT Swish) — paper shows Sigmoid significantly outperforms Swish gating
- **Low-rank output gate**: Same low-rank structure as forget gate (fair parameter comparison)
- **Head-wise RMSNorm**: Applied before gating
- The output gate alleviates the **Attention Sink** problem

---

## 5. Model Architecture

### 5.1 Hybrid Design: 3:1 KDA:MLA Ratio
- Every 4 layers: 3 KDA layers + 1 full MLA layer
- **Layerwise hybrid** (not headwise) — chosen for infrastructure simplicity and training stability
- The MLA layers preserve **global information flow** for long-range dependencies

### 5.2 NoPE for MLA Layers
- All full attention (MLA) layers use **No Position Encoding**
- KDA layers handle ALL positional information (as learnable position embeddings)
- Benefits:
  - MLA layers can be converted to pure MQA during inference (more efficient)
  - Simplifies long-context training (no RoPE frequency tuning, no YaRN)
  - More balanced positional bias across depth

### 5.3 Scale
- 48B total / 3B active parameters
- 8 out of 256 experts activated (1 shared expert)
- First layer is dense (no MoE) — ensures stable training
- Architecture follows Moonlight (from Moonshot AI)
- Muon optimizer (MuonClip variant)

---

## 6. Training & Evaluation Details

### 6.1 Pre-training
- **1.4T tokens** (comparison experiments) / **5.7T tokens** (final released model)
- 4096 context window for pre-training
- MuonClip optimizer, WSD learning rate schedule
- LR = 1.1e-3, batch size = 32M tokens
- Same annealing schedule as Kimi K2

### 6.2 Post-training
- **Multi-stage SFT**: General instruction-following -> reasoning-intensive data
- **RL with RLVR**: Math, code, STEM prompts
  - Truncated importance sampling (mitigates policy mismatch between rollout and training)
  - Dynamic KL penalty adjustment
  - Dynamic mini-batch size (prevents entropy collapse)
  - PTX loss (concurrent SFT during RL to prevent capability degeneration)

### 6.3 Evaluation Benchmarks

**Short-context**: HellaSwag, ARC-C, Winogrande, MMLU, TriviaQA, MMLU-Pro, MMLU-Redux, GPQA-Diamond, BBH, LiveBench
**Math & Code**: AIME 2025, MATH500, HMMT 2025, PolyMath-en, LiveCodeBench v6, EvalPlus
**Long-context**: RULER, MRCR, HELMET-ICL, LongBench V2, Frames, RepoQA, Long Code Arena
**Chinese**: C-Eval, CMMLU

### 6.4 Evaluation Configuration
- Temperature 1.0 for all evaluations
- Avg@k for high-variance benchmarks
- GPQA-Diamond: mean across 8 independent runs
- LM-Harness-Evaluation framework (internal fork)

---

## 7. Key Experimental Results

### 7.1 Synthetic Tasks (KDA vs GDN vs Mamba2)
- **Palindrome** (reverse copying): KDA >> GDN > Mamba2 (fails)
- **MQAR** (multi-query associative recall): KDA >> GDN > Mamba2 (fails)
- **Stack** (LIFO state tracking, 64 stacks): KDA > GDN > Mamba2 (fails)
- KDA converges significantly faster than GDN on all tasks
- Mamba2 fails completely — no delta rule means no precise memory retrieval

### 7.2 Ablation Study Results

| Configuration | Train PPL | Val PPL |
|---------------|-----------|---------|
| **3:1 (final)** | **9.23** | **5.65** |
| 0:1 (full attention only) | 9.45 | 5.77 |
| 1:1 | 9.29 | 5.66 |
| 7:1 | 9.23 | 5.70 |
| 15:1 | 9.34 | 5.82 |
| w/o output gate | 9.25 | 5.67 |
| w/ Swish output gate | 9.43 | 5.81 |
| w/o convolution layer | 9.29 | 5.70 |

**Key takeaways**:
- 3:1 ratio is optimal (best validation PPL)
- 7:1 matches training PPL but hurts generalization
- Full attention (0:1) performs worst — linear attention helps!
- Sigmoid gate >> Swish gate >> no gate
- Short convolutions contribute meaningfully even in hybrid models

### 7.3 Scaling Law
- KDA achieves **~1.16x compute efficiency** vs MLA baselines with compute-optimal training
- 5 model sizes tested with Chinchilla methodology
- Same training configuration used (only 3:1 ratio changes)

### 7.4 Pre-training Results (1.4T tokens, 48B/3B model)

| Benchmark | MLA | GDN-H | **Kimi Linear** |
|-----------|-----|-------|-----------------|
| HellaSwag | 81.7 | 82.2 | **82.9** |
| MMLU | 71.6 | 72.2 | **73.8** |
| MMLU-Pro | 47.2 | 47.9 | **51.0** |
| BBH | 71.6 | 70.6 | **72.9** |
| TriviaQA | 68.9 | 70.1 | **71.7** |
| GSM8K | 83.7 | 81.7 | **83.9** |

Kimi Linear wins on **all** general and most math/code benchmarks.

### 7.5 Long-Context Results (128K, post-SFT)

| Benchmark | MLA | GDN-H | KL (RoPE) | **Kimi Linear** |
|-----------|-----|-------|-----------|-----------------|
| RULER | 81.3 | 80.5 | 78.8 | **84.3** |
| MRCR | 22.6 | 23.9 | 22.0 | **29.6** |
| HELMET-ICL | 88.0 | 85.5 | 88.0 | **90.0** |
| RepoQA | 63.0 | 63.0 | 66.5 | **68.5** |
| **Average** | 52.2 | 51.2 | 51.8 | **54.5** |

NoPE + KDA >> RoPE + KDA for long-context tasks.

### 7.6 RL Results
- Kimi Linear has **higher training accuracy growth rate** than MLA during RLVR
- Gap **widens over time** — KDA benefits more from RL scaling
- On MATH500 and AIME2025: faster and better improvement vs MLA
- This is the first evidence that **linear attention scales better under RL** than full attention

### 7.7 Efficiency Results (48B/3B model)

| Context Length | Prefill Speedup | Decode Speedup |
|----------------|-----------------|----------------|
| 4K-16K | ~1x (comparable) | ~1x |
| 128K | ~1.5x | ~2x |
| 512K | 2.3x | ~4x |
| 1M | 2.9x | **6x** (batch-optimal: 6.3x) |

### 7.8 5.7T Token Results
- RULER score: **94.8 at 1M context length**
- Consistently outperforms Moonlight across nearly all benchmarks

---

## 8. Monitoring, Measurement & Evaluation Techniques

### 8.1 Metrics Used
1. **Training PPL / Validation PPL**: Standard language modeling metrics
2. **Benchmark suites**: HellaSwag, MMLU, BBH, GSM8K, etc. (see Section 6.3)
3. **Scaling law fitting**: Chinchilla methodology (5 model sizes)
4. **RL convergence curves**: Training accuracy, test accuracy over RL steps
5. **Prefill/decode latency**: Wall-clock timing at various context lengths
6. **Kernel speed benchmarks**: Comparison of KDA vs DPLR kernel speeds

### 8.2 Evaluation Methodology
- **Fair comparison**: Same architecture, parameter count, training setup, token budget
- **Temperature 1.0**: For all evaluations
- **Avg@k**: For high-variance benchmarks (AIME: Avg@64, HMMT: Avg@32, GPQA: Avg@8, PolyMath: Avg@4)
- **Perplexity-based** for MMLU, MMLU-Redux, GPQA-Diamond, C-Eval (base model)
- **Generation-based** for all other benchmarks
- **Validation on distribution-shifted data**: Val set distribution "differs significantly" from pre-training corpus

### 8.3 Ablation Design
- All ablations use **first-scale scaling law model** (16 heads, 16 layers)
- Same FLOPs budget and hyperparameters across all compared models
- Variables tested: hybrid ratio, output gate type, convolution layer, NoPE vs RoPE

### 8.4 Synthetic Benchmarks for Linear Attention
Three synthetic tasks to probe specific capabilities:
1. **Palindrome**: Tests precise reverse-order retrieval (hardest for compressed memory)
2. **MQAR (Multi-Query Associative Recall)**: Tests multi-position value retrieval (correlated with LM performance)
3. **Stack (64 LIFO stacks)**: Tests state tracking across many parallel entities

All use: 2 layers, 2 heads, d_h=128. Training length 256-2048, grid search over 4 LRs, max 20K steps.

---

## 9. Connection to NanoSeek Project

### 9.1 Direct Architectural Relevance

NanoSeek uses **MLA (Multi-head Latent Attention)** — the same attention mechanism that Kimi Linear hybridizes with. The Kimi Linear paper provides a compelling case for:

1. **Hybrid KDA+MLA as a next-generation attention architecture**: NanoSeek currently uses full MLA everywhere. The paper shows that replacing 75% of MLA layers with KDA layers **improves quality** while dramatically reducing inference cost.

2. **NoPE for MLA layers**: Kimi Linear applies No Position Encoding to MLA layers, letting KDA handle all positional information. This simplifies long-context extension (no RoPE tuning needed).

3. **MoE compatibility**: Kimi Linear uses the **same Moonlight MoE architecture** (expert parallelism, shared experts, top-8 routing) as NanoSeek. The paper validates that linear attention works well with MoE at the 48B/3B scale.

### 9.2 Potential NanoSeek Experiments

**Experiment 1: Hybrid KDA+MLA NanoSeek**
- Replace 75% of MLA layers with KDA layers in NanoSeek's 16-layer decoder
- Layout: [KDA, KDA, KDA, MLA] x 4 = 12 KDA + 4 MLA layers
- Expected: Better quality + 75% KV cache reduction
- Risk: KDA kernel availability (need flash-linear-attention library)

**Experiment 2: NoPE for MLA layers**
- Remove RoPE/YaRN from MLA layers, keep RoPE only for KDA (or use KDA's native positional encoding)
- Expected: Simpler long-context extension, comparable short-context quality
- This would eliminate NanoSeek's YaRN complexity for Phase 2

**Experiment 3: Scaling law comparison**
- Run NanoSeek's muP scaling law experiments with both MLA-only and KDA+MLA configurations
- Test if the 1.16x compute efficiency transfers to NanoSeek's scale (1B active)

**Experiment 4: RL scaling comparison**
- The paper's most surprising finding: KDA benefits MORE from RL than MLA
- Test if NanoSeek's GRPO pipeline shows similar accelerated improvement with KDA+MLA

### 9.3 Implementation Considerations

**What would change in NanoSeek's model.py**:
- New `KDALayer` class implementing the recurrent + chunkwise algorithm
- Modified `DecoderLayer` to support both MLA and KDA attention types
- Layer assignment logic: `layer_type = 'kda' if (i % 4 != 3) else 'mla'`
- KDA needs: ShortConv, L2Norm, Sigmoid gate, low-rank alpha/beta projections
- State management: fixed-size `d_k x d_v` state per head (no KV cache for KDA layers)

**What stays the same**:
- MoE expert architecture (unchanged)
- MTP (Multi-Token Prediction, unchanged)
- FIM (Fill-in-the-Middle, unchanged)
- Training infrastructure (EMA, muP, checkpoint management)
- Evaluation framework

### 9.4 Key Insights for NanoSeek Development

1. **Output gate matters**: Sigmoid >> Swish >> None. Check NanoSeek's gating choices.

2. **Short convolutions matter even in hybrid models**: Don't skip ShortConv in KDA layers.

3. **The delta rule is critical for retrieval**: Mamba2 (no delta rule) fails synthetic tasks. GLA (no delta rule) is weaker. The delta rule is what enables precise memory retrieval in linear attention.

4. **Per-channel gating >> per-head gating**: This is the core insight — finer-grained forgetting enables better use of the finite-state memory.

5. **The 3:1 ratio is optimal**: Not 1:1, not 7:1. The sweet spot is 75% linear + 25% full attention.

6. **NoPE works when linear attention handles position**: Positional encoding in MLA layers is actually **harmful** for long-context performance when KDA already provides positional information.

7. **Linear attention + RL synergy**: Most surprising finding. Suggests that linear attention's compressed representation may be more amenable to RL optimization than full attention's redundant representation.

---

## 10. Comparison Table: All Linear Attention Variants

| Method | Gate | Delta Rule | Update Rule |
|--------|------|------------|-------------|
| Linear Attn | None | No | S_t = S_{t-1} + k v^T |
| RetNet | Scalar (static) | No | S_t = alpha S_{t-1} + beta k v^T |
| Mamba2 | Scalar (dynamic) | No | S_t = alpha_t S_{t-1} + beta_t k v^T |
| GLA | Diag (dynamic) | No | S_t = Diag(alpha_t) S_{t-1} + k v^T |
| DeltaNet | None | Yes | S_t = (I - beta k k^T) S_{t-1} + beta k v^T |
| GDN | Scalar (dynamic) | Yes | S_t = alpha_t (I - beta k k^T) S_{t-1} + beta k v^T |
| RWKV7 | Diag (dynamic) | Yes (DPLR) | S_t = (Diag(alpha_t) - (b*k_hat) k_hat^T) S_{t-1} + k v^T |
| **KDA** | **Diag (dynamic)** | **Yes** | **S_t = (I - beta k k^T) Diag(alpha_t) S_{t-1} + beta k v^T** |

---

## 11. Open Questions & Future Directions

1. **Sparse + Linear hybrid**: Can NSA/MoBA sparse attention be combined with KDA linear attention for even better quality/efficiency?
2. **State expansion**: Can KDA's finite state be expanded (via techniques from MoM, Log-Linear) to match full attention on exact copying?
3. **Scaling beyond 48B**: Does the 1.16x compute efficiency hold at 100B+ scale?
4. **KDA for vision**: Can the delta rule + channel-wise gating help in vision transformers?
5. **Optimal hybrid ratio at different scales**: Is 3:1 universally optimal, or scale-dependent?

---

*Summary generated from arXiv:2510.26692 TeX source. Analysis contextualized for NanoSeek project.*

Sources:
- [Kimi Linear Paper (arXiv)](https://arxiv.org/abs/2510.26692)
- [GitHub Repository](https://github.com/MoonshotAI/Kimi-Linear)
- [HuggingFace Model](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)
