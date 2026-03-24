# NanoSeek: Pre-Training Research Plan
## MoE Training Dynamics at Nano Scale
### March 2026

---

## Thesis

Train MLA + MoE + MTP models at 3 scales (55M → 441M → 1.08B active) using
the proven DeepSeek V3.2 architecture with MuonAdamW optimizer. Produce the
deepest public analysis of MoE training dynamics — expert specialization,
routing stability, collapse prediction — and validate muP HP transfer for MoE.

**What we don't do**: We don't invent new architectures. MLA, MoE, MTP, and Muon
are all validated at scale by multiple labs. KDA, QK-Clip, and hybrid attention
are interesting but unimplemented, unvalidated below 3B, and weeks of engineering
away. They are future work, not this plan.

**What we produce**: Trained models, a reusable instrumented training pipeline,
and a training dynamics report that doesn't exist anywhere in open literature.

---

## Why This Focus

### What the frontier actually needs (2026)

Every lab running MoE at scale fights the same problems daily:

1. **Expert collapse** — routing degenerates, experts die silently
2. **Stability** — loss spikes, gradient explosions, mysterious divergences
3. **HP transfer** — muP promises transfer but breaks in practice for MoE
4. **Routing dynamics** — when do experts specialize? what predicts collapse?
5. **Data interaction** — how does data mixture affect expert specialization?

These problems are **scale-universal**. Expert collapse at 55M active params
exhibits the same signatures as expert collapse at 100B. Someone who can look
at a W&B dashboard and say "expert 47 in layer 12 is about to die, here's why"
is immediately useful at any scale.

### Where small-scale work has outsized impact

| Research Area | Transfers to 100B+? | Cost at 1B | Novelty |
|---|---|---|---|
| New architecture (KDA, etc.) | Low — behavior changes with scale | High (weeks of eng) | Low (Kimi did it) |
| MoE training dynamics | **Very high** — phenomena are universal | Low | **High** (poorly documented) |
| Data mixture optimization | **Very high** — ratios transfer directly | Low | Medium |
| HP transfer validation | **High** — methodology transfers | Medium | Medium-high |
| Stability detection tools | **Very high** — tools transfer directly | Low | High |

### What we already have (inventory)

Built and tested (15K lines, 120+ tests passing):

| Component | Status | Notes |
|-----------|--------|-------|
| MLA attention (23x KV compression) | Done | Sections 1-7 in model.py |
| MoE (64 experts, top-8, grouped routing) | Done | Aux-loss-free balancing |
| MTP (speculative decoding) | Done | Lambda annealing 0.3→0.1 |
| MuonAdamW + DistMuonAdamW | Done | Newton-Schulz + muP scaling |
| EMA tracking (decay=0.9999) | Done | Karras warmup |
| FIM 10% PSM | Done | In dataloader from token 1 |
| Batch warmup (1/5→1x) | Done | First 10% of steps |
| Checkpoint resume | Done | Model + optimizer + EMA + dataloader state |
| TrainingHealthMonitor | Done | Gradient z-score, loss spike, H_load alerts |
| I_spec computation | Done | MI(expert; domain) via gate hooks |
| Dead expert detection | Done | Per-layer utilization tracking |
| MTP acceptance rate | Done | Speculative acceptance measurement |
| Domain BPB | Done | code/math/science/web/books |
| Scaling law fitter | Done | Chinchilla L(N) with LOOCV |
| Configs (anchor/500M/1B) | Done | muP-aligned, gamma_freeze=0.95 |
| Ablation flags | Done | --aux-loss-type, --inject-bad-batch, --num-experts, etc. |
| 7 critical training bugs fixed | Done | See TRAINING_BUGS_POSTMORTEM.md |

**What's NOT done**: Zero completed training runs. Smoke test v5 ran 38 steps.

---

## The Plan

### Phase 0: Build Instrumentation Infrastructure (Days 0-1)

**Goal**: All dynamics collection code exists, tested, and ready BEFORE any
training run fires. Every run — including throwaway HP search — produces dynamics
data. The instrumentation IS the infrastructure.

**Why this comes first** (Jensen's co-design principle): NVIDIA builds monitoring,
cooling, and power delivery BEFORE populating a rack with GPUs. Our equivalent:
build the MoE dynamics collector and expert gradient tracker before populating
W&B with runs. The HP grid search produces 12 runs — that's 12 data points for
dynamics analysis that we get for free if instrumentation exists on Day 1.

#### Day 0: MoEDynamicsCollector + Expert Gradient Tracker

Write and test two modules (~230 lines total, zero new dependencies):

```python
# nanoseek/eval/moe_dynamics.py (~150 lines)
class MoEDynamicsCollector:
    """Hook-based MoE routing dynamics for every eval step.

    Attaches to Gate modules. Computes per-layer:
    - routing_entropy: H(expert | layer) — uniformity of selection
    - expert_gini: Gini coefficient of load counts — inequality
    - routing_churn: fraction of tokens that changed top-1 expert vs prev eval
    - expert_output_diversity: mean pairwise cosine distance of expert hidden states
    - gate_logit_stats: max, std, kurtosis of pre-sigmoid gate logits
    - mtp_x_routing: correlation between MTP acceptance and routing entropy
    """

# Addition to pre_train.py eval loop (~80 lines)
def compute_expert_gradient_norms(model):
    """Per-expert gradient L2 norm. Called after backward(), before step().
    Returns {layer_i: {expert_j: grad_norm}}.
    """
```

**Integration points** (changes to pre_train.py):
- Instantiate `MoEDynamicsCollector` at training start
- Call `collector.step(model)` inside the existing eval loop
- Call `compute_expert_gradient_norms(model)` after backward on eval steps
- Log all metrics to W&B with `moe_dynamics/` prefix

**Pass criteria**:
- `pytest tests/test_moe_dynamics.py` — unit tests for collector on toy model
- Collector adds <2% wall-clock overhead (measured on 100-step anchor run)
- All metrics appear in W&B panel after a 10-step smoke test

#### Day 0: W&B Project Organization

Set up structured W&B project before any runs:
```
nanoseek-research/
├── hp-search/          # 12 HP grid runs (group by mlr×elr)
├── anchor-full/        # Best HP, full token budget
├── 500m-transfer/      # muP validation
├── stability/          # Bad batch, aux-loss, routing sensitivity
├── 1b-flagship/        # Final model
└── data-mixture/       # Domain ratio experiments (Phase 2.5)
```

Tag every run with: `scale`, `matrix_lr`, `embedding_lr`, `experiment_type`.
This enables cross-run comparison from Day 1.

#### Day 1: MTP Dynamics Extension

MTP acceptance rate is already tracked but never analyzed as a dynamics signal.
Add MTP to the dynamics framework:

```python
# In MoEDynamicsCollector:
# - mtp_acceptance_vs_step: standard tracking (already exists)
# - mtp_acceptance_per_domain: acceptance rate on code/math/science/web/books
# - mtp_x_routing_entropy: Pearson correlation between MTP acceptance and
#   mean routing entropy across layers (same eval batch)
# - mtp_x_ispec: correlation between MTP acceptance and I_spec
```

**Why this matters** (Jensen's "inference is thinking" insight): MTP measures how
well the model predicts its own future tokens — a proxy for reasoning vs memorizing.
If MTP acceptance and expert specialization (I_spec) have correlated dynamics,
that connects routing quality to inference quality. This is a publishable finding
about the relationship between routing and reasoning in MoE models.

---

### Phase 1: HP Search + First Converged Model (Days 2-6)

**Goal**: Find good hyperparameters at anchor scale and produce our first
fully-converged training run with deep instrumentation.

#### Day 2-3: Anchor HP Grid Search

```bash
# 12 runs: 4 matrix_lr × 3 embedding_lr
# ~2.5 hrs each on A6000, run 4 in parallel
for mlr in 0.005 0.01 0.02 0.04; do
  for elr in 0.1 0.3 0.5; do
    python -m nanoseek.scripts.pre_train \
        --run "hp-mlr${mlr}-elr${elr}" --scale anchor \
        --matrix-lr $mlr --embedding-lr $elr \
        --num-iterations 4000 --eval-every 200 --save-every -1 --seed 42 &
    sleep 5
  done; wait
done
```

**What to track** (pre_train.py existing + Phase 0 dynamics collector):
- `ema_val_bpb` — primary quality metric
- `train_loss` — convergence curve
- `H_load` — expert balance (must stay > 2.0 bits)
- `grad_norm` — stability signal
- `moe_dynamics/*` — all Phase 0 instrumentation (routing entropy, Gini, churn, etc.)
- `mtp_dynamics/*` — MTP acceptance, per-domain MTP, MTP×routing correlation
- Health monitor alerts

Because instrumentation exists from Phase 0, even these short HP search runs produce
dynamics data. 12 runs × 4000 steps × 20 eval points = 240 dynamics snapshots across
the HP landscape. This reveals: do routing dynamics change with learning rate? Does
higher matrix_lr cause earlier specialization? Free insight.

**Pass criteria**:
- At least 8/12 runs converge (no NaN, no H_load collapse)
- Best run achieves ema_val_bpb < 2.0 (reasonable for 55M MoE on 1.1B tokens)
- Clear winner in (matrix_lr, embedding_lr) space
- Dynamics collector ran on all 12 runs (spot-check W&B panels)

#### Day 4: Analysis + Best HP Selection

Pick the (matrix_lr, embedding_lr) that minimizes ema_val_bpb.
Document the HP landscape: which combinations diverged? Why?

**New analysis (from Phase 0 instrumentation across HP runs)**:
- HP sensitivity of routing dynamics: does matrix_lr affect I_spec trajectory?
- Diverged runs: did dynamics metrics predict divergence before loss did?
- Early warning validation: did TrainingHealthMonitor fire before visible problems?

#### Day 5-6: Instrumented Full Anchor Run

Re-run the best HP config with full token budget. All dynamics collection is already
active from Phase 0 — this run just produces a complete training trajectory.

This run produces the first full-training dataset for dynamics analysis.

---

### Phase 2: Scale Up + Stability Experiments (Days 6-12)

**Goal**: Train at 500M, validate HP transfer, run targeted stability experiments.

#### Day 6-8: nano-500M with muP Transfer

```bash
python -m nanoseek.scripts.pre_train \
    --run "500m-transfer" --scale 500m \
    --matrix-lr <best_from_anchor> --embedding-lr <best_from_anchor> \
    --eval-every 500 --save-every 2000 --device-batch-size 8 --seed 42
```

**The research question**: Does muP HP transfer work for MoE?

muP scales:
- `matrix_lr *= anchor_width / target_width` (1/width rule)
- `embedding_lr` stays constant
- `batch_size *= (target_width / anchor_width)^2` (√B rule)

If 500M converges with transferred HPs → clean validation of muP for MoE.
If it diverges → document WHICH scaling rule broke and WHY. Both outcomes
are publishable. This validates arXiv:2508.09752 at a new architecture point.

Same deep instrumentation. Compare dynamics: do experts specialize at the
same training fraction as anchor? Is routing entropy evolution similar?

#### Day 9-10: Stability Experiments at Anchor Scale

Three targeted experiments. Each ~4-6 hours. These use the existing
ablation flags in pre_train.py.

**Experiment 1: Bad Batch Recovery**
```bash
python -m nanoseek.scripts.pre_train \
    --run "stability-badbatch" --scale anchor \
    --matrix-lr <best> --embedding-lr <best> \
    --inject-bad-batch 1500 --num-iterations 4000 \
    --eval-every 100 --save-every -1 --seed 42
```

Measure:
- How quickly does loss recover? (steps to return to pre-spike EMA)
- Does routing change permanently? (H_load, I_spec before vs after)
- Do any experts die from the spike?
- What does the health monitor detect? (validate our alerting)

**Experiment 2: Aux-Loss-Free vs Classic Auxiliary Loss**
```bash
python -m nanoseek.scripts.pre_train \
    --run "stability-classic-aux" --scale anchor \
    --matrix-lr <best> --embedding-lr <best> \
    --aux-loss-type classic --num-iterations 4000 \
    --eval-every 100 --save-every -1 --seed 42
```

Measure:
- ema_val_bpb: does aux-loss-free (bias method) match classic aux loss?
- I_spec: does aux-loss-free allow MORE expert specialization?
  (This is the key claim from DeepSeek V3 — removing aux loss lets
  experts specialize more freely. Nobody has verified this independently.)
- H_load trajectory: does classic aux loss produce more uniform routing?
- Expert diversity: which method produces more diverse expert outputs?

**Experiment 3: Routing Sensitivity (if time permits)**
```bash
# Fewer experts: 16 experts, top-4, n_group=4
python -m nanoseek.scripts.pre_train \
    --run "stability-16experts" --scale anchor \
    --matrix-lr <best> --embedding-lr <best> \
    --num-experts 16 --top-k 4 --n-group 4 --topk-group 2 \
    --num-iterations 4000 --eval-every 100 --save-every -1 --seed 42
```

Compare 16-expert vs 64-expert dynamics:
- When do experts specialize with fewer experts?
- Is I_spec higher or lower with 16 experts?
- How does routing entropy evolve differently?

#### Day 11: Data↔Routing Interaction (The "Supply Chain" Experiment)

Jensen shapes hardware supply chains years before demand materializes. Our equivalent:
understand how data mixture affects expert routing BEFORE committing to 500M/1B data
mixtures. This is cheap at anchor scale (~3 hours per run) and directly determines
data strategy for the flagship.

```bash
# Baseline: default ClimbMix ratios (whatever the shards produce)
# Already measured from the full anchor run (Day 5-6)

# Experiment: code-heavy mixture (2× code fraction)
python -m nanoseek.scripts.pre_train \
    --run "data-code-heavy" --scale anchor \
    --matrix-lr <best> --embedding-lr <best> \
    --data-domain-weights "code:2.0,math:1.0,science:1.0,web:0.5,books:0.5" \
    --num-iterations 4000 --eval-every 100 --save-every -1 --seed 42

# Experiment: math-heavy mixture (2× math fraction)
python -m nanoseek.scripts.pre_train \
    --run "data-math-heavy" --scale anchor \
    --matrix-lr <best> --embedding-lr <best> \
    --data-domain-weights "code:1.0,math:2.0,science:1.0,web:0.5,books:0.5" \
    --num-iterations 4000 --eval-every 100 --save-every -1 --seed 42
```

**What to measure** (all from Phase 0 instrumentation, no new code):
- I_spec trajectory: does domain-heavy data produce faster/stronger specialization?
- Expert-domain affinity: do specific experts lock onto the boosted domain?
- Cross-domain BPB: does code-heavy training hurt math BPB? (Or help, via transfer?)
- Routing entropy: does skewed data produce less uniform routing (lower H_load)?

**Why this matters**: If domain ratio X → I_spec pattern Y, we can **engineer the
data mixture** for 1B training instead of guessing. This is the highest-leverage
cheap experiment in the plan — 6 GPU-hours to inform a 14-hour flagship run.

**Implementation note**: The `--data-domain-weights` flag needs to be added to
pre_train.py and dataloader.py. This is ~30 lines: weighted sampling from domain-tagged
shards. ClimbMix shards are already tagged by source domain in the parquet metadata.

#### Day 12-13: Analyze Phase 2 Results

Key analyses:
1. **HP transfer validation**: Did 500M converge? What's the ema_val_bpb delta
   vs what anchor scaling law predicts?
2. **Scale invariance check**: Plot I_spec vs training fraction at 55M and 441M
   on same axes. Same curve → dynamics are scale-invariant (big finding).
3. **Stability playbook**: From the 3 experiments, what works for recovery?
   What predicts collapse? Concrete recommendations.
4. **Aux-loss-free verification**: Is DeepSeek's claim correct?
5. **Data↔routing map**: How does domain mixture affect I_spec, expert affinity,
   and cross-domain BPB? Concrete recommendation for 1B data strategy.
6. **MTP×routing correlation**: From all runs, compute Pearson correlation between
   MTP acceptance rate and mean routing entropy / I_spec. If correlated (r > 0.5),
   this connects routing quality to inference quality — a novel finding.

---

### Phase 3: Flagship + Final Analysis (Days 14-20)

**Goal**: Train NanoSeek-1B, complete 3-point scaling analysis, produce deliverables.

#### Day 14-15: NanoSeek-1B Training

```bash
torchrun --nproc_per_node=8 -m nanoseek.scripts.pre_train \
    --run "nanoseek-1b" --scale 1b \
    --matrix-lr <best> --embedding-lr <best> \
    --eval-every 500 --save-every 2000 --device-batch-size 16 --seed 42
```

8×H100, ~14 hours, 22B tokens. Full Phase 0 instrumentation active.
Use data mixture informed by Day 11 data↔routing experiment.

#### Day 16-17: Complete Analysis

**1. Scaling Law Fit** (using existing scaling_law.py)
- 3 points: (55M, ema_val_bpb_55m), (441M, ema_val_bpb_441m), (1.08B, ema_val_bpb_1b)
- Fit L(N) = E + A × N^(-α)
- With only 3 points, report LOOCV error (already implemented)
- Compare α to literature values for dense transformers and MoE

**2. Training Dynamics Report** (the novel contribution)

For each scale, analyze and compare:

| Metric | What It Reveals | Comparison |
|--------|----------------|------------|
| I_spec vs training fraction | When experts specialize | Same fraction across scales → universal |
| H_load trajectory | Routing balance evolution | Early collapse at any scale? |
| Routing churn decay | How fast routing stabilizes | Exponential decay? Power law? |
| Expert gradient norm variance | Learning equality across experts | Gini coefficient over training |
| Gate logit growth | Stability signal | Linear? Exponential? Bounded? |
| MoE/MLA gradient ratio | Layer type interaction | Constant or evolving? |

**3. Expert Specialization Analysis**

Per-layer, per-scale:
- Which layers specialize most? (Expect: early layers general, later layers specialized)
- Do the same experts handle similar domains across scales?
- Is there a "specialization phase transition" (sudden I_spec jump)?

**4. MTP × Routing Correlation Analysis**

Across all runs and scales:
- Plot MTP acceptance rate vs mean I_spec at each eval point
- Compute Pearson/Spearman correlation per scale
- Per-domain MTP acceptance: does code MTP improve when code-specialized experts emerge?
- If MTP and I_spec are correlated (r > 0.5): routing quality → inference quality
- If uncorrelated: MTP and routing are independent signals (also interesting)

**5. Data↔Routing Interaction Report**

From the Day 11 experiment + all other runs:
- Domain mixture → expert affinity matrix (which experts lock onto which domains)
- Optimal data strategy recommendation for 1B training
- Cross-domain transfer effects (does boosting code help or hurt math?)

**6. Practical Monitoring Guide**

From all training runs, distill:
- What are the early warning signals for expert collapse? (Lead time in steps)
- What's the minimum monitoring cadence that catches problems?
- Which metrics are redundant? (Don't need to track everything)
- Decision tree: "If you see X, do Y"

#### Day 18-20: Figures and Write-Up

**Figure 1: Scaling Curve**
- L(N_active) for 3 scales with fit line
- Compare to dense transformer scaling law (if available)

**Figure 2: Expert Specialization Timeline**
- I_spec vs training fraction, all 3 scales on same axes
- Highlight phase transitions

**Figure 3: Routing Stability**
- Routing entropy per layer over training (heatmap: layer × step)
- One panel per scale

**Figure 4: HP Transfer Validation**
- Predicted vs actual ema_val_bpb at 500M and 1B
- Error bars from LOOCV

**Figure 5: Stability Experiment Results**
- Bad batch: loss/H_load/I_spec before, during, after spike
- Aux-loss comparison: I_spec and ema_val_bpb head-to-head

**Figure 6: MTP × Routing Dynamics**
- MTP acceptance rate vs I_spec, scatter plot, all scales colored differently
- Per-domain MTP acceptance rate over training (lines per domain)
- Correlation coefficient annotated

**Figure 7: Data↔Routing Interaction**
- Expert-domain affinity heatmap for default vs code-heavy vs math-heavy mixtures
- I_spec trajectory comparison across data mixtures (same axes)
- Cross-domain BPB impact (bar chart: does boosting one domain hurt another?)

**Figure 8: Expert Health Dashboard**
- The monitoring view: what to watch during MoE training
- Annotated with "healthy", "warning", "critical" zones

---

## What We Explicitly Don't Do

| Dropped | Reason |
|---------|--------|
| KDA / hybrid linear attention | Not implemented, weeks of engineering, unvalidated < 3B. Future work. |
| QK-Clip | Consensus: unnecessary below 10B. Even K2 paper says it self-deactivates. |
| 128-expert ablation | Interesting but doubles memory and adds variables. Month 2 work. |
| Nano-150M intermediate scale | 3 points suffice for power law. Adding cost without critical info. |
| Cross-architecture scaling laws | We have 1 architecture. Do it right. |
| RL post-training | Requires a trained model first. Month 2 after this plan completes. |
| Scaling to 3B/7B | Budget and time constrained. The science works at 1B. |
| Benchmark evaluation (MMLU, etc.) | At 1B, absolute numbers are meaningless. Relative dynamics are the value. |

---

## Compute Budget

| Run | Scale | Time | Hardware | Cost |
|-----|-------|------|----------|------|
| Phase 0 smoke test (instrumentation) | 55M | ~0.5 hrs | A6000 | ~$1 |
| HP grid (12 runs) | 55M | ~30 hrs total | A6000 | ~$60 |
| Best HP full run | 55M | ~6 hrs | A6000 | ~$12 |
| 500M transfer | 441M | ~14 hrs | A6000 | ~$28 |
| Stability exp ×3 | 55M | ~15 hrs total | A6000 | ~$30 |
| Data mixture exp ×2 | 55M | ~6 hrs total | A6000 | ~$12 |
| NanoSeek-1B | 1.08B | ~14 hrs | 8×H100 | ~$350 |
| **Total** | | **~86 GPU-hrs** | | **~$493** |

Conservative buffer for reruns: **~$650 total** (20 days, not 18).

---

## New Code Required

The plan requires minimal new code. Everything is additive to the existing
training loop (no architecture changes, no new modules).

### 1. MoE Dynamics Collector (~150 lines) — Phase 0, Day 0

```python
# nanoseek/eval/moe_dynamics.py
class MoEDynamicsCollector:
    """Collects per-step MoE routing dynamics for training analysis.

    Attaches lightweight hooks to Gate modules. Computes:
    - Per-layer routing entropy
    - Per-layer expert utilization Gini coefficient
    - Routing churn (token-level routing change vs previous step)
    - Expert output diversity (pairwise cosine distance)
    - Gate logit statistics (max, std, distribution moments)
    - MTP × routing correlation (acceptance rate vs entropy/I_spec)
    """
```

### 2. Expert Gradient Tracker (~80 lines) — Phase 0, Day 0

```python
# Addition to pre_train.py eval loop
def compute_expert_gradient_norms(model):
    """Per-expert gradient L2 norm for learning equality analysis.

    Returns dict: {layer_i: {expert_j: grad_norm}} for all MoE layers.
    Called after backward(), before optimizer.step().
    """
```

### 3. MTP Dynamics Extension (~50 lines) — Phase 0, Day 1

```python
# Addition to MoEDynamicsCollector or standalone in eval/mtp_dynamics.py
def compute_mtp_domain_acceptance(model, eval_batches, domain_labels):
    """MTP acceptance rate broken down by domain (code/math/science/web/books).

    Also computes:
    - Pearson correlation: MTP acceptance vs mean routing entropy
    - Pearson correlation: MTP acceptance vs I_spec
    """
```

### 4. Data Domain Weighting (~30 lines) — Phase 2, Day 11

```python
# Addition to pre_train.py CLI + dataloader.py
# --data-domain-weights "code:2.0,math:1.0,..." flag
# Weighted sampling from domain-tagged parquet shards
# ClimbMix shards already tagged by source domain in metadata
```

### 5. Analysis Notebook (~250 lines)

```python
# notebooks/moe_dynamics_analysis.py
# Pulls W&B data, produces all 8 figures
# Scaling law fit, dynamics comparison, stability analysis,
# MTP correlation, data↔routing interaction
```

**Total new code: ~560 lines.** No new dependencies. No architecture changes.
~260 lines are Phase 0 (must exist before first training run).

---

## Deliverables

When this plan completes, we have:

### Models
- NanoSeek-55M (EMA weights, best HP config)
- NanoSeek-441M (EMA weights, muP-transferred HPs)
- NanoSeek-1.08B (EMA weights, flagship)
- All with reproducible configs and complete W&B logs

### Research Artifacts
- **HP Transfer Report**: Does muP work for MoE? Which rules hold, which break?
  Includes HP sensitivity of routing dynamics (from 12-run grid).
- **Training Dynamics Report**: Expert specialization timeline, routing stability,
  scale-invariance analysis, MTP×routing correlation — the deepest public analysis
  of MoE training dynamics
- **Stability Playbook**: Bad batch recovery, aux-loss-free vs classic, early
  warning signals, practical monitoring recommendations
- **Data↔Routing Report**: How domain mixture affects expert specialization, expert
  affinity mapping, concrete data strategy recommendation for large-scale training
- **Scaling Law Fit**: L(N_active) from 3 points with LOOCV validation

### Engineering Artifacts
- Instrumented MoE training pipeline (reusable at any scale)
- MoE dynamics collector (hook-based, zero overhead when disabled)
- Training health monitor (already built, validated by real runs)
- Analysis pipeline (W&B → figures, reproducible)

### Figures (8 paper-quality)
1. Scaling curve L(N) with fit
2. Expert specialization timeline (I_spec vs training fraction, 3 scales)
3. Routing stability heatmap (layer × step)
4. HP transfer validation
5. Stability experiment comparison
6. MTP × routing dynamics (acceptance vs I_spec, per-domain MTP)
7. Data↔routing interaction (expert-domain affinity, mixture comparison)
8. Expert health monitoring dashboard

---

## Success Criteria

| Criterion | Threshold | Notes |
|-----------|-----------|-------|
| Anchor HP search completes | ≥8/12 runs converge | If <8, training infra has bugs |
| Best anchor ema_val_bpb | < 2.0 | Reasonable for 55M MoE on 1.1B tokens |
| 500M trains with transferred HPs | No divergence, ema_val_bpb < 1.5 | muP transfer works |
| 1B trains to completion | 22B tokens, no restart needed | Production pipeline quality |
| H_load > 2.0 bits in all runs | All runs, all steps | No expert collapse |
| I_spec increases over training | Monotonic at all 3 scales | Experts are specializing |
| Scale-invariant dynamics | I_spec curves overlap when plotted vs fraction | Big finding if true |
| MTP×routing correlation measurable | Pearson r > 0.3 between MTP acceptance and I_spec | Novel connection between routing and inference |
| Data mixture affects routing | I_spec trajectory differs between default/code-heavy/math-heavy | Validates data↔routing interaction |
| Stability experiments produce clear signal | Bad batch recovery, aux-loss comparison measurable | If no signal, experiments were too weak |
| Phase 0 instrumentation ready before training | All dynamics metrics in W&B from first HP run | Infrastructure-first principle |

---

## Risk Mitigation

| Risk | Probability | Mitigation |
|------|-------------|------------|
| HP search finds no good config | Low | Widen search range; try weight_decay axis |
| muP transfer fails at 500M | Medium | Document failure mode (valuable). Re-tune at 500M. Budget 1 extra day. |
| Expert collapse in early training | Medium | Already have health monitor. Reduce matrix_lr, increase gamma. |
| A6000 OOM at 500M | Low | Reduce device_batch_size to 2. Gradient checkpointing already in model. |
| 1B training diverges on H100 | Low | Start from 500M checkpoint if needed. Have checkpoint resume. |
| Dynamics instrumentation adds overhead | Low | Gate hooks are <1% overhead. Gradient norms computed from existing tensors. |
| Data domain weighting not in dataloader | Low | ~30 lines. ClimbMix parquet has domain tags. Fallback: manual shard selection. |
| MTP×routing correlation is near zero | Medium | Still publishable as a null result. Proves MTP and routing are independent signals. |

---

## What Comes After (Month 2+, Not This Plan)

If this plan succeeds, natural next steps (in priority order):

1. **Deep data mixture optimization** — use anchor model as proxy, test 10+ mixtures
   (Phase 2.5 gives us initial signal; Month 2 does systematic sweep)
2. **High-sparsity MoE ablation** (128 experts) — config change only, cheap to test.
   Compare dynamics: do 128 experts specialize differently than 64?
3. **RL post-training** (GRPO) — single-stage on best 500M model.
   Key question: does RL disrupt expert specialization patterns?
   Measure I_spec before/after RL to answer.
4. **KDA hybrid attention** — if dynamics report suggests attention is the bottleneck
5. **Scaling to 3B** — if 1B produces clean results worth extending

These are earned by completing the baseline, not planned in advance.

---

## Connection to Anthropic Pre-Training RE Role

| JD Requirement | What This Plan Demonstrates |
|---|---|
| "Model architecture research" | MLA + MoE at 3 scales with deep dynamics analysis |
| "Algorithms, optimizers" | MuonAdamW + muP transfer validation for MoE |
| "Design, run, analyze experiments" | HP search + stability experiments + dynamics analysis |
| "Scale training infrastructure" | Single GPU → 8×H100, checkpoint resume, health monitoring |
| "Dev tooling" | MoE dynamics collector, health monitor, analysis pipeline |
| "Optimizing throughput" | torch.compile, gradient checkpointing, MFU tracking |
| "Entire stack" | Low-level (gate hooks, gradient tracking) → high-level (scaling law, dynamics report) |

The difference between this and a typical small-scale project: **we don't just train
a model — we instrument it deeply enough to produce insights about MoE training
that transfer to any scale.** The model is the means. The understanding is the end.

---

## Engineering Philosophy (What Shaped This Plan)

Three principles drive every decision:

1. **Instrumentation is infrastructure, not afterthought.** Phase 0 exists because
   every training run should produce dynamics data. Even throwaway HP search runs
   become research data when the collector is ready on Day 0. (Jensen: "NVIDIA builds
   monitoring into the rack, not on top of it.")

2. **The unit of value is insight per token trained.** A 55M anchor run that reveals
   how data mixture affects expert affinity is worth more than a 1B run that just
   produces a checkpoint. Every phase extracts maximum information per GPU-hour.

3. **Cheap experiments inform expensive ones.** Data↔routing interaction at anchor
   scale (6 GPU-hours) determines data strategy for the 1B flagship (14 hours on
   8×H100). MTP×routing correlation at anchor scale tells us whether to instrument
   MTP more deeply at 1B. We never commit expensive compute without cheap validation.
