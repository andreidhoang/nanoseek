# NanoSeek: Pre-Training Research Plan
## MoE Training Dynamics at Nano Scale
### March 2026 (revised March 25 — ablation-first, d=1280 primary scale)

---

## Thesis

Train MLA + MoE + MTP models at the **ablation scale** (d=1280, 16 layers, ~410M active / ~1.95B total)
to produce the deepest public analysis of MoE training dynamics — expert specialization,
routing stability, collapse prediction. Graduate the best config to 1B for a final flagship run.

**Why d=1280, 16 layers**: DeepSeek validated their MoE design at d=1280
(DeepSeekMoE 2B ablation: d=1280, 64 experts, ~300M active). We use 16 layers
(matching 1B) so only WIDTH varies between scales — HPs transfer cleanly.
At ~$35 per full run, we can afford 18+ experiments — enough to do real science.
At our old 1B scale ($350/run), we could afford 1.5 experiments — that's a demo, not research.

**What we produce**: A trained ablation-scale model, a reusable instrumented training pipeline,
a training dynamics report that doesn't exist anywhere in open literature, and a 1B graduation model.

---

## Why This Focus

Every lab running MoE at scale fights the same problems daily:

1. **Expert collapse** — routing degenerates, experts die silently
2. **Stability** — loss spikes, gradient explosions, mysterious divergences
3. **Routing dynamics** — when do experts specialize? what predicts collapse?
4. **Data interaction** — how does data mixture affect expert specialization?

These problems are **scale-universal**. The instrumentation and insights
transfer directly to 100B+ scale.

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
| Configs (ablation d=1280 / 1B d=2048) | Done | Correct MLA geometry, same depth (16L) |
| Ablation flags | Done | --aux-loss-type, --inject-bad-batch, etc. |
| 7 critical training bugs fixed | Done | See TRAINING_BUGS_POSTMORTEM.md |

**What's NOT done**: Zero completed training runs. Smoke test v5 ran 38 steps.

---

## Two Scales, Clear Roles

| Scale | d_model | Layers | Active Params | Total Params | Cost/Full Run | Role |
|-------|---------|--------|---------------|--------------|---------------|------|
| **ablation** | 1280 | 16 | ~410M | ~1.95B | ~$35 | **PRIMARY**: HP search, dynamics, ALL experiments |
| **1b** | 2048 | 16 | 1.08B | 4.75B | ~$350 | Graduation run (once, after experiments prove out) |

**Why only two scales**: Both share 16 layers — only width varies (1280 → 2048).
This means HPs transfer cleanly with no depth mismatch. Running stability experiments
at ablation scale ($35) instead of a separate anchor ($10) costs $25 more per experiment,
but eliminates scale-transfer assumptions and makes results directly comparable to main runs.

**Why ablation is primary**: DeepSeek's own 2B ablation used d=1280, ~300M active.
This is the smallest scale where MoE routing dynamics are proven meaningful.
At $35/run we can afford 18+ experiments vs 1.5 at 1B.

**Why keep 1B**: The graduation run proves our best HPs and techniques work at
production scale. One big run after we know what works.

---

## The Plan

### Phase 0: Build Instrumentation (Days 0-1)

**Goal**: All dynamics collection code exists and tested BEFORE any training run.

#### Day 0: MoEDynamicsCollector + Expert Gradient Tracker

```python
# nanoseek/eval/moe_dynamics.py (~150 lines)
class MoEDynamicsCollector:
    """Hook-based MoE routing dynamics for every eval step.
    - routing_entropy, expert_gini, routing_churn
    - expert_output_diversity, gate_logit_stats
    - mtp_x_routing correlation
    """

# Addition to pre_train.py (~80 lines)
def compute_expert_gradient_norms(model):
    """Per-expert gradient L2 norm for learning equality analysis."""
```

#### Day 0: W&B Project Organization

```
nanoseek-research/
├── smoke-tests/        # Gate 1 verification runs
├── hp-search/          # 6 ablation HP runs (500 steps each)
├── ablation-full/      # Full ablation training runs
├── stability/          # Bad batch, aux-loss experiments (at anchor scale)
├── data-mixture/       # Domain ratio experiments (at ablation scale)
└── 1b-graduation/      # Final 1B training
```

#### Day 1: MTP Dynamics Extension + Smoke Test

MTP per-domain acceptance, MTP×I_spec correlation.
Run 100-step smoke test at ablation scale to verify infra works.

---

### Phase 1: HP Search at Ablation Scale (Days 2-4)

**Goal**: Find optimal (matrix_lr, embedding_lr) by running 6 short ablation-scale runs.

#### Why direct search at ablation scale

Published HPs from DeepSeek V3, V2-Lite, and Kimi K2 already bracket the optimal range:
- DeepSeek V2-Lite (2.4B active, d=2048): lr=4.2e-4, beta2=0.95
- DeepSeek V3 (37B active, d=7168): lr=4e-4, beta2=0.95
- Our ablation scale sits between their 2B ablation and V2-Lite

A 6-run grid at 500 steps each costs ~$30 total. This finds the actual
ablation-scale optimum with zero transfer risk.

#### Days 2-3: 6 HP Runs at Ablation Scale (500 steps each)

```bash
# 6 runs: 3 matrix_lr × 2 embedding_lr
# ~15 min each on A6000, sequential → ~1.5 hrs total
for mlr in 0.005 0.01 0.02; do
  for elr in 0.2 0.5; do
    python -m nanoseek.scripts.pre_train \
        --run "hp-abl-mlr${mlr}-elr${elr}" --scale ablation \
        --matrix-lr $mlr --embedding-lr $elr \
        --num-iterations 500 --eval-every 100 --save-every -1 --seed 42
  done
done
```

**Cost**: 6 runs × ~$7 = ~$42

**Selection criterion**: Lowest `ema_val_bpb` at step 500.
Also check: H_load > 2.0, no NaN, grad_norm stable.

#### Day 4: Analyze + Select

Pick best (matrix_lr, embedding_lr). Check:
- Did any run diverge? (signals instability at that HP)
- Do routing dynamics differ across HPs? (preliminary dynamics signal)
- Is there a clear winner or a flat basin? (flat = robust)

---

### Phase 2: Full Ablation Training (Days 5-8)

**Goal**: Train ablation model to completion with full dynamics instrumentation.

#### Days 5-7: Full Ablation Training

```bash
python -m nanoseek.scripts.pre_train \
    --run "nanoseek-ablation-v1" --scale ablation \
    --matrix-lr <best> --embedding-lr <best> \
    --eval-every 250 --save-every 1000 --device-batch-size 16 --seed 42
```

1-4× A6000, ~10-14 hours, 8.2B tokens. Full Phase 0 instrumentation active.

**Logged at every eval step (every 250 steps)**:
- ema_val_bpb, train_loss, grad_norm
- H_load, I_spec (per layer)
- Per-layer routing entropy, Gini, churn
- Expert gradient norms (per expert per layer)
- MTP acceptance rate (overall + per domain)
- MTP×I_spec correlation
- Gate logit statistics
- Domain BPB (code/math/science/web/books)

**Pass criteria**:
- Trains to completion (8.2B tokens, no restart)
- H_load > 2.0 throughout
- I_spec increases over training
- ema_val_bpb is reasonable for 400M-class MoE

---

### Phase 3: Stability & Data Experiments (Days 9-12)

**Goal**: Run targeted experiments at ablation scale (same scale as main training).
Results are directly comparable — no scale-transfer assumptions.

#### Day 9: Bad Batch Recovery

```bash
python -m nanoseek.scripts.pre_train \
    --run "stability-badbatch" --scale ablation \
    --matrix-lr <best> --embedding-lr <best> \
    --inject-bad-batch 1500 --num-iterations 4000 \
    --eval-every 100 --save-every -1 --seed 42
```

Measure: recovery speed, routing permanence, expert death.

#### Day 10: Aux-Loss-Free vs Classic

```bash
python -m nanoseek.scripts.pre_train \
    --run "stability-classic-aux" --scale ablation \
    --matrix-lr <best> --embedding-lr <best> \
    --aux-loss-type classic --num-iterations 4000 \
    --eval-every 100 --save-every -1 --seed 42
```

Independently verify DeepSeek V3's claim: does removing aux loss
allow more expert specialization (higher I_spec)?

#### Day 11: Data↔Routing Interaction

```bash
python -m nanoseek.scripts.pre_train \
    --run "data-code-heavy" --scale ablation \
    --matrix-lr <best> --embedding-lr <best> \
    --data-domain-weights "code:2.0,math:1.0,science:1.0,web:0.5,books:0.5" \
    --num-iterations 4000 --eval-every 100 --save-every -1 --seed 42
```

Measure: I_spec trajectory, expert-domain affinity, cross-domain BPB.

#### Day 12: Analyze Stability Results

Compile stability playbook: what predicts collapse, what recovers, what doesn't.

---

### Phase 4: 1B Graduation Run (Days 13-14)

**Goal**: Train NanoSeek-1B with best HPs and monitoring from ablation experiments.

```bash
torchrun --nproc_per_node=8 -m nanoseek.scripts.pre_train \
    --run "nanoseek-1b-v1" --scale 1b \
    --matrix-lr <best> --embedding-lr <best> \
    --eval-every 500 --save-every 2000 --device-batch-size 16 --seed 42
```

8×H100, ~14 hours, 22B tokens. All instrumentation from ablation experiments active.

**Pass criteria**:
- Trains to completion (22B tokens, no restart)
- H_load > 2.0 throughout
- I_spec trajectory matches ablation-scale pattern
- ema_val_bpb is reasonable for 1B-class MoE

---

### Phase 5: Analysis + Write-Up (Days 15-17)

**Goal**: Produce the training dynamics report and figures.

#### Key Analyses

1. **Expert Specialization Timeline** — I_spec vs training fraction at ablation + 1B
2. **Routing Stability** — per-layer entropy/churn heatmaps over training
3. **Expert Gradient Equality** — Gini coefficient of per-expert grad norms
4. **MTP × Routing Correlation** — does MTP acceptance track I_spec?
5. **Stability Playbook** — bad batch recovery, aux-loss comparison
6. **Data↔Routing Map** — how domain mixture affects specialization
7. **Scale Transfer** — do ablation-scale dynamics predict 1B dynamics?

#### 7 Paper-Quality Figures

1. Expert specialization timeline (I_spec vs training fraction, ablation + 1B)
2. Routing stability heatmap (layer × step)
3. HP sensitivity (6-run landscape at ablation scale)
4. Stability experiment comparison (bad batch, aux-loss)
5. MTP × routing dynamics (acceptance vs I_spec)
6. Data↔routing interaction (expert-domain affinity)
7. Ablation→1B dynamics comparison (does the pattern transfer?)

---

## What We Explicitly Don't Do

| Dropped | Reason |
|---------|--------|
| muP proxy/anchor HP transfer | Direct ablation search is cheaper and zero transfer risk |
| 3-point scaling law fit | Only 2 scales trained. Not enough points. |
| Benchmark evaluation (MMLU etc.) | At 300M-1B, absolute numbers are meaningless |
| RL post-training | Requires trained model. Month 2. |

---

## Compute Budget

| Run | Scale | Time | Hardware | Cost |
|-----|-------|------|----------|------|
| Phase 0 smoke test | ablation | ~0.5 hrs | A6000 | ~$7 |
| HP search (6 × 500 steps) | ablation | ~1.5 hrs | A6000 | ~$42 |
| Full ablation training | ablation | ~12 hrs | A6000 | ~$35 |
| Stability exp ×2 | ablation | ~6 hrs | A6000 | ~$20 |
| Data mixture exp | ablation | ~3 hrs | A6000 | ~$10 |
| **1B Graduation** | **1B** | **~14 hrs** | **8×H100** | **~$350** |
| **Total** | | **~37 hrs** | | **~$464** |

Conservative buffer: **~$600 total** (17 days).

**Key insight**: Two scales, same depth (16L). Only width varies.
HPs transfer cleanly. 18+ experiments at ablation before one big 1B bet.

---

## New Code Required

~310 lines total. No new dependencies. No architecture changes.

1. **MoE Dynamics Collector** (~150 lines) — Phase 0, Day 0
2. **Expert Gradient Tracker** (~80 lines) — Phase 0, Day 0
3. **MTP Dynamics Extension** (~50 lines) — Phase 0, Day 1
4. **Data Domain Weighting** (~30 lines) — Phase 3, Day 11

---

## Deliverables

### Models
- NanoSeek-Ablation (EMA weights, full dynamics, 8.2B tokens)
- NanoSeek-1B (EMA weights, full dynamics, 22B tokens)
- Reproducible configs and complete W&B logs

### Research Artifacts
- **Training Dynamics Report**: Expert specialization timeline, routing stability,
  MTP×routing correlation — deepest public analysis of MoE training dynamics
- **Stability Playbook**: Bad batch recovery, aux-loss-free verification,
  early warning signals, monitoring recommendations
- **Data↔Routing Report**: How domain mixture affects expert specialization
- **HP Sensitivity**: 6-point landscape at ablation scale
- **Scale Transfer**: Do ablation dynamics predict 1B dynamics?

### Engineering Artifacts
- Instrumented MoE training pipeline (reusable at any scale)
- MoE dynamics collector (hook-based)
- Training health monitor (validated by real runs)

---

## Success Criteria

| Criterion | Threshold |
|-----------|-----------|
| HP search: ≥4/6 runs converge | If <4, training infra has bugs |
| Ablation trains to completion | 8.2B tokens, no restart needed |
| 1B trains to completion | 22B tokens, no restart needed |
| H_load > 2.0 bits in all runs | No expert collapse |
| I_spec increases over training | Experts are specializing |
| MTP×routing correlation | Pearson r > 0.3 (novel finding) |
| Data mixture affects routing | I_spec trajectory differs |
| Ablation→1B dynamics transfer | Same qualitative patterns |

---

## What Comes After (Month 2+)

1. **V2-Lite + Engram** — Load pre-trained V2-Lite (15.7B), add Engram conditional
   memory, continue-train. Highest-ROI architecture experiment.
2. **RL post-training** (GRPO) — on best 1B model
3. **Deep data mixture optimization** — 10+ mixtures at ablation scale
4. **Scaling to 3B** — if 1B produces clean results

---

## Engineering Philosophy

1. **Instrumentation is infrastructure.** Every run produces dynamics data.
2. **Cheap experiments first.** Run 20 experiments at $25 before 1 at $350.
3. **Match proven scales.** d=1280 is DeepSeek's validated ablation width.
4. **Don't transfer when you can measure directly.** $30 of direct HP search
   beats $120 of proxy search with transfer risk.
5. **Graduate the winner.** Only the 1B run that uses battle-tested HPs.
