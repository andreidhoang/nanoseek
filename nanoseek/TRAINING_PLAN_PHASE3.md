# NanoSeek Phase 3: Training & Ablation Launch Plan
## From First Principles — GPU Selection, Ablation Design, RunPod Setup
### March 2026 | Lead: Senior AI Research Engineer

---

## 0. Stage 1 Review Verdict

### What's Built (Excellent)

The codebase is **production-ready for training**. Key strengths:

1. **Architecture correctness**: MLA (23× KV compression), MoE (64 experts, top-8 sigmoid routing), MTP (λ=0.3→0.1 schedule) — all 7 sections implemented with 115+ passing tests. The DeepSeek V3.2 paper spec is faithfully preserved.

2. **Training infra is complete for Phase 1**:
   - EMA tracking (decay=0.9999, every 10 steps) — RULE 1 enforced
   - Batch warmup (1/5→1× over first 10%) — V3 methodology
   - FIM 10% PSM from token 1 — RULE 6 enforced
   - muP transfer: √(B/B_ref) × (w_ref/w) for hidden weights, T_epoch weight decay scaling
   - MuonAdamW + DistMuonAdamW with Polar Express orthogonalization + NorMuon variance reduction
   - Checkpoint resume with full state (model + optimizer + EMA + dataloader position)
   - Config validation catches CLAUDE.md rule violations before any GPU hours burn

3. **Eval wiring is ready**: I_spec, H_load, MTP acceptance rate, dead experts, domain BPB — all hooked into the training loop and logged to W&B.

4. **Optimizer is state-of-the-art**: Fused Muon with Polar Express (not Newton-Schulz), NorMuon variance reduction, cautious weight decay mask. This is 2026-correct, not 2024-correct.

### What Needs Attention Before Launch

| Issue | Severity | Action |
|-------|----------|--------|
| Gate 1 manual checks not yet run on GPU | **Blocker** | First 100-step smoke test on RunPod |
| No `scaling_law_lab/` directory or configs yet | Medium | Create sweep configs before Series A |
| Sections 8-9 (Indexer + DSA) empty | Low (Phase 2 only) | Not needed for 4K training |
| `test_moe_standalone.py` skipped | Low | API rewrite needed, not blocking |
| Data pipeline untested on real Parquet data | Medium | Validate dataloader on actual training data |

---

## 1. GPU Requirements — From First Principles

### 1.1 The Physics: Why These GPUs, Not Others

**The fundamental constraint for MoE training is memory, not compute.**

For a dense model, memory ≈ `16 × N_params` bytes (2B model + 2B grad + 12B optimizer states in mixed precision). But MoE breaks this because **all experts must be resident** even though only top-k fire per token. This means:

```
Memory = 16 × N_total (not N_active)
```

For NanoSeek-1B: `16 × 4.75B = 76 GB` — this is why we need 80GB GPUs.

Activations add to this. Per-layer activation memory for a transformer:

```
A_layer ≈ 2 × B × T × d × sizeof(dtype)    [attention QKV + FFN intermediates]
```

For MoE layers, the expert intermediates are only computed for top-k, but the dispatch/combine buffers and router logits add overhead. At 4K context, batch 16, 2048 hidden:

```
A ≈ 16_layers × 2 × 16 × 4096 × 2048 × 2 bytes ≈ 8.6 GB
```

**Total for 1B training on single GPU**: ~76 + 8.6 + ~4 (misc) ≈ **~89 GB** → requires multi-GPU or gradient checkpointing.

### 1.2 Per-Scale Memory & Compute Budget

| Scale | N_active | N_total | Model (bf16) | Optimizer | Activations | Total | Min GPU | Tokens | FLOPs (6ND) | Time (1×H100, 47% MFU) |
|-------|----------|---------|-------------|-----------|-------------|-------|---------|--------|-------------|------------------------|
| **Anchor** | ~55M | ~240M | 0.48 GB | 2.9 GB | 0.8 GB | **~4.2 GB** | 1× any 24GB | 1.1B | 3.6e17 | **~13 min** |
| **nano-500M** | ~440M | ~1.9B | 3.8 GB | 22.8 GB | 3.2 GB | **~30 GB** | 1× H100 80GB | 8.8B | 2.3e19 | **~14 hrs** |
| **NanoSeek-1B** | 1.08B | 4.75B | 9.5 GB | 57 GB | 8.6 GB | **~75 GB** | 1× H100 80GB (tight) | 22B | 1.4e20 | **~85 hrs** (1 GPU) |

**Time derivation** (showing the math):
```
H100 peak bf16: 990 TFLOPS
MFU target: 47% (realistic for MoE with expert dispatch overhead)
Effective throughput: 990 × 0.47 = 465 TFLOPS

NanoSeek-1B on 1× H100:
  Total FLOPs = 6 × 1.08e9 × 22e9 = 1.43e20
  Time = 1.43e20 / 465e12 = 307,500 sec ≈ 85.4 hours

NanoSeek-1B on 8× H100 (with 90% scaling efficiency):
  Time = 85.4 / (8 × 0.90) ≈ 11.9 hours ≈ 12 hours
```

### 1.3 Why H100 over A100 (or Others)

| GPU | bf16 TFLOPS | HBM | $/hr (RunPod) | TFLOPS/$ | Decision |
|-----|------------|-----|---------------|----------|----------|
| A100 40GB | 312 | 40 GB | ~$1.64 | 190 | ❌ 40GB too small for 500M+ |
| A100 80GB | 312 | 80 GB | ~$2.21 | 141 | ✅ OK for anchor + 500M |
| **H100 80GB** | 990 | 80 GB | ~$3.29 | **301** | ✅ **Best TFLOPS/$ + enough memory** |
| H100 SXM | 990 | 80 GB | ~$3.89 | 254 | ✅ Slightly better interconnect |
| A6000 48GB | 155 | 48 GB | ~$0.79 | 196 | ✅ Cheapest for anchor HP search |

**First-principles reasoning:**

1. **Anchor HP search (20+ runs × ~13 min each)**: Any GPU with ≥24GB works. A6000 at $0.79/hr is optimal — total cost ~$5 for all HP runs. Even a 4090 works.

2. **Stability ablations (3 runs × ~13 min each)**: Same — any 24GB GPU. ~$1 total.

3. **nano-500M validation (1 run × ~14 hrs)**: Needs 30GB+ memory. A100 80GB works but H100 is 3.2× faster for 1.5× the price — H100 wins. ~$23 on 1×H100.

4. **NanoSeek-1B (1 run × ~12 hrs on 8×H100)**: This is the expensive run. 8×H100 at $3.29/hr × 8 = $26.32/hr × 12 hrs = **~$316**. On 8×A100 80GB it would take ~32 hrs at $17.68/hr = ~$566. **H100 is 44% cheaper for the main run.**

### 1.4 Why Not 4-bit Quantized Training / Lower Precision?

At 1B scale, the memory savings from QLoRA or 4-bit training are marginal (~2×) but the precision loss is unacceptable for a **research project measuring scaling exponents to 2% accuracy**. The scaling law fits depend on clean loss curves — any quantization noise corrupts α and δ estimates. Full bf16 mixed precision is the correct choice.

---

## 2. Ablation Design — What We Run and Why

### 2.1 The Scientific Questions (Ordered by Information Value)

Each ablation answers a specific, falsifiable question. We never run ablations "to see what happens" — each has a pre-registered hypothesis and success metric.

### Ablation Group 1: Hyperparameter Transfer (muP Validation)
**Question**: Do muP scaling rules (√B + 1/width + T_epoch) transfer correctly to MoE?

This is THE critical experiment. If muP transfer fails, we cannot trust any HP tuned at anchor scale.

| Run ID | Scale | Hidden | LR scaling | Purpose |
|--------|-------|--------|-----------|---------|
| `hp-anchor-grid` | Anchor (55M) | 480 | base | Grid search: matrix_lr × {0.005, 0.01, 0.02, 0.04}, embedding_lr × {0.1, 0.3, 0.5} |
| `hp-500m-transfer` | 500M | 1280 | muP-scaled | Verify: does best anchor LR, when muP-scaled, also be best at 500M? |
| `hp-500m-grid` | 500M | 1280 | grid | Sanity: small grid at 500M to confirm muP-scaled HP is within 0.02 BPB of local optimum |

**Pre-registered hypothesis**: muP-scaled anchor HP achieves ema_val_bpb within 0.02 of the 500M grid-searched optimum.

**If hypothesis FAILS**: The 1/width scaling for Muon (not AdamW) may need correction. Muon's orthogonalization changes the effective update scale — the standard muP 1/width rule was derived for SGD/Adam. Document which parameter group's scaling broke and propose a Muon-aware correction.

**GPU budget**: Anchor grid (12 runs × 13 min = 2.6 hrs), 500M transfer (1 × 14 hrs), 500M grid (4 runs × 14 hrs = 56 hrs). **Total: ~73 GPU-hours on H100.**

### Ablation Group 2: Training Stability (MoE-Specific)
**Question**: How do DeepSeek V3.2's stabilization techniques interact, and which are necessary vs. redundant at 1B scale?

| Run ID | Config | What's Different | Measures |
|--------|--------|-----------------|----------|
| `stab-A` (baseline) | All V3.2 techniques ON | aux-loss-free bias + seq_aux(α=1e-4) + γ_freeze=0.95 + grad_clip=1.0 | H_load, I_spec, loss curve, spike recovery |
| `stab-C` | Remove seq_aux loss | aux-loss-free bias only (no α term) | Does seq_aux actually help? Measure I_spec divergence |
| `stab-D` | Remove grad clipping | Everything else ON, grad_clip=∞ | Is grad_clip=1.0 necessary for MoE stability? |
| `stab-E` | Use standard aux loss | Replace bias-based balancing with classic aux loss (α=0.01) | aux-loss-free vs. classic: I_spec comparison |
| `stab-F` | Bad batch injection | Inject 10× normal gradient at step 1500 into Run A | Recovery dynamics: how many steps to recover? |

**Pre-registered hypotheses**:
- A vs C: Removing seq_aux drops I_spec by >0.1 nats (seq_aux helps specialization)
- A vs D: Removing grad clip causes a spike at step ~500-1000 (MoE gradient variance without clip)
- A vs E: Aux-loss-free achieves I_spec >0.1 nats higher than classic aux loss (the core V3 claim)
- F: Recovery within 50 steps (grad clip + bias reset provides robustness)

**GPU budget**: 5 runs × 13 min = ~1.1 hrs on any 24GB GPU. **Trivially cheap.**

### Ablation Group 3: Architecture Sensitivity
**Question**: Which architectural choices have the highest marginal impact on loss?

| Run ID | What's Changed | Control | Isolates |
|--------|---------------|---------|----------|
| `arch-no-mtp` | MTP disabled (λ=0) | stab-A | MTP contribution to final loss |
| `arch-no-shared` | Shared experts removed | stab-A | Shared expert contribution |
| `arch-no-mla` | Standard MHA instead of MLA | stab-A | MLA vs MHA at fixed params |
| `arch-fewer-experts` | 16 experts, top-2 (same N_active) | stab-A | Expert granularity effect |

**Pre-registered hypotheses**:
- No MTP: loss increases by ~0.05 BPB (MTP provides training signal, not just inference speed)
- No shared: loss increases by ~0.02-0.04 BPB (shared experts handle common patterns)
- No MLA: loss approximately equal (MLA saves KV cache, not training quality at 4K context)
- Fewer experts: loss increases by ~0.03 BPB (granularity effect from log(E) term)

**GPU budget**: 4 runs × 13 min = ~52 min. **Trivially cheap.**

### 2.2 Total Ablation Budget Summary

| Phase | Runs | GPU-Hours (H100) | Cost |
|-------|------|------------------|------|
| Anchor HP grid | 12 | 2.6 | $8.55 |
| Stability ablations | 5 | 1.1 | $3.62 |
| Architecture ablations | 4 | 0.9 | $2.96 |
| **nano-500M transfer** | 1 | 14 | $46.06 |
| nano-500M sanity grid | 4 | 56 | $184.24 |
| **NanoSeek-1B full** | 1 | 96 (8×12h) | $316 |
| **Total** | 27 | ~171 | **~$561** |

**Cost optimization**: Run anchor + stability + architecture ablations on A6000 ($0.79/hr) instead of H100. Saves ~$10. Run nano-500M grid on 2×H100 instead of 1× — halves wall time, same GPU-hours.

---

## 3. Ablation Execution Order (Critical Path)

```
Day 1 (4 hours) — Anchor scale on 1× A6000/A100
═══════════════════════════════════════════════
  [0:00] Gate 1 smoke test (100 steps, verify H_load > 4 bits, MTP ~50%, FIM ~10%)
  [0:15] HP grid search (12 runs parallel on data, sequential on GPU)
  [2:45] Stability ablations A, C, D, E, F
  [3:50] Architecture ablations (no-mtp, no-shared, no-mla, fewer-experts)
  [4:00] DONE — analyze results, pick best HP, verify pre-registered hypotheses

Day 2 (18 hours) — 500M validation on 1× H100
═══════════════════════════════════════════════
  [0:00] hp-500m-transfer (best anchor HP, muP-scaled)
  [14:00] Run complete — check ema_val_bpb
  [14:00] IF muP hypothesis holds: proceed to 1B
           IF fails: launch hp-500m-grid (4 runs, need another ~56 hrs)

Day 3 (14 hours) — NanoSeek-1B on 8× H100
═══════════════════════════════════════════════
  [0:00] Launch full 22B token training
  [12:00] Training complete
  [12:00] Run EMA evaluation: domain BPB, I_spec, MTP acceptance, dead experts
  [13:00] Compare predicted loss (from scaling law fit) vs actual
  [14:00] DONE — Phase 3 complete
```

**Total wall time: ~3 days. Total cost: ~$400-560.**

---

## 4. RunPod Setup — Exact Configuration

### 4.1 Pod Configuration

**Pod 1: Anchor Ablations** (Day 1)
```yaml
gpu_type: "NVIDIA A6000"       # 48GB, $0.79/hr — overkill for 55M model
gpu_count: 1
cloud_type: "COMMUNITY"        # cheapest tier
container_image: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
volume_size: 50                 # GB — for data + checkpoints
volume_mount: "/workspace"
env:
  NANOSEEK_DATA_DIR: "/workspace/data"
  WANDB_API_KEY: "<your-key>"
  PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True"
```

**Pod 2: 500M Validation** (Day 2)
```yaml
gpu_type: "NVIDIA H100 80GB HBM3"
gpu_count: 1
cloud_type: "SECURE"           # better reliability for 14hr run
container_image: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
volume_size: 100               # GB — larger model + more checkpoints
```

**Pod 3: NanoSeek-1B Full Training** (Day 3)
```yaml
gpu_type: "NVIDIA H100 80GB HBM3"
gpu_count: 8
cloud_type: "SECURE"           # MUST be secure for multi-GPU NVLink
container_image: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
volume_size: 200               # GB — 4.75B model checkpoints are large
```

### 4.2 Environment Setup Script

```bash
#!/bin/bash
# setup_runpod.sh — Run once after pod starts

set -euxo pipefail

# ── System deps ──
apt-get update && apt-get install -y git htop nvtop tmux

# ── Clone repo ──
cd /workspace
git clone <your-repo-url> nanoseek
cd nanoseek

# ── Python deps ──
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install wandb transformers datasets tokenizers numpy scipy fasttext-wheel \
            datasketch mmh3 pyarrow sentencepiece

# ── Verify GPU ──
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print(f'BF16: {torch.cuda.is_bf16_supported()}')
"

# ── Download training data ──
# Option A: HuggingFace datasets (e.g., SlimPajama, FineWeb)
python -c "
from datasets import load_dataset
# Download ~50GB of FineWeb-Edu for training
ds = load_dataset('HuggingFaceFW/fineweb-edu', split='train', streaming=False,
                  num_proc=8, cache_dir='/workspace/data/cache')
ds.to_parquet('/workspace/data/fineweb_edu/')
print(f'Downloaded {len(ds)} samples')
"

# ── W&B login ──
wandb login $WANDB_API_KEY

# ── Verify training can start ──
cd /workspace/nanoseek/nanoseek
python -m pytest tests/ -v --tb=short
echo "✅ All tests passed — ready to train"
```

### 4.3 Launch Commands

**Gate 1 Smoke Test** (MUST pass before any real training):
```bash
cd /workspace/nanoseek/nanoseek

# 100-step smoke test at anchor scale
python -m nanoseek.scripts.pre_train \
    --run "gate1-smoke" \
    --scale anchor \
    --num-iterations 100 \
    --eval-every 50 \
    --save-every 100 \
    --device-batch-size 16 \
    --seed 42

# Check W&B for:
# ✅ H_load > 4 bits at step 0 (random routing → uniform)
# ✅ MTP acceptance ~50% at step 0 (random → coin flip)
# ✅ FIM fraction ~10% in logs
# ✅ EMA checkpoint saved at step 100
# ✅ No NaN/Inf in loss
# ✅ MFU reasonable (>30% on A6000, >40% on H100)
```

**HP Grid Search** (anchor scale):
```bash
# Run grid search — 12 combinations
for mlr in 0.005 0.01 0.02 0.04; do
  for elr in 0.1 0.3 0.5; do
    python -m nanoseek.scripts.pre_train \
        --run "hp-anchor-mlr${mlr}-elr${elr}" \
        --scale anchor \
        --matrix-lr $mlr \
        --embedding-lr $elr \
        --eval-every 100 \
        --save-every -1 \
        --seed 42 &
    # Stagger launches by 5 sec to avoid OOM from concurrent init
    sleep 5
  done
  wait  # wait for batch of 3 to finish before next mlr
done
```

**Stability Ablations**:
```bash
# Run A: Full V3.2 baseline
python -m nanoseek.scripts.pre_train \
    --run "stab-A-baseline" \
    --scale anchor \
    --seed 42

# Run C: No seq_aux (requires code flag — add --no-seq-aux)
python -m nanoseek.scripts.pre_train \
    --run "stab-C-no-seq-aux" \
    --scale anchor \
    --seed 42

# Run D: No grad clipping
python -m nanoseek.scripts.pre_train \
    --run "stab-D-no-gradclip" \
    --scale anchor \
    --seed 42

# ... etc for E, F
```

**nano-500M Transfer Validation**:
```bash
# Uses muP-scaled HPs from best anchor run
python -m nanoseek.scripts.pre_train \
    --run "hp-500m-transfer" \
    --scale 500m \
    --matrix-lr <best_anchor_mlr> \
    --embedding-lr <best_anchor_elr> \
    --eval-every 500 \
    --save-every 2000 \
    --seed 42
```

**NanoSeek-1B Full Training** (8× H100):
```bash
torchrun --nproc_per_node=8 -m nanoseek.scripts.pre_train \
    --run "nanoseek-1b-v1" \
    --scale 1b \
    --matrix-lr <best_anchor_mlr> \
    --embedding-lr <best_anchor_elr> \
    --device-batch-size 16 \
    --eval-every 500 \
    --save-every 2000 \
    --seed 42
```

---

## 5. Pre-Flight Checklist (Before Spending Any Money)

### 5.1 Code Readiness

- [ ] All 115+ tests pass locally
- [ ] `pre_train.py` runs 10 steps on CPU without errors
- [ ] W&B logging produces expected keys (train_loss, ema_val_bpb, h_load, i_spec, mtp_acceptance, fim_fraction)
- [ ] Checkpoint save/load round-trips correctly (save at step 50, resume at step 50, loss matches)
- [ ] Config validation rejects gamma_freeze_ratio=0.80

### 5.2 Data Readiness

- [ ] Training data in Parquet format at NANOSEEK_DATA_DIR
- [ ] Dataloader produces sequences of correct length (4096 tokens)
- [ ] FIM transform fires at ~10% rate
- [ ] BOS token at position 0 of every sequence
- [ ] Val split (last parquet file) is separate from train

### 5.3 Ablation Readiness

- [ ] CLI flags exist for all ablation variants (--no-seq-aux, --no-grad-clip, --aux-loss-type, --no-mtp, etc.)
- [ ] Bad batch injection code ready (multiply grad by 10× at step 1500)
- [ ] W&B project "nanoseek" created with proper team access
- [ ] Pre-registered hypotheses written down BEFORE seeing any results

### 5.4 RunPod Readiness

- [ ] SSH key uploaded to RunPod account
- [ ] Billing method verified (budget: $600 max)
- [ ] Template saved for each pod config (1×A6000, 1×H100, 8×H100)
- [ ] Network volume created in preferred region for data persistence
- [ ] `setup_runpod.sh` tested on a cheap pod first

---

## 6. What "Success" Looks Like After Phase 3

### Minimum Viable Success (must achieve all):

1. **muP Transfer Validated**: Best anchor HP, when muP-scaled to 500M, achieves ema_val_bpb within 0.02 of 500M grid-searched optimum
2. **Stability Characterized**: A vs C vs D vs E I_spec comparison documented, pre-registered hypotheses confirmed or falsified
3. **NanoSeek-1B Trained**: 22B tokens, ema_val_bpb converged, no NaN/divergence, H_load > 2 bits throughout
4. **Scaling Law Point**: NanoSeek-1B actual loss matches predicted loss (from anchor + 500M fit) within 5%

### Stretch Goals:

5. **MTP Acceptance**: >75% by end of training (speculative decoding works)
6. **Expert Specialization**: I_spec > 0.4 nats (experts learned meaningful roles)
7. **MFU**: >45% on 8×H100 (well-optimized training loop)
8. **Scaling Exponents**: α ≈ 0.34 ± 0.03, δ ≈ 0.28 ± 0.03 (consistent with literature)

---

## 7. Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| muP transfer fails for Muon | 30% | High — can't trust anchor HPs | Run 500M grid as backup; derive Muon-aware correction |
| Expert collapse (H_load < 2) | 15% | High — model degenerates to dense | Monitor every 100 steps; increase seq_aux α if needed |
| RunPod preemption during 1B training | 20% | Medium — lose partial run | Checkpoint every 2000 steps; resume from checkpoint |
| Data quality issues | 10% | Medium — noisy loss curves | Validate data pipeline offline; spot-check samples |
| OOM on 8×H100 for 1B | 5% | Low — need to reduce batch size | Start with device_batch_size=8, increase if memory allows |
| MTP doesn't improve loss | 25% | Low — expected from ablation | Document and proceed; MTP value is inference speed, not training |

---

## 8. Implementation TODOs Before Launch

### Must-Do (Before Day 1):

1. **Add ablation CLI flags to `pre_train.py`**:
   - `--no-seq-aux` (disable sequence auxiliary loss)
   - `--no-grad-clip` (set grad_clip to inf)
   - `--aux-loss-type {bias,classic}` (switch balancing strategy)
   - `--no-mtp` (disable MTP, set λ=0)
   - `--no-shared-experts` (remove shared experts)
   - `--inject-bad-batch STEP` (multiply grad by 10× at specified step)

2. **Create `scaling_law_lab/configs/`** with YAML configs for all sweep points

3. **Validate data pipeline end-to-end** on a small Parquet sample

4. **Write HP grid analysis script** that reads W&B runs and picks best config

### Nice-to-Have (Can Do During Training):

5. **Automated spike detection** in W&B (alert if loss increases >2× between steps)
6. **Live MFU dashboard** in W&B
7. **Scaling law fitting script** (fit after Series A completes)

---

*This plan was designed to maximize information per GPU-dollar while maintaining scientific rigor. Every ablation has a pre-registered hypothesis. Every GPU choice is justified from memory and compute first principles. Let's train.*
