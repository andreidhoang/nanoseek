

# Karpathy Method × NanoSeek: Deep Analysis Report

Both agents completed exhaustive analysis. Here's the synthesized result — structured as Karpathy would think about it.

---

## I. THE 7 COMMANDMENTS (from 320+ experiments)

1. **Wall clock to capability is THE metric** — not per-step loss, not MFU, not micro-benchmarks
2. **Validate at target scale** — d12 tuning actively hurt d20. Anchor scale is for muP verification only
3. **Re-run scaling laws after architecture changes** — the compute-optimal frontier shifts
4. **Every feature must earn its complexity budget** — if it doesn't move wall clock, kill it
5. **torch.compile is mandatory for FP8** — without it, FP8 is 4x *slower*
6. **Weight decay scales as 1/width²** — measured empirically, exponent ≈ 1.97
7. **Sharp cliffs exist in HP space** — x0_beta1 flat at 0.90-0.96, catastrophic at 0.98

---

## II. PRIORITY 1: Implement Immediately

| # | Technique | Expected Impact | File |
|---|-----------|----------------|------|
| 1 | **GC management** (disable after step 1, manual collect every 5000) | 1-3% wall-clock free | `pre_train.py` |
| 2 | **COMPUTE_DTYPE global** (kill any remaining autocast) | 2-5% throughput | `common.py` |
| 3 | **torch.compile dynamic=False** (fixed shapes, no recompilation) | 15-30% throughput | `pre_train.py` |

These are all adopted by nanochat, already validated, and require minimal code changes. The GC fix is especially important for MoE — 64 experts create many small tensors that trigger GC pauses.

---

## III. PRIORITY 2: Before Full Training

| # | Technique | Impact | Notes |
|---|-----------|--------|-------|
| 4 | **Logit softcap tuning** — NanoSeek uses 30, Karpathy found 20 optimal | ~1e-3 val_bpb + stability | Easy ablation: try 15, 20, 30 |
| 5 | **WD cosine schedule** (1.0 → 0.0 linear decay) | 0.5-1% val_bpb | Already have cautious WD, add schedule |
| 6 | **WD scaling law** — WD ∝ 1/width² | Correctness | Ablation (1280): ~0.08, 1B (2048): ~0.031 |
| 7 | **Auto batch size** (Power Lines: B_opt ∝ D^0.383) | 10-20% if current B is wrong | ablation→1B: B grows 1.44x |
| 8 | **Muon momentum schedule** (warmup 0.85→0.97, warmdown to 0.90) | Stability for MoE | High gradient variance in MoE benefits |
| 9 | **FP8 rebuild with compile** (tensorwise, E4M3/E5M2) | ~1.4x H100 throughput | But capability-matched speedup is only ~5% |
| 10 | **Scaling law param counting** — add `num_scaling_params()` to model | Critical for Phase 3 | Use Kaplan-style: N_active projections + lm_head |

---

## IV. DO NOT ADOPT (Negative or Inapplicable)

| Technique | Why Not |
|-----------|---------|
| **x0/resid lambdas** | DeepSeek V3 init is designed for this architecture; lambdas would interfere |
| **Value Embeddings** | MLA already provides value enhancement via compressed latent KV; nanoseek is not capacity-starved at 4.75B |
| **Sliding window (SSSL)** | MLA's 23x KV compression already solves what sliding window addresses |
| **SwiGLU → ReLU²** | nanochat found ReLU² better, but nanoseek's MoE experts are *designed* for SwiGLU |
| **Bigram hash embeddings** | Reverted at scale in nanochat |
| **Varlen attention** | 0.0002 bpb, not worth complexity |
| **No gradient clipping** | Works for Muon+dense, but MoE routing creates gradient spikes — keep clipping for nanoseek |

---

## V. EXPERIMENTAL METHODOLOGY (The Karpathy Process)

### The Progression Discipline
```
d12 (5 min)  →  d16 (intermediate)  →  d20 (target)
   ↓                    ↓                    ↓
 "does it work?"    "scale check"      "THE truth"
 Most ideas die     Catches scale-      Fine-tuned d12
 here               dependent           values ACTIVELY
                    behavior            HURT at d20
```

### For NanoSeek, this maps to:
```
anchor (768h)  →  ablation (1280h, 410M active)  →  1B (2048h, 1.08B active)
     ↓                       ↓                              ↓
  muP transfer           HP search                    graduation run
  verification           happens HERE                 (transfer, don't retune)
  only                   (500 steps, ~$7/run)
```

### The Autoresearch Loop
1. Train reference (ablation, 500 steps) → measure wall-clock + val_bpb
2. Change ONE thing
3. Measure: time to reach reference val_bpb OR val_bpb at reference time
4. Accept/reject on wall clock
5. Also track H_load + I_spec (MoE-specific — don't accept a change that collapses experts)

---

## VI. SCALING LAWS AS SCIENTIFIC INSTRUMENT

Karpathy tested 3 counting methods. **Kaplan-style wins**:

| Method | N exponent | D exponent | Ratio stability |
|--------|-----------|-----------|----------------|
| **Kaplan** (projections + lm_head) | C^0.54 | C^0.49 | **~10.5x stable** |
| Chinchilla (all params) | C^0.37 | C^0.50 | 3.0-4.0x unstable |
| Transformer-only | C^0.70 | C^0.41 | 17-8.5x very unstable |

**For NanoSeek MoE**: Use Kaplan-style counting with N_active (top-8 experts + 2 shared + attention + lm_head). Run 4-point FLOPs sweep (1e18, 2e18, 5e18, 1e19) to calibrate the tokens:params ratio for MoE.

---

## VII. THE KARPATHY PLAN FOR NANOSEEK

### Phase 1: Gate 1 (100 steps)
- Loss decreases? No NaN? No expert collapse? MFU reasonable?
- H_load > 2 bits? FIM at ~10%? EMA tracking works?

### Phase 2: HP Search (6 runs, 500 steps each, ~$42)
- Sweep: matrix_lr × embedding_lr (3×2 grid)
- **Do NOT over-tune.** The 320-experiment lesson: only `x0_beta1=0.96` mattered at target scale

### Phase 3: Autoresearch Candidates (in order)
1. torch.compile dynamic=False (largest expected win)
2. GC management
3. Softcap tuning (15 vs 20 vs 30)
4. WD cosine schedule
5. Muon momentum schedule
6. Batch size validation (current vs 2x vs 0.5x)

### Phase 4: Scaling Laws
- 4-point FLOPs sweep with Kaplan-style counting
- Determine optimal tokens:params ratio for MoE
- **Re-run after any architecture change**

### Phase 5: 1B Graduation
- Transfer HPs via muP (don't retune!)
- WD scales: WD_1b = WD_ablation × (1280/2048)² ≈ 0.031
- Batch size via Power Lines: B_1b ≈ 1.44 × B_ablation
- FP8 on H100 only (not A6000/A100)

---

## VIII. WHAT'S ALREADY GOOD IN NANOSEEK

The agents found that nanoseek's optimizer is **fully caught up** with nanochat:
- Polar Express orthogonalization
- NorMuon variance reduction  
- Cautious weight decay
- Fused kernels with 0-D CPU tensors
- Double-buffered dataloader (ahead of nanochat)
- BOS-aligned packing
- Same tokenizer + number split pattern

The main gaps are **training loop infrastructure** (GC, compile flags, scheduling) and the **experimental methodology** (wall-clock measurement discipline, autoresearch loop).

---

Want me to start implementing the Priority 1 items (GC management, COMPUTE_DTYPE, compile flags)? Or would you prefer to discuss any of these findings first?