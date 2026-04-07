# Pretraining Execution Backlog
## NanoSeek v4 Launch Backlog

**Version**: 2026-04-04  
**Status**: Execution backlog derived from the v4 masterplan and preregistration.  
**Goal**: Convert the dense-first hybrid-attention program into a launchable sequence tied to the current repo state.

---

## 0. Current State

### What is already real
- MLA model path exists and can instantiate.
- Training loop, dataloader, and checkpoint manager exist.
- KDA ablation scaffolding exists in configs, tests, and `pre_train.py`.
- Atomic checkpoint writes and resumable dataloader state are partially implemented.

### What is not launch-ready
- `nanoseek/nanoseek/kda.py` is still a stub with `NotImplementedError`.
- `nanoseek/nanoseek/config.py` does not define `get_nanokda_anchor_config` or `get_nanokda_ablation_config`, even though `pre_train.py` and `validate_kda_setup.py` expect them.
- There is no production in-tree `GatedDeltaNet` implementation or `--arch` path.
- The current KDA ablation config still reflects the old `MLA vs KDA+NoPE vs KDA+RoPE` story, not the new dense-first confirmatory design.
- The repo still defaults to a MoE-centric training path, while the v4 plan is dense-first.

### Verified check
Command run:

```bash
python3 -u -m nanoseek.scripts.validate_kda_setup
```

Observed failures:
- config import failure for missing `get_nanokda_anchor_config`
- `KDAAttention` not implemented

This means the hybrid study is not launchable yet.

---

## 1. Stop List

Do not do these before the confirmatory core is ready:

- Do not launch the old MoE-first KDA ablation as if it were the v4 study.
- Do not start GRPO or reasoning post-training work.
- Do not add sparse transfer or alignment-data ramps before a dense winner exists.
- Do not treat `nanokda` config scaffolding as evidence that KDA is implemented.
- Do not benchmark throughput across arms until the systems contract is frozen.

---

## 2. Priority Order

### P0: Launch Blockers
1. Make the dense-first study runnable in code.
2. Make at least one hybrid arm implementation-ready.
3. Freeze the systems and eval contract in executable artifacts.

### P1: Confirmatory Launch
4. Run the dense anchor study.
5. Run dense validation.
6. Run the long-context extension probe.

### P2: Follow-On Studies
7. Sparse transfer check.
8. Late-stage data intervention.

---

## 3. P0 Backlog

### P0.1 Dense-first config family
**Why**: The v4 plan is dense-first, but the repo still assumes MoE as the primary backbone.

**Files**:
- `nanoseek/nanoseek/config.py`
- `nanoseek/scripts/pre_train.py`
- `nanoseek/scaling_law_lab/configs/kda_ablations.yaml`

**Tasks**:
- Add dense config factories for anchor and validation scales.
- Make dense mode explicit instead of implied.
- Add architecture names for the v4 study:
  - `mla-dense`
  - `kda-mla-dense`
  - `gdn-mla-dense`
- Update launch configs so they reflect the dense-first 3-arm study, not `NoPE` vs `RoPE`.

**Implementation note**:
- The lowest-friction dense path is to reuse the existing model and set `config.moe.first_k_dense_replace = config.num_layers`, so every decoder layer uses the dense SwiGLU branch already present in `model.py`.

**Acceptance**:
- `pre_train.py --arch mla-dense` builds a dense model without MoE dispatch.
- Anchor and validation config factories exist and round-trip.
- Old KDA ablation YAML no longer defines the wrong confirmatory study.

### P0.2 KDA reference path
**Why**: The KDA arm is still paperware.

**Files**:
- `nanoseek/nanoseek/kda.py`
- `nanoseek/tests/test_kda.py`
- `nanoseek/scripts/validate_kda_setup.py`

**Tasks**:
- Implement `KDAAttention.__init__`.
- Implement `KDAAttention.forward`.
- Implement `kda_kernel_reference`.
- Implement kernel dispatch fallback logic in `kda_kernel`.
- Make `validate_kda_setup.py` assert real reference-path correctness, not just importability.

**Acceptance**:
- `python3 -u -m nanoseek.scripts.validate_kda_setup` passes the KDA module section.
- `test_kda.py` covers:
  - output shape
  - causal behavior
  - recurrent vs chunk equivalence
  - state continuity
  - model integration

### P0.3 GatedDeltaNet integration
**Why**: The v4 confirmatory plan is 3-arm. Right now there is no in-tree GatedDeltaNet arm.

**Files**:
- new module, likely `nanoseek/nanoseek/gated_deltanet.py`
- `nanoseek/nanoseek/model.py`
- `nanoseek/nanoseek/config.py`
- `nanoseek/scripts/pre_train.py`
- new tests, likely `nanoseek/tests/test_gated_deltanet.py`

**Tasks**:
- Add a reference-correct GatedDeltaNet module with the same attention interface used by decoder layers.
- Add an `attention_type` path in the model and arch/config plumbing.
- Add a validation script for GatedDeltaNet readiness parallel to KDA.

**Acceptance**:
- `gdn-mla-dense` model instantiates.
- Reference forward pass works on CPU.
- Streaming and full-sequence outputs agree within tolerance on a fixed test.

### P0.4 Implementation-readiness gate
**Why**: The docs are now honest about launch readiness; the code needs the same gate.

**Files**:
- `nanoseek/scripts/validate_kda_setup.py`
- new readiness script, likely `nanoseek/scripts/validate_hybrid_arms.py`
- `nanoseek/scaling_law_lab/launch_sweep.py`

**Tasks**:
- Replace the old KDA-only readiness check with an arm-level gate:
  - baseline dense MLA builds
  - KDA arm builds and passes reference tests
  - GatedDeltaNet arm builds and passes reference tests
- Make launch tooling fail closed if an arm does not pass readiness.

**Acceptance**:
- Launch scripts refuse to start a study with unready arms.
- If only one hybrid arm is ready, the run metadata records the missing arm as a readiness failure, not a scientific loss.

### P0.5 Systems contract artifact
**Why**: Throughput claims are confounded unless the systems contract is frozen in code and emitted per run.

**Files**:
- `nanoseek/scripts/pre_train.py`
- `nanoseek/nanoseek/report.py`
- possibly new artifact helper, e.g. `nanoseek/nanoseek/run_manifest.py`

**Tasks**:
- Emit a `systems_contract.json` artifact per run containing:
  - GPU SKU and count
  - world size
  - precision mode
  - activation checkpointing setting
  - compile setting
  - dataloader worker/prefetch settings
  - warmup and benchmark windows
- Emit measured:
  - `tokens/sec`
  - `tokens/sec/GPU`
  - `step_time_ms`
  - MFU
  - peak HBM
  - comm fraction

**Acceptance**:
- Every run directory contains the same machine-readable systems contract.
- Benchmark numbers are collected after a fixed warmup, not ad hoc.

### P0.6 Replay and restart-equivalence proof
**Why**: Current checkpointing is better than average, but the v4 standard requires restart equivalence, not just save/load.

**Files**:
- `nanoseek/nanoseek/checkpoint_manager.py`
- `nanoseek/nanoseek/dataloader.py`
- `nanoseek/scripts/pre_train.py`
- new test harness, likely `nanoseek/tests/test_restart_equivalence.py`

**Tasks**:
- Extend checkpoint metadata to include:
  - consumed token count
  - dataloader cursor
  - CPU RNG
  - CUDA RNG
  - grad-accum microstep
  - scheduler state
  - EMA state
- Add a fault-injection test:
  - run uninterrupted control for N steps
  - kill resumed run midstream
  - resume from checkpoint
  - compare next 100 optimizer steps within fixed loss tolerance

**Acceptance**:
- Resume preserves token order.
- Resume tracks uninterrupted control within the declared tolerance.

### P0.7 Fixed eval and contamination contract
**Why**: The docs are stricter now; the repo needs the same discipline.

**Files**:
- `nanoseek/scripts/base_eval.py`
- `nanoseek/scripts/chat_eval.py`
- `nanoseek/eval/domain_bpb.py`
- new manifest files under `eval/` or `docs/`

**Tasks**:
- Add one locked eval manifest covering:
  - named suites
  - prompt templates
  - decoding policy
  - scoring policy
  - seed aggregation
  - denylist / contamination manifest
- Make run outputs include the eval manifest hash.

**Acceptance**:
- Architecture runs and data-intervention runs use declared eval artifacts, not ad hoc prompts.

---

## 4. P1 Confirmatory Launch

### P1.1 Dense anchor study
**Goal**: Run `A0`, `A1`, `A2` at 175M if all three arms pass readiness.

**Required before launch**:
- P0.1 through P0.7 complete
- measured parity report for parameters and FLOPs/token
- fixed systems contract emitted

**Outputs**:
- per-seed `ema_val_bpb`
- loss-vs-token
- `tokens_to_target_loss`
- systems metrics
- prefill/decode harness metrics

**Kill gates**:
- irrecoverable divergence
- replay failure
- restart mismatch
- train-cost mismatch above tolerance

### P1.2 Dense validation study
**Goal**: Repeat the surviving dense arms at 410M under the same contract.

**Rules**:
- do not prune for convenience and still call it confirmatory
- carry forward the same eval and systems artifacts

### P1.3 Long-context extension probe
**Goal**: Compare `A0` vs best hybrid under one frozen context-extension recipe.

**Rule**:
- label as exploratory unless separately preregistered

---

## 5. P2 Follow-On Studies

### P2.1 Sparse transfer check
**Goal**: Test whether the dense winner survives transfer to a sparse backbone.

**Files**:
- `nanoseek/nanoseek/config.py`
- `nanoseek/scripts/pre_train.py`
- possibly sparse-specific launch configs

**Rule**:
- this is a new study, not an extension of confirmatory force from dense runs

### P2.2 Late-stage data intervention
**Goal**: Run one token-conserving late-stage replacement on the architecture winner.

**Files**:
- `nanoseek/nanoseek/data_curation/mixture.py`
- `nanoseek/nanoseek/data_curation/quality_classifier.py`
- separate preregistration doc

**Rule**:
- pick one intervention class only
- keep total tokens and steps fixed

---

## 6. Recommended Execution Order

### Week 1
1. Implement dense config family.
2. Replace old KDA ablation YAML with v4 dense-first launch configs.
3. Add implementation-readiness gate.

### Week 2
4. Implement KDA reference path and tests.
5. Add GatedDeltaNet module and tests.

### Week 3
6. Freeze systems contract artifacts.
7. Add restart-equivalence and replay tests.
8. Lock eval manifest and contamination manifest.

### Week 4
9. Run dense anchor study.
10. Review losses, sample efficiency, and systems parity.

### Week 5
11. Run dense validation.
12. Run long-context probe.

### Week 6
13. Decide whether sparse transfer is worth doing.
14. If yes, preregister and launch it.
15. Separately preregister the late-stage data intervention.

---

## 7. Honest Launch Decision

### Launch now?
No.

### Minimum condition to launch the study honestly
- baseline dense MLA arm ready
- at least one hybrid arm ready
- systems contract emitted
- replay and restart-equivalence proven
- eval manifest frozen

### Strong condition to launch the intended 3-arm confirmatory study
- `mla-dense`
- `kda-mla-dense`
- `gdn-mla-dense`

all pass implementation-readiness and parity gates.

If only one hybrid arm is ready, run the 2-arm study and record the missing arm as a readiness failure.

---

## 8. Deliverables to Produce Next

1. Dense config factories in code.
2. Hybrid arm readiness script.
3. Updated dense-first launch YAML.
4. Systems contract artifact emission.
5. Restart-equivalence test harness.
6. Eval denylist / contamination manifest.

That is the shortest path from v4 theory to runnable science.
