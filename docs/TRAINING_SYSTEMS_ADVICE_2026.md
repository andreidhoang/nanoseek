# NanoSeek Training Systems Review (2026) — DDP/FSDP, Scaling, Reliability

## Purpose

This document captures concrete engineering advice for NanoSeek’s training stack based on first-principles analysis of the current codebase and 2026 frontier-lab practices.

It is meant to answer:
1. What distributed method NanoSeek currently uses (DDP vs FSDP) and why.
2. Whether current model sizing and scaling-law assumptions are coherent.
3. What is still missing for robust large-run research operations (resume, crash recovery, checkpoint safety, hardware specs, observability).

---

## 1) Current distributed training method: DDP (not FSDP)

### What is implemented now

NanoSeek training uses **PyTorch DDP** when launched with `torchrun`:
- Distributed world initialized via `dist.init_process_group(backend='nccl')`
- Per-rank device assignment from `LOCAL_RANK`
- Model wrapped with `DDP(model, device_ids=[ddp_local_rank])`

### Why this was chosen

The current script comments explicitly argue that for this model scale, DDP is sufficient and simpler than FSDP, with lower integration complexity and no parameter-shard orchestration burden.

### Important nuance

Although the model wrapper is DDP, NanoSeek uses distributed custom optimizers (`DistAdamW`, `DistMuon`) with reduce-scatter / gather style behavior, so memory/communication behavior is not naive "replicated Adam everywhere".

---

## 2) Model sizing and scaling-law alignment

## 2.1 Current sizing assumptions are coherent

The default NanoSeek-1B config is documented as:
- ~1.08B active parameters
- ~4.75B total parameters (MoE expansion)
- 64 routed experts, 8 active per token
- Chinchilla-style target token budget (~20x active params)

This is internally consistent with the config narrative and intended educational/research scope.

## 2.2 Why this is a good operating point

From first principles, this size is a useful compromise:
- Large enough to expose real MoE routing and sparse-attention dynamics.
- Small enough to iterate on architecture + training recipes without frontier-scale infrastructure.
- Supports transfer learning of engineering insights (routing stability, indexer behavior, multiphase scheduling).

---

## 3) MoE / DSA experiment readiness: what is strong

NanoSeek already has a strong base:
- MLA + MoE + MTP + DSA integrated in one training stack.
- Multi-phase training hooks (dense MLA -> sparse DSA).
- Checkpoint manager that stores model + per-rank optimizer + metadata.
- Streaming dataloader with resumable state dictionary support.

This is enough to run meaningful architecture and systems ablations.

---

## 4) Gaps to close for top-tier 2026 research reliability

## P0 — Reliability and resume ergonomics

1. **Auto-resume latest checkpoint**
   - Current flow requires explicit `resume_from_step`.
   - Add `--resume=latest` path that uses checkpoint discovery.

2. **Atomic checkpoint writes + integrity checks**
   - Write to temp files then rename.
   - Add checksums and validation before marking checkpoint "ready".

3. **Strict resume invariants**
   - Validate model config hash, tokenizer hash, data split version, and optimizer schema before resume.

## P0 — Crash tolerance and distributed robustness

4. **Recovery smoke test in CI**
   - Start short distributed run, kill a worker, restart, verify step continuity and metric drift bounds.

5. **Elastic distributed support (optional)**
   - Add torch elastic launcher path for preemptible environments.

## P1 — Checkpoint lifecycle management

6. **Retention policy**
   - `latest-k`, `best-val`, and `phase-boundary` checkpoint classes.

7. **Promotion rules**
   - Promote only if no regression gates fail (loss, stability, routing health).

## P1 — Observability and experiment governance

8. **MoE routing telemetry**
   - Expert utilization entropy
   - Token-to-expert Gini
   - Overflow/drop rates
   - Per-expert gradient norms

9. **Run manifest / reproducibility card**
   - Git SHA, command line, CUDA/PyTorch versions, world size, tokenizer checksum, dataset snapshot.

10. **Phase-transition diagnostics**
    - Explicit dashboard around dense->sparse switch: loss jump, throughput change, memory change, routing behavior.

## P2 — Hardware profile clarity

11. **Published “known-good” hardware matrices**
    - 1xGPU dev profile
    - 8xH100 profile
    - expected throughput ranges
    - safe batch-size defaults and OOM fallback guidance

---

## 5) Recommended strategy decision (project-level)

Keep NanoSeek’s **architecture-native training path** as the primary research kernel.

Then add a second lane for external checkpoints only if needed for rapid post-training experimentation.

Rationale:
- Primary lane preserves first-principles architecture learning value.
- Secondary lane accelerates post-training/inference-time-compute research without distorting core design goals.

---

## 6) Suggested 4-week execution plan

### Week 1
- Implement `--resume=latest`
- Add checkpoint atomic write + checksum metadata

### Week 2
- Add run manifest output and config/data/tokenizer hashes
- Add checkpoint integrity preflight checks

### Week 3
- Add MoE routing telemetry + phase transition dashboard counters
- Add retention/promotion checkpoint policy

### Week 4
- Add distributed crash-recovery smoke test
- Publish hardware profile table with tested settings and throughput

---

## 7) Definition of done

A training stack is “research-grade reliable” when:
1. Crash/resume is one-flag and deterministic enough for long runs.
2. Checkpoints are integrity-validated and policy-managed.
3. Distributed behavior is observable and diagnosable.
4. Results are reproducible with complete run metadata.
5. Phase transitions (dense->sparse) are governed by explicit regression gates.

