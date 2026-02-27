# 06 — Multi-Phase Training Orchestration for NanoSeek

## Principal Engineer's Implementation Guide — February 2026

**Author**: Distinguished AI Researcher & Principal Engineer, Foundation Models Division
**Scope**: Complete training orchestration — phase transitions, dynamic schedules, monitoring, and crash recovery
**Prerequisites**: `scripts/pre-train.py` (training loop), `model/config.py` (schedule parameters)

---

## 1. Problem Statement

Training NanoSeek requires orchestrating **five independent dynamic schedules** while executing a
**two-phase pipeline** that reconfigures the model mid-run. The current code in `pre-train.py`
(lines 1391–1461) handles this with inline conditionals — fragile, untestable, no recovery path:

| Gap | Risk | Severity |
|-----|------|----------|
| No transition atomicity | Crash during P1→P2 = corrupt model | **CRITICAL** |
| Schedules scattered across 5 locations | One missed update = silent drift | HIGH |
| No loss spike / gradient anomaly detection | Divergence wastes hours of compute | HIGH |
| No expert collapse detection | Dead experts reduce effective capacity | HIGH |
| No atomic checkpoint writes | Partial write on OOM/kill = lost state | MEDIUM |
| No rollback capability | Failed transition requires manual restart | HIGH |

**Cost of failure**: A 14-hour run on 8×H100 costs ~$300. A crash at hour 11 (the Phase 1→2
transition at 80%) wastes ~$240 if the checkpoint is corrupted.

---

## 2. First Principles: Why Multi-Phase Training

### Dense-Then-Sparse

Dense attention builds strong representations; sparse attention adapts them for efficiency.
Training sparse from the start fails because the Lightning Indexer has no attention patterns
to learn from. The two-phase pipeline solves this:

- **Phase 1** (80% tokens, 4K ctx): Dense MLA. Indexer trains via KL-divergence auxiliary loss.
- **Phase 2** (20% tokens, 8K ctx): Sparse DSA active. YaRN extends RoPE. Lower LR (0.33×).

### Chinchilla Ratios

For 1.08B active params, Chinchilla-optimal is 22B tokens (20×). The 80/20 split gives 17.6B
tokens for representation building and 4.4B for long-context adaptation — matching DeepSeek V3's
proven dense/sparse ratio at 14.8T scale.

### Why Dynamic MTP Weight (0.3 → 0.1 at 60%)

High λ early provides stronger gradient signal while the model learns token distributions.
Late in training, MTP predictions correlate with the main head and high λ interferes with
fine-grained loss landscape navigation. The step-function transition at 60% matches V3.

### Why Freeze Gamma at 80%

The bias rule `bias[i] -= γ * (load[i] - mean_load) / mean_load` actively redistributes tokens.
Late in training, routing has converged — continuing updates causes load oscillation. Freezing
γ→0 at 80% lets the router fine-tune on a fixed landscape.

---

## 3. Production Code

### 6a. Training Orchestrator (`scripts/training_orchestrator.py`)

```python
"""
Training Orchestrator — state machine + unified schedules + crash recovery.
Wraps pre-train.py main() without modifying it.
"""
import os, time, math, json, shutil, logging, threading
from enum import Enum, auto
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from collections import deque
from contextlib import contextmanager
import torch, torch.distributed as dist

logger = logging.getLogger("nanoseek.orchestrator")

class PhaseState(Enum):
    PHASE_1_DENSE = auto()
    TRANSITIONING = auto()
    PHASE_2_SPARSE = auto()
    ROLLED_BACK = auto()
    COMPLETED = auto()

@dataclass
class TransitionRecord:
    timestamp: float; from_phase: PhaseState; to_phase: PhaseState
    global_step: int; success: bool
    rollback_path: Optional[str] = None; error: Optional[str] = None


class ScheduleManager:
    """Single source of truth for all dynamic schedules. Stateless — derives everything from step."""

    def __init__(self, total_steps: int, warmup_ratio: float = 0.0, warmdown_ratio: float = 0.2,
                 final_lr_frac: float = 0.0, mtp_initial: float = 0.3, mtp_final: float = 0.1,
                 mtp_transition_ratio: float = 0.60, gamma: float = 0.001,
                 gamma_freeze_ratio: float = 0.80, muon_momentum_start: float = 0.85,
                 muon_momentum_end: float = 0.95, muon_warmup_steps: int = 300):
        self.total_steps = total_steps
        self.warmup_ratio = warmup_ratio; self.warmdown_ratio = warmdown_ratio
        self.final_lr_frac = final_lr_frac
        self.mtp_initial = mtp_initial; self.mtp_final = mtp_final
        self.mtp_transition_ratio = mtp_transition_ratio
        self.gamma_base = gamma; self.gamma_freeze_ratio = gamma_freeze_ratio
        self.muon_start = muon_momentum_start; self.muon_end = muon_momentum_end
        self.muon_warmup_steps = muon_warmup_steps

    def get_lr_multiplier(self, step: int) -> float:
        warmup = round(self.warmup_ratio * self.total_steps)
        warmdown = round(self.warmdown_ratio * self.total_steps)
        if step < warmup:
            return (step + 1) / warmup
        elif step <= self.total_steps - warmdown:
            return 1.0
        else:
            p = (self.total_steps - step) / warmdown
            return p + (1 - p) * self.final_lr_frac

    def get_mtp_weight(self, step: int) -> float:
        return self.mtp_initial if step < int(self.mtp_transition_ratio * self.total_steps) else self.mtp_final

    def get_gamma(self, step: int) -> float:
        return self.gamma_base if step < int(self.gamma_freeze_ratio * self.total_steps) else 0.0

    def get_muon_momentum(self, step: int) -> float:
        f = min(step / self.muon_warmup_steps, 1.0)
        return (1 - f) * self.muon_start + f * self.muon_end

    def get_all(self, step: int) -> Dict[str, float]:
        """All schedules in one call — no drift possible."""
        return {"lr_multiplier": self.get_lr_multiplier(step),
                "mtp_weight": self.get_mtp_weight(step),
                "gamma": self.get_gamma(step),
                "muon_momentum": self.get_muon_momentum(step),
                "progress": step / self.total_steps}

    def export_timeline(self, num_points: int = 500) -> List[Dict[str, float]]:
        timeline = []
        for i in range(num_points + 1):
            step = int(i * self.total_steps / num_points)
            entry = self.get_all(step); entry["step"] = step
            timeline.append(entry)
        return timeline


class _TransactionHandle:
    def __init__(self): self.applied = False
    def mark_applied(self): self.applied = True

class PhaseTransitionError(Exception):
    def __init__(self, msg, rollback_path):
        super().__init__(msg); self.rollback_path = rollback_path


class TrainingOrchestrator:
    """State machine for multi-phase training with atomic transitions and rollback."""

    def __init__(self, checkpoint_dir: Path, schedule: ScheduleManager,
                 phase1_steps: int, total_steps: int, rank: int = 0):
        self.checkpoint_dir = Path(checkpoint_dir); self.schedule = schedule
        self.phase1_steps = phase1_steps; self.total_steps = total_steps; self.rank = rank
        self.state = PhaseState.PHASE_1_DENSE; self.phase_step = 0
        self.transition_history: List[TransitionRecord] = []

    def should_transition(self, step: int) -> bool:
        return self.state == PhaseState.PHASE_1_DENSE and self.phase_step >= self.phase1_steps

    @contextmanager
    def atomic_transition(self, step, model, optimizers, config, save_fn):
        """
        Atomic phase transition protocol:
        1. Save P1 checkpoint (rollback point)
        2. Yield for caller to apply P2 config
        3. On success → PHASE_2; on failure → rollback to P1
        """
        rollback_dir = self.checkpoint_dir / "phase1_final"
        rollback_dir.mkdir(parents=True, exist_ok=True)
        self.state = PhaseState.TRANSITIONING
        record = TransitionRecord(time.time(), PhaseState.PHASE_1_DENSE,
                                  PhaseState.PHASE_2_SPARSE, step, False, str(rollback_dir))

        if self.rank == 0: logger.info(f"Phase transition: saving P1 at step {step}")
        save_fn(rollback_dir, step, model, optimizers, config)

        tx = _TransactionHandle()
        try:
            yield tx
            if not tx.applied:
                raise RuntimeError("P2 config not applied (tx.mark_applied() missing)")
            record.success = True; self.state = PhaseState.PHASE_2_SPARSE; self.phase_step = 0
            if self.rank == 0: logger.info("Phase transition succeeded")
        except Exception as e:
            record.success = False; record.error = str(e)
            self.state = PhaseState.ROLLED_BACK
            if self.rank == 0: logger.error(f"Phase transition failed: {e}")
            raise PhaseTransitionError(str(e), rollback_dir) from e
        finally:
            self.transition_history.append(record)

    def step(self): self.phase_step += 1

    def get_schedules(self, step: int) -> Dict[str, float]:
        return self.schedule.get_all(step)

    def state_dict(self) -> Dict[str, Any]:
        return {"state": self.state.name, "phase_step": self.phase_step,
                "history": [{"step": r.global_step, "success": r.success, "error": r.error}
                            for r in self.transition_history]}

    def load_state_dict(self, d: Dict[str, Any]):
        self.state = PhaseState[d["state"]]; self.phase_step = d["phase_step"]
```

### 6b. Schedule Visualizer

```python
"""Schedule Visualizer — matplotlib + W&B/TensorBoard export."""
import json
from pathlib import Path
from typing import Dict, List

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt; HAS_MPL = True
except ImportError: HAS_MPL = False


class ScheduleVisualizer:
    def __init__(self, schedule_manager, output_dir: Path):
        self.schedule = schedule_manager
        self.output_dir = Path(output_dir); self.output_dir.mkdir(parents=True, exist_ok=True)
        self._history: List[Dict[str, float]] = []

    def record(self, step: int, values: Dict[str, float]):
        self._history.append({"step": step, **values})

    def plot_timeline(self, filename="schedule_timeline.png", num_points=500):
        if not HAS_MPL:
            self._export_json(filename.replace(".png", ".json"), num_points); return

        tl = self.schedule.export_timeline(num_points)
        steps = [t["step"] for t in tl]
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        fig.suptitle("NanoSeek Training Schedule Timeline", fontsize=14, fontweight="bold")
        specs = [("lr_multiplier", "LR Multiplier", "#2196F3"),
                 ("mtp_weight", "MTP Weight (λ)", "#FF9800"),
                 ("gamma", "MoE γ", "#4CAF50"),
                 ("muon_momentum", "Muon Momentum", "#9C27B0")]
        for (key, label, color), ax in zip(specs, axes):
            ax.plot(steps, [t[key] for t in tl], color=color, lw=2, label=label)
            ax.set_ylabel(label, fontsize=9); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
            if self._history:
                ax.scatter([h["step"] for h in self._history if key in h],
                           [h[key] for h in self._history if key in h], color=color, s=4, alpha=0.5)
        axes[-1].set_xlabel("Training Step")
        plt.tight_layout(); plt.savefig(self.output_dir / filename, dpi=150); plt.close()

    def export_wandb(self, run, step, values):
        run.log({f"schedule/{k}": v for k, v in values.items()}, step=step)

    def export_tensorboard(self, writer, step, values):
        for k, v in values.items(): writer.add_scalar(f"schedule/{k}", v, step)

    def _export_json(self, fn, n):
        with open(self.output_dir / fn, "w") as f: json.dump(self.schedule.export_timeline(n), f)
```

### 6c. Enhanced Checkpoint Manager

```python
"""Atomic checkpoint writes + async checkpointing + validation."""
import os, shutil, threading, logging, math, json
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch, torch.distributed as dist

logger = logging.getLogger("nanoseek.checkpoint")


class AtomicCheckpointManager:
    """
    Atomic protocol: write to .tmp dir → os.rename to final (atomic on POSIX).
    Prevents partial checkpoints from OOM kills, SIGTERM, or NFS failures.
    """
    def __init__(self, checkpoint_dir: Path, rank=0, world_size=1, max_checkpoints=3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.rank = rank; self.world_size = world_size
        self.max_checkpoints = max_checkpoints
        self._async_thread: Optional[threading.Thread] = None
        self._async_error: Optional[Exception] = None

    def save_atomic(self, step: int, model_state: Dict, optim_states: List[Dict], metadata: Dict):
        final_dir = self.checkpoint_dir / f"step_{step:06d}"
        tmp_dir = self.checkpoint_dir / f".tmp_step_{step:06d}_rank{self.rank}"
        try:
            if tmp_dir.exists(): shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True)
            if self.rank == 0:
                torch.save(model_state, tmp_dir / "model.pt")
                with open(tmp_dir / "metadata.json", "w") as f:
                    json.dump(_safe_serialize(metadata), f, indent=2)
            torch.save(optim_states, tmp_dir / f"optim_rank{self.rank}.pt")
            if self.rank == 0:
                if final_dir.exists(): shutil.rmtree(final_dir)
                os.rename(str(tmp_dir), str(final_dir))
            else:
                final_dir.mkdir(parents=True, exist_ok=True)
                os.rename(str(tmp_dir / f"optim_rank{self.rank}.pt"),
                          str(final_dir / f"optim_rank{self.rank}.pt"))
                shutil.rmtree(tmp_dir, ignore_errors=True)
            if dist.is_initialized(): dist.barrier()
            self._cleanup_old()
            logger.info(f"Checkpoint saved atomically: step {step:06d}")
        except Exception as e:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.error(f"Checkpoint save failed at step {step}: {e}"); raise

    def save_async(self, step, model_state, optim_states, metadata):
        """Clone to CPU, then write in background thread (overlaps with training)."""
        self.wait_for_async()
        cpu_model = {k: v.cpu().clone() for k, v in model_state.items()}
        cpu_opts = [{k: (v.cpu().clone() if isinstance(v, torch.Tensor) else v)
                     for k, v in s.items()} for s in optim_states]
        self._async_thread = threading.Thread(
            target=self._worker, args=(step, cpu_model, cpu_opts, metadata), daemon=True)
        self._async_thread.start()

    def wait_for_async(self):
        if self._async_thread:
            self._async_thread.join(); self._async_thread = None
            if self._async_error:
                e = self._async_error; self._async_error = None
                raise RuntimeError(f"Async checkpoint failed: {e}")

    def validate_checkpoint(self, step, model_cls=None, model_config=None, val_batch=None):
        """Load checkpoint and optionally verify loss on a small batch."""
        p = self.checkpoint_dir / f"step_{step:06d}" / "model.pt"
        try:
            state = torch.load(p, map_location="cpu", weights_only=True)
            if val_batch is not None and model_cls:
                m = model_cls(model_config)
                m.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in state.items()})
                m.eval()
                with torch.no_grad():
                    loss = m(input_ids=val_batch[0].cpu(), labels=val_batch[1].cpu())["loss"].item()
                    if not math.isfinite(loss): return False
            return True
        except Exception as e:
            logger.error(f"Validation failed: {e}"); return False

    def _worker(self, step, ms, os_, meta):
        try: self.save_atomic(step, ms, os_, meta)
        except Exception as e: self._async_error = e

    def _cleanup_old(self):
        if self.rank != 0: return
        ckpts = sorted(self.checkpoint_dir.glob("step_*"), key=lambda p: p.name)
        while len(ckpts) > self.max_checkpoints:
            shutil.rmtree(ckpts.pop(0), ignore_errors=True)

def _safe_serialize(d):
    r = {}
    for k, v in d.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): r[k] = str(v)
        elif isinstance(v, dict): r[k] = _safe_serialize(v)
        elif isinstance(v, (int, float, str, bool, list)): r[k] = v
        else: r[k] = str(v)
    return r
```

### 6d. Training Monitor & Early Stopping

```python
"""Training Monitor — loss spike, gradient anomaly, expert collapse detection + remediation."""
import math, logging
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, List
from enum import Enum, auto
import torch

logger = logging.getLogger("nanoseek.monitor")

class AnomalyType(Enum):
    LOSS_SPIKE = auto(); GRADIENT_NAN = auto(); GRADIENT_INF = auto()
    GRADIENT_EXPLOSION = auto(); EXPERT_COLLAPSE = auto(); LOSS_DIVERGENCE = auto()

@dataclass
class Anomaly:
    step: int; anomaly_type: AnomalyType; severity: float; details: str; remediation: str

@dataclass
class MonitorConfig:
    loss_window: int = 100; loss_sigma: float = 3.0
    grad_window: int = 50; grad_spike_factor: float = 10.0
    expert_min_load_frac: float = 0.01; max_consecutive_spikes: int = 3


class TrainingMonitor:
    def __init__(self, config: Optional[MonitorConfig] = None):
        self.cfg = config or MonitorConfig()
        self._losses: deque = deque(maxlen=self.cfg.loss_window)
        self._grads: deque = deque(maxlen=self.cfg.grad_window)
        self._anomalies: List[Anomaly] = []
        self._consec_spikes = 0

    def check_loss(self, step: int, loss: float) -> Optional[Anomaly]:
        """Detect loss spikes >3σ from rolling mean, or non-finite values."""
        if not math.isfinite(loss):
            a = Anomaly(step, AnomalyType.LOSS_DIVERGENCE, 1.0,
                        f"Non-finite loss: {loss}", "rollback")
            self._anomalies.append(a); return a
        self._losses.append(loss)
        if len(self._losses) < 10: return None
        mean = sum(self._losses) / len(self._losses)
        std = math.sqrt(sum((x - mean)**2 for x in self._losses) / len(self._losses)) or 1e-8
        if loss > mean + self.cfg.loss_sigma * std:
            self._consec_spikes += 1
            rem = "reduce_lr" if self._consec_spikes >= self.cfg.max_consecutive_spikes else "log"
            a = Anomaly(step, AnomalyType.LOSS_SPIKE, min((loss-mean)/std/10, 1.0),
                        f"Loss {loss:.4f} > μ+{self.cfg.loss_sigma}σ (μ={mean:.4f}, σ={std:.4f}), "
                        f"consecutive={self._consec_spikes}", rem)
            self._anomalies.append(a); return a
        self._consec_spikes = 0; return None

    def check_gradients(self, step: int, norm: float) -> Optional[Anomaly]:
        """Detect NaN, Inf, or >10× rolling mean gradient norm."""
        if math.isnan(norm):
            a = Anomaly(step, AnomalyType.GRADIENT_NAN, 1.0, "NaN grad norm", "skip_and_reduce_lr")
            self._anomalies.append(a); return a
        if math.isinf(norm):
            a = Anomaly(step, AnomalyType.GRADIENT_INF, 1.0, "Inf grad norm", "skip_and_reduce_lr")
            self._anomalies.append(a); return a
        self._grads.append(norm)
        if len(self._grads) < 10: return None
        mean = sum(self._grads) / len(self._grads)
        if mean > 0 and norm > mean * self.cfg.grad_spike_factor:
            a = Anomaly(step, AnomalyType.GRADIENT_EXPLOSION,
                        min(norm / (mean * self.cfg.grad_spike_factor), 1.0),
                        f"Grad norm {norm:.4f} > {self.cfg.grad_spike_factor}× mean {mean:.4f}",
                        "increase_grad_clip")
            self._anomalies.append(a); return a
        return None

    def check_expert_balance(self, step: int, loads: torch.Tensor) -> Optional[Anomaly]:
        """Detect expert collapse: any expert receiving <1% of expected load."""
        expected = loads.sum() / loads.numel()
        collapsed = (loads < expected * self.cfg.expert_min_load_frac).nonzero(as_tuple=True)[0]
        if len(collapsed) > 0:
            ids = collapsed.tolist()
            a = Anomaly(step, AnomalyType.EXPERT_COLLAPSE, len(ids)/loads.numel(),
                        f"{len(ids)} experts collapsed: {ids[:10]}{'...' if len(ids)>10 else ''}",
                        "increase_gamma")
            self._anomalies.append(a); return a
        return None

    def get_remediation(self, anomaly: Anomaly) -> Dict:
        """Map anomaly → concrete action dict for the training loop."""
        actions = {
            "log": {"action": "log", "msg": anomaly.details},
            "reduce_lr": {"action": "scale_lr", "factor": 0.5, "msg": anomaly.details},
            "skip_and_reduce_lr": {"action": "skip_and_scale_lr", "factor": 0.1, "msg": anomaly.details},
            "increase_grad_clip": {"action": "scale_grad_clip", "factor": 0.5, "msg": anomaly.details},
            "increase_gamma": {"action": "set_gamma", "value": 0.01, "steps": 100, "msg": anomaly.details},
            "rollback": {"action": "rollback", "msg": anomaly.details},
        }
        return actions.get(anomaly.remediation, actions["log"])

    @property
    def anomaly_count(self): return len(self._anomalies)
    @property
    def recent(self): return self._anomalies[-10:]
```

### 6e. Comprehensive Training Config

```python
"""Ratio-based training config with scaling-law validation and serialization."""
import json, math
from dataclasses import dataclass, asdict
from typing import Dict, Any
from pathlib import Path

@dataclass
class OrchestratedTrainingConfig:
    """All schedules parameterized with ratios — portable across any token budget."""
    # Phase splits
    phase1_token_fraction: float = 0.80
    phase1_context_length: int = 4096; phase2_context_length: int = 8192
    phase2_lr_scale: float = 0.33; phase2_warmup_steps: int = 100
    # LR schedule (ratios of total steps)
    lr_warmup_ratio: float = 0.0; lr_warmdown_ratio: float = 0.20; lr_final_frac: float = 0.0
    # MTP schedule
    mtp_initial: float = 0.3; mtp_final: float = 0.1; mtp_transition_ratio: float = 0.60
    # MoE bias schedule
    gamma_initial: float = 0.001; gamma_freeze_ratio: float = 0.80
    # Muon momentum
    muon_start: float = 0.85; muon_end: float = 0.95; muon_warmup_steps: int = 300
    # Monitoring
    loss_spike_sigma: float = 3.0; grad_spike_factor: float = 10.0
    expert_min_load_frac: float = 0.01; max_consecutive_spikes: int = 3
    # Checkpointing
    ckpt_every_ratio: float = 0.05; max_ckpts: int = 3
    async_ckpt: bool = True; validate_ckpts: bool = True

    def validate_scaling(self, active_params: int, total_tokens: int) -> Dict[str, Any]:
        diag = {"warnings": [], "info": []}
        ratio = total_tokens / active_params
        if ratio < 10: diag["warnings"].append(f"Under-trained: {ratio:.1f}× (optimal ~20×)")
        elif ratio > 40: diag["warnings"].append(f"Over-trained: {ratio:.1f}×")
        else: diag["info"].append(f"Chinchilla ratio: {ratio:.1f}×")
        if self.mtp_transition_ratio >= self.gamma_freeze_ratio:
            diag["warnings"].append("MTP transition overlaps gamma freeze — potential instability")
        p1_tok = total_tokens * self.phase1_token_fraction
        if p1_tok < active_params * 10:
            diag["warnings"].append(f"Phase 1 tokens ({p1_tok/1e9:.1f}B) < 10× active params")
        return diag

    def save(self, path: Path):
        with open(path, "w") as f: json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path):
        with open(path) as f: return cls(**json.load(f))
```

### Integration: Wrapping pre-train.py

```python
"""
Integration layer — replaces the inline phase transition + scattered schedules
in pre-train.py (lines 1391–1461) with orchestrated calls.
"""

def orchestrated_step(step, orchestrator, monitor, visualizer,
                      model, optimizers, config,
                      train_loss, grad_norm, expert_loads=None, wandb_run=None):
    """Called each step after backward + optimizer.step(). Returns 'continue' or 'transition_needed'."""
    s = orchestrator.get_schedules(step)

    # Apply LR schedule
    for opt in optimizers:
        if opt is not None:
            for g in opt.param_groups: g["lr"] = g["initial_lr"] * s["lr_multiplier"]

    # Apply Muon momentum
    if len(optimizers) > 1 and optimizers[1] is not None:
        for g in optimizers[1].param_groups: g["momentum"] = s["muon_momentum"]

    # Monitor anomalies
    for anomaly in filter(None, [
        monitor.check_loss(step, train_loss),
        monitor.check_gradients(step, grad_norm),
        monitor.check_expert_balance(step, expert_loads) if expert_loads is not None else None,
    ]):
        _apply_remediation(monitor.get_remediation(anomaly), optimizers, config, step)

    # Visualization
    visualizer.record(step, s)
    if wandb_run and step % 100 == 0: visualizer.export_wandb(wandb_run, step, s)

    if orchestrator.should_transition(step): return "transition_needed"
    orchestrator.step(); return "continue"


def _apply_remediation(action, optimizers, config, step):
    act = action.get("action", "log")
    if act == "log":
        logger.warning(f"[{step}] {action['msg']}")
    elif act == "scale_lr":
        for o in optimizers:
            if o: [g.__setitem__("lr", g["lr"] * action["factor"]) for g in o.param_groups]
        logger.warning(f"[{step}] LR scaled by {action['factor']}: {action['msg']}")
    elif act == "skip_and_scale_lr":
        for o in optimizers:
            if o:
                o.zero_grad(set_to_none=True)
                [g.__setitem__("lr", g["lr"] * action["factor"]) for g in o.param_groups]
        logger.warning(f"[{step}] Step skipped, LR×{action['factor']}: {action['msg']}")
    elif act == "scale_grad_clip" and hasattr(config, "grad_clip"):
        config.grad_clip *= action["factor"]
        logger.warning(f"[{step}] Grad clip → {config.grad_clip}: {action['msg']}")
    elif act == "rollback":
        raise RuntimeError(f"Rollback requested at step {step}: {action['msg']}")
```

---

## 4. File Placement

```
nanoseek/
├── scripts/
│   ├── pre-train.py                        # Existing (add orchestrator hooks)
│   ├── training_orchestrator.py            # §6a Orchestrator + ScheduleManager
│   └── schedule_visualizer.py              # §6b ScheduleVisualizer
├── model/eval/
│   ├── checkpoint_manager.py               # Existing (base I/O)
│   ├── atomic_checkpoint_manager.py        # §6c AtomicCheckpointManager
│   └── monitoring.py                       # §6d TrainingMonitor
└── implementation_docs/
    └── 06_MULTI_PHASE_TRAINING_ORCHESTRATION.md
```

## 5. Verification

```python
def test_schedule_boundaries():
    sm = ScheduleManager(total_steps=1000, warmup_ratio=0.05, warmdown_ratio=0.2)
    assert abs(sm.get_lr_multiplier(50) - 1.0) < 1e-6          # warmup complete
    assert sm.get_mtp_weight(599) == 0.3                         # before transition
    assert sm.get_mtp_weight(600) == 0.1                         # after transition
    assert sm.get_gamma(799) == 0.001                             # before freeze
    assert sm.get_gamma(800) == 0.0                               # after freeze
    assert abs(sm.get_muon_momentum(300) - 0.95) < 1e-6          # warmup done

def test_loss_spike_detection():
    m = TrainingMonitor()
    for i in range(50): m.check_loss(i, 3.0 + i*0.001)
    assert m.check_loss(50, 10.0) is not None                    # spike detected

def test_expert_collapse():
    m = TrainingMonitor()
    loads = torch.ones(64) * 100.0; loads[5] = 0.1; loads[42] = 0.05
    a = m.check_expert_balance(100, loads)
    assert a is not None and "2 experts" in a.details

def test_atomic_checkpoint(tmp_path):
    mgr = AtomicCheckpointManager(tmp_path)
    mgr.save_atomic(42, {"w": torch.randn(10)}, [{}], {"step": 42})
    loaded = torch.load(tmp_path / "step_000042" / "model.pt", weights_only=True)
    assert "w" in loaded
```

## 6. Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Phase transition | <30s | Save + verify + apply + rebuild dataloader |
| Atomic ckpt overhead | <5% step time | Async write overlaps compute |
| Monitor per step | <1ms | Rolling stats, no GPU sync |
| Rollback recovery | <60s | Load ckpt + rebuild optim states |

## 7. Gotchas & Edge Cases

1. **Crash during transition**: The atomic protocol saves P1 **before** applying P2. On restart, detect `phase1_final/` directory → load and retry.

2. **Async checkpoint OOM**: `save_async()` clones model to CPU (~9.5 GB). If host RAM is tight, use synchronous `save_atomic()` or shard the CPU clone.

3. **DDP barrier deadlock**: Checkpoint uses `dist.barrier()` after rank 0 rename. Always checkpoint at deterministic points (start of step, not mid-accumulation).

4. **Schedule drift after rollback**: `ScheduleManager` is stateless — it derives everything from `step`. Restoring `step` from checkpoint produces identical schedules.

5. **Expert collapse false positives during warmup**: Early routing is random with high load variance. Skip `check_expert_balance()` for `step < warmup_steps`.

6. **MTP step-function spike**: The λ=0.3→0.1 transition is a step function (matching V3). This causes a 1–2 step loss bump that the monitor's 100-step window absorbs. Do not smooth this transition — it requires gradient correction terms.

7. **Phase 2 grad accumulation**: When context doubles (4K→8K), tokens/micro-batch doubles, so grad_accum must halve. Validate `total_batch_size % new_world_tokens == 0` before transition or training crashes.

---

*"The orchestrator doesn't make training faster. It makes training reliable. The difference between a 14-hour run that completes and one that crashes at hour 11 is worth more than any kernel optimization."*

— Principal Engineer's Note, Foundation Models Division, 2026
