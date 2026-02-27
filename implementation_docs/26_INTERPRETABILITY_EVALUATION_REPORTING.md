# 26 — Interpretability Evaluation, Metrics & Diagnostic Reporting

## Principal Engineer's Implementation Guide — February 2026

**Author**: Distinguished AI Researcher & Principal Engineer, Foundation Models Division
**Scope**: Complete interpretability evaluation suite — quantitative faithfulness/completeness/minimality/consistency metrics, structured diagnostic reporting (JSON + HTML + Markdown), and human evaluation framework for NanoSeek's MLA + MoE + MTP + DSA architecture
**Prerequisite**: `00_MASTER_PLAN.md`, `23_INTERPRETABILITY_FRAMEWORK.md` (hook architecture), `24_MECHANISTIC_ANALYSIS.md` (patching, SAE, circuit discovery), `25_VISUALIZATION_TOOLKIT.md` (visualization)
**Criticality**: **HIGH** — Interpretability without evaluation is storytelling. Evaluation without reporting is invisible. This doc closes the loop.

---

## 1. Problem Statement

You have built the interpretability framework (Doc 23), run mechanistic analyses (Doc 24), and generated visualizations (Doc 25). You can hook into any layer, patch activations, train sparse autoencoders, discover circuits, and render attention heatmaps. You have produced explanations.

**But how do you know your explanations are correct?**

This is the **meta-evaluation problem** — the problem of evaluating the evaluators. It is the most neglected question in interpretability research, and the one that separates rigorous science from pattern-matching on cherry-picked examples.

### The Qualitative Trap

Most interpretability work follows a seductive but dangerous pattern:

```
1. Run analysis on a carefully chosen input
2. Find a pattern that looks meaningful
3. Tell a compelling narrative about what the model "does"
4. Publish with beautiful visualizations
5. Never check whether the explanation is actually correct
```

Adebayo et al. (2018) — "Sanity Checks for Saliency Maps" — demonstrated that several popular saliency methods produce attributions indistinguishable from random noise when tested against randomized model weights. The methods had been cited thousands of times. The visualizations were compelling. The explanations were wrong.

| Failure Mode | What Happened | Why Evaluation Would Have Caught It |
|---|---|---|
| Saliency ≈ edge detector | Gradient saliency highlighted input edges, not model-relevant features | Faithfulness test: attributions unchanged after randomizing weights → score ≈ 0 |
| Cherry-picked circuits | IOI circuit shown on 5 hand-crafted examples | Completeness test: circuit fails on 40% of held-out examples |
| Over-specified explanations | "Critical" attention head ablatable with no effect | Minimality test: ablating 80% of "critical" heads has no effect |
| Inconsistent SAE features | Feature semantics shift across random seeds | Consistency test: seed variance > 0.3 → unstable |
| Unfalsifiable narratives | "This layer does abstraction" — untestable | Counterfactual validity: no prediction → no evaluation possible |

### What We Need

| Capability | Question It Answers | Where Used |
|---|---|---|
| **Quantitative metrics** | Is this explanation faithful? Complete? Minimal? Consistent? | CI/CD gates, regression detection |
| **Structured reporting** | What did the analysis find, and can someone reproduce it? | Peer review, audit trails |
| **Human evaluation** | Do domain experts agree the explanation is meaningful? | Ground truth calibration |
| **Pipeline integration** | Does interpretability inform ship/reject decisions? | Doc 17 eval gate extension |

### Current State

- Docs 23–25 provide hook infrastructure, mechanistic analysis, and visualization
- **No quantitative evaluation of interpretability claims exists**
- **No structured reporting infrastructure exists**
- **No human evaluation framework or eval gate integration exists**

---

## 2. First Principles — Evaluating Interpretability

Interpretability claims are scientific hypotheses about model internals. Like all scientific claims, they must be testable, quantifiable, and reproducible.

### 2a. Faithfulness

**Does the explanation actually reflect model behavior?** (Jacovi & Goldberg 2020)

**Sufficiency**: Running *only* the identified circuit reproduces model output.

```
Sufficiency(circuit, input) = 1 - KL(P_full || P_circuit)
  = 1.0 → circuit fully reproduces behavior;  = 0.0 → maximally different
```

**Comprehensiveness**: Ablating *only* the identified circuit destroys model output.

```
Comprehensiveness(circuit, input) = KL(P_full || P_ablated)
  → ∞ = circuit is necessary;  → 0 = circuit is irrelevant
```

**Faithfulness correlation**: Spearman rank correlation between attribution scores and actual perturbation effects — do the rankings agree?

### 2b. Completeness

**Does the explanation account for all important model behavior?**

```
CircuitRecoveryRate = (# inputs where circuit reproduces output within ε) / (# total inputs)
UnexplainedVariance = 1 - R²(predicted_behavior, actual_behavior)
```

### 2c. Minimality

**Is the explanation as simple as possible?** The trivial circuit "everything matters" is faithful but useless.

```
Minimality = 1 - (|circuit components| / |total components|)
```

Fundamental tension: faithfulness vs. minimality. Report the Pareto frontier — the smallest circuit achieving >80% sufficiency.

### 2d. Consistency

**Input invariance**: Paraphrased inputs should yield similar explanations (cosine similarity of explanation vectors). **Seed invariance**: Re-running with different random seeds should produce similar results (1 - coefficient of variation).

### 2e. Human Evaluability

The HIVE framework proposes structured tasks: circuit validation (binary judgment), feature labeling (free-text + agreement), failure prediction (tests whether explanation enables prediction). Expensive (~$50–200/batch) but provides ground truth that automatic metrics approximate.

### 2f. Counterfactual Validity

The strongest test: predictions about interventions must hold. If "head 3.7 implements induction," then ablating it should degrade induction tasks specifically. Requires pre-registration to avoid post-hoc rationalization.

### 2g. Feature Recovery Rate (SAE-Specific)

When ground truth features are available (positional, syntactic, semantic, routing), feature recovery = fraction of known features with a matched SAE feature (correlation > τ = 0.7).

---

## 3. Production Code

### 3a. Interpretability Metrics (`fms/interpretability/metrics.py`)

```python
"""
Interpretability evaluation metrics for NanoSeek.

Quantitative metrics: faithfulness (sufficiency, comprehensiveness, correlation),
consistency (input/seed invariance), completeness (circuit recovery, unexplained
variance), SAE quality (reconstruction, sparsity, dead features, purity), and
MoE interpretability (expert consistency, specialization, routing predictability).

References: Jacovi & Goldberg 2020, DeYoung et al. 2020, Bricken et al. 2023.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class MetricResult:
    """Container for a single metric evaluation result."""
    name: str
    value: float
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    n_samples: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {"name": self.name, "value": round(self.value, 6), "n_samples": self.n_samples}
        if self.ci_lower is not None:
            d["ci_lower"] = round(self.ci_lower, 6)
            d["ci_upper"] = round(self.ci_upper, 6)
        if self.metadata:
            d["metadata"] = self.metadata
        return d


def _bootstrap_ci(values: List[float], n_boot: int = 1000, ci: float = 0.95, seed: int = 42) -> Tuple[float, float, float]:
    """Bootstrap mean and confidence interval. Returns (mean, lo, hi)."""
    if len(values) <= 1:
        v = values[0] if values else 0.0
        return v, v, v
    rng = torch.Generator().manual_seed(seed)
    t = torch.tensor(values, dtype=torch.float64)
    means = sorted(t[torch.randint(0, len(t), (len(t),), generator=rng)].mean().item() for _ in range(n_boot))
    alpha = (1 - ci) / 2
    return t.mean().item(), means[int(alpha * n_boot)], means[int((1 - alpha) * n_boot)]


def _kl_divergence(logits_p: Tensor, logits_q: Tensor) -> float:
    """KL(P || Q) from unnormalized logits."""
    log_p = F.log_softmax(logits_p.float(), dim=-1)
    log_q = F.log_softmax(logits_q.float(), dim=-1)
    return max((log_p.exp() * (log_p - log_q)).sum(-1).mean().item(), 0.0)


class FaithfulnessMetrics:
    """Measure whether explanations faithfully represent model behavior."""

    def __init__(self, ablation_value: str = "zero", n_bootstrap: int = 1000, device: str = "cpu"):
        assert ablation_value in ("zero", "mean")
        self.ablation_value = ablation_value
        self.n_bootstrap = n_bootstrap
        self.device = device

    def _apply_circuit_mask(self, model, input_ids, circuit_mask, keep_circuit, mean_cache=None):
        """Run model with circuit components either kept or ablated."""
        handles = []
        def _make_hook(mask, mean_act):
            def hook_fn(module, inp, output):
                act = output[0] if isinstance(output, tuple) else output
                abl_mask = mask if not keep_circuit else ~mask
                repl = mean_act.to(act.device).expand_as(act) if (self.ablation_value == "mean" and mean_act is not None) else torch.zeros_like(act)
                act = torch.where(abl_mask.unsqueeze(0).expand_as(act).to(act.device), repl, act)
                return (act, *output[1:]) if isinstance(output, tuple) else act
            return hook_fn
        try:
            for name, mask in circuit_mask.items():
                mod = dict(model.named_modules()).get(name)
                if mod is None:
                    continue
                handles.append(mod.register_forward_hook(_make_hook(mask, (mean_cache or {}).get(name))))
            with torch.no_grad():
                out = model(input_ids.to(self.device))
            return out.logits if hasattr(out, "logits") else out
        finally:
            for h in handles:
                h.remove()

    def _full_logits(self, model, input_ids):
        with torch.no_grad():
            out = model(input_ids.to(self.device))
            return out.logits if hasattr(out, "logits") else out

    def sufficiency_score(self, model, circuit_mask, input_ids, mean_cache=None) -> MetricResult:
        """Output match when running only the identified circuit. 1.0 = perfect."""
        scores = []
        for i in range(input_ids.size(0)):
            ids = input_ids[i:i+1]
            kl = _kl_divergence(self._full_logits(model, ids), self._apply_circuit_mask(model, ids, circuit_mask, True, mean_cache))
            scores.append(max(1.0 - kl, 0.0))
        mean, lo, hi = _bootstrap_ci(scores, self.n_bootstrap)
        return MetricResult("sufficiency", mean, lo, hi, len(scores), {"ablation": self.ablation_value})

    def comprehensiveness_score(self, model, circuit_mask, input_ids, mean_cache=None) -> MetricResult:
        """Output degradation when ablating the circuit. Higher = more necessary."""
        scores = []
        for i in range(input_ids.size(0)):
            ids = input_ids[i:i+1]
            scores.append(_kl_divergence(self._full_logits(model, ids), self._apply_circuit_mask(model, ids, circuit_mask, False, mean_cache)))
        mean, lo, hi = _bootstrap_ci(scores, self.n_bootstrap)
        return MetricResult("comprehensiveness", mean, lo, hi, len(scores), {"ablation": self.ablation_value})

    def faithfulness_correlation(self, attributions: Tensor, perturbation_effects: Tensor) -> MetricResult:
        """Spearman correlation between attribution scores and actual perturbation effects."""
        n = attributions.numel()
        assert n == perturbation_effects.numel() and n >= 3
        def _rank(t):
            _, idx = t.flatten().float().sort()
            r = torch.empty_like(idx, dtype=torch.float32)
            r[idx] = torch.arange(len(idx), dtype=torch.float32)
            return r
        d = _rank(attributions) - _rank(perturbation_effects)
        rho = 1.0 - 6.0 * (d**2).sum().item() / (n * (n**2 - 1))
        return MetricResult("faithfulness_correlation", rho, n_samples=n, metadata={"method": "spearman"})


class ConsistencyMetrics:
    """Measure explanation stability across similar inputs."""

    def __init__(self, device: str = "cpu", n_bootstrap: int = 1000):
        self.device = device
        self.n_bootstrap = n_bootstrap

    def input_invariance_score(self, model, interpreter, paraphrases) -> MetricResult:
        """Cosine similarity of explanations on paraphrased inputs."""
        sims = []
        for orig, para in paraphrases:
            with torch.no_grad():
                e1 = interpreter(model, orig.to(self.device)).float().flatten()
                e2 = interpreter(model, para.to(self.device)).float().flatten()
            ml = min(len(e1), len(e2))
            if e1[:ml].norm() < 1e-12 or e2[:ml].norm() < 1e-12:
                sims.append(0.0)
            else:
                sims.append(F.cosine_similarity(e1[:ml].unsqueeze(0), e2[:ml].unsqueeze(0)).item())
        mean, lo, hi = _bootstrap_ci(sims, self.n_bootstrap)
        return MetricResult("input_invariance", mean, lo, hi, len(sims))

    def seed_invariance_score(self, model, interpreter, input_ids, n_seeds=5) -> MetricResult:
        """1 - coefficient of variation across random seeds. Higher = more stable."""
        vals = []
        for s in range(n_seeds):
            with torch.no_grad():
                r = interpreter(model, input_ids.to(self.device), s)
                vals.append(r.float().mean().item() if isinstance(r, Tensor) else float(r))
        t = torch.tensor(vals, dtype=torch.float64)
        cv = t.std().item() / (abs(t.mean().item()) + 1e-12)
        return MetricResult("seed_invariance", max(1.0 - cv, 0.0), n_samples=n_seeds,
                            metadata={"cv": round(cv, 6), "per_seed": [round(v, 6) for v in vals]})


class CompletenessMetrics:
    """Measure whether explanation accounts for full model behavior."""

    def __init__(self, match_mode="argmax", epsilon=0.05, device="cpu", n_bootstrap=1000):
        assert match_mode in ("argmax", "top5", "kl")
        self.match_mode, self.epsilon, self.device, self.n_bootstrap = match_mode, epsilon, device, n_bootstrap

    def _match(self, full, circ):
        if self.match_mode == "argmax":
            return full.argmax(-1).eq(circ.argmax(-1)).all().item()
        elif self.match_mode == "top5":
            return len(set(full.topk(5,-1).indices.flatten().tolist()) & set(circ.topk(5,-1).indices.flatten().tolist())) / 5 >= 0.8
        return _kl_divergence(full, circ) < self.epsilon

    def circuit_recovery_rate(self, model, circuit_runner, test_inputs) -> MetricResult:
        """Fraction of test inputs where circuit reproduces model output."""
        matches = []
        for i in range(test_inputs.size(0)):
            ids = test_inputs[i:i+1].to(self.device)
            with torch.no_grad():
                full = model(ids); fl = (full.logits if hasattr(full, "logits") else full)[:, -1, :]
                cl = circuit_runner(model, ids); cl = cl[:, -1, :] if cl.dim() == 3 else cl
            matches.append(1.0 if self._match(fl, cl) else 0.0)
        mean, lo, hi = _bootstrap_ci(matches, self.n_bootstrap)
        return MetricResult("circuit_recovery_rate", mean, lo, hi, len(matches), {"match_mode": self.match_mode})

    def unexplained_variance(self, model, explanation_predictor, dataset) -> MetricResult:
        """1 - R² between explanation predictions and actual outputs. Lower = better."""
        actuals, preds = [], []
        for i in range(dataset.size(0)):
            ids = dataset[i:i+1].to(self.device)
            with torch.no_grad():
                out = model(ids); a = (out.logits if hasattr(out, "logits") else out)[:, -1, :]
                p = explanation_predictor(ids); p = p[:, -1, :] if p.dim() == 3 else p
            actuals.append(a.float().cpu()); preds.append(p.float().cpu())
        a_cat, p_cat = torch.cat(actuals).flatten(), torch.cat(preds).flatten()
        r2 = 1.0 - ((a_cat - p_cat)**2).sum().item() / ((a_cat - a_cat.mean())**2).sum().item() + 1e-12
        return MetricResult("unexplained_variance", max(1.0 - r2, 0.0), n_samples=dataset.size(0))


class SAEMetrics:
    """Evaluate Sparse Autoencoder quality."""

    def __init__(self, device="cpu", n_bootstrap=1000):
        self.device, self.n_bootstrap = device, n_bootstrap

    def _get_features(self, sae, activations):
        out = sae(activations.to(self.device))
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            return out[0], out[1]
        return out, out

    def reconstruction_loss(self, sae, activations) -> MetricResult:
        """MSE between original and reconstructed activations."""
        recon, _ = self._get_features(sae, activations)
        mse = F.mse_loss(recon.float(), activations.to(self.device).float()).item()
        return MetricResult("sae_reconstruction_loss", mse, n_samples=activations.size(0),
                            metadata={"relative_mse": round(mse / (activations.float().var().item() + 1e-12), 6)})

    def feature_sparsity(self, sae, activations, threshold=1e-5) -> MetricResult:
        """Average L0 norm — number of active features per input."""
        _, feats = self._get_features(sae, activations)
        l0 = (feats.abs() > threshold).float().sum(-1)
        return MetricResult("sae_feature_sparsity", l0.mean().item(), n_samples=activations.size(0),
                            metadata={"n_features": feats.size(-1), "median_l0": round(l0.median().item(), 4)})

    def dead_feature_fraction(self, sae, activations, threshold=1e-5) -> MetricResult:
        """Fraction of features that never activate. <5% healthy, >20% failure."""
        _, feats = self._get_features(sae, activations)
        n_dead = (~(feats.abs() > threshold).any(0)).sum().item()
        n_total = feats.size(-1)
        return MetricResult("sae_dead_feature_fraction", n_dead / max(n_total, 1),
                            n_samples=activations.size(0), metadata={"n_dead": n_dead, "n_total": n_total})

    def feature_purity(self, sae, activations, labels, threshold=1e-5) -> MetricResult:
        """1 - normalized entropy of label distribution per active feature."""
        _, feats = self._get_features(sae, activations)
        labels = labels.to(self.device)
        n_classes = int(labels.max().item()) + 1
        max_ent = math.log(n_classes + 1e-12)
        purities = []
        for f in range(feats.size(-1)):
            mask = feats[:, f].abs() > threshold
            if mask.sum() < 5:
                continue
            counts = torch.bincount(labels[mask].long(), minlength=n_classes).float()
            p = counts / counts.sum(); p = p[p > 0]
            purities.append(1.0 - (-(p * p.log()).sum().item()) / max_ent)
        if not purities:
            return MetricResult("sae_feature_purity", 0.0, n_samples=0)
        mean, lo, hi = _bootstrap_ci(purities, self.n_bootstrap)
        return MetricResult("sae_feature_purity", mean, lo, hi, len(purities))


class MoEInterpretabilityMetrics:
    """MoE-specific interpretability metrics for NanoSeek (64+2 experts, 8 active/token)."""

    def __init__(self, n_experts=64, top_k=8, device="cpu", n_bootstrap=1000):
        self.n_experts, self.top_k, self.device, self.n_bootstrap = n_experts, top_k, device, n_bootstrap

    def expert_consistency_score(self, routing_data: List[Dict[str, Tensor]]) -> MetricResult:
        """Do similar tokens consistently route to the same expert? (Jaccard similarity)"""
        from collections import defaultdict
        token_sets: Dict[int, List[set]] = defaultdict(list)
        for rd in routing_data:
            tids = rd["token_ids"].flatten().cpu()
            eidxs = rd["expert_indices"].cpu()
            for i in range(len(tids)):
                token_sets[tids[i].item()].append(set(eidxs[i].tolist()))
        scores = []
        for sets in token_sets.values():
            if len(sets) < 2:
                continue
            jaccards = [len(sets[i] & sets[j]) / max(len(sets[i] | sets[j]), 1) for i in range(len(sets)) for j in range(i+1, len(sets))]
            if jaccards:
                scores.append(sum(jaccards) / len(jaccards))
        if not scores:
            return MetricResult("expert_consistency", 0.0, n_samples=0)
        mean, lo, hi = _bootstrap_ci(scores, self.n_bootstrap)
        return MetricResult("expert_consistency", mean, lo, hi, len(scores))

    def expert_specialization_entropy(self, routing_data: List[Dict[str, Tensor]]) -> MetricResult:
        """How specialized is each expert? Low entropy = high specialization."""
        counts = [{} for _ in range(self.n_experts)]
        for rd in routing_data:
            tids = rd["token_ids"].flatten().cpu()
            eidxs = rd["expert_indices"].cpu()
            for i in range(len(tids)):
                tid = tids[i].item()
                for e in eidxs[i].tolist():
                    if 0 <= e < self.n_experts:
                        counts[e][tid] = counts[e].get(tid, 0) + 1
        entropies = []
        for c in counts:
            if not c:
                continue
            total = sum(c.values())
            probs = [v / total for v in c.values()]
            ent = -sum(p * math.log(p + 1e-12) for p in probs)
            entropies.append(ent / (math.log(len(c) + 1e-12) + 1e-12))
        if not entropies:
            return MetricResult("expert_specialization_entropy", 1.0, n_samples=0)
        mean, lo, hi = _bootstrap_ci(entropies, self.n_bootstrap)
        return MetricResult("expert_specialization_entropy", mean, lo, hi, len(entropies))

    def routing_predictability(self, features: Tensor, routing_decisions: Tensor) -> MetricResult:
        """Can we predict routing from input features? Linear probe accuracy."""
        features = features.to(self.device).float()
        routing_decisions = routing_decisions.to(self.device).long()
        n = features.size(0)
        split = int(0.8 * n)
        n_classes = int(routing_decisions.max().item()) + 1
        probe = nn.Linear(features.size(1), n_classes).to(self.device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        probe.train()
        for _ in range(50):
            for s in range(0, split, 256):
                loss = F.cross_entropy(probe(features[s:s+256]), routing_decisions[s:s+256])
                opt.zero_grad(); loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            acc = (probe(features[split:]).argmax(-1) == routing_decisions[split:]).float().mean().item()
        return MetricResult("routing_predictability", acc, n_samples=n,
                            metadata={"random_baseline": round(1.0 / max(n_classes, 1), 4), "n_classes": n_classes})
```

### 3b. Diagnostic Report Generator (`fms/interpretability/report.py`)

```python
"""
Interpretability diagnostic report generator for NanoSeek.

Structured reports combining quantitative metrics, qualitative analysis,
and embedded visualizations. Three output formats: JSON (CI/CD), HTML
(human-readable), Markdown (Git-trackable).
"""

from __future__ import annotations

import base64, datetime, json, os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ReportSection:
    """A single section in an interpretability report."""
    title: str
    content: str = ""
    figures: List[Dict[str, str]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    subsections: List["ReportSection"] = field(default_factory=list)
    severity: str = "info"
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = {"title": self.title, "content": self.content, "metrics": self.metrics, "severity": self.severity}
        if self.figures: d["figures"] = self.figures
        if self.subsections: d["subsections"] = [s.to_dict() for s in self.subsections]
        if self.recommendations: d["recommendations"] = self.recommendations
        return d


class InterpretabilityReport:
    """Structured report combining quantitative metrics and qualitative analysis."""

    def __init__(self, model_name: str, analysis_config: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.analysis_config = analysis_config or {}
        self.sections: List[ReportSection] = []
        self.metadata = {
            "model_name": model_name,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "analysis_config": self.analysis_config,
            "generator": "nanoseek-interpretability-v1",
        }
        self._global_metrics: Dict[str, Any] = {}

    def add_section(self, title, content="", figures=None, metrics=None,
                    severity="info", recommendations=None) -> ReportSection:
        """Add a top-level section to the report."""
        section = ReportSection(title=title, content=content, figures=figures or [],
                                metrics=metrics or {}, severity=severity,
                                recommendations=recommendations or [])
        self.sections.append(section)
        for k, v in (metrics or {}).items():
            self._global_metrics[f"{title}/{k}"] = v
        return section

    def add_global_metric(self, name: str, value: Any) -> None:
        self._global_metrics[name] = value

    def _summary(self) -> Dict[str, Any]:
        crit = sum(1 for s in self.sections if s.severity == "critical")
        warn = sum(1 for s in self.sections if s.severity == "warning")
        findings = [f"[{s.severity.upper()}] {s.title}: {s.content[:200]}" for s in self.sections if s.severity in ("critical", "warning")]
        recs = [r for s in self.sections for r in s.recommendations]
        return {"n_sections": len(self.sections), "n_critical": crit, "n_warnings": warn,
                "top_findings": findings[:10], "recommendations": recs[:20],
                "overall_status": "FAIL" if crit else ("WARN" if warn else "PASS")}

    def generate_json(self, output_path: Union[str, Path]) -> Dict[str, Any]:
        """Serialize report to JSON."""
        d = {"metadata": self.metadata, "global_metrics": self._global_metrics,
             "sections": [s.to_dict() for s in self.sections], "summary": self._summary()}
        p = Path(output_path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d, indent=2, default=str))
        return d

    def generate_html(self, output_path: Union[str, Path]) -> str:
        """Render report as self-contained HTML with embedded visualizations."""
        summary = self._summary()
        status_cls = {"PASS": "status-pass", "WARN": "status-warn", "FAIL": "status-fail"}[summary["overall_status"]]
        badge = {"PASS": "badge-pass", "WARN": "badge-warning", "FAIL": "badge-critical"}[summary["overall_status"]]

        metric_rows = "".join(f"<tr><td>{k}</td><td>{v:.4f if isinstance(v, float) else v}</td></tr>\n" for k, v in self._global_metrics.items())

        def _section_html(sec, depth=1):
            tag = f"h{min(depth+1, 6)}"
            bcls = {"info": "badge-info", "warning": "badge-warning", "critical": "badge-critical"}.get(sec.severity, "badge-info")
            parts = [f'<div class="section"><{tag}>{sec.title} <span class="badge {bcls}">{sec.severity}</span></{tag}>']
            if sec.content: parts.append(f"<p>{sec.content}</p>")
            if sec.metrics:
                parts.append('<table class="metric-table"><tr><th>Metric</th><th>Value</th></tr>')
                parts.extend(f"<tr><td>{k}</td><td>{v:.4f if isinstance(v, float) else v}</td></tr>" for k, v in sec.metrics.items())
                parts.append("</table>")
            for fig in sec.figures:
                src = fig.get("path", "")
                parts.append(f'<div class="figure">')
                if os.path.exists(src):
                    with open(src, "rb") as f: b64 = base64.b64encode(f.read()).decode()
                    parts.append(f'<img src="data:image/png;base64,{b64}" alt="{fig.get("alt_text","")}" />')
                parts.append(f'<figcaption>{fig.get("caption","")}</figcaption></div>')
            if sec.recommendations:
                parts.append('<ul class="recs">' + "".join(f"<li>{r}</li>" for r in sec.recommendations) + "</ul>")
            for sub in sec.subsections: parts.append(_section_html(sub, depth + 1))
            parts.append("</div>")
            return "\n".join(parts)

        html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>Interpretability Report — {self.model_name}</title>
<style>
:root{{--bg:#0d1117;--fg:#c9d1d9;--accent:#58a6ff;--warn:#d29922;--crit:#f85149;--ok:#3fb950;--card:#161b22;--border:#30363d}}
*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:-apple-system,sans-serif;background:var(--bg);color:var(--fg);line-height:1.6;padding:2rem;max-width:1200px;margin:0 auto}}
h1{{color:var(--accent)}}h2{{color:var(--fg);margin:1.5rem 0 .5rem;border-bottom:1px solid var(--border);padding-bottom:.3rem}}
.meta{{background:var(--card);border:1px solid var(--border);border-radius:6px;padding:1rem;margin:1rem 0;font-size:.9rem}}
.metric-table{{width:100%;border-collapse:collapse;margin:1rem 0}}.metric-table th,.metric-table td{{padding:.5rem 1rem;text-align:left;border-bottom:1px solid var(--border)}}.metric-table th{{background:var(--card);color:var(--accent)}}
.badge{{display:inline-block;padding:2px 8px;border-radius:12px;font-size:.8rem;font-weight:600}}.badge-info{{background:#1f6feb33;color:var(--accent)}}.badge-warning{{background:#d2992233;color:var(--warn)}}.badge-critical{{background:#f8514933;color:var(--crit)}}.badge-pass{{background:#3fb95033;color:var(--ok)}}
.section{{background:var(--card);border:1px solid var(--border);border-radius:6px;padding:1.5rem;margin:1rem 0}}.recs{{padding-left:1.5rem}}.recs li{{color:var(--warn)}}
.figure{{margin:1rem 0;text-align:center}}.figure img{{max-width:100%;border-radius:6px}}.figure figcaption{{font-size:.85rem;color:#8b949e}}
.status-pass{{color:var(--ok)}}.status-warn{{color:var(--warn)}}.status-fail{{color:var(--crit)}}
</style></head><body>
<h1>Interpretability Report</h1><p style="font-size:1.1rem;color:#8b949e">Model: <strong>{self.model_name}</strong></p>
<div class="meta"><strong>Generated:</strong> {self.metadata['timestamp']}<br><strong>Config:</strong> <code>{json.dumps(self.analysis_config, separators=(',',':'))}</code></div>
<h2>Summary <span class="badge {badge}">{summary['overall_status']}</span></h2>
<p>{summary['n_sections']} sections | {summary['n_critical']} critical | {summary['n_warnings']} warnings</p>
<table class="metric-table"><tr><th>Metric</th><th>Value</th></tr>{metric_rows}</table>
{"".join(_section_html(s, 1) for s in self.sections)}
<hr style="border-color:var(--border);margin:2rem 0"><p style="font-size:.8rem;color:#8b949e">Generated by NanoSeek Interpretability Suite — {self.metadata['timestamp']}</p>
</body></html>"""
        p = Path(output_path); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(html)
        return html

    def generate_markdown(self, output_path: Union[str, Path]) -> str:
        """Render report as Markdown."""
        def _sec_md(sec, depth=2):
            h = "#" * min(depth, 6)
            emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(sec.severity, "")
            lines = [f"{h} {emoji} {sec.title}\n", sec.content + "\n" if sec.content else ""]
            if sec.metrics:
                lines += ["| Metric | Value |", "|--------|-------|"]
                lines += [f"| {k} | {v:.4f if isinstance(v, float) else v} |" for k, v in sec.metrics.items()]
                lines.append("")
            for fig in sec.figures:
                lines.append(f"![{fig.get('alt_text', '')}]({fig.get('path', '')})")
            if sec.recommendations:
                lines += ["**Recommendations:**"] + [f"- {r}" for r in sec.recommendations] + [""]
            for sub in sec.subsections:
                lines.append(_sec_md(sub, depth + 1))
            return "\n".join(lines)

        metric_table = "## Metric Summary\n\n| Metric | Value |\n|--------|-------|\n" + \
            "\n".join(f"| {k} | {v:.4f if isinstance(v, float) else v} |" for k, v in self._global_metrics.items())
        md = f"# Interpretability Report: {self.model_name}\n\n*Generated: {self.metadata['timestamp']}*\n\n{metric_table}\n\n" + \
             "\n".join(_sec_md(s) for s in self.sections)
        p = Path(output_path); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(md)
        return md


@dataclass
class DiagnosticConfig:
    """Configuration for the diagnostic runner."""
    n_samples: int = 100
    max_seq_len: int = 128
    batch_size: int = 8
    device: str = "cpu"
    include_attention: bool = True
    include_experts: bool = True
    include_representations: bool = True
    include_safety: bool = True
    output_dir: str = "interpretability_reports"
    random_seed: int = 42


class DiagnosticRunner:
    """Run complete interpretability diagnostic suite and generate report.

    Orchestrates: (1) capture activations via hooks, (2) attention diagnostics,
    (3) expert diagnostics, (4) representation diagnostics, (5) safety diagnostics,
    (6) assemble into InterpretabilityReport with JSON/HTML/MD output.
    """

    def __init__(self, model: nn.Module, tokenizer: Any, config: Optional[DiagnosticConfig] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or DiagnosticConfig()
        self.model.eval()
        self._rng = torch.Generator().manual_seed(self.config.random_seed)

    def _gen_inputs(self, n):
        vs = getattr(getattr(self.model, "config", None), "vocab_size", 32000)
        return torch.randint(1, vs, (n, self.config.max_seq_len), generator=self._rng)

    def run_attention_diagnostics(self) -> Dict[str, Any]:
        """Capture attention entropy per head across layers."""
        results, attn_maps, handles = {"layers": {}}, {}, []
        inputs = self._gen_inputs(min(self.config.n_samples, 32)).to(self.config.device)
        def _hook(name):
            def fn(m, i, o):
                if isinstance(o, tuple) and len(o) >= 2 and o[1] is not None:
                    attn_maps.setdefault(name, []).append(o[1].detach().cpu())
            return fn
        for n, m in self.model.named_modules():
            if "attn" in n.lower():
                handles.append(m.register_forward_hook(_hook(n)))
        try:
            with torch.no_grad():
                for i in range(0, inputs.size(0), self.config.batch_size):
                    self.model(inputs[i:i+self.config.batch_size])
        finally:
            for h in handles: h.remove()
        for ln, al in attn_maps.items():
            a = torch.cat(al, 0)
            if a.dim() >= 4:
                ent = -(a.clamp(min=1e-12) * a.clamp(min=1e-12).log()).sum(-1).mean((0, 2))
                results["layers"][ln] = {"mean_entropy": ent.mean().item(), "n_heads": a.size(1)}
        results["n_layers_captured"] = len(attn_maps)
        return results

    def run_expert_diagnostics(self) -> Dict[str, Any]:
        """Capture MoE routing statistics."""
        results, captures, handles = {"layers": {}}, {}, []
        inputs = self._gen_inputs(min(self.config.n_samples, 32)).to(self.config.device)
        def _hook(name):
            def fn(m, i, o):
                captures.setdefault(name, []).append({"shape": list(o[0].shape) if isinstance(o, tuple) else list(o.shape)})
            return fn
        for n, m in self.model.named_modules():
            if any(k in n.lower() for k in ("moe", "expert", "gate")):
                handles.append(m.register_forward_hook(_hook(n)))
        try:
            with torch.no_grad():
                for i in range(0, inputs.size(0), self.config.batch_size):
                    self.model(inputs[i:i+self.config.batch_size])
        finally:
            for h in handles: h.remove()
        for ln, caps in captures.items():
            results["layers"][ln] = {"n_captures": len(caps)}
        results["n_moe_layers_captured"] = len(captures)
        return results

    def run_representation_diagnostics(self) -> Dict[str, Any]:
        """Per-layer activation statistics and cosine similarity trajectory."""
        results, acts, handles = {"layers": {}}, {}, []
        inputs = self._gen_inputs(min(self.config.n_samples, 16)).to(self.config.device)
        def _hook(name):
            def fn(m, i, o):
                a = o[0] if isinstance(o, tuple) else o
                if isinstance(a, Tensor) and a.dim() >= 2:
                    acts.setdefault(name, []).append(a.detach().float().cpu())
            return fn
        for n, m in self.model.named_modules():
            if any(k in n.lower() for k in ("norm", "ln_", "rmsnorm")):
                handles.append(m.register_forward_hook(_hook(n)))
        try:
            with torch.no_grad():
                for i in range(0, inputs.size(0), self.config.batch_size):
                    self.model(inputs[i:i+self.config.batch_size])
        finally:
            for h in handles: h.remove()
        prev = None
        for ln in sorted(acts):
            a = torch.cat(acts[ln], 0)
            info = {"mean": a.mean().item(), "std": a.std().item(), "norm": a.norm(-1).mean().item()}
            cur = a.mean((0, 1)) if a.dim() == 3 else a.mean(0)
            if prev is not None and cur.shape == prev.shape:
                info["cosine_sim_to_prev"] = F.cosine_similarity(cur.unsqueeze(0), prev.unsqueeze(0)).item()
            prev = cur
            results["layers"][ln] = info
        results["n_layers_captured"] = len(acts)
        return results

    def run_safety_diagnostics(self) -> Dict[str, Any]:
        """Test explanation stability under input perturbation."""
        inputs = self._gen_inputs(16).to(self.config.device)
        with torch.no_grad():
            orig = self.model(inputs[:4])
            lo = orig.logits if hasattr(orig, "logits") else orig
        pert = inputs[:4].clone()
        mid = pert.size(1) // 2
        pert[:, mid], pert[:, mid+1] = pert[:, mid+1].clone(), pert[:, mid].clone()
        with torch.no_grad():
            po = self.model(pert.to(self.config.device))
            lp = po.logits if hasattr(po, "logits") else po
        kl = max((F.log_softmax(lo.float(), -1).exp() * (F.log_softmax(lo.float(), -1) - F.log_softmax(lp.float(), -1))).sum(-1).mean().item(), 0.0)
        return {"perturbation_kl": round(kl, 6), "assessment": "stable" if kl < 0.5 else "sensitive"}

    def run_full_diagnostic(self) -> InterpretabilityReport:
        """Run all diagnostics and assemble into a complete report."""
        report = InterpretabilityReport("nanoseek-diagnostic", {"n_samples": self.config.n_samples, "device": self.config.device})
        n_params = sum(p.numel() for p in self.model.parameters())
        report.add_section("Model Overview", f"NanoSeek with {n_params/1e9:.2f}B params.", metrics={"total_params": n_params})
        if self.config.include_attention:
            ar = self.run_attention_diagnostics()
            report.add_section("Attention Analysis", f"{ar['n_layers_captured']} layers analyzed.", metrics={"n_layers": ar["n_layers_captured"]})
        if self.config.include_experts:
            er = self.run_expert_diagnostics()
            report.add_section("Expert Routing", f"{er['n_moe_layers_captured']} MoE layers.", metrics={"n_moe_layers": er["n_moe_layers_captured"]})
        if self.config.include_representations:
            rr = self.run_representation_diagnostics()
            report.add_section("Representations", f"{rr['n_layers_captured']} norm layers.", metrics={"n_norm_layers": rr["n_layers_captured"]})
        if self.config.include_safety:
            sr = self.run_safety_diagnostics()
            sev = "info" if sr["assessment"] == "stable" else "warning"
            report.add_section("Explanation Robustness", f"Perturbation: {sr['assessment']}.", metrics=sr, severity=sev)
        out = Path(self.config.output_dir); out.mkdir(parents=True, exist_ok=True)
        report.generate_json(out / "diagnostic_report.json")
        report.generate_html(out / "diagnostic_report.html")
        report.generate_markdown(out / "diagnostic_report.md")
        return report
```

### 3c. Human Evaluation Framework (`fms/interpretability/human_eval.py`)

```python
"""
Human evaluation framework for NanoSeek interpretability.

Generates structured HIVE-inspired tasks (circuit validation, expert labeling,
failure detection) and scores annotations (Fleiss' kappa, explanation usefulness).
"""

from __future__ import annotations

import json, uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class TaskItem:
    """A single human evaluation task item."""
    task_id: str = ""
    task_type: str = ""
    prompt: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    options: List[str] = field(default_factory=list)
    expected_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.task_id: self.task_id = str(uuid.uuid4())[:8]

    def to_dict(self):
        d = {"task_id": self.task_id, "task_type": self.task_type, "prompt": self.prompt, "context": self.context}
        if self.options: d["options"] = self.options
        if self.expected_answer is not None: d["expected_answer"] = self.expected_answer
        if self.metadata: d["metadata"] = self.metadata
        return d


@dataclass
class TaskBatch:
    """A batch of human evaluation tasks."""
    batch_id: str = ""
    task_type: str = ""
    model_name: str = ""
    instructions: str = ""
    items: List[TaskItem] = field(default_factory=list)
    estimated_time_minutes: float = 0.0

    def __post_init__(self):
        if not self.batch_id: self.batch_id = str(uuid.uuid4())[:8]

    def to_dict(self):
        return {"batch_id": self.batch_id, "task_type": self.task_type, "model_name": self.model_name,
                "instructions": self.instructions, "n_items": len(self.items),
                "estimated_time_minutes": self.estimated_time_minutes,
                "items": [i.to_dict() for i in self.items]}


class HumanEvalTask:
    """Generate human-evaluable interpretability tasks (inspired by HIVE)."""

    def __init__(self, model_name: str = "nanoseek"):
        self.model_name = model_name
        self._batches: List[TaskBatch] = []

    def generate_circuit_validation_task(self, circuit: Dict[str, Any], examples: List[Dict[str, str]]) -> TaskBatch:
        """Present annotators with circuit description + examples, ask if circuit explains behavior."""
        items = []
        for i, ex in enumerate(examples):
            items.append(TaskItem(
                task_type="circuit_validation",
                prompt=f"Circuit: {circuit.get('name','?')} — {circuit.get('description','')}\n"
                       f"Components: {', '.join(circuit.get('components',[]))}\n"
                       f"Mechanism: {circuit.get('mechanism','')}\n\n"
                       f"Input: {ex.get('input','')}\nOutput: {ex.get('output','')}\n\n"
                       f"Does this circuit explain the model's behavior?",
                context={"circuit": circuit, "example": ex},
                options=["Yes — clearly explains", "Partially", "No", "Unsure"],
                expected_answer=ex.get("expected_answer"),
                metadata={"circuit_active": ex.get("circuit_active", True)}))
        batch = TaskBatch(task_type="circuit_validation", model_name=self.model_name,
                          instructions="Judge whether the described circuit explains the model's input→output behavior.",
                          items=items, estimated_time_minutes=len(items) * 2.0)
        self._batches.append(batch)
        return batch

    def generate_expert_labeling_task(self, expert_id: int, top_tokens: List[Dict[str, Any]], layer: int = 0) -> TaskBatch:
        """Present top-activating tokens for an expert; ask annotator to label specialization."""
        token_list = "\n".join(f"  {i+1}. \"{t['token']}\" (act={t.get('activation',0):.3f}) ctx: \"{t.get('context','')}\"" for i, t in enumerate(top_tokens[:20]))
        item = TaskItem(task_type="expert_labeling",
                        prompt=f"Expert {expert_id} (layer {layer}) top tokens:\n{token_list}\n\nWhat does this expert specialize in? Provide: label, confidence (1-5), reasoning.",
                        context={"expert_id": expert_id, "layer": layer, "top_tokens": top_tokens[:20]})
        batch = TaskBatch(task_type="expert_labeling", model_name=self.model_name,
                          instructions="Identify the concept/pattern this MoE expert specializes in.",
                          items=[item], estimated_time_minutes=5.0)
        self._batches.append(batch)
        return batch

    def generate_failure_detection_task(self, model_description: str, explanation: Dict[str, str],
                                        test_inputs: List[Dict[str, Any]]) -> TaskBatch:
        """Test if explanation enables humans to predict model success/failure on novel inputs."""
        items = []
        for test in test_inputs:
            items.append(TaskItem(
                task_type="failure_detection",
                prompt=f"Task: {model_description}\nExplanation: {explanation.get('summary','')}\n\n"
                       f"New input: {test.get('input','')}\nWill the model succeed or fail?",
                context={"explanation": explanation, "test_input": test.get("input", "")},
                options=["Succeed", "Fail", "Unsure"],
                expected_answer="Succeed" if test.get("model_succeeds", True) else "Fail",
                metadata={"ground_truth_succeeds": test.get("model_succeeds", True)}))
        batch = TaskBatch(task_type="failure_detection", model_name=self.model_name,
                          instructions="Given an explanation of model behavior, predict success/failure on new inputs.",
                          items=items, estimated_time_minutes=len(items) * 1.5)
        self._batches.append(batch)
        return batch

    def export_tasks(self, output_dir: Union[str, Path]) -> None:
        """Export all task batches as JSON files with manifest."""
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        manifest = []
        for batch in self._batches:
            fn = f"{batch.task_type}_{batch.batch_id}.json"
            (output_dir / fn).write_text(json.dumps(batch.to_dict(), indent=2, default=str))
            manifest.append({"batch_id": batch.batch_id, "task_type": batch.task_type,
                             "n_items": len(batch.items), "file": fn})
        (output_dir / "manifest.json").write_text(json.dumps(
            {"model_name": self.model_name, "n_batches": len(manifest),
             "total_items": sum(m["n_items"] for m in manifest), "batches": manifest}, indent=2))


class HumanEvalScorer:
    """Score human evaluation results."""

    def compute_inter_annotator_agreement(self, annotations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Fleiss' kappa for multi-annotator agreement."""
        from collections import defaultdict
        task_anns: Dict[str, List[str]] = defaultdict(list)
        for a in annotations: task_anns[a["task_id"]].append(a["answer"])
        if not task_anns:
            return {"fleiss_kappa": 0.0, "n_tasks": 0}
        all_cats = sorted(set(a["answer"] for a in annotations))
        cat_idx = {c: i for i, c in enumerate(all_cats)}
        k = len(all_cats)
        matrix = []
        for answers in task_anns.values():
            row = [0] * k
            for a in answers:
                if a in cat_idx: row[cat_idx[a]] += 1
            matrix.append(row)
        N, n = len(matrix), max(sum(r) for r in matrix)
        p_j = [sum(matrix[i][j] for i in range(N)) / (N * n) for j in range(k)]
        P_i = [(sum(r[j]**2 for j in range(k)) - sum(r)) / (sum(r) * (sum(r) - 1)) if sum(r) > 1 else 1.0 for r in matrix]
        P_bar, P_e = sum(P_i) / N, sum(p**2 for p in p_j)
        kappa = (P_bar - P_e) / (1 - P_e) if abs(1 - P_e) > 1e-12 else (1.0 if abs(P_bar - 1) < 1e-12 else 0.0)
        return {"fleiss_kappa": round(kappa, 4), "n_tasks": N, "n_categories": k,
                "interpretation": _interpret_kappa(kappa)}

    def compute_explanation_usefulness(self, annotations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Accuracy of human failure predictions given explanations."""
        if not annotations: return {"accuracy": 0.0, "n_annotations": 0}
        tp = fp = tn = fn = 0
        for a in annotations:
            pred_fail = "fail" in a.get("answer", "").lower()
            gt_succ = a.get("ground_truth_succeeds", True)
            if pred_fail and not gt_succ: tp += 1
            elif pred_fail and gt_succ: fp += 1
            elif not pred_fail and gt_succ: tn += 1
            else: fn += 1
        total = tp + fp + tn + fn
        acc = (tp + tn) / max(total, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        return {"accuracy": round(acc, 4), "precision": round(prec, 4), "recall": round(rec, 4),
                "f1": round(2*prec*rec / max(prec+rec, 1e-12), 4), "n_annotations": total}


def _interpret_kappa(k):
    if k < 0: return "poor"
    if k < 0.20: return "slight"
    if k < 0.40: return "fair"
    if k < 0.60: return "moderate"
    if k < 0.80: return "substantial"
    return "almost perfect"
```

---

## 4. Complete Diagnostic Pipeline

```
┌────────────────────────────────────────────────────────────────────┐
│              Interpretability Diagnostic Pipeline                  │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Raw Model      Hook Capture      Mechanistic Analysis             │
│  (NanoSeek)  ──▶  (Doc 23)    ──▶   (Doc 24)                      │
│  MLA+MoE+        Activations       Act. Patching                   │
│  MTP+DSA         Gradients          SAE Training                   │
│  ~4.75B          Routing            Circuit Discovery               │
│                                          │                         │
│              ┌───────────────────────────┤                         │
│              ▼                           ▼                         │
│   Visualization (Doc 25)     Metrics & Evaluation (This Doc)       │
│   Attention heatmaps         Faithfulness, Consistency,            │
│   Expert routing diagrams    Completeness, SAE quality,            │
│   Feature dashboards         MoE interpretability                  │
│              │                           │                         │
│              └───────────┬───────────────┘                         │
│                          ▼                                         │
│              Report Generation (This Doc)                          │
│              ┌──────┐ ┌──────┐ ┌──────────┐                       │
│              │ JSON │ │ HTML │ │ Markdown │                        │
│              │(CI)  │ │(view)│ │ (Git)    │                        │
│              └──┬───┘ └──────┘ └──────────┘                       │
│                 │                                                  │
│                 ▼                                                  │
│     ┌──────────────────────────┐    ┌──────────────────────┐       │
│     │ Human Evaluation         │    │ Eval Gate (Doc 17)   │       │
│     │ Circuit validation       │    │ interpretability ≥ θ │       │
│     │ Expert labeling          │    │ → SHIP / REJECT      │       │
│     │ Failure detection        │    └──────────────────────┘       │
│     └──────────────────────────┘                                   │
└────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Hook Capture → attention_weights, expert_routing, hidden_states
  ↓
Mechanistic Analysis → circuit_mask, sae, patching_effects
  ↓
Metrics → MetricResult(sufficiency=0.87), MetricResult(expert_consistency=0.72), ...
  ↓
Report → diagnostic_report.{json,html,md}
  ↓
Human Eval → tasks/*.json → annotations → agreement scores
  ↓
Eval Gate → SHIP / REJECT / REVIEW
```

---

## 5. File Placement

```
nanoseek/
├── fms/
│   ├── interpretability/           # NEW: Interpretability evaluation suite
│   │   ├── __init__.py
│   │   ├── metrics.py              # 3a: All metric classes
│   │   ├── report.py               # 3b: Report generator + DiagnosticRunner
│   │   └── human_eval.py           # 3c: HIVE tasks + annotation scoring
│   └── eval_harness/
│       └── gate.py                 # MODIFIED: Add interpretability checks
├── tests/
│   └── test_interpretability/
│       ├── test_metrics.py
│       ├── test_report.py
│       └── test_human_eval.py
└── scripts/
    └── run_interpretability.py     # CLI entry point
```

---

## 6. Example Reports

### Example JSON Output — Full Diagnostic Run

```json
{
  "metadata": {
    "model_name": "nanoseek-1b-dpo-v3",
    "timestamp": "2026-02-27T14:32:18Z",
    "analysis_config": {
      "n_samples": 500, "max_seq_len": 512, "device": "cuda:0",
      "faithfulness_ablation": "mean", "sae_checkpoint": "checkpoints/sae_layer8_16k.pt"
    }
  },
  "global_metrics": {
    "Faithfulness/sufficiency": 0.87,
    "Faithfulness/comprehensiveness": 2.34,
    "Faithfulness/correlation": 0.68,
    "Expert Routing/expert_consistency": 0.72,
    "Expert Routing/specialization_entropy": 0.61,
    "Expert Routing/routing_predictability": 0.43,
    "SAE Quality/reconstruction_loss": 0.0023,
    "SAE Quality/dead_feature_fraction": 0.032,
    "SAE Quality/feature_purity": 0.78,
    "Consistency/input_invariance": 0.84,
    "Consistency/seed_invariance": 0.91
  },
  "sections": [
    {
      "title": "Model Overview",
      "content": "NanoSeek-1B (DPO v3): 4.87B total / 1.08B active params, 16 layers, 64+2 MoE experts.",
      "metrics": {"total_params": 4870000000, "hidden_size": 2048, "num_layers": 16}
    },
    {
      "title": "Attention Analysis",
      "content": "Layers 0-3: low entropy (positional). Layers 12-15: high entropy (semantic). 3 redundant heads.",
      "metrics": {"mean_entropy": 3.42, "n_redundant_heads": 3},
      "recommendations": ["Investigate redundant heads for pruning (~0.3% compute savings each)"]
    },
    {
      "title": "Expert Routing Analysis",
      "content": "Consistency 0.72, specialization entropy 0.61, routing predictability 0.43.",
      "metrics": {"expert_consistency": 0.72, "most_specialized": "Expert 17 (Python keywords)"}
    },
    {
      "title": "Faithfulness Evaluation",
      "content": "Sufficiency 0.87, comprehensiveness 2.34 nats, correlation 0.68.",
      "metrics": {"sufficiency": 0.87, "comprehensiveness": 2.34, "circuit_size": 0.15}
    },
    {
      "title": "Identified Failure Modes",
      "severity": "warning",
      "content": "Expert collapse on code-switching, attention sink on long sequences, SAE dead feature cluster.",
      "recommendations": [
        "Investigate routing stability on code-switching inputs",
        "Consider StreamingLLM-style sink token handling",
        "Re-initialize dead features via ghost gradient trick"
      ]
    }
  ],
  "summary": {
    "overall_status": "WARN",
    "n_sections": 5, "n_critical": 0, "n_warnings": 1,
    "recommendations": ["Add interpretability_score ≥ 0.6 as soft constraint in eval gate"]
  }
}
```

---

## 7. Integration with FMS Pipeline

### Extending the Eval Gate (Doc 17) with Interpretability Metrics

```
Existing Gate (Doc 17):              Extended Gate:
  HARD:                                HARD:
    safety ≥ baseline                    safety ≥ baseline
    p95_latency ≤ budget                 p95_latency ≤ budget
    cost ≤ budget                        cost ≤ budget
  SOFT:                                  expert_consistency ≥ 0.3  (NEW)
    quality improvement                SOFT:
    throughput improvement               quality improvement
                                         throughput improvement
                                         interpretability_score ≥ 0.6  (NEW)
                                         no faithfulness regression     (NEW)
```

### Gate Integration Code

```python
def check_interpretability(interp_results, baseline_results=None, thresholds=None):
    """Check interpretability metrics against thresholds for ship/reject gate.

    Returns {"passed": bool, "hard_failures": [...], "soft_warnings": [...],
             "metrics": {...}, "recommendation": str}
    """
    thresholds = thresholds or {}
    expert_floor = thresholds.get("expert_consistency_floor", 0.3)
    interp_target = thresholds.get("interpretability_target", 0.6)
    faith_tol = thresholds.get("faithfulness_regression_tolerance", 0.05)

    hard, soft = [], []
    ec = interp_results.get("expert_consistency", 0.0)
    if ec < expert_floor:
        hard.append(f"expert_consistency={ec:.3f} < floor={expert_floor} — opaque routing")

    suff = interp_results.get("sufficiency", 0.0)
    fp = interp_results.get("sae_feature_purity", 0.0)
    score = 0.4 * suff + 0.3 * ec + 0.3 * fp
    if score < interp_target:
        soft.append(f"interpretability_score={score:.3f} < target={interp_target}")

    if baseline_results:
        bs = baseline_results.get("sufficiency", 0.0)
        if suff < bs - faith_tol:
            soft.append(f"sufficiency regressed: {bs:.3f} → {suff:.3f}")

    return {"passed": len(hard) == 0, "hard_failures": hard, "soft_warnings": soft,
            "metrics": {"expert_consistency": ec, "interpretability_score": round(score, 4)},
            "recommendation": "REJECT" if hard else ("SHIP with warnings" if soft else "SHIP")}
```

### Regression Detection Across Training

Track interpretability metrics at each checkpoint to detect degradation:

```
ckpt_0 → ckpt_5k → ckpt_10k → ckpt_20k → SFT → DPO
Sufficiency:  0.42    0.61      0.74      0.82   0.87  0.87  ✅ monotonic
Expert Cons:  0.55    0.58      0.65      0.70   0.72  0.72  ✅ stable
Dead Feats:   0.45    0.22      0.10      0.05   0.03  0.03  ✅ improving

Key questions:
  - Does SFT damage interpretability? (routing structure may change)
  - Does DPO damage interpretability? (preference ≠ interpretability)
  - Are early circuits preserved at final checkpoint?
```

---

## 8. Gotchas & Edge Cases

### 8a. Faithfulness Metrics Can Be Gamed by Trivially Large Circuits

A circuit containing 95% of components achieves sufficiency ≈ 1.0 trivially. **Always report minimality alongside faithfulness.** The correct evaluation object is the Pareto frontier of sufficiency vs. minimality. Report the "knee" — the smallest circuit achieving >80% sufficiency.

### 8b. Human Evaluation Is Expensive — Use Proxy Metrics for CI/CD

A single human eval batch costs $50–200 and takes hours. **Use automatic metrics for CI/CD gates**, reserve human evaluation for milestone releases and publication-quality results. Cadence: automatic on every commit; spot-check weekly; full human eval at milestones.

### 8c. SAE Reconstruction Loss Doesn't Guarantee Feature Quality

Low reconstruction loss (0.001 MSE) can coexist with uninterpretable features. Reconstruction measures information preservation, not disentanglement. Always evaluate feature purity and dead features alongside reconstruction.

| Metric | Healthy | Concerning | Failure |
|---|---|---|---|
| Dead features | <5% | 5–20% | >20% |
| Feature purity | >0.7 | 0.5–0.7 | <0.5 |
| Mean L0 | 10–100 | 100–500 | >500 or <5 |

### 8d. Expert Consistency Varies Dramatically Between Domains

Expert consistency of 0.72 on English Wikipedia ≠ 0.72 on code, legal text, or mixed input. MoE routing is domain-sensitive. **Always report the evaluation domain.** Example: same model scores 0.85 on Python, 0.58 on legal, 0.31 on mixed code+prose.

### 8e. Report File Sizes Can Be Enormous with Embedded Visualizations

HTML reports with 16 attention heatmaps + 64 expert diagrams + 100 SAE dashboards can exceed 200MB. Mitigate with: SVG over PNG (10–50× smaller), lazy loading, external figure references for JSON/Markdown, configurable resolution (128×128 for CI, full-res for publication).

### 8f. Seed Variance in Interpretability Metrics Requires Bootstrap CIs

Never report a single value without uncertainty. All metric classes compute bootstrap CIs (1000 resamples, 95% CI). Rule of thumb: CI width <5% = stable; 5–15% = noisy; >15% = unreliable (increase n_samples). For SAE features, train 3+ SAEs with different seeds — a feature is only real if it appears in >2/3 of runs.

### 8g. Faithfulness Correlation Assumes Monotonic Relationships

Spearman correlation fails when attribution→effect is non-monotonic (feature interactions). Supplement with **ablation path analysis**: sweep ablated components from 0 to N, plot degradation curve — faithful attributions degrade faster when ablating high-attribution components first.

### 8h. Counterfactual Validity Requires Pre-Registration

Post-hoc rationalization: ablating any circuit degrades *something*. Pre-registration protocol: before running an intervention, record the predicted effect and predicted null effect. Only count timestamp-verified predictions in the counterfactual validity score.

---

*"Interpretability without evaluation is literature. Evaluation without reporting is ephemeral. Reporting without integration is academic. Only when the interpretability pipeline feeds into the ship/reject gate — when opaque expert routing blocks a deployment, when faithfulness regression triggers a review — does interpretability become engineering. And engineering is what ships."*

— Principal Engineer's Note, Foundation Models Division, 2026
