"""
NanoSeek Phase 3: Ablation Analysis & Hypothesis Testing.

Pulls completed runs from W&B, compares against pre-registered hypotheses,
generates a summary table for the paper, and fits scaling laws.

Usage:
    # Analyze stability ablations
    python -m nanoseek.scripts.analyze_ablations --group stability-anchor

    # Analyze HP transfer (muP validation)
    python -m nanoseek.scripts.analyze_ablations --group hp-anchor --hp-transfer

    # Full Phase 3 report (all groups)
    python -m nanoseek.scripts.analyze_ablations --full-report

    # Export results to JSON for the paper
    python -m nanoseek.scripts.analyze_ablations --full-report --output results/phase3_results.json
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ═══════════════════════════════════════════════════════════════════
# Pre-Registered Hypotheses (from TRAINING_PLAN_PHASE3.md Section 2)
# ═══════════════════════════════════════════════════════════════════
# These are FROZEN before seeing any results. Do not modify after
# training starts. If a hypothesis is wrong, document it — don't
# retroactively "fix" it. That's how science works.
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Hypothesis:
    """A pre-registered, falsifiable hypothesis."""
    id: str
    group: str           # stability, architecture, hp-transfer
    description: str
    metric: str          # W&B metric key to compare
    baseline_run: str    # run name pattern for baseline
    variant_run: str     # run name pattern for variant
    direction: str       # "higher", "lower", "within"
    threshold: float     # magnitude of expected difference
    unit: str            # "nats", "BPB", "bits", "steps"
    rationale: str       # WHY we expect this

    def check(self, baseline_val: float, variant_val: float) -> dict:
        """Test hypothesis against actual values. Returns verdict dict."""
        diff = variant_val - baseline_val

        if self.direction == "higher":
            passed = diff > self.threshold
            actual_diff = diff
        elif self.direction == "lower":
            passed = diff < -self.threshold
            actual_diff = -diff
        elif self.direction == "within":
            passed = abs(diff) <= self.threshold
            actual_diff = abs(diff)
        else:
            raise ValueError(f"Unknown direction: {self.direction}")

        return {
            "hypothesis_id": self.id,
            "passed": passed,
            "baseline_val": baseline_val,
            "variant_val": variant_val,
            "expected_diff": self.threshold,
            "actual_diff": actual_diff,
            "direction": self.direction,
            "unit": self.unit,
            "verdict": "CONFIRMED" if passed else "FALSIFIED",
        }


# Pre-registered hypotheses — DO NOT MODIFY after training starts
HYPOTHESES = [
    # ─── Stability ablations ───
    Hypothesis(
        id="stab-A-vs-C",
        group="stability",
        description="Removing seq_aux drops I_spec by >0.1 nats",
        metric="eval/i_spec_mean",
        baseline_run="stab-A",
        variant_run="stab-C",
        direction="lower",
        threshold=0.1,
        unit="nats",
        rationale="seq_aux encourages load balance → diverse expert roles → higher I_spec",
    ),
    Hypothesis(
        id="stab-A-vs-D",
        group="stability",
        description="Removing grad clip causes a loss spike at step ~500-1000",
        metric="train/loss",
        baseline_run="stab-A",
        variant_run="stab-D",
        direction="higher",  # spike means higher loss
        threshold=0.5,       # spike should be noticeable
        unit="BPB",
        rationale="MoE gradient variance without clip → occasional spikes in bf16",
    ),
    Hypothesis(
        id="stab-A-vs-E",
        group="stability",
        description="Aux-loss-free achieves I_spec >0.1 nats higher than classic aux loss",
        metric="eval/i_spec_mean",
        baseline_run="stab-A",
        variant_run="stab-E",
        direction="higher",
        threshold=0.1,
        unit="nats",
        rationale="Classic aux loss penalizes specialization; bias-based doesn't",
    ),
    Hypothesis(
        id="stab-F-recovery",
        group="stability",
        description="Recovery within 50 steps after 10x gradient injection",
        metric="_custom_spike_recovery",  # needs custom logic
        baseline_run="stab-A",
        variant_run="stab-F",
        direction="within",
        threshold=50,
        unit="steps",
        rationale="Grad clip + bias reset provides robustness",
    ),

    # ─── Architecture ablations ───
    Hypothesis(
        id="arch-no-mtp",
        group="architecture",
        description="Disabling MTP increases loss by ~0.05 BPB",
        metric="ema_val/bpb",
        baseline_run="stab-A",
        variant_run="arch-no-mtp",
        direction="higher",
        threshold=0.02,  # conservative lower bound
        unit="BPB",
        rationale="MTP provides training signal (next-token auxiliary task)",
    ),
    Hypothesis(
        id="arch-no-shared",
        group="architecture",
        description="Removing shared experts increases loss by ~0.02-0.04 BPB",
        metric="ema_val/bpb",
        baseline_run="stab-A",
        variant_run="arch-no-shared",
        direction="higher",
        threshold=0.01,  # conservative lower bound
        unit="BPB",
        rationale="Shared experts handle common patterns across all tokens",
    ),
    Hypothesis(
        id="arch-no-mla",
        group="architecture",
        description="MLA vs MHA: approximately equal loss at 4K context",
        metric="ema_val/bpb",
        baseline_run="stab-A",
        variant_run="arch-no-mla",
        direction="within",
        threshold=0.02,
        unit="BPB",
        rationale="MLA saves KV cache, not training quality at short context",
    ),

    # ─── HP transfer ───
    Hypothesis(
        id="mup-transfer",
        group="hp-transfer",
        description="muP-scaled anchor HP achieves ema_val_bpb within 0.02 of 500M grid optimum",
        metric="ema_val/bpb",
        baseline_run="hp-500m-transfer",
        variant_run="hp-500m-grid",  # best grid run
        direction="within",
        threshold=0.02,
        unit="BPB",
        rationale="muP theory: √B × 1/width scaling preserves HP optima across widths",
    ),
]


def fetch_wandb_runs(project: str = "nanoseek", group: Optional[str] = None,
                     tags: Optional[List[str]] = None) -> List[dict]:
    """Fetch completed runs from W&B API."""
    if not HAS_WANDB:
        print("ERROR: wandb not installed. Install with: pip install wandb")
        sys.exit(1)

    api = wandb.Api()
    filters = {"state": "finished"}
    if group:
        filters["group"] = group

    runs = api.runs(project, filters=filters)
    results = []

    for run in runs:
        # Skip if tag filter specified and not matched
        if tags and not any(t in run.tags for t in tags):
            continue

        run_data = {
            "name": run.name,
            "id": run.id,
            "group": run.group,
            "tags": run.tags,
            "config": dict(run.config),
            "summary": dict(run.summary),
            "created_at": str(run.created_at),
            "runtime_seconds": run.summary.get("_runtime", 0),
        }
        results.append(run_data)

    return results


def find_run(runs: List[dict], name_pattern: str) -> Optional[dict]:
    """Find a run by name prefix match."""
    matches = [r for r in runs if r["name"].startswith(name_pattern)]
    if not matches:
        return None
    # If multiple matches, take the most recent
    return sorted(matches, key=lambda r: r["created_at"], reverse=True)[0]


def get_final_metric(run: dict, metric: str) -> Optional[float]:
    """Get the final value of a metric from a run's summary."""
    return run["summary"].get(metric)


def test_hypotheses(runs: List[dict], group: Optional[str] = None) -> List[dict]:
    """Test all pre-registered hypotheses against actual run data."""
    results = []

    for hyp in HYPOTHESES:
        if group and hyp.group != group:
            continue

        baseline = find_run(runs, hyp.baseline_run)
        variant = find_run(runs, hyp.variant_run)

        if baseline is None or variant is None:
            results.append({
                "hypothesis_id": hyp.id,
                "verdict": "INCOMPLETE",
                "reason": f"Missing runs: baseline={hyp.baseline_run} "
                          f"({'found' if baseline else 'MISSING'}), "
                          f"variant={hyp.variant_run} "
                          f"({'found' if variant else 'MISSING'})",
            })
            continue

        # Special handling for spike recovery (stab-F)
        if hyp.metric == "_custom_spike_recovery":
            results.append({
                "hypothesis_id": hyp.id,
                "verdict": "MANUAL_CHECK",
                "reason": "Spike recovery requires time-series analysis. "
                          "Check W&B: compare loss curves around injection step.",
                "baseline_run": baseline["name"],
                "variant_run": variant["name"],
            })
            continue

        baseline_val = get_final_metric(baseline, hyp.metric)
        variant_val = get_final_metric(variant, hyp.metric)

        if baseline_val is None or variant_val is None:
            results.append({
                "hypothesis_id": hyp.id,
                "verdict": "METRIC_MISSING",
                "reason": f"Metric '{hyp.metric}' not found in "
                          f"baseline={baseline_val}, variant={variant_val}",
            })
            continue

        verdict = hyp.check(baseline_val, variant_val)
        verdict["description"] = hyp.description
        verdict["rationale"] = hyp.rationale
        results.append(verdict)

    return results


def find_best_hp_run(runs: List[dict], prefix: str = "hp-anchor") -> Optional[dict]:
    """Find the HP grid run with lowest final ema_val_bpb."""
    hp_runs = [r for r in runs if r["name"].startswith(prefix)]
    if not hp_runs:
        return None

    best = None
    best_bpb = float("inf")
    for run in hp_runs:
        bpb = get_final_metric(run, "ema_val/bpb")
        if bpb is not None and bpb < best_bpb:
            best_bpb = bpb
            best = run

    return best


def generate_comparison_table(runs: List[dict], group: str) -> str:
    """Generate a markdown comparison table for a group of runs."""
    group_runs = [r for r in runs if r.get("group") == group]
    if not group_runs:
        return f"No runs found for group: {group}\n"

    # Collect all metrics of interest
    metrics = ["ema_val/bpb", "eval/i_spec_mean", "train/H_load",
               "eval/dead_expert_count", "eval/mtp_acceptance_rate"]

    lines = []
    lines.append(f"## {group}")
    lines.append("")

    # Header
    header = "| Run | " + " | ".join(m.split("/")[-1] for m in metrics) + " | Runtime |"
    sep = "|-----|" + "|".join(["------"] * len(metrics)) + "|---------|"
    lines.append(header)
    lines.append(sep)

    for run in sorted(group_runs, key=lambda r: r["name"]):
        vals = []
        for m in metrics:
            v = get_final_metric(run, m)
            if v is not None:
                vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
            else:
                vals.append("—")
        runtime = run.get("runtime_seconds", 0)
        runtime_str = f"{runtime / 60:.1f}m" if runtime < 3600 else f"{runtime / 3600:.1f}h"
        lines.append(f"| {run['name']} | " + " | ".join(vals) + f" | {runtime_str} |")

    lines.append("")
    return "\n".join(lines)


def generate_cost_report(runs: List[dict]) -> str:
    """Estimate GPU cost from runtime and scale."""
    # Approximate $/hr by GPU type (RunPod 2026 pricing)
    cost_per_hour = {
        "anchor": 0.44,   # 1x RTX 4090
        "500m": 3.29,     # 1x H100
        "1b": 26.32,      # 8x H100
    }

    lines = ["## Cost Report", ""]
    lines.append("| Run | Scale | Runtime | Est. Cost |")
    lines.append("|-----|-------|---------|-----------|")

    total_cost = 0.0
    for run in sorted(runs, key=lambda r: r["name"]):
        scale = run["config"].get("scale", "anchor")
        runtime_s = run.get("runtime_seconds", 0)
        runtime_h = runtime_s / 3600
        cost = runtime_h * cost_per_hour.get(scale, 1.0)
        total_cost += cost
        lines.append(f"| {run['name']} | {scale} | {runtime_h:.2f}h | ${cost:.2f} |")

    lines.append(f"| **TOTAL** | | | **${total_cost:.2f}** |")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze NanoSeek Phase 3 ablation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test stability hypotheses
  python -m nanoseek.scripts.analyze_ablations --group stability-anchor

  # Find best HP from anchor grid search
  python -m nanoseek.scripts.analyze_ablations --group hp-anchor --hp-transfer

  # Full Phase 3 report
  python -m nanoseek.scripts.analyze_ablations --full-report --output results/phase3.json
        """,
    )
    parser.add_argument("--project", default="nanoseek", help="W&B project name")
    parser.add_argument("--group", default=None, help="Filter by W&B group")
    parser.add_argument("--hp-transfer", action="store_true",
                        help="Analyze HP grid search and muP transfer")
    parser.add_argument("--full-report", action="store_true",
                        help="Generate complete Phase 3 analysis")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--offline", default=None,
                        help="Load runs from local JSON file instead of W&B API")

    args = parser.parse_args()

    # Fetch runs
    if args.offline:
        with open(args.offline, "r") as f:
            runs = json.load(f)
        print(f"Loaded {len(runs)} runs from {args.offline}")
    else:
        print(f"Fetching runs from W&B project '{args.project}'...")
        runs = fetch_wandb_runs(args.project, group=args.group)
        print(f"Found {len(runs)} completed runs")

    if not runs:
        print("No runs found. Check W&B project name and filters.")
        return

    report = {}

    # HP transfer analysis
    if args.hp_transfer or args.full_report:
        print("\n" + "=" * 60)
        print("HP GRID SEARCH ANALYSIS")
        print("=" * 60)
        best = find_best_hp_run(runs)
        if best:
            print(f"Best anchor HP run: {best['name']}")
            print(f"  ema_val_bpb: {get_final_metric(best, 'ema_val/bpb'):.4f}")
            print(f"  matrix_lr:   {best['config'].get('matrix_lr')}")
            print(f"  embedding_lr: {best['config'].get('embedding_lr')}")
            report["best_hp"] = {
                "run": best["name"],
                "ema_val_bpb": get_final_metric(best, "ema_val/bpb"),
                "matrix_lr": best["config"].get("matrix_lr"),
                "embedding_lr": best["config"].get("embedding_lr"),
            }
        else:
            print("No HP grid runs found.")

    # Hypothesis testing
    if args.group or args.full_report:
        print("\n" + "=" * 60)
        print("HYPOTHESIS TESTING")
        print("=" * 60)
        group_filter = args.group.split("-")[0] if args.group else None
        verdicts = test_hypotheses(runs, group=group_filter)
        report["hypotheses"] = verdicts

        for v in verdicts:
            status = v["verdict"]
            marker = {"CONFIRMED": "✓", "FALSIFIED": "✗",
                       "INCOMPLETE": "?", "MANUAL_CHECK": "⚠",
                       "METRIC_MISSING": "!"}
            print(f"  [{marker.get(status, '?')}] {v['hypothesis_id']}: {status}")
            if "description" in v:
                print(f"      {v['description']}")
            if "actual_diff" in v:
                print(f"      expected: {v['direction']} {v['expected_diff']} {v.get('unit', '')}, "
                      f"actual: {v['actual_diff']:.4f}")
            if "reason" in v:
                print(f"      {v['reason']}")

    # Comparison tables
    if args.full_report:
        print("\n" + "=" * 60)
        print("COMPARISON TABLES")
        print("=" * 60)

        groups = set(r.get("group", "") for r in runs if r.get("group"))
        tables = {}
        for g in sorted(groups):
            table = generate_comparison_table(runs, g)
            tables[g] = table
            print(table)

        report["comparison_tables"] = tables

        # Cost report
        cost = generate_cost_report(runs)
        print(cost)
        report["cost_report"] = cost

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
