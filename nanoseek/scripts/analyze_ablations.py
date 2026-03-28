"""
NanoSeek Phase 3: Ablation Analysis & Hypothesis Testing.

Pulls completed runs from W&B, compares against pre-registered hypotheses,
generates a summary table for the paper, and fits scaling laws.

Usage:
    # Analyze stability ablations
    python -m nanoseek.scripts.analyze_ablations --group stability-anchor

    # Analyze HP transfer (muP validation) — single-seed Round 1/2
    python -m nanoseek.scripts.analyze_ablations --group hp-anchor --hp-transfer

    # Analyze HP transfer with multi-seed aggregation (Round 3)
    python -m nanoseek.scripts.analyze_ablations --group hp-anchor --hp-transfer --multiseed

    # Full Phase 3 report (all groups)
    python -m nanoseek.scripts.analyze_ablations --full-report

    # Export results to JSON for the paper
    python -m nanoseek.scripts.analyze_ablations --full-report --output results/phase3_results.json
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
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
        baseline_run="hp-ablation-transfer",
        variant_run="hp-ablation-grid",  # best grid run
        direction="within",
        threshold=0.02,
        unit="BPB",
        rationale="muP theory: √B × 1/width scaling preserves HP optima across widths",
    ),
]


# ═══════════════════════════════════════════════════════════════════
# W&B Data Fetching
# ═══════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════
# HP Grid Search Analysis — Multi-Round, Multi-Seed
# ═══════════════════════════════════════════════════════════════════

def _parse_hp_run_name(name: str) -> Optional[dict]:
    """Extract (round, matrix_lr, embedding_lr, seed, weight_decay) from run name.

    Supported patterns:
      hp-r1-mlr0.02-elr0.3          → round=1, seed=42 (implicit)
      hp-r2-mlr0.025-elr0.225       → round=2, seed=42 (implicit)
      hp-seed-mlr0.02-elr0.3-s137   → round=3 (seed validation)
      hp-wd-mlr0.02-elr0.3-wd0.0    → round=4 (weight decay)
      hp-anchor-mlr0.02-elr0.3      → round=1 (legacy naming)
    """
    parsed = {"round": None, "matrix_lr": None, "embedding_lr": None,
              "seed": 42, "weight_decay": 0.1}

    # Round 3: multi-seed
    m = re.match(r"hp-seed-mlr([\d.]+)-elr([\d.]+)-s(\d+)", name)
    if m:
        parsed["round"] = 3
        parsed["matrix_lr"] = float(m.group(1))
        parsed["embedding_lr"] = float(m.group(2))
        parsed["seed"] = int(m.group(3))
        return parsed

    # Round 4: weight decay
    m = re.match(r"hp-wd-mlr([\d.]+)-elr([\d.]+)-wd([\d.]+)", name)
    if m:
        parsed["round"] = 4
        parsed["matrix_lr"] = float(m.group(1))
        parsed["embedding_lr"] = float(m.group(2))
        parsed["weight_decay"] = float(m.group(3))
        return parsed

    # Round 2: fine grid
    m = re.match(r"hp-r2-mlr([\d.]+)-elr([\d.]+)", name)
    if m:
        parsed["round"] = 2
        parsed["matrix_lr"] = float(m.group(1))
        parsed["embedding_lr"] = float(m.group(2))
        return parsed

    # Round 1: coarse grid (new naming)
    m = re.match(r"hp-r1-mlr([\d.]+)-elr([\d.]+)", name)
    if m:
        parsed["round"] = 1
        parsed["matrix_lr"] = float(m.group(1))
        parsed["embedding_lr"] = float(m.group(2))
        return parsed

    # Legacy naming (hp-anchor-mlr...)
    m = re.match(r"hp-anchor-mlr([\d.]+)-elr([\d.]+)", name)
    if m:
        parsed["round"] = 1
        parsed["matrix_lr"] = float(m.group(1))
        parsed["embedding_lr"] = float(m.group(2))
        return parsed

    return None


def _config_key(mlr: float, elr: float) -> str:
    """Canonical string key for a (matrix_lr, embedding_lr) config."""
    return f"mlr={mlr},elr={elr}"


def find_best_hp_run(runs: List[dict], prefix: str = "hp-") -> Optional[dict]:
    """Find the HP grid run with lowest final ema_val_bpb (single-seed, any round)."""
    hp_runs = [r for r in runs if r["name"].startswith(prefix)
               and _parse_hp_run_name(r["name"]) is not None]
    if not hp_runs:
        return None

    best = None
    best_bpb = float("inf")
    for run in hp_runs:
        parsed = _parse_hp_run_name(run["name"])
        # Skip weight decay and multi-seed runs for basic best-HP selection
        if parsed and parsed["round"] in (3, 4):
            continue
        bpb = get_final_metric(run, "ema_val/bpb")
        if bpb is not None and bpb < best_bpb:
            best_bpb = bpb
            best = run

    return best


def find_top_n_hp_configs(runs: List[dict], n: int = 3) -> List[dict]:
    """Find top-N HP configs by ema_val_bpb from Round 1+2 runs.

    Returns list of dicts: [{"matrix_lr": ..., "embedding_lr": ..., "bpb": ..., "run": ...}]
    sorted by bpb ascending (best first).
    """
    hp_runs = []
    for r in runs:
        parsed = _parse_hp_run_name(r["name"])
        if parsed is None or parsed["round"] not in (1, 2):
            continue
        bpb = get_final_metric(r, "ema_val/bpb")
        if bpb is None:
            continue
        hp_runs.append({
            "matrix_lr": parsed["matrix_lr"],
            "embedding_lr": parsed["embedding_lr"],
            "bpb": bpb,
            "run": r["name"],
            "round": parsed["round"],
        })

    hp_runs.sort(key=lambda x: x["bpb"])
    return hp_runs[:n]


def analyze_multiseed(runs: List[dict]) -> dict:
    """Aggregate Round 3 multi-seed runs into mean ± std per config.

    Returns dict keyed by config_key → {
        "matrix_lr", "embedding_lr",
        "mean_bpb", "std_bpb", "seeds", "runs",
        "n_seeds", "is_significant" (std < mean diff to next config)
    }
    """
    # Group seed runs by (mlr, elr) config
    config_seeds: Dict[str, List[dict]] = defaultdict(list)

    for r in runs:
        parsed = _parse_hp_run_name(r["name"])
        if parsed is None or parsed["round"] != 3:
            continue
        bpb = get_final_metric(r, "ema_val/bpb")
        if bpb is None:
            continue
        key = _config_key(parsed["matrix_lr"], parsed["embedding_lr"])
        config_seeds[key].append({
            "seed": parsed["seed"],
            "bpb": bpb,
            "run": r["name"],
        })

    if not config_seeds:
        return {}

    # Compute stats per config
    results = {}
    for key, seed_runs in config_seeds.items():
        bpbs = [s["bpb"] for s in seed_runs]
        n = len(bpbs)
        mean_bpb = sum(bpbs) / n
        std_bpb = math.sqrt(sum((b - mean_bpb) ** 2 for b in bpbs) / max(n - 1, 1)) if n > 1 else float("inf")

        # Parse mlr/elr from key
        parts = key.split(",")
        mlr = float(parts[0].split("=")[1])
        elr = float(parts[1].split("=")[1])

        results[key] = {
            "matrix_lr": mlr,
            "embedding_lr": elr,
            "mean_bpb": mean_bpb,
            "std_bpb": std_bpb,
            "n_seeds": n,
            "seeds": sorted([s["seed"] for s in seed_runs]),
            "runs": [s["run"] for s in seed_runs],
            "bpbs": bpbs,
        }

    # Rank by mean_bpb and check significance
    ranked = sorted(results.values(), key=lambda x: x["mean_bpb"])
    for i, cfg in enumerate(ranked):
        cfg["rank"] = i + 1
        if i < len(ranked) - 1:
            next_mean = ranked[i + 1]["mean_bpb"]
            gap = next_mean - cfg["mean_bpb"]
            # Winner is "significant" if gap > std of the winner
            cfg["gap_to_next"] = gap
            cfg["is_significant"] = gap > cfg["std_bpb"]
        else:
            cfg["gap_to_next"] = None
            cfg["is_significant"] = None

    return {_config_key(c["matrix_lr"], c["embedding_lr"]): c for c in ranked}


def analyze_weight_decay(runs: List[dict]) -> List[dict]:
    """Analyze Round 4 weight decay sensitivity runs.

    Returns list of dicts with WD value and bpb, sorted by bpb.
    """
    wd_runs = []
    for r in runs:
        parsed = _parse_hp_run_name(r["name"])
        if parsed is None or parsed["round"] != 4:
            continue
        bpb = get_final_metric(r, "ema_val/bpb")
        if bpb is None:
            continue
        wd_runs.append({
            "weight_decay": parsed["weight_decay"],
            "bpb": bpb,
            "run": r["name"],
            "matrix_lr": parsed["matrix_lr"],
            "embedding_lr": parsed["embedding_lr"],
        })

    wd_runs.sort(key=lambda x: x["bpb"])
    return wd_runs


# ═══════════════════════════════════════════════════════════════════
# Hypothesis Testing
# ═══════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════

def generate_comparison_table(runs: List[dict], group: str) -> str:
    """Generate a markdown comparison table for a group of runs."""
    group_runs = [r for r in runs if r.get("group") == group]
    if not group_runs:
        available = sorted(set(r.get("group", "") for r in runs if r.get("group")))
        return f"No runs found for group: {group}\nAvailable groups: {available}\n"

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


def generate_hp_grid_table(runs: List[dict]) -> str:
    """Generate a dedicated HP grid results table, organized by round."""
    lines = ["## HP Grid Search Results", ""]

    # Collect all HP runs by round
    rounds: Dict[int, List[dict]] = defaultdict(list)
    for r in runs:
        parsed = _parse_hp_run_name(r["name"])
        if parsed is None:
            continue
        bpb = get_final_metric(r, "ema_val/bpb")
        rounds[parsed["round"]].append({
            **parsed,
            "bpb": bpb,
            "run": r["name"],
            "runtime_seconds": r.get("runtime_seconds", 0),
        })

    round_names = {1: "Round 1 (Coarse Grid)", 2: "Round 2 (Fine Grid)",
                   3: "Round 3 (Multi-Seed)", 4: "Round 4 (Weight Decay)"}

    for round_num in sorted(rounds.keys()):
        round_runs = rounds[round_num]
        lines.append(f"### {round_names.get(round_num, f'Round {round_num}')}")
        lines.append("")

        if round_num in (1, 2):
            lines.append("| Run | matrix_lr | embedding_lr | ema_val_bpb | Runtime |")
            lines.append("|-----|-----------|-------------|-------------|---------|")
            for entry in sorted(round_runs, key=lambda x: x.get("bpb") or float("inf")):
                bpb_str = f"{entry['bpb']:.4f}" if entry['bpb'] is not None else "—"
                rt = entry["runtime_seconds"]
                rt_str = f"{rt / 60:.1f}m" if rt < 3600 else f"{rt / 3600:.1f}h"
                lines.append(f"| {entry['run']} | {entry['matrix_lr']} | "
                            f"{entry['embedding_lr']} | {bpb_str} | {rt_str} |")

        elif round_num == 3:
            lines.append("| Run | matrix_lr | embedding_lr | seed | ema_val_bpb |")
            lines.append("|-----|-----------|-------------|------|-------------|")
            for entry in sorted(round_runs, key=lambda x: (x["matrix_lr"], x["embedding_lr"], x["seed"])):
                bpb_str = f"{entry['bpb']:.4f}" if entry['bpb'] is not None else "—"
                lines.append(f"| {entry['run']} | {entry['matrix_lr']} | "
                            f"{entry['embedding_lr']} | {entry['seed']} | {bpb_str} |")

            # Add aggregated stats
            multiseed = analyze_multiseed(runs)
            if multiseed:
                lines.append("")
                lines.append("**Multi-Seed Aggregation:**")
                lines.append("")
                lines.append("| Config | mean(bpb) | std(bpb) | n_seeds | Gap to Next | Significant? |")
                lines.append("|--------|-----------|---------|---------|-------------|-------------|")
                for key, stats in sorted(multiseed.items(), key=lambda x: x[1]["mean_bpb"]):
                    gap_str = f"{stats['gap_to_next']:.4f}" if stats['gap_to_next'] is not None else "—"
                    sig_str = "YES" if stats['is_significant'] else ("NO" if stats['is_significant'] is not None else "—")
                    lines.append(f"| {key} | {stats['mean_bpb']:.4f} | {stats['std_bpb']:.4f} | "
                                f"{stats['n_seeds']} | {gap_str} | {sig_str} |")

        elif round_num == 4:
            lines.append("| Run | weight_decay | ema_val_bpb |")
            lines.append("|-----|-------------|-------------|")
            for entry in sorted(round_runs, key=lambda x: x.get("bpb") or float("inf")):
                bpb_str = f"{entry['bpb']:.4f}" if entry['bpb'] is not None else "—"
                lines.append(f"| {entry['run']} | {entry['weight_decay']} | {bpb_str} |")

        lines.append("")

    return "\n".join(lines)


def generate_cost_report(runs: List[dict]) -> str:
    """Estimate GPU cost from runtime and scale."""
    # Approximate $/hr by GPU type (RunPod 2026 pricing)
    cost_per_hour = {
        "anchor": 0.44,   # 1x RTX 4090
        "ablation": 3.29,     # 1x H100
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


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Analyze NanoSeek Phase 3 ablation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test stability hypotheses
  python -m nanoseek.scripts.analyze_ablations --group stability-anchor

  # Find best HP from anchor grid search (Round 1+2)
  python -m nanoseek.scripts.analyze_ablations --group hp-anchor --hp-transfer

  # Multi-seed analysis (Round 3)
  python -m nanoseek.scripts.analyze_ablations --group hp-anchor --hp-transfer --multiseed

  # Full Phase 3 report
  python -m nanoseek.scripts.analyze_ablations --full-report --output results/phase3.json
        """,
    )
    parser.add_argument("--project", default="nanoseek", help="W&B project name")
    parser.add_argument("--group", default=None, help="Filter by W&B group")
    parser.add_argument("--hp-transfer", action="store_true",
                        help="Analyze HP grid search and muP transfer")
    parser.add_argument("--multiseed", action="store_true",
                        help="Include multi-seed aggregation in HP analysis (Round 3)")
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

        # ─── Round 1+2: Best single-seed run ───
        best = find_best_hp_run(runs)
        if best:
            parsed = _parse_hp_run_name(best["name"])
            print(f"\nBest single-seed HP run: {best['name']}")
            print(f"  ema_val_bpb:  {get_final_metric(best, 'ema_val/bpb'):.4f}")
            print(f"  matrix_lr:    {parsed['matrix_lr'] if parsed else best['config'].get('matrix_lr')}")
            print(f"  embedding_lr: {parsed['embedding_lr'] if parsed else best['config'].get('embedding_lr')}")
            report["best_hp_single_seed"] = {
                "run": best["name"],
                "ema_val_bpb": get_final_metric(best, "ema_val/bpb"),
                "matrix_lr": parsed["matrix_lr"] if parsed else best["config"].get("matrix_lr"),
                "embedding_lr": parsed["embedding_lr"] if parsed else best["config"].get("embedding_lr"),
            }
        else:
            print("No HP grid runs found (looking for hp-r1-*, hp-r2-*, hp-anchor-* prefixes).")

        # ─── Top-3 configs for seed validation ───
        top3 = find_top_n_hp_configs(runs, n=3)
        if top3:
            print(f"\nTop-3 configs (for multi-seed validation):")
            for i, cfg in enumerate(top3):
                print(f"  #{i+1}: mlr={cfg['matrix_lr']}, elr={cfg['embedding_lr']} "
                      f"→ bpb={cfg['bpb']:.4f} (R{cfg['round']}: {cfg['run']})")
            # Print the TOP3_CONFIGS env var for convenience
            top3_str = ";".join(f"{c['matrix_lr']},{c['embedding_lr']}" for c in top3)
            print(f"\n  Export for hp-seeds: TOP3_CONFIGS='{top3_str}'")
            report["top3_configs"] = top3

        # ─── Round 3: Multi-seed aggregation ───
        if args.multiseed or args.full_report:
            multiseed = analyze_multiseed(runs)
            if multiseed:
                print(f"\nMulti-Seed Results ({len(multiseed)} configs):")
                ranked = sorted(multiseed.values(), key=lambda x: x["mean_bpb"])
                for cfg in ranked:
                    sig_marker = " ***" if cfg["is_significant"] else ""
                    seeds_str = ",".join(str(s) for s in cfg["seeds"])
                    print(f"  #{cfg['rank']}: mlr={cfg['matrix_lr']}, elr={cfg['embedding_lr']} "
                          f"→ mean={cfg['mean_bpb']:.4f} ± {cfg['std_bpb']:.4f} "
                          f"(seeds: {seeds_str}){sig_marker}")

                winner = ranked[0]
                print(f"\n  CONFIRMED WINNER: mlr={winner['matrix_lr']}, elr={winner['embedding_lr']}")
                print(f"    mean(bpb) = {winner['mean_bpb']:.4f} ± {winner['std_bpb']:.4f}")
                if winner["is_significant"]:
                    print(f"    Gap to #2 ({winner['gap_to_next']:.4f}) > std ({winner['std_bpb']:.4f}) → SIGNIFICANT")
                elif winner["is_significant"] is not None:
                    print(f"    Gap to #2 ({winner['gap_to_next']:.4f}) ≤ std ({winner['std_bpb']:.4f}) "
                          f"→ NOT SIGNIFICANT (pick lower LR = more conservative)")

                report["multiseed"] = {k: {**v, "bpbs": v["bpbs"]} for k, v in multiseed.items()}
                report["confirmed_winner"] = {
                    "matrix_lr": winner["matrix_lr"],
                    "embedding_lr": winner["embedding_lr"],
                    "mean_bpb": winner["mean_bpb"],
                    "std_bpb": winner["std_bpb"],
                    "is_significant": winner["is_significant"],
                }
            else:
                print("\nNo multi-seed runs found (looking for hp-seed-* prefix).")

        # ─── Round 4: Weight decay sensitivity ───
        wd_results = analyze_weight_decay(runs)
        if wd_results:
            print(f"\nWeight Decay Sensitivity:")
            for wd_run in wd_results:
                print(f"  WD={wd_run['weight_decay']}: bpb={wd_run['bpb']:.4f} ({wd_run['run']})")

            # Compare to WD=0.1 baseline from seed runs
            baseline_wd01 = None
            if "confirmed_winner" in report:
                mlr = report["confirmed_winner"]["matrix_lr"]
                elr = report["confirmed_winner"]["embedding_lr"]
                baseline_wd01 = report["confirmed_winner"]["mean_bpb"]
                print(f"  WD=0.1 (baseline, mean of seeds): bpb={baseline_wd01:.4f}")

                for wd_run in wd_results:
                    diff = wd_run["bpb"] - baseline_wd01
                    print(f"  WD={wd_run['weight_decay']} vs WD=0.1: "
                          f"{'better' if diff < 0 else 'worse'} by {abs(diff):.4f} BPB")

            report["weight_decay"] = wd_results

        # ─── HP grid table ───
        hp_table = generate_hp_grid_table(runs)
        if hp_table.strip():
            print(f"\n{hp_table}")
            report["hp_grid_table"] = hp_table

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
