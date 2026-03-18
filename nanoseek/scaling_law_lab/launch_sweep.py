#!/usr/bin/env python3
"""
Launch sweep runs from YAML config files.

Usage:
    # Launch all Series A configs sequentially
    python -m scaling_law_lab.launch_sweep configs/scaling_sweep/series_a_scale.yaml

    # Launch a specific config within a multi-config YAML
    python -m scaling_law_lab.launch_sweep configs/scaling_sweep/series_a_scale.yaml --only nano_80M

    # Dry run (print commands without executing)
    python -m scaling_law_lab.launch_sweep configs/scaling_sweep/series_a_scale.yaml --dry-run

    # Launch with custom prefix
    python -m scaling_law_lab.launch_sweep configs/scaling_sweep/series_b_experts.yaml --prefix scaling-b
"""

import argparse
import subprocess
import sys
import yaml
from pathlib import Path


def build_cli_args(config_name: str, cfg: dict, base: dict | None = None, prefix: str = "") -> list[str]:
    """Convert a YAML config dict into CLI arguments for pre_train.py."""
    args = ["python", "-m", "nanoseek.scripts.pre_train"]

    # Run name
    run_name = cfg.get("run_name", f"{prefix}-{config_name}" if prefix else config_name)
    args.extend(["--run", run_name])

    # Scale (default to anchor for sweep configs)
    scale = cfg.get("scale", "anchor")
    args.extend(["--scale", scale])

    # Seed
    seed = cfg.get("seed", base.get("seed", 42) if base else 42)
    args.extend(["--seed", str(seed)])

    # Top-level overrides
    if "total_tokens" in cfg:
        args.extend(["--total-tokens", str(cfg["total_tokens"])])
    if "hidden_size" in cfg:
        args.extend(["--hidden-size", str(cfg["hidden_size"])])

    # MoE overrides
    moe = cfg.get("moe", {})
    if "n_routed_experts" in moe:
        args.extend(["--num-experts", str(moe["n_routed_experts"])])
    if "num_experts_per_tok" in moe:
        args.extend(["--top-k", str(moe["num_experts_per_tok"])])
    if "n_group" in moe:
        args.extend(["--n-group", str(moe["n_group"])])
    if "topk_group" in moe:
        args.extend(["--topk-group", str(moe["topk_group"])])

    # Eval and save
    eval_every = cfg.get("eval_every", base.get("eval_every", 250) if base else 250)
    save_every = cfg.get("save_every", base.get("save_every", -1) if base else -1)
    args.extend(["--eval-every", str(eval_every)])
    args.extend(["--save-every", str(save_every)])

    # Training overrides
    if "matrix_lr" in cfg:
        args.extend(["--matrix-lr", str(cfg["matrix_lr"])])
    if "embedding_lr" in cfg:
        args.extend(["--embedding-lr", str(cfg["embedding_lr"])])

    # Ablation flags
    if cfg.get("no_seq_aux"):
        args.append("--no-seq-aux")
    if cfg.get("no_grad_clip"):
        args.append("--no-grad-clip")
    if cfg.get("no_mtp"):
        args.append("--no-mtp")
    if cfg.get("no_shared_experts"):
        args.append("--no-shared-experts")
    if "aux_loss_type" in cfg:
        args.extend(["--aux-loss-type", cfg["aux_loss_type"]])
    if "inject_bad_batch" in cfg and cfg["inject_bad_batch"] >= 0:
        args.extend(["--inject-bad-batch", str(cfg["inject_bad_batch"])])

    # Full YAML config path (for overrides not covered by CLI flags)
    # The --config-yaml flag handles mla, mtp, moe_intermediate_size, etc.
    # We write a temporary per-run YAML for this.
    return args, cfg


def main():
    parser = argparse.ArgumentParser(description="Launch scaling law sweep from YAML")
    parser.add_argument("yaml_path", type=str, help="Path to sweep YAML config")
    parser.add_argument("--only", type=str, default="", help="Run only this config name")
    parser.add_argument("--prefix", type=str, default="", help="Run name prefix")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--temp-dir", type=str, default="/tmp/nanoseek_sweep",
                        help="Directory for temporary per-run YAML configs")
    args = parser.parse_args()

    with open(args.yaml_path, 'r') as f:
        sweep = yaml.safe_load(f)

    configs = sweep.get("configs", {})
    base = sweep.get("base", {})

    if not configs:
        print(f"No 'configs' section found in {args.yaml_path}")
        sys.exit(1)

    # Create temp dir for per-run YAML files
    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    for name, cfg in configs.items():
        if args.only and name != args.only:
            continue

        # Merge base config with per-run config (per-run takes precedence)
        merged = {**base, **cfg}
        for section in ['moe', 'mla', 'mtp']:
            if section in base and section in cfg:
                merged[section] = {**base[section], **cfg[section]}
            elif section in base:
                merged[section] = base[section]

        cli_args, full_cfg = build_cli_args(name, merged, base, args.prefix)

        # Write per-run YAML for --config-yaml (handles mla, mtp overrides)
        per_run_yaml = temp_dir / f"{name}.yaml"
        with open(per_run_yaml, 'w') as f:
            yaml.dump(full_cfg, f, default_flow_style=False)
        cli_args.extend(["--config-yaml", str(per_run_yaml)])

        cmd = " ".join(cli_args)
        print(f"\n{'='*60}")
        print(f"Config: {name}")
        print(f"Command: {cmd}")
        print(f"{'='*60}")

        if not args.dry_run:
            result = subprocess.run(cli_args, cwd=str(Path(__file__).parent.parent))
            if result.returncode != 0:
                print(f"FAILED: {name} (exit code {result.returncode})")
                print("Continuing to next config...")
            else:
                print(f"COMPLETED: {name}")


if __name__ == "__main__":
    main()
