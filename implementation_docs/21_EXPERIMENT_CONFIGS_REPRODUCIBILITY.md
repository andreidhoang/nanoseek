# 21 — Experiment Configuration & Reproducibility System

## Principal Engineer's Implementation Guide — February 2026

**Author**: Distinguished AI Researcher & Principal Engineer, Foundation Models Division
**Scope**: Unified experiment configuration system — YAML-based single source of truth, Pydantic validation, config hashing for deduplication, seed management, diff utilities, and CLI override mechanism
**Prerequisite**: `00_MASTER_PLAN.md` (context), `model/config.py` (NanoSeekConfig dataclass hierarchy), `scripts/pre-train.py` (training loop)
**Criticality**: **MAXIMUM** — An irreproducible experiment is an experiment that never happened

---

## 1. Problem Statement

NanoSeek's training pipeline spans five distinct phases: pre-training, SFT, DPO, evaluation, and serving. Each phase has dozens of hyperparameters — learning rates, batch sizes, LoRA ranks, quantization modes, safety thresholds. Today these parameters live in four separate locations:

| Location | What Lives There | Problem |
|---|---|---|
| `model/config.py` | Architecture config (NanoSeekConfig dataclass) | Python-only, no validation beyond `assert`, no serialization |
| `scripts/pre-train.py` | Training hyperparameters (TrainingConfig dataclass) | Hardcoded defaults, CLI overrides via string parsing |
| Command-line arguments | Per-run overrides (`--model_size=1b --lr=3e-4`) | Ephemeral — lost after the run unless manually logged |
| W&B / TensorBoard | Logged metrics + some config | Post-hoc only — cannot reconstruct exact run from logs |

### What Goes Wrong

**Config drift**: Engineer A trains with `lr=3e-4`, Engineer B trains with `lr=2e-4`. Both claim "NanoSeek-1B results." Neither recorded which config produced which checkpoint. Six months later, the team cannot determine which learning rate produced the published results.

**Irreproducible experiments**: A training run uses `seed=42`, `batch_size=524288`, `lr=3e-4`. The run crashes at step 8,000. The engineer restarts with slightly different settings (larger batch to avoid OOM) but doesn't update the config file. The "reproduced" run is actually a different experiment.

**Scattered hyperparameters**: SFT uses `lora_rank=16`, DPO uses `lora_rank=8`. These values are hardcoded in separate scripts. When someone changes the SFT rank, they forget to update the DPO script. The post-training pipeline silently trains with mismatched architectures.

**No experiment identity**: Two runs with identical configs produce identical checkpoints, but nobody knows they're duplicates. $300 of H100 time is wasted re-running an experiment that already completed successfully.

**Phantom parameters**: A YAML config says `lr: 3e-4`, but a CLI override `--lr=1e-4` silently replaces it. The saved config shows `3e-4` because the override wasn't persisted. The actual training used `1e-4`. This is the most insidious failure mode — the records actively lie.

### What the FMS Lab Plan Requires

A single YAML file per experiment that:

1. Defines **every** hyperparameter for **every** phase (pretrain, SFT, DPO, eval, serving)
2. Is **validated** at load time — type errors, range violations, and cross-field inconsistencies are caught before any GPU is allocated
3. Is **immutable** after experiment launch — the config is hashed and the hash is embedded in the checkpoint
4. Is **diffable** — two experiment configs can be compared field-by-field to see exactly what changed
5. Is **version-controlled** — configs live in `fms/configs/` and are committed to git alongside code

### Targets

- **Config load time**: <100ms (including validation)
- **Hash collision probability**: <1e-18 (SHA-256)
- **Validation coverage**: 100% of fields have type + range checks
- **Reproducibility guarantee**: Same config + same code commit = same training dynamics (modulo hardware non-determinism)

---

## 2. First Principles

### 2a. Single Source of Truth

Every hyperparameter that affects training dynamics must live in exactly one place. If it's in the YAML, it's not also hardcoded in Python. If it's computed from other values, the computation is explicit and deterministic.

```
WRONG (scattered):
  model/config.py:     hidden_size = 2048
  scripts/pre-train.py: lr = 3e-4
  CLI:                  --batch_size=524288

RIGHT (unified):
  fms/configs/pretrain.yaml:
    model:
      hidden_size: 2048
    training:
      lr: 3e-4
      batch_size: 524288
```

### 2b. Config Immutability After Experiment Launch

Once an experiment begins (first optimizer step), the config is frozen. Any modification requires creating a new experiment with a new name. Enforced by hashing the config at launch time (SHA-256 of canonical YAML), embedding the hash in every checkpoint, and validating the hash at resume time.

```python
# At launch:
config_hash = hash_config(config)
checkpoint.metadata["config_hash"] = config_hash

# At resume — refuses to continue if config was modified:
verify_config_hash(config, checkpoint.metadata["config_hash"])
```

### 2c. Seed Management for Reproducibility

Three sources of randomness must be controlled:

| Source | What It Affects | How to Control |
|---|---|---|
| Model initialization | Weight values at step 0 | `torch.manual_seed(seed)` before `model.apply(_init_weights)` |
| Data ordering | Which examples appear in which batch | `generator = torch.Generator().manual_seed(seed)` for dataloader |
| Dropout masks | Which neurons are dropped | `torch.manual_seed(seed + step)` per step for exact reproducibility |

NanoSeek uses `dropout=0.0` (following DeepSeek V3), so dropout seeds are currently irrelevant — but the system handles them for future experiments. **CUDA non-determinism caveat**: even with all seeds set, CUDA atomicAdd and cuBLAS introduce hardware-level non-determinism. The config exposes `experiment.deterministic: bool` to force deterministic algorithms at a ~15% performance cost.

### 2d. Config Inheritance and Overrides

Most experiments share 95% of their config with a base config. The system supports layered overrides:

```
Priority (highest to lowest):
  1. CLI overrides:        --pretrain.lr=1e-4
  2. Experiment YAML:      fms/configs/experiments/lr_sweep.yaml
  3. Base config YAML:     fms/configs/pretrain.yaml (via base_config field)
  4. Schema defaults:      Pydantic model defaults
```

An experiment YAML only needs to specify the fields that differ from defaults.

### 2e. Version Control for Configs

Configs are committed to git alongside the code they configure. Every checkpoint records its config hash and git commit, creating an unambiguous mapping from any checkpoint back to exact code + config.

### 2f. Why YAML Over JSON / TOML / Python

| Format | Comments | Anchors | Readability | Footguns |
|--------|---------|---------|-------------|----------|
| JSON | No | No | Dense, noisy syntax | Trailing commas, no comments |
| TOML | Yes | No | Good for flat configs | Deep nesting is awkward |
| Python | Yes | N/A | Familiar | Turing-complete = unpredictable |
| **YAML** | **Yes** | **Yes** | **Best for hierarchical config** | **Type coercion (see §7)** |

YAML wins because: (1) comments let every hyperparameter have an inline explanation, (2) anchors enable DRY configs, (3) indentation-based nesting maps naturally to dataclass hierarchies, (4) every ML framework already uses YAML. The type coercion footgun is mitigated by Pydantic validation at load time.

---

## 3. Production Code

### 3a. Config Schema — Pydantic Models for Validation

File: `fms/configs/schema.py`

```python
"""
Experiment configuration schema with Pydantic validation.
Every field has a type annotation, default value, and range validation.
"""

from __future__ import annotations
import hashlib, json, os, random
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import numpy as np
import torch
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class OptimizerType(str, Enum):
    ADAMW = "adamw"
    MUON_ADAMW = "muon+adamw"
    SGD = "sgd"

class QuantizationType(str, Enum):
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    FP8 = "fp8"


class ExperimentConfig(BaseModel):
    """Top-level experiment metadata."""
    name: str = Field(..., min_length=1, max_length=128)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    hardware: str = Field(default="8xH100")
    deterministic: bool = Field(default=False)
    base_config: Optional[str] = Field(default=None)
    tags: List[str] = Field(default_factory=list)
    description: str = Field(default="")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        import re
        if not re.match(r'^[a-zA-Z0-9_\-\.]+$', v):
            raise ValueError(f"Experiment name '{v}' contains invalid characters.")
        return v


class PretrainConfig(BaseModel):
    """Pre-training hyperparameters."""
    model_size: str = Field(default="1b")
    total_tokens: int = Field(default=22_000_000_000, gt=0)
    batch_size: int = Field(default=524288, gt=0)
    seq_len: int = Field(default=4096, gt=0, le=131072)
    lr: float = Field(default=3e-4, gt=0, lt=1.0)
    lr_min: float = Field(default=3e-5, ge=0, lt=1.0)
    weight_decay: float = Field(default=0.1, ge=0, le=1.0)
    optimizer: OptimizerType = Field(default=OptimizerType.MUON_ADAMW)
    warmup_steps: int = Field(default=1000, ge=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    checkpoint_every: int = Field(default=1000, gt=0)
    dtype: str = Field(default="bfloat16")
    gradient_checkpointing: bool = Field(default=True)
    compile_model: bool = Field(default=True)

    @model_validator(mode="after")
    def validate_lr_ordering(self) -> "PretrainConfig":
        if self.lr_min >= self.lr:
            raise ValueError(f"lr_min ({self.lr_min}) must be < lr ({self.lr})")
        return self

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        valid = {"bfloat16", "float16", "float32"}
        if v not in valid:
            raise ValueError(f"dtype must be one of {valid}, got '{v}'")
        return v

    @field_validator("batch_size")
    @classmethod
    def validate_batch_alignment(cls, v: int) -> int:
        if v % 1024 != 0:
            raise ValueError(f"batch_size ({v}) should be aligned to 1024")
        return v


class SFTConfig(BaseModel):
    """Supervised Fine-Tuning hyperparameters."""
    dataset: str = Field(default="ultrachat-200k", min_length=1)
    epochs: int = Field(default=3, gt=0, le=100)
    lr: float = Field(default=2e-5, gt=0, lt=1.0)
    batch_size: int = Field(default=4, gt=0)
    seq_len: int = Field(default=2048, gt=0, le=131072)
    lora_rank: int = Field(default=16, gt=0, le=256)
    lora_alpha: float = Field(default=32.0, gt=0)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1.0)
    lora_targets: List[str] = Field(
        default=["wq_a", "wq_b", "wkv_a", "wkv_b", "wo"], min_length=1
    )
    warmup_ratio: float = Field(default=0.03, ge=0, le=1.0)
    weight_decay: float = Field(default=0.0, ge=0, le=1.0)

    @model_validator(mode="after")
    def validate_lora_alpha_rank_ratio(self) -> "SFTConfig":
        ratio = self.lora_alpha / self.lora_rank
        if ratio < 0.5 or ratio > 8.0:
            raise ValueError(
                f"lora_alpha/lora_rank ratio ({ratio:.1f}) is unusual. "
                f"Typical range: 1.0-4.0."
            )
        return self


class DPOConfig(BaseModel):
    """Direct Preference Optimization hyperparameters."""
    dataset: str = Field(default="ultrafeedback-60k", min_length=1)
    epochs: int = Field(default=1, gt=0, le=10)
    lr: float = Field(default=5e-7, gt=0, lt=1e-4)
    beta: float = Field(default=0.1, gt=0, le=1.0)
    batch_size: int = Field(default=2, gt=0)
    seq_len: int = Field(default=2048, gt=0, le=131072)
    ref_model: str = Field(default="sft")
    label_smoothing: float = Field(default=0.0, ge=0, le=0.5)
    max_prompt_length: int = Field(default=512, gt=0)

    @field_validator("beta")
    @classmethod
    def validate_beta_range(cls, v: float) -> float:
        if v < 0.01:
            raise ValueError(
                f"DPO beta ({v}) is dangerously small. Values < 0.01 cause "
                f"reward hacking. Minimum recommended: 0.05"
            )
        return v


class EvalConfig(BaseModel):
    """Evaluation benchmark configuration."""
    benchmarks: List[str] = Field(
        default=["mmlu", "gsm8k", "humaneval", "truthfulqa"], min_length=1
    )
    shots: Dict[str, int] = Field(
        default={"mmlu": 5, "gsm8k": 5, "humaneval": 0, "truthfulqa": 0}
    )
    safety_set: str = Field(default="safety/refusal_50.jsonl")
    leakage_check: bool = Field(default=True)
    eval_batch_size: int = Field(default=8, gt=0)
    max_eval_samples: Optional[int] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_shots_match_benchmarks(self) -> "EvalConfig":
        for benchmark in self.benchmarks:
            if benchmark not in self.shots:
                raise ValueError(
                    f"Benchmark '{benchmark}' missing from shots dict."
                )
        return self


class ServingConfig(BaseModel):
    """Inference serving configuration."""
    quantization: QuantizationType = Field(default=QuantizationType.INT8)
    max_batch_size: int = Field(default=32, gt=0, le=512)
    max_seq_len: int = Field(default=4096, gt=0, le=131072)
    speculative_decoding: bool = Field(default=True)
    port: int = Field(default=8000, ge=1024, le=65535)
    host: str = Field(default="0.0.0.0")
    timeout: int = Field(default=300, gt=0)
    num_workers: int = Field(default=1, gt=0, le=32)


class GateConfig(BaseModel):
    """Ship/reject gate thresholds (see Doc 17)."""
    quality_threshold: str = Field(default="+2% MMLU over baseline")
    quality_min_delta: float = Field(default=0.02, ge=0, le=1.0)
    safety_threshold: str = Field(default=">=baseline refusal accuracy")
    safety_regression_tolerance: float = Field(default=0.0, ge=0, le=0.1)
    p95_latency_budget_ms: float = Field(default=200.0, gt=0)
    cost_budget: str = Field(default="$5/1M tokens")
    cost_per_million_tokens: float = Field(default=5.0, gt=0)

    @model_validator(mode="after")
    def validate_safety_strictness(self) -> "GateConfig":
        if self.safety_regression_tolerance > 0.05:
            raise ValueError(
                f"safety_regression_tolerance ({self.safety_regression_tolerance}) "
                f"is dangerously high. See Doc 17 §2b."
            )
        return self


class FMSConfig(BaseModel):
    """Root configuration — single source of truth for an FMS experiment."""
    experiment: ExperimentConfig
    pretrain: PretrainConfig = Field(default_factory=PretrainConfig)
    sft: SFTConfig = Field(default_factory=SFTConfig)
    dpo: DPOConfig = Field(default_factory=DPOConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    gate: GateConfig = Field(default_factory=GateConfig)

    def to_yaml(self) -> str:
        """Serialize to canonical YAML (sorted keys for deterministic hashing)."""
        return yaml.dump(self.model_dump(mode="json"), sort_keys=True)

    def to_flat_dict(self) -> Dict[str, Any]:
        """Flatten nested config to dot-notation dict for W&B logging."""
        flat: Dict[str, Any] = {}
        _flatten_dict(self.model_dump(mode="json"), "", flat)
        return flat


def _flatten_dict(d: Dict, prefix: str, out: Dict):
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _flatten_dict(v, key, out)
        else:
            out[key] = v
```

### 3b. Config Loader — Loading, Merging, CLI Override

File: `fms/configs/loader.py`

```python
"""
Config loader with inheritance, merging, and CLI override support.

Loading priority (highest to lowest):
  1. CLI overrides:      --pretrain.lr=1e-4
  2. Experiment YAML:    the specified config file
  3. Base config YAML:   via experiment.base_config field
  4. Schema defaults:    Pydantic model defaults
"""

from __future__ import annotations
import copy, sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from fms.configs.schema import FMSConfig


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load YAML with safe loader (no arbitrary Python execution)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(data)}")
    return data


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dicts. Override values take precedence."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _parse_cli_overrides(overrides: List[str]) -> Dict[str, Any]:
    """
    Parse CLI overrides in dot-notation into a nested dict.

    Examples:
        ["pretrain.lr=1e-4", "experiment.seed=123"]
        → {"pretrain": {"lr": 0.0001}, "experiment": {"seed": 123}}
    """
    result: Dict[str, Any] = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected: key=value")
        key, value_str = override.split("=", 1)
        key = key.lstrip("-")
        value = _infer_type(value_str)
        parts = key.split(".")
        current = result
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return result


def _infer_type(value_str: str) -> Any:
    """Infer Python type from CLI string value."""
    if value_str.lower() in ("true", "yes", "on"):
        return True
    if value_str.lower() in ("false", "no", "off"):
        return False
    if value_str.lower() in ("none", "null", "~"):
        return None
    try:
        return int(value_str.replace("_", "")) if "_" in value_str else int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass
    if value_str.startswith("[") and value_str.endswith("]"):
        inner = value_str[1:-1].strip()
        return [_infer_type(i.strip().strip('"').strip("'")) for i in inner.split(",")] if inner else []
    return value_str


def load_config(
    config_path: str | Path,
    overrides: Optional[List[str]] = None,
    config_dir: Optional[str | Path] = None,
) -> FMSConfig:
    """Load, merge, and validate an experiment config."""
    config_path = Path(config_path)
    config_dir = Path(config_dir) if config_dir else config_path.parent

    # Load experiment YAML
    data = _load_yaml(config_path)

    # Merge base config if specified
    base_ref = (data.get("experiment") or {}).get("base_config")
    if base_ref:
        data = _deep_merge(_load_yaml(config_dir / base_ref), data)

    # Apply CLI overrides (highest priority)
    if overrides:
        data = _deep_merge(data, _parse_cli_overrides(overrides))

    # Validate with Pydantic
    return FMSConfig.model_validate(data)


def load_config_from_cli() -> FMSConfig:
    """Load config from sys.argv: <config.yaml> [--key=value ...]"""
    if len(sys.argv) < 2:
        print("Usage: python -m fms.configs.loader <config.yaml> [--key=value ...]")
        sys.exit(1)
    overrides = [arg.lstrip("-") for arg in sys.argv[2:] if "=" in arg]
    return load_config(sys.argv[1], overrides=overrides)
```

### 3c. YAML Config Files

#### File: `fms/configs/pretrain.yaml`

```yaml
# ===========================================================================
# NanoSeek Pre-training Configuration — Single Source of Truth
# All values validated by fms/configs/schema.py at load time.
# ===========================================================================

experiment:
  name: "nanoseek-1b-pretrain-v1"
  seed: 42
  hardware: "8xH100"
  deterministic: false
  tags: ["pretrain", "nanoseek-1b", "chinchilla-optimal"]
  description: >
    NanoSeek-1B pre-training with DeepSeek V3 architecture.
    Chinchilla-optimal: 22B tokens for 1.08B active params.

pretrain:
  model_size: "1b"
  total_tokens: 22_000_000_000    # 22B = 20x active params (Chinchilla)
  batch_size: 524288               # 128 sequences × 4096 tokens
  seq_len: 4096                    # 4K context, extend via YaRN at inference
  lr: 3e-4                         # Peak LR (DeepSeek V3 style)
  lr_min: 3e-5                     # 10% of peak (cosine decay floor)
  weight_decay: 0.1
  optimizer: "muon+adamw"          # Muon for matrices, AdamW for embeddings
  warmup_steps: 1000
  max_grad_norm: 1.0
  checkpoint_every: 1000
  dtype: "bfloat16"
  gradient_checkpointing: true
  compile_model: true

sft:
  dataset: "ultrachat-200k"
  epochs: 3
  lr: 2e-5
  batch_size: 4
  seq_len: 2048
  lora_rank: 16
  lora_targets: ["wq_a", "wq_b", "wkv_a", "wkv_b", "wo"]

dpo:
  dataset: "ultrafeedback-60k"
  epochs: 1
  lr: 5e-7
  beta: 0.1
  batch_size: 2
  seq_len: 2048
  ref_model: "sft"

eval:
  benchmarks: ["mmlu", "gsm8k", "humaneval", "truthfulqa"]
  shots: {mmlu: 5, gsm8k: 5, humaneval: 0, truthfulqa: 0}
  safety_set: "safety/refusal_50.jsonl"
  leakage_check: true

serving:
  quantization: "int8"
  max_batch_size: 32
  max_seq_len: 4096
  speculative_decoding: true
  port: 8000

gate:
  quality_threshold: "+2% MMLU over baseline"
  quality_min_delta: 0.02
  safety_threshold: ">=baseline refusal accuracy"
  safety_regression_tolerance: 0.0
  p95_latency_budget_ms: 200.0
  cost_budget: "$5/1M tokens"
  cost_per_million_tokens: 5.0
```

#### File: `fms/configs/sft.yaml`

```yaml
# ===========================================================================
# NanoSeek SFT Configuration
# Inherits from pretrain.yaml, overrides SFT-specific settings.
# ===========================================================================

experiment:
  name: "nanoseek-1b-sft-v1"
  seed: 42
  hardware: "8xH100"
  base_config: "pretrain.yaml"
  tags: ["sft", "lora", "ultrachat"]
  description: "SFT with LoRA rank=16 on MLA projections. ~40 min on 8xH100."

sft:
  dataset: "ultrachat-200k"
  epochs: 3
  lr: 2e-5
  batch_size: 4
  seq_len: 2048
  lora_rank: 16
  lora_alpha: 32.0
  lora_dropout: 0.05
  lora_targets: ["wq_a", "wq_b", "wkv_a", "wkv_b", "wo"]
  warmup_ratio: 0.03
  weight_decay: 0.0
```

#### File: `fms/configs/dpo.yaml`

```yaml
# ===========================================================================
# NanoSeek DPO Configuration
# Direct Preference Optimization with SFT reference model.
# ===========================================================================

experiment:
  name: "nanoseek-1b-dpo-v1"
  seed: 42
  hardware: "8xH100"
  base_config: "pretrain.yaml"
  tags: ["dpo", "preference-alignment", "ultrafeedback"]
  description: "DPO with beta=0.1, UltraFeedback-60K. ~15 min on 8xH100."

dpo:
  dataset: "ultrafeedback-60k"
  epochs: 1
  lr: 5e-7
  beta: 0.1
  batch_size: 2
  seq_len: 2048
  ref_model: "sft"
  label_smoothing: 0.0
  max_prompt_length: 512
```

#### File: `fms/configs/serving.yaml`

```yaml
# ===========================================================================
# NanoSeek Serving Configuration
# INT8 quantization + MTP speculative decoding for production.
# ===========================================================================

experiment:
  name: "nanoseek-1b-serving-v1"
  seed: 42
  hardware: "1xH100"
  tags: ["serving", "int8", "speculative"]
  description: "Production serving: INT8 quant, speculative decoding."

serving:
  quantization: "int8"
  max_batch_size: 32
  max_seq_len: 4096
  speculative_decoding: true
  port: 8000
  host: "0.0.0.0"
  timeout: 300
  num_workers: 1

gate:
  p95_latency_budget_ms: 200.0
  cost_per_million_tokens: 5.0
```

#### File: `fms/configs/eval.yaml`

```yaml
# ===========================================================================
# NanoSeek Evaluation Configuration
# Quality benchmarks + safety + data leakage detection.
# ===========================================================================

experiment:
  name: "nanoseek-1b-eval-v1"
  seed: 42
  hardware: "1xH100"
  tags: ["eval", "benchmarks", "safety"]
  description: "Full eval sweep: MMLU, GSM8K, HumanEval, TruthfulQA + safety."

eval:
  benchmarks: ["mmlu", "gsm8k", "humaneval", "truthfulqa"]
  shots: {mmlu: 5, gsm8k: 5, humaneval: 0, truthfulqa: 0}
  safety_set: "safety/refusal_50.jsonl"
  leakage_check: true
  eval_batch_size: 8

gate:
  quality_threshold: "+2% MMLU over baseline"
  quality_min_delta: 0.02
  safety_threshold: ">=baseline refusal accuracy"
  safety_regression_tolerance: 0.0
  p95_latency_budget_ms: 200.0
  cost_per_million_tokens: 5.0
```

### 3d. Seed Management Utilities

File: `fms/configs/seed.py`

```python
"""
Seed management for reproducible experiments.
Controls model init, data ordering, and dropout randomness.
"""

from __future__ import annotations
import os, random
import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Global seed (0 to 2^32-1)
        deterministic: Force deterministic CUDA ops (~15% slower)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_dataloader_seed(base_seed: int, epoch: int) -> int:
    """Deterministic seed per epoch: same epoch = same data order across runs."""
    return (base_seed + epoch * 1000003) % (2**32)


def get_step_seed(base_seed: int, step: int) -> int:
    """Deterministic seed per step for exact dropout mask reproducibility."""
    return (base_seed + step * 7919) % (2**32)


def get_worker_seed(base_seed: int, worker_id: int) -> int:
    """Deterministic seed per dataloader worker to avoid identical batches."""
    return (base_seed + worker_id * 104729) % (2**32)


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker_init_fn for deterministic multi-worker loading."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
```

### 3e. Config Hashing for Experiment Deduplication

File: `fms/configs/hashing.py`

```python
"""
Config hashing for experiment identity and deduplication.
SHA-256 of canonical JSON representation. Two configs with the same
hash are guaranteed identical (collision probability < 1e-18).
"""

from __future__ import annotations
import hashlib, json
from typing import Any, Dict
from fms.configs.schema import FMSConfig


def hash_config(config: FMSConfig) -> str:
    """Compute SHA-256 hash of config for identity and dedup."""
    canonical = json.dumps(
        config.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def hash_config_short(config: FMSConfig, length: int = 12) -> str:
    """Short hash for display (default 12 hex chars = 48 bits)."""
    return hash_config(config)[:length]


def verify_config_hash(config: FMSConfig, expected_hash: str, strict: bool = True) -> bool:
    """
    Verify config matches checkpoint hash. Catches modified-config-resume bugs.

    Args:
        config: Current config
        expected_hash: Hash from checkpoint metadata
        strict: Raise on mismatch (True) or warn (False)
    """
    current_hash = hash_config(config)
    if current_hash != expected_hash:
        msg = (
            f"Config hash mismatch!\n"
            f"  Expected (checkpoint): {expected_hash[:16]}...\n"
            f"  Current:               {current_hash[:16]}...\n"
            f"Config was modified since experiment launch. "
            f"Create a new experiment instead."
        )
        if strict:
            raise ConfigHashMismatchError(msg)
        import warnings
        warnings.warn(msg, ConfigHashWarning)
        return False
    return True


class ConfigHashMismatchError(Exception):
    """Raised when config hash doesn't match checkpoint."""
    pass

class ConfigHashWarning(UserWarning):
    """Warned on non-strict config hash mismatch."""
    pass
```

### 3f. Config Diff Utility

File: `fms/configs/diff.py`

```python
"""
Config diff utility — compare two experiment configs field-by-field.

Usage:
    python -m fms.configs.diff baseline.yaml experiment.yaml
"""

from __future__ import annotations
import sys
from dataclasses import dataclass
from typing import Any, Dict, List
from fms.configs.loader import load_config
from fms.configs.schema import FMSConfig


@dataclass
class ConfigChange:
    path: str          # Dot-separated field path
    old_value: Any
    new_value: Any


def diff_configs(config_a: FMSConfig, config_b: FMSConfig) -> List[ConfigChange]:
    """Compute field-by-field diff. Returns only changed fields, sorted by path."""
    changes: List[ConfigChange] = []
    _diff_recursive(
        config_a.model_dump(mode="json"),
        config_b.model_dump(mode="json"),
        "", changes,
    )
    return sorted(changes, key=lambda c: c.path)


def _diff_recursive(a: Any, b: Any, path: str, changes: List[ConfigChange]):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            _diff_recursive(a.get(key), b.get(key), f"{path}.{key}" if path else key, changes)
    elif a != b:
        changes.append(ConfigChange(path=path, old_value=a, new_value=b))


def print_diff(changes: List[ConfigChange], name_a: str = "baseline", name_b: str = "experiment") -> str:
    if not changes:
        return f"No differences between {name_a} and {name_b}."
    lines = [
        f"Config diff: {name_a} → {name_b}",
        f"{'─' * 72}",
        f"{'Field':<35} {'Old':>15} {'New':>15}",
        f"{'─' * 35} {'─' * 15} {'─' * 15}",
    ]
    for c in changes:
        old_s = f"{c.old_value:.2e}" if isinstance(c.old_value, float) and abs(c.old_value) < 0.001 and c.old_value != 0 else str(c.old_value)
        new_s = f"{c.new_value:.2e}" if isinstance(c.new_value, float) and abs(c.new_value) < 0.001 and c.new_value != 0 else str(c.new_value)
        lines.append(f"{c.path:<35} {old_s:>15} {new_s:>15}")
    lines.append(f"{'─' * 72}")
    lines.append(f"Total changes: {len(changes)}")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 3:
        print("Usage: python -m fms.configs.diff <baseline.yaml> <experiment.yaml>")
        sys.exit(1)
    config_a = load_config(sys.argv[1])
    config_b = load_config(sys.argv[2])
    changes = diff_configs(config_a, config_b)
    print(print_diff(changes, sys.argv[1], sys.argv[2]))
    from fms.configs.hashing import hash_config_short
    print(f"\nHashes: {sys.argv[1]}={hash_config_short(config_a)}  {sys.argv[2]}={hash_config_short(config_b)}")


if __name__ == "__main__":
    main()
```

---

## 4. Config Hierarchy Visualization

```
FMSConfig (root)
│
├── experiment: ExperimentConfig
│   ├── name: str                    "nanoseek-1b-pretrain-v1"
│   ├── seed: int                    42
│   ├── hardware: str                "8xH100"
│   ├── deterministic: bool          false
│   ├── base_config: Optional[str]   null
│   ├── tags: List[str]              ["pretrain", "nanoseek-1b"]
│   └── description: str             "NanoSeek-1B pre-training..."
│
├── pretrain: PretrainConfig
│   ├── model_size: str              "1b"
│   ├── total_tokens: int            22_000_000_000
│   ├── batch_size: int              524288
│   ├── seq_len: int                 4096
│   ├── lr / lr_min: float           3e-4 / 3e-5
│   ├── optimizer: OptimizerType     "muon+adamw"
│   ├── warmup_steps: int            1000
│   ├── checkpoint_every: int        1000
│   └── dtype: str                   "bfloat16"
│
├── sft: SFTConfig
│   ├── dataset: str                 "ultrachat-200k"
│   ├── epochs: int                  3
│   ├── lr: float                    2e-5
│   ├── lora_rank: int               16
│   ├── lora_targets: List[str]      ["wq_a", "wq_b", ...]
│   └── warmup_ratio: float          0.03
│
├── dpo: DPOConfig
│   ├── dataset: str                 "ultrafeedback-60k"
│   ├── lr: float                    5e-7
│   ├── beta: float                  0.1
│   ├── ref_model: str               "sft"
│   └── label_smoothing: float       0.0
│
├── eval: EvalConfig
│   ├── benchmarks: List[str]        ["mmlu", "gsm8k", ...]
│   ├── shots: Dict[str, int]        {mmlu: 5, gsm8k: 5, ...}
│   ├── safety_set: str              "safety/refusal_50.jsonl"
│   └── leakage_check: bool          true
│
├── serving: ServingConfig
│   ├── quantization: QuantType      "int8"
│   ├── max_batch_size: int          32
│   ├── speculative_decoding: bool   true
│   └── port: int                    8000
│
└── gate: GateConfig
    ├── quality_min_delta: float     0.02
    ├── safety_regression_tolerance: 0.0
    ├── p95_latency_budget_ms: float 200.0
    └── cost_per_million_tokens: float 5.0
```

### Loading Flow

```
  Schema Defaults          Base YAML              Experiment YAML         CLI Overrides
  (Pydantic defaults) ──▶  (pretrain.yaml)   ──▶  (lr_sweep.yaml)   ──▶  (--pretrain.lr=1e-4)
                         deep_merge             deep_merge             deep_merge
                                                                            │
                                                                     ┌──────▼──────┐
                                                                     │  Pydantic   │
                                                                     │ Validation  │
                                                                     └──────┬──────┘
                                                                            │
                                                                     ┌──────▼──────┐
                                                                     │  FMSConfig  │
                                                                     │  (frozen)   │
                                                                     └─────────────┘
```

---

## 5. File Placement

```
fms/
├── configs/
│   ├── __init__.py              # Re-exports load_config, FMSConfig
│   ├── schema.py                # Pydantic models (§3a)
│   ├── loader.py                # YAML loading + merging + CLI (§3b)
│   ├── seed.py                  # Seed management (§3d)
│   ├── hashing.py               # Config hashing + verification (§3e)
│   ├── diff.py                  # Config diff utility (§3f)
│   ├── pretrain.yaml            # Default pretrain config (§3c)
│   ├── sft.yaml                 # Default SFT config
│   ├── dpo.yaml                 # Default DPO config
│   ├── serving.yaml             # Default serving config
│   ├── eval.yaml                # Default eval config
│   └── experiments/             # Per-experiment overrides
│       ├── lr_sweep_1e4.yaml
│       └── lora_rank_ablation.yaml
```

### Relationship to Existing Files

| Existing File | Relationship | Migration Path |
|---|---|---|
| `model/config.py` | Architecture config (NanoSeekConfig) | Stays as-is; `FMSConfig.pretrain.model_size` selects the factory function |
| `scripts/pre-train.py` | Training script with TrainingConfig | Gradually migrated: reads FMSConfig, maps to TrainingConfig internally |
| `pyproject.toml` | Package config | Add `pydantic` and `pyyaml` to dependencies |

The migration is **non-breaking**: existing scripts continue working with their current argument parsing. New experiments use `fms/configs/` for full reproducibility.

---

## 6. Usage Examples

### 6a. Load and Validate

```python
from fms.configs.loader import load_config
from fms.configs.hashing import hash_config_short

config = load_config("fms/configs/pretrain.yaml")
print(f"Experiment: {config.experiment.name}")
print(f"LR: {config.pretrain.lr}")
print(f"Hash: {hash_config_short(config)}")
```

### 6b. CLI Overrides

```bash
python -m fms.train.pretrain \
    --config fms/configs/pretrain.yaml \
    --pretrain.lr=1e-4 \
    --pretrain.warmup_steps=2000 \
    --experiment.seed=123
```

### 6c. Experiment with Inheritance

```yaml
# fms/configs/experiments/lr_sweep_1e4.yaml
experiment:
  name: "nanoseek-1b-lr-sweep-1e4"
  seed: 42
  base_config: "pretrain.yaml"     # Inherits ALL defaults
  tags: ["ablation", "lr-sweep"]

pretrain:
  lr: 1e-4                         # Only override this field
  lr_min: 1e-5
```

```python
config = load_config("fms/configs/experiments/lr_sweep_1e4.yaml")
assert config.pretrain.lr == 1e-4          # Overridden
assert config.pretrain.batch_size == 524288  # Inherited
```

### 6d. Validation Catches Errors Before GPU Allocation

```python
from pydantic import ValidationError
from fms.configs.schema import FMSConfig

try:
    FMSConfig.model_validate({
        "experiment": {"name": "test"},
        "pretrain": {"lr": -0.001, "lr_min": 0.01, "batch_size": 1000},
        "dpo": {"beta": 0.001},
    })
except ValidationError as e:
    print(e)
    # pretrain.lr: must be > 0
    # pretrain: lr_min (0.01) must be < lr (-0.001)
    # pretrain.batch_size: should be aligned to 1024
    # dpo.beta: dangerously small (< 0.01)
```

### 6e. Hash and Deduplicate

```python
from fms.configs.hashing import hash_config

hash_a = hash_config(load_config("experiments/run_a.yaml"))
hash_b = hash_config(load_config("experiments/run_b.yaml"))

if hash_a == hash_b:
    print("WARNING: Duplicate experiment — skipping to save $300")
```

### 6f. Diff Two Experiments

```bash
$ python -m fms.configs.diff pretrain.yaml experiments/lr_sweep_1e4.yaml

Config diff: pretrain.yaml → lr_sweep_1e4.yaml
────────────────────────────────────────────────────────────────────────
Field                                       Old             New
─────────────────────────────── ─────────────── ───────────────
experiment.name                 nanoseek-1...  nanoseek-1...
pretrain.lr                          3.00e-04      1.00e-04
pretrain.lr_min                      3.00e-05      1.00e-05
────────────────────────────────────────────────────────────────────────
Total changes: 3
```

### 6g. Checkpoint Integration

```python
from fms.configs.hashing import hash_config, verify_config_hash

# At launch: embed hash in checkpoint
config_hash = hash_config(config)
checkpoint["config_hash"] = config_hash

# At resume: verify config hasn't changed
verify_config_hash(config, checkpoint["config_hash"])
# Raises ConfigHashMismatchError if modified
```

---

## 7. Gotchas

### 7a. YAML Type Coercion — The Norway Problem

YAML 1.1 aggressively coerces values: `on` → `True`, `no` → `False` (the Norway ISO code `NO` becomes boolean `False`). Always quote string values that could be misinterpreted: `dataset: "on-policy-v2"` (not `dataset: on-policy-v2`). Pydantic validation catches type mismatches, but catching them in the YAML itself is cleaner.

| YAML Value | Expected | Actual | Fix |
|---|---|---|---|
| `on` | `"on"` | `True` | Quote it: `"on"` |
| `no` | `"no"` | `False` | Quote it: `"no"` |
| `0o10` | `"0o10"` | `8` | Quote it |
| `1e-4` | `0.0001` | `0.0001` | Correct ✓ |

### 7b. Float Precision in Config Hashing

`3e-4` and `0.0003` are the same float and produce the same hash. But `0.00030000000000000001` is a different float in IEEE 754 and may produce a different hash. **Rule**: always use scientific notation for small values (`3e-4` not `0.0003`).

### 7c. Path Resolution in Base Configs

`base_config: "pretrain.yaml"` resolves relative to the config file's directory, not the working directory. For experiment configs in `experiments/` referencing parent configs, use `base_config: "../pretrain.yaml"`.

### 7d. CLI Override Type Inference

The parser infers types heuristically: `--experiment.name=42` becomes integer `42`. Pydantic coerces int → str for string fields, but for stricter behavior, quote the value: `--experiment.name='"42"'`.

### 7e. Seed Reproducibility Across PyTorch Versions

`torch.manual_seed(42)` on PyTorch 2.2 may produce different weight initializations than PyTorch 2.4. Experiments are only reproducible on the same PyTorch version. Record `torch.__version__` in checkpoint metadata.

### 7f. Lists Are Replaced, Not Merged

Deep merge treats lists as atomic: an override replaces the entire list. `benchmarks: ["mmlu"]` in the experiment YAML replaces the base's full benchmark list, it does not append. This is intentional — list merge semantics (append? prepend?) are ambiguous.

### 7g. Validation Runs Before GPU Allocation

Pydantic validation in `load_config()` takes <100ms on CPU. Config errors are caught before spending $4/hour on GPU instances. But validation cannot check external constraints (e.g., "does the dataset path exist?") — those are runtime checks in the training script.

### 7h. Immutability Is Convention, Not Type-Enforced

`FMSConfig` is a Pydantic model that allows mutation after construction. The config hash catches modifications at resume time, but not at runtime. For strict immutability, set `model_config = ConfigDict(frozen=True)` in each model class.

---

*"The most expensive bug in machine learning is not a wrong gradient — it's a right gradient applied to the wrong hyperparameters. A config system doesn't make your experiments better. It makes your experiments real."*

— Principal Engineer's Note, Foundation Models Division, 2026
