# NanoSeek

NanoSeek is a compact, from-scratch DeepSeek-style language model project focused on:

- MLA (Multi-head Latent Attention)
- MoE (Mixture of Experts)
- MTP (Multi-Token Prediction)

Inspired by nanochat: minimal, hackable, and research-friendly training code.

The codebase is built for fast iteration: readable modules, strong tests, and one main training entrypoint.

## Current Scope

This repository is centered on pre-training experiments and architecture validation.

- Main package: `nanoseek/`
- Main trainer: `scripts/pre_train.py`
- Main model: `nanoseek/model.py`
- Main config system: `nanoseek/config.py`

## Project Layout

- `nanoseek/`: core model, config, optimizer, tokenizer, dataloader, checkpoint manager
- `scripts/`: training and utility scripts
- `eval/`: evaluation metrics and diagnostics
- `tests/`: unit and integration tests
- `runs/`: run helper scripts

## Quick Start

### 1) Install

```bash
cd nanoseek
pip install -e .
```

Optional training dependencies:

```bash
pip install -e .[training]
```

Optional dev/test dependencies:

```bash
pip install -e .[dev]
```

### 2) Run tests

```bash
pytest tests/ -v
```

### 3) Smoke train (CPU/Mac friendly)

```bash
python -m nanoseek.scripts.pre_train \
  --scale ablation \
  --num-iterations 20 \
  --device-batch-size 1 \
  --eval-tokens 512 \
  --eval-every -1
```

### 4) Standard training run

```bash
python -m nanoseek.scripts.pre_train \
  --run gate1-smoke \
  --scale ablation \
  --seed 42 \
  --num-iterations 100 \
  --eval-every 50 \
  --save-every 100 \
  --device-batch-size 4
```

## Key CLI Flags

- `--scale`: `anchor`, `ablation`, `1b`, or `d<N>`
- `--depth`: override scale with depth-based config
- `--target-flops`: FLOPs-budgeted run
- `--target-param-data-ratio`: token budget from scaling params
- `--no-mtp`: disable MTP ablation
- `--aux-loss-type {bias,classic}`: MoE balancing mode
- `--resume-from-step`: resume from checkpoint

For full ablation controls, refer to `scripts/pre_train.py` and `nanoseek/config.py`.

## Notes

- Checkpoints are written under `checkpoints/`.
- The training script supports single GPU and distributed launches.
- EMA weights are tracked for evaluation stability.

## License

Internal/research use unless specified otherwise by the repository owner.
