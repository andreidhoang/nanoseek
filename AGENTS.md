# AGENTS.md

## Cursor Cloud specific instructions

### Overview

NanoSeek is a single-package Python ML project (educational DeepSeek V3.2 implementation). No Docker, databases, or external services are required. See `README.md` for architecture details and `CONTRIBUTING.md` for code style guidelines.

### Running the project

- **Install**: `pip install -e ".[dev,training]"` (also install lint tools: `pip install black isort mypy`)
- **Verify**: `python scripts/verify_install.py`
- **Tests**: `pytest tests/ -v -m "not slow"` (fast suite, ~10s); `pytest tests/ -v` (full suite)
- **Lint**: `black --check model/ scripts/ tests/`, `isort --check-only model/ scripts/ tests/`, `mypy model/`

### Gotchas

- The default model config (`get_nanoseek_config()`) creates a ~4.75B param model that requires ~20GB+ RAM. For quick testing, override config fields as done in `scripts/verify_install.py` (hidden_size=256, num_hidden_layers=2, etc.).
- When passing `labels` to the model's forward method with a reduced `hidden_size`, the MTP module will fail because its cross-attention layers are initialized with the full 2048 hidden size. For small configs, compute loss manually using `F.cross_entropy` on the logits instead of passing `labels`.
- `test_rotation_is_invertible` in `tests/test_rope.py` is flaky due to floating-point precision (max diff ~4.8e-7 vs rtol=1e-4). It passes in isolation but may intermittently fail when run with the full suite.
- `~/.local/bin` must be on `PATH` for `pytest`, `black`, `isort`, `mypy`, and other pip-installed scripts to be found.
- No GPU is available in this environment; all tests run on CPU. Tests marked `@pytest.mark.gpu` are automatically skipped.
- `mypy model/` reports a false positive from torch internals (pattern matching syntax error in `torch._inductor`). This is not a project issue.
