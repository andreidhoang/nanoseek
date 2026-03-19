# NanoSeek Setup Guide
## Complete Environment Setup for AI Agents and Engineers on RunPod
### Copy-paste ready. Tested March 2026.

---

## Quick Start (5 minutes)

```bash
# === STEP 1: Clone and enter project ===
git clone <repo-url> /workspace/nanoseek
cd /workspace/nanoseek/nanoseek

# === STEP 2: Install all dependencies ===
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install wandb tiktoken scikit-learn pyarrow requests filelock scipy psutil

# === STEP 3: Download training data (170 shards = enough for anchor + 500M) ===
python -m nanoseek.nanoseek.dataset -n 170

# === STEP 4: Set W&B API key ===
export WANDB_API_KEY="<your-key>"

# === STEP 5: Verify everything works ===
python -m pytest tests/ -v                    # 120 tests should pass
python -c "
from nanoseek.nanoseek.tokenizer import get_tokenizer
t = get_tokenizer()
print(f'Tokenizer OK: vocab={t.get_vocab_size()}')
"

# === STEP 6: Run training ===
cd /workspace/nanoseek
python -m nanoseek.scripts.pre_train \
    --run gate1-smoke \
    --scale anchor \
    --seed 42 \
    --num-iterations 100 \
    --eval-every 50 \
    --save-every 100 \
    --device-batch-size 4
```

---

## Detailed Setup

### 1. System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.9+ | 3.11 |
| GPU (anchor) | Any 24GB NVIDIA | A6000 (48GB) |
| GPU (500M) | 1x 80GB | 1x H100 |
| GPU (1B) | 1x 80GB (tight) | 8x H100 |
| Disk | 20 GB | 100 GB |
| RAM | 32 GB | 64 GB |

### 2. RunPod Pod Configuration

**For Anchor Scale Training (Phase 3A)**:
```yaml
gpu_type: "NVIDIA A6000"
gpu_count: 1
cloud_type: "COMMUNITY"
container_image: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
volume_size: 50
volume_mount_path: "/workspace"
```

**For 500M / 1B Training (Phase 3B/3D)**:
```yaml
gpu_type: "NVIDIA H100 80GB HBM3"
gpu_count: 1    # or 8 for NanoSeek-1B
cloud_type: "SECURE"    # MUST be SECURE for multi-GPU NVLink
container_image: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
volume_size: 200
volume_mount_path: "/workspace"
```

### 3. Package Dependencies

#### Required (Training Will Fail Without These)

| Package | Version | Why |
|---------|---------|-----|
| `torch` | >= 2.0.0 | Core framework. Use cu124 index for CUDA 12.4 |
| `pyarrow` | any | Reads training data (parquet format) |
| `tiktoken` | any | Tokenizer backend (BPE) |
| `requests` | any | Downloads dataset shards from HuggingFace |
| `filelock` | any | Thread-safe file downloads |
| `wandb` | >= 0.15.0 | Training metrics logging (set `--run dummy` to skip) |

#### Required for Full Evaluation

| Package | Version | Why |
|---------|---------|-----|
| `scikit-learn` | any | I_spec computation (expert specialization metric) |
| `scipy` | any | Scaling law curve fitting |
| `numpy` | >= 1.20 | Numerical operations (installed with torch) |

#### Optional

| Package | Version | Why |
|---------|---------|-----|
| `pytest` | >= 7.0 | Running test suite |
| `psutil` | any | System monitoring |
| `fasttext-wheel` | any | Data curation quality classifier (not needed for training) |

#### One-Line Install
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124 && \
pip install wandb tiktoken scikit-learn pyarrow requests filelock scipy psutil pytest
```

### 4. Training Data Setup

**Dataset**: ClimbMix-400B (Karpathy's curated web text)
**Source**: `https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle`
**Format**: Parquet shards, ~60 MB each

**Download**:
```bash
cd /workspace/nanoseek/nanoseek

# For anchor scale experiments (170 shards, ~10 GB)
python -m nanoseek.nanoseek.dataset -n 170

# For full 1B training (all 6543 shards, ~400 GB)
python -m nanoseek.nanoseek.dataset -n -1
```

**Data location**: `~/.cache/nanochat/base_data_climbmix/`

**If data is on a different path** (e.g., mounted volume):
```bash
# Symlink to expected location
mkdir -p ~/.cache/nanochat
ln -sf /workspace/data/base_data_climbmix ~/.cache/nanochat/base_data_climbmix
```

**Verify data**:
```bash
ls ~/.cache/nanochat/base_data_climbmix/ | head -5
# Should show: shard_00000.parquet, shard_00001.parquet, ...
```

### 5. Tokenizer

The tokenizer auto-downloads on first use. No manual setup needed.

**Location**: `~/.cache/nanochat/tokenizer.json`
**Vocab size**: 32,768 tokens
**Special tokens**: BOS, EOS, FIM (prefix/suffix/middle), chat tokens, pad

**Verify**:
```bash
python -c "
from nanoseek.nanoseek.tokenizer import get_tokenizer
t = get_tokenizer()
print(f'Vocab: {t.get_vocab_size()}')
print(f'FIM tokens: {t.get_fim_tokens()}')
"
```

### 6. Environment Variables

| Variable | Required? | Default | Purpose |
|----------|-----------|---------|---------|
| `WANDB_API_KEY` | Yes (unless `--run dummy`) | none | W&B authentication |
| `NANOCHAT_BASE_DIR` | No | `~/.cache/nanochat/` | Data/tokenizer cache location |
| `CUDA_VISIBLE_DEVICES` | No | all GPUs | Limit visible GPUs |

### 7. Directory Structure After Setup

```
/workspace/nanoseek/
├── nanoseek/                     # Package root
│   ├── nanoseek/                 # Core code
│   │   ├── model.py              # Architecture (1,999 lines)
│   │   ├── config.py             # Configs (anchor/500m/1b)
│   │   ├── dataloader.py         # Data loading + FIM
│   │   ├── optim.py              # MuonAdamW optimizer
│   │   ├── tokenizer.py          # Tokenizer
│   │   ├── dataset.py            # Data download
│   │   ├── common.py             # DDP utilities
│   │   └── eval/                 # Evaluation modules
│   ├── scripts/
│   │   └── pre_train.py          # Main training script
│   └── tests/                    # 120 tests
├── docs/                         # Documentation
├── checkpoints/                  # Created during training
├── logs/                         # Created during training
└── wandb/                        # Created during training
```

---

## Verification Checklist

Run these checks before starting any training:

```bash
cd /workspace/nanoseek/nanoseek

# 1. Python + PyTorch
python -c "
import torch
print(f'Python OK')
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    t = torch.zeros(1).cuda()
    print(f'GPU tensor OK')
"

# 2. All imports work
python -c "
import pyarrow, wandb, sklearn, scipy, tiktoken, requests, filelock
print('All packages OK')
"

# 3. Test suite passes
python -m pytest tests/ -v --tb=short
# Expected: 120 passed, 7 skipped

# 4. Tokenizer loads
python -c "
from nanoseek.nanoseek.tokenizer import get_tokenizer
t = get_tokenizer()
print(f'Tokenizer OK: {t.get_vocab_size()} tokens')
"

# 5. Data exists
python -c "
from nanoseek.nanoseek.dataset import list_parquet_files
files = list_parquet_files()
print(f'Data OK: {len(files)} shards found')
"

# 6. Model builds correctly (meta device path)
python -c "
import torch, math
from nanoseek.nanoseek.config import get_nanoseek_anchor_config
from nanoseek.nanoseek.model import NanoSeekModel
config = get_nanoseek_anchor_config()
with torch.device('meta'):
    model = NanoSeekModel(config)
model.to_empty(device='cpu')
model.init_weights()
model.train()
x = torch.randint(0, config.vocab_size, (1, 32))
out = model(x, labels=x, mtp_lambda=0.3)
ln_V = math.log(config.vocab_size)
assert abs(out['main_loss'].item() - ln_V) > 0.01, 'BROKEN: loss = ln(V), init failed'
assert abs(out['mtp_loss'].item() - ln_V) > 0.01, 'BROKEN: MTP loss = ln(V), init failed'
print(f'Model OK: main_loss={out[\"main_loss\"].item():.2f}, mtp_loss={out[\"mtp_loss\"].item():.2f} (random={ln_V:.2f})')
"

# 7. Quick 10-step GPU training
cd /workspace/nanoseek
python -m nanoseek.scripts.pre_train \
    --run sanity \
    --scale anchor \
    --num-iterations 10 \
    --eval-every -1 \
    --save-every -1 \
    --device-batch-size 4
# Check: loss decreasing, H_load > 4, no errors
```

If ALL checks pass, you're ready to train.

---

## Exact Training Commands (Tested & Working)

These are the EXACT commands that produced successful training runs. Copy-paste directly.

### Gate 1 Smoke Test (100 steps, ~33 min on A6000)
```bash
cd /workspace/nanoseek

WANDB_API_KEY="<your-key>" python -m nanoseek.scripts.pre_train \
    --run gate1-smoke \
    --scale anchor \
    --seed 42 \
    --num-iterations 100 \
    --eval-every 50 \
    --save-every 100 \
    --device-batch-size 4
```

**Expected output at step 0**:
```
loss: ~14.15 (main:~10.89 mtp:~10.88 aux:0.000001) | H_load: ~5.97
```
If you see `main:10.3972` and `mtp:10.3972` exactly, init is broken. STOP.

**Expected output at step 30**:
```
loss: ~9.59 (main:~6.53 mtp:~6.86) | H_load: ~5.70
```

### HP Grid Search (12 runs, ~3 hrs on A6000)
```bash
cd /workspace/nanoseek

for mlr in 0.005 0.01 0.02 0.04; do
  for elr in 0.1 0.3 0.5; do
    WANDB_API_KEY="<your-key>" python -m nanoseek.scripts.pre_train \
        --run "hp-r1-mlr${mlr}-elr${elr}" \
        --scale anchor \
        --matrix-lr $mlr \
        --embedding-lr $elr \
        --eval-every 100 \
        --save-every -1 \
        --seed 42 &
    sleep 5
  done
  wait
done
```

### Stability Ablations (5 runs, use best HP from grid search)
```bash
cd /workspace/nanoseek
BEST_MLR=0.02   # replace with winner from HP search
BEST_ELR=0.3    # replace with winner from HP search

# Run A: Full V3.2 baseline
WANDB_API_KEY="<your-key>" python -m nanoseek.scripts.pre_train \
    --run stab-A --scale anchor --matrix-lr $BEST_MLR --embedding-lr $BEST_ELR --seed 42

# Run C: Remove seq_aux
WANDB_API_KEY="<your-key>" python -m nanoseek.scripts.pre_train \
    --run stab-C --scale anchor --matrix-lr $BEST_MLR --embedding-lr $BEST_ELR --no-seq-aux --seed 42

# Run D: Remove grad clip
WANDB_API_KEY="<your-key>" python -m nanoseek.scripts.pre_train \
    --run stab-D --scale anchor --matrix-lr $BEST_MLR --embedding-lr $BEST_ELR --no-grad-clip --seed 42

# Run E: Classic aux loss
WANDB_API_KEY="<your-key>" python -m nanoseek.scripts.pre_train \
    --run stab-E --scale anchor --matrix-lr $BEST_MLR --embedding-lr $BEST_ELR --aux-loss-type classic --seed 42

# Run F: Bad batch injection at step 1500
WANDB_API_KEY="<your-key>" python -m nanoseek.scripts.pre_train \
    --run stab-F --scale anchor --matrix-lr $BEST_MLR --embedding-lr $BEST_ELR --inject-bad-batch 1500 --seed 42
```

### nano-500M Validation (single run, ~14 hrs on H100)
```bash
cd /workspace/nanoseek

WANDB_API_KEY="<your-key>" python -m nanoseek.scripts.pre_train \
    --run hp-500m-transfer \
    --scale 500m \
    --matrix-lr $BEST_MLR \
    --embedding-lr $BEST_ELR \
    --eval-every 500 \
    --save-every 2000 \
    --device-batch-size 8 \
    --seed 42
```

### NanoSeek-1B Full Training (8x H100, ~12 hrs)
```bash
cd /workspace/nanoseek

WANDB_API_KEY="<your-key>" torchrun --nproc_per_node=8 -m nanoseek.scripts.pre_train \
    --run nanoseek-1b-v1 \
    --scale 1b \
    --matrix-lr $BEST_MLR \
    --embedding-lr $BEST_ELR \
    --eval-every 500 \
    --save-every 2000 \
    --device-batch-size 16 \
    --seed 42
```

### Resume from Checkpoint (after crash or preemption)
```bash
cd /workspace/nanoseek

# Resume from latest checkpoint
WANDB_API_KEY="<your-key>" python -m nanoseek.scripts.pre_train \
    --run gate1-smoke \
    --scale anchor \
    --seed 42 \
    --num-iterations 100 \
    --eval-every 50 \
    --save-every 100 \
    --device-batch-size 4 \
    --resume-from-step 0    # 0 = latest checkpoint

# Resume from specific step
    --resume-from-step 50   # resume from step 50 checkpoint
```

### Skip W&B (offline/no-login mode)
```bash
# Option 1: Offline mode (logs locally, sync later)
WANDB_MODE=offline python -m nanoseek.scripts.pre_train --run gate1-smoke --scale anchor ...

# Option 2: Dummy mode (no W&B at all)
python -m nanoseek.scripts.pre_train --run dummy --scale anchor ...
```

### Data Download Commands
```bash
cd /workspace/nanoseek/nanoseek

# Download 10 shards (quick test, ~600 MB)
python -m nanoseek.nanoseek.dataset -n 10

# Download 170 shards (enough for anchor + 500M, ~10 GB)
python -m nanoseek.nanoseek.dataset -n 170

# Download ALL shards (full 1B training, ~400 GB)
python -m nanoseek.nanoseek.dataset -n -1
```

### Important: Working Directory Matters
```bash
# CORRECT — run from /workspace/nanoseek (parent of nanoseek package)
cd /workspace/nanoseek
python -m nanoseek.scripts.pre_train ...

# WRONG — will get ModuleNotFoundError
cd /workspace/nanoseek/nanoseek
python -m scripts.pre_train ...      # FAILS
python -m nanoseek.scripts.pre_train ...  # ALSO FAILS from here
```

See `docs/TRAINING_EXECUTION_PLAN.md` for what to run next.

---

## Troubleshooting

### "No module named 'nanoseek'"
You must run from `/workspace/nanoseek` (NOT `/workspace/nanoseek/nanoseek`):
```bash
cd /workspace/nanoseek
python -m nanoseek.scripts.pre_train ...
```

### "CUDA error: no kernel image is available"
PyTorch version doesn't support your GPU architecture. Install correct version:
```bash
# For Ampere (A100, A6000, RTX 30xx) — CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124

# For Hopper (H100) — CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124

# For Blackwell (RTX PRO 4500, B200) — CUDA 12.8
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### "No API key configured" (wandb)
```bash
export WANDB_API_KEY="your-key-here"
# Or to skip W&B entirely:
WANDB_MODE=offline python -m nanoseek.scripts.pre_train --run dummy ...
```

### "No dataset parquet files found"
```bash
cd /workspace/nanoseek/nanoseek
python -m nanoseek.nanoseek.dataset -n 170
```

### "ModuleNotFoundError: No module named 'tiktoken'"
```bash
pip install tiktoken
```

### "sklearn not available, using simplified I_spec"
```bash
pip install scikit-learn
```

### Loss = 10.3972 at step 0 (exactly ln(32768))
Model initialization is broken. This was Bug 1-2 from the postmortem.
Verify the fix is present:
```bash
python -c "
import torch
from nanoseek.nanoseek.config import get_nanoseek_anchor_config
from nanoseek.nanoseek.model import NanoSeekModel, RMSNorm
config = get_nanoseek_anchor_config()
with torch.device('meta'):
    m = NanoSeekModel(config)
m.to_empty(device='cpu')
m.init_weights()
# Check RMSNorm
for name, mod in m.named_modules():
    if isinstance(mod, RMSNorm):
        assert torch.all(mod.weight == 1.0), f'{name} weight not ones!'
        break
# Check RoPE
f = m.layers[0].self_attn.freqs_cis
assert f.abs().sum() > 0, 'freqs_cis is zeros!'
print('Init OK')
"
```

### OOM (Out of Memory)
Reduce batch size:
```bash
--device-batch-size 2   # or even 1
```
Gradient accumulation compensates automatically to maintain effective batch size.

---

## For AI Agents: Automated Setup Script

Copy-paste this entire block to set up and verify in one shot:

```bash
#!/bin/bash
set -e

echo "=== NanoSeek Automated Setup ==="

# Install dependencies
pip install -q torch --index-url https://download.pytorch.org/whl/cu124
pip install -q wandb tiktoken scikit-learn pyarrow requests filelock scipy psutil pytest

# Enter project
cd /workspace/nanoseek/nanoseek

# Download minimal data
python -m nanoseek.nanoseek.dataset -n 10

# Run tests
python -m pytest tests/ -v --tb=short -q

# Verify model init
python -c "
import torch, math
from nanoseek.nanoseek.config import get_nanoseek_anchor_config
from nanoseek.nanoseek.model import NanoSeekModel
config = get_nanoseek_anchor_config()
with torch.device('meta'):
    model = NanoSeekModel(config)
model.to_empty(device='cpu')
model.init_weights()
model.train()
x = torch.randint(0, config.vocab_size, (1, 32))
out = model(x, labels=x, mtp_lambda=0.3)
ln_V = math.log(config.vocab_size)
assert abs(out['main_loss'].item() - ln_V) > 0.01
assert abs(out['mtp_loss'].item() - ln_V) > 0.01
print('Model init: PASS')
"

echo "=== Setup Complete ==="
echo "Next: cd /workspace/nanoseek && python -m nanoseek.scripts.pre_train --run gate1-smoke --scale anchor --seed 42 --num-iterations 100 --eval-every 50 --save-every 100 --device-batch-size 4"
```
