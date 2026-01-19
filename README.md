<div align="center">

# NanoSeek

### DeepSeek V3.2 at Nano Scale

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-100%2B%20passing-brightgreen.svg)](tests/)

**Educational implementation of DeepSeek V3.2 architecture**

*For educational purposes only. Inspired by [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT).*

[Quick Start](#-quick-start) •
[Architecture](#-architecture) •
[Training](#-training) •
[Validation](#-validation-run-100m-tokens) •
[Contributing](#-contributing)

</div>

---

## Why NanoSeek?

> *"DeepSeek is probably the single most important paper that most Silicon Valley researchers read from in the last couple years."* — **Jensen Huang, 2026**

NanoSeek is an **educational project** implementing the four breakthrough innovations from DeepSeek V3.2 at a trainable scale (~1B active parameters). Inspired by Andrej Karpathy's philosophy of learning by building, this project enables you to:

- **Learn** frontier LLM architecture through hands-on implementation
- **Train** your own model on consumer/cloud hardware
- **Experiment** with MLA, MoE, MTP, and sparse attention
- **Understand** the internals of modern large language models

| Innovation | What It Does | Our Implementation |
|------------|--------------|-------------------|
| **MLA** | Smaller KV cache | ✅ Complete |
| **MoE** | Sparse expert routing | ✅ Complete |
| **MTP** | Multi-token prediction | ✅ Complete |
| **DSA** | Sparse attention | ✅ Complete |

### Scaling Laws & DeepSeek Ratios

We strictly preserve DeepSeek's architectural ratios to ensure insights transfer across scales:

| Ratio | DeepSeek V3 | NanoSeek | Preserved |
|-------|-------------|----------|-----------|
| `kv_lora_rank / hidden` | 0.07× | 0.07× | ✅ |
| `q_lora_rank / hidden` | 0.21× | 0.21× | ✅ |
| `experts_per_token / total_experts` | 8/256 | 8/64 | ✅ |
| `shared_experts` | 1-2 | 2 | ✅ |
| `mtp_loss_weight` | 0.3→0.1 | 0.3→0.1 | ✅ |

Training follows Chinchilla scaling: **20× tokens per active parameter** (22B tokens for 1.08B active params).

---

## 🎯 Key Features

### Architecture Innovations

- **Multi-Head Latent Attention (MLA)**: Low-rank KV compression (kv_lora_rank + rope_dim per layer)
- **DeepSeekMoE**: 64 routed + 2 shared experts with auxiliary-loss-free load balancing
- **Multi-Token Prediction (MTP)**: Predicts multiple future tokens for richer training signal
- **DeepSeek Sparse Attention (DSA)**: Lightning Indexer for selective token attention

### Training Features

- **FLOP-based training**: Compute-optimal scheduling
- **Multi-phase training**: Dense → Sparse attention transition
- **Distributed ready**: torchrun for multi-GPU training
- **Memory efficient**: Gradient checkpointing, optimized data loading

### Developer Experience

- **100+ tests**: Comprehensive test suite with numerical validation
- **Clean APIs**: `create_nanoseek(config)` — that's it
- **Type hints**: Full type annotations throughout

---

## 📊 Model Specifications

### NanoSeek-1B (Default)

| Parameter | Value |
|-----------|-------|
| Hidden Size | 2048 |
| Layers | 16 |
| Attention Heads | 16 |
| **Active Parameters** | ~1.08B |
| **Total Parameters** | ~4.87B |
| Experts (Routed + Shared) | 64 + 2 |
| Active Experts | 8 |
| KV Compression | 23× |
| Context Length | 4K (train) / 32K (infer) |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nanoseek.git
cd nanoseek

# Install in development mode
pip install -e ".[dev,training]"

# Verify installation
pytest tests/test_mla.py -v
```

### Quick Test

```python
import torch
from model.model import create_nanoseek
from model.config import get_nanoseek_config

# Create model
config = get_nanoseek_config()
model = create_nanoseek(config)

# Forward pass
batch_size, seq_len = 2, 512
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

outputs = model(input_ids)
print(f"Output shape: {outputs['logits'].shape}")
print(f"Active params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
```

### Quick Training

```bash
# CPU/MacBook test (small model, short sequences)
python scripts/pre-train.py \
  --model_size=125m \
  --max_seq_len=512 \
  --device_batch_size=1 \
  --num_iterations=100

# Single GPU training
python scripts/pre-train.py \
  --model_size=1b \
  --max_seq_len=4096

# Multi-GPU distributed training (8×H100)
torchrun --nproc_per_node=8 scripts/pre-train.py
```

---

## 🏗️ Architecture

### Overview

```
Input Tokens
     │
     ▼
┌─────────────────┐
│   Embedding     │  (65536 vocab → 2048 hidden)
└─────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│              NanoSeek Decoder Layer (×16)           │
│  ┌───────────────────────────────────────────────┐  │
│  │ RMSNorm → MLA Attention → Residual            │  │
│  │          (23× KV compression)                 │  │
│  └───────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────┐  │
│  │ RMSNorm → MoE FFN → Residual                  │  │
│  │          (64+2 experts, 8 active)             │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────┐
│    RMSNorm      │
└─────────────────┘
     │
     ├──────────────────────────┐
     ▼                          ▼
┌─────────────────┐    ┌─────────────────┐
│   Main Head     │    │   MTP Head      │
│   (next token)  │    │   (+1 token)    │
└─────────────────┘    └─────────────────┘
```

### Multi-Head Latent Attention (MLA)

The key innovation enabling massive KV cache reduction:

```python
# Standard Multi-Head Attention (MHA)
kv_cache_size = 2 × num_heads × head_dim × seq_len
#             = 2 × 16 × 128 × 4096 = 16.7M values

# Multi-Head Latent Attention (MLA)
kv_cache_size = (kv_lora_rank + qk_rope_head_dim) × seq_len
#             = (143 + 32) × 4096 = 717K values

# Compression ratio: 16.7M / 717K ≈ 23×
```

**How it works:**
1. Compress K, V through low-rank projection (kv_lora_rank=143)
2. Share RoPE component across all heads (qk_rope_head_dim=32)
3. Reconstruct full attention at computation time

### DeepSeekMoE

Sparse expert architecture with auxiliary-loss-free balancing:

```python
# Configuration
num_experts = 64          # Routed experts
num_shared_experts = 2    # Always active
num_experts_per_tok = 8   # Active per token

# Key innovations:
# 1. Sigmoid scoring (NOT softmax) - allows variable expert counts
# 2. Group-based routing - reduces communication overhead
# 3. Bias-based load balancing - no auxiliary loss needed

# Load balancing update (the key insight!)
bias[i] -= gamma * (actual_load[i] - expected_load)
# gamma = 0.001, frozen at 80% training
```

### Multi-Token Prediction (MTP)

Predict multiple future tokens for both training and inference:

```python
# Training: Additional supervision signal
mtp_loss_weight = 0.3 → 0.1  # Decays during training
total_loss = main_loss + mtp_loss_weight * mtp_loss

# Inference: Speculative decoding
# MTP head proposes draft tokens
# Main head verifies in parallel
# Accept multiple tokens per forward pass
```

---

## 📂 Project Structure

```
nanoseek/
├── model/                       # Core implementation
│   ├── config.py               # Configuration
│   ├── model.py                # Full model
│   └── optimizer/              # Muon + AdamW
│
├── scripts/                    # Training & utilities
│   ├── pre-train.py           # Main training script
│   ├── dataloader.py          # Data loading
│   └── scheduler.py           # LR scheduling
│
└── tests/                      # Comprehensive tests
    ├── test_mla.py            # MLA tests
    ├── test_moe.py            # MoE tests
    ├── test_mtp.py            # MTP tests
    ├── test_dsa.py            # Sparse attention tests
    ├── test_rope.py           # RoPE/YaRN tests
    ├── test_numerical.py      # Numerical stability
    └── test_integration.py    # End-to-end tests
```

---

## ⚙️ Configuration

### Default Configuration

```python
from model.config import get_nanoseek_config

config = get_nanoseek_config()

# Architecture
config.hidden_size          # 2048
config.num_hidden_layers    # 16
config.num_attention_heads  # 16
config.vocab_size           # 65536
config.max_position_embeddings  # 4096

# MLA
config.mla.q_lora_rank      # 430 (0.21 × hidden)
config.mla.kv_lora_rank     # 143 (0.07 × hidden)
config.mla.qk_rope_head_dim # 32

# MoE
config.moe.num_experts      # 64
config.moe.num_shared_experts  # 2
config.moe.num_experts_per_tok # 8
config.moe.use_sigmoid_scoring # True
config.moe.aux_loss_free    # True

# MTP
config.mtp.num_nextn_predict_layers  # 1
config.mtp.mtp_loss_weight  # 0.3 → 0.1

# Training
config.training.total_tokens    # 22B
config.training.global_batch_size  # 128
config.training.learning_rate   # 3e-4 → 3e-5
```

### Custom Configuration

```python
from model.config import NanoSeekConfig, MLAConfig, MoEConfig

config = NanoSeekConfig(
    hidden_size=1024,
    num_hidden_layers=12,
    mla=MLAConfig(
        kv_lora_rank=71,  # Maintain 0.07× ratio
    ),
    moe=MoEConfig(
        num_experts=32,
        num_experts_per_tok=4,
    ),
)
```

### Preset Configurations

```python
from model.config import (
    get_nanoseek_config,      # Default 1B
    get_nanoseek_500m_config, # Smaller variant
)
```

---

## 🏋️ Training

### Prerequisites

1. **Data**: Download FineWeb-Edu dataset
   ```bash
   # Using HuggingFace datasets
   python scripts/download_data.py --dataset=fineweb-edu --tokens=100B
   ```

2. **Hardware**:
   - Minimum: 1× A100/H100 (80GB)
   - Recommended: 8× H100 for full training
   - Development: Any GPU or CPU (with reduced model)

### Training Commands

```bash
# Development (CPU, reduced model)
python scripts/pre-train.py \
  --model_size=125m \
  --max_seq_len=512 \
  --device_batch_size=1 \
  --num_iterations=100 \
  --device=cpu

# Single GPU (A100/H100)
python scripts/pre-train.py \
  --model_size=1b \
  --max_seq_len=4096 \
  --device_batch_size=4 \
  --gradient_accumulation_steps=8

# Multi-GPU (8× H100)
torchrun --nproc_per_node=8 scripts/pre-train.py \
  --model_size=1b \
  --max_seq_len=4096 \
  --device_batch_size=4

# Multi-phase training (dense → sparse)
python scripts/pre-train.py \
  --model_size=1b \
  --enable_dsa \
  --phase1_tokens=17.6B \
  --phase2_tokens=4.4B
```

### Training Monitoring

```bash
# Enable WandB logging
pip install wandb
wandb login

python scripts/pre-train.py --wandb_project=nanoseek

# Or TensorBoard
python scripts/pre-train.py --tensorboard_dir=./logs
tensorboard --logdir=./logs
```

### Training Metrics

The training script tracks:
- **Loss**: Main loss, MTP loss, auxiliary losses
- **MFU**: Model FLOPs Utilization
- **Expert Balance**: Load distribution across experts
- **Memory**: GPU memory usage, KV cache size
- **Speed**: Tokens/second, iterations/second

---

## 🔬 Validation Run (100M Tokens)

Before full training, we recommend a **100M token validation run** to verify the architecture works correctly and observe initial loss curves.

### Why 100M Tokens?

- Fast enough to run on a single GPU
- Long enough to see meaningful loss decrease
- Validates all components (MLA, MoE, MTP) work together
- Catches configuration issues early

### Run Validation

```bash
# Single GPU validation (recommended first step)
python scripts/pre-train.py \
  --model_size=500m \
  --max_seq_len=2048 \
  --total_tokens=100_000_000 \
  --device_batch_size=4 \
  --wandb_project=nanoseek-validation

# Even smaller test (CPU/laptop)
python scripts/pre-train.py \
  --model_size=125m \
  --max_seq_len=512 \
  --total_tokens=10_000_000 \
  --device_batch_size=1 \
  --device=cpu
```

### What to Look For

| Metric | Expected Behavior |
|--------|-------------------|
| Loss | Steady decrease from ~10 to ~4-5 |
| MoE Balance | Expert load variance < 20% |
| Gradients | No NaN/Inf values |
| Memory | Stable, no OOM |

### Sample Output

```
Step 1000 | Loss: 6.23 | MTP Loss: 5.89 | Expert Var: 0.15
Step 2000 | Loss: 5.41 | MTP Loss: 5.12 | Expert Var: 0.13
Step 3000 | Loss: 4.87 | MTP Loss: 4.62 | Expert Var: 0.12
...
```

If validation succeeds, proceed to full training with your target configuration.

---

## 🧪 Testing

### Run All Tests

```bash
# Full test suite
pytest tests/ -v

# Fast tests only (skip slow)
pytest tests/ -v -m "not slow"

# Specific component
pytest tests/test_mla.py -v
pytest tests/test_moe.py -v
pytest tests/test_mtp.py -v
```

### Test Categories

| Test | Description | Time |
|------|-------------|------|
| `test_mla.py` | MLA attention, KV compression | ~10s |
| `test_moe.py` | Expert routing, load balancing | ~15s |
| `test_mtp.py` | Multi-token prediction | ~10s |
| `test_dsa.py` | Sparse attention patterns | ~10s |
| `test_rope.py` | Position embeddings | ~5s |
| `test_numerical.py` | Gradient stability | ~10s |
| `test_integration.py` | End-to-end forward/backward | ~30s |

### Test Markers

```bash
# GPU-only tests
pytest tests/ -v -m gpu

# Slow tests (full model)
pytest tests/ -v -m slow

# Numerical validation
pytest tests/test_numerical.py -v
```

---

## 📚 References

**DeepSeek Papers:**
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) - MLA innovation
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437) - Full architecture
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066) - Expert routing

---

## 🛠️ Development Roadmap

### Phase 1: Foundation ✅
- [x] MLA implementation with KV compression
- [x] MoE with auxiliary-loss-free balancing
- [x] MTP with speculative decoding support
- [x] DSA with Lightning Indexer
- [x] Comprehensive test suite
- [x] Training infrastructure

### Phase 2: Optimization (In Progress)
- [ ] FP8 training support
- [ ] Multi-node distributed training
- [ ] Gradient checkpointing optimization
- [ ] Performance profiling and tuning

### Phase 3: Scaling
- [ ] NanoSeek-5B variant
- [ ] NanoSeek-20B variant
- [ ] Expert parallelism
- [ ] Pipeline parallelism

### Phase 4: Applications
- [ ] Instruction tuning dataset
- [ ] GRPO post-training
- [ ] Domain specialization examples
- [ ] Inference optimization

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Getting Started

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Run tests**
   ```bash
   pytest tests/ -v
   ```
5. **Submit a pull request**

### Contribution Areas

| Area | Description | Difficulty |
|------|-------------|------------|
| **Training** | Distributed training improvements | Medium |
| **Testing** | Additional test coverage | Easy |
| **Benchmarks** | Performance comparisons | Medium |
| **Documentation** | Improve docs and examples | Easy |

### Code Style

- **Python**: Follow PEP 8, use type hints
- **Documentation**: Docstrings for all public functions
- **Tests**: Every feature needs tests
- **Commits**: Clear, descriptive commit messages

### Pull Request Guidelines

1. **Title**: Clear description of the change
2. **Description**: What and why (not just how)
3. **Tests**: All tests passing
4. **Documentation**: Update relevant docs

---

## ❓ FAQ

### Why "NanoSeek"?

Inspired by Andrej Karpathy's nanoGPT, "NanoSeek" applies the same philosophy to DeepSeek V3.2 — implementing frontier architecture at "nano" scale for educational purposes, small enough to train on consumer hardware while preserving all innovations.

### What hardware do I need?

- **Validation run**: Single GPU (any size) with reduced model
- **Full training**: 1× A100/H100 (80GB) or multiple GPUs
- **Development**: CPU works for small tests

### Can I use this for production?

This project is for **educational purposes only**. While the implementation follows production patterns, it is intended as a learning and research tool, not for production deployment.

### How does this compare to official DeepSeek?

This is an educational reimplementation at smaller scale. The architecture is faithful, but training compute and data are much smaller than production DeepSeek.

---

## 📄 License

License pending. Please check back for updates.

---

## 🙏 Acknowledgments

- **[Andrej Karpathy](https://github.com/karpathy)** for [nanoGPT](https://github.com/karpathy/nanoGPT), which inspired this educational nano-scale approach
- **DeepSeek AI** for the groundbreaking research and open publications
- **PyTorch team** for the excellent deep learning framework

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. It is an independent reimplementation inspired by published research papers and is not affiliated with DeepSeek AI. Use at your own risk.

---

## 📬 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/nanoseek/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/nanoseek/discussions)

---

<div align="center">

**Built with ❤️ for the AI research community**

*Star ⭐ this repo if you find it useful!*

</div>
