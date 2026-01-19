# Contributing to NanoSeek

Thank you for your interest in contributing to NanoSeek! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Contributions](#making-contributions)
- [Pull Request Process](#pull-request-process)
- [Contribution Areas](#contribution-areas)
- [Code Style](#code-style)
- [Testing Guidelines](#testing-guidelines)

---

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- Be respectful and constructive in discussions
- Focus on the technical merits of contributions
- Help others learn and grow
- Report any unacceptable behavior

---

## Getting Started

### Prerequisites

- Python 3.11+
- PyTorch 2.0+
- Git
- (Optional) CUDA-capable GPU for testing

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/nanoseek.git
cd nanoseek

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/nanoseek.git
```

---

## Development Setup

### Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install in development mode with all extras
pip install -e ".[dev,training]"

# Verify installation
pytest tests/test_mla.py -v
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run fast tests only
pytest tests/ -v -m "not slow"

# Run specific test file
pytest tests/test_mla.py -v

# Run with coverage
pytest tests/ -v --cov=model --cov-report=html
```

### Code Quality Checks

```bash
# Type checking (if using mypy)
mypy model/

# Linting (if using ruff/flake8)
ruff check model/

# Formatting (if using black)
black model/ scripts/ tests/
```

---

## Making Contributions

### Types of Contributions

| Type | Description | Difficulty |
|------|-------------|------------|
| Bug fixes | Fix issues in existing code | Variable |
| Tests | Add test coverage | Easy-Medium |
| Features | New functionality | Medium-Hard |
| Kernels | CUDA/Triton optimizations | Hard |
| Performance | Optimization improvements | Medium-Hard |

### Workflow

1. **Check existing issues** - See if your idea is already discussed
2. **Open an issue** - Discuss significant changes before implementing
3. **Create a branch** - Branch from `main` with a descriptive name
4. **Make changes** - Follow code style and testing guidelines
5. **Submit PR** - Open a pull request with clear description

### Branch Naming

```
feature/mla-optimization    # New features
fix/moe-routing-bug         # Bug fixes
docs/readme-update          # Documentation
test/integration-coverage   # Testing improvements
perf/kv-cache-memory        # Performance improvements
```

---

## Pull Request Process

### Before Submitting

- [ ] All tests pass: `pytest tests/ -v`
- [ ] Code follows style guidelines
- [ ] Documentation updated if needed
- [ ] Commit messages are clear and descriptive
- [ ] Branch is up-to-date with main

### PR Template

```markdown
## Summary
Brief description of changes

## Motivation
Why this change is needed

## Changes
- List of specific changes
- File modifications

## Testing
How the changes were tested

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### Review Process

1. **Automated checks** - CI runs tests
2. **Code review** - Maintainers review the code
3. **Feedback** - Address any requested changes
4. **Approval** - Get approval from maintainer
5. **Merge** - PR is merged to main

---

## Contribution Areas

### 1. Core Model (`model/`)

**High Impact Areas:**
- MLA attention optimization
- MoE routing efficiency
- MTP speculative decoding improvements
- Memory optimization

**Guidelines:**
- Preserve DeepSeek architecture ratios
- Maintain numerical stability
- Add tests for new functionality
- Document mathematical operations

**Example:**
```python
def improved_mla_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Improved MLA forward pass with better memory efficiency.

    Mathematical formulation:
    1. Compress KV: c_kv = W_dkv @ x  (d_model → kv_lora_rank)
    2. Share RoPE: k_rope = W_kr @ c_kv  (shared across heads)
    3. Reconstruct: K = [k_content; k_rope]

    Args:
        hidden_states: Input tensor [batch, seq_len, hidden_size]

    Returns:
        Output tensor [batch, seq_len, hidden_size]
    """
    # Implementation with clear comments
    ...
```

### 2. CUDA Kernels (`kernels/`)

**Priority Order:**
1. FlashMLA (attention kernel)
2. FP8 GEMM (training speedup)
3. MoE Dispatch/Combine (expert routing)
4. Grouped GEMM (batched operations)

**Guidelines:**
- Start with Triton, optimize with CUDA C++ later
- Target H100/Hopper architecture
- Follow kernel documentation in `kernels/docs/`
- Benchmark against PyTorch baseline

**Example Triton kernel:**
```python
@triton.jit
def mla_fwd_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    ...
):
    """
    FlashMLA forward kernel.

    Block structure:
    - BLOCK_M: Query block size
    - BLOCK_N: Key/Value block size
    - BLOCK_D: Hidden dimension
    """
    # Implementation
    ...
```

### 3. Training (`scripts/`)

**Areas for Improvement:**
- Distributed training optimization
- Memory-efficient gradient checkpointing
- Data loading performance
- Logging and monitoring

**Guidelines:**
- Maintain FLOP-based training paradigm
- Support both single and multi-GPU
- Keep training reproducible (seeds, determinism)

### 4. Tests (`tests/`)

**Coverage Goals:**
- >90% line coverage for core model
- Numerical validation against reference
- Edge cases and error handling
- Performance regression tests

**Test Structure:**
```python
class TestMLA:
    """Tests for Multi-Head Latent Attention."""

    def test_kv_compression_ratio(self, config):
        """Verify KV cache is compressed correctly."""
        ...

    def test_output_matches_mha(self, config):
        """Verify MLA output approximates MHA output."""
        ...

    def test_incremental_decoding(self, config):
        """Verify incremental decoding works correctly."""
        ...

    @pytest.mark.slow
    def test_full_model_integration(self, config):
        """Full integration test with large model."""
        ...
```

---

## Code Style

### Python Style

```python
# Use type hints
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Forward pass documentation.

    Args:
        hidden_states: Input tensor of shape [batch, seq, hidden]
        attention_mask: Optional attention mask

    Returns:
        Tuple of (output, attention_weights)
    """
    ...

# Clear variable names
kv_compressed = self.kv_down_proj(hidden_states)  # Good
x = self.proj(h)  # Avoid unless context is clear

# Document complex operations
# Apply RoPE to query and key
# q_rope: [batch, heads, seq, rope_dim]
q_rope = apply_rotary_pos_emb(q_for_rope, cos, sin)
```

### Commit Messages

```
# Format: <type>: <description>

feat: Add FP8 support for MLA attention
fix: Correct expert load balancing calculation
docs: Update README with new benchmarks
test: Improve MTP speculative decoding coverage
perf: Optimize KV cache memory allocation
refactor: Simplify MoE routing logic
```

### File Organization

```
model/
├── __init__.py          # Public exports
├── model.py             # Main model class
├── config.py            # Configuration
├── attention/           # Attention variants
│   ├── mla.py          # Multi-head latent attention
│   └── sparse.py       # Sparse attention
└── moe/                 # Mixture of experts
    ├── router.py       # Expert routing
    └── experts.py      # Expert networks
```

---

## Testing Guidelines

### Test Categories

```python
# Unit tests - test individual functions
def test_kv_compression():
    """Test that KV projection compresses correctly."""
    ...

# Integration tests - test component interactions
def test_mla_in_decoder_layer():
    """Test MLA within full decoder layer."""
    ...

# Numerical tests - validate correctness
def test_mla_matches_reference():
    """Test MLA output matches reference implementation."""
    ...

# Performance tests - benchmark
@pytest.mark.benchmark
def test_mla_forward_speed():
    """Benchmark MLA forward pass speed."""
    ...
```

### Test Markers

```python
@pytest.mark.slow       # Long-running tests
@pytest.mark.gpu        # Requires GPU
@pytest.mark.benchmark  # Performance benchmarks
@pytest.mark.numerical  # Numerical validation
```

### Fixtures

```python
# Use fixtures from conftest.py
def test_with_config(minimal_config):
    """Test using minimal configuration fixture."""
    model = create_nanoseek(minimal_config)
    ...

def test_on_device(device):
    """Test on available device (CPU or CUDA)."""
    ...
```

---

## Docstring Guidelines

```python
def function_name(arg1: Type1, arg2: Type2) -> ReturnType:
    """
    Brief one-line description.

    Longer description if needed, explaining the algorithm,
    mathematical formulation, or important details.

    Mathematical formulation:
        output = softmax(Q @ K.T / sqrt(d)) @ V

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: When invalid input provided

    Example:
        >>> result = function_name(x, y)
        >>> print(result.shape)
        torch.Size([2, 512, 2048])

    Note:
        Important implementation notes or caveats.

    See Also:
        - Related function or class
        - External reference
    """
```

---

## Questions?

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers for sensitive issues

---

Thank you for contributing to NanoSeek! Your efforts help make frontier LLM architecture accessible to everyone.
