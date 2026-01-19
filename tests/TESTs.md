# NanoSeek 350M Pre-Training Validation Plan
## DeepSeek Researcher/Engineer Perspective

**Objective**: Comprehensive component validation, logging, and monitoring to ensure absolute correctness before training a 350M parameter NanoSeek model.

---

## Executive Summary

A DeepSeek researcher would validate **every architectural innovation** with:
1. **Unit Tests**: Mathematical correctness of each component
2. **Gradient Sanity Checks**: Verify gradients flow correctly
3. **Numerical Stability Tests**: Check for NaN/Inf under edge cases
4. **Equivalence Tests**: Compare against reference implementations
5. **Integration Tests**: End-to-end forward/backward passes
6. **Runtime Monitoring**: Continuous validation during training

---

## 1. RoPE + YaRN Validation

### 1.1 Frequency Computation Tests
```python
def test_rope_frequencies():
    """Verify RoPE frequencies match DeepSeek V3 specification."""
    dim = 16  # qk_rope_head_dim
    theta = 10000.0

    # Test 1: Standard frequency computation
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # Verify exponential decay pattern
    assert torch.all(freqs[:-1] > freqs[1:]), "Frequencies must decay"

    # Verify range
    assert freqs[0] == 1.0  # First frequency = 1/theta^0 = 1
    assert freqs[-1] < 0.01  # Last frequency should be small
```

### 1.2 Rotation Correctness
```python
def test_rope_rotation_invariance():
    """Verify RoPE preserves relative position information."""
    # Key property: cos(a-b) pattern for dot product
    x = torch.randn(1, 10, 1, 16)  # (B, S, H, D)

    freqs = precompute_freqs_cis(16, 20)

    # Rotate x at position 5
    x_rotated_5 = apply_rotary_emb(x, freqs[5:15])

    # Rotate x at position 0, then check relative distance
    x_rotated_0 = apply_rotary_emb(x, freqs[:10])

    # Dot product should depend only on relative distance (5 positions)
    # Not absolute positions
```

### 1.3 YaRN Interpolation Tests
```python
def test_yarn_interpolation():
    """Verify YaRN extends context correctly."""
    # Train at 4K, extend to 32K (factor=8)
    original_len = 4096
    extended_len = 32768
    scaling_factor = 8.0

    # Verify frequencies are interpolated, not extrapolated
    freqs_original = precompute_freqs_cis(16, original_len)
    freqs_extended = precompute_freqs_cis(
        16, extended_len,
        scaling_factor=scaling_factor,
        original_max_position_embeddings=original_len
    )

    # Low frequencies should be scaled down (for longer context)
    # High frequencies should remain unchanged
    # Check smooth interpolation via linear_ramp_factor
```

### 1.4 Logging & Monitoring
- **Log**: Max/min frequencies, correction dim range
- **Monitor**: Attention entropy at different positions (should be stable)
- **Alert**: If position > max_position_embeddings during inference

---

## 2. MLA (Multi-head Latent Attention) Validation

### 2.1 KV Compression Correctness
```python
def test_mla_kv_compression():
    """Verify KV compression preserves information."""
    mla = MultiHeadLatentAttention(
        hidden_size=1024, num_heads=16,
        q_lora_rank=220, kv_lora_rank=72,
        qk_nope_head_dim=48, qk_rope_head_dim=16,
        v_head_dim=48
    )

    x = torch.randn(2, 128, 1024)

    # Forward pass
    output, cache = mla(x, use_cache=True)

    # Cache should be compressed
    kv_compressed, k_pe = cache
    assert kv_compressed.shape == (2, 128, 72)  # kv_lora_rank
    assert k_pe.shape == (2, 128, 1, 16)  # qk_rope_head_dim (shared!)

    # Verify compression ratio
    standard_kv_size = 2 * 16 * 64  # 2 * num_heads * head_dim = 2048
    mla_kv_size = 72 + 16  # kv_lora_rank + qk_rope_head_dim = 88
    compression_ratio = standard_kv_size / mla_kv_size
    assert compression_ratio > 20, f"Expected >20x, got {compression_ratio}x"
```

### 2.2 Attention Pattern Tests
```python
def test_mla_attention_patterns():
    """Verify attention patterns are reasonable."""
    mla = create_mla_from_config(config)

    x = torch.randn(1, 64, 1024)
    mask = create_causal_mask(64)

    # Get attention weights (modify forward to return them)
    output, _ = mla(x, attention_mask=mask)

    # Check 1: Row sums should equal 1 (softmax)
    # Check 2: Causal: No attention to future positions
    # Check 3: No extreme values (all zeros or all ones on single token)
```

### 2.3 Gradient Flow Tests
```python
def test_mla_gradient_flow():
    """Verify gradients flow through all projection paths."""
    mla = create_mla_from_config(config)

    x = torch.randn(2, 64, 1024, requires_grad=True)
    output, _ = mla(x)
    loss = output.sum()
    loss.backward()

    # All projections should have non-zero gradients
    assert mla.wq_a.weight.grad is not None
    assert mla.wq_b.weight.grad is not None
    assert mla.wkv_a.weight.grad is not None
    assert mla.wkv_b.weight.grad is not None
    assert mla.wo.weight.grad is not None

    # Check gradient magnitudes are reasonable (not exploding/vanishing)
    for name, param in mla.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm()
            assert 1e-6 < grad_norm < 1e6, f"{name} grad norm: {grad_norm}"
```

### 2.4 Incremental Decoding Consistency
```python
def test_mla_incremental_consistency():
    """Verify incremental decoding matches full forward pass."""
    mla = create_mla_from_config(config)
    mla.eval()

    # Full sequence forward
    x_full = torch.randn(1, 128, 1024)
    output_full, _ = mla(x_full)

    # Incremental: process prefix, then extend
    x_prefix = x_full[:, :64]
    x_suffix = x_full[:, 64:]

    out_prefix, cache = mla(x_prefix, use_cache=True)
    out_suffix, _ = mla(x_suffix, past_key_value=cache, use_cache=True)

    output_incremental = torch.cat([out_prefix, out_suffix], dim=1)

    # Should match within numerical precision
    assert torch.allclose(output_full, output_incremental, atol=1e-5)
```

### 2.5 Logging & Monitoring
- **Log**: Q/K/V norms, attention entropy per head, softmax temperature
- **Monitor**: KV cache growth rate, compression ratio at runtime
- **Alert**: Attention collapse (single token gets all weight)

---

## 3. MoE (Mixture of Experts) Validation

### 3.1 Router Behavior Tests
```python
def test_moe_router():
    """Verify router selects experts correctly."""
    moe = create_moe_from_config(config)

    x = torch.randn(2, 128, 1024)

    # Get router decisions
    weights, indices = moe.gate(x.view(-1, 1024))

    # Check 1: Correct number of experts selected
    assert indices.shape[-1] == 4  # num_experts_per_tok

    # Check 2: Weights sum to reasonable value
    # With sigmoid scoring, weights can vary more
    assert weights.sum(dim=-1).mean() > 0.5

    # Check 3: No expert index out of range
    assert indices.min() >= 0
    assert indices.max() < 32  # n_routed_experts
```

### 3.2 Load Balancing Tests
```python
def test_moe_load_balance():
    """Verify auxiliary-loss-free balancing works."""
    moe = create_moe_from_config(config)

    # Simulate training with imbalanced initial routing
    for step in range(100):
        x = torch.randn(2, 128, 1024)
        output, aux = moe(x)
        moe.update_load_balance_bias(gamma=0.001)

    # After training, loads should be more balanced
    load_stats = moe.get_expert_load_stats()
    loads = load_stats['expert_load']

    # Coefficient of variation should decrease
    cv = loads.std() / loads.mean()
    assert cv < 0.5, f"Load imbalance too high: CV={cv:.3f}"
```

### 3.3 Expert Specialization Check
```python
def test_expert_outputs_differ():
    """Verify different experts produce different outputs."""
    moe = create_moe_from_config(config)

    x = torch.randn(1, 1, 1024)

    # Get output from each expert individually
    expert_outputs = []
    for expert in moe.experts:
        expert_outputs.append(expert(x.view(-1, 1024)))

    # Experts should produce different outputs
    for i in range(len(expert_outputs)):
        for j in range(i+1, len(expert_outputs)):
            similarity = F.cosine_similarity(
                expert_outputs[i].flatten(),
                expert_outputs[j].flatten(),
                dim=0
            )
            assert similarity < 0.99, f"Experts {i} and {j} too similar"
```

### 3.4 Gradient Propagation Through Router
```python
def test_moe_gradient_through_router():
    """Verify gradients flow through routing decisions."""
    moe = create_moe_from_config(config)

    x = torch.randn(2, 64, 1024, requires_grad=True)
    output, aux = moe(x)
    loss = output.sum() + aux.get('seq_aux_loss', 0)
    loss.backward()

    # Router weights should get gradients
    assert moe.gate.weight.grad is not None
    assert moe.gate.weight.grad.abs().mean() > 0

    # All experts should get some gradient (statistically)
    for i, expert in enumerate(moe.experts):
        # At least some experts should be updated
        if expert.gate_proj.weight.grad is not None:
            assert expert.gate_proj.weight.grad.abs().mean() > 0
```

### 3.5 Logging & Monitoring
```python
# Add to MoE forward pass
def forward_with_logging(self, x):
    output, aux = original_forward(x)

    # Log expert utilization
    aux['expert_utilization'] = {
        'load': self.gate.expert_load.clone(),
        'bias': self.gate.expert_bias.clone(),
        'load_std': self.gate.expert_load.std().item(),
        'load_mean': self.gate.expert_load.mean().item(),
        'min_load': self.gate.expert_load.min().item(),
        'max_load': self.gate.expert_load.max().item(),
    }
    return output, aux
```

- **Log**: Per-expert load, bias values, routing entropy
- **Monitor**: Load imbalance coefficient, dead expert detection
- **Alert**: Expert receiving <1% of tokens, bias magnitude explosion

---

## 4. MTP (Multi-Token Prediction) Validation

### 4.1 Token Alignment Tests
```python
def test_mtp_token_alignment():
    """Verify MTP predicts correct token positions."""
    mtp = MultiTokenPrediction(
        hidden_size=1024, vocab_size=65536,
        num_mtp_modules=1
    )

    main_hidden = torch.randn(2, 128, 1024)
    labels = torch.randint(0, 65536, (2, 128))

    results = mtp(main_hidden, labels=labels)

    # MTP module 0 should predict token at position i+2
    # (main model predicts i+1, MTP predicts i+2)
    mtp_logits = results['mtp_logits'][0]

    # Shape should allow predicting shifted positions
    assert mtp_logits.shape[1] <= 126  # seq_len - 2 for offset
```

### 4.2 Loss Weight Schedule Tests
```python
def test_mtp_loss_schedule():
    """Verify dynamic loss weight schedule."""
    model = NanoSeekModel(config)

    # Early training: weight = 0.3
    model.tokens_processed.fill_(0)
    early_weight = model.get_mtp_loss_weight()
    assert early_weight == 0.3

    # After 60%: weight = 0.1
    model.tokens_processed.fill_(int(config.total_tokens * 0.61))
    late_weight = model.get_mtp_loss_weight()
    assert late_weight == 0.1
```

### 4.3 Speculative Decoding Consistency
```python
def test_mtp_speculative_decode():
    """Verify speculative decoding produces valid tokens."""
    model = NanoSeekModel(config)
    model.eval()

    # Get main model hidden states
    x = torch.randint(0, 65536, (1, 10))
    outputs = model(x)
    main_hidden = outputs['hidden_states']

    # Speculative decode
    if model.mtp is not None:
        draft_tokens, draft_probs = model.mtp.speculative_decode(
            main_hidden, temperature=0.0
        )

        # Tokens should be valid
        assert draft_tokens.min() >= 0
        assert draft_tokens.max() < 65536

        # Greedy should have high probability
        assert draft_probs.min() > 0.0
```

### 4.4 Logging & Monitoring
- **Log**: MTP loss per module, acceptance rate in speculative decoding
- **Monitor**: MTP/main loss ratio (should decrease during training)
- **Alert**: MTP loss > 2x main loss (possible misalignment)

---

## 5. DSA (DeepSeek Sparse Attention) Validation

### 5.1 Indexer Training Tests
```python
def test_indexer_training():
    """Verify indexer learns to predict attention patterns."""
    dsa = DSASparseAttention(
        hidden_size=1024, num_heads=16,
        q_lora_rank=220, kv_lora_rank=72,
        qk_nope_head_dim=48, qk_rope_head_dim=16,
        v_head_dim=48,
        sparse_config=SparseAttentionConfig(enabled=False)
    )

    x = torch.randn(2, 256, 1024, requires_grad=True)

    # Get indexer loss (should be computed even when DSA disabled)
    output, _, aux = dsa(x, output_indexer_loss=True)

    assert 'indexer_loss' in aux
    assert aux['indexer_loss'].item() > 0

    # Indexer loss should have gradients
    aux['indexer_loss'].backward(retain_graph=True)
    assert dsa.indexer.q_proj.weight.grad is not None
```

### 5.2 Sparse vs Dense Equivalence (Short Sequences)
```python
def test_dsa_sparse_dense_equivalence():
    """For short sequences, sparse attention should match dense."""
    config_with_sparse = SparseAttentionConfig(
        enabled=True,
        activation_threshold=64,  # Low threshold for testing
        topk_tokens=128
    )

    dsa = DSASparseAttention(..., sparse_config=config_with_sparse)

    # Short sequence: should use dense
    x_short = torch.randn(1, 32, 1024)
    out_short, _, _ = dsa(x_short)

    # Long sequence: uses sparse
    x_long = torch.randn(1, 128, 1024)
    out_long, _, _ = dsa(x_long)

    # Compare first 32 positions (if context is same)
    # Note: Due to sparse selection, may not be exactly equal
```

### 5.3 Token Selection Quality
```python
def test_indexer_selects_important_tokens():
    """Verify indexer selects tokens that get high attention."""
    dsa = DSASparseAttention(...)

    # Create input with clear attention pattern
    # (e.g., repeat a distinctive token that should be attended to)
    x = torch.randn(1, 256, 1024)

    # Get indexer scores
    q_compressed, kv_compressed, _ = dsa._get_compressed_representations(x)
    index_scores = dsa.indexer(q_compressed, kv_compressed)

    # Compare to actual attention scores from dense MLA
    # Selected tokens should correlate with high-attention tokens
```

### 5.4 Logging & Monitoring
- **Log**: Indexer entropy, top-k token positions, selection overlap with dense attention
- **Monitor**: Sparse attention sparsity ratio, indexer loss convergence
- **Alert**: Indexer selecting only recent tokens (locality collapse)

---

## 6. Full Model Integration Tests

### 6.1 Forward Pass Sanity
```python
def test_full_forward():
    """Verify complete forward pass works."""
    model = NanoSeekModel(get_nanoseek_350m_config())

    batch, seq = 2, 128
    x = torch.randint(0, 65536, (batch, seq))
    labels = torch.randint(0, 65536, (batch, seq))

    outputs = model(x, labels=labels)

    # Check all expected outputs
    assert 'logits' in outputs
    assert 'loss' in outputs
    assert 'main_loss' in outputs
    assert outputs['logits'].shape == (batch, seq, 65536)

    # Loss should be reasonable (not NaN, not too high)
    assert not torch.isnan(outputs['loss'])
    assert outputs['loss'].item() < 15  # Random init ~ln(vocab_size) ≈ 11
```

### 6.2 Backward Pass Stability
```python
def test_backward_stability():
    """Verify backward pass doesn't have gradient issues."""
    model = NanoSeekModel(get_nanoseek_350m_config())

    x = torch.randint(0, 65536, (4, 256))
    labels = torch.randint(0, 65536, (4, 256))

    outputs = model(x, labels=labels)
    outputs['loss'].backward()

    # Check no NaN gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN grad in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf grad in {name}"

    # Check gradient norms are reasonable
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert total_norm < 1000, f"Gradient norm too high: {total_norm}"
```

### 6.3 Memory Efficiency Check
```python
def test_memory_efficiency():
    """Verify MLA KV cache is actually smaller."""
    model = NanoSeekModel(get_nanoseek_350m_config())
    model.eval()

    x = torch.randint(0, 65536, (1, 512))

    with torch.no_grad():
        outputs = model(x, use_cache=True)

    cache = outputs['past_key_values']

    # Calculate actual cache size
    cache_size = 0
    for layer_cache in cache:
        kv_compressed, k_pe = layer_cache
        cache_size += kv_compressed.numel() * kv_compressed.element_size()
        cache_size += k_pe.numel() * k_pe.element_size()

    # Compare to theoretical MHA cache
    # MHA: 2 * layers * seq * heads * head_dim * 2 (bf16)
    mha_cache = 2 * 18 * 512 * 12 * 64 * 2  # ~27MB

    # MLA should be ~24x smaller
    compression = mha_cache / cache_size
    print(f"KV Cache compression: {compression:.1f}x")
    assert compression > 15, "KV cache not compressed enough"
```

### 6.4 Parameter Count Validation
```python
def test_parameter_count():
    """Verify actual params match config estimates."""
    config = get_nanoseek_350m_config()
    model = NanoSeekModel(config)

    actual_total = sum(p.numel() for p in model.parameters())
    estimated = config.estimated_total_params

    # Should be within 10%
    ratio = actual_total / estimated
    assert 0.9 < ratio < 1.1, f"Param count mismatch: {actual_total} vs {estimated}"

    print(f"Total params: {actual_total:,}")
    print(f"Estimated: {estimated:,}")
    print(f"Active params: {config.estimated_active_params:,}")
```

---

## 7. Training Loop Monitoring

### 7.1 Comprehensive Logging Schema
```python
class TrainingMonitor:
    """Monitor all critical metrics during training."""

    def log_step(self, step, model, outputs, optimizer):
        metrics = {
            # Loss components
            'loss/total': outputs['loss'].item(),
            'loss/main': outputs['main_loss'].item(),
            'loss/mtp': outputs.get('mtp_loss', 0),
            'loss/aux': outputs.get('aux_loss', 0),
            'loss/indexer': outputs.get('indexer_loss', 0),

            # Gradient health
            'grad/norm': self.compute_grad_norm(model),
            'grad/max': self.compute_max_grad(model),
            'grad/nan_count': self.count_nan_grads(model),

            # MoE health (per layer)
            'moe/load_std': [],
            'moe/min_load': [],
            'moe/max_load': [],
            'moe/dead_experts': [],

            # Attention health
            'attn/entropy': [],
            'attn/max_weight': [],

            # MTP health
            'mtp/weight': model.get_mtp_loss_weight(),
            'mtp/gamma': model.get_gamma(),

            # Learning rate
            'lr': optimizer.param_groups[0]['lr'],

            # Tokens processed
            'tokens': model.tokens_processed.item(),
        }

        # Collect MoE stats
        for layer_idx, stats in model.get_expert_load_stats().items():
            load = stats['expert_load']
            metrics['moe/load_std'].append(load.std().item())
            metrics['moe/min_load'].append(load.min().item())
            metrics['moe/max_load'].append(load.max().item())
            metrics['moe/dead_experts'].append((load == 0).sum().item())

        return metrics
```

### 7.2 Alert Conditions
```python
ALERT_CONDITIONS = {
    'loss_spike': lambda m: m['loss/total'] > 10,
    'nan_detected': lambda m: m['grad/nan_count'] > 0,
    'gradient_explosion': lambda m: m['grad/norm'] > 100,
    'gradient_vanishing': lambda m: m['grad/norm'] < 1e-7,
    'dead_experts': lambda m: any(d > 0 for d in m['moe/dead_experts']),
    'attention_collapse': lambda m: any(e < 0.1 for e in m['attn/entropy']),
    'moe_imbalance': lambda m: any(s > 1.0 for s in m['moe/load_std']),
}
```

### 7.3 Checkpoint Validation
```python
def validate_checkpoint(ckpt_path):
    """Validate checkpoint before resuming training."""
    state = torch.load(ckpt_path)

    # Check model state
    model_state = state['model_state_dict']
    for name, tensor in model_state.items():
        assert not torch.isnan(tensor).any(), f"NaN in {name}"
        assert not torch.isinf(tensor).any(), f"Inf in {name}"

    # Check optimizer state
    opt_state = state['optimizer_state_dict']
    for param_id, state in opt_state['state'].items():
        if 'exp_avg' in state:
            assert not torch.isnan(state['exp_avg']).any()
        if 'exp_avg_sq' in state:
            assert not torch.isnan(state['exp_avg_sq']).any()

    # Check training progress
    assert state['tokens_processed'] > 0
    assert state['step'] > 0
```

---

## 8. Pre-Training Checklist

### Before Training Begins:
- [ ] Run all unit tests (`pytest tests/`)
- [ ] Run gradient check on small batch
- [ ] Verify parameter count matches estimates
- [ ] Confirm KV cache compression ratio >20x
- [ ] Test incremental decoding consistency
- [ ] Validate MoE load balancing starts balanced
- [ ] Check MTP loss computation for first batch
- [ ] Verify DSA indexer gradients flow
- [ ] Profile memory usage at max sequence length
- [ ] Test checkpoint save/load cycle
- [ ] Validate learning rate schedule
- [ ] Confirm total tokens matches config

### During Training (Continuous):
- [ ] Monitor loss curves for instability
- [ ] Track MoE expert utilization
- [ ] Log gradient norms per layer
- [ ] Watch for attention entropy collapse
- [ ] Validate MTP loss ratio
- [ ] Check for dead experts
- [ ] Monitor memory consumption
- [ ] Verify checkpoint integrity

---

## 9. Implementation Files (pytest + Full Monitoring)

### Test Suite Structure
```
tests/
├── conftest.py              # Shared fixtures (configs, models, sample data)
├── test_rope.py             # RoPE + YaRN validation (6 tests)
├── test_mla.py              # MLA compression tests (8 tests)
├── test_moe.py              # MoE routing + load balance (7 tests)
├── test_mtp.py              # MTP alignment + loss schedule (5 tests)
├── test_dsa.py              # DSA sparse attention (6 tests)
├── test_integration.py      # Full model tests (8 tests)
└── test_numerical.py        # NaN/Inf/gradient stability (4 tests)
```

### Training Infrastructure
```
training/
├── monitor.py               # TrainingMonitor class with metrics logging
├── alerts.py                # Alert conditions and handlers
└── validation.py            # Pre-training validation suite
```

### Scripts
```
scripts/
├── validate_checkpoint.py   # Checkpoint integrity validation
├── run_pretrain_checks.py   # All pre-training validation in one script
└── profile_memory.py        # Memory profiling at max seq length
```

### pytest Configuration (pyproject.toml)
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "gpu: marks tests requiring GPU",
    "integration: marks integration tests",
]
```

### Key Fixtures (conftest.py)
```python
@pytest.fixture
def config_350m():
    return get_nanoseek_350m_config()

@pytest.fixture
def model_350m(config_350m):
    return NanoSeekModel(config_350m)

@pytest.fixture
def sample_batch():
    return {
        'input_ids': torch.randint(0, 65536, (2, 128)),
        'labels': torch.randint(0, 65536, (2, 128)),
    }
```

---

## 10. Critical Metrics to Track

```
Phase 1 (Dense, 80% tokens):
├── Loss: main_loss, mtp_loss, seq_aux_loss, indexer_loss
├── MoE: expert_load_std, dead_expert_count, bias_magnitude
├── Gradients: global_norm, per_layer_norm, nan_count
└── Attention: entropy_per_head, kv_cache_size

Phase 2 (Sparse, 20% tokens):
├── All Phase 1 metrics +
├── DSA: sparse_ratio, top_k_overlap_with_dense
├── YaRN: position_extrapolation_errors
└── Performance: tokens_per_second, memory_peak
```

This plan ensures **every architectural innovation** is validated before committing GPU hours to training. A DeepSeek researcher would not proceed without these checks.
