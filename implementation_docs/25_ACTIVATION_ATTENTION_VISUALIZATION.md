# 25 — Activation & Attention Visualization Toolkit

## Distinguished Engineer's Assessment — February 2026

> "NanoSeek's four simultaneous innovations — MLA, MoE, MTP, DSA — each introduce opaque
> internal mechanics that no existing visualization tool can handle. BertViz doesn't
> understand MLA's split Q/K. TransformerLens can't hook into custom MoE dispatch.
> We need a purpose-built toolkit: 17 functions covering attention, routing, latent spaces,
> MTP agreement, and sparse masks — production-quality, fully typed, ready for notebooks
> and automated reports."

**Author**: Distinguished AI Researcher & Principal Engineer, Interpretability Division
**Scope**: Complete interpretability visualization toolkit for NanoSeek
**Dependencies**: `matplotlib`, `plotly`, `numpy`, `torch`, `scikit-learn`

---

## Section 1: Problem Statement

### 1.1 Why Visualization Matters for NanoSeek

NanoSeek implements MLA, MoE, MTP, and DSA simultaneously. Without visualization:

- **MoE routing is invisible.** 8 of 64 experts activate per token across 14 layers — `C(64,8)^14 ≈ 10^{132}` combinatorial space. Dead experts, specialization patterns, and routing consistency require direct visual inspection.
- **MLA attention is non-standard.** Queries/keys split into nope (64-dim) + rope (32-dim) components. Standard tools (BertViz) assume a single Q-K space and produce misleading results.
- **DSA sparse patterns need mask visualization.** The Lightning Indexer's top-k selection creates fundamentally different patterns from dense causal attention.
- **MTP agreement reveals training quality.** Disagreement between main and MTP heads signals undertrained modules or ambiguous continuations.
- **Latent space evolution tracks representation health.** The "logit lens" shows how predictions form across layers.

### 1.2 Limitations of Existing Tools

| Tool | Why It Fails for NanoSeek |
|------|--------------------------|
| **BertViz** | No MLA split Q/K, no compressed KV, no MoE |
| **TransformerLens** | Assumes HuggingFace structure, no custom MLA/MoE/DSA |
| **Circuitsvis** | No MoE routing, no sparse attention masks |
| **Ecco** | Vanilla transformers only |

### 1.3 Goal

Build `fms/interpretability/visualize.py`: a self-contained module that hooks into `model/model.py` directly, handles MLA's split Q/K, visualizes MoE routing, renders DSA sparse masks, projects latent spaces, and generates HTML dashboards.

---

## Section 2: First Principles

### 2.1 Information Visualization for Neural Networks

Following Olah et al. (2018, "The Building Blocks of Interpretability"), effective visualization targets intermediate representations — the "sweet spot" between raw weights and final predictions. For NanoSeek:

| Representation | Source | What It Reveals |
|---------------|--------|-----------------|
| Attention weights | `MultiHeadLatentAttention.forward()` | Which tokens attend to which |
| Expert indices | `Gate.forward()` | Which experts process which tokens |
| Compressed KV | `wkv_a` output, `[B, T, 143]` | What survives KV compression |
| Residual stream | Each layer's output | How representations evolve |
| Indexer scores | `LightningIndexer.forward()` | DSA token importance |
| MTP logits | `MTPModule.forward()` | Multi-token prediction quality |

### 2.2 Attention Weights ≠ Attribution

Per Jain & Wallace (2019) and Wiegreffe & Pinter (2019): **attention weights show information flow, not causation.** For NanoSeek: (1) attention heatmaps are descriptive, not causal; (2) MoE routing is more directly causal since it determines which parameters process each token; (3) the logit lens is better for understanding prediction formation.

### 2.3 Color Theory and Visualization Choices

- **`viridis`**: Perceptually uniform, colorblind-friendly — for continuous data (attention weights)
- **`tab20`**: Categorical — for discrete data (expert assignments)
- **`RdBu`**: Diverging — for data with meaningful center (logit differences)
- **Avoid**: `jet`, `rainbow` — perceptual artifacts

### 2.4 Interactive vs Static

Static (matplotlib PNG) for publications, reports, automation. Interactive (plotly HTML) for notebook exploration. Our toolkit defaults to matplotlib `Figure` objects.

### 2.5 Dimensionality Reduction

| Method | Preserves | Use When |
|--------|-----------|----------|
| **PCA** | Global variance (deterministic) | Quick overview, automated reports |
| **t-SNE** | Local neighborhoods (stochastic) | Cluster discovery — always set `random_state` |
| **UMAP** | Both local and global | Large datasets, interactive |

---

## Section 3: Production Code — `fms/interpretability/visualize.py`

### File Structure

```
fms/
└── interpretability/
    ├── __init__.py
    └── visualize.py          ← All visualization functions
```

### Complete Implementation

```python
"""
NanoSeek Activation & Attention Visualization Toolkit.

A comprehensive visualization module for understanding the internal mechanics
of NanoSeek's MLA attention, MoE routing, MTP predictions, and DSA sparse
attention patterns.

Requirements:
    pip install matplotlib plotly scikit-learn wordcloud

Usage:
    from fms.interpretability.visualize import (
        visualize_mla_attention,
        visualize_expert_routing,
        generate_interpretability_dashboard,
    )

    model = create_nanoseek(config)
    tokenizer = ...  # Any tokenizer with encode/decode

    fig = visualize_mla_attention(model, tokenizer, "Hello world", layer_idx=0)
    fig.savefig("attention.png")
"""

from __future__ import annotations

import math
import os
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from torch import Tensor

try:
    import plotly.graph_objects as go
    import plotly.subplots as sp
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from wordcloud import WordCloud

    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tokenize(tokenizer: Any, text: str) -> Tuple[Tensor, List[str]]:
    """Tokenize text → (input_ids [1, T], token_strings)."""
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(text)
        if isinstance(ids, list):
            ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        elif isinstance(ids, Tensor) and ids.dim() == 1:
            ids = ids.unsqueeze(0)
    else:
        raise ValueError("Tokenizer must have an `encode` method")

    token_strs = []
    for i in range(ids.shape[1]):
        tok_id = ids[0, i].item()
        if hasattr(tokenizer, "decode"):
            token_strs.append(tokenizer.decode([tok_id]))
        else:
            token_strs.append(str(tok_id))

    return ids, token_strs


def _get_device(model: torch.nn.Module) -> torch.device:
    """Return the device of the first parameter in the model."""
    return next(model.parameters()).device


@torch.no_grad()
def _run_with_cache(
    model: torch.nn.Module,
    input_ids: Tensor,
) -> Dict[str, Any]:
    """Forward pass with hooks capturing attn_weights, residuals, expert_indices/weights,
    gate_scores, kv_compressed, and indexer_scores per layer."""
    cache: Dict[str, Any] = {
        "attn_weights": {},
        "residuals": {},
        "expert_indices": {},
        "expert_weights": {},
        "gate_scores": {},
        "kv_compressed": {},
        "indexer_scores": {},
    }
    hooks: list = []

    for layer_idx, layer in enumerate(model.layers):

        # --- Attention weight hook ----------------------------------------
        attn_module = layer.self_attn
        if hasattr(attn_module, "mla"):
            attn_core = attn_module.mla
        else:
            attn_core = attn_module

        def _make_attn_hook(idx, core):
            orig_forward = core.forward

            def hooked_forward(*args, **kwargs):
                result = orig_forward(*args, **kwargs)
                output_tensor = result[0]
                # Recompute attention weights for capture
                hidden = args[0] if args else kwargs.get("hidden_states")
                if hidden is not None:
                    B, T, _ = hidden.shape
                    q = core.wq_a(hidden)
                    q = core.q_norm(q)
                    q = core.wq_b(q)
                    q = q.view(B, T, core.num_heads, core.qk_head_dim)
                    q_nope, q_pe = torch.split(
                        q, [core.qk_nope_head_dim, core.qk_rope_head_dim], dim=-1
                    )
                    kv = core.wkv_a(hidden)
                    kv_c, k_pe_raw = torch.split(
                        kv, [core.kv_lora_rank, core.qk_rope_head_dim], dim=-1
                    )
                    cache["kv_compressed"][idx] = kv_c.detach().cpu()
                    kv_c = core.kv_norm(kv_c)
                    from model.model import apply_rotary_emb

                    freqs = core.freqs_cis[:T]
                    k_pe_raw = k_pe_raw.unsqueeze(2)
                    k_pe = apply_rotary_emb(k_pe_raw, freqs, interleaved=True)
                    kv_exp = core.wkv_b(kv_c)
                    kv_exp = kv_exp.view(
                        B, T, core.num_heads,
                        core.qk_nope_head_dim + core.v_head_dim,
                    )
                    k_nope = kv_exp[..., : core.qk_nope_head_dim]
                    q_pe = apply_rotary_emb(q_pe, freqs, interleaved=True)
                    q_full = torch.cat([q_nope, q_pe], dim=-1)
                    k_pe_exp = k_pe.expand(-1, -1, core.num_heads, -1)
                    k_full = torch.cat([k_nope, k_pe_exp], dim=-1)
                    q_t = q_full.transpose(1, 2)
                    k_t = k_full.transpose(1, 2)
                    scores = torch.matmul(q_t, k_t.transpose(-2, -1))
                    scores = scores * core.softmax_scale
                    causal = torch.triu(
                        torch.ones(T, T, device=hidden.device), diagonal=1
                    ).bool()
                    scores.masked_fill_(causal.unsqueeze(0).unsqueeze(0), float("-inf"))
                    weights = F.softmax(scores, dim=-1, dtype=torch.float32)
                    cache["attn_weights"][idx] = weights.detach().cpu()
                return result

            core.forward = hooked_forward
            hooks.append((core, "forward", orig_forward))

        _make_attn_hook(layer_idx, attn_core)

        # --- DSA indexer hook ---------------------------------------------
        if hasattr(attn_module, "indexer"):
            indexer = attn_module.indexer
            orig_indexer_fwd = indexer.forward

            def _make_indexer_hook(idx, idxr, orig_fwd):
                def hooked_indexer(*args, **kwargs):
                    result = orig_fwd(*args, **kwargs)
                    cache["indexer_scores"][idx] = result.detach().cpu()
                    return result

                idxr.forward = hooked_indexer
                hooks.append((idxr, "forward", orig_fwd))

            _make_indexer_hook(layer_idx, indexer, orig_indexer_fwd)

        # --- Residual stream hook -----------------------------------------
        orig_layer_fwd = layer.forward

        def _make_residual_hook(idx, lyr, orig_fwd):
            def hooked_layer(*args, **kwargs):
                result = orig_fwd(*args, **kwargs)
                cache["residuals"][idx] = result[0].detach().cpu()
                return result

            lyr.forward = hooked_layer
            hooks.append((lyr, "forward", orig_fwd))

        _make_residual_hook(layer_idx, layer, orig_layer_fwd)

        # --- MoE gate hook ------------------------------------------------
        if layer.is_moe_layer:
            gate = layer.ffn.gate
            orig_gate_fwd = gate.forward

            def _make_gate_hook(idx, gt, orig_fwd):
                def hooked_gate(x):
                    weights, indices = orig_fwd(x)
                    cache["expert_indices"][idx] = indices.detach().cpu()
                    cache["expert_weights"][idx] = weights.detach().cpu()
                    scores_raw = F.linear(x, gt.weight)
                    cache["gate_scores"][idx] = scores_raw.detach().cpu()
                    return weights, indices

                gt.forward = hooked_gate
                hooks.append((gt, "forward", orig_fwd))

            _make_gate_hook(layer_idx, gate, orig_gate_fwd)

    # Run forward pass
    device = _get_device(model)
    input_ids = input_ids.to(device)
    model.eval()
    outputs = model(input_ids, output_hidden_states=True)
    cache["outputs"] = {k: v.detach().cpu() if isinstance(v, Tensor) else v
                        for k, v in outputs.items()}

    # Restore original forwards
    for obj, attr, orig in hooks:
        setattr(obj, attr, orig)

    return cache


# =========================================================================
# Section 3a: Attention Pattern Visualization
# =========================================================================


def visualize_mla_attention(
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
    layer_idx: int,
    head_idx: Optional[int] = None,
    show_compressed: bool = False,
) -> Figure:
    """MLA attention heatmap. Shows softmax weights for layer/head.
    If show_compressed=True, adds compressed KV norm panel (143-dim).

    Args:
        model: NanoSeekModel.  tokenizer: encode/decode.  text: input.
        layer_idx: 0–15.  head_idx: single head or None=average.
    Returns: matplotlib Figure.

    Example: ``fig = visualize_mla_attention(model, tok, "Hi", layer_idx=0)``
    """
    input_ids, tokens = _tokenize(tokenizer, text)
    cache = _run_with_cache(model, input_ids)

    attn = cache["attn_weights"].get(layer_idx)
    if attn is None:
        raise ValueError(f"No attention weights captured for layer {layer_idx}")

    # attn shape: [B, H, T, T] — take first batch element
    attn = attn[0]  # [H, T, T]

    ncols = 2 if show_compressed else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    if head_idx is not None:
        attn_map = attn[head_idx].numpy()
        title = f"MLA Attention — Layer {layer_idx}, Head {head_idx}"
    else:
        attn_map = attn.mean(dim=0).numpy()
        title = f"MLA Attention (avg heads) — Layer {layer_idx}"

    im = axes[0].imshow(attn_map, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    axes[0].set_title(title, fontsize=11)
    axes[0].set_xlabel("Key position")
    axes[0].set_ylabel("Query position")
    if len(tokens) <= 32:
        axes[0].set_xticks(range(len(tokens)))
        axes[0].set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
        axes[0].set_yticks(range(len(tokens)))
        axes[0].set_yticklabels(tokens, fontsize=7)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    if show_compressed:
        kv_c = cache["kv_compressed"].get(layer_idx)
        if kv_c is not None:
            norms = kv_c[0].norm(dim=-1).numpy()  # [T]
            axes[1].bar(range(len(norms)), norms, color="steelblue", alpha=0.8)
            axes[1].set_title(
                f"Compressed KV Norms (143-dim) — Layer {layer_idx}", fontsize=11
            )
            axes[1].set_xlabel("Token position")
            axes[1].set_ylabel("L2 norm")
            if len(tokens) <= 32:
                axes[1].set_xticks(range(len(tokens)))
                axes[1].set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
        else:
            axes[1].text(0.5, 0.5, "No compressed KV data",
                         transform=axes[1].transAxes, ha="center")

    fig.tight_layout()
    return fig


def visualize_attention_across_layers(
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
) -> Figure:
    """num_layers × num_heads grid of attention heatmaps.
    Returns: matplotlib Figure.
    Example: ``fig = visualize_attention_across_layers(model, tok, "Hi")``
    """
    input_ids, tokens = _tokenize(tokenizer, text)
    cache = _run_with_cache(model, input_ids)

    num_layers = len(model.layers)
    num_heads = model.config.num_heads

    fig, axes = plt.subplots(
        num_layers, num_heads,
        figsize=(num_heads * 1.2, num_layers * 1.2),
    )

    for l_idx in range(num_layers):
        attn = cache["attn_weights"].get(l_idx)
        for h_idx in range(num_heads):
            ax = axes[l_idx, h_idx] if num_layers > 1 else axes[h_idx]
            if attn is not None:
                attn_map = attn[0, h_idx].numpy()
                ax.imshow(attn_map, cmap="viridis", aspect="auto", vmin=0, vmax=1)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=6)
            ax.set_xticks([])
            ax.set_yticks([])
            if l_idx == 0:
                ax.set_title(f"H{h_idx}", fontsize=6)
            if h_idx == 0:
                ax.set_ylabel(f"L{l_idx}", fontsize=6)

    fig.suptitle("Attention Patterns: All Layers × All Heads", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def visualize_attention_entropy(
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
) -> Figure:
    """Attention entropy per head per layer. Low=sharp, high=diffuse.
    Identifies positional heads, global heads, and degenerate heads.
    Returns: matplotlib Figure (layers × heads heatmap).
    Example: ``fig = visualize_attention_entropy(model, tok, "Hello")``
    """
    input_ids, tokens = _tokenize(tokenizer, text)
    cache = _run_with_cache(model, input_ids)

    num_layers = len(model.layers)
    num_heads = model.config.num_heads

    entropy_matrix = np.zeros((num_layers, num_heads))

    for l_idx in range(num_layers):
        attn = cache["attn_weights"].get(l_idx)
        if attn is None:
            continue
        attn = attn[0]  # [H, T, T]
        for h_idx in range(num_heads):
            w = attn[h_idx]  # [T, T]
            w_clamped = w.clamp(min=1e-10)
            ent = -(w_clamped * w_clamped.log()).sum(dim=-1).mean().item()
            entropy_matrix[l_idx, h_idx] = ent

    fig, ax = plt.subplots(figsize=(max(8, num_heads * 0.6), max(5, num_layers * 0.4)))
    im = ax.imshow(entropy_matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(num_heads))
    ax.set_yticks(range(num_layers))
    ax.set_title("Attention Entropy (low = sharp, high = diffuse)")
    fig.colorbar(im, ax=ax, label="Entropy (nats)")

    for l in range(num_layers):
        for h in range(num_heads):
            val = entropy_matrix[l, h]
            ax.text(h, l, f"{val:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if val > entropy_matrix.max() * 0.6 else "black")

    fig.tight_layout()
    return fig


# =========================================================================
# Section 3b: MoE Expert Routing Visualization
# =========================================================================


def visualize_expert_routing(
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
) -> Figure:
    """Color-coded token-to-expert map (top-1 expert per token per MoE layer).
    Returns: matplotlib Figure. Example: ``visualize_expert_routing(m, t, "Hi")``
    """
    input_ids, tokens = _tokenize(tokenizer, text)
    cache = _run_with_cache(model, input_ids)

    moe_layers = sorted(cache["expert_indices"].keys())
    if not moe_layers:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "No MoE layers found", transform=ax.transAxes, ha="center")
        return fig

    seq_len = len(tokens)
    n_routed = model.config.moe.n_routed_experts

    cmap = plt.cm.get_cmap("tab20", n_routed)

    fig, axes = plt.subplots(
        len(moe_layers), 1,
        figsize=(max(10, seq_len * 0.5), len(moe_layers) * 1.0 + 1),
        squeeze=False,
    )

    for row, l_idx in enumerate(moe_layers):
        ax = axes[row, 0]
        indices = cache["expert_indices"][l_idx]  # [B*T, K]
        top1 = indices[:seq_len, 0].numpy()

        colors = [cmap(e / n_routed) for e in top1]
        ax.bar(range(seq_len), [1] * seq_len, color=colors, width=1.0, edgecolor="none")
        ax.set_xlim(-0.5, seq_len - 0.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(f"L{l_idx}", fontsize=8, rotation=0, labelpad=20)

        if row == len(moe_layers) - 1 and seq_len <= 40:
            ax.set_xticks(range(seq_len))
            ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=6)
        elif row == len(moe_layers) - 1:
            ax.set_xlabel("Token position")
        else:
            ax.set_xticks([])

    fig.suptitle("Expert Routing (top-1 expert per token, color = expert ID)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def visualize_expert_load_distribution(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: List[str],
) -> Figure:
    """Bar chart of token counts per expert across a corpus. Red line = expected
    balanced load. Identifies dead/overloaded experts.
    Returns: matplotlib Figure. Example: ``visualize_expert_load_distribution(m, t, texts)``
    """
    n_routed = model.config.moe.n_routed_experts
    load_counts = np.zeros(n_routed)

    total_tokens = 0
    for text in texts:
        input_ids, _ = _tokenize(tokenizer, text)
        cache = _run_with_cache(model, input_ids)
        seq_len = input_ids.shape[1]
        total_tokens += seq_len

        for l_idx, indices in cache["expert_indices"].items():
            flat = indices[:seq_len].numpy().flatten()
            for e in flat:
                load_counts[int(e)] += 1

    n_moe_layers = len([l for l in model.layers if l.is_moe_layer])
    k = model.config.moe.num_experts_per_tok
    expected = (total_tokens * k * n_moe_layers) / n_routed

    fig, ax = plt.subplots(figsize=(max(12, n_routed * 0.25), 5))

    colors = ["#e74c3c" if c < expected * 0.1 else
              "#f39c12" if c > expected * 2 else
              "#3498db" for c in load_counts]
    ax.bar(range(n_routed), load_counts, color=colors, edgecolor="none", alpha=0.85)
    ax.axhline(y=expected, color="red", linestyle="--", linewidth=1.5,
               label=f"Expected load ({expected:.0f})")
    ax.set_xlabel("Expert ID")
    ax.set_ylabel("Token count (across all layers)")
    ax.set_title(f"Expert Load Distribution ({len(texts)} texts, {total_tokens} tokens)")
    ax.legend()
    fig.tight_layout()
    return fig


def visualize_expert_coactivation_matrix(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: List[str],
) -> Figure:
    """n_experts × n_experts heatmap of co-selection frequency. Reveals expert
    cliques, complementary pairs, and redundancy. Returns: matplotlib Figure.
    """
    n_routed = model.config.moe.n_routed_experts
    coactivation = np.zeros((n_routed, n_routed), dtype=np.float64)

    for text in texts:
        input_ids, _ = _tokenize(tokenizer, text)
        cache = _run_with_cache(model, input_ids)
        seq_len = input_ids.shape[1]

        for l_idx, indices in cache["expert_indices"].items():
            for t in range(min(seq_len, indices.shape[0])):
                experts = indices[t].numpy()
                for i in range(len(experts)):
                    for j in range(i + 1, len(experts)):
                        ei, ej = int(experts[i]), int(experts[j])
                        coactivation[ei, ej] += 1
                        coactivation[ej, ei] += 1

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(coactivation, cmap="YlOrRd", aspect="equal")
    ax.set_xlabel("Expert ID")
    ax.set_ylabel("Expert ID")
    ax.set_title("Expert Co-activation Matrix")
    fig.colorbar(im, ax=ax, label="Co-activation count")
    fig.tight_layout()
    return fig


def visualize_expert_specialization_wordcloud(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: List[str],
    expert_id: int,
) -> Figure:
    """Word cloud of tokens most frequently routed to expert_id (0–63).
    Word size ∝ routing frequency. Returns: matplotlib Figure.
    """
    if not HAS_WORDCLOUD:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Install `wordcloud` package:\npip install wordcloud",
                transform=ax.transAxes, ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        return fig

    token_freq: Dict[str, int] = defaultdict(int)

    for text in texts:
        input_ids, token_strs = _tokenize(tokenizer, text)
        cache = _run_with_cache(model, input_ids)
        seq_len = input_ids.shape[1]

        for l_idx, indices in cache["expert_indices"].items():
            for t in range(min(seq_len, indices.shape[0])):
                experts = indices[t].numpy().tolist()
                if expert_id in experts:
                    tok = token_strs[t].strip()
                    if tok:
                        token_freq[tok] += 1

    fig, ax = plt.subplots(figsize=(10, 5))
    if token_freq:
        wc = WordCloud(
            width=1000, height=500,
            background_color="white",
            max_words=100,
            colormap="viridis",
        ).generate_from_frequencies(token_freq)
        ax.imshow(wc, interpolation="bilinear")
    else:
        ax.text(0.5, 0.5, f"No tokens routed to Expert {expert_id}",
                transform=ax.transAxes, ha="center", va="center", fontsize=14)
    ax.set_axis_off()
    ax.set_title(f"Expert {expert_id} — Token Specialization", fontsize=13)
    fig.tight_layout()
    return fig


def visualize_routing_across_layers(
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
) -> Figure:
    """Top-1 expert per token across MoE layers as color-coded grid.
    Reveals cross-layer expert affinity. Returns: matplotlib Figure.
    """
    input_ids, tokens = _tokenize(tokenizer, text)
    cache = _run_with_cache(model, input_ids)

    moe_layers = sorted(cache["expert_indices"].keys())
    if not moe_layers:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "No MoE layers found", transform=ax.transAxes, ha="center")
        return fig

    seq_len = len(tokens)
    n_routed = model.config.moe.n_routed_experts
    cmap = plt.cm.get_cmap("tab20", n_routed)

    routing_matrix = np.zeros((len(moe_layers), seq_len), dtype=int)
    for row, l_idx in enumerate(moe_layers):
        indices = cache["expert_indices"][l_idx]
        routing_matrix[row] = indices[:seq_len, 0].numpy()

    fig, ax = plt.subplots(figsize=(max(10, seq_len * 0.5), max(5, len(moe_layers) * 0.6)))

    im = ax.imshow(
        routing_matrix, cmap=cmap, aspect="auto",
        vmin=0, vmax=n_routed - 1, interpolation="nearest",
    )
    ax.set_xlabel("Token position")
    ax.set_ylabel("MoE Layer")
    ax.set_yticks(range(len(moe_layers)))
    ax.set_yticklabels([f"L{l}" for l in moe_layers], fontsize=8)

    if seq_len <= 40:
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=6)

    ax.set_title("Top-1 Expert Routing Across Layers (color = expert ID)")
    fig.colorbar(im, ax=ax, label="Expert ID", ticks=range(0, n_routed, max(1, n_routed // 8)))
    fig.tight_layout()
    return fig


# =========================================================================
# Section 3c: Latent Space & Activation Visualization
# =========================================================================


def visualize_residual_stream(
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
    method: str = "pca",
) -> Figure:
    """Project 2048-dim residual stream to 2D at each layer (pca/tsne/umap).
    Reveals cluster formation, collapse, and layer transitions.
    Returns: matplotlib Figure.
    """
    if not HAS_SKLEARN:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Install scikit-learn: pip install scikit-learn",
                transform=ax.transAxes, ha="center")
        return fig

    input_ids, tokens = _tokenize(tokenizer, text)
    cache = _run_with_cache(model, input_ids)

    num_layers = len(model.layers)
    all_acts = []
    layer_labels = []
    for l_idx in range(num_layers):
        res = cache["residuals"].get(l_idx)
        if res is not None:
            acts = res[0].numpy()  # [T, D]
            all_acts.append(acts)
            layer_labels.extend([l_idx] * acts.shape[0])

    if not all_acts:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No residual activations captured", transform=ax.transAxes, ha="center")
        return fig

    all_acts = np.concatenate(all_acts, axis=0)  # [num_layers * T, D]
    layer_labels = np.array(layer_labels)

    if method == "pca":
        reducer = PCA(n_components=2)
        proj = reducer.fit_transform(all_acts)
    elif method == "tsne":
        perplexity = min(30, max(2, all_acts.shape[0] // 4))
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        proj = reducer.fit_transform(all_acts)
    elif method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            proj = reducer.fit_transform(all_acts)
        except ImportError:
            warnings.warn("UMAP not available, falling back to PCA")
            reducer = PCA(n_components=2)
            proj = reducer.fit_transform(all_acts)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca', 'tsne', or 'umap'.")

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.get_cmap("viridis", num_layers)

    for l_idx in range(num_layers):
        mask = layer_labels == l_idx
        ax.scatter(
            proj[mask, 0], proj[mask, 1],
            c=[cmap(l_idx / num_layers)] * mask.sum(),
            label=f"Layer {l_idx}",
            s=30, alpha=0.7, edgecolors="white", linewidths=0.3,
        )

    ax.set_title(f"Residual Stream Evolution ({method.upper()})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def visualize_logit_lens(
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
) -> Figure:
    """Logit lens: project each layer's residual through lm_head to show
    top-1 prediction at each (layer, position). Green=confident.
    Returns: matplotlib Figure.
    """
    input_ids, tokens = _tokenize(tokenizer, text)
    cache = _run_with_cache(model, input_ids)

    num_layers = len(model.layers)
    seq_len = len(tokens)

    lm_head_weight = model.lm_head.weight.detach().cpu()  # [V, D]
    lm_head_bias = None
    if model.lm_head.bias is not None:
        lm_head_bias = model.lm_head.bias.detach().cpu()

    # norm weight for applying RMSNorm before lm_head
    norm_weight = model.norm.weight.detach().cpu()
    norm_eps = model.norm.eps

    predictions = np.empty((num_layers, seq_len), dtype=object)
    confidence = np.zeros((num_layers, seq_len))

    for l_idx in range(num_layers):
        res = cache["residuals"].get(l_idx)
        if res is None:
            for t in range(seq_len):
                predictions[l_idx, t] = "?"
            continue

        h = res[0].float()  # [T, D]
        # Apply RMSNorm
        rms = torch.sqrt(h.pow(2).mean(-1, keepdim=True) + norm_eps)
        h_normed = (h / rms) * norm_weight

        logits = F.linear(h_normed, lm_head_weight, lm_head_bias)  # [T, V]
        probs = F.softmax(logits, dim=-1)
        top_probs, top_ids = probs.max(dim=-1)

        for t in range(seq_len):
            tok_id = top_ids[t].item()
            if hasattr(tokenizer, "decode"):
                pred_tok = tokenizer.decode([tok_id])
            else:
                pred_tok = str(tok_id)
            predictions[l_idx, t] = pred_tok[:8]
            confidence[l_idx, t] = top_probs[t].item()

    fig, ax = plt.subplots(figsize=(max(10, seq_len * 0.8), max(6, num_layers * 0.5)))
    im = ax.imshow(confidence, cmap="YlGn", aspect="auto", vmin=0, vmax=1)

    for l in range(num_layers):
        for t in range(seq_len):
            pred = predictions[l, t]
            conf = confidence[l, t]
            color = "white" if conf > 0.5 else "black"
            ax.text(t, l, pred, ha="center", va="center",
                    fontsize=max(4, min(7, 80 // seq_len)), color=color,
                    fontweight="bold" if conf > 0.8 else "normal")

    ax.set_xlabel("Token position")
    ax.set_ylabel("Layer")
    ax.set_yticks(range(num_layers))
    ax.set_title("Logit Lens — Top-1 prediction at each layer (brightness = confidence)")
    if seq_len <= 32:
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
    fig.colorbar(im, ax=ax, label="Top-1 probability")
    fig.tight_layout()
    return fig


def visualize_mla_compressed_space(
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
    layer_idx: int,
    method: str = "pca",
) -> Figure:
    """Project 143-dim compressed KV space to 2D (pca/tsne). Shows what
    survives MLA compression. Returns: matplotlib Figure.
    """
    if not HAS_SKLEARN:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Install scikit-learn: pip install scikit-learn",
                transform=ax.transAxes, ha="center")
        return fig

    input_ids, token_strs = _tokenize(tokenizer, text)
    cache = _run_with_cache(model, input_ids)

    kv_c = cache["kv_compressed"].get(layer_idx)
    if kv_c is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No compressed KV for layer {layer_idx}",
                transform=ax.transAxes, ha="center")
        return fig

    data = kv_c[0].numpy()  # [T, kv_lora_rank]
    seq_len = data.shape[0]

    if method == "pca":
        n_comp = min(2, data.shape[0], data.shape[1])
        reducer = PCA(n_components=n_comp)
        proj = reducer.fit_transform(data)
        if n_comp < 2:
            proj = np.column_stack([proj, np.zeros(seq_len)])
        explained = reducer.explained_variance_ratio_
    elif method == "tsne":
        perplexity = min(30, max(2, seq_len // 4))
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        proj = reducer.fit_transform(data)
        explained = None
    else:
        raise ValueError(f"Method must be 'pca' or 'tsne', got '{method}'")

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = np.arange(seq_len)
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=colors, cmap="viridis",
                    s=60, edgecolors="black", linewidths=0.5, zorder=5)

    for i, tok in enumerate(token_strs):
        ax.annotate(tok[:10], (proj[i, 0], proj[i, 1]),
                    fontsize=6, alpha=0.8,
                    xytext=(4, 4), textcoords="offset points")

    title = f"Compressed KV Space (143-dim → 2D, {method.upper()}) — Layer {layer_idx}"
    if explained is not None:
        title += f"\nExplained var: {explained[0]:.1%}, {explained[1]:.1%}"
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.colorbar(sc, ax=ax, label="Token position")
    fig.tight_layout()
    return fig


def visualize_activation_norms(
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
) -> Figure:
    """Mean L2 norm of residual stream per layer. Detects collapse or
    explosion. Returns: matplotlib Figure.
    """
    input_ids, tokens = _tokenize(tokenizer, text)
    cache = _run_with_cache(model, input_ids)

    num_layers = len(model.layers)
    mean_norms = []
    min_norms = []
    max_norms = []

    for l_idx in range(num_layers):
        res = cache["residuals"].get(l_idx)
        if res is not None:
            norms = res[0].norm(dim=-1)  # [T]
            mean_norms.append(norms.mean().item())
            min_norms.append(norms.min().item())
            max_norms.append(norms.max().item())
        else:
            mean_norms.append(0)
            min_norms.append(0)
            max_norms.append(0)

    layers = list(range(num_layers))
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.fill_between(layers, min_norms, max_norms, alpha=0.2, color="steelblue",
                    label="Min–Max range")
    ax.plot(layers, mean_norms, "o-", color="steelblue", linewidth=2,
            markersize=6, label="Mean norm")

    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 Norm")
    ax.set_title("Residual Stream Activation Norms Across Layers")
    ax.set_xticks(layers)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# =========================================================================
# Section 3d: MTP Prediction Visualization
# =========================================================================


def visualize_mtp_predictions(
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
) -> Figure:
    """Main head vs MTP head top-1 predictions. Green=agree, red=disagree.
    Returns: matplotlib Figure.
    """
    input_ids, tokens = _tokenize(tokenizer, text)
    device = _get_device(model)
    input_ids = input_ids.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    logits = outputs["logits"].cpu()  # [B, T, V]
    main_preds = logits[0].argmax(dim=-1)  # [T]

    mtp_preds = None
    if model.mtp is not None:
        hidden = outputs["hidden_states"].cpu()
        with torch.no_grad():
            mtp_out = model.mtp(hidden.to(device), labels=input_ids)
        if mtp_out["mtp_logits"]:
            mtp_logits = mtp_out["mtp_logits"][0].cpu()  # [B, T', V]
            mtp_preds = mtp_logits[0].argmax(dim=-1)  # [T']

    seq_len = len(tokens)
    fig, ax = plt.subplots(figsize=(max(12, seq_len * 0.6), 4))
    ax.set_axis_off()

    cell_w = 1.0 / (seq_len + 1)
    cell_h = 0.25

    # Header row
    ax.text(0, 0.9, "Position", fontsize=7, fontweight="bold", va="center")
    ax.text(0, 0.65, "Main pred", fontsize=7, fontweight="bold", va="center")
    ax.text(0, 0.4, "MTP pred", fontsize=7, fontweight="bold", va="center")
    ax.text(0, 0.15, "Agree?", fontsize=7, fontweight="bold", va="center")

    display_len = min(seq_len - 1, 40)
    for t in range(display_len):
        x = (t + 1) * cell_w

        # Position label
        tok_str = tokens[t][:6] if t < len(tokens) else "?"
        ax.text(x, 0.9, tok_str, fontsize=5, va="center", ha="center")

        # Main prediction
        main_id = main_preds[t].item()
        main_str = tokenizer.decode([main_id])[:6] if hasattr(tokenizer, "decode") else str(main_id)
        ax.text(x, 0.65, main_str, fontsize=5, va="center", ha="center")

        # MTP prediction
        if mtp_preds is not None and t < mtp_preds.shape[0]:
            mtp_id = mtp_preds[t].item()
            mtp_str = tokenizer.decode([mtp_id])[:6] if hasattr(tokenizer, "decode") else str(mtp_id)
            ax.text(x, 0.4, mtp_str, fontsize=5, va="center", ha="center")

            agree = main_id == mtp_id
            color = "#2ecc71" if agree else "#e74c3c"
            ax.add_patch(plt.Rectangle(
                (x - cell_w / 2, 0.05), cell_w, 0.2,
                facecolor=color, alpha=0.4, edgecolor="none",
            ))
            ax.text(x, 0.15, "✓" if agree else "✗", fontsize=7,
                    va="center", ha="center", color=color, fontweight="bold")
        else:
            ax.text(x, 0.4, "—", fontsize=5, va="center", ha="center")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1)
    ax.set_title("Main Head vs MTP Head Predictions", fontsize=11)
    fig.tight_layout()
    return fig


def visualize_mtp_agreement(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: List[str],
) -> Figure:
    """Agreement rate between main and MTP predictions vs token position.
    Aggregated across corpus. Returns: matplotlib Figure.
    """
    if model.mtp is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Model has no MTP module", transform=ax.transAxes, ha="center")
        return fig

    agree_by_pos: Dict[int, List[bool]] = defaultdict(list)
    device = _get_device(model)

    for text in texts:
        input_ids, _ = _tokenize(tokenizer, text)
        input_ids = input_ids.to(device)
        model.eval()

        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)

        logits = outputs["logits"].cpu()
        main_preds = logits[0].argmax(dim=-1)

        hidden = outputs["hidden_states"].cpu()
        with torch.no_grad():
            mtp_out = model.mtp(hidden.to(device), labels=input_ids)
        if mtp_out["mtp_logits"]:
            mtp_logits = mtp_out["mtp_logits"][0].cpu()
            mtp_preds = mtp_logits[0].argmax(dim=-1)

            compare_len = min(main_preds.shape[0], mtp_preds.shape[0])
            for t in range(compare_len):
                agree_by_pos[t].append(main_preds[t].item() == mtp_preds[t].item())

    if not agree_by_pos:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No MTP predictions captured", transform=ax.transAxes, ha="center")
        return fig

    positions = sorted(agree_by_pos.keys())
    rates = [np.mean(agree_by_pos[p]) for p in positions]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(positions, rates, "o-", color="steelblue", linewidth=2, markersize=4)
    ax.axhline(y=np.mean(rates), color="red", linestyle="--", alpha=0.5,
               label=f"Overall: {np.mean(rates):.1%}")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Agreement rate")
    ax.set_title(f"Main ↔ MTP Agreement Rate ({len(texts)} texts)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# =========================================================================
# Section 3e: DSA Sparse Attention Patterns
# =========================================================================


def visualize_sparse_attention_pattern(
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
    layer_idx: int,
) -> Figure:
    """Dense causal mask vs DSA sparse mask side-by-side. Shows sparsity
    pattern from Lightning Indexer top-k selection. Returns: matplotlib Figure.
    """
    input_ids, tokens = _tokenize(tokenizer, text)
    cache = _run_with_cache(model, input_ids)
    seq_len = len(tokens)

    indexer_scores = cache["indexer_scores"].get(layer_idx)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: causal mask (dense baseline)
    causal = np.tril(np.ones((seq_len, seq_len)))
    axes[0].imshow(causal, cmap="Greys", aspect="auto", vmin=0, vmax=1)
    axes[0].set_title(f"Dense Causal Mask — Layer {layer_idx}", fontsize=10)
    axes[0].set_xlabel("Key position")
    axes[0].set_ylabel("Query position")

    # Panel 2: sparse mask from indexer
    if indexer_scores is not None:
        scores = indexer_scores[0].numpy()  # [T, T]
        topk = model.config.sparse.topk_tokens
        k = min(topk, seq_len)

        sparse_mask = np.zeros_like(scores)
        for q in range(seq_len):
            valid_scores = scores[q, :q + 1].copy()
            if len(valid_scores) > k:
                threshold = np.sort(valid_scores)[-k]
                sparse_mask[q, :q + 1] = (scores[q, :q + 1] >= threshold).astype(float)
            else:
                sparse_mask[q, :q + 1] = 1.0

        axes[1].imshow(sparse_mask, cmap="Blues", aspect="auto", vmin=0, vmax=1)
        sparsity = 1.0 - sparse_mask.sum() / causal.sum()
        axes[1].set_title(
            f"DSA Sparse Mask — Layer {layer_idx} (sparsity: {sparsity:.1%})",
            fontsize=10,
        )
    else:
        axes[1].imshow(causal, cmap="Blues", aspect="auto", vmin=0, vmax=1)
        axes[1].set_title(
            f"Layer {layer_idx} — No indexer (dense fallback)", fontsize=10
        )

    axes[1].set_xlabel("Key position")
    axes[1].set_ylabel("Query position")

    if seq_len <= 32:
        for ax in axes:
            ax.set_xticks(range(seq_len))
            ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=6)
            ax.set_yticks(range(seq_len))
            ax.set_yticklabels(tokens, fontsize=6)

    fig.suptitle("Dense vs Sparse Attention Mask Comparison", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def visualize_indexer_scores(
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
    layer_idx: int,
) -> Figure:
    """Raw Lightning Indexer scores before top-k masking. Shows which tokens
    the indexer considers important per query. Returns: matplotlib Figure.
    """
    input_ids, tokens = _tokenize(tokenizer, text)
    cache = _run_with_cache(model, input_ids)
    seq_len = len(tokens)

    indexer_scores = cache["indexer_scores"].get(layer_idx)

    fig, ax = plt.subplots(figsize=(8, 7))

    if indexer_scores is not None:
        scores = indexer_scores[0].numpy()  # [T, T]
        # Apply causal masking for display
        causal = np.triu(np.ones_like(scores), k=1)
        scores_masked = np.where(causal, np.nan, scores)

        im = ax.imshow(scores_masked, cmap="inferno", aspect="auto")
        ax.set_title(f"Lightning Indexer Scores — Layer {layer_idx}", fontsize=11)
        fig.colorbar(im, ax=ax, label="Indexer score")
    else:
        ax.text(0.5, 0.5, f"No indexer scores for layer {layer_idx}\n"
                           "(DSA may be disabled)",
                transform=ax.transAxes, ha="center", va="center", fontsize=12)
        ax.set_title(f"Indexer Scores — Layer {layer_idx} (not available)", fontsize=11)

    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")

    if seq_len <= 32:
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(seq_len))
        ax.set_yticklabels(tokens, fontsize=7)

    fig.tight_layout()
    return fig


# =========================================================================
# Section 3f: Comprehensive Dashboard
# =========================================================================


def generate_interpretability_dashboard(
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
    output_dir: str,
) -> None:
    """Generate full HTML dashboard with all visualizations to output_dir.
    Creates PNGs + index.html. Example: ``generate_interpretability_dashboard(m, t, "Hi", "./out")``
    """
    os.makedirs(output_dir, exist_ok=True)
    figures: Dict[str, str] = {}

    def _save(name: str, fig_fn, **kwargs):
        try:
            fig = fig_fn(**kwargs)
            path = os.path.join(output_dir, f"{name}.png")
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            figures[name] = f"{name}.png"
        except Exception as e:
            warnings.warn(f"Failed to generate {name}: {e}")

    # Attention visualizations
    _save("attn_layer0", visualize_mla_attention,
          model=model, tokenizer=tokenizer, text=text, layer_idx=0, show_compressed=True)
    _save("attn_entropy", visualize_attention_entropy,
          model=model, tokenizer=tokenizer, text=text)
    num_layers = len(model.layers)
    if num_layers <= 20 and len(text.split()) <= 50:
        _save("attn_grid", visualize_attention_across_layers,
              model=model, tokenizer=tokenizer, text=text)

    # Expert routing
    _save("expert_routing", visualize_expert_routing,
          model=model, tokenizer=tokenizer, text=text)
    _save("expert_load", visualize_expert_load_distribution,
          model=model, tokenizer=tokenizer, texts=[text])
    _save("expert_coactivation", visualize_expert_coactivation_matrix,
          model=model, tokenizer=tokenizer, texts=[text])
    _save("routing_flow", visualize_routing_across_layers,
          model=model, tokenizer=tokenizer, text=text)

    # Latent space
    _save("residual_pca", visualize_residual_stream,
          model=model, tokenizer=tokenizer, text=text, method="pca")
    _save("logit_lens", visualize_logit_lens,
          model=model, tokenizer=tokenizer, text=text)
    _save("compressed_kv", visualize_mla_compressed_space,
          model=model, tokenizer=tokenizer, text=text, layer_idx=0, method="pca")
    _save("activation_norms", visualize_activation_norms,
          model=model, tokenizer=tokenizer, text=text)

    # MTP
    _save("mtp_predictions", visualize_mtp_predictions,
          model=model, tokenizer=tokenizer, text=text)
    _save("mtp_agreement", visualize_mtp_agreement,
          model=model, tokenizer=tokenizer, texts=[text])

    # DSA
    _save("sparse_mask_l4", visualize_sparse_attention_pattern,
          model=model, tokenizer=tokenizer, text=text, layer_idx=4)
    _save("indexer_scores_l4", visualize_indexer_scores,
          model=model, tokenizer=tokenizer, text=text, layer_idx=4)

    # Generate HTML dashboard
    sections = {
        "Attention Patterns": ["attn_layer0", "attn_entropy", "attn_grid"],
        "Expert Routing": ["expert_routing", "expert_load", "expert_coactivation", "routing_flow"],
        "Latent Space": ["residual_pca", "logit_lens", "compressed_kv", "activation_norms"],
        "MTP Predictions": ["mtp_predictions", "mtp_agreement"],
        "DSA Sparse Attention": ["sparse_mask_l4", "indexer_scores_l4"],
    }

    css = ("body{font-family:system-ui;max-width:1400px;margin:0 auto;padding:20px}"
           ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(500px,1fr));gap:20px}"
           ".card{background:#fff;border-radius:8px;padding:16px;box-shadow:0 2px 4px rgba(0,0,0,.1)}"
           ".card img{width:100%;border-radius:4px}")
    html_parts = [f"<!DOCTYPE html><html><head><style>{css}</style></head><body>",
                  f"<h1>NanoSeek Interpretability Dashboard</h1>",
                  f"<p><b>Input:</b> <code>{text[:200]}</code></p>"]
    for title, keys in sections.items():
        html_parts.append(f"<h2>{title}</h2><div class='grid'>")
        for k in keys:
            if k in figures:
                html_parts.append(f"<div class='card'><h3>{k.replace('_',' ').title()}</h3>"
                                  f"<img src='{figures[k]}'></div>")
        html_parts.append("</div>")
    html_parts.append("</body></html>")

    html_path = os.path.join(output_dir, "index.html")
    with open(html_path, "w") as f:
        f.write("\n".join(html_parts))

    print(f"Dashboard written to {html_path}")
    print(f"Generated {len(figures)} visualizations")
```

---

## Section 4: Visualization Gallery

ASCII representations of key visualizations, annotated with what to look for.

### 4.1 MLA Attention Heatmap

```
              Key position →
              The  cat  sat  on  the  mat
         ┌─────────────────────────────────┐
    The  │ ██░░ ░░░░ ░░░░ ░░░░ ░░░░ ░░░░  │  ← Self-attention only
    cat  │ ▓▓░░ ██░░ ░░░░ ░░░░ ░░░░ ░░░░  │  ← Attends to "The"
Q   sat  │ ░░░░ ▓▓░░ ██░░ ░░░░ ░░░░ ░░░░  │  ← Attends to "cat"
    the  │ ▓▓░░ ░░░░ ░░░░ ░░░░ ██░░ ░░░░  │  ← Links to first "The"
    mat  │ ░░░░ ▓▓░░ ░░░░ ░░░░ ░░░░ ██░░  │  ← Noun-noun link
         └─────────────────────────────────┘
    ██ high (>0.3)  ▓▓ medium (0.1-0.3)  ░░ low (<0.1)
    LOOK FOR: diagonal=self-attn, vertical stripes=anchor tokens
```

### 4.2 Expert Load Distribution

```
    Count ┌───────────────────────────────────┐
    600   │   ▓   ▓   ▓       ▓   ▓   ▓      │ ← Expected line
    400 - │ - ▓ - ▓ - ▓ - ▓ - ▓ - ▓ - ▓ - ▓ -│ --------
    200   │   ▓   ▓   ▓   ▓   ▓   ▓   ▓   ▓  │
      0   │   ░                            ░   │ ← Dead experts!
          └───────────────────────────────────┘
           E0  E8  E16 E24 E32 E40 E48  E63
    Red=dead (<10% expected), Orange=overloaded (>200%), Blue=balanced
```

### 4.3 Logit Lens

```
    Position →  The   capital  of   France  is   [pred]
              ┌───────────────────────────────────────┐
    Layer 0   │  a     the     of    the    the    a  │ ← Noise
    Layer 8   │  the   of    France France  is   Paris│ ← "Paris" emerges
    Layer 15  │  cat   of    France France  is   Paris│ ← Final prediction
              └───────────────────────────────────────┘
    Green=high confidence, White=low. LOOK FOR: at which layer does
    the correct prediction appear? Gradual or sudden?
```

### 4.4 Sparse Attention Mask

```
    DENSE CAUSAL               DSA SPARSE (top-k=4)
    ┌────────────┐             ┌────────────┐
    │ ■          │             │ ■          │
    │ ■ ■        │             │ ■ ■        │
    │ ■ ■ ■      │             │   ■ ■      │ ← Skips token 0
    │ ■ ■ ■ ■    │             │ ■   ■ ■    │
    │ ■ ■ ■ ■ ■  │             │ ■     ■ ■  │
    └────────────┘             └────────────┘
    O(T²)                       O(T×k), sparsity ≈ 50-75%
    LOOK FOR: recent tokens always selected, anchor tokens from far back
```

---

## Section 5: File Placement

```
fms/interpretability/visualize.py    ← ALL VISUALIZATION CODE (this doc's Section 3)
fms/interpretability/__init__.py     ← Re-exports all 17 public functions
tests/test_visualize.py              ← Visualization unit tests
```

**Dependencies** (`pyproject.toml` extra):
```toml
[project.optional-dependencies]
interpretability = ["matplotlib>=3.7", "plotly>=5.15", "scikit-learn>=1.3", "wordcloud>=1.9"]
```

---

## Section 6: Usage Examples

### 6.1 Analyze a Single Prompt End-to-End

```python
import torch
from model.model import create_nanoseek
from model.config import NanoSeekConfig, MLAConfig, MoEConfig, MTPConfig

config = NanoSeekConfig(
    hidden_size=256, num_layers=4, num_heads=4, vocab_size=1024,
    intermediate_size=512,
    mla=MLAConfig(q_lora_rank=64, kv_lora_rank=32, qk_nope_head_dim=32,
                  qk_rope_head_dim=16, v_head_dim=32),
    moe=MoEConfig(n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1,
                  moe_intermediate_size=128, n_group=2, topk_group=2,
                  first_k_dense_replace=1),
    mtp=MTPConfig(num_mtp_modules=1, mtp_num_heads=4),
)
model = create_nanoseek(config)

class SimpleTokenizer:
    def encode(self, text): return [hash(c) % 1024 for c in text.split()]
    def decode(self, ids): return "".join([chr(65 + (i % 26)) for i in ids])

tokenizer = SimpleTokenizer()
text = "The quick brown fox jumps over the lazy dog"

from fms.interpretability.visualize import generate_interpretability_dashboard
generate_interpretability_dashboard(model, tokenizer, text, output_dir="./viz_output")
```

### 6.2 Compare Expert Routing Across Text Domains

```python
from fms.interpretability.visualize import (
    visualize_expert_load_distribution, visualize_expert_specialization_wordcloud,
)

code_texts = ["def fib(n): return fib(n-1)+fib(n-2)", "import torch"]
english_texts = ["The quick brown fox", "Scientists discovered a species"]
math_texts = ["Let f(x) = x^2 + 3x", "The integral of sin(x) equals 2"]

for name, texts in [("code", code_texts), ("eng", english_texts), ("math", math_texts)]:
    fig = visualize_expert_load_distribution(model, tokenizer, texts)
    fig.savefig(f"expert_load_{name}.png")

for eid in [0, 7, 15, 31]:
    fig = visualize_expert_specialization_wordcloud(
        model, tokenizer, code_texts + english_texts + math_texts, eid)
    fig.savefig(f"expert_{eid}_wordcloud.png")
```

### 6.3 Track Representation Evolution Through Layers

```python
from fms.interpretability.visualize import (
    visualize_residual_stream, visualize_activation_norms,
    visualize_logit_lens, visualize_mla_compressed_space,
)

text = "The capital of France is"

visualize_residual_stream(model, tokenizer, text, method="pca").savefig("residual_pca.png")
visualize_residual_stream(model, tokenizer, text, method="tsne").savefig("residual_tsne.png")
visualize_activation_norms(model, tokenizer, text).savefig("activation_norms.png")
visualize_logit_lens(model, tokenizer, text).savefig("logit_lens.png")

for layer in [0, 4, 8, 15]:
    visualize_mla_compressed_space(model, tokenizer, text, layer_idx=layer).savefig(
        f"compressed_kv_L{layer}.png")
```

### 6.4 DSA Sparse Attention Analysis

```python
from model.config import SparseAttentionConfig
from fms.interpretability.visualize import (
    visualize_sparse_attention_pattern, visualize_indexer_scores,
)

# Enable DSA with low threshold for demo
config.sparse = SparseAttentionConfig(
    enabled=True, topk_tokens=8, activation_threshold=4, dense_warmup_steps=0)
model_dsa = create_nanoseek(config)

text = "The quick brown fox jumps over the lazy dog near the old bridge"
for layer_idx in [1, 2, 3]:
    visualize_sparse_attention_pattern(model_dsa, tokenizer, text, layer_idx).savefig(
        f"sparse_L{layer_idx}.png")
    visualize_indexer_scores(model_dsa, tokenizer, text, layer_idx).savefig(
        f"indexer_L{layer_idx}.png")
```

---

## Section 7: Gotchas

### 7.1 Matplotlib Backend in Headless Environments

The module sets `matplotlib.use("Agg")` at import time. If you import `matplotlib.pyplot` elsewhere *before* importing the visualization module, the backend may be locked to a different choice. **Import the visualization module first**, or set the backend manually.

### 7.2 Memory Explosion with Full Attention Capture

Attention weights for 16 layers at seq_len=4096: `16 × [1, 16, 4096, 4096] × 4B = 17.1 GB`. **Mitigations**: use short sequences (<512), visualize one layer at a time, or increase system RAM. The cache stores on CPU to avoid GPU OOM.

### 7.3 t-SNE Non-Determinism

t-SNE produces different layouts each run — unsuitable for automated testing or cross-run comparison. **Always set `random_state=42`** (the module does this). Prefer PCA for reproducible reports.

### 7.4 MLA Attention Has Both Nope and Rope Components

`score = q_nope · k_nope^T + q_rope · k_rope^T` — semantic + positional signals are mixed. Visualizations show the combined score. The `show_compressed=True` option provides partial insight via compressed KV norms. Fully disentangling nope vs. rope contributions is a future extension.

### 7.5 Shared Experts Not Shown in Routing Visualizations

The 2 shared experts process every token unconditionally and are not selected via the gate. Routing visualizations show only routed expert assignments. A "dead" routed expert may be fine if shared experts cover its functionality.

### 7.6 Plotly HTML Files Can Be Very Large

For attention matrices at seq_len=512: `512² × 16 × 16 = 67M values → ~200MB plotly HTML`. Use matplotlib (PNG) for long sequences. Dashboard uses PNGs by default.

### 7.7 Gate Scores vs. Gate Weights

`gate_scores` = raw logits pre-sigmoid. `expert_weights` = final weights after top-k + scaling. `expert_indices` = which experts selected. Use indices for routing decisions, weights for confidence, scores for understanding rejections.

### 7.8 Untrained Models Produce Noise

Random init → meaningless visualizations. Expect interpretable patterns after ~100M tokens for the default config, fewer for reduced configs.

---

## Closing

> *"The purpose of visualization is insight, not pictures."*
> — **Ben Shneiderman** (1999)

17 functions, one dashboard generator, all four NanoSeek innovations (MLA, MoE, MTP, DSA). Each function answers a specific question: attention heatmaps → "where does the model look?", expert routing → "which parameters process this token?", logit lens → "when does the model know the answer?", activation norms → "is the representation healthy?" Use these tools to systematically interrogate NanoSeek's computational strategy — not as an end, but as a beginning.
