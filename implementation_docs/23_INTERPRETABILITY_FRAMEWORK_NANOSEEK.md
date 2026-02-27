# 23 — Interpretability & Visualization Framework for NanoSeek

## Principal Engineer's Implementation Guide — February 2026

**Author**: Distinguished AI Researcher & Principal Engineer, Interpretability Division
**Scope**: Complete mechanistic interpretability framework — hook-based activation capture, MLA latent space extraction, MoE routing analysis, MTP prediction alignment, DSA sparse attention visualization, and full-suite reporting
**Prerequisites**: `00_MASTER_PLAN.md`, `11_COMPLETE_FORWARD_BACKWARD_WALKTHROUGH.md` (tensor flow), `03_FUSED_MOE_EXPERT_PARALLELISM.md` (MoE internals), `02_FLASH_MLA_ATTENTION_KERNEL.md` (MLA internals), `08_SPECULATIVE_DECODING_MTP.md` (MTP internals)
**Outcome**: A production-grade `fms/interpretability/framework.py` module that instruments any NanoSeek checkpoint, captures activations from all four innovations (MLA, MoE, MTP, DSA), runs standardized analyses, and produces structured `InterpretabilityReport` artifacts

---

## 1. Problem Statement

NanoSeek implements four architectural innovations that each introduce distinct opacity. Understanding what the model learns — and where it fails — requires interpretability tools designed specifically for these innovations. Generic transformer techniques are necessary but insufficient because NanoSeek departs from vanilla transformers in ways that change what "attention" and "neuron" mean.

### 1a. Why Interpretability Matters for NanoSeek Specifically

**MoE routing opacity**: 64 routed experts, 8 active per token. The gate's sigmoid scoring with bias-based load balancing creates implicit expert specialization. What do the experts specialize in — syntax, semantics, domain knowledge, positional patterns? Without interpretability tools, the gate is a black box. If one expert is systematically over-loaded, the only symptom is subtle quality degradation benchmarks might miss.

**MLA compression information loss**: MLA projects K,V through a `kv_lora_rank=143` bottleneck — compressing 4096 dimensions into 175 dimensions (143 + 32 RoPE). This 23× compression is the efficiency win, but the model must encode all attention-relevant information into a space 23× smaller. What information survives? What gets lost?

**MTP planning behavior**: Multi-Token Prediction forces the model to predict token `t+2` via a cross-attention module. Does this create genuine multi-step planning, or does MTP learn a shallow shortcut? Comparing MTP and main hidden states reveals whether distinct planning representations emerge.

**DSA attention sparsity**: The Lightning Indexer selects `top-k` important tokens for sparse attention. Which tokens does it consider important — structural patterns (paragraph starts, punctuation) or semantic patterns (topically relevant tokens)?

### 1b. Questions We Want to Answer

| Question | Component | Method |
|----------|-----------|--------|
| Which experts specialize in what content types? | MoE | Expert activation profiling by input category |
| Do any experts become "dead" (never activated)? | MoE | Expert load distribution analysis |
| What information survives KV compression? | MLA | Latent space probing classifiers |
| Are some MLA heads redundant after compression? | MLA | Head importance scoring via ablation |
| Does MTP create genuine planning representations? | MTP | CKA similarity between MTP and main hidden states |
| Which tokens does DSA consider "important"? | DSA | Lightning Indexer selection pattern analysis |
| Where do model failures concentrate? | All | Error case activation clustering |
| Does the model use circuits for specific tasks? | All | Activation patching / causal tracing |

### 1c. Interpretability vs Opacity by Component

| Component | What's Interpretable | What's Opaque |
|-----------|---------------------|---------------|
| **MLA** | Attention weights (post-decompression), RoPE frequencies | Compressed latent space semantics, information loss in `wkv_a` |
| **MoE** | Expert selection per token (indices/weights), load distribution | Expert internal specialization, gate decision boundary geometry |
| **MTP** | Predicted token distributions, loss per module | Cross-attention learned alignment, planning depth |
| **DSA** | Selected token indices, indexer scores | Why specific tokens are selected, indexer-attention interaction |
| **Residual stream** | Hidden state norms, layer-wise contribution | Feature superposition, polysemantic neurons |

---

## 2. First Principles — Interpretability Taxonomy

### 2a. Post-Hoc Explainability vs Mechanistic Interpretability

**Post-hoc explainability** treats the model as a black box: "what correlates with this output?" Methods include gradient-based attribution, LIME/SHAP, and attention visualization. Useful for debugging individual predictions but does not explain *how* the model computes answers — correlation, not causation.

**Mechanistic interpretability** treats the model as a program: "what algorithm does this implement?" The goal is to reverse-engineer computational subgraphs into human-understandable features, circuits, and algorithms. Key references:

- **Olah et al. (2020)** "Zoom In: An Introduction to Circuits" — Neural networks implement modular, reusable computational subgraphs (circuits) that compose to produce behavior.
- **Elhage et al. (2022)** "Toy Models of Superposition" — Neural networks represent more features than dimensions by encoding features in superposition (overlapping, non-orthogonal directions). The central challenge of mechanistic interpretability.
- **Bricken et al. (2023)** / **Templeton et al. (2024)** "Scaling Monosemanticity" — Sparse Autoencoders (SAEs) extract interpretable features from superposed representations at scale, producing millions of human-interpretable features.

| Approach | NanoSeek Application | When to Use |
|----------|---------------------|-------------|
| Post-hoc | Expert load heatmaps, attention pattern visualization | Quick diagnostics, monitoring |
| Mechanistic | Latent space probing, circuit identification, causal tracing | Deep analysis, failure diagnosis |

### 2b. Why MoE Models Are Uniquely Suited for Interpretability

MoE architectures have a structural advantage: **natural modularity**. Each expert is a self-contained FFN analyzable independently:

1. **Expert-as-unit analysis**: Instead of polysemantic neurons, analyze entire experts. An expert that activates consistently for code tokens is more interpretable than individual neurons within a dense FFN.
2. **Routing as clustering signal**: The gate's selection function *already partitions* the token space — a learned, natural lens for understanding what features the model considers important.
3. **Ablation without retraining**: Removing a single expert (gate score → -∞) tests its role without retraining. In dense models, ablating a neuron disrupts all activations passing through it.

DeepSeek's own analysis (V3 Technical Report, §5.4) found experts develop specialization along topic dimensions (science, code, multilingual) and functional dimensions (syntax, reasoning, factual recall).

### 2c. MLA's Latent Space: Challenge and Opportunity

**Challenge — Compressed representations are opaque**: `wkv_a` maps `2048 → 175`. Unlike standard MHA where K/V are per-head projections, MLA's compressed representation mixes all heads' information into a shared bottleneck. You cannot "read off" what a head attends to from the compressed KV.

**Opportunity — Low rank = fewer independent features**: The compressed KV lies on a 143-dimensional manifold. Linear probes can test what information is retained with relatively few parameters. PCA/SVD reveals dominant axes of variation. Comparing spectra of compressed KV vs original hidden states reveals what gets amplified and suppressed.

### 2d. Feature Superposition and Why It Matters for MoE

Elhage et al. (2022) showed neural networks in the superposition regime encode `M >> N` features in `N` dimensions using almost-orthogonal directions, making neurons **polysemantic**. For MoE, the gate's routing decision is based on superposed features — a token activating expert #17 might do so because of superposed [scientific terminology] + [formal register] + [long-range dependency].

SAEs address this by learning an overcomplete basis decomposing superposed representations into monosemantic features. Applying SAEs to gate inputs reveals what features drive routing; on expert outputs, what each expert contributes to the residual stream.

### 2e. The Interpretability Hierarchy

```
Level 0: WEIGHTS    — Raw parameter matrices. Interpretable for embeddings only.
Level 1: NEURONS    — Individual activation dimensions. Often polysemantic.
Level 2: FEATURES   — Learned directions in activation space. SAEs extract these.
Level 3: CIRCUITS   — Connected subgraphs implementing a computation.
Level 4: BEHAVIORS  — Human-observable capabilities (evaluation benchmarks, Doc 10).
```

Our framework operates at levels 1–3.

### 2f. Causal Intervention vs Correlation-Based Analysis

**Correlation**: "Expert #17 activates more for code tokens" → might be co-activated with the actual code expert.
**Causal**: "Zeroing expert #17's output for code tokens drops code quality 30%" → causally necessary.

The framework supports both: correlation via activation capture and probing; causal via activation patching (Conmy et al. 2023), causal tracing (Meng et al. 2022), and expert ablation.

### 2g. Published Methods Applied to NanoSeek

| Method | Citation | NanoSeek Application |
|--------|----------|---------------------|
| **Sparse Autoencoders** | Bricken et al. 2023, Anthropic | Extract monosemantic features from residual stream and gate inputs |
| **Activation Patching** | Conmy et al. 2023 | Identify causal circuits for specific behaviors |
| **Logit Lens** | nostalgebraist 2020 | Project intermediate hidden states to vocab space layer-by-layer |
| **Causal Tracing** | Meng et al. 2022 | Locate where factual knowledge is stored |
| **Probing Classifiers** | Belinkov & Glass 2019 | Linear probes on compressed KV for linguistic information |
| **CKA Similarity** | Kornblith et al. 2019 | Compare MTP hidden states to main model hidden states |
| **Direct Logit Attribution** | Elhage et al. 2021 | Decompose logit predictions into per-head, per-layer contributions |

---

## 3. NanoSeek Interpretability Research Plan

### 3a. Core Framework: `fms/interpretability/framework.py`

```python
"""
NanoSeek Interpretability Framework — hook-based activation capture and analysis
for MLA, MoE, MTP, and DSA components.

Design: Hooks are registered lazily and removed after analysis to prevent leaks.
All captured tensors are detached and moved to CPU by default. Analysis methods
return structured dicts with documented tensor shapes. The framework never
modifies model weights (read-only introspection).

Reference architecture (model/model.py):
    NanoSeekModel
    ├── embed_tokens: Embedding(65536, 2048)
    ├── layers: ModuleList[NanoSeekDecoderLayer × 16]
    │   ├── self_attn: MLA | DSA
    │   │   ├── wq_a [2048→430], q_norm, wq_b [430→16×160]
    │   │   ├── wkv_a [2048→175], kv_norm, wkv_b [143→16×256]
    │   │   └── wo [16×128→2048]
    │   └── ffn: MoE | MLP
    │       ├── gate: Gate(2048→64), experts: [SwiGLUFFN × 64]
    │       └── shared_experts: [SwiGLUFFN × 2]
    ├── norm, lm_head [2048→65536]
    └── mtp: MultiTokenPrediction → MTPModule × 1
"""

from __future__ import annotations
import gc, json, os, warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class InterpretabilityConfig:
    """Controls which layers, heads, experts, and methods to analyze.
    
    Why a config object instead of method arguments? Because interpretability
    runs are expensive (full forward passes with hooks), and we want to declare
    the full analysis plan upfront so we can:
    1. Pre-allocate capture buffers of known size
    2. Register only the hooks we need (fewer hooks = less overhead)
    3. Produce reproducible analysis runs via serializable configs
    """
    # ── Target selection ──────────────────────────────────────────
    # Which layers to instrument. None = all layers.
    # Typical choices: [0, 7, 15] for first/middle/last, or None for sweep.
    target_layers: Optional[List[int]] = None

    # Which attention heads to capture. None = all heads.
    target_heads: Optional[List[int]] = None

    # Which experts to analyze in detail. None = all 64 routed + 2 shared.
    target_experts: Optional[List[int]] = None

    # ── Analysis methods ──────────────────────────────────────────
    # Each flag enables a specific analysis pass. Disable unused methods
    # to save memory and compute.
    analyze_mla: bool = True          # MLA latent space extraction
    analyze_moe: bool = True          # MoE routing analysis
    analyze_mtp: bool = True          # MTP prediction alignment
    analyze_dsa: bool = True          # DSA sparse attention patterns
    analyze_residual: bool = True     # Residual stream evolution
    analyze_logit_lens: bool = True   # Layer-wise token predictions

    # ── Capture settings ──────────────────────────────────────────
    # Move captured tensors to CPU immediately to prevent GPU OOM.
    capture_to_cpu: bool = True

    # Maximum tokens to capture activations for.
    # Memory budget: ~1GB per 1024 tokens per 4 analyzed layers.
    max_capture_tokens: int = 2048

    # Whether to capture attention weight matrices (expensive: O(n²) per layer).
    capture_attention_weights: bool = True

    # ── Output settings ───────────────────────────────────────────
    output_dir: str = "artifacts/interpretability"
    save_raw_activations: bool = False
    save_dtype: str = "float32"

    def get_target_layers(self, num_layers: int) -> List[int]:
        """Resolve target layers, defaulting to all if None."""
        if self.target_layers is None:
            return list(range(num_layers))
        return [l for l in self.target_layers if l < num_layers]

    def get_target_heads(self, num_heads: int) -> List[int]:
        """Resolve target heads, defaulting to all if None."""
        if self.target_heads is None:
            return list(range(num_heads))
        return [h for h in self.target_heads if h < num_heads]


class ActivationStore:
    """Stores captured activations from forward-pass hooks.

    Why not just append to a list? Because hooks fire during autograd's forward
    pass, which may be inside torch.no_grad() or inference_mode(). The store:
    1. Detaches tensors from computation graph (prevents memory leaks)
    2. Optionally moves to CPU (prevents GPU OOM during long analyses)
    3. Tracks which layer/component produced each tensor (structured access)
    """

    def __init__(self, to_cpu: bool = True):
        self._store: Dict[str, List[Tensor]] = defaultdict(list)
        self._to_cpu = to_cpu

    def capture(self, key: str, tensor: Tensor) -> None:
        """Capture a tensor under the given key. Detaches and optionally moves to CPU."""
        t = tensor.detach()
        if self._to_cpu:
            t = t.cpu()
        self._store[key].append(t)

    def get(self, key: str) -> List[Tensor]:
        """Retrieve all captured tensors for a key."""
        return self._store.get(key, [])

    def get_single(self, key: str) -> Optional[Tensor]:
        """Retrieve first capture for a key. Returns None if not captured."""
        tensors = self.get(key)
        return tensors[0] if tensors else None

    def get_stacked(self, key: str) -> Optional[Tensor]:
        """Stack all captures for a key into a single tensor (batch dim)."""
        tensors = self.get(key)
        return torch.stack(tensors, dim=0) if tensors else None

    def keys(self) -> Set[str]:
        return set(self._store.keys())

    def clear(self) -> None:
        """Release all captured tensors."""
        self._store.clear()
        gc.collect()

    def memory_usage_bytes(self) -> int:
        """Estimate total memory usage of stored tensors."""
        return sum(t.element_size() * t.nelement()
                   for tensors in self._store.values() for t in tensors)

    def summary(self) -> Dict[str, Dict[str, Any]]:
        """Return summary: key → {count, shape, dtype, bytes}."""
        return {key: {
            "count": len(ts), "shape": list(ts[0].shape),
            "dtype": str(ts[0].dtype),
            "bytes": sum(t.element_size() * t.nelement() for t in ts),
        } for key, ts in self._store.items() if ts}


@dataclass
class InterpretabilityReport:
    """Structured output from a complete interpretability analysis run."""
    mla_results: Dict[str, Any] = field(default_factory=dict)
    moe_results: Dict[str, Any] = field(default_factory=dict)
    mtp_results: Dict[str, Any] = field(default_factory=dict)
    dsa_results: Dict[str, Any] = field(default_factory=dict)
    residual_stream: Dict[str, Any] = field(default_factory=dict)
    logit_lens: Dict[str, Any] = field(default_factory=dict)
    config: Optional[InterpretabilityConfig] = None
    prompts: List[str] = field(default_factory=list)
    model_config: Optional[Dict[str, Any]] = None

    def save(self, output_dir: str) -> None:
        """Save report: metadata.json + per-component .pt tensor files."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "metadata.json", "w") as f:
            json.dump({"prompts": self.prompts, "model_config": self.model_config},
                      f, indent=2, default=str)
        for name, results in [("mla", self.mla_results), ("moe", self.moe_results),
                              ("mtp", self.mtp_results), ("dsa", self.dsa_results),
                              ("residual", self.residual_stream), ("logit_lens", self.logit_lens)]:
            comp_dir = out / name
            comp_dir.mkdir(exist_ok=True)
            for key, value in results.items():
                if isinstance(value, Tensor):
                    torch.save(value, comp_dir / f"{key}.pt")
                else:
                    with open(comp_dir / f"{key}.json", "w") as f:
                        json.dump(value, f, indent=2, default=str)


class NanoSeekInterpreter:
    """Main interpretability interface for NanoSeek models.
    
    Coordinates hook registration, activation capture, and component-specific
    analysis. Understands NanoSeek's module hierarchy and knows where to attach
    hooks for each component (MLA, MoE, MTP, DSA, residual stream).
    
    NOT thread-safe — hook registration modifies the model's forward pass.
    """

    def __init__(self, model: nn.Module, tokenizer: Any,
                 config: Optional[InterpretabilityConfig] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or InterpretabilityConfig()
        self.store = ActivationStore(to_cpu=self.config.capture_to_cpu)
        self._hook_handles: List[torch.utils.hooks.RemovableHook] = []
        self._model_config = self._extract_model_config()
        num_layers = len(model.layers) if hasattr(model, "layers") else 0
        self._target_layers = self.config.get_target_layers(num_layers)
        cfg = getattr(model, "config", None)
        self._num_heads = cfg.num_heads if cfg else 16

    def _extract_model_config(self) -> Dict[str, Any]:
        cfg = getattr(self.model, "config", None)
        if cfg is None:
            return {}
        mla = getattr(cfg, "mla", None)
        moe = getattr(cfg, "moe", None)
        mtp = getattr(cfg, "mtp", None)
        return {
            "hidden_size": getattr(cfg, "hidden_size", None),
            "num_layers": getattr(cfg, "num_layers", None),
            "num_heads": getattr(cfg, "num_heads", None),
            "vocab_size": getattr(cfg, "vocab_size", None),
            "kv_lora_rank": getattr(mla, "kv_lora_rank", None) if mla else None,
            "n_routed_experts": getattr(moe, "n_routed_experts", None) if moe else None,
            "num_experts_per_tok": getattr(moe, "num_experts_per_tok", None) if moe else None,
            "num_mtp_modules": getattr(mtp, "num_mtp_modules", None) if mtp else None,
        }

    # ─── Hook Registration ────────────────────────────────────────

    def register_hooks(self, target_layers: Optional[List[int]] = None,
                       target_components: Optional[List[str]] = None) -> int:
        """Register forward hooks on model components for activation capture.
        
        Args:
            target_layers: Layer indices to instrument. None = use config default.
            target_components: Component names to hook. Supported values:
                ["mla", "moe", "mtp", "dsa", "residual", "embeddings", "logits"]
                None = hook all components enabled in config.
        
        Returns:
            Number of hooks registered.
        
        Hook attachment points and captured tensor shapes (B=batch, S=seq):
        ┌──────────────┬─────────────────────────┬────────────────────────┐
        │ Component    │ Hook Point              │ Captured Shape         │
        ├──────────────┼─────────────────────────┼────────────────────────┤
        │ embeddings   │ embed_tokens output      │ [B, S, 2048]          │
        │ mla.q_comp   │ wq_a output              │ [B, S, 430]           │
        │ mla.kv_comp  │ wkv_a output             │ [B, S, 175]           │
        │ mla.kv_decomp│ wkv_b output             │ [B, S, 16×256]        │
        │ mla.attn_out │ wo output                │ [B, S, 2048]          │
        │ moe.gate     │ Gate.forward output      │ [N,8] + [N,8]        │
        │ mtp.logits   │ MTPModule output         │ [B, S', 65536]        │
        │ mtp.hidden   │ MTPModule hidden         │ [B, S', H]           │
        │ dsa.scores   │ indexer output            │ [B, S, kv_len]       │
        │ residual.L_i │ layer_i output           │ [B, S, 2048]          │
        │ logits       │ lm_head output           │ [B, S, 65536]         │
        └──────────────┴─────────────────────────┴────────────────────────┘
        """
        layers = target_layers or self._target_layers
        components = target_components or self._resolve_components()
        count = 0

        if "embeddings" in components:
            count += self._register_hook(self.model.embed_tokens, "embeddings")

        for li in layers:
            if li >= len(self.model.layers):
                continue
            layer = self.model.layers[li]
            if "mla" in components:
                count += self._hook_mla(layer, li)
            if "moe" in components and layer.is_moe_layer:
                count += self._hook_moe(layer, li)
            if "residual" in components:
                count += self._hook_residual(layer, li)
            if "dsa" in components and layer.use_sparse_attention:
                count += self._hook_dsa(layer, li)

        if "mtp" in components and hasattr(self.model, "mtp") and self.model.mtp is not None:
            count += self._hook_mtp()
        if "logits" in components:
            count += self._register_hook(self.model.lm_head, "logits")
        return count

    def _resolve_components(self) -> List[str]:
        c = ["embeddings", "residual", "logits"]
        if self.config.analyze_mla: c.append("mla")
        if self.config.analyze_moe: c.append("moe")
        if self.config.analyze_mtp: c.append("mtp")
        if self.config.analyze_dsa: c.append("dsa")
        return c

    def _register_hook(self, module: nn.Module, key: str,
                       extract_fn: Optional[Callable] = None) -> int:
        def hook_fn(mod, inp, out):
            if extract_fn is not None:
                tensor = extract_fn(mod, inp, out)
            elif isinstance(out, Tensor):
                tensor = out
            elif isinstance(out, tuple) and isinstance(out[0], Tensor):
                tensor = out[0]
            else:
                return
            self.store.capture(key, tensor)
        handle = module.register_forward_hook(hook_fn)
        self._hook_handles.append(handle)
        return 1

    def _hook_mla(self, layer: nn.Module, li: int) -> int:
        """Hook MLA: q_compressed [B,S,430], kv_compressed [B,S,175],
        kv_decompressed [B,S,4096], attn_output [B,S,2048]."""
        attn = layer.self_attn
        mla = attn.mla if hasattr(attn, "mla") else attn
        count = self._register_hook(mla.wq_a, f"mla.q_compressed.layer_{li}")
        count += self._register_hook(mla.wkv_a, f"mla.kv_compressed.layer_{li}")
        count += self._register_hook(mla.wkv_b, f"mla.kv_decompressed.layer_{li}")
        count += self._register_hook(mla.wo, f"mla.attn_output.layer_{li}")
        return count

    def _hook_moe(self, layer: nn.Module, li: int) -> int:
        """Hook MoE gate (weights+indices) and combined output."""
        moe = layer.ffn
        k = self._model_config.get("num_experts_per_tok", 8)
        def extract_gate(mod, inp, out):
            weights, indices = out
            return torch.cat([weights.detach().float(), indices.detach().float()], dim=-1)
        count = self._register_hook(moe.gate, f"moe.gate_output.layer_{li}",
                                    extract_fn=extract_gate)
        def extract_moe_out(mod, inp, out):
            return out[0]
        count += self._register_hook(moe, f"moe.output.layer_{li}",
                                     extract_fn=extract_moe_out)
        return count

    def _hook_mtp(self) -> int:
        """Hook MTP logits [B,S',V] and hidden states [B,S',H]."""
        count = 0
        for i, module in enumerate(self.model.mtp.mtp_modules):
            def make_extract(idx):
                def fn(mod, inp, out): return out[0]  # logits
                return fn
            count += self._register_hook(module, f"mtp.logits.module_{i}",
                                         extract_fn=make_extract(i))
            count += self._register_hook(module.output_norm, f"mtp.hidden.module_{i}")
        return count

    def _hook_dsa(self, layer: nn.Module, li: int) -> int:
        """Hook DSA indexer scores [B,S,kv_len]."""
        return self._register_hook(layer.self_attn.indexer,
                                   f"dsa.indexer_scores.layer_{li}")

    def _hook_residual(self, layer: nn.Module, li: int) -> int:
        """Hook layer output (post residual) [B,S,2048]."""
        def extract(mod, inp, out): return out[0]
        return self._register_hook(layer, f"residual.layer_{li}", extract_fn=extract)

    def remove_hooks(self) -> int:
        count = len(self._hook_handles)
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        return count

    def cleanup(self) -> None:
        self.remove_hooks()
        self.store.clear()

    # ─── Forward Pass ─────────────────────────────────────────────

    def _tokenize(self, text: str) -> Tensor:
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for text input.")
        tokens = self.tokenizer.encode(text)
        t = torch.tensor([tokens], dtype=torch.long) if isinstance(tokens, list) else tokens
        return t.unsqueeze(0) if t.dim() == 1 else t

    @torch.inference_mode()
    def _forward_with_hooks(self, input_ids: Tensor) -> Dict[str, Tensor]:
        self.model.eval()
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        if input_ids.shape[1] > self.config.max_capture_tokens:
            input_ids = input_ids[:, :self.config.max_capture_tokens]
        return self.model(input_ids, output_hidden_states=True)

    # ─── Component Analysis Methods ──────────────────────────────

    def extract_mla_latent_space(self, input_ids: Tensor) -> Dict[str, Any]:
        """Extract and analyze MLA's compressed KV representations.

        Captures representations at three key pipeline points:
        1. Pre-compression:   hidden_states [B, S, 2048]
        2. Post-compression:  wkv_a output  [B, S, 175] (the bottleneck)
        3. Post-decompression: wkv_b output [B, S, 16 × 256]

        Analysis:
        - Singular value spectrum of compressed KV → effective rank
        - Per-head contribution norms after decompression → head redundancy
        - Q compressed space for query representation quality

        Returns dict with kv_compressed, kv_decompressed, q_compressed,
        singular_values, effective_rank, per_head_norms — keyed by layer_idx.
        """
        self.store.clear()
        self.register_hooks(target_components=["mla", "embeddings"])
        try:
            self._forward_with_hooks(input_ids)
        finally:
            self.remove_hooks()

        results = {k: {} for k in ["kv_compressed", "kv_decompressed", "q_compressed",
                                    "singular_values", "effective_rank", "per_head_norms"]}
        kv_lora_rank = self._model_config.get("kv_lora_rank", 143)

        for li in self._target_layers:
            kv_comp = self.store.get_single(f"mla.kv_compressed.layer_{li}")
            if kv_comp is not None:
                results["kv_compressed"][li] = kv_comp
                kv_content = kv_comp[:, :, :kv_lora_rank].reshape(-1, kv_lora_rank).float()
                if kv_content.shape[0] > 0:
                    try:
                        _, sv, _ = torch.svd_lowrank(kv_content, q=min(64, kv_lora_rank))
                        results["singular_values"][li] = sv
                        results["effective_rank"][li] = (sv > sv[0] * 0.01).sum().item()
                    except RuntimeError:
                        pass

            q_comp = self.store.get_single(f"mla.q_compressed.layer_{li}")
            if q_comp is not None:
                results["q_compressed"][li] = q_comp

            kv_decomp = self.store.get_single(f"mla.kv_decompressed.layer_{li}")
            if kv_decomp is not None:
                results["kv_decompressed"][li] = kv_decomp
                per_head_dim = kv_decomp.shape[-1] // self._num_heads
                kv_ph = kv_decomp.view(*kv_decomp.shape[:-1], self._num_heads, per_head_dim)
                results["per_head_norms"][li] = kv_ph.float().norm(dim=-1).mean(dim=(0, 1))
        return results

    def capture_expert_routing(self, input_ids: Tensor) -> Dict[str, Any]:
        """Capture MoE routing: expert indices/weights, load distribution,
        routing entropy, co-activation matrix, top-1 expert per token."""
        self.store.clear()
        self.register_hooks(target_components=["moe"])
        try:
            self._forward_with_hooks(input_ids)
        finally:
            self.remove_hooks()

        results = {k: {} for k in ["expert_indices", "expert_weights", "expert_load",
                                    "expert_entropy", "co_activation_matrix", "top1_expert_per_token"]}
        n_exp = self._model_config.get("n_routed_experts", 64)
        k = self._model_config.get("num_experts_per_tok", 8)

        for li in self._target_layers:
            gate_out = self.store.get_single(f"moe.gate_output.layer_{li}")
            if gate_out is None:
                continue
            weights, indices = gate_out[:, :k], gate_out[:, k:].long()
            results["expert_indices"][li] = indices
            results["expert_weights"][li] = weights

            load = torch.zeros(n_exp)
            for eid in range(n_exp):
                load[eid] = (indices == eid).sum().float()
            results["expert_load"][li] = load

            lp = load / load.sum().clamp(min=1e-8)
            results["expert_entropy"][li] = -(lp * (lp + 1e-10).log()).sum().item()

            co = torch.zeros(n_exp, n_exp)
            for ti in range(indices.shape[0]):
                active = indices[ti].tolist()
                for a in range(len(active)):
                    for b in range(a + 1, len(active)):
                        co[active[a], active[b]] += 1
                        co[active[b], active[a]] += 1
            results["co_activation_matrix"][li] = co

            top1_idx = weights.argmax(dim=-1)
            results["top1_expert_per_token"][li] = indices.gather(1, top1_idx.unsqueeze(-1)).squeeze(-1)
        return results

    def analyze_mtp_predictions(self, input_ids: Tensor) -> Dict[str, Any]:
        """Analyze MTP: prediction agreement with main model, entropy,
        hidden state norms, CKA similarity to main model representations."""
        self.store.clear()
        self.register_hooks(target_components=["mtp", "residual", "logits"])
        try:
            self._forward_with_hooks(input_ids)
        finally:
            self.remove_hooks()

        results = {k: {} for k in ["mtp_logits", "mtp_top_tokens", "agreement_rate",
                                    "mtp_entropy", "mtp_hidden_norms", "cka_similarity"]}
        main_logits = self.store.get_single("logits")
        main_top = main_logits.argmax(dim=-1) if main_logits is not None else None
        results["main_top_tokens"] = main_top

        if self.model.mtp is not None:
            for i in range(len(self.model.mtp.mtp_modules)):
                mtp_logits = self.store.get_single(f"mtp.logits.module_{i}")
                if mtp_logits is None:
                    continue
                results["mtp_logits"][i] = mtp_logits
                mtp_top = mtp_logits.argmax(dim=-1)
                results["mtp_top_tokens"][i] = mtp_top

                probs = F.softmax(mtp_logits.float(), dim=-1)
                results["mtp_entropy"][i] = -(probs * (probs + 1e-10).log()).sum(dim=-1)

                if main_top is not None:
                    offset = i + 1
                    main_shifted = main_top[:, offset:offset + mtp_top.shape[1]]
                    ml = min(main_shifted.shape[1], mtp_top.shape[1])
                    if ml > 0:
                        results["agreement_rate"][i] = (
                            main_shifted[:, :ml] == mtp_top[:, :ml]).float().mean().item()

                mtp_h = self.store.get_single(f"mtp.hidden.module_{i}")
                if mtp_h is not None:
                    results["mtp_hidden_norms"][i] = mtp_h.float().norm(dim=-1)
                    last_l = max(self._target_layers) if self._target_layers else 0
                    main_h = self.store.get_single(f"residual.layer_{last_l}")
                    if main_h is not None:
                        ms = min(mtp_h.shape[1], main_h.shape[1])
                        if ms > 0:
                            results["cka_similarity"][i] = self._compute_linear_cka(
                                mtp_h[:, :ms].reshape(-1, mtp_h.shape[-1]),
                                main_h[:, :ms].reshape(-1, main_h.shape[-1]))
        return results

    def analyze_sparse_attention(self, input_ids: Tensor) -> Dict[str, Any]:
        """Analyze DSA: indexer scores, score statistics, position selection
        frequency, selection entropy."""
        self.store.clear()
        self.register_hooks(target_components=["dsa"])
        try:
            self._forward_with_hooks(input_ids)
        finally:
            self.remove_hooks()

        results = {k: {} for k in ["indexer_scores", "score_statistics",
                                    "position_selection_frequency", "selection_entropy"]}
        for li in self._target_layers:
            scores = self.store.get_single(f"dsa.indexer_scores.layer_{li}")
            if scores is None:
                continue
            results["indexer_scores"][li] = scores
            sf = scores.float()
            results["score_statistics"][li] = {
                "mean": sf.mean().item(), "std": sf.std().item(),
                "max": sf.max().item(), "min": sf.min().item(),
            }
            probs = F.softmax(sf, dim=-1)
            results["position_selection_frequency"][li] = probs.mean(dim=(0, 1))
            log_p = F.log_softmax(sf, dim=-1)
            results["selection_entropy"][li] = -(probs * log_p).sum(dim=-1)
        return results

    def analyze_residual_stream(self, input_ids: Tensor) -> Dict[str, Any]:
        """Analyze residual stream: norms, inter-layer cosine similarity,
        logit lens (project each layer to vocab space)."""
        self.store.clear()
        self.register_hooks(target_components=["embeddings", "residual", "logits"])
        try:
            self._forward_with_hooks(input_ids)
        finally:
            self.remove_hooks()

        results = {"hidden_states": {}, "norms": {}, "layer_similarity": [],
                   "logit_lens_top_tokens": {}, "logit_lens_probs": {}}
        emb = self.store.get_single("embeddings")
        prev = emb
        if emb is not None:
            results["norms"][-1] = emb.float().norm(dim=-1)

        for li in sorted(self._target_layers):
            h = self.store.get_single(f"residual.layer_{li}")
            if h is None:
                continue
            results["hidden_states"][li] = h
            results["norms"][li] = h.float().norm(dim=-1)
            if prev is not None and h.shape == prev.shape:
                cs = F.cosine_similarity(h.float().reshape(-1, h.shape[-1]),
                                         prev.float().reshape(-1, prev.shape[-1]),
                                         dim=-1).mean().item()
                results["layer_similarity"].append(cs)
            # Logit lens
            if hasattr(self.model, "norm") and hasattr(self.model, "lm_head"):
                dev = next(self.model.parameters()).device
                normed = self.model.norm(h.to(dev))
                logits = self.model.lm_head(normed).detach().cpu()
                results["logit_lens_top_tokens"][li] = logits.argmax(dim=-1)
                results["logit_lens_probs"][li] = F.softmax(logits.float(), dim=-1).max(dim=-1).values
            prev = h
        return results

    def run_full_analysis(self, prompts: List[str]) -> InterpretabilityReport:
        """Run the complete interpretability suite on a list of prompts.

        This is the main entry point for comprehensive analysis. Each prompt is
        processed independently (no batching across prompts to avoid padding
        artifacts in the analysis).

        Memory usage estimate (per prompt, default config, 1024 tokens):
            MLA:      ~200 MB (compressed + decompressed KV × 16 layers)
            MoE:      ~50 MB  (gate outputs × MoE layers)
            MTP:      ~30 MB  (logits are large but only 1 module)
            DSA:      ~100 MB (indexer scores × DSA layers)
            Residual: ~500 MB (full hidden states × 16 layers)
            Total:    ~880 MB per prompt
        """
        report = InterpretabilityReport(config=self.config, prompts=prompts,
                                        model_config=self._model_config)
        for pi, prompt in enumerate(prompts):
            input_ids = self._tokenize(prompt)
            pfx = f"prompt_{pi}"
            if self.config.analyze_mla:
                for k, v in self.extract_mla_latent_space(input_ids).items():
                    report.mla_results[f"{pfx}.{k}"] = v
            if self.config.analyze_moe:
                for k, v in self.capture_expert_routing(input_ids).items():
                    report.moe_results[f"{pfx}.{k}"] = v
            if self.config.analyze_mtp:
                for k, v in self.analyze_mtp_predictions(input_ids).items():
                    report.mtp_results[f"{pfx}.{k}"] = v
            if self.config.analyze_dsa:
                for k, v in self.analyze_sparse_attention(input_ids).items():
                    report.dsa_results[f"{pfx}.{k}"] = v
            if self.config.analyze_residual or self.config.analyze_logit_lens:
                for k, v in self.analyze_residual_stream(input_ids).items():
                    target = report.logit_lens if k.startswith("logit_lens") else report.residual_stream
                    target[f"{pfx}.{k}"] = v
        if self.config.output_dir:
            report.save(self.config.output_dir)
        return report

    @staticmethod
    def _compute_linear_cka(x: Tensor, y: Tensor) -> float:
        """Linear CKA (Kornblith et al. 2019) — representational similarity
        invariant to orthogonal transforms. Returns float in [0, 1]."""
        x, y = x.float() - x.float().mean(0, keepdim=True), y.float() - y.float().mean(0, keepdim=True)
        hsic_xy = (x @ x.T * (y @ y.T)).sum()
        denom = ((x @ x.T).pow(2).sum() * (y @ y.T).pow(2).sum()).sqrt()
        return (hsic_xy / denom).item() if denom > 1e-10 else 0.0
```

### 3b. Example Usage

```python
import torch
from model.model import create_nanoseek
from model.config import NanoSeekConfig, MLAConfig, MoEConfig, MTPConfig

# Small model for fast analysis
config = NanoSeekConfig(
    hidden_size=256, num_layers=4, num_heads=4, vocab_size=1024,
    mla=MLAConfig(q_lora_rank=54, kv_lora_rank=18, qk_nope_head_dim=32,
                  qk_rope_head_dim=16, v_head_dim=32),
    moe=MoEConfig(n_routed_experts=16, num_experts_per_tok=4, n_shared_experts=1),
    mtp=MTPConfig(num_mtp_modules=1, mtp_num_heads=4),
)
model = create_nanoseek(config)

interp_config = InterpretabilityConfig(
    target_layers=[0, 1, 2, 3], analyze_dsa=False,
    output_dir="artifacts/demo_analysis",
)
interp = NanoSeekInterpreter(model, tokenizer=None, config=interp_config)

input_ids = torch.randint(0, config.vocab_size, (1, 64))

mla_results = interp.extract_mla_latent_space(input_ids)
print(f"MLA effective rank by layer: {mla_results['effective_rank']}")

moe_results = interp.capture_expert_routing(input_ids)
for li, entropy in moe_results["expert_entropy"].items():
    print(f"MoE routing entropy layer {li}: {entropy:.3f}")

mtp_results = interp.analyze_mtp_predictions(input_ids)
for mi, rate in mtp_results["agreement_rate"].items():
    print(f"MTP module {mi} agreement: {rate:.1%}")

interp.cleanup()
```

---

## 4. Component-Specific Interpretability Strategies

### 4a. MLA — Multi-Head Latent Attention

**What to measure and why**:

| Metric | Purpose | Method |
|--------|---------|--------|
| KV compressed spectrum (SVD) | How much of the 143-dim space is used? | `torch.linalg.svdvals` on `wkv_a` output |
| Per-head contribution after decompression | Are heads redundant? | L2 norm of per-head slices of `wkv_b` output |
| Compression reconstruction error | How lossy is the bottleneck? | Compare `wkv_b(wkv_a(x))` to ideal `W_full @ x` |
| Attention pattern comparison | How do MLA patterns differ from MHA? | Visualize attention weights, compare to baseline |

**Published methods**: Probing classifiers (Belinkov & Glass 2019) on compressed KV (dim=143) to classify token properties (POS tags, named entities, dependency relations). If probes achieve high accuracy, the information survives compression. Singular value analysis: if top-k singular values capture >95% of variance, the effective dimensionality is k, not 143 — the model uses fewer independent features than the bottleneck allows.

**Expected findings**: Early layers have lower effective rank (broad, shared features). Later layers have higher effective rank (specialized, token-specific). Position info concentrates in RoPE (32 dims); semantic info in compressed KV (143 dims). Some heads will have near-zero contribution norms post-decompression, indicating redundancy that the compression has exposed.

```
                    MLA Probe Points
                    ════════════════

hidden [B,S,2048]
    ├── wq_a ──→ ⊕ PROBE A: q_compressed [B,S,430]
    │             └── wq_b ──→ Q [B,S,16,160]
    │
    └── wkv_a ─→ ⊕ PROBE B: kv_compressed [B,S,175]
                  ├── kv_content [B,S,143]  ← SVD target
                  └── k_pe [B,S,32]         ← RoPE component
                      │
                   kv_norm → wkv_b ──→ ⊕ PROBE C: kv_decomp [B,S,16,256]
                                        ├── k_nope [B,S,16,128]
                                        └── v [B,S,16,128]
                                            │
                                         attention → wo → ⊕ PROBE D [B,S,2048]
```

### 4b. MoE — Mixture of Experts

**What to measure and why**:

| Metric | Purpose | Method |
|--------|---------|--------|
| Expert load distribution | Are experts balanced? | `torch.bincount` on gate indices |
| Expert specialization by input type | What triggers each expert? | Correlation: activation vs token categories |
| Routing entropy | How confident are routing decisions? | Shannon entropy of gate scores |
| Expert co-activation | Which experts work together? | Co-occurrence matrix from indices |
| Expert ablation | Is each expert necessary? | Force weight to 0, measure loss delta |
| Gate decision geometry | How does gate partition input space? | PCA/t-SNE of gate weight matrix |

**Published methods**: Expert specialization profiling (Fedus et al. 2022 — Switch Transformers) — tag input tokens by category and measure per-expert activation frequency. SAEs on gate inputs (Bricken et al. 2023) — decompose hidden states before the gate projection into monosemantic features, revealing *which features* drive routing rather than just *which experts* are selected. Expert ablation (causal intervention) — zero out individual expert contributions and measure downstream loss; distinguishes "specialist" experts (high loss increase) from "generalist" experts (minimal impact).

**Expected findings**: Several experts specialize in high-frequency patterns (function words, punctuation) and process the most tokens. A small number handle rare patterns (technical vocabulary, numbers). Co-activation reveals "expert teams" handling complex tokens collaboratively. The bias term shifts load toward uniformity, making the distribution more balanced than raw sigmoid scores would produce.

```
                    MoE Probe Points
                    ════════════════

hidden [N, 2048]
    │
    ├── gate.weight ──→ sigmoid+bias ──→ ⊕ PROBE G: scores [N,64]
    │                                     │
    │                                  topk(8) ──→ ⊕ PROBE H: indices [N,8], weights [N,8]
    │                                     │
    │                              token_centric_dispatch
    │                              ┌──┴──┬──┴── ... ──┐
    │                           exp_0  exp_1        exp_63
    │                              └──┬──┴──┬── ... ──┘
    │                            weighted_sum ──→ ⊕ PROBE J: routed [N,2048]
    │
    └── shared_experts ──→ ⊕ PROBE F: shared [N,2048]
                            │
                          + routed ──→ output [B,S,2048]
```

### 4c. MTP — Multi-Token Prediction

**What to measure and why**:

| Metric | Purpose | Method |
|--------|---------|--------|
| Prediction agreement (MTP vs main) | Does MTP learn the same distribution? | Compare argmax tokens |
| Prediction entropy | How confident is MTP? | Entropy of softmax distribution |
| CKA similarity | Are MTP representations similar to main? | Linear CKA between hidden states |
| Cross-attention patterns | What does MTP attend to? | Capture cross-attention weights |
| Speculative decode acceptance rate | Would MTP drafts be accepted? | Simulate verify step |

**Published methods**: CKA (Kornblith et al. 2019) measures representational similarity between MTP hidden states and main model hidden states. High CKA (>0.8) suggests MTP learns similar representations; low CKA suggests distinct planning features emerge. Logit lens applied to MTP: project MTP hidden states through the shared `lm_head` at intermediate processing stages (after cross-attention, after self-attention, after FFN) to see how the MTP prediction evolves.

**Expected findings**: Higher agreement for predictable next-next tokens (function words, common phrases), lower for content words. MTP hidden states will have lower CKA with early main model layers and higher with later layers. MTP entropy will be higher than main model entropy on average (less confident about t+2 than t+1). Cross-attention focuses on the most recent tokens (local context is most informative).

```
                    MTP Probe Points
                    ════════════════

main_hidden [B,S,2048] ─────────────────────────────┐
    │                                                 │
 hidden_norm ── concat ── embed_norm(target_tokens)  │
    ⊕ PROBE K: [B,S,4096]                           │
    │                                                 │
 concat_proj [4096→2048]                             │
    │                                                 │
 MTPBlock: cross_attn(q=hidden, kv=main) ←───────────┘
    ⊕ PROBE L: cross-attn output
    │
 self_attn (causal) → ⊕ PROBE M
    │
 SwiGLU FFN → ⊕ PROBE N
    │
 output_norm → ⊕ PROBE O: final hidden [B,S',H]
    │
 lm_head (shared) → ⊕ PROBE P: logits [B,S',65536]
```

### 4d. DSA — DeepSeek Sparse Attention

**What to measure and why**:

| Metric | Purpose | Method |
|--------|---------|--------|
| Indexer selection frequency by position | Does DSA prefer recent/early tokens? | Histogram of selected indices |
| Selection overlap across layers | Do layers attend to the same tokens? | Jaccard similarity of selected sets |
| Score distribution shape | How selective is the indexer? | Score histogram, entropy |
| Dense vs sparse output difference | What changes with sparsity? | L2 distance: dense vs sparse output |
| Indexer head weights | Which indexer heads matter most? | Inspect `head_weights` parameter |

**Published methods**: Attention pattern categorization (Clark et al. 2019, "What Does BERT Look At?") — classify DSA selection patterns into types: local (recent tokens), global (beginning-of-sequence), syntactic (matching brackets/keywords), semantic (topically related tokens). Counterfactual analysis: force the indexer to select different token subsets and measure impact on model output — reveals whether the indexer's selection is critical (large output change) or redundant (model can compute correctly from any subset).

**Expected findings**: The indexer will prefer recent tokens (recency bias is universal in attention). Early tokens (positions 0-5) will be selected disproportionately (the "attention sink" phenomenon, Xiao et al. 2023). Selection patterns will differ between layers: early layers select based on position, later layers on content. The `head_weights` parameter will be non-uniform, with 1-2 heads dominating the importance scoring.

```
                    DSA Probe Points
                    ════════════════

hidden [B,S,2048]
    │
    ├── wq_a → q_compressed [B,S,430]
    │                │
    │          indexer.q_proj → [B,S,4,64]
    │                │
    └── wkv_a → kv_compressed [B,S,143]
                     │
               indexer.k_proj → [B,kv,4,64]
                     │
              ⊕ PROBE Q: einsum(q,k) → scores [B,4,S,kv]
                     │
              relu → head_weight_sum → ⊕ PROBE R: [B,S,kv_len]
                     │
              topk → ⊕ PROBE S: selected indices [B,S,topk]
                     │
              sparse attention on selected subset
              → ⊕ PROBE T: sparse attn weights
              → wo → ⊕ PROBE U: output [B,S,2048]
```

---

## 5. Data Flow & Hook Architecture

### 5a. Complete Forward Pass — All Hook Attachment Points

```
input_ids [B, S]
     │
  embed_tokens ◄── HOOK "embeddings" → [B, S, 2048]
     │
═══ FOR layer_idx IN range(16): ════════════════════════════════
│                                                               │
│   input_layernorm                                             │
│        │                                                      │
│   self_attn (MLA or DSA)                                      │
│   ├── HOOK "mla.q_compressed.L_i"    → [B, S, 430]          │
│   ├── HOOK "mla.kv_compressed.L_i"   → [B, S, 175]          │
│   ├── HOOK "mla.kv_decompressed.L_i" → [B, S, 4096]         │
│   ├── HOOK "mla.attn_output.L_i"     → [B, S, 2048]         │
│   └── HOOK "dsa.indexer_scores.L_i"  → [B, S, kv_len] (DSA) │
│        │                                                      │
│   + residual → post_attn_layernorm                            │
│        │                                                      │
│   ffn (MoE or MLP)                                            │
│   ├── HOOK "moe.gate_output.L_i" → [N, 16] (8w + 8i)       │
│   └── HOOK "moe.output.L_i"     → [B, S, 2048]              │
│        │                                                      │
│   + residual ◄── HOOK "residual.layer_i" → [B, S, 2048]     │
│                                                               │
═══ END FOR ════════════════════════════════════════════════════
     │
  norm → lm_head ◄── HOOK "logits" → [B, S, 65536]
     │
  mtp (if enabled)
  ├── HOOK "mtp.logits.module_j"  → [B, S', 65536]
  └── HOOK "mtp.hidden.module_j"  → [B, S', 2048]
```

### 5b. Hook Data Flow — From Capture to Report

```
Model Forward Pass            ActivationStore               Analysis Methods
──────────────────            ────────────────               ────────────────

nn.Module.forward()           store.capture(key, tensor)
       │                            │
       ├── hook_fn fires ──────────►│ detach() → .cpu()
       │                            │ append to _store[key]
       │                            │
       └── forward complete         │
                                    │
                            store.get(key) ────────►  extract_mla_latent_space()
                            store.get(key) ────────►  capture_expert_routing()
                            store.get(key) ────────►  analyze_mtp_predictions()
                            store.get(key) ────────►  analyze_sparse_attention()
                            store.get(key) ────────►  analyze_residual_stream()
                                    │
                            store.clear() ◄────────  cleanup()

                                                          │
                                                          ▼
                                                InterpretabilityReport
                                                ├── mla_results
                                                ├── moe_results
                                                ├── mtp_results
                                                ├── dsa_results
                                                ├── residual_stream
                                                └── logit_lens
                                                          │
                                                   report.save()
                                                          ▼
                                                artifacts/
                                                ├── metadata.json
                                                ├── mla/*.pt
                                                ├── moe/*.pt
                                                └── ...
```

### 5c. Tensor Shape Reference

For default NanoSeek-1B: hidden=2048, heads=16, kv_lora_rank=143, rope=32, experts=64, k=8, vocab=65536.

```
┌───────────────────────────────┬──────────────────────────────────┐
│ Capture Point                 │ Shape                            │
├───────────────────────────────┼──────────────────────────────────┤
│ embeddings                    │ [B, S, 2048]                     │
│ mla.q_compressed.layer_i      │ [B, S, 430]                      │
│ mla.kv_compressed.layer_i     │ [B, S, 175] (143 + 32 RoPE)     │
│ mla.kv_decompressed.layer_i   │ [B, S, 4096] (16 × 256)         │
│ mla.attn_output.layer_i       │ [B, S, 2048]                     │
│ moe.gate_output.layer_i       │ [N, 16] (8 weights + 8 indices)  │
│ moe.output.layer_i            │ [B, S, 2048]                     │
│ mtp.logits.module_j           │ [B, S', 65536]                   │
│ mtp.hidden.module_j           │ [B, S', 2048]                    │
│ dsa.indexer_scores.layer_i    │ [B, S, kv_len]                   │
│ residual.layer_i              │ [B, S, 2048]                     │
│ logits                        │ [B, S, 65536]                    │
├───────────────────────────────┼──────────────────────────────────┤
│ DERIVED                       │                                  │
├───────────────────────────────┼──────────────────────────────────┤
│ singular_values[layer_i]      │ [min(B*S, 143)]                  │
│ per_head_norms[layer_i]       │ [16]                             │
│ expert_load[layer_i]          │ [64]                             │
│ co_activation_matrix[layer_i] │ [64, 64]                         │
│ layer_similarity              │ [num_layers - 1]                 │
│ logit_lens_top_tokens[layer_i]│ [B, S]                           │
│ mtp_entropy[module_j]         │ [B, S']                          │
│ selection_entropy[layer_i]    │ [B, S]                           │
└───────────────────────────────┴──────────────────────────────────┘
N = B × S.  S' = S - (module_idx + 1).  kv_len ≥ S.
```

---

## 6. File Placement

```
fms/interpretability/
├── __init__.py           # Package exports
├── framework.py          # Core: InterpretabilityConfig, ActivationStore,
│                         #   NanoSeekInterpreter, InterpretabilityReport (this doc)
├── hooks.py              # Low-level hook utilities: HookManager, BatchedHookCapture,
│                         #   ConditionalHook, GradientHook (backward pass capture)
├── mla_analysis.py       # MLA deep analysis (Doc 24): LatentSpaceProbe,
│                         #   CompressionAnalyzer, HeadRedundancyScorer
├── moe_analysis.py       # MoE expert analysis (Doc 24): ExpertSpecializationProfiler,
│                         #   ExpertAblation, GateGeometryAnalyzer, SAEonGateInputs
├── visualize.py          # Visualization toolkit (Doc 25): AttentionHeatmap,
│                         #   ExpertLoadChart, ResidualStreamPlot, LogitLensGrid
├── evaluate.py           # Evaluation metrics (Doc 26): ProbeAccuracy,
│                         #   CircuitFaithfulness, FeatureCompleteness
└── README.md             # Module docs and quick-start guide
```

### Dependency Flow

```
framework.py ──► hooks.py         (hook registration primitives)
     ├──────────► mla_analysis.py  (MLA methods extend framework)
     ├──────────► moe_analysis.py  (MoE methods extend framework)
     └──────────► visualize.py     (consumes InterpretabilityReport)
                      └──► evaluate.py  (metrics for interpretability quality)
```

### Integration

The framework imports from but does **not modify** existing model code. Uses PyTorch's `register_forward_hook` API — no changes to `model/model.py` required.

---

## 7. Gotchas

### 7a. Hook Memory Explosion with Full Activation Capture

Capturing all activations for 16 layers on 4096 tokens requires ~8GB CPU memory. The logit lens is the killer: projecting every layer to vocab space (65536 dims) produces ~16GB of tensors. **Mitigation**: The framework captures logit lens results as argmax token IDs + top-1 probability only, reducing 16GB → ~512KB. Use `target_layers` to limit scope. Budget: ~1GB per 1024 tokens per 4 analyzed layers.

### 7b. MLA Compressed Space Is NOT Attention Weights

A common mistake: visualize `wkv_a` output `[B, S, 175]` as an "attention pattern." This is wrong — it is a feature vector, not a weight matrix. It has shape `[S, 175]`, not `[S, S]`. To see actual attention weights, you must decompress via `wkv_b`, compute Q through the full path, apply RoPE, and compute `Q @ K^T * scale + softmax`. The framework's `capture_attention_weights` flag handles this correctly.

### 7c. MoE Expert Counts Change Meaning with Group Routing

NanoSeek uses `n_expert_groups=4` and `n_limited_groups=2`: the gate first selects top-2 groups (16 experts each), then top-8 within those groups. Expert indices are NOT uniformly distributed over [0, 63]. Co-activation patterns are biased by group structure. "Dead expert" analysis must account for group exclusion — an expert with zero load might be in a never-selected group, not a dead expert.

### 7d. Superposition Makes Neuron-Level Analysis Misleading

Individual neurons are polysemantic (Elhage et al. 2022). Analyzing `hidden_states[:, :, 42]` across inputs shows it activating for seemingly unrelated concepts because it encodes multiple features in superposition. **What to do instead**: Use SAEs to decompose into monosemantic features, or analyze at the level of entire expert outputs (MoE provides natural modularity). Never report "neuron X responds to concept Y" without controlling for superposition via interventional experiments.

### 7e. DSA Attention Patterns Depend on Training Phase

During Phase 1 (dense training, first 80% of tokens), the DSA indexer trains via auxiliary loss but does not control attention. During Phase 2 (sparse fine-tuning, last 20%), the indexer actively selects tokens. Phase 1 checkpoints have meaningful indexer scores but they don't reflect actual attention behavior. Only Phase 2+ checkpoints show the indexer's real impact. The `training_step` buffer on `DSASparseAttention` tracks phase.

### 7f. MTP Hidden States May Be Lower-Dimensional

When `mtp_hidden_size != hidden_size`, MTP operates in a different space. CKA comparison requires careful dimension handling. SAEs trained on main model hidden states cannot be applied to MTP hidden states. When `mtp_hidden_size != hidden_size`, MTP creates its own `lm_head` and `embed_tokens`, decoupling the prediction head's weight space from the main model.

### 7g. Gradient Checkpointing Interferes with Hook Capture

With `gradient_checkpointing=True`, PyTorch re-runs forward passes during backward, causing hooks to fire twice (duplicate captures). **Mitigation**: Run analysis in `torch.inference_mode()` with `model.eval()`, which disables gradient checkpointing. Never run interpretability analysis with `model.train()` unless you specifically want training-time behavior (dropout effects).

### 7h. Hook Ordering Not Guaranteed Across PyTorch Versions

PyTorch's hook execution order can vary across versions (especially with `torch.compile`). The framework avoids this by running component-specific analyses in separate forward passes (`extract_mla_latent_space` is a separate call from `capture_expert_routing`). The `run_full_analysis` method orchestrates these sequentially.

---

*"The models we cannot interpret are the models we cannot trust. And the models we cannot trust are the models that will eventually surprise us — in the worst possible way, at the worst possible time. Interpretability is not an academic luxury; it is the engineering discipline that separates 'a model that works' from 'a model we understand.' Only the latter deserves to be deployed."*

— Principal Engineer's Note, Interpretability Division, 2026
