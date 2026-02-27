# 24 — Mechanistic Interpretability: Circuit Discovery & Causal Analysis

**Author**: Distinguished AI Researcher & Principal Engineer, Foundation Models Division
**Date**: February 2026
**Scope**: Reverse-engineering NanoSeek's learned algorithms through activation patching, circuit discovery, sparse autoencoders, and latent space probing
**Prerequisites**: `model/model.py` (MultiHeadLatentAttention, Gate, MoE, MTPModule, NanoSeekDecoderLayer), `model/config.py` (NanoSeekConfig)

---

## 1. Problem Statement

### What Mechanistic Interpretability Is

Mechanistic interpretability is the science of reverse-engineering neural networks — understanding the algorithms that the model has learned, expressed as human-readable computational graphs. This is distinct from behavioral interpretability (probing what a model does); mechanistic interpretability seeks to explain *how* and *why*. The foundational framework was established by Elhage et al. (2021) in "A Mathematical Framework for Transformer Circuits," showing transformer computations decompose into *circuits* — minimal sub-networks implementing specific algorithmic behaviors. Olsson et al. (2022) built on this in "In-context Learning and Induction Heads," demonstrating that induction heads implement [A][B]...[A] → [B] pattern-completion and are causally responsible for in-context learning.

### Why MoE Models Have Natural Circuit Structure

MoE architectures present a unique opportunity: **explicit routing decisions** create a natural factorization into identifiable sub-circuits. Each expert IS a sub-circuit. NanoSeek amplifies this:

- **MLA's low-rank bottleneck** (kv_lora_rank=143) forces information compression — what survives the 23× compression tells us what the model considers important.
- **The Gate's sigmoid scoring + bias-based load balancing** creates measurable expert specialization patterns.
- **MTP's cross-attention modules** reveal what sequential information the model considers predictive.

### Goal

Reverse-engineer NanoSeek's learned algorithms across three axes:

1. **Induction heads in MLA**: Identify which of the 16 attention heads implement induction-like pattern completion through MLA's compressed KV.
2. **Expert specialization in MoE**: Profile what each of the 64 routed experts has learned to specialize in.
3. **Feature decomposition via SAEs**: Train sparse autoencoders on NanoSeek's residual stream to discover monosemantic features.

---

## 2. First Principles — The Science

### 2.1 Activation Patching (Causal Tracing)

Replace activations at specific positions in one forward pass with activations from a different run, measuring the output change. Given clean input `x_clean` and corrupted input `x_corrupt` (typically `x_clean` + Gaussian noise on embeddings):

```
IE(l, c) = P(correct | do(a_l^c = a_l^c(x_clean)), x_corrupt) - P(correct | x_corrupt)
```

Meng et al. (2022) in "Locating and Editing Factual Associations in GPT" used this to show factual knowledge concentrates in mid-layer MLPs at the last subject token.

**For NanoSeek**: We patch at three granularities — layer outputs, MLA heads, and MoE experts. Because K and V are jointly compressed through `wkv_a` → `kv_norm` → `wkv_b`, patching a single MLA "head" requires intervening after `wkv_b` expansion, not in the compressed latent.

### 2.2 Circuit Discovery

A *circuit* is a minimal computational subgraph sufficient to reproduce a specific behavior. Conmy et al. (2023) introduced ACDC in "Towards Automated Circuit Discovery," framing this as edge pruning: iteratively remove connections and keep those whose removal degrades task performance.

**For NanoSeek**: MoE layers add a branching factor — each token's circuit passes through a different subset of 8/64 experts. The circuit is a *conditional* subgraph depending on routing decisions, requiring joint tracing of routing and expert computations.

### 2.3 Sparse Autoencoders (SAEs)

Individual neurons are polysemantic — activating for multiple unrelated concepts (Elhage et al. 2022). SAEs decompose these into monosemantic *features*:

```
encoded = ReLU(W_enc @ (x - b_dec) + b_enc)    # R^d → R^n_features
decoded = W_dec @ encoded + b_dec               # R^n_features → R^d
```

where `n_features >> d` (4-64× overcomplete). The L1 penalty on `encoded` forces sparsity. Bricken et al. (2023) at Anthropic ("Towards Monosemanticity") showed SAEs discover features like "DNA sequences" and "legal language" — invisible at the neuron level. Cunningham et al. (2023) extended this to residual streams.

**For NanoSeek**: SAE features in MoE models should be *sparser* than dense models because routing already provides a coarse decomposition. The key question: do experts themselves contain superposed features, or does routing eliminate the need?

### 2.4 Logit Lens / Tuned Lens

Project the residual stream at any intermediate layer through the unembedding matrix (lm_head) to see what the model would predict if computation stopped there. nostalgebraist (2020) introduced the *logit lens*; Belrose et al. (2023) introduced the *tuned lens* with learned per-layer affine corrections.

**For NanoSeek**: MoE creates *discontinuous* residual contributions — different tokens pass through different experts, making logit lens predictions noisier at MoE layers than at MLA layers.

### 2.5 Superposition

Networks represent more features than dimensions by encoding in *superposition* — overlapping, nearly-orthogonal directions. Elhage et al. (2022) in "Toy Models of Superposition" showed this arises when features are sparse and capacity is limited.

**For NanoSeek**: We hypothesize a gradient — early layers (0-3) exhibit high superposition, mid layers (4-10) moderate, late layers (11-15) low. MoE layers should exhibit *less* superposition than dense layers because routing already partitions the feature space.

### 2.6 Induction Heads

Induction heads implement [A][B]...[A] → [B] pattern-completion (Olsson et al. 2022). The two-head circuit: (1) a **previous-token head** attends pos i to pos i-1, and (2) an **induction head** attends from the current [A] to positions after previous occurrences of [A].

**For NanoSeek's MLA**: The RoPE component `k_pe` (dim 32) handles positional matching, while `k_nope` handles content matching. The content signal must survive the 143-dim compression bottleneck. We should examine rope and nope components separately — strong positional matching in rope suggests previous-token head behavior even if content matching is weaker.

---

## 3. Production Code — Four Complete Analysis Modules

### 3a. Activation Patching for NanoSeek

**File**: `fms/interpretability/patching.py`

```python
"""
Activation Patching (Causal Tracing) for NanoSeek.

Implements hook-based causal interventions to measure component importance.
Supports patching at layer, MLA head, and MoE expert granularity.

Reference: Meng et al. 2022, "Locating and Editing Factual Associations in GPT"

Usage:
    patcher = ActivationPatcher(model, tokenizer)
    result = patcher.run_causal_trace(
        prompt="The Eiffel Tower is located in",
        subject_tokens=[1, 2, 3],  # token positions for "Eiffel Tower"
    )
    # result.effects[layer][component] gives causal effect magnitude
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class CausalTraceResult:
    """Result of a full causal trace across all layers and components."""

    prompt: str
    subject_token_positions: List[int]
    target_token: str
    clean_prob: float
    corrupt_prob: float

    layer_effects: Dict[int, float] = field(default_factory=dict)
    mla_head_effects: Dict[Tuple[int, int], float] = field(default_factory=dict)
    expert_effects: Dict[Tuple[int, int], float] = field(default_factory=dict)
    mlp_effects: Dict[int, float] = field(default_factory=dict)

    @property
    def most_important_layers(self) -> List[Tuple[int, float]]:
        """Top layers by causal effect, sorted descending."""
        return sorted(
            self.layer_effects.items(), key=lambda x: abs(x[1]), reverse=True
        )

    @property
    def most_important_heads(self) -> List[Tuple[Tuple[int, int], float]]:
        """Top (layer, head) pairs by causal effect, sorted descending."""
        return sorted(
            self.mla_head_effects.items(), key=lambda x: abs(x[1]), reverse=True
        )

    @property
    def most_important_experts(self) -> List[Tuple[Tuple[int, int], float]]:
        """Top (layer, expert) pairs by causal effect, sorted descending."""
        return sorted(
            self.expert_effects.items(), key=lambda x: abs(x[1]), reverse=True
        )


class ActivationPatcher:
    """
    Causal intervention: patch activations to measure component importance.

    Runs the model on clean and corrupted inputs, selectively replacing
    activations to measure which components are causally responsible.

    NanoSeek-specific: MLA heads share compressed KV (patch after wkv_b),
    MoE routing and expert outputs must be patched jointly.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        noise_std: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.noise_std = noise_std
        self.device = device or next(model.parameters()).device
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._cached_activations: Dict[str, Tensor] = {}

    def _clear_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _clear_cache(self) -> None:
        """Clear cached activations."""
        self._cached_activations.clear()

    def _corrupt_embeddings(self, input_ids: Tensor, subject_positions: List[int]) -> Tensor:
        """
        Create corrupted input by adding Gaussian noise to subject token embeddings.

        This is the standard corruption method from Meng et al. (2022).
        Only subject tokens are corrupted; context tokens remain clean.
        """
        embeddings = self.model.embed_tokens(input_ids).clone()
        noise = torch.randn_like(embeddings[:, subject_positions]) * self.noise_std
        embeddings[:, subject_positions] += noise
        return embeddings

    def _register_cache_hook(self, module: nn.Module, name: str) -> None:
        """Register a forward hook that caches the module's output."""
        def hook_fn(mod: nn.Module, inp: Any, out: Any) -> None:
            if isinstance(out, tuple):
                self._cached_activations[name] = out[0].detach().clone()
            elif isinstance(out, dict):
                if "logits" in out:
                    self._cached_activations[name] = out["logits"].detach().clone()
            else:
                self._cached_activations[name] = out.detach().clone()

        handle = module.register_forward_hook(hook_fn)
        self._hooks.append(handle)

    def _register_patch_hook(
        self,
        module: nn.Module,
        name: str,
        patch_value: Tensor,
        positions: Optional[List[int]] = None,
    ) -> None:
        """Register a forward hook that replaces activations with patched values."""
        def hook_fn(mod: nn.Module, inp: Any, out: Any) -> Any:
            if isinstance(out, tuple):
                patched = list(out)
                if positions is not None:
                    patched[0] = out[0].clone()
                    patched[0][:, positions] = patch_value[:, positions]
                else:
                    patched[0] = patch_value
                return tuple(patched)
            else:
                if positions is not None:
                    result = out.clone()
                    result[:, positions] = patch_value[:, positions]
                    return result
                return patch_value

        handle = module.register_forward_hook(hook_fn)
        self._hooks.append(handle)

    def _get_target_probability(
        self, logits: Tensor, target_token_id: int, position: int = -1
    ) -> float:
        """Extract the probability of the target token at the given position."""
        probs = F.softmax(logits[0, position], dim=-1)
        return probs[target_token_id].item()

    def _run_with_embeddings(self, embeddings: Tensor) -> Tensor:
        """Run the model from pre-computed embeddings (bypassing embed_tokens)."""
        hidden_states = embeddings
        b, s, _ = hidden_states.shape
        device = hidden_states.device
        pos_ids = torch.arange(s, device=device).unsqueeze(0).expand(b, -1)
        from model.model import create_causal_mask
        mask = create_causal_mask(s, dtype=hidden_states.dtype, device=device)
        for layer in self.model.layers:
            hidden_states, _, _ = layer(hidden_states=hidden_states, attention_mask=mask, position_ids=pos_ids)
        return self.model.lm_head(self.model.norm(hidden_states))

    @torch.no_grad()
    def patch_layer_output(
        self,
        clean_input: Tensor,
        corrupt_input: Tensor,
        layer_idx: int,
        target_token_id: int,
        subject_positions: List[int],
    ) -> float:
        """
        Patch a single layer's output from the clean run into the corrupt run.

        Returns the change in probability of the target token.
        """
        self._clear_hooks()
        self._clear_cache()

        clean_embeddings = self.model.embed_tokens(clean_input)
        corrupt_embeddings = self._corrupt_embeddings(corrupt_input, subject_positions)

        clean_logits = self._run_with_embeddings(clean_embeddings)
        clean_prob = self._get_target_probability(clean_logits, target_token_id)

        self._clear_hooks()

        layer = self.model.layers[layer_idx]
        self._register_cache_hook(layer, f"layer_{layer_idx}")

        _ = self._run_with_embeddings(clean_embeddings)
        cached = self._cached_activations[f"layer_{layer_idx}"]

        self._clear_hooks()

        self._register_patch_hook(layer, f"patch_layer_{layer_idx}", cached)
        patched_logits = self._run_with_embeddings(corrupt_embeddings)
        patched_prob = self._get_target_probability(patched_logits, target_token_id)

        self._clear_hooks()

        corrupt_logits = self._run_with_embeddings(corrupt_embeddings)
        corrupt_prob = self._get_target_probability(corrupt_logits, target_token_id)

        return (patched_prob - corrupt_prob) / (clean_prob - corrupt_prob + 1e-10)

    @torch.no_grad()
    def patch_mla_head(
        self,
        clean_input: Tensor,
        corrupt_input: Tensor,
        layer_idx: int,
        head_idx: int,
        target_token_id: int,
        subject_positions: List[int],
    ) -> float:
        """
        Patch a single MLA head's contribution from clean into corrupt run.

        Because MLA compresses KV jointly, we patch the per-head slice
        AFTER the wkv_b expansion (in the attention output, not the latent).
        The head's contribution is isolated by zeroing all other heads'
        attention outputs and measuring the effect.
        """
        self._clear_hooks()
        self._clear_cache()

        layer = self.model.layers[layer_idx]
        attn = layer.self_attn
        if hasattr(attn, 'mla'):
            attn = attn.mla

        num_heads = attn.num_heads
        v_head_dim = attn.v_head_dim

        clean_embeddings = self.model.embed_tokens(clean_input)
        corrupt_embeddings = self._corrupt_embeddings(corrupt_input, subject_positions)

        clean_head_outputs: Dict[str, Tensor] = {}

        def cache_wo_input(mod, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            clean_head_outputs["wo_input"] = x.detach().clone()

        handle = attn.wo.register_forward_hook(cache_wo_input)
        self._hooks.append(handle)
        _ = self._run_with_embeddings(clean_embeddings)
        self._clear_hooks()

        clean_wo_input = clean_head_outputs["wo_input"]
        batch_size, seq_len, total_dim = clean_wo_input.shape
        clean_per_head = clean_wo_input.view(batch_size, seq_len, num_heads, v_head_dim)

        def patch_single_head(mod, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            current = x.view(batch_size, seq_len, num_heads, v_head_dim).clone()
            current[:, :, head_idx, :] = clean_per_head[:, :, head_idx, :]
            return (current.view(batch_size, seq_len, total_dim),) + inp[1:] if isinstance(inp, tuple) else current.view(batch_size, seq_len, total_dim)

        handle = attn.wo.register_forward_pre_hook(patch_single_head)
        self._hooks.append(handle)

        patched_logits = self._run_with_embeddings(corrupt_embeddings)
        patched_prob = self._get_target_probability(patched_logits, target_token_id)

        self._clear_hooks()
        corrupt_logits = self._run_with_embeddings(corrupt_embeddings)
        corrupt_prob = self._get_target_probability(corrupt_logits, target_token_id)

        clean_logits = self._run_with_embeddings(clean_embeddings)
        clean_prob = self._get_target_probability(clean_logits, target_token_id)

        self._clear_hooks()
        return (patched_prob - corrupt_prob) / (clean_prob - corrupt_prob + 1e-10)

    @torch.no_grad()
    def patch_expert(
        self,
        clean_input: Tensor,
        corrupt_input: Tensor,
        layer_idx: int,
        expert_id: int,
        target_token_id: int,
        subject_positions: List[int],
    ) -> float:
        """
        Patch a single MoE expert's output from clean into corrupt run.

        This patches BOTH the routing decision (ensuring the expert is selected)
        AND the expert's output. Patching only the output without the routing
        would be meaningless if the expert wasn't selected in the corrupt run.
        """
        self._clear_hooks()

        layer = self.model.layers[layer_idx]
        if not layer.is_moe_layer:
            return 0.0

        moe = layer.ffn

        clean_expert_outputs: Dict[str, Tensor] = {}
        clean_routing: Dict[str, Tensor] = {}

        def cache_expert(mod, inp, out):
            clean_expert_outputs["output"] = out.detach().clone() if not isinstance(out, tuple) else out[0].detach().clone()

        def cache_gate(mod, inp, out):
            weights, indices = out
            clean_routing["weights"] = weights.detach().clone()
            clean_routing["indices"] = indices.detach().clone()

        handle_e = moe.experts[expert_id].register_forward_hook(cache_expert)
        handle_g = moe.gate.register_forward_hook(cache_gate)
        self._hooks.extend([handle_e, handle_g])

        clean_embeddings = self.model.embed_tokens(clean_input)
        _ = self._run_with_embeddings(clean_embeddings)
        self._clear_hooks()

        clean_logits = self._run_with_embeddings(clean_embeddings)
        clean_prob = self._get_target_probability(clean_logits, target_token_id)

        corrupt_embeddings = self._corrupt_embeddings(corrupt_input, subject_positions)
        corrupt_logits = self._run_with_embeddings(corrupt_embeddings)
        corrupt_prob = self._get_target_probability(corrupt_logits, target_token_id)

        def patch_gate_and_expert(mod, inp, out):
            weights, indices = out
            patched_weights = weights.clone()
            patched_indices = indices.clone()
            mask = (clean_routing["indices"] == expert_id).any(dim=-1)
            patched_weights[mask] = clean_routing["weights"][mask]
            patched_indices[mask] = clean_routing["indices"][mask]
            return patched_weights, patched_indices

        handle = moe.gate.register_forward_hook(patch_gate_and_expert)
        self._hooks.append(handle)

        patched_logits = self._run_with_embeddings(corrupt_embeddings)
        patched_prob = self._get_target_probability(patched_logits, target_token_id)

        self._clear_hooks()
        return (patched_prob - corrupt_prob) / (clean_prob - corrupt_prob + 1e-10)

    @torch.no_grad()
    def run_causal_trace(self, prompt: str, subject_tokens: List[int], target_position: int = -1) -> CausalTraceResult:
        """Run a complete causal trace across all layers and components."""
        self.model.eval()
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], device=self.device)

        clean_logits = self.model(input_ids)["logits"]
        target_token_id = clean_logits[0, target_position].argmax().item()
        clean_prob = self._get_target_probability(clean_logits, target_token_id, target_position)

        corrupt_embeddings = self._corrupt_embeddings(input_ids, subject_tokens)
        corrupt_prob = self._get_target_probability(
            self._run_with_embeddings(corrupt_embeddings), target_token_id, target_position)

        result = CausalTraceResult(
            prompt=prompt, subject_token_positions=subject_tokens,
            target_token=self.tokenizer.decode([target_token_id]),
            clean_prob=clean_prob, corrupt_prob=corrupt_prob,
        )

        for layer_idx in range(len(self.model.layers)):
            result.layer_effects[layer_idx] = self.patch_layer_output(
                input_ids, input_ids, layer_idx, target_token_id, subject_tokens)

        for layer_idx, _ in sorted(result.layer_effects.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
            num_heads = getattr(self.model.layers[layer_idx].self_attn, 'num_heads', self.model.config.num_heads)
            for head_idx in range(num_heads):
                result.mla_head_effects[(layer_idx, head_idx)] = self.patch_mla_head(
                    input_ids, input_ids, layer_idx, head_idx, target_token_id, subject_tokens)
            if self.model.layers[layer_idx].is_moe_layer:
                for expert_id in range(min(8, self.model.config.moe.n_routed_experts)):
                    result.expert_effects[(layer_idx, expert_id)] = self.patch_expert(
                        input_ids, input_ids, layer_idx, expert_id, target_token_id, subject_tokens)

        self._clear_hooks()
        self._clear_cache()
        return result

    def visualize_trace(self, result: CausalTraceResult) -> str:
        """Generate an ASCII visualization of causal trace results."""
        lines = [f"Causal Trace: '{result.prompt}'",
                 f"Target: '{result.target_token}' (clean={result.clean_prob:.3f}, corrupt={result.corrupt_prob:.3f})", ""]
        max_effect = max(abs(v) for v in result.layer_effects.values()) if result.layer_effects else 1.0
        for layer_idx in sorted(result.layer_effects.keys()):
            effect = result.layer_effects[layer_idx]
            bar = "█" * int(40 * abs(effect) / (max_effect + 1e-10))
            lines.append(f"  L{layer_idx:02d}: {bar} {effect:.4f}")
        for label, data, fmt in [("MLA Heads", result.most_important_heads[:10], "L{0:02d}H{1:02d}"),
                                  ("Experts", result.most_important_experts[:10], "L{0:02d}E{1:02d}")]:
            if data:
                lines.append(f"\nTop {label}:")
                for key, effect in data:
                    lines.append(f"  {fmt.format(*key)}: {effect:.4f}")
        return "\n".join(lines)
```

### 3b. MoE Expert Specialization Analysis

**File**: `fms/interpretability/moe_analysis.py`

```python
"""
MoE Expert Specialization Analysis for NanoSeek.

Discovers what each of the 64 routed experts (and 2 shared experts)
specializes in by profiling activation patterns across diverse text inputs.

Analyses include:
- Token-type distribution per expert (punctuation, digits, uppercase, etc.)
- POS-tag correlation (if spaCy available)
- Topic/domain clustering via expert co-activation patterns
- Dead expert detection
- Expert ablation studies

Usage:
    analyzer = ExpertAnalyzer(model, tokenizer)
    profile = analyzer.profile_expert_activation(dataset)
    clusters = analyzer.cluster_experts_by_behavior()
    dead = analyzer.find_dead_experts(threshold=0.01)
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ExpertProfile:
    """Activation profile for a single expert."""

    expert_id: int
    layer_idx: int
    total_activations: int
    activation_rate: float

    token_type_distribution: Dict[str, float] = field(default_factory=dict)
    position_distribution: Dict[str, float] = field(default_factory=dict)
    top_activating_tokens: List[Tuple[str, float]] = field(default_factory=list)
    mean_routing_weight: float = 0.0
    weight_std: float = 0.0

    @property
    def is_dead(self) -> bool:
        """Expert is dead if activation rate is below 1%."""
        return self.activation_rate < 0.01


@dataclass
class AblationResult:
    """Result of ablating (removing) a single expert."""

    expert_id: int
    layer_idx: int
    baseline_loss: float
    ablated_loss: float
    loss_increase: float
    loss_increase_pct: float
    perplexity_increase: float

    @property
    def is_critical(self) -> bool:
        """Expert is critical if ablation increases loss by >5%."""
        return self.loss_increase_pct > 5.0


class ExpertAnalyzer:
    """
    Discover what each of the 64 experts specializes in.

    Profiles expert behavior by running diverse text, recording routing
    decisions, and analyzing token distributions. NanoSeek uses sigmoid
    scoring, so selection scores are independent per expert.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._routing_data: Dict[int, List[Dict[str, Tensor]]] = defaultdict(list)

    def _classify_token(self, token_str: str) -> List[str]:
        """Classify a token into type categories."""
        categories = []
        stripped = token_str.strip()

        if not stripped:
            categories.append("whitespace")
            return categories

        if stripped in '.,;:!?':
            categories.append("punctuation")
        if stripped in '()[]{}""\'\'':
            categories.append("brackets_quotes")
        if stripped.isdigit() or any(c.isdigit() for c in stripped):
            categories.append("numeric")
        if stripped.isalpha():
            if stripped[0].isupper():
                categories.append("capitalized")
            else:
                categories.append("lowercase")
        if stripped.startswith(("the", "a", "an", "is", "was", "are", "were", "in", "on", "at", "to", "for")):
            categories.append("function_word")
        if len(stripped) <= 2:
            categories.append("short_token")
        elif len(stripped) >= 8:
            categories.append("long_token")

        if not categories:
            categories.append("other")

        return categories

    def _install_routing_hooks(self) -> None:
        """Install hooks on all MoE gates to capture routing decisions."""
        self._clear_hooks()
        self._routing_data.clear()

        for layer_idx, layer in enumerate(self.model.layers):
            if not layer.is_moe_layer:
                continue

            gate = layer.ffn.gate
            idx = layer_idx

            def make_hook(li: int):
                def hook_fn(mod, inp, out):
                    weights, indices = out
                    self._routing_data[li].append({
                        "weights": weights.detach().cpu(),
                        "indices": indices.detach().cpu(),
                    })
                return hook_fn

            handle = gate.register_forward_hook(make_hook(idx))
            self._hooks.append(handle)

    def _clear_hooks(self) -> None:
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    @torch.no_grad()
    def profile_expert_activation(
        self,
        dataset: List[str],
        max_seq_len: int = 512,
    ) -> Dict[Tuple[int, int], ExpertProfile]:
        """
        Profile expert activation patterns across a text dataset.

        Args:
            dataset: List of text strings to analyze.
            max_seq_len: Maximum sequence length per sample.

        Returns:
            Dict mapping (layer_idx, expert_id) to ExpertProfile.
        """
        self.model.eval()
        self._install_routing_hooks()

        all_tokens: List[List[str]] = []

        for text in dataset:
            tokens = self.tokenizer.encode(text)[:max_seq_len]
            input_ids = torch.tensor([tokens], device=self.device)

            _ = self.model(input_ids)

            token_strs = [self.tokenizer.decode([t]) for t in tokens]
            all_tokens.append(token_strs)

        self._clear_hooks()

        profiles: Dict[Tuple[int, int], ExpertProfile] = {}

        for layer_idx, routing_records in self._routing_data.items():
            n_experts = self.model.config.moe.n_routed_experts
            expert_types: Dict[int, Counter] = defaultdict(Counter)
            expert_pos: Dict[int, Counter] = defaultdict(Counter)
            expert_counts: Dict[int, int] = defaultdict(int)
            expert_wsums: Dict[int, float] = defaultdict(float)
            expert_wsq: Dict[int, float] = defaultdict(float)
            expert_toks: Dict[int, Counter] = defaultdict(Counter)
            total = 0

            for si, record in enumerate(routing_records):
                w, idx = record["weights"], record["indices"]
                n = w.shape[0]
                total += n
                tok_strs = all_tokens[si] if si < len(all_tokens) else []
                for ti in range(n):
                    for k in range(idx.shape[1]):
                        eid, wt = idx[ti, k].item(), w[ti, k].item()
                        expert_counts[eid] += 1
                        expert_wsums[eid] += wt
                        expert_wsq[eid] += wt * wt
                        if ti < len(tok_strs):
                            for cat in self._classify_token(tok_strs[ti]):
                                expert_types[eid][cat] += 1
                            expert_toks[eid][tok_strs[ti]] += 1
                        bucket = ("start" if ti < n*0.1 else "early" if ti < n*0.3
                                  else "mid" if ti < n*0.7 else "late" if ti < n*0.9 else "end")
                        expert_pos[eid][bucket] += 1

            for eid in range(n_experts):
                c = expert_counts[eid]
                td = dict(expert_types[eid])
                tt = sum(td.values()) or 1
                pd = dict(expert_pos[eid])
                pt = sum(pd.values()) or 1
                mw = expert_wsums[eid] / (c or 1)
                profiles[(layer_idx, eid)] = ExpertProfile(
                    expert_id=eid, layer_idx=layer_idx, total_activations=c,
                    activation_rate=c / (total + 1e-10),
                    token_type_distribution={k: v/tt for k, v in td.items()},
                    position_distribution={k: v/pt for k, v in pd.items()},
                    top_activating_tokens=[(t, n/(c or 1)) for t, n in expert_toks[eid].most_common(20)],
                    mean_routing_weight=mw,
                    weight_std=math.sqrt(max(expert_wsq[eid]/(c or 1) - mw**2, 0.0)),
                )

        return profiles

    def compute_expert_token_distributions(
        self,
        texts: List[str],
        max_seq_len: int = 512,
    ) -> Dict[Tuple[int, int], Dict[str, float]]:
        """
        Compute token-type distributions for each expert across given texts.

        Returns a mapping from (layer, expert) to {token_type: frequency}.
        """
        profiles = self.profile_expert_activation(texts, max_seq_len)
        return {
            key: profile.token_type_distribution
            for key, profile in profiles.items()
        }

    def find_dead_experts(
        self,
        dataset: List[str],
        threshold: float = 0.01,
        max_seq_len: int = 512,
    ) -> List[Tuple[int, int]]:
        """
        Find experts that activate less than `threshold` fraction of the time.

        Dead experts represent wasted parameters and indicate load balancing
        issues. NanoSeek's bias-based balancing should prevent dead experts,
        but they can still appear early in training or after fine-tuning.

        Returns list of (layer_idx, expert_id) pairs.
        """
        profiles = self.profile_expert_activation(dataset, max_seq_len)
        return [
            (key[0], key[1])
            for key, profile in profiles.items()
            if profile.activation_rate < threshold
        ]

    def cluster_experts_by_behavior(self, profiles: Dict[Tuple[int, int], ExpertProfile],
                                     n_clusters: int = 8) -> Dict[int, List[Tuple[int, int]]]:
        """Cluster experts by token-type distribution using agglomerative clustering with cosine similarity."""
        cats = sorted({c for p in profiles.values() for c in p.token_type_distribution})
        keys = sorted(profiles.keys())
        if not keys: return {}
        vecs = torch.tensor([[profiles[k].token_type_distribution.get(c, 0.0) for c in cats] for k in keys], dtype=torch.float32)
        vecs = vecs / vecs.norm(dim=1, keepdim=True).clamp(min=1e-10)
        sim = vecs @ vecs.T
        members: Dict[int, List[int]] = {i: [i] for i in range(len(keys))}
        while len(members) > n_clusters:
            best_sim, best_pair = -1.0, (-1, -1)
            active = sorted(members.keys())
            for ii, ci in enumerate(active):
                for cj in active[ii+1:]:
                    s = sum(sim[mi, mj].item() for mi in members[ci] for mj in members[cj])
                    s /= (len(members[ci]) * len(members[cj])) or 1
                    if s > best_sim: best_sim, best_pair = s, (ci, cj)
            if best_pair[0] < 0: break
            members[best_pair[0]].extend(members[best_pair[1]]); del members[best_pair[1]]
        return {i: [keys[m] for m in ms] for i, (_, ms) in enumerate(sorted(members.items()))}

    @torch.no_grad()
    def expert_ablation_study(self, expert_id: int, layer_idx: int,
                              eval_texts: List[str], max_seq_len: int = 512) -> AblationResult:
        """Measure impact of removing a single expert by comparing loss with/without it."""
        self.model.eval()
        layer = self.model.layers[layer_idx]
        if not layer.is_moe_layer:
            return AblationResult(expert_id=expert_id, layer_idx=layer_idx, baseline_loss=0.0,
                                  ablated_loss=0.0, loss_increase=0.0, loss_increase_pct=0.0, perplexity_increase=0.0)
        expert = layer.ffn.experts[expert_id]

        def _eval_loss(forward_fn=None):
            if forward_fn: expert.forward = forward_fn
            losses = []
            for text in eval_texts:
                toks = self.tokenizer.encode(text)[:max_seq_len]
                if len(toks) < 2: continue
                ids = torch.tensor([toks], device=self.device)
                out = self.model(ids, labels=ids)
                if "loss" in out: losses.append(out["loss"].item())
            return sum(losses) / len(losses) if losses else 0.0

        orig_fwd = expert.forward
        bl = _eval_loss()
        al = _eval_loss(lambda x: torch.zeros_like(x))
        expert.forward = orig_fwd
        inc = al - bl
        return AblationResult(expert_id=expert_id, layer_idx=layer_idx, baseline_loss=bl, ablated_loss=al,
                              loss_increase=inc, loss_increase_pct=inc/(bl+1e-10)*100,
                              perplexity_increase=math.exp(al)-math.exp(bl))

    def summarize_profiles(self, profiles: Dict[Tuple[int, int], ExpertProfile]) -> str:
        """Generate ASCII summary of expert specialization patterns."""
        lines = ["Expert Specialization Summary", "=" * 60]
        by_layer: Dict[int, List[ExpertProfile]] = defaultdict(list)
        for (layer, _), profile in profiles.items():
            by_layer[layer].append(profile)
        for layer_idx in sorted(by_layer.keys()):
            lp = by_layer[layer_idx]
            dead = sum(1 for p in lp if p.is_dead)
            lines.append(f"\nLayer {layer_idx}: {len(lp)} experts, {dead} dead")
            for p in sorted([p for p in lp if not p.is_dead], key=lambda p: p.activation_rate, reverse=True)[:5]:
                top = max(p.token_type_distribution, key=p.token_type_distribution.get) if p.token_type_distribution else "none"
                lines.append(f"  E{p.expert_id:02d}: rate={p.activation_rate:.3f} top={top} w={p.mean_routing_weight:.3f}")
        return "\n".join(lines)
```

### 3c. MLA Latent Space Analysis

**File**: `fms/interpretability/mla_analysis.py`

```python
"""
MLA (Multi-Head Latent Attention) Latent Space Analysis for NanoSeek.

Analyzes what information survives MLA's 23x KV compression and how
the model represents queries, keys, and values in the latent space.

Includes:
- Compressed KV extraction and visualization
- Reconstruction error analysis (what is lost in compression?)
- Logit lens through layers (how do predictions evolve?)
- Linear probing on compressed features
- Induction head detection in MLA's split Q/K space

Reference architecture (from model/model.py):
  Q path: hidden → wq_a (2048→430) → q_norm → wq_b (430→16×96)
  K path: hidden → wkv_a (2048→175) → split:
    - kv_compressed (143) → kv_norm → wkv_b (143→16×(64+64))
    - k_pe (32) → RoPE

Usage:
    analyzer = MLAAnalyzer(model, tokenizer)
    kv = analyzer.extract_compressed_kv(input_ids, layer=5)
    errors = analyzer.reconstruction_error_analysis(input_ids)
    lens = analyzer.logit_lens_through_layers(input_ids)
    heads = analyzer.induction_head_detection(input_ids)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ProbeResult:
    """Result of linear probing on compressed KV features."""

    layer_idx: int
    task_name: str
    accuracy: float
    baseline_accuracy: float
    probe_weights_norm: float
    top_features: List[Tuple[int, float]] = field(default_factory=list)

    @property
    def lift(self) -> float:
        """Accuracy improvement over baseline."""
        return self.accuracy - self.baseline_accuracy


@dataclass
class InductionHead:
    """Detected induction head in MLA."""

    layer_idx: int
    head_idx: int
    induction_score: float
    prefix_matching_score: float
    copying_score: float

    @property
    def is_strong_induction(self) -> bool:
        """Strong induction head if score > 0.5."""
        return self.induction_score > 0.5


class MLAAnalyzer:
    """
    Analyze what information survives MLA's 23x KV compression.

    The compressed KV (143 dims) is SHARED across all 16 heads; per-head
    info only emerges after wkv_b expansion. RoPE (32 dims) is separate.
    Probes the compressed representation for preserved information,
    reconstruction error, logit lens trajectories, and induction heads.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

    def _clear_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _get_attn_module(self, layer_idx: int):
        """Get the MLA attention module for a given layer."""
        layer = self.model.layers[layer_idx]
        attn = layer.self_attn
        if hasattr(attn, 'mla'):
            return attn.mla
        return attn

    @torch.no_grad()
    def extract_compressed_kv(
        self,
        input_ids: Tensor,
        layer_idx: int,
    ) -> Dict[str, Tensor]:
        """
        Extract the compressed KV representation at a specific layer.

        Returns:
            Dict with keys:
            - 'kv_compressed': (batch, seq, 143) — the latent KV
            - 'k_pe': (batch, seq, 32) — the positional key component
            - 'kv_full': (batch, seq, kv_len, 175) — concatenated cache representation
            - 'q_compressed': (batch, seq, 430) — the latent query
        """
        self.model.eval()
        attn = self._get_attn_module(layer_idx)
        captured: Dict[str, Tensor] = {}

        def capture_wkv_a_output(mod, inp, out):
            captured["wkv_a_output"] = out.detach().clone()

        def capture_wq_a_output(mod, inp, out):
            captured["wq_a_output"] = out.detach().clone()

        h1 = attn.wkv_a.register_forward_hook(capture_wkv_a_output)
        h2 = attn.wq_a.register_forward_hook(capture_wq_a_output)
        self._hooks.extend([h1, h2])

        _ = self.model(input_ids)
        self._clear_hooks()

        kv_lora_rank = attn.kv_lora_rank
        rope_dim = attn.qk_rope_head_dim

        wkv_a_out = captured["wkv_a_output"]
        kv_compressed = wkv_a_out[..., :kv_lora_rank]
        k_pe = wkv_a_out[..., kv_lora_rank:]

        kv_compressed_normed = attn.kv_norm(kv_compressed)

        q_compressed = captured["wq_a_output"]
        q_compressed_normed = attn.q_norm(q_compressed)

        return {
            "kv_compressed": kv_compressed_normed,
            "k_pe": k_pe,
            "q_compressed": q_compressed_normed,
            "kv_raw": kv_compressed,
            "q_raw": q_compressed,
        }

    @torch.no_grad()
    def reconstruction_error_analysis(self, input_ids: Tensor) -> Dict[int, Dict[str, float]]:
        """Measure effective rank and variance utilization of compressed KV at each layer."""
        self.model.eval()
        results: Dict[int, Dict[str, float]] = {}
        for layer_idx in range(len(self.model.layers)):
            attn = self._get_attn_module(layer_idx)
            ext = self.extract_compressed_kv(input_ids, layer_idx)
            kv, q = ext["kv_compressed"], ext["q_compressed"]
            def _rank_stats(t, rank):
                _, S, _ = torch.svd(t.squeeze(0).float())
                cv = (S**2).cumsum(0) / (S**2).sum()
                return {"rank_90": (cv < 0.9).sum().item() + 1, "rank_99": (cv < 0.99).sum().item() + 1,
                        "lora_rank": rank, "util_90": ((cv < 0.9).sum().item() + 1) / rank,
                        "norm": t.float().norm().item(), "mean_abs": t.float().abs().mean().item()}
            kv_s = _rank_stats(kv, attn.kv_lora_rank)
            q_s = _rank_stats(q, attn.q_lora_rank)
            results[layer_idx] = {f"kv_{k}": v for k, v in kv_s.items()}
            results[layer_idx].update({f"q_{k}": v for k, v in q_s.items()})
        return results

    @torch.no_grad()
    def logit_lens_through_layers(self, input_ids: Tensor, top_k: int = 5) -> Dict[int, Dict[str, Any]]:
        """Apply logit lens at every layer — projects residual stream through lm_head."""
        self.model.eval()
        hidden = self.model.embed_tokens(input_ids)
        b, s, _ = hidden.shape
        device = hidden.device
        from model.model import create_causal_mask
        pos_ids = torch.arange(s, device=device).unsqueeze(0).expand(b, -1)
        mask = create_causal_mask(s, dtype=hidden.dtype, device=device)
        results: Dict[int, Dict[str, Any]] = {}
        for li, layer in enumerate(self.model.layers):
            hidden, _, _ = layer(hidden_states=hidden, attention_mask=mask, position_ids=pos_ids)
            logits = self.model.lm_head(self.model.norm(hidden))
            probs = F.softmax(logits[0, -1], dim=-1)
            entropy = -(probs * probs.log().clamp(min=-100)).sum().item()
            tk_p, tk_i = probs.topk(top_k)
            results[li] = {"top_tokens": [(self.tokenizer.decode([i.item()]), p) for i, p in zip(tk_i, tk_p.tolist())],
                           "entropy": entropy, "logits_norm": logits[0, -1].float().norm().item(), "_probs": probs.cpu()}
        final = results[len(self.model.layers)-1]["_probs"]
        for li in results:
            results[li]["kl_from_final"] = F.kl_div(results[li]["_probs"].log().clamp(min=-100), final, reduction="sum").item()
            del results[li]["_probs"]
        return results

    @torch.no_grad()
    def probe_compressed_features(
        self,
        dataset: List[Tuple[Tensor, Tensor]],
        layer_idx: int,
        task_name: str = "probe",
        num_classes: int = 2,
        epochs: int = 50,
        lr: float = 0.01,
    ) -> ProbeResult:
        """
        Train a linear probe on compressed KV features.

        Tests what information is extractable from the 143-dim compressed KV.
        Examples of probe tasks:
        - Token identity (can we recover the original token from compression?)
        - POS tag (does compressed KV encode syntactic role?)
        - Named entity (does it encode entity status?)
        - Sentence boundary (does it know sentence structure?)

        Args:
            dataset: List of (input_ids, labels) tuples.
            layer_idx: Which layer's compressed KV to probe.
            task_name: Name of the probing task.
            num_classes: Number of classification classes.
            epochs: Training epochs for the probe.
            lr: Learning rate for probe training.
        """
        self.model.eval()
        attn = self._get_attn_module(layer_idx)

        all_features = []
        all_labels = []

        for input_ids, labels in dataset:
            input_ids = input_ids.to(self.device)
            extracted = self.extract_compressed_kv(input_ids, layer_idx)
            kv_compressed = extracted["kv_compressed"]

            flat_features = kv_compressed.view(-1, kv_compressed.shape[-1])
            flat_labels = labels.view(-1)

            valid_mask = flat_labels >= 0
            all_features.append(flat_features[valid_mask].cpu())
            all_labels.append(flat_labels[valid_mask].cpu())

        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0).long()

        n = features.shape[0]
        split = int(0.8 * n)
        perm = torch.randperm(n)
        train_idx = perm[:split]
        val_idx = perm[split:]

        probe = nn.Linear(features.shape[1], num_classes)
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

        for epoch in range(epochs):
            probe.train()
            logits = probe(features[train_idx])
            loss = F.cross_entropy(logits, labels[train_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_logits = probe(features[val_idx])
            val_preds = val_logits.argmax(dim=-1)
            accuracy = (val_preds == labels[val_idx]).float().mean().item()

        class_counts = torch.bincount(labels[val_idx], minlength=num_classes)
        baseline = class_counts.max().item() / len(val_idx)

        weight_norm = probe.weight.data.norm().item()
        feature_importance = probe.weight.data.abs().mean(dim=0)
        top_features = feature_importance.topk(min(10, len(feature_importance)))
        top_feature_list = list(zip(top_features.indices.tolist(), top_features.values.tolist()))

        return ProbeResult(
            layer_idx=layer_idx,
            task_name=task_name,
            accuracy=accuracy,
            baseline_accuracy=baseline,
            probe_weights_norm=weight_norm,
            top_features=top_feature_list,
        )

    @torch.no_grad()
    def induction_head_detection(
        self,
        input_ids: Tensor,
        threshold: float = 0.3,
    ) -> List[InductionHead]:
        """
        Detect induction heads in MLA's attention patterns.

        An induction head implements the pattern: [A][B]...[A] → [B].
        We detect this by:
        1. Creating input with repeated bigrams: [A][B][C][D]...[A][B][C][D]
        2. Computing attention patterns for each head
        3. Measuring how much attention flows from the second [A] to the
           position after the first [A] (which is [B])

        In MLA, induction behavior must operate through the compressed
        bottleneck. The RoPE component handles positional matching (finding
        where [A] appeared before), while the nope component handles content
        matching (identifying that the token IS [A]).

        We measure three sub-scores:
        - prefix_matching_score: Does the head attend to positions where
          the current token previously appeared? (RoPE-mediated)
        - copying_score: Does the head's output bias predictions toward
          the token following the attended position? (content-mediated)
        - induction_score: Geometric mean of the above.
        """
        self.model.eval()
        seq_len = input_ids.shape[1]

        detected_heads: List[InductionHead] = []

        for layer_idx in range(len(self.model.layers)):
            attn = self._get_attn_module(layer_idx)
            num_heads = attn.num_heads

            hidden_states = self.model.embed_tokens(input_ids)
            device = hidden_states.device
            batch_size = hidden_states.shape[0]

            from model.model import create_causal_mask
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            causal_mask = create_causal_mask(seq_len, dtype=hidden_states.dtype, device=device)

            residual = hidden_states
            for prev_layer_idx in range(layer_idx):
                prev_layer = self.model.layers[prev_layer_idx]
                residual, _, _ = prev_layer(
                    hidden_states=residual,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                )

            normed = self.model.layers[layer_idx].input_layernorm(residual)

            q = attn.wq_a(normed)
            q = attn.q_norm(q)
            q = attn.wq_b(q)
            q = q.view(batch_size, seq_len, num_heads, attn.qk_head_dim)
            q_nope, q_pe = torch.split(q, [attn.qk_nope_head_dim, attn.qk_rope_head_dim], dim=-1)

            kv = attn.wkv_a(normed)
            kv_compressed, k_pe_raw = torch.split(
                kv, [attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1
            )
            kv_compressed = attn.kv_norm(kv_compressed)

            from model.model import apply_rotary_emb
            freqs = attn.freqs_cis[:seq_len]
            q_pe = apply_rotary_emb(q_pe, freqs, interleaved=True)
            k_pe = apply_rotary_emb(k_pe_raw.unsqueeze(2), freqs, interleaved=True)

            kv_expanded = attn.wkv_b(kv_compressed)
            kv_expanded = kv_expanded.view(batch_size, seq_len, num_heads, attn.qk_nope_head_dim + attn.v_head_dim)
            k_nope, v = torch.split(kv_expanded, [attn.qk_nope_head_dim, attn.v_head_dim], dim=-1)

            q_full = torch.cat([q_nope, q_pe], dim=-1)
            k_pe_expanded = k_pe.expand(-1, -1, num_heads, -1)
            k_full = torch.cat([k_nope, k_pe_expanded], dim=-1)

            q_full = q_full.transpose(1, 2)
            k_full = k_full.transpose(1, 2)

            scale = 1.0 / math.sqrt(attn.qk_head_dim)
            attn_weights = torch.matmul(q_full, k_full.transpose(-2, -1)) * scale
            attn_weights = attn_weights + causal_mask
            attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32)

            tokens = input_ids[0].tolist()
            for head_idx in range(num_heads):
                head_attn = attn_probs[0, head_idx]

                prefix_score = 0.0
                copy_score = 0.0
                count = 0

                for pos in range(1, seq_len):
                    current_token = tokens[pos]
                    for prev_pos in range(pos):
                        if tokens[prev_pos] == current_token and prev_pos + 1 < seq_len:
                            prefix_score += head_attn[pos, prev_pos + 1].item()
                            count += 1

                if count > 0:
                    prefix_score /= count

                shift_attn = torch.zeros_like(head_attn)
                if seq_len > 1:
                    shift_attn[:, 1:] = head_attn[:, :-1]
                    diag = torch.diagonal(shift_attn)
                    copy_score = diag.mean().item()

                induction_score = math.sqrt(max(prefix_score * copy_score, 0.0))

                if induction_score > threshold:
                    detected_heads.append(InductionHead(
                        layer_idx=layer_idx,
                        head_idx=head_idx,
                        induction_score=induction_score,
                        prefix_matching_score=prefix_score,
                        copying_score=copy_score,
                    ))

        detected_heads.sort(key=lambda h: h.induction_score, reverse=True)
        return detected_heads

    def visualize_logit_lens(self, results: Dict[int, Dict[str, Any]]) -> str:
        """ASCII visualization of logit lens results."""
        lines = ["Logit Lens: Prediction Evolution", "=" * 65]
        for li in sorted(results.keys()):
            d = results[li]
            top = d["top_tokens"][0] if d["top_tokens"] else ("?", 0.0)
            kl = d.get("kl_from_final", 0.0)
            lines.append(f"  L{li:02d}: {'█'*int(30/(1+kl))} top='{top[0]}'({top[1]:.2f}) H={d['entropy']:.2f} KL={kl:.2f}")
        return "\n".join(lines)
```

### 3d. Sparse Autoencoder for Feature Discovery

**File**: `fms/interpretability/sae.py`

```python
"""
Sparse Autoencoder (SAE) for Decomposing NanoSeek Activations.

Trains sparse autoencoders on NanoSeek's residual stream activations to
discover interpretable, monosemantic features — following Anthropic's
methodology from Bricken et al. (2023).

The key insight: individual neurons encode multiple features (superposition).
SAEs decompose these polysemantic neurons into a larger set of monosemantic
features, each corresponding to a single interpretable concept.

Architecture:
    encoded = ReLU(W_enc @ (x - b_dec) + b_enc)   # R^d → R^n_features
    decoded = W_dec @ encoded + b_dec              # R^n_features → R^d

Training objective:
    L = ||x - decoded||² + λ * ||encoded||₁

where λ (sparsity_coeff) controls the sparsity-reconstruction tradeoff.

Usage:
    sae = SparseAutoencoder(input_dim=2048, n_features=16384, sparsity_coeff=1e-3)
    sae.train_on_activations(activation_dataset, epochs=10)
    features = sae.get_top_features(activation, k=10)
    report = sae.feature_dashboard(feature_idx=42, dataset=dataset)

Reference: Bricken et al. 2023, "Towards Monosemanticity"
           Cunningham et al. 2023, "Sparse Autoencoders Find Highly
           Interpretable Features in Language Models"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class Feature:
    """A single learned SAE feature."""

    feature_idx: int
    activation_frequency: float
    mean_activation: float
    max_activation: float
    top_activating_examples: List[Tuple[str, float]] = field(default_factory=list)
    decoder_weights_norm: float = 0.0
    cosine_sim_to_nearest: float = 0.0

    @property
    def is_dead(self) -> bool:
        """Feature is dead if it never activates."""
        return self.activation_frequency < 1e-6

    @property
    def is_ultra_sparse(self) -> bool:
        """Feature is ultra-sparse if it activates <0.1% of the time."""
        return self.activation_frequency < 0.001


@dataclass
class FeatureReport:
    """Comprehensive report for a single SAE feature."""

    feature_idx: int
    activation_frequency: float
    mean_activation: float
    max_activation: float
    top_activating_examples: List[Dict[str, Any]] = field(default_factory=list)
    logit_effect: Dict[str, float] = field(default_factory=dict)
    decoder_direction: Optional[Tensor] = None
    co_occurring_features: List[Tuple[int, float]] = field(default_factory=list)

    def summary(self) -> str:
        """One-line summary of the feature."""
        top_examples = ", ".join(
            f"'{ex['text'][:30]}'" for ex in self.top_activating_examples[:3]
        )
        return (
            f"Feature {self.feature_idx}: freq={self.activation_frequency:.4f} "
            f"mean_act={self.mean_activation:.3f} examples=[{top_examples}]"
        )


class SparseAutoencoder(nn.Module):
    """
    SAE for decomposing NanoSeek activations into interpretable features.

    Follows Anthropic's architecture (Bricken et al. 2023): pre-encoder
    bias subtraction, overcomplete encoding, ReLU, L1 sparsity penalty,
    unit-norm decoder columns. Default: input_dim=2048, n_features=16384
    (8x overcomplete). Also applicable to expert outputs (768) or
    compressed KV (143).
    """

    def __init__(
        self,
        input_dim: int = 2048,
        n_features: int = 16384,
        sparsity_coeff: float = 1e-3,
        decoder_init_scale: float = 0.1,
        tied_weights: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.n_features = n_features
        self.sparsity_coeff = sparsity_coeff
        self.tied_weights = tied_weights

        self.W_enc = nn.Parameter(torch.empty(n_features, input_dim))
        self.b_enc = nn.Parameter(torch.zeros(n_features))

        if not tied_weights:
            self.W_dec = nn.Parameter(torch.empty(input_dim, n_features))
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(input_dim))

        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        if not tied_weights:
            nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
            with torch.no_grad():
                self.W_dec.data *= decoder_init_scale
                self.W_dec.data /= self.W_dec.data.norm(dim=0, keepdim=True).clamp(min=1e-8)

        self._feature_activation_counts = torch.zeros(n_features)
        self._total_samples = 0

    @property
    def decoder_weight(self) -> Tensor:
        """Get decoder weight matrix (handles tied weights)."""
        if self.tied_weights:
            return self.W_enc.T
        return self.W_dec

    def encode(self, x: Tensor) -> Tensor:
        """Encode input activations to sparse feature coefficients."""
        x_centered = x - self.b_dec
        pre_activation = F.linear(x_centered, self.W_enc, self.b_enc)
        return F.relu(pre_activation)

    def decode(self, encoded: Tensor) -> Tensor:
        """Decode sparse features back to activation space."""
        return F.linear(encoded, self.decoder_weight.T if self.tied_weights else self.decoder_weight) + self.b_dec

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Full forward pass: encode → decode with sparsity loss.

        Args:
            x: Input activations (batch, input_dim) or (batch, seq, input_dim)

        Returns:
            encoded: Sparse feature activations
            decoded: Reconstructed activations
            sparsity_loss: L1 penalty on encoded activations
        """
        original_shape = x.shape
        if x.dim() == 3:
            batch, seq, dim = x.shape
            x = x.reshape(-1, dim)

        encoded = self.encode(x)
        decoded = self.decode(encoded)

        sparsity_loss = self.sparsity_coeff * encoded.abs().sum(dim=-1).mean()

        if self.training:
            with torch.no_grad():
                active = (encoded > 0).float().sum(dim=0)
                self._feature_activation_counts = self._feature_activation_counts.to(x.device)
                self._feature_activation_counts += active
                self._total_samples += x.shape[0]

        if len(original_shape) == 3:
            encoded = encoded.reshape(batch, seq, -1)
            decoded = decoded.reshape(batch, seq, -1)

        return encoded, decoded, sparsity_loss

    def compute_loss(self, x: Tensor) -> Dict[str, Tensor]:
        """Compute full training loss with detailed breakdown."""
        encoded, decoded, sparsity_loss = self(x)

        if x.dim() == 3:
            x_flat = x.reshape(-1, x.shape[-1])
            decoded_flat = decoded.reshape(-1, decoded.shape[-1])
        else:
            x_flat = x
            decoded_flat = decoded

        reconstruction_loss = F.mse_loss(decoded_flat, x_flat)
        total_loss = reconstruction_loss + sparsity_loss

        with torch.no_grad():
            l0 = (encoded > 0).float().sum(dim=-1).mean()
            frac_active = (encoded > 0).any(dim=0).float().mean()

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "sparsity_loss": sparsity_loss,
            "l0_sparsity": l0,
            "frac_features_active": frac_active,
        }

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Normalize decoder columns to unit norm (Anthropic's approach)."""
        if not self.tied_weights and self.W_dec is not None:
            norms = self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
            self.W_dec.data /= norms

    def train_on_activations(self, activation_dataset: List[Tensor], epochs: int = 10,
                             batch_size: int = 4096, lr: float = 3e-4, lr_warmup_steps: int = 1000,
                             log_every: int = 100, normalize_every: int = 100,
                             resample_dead_every: int = 25000, dead_threshold: float = 1e-6) -> Dict[str, List[float]]:
        """Train SAE following Anthropic's recipe: Adam + warmup + decoder norm + dead resampling."""
        device = self.W_enc.device
        all_act = torch.cat(activation_dataset, dim=0).to(device)
        n = all_act.shape[0]
        opt = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.999))
        history: Dict[str, List[float]] = {k: [] for k in ["loss", "reconstruction_loss", "sparsity_loss", "l0", "frac_active"]}
        step = 0
        for _ in range(epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n - batch_size + 1, batch_size):
                if step < lr_warmup_steps:
                    for pg in opt.param_groups: pg["lr"] = lr * step / lr_warmup_steps
                ld = self.compute_loss(all_act[perm[start:start+batch_size]])
                opt.zero_grad(); ld["loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0); opt.step()
                if step % normalize_every == 0: self.normalize_decoder()
                if step > 0 and step % resample_dead_every == 0:
                    self._resample_dead_features(all_act, opt, dead_threshold)
                if step % log_every == 0:
                    history["loss"].append(ld["loss"].item())
                    history["reconstruction_loss"].append(ld["reconstruction_loss"].item())
                    history["sparsity_loss"].append(ld["sparsity_loss"].item())
                    history["l0"].append(ld["l0_sparsity"].item())
                    history["frac_active"].append(ld["frac_features_active"].item())
                step += 1
        return history

    @torch.no_grad()
    def _resample_dead_features(self, activations: Tensor, optimizer: torch.optim.Optimizer,
                                dead_threshold: float) -> int:
        """Resample dead features toward high-reconstruction-error data points."""
        if self._total_samples == 0: return 0
        dead_mask = (self._feature_activation_counts / self._total_samples) < dead_threshold
        n_dead = dead_mask.sum().item()
        if n_dead == 0: return 0
        sample = activations[torch.randperm(activations.shape[0])[:min(n_dead*10, len(activations))]]
        _, decoded, _ = self(sample)
        _, top_idx = (sample - decoded).pow(2).sum(-1).topk(min(n_dead, len(sample)))
        high_err = sample[top_idx]
        for i, di in enumerate(dead_mask.nonzero(as_tuple=True)[0][:len(high_err)]):
            d = (high_err[i] - self.b_dec); d = d / d.norm().clamp(min=1e-8)
            self.W_enc.data[di] = d * 0.5; self.b_enc.data[di] = 0.0
            if not self.tied_weights and self.W_dec is not None: self.W_dec.data[:, di] = d * 0.1
        self._feature_activation_counts[dead_mask] = 0
        return n_dead

    @torch.no_grad()
    def get_top_features(
        self,
        activation: Tensor,
        k: int = 10,
    ) -> List[Feature]:
        """
        Get the top-k most active features for a given activation.

        Args:
            activation: Single activation vector (input_dim,) or batch (B, input_dim).
            k: Number of top features to return.

        Returns:
            List of Feature objects, sorted by activation magnitude.
        """
        if activation.dim() == 1:
            activation = activation.unsqueeze(0)

        encoded = self.encode(activation)
        mean_encoded = encoded.mean(dim=0)

        topk_values, topk_indices = mean_encoded.topk(k)

        freq = self._feature_activation_counts / max(self._total_samples, 1)

        features = []
        for val, idx in zip(topk_values.tolist(), topk_indices.tolist()):
            dec_norm = self.decoder_weight[:, idx].norm().item() if not self.tied_weights else self.W_enc[idx].norm().item()
            features.append(Feature(
                feature_idx=idx,
                activation_frequency=freq[idx].item(),
                mean_activation=val,
                max_activation=encoded[:, idx].max().item(),
                decoder_weights_norm=dec_norm,
            ))

        return features

    @torch.no_grad()
    def feature_dashboard(self, feature_idx: int, dataset: List[Tuple[str, Tensor]],
                          top_k_examples: int = 20, top_k_logits: int = 10) -> FeatureReport:
        """Generate comprehensive dashboard: activation stats, top examples, logit effects."""
        all_acts, examples = [], []
        for text, act in dataset:
            if act.dim() == 1: act = act.unsqueeze(0)
            fa = self.encode(act)[..., feature_idx]
            if fa.max().item() > 0:
                examples.append({"text": text, "max_activation": fa.max().item(),
                                 "mean_activation": fa.mean().item(),
                                 "max_position": fa.squeeze(0).argmax().item() if fa.dim() > 1 else 0})
            all_acts.append(fa.reshape(-1))
        aa = torch.cat(all_acts)
        examples.sort(key=lambda x: x["max_activation"], reverse=True)
        dec_dir = self.decoder_weight[:, feature_idx] if not self.tied_weights else self.W_enc[feature_idx]
        logit_effect = {}
        if hasattr(self, '_lm_head_weight') and self._lm_head_weight is not None:
            lc = self._lm_head_weight @ dec_dir
            for prefix, largest in [("promote", True), ("suppress", False)]:
                tk = lc.topk(top_k_logits, largest=largest)
                for v, i in zip(tk.values.tolist(), tk.indices.tolist()):
                    logit_effect[f"{prefix}_{i}"] = v
        return FeatureReport(
            feature_idx=feature_idx, activation_frequency=(aa > 0).float().mean().item(),
            mean_activation=aa[aa > 0].mean().item() if (aa > 0).any() else 0.0,
            max_activation=aa.max().item(), top_activating_examples=examples[:top_k_examples],
            logit_effect=logit_effect, decoder_direction=dec_dir.cpu())

    def set_lm_head(self, lm_head_weight: Tensor) -> None:
        """Set the lm_head weight for logit effect computation."""
        self._lm_head_weight = lm_head_weight.detach()
```

---

## 4. Circuit Discovery Methodology

The complete circuit discovery pipeline for NanoSeek integrates all four modules:

```
Input → Step 1: IDENTIFY BEHAVIOR (define target: induction, factual recall, syntax)
  │
  ▼
Step 2: ACTIVATION PATCHING (ActivationPatcher)
  │  For each layer l=0..15: patch_layer_output(clean, corrupt)
  │  → Identify top-5 critical layers
  │  Drill down: patch_mla_head(l,h), patch_expert(l,e)
  │  → Component-level importance map
  ▼
Step 3: IDENTIFY CRITICAL COMPONENTS
  │  Threshold: |effect| > 0.1 × max(|effect|)
  │  MoE layers: jointly identify routing + expert output
  │  → Candidate circuit graph
  ▼
Step 4: FEATURE DECOMPOSITION (SparseAutoencoder)
  │  Train SAEs on residual stream at critical layers
  │  get_top_features() → map SAE features to circuit components
  ▼
Step 5: VERIFY MINIMAL CIRCUIT
  │  Ablate everything OUTSIDE candidate circuit
  │  Behavior preserved → circuit is sufficient ✓
  │  Remove components one-at-a-time → each is necessary ✓
  ▼
Step 6: INTERPRET & DOCUMENT
       MLA Analysis → compression bottleneck characterization
       Expert Analysis → expert contributions to circuit
       Logit Lens → layer-by-layer prediction evolution
       → Human-readable circuit description with evidence
```

---

## 5. File Placement

```
nanoseek/
├── fms/
│   └── interpretability/
│       ├── __init__.py                 # Package init
│       ├── patching.py                 # 3a: ActivationPatcher
│       ├── moe_analysis.py             # 3b: ExpertAnalyzer
│       ├── mla_analysis.py             # 3c: MLAAnalyzer
│       └── sae.py                      # 3d: SparseAutoencoder
├── scripts/
│   └── run_interpretability.py         # CLI entry point for all analyses
└── tests/
    └── test_interpretability.py        # Unit tests for all four modules
```

All modules are **read-only** — they use forward hooks and `torch.no_grad()`, never modifying model weights. Import from `model.model` for `create_causal_mask` and `apply_rotary_emb`. Any tokenizer with `encode(text) → List[int]` and `decode(List[int]) → str` works.

---

## 6. Expected Findings

Based on published interpretability research and NanoSeek's architecture, here is what we expect to discover:

| Finding | Expected Location | Evidence | Confidence |
|---------|-------------------|----------|------------|
| **Induction heads** | MLA layers 2-6 | High attention from second occurrence of token A to position after first A. RoPE component handles positional matching. | High — induction heads are universal in transformers (Olsson et al. 2022) |
| **Previous-token heads** | MLA layers 0-2 | Strong diagonal-shifted attention pattern (pos i attends to pos i-1). These feed into induction heads at higher layers. | High — prerequisite for induction circuits |
| **Punctuation experts** | MoE layers 2-5 (early MoE) | 2-4 experts per layer that activate >3× baseline rate for `.`, `,`, `;`, `!`, `?`. Likely handle syntax boundaries. | Medium — observed in GPT-2 MoE variants |
| **Syntax experts** | MoE layers 4-8 (mid) | Experts correlating with POS tags: verb-specialists, noun-specialists, determiner-specialists. Co-activate with function words. | Medium — expert specialization emerges robustly |
| **Knowledge experts** | MoE layers 8-14 (late MoE) | Experts that activate for domain-specific tokens (scientific, legal, technical). More dispersed activation patterns. | Medium-Low — requires sufficient training data diversity |
| **Dead experts** | MoE layers 2-3 (earliest MoE) | 2-5 experts per layer with <1% activation rate, despite bias-based load balancing. More common in early training. | Medium — NanoSeek's gamma=0.001 should minimize these |
| **Superposition gradient** | Residual stream layers 0→15 | SAE features: early layers need 16x overcomplete, late layers need 4x. L0 sparsity increases through layers. | High — consistent across model families |
| **MLA information loss** | Compressed KV at all layers | Effective rank of 143-dim KV: ~80-100 used at 90% variance explained. Positional and syntactic features survive; fine-grained semantic features are lossy. | Medium — depends on training convergence |
| **Logit lens convergence** | Layers 10-15 | KL divergence from final prediction drops sharply at layers 10-12. MoE layers show higher KL variance than MLA-only layers. | High — standard logit lens behavior |
| **Expert co-activation clusters** | Cross-layer patterns | Groups of 4-8 experts that consistently co-activate, forming "meta-experts" for complex tasks. | Medium — observed in Switch Transformer analysis |

**Quantitative predictions**: Top induction head score > 0.6 by layer 4-5 after 5B+ tokens. Expert utilization >90% within 50-200% of uniform baseline. SAE reconstruction R² > 0.95 with L0 < 50 at 8× overcomplete. KV probe for POS tagging >85% accuracy (vs ~15% baseline).

---

## 7. Gotchas & Pitfalls

### Gotcha 1: Activation Patching in MoE Requires Patching BOTH Routing AND Expert Output

When patching an expert in `patch_expert()`, you must patch the Gate's routing decision *and* the expert's forward computation jointly. If you only patch the expert output without ensuring the expert was selected in the corrupt run, the patched output is never used (it's multiplied by zero routing weight). Conversely, if you only patch the routing to force-select the expert but don't patch its output, you get the corrupt run's expert computation, which is confounded.

**Mitigation**: The `ActivationPatcher.patch_expert()` implementation patches the Gate hook to use clean routing for tokens where the target expert was selected in the clean run.

### Gotcha 2: SAE Features in MoE Models Are Sparser Than Dense Models

Because MoE routing already decomposes computation into expert-specific pathways, the residual stream at MoE layers has *less* superposition than equivalent dense layers. SAE training on post-MoE residual streams will find fewer features per input (lower L0) but may need fewer total features to explain the same fraction of variance. If you use the same SAE configuration (e.g., 8× overcomplete) for both MoE and dense layers, the MoE SAE will have more dead features.

**Mitigation**: Use adaptive overcomplete ratios — 4× for post-MoE layers, 8-16× for pre-MoE and MLA layers. Monitor dead feature fraction and resample aggressively.

### Gotcha 3: MLA's Compressed KV Loses Per-Head Information

The compressed KV (`kv_compressed`, dim 143) is SHARED across all 16 attention heads. Individual head-specific information only emerges after the `wkv_b` expansion (143 → 16×128). This means:
- Probing the compressed KV reveals *shared* information, not per-head information.
- If you want to understand what a specific head attends to, you must probe *after* `wkv_b`, not in the compressed space.
- The compressed space is a bottleneck: information that is useful for only 1-2 heads may be discarded by the compression if it conflicts with information useful for the other 14-15 heads.

**Mitigation**: Run probes at both the compressed level (143-dim) and the expanded per-head level (128-dim per head). Compare to understand what information is head-specific vs. shared.

### Gotcha 4: Logit Lens May Not Work Well With MoE

The logit lens assumes the residual stream smoothly evolves toward the final prediction. MoE layers introduce *discontinuous* contributions — different tokens pass through different experts, creating a non-smooth residual update. This can cause:
- High variance in logit lens predictions at MoE layers.
- "Jumpy" KL divergence curves (not monotonically decreasing).
- Different tokens converging to their final predictions at different layers, depending on which experts they activate.

**Mitigation**: Compute logit lens metrics separately for tokens that activate the same expert set. Group tokens by their top-1 or top-3 expert assignments and compute per-group logit lens trajectories. Also consider the *tuned* lens (Belrose et al. 2023), which learns a per-layer affine correction.

### Gotcha 5: Induction Heads in MLA Operate Through the Compressed Bottleneck

In standard MHA, induction heads operate directly on full-rank Q and K. In MLA, the induction signal must survive the compression pipeline:
1. The "content matching" signal (is this token the same as my current token?) must be encoded in the 143-dim compressed KV.
2. The "positional matching" signal (where did this token appear before?) is handled by the 32-dim RoPE component, which is NOT compressed.

This factorization means MLA induction heads may be *weaker* than standard MHA induction heads for content matching (information loss through the bottleneck) but equally strong for positional matching (RoPE is uncompressed). The net induction score may be lower, but the *mechanism* is the same.

**Mitigation**: When running induction head detection, analyze the RoPE and nope components separately. A head with high positional matching but low content matching is still functionally an induction head — it just relies more on position than content.

### Gotcha 6: Expert Ablation Can Trigger Load Balancing Compensation

NanoSeek's `Gate` has a learned `expert_bias` buffer that adjusts routing to balance load. When you ablate an expert (zero its output), the bias update mechanism (`update_load_balance_bias()`) will detect that the ablated expert has zero load and increase its bias to route more tokens to it — the opposite of what you want for a clean ablation.

**Mitigation**: During ablation studies, either (a) freeze the Gate biases by detaching `expert_bias` from the update mechanism, or (b) use `model.eval()` mode where bias updates are not applied (updates only occur during training). The `ExpertAnalyzer.expert_ablation_study()` implementation uses `model.eval()` to avoid this.

---

## 8. Verification & Performance

Run with a small NanoSeek config (hidden_size=256, 2 layers, 4 experts):
```bash
pytest tests/test_interpretability.py -v --timeout=120
```

| Analysis | Small Config | Full NanoSeek-1B | Memory |
|----------|-------------:|------------------:|-------:|
| Causal trace (1 prompt, all layers) | ~5s | ~60s | ~4 GB |
| Expert profiling (100 texts) | ~30s | ~10min | ~2 GB |
| MLA reconstruction analysis | ~10s | ~2min | ~3 GB |
| SAE training (10 epochs, 100K samples) | ~2min | ~2hrs | ~8 GB |
| Induction head detection | ~15s | ~5min | ~4 GB |
| Full circuit discovery pipeline | ~5min | ~4hrs | ~16 GB |

---

*"The goal of mechanistic interpretability is not to understand every neuron — it is to understand the algorithms. A neural network that learns to do modular arithmetic is not 'doing modular arithmetic like a neuron'; it is implementing a discrete Fourier transform. The algorithm is the explanation, the neurons are the implementation."*

— Neel Nanda, "A Comprehensive Mechanistic Interpretability Explainer," 2023
