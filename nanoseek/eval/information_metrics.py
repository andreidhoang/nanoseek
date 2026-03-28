"""
Information-theoretic metrics for MoE expert specialization.

I_spec (Mutual Information between tokens and experts):
  I_spec = H(expert) - H(expert | token_cluster)

Measures whether experts are learning semantic specializations.
Too low (< 0.3) = experts interchangeable (redundant)
Too high (> 0.7) = experts too specialized (fragile routing)

Reference: Krajewski et al., "Scaling Laws for Fine-Grained Mixture of Experts" (2024)
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class GateActivationCollector:
    """
    Collects gate activations via forward hooks for I_spec computation.

    Registers hooks on Gate modules to capture router inputs and expert selections
    during a forward pass. Removes hooks after collection is complete.
    """

    def __init__(self):
        self.router_inputs: list[torch.Tensor] = []  # [N, D] per layer
        self.expert_indices: list[torch.Tensor] = []  # [N, K] per layer
        self._hooks: list = []

    def register_hooks(self, model: nn.Module):
        """Register forward hooks on all Gate modules."""
        for name, module in model.named_modules():
            if module.__class__.__name__ == 'Gate':
                hook = module.register_forward_hook(self._gate_hook)
                self._hooks.append(hook)
        logger.info(f"Registered {len(self._hooks)} gate hooks for I_spec collection")

    def _gate_hook(self, module, input, output):
        """Capture router input and expert indices from Gate.forward()."""
        # Gate.forward() input: hidden_states [N, D]
        # Gate.forward() output: (weights, indices, aux_loss, metadata)
        if isinstance(input, tuple):
            router_input = input[0]
        else:
            router_input = input

        weights, indices, aux_loss, metadata = output
        self.router_inputs.append(router_input.detach().cpu())
        self.expert_indices.append(indices.detach().cpu())

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def clear(self):
        """Clear collected activations."""
        self.router_inputs.clear()
        self.expert_indices.clear()


@torch.no_grad()
def compute_i_spec(
    model: nn.Module,
    dataloader,
    device: torch.device,
    n_tokens: int = 10000,
    n_clusters: int = 64,
    n_experts: int = 64,
) -> dict[str, float]:
    """
    Compute I_spec: mutual information between token clusters and expert assignments.

    I_spec = H(expert) - H(expert | cluster)

    where clusters are obtained by k-means on router input embeddings.

    Args:
        model: NanoSeekModel in eval mode.
        dataloader: Yields (input_ids, targets) batches.
        device: Compute device.
        n_tokens: Number of tokens to collect activations from.
        n_clusters: Number of k-means clusters for token grouping.
        n_experts: Number of routed experts.

    Returns:
        Dict with 'i_spec_mean', 'i_spec_per_layer', 'h_expert' keys.
    """
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError:
        logger.warning("sklearn not available, using simplified I_spec (no clustering)")
        return _compute_i_spec_simple(model, dataloader, device, n_tokens, n_experts)

    model.eval()
    collector = GateActivationCollector()
    collector.register_hooks(model)

    # Collect activations
    tokens_collected = 0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        model(batch_x)
        tokens_collected += batch_x.numel()
        if tokens_collected >= n_tokens:
            break

    collector.remove_hooks()

    if not collector.router_inputs:
        logger.warning("No gate activations collected — model may not have MoE layers")
        return {'i_spec_mean': 0.0, 'i_spec_per_layer': [], 'h_expert': 0.0}

    # Compute I_spec per layer
    n_layers = len(collector.router_inputs)
    i_spec_per_layer = []
    h_expert = 0.0  # default if no layers (prevents NameError at line 172)

    for layer_idx in range(n_layers):
        router_input = collector.router_inputs[layer_idx].float().numpy()  # [N, D]
        expert_idx = collector.expert_indices[layer_idx].int().numpy()    # [N, K]

        N = router_input.shape[0]
        if N < n_clusters:
            i_spec_per_layer.append(0.0)
            continue

        # K-means clustering on router inputs
        kmeans = MiniBatchKMeans(n_clusters=min(n_clusters, N), random_state=42, n_init=1)
        cluster_labels = kmeans.fit_predict(router_input)

        # H(expert): marginal entropy of expert assignments
        expert_counts = np.zeros(n_experts)
        for row in expert_idx:
            for e in row:
                if 0 <= e < n_experts:
                    expert_counts[e] += 1
        expert_probs = expert_counts / max(expert_counts.sum(), 1)
        nonzero = expert_probs[expert_probs > 0]
        h_expert = -np.sum(nonzero * np.log(nonzero + 1e-10))

        # H(expert | cluster): conditional entropy
        h_expert_given_cluster = 0.0
        for c in range(min(n_clusters, N)):
            mask = cluster_labels == c
            n_c = mask.sum()
            if n_c == 0:
                continue
            cluster_expert_counts = np.zeros(n_experts)
            for row in expert_idx[mask]:
                for e in row:
                    if 0 <= e < n_experts:
                        cluster_expert_counts[e] += 1
            cluster_probs = cluster_expert_counts / max(cluster_expert_counts.sum(), 1)
            nz = cluster_probs[cluster_probs > 0]
            h_c = -np.sum(nz * np.log(nz + 1e-10))
            h_expert_given_cluster += (n_c / N) * h_c

        i_spec = h_expert - h_expert_given_cluster
        i_spec_per_layer.append(round(float(i_spec), 4))

    results = {
        'i_spec_mean': round(float(np.mean(i_spec_per_layer)), 4),
        'i_spec_per_layer': i_spec_per_layer,
        'h_expert': round(float(h_expert), 4),
        'n_tokens_collected': tokens_collected,
        'n_clusters': n_clusters,
    }

    logger.info(f"I_spec mean: {results['i_spec_mean']:.4f} (healthy: 0.3-0.7)")
    return results


def _compute_i_spec_simple(
    model: nn.Module,
    dataloader,
    device: torch.device,
    n_tokens: int,
    n_experts: int,
) -> dict[str, float]:
    """Simplified I_spec without sklearn — uses expert frequency only."""
    model.eval()
    collector = GateActivationCollector()
    collector.register_hooks(model)

    tokens_collected = 0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        model(batch_x)
        tokens_collected += batch_x.numel()
        if tokens_collected >= n_tokens:
            break

    collector.remove_hooks()

    if not collector.expert_indices:
        return {'i_spec_mean': 0.0, 'i_spec_per_layer': [], 'h_expert': 0.0}

    # Just compute H(expert) as a proxy
    h_experts = []
    for expert_idx in collector.expert_indices:
        expert_counts = np.zeros(n_experts)
        for row in expert_idx.numpy():
            for e in row:
                if 0 <= e < n_experts:
                    expert_counts[e] += 1
        expert_probs = expert_counts / max(expert_counts.sum(), 1)
        h = -sum(p * np.log(p + 1e-10) for p in expert_probs if p > 0)
        h_experts.append(float(h))

    return {
        'i_spec_mean': 0.0,  # Cannot compute without clustering
        'h_expert': round(float(np.mean(h_experts)), 4),
        'i_spec_per_layer': [],
        'n_tokens_collected': tokens_collected,
        'note': 'simplified (no sklearn) — only H(expert) computed',
    }
