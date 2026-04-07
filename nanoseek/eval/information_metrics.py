"""
Information-theoretic metrics for MoE expert specialization.

Two I_spec variants:
  1. Labeled-domain I_spec (PRIMARY — masterplan Workstream 1):
     I_spec = I(E; D) where D is one of 5 labeled domains (code/math/science/web/books).
     Computed from the joint distribution P(expert_id, domain_label).
     This is the confirmatory metric for the specialization study.

  2. Cluster-based I_spec (SECONDARY):
     I_spec = H(expert) - H(expert | token_cluster)
     Uses k-means clustering on router inputs. Reported as a secondary metric.

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
        raise ImportError(
            "I_spec computation requires scikit-learn: pip install scikit-learn\n"
            "Without sklearn, I_spec returns 0.0 which is scientifically useless.\n"
            "Install with: pip install 'nanoseek[training]'"
        )

    model.eval()
    collector = GateActivationCollector()
    collector.register_hooks(model)

    # Collect activations (try/finally ensures hooks are removed even on OOM/NaN)
    try:
        tokens_collected = 0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            model(batch_x)
            tokens_collected += batch_x.numel()
            if tokens_collected >= n_tokens:
                break
    finally:
        collector.remove_hooks()

    if not collector.router_inputs:
        logger.warning("No gate activations collected — model may not have MoE layers")
        return {'i_spec_mean': 0.0, 'i_spec_per_layer': [], 'h_expert': 0.0}

    # Compute I_spec per layer
    n_layers = len(collector.router_inputs)
    i_spec_per_layer = []
    h_expert_per_layer = []
    h_expert_given_cluster_per_layer = []

    for layer_idx in range(n_layers):
        router_input = collector.router_inputs[layer_idx].float().numpy()  # [N, D]
        expert_idx = collector.expert_indices[layer_idx].int().numpy()    # [N, K]

        N = router_input.shape[0]
        if N < n_clusters:
            i_spec_per_layer.append(0.0)
            h_expert_per_layer.append(0.0)
            h_expert_given_cluster_per_layer.append(0.0)
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
        h_expert_layer = float(-np.sum(nonzero * np.log(nonzero + 1e-10)))

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

        i_spec = h_expert_layer - h_expert_given_cluster
        i_spec_per_layer.append(round(float(i_spec), 4))
        h_expert_per_layer.append(round(h_expert_layer, 4))
        h_expert_given_cluster_per_layer.append(round(float(h_expert_given_cluster), 4))

    results = {
        'i_spec_mean': round(float(np.mean(i_spec_per_layer)), 4) if i_spec_per_layer else 0.0,
        'i_spec_per_layer': i_spec_per_layer,
        'h_expert_per_layer': h_expert_per_layer,
        'h_expert_given_cluster_per_layer': h_expert_given_cluster_per_layer,
        'h_expert_mean': round(float(np.mean(h_expert_per_layer)), 4) if h_expert_per_layer else 0.0,
        'n_tokens_collected': tokens_collected,
        'n_clusters': n_clusters,
        'variant': 'cluster-based',
    }

    logger.info(f"I_spec (cluster) mean: {results['i_spec_mean']:.4f} (healthy: 0.3-0.7)")
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

    # try/finally ensures hooks are removed even on OOM/NaN
    try:
        tokens_collected = 0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            model(batch_x)
            tokens_collected += batch_x.numel()
            if tokens_collected >= n_tokens:
                break
    finally:
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
        'h_expert_mean': round(float(np.mean(h_experts)), 4),
        'i_spec_per_layer': [],
        'n_tokens_collected': tokens_collected,
        'note': 'simplified (no sklearn) — only H(expert) computed',
    }


# ═══════════════════════════════════════════════════════════════════
# Labeled-Domain I_spec — PRIMARY metric (Masterplan Workstream 1)
# ═══════════════════════════════════════════════════════════════════
#
# I_spec = I(E; D) = H(E) - H(E|D)
#
# where D ∈ {code, math, science, web, books} are labeled domains.
#
# This is the CONFIRMATORY metric. Unlike cluster-based I_spec, it
# uses ground-truth domain labels so the mutual information measures
# genuine expert-domain association, not k-means artifacts.
#
# The joint distribution P(expert_id, domain_label) is built by:
#   1. Running domain-labeled tokens through the model
#   2. Recording which experts each token is routed to (from Gate hooks)
#   3. Counting co-occurrences in a [n_experts × n_domains] matrix
#
# Novel: No published work tracks this metric during training as an
# online diagnostic. DeepSeek V3 published post-hoc heatmaps only.
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_i_spec_labeled(
    model: nn.Module,
    domain_texts: dict[str, list[str]],
    tokenizer,
    device: torch.device,
    n_experts: int = 64,
    max_tokens_per_domain: int = 2000,
    max_length: int = 2048,
) -> dict[str, float]:
    """
    Compute labeled-domain I_spec: I(E; D) using ground-truth domain labels.

    This is the PRIMARY I_spec metric (masterplan Workstream 1). It uses
    labeled domain data rather than k-means clustering, producing a more
    interpretable and reproducible mutual information estimate.

    Args:
        model: NanoSeekModel in eval mode.
        domain_texts: Dict mapping domain name -> list of text samples.
                      Expected domains: code, math, science, web, books.
        tokenizer: Tokenizer with encode() method.
        device: Compute device.
        n_experts: Number of routed experts.
        max_tokens_per_domain: Max tokens to process per domain (for balance).
        max_length: Max sequence length per sample.

    Returns:
        Dict with:
          - i_spec_labeled_mean: mean I(E;D) across MoE layers
          - i_spec_labeled_per_layer: list of I(E;D) per layer
          - h_expert_per_layer: H(E) per layer
          - h_expert_given_domain_per_layer: H(E|D) per layer
          - h_domain: H(D) — domain marginal entropy
          - n_domains: number of domains used
          - n_tokens_per_domain: dict of actual tokens processed per domain
    """
    model.eval()
    domains = list(domain_texts.keys())
    n_domains = len(domains)

    if n_domains < 2:
        logger.warning(f"Only {n_domains} domain(s) — need ≥2 for meaningful I_spec")
        return {
            'i_spec_labeled_mean': 0.0,
            'i_spec_labeled_per_layer': [],
            'h_expert_per_layer': [],
            'h_expert_given_domain_per_layer': [],
            'h_domain': 0.0,
            'n_domains': n_domains,
            'n_tokens_per_domain': {},
            'variant': 'labeled-domain',
        }

    # Collect expert indices per domain, per layer
    # Structure: domain_expert_counts[layer_idx][domain_idx] = np.array([n_experts])
    # We'll build the joint distribution P(E, D) per layer.

    collector = GateActivationCollector()
    collector.register_hooks(model)

    # Per-domain token counts and per-layer joint counts
    # We process one domain at a time to associate expert assignments with domains
    tokens_per_domain = {}
    # Accumulate: layer_joint[layer_idx] = np.array([n_experts, n_domains])
    layer_joint = {}

    try:
        for domain_idx, domain_name in enumerate(domains):
            texts = domain_texts[domain_name]
            domain_tokens_collected = 0

            for text in texts:
                if domain_tokens_collected >= max_tokens_per_domain:
                    break

                tokens = tokenizer.encode(text)
                if len(tokens) < 2:
                    continue
                if len(tokens) > max_length:
                    tokens = tokens[:max_length]

                # Clear collector for this forward pass
                collector.clear()

                input_ids = torch.tensor([tokens], device=device)
                model(input_ids)

                seq_tokens = input_ids.numel()
                domain_tokens_collected += seq_tokens

                # Record expert selections per layer for this domain
                for layer_idx, expert_idx_tensor in enumerate(collector.expert_indices):
                    expert_idx = expert_idx_tensor.int().numpy()  # [N, K]

                    if layer_idx not in layer_joint:
                        layer_joint[layer_idx] = np.zeros((n_experts, n_domains))

                    # Count expert assignments for this domain
                    for row in expert_idx:
                        for e in row:
                            if 0 <= e < n_experts:
                                layer_joint[layer_idx][e, domain_idx] += 1

            tokens_per_domain[domain_name] = domain_tokens_collected
            logger.debug(f"Domain '{domain_name}': {domain_tokens_collected} tokens")

    finally:
        collector.remove_hooks()

    if not layer_joint:
        logger.warning("No gate activations collected — model may not have MoE layers")
        return {
            'i_spec_labeled_mean': 0.0,
            'i_spec_labeled_per_layer': [],
            'h_expert_per_layer': [],
            'h_expert_given_domain_per_layer': [],
            'h_domain': 0.0,
            'n_domains': n_domains,
            'n_tokens_per_domain': tokens_per_domain,
            'variant': 'labeled-domain',
        }

    # Compute I(E; D) = H(E) - H(E|D) per layer
    i_spec_per_layer = []
    h_expert_per_layer = []
    h_e_given_d_per_layer = []

    for layer_idx in sorted(layer_joint.keys()):
        joint = layer_joint[layer_idx]  # [n_experts, n_domains]
        total = joint.sum()
        if total == 0:
            i_spec_per_layer.append(0.0)
            h_expert_per_layer.append(0.0)
            h_e_given_d_per_layer.append(0.0)
            continue

        # P(E, D) joint distribution
        p_joint = joint / total

        # P(E) marginal: sum over domains
        p_expert = p_joint.sum(axis=1)  # [n_experts]
        nonzero_e = p_expert[p_expert > 0]
        h_expert = float(-np.sum(nonzero_e * np.log(nonzero_e + 1e-10)))

        # P(D) marginal: sum over experts
        p_domain = p_joint.sum(axis=0)  # [n_domains]

        # H(E|D) = sum_d P(D=d) * H(E|D=d)
        h_e_given_d = 0.0
        for d_idx in range(n_domains):
            p_d = p_domain[d_idx]
            if p_d < 1e-10:
                continue
            # P(E|D=d) = P(E,D=d) / P(D=d)
            p_e_given_d = joint[:, d_idx] / max(joint[:, d_idx].sum(), 1)
            nz = p_e_given_d[p_e_given_d > 0]
            h_e_given_d += p_d * float(-np.sum(nz * np.log(nz + 1e-10)))

        i_spec = h_expert - h_e_given_d
        i_spec_per_layer.append(round(i_spec, 4))
        h_expert_per_layer.append(round(h_expert, 4))
        h_e_given_d_per_layer.append(round(h_e_given_d, 4))

    # H(D) — domain marginal entropy (for reference)
    total_all = sum(layer_joint[k].sum() for k in layer_joint)
    if total_all > 0:
        # Use the first layer's domain marginal (same across layers if balanced)
        first_layer = sorted(layer_joint.keys())[0]
        p_d_global = layer_joint[first_layer].sum(axis=0)
        p_d_global = p_d_global / max(p_d_global.sum(), 1)
        nz_d = p_d_global[p_d_global > 0]
        h_domain = float(-np.sum(nz_d * np.log(nz_d + 1e-10)))
    else:
        h_domain = 0.0

    results = {
        'i_spec_labeled_mean': round(float(np.mean(i_spec_per_layer)), 4) if i_spec_per_layer else 0.0,
        'i_spec_labeled_per_layer': i_spec_per_layer,
        'h_expert_per_layer': h_expert_per_layer,
        'h_expert_given_domain_per_layer': h_e_given_d_per_layer,
        'h_domain': round(h_domain, 4),
        'n_domains': n_domains,
        'n_tokens_per_domain': tokens_per_domain,
        'variant': 'labeled-domain',
    }

    logger.info(
        f"I_spec (labeled) mean: {results['i_spec_labeled_mean']:.4f} "
        f"across {len(i_spec_per_layer)} layers, {n_domains} domains"
    )
    return results
