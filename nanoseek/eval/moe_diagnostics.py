"""
Enhanced MoE diagnostics beyond base_eval.py's load_entropy/load_cv.

Adds:
- Dead expert detection (experts with <1% traffic)
- Expert utilization Gini coefficient
- Per-layer expert load heatmap data
- MTP acceptance rate measurement

These metrics are critical for detecting expert collapse during training
and verifying MoE health during RL post-training (RULE 7, RULE 9).
"""

import math
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@torch.no_grad()
def compute_dead_experts(
    model: nn.Module,
    dataloader,
    device: torch.device,
    n_tokens: int = 50000,
    dead_threshold: float = 0.01,
) -> dict:
    """
    Detect dead experts (experts receiving < threshold fraction of tokens).

    Args:
        model: NanoSeekModel in eval mode.
        dataloader: Yields (input_ids, targets) batches.
        device: Compute device.
        n_tokens: Tokens to process for statistics.
        dead_threshold: Fraction below which an expert is considered dead.

    Returns:
        Dict with dead_count, dead_experts (list of indices), utilization stats.
    """
    model.eval()

    # Collect load stats across multiple batches
    all_layer_counts = {}  # layer_idx -> accumulated counts
    tokens_processed = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        model(batch_x)
        tokens_processed += batch_x.numel()

        # Get per-layer load stats from model
        load_stats = model.get_expert_load_stats()
        per_layer = load_stats.get("per_layer", {})

        for layer_idx, layer_data in per_layer.items():
            counts = layer_data.get("load_counts", None)
            if counts is not None:
                if layer_idx not in all_layer_counts:
                    all_layer_counts[layer_idx] = counts.clone().float()
                else:
                    all_layer_counts[layer_idx] += counts.float()

        if tokens_processed >= n_tokens:
            break

    # If model doesn't expose per-layer data, use aggregate
    if not all_layer_counts:
        aggregate = model.get_expert_load_stats()
        counts = aggregate.get("load_per_expert", None)
        if counts is not None:
            all_layer_counts[0] = counts.float()

    results = {
        'dead_experts_per_layer': {},
        'total_dead_count': 0,
        'utilization_gini_per_layer': {},
        'tokens_processed': tokens_processed,
    }

    for layer_idx, counts in all_layer_counts.items():
        total = counts.sum().item()
        if total == 0:
            continue

        fractions = counts / total
        n_experts = len(counts)

        # Dead experts: fraction < threshold
        dead_mask = fractions < dead_threshold
        dead_indices = torch.where(dead_mask)[0].tolist()
        results['dead_experts_per_layer'][layer_idx] = {
            'dead_indices': dead_indices,
            'dead_count': len(dead_indices),
            'fractions': fractions.tolist(),
        }
        results['total_dead_count'] += len(dead_indices)

        # Gini coefficient of expert utilization
        gini = _gini_coefficient(fractions.numpy())
        results['utilization_gini_per_layer'][layer_idx] = round(gini, 4)

    if results['total_dead_count'] > 0:
        logger.warning(
            f"Found {results['total_dead_count']} dead experts "
            f"(< {dead_threshold:.0%} traffic) across {len(all_layer_counts)} layers"
        )
    else:
        logger.info("No dead experts detected")

    return results


def _gini_coefficient(values) -> float:
    """Compute Gini coefficient (0 = perfect equality, 1 = maximum inequality)."""
    import numpy as np
    values = np.array(values, dtype=float)
    if values.sum() == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_vals) / (n * np.sum(sorted_vals))) - (n + 1) / n)


@torch.no_grad()
def compute_mtp_acceptance_rate(
    model: nn.Module,
    dataloader,
    device: torch.device,
    n_tokens: int = 10000,
) -> dict[str, float]:
    """
    Measure Multi-Token Prediction acceptance rate.

    Compares MTP head predictions against actual next tokens.
    This is a test-time scaling signal (RULE 9): higher acceptance rate
    means speculative decoding will be more effective.

    Args:
        model: NanoSeekModel with MTP head.
        dataloader: Yields (input_ids, targets) batches.
        device: Compute device.
        n_tokens: Tokens to evaluate.

    Returns:
        Dict with overall acceptance rate and per-position rates.
    """
    model.eval()

    total_correct = 0
    total_predictions = 0
    per_position_correct = {}
    per_position_total = {}
    tokens_processed = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_x)
        if not isinstance(outputs, dict):
            logger.warning("Model does not return dict — cannot compute MTP acceptance")
            return {'acceptance_rate': 0.0, 'note': 'model does not return dict outputs'}

        # Check if model has MTP logits
        mtp_logits = outputs.get('mtp_logits', None)
        if mtp_logits is None:
            # Try to get MTP predictions from auxiliary outputs
            logger.info("No mtp_logits in outputs — MTP may not be enabled")
            return {'acceptance_rate': 0.0, 'note': 'no mtp_logits in model output'}

        # mtp_logits shape: [n_predict, B, S, V] or [B, S, V] for single prediction
        if mtp_logits.dim() == 4:
            n_predict = mtp_logits.shape[0]
            for pos in range(n_predict):
                pos_logits = mtp_logits[pos]  # [B, S, V]
                pos_preds = pos_logits.argmax(dim=-1)  # [B, S]

                # Target for position k is tokens shifted by k+1
                if pos + 1 < batch_y.shape[1]:
                    pos_targets = batch_y[:, pos + 1:]  # shifted targets
                    min_len = min(pos_preds.shape[1], pos_targets.shape[1])
                    correct = (pos_preds[:, :min_len] == pos_targets[:, :min_len]).sum().item()
                    total = min_len * batch_y.shape[0]

                    per_position_correct[pos] = per_position_correct.get(pos, 0) + correct
                    per_position_total[pos] = per_position_total.get(pos, 0) + total
                    total_correct += correct
                    total_predictions += total
        elif mtp_logits.dim() == 3:
            # Single MTP prediction
            preds = mtp_logits.argmax(dim=-1)
            if batch_y.shape[1] > 1:
                targets = batch_y[:, 1:]
                min_len = min(preds.shape[1], targets.shape[1])
                correct = (preds[:, :min_len] == targets[:, :min_len]).sum().item()
                total = min_len * batch_y.shape[0]
                total_correct += correct
                total_predictions += total
                per_position_correct[0] = per_position_correct.get(0, 0) + correct
                per_position_total[0] = per_position_total.get(0, 0) + total

        tokens_processed += batch_x.numel()
        if tokens_processed >= n_tokens:
            break

    overall_rate = total_correct / max(total_predictions, 1)
    per_position_rates = {
        pos: per_position_correct[pos] / max(per_position_total[pos], 1)
        for pos in sorted(per_position_correct.keys())
    }

    results = {
        'acceptance_rate': round(overall_rate, 4),
        'per_position_rates': {str(k): round(v, 4) for k, v in per_position_rates.items()},
        'total_predictions': total_predictions,
        'tokens_processed': tokens_processed,
    }

    logger.info(f"MTP acceptance rate: {overall_rate:.2%} (target: >75% by end of training)")
    return results


def get_expert_load_heatmap(model: nn.Module) -> dict:
    """
    Extract per-layer expert load data for heatmap visualization.

    Returns dict suitable for logging as a wandb table or matplotlib heatmap.
    """
    load_stats = model.get_expert_load_stats()
    per_layer = load_stats.get("per_layer", {})

    heatmap_data = {}
    for layer_idx, layer_data in per_layer.items():
        counts = layer_data.get("load_counts", None)
        if counts is not None:
            total = counts.sum().item()
            if total > 0:
                heatmap_data[str(layer_idx)] = (counts / total).tolist()

    # Fallback to aggregate if per-layer not available
    if not heatmap_data:
        counts = load_stats.get("load_per_expert", None)
        if counts is not None and counts.sum() > 0:
            heatmap_data["aggregate"] = (counts / counts.sum()).tolist()

    return heatmap_data
