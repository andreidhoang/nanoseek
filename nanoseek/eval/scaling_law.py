"""
Chinchilla scaling law fitting with LOOCV validation.

Fits L(N) = E + A * N^(-alpha) from training run data points.

Critical insight: With 3 data points and 3 parameters, R² = 1.0 trivially
(exactly determined system). Use LOOCV instead: fix E to literature prior,
fit on 2 points, predict the held-out point, measure relative error.

Reference: Hoffmann et al., "Training Compute-Optimal Large Language Models" (2022)
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


@dataclass
class ScalingLawFit:
    """Results from scaling law fitting."""
    E: float          # Irreducible entropy (nats)
    A: float          # Scale coefficient
    alpha: float      # Scaling exponent
    loocv_errors: list[float]  # Relative errors from leave-one-out
    mean_loocv_error: float    # Mean LOOCV relative error
    data_points: list[tuple[int, float]]  # (N, loss) pairs used
    r_squared: float  # R² (note: trivially 1.0 with 3 points)


def _chinchilla_loss(N, E, A, alpha):
    """Chinchilla scaling law: L(N) = E + A * N^(-alpha)."""
    return E + A * np.power(N, -alpha)


def fit_scaling_law(
    data_points: list[tuple[int, float]],
    E_prior: float = 1.69,
    bounds: tuple = ((1.0, 0.1, 0.01), (3.0, 100.0, 1.0)),
) -> ScalingLawFit:
    """
    Fit Chinchilla scaling law L(N) = E + A * N^(-alpha).

    Args:
        data_points: List of (param_count, final_ema_val_bpb) tuples.
        E_prior: Literature prior for irreducible entropy (Chinchilla: ~1.69).
        bounds: Parameter bounds ((E_min, A_min, alpha_min), (E_max, A_max, alpha_max)).

    Returns:
        ScalingLawFit with fitted parameters and LOOCV errors.
    """
    N_vals = np.array([p[0] for p in data_points], dtype=float)
    L_vals = np.array([p[1] for p in data_points], dtype=float)

    # Full fit (all data points)
    try:
        popt, pcov = curve_fit(
            _chinchilla_loss, N_vals, L_vals,
            p0=[E_prior, 5.0, 0.3],
            bounds=bounds,
            maxfev=10000,
        )
        E, A, alpha = popt
    except RuntimeError as e:
        logger.warning(f"Full fit failed: {e}. Using prior E={E_prior}")
        E, A, alpha = E_prior, 5.0, 0.3

    # R² (informational only — trivially 1.0 with 3 points, 3 params)
    L_pred = _chinchilla_loss(N_vals, E, A, alpha)
    ss_res = np.sum((L_vals - L_pred) ** 2)
    ss_tot = np.sum((L_vals - np.mean(L_vals)) ** 2)
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-10)

    # LOOCV with fixed E prior
    # Fix E to literature value, fit only A and alpha on n-1 points
    loocv_errors = []

    for i in range(len(data_points)):
        # Leave one out
        train_N = np.delete(N_vals, i)
        train_L = np.delete(L_vals, i)
        test_N = N_vals[i]
        test_L = L_vals[i]

        if len(train_N) < 2:
            loocv_errors.append(float('inf'))
            continue

        # Fit with fixed E
        def _fixed_E_loss(N, A_fit, alpha_fit):
            return E_prior + A_fit * np.power(N, -alpha_fit)

        try:
            popt_loo, _ = curve_fit(
                _fixed_E_loss, train_N, train_L,
                p0=[5.0, 0.3],
                bounds=([0.1, 0.01], [100.0, 1.0]),
                maxfev=10000,
            )
            A_loo, alpha_loo = popt_loo
            pred_L = E_prior + A_loo * np.power(test_N, -alpha_loo)
            rel_error = abs(pred_L - test_L) / max(abs(test_L), 1e-10)
            loocv_errors.append(float(rel_error))
        except RuntimeError:
            loocv_errors.append(float('inf'))

    mean_loocv = np.mean([e for e in loocv_errors if np.isfinite(e)]) if loocv_errors else float('inf')

    result = ScalingLawFit(
        E=float(E),
        A=float(A),
        alpha=float(alpha),
        loocv_errors=loocv_errors,
        mean_loocv_error=float(mean_loocv),
        data_points=data_points,
        r_squared=float(r_squared),
    )

    # Interpretation
    logger.info(f"Scaling law fit: L(N) = {E:.4f} + {A:.4f} * N^(-{alpha:.4f})")
    logger.info(f"R² = {r_squared:.6f} (NOTE: trivially 1.0 with {len(data_points)} points, 3 params)")
    logger.info(f"LOOCV mean relative error: {mean_loocv:.4f} (healthy: < 0.15)")

    if mean_loocv > 0.15:
        logger.warning(
            "LOOCV error > 15%: scaling law may not extrapolate well. "
            "Consider: (1) more data points, (2) different E prior, (3) training longer"
        )

    return result


def predict_loss(fit: ScalingLawFit, N: int) -> float:
    """Predict loss for a model with N active parameters."""
    return fit.E + fit.A * N ** (-fit.alpha)


def compute_optimal_tokens(fit: ScalingLawFit, N: int, budget_flops: float) -> float:
    """
    Compute optimal number of training tokens given a FLOPs budget.

    Uses Chinchilla's optimal compute allocation: D* = budget / (6 * N).
    """
    return budget_flops / (6 * N)


def main():
    """CLI entry point for scaling law fitting."""
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Fit Chinchilla scaling law from training runs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--runs', nargs='+', required=True,
        help='Run data as name:params:loss (e.g., anchor:55e6:2.31 500m:500e6:1.85)',
    )
    parser.add_argument('--E-prior', type=float, default=1.69, help='Literature prior for E')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    parser.add_argument(
        '--predict', type=float, nargs='*', default=[],
        help='Predict loss for these param counts (e.g., 3e9 7e9)',
    )

    args = parser.parse_args()

    # Parse run data
    data_points = []
    for run in args.runs:
        parts = run.split(':')
        if len(parts) != 3:
            parser.error(f"Invalid run format: {run}. Use name:params:loss")
        name, params, loss = parts
        data_points.append((float(params), float(loss)))
        logger.info(f"Run '{name}': N={float(params):.0f}, L={float(loss):.4f}")

    # Fit
    result = fit_scaling_law(data_points, E_prior=args.E_prior)

    # Predictions
    predictions = {}
    for N in args.predict:
        pred = predict_loss(result, N)
        predictions[f"{N:.0e}"] = pred
        logger.info(f"Predicted L({N:.0e}) = {pred:.4f}")

    # Output
    output = {
        'fit': {
            'E': result.E,
            'A': result.A,
            'alpha': result.alpha,
        },
        'validation': {
            'r_squared': result.r_squared,
            'r_squared_note': 'trivially 1.0 with 3 points and 3 params',
            'loocv_errors': result.loocv_errors,
            'mean_loocv_error': result.mean_loocv_error,
            'loocv_healthy': result.mean_loocv_error < 0.15,
        },
        'data_points': [{'N': int(n), 'L': l} for n, l in result.data_points],
        'predictions': predictions,
    }

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results written to {args.output}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
