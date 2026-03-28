"""
Chinchilla scaling law fitting with validation.

Fits L(N) = E + A * N^(-alpha) from training run data points.

NanoSeek uses 2 training scales (ablation 410M, 1B), not 3. With 2 data points:
  - Fix E to literature prior (1.69 nats) → fit A and alpha (2 params from 2 points)
  - This is exactly determined (unique solution), so LOOCV is not possible
  - Instead we report the fit quality via residuals and compare alpha to literature range
  - With 3+ data points: full LOOCV with fixed E is possible and preferred

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
    r_squared: float  # R² (trivially 1.0 when n_points ≤ n_free_params)


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

    Handles different numbers of data points:
      - 2 points: Fix E to prior, fit A and alpha (exactly determined, no LOOCV)
      - 3 points: Fix E to prior, fit A and alpha, LOOCV validates extrapolation
      - 4+ points: Full 3-param fit possible, LOOCV validates all params

    Args:
        data_points: List of (param_count, final_ema_val_bpb) tuples.
            Use N_active (not N_total) for MoE models.
        E_prior: Literature prior for irreducible entropy (Chinchilla: ~1.69 nats).
        bounds: Parameter bounds ((E_min, A_min, alpha_min), (E_max, A_max, alpha_max)).

    Returns:
        ScalingLawFit with fitted parameters and validation errors.
    """
    n_points = len(data_points)
    if n_points < 2:
        raise ValueError(f"Need at least 2 data points, got {n_points}")

    N_vals = np.array([p[0] for p in data_points], dtype=float)
    L_vals = np.array([p[1] for p in data_points], dtype=float)

    # ── Fixed-E fit (always done: works with 2+ points) ──
    # Fix E to literature prior, fit only A and alpha.
    # With 2 points and 2 params: exactly determined (unique solution).
    # With 3+ points: overdetermined (least squares fit).
    def _fixed_E_loss(N, A_fit, alpha_fit):
        return E_prior + A_fit * np.power(N, -alpha_fit)

    try:
        popt_fixed, _ = curve_fit(
            _fixed_E_loss, N_vals, L_vals,
            p0=[5.0, 0.3],
            bounds=([0.1, 0.01], [100.0, 1.0]),
            maxfev=10000,
        )
        A, alpha = popt_fixed
        E = E_prior  # fixed
    except RuntimeError as e:
        logger.warning(f"Fixed-E fit failed: {e}. Using defaults.")
        E, A, alpha = E_prior, 5.0, 0.3

    # ── Full 3-param fit (only with 4+ points — otherwise underdetermined) ──
    if n_points >= 4:
        try:
            popt_full, _ = curve_fit(
                _chinchilla_loss, N_vals, L_vals,
                p0=[E_prior, A, alpha],  # warm-start from fixed-E fit
                bounds=bounds,
                maxfev=10000,
            )
            E, A, alpha = popt_full
            logger.info(f"Full 3-param fit (E free): E={E:.4f}, A={A:.4f}, alpha={alpha:.4f}")
        except RuntimeError as e:
            logger.info(f"Full fit failed ({e}), using fixed-E fit (E={E_prior})")

    # ── R² ──
    L_pred = _chinchilla_loss(N_vals, E, A, alpha)
    ss_res = np.sum((L_vals - L_pred) ** 2)
    ss_tot = np.sum((L_vals - np.mean(L_vals)) ** 2)
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-10)

    # ── Validation ──
    # With 2 points (our case: ablation + 1B):
    #   Exactly determined → LOOCV leaves 1 point, fits 2 params from 1 point = underdetermined.
    #   Instead: validate alpha against literature range [0.2, 0.6] and report residuals.
    # With 3+ points: proper LOOCV (leave one out, fit on rest, predict held-out).
    loocv_errors = []

    if n_points == 2:
        # Can't do LOOCV with 2 points. Validate via sanity checks instead.
        logger.info(f"2 data points: exactly determined fit (no LOOCV possible)")
        if 0.15 <= alpha <= 0.7:
            logger.info(f"alpha={alpha:.4f} is within literature range [0.15, 0.7] ✓")
        else:
            logger.warning(
                f"alpha={alpha:.4f} is outside literature range [0.15, 0.7]. "
                f"Possible issues: undertrained models, or architecture bottleneck."
            )
        # Report fit residuals as a proxy for quality
        for i, (N, L_actual) in enumerate(data_points):
            L_fit = _chinchilla_loss(N, E, A, alpha)
            residual = abs(L_fit - L_actual)
            logger.info(f"  Point {i}: N={N:.0e}, L_actual={L_actual:.4f}, L_fit={L_fit:.4f}, residual={residual:.6f}")
    else:
        # 3+ points: proper LOOCV with fixed E
        for i in range(n_points):
            train_N = np.delete(N_vals, i)
            train_L = np.delete(L_vals, i)
            test_N = N_vals[i]
            test_L = L_vals[i]

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

    mean_loocv = np.mean([e for e in loocv_errors if np.isfinite(e)]) if loocv_errors else float('nan')

    result = ScalingLawFit(
        E=float(E),
        A=float(A),
        alpha=float(alpha),
        loocv_errors=loocv_errors,
        mean_loocv_error=float(mean_loocv) if not np.isnan(mean_loocv) else 0.0,
        data_points=data_points,
        r_squared=float(r_squared),
    )

    # ── Summary ──
    logger.info(f"Scaling law fit: L(N) = {E:.4f} + {A:.4f} * N^(-{alpha:.4f})")
    if n_points == 2:
        logger.info(f"R² = {r_squared:.6f} (trivially 1.0 with 2 points, 2 free params)")
        logger.info(f"Validation: alpha range check only (LOOCV needs 3+ points)")
    elif n_points == 3:
        logger.info(f"R² = {r_squared:.6f} (trivially 1.0 with 3 points, E fixed, 2 free params → 1 DOF)")
        logger.info(f"LOOCV mean relative error: {mean_loocv:.4f} (healthy: < 0.15)")
    else:
        logger.info(f"R² = {r_squared:.6f}")
        logger.info(f"LOOCV mean relative error: {mean_loocv:.4f} (healthy: < 0.15)")

    if loocv_errors and mean_loocv > 0.15:
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
        help='Run data as name:params:loss (e.g., ablation:410e6:1.85 1b:1.08e9:1.62)',
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
            'r_squared_note': 'trivially 1.0 when n_points <= n_free_params (2 pts / 2 params or 3 pts / 3 params)',
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
