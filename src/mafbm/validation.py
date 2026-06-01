"""Statistical validation of MA-fBM paths against theoretical fBM properties.

Key checks:
1. Hurst exponent recovery via log-log regression of quadratic variation.
2. Variance scaling: E[|B_t^H|²] ≈ c_H · t^{2H}.
3. Kernel approximation error vs the true Volterra kernel.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import linregress
from typing import Dict, Tuple


def compute_quadratic_variation(paths: np.ndarray, dt: float, q: int = 2) -> np.ndarray:
    """Compute the q-th variation of each path.

    The q-th variation of fBM scales as T^{qH}, so for q=2 we get the
    standard quadratic variation which scales as T^{2H}.

    Parameters
    ----------
    paths : ndarray of shape (n_paths, n_steps + 1)
    dt : float
        Time step size.
    q : int
        Order of the variation (default 2 for quadratic).

    Returns
    -------
    qv : ndarray of shape (n_paths,)
        q-th variation estimate for each path.
    """
    increments = np.diff(paths, axis=1)
    qv = (np.abs(increments) ** q).sum(axis=1)
    return qv


def estimate_hurst_exponent(
    paths: np.ndarray,
    dt: float,
    scales: int = 8,
) -> Tuple[float, float]:
    """Estimate the Hurst exponent via log-log regression of quadratic variation.

    Uses multiple sub-sampling scales to regress
      log E[|ΔB_Δt|²]  vs  log(Δt)
    The slope gives 2H.

    Parameters
    ----------
    paths : ndarray of shape (n_paths, n_steps + 1)
    dt : float
        Base time step.
    scales : int
        Number of log-spacing scales to include in the regression.

    Returns
    -------
    H_est : float
        Estimated Hurst exponent.
    se : float
        Standard error of the regression slope.
    """
    n_steps = paths.shape[1] - 1
    log_dt_list = []
    log_var_list = []

    for s in range(1, scales + 1):
        step = 2 ** s
        if step >= n_steps:
            break
        # Compute increments at lag `step`
        sub_paths = paths[:, ::step]
        incs = np.diff(sub_paths, axis=1)
        var_est = np.mean(incs ** 2)
        log_dt_list.append(np.log(step * dt))
        log_var_list.append(np.log(var_est))

    slope, intercept, r_value, p_value, se = linregress(log_dt_list, log_var_list)
    H_est = slope / 2.0
    return H_est, se / 2.0


def validate_hurst_exponent(
    paths: np.ndarray,
    H_target: float,
    dt: float,
    tol: float = 0.05,
) -> Dict:
    """Estimate H from paths and check it matches the target within tolerance.

    Returns a dict with keys: H_estimated, H_target, error, passed.
    """
    H_est, se = estimate_hurst_exponent(paths, dt)
    error = abs(H_est - H_target)
    return {
        "H_estimated": H_est,
        "H_target": H_target,
        "standard_error": se,
        "error": error,
        "passed": error < tol,
    }


def variance_scaling_check(
    paths: np.ndarray,
    H_target: float,
    dt: float,
) -> Dict:
    """Check E[|B_t^H|²] ∝ t^{2H} by regressing log-variance vs log-time.

    Returns slope (should be ≈ 2H) and R² of the fit.
    """
    n_steps = paths.shape[1] - 1
    t_grid = np.arange(1, n_steps + 1) * dt
    var_t = np.var(paths[:, 1:], axis=0)

    mask = var_t > 0
    slope, intercept, r, p, se = linregress(np.log(t_grid[mask]), np.log(var_t[mask]))
    return {
        "slope": slope,
        "expected_slope": 2 * H_target,
        "r_squared": r ** 2,
        "intercept": intercept,
    }


def full_validation_report(
    paths: np.ndarray,
    H_target: float,
    dt: float,
    kernel_weights: "MAFBMWeights" = None,  # noqa: F821 — avoid circular import
) -> None:
    """Print a human-readable validation summary for a batch of MA-fBM paths."""
    hurst_check = validate_hurst_exponent(paths, H_target, dt)
    var_check = variance_scaling_check(paths, H_target, dt)

    print(f"\n=== MA-fBM Validation  (H_target = {H_target:.2f}) ===")
    print(f"  Estimated Hurst:  {hurst_check['H_estimated']:.4f} ± {hurst_check['standard_error']:.4f}")
    print(f"  Estimation error: {hurst_check['error']:.4f}  (tol 0.05)  {'PASS' if hurst_check['passed'] else 'FAIL'}")
    print(f"  Variance scaling: slope={var_check['slope']:.4f}, expected={var_check['expected_slope']:.4f}, R²={var_check['r_squared']:.4f}")
    if kernel_weights is not None:
        print(f"  Kernel L² residual: {kernel_weights.residual:.4e}")
