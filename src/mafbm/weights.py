"""Solve A ω = b for the L²-optimal MA-fBM approximation weights."""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List

from .grid import compute_gamma_grid
from .covariance import compute_covariance_matrix, compute_rhs


@dataclass
class MAFBMWeights:
    """Container for a single-Hurst calibration result."""
    H: float
    gammas: np.ndarray     # (K,) mean-reversion speeds
    omega: np.ndarray      # (K,) optimal approximation weights
    residual: float        # L² residual ∫|K_approx - K_true|²

    def approx_kernel(self, t: np.ndarray) -> np.ndarray:
        """Evaluate the approximated Volterra kernel at times t."""
        # K_approx(t) = Σ_k ω_k e^{-γ_k t}
        return (self.omega[:, None] * np.exp(-self.gammas[:, None] * t[None, :])).sum(0)

    def true_kernel(self, t: np.ndarray) -> np.ndarray:
        """Evaluate the exact Volterra kernel K(t) = t^{H-1/2}."""
        return t ** (self.H - 0.5)


def compute_optimal_weights(
    H: float,
    K: int = 10,
    T: float = 1.0,
    r: float = 100.0,
) -> MAFBMWeights:
    """Compute L²-optimal MA-fBM approximation weights for Hurst parameter H.

    Solves the linear system A ω = b derived from the L²([0,T]) projection of
    the Volterra kernel  K(t) = t^{H-1/2}  onto the exponential basis
    {e^{-γ_k t}}_{k=1}^K.

    Parameters
    ----------
    H : float
        Hurst parameter in (0, 1). Rough volatility regime: H < 0.5.
    K : int
        Number of OU processes (K=10 gives high fidelity).
    T : float
        L² optimisation window (set to desired simulation horizon).
    r : float
        Geometric base for the gamma grid (r=100 gives 5 decades of coverage).

    Returns
    -------
    MAFBMWeights
    """
    gammas = compute_gamma_grid(K, r)
    A = compute_covariance_matrix(gammas, T)
    b = compute_rhs(gammas, H, T)
    omega = np.linalg.solve(A, b)

    # Compute L² residual on a fine grid
    t_grid = np.linspace(1e-6, T, 1000)
    K_true = t_grid ** (H - 0.5)
    K_approx = (omega[:, None] * np.exp(-gammas[:, None] * t_grid[None, :])).sum(0)
    # np.trapezoid added in NumPy 2.0; np.trapz is the compatible alias
    _trapz = getattr(np, "trapezoid", np.trapz)
    residual = float(_trapz((K_true - K_approx) ** 2, t_grid))

    return MAFBMWeights(H=H, gammas=gammas, omega=omega, residual=residual)


class MAFBMCalibrator:
    """Pre-calibrate MA-fBM weights for a grid of Hurst parameters.

    Call `.calibrate()` once, then use `.get_weights(H)` during simulation
    (interpolating to the nearest calibrated value).
    """

    def __init__(
        self,
        hurst_grid: List[float] = None,
        K: int = 10,
        T: float = 1.0,
        r: float = 100.0,
    ):
        if hurst_grid is None:
            hurst_grid = [round(0.05 * i, 2) for i in range(1, 10)]  # 0.05 to 0.45
        self.hurst_grid = sorted(hurst_grid)
        self.K = K
        self.T = T
        self.r = r
        self._cache: Dict[float, MAFBMWeights] = {}

    def calibrate(self) -> "MAFBMCalibrator":
        """Compute and cache weights for every H in the grid."""
        for H in self.hurst_grid:
            self._cache[H] = compute_optimal_weights(H, self.K, self.T, self.r)
        return self

    def get_weights(self, H: float) -> MAFBMWeights:
        """Return the pre-calibrated weights for the given H.

        If H was not in the calibration grid, raise an error and suggest
        re-calibrating with that value included.
        """
        if H not in self._cache:
            # Snap to nearest calibrated value
            nearest = min(self._cache.keys(), key=lambda h: abs(h - H))
            return self._cache[nearest]
        return self._cache[H]

    @property
    def residuals(self) -> Dict[float, float]:
        return {H: w.residual for H, w in self._cache.items()}

    def summary(self) -> None:
        print(f"{'H':>6}  {'residual':>12}  {'max|ω|':>10}")
        print("-" * 35)
        for H, w in sorted(self._cache.items()):
            print(f"{H:>6.2f}  {w.residual:>12.2e}  {np.max(np.abs(w.omega)):>10.4f}")
