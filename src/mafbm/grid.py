"""Geometric grid of mean-reversion speeds for the OU process ensemble."""
import numpy as np


def compute_gamma_grid(K: int, r: float = 100.0) -> np.ndarray:
    """Return geometrically spaced OU mean-reversion speeds γ_k = r^(k - n).

    Parameters
    ----------
    K : int
        Number of OU processes.
    r : float
        Geometric base (r > 1 ensures log-uniform spacing).

    Returns
    -------
    gammas : ndarray of shape (K,)
        Sorted ascending mean-reversion speeds.
    """
    n = (K + 1) / 2.0  # centering parameter so speeds straddle 1
    gammas = np.array([r ** (k - n) for k in range(1, K + 1)], dtype=np.float64)
    return gammas
