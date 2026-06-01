"""L²-optimal covariance matrix and RHS vector for MA-fBM calibration.

The MA-fBM approximates the Volterra kernel K(t) = t^{H-1/2} by a sum of
exponentials:  K(t) ≈ Σ_k ω_k e^{-γ_k t}.

The L²([0,T]) optimal weights minimise
    ∫_0^T |Σ_k ω_k e^{-γ_k t} - t^{H-1/2}|² dt

which gives the normal equations  A ω = b  with:

    A_{i,j} = ∫_0^T e^{-(γ_i + γ_j)t} dt
             = (1 - e^{-(γ_i+γ_j)T}) / (γ_i + γ_j)

    b_k = ∫_0^T e^{-γ_k t} t^{H-1/2} dt
        = γ_k^{-(H+1/2)} · γ(H+1/2, γ_k T)

where γ(a, x) is the *lower* incomplete gamma function.

References
----------
Carmona, Coutin, Montseny (2000); Gatheral, Jaisson, Rosenbaum (2018).
"""
import numpy as np
from scipy.special import gamma as gamma_func, gammainc


def compute_covariance_matrix(gammas: np.ndarray, T: float) -> np.ndarray:
    """Compute the Gram matrix A for the exponential basis {e^{-γ_k t}}_{k=1}^K.

    Parameters
    ----------
    gammas : ndarray of shape (K,)
        OU mean-reversion speeds.
    T : float
        Terminal time for the L² optimisation window.

    Returns
    -------
    A : ndarray of shape (K, K)
        Symmetric positive-definite covariance matrix.
    """
    K = len(gammas)
    gi = gammas[:, None]   # (K, 1)
    gj = gammas[None, :]   # (1, K)
    A = (1.0 - np.exp(-(gi + gj) * T)) / (gi + gj)
    return A


def compute_rhs(gammas: np.ndarray, H: float, T: float) -> np.ndarray:
    """Compute the RHS vector b_k = ∫_0^T e^{-γ_k t} t^{H-1/2} dt.

    Uses the relation  ∫_0^T e^{-γ t} t^{α-1} dt = γ^{-α} Γ(α) P(α, γT)
    where P(α, x) = γ(α, x)/Γ(α) is the regularised lower incomplete gamma
    (= scipy.special.gammainc).

    Parameters
    ----------
    gammas : ndarray of shape (K,)
    H : float
        Hurst parameter in (0, 1).
    T : float

    Returns
    -------
    b : ndarray of shape (K,)
    """
    alpha = H + 0.5  # exponent: t^{H-1/2} = t^{alpha-1}
    x = gammas * T   # (K,)
    # gammainc(a, x) = P(a, x) = regularised lower incomplete gamma
    reg_lower = gammainc(alpha, x)
    full_gamma = gamma_func(alpha)
    b = (full_gamma * reg_lower) / (gammas ** alpha)
    return b
