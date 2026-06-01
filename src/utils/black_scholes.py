"""Black-Scholes pricing and Greeks for European options.

All functions accept array inputs and broadcast via NumPy.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _d1(S: np.ndarray, K: float, T: float, r: float, sigma: float) -> np.ndarray:
    T = np.maximum(T, 1e-10)
    sigma = np.maximum(sigma, 1e-10)
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def _d2(S: np.ndarray, K: float, T: float, r: float, sigma: float) -> np.ndarray:
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(np.maximum(T, 1e-10))


def bs_price(
    S: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: float,
    call: bool = True,
) -> np.ndarray:
    """Black-Scholes European option price.

    Parameters
    ----------
    S : spot price
    K : strike price
    T : time to maturity in years
    r : risk-free rate
    sigma : implied volatility
    call : True for call, False for put

    Returns
    -------
    price : ndarray (same shape as S)
    """
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    disc = np.exp(-r * T)
    if call:
        return S * norm.cdf(d1) - K * disc * norm.cdf(d2)
    else:
        return K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(
    S: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: float,
    call: bool = True,
) -> np.ndarray:
    """Black-Scholes delta (∂V/∂S).

    For a long call: Δ ∈ [0, 1].
    For a long put:  Δ ∈ [-1, 0].
    """
    d1 = _d1(S, K, T, r, sigma)
    if call:
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1.0


def bs_gamma(
    S: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> np.ndarray:
    """Black-Scholes gamma (∂²V/∂S²), same for calls and puts."""
    d1 = _d1(S, K, T, r, sigma)
    T_safe = np.maximum(T, 1e-10)
    sigma_safe = np.maximum(sigma, 1e-10)
    return norm.pdf(d1) / (S * sigma_safe * np.sqrt(T_safe))


def bs_vega(
    S: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> np.ndarray:
    """Black-Scholes vega (∂V/∂σ), same for calls and puts.

    Returns vega per unit change in volatility (not per 1 pp).
    """
    d1 = _d1(S, K, T, r, sigma)
    T_safe = np.maximum(T, 1e-10)
    return S * norm.pdf(d1) * np.sqrt(T_safe)


def implied_vol(
    price_target: float,
    S: float,
    K: float,
    T: float,
    r: float,
    call: bool = True,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Invert the BS formula to find implied volatility via Newton-Raphson."""
    sigma = 0.3  # initial guess
    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma, call)
        vega = bs_vega(S, K, T, r, sigma)
        if abs(vega) < 1e-14:
            break
        diff = price - price_target
        if abs(diff) < tol:
            break
        sigma = sigma - diff / vega
        sigma = max(1e-6, min(sigma, 10.0))
    return sigma
