"""Simulate MA-fBM paths from the ensemble of coupled OU processes.

Key insight: all K OU processes share the *same* driving Brownian increments.
This coupling is what makes the linear combination Σ_k ω_k Y_t^(k) reproduce
the correlated-increment structure of fractional Brownian motion.

Exact discrete update for the k-th OU process driven by dB_t:

    Y_{t+Δ}^(k) = e^{-γ_k Δ} Y_t^(k)  +  ξ_k(Δ) · ε_t

where  ξ_k(Δ) = sqrt((1 - e^{-2γ_k Δ}) / (2γ_k))  and ε_t ~ N(0,1) is
SHARED across all k.

B̂_t^H ≈ Σ_k ω_k Y_t^(k)
"""
from __future__ import annotations

import numpy as np
from typing import Optional

from .weights import MAFBMWeights, MAFBMCalibrator


class MAFBMSimulator:
    """Generate batches of MA-fBM paths for a given Hurst parameter.

    Parameters
    ----------
    calibrator : MAFBMCalibrator
        Pre-calibrated weights (call `.calibrate()` before passing).
    dt : float
        Time-step size (e.g. 1/252 for daily steps over one year).
    n_steps : int
        Number of time steps per path.
    seed : int, optional
        NumPy random seed for reproducibility.
    """

    def __init__(
        self,
        calibrator: MAFBMCalibrator,
        dt: float = 1.0 / 252.0,
        n_steps: int = 252,
        seed: Optional[int] = None,
    ):
        self.calibrator = calibrator
        self.dt = dt
        self.n_steps = n_steps
        self.rng = np.random.default_rng(seed)

    def simulate(
        self,
        H: float,
        n_paths: int,
    ) -> np.ndarray:
        """Generate n_paths independent MA-fBM sample paths of length n_steps.

        Parameters
        ----------
        H : float
            Hurst parameter.
        n_paths : int
            Number of independent paths to generate.

        Returns
        -------
        paths : ndarray of shape (n_paths, n_steps + 1)
            MA-fBM paths starting at 0.  paths[:, 0] == 0.
        """
        weights = self.calibrator.get_weights(H)
        gammas = weights.gammas   # (K,)
        omega = weights.omega     # (K,)
        K = len(gammas)
        dt = self.dt

        # Precompute decay factors and noise scale per OU process
        decay = np.exp(-gammas * dt)                          # (K,)
        noise_scale = np.sqrt((1.0 - np.exp(-2.0 * gammas * dt)) / (2.0 * gammas))  # (K,)

        # OU state: (n_paths, K)
        Y = np.zeros((n_paths, K))

        # Output: (n_paths, n_steps+1) — store the MA-fBM value at each step
        paths = np.zeros((n_paths, self.n_steps + 1))

        for t in range(self.n_steps):
            # Shared Brownian increment: (n_paths,)
            eps = self.rng.standard_normal(n_paths)

            # Update all OU processes simultaneously
            # Y_new[:, k] = decay[k] * Y[:, k] + noise_scale[k] * eps
            Y = decay[None, :] * Y + noise_scale[None, :] * eps[:, None]  # (n_paths, K)

            # MA-fBM value: Σ_k ω_k Y_{t+1}^(k)
            paths[:, t + 1] = Y @ omega  # (n_paths,)

        return paths

    def simulate_increments(
        self,
        H: float,
        n_paths: int,
    ) -> "tuple[np.ndarray, np.ndarray]":
        """Return both paths and their increments.

        Returns
        -------
        paths : ndarray of shape (n_paths, n_steps + 1)
        increments : ndarray of shape (n_paths, n_steps)
        """
        paths = self.simulate(H, n_paths)
        increments = np.diff(paths, axis=1)
        return paths, increments

    def simulate_ou_states(
        self,
        H: float,
        n_paths: int,
    ) -> "tuple[np.ndarray, np.ndarray]":
        """Return the full OU state matrix alongside the MA-fBM paths.

        The OU states are the Markovian sufficient statistics that allow the
        RL agent to condition on the current rough-vol regime without needing
        the full path history.

        Returns
        -------
        paths : ndarray of shape (n_paths, n_steps + 1)
        ou_states : ndarray of shape (n_paths, n_steps + 1, K)
            OU process values at each time step.
        """
        weights = self.calibrator.get_weights(H)
        gammas = weights.gammas
        omega = weights.omega
        K = len(gammas)
        dt = self.dt

        decay = np.exp(-gammas * dt)
        noise_scale = np.sqrt((1.0 - np.exp(-2.0 * gammas * dt)) / (2.0 * gammas))

        Y = np.zeros((n_paths, K))
        paths = np.zeros((n_paths, self.n_steps + 1))
        ou_states = np.zeros((n_paths, self.n_steps + 1, K))
        ou_states[:, 0, :] = Y

        for t in range(self.n_steps):
            eps = self.rng.standard_normal(n_paths)
            Y = decay[None, :] * Y + noise_scale[None, :] * eps[:, None]
            paths[:, t + 1] = Y @ omega
            ou_states[:, t + 1, :] = Y

        return paths, ou_states
