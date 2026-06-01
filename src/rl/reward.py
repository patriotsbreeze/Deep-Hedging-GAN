"""Reward functions for the deep hedging agent.

The primary reward is the absolute-value local downside-shortfall reward:

    r_{t+1} = scale * (base + PnL_{t+1}^{(100)} - κ * |PnL_{t+1}^{(100)}|^α)

where PnL^{(100)} = PnL * 100, κ penalises downside deviations, and α=1
gives a stable, non-explosive gradient signal.

The dynamic penalty κ is linked to Expected Shortfall (ES) of recent PnL
history.  When ES is large (bad tail risk), κ escalates, forcing the agent
to prioritise jump protection over exploiting the empirical leverage effect.
"""
from __future__ import annotations

import numpy as np
from collections import deque


class HedgingReward:
    """Stateful reward computer that tracks recent PnL for dynamic κ.

    Parameters
    ----------
    scale : float
        Outer scaling factor (default 10).
    base : float
        Daily base revenue assumption (default 0.03 = 3 bps).
    alpha : float
        Polynomial exponent for the downside penalty (1 = absolute value).
    kappa_base : float
        Base penalty multiplier.
    es_quantile : float
        Quantile for Expected Shortfall (0.05 = 5th percentile → CVaR_95).
    es_window : int
        Rolling window length for ES estimation.
    dynamic_kappa : bool
        If True, scale κ by the current ES estimate.
    """

    def __init__(
        self,
        scale: float = 10.0,
        base: float = 0.03,
        alpha: float = 1.0,
        kappa_base: float = 1.0,
        es_quantile: float = 0.05,
        es_window: int = 20,
        dynamic_kappa: bool = True,
    ):
        self.scale = scale
        self.base = base
        self.alpha = alpha
        self.kappa_base = kappa_base
        self.es_quantile = es_quantile
        self.dynamic_kappa = dynamic_kappa
        self._pnl_history: deque = deque(maxlen=es_window)

    def reset(self) -> None:
        self._pnl_history.clear()

    def _expected_shortfall(self) -> float:
        """Estimate ES from the rolling PnL window (lower = worse tail)."""
        if len(self._pnl_history) < 5:
            return 0.0
        pnl_arr = np.array(self._pnl_history)
        threshold = np.quantile(pnl_arr, self.es_quantile)
        tail = pnl_arr[pnl_arr <= threshold]
        if len(tail) == 0:
            return 0.0
        return float(np.mean(tail))   # negative value

    def kappa(self) -> float:
        """Return the current effective penalty multiplier."""
        if not self.dynamic_kappa:
            return self.kappa_base
        es = self._expected_shortfall()
        # Amplify κ when ES is very negative (rough regime detected)
        amplification = max(1.0, 1.0 + abs(es) / (self.base + 1e-8))
        return self.kappa_base * amplification

    def compute(self, pnl: float) -> float:
        """Compute the reward for a single step PnL value.

        Parameters
        ----------
        pnl : float
            Raw daily PnL (option P&L + hedge P&L + transaction costs).

        Returns
        -------
        reward : float
        """
        pnl_scaled = pnl * 100.0  # PnL^{(100)}
        self._pnl_history.append(pnl_scaled)
        kappa = self.kappa()
        reward = self.scale * (self.base + pnl_scaled - kappa * (abs(pnl_scaled) ** self.alpha))
        return float(reward)

    def compute_batch(self, pnl_arr: np.ndarray) -> np.ndarray:
        """Compute rewards for an array of PnL values (no history update)."""
        pnl_scaled = pnl_arr * 100.0
        kappa = self.kappa()
        return self.scale * (self.base + pnl_scaled - kappa * (np.abs(pnl_scaled) ** self.alpha))
