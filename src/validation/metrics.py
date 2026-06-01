"""Risk and performance metrics for the hedging validation study.

Three primary metrics from the project plan:
  1. Accumulated Reward — total sum of the local downside-shortfall reward.
  2. Terminal Downside Variance — variance of the final-day negative PnL.
  3. 5th-percentile CVaR — Expected Shortfall at the 95% confidence level.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List


class HedgingMetrics:
    """Compute and aggregate hedging performance metrics.

    Parameters
    ----------
    pnl_paths : ndarray of shape (n_episodes, n_steps)
        Daily PnL (after transaction costs) for each episode.
    reward_paths : ndarray of shape (n_episodes, n_steps)
        Reward realised at each step.
    """

    def __init__(self, pnl_paths: np.ndarray, reward_paths: np.ndarray):
        self.pnl = pnl_paths           # (N, T)
        self.rewards = reward_paths    # (N, T)
        self.terminal_pnl = pnl_paths[:, -1]  # (N,) — last-day PnL
        self.cumulative_pnl = pnl_paths.sum(axis=1)  # (N,)

    # ------------------------------------------------------------------
    # Primary metrics
    # ------------------------------------------------------------------

    def accumulated_reward(self) -> float:
        """Mean total reward across episodes."""
        return float(self.rewards.sum(axis=1).mean())

    def terminal_downside_variance(self, quantile: float = 0.0) -> float:
        """Variance of the terminal PnL distribution below quantile q.

        With q=0 this is the variance of the entire terminal PnL distribution.
        With q=0.5 this focuses on the lower half, matching the downside-only
        objective used during training.
        """
        tpnl = self.terminal_pnl
        if quantile > 0:
            threshold = np.quantile(tpnl, quantile)
            tpnl = tpnl[tpnl <= threshold]
        return float(np.var(tpnl))

    def cvar(self, quantile: float = 0.05) -> float:
        """Compute the Expected Shortfall (CVaR) at the given quantile.

        CVaR_q = E[PnL | PnL ≤ VaR_q]

        Parameters
        ----------
        quantile : float
            Tail probability level (0.05 → CVaR_95).

        Returns
        -------
        cvar : float
            Mean of all PnL realisations below the q-th percentile.
            More negative = worse tail risk.
        """
        cumulative_pnl = self.cumulative_pnl
        threshold = np.quantile(cumulative_pnl, quantile)
        tail = cumulative_pnl[cumulative_pnl <= threshold]
        if len(tail) == 0:
            return float(threshold)
        return float(np.mean(tail))

    def var(self, quantile: float = 0.05) -> float:
        """Value at Risk at the given quantile."""
        return float(np.quantile(self.cumulative_pnl, quantile))

    def sharpe_ratio(self, annualise: bool = True, periods_per_year: int = 252) -> float:
        """Daily Sharpe ratio of the cumulative PnL distribution."""
        daily_pnl = self.pnl.mean(axis=0)  # average across episodes
        mu = np.mean(daily_pnl)
        sigma = np.std(daily_pnl) + 1e-10
        sharpe = mu / sigma
        if annualise:
            sharpe *= np.sqrt(periods_per_year)
        return float(sharpe)

    def summary(self) -> Dict[str, float]:
        return {
            "accumulated_reward": self.accumulated_reward(),
            "terminal_downside_variance": self.terminal_downside_variance(),
            "cvar_5": self.cvar(0.05),
            "var_5": self.var(0.05),
            "sharpe_ratio": self.sharpe_ratio(),
            "mean_cumulative_pnl": float(self.cumulative_pnl.mean()),
            "std_cumulative_pnl": float(self.cumulative_pnl.std()),
        }

    def print_summary(self, label: str = "Strategy") -> None:
        m = self.summary()
        print(f"\n{'─' * 40}")
        print(f"  {label}")
        print(f"{'─' * 40}")
        for k, v in m.items():
            print(f"  {k:<30} {v:>+10.4f}")
        print(f"{'─' * 40}")


def compare_strategies(strategies: Dict[str, "HedgingMetrics"]) -> None:
    """Print a side-by-side comparison table for multiple strategies."""
    if not strategies:
        return
    keys = list(strategies.values())[0].summary().keys()
    header = f"{'Metric':<32}" + "".join(f"{k:>18}" for k in strategies)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for metric in keys:
        row = f"{metric:<32}"
        for label, m in strategies.items():
            row += f"{m.summary()[metric]:>+18.4f}"
        print(row)
    print("=" * len(header))
