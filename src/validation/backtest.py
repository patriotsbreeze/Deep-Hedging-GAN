"""Walk-forward backtester for out-of-sample validation (Phase V).

Design:
  - Training data: all data up to train_end (e.g. 2021-12-31).
  - Test data: data from test_start through test_end (e.g. 2022-2023).
  - The test set is never seen during MA-fBM/SigGAN or RL training.
  - The distilled symbolic policy, the raw SigFormer neural policy, the BS
    benchmark, and the naive TD3 agent are all evaluated on the same test paths.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List, Callable

from .metrics import HedgingMetrics, compare_strategies
from .benchmarks import BlackScholesBenchmark


class WalkForwardBacktester:
    """Orchestrate the out-of-sample walk-forward performance study.

    Parameters
    ----------
    test_env : HedgingEnv
        Environment seeded with held-out historical test data.
    reward_fn : HedgingReward
        Reward function instance (for computing accumulated rewards).
    K : float
        Option strike price.
    ttm_years : float
        Option time to maturity in years.
    dt : float
        Time step size in years (1/252 for daily).
    """

    def __init__(
        self,
        test_env,
        reward_fn,
        K: float = 1.0,
        ttm_years: float = 0.25,
        dt: float = 1.0 / 252.0,
    ):
        self.env = test_env
        self.reward_fn = reward_fn
        self.K = K
        self.ttm_years = ttm_years
        self.dt = dt
        self._results: Dict[str, HedgingMetrics] = {}

    # ------------------------------------------------------------------
    # Run individual strategies
    # ------------------------------------------------------------------

    def _rollout_agent(self, agent, n_episodes: int, label: str) -> HedgingMetrics:
        """Roll out an RL agent (or any callable) on the test environment."""
        pnl_list = []
        reward_list = []

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            self.reward_fn.reset()
            ep_pnl = []
            ep_reward = []
            done = False
            while not done:
                if callable(agent):
                    action = agent(obs)
                else:
                    action = agent.select_action(obs, explore=False)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                ep_pnl.append(info.get("total_pnl", 0.0))
                ep_reward.append(reward)
            pnl_list.append(ep_pnl)
            reward_list.append(ep_reward)

        max_len = max(len(p) for p in pnl_list)
        pnl_matrix = np.array([p + [0.0] * (max_len - len(p)) for p in pnl_list])
        reward_matrix = np.array([r + [0.0] * (max_len - len(r)) for r in reward_list])
        m = HedgingMetrics(pnl_matrix, reward_matrix)
        self._results[label] = m
        return m

    def run_neural_policy(
        self,
        agent,
        n_episodes: int = 1000,
        label: str = "SigFormer TD3",
    ) -> HedgingMetrics:
        """Evaluate the trained SigFormer TD3 agent."""
        return self._rollout_agent(agent, n_episodes, label)

    def run_symbolic_policy(
        self,
        formula_fn: Callable,
        n_episodes: int = 1000,
        label: str = "Distilled Formula",
    ) -> HedgingMetrics:
        """Evaluate the distilled symbolic formula.

        Parameters
        ----------
        formula_fn : callable
            Takes obs (ndarray) and returns action (ndarray shape (1,)).
        """
        return self._rollout_agent(formula_fn, n_episodes, label)

    def run_black_scholes(
        self,
        spot_paths: np.ndarray,
        iv_paths: np.ndarray,
        n_episodes: Optional[int] = None,
        label: str = "Black-Scholes",
    ) -> HedgingMetrics:
        """Evaluate the Black-Scholes delta hedge benchmark."""
        n_episodes = n_episodes or spot_paths.shape[0]
        bs = BlackScholesBenchmark(risk_free_rate=0.0)
        pnl_matrix, _ = bs.run_backtest(
            spot_paths[:n_episodes],
            iv_paths[:n_episodes],
            self.K,
            self.ttm_years,
            self.dt,
        )
        reward_matrix = np.zeros_like(pnl_matrix)
        for i in range(pnl_matrix.shape[0]):
            self.reward_fn.reset()
            for t in range(pnl_matrix.shape[1]):
                reward_matrix[i, t] = self.reward_fn.compute(pnl_matrix[i, t])

        m = HedgingMetrics(pnl_matrix, reward_matrix)
        self._results[label] = m
        return m

    def run_naive_td3(
        self,
        agent,
        n_episodes: int = 1000,
        label: str = "Naive TD3",
    ) -> HedgingMetrics:
        """Evaluate the naive TD3 benchmark (trained without rough augmentation)."""
        return self._rollout_agent(agent, n_episodes, label)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_comparison(self) -> None:
        """Print a formatted comparison table for all evaluated strategies."""
        if not self._results:
            print("No results yet. Run one or more strategy evaluations first.")
            return
        compare_strategies(self._results)

    def results_dataframe(self) -> pd.DataFrame:
        """Return a tidy DataFrame with one row per strategy."""
        rows = []
        for label, m in self._results.items():
            row = {"strategy": label}
            row.update(m.summary())
            rows.append(row)
        return pd.DataFrame(rows).set_index("strategy")

    def save_results(self, path: str) -> None:
        df = self.results_dataframe()
        df.to_csv(path)
        print(f"Results saved to {path}")
