"""Benchmark hedging strategies for out-of-sample comparison.

Two benchmarks per the project plan:
  1. Black-Scholes daily delta hedge — the classical parametric baseline.
  2. Naive TD3 — a deep hedging agent trained ONLY on historical data
     (no MA-fBM / SigGAN augmentation), to isolate the benefit of rough-
     volatility stress testing.
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from ..utils.black_scholes import bs_delta, bs_price


class BlackScholesBenchmark:
    """Classical Black-Scholes daily delta hedging benchmark.

    At each step, the agent holds exactly Δ^BS units of the underlying,
    computed from the current spot price, IV surface average, and TTM.
    No learning or optimisation is involved.

    Parameters
    ----------
    transaction_cost : float
        Proportional transaction cost on hedge trades.
    """

    def __init__(self, transaction_cost: float = 0.001, risk_free_rate: float = 0.0):
        self.tc = transaction_cost
        self.r = risk_free_rate

    def run_episode(
        self,
        spot_path: np.ndarray,   # (T+1,) log-spot prices
        iv_path: np.ndarray,     # (T+1,) average implied volatility
        K: float,
        ttm_years: float,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate one hedging episode under the BS delta strategy.

        Returns
        -------
        pnl : ndarray (T,) — daily PnL at each step
        inventory : ndarray (T+1,) — hedge position at each step
        """
        T = len(spot_path) - 1
        pnl = np.zeros(T)
        inventory = np.zeros(T + 1)

        for t in range(T):
            S = np.exp(spot_path[t])
            S_next = np.exp(spot_path[t + 1])
            sigma = iv_path[t]
            ttm = max(ttm_years - t * dt, 1e-6)
            ttm_next = max(ttm_years - (t + 1) * dt, 1e-6)

            delta = bs_delta(S, K, ttm, self.r, sigma, call=True)
            trade = delta - inventory[t]
            tc = self.tc * abs(trade) * S

            option_pnl = (
                bs_price(S_next, K, ttm_next, self.r, sigma, call=True)
                - bs_price(S, K, ttm, self.r, sigma, call=True)
            )
            hedge_pnl = -inventory[t] * (S_next - S)

            pnl[t] = option_pnl + hedge_pnl - tc
            inventory[t + 1] = delta

        return pnl, inventory

    def run_backtest(
        self,
        spot_paths: np.ndarray,  # (N, T+1)
        iv_paths: np.ndarray,    # (N, T+1)
        K: float,
        ttm_years: float,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the benchmark across all episodes.

        Returns
        -------
        pnl_matrix : ndarray (N, T)
        inventory_matrix : ndarray (N, T+1)
        """
        N, T1 = spot_paths.shape
        pnl_matrix = np.zeros((N, T1 - 1))
        inv_matrix = np.zeros((N, T1))
        for i in range(N):
            pnl_matrix[i], inv_matrix[i] = self.run_episode(
                spot_paths[i], iv_paths[i], K, ttm_years, dt
            )
        return pnl_matrix, inv_matrix


class NaiveTD3Benchmark:
    """Naive TD3 agent trained purely on historical data (no synthetic paths).

    This benchmark isolates the contribution of MA-fBM / SigGAN data
    augmentation by comparing a similarly architected but non-augmented agent.

    The agent is loaded from a pre-saved checkpoint trained without the
    synthetic rough environment.

    Parameters
    ----------
    agent : TD3Agent
        Trained TD3 agent (loaded from a historical-only training run).
    transaction_cost : float
    """

    def __init__(self, agent, transaction_cost: float = 0.001):
        self.agent = agent
        self.tc = transaction_cost

    def run_backtest(
        self,
        env,
        n_episodes: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Roll out the naive agent on the test environment.

        Returns
        -------
        pnl_matrix : ndarray (N, T)
        reward_matrix : ndarray (N, T)
        """
        n_episodes = n_episodes or env.n_episodes
        pnl_list = []
        reward_list = []

        for _ in range(n_episodes):
            obs, _ = env.reset()
            episode_pnl = []
            episode_reward = []
            done = False
            while not done:
                action = self.agent.select_action(obs, explore=False)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_pnl.append(info.get("total_pnl", 0.0))
                episode_reward.append(reward)
            pnl_list.append(episode_pnl)
            reward_list.append(episode_reward)

        max_len = max(len(p) for p in pnl_list)
        pnl_matrix = np.array([p + [0.0] * (max_len - len(p)) for p in pnl_list])
        reward_matrix = np.array([r + [0.0] * (max_len - len(r)) for r in reward_list])
        return pnl_matrix, reward_matrix
