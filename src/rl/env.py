"""Gymnasium environment for the dynamic option hedging problem.

State space:
  - Current hedge inventory (fractional units short of underlying)
  - Option moneyness (log S/K)
  - Time to maturity (TTM / T_max, normalised to [0,1])
  - Flattened IV surface grid (n_strikes × n_maturities)
  - Truncated path signature of recent (spot, IV) co-movements

Action space:
  - Continuous scalar δ_t ∈ [-1, 1] representing the target hedge ratio
    (fraction of the option's Black-Scholes delta to hold).

Reward:
  - Local downside-shortfall reward with dynamic-κ ES linkage.

The environment can be seeded with synthetic MA-fBM paths (from SigGAN)
or real historical data for walk-forward validation.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from .reward import HedgingReward
from ..utils.black_scholes import bs_delta, bs_price


class HedgingEnv(gym.Env):
    """Option hedging environment for TD3 / SAC agents.

    Parameters
    ----------
    market_paths : ndarray of shape (n_episodes, n_steps + 1)
        Simulated (or historical) log spot-price paths.
    iv_surfaces : ndarray of shape (n_episodes, n_steps + 1, n_strikes, n_maturities)
        Implied volatility surface at each step.
    option_params : dict
        Keys: strike K, initial spot S0, risk_free_rate r, ttm (days).
    reward_cfg : dict, optional
        Keyword arguments forwarded to HedgingReward.
    history_window : int
        Length of the signature history window.
    sig_depth : int
        Path signature truncation depth for the state embedding.
    transaction_cost : float
        Proportional transaction cost on the hedge trade (fraction of notional).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        market_paths: np.ndarray,
        iv_surfaces: np.ndarray,
        option_params: Optional[dict] = None,
        reward_cfg: Optional[dict] = None,
        history_window: int = 20,
        sig_depth: int = 3,
        transaction_cost: float = 0.001,
    ):
        super().__init__()

        self.market_paths = market_paths     # (N, T+1)
        self.iv_surfaces = iv_surfaces       # (N, T+1, n_strikes, n_maturities)
        self.n_episodes, self.n_steps = market_paths.shape[0], market_paths.shape[1] - 1

        opt = option_params or {}
        self.K = opt.get("strike", 1.0)           # normalised strike
        self.r = opt.get("risk_free_rate", 0.0)
        self.ttm_max = opt.get("ttm", 63) / 252.0  # in years
        self.transaction_cost = transaction_cost
        self.history_window = history_window
        self.sig_depth = sig_depth

        # IV surface grid size
        self.n_strikes, self.n_maturities = iv_surfaces.shape[2], iv_surfaces.shape[3]
        self.iv_flat_dim = self.n_strikes * self.n_maturities

        # Path signature dimension for 2-channel (spot, avg_iv) history
        # d=2, depth=3, with time → d_eff=3 → sig_dim = 3 + 9 + 27 = 39
        d_eff = 3  # time + spot + avg_iv
        self.sig_dim = sum(d_eff ** k for k in range(1, sig_depth + 1))

        # Observation: [inventory (1), moneyness (1), ttm_frac (1), iv_surface (flat), sig]
        obs_dim = 1 + 1 + 1 + self.iv_flat_dim + self.sig_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Action: continuous hedge ratio in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.reward_fn = HedgingReward(**(reward_cfg or {}))

        # Episode state
        self._ep_idx: int = 0
        self._step: int = 0
        self._inventory: float = 0.0
        self._spot_history: list = []
        self._iv_history: list = []

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._ep_idx = self.np_random.integers(0, self.n_episodes)
        self._step = 0
        self._inventory = 0.0
        self.reward_fn.reset()

        # Initialise history buffers
        S0 = float(self.market_paths[self._ep_idx, 0])
        iv0 = self.iv_surfaces[self._ep_idx, 0].mean()
        self._spot_history = [S0] * self.history_window
        self._iv_history = [iv0] * self.history_window

        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        delta_target = float(np.clip(action[0], -1.0, 1.0))

        log_spot_curr = float(self.market_paths[self._ep_idx, self._step])
        log_spot_next = float(self.market_paths[self._ep_idx, self._step + 1])
        spot_curr = np.exp(log_spot_curr)   # actual spot price for BS/PnL
        spot_next = np.exp(log_spot_next)   # actual spot price for BS/PnL
        iv_curr = self.iv_surfaces[self._ep_idx, self._step]

        ttm_curr = self.ttm_max * (1.0 - self._step / self.n_steps)
        ttm_next = self.ttm_max * (1.0 - (self._step + 1) / self.n_steps)

        # Black-Scholes delta as the reference target
        moneyness = np.log(spot_curr / self.K)
        sigma = float(iv_curr.mean())
        bs_delta_curr = bs_delta(spot_curr, self.K, ttm_curr, self.r, sigma, call=True)

        # Agent's target inventory = delta_target * bs_delta (haircut interpretation)
        new_inventory = delta_target * bs_delta_curr
        trade = new_inventory - self._inventory
        tc = self.transaction_cost * abs(trade) * spot_curr

        # Option PnL (long call): change in BS price
        option_pnl = bs_price(spot_next, self.K, ttm_next, self.r, sigma, call=True) \
                   - bs_price(spot_curr, self.K, ttm_curr, self.r, sigma, call=True)

        # Hedge PnL: short position in underlying
        hedge_pnl = -self._inventory * (spot_next - spot_curr)

        # Total PnL net of transaction costs
        total_pnl = option_pnl + hedge_pnl - tc

        reward = self.reward_fn.compute(total_pnl)

        # Update state
        self._inventory = new_inventory
        self._step += 1
        self._spot_history.append(log_spot_next)  # history stores log-spot (already in log space)
        self._iv_history.append(float(self.iv_surfaces[self._ep_idx, self._step].mean()))
        if len(self._spot_history) > self.history_window:
            self._spot_history = self._spot_history[-self.history_window:]
            self._iv_history = self._iv_history[-self.history_window:]

        terminated = self._step >= self.n_steps
        obs = self._get_obs()

        info = {
            "option_pnl": option_pnl,
            "hedge_pnl": hedge_pnl,
            "transaction_cost": tc,
            "total_pnl": total_pnl,
            "bs_delta": bs_delta_curr,
            "inventory": self._inventory,
        }
        return obs, reward, terminated, False, info

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def _compute_signature(self) -> np.ndarray:
        """Compute truncated path signature of the (time, log-spot, avg-iv) history."""
        T = len(self._spot_history)
        time_ch = np.linspace(0, 1, T)
        path = np.column_stack([time_ch, self._spot_history, self._iv_history])  # (T, 3)
        return self._sig_numpy(path, self.sig_depth)

    @staticmethod
    def _sig_numpy(path: np.ndarray, depth: int) -> np.ndarray:
        """Minimal signature computation (levels 1-3) without external deps."""
        T, d = path.shape
        incs = np.diff(path, axis=0)

        # Level 1
        s1 = incs.sum(0)

        # Level 2: iterated sums
        partial1 = np.zeros(d)
        s2_acc = np.zeros(d * d)
        for t in range(T - 1):
            s2_acc += np.outer(partial1, incs[t]).ravel()
            partial1 += incs[t]
        s2 = s2_acc

        if depth == 2:
            return np.concatenate([s1, s2])

        # Level 3
        partial2 = np.zeros(d * d)
        partial1 = np.zeros(d)
        s3_acc = np.zeros(d ** 3)
        for t in range(T - 1):
            s3_acc += np.outer(partial2, incs[t]).ravel()
            partial2 += np.outer(partial1, incs[t]).ravel()
            partial1 += incs[t]
        s3 = s3_acc

        return np.concatenate([s1, s2, s3])

    def _get_obs(self) -> np.ndarray:
        step = min(self._step, self.n_steps)
        log_spot = float(self.market_paths[self._ep_idx, step])
        moneyness = float(log_spot - np.log(self.K))  # log(actual_spot / K)
        ttm_frac = 1.0 - step / self.n_steps
        iv_flat = self.iv_surfaces[self._ep_idx, step].ravel()
        sig = self._compute_signature()

        obs = np.concatenate([
            [self._inventory],
            [moneyness],
            [ttm_frac],
            iv_flat.astype(np.float32),
            sig.astype(np.float32),
        ]).astype(np.float32)
        return obs
