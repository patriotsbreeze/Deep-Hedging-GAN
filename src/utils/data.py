"""Market data loading and preprocessing utilities.

Supports:
  - Yahoo Finance via yfinance (spot prices)
  - Synthetic data generation (for testing without real data)
  - Implied volatility surface construction from option chain data
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict
from datetime import datetime

from .black_scholes import bs_price, implied_vol


class MarketDataLoader:
    """Load and preprocess historical market data for Phase II and Phase V.

    Parameters
    ----------
    ticker : str
        Underlying asset ticker (e.g. "^GSPC" for S&P 500).
    start : str
        Start date (YYYY-MM-DD).
    end : str
        End date (YYYY-MM-DD).
    cache_dir : str, optional
        Directory to cache downloaded data as CSV.
    """

    def __init__(
        self,
        ticker: str = "^GSPC",
        start: str = "2015-01-01",
        end: str = "2021-12-31",
        cache_dir: Optional[str] = None,
    ):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._spot: Optional[pd.Series] = None
        self._log_returns: Optional[pd.Series] = None

    def load_spot(self) -> pd.Series:
        """Download or load cached spot price series."""
        if self._spot is not None:
            return self._spot

        cache_path = self.cache_dir / f"{self.ticker.replace('^', '')}_{self.start}_{self.end}.csv" \
            if self.cache_dir else None

        if cache_path and cache_path.exists():
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            self._spot = df["Close"]
        else:
            try:
                import yfinance as yf
                data = yf.download(self.ticker, start=self.start, end=self.end, progress=False)
                self._spot = data["Close"]
                if cache_path:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    self._spot.to_csv(cache_path)
            except ImportError:
                raise ImportError("yfinance is required for market data. pip install yfinance")

        return self._spot

    def log_returns(self) -> pd.Series:
        if self._log_returns is None:
            spot = self.load_spot()
            self._log_returns = np.log(spot / spot.shift(1)).dropna()
        return self._log_returns

    def rolling_realised_vol(self, window: int = 21) -> pd.Series:
        """Annualised realised volatility using a rolling window."""
        lr = self.log_returns()
        return lr.rolling(window).std() * np.sqrt(252)

    def construct_path_windows(
        self,
        window: int = 63,
        step: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Slice the spot series into overlapping windows.

        Returns
        -------
        log_spot_windows : ndarray (N, window + 1)
        iv_windows : ndarray (N, window + 1)
            Proxy IV from rolling realised volatility.
        """
        spot = self.load_spot().values
        rv = self.rolling_realised_vol().values
        log_spot = np.log(spot)

        windows_s = []
        windows_iv = []
        for i in range(0, len(spot) - window, step):
            if np.any(np.isnan(rv[i: i + window + 1])):
                continue
            windows_s.append(log_spot[i: i + window + 1])
            windows_iv.append(rv[i: i + window + 1])

        return np.array(windows_s), np.array(windows_iv)


# ---------------------------------------------------------------------------
# Synthetic data generator (for testing without real market data)
# ---------------------------------------------------------------------------

def generate_synthetic_market(
    n_paths: int = 1000,
    n_steps: int = 63,
    S0: float = 100.0,
    mu: float = 0.05,
    sigma_base: float = 0.20,
    dt: float = 1.0 / 252.0,
    n_strikes: int = 5,
    n_maturities: int = 3,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate simple GBM spot paths and flat IV surfaces for testing.

    Parameters
    ----------
    n_paths : int
    n_steps : int
    S0 : float
    mu, sigma_base : float
    dt : float
    n_strikes : int
        Number of IV surface strike grid points.
    n_maturities : int
        Number of IV surface maturity grid points.
    seed : int, optional

    Returns
    -------
    spot_paths : ndarray (n_paths, n_steps + 1)  — log-spot prices
    iv_surfaces : ndarray (n_paths, n_steps + 1, n_strikes, n_maturities)
    """
    rng = np.random.default_rng(seed)
    log_S0 = np.log(S0)

    # GBM paths
    increments = rng.normal(
        (mu - 0.5 * sigma_base ** 2) * dt,
        sigma_base * np.sqrt(dt),
        size=(n_paths, n_steps),
    )
    log_spot = np.zeros((n_paths, n_steps + 1))
    log_spot[:, 0] = log_S0
    log_spot[:, 1:] = log_S0 + np.cumsum(increments, axis=1)

    # Simple flat IV surface with small random perturbations
    iv_base = sigma_base + rng.normal(0, 0.02, size=(n_paths, n_steps + 1))
    iv_surfaces = np.broadcast_to(
        iv_base[:, :, None, None],
        (n_paths, n_steps + 1, n_strikes, n_maturities),
    ).copy()
    # Add a mild smile: IV increases away from ATM
    smile = np.linspace(-0.03, 0.03, n_strikes)  # (n_strikes,)
    iv_surfaces += np.abs(smile)[None, None, :, None]
    iv_surfaces = np.clip(iv_surfaces, 0.05, 1.0)

    return log_spot, iv_surfaces
