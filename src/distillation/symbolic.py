"""Policy distillation via symbolic regression (PySR).

The trained SigFormer (or fRNN) neural policy is treated as a black-box
function f(s) → a.  We generate a large dataset of (state, action) pairs
and run PySR's genetic algorithm to find the compact algebraic expression
that best approximates f on this dataset.

The output is a Pareto frontier of mathematical expressions trading off
accuracy against complexity.  The analyst selects the simplest formula that
retains sufficient accuracy.

Requires: pip install pysr
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn


class PolicyDistiller:
    """Generate (state, action) pairs and run PySR to distill the neural policy.

    Parameters
    ----------
    actor : nn.Module
        Trained actor network (SigFormerActor or FRNNActor).
    env : HedgingEnv
        Environment used to sample states (avoid using the same data as training).
    state_feature_names : list of str, optional
        Names for the state features used by PySR (aids interpretability).
    device : str
    """

    def __init__(
        self,
        actor: nn.Module,
        env,
        state_feature_names: Optional[List[str]] = None,
        device: str = "cpu",
    ):
        self.actor = actor.to(device)
        self.actor.eval()
        self.env = env
        self.feature_names = state_feature_names
        self.device = device
        self._dataset: Optional[pd.DataFrame] = None
        self._sr_model = None

    # ------------------------------------------------------------------
    # Step 1: Collect state-action dataset
    # ------------------------------------------------------------------

    @torch.no_grad()
    def collect_dataset(self, n_samples: int = 100_000) -> pd.DataFrame:
        """Roll out the policy and record (state, action) pairs.

        Parameters
        ----------
        n_samples : int
            Number of state-action pairs to collect.

        Returns
        -------
        df : DataFrame with columns [state_0, …, state_d, action]
        """
        states_list = []
        actions_list = []
        collected = 0

        while collected < n_samples:
            obs, _ = self.env.reset()
            done = False
            while not done and collected < n_samples:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                out = self.actor(obs_t)
                action = (out[0] if isinstance(out, tuple) else out).squeeze().cpu().numpy()
                states_list.append(obs.copy())
                actions_list.append(float(action))
                next_obs, _, terminated, truncated, _ = self.env.step(
                    np.array([float(action)])
                )
                done = terminated or truncated
                obs = next_obs
                collected += 1

        states = np.stack(states_list)
        actions = np.array(actions_list)

        d = states.shape[1]
        if self.feature_names and len(self.feature_names) == d:
            cols = self.feature_names
        else:
            cols = [f"s{i}" for i in range(d)]

        df = pd.DataFrame(states, columns=cols)
        df["action"] = actions
        self._dataset = df
        print(f"Collected {len(df)} state-action pairs.")
        return df

    # ------------------------------------------------------------------
    # Step 2: Run symbolic regression
    # ------------------------------------------------------------------

    def fit_symbolic(
        self,
        df: Optional[pd.DataFrame] = None,
        feature_cols: Optional[List[str]] = None,
        niterations: int = 100,
        maxsize: int = 25,
        populations: int = 15,
        binary_operators: Optional[List[str]] = None,
        unary_operators: Optional[List[str]] = None,
        model_selection: str = "best",
        verbosity: int = 0,
    ):
        """Run PySR to find a compact expression for the policy.

        Parameters
        ----------
        df : DataFrame, optional
            State-action dataset (uses self._dataset if None).
        feature_cols : list of str, optional
            Which state features to include in the regression.
            Using a subset (e.g., just inventory, moneyness, TTM, avg_IV)
            promotes simpler, more interpretable formulas.
        niterations : int
            Number of PySR evolution iterations.
        maxsize : int
            Maximum complexity of the symbolic expression.
        Returns
        -------
        sr_model : PySRRegressor
            Fitted model with .equations_ DataFrame and .predict() method.
        """
        try:
            from pysr import PySRRegressor
        except ImportError as e:
            raise ImportError(
                "PySR is required for symbolic regression. "
                "Install with: pip install pysr"
            ) from e

        data = df if df is not None else self._dataset
        if data is None:
            raise ValueError("No dataset available. Call collect_dataset() first.")

        if feature_cols is None:
            # Use a reduced set of interpretable state features
            # Focus on the low-dimensional features most relevant to the hedge ratio
            feature_cols = [c for c in data.columns if c != "action" and not c.startswith("s")]
            if not feature_cols:
                # Fall back to all columns except action, but warn
                feature_cols = [c for c in data.columns if c != "action"]
                print(
                    f"Warning: using all {len(feature_cols)} state features. "
                    "Consider providing feature_cols to focus on interpretable features."
                )

        X = data[feature_cols].values.astype(np.float32)
        y = data["action"].values.astype(np.float32)

        if binary_operators is None:
            binary_operators = ["+", "-", "*", "/"]
        if unary_operators is None:
            unary_operators = ["exp", "log", "abs", "sqrt"]

        sr = PySRRegressor(
            niterations=niterations,
            populations=populations,
            maxsize=maxsize,
            binary_operators=binary_operators,
            unary_operators=unary_operators,
            model_selection=model_selection,
            verbosity=verbosity,
            feature_names_in=feature_cols,
        )
        print(f"Running PySR with {niterations} iterations on {len(X)} samples ...")
        sr.fit(X, y)
        self._sr_model = sr
        print("Symbolic regression complete.")
        return sr

    # ------------------------------------------------------------------
    # Step 3: Analyse the Pareto frontier
    # ------------------------------------------------------------------

    def pareto_frontier(self) -> pd.DataFrame:
        """Return the Pareto frontier of (complexity, accuracy) trade-offs.

        Returns
        -------
        frontier : DataFrame with columns [complexity, loss, equation]
        """
        if self._sr_model is None:
            raise ValueError("Run fit_symbolic() first.")
        eqs = self._sr_model.equations_
        cols = ["complexity", "loss", "equation"]
        available = [c for c in cols if c in eqs.columns]
        return eqs[available].sort_values("complexity").reset_index(drop=True)

    def best_formula(self) -> str:
        """Return the string representation of the selected best equation."""
        if self._sr_model is None:
            raise ValueError("Run fit_symbolic() first.")
        return str(self._sr_model.sympy())

    def evaluate_accuracy(
        self,
        df: Optional[pd.DataFrame] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> dict:
        """Compute R² and MAE of the distilled formula against the neural policy."""
        if self._sr_model is None:
            raise ValueError("Run fit_symbolic() first.")

        data = df if df is not None else self._dataset
        if data is None:
            raise ValueError("No dataset available.")

        if feature_cols is None:
            eqs = self._sr_model.equations_
            feature_cols = self._sr_model.feature_names_in_

        X = data[feature_cols].values.astype(np.float32)
        y = data["action"].values.astype(np.float32)
        y_pred = self._sr_model.predict(X)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        mae = np.mean(np.abs(y - y_pred))

        return {"r2": float(r2), "mae": float(mae), "formula": self.best_formula()}

    def save_equations(self, path: str) -> None:
        """Save the full equation table to CSV."""
        if self._sr_model is None:
            raise ValueError("Run fit_symbolic() first.")
        self._sr_model.equations_.to_csv(path, index=False)
        print(f"Equations saved to {path}")
