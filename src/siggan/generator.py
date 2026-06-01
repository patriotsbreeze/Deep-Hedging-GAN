"""Auto-Regressive Feed-Forward Neural Network (AR-FNN) Generator.

The generator maps MA-fBM conditioning paths Z ∈ ℝ^{T×K} to synthetic
market paths X̂ ∈ ℝ^{T×d_out} (spot, IV surface, volume, …).

Architecture:
  1. Project the MA-fBM state (K OU values) at each step to a latent code.
  2. At each time step t, the AR-FNN maps (z_t, x̂_{t-1}) → x̂_t via a
     shared MLP, making it auto-regressive (each output depends on the prior).
  3. A final affine layer maps to the output channels.

The conditioning on MA-fBM paths (rather than iid Gaussian noise) ensures the
generated sequences inherit the anti-persistent, rough character required to
stress-test the RL hedging agent.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class ARFNNGenerator(nn.Module):
    """Conditional Auto-Regressive FNN Generator conditioned on MA-fBM paths.

    Parameters
    ----------
    noise_dim : int
        Dimension of the MA-fBM conditioning vector at each step (= K OU states).
    output_dim : int
        Number of output channels (e.g. 1 spot + n_strikes*n_maturities IV + 1 vol).
    hidden_dim : int
        Width of the shared hidden layers.
    n_layers : int
        Number of hidden layers.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        noise_dim: int = 10,
        output_dim: int = 6,
        hidden_dim: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Input: [MA-fBM conditioning (noise_dim)] + [previous output (output_dim)]
        in_dim = noise_dim + output_dim

        layers = [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor, x0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate a synthetic market path auto-regressively.

        Parameters
        ----------
        z : Tensor of shape (B, T, noise_dim)
            MA-fBM conditioning paths (K OU process values at each step).
        x0 : Tensor of shape (B, output_dim), optional
            Initial market state. Defaults to zeros.

        Returns
        -------
        x_hat : Tensor of shape (B, T, output_dim)
            Generated market path (log-returns, log-IV, etc.).
        """
        B, T, _ = z.shape
        device = z.device

        if x0 is None:
            x_prev = torch.zeros(B, self.output_dim, device=device)
        else:
            x_prev = x0

        outputs = []
        for t in range(T):
            inp = torch.cat([z[:, t, :], x_prev], dim=-1)  # (B, noise_dim + output_dim)
            x_t = self.net(inp)                             # (B, output_dim)
            outputs.append(x_t)
            x_prev = x_t

        return torch.stack(outputs, dim=1)  # (B, T, output_dim)


class ConditionalARFNNGenerator(nn.Module):
    """Extended generator that also accepts a global regime conditioning vector.

    Useful for explicitly conditioning on the Hurst parameter H so the
    generator can smoothly interpolate between rough and smooth regimes.
    """

    def __init__(
        self,
        noise_dim: int = 10,
        output_dim: int = 6,
        hidden_dim: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1,
        condition_dim: int = 1,   # e.g. Hurst parameter H
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.output_dim = output_dim
        self.condition_dim = condition_dim

        in_dim = noise_dim + output_dim + condition_dim
        layers = [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        z: torch.Tensor,
        c: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z : Tensor (B, T, noise_dim)
        c : Tensor (B, condition_dim)   — broadcast across time steps
        x0 : Tensor (B, output_dim), optional
        """
        B, T, _ = z.shape
        device = z.device
        c_expanded = c.unsqueeze(1).expand(-1, T, -1)  # (B, T, condition_dim)

        if x0 is None:
            x_prev = torch.zeros(B, self.output_dim, device=device)
        else:
            x_prev = x0

        outputs = []
        for t in range(T):
            inp = torch.cat([z[:, t, :], x_prev, c_expanded[:, t, :]], dim=-1)
            x_t = self.net(inp)
            outputs.append(x_t)
            x_prev = x_t

        return torch.stack(outputs, dim=1)
