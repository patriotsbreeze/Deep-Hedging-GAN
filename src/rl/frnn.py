"""Fractional RNN (fRNN) actor/critic — an alternative to SigFormer.

The fRNN relaxes the Markov assumption via a GRU-based recurrent core that
maintains an internal state across the hedging episode.  This allows the
agent to track the latent path trajectory of the rough volatility process.

Compared to the SigFormer the fRNN is:
  + More parameter-efficient
  + Naturally sequential (suited for online deployment)
  - Potentially slower to converge on very long episodes
  - Less expressive for long-range dependencies

In the project, both architectures are benchmarked under identical TD3
training conditions on the same synthetic MA-fBM environments.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple


class FRNNCore(nn.Module):
    """Shared GRU-based recurrent backbone.

    Processes a single-step observation and updates a hidden state.
    For episode resets, the hidden state must be manually zeroed.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        obs : Tensor (B, obs_dim) — single-step observation
        hidden : Tensor (n_layers, B, hidden_dim) or None

        Returns
        -------
        features : Tensor (B, hidden_dim)
        new_hidden : Tensor (n_layers, B, hidden_dim)
        """
        # GRU expects (B, seq_len, input_size) — seq_len=1 for online mode
        x = obs.unsqueeze(1)          # (B, 1, obs_dim)
        out, hidden = self.gru(x, hidden)
        features = self.norm(out.squeeze(1))  # (B, hidden_dim)
        return features, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)


class FRNNActor(nn.Module):
    """fRNN actor network: (obs, hidden) → (action, new_hidden).

    The hidden state acts as the agent's memory of past volatility regimes,
    enabling it to modulate the delta haircut in response to path roughness.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 1,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.core = FRNNCore(obs_dim, hidden_dim, n_layers, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (action, new_hidden)."""
        features, new_hidden = self.core(obs, hidden)
        action = self.head(features)
        return action, new_hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.core.init_hidden(batch_size, device)


class FRNNCritic(nn.Module):
    """Twin fRNN critic: independent Q-networks with recurrent cores.

    For TD3, the critic uses the full state-action sequence; here we use
    a flat (obs, action) as input without an explicit hidden state rollout
    (the signature in obs already carries temporal information).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 1,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        in_dim = obs_dim + action_dim

        self.core1 = FRNNCore(in_dim, hidden_dim, n_layers, dropout)
        self.head1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.core2 = FRNNCore(in_dim, hidden_dim, n_layers, dropout)
        self.head2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        h1: Optional[torch.Tensor] = None,
        h2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        f1, _ = self.core1(x, h1)
        f2, _ = self.core2(x, h2)
        return self.head1(f1), self.head2(f2)

    def Q1(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        h1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        f1, _ = self.core1(x, h1)
        return self.head1(f1)
