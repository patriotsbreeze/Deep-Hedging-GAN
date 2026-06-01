"""SigFormer: Signature Transformer actor/critic for TD3.

The SigFormer processes the flattened observation vector through:
  1. A linear projection into a d_model-dimensional embedding.
  2. A positional encoding (the raw signature already captures ordering,
     so we use a learnable scalar position embedding).
  3. N Transformer encoder layers (multi-head self-attention + FFN).
  4. A final linear head to produce the action (actor) or Q-value (critic).

For the actor, the output is tanh-squashed to [-1, 1].
For the twin critics, two independent networks output scalar Q-values.

Note: In the deep hedging context, the "sequence" is not time steps but
the feature groups in the state vector.  The Transformer allows the agent
to attend across inventory, moneyness, TTM, IV surface, and signature
features simultaneously, weighting their contributions dynamically.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PositionalEncoding(nn.Module):
    """Learnable positional embedding added to the input tokens."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        return x + self.pe(positions)


class SigFormerBlock(nn.Module):
    """Single Transformer encoder block with pre-LN architecture."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # Pre-LN FFN
        x = x + self.ffn(self.norm2(x))
        return x


class SigFormerBackbone(nn.Module):
    """Shared Transformer backbone for the actor and critic.

    Splits the flat observation into n_tokens tokens of equal size and
    processes them through N Transformer blocks.
    """

    def __init__(
        self,
        obs_dim: int,
        d_model: int = 256,
        n_layers: int = 5,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        n_tokens: int = 8,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model

        # Pad obs_dim to be divisible by n_tokens
        self.pad_dim = (-obs_dim % n_tokens)
        token_dim = (obs_dim + self.pad_dim) // n_tokens

        self.input_proj = nn.Linear(token_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=n_tokens)
        self.blocks = nn.ModuleList([
            SigFormerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        obs : Tensor (B, obs_dim)

        Returns
        -------
        out : Tensor (B, d_model)   — CLS-style mean-pooled representation
        """
        B, D = obs.shape
        # Pad to make obs_dim divisible by n_tokens
        if self.pad_dim > 0:
            obs = F.pad(obs, (0, self.pad_dim))
        # Reshape into tokens: (B, n_tokens, token_dim)
        tokens = obs.view(B, self.n_tokens, -1)
        x = self.input_proj(tokens)          # (B, n_tokens, d_model)
        x = self.pos_enc(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # Mean-pool across token dimension
        return x.mean(dim=1)                 # (B, d_model)


class SigFormerActor(nn.Module):
    """Actor network: maps observation → hedge ratio in [-1, 1].

    Parameters
    ----------
    obs_dim : int
        Dimension of the state observation.
    action_dim : int
        Dimension of the action (1 for scalar hedge ratio).
    d_model, n_layers, n_heads, d_ff, dropout : int/float
        SigFormer backbone hyperparameters.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 1,
        d_model: int = 256,
        n_layers: int = 5,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = SigFormerBackbone(obs_dim, d_model, n_layers, n_heads, d_ff, dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, action_dim),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return action in [-1, 1] for given state observation."""
        features = self.backbone(obs)
        return self.head(features)


class SigFormerCritic(nn.Module):
    """Twin critic for TD3: two independent Q-networks.

    Each Q-network maps (obs, action) → scalar Q-value.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 1,
        d_model: int = 256,
        n_layers: int = 5,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Q1
        self.backbone1 = SigFormerBackbone(
            obs_dim + action_dim, d_model, n_layers, n_heads, d_ff, dropout
        )
        self.head1 = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        # Q2
        self.backbone2 = SigFormerBackbone(
            obs_dim + action_dim, d_model, n_layers, n_heads, d_ff, dropout
        )
        self.head2 = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (Q1, Q2) estimates."""
        x = torch.cat([obs, action], dim=-1)
        q1 = self.head1(self.backbone1(x))
        q2 = self.head2(self.backbone2(x))
        return q1, q2

    def Q1(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return only Q1 (used for actor update)."""
        x = torch.cat([obs, action], dim=-1)
        return self.head1(self.backbone1(x))
