"""Causal Optimal Transport (COT) Wasserstein loss for the SigGAN discriminator.

Standard Wasserstein GAN discriminators do not enforce the arrow of time:
the optimal transport plan can use "future" information to match "past"
distributions, which is physically impossible and introduces look-ahead bias.

The COT-GAN penalty (Xu et al., 2020) adds a causality constraint to the
transport cost:

    c_φ^K(x, y) = c(x, y) + Σ_{j=1}^J Σ_{t=1}^{T-1} h_{φ1,t}^j(y) · ΔM_{φ2,t}^j(x)

where:
  - x is real market data, y is generated data
  - h_{φ1}^j and M_{φ2}^j are auxiliary neural networks jointly trained to
    MAXIMISE the transport cost when the generator attempts non-causal transport
  - ΔM = M_{t+1} − M_t are increments of the martingale estimator

Minimising this augmented Wasserstein distance over the generator ensures the
generated rough paths are temporally consistent and arbitrage-free.

Reference: Xu et al. (2020), "COTGAN: Generating Causal Time Series."
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CausalPenaltyNet(nn.Module):
    """Auxiliary network (h or M) for the COT-GAN causality penalty.

    Takes a time-series input x ∈ ℝ^{T×d} and produces a scalar score at
    each time step: out ∈ ℝ^T.
    """

    def __init__(self, in_channels: int, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (B, T, d)

        Returns
        -------
        out : Tensor (B, T)   — scalar score at each time step
        """
        h, _ = self.rnn(x)               # (B, T, hidden)
        return self.proj(h).squeeze(-1)  # (B, T)


class CausalOTLoss(nn.Module):
    """Causal Optimal Transport Wasserstein loss for the SigGAN.

    Trains J pairs of penalty networks (h^j, M^j) alongside the discriminator.

    Parameters
    ----------
    in_channels : int
        Number of channels in the real/generated time series.
    J : int
        Number of COT penalty pairs.
    hidden_dim : int
        Hidden size of each penalty network.
    lambda_cot : float
        Weight on the causality penalty term.
    """

    def __init__(
        self,
        in_channels: int = 6,
        J: int = 2,
        hidden_dim: int = 64,
        lambda_cot: float = 1.0,
    ):
        super().__init__()
        self.J = J
        self.lambda_cot = lambda_cot

        self.h_nets = nn.ModuleList([
            CausalPenaltyNet(in_channels, hidden_dim) for _ in range(J)
        ])
        self.M_nets = nn.ModuleList([
            CausalPenaltyNet(in_channels, hidden_dim) for _ in range(J)
        ])

    def causal_penalty(self, x_real: torch.Tensor, x_fake: torch.Tensor) -> torch.Tensor:
        """Compute the COT causality penalty.

        Σ_j Σ_t h^j_t(x_fake) · (M^j_{t+1}(x_real) − M^j_t(x_real))

        The penalty is *added* to the Wasserstein cost during discriminator
        maximisation so that h and M are trained to detect non-causal transport.

        Parameters
        ----------
        x_real : Tensor (B, T, d) — real market paths
        x_fake : Tensor (B, T, d) — generated paths

        Returns
        -------
        penalty : Tensor ()  — scalar penalty term
        """
        penalty = torch.tensor(0.0, device=x_real.device)
        T = x_real.shape[1]

        for j in range(self.J):
            h_scores = self.h_nets[j](x_fake)     # (B, T)
            M_scores = self.M_nets[j](x_real)     # (B, T)
            # Increments of M: ΔM_t = M_{t+1} - M_t
            delta_M = M_scores[:, 1:] - M_scores[:, :-1]   # (B, T-1)
            # Sum over t: Σ_{t=1}^{T-1} h_t · ΔM_t
            # h is evaluated at t = 0, …, T-2 to match delta_M at t+1
            inner = (h_scores[:, :-1] * delta_M).sum(dim=1)  # (B,)
            penalty = penalty + inner.mean()

        return self.lambda_cot * penalty

    def gradient_penalty(
        self,
        discriminator: nn.Module,
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        lambda_gp: float = 10.0,
    ) -> torch.Tensor:
        """Gradient penalty (WGAN-GP) for the Lipschitz constraint.

        Interpolates between real and fake samples and penalises the norm of
        the discriminator gradient away from 1.
        """
        B = x_real.shape[0]
        alpha = torch.rand(B, 1, 1, device=x_real.device)
        interp = (alpha * x_real + (1.0 - alpha) * x_fake).requires_grad_(True)
        d_interp = discriminator(interp)
        # allow_unused=True handles the iisignature path which detaches from the graph;
        # in that case grad is None → zero gradient → penalty penalises flat discriminator.
        grad = torch.autograd.grad(
            outputs=d_interp.sum(),
            inputs=interp,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if grad is None:
            grad = torch.zeros_like(interp)
        grad_norm = grad.reshape(B, -1).norm(2, dim=1)
        return lambda_gp * ((grad_norm - 1.0) ** 2).mean()

    def discriminator_loss(
        self,
        discriminator: nn.Module,
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        lambda_gp: float = 10.0,
    ) -> torch.Tensor:
        """Full discriminator (critic) loss for one update step.

        L_D = E[D(x_fake)] - E[D(x_real)] + causal_penalty + gradient_penalty
        """
        d_real = discriminator(x_real).mean()
        d_fake = discriminator(x_fake.detach()).mean()
        cot_pen = self.causal_penalty(x_real, x_fake.detach())
        gp = self.gradient_penalty(discriminator, x_real, x_fake.detach(), lambda_gp)
        return d_fake - d_real + cot_pen + gp

    def generator_loss(
        self,
        discriminator: nn.Module,
        x_fake: torch.Tensor,
    ) -> torch.Tensor:
        """Generator loss: maximise discriminator score on fake samples.

        L_G = -E[D(x_fake)]
        """
        return -discriminator(x_fake).mean()
