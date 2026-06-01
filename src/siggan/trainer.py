"""SigGAN training orchestration.

Training loop:
  1. Sample MA-fBM conditioning paths Z from the pre-calibrated simulator.
  2. Feed Z into the AR-FNN generator to produce fake market paths X̂.
  3. Update the discriminator (+ COT penalty nets) n_critic_steps times.
  4. Update the generator once.
  5. Repeat until discriminator can no longer distinguish real from fake.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .generator import ARFNNGenerator
from .discriminator import SignatureDiscriminator
from .cot_loss import CausalOTLoss


class SigGANTrainer:
    """Orchestrates the SigGAN training loop.

    Parameters
    ----------
    generator : ARFNNGenerator
    discriminator : SignatureDiscriminator
    cot_loss : CausalOTLoss
    real_data : ndarray of shape (N, T, d)
        Historical market paths for the real distribution.
    mafbm_simulator : MAFBMSimulator
        Pre-calibrated MA-fBM simulator for conditioning noise.
    hurst_range : tuple(float, float)
        Range of Hurst parameters to sample from during training.
    device : str
    config : dict
        Training hyperparameters (lr, batch_size, n_epochs, …).
    save_dir : str, optional
        Directory to save model checkpoints.
    """

    def __init__(
        self,
        generator: ARFNNGenerator,
        discriminator: SignatureDiscriminator,
        cot_loss: CausalOTLoss,
        real_data: np.ndarray,
        mafbm_simulator,          # MAFBMSimulator — avoid circular import
        hurst_range: tuple = (0.05, 0.45),
        device: str = "cpu",
        config: Optional[dict] = None,
        save_dir: Optional[str] = None,
    ):
        self.gen = generator.to(device)
        self.disc = discriminator.to(device)
        self.cot = cot_loss.to(device)
        self.simulator = mafbm_simulator
        self.hurst_range = hurst_range
        self.device = device
        self.save_dir = Path(save_dir) if save_dir else None

        cfg = config or {}
        self.n_epochs = cfg.get("n_epochs", 1000)
        self.batch_size = cfg.get("batch_size", 128)
        self.lr_gen = cfg.get("lr_gen", 1e-4)
        self.lr_disc = cfg.get("lr_disc", 1e-4)
        self.n_critic_steps = cfg.get("n_critic_steps", 5)
        self.lambda_gp = cfg.get("lambda_gp", 10.0)

        self.opt_gen = torch.optim.Adam(self.gen.parameters(), lr=self.lr_gen, betas=(0.5, 0.9))
        self.opt_disc = torch.optim.Adam(
            list(self.disc.parameters()) + list(self.cot.parameters()),
            lr=self.lr_disc, betas=(0.5, 0.9),
        )

        # Wrap real data in a DataLoader
        real_tensor = torch.tensor(real_data, dtype=torch.float32)
        self._real_loader = DataLoader(
            TensorDataset(real_tensor),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        self._real_iter = iter(self._real_loader)

        self.history = {"disc_loss": [], "gen_loss": [], "epoch_time": []}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_real(self) -> torch.Tensor:
        try:
            (batch,) = next(self._real_iter)
        except StopIteration:
            self._real_iter = iter(self._real_loader)
            (batch,) = next(self._real_iter)
        return batch.to(self.device)

    def _sample_noise(self, batch_size: int) -> torch.Tensor:
        """Sample MA-fBM conditioning paths with random H in hurst_range."""
        H = np.random.uniform(*self.hurst_range)
        # OU states: (batch_size, n_steps, K)
        _, ou_states = self.simulator.simulate_ou_states(H, batch_size)
        # ou_states has shape (batch_size, n_steps+1, K); drop t=0
        z = torch.tensor(ou_states[:, 1:, :], dtype=torch.float32).to(self.device)
        return z

    # ------------------------------------------------------------------
    # Training steps
    # ------------------------------------------------------------------

    def _discriminator_step(self) -> float:
        self.disc.train()
        self.gen.eval()

        x_real = self._sample_real()
        B = x_real.shape[0]
        z = self._sample_noise(B)

        with torch.no_grad():
            x_fake = self.gen(z)

        self.opt_disc.zero_grad()
        loss = self.cot.discriminator_loss(self.disc, x_real, x_fake, self.lambda_gp)
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.disc.parameters()) + list(self.cot.parameters()), 1.0
        )
        self.opt_disc.step()
        return loss.item()

    def _generator_step(self) -> float:
        self.gen.train()
        self.disc.eval()

        x_real = self._sample_real()
        B = x_real.shape[0]
        z = self._sample_noise(B)

        self.opt_gen.zero_grad()
        x_fake = self.gen(z)
        loss = self.cot.generator_loss(self.disc, x_fake)
        loss.backward()
        nn.utils.clip_grad_norm_(self.gen.parameters(), 1.0)
        self.opt_gen.step()
        return loss.item()

    # ------------------------------------------------------------------
    # Public training interface
    # ------------------------------------------------------------------

    def train(self, verbose: bool = True) -> dict:
        """Run the full GAN training loop.

        Returns
        -------
        history : dict with disc_loss, gen_loss, epoch_time lists.
        """
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.n_epochs + 1):
            t0 = time.time()
            disc_losses = []
            for _ in range(self.n_critic_steps):
                d_loss = self._discriminator_step()
                disc_losses.append(d_loss)
            g_loss = self._generator_step()

            elapsed = time.time() - t0
            self.history["disc_loss"].append(np.mean(disc_losses))
            self.history["gen_loss"].append(g_loss)
            self.history["epoch_time"].append(elapsed)

            if verbose and epoch % max(1, self.n_epochs // 20) == 0:
                print(
                    f"Epoch {epoch:5d}/{self.n_epochs}  "
                    f"D={np.mean(disc_losses):+.4f}  "
                    f"G={g_loss:+.4f}  "
                    f"t={elapsed:.1f}s"
                )

            if self.save_dir and epoch % max(1, self.n_epochs // 10) == 0:
                self._save_checkpoint(epoch)

        return self.history

    def _save_checkpoint(self, epoch: int) -> None:
        ckpt = {
            "epoch": epoch,
            "gen_state": self.gen.state_dict(),
            "disc_state": self.disc.state_dict(),
            "cot_state": self.cot.state_dict(),
            "opt_gen": self.opt_gen.state_dict(),
            "opt_disc": self.opt_disc.state_dict(),
        }
        path = self.save_dir / f"siggan_epoch_{epoch:05d}.pt"
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.gen.load_state_dict(ckpt["gen_state"])
        self.disc.load_state_dict(ckpt["disc_state"])
        self.cot.load_state_dict(ckpt["cot_state"])
        self.opt_gen.load_state_dict(ckpt["opt_gen"])
        self.opt_disc.load_state_dict(ckpt["opt_disc"])
        return ckpt["epoch"]

    @torch.no_grad()
    def generate(self, n_paths: int) -> np.ndarray:
        """Generate n_paths synthetic market paths using the trained generator."""
        self.gen.eval()
        z = self._sample_noise(n_paths)
        x_fake = self.gen(z)
        return x_fake.cpu().numpy()
