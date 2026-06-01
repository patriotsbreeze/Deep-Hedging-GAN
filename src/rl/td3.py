"""Twin Delayed Deep Deterministic Policy Gradient (TD3) agent.

TD3 improvements over DDPG:
  1. Twin critics — take the minimum Q-value to reduce overestimation bias.
  2. Delayed policy update — actor updated every `policy_delay` critic steps.
  3. Target policy smoothing — add clipped noise to target actions during
     critic update to prevent exploitation of Q-function spikes.

The agent supports both SigFormer and fRNN actor/critic networks.

References
----------
Fujimoto et al. (2018), "Addressing Function Approximation Error in
Actor-Critic Methods."
"""
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Optional, Tuple, Type, Union
import random


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-size circular buffer storing (s, a, r, s', done) transitions."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, batch_size)
        return (
            torch.tensor(self.obs[idx], device=self.device),
            torch.tensor(self.actions[idx], device=self.device),
            torch.tensor(self.rewards[idx], device=self.device),
            torch.tensor(self.next_obs[idx], device=self.device),
            torch.tensor(self.dones[idx], device=self.device),
        )

    def __len__(self) -> int:
        return self.size


# ---------------------------------------------------------------------------
# TD3 Agent
# ---------------------------------------------------------------------------

class TD3Agent:
    """TD3 agent supporting SigFormer or fRNN actor/critic networks.

    Parameters
    ----------
    actor : nn.Module
        Actor network (SigFormerActor or FRNNActor).
    critic : nn.Module
        Twin-critic network (SigFormerCritic or FRNNCritic).
    obs_dim : int
    action_dim : int
    config : dict
        TD3 hyperparameters (gamma, tau, policy_noise, noise_clip,
        policy_delay, lr_actor, lr_critic, buffer_size, batch_size).
    device : str
    """

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        obs_dim: int,
        action_dim: int,
        config: Optional[dict] = None,
        device: str = "cpu",
    ):
        cfg = config or {}
        self.gamma = cfg.get("gamma", 0.99)
        self.tau = cfg.get("tau", 0.005)
        self.policy_noise = cfg.get("policy_noise", 0.2)
        self.noise_clip = cfg.get("noise_clip", 0.5)
        self.policy_delay = cfg.get("policy_delay", 2)
        self.exploration_noise = cfg.get("exploration_noise", 0.1)
        self.batch_size = cfg.get("batch_size", 256)
        self.device = device

        self.actor = actor.to(device)
        self.actor_target = copy.deepcopy(actor).to(device)
        self.critic = critic.to(device)
        self.critic_target = copy.deepcopy(critic).to(device)

        # Freeze targets (updated via soft polyak averaging)
        for p in self.actor_target.parameters():
            p.requires_grad_(False)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        lr_a = cfg.get("lr_actor", 3e-4)
        lr_c = cfg.get("lr_critic", 3e-4)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_c)

        self.buffer = ReplayBuffer(
            capacity=cfg.get("buffer_size", 1_000_000),
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
        )

        self._update_step = 0
        self.training_stats = {
            "critic_loss": [],
            "actor_loss": [],
            "episode_reward": [],
            "episode_pnl": [],
        }

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        """Select an action for the given observation.

        Parameters
        ----------
        obs : ndarray of shape (obs_dim,)
        explore : bool
            If True, add Gaussian exploration noise.

        Returns
        -------
        action : ndarray of shape (action_dim,)
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        # Handle both SigFormer (returns action) and fRNN (returns action, hidden)
        out = self.actor(obs_t)
        if isinstance(out, tuple):
            action = out[0]
        else:
            action = out
        action = action.squeeze(0).cpu().numpy()

        if explore:
            noise = np.random.randn(*action.shape) * self.exploration_noise
            action = np.clip(action + noise, -1.0, 1.0)
        return action

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self) -> None:
        """Perform one TD3 update step (critic + optionally actor)."""
        if len(self.buffer) < self.batch_size:
            return

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        # --- Critic update ---
        with torch.no_grad():
            # Target policy smoothing
            target_out = self.actor_target(next_obs)
            if isinstance(target_out, tuple):
                target_actions = target_out[0]
            else:
                target_actions = target_out

            noise = (
                torch.randn_like(target_actions) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            target_actions = (target_actions + noise).clamp(-1.0, 1.0)

            # Twin target Q-values
            q1_target, q2_target = self.critic_target(next_obs, target_actions)
            q_target = rewards + self.gamma * (1.0 - dones) * torch.min(q1_target, q2_target)

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.opt_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.opt_critic.step()

        self._update_step += 1
        self.training_stats["critic_loss"].append(critic_loss.item())

        # --- Delayed actor update ---
        if self._update_step % self.policy_delay == 0:
            actor_out = self.actor(obs)
            if isinstance(actor_out, tuple):
                actor_actions = actor_out[0]
            else:
                actor_actions = actor_out

            actor_loss = -self.critic.Q1(obs, actor_actions).mean()

            self.opt_actor.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.opt_actor.step()

            # Soft target updates
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

            self.training_stats["actor_loss"].append(actor_loss.item())

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for p, p_tgt in zip(source.parameters(), target.parameters()):
            p_tgt.data.mul_(1.0 - self.tau).add_(p.data * self.tau)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        env,
        n_episodes: int = 10_000,
        warmup_steps: int = 10_000,
        updates_per_step: int = 1,
        verbose: bool = True,
        save_path: Optional[str] = None,
    ) -> dict:
        """Run the full TD3 training loop.

        Parameters
        ----------
        env : HedgingEnv (Gymnasium-compatible)
        n_episodes : int
        warmup_steps : int
            Steps with random actions before gradient updates begin.
        updates_per_step : int
            Gradient updates per environment step.
        verbose : bool
        save_path : str, optional
            Path to save the best model checkpoint.

        Returns
        -------
        training_stats : dict
        """
        total_steps = 0
        best_reward = -np.inf

        for ep in range(1, n_episodes + 1):
            obs, _ = env.reset()
            ep_reward = 0.0
            ep_pnl = 0.0
            done = False

            while not done:
                if total_steps < warmup_steps:
                    action = env.action_space.sample()
                else:
                    action = self.select_action(obs, explore=True)

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                self.buffer.add(obs, action, reward, next_obs, float(done))
                obs = next_obs
                ep_reward += reward
                ep_pnl += info.get("total_pnl", 0.0)
                total_steps += 1

                if total_steps >= warmup_steps:
                    for _ in range(updates_per_step):
                        self.update()

            self.training_stats["episode_reward"].append(ep_reward)
            self.training_stats["episode_pnl"].append(ep_pnl)

            if save_path and ep_reward > best_reward:
                best_reward = ep_reward
                self.save(save_path)

            if verbose and ep % max(1, n_episodes // 20) == 0:
                recent = self.training_stats["episode_reward"][-100:]
                print(
                    f"Episode {ep:6d}/{n_episodes}  "
                    f"Reward={ep_reward:+.2f}  "
                    f"Avg100={np.mean(recent):+.2f}  "
                    f"Steps={total_steps}"
                )

        return self.training_stats

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.opt_actor.load_state_dict(ckpt["opt_actor"])
        self.opt_critic.load_state_dict(ckpt["opt_critic"])
