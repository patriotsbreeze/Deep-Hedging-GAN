"""Unit tests for Phase III: RL environment, reward, and actor/critic architectures.

Tests cover:
  - HedgingEnv reset/step shapes and BS convention (log-spot → actual spot)
  - HedgingReward compute and dynamic kappa
  - SigFormerActor/Critic forward passes
  - FRNNActor/Critic forward passes and hidden-state threading
  - TD3Agent hidden-state preservation for fRNN
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from src.rl.env import HedgingEnv
from src.rl.reward import HedgingReward
from src.rl.sigformer import SigFormerActor, SigFormerCritic
from src.rl.frnn import FRNNActor, FRNNCritic, FRNNCore
from src.rl.td3 import TD3Agent
from src.utils.data import generate_synthetic_market


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env(n_paths=20, n_steps=10, n_strikes=2, n_maturities=2, seed=0):
    log_spot, iv_surfaces = generate_synthetic_market(
        n_paths=n_paths,
        n_steps=n_steps,
        n_strikes=n_strikes,
        n_maturities=n_maturities,
        seed=seed,
    )
    env = HedgingEnv(
        market_paths=log_spot,
        iv_surfaces=iv_surfaces,
        option_params={"strike": 100.0, "ttm": 10},
        history_window=5,
        sig_depth=2,
        transaction_cost=0.001,
    )
    return env


# ---------------------------------------------------------------------------
# HedgingEnv
# ---------------------------------------------------------------------------

class TestHedgingEnv:
    def test_reset_obs_shape(self):
        env = make_env()
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32

    def test_reset_inventory_zero(self):
        env = make_env()
        env.reset()
        assert env._inventory == 0.0

    def test_step_obs_shape(self):
        env = make_env()
        obs, _ = env.reset()
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert next_obs.shape == obs.shape

    def test_step_returns_float_reward(self):
        env = make_env()
        env.reset()
        action = np.array([0.5], dtype=np.float32)
        _, reward, _, _, _ = env.step(action)
        assert isinstance(reward, float)

    def test_step_info_keys(self):
        env = make_env()
        env.reset()
        _, _, _, _, info = env.step(np.array([0.0]))
        for key in ("option_pnl", "hedge_pnl", "transaction_cost", "total_pnl", "bs_delta"):
            assert key in info

    def test_episode_terminates(self):
        env = make_env(n_steps=5)
        env.reset()
        terminated = False
        for _ in range(10):
            _, _, terminated, _, _ = env.step(env.action_space.sample())
            if terminated:
                break
        assert terminated

    def test_full_episode_no_nan(self):
        env = make_env(n_steps=20)
        obs, _ = env.reset()
        assert not np.any(np.isnan(obs))
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            assert not np.any(np.isnan(obs)), "NaN in observation"
            assert np.isfinite(reward), f"Non-finite reward: {reward}"

    def test_bs_convention_uses_actual_spot(self):
        # BS delta for ATM option should be ~0.5; if we accidentally pass log-spot
        # to bs_delta the result would be wildly wrong (log(100)≈4.6 → near 1.0).
        from src.utils.black_scholes import bs_delta
        env = make_env(n_steps=5)
        env.reset()
        action = np.array([1.0])  # request full BS delta as target
        _, _, _, _, info = env.step(action)
        # bs_delta for actual spot ~100, K=100 should be in [0, 1]
        assert 0.0 <= info["bs_delta"] <= 1.0, (
            f"bs_delta={info['bs_delta']} suggests wrong spot convention"
        )

    def test_spot_history_stores_log_spot(self):
        env = make_env(n_steps=5)
        env.reset()
        env.step(np.array([0.0]))
        # Log-spot for S0=100 is ~4.6; values should be near log(100), not log(log(100))
        latest = env._spot_history[-1]
        assert 2.0 < latest < 8.0, f"Unexpected log-spot value {latest}; may be double-logged"


# ---------------------------------------------------------------------------
# HedgingReward
# ---------------------------------------------------------------------------

class TestHedgingReward:
    def test_compute_returns_float(self):
        rw = HedgingReward()
        r = rw.compute(0.01)
        assert isinstance(r, float)

    def test_negative_pnl_negative_reward(self):
        rw = HedgingReward(scale=10.0, base=0.0, kappa_base=2.0, dynamic_kappa=False)
        r = rw.compute(-1.0)
        assert r < 0.0

    def test_dynamic_kappa_escalates(self):
        rw = HedgingReward(dynamic_kappa=True, kappa_base=1.0, es_window=10)
        # Feed a series of large losses to fill the history
        for _ in range(15):
            rw.compute(-1.0)
        kappa_bad = rw.kappa()
        rw2 = HedgingReward(dynamic_kappa=True, kappa_base=1.0, es_window=10)
        for _ in range(15):
            rw2.compute(0.01)
        kappa_good = rw2.kappa()
        assert kappa_bad > kappa_good

    def test_reset_clears_history(self):
        rw = HedgingReward(dynamic_kappa=True)
        for _ in range(20):
            rw.compute(-1.0)
        rw.reset()
        assert len(rw._pnl_history) == 0


# ---------------------------------------------------------------------------
# SigFormer Actor / Critic
# ---------------------------------------------------------------------------

class TestSigFormerActor:
    def test_output_shape(self):
        obs_dim, action_dim, B = 50, 1, 8
        actor = SigFormerActor(obs_dim=obs_dim, action_dim=action_dim,
                               d_model=32, n_layers=1, n_heads=4, d_ff=64)
        obs = torch.randn(B, obs_dim)
        action = actor(obs)
        assert action.shape == (B, action_dim)

    def test_output_in_range(self):
        obs_dim, B = 40, 16
        actor = SigFormerActor(obs_dim=obs_dim, d_model=32, n_layers=1, n_heads=4, d_ff=64)
        obs = torch.randn(B, obs_dim)
        with torch.no_grad():
            action = actor(obs)
        assert action.min().item() >= -1.0 - 1e-6
        assert action.max().item() <= 1.0 + 1e-6

    def test_returns_tensor_not_tuple(self):
        actor = SigFormerActor(obs_dim=30, d_model=32, n_layers=1, n_heads=4, d_ff=64)
        obs = torch.randn(2, 30)
        out = actor(obs)
        assert isinstance(out, torch.Tensor)


class TestSigFormerCritic:
    def test_forward_shapes(self):
        obs_dim, action_dim, B = 50, 1, 6
        critic = SigFormerCritic(obs_dim=obs_dim, action_dim=action_dim,
                                 d_model=32, n_layers=1, n_heads=4, d_ff=64)
        obs = torch.randn(B, obs_dim)
        action = torch.randn(B, action_dim)
        q1, q2 = critic(obs, action)
        assert q1.shape == (B, 1)
        assert q2.shape == (B, 1)

    def test_q1_shape(self):
        obs_dim, action_dim, B = 40, 1, 5
        critic = SigFormerCritic(obs_dim=obs_dim, action_dim=action_dim,
                                 d_model=32, n_layers=1, n_heads=4, d_ff=64)
        obs = torch.randn(B, obs_dim)
        action = torch.randn(B, action_dim)
        q1 = critic.Q1(obs, action)
        assert q1.shape == (B, 1)

    def test_twin_critics_differ(self):
        obs_dim, action_dim, B = 40, 1, 4
        critic = SigFormerCritic(obs_dim=obs_dim, action_dim=action_dim,
                                 d_model=32, n_layers=1, n_heads=4, d_ff=64)
        obs = torch.randn(B, obs_dim)
        action = torch.randn(B, action_dim)
        q1, q2 = critic(obs, action)
        assert not torch.allclose(q1, q2)


# ---------------------------------------------------------------------------
# fRNN Actor / Critic
# ---------------------------------------------------------------------------

class TestFRNNActor:
    def test_output_shape(self):
        obs_dim, action_dim, B = 50, 1, 4
        actor = FRNNActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=32, n_layers=1)
        obs = torch.randn(B, obs_dim)
        action, hidden = actor(obs)
        assert action.shape == (B, action_dim)

    def test_output_in_range(self):
        actor = FRNNActor(obs_dim=40, hidden_dim=32, n_layers=1)
        obs = torch.randn(8, 40)
        with torch.no_grad():
            action, _ = actor(obs)
        assert action.min().item() >= -1.0 - 1e-6
        assert action.max().item() <= 1.0 + 1e-6

    def test_returns_tuple(self):
        actor = FRNNActor(obs_dim=30, hidden_dim=32, n_layers=1)
        obs = torch.randn(2, 30)
        out = actor(obs)
        assert isinstance(out, tuple) and len(out) == 2

    def test_hidden_state_propagates(self):
        # Output should differ when we pass a non-zero hidden state vs zero hidden
        obs_dim, B = 30, 2
        actor = FRNNActor(obs_dim=obs_dim, hidden_dim=32, n_layers=1)
        actor.eval()
        obs = torch.randn(B, obs_dim)
        with torch.no_grad():
            action1, hidden1 = actor(obs)           # first step: hidden=None (zeros)
            action2, hidden2 = actor(obs, hidden1)  # second step: pass hidden
        assert not torch.allclose(action1, action2)

    def test_hidden_shape(self):
        n_layers, hidden_dim, B = 2, 32, 4
        actor = FRNNActor(obs_dim=20, hidden_dim=hidden_dim, n_layers=n_layers)
        obs = torch.randn(B, 20)
        _, hidden = actor(obs)
        assert hidden.shape == (n_layers, B, hidden_dim)


class TestFRNNCritic:
    def test_forward_shapes(self):
        obs_dim, action_dim, B = 50, 1, 6
        critic = FRNNCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=32, n_layers=1)
        obs = torch.randn(B, obs_dim)
        action = torch.randn(B, action_dim)
        q1, q2 = critic(obs, action)
        assert q1.shape == (B, 1)
        assert q2.shape == (B, 1)

    def test_q1_shape(self):
        obs_dim, action_dim, B = 40, 1, 5
        critic = FRNNCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=32, n_layers=1)
        obs = torch.randn(B, obs_dim)
        action = torch.randn(B, action_dim)
        q1 = critic.Q1(obs, action)
        assert q1.shape == (B, 1)


# ---------------------------------------------------------------------------
# TD3Agent — hidden state for fRNN
# ---------------------------------------------------------------------------

class TestTD3AgentHiddenState:
    def test_sigformer_hidden_stays_none(self):
        obs_dim, action_dim = 40, 1
        actor = SigFormerActor(obs_dim=obs_dim, d_model=32, n_layers=1, n_heads=4, d_ff=64)
        critic = SigFormerCritic(obs_dim=obs_dim, d_model=32, n_layers=1, n_heads=4, d_ff=64)
        agent = TD3Agent(actor, critic, obs_dim=obs_dim, action_dim=action_dim)
        obs = np.random.randn(obs_dim).astype(np.float32)
        agent.select_action(obs, explore=False)
        agent.select_action(obs, explore=False)
        assert agent._actor_hidden is None  # SigFormer never sets hidden

    def test_frnn_hidden_updated_after_step(self):
        obs_dim, action_dim = 40, 1
        actor = FRNNActor(obs_dim=obs_dim, hidden_dim=32, n_layers=1)
        critic = FRNNCritic(obs_dim=obs_dim, hidden_dim=32, n_layers=1)
        agent = TD3Agent(actor, critic, obs_dim=obs_dim, action_dim=action_dim)
        obs = np.random.randn(obs_dim).astype(np.float32)
        assert agent._actor_hidden is None
        agent.select_action(obs, explore=False)
        assert agent._actor_hidden is not None  # hidden updated after first step

    def test_frnn_hidden_changes_output(self):
        obs_dim, action_dim = 40, 1
        actor = FRNNActor(obs_dim=obs_dim, hidden_dim=32, n_layers=1)
        critic = FRNNCritic(obs_dim=obs_dim, hidden_dim=32, n_layers=1)
        agent = TD3Agent(actor, critic, obs_dim=obs_dim, action_dim=action_dim)
        obs = np.random.randn(obs_dim).astype(np.float32)
        a1 = agent.select_action(obs, explore=False)  # step 1: hidden=None
        a2 = agent.select_action(obs, explore=False)  # step 2: hidden from step 1
        # Same obs but different hidden → different action
        assert not np.allclose(a1, a2)

    def test_reset_episode_clears_hidden(self):
        obs_dim, action_dim = 40, 1
        actor = FRNNActor(obs_dim=obs_dim, hidden_dim=32, n_layers=1)
        critic = FRNNCritic(obs_dim=obs_dim, hidden_dim=32, n_layers=1)
        agent = TD3Agent(actor, critic, obs_dim=obs_dim, action_dim=action_dim)
        obs = np.random.randn(obs_dim).astype(np.float32)
        agent.select_action(obs, explore=False)
        assert agent._actor_hidden is not None
        agent.reset_episode()
        assert agent._actor_hidden is None

    def test_action_shape(self):
        env = make_env(n_steps=5)
        obs, _ = env.reset()
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        actor = SigFormerActor(obs_dim=obs_dim, d_model=32, n_layers=1, n_heads=4, d_ff=64)
        critic = SigFormerCritic(obs_dim=obs_dim, d_model=32, n_layers=1, n_heads=4, d_ff=64)
        agent = TD3Agent(actor, critic, obs_dim=obs_dim, action_dim=action_dim)
        action = agent.select_action(obs.astype(np.float32), explore=False)
        assert action.shape == (action_dim,)
