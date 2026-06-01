"""Phase III-B: Train the naive TD3 baseline (no synthetic data augmentation).

This script trains a SigFormer TD3 agent on plain GBM paths only, without
any MA-fBM or SigGAN augmentation.  The resulting agent is the baseline for
Phase V comparisons: it should underperform the SigGAN-augmented agent on
rough-regime test paths.

Usage
-----
    python scripts/phase3b_train_naive_td3.py --n_episodes 5000
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch

from src.rl import HedgingEnv, SigFormerActor, SigFormerCritic, TD3Agent
from src.utils.data import generate_synthetic_market


def main(args):
    device = args.device
    print(f"=== Phase III-B: Naive TD3 Baseline ===  device={device}\n")

    # ----------------------------------------------------------------
    # 1. Generate plain GBM training paths (no rough augmentation)
    # ----------------------------------------------------------------
    print("Generating GBM paths (no MA-fBM augmentation) ...")
    log_spot, iv_surfaces = generate_synthetic_market(
        n_paths=args.n_train_paths,
        n_steps=63,
        S0=100.0,
        mu=0.05,
        sigma_base=0.20,
        seed=42,
    )
    print(f"  Spot paths shape: {log_spot.shape}")
    print(f"  IV surfaces shape: {iv_surfaces.shape}")

    # ----------------------------------------------------------------
    # 2. Build environment
    # ----------------------------------------------------------------
    env = HedgingEnv(
        market_paths=log_spot,
        iv_surfaces=iv_surfaces,
        option_params={"strike": 100.0, "ttm": 63},
        history_window=20,
        sig_depth=3,
        transaction_cost=0.001,
    )
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"\n  Observation dim: {obs_dim}")
    print(f"  Action dim:      {action_dim}")

    # ----------------------------------------------------------------
    # 3. Build SigFormer actor / critic
    # ----------------------------------------------------------------
    arch_cfg = {
        "d_model": 128,
        "n_layers": 3,
        "n_heads": 4,
        "d_ff": 512,
        "dropout": 0.1,
    }
    actor = SigFormerActor(obs_dim=obs_dim, action_dim=action_dim, **arch_cfg)
    critic = SigFormerCritic(obs_dim=obs_dim, action_dim=action_dim, **arch_cfg)

    n_actor = sum(p.numel() for p in actor.parameters())
    n_critic = sum(p.numel() for p in critic.parameters())
    print(f"\n  Actor params:  {n_actor:,}")
    print(f"  Critic params: {n_critic:,}")

    # ----------------------------------------------------------------
    # 4. Train TD3
    # ----------------------------------------------------------------
    td3_cfg = {
        "gamma": 0.99,
        "tau": 0.005,
        "policy_noise": 0.2,
        "noise_clip": 0.5,
        "policy_delay": 2,
        "lr_actor": 3e-4,
        "lr_critic": 3e-4,
        "buffer_size": 500_000,
        "batch_size": 256,
        "exploration_noise": 0.1,
    }

    agent = TD3Agent(
        actor=actor,
        critic=critic,
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=td3_cfg,
        device=device,
    )

    os.makedirs("models/rl_naive", exist_ok=True)
    save_path = "models/rl_naive/best_agent.pt"

    print(f"\nStarting naive TD3 training for {args.n_episodes} episodes ...")
    stats = agent.train(
        env=env,
        n_episodes=args.n_episodes,
        warmup_steps=args.warmup_steps,
        verbose=True,
        save_path=save_path,
    )

    print(f"\nTraining complete. Best model saved to {save_path}")
    final_avg = np.mean(stats["episode_reward"][-100:])
    print(f"Final 100-episode avg reward: {final_avg:+.4f}")

    # Save training curve
    os.makedirs("results", exist_ok=True)
    np.save("results/naive_td3_rewards.npy", np.array(stats["episode_reward"]))
    print("Training rewards saved to results/naive_td3_rewards.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase III-B: Naive TD3 baseline")
    parser.add_argument("--n_episodes", type=int, default=5000)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--n_train_paths", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    main(args)
