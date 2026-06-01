"""Phase III: Train the SigFormer TD3 hedging agent.

IMPORTANT: RL training is computationally expensive (~hours).
Confirm with the user before running.

Usage
-----
    python scripts/phase3_train_rl.py --arch sigformer --n_episodes 10000
    python scripts/phase3_train_rl.py --arch frnn --n_episodes 10000
"""
import argparse
import sys
import os
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch

from src.mafbm import MAFBMCalibrator, MAFBMSimulator
from src.rl import HedgingEnv, HedgingReward, SigFormerActor, SigFormerCritic, FRNNActor, FRNNCritic, TD3Agent
from src.utils.data import generate_synthetic_market, gan_output_to_env_inputs


def build_env(spot_paths, iv_surfaces, reward_cfg=None):
    """Construct a HedgingEnv from market path arrays."""
    return HedgingEnv(
        market_paths=spot_paths,
        iv_surfaces=iv_surfaces,
        option_params={"strike": np.exp(spot_paths[:, 0].mean()), "ttm": 63},
        reward_cfg=reward_cfg,
        history_window=20,
        sig_depth=3,
        transaction_cost=0.001,
    )


def main(args):
    device = args.device
    print(f"=== Phase III: TD3 Training ({args.arch}) ===  device={device}\n")

    # ----------------------------------------------------------------
    # 1. Load or generate training market paths
    # ----------------------------------------------------------------
    synthetic_path = "data/synthetic/market_paths.npy"
    if os.path.exists(synthetic_path):
        print(f"Loading synthetic paths from {synthetic_path} ...")
        raw = np.load(synthetic_path)   # (N, T, 2): [log_returns, avg_iv]
        log_spot, iv_surfaces = gan_output_to_env_inputs(raw, S0=100.0, n_strikes=1, n_maturities=1)
    else:
        print("Synthetic data not found; generating GBM paths for testing ...")
        log_spot, iv_surfaces = generate_synthetic_market(
            n_paths=args.n_train_paths,
            n_steps=63,
            seed=42,
        )

    print(f"  Spot paths shape: {log_spot.shape}")
    print(f"  IV surfaces shape: {iv_surfaces.shape}")

    # ----------------------------------------------------------------
    # 2. Build environment
    # ----------------------------------------------------------------
    env = build_env(log_spot, iv_surfaces)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"\n  Observation dim: {obs_dim}")
    print(f"  Action dim:      {action_dim}")

    # ----------------------------------------------------------------
    # 3. Build actor / critic
    # ----------------------------------------------------------------
    arch_cfg = {
        "d_model": 128,
        "n_layers": 3,
        "n_heads": 4,
        "d_ff": 512,
        "dropout": 0.1,
    }

    if args.arch == "sigformer":
        actor = SigFormerActor(obs_dim=obs_dim, action_dim=action_dim, **arch_cfg)
        critic = SigFormerCritic(obs_dim=obs_dim, action_dim=action_dim, **arch_cfg)
    elif args.arch == "frnn":
        actor = FRNNActor(obs_dim=obs_dim, action_dim=action_dim,
                          hidden_dim=256, n_layers=2, dropout=0.1)
        critic = FRNNCritic(obs_dim=obs_dim, action_dim=action_dim,
                            hidden_dim=256, n_layers=2, dropout=0.1)
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")

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

    os.makedirs(f"models/rl_{args.arch}", exist_ok=True)
    save_path = f"models/rl_{args.arch}/best_agent.pt"

    print(f"\nStarting TD3 training for {args.n_episodes} episodes ...")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase III: RL training")
    parser.add_argument("--arch", choices=["sigformer", "frnn"], default="sigformer")
    parser.add_argument("--n_episodes", type=int, default=5000)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--n_train_paths", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    main(args)
