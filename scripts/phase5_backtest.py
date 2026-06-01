"""Phase V: Out-of-sample walk-forward validation.

Evaluates all strategies on the held-out 2022-2023 test period and prints
a comprehensive performance comparison table.

Usage
-----
    python scripts/phase5_backtest.py \
        --sigformer_path models/rl_sigformer/best_agent.pt \
        --naive_path models/rl_naive/best_agent.pt
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch

from src.rl import HedgingEnv, HedgingReward, SigFormerActor, SigFormerCritic, FRNNActor, FRNNCritic, TD3Agent
from src.validation import WalkForwardBacktester
from src.validation.metrics import compare_strategies
from src.utils.data import MarketDataLoader, generate_synthetic_market


def load_agent(arch, model_path, obs_dim, action_dim, device):
    arch_cfg = dict(d_model=128, n_layers=3, n_heads=4, d_ff=512, dropout=0.1)
    if arch == "sigformer":
        actor = SigFormerActor(obs_dim=obs_dim, action_dim=action_dim, **arch_cfg)
        critic = SigFormerCritic(obs_dim=obs_dim, action_dim=action_dim, **arch_cfg)
    else:
        actor = FRNNActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=256, n_layers=2)
        critic = FRNNCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=256, n_layers=2)

    agent = TD3Agent(actor=actor, critic=critic, obs_dim=obs_dim,
                     action_dim=action_dim, device=device)
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded {arch} from {model_path}")
    else:
        print(f"WARNING: {model_path} not found. Using random weights.")
    return agent


def main(args):
    device = args.device
    print("=== Phase V: Walk-Forward Backtest ===\n")

    # ----------------------------------------------------------------
    # 1. Load test market data (2022-2023 hold-out)
    # ----------------------------------------------------------------
    if args.synthetic_test:
        print("Using synthetic test data (2022-2023 hold-out simulation)")
        log_spot, iv_surfaces = generate_synthetic_market(
            n_paths=args.n_episodes,
            n_steps=63,
            seed=2022,
        )
    else:
        print("Loading 2022-2023 historical test data ...")
        try:
            loader = MarketDataLoader(
                ticker="^GSPC",
                start="2022-01-01",
                end="2023-12-31",
                cache_dir="data/",
            )
            log_spot, iv_surfaces = loader.construct_path_windows(window=63)
            print(f"  Loaded {len(log_spot)} test windows.")
        except Exception as e:
            print(f"Could not load historical data ({e}); using synthetic.")
            log_spot, iv_surfaces = generate_synthetic_market(
                n_paths=args.n_episodes, n_steps=63, seed=2022
            )

    # ----------------------------------------------------------------
    # 2. Build test environment
    # ----------------------------------------------------------------
    K = np.exp(log_spot[:, 0].mean())
    env = HedgingEnv(
        market_paths=log_spot,
        iv_surfaces=iv_surfaces,
        option_params={"strike": K, "ttm": 63},
        history_window=20,
        sig_depth=3,
    )
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    reward_fn = HedgingReward(dynamic_kappa=True)

    backtester = WalkForwardBacktester(
        test_env=env,
        reward_fn=reward_fn,
        K=K,
        ttm_years=63.0 / 252.0,
        dt=1.0 / 252.0,
    )

    # ----------------------------------------------------------------
    # 3. Evaluate all strategies
    # ----------------------------------------------------------------
    n_ep = min(args.n_episodes, env.n_episodes)

    # (a) Black-Scholes benchmark
    print("Running Black-Scholes benchmark ...")
    avg_iv = iv_surfaces[:n_ep, :, 0, 0]  # (N, T+1)
    backtester.run_black_scholes(log_spot[:n_ep], avg_iv, n_ep)

    # (b) SigFormer TD3 (trained with rough augmentation)
    if args.sigformer_path:
        print("Running SigFormer TD3 ...")
        agent_sf = load_agent("sigformer", args.sigformer_path, obs_dim, action_dim, device)
        backtester.run_neural_policy(agent_sf, n_ep, label="SigFormer TD3")

    # (c) Naive TD3 (trained without rough augmentation)
    if args.naive_path:
        print("Running Naive TD3 benchmark ...")
        agent_naive = load_agent("sigformer", args.naive_path, obs_dim, action_dim, device)
        backtester.run_naive_td3(agent_naive, n_ep)

    # ----------------------------------------------------------------
    # 4. Print results
    # ----------------------------------------------------------------
    backtester.print_comparison()

    os.makedirs("results", exist_ok=True)
    backtester.save_results("results/backtest_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase V: Walk-forward backtest")
    parser.add_argument("--sigformer_path", type=str,
                        default="models/rl_sigformer/best_agent.pt")
    parser.add_argument("--naive_path", type=str, default=None)
    parser.add_argument("--n_episodes", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--synthetic_test", action="store_true")
    args = parser.parse_args()
    main(args)
