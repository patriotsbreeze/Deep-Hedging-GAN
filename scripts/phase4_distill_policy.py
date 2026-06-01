"""Phase IV: Distil the neural hedging policy via symbolic regression (PySR).

IMPORTANT: This requires a trained RL model from Phase III and takes
moderate compute (~minutes to hours depending on PySR iterations).

Usage
-----
    python scripts/phase4_distill_policy.py --arch sigformer \
        --model_path models/rl_sigformer/best_agent.pt \
        --n_samples 50000 --niterations 50
"""
import argparse
import sys
import os
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch

from src.rl import HedgingEnv, SigFormerActor, FRNNActor, TD3Agent, SigFormerCritic, FRNNCritic
from src.distillation import PolicyDistiller
from src.utils.data import generate_synthetic_market


def main(args):
    device = args.device
    print(f"=== Phase IV: Policy Distillation ===\n")

    # ----------------------------------------------------------------
    # 1. Build the test environment (held-out data or synthetic)
    # ----------------------------------------------------------------
    log_spot, iv_surfaces = generate_synthetic_market(
        n_paths=2000, n_steps=63, seed=99,
    )
    env = HedgingEnv(
        market_paths=log_spot,
        iv_surfaces=iv_surfaces,
        option_params={"strike": np.exp(log_spot[:, 0].mean()), "ttm": 63},
        history_window=20,
        sig_depth=3,
    )
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # ----------------------------------------------------------------
    # 2. Load trained actor
    # ----------------------------------------------------------------
    arch_cfg = dict(d_model=128, n_layers=3, n_heads=4, d_ff=512, dropout=0.1)
    if args.arch == "sigformer":
        actor = SigFormerActor(obs_dim=obs_dim, action_dim=action_dim, **arch_cfg)
        critic = SigFormerCritic(obs_dim=obs_dim, action_dim=action_dim, **arch_cfg)
    else:
        actor = FRNNActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=256, n_layers=2)
        critic = FRNNCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=256, n_layers=2)

    agent = TD3Agent(actor=actor, critic=critic, obs_dim=obs_dim,
                     action_dim=action_dim, device=device)

    if os.path.exists(args.model_path):
        agent.load(args.model_path)
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"WARNING: model path {args.model_path} not found. Using random weights.")

    # ----------------------------------------------------------------
    # 3. Collect state-action dataset
    # ----------------------------------------------------------------
    # Use interpretable state feature names
    feature_names = (
        ["inventory", "moneyness", "ttm_frac"]
        + [f"iv_{i}" for i in range(env.iv_flat_dim)]
        + [f"sig_{i}" for i in range(env.sig_dim)]
    )

    distiller = PolicyDistiller(
        actor=agent.actor,
        env=env,
        state_feature_names=feature_names,
        device=device,
    )

    print(f"Collecting {args.n_samples} state-action pairs ...")
    df = distiller.collect_dataset(n_samples=args.n_samples)
    df.to_csv("results/state_action_dataset.csv", index=False)
    print("Dataset saved to results/state_action_dataset.csv")

    # ----------------------------------------------------------------
    # 4. Symbolic regression on interpretable features
    # ----------------------------------------------------------------
    # Focus on the 3 most interpretable features
    feature_cols = ["inventory", "moneyness", "ttm_frac"]
    if "iv_0" in df.columns:
        feature_cols.append("iv_0")  # first strike IV as rough vol proxy

    print(f"\nRunning PySR on features: {feature_cols}")
    print(f"  niterations={args.niterations}, maxsize={args.maxsize}")

    sr_model = distiller.fit_symbolic(
        feature_cols=feature_cols,
        niterations=args.niterations,
        maxsize=args.maxsize,
        populations=15,
        verbosity=1,
    )

    # ----------------------------------------------------------------
    # 5. Report results
    # ----------------------------------------------------------------
    print("\n=== Pareto Frontier ===")
    print(distiller.pareto_frontier().to_string(index=False))

    print(f"\n=== Best Formula ===\n  {distiller.best_formula()}")

    accuracy = distiller.evaluate_accuracy(feature_cols=feature_cols)
    print(f"\n  R²  = {accuracy['r2']:.4f}")
    print(f"  MAE = {accuracy['mae']:.6f}")

    os.makedirs("results", exist_ok=True)
    distiller.save_equations("results/symbolic_equations.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase IV: Symbolic regression distillation")
    parser.add_argument("--arch", choices=["sigformer", "frnn"], default="sigformer")
    parser.add_argument("--model_path", type=str,
                        default="models/rl_sigformer/best_agent.pt")
    parser.add_argument("--n_samples", type=int, default=50_000)
    parser.add_argument("--niterations", type=int, default=50)
    parser.add_argument("--maxsize", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    main(args)
