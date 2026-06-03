"""Phase II: Train the Conditional SigGAN on MA-fBM-conditioned paths.

IMPORTANT: Training this GAN is computationally expensive (~hours on GPU).
Confirm with the user before running.

Usage
-----
    python scripts/phase2_train_siggan.py --n_epochs 1000 --device cuda

The trained generator can then produce 10^6 synthetic market episodes for RL.
"""
import argparse
import sys
import os
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch

from src.mafbm import MAFBMCalibrator, MAFBMSimulator
from src.siggan import ARFNNGenerator, SignatureDiscriminator, CausalOTLoss, SigGANTrainer
from src.utils.data import MarketDataLoader, generate_synthetic_market


def main(args):
    device = args.device
    print(f"=== Phase II: SigGAN Training ===  device={device}\n")

    # ----------------------------------------------------------------
    # 1. Load MA-fBM calibrator
    # ----------------------------------------------------------------
    if os.path.exists("models/mafbm_calibrator.pkl"):
        with open("models/mafbm_calibrator.pkl", "rb") as f:
            calibrator = pickle.load(f)
        print("Loaded calibrator from models/mafbm_calibrator.pkl")
    else:
        print("Calibrator not found. Running Phase I first ...")
        calibrator = MAFBMCalibrator(K=10, T=1.0, r=100.0)
        calibrator.calibrate()

    simulator = MAFBMSimulator(calibrator, dt=1.0 / 252.0, n_steps=args.n_steps, seed=0)

    # ----------------------------------------------------------------
    # 2. Load real market data
    # ----------------------------------------------------------------
    print("Loading market data ...")
    if args.synthetic_data:
        print("  Using synthetic GBM data (no real data download required)")
        spot_paths, iv_surfaces = generate_synthetic_market(
            n_paths=args.n_real_paths,
            n_steps=args.n_steps,
            seed=42,
        )
    else:
        loader = MarketDataLoader(
            ticker="^GSPC",
            start=args.train_start,
            end=args.train_end,
            cache_dir="data/",
        )
        spot_paths, iv_surfaces = loader.construct_path_windows(window=args.n_steps)
        print(f"  Loaded {len(spot_paths)} historical windows.")

    # Build real data tensor: [log_spot_returns, avg_iv]
    # Shape: (N, n_steps, output_dim)
    log_returns = np.diff(spot_paths, axis=1)         # (N, n_steps)
    avg_iv = iv_surfaces[:, 1:, :, :].mean(axis=(-1, -2))  # (N, n_steps)
    real_data = np.stack([log_returns, avg_iv], axis=-1)  # (N, n_steps, 2)
    output_dim = real_data.shape[-1]
    print(f"  Real data shape: {real_data.shape}")

    # Normalise each channel to zero mean, unit std so the GAN learns at a
    # consistent scale.  Stats saved alongside the model for denormalisation.
    data_mean = real_data.mean(axis=(0, 1), keepdims=True)   # (1, 1, 2)
    data_std  = real_data.std(axis=(0, 1), keepdims=True) + 1e-8
    real_data_norm = (real_data - data_mean) / data_std
    print(f"  Channel means (before norm): {data_mean.squeeze().tolist()}")
    print(f"  Channel stds  (before norm): {data_std.squeeze().tolist()}")

    # ----------------------------------------------------------------
    # 3. Construct networks
    # ----------------------------------------------------------------
    noise_dim = 10  # K OU process states

    gen = ARFNNGenerator(noise_dim=noise_dim, output_dim=output_dim,
                         hidden_dim=256, n_layers=4, dropout=0.1)
    disc = SignatureDiscriminator(in_channels=output_dim, sig_depth=3,
                                  hidden_dim=256, n_layers=3)
    cot = CausalOTLoss(in_channels=output_dim, J=2, hidden_dim=64, lambda_cot=1.0)

    n_gen = sum(p.numel() for p in gen.parameters())
    n_disc = sum(p.numel() for p in disc.parameters())
    print(f"\n  Generator params:      {n_gen:,}")
    print(f"  Discriminator params:  {n_disc:,}")
    print(f"  COT penalty params:    {sum(p.numel() for p in cot.parameters()):,}\n")

    # ----------------------------------------------------------------
    # 4. Train
    # ----------------------------------------------------------------
    config = {
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "lr_gen": 1e-4,
        "lr_disc": 1e-4,
        "n_critic_steps": 5,
        "lambda_gp": 10.0,
    }
    trainer = SigGANTrainer(
        generator=gen,
        discriminator=disc,
        cot_loss=cot,
        real_data=real_data_norm,   # train on normalised data
        mafbm_simulator=simulator,
        hurst_range=(0.05, 0.45),
        device=device,
        config=config,
        save_dir="models/siggan/",
    )

    print(f"Starting SigGAN training for {args.n_epochs} epochs ...")
    history = trainer.train(verbose=True)

    # ----------------------------------------------------------------
    # 5. Generate synthetic dataset for RL
    # ----------------------------------------------------------------
    print(f"\nGenerating {args.n_generated:,} synthetic market paths ...")
    synthetic_norm = trainer.generate(args.n_generated)   # (N, T, 2), normalised scale
    # Denormalise back to natural units (log-returns, avg-IV)
    synthetic = synthetic_norm * data_std[0] + data_mean[0]
    os.makedirs("data/synthetic", exist_ok=True)
    np.save("data/synthetic/market_paths.npy", synthetic)
    # Save normalisation stats for reference
    np.save("data/synthetic/norm_mean.npy", data_mean)
    np.save("data/synthetic/norm_std.npy", data_std)
    print(f"Saved synthetic paths to data/synthetic/market_paths.npy")
    print(f"  Generated log_returns min/max/mean: {synthetic[:,:,0].min():.4f} / {synthetic[:,:,0].max():.4f} / {synthetic[:,:,0].mean():.4f}")
    print(f"  Generated avg_iv     min/max/mean: {synthetic[:,:,1].min():.4f} / {synthetic[:,:,1].max():.4f} / {synthetic[:,:,1].mean():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase II: SigGAN training")
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_steps", type=int, default=63)
    parser.add_argument("--n_real_paths", type=int, default=5000)
    parser.add_argument("--n_generated", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cpu",
                        help="Compute device: cpu or cuda")
    parser.add_argument("--train_start", type=str, default="2015-01-01")
    parser.add_argument("--train_end", type=str, default="2021-12-31")
    parser.add_argument("--synthetic_data", action="store_true",
                        help="Use synthetic GBM data instead of real market data")
    args = parser.parse_args()
    main(args)
