"""Phase I: Calibrate MA-fBM weights and validate path statistics.

Run this script to:
  1. Compute L²-optimal MA-fBM weights for H ∈ {0.05, …, 0.45}.
  2. Simulate a small set of paths and validate the estimated Hurst exponent.
  3. Save the calibrated weights to disk for downstream use.

Usage
-----
    python scripts/phase1_calibrate_mafbm.py --validate --plot

NOTE: This script does NOT run training or large-scale simulation.
The --validate flag triggers ~10,000 path validation runs (lightweight).
"""
import argparse
import sys
import os
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt

from src.mafbm import MAFBMCalibrator, MAFBMSimulator, validate_hurst_exponent
from src.mafbm.validation import full_validation_report


def main(args):
    # ----------------------------------------------------------------
    # 1. Calibrate weights
    # ----------------------------------------------------------------
    print("=== Phase I: MA-fBM Calibration ===\n")

    hurst_grid = [round(0.05 * i, 2) for i in range(1, 10)]  # 0.05 to 0.45
    calibrator = MAFBMCalibrator(
        hurst_grid=hurst_grid,
        K=10,       # 10 OU processes
        T=1.0,      # 1-year L² window
        r=100.0,    # geometric spacing base
    )
    print(f"Calibrating MA-fBM weights for H ∈ {hurst_grid} ...")
    calibrator.calibrate()
    calibrator.summary()

    # Save calibrated weights
    os.makedirs("models", exist_ok=True)
    with open("models/mafbm_calibrator.pkl", "wb") as f:
        pickle.dump(calibrator, f)
    print("\nCalibrator saved to models/mafbm_calibrator.pkl")

    # ----------------------------------------------------------------
    # 2. (Optional) Validate path statistics
    # ----------------------------------------------------------------
    if args.validate:
        print("\n=== Validation ===")
        simulator = MAFBMSimulator(calibrator, dt=1.0 / 252.0, n_steps=252, seed=42)

        for H_test in [0.1, 0.25, 0.45]:
            print(f"\nSimulating {args.n_paths} paths for H = {H_test} ...")
            paths = simulator.simulate(H_test, n_paths=args.n_paths)
            weights = calibrator.get_weights(H_test)
            full_validation_report(paths, H_test, dt=1.0 / 252.0, kernel_weights=weights)

    # ----------------------------------------------------------------
    # 3. (Optional) Plot kernel approximation quality
    # ----------------------------------------------------------------
    if args.plot:
        os.makedirs("results", exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        t = np.linspace(1e-4, 1.0, 500)

        for ax, H in zip(axes, [0.1, 0.25, 0.45]):
            w = calibrator.get_weights(H)
            k_true = w.true_kernel(t)
            k_approx = w.approx_kernel(t)
            ax.plot(t, k_true, "k-", lw=2, label="True K(t)")
            ax.plot(t, k_approx, "r--", lw=1.5, label="MA-fBM approx")
            ax.set_title(f"H = {H:.2f}, L² residual = {w.residual:.2e}")
            ax.set_xlabel("t")
            ax.set_ylabel("K(t)")
            ax.legend(fontsize=8)

        fig.suptitle("Volterra Kernel Approximation Quality")
        plt.tight_layout()
        fig.savefig("results/kernel_approximation.png", dpi=150)
        print("\nKernel plot saved to results/kernel_approximation.png")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase I: MA-fBM calibration")
    parser.add_argument("--validate", action="store_true",
                        help="Run Hurst exponent validation on simulated paths")
    parser.add_argument("--n_paths", type=int, default=2000,
                        help="Number of paths for validation (default 2000)")
    parser.add_argument("--plot", action="store_true",
                        help="Plot kernel approximation quality")
    args = parser.parse_args()
    main(args)
