"""Unit tests for Phase I: MA-fBM calibration and simulation.

These tests run in ~seconds (no GPU required, no training).
"""
import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.mafbm.grid import compute_gamma_grid
from src.mafbm.covariance import compute_covariance_matrix, compute_rhs
from src.mafbm.weights import compute_optimal_weights, MAFBMCalibrator
from src.mafbm.simulation import MAFBMSimulator
from src.mafbm.validation import estimate_hurst_exponent, validate_hurst_exponent


class TestGammaGrid:
    def test_shape(self):
        gammas = compute_gamma_grid(K=10, r=100.0)
        assert gammas.shape == (10,)

    def test_geometric_spacing(self):
        gammas = compute_gamma_grid(K=10, r=100.0)
        # Consecutive ratios should all equal r
        ratios = gammas[1:] / gammas[:-1]
        np.testing.assert_allclose(ratios, 100.0, rtol=1e-10)

    def test_center_near_one(self):
        # With K=11, n=6, k=6 gives r^0 = 1
        gammas = compute_gamma_grid(K=11, r=10.0)
        assert any(np.isclose(gammas, 1.0, rtol=1e-8))


class TestCovarianceMatrix:
    def test_symmetric(self):
        gammas = compute_gamma_grid(K=5, r=10.0)
        A = compute_covariance_matrix(gammas, T=1.0)
        np.testing.assert_allclose(A, A.T, atol=1e-12)

    def test_positive_definite(self):
        gammas = compute_gamma_grid(K=5, r=10.0)
        A = compute_covariance_matrix(gammas, T=1.0)
        eigvals = np.linalg.eigvalsh(A)
        assert np.all(eigvals > 0)

    def test_shape(self):
        gammas = compute_gamma_grid(K=7, r=10.0)
        A = compute_covariance_matrix(gammas, T=1.0)
        assert A.shape == (7, 7)


class TestRHSVector:
    def test_positive(self):
        gammas = compute_gamma_grid(K=5, r=10.0)
        b = compute_rhs(gammas, H=0.1, T=1.0)
        assert np.all(b > 0)

    def test_decreasing_with_gamma(self):
        # For small H (rough), the kernel K(t)=t^{H-1/2} is singular at 0.
        # Faster decaying exponentials (large γ) should have smaller ∫e^{-γt}t^{H-1/2}
        gammas = compute_gamma_grid(K=10, r=100.0)
        b = compute_rhs(gammas, H=0.1, T=1.0)
        # Not strictly monotone for all cases but the largest γ should be small
        assert b[-1] < b[0]


class TestOptimalWeights:
    @pytest.mark.parametrize("H", [0.1, 0.25, 0.45])
    def test_solve(self, H):
        w = compute_optimal_weights(H, K=10, T=1.0, r=100.0)
        assert w.H == H
        assert len(w.omega) == 10
        assert w.residual >= 0

    def test_residual_small(self):
        w = compute_optimal_weights(H=0.25, K=10, T=1.0, r=100.0)
        assert w.residual < 0.1

    def test_kernel_approximation(self):
        w = compute_optimal_weights(H=0.1, K=10, T=1.0, r=100.0)
        t = np.linspace(0.01, 1.0, 100)
        k_true = w.true_kernel(t)
        k_approx = w.approx_kernel(t)
        rel_err = np.abs(k_true - k_approx) / (np.abs(k_true) + 1e-8)
        # Allow up to 20% relative error (rough H is hardest to approximate)
        assert rel_err.mean() < 0.2


class TestMAFBMCalibrator:
    def test_calibrate(self):
        cal = MAFBMCalibrator(hurst_grid=[0.1, 0.25, 0.45], K=5, T=1.0)
        cal.calibrate()
        assert len(cal._cache) == 3

    def test_get_weights_snap(self):
        cal = MAFBMCalibrator(hurst_grid=[0.1, 0.3], K=5, T=1.0)
        cal.calibrate()
        # Requesting H=0.2 should snap to nearest (0.1 or 0.3)
        w = cal.get_weights(0.2)
        assert w.H in [0.1, 0.3]


class TestMAFBMSimulator:
    def test_shape(self):
        cal = MAFBMCalibrator(hurst_grid=[0.25], K=5, T=1.0)
        cal.calibrate()
        sim = MAFBMSimulator(cal, dt=1.0 / 252.0, n_steps=50, seed=0)
        paths = sim.simulate(H=0.25, n_paths=10)
        assert paths.shape == (10, 51)

    def test_starts_at_zero(self):
        cal = MAFBMCalibrator(hurst_grid=[0.25], K=5, T=1.0)
        cal.calibrate()
        sim = MAFBMSimulator(cal, dt=1.0 / 252.0, n_steps=50, seed=0)
        paths = sim.simulate(H=0.25, n_paths=100)
        np.testing.assert_allclose(paths[:, 0], 0.0, atol=1e-10)

    def test_hurst_recovery_coarse(self):
        """Rough sanity check: estimated H should be in the right ballpark."""
        cal = MAFBMCalibrator(hurst_grid=[0.1, 0.45], K=10, T=1.0)
        cal.calibrate()
        sim = MAFBMSimulator(cal, dt=1.0 / 252.0, n_steps=252, seed=7)

        for H_target in [0.1, 0.45]:
            paths = sim.simulate(H_target, n_paths=500)
            H_est, se = estimate_hurst_exponent(paths, dt=1.0 / 252.0, scales=5)
            # Accept wide tolerance given short paths
            assert abs(H_est - H_target) < 0.2, f"H_est={H_est:.3f}, target={H_target}"


class TestBlackScholes:
    def test_put_call_parity(self):
        from src.utils.black_scholes import bs_price
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        call = bs_price(S, K, T, r, sigma, call=True)
        put = bs_price(S, K, T, r, sigma, call=False)
        # Put-call parity: C - P = S - K * exp(-rT)
        parity = S - K * np.exp(-r * T)
        np.testing.assert_allclose(call - put, parity, rtol=1e-6)

    def test_delta_bounds(self):
        from src.utils.black_scholes import bs_delta
        S = np.array([80.0, 100.0, 120.0])
        delta = bs_delta(S, K=100.0, T=0.5, r=0.02, sigma=0.25, call=True)
        assert np.all(delta >= 0) and np.all(delta <= 1)
