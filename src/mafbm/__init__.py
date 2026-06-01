"""Phase I: Markov Approximation of Fractional Brownian Motion (MA-fBM)."""
from .weights import compute_optimal_weights, MAFBMCalibrator
from .simulation import MAFBMSimulator
from .validation import validate_hurst_exponent, compute_quadratic_variation

__all__ = [
    "compute_optimal_weights",
    "MAFBMCalibrator",
    "MAFBMSimulator",
    "validate_hurst_exponent",
    "compute_quadratic_variation",
]
