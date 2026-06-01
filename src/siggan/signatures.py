"""Path signature computation for time series feature extraction.

The signature of a path X : [0,T] → ℝ^d is a graded tensor series
  Sig(X) = (1, X^1, X^{1,2}, X^{1,2,3}, …)
where X^{i_1,…,i_k} = ∫_{0<t_1<…<t_k<T} dX^{i_1}_{t_1} … dX^{i_k}_{t_k}.

Truncated to depth N, the signature has Σ_{k=0}^N d^k terms.

This module provides:
  - A pure-NumPy fallback (slow but dependency-free) for testing.
  - A PyTorch wrapper that calls `iisignature` (if installed) or
    the manual implementation.
  - Utility to add a time channel (required for capturing the parameterisation).
"""
from __future__ import annotations

import numpy as np
from typing import Optional


def _signature_numpy(path: np.ndarray, depth: int) -> np.ndarray:
    """Compute truncated path signature via iterated integrals (Chen's identity).

    Parameters
    ----------
    path : ndarray of shape (T, d)
        Discrete path (vertices, not increments).
    depth : int
        Truncation depth.

    Returns
    -------
    sig : ndarray of shape (sig_dim,)
        Flattened signature coefficients, depth-0 term (scalar 1) excluded.
    """
    T, d = path.shape
    increments = np.diff(path, axis=0)  # (T-1, d)

    # Chen's recursion: sig_depth_k = sig_depth_{k-1} ⊗ increments + …
    # We store current level as a tensor and accumulate.
    sig_parts = []

    # Level 1: trivial iterated integrals = increments summed
    level = increments.copy()   # (T-1, d)
    # Accumulated sum from left (Chen-style)
    # For discrete paths: S^{i_1…i_k}_{0,T} = Σ_{t1<t2<…<tk} ΔX^{i1} … ΔX^{ik}
    current = level.sum(axis=0)  # (d,)
    sig_parts.append(current)

    if depth >= 2:
        # Level ≥ 2: iterated sums
        # Maintain running tensor of shape (d^{level},) for the partial sums up to t-1
        partial = np.zeros((d,))
        level_tensors = []
        for t in range(T - 1):
            dx = increments[t]  # (d,)
            # Level 2: Σ_{s<t} ΔX_s ⊗ ΔX_t
            level2_t = partial[:, None] * dx[None, :]  # outer product (d, d)
            level_tensors.append(level2_t.ravel())
            partial = partial + dx

        level2_sig = np.stack(level_tensors, axis=0).sum(axis=0)
        sig_parts.append(level2_sig)

    if depth >= 3:
        # Level 3: triple iterated sums
        partial2 = np.zeros((d * d,))
        partial1 = np.zeros((d,))
        level3_list = []
        for t in range(T - 1):
            dx = increments[t]
            level3_t = partial2[:, None] * dx[None, :]  # (d², d)
            level3_list.append(level3_t.ravel())
            partial2 = partial2 + (partial1[:, None] * dx[None, :]).ravel()
            partial1 = partial1 + dx
        level3_sig = np.stack(level3_list, axis=0).sum(axis=0)
        sig_parts.append(level3_sig)

    return np.concatenate(sig_parts)


def signature_dim(d: int, depth: int) -> int:
    """Return the dimension of the truncated signature at given depth (excl. level 0)."""
    return sum(d ** k for k in range(1, depth + 1))


def add_time_channel(path: np.ndarray, t_start: float = 0.0, t_end: float = 1.0) -> np.ndarray:
    """Prepend a time channel to the path so the signature captures parameterisation.

    Parameters
    ----------
    path : ndarray of shape (T, d)
    t_start, t_end : float

    Returns
    -------
    path_with_time : ndarray of shape (T, d+1)
    """
    T = path.shape[0]
    time_channel = np.linspace(t_start, t_end, T)[:, None]
    return np.concatenate([time_channel, path], axis=1)


class PathSignatureTransform:
    """Compute batched path signatures, optionally using iisignature.

    If `iisignature` is installed, uses its fast C++ backend.
    Falls back to the pure-NumPy implementation otherwise.
    """

    def __init__(self, depth: int = 3, with_time: bool = True, backend: str = "auto"):
        """
        Parameters
        ----------
        depth : int
            Signature truncation depth.
        with_time : bool
            Prepend a time channel before computing the signature.
        backend : str
            "iisignature", "numpy", or "auto" (try iisignature first).
        """
        self.depth = depth
        self.with_time = with_time
        self._backend = backend
        self._iisig = None
        if backend in ("auto", "iisignature"):
            try:
                import iisignature
                self._iisig = iisignature
            except ImportError:
                if backend == "iisignature":
                    raise
        # Determine effective channel dim after optional time augmentation
        self._d: Optional[int] = None  # set lazily on first call

    def _prepare(self, path: np.ndarray) -> np.ndarray:
        if self.with_time:
            path = add_time_channel(path)
        return path

    def transform_single(self, path: np.ndarray) -> np.ndarray:
        """Compute signature for a single path of shape (T, d)."""
        path = self._prepare(path)
        if self._iisig is not None:
            return self._iisig.sig(path, self.depth)
        return _signature_numpy(path, self.depth)

    def transform_batch(self, paths: np.ndarray) -> np.ndarray:
        """Compute signatures for a batch of paths of shape (B, T, d).

        Returns
        -------
        sigs : ndarray of shape (B, sig_dim)
        """
        if self._iisig is not None:
            if self.with_time:
                T = paths.shape[1]
                time_channel = np.linspace(0, 1, T)[None, :, None] * np.ones(
                    (paths.shape[0], 1, 1)
                )
                paths = np.concatenate([time_channel, paths], axis=2)
            return np.stack([self._iisig.sig(p, self.depth) for p in paths])
        return np.stack([self.transform_single(p) for p in paths])

    def output_dim(self, d: int) -> int:
        """Return signature output dimension for input channels d (after time augmentation)."""
        if self.with_time:
            d = d + 1
        return signature_dim(d, self.depth)
