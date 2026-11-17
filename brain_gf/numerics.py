"""Small numeric helpers. NumPy only - no autograd anywhere in this package."""

from __future__ import annotations

import numpy as np


def unit_cols(M: np.ndarray) -> np.ndarray:
    """Scale every COLUMN to unit L2 norm (feature directions on the sphere)."""
    n = np.linalg.norm(M, axis=0, keepdims=True)
    return M / np.maximum(n, 1e-9)


def unit_rows(M: np.ndarray) -> np.ndarray:
    """Scale every ROW to unit L2 norm (memory slots on the sphere)."""
    n = np.linalg.norm(M, axis=-1, keepdims=True)
    return M / np.maximum(n, 1e-9)


def orth_cols(M: np.ndarray) -> np.ndarray:
    """QR-orthonormalise columns so the frozen feature map covers the space evenly.

    Random features with orthogonal columns avoid the 'two columns point the same way'
    blind spot without needing any training.
    """
    q, r = np.linalg.qr(M)
    s = np.sign(np.diag(r))
    s[s == 0] = 1.0
    return q * s[None, :]


def feature_map(rng, n_in: int, n_out: int) -> np.ndarray:
    """Random frozen projection, shape (n_in, n_out).

    Orthogonal columns are only possible when n_out <= n_in (R^n_in holds at most
    n_in mutually orthogonal directions). Asking QR for more just silently returns
    n_in columns, which is how (8, 128) quietly became (8, 8). So: orthonormalise
    when it is meaningful, otherwise fall back to unit-length columns.
    """
    M = rng.standard_normal((n_in, n_out))
    if n_out <= n_in:
        return orth_cols(M)
    return unit_cols(M)


def softmax(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    s = s - np.max(s)
    e = np.exp(s)
    return e / e.sum()


def r2(y: np.ndarray, p: np.ndarray) -> float:
    """Coefficient of determination over all output dims pooled."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
        p = p[:, None]
    ss_res = ((y - p) ** 2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    # a constant-in-time target has no variance to explain
    ok = ss_tot > 1e-12
    if not np.any(ok):
        return float("nan")
    return float(1.0 - ss_res[ok].sum() / ss_tot[ok].sum())


def rolling_mse(y: np.ndarray, p: np.ndarray, window: int = 50) -> np.ndarray:
    """Causal rolling mean of per-step MSE (no future leakage)."""
    err = ((np.asarray(y, np.float64) - np.asarray(p, np.float64)) ** 2).mean(axis=1)
    c = np.cumsum(np.insert(err, 0, 0.0))
    n = err.shape[0]
    out = np.empty(n, dtype=np.float64)
    for t in range(n):
        lo = max(0, t - window + 1)
        out[t] = (c[t + 1] - c[lo]) / (t + 1 - lo)
    return out
