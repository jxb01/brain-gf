"""Non-stationary stream: the system is asked to track regime A -> B -> A.

Each regime is a *different* 8-dim nonlinear map from sensor state to decision
target. Nothing tells the model that the world changed; the only signal is the
prediction error. This is the only benchmark worth running for these brains -
on a single stationary map every online least-squares fitter converges anyway,
so a static R^2 number proves nothing about adaptation.
"""

from __future__ import annotations

import numpy as np


def regime_a(x: np.ndarray) -> np.ndarray:
    """Smooth, almost-linear-plus-interaction map."""
    return np.sin(1.3 * x) + 0.5 * np.roll(x, -1) * np.roll(x, -2)


def regime_b(x: np.ndarray) -> np.ndarray:
    """Saturating, sign-splitting map - shares almost no structure with A."""
    return np.tanh(2.0 * x - np.roll(x, -3)) - 0.7 * np.cos(1.7 * x)


_REGIMES = {"A": regime_a, "B": regime_b}


def make_stream(
    segments=((("A", 900), ("B", 700), ("A", 700))),
    dim: int = 8,
    noise: float = 0.02,
    seed: int = 0,
):
    """Return (X, Y, regimes) as float64 arrays of shape (N, dim)."""
    rng = np.random.default_rng(seed)
    xs, ys, tags = [], [], []
    for name, n in segments:
        f = _REGIMES[name]
        x = rng.standard_normal((n, dim))
        y = f(x) + noise * rng.standard_normal((n, dim))
        xs.append(x)
        ys.append(y)
        tags.extend([name] * n)
    return np.concatenate(xs, 0), np.concatenate(ys, 0), tags


QUICK_SEGMENTS = (("A", 300), ("B", 250), ("A", 250))
DEFAULT_SEGMENTS = (("A", 900), ("B", 700), ("A", 700))
