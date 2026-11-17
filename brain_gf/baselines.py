"""Gradient-free reference learners. NumPy only - the whole package runs without torch.

    RidgeOracle - closed-form least squares on frozen features (offline ceiling)
    ESN         - frozen random reservoir + ridge readout (a memory baseline that
                  still needs no gradients)
    ARLag       - lagged linear features (the classical time-series baseline)

None of these is trained by backpropagation; they exist so the online learner can
be compared against something that is not itself learning by gradient descent.
"""

from __future__ import annotations

import numpy as np


class RidgeOracle:
    """Closed-form least squares on frozen features.

    Not an online learner. It answers: how much does the online write give up
    compared with fitting the whole regime at once?
    """

    def __init__(self, feature_fn=None, lam: float = 1e-3):
        self.feature_fn = feature_fn
        self.lam = lam
        self.W = None

    def fit(self, F: np.ndarray, Y: np.ndarray):
        A = F.T @ F + self.lam * np.eye(F.shape[1])
        self.W = np.linalg.solve(A, F.T @ Y)
        return self

    def predict(self, F: np.ndarray) -> np.ndarray:
        return F @ self.W


class ESN:
    """Frozen random reservoir; only the linear readout is ever fitted."""

    def __init__(self, n_in: int, n_res: int = 300, spectral: float = 0.9, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.Win = rng.uniform(-1, 1, (n_in, n_res))
        W = rng.standard_normal((n_res, n_res))
        W[rng.random((n_res, n_res)) > 0.1] = 0.0
        sr = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (spectral / max(sr, 1e-9))
        self.n_res = n_res

    def run(self, Z: np.ndarray) -> np.ndarray:
        """Features = [reservoir state, current input] for every sample."""
        h = np.zeros(self.n_res)
        out = []
        for z in Z:
            h = np.tanh(z @ self.Win + self.W @ h)
            out.append(np.concatenate([h, z]))
        return np.array(out)


def lag_features(Z: np.ndarray, k: int) -> np.ndarray:
    """Stack the last k+1 samples as one feature vector (the classical AR view)."""
    pad = np.repeat(Z[:1], k, axis=0)
    Zp = np.vstack([pad, Z])
    return np.stack([Zp[k - j: k - j + len(Z)] for j in range(k + 1)], axis=1).reshape(len(Z), -1)
