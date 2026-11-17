"""Gradient-free brains.

Design rules (the whole point of the package):

1.  No autograd, no loss.backward(), no optimizer state. Every update is a physical
    write:  param <- param * decay + lr * (feature outer error) / feature_energy.
    That is NLMS (normalised least mean squares) - the exact online least-squares
    step, derived without any gradient machinery.

2.  Frozen feature map. K1/K2 are random, orthonormalised once, never trained.
    Freezing the map is what keeps the online learner stable: if the coordinate
    system moved, every memory written so far would be invalidated at once.

3.  Lock the direction, leave the magnitude alone.
    - Feature map columns: unit norm (frozen directions).
    - Attention memory rows: unit norm after every write (sphere lock).
    - Readout matrices: free magnitude, only a global energy budget (clipping).
    A pure attention readout lives inside the convex hull of its slots, so it can
    never output |y| beyond the slot radius. The readout *matrix multiply* is what
    unlocks amplitude; the sphere lock alone would cap it.

4.  Decay (forgetting) is mandatory for non-stationary streams. Without it the
    readout accumulates every regime it has ever seen and becomes a stitched
    average of incompatible ones.

Two concrete brains:

    FeatureReadoutBrain  - the one that actually fits: random nonlinear features
                           (tanh + cos, so even functions are representable) and a
                           free linear readout per organ. Cheap, stable, R^2 ~ 0.95.
    PureAttentionBrain   - the associative-memory variant: input acts as its own
                           probe, no Q/K, softmax over unit slots, residual stack.
                           Bounded by the convex hull, so weaker in amplitude, but
                           it is the closest thing to a biological attractor net.
"""

from __future__ import annotations

import numpy as np

from .numerics import feature_map, softmax, unit_cols, unit_rows


class FeatureReadoutBrain:
    """Frozen random features + free readout, written by NLMS.

    Two organs share one feature vector:
        sensory  : reconstruct the input   (always on, fixed high gain)
        decision : predict the target      (non-saturating stress response)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hid: int = 128,
        hid2: int = 128,
        lr_sensory: float = 0.15,
        lr_decision: float = 0.05,
        decay: float = 0.999,
        budget: float = 12.0,
        use_cos: bool = True,
        use_stress: bool = True,
        seed: int = 0,
        dtype=np.float64,
    ):
        rng = np.random.default_rng(seed)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hid = int(hid)
        self.hid2 = int(hid2)
        self.use_cos = bool(use_cos)
        self.use_stress = bool(use_stress)
        self.lr_sensory = float(lr_sensory)
        self.lr_decision = float(lr_decision)
        self.decay = float(decay)
        self.budget = float(budget)
        self.eps = 1e-6
        self.dtype = dtype

        # --- frozen feature map (never trained) ---
        self.K1 = feature_map(rng, self.in_dim, self.hid).astype(dtype)
        f1 = (2 * self.hid) if self.use_cos else self.hid
        self.K2 = feature_map(rng, f1, self.hid2).astype(dtype)
        self.fdim = f1 + self.hid2

        # --- free readouts (the only things written online) ---
        self.Ws = np.zeros((self.fdim, self.in_dim), dtype=dtype)  # sensory: phi -> x
        self.Wd = np.zeros((self.fdim, self.out_dim), dtype=dtype)  # decision: phi -> y

    # ---------------- forward ----------------
    def features(self, x: np.ndarray) -> np.ndarray:
        z1 = x @ self.K1
        if self.use_cos:
            # cos adds the even part: tanh alone is odd, so x*x style interactions
            # are simply not representable by an odd gate.
            h1 = np.concatenate([np.tanh(z1), np.cos(z1)], axis=-1)
        else:
            h1 = np.tanh(z1)
        h2 = np.tanh(h1 @ self.K2)
        return np.concatenate([h1, h2], axis=-1)

    def predict(self, x: np.ndarray):
        phi = self.features(x)
        return phi @ self.Ws, phi @ self.Wd, phi

    # ---------------- write ----------------
    def _write(self, W: np.ndarray, phi: np.ndarray, err: np.ndarray, lr: float) -> np.ndarray:
        energy = float(phi @ phi) + self.eps
        W = W * self.decay + lr * np.outer(phi, err) / energy
        # energy budget: never let a readout column leave the safe box
        n = np.linalg.norm(W, axis=0, keepdims=True)
        W = W * np.minimum(1.0, self.budget / np.maximum(n, 1e-12))
        return W

    def step(self, x: np.ndarray, y: np.ndarray | None = None):
        s_pred, d_pred, phi = self.predict(x)

        # sensory organ: keep an honest picture of the world, always, at high gain
        self.Ws = self._write(self.Ws, phi, x - s_pred, self.lr_sensory)

        # decision organ: quiet on small errors, violent on big ones, but the
        # magnitude of the error is never thrown away (tanh alone saturates at 0.15)
        if y is not None:
            err = y - d_pred
            if self.use_stress:
                err = err * (1.0 + 2.0 * np.tanh(np.abs(err)))
            self.Wd = self._write(self.Wd, phi, err, self.lr_decision)

        return s_pred, d_pred

    # ---------------- bookkeeping ----------------
    def counts(self) -> dict:
        trainable = int(self.Ws.size + self.Wd.size)
        frozen = int(self.K1.size + self.K2.size)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}


class PureAttentionBrain:
    """Content-addressed memory: the input is its own probe, there is no Q and no K.

    forward : h <- h + softmax(unit(h) @ V_l^T / tau) @ V_l     (residual stack)
              pred = h @ Rout
    write   : V_l <- normalize_rows(V_l * decay + lr * outer(a_l, e_l) / |a_l|^2)

    Error is carried backwards by projection through the very same matrices
    (transpose symmetry), never by differentiating anything.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dim: int = 32,
        layers: int = 3,
        slots: int = 32,
        lr: float = 0.2,
        decay: float = 0.999,
        tau: float = 0.5,
        residual: bool = True,
        seed: int = 0,
        dtype=np.float64,
    ):
        rng = np.random.default_rng(seed)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.dim = int(dim)
        self.layers = int(layers)
        self.slots = int(slots)
        self.lr = float(lr)
        self.decay = float(decay)
        self.tau = float(tau)
        self.residual = bool(residual)
        self.eps = 1e-6
        self.dtype = dtype

        self.Rin = unit_cols(rng.standard_normal((self.in_dim, self.dim))).astype(dtype)
        self.Rout = unit_cols(rng.standard_normal((self.dim, self.out_dim))).astype(dtype)
        self.V = [
            unit_rows(rng.standard_normal((self.slots, self.dim))).astype(dtype)
            for _ in range(self.layers)
        ]

    def forward(self, x: np.ndarray):
        h = x @ self.Rin
        acts = []
        for V in self.V:
            p = h / (np.linalg.norm(h) + 1e-9)
            a = softmax(p @ V.T / self.tau)
            o = a @ V
            acts.append(a)
            h = h + o if self.residual else o
        pred = h @ self.Rout
        return pred, h, acts

    def step(self, x: np.ndarray, y: np.ndarray | None = None):
        pred, h, acts = self.forward(x)
        if y is None:
            return pred
        err = np.asarray(y, dtype=np.float64) - pred
        # error lives in output space -> push it into the latent space with Rout^T
        e = err @ self.Rout.T

        for l in range(self.layers - 1, -1, -1):
            V = self.V[l]
            a = acts[l]
            den = float(a @ a) + self.eps
            V *= self.decay
            V += self.lr * np.outer(a, e) / den
            self.V[l] = unit_rows(V)  # sphere lock: direction only
            # carry the blame one layer down: residual identity + projection
            e = e + (e @ self.V[l].T @ self.V[l]) / max(self.slots, 1)

        return pred

    def counts(self) -> dict:
        trainable = int(sum(v.size for v in self.V))
        frozen = int(self.Rin.size + self.Rout.size)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}
