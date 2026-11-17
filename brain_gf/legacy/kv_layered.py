"""The two-layer K/V brain, dimensionally repaired, keeping the original intent.

Original intent (unchanged):
    layer1: X  --K1--> strength(attention)  --V1--> content h1
    layer2: h1 --K2--> strength(attention)  --V2--> prediction
    no gradients: every write is  outer(attention * blame, input direction)

Repaired points (the verbatim version cannot run):
    1. V1 must emit hid dims and K2 must consume hid dims; otherwise layer 2
       receives 8 dims while K2 has 16 columns.
    2. V2 must emit dim (target width), not hid.
    3. x_norm / h1_norm lived inside forward() only -> NameError in step().
    4. blame must travel back through K2:  e_slots = V2 @ err, e_h1 = e_slots @ K2.
       err @ V2.T lands in slot space, not in h1 space, so the original line
       mixed two spaces and then broadcast badly.
"""

from __future__ import annotations

import numpy as np


class LayeredKVBrain:
    def __init__(
        self,
        dim: int = 8,
        hid: int = 16,
        slots: int = 16,
        lr_k: float = 0.02,
        lr_v: float = 0.05,
        budget: float = 10.0,
        decay: float = 1.0,
        update_k: bool = True,
        normalize_step: bool = True,
        update_layer1: bool = True,
        tau: float = 1.0,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        self.slots = slots
        self.lr_k = lr_k
        self.lr_v = lr_v
        self.budget = budget
        self.decay = decay
        self.update_k = update_k
        self.normalize_step = normalize_step
        self.update_layer1 = update_layer1
        self.tau = float(tau)

        self.K1 = rng.standard_normal((slots, dim)) * 0.5
        self.V1 = rng.standard_normal((slots, hid)) * 0.5
        self.K2 = rng.standard_normal((slots, hid)) * 0.5
        self.V2 = rng.standard_normal((slots, dim)) * 0.5

    @staticmethod
    def _softmax(s):
        e = np.exp(s - np.max(s, axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    @staticmethod
    def _unit(v):
        return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-6)

    def forward(self, x):
        x = np.atleast_2d(x)
        x_norm = self._unit(x)
        a1 = self._softmax(x_norm @ self.K1.T / self.tau)
        h1 = a1 @ self.V1
        h1_norm = self._unit(h1)
        a2 = self._softmax(h1_norm @ self.K2.T / self.tau)
        pred = a2 @ self.V2
        return pred, {"x_norm": x_norm, "a1": a1, "h1": h1, "h1_norm": h1_norm, "a2": a2}

    def step(self, x, y):
        y = np.atleast_2d(y)
        pred, c = self.forward(x)
        err = y - pred                                    # (1, dim)

        # blame in slot space: whose V2 row is responsible
        e2 = self.V2 @ err[0]                             # (slots,)

        # Without this, the prediction moves by lr * |a|^2 * err per step, and
        # |a|^2 ~ 1/slots for a flat softmax -> 16x too slow. Dividing by the
        # attention energy makes the prediction move by exactly lr * err.
        g2 = 1.0 / (float(c["a2"][0] @ c["a2"][0]) + 1e-9) if self.normalize_step else 1.0
        g1 = 1.0 / (float(c["a1"][0] @ c["a1"][0]) + 1e-9) if self.normalize_step else 1.0

        if self.update_k:
            self.K2 *= self.decay
            self.K2 += self.lr_k * g2 * np.outer(c["a2"][0] * e2, c["h1_norm"][0])
        self.V2 *= self.decay
        self.V2 += self.lr_v * g2 * np.outer(c["a2"][0], err[0])

        # carry blame back into h1 space through the same K2
        e_h1 = e2 @ self.K2                               # (hid,)
        e1 = self.V1 @ e_h1                               # (slots,)

        if self.update_layer1:
            if self.update_k:
                self.K1 *= self.decay
                self.K1 += self.lr_k * g1 * np.outer(c["a1"][0] * e1, c["x_norm"][0])
            self.V1 *= self.decay
            self.V1 += self.lr_v * g1 * np.outer(c["a1"][0], e_h1)

        # energy budget (their original lock)
        for M in (self.K1, self.V1, self.K2, self.V2):
            n = np.linalg.norm(M, axis=-1, keepdims=True)
            M *= np.minimum(1.0, self.budget / np.maximum(n, 1e-9))

        return pred[0]

    def counts(self):
        n = int(self.K1.size + self.V1.size + self.K2.size + self.V2.size)
        return {"trainable": n, "frozen": 0, "total": n}
