"""SparseDeepBrain repaired. Design kept: per-layer K(strength)/V(content), top-k sparse
attention, no gradients, energy-budget lock.

Repaired:
    1. every layer must stay in dim space: K_l, V_l are (slots, dim). The verbatim
       version bounced the error through V.T, turning a (1,dim) blame into (1,slots)
       and then tried to write it into a (slots, dim) matrix.
    2. y_target was ignored (always reconstructed x). Now target=None -> reconstruct.
    3. blame now goes back through the Jacobian transpose of the attention read:
           e_slots = a * (V @ e)            # only the slots that actually fired
           e_prev  = K.T @ e_slots / |h|    # back into the previous layer's space
    4. V is snapshotted before writing, so the blame uses the matrix that produced
       the forward pass, not the one it just became.
    5. optional attention-energy normalisation: without it the prediction moves by
       lr * sum(a^2) * err ~ lr/k * err, i.e. top-k makes it k times too slow.
"""

from __future__ import annotations

import numpy as np


class SparseDeepBrain:
    def __init__(
        self,
        dim: int = 8,
        slots: int = 128,
        layers: int = 4,
        topk: int = 4,
        lr: float = 0.05,
        lr_k: float = 0.02,
        budget: float = 10.0,
        decay: float = 1.0,
        residual: bool = False,
        normalize_step: bool = True,
        update_k: bool = True,
        write_top_only: bool = False,
        err_gain: float = 1.0,
        use_readout: bool = False,
        lr_w: float = 0.3,
        weight_scale: float = 1.0,
        layer_gain: float = 1.0,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        self.dim, self.slots, self.layers, self.topk = dim, slots, layers, min(topk, slots)
        self.lr, self.lr_k, self.budget, self.decay = lr, lr_k, budget, decay
        self.residual, self.normalize_step, self.update_k = residual, normalize_step, update_k
        self.write_top_only = write_top_only
        self.err_gain = float(err_gain)
        self.use_readout = bool(use_readout)
        self.lr_w = float(lr_w)
        self.weight_scale = float(weight_scale)
        self.layer_gain = float(layer_gain)
        self.W = np.zeros((dim, dim)) if use_readout else None

        self.Ks = [rng.standard_normal((slots, dim)) * 0.3 for _ in range(layers)]
        self.Vs = [rng.standard_normal((slots, dim)) * 0.3 for _ in range(layers)]

    @staticmethod
    def _softmax(s):
        e = np.exp(s - np.max(s, axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def _sparse(self, scores):
        a = self._softmax(scores)
        idx = np.argsort(a)[-self.topk:]
        mask = np.zeros_like(a)
        mask[idx] = 1.0
        a = a * mask
        return a / (a.sum() + 1e-8), idx

    def forward(self, x):
        h = np.atleast_2d(x).astype(np.float64)
        attns, idxs, hs, hnorms = [], [], [], []
        for l in range(self.layers):
            hn = h / (np.linalg.norm(h, axis=-1, keepdims=True) + 1e-8)
            a, idx = self._sparse((hn @ self.Ks[l].T)[0])
            a = a * self.weight_scale
            o = a @ self.Vs[l]
            attns.append(a)
            idxs.append(idx)
            hs.append(o.copy())
            hnorms.append(hn[0].copy())
            h = h + o if self.residual else o.reshape(1, -1)
        return h, attns, idxs, hs, hnorms

    def step(self, x, y_target=None):
        x2 = np.atleast_2d(x).astype(np.float64)
        pred, attns, idxs, hs, hnorms = self.forward(x2)
        target = x2 if y_target is None else np.atleast_2d(y_target)

        if self.use_readout:
            h = pred[0]                                    # latent state
            y_hat = h @ self.W
            e_out = (target - y_hat.reshape(1, -1))[0]
            energy = float(h @ h) + 1e-9
            self.W += self.lr_w * np.outer(h, e_out) / energy
            e = (e_out @ self.W.T).copy()                  # blame pushed into latent space
            pred = y_hat.reshape(1, -1)
        else:
            e = (target - pred)[0].copy()                  # (dim,)

        for l in range(self.layers - 1, -1, -1):
            V_old = self.Vs[l]
            idx = idxs[l]
            a = attns[l]
            a_act = a[idx]
            g = 1.0 / (float(a_act @ a_act) + 1e-9) if self.normalize_step else 1.0
            g *= self.layer_gain ** (self.layers - 1 - l)   # 顶层 1.0，越深越小

            # write content: only the slots that fired
            writable = (l == self.layers - 1) or (not self.write_top_only)
            dV = np.zeros_like(self.Vs[l])
            if writable:
                dV[idx] = a_act[:, None] * e
            self.Vs[l] = V_old * self.decay + self.lr * g * dV

            if self.update_k and writable:
                dK = np.zeros_like(self.Ks[l])
                dK[idx] = a_act[:, None] * (e @ V_old[idx].T)[:, None] * hnorms[l]
                self.Ks[l] = self.Ks[l] * self.decay + self.lr_k * g * dK

            # blame one layer down (Jacobian transpose of the attention read)
            if l > 0:
                e_slots = a * (V_old @ e)
                e = self.err_gain * (self.Ks[l].T @ e_slots) / (np.linalg.norm(hnorms[l]) + 1e-8)

            for M in (self.Ks[l], self.Vs[l]):
                n = np.linalg.norm(M, axis=-1, keepdims=True)
                M *= np.minimum(1.0, self.budget / np.maximum(n, 1e-9))

        return pred

    def counts(self):
        n = int(sum(m.size for m in self.Ks) + sum(m.size for m in self.Vs))
        return {"trainable": n, "frozen": 0, "total": n}
