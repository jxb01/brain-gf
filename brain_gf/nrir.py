"""NRIR brain, fixed and runnable.

Kept from the original design:
    - 4 layers, each with its own K and V
    - sparse top-k attention (only k slots are read / written)
    - spherical interpolation: h <- normalize((1-g) h + g * (attn @ V))
    - energy budget clipping

Changed (each one was a real defect):
    1. sparse_attn      : argsort on a (1, slots) array returned (k, slots) -> IndexError
    2. output head W    : the original returned h itself (unit norm, cannot express amplitude)
    3. error source     : y_target was accepted and then never used
    4. decay            : there was none, so a regime switch could never be forgotten
    5. back-projection  : err @ V.T had the wrong shape, and with column-orthonormal V
                          the term err @ V.T @ V is exactly err, so alpha=0.5 made the
                          whole thing a no-op. Now the residual path is explicit.
    6. periodic QR      : re-orthonormalising V every 100 layer-updates erases what was
                          just learned (QR replaces V by Q, which is not a rotation).
    7. spectral norm    : norm(V.T, 2) per layer per step is an SVD of a 128x8 matrix;
                          with orthonormal columns it is exactly 1, so alpha was a constant.
"""

from __future__ import annotations

import numpy as np


def _unit_rows(M):
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-9)


def _orth_cols(M):
    q, r = np.linalg.qr(M)
    s = np.sign(np.diag(r))
    s[s == 0] = 1.0
    return q * s[None, :]


class NRIRBrain:
    def __init__(
        self,
        dim: int = 8,
        out_dim: int = 1,
        slots: int = 128,
        layers: int = 4,
        topk: int = 8,
        lr_v: float = 0.05,
        lr_w: float = 0.15,
        lr_k: float = 0.0,     # 0 = freeze the routing map (the stable choice)
        decay: float = 0.999,
        g: float = 0.5,
        budget: float = 10.0,
        readout: str = "final",   # "final" = 只用最后一层 h；"concat" = 拼所有层
        renorm: bool = True,      # False = 不再把每层 h 拉回单位球
        residual: bool = False,   # True = h <- h + g*delta（累积），False = 凸混合
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        self.readout = readout
        self.renorm = renorm
        self.residual = residual
        self.dim, self.out_dim = int(dim), int(out_dim)
        self.slots, self.layers, self.topk = int(slots), int(layers), int(topk)
        self.lr_v, self.lr_w, self.lr_k = float(lr_v), float(lr_w), float(lr_k)
        self.decay, self.g, self.budget = float(decay), float(g), float(budget)

        # memory: rows are slots, columns are the dim feature axes (orthonormal at init)
        self.Vs = [_orth_cols(rng.standard_normal((self.slots, self.dim)) * 0.3) for _ in range(layers)]
        self.Ks = [_orth_cols(rng.standard_normal((self.slots, self.dim)) * 0.3) for _ in range(layers)]
        # readout: free magnitude, this is what lets |pred| exceed 1
        nfeat = self.dim if readout == "final" else self.dim * self.layers
        self.W = np.zeros((nfeat, self.out_dim))
        self.steps = 0

    # ---------------- attention ----------------
    @staticmethod
    def _softmax_row(s):
        s = s - s.max()
        e = np.exp(s)
        return e / e.sum()

    def sparse_attn(self, scores):
        """scores: (slots,) -> (attn (slots,), idx (k,))"""
        a = self._softmax_row(scores)
        k = min(self.topk, self.slots)
        idx = np.argsort(a)[-k:]          # FIX 1: was argsort on a 2-D array
        mask = np.zeros_like(a)
        mask[idx] = 1.0
        a = a * mask
        return a / (a.sum() + 1e-8), idx

    # ---------------- forward ----------------
    def forward(self, x):
        h = np.asarray(x, np.float64).reshape(-1)
        h = h / (np.linalg.norm(h) + 1e-8)

        attns, idxs, states = [], [], [h.copy()]
        for l in range(self.layers):
            hs = h / (np.linalg.norm(h) + 1e-8)   # 打分只看方向，幅度无关
            attn, idx = self.sparse_attn(hs @ self.Ks[l].T)
            delta = attn @ self.Vs[l]
            h_new = h + self.g * delta if self.residual else (1.0 - self.g) * h + self.g * delta
            h = h_new / (np.linalg.norm(h_new) + 1e-8) if self.renorm else h_new
            attns.append(attn)
            idxs.append(idx)
            states.append(h.copy())

        feats = self._features(h, states)
        pred = feats @ self.W          # FIX 2: real readout, magnitude free
        return pred, feats, attns, idxs, states

    def _features(self, h, states):
        if self.readout == "concat":
            return np.concatenate(states[1:], axis=-1)   # layers * dim
        return h

    # ---------------- write ----------------
    def _write_V(self, l, idx, attn_active, e):
        V = self.Vs[l]
        V *= self.decay                # FIX 4: forgetting
        V[idx] += self.lr_v * attn_active[:, None] * e
        self.Vs[l] = self._clip_rows(V)

    def _clip_rows(self, M):
        n = np.linalg.norm(M, axis=-1, keepdims=True)
        return M * np.minimum(1.0, self.budget / np.maximum(n, 1e-9))

    def step(self, x, y=None):
        pred, feats, attns, idxs, states = self.forward(x)

        # 每个层分到的潜在误差（concat 模式下按层切块）
        per_layer = []
        if y is not None:
            err = np.asarray(y, np.float64).reshape(-1) - pred      # FIX 3: real target

            # readout: NLMS on the feature vector (step independent of scale)
            energy = float(feats @ feats) + 1e-8
            self.W = self.W * self.decay + self.lr_w * np.outer(feats, err) / energy

            # latent error: back through the same readout, transposed
            e_all = err @ self.W.T
            if self.readout == "concat":
                per_layer = [e_all[l * self.dim : (l + 1) * self.dim] for l in range(self.layers)]
                e = per_layer[-1]
            else:
                e = e_all
        else:
            # no target: keep an honest self-model of the input (sensory organ)
            e = (np.asarray(x, np.float64).reshape(-1) / (np.linalg.norm(x) + 1e-8)) - feats

        for l in range(self.layers - 1, -1, -1):
            idx = idxs[l]
            a_act = attns[l][idx]
            a_act = a_act / (a_act.sum() + 1e-8)   # renormalise over the selected k
            e_l = e if not per_layer else per_layer[l]
            self._write_V(l, idx, a_act, e_l)

            if self.lr_k > 0.0:                     # optional, unstable by default
                h_in = states[l]
                K = self.Ks[l]
                K *= self.decay
                K[idx] += self.lr_k * a_act[:, None] * np.outer(e_l, h_in)[: len(idx)]
                self.Ks[l] = self._clip_rows(K)

            # FIX 5: residual chain => the downstream error reaches every layer unchanged.
            # (err @ V.T @ V is exactly err for column-orthonormal V, so write it plainly.)
            # e stays as is.

        self.steps += 1
        return pred


# ---------------------------------- self test ----------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--decay", type=float, default=0.999)
    ap.add_argument("--switch", action="store_true", help="regime switch A -> B -> A test")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    dim = 8

    def task_a(x):
        return (np.sin(x[:, 0]) + x[:, 1] * x[:, 2])[:, None]

    def task_b(x):
        return (np.tanh(2.0 * x[:, 3] - x[:, 4]) - 0.7 * np.cos(1.7 * x[:, 0]))[:, None]

    if not args.switch:
        X = rng.standard_normal((args.n, dim))
        Y = task_a(X)
        brain = NRIRBrain(dim=dim, out_dim=1, decay=args.decay, seed=0)
        print("NRIR fixed | task A | one pass per epoch")
        for ep in range(args.epochs):
            preds = np.empty((len(X), 1))
            for i in range(len(X)):
                preds[i] = brain.step(X[i], Y[i])
            r2 = 1 - ((Y - preds) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
            print(f"  epoch {ep:02d}   R2 = {r2:+.4f}")
    else:
        segs = [("A", 1200), ("B", 1000), ("A", 1000)]
        Xs, Ys, tags = [], [], []
        for name, n in segs:
            x = rng.standard_normal((n, dim))
            y = task_a(x) if name == "A" else task_b(x)
            Xs.append(x)
            Ys.append(y)
            tags += [name] * n
        X, Y = np.concatenate(Xs), np.concatenate(Ys)
        for tag, dec in (("decay 0.999", 0.999), ("decay 1.000", 1.0)):
            rng2 = np.random.default_rng(0)
            brain = NRIRBrain(dim=dim, out_dim=1, decay=dec, seed=0)
            preds = np.empty_like(Y)
            for i in range(len(X)):
                preds[i] = brain.step(X[i], Y[i])
            parts = []
            s = 0
            for name, n in segs:
                sl = slice(s, s + n)
                r2 = 1 - ((Y[sl] - preds[sl]) ** 2).sum() / ((Y[sl] - Y[sl].mean(0)) ** 2).sum()
                parts.append(f"{name}:{r2:+.3f}")
                s += n
            print(f"{tag:<12} " + "  ".join(parts))
