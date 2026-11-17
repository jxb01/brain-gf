"""NEST-Brain: Neuromodulated Evolutionary Sparse Tree Brain.

Faithful to the pseudocode, with two switches that must be separated by experiment:

    k_update  : whether K follows the dopamine rule at all
    dopamine  : "scalar" = the spec's  -||e||^2   (sign-blind, same direction for every column)
                "vector" = per-column alignment of that column's own contribution with the error

Also adds the readout head the pseudocode is missing: without it the prediction is
h itself, and h is L2-normalised every layer, so |pred| <= 1 and any target with
amplitude > 1 is unreachable (the exact failure we already measured on NRIR).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- primitives

def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class GRUCell:
    """Single GRU cell (numpy). Weights frozen: only K is ever written."""

    def __init__(self, in_dim, hid, rng, scale=0.3):
        self.in_dim, self.hid = in_dim, hid
        s = scale / np.sqrt(in_dim + hid)
        self.Wz = rng.standard_normal((in_dim, hid)) * s
        self.Wr = rng.standard_normal((in_dim, hid)) * s
        self.Wn = rng.standard_normal((in_dim, hid)) * s
        self.Uz = rng.standard_normal((hid, hid)) * s
        self.Ur = rng.standard_normal((hid, hid)) * s
        self.Un = rng.standard_normal((hid, hid)) * s
        self.bz = np.zeros(hid)
        self.br = np.zeros(hid)
        self.bn = np.zeros(hid)
        self.n_params = sum(a.size for a in
                            (self.Wz, self.Wr, self.Wn, self.Uz, self.Ur, self.Un,
                             self.bz, self.br, self.bn))

    def step(self, x, h):
        z = _sig(x @ self.Wz + h @ self.Uz + self.bz)
        r = _sig(x @ self.Wr + h @ self.Ur + self.br)
        n = np.tanh(x @ self.Wn + (r * h) @ self.Un + self.bn)
        return (1.0 - z) * h + z * n


class CorticalColumn:
    """One V slot: n_neurons GRUs + an output projection."""

    def __init__(self, n_neurons, input_dim, hidden_dim, rng, diverse=False):
        self.n_neurons = n_neurons
        self.hidden_dim = hidden_dim
        self.diverse = diverse
        # 每个柱看到不同的输入投影 => 柱之间不再可互换，路由才有东西可选
        self.Pin = (rng.standard_normal((input_dim, input_dim)) * 0.7 if diverse else None)
        self.grus = [GRUCell(input_dim, hidden_dim, rng) for _ in range(n_neurons)]
        self.h_states = [np.zeros(hidden_dim) for _ in range(n_neurons)]
        self.output_proj = rng.standard_normal(
            (n_neurons * hidden_dim, hidden_dim)) / np.sqrt(n_neurons * hidden_dim)
        self.n_params = sum(g.n_params for g in self.grus) + self.output_proj.size

    def forward(self, x):
        if self.Pin is not None:
            x = np.tanh(x @ self.Pin)
        for r, gru in enumerate(self.grus):
            self.h_states[r] = gru.step(x, self.h_states[r])
        return np.concatenate(self.h_states) @ self.output_proj


class NESTLayer:
    def __init__(self, n_columns, input_dim, hidden_dim, topk, n_neurons, rng, diverse=False):
        self.n_columns = n_columns
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.topk = topk
        self.K = rng.standard_normal((input_dim, n_columns)) * 0.1          # trainable routing
        self.columns = [CorticalColumn(n_neurons, input_dim, hidden_dim, rng, diverse=diverse)
                        for _ in range(n_columns)]
        self.survival = np.zeros(n_columns)
        self.dormant = np.zeros(n_columns, dtype=bool)
        self.act_count = np.zeros(n_columns)      # for exp4 statistics

    def forward(self, h):
        s = h @ self.K
        s = np.where(self.dormant, -1e9, s)
        k = min(self.topk, self.n_columns)
        idx = np.argpartition(-s, k - 1)[:k]
        z = s[idx] - s[idx].max()
        w = np.exp(z)
        w = w / (w.sum() + 1e-12)
        a = np.zeros(self.n_columns)
        a[idx] = w
        self.act_count[idx] += 1.0

        out = np.zeros(self.hidden_dim)
        col_out = {}
        parts = []
        for j, wij in zip(idx, w):
            o = self.columns[j].forward(h)
            col_out[int(j)] = o
            out += wij * o
            parts.append(wij * o)
        feats = np.concatenate(parts) if parts else np.zeros(self.topk * self.hidden_dim)
        return out, a, idx, col_out, feats


class NESTBrain:
    def __init__(self, layers=1, input_dim=8, hidden_dim=8, n_columns=128,
                 topk=8, n_neurons=4, out_dim=8, seed=0, renorm=True,
                 budget=None, pre_ln=False, diverse=False, read_mode="mix"):
        rng = np.random.default_rng(seed)
        self.hidden_dim = hidden_dim
        self.renorm = renorm
        self.budget = budget      # 只在 renorm=False 时生效：||h|| 上限
        self.pre_ln = pre_ln      # 读出前做无参数 LayerNorm
        # residual needs matching dims: project the input when hidden != input
        self.Rin = (rng.standard_normal((input_dim, hidden_dim)) / np.sqrt(input_dim)
                    if input_dim != hidden_dim else None)
        self.read_mode = read_mode
        dim0 = hidden_dim if self.Rin is not None else input_dim
        self.layers = [
            NESTLayer(n_columns, dim0 if l == 0 else hidden_dim,
                      hidden_dim, topk, n_neurons, rng, diverse=diverse)
            for l in range(layers)
        ]
        n_feat = (topk * hidden_dim * layers) if read_mode == "concat" else hidden_dim
        self.W = np.zeros((n_feat, out_dim))        # readout, free magnitude
        self.Ws = np.zeros((hidden_dim, hidden_dim))  # sensory self-model (optional)
        self.steps = 0

    # ------------- forward -------------
    def forward(self, x):
        h = np.asarray(x, np.float64)
        if self.Rin is not None:
            h = h @ self.Rin
        cache = []
        feats_all = []
        for layer in self.layers:
            h_new, a, idx, col_out, lf = layer.forward(h)
            feats_all.append(lf)
            cache.append((a, idx, h.copy(), col_out))
            h = h_new + h
            if self.renorm:
                h = h / (np.linalg.norm(h) + 1e-8)
            elif self.budget is not None:
                nh = np.linalg.norm(h)
                if nh > self.budget:
                    h = h * (self.budget / nh)
        self._feats = np.concatenate(feats_all) if self.read_mode == "concat" else h
        return h, cache

    # ------------- one online step -------------
    def step(self, x, y=None, eta_k=0.01, eta_s=0.01, alpha_s=0.01,
             lam_cost=0.1, theta_sleep=-0.5, wake_interval=500,
             k_update=True, dopamine="scalar", readout=True, survival="spec",
             lr_w=0.15):
        h, cache = self.forward(x)
        feats = self._feats
        if self.pre_ln:
            feats = (h - h.mean()) / (h.std() + 1e-8)   # 无参数 LayerNorm
        pred = feats @ self.W if readout else h

        if y is None:
            self.steps += 1
            return pred

        e = np.asarray(y, np.float64) - pred

        if readout and lr_w > 0.0:
            energy = float(feats @ feats) + 1e-8
            self.W += lr_w * np.outer(feats, e) / energy   # NLMS, no gradient

        scalar_dop = -float(e @ e)                        # exactly the pseudocode

        for l in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[l]
            a, idx, h_input, col_out = cache[l]

            if k_update:
                if dopamine == "scalar":
                    for j in idx:
                        layer.K[:, j] += eta_k * scalar_dop * a[j] * h_input
                        layer.K[:, j] /= np.linalg.norm(layer.K[:, j]) + 1e-8
                elif dopamine == "proj":
                    # 误差在该列输出方向上的投影（带符号），一阶版烧蚀 credit
                    contrib = np.zeros(len(idx))
                    for t, j in enumerate(idx):
                        c = col_out[int(j)] @ self.W            # 该列对输出的贡献向量
                        contrib[t] = float(e @ c)               # >0 = 这列在帮忙
                    scale = np.abs(contrib).max() + 1e-12
                    for t, j in enumerate(idx):
                        layer.K[:, j] += eta_k * (contrib[t] / scale) * a[j] * h_input
                        layer.K[:, j] /= np.linalg.norm(layer.K[:, j]) + 1e-8
                else:  # "vector": scaled by a_j
                    contrib = np.zeros(len(idx))
                    for t, j in enumerate(idx):
                        contrib[t] = float(a[j] * (col_out[int(j)] @ self.W) @ e)
                    scale = np.abs(contrib).max() + 1e-12
                    for t, j in enumerate(idx):
                        layer.K[:, j] += eta_k * (contrib[t] / scale) * a[j] * h_input
                        layer.K[:, j] /= np.linalg.norm(layer.K[:, j]) + 1e-8

            # ---- survival bookkeeping ----
            if survival == "spec":
                # as written: activation strength x (negative) global signal, minus cost.
                # Sign-blind: the more a column works, the faster it dies.
                for j in range(layer.n_columns):
                    contrib = a[j] * scalar_dop
                    cost = a[j] * 1.0
                    layer.survival[j] = ((1.0 - alpha_s) * layer.survival[j]
                                         + alpha_s * (contrib - lam_cost * cost))
            else:
                # "ablate": credit = how much the error grows when THIS column's
                # contribution is removed. Per-column, signed, and cheap: the column
                # output is already in col_out, so e_{-j} = e + a_j * (col_out_j @ W).
                for j in range(layer.n_columns):
                    if a[j] > 0.0:
                        e_minus = e + a[j] * (col_out[int(j)] @ self.W)
                        credit = float(e @ e) - float(e_minus @ e_minus)   # >0 = helped
                    else:
                        credit = 0.0
                    cost = a[j] * 1.0
                    layer.survival[j] = ((1.0 - alpha_s) * layer.survival[j]
                                         + alpha_s * (credit - lam_cost * cost))
            layer.dormant = layer.survival < theta_sleep

        self.steps += 1
        if wake_interval and self.steps % wake_interval == 0:
            for layer in self.layers:
                d = np.flatnonzero(layer.dormant)
                if d.size:
                    j = int(d[0])
                    layer.K[:, j] += np.random.default_rng(0).standard_normal(layer.K.shape[0]) * 0.01
                    layer.dormant[j] = False
                    layer.survival[j] = 0.0
        return pred

    def counts(self):
        tr = int(sum(l.K.size for l in self.layers) + self.W.size)
        fr = int(sum(c.n_params for l in self.layers for c in l.columns))
        return {"trainable": tr, "frozen": fr, "total": tr + fr}


# ---------------------------------------------------------------- experiments

def signals(n, dim=8, seed=0, kind="sine"):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    freqs = np.linspace(0.02, 0.16, dim)
    if kind == "sine":
        base = np.sin(2 * np.pi * freqs[None, :] * t[:, None])
    else:  # square
        base = np.sign(np.sin(2 * np.pi * freqs[None, :] * t[:, None]))
    return base + 0.02 * rng.standard_normal((n, dim))


def r2(y, p):
    return 1.0 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()


def run_sequence(brain, X, Y, **kw):
    import time
    preds = np.empty_like(Y)
    t0 = time.perf_counter()
    for i in range(len(X)):
        preds[i] = brain.step(X[i], Y[i], **kw)
    us = 1e6 * (time.perf_counter() - t0) / len(X)
    return preds, us
