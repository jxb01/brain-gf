"""NEST tree: k independent branches / tracks, each routed on its own, concat at the end.

Difference from nest.py (the flattened version):
    old    : h = sum_j a_j * col_j(x)          -> one vector, depth collapses
    tree   : k branch vectors, each routed independently at the next layer,
             concatenated only at the readout     -> depth has room to work

Branch count is kept at top-k (way C). Each (track, layer) keeps its own column
states, otherwise the GRUs of a shared bank would interleave the k tracks.
"""

from __future__ import annotations

import numpy as np


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class GRUCell:
    def __init__(self, in_dim, hid, rng, scale=0.3):
        self.in_dim, self.hid = in_dim, hid
        s = scale / np.sqrt(in_dim + hid)
        self.Wz = rng.standard_normal((in_dim, hid)) * s
        self.Wr = rng.standard_normal((in_dim, hid)) * s
        self.Wn = rng.standard_normal((in_dim, hid)) * s
        self.Uz = rng.standard_normal((hid, hid)) * s
        self.Ur = rng.standard_normal((hid, hid)) * s
        self.Un = rng.standard_normal((hid, hid)) * s
        self.bz, self.br, self.bn = np.zeros(hid), np.zeros(hid), np.zeros(hid)
        self.n_params = sum(a.size for a in (self.Wz, self.Wr, self.Wn,
                                             self.Uz, self.Ur, self.Un,
                                             self.bz, self.br, self.bn))

    def step(self, x, h):
        z = _sig(x @ self.Wz + h @ self.Uz + self.bz)
        r = _sig(x @ self.Wr + h @ self.Ur + self.br)
        n = np.tanh(x @ self.Wn + (r * h) @ self.Un + self.bn)
        return (1.0 - z) * h + z * n


class Column:
    def __init__(self, input_dim, hidden_dim, n_neurons, rng):
        self.hidden_dim = hidden_dim
        self.n_neurons = n_neurons
        self.grus = [GRUCell(input_dim, hidden_dim, rng) for _ in range(n_neurons)]
        self.proj = rng.standard_normal((n_neurons * hidden_dim, hidden_dim)) / np.sqrt(n_neurons * hidden_dim)
        self.n_params = sum(g.n_params for g in self.grus) + self.proj.size

    def forward(self, x, hs=None):
        if hs is None:
            hs = [np.zeros(self.hidden_dim) for _ in range(self.n_neurons)]
        for r, g in enumerate(self.grus):
            hs[r] = g.step(x, hs[r])
        return np.concatenate(hs) @ self.proj, hs


class TreeLayer:
    def __init__(self, n_columns, input_dim, hidden_dim, topk, n_neurons, rng):
        self.n_columns, self.input_dim, self.hidden_dim = n_columns, input_dim, hidden_dim
        self.topk, self.n_neurons = topk, n_neurons
        self.K = rng.standard_normal((input_dim, n_columns)) * 0.1
        self.cols = [Column(input_dim, hidden_dim, n_neurons, rng) for _ in range(n_columns)]
        self.weight_branches = False
        self.survival = np.zeros(n_columns)
        self.dormant = np.zeros(n_columns, dtype=bool)
        self.act = np.zeros(n_columns)

    def forward(self, x, state):
        """state: dict {column_index -> [neuron hidden states]} for THIS track."""
        s = np.where(self.dormant, -1e9, x @ self.K)
        k = min(self.topk, self.n_columns)
        idx = np.argpartition(-s, k - 1)[:k]
        z = s[idx] - s[idx].max()
        w = np.exp(z)
        w /= w.sum() + 1e-12
        self.act[idx] += 1.0

        outs, mix = [], np.zeros(self.hidden_dim)
        for j, wj in zip(idx, w):
            j = int(j)
            o, hs = self.cols[j].forward(x, state.get(j))
            state[j] = hs
            # 分支不再乘 a_j：读出是线性的，权重冗余；乘上去只会让幅度逐层衰减
            outs.append(o if not self.weight_branches else wj * o)
            mix += wj * o
        a = np.zeros(self.n_columns)
        a[idx] = w
        return mix, outs, a, idx

    def counts(self):
        return sum(c.n_params for c in self.cols) + self.K.size


class TreeBrain:
    def __init__(self, layers=1, input_dim=8, hidden_dim=8, n_columns=128,
                 topk=8, n_neurons=4, out_dim=8, seed=0, agg="concat", theta=0.5):
        rng = np.random.default_rng(seed)
        self.topk, self.hidden_dim, self.layers_n = topk, hidden_dim, layers
        self.layers = [
            TreeLayer(n_columns, input_dim if l == 0 else hidden_dim,
                      hidden_dim, topk, n_neurons, rng)
            for l in range(layers)
        ]
        self.agg, self.theta = agg, theta
        n_feat = (topk * hidden_dim) if agg == "concat" else hidden_dim
        self.W = np.zeros((n_feat, out_dim))
        self.last_consensus = None
        # per (track, layer) column states
        self.states = [dict() for _ in range(topk)]
        self.root_state = {}
        self.steps = 0

    def forward(self, x):
        x = np.asarray(x, np.float64)
        mix0, outs, a0, idx0 = self.layers[0].forward(x, self.root_state)
        tracks = [o / (np.linalg.norm(o) + 1e-8) for o in outs]   # 逐轨归一化，稳住尺度

        for l in range(1, len(self.layers)):
            new = []
            for t, b in enumerate(tracks):
                st = self.states[t].setdefault(l, {})
                m, subs, a, idx = self.layers[l].forward(b, st)
                new.append(m / (np.linalg.norm(m) + 1e-8))   # one normalized vector per track
            tracks = new

        self.last_tracks = tracks
        if self.agg == "concat":
            feats = np.concatenate(tracks)
        elif self.agg == "sum":
            feats = np.mean(tracks, axis=0)
        else:  # consensus: 按"有多少叶子和它相似"加权，簇大的说话更响
            M = len(tracks)
            Xm = np.stack(tracks)
            Xm = Xm / (np.linalg.norm(Xm, axis=1, keepdims=True) + 1e-8)
            S = Xm @ Xm.T
            np.fill_diagonal(S, 0.0)
            consensus = np.array([(S[i] > self.theta).sum() + 1 for i in range(M)], dtype=np.float64)
            w = consensus / consensus.sum()
            self.last_consensus = w
            feats = w @ Xm
        return feats @ self.W, feats

    def step(self, x, y=None, lr_w=0.15, decay=1.0):
        pred, feats = self.forward(x)
        if y is not None:
            e = np.asarray(y, np.float64) - pred
            energy = float(feats @ feats) + 1e-8
            self.W = self.W * decay + lr_w * np.outer(feats, e) / energy
        self.steps += 1
        return pred

    def counts(self):
        tr = int(self.W.size)
        fr = int(sum(l.counts() for l in self.layers))
        return {"trainable": tr, "frozen": fr, "total": tr + fr}


# --------------------------------------------------------------- experiments
def r2(y, p):
    return 1.0 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()


def signals(n, dim=8, seed=0, kind="sine"):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    f = np.linspace(0.02, 0.16, dim)
    base = (np.sin(2 * np.pi * f[None, :] * t[:, None]) if kind == "sine"
            else np.sign(np.sin(2 * np.pi * f[None, :] * t[:, None])))
    return base + 0.02 * rng.standard_normal((n, dim))
