"""Math-op columns + cluster readout.

Each column holds 4 distinct hand-written operators on a frozen random projection
(linear / quadratic / abs / threshold / sin / diff / cumsum / normalize / eml).
No GRU, no training of the operators - the only learner is the readout.

Readouts compared:
    concat  : k branch vectors concatenated (capacity = k*hidden)
    sum     : their mean (capacity = hidden)
    cluster : greedy cosine clustering, each cluster summed, cluster size as weight
              - this is still a weighted SUM of vectors inside the same hull
"""

from __future__ import annotations

import numpy as np

OPS = ("linear", "quadratic", "abs", "threshold", "sin", "cos", "diff", "cumsum",
       "normalize", "eml")


def _unit_cols(M):
    return M / np.maximum(np.linalg.norm(M, axis=0, keepdims=True), 1e-9)


def _softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


class OpColumn:
    """4 distinct operators over frozen random projections; output = their mean."""

    def __init__(self, in_dim, hid, rng, n_ops=4, combine="mean", op_set=None):
        self.in_dim, self.hid = in_dim, hid
        self.combine = combine
        self.out_dim = hid * n_ops if combine == "concat" else hid
        self.ops = []
        chosen = (list(op_set) if op_set is not None
                  else [OPS[int(t)] for t in rng.choice(len(OPS), size=n_ops, replace=False)])
        for name in chosen:
            self.ops.append((name, _unit_cols(rng.standard_normal((in_dim, hid)))))
        self.prev_z = [np.zeros(hid) for _ in self.ops]
        self.acc = [np.zeros(hid) for _ in self.ops]

    def _one(self, k, name, W, x):
        z = x @ W
        if name == "linear":
            out = z
        elif name == "quadratic":
            out = z * z
        elif name == "abs":
            out = np.abs(z)
        elif name == "threshold":
            out = np.maximum(0.0, z - 0.5)
        elif name == "sin":
            out = np.sin(z)
        elif name == "cos":
            out = np.cos(z)
        elif name == "diff":
            out = z - self.prev_z[k]
            self.prev_z[k] = z
        elif name == "cumsum":
            self.acc[k] = 0.995 * self.acc[k] + z          # 泄漏积分，防止发散
            out = self.acc[k]
        elif name == "normalize":
            out = (z - z.mean()) / (z.std() + 1e-8)
        else:  # eml
            zc = np.clip(z, -6.0, 6.0)
            out = np.exp(zc) - np.log(_softplus(zc) + 1e-6) + 0.5
            out = np.clip(out, -20.0, 20.0)
        return out

    def forward(self, x):
        outs = [self._one(k, name, W, x) for k, (name, W) in enumerate(self.ops)]
        return np.concatenate(outs) if self.combine == "concat" else np.mean(outs, axis=0)


class OpLayer:
    def __init__(self, n_columns, in_dim, hid, topk, rng, n_ops=4, combine="mean", op_set=None):
        self.topk, self.hid = topk, hid
        self.combine = combine
        self.col_out_dim = hid * n_ops if combine == "concat" else hid
        self.K = rng.standard_normal((in_dim, n_columns)) * 0.1
        self.cols = [OpColumn(in_dim, hid, rng, n_ops=n_ops, combine=combine, op_set=op_set)
                     for _ in range(n_columns)]
        self.static_mask = False
        self._fixed_idx = np.arange(min(topk, n_columns))
        self.act = np.zeros(n_columns)

    def forward(self, x, weight_branches=False):
        s = x @ self.K
        k = min(self.topk, len(self.cols))
        if self.static_mask:
            idx = self._fixed_idx
            w = np.full(k, 1.0 / k)
        else:
            idx = np.argpartition(-s, k - 1)[:k]
            z = s[idx] - s[idx].max()
            w = np.exp(z)
            w /= w.sum() + 1e-12
        self.act[idx] += 1.0
        outs = []
        for j, wj in zip(idx, w):
            o = self.cols[int(j)].forward(x)
            outs.append(wj * o if weight_branches else o)
        a = np.zeros(len(self.cols))
        a[idx] = w
        return outs, a


def cluster_readout(leaf_outs, theta=0.5):
    M = len(leaf_outs)
    T = np.stack([o / (np.linalg.norm(o) + 1e-8) for o in leaf_outs])
    S = T @ T.T
    visited = [False] * M
    clusters = []
    for i in range(M):
        if visited[i]:
            continue
        c = [i]
        visited[i] = True
        for j in range(i + 1, M):
            if not visited[j] and S[i, j] > theta:
                c.append(j)
                visited[j] = True
        clusters.append(c)
    outputs = [np.sum([leaf_outs[i] for i in c], axis=0) for c in clusters]
    scores = [len(c) / M for c in clusters]
    return outputs, scores


class OpBrain:
    def __init__(self, layers=1, in_dim=8, hid=8, n_columns=128, topk=8,
                 out_dim=8, agg="concat", theta=0.5, n_ops=4, seed=0, combine="mean",
                 op_set=None, standardize=False, stage2=0, static_mask=False):
        rng = np.random.default_rng(seed)
        self.topk, self.hid, self.agg, self.theta = topk, hid, agg, theta
        self.standardize = standardize
        # 第二级随机混合：把一级特征再投影一次并过非线性，相当于 FR 的 K2 那一级
        self.stage2 = int(stage2)
        self.static_mask = static_mask
        self._rng = rng
        self.combine = combine
        col_out = hid * n_ops if combine == "concat" else hid
        self.col_out = col_out
        self.layers = [OpLayer(n_columns, in_dim if l == 0 else col_out, hid, topk, rng,
                               n_ops, combine=combine, op_set=op_set) for l in range(layers)]
        for lay in self.layers:
            lay.static_mask = static_mask
        n_feat = topk * col_out if agg == "concat" else col_out
        self.K2 = (rng.standard_normal((n_feat, stage2)) / np.sqrt(n_feat)
                   if stage2 else None)
        self.W = np.zeros((n_feat + int(stage2), out_dim))
        self.n_clusters = []

    def forward(self, x):
        x = np.asarray(x, np.float64)
        outs, a = self.layers[0].forward(x)
        tracks = [o / (np.linalg.norm(o) + 1e-8) for o in outs]
        for l in range(1, len(self.layers)):
            new = []
            for b in tracks:
                sub, _a = self.layers[l].forward(b)
                new.append(np.mean(sub, axis=0))
            tracks = [o / (np.linalg.norm(o) + 1e-8) for o in new]

        if self.agg == "concat":
            feats = np.concatenate(tracks)
        elif self.agg == "sum":
            feats = np.mean(tracks, axis=0)
        else:
            outs_c, scores = cluster_readout(tracks, self.theta)
            w = np.array(scores) / np.sum(scores)
            feats = np.sum([w[k] * outs_c[k] for k in range(len(outs_c))], axis=0)
            self.n_clusters.append(len(outs_c))
        if self.K2 is not None:
            feats = np.concatenate([feats, np.tanh(feats @ self.K2)])
        if self.standardize:            # 特征条件数：把尺度统一到一个量级
            feats = (feats - feats.mean()) / (feats.std() + 1e-8)
        return feats @ self.W, feats

    def step(self, x, y=None, lr=0.15, decay=1.0, dead_err=0.0, dead_delta=0.0):
        """dead_err  : skip the write when max|e| is below this (dead-zone on the error)
        dead_delta: skip the write when every |delta| is below this (dead-zone on the step)
        Both are noise gates: they stop the readout from chasing jitter it cannot fix."""
        pred, feats = self.forward(x)
        self.skipped = 0
        if y is not None:
            e = np.asarray(y, np.float64) - pred
            if dead_err > 0.0 and np.max(np.abs(e)) < dead_err:
                self.skipped = 1
                return pred
            energy = float(feats @ feats) + 1e-8
            delta = lr * np.outer(feats, e) / energy
            if dead_delta > 0.0 and np.max(np.abs(delta)) < dead_delta:
                self.skipped = 1
                return pred
            self.W = self.W * decay + delta
        return pred


def r2(y, p):
    return 1.0 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()


def signals(n, dim=8, seed=0, kind="sine"):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    f = np.linspace(0.02, 0.16, dim)
    base = (np.sin(2 * np.pi * f[None, :] * t[:, None]) if kind == "sine"
            else np.sign(np.sin(2 * np.pi * f[None, :] * t[:, None])))
    return base + 0.02 * rng.standard_normal((n, dim))
