"""Consensus applied where it belongs: on PREDICTIONS (one readout per leaf), not on hidden states."""
import numpy as np
from brain_gf.nest_tree import TreeBrain, signals, r2

N = 1200
A_ = signals(N + 1, dim=8, seed=0, kind="sine")
X, Y = A_[:-1], A_[1:]

def make():
    return TreeBrain(1, 8, 8, 128, 8, 4, 8, seed=0, agg="sum")

def run(mode, theta=0.5, lr=0.15):
    b = make()
    M, hid, d = 8, 8, 8
    Wi = np.zeros((M, hid, d))
    preds = np.empty_like(Y)
    wlog = []
    for i in range(N):
        _p, _f = b.forward(X[i])
        T = np.stack(b.last_tracks)          # (M, hid), already unit-normalised
        pi = np.einsum("mh,mhd->md", T, Wi)  # each leaf's own prediction
        S = T @ T.T
        np.fill_diagonal(S, 0.0)
        cons = np.array([(S[k] > theta).sum() + 1 for k in range(M)], dtype=float)
        w = cons / cons.sum() if mode == "consensus" else np.full(M, 1.0 / M)
        wlog.append(w)
        yhat = w @ pi
        preds[i] = yhat
        e = Y[i] - yhat
        for k in range(M):                   # each leaf learns from the shared error
            den = float(T[k] @ T[k]) + 1e-8
            Wi[k] += lr * np.outer(T[k], e) / den
        b.W = np.zeros_like(b.W)
    return preds, np.array(wlog)

for mode in ("mean", "consensus"):
    p, W = run(mode)
    print(f"  per-leaf readouts, {mode:<10} R2={r2(Y, p):+.3f}   "
          f"weights min/max/std={W[:,-1].min():.3f}/{W[:,-1].max():.3f}/{W[:,-1].std():.4f}")

b = TreeBrain(1, 8, 8, 128, 8, 4, 8, seed=0, agg="concat")
p = np.empty_like(Y)
for i in range(N):
    p[i] = b.step(X[i], Y[i])
print(f"  reference: concat readout          R2={r2(Y, p):+.3f}")
