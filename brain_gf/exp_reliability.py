"""Replace "agreement" with "reliability": weight each leaf by its own recent error.

Agreement between leaves is not evidence of correctness (their errors are only
0.19-correlated and similarity says nothing about who is right). The cheap,
principled alternative is: track each leaf's running MSE and softmax(-mse/tau).
"""
import numpy as np
from brain_gf.nest_tree import TreeBrain, signals, r2

N = 1200
A_ = signals(N + 1, dim=8, seed=0, kind="sine")
X, Y = A_[:-1], A_[1:]

def run(mode, tau=0.05, lr=0.15, theta=0.5):
    b = TreeBrain(1, 8, 8, 128, 8, 4, 8, seed=0, agg="sum")
    M, hid, d = 8, 8, 8
    Wi = np.zeros((M, hid, d))
    mse = np.ones(M) * 1.0
    preds = np.empty_like(Y)
    wlast = None
    for i in range(N):
        b.forward(X[i])
        T = np.stack(b.last_tracks)
        pi = np.einsum("mh,mhd->md", T, Wi)
        if mode == "mean":
            w = np.full(M, 1.0 / M)
        elif mode == "consensus":
            S = T @ T.T
            np.fill_diagonal(S, 0.0)
            cons = np.array([(S[k] > theta).sum() + 1 for k in range(M)], float)
            w = cons / cons.sum()
        else:  # reliability
            z = -mse / tau
            w = np.exp(z - z.max())
            w /= w.sum()
        wlast = w
        yhat = w @ pi
        preds[i] = yhat
        e = Y[i] - yhat
        for k in range(M):
            den = float(T[k] @ T[k]) + 1e-8
            Wi[k] += lr * np.outer(T[k], e) / den
        # each leaf's own error, EMA
        own = ((pi - Y[i]) ** 2).mean(axis=1)
        mse = 0.98 * mse + 0.02 * own
    return preds, wlast, mse

print(f"{'mode':<14} {'R2':>8}   final weights (min/max)")
for mode in ("mean", "consensus", "reliability"):
    p, w, mse = run(mode)
    print(f"{mode:<14} {r2(Y, p):>+8.3f}   {w.min():.3f}/{w.max():.3f}")

print()
print("per-leaf own MSE at the end (reliability run):", np.round(np.sort(mse), 3))
