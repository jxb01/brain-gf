"""Ordered deflation: each branch fits the residual of the FROZEN prefix.

    r_0 = y
    branch j fits r_j by ridge  ->  contrib_j,  r_{j+1} = r_j - contrib_j

No feedback (earlier branches never move again), so it cannot diverge, and every
branch's contribution is individually meaningful: it explains what the prefix left.
"""
import numpy as np
from brain_gf.nest_ops import OpColumn, signals

def build():
    Xs, Ys = [], []
    for kind in ("sine", "square", "sine"):
        s = signals(1000 + 1, dim=8, seed=0, kind=kind)
        Xs.append(s[:-1]); Ys.append(s[1:])
    return np.concatenate(Xs), np.concatenate(Ys)

X, Y = build()
split = 2500

rng = np.random.default_rng(0)
cols = [OpColumn(8, 8, rng, n_ops=4, combine="concat") for _ in range(8)]
F = [np.stack([c.forward(x) for x in X]) for c in cols]

def r2(y, p):
    return 1.0 - ((y-p)**2).sum() / ((y-y.mean(0))**2).sum()

def ridge(Fm, T, lam=1e-2):
    return np.linalg.solve(Fm.T@Fm + lam*np.eye(Fm.shape[1]), Fm.T@T)

resid = Y.copy()
cum = np.zeros_like(Y)
print(f"{'after branch':>13} {'train R2':>9} {'test R2':>9} {'branch gain':>12}")
for j in range(8):
    W = ridge(F[j][:split], resid[:split])
    contrib = F[j] @ W
    gain_tr = r2(resid[:split], contrib[:split])
    resid = resid - contrib
    cum = cum + contrib
    print(f"{j:>13} {r2(Y[:split], cum[:split]):>+9.3f} {r2(Y[split:], cum[split:]):>+9.3f} {gain_tr:>+12.3f}")

print()
print("reference: single readout on the concatenation of all 8 blocks (ridge) = "
      f"{r2(Y[split:], np.stack([np.concatenate([F[k][i] for k in range(8)]) for i in range(len(X))])[split:] @ ridge(np.stack([np.concatenate([F[k][i] for k in range(8)]) for i in range(len(X))])[:split], Y[:split])):+.3f}")
