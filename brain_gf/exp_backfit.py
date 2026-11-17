"""Independent branches, no gating, additive output, leave-one-out credit.

    pred      = sum_j contrib_j                     (plain sum, no gates)
    contrib_j = phi_j(x) @ w_j
    e         = y - pred
    e_j       = e + contrib_j                       (what the others did NOT explain)
    w_j      += lr * phi_j (x) e_j / |phi_j|^2      (each branch fits its own leftover)

This is backfitting: every branch is responsible for its own residual, so adding a
branch does not re-weight the others (its readout starts at 0 => contrib 0), and no
branch ever "explains the same part twice".
"""
import numpy as np
from brain_gf.nest_ops import OpBrain, signals, r2
from brain_gf.nest_ops import OpColumn

SEG = 1000
def build():
    Xs, Ys = [], []
    for kind in ("sine", "square", "sine"):
        s = signals(SEG + 1, dim=8, seed=0, kind=kind)
        Xs.append(s[:-1]); Ys.append(s[1:])
    return np.concatenate(Xs), np.concatenate(Ys)

X, Y = build()
POOL = 32

def bank(seed=0):
    rng = np.random.default_rng(seed)
    return [OpColumn(8, 8, rng, n_ops=4, combine="concat") for _ in range(POOL)]

def run(n0, n_max, lr=0.15, grow=None, cooldown=120, seed=0):
    cols = bank(seed)
    cd = cols[0].out_dim
    wi = np.zeros((POOL, cd, 8))
    n = n0
    last = -10**9; var = 1.0
    preds = np.empty_like(Y); events = []
    drift = None
    base_snapshot = None
    for i in range(len(X)):
        x = X[i]
        phis = [cols[j].forward(x) for j in range(n)]
        contrib = np.array([phis[j] @ wi[j] for j in range(n)])
        pred = contrib.sum(0)
        preds[i] = pred
        e = Y[i] - pred

        for j in range(n):
            e_j = e + contrib[j]                      # leave-one-out residual
            wi[j] += lr * np.outer(phis[j], e_j) / (float(phis[j] @ phis[j]) + 1e-8)

        var = 0.98 * var + 0.02 * float(e @ e) / e.size
        if grow is not None and n < n_max and np.sqrt(var) > grow and (i - last) > cooldown:
            n += 1; last = i; events.append(i)
    seg = [r2(Y[k*SEG:(k+1)*SEG], preds[k*SEG:(k+1)*SEG]) for k in range(3)]
    bw = [r2(Y[SEG+a:SEG+b_], preds[SEG+a:SEG+b_]) for a, b_ in ((0,100),(800,1000))]
    return seg, bw, n, events

def run_flat(k=8, seed=0):
    b = OpBrain(1, 8, 8, 128, k, 8, agg="concat", combine="concat", seed=seed, static_mask=True)
    p = np.empty_like(Y)
    for i in range(len(X)):
        p[i] = b.step(X[i], Y[i])
    seg = [r2(Y[j*SEG:(j+1)*SEG], p[j*SEG:(j+1)*SEG]) for j in range(3)]
    bw = [r2(Y[SEG+a:SEG+b_], p[SEG+a:SEG+b_]) for a, b_ in ((0,100),(800,1000))]
    return seg, bw

print(f"{'config':<30} {'A':>7} {'B':>7} {'A2':>7} {'B[0:100]':>9} {'B[800:]':>8} {'n':>3}")
seg, bw = run_flat()
print(f"{'flat concat (reference)':<30} {seg[0]:>+7.3f} {seg[1]:>+7.3f} {seg[2]:>+7.3f} "
      f"{bw[0]:>+9.3f} {bw[1]:>+8.3f} {'-':>3}")
for tag, kw in [
    ("backfit n=8", dict(n0=8, n_max=8)),
    ("backfit n=16", dict(n0=16, n_max=16)),
    ("backfit n=32", dict(n0=32, n_max=32)),
    ("backfit grow 8..32 t=0.6", dict(n0=8, n_max=32, grow=0.6)),
]:
    seg, bw, n, ev = run(**kw)
    print(f"{tag:<30} {seg[0]:>+7.3f} {seg[1]:>+7.3f} {seg[2]:>+7.3f} "
          f"{bw[0]:>+9.3f} {bw[1]:>+8.3f} {n:>3}")
