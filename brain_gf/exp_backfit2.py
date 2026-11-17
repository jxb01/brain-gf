"""Three ways to run independent additive branches, and which one is stable.

    parallel LOO      : every branch updated each step on e_j = e + contrib_j   (diverges)
    damped LOO        : same but e_j = e + alpha*contrib_j, alpha<1
    sequential        : one branch per step, round-robin (classic backfitting)
"""
import numpy as np
from brain_gf.nest_ops import OpColumn, signals, r2
from brain_gf.nest_ops import OpBrain

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

def run(n, mode="sequential", lr=0.15, alpha=0.3, seed=0, floor=0.05):
    cols = bank(seed)
    cd = cols[0].out_dim
    wi = np.zeros((POOL, cd, 8))
    preds = np.empty_like(Y)
    rr = 0
    for i in range(len(X)):
        x = X[i]
        phis = [cols[j].forward(x) for j in range(n)]
        contrib = np.array([phis[j] @ wi[j] for j in range(n)])
        pred = contrib.sum(0)
        preds[i] = pred
        e = Y[i] - pred

        if mode == "parallel":
            for j in range(n):
                e_j = e + contrib[j]
                wi[j] += lr * np.outer(phis[j], e_j) / (float(phis[j] @ phis[j]) + floor)
        elif mode == "damped":
            for j in range(n):
                e_j = e + alpha * contrib[j]
                wi[j] += lr * np.outer(phis[j], e_j) / (float(phis[j] @ phis[j]) + floor)
        else:  # sequential: exactly one branch per step
            j = rr % n; rr += 1
            e_j = e + contrib[j]
            wi[j] += lr * np.outer(phis[j], e_j) / (float(phis[j] @ phis[j]) + floor)

        if not np.isfinite(pred).all():
            return None, None
    seg = [r2(Y[k*SEG:(k+1)*SEG], preds[k*SEG:(k+1)*SEG]) for k in range(3)]
    bw = [r2(Y[SEG+a:SEG+b_], preds[SEG+a:SEG+b_]) for a, b_ in ((0,100),(800,1000))]
    return seg, bw

print(f"{'config':<34} {'A':>9} {'B':>9} {'A2':>9} {'B[800:]':>9}")
for tag, mode, lr, al in [
    ("parallel LOO lr=0.05 floor=.05", "parallel", 0.05, 1.0),
    ("damped LOO a=0.3 lr=0.15", "damped", 0.15, 0.3),
    ("sequential lr=0.15", "sequential", 0.15, 1.0),
    ("sequential lr=0.3", "sequential", 0.3, 1.0),
]:
    seg, bw = run(8, mode=mode, lr=lr, alpha=al)
    if seg is None:
        print(f"{tag:<34} {'DIVERGED':>9}")
    else:
        print(f"{tag:<34} {seg[0]:>+9.3f} {seg[1]:>+9.3f} {seg[2]:>+9.3f} {bw[1]:>+9.3f}")

b = OpBrain(1, 8, 8, 128, 8, 8, agg="concat", combine="concat", seed=0, static_mask=True)
p = np.empty_like(Y)
for i in range(len(X)):
    p[i] = b.step(X[i], Y[i])
seg = [r2(Y[k*SEG:(k+1)*SEG], p[k*SEG:(k+1)*SEG]) for k in range(3)]
bw = r2(Y[SEG+800:SEG+1000], p[SEG+800:SEG+1000])
print(f"{'flat concat (reference)':<34} {seg[0]:>+9.3f} {seg[1]:>+9.3f} {seg[2]:>+9.3f} {bw:>+9.3f}")
