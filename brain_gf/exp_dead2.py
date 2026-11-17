"""Does the dead-zone hurt adaptation? And does skipping actually save time?"""
import time
import numpy as np
from brain_gf.nest_ops import OpBrain, signals, r2

SEG = 1000
def build():
    Xs, Ys = [], []
    for kind in ("sine", "square", "sine"):
        s = signals(SEG + 1, dim=8, seed=0, kind=kind)
        Xs.append(s[:-1]); Ys.append(s[1:])
    return np.concatenate(Xs), np.concatenate(Ys)

X, Y = build()

def run(dead_err):
    b = OpBrain(1, 8, 8, 128, 16, 8, agg="concat", combine="concat",
                seed=0, static_mask=True)
    p = np.empty_like(Y)
    skip = 0
    t0 = time.perf_counter()
    for i in range(len(X)):
        p[i] = b.step(X[i], Y[i], dead_err=dead_err)
        skip += b.skipped
    us = 1e6 * (time.perf_counter() - t0) / len(X)
    seg = [r2(Y[i*SEG:(i+1)*SEG], p[i*SEG:(i+1)*SEG]) for i in range(3)]
    bw = [r2(Y[SEG+a:SEG+b_], p[SEG+a:SEG+b_]) for a, b_ in ((0,100),(100,300),(800,1000))]
    return seg, bw, 100.0*skip/len(X), us

print(f"{'dead_err':>9} {'A':>7} {'B':>7} {'A2':>7} {'B[0:100]':>9} {'B[800:]':>8} {'skip%':>7} {'us':>6}")
for de in (0.0, 0.1, 0.2, 0.4):
    seg, bw, sk, us = run(de)
    print(f"{de:>9} {seg[0]:>+7.3f} {seg[1]:>+7.3f} {seg[2]:>+7.3f} "
          f"{bw[0]:>+9.3f} {bw[2]:>+8.3f} {sk:>6.1f}% {us:>6.0f}")
