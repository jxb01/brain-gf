"""Tree version after the two fixes: no a_j on branches, per-track normalisation."""
import time
import numpy as np
from brain_gf.nest_tree import TreeBrain, signals, r2

N = 1200
A_ = signals(N + 1, dim=8, seed=0, kind="sine")
X, Y = A_[:-1], A_[1:]

def run(b, X, Y, **kw):
    p = np.empty_like(Y)
    t0 = time.perf_counter()
    for i in range(len(X)):
        p[i] = b.step(X[i], Y[i], **kw)
    return p, 1e6 * (time.perf_counter() - t0) / len(X)

print("experiment 1: single layer (flattened baseline +0.584, concat-readout +0.454)")
b = TreeBrain(1, 8, 8, 128, 8, 4, 8, seed=0)
p, us = run(b, X, Y)
print(f"  tree layers=1  R2={r2(Y, p):+.3f}  {us:.0f} us/step  trainable={b.counts()['trainable']}")

print()
print("experiment 2: depth, 32 columns (flattened was 0.584/0.585/0.584)")
print(f"  {'layers':>7} {'R2':>8} {'us/step':>9}")
for L in (1, 2, 4):
    b = TreeBrain(L, 8, 8, 32, 8, 4, 8, seed=0)
    p, us = run(b, X, Y)
    print(f"  {L:>7} {r2(Y, p):>+8.3f} {us:>9.0f}")
