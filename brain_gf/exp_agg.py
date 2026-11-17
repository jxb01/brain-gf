"""Aggregation comparison: concat vs sum vs consensus, and whether consensus ever differs.

If the leaves are near-identical (all mixtures of the same random bank), the consensus
weights collapse to uniform and the method degenerates to plain sum.
"""
import time
import numpy as np
from brain_gf.nest_tree import TreeBrain, signals, r2

N = 1200
A_ = signals(N + 1, dim=8, seed=0, kind="sine")
X, Y = A_[:-1], A_[1:]

def run(b, X, Y):
    p = np.empty_like(Y)
    t0 = time.perf_counter()
    for i in range(len(X)):
        p[i] = b.step(X[i], Y[i])
    return p, 1e6 * (time.perf_counter() - t0) / len(X)

for L in (1, 2):
    cols = 128 if L == 1 else 32
    print(f"--- layers={L}, columns={cols}, k=8 ---")
    for agg in ("concat", "sum", "consensus"):
        b = TreeBrain(L, 8, 8, cols, 8, 4, 8, seed=0, agg=agg)
        p, us = run(b, X, Y)
        w = b.last_consensus
        extra = ""
        if w is not None:
            extra = f"  consensus w: min={w.min():.3f} max={w.max():.3f} std={w.std():.4f}"
        print(f"  agg={agg:<10} R2={r2(Y, p):+.3f}  {us:6.0f} us/step{extra}")

print()
print("--- leaf diversity at the last layer (cosine between branches) ---")
b = TreeBrain(1, 8, 8, 128, 8, 4, 8, seed=0, agg="consensus")
feats_log = []
for i in range(300):
    b.step(X[i], Y[i])
    if i % 50 == 0:
        _p, _f = b.forward(X[i])
print("  (consensus weights over 1200 steps: see above; uniform = 0.125 each)")
b2 = TreeBrain(1, 8, 8, 128, 8, 4, 8, seed=0, agg="sum")
tracks_cos = []
for i in range(300):
    b2.step(X[i], Y[i])
_, f0 = b2.forward(X[0])
print(f"  readout feature dim: concat={(8*8)}, sum/consensus={8}")
