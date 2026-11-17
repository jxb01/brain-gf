"""Math-op bank vs GRU bank; concat vs sum vs cluster readout."""
import time
import numpy as np
from brain_gf.nest_ops import OpBrain, signals, r2, cluster_readout

N = 1200
A_ = signals(N + 1, dim=8, seed=0, kind="sine")
X, Y = A_[:-1], A_[1:]

def run(b, X, Y):
    p = np.empty_like(Y)
    t0 = time.perf_counter()
    for i in range(len(X)):
        p[i] = b.step(X[i], Y[i])
    return p, 1e6 * (time.perf_counter() - t0) / len(X)

print("reference points from earlier runs (same data, single pass):")
print("  GRU bank + concat   = +0.560")
print("  frozen tanh/cos bank (FeatureReadout, lr 0.15) = +0.839")
print()
print(f"{'bank / agg':<26} {'R2':>8} {'us/step':>9} {'clusters':>9}")
for agg in ("concat", "sum", "cluster"):
    b = OpBrain(1, 8, 8, 128, 8, 8, agg=agg, seed=0)
    p, us = run(b, X, Y)
    nc = np.mean(b.n_clusters) if b.n_clusters else 0
    print(f"{'math-op 1 layer / ' + agg:<26} {r2(Y, p):>+8.3f} {us:>9.0f} {nc:>9.2f}")

for L in (2, 3):
    b = OpBrain(L, 8, 8, 64, 8, 8, agg="concat", seed=0)
    p, us = run(b, X, Y)
    print(f"{'math-op ' + str(L) + ' layers / concat':<26} {r2(Y, p):>+8.3f} {us:>9.0f}")
