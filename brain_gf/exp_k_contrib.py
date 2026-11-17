"""Does k contribute, and through what?

Three separate questions:
  1. static selection, k = 2..64          -> effect of #columns (and #features)
  2. fixed 512 features, split differently -> 8x8 ops vs 16x4 vs 32x2
  3. dynamic vs static at the same k      -> effect of the routing mechanism itself
"""
import numpy as np
from brain_gf.nest_ops import OpBrain, signals, r2

N = 1200
A_ = signals(N + 1, dim=8, seed=0, kind="sine")
X, Y = A_[:-1], A_[1:]

def run(k, n_ops=4, static=True, seg=None):
    b = OpBrain(1, 8, 8, 128, k, 8, agg="concat", combine="concat",
                seed=0, static_mask=static, n_ops=n_ops)
    p = np.empty_like(Y)
    for i in range(len(X)):
        p[i] = b.step(X[i], Y[i])
    return r2(Y, p), b.W.shape[0]

print("1) static selection, vary k (features = k * 4 ops * 8 dims)")
print(f"{'k':>4} {'features':>9} {'R2':>8}")
for k in (2, 4, 8, 16, 32, 64):
    r, d = run(k)
    print(f"{k:>4} {d:>9} {r:>+8.3f}")

print()
print("2) fixed 512 features, different column/op split (same #features)")
print(f"{'k':>4} {'ops/col':>8} {'features':>9} {'R2':>8}")
for k, ops in ((64, 1), (32, 2), (16, 4), (8, 8)):
    r, d = run(k, n_ops=ops)
    print(f"{k:>4} {ops:>8} {d:>9} {r:>+8.3f}")
