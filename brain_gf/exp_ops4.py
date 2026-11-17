"""Feature conditioning: does standardising the mixed-scale features explain the gap?"""
import time
import numpy as np
from brain_gf.nest_ops import OpBrain, signals, r2

N = 1200
A_ = signals(N + 1, dim=8, seed=0, kind="sine")
X, Y = A_[:-1], A_[1:]

def run(b, X, Y):
    p = np.empty_like(Y)
    t0 = time.perf_counter()
    for i in range(len(X)):
        p[i] = b.step(X[i], Y[i])
    return p, 1e6 * (time.perf_counter() - t0) / len(X)

print(f"{'config':<48} {'R2':>8} {'us/step':>9}")
for stdz in (False, True):
    for k in (8, 16):
        b = OpBrain(1, 8, 8, 128, k, 8, agg="concat", combine="concat",
                    seed=0, standardize=stdz)
        p, us = run(b, X, Y)
        print(f"{'math-op k=' + str(k) + ', ops=concat, standardize=' + str(stdz):<48} "
              f"{r2(Y, p):>+8.3f} {us:>9.0f}")
print()
print("frozen tanh/cos bank (all features already in [-1,1]) = +0.839")
