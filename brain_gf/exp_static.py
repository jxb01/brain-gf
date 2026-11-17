"""Is per-step routing the bottleneck? Compare dynamic vs static column selection."""
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

print(f"{'selection':<42} {'R2':>8} {'us/step':>9} {'dim':>6}")
for static in (False, True):
    for k in (4, 8, 16):
        b = OpBrain(1, 8, 8, 128, k, 8, agg="concat", combine="concat",
                    seed=0, static_mask=static)
        p, us = run(b, X, Y)
        tag = ("static (fixed " + str(k) + " of 128)") if static else ("dynamic top-" + str(k) + " of 128")
        print(f"{tag:<42} {r2(Y, p):>+8.3f} {us:>9.0f} {b.W.shape[0]:>6}")
print()
print("frozen tanh/cos bank (all 384 features, always on) = +0.839")
