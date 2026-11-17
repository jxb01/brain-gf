"""Does concatenating the 4 operators inside each column fix it?"""
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

print(f"{'config':<40} {'R2':>8} {'us/step':>9} {'feat dim':>9}")
for combine in ("mean", "concat"):
    for agg in ("concat", "sum", "cluster"):
        b = OpBrain(1, 8, 8, 128, 8, 8, agg=agg, combine=combine, seed=0)
        p, us = run(b, X, Y)
        print(f"{'1 layer, ops=' + combine + ', readout=' + agg:<40} {r2(Y, p):>+8.3f} "
              f"{us:>9.0f} {b.W.shape[0]:>9}")

print()
for L in (2, 3):
    b = OpBrain(L, 8, 8, 64, 8, 8, agg="concat", combine="concat", seed=0)
    p, us = run(b, X, Y)
    print(f"{str(L) + ' layers, ops=concat, readout=concat':<40} {r2(Y, p):>+8.3f} "
          f"{us:>9.0f} {b.W.shape[0]:>9}")
