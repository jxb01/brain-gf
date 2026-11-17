"""Is the top-k router itself the bottleneck? Sweep k up to dense (all columns)."""
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

print("math-op bank, 1 layer, ops concatenated in each column, readout = concat")
print(f"{'topk':>6} {'R2':>8} {'us/step':>9} {'feat dim':>9}")
for k in (4, 8, 16, 32, 64, 128):
    b = OpBrain(1, 8, 8, 128, k, 8, agg="concat", combine="concat", seed=0)
    p, us = run(b, X, Y)
    print(f"{k:>6} {r2(Y, p):>+8.3f} {us:>9.0f} {b.W.shape[0]:>9}")

print()
print("reference: frozen tanh/cos bank (FeatureReadout, 384 features, no routing) = +0.839")
