"""Does matching the operator family to the task close the gap to 0.839?"""
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

sets = {
    "random (9 ops, pick 4)": None,
    "linear+sin+cos+diff": ("linear", "sin", "cos", "diff"),
    "linear+sin+cos+cumsum": ("linear", "sin", "cos", "cumsum"),
    "linear+cos+cos+cos": ("linear", "cos", "cos", "cos"),
}
print(f"{'op set':<28} {'R2':>8} {'us/step':>9}")
for tag, ops in sets.items():
    b = OpBrain(1, 8, 8, 128, 8, 8, agg="concat", combine="concat", seed=0, op_set=ops)
    p, us = run(b, X, Y)
    print(f"{tag:<28} {r2(Y, p):>+8.3f} {us:>9.0f}")
print()
print("frozen tanh/cos bank reference (3 ops, 384 dims, no routing) = +0.839")
