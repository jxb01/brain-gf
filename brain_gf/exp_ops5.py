"""Does a second random mixing stage close the gap to the frozen tanh/cos bank?"""
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

print(f"{'config':<52} {'R2':>8} {'us/step':>9} {'dim':>6}")
for st2 in (0, 128, 256):
    for k in (8, 16):
        b = OpBrain(1, 8, 8, 128, k, 8, agg="concat", combine="concat",
                    seed=0, stage2=st2)
        p, us = run(b, X, Y)
        print(f"{'math-op k=' + str(k) + ' stage2=' + str(st2):<52} {r2(Y, p):>+8.3f} "
              f"{us:>9.0f} {b.W.shape[0]:>6}")
print()
print("frozen tanh/cos bank reference = +0.839")
