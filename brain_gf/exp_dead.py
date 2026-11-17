"""Dead-zone on the write: does gating small updates help?

Two axes:
  - accuracy (R2) and jitter (std of the per-step error)
  - fraction of steps skipped (the compute saving that makes the firmware story real)
"""
import numpy as np
from brain_gf.nest_ops import OpBrain, signals, r2

N = 1500
A_ = signals(N + 1, dim=8, seed=0, kind="sine")
Xs, Ys = A_[:-1], A_[1:]

def run(dead_err=0.0, dead_delta=0.0, k=16):
    b = OpBrain(1, 8, 8, 128, k, 8, agg="concat", combine="concat",
                seed=0, static_mask=True)
    p = np.empty_like(Ys)
    skip = 0
    for i in range(len(Xs)):
        p[i] = b.step(Xs[i], Ys[i], dead_err=dead_err, dead_delta=dead_delta)
        skip += b.skipped
    err = (Ys - p)
    return r2(Ys, p), err.std(), 100.0 * skip / len(Xs)

print("stationary stream, static16 + concat readout")
print(f"{'dead_err':>9} {'dead_delta':>11} {'R2':>8} {'err std':>9} {'skipped%':>9}")
for de in (0.0, 0.01, 0.02, 0.05, 0.1, 0.2):
    r, sd, sk = run(dead_err=de)
    print(f"{de:>9} {0.0:>11} {r:>+8.3f} {sd:>9.4f} {sk:>8.1f}%")
for dd in (0.001, 0.005, 0.02):
    r, sd, sk = run(dead_delta=dd)
    print(f"{0.0:>9} {dd:>11} {r:>+8.3f} {sd:>9.4f} {sk:>8.1f}%")
