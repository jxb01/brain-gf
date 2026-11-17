"""Where a gate actually pays: long stationary stretch, measure skip% near convergence."""
import numpy as np
from brain_gf.nest_ops import OpBrain, signals, r2

N = 4000
A_ = signals(N + 1, dim=8, seed=0, kind="sine")
X, Y = A_[:-1], A_[1:]

def run(scale):
    b = OpBrain(1, 8, 8, 128, 16, 8, agg="concat", combine="concat",
                seed=0, static_mask=True)
    p = np.empty_like(Y)
    skipped = np.zeros(N, dtype=bool)
    var = 1.0
    for i in range(N):
        theta = scale * np.sqrt(var)
        p[i] = b.step(X[i], Y[i], dead_err=theta)
        skipped[i] = bool(b.skipped)
        e = Y[i] - p[i]
        var = 0.98 * var + 0.02 * float(e @ e) / e.size
    head = 100.0 * skipped[:1000].mean()
    tail = 100.0 * skipped[-1000:].mean()
    return r2(Y, p), r2(Y[-1000:], p[-1000:]), head, tail

print(f"{'gate':<14} {'R2 all':>8} {'R2 tail':>8} {'skip% first1k':>14} {'skip% last1k':>13}")
for sc in (0.0, 0.5, 1.0, 1.5):
    r_all, r_tail, h, t = run(sc)
    print(f"{('none' if sc == 0 else 'c=' + str(sc)):<14} {r_all:>+8.3f} {r_tail:>+8.3f} "
          f"{h:>13.1f}% {t:>12.1f}%")
