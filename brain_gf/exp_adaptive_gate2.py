"""Self-tuning gate, causal (uses the previous step's error, one forward per step)."""
import numpy as np
from brain_gf.nest_ops import OpBrain, signals, r2

SEG = 1000
def build():
    Xs, Ys = [], []
    for kind in ("sine", "square", "sine"):
        s = signals(SEG + 1, dim=8, seed=0, kind=kind)
        Xs.append(s[:-1]); Ys.append(s[1:])
    return np.concatenate(Xs), np.concatenate(Ys)

X, Y = build()

def run(scale):
    b = OpBrain(1, 8, 8, 128, 16, 8, agg="concat", combine="concat",
                seed=0, static_mask=True)
    p = np.empty_like(Y)
    skip = 0
    var = 1.0
    for i in range(len(X)):
        theta = scale * np.sqrt(var)
        p[i] = b.step(X[i], Y[i], dead_err=theta)
        skip += b.skipped
        e = Y[i] - p[i]
        var = 0.98 * var + 0.02 * float(e @ e) / e.size
    seg = [r2(Y[i*SEG:(i+1)*SEG], p[i*SEG:(i+1)*SEG]) for i in range(3)]
    bw = [r2(Y[SEG+a:SEG+b_], p[SEG+a:SEG+b_]) for a, b_ in ((0,100),(100,300),(800,1000))]
    return seg, bw, 100.0*skip/len(X)

print(f"{'gate (theta = c * rms(e))':<26} {'A':>7} {'B':>7} {'A2':>7} {'B[0:100]':>9} {'B[800:]':>8} {'skip%':>7}")
for sc in (0.0, 0.25, 0.5, 0.75, 1.0):
    seg, bw, sk = run(sc)
    tag = "none" if sc == 0 else f"c = {sc}"
    print(f"{tag:<26} {seg[0]:>+7.3f} {seg[1]:>+7.3f} {seg[2]:>+7.3f} "
          f"{bw[0]:>+9.3f} {bw[2]:>+8.3f} {sk:>6.1f}%")
