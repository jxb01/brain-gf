"""Self-tuning gate: dead_err = 0.5 * running std of the recent error.

After a regime switch the error explodes -> the gate opens by itself.
Once the model settles, the gate closes and the writes stop.
"""
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

def run(mode, scale=0.5, fixed=0.0):
    b = OpBrain(1, 8, 8, 128, 16, 8, agg="concat", combine="concat",
                seed=0, static_mask=True)
    p = np.empty_like(Y)
    skip = 0
    var = 1.0                     # running second moment of the error
    for i in range(len(X)):
        pred, _ = b.forward(X[i])
        e = Y[i] - pred
        # running scale of the error (EMA of squared error)
        var = 0.98 * var + 0.02 * float(e @ e) / e.size
        theta = fixed if mode == "fixed" else scale * np.sqrt(var)
        p[i] = b.step(X[i], Y[i], dead_err=theta)
        skip += b.skipped
    seg = [r2(Y[i*SEG:(i+1)*SEG], p[i*SEG:(i+1)*SEG]) for i in range(3)]
    bw = [r2(Y[SEG+a:SEG+b_], p[SEG+a:SEG+b_]) for a, b_ in ((0,100),(100,300),(800,1000))]
    return seg, bw, 100.0*skip/len(X), theta

print(f"{'gate':<22} {'A':>7} {'B':>7} {'A2':>7} {'B[0:100]':>9} {'B[800:]':>8} {'skip%':>7}")
seg, bw, sk, _ = run("fixed", fixed=0.0)
print(f"{'none':<22} {seg[0]:>+7.3f} {seg[1]:>+7.3f} {seg[2]:>+7.3f} {bw[0]:>+9.3f} {bw[2]:>+8.3f} {sk:>6.1f}%")
for sc in (0.3, 0.5, 1.0):
    seg, bw, sk, th = run("adapt", scale=sc)
    print(f"{'adaptive x' + str(sc):<22} {seg[0]:>+7.3f} {seg[1]:>+7.3f} {seg[2]:>+7.3f} "
          f"{bw[0]:>+9.3f} {bw[2]:>+8.3f} {sk:>6.1f}%")
