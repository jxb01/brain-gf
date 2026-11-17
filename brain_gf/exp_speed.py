"""Adaptation speed on the switch: windowed R2 inside regime B."""
import numpy as np
from brain_gf.nest_ops import OpBrain, signals, r2
from brain_gf.gf_brain import FeatureReadoutBrain

SEG = 1000
def build():
    Xs, Ys = [], []
    for kind in ("sine", "square", "sine"):
        s = signals(SEG + 1, dim=8, seed=0, kind=kind)
        Xs.append(s[:-1]); Ys.append(s[1:])
    return np.concatenate(Xs), np.concatenate(Ys)

X, Y = build()

def run_op(decay=1.0, static=True, k=16, agg="concat"):
    b = OpBrain(1, 8, 8, 128, k, 8, agg=agg, combine="concat", seed=0, static_mask=static)
    p = np.empty_like(Y)
    for i in range(len(X)):
        p[i] = b.step(X[i], Y[i], decay=decay)
    return p

def run_fr(decay=1.0):
    b = FeatureReadoutBrain(8, 8, lr_decision=0.15, decay=decay, seed=0)
    p = np.empty_like(Y)
    for i in range(len(X)):
        _s, p[i] = b.step(X[i], Y[i])
    return p

def windows(p, name):
    s = {"A": 0, "B": SEG, "A2": 2 * SEG}[name]
    out = []
    for a, b_ in ((0, 100), (100, 300), (300, 600), (800, 1000)):
        out.append(r2(Y[s + a:s + b_], p[s + a:s + b_]))
    return out

print("R2 inside regime B at increasing distance from the switch")
print(f"{'config':<28} {'B[0:100]':>9} {'B[100:300]':>11} {'B[300:600]':>11} {'B[800:1k]':>10}")
for tag, p in [
    ("op static16 decay=1.0", run_op()),
    ("op dynamic16 decay=1.0", run_op(static=False)),
    ("op static16 cluster", run_op(agg="cluster")),
    ("FR decay=1.0", run_fr()),
]:
    w = windows(p, "B")
    print(f"{tag:<28} {w[0]:>+9.3f} {w[1]:>+11.3f} {w[2]:>+11.3f} {w[3]:>+10.3f}")
