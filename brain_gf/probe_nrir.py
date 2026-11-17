"""Probe: where does NRIR lose the R^2 ? readout width, or the sphere constraint?"""
import numpy as np
from brain_gf.nrir import NRIRBrain

rng_master = 0

def task_a(x):
    return (np.sin(x[:, 0]) + x[:, 1] * x[:, 2])[:, None]

def task_b(x):
    return (np.tanh(2.0 * x[:, 3] - x[:, 4]) - 0.7 * np.cos(1.7 * x[:, 0]))[:, None]

def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()

def run(brain, X, Y, epochs):
    hist = []
    for _ in range(epochs):
        preds = np.empty_like(Y)
        for i in range(len(X)):
            preds[i] = brain.step(X[i], Y[i])
        hist.append(r2(Y, preds))
    return hist

N, EP = 800, 6
rng = np.random.default_rng(rng_master)
X = rng.standard_normal((N, 8))
Y = task_a(X)

print("== readout / renorm ==")
for readout in ("final", "concat"):
    for renorm in (True, False):
        b = NRIRBrain(dim=8, out_dim=1, readout=readout, renorm=renorm, seed=0)
        h = run(b, X, Y, EP)
        print(f"  readout={readout:<7} renorm={str(renorm):<5}  R2 last={h[-1]:+.3f}  curve={[round(v,3) for v in h]}")

print()
print("== decay sweep (A -> B -> A, 1200/1000/1000) ==")
segs = [("A", 1200), ("B", 1000), ("A", 1000)]
rng2 = np.random.default_rng(rng_master)
Xs, Ys = [], []
for name, n in segs:
    x = rng2.standard_normal((n, 8))
    Xs.append(x)
    Ys.append(task_a(x) if name == "A" else task_b(x))
X2, Y2 = np.concatenate(Xs), np.concatenate(Ys)

for decay in (1.0, 0.9999, 0.999):
    for readout in ("final", "concat"):
        b = NRIRBrain(dim=8, out_dim=1, readout=readout, decay=decay, seed=0)
        preds = np.empty_like(Y2)
        for i in range(len(X2)):
            preds[i] = b.step(X2[i], Y2[i])
        parts, s = [], 0
        for name, n in segs:
            sl = slice(s, s + n)
            parts.append(f"{name}:{r2(Y2[sl], preds[sl]):+.3f}")
            s += n
        print(f"  decay={decay:<7} readout={readout:<7} " + "  ".join(parts))
