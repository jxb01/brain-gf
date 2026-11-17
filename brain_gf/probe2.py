"""Round 2: is the convex blend the bottleneck? And how does the feature-readout brain do
on the exact same stream?"""
import numpy as np
from brain_gf.nrir import NRIRBrain
from brain_gf.gf_brain import FeatureReadoutBrain

def task_a(x):
    return (np.sin(x[:, 0]) + x[:, 1] * x[:, 2])[:, None]

def task_b(x):
    return (np.tanh(2.0 * x[:, 3] - x[:, 4]) - 0.7 * np.cos(1.7 * x[:, 0]))[:, None]

def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()

def run(brain, X, Y, epochs, attn=False):
    hist = []
    for _ in range(epochs):
        preds = np.empty_like(Y)
        for i in range(len(X)):
            if attn:
                preds[i] = brain.step(X[i], Y[i])
            else:
                _s, preds[i] = brain.step(X[i], Y[i])
        hist.append(r2(Y, preds))
    return hist

N, EP = 800, 6
rng = np.random.default_rng(0)
X = rng.standard_normal((N, 8))
Y = task_a(X)

print("== blend vs residual (dim=8, layers=4, g=0.5) ==")
for residual in (False, True):
    for renorm in (True, False):
        b = NRIRBrain(dim=8, out_dim=1, residual=residual, renorm=renorm, seed=0)
        h = run(b, X, Y, EP, attn=True)
        print(f"  residual={str(residual):<5} renorm={str(renorm):<5}  R2={h[-1]:+.3f}")

print()
print("== same task, feature-readout brain ==")
for decay in (1.0, 0.999):
    b = FeatureReadoutBrain(8, 1, seed=0, decay=decay, lr_decision=0.05)
    h = run(b, X, Y, EP)
    print(f"  feature_readout decay={decay:<6} R2={h[-1]:+.3f}  curve={[round(float(v),3) for v in h]}")

print()
print("== A->B->A switch, same stream ==")
segs = [("A", 1200), ("B", 1000), ("A", 1000)]
rng2 = np.random.default_rng(0)
Xs, Ys = [], []
for name, n in segs:
    x = rng2.standard_normal((n, 8))
    Xs.append(x); Ys.append(task_a(x) if name == "A" else task_b(x))
X2, Y2 = np.concatenate(Xs), np.concatenate(Ys)

def report(tag, preds):
    parts, s = [], 0
    for name, n in segs:
        sl = slice(s, s + n)
        parts.append(f"{name}:{r2(Y2[sl], preds[sl]):+.3f}")
        s += n
    print(f"  {tag:<34} " + "  ".join(parts))

for decay in (1.0, 0.9999, 0.999):
    b = FeatureReadoutBrain(8, 1, seed=0, decay=decay)
    preds = np.empty_like(Y2)
    for i in range(len(X2)):
        _s, preds[i] = b.step(X2[i], Y2[i])
    report(f"feature_readout decay={decay}", preds)

for residual in (False, True):
    b = NRIRBrain(dim=8, out_dim=1, residual=residual, renorm=not residual, seed=0)
    preds = np.empty_like(Y2)
    for i in range(len(X2)):
        preds[i] = b.step(X2[i], Y2[i])
    report(f"nrir residual={residual}", preds)
