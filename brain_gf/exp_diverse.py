"""Does breaking column exchangeability give routing something to choose?"""
import numpy as np
from brain_gf.nest import NESTBrain, signals, r2

def run(b, X, Y, **kw):
    p = np.empty_like(Y)
    for i in range(len(X)):
        p[i] = b.step(X[i], Y[i], **kw)
    return p

N = 1500
A_ = signals(N + 1, dim=8, seed=0, kind="sine")
X, Y = A_[:-1], A_[1:]

print("exp1, single pass (old best = +0.584)")
for div in (False, True):
    for k_up in (False, True):
        b = NESTBrain(1, 8, 8, 128, 8, 4, 8, seed=0, diverse=div)
        p = run(b, X, Y, readout=True, k_update=k_up,
                dopamine="proj" if k_up else "scalar", lr_w=0.15)
        print(f"  diverse={str(div):<5} k_update={str(k_up):<5} R2={r2(Y, p):+.3f}")

print()
print("=== E again: warm up on A, freeze W, switch to B ===")
A = signals(800, dim=8, seed=1, kind="sine")
B = signals(800, dim=8, seed=2, kind="square")
XA, YA = A[:-1], A[1:]
XB, YB = B[:-1], B[1:]

for div in (False, True):
    base = NESTBrain(1, 8, 8, 128, 8, 4, 8, seed=0, diverse=div)
    run(base, XA, YA, readout=True, k_update=False, lr_w=0.15)
    W0 = base.W.copy()
    for tag, kw in [("K off", dict(k_update=False, dopamine="proj")),
                    ("K proj", dict(k_update=True, dopamine="proj"))]:
        b = NESTBrain(1, 8, 8, 128, 8, 4, 8, seed=0, diverse=div)
        run(b, XA, YA, readout=True, k_update=False, lr_w=0.15)
        b.W = W0.copy()
        pb = run(b, XB, YB, readout=True, lr_w=0.0, **kw)
        print(f"  diverse={str(div):<5} {tag:<7} R2(B first100)={r2(YB[:100], pb[:100]):+.3f} "
              f" R2(B last400)={r2(YB[-400:], pb[-400:]):+.3f}")
