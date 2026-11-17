"""Does reading the columns individually (instead of their softmax mixture) unlock it?"""
import numpy as np
from brain_gf.nest import NESTBrain, signals, r2

N = 1500
A_ = signals(N + 1, dim=8, seed=0, kind="sine")
X, Y = A_[:-1], A_[1:]

def run(b, X, Y, **kw):
    p = np.empty_like(Y)
    for i in range(len(X)):
        p[i] = b.step(X[i], Y[i], **kw)
    return p

print("single layer, hidden=8, 128 columns, top-k=8")
for mode in ("mix", "concat"):
    for h in (8, 32):
        b = NESTBrain(1, 8, h, 128, 8, 4, 8, seed=0, read_mode=mode)
        p = run(b, X, Y, readout=True, k_update=False, lr_w=0.15)
        print(f"  read_mode={mode:<7} hidden={h:<3} R2={r2(Y, p):+.3f}")

print()
print("depth under concat readout")
for L in (1, 2, 4):
    b = NESTBrain(L, 8, 8, 64, 8, 4, 8, seed=0, read_mode="concat")
    p = run(b, X, Y, readout=True, k_update=False, lr_w=0.15)
    print(f"  layers={L} R2={r2(Y, p):+.3f}")
