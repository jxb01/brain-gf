"""Is it the routing, or the information capacity of the state?

Same brain, same stream, fewer oscillators to hold:
  out_dim = 8 -> 4 -> 2 -> 1   (state is 8-dim either way)
If R2 climbs as out_dim falls, the ceiling is capacity, not routing.
"""
import numpy as np
from brain_gf.nest import NESTBrain, signals, r2

N = 1500
A_ = signals(N + 1, dim=8, seed=0, kind="sine")
X, Yfull = A_[:-1], A_[1:]

def run(b, X, Y, **kw):
    p = np.empty_like(Y)
    for i in range(len(X)):
        p[i] = b.step(X[i], Y[i], **kw)
    return p

print("hidden=8, single layer, predicting k of the 8 oscillators")
for k in (8, 4, 2, 1):
    b = NESTBrain(1, 8, 8, 128, 8, 4, k, seed=0)
    p = run(b, X, Yfull[:, :k], readout=True, k_update=False, lr_w=0.15)
    print(f"  out_dim={k}   R2={r2(Yfull[:, :k], p):+.3f}")

print()
print("hidden=32/64, predicting all 8 (state widened, same task)")
for h in (8, 32, 64):
    b = NESTBrain(1, 8, h, 128, 8, 4, 8, seed=0)
    p = run(b, X, Yfull, readout=True, k_update=False, lr_w=0.15)
    print(f"  hidden={h:<3} R2={r2(Yfull, p):+.3f}")

print()
print("reference: same task with the direct nonlinear feature bank (no recurrence)")
from brain_gf.gf_brain import FeatureReadoutBrain
for k in (8, 2):
    b = FeatureReadoutBrain(8, k, lr_decision=0.15, decay=1.0, seed=0)
    p = np.empty_like(Yfull[:, :k])
    for i in range(len(X)):
        _s, p[i] = b.step(X[i], Yfull[i, :k])
    print(f"  feature_readout out_dim={k} R2={r2(Yfull[:, :k], p):+.3f}")
