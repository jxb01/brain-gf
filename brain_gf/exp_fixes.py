"""Run the three corrections in the order we agreed.

  E : mute W after a warm-up, then ask whether K alone can take over on a NEW regime
  3 : drop per-layer normalisation, add a loose energy budget, LayerNorm before readout
  4 : projection-form dopamine (first-order ablation credit)
  2 : ablation survival credit with lambda matched to the credit scale
"""
import numpy as np
from brain_gf.nest import NESTBrain, signals, r2

def seq(n, kind, seed, dim=8):
    return signals(n, dim=dim, seed=seed, kind=kind)

def run(brain, X, Y, **kw):
    p = np.empty_like(Y)
    for i in range(len(X)):
        p[i] = brain.step(X[i], Y[i], **kw)
    return p

# ---------------------------------------------------------------- E
print("=== E: W warmed up then frozen; does K adapt on a new regime? ===")
A = seq(800, "sine", 1)
B = seq(800, "square", 2)
XA, YA = A[:-1], A[1:]
XB, YB = B[:-1], B[1:]

base = NESTBrain(1, 8, 8, 128, 8, 4, 8, seed=0)
run(base, XA, YA, readout=True, k_update=False, lr_w=0.15)          # warm-up
W_frozen = base.W.copy()
print(f"  after warm-up on A: R2(A last 300) = {r2(YA[-300:], run(base, XA[-300:], YA[-300:], readout=True, k_update=False, lr_w=0.0)):+.3f}")

for tag, kw in [
    ("K off (only frozen W)", dict(k_update=False, dopamine="proj")),
    ("K proj dopamine",       dict(k_update=True,  dopamine="proj")),
    ("K scalar (as written)", dict(k_update=True,  dopamine="scalar")),
]:
    b = NESTBrain(1, 8, 8, 128, 8, 4, 8, seed=0)
    run(b, XA, YA, readout=True, k_update=False, lr_w=0.15)          # same warm-up
    b.W = W_frozen.copy()
    pb = run(b, XB, YB, readout=True, lr_w=0.0, **kw)                # W frozen now
    print(f"  {tag:<24} R2(B first100)={r2(YB[:100], pb[:100]):+.3f}  "
          f"R2(B last400)={r2(YB[-400:], pb[-400:]):+.3f}")

# ---------------------------------------------------------------- fix 3 + fix 4
print()
print("=== fix 3 (+4): no per-layer norm, budget=10, LayerNorm before readout ===")
N = 1500
A_ = seq(N + 1, "sine", 0)
X, Y = A_[:-1], A_[1:]

cfgs = [
    ("renorm=True  (old)",           dict(renorm=True)),
    ("renorm=False budget=10 preLN", dict(renorm=False, budget=10.0, pre_ln=True)),
    ("renorm=False budget=20 preLN", dict(renorm=False, budget=20.0, pre_ln=True)),
    ("renorm=False budget=50 preLN", dict(renorm=False, budget=50.0, pre_ln=True)),
]
for tag, c in cfgs:
    b = NESTBrain(1, 8, 8, 128, 8, 4, 8, seed=0, **c)
    p = run(b, X, Y, readout=True, k_update=False, lr_w=0.15)
    print(f"  {tag:<30} R2={r2(Y, p):+.3f}")

print()
print("  depth under the fixed variant (budget=10, preLN):")
for L in (1, 2, 4):
    b = NESTBrain(L, 8, 8, 64, 8, 4, 8, seed=0, renorm=False, budget=10.0, pre_ln=True)
    p = run(b, X, Y, readout=True, k_update=True, dopamine="proj", lr_w=0.15)
    print(f"    layers={L}  R2={r2(Y, p):+.3f}")

# ---------------------------------------------------------------- fix 2
print()
print("=== fix 2: ablation survival credit, lambda matched to credit scale ===")
rng = np.random.default_rng(0)
n = 4000
t = np.arange(n)
sig1 = np.sin(2 * np.pi * 0.18 * t) + 0.5 * np.sin(2 * np.pi * 0.01 * t)
sig1[rng.integers(0, n, 60)] += 3.0
Xm = np.stack([sig1, np.roll(sig1, 1), np.roll(sig1, 2)], 1)
Xm = np.concatenate([Xm, rng.standard_normal((n, 5)) * 0.01], 1)
Ym = np.roll(sig1, -1)[:, None]

for mode, lam in (("spec", 0.1), ("ablate", 0.1), ("ablate", 0.001), ("ablate", 0.0)):
    b = NESTBrain(1, 8, 8, 128, 8, 4, 1, seed=0)
    run(b, Xm, Ym, readout=True, k_update=True, dopamine="proj",
        survival=mode, lam_cost=lam, wake_interval=100000)
    lay = b.layers[0]
    act, surv = lay.act_count, lay.survival
    live = act > 0
    print(f"  survival={mode:<7} lam={lam:<6} corr(act,surv)={np.corrcoef(act, surv)[0,1]:+.3f}  "
          f"mean surv(fired)={surv[live].mean():+.4f}")
