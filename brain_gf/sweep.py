"""Capacity x plasticity sweep on the A -> B -> A stream (single pass, no epochs)."""
import numpy as np
from brain_gf.gf_brain import FeatureReadoutBrain
from brain_gf.nrir import NRIRBrain

def task_a(x): return (np.sin(x[:, 0]) + x[:, 1] * x[:, 2])[:, None]
def task_b(x): return (np.tanh(2.0 * x[:, 3] - x[:, 4]) - 0.7 * np.cos(1.7 * x[:, 0]))[:, None]
def r2(y, p): return 1 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()

SEGS = [("A", 1200), ("B", 1000), ("A", 1000)]
rng = np.random.default_rng(0)
Xs, Ys = [], []
for name, n in SEGS:
    x = rng.standard_normal((n, 8))
    Xs.append(x); Ys.append(task_a(x) if name == "A" else task_b(x))
X, Y = np.concatenate(Xs), np.concatenate(Ys)

def score(preds):
    out, s = [], 0
    for name, n in SEGS:
        sl = slice(s, s + n)
        out.append(r2(Y[sl], preds[sl])); s += n
    return out

def run_fr(hid, hid2, decay):
    b = FeatureReadoutBrain(8, 1, hid=hid, hid2=hid2, decay=decay, seed=0)
    p = np.empty_like(Y)
    for i in range(len(X)):
        _s, p[i] = b.step(X[i], Y[i])
    return score(p)

print("== feature_readout: plasticity knob (hid=128/128) ==")
print(f"{'decay':>8} {'A':>7} {'B':>7} {'A2':>7} {'mean':>7}")
for decay in (1.0, 0.9999, 0.9995, 0.999):
    a, b_, a2 = run_fr(128, 128, decay)
    print(f"{decay:>8} {a:>+7.3f} {b_:>+7.3f} {a2:>+7.3f} {(a+b_+a2)/3:>+7.3f}")

print()
print("== feature_readout: capacity knob (decay=1.0) ==")
print(f"{'hid':>8} {'A':>7} {'B':>7} {'A2':>7} {'mean':>7} {'trainable':>10}")
for hid in (8, 32, 128):
    a, b_, a2 = run_fr(hid, hid, 1.0)
    n = (2*hid + hid) * (8 + 1)
    print(f"{hid:>8} {a:>+7.3f} {b_:>+7.3f} {a2:>+7.3f} {(a+b_+a2)/3:>+7.3f} {n:>10}")

print()
print("== nrir: interpolation knob g (blend, renorm=False) ==")
print(f"{'g':>8} {'A':>7} {'B':>7} {'A2':>7} {'mean':>7}")
for g in (0.2, 0.5, 0.8):
    b = NRIRBrain(dim=8, out_dim=1, g=g, renorm=False, seed=0)
    p = np.empty_like(Y)
    for i in range(len(X)):
        p[i] = b.step(X[i], Y[i])
    a_, b_, a2 = score(p)
    print(f"{g:>8} {a_:>+7.3f} {b_:>+7.3f} {a2:>+7.3f} {(a_+b_+a2)/3:>+7.3f}")
