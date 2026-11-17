import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brain_gf.legacy.kv_layered import LayeredKVBrain
from brain_gf.gf_brain import FeatureReadoutBrain
from brain_gf.numerics import r2, rolling_mse
from brain_gf.tasks import make_stream, DEFAULT_SEGMENTS


def run(stepper, X, Y):
    P = np.zeros_like(Y)
    t0 = time.perf_counter()
    for t in range(len(X)):
        P[t] = stepper(X[t], Y[t])
    return P, 1e6 * (time.perf_counter() - t0) / len(X)


print("=== stationary regime A: does it learn at all? ===")
Xs, Ys, _ = make_stream(segments=(("A", 2000),), dim=8, seed=0)
variants = [
    ("K+V updated", lambda: LayeredKVBrain(8, 16, 16, update_k=True, seed=0)),
    ("V only", lambda: LayeredKVBrain(8, 16, 16, update_k=False, seed=0)),
    ("K+V decay", lambda: LayeredKVBrain(8, 16, 16, update_k=True, decay=0.999, seed=0)),
    ("V only decay", lambda: LayeredKVBrain(8, 16, 16, update_k=False, decay=0.999, seed=0)),
]
for name, mk in variants:
    b = mk()
    P, us = run(b.step, Xs, Ys)
    blocks = [r2(Ys[i:i + 500], P[i:i + 500]) for i in (0, 500, 1000, 1500)]
    txt = " ".join("%+.2f" % v for v in blocks)
    print("%-14s R2 by block: %s   tail=%+.3f  %6.1f us/step"
          % (name, txt, r2(Ys[-500:], P[-500:]), us))

print("")
print("=== non-stationary A -> B -> A ===")
X, Y, _ = make_stream(segments=DEFAULT_SEGMENTS, dim=8, seed=0)
segs = [(0, 900), (900, 1600), (1600, 2300)]


def report(name, P, us, params):
    parts = ["%+.3f" % r2(Y[a:b], P[a:b]) for a, b in segs]
    rm = rolling_mse(Y, P, 50)
    rec = []
    for s in (900, 1600):
        base = float(np.median(rm[s - 50:s]))
        thr = 1.5 * base
        hit = None
        for t in range(s + 1, min(s + 400, len(rm))):
            if rm[t] <= thr:
                hit = t - s
                break
        rec.append(hit)
    rb = "-" if rec[0] is None else str(rec[0])
    rc = "-" if rec[1] is None else str(rec[1])
    print("%-16s R2 A1/B/A2: %s   rec@B=%3s rec@A2=%3s  %6.1f us/step  params=%d"
          % (name, " ".join(parts), rb, rc, us, params))


for name, b in [
    ("kv K+V", LayeredKVBrain(8, 16, 16, update_k=True, seed=0)),
    ("kv V-only", LayeredKVBrain(8, 16, 16, update_k=False, seed=0)),
]:
    P, us = run(b.step, X, Y)
    report(name, P, us, b.counts()["trainable"])

fb = FeatureReadoutBrain(8, 8, hid=256, lr_decision=0.1, decay=1.0, seed=0)
P, us = run(lambda x, y: fb.step(x, y)[1], X, Y)
report("gf_full(256)", P, us, fb.counts()["trainable"])
