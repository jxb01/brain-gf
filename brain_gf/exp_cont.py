"""Continual learning test: A (sine) -> B (square) -> A2 (sine), 1000 steps each.

The whole point of the design is adaptation, so this is the decisive benchmark:
per-segment R2, recovery steps after each switch (rolling MSE back under 1.5x the
pre-switch level), and the worst segment.
"""
import time
import numpy as np
from brain_gf.nest_ops import OpBrain, signals, r2
from brain_gf.gf_brain import FeatureReadoutBrain

SEG = 1000

def build_stream(kind_a="sine", kind_b="square", seed=0):
    Xs, Ys = [], []
    for kind in (kind_a, kind_b, kind_a):
        s = signals(SEG + 1, dim=8, seed=seed, kind=kind)
        Xs.append(s[:-1])
        Ys.append(s[1:])
    return np.concatenate(Xs), np.concatenate(Ys)

X, Y = build_stream()

def roll(y, p, w=50):
    err = ((y - p) ** 2).mean(axis=1)
    c = np.cumsum(np.insert(err, 0, 0.0))
    out = np.empty_like(err)
    for t in range(len(err)):
        lo = max(0, t - w + 1)
        out[t] = (c[t + 1] - c[lo]) / (t + 1 - lo)
    return out

def evaluate(p):
    segs, s = [], 0
    for name in ("A", "B", "A2"):
        sl = slice(s, s + SEG)
        segs.append(r2(Y[sl], p[sl]))
        s += SEG
    rm = roll(Y, p)
    rec = []
    s = 0
    for i in range(3):
        if i > 0:
            base = float(np.median(rm[s - 50:s]))
            hit = None
            for t in range(s + 1, min(s + SEG, len(rm))):
                if rm[t] <= 1.5 * base:
                    hit = t - s
                    break
            rec.append(hit)
        s += SEG
    return segs, rec

def run_op(decay, static=True, agg="concat", k=16):
    b = OpBrain(1, 8, 8, 128, k, 8, agg=agg, combine="concat", seed=0, static_mask=static)
    p = np.empty_like(Y)
    t0 = time.perf_counter()
    for i in range(len(X)):
        p[i] = b.step(X[i], Y[i], decay=decay)
    return p, 1e6 * (time.perf_counter() - t0) / len(X)

def run_fr(decay):
    b = FeatureReadoutBrain(8, 8, lr_decision=0.15, decay=decay, seed=0)
    p = np.empty_like(Y)
    t0 = time.perf_counter()
    for i in range(len(X)):
        _s, p[i] = b.step(X[i], Y[i])
    return p, 1e6 * (time.perf_counter() - t0) / len(X)

rows = []
for tag, fn in [
    ("op static16  decay=1.0", lambda: run_op(1.0)),
    ("op static16  decay=0.9999", lambda: run_op(0.9999)),
    ("op static16  decay=0.999", lambda: run_op(0.999)),
    ("op dynamic16 decay=1.0", lambda: run_op(1.0, static=False)),
    ("op static16 cluster readout", lambda: run_op(1.0, agg="cluster")),
    ("FR           decay=1.0", lambda: run_fr(1.0)),
    ("FR           decay=0.9999", lambda: run_fr(0.9999)),
]:
    p, us = fn()
    segs, rec = evaluate(p)
    rows.append((tag, segs, rec, us))

print(f"{'config':<30} {'A':>7} {'B':>7} {'A2':>7} {'worst':>7} {'rec@B':>6} {'rec@A2':>7} {'us':>6}")
for tag, segs, rec, us in rows:
    print(f"{tag:<30} {segs[0]:>+7.3f} {segs[1]:>+7.3f} {segs[2]:>+7.3f} "
          f"{min(segs):>+7.3f} {str(rec[0]):>6} {str(rec[1]):>7} {us:>6.0f}")
