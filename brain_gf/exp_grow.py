"""Structural plasticity: let the tree grow a column when the error stays high.

Rules taken from everything measured so far:
  - slots are stable (slot == column index), inactive slots are zero
  - growth only APPENDS; the new column's readout block starts at 0, so the
    prediction never jumps
  - growth needs sustained high error + a cooldown (no per-step flicker)
  - no pruning (credit-based pruning measured inverted)
"""
import numpy as np
from brain_gf.nest_ops import OpBrain, signals, r2

SEG = 1000
def build():
    Xs, Ys = [], []
    for kind in ("sine", "square", "sine"):
        s = signals(SEG + 1, dim=8, seed=0, kind=kind)
        Xs.append(s[:-1]); Ys.append(s[1:])
    return np.concatenate(Xs), np.concatenate(Ys)

X, Y = build()
POOL_MAX = 32
CORE = 8

def run(active0, active_max, grow_thresh=None, cooldown=150, lr=0.15):
    b = OpBrain(1, 8, 8, 128, POOL_MAX, 8, agg="concat", combine="concat",
                seed=0, static_mask=True)
    layer = b.layers[0]
    cd = layer.col_out_dim
    W = np.zeros((POOL_MAX * cd, 8))
    active = active0
    last_grow = -10**9
    var = 1.0; base = None
    preds = np.empty_like(Y)
    trace = []
    for i in range(len(X)):
        x = X[i]
        feats = np.zeros(POOL_MAX * cd)
        for j in range(active):
            feats[j * cd:(j + 1) * cd] = layer.cols[j].forward(x)
        pred = feats @ W
        preds[i] = pred
        e = Y[i] - pred
        W = W + lr * np.outer(feats, e) / (float(feats @ feats) + 1e-8)
        var = 0.98 * var + 0.02 * float(e @ e) / e.size

        if grow_thresh is not None and active < active_max:
            rms = np.sqrt(var)
            if base is None:
                base = rms
            # sustained: error must be clearly above the recent baseline AND cooldown passed
            if rms > grow_thresh and (i - last_grow) > cooldown:
                active += 1
                last_grow = i
                trace.append(i)
                base = rms
    seg = [r2(Y[k*SEG:(k+1)*SEG], preds[k*SEG:(k+1)*SEG]) for k in range(3)]
    bw = [r2(Y[SEG+a:SEG+b_], preds[SEG+a:SEG+b_]) for a, b_ in ((0,100),(800,1000))]
    return seg, bw, active, trace

print(f"{'config':<30} {'A':>7} {'B':>7} {'A2':>7} {'B[0:100]':>9} {'B[800:]':>8} {'end k':>6}")
for tag, kw in [
    ("fixed 8", dict(active0=8, active_max=8)),
    ("fixed 16", dict(active0=16, active_max=16)),
    ("fixed 32", dict(active0=32, active_max=32)),
    ("grow 8->32 thresh=0.6", dict(active0=8, active_max=32, grow_thresh=0.6)),
    ("grow 8->32 thresh=0.4", dict(active0=8, active_max=32, grow_thresh=0.4)),
]:
    seg, bw, act, tr = run(**kw)
    print(f"{tag:<30} {seg[0]:>+7.3f} {seg[1]:>+7.3f} {seg[2]:>+7.3f} "
          f"{bw[0]:>+9.3f} {bw[1]:>+8.3f} {act:>6}")
print()
seg, bw, act, tr = run(active0=8, active_max=32, grow_thresh=0.6)
print(f"growth events (thresh=0.6): {tr}")
