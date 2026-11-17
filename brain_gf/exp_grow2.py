"""Fair test for growth: a task where the starting capacity is genuinely too small.

d=64 random teacher, 3000 samples, single pass. Earlier: 512 features -> 0.368 online.
Start at 8 columns (256 features) and let it grow to 32 (1024).
"""
import numpy as np
from brain_gf.nest_ops import OpBrain

def r2(y, p): return 1.0 - ((y - p)**2).sum() / ((y - y.mean(0))**2).sum()

d = 64
rng0 = np.random.default_rng(0)
W1 = rng0.standard_normal((d, 32)) / np.sqrt(d)
W2 = rng0.standard_normal((32, 8)) / np.sqrt(32)
f = lambda x: np.tanh(x @ W1) @ W2

N = 3000
rng = np.random.default_rng(1)
X = rng.standard_normal((N, d))
Y = f(X)
POOL_MAX = 32

def run(active0, active_max, grow_thresh=None, cooldown=120, lr=0.15, seed=0):
    b = OpBrain(1, d, 8, 128, POOL_MAX, 8, agg="concat", combine="concat",
                seed=seed, static_mask=True)
    layer = b.layers[0]; cd = layer.col_out_dim
    W = np.zeros((POOL_MAX * cd, 8))
    active = active0; last_grow = -10**9; var = 1.0
    preds = np.empty_like(Y); trace = []
    for i in range(N):
        x = X[i]
        feats = np.zeros(POOL_MAX * cd)
        for j in range(active):
            feats[j*cd:(j+1)*cd] = layer.cols[j].forward(x)
        pred = feats @ W
        preds[i] = pred
        e = Y[i] - pred
        W = W + lr * np.outer(feats, e) / (float(feats @ feats) + 1e-8)
        var = 0.98*var + 0.02*float(e @ e)/e.size
        if grow_thresh is not None and active < active_max:
            if np.sqrt(var) > grow_thresh and (i - last_grow) > cooldown:
                active += 1; last_grow = i; trace.append(i)
    return r2(Y, preds), r2(Y[-1000:], preds[-1000:]), active, trace

print(f"{'config':<26} {'last1k mean':>12} {'spread':>8} {'end k':>6}")
for tag, kw in [
    ("fixed 8 (256 feats)", dict(active0=8, active_max=8)),
    ("fixed 32 (1024 feats)", dict(active0=32, active_max=32)),
    ("grow 8->32 t=0.6", dict(active0=8, active_max=32, grow_thresh=0.6)),
]:
    vals, ends = [], []
    for sd in (0, 1, 2):
        _a, r_tail, act, _tr = run(seed=sd, **kw)
        vals.append(r_tail); ends.append(act)
    vals = np.array(vals)
    print(f"{tag:<26} {vals.mean():>+12.3f} {vals.std():>8.3f} {np.mean(ends):>6.1f}")
