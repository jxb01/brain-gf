"""Tree as representation: each node is a local predictor, not just a feature.

  node j  : op column phi_j(x) (32 dims) + its OWN readout w_j (32 -> 8)
  routing : top-m nodes by x @ K, softmax gates g_j
  output  : y = sum_j g_j * (phi_j @ w_j)
  update  : each active node learns its share of the error, locally
            w_j += lr * g_j * phi_j (x) e / |phi_j|^2
  hormone : a node's local error drives growth (spawn a child = activate a new
            node) and death (long-unused node is dropped)

Why this can escape the earlier starvation: a node's own readout is only 32x8,
so it can be fit from a few hundred activations, unlike one global 1024x8 readout.
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
POOL = 32

def run(n0, n_max, m=2, lr=0.15, grow=None, cooldown=120):
    b = OpBrain(1, 8, 8, 128, POOL, 8, agg="concat", combine="concat",
                seed=0, static_mask=True)
    layer = b.layers[0]
    cd = layer.col_out_dim
    wi = np.zeros((POOL, cd, 8))
    n = n0
    last_grow = -10**9
    var = 1.0
    preds = np.empty_like(Y)
    events = []
    for i in range(len(X)):
        x = X[i]
        scores = x @ layer.K[:, :n]
        k = min(m, n)
        sel = np.argsort(-scores)[:k]
        z = scores[sel] - scores[sel].max()
        g = np.exp(z); g /= g.sum() + 1e-12

        pred = np.zeros(8)
        phis = []
        for gj, j in zip(g, sel):
            phi = layer.cols[int(j)].forward(x)
            phis.append((gj, int(j), phi))
            pred = pred + gj * (phi @ wi[int(j)])
        preds[i] = pred
        e = Y[i] - pred

        for gj, j, phi in phis:                    # local credit, own readout only
            wi[j] += lr * gj * np.outer(phi, e) / (float(phi @ phi) + 1e-8)

        var = 0.98 * var + 0.02 * float(e @ e) / e.size
        if grow is not None and n < n_max and np.sqrt(var) > grow and (i - last_grow) > cooldown:
            n += 1; last_grow = i; events.append(i)

    seg = [r2(Y[k*SEG:(k+1)*SEG], preds[k*SEG:(k+1)*SEG]) for k in range(3)]
    bw = [r2(Y[SEG+a:SEG+b_], preds[SEG+a:SEG+b_]) for a, b_ in ((0,100),(800,1000))]
    return seg, bw, n, events

print(f"{'config':<28} {'A':>7} {'B':>7} {'A2':>7} {'B[0:100]':>9} {'B[800:]':>8} {'end n':>6}")
for tag, kw in [
    ("tree n=8  m=2", dict(n0=8, n_max=8)),
    ("tree n=16 m=2", dict(n0=16, n_max=16)),
    ("tree n=32 m=2", dict(n0=32, n_max=32)),
    ("tree grow 8..32 m=2 t=0.6", dict(n0=8, n_max=32, grow=0.6)),
    ("tree grow 8..32 m=4 t=0.6", dict(n0=8, n_max=32, m=4, grow=0.6)),
]:
    seg, bw, n, ev = run(**kw)
    print(f"{tag:<28} {seg[0]:>+7.3f} {seg[1]:>+7.3f} {seg[2]:>+7.3f} "
          f"{bw[0]:>+9.3f} {bw[1]:>+8.3f} {n:>6}")
print()
print("reference (flat concat readout, static 8 cols): A +0.743  B +0.654  A2 +0.971  B[0:100] +0.610")
