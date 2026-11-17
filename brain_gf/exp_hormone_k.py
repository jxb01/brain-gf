"""Does a hormone-modulated k help?

k_t = clip(k_min + round(c * rms(e)), k_min, k_max)
  - a fixed CORE of k_min columns is always on (keeps the readout's sample budget)
  - the extra columns are chosen dynamically and only while the error is large
That is the neuromodulation reading: arousal widens the ensemble, calm narrows it.
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

def run(k_fixed=None, k_min=8, k_max=32, c=0.6, lr=0.15):
    b = OpBrain(1, 8, 8, 128, k_max, 8, agg="concat", combine="concat",
                seed=0, static_mask=True)
    layer = b.layers[0]
    W = np.zeros((k_max * layer.col_out_dim, 8))
    core = np.arange(k_min)
    var = 1.0
    preds = np.empty_like(Y)
    ks = []
    for i in range(len(X)):
        x = X[i]
        if k_fixed is None:                      # hormone-modulated k
            theta = c * np.sqrt(var)
            k_t = int(np.clip(k_min + round(theta * k_max), k_min, k_max))
        else:
            k_t = k_fixed
        ks.append(k_t)

        scores = x @ layer.K
        rest = np.setdiff1d(np.arange(layer.K.shape[1]), core, assume_unique=False)
        n_extra = max(0, k_t - k_min)
        extra = rest[np.argsort(-scores[rest])[:n_extra]] if n_extra else np.array([], int)
        idx = np.concatenate([core, extra]).astype(int)

        outs = [layer.cols[j].forward(x) for j in idx]
        feats = np.zeros(k_max * layer.col_out_dim)
        for t, j in enumerate(idx):
            feats[t * layer.col_out_dim:(t + 1) * layer.col_out_dim] = outs[t]
        pred = feats @ W
        preds[i] = pred
        e = Y[i] - pred
        W = W + lr * np.outer(feats, e) / (float(feats @ feats) + 1e-8)
        var = 0.98 * var + 0.02 * float(e @ e) / e.size
    seg = [r2(Y[i*SEG:(i+1)*SEG], preds[i*SEG:(i+1)*SEG]) for i in range(3)]
    bw = [r2(Y[SEG+a:SEG+b_], preds[SEG+a:SEG+b_]) for a, b_ in ((0,100),(800,1000))]
    cost = 100.0 * np.mean(ks) / k_max
    return seg, bw, np.mean(ks), cost

print(f"{'config':<26} {'A':>7} {'B':>7} {'A2':>7} {'B[0:100]':>9} {'B[800:]':>8} {'mean k':>7} {'cost%':>6}")
for tag, kw in [
    ("fixed k=8", dict(k_fixed=8)),
    ("fixed k=16", dict(k_fixed=16)),
    ("fixed k=32", dict(k_fixed=32)),
    ("hormone k 8..32 c=0.6", dict(k_fixed=None, c=0.6)),
    ("hormone k 8..32 c=1.0", dict(k_fixed=None, c=1.0)),
]:
    seg, bw, mk, cost = run(**kw)
    print(f"{tag:<26} {seg[0]:>+7.3f} {seg[1]:>+7.3f} {seg[2]:>+7.3f} "
          f"{bw[0]:>+9.3f} {bw[1]:>+8.3f} {mk:>7.1f} {cost:>5.0f}%")
