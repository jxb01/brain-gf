"""Hormone-modulated k, slot-stable (slot == column index; inactive slots are zero).

The first attempt let the selected columns shift between feature slots every step,
so the readout applied its weights to the wrong column - that polluted every
variable-k number, including the fixed-k=16/32 ones.
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
POOL = 32                      # candidate columns; slot i <-> column i
CORE = 8

def run(k_spec, c=0.6, lr=0.15):
    b = OpBrain(1, 8, 8, 128, POOL, 8, agg="concat", combine="concat",
                seed=0, static_mask=True)
    layer = b.layers[0]
    cd = layer.col_out_dim
    W = np.zeros((POOL * cd, 8))
    var = 1.0
    preds = np.empty_like(Y)
    ks = []
    for i in range(len(X)):
        x = X[i]
        if k_spec == "hormone":
            k_t = int(np.clip(CORE + round(c * np.sqrt(var) * POOL), CORE, POOL))
            scores = x @ layer.K
            extras = np.argsort(-scores[CORE:POOL])[: k_t - CORE] + CORE
            idx = np.concatenate([np.arange(CORE), extras]).astype(int)
        else:
            idx = np.arange(int(k_spec))
        ks.append(len(idx))

        feats = np.zeros(POOL * cd)
        for j in idx:
            feats[j * cd:(j + 1) * cd] = layer.cols[int(j)].forward(x)
        pred = feats @ W
        preds[i] = pred
        e = Y[i] - pred
        W = W + lr * np.outer(feats, e) / (float(feats @ feats) + 1e-8)
        var = 0.98 * var + 0.02 * float(e @ e) / e.size
    seg = [r2(Y[i*SEG:(i+1)*SEG], preds[i*SEG:(i+1)*SEG]) for i in range(3)]
    bw = [r2(Y[SEG+a:SEG+b_], preds[SEG+a:SEG+b_]) for a, b_ in ((0,100),(800,1000))]
    return seg, bw, np.mean(ks)

print("slot-stable, pool=32 columns, 32 dims each -> 1024 feature slots")
print(f"{'config':<24} {'A':>7} {'B':>7} {'A2':>7} {'B[0:100]':>9} {'B[800:]':>8} {'mean k':>7}")
for tag, spec in [("fixed k=8", 8), ("fixed k=16", 16), ("fixed k=32", 32)]:
    seg, bw, mk = run(spec)
    print(f"{tag:<24} {seg[0]:>+7.3f} {seg[1]:>+7.3f} {seg[2]:>+7.3f} {bw[0]:>+9.3f} {bw[1]:>+8.3f} {mk:>7.1f}")
for c in (0.4, 0.8, 1.5):
    seg, bw, mk = run("hormone", c=c)
    print(f"{'hormone 8..32 c=' + str(c):<24} {seg[0]:>+7.3f} {seg[1]:>+7.3f} {seg[2]:>+7.3f} "
          f"{bw[0]:>+9.3f} {bw[1]:>+8.3f} {mk:>7.1f}")
