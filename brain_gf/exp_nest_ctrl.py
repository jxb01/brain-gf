"""Control: is 0.59 the brain's fault or the task's?"""
import numpy as np
from brain_gf.nest import NESTBrain, signals, r2, run_sequence
from brain_gf.gf_brain import FeatureReadoutBrain

N = 1500
A_ = signals(N + 1, dim=8, seed=0, kind="sine")
X, Y = A_[:-1], A_[1:]

print("single pass, same data (next-frame prediction, 8-dim sines)")
for lr in (0.05, 0.15, 0.3):
    b = FeatureReadoutBrain(8, 8, lr_decision=lr, decay=1.0, seed=0)
    p = np.empty_like(Y)
    for i in range(len(X)):
        _s, p[i] = b.step(X[i], Y[i])
    print(f"  feature_readout lr={lr:<5} R2={r2(Y, p):+.3f}")

b = NESTBrain(layers=1, input_dim=8, hidden_dim=8, n_columns=128, topk=8,
              n_neurons=4, out_dim=8, seed=0)
p, us = run_sequence(b, X, Y, readout=True, k_update=True, dopamine="vector")
print(f"  nest vector-dopamine             R2={r2(Y, p):+.3f}   ({us:.0f} us/step)")

print()
print("== experiment 3: A(sine) -> B(square) -> A, 1000 steps each ==")
seg = 1000
Xs = signals(seg, dim=8, seed=1, kind="sine")
Xb = signals(seg, dim=8, seed=2, kind="square")
Xall = np.concatenate([Xs, Xb, Xs])
Yall = np.concatenate([Xs, Xb, Xs])[1:] if False else np.concatenate([Xs, Xb, Xs])
Xin = np.concatenate([Xs, Xb, Xs])
# target = next sample of the same stream
Xin_full = np.concatenate([Xs, Xb, Xs])
shifted = np.concatenate([Xs[1:], Xb[:1], Xb[1:], Xs[:1], Xs[1:], Xs[:1]])[: 3 * seg]
Xin = Xin_full
Yt = shifted

def r2seg(preds):
    out, s = [], 0
    for name in ("A", "B", "A2"):
        sl = slice(s, s + seg)
        out.append(r2(Yt[sl], preds[sl]))
        s += seg
    return out

for tag, kw, mk in [
    ("nest scalar dop", dict(readout=True, k_update=True, dopamine="scalar"),
     lambda: NESTBrain(1, 8, 8, 128, 8, 4, 8, seed=0)),
    ("nest vector dop", dict(readout=True, k_update=True, dopamine="vector"),
     lambda: NESTBrain(1, 8, 8, 128, 8, 4, 8, seed=0)),
    ("nest frozen (k off)", dict(readout=True, k_update=False, dopamine="scalar"),
     lambda: NESTBrain(1, 8, 8, 128, 8, 4, 8, seed=0)),
]:
    b = mk()
    p = np.empty_like(Yt)
    for i in range(len(Xin)):
        p[i] = b.step(Xin[i], Yt[i], **kw)
    a, bb, a2 = r2seg(p)
    print(f"  {tag:<22} A:{a:+.3f}  B:{bb:+.3f}  A2:{a2:+.3f}")
