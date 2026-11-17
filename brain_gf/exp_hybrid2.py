"""Hybrid: static features (fast transfer) + dynamic features (higher ceiling)."""
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

class Hybrid:
    def __init__(self, k=16, lr=0.15, decay=1.0):
        self.a = OpBrain(1, 8, 8, 128, k, 8, agg="concat", combine="concat",
                         seed=0, static_mask=True)
        self.b = OpBrain(1, 8, 8, 128, k, 8, agg="concat", combine="concat",
                         seed=0, static_mask=False)
        self.W = np.zeros((self.a.W.shape[0] + self.b.W.shape[0], 8))
        self.lr, self.decay = lr, decay

    def step(self, x, y):
        _, f1 = self.a.forward(x)
        _, f2 = self.b.forward(x)
        f = np.concatenate([f1, f2])
        pred = f @ self.W
        e = y - pred
        self.W = self.W * self.decay + self.lr * np.outer(f, e) / (float(f @ f) + 1e-8)
        return pred

h = Hybrid()
p = np.empty_like(Y)
for i in range(len(X)):
    p[i] = h.step(X[i], Y[i])

def win(p, name, spans):
    s = {"A": 0, "B": SEG, "A2": 2 * SEG}[name]
    return [r2(Y[s + a:s + b_], p[s + a:s + b_]) for a, b_ in spans]

spans = [(0, 100), (100, 300), (300, 600), (800, 1000)]
print("hybrid (static16 + dynamic16, shared readout)")
print(f"  B windows: " + "  ".join(f"{v:+.3f}" for v in win(p, "B", spans)))
print("  reference:")
print("    static16  B: +0.573  +0.597  +0.626  +0.658")
print("    dynamic16 B: +0.271  +0.487  +0.632  +0.719")
seg = [r2(Y[i*SEG:(i+1)*SEG], p[i*SEG:(i+1)*SEG]) for i in range(3)]
print(f"  segments A/B/A2 = {seg[0]:+.3f} / {seg[1]:+.3f} / {seg[2]:+.3f}")
