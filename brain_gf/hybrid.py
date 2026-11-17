"""Two-timescale hybrid: wide frozen features (capacity) + small associative state (plasticity)."""
import numpy as np
from brain_gf.gf_brain import FeatureReadoutBrain
from brain_gf.nrir import NRIRBrain

def task_a(x): return (np.sin(x[:, 0]) + x[:, 1] * x[:, 2])[:, None]
def task_b(x): return (np.tanh(2.0 * x[:, 3] - x[:, 4]) - 0.7 * np.cos(1.7 * x[:, 0]))[:, None]
def r2(y, p): return 1 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()

class Hybrid:
    """pred = phi_fast @ W_fast + h_slow @ W_slow, two independent NLMS writes."""

    def __init__(self, dim=8, out_dim=1, lr_slow=0.05, lr_fast=0.5,
                 decay_slow=1.0, decay_fast=0.98, seed=0):
        self.fr = FeatureReadoutBrain(dim, out_dim, lr_decision=0.0, seed=seed)
        self.nr = NRIRBrain(dim=dim, out_dim=out_dim, renorm=False, seed=seed)
        # the FR organ must not write on its own; the hybrid owns the readout
        self.Ws = None
        self.slow = FeatureReadoutBrain(dim, out_dim, seed=seed)
        self.W_slow = np.zeros((self.slow.fdim, out_dim))
        self.W_fast = np.zeros((dim, out_dim))
        self.lr_slow, self.lr_fast = lr_slow, lr_fast
        self.decay_slow, self.decay_fast = decay_slow, decay_fast

    def step(self, x, y):
        phi = self.slow.features(x)
        h, _attns, _idxs, _states = self.fr_features(x)
        pred = phi @ self.W_slow + h @ self.W_fast
        err = y - pred
        e1 = float(phi @ phi) + 1e-8
        self.W_slow = self.W_slow * self.decay_slow + self.lr_slow * np.outer(phi, err) / e1
        e2 = float(h @ h) + 1e-8
        self.W_fast = self.W_fast * self.decay_fast + self.lr_fast * np.outer(h, err) / e2
        # let the associative memory learn from the residual as well
        self.nr.step(x, err.reshape(1))
        return pred

    def fr_features(self, x):
        h, attns, idxs, states = self.nr.forward(x)[1:]
        return h, attns, idxs, states

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

def run(mk):
    b = mk()
    p = np.empty_like(Y)
    for i in range(len(X)):
        out = b.step(X[i], Y[i])
        if isinstance(out, tuple):
            out = out[1]
        p[i] = np.asarray(out).reshape(-1)
    return score(p)

print(f"{'model':<28} {'A':>7} {'B':>7} {'A2':>7} {'mean':>7} {'worst':>7}")
class _NrirWrap:
    def __init__(self):
        self.b = NRIRBrain(dim=8, out_dim=1, g=0.5, renorm=False, seed=0)

    def step(self, x, y):
        return self.b.step(x, y)


for tag, mk in [
    ("fr decay=1.0", lambda: FeatureReadoutBrain(8, 1, decay=1.0, seed=0)),
    ("fr decay=0.9999", lambda: FeatureReadoutBrain(8, 1, decay=0.9999, seed=0)),
    ("nrir g=0.5", lambda: _NrirWrap()),
    ("hybrid slow+fast", lambda: Hybrid(seed=0)),
]:
    a, b_, a2 = run(mk)
    print(f"{tag:<28} {a:>+7.3f} {b_:>+7.3f} {a2:>+7.3f} {(a+b_+a2)/3:>+7.3f} {min(a,b_,a2):>+7.3f}")

