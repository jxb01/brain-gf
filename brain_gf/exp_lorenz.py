"""Lorenz-96: does the frozen-feature family survive a task that needs memory?

Models, all sharing the same ridge readout where applicable:
  A  op-bank features (static 16 cols x 4 ops = 512 dims), ridge          (offline ceiling)
  B  same features, NLMS single pass then frozen                          (deployed)
  C  ESN reservoir (300 units, frozen random recurrence), ridge           (memory baseline)
  D  lagged linear features (k lags), ridge                               (classical AR)
Tasks: one-step (x_{t+1}) and five-step (x_{t+5}) prediction, test split in time.
"""
import numpy as np

F = 8.0
N_DIM = 10

def rhs(x):
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F

def simulate(steps=60000, dt=0.01, every=5, seed=0):
    rng = np.random.default_rng(seed)
    x = F + 0.1 * rng.standard_normal(N_DIM)
    out = []
    for t in range(steps):
        k1 = rhs(x); k2 = rhs(x + dt / 2 * k1); k3 = rhs(x + dt / 2 * k2); k4 = rhs(x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if t % every == 0:
            out.append(x.copy())
    return np.array(out)

def r2(y, p):
    return 1.0 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()

def ridge_fit(Fm, Y, lam=1e-3):
    A = Fm.T @ Fm + lam * np.eye(Fm.shape[1])
    return np.linalg.solve(A, Fm.T @ Y)

# ---------------------------------------------------------------- data
Xraw = simulate()
Xall = Xraw[2000:]
mu, sd = Xall[: int(0.7 * len(Xall))].mean(0), Xall[: int(0.7 * len(Xall))].std(0)
Z = (Xall - mu) / sd
split = int(0.7 * len(Z))
H = {"one": 1, "five": 5}

print(f"samples={len(Z)}  train={split}  test={len(Z)-split}  dim={N_DIM}")

# ---------------------------------------------------------------- feature banks
from brain_gf.nest_ops import OpBrain

def op_features(Z, d):
    b = OpBrain(1, d, 8, 128, 16, 8, agg="concat", combine="concat", seed=0, static_mask=True)
    return np.stack([b.forward(z)[1] for z in Z])

class ESN:
    def __init__(self, n_in, n_res=300, spectral=0.9, seed=0):
        rng = np.random.default_rng(seed)
        self.Win = rng.uniform(-1, 1, (n_in, n_res)) * 1.0
        W = rng.standard_normal((n_res, n_res))
        W[rng.random((n_res, n_res)) > 0.1] = 0.0
        sr = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (spectral / max(sr, 1e-9))
        self.n_res = n_res

    def run(self, Z, warm=100):
        h = np.zeros(self.n_res)
        outs = []
        for t, z in enumerate(Z):
            h = np.tanh(z @ self.Win + self.W @ h)
            outs.append(np.concatenate([h, z]))     # state + current input
        return np.array(outs)

def lag_features(Z, k):
    pad = np.repeat(Z[:1], k, axis=0)
    Zp = np.vstack([pad, Z])
    return np.stack([Zp[k - j: k - j + len(Z)] for j in range(k + 1)], axis=1).reshape(len(Z), -1)

banks = {
    "A op-features (512)": op_features(Z, N_DIM),
    "C ESN (300+10)": ESN(N_DIM).run(Z),
    "D lag k=1": lag_features(Z, 1),
    "D lag k=5": lag_features(Z, 5),
    "D lag k=10": lag_features(Z, 10),
}

print(f"{'model':<24} {'1-step R2':>10} {'5-step R2':>10} {'dim':>6}")
rows = []
for name, Fm in banks.items():
    res = []
    for key, h in H.items():
        Yt = Z[h:]
        Ft = Fm[: len(Z) - h]
        W = ridge_fit(Ft[:split], Yt[:split])
        res.append(r2(Yt[split:], Ft[split:] @ W))
    print(f"{name:<24} {res[0]:>+10.3f} {res[1]:>+10.3f} {Fm.shape[1]:>6}")
    rows.append((name, res))

# ---------------------------------------------------------------- deployed: NLMS on the same features
b = OpBrain(1, N_DIM, 8, 128, 16, N_DIM, agg="concat", combine="concat", seed=0, static_mask=True)
Wnlms = np.zeros((b.W.shape[0], N_DIM))
for i in range(split - 1):                     # train part only, one pass
    pred, feats = b.forward(Z[i])
    e = Z[i + 1] - pred
    Wnlms = Wnlms + 0.15 * np.outer(feats, e) / (float(feats @ feats) + 1e-8)
# evaluate frozen readout on the test part
Fte = np.stack([b.forward(z)[1] for z in Z[split:]])
pred1 = Fte @ Wnlms
print(f"{'B NLMS (frozen, 1 pass)':<24} {r2(Z[split+1:], pred1[:-1]):>+10.3f} {'-':>10} {Fte.shape[1]:>6}")
