"""The real memory test: observe only 3 of the 10 Lorenz-96 coordinates.

With the full state the system is first-order Markov, so a memoryless feature map
is enough (that is why A beat the ESN on 5-step). Hide 7 coordinates and the past
becomes necessary - now a reservoir should win if memory matters.
"""
import numpy as np
from brain_gf.nest_ops import OpBrain

F = 8.0
N_DIM = 10
KEEP = 3

def rhs(x):
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F

def simulate(steps=60000, dt=0.01, every=5, seed=0):
    rng = np.random.default_rng(seed)
    x = F + 0.1 * rng.standard_normal(N_DIM)
    out = []
    for t in range(steps):
        k1 = rhs(x); k2 = rhs(x + dt/2*k1); k3 = rhs(x + dt/2*k2); k4 = rhs(x + dt*k3)
        x = x + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        if t % every == 0:
            out.append(x.copy())
    return np.array(out)

def r2(y, p):
    return 1.0 - ((y - p)**2).sum() / ((y - y.mean(0))**2).sum()

def ridge_fit(Fm, Y, lam=1e-3):
    return np.linalg.solve(Fm.T @ Fm + lam*np.eye(Fm.shape[1]), Fm.T @ Y)

class ESN:
    def __init__(self, n_in, n_res=300, spectral=0.9, seed=0):
        rng = np.random.default_rng(seed)
        self.Win = rng.uniform(-1, 1, (n_in, n_res))
        W = rng.standard_normal((n_res, n_res))
        W[rng.random((n_res, n_res)) > 0.1] = 0.0
        self.W = W * (spectral / max(np.max(np.abs(np.linalg.eigvals(W))), 1e-9))
        self.n_res = n_res
    def run(self, Z):
        h = np.zeros(self.n_res); outs = []
        for z in Z:
            h = np.tanh(z @ self.Win + self.W @ h)
            outs.append(np.concatenate([h, z]))
        return np.array(outs)

Xraw = simulate()[2000:]
Zfull = Xraw
Z = Zfull[:, :KEEP]
split = int(0.7 * len(Z))

def lag(Z, k):
    pad = np.repeat(Z[:1], k, 0)
    Zp = np.vstack([pad, Z])
    return np.stack([Zp[k-j:k-j+len(Z)] for j in range(k+1)], 1).reshape(len(Z), -1)

def opf(Z, d):
    b = OpBrain(1, d, 8, 128, 16, KEEP, agg="concat", combine="concat", seed=0, static_mask=True)
    return np.stack([b.forward(z)[1] for z in Z]), b

print(f"observed dims={KEEP} of {N_DIM}, samples={len(Z)}, train={split}, test={len(Z)-split}")
print(f"{'model':<26} {'1-step':>8} {'5-step':>8} {'dim':>6}")
Fa, _ = opf(Z, KEEP)
Fesn = ESN(KEEP).run(Z)
banks = {"A op-features (no memory)": Fa, "C ESN (memory)": Fesn,
         "D lag k=5": lag(Z, 5), "D lag k=20": lag(Z, 20)}
for name, Fm in banks.items():
    out = []
    for h in (1, 5):
        Yt = Z[h:]; Ft = Fm[:len(Z)-h]
        W = ridge_fit(Ft[:split], Yt[:split])
        out.append(r2(Yt[split:], Ft[split:] @ W))
    print(f"{name:<26} {out[0]:>+8.3f} {out[1]:>+8.3f} {Fm.shape[1]:>6}")
