"""Capacity frontier: how complex a function can this frozen-feature + NLMS family fit?

Ladder: input dim 8/16/32/64, teacher = tanh(x W1) W2 (hidden 32, 8 outputs).
Three numbers per rung:
  online 1 pass  - the actual deployed behaviour
  online 4 passes- same stream repeated (sample budget effect)
  ridge ceiling  - closed-form least squares on the SAME features (offline),
                   i.e. the best this feature map could ever do
"""
import numpy as np
from brain_gf.nest_ops import OpBrain, signals
from brain_gf.gf_brain import FeatureReadoutBrain

def r2(y, p):
    return 1.0 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()

def teacher(d, seed=0):
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((d, 32)) / np.sqrt(d)
    W2 = rng.standard_normal((32, 8)) / np.sqrt(32)
    return lambda x: np.tanh(x @ W1) @ W2

def feats_of(brain, X):
    return np.stack([brain.forward(x)[1] for x in X])

def ridge(F, Y, lam=1e-2):
    A = F.T @ F + lam * np.eye(F.shape[1])
    return np.linalg.solve(A, F.T @ Y)

print(f"{'in_dim':>7} {'online1':>8} {'online4':>8} {'ridge(op)':>10} {'ridge(FR)':>10} {'dim(op)':>8}")
for d in (8, 16, 32, 64):
    f = teacher(d)
    N = 3000
    rng = np.random.default_rng(1)
    X = rng.standard_normal((N, d))
    Y = f(X)

    b = OpBrain(1, d, 8, 128, 16, 8, agg="concat", combine="concat",
                seed=0, static_mask=True)
    p1 = np.stack([b.step(X[i], Y[i]) for i in range(N)])
    r_one = r2(Y, p1)
    p4 = p1
    for _ in range(3):
        p4 = np.stack([b.step(X[i], Y[i]) for i in range(N)])
    r_four = r2(Y, p4)

    b2 = OpBrain(1, d, 8, 128, 16, 8, agg="concat", combine="concat",
                 seed=0, static_mask=True)
    F = feats_of(b2, X)
    Wo = ridge(F, Y)
    r_ridge = r2(Y, F @ Wo)

    fr = FeatureReadoutBrain(d, 8, seed=0)
    Ffr = np.stack([fr.features(x) for x in X])
    Wf = ridge(Ffr, Y)
    r_ridge_fr = r2(Y, Ffr @ Wf)

    print(f"{d:>7} {r_one:>+8.3f} {r_four:>+8.3f} {r_ridge:>+10.3f} {r_ridge_fr:>+10.3f} {F.shape[1]:>8}")
