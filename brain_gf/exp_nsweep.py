"""How much data does the online learner need to reach the feature map's ceiling?

Fixed d=32, fixed 512 features, vary N. online(1 pass) vs ridge on the same features.
"""
import numpy as np
from brain_gf.nest_ops import OpBrain

def r2(y, p):
    return 1.0 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()

d = 32
rng0 = np.random.default_rng(0)
W1 = rng0.standard_normal((d, 32)) / np.sqrt(d)
W2 = rng0.standard_normal((32, 8)) / np.sqrt(32)
f = lambda x: np.tanh(x @ W1) @ W2

print(f"{'N':>7} {'online1':>9} {'online3':>9} {'ridge':>8} {'N/features':>11}")
for N in (500, 1500, 3000, 8000):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((N, d))
    Y = f(X)

    b = OpBrain(1, d, 8, 128, 16, 8, agg="concat", combine="concat",
                seed=0, static_mask=True)
    p1 = np.stack([b.step(X[i], Y[i]) for i in range(N)])
    r1 = r2(Y, p1)
    p3 = p1
    for _ in range(2):
        p3 = np.stack([b.step(X[i], Y[i]) for i in range(N)])
    r3 = r2(Y, p3)

    b2 = OpBrain(1, d, 8, 128, 16, 8, agg="concat", combine="concat",
                 seed=0, static_mask=True)
    F = np.stack([b2.forward(x)[1] for x in X])
    A = F.T @ F + 1e-2 * np.eye(F.shape[1])
    W = np.linalg.solve(A, F.T @ Y)
    rr = r2(Y, F @ W)

    print(f"{N:>7} {r1:>+9.3f} {r3:>+9.3f} {rr:>+8.3f} {N/512:>11.1f}")
