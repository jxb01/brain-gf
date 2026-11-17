"""Why consensus cannot help here: measure how correlated the leaves' errors are.

Majority vote reduces error only when the voters' errors are independent.
Ensemble variance = rho*sigma^2 + (1-rho)*sigma^2/M.  If rho ~ 1 there is nothing to average.
"""
import numpy as np
from brain_gf.nest_tree import TreeBrain, signals

N = 600
A_ = signals(N + 1, dim=8, seed=0, kind="sine")
X, Y = A_[:-1], A_[1:]

b = TreeBrain(1, 8, 8, 128, 8, 4, 8, seed=0, agg="sum")
resid = []
for i in range(N):
    pred, feats = b.forward(X[i])
    b.step(X[i], Y[i])
    # per-leaf output: each branch goes through the same hidden->out readout
    leaves = []
    for t in range(8):
        leaf = feats if t == 0 else None
    resid.append(None)

# rebuild properly: capture per-track vectors
b2 = TreeBrain(1, 8, 8, 128, 8, 4, 8, seed=0, agg="sum")
errs = []
for i in range(N):
    pred, feats = b2.forward(X[i])
    # reconstruct the 8 normalized track vectors
    mix0, outs, a0, idx0 = b2.layers[0].forward(X[i], b2.root_state)
    tracks = np.stack([o / (np.linalg.norm(o) + 1e-8) for o in outs])
    leaf_pred = tracks @ b2.W            # (8, out_dim)
    errs.append((leaf_pred - Y[i]).ravel())
    b2.W = b2.W + 0.15 * np.outer(feats, Y[i] - pred) / (feats @ feats + 1e-8)

E = np.array(errs)                      # (N, 8*out)
M = 8
cors = []
for i in range(M):
    for j in range(i + 1, M):
        a = E[:, i * 8:(i + 1) * 8].ravel()
        c = E[:, j * 8:(j + 1) * 8].ravel()
        cors.append(np.corrcoef(a, c)[0, 1])
cors = np.array(cors)
print(f"pairwise leaf-error correlation: mean={cors.mean():+.3f}  min={cors.min():+.3f}  max={cors.max():+.3f}")
lam = cors.mean()
print(f"majority-vote variance factor = rho + (1-rho)/M = {lam + (1-lam)/M:.3f}   (1.0 = no gain)")
print()
print("interpreting: rho near 1 means all leaves fail together (same bank, same input),")
print("so re-weighting them by agreement cannot reduce error - only concat (more features) can.")
