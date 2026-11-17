"""Why does every "each branch fits its own leftover" variant diverge?

Backfitting is block Gauss-Seidel on the normal equations. It converges iff the
spectral radius of the iteration matrix is < 1. For block-wise least squares with
blocks F_1..F_n and A = F^T F, D = blockdiag(A):

    M = I - D^{-1} A          rho(M) < 1  <=>  converges

Correlated blocks (all branches see the same x) push rho(M) above 1, and then no
step size < 1 saves it.
"""
import numpy as np
from brain_gf.nest_ops import OpColumn, signals

def build_stream():
    Xs, Ys = [], []
    for kind in ("sine", "square", "sine"):
        s = signals(1000 + 1, dim=8, seed=0, kind=kind)
        Xs.append(s[:-1]); Ys.append(s[1:])
    return np.concatenate(Xs), np.concatenate(Ys)

X, Y = build_stream()

rng = np.random.default_rng(0)
cols = [OpColumn(8, 8, rng, n_ops=4, combine="concat") for _ in range(8)]
N = 1500
blocks = []
for j in range(8):
    F = np.stack([cols[j].forward(x) for x in X[:N]])
    blocks.append(F)

# A = F^T F, block-wise
sizes = [b.shape[1] for b in blocks]
A = np.zeros((sum(sizes), sum(sizes)))
for i, bi in enumerate(blocks):
    for k, bk in enumerate(blocks):
        A[i*32:(i+1)*32, k*32:(k+1)*32] = bi.T @ bk
D = np.zeros_like(A)
for i in range(8):
    D[i*32:(i+1)*32, i*32:(i+1)*32] = A[i*32:(i+1)*32, i*32:(i+1)*32]

M = np.eye(A.shape[0]) - np.linalg.solve(D, A)
rho = np.max(np.abs(np.linalg.eigvals(M)))
print(f"blocks=8, features=32 each, N={N}")
print(f"spectral radius of the backfitting iteration : {rho:.3f}   (must be < 1)")

# how correlated are the blocks themselves?
Fsum = sum(blocks)
corr = []
for i in range(8):
    for k in range(i+1, 8):
        ci = blocks[i] @ np.linalg.lstsq(Fsum, Y[:N], rcond=None)[0]
        ck = blocks[k] @ np.linalg.lstsq(Fsum, Y[:N], rcond=None)[0]
        corr.append(np.corrcoef(ci.ravel(), ck.ravel())[0, 1])
print(f"pairwise correlation of already-fit block contributions: mean={np.mean(corr):+.3f}  "
      f"min={np.min(corr):+.3f}  max={np.max(corr):+.3f}")
print()
print("reading: rho > 1 means block Gauss-Seidel diverges for ANY step size -")
print("that is why parallel / damped / sequential / floored all blew up.")
