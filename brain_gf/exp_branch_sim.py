"""Are the k leaves actually different? Measure it instead of assuming.

  - pairwise cosine between branch vectors, per layer
  - effective rank of the k branches (SVD, participation ratio)
  - Jaccard overlap of the top-k column sets chosen by different tracks
"""
import numpy as np
from brain_gf.nest_tree import TreeBrain, signals

N = 200
A_ = signals(N + 1, dim=8, seed=0, kind="sine")
X = A_[:-1]

b = TreeBrain(2, 8, 8, 64, 8, 4, 8, seed=0, agg="concat")

def cosmat(T):
    Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-8)
    S = Tn @ Tn.T
    iu = np.triu_indices(len(T), 1)
    return S[iu].mean(), S[iu].min(), S[iu].max()

def part_ratio(T):
    s = np.linalg.svd(T, compute_uv=False)
    s = s / (s.sum() + 1e-12)
    return 1.0 / (np.sum(s ** 2) + 1e-12)      # 1 = all energy in one direction

m0, m1 = [], []
r0, r1 = [], []
jac = []
for i in range(N):
    mix0, outs0, a0, idx0 = b.layers[0].forward(X[i], b.root_state)
    T0 = np.stack([o / (np.linalg.norm(o) + 1e-8) for o in outs0])
    m0.append(cosmat(T0)); r0.append(part_ratio(T0))

    T1, sets = [], []
    for t in range(len(T0)):
        st = b.states[t].setdefault(1, {})
        m, subs, a, idx = b.layers[1].forward(T0[t], st)
        T1.append(m / (np.linalg.norm(m) + 1e-8))
        sets.append(set(int(j) for j in idx))
    T1 = np.stack(T1)
    m1.append(cosmat(T1)); r1.append(part_ratio(T1))
    for a_i in range(len(sets)):
        for b_i in range(a_i + 1, len(sets)):
            inter = len(sets[a_i] & sets[b_i])
            union = len(sets[a_i] | sets[b_i])
            jac.append(inter / union)

print(f"layer 0 branches: cosine mean/min/max = {np.mean([x[0] for x in m0]):+.3f} / "
      f"{np.mean([x[1] for x in m0]):+.3f} / {np.mean([x[2] for x in m0]):+.3f}")
print(f"layer 1 branches: cosine mean/min/max = {np.mean([x[0] for x in m1]):+.3f} / "
      f"{np.mean([x[1] for x in m1]):+.3f} / {np.mean([x[2] for x in m1]):+.3f}")
print(f"effective rank  : layer0={np.mean(r0):.2f}  layer1={np.mean(r1):.2f}   (k=8)")
print(f"top-k column overlap between tracks (Jaccard, layer1) = {np.mean(jac):.3f}  "
      f"(0=完全不同的列, 1=完全相同的列)")
