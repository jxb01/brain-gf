"""Experiment 4-lite: does the survival score select the right columns?

Mixed input (fast / slow / spike), 4000 steps, then ask:
  - which columns actually carried the signal (activation count)
  - which ones the survival rule decided to keep
"""
import numpy as np
from brain_gf.nest import NESTBrain, r2

rng = np.random.default_rng(0)
n = 4000
t = np.arange(n)
fast = np.sin(2 * np.pi * 0.18 * t)
slow = 0.5 * np.sin(2 * np.pi * 0.01 * t)
spike = np.zeros(n)
spike[rng.integers(0, n, 60)] = 3.0
sig1 = fast + slow + spike
X = np.stack([sig1, np.roll(sig1, 1), np.roll(sig1, 2)], 1)
X = np.concatenate([X, rng.standard_normal((n, 5)) * 0.01], 1)
Y = np.roll(sig1, -1)[:, None]

b = NESTBrain(layers=1, input_dim=8, hidden_dim=8, n_columns=128, topk=8,
              n_neurons=4, out_dim=1, seed=0)
p = np.empty_like(Y)
for i in range(n):
    p[i] = b.step(X[i], Y[i], readout=True, k_update=True, dopamine="scalar",
                  wake_interval=100000)   # no waking: we want to see deaths

layer = b.layers[0]
act = layer.act_count
surv = layer.survival
dorm = layer.dormant
order = np.argsort(-act)

print(f"R2 (last 1000 steps) = {r2(Y[-1000:], p[-1000:]):+.3f}")
print(f"dormant columns      = {dorm.sum()} / {len(dorm)}")
print()
print("  rank   activations   survival    dormant")
for r, j in enumerate(order[:6]):
    print(f"  {r:<5} {act[j]:>10.0f}   {surv[j]:>+8.4f}    {bool(dorm[j])}")
print("  ...")
for j in order[-4:]:
    print(f"  {int(np.where(order == j)[0][0]):<5} {act[j]:>10.0f}   {surv[j]:>+8.4f}    {bool(dorm[j])}")

live = act > 0
print()
print(f"mean survival of columns that NEVER fired : {surv[~live].mean():+.4f}  (n={int((~live).sum())})")
print(f"mean survival of columns that fired       : {surv[live].mean():+.4f}  (n={int(live.sum())})")
corr = np.corrcoef(act, surv)[0, 1]
print(f"corr(activation count, survival)          : {corr:+.3f}")
print()
print("reading: survival should reward columns that earned it;")
print("         a strongly negative correlation means the rule kills its own best columns.")
