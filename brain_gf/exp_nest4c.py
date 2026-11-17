"""Is it the credit, or the cost term's scale?"""
import numpy as np
from brain_gf.nest import NESTBrain, r2

rng = np.random.default_rng(0)
n = 4000
t = np.arange(n)
sig1 = np.sin(2 * np.pi * 0.18 * t) + 0.5 * np.sin(2 * np.pi * 0.01 * t)
sig1[rng.integers(0, n, 60)] += 3.0
X = np.stack([sig1, np.roll(sig1, 1), np.roll(sig1, 2)], 1)
X = np.concatenate([X, rng.standard_normal((n, 5)) * 0.01], 1)
Y = np.roll(sig1, -1)[:, None]

print(f"{'survival':<10} {'lam_cost':>9} {'corr(act,surv)':>15} {'mean surv fired':>16}")
for mode in ("spec", "ablate"):
    for lam in (0.1, 0.0):
        b = NESTBrain(layers=1, input_dim=8, hidden_dim=8, n_columns=128, topk=8,
                      n_neurons=4, out_dim=1, seed=0)
        p = np.empty_like(Y)
        for i in range(n):
            p[i] = b.step(X[i], Y[i], readout=True, k_update=True, dopamine="scalar",
                          wake_interval=100000, survival=mode, lam_cost=lam)
        layer = b.layers[0]
        act, surv = layer.act_count, layer.survival
        live = act > 0
        print(f"{mode:<10} {lam:>9} {np.corrcoef(act, surv)[0,1]:>+15.3f} {surv[live].mean():>+16.4f}")
