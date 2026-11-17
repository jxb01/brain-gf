"""Experiment 1: can a single-layer NEST-Brain learn online?

Three variants, same data, same budget:
  A. spec            - pred = h (no readout), K updated by scalar dopamine
  B. spec + readout  - pred = h @ W, K updated by scalar dopamine
  C. readout only    - K frozen, only W written (the control)
  D. vector dopamine - pred = h @ W, K credited by per-column alignment
"""
import numpy as np
from brain_gf.nest import NESTBrain, signals, r2, run_sequence

N = 1500
X_all = signals(N + 1, dim=8, seed=0, kind="sine")
X, Y = X_all[:-1], X_all[1:]

def build(**kw):
    return NESTBrain(layers=1, input_dim=8, hidden_dim=8, n_columns=128,
                     topk=8, n_neurons=4, out_dim=8, seed=0, **kw)

cases = [
    ("A spec (no readout, scalar dop)", dict(readout=False, k_update=True,  dopamine="scalar")),
    ("B spec+readout (scalar dop)",     dict(readout=True,  k_update=True,  dopamine="scalar")),
    ("C readout only (K frozen)",       dict(readout=True,  k_update=False, dopamine="scalar")),
    ("D vector dopamine",               dict(readout=True,  k_update=True,  dopamine="vector")),
]

print(f"{'variant':<34} {'R2':>8} {'us/step':>9} {'trainable':>10} {'frozen':>9} {'dormant%':>9}")
for tag, kw in cases:
    b = build()
    preds, us = run_sequence(b, X, Y, **kw)
    c = b.counts()
    dor = 100.0 * sum(l.dormant.sum() for l in b.layers) / (128 * len(b.layers))
    print(f"{tag:<34} {r2(Y, preds):>+8.3f} {us:>9.1f} {c['trainable']:>10} {c['frozen']:>9} {dor:>8.1f}%")

print()
print("note: with 8-dim input, hidden 8, the GRU columns give the brain a memory;")
print("the only thing that can learn a mapping is K (routing) and W (readout).")
