"""Experiment 1b: is the per-layer L2 normalisation the ceiling?"""
import numpy as np
from brain_gf.nest import NESTBrain, signals, r2, run_sequence

N = 1500
A_ = signals(N + 1, dim=8, seed=0, kind="sine")
X, Y = A_[:-1], A_[1:]

def run(hidden, renorm, tag):
    b = NESTBrain(layers=1, input_dim=8, hidden_dim=hidden, n_columns=128,
                  topk=8, n_neurons=4, out_dim=8, seed=0, renorm=renorm)
    preds, us = run_sequence(b, X, Y, readout=True, k_update=False, dopamine="scalar")
    print(f"{tag:<30} R2={r2(Y, preds):+.3f}  {us:7.1f} us/step")
    return r2(Y, preds)

print("hidden=8  (pred = unit vector direction only)")
run(8, True,  "hidden=8  renorm=True")
run(8, False, "hidden=8  renorm=False")
print("hidden=32 (input projected, 32-dim state)")
run(32, True,  "hidden=32 renorm=True")
run(32, False, "hidden=32 renorm=False")
print("hidden=64")
run(64, False, "hidden=64 renorm=False")
