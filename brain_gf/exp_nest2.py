"""Experiment 2: does depth help? (layers 1 / 2 / 4 / 8, 64 columns each)"""
import numpy as np
from brain_gf.nest import NESTBrain, signals, r2, run_sequence

N = 1200
A_ = signals(N + 1, dim=8, seed=0, kind="sine")
X, Y = A_[:-1], A_[1:]

print(f"{'layers':>7} {'R2':>8} {'us/step':>9} {'params(frozen)':>15}")
for L in (1, 2, 4, 8):
    b = NESTBrain(layers=L, input_dim=8, hidden_dim=8, n_columns=64, topk=8,
                  n_neurons=4, out_dim=8, seed=0)
    p, us = run_sequence(b, X, Y, readout=True, k_update=True, dopamine="vector")
    c = b.counts()
    print(f"{L:>7} {r2(Y, p):>+8.3f} {us:>9.1f} {c['frozen']:>15}")
