"""Tune the fast path of the hybrid (window = 1/(1-decay) must exceed the regime length)."""
import numpy as np
from brain_gf.gf_brain import FeatureReadoutBrain
from brain_gf.nrir import NRIRBrain
from brain_gf.hybrid import Hybrid, task_a, task_b, SEGS, X, Y, score

print(f"{'lr_fast':>8} {'decay_fast':>11} {'A':>7} {'B':>7} {'A2':>7} {'mean':>7}")
for lr_fast in (0.2, 0.5):
    for decay_fast in (0.98, 0.995, 1.0):
        b = Hybrid(lr_slow=0.05, lr_fast=lr_fast, decay_slow=1.0, decay_fast=decay_fast, seed=0)
        p = np.empty_like(Y)
        for i in range(len(X)):
            p[i] = np.asarray(b.step(X[i], Y[i])).reshape(-1)
        a_, b_, a2 = score(p)
        print(f"{lr_fast:>8} {decay_fast:>11} {a_:>+7.3f} {b_:>+7.3f} {a2:>+7.3f} {(a_+b_+a2)/3:>+7.3f}")
