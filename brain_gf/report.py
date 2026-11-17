"""One-command summary of the measured claims (about 15 s).

    python -m brain_gf.report

Prints four blocks that correspond to RESULTS.md sections A/D/E/F.
"""

from __future__ import annotations

import time

import numpy as np

from brain_gf.gf_brain import FeatureReadoutBrain
from brain_gf.nest_ops import OpBrain, OpColumn, signals


def r2(y, p):
    return 1.0 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()


def run_brain(b, X, Y, **kw):
    p = np.empty_like(Y)
    t0 = time.perf_counter()
    for i in range(len(X)):
        p[i] = b.step(X[i], Y[i], **kw)
    return p, 1e6 * (time.perf_counter() - t0) / len(X)


def main():
    # ---------------------------------------------------------------- A. stationary
    print("=" * 74)
    print("A. stationary next-frame prediction, 8-dim sines, single pass, N=1200")
    N = 1200
    s = signals(N + 1, dim=8, seed=0, kind="sine")
    X, Y = s[:-1], s[1:]

    b = OpBrain(1, 8, 8, 128, 16, 8, agg="concat", combine="concat", seed=0, static_mask=True)
    p, us = run_brain(b, X, Y)
    print(f"   op bank, static 16 cols, concat readout : R2={r2(Y, p):+.3f}   {us:5.0f} us/step")

    fr = FeatureReadoutBrain(8, 8, lr_decision=0.15, decay=1.0, seed=0)
    pf = np.empty_like(Y)
    for i in range(len(X)):
        _s, pf[i] = fr.step(X[i], Y[i])
    print(f"   tanh/cos feature bank reference          : R2={r2(Y, pf):+.3f}")

    # ---------------------------------------------------------------- D. A -> B -> A
    print()
    print("=" * 74)
    print("D. non-stationary A(sine) -> B(square) -> A, 1000 steps each")
    SEG = 1000
    segs = []
    for kind in ("sine", "square", "sine"):
        ss = signals(SEG + 1, dim=8, seed=0, kind=kind)
        segs.append((ss[:-1], ss[1:]))
    Xs = np.concatenate([a for a, _ in segs])
    Ys = np.concatenate([b_ for _, b_ in segs])


    def window_r2(p, name, a, b_):
        base = {"A": 0, "B": SEG, "A2": 2 * SEG}[name]
        return r2(Ys[base + a:base + b_], p[base + a:base + b_])


    def seg_r2(p):
        return [r2(Ys[k * SEG:(k + 1) * SEG], p[k * SEG:(k + 1) * SEG]) for k in range(3)]


    b = OpBrain(1, 8, 8, 128, 16, 8, agg="concat", combine="concat", seed=0, static_mask=True)
    p_static, us_s = run_brain(b, Xs, Ys)
    b = OpBrain(1, 8, 8, 128, 16, 8, agg="concat", combine="concat", seed=0, static_mask=False)
    p_dyn, us_d = run_brain(b, Xs, Ys)

    a_ = OpBrain(1, 8, 8, 128, 16, 8, agg="concat", combine="concat", seed=0, static_mask=True)
    d_ = OpBrain(1, 8, 8, 128, 16, 8, agg="concat", combine="concat", seed=0, static_mask=False)
    W = np.zeros((a_.W.shape[0] + d_.W.shape[0], 8))
    p_hyb = np.empty_like(Ys)
    for i in range(len(Xs)):
        _, f1 = a_.forward(Xs[i])
        _, f2 = d_.forward(Xs[i])
        feats = np.concatenate([f1, f2])
        pred = feats @ W
        p_hyb[i] = pred
        e = Ys[i] - pred
        W = W + 0.15 * np.outer(feats, e) / (float(feats @ feats) + 1e-8)

    print(f"   {'config':<24}{'A':>8}{'B':>8}{'A2':>8}{'B[0:100]':>10}{'B[800:]':>9}")
    for tag, p in (("static 16", p_static), ("dynamic 16", p_dyn), ("static+dynamic", p_hyb)):
        sg = seg_r2(p)
        print(f"   {tag:<24}{sg[0]:>+8.3f}{sg[1]:>+8.3f}{sg[2]:>+8.3f}"
              f"{window_r2(p, 'B', 0, 100):>+10.3f}{window_r2(p, 'B', 800, 1000):>+9.3f}")

    # ---------------------------------------------------------------- E. deflation
    print()
    print("=" * 74)
    print("E. ordered deflation: each branch fits the residual of the frozen prefix")
    rng = np.random.default_rng(0)
    cols = [OpColumn(8, 8, rng, n_ops=4, combine="concat") for _ in range(8)]
    F = [np.stack([c.forward(x) for x in Xs]) for c in cols]
    split = 2500
    resid = Ys.copy()
    cum = np.zeros_like(Ys)
    print(f"   {'branch':<10}{'test R2':>10}{'own gain':>10}")
    gains = []
    for j in range(8):
        A_ = F[j][:split].T @ F[j][:split] + 1e-2 * np.eye(F[j].shape[1])
        Wj = np.linalg.solve(A_, F[j][:split].T @ resid[:split])
        contrib = F[j] @ Wj
        gains.append(r2(resid[:split], contrib[:split]))
        resid = resid - contrib
        cum = cum + contrib
        if j in (0, 1, 3, 7):
            print(f"   {j + 1:<10}{r2(Ys[split:], cum[split:]):>+10.3f}{gains[-1]:>+10.3f}")

    # ---------------------------------------------------------------- F. dead zone
    print()
    print("=" * 74)
    print("F. dead-zone write gate, theta = c * rms(e), 3000 steps, theta=0 for the first 200")
    N = 3000
    s2 = signals(N + 1, dim=8, seed=1, kind="sine")
    X2, Y2 = s2[:-1], s2[1:]
    for c in (0.0, 1.0):
        b = OpBrain(1, 8, 8, 128, 16, 8, agg="concat", combine="concat", seed=0, static_mask=True)
        var, skip = 1.0, 0
        p = np.empty_like(Y2)
        for i in range(N):
            theta = 0.0 if i < 200 else c * np.sqrt(var)
            p[i] = b.step(X2[i], Y2[i], dead_err=theta)
            skip += b.skipped
            e = Y2[i] - p[i]
            var = 0.98 * var + 0.02 * float(e @ e) / e.size
        tail_skip = 0
        print(f"   c={c:<4} R2={r2(Y2, p):+.3f}   writes skipped={100.0 * skip / N:5.1f}%")

    print()
    print("See RESULTS.md for the full log, including the negative results.")


if __name__ == "__main__":
    main()
