"""Non-stationary benchmark: regime A -> B -> A.

Run:
    python -m brain_gf.bench_ns --quick
    python brain_gf/bench_ns.py --outdir D:/cp_dan/runs/brain_gf

Outputs (in --outdir):
    results.json   all metrics
    rolling.csv    causal rolling MSE per method
    fig_ns.png     rolling MSE + per-regime R^2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from brain_gf import baselines as bl
from brain_gf.gf_brain import FeatureReadoutBrain, PureAttentionBrain
from brain_gf.numerics import r2, rolling_mse
from brain_gf.tasks import DEFAULT_SEGMENTS, QUICK_SEGMENTS, make_stream

PLOT_ORDER = ["ridge_refit", "gf_full", "gf_nodecay", "attn_resid", "attn_plain"]


def run_stream(stepper, X: np.ndarray, Y: np.ndarray):
    preds = np.zeros_like(Y)
    t0 = time.perf_counter()
    for t in range(len(X)):
        preds[t] = stepper(X[t], Y[t])
    dt = time.perf_counter() - t0
    return preds, 1e6 * dt / len(X)


def brain_stepper(brain):
    def f(x, y):
        _s, d = brain.step(x, y)
        return d

    return f


def attn_stepper(brain):
    def f(x, y):
        return brain.step(x, y)

    return f


def ridge_predictions(feature_fn, X, Y, segments):
    """Refit the closed-form readout per regime (offline upper bound, no gradients)."""
    F = np.stack([feature_fn(x) for x in X])
    preds = np.full_like(Y, np.nan)
    s = 0
    for _name, n in segments:
        half = n // 2
        # fit on the first half of the regime, score on the second half: no leakage
        oracle = bl.RidgeOracle().fit(F[s:s + half], Y[s:s + half])
        preds[s + half:s + n] = oracle.predict(F[s + half:s + n])
        s += n
    return preds


def evaluate(preds: np.ndarray, Y: np.ndarray, segments, window: int):
    seg_reports = []
    s = 0
    for name, n in segments:
        sl = slice(s, s + n)
        tail = min(200, max(1, n // 2))
        P, T = preds[sl], Y[sl]
        ok = np.isfinite(P).all(axis=1)
        Pt, Tt = preds[s + n - tail:s + n], Y[s + n - tail:s + n]
        okt = np.isfinite(Pt).all(axis=1)
        seg_reports.append(
            {
                "regime": name,
                "r2": r2(T[ok], P[ok]) if ok.any() else float("nan"),
                "r2_tail": r2(Tt[okt], Pt[okt]) if okt.any() else float("nan"),
                "mse": float(((T[ok] - P[ok]) ** 2).mean()) if ok.any() else float("nan"),
            }
        )
        s += n

    safe = np.where(np.isfinite(preds), preds, Y)   # NaN rows contribute zero error
    rm = rolling_mse(Y, safe, window)
    recovery = []
    s = 0
    for i, (_name, n) in enumerate(segments):
        if i == 0:
            s += n
            continue
        base = float(np.median(rm[max(0, s - window) : s]))
        thr = 1.5 * base
        hit = None
        for t in range(s + 1, min(s + n, len(rm))):
            if rm[t] <= thr:
                hit = t - s
                break
        recovery.append(
            {"switch_at": int(s), "steps": None if hit is None else int(hit), "baseline_mse": base}
        )
        s += n
    return seg_reports, recovery, rm


def build_methods(dim: int, seed: int):
    methods = {
        "gf_full": ("brain", FeatureReadoutBrain(dim, dim, seed=seed)),
        "gf_nodecay": ("brain", FeatureReadoutBrain(dim, dim, decay=1.0, seed=seed)),
        "gf_nocos": ("brain", FeatureReadoutBrain(dim, dim, use_cos=False, seed=seed)),
        "gf_nostress": ("brain", FeatureReadoutBrain(dim, dim, use_stress=False, seed=seed)),
        "attn_resid": ("attn", PureAttentionBrain(dim, dim, layers=3, slots=32, seed=seed)),
        "attn_plain": (
            "attn",
            PureAttentionBrain(dim, dim, layers=3, slots=32, residual=False, seed=seed),
        ),
    }
    # honest offline reference: fit on the first half of each regime, score the second
    methods["ridge_refit"] = ("ridge", None)
    return methods


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="short stream, no frozen pretrain")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=8)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument(
        "--outdir", default=os.path.join("D:/cp_dan/runs/brain_gf"), help="where to write results"
    )
    args = ap.parse_args(argv)

    segments = QUICK_SEGMENTS if args.quick else DEFAULT_SEGMENTS
    X, Y, _tags = make_stream(segments=segments, dim=args.dim, seed=args.seed)
    n0 = segments[0][1]

    feature_brain = FeatureReadoutBrain(args.dim, args.dim, seed=args.seed)
    methods = build_methods(args.dim, args.seed)

    results = {}
    rolling = {}
    for name, (kind, obj) in methods.items():
        t_start = time.perf_counter()
        if kind == "brain":
            preds, us = run_stream(brain_stepper(obj), X, Y)
            counts = obj.counts()
        elif kind == "attn":
            preds, us = run_stream(attn_stepper(obj), X, Y)
            counts = obj.counts()
        elif kind == "ridge":
            preds = ridge_predictions(feature_brain.features, X, Y, segments)
            us = float("nan")
            counts = {"trainable": int(feature_brain.fdim * args.dim), "frozen": 0}
            counts["total"] = counts["trainable"]
        else:  # pragma: no cover
            continue

        seg_reports, recovery, rm = evaluate(preds, Y, segments, args.window)
        rolling[name] = rm
        results[name] = {
            "params": counts,
            "us_per_step": None if np.isnan(us) else float(us),
            "wall_s": float(time.perf_counter() - t_start),
            "segments": seg_reports,
            "recovery": recovery,
            "mse_overall": float(((Y - preds) ** 2).mean()),
        }
        print(f"  {name:<14} done in {results[name]['wall_s']:6.2f}s")

    # ---------------- report ----------------
    print()
    hdr = f"{'method':<14}{'R2.A1':>8}{'R2.B':>8}{'R2.A2':>8}{'rec@B':>7}{'rec@A2':>8}{'us/step':>10}{'params':>9}"
    print(hdr)
    print("-" * len(hdr))
    for name in methods:
        r = results[name]
        segs = r["segments"]
        rec = r["recovery"]
        rec_b = rec[0]["steps"] if len(rec) > 0 and rec[0]["steps"] is not None else -1
        rec_c = rec[1]["steps"] if len(rec) > 1 and rec[1]["steps"] is not None else -1
        us = "-" if r["us_per_step"] is None else f"{r['us_per_step']:.1f}"
        print(
            f"{name:<14}{segs[0]['r2']:>8.3f}{segs[1]['r2']:>8.3f}{segs[2]['r2']:>8.3f}"
            f"{rec_b:>7}{rec_c:>8}{us:>10}{r['params']['trainable']:>9}"
        )

    os.makedirs(args.outdir, exist_ok=True)
    payload = {
        "config": {"segments": list(segments), "dim": args.dim, "window": args.window, "seed": args.seed},
        "methods": results,
    }
    with open(os.path.join(args.outdir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    names = list(rolling)
    with open(os.path.join(args.outdir, "rolling.csv"), "w", encoding="utf-8") as fh:
        fh.write("method,t,rolling_mse\n")
        for nm in names:
            arr = rolling[nm]
            for t, v in enumerate(arr):
                fh.write(f"{nm},{t},{v:.8g}\n")

    # ---------------- figure ----------------
    plot_names = [n for n in PLOT_ORDER if n in rolling]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    for nm in plot_names:
        ax.semilogy(rolling[nm], label=nm, linewidth=1.4)
    s = 0
    for i, (_nm, n) in enumerate(segments):
        s += n
        if i < len(segments) - 1:
            ax.axvline(s, color="k", linestyle="--", alpha=0.35)
    ax.set_xlabel("step")
    ax.set_ylabel("rolling MSE (log)")
    ax.set_title("Non-stationary tracking (A -> B -> A)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    ax = axes[1]
    width = 0.8 / max(len(plot_names), 1)
    xs = np.arange(len(segments))
    for k, nm in enumerate(plot_names):
        vals = [d["r2"] for d in results[nm]["segments"]]
        ax.bar(xs + k * width, vals, width=width, label=nm)
    ax.set_xticks(xs + width * (len(plot_names) - 1) / 2)
    ax.set_xticklabels([nm for nm, _ in segments])
    ax.set_ylim(min(0.0, ax.get_ylim()[0]), 1.05)
    ax.set_ylabel("R^2 per regime")
    ax.set_title("Accuracy per regime")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig_path = os.path.join(args.outdir, "fig_ns.png")
    fig.savefig(fig_path, dpi=140)
    print(f"\nwrote {args.outdir}/results.json, rolling.csv, fig_ns.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
