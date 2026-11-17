# brain-gf

**Gradient-free random-feature adaptive filter.** NumPy only. No backprop, no autograd,
no optimizer state — every "learning" step is one error outer-product written into a readout.

```bash
pip install -e .
python -m brain_gf.report        # reproduces the headline numbers (~15 s)
python -m brain_gf.bench_ns      # A(sine) -> B(square) -> A benchmark
```

## What it is

A kernel machine with a frozen feature map:

| stage | what it is | trained? |
|---|---|---|
| feature bank | 16 columns × 4 hand-written operators (linear / quadratic / abs / threshold / sin / cos / diff / cumsum / normalize / eml) on frozen random projections | **frozen** |
| aggregation | `concat` of all column outputs | — |
| readout | one linear map, written by NLMS online or solved by ridge offline | **the only learner** |

It reaches 0.99 one-step / 0.85 five-step on Lorenz-96, 0.68 on the new regime of a
sine→square switch, at ~330 µs/step in pure Python on one core.

## What it is not

It is **not** a replacement for gradient descent, and this repository is mostly a record of
what does *not* work. Measured and reproducible:

| mechanism | result |
|---|---|
| per-step top-k routing | −0.16 R² (each feature's readout row only updates k/n of the steps) |
| hormone-modulated k | −0.05 … −0.25 |
| structural growth / pruning | within noise (±0.005) |
| survival-score pruning | corr(activation, survival) = −0.85 — it ranks its own best columns for death |
| consensus / cluster pooling | −0.2 and worse; a weighted sum stays inside the same convex hull |
| dopamine update of the routing matrix | +0.004 |
| decay / forgetting | −0.03 … −0.13 (NLMS re-fits by itself) |
| gated tree of local predictors | 0.01 … 0.54 vs 0.74 flat |

Two facts that save time:

1. An additive tree **is** the concatenated readout: `Σ φ_j w_j = [φ_1…φ_n]·[w_1…w_n]`.
   Only gating adds a new function class, and gating cannot learn "who is right" without gradients.
2. Per-branch backfitting on this design **diverges**: the block Gauss-Seidel iteration has
   spectral radius **6.911** (must be < 1). The convergent form is ordered deflation
   (0.960 vs 0.986 for a single readout).

Full measurement log: [brain_gf/RESULTS.md](brain_gf/RESULTS.md).
Design notes and API: [brain_gf/README.md](brain_gf/README.md).

## Layout

```
brain_gf/            the package (library, benchmarks, every experiment that backs a claim)
  nest_ops.py        operator bank + concat readout  (the main object)
  gf_brain.py        earlier orthogonal take: random-feature readout, pure-attention memory
  bench_ns.py        non-stationary benchmark
  report.py          one-command summary
  exp_*.py           the experiment behind each number
  legacy/            early attempts, frozen for provenance
  c_port/            C reference implementation (not compiled here)
```

## License

MIT — see [LICENSE](LICENSE).
