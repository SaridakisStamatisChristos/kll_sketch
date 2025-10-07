# KLL Sketch Benchmarking Guide

This guide explains how to generate baseline performance numbers for the local `kll_sketch` implementation, analyse the results, and interpret the reported error metrics.

## Installation

The benchmarks rely on a handful of scientific Python packages. Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[bench]
```

> Windows PowerShell activation: `.venv\Scripts\Activate.ps1`

If you prefer an ad-hoc install without extras, install the runtime packages manually: `pip install -U numpy pandas matplotlib pytest pytest-benchmark jupyter`.

## Running the CLI benchmarks

The entry point lives at `benchmarks/bench_kll.py`. It generates synthetic datasets, measures update throughput, query latency, accuracy against `numpy.quantile`, and merge performance. By default it sweeps the recommended population sizes, sketch capacities, quantiles, and distributions. All artifacts are written beneath `bench_out/`.

Example command (matches the defaults):

```bash
python benchmarks/bench_kll.py \
  --module kll_sketch --class KLLSketch \
  --outdir bench_out \
  --Ns 1e5 1e6 \
  --capacities 200 400 800 \
  --distributions uniform normal exponential pareto bimodal \
  --qs 0.01 0.05 0.1 0.25 0.5 0.75 0.9 0.95 0.99 \
  --shards 8
```

The script produces four CSV files:

- `bench_out/accuracy.csv`: absolute value error for each `(distribution, N, capacity, q)` pair. A `mode` column distinguishes single-pass sketches from merged sketches.
- `bench_out/update_throughput.csv`: update times and inserts/sec for each `(distribution, N, capacity)` combination.
- `bench_out/query_latency.csv`: per-quantile latency measurements in microseconds.
- `bench_out/merge.csv`: merge timings versus shard counts.

## Pytest micro-benchmarks

The micro-benchmarks live under `tests/bench/` and are marked with `@pytest.mark.benchmark` so they are skipped unless explicitly requested. They use `pytest-benchmark` to record update throughput for `N ∈ {1e5, 1e6}` and `capacity ∈ {200, 400, 800}` on uniform and normal data. A lightweight accuracy assertion verifies that the median absolute error stays within `~1%` of the data standard deviation to avoid flaky failures.

Run them with:

```bash
pytest -m benchmark -q --benchmark-json=bench_out/pytest/results.json
```

The `bench_out/pytest/` directory is created automatically so the JSON export succeeds.

## Plotting the results

Open the Jupyter notebook at `benchmarks/bench_plots.ipynb` to load the CSV outputs, generate the summary plots, and inspect basic accuracy tables:

```bash
jupyter notebook benchmarks/bench_plots.ipynb
```

The notebook provides:

- Absolute value error versus quantile, faceted by distribution with one line per capacity.
- Updates/sec versus `N` on a log-scaled horizontal axis, grouped by capacity.
- Query latency (µs) versus capacity, grouped by quantile.
- Merge time versus the number of shards.
- Summary tables for mean/median error and best/worst quantiles.

## Interpreting the metrics

- **Value error vs rank error**: The benchmark tracks *value* error—`|approximate_quantile - exact_quantile|`. This directly measures how far the returned value is from the true quantile. Rank error (difference between approximate and exact ranks) is another perspective, but is not reported here. Large value errors often occur in heavy-tailed regions even when rank error is acceptable.
- **Capacity (`k`)**: The sketch capacity controls the maximum buffer size (and therefore the accuracy/memory trade-off). Higher capacities retain more samples, reducing approximation error at the cost of higher memory usage and slightly slower updates. You should observe smaller value errors as `k` increases, especially for central quantiles.
- **Throughput and latency**: Updates/sec should scale roughly linearly with the input size until memory/cache effects kick in. Query latency (µs per quantile) should remain in the single digits to low tens for the provided capacities.
- **Merge performance**: Shard merges should be significantly faster than building a sketch from scratch. Post-merge accuracy should match single-pass sketches within noise.

## Repository layout

```
kll_sketch/
├── benchmarks/
│   ├── bench_kll.py        # CLI benchmark runner
│   └── bench_plots.ipynb   # Notebook for plotting CSV outputs
├── tests/
│   └── bench/              # Pytest micro-benchmarks (marked)
├── bench_out/              # Outputs (ignored by git)
└── docs/
    └── benchmarks.md       # This guide
```
