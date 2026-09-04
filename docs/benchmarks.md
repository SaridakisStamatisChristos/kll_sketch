# Benchmark and characterization methodology

## Primary metric: normalized rank error

KLL is a rank sketch. The primary accuracy measurement is therefore realized normalized rank error, not absolute numerical distance from an exact quantile value.

For returned estimate `x_hat` at requested quantile `q`, the harness finds the interval of true ranks occupied by `x_hat` (important for duplicates) and reports the distance from target rank `q*(N-1)` to that interval divided by `N`.

This makes the metric meaningful across uniform, heavy-tailed, discrete and arbitrarily scaled distributions.

## Distributions

The default sweep includes:

- uniform;
- Gaussian;
- exponential;
- Pareto;
- bimodal mixture;
- duplicate-heavy discrete streams.

The deterministic test suite separately covers monotone, reverse-monotone, all-equal and alternating-extreme inputs.

## Outputs

`bench_kll.py` writes:

- `accuracy_rank.csv`
- `update_throughput.csv`
- `query_latency.csv`
- `merge.csv`
- `footprint.csv`

Each accuracy row includes the sketch's empirical error-model value so observed error/model ratios can be gated in CI.

## CI smoke gate

The CI benchmark is deliberately small enough for shared runners. It checks:

- realized normalized rank error against a conservative multiple of the empirical model;
- a broad pure-Python update-throughput floor;
- cached query p95 latency;
- merge completion;
- non-empty retained-footprint sanity.

Shared-runner timing is not presented as portable benchmark evidence. Release performance claims should use dedicated machines and preserve machine metadata alongside raw CSV artifacts.

## External comparison

`compare_datasketches.py` optionally compares this implementation with Apache DataSketches KLL. It reports throughput, retained population and realized rank error on the same data.

For serious comparisons, use both equal-`k` and equal-serialized-size runs. Equal `k` alone is not a complete memory-efficiency comparison because implementations may store different metadata and item representations.
