# Benchmark and characterization methodology

Performance and accuracy are treated as **evidence**, not adjectives. Every published
number must identify the code revision, peer implementation/version, runtime, workload,
trial policy, and machine/runner class. Shared-runner measurements are scoped to that
runner and are never portable performance guarantees.

## 1. Accuracy metric: normalized rank error

KLL is a rank sketch. The primary accuracy measurement is realized normalized rank
error, not absolute value-space distance.

For returned estimate `x_hat` at requested quantile `q`, the harness finds the interval
of true ranks occupied by `x_hat` (important for duplicates) and reports the distance
from target rank `q * (N - 1)` to that interval, divided by `N`.

This remains meaningful across uniform, Gaussian, heavy-tailed, multimodal, and
duplicate-heavy distributions. Accuracy is stochastic: compare distributions over
repeated trials, not one sketch instance.

## 2. Internal characterization harness

`benchmarks/bench_kll.py` writes:

- `accuracy_rank.csv`;
- `update_throughput.csv`;
- `query_latency.csv`;
- `merge.csv`;
- `footprint.csv`.

The default sweep covers uniform, Gaussian, exponential, Pareto, bimodal, and
duplicate-heavy data. Deterministic tests separately cover monotone, reverse-monotone,
all-equal, alternating-extreme, signed-zero, malformed serialization, and fallback
semantics.

Each accuracy row includes the implementation's empirical error-model value so observed
error/model ratios can be validated by `benchmarks/validate_benchmarks.py`.

## 3. Performance-regression CI

`benchmarks/performance_regression.py` compares the optional resident native backend to
the pure-Python semantic reference **inside one process on identical deterministic
inputs and seeds**. It covers:

- bulk ingestion;
- repeated batched quantile queries;
- multi-shard merge.

The script also requires byte-identical pure/native KLL2 state for the compared build and
merge fixtures. CI gates on conservative relative speedups rather than absolute wall
clock values. This makes the regression signal more robust to shared-runner load while
keeping correctness/state parity non-negotiable.

Run locally after building the native extension:

```bash
python -m kll_sketch._native_build
python benchmarks/performance_regression.py
```

## 4. Apache DataSketches KLL: primary peer

The primary peer is `datasketches.kll_doubles_sketch`. The focused harness
`benchmarks/competitive_kll_focus.py` uses the same `k`, input arrays, quantile set, and
process for both implementations. Implementation order alternates between trials to
reduce systematic thermal/scheduler bias; short merge operations are amplified over
many pre-created destinations.

The retained v3.2 focused characterization pins:

```bash
python -m pip install numpy==2.5.2 datasketches==5.2.0
python -m kll_sketch._native_build
python benchmarks/competitive_kll_focus.py \
  --N 250000 --k 200 --seed 7331 --trials 5 \
  --query-loops 2000 --shards 8 --merge-loops 200
python benchmarks/competitive_kll_cold_merge.py
```

Equal `k` is not an equal-memory claim. Serialized size is therefore reported alongside
throughput/error. For memory-efficiency studies, also compare equal serialized size or
an explicitly stated memory budget.

## 5. Multi-k / multi-N benchmark matrix

`benchmarks/competitive_kll_matrix.py` expands the Apache comparison over multiple
`N`, `k`, and shard counts. It emits:

- `results.json` — complete environment, methodology, per-cell details, and merge scaling;
- `matrix.csv` — compact `N × k` throughput/query/error/serialized-size table;
- `merge_scaling.csv` — sharded merge scaling table.

The release workflow defaults to:

- `N ∈ {50,000, 250,000, 1,000,000}`;
- `k ∈ {100, 200, 400, 800}`;
- merge shards `∈ {2, 4, 8, 16, 32}`;
- uniform, normal, and duplicate-heavy distributions;
- three paired trials per matrix cell.

Run the same matrix locally:

```bash
python benchmarks/competitive_kll_matrix.py \
  --Ns 50000 250000 1000000 \
  --ks 100 200 400 800 \
  --shards 2 4 8 16 32 \
  --merge-N 250000 --merge-k 200 \
  --distributions uniform normal duplicates \
  --trials 3 --query-loops 1000 --merge-loops 96 \
  --outdir benchmark_matrix
```

GitHub Actions uploads the JSON/CSV directory as the
`apache-kll-benchmark-matrix` artifact. The JSON records the source commit (`GITHUB_SHA`)
and runtime/library versions when available.

## 6. Claim policy

A README/release performance statement is acceptable only when all of the following are
true:

1. it is generated from public APIs on the exact release code or an identified ancestor;
2. peer versions and benchmark dependencies are pinned;
3. raw machine-readable artifacts are retained;
4. repeated trials and measurement ordering are documented;
5. accuracy and serialized footprint are reported where relevant;
6. the statement says exactly which workloads/parameters won, tied, or lost;
7. no shared-runner result is presented as a universal hardware-independent guarantee.

The benchmark workflows are intentionally split: semantic/performance regression is a
merge gate, while Apache peer timing is release characterization. That separation avoids
turning noisy third-party shared-runner timing into a correctness oracle.
