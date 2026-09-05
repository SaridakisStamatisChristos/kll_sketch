# KLL Streaming Quantile Sketch

A high-integrity, mergeable **KLL quantile sketch** for Python with reproducible seeded
randomness, exact extrema, strict versioned serialization, rank-space validation, zero
runtime dependencies, and an optional **resident C++17 / SIMD acceleration backend**.

Version **3.2.0** keeps the public `KLL` API and `KLL2` wire format stable while moving
compatible hot paths onto persistent C++ state. Pure Python remains the canonical
semantic reference/fallback: removing or disabling the extension changes performance,
not the public contract.

## Highlights

- deterministic SplitMix64 compaction with seeded reproducibility;
- exact external minimum/maximum and exact represented mass;
- mergeable KLL hierarchy with inherited `min_k` quality tracking;
- checksummed `KLL2` serialization plus historical `KLL1` read compatibility;
- batched quantile/rank/CDF/PMF queries and positive integer weighted updates;
- persistent resident native state for ingestion, queries, and merges;
- exact-sequence, structurally preflighted resident merge compaction;
- direct CPython C-level hot query/merge dispatch with canonical fallbacks;
- runtime-dispatched AVX2 finite/extrema scanning on supported GCC/Clang x86 builds;
- deterministic, byte-for-byte Python/native KLL2 parity tests;
- pure Python support on Python 3.10–3.14 across Linux, macOS, and Windows;
- self-hosted PEP 517 build backend with no runtime dependency and no third-party native
  build framework.

## Quick start

```python
from kll_sketch import KLL, native_backend_info

sk = KLL(capacity=200, rng_seed=7331)
sk.extend([1, 5, 2, 9, 3, 6, 4, 8, 7])

print(sk.median())
print(sk.quantiles_at([0.1, 0.5, 0.9]))
print(sk.normalized_rank(5))
print(sk.min_value, sk.max_value)
print(native_backend_info())
```

Merge and validate:

```python
a = KLL(200, rng_seed=1)
b = KLL(200, rng_seed=2)
a.extend(range(10_000))
b.extend(range(10_000, 20_000))
a.merge(b)
a.validate()
```

Strict round-trip serialization:

```python
blob = a.to_bytes()
restored = KLL.from_bytes(blob)
assert restored.to_bytes() == blob
```

Version 3.2.0 does **not** introduce a new serialization format.

## Public API

| API | Meaning |
| --- | --- |
| `add(x, weight=1)` | Insert one value or a positive integer-weighted value |
| `extend(xs)` / `update_many(xs)` | Bulk ingestion |
| `quantile(q)` / `quantiles_at(qs)` | Single or batched quantile queries |
| `quantiles(m)` / `median()` | Equal-mass cuts / median |
| `rank(x)` / `ranks(xs)` | Approximate absolute rank queries |
| `normalized_rank(x)` | Rank divided by represented mass |
| `cdf(xs)` / `pmf(cuts)` | Distribution queries |
| `merge(other)` | Merge another sketch without changing destination `k` |
| `normalized_rank_error()` | Empirical normalized-rank error model |
| `quantile_lower_bound/upper_bound` | Error-model-derived bounds |
| `to_bytes()` / `from_bytes()` | Strict KLL2 serialization / KLL1+KLL2 reading |
| `validate()` / `debug_state()` | Structural diagnostics |
| `native_available()` / `native_enabled()` | Native backend state |
| `native_backend_info()` | Compiler/SIMD/backend diagnostics |
| `set_native_enabled(bool)` | Process-local backend switch |

`KLLSketch` remains a direct alias of `KLL`. See [`docs/api.md`](docs/api.md) for the
stable signatures and edge semantics.

## Accuracy model

KLL controls **rank error**, not value-space distance. The implementation exposes the
established empirical model:

```text
single-sided:  2.296 / k^0.9723
PMF:           2.446 / k^0.9433
```

Representative single-sided characterization:

| k | Approx. normalized rank error |
|---:|---:|
| 100 | 2.61% |
| 200 | 1.33% |
| 400 | 0.68% |
| 800 | 0.35% |

These are engineering characterization values, not deterministic per-instance bounds.
See [`docs/algorithm.md`](docs/algorithm.md) and
[`docs/benchmarks.md`](docs/benchmarks.md).

## Native engine

The optional extension uses the **CPython C API + C++17** directly. It does not require
pybind11, Cython, NumPy, setuptools, Meson, or scikit-build.

### Resident state and merge engine

Compatible sketches can keep an opaque C++ state resident across hot operations instead
of reconstructing native vectors from Python lists on every call. The native state
preserves level capacities, lazy-compaction policy, SplitMix64 bit consumption,
signed-zero ordering rules, retained accounting, exact extrema, and serialized state.

The v3.2 merge path includes structural preflight, compaction-aware raw-write elision,
pre-recorded exact compaction sequencing, cached `min_k` synchronization, and direct
resident-to-resident dispatch. Empty-destination adoption deliberately retains the
destination RNG state and compaction count so later evolution remains identical to the
Python reference.

The query path keeps a mutation-invalidated weighted query view in C++ and uses direct
C-level batched quantile dispatch. Unsupported semantics—including represented mass
beyond exact binary64 integer rank representation—fall back to the canonical Python
path.

### SIMD policy

On GCC/Clang x86 builds, compatible contiguous native-`double` buffers use a
runtime-dispatched **AVX2** finite/extrema scan when AVX2 is available. The extension is
not globally compiled with `-mavx2`; unsupported CPUs retain the scalar path. Non-x86
and current MSVC builds use scalar scanning while retaining the C++ state engine.

```python
from kll_sketch import native_backend_info
print(native_backend_info())
```

See [`docs/native.md`](docs/native.md) for the exact compatibility and synchronization
contract.

## Build and package

Pure source checkout:

```bash
python -m pip install .
```

Build the default universal wheel:

```bash
python -m pip wheel .
# kll_sketch-3.2.0-py3-none-any.whl
```

Build the optional native extension in a checkout:

```bash
python -m kll_sketch._native_build
python -c "from kll_sketch import native_backend_info; print(native_backend_info())"
```

Or build an explicit platform-local native wheel:

```bash
python -m pip wheel . --config-settings native=true
```

A C++17 compiler and Python development headers are required for native compilation.
Runtime native wheels exclude native implementation sources/build helpers; the source
distribution retains those sources plus release/research metadata (`CITATION.cff`,
`CONTRIBUTING.md`, and `SECURITY.md`). Force pure Python with
`KLL_SKETCH_DISABLE_NATIVE=1` or `set_native_enabled(False)`.

## Apache DataSketches KLL performance evidence

Apache DataSketches `kll_doubles_sketch` is the primary peer comparison. Measurements
use public APIs, identical input arrays/`k`, repeated paired trials, and pinned peer
versions. **Every number below is a characterization of the stated GitHub-hosted runner
and workload—not a portable performance guarantee.**

Retained focused gate: Ubuntu 24.04 GitHub-hosted runner, CPython 3.13.15,
Apache DataSketches 5.2.0, `N=250,000`, `k=200`, seven distributions, eight merge shards:

| Metric | kll-sketch 3.2 native | Apache KLL | Relative result |
| --- | ---: | ---: | ---: |
| Geometric-mean bulk ingestion | 30.81 M updates/s | 29.62 M updates/s | **1.040×** |
| Repeated batched quantile query | 0.362 µs | 0.541 µs | **1.493× speed** |
| Repeated 8-way merge | 43.92 µs | 47.86 µs | **1.090× speed** |
| Serialized bytes | 4,933 | 4,864 | Apache ~1.4% smaller |

A separate fresh-destination gate amplified each measurement over 128 destinations and
alternated implementation order over 31 paired trials:

| Metric | kll-sketch 3.2 native | Apache KLL |
| --- | ---: | ---: |
| Median fresh 8-way merge | **32.61 µs** | 34.31 µs |
| Paired trials won | **30 / 31** | 1 / 31 |
| Median speed ratio | **1.049×** | — |

### Multi-`N` / multi-`k` release matrix

A broader release-candidate run on Ubuntu 24.04 / CPython 3.13.15 with
NumPy 2.5.2 and Apache DataSketches 5.2.0 swept `N={50k,250k,1m}` and
`k={100,200,400,800}` over uniform, normal, and duplicate-heavy inputs (three paired
trials per cell):

- ingestion was faster in **9/12** `(N, k)` cells; the other three were near parity,
  with ratios of 0.983×–0.998× Apache;
- repeated batched quantile queries were faster in **12/12** cells, by approximately
  **1.52×–1.58×**;
- median serialized footprint across the tested distributions stayed within roughly
  **-3.0% to +2.5%** of Apache, depending on the cell;
- observed rank error was mixed across cells, as expected for stochastic sketches; the
  matrix is not evidence of universal accuracy dominance by either implementation.

Sharded merge scaling at `N=250,000`, `k=200` showed the trade-off directly:

| Shards | kll-sketch | Apache KLL | kll-sketch / Apache speed |
| ---: | ---: | ---: | ---: |
| 2 | 7.89 µs | 8.21 µs | **1.041×** |
| 4 | 20.77 µs | **19.22 µs** | **0.925×** |
| 8 | **42.31 µs** | 50.10 µs | **1.184×** |
| 16 | **77.65 µs** | 85.28 µs | **1.098×** |
| 32 | **142.54 µs** | 155.42 µs | **1.090×** |

So the defensible release claim is intentionally bounded: **on the tested runner and
workloads, kll-sketch 3.2 was consistently faster for repeated batched quantile queries,
usually faster for ingestion, and faster for four of five measured shard counts; Apache
won the 4-shard merge point.** Serialized footprint and stochastic rank error were
competitive rather than uniformly superior.

Reproduce the focused evidence:

```bash
python -m pip install numpy==2.5.2 datasketches==5.2.0
python -m kll_sketch._native_build
python benchmarks/competitive_kll_focus.py
python benchmarks/competitive_kll_cold_merge.py
```

For release-grade breadth, `benchmarks/competitive_kll_matrix.py` sweeps
`N={50k,250k,1m}`, `k={100,200,400,800}`, and merge shards `{2,4,8,16,32}` by default,
emitting JSON/CSV artifacts with the exact source commit and GitHub run id. See
[`docs/benchmarks.md`](docs/benchmarks.md). The workflow
`.github/workflows/benchmark-matrix.yml` preserves those artifacts; it is
characterization, not a noisy third-party timing merge gate.

## Performance-regression CI

`benchmarks/performance_regression.py` provides the stable performance gate. It compares
resident native execution against the pure-Python reference in the same process on
identical deterministic fixtures and separately covers ingestion, repeated batched
queries, and multi-shard merge. It also requires exact pure/native serialized-state
parity for the benchmark fixtures.

This design catches substantial native regressions without pretending that absolute
shared-runner microseconds are stable across machines.

## Validation and CI

```bash
python -m pip install -r kll_sketch/requirements-test.txt
python -m pytest -q kll_sketch/tests --cov=kll_sketch --cov-report=term-missing
```

The release validation surface includes:

- pure Python: Linux/macOS/Windows × Python 3.10–3.14;
- native C++: Linux/macOS/Windows on representative supported versions;
- native/Python byte-level state parity and deterministic RNG evolution;
- strict KLL1/KLL2 serialization and hostile-input behavior;
- invalid-input fallback, one-shot iterator safety, and signed-zero stability;
- enormous-rank (`n > 2**53`) query fallback;
- universal pure wheel, explicit native wheel, and offline source-install gates;
- rank-space accuracy regression and same-process native performance regression;
- focused Apache KLL, robust fresh-merge, and multi-`k`/multi-`N` release evidence;
- pre-tag release artifact build/content/install verification.

See [`docs/production-readiness.md`](docs/production-readiness.md) and
[`docs/release-checklist.md`](docs/release-checklist.md).

## Weighted updates

Positive integer weights are represented through binary level placement. Total
represented mass remains exact. Applications requiring the ordinary unweighted KLL
statistical model should ingest observations individually; a weighted stream can have a
different compaction history from replaying an expanded interleaved stream.

## Security, contributing, and citation

- Security model/reporting: [`SECURITY.md`](SECURITY.md)
- Contribution/benchmark rules: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Citation metadata: [`CITATION.cff`](CITATION.cff)
- v3.2.0 release notes: [`docs/release-notes-v3.2.0.md`](docs/release-notes-v3.2.0.md)

`CITATION.cff` intentionally contains no DOI until a real archival DOI exists. The
repository is prepared for a Zenodo archive of the tagged GitHub release.

## Compatibility notes for 3.2.0

- Python 3.10+ remains supported.
- `KLLSketch is KLL` remains true.
- KLL2 serialization is unchanged and KLL1 remains readable.
- deterministic seeded semantics and Python/native parity are preserved.
- native acceleration is optional; unsupported inputs route through canonical fallback.
- the canonical distribution remains the pure `py3-none-any` wheel plus source
  distribution; native builds remain explicit.

## References

- Zohar Karnin, Kevin Lang, Edo Liberty, **Optimal Quantile Approximation in Streams**,
  FOCS 2016.
- Apache DataSketches KLL implementations and characterization methodology.

## License

Apache License 2.0.
