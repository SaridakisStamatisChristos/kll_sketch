# KLL Streaming Quantile Sketch

A high-integrity, mergeable **KLL quantile sketch** for Python with reproducible seeded randomness, exact extrema, strict versioned serialization, rank-space validation, zero runtime dependencies, and an optional **resident C++17 / SIMD acceleration backend**.

Version **3.2** keeps the public `KLL` API and `KLL2` wire format stable while moving native hot paths onto persistent C++ state. Pure Python remains the canonical semantic fallback: the extension can be absent or disabled without changing the API.

## Highlights

- deterministic SplitMix64 compaction with seeded reproducibility;
- exact external minimum/maximum;
- mergeable KLL hierarchy with inherited `min_k` quality tracking;
- checksummed `KLL2` serialization plus `KLL1` read compatibility;
- batched quantile/rank/CDF/PMF queries;
- positive integer weighted updates;
- persistent resident native state for ingestion, queries, and merges;
- direct CPython C-level descriptors for hot query/merge calls;
- compaction-aware resident merging with exact structural preflight;
- runtime-dispatched AVX2 finite-value scanning on supported GCC/Clang x86 builds;
- byte-for-byte Python/native state parity tests;
- pure Python support on Python 3.10–3.14 across Linux, macOS, and Windows;
- no runtime dependency and no third-party native build framework.

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

### Merge

```python
a = KLL(200, rng_seed=1)
b = KLL(200, rng_seed=2)
a.extend(range(10_000))
b.extend(range(10_000, 20_000))
a.merge(b)
a.validate()
```

### Serialize

```python
blob = a.to_bytes()
restored = KLL.from_bytes(blob)
assert restored.to_bytes() == blob
```

Version 3.2 does **not** introduce a new serialization format.

## Public API

| API | Meaning |
| --- | --- |
| `add(x, weight=1)` | Insert one value or a positive integer-weighted value |
| `extend(xs)` / `update_many(xs)` | Bulk ingestion |
| `quantile(q)` / `quantiles_at(qs)` | Single or batched quantile queries |
| `quantiles(m)` / `median()` | Equal-mass cuts / median |
| `rank(x)` / `ranks(xs)` | Absolute rank queries |
| `normalized_rank(x)` | Rank divided by `n` |
| `cdf(xs)` / `pmf(cuts)` | Distribution queries |
| `merge(other)` | Merge another sketch |
| `normalized_rank_error()` | Empirical normalized-rank error model |
| `quantile_lower_bound/upper_bound` | Error-model-derived bounds |
| `to_bytes()` / `from_bytes()` | Strict KLL2 serialization / KLL1+KLL2 reading |
| `validate()` / `debug_state()` | Structural diagnostics |
| `native_available()` | Whether the compiled extension is importable |
| `native_enabled()` | Whether native dispatch is active |
| `native_backend_info()` | Compiler/SIMD/backend diagnostics |
| `set_native_enabled(bool)` | Process-local backend switch |

`KLLSketch` remains a direct alias of `KLL`.

## Accuracy model

KLL controls **rank error**, not value-space distance. The implementation exposes the established empirical model:

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

These are characterization values, not per-instance proofs. See [`docs/algorithm.md`](docs/algorithm.md) and [`docs/benchmarks.md`](docs/benchmarks.md).

## Native engine

The optional extension uses the **CPython C API + C++17** directly. It does not require pybind11, Cython, NumPy, setuptools, Meson, or scikit-build.

### Resident state

Version 3.2 keeps an opaque C++ sketch state resident across hot operations instead of reconstructing native vectors from Python lists for every call. The native state preserves the same level capacities, lazy-compaction policy, SplitMix64 bit consumption, signed-zero ordering rules, retained accounting, and serialized state as the Python implementation.

The merge path adds:

- structural preflight before mutation;
- compaction-aware raw-write elision for levels that will immediately compact;
- an exact preflight-recorded compaction sequence for resident merges;
- cached `min_k` metadata with fallback-safe synchronization;
- cached Python slot keys for low-overhead visible-state mirroring;
- direct resident-to-resident C++ merge dispatch.

The query path keeps the weighted query view in C++ and exposes direct C-level batched quantile dispatch, with canonical fallback for unsupported semantics such as ranks beyond exact binary64 integer representation.

### SIMD policy

On GCC/Clang x86 builds, compatible contiguous native-`double` buffers use a runtime-dispatched **AVX2** finite-value scan when AVX2 is available. The extension is not globally compiled with `-mavx2`; unsupported CPUs retain the scalar path. Non-x86 and current MSVC builds use scalar scanning while retaining the rest of the native engine.

```python
from kll_sketch import native_backend_info
print(native_backend_info())
# Example:
# {'available': True, 'enabled': True, 'compiler': 'gcc',
#  'simd': 'avx2-runtime', 'api_version': 1, 'persistent_state': True}
```

### Build native in a checkout

```bash
python -m kll_sketch._native_build
python -c "from kll_sketch import native_backend_info; print(native_backend_info())"
```

A C++17 compiler and Python development headers for the active interpreter are required. On Windows, use an MSVC developer environment.

### Pure and native wheels

The default wheel remains universal and pure Python:

```bash
python -m pip wheel .
# kll_sketch-3.2.0-py3-none-any.whl
```

Native compilation is explicit:

```bash
python -m pip wheel . --config-settings native=true
```

Native wheels are tagged for the active CPython/platform and contain the compiled extension only; native implementation sources/build helpers are intentionally excluded from runtime wheels. The source distribution retains the C++ sources needed to build native artifacts.

### Force pure Python

```bash
KLL_SKETCH_DISABLE_NATIVE=1 python your_program.py
```

or:

```python
from kll_sketch import set_native_enabled
set_native_enabled(False)
```

## Apache DataSketches performance validation

The repository contains public-API same-process comparison gates against `datasketches.kll_doubles_sketch`. Performance is hardware/compiler/workload dependent; the numbers below are **characterization of GitHub-hosted Ubuntu 24.04 runners, not portable guarantees**.

Focused KLL gate, CPython 3.13.15, `N=250,000`, `k=200`, 7 distributions, 8 merge shards:

| Metric | kll-sketch 3.2 native | Apache KLL | Relative result |
| --- | ---: | ---: | ---: |
| Geometric-mean bulk ingestion | 30.81 M updates/s | 29.62 M updates/s | **1.040×** |
| Batched quantile query | 0.362 µs | 0.541 µs | **1.493× speed** |
| 8-way merge | 43.92 µs | 47.86 µs | **1.090× speed** |
| Serialized bytes | 4,933 | 4,864 | Apache ~1.4% smaller |

A separate fresh-destination merge gate amplifies each measurement over 128 destinations and alternates implementation order over 31 paired trials:

| Metric | kll-sketch 3.2 native | Apache KLL |
| --- | ---: | ---: |
| Median fresh 8-way merge | **32.61 µs** | 34.31 µs |
| Paired trials won | **30 / 31** | 1 / 31 |
| Median speed ratio | **1.049×** | — |

These measurements establish a win for the tested hot paths and parameters only. KLL accuracy remains stochastic; compare error distributions over repeated trials rather than treating a single run's worst error as a universal ordering.

Run the gates yourself:

```bash
python -m pip install numpy==2.5.2 datasketches==5.2.0
python -m kll_sketch._native_build
python benchmarks/competitive_kll_focus.py
python benchmarks/competitive_kll_cold_merge.py
```

The broader ecosystem benchmark additionally characterizes Apache t-digest, DDSketch, and standalone Python t-digest:

```bash
python benchmarks/competitive_quantiles.py
```

## Weighted updates

Positive integer weights are represented through binary level placement. Total represented mass remains exact. Applications requiring the ordinary unweighted KLL statistical model should ingest observations individually; a weighted stream can have a different compaction history from replaying an expanded interleaved stream.

## Validation and CI

```bash
python -m pip install -r kll_sketch/requirements-test.txt
python -m pytest -q kll_sketch/tests --cov=kll_sketch --cov-report=term-missing
```

CI validates:

- pure Python on Linux/macOS/Windows × Python 3.10–3.14;
- native C++ on Linux/macOS/Windows across representative supported versions;
- native/Python byte-level state parity;
- serialization and hostile-input behavior;
- invalid-input fallback and one-shot iterator safety;
- signed-zero stability;
- enormous-rank (`n > 2**53`) query fallback;
- universal pure wheel build/install;
- platform native wheel build/install;
- runtime-wheel source hygiene;
- no-index pure source installation;
- rank-space regression gates;
- native speed regression gates;
- focused Apache KLL and robust fresh-merge characterization workflows.

## Offline pure installation

The in-tree PEP 517 backend has no third-party build dependency:

```bash
python -m venv .venv
. .venv/bin/activate
PIP_NO_INDEX=1 python -m pip install --no-index .
```

Native compilation is always explicit.

## Compatibility notes for 3.2

- Python 3.10+ remains supported.
- `KLL` / `KLLSketch` type identity is unchanged.
- the scalar API remains backward compatible;
- KLL2 serialization is unchanged and KLL1 remains readable;
- native acceleration is optional and removable without loss of functionality;
- unsupported native inputs route through the canonical Python path;
- default wheels remain `py3-none-any`;
- native wheels contain the extension but not implementation source files.

## References

- Zohar Karnin, Kevin Lang, Edo Liberty, **Optimal Quantile Approximation in Streams**, FOCS 2016.
- Apache DataSketches KLL implementations and characterization methodology.

## License

Apache License 2.0.
