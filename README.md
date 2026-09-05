# KLL Streaming Quantile Sketch

A high-integrity, mergeable **KLL quantile sketch** for Python with reproducible seeded randomness, exact extrema, strict versioned serialization, rank-space validation, zero runtime dependencies, and an **optional C++17 native acceleration backend**.

Version **3.2** keeps the v2/v3 public `KLL` API and `KLL2` wire format intact while specializing the resident native merge engine. A fresh empty destination can now adopt an already-valid same-`k` resident source hierarchy directly, avoiding the v3.1 empty-state bootstrap and general merge planner. The destination keeps its own SplitMix64 RNG and compaction count, so later compactions remain byte-identical to the pure-Python reference. Non-empty merges stay on the proven v3.1 preflighted engine.

## Core guarantees

- normalized **rank error** is the primary accuracy metric;
- retained level `h` items have implicit weight `2**h`;
- `n == sum(2**h * len(level[h]))` is validated structurally;
- minimum and maximum remain exact after compaction;
- seeded SplitMix64 compaction is reproducible;
- `min_k` tracks inherited merge quality without changing configured `k`;
- repeated queries share a mutation-invalidated weighted view;
- `KLL2` payloads are checksummed and hostile input fails closed;
- historical `KLL1` payloads remain readable;
- native execution is differential-tested against Python down to serialized state;
- disabling the extension changes performance, not the public semantics.

## Quick start

```python
from kll_sketch import KLL, native_backend_info

sk = KLL(capacity=200, rng_seed=7331)
sk.extend([1, 5, 2, 9, 3, 6, 4, 8, 7])

print(sk.median())
print(sk.quantiles_at([.1, .5, .9]))
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
| `add(x, weight=1)` | Insert one value or positive integer-weighted value |
| `extend(xs)` / `update_many(xs)` | Bulk ingestion; native when a safe sized sequence/buffer is available |
| `quantile(q)` / `quantiles_at(qs)` | Single or batched quantile queries |
| `quantiles(m)` / `median()` | Equal-mass cuts / median |
| `rank(x)` / `ranks(xs)` | Absolute rank queries |
| `normalized_rank(x)` | Rank divided by `n` |
| `cdf(xs)` / `pmf(cuts)` | Distribution queries |
| `merge(other)` | Merge another sketch |
| `normalized_rank_error()` | Empirical ~99% normalized-rank error model |
| `quantile_lower_bound/upper_bound` | Error-model-derived bounds |
| `to_bytes()` / `from_bytes()` | Strict KLL2 serialization / KLL1+KLL2 reading |
| `validate()` / `debug_state()` | Structural diagnostics |
| `native_available()` | Is the compiled extension importable? |
| `native_enabled()` | Is native dispatch active? |
| `native_backend_info()` | Compiler/SIMD/backend diagnostics |
| `set_native_enabled(bool)` | Process-local backend switch |

`KLLSketch` remains a direct alias of `KLL`.

## Accuracy model

KLL controls rank error rather than value-space distance. The implementation exposes the established empirical model:

```text
single-sided:  2.296 / k^0.9723
PMF:           2.446 / k^0.9433
```

Representative single-sided values:

| k | Approx. normalized rank error |
|---:|---:|
| 100 | 2.61% |
| 200 | 1.33% |
| 400 | 0.68% |
| 800 | 0.35% |

These are characterization values, not per-instance proofs. See [`docs/algorithm.md`](docs/algorithm.md) and [`docs/benchmarks.md`](docs/benchmarks.md).

## Native acceleration

The optional extension uses the **CPython C API + C++17** directly. There is no pybind11, Cython, NumPy, setuptools, Meson, or scikit-build dependency.

### Resident-state execution

When the native backend is installed and enabled, compatible operations can transition a sketch into an opaque resident C++ state. While resident:

- bulk ingestion evolves the C++ KLL state directly;
- a merge into a fresh empty same-`k` destination can adopt the source hierarchy directly while preserving destination RNG/compaction state;
- non-empty resident merges use the v3.1 structural preflight and exact compaction engine;
- rank and quantile queries use a native weighted query view;
- the hot `quantiles_at` method enters C++ through a direct CPython method descriptor;
- successful resident merges return only the metadata Python needs immediately instead of rebuilding a full state tuple;
- serialization, validation, copying, weighted/scalar fallbacks, or disabling native materialize the canonical Python levels on demand.

This keeps `KLL` as the same Python class and preserves the pure implementation as an executable reference rather than exposing a separate native sketch type.

The native state machine consumes the same SplitMix64 bits in the same places as Python. CI checks byte-identical `to_bytes()` results after native execution, including continued ingestion after resident merges. The v3.2 empty-destination regression uses different source/destination seeds and forces later compactions specifically to prove that source RNG state is not accidentally inherited.

### SIMD policy

On GCC/Clang x86 builds, compatible contiguous native-`double` buffers use a **runtime-dispatched AVX2 scan** that validates finiteness and reduces extrema four doubles at a time. Signed-zero batches take the tie-preserving path required for Python-equivalent serialized state. The extension is not globally compiled with `-mavx2`, so older x86 CPUs remain safe. Non-x86 and current MSVC builds use the scalar scanner while retaining the rest of the C++ engine.

```python
from kll_sketch import native_backend_info
print(native_backend_info())
# {'available': True, 'enabled': True, 'compiler': 'gcc',
#  'simd': 'avx2-runtime', 'api_version': 1, 'persistent_state': True}
```

### Build native in a checkout

```bash
python -m kll_sketch._native_build
python -c "from kll_sketch import native_backend_info; print(native_backend_info())"
```

A C++17 compiler and Python development headers for the active interpreter are required. On Windows use an MSVC developer environment.

### Pure and native wheels

The default wheel stays universal and pure Python:

```bash
python -m pip wheel .
# kll_sketch-3.2.0-py3-none-any.whl
```

Native compilation is explicit:

```bash
python -m pip wheel . --config-settings native=true
```

That produces a wheel tagged for the active CPython/platform and containing the extension. Native wheels built this way are platform-local artifacts; normal release portability still comes from the universal pure wheel.

### Force pure Python

```bash
KLL_SKETCH_DISABLE_NATIVE=1 python your_program.py
```

or:

```python
from kll_sketch import set_native_enabled
set_native_enabled(False)
```

See [`docs/native.md`](docs/native.md).

## Weighted updates

Positive integer weights are represented through binary level placement. Total represented mass remains exact. Applications that require the ordinary unweighted KLL statistical model should ingest observations individually; a weighted stream can have a different compaction history from replaying an expanded interleaved stream.

## Benchmarking

Pure rank-space characterization:

```bash
python benchmarks/bench_kll.py \
  --outdir bench_out \
  --Ns 1e5 \
  --capacities 100 200 400 800 \
  --distributions uniform normal exponential pareto bimodal duplicates \
  --trials 5
python benchmarks/validate_benchmarks.py bench_out
```

Native differential/performance gate:

```bash
python -m kll_sketch._native_build
python benchmarks/bench_native.py --N 300000 --k 200
```

Focused public-API comparison against Apache DataSketches KLL:

```bash
python -m pip install numpy datasketches
python benchmarks/competitive_kll_focus.py --N 250000 --k 200 --trials 5
```

Broader multi-library comparison:

```bash
python benchmarks/competitive_quantiles.py --help
```

### v3.2 merge characterization

On the v3.2 candidate gate (Ubuntu 24.04, CPython 3.13, `N=250000`, `k=200`, eight shards), the focused 200-destination public-API benchmark measured:

| metric | kll-sketch native | Apache KLL 5.2 |
|---|---:|---:|
| geo-mean ingestion | 31.45M updates/s | 27.96M updates/s |
| repeated query | 0.398 us | 0.653 us |
| repeated eight-way merge | 50.19 us | 50.49 us |

The broader two-trial cold-destination characterization measured 86.73 us for `kll-sketch-native` versus 66.45 us for Apache KLL. That cold benchmark is intentionally reported rather than hidden: v3.2 removes a meaningful first-merge bootstrap cost and reaches parity in the repeated merge gate, but Apache remains faster for the measured cold one-shot merge workload.

These are observations of GitHub-hosted runners, not portable performance guarantees. The native benchmark aborts if native and Python serialized states differ.

## Validation and CI

```bash
python -m pip install -r kll_sketch/requirements-test.txt
python -m pytest -q kll_sketch/tests --cov=kll_sketch --cov-report=term-missing
```

CI validates:

- pure Python on Linux/macOS/Windows × Python 3.10–3.14;
- native C++ on Linux/macOS/Windows across representative supported Python versions;
- native/Python byte-level state parity;
- resident merge → continued native ingestion → serialization parity;
- empty-destination native merge → later compaction with different source/destination seeds → exact byte parity;
- native-disable synchronization after resident operations;
- invalid-input fallback and one-shot iterator safety;
- signed-zero stability;
- enormous-rank (`n > 2**53`) query fallback;
- universal pure wheel build/install;
- platform native wheel build/install;
- no-index pure source installation;
- rank-space regression gates;
- native speed regression gates.

## Compatibility

Version 3.2 preserves:

- the `KLL` / `KLLSketch` public class identity;
- pure-Python behavior when native acceleration is unavailable or disabled;
- KLL2 serialization bytes for equivalent canonical state;
- KLL1 read compatibility;
- scalar and weighted update semantics;
- merge `min_k` semantics;
- seeded deterministic compaction semantics.

See [`docs/CHANGELOG.md`](docs/CHANGELOG.md) for release details.
