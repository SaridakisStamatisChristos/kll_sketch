# KLL Streaming Quantile Sketch

A high-integrity, mergeable **KLL quantile sketch** for Python with reproducible seeded randomness, exact extrema, strict versioned serialization, rank-space validation, zero runtime dependencies, and an **optional C++17 native acceleration backend**.

Version **3.0** preserves the v2 algorithm, public `KLL` class, and `KLL2` wire format. Native code is an optimization layer: if it is absent or disabled, the same API runs through the canonical pure-Python implementation.

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
- native execution is differential-tested against Python down to serialized state.

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

Version 3.0 does **not** introduce a new serialization format.

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

Accelerated operations include:

- transactional bulk ingestion and KLL compaction;
- contiguous native-`double` buffer ingestion such as `array('d')`;
- stable level compaction sorting;
- weighted query-view materialization;
- batched rank lookup;
- batched quantile lookup.

The native state machine consumes the same SplitMix64 bits in the same places as Python. CI therefore checks byte-identical `to_bytes()` results, not merely statistically similar quantiles.

### SIMD policy

On GCC/Clang x86 builds, compatible contiguous `double` buffers use a **runtime-dispatched AVX2 finite-value scan** when AVX2 is present. Extrema are then reduced with Python-equivalent comparison/tie semantics so signed zero cannot change serialized state. The extension is not globally compiled with `-mavx2`, so older x86 CPUs retain a scalar path. Non-x86 and current MSVC builds use the scalar scan while retaining the rest of the C++ engine.

```python
from kll_sketch import native_backend_info
print(native_backend_info())
# {'available': True, 'enabled': True, 'compiler': 'gcc',
#  'simd': 'avx2-runtime', 'api_version': 1}
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
# kll_sketch-3.0.0-py3-none-any.whl
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

The native benchmark aborts if native and Python serialized states differ.

Optional Apache DataSketches comparison:

```bash
python -m pip install '.[compare]'
python benchmarks/compare_datasketches.py --N 1000000 --k 200
```

## Validation and CI

```bash
python -m pip install -r kll_sketch/requirements-test.txt
python -m pytest -q kll_sketch/tests --cov=kll_sketch --cov-report=term-missing
```

CI validates:

- pure Python on Linux/macOS/Windows × Python 3.10–3.14;
- native C++ on Linux/macOS/Windows across representative supported Python versions;
- native/Python byte-level state parity;
- invalid-input fallback and one-shot iterator safety;
- signed-zero stability;
- enormous-rank (`n > 2**53`) query fallback;
- universal pure wheel build/install;
- platform native wheel build/install;
- no-index pure source installation;
- rank-space regression gates;
- native speed regression gates.

## Offline pure installation

The in-tree PEP 517 backend has no third-party build dependency. A normal source install remains index-independent:

```bash
python -m venv .venv
. .venv/bin/activate
PIP_NO_INDEX=1 python -m pip install --no-index .
```

Native compilation is always explicit.

## Compatibility notes for 3.0

- Python 3.10+ remains supported.
- `KLL`/`KLLSketch` identity is unchanged.
- KLL2 serialization is unchanged; KLL1 remains readable.
- Native acceleration is optional and removable without losing functionality.
- unsupported native inputs replay through the canonical Python path;
- quantile lookup falls back to Python when cumulative integer ranks exceed exact binary64 integer range (`2**53`);
- default wheels remain `py3-none-any`.

## References

- Zohar Karnin, Kevin Lang, Edo Liberty, **Optimal Quantile Approximation in Streams**, FOCS 2016.
- Apache DataSketches KLL implementations and characterization methodology.

## License

Apache License 2.0.
