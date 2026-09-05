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

AVX2 is used only through runtime dispatch on supported x86 GCC/Clang builds. The extension is never compiled globally with `-mavx2`; unsupported CPUs and non-x86 targets use scalar code.

## Build modes

### Pure Python (default)

```bash
python -m pip install .
```

The default wheel is universal and remains `py3-none-any`.

### Native extension

```bash
python -m pip wheel . --no-deps --no-build-isolation --config-settings native=true
```

The native build uses the active interpreter's compiler/sysconfig settings and emits a platform-local CPython wheel.

Native acceleration can be disabled at runtime with:

```bash
KLL_SKETCH_DISABLE_NATIVE=1 python your_program.py
```

or:

```python
from kll_sketch import set_native_enabled
set_native_enabled(False)
```

## Development

```bash
python -m pip install -r requirements-test.txt
pytest -q
python benchmarks/bench_native.py --N 500000
python benchmarks/competitive_kll_focus.py --N 250000 --k 200 --trials 5
```

See [`docs/native.md`](docs/native.md) and [`docs/release-checklist.md`](docs/release-checklist.md) for native and release details.

## License

Apache-2.0
