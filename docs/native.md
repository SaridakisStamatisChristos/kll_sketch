# Native acceleration architecture

KLL v3 adds an optional **CPython C API / C++17** acceleration layer while keeping the Python implementation authoritative.

## Compatibility contract

The extension is an optimization layer, not a second KLL algorithm.

1. `KLL` remains the same Python class in both modes.
2. KLL2 serialization and KLL1 read compatibility are unchanged.
3. Native batch compaction consumes the same SplitMix64 bits in the same order as the Python reference.
4. Native sorting is stable so equal values (including `+0.0` / `-0.0`) preserve Python ordering semantics.
5. Native bulk attempts are transactional. A rejected attempt commits no sketch state, then the input is replayed through the canonical Python method.
6. Only replay-safe indexed/sized inputs are eligible for native bulk probing; generic/one-shot iterators stay on the incremental Python path.
7. Users can force Python with `KLL_SKETCH_DISABLE_NATIVE=1` or `set_native_enabled(False)`.
8. Quantile lookup falls back to Python when `n > 2**53`, retaining Python's exact int/float rank-comparison semantics at enormous cumulative weights.

## Accelerated paths

The extension implements:

- **`ingest_batch`** — bulk level-0 insertion and complete KLL compaction/state transition;
- **`compact_level`** — stable sorting and parity-based promotion for Python-driven scalar/weighted/merge paths;
- **`materialize`** — flatten retained levels, attach powers-of-two weights, stable-sort, and build cumulative weights;
- **`ranks_many`** — batched binary-search ranks;
- **`quantiles_many`** — batched cumulative-weight quantile lookup for exactly representable rank ranges.

All retained values crossing the C boundary are finite Python `float`/C `double` values, matching the core runtime's coercion policy.

## SIMD policy

For one-dimensional C-contiguous buffers with native PEP 3118 `double` format (`d`), data is copied once into native storage.

On GCC/Clang x86 builds:

- a function-specific AVX2 implementation validates finiteness four doubles at a time;
- AVX2 is selected only after runtime CPU feature detection;
- the module itself is **not** globally compiled with `-mavx2`;
- extrema are reduced afterward using scalar `<` / `>` comparisons initialized from the first element, reproducing Python's tie/signed-zero behavior.

If AVX2 is unavailable, or on non-x86 / current MSVC builds, the buffer scan is scalar. The rest of the KLL state machine still runs in C++.

`native_backend_info()` exposes `simd='avx2-runtime'` or `simd='scalar'` plus compiler and API version.

## Memory and failure safety

The native extension reconstructs the current levels into local C++ vectors, validates the incoming batch, then evolves only that local state. Python sketch fields are replaced only after a successful result returns.

This design intentionally costs one state conversion per bulk call in exchange for a strong transactional boundary: invalid values, unsupported capacity depths, or arithmetic overflow cannot leave a half-native Python object.

The compactor avoids references into the outer `std::vector` across a resize, because adding a new KLL level can relocate that outer vector. This invariant is regression-tested through the large-stream native matrix.

## Build model

Normal builds remain pure Python:

```bash
python -m pip wheel .
```

Native builds are explicit:

```bash
python -m pip wheel . --config-settings native=true
```

For a source checkout:

```bash
python -m kll_sketch._native_build
```

The in-tree builder invokes the platform compiler directly using Python `sysconfig` information:

- GCC/Clang: `-O3 -DNDEBUG -std=c++17 -fPIC -shared`;
- macOS adds `-undefined dynamic_lookup`;
- MSVC: `/O2 /DNDEBUG /EHsc /std:c++17 /LD` plus the active Python import library.

No pybind11, Cython, setuptools, Meson, scikit-build, or NumPy is required.

Native wheels produced by this backend are tagged for the active CPython/platform and are intended as platform-local artifacts. The default pure wheel remains the portable distribution.

## Differential validation

`kll_sketch/tests/test_native.py` compares native and pure execution using the same seeds and data, including:

- `debug_state()` and full `to_bytes()` equality;
- quantiles, ranks, and CDF;
- list/range ingestion;
- contiguous `array('d')` ingestion;
- signed-zero stability;
- invalid-batch partial-progress replay;
- one-shot iterator safety;
- `n > 2**53` query fallback;
- class identity and `copy()` behavior.

`benchmarks/bench_native.py` also requires byte-identical serialized state before it evaluates speed. A fast but semantically divergent build therefore fails rather than producing a flattering benchmark.

## Scope boundaries

Native acceleration does not change:

- the KLL error model;
- weighted-update semantics;
- merge semantics or `min_k`;
- KLL1/KLL2 formats;
- the canonical pure-Python implementation.

Disabling or deleting the extension reduces performance, not capability.
