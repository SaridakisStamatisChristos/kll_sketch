# Changelog

## 3.0.0

### Native acceleration

- Added an optional CPython C API / C++17 acceleration module without changing the public `KLL` class.
- Added transactional native bulk ingestion that mirrors the Python KLL state machine, including SplitMix64 compaction-bit consumption.
- Added native stable compaction sorting, weighted query-view materialization, batched rank lookup, and batched quantile lookup.
- Added contiguous native-`double` buffer ingestion.
- Added runtime-dispatched AVX2 finite-value scanning on supported GCC/Clang x86 builds with a scalar fallback.
- Preserved Python-equivalent extrema and tie behavior, including signed-zero stability.
- Added runtime controls: `native_available`, `native_enabled`, `native_backend_info`, and `set_native_enabled`.
- Added `KLL_SKETCH_DISABLE_NATIVE=1` for process-level forced pure-Python operation.
- Restricted native batch probing to replay-safe indexed/sized inputs so one-shot iterators are never consumed by a failed optimization attempt.
- Added exact Python fallback for quantile target comparisons when `n > 2**53`.

### Packaging

- Preserved the default universal `py3-none-any` pure-Python wheel.
- Added explicit native wheel builds via `--config-settings native=true`.
- Added a dependency-free native compiler driver using the active interpreter's `sysconfig` paths.
- Kept native source/build helpers out of runtime wheels.
- Kept normal source installation fully no-index capable and free of third-party build dependencies.

### Validation / CI

- Added Linux/macOS/Windows native build and differential-test coverage across representative supported Python versions.
- Added byte-identical native/Python serialization parity tests for list, range, and contiguous-double ingestion.
- Added signed-zero, invalid-input replay, one-shot iterator, enormous-rank, class-identity, and copy-contract regressions.
- Added a native speed benchmark that fails before reporting performance if serialized state diverges from Python.
- Added pure-wheel and native-wheel install tests outside the source tree.
- Kept the existing 15-job pure-Python OS/Python compatibility matrix and 90% core-Python coverage gate separate from native tooling coverage.

### Compatibility

- `KLL2` serialization is unchanged. `KLL1` remains readable.
- `KLLSketch` remains a direct alias of `KLL`.
- Weighted updates, merge semantics, `min_k`, error reporting, and the pure-Python algorithm are unchanged.
- Native acceleration is optional: removing or disabling the extension changes performance, not functionality.

## 2.0.0

### Algorithm

- Replaced the 1.x KLL-inspired compactor with a KLL-style geometric hierarchy.
- Added one-parity-per-compaction randomized halving.
- Added exact external min/max tracking.
- Added deterministic integer level-capacity calculation.
- Added persistent SplitMix64 RNG state for reproducible seeded operation.
- Added explicit `min_k` merge-quality tracking while preserving destination configured `k`.
- Retained integer-weighted updates using binary level placement.

### Queries

- Added mutation-invalidated cached sorted query views.
- Added batched `ranks`, `normalized_rank`, `pmf`, error-model reporting and quantile bounds.
- Made `q=0` and `q=1` return exact extrema.

### Serialization

- Added strict checksummed `KLL2` format.
- Added payload length, CRC32, RNG state, extrema, `min_k`, compaction count and retained count.
- Added hostile-input validation and `SerializationError`.
- Preserved strict read compatibility with historical `KLL1` payloads.

### Validation

- Reworked accuracy testing and benchmarks around normalized rank error.
- Added adversarial, merge-tree, corruption, byte-stability and property tests.
- Added explicit structural `validate()` and `debug_state()` APIs.
- Raised deterministic suite branch coverage target to 90%.

### Packaging / CI

- Fixed Core Metadata versioning for `License-Expression` / `License-File`.
- Excluded tests and build-backend internals from runtime wheels.
- Repaired offline source installation validation.
- Replaced stale value-error performance gates with rank-space characterization.
- Raised supported Python baseline to 3.10 and added 3.13/3.14 coverage.

### Compatibility changes

- `to_bytes()` now emits KLL2. `from_bytes()` still reads KLL1.
- `KLLSketch` is now a direct alias of `KLL` instead of an empty subclass.
- Merge no longer overwrites configured `k`; inherited lower-quality estimation is represented by `min_k`.
