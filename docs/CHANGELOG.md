# Changelog

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
