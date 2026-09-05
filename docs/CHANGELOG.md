# Changelog

## 3.2.0

### Merge engine

- Added a specialized native fast path for merging an already-resident source into a fresh empty destination with the same configured `k`.
- The destination adopts the source's already-valid level hierarchy and capacity metadata directly, avoiding empty-state bootstrap plus the general merge planner for the first merge.
- Preserved destination SplitMix64 RNG state and destination compaction count rather than copying those fields from the source, matching the pure-Python merge semantics for all later compactions.
- Preserved `min_k`, exact extrema, retained count, source immutability, and the existing non-empty resident merge engine.
- Kept keyword, disabled-native, nonresident, mixed-`k`, unsupported, and fallback behavior on the established semantic paths.

### Validation

- Added an exact-state regression with different source/destination seeds that performs empty-destination adoption, forces later compactions, and requires byte-identical serialization against pure Python.
- Verified the source sketch remains unchanged after the optimized merge.
- Revalidated pure and native execution on Linux, macOS, and Windows across the existing supported Python matrix, including native-wheel and offline-install gates.

### Performance characterization

- On the retained Ubuntu 24.04 / CPython 3.13 focused gate (`N=250000`, `k=200`, eight shards), measured repeated eight-way merge at 50.19 us versus 50.49 us for Apache DataSketches KLL 5.2 on that runner.
- The same focused run measured 31.45M updates/s versus 27.96M/s and 0.398 us versus 0.653 us for the repeated query set.
- A separate broad cold-destination two-trial run measured 86.73 us versus Apache's 66.45 us; this remaining cold one-shot gap is documented rather than hidden.
- Competitive timings remain runner observations, not portable guarantees.

### Rejected experiments

- Rejected lazy sorted-shadow invalidation/rebuild after it regressed focused merge to roughly 152 us versus roughly 49 us for Apache.
- Rejected skipping the planner for non-compacting resident merges after it regressed the focused merge ratio.
- Rejected peak-allocation pre-reservation after it also regressed the focused merge ratio.
- Retained only the empty-destination adoption optimization that demonstrated a measurable benefit while preserving exact semantics.

### Compatibility

- Public `KLL` / `KLLSketch` API identity is unchanged.
- `KLL2` serialization is unchanged and `KLL1` remains readable.
- Default wheel remains pure `py3-none-any`; native wheels remain explicit platform-local builds.
- Removing or disabling the native extension changes performance, not functionality.

## 3.1.0

### Resident native state

- Added a persistent C++ `SketchState` so compatible sketches can remain native across ingestion, merge, rank, and quantile calls instead of reconstructing native state for every operation.
- Kept `KLL` as the same public Python class; resident state is an internal optimization and pure Python remains the canonical fallback/reference implementation.
- Added on-demand synchronization back to canonical Python levels for serialization, validation, copying, weighted/scalar fallback, and native-disable boundaries.
- Preserved seeded SplitMix64 compaction state, exact extrema, retained-count accounting, and KLL2 serialization semantics across resident execution.

### Query path

- Added a direct CPython method descriptor for hot `KLL.quantiles_at` calls when resident native state is available.
- Retained the Python runtime dispatcher as the semantic fallback for cold state, disabled native mode, invalid/unsupported inputs, and represented mass above `2**53`.
- Kept mutation-invalidated resident weighted query views for repeated rank/quantile workloads.

### Merge path

- Added journaled resident-state merge rollback so successful merges avoid cloning the entire destination sketch while failed native operations can restore destination state.
- Added a compact successful-merge handoff that returns only the retained count needed immediately by Python; native RNG and compaction internals remain authoritative until synchronization.
- Added exact resident merge parity tests covering continued native ingestion after merge and synchronization immediately after native is disabled.
- Rejected benchmark-regressing alternatives during development, including fixed 64-level payload journals, direct public C-level merge dispatch, and blanket deferred higher-level sorting.

### SIMD / ingestion

- Fused AVX2 finiteness validation and extrema reduction for compatible contiguous native-double buffers on supported GCC/Clang x86 builds.
- Preserved signed-zero/extrema tie semantics with the canonical tie-preserving path.
- Retained runtime AVX2 detection and scalar fallback; the extension is still not globally built with `-mavx2`.

### Benchmarking / CI

- Added a focused same-runner public-API benchmark against Apache DataSketches KLL covering ingestion, repeated batched quantiles, eight-way merge, serialized size, and observed rank error.
- Added a broader multi-library competitive quantile benchmark workflow.
- Added same-process optimization controls during development to distinguish implementation effects from shared-runner variance; experimental controls are not part of the production benchmark path.
- Extended native regression coverage for resident merge state and native-disable synchronization.
- Preserved Linux/macOS/Windows native and pure-Python matrices, packaging checks, offline installation, native differential speed gates, and byte-level parity validation.

### Compatibility

- Public `KLL` / `KLLSketch` API identity is unchanged.
- `KLL2` serialization is unchanged and `KLL1` remains readable.
- Default wheel remains pure `py3-none-any`; native wheels remain explicit platform-local builds.
- Removing or disabling the native extension changes performance, not functionality.

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
