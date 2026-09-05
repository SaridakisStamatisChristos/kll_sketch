# Changelog

## 3.2.0

### Resident merge engine

- Added a specialized native fast path for merging an already-resident source into a fresh empty destination with the same configured `k`.
- Preserved the destination SplitMix64 RNG state and destination compaction count during empty adoption rather than copying source execution history.
- Added structural merge preflight that predicts level growth, retained count, capacity evolution, and the exact sequence of compacted levels before mutation.
- Added compaction-aware raw-write elision for higher levels that are proven to compact immediately.
- Added an exact-sequence resident merge executor so the hot path does not rescan the hierarchy after every compaction to rediscover work already proven by preflight.
- Added resident `min_k` caching with synchronization guards for fallback/keyword paths that can tighten Python-visible merge quality metadata.
- Added interned/cached Python slot keys for resident lookup, merge visibility mirroring, empty adoption, and `min_k` synchronization, reducing CPython framing cost without changing sketch mathematics.
- Kept self-merge, keyword, disabled-native, nonresident, mixed-`k`, unsupported, and error paths on the canonical fallback semantics.

### Query path

- Kept the direct C-level `quantiles_at` method descriptor introduced during resident-state work and made resident query dispatch part of the v3.2 production path.
- Native query state remains mutation-invalidated and resident between compatible calls.
- Exact Python fallback remains mandatory for represented mass above `2**53` and for unsupported public-input semantics.

### Validation

- Added exact native/Python state regressions for empty adoption with distinct source/destination seeds followed by later compaction.
- Added resident merge parity tests covering continued ingestion, native-disable synchronization, source immutability, `min_k` inheritance, signed zero, and byte-identical KLL2 serialization.
- Revalidated native compilation and execution across Linux, macOS, and Windows on supported representative Python versions; pure Python remains covered on Python 3.10–3.14 across all three operating systems.
- Added a robust cold-merge benchmark that alternates implementation order over paired trials and amplifies each measurement over many fresh destinations.
- Strengthened wheel-content checks so runtime wheels contain neither `_native*.cpp` nor `_native*.inc` implementation sources/build helpers.

### Performance characterization

A retained public-API comparison against Apache DataSketches KLL 5.2 on a GitHub-hosted Ubuntu 24.04 / CPython 3.13.15 runner (`N=250000`, `k=200`, seven distributions, eight merge shards) measured:

- 30.81M updates/s for `kll-sketch` versus 29.62M/s for Apache (1.040x);
- 0.362 us versus 0.541 us for the repeated batched quantile set (1.493x speed);
- 43.92 us versus 47.86 us for repeated eight-way merge (1.090x speed).

A separate 31-trial fresh-destination gate (128 fresh destinations per trial, alternating implementation order) measured a 32.61 us median for `kll-sketch` versus 34.31 us for Apache; `kll-sketch` won 30 of 31 paired trials, with a 1.049x median speed ratio.

These are workload/runner characterizations, not portable performance guarantees. Serialized state remains slightly larger than Apache's official serializer in the retained comparison (4933 versus 4864 bytes), and stochastic KLL rank error must be compared over repeated trials rather than interpreted from one run.

### Rejected experiments

- Rejected lazy sorted-shadow invalidation/rebuild after a large merge regression.
- Rejected skipping structural planning for non-compacting resident merges after it regressed the focused merge ratio.
- Rejected peak-allocation pre-reservation after it regressed focused merge.
- Rejected a no-shadow/sort-in-place resident engine after substantial merge regression.
- Rejected reusable thread-local merge scratch storage after it regressed ingestion and merge throughput.

Only optimizations that preserved exact state semantics and passed the measured public-API gates were retained.

### Packaging / compatibility

- Public `KLL` / `KLLSketch` API identity is unchanged.
- `KLL2` serialization is unchanged and `KLL1` remains readable.
- Default wheel remains pure `py3-none-any`; native wheels remain explicit platform-local builds.
- Runtime wheels exclude native C++/include implementation sources and the native build helper; source distributions retain build sources.
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
