# Native acceleration architecture

KLL v3.2 uses an optional **CPython C API / C++17** acceleration layer while keeping the Python implementation as the portable reference and fallback.

## Compatibility contract

The extension is an optimization layer, not a different sketch algorithm.

1. `KLL` remains the same Python class in pure and native modes.
2. KLL2 serialization and KLL1 read compatibility are unchanged.
3. Native compaction consumes the same SplitMix64 bits in the same order as the Python reference.
4. Equal-value ordering, including `+0.0` / `-0.0`, is preserved where it affects canonical state.
5. Failed native operations do not silently commit partial native state; merge paths retain explicit preflight/fallback semantics and unsupported/rejected operations fall back through the canonical Python path.
6. Only replay-safe indexed/sized inputs are eligible for native bulk probing; generic one-shot iterators stay on the incremental Python path.
7. Users can force Python with `KLL_SKETCH_DISABLE_NATIVE=1` or `set_native_enabled(False)`.
8. Quantile lookup falls back to Python when `n > 2**53`, retaining Python's exact int/float rank-comparison semantics at enormous cumulative weights.

## Resident native state

Version 3.1 introduced persistent native execution. Version 3.2 keeps that model and specializes the first resident merge into an empty destination.

A compatible `KLL` can keep an opaque `SketchState` capsule resident across calls instead of converting Python levels to C++ and back for every operation. The resident state owns:

- retained KLL levels;
- sorted level shadows used by compaction;
- level capacities;
- `n`, retained count, SplitMix64 state and compaction count;
- exact minimum and maximum;
- a mutation-invalidated native weighted query view.

The Python object remains the public identity and stores the resident handle through its existing internal cache fields. Python levels are materialized again only at synchronization boundaries such as serialization, validation, copying, Python-only fallback, or disabling native dispatch.

This arrangement avoids exposing a second public sketch type and keeps the Python implementation useful as an executable specification.

## Accelerated paths

The extension provides both legacy stateless helpers and the resident engine. The hot paths include:

- **bulk ingestion** — validates a safe sequence/buffer and evolves level 0 plus all required compactions in C++;
- **empty-destination resident merge adoption** — when a fresh destination and resident source have the same configured `k`, v3.2 copies the already-valid source hierarchy directly into a new destination resident state instead of bootstrapping an empty state and running the general merge planner;
- **non-empty resident merge** — retains the v3.1 structural preflight and exact native compaction path;
- **resident quantiles** — uses the native weighted query view;
- **resident ranks** — binary-searches the native query view;
- **direct `quantiles_at` descriptor** — once native state is resident, the hot public batched-quantile call enters the extension directly, with the Python dispatcher retained as fallback;
- **native level compaction/materialization helpers** — retained for fallback and compatibility paths.

### Empty-destination adoption semantics

The v3.2 shortcut is deliberately narrow. It is used only when:

- native dispatch is enabled;
- the call is the ordinary positional `merge(other)` hot path;
- destination is a fresh empty, nonresident `KLL`;
- source is already resident and non-empty;
- source and destination have the same configured `k`.

The source hierarchy is already compressed under the same capacity geometry, so copying it into an empty destination cannot itself require compaction. Crucially, v3.2 copies the source levels/capacity metadata but **does not copy the source RNG state or source compaction count**. The destination retains its own SplitMix64 state and compaction counter exactly as the Python reference merge does. `min_k` inheritance still runs through the public merge semantics.

A dedicated regression test builds source and destination with different seeds, performs the empty merge, then ingests enough additional data to force later compactions. Final native serialization must be byte-identical to the pure-Python reference, and the source sketch must remain unchanged.

All non-eligible merges tail-call or continue through the v3.1 semantics.

## SIMD policy

For one-dimensional C-contiguous buffers with native PEP 3118 `double` format (`d`), the resident ingestion path reads the native buffer directly.

On GCC/Clang x86 builds:

- a function-specific AVX2 scanner validates finiteness four doubles at a time;
- the same scan reduces batch extrema;
- AVX2 is selected only after runtime CPU feature detection;
- the extension is **not** globally compiled with `-mavx2`;
- zero-containing batches use the tie-preserving scalar extrema pass required to reproduce Python signed-zero behavior.

If AVX2 is unavailable, or on non-x86 / current MSVC builds, the scan is scalar. The rest of the resident KLL engine still executes in C++.

`native_backend_info()` reports `simd='avx2-runtime'` or `simd='scalar'`, compiler information, API version, and whether persistent state is available.

## Merge transaction model

Non-empty resident merge avoids cloning the entire destination sketch before every successful merge. Its v3.1 structural preflight proves compaction sizes and capacity evolution before mutation, then runs the ordinary exact compactor. Unsupported or replay-safe preflight failures can use the canonical fallback path; impossible post-mutation internal failures are surfaced rather than replayed.

The v3.2 empty-destination adoption path has no destructive destination payload to roll back: it allocates and populates a new native state first, then installs that resident state only after construction succeeds.

Several alternatives were benchmarked and rejected rather than accumulated:

- lazy invalidation/rebuild of sorted shadows regressed focused merge to roughly 152 us versus roughly 49 us for Apache;
- skipping the structural planner for non-compacting resident merges regressed the focused ratio;
- pre-reserving predicted compaction peaks also regressed the focused merge ratio;
- a `min_k` cache experiment from v3.1 was likewise rejected after a measured regression.

The release therefore retains only the optimization that produced a repeatable benefit: direct empty-destination adoption.

## Query dispatch

`KLL.quantiles_at` has a native method descriptor when the compiled backend exposes the resident type fastpath. The descriptor uses resident native state directly when possible and otherwise tail-calls the Python runtime dispatcher.

The direct path preserves important fallbacks:

- empty probability collections remain valid;
- empty sketches still raise on non-empty quantile requests;
- generic iterables are materialized once;
- invalid probabilities fail with the public error behavior;
- sketches above `2**53` represented mass use the exact Python comparison path;
- `set_native_enabled(False)` disables the direct descriptor as well as normal native dispatch.

## Synchronization boundaries

While resident, native state is authoritative for internal compaction/RNG details. Python-visible fields required by immediate public properties are mirrored after successful operations. Before a Python-only operation needs canonical levels, `_sync_state()` exports the resident levels and complete statistics back into the Python object.

Regression coverage verifies:

1. resident merge;
2. continued resident ingestion after the merge;
3. byte-identical serialization against the pure-Python reference;
4. disabling native immediately after a resident merge and synchronizing cleanly;
5. v3.2 empty-destination adoption with distinct source/destination seeds followed by later compaction.

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

No pybind11, Cython, setuptools, Meson, scikit-build, or NumPy is required by the package or native build driver.

Native wheels are tagged for the active CPython/platform. The default pure wheel remains the portable distribution.

## Differential validation

Native CI compares resident and pure execution using identical seeds and input streams. Coverage includes:

- `debug_state()` and full `to_bytes()` equality;
- quantiles, ranks, CDF and PMF behavior;
- list/range ingestion;
- contiguous `array('d')` ingestion;
- signed-zero stability;
- invalid-batch replay semantics;
- one-shot iterator safety;
- `n > 2**53` exact-query fallback;
- class identity and copy behavior;
- resident merge exact-state parity;
- continued native ingestion after merge;
- synchronization after native is disabled;
- empty-destination merge adoption followed by later compaction with different source/destination seeds.

`benchmarks/bench_native.py` requires byte-identical serialized state before reporting native speed. `benchmarks/competitive_kll_focus.py` compares public APIs against Apache DataSketches KLL on the same process/runner. Competitive timings are observations of the measured environment, not portable performance guarantees.

## v3.2 benchmark interpretation

On the retained v3.2 candidate gate (Ubuntu 24.04, CPython 3.13, `N=250000`, `k=200`, eight shards), the focused benchmark measured 50.19 us for the repeated eight-way `kll-sketch` merge versus 50.49 us for Apache KLL, while preserving the existing ingestion/query advantage on that runner.

A separate broad two-trial cold-destination run measured 86.73 us for `kll-sketch` versus 66.45 us for Apache. The latter is intentionally kept visible: direct adoption materially reduces first-merge bootstrap work, but Apache still has an advantage for the measured cold one-shot sequence.

## Scope boundaries

Native acceleration does not change:

- the KLL error model;
- weighted-update semantics;
- merge `min_k` semantics;
- KLL1/KLL2 formats;
- the canonical pure-Python implementation;
- deterministic seeded behavior for equivalent canonical execution.

Disabling or deleting the extension reduces performance, not capability.
