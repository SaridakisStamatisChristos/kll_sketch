# Native acceleration architecture

KLL v3.2 uses an optional **CPython C API / C++17** acceleration layer while keeping the Python implementation as the portable reference and semantic fallback.

## Compatibility contract

The extension is an optimization layer, not a different sketch algorithm.

1. `KLL` remains the same Python class in pure and native modes.
2. KLL2 serialization and KLL1 read compatibility are unchanged.
3. Native compaction consumes the same SplitMix64 bits in the same order as the Python reference.
4. Equal-value ordering, including `+0.0` / `-0.0`, is preserved where it affects canonical state.
5. Merge fast paths preflight before mutation; unsupported public semantics route through the canonical Python dispatcher.
6. Only replay-safe indexed/sized inputs are eligible for native bulk probing; generic one-shot iterators remain safe.
7. Users can force Python with `KLL_SKETCH_DISABLE_NATIVE=1` or `set_native_enabled(False)`.
8. Quantile lookup falls back to Python when represented mass exceeds `2**53`, preserving exact integer-rank comparison semantics.

## Resident native state

A compatible `KLL` can keep an opaque C++ `SketchState` resident across calls instead of translating Python levels to C++ and back for every operation. Resident state owns:

- retained KLL levels;
- sorted level shadows used by compaction;
- deterministic level capacities;
- `n`, retained count, SplitMix64 state, and compaction count;
- exact minimum and maximum;
- lazily cached `min_k` merge-quality metadata;
- a mutation-invalidated weighted query view.

The Python object remains the public identity. Canonical Python levels are materialized again only at synchronization boundaries such as serialization, validation, copying, Python-only fallback, or disabling native dispatch.

## Native ingestion

Compatible sized sequences and contiguous native-`double` buffers enter the resident C++ engine directly. The engine follows the same capacity geometry and lazy-compaction state machine as Python.

For one-dimensional C-contiguous PEP 3118 `double` (`d`) buffers on GCC/Clang x86 builds, finite-value validation and batch-extrema scanning use runtime-dispatched AVX2 when available. The extension is not globally compiled with `-mavx2`; unsupported CPUs retain the scalar path. Zero-containing batches preserve Python signed-zero extrema semantics through the tie-preserving path. Non-x86 and current MSVC builds use scalar scanning while retaining the rest of the C++ engine.

## Direct query dispatch

`KLL.quantiles_at` is replaced at installation time with a CPython method descriptor when the native backend supports resident type fast paths. When state is resident, the call enters C++ directly and reuses the native weighted query view.

The descriptor falls back to the Python runtime dispatcher for cold/nonresident state, disabled native mode, unsupported inputs, and represented mass above `2**53`. Empty probability collections, invalid probabilities, and empty-sketch behavior remain API-compatible.

## Resident merge engine

Version 3.2 specializes both the first merge and subsequent resident-to-resident merges.

### Empty-destination adoption

For ordinary positional `merge(other)` when the destination is fresh/empty, the source is resident/non-empty, and configured `k` matches, the destination can adopt a copy of the source's already-valid hierarchy without bootstrapping an empty resident state and running the general append planner.

The source levels/capacity metadata are copied, but the destination **retains its own RNG state and compaction count**. This matches Python semantics for later compactions. `min_k`, extrema, represented mass, and source immutability remain preserved.

### Structural preflight

For resident-to-resident merge, v3.2 computes the post-append structural evolution before mutation. The preflight determines:

- required level count and capacity geometry;
- total retained count after each structural compaction;
- final compaction count;
- exactly which levels will compact;
- the exact compact-level execution sequence for normal-sized plans.

Compaction sizes depend on level cardinality, not on the random parity/value choice, so the sequence can be proven without consuming RNG bits.

### Raw-write elision

If preflight proves that a higher level will immediately compact, the merge need not materialize source values into that level's canonical raw vector only to destroy them moments later. The sorted shadow remains authoritative transiently; compaction restores raw/sorted parity at the boundary leftover.

### Exact-sequence execution

After preflight, normal resident merges execute the recorded compact-level sequence directly rather than rescanning every level after each compaction to rediscover the same next compactable level. Deep/pathological plans that exceed the fixed fast-plan capacity fall back before mutation to the established v3.2 path.

The executor still consumes SplitMix64 bits only inside the canonical compactor, so deterministic state and byte-level serialization remain unchanged.

### Cached Python slot keys

The hot native descriptor touches the same slotted Python fields on every merge: resident handles, `n`, retained count, extrema, RNG/compaction metadata during adoption, and `min_k` when needed. v3.2 interns those slot-name `PyUnicode` objects once when the fast path installs and uses `PyObject_GetAttr` / `PyObject_SetAttr` with the cached keys instead of repeatedly resolving C-string attribute names.

This layer changes no KLL mathematics. It reduces CPython framing cost and is especially material in fresh-destination and short merge sequences.

### `min_k` synchronization

Resident state lazily caches `min_k`. Same-quality hot merges remain native without rereading Python metadata. If a source can tighten the bound, the destination's authoritative Python value is rechecked first so deliberate keyword/fallback merges cannot leave stale resident metadata.

### Fallback boundaries

Self-merge, keyword variants, disabled-native calls, incompatible/nonresident inputs, mixed semantics, and rejected preflight cases retain the canonical runtime fallback. Once mutation begins, impossible internal failures are surfaced rather than replayed over partially mutated state.

## Synchronization boundaries

While resident, C++ state is authoritative for compaction/RNG details. Python-visible fields needed by immediate public properties are mirrored after successful operations. Before a Python-only operation requires canonical levels, runtime synchronization exports the resident hierarchy and complete statistics back into the Python object.

Regression coverage includes:

- resident ingestion and later compaction;
- resident-to-resident merge;
- empty-destination adoption with distinct source/destination seeds;
- continued ingestion after merge;
- `min_k` inheritance and fallback synchronization;
- source immutability;
- signed-zero stability;
- disabling native immediately after merge;
- byte-identical serialization against pure Python.

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

Default wheels are `py3-none-any`. Explicit native wheels are CPython/platform tagged and contain the compiled extension but intentionally exclude `_native*.cpp`, `_native*.inc`, and `_native_build.py`. Source distributions retain the implementation/build sources.

## Differential validation

Native CI compares resident and pure execution using identical seeds and input streams. Coverage includes full `to_bytes()` equality, `debug_state()`, quantiles, ranks, CDF/PMF behavior, signed zero, invalid-batch replay, one-shot iterator safety, enormous-rank fallback, class identity, copy behavior, resident merge parity, native-disable synchronization, and packaging hygiene.

`benchmarks/bench_native.py` requires serialized-state parity before reporting speed.

`benchmarks/competitive_kll_focus.py` compares public KLL APIs against Apache DataSketches KLL on the same process/runner. `benchmarks/competitive_kll_cold_merge.py` isolates fresh-destination eight-way merge with 31 paired trials, 128 fresh destinations per trial, and alternating implementation order. `benchmarks/competitive_quantiles.py` provides the broader ecosystem comparison.

## v3.2 performance snapshot

On a retained GitHub-hosted Ubuntu 24.04 / CPython 3.13.15 focused gate (`N=250000`, `k=200`, seven distributions, eight shards), the native engine measured:

- 30.81M updates/s versus 29.62M/s for Apache KLL;
- 0.362 us versus 0.541 us for the repeated batched quantile set;
- 43.92 us versus 47.86 us for repeated eight-way merge.

The robust fresh-destination merge gate measured 32.61 us median for `kll-sketch` versus 34.31 us for Apache, with `kll-sketch` winning 30 of 31 paired trials.

These are runner/workload observations, not portable performance guarantees. Serialized size and stochastic rank-error distributions remain separate dimensions of comparison.

## Scope boundaries

Native acceleration does not change:

- the KLL error model;
- weighted-update semantics;
- merge `min_k` semantics;
- KLL1/KLL2 formats;
- the canonical pure-Python implementation;
- deterministic seeded behavior for equivalent canonical execution.

Disabling or deleting the extension reduces performance, not capability.
