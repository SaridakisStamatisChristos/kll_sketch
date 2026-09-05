# API reference

The stable public surface is exported from `kll_sketch`. `KLLSketch` is a direct alias
of `KLL`.

## Construction

```python
KLL(capacity: int = 200, rng_seed: int = 0xA5B357)
```

`capacity` is the KLL accuracy/memory parameter conventionally called `k` and must be at
least 40. `rng_seed` selects the deterministic SplitMix64 compaction stream.

## Ingestion

- `add(x, weight=1.0) -> None` — ingest one finite real value. Positive integer weights
  are supported by the compatibility weighted-update path.
- `extend(xs) -> None` — ingest an iterable.
- `update_many(xs) -> None` — alias of `extend`.

For the ordinary unweighted KLL statistical model, ingest observations individually.
Weighted binary placement preserves exact represented mass but can have a different
compaction history from replaying an expanded interleaved stream.

## Quantiles and ranks

- `quantile(q) -> float` — estimate one quantile for `q` in `[0, 1]`.
- `quantiles_at(probabilities) -> list[float]` — batched arbitrary probabilities.
- `quantiles(m) -> list[float]` — equal-mass interior cuts. Historical `m == 1`
  behavior returns `[median()]`.
- `median() -> float`.
- `rank(x, *, inclusive=True) -> float` — approximate absolute represented rank.
- `ranks(xs, *, inclusive=True) -> list[float]`.
- `normalized_rank(x, *, inclusive=True) -> float`.
- `cdf(xs, *, inclusive=True) -> list[float]`.
- `pmf(split_points, *, inclusive=True) -> list[float]`; split points must be strictly
  increasing.

Empty-sketch quantile/min/max operations raise `ValueError`; empty rank/CDF operations
return zero-valued results according to the existing API.

## Error characterization

- `normalized_rank_error(*, pmf=False) -> float`.
- `quantile_lower_bound(q) -> float`.
- `quantile_upper_bound(q) -> float`.

The exposed model is an engineering characterization of normalized rank error, not a
per-instance deterministic guarantee.

## Merge

```python
destination.merge(source)
```

`source` must be another `KLL` and cannot be `destination` itself. The destination keeps
its configured `k`. When estimation-mode inputs with lower effective quality are merged,
`min_k` records the tighter inherited error parameter.

## Serialization

- `to_bytes() -> bytes` emits checksummed `KLL2`.
- `KLL.from_bytes(data) -> KLL` reads `KLL2` and historical `KLL1`.
- `serialization_version` returns `2`.
- malformed or inconsistent payloads raise `SerializationError`.

Version 3.2 does not introduce a new wire format.

## State and diagnostics

Read-only properties:

- `k`, `min_k`, `n`, `num_retained`, `is_estimation_mode`;
- `min_value`, `max_value`;
- `serialization_version`.

Methods:

- `size() -> int`;
- `copy() -> KLL`;
- `validate() -> None`;
- `debug_state() -> dict`.

## Native backend control

The package exports:

- `native_available()`;
- `native_enabled()`;
- `native_backend_info()`;
- `set_native_enabled(bool)`.

The extension is optional. Disabling or removing it changes performance, not the public
semantic contract. `KLL_SKETCH_DISABLE_NATIVE=1` forces the Python path at import/runtime
dispatch boundaries.
