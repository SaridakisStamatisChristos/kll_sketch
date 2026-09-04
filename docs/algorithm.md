# Algorithm and invariants

## Representation

The sketch stores levels `L_0, L_1, ...`. An item in level `h` represents weight `2^h`.

The central invariant is therefore:

```text
n == sum((2 ** h) * len(L_h) for h in levels)
```

`validate()` checks this invariant together with finite retained values, retained-count accounting, extrema consistency and effective-`k` bounds.

## Level capacities

For `H` current levels, level `h` has depth:

```text
depth = H - h - 1
```

and nominal capacity:

```text
max(m, round(k * (2/3) ** depth))
```

with `m = 8`.

The implementation performs the rounding with exact integer arithmetic:

```text
round(k * 2^depth / 3^depth)
```

so the capacity policy is reproducible across platforms.

The top level has capacity `k`; capacities shrink geometrically toward level zero. Compaction is lazy: compression runs only when the total retained population exceeds the sum of current level capacities.

## Compaction

To compact level `h`:

1. Sort the level.
2. If its population is odd, use one pseudo-random bit to keep either the first or last item in the level.
3. The remaining compactable block is even.
4. Draw one parity bit for the **entire compaction**.
5. Promote either all even-position or all odd-position items to `h+1`.
6. Each promoted item doubles implicit weight because it moved up one level.

This is intentionally different from the old 1.x implementation, which independently randomized choices inside pairs and preserved both boundaries inside every compaction. Version 2 tracks global extrema separately instead.

## Randomness and reproducibility

KLL is a randomized sketch. Version 2 uses SplitMix64 as a small internal pseudo-random stream. The seed is user-configurable and the RNG state is serialized in KLL2, so replay is byte-stable for a fixed sequence of operations.

Seeded reproducibility is an engineering property; it does not turn the randomized error model into a deterministic worst-case bound.

## Exact extrema

`min_value` and `max_value` are maintained independently of retained levels. Consequently:

```text
quantile(0.0) == exact stream minimum
quantile(1.0) == exact stream maximum
```

Compaction is free to discard endpoint samples because exact extrema do not depend on compactor retention.

## Query view

Queries materialize retained `(value, weight)` pairs into one sorted view and cumulative-weight array. The result is cached against a mutation generation counter.

A repeated query without an intervening update or merge therefore avoids sorting and rebuilding the weighted representation.

## Merge semantics and min_k

`k` is the configured capacity of the destination sketch. It is not silently changed during merge.

`min_k` tracks the smallest effective `k` inherited from any **estimation-mode** sketch merged into the destination. Error reporting uses `min_k`, while future storage capacity still uses configured `k`.

This distinction avoids throwing away capacity merely because a small, still-exact sketch was merged into a larger destination.

## Weighted updates

For integer weight `w`, the binary representation of `w` determines which levels receive one copy of the value. For example:

```text
13 = 8 + 4 + 1
```

places the value in levels 3, 2 and 0, representing total mass 13 with three retained items before any necessary recompression.

This preserves exact total mass and matches the structural method used by modern weighted KLL implementations. It need not reproduce the exact compaction history of literally replaying an interleaved expanded stream.
