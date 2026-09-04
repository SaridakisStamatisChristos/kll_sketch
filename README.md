# KLL Streaming Quantile Sketch

A high-integrity, mergeable **KLL quantile sketch** for Python with deterministic seeded randomness, exact extrema, strict versioned serialization, rank-space accuracy characterization, and zero runtime dependencies.

Version **2.0** replaces the original KLL-inspired compactor with a KLL-style hierarchical engine aligned with established implementations: geometric level capacities, lazy compaction, one unbiased parity choice per compaction, explicit effective-`k` tracking across merges, and exact mass conservation.

## Why this implementation exists

Approximate quantiles are only useful when their semantics are explicit. This repository therefore treats the following as first-class engineering requirements:

- **normalized rank error**, not value-space error, is the primary accuracy metric;
- every retained item has exact implicit weight `2**level`;
- `n == sum(2**level * len(level))` is continuously testable;
- global minimum and maximum are exact even after compaction;
- merge history is represented by `min_k` without silently changing configured `k`;
- repeated queries reuse a cached sorted weighted view;
- corrupted serialized input fails closed with `SerializationError`;
- historical `KLL1` payloads remain readable while new sketches emit checksummed `KLL2`.

## Quick start

```python
from kll_sketch import KLL

sk = KLL(capacity=200, rng_seed=7331)
sk.extend([1, 5, 2, 9, 3, 6, 4, 8, 7])

print(sk.n)                    # 9
print(sk.median())             # 5.0
print(sk.quantile(0.9))
print(sk.quantiles_at([.1, .5, .9]))
print(sk.normalized_rank(5))
print(sk.min_value, sk.max_value)
```

### Merge

```python
a = KLL(200, rng_seed=1)
b = KLL(200, rng_seed=2)
a.extend(range(10_000))
b.extend(range(10_000, 20_000))
a.merge(b)

a.validate()
print(a.n, a.num_retained, a.min_k)
```

### Serialize

```python
blob = a.to_bytes()            # writes KLL2 + CRC32
restored = KLL.from_bytes(blob)
assert restored.to_bytes() == blob
```

`from_bytes()` also accepts historical `KLL1` snapshots produced by the 1.x series.

## Public API

| API | Meaning |
| --- | --- |
| `add(x, weight=1)` | Insert one value or a positive integer-weighted value |
| `extend(xs)` / `update_many(xs)` | Bulk ingestion |
| `quantile(q)` | Approximate quantile for `q in [0,1]` |
| `quantiles_at(qs)` | Batched quantiles using one cached query view |
| `quantiles(m)` | Interior cuts for `m` equal-mass buckets |
| `median()` | `quantile(0.5)` |
| `rank(x, inclusive=True)` | Approximate absolute rank in `[0,n]` |
| `ranks(xs, inclusive=True)` | Batched rank queries |
| `normalized_rank(x)` | Approximate rank divided by `n` |
| `cdf(xs)` / `pmf(cuts)` | Distribution queries |
| `merge(other)` | Merge another sketch into this one |
| `normalized_rank_error()` | Conventional empirical ~99% rank-error model |
| `quantile_lower_bound(q)` / `quantile_upper_bound(q)` | Error-model-derived quantile bounds |
| `to_bytes()` / `from_bytes()` | Strict versioned serialization |
| `validate()` | Structural invariant checker |
| `debug_state()` | JSON-friendly internal diagnostics |

`KLLSketch` remains available as an alias of `KLL` for compatibility.

## Accuracy model

KLL controls **rank error**. It does not promise a bounded numerical distance between the returned value and an exact quantile value.

This implementation exposes the conventional empirical model used by established KLL implementations:

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

These are characterization values, not per-instance proofs. The benchmark harness measures realized normalized rank error directly against exact ranks.

See [`docs/algorithm.md`](docs/algorithm.md) and [`docs/benchmarks.md`](docs/benchmarks.md).

## Weighted updates

Integer-weighted KLL updates are supported. A weight is decomposed into binary level contributions, the same structural idea used by modern weighted KLL implementations. Total represented mass remains exact.

For applications whose statistical assumptions require an ordinary unweighted stream, ingest observations individually. Weighted inputs preserve weight semantics but can produce a different compaction history than literally replaying an interleaved expanded stream.

## Performance architecture

The pure-Python implementation avoids several common hot-path costs:

- no `random.Random` object allocation per compaction;
- deterministic SplitMix64 RNG state lives inside the sketch;
- retained-item count is maintained incrementally;
- level capacities use deterministic integer arithmetic;
- repeated quantile/rank/CDF calls share a mutation-invalidated sorted cache;
- batched rank/CDF queries materialize the sketch only once.

There are **no runtime dependencies**.

## Benchmarking

Run a reproducible characterization sweep:

```bash
python benchmarks/bench_kll.py \
  --outdir bench_out \
  --Ns 1e5 \
  --capacities 100 200 400 800 \
  --distributions uniform normal exponential pareto bimodal duplicates \
  --trials 5

python benchmarks/validate_benchmarks.py bench_out
```

Artifacts include:

- `accuracy_rank.csv` — realized normalized rank errors;
- `update_throughput.csv` — updates/sec;
- `query_latency.csv` — steady-state cached query latency;
- `merge.csv` — merge timing and retained counts;
- `footprint.csv` — retained items, level count, serialized size.

Optional Apache DataSketches comparison:

```bash
python -m pip install '.[compare]'
python benchmarks/compare_datasketches.py --N 1000000 --k 200
```

The comparison is intentionally external and optional; `datasketches` is not a runtime dependency.

## Validation

```bash
python -m pip install -r kll_sketch/requirements-test.txt
python -m pytest -q kll_sketch/tests --cov=kll_sketch --cov-report=term-missing
```

The deterministic suite covers:

- exact mode;
- weight conservation;
- exact extrema;
- merge trees and mixed `k`;
- rank/CDF/PMF semantics;
- cache invalidation;
- adversarial sorted/reverse/duplicate/extreme streams;
- KLL2 corruption and checksum rejection;
- KLL1 backward reading;
- byte-stable KLL2 round trips;
- property-based invariant and rank-error tests when Hypothesis is installed.

## Packaging and offline builds

The source tree keeps a small in-tree PEP 517 backend with **no build-time third-party dependencies**. This allows source installation without contacting a package index:

```bash
python -m venv .venv
. .venv/bin/activate
PIP_NO_INDEX=1 python -m pip install --no-index .
```

Runtime remains pure Python and dependency-free. Distribution metadata is emitted as Core Metadata 2.4 so SPDX `License-Expression` and `License-File` fields are standards-correct.

## Compatibility notes for 2.0

- Python 3.10+ is supported.
- `to_bytes()` now emits `KLL2`; `KLL1` remains readable.
- configured `k` no longer gets overwritten merely because another sketch with smaller `k` is merged; `min_k` tracks inherited estimation quality separately.
- exact `q=0` and `q=1` answers come from separately tracked extrema.
- benchmark accuracy is now rank-based rather than absolute value error.

See [`docs/CHANGELOG.md`](docs/CHANGELOG.md).

## References

- Zohar Karnin, Kevin Lang, Edo Liberty, **Optimal Quantile Approximation in Streams**, FOCS 2016.
- Apache DataSketches KLL implementations and characterization methodology.

## License

Apache License 2.0.
