# Operational guide

This library is an in-process sketch, not a service. Operational concerns therefore center on input rate, retained footprint, merge topology, query latency, serialization compatibility, and accuracy configuration.

## Metrics worth exposing

For long-lived sketches, record at least:

- input mass `n`;
- configured `k` and effective `min_k`;
- `num_retained`;
- number of levels and per-level sizes from `debug_state()`;
- compaction count;
- update throughput;
- merge latency;
- query p50/p95/p99;
- serialized byte size;
- `SerializationError` count on ingesting persisted sketches.

Do not log retained values when they may contain sensitive data. Structural diagnostics are usually sufficient.

## Choosing k

Use `normalized_rank_error()` as the engineering starting point rather than the old rule of thumb `epsilon ~= 1/k`.

Representative single-sided characterization values are approximately:

| k | normalized rank error |
|---:|---:|
| 100 | 2.61% |
| 200 | 1.33% |
| 400 | 0.68% |
| 800 | 0.35% |

Measure your actual distributions and merge topology with `benchmarks/bench_kll.py` before setting production SLOs.

## Merge topology

Balanced merge trees usually provide a more predictable operational profile than repeatedly merging every shard into one hot accumulator.

Monitor `min_k`: it records lower-`k` estimation history inherited from merged sketches. A destination can have configured `k=400` while `min_k=100`; storage continues using 400, but the inherited error model must remain conservative at 100.

Avoid treating a larger destination `k` as a way to recover information already discarded by a lower-`k` estimation-mode source.

## Serialization rollout

Version 2 **reads KLL1 and KLL2 but writes KLL2**.

This has an important deployment consequence:

- upgrade readers before writers if persisted blobs cross process/version boundaries;
- once a v2 writer emits KLL2, a v1 reader cannot consume that blob;
- rollback from v2 to v1 therefore requires either suppressing KLL2 writes during the canary or keeping v2 readers available for persisted state.

Never assume bidirectional wire compatibility merely because the new version can read old snapshots.

## Corruption handling

Treat `SerializationError` as a data-integrity event. Preserve the failing blob and surrounding metadata for forensic analysis, but do not repeatedly retry the same corrupt payload.

KLL2 validates length, checksum, flags, level bounds, retained mass, extrema, and trailing bytes before accepting state.

## Upgrade playbook

1. Run the exact release commit through CI.
2. Run `benchmarks/bench_kll.py` on representative production-like data.
3. Verify KLL1 fixtures from the currently deployed version load successfully in v2.
4. Canary v2 readers before enabling KLL2 persistence.
5. Observe update/query latency, `num_retained`, serialized bytes, `min_k`, and error-model values.
6. Enable v2 writers only after all consumers are KLL2-capable.
7. Keep raw benchmark artifacts and release hashes with the deployment record.

## Incident diagnostics

`debug_state()` is intentionally JSON-friendly. Capture it alongside application version, git SHA, workload identity, and serialized blob size.

For reproducibility, also record the configured seed when deterministic replay matters. KLL2 stores RNG state, so a restored sketch continues from the same pseudo-random stream.
