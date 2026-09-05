# v3.2.0 release notes

Release target: `v3.2.0`

Production baseline for final release work:
`6a762ad4f76f8267bf1e8a78d9191ca39dd992ab`.

## What v3.2.0 is

Version 3.2.0 is the production release of the resident native KLL architecture. It
keeps the existing Python reference implementation, public `KLL`/`KLLSketch` identity,
and KLL2 wire format while accelerating compatible ingestion, query, and merge paths
through persistent C++17 state.

The native engine preserves the Python compaction/RNG semantics and is removable: pure
Python remains the canonical fallback.

## Performance evidence

The retained focused comparison against Apache DataSketches KLL 5.2.0 on a GitHub-hosted
Ubuntu 24.04 / CPython 3.13.15 runner used `N=250,000`, `k=200`, seven distributions and
eight merge shards. It measured:

| Metric | kll-sketch 3.2 native | Apache KLL 5.2.0 | Relative |
| --- | ---: | ---: | ---: |
| Geometric-mean ingestion | 30.81 M updates/s | 29.62 M updates/s | 1.040x |
| Repeated batched quantile query | 0.362 us | 0.541 us | 1.493x speed |
| Repeated 8-way merge | 43.92 us | 47.86 us | 1.090x speed |
| Serialized bytes | 4,933 | 4,864 | Apache ~1.4% smaller |

A separate 31-trial fresh-destination merge gate (128 destinations per trial,
alternating implementation order) measured a 32.61 us median versus 34.31 us for
Apache, with kll-sketch winning 30/31 paired trials and a 1.049x median speed ratio.

The broader release-candidate matrix swept `N={50k,250k,1m}` and
`k={100,200,400,800}` over uniform, normal and duplicate-heavy inputs. On that Ubuntu
24.04 / CPython 3.13.15 run:

- ingestion was faster in 9/12 cells; the three non-wins were near parity at
  0.983x-0.998x Apache;
- repeated batched quantile query was faster in all 12 cells, about 1.52x-1.58x;
- serialized footprint stayed within approximately -3.0% to +2.5% of Apache across the
  measured cells;
- rank-error observations were mixed and are not evidence of universal accuracy
  superiority by either implementation.

Sharded merge scaling at `N=250,000`, `k=200` measured:

| Shards | kll-sketch | Apache KLL | Relative speed |
| ---: | ---: | ---: | ---: |
| 2 | 7.89 us | 8.21 us | 1.041x |
| 4 | 20.77 us | **19.22 us** | **0.925x** |
| 8 | **42.31 us** | 50.10 us | **1.184x** |
| 16 | **77.65 us** | 85.28 us | **1.098x** |
| 32 | **142.54 us** | 155.42 us | **1.090x** |

These are runner/workload characterizations, not portable guarantees. The release claim
is deliberately bounded: query performance was consistently stronger on the measured
matrix, ingestion was usually stronger, merge won four of five measured shard counts,
and Apache won the 4-shard merge point. The matrix workflow preserves JSON/CSV artifacts
with peer/runtime metadata and the source commit.

## Release engineering

- same-process native-vs-reference performance regression CI with exact serialized-state
  parity checks;
- multi-`N`/multi-`k` Apache DataSketches KLL and sharded-merge characterization;
- pre-tag universal wheel/sdist content, install and SHA-256 artifact validation;
- source distribution includes `CITATION.cff`, `CONTRIBUTING.md`, and `SECURITY.md`;
- PyPI publication is prepared for Trusted Publishing/OIDC and is triggered only by a
  published GitHub Release using its exact tag;
- Zenodo-ready citation metadata is present without inventing a DOI before an archive
  exists.

## Compatibility

- Python 3.10+.
- `KLLSketch is KLL`.
- KLL2 serialization unchanged; KLL1 remains readable.
- deterministic seeded Python/native state parity preserved.
- pure `py3-none-any` wheel remains the default publication artifact.
- native compilation remains explicit.

## Release gate

Before creating the `v3.2.0` tag:

1. all cross-platform CI jobs must be green for the exact release commit;
2. `Performance Regression` must pass its same-process native/reference gates;
3. `Apache KLL Benchmark Matrix` must complete and its artifact must be retained;
4. `Release Artifacts` must build and smoke-test the universal wheel and sdist;
5. the tag must point to the exact validated commit and match package version `3.2.0`.

No benchmark headline should be broadened beyond the measured environment and workload.
