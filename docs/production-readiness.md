# Production-readiness criteria

Version 3.2 treats "production ready" as an **evidence claim**. A release candidate is
ready only when the exact release commit satisfies the gates below; a historical green
run does not certify later commits.

## Semantic and compatibility gates

- unit, property, fallback, and malformed-input tests pass;
- branch-aware runtime coverage remains at least 90%;
- pure Python is tested on Linux/macOS/Windows across Python 3.10–3.14;
- native C++ is tested on all three operating systems on representative supported Python
  versions;
- native and Python execution retain exact seeded-state/KLL2 parity on covered paths;
- KLL2 remains checksummed/strict and historical KLL1 remains readable;
- signed zero, one-shot iterators, huge represented ranks, merge `min_k`, and source
  immutability remain covered.

## Packaging gates

- the default wheel is `py3-none-any` and contains no native/build/test sources;
- a native wheel can be built explicitly and contains only the compiled extension plus
  runtime package files;
- source distribution contains the C++ sources required for an explicit native build;
- pure wheel and sdist install/smoke tests run outside the source checkout;
- `pip --no-index .` works for the dependency-free pure source install;
- release artifacts receive SHA-256 checksums;
- tag `vX.Y.Z` must equal package `__version__ == X.Y.Z`.

## Performance and research evidence gates

- rank-space characterization passes its conservative accuracy checks;
- same-process native-vs-reference ingestion/query/merge regression ratios pass without
  sacrificing byte-state parity;
- the focused Apache DataSketches KLL benchmark remains reproducible with pinned peer
  dependencies;
- the release benchmark matrix spans multiple `k`, `N`, and shard counts and retains raw
  JSON/CSV artifacts;
- README/release claims are limited to measured workloads and include losses/trade-offs,
  including serialized-size differences.

## Open-source release gates

- README, API reference, benchmark methodology, changelog, security policy, contributing
  guide, release notes, and release checklist agree on the public contract;
- `CITATION.cff` validates as CFF 1.2 and carries the release version;
- the GitHub release is made from the exact validated tag;
- PyPI publication uses Trusted Publishing/OIDC and publishes the canonical pure wheel
  plus source distribution;
- Zenodo integration may archive the GitHub release after it is enabled for the repo;
  a DOI is added to citation metadata only after one actually exists.

## Authoritative workflows

- `.github/workflows/ci.yml` — cross-platform correctness/package baseline;
- `.github/workflows/performance-regression.yml` — same-process native performance gate;
- `.github/workflows/benchmark-matrix.yml` — Apache KLL matrix/scaling evidence;
- `.github/workflows/release-artifacts.yml` — exact release artifact verification;
- `.github/workflows/publish-pypi.yml` — OIDC-gated PyPI publication.
