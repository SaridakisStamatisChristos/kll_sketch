# v3.2.0 final release-readiness audit

Audit target: **kll-sketch 3.2.0**

Immutable production/native baseline:
`6a762ad4f76f8267bf1e8a78d9191ca39dd992ab`.

This audit treats that baseline as the semantic engine to preserve. Final release work is
limited to release engineering, packaging metadata, benchmark tooling/evidence, CI,
documentation, security/contribution policy, and citation metadata. The KLL algorithm,
resident native state/compaction engine, RNG consumption, public KLL semantics, and KLL2
wire format are not redesigned by this release-hardening work.

## Audit result

| Area | Result | Evidence / release decision |
| --- | --- | --- |
| Core/native semantic preservation | PASS | PR #19 does not change `kll_sketch.py`, native C++ state/compaction sources, RNG logic, or KLL2 format. The only package-code-adjacent change is the sdist file list in `_build_backend.py`. |
| Cross-platform correctness | PASS on release-candidate lineage; final `main` rerun required before tag | Existing CI covers pure Python on Linux/macOS/Windows across Python 3.10–3.14 and native builds on representative versions. No release-hardening change weakens those jobs. |
| Deterministic Python/native parity | PASS | Existing parity suite remains authoritative; the new performance gate additionally requires byte-identical KLL2 state for ingestion, query fixtures, merge sources, and merge destinations. |
| Serialization / hostile input robustness | PASS | KLL2 remains checksummed, length-checked and structurally validated; bounded level counts, finite-value checks, represented-mass/retained-count invariants and historical KLL1 compatibility remain intact. |
| Performance-regression CI | PASS | Same-process native/reference benchmark uses deterministic fixtures, paired trials with alternating implementation order, exact state parity and conservative release floors of 50x ingestion / 5x repeated-query / 5x merge. On PR-head run 33989107659 it measured 122.0x / 16.7x / 12.4x respectively. |
| Apache DataSketches primary-peer comparison | PASS | Peer is pinned to `datasketches==5.2.0` with `numpy==2.5.2`; focused and matrix benchmarks use public APIs and shared input arrays/`k`, repeated paired measurements and explicit runner/workload scoping. |
| Multi-`k` / multi-`N` evidence | PASS | Matrix covers `N={50k,250k,1m}` and `k={100,200,400,800}`. The first durable release-candidate snapshot records ingestion wins in 9/12 cells and repeated batched-query wins in 12/12, while presenting size/error trade-offs rather than claiming universal superiority. |
| Sharded merge scaling | PASS | Scaling covers 2/4/8/16/32 shards at `N=250k,k=200`. The durable snapshot reports wins at 2/8/16/32 and an Apache win at 4 shards; README/release notes disclose the loss. |
| Durable benchmark artifacts | PASS | Versioned CSV evidence is retained under `benchmarks/results/v3.2.0/`; richer JSON/CSV artifacts are also uploaded by Actions with source SHA, runtime/peer metadata and run id. |
| README claim discipline | PASS | Claims are explicitly limited to measured runner/workloads, disclose Apache wins and serialized-size differences, and distinguish stochastic rank-error observations from guarantees. |
| Packaging / PyPI readiness | PASS, publication intentionally gated | Canonical pure `py3-none-any` wheel and sdist are built and smoke-tested. The sdist includes native build sources plus `CITATION.cff`, `CONTRIBUTING.md` and `SECURITY.md`. PyPI publication is release-only via Trusted Publishing/OIDC and stores no long-lived API token. |
| Release artifact integrity | PASS | Release workflow validates version/metadata/content, performs isolated wheel install + KLL2 round-trip smoke test, generates SHA-256 sums and uploads artifacts. PR-head run 33989107664 passed. |
| API/documentation polish | PASS | Stable API reference, algorithm/native/benchmark methodology, production-readiness criteria, release checklist and v3.2.0 release notes are present. |
| Security / edge-case audit | PASS | Existing malformed-payload, fallback, one-shot iterator, signed-zero, huge-rank and native-parity coverage is retained. `SECURITY.md` documents the in-process trust model and private vulnerability-reporting path. |
| Research citation metadata | PASS | `CITATION.cff` uses CFF 1.2.0, identifies software/version/repository/license/author and intentionally omits a DOI until a real archival DOI exists. |
| Zenodo readiness | PASS, archive creation external | `CITATION.cff` is sufficient repository metadata for Zenodo ingestion; the repository must be enabled in the maintainer's Zenodo GitHub integration before the GitHub Release if automatic archival is desired. |
| Tag/release preparation | PASS, tag not cut during audit | Package version is 3.2.0; tag workflow enforces `v3.2.0 == __version__`. The signed tag must point to the exact post-merge `main` commit after its direct CI/performance/matrix/artifact runs are green. |

## Benchmark claim approved for v3.2.0

The release may claim that, **on the documented GitHub-hosted Ubuntu/CPython workloads**,
kll-sketch 3.2 native was consistently faster for repeated batched quantile queries,
usually faster for ingestion, and faster at four of five measured shard counts in the
release matrix. Apache DataSketches won the 4-shard merge point. Serialized footprint
and observed stochastic rank error were competitive/mixed rather than uniformly better.

Do not shorten this to "faster than Apache DataSketches" without the workload scope and
trade-offs.

## Final pre-tag gate

After PR #19 is merged, use the exact resulting `main` SHA as the release candidate and
require all of the following to complete successfully on that SHA before creating the
tag:

1. `CI`;
2. `Performance Regression`;
3. `Apache KLL Benchmark Matrix`;
4. `Release Artifacts`.

Then create signed tag `v3.2.0` at that exact SHA. Tag-triggered artifact/matrix jobs must
also pass before publishing the GitHub Release.

## External account prerequisites (not repository defects)

- A signed Git tag requires the maintainer's signing key; the repository must not
  impersonate that signature.
- PyPI needs a Trusted Publisher mapping for owner `SaridakisStamatisChristos`, repository
  `kll_sketch`, workflow `publish-pypi.yml`, environment `pypi`; protect that GitHub
  environment before publication.
- Zenodo archival requires the maintainer to enable this repository in the relevant
  Zenodo GitHub account/integration. Add a DOI only after Zenodo actually creates one.

Until those account-level actions are deliberately performed, the repository is
**release-ready but not falsely represented as already published or archived**.
