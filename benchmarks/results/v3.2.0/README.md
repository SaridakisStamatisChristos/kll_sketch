# v3.2.0 Apache KLL release-candidate evidence

This directory is a durable compact snapshot of the first full release-candidate matrix
run. The richer per-distribution JSON remains available in the GitHub Actions artifact
for the recorded run; these CSV files preserve the complete top-level matrix and merge
scaling results in the repository itself.

- production native/core baseline: `6a762ad4f76f8267bf1e8a78d9191ca39dd992ab`
- release-candidate head measured: `d3c21d917f3441f3e67ac540fe2f57255d4bb478`
- GitHub Actions run: `33988540273` (`Apache KLL Benchmark Matrix`, run 1)
- GitHub PR synthetic merge SHA used by that first run: `ae3b51b98722b56f098e1a5f041363616709cbc7`
- runner: Ubuntu 24.04, CPython 3.13.15, x86_64, GCC native backend with runtime AVX2
- peer: Apache DataSketches `5.2.0`
- NumPy: `2.5.2`
- seed: `7331`
- matrix distributions: uniform, normal, duplicate-heavy
- matrix trials: 3 paired trials per cell
- query loops: 1000
- merge loops: 96

The subsequent release-engineering commits do not redesign the KLL/native engine. The
benchmark harness was later amended only to record the actual PR head SHA directly in
future artifacts.

Interpretation is deliberately limited to this runner/workload. In this snapshot,
kll-sketch ingestion won 9/12 `(N,k)` cells, repeated batched query won 12/12 cells,
and merge won at 2/8/16/32 shards while Apache won at 4 shards. Rank-error observations
were mixed and are not evidence of universal accuracy superiority by either sketch.

Files:

- `matrix.csv` — all 12 `N × k` top-level comparison cells;
- `merge_scaling.csv` — all five sharded merge points.

The original Actions artifact was uploaded with digest
`sha256:47098b95f2a3d7ceebf586b34a76e02d6caf8cdccee8010b793f1aa60b57b4a3`.
