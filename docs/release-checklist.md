# v3.2.0 release checklist

This checklist is intentionally commit-specific. Release only from a commit whose CI and
benchmark evidence can be traced directly to the `v3.2.0` tag.

Production baseline entering final release work:
`6a762ad4f76f8267bf1e8a78d9191ca39dd992ab`.

## 1. Start clean and verify identity

```bash
git fetch --tags --prune
git checkout <release-commit>
git status --short
python - <<'PY'
from kll_sketch import __version__
assert __version__ == "3.2.0"
print(__version__)
PY
```

The final release commit must descend from the production baseline and must not contain a
core-engine semantic redesign.

## 2. Run correctness and coverage

```bash
python -m pip install -r kll_sketch/requirements-test.txt
KLL_SKETCH_DISABLE_NATIVE=1 python -m pytest -q kll_sketch/tests \
  --cov=kll_sketch --cov-report=term-missing
python -m kll_sketch._native_build
python -m pytest -q kll_sketch/tests
```

Runtime branch-aware coverage must remain at least 90%.

## 3. Run rank-space and native performance gates

```bash
KLL_SKETCH_DISABLE_NATIVE=1 python benchmarks/bench_kll.py \
  --outdir bench_out --Ns 1e5 --capacities 100 200 400 800 --trials 5
python benchmarks/validate_benchmarks.py bench_out
python benchmarks/performance_regression.py
```

Do not replace parity checks with timing checks: native state must remain byte-identical
to the Python reference on the covered fixtures.

## 4. Reproduce Apache DataSketches evidence

```bash
python -m pip install numpy==2.5.2 datasketches==5.2.0
python benchmarks/competitive_kll_focus.py \
  --N 250000 --k 200 --seed 7331 --trials 5 \
  --query-loops 2000 --shards 8 --merge-loops 200
python benchmarks/competitive_kll_cold_merge.py
python benchmarks/competitive_kll_matrix.py \
  --Ns 50000 250000 1000000 --ks 100 200 400 800 \
  --shards 2 4 8 16 32 --merge-N 250000 --merge-k 200 \
  --distributions uniform normal duplicates --trials 3 \
  --query-loops 1000 --merge-loops 96 --outdir benchmark_matrix
```

Preserve JSON/CSV outputs. Claims in README/release notes must match measured artifacts
and remain explicitly runner/workload scoped.

## 5. Build canonical release artifacts

The default publication artifacts are the pure universal wheel and source distribution:

```bash
rm -rf dist && mkdir dist
python - <<'PY'
from kll_sketch._build_backend import build_sdist, build_wheel
print(build_wheel("dist"))
print(build_sdist("dist"))
PY
```

`.github/workflows/release-artifacts.yml` performs the authoritative artifact-content,
metadata, install, smoke, and SHA-256 checks. Native wheels are explicit platform-local
build products; they are not silently substituted for the canonical pure PyPI wheel.

## 6. PyPI Trusted Publisher prerequisite

Before publishing the GitHub Release, configure a PyPI Trusted Publisher for:

- owner: `SaridakisStamatisChristos`;
- repository: `kll_sketch`;
- workflow: `publish-pypi.yml`;
- environment: `pypi`.

Protect the GitHub `pypi` environment. The workflow requests `id-token: write` only in
the publish job and stores no long-lived PyPI API token.

If the project name does not yet exist on PyPI, configure a pending Trusted Publisher in
PyPI first. Publication itself is intentionally not attempted during ordinary CI.

## 7. Citation / Zenodo prerequisite

`CITATION.cff` is the canonical in-repository citation metadata. Enable this repository
in Zenodo's GitHub integration **before** creating the GitHub Release if automatic archive
creation is desired. Do not invent or pre-fill a DOI; add the Zenodo DOI only after the
archive exists.

## 8. Tag and GitHub Release

Only after every required workflow is green for the exact commit:

```bash
git tag -s v3.2.0 <release-commit> -m "kll-sketch v3.2.0"
git push origin v3.2.0
```

Create the GitHub Release from tag `v3.2.0` using
`docs/release-notes-v3.2.0.md`. The tag push re-runs release artifact and benchmark-matrix
workflows; publishing the GitHub Release triggers the OIDC PyPI workflow.

## 9. Post-release verification

- confirm GitHub Release points to the intended tag/commit;
- confirm wheel/sdist hashes and installed `__version__`;
- confirm the PyPI project exposes `3.2.0` and the universal wheel/sdist;
- install from PyPI into a clean environment and run a KLL2 round-trip smoke test;
- confirm Zenodo archive/DOI if integration was enabled;
- if a DOI exists, update citation metadata in the next documented metadata commit rather
  than rewriting the released tag.
