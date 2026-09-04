# Release checklist

The release process is designed so the exact source commit can be validated and packaged without build-time third-party dependencies.

## 1. Start clean

```bash
git fetch --tags --prune
git checkout <release-commit-or-tag>
python -m venv .venv
. .venv/bin/activate
```

## 2. Run validation

```bash
python -m pip install -r kll_sketch/requirements-test.txt
python -m pytest -q kll_sketch/tests --cov=kll_sketch --cov-report=term-missing
python benchmarks/bench_kll.py --outdir bench_out --Ns 1e5 --capacities 100 200 400 800 --trials 5
python benchmarks/validate_benchmarks.py bench_out
```

The runtime coverage gate is 90%.

## 3. Build wheel and sdist with the in-tree backend

```bash
rm -rf dist
mkdir dist
python - <<'PY'
from kll_sketch._build_backend import build_sdist, build_wheel
print(build_wheel("dist"))
print(build_sdist("dist"))
PY
```

## 4. Inspect and install artifacts

Check that the wheel contains only runtime package files plus dist-info metadata, not tests or build-backend internals.

```bash
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile
wheel = next(Path("dist").glob("*.whl"))
with ZipFile(wheel) as zf:
    names = zf.namelist()
    assert "kll_sketch/kll_sketch.py" in names
    assert "kll_sketch/__init__.py" in names
    assert not any("/tests/" in n for n in names)
    metadata = zf.read(next(n for n in names if n.endswith("/METADATA"))).decode()
    assert "Metadata-Version: 2.4" in metadata
    assert "License-Expression: Apache-2.0" in metadata
PY
```

Then install the wheel into a clean environment and run a round-trip smoke test.

## 5. Verify offline source installation

```bash
python -m venv .venv-offline
PIP_NO_INDEX=1 .venv-offline/bin/pip install --no-index .
```

This is a separate gate from wheel installation.

## 6. Capture hashes

```bash
python - <<'PY'
from pathlib import Path
import hashlib
with open("dist/SHA256SUMS", "w", encoding="utf-8") as out:
    for artifact in sorted(Path("dist").glob("kll_sketch-*")):
        out.write(f"{hashlib.sha256(artifact.read_bytes()).hexdigest()}  {artifact.name}\n")
PY
```

Optionally sign `SHA256SUMS` with the project's release-signing process.

## 7. Serialization compatibility gate

Before release:

- load committed/archived KLL1 fixtures with the v2 reader;
- verify KLL2 round trips byte-for-byte;
- verify corruption tests reject modified payloads;
- document that v1 readers cannot read KLL2 if a downgrade is planned.

## 8. Tag and publish

Create the signed release tag only after CI is green for the exact source commit and artifacts have been verified.
