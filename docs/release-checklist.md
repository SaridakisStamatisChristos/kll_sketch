# Signed Release Checklist

This playbook produces reproducible artifacts that can be verified offline.

1. **Start from a clean clone**
   - `git fetch --tags --prune`
   - `git checkout vX.Y.Z` (or the commit to be released)
   - `python -m venv .venv && source .venv/bin/activate`
   - `python -m pip install --upgrade pip`
2. **Build artifacts with the in-tree backend**
   - `python -m pip install build==1.2.1` *(only needed on the build host)*
   - `python -m build --wheel --sdist`
3. **Capture hashes**
   - `cd dist`
   - `python - <<'PY'`
     ```python
     from pathlib import Path
     import hashlib

     with open("CHECKSUMS.txt", "w", encoding="utf-8") as out:
         for artifact in sorted(Path(".").glob("kll_sketch-*")):
             data = artifact.read_bytes()
             sha256 = hashlib.sha256(data).hexdigest()
             out.write(f"{sha256}  {artifact.name}\n")
     ```
     PY
4. **Sign the checksum file**
   - `gpg --armor --detach-sign CHECKSUMS.txt`
   - Publish `CHECKSUMS.txt` and `CHECKSUMS.txt.asc` alongside the release assets.
5. **Verify offline (CI already does this, but double-check before uploading)**
   - Transfer `dist/` to an isolated machine.
   - `python -m pip install --no-index ./kll_sketch-X.Y.Z-py3-none-any.whl`
   - Run `python - <<'PY'` to sanity-check the install:
     ```python
     from kll_sketch import KLL

     sketch = KLL(capacity=128)
     sketch.extend(range(10_000))
     assert abs(sketch.quantile(0.5) - 4999.5) < 5
     print("offline install ok")
     ```
     PY
6. **Upload & tag**
   - `twine upload dist/*`
   - `git tag -s vX.Y.Z -m "kll-sketch vX.Y.Z"`
   - `git push origin vX.Y.Z`

Document any deviations here so the next release engineer has the full context.
