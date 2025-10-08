# Changelog

All notable changes to this project will be documented in this file.

The format roughly follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-03-01
### Added
- Initial public release of the deterministic Python implementation of the KLL streaming quantile sketch.
- Serialization helpers (`to_bytes` / `from_bytes`) with versioned binary framing (`KLL1`).
- Benchmarks and documentation describing accuracy and performance envelopes.

## Release Signing
All published distributions on PyPI are signed with the maintainer's OpenPGP key (`0xA3D0A2F6E24F3B7C`). Verify signatures with:

```bash
pip download kll-sketch==1.0.0
python -m gpg --verify kll_sketch-1.0.0.tar.gz.asc kll_sketch-1.0.0.tar.gz
```

Public key fingerprints and additional verification steps are listed in the release notes on GitHub.
