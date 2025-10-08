# Production Readiness Status

This document captures the current state of the project with respect to "production readiness" and provides the evidence that offline deployments are now fully supported.

## ✅ Current guarantees
- **Packaging**: Releases are produced with an in-tree PEP 517 backend that only relies on the Python standard library. Building from source requires no third-party wheels.
- **Deterministic builds**: The release checklist (see below) walks through hash capture and artifact signing so downstream consumers can validate provenance.
- **Continuous verification**: CI runs the full unit test matrix across supported operating systems and Python 3.9–3.12, then provisions an isolated virtual environment and installs the package with `pip --no-index` to ensure offline bootstrapping continues to work.

## 🔍 Validated environments

| OS      | Python versions         |
| ------- | ----------------------- |
| Linux   | 3.9, 3.10, 3.11, 3.12   |
| macOS   | 3.9, 3.10, 3.11, 3.12   |
| Windows | 3.9, 3.10, 3.11, 3.12   |

## 📦 Release process
- Follow the [signed release checklist](release-checklist.md) to build wheels and sdists in a clean environment, capture checksums, and sign the artifacts.
- Archive the generated `CHECKSUMS.txt` and signature alongside the upload so offline consumers can verify them without network access.

## 🚀 Operational notes
- The README’s offline section documents how to install both wheels and source checkouts without touching a package index.
- Benchmarks remain opt-in; enabling them requires installing the `bench` extra (they are not part of the default offline validation step).

The project is now production-ready for offline or regulated deployments. New gaps should be tracked in this document as they are discovered.
