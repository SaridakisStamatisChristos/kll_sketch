# Production-readiness criteria

Version 2 treats "production ready" as an evidence claim, not a README adjective.

A release candidate is considered ready only when the following gates pass for the exact release commit:

- unit and property tests;
- >= 90% branch-aware coverage for runtime code;
- Linux/macOS/Windows test matrix on supported Python versions;
- strict KLL1/KLL2 serialization tests;
- offline `pip --no-index .` source installation;
- wheel content inspection and installed-package smoke test;
- rank-space benchmark smoke validation;
- raw benchmark artifacts retained from CI;
- no unsupported value-error or portable-performance claims.

The current workflow definition is `.github/workflows/ci.yml`. GitHub Actions status for the release commit is the authoritative current evidence; this document intentionally does not hard-code a historical green badge as if it guaranteed future commits.
