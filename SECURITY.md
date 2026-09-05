# Security policy

## Supported versions

Security fixes are provided for the current `3.2.x` release line. Older snapshots may
receive fixes only when a report demonstrates that the issue also affects the current
release.

## Reporting a vulnerability

Please do **not** open a public issue for an unpatched vulnerability. Use GitHub's
private vulnerability reporting/security-advisory interface when it is available for
this repository. If that interface is unavailable, contact the maintainer at
`stamatis@saridakis.dev` with the subject `kll-sketch security report`.

Include:

- affected version/commit and platform;
- a minimal reproducer or malformed payload when safe to share;
- expected versus observed behavior;
- impact and whether untrusted input is required.

## Security model and trust boundaries

`kll-sketch` is an in-process data structure. It does not perform networking, spawn
subprocesses, load plugins, use `pickle`, or execute code from serialized sketches.

The main untrusted-input boundary is `KLL.from_bytes()`. KLL2 payloads are length
checked, checksummed, parsed with bounded level counts, validated for finite values,
and checked against represented-mass/retained-count invariants before a sketch is
returned. Historical KLL1 payloads remain readable for compatibility and are also
structurally validated, but they do not carry the KLL2 checksum.

The optional native extension is a performance layer. Unsupported public inputs must
fall back to the Python semantic path; native acceleration must not relax validation,
serialization, or exact-state invariants.

Performance benchmark files and JSON/CSV artifacts are data, not trusted executable
inputs. Benchmark claims should be reproduced from the pinned workflow environment
before being treated as evidence on another machine.
