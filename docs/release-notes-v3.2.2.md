# v3.2.2 release notes

Release target: `v3.2.2`

## What changed

Version 3.2.2 is a packaging-only correction. The universal wheel now stores its
declared `LICENSE` file at `.dist-info/licenses/LICENSE`, matching Core Metadata 2.4 and
PyPI's distribution validation requirements.

Release validation now checks the declared wheel license path, contents, metadata, and
`RECORD` entry before publication. The source-distribution `PKG-INFO` correction from
3.2.1 remains intact.

## Compatibility

- The public `KLL` / `KLLSketch` API is unchanged.
- KLL2 serialization is unchanged; KLL1 remains readable.
- Deterministic seeded Python/native state parity is unchanged.
- The canonical wheel remains pure `py3-none-any`; native compilation remains explicit.
- Python 3.10+ remains supported.

There are no algorithm, accuracy-model, performance-path, or wire-format changes in this
patch release.
