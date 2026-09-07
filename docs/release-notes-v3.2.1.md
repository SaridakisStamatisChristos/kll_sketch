# v3.2.1 release notes

Release target: `v3.2.1`

## What changed

Version 3.2.1 is a packaging-only patch. Source distributions now contain the
standards-required root `PKG-INFO` file, generated from the same canonical metadata as
the universal wheel.

Release validation now checks the source distribution's metadata version, package name,
package version, and license expression before publication.

## Compatibility

- The public `KLL` / `KLLSketch` API is unchanged.
- KLL2 serialization is unchanged; KLL1 remains readable.
- Deterministic seeded Python/native state parity is unchanged.
- The canonical wheel remains pure `py3-none-any`; native compilation remains explicit.
- Python 3.10+ remains supported.

There are no algorithm, accuracy-model, performance-path, or wire-format changes in this
patch release.
