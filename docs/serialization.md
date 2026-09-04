# Serialization

## KLL2

Version 2 writes a strict, checksummed binary envelope.

All integers are unsigned big-endian. Floating-point values are IEEE-754 binary64 big-endian.

### Header

| Field | Size |
|---|---:|
| magic `KLL2` | 4 bytes |
| payload length | uint32 |
| CRC32(payload) | uint32 |

### Payload

| Field | Type |
|---|---|
| flags (`bit0 = non-empty`) | uint8 |
| configured `k` | uint32 |
| effective `min_k` | uint32 |
| total represented `n` | uint64 |
| level count | uint32 |
| original seed | uint64 |
| current SplitMix64 state | uint64 |
| compaction count | uint64 |
| retained-item count | uint64 |
| exact minimum | float64 |
| exact maximum | float64 |
| each level length | uint32 |
| each level payload | `length * float64` |

For empty sketches the extrema slots are serialized as zero and ignored because the non-empty flag is clear.

## Validation on read

`from_bytes()` rejects:

- unsupported magic;
- truncated headers or payloads;
- payload length mismatch;
- trailing bytes;
- CRC mismatch;
- unsupported feature flags;
- illegal level counts;
- level lengths larger than remaining input;
- non-finite retained values;
- invalid `k` / `min_k` relationships;
- retained-count mismatches;
- weighted-level mass not equal to `n`;
- inconsistent empty/non-empty state;
- reversed or non-finite extrema;
- retained values outside the declared extrema.

All format failures are normalized to `SerializationError` rather than exposing low-level `struct.error` details.

A 512 MiB payload safety limit prevents hostile length fields from requesting unreasonable work.

## KLL1 compatibility

Version 2 retains a strict reader for historical `KLL1` payloads:

```text
magic, k, n, level_count, seed, [level_length, values]...
```

KLL1 had no checksum or explicit extrema. The 1.x implementation preserved boundary values during compaction, so the compatibility reader reconstructs extrema from retained values and then validates represented mass.

New writes always use KLL2.

## Compatibility policy

- Readers should remain backward-compatible with every format published by this project.
- Breaking wire changes require a new magic/version.
- New readers fail closed on unsupported feature flags.
- Golden byte fixtures and mutation tests should accompany any future format revision.
