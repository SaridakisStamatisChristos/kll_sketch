#!/usr/bin/env python3
"""Deterministic differential benchmark for the optional native backend."""
from __future__ import annotations

import argparse
from array import array
from pathlib import Path
import sys
import time

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kll_sketch import KLL, native_backend_info, native_available, set_native_enabled


def _time_build(data, *, enabled: bool, k: int, seed: int) -> tuple[KLL, float]:
    set_native_enabled(enabled)
    sketch = KLL(k, seed)
    start = time.perf_counter()
    sketch.extend(data)
    elapsed = time.perf_counter() - start
    sketch.validate()
    return sketch, elapsed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=300_000)
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--seed", type=int, default=7331)
    p.add_argument("--min-speedup", type=float, default=1.25)
    args = p.parse_args()
    if args.N <= 0:
        raise SystemExit("N must be positive")
    if not native_available():
        raise SystemExit("native extension is not available")

    values = [((i * 1103515245 + 12345) & 0xFFFFFFFF) / 2**32 for i in range(args.N)]
    buffer_values = array("d", values)

    try:
        pure, pure_s = _time_build(values, enabled=False, k=args.k, seed=args.seed)
        native, native_s = _time_build(values, enabled=True, k=args.k, seed=args.seed)
        buffered, buffered_s = _time_build(buffer_values, enabled=True, k=args.k, seed=args.seed)
    finally:
        set_native_enabled(True)

    if native.to_bytes() != pure.to_bytes():
        raise SystemExit("native/list state diverged from Python reference")
    if buffered.to_bytes() != pure.to_bytes():
        raise SystemExit("native/buffer state diverged from Python reference")

    pure_rate = args.N / pure_s
    native_rate = args.N / native_s
    buffer_rate = args.N / buffered_s
    speedup = native_rate / pure_rate
    buffer_speedup = buffer_rate / pure_rate

    info = native_backend_info()
    print(f"backend={info}")
    print(f"python: {pure_rate:,.0f} updates/s ({pure_s:.3f}s)")
    print(f"native-list: {native_rate:,.0f} updates/s ({native_s:.3f}s), {speedup:.2f}x")
    print(f"native-buffer: {buffer_rate:,.0f} updates/s ({buffered_s:.3f}s), {buffer_speedup:.2f}x")

    if speedup < args.min_speedup:
        raise SystemExit(
            f"native acceleration regression: {speedup:.2f}x < required {args.min_speedup:.2f}x"
        )


if __name__ == "__main__":
    main()
