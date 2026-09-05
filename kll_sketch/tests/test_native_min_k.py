from __future__ import annotations

import pytest

from kll_sketch import KLL, native_available, set_native_enabled


def _build(k: int, seed: int, start: int, count: int, *, native: bool) -> KLL:
    set_native_enabled(native)
    sketch = KLL(k, seed)
    sketch.extend(float(start + i) for i in range(count))
    return sketch


@pytest.mark.skipif(not native_available(), reason="optional native extension not built")
def test_native_mixed_k_merge_matches_python_min_k_and_bytes() -> None:
    try:
        pure_dst = _build(200, 7001, 0, 5_000, native=False)
        pure_src = _build(80, 7002, 10_000, 5_000, native=False)
        pure_dst.merge(pure_src)

        native_dst = _build(200, 7001, 0, 5_000, native=True)
        native_src = _build(80, 7002, 10_000, 5_000, native=True)
        native_dst.merge(native_src)

        assert native_dst.min_k == pure_dst.min_k == 80
        assert native_dst.debug_state() == pure_dst.debug_state()
        assert native_dst.to_bytes() == pure_dst.to_bytes()
    finally:
        set_native_enabled(True)


@pytest.mark.skipif(not native_available(), reason="optional native extension not built")
def test_native_min_k_cache_survives_keyword_fallback_then_fast_merge() -> None:
    try:
        # Build the canonical Python reference sequence first.
        pure_dst = _build(200, 8100, 0, 4_000, native=False)
        pure_same = _build(200, 8101, 10_000, 4_000, native=False)
        pure_low = _build(80, 8102, 20_000, 4_000, native=False)
        pure_lower = _build(40, 8103, 30_000, 4_000, native=False)
        pure_dst.merge(pure_same)
        pure_dst.merge(other=pure_low)
        pure_dst.merge(pure_lower)

        # Positional merge populates the resident fast-path cache. Keyword merge
        # intentionally enters the Python fallback; the following positional
        # merge must reload any invalidated min_k cache rather than use stale C++
        # metadata.
        native_dst = _build(200, 8100, 0, 4_000, native=True)
        native_same = _build(200, 8101, 10_000, 4_000, native=True)
        native_low = _build(80, 8102, 20_000, 4_000, native=True)
        native_lower = _build(40, 8103, 30_000, 4_000, native=True)
        native_dst.merge(native_same)
        native_dst.merge(other=native_low)
        native_dst.merge(native_lower)

        assert native_dst.min_k == pure_dst.min_k == 40
        assert native_dst.debug_state() == pure_dst.debug_state()
        assert native_dst.to_bytes() == pure_dst.to_bytes()
    finally:
        set_native_enabled(True)
