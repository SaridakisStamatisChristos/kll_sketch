from __future__ import annotations

from array import array

import pytest

from kll_sketch import KLL, native_available, set_native_enabled


pytestmark = pytest.mark.skipif(not native_available(), reason="native extension is not built")


def _pure_reference(left, right, tail):
    set_native_enabled(False)
    dst = KLL(80, 7331)
    src = KLL(80, 7332)
    dst.extend(left)
    src.extend(right)
    dst.merge(src)
    visible = (dst.n, dst.num_retained, dst.min_value, dst.max_value, dst.min_k)
    dst.extend(tail)
    return visible, dst.to_bytes(), dst.quantiles_at((0.01, 0.5, 0.99))


def test_resident_merge_compact_result_preserves_exact_state() -> None:
    left = array("d", ((i * 17) % 1009 - 500.0 for i in range(12_000)))
    right = array("d", ((i * 29) % 1301 - 650.0 for i in range(9_000)))
    tail = array("d", (-777.0, -1.5, 0.0, 3.25, 999.0))

    try:
        expected_visible, expected_bytes, expected_qs = _pure_reference(left, right, tail)

        set_native_enabled(True)
        dst = KLL(80, 7331)
        src = KLL(80, 7332)
        dst.extend(left)
        src.extend(right)

        # Both sketches are resident here. The compact merge return path mirrors
        # public fields without materializing Python levels or a full stats tuple.
        dst.merge(src)
        assert (dst.n, dst.num_retained, dst.min_value, dst.max_value, dst.min_k) == expected_visible

        # Continuing natively after the merge must preserve the exact canonical
        # state, including RNG/compaction state that remains authoritative in C++.
        dst.extend(tail)
        assert dst.to_bytes() == expected_bytes
        assert dst.quantiles_at((0.01, 0.5, 0.99)) == expected_qs
    finally:
        set_native_enabled(True)


def test_resident_merge_syncs_cleanly_when_native_is_disabled() -> None:
    left = array("d", (float(i) for i in range(4000)))
    right = array("d", (float(-i) for i in range(3500)))

    try:
        set_native_enabled(False)
        pure_dst = KLL(96, 101)
        pure_src = KLL(96, 102)
        pure_dst.extend(left)
        pure_src.extend(right)
        pure_dst.merge(pure_src)
        expected = pure_dst.to_bytes()

        set_native_enabled(True)
        native_dst = KLL(96, 101)
        native_src = KLL(96, 102)
        native_dst.extend(left)
        native_src.extend(right)
        native_dst.merge(native_src)

        set_native_enabled(False)
        assert native_dst.to_bytes() == expected
    finally:
        set_native_enabled(True)
