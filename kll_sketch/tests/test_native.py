from __future__ import annotations

from array import array
import math
import random

import pytest

from kll_sketch import (
    KLL,
    native_available,
    native_backend_info,
    native_enabled,
    set_native_enabled,
)


def _build(data, *, enabled: bool, k: int = 128, seed: int = 7331) -> KLL:
    set_native_enabled(enabled)
    sketch = KLL(k, seed)
    sketch.extend(data)
    sketch.validate()
    return sketch


def test_backend_info_is_stable_without_extension() -> None:
    info = native_backend_info()
    assert info["available"] is native_available()
    assert info["enabled"] is native_enabled()
    if info["available"]:
        assert info["api_version"] == 1
        assert info["compiler"] in {"gcc", "clang", "msvc", "unknown"}
        assert info["simd"] in {"scalar", "avx2-runtime"}
    else:
        assert info["api_version"] is None
        with pytest.raises(RuntimeError):
            set_native_enabled(True)


@pytest.mark.skipif(not native_available(), reason="optional native extension not built")
def test_native_list_batch_is_byte_identical_to_python() -> None:
    rng = random.Random(9917)
    data = [rng.gauss(0.0, 3.0) for _ in range(80_000)]
    try:
        pure = _build(data, enabled=False)
        native = _build(data, enabled=True)
        assert native.debug_state() == pure.debug_state()
        assert native.to_bytes() == pure.to_bytes()
        qs = [.001, .01, .1, .5, .9, .99, .999]
        assert native.quantiles_at(qs) == pure.quantiles_at(qs)
        points = [-5.0, -1.0, 0.0, 1.0, 5.0]
        assert native.ranks(points) == pure.ranks(points)
        assert native.cdf(points) == pure.cdf(points)
    finally:
        set_native_enabled(True)


@pytest.mark.skipif(not native_available(), reason="optional native extension not built")
def test_native_contiguous_double_buffer_matches_python() -> None:
    data = array("d", (math.sin(i * 0.013) * 100.0 + i % 17 for i in range(50_000)))
    try:
        pure = _build(data, enabled=False, k=200, seed=42)
        native = _build(data, enabled=True, k=200, seed=42)
        assert native.to_bytes() == pure.to_bytes()
        assert native.min_value == pure.min_value
        assert native.max_value == pure.max_value
    finally:
        set_native_enabled(True)


@pytest.mark.skipif(not native_available(), reason="optional native extension not built")
def test_native_signed_zero_buffer_preserves_exact_python_state() -> None:
    # std::sort and SIMD min/max can reorder/equalize signed zero in ways Python
    # does not. v3 deliberately uses stable sorting and first-value extrema
    # semantics, so this is a byte-level regression test rather than ~=.
    data = array("d", [0.0, -0.0, 2.0, -2.0] * 20_000)
    try:
        pure = _build(data, enabled=False, k=128, seed=117)
        native = _build(data, enabled=True, k=128, seed=117)
        assert native.to_bytes() == pure.to_bytes()
        assert math.copysign(1.0, native.min_value) == math.copysign(1.0, pure.min_value)
        assert math.copysign(1.0, native.max_value) == math.copysign(1.0, pure.max_value)
    finally:
        set_native_enabled(True)


@pytest.mark.skipif(not native_available(), reason="optional native extension not built")
def test_native_range_batch_matches_python() -> None:
    data = range(-25_000, 25_000)
    try:
        pure = _build(data, enabled=False, k=256, seed=123)
        native = _build(data, enabled=True, k=256, seed=123)
        assert native.to_bytes() == pure.to_bytes()
    finally:
        set_native_enabled(True)


@pytest.mark.skipif(not native_available(), reason="optional native extension not built")
def test_native_invalid_batch_replays_python_partial_progress_semantics() -> None:
    try:
        set_native_enabled(True)
        sketch = KLL(128, 7)
        with pytest.raises(ValueError):
            sketch.extend([1.0, 2.0, float("nan"), 4.0])
        assert sketch.n == 2
        assert sketch.min_value == 1.0
        assert sketch.max_value == 2.0
        sketch.validate()
    finally:
        set_native_enabled(True)


@pytest.mark.skipif(not native_available(), reason="optional native extension not built")
def test_sized_one_shot_iterable_is_not_consumed_by_native_probe() -> None:
    class SizedOneShot:
        def __init__(self) -> None:
            self._used = False

        def __len__(self) -> int:
            return 4

        def __iter__(self):
            if self._used:
                return iter(())
            self._used = True
            return iter((1.0, 2.0, 3.0, 4.0))

    try:
        set_native_enabled(True)
        sketch = KLL(128, 3)
        sketch.extend(SizedOneShot())
        assert sketch.n == 4
        assert sketch.quantiles_at([0.0, 0.5, 1.0]) == [1.0, 2.0, 4.0]
    finally:
        set_native_enabled(True)


@pytest.mark.skipif(not native_available(), reason="optional native extension not built")
def test_enormous_weight_quantile_uses_exact_python_rank_comparison() -> None:
    try:
        set_native_enabled(True)
        sketch = KLL(200, 17)
        sketch.add(1.0, 1 << 54)
        sketch.add(2.0)
        assert sketch.n > 1 << 53
        assert sketch.quantile(0.5) == 1.0
        sketch.validate()
    finally:
        set_native_enabled(True)


@pytest.mark.skipif(not native_available(), reason="optional native extension not built")
def test_runtime_switch_preserves_class_identity_and_copy_contract() -> None:
    try:
        set_native_enabled(True)
        sketch = KLL(128, 5)
        sketch.extend(range(10_000))
        clone = sketch.copy()
        assert type(clone) is KLL
        assert clone.to_bytes() == sketch.to_bytes()
        set_native_enabled(False)
        assert type(sketch) is KLL
        assert type(clone) is KLL
    finally:
        set_native_enabled(True)
