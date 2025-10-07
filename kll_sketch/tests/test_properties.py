"""Property-based tests exercising probabilistic guarantees of :mod:`kll_sketch`."""
from __future__ import annotations

import bisect
from typing import Sequence

import math

import pytest

hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies
given = hypothesis.given
settings = hypothesis.settings

from kll_sketch import KLL


def _sorted_list(seq: Sequence[float]) -> list[float]:
    ordered = list(seq)
    ordered.sort()
    return ordered


@given(
    st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=2_000,
    ),
    st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=75, deadline=None)
def test_quantile_rank_error_is_bounded(xs: list[float], q: float) -> None:
    sketch = KLL(capacity=256)
    sketch.extend(xs)

    estimate = sketch.quantile(q)
    ordered = _sorted_list(xs)
    target_rank = q * (len(xs) - 1)

    # Compute the realised rank interval for the estimate in the truth data.
    left = bisect.bisect_left(ordered, estimate)
    right = bisect.bisect_right(ordered, estimate)

    # Allow a tolerance proportional to 1/k (here ~0.004) plus a small constant
    # for discrete datasets.  The assert keeps the property coarse but useful.
    slack = max(3.0, 0.04 * len(xs))
    assert left <= target_rank + slack
    assert right >= target_rank - slack


@given(
    st.lists(
        st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
        min_size=0,
        max_size=1_000,
    ),
    st.lists(
        st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
        min_size=0,
        max_size=1_000,
    ),
)
@settings(max_examples=60, deadline=None)
def test_merge_matches_extending(xs: list[float], ys: list[float]) -> None:
    combined = xs + ys

    serial = KLL(capacity=128)
    serial.extend(combined)

    a = KLL(capacity=128)
    b = KLL(capacity=128)
    a.extend(xs)
    b.extend(ys)
    a.merge(b)

    if not combined:
        assert serial.size() == 0
        assert a.size() == 0
        assert b.size() == 0
        return

    for q in [0.0, 0.1, 0.5, 0.9, 1.0]:
        assert math.isclose(a.quantile(q), serial.quantile(q), rel_tol=0.05, abs_tol=0.05)


@given(
    st.lists(
        st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
        min_size=0,
        max_size=1_500,
    )
)
@settings(max_examples=60, deadline=None)
def test_serialization_roundtrip_matches_levels(xs: list[float]) -> None:
    sketch = KLL(capacity=200)
    sketch.extend(xs)
    payload = sketch.to_bytes()
    restored = KLL.from_bytes(payload)

    assert restored.size() == sketch.size()
    assert restored._levels == sketch._levels

    if xs:
        for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
            restored_q = restored.quantile(q)
            sketch_q = sketch.quantile(q)
            assert math.isclose(restored_q, sketch_q, rel_tol=1e-9, abs_tol=1e-9)
