"""Property and randomized differential-style tests for the KLL core."""
from __future__ import annotations

import bisect
import random

import pytest

from kll_sketch import KLL

hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies
given = hypothesis.given
settings = hypothesis.settings


def _rank_error(ordered: list[float], estimate: float, q: float) -> float:
    target = q * (len(ordered) - 1)
    lo = bisect.bisect_left(ordered, estimate)
    hi = bisect.bisect_right(ordered, estimate) - 1
    if target < lo:
        return (lo - target) / len(ordered)
    if target > hi:
        return (target - hi) / len(ordered)
    return 0.0


@given(
    st.lists(
        st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=3000,
    )
)
@settings(max_examples=80, deadline=None)
def test_structural_invariants_hold(xs: list[float]) -> None:
    sketch = KLL(capacity=128, rng_seed=7331)
    sketch.extend(xs)
    sketch.validate()
    assert sketch.min_value == min(xs)
    assert sketch.max_value == max(xs)
    assert sketch.num_retained <= sketch.debug_state()["total_capacity"]


@given(
    st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=2500,
    ),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=80, deadline=None)
def test_quantile_contract_and_rank_error(xs: list[float], q: float) -> None:
    sketch = KLL(capacity=200, rng_seed=19)
    sketch.extend(xs)
    ordered = sorted(xs)
    estimate = sketch.quantile(q)

    if not sketch.is_estimation_mode:
        # Before the first compaction the sketch is exact. Quantile semantics are
        # the lower order statistic at floor(q * (n - 1)); do not apply a
        # continuous-rank error metric to tiny exact samples such as [0, 1].
        expected = ordered[int(q * (len(ordered) - 1))]
        assert estimate == expected
        return

    # Once compaction starts, validate the actual KLL contract in normalized-rank
    # space. Release characterization provides the tighter p95/p99 envelope.
    assert _rank_error(ordered, estimate, q) <= 0.05


@given(
    st.lists(st.integers(-10000, 10000), min_size=0, max_size=1500),
    st.lists(st.integers(-10000, 10000), min_size=0, max_size=1500),
)
@settings(max_examples=60, deadline=None)
def test_merge_preserves_mass_extrema_and_query_order(xs: list[int], ys: list[int]) -> None:
    left = KLL(128, 1)
    right = KLL(128, 2)
    left.extend(xs)
    right.extend(ys)
    left.merge(right)
    left.validate()
    combined = xs + ys
    assert left.n == len(combined)
    if combined:
        assert left.min_value == min(combined)
        assert left.max_value == max(combined)
        qs = [0.0, 0.1, 0.5, 0.9, 1.0]
        answers = left.quantiles_at(qs)
        assert answers == sorted(answers)


@given(st.lists(st.integers(-1000, 1000), min_size=0, max_size=2000))
@settings(max_examples=60, deadline=None)
def test_serialization_is_byte_stable(xs: list[int]) -> None:
    sketch = KLL(200, 123456789)
    sketch.extend(xs)
    payload = sketch.to_bytes()
    restored = KLL.from_bytes(payload)
    restored.validate()
    assert restored.to_bytes() == payload
    assert restored._levels == sketch._levels


def test_merge_tree_characterization_smoke() -> None:
    rng = random.Random(42)
    xs = [rng.random() for _ in range(50_000)]
    shards = []
    for i in range(8):
        shard = KLL(200, i + 1)
        shard.extend(xs[i::8])
        shards.append(shard)
    while len(shards) > 1:
        nxt = []
        for i in range(0, len(shards), 2):
            shards[i].merge(shards[i + 1])
            nxt.append(shards[i])
        shards = nxt
    merged = shards[0]
    merged.validate()
    ordered = sorted(xs)
    for q in [0.01, 0.1, 0.5, 0.9, 0.99]:
        assert _rank_error(ordered, merged.quantile(q), q) < 0.05
