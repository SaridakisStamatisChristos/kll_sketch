"""Deterministic regression tests for :mod:`kll_sketch`."""
from __future__ import annotations

import bisect
import math
import random
from typing import Iterable

import pytest

from kll_sketch import KLL


def _truth_quantile(xs: Iterable[float], q: float) -> float:
    ordered = sorted(xs)
    if not ordered:
        raise ValueError("empty iterable")
    idx = int(q * (len(ordered) - 1))
    return ordered[idx]


def test_basic_quantiles_regression() -> None:
    """The sketch stays within a tight absolute error on a pseudo-random stream."""
    rng = random.Random(1)
    xs = [rng.random() for _ in range(20_000)]
    truth = sorted(xs)

    sketch = KLL(capacity=256)
    sketch.extend(xs)

    # Deterministic checkpoints chosen to exercise both tails and the median.
    for q in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
        estimate = sketch.quantile(q)
        reference = truth[int(q * (len(xs) - 1))]
        assert abs(estimate - reference) <= 0.015


@pytest.mark.parametrize("capacity", [64, 128, 256])
def test_quantile_rank_consistency(capacity: int) -> None:
    """Quantile answers are compatible with the public ``rank`` helper."""
    rng = random.Random(capacity)
    xs = [rng.uniform(-5.0, 5.0) for _ in range(2_000)]

    sketch = KLL(capacity=capacity)
    sketch.extend(xs)

    for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
        estimate = sketch.quantile(q)
        approx_rank = sketch.rank(estimate)
        target_rank = q * sketch.size()
        # rank() returns a value in [0, n]; allow a small tolerance in samples.
        assert math.isclose(approx_rank, target_rank, rel_tol=0.05, abs_tol=2.0)


def test_rank_and_cdf_are_monotone() -> None:
    rng = random.Random(7)
    xs = sorted(rng.uniform(-1.0, 1.0) for _ in range(500))

    sketch = KLL(capacity=200)
    sketch.extend(xs)

    last_rank = -1.0
    for x in xs:
        rank = sketch.rank(x)
        assert rank >= last_rank
        last_rank = rank

    cdf_values = sketch.cdf(xs)
    assert all(0.0 <= value <= 1.0 for value in cdf_values)
    assert cdf_values == sorted(cdf_values)


def test_weight_conservation() -> None:
    rng = random.Random(0)
    sketch = KLL(capacity=64, rng_seed=777)
    xs = [rng.random() for _ in range(50_000)]
    sketch.extend(xs)
    vals, wts = sketch._materialize_aligned()
    assert len(vals) == len(wts)
    assert abs(sum(wts) - sketch.size()) < 1e-9


def test_merge_matches_single_stream() -> None:
    rng = random.Random(321)
    left = [rng.random() for _ in range(5_000)]
    right = [rng.random() for _ in range(5_000)]

    merged = KLL(capacity=200)
    merged.extend(left)
    merged.extend(right)

    a = KLL(capacity=200)
    b = KLL(capacity=200)
    a.extend(left)
    b.extend(right)
    a.merge(b)

    checkpoints = [0.01, 0.1, 0.5, 0.9, 0.99]
    for q in checkpoints:
        assert math.isclose(a.quantile(q), merged.quantile(q), rel_tol=0.02, abs_tol=0.01)


@pytest.mark.parametrize(
    "sample",
    [
        [],
        [0.0],
        [0.0, 0.0, 0.0],
        [-1.5, 0.0, 1.5],
        [float(i) for i in range(10)],
    ],
)
def test_serialization_roundtrip(sample: list[float]) -> None:
    sketch = KLL(capacity=64)
    sketch.extend(sample)

    restored = KLL.from_bytes(sketch.to_bytes())
    assert restored.size() == sketch.size()
    assert restored._levels == sketch._levels
    if sketch.size():
        assert restored.quantile(0.5) == pytest.approx(sketch.quantile(0.5))
    else:
        with pytest.raises(ValueError):
            restored.quantile(0.5)


def test_invalid_inputs_raise() -> None:
    sketch = KLL(capacity=64)
    with pytest.raises(ValueError):
        sketch.add(float("nan"))
    with pytest.raises(ValueError):
        sketch.add(float("inf"))
    with pytest.raises(ValueError):
        sketch.quantile(-0.01)
    with pytest.raises(ValueError):
        sketch.quantile(1.5)
    with pytest.raises(ValueError):
        sketch.quantile(0.5)  # empty sketch


@pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("values", [[1.0], [1.0, 2.0, 3.0], [5.0] * 10])
def test_quantile_matches_truth_for_small_inputs(values: list[float], q: float) -> None:
    sketch = KLL(capacity=64)
    sketch.extend(values)
    truth = _truth_quantile(values, q)
    assert sketch.quantile(q) == pytest.approx(truth)
    rank_estimate = sketch.rank(truth)
    assert 0.0 <= rank_estimate <= len(values)


def test_rank_brackets_quantile() -> None:
    rng = random.Random(99)
    xs = [rng.uniform(-10, 10) for _ in range(3_000)]

    sketch = KLL(capacity=128)
    sketch.extend(xs)

    for q in [0.0, 0.1, 0.5, 0.9, 1.0]:
        estimate = sketch.quantile(q)
        ordered = sorted(xs)
        lower = bisect.bisect_left(ordered, estimate)
        upper = bisect.bisect_right(ordered, estimate)
        target_rank = q * (len(xs) - 1)
        assert lower <= target_rank + 200
        assert upper >= target_rank - 200
