"""Regression tests that exercise the sketch under multiple RNG seeds."""

from __future__ import annotations

import math
import random
import statistics
from typing import Iterable

import pytest

from kll_sketch import KLL


def _exact_quantile(values: Iterable[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("_exact_quantile requires a non-empty iterable")
    # Linear interpolation between neighbouring order statistics to mirror the
    # behaviour of NumPy's ``quantile`` default method.
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight_upper = position - lower
    weight_lower = 1.0 - weight_upper
    return ordered[lower] * weight_lower + ordered[upper] * weight_upper


@pytest.mark.parametrize("seed", [3, 17, 221, 1987, 4096])
def test_quantile_accuracy_across_random_seeds(seed: int) -> None:
    rng = random.Random(seed)
    samples = [rng.gauss(0.0, 1.0) for _ in range(8_000)]

    sketch = KLL(capacity=256, rng_seed=seed)
    sketch.extend(samples)

    approx = sketch.quantile(0.5)
    exact = _exact_quantile(samples, 0.5)
    dispersion = statistics.pstdev(samples)
    tolerance = max(1e-9, 0.12 * dispersion)

    print(
        "seed={seed}: approx median={approx:.6f}, exact={exact:.6f}, tolerance={tolerance:.6f}".format(
            seed=seed, approx=approx, exact=exact, tolerance=tolerance
        )
    )

    assert abs(approx - exact) <= tolerance


def test_deterministic_compactions_for_fixed_seed() -> None:
    seed = 123_456
    rng = random.Random(seed)
    payload = [rng.uniform(-5.0, 5.0) for _ in range(5_000)]

    a = KLL(capacity=200, rng_seed=seed)
    b = KLL(capacity=200, rng_seed=seed)
    for value in payload:
        a.add(value)
        b.add(value)

    print(
        "deterministic compaction: sketch bytes size={size}, levels={levels}".format(
            size=len(a.to_bytes()),
            levels=sum(len(level) for level in a._levels),
        )
    )

    assert a._levels == b._levels
    assert a.to_bytes() == b.to_bytes()
    for q in [0.05, 0.5, 0.95]:
        assert math.isclose(a.quantile(q), b.quantile(q), rel_tol=1e-12, abs_tol=1e-12)
