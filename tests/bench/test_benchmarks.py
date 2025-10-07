"""Pytest micro-benchmarks for the KLL sketch."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("numpy")
import numpy as np

from kll_sketch import KLLSketch

OUTPUT_DIR = Path("bench_out/pytest")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pytestmark = pytest.mark.benchmark


def _generate_data(dist: str, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if dist == "uniform":
        return rng.uniform(0.0, 1.0, size)
    if dist == "normal":
        return rng.normal(0.0, 1.0, size)
    raise ValueError(f"Unsupported distribution for pytest benchmarks: {dist}")


@pytest.mark.parametrize("distribution", ["uniform", "normal"])
@pytest.mark.parametrize("N", [int(1e5), int(1e6)])
@pytest.mark.parametrize("capacity", [200, 400, 800])
def test_update_throughput(distribution: str, N: int, capacity: int, benchmark) -> None:
    data = _generate_data(distribution, N, seed=42)

    def build_sketch() -> KLLSketch:
        sketch = KLLSketch(capacity=capacity, rng_seed=42)
        for value in data:
            sketch.add(float(value))
        return sketch

    sketch = benchmark(build_sketch)

    approx = sketch.quantile(0.5)
    exact = float(np.quantile(data, 0.5))
    std = float(np.std(data))
    scale = 0.02 * std * (200 / capacity) ** 0.5 if std > 0 else 0.0
    tolerance = max(1e-9, scale)
    assert abs(approx - exact) <= tolerance
