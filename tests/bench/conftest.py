from __future__ import annotations

from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    # Ensure the default pytest-benchmark output directory exists when users
    # pass ``--benchmark-json=bench_out/pytest/results.json``.
    Path("bench_out/pytest").mkdir(parents=True, exist_ok=True)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    marker_expr = config.getoption("-m")
    if marker_expr:
        # Users explicitly requested a marker expression; respect it.
        return
    skip_marker = pytest.mark.skip(reason="benchmark tests are opt-in; run with -m benchmark")
    for item in items:
        if "benchmark" in item.keywords:
            item.add_marker(skip_marker)
