#!/usr/bin/env python3
"""Stable same-process performance regression gates for the optional native backend.

The benchmark compares native and pure-Python execution in the same interpreter,
on identical deterministic inputs and seeds. It is intentionally conservative:
ratios, not absolute wall-clock numbers, are gated on shared CI runners.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kll_sketch import KLL, __version__, native_backend_info, native_available, set_native_enabled

QS = (0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999)


def _values(n: int) -> list[float]:
    return [((i * 1103515245 + 12345) & 0xFFFFFFFF) / 2**32 for i in range(n)]


def _median_time(callable_, repeats: int) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        callable_()
        samples.append((time.perf_counter_ns() - start) / 1e9)
    return statistics.median(samples)


def _build(data: list[float], *, enabled: bool, k: int, seed: int) -> KLL:
    set_native_enabled(enabled)
    sk = KLL(k, seed)
    sk.extend(data)
    sk.validate()
    return sk


def _ingestion(data: list[float], *, enabled: bool, k: int, seed: int, trials: int) -> tuple[float, bytes]:
    rates: list[float] = []
    last = b""
    for i in range(trials):
        set_native_enabled(enabled)
        start = time.perf_counter_ns()
        sk = KLL(k, seed + i)
        sk.extend(data)
        elapsed = (time.perf_counter_ns() - start) / 1e9
        sk.validate()
        rates.append(len(data) / elapsed)
        last = sk.to_bytes()
    return statistics.median(rates), last


def _query(sk: KLL, *, enabled: bool, loops: int, trials: int) -> float:
    latencies: list[float] = []
    for _ in range(trials):
        set_native_enabled(enabled)
        sk.quantiles_at(QS)
        start = time.perf_counter_ns()
        for _ in range(loops):
            sk.quantiles_at(QS)
        latencies.append((time.perf_counter_ns() - start) / loops / 1e3)
    return statistics.median(latencies)


def _sources(data: list[float], *, enabled: bool, k: int, seed: int, shards: int) -> list[KLL]:
    step = (len(data) + shards - 1) // shards
    out: list[KLL] = []
    set_native_enabled(enabled)
    for i in range(shards):
        part = data[i * step : min(len(data), (i + 1) * step)]
        if not part:
            break
        sk = KLL(k, seed + 1000 + i)
        sk.extend(part)
        sk.validate()
        out.append(sk)
    return out


def _merge(
    sources: list[KLL],
    *,
    enabled: bool,
    k: int,
    seed: int,
    loops: int,
    trials: int,
    expected_n: int,
) -> tuple[float, bytes]:
    timings: list[float] = []
    representative = b""
    for trial in range(trials):
        set_native_enabled(enabled)
        destinations = [KLL(k, seed + 50000 + trial * loops + i) for i in range(loops)]
        start = time.perf_counter_ns()
        for dst in destinations:
            for src in sources:
                dst.merge(src)
        elapsed_us = (time.perf_counter_ns() - start) / loops / 1e3
        if any(dst.n != expected_n for dst in destinations):
            raise AssertionError("merge produced incorrect represented mass")
        timings.append(elapsed_us)
        representative = destinations[0].to_bytes()
    return statistics.median(timings), representative


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=250_000)
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--seed", type=int, default=7331)
    p.add_argument("--shards", type=int, default=8)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--query-loops", type=int, default=2000)
    p.add_argument("--merge-loops", type=int, default=96)
    p.add_argument("--min-ingestion-speedup", type=float, default=1.15)
    p.add_argument("--min-query-speedup", type=float, default=1.05)
    p.add_argument("--min-merge-speedup", type=float, default=1.05)
    p.add_argument("--out", type=Path, default=Path("performance_regression.json"))
    args = p.parse_args()

    if args.N <= 0 or args.trials < 3 or args.query_loops <= 0 or args.merge_loops <= 0:
        raise SystemExit("N/query-loops/merge-loops must be positive and trials must be >= 3")
    if args.shards < 2:
        raise SystemExit("shards must be >= 2")
    if not native_available():
        raise SystemExit("native extension is required")

    data = _values(args.N)
    info = native_backend_info()
    try:
        pure_ingest, pure_bytes = _ingestion(
            data, enabled=False, k=args.k, seed=args.seed, trials=args.trials
        )
        native_ingest, native_bytes = _ingestion(
            data, enabled=True, k=args.k, seed=args.seed, trials=args.trials
        )
        if pure_bytes != native_bytes:
            raise AssertionError("native ingestion state diverged from pure Python")

        pure_query_sk = _build(data, enabled=False, k=args.k, seed=args.seed + 777)
        native_query_sk = _build(data, enabled=True, k=args.k, seed=args.seed + 777)
        if pure_query_sk.to_bytes() != native_query_sk.to_bytes():
            raise AssertionError("query fixtures do not have identical state")
        native_query = _query(native_query_sk, enabled=True, loops=args.query_loops, trials=args.trials)
        pure_query = _query(pure_query_sk, enabled=False, loops=args.query_loops, trials=args.trials)

        pure_sources = _sources(data, enabled=False, k=args.k, seed=args.seed, shards=args.shards)
        native_sources = _sources(data, enabled=True, k=args.k, seed=args.seed, shards=args.shards)
        if [s.to_bytes() for s in pure_sources] != [s.to_bytes() for s in native_sources]:
            raise AssertionError("merge source states diverged")
        native_merge, native_merge_bytes = _merge(
            native_sources,
            enabled=True,
            k=args.k,
            seed=args.seed,
            loops=args.merge_loops,
            trials=args.trials,
            expected_n=args.N,
        )
        pure_merge, pure_merge_bytes = _merge(
            pure_sources,
            enabled=False,
            k=args.k,
            seed=args.seed,
            loops=args.merge_loops,
            trials=args.trials,
            expected_n=args.N,
        )
        if native_merge_bytes != pure_merge_bytes:
            raise AssertionError("native merge state diverged from pure Python")
    finally:
        set_native_enabled(True)

    speedups = {
        "ingestion": native_ingest / pure_ingest,
        "query": pure_query / native_query,
        "merge": pure_merge / native_merge,
    }
    result = {
        "schema": 1,
        "project_version": __version__,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "native": info,
            "N": args.N,
            "k": args.k,
            "shards": args.shards,
            "trials": args.trials,
            "query_loops": args.query_loops,
            "merge_loops": args.merge_loops,
        },
        "medians": {
            "pure_updates_per_s": pure_ingest,
            "native_updates_per_s": native_ingest,
            "pure_query_us": pure_query,
            "native_query_us": native_query,
            "pure_merge_us": pure_merge,
            "native_merge_us": native_merge,
        },
        "speedups_native_over_python": speedups,
        "thresholds": {
            "ingestion": args.min_ingestion_speedup,
            "query": args.min_query_speedup,
            "merge": args.min_merge_speedup,
        },
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))

    failures = [
        f"{name} {speedups[name]:.3f}x < {threshold:.3f}x"
        for name, threshold in (
            ("ingestion", args.min_ingestion_speedup),
            ("query", args.min_query_speedup),
            ("merge", args.min_merge_speedup),
        )
        if speedups[name] < threshold
    ]
    if failures:
        raise SystemExit("performance regression: " + "; ".join(failures))


if __name__ == "__main__":
    main()
