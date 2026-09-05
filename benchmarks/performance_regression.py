#!/usr/bin/env python3
"""Stable same-process performance regression gates for the optional native backend.

The benchmark compares native and pure-Python execution in the same interpreter on
identical deterministic inputs and seeds. Trials alternate implementation order to
reduce thermal/scheduler bias. CI gates on conservative ratios, not absolute wall-clock
numbers, and exact serialized-state parity is required throughout.
"""
from __future__ import annotations

import argparse
import json
import os
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


def _source_sha() -> str | None:
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if event_path:
        try:
            payload = json.loads(Path(event_path).read_text(encoding="utf-8"))
            pull_request = payload.get("pull_request") or {}
            head = pull_request.get("head") or {}
            sha = head.get("sha")
            if sha:
                return str(sha)
        except (OSError, ValueError, TypeError):
            pass
    return os.environ.get("GITHUB_SHA")


def _values(n: int) -> list[float]:
    return [((i * 1103515245 + 12345) & 0xFFFFFFFF) / 2**32 for i in range(n)]


def _build(data: list[float], *, enabled: bool, k: int, seed: int) -> KLL:
    set_native_enabled(enabled)
    sk = KLL(k, seed)
    sk.extend(data)
    sk.validate()
    return sk


def _ingestion_once(data: list[float], *, enabled: bool, k: int, seed: int) -> tuple[float, bytes]:
    set_native_enabled(enabled)
    start = time.perf_counter_ns()
    sk = KLL(k, seed)
    sk.extend(data)
    elapsed = (time.perf_counter_ns() - start) / 1e9
    sk.validate()
    return len(data) / elapsed, sk.to_bytes()


def _paired_ingestion(data: list[float], *, k: int, seed: int, trials: int) -> tuple[float, float]:
    pure_rates: list[float] = []
    native_rates: list[float] = []
    for trial in range(trials):
        measurements: dict[bool, tuple[float, bytes]] = {}
        order = (False, True) if trial % 2 == 0 else (True, False)
        for enabled in order:
            measurements[enabled] = _ingestion_once(
                data, enabled=enabled, k=k, seed=seed + trial
            )
        if measurements[False][1] != measurements[True][1]:
            raise AssertionError("native ingestion state diverged from pure Python")
        pure_rates.append(measurements[False][0])
        native_rates.append(measurements[True][0])
    return statistics.median(pure_rates), statistics.median(native_rates)


def _query_once(sk: KLL, *, enabled: bool, loops: int) -> float:
    set_native_enabled(enabled)
    sk.quantiles_at(QS)
    start = time.perf_counter_ns()
    for _ in range(loops):
        sk.quantiles_at(QS)
    return (time.perf_counter_ns() - start) / loops / 1e3


def _paired_query(
    pure_sk: KLL,
    native_sk: KLL,
    *,
    loops: int,
    trials: int,
) -> tuple[float, float]:
    if pure_sk.to_bytes() != native_sk.to_bytes():
        raise AssertionError("query fixtures do not have identical state")
    pure_latencies: list[float] = []
    native_latencies: list[float] = []
    sketches = {False: pure_sk, True: native_sk}
    for trial in range(trials):
        measurements: dict[bool, float] = {}
        order = (False, True) if trial % 2 == 0 else (True, False)
        for enabled in order:
            measurements[enabled] = _query_once(sketches[enabled], enabled=enabled, loops=loops)
        pure_latencies.append(measurements[False])
        native_latencies.append(measurements[True])
    return statistics.median(pure_latencies), statistics.median(native_latencies)


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


def _merge_once(
    sources: list[KLL],
    *,
    enabled: bool,
    k: int,
    seed: int,
    loops: int,
    expected_n: int,
) -> tuple[float, bytes]:
    set_native_enabled(enabled)
    destinations = [KLL(k, seed + i) for i in range(loops)]
    start = time.perf_counter_ns()
    for dst in destinations:
        for src in sources:
            dst.merge(src)
    elapsed_us = (time.perf_counter_ns() - start) / loops / 1e3
    if any(dst.n != expected_n for dst in destinations):
        raise AssertionError("merge produced incorrect represented mass")
    return elapsed_us, destinations[0].to_bytes()


def _paired_merge(
    pure_sources: list[KLL],
    native_sources: list[KLL],
    *,
    k: int,
    seed: int,
    loops: int,
    trials: int,
    expected_n: int,
) -> tuple[float, float]:
    if [s.to_bytes() for s in pure_sources] != [s.to_bytes() for s in native_sources]:
        raise AssertionError("merge source states diverged")
    pure_timings: list[float] = []
    native_timings: list[float] = []
    sources = {False: pure_sources, True: native_sources}
    for trial in range(trials):
        measurements: dict[bool, tuple[float, bytes]] = {}
        order = (False, True) if trial % 2 == 0 else (True, False)
        trial_seed = seed + 50_000 + trial * loops
        for enabled in order:
            measurements[enabled] = _merge_once(
                sources[enabled],
                enabled=enabled,
                k=k,
                seed=trial_seed,
                loops=loops,
                expected_n=expected_n,
            )
        if measurements[False][1] != measurements[True][1]:
            raise AssertionError("native merge state diverged from pure Python")
        pure_timings.append(measurements[False][0])
        native_timings.append(measurements[True][0])
    return statistics.median(pure_timings), statistics.median(native_timings)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=250_000)
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--seed", type=int, default=7331)
    p.add_argument("--shards", type=int, default=8)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--query-loops", type=int, default=2000)
    p.add_argument("--merge-loops", type=int, default=96)
    p.add_argument("--min-ingestion-speedup", type=float, default=50.0)
    p.add_argument("--min-query-speedup", type=float, default=5.0)
    p.add_argument("--min-merge-speedup", type=float, default=5.0)
    p.add_argument("--out", type=Path, default=Path("performance_regression.json"))
    args = p.parse_args()

    if args.N <= 0 or args.trials < 3 or args.query_loops <= 0 or args.merge_loops <= 0:
        raise SystemExit("N/query-loops/merge-loops must be positive and trials must be >= 3")
    if args.k < 40:
        raise SystemExit("k must be >= 40")
    if args.shards < 2 or args.shards > args.N:
        raise SystemExit("shards must be between 2 and N")
    if min(args.min_ingestion_speedup, args.min_query_speedup, args.min_merge_speedup) <= 0:
        raise SystemExit("performance thresholds must be positive")
    if not native_available():
        raise SystemExit("native extension is required")

    data = _values(args.N)
    info = native_backend_info()
    try:
        pure_ingest, native_ingest = _paired_ingestion(
            data, k=args.k, seed=args.seed, trials=args.trials
        )

        pure_query_sk = _build(data, enabled=False, k=args.k, seed=args.seed + 777)
        native_query_sk = _build(data, enabled=True, k=args.k, seed=args.seed + 777)
        pure_query, native_query = _paired_query(
            pure_query_sk,
            native_query_sk,
            loops=args.query_loops,
            trials=args.trials,
        )

        pure_sources = _sources(data, enabled=False, k=args.k, seed=args.seed, shards=args.shards)
        native_sources = _sources(data, enabled=True, k=args.k, seed=args.seed, shards=args.shards)
        pure_merge, native_merge = _paired_merge(
            pure_sources,
            native_sources,
            k=args.k,
            seed=args.seed,
            loops=args.merge_loops,
            trials=args.trials,
            expected_n=args.N,
        )
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
            "source_sha": _source_sha(),
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "ordering": "paired trials with alternating implementation order",
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
