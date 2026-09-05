#!/usr/bin/env python3
"""Robust fresh-destination merge benchmark: kll-sketch native vs Apache KLL.

The broad ecosystem benchmark intentionally keeps its trial count low because
pure-Python t-digest dominates wall time. This focused gate isolates the one
short operation that is most vulnerable to allocator/scheduler noise: merging
pre-built shards into a fresh destination.

Both implementations receive the same float64 shards and k. Destination
construction stays outside the timed region, implementation order alternates
per trial, and each timing is amplified across many fresh destinations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from datasketches import kll_doubles_sketch
from kll_sketch import KLL, native_backend_info


def normal_data(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed * 1315423911 + 2 * 2654435761)
    return np.ascontiguousarray(rng.normal(size=n), dtype=np.float64)


def build_sources(data: np.ndarray, k: int, seed: int, shards: int):
    pieces = [np.ascontiguousarray(x, dtype=np.float64) for x in np.array_split(data, shards)]
    ours = []
    apache = []
    for i, part in enumerate(pieces):
        a = KLL(k, seed + 1000 + i)
        a.extend(part)
        ours.append(a)

        b = kll_doubles_sketch(k)
        b.update(part)
        apache.append(b)
    return ours, apache


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    x = q * (len(ordered) - 1)
    lo = int(x)
    hi = min(lo + 1, len(ordered) - 1)
    frac = x - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median_us": statistics.median(values),
        "p10_us": percentile(values, 0.10),
        "p90_us": percentile(values, 0.90),
        "min_us": min(values),
        "max_us": max(values),
    }


def benchmark(
    data: np.ndarray,
    k: int,
    seed: int,
    shards: int,
    trials: int,
    loops: int,
) -> dict:
    ours_sources, apache_sources = build_sources(data, k, seed, shards)
    expected_n = len(data)
    ours_times: list[float] = []
    apache_times: list[float] = []
    paired_ratios: list[float] = []

    for trial in range(trials):
        ours_dsts = [KLL(k, seed + 90000 + trial * loops + i) for i in range(loops)]
        apache_dsts = [kll_doubles_sketch(k) for _ in range(loops)]
        order = ("ours", "apache") if trial % 2 == 0 else ("apache", "ours")
        measured: dict[str, float] = {}

        for which in order:
            if which == "ours":
                t0 = time.perf_counter_ns()
                for dst in ours_dsts:
                    for src in ours_sources:
                        dst.merge(src)
                measured["ours"] = (time.perf_counter_ns() - t0) / loops / 1e3
                if any(dst.n != expected_n for dst in ours_dsts):
                    raise AssertionError("kll-sketch cold merge produced wrong n")
            else:
                t0 = time.perf_counter_ns()
                for dst in apache_dsts:
                    for src in apache_sources:
                        dst.merge(src)
                measured["apache"] = (time.perf_counter_ns() - t0) / loops / 1e3
                if any(dst.n != expected_n for dst in apache_dsts):
                    raise AssertionError("Apache KLL cold merge produced wrong n")

        ours_times.append(measured["ours"])
        apache_times.append(measured["apache"])
        paired_ratios.append(measured["apache"] / measured["ours"])

    ours_summary = summarize(ours_times)
    apache_summary = summarize(apache_times)
    paired_wins = sum(o < a for o, a in zip(ours_times, apache_times))
    return {
        "ours": ours_summary,
        "apache": apache_summary,
        "paired": {
            "ours_faster_trials": paired_wins,
            "apache_faster_or_tied_trials": trials - paired_wins,
            "median_speed_ratio_ours_over_apache": statistics.median(paired_ratios),
            "p10_speed_ratio_ours_over_apache": percentile(paired_ratios, 0.10),
            "p90_speed_ratio_ours_over_apache": percentile(paired_ratios, 0.90),
        },
        "raw_us": {"ours": ours_times, "apache": apache_times},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=250_000)
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--seed", type=int, default=7331)
    p.add_argument("--shards", type=int, default=8)
    p.add_argument("--trials", type=int, default=31)
    p.add_argument("--loops", type=int, default=128)
    p.add_argument("--out", type=Path, default=Path("competitive_cold_merge.json"))
    args = p.parse_args()

    if args.trials < 3 or args.loops < 1 or args.shards < 2:
        raise SystemExit("require trials >= 3, loops >= 1, shards >= 2")

    info = native_backend_info()
    if not (info.get("available") and info.get("enabled") and info.get("persistent_state")):
        raise SystemExit(f"persistent native backend required: {info}")

    data = normal_data(args.N, args.seed)
    result = benchmark(data, args.k, args.seed, args.shards, args.trials, args.loops)
    result["environment"] = {
        "native": info,
        "N": args.N,
        "k": args.k,
        "seed": args.seed,
        "shards": args.shards,
        "trials": args.trials,
        "loops": args.loops,
        "destination_construction_timed": False,
        "implementation_order_alternates": True,
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
