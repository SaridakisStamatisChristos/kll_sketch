#!/usr/bin/env python3
"""Fast apples-to-apples public-API benchmark: kll-sketch native vs Apache KLL.

This exists for optimization and regression work. The broader
``competitive_quantiles.py`` remains the multi-library benchmark. Both KLL
implementations use the same ``k``, input arrays, quantile set, and process.

Short merge timings are amplified over many pre-created destinations so a
single scheduler interruption on a shared CI runner cannot dominate the
headline merge result. A position-by-position merge diagnostic is also emitted
to separate first-merge/bootstrap cost from later compaction cost.
"""
from __future__ import annotations

import argparse
import json
import math
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

QS = (0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999)
DISTS = ("uniform", "normal", "lognormal", "exponential", "pareto", "bimodal", "duplicates")


def data_for(name: str, n: int, seed: int) -> np.ndarray:
    # Stable independent streams without Python's randomized hash().
    labels = {name: i + 1 for i, name in enumerate(DISTS)}
    rng = np.random.default_rng(seed * 1315423911 + labels[name] * 2654435761)
    if name == "uniform":
        x = rng.random(n)
    elif name == "normal":
        x = rng.normal(size=n)
    elif name == "lognormal":
        x = rng.lognormal(0.0, 1.25, n)
    elif name == "exponential":
        x = rng.exponential(size=n)
    elif name == "pareto":
        x = rng.pareto(2.0, n) + 1.0
    elif name == "bimodal":
        pick = rng.random(n) < 0.55
        x = np.empty(n, dtype=np.float64)
        x[pick] = rng.normal(-2.0, 0.45, int(pick.sum()))
        x[~pick] = rng.normal(3.0, 1.1, int((~pick).sum()))
    elif name == "duplicates":
        x = rng.choice(np.array([-10., -1., 0., 0., 0., 1., 10.]), n)
    else:
        raise ValueError(name)
    return np.ascontiguousarray(x, dtype=np.float64)


def rank_error(ordered: np.ndarray, value: float, q: float) -> float:
    target = q * (len(ordered) - 1)
    lo = int(np.searchsorted(ordered, value, side="left"))
    hi = int(np.searchsorted(ordered, value, side="right")) - 1
    return max(lo - target, target - hi, 0.0) / len(ordered)


def geometric_mean(xs: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(x) for x in xs))


def bench_one(data: np.ndarray, k: int, seed: int, trials: int, query_loops: int):
    ordered = np.sort(data)
    out = {"ours": [], "apache": []}
    for trial in range(trials):
        # Alternate order to reduce systematic thermal/scheduler bias.
        order = ("ours", "apache") if trial % 2 == 0 else ("apache", "ours")
        for which in order:
            if which == "ours":
                t0 = time.perf_counter_ns()
                sk = KLL(k, seed + trial)
                sk.extend(data)
                elapsed = (time.perf_counter_ns() - t0) / 1e9
                estimates = [float(x) for x in sk.quantiles_at(QS)]
                t0 = time.perf_counter_ns()
                for _ in range(query_loops):
                    sk.quantiles_at(QS)
                query_us = (time.perf_counter_ns() - t0) / query_loops / 1e3
                size = len(sk.to_bytes())
            else:
                t0 = time.perf_counter_ns()
                sk = kll_doubles_sketch(k)
                sk.update(data)
                elapsed = (time.perf_counter_ns() - t0) / 1e9
                estimates = [float(x) for x in sk.get_quantiles(list(QS))]
                t0 = time.perf_counter_ns()
                for _ in range(query_loops):
                    sk.get_quantiles(list(QS))
                query_us = (time.perf_counter_ns() - t0) / query_loops / 1e3
                size = len(sk.serialize())
            err = max(rank_error(ordered, x, q) for x, q in zip(estimates, QS))
            out[which].append({
                "updates_per_s": len(data) / elapsed,
                "query_us": query_us,
                "rank_error": err,
                "size_bytes": size,
            })
    return out


def _merge_sources(data: np.ndarray, k: int, seed: int, shards: int):
    pieces = [np.ascontiguousarray(x, dtype=np.float64) for x in np.array_split(data, shards)]
    ours_sources = []
    apache_sources = []
    for i, part in enumerate(pieces):
        a = KLL(k, seed + 1000 + i)
        a.extend(part)
        ours_sources.append(a)
        b = kll_doubles_sketch(k)
        b.update(part)
        apache_sources.append(b)
    return ours_sources, apache_sources


def bench_merge(
    data: np.ndarray,
    k: int,
    seed: int,
    shards: int,
    trials: int,
    merge_loops: int,
):
    ours_sources, apache_sources = _merge_sources(data, k, seed, shards)
    expected_n = len(data)
    result = {"ours": [], "apache": []}
    for trial in range(trials):
        # Destination construction is deliberately outside the timed region.
        # A range of seeds samples different destination compaction-bit streams.
        ours_dsts = [KLL(k, seed + 9000 + trial * merge_loops + i) for i in range(merge_loops)]
        apache_dsts = [kll_doubles_sketch(k) for _ in range(merge_loops)]

        order = ("ours", "apache") if trial % 2 == 0 else ("apache", "ours")
        for which in order:
            if which == "ours":
                t0 = time.perf_counter_ns()
                for dst in ours_dsts:
                    for src in ours_sources:
                        dst.merge(src)
                us = (time.perf_counter_ns() - t0) / merge_loops / 1e3
                if any(dst.n != expected_n for dst in ours_dsts):
                    raise AssertionError("kll-sketch merge produced wrong n")
            else:
                t0 = time.perf_counter_ns()
                for dst in apache_dsts:
                    for src in apache_sources:
                        dst.merge(src)
                us = (time.perf_counter_ns() - t0) / merge_loops / 1e3
                if any(dst.n != expected_n for dst in apache_dsts):
                    raise AssertionError("Apache KLL merge produced wrong n")
            result[which].append(us)
    return result


def bench_merge_positions(
    data: np.ndarray,
    k: int,
    seed: int,
    shards: int,
    merge_loops: int,
):
    """Measure each source position across many destinations.

    This is diagnostic rather than a headline score.  Destination construction
    remains outside timing, and every position is measured after all prior
    positions have been applied to the same destinations.  That exposes whether
    the gap is bootstrap/state creation or later KLL compaction work.
    """
    ours_sources, apache_sources = _merge_sources(data, k, seed + 31337, shards)
    ours_dsts = [KLL(k, seed + 70000 + i) for i in range(merge_loops)]
    apache_dsts = [kll_doubles_sketch(k) for _ in range(merge_loops)]
    result = {"ours_us": [], "apache_us": [], "ratio_ours_over_apache_speed": []}

    for ours_src, apache_src in zip(ours_sources, apache_sources):
        # Alternate implementation order by position to reduce systematic bias.
        position = len(result["ours_us"])
        order = ("ours", "apache") if position % 2 == 0 else ("apache", "ours")
        measured: dict[str, float] = {}
        for which in order:
            if which == "ours":
                t0 = time.perf_counter_ns()
                for dst in ours_dsts:
                    dst.merge(ours_src)
                measured["ours"] = (time.perf_counter_ns() - t0) / merge_loops / 1e3
            else:
                t0 = time.perf_counter_ns()
                for dst in apache_dsts:
                    dst.merge(apache_src)
                measured["apache"] = (time.perf_counter_ns() - t0) / merge_loops / 1e3
        result["ours_us"].append(measured["ours"])
        result["apache_us"].append(measured["apache"])
        result["ratio_ours_over_apache_speed"].append(measured["apache"] / measured["ours"])

    expected_n = len(data)
    if any(dst.n != expected_n for dst in ours_dsts):
        raise AssertionError("position-profile kll-sketch merge produced wrong n")
    if any(dst.n != expected_n for dst in apache_dsts):
        raise AssertionError("position-profile Apache merge produced wrong n")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=250_000)
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--seed", type=int, default=7331)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--query-loops", type=int, default=2000)
    p.add_argument("--shards", type=int, default=8)
    p.add_argument("--merge-loops", type=int, default=200)
    p.add_argument("--out", type=Path, default=Path("competitive_focus.json"))
    args = p.parse_args()

    if args.merge_loops < 1:
        raise SystemExit("--merge-loops must be >= 1")

    info = native_backend_info()
    if not (info.get("available") and info.get("enabled")):
        raise SystemExit(f"native backend required: {info}")

    per_dist: dict[str, dict] = {}
    ours_thr: list[float] = []
    apache_thr: list[float] = []
    ours_query: list[float] = []
    apache_query: list[float] = []
    ours_err: list[float] = []
    apache_err: list[float] = []
    ours_sizes: list[float] = []
    apache_sizes: list[float] = []
    normal = None
    for name in DISTS:
        data = data_for(name, args.N, args.seed)
        if name == "normal":
            normal = data
        r = bench_one(data, args.k, args.seed, args.trials, args.query_loops)
        summary = {}
        for which in ("ours", "apache"):
            rows = r[which]
            summary[which] = {
                "updates_per_s": statistics.median(x["updates_per_s"] for x in rows),
                "query_us": statistics.median(x["query_us"] for x in rows),
                "worst_rank_error": max(x["rank_error"] for x in rows),
                "size_bytes": statistics.median(x["size_bytes"] for x in rows),
            }
        per_dist[name] = summary
        ours_thr.append(summary["ours"]["updates_per_s"])
        apache_thr.append(summary["apache"]["updates_per_s"])
        ours_query.append(summary["ours"]["query_us"])
        apache_query.append(summary["apache"]["query_us"])
        ours_err.append(summary["ours"]["worst_rank_error"])
        apache_err.append(summary["apache"]["worst_rank_error"])
        ours_sizes.append(summary["ours"]["size_bytes"])
        apache_sizes.append(summary["apache"]["size_bytes"])

    assert normal is not None
    merge = bench_merge(normal, args.k, args.seed, args.shards, args.trials, args.merge_loops)
    merge_positions = bench_merge_positions(normal, args.k, args.seed, args.shards, args.merge_loops)
    result = {
        "environment": {
            "native": info,
            "N": args.N,
            "k": args.k,
            "trials": args.trials,
            "query_loops": args.query_loops,
            "shards": args.shards,
            "merge_loops": args.merge_loops,
        },
        "ours": {
            "geo_mean_updates_per_s": geometric_mean(ours_thr),
            "median_query_us": statistics.median(ours_query),
            "median_merge_us": statistics.median(merge["ours"]),
            "worst_rank_error": max(ours_err),
            "median_size_bytes": statistics.median(ours_sizes),
        },
        "apache": {
            "geo_mean_updates_per_s": geometric_mean(apache_thr),
            "median_query_us": statistics.median(apache_query),
            "median_merge_us": statistics.median(merge["apache"]),
            "worst_rank_error": max(apache_err),
            "median_size_bytes": statistics.median(apache_sizes),
        },
        "per_distribution": per_dist,
        "diagnostics": {"merge_position": merge_positions},
    }
    result["ratios_ours_over_apache"] = {
        "ingestion": result["ours"]["geo_mean_updates_per_s"] / result["apache"]["geo_mean_updates_per_s"],
        "query_speed": result["apache"]["median_query_us"] / result["ours"]["median_query_us"],
        "merge_speed": result["apache"]["median_merge_us"] / result["ours"]["median_merge_us"],
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
