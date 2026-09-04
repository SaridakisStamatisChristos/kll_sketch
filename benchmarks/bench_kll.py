#!/usr/bin/env python3
"""Reproducible KLL characterization and performance harness.

Accuracy is measured in normalized rank space, the metric KLL actually
controls. Value-space error is intentionally not used as a correctness gate.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Callable

import random

# Make the documented ``python benchmarks/bench_kll.py`` command work directly
# from a source checkout without requiring an editable install first.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kll_sketch import KLL


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default="bench_out")
    p.add_argument("--Ns", nargs="+", type=lambda x: int(float(x)), default=[100_000])
    p.add_argument("--capacities", nargs="+", type=int, default=[100, 200, 400, 800])
    p.add_argument("--distributions", nargs="+", default=["uniform", "normal", "exponential", "pareto", "bimodal", "duplicates"])
    p.add_argument("--qs", nargs="+", type=float, default=[.001,.01,.05,.1,.25,.5,.75,.9,.95,.99,.999])
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--shards", type=int, default=8)
    p.add_argument("--seed", type=int, default=7331)
    return p.parse_args()


def _seed(base: int, *parts: object) -> int:
    raw = "|".join(map(str, (base,) + parts)).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _uniform(r: random.Random, n: int) -> list[float]:
    return [r.random() for _ in range(n)]


def _normal(r: random.Random, n: int) -> list[float]:
    return [r.gauss(0.0, 1.0) for _ in range(n)]


def _exponential(r: random.Random, n: int) -> list[float]:
    return [r.expovariate(1.0) for _ in range(n)]


def _pareto(r: random.Random, n: int) -> list[float]:
    return [r.paretovariate(1.5) - 1.0 for _ in range(n)]


def _bimodal(r: random.Random, n: int) -> list[float]:
    out = [r.gauss(-2.0, 1.0) if i & 1 else r.gauss(2.0, .5) for i in range(n)]
    r.shuffle(out)
    return out


def _duplicates(r: random.Random, n: int) -> list[float]:
    return [float(r.randrange(32)) for _ in range(n)]


GENERATORS: dict[str, Callable[[random.Random, int], list[float]]] = {
    "uniform": _uniform,
    "normal": _normal,
    "exponential": _exponential,
    "pareto": _pareto,
    "bimodal": _bimodal,
    "duplicates": _duplicates,
}


def _rank_error(ordered: list[float], estimate: float, q: float) -> float:
    n = len(ordered)
    target = q * (n - 1)
    left = bisect.bisect_left(ordered, estimate)
    right = bisect.bisect_right(ordered, estimate) - 1
    if target < left:
        return (left - target) / n
    if target > right:
        return (target - right) / n
    return 0.0


def _write(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _args()
    unknown = sorted(set(args.distributions) - GENERATORS.keys())
    if unknown:
        raise SystemExit(f"unknown distributions: {', '.join(unknown)}")
    if args.trials <= 0 or args.shards <= 0:
        raise SystemExit("trials and shards must be positive")
    if any(not 0 <= q <= 1 for q in args.qs):
        raise SystemExit("all quantiles must be in [0,1]")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    accuracy: list[dict[str, object]] = []
    throughput: list[dict[str, object]] = []
    latency: list[dict[str, object]] = []
    merge: list[dict[str, object]] = []
    footprint: list[dict[str, object]] = []

    for dist in args.distributions:
        generator = GENERATORS[dist]
        for n in args.Ns:
            for trial in range(args.trials):
                data_seed = _seed(args.seed, dist, n, trial)
                data = generator(random.Random(data_seed), n)
                ordered = sorted(data)

                for k in args.capacities:
                    sketch_seed = _seed(args.seed, "sketch", dist, n, trial, k)
                    sketch = KLL(k, sketch_seed)
                    start = time.perf_counter_ns()
                    sketch.extend(data)
                    update_ns = time.perf_counter_ns() - start
                    sketch.validate()
                    updates_per_sec = n / (update_ns / 1e9) if update_ns else math.inf
                    throughput.append({
                        "distribution": dist, "N": n, "trial": trial, "k": k,
                        "updates_per_sec": updates_per_sec, "update_time_s": update_ns / 1e9,
                    })

                    errors = []
                    # Warm cache once; per-query measurements then represent steady-state queries.
                    sketch.quantiles_at(args.qs)
                    for q in args.qs:
                        t0 = time.perf_counter_ns()
                        estimate = sketch.quantile(q)
                        q_ns = time.perf_counter_ns() - t0
                        err = _rank_error(ordered, estimate, q)
                        errors.append(err)
                        latency.append({
                            "distribution": dist, "N": n, "trial": trial, "k": k,
                            "q": q, "latency_us": q_ns / 1e3,
                        })
                        accuracy.append({
                            "distribution": dist, "N": n, "trial": trial, "k": k,
                            "mode": "single", "q": q, "estimate": estimate,
                            "normalized_rank_error": err,
                            "model_99_error": sketch.normalized_rank_error(),
                        })

                    footprint.append({
                        "distribution": dist, "N": n, "trial": trial, "k": k,
                        "num_retained": sketch.num_retained,
                        "serialized_bytes": len(sketch.to_bytes()),
                        "levels": len(sketch._levels),
                        "max_rank_error": max(errors, default=0.0),
                    })

                    shards = []
                    for shard_idx in range(args.shards):
                        shard = KLL(k, _seed(sketch_seed, "shard", shard_idx))
                        shard.extend(data[shard_idx::args.shards])
                        shards.append(shard)
                    target = shards[0].copy()
                    t0 = time.perf_counter_ns()
                    for shard in shards[1:]:
                        target.merge(shard)
                    merge_ns = time.perf_counter_ns() - t0
                    target.validate()
                    merge.append({
                        "distribution": dist, "N": n, "trial": trial, "k": k,
                        "shards": args.shards, "merge_time_s": merge_ns / 1e9,
                        "num_retained": target.num_retained,
                    })
                    for q in args.qs:
                        estimate = target.quantile(q)
                        accuracy.append({
                            "distribution": dist, "N": n, "trial": trial, "k": k,
                            "mode": "merged", "q": q, "estimate": estimate,
                            "normalized_rank_error": _rank_error(ordered, estimate, q),
                            "model_99_error": target.normalized_rank_error(),
                        })

    _write(outdir / "accuracy_rank.csv", ["distribution","N","trial","k","mode","q","estimate","normalized_rank_error","model_99_error"], accuracy)
    _write(outdir / "update_throughput.csv", ["distribution","N","trial","k","updates_per_sec","update_time_s"], throughput)
    _write(outdir / "query_latency.csv", ["distribution","N","trial","k","q","latency_us"], latency)
    _write(outdir / "merge.csv", ["distribution","N","trial","k","shards","merge_time_s","num_retained"], merge)
    _write(outdir / "footprint.csv", ["distribution","N","trial","k","num_retained","serialized_bytes","levels","max_rank_error"], footprint)

    max_err = max(float(row["normalized_rank_error"]) for row in accuracy) if accuracy else 0.0
    p95_latency = statistics.quantiles([float(r["latency_us"]) for r in latency], n=20)[18] if len(latency) >= 20 else 0.0
    min_ups = min(float(row["updates_per_sec"]) for row in throughput) if throughput else math.inf
    print(f"max normalized rank error: {max_err:.6f}")
    print(f"minimum update throughput: {min_ups:,.0f} updates/s")
    print(f"steady-state query latency p95: {p95_latency:.2f} us")
    print(f"artifacts: {outdir}")


if __name__ == "__main__":
    main()
