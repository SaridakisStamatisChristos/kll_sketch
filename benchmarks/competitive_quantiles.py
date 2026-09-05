#!/usr/bin/env python3
"""Competitive quantile-sketch benchmark.

This harness deliberately compares public, ready-to-query APIs rather than
internal kernels.  KLL implementations are parameter-matched at k=200 by
default.  DDSketch uses its relative-accuracy parameter, so its accuracy model
is different and both rank-space and value-space errors are reported.

The benchmark is intended for reproducible characterization, not a universal
ranking.  Data generation and input-format conversion are outside timed update
sections.  Native/vectorized bulk APIs are used when the library exposes them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from kll_sketch import KLL, native_backend_info
from datasketches import kll_doubles_sketch, tdigest_double
from ddsketch.ddsketch import DDSketch

try:
    from tdigest import TDigest
except Exception:  # optional legacy package
    TDigest = None  # type: ignore[assignment,misc]


DEFAULT_QS = (0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999)


def stable_seed(base: int, label: str) -> int:
    digest = hashlib.sha256(f"{base}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def generate_distribution(name: str, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(seed, name))
    if name == "uniform":
        x = rng.random(n)
    elif name == "normal":
        x = rng.normal(0.0, 1.0, n)
    elif name == "lognormal":
        x = rng.lognormal(0.0, 1.25, n)
    elif name == "exponential":
        x = rng.exponential(1.0, n)
    elif name == "pareto":
        x = rng.pareto(2.0, n) + 1.0
    elif name == "bimodal":
        choose = rng.random(n) < 0.55
        x = np.empty(n, dtype=np.float64)
        x[choose] = rng.normal(-2.0, 0.45, int(choose.sum()))
        x[~choose] = rng.normal(3.0, 1.1, int((~choose).sum()))
    elif name == "duplicates":
        values = np.array([-10.0, -1.0, 0.0, 0.0, 0.0, 1.0, 10.0], dtype=np.float64)
        x = rng.choice(values, size=n, replace=True)
    else:
        raise ValueError(f"unknown distribution: {name}")
    return np.ascontiguousarray(x, dtype=np.float64)


def exact_order_quantiles(ordered: np.ndarray, qs: tuple[float, ...]) -> list[float]:
    n = len(ordered)
    return [float(ordered[int(q * (n - 1))]) for q in qs]


def rank_error(ordered: np.ndarray, estimate: float, q: float) -> float:
    n = len(ordered)
    target = q * (n - 1)
    lo = int(np.searchsorted(ordered, estimate, side="left"))
    hi = int(np.searchsorted(ordered, estimate, side="right")) - 1
    return max(lo - target, target - hi, 0.0) / n


def accuracy_metrics(
    ordered: np.ndarray,
    estimates: list[float],
    qs: tuple[float, ...],
) -> tuple[float, float, float, float]:
    exact = exact_order_quantiles(ordered, qs)
    rank_errors = [rank_error(ordered, e, q) for e, q in zip(estimates, qs)]
    data_range = max(float(ordered[-1] - ordered[0]), 1e-300)
    range_errors = [abs(e - x) / data_range for e, x in zip(estimates, exact)]
    tail_rel = [
        abs(e - x) / max(abs(x), 1e-300)
        for e, x, q in zip(estimates, exact, qs)
        if q >= 0.99 and abs(x) > 1e-12
    ]
    return (
        max(rank_errors),
        statistics.fmean(rank_errors),
        max(range_errors),
        max(tail_rel) if tail_rel else math.nan,
    )


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


class Impl:
    def __init__(
        self,
        name: str,
        params: str,
        build: Callable[[np.ndarray, list[float], int], Any],
        quantiles: Callable[[Any, tuple[float, ...]], list[float]],
        empty: Callable[[int], Any],
        merge: Callable[[Any, Any], None],
        size: Callable[[Any], tuple[int | None, str]],
        retained: Callable[[Any], int | None],
    ) -> None:
        self.name = name
        self.params = params
        self.build = build
        self.quantiles = quantiles
        self.empty = empty
        self.merge = merge
        self.size = size
        self.retained = retained


def implementations(k: int, seed: int, dd_rel: float) -> list[Impl]:
    def ours_build(arr: np.ndarray, _lst: list[float], trial: int) -> KLL:
        sk = KLL(k, seed + trial)
        sk.extend(arr)
        return sk

    def ours_empty(trial: int) -> KLL:
        return KLL(k, seed + 100_000 + trial)

    def apkll_build(arr: np.ndarray, _lst: list[float], _trial: int):
        sk = kll_doubles_sketch(k)
        sk.update(arr)
        return sk

    def apkll_empty(_trial: int):
        return kll_doubles_sketch(k)

    def aptd_build(arr: np.ndarray, _lst: list[float], _trial: int):
        sk = tdigest_double(k)
        sk.update(arr)
        sk.compress()
        return sk

    def aptd_empty(_trial: int):
        return tdigest_double(k)

    def dd_build(_arr: np.ndarray, lst: list[float], _trial: int):
        sk = DDSketch(relative_accuracy=dd_rel)
        for x in lst:
            sk.add(x)
        return sk

    def dd_empty(_trial: int):
        return DDSketch(relative_accuracy=dd_rel)

    impls = [
        Impl(
            "kll-sketch-native",
            f"k={k}",
            ours_build,
            lambda sk, qs: [float(x) for x in sk.quantiles_at(qs)],
            ours_empty,
            lambda dst, src: dst.merge(src),
            lambda sk: (len(sk.to_bytes()), "official-kll2"),
            lambda sk: int(sk.num_retained),
        ),
        Impl(
            "apache-kll-double",
            f"k={k}",
            apkll_build,
            lambda sk, qs: [float(x) for x in sk.get_quantiles(list(qs))],
            apkll_empty,
            lambda dst, src: dst.merge(src),
            lambda sk: (len(sk.serialize()), "official"),
            lambda sk: int(sk.num_retained),
        ),
        Impl(
            "apache-tdigest-double",
            f"k={k}",
            aptd_build,
            lambda sk, qs: [float(sk.get_quantile(q)) for q in qs],
            aptd_empty,
            lambda dst, src: dst.merge(src),
            lambda sk: (len(sk.serialize()), "official"),
            lambda _sk: None,
        ),
        Impl(
            "datadog-ddsketch",
            f"relative_accuracy={dd_rel}",
            dd_build,
            lambda sk, qs: [float(sk.get_quantile_value(q)) for q in qs],
            dd_empty,
            lambda dst, src: dst.merge(src),
            lambda _sk: (None, "n/a"),
            lambda _sk: None,
        ),
    ]

    if TDigest is not None:
        def pytd_build(_arr: np.ndarray, lst: list[float], _trial: int):
            sk = TDigest()
            sk.batch_update(lst)
            return sk

        def pytd_empty(_trial: int):
            return TDigest()

        def pytd_merge(dst, src) -> None:
            dst.update_centroids_from_list(src.centroids_to_list())

        impls.append(
            Impl(
                "python-tdigest",
                "default delta=0.01,K=25",
                pytd_build,
                lambda sk, qs: [float(sk.percentile(q * 100.0)) for q in qs],
                pytd_empty,
                pytd_merge,
                lambda _sk: (None, "n/a"),
                lambda sk: len(sk.centroids_to_list()),
            )
        )
    return impls


def geometric_mean(values: list[float]) -> float:
    vals = [v for v in values if v > 0 and math.isfinite(v)]
    return math.exp(statistics.fmean(math.log(v) for v in vals)) if vals else math.nan


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def write_summary(rows: list[dict[str, Any]], merge_rows: list[dict[str, Any]], outdir: Path) -> str:
    names = sorted({r["implementation"] for r in rows})
    lines = [
        "# Competitive quantile benchmark",
        "",
        "Results are characterization of one GitHub Actions runner, not portable performance guarantees.",
        "KLL and Apache t-digest use k=200 by default; DDSketch uses relative_accuracy=0.01.",
        "Serialized bytes are compared only where each library exposes an official serializer.",
        "",
        "| implementation | geo-mean updates/s | worst rank err | median rank err | median query set us | median official bytes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    summary: dict[str, dict[str, Any]] = {}
    for name in names:
        rs = [r for r in rows if r["implementation"] == name]
        # Per-distribution medians avoid overweighting repeated timing trials.
        dists = sorted({r["distribution"] for r in rs})
        per_dist_thr = [statistics.median([r["updates_per_s"] for r in rs if r["distribution"] == d]) for d in dists]
        rank_vals = [r["max_rank_error"] for r in rs]
        query_vals = [r["query_set_us"] for r in rs]
        official_sizes = [r["size_bytes"] for r in rs if r["size_bytes"] is not None and r["size_basis"].startswith("official")]
        summary[name] = {
            "geo_mean_updates_per_s": geometric_mean(per_dist_thr),
            "worst_rank_error": max(rank_vals),
            "median_rank_error": statistics.median(rank_vals),
            "median_query_set_us": statistics.median(query_vals),
            "median_official_size_bytes": statistics.median(official_sizes) if official_sizes else None,
        }
        s = summary[name]
        lines.append(
            f"| {name} | {s['geo_mean_updates_per_s']:,.0f} | {s['worst_rank_error']:.6f} | "
            f"{s['median_rank_error']:.6f} | {s['median_query_set_us']:.2f} | "
            f"{fmt(s['median_official_size_bytes'], 0)} |"
        )

    lines.extend([
        "",
        "## Merge benchmark (normal distribution, pre-built shards)",
        "",
        "| implementation | median merge us | represented items | merge items/s |",
        "|---|---:|---:|---:|",
    ])
    for name in sorted({r["implementation"] for r in merge_rows}):
        rs = [r for r in merge_rows if r["implementation"] == name]
        med_us = statistics.median(r["merge_us"] for r in rs)
        items = int(rs[0]["n"])
        lines.append(f"| {name} | {med_us:.2f} | {items:,} | {items / (med_us / 1e6):,.0f} |")

    lines.extend([
        "",
        "## Interpretation guardrails",
        "",
        "- `kll-sketch-native` and `apache-kll-double` are the closest algorithm/parameter match.",
        "- Apache t-digest prioritizes tail behavior and has a different error profile.",
        "- DDSketch provides relative value-error guarantees rather than KLL-style normalized-rank guarantees.",
        "- Standalone `python-tdigest`, when installable, is a pure-Python package and is included as an ecosystem reference, not a native peer.",
        "- Update timings exclude random-data generation and list/array conversion.",
    ])
    text = "\n".join(lines) + "\n"
    (outdir / "summary.md").write_text(text)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=250_000)
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--seed", type=int, default=7331)
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--query-loops", type=int, default=500)
    p.add_argument("--shards", type=int, default=8)
    p.add_argument("--dd-relative-accuracy", type=float, default=0.01)
    p.add_argument("--outdir", type=Path, default=Path("competitive_out"))
    p.add_argument(
        "--distributions",
        nargs="+",
        default=["uniform", "normal", "lognormal", "exponential", "pareto", "bimodal", "duplicates"],
    )
    args = p.parse_args()
    if args.N < 1000 or args.trials < 1 or args.query_loops < 1 or args.shards < 2:
        raise SystemExit("invalid benchmark dimensions")

    args.outdir.mkdir(parents=True, exist_ok=True)
    qs = DEFAULT_QS
    impls = implementations(args.k, args.seed, args.dd_relative_accuracy)
    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": package_version("numpy"),
        "datasketches": package_version("datasketches"),
        "ddsketch": package_version("ddsketch"),
        "tdigest": package_version("tdigest"),
        "native_backend": native_backend_info(),
        "N": args.N,
        "k": args.k,
        "seed": args.seed,
        "trials": args.trials,
        "query_loops": args.query_loops,
        "shards": args.shards,
        "quantiles": qs,
        "distributions": args.distributions,
        "implementations": {i.name: i.params for i in impls},
    }
    (args.outdir / "environment.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))

    rows: list[dict[str, Any]] = []
    normal_data: np.ndarray | None = None
    for dist in args.distributions:
        data = generate_distribution(dist, args.N, args.seed)
        if dist == "normal":
            normal_data = data
        data_list = data.tolist()  # conversion deliberately outside timed update sections
        ordered = np.sort(data)
        print(f"\n[{dist}] N={len(data):,}")
        for impl in impls:
            for trial in range(args.trials):
                t0 = time.perf_counter()
                sk = impl.build(data, data_list, trial)
                build_s = time.perf_counter() - t0
                estimates = impl.quantiles(sk, qs)
                if len(estimates) != len(qs) or not all(math.isfinite(x) for x in estimates):
                    raise RuntimeError(f"{impl.name} produced invalid quantiles on {dist}")
                max_rank, mean_rank, max_range, max_tail_rel = accuracy_metrics(ordered, estimates, qs)

                t0 = time.perf_counter()
                for _ in range(args.query_loops):
                    impl.quantiles(sk, qs)
                query_set_us = (time.perf_counter() - t0) * 1e6 / args.query_loops
                size_bytes, size_basis = impl.size(sk)
                retained = impl.retained(sk)
                row = {
                    "implementation": impl.name,
                    "parameters": impl.params,
                    "distribution": dist,
                    "trial": trial,
                    "N": args.N,
                    "build_s": build_s,
                    "updates_per_s": args.N / build_s,
                    "max_rank_error": max_rank,
                    "mean_rank_error": mean_rank,
                    "max_range_normalized_value_error": max_range,
                    "max_tail_relative_value_error": max_tail_rel,
                    "query_set_us": query_set_us,
                    "size_bytes": size_bytes,
                    "size_basis": size_basis,
                    "retained": retained,
                }
                rows.append(row)
                print(
                    f"  {impl.name:24s} trial={trial} "
                    f"{row['updates_per_s']:>12,.0f}/s rank={max_rank:.6f} "
                    f"qset={query_set_us:8.2f}us size={size_bytes if size_bytes is not None else 'n/a'}"
                )

    if normal_data is None:
        normal_data = generate_distribution("normal", args.N, args.seed)
    shards_np = [np.ascontiguousarray(x, dtype=np.float64) for x in np.array_split(normal_data, args.shards)]
    shards_list = [x.tolist() for x in shards_np]
    merge_rows: list[dict[str, Any]] = []
    print("\n[merge] normal distribution, pre-built shards")
    for impl in impls:
        source_sketches = [impl.build(a, b, 10_000 + idx) for idx, (a, b) in enumerate(zip(shards_np, shards_list))]
        for trial in range(args.trials):
            dst = impl.empty(20_000 + trial)
            t0 = time.perf_counter()
            for src in source_sketches:
                impl.merge(dst, src)
            merge_us = (time.perf_counter() - t0) * 1e6
            estimates = impl.quantiles(dst, qs)
            if not all(math.isfinite(x) for x in estimates):
                raise RuntimeError(f"{impl.name} merge produced invalid quantiles")
            merge_rows.append({"implementation": impl.name, "trial": trial, "merge_us": merge_us, "n": args.N})
            print(f"  {impl.name:24s} trial={trial} {merge_us:12.2f} us")

    with (args.outdir / "results.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (args.outdir / "merge.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(merge_rows[0].keys()))
        writer.writeheader()
        writer.writerows(merge_rows)

    summary = write_summary(rows, merge_rows, args.outdir)
    print("\n" + summary)


if __name__ == "__main__":
    main()
