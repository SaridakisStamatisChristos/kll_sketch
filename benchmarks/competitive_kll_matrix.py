#!/usr/bin/env python3
"""Reproducible Apache DataSketches KLL matrix and sharded-merge scaling benchmark."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from competitive_kll_focus import (
    QS,
    bench_merge,
    bench_one,
    data_for,
    geometric_mean,
)
from kll_sketch import __version__, native_backend_info

DEFAULT_DISTS = ("uniform", "normal", "duplicates")


def _positive(values: list[int], name: str) -> None:
    if not values or any(v <= 0 for v in values):
        raise SystemExit(f"{name} must contain positive integers")


def _source_sha() -> str | None:
    """Return the source commit rather than GitHub's synthetic PR merge SHA."""
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


def _environment(args: argparse.Namespace) -> dict:
    return {
        "project_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": importlib.metadata.version("numpy"),
        "datasketches": importlib.metadata.version("datasketches"),
        "native": native_backend_info(),
        "seed": args.seed,
        "trials": args.trials,
        "query_loops": args.query_loops,
        "merge_loops": args.merge_loops,
        "source_sha": _source_sha(),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
    }


def _summarize(rows: list[dict]) -> dict:
    return {
        "updates_per_s": statistics.median(r["updates_per_s"] for r in rows),
        "query_us": statistics.median(r["query_us"] for r in rows),
        "worst_rank_error": max(r["rank_error"] for r in rows),
        "size_bytes": statistics.median(r["size_bytes"] for r in rows),
    }


def _matrix_row(n: int, k: int, seed: int, trials: int, query_loops: int, dists: tuple[str, ...]) -> dict:
    ours_thr: list[float] = []
    apache_thr: list[float] = []
    ours_query: list[float] = []
    apache_query: list[float] = []
    ours_err: list[float] = []
    apache_err: list[float] = []
    ours_size: list[float] = []
    apache_size: list[float] = []
    per_distribution: dict[str, dict] = {}

    for name in dists:
        data = data_for(name, n, seed)
        raw = bench_one(data, k, seed, trials, query_loops)
        ours = _summarize(raw["ours"])
        apache = _summarize(raw["apache"])
        per_distribution[name] = {"ours": ours, "apache": apache}
        ours_thr.append(ours["updates_per_s"])
        apache_thr.append(apache["updates_per_s"])
        ours_query.append(ours["query_us"])
        apache_query.append(apache["query_us"])
        ours_err.append(ours["worst_rank_error"])
        apache_err.append(apache["worst_rank_error"])
        ours_size.append(ours["size_bytes"])
        apache_size.append(apache["size_bytes"])

    ours_ingest = geometric_mean(ours_thr)
    apache_ingest = geometric_mean(apache_thr)
    ours_query_med = statistics.median(ours_query)
    apache_query_med = statistics.median(apache_query)
    ours_size_med = statistics.median(ours_size)
    apache_size_med = statistics.median(apache_size)
    return {
        "N": n,
        "k": k,
        "distributions": list(dists),
        "ours_updates_per_s": ours_ingest,
        "apache_updates_per_s": apache_ingest,
        "ingestion_ratio_ours_over_apache": ours_ingest / apache_ingest,
        "ours_query_us": ours_query_med,
        "apache_query_us": apache_query_med,
        "query_speed_ratio_ours_over_apache": apache_query_med / ours_query_med,
        "ours_worst_rank_error": max(ours_err),
        "apache_worst_rank_error": max(apache_err),
        "ours_size_bytes": ours_size_med,
        "apache_size_bytes": apache_size_med,
        "size_ratio_ours_over_apache": ours_size_med / apache_size_med,
        "per_distribution": per_distribution,
    }


def _merge_row(n: int, k: int, shards: int, seed: int, trials: int, merge_loops: int) -> dict:
    data = data_for("normal", n, seed + 4242)
    raw = bench_merge(data, k, seed, shards, trials, merge_loops)
    ours = statistics.median(raw["ours"])
    apache = statistics.median(raw["apache"])
    return {
        "N": n,
        "k": k,
        "shards": shards,
        "ours_merge_us": ours,
        "apache_merge_us": apache,
        "merge_speed_ratio_ours_over_apache": apache / ours,
    }


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--Ns", nargs="+", type=int, default=[50_000, 250_000, 1_000_000])
    p.add_argument("--ks", nargs="+", type=int, default=[100, 200, 400, 800])
    p.add_argument("--shards", nargs="+", type=int, default=[2, 4, 8, 16, 32])
    p.add_argument("--merge-N", type=int, default=250_000)
    p.add_argument("--merge-k", type=int, default=200)
    p.add_argument("--distributions", nargs="+", default=list(DEFAULT_DISTS))
    p.add_argument("--seed", type=int, default=7331)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--query-loops", type=int, default=1000)
    p.add_argument("--merge-loops", type=int, default=96)
    p.add_argument("--outdir", type=Path, default=Path("benchmark_matrix"))
    args = p.parse_args()

    _positive(args.Ns, "Ns")
    _positive(args.ks, "ks")
    _positive(args.shards, "shards")
    if args.trials < 2 or args.query_loops <= 0 or args.merge_loops <= 0:
        raise SystemExit("trials must be >= 2 and loop counts must be positive")
    if any(s > args.merge_N for s in args.shards):
        raise SystemExit("shard count cannot exceed merge-N")

    info = native_backend_info()
    if not (info.get("available") and info.get("enabled")):
        raise SystemExit(f"native backend required: {info}")

    dists = tuple(args.distributions)
    allowed = {"uniform", "normal", "lognormal", "exponential", "pareto", "bimodal", "duplicates"}
    if not dists or any(d not in allowed for d in dists):
        raise SystemExit(f"unsupported distribution; choose from {sorted(allowed)}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    matrix_rows = [
        _matrix_row(n, k, args.seed, args.trials, args.query_loops, dists)
        for n in args.Ns
        for k in args.ks
    ]
    merge_rows = [
        _merge_row(args.merge_N, args.merge_k, shards, args.seed, args.trials, args.merge_loops)
        for shards in args.shards
    ]
    payload = {
        "schema": 1,
        "environment": _environment(args),
        "methodology": {
            "peer": "Apache DataSketches kll_doubles_sketch",
            "quantiles": list(QS),
            "matrix_distributions": list(dists),
            "ordering": "paired same-process measurements; implementation order alternates inside focused harness",
            "claims": "runner/workload characterization only; no portable performance guarantee",
        },
        "matrix": matrix_rows,
        "merge_scaling": merge_rows,
    }
    (args.outdir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(
        args.outdir / "matrix.csv",
        matrix_rows,
        [
            "N", "k", "ours_updates_per_s", "apache_updates_per_s",
            "ingestion_ratio_ours_over_apache", "ours_query_us", "apache_query_us",
            "query_speed_ratio_ours_over_apache", "ours_worst_rank_error",
            "apache_worst_rank_error", "ours_size_bytes", "apache_size_bytes",
            "size_ratio_ours_over_apache",
        ],
    )
    _write_csv(
        args.outdir / "merge_scaling.csv",
        merge_rows,
        [
            "N", "k", "shards", "ours_merge_us", "apache_merge_us",
            "merge_speed_ratio_ours_over_apache",
        ],
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
