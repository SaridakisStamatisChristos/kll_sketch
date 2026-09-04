#!/usr/bin/env python3
"""Validate KLL benchmark artifacts with rank-space and regression gates."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"missing benchmark artifact: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("outdir", nargs="?", default="bench_out")
    p.add_argument("--summary", default="bench_summary.md")
    p.add_argument("--throughput-floor", type=float, default=50_000.0)
    p.add_argument("--query-p95-us", type=float, default=100.0)
    p.add_argument("--model-multiplier", type=float, default=2.0)
    args = p.parse_args()
    root = Path(args.outdir)

    accuracy = _rows(root / "accuracy_rank.csv")
    throughput = _rows(root / "update_throughput.csv")
    latency = _rows(root / "query_latency.csv")
    merge = _rows(root / "merge.csv")
    footprint = _rows(root / "footprint.csv")

    worst_ratio = 0.0
    worst_error = 0.0
    for row in accuracy:
        err = float(row["normalized_rank_error"])
        model = float(row["model_99_error"])
        worst_error = max(worst_error, err)
        worst_ratio = max(worst_ratio, err / model if model else 0.0)

    min_ups = min((float(r["updates_per_sec"]) for r in throughput), default=float("inf"))
    latencies = sorted(float(r["latency_us"]) for r in latency)
    p95 = latencies[min(len(latencies)-1, int(.95 * len(latencies)))] if latencies else 0.0
    max_merge = max((float(r["merge_time_s"]) for r in merge), default=0.0)
    over_capacity = [r for r in footprint if int(r["num_retained"]) <= 0 and int(r["N"]) > 0]

    checks = [
        ("Rank error / empirical model", worst_ratio <= args.model_multiplier, f"{worst_ratio:.3f}x", f"<= {args.model_multiplier:.2f}x"),
        ("Update throughput", min_ups >= args.throughput_floor, f"{min_ups:,.0f}/s", f">= {args.throughput_floor:,.0f}/s"),
        ("Cached query p95", p95 <= args.query_p95_us, f"{p95:.2f} us", f"<= {args.query_p95_us:.2f} us"),
        ("Merge smoke", max_merge <= 2.0, f"{max_merge:.3f} s", "<= 2.0 s"),
        ("Footprint sanity", not over_capacity, "OK" if not over_capacity else "invalid", "retained > 0"),
    ]

    lines = ["# Benchmark validation summary", "", f"Worst normalized rank error: **{worst_error:.6f}**", "", "| Gate | Observed | Threshold | Status |", "|---|---:|---:|:---:|"]
    for name, ok, observed, threshold in checks:
        lines.append(f"| {name} | {observed} | {threshold} | {'PASS' if ok else 'FAIL'} |")
    summary = "\n".join(lines) + "\n"
    path = root / args.summary
    path.write_text(summary, encoding="utf-8")
    print(summary)
    if not all(ok for _, ok, _, _ in checks):
        raise SystemExit("benchmark regression detected")


if __name__ == "__main__":
    main()
