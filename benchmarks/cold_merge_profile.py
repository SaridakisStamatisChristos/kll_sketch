from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from datasketches import kll_doubles_sketch

from kll_sketch import KLL, native_backend_info
from kll_sketch import _native as _native_impl


def pct(values: list[float], p: float) -> float:
    xs = sorted(values)
    if not xs:
        return float("nan")
    i = (len(xs) - 1) * p
    lo = int(i)
    hi = min(lo + 1, len(xs) - 1)
    f = i - lo
    return xs[lo] * (1.0 - f) + xs[hi] * f


def build_sources(n: int, k: int, seed: int, shards: int):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=n).astype(np.float64, copy=False)
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


def one_ours(sources, k: int, seed: int):
    dst = KLL(k, seed)
    positions = []
    t_all = time.perf_counter_ns()
    for src in sources:
        t0 = time.perf_counter_ns()
        dst.merge(src)
        positions.append((time.perf_counter_ns() - t0) / 1e3)
    total = (time.perf_counter_ns() - t_all) / 1e3
    return total, positions, dst.n


def one_apache(sources, k: int):
    dst = kll_doubles_sketch(k)
    positions = []
    t_all = time.perf_counter_ns()
    for src in sources:
        t0 = time.perf_counter_ns()
        dst.merge(src)
        positions.append((time.perf_counter_ns() - t0) / 1e3)
    total = (time.perf_counter_ns() - t_all) / 1e3
    return total, positions, dst.n


def level_sizes(sk: KLL) -> list[int]:
    capsule = sk._cache_prefix
    levels, _stats = _native_impl.state_export(capsule)
    return [len(level) for level in levels]


def structure_ours(sources, k: int, seed: int):
    """Untimed pass exposing exact merge/compaction level-shape transitions."""
    dst = KLL(k, seed)
    rows = []
    previous_compactions = 0
    for i, src in enumerate(sources, 1):
        src_sizes = level_sizes(src)
        if i == 1:
            pre_sizes = [0]
        else:
            pre_sizes = level_sizes(dst)
        dst.merge(src)
        capsule = dst._cache_prefix
        n, retained, rng_state, compactions, min_value, max_value, num_levels = _native_impl.state_stats(capsule)
        post_sizes = level_sizes(dst)
        compactions = int(compactions)
        rows.append({
            "position": i,
            "n": int(n),
            "retained": int(retained),
            "num_levels": int(num_levels),
            "compactions_total": compactions,
            "compactions_delta": compactions - previous_compactions,
            "dst_pre_level_sizes": pre_sizes,
            "src_level_sizes": src_sizes,
            "dst_post_level_sizes": post_sizes,
        })
        previous_compactions = compactions
    return rows


def summarize(xs: list[float]):
    return {
        "first_us": xs[0],
        "median_us": statistics.median(xs),
        "p10_us": pct(xs, 0.10),
        "p90_us": pct(xs, 0.90),
        "min_us": min(xs),
        "max_us": max(xs),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=250_000)
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--seed", type=int, default=7331)
    p.add_argument("--shards", type=int, default=8)
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--out", type=Path, default=Path("cold_merge_profile.json"))
    args = p.parse_args()

    info = native_backend_info()
    if not (info.get("available") and info.get("enabled")):
        raise SystemExit(f"native backend required: {info}")

    ours_sources, apache_sources = build_sources(args.N, args.k, args.seed, args.shards)
    ours_total: list[float] = []
    apache_total: list[float] = []
    ours_pos = [[] for _ in range(args.shards)]
    apache_pos = [[] for _ in range(args.shards)]

    for trial in range(args.trials):
        order = ("ours", "apache") if trial % 2 == 0 else ("apache", "ours")
        for which in order:
            if which == "ours":
                total, positions, n = one_ours(ours_sources, args.k, args.seed + 50_000 + trial)
                ours_total.append(total)
                for i, us in enumerate(positions):
                    ours_pos[i].append(us)
            else:
                total, positions, n = one_apache(apache_sources, args.k)
                apache_total.append(total)
                for i, us in enumerate(positions):
                    apache_pos[i].append(us)
            if n != args.N:
                raise AssertionError((which, n, args.N))

    result = {
        "environment": {"native": info, "N": args.N, "k": args.k, "shards": args.shards, "trials": args.trials},
        "ours_total": summarize(ours_total),
        "apache_total": summarize(apache_total),
        "ratio_median_speed": statistics.median(apache_total) / statistics.median(ours_total),
        "positions": [
            {
                "position": i + 1,
                "ours": summarize(ours_pos[i]),
                "apache": summarize(apache_pos[i]),
                "ratio_median_speed": statistics.median(apache_pos[i]) / statistics.median(ours_pos[i]),
            }
            for i in range(args.shards)
        ],
        "structure": structure_ours(ours_sources, args.k, args.seed + 90_000),
        "raw": {"ours_total_us": ours_total, "apache_total_us": apache_total},
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
