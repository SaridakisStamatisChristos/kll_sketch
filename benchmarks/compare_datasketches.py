#!/usr/bin/env python3
"""Optional differential/performance comparison with Apache DataSketches.

Install ``datasketches`` explicitly to run this file. It is deliberately not a
runtime dependency of kll-sketch.
"""
from __future__ import annotations

import argparse
import bisect
import random
import time

from kll_sketch import KLL

try:
    from datasketches import kll_floats_sketch
except ImportError as exc:  # pragma: no cover - optional tool
    raise SystemExit("install the optional 'datasketches' package to run this comparison") from exc


def rank_error(ordered: list[float], estimate: float, q: float) -> float:
    target = q * (len(ordered) - 1)
    lo = bisect.bisect_left(ordered, estimate)
    hi = bisect.bisect_right(ordered, estimate) - 1
    return max(lo-target, target-hi, 0.0) / len(ordered)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=1_000_000)
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--seed", type=int, default=7331)
    args = p.parse_args()
    rng = random.Random(args.seed)
    data = [rng.random() for _ in range(args.N)]
    ordered = sorted(data)
    qs = [.01,.1,.5,.9,.99]

    ours = KLL(args.k, args.seed)
    t0=time.perf_counter(); ours.extend(data); ours_t=time.perf_counter()-t0
    apache = kll_floats_sketch(args.k)
    t0=time.perf_counter()
    for x in data: apache.update(x)
    apache_t=time.perf_counter()-t0

    print("implementation,updates/s,retained,max_rank_error")
    print(f"kll-sketch,{args.N/ours_t:.0f},{ours.num_retained},{max(rank_error(ordered, ours.quantile(q), q) for q in qs):.6f}")
    print(f"datasketches,{args.N/apache_t:.0f},{apache.num_retained},{max(rank_error(ordered, apache.get_quantile(q), q) for q in qs):.6f}")


if __name__ == "__main__":
    main()
