#!/usr/bin/env python3
"""Benchmark runner for the local kll_sketch implementation."""

from __future__ import annotations

import argparse
import importlib
import hashlib
import math
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", default="kll_sketch", help="Module that exports the sketch class")
    parser.add_argument("--class", dest="cls", default="KLLSketch", help="Sketch class name inside the module")
    parser.add_argument("--outdir", default="bench_out", help="Directory for benchmark CSV outputs")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed for reproducibility")
    parser.add_argument("--Ns", nargs="+", default=["1e5", "1e6"], help="Population sizes to benchmark")
    parser.add_argument(
        "--capacities", nargs="+", default=["200", "400", "800"], help="Sketch capacities to benchmark"
    )
    parser.add_argument(
        "--distributions",
        nargs="+",
        default=["uniform", "normal", "exponential", "pareto", "bimodal"],
        help="Synthetic data distributions to sample",
    )
    parser.add_argument(
        "--qs",
        nargs="+",
        default=["0.01", "0.05", "0.1", "0.25", "0.5", "0.75", "0.9", "0.95", "0.99"],
        help="Quantiles to evaluate",
    )
    parser.add_argument("--shards", type=int, default=8, help="Number of shards for the merge benchmark")
    return parser.parse_args()


def _to_int_list(values: Iterable[str]) -> List[int]:
    return [int(float(v)) for v in values]


def _to_float_list(values: Iterable[str]) -> List[float]:
    return [float(v) for v in values]


def _hash_seed(seed: int, *parts: object) -> int:
    material = "::".join(str(p) for p in (seed,) + parts)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _uniform(rng: np.random.Generator, size: int) -> np.ndarray:
    return rng.uniform(0.0, 1.0, size)


def _normal(rng: np.random.Generator, size: int) -> np.ndarray:
    return rng.normal(0.0, 1.0, size)


def _exponential(rng: np.random.Generator, size: int) -> np.ndarray:
    return rng.exponential(scale=1.0, size=size)


def _pareto(rng: np.random.Generator, size: int) -> np.ndarray:
    return rng.pareto(a=1.5, size=size)


def _bimodal(rng: np.random.Generator, size: int) -> np.ndarray:
    left = size // 2
    right = size - left
    first = rng.normal(-2.0, 1.0, left)
    second = rng.normal(2.0, 0.5, right)
    data = np.concatenate([first, second]) if size else np.empty(0, dtype=float)
    rng.shuffle(data)
    return data


DATA_GENERATORS: Dict[str, Callable[[np.random.Generator, int], np.ndarray]] = {
    "uniform": _uniform,
    "normal": _normal,
    "exponential": _exponential,
    "pareto": _pareto,
    "bimodal": _bimodal,
}


def _validate_distributions(names: Sequence[str]) -> None:
    unknown = sorted(set(names) - DATA_GENERATORS.keys())
    if unknown:
        raise ValueError(f"Unknown distributions requested: {', '.join(unknown)}")


def _instantiate_sketch(sketch_cls, capacity: int, seed: int):
    try:
        return sketch_cls(capacity=capacity, rng_seed=seed)
    except TypeError:
        # Older signatures might use positional arguments only.
        return sketch_cls(capacity)


def main() -> None:
    args = _parse_args()

    Ns = _to_int_list(args.Ns)
    capacities = _to_int_list(args.capacities)
    qs = _to_float_list(args.qs)
    _validate_distributions(args.distributions)

    module = importlib.import_module(args.module)
    cls_name = args.cls
    if not hasattr(module, cls_name):
        fallback_names = []
        if not cls_name.endswith("Sketch"):
            fallback_names.append(f"{cls_name}Sketch")
        else:
            base = cls_name[: -len("Sketch")]
            if base:
                fallback_names.append(base)
        fallback_names.extend(["KLLSketch", "KLL"])
        for candidate in fallback_names:
            if hasattr(module, candidate):
                cls_name = candidate
                break
        else:
            available = ", ".join(sorted(attr for attr in dir(module) if not attr.startswith("_")))
            raise AttributeError(
                f"{module.__name__!r} does not define {args.cls!r}. Available attributes: {available}"
            )

    sketch_cls = getattr(module, cls_name)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    accuracy_records: List[Dict[str, object]] = []
    throughput_records: List[Dict[str, object]] = []
    latency_records: List[Dict[str, object]] = []
    merge_records: List[Dict[str, object]] = []

    for dist in args.distributions:
        for N in Ns:
            combo_seed = _hash_seed(args.seed, dist, N)
            data_rng = np.random.default_rng(combo_seed)
            generator = DATA_GENERATORS[dist]
            data = generator(data_rng, N).astype(float, copy=False)
            if data.size != N:
                data = np.resize(data, N)

            exact_quantiles = np.quantile(data, qs, method="linear")
            exact_map = dict(zip(qs, exact_quantiles))

            for capacity in capacities:
                sketch = _instantiate_sketch(sketch_cls, capacity, args.seed)
                start = time.perf_counter()
                for value in data:
                    sketch.add(float(value))
                update_elapsed = time.perf_counter() - start
                updates_per_sec = (N / update_elapsed) if update_elapsed > 0 else math.inf

                throughput_records.append(
                    {
                        "distribution": dist,
                        "N": int(N),
                        "capacity": int(capacity),
                        "update_time_s": update_elapsed,
                        "updates_per_sec": updates_per_sec,
                    }
                )

                for q in qs:
                    q_float = float(q)
                    q_start = time.perf_counter()
                    approx = sketch.quantile(q_float)
                    q_elapsed = time.perf_counter() - q_start
                    latency_records.append(
                        {
                            "distribution": dist,
                            "N": int(N),
                            "capacity": int(capacity),
                            "q": q_float,
                            "latency_us": q_elapsed * 1e6,
                        }
                    )
                    accuracy_records.append(
                        {
                            "distribution": dist,
                            "N": int(N),
                            "capacity": int(capacity),
                            "mode": "single",
                            "q": q_float,
                            "estimate": approx,
                            "exact": exact_map[q_float],
                            "abs_error": abs(approx - exact_map[q_float]),
                        }
                    )

                shard_arrays = np.array_split(data, args.shards)
                shard_sketches = []
                for shard_idx, shard in enumerate(shard_arrays):
                    shard_sketch = _instantiate_sketch(
                        sketch_cls, capacity, args.seed + shard_idx + 1
                    )
                    for value in shard:
                        shard_sketch.add(float(value))
                    shard_sketches.append(shard_sketch)

                merge_target = _instantiate_sketch(sketch_cls, capacity, args.seed)
                merge_start = time.perf_counter()
                for shard_sketch in shard_sketches:
                    merge_target.merge(shard_sketch)
                merge_elapsed = time.perf_counter() - merge_start

                merge_records.append(
                    {
                        "distribution": dist,
                        "N": int(N),
                        "capacity": int(capacity),
                        "shards": int(args.shards),
                        "merge_time_s": merge_elapsed,
                    }
                )

                for q in qs:
                    q_float = float(q)
                    approx = merge_target.quantile(q_float)
                    accuracy_records.append(
                        {
                            "distribution": dist,
                            "N": int(N),
                            "capacity": int(capacity),
                            "mode": "merged",
                            "q": q_float,
                            "estimate": approx,
                            "exact": exact_map[q_float],
                            "abs_error": abs(approx - exact_map[q_float]),
                        }
                    )

    accuracy_path = outdir / "accuracy.csv"
    throughput_path = outdir / "update_throughput.csv"
    latency_path = outdir / "query_latency.csv"
    merge_path = outdir / "merge.csv"

    pd.DataFrame.from_records(accuracy_records).to_csv(accuracy_path, index=False)
    pd.DataFrame.from_records(throughput_records).to_csv(throughput_path, index=False)
    pd.DataFrame.from_records(latency_records).to_csv(latency_path, index=False)
    pd.DataFrame.from_records(merge_records).to_csv(merge_path, index=False)

    print("Benchmark artifacts written to:")
    print(f"  {accuracy_path}")
    print(f"  {throughput_path}")
    print(f"  {latency_path}")
    print(f"  {merge_path}")


if __name__ == "__main__":
    main()
