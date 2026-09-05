# Contributing

Contributions are welcome when they preserve the project's core contract: deterministic
seeded semantics, exact Python/native state parity, stable public APIs, and strict
serialization behavior.

## Development setup

```bash
python -m pip install -r kll_sketch/requirements-test.txt
python -m pytest -q kll_sketch/tests --cov=kll_sketch --cov-report=term-missing
```

To exercise the optional native backend:

```bash
python -m kll_sketch._native_build
python -m pytest -q kll_sketch/tests
```

## Change rules

1. Pure Python is the executable semantic reference. A native optimization must produce
   the same observable results and, where covered by the compatibility contract,
   byte-identical KLL2 state.
2. Do not change RNG consumption, compaction ordering, signed-zero behavior, `min_k`
   inheritance, or wire-format semantics as a performance shortcut.
3. New fast paths require differential tests against the Python path, including fallback
   and invalid-input cases.
4. Performance claims require raw artifacts, pinned peer versions, machine/runtime
   metadata, repeated trials, and wording scoped to the measured workloads.
5. Shared-runner timing gates should use same-process ratios and conservative thresholds;
   absolute timings belong in characterization artifacts, not correctness CI.
6. Backward-incompatible API or serialization changes require an explicit major/minor
   compatibility decision and migration documentation.

## Benchmarking

For rank-space correctness characterization:

```bash
python benchmarks/bench_kll.py --outdir bench_out --Ns 1e5 --capacities 100 200 400 800 --trials 5
python benchmarks/validate_benchmarks.py bench_out
```

For Apache DataSketches comparison:

```bash
python -m pip install numpy==2.5.2 datasketches==5.2.0
python -m kll_sketch._native_build
python benchmarks/competitive_kll_focus.py
python benchmarks/competitive_kll_matrix.py
```

Never tune or merge an optimization solely against one benchmark point. Re-run semantic
parity, the focused peer comparison, and the multi-`k`/multi-`N` matrix.

## Pull requests

Keep changes narrow. Describe the compatibility surface touched, tests executed, and any
benchmark methodology changes. If performance changes, attach or link the generated
JSON/CSV artifact rather than only reporting a headline number.
