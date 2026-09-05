# KLL Streaming Quantile Sketch

A high-integrity, mergeable **KLL quantile sketch** for Python with reproducible seeded randomness, exact extrema, strict versioned serialization, rank-space validation, zero runtime dependencies, and an **optional C++17 native acceleration backend**.

Version **3.2** keeps the v2/v3 public `KLL` API and `KLL2` wire format intact while specializing the resident native merge engine. A fresh empty destination can now adopt an already-valid same-`k` resident source hierarchy directly, avoiding the v3.1 empty-state bootstrap and general merge planner. The destination keeps its own SplitMix64 RNG and compaction count, so later compactions remain byte-identical to the pure-Python reference. Non-empty merges stay on the proven v3.1 preflighted engine.
