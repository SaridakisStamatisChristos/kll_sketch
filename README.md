# KLL Streaming Quantile Sketch

A high-integrity, mergeable **KLL quantile sketch** for Python with reproducible seeded randomness, exact extrema, strict versioned serialization, rank-space validation, zero runtime dependencies, and an **optional C++17 native acceleration backend**.

Version **3.2** keeps the v2/v3 public `KLL` API and `KLL2` wire format intact while specializing the resident native merge engine.
