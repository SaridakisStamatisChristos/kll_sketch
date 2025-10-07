Here’s a polished, stars-friendly **README.md** you can paste on your repo’s front page. I tailored it for your project and authorship.


# KLL Streaming Quantile Sketch (Python)
Fast, mergeable **KLL** sketch for streaming quantiles — deterministic, zero deps, production-ready.

[![CI](https://github.com/SaridakisStamatisChristos/kll_sketch/actions/workflows/ci.yml/badge.svg)](https://github.com/SaridakisStamatisChristos/kll_sketch/actions/workflows/ci.yml)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

> Author: **Stamatis-Christos Saridakis**

---

## ✨ Features
- **Accurate ε-quantiles** under tight memory bounds (KLL algorithm)
- **Deterministic compaction** (parity sampling) + **weight conservation**
- **Weighted ingestion** via `add(x, weight)` for aggregated data
- **Mergeable** sketches for distributed/parallel ingestion
- **Serializable** (`to_bytes` / `from_bytes`)
- **Convenience helpers** such as `quantiles(m)` and `quantiles_at(qs)` for
  evenly spaced or ad-hoc cuts
- **Zero dependencies**, Python 3.9+

---

## 🚀 Quick Start
```python
from kll_sketch import KLL

sk = KLL(capacity=200)
sk.extend([1, 5, 2, 9, 3, 6, 4, 8, 7])

print("n =", sk.size())
print("median ≈", sk.median())
print("q(0.9) ≈", sk.quantile(0.9))
print("quartiles ≈", sk.quantiles(4))
```

### Merge & Serialize

```python
a, b = KLL(200), KLL(200)
a.extend(range(10_000))
b.extend(range(10_000, 20_000))

a.merge(b)            # in-place merge
blob = a.to_bytes()   # serialize
a2 = KLL.from_bytes(blob)
assert abs(a2.quantile(0.5) - a.quantile(0.5)) < 1e-12
```

---

## 🧰 API (minimal)

| Method                        | Description                             |
| ----------------------------- | --------------------------------------- |
| `add(x, weight=1)`            | Ingest one value with optional weight.  |
| `extend(xs)`                  | Ingest an iterable of values.           |
| `size()`                      | Total number of ingested items `n`.     |
| `quantile(q)`                 | Approximate `q`-quantile for `q∈[0,1]`. |
| `quantiles(m)`                | Evenly spaced cut points.               |
| `quantiles_at(qs)`            | Batched quantiles for arbitrary `qs`.   |
| `median()`                    | Convenience for `quantile(0.5)`.        |
| `rank(x)`                     | Approximate rank of `x` in `[0, n]`.    |
| `cdf(xs)`                     | CDF values for a sequence `xs`.         |
| `merge(other)`                | In-place merge with another sketch.     |
| `to_bytes()` / `from_bytes()` | Serialize / deserialize.                |

---

## 📐 Theory (KLL in one minute)

This implementation follows **Karnin–Lang–Liberty (2016)**: a space-optimal streaming algorithm for quantile approximation. Items are stored in **levels**; compaction randomly keeps one item from each pair and **promotes** it upward, doubling its weight. This achieves tight error bounds with **O(k)** space and **amortized O(1)** update cost.

> Reference: *Optimal Quantile Approximation in Streams*, FOCS 2016.

---

## 📊 Accuracy & Performance

* Typical error ≈ **O(1/k)** in rank space (increase `capacity` to tighten ε).
* Updates amortized **O(1)** with occasional compactions.
* Queries merge level buffers (**k-way**) and scan weights to the target rank.
  Use `quantiles_at` to answer multiple quantiles with a single scan.

> Tip: For heavy query loads, cache materialized arrays between queries.

---

## ✅ Tests

```bash
python -m pytest -q
```

---

## 🗺️ Roadmap

* Optional NumPy/C hot paths for sort/merge.

---

## 📝 License

Licensed under **Apache-2.0**.

---

## 🙌 Acknowledgments

Based on the KLL algorithm by Z. Karnin, E. Liberty, and L. Lang.

```


