# KLL Streaming Quantile Sketch (Python)

A fast, mergeable **KLL** sketch for streaming quantiles.

- Deterministic compaction (parity sampling), aligned weights, weight conservation
- Mergeable, serializable, **zero deps**, Python 3.9+
- API: `add`, `extend`, `quantile`, `rank`, `cdf`, `merge`, `to_bytes`, `from_bytes`

## Install
Just drop `kll_sketch.py` into your project (or package via `pyproject.toml`).

## Quick start
```python
from kll_sketch import KLL
sk = KLL(capacity=200)
sk.extend([1,2,3,4,5])
print(sk.quantile(0.5))  # ~3
```

## Tests
```bash
python -m pytest -q
```

## License
Apache-2.0
