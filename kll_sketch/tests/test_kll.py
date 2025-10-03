import random
from kll_sketch import KLL

def test_basic_quantiles():
    rng = random.Random(1)
    xs = [rng.random() for _ in range(50_000)]
    srt = sorted(xs)
    sk = KLL(capacity=200); sk.extend(xs)
    for q in [0.01,0.1,0.25,0.5,0.75,0.9,0.99]:
        est = sk.quantile(q); tru = srt[int(q*(len(xs)-1))]
        assert abs(est - tru) <= 0.02

def test_weight_conservation():
    rng = random.Random(0)
    sk = KLL(capacity=64, rng_seed=777)
    xs = [rng.random() for _ in range(200_000)]
    sk.extend(xs)
    vals, wts = sk._materialize_aligned()
    assert abs(sum(wts) - sk.size()) < 1e-9
