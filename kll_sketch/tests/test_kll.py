import bisect
import math
import random
import struct

import pytest

from kll_sketch import KLL, SerializationError


def realized_rank_error(xs, estimate, q):
    ordered = sorted(xs)
    target = q * (len(xs) - 1)
    lo = bisect.bisect_left(ordered, estimate)
    hi = bisect.bisect_right(ordered, estimate) - 1
    if target < lo:
        return (lo - target) / len(xs)
    if target > hi:
        return (target - hi) / len(xs)
    return 0.0


def test_exact_mode_and_extrema():
    xs = [3.0, 1.0, 8.0, 2.0, 2.0]
    sk = KLL(40, 7)
    sk.extend(xs)
    assert sk.min_value == 1.0
    assert sk.max_value == 8.0
    assert not sk.is_estimation_mode
    for q in [0, .25, .5, .75, 1]:
        truth = sorted(xs)[int(q*(len(xs)-1))]
        assert sk.quantile(q) == truth
    sk.validate()


def test_large_stream_invariants_and_reasonable_rank_accuracy():
    rng = random.Random(123)
    xs = [rng.random() for _ in range(100_000)]
    sk = KLL(200, 9)
    sk.extend(xs)
    sk.validate()
    assert sk.num_retained < 1000
    for q in [.001,.01,.1,.5,.9,.99,.999]:
        err = realized_rank_error(xs, sk.quantile(q), q)
        assert err < 0.04


def test_sorted_reverse_duplicates_adversarial():
    for xs in [list(map(float, range(50000))), list(map(float, range(50000,0,-1))), [1.0]*50000, ([0.0, 1e30]*25000)]:
        sk = KLL(200, 42)
        sk.extend(xs)
        sk.validate()
        assert sk.min_value == min(xs)
        assert sk.max_value == max(xs)
        assert sk.quantile(0) >= min(xs)
        assert sk.quantile(1) <= max(xs)


def test_cache_reused_and_invalidated():
    sk = KLL(64)
    sk.extend(range(1000))
    a = sk._query_view()
    b = sk._query_view()
    assert a[0] is b[0]
    sk.add(1001)
    c = sk._query_view()
    assert c[0] is not a[0]


def test_rank_cdf_pmf_semantics():
    sk = KLL(64)
    sk.extend([1,1,2,3,4])
    assert sk.rank(1) == 2
    assert sk.rank(1, inclusive=False) == 0
    assert sk.normalized_rank(2) == 3/5
    assert sk.cdf([1,2,4]) == pytest.approx([2/5,3/5,1])
    pmf = sk.pmf([1,3])
    assert math.isclose(sum(pmf), 1.0)
    assert pmf == pytest.approx([2/5,2/5,1/5])


def test_merge_and_weight_conservation():
    rng = random.Random(11)
    xs = [rng.gauss(0,1) for _ in range(40000)]
    a,b = KLL(200,1), KLL(200,2)
    a.extend(xs[:20000]); b.extend(xs[20000:])
    a.merge(b)
    a.validate()
    assert a.n == 40000
    vals,wts = a._materialize_aligned()
    assert sum(wts) == 40000
    assert len(vals) == a.num_retained


def test_merge_different_k_adopts_tighter():
    a,b = KLL(400),KLL(100)
    a.extend(range(10000)); b.extend(range(10000,20000))
    a.merge(b)
    assert a.k == 400
    assert a.min_k == 100
    a.validate()


def test_weighted_small_exact():
    sk = KLL(128)
    sk.add(-1, weight=3)
    sk.add(2.5, weight=5)
    sk.add(10, weight=2)
    assert sk.n == 10
    assert sk.quantile(.5) == 2.5
    sk.validate()


def test_seed_normalization_serializes_negative_seed():
    sk = KLL(64, -1)
    sk.extend(range(1000))
    restored = KLL.from_bytes(sk.to_bytes())
    assert restored.quantiles_at([0,.5,1]) == sk.quantiles_at([0,.5,1])


def test_kll2_roundtrip_and_checksum():
    sk = KLL(100, 999)
    sk.extend(range(5000))
    blob = sk.to_bytes()
    assert blob[:4] == b"KLL2"
    restored = KLL.from_bytes(blob)
    assert restored.debug_state() == sk.debug_state()
    assert restored._levels == sk._levels
    assert restored.to_bytes() == blob
    bad = bytearray(blob); bad[-1] ^= 1
    with pytest.raises(SerializationError, match="checksum"):
        KLL.from_bytes(bad)


def test_kll2_rejects_trailing_and_truncated():
    sk = KLL(64); sk.extend(range(1000)); blob=sk.to_bytes()
    with pytest.raises(SerializationError): KLL.from_bytes(blob+b"x")
    with pytest.raises(SerializationError): KLL.from_bytes(blob[:-1])
    with pytest.raises(SerializationError): KLL.from_bytes(b"bad")


def test_kll1_backward_read():
    # Legacy exact sketch: k=64, n=3, L=1, seed=7, level [1,2,3]
    blob = b"KLL1" + struct.pack(">I Q I Q I 3d", 64,3,1,7,3,1.0,2.0,3.0)
    sk = KLL.from_bytes(blob)
    assert sk.n == 3
    assert sk.quantile(.5) == 2.0
    assert sk.min_value == 1.0 and sk.max_value == 3.0
    sk.validate()


def test_invalid_inputs():
    with pytest.raises(ValueError): KLL(39)
    sk=KLL(64)
    for bad in [float("nan"),float("inf"),-float("inf")]:
        with pytest.raises(ValueError): sk.add(bad)
    for bad in [0,-1,1.5,float("inf")]:
        with pytest.raises(ValueError): sk.add(1, bad)
    for bad in [-.1,1.1,float("nan")]:
        with pytest.raises(ValueError): sk.quantile(bad)
    with pytest.raises(ValueError): sk.quantile(.5)
    with pytest.raises(ValueError): sk.pmf([2,1])
    with pytest.raises(ValueError): sk.merge(sk)


def test_error_model_matches_reference_values():
    expected={100:.02608,200:.01329,400:.00678,800:.00345}
    for k,e in expected.items():
        got=KLL(k).normalized_rank_error()
        assert got == pytest.approx(e, abs=5e-5)


def _rewrite_kll2(blob: bytes, offset: int, replacement: bytes) -> bytes:
    import zlib
    payload = bytearray(blob[12:])
    payload[offset:offset+len(replacement)] = replacement
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    return b"KLL2" + struct.pack(">II", len(payload), crc) + payload


def test_public_api_edge_paths():
    sk = KLL(64, 17)
    assert len(sk) == sk.size() == sk.n == 0
    assert sk.serialization_version == 2
    assert sk.rank(1) == 0
    assert sk.ranks([1, 2]) == [0, 0]
    assert sk.normalized_rank(1) == 0
    assert sk.cdf([1, 2]) == [0, 0]
    assert sk.pmf([1, 2]) == [0, 0, 0]
    assert sk.quantiles_at([]) == []
    with pytest.raises(ValueError):
        _ = sk.min_value
    with pytest.raises(ValueError):
        _ = sk.max_value
    sk.update_many([1,2,3,4])
    assert sk.median() == 2
    assert sk.quantiles(1) == [2]
    assert sk.quantiles(4) == [1,2,3]
    assert sk.rank(-100) == 0
    assert sk.ranks([-100, 2, 100]) == [0, 2, 4]
    assert sk.normalized_rank(100) == 1
    assert sk.normalized_rank_error(pmf=True) > sk.normalized_rank_error()
    lo = sk.quantile_lower_bound(.5)
    hi = sk.quantile_upper_bound(.5)
    assert lo <= sk.quantile(.5) <= hi
    clone = sk.copy()
    assert clone.debug_state() == sk.debug_state()
    clone.add(99)
    assert clone.n == sk.n + 1
    assert sk.n == 4


def test_argument_coercion_and_merge_edges():
    for bad in [True, 39, 40.5, "x"]:
        with pytest.raises((TypeError, ValueError)):
            KLL(bad)
    with pytest.raises(ValueError):
        KLL(0x1_0000_0000)
    for seed in [True, 1.5, "x"]:
        with pytest.raises((TypeError, ValueError)):
            KLL(64, seed)
    sk = KLL(64)
    for bad in [None, object()]:
        with pytest.raises(TypeError):
            sk.add(bad)
    for bad in [None, object()]:
        with pytest.raises(TypeError):
            sk.add(1, bad)
    with pytest.raises(TypeError):
        sk.quantile(None)
    for bad_m in [True, 0, -1, 1.5, "x"]:
        with pytest.raises((TypeError, ValueError)):
            sk.quantiles(bad_m)
    with pytest.raises(TypeError):
        sk.merge(object())
    other = KLL(64)
    sk.merge(other)  # empty source is a no-op
    assert sk.n == 0


def test_serialization_type_and_structural_rejections():
    with pytest.raises(TypeError):
        KLL.from_bytes("not-bytes")
    with pytest.raises(SerializationError):
        KLL.from_bytes(b"")
    with pytest.raises(SerializationError):
        KLL.from_bytes(b"NOPE")

    sk = KLL(64, 4)
    sk.extend([1.0, 2.0, 3.0])
    blob = sk.to_bytes()
    with pytest.raises(SerializationError, match="feature flags"):
        KLL.from_bytes(_rewrite_kll2(blob, 0, b"\x02"))
    with pytest.raises(SerializationError, match="min_k"):
        KLL.from_bytes(_rewrite_kll2(blob, 5, struct.pack(">I", 39)))
    with pytest.raises(SerializationError, match="level count"):
        KLL.from_bytes(_rewrite_kll2(blob, 17, struct.pack(">I", 0)))
    with pytest.raises(SerializationError, match="retained-item"):
        KLL.from_bytes(_rewrite_kll2(blob, 45, struct.pack(">Q", 999)))
    with pytest.raises(SerializationError, match="level weights"):
        KLL.from_bytes(_rewrite_kll2(blob, 9, struct.pack(">Q", 999)))
    with pytest.raises(SerializationError, match="empty/non-empty"):
        KLL.from_bytes(_rewrite_kll2(blob, 0, b"\x00"))
    with pytest.raises(SerializationError, match="reversed"):
        bad = _rewrite_kll2(blob, 53, struct.pack(">d", 10.0))
        bad = _rewrite_kll2(bad, 61, struct.pack(">d", -10.0))
        KLL.from_bytes(bad)
    with pytest.raises(SerializationError, match="outside"):
        KLL.from_bytes(_rewrite_kll2(blob, 73, struct.pack(">d", 1000.0)))
    with pytest.raises(SerializationError, match="remaining"):
        KLL.from_bytes(_rewrite_kll2(blob, 69, struct.pack(">I", 999999)))


def test_empty_serialization_roundtrip_and_legacy_rejections():
    empty = KLL(64)
    restored = KLL.from_bytes(empty.to_bytes())
    assert restored.n == 0
    restored.validate()

    legacy = b"KLL1" + struct.pack(">I Q I Q I d", 64, 1, 1, 7, 1, 2.0)
    with pytest.raises(SerializationError, match="trailing"):
        KLL.from_bytes(legacy + b"x")
    with pytest.raises(SerializationError):
        KLL.from_bytes(legacy[:-2])
