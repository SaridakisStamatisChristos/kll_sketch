# KLL Streaming Quantile Sketch (Python)
# Production-ready implementation with:
# - Named constants (caps, loop guards)
# - Deterministic RNG salting (seed + salt mixer)
# - Weight-conserving compaction (both boundaries preserved)
# - Aligned materialization (values + weights stay in sync)
# - Merge + serialize/deserialize
# Python 3.9+

from __future__ import annotations
import math
import struct
import random
import heapq
from bisect import bisect_left, bisect_right
from typing import Iterable, List, Tuple, Optional


SERIAL_FORMAT_MAGIC = b"KLL1"
SERIAL_FORMAT_VERSION = 1


class KLL:
    """
    KLL streaming quantile sketch (supports weighted ingestion, mergeable, serializable).

    Paper:
      - Karnin, Zohar, Edo Liberty, and Liran Lang. "Optimal quantile approximation
        in streams." FOCS 2016.

    Strategy (high level):
      - Maintain multiple levels of buffers; level ℓ items represent weight 2^ℓ.
      - When over capacity, compact a level by sampling one element from each
        disjoint pair (parity-controlled), promoting the sampled elements to the
        next level (thus doubling their weight).
      - Boundary elements NOT in any pair are preserved at the current level.
        This guarantees total weight conservation:  Σ(weights) == n.

    Compatibility and upgrade policy:
      - The serialized binary format is versioned via the ``KLL{SERIAL_FORMAT_VERSION}``
        magic header. New minor releases preserve backwards compatibility with
        previously published format versions; breaking changes bump the header
        and provide a migration path through :meth:`from_bytes`.
      - The Python API follows semantic versioning. Patch releases may add
        methods or keyword arguments but will not change behaviour of existing
        calls. Major releases are reserved for intentional, documented
        compatibility breaks.

    Public API:
      add(x, weight=1), extend(xs), quantile(q), quantiles(m), quantiles_at(qs),
      median(), rank(x), cdf(xs), merge(other), to_bytes(), from_bytes()
    """

    # ---------------------------- Tunable constants ----------------------------
    _MIN_CAPACITY: int = 40                 # minimal allowed capacity for accuracy
    _SOFT_CAP_FACTOR: float = 1.15          # global soft overfill before compaction
    _LEVEL_BASE_MIN: int = 8                # per-level base cap (~k//8, min 8)
    _STALL_BREAK_LOOPS: int = 16            # break if no progress after this many loops
    _MAX_COMPACT_LOOPS: int = 10_000        # absolute safety bound on compaction loops
    _DEFAULT_SEED: int = 0xA5B357           # deterministic default RNG seed

    # 64-bit odd constant (golden ratio scaled) for hashing the RNG salt.
    # We mix the configured seed with an evolving salt (level + n + buffer size)
    # to get stable but well-dispersed pseudo-randomness per compaction event.
    _SALT_MIX64: int = 0x9E3779B185EBCA87

    __slots__ = ("_levels", "_n", "_k", "_rng_seed")

    def __init__(self, capacity: int = 200, rng_seed: int = _DEFAULT_SEED):
        if capacity < self._MIN_CAPACITY:
            raise ValueError(f"capacity must be >= {self._MIN_CAPACITY} for good accuracy")
        self._k = int(capacity)
        self._levels: List[List[float]] = [[]]  # level 0 buffer
        self._n = 0
        self._rng_seed = int(rng_seed)

    # ------------------------------- Public API --------------------------------
    def add(self, x: float, weight: float = 1.0) -> None:
        """Ingest a value with an optional (integer) weight."""

        # ``weight`` may be provided as an ``int`` or any float that rounds to an
        # integer (for compatibility with NumPy scalars).  Each unit of weight is
        # equivalent to inserting ``x`` once; internally we fold the binary
        # decomposition of ``weight`` across the sketch levels to avoid O(weight)
        # work for large aggregates.

        xv = float(x)
        if math.isnan(xv) or math.isinf(xv):
            raise ValueError("x must be finite")

        wv = float(weight)
        if math.isnan(wv) or math.isinf(wv):
            raise ValueError("weight must be finite")
        if wv <= 0.0:
            raise ValueError("weight must be > 0")

        rounded = int(round(wv))
        if abs(wv - rounded) > 1e-9:
            raise ValueError("weight must be an integer")
        if rounded <= 0:
            raise ValueError("weight must be > 0")

        self._ingest_weighted_value(xv, rounded)

    def extend(self, xs: Iterable[float]) -> None:
        for x in xs:
            self.add(x)

    def size(self) -> int:
        return self._n

    def median(self) -> float:
        return self.quantile(0.5)

    def quantile(self, q: float) -> float:
        if not (0.0 <= q <= 1.0):
            raise ValueError("q must be in [0,1]")
        return self._batched_quantiles([q])[0]

    def quantiles_at(self, probabilities: Iterable[float]) -> List[float]:
        """Return the approximate quantiles for each entry in ``probabilities``.

        This method evaluates all requested quantiles using a single materialized
        pass through the sketch, which is significantly faster than issuing
        repeated :meth:`quantile` calls for large query batches.
        """

        qs = [float(q) for q in probabilities]
        if any(not (0.0 <= q <= 1.0) for q in qs):
            raise ValueError("all probabilities must be in [0,1]")
        if not qs:
            return []
        return self._batched_quantiles(qs)

    def rank(self, x: float) -> float:
        """Approximate rank in [0, n]."""
        if self._n == 0:
            return 0.0
        vals, wts = self._materialize_aligned()
        cum = 0.0
        for v, w in zip(vals, wts):
            if x < v:
                return max(0.0, min(float(self._n), cum))
            cum += w
        return float(self._n)

    def cdf(self, xs: Iterable[float]) -> List[float]:
        n = max(1, self._n)
        return [self.rank(x) / n for x in xs]

    def merge(self, other: "KLL") -> None:
        if not isinstance(other, KLL):
            raise TypeError("merge expects KLL")
        if other._n == 0:
            return
        # Choose tighter capacity deterministically
        self._k = min(self._k, other._k)
        # Append levelwise; compaction will enforce caps.
        self._ensure_levels(max(len(self._levels), len(other._levels)))
        for lvl, arr in enumerate(other._levels):
            if arr:
                self._ensure_levels(lvl + 1)
                self._levels[lvl].extend(arr)
        self._n += other._n
        self._compress_until_ok()

    def to_bytes(self) -> bytes:
        """
        Serialize the sketch into the versioned ``KLL1`` binary envelope.

        The layout is:
          magic 'KLL1' (4B), k(uint32), n(uint64), L(uint32), seed(uint64),
          then for each level: len(uint32) followed by len doubles.

        The header version is bumped only when the on-wire format changes in a
        backwards-incompatible way. Minor library upgrades keep emitting
        ``KLL1`` payloads so downstream systems can safely deserialize historical
        snapshots.
        """
        out = bytearray()
        out += SERIAL_FORMAT_MAGIC
        out += struct.pack(">I", self._k)
        out += struct.pack(">Q", self._n)
        out += struct.pack(">I", len(self._levels))
        out += struct.pack(">Q", self._rng_seed)
        for arr in self._levels:
            out += struct.pack(">I", len(arr))
            if arr:
                out += struct.pack(">" + "d" * len(arr), *arr)
        return bytes(out)

    @classmethod
    def from_bytes(cls, b: bytes) -> "KLL":
        """Rehydrate a :class:`KLL` instance from :meth:`to_bytes` output."""
        mv = memoryview(b)
        if mv[:4].tobytes() != SERIAL_FORMAT_MAGIC:
            raise ValueError(
                "Unsupported serialization header. The 1.x reader only understands "
                f"{SERIAL_FORMAT_MAGIC!r}."
            )
        off = 4
        k = struct.unpack_from(">I", mv, off)[0]; off += 4
        n = struct.unpack_from(">Q", mv, off)[0]; off += 8
        L = struct.unpack_from(">I", mv, off)[0]; off += 4
        seed = struct.unpack_from(">Q", mv, off)[0]; off += 8
        self = cls(k, seed)
        self._levels = []
        for _ in range(L):
            ln = struct.unpack_from(">I", mv, off)[0]; off += 4
            if ln:
                arr = list(struct.unpack_from(">" + "d" * ln, mv, off))
                off += 8 * ln
            else:
                arr = []
            self._levels.append(arr)
        self._n = int(n)
        return self

    # ------------------------------- Internals ---------------------------------
    def _ensure_levels(self, L: int) -> None:
        while len(self._levels) < L:
            self._levels.append([])

    def _total_items(self) -> int:
        return sum(len(a) for a in self._levels)

    def _capacity_exceeded(self) -> bool:
        # Global soft cap to reduce churn
        return self._total_items() > int(self._k * self._SOFT_CAP_FACTOR)

    def _level_capacity(self, level: int) -> int:
        # Preserve exactness while the total population fits inside ``k`` by
        # deferring compaction.  This mirrors the streaming ingestion behaviour
        # where no compression is triggered before exceeding ``k`` items, which
        # keeps small merges numerically identical to simple extension.
        if self._n <= self._k:
            return self._k

        # Geometric schedule; sum over levels ≈ O(k)
        base = max(self._LEVEL_BASE_MIN, self._k // 8)
        return base * (1 << max(0, level))

    def _rng(self, salt: int) -> random.Random:
        # Deterministic per-event RNG using a 64-bit mix of seed and salt.
        # SALT combines level + current n + buffer length to vary across events.
        mix = (self._rng_seed * self._SALT_MIX64 + (salt & 0xFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
        return random.Random(mix)

    def _find_compactable_level(self) -> Optional[int]:
        """
        Return a level index that can form at least one pair.
        Preference: first level exceeding its cap (and len>=2), else the lowest
        level with len>=2; else None.
        """
        for lvl, arr in enumerate(self._levels):
            if len(arr) > self._level_capacity(lvl) and len(arr) >= 2:
                return lvl
        for lvl, arr in enumerate(self._levels):
            if len(arr) >= 2:
                return lvl
        return None

    def _compress_once(self) -> bool:
        """Do one compaction step. Returns True if any items were promoted."""
        if len(self._levels) == 1:
            self._levels.append([])

        lvl = self._find_compactable_level()
        if lvl is None:
            return False

        buf = self._levels[lvl]
        buf.sort()
        if len(buf) < 3:
            # Not enough items to compact while preserving min/max boundaries.
            return False

        rng = self._rng(salt=lvl + self._n + len(buf))
        keep_odd = rng.getrandbits(1) == 1
        start = 1 if keep_odd else 0

        # Always preserve explicit boundary elements.
        core = buf[1:-1]
        if len(core) < 2:
            return False

        if len(core) - start < 2:
            keep_odd = not keep_odd
            start = 1 if keep_odd else 0
            if len(core) - start < 2:
                return False

        promoted: List[float] = []
        for i in range(start, len(core) - 1, 2):
            promoted.append(core[i] if rng.getrandbits(1) else core[i + 1])

        if not promoted:
            return False

        leftover: List[float] = [buf[0]]
        if start == 1:
            leftover.append(core[0])
        if (len(core) - start) % 2 == 1:
            leftover.append(core[-1])
        leftover.append(buf[-1])

        self._levels[lvl] = leftover
        self._ensure_levels(lvl + 2)
        self._levels[lvl + 1].extend(promoted)
        return True

    def _compress_until_ok(self) -> None:
        loops = 0
        while (
            (
                self._capacity_exceeded()
                or any(len(a) > self._level_capacity(i) for i, a in enumerate(self._levels))
            )
            and self._find_compactable_level() is not None
        ):
            progressed = self._compress_once()
            loops += 1
            if not progressed and loops > self._STALL_BREAK_LOOPS:
                # Extremely rare; break to avoid spinning. Next insert/merge will retry.
                break
            if loops > self._MAX_COMPACT_LOOPS:
                raise RuntimeError("compaction did not converge")

    # --------- materialization with aligned weights (values, weights) ----------
    def _materialize_aligned(self) -> Tuple[List[float], List[float]]:
        per_level: List[Tuple[List[float], float]] = []
        for lvl, arr in enumerate(self._levels):
            if not arr:
                continue
            a = sorted(arr)
            w = float(1 << lvl)
            per_level.append((a, w))

        if not per_level:
            return [], []

        # Small-k fast path: manual k-way merge while pushing aligned weights.
        if len(per_level) <= 3:
            idx = [0] * len(per_level)
            out_v: List[float] = []
            out_w: List[float] = []
            while True:
                best: Optional[float] = None
                best_j = -1
                for j, (arr, w) in enumerate(per_level):
                    i = idx[j]
                    if i < len(arr):
                        v = arr[i]
                        if best is None or v < best:
                            best, best_j = v, j
                if best_j < 0:
                    break
                out_v.append(best)                  # value
                out_w.append(per_level[best_j][1])  # weight aligned to source level
                idx[best_j] += 1
            return out_v, out_w

        # General path: heap-merge for O(n log L); keep (value, j, i, weight)
        heap: List[Tuple[float, int, int, float]] = []
        for j, (arr, w) in enumerate(per_level):
            heap.append((arr[0], j, 0, w))
        heapq.heapify(heap)

        out_v2: List[float] = []
        out_w2: List[float] = []
        while heap:
            v, j, i, w = heapq.heappop(heap)
            out_v2.append(v)
            out_w2.append(w)
            i += 1
            arr = per_level[j][0]
            if i < len(arr):
                heapq.heappush(heap, (arr[i], j, i, w))
        return out_v2, out_w2

    def quantiles(self, m: int) -> List[float]:
        """Return evenly spaced quantile cut points.

        ``m`` corresponds to the number of equal-mass buckets.  For ``m > 1`` the
        return value contains ``m-1`` interior cut points.  ``m == 1`` yields the
        median for convenience.
        """

        if m <= 0:
            raise ValueError("m must be positive")
        if self._n == 0:
            raise ValueError("empty sketch")
        if m == 1:
            return [self.quantile(0.5)]
        step = 1.0 / m
        qs = [step * i for i in range(1, m)]
        return self.quantiles_at(qs)

    # ---------------------- weighted ingestion internals ----------------------
    def _ingest_weighted_value(self, value: float, weight: int) -> None:
        """Fold ``weight`` copies of ``value`` into the level buffers."""

        remaining = weight
        level = 0
        while remaining:
            if remaining & 1:
                self._ensure_levels(level + 1)
                self._levels[level].append(value)
            remaining >>= 1
            level += 1
        self._n += weight
        if self._capacity_exceeded():
            self._compress_until_ok()

    def _batched_quantiles(self, qs: List[float]) -> List[float]:
        if self._n == 0:
            raise ValueError("empty sketch")
        vals, wts = self._materialize_aligned()
        if not vals:
            raise ValueError("empty sketch")

        prefix: List[float] = []
        total = 0.0
        for w in wts:
            total += w
            prefix.append(total)

        ordered = sorted(enumerate(qs), key=lambda item: item[1])
        out = [0.0] * len(qs)
        search_lo = 0
        for idx, q in ordered:
            target = q * (self._n - 1)
            pos = bisect_right(prefix, target, lo=search_lo)
            if pos >= len(vals):
                pos = len(vals) - 1
            out[idx] = vals[pos]
            search_lo = pos
        return out


# ----------------------------- quick self-test --------------------------------
if __name__ == "__main__":
    import random

    rng = random.Random(42)
    xs = [rng.gauss(0, 1) for _ in range(20000)]
    sk = KLL(capacity=200)
    sk.extend(xs)
    for q in [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0]:
        est = sk.quantile(q)
        truth = sorted(xs)[int(q * (len(xs) - 1))]
        err = abs(est - truth)
        print(f"q={q:>4}: est={est:+.4f} truth={truth:+.4f} |err|={err:.4f}")

    # Weight conservation sanity check
    _, wts = sk._materialize_aligned()
    assert abs(sum(wts) - sk.size()) < 1e-9, "weight conservation violated"