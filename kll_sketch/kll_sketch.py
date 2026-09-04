"""High-integrity KLL quantile sketch.

The implementation keeps the original public surface of this project while
replacing the compaction engine with a KLL-style hierarchy:

* level capacities decrease geometrically toward level zero;
* overfull levels are compacted with one unbiased parity choice;
* odd compactors keep one boundary item and compact the remaining even block;
* exact min/max are tracked outside the compactors;
* total represented weight is an invariant: ``n == sum(2**h * len(level[h]))``;
* repeated queries share one cached, sorted weighted view;
* serialization is strict, checksummed and backwards-readable from ``KLL1``.

The sketch is randomized in the algorithmic sense, but reproducible for a
fixed ``rng_seed`` because it uses an internal SplitMix64 stream.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
import math
import struct
from typing import Iterable, List, Optional, Sequence, Tuple
import zlib


SERIAL_FORMAT_MAGIC = b"KLL2"
SERIAL_FORMAT_VERSION = 2
LEGACY_SERIAL_FORMAT_MAGIC = b"KLL1"

_U64_MASK = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB


class SerializationError(ValueError):
    """Raised when a serialized KLL payload is malformed or inconsistent."""


class KLL:
    """Mergeable KLL-style streaming quantile sketch.

    Parameters
    ----------
    capacity:
        Accuracy / memory control parameter conventionally called ``k``.
        Larger values retain more items and reduce normalized rank error.
    rng_seed:
        Seed for the sketch's reproducible pseudo-random compaction stream.

    Notes
    -----
    ``rank(x)`` preserves the historical API and returns an approximate *absolute*
    inclusive rank in ``[0, n]``. ``normalized_rank`` exposes the statistically
    natural ``[0, 1]`` form.

    ``add(x, weight)`` is retained for backwards compatibility. Integer weights
    are represented exactly in total mass, using binary level placement. This is
    a useful weighted extension, but applications that require the ordinary
    unweighted KLL statistical model should ingest observations individually.
    """

    _MIN_CAPACITY = 40
    _MIN_LEVEL_CAPACITY = 8
    _MAX_LEVELS = 64
    _DEFAULT_SEED = 0xA5B357
    _MAX_SERIALIZED_BYTES = 512 * 1024 * 1024

    # DataSketches-compatible empirical 99% normalized-rank error model.
    # Single-sided rank/quantile error: 2.296 / k**0.9723
    # Double-sided PMF error:         2.446 / k**0.9433
    _SINGLE_SIDED_ERROR_A = 2.296
    _SINGLE_SIDED_ERROR_B = 0.9723
    _PMF_ERROR_A = 2.446
    _PMF_ERROR_B = 0.9433

    __slots__ = (
        "_levels",
        "_n",
        "_k",
        "_min_k",
        "_seed",
        "_rng_state",
        "_num_retained",
        "_min_value",
        "_max_value",
        "_generation",
        "_cache_generation",
        "_cache_values",
        "_cache_prefix",
        "_compaction_count",
    )

    def __init__(self, capacity: int = 200, rng_seed: int = _DEFAULT_SEED):
        if isinstance(capacity, bool):
            raise TypeError("capacity must be an integer")
        try:
            k = int(capacity)
        except (TypeError, ValueError) as exc:
            raise TypeError("capacity must be an integer") from exc
        if k != capacity:
            raise ValueError("capacity must be an integer")
        if k < self._MIN_CAPACITY:
            raise ValueError(f"capacity must be >= {self._MIN_CAPACITY}")
        if k > 0xFFFFFFFF:
            raise ValueError("capacity is too large for the serialized format")

        seed = self._coerce_seed(rng_seed)
        self._k = k
        self._min_k = k
        self._seed = seed
        self._rng_state = seed
        self._levels: List[List[float]] = [[]]
        self._n = 0
        self._num_retained = 0
        self._min_value: Optional[float] = None
        self._max_value: Optional[float] = None
        self._generation = 0
        self._cache_generation = -1
        self._cache_values: List[float] = []
        self._cache_prefix: List[int] = []
        self._compaction_count = 0

    # ------------------------------------------------------------------
    # Public state and compatibility helpers
    # ------------------------------------------------------------------
    @property
    def k(self) -> int:
        return self._k

    @property
    def min_k(self) -> int:
        """Smallest effective k inherited from estimation-mode merge inputs."""
        return self._min_k

    @property
    def n(self) -> int:
        return self._n

    @property
    def num_retained(self) -> int:
        return self._num_retained

    @property
    def is_estimation_mode(self) -> bool:
        return self._n > self._k or len(self._levels) > 1

    @property
    def min_value(self) -> float:
        if self._min_value is None:
            raise ValueError("empty sketch")
        return self._min_value

    @property
    def max_value(self) -> float:
        if self._max_value is None:
            raise ValueError("empty sketch")
        return self._max_value

    @property
    def serialization_version(self) -> int:
        return SERIAL_FORMAT_VERSION

    def __len__(self) -> int:
        return self._n

    def size(self) -> int:
        """Return total represented input weight."""
        return self._n

    def copy(self) -> "KLL":
        """Return an independent in-memory copy preserving RNG state."""
        other = KLL(self._k, self._seed)
        other._levels = [level.copy() for level in self._levels]
        other._n = self._n
        other._min_k = self._min_k
        other._rng_state = self._rng_state
        other._num_retained = self._num_retained
        other._min_value = self._min_value
        other._max_value = self._max_value
        other._generation = self._generation
        other._compaction_count = self._compaction_count
        return other

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def add(self, x: float, weight: float = 1.0) -> None:
        """Ingest ``x`` with a positive integer weight.

        ``weight=1`` follows the ordinary unweighted KLL path. Larger integer
        weights use the backwards-compatible weighted extension.
        """
        value = self._coerce_value(x)
        iw = self._coerce_weight(weight)
        if iw == 1:
            self._add_one(value)
        else:
            self._add_weighted_binary(value, iw)

    def extend(self, xs: Iterable[float]) -> None:
        """Bulk-ingest an iterable with one validation/dispatch path per item."""
        for raw in xs:
            self._add_one(self._coerce_value(raw))

    def update_many(self, xs: Iterable[float]) -> None:
        """Alias for :meth:`extend` for users coming from sketch APIs."""
        self.extend(xs)

    # ------------------------------------------------------------------
    # Quantile, rank, CDF and PMF queries
    # ------------------------------------------------------------------
    def median(self) -> float:
        return self.quantile(0.5)

    def quantile(self, q: float) -> float:
        qf = self._coerce_probability(q, "q")
        return self._quantiles_from_probabilities([qf])[0]

    def quantiles_at(self, probabilities: Iterable[float]) -> List[float]:
        qs = [self._coerce_probability(q, "probability") for q in probabilities]
        if not qs:
            return []
        return self._quantiles_from_probabilities(qs)

    def quantiles(self, m: int) -> List[float]:
        """Return ``m-1`` interior equal-mass cut points.

        Historical behavior is retained for ``m == 1``: it returns a one-item
        list containing the median.
        """
        if isinstance(m, bool):
            raise TypeError("m must be an integer")
        try:
            mi = int(m)
        except (TypeError, ValueError) as exc:
            raise TypeError("m must be an integer") from exc
        if mi != m:
            raise ValueError("m must be an integer")
        if mi <= 0:
            raise ValueError("m must be positive")
        self._require_nonempty()
        if mi == 1:
            return [self.median()]
        return self.quantiles_at(i / mi for i in range(1, mi))

    def rank(self, x: float, *, inclusive: bool = True) -> float:
        """Return approximate absolute rank in ``[0, n]``.

        By default the rank is inclusive (count of represented mass ``<= x``),
        matching the historical behavior. Set ``inclusive=False`` for ``< x``.
        """
        if self._n == 0:
            return 0.0
        value = self._coerce_value(x)
        values, prefix = self._query_view()
        pos = bisect_right(values, value) if inclusive else bisect_left(values, value)
        if pos <= 0:
            return 0.0
        return float(prefix[pos - 1])

    def ranks(self, xs: Iterable[float], *, inclusive: bool = True) -> List[float]:
        if self._n == 0:
            return [0.0 for _ in xs]
        values, prefix = self._query_view()
        out: List[float] = []
        finder = bisect_right if inclusive else bisect_left
        for raw in xs:
            value = self._coerce_value(raw)
            pos = finder(values, value)
            out.append(0.0 if pos <= 0 else float(prefix[pos - 1]))
        return out

    def normalized_rank(self, x: float, *, inclusive: bool = True) -> float:
        if self._n == 0:
            return 0.0
        return self.rank(x, inclusive=inclusive) / self._n

    def cdf(self, xs: Iterable[float], *, inclusive: bool = True) -> List[float]:
        if self._n == 0:
            return [0.0 for _ in xs]
        inv_n = 1.0 / self._n
        return [r * inv_n for r in self.ranks(xs, inclusive=inclusive)]

    def pmf(self, split_points: Iterable[float], *, inclusive: bool = True) -> List[float]:
        """Return probability masses separated by monotonically increasing cuts."""
        points = [self._coerce_value(x) for x in split_points]
        if any(a >= b for a, b in zip(points, points[1:])):
            raise ValueError("split_points must be strictly increasing")
        if self._n == 0:
            return [0.0] * (len(points) + 1)
        cuts = self.cdf(points, inclusive=inclusive)
        out: List[float] = []
        previous = 0.0
        for cut in cuts:
            out.append(max(0.0, cut - previous))
            previous = cut
        out.append(max(0.0, 1.0 - previous))
        return out

    # ------------------------------------------------------------------
    # Error model and confidence helpers
    # ------------------------------------------------------------------
    def normalized_rank_error(self, *, pmf: bool = False) -> float:
        """Return the conventional empirical ~99% normalized-rank error model.

        This is an engineering characterization, not a per-instance proof.
        ``pmf=True`` uses the more conservative double-sided error model.
        """
        if pmf:
            return self._PMF_ERROR_A / (self._min_k ** self._PMF_ERROR_B)
        return self._SINGLE_SIDED_ERROR_A / (self._min_k ** self._SINGLE_SIDED_ERROR_B)

    def quantile_lower_bound(self, q: float) -> float:
        qf = self._coerce_probability(q, "q")
        eps = self.normalized_rank_error()
        return self.quantile(max(0.0, qf - eps))

    def quantile_upper_bound(self, q: float) -> float:
        qf = self._coerce_probability(q, "q")
        eps = self.normalized_rank_error()
        return self.quantile(min(1.0, qf + eps))

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------
    def merge(self, other: "KLL") -> None:
        """Merge ``other`` into this sketch.

        The destination keeps its configured ``k``. If ``other`` is already in
        estimation mode, ``min_k`` records the tighter inherited error parameter.
        """
        if not isinstance(other, KLL):
            raise TypeError("merge expects KLL")
        if other is self:
            raise ValueError("cannot merge a sketch with itself")
        if other._n == 0:
            return

        if other.is_estimation_mode:
            self._min_k = min(self._min_k, other._min_k)
        self._ensure_levels(len(other._levels))
        for idx, level in enumerate(other._levels):
            if level:
                self._levels[idx].extend(level)
        self._n += other._n
        self._num_retained += other._num_retained
        if self._min_value is None or (other._min_value is not None and other._min_value < self._min_value):
            self._min_value = other._min_value
        if self._max_value is None or (other._max_value is not None and other._max_value > self._max_value):
            self._max_value = other._max_value
        self._mark_mutated()
        self._compress_while_needed()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_bytes(self) -> bytes:
        """Serialize to strict checksummed ``KLL2`` format."""
        flags = 1 if self._n else 0
        payload = bytearray()
        payload += struct.pack(">B", flags)
        payload += struct.pack(">I", self._k)
        payload += struct.pack(">I", self._min_k)
        payload += struct.pack(">Q", self._n)
        payload += struct.pack(">I", len(self._levels))
        payload += struct.pack(">Q", self._seed)
        payload += struct.pack(">Q", self._rng_state)
        payload += struct.pack(">Q", self._compaction_count)
        payload += struct.pack(">Q", self._num_retained)
        payload += struct.pack(">d", 0.0 if self._min_value is None else self._min_value)
        payload += struct.pack(">d", 0.0 if self._max_value is None else self._max_value)
        for level in self._levels:
            payload += struct.pack(">I", len(level))
            if level:
                payload += struct.pack(">" + "d" * len(level), *level)

        if len(payload) > self._MAX_SERIALIZED_BYTES:
            raise SerializationError("serialized sketch exceeds safety limit")
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        return SERIAL_FORMAT_MAGIC + struct.pack(">II", len(payload), crc) + payload

    @classmethod
    def from_bytes(cls, data: bytes) -> "KLL":
        """Deserialize ``KLL2`` or a historical ``KLL1`` payload."""
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("serialized sketch must be bytes-like")
        blob = bytes(data)
        if len(blob) < 4:
            raise SerializationError("serialized sketch is truncated")
        magic = blob[:4]
        if magic == SERIAL_FORMAT_MAGIC:
            return cls._from_v2(blob)
        if magic == LEGACY_SERIAL_FORMAT_MAGIC:
            return cls._from_v1(blob)
        raise SerializationError(f"unsupported serialization header: {magic!r}")

    @classmethod
    def _from_v2(cls, blob: bytes) -> "KLL":
        if len(blob) < 12:
            raise SerializationError("KLL2 header is truncated")
        payload_len, expected_crc = struct.unpack_from(">II", blob, 4)
        if payload_len > cls._MAX_SERIALIZED_BYTES:
            raise SerializationError("declared payload exceeds safety limit")
        if len(blob) != 12 + payload_len:
            raise SerializationError("KLL2 payload length mismatch or trailing bytes")
        payload = memoryview(blob)[12:]
        actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise SerializationError("KLL2 checksum mismatch")

        reader = _Reader(payload)
        try:
            flags = reader.u8()
            if flags & ~1:
                raise SerializationError("KLL2 contains unsupported feature flags")
            k = reader.u32()
            min_k = reader.u32()
            n = reader.u64()
            level_count = reader.u32()
            seed = reader.u64()
            rng_state = reader.u64()
            compaction_count = reader.u64()
            retained = reader.u64()
            min_value = reader.f64()
            max_value = reader.f64()
            levels = cls._read_levels(reader, level_count)
            reader.require_eof()
        except (struct.error, OverflowError) as exc:
            raise SerializationError("KLL2 payload is truncated or malformed") from exc

        self = cls._restore_validated(
            k=k,
            min_k=min_k,
            n=n,
            seed=seed,
            rng_state=rng_state,
            compaction_count=compaction_count,
            retained=retained,
            levels=levels,
            has_values=bool(flags & 1),
            min_value=min_value,
            max_value=max_value,
        )
        return self

    @classmethod
    def _from_v1(cls, blob: bytes) -> "KLL":
        reader = _Reader(memoryview(blob), start=4)
        try:
            k = reader.u32()
            n = reader.u64()
            level_count = reader.u32()
            seed = reader.u64()
            levels = cls._read_levels(reader, level_count)
            reader.require_eof()
        except (struct.error, OverflowError) as exc:
            raise SerializationError("legacy KLL1 payload is truncated or malformed") from exc

        retained = sum(len(level) for level in levels)
        all_values = [v for level in levels for v in level]
        has_values = n > 0
        min_value = min(all_values) if all_values else 0.0
        max_value = max(all_values) if all_values else 0.0
        self = cls._restore_validated(
            k=k,
            min_k=k,
            n=n,
            seed=seed,
            rng_state=seed,
            compaction_count=0,
            retained=retained,
            levels=levels,
            has_values=has_values,
            min_value=min_value,
            max_value=max_value,
            legacy=True,
        )
        return self

    # ------------------------------------------------------------------
    # Diagnostics / invariants
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Raise ``AssertionError`` if internal representation invariants fail."""
        assert self._k >= self._MIN_CAPACITY
        assert self._MIN_CAPACITY <= self._min_k <= self._k
        assert 1 <= len(self._levels) <= self._MAX_LEVELS
        assert self._num_retained == sum(len(level) for level in self._levels)
        represented = 0
        for h, level in enumerate(self._levels):
            weight = 1 << h
            represented += weight * len(level)
            for value in level:
                assert math.isfinite(value)
        assert represented == self._n
        if self._n == 0:
            assert self._min_value is None and self._max_value is None
            assert self._num_retained == 0
        else:
            assert self._min_value is not None and self._max_value is not None
            assert math.isfinite(self._min_value) and math.isfinite(self._max_value)
            assert self._min_value <= self._max_value
            for level in self._levels:
                for value in level:
                    assert self._min_value <= value <= self._max_value

    def debug_state(self) -> dict:
        """Return JSON-friendly structural diagnostics."""
        return {
            "k": self._k,
            "min_k": self._min_k,
            "n": self._n,
            "num_retained": self._num_retained,
            "num_levels": len(self._levels),
            "level_sizes": [len(level) for level in self._levels],
            "level_capacities": [self._level_capacity(i) for i in range(len(self._levels))],
            "total_capacity": self._total_capacity(),
            "estimation_mode": self.is_estimation_mode,
            "min": self._min_value,
            "max": self._max_value,
            "compactions": self._compaction_count,
        }

    # ------------------------------------------------------------------
    # Core KLL engine
    # ------------------------------------------------------------------
    def _add_one(self, value: float) -> None:
        if self._n == _U64_MASK:
            raise OverflowError("total sketch weight exceeds uint64 serialization range")
        self._update_extrema(value)
        self._levels[0].append(value)
        self._n += 1
        self._num_retained += 1
        self._mark_mutated()
        self._compress_while_needed()

    def _add_weighted_binary(self, value: float, weight: int) -> None:
        if weight > _U64_MASK - self._n:
            raise OverflowError("total sketch weight exceeds uint64 serialization range")
        self._update_extrema(value)
        remaining = weight
        level = 0
        while remaining:
            if remaining & 1:
                if level >= self._MAX_LEVELS:
                    raise OverflowError("weighted update exceeds maximum representable level")
                self._ensure_levels(level + 1)
                self._levels[level].append(value)
                self._num_retained += 1
            remaining >>= 1
            level += 1
        self._n += weight
        self._mark_mutated()
        self._compress_while_needed()

    def _compress_while_needed(self) -> None:
        guard = 0
        while self._num_retained > self._total_capacity():
            level = self._find_overfull_level()
            if level is None:
                # This should be impossible: if every level is within its own
                # capacity, their sum cannot exceed total capacity.
                raise RuntimeError("KLL capacity accounting became inconsistent")
            self._compact_level(level)
            guard += 1
            if guard > 10_000:
                raise RuntimeError("KLL compaction did not converge")

    def _find_overfull_level(self) -> Optional[int]:
        for level, items in enumerate(self._levels):
            if len(items) > self._level_capacity(level):
                return level
        return None

    def _compact_level(self, level: int) -> None:
        items = self._levels[level]
        if len(items) < 2:
            raise RuntimeError("attempted to compact a non-compactable level")
        items.sort()

        leftover: List[float]
        compactable: Sequence[float]
        if len(items) & 1:
            if self._next_bit():
                leftover = [items[-1]]
                compactable = items[:-1]
            else:
                leftover = [items[0]]
                compactable = items[1:]
        else:
            leftover = []
            compactable = items

        # One parity bit is chosen for the whole compaction. This is the key
        # unbiased KLL compactor operation; do not independently randomize pairs.
        offset = self._next_bit()
        promoted = list(compactable[offset::2])
        if not promoted:
            raise RuntimeError("KLL compaction produced no promoted items")

        self._levels[level] = leftover
        self._ensure_levels(level + 2)
        self._levels[level + 1].extend(promoted)
        old_count = len(items)
        self._num_retained += len(leftover) + len(promoted) - old_count
        self._compaction_count += 1
        self._mark_mutated()

    def _level_capacity(self, level: int) -> int:
        """Return the KLL geometric capacity for one level.

        The top level has capacity ``k``. Moving one level downward multiplies
        target capacity by 2/3, bounded by ``_MIN_LEVEL_CAPACITY``. Integer
        arithmetic avoids platform-dependent floating rounding in this policy.
        """
        if not (0 <= level < len(self._levels)):
            raise IndexError("level out of range")
        depth = len(self._levels) - level - 1
        numerator = self._k * (2 ** depth)
        denominator = 3 ** depth
        rounded = (numerator + denominator // 2) // denominator
        return max(self._MIN_LEVEL_CAPACITY, int(rounded))

    def _total_capacity(self) -> int:
        return sum(self._level_capacity(i) for i in range(len(self._levels)))

    def _ensure_levels(self, count: int) -> None:
        if count > self._MAX_LEVELS:
            raise OverflowError("sketch exceeds maximum supported level count")
        while len(self._levels) < count:
            self._levels.append([])

    # ------------------------------------------------------------------
    # Query materialization cache
    # ------------------------------------------------------------------
    def _query_view(self) -> Tuple[List[float], List[int]]:
        if self._cache_generation == self._generation:
            return self._cache_values, self._cache_prefix
        self._require_nonempty()

        weighted: List[Tuple[float, int]] = []
        weighted_extend = weighted.extend
        for level, items in enumerate(self._levels):
            if not items:
                continue
            weight = 1 << level
            weighted_extend((value, weight) for value in items)
        weighted.sort(key=lambda pair: pair[0])

        values: List[float] = []
        prefix: List[int] = []
        cumulative = 0
        for value, weight in weighted:
            cumulative += weight
            values.append(value)
            prefix.append(cumulative)

        if cumulative != self._n:
            raise RuntimeError("materialized KLL weight does not equal n")
        self._cache_values = values
        self._cache_prefix = prefix
        self._cache_generation = self._generation
        return values, prefix

    def _materialize_aligned(self) -> Tuple[List[float], List[float]]:
        """Backwards-compatible diagnostic helper returning value/weight arrays."""
        values, prefix = self._query_view()
        weights: List[float] = []
        previous = 0
        for current in prefix:
            weights.append(float(current - previous))
            previous = current
        return values.copy(), weights

    def _quantiles_from_probabilities(self, qs: List[float]) -> List[float]:
        self._require_nonempty()
        values, prefix = self._query_view()
        ordered = sorted(enumerate(qs), key=lambda item: item[1])
        out = [0.0] * len(qs)
        lo = 0
        for original_index, q in ordered:
            if q <= 0.0:
                out[original_index] = self.min_value
                continue
            if q >= 1.0:
                out[original_index] = self.max_value
                lo = len(values) - 1
                continue
            target = q * (self._n - 1)
            pos = bisect_right(prefix, target, lo=lo)
            if pos >= len(values):
                pos = len(values) - 1
            out[original_index] = values[pos]
            lo = pos
        return out

    # ------------------------------------------------------------------
    # Validation / coercion / RNG helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_seed(seed: int) -> int:
        if isinstance(seed, bool):
            raise TypeError("rng_seed must be an integer")
        try:
            value = int(seed)
        except (TypeError, ValueError) as exc:
            raise TypeError("rng_seed must be an integer") from exc
        if value != seed:
            raise ValueError("rng_seed must be an integer")
        return value & _U64_MASK

    @staticmethod
    def _coerce_value(value: float) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError("x must be a real number") from exc
        if not math.isfinite(out):
            raise ValueError("x must be finite")
        return out

    @staticmethod
    def _coerce_weight(weight: float) -> int:
        try:
            value = float(weight)
        except (TypeError, ValueError) as exc:
            raise TypeError("weight must be a positive integer") from exc
        if not math.isfinite(value):
            raise ValueError("weight must be finite")
        if value <= 0:
            raise ValueError("weight must be > 0")
        rounded = int(round(value))
        if abs(value - rounded) > 1e-9:
            raise ValueError("weight must be an integer")
        return rounded

    @staticmethod
    def _coerce_probability(value: float, name: str) -> float:
        try:
            q = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a real number") from exc
        if not math.isfinite(q) or not (0.0 <= q <= 1.0):
            raise ValueError(f"{name} must be in [0,1]")
        return q

    def _update_extrema(self, value: float) -> None:
        if self._min_value is None or value < self._min_value:
            self._min_value = value
        if self._max_value is None or value > self._max_value:
            self._max_value = value

    def _mark_mutated(self) -> None:
        self._generation += 1
        self._cache_generation = -1

    def _require_nonempty(self) -> None:
        if self._n == 0:
            raise ValueError("empty sketch")

    def _next_u64(self) -> int:
        self._rng_state = (self._rng_state + _SPLITMIX_GAMMA) & _U64_MASK
        z = self._rng_state
        z = ((z ^ (z >> 30)) * _SPLITMIX_M1) & _U64_MASK
        z = ((z ^ (z >> 27)) * _SPLITMIX_M2) & _U64_MASK
        return (z ^ (z >> 31)) & _U64_MASK

    def _next_bit(self) -> int:
        return self._next_u64() & 1

    # ------------------------------------------------------------------
    # Deserialization helpers
    # ------------------------------------------------------------------
    @classmethod
    def _read_levels(cls, reader: "_Reader", level_count: int) -> List[List[float]]:
        if not (1 <= level_count <= cls._MAX_LEVELS):
            raise SerializationError("invalid level count")
        levels: List[List[float]] = []
        for _ in range(level_count):
            length = reader.u32()
            if length > reader.remaining // 8:
                raise SerializationError("level length exceeds remaining payload")
            values = reader.f64_array(length)
            if any(not math.isfinite(v) for v in values):
                raise SerializationError("serialized sketch contains non-finite values")
            levels.append(values)
        return levels

    @classmethod
    def _restore_validated(
        cls,
        *,
        k: int,
        min_k: int,
        n: int,
        seed: int,
        rng_state: int,
        compaction_count: int,
        retained: int,
        levels: List[List[float]],
        has_values: bool,
        min_value: float,
        max_value: float,
        legacy: bool = False,
    ) -> "KLL":
        try:
            self = cls(k, seed)
        except (TypeError, ValueError) as exc:
            raise SerializationError("serialized sketch has invalid capacity") from exc
        if not (cls._MIN_CAPACITY <= min_k <= k):
            raise SerializationError("serialized sketch has invalid min_k")

        actual_retained = sum(len(level) for level in levels)
        if retained != actual_retained:
            raise SerializationError("retained-item count mismatch")
        represented = sum((1 << h) * len(level) for h, level in enumerate(levels))
        if represented != n:
            raise SerializationError("serialized level weights do not equal n")
        if has_values != (n > 0):
            raise SerializationError("serialized empty/non-empty flag is inconsistent")
        if n == 0 and actual_retained != 0:
            raise SerializationError("empty sketch cannot contain retained items")
        if n > 0 and actual_retained == 0:
            raise SerializationError("non-empty sketch has no retained items")

        if n > 0:
            if not legacy and (not math.isfinite(min_value) or not math.isfinite(max_value)):
                raise SerializationError("serialized extrema must be finite")
            if min_value > max_value:
                raise SerializationError("serialized extrema are reversed")
            for level in levels:
                for value in level:
                    if value < min_value or value > max_value:
                        raise SerializationError("retained item lies outside serialized extrema")

        self._levels = levels
        self._min_k = min_k
        self._n = n
        self._rng_state = rng_state & _U64_MASK
        self._compaction_count = compaction_count
        self._num_retained = actual_retained
        self._min_value = min_value if n else None
        self._max_value = max_value if n else None
        self._generation = 1
        self._cache_generation = -1
        self._cache_values = []
        self._cache_prefix = []
        self.validate()
        return self


class _Reader:
    __slots__ = ("_view", "_offset")

    def __init__(self, view: memoryview, start: int = 0):
        self._view = view
        self._offset = start

    @property
    def remaining(self) -> int:
        return len(self._view) - self._offset

    def _take(self, size: int) -> int:
        if size < 0 or self._offset + size > len(self._view):
            raise SerializationError("serialized sketch is truncated")
        start = self._offset
        self._offset += size
        return start

    def u8(self) -> int:
        return struct.unpack_from(">B", self._view, self._take(1))[0]

    def u32(self) -> int:
        return struct.unpack_from(">I", self._view, self._take(4))[0]

    def u64(self) -> int:
        return struct.unpack_from(">Q", self._view, self._take(8))[0]

    def f64(self) -> float:
        return struct.unpack_from(">d", self._view, self._take(8))[0]

    def f64_array(self, count: int) -> List[float]:
        if count == 0:
            return []
        start = self._take(8 * count)
        return list(struct.unpack_from(">" + "d" * count, self._view, start))

    def require_eof(self) -> None:
        if self.remaining != 0:
            raise SerializationError("serialized sketch contains trailing bytes")
