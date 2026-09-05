"""Runtime dispatch for the optional native KLL accelerator.

The public :class:`KLL` object remains the API boundary. When the native
backend is present, bulk updates, quantile/rank queries, and merges can keep an
opaque C++ state resident across calls. Python levels are materialized only
when a Python-only operation actually needs them.
"""
from __future__ import annotations

import os
from typing import Any

from .kll_sketch import KLL

try:
    from . import _native as _native_impl
except ImportError:
    _native_impl = None

_DISABLED_BY_ENV = os.environ.get("KLL_SKETCH_DISABLE_NATIVE", "").strip().lower() in {
    "1", "true", "yes", "on"
}
_ENABLED = _native_impl is not None and not _DISABLED_BY_ENV
_INSTALLED = False
_MAX_EXACT_DOUBLE_INTEGER = 1 << 53
_STATE_SENTINEL = -2
_HAS_PERSISTENT_STATE = bool(
    _native_impl is not None
    and all(
        hasattr(_native_impl, name)
        for name in (
            "state_create",
            "state_export",
            "state_ingest",
            "state_merge",
            "state_quantiles",
            "state_ranks",
        )
    )
)
_HAS_COMPACT_MERGE_RESULT = bool(
    _native_impl is not None and hasattr(_native_impl, "state_merge_retained")
)

_ORIGINAL_ADD = KLL.add
_ORIGINAL_COPY = KLL.copy
_ORIGINAL_TO_BYTES = KLL.to_bytes
_ORIGINAL_VALIDATE = KLL.validate
_ORIGINAL_DEBUG_STATE = KLL.debug_state
_ORIGINAL_EXTEND = KLL.extend
_ORIGINAL_MERGE = KLL.merge
_ORIGINAL_COMPACT_LEVEL = KLL._compact_level
_ORIGINAL_QUERY_VIEW = KLL._query_view
_ORIGINAL_RANK = KLL.rank
_ORIGINAL_RANKS = KLL.ranks
_ORIGINAL_QUANTILES_AT = KLL.quantiles_at
_ORIGINAL_QUANTILES_FROM_PROBABILITIES = KLL._quantiles_from_probabilities


class _NativeStateHandle:
    __slots__ = ("capsule",)

    def __init__(self, capsule: object) -> None:
        self.capsule = capsule


def native_available() -> bool:
    """Return whether the compiled extension can be imported."""
    return _native_impl is not None


def native_enabled() -> bool:
    """Return whether runtime dispatch currently uses the native backend."""
    return bool(_ENABLED and _native_impl is not None)


def set_native_enabled(enabled: bool) -> None:
    """Enable or disable native dispatch for this process."""
    global _ENABLED
    if enabled and _native_impl is None:
        raise RuntimeError("kll-sketch native extension is not available")
    _ENABLED = bool(enabled)


def native_backend_info() -> dict[str, Any]:
    """Return JSON-friendly backend diagnostics."""
    if _native_impl is None:
        return {
            "available": False,
            "enabled": False,
            "api_version": None,
            "compiler": None,
            "simd": None,
            "persistent_state": False,
        }
    info = dict(_native_impl.info())
    info["available"] = True
    info["enabled"] = native_enabled()
    info["persistent_state"] = _HAS_PERSISTENT_STATE
    return info


def _candidate_for_native_batch(xs: object) -> bool:
    """Return whether probing *xs* cannot consume a one-shot iterator."""
    if isinstance(xs, (str, bytes, bytearray)):
        return False
    try:
        len(xs)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        return False
    return callable(getattr(xs, "__getitem__", None))


def _state_handle(self: KLL) -> _NativeStateHandle | None:
    if self._cache_generation != _STATE_SENTINEL:
        return None
    value = self._cache_values
    return value if isinstance(value, _NativeStateHandle) else None


def _store_state(self: KLL, handle: _NativeStateHandle) -> None:
    # The Python handle remains the ownership/fallback representation. The raw
    # capsule is mirrored into the otherwise-unused resident cache-prefix slot
    # so direct C-level methods can recover SketchState with one slot lookup
    # instead of traversing Python handle attributes on every hot call.
    self._cache_values = handle  # type: ignore[assignment]
    self._cache_prefix = handle.capsule  # type: ignore[assignment]
    self._cache_generation = _STATE_SENTINEL


def _apply_stats(self: KLL, handle: _NativeStateHandle, stats: object, *, changed: bool) -> None:
    n, retained, rng_state, compactions, min_value, max_value, _num_levels = stats  # type: ignore[misc]
    self._n = int(n)
    self._num_retained = int(retained)
    self._rng_state = int(rng_state)
    self._compaction_count = int(compactions)
    self._min_value = min_value
    self._max_value = max_value
    if changed:
        self._generation += 1
    _store_state(self, handle)


def _sync_state(self: KLL) -> None:
    """Materialize an opaque native state back into canonical Python levels."""
    handle = _state_handle(self)
    if handle is None:
        return
    assert _native_impl is not None
    levels, stats = _native_impl.state_export(handle.capsule)
    n, retained, rng_state, compactions, min_value, max_value, _num_levels = stats
    self._levels = levels
    self._n = int(n)
    self._num_retained = int(retained)
    self._rng_state = int(rng_state)
    self._compaction_count = int(compactions)
    self._min_value = min_value
    self._max_value = max_value
    self._cache_values = []
    self._cache_prefix = []
    self._cache_generation = -1


def _ensure_state(self: KLL) -> _NativeStateHandle:
    handle = _state_handle(self)
    if handle is not None:
        return handle
    if not (_HAS_PERSISTENT_STATE and native_enabled()):
        raise RuntimeError("persistent native state is not available")
    assert _native_impl is not None
    capsule = _native_impl.state_create(
        self._levels,
        self._n,
        self._k,
        self._rng_state,
        self._compaction_count,
        self._num_retained,
        self._min_value,
        self._max_value,
        self._MIN_LEVEL_CAPACITY,
        self._MAX_LEVELS,
    )
    handle = _NativeStateHandle(capsule)
    _store_state(self, handle)
    return handle


def _legacy_native_extend(self: KLL, xs: object) -> bool:
    if _native_impl is None or not hasattr(_native_impl, "ingest_batch"):
        return False
    try:
        result = _native_impl.ingest_batch(
            self._levels,
            self._n,
            self._k,
            self._rng_state,
            self._compaction_count,
            self._num_retained,
            self._min_value,
            self._max_value,
            xs,
            self._MIN_LEVEL_CAPACITY,
            self._MAX_LEVELS,
        )
    except (TypeError, ValueError, OverflowError):
        return False
    levels, n, retained, rng_state, compactions, min_value, max_value = result
    changed = int(n) != self._n
    self._levels = levels
    self._n = int(n)
    self._num_retained = int(retained)
    self._rng_state = int(rng_state)
    self._compaction_count = int(compactions)
    self._min_value = min_value
    self._max_value = max_value
    if changed:
        self._mark_mutated()
    return True


def _native_extend(self: KLL, xs: object) -> None:
    if not native_enabled() or not _candidate_for_native_batch(xs):
        _sync_state(self)
        _ORIGINAL_EXTEND(self, xs)  # type: ignore[arg-type]
        return

    if _HAS_PERSISTENT_STATE:
        assert _native_impl is not None
        handle = _ensure_state(self)
        old_n = self._n
        try:
            stats = _native_impl.state_ingest(handle.capsule, xs)
        except (TypeError, ValueError, OverflowError):
            _sync_state(self)
            _ORIGINAL_EXTEND(self, xs)  # type: ignore[arg-type]
            return
        _apply_stats(self, handle, stats, changed=int(stats[0]) != old_n)
        return

    if not _legacy_native_extend(self, xs):
        _ORIGINAL_EXTEND(self, xs)  # type: ignore[arg-type]


def _merge_visible_state(self: KLL, other: KLL, retained: int, old_n: int) -> None:
    """Mirror Python-visible fields while resident C++ state remains authoritative."""
    self._n = old_n + other._n
    self._num_retained = retained
    if self._min_value is None:
        self._min_value = other._min_value
        self._max_value = other._max_value
    else:
        if other._min_value is not None and other._min_value < self._min_value:
            self._min_value = other._min_value
        if other._max_value is not None and (self._max_value is None or other._max_value > self._max_value):
            self._max_value = other._max_value
    self._generation += 1


def _native_merge(self: KLL, other: KLL) -> None:
    if not isinstance(other, KLL):
        raise TypeError("merge expects KLL")
    if other is self:
        raise ValueError("cannot merge a sketch with itself")
    if other._n == 0:
        return
    if not (native_enabled() and _HAS_PERSISTENT_STATE):
        _sync_state(self)
        _sync_state(other)
        _ORIGINAL_MERGE(self, other)
        return

    if other.is_estimation_mode:
        self._min_k = min(self._min_k, other._min_k)
    assert _native_impl is not None
    dst = _ensure_state(self)
    src = _ensure_state(other)
    old_n = self._n
    try:
        if _HAS_COMPACT_MERGE_RESULT:
            retained = int(_native_impl.state_merge_retained(dst.capsule, src.capsule))
            _merge_visible_state(self, other, retained, old_n)
            _store_state(self, dst)
            return
        stats = _native_impl.state_merge(dst.capsule, src.capsule)
    except (TypeError, ValueError, OverflowError):
        _sync_state(self)
        _sync_state(other)
        _ORIGINAL_MERGE(self, other)
        return
    _apply_stats(self, dst, stats, changed=int(stats[0]) != old_n)


def _native_quantiles_at(self: KLL, probabilities) -> list[float]:
    if not (native_enabled() and _HAS_PERSISTENT_STATE) or self._n > _MAX_EXACT_DOUBLE_INTEGER:
        _sync_state(self)
        return _ORIGINAL_QUANTILES_AT(self, probabilities)
    if not _candidate_for_native_batch(probabilities):
        _sync_state(self)
        return _ORIGINAL_QUANTILES_AT(self, probabilities)
    if len(probabilities) == 0:
        return []
    if self._n == 0:
        return _ORIGINAL_QUANTILES_AT(self, probabilities)
    assert _native_impl is not None
    handle = _ensure_state(self)
    try:
        return _native_impl.state_quantiles(handle.capsule, probabilities)
    except (TypeError, ValueError, OverflowError):
        _sync_state(self)
        return _ORIGINAL_QUANTILES_AT(self, probabilities)


def _native_quantiles_from_probabilities(self: KLL, qs: list[float]) -> list[float]:
    if not (native_enabled() and _HAS_PERSISTENT_STATE) or self._n > _MAX_EXACT_DOUBLE_INTEGER:
        _sync_state(self)
        return _ORIGINAL_QUANTILES_FROM_PROBABILITIES(self, qs)
    self._require_nonempty()
    assert _native_impl is not None
    handle = _ensure_state(self)
    try:
        return _native_impl.state_quantiles(handle.capsule, qs)
    except (TypeError, ValueError, OverflowError):
        _sync_state(self)
        return _ORIGINAL_QUANTILES_FROM_PROBABILITIES(self, qs)


def _native_ranks(self: KLL, xs, *, inclusive: bool = True) -> list[float]:
    if self._n == 0:
        return [0.0 for _ in xs]
    if not (native_enabled() and _HAS_PERSISTENT_STATE) or not _candidate_for_native_batch(xs):
        _sync_state(self)
        return _ORIGINAL_RANKS(self, xs, inclusive=inclusive)
    assert _native_impl is not None
    handle = _ensure_state(self)
    try:
        return _native_impl.state_ranks(handle.capsule, xs, inclusive)
    except (TypeError, ValueError, OverflowError):
        _sync_state(self)
        return _ORIGINAL_RANKS(self, xs, inclusive=inclusive)


def _native_rank(self: KLL, x: float, *, inclusive: bool = True) -> float:
    if self._n == 0:
        return 0.0
    if not (native_enabled() and _HAS_PERSISTENT_STATE):
        _sync_state(self)
        return _ORIGINAL_RANK(self, x, inclusive=inclusive)
    assert _native_impl is not None
    handle = _ensure_state(self)
    try:
        return float(_native_impl.state_ranks(handle.capsule, (x,), inclusive)[0])
    except (TypeError, ValueError, OverflowError):
        _sync_state(self)
        return _ORIGINAL_RANK(self, x, inclusive=inclusive)


def _native_query_view(self: KLL):
    _sync_state(self)
    return _ORIGINAL_QUERY_VIEW(self)


def _native_compact_level(self: KLL, level: int) -> None:
    _sync_state(self)
    if not native_enabled() or _native_impl is None or not hasattr(_native_impl, "compact_level"):
        _ORIGINAL_COMPACT_LEVEL(self, level)
        return
    items = self._levels[level]
    if len(items) < 2:
        raise RuntimeError("attempted to compact a non-compactable level")
    keep_high = bool(self._next_bit()) if len(items) & 1 else False
    offset = self._next_bit()
    leftover, promoted = _native_impl.compact_level(items, keep_high, offset)
    if not promoted:
        raise RuntimeError("KLL compaction produced no promoted items")
    old_count = len(items)
    self._levels[level] = leftover
    self._ensure_levels(level + 2)
    self._levels[level + 1].extend(promoted)
    self._num_retained += len(leftover) + len(promoted) - old_count
    self._compaction_count += 1
    self._mark_mutated()


def _sync_then(original):
    def wrapped(self: KLL, *args, **kwargs):
        _sync_state(self)
        return original(self, *args, **kwargs)
    return wrapped


_NATIVE_ADD = _sync_then(_ORIGINAL_ADD)
_NATIVE_COPY = _sync_then(_ORIGINAL_COPY)
_NATIVE_TO_BYTES = _sync_then(_ORIGINAL_TO_BYTES)
_NATIVE_VALIDATE = _sync_then(_ORIGINAL_VALIDATE)
_NATIVE_DEBUG_STATE = _sync_then(_ORIGINAL_DEBUG_STATE)


def install_native_acceleration() -> None:
    """Install dispatch wrappers exactly once without replacing ``KLL`` itself."""
    global _INSTALLED
    if _INSTALLED:
        return
    KLL.add = _NATIVE_ADD  # type: ignore[method-assign]
    KLL.copy = _NATIVE_COPY  # type: ignore[method-assign]
    KLL.to_bytes = _NATIVE_TO_BYTES  # type: ignore[method-assign]
    KLL.validate = _NATIVE_VALIDATE  # type: ignore[method-assign]
    KLL.debug_state = _NATIVE_DEBUG_STATE  # type: ignore[method-assign]
    KLL.extend = _native_extend  # type: ignore[method-assign]
    KLL.update_many = _native_extend  # type: ignore[method-assign]
    KLL.merge = _native_merge  # type: ignore[method-assign]
    KLL._compact_level = _native_compact_level  # type: ignore[method-assign]
    KLL._query_view = _native_query_view  # type: ignore[method-assign]
    KLL.rank = _native_rank  # type: ignore[method-assign]
    KLL.ranks = _native_ranks  # type: ignore[method-assign]
    KLL.quantiles_at = _native_quantiles_at  # type: ignore[method-assign]
    KLL._quantiles_from_probabilities = _native_quantiles_from_probabilities  # type: ignore[method-assign]
    _INSTALLED = True
