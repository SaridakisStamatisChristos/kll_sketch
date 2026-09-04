"""Runtime dispatch for the optional :mod:`kll_sketch._native` extension.

The public ``KLL`` class is never replaced. Instead, this module installs thin
method wrappers on the existing class and keeps the original Python methods as
fallbacks. This preserves class identity, serialization behavior, ``copy()``,
and the zero-dependency runtime path.
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

_ORIGINAL_EXTEND = KLL.extend
_ORIGINAL_COMPACT_LEVEL = KLL._compact_level
_ORIGINAL_QUERY_VIEW = KLL._query_view
_ORIGINAL_RANKS = KLL.ranks
_ORIGINAL_QUANTILES_FROM_PROBABILITIES = KLL._quantiles_from_probabilities


def native_available() -> bool:
    """Return whether the compiled extension can be imported."""
    return _native_impl is not None


def native_enabled() -> bool:
    """Return whether runtime dispatch currently uses the native backend."""
    return bool(_ENABLED and _native_impl is not None)


def set_native_enabled(enabled: bool) -> None:
    """Enable or disable native dispatch for this process.

    Enabling requires the compiled extension to be importable. This switch is
    primarily useful for deterministic differential tests and benchmarks.
    """
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
        }
    info = dict(_native_impl.info())
    info["available"] = True
    info["enabled"] = native_enabled()
    return info


def _candidate_for_native_batch(xs: object) -> bool:
    if isinstance(xs, (str, bytes, bytearray)):
        return False
    try:
        len(xs)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        return False
    return True


def _native_extend(self: KLL, xs: object) -> None:
    if not native_enabled() or not _candidate_for_native_batch(xs):
        _ORIGINAL_EXTEND(self, xs)  # type: ignore[arg-type]
        return
    assert _native_impl is not None
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
        # The native attempt is transactional: no Python sketch state is
        # mutated before success. Replay through the canonical Python path so
        # coercion errors and partial-progress semantics remain unchanged.
        _ORIGINAL_EXTEND(self, xs)  # type: ignore[arg-type]
        return

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


def _native_compact_level(self: KLL, level: int) -> None:
    if not native_enabled():
        _ORIGINAL_COMPACT_LEVEL(self, level)
        return
    assert _native_impl is not None
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


def _native_query_view(self: KLL):
    if not native_enabled():
        return _ORIGINAL_QUERY_VIEW(self)
    if self._cache_generation == self._generation:
        return self._cache_values, self._cache_prefix
    self._require_nonempty()
    assert _native_impl is not None
    values, prefix = _native_impl.materialize(self._levels, self._n)
    self._cache_values = values
    self._cache_prefix = prefix
    self._cache_generation = self._generation
    return values, prefix


def _native_ranks(self: KLL, xs, *, inclusive: bool = True):
    if not native_enabled() or not _candidate_for_native_batch(xs):
        return _ORIGINAL_RANKS(self, xs, inclusive=inclusive)
    if self._n == 0:
        return [0.0 for _ in xs]
    assert _native_impl is not None
    values, prefix = self._query_view()
    try:
        return _native_impl.ranks_many(values, prefix, xs, inclusive)
    except (TypeError, ValueError):
        return _ORIGINAL_RANKS(self, xs, inclusive=inclusive)


def _native_quantiles_from_probabilities(self: KLL, qs: list[float]):
    if not native_enabled():
        return _ORIGINAL_QUANTILES_FROM_PROBABILITIES(self, qs)
    self._require_nonempty()
    assert _native_impl is not None
    values, prefix = self._query_view()
    return _native_impl.quantiles_many(
        values,
        prefix,
        self._n,
        qs,
        self.min_value,
        self.max_value,
    )


def install_native_acceleration() -> None:
    """Install dispatch wrappers exactly once."""
    global _INSTALLED
    if _INSTALLED:
        return
    KLL.extend = _native_extend  # type: ignore[method-assign]
    KLL.update_many = _native_extend  # type: ignore[method-assign]
    KLL._compact_level = _native_compact_level  # type: ignore[method-assign]
    KLL._query_view = _native_query_view  # type: ignore[method-assign]
    KLL.ranks = _native_ranks  # type: ignore[method-assign]
    KLL._quantiles_from_probabilities = _native_quantiles_from_probabilities  # type: ignore[method-assign]
    _INSTALLED = True
