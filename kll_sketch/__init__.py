# kll_sketch/__init__.py
from .kll_sketch import KLL


class KLLSketch(KLL):
    """Alias for :class:`KLL` used by the benchmarking utilities."""


__all__ = ["KLL", "KLLSketch"]
