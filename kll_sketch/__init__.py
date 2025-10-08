"""kll_sketch package public API."""
from ._metadata import __version__
from .kll_sketch import KLL


class KLLSketch(KLL):
    """Alias for :class:`KLL` used by the benchmarking utilities."""


__all__ = ["KLL", "KLLSketch", "__version__"]
