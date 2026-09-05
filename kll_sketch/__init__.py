"""Public API for :mod:`kll_sketch`."""

from .kll_sketch import (
    KLL,
    LEGACY_SERIAL_FORMAT_MAGIC,
    SERIAL_FORMAT_MAGIC,
    SERIAL_FORMAT_VERSION,
    SerializationError,
)
from ._native_runtime import (
    _NativeStateHandle,
    install_native_acceleration,
    native_available,
    native_backend_info,
    native_enabled,
    set_native_enabled as _set_native_enabled,
)

install_native_acceleration()

try:
    from . import _native as _native_impl
except ImportError:
    _native_impl = None

# Keep the Python runtime dispatcher as the canonical semantic fallback, then
# replace only proven resident-state hot methods with direct C-level
# descriptors. Keyword/disabled/nonresident/error paths tail-call these Python
# fallbacks, preserving the public API rather than creating a second API.
if _native_impl is not None and hasattr(_native_impl, "install_type_fastpaths"):
    _native_quantiles_fallback = KLL.quantiles_at
    _native_merge_fallback = KLL.merge
    _native_impl.install_type_fastpaths(
        KLL,
        _native_quantiles_fallback,
        _native_merge_fallback,
        _NativeStateHandle,
    )
    if hasattr(_native_impl, "install_v32_merge_fastpath"):
        _native_impl.install_v32_merge_fastpath()
    if hasattr(_native_impl, "install_v32_sequence_fastpath"):
        _native_impl.install_v32_sequence_fastpath()
    if hasattr(_native_impl, "install_v32_slot_fastpath"):
        _native_impl.install_v32_slot_fastpath()
    if hasattr(_native_impl, "set_type_fastpaths_enabled"):
        _native_impl.set_type_fastpaths_enabled(native_enabled())
else:
    _native_quantiles_fallback = None
    _native_merge_fallback = None


def set_native_enabled(enabled: bool) -> None:
    """Enable or disable native dispatch, including direct C-level hot paths."""
    _set_native_enabled(enabled)
    if _native_impl is not None and hasattr(_native_impl, "set_type_fastpaths_enabled"):
        _native_impl.set_type_fastpaths_enabled(native_enabled())


__version__ = "3.2.0"

# Historical benchmark-facing name. Alias rather than subclass so isinstance,
# serialization and type identity stay simple and predictable.
KLLSketch = KLL

__all__ = [
    "KLL",
    "KLLSketch",
    "SerializationError",
    "SERIAL_FORMAT_MAGIC",
    "SERIAL_FORMAT_VERSION",
    "LEGACY_SERIAL_FORMAT_MAGIC",
    "native_available",
    "native_enabled",
    "native_backend_info",
    "set_native_enabled",
    "__version__",
]
