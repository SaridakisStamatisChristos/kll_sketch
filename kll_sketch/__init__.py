"""Public API for :mod:`kll_sketch`."""

from .kll_sketch import (
    KLL,
    LEGACY_SERIAL_FORMAT_MAGIC,
    SERIAL_FORMAT_MAGIC,
    SERIAL_FORMAT_VERSION,
    SerializationError,
)
from ._native_runtime import (
    install_native_acceleration,
    native_available,
    native_backend_info,
    native_enabled,
    set_native_enabled,
)

install_native_acceleration()

__version__ = "3.0.0"

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
