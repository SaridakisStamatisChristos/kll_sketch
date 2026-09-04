"""Public API for :mod:`kll_sketch`."""

from .kll_sketch import (
    KLL,
    LEGACY_SERIAL_FORMAT_MAGIC,
    SERIAL_FORMAT_MAGIC,
    SERIAL_FORMAT_VERSION,
    SerializationError,
)

__version__ = "2.0.0"

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
    "__version__",
]
