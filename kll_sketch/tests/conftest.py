"""Pytest configuration ensuring the package is importable during tests."""
from __future__ import annotations

import sys
from pathlib import Path

# When pytest collects tests inside the installed package directory, the
# repository root (which contains the ``kll_sketch`` package) might not be on
# ``sys.path``.  Add it explicitly so ``from kll_sketch import KLL`` works even
# when the tests are executed without installing the project as a package.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
