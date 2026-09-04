"""Project metadata shared by the runtime-independent build backend."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Author:
    name: str
    email: str | None = None


PROJECT_METADATA: Mapping[str, object] = {
    "name": "kll-sketch",
    "version": "2.0.0",
    "summary": "High-integrity KLL streaming quantile sketch with strict serialization and rank-space validation",
    "readme": {"path": _PROJECT_ROOT / "README.md", "content_type": "text/markdown"},
    "requires_python": ">=3.10",
    "license": {"text": "Apache-2.0", "files": ["LICENSE"]},
    "authors": [Author(name="Stamatis-Christos Saridakis", email="stamatis@saridakis.dev")],
    "keywords": ["quantiles", "sketch", "streaming", "kll", "data-structures", "approximate-algorithms"],
    "classifiers": [
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    "urls": {
        "Homepage": "https://github.com/SaridakisStamatisChristos/kll_sketch",
        "Repository": "https://github.com/SaridakisStamatisChristos/kll_sketch",
        "Documentation": "https://github.com/SaridakisStamatisChristos/kll_sketch/tree/main/docs",
        "Issues": "https://github.com/SaridakisStamatisChristos/kll_sketch/issues",
    },
    "optional-dependencies": {
        "test": ["pytest>=7.4", "hypothesis>=6.88", "pytest-cov>=4.1"],
        "compare": ["datasketches>=5.0"],
    },
}

SUPPORTED_PYTHON_VERSIONS: List[str] = ["3.10", "3.11", "3.12", "3.13", "3.14"]
SUPPORTED_PLATFORMS: List[str] = ["Linux", "macOS", "Windows"]
__version__ = str(PROJECT_METADATA["version"])
