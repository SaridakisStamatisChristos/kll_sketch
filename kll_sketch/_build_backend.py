"""Small standards-oriented PEP 517 backend with no third-party dependencies."""
from __future__ import annotations

import base64
from hashlib import sha256
from pathlib import Path
import shutil
import tarfile
import tempfile
from typing import Iterable, Mapping
from zipfile import ZIP_DEFLATED, ZipFile

from ._metadata import PROJECT_METADATA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_NAME = str(PROJECT_METADATA["name"])
_VERSION = str(PROJECT_METADATA["version"])
_NORMALIZED_NAME = PACKAGE_NAME.replace("-", "_")
_DIST_INFO = f"{_NORMALIZED_NAME}-{_VERSION}.dist-info"


def _build_metadata_text() -> str:
    readme_info = PROJECT_METADATA["readme"]
    readme_path = Path(readme_info["path"])  # type: ignore[index]
    readme_content_type = str(readme_info["content_type"])  # type: ignore[index]
    lines = ["Metadata-Version: 2.4", f"Name: {PACKAGE_NAME}", f"Version: {_VERSION}"]
    summary = PROJECT_METADATA.get("summary")
    if summary:
        lines.append(f"Summary: {summary}")
    authors = PROJECT_METADATA.get("authors", [])
    if authors:
        author = authors[0]  # type: ignore[index]
        name = getattr(author, "name", "")
        email = getattr(author, "email", None)
        if name:
            lines.append(f"Author: {name}")
        if email:
            lines.append(f"Author-email: {name} <{email}>")
    requires_python = PROJECT_METADATA.get("requires_python")
    if requires_python:
        lines.append(f"Requires-Python: {requires_python}")
    keywords = PROJECT_METADATA.get("keywords", [])
    if keywords:
        lines.append(f"Keywords: {', '.join(keywords)}")
    for classifier in PROJECT_METADATA.get("classifiers", []):
        lines.append(f"Classifier: {classifier}")
    for label, url in PROJECT_METADATA.get("urls", {}).items():  # type: ignore[union-attr]
        lines.append(f"Project-URL: {label}, {url}")
    license_info = PROJECT_METADATA.get("license", {})
    if isinstance(license_info, dict):
        expression = license_info.get("text")
        if expression:
            lines.append(f"License-Expression: {expression}")
        for license_file in license_info.get("files", []):
            lines.append(f"License-File: {license_file}")
    for extra, requirements in PROJECT_METADATA.get("optional-dependencies", {}).items():  # type: ignore[union-attr]
        lines.append(f"Provides-Extra: {extra}")
        for requirement in requirements:
            lines.append(f"Requires-Dist: {requirement}; extra == '{extra}'")
    lines.extend([f"Description-Content-Type: {readme_content_type}", ""])
    return "\n".join(lines) + "\n" + readme_path.read_text(encoding="utf-8")


def _write_metadata(dist_info: Path) -> None:
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(_build_metadata_text(), encoding="utf-8")
    (dist_info / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: kll-sketch self-hosted backend\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        encoding="utf-8",
    )
    license_info = PROJECT_METADATA.get("license", {})
    if isinstance(license_info, dict):
        for rel_path in license_info.get("files", []):
            source = PROJECT_ROOT / rel_path
            (dist_info / Path(rel_path).name).write_bytes(source.read_bytes())


def _iter_files(directory: Path) -> Iterable[Path]:
    return (p for p in sorted(directory.rglob("*")) if p.is_file())


def _record_for(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    digest = base64.urlsafe_b64encode(sha256(path.read_bytes()).digest()).decode().rstrip("=")
    return f"{rel},sha256={digest},{path.stat().st_size}"


def _write_record(dist_info: Path, wheel_root: Path) -> None:
    record = dist_info / "RECORD"
    lines = [_record_for(p, wheel_root) for p in _iter_files(wheel_root) if p != record]
    lines.append(f"{_DIST_INFO}/RECORD,,")
    record.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_wheel(
    wheel_directory: str,
    config_settings: Mapping[str, object] | None = None,
    metadata_directory: str | None = None,
) -> str:
    del config_settings, metadata_directory
    with tempfile.TemporaryDirectory() as tmpdir:
        wheel_root = Path(tmpdir)
        shutil.copytree(
            PROJECT_ROOT / "kll_sketch",
            wheel_root / "kll_sketch",
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", "*.pyo", "tests", "requirements-test.txt", "LICENSE",
                "_build_backend.py", "_metadata.py",
            ),
        )
        dist_info = wheel_root / _DIST_INFO
        _write_metadata(dist_info)
        _write_record(dist_info, wheel_root)
        wheel_path = Path(wheel_directory) / f"{_NORMALIZED_NAME}-{_VERSION}-py3-none-any.whl"
        with ZipFile(wheel_path, "w", ZIP_DEFLATED) as zf:
            for file in _iter_files(wheel_root):
                zf.write(file, file.relative_to(wheel_root).as_posix())
        return wheel_path.name


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: Mapping[str, object] | None = None,
) -> str:
    del config_settings
    dist_info = Path(metadata_directory) / _DIST_INFO
    _write_metadata(dist_info)
    return dist_info.name


def build_sdist(sdist_directory: str, config_settings: Mapping[str, object] | None = None) -> str:
    del config_settings
    with tempfile.TemporaryDirectory() as tmpdir:
        sdist_root = Path(tmpdir) / f"{_NORMALIZED_NAME}-{_VERSION}"
        sdist_root.mkdir()
        for item in ["README.md", "LICENSE", "pyproject.toml"]:
            source = PROJECT_ROOT / item
            if source.exists():
                shutil.copy2(source, sdist_root / source.name)
        for directory in ["kll_sketch", "docs", "benchmarks", "tests"]:
            source = PROJECT_ROOT / directory
            if source.exists():
                shutil.copytree(source, sdist_root / directory, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
        archive = Path(sdist_directory) / f"{_NORMALIZED_NAME}-{_VERSION}.tar.gz"
        with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT) as tf:
            tf.add(sdist_root, arcname=sdist_root.name)
        return archive.name
