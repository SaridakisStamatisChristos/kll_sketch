"""Self-contained PEP 517 backend with no external dependencies."""
from __future__ import annotations

import base64
import shutil
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Mapping
from zipfile import ZipFile, ZIP_DEFLATED
import tarfile

from ._metadata import PROJECT_METADATA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_NAME = PROJECT_METADATA["name"]  # type: ignore[index]
_VERSION = PROJECT_METADATA["version"]  # type: ignore[index]
_NORMALIZED_NAME = PACKAGE_NAME.replace("-", "_")
_DIST_INFO = f"{_NORMALIZED_NAME}-{_VERSION}.dist-info"


def _build_metadata_text() -> str:
    readme_info = PROJECT_METADATA["readme"]  # type: ignore[index]
    readme_path = Path(readme_info["path"])  # type: ignore[index]
    readme_content_type = readme_info["content_type"]  # type: ignore[index]
    summary = PROJECT_METADATA.get("summary", "")
    keywords = PROJECT_METADATA.get("keywords", [])
    classifiers = PROJECT_METADATA.get("classifiers", [])
    urls = PROJECT_METADATA.get("urls", {})
    authors = PROJECT_METADATA.get("authors", [])
    license_info = PROJECT_METADATA.get("license", {})
    requires_python = PROJECT_METADATA.get("requires_python")
    optional = PROJECT_METADATA.get("optional-dependencies", {})

    lines: list[str] = [
        "Metadata-Version: 2.1",
        f"Name: {PACKAGE_NAME}",
        f"Version: {_VERSION}",
    ]
    if summary:
        lines.append(f"Summary: {summary}")
    if authors:
        primary = authors[0]
        name = getattr(primary, "name", None) or primary.get("name")  # type: ignore[union-attr]
        email = getattr(primary, "email", None) or primary.get("email")  # type: ignore[union-attr]
        if name:
            lines.append(f"Author: {name}")
        if email:
            lines.append(f"Author-email: {name} <{email}>")
    if requires_python:
        lines.append(f"Requires-Python: {requires_python}")
    if keywords:
        lines.append(f"Keywords: {', '.join(keywords)}")
    for classifier in classifiers:
        lines.append(f"Classifier: {classifier}")
    for label, url in urls.items():
        lines.append(f"Project-URL: {label}, {url}")

    license_text = license_info.get("text") if isinstance(license_info, dict) else None
    license_files = []
    if isinstance(license_info, dict):
        license_files = license_info.get("files", [])
    if license_text:
        lines.append(f"License-Expression: {license_text}")
    for lf in license_files:
        lines.append(f"License-File: {lf}")

    for extra, requirements in optional.items():
        lines.append(f"Provides-Extra: {extra}")
        for requirement in requirements:
            lines.append(f"Requires-Dist: {requirement}; extra == '{extra}'")

    lines.append(f"Description-Content-Type: {readme_content_type}")
    lines.append("")
    description = readme_path.read_text(encoding="utf-8")
    return "\n".join(lines) + "\n" + description


def _write_metadata(dist_info: Path) -> None:
    dist_info.mkdir(parents=True, exist_ok=True)
    metadata_path = dist_info / "METADATA"
    metadata_path.write_text(_build_metadata_text(), encoding="utf-8")
    wheel_path = dist_info / "WHEEL"
    wheel_path.write_text(
        "\n".join(
            [
                "Wheel-Version: 1.0",
                "Generator: kll-sketch self-hosted backend",
                "Root-Is-Purelib: true",
                "Tag: py3-none-any",
                "",
            ]
        ),
        encoding="utf-8",
    )
    license_files = PROJECT_METADATA.get("license", {})
    if isinstance(license_files, dict):
        for rel_path in license_files.get("files", []):
            source = PROJECT_ROOT / rel_path
            target = dist_info / Path(rel_path).name
            target.write_bytes(source.read_bytes())


def _iter_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            yield path


def _record_for(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    with path.open("rb") as fh:
        digest = base64.urlsafe_b64encode(sha256(fh.read()).digest()).decode().rstrip("=")
    size = path.stat().st_size
    return f"{relative},sha256={digest},{size}"


def _write_record(dist_info: Path, wheel_root: Path) -> None:
    record_lines = [
        _record_for(path, wheel_root)
        for path in _iter_files(wheel_root)
        if path != dist_info / "RECORD"
    ]
    record_lines.append(f"{_DIST_INFO}/RECORD,,")
    (dist_info / "RECORD").write_text("\n".join(record_lines) + "\n", encoding="utf-8")


def build_wheel(wheel_directory: str, config_settings: Mapping[str, object] | None = None, metadata_directory: str | None = None) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        wheel_root = Path(tmpdir)
        package_target = wheel_root / "kll_sketch"
        shutil.copytree(
            PROJECT_ROOT / "kll_sketch",
            package_target,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "*.swp"),
        )
        dist_info = wheel_root / _DIST_INFO
        _write_metadata(dist_info)
        _write_record(dist_info, wheel_root)
        wheel_path = Path(wheel_directory) / f"{_NORMALIZED_NAME}-{_VERSION}-py3-none-any.whl"
        with ZipFile(wheel_path, "w", ZIP_DEFLATED) as zf:
            for file in _iter_files(wheel_root):
                zf.write(file, file.relative_to(wheel_root).as_posix())
        return wheel_path.name


def prepare_metadata_for_build_wheel(metadata_directory: str, config_settings: Mapping[str, object] | None = None) -> str:
    dist_info = Path(metadata_directory) / _DIST_INFO
    _write_metadata(dist_info)
    # RECORD will be regenerated during build_wheel
    return dist_info.name


def build_sdist(sdist_directory: str, config_settings: Mapping[str, object] | None = None) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        sdist_root = Path(tmpdir) / f"{PACKAGE_NAME}-{_VERSION}"
        sdist_root.mkdir()
        for item in ["README.md", "LICENSE", "pyproject.toml"]:
            source = PROJECT_ROOT / item
            if source.exists():
                shutil.copy2(source, sdist_root / source.name)
        for directory in ["kll_sketch", "docs", "benchmarks", "tests"]:
            source_dir = PROJECT_ROOT / directory
            if not source_dir.exists():
                continue
            target_dir = sdist_root / directory
            shutil.copytree(
                source_dir,
                target_dir,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "*.swp"),
            )
        archive_path = Path(sdist_directory) / f"{PACKAGE_NAME}-{_VERSION}.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tf:
            tf.add(sdist_root, arcname=sdist_root.name)
        return archive_path.name
