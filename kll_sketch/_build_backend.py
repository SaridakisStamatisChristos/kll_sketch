"""Small standards-oriented PEP 517 backend with no third-party dependencies."""
from __future__ import annotations

import base64
from hashlib import sha256
import os
from pathlib import Path
import shutil
import sys
import sysconfig
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


def _truthy(value: object) -> bool:
    if isinstance(value, (list, tuple)):
        return any(_truthy(item) for item in value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _native_requested(config_settings: Mapping[str, object] | None) -> bool:
    if _truthy(os.environ.get("KLL_SKETCH_BUILD_NATIVE", "")):
        return True
    if not config_settings:
        return False
    return any(
        key in config_settings and _truthy(config_settings[key])
        for key in ("native", "--native", "kll-native")
    )


def _wheel_tag(native: bool) -> str:
    if not native:
        return "py3-none-any"
    interpreter = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    return f"{interpreter}-{interpreter}-{platform}"


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


def _write_metadata(dist_info: Path, *, tag: str = "py3-none-any", pure: bool = True) -> None:
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(_build_metadata_text(), encoding="utf-8")
    (dist_info / "WHEEL").write_text(
        "Wheel-Version: 1.0\n"
        "Generator: kll-sketch self-hosted backend\n"
        f"Root-Is-Purelib: {'true' if pure else 'false'}\n"
        f"Tag: {tag}\n",
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
    del metadata_directory
    native = _native_requested(config_settings)
    tag = _wheel_tag(native)
    with tempfile.TemporaryDirectory() as tmpdir:
        wheel_root = Path(tmpdir)
        package_root = wheel_root / "kll_sketch"
        shutil.copytree(
            PROJECT_ROOT / "kll_sketch",
            package_root,
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", "*.pyo", "tests", "requirements-test.txt", "LICENSE",
                "_build_backend.py", "_metadata.py", "_native_build.py",
                "_native*.cpp", "_native*.inc", "_native*.so", "_native*.pyd",
            ),
        )
        if native:
            from ._native_build import build_native

            build_native(package_root)
        dist_info = wheel_root / _DIST_INFO
        _write_metadata(dist_info, tag=tag, pure=not native)
        _write_record(dist_info, wheel_root)
        wheel_path = Path(wheel_directory) / f"{_NORMALIZED_NAME}-{_VERSION}-{tag}.whl"
        wheel_path.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(wheel_path, "w", ZIP_DEFLATED) as zf:
            for file in _iter_files(wheel_root):
                zf.write(file, file.relative_to(wheel_root).as_posix())
        return wheel_path.name


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: Mapping[str, object] | None = None,
) -> str:
    native = _native_requested(config_settings)
    dist_info = Path(metadata_directory) / _DIST_INFO
    _write_metadata(dist_info, tag=_wheel_tag(native), pure=not native)
    return dist_info.name


def build_sdist(sdist_directory: str, config_settings: Mapping[str, object] | None = None) -> str:
    del config_settings
    with tempfile.TemporaryDirectory() as tmpdir:
        sdist_root = Path(tmpdir) / f"{_NORMALIZED_NAME}-{_VERSION}"
        sdist_root.mkdir()
        for item in [
            "README.md",
            "LICENSE",
            "pyproject.toml",
            "CITATION.cff",
            "CONTRIBUTING.md",
            "SECURITY.md",
        ]:
            source = PROJECT_ROOT / item
            if source.exists():
                shutil.copy2(source, sdist_root / source.name)
        for directory in ["kll_sketch", "docs", "benchmarks", "tests"]:
            source = PROJECT_ROOT / directory
            if source.exists():
                shutil.copytree(
                    source,
                    sdist_root / directory,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "_native*.so", "_native*.pyd"),
                )
        archive = Path(sdist_directory) / f"{_NORMALIZED_NAME}-{_VERSION}.tar.gz"
        Path(sdist_directory).mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT) as tf:
            tf.add(sdist_root, arcname=sdist_root.name)
        return archive.name
