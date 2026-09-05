"""Dependency-free builder for the optional C++17 acceleration module."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig

PACKAGE_DIR = Path(__file__).resolve().parent
SOURCES = (PACKAGE_DIR / "_native.cpp", PACKAGE_DIR / "_native_state.cpp")


class NativeBuildError(RuntimeError):
    """Raised when the optional native extension cannot be compiled."""


def extension_filename() -> str:
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not suffix:
        suffix = ".pyd" if os.name == "nt" else ".so"
    return "_native" + str(suffix)


def _run(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise NativeBuildError(f"native compiler not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise NativeBuildError(f"native compilation failed with exit code {exc.returncode}") from exc


def _check_sources() -> None:
    missing = [str(source) for source in SOURCES if not source.exists()]
    if missing:
        raise NativeBuildError(f"missing native source: {', '.join(missing)}")


def _unix_command(target: Path) -> list[str]:
    raw_cxx = os.environ.get("CXX") or sysconfig.get_config_var("CXX") or "c++"
    compiler = shlex.split(str(raw_cxx)) or ["c++"]
    include = sysconfig.get_paths()["include"]
    command = compiler + [
        "-O3",
        "-DNDEBUG",
        "-std=c++17",
        "-fPIC",
        "-shared",
        "-DPyInit__native=PyInit__native_base",
        f"-I{include}",
        *(str(source) for source in SOURCES),
        "-o",
        str(target),
    ]
    if sys.platform == "darwin":
        command.extend(["-undefined", "dynamic_lookup"])
    extra = os.environ.get("KLL_SKETCH_NATIVE_CXXFLAGS")
    if extra:
        command[1:1] = shlex.split(extra)
    return command


def _windows_command(target: Path) -> list[str]:
    compiler = os.environ.get("CXX", "cl")
    include = sysconfig.get_paths()["include"]
    libs = Path(sys.base_prefix) / "libs"
    library = f"python{sys.version_info.major}{sys.version_info.minor}.lib"
    command = [
        compiler,
        "/nologo",
        "/O2",
        "/DNDEBUG",
        "/EHsc",
        "/std:c++17",
        "/LD",
        "/DPyInit__native=PyInit__native_base",
        f"/I{include}",
        *(str(source) for source in SOURCES),
        "/link",
        f"/LIBPATH:{libs}",
        library,
        f"/OUT:{target}",
    ]
    extra = os.environ.get("KLL_SKETCH_NATIVE_CXXFLAGS")
    if extra:
        command[1:1] = shlex.split(extra, posix=False)
    return command


def build_native(output_dir: str | os.PathLike[str] | None = None, *, force: bool = True) -> Path:
    """Compile ``_native`` into *output_dir* and return the extension path."""
    _check_sources()
    destination = Path(output_dir) if output_dir is not None else PACKAGE_DIR
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / extension_filename()
    if target.exists() and not force:
        return target
    if target.exists():
        target.unlink()
    command = _windows_command(target) if os.name == "nt" else _unix_command(target)
    _run(command)
    if not target.exists():
        raise NativeBuildError("compiler reported success but extension was not produced")
    return target


def clean_native(output_dir: str | os.PathLike[str] | None = None) -> None:
    destination = Path(output_dir) if output_dir is not None else PACKAGE_DIR
    target = destination / extension_filename()
    if target.exists():
        target.unlink()
    for pattern in ("_native.obj", "_native.exp", "_native.lib", "_native_state.obj"):
        candidate = destination / pattern
        if candidate.exists():
            candidate.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-force", action="store_true")
    args = parser.parse_args()
    if args.clean:
        clean_native(args.output_dir)
        return
    target = build_native(args.output_dir, force=not args.no_force)
    print(target)


if __name__ == "__main__":
    main()
