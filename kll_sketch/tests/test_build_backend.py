from email.parser import Parser
import tarfile
from zipfile import ZipFile

from kll_sketch import __version__
from kll_sketch._build_backend import PROJECT_ROOT, build_sdist, build_wheel


def test_wheel_places_declared_license_in_licenses_directory(tmp_path) -> None:
    filename = build_wheel(str(tmp_path))

    assert filename == f"kll_sketch-{__version__}-py3-none-any.whl"
    dist_info = f"kll_sketch-{__version__}.dist-info"
    license_path = f"{dist_info}/licenses/LICENSE"
    with ZipFile(tmp_path / filename) as archive:
        names = archive.namelist()
        metadata = Parser().parsestr(archive.read(f"{dist_info}/METADATA").decode("utf-8"))
        record = archive.read(f"{dist_info}/RECORD").decode("utf-8").splitlines()

        assert license_path in names
        assert f"{dist_info}/LICENSE" not in names
        assert archive.read(license_path) == (PROJECT_ROOT / "LICENSE").read_bytes()
        assert metadata.get_all("License-File") == ["LICENSE"]
        assert any(line.startswith(f"{license_path},sha256=") for line in record)


def test_sdist_contains_valid_core_metadata(tmp_path) -> None:
    filename = build_sdist(str(tmp_path))

    assert filename == f"kll_sketch-{__version__}.tar.gz"
    with tarfile.open(tmp_path / filename, "r:gz") as archive:
        pkg_info = archive.extractfile(f"kll_sketch-{__version__}/PKG-INFO")
        assert pkg_info is not None
        metadata = Parser().parsestr(pkg_info.read().decode("utf-8"))

    assert metadata["Metadata-Version"] == "2.4"
    assert metadata["Name"] == "kll-sketch"
    assert metadata["Version"] == __version__
    assert metadata["License-Expression"] == "Apache-2.0"
