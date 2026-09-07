from email.parser import Parser
import tarfile

from kll_sketch import __version__
from kll_sketch._build_backend import build_sdist


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
