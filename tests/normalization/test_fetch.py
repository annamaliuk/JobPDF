import zipfile
from pathlib import Path

import pytest

from jobpdf.normalization.esco_fetch import (
    MARKER_NAME,
    REQUIRED_FILES,
    FetchError,
    fetch_from_dir,
    fetch_from_url,
    fetch_from_zip,
    sha256_file,
)

FAKE_CSV = "conceptUri,preferredLabel\nhttp://example.org/skill/1,fake skill\n"


def make_zip(path: Path, members: dict[str, str] | None = None) -> Path:
    if members is None:
        members = {f"ESCO dataset/{name}": FAKE_CSV for name in REQUIRED_FILES}
        members["ESCO dataset/readme.txt"] = "not a csv"
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


def test_zip_is_flattened_and_marker_written(tmp_path: Path) -> None:
    zip_path = make_zip(tmp_path / "esco.zip")
    dest = tmp_path / "esco" / "v1.2.1"

    result = fetch_from_zip(zip_path, dest, expected_sha256="")

    assert not result.skipped
    assert result.source_sha256 == sha256_file(zip_path)
    assert sorted(p.name for p in dest.glob("*.csv")) == sorted(REQUIRED_FILES)
    assert not (dest / "readme.txt").exists()
    assert (dest / MARKER_NAME).is_file()


def test_second_run_skips_unless_forced(tmp_path: Path) -> None:
    zip_path = make_zip(tmp_path / "esco.zip")
    dest = tmp_path / "out"
    fetch_from_zip(zip_path, dest, expected_sha256="")

    assert fetch_from_zip(zip_path, dest, expected_sha256="").skipped
    assert not fetch_from_zip(zip_path, dest, expected_sha256="", force=True).skipped


def test_tampered_file_triggers_refetch(tmp_path: Path) -> None:
    zip_path = make_zip(tmp_path / "esco.zip")
    dest = tmp_path / "out"
    fetch_from_zip(zip_path, dest, expected_sha256="")
    (dest / "skills_en.csv").write_text("tampered")

    result = fetch_from_zip(zip_path, dest, expected_sha256="")

    assert not result.skipped
    assert (dest / "skills_en.csv").read_text() == FAKE_CSV


def test_checksum_mismatch_aborts_without_touching_dest(tmp_path: Path) -> None:
    zip_path = make_zip(tmp_path / "esco.zip")
    dest = tmp_path / "out"

    with pytest.raises(FetchError, match="SHA-256 mismatch"):
        fetch_from_zip(zip_path, dest, expected_sha256="0" * 64)
    assert not dest.exists()


def test_unsafe_zip_path_is_rejected(tmp_path: Path) -> None:
    members = {name: FAKE_CSV for name in REQUIRED_FILES}
    members["../evil.csv"] = "x"
    zip_path = make_zip(tmp_path / "evil.zip", members)

    with pytest.raises(FetchError, match="Unsafe path"):
        fetch_from_zip(zip_path, tmp_path / "out", expected_sha256="")
    assert not (tmp_path / "evil.csv").exists()


def test_missing_required_file(tmp_path: Path) -> None:
    zip_path = make_zip(tmp_path / "esco.zip", {"skills_en.csv": FAKE_CSV})

    with pytest.raises(FetchError, match="skills_uk.csv"):
        fetch_from_zip(zip_path, tmp_path / "out", expected_sha256="")


def test_folder_source(tmp_path: Path) -> None:
    src = tmp_path / "export"
    src.mkdir()
    for name in REQUIRED_FILES:
        (src / name).write_text(FAKE_CSV)
    dest = tmp_path / "out"

    first = fetch_from_dir(src, dest)
    assert not first.skipped and first.source_sha256 is None
    assert fetch_from_dir(src, dest).skipped

    (src / "skills_en.csv").write_text(FAKE_CSV + "http://example.org/skill/2,new\n")
    assert not fetch_from_dir(src, dest).skipped  # source changed -> copied again


def test_url_mode_needs_url_and_checksum(tmp_path: Path) -> None:
    with pytest.raises(FetchError, match="--zip"):
        fetch_from_url(tmp_path / "out", url="", expected_sha256="")
