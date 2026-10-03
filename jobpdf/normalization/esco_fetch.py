"""Get the ESCO CSV export into data/esco/<version>/, verified and idempotent.

ESCO's portal only emails a download link, so the zip is downloaded once by
hand and published as a GitHub Release asset; this module automates the rest.
Three sources are supported: the release URL (default), a local zip, or an
already-unzipped local folder.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

ESCO_VERSION = "v1.2.1"
# Placeholders until the zip is published as a GitHub Release asset.
ESCO_ZIP_URL = ""
ESCO_ZIP_SHA256 = ""

DEFAULT_DATA_ROOT = Path("data/esco")
MARKER_NAME = ".source.json"
# The loader cannot work without these; fail at fetch time, not build time.
REQUIRED_FILES = ("skills_en.csv", "skills_uk.csv", "digitalSkillsCollection_en.csv")
DOWNLOAD_TIMEOUT_S = 120
_CHUNK = 1024 * 1024

SourceKind = Literal["url", "zip", "dir"]


class FetchError(Exception):
    """A problem the user must fix (bad checksum, missing file, no URL set)."""


@dataclass(frozen=True)
class FetchResult:
    dest: Path
    source: SourceKind
    source_sha256: str | None  # the zip's SHA-256; None for a folder source
    files: dict[str, str]  # CSV name -> SHA-256
    skipped: bool  # True when the destination was already up to date


def esco_dir(root: Path = DEFAULT_DATA_ROOT) -> Path:
    return root / ESCO_VERSION


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def use_existing(dest: Path) -> FetchResult | None:
    """Data already fetched (from any source) and intact, or None.

    Lets build_taxonomy.py run without arguments after a one-off
    ``fetch_esco.py --zip`` while the release URL isn't set yet.
    """
    marker = _read_marker(dest)
    if marker and _files_intact(dest, marker):
        return _result(dest, marker, skipped=True)
    return None


def fetch_from_url(
    dest: Path,
    url: str = ESCO_ZIP_URL,
    expected_sha256: str = ESCO_ZIP_SHA256,
    force: bool = False,
) -> FetchResult:
    """Download the release zip, verify it, and unzip it into ``dest``."""
    if not url or not expected_sha256:
        raise FetchError(
            "ESCO_ZIP_URL / ESCO_ZIP_SHA256 are not set yet in "
            "jobpdf/normalization/esco_fetch.py. Use --zip PATH or --dir PATH for now."
        )
    marker = _read_marker(dest)
    if not force and marker and marker.get("source_sha256") == expected_sha256:
        if _files_intact(dest, marker):
            return _result(dest, marker, skipped=True)

    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        zip_path = Path(tmp) / f"esco_{ESCO_VERSION}.zip"
        _download(url, zip_path)
        return fetch_from_zip(zip_path, dest, expected_sha256, force=True, source="url")


def fetch_from_zip(
    zip_path: Path,
    dest: Path,
    expected_sha256: str = ESCO_ZIP_SHA256,
    force: bool = False,
    source: SourceKind = "zip",
) -> FetchResult:
    """Verify a local zip (if a checksum is pinned) and unzip its CSVs into ``dest``."""
    if not zip_path.is_file():
        raise FetchError(f"Zip not found: {zip_path}")
    actual = sha256_file(zip_path)
    if expected_sha256 and actual != expected_sha256:
        raise FetchError(
            f"SHA-256 mismatch for {zip_path}:\n  expected {expected_sha256}\n  actual   {actual}"
        )
    marker = _read_marker(dest)
    if not force and marker and marker.get("source_sha256") == actual:
        if _files_intact(dest, marker):
            return _result(dest, marker, skipped=True)

    with zipfile.ZipFile(zip_path) as archive:
        members = _csv_members(archive)

        def write(staging: Path) -> None:
            for name, info in members.items():
                with archive.open(info) as src, (staging / name).open("wb") as out:
                    shutil.copyfileobj(src, out, _CHUNK)

        files = _replace_dest(dest, write)
    marker = _write_marker(dest, source, actual, files)
    return _result(dest, marker, skipped=False)


def fetch_from_dir(src_dir: Path, dest: Path, force: bool = False) -> FetchResult:
    """Copy the CSVs of an already-unzipped ESCO export into ``dest``."""
    if not src_dir.is_dir():
        raise FetchError(f"Folder not found: {src_dir}")
    sources = {p.name: p for p in sorted(src_dir.glob("*.csv"))}
    _check_required(sources)
    marker = _read_marker(dest)
    if not force and marker and _files_intact(dest, marker):
        if {name: sha256_file(p) for name, p in sources.items()} == marker["files"]:
            return _result(dest, marker, skipped=True)

    def write(staging: Path) -> None:
        for name, path in sources.items():
            shutil.copyfile(path, staging / name)

    files = _replace_dest(dest, write)
    marker = _write_marker(dest, "dir", None, files)
    return _result(dest, marker, skipped=False)


def _download(url: str, target: Path) -> None:
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response:
            with target.open("wb") as out:
                shutil.copyfileobj(response, out, _CHUNK)
    except OSError as exc:  # URLError is an OSError
        raise FetchError(f"Download failed from {url}: {exc}") from exc


def _csv_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """CSV members by flat file name; the export may sit in a sub-folder."""
    members: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        path = PurePosixPath(info.filename.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or ":" in info.filename:
            raise FetchError(f"Unsafe path in zip, refusing to extract: {info.filename}")
        if info.is_dir() or path.suffix.lower() != ".csv":
            continue
        if path.name in members:
            raise FetchError(f"Zip contains {path.name} twice")
        members[path.name] = info
    _check_required(members)
    return members


def _check_required(names: dict[str, object]) -> None:
    missing = [name for name in REQUIRED_FILES if name not in names]
    if missing:
        raise FetchError(f"ESCO export is missing required files: {', '.join(missing)}")


def _replace_dest(dest: Path, write: Callable[[Path], None]) -> dict[str, str]:
    """Write into a staging folder, then swap it in, so a failure never leaves half a dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=dest.parent, prefix=f".{dest.name}-"))
    try:
        write(staging)
        files = {p.name: sha256_file(p) for p in sorted(staging.glob("*.csv"))}
        if dest.exists():
            shutil.rmtree(dest)
        staging.rename(dest)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return files


def _read_marker(dest: Path) -> dict | None:
    try:
        return json.loads((dest / MARKER_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_marker(
    dest: Path, source: SourceKind, source_sha256: str | None, files: dict[str, str]
) -> dict:
    marker = {
        "esco_version": ESCO_VERSION,
        "source": source,
        "source_sha256": source_sha256,
        "files": files,
    }
    (dest / MARKER_NAME).write_text(json.dumps(marker, indent=2), encoding="utf-8")
    return marker


def _files_intact(dest: Path, marker: dict) -> bool:
    files: dict[str, str] = marker.get("files", {})
    return bool(files) and all(
        (dest / name).is_file() and sha256_file(dest / name) == sha
        for name, sha in files.items()
    )


def _result(dest: Path, marker: dict, skipped: bool) -> FetchResult:
    return FetchResult(
        dest=dest,
        source=marker["source"],
        source_sha256=marker.get("source_sha256"),
        files=dict(marker["files"]),
        skipped=skipped,
    )
