"""Fetch the ESCO export into data/esco/<version>/ (JM-16).

Usage (from the repo root):
    uv run python scripts/fetch_esco.py                 # release URL (once it is set)
    uv run python scripts/fetch_esco.py --zip PATH      # local zip; prints its SHA-256
    uv run python scripts/fetch_esco.py --dir PATH      # already-unzipped export folder
Add --force to re-extract even when data/esco/<version>/ is up to date.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jobpdf.normalization.esco_fetch import (
    DEFAULT_DATA_ROOT,
    FetchError,
    FetchResult,
    esco_dir,
    fetch_from_dir,
    fetch_from_url,
    fetch_from_zip,
)


def run(zip_path: Path | None, src_dir: Path | None, data_root: Path, force: bool) -> FetchResult:
    """Shared by build_taxonomy.py so both scripts fetch the same way."""
    dest = esco_dir(data_root)
    if zip_path is not None:
        result = fetch_from_zip(zip_path, dest, force=force)
    elif src_dir is not None:
        result = fetch_from_dir(src_dir, dest, force=force)
    else:
        result = fetch_from_url(dest, force=force)

    state = "already up to date, skipped" if result.skipped else "written"
    print(f"ESCO {dest.name}: {len(result.files)} CSV files in {result.dest} ({state})")
    if result.source_sha256:
        print(f"zip SHA-256: {result.source_sha256}")
        print("  -> paste into ESCO_ZIP_SHA256 in jobpdf/normalization/esco_fetch.py")
    return result


def add_source_args(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--zip", type=Path, help="local ESCO zip instead of the release URL")
    source.add_argument("--dir", type=Path, help="local unzipped ESCO export folder")
    parser.add_argument("--force", action="store_true", help="re-extract even if up to date")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch the ESCO CSV export.")
    add_source_args(parser)
    args = parser.parse_args()
    try:
        run(args.zip, args.dir, args.data_root, args.force)
    except FetchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
