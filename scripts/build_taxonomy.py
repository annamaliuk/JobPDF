"""Build data/taxonomy/{concepts.csv, aliases.csv, manifest.json} (JM-16).

Usage (from the repo root):
    uv run python scripts/build_taxonomy.py              # ESCO from data/esco/ or the release URL
    uv run python scripts/build_taxonomy.py --zip PATH   # or --dir PATH, as in fetch_esco.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fetch_esco import add_source_args
from fetch_esco import run as fetch

from jobpdf.normalization.build import DEFAULT_CUSTOM, DEFAULT_OUT_DIR, build_taxonomy
from jobpdf.normalization.custom import CustomSkillsError
from jobpdf.normalization.esco_fetch import FetchError

SHOW_AMBIGUOUS = 10


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the skills taxonomy files.")
    add_source_args(parser)
    parser.add_argument("--custom", type=Path, default=DEFAULT_CUSTOM)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    try:
        fetched = fetch(args.zip, args.dir, args.data_root, args.force)
        result = build_taxonomy(fetched.dest, args.custom, args.out_dir)
    except (FetchError, CustomSkillsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    counts = result.manifest["counts"]
    ambiguous = result.manifest["ambiguous_aliases"]
    collisions = result.manifest["stripped_collisions_skipped"]
    print(f"\nWrote {args.out_dir}/concepts.csv, aliases.csv, manifest.json")
    print(f"concepts: {counts['concepts']}")
    print(f"aliases:  {counts['aliases_total']} by lang {counts['aliases_by_lang']}")
    print(f"          by kind {counts['aliases_by_kind']}, exact_only {counts['exact_only']}")
    print(f"stripped-alias collisions skipped: {collisions['count']}")
    print(f"ambiguous aliases (one name, several concepts): {ambiguous['count']}")
    for example in ambiguous["examples"][:SHOW_AMBIGUOUS]:
        print(f"  {example['alias_norm']!r}: {len(example['canonical_ids'])} concepts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
