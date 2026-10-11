"""Build the taxonomy files: concepts.csv, aliases.csv, manifest.json (JM-16)."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

from jobpdf.normalization.custom import load_custom, merge_custom
from jobpdf.normalization.esco_fetch import ESCO_VERSION, MARKER_NAME, sha256_file
from jobpdf.normalization.taxonomy import (
    DIGITAL_COLLECTION,
    Alias,
    Concept,
    LoadStats,
    StripCollision,
    load_esco_with_stats,
    strip_parentheticals,
)

DEFAULT_OUT_DIR = Path("data/taxonomy")
DEFAULT_CUSTOM = DEFAULT_OUT_DIR / "custom_skills.csv"
LANGUAGES = ("en", "uk")
# The manifest lists this many examples; the counts always cover everything.
MAX_EXAMPLES = 50


@dataclass(frozen=True)
class BuildResult:
    concepts: list[Concept]
    aliases: list[Alias]
    manifest: dict


def build_taxonomy(
    esco_dir: Path, custom_path: Path = DEFAULT_CUSTOM, out_dir: Path = DEFAULT_OUT_DIR
) -> BuildResult:
    """load ESCO -> strip parentheticals -> merge custom -> write files."""
    concepts, aliases, stats = load_esco_with_stats(esco_dir, LANGUAGES)
    aliases, collisions = strip_parentheticals(aliases)
    custom_concepts, custom_aliases = load_custom(
        custom_path, {c.canonical_id: c for c in concepts}
    )
    concepts, aliases = merge_custom(concepts, aliases, custom_concepts, custom_aliases)

    inputs = [esco_dir / f"skills_{lang}.csv" for lang in LANGUAGES]
    inputs += [esco_dir / DIGITAL_COLLECTION, custom_path]
    manifest = build_manifest(concepts, aliases, stats, collisions, inputs, esco_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / "concepts.csv", concepts)
    write_rows(out_dir / "aliases.csv", aliases)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return BuildResult(concepts, aliases, manifest)


def find_ambiguous(aliases: list[Alias]) -> dict[str, list[str]]:
    """alias_norm -> canonical_ids, for every alias_norm that names several concepts.

    These are not resolved here: JM-18 routes them to the LLM.
    """
    owners: dict[str, set[str]] = defaultdict(set)
    for alias in aliases:
        owners[alias.alias_norm].add(alias.canonical_id)
    return {norm: sorted(ids) for norm, ids in sorted(owners.items()) if len(ids) > 1}


def build_manifest(
    concepts: list[Concept],
    aliases: list[Alias],
    stats: LoadStats,
    collisions: list[StripCollision],
    inputs: list[Path],
    esco_dir: Path,
) -> dict:
    ambiguous = find_ambiguous(aliases)
    # Most-contested aliases first: they matter most for JM-18.
    examples = sorted(ambiguous.items(), key=lambda item: (-len(item[1]), item[0]))
    return {
        "esco_version": ESCO_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "esco_zip_sha256": _esco_zip_sha256(esco_dir),
            "files": {path.name: sha256_file(path) for path in inputs},
        },
        "counts": {
            "concepts": dict(sorted(Counter(c.source for c in concepts).items())),
            "aliases_total": len(aliases),
            "aliases_by_lang": dict(sorted(Counter(a.lang for a in aliases).items())),
            "aliases_by_kind": dict(sorted(Counter(a.alias_kind for a in aliases).items())),
            "exact_only": sum(a.exact_only for a in aliases),
        },
        "esco_rows": {
            "read": stats.rows_read,
            "duplicate_rows": stats.duplicate_rows,
            "dropped_rows": stats.dropped_rows,
            "unknown_uris": stats.unknown_uris,
        },
        "stripped_collisions_skipped": {
            "count": len(collisions),
            "examples": [
                {"alias_norm": c.alias_norm, "canonical_ids": list(c.canonical_ids)}
                for c in collisions[:MAX_EXAMPLES]
            ],
        },
        "ambiguous_aliases": {
            "count": len(ambiguous),
            "examples": [
                {"alias_norm": norm, "canonical_ids": ids} for norm, ids in examples[:MAX_EXAMPLES]
            ],
        },
    }


def write_rows(path: Path, rows: list[Concept] | list[Alias]) -> None:
    """CSV with one column per dataclass field; booleans as true/false, None as empty."""
    if not rows:
        raise ValueError(f"refusing to write empty {path.name}")
    columns = [f.name for f in fields(rows[0])]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(value) for key, value in asdict(row).items()})


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _esco_zip_sha256(esco_dir: Path) -> str | None:
    try:
        marker = json.loads((esco_dir / MARKER_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return marker.get("source_sha256")
