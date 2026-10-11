"""Load ESCO skills into concepts and aliases (JM-16).

Facts about the v1.2.1 CSV export this relies on (checked on the real files):
- skills_<lang>.csv are mono-lingual; languages join on ``conceptUri``.
- altLabels / hiddenLabels hold several labels separated by line breaks inside
  one quoted cell, so files must be read with the csv module.
- skills_uk.csv only translates preferredLabel and description; its altLabels
  and hiddenLabels are empty.
- A few URIs appear twice, differing only in modifiedDate.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from jobpdf.normalization.text import normalize

AliasKind = Literal["preferred", "alt", "hidden", "stripped", "custom"]
ConceptSource = Literal["esco", "custom"]

# Aliases this short ("r", "c", "go") are too ambiguous to embed; JM-17 skips them
# and JM-18 only uses them for exact lookups.
EXACT_ONLY_MAX_LEN = 2
SKILL_CONCEPT_TYPE = "KnowledgeSkillCompetence"
RELEASED = "released"
BASE_LANGUAGE = "en"  # concepts are defined by the English file
DIGITAL_COLLECTION = "digitalSkillsCollection_en.csv"

# When the same (concept, alias_norm) comes from several labels, keep the strongest.
_KIND_RANK: dict[str, int] = {"preferred": 0, "alt": 1, "hidden": 2, "stripped": 3, "custom": 4}
_TRAILING_PARENTHETICAL = re.compile(r"^(.*?\S)\s*\([^()]*\)$")


@dataclass(frozen=True)
class Concept:
    canonical_id: str
    name_en: str
    name_uk: str | None
    description: str | None
    skill_type: str | None
    reuse_level: str | None
    is_digital: bool
    source: ConceptSource


@dataclass(frozen=True)
class Alias:
    canonical_id: str
    alias: str  # as written in the source
    alias_norm: str  # normalize(alias): the lookup key
    lang: str
    alias_kind: AliasKind
    exact_only: bool


@dataclass
class LoadStats:
    rows_read: dict[str, int] = field(default_factory=dict)  # per language
    duplicate_rows: int = 0  # same URI twice; the latest modifiedDate wins
    dropped_rows: int = 0  # not a released individual skill
    unknown_uris: dict[str, int] = field(default_factory=dict)  # non-base rows without concept


def make_alias(canonical_id: str, alias: str, lang: str, kind: AliasKind) -> Alias:
    norm = normalize(alias)
    return Alias(
        canonical_id=canonical_id,
        alias=alias.strip(),
        alias_norm=norm,
        lang=lang,
        alias_kind=kind,
        exact_only=len(norm) <= EXACT_ONLY_MAX_LEN,
    )


def load_esco(
    esco_dir: Path, languages: tuple[str, ...] = ("en", "uk")
) -> tuple[list[Concept], list[Alias]]:
    """Released individual skills as concepts, every label as an alias, deduplicated."""
    concepts, aliases, _ = load_esco_with_stats(esco_dir, languages)
    return concepts, aliases


def load_esco_with_stats(
    esco_dir: Path, languages: tuple[str, ...] = ("en", "uk")
) -> tuple[list[Concept], list[Alias], LoadStats]:
    """Like load_esco, plus the counts the build manifest reports."""
    if BASE_LANGUAGE not in languages:
        raise ValueError(f"languages must include {BASE_LANGUAGE!r}: concepts come from it")
    stats = LoadStats()
    digital = {row["conceptUri"] for row in read_csv(esco_dir / DIGITAL_COLLECTION)}

    rows_by_lang = {lang: _skill_rows(esco_dir, lang, stats) for lang in languages}
    base = rows_by_lang[BASE_LANGUAGE]
    uk = rows_by_lang.get("uk", {})

    concepts = [
        Concept(
            canonical_id=uri,
            name_en=row["preferredLabel"].strip(),
            name_uk=uk[uri]["preferredLabel"].strip() if uri in uk else None,
            description=row["description"].strip() or None,
            skill_type=row["skillType"] or None,
            reuse_level=row["reuseLevel"] or None,
            is_digital=uri in digital,
            source="esco",
        )
        for uri, row in sorted(base.items())
    ]

    aliases: list[Alias] = []
    for lang, rows in rows_by_lang.items():
        unknown = 0
        for uri, row in rows.items():
            if uri not in base:
                unknown += 1  # translated row for a concept the base file doesn't release
                continue
            aliases.extend(_row_aliases(uri, row, lang))
        if lang != BASE_LANGUAGE:
            stats.unknown_uris[lang] = unknown
    return concepts, dedupe_aliases(aliases), stats


def read_csv(path: Path) -> list[dict[str, str]]:
    # Descriptions can be long; the csv default field limit is 128 KB.
    csv.field_size_limit(16 * 1024 * 1024)
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _skill_rows(esco_dir: Path, lang: str, stats: LoadStats) -> dict[str, dict[str, str]]:
    """Released KnowledgeSkillCompetence rows by URI, latest modifiedDate per URI."""
    rows = read_csv(esco_dir / f"skills_{lang}.csv")
    stats.rows_read[lang] = len(rows)
    by_uri: dict[str, dict[str, str]] = {}
    for row in rows:
        if row["conceptType"] != SKILL_CONCEPT_TYPE or row["status"] != RELEASED:
            stats.dropped_rows += 1
            continue
        uri = row["conceptUri"]
        previous = by_uri.get(uri)
        if previous is not None:
            stats.duplicate_rows += 1
            # ISO timestamps compare correctly as strings.
            if previous["modifiedDate"] >= row["modifiedDate"]:
                continue
        by_uri[uri] = row
    return by_uri


def _row_aliases(uri: str, row: dict[str, str], lang: str) -> list[Alias]:
    labels: list[tuple[str, AliasKind]] = [(row["preferredLabel"], "preferred")]
    labels += [(label, "alt") for label in split_labels(row["altLabels"])]
    labels += [(label, "hidden") for label in split_labels(row["hiddenLabels"])]
    return [make_alias(uri, label, lang, kind) for label, kind in labels if label.strip()]


def split_labels(cell: str) -> list[str]:
    """Labels in one cell are separated by line breaks."""
    return [label.strip() for label in cell.splitlines() if label.strip()]


def dedupe_aliases(aliases: list[Alias]) -> list[Alias]:
    """One alias per (concept, alias_norm): the strongest kind, then the base language."""
    best: dict[tuple[str, str], Alias] = {}
    for alias in aliases:
        key = (alias.canonical_id, alias.alias_norm)
        current = best.get(key)
        if current is None or _rank(alias) < _rank(current):
            best[key] = alias
    return sorted(best.values(), key=lambda a: (a.canonical_id, a.alias_norm))


def _rank(alias: Alias) -> tuple[int, int]:
    return (_KIND_RANK[alias.alias_kind], 0 if alias.lang == BASE_LANGUAGE else 1)


@dataclass(frozen=True)
class StripCollision:
    alias_norm: str  # the stripped form that was not added
    canonical_ids: tuple[str, ...]  # every concept that claims it


def strip_parentheticals(aliases: list[Alias]) -> tuple[list[Alias], list[StripCollision]]:
    """Add "Python" for "Python (computer programming)", unless it would be ambiguous.

    A stripped alias is skipped when its alias_norm already belongs to a
    different concept, or when two concepts strip to the same text ("Go
    (programming)" vs "Go (game)"). Candidates are all computed first, so the
    result doesn't depend on row order.
    """
    owners: dict[str, set[str]] = defaultdict(set)
    for alias in aliases:
        owners[alias.alias_norm].add(alias.canonical_id)

    candidates: dict[str, dict[str, Alias]] = defaultdict(dict)  # norm -> concept -> alias
    for alias in aliases:
        if alias.alias_kind in ("stripped", "custom"):
            continue
        match = _TRAILING_PARENTHETICAL.match(alias.alias)
        if match is None:
            continue
        stripped = make_alias(alias.canonical_id, match.group(1), alias.lang, "stripped")
        if alias.canonical_id in owners.get(stripped.alias_norm, set()):
            continue  # the concept already has this alias ("Python" is an ESCO alt label)
        candidates[stripped.alias_norm].setdefault(alias.canonical_id, stripped)

    added: list[Alias] = []
    collisions: list[StripCollision] = []
    for norm, by_concept in sorted(candidates.items()):
        claimants = set(by_concept) | owners.get(norm, set())
        if len(claimants) > 1:
            collisions.append(StripCollision(norm, tuple(sorted(claimants))))
            continue
        added.extend(by_concept.values())
    return dedupe_aliases([*aliases, *added]), collisions
