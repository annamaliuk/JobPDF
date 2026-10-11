"""Hand-curated taxonomy extension: data/taxonomy/custom_skills.csv (JM-16).

Each row adds one alias. ``canonical_id`` is either an ESCO concept URI (the
row adds an alias to it) or ``jobpdf:skill/<slug>`` (a concept ESCO lacks; its
first row carries ``canonical_name``). Validation collects every problem with
its line number, so one run shows everything that needs fixing.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from jobpdf.normalization.taxonomy import Alias, Concept, dedupe_aliases, make_alias

COLUMNS = ("canonical_id", "canonical_name", "alias", "lang")
LANGUAGES = ("en", "uk")
CUSTOM_PREFIX = "jobpdf:skill/"
_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_URI = re.compile(r"^https?://\S+$")


class CustomSkillsError(ValueError):
    """custom_skills.csv is invalid; ``errors`` lists every problem found."""

    def __init__(self, path: Path, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"{path}: {len(errors)} problem(s)\n  " + "\n  ".join(errors))


def load_custom(
    path: Path, esco_concepts: dict[str, Concept]
) -> tuple[list[Concept], list[Alias]]:
    """Validate the custom file and return new concepts plus all custom aliases."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise CustomSkillsError(path, [f"missing columns: {', '.join(missing)}"])
        rows = [(reader.line_num, row) for row in reader]

    errors: list[str] = []
    names: dict[str, str] = {}  # jobpdf concept -> canonical_name
    aliases: list[Alias] = []
    for line, row in rows:
        cid, name, alias, lang = (row[c].strip() for c in COLUMNS)
        problem = _check_row(cid, name, alias, lang, names, esco_concepts)
        if problem:
            errors.append(f"line {line}: {problem}")
            continue
        if cid.startswith(CUSTOM_PREFIX) and cid not in names:
            names[cid] = name
            aliases.append(make_alias(cid, name, lang, "custom"))
        if alias:
            aliases.append(make_alias(cid, alias, lang, "custom"))

    errors += _conflicting_aliases(aliases)
    if errors:
        raise CustomSkillsError(path, errors)

    concepts = [
        Concept(
            canonical_id=cid,
            name_en=name,
            name_uk=None,
            description=None,
            skill_type=None,
            reuse_level=None,
            is_digital=True,  # the extension only covers tech tools
            source="custom",
        )
        for cid, name in sorted(names.items())
    ]
    return concepts, dedupe_aliases(aliases)


def merge_custom(
    concepts: list[Concept],
    aliases: list[Alias],
    custom_concepts: list[Concept],
    custom_aliases: list[Alias],
) -> tuple[list[Concept], list[Alias]]:
    """Add custom concepts and aliases. Clashes with ESCO aliases are kept, not
    resolved: the manifest reports them as ambiguous and JM-18 asks the LLM."""
    return [*concepts, *custom_concepts], dedupe_aliases([*aliases, *custom_aliases])


def _check_row(
    cid: str,
    name: str,
    alias: str,
    lang: str,
    names: dict[str, str],
    esco_concepts: dict[str, Concept],
) -> str | None:
    if lang not in LANGUAGES:
        return f"lang must be one of {', '.join(LANGUAGES)}, got {lang!r}"
    if cid.startswith(CUSTOM_PREFIX):
        slug = cid[len(CUSTOM_PREFIX) :]
        if not _SLUG.match(slug):
            return f"malformed slug {slug!r} (use lowercase letters, digits and single hyphens)"
        if cid not in names and not name:
            return f"first row of {cid} must have canonical_name"
        if cid in names and name and name != names[cid]:
            return f"canonical_name {name!r} conflicts with {names[cid]!r} for {cid}"
        return None
    if not _URI.match(cid):
        return f"canonical_id must be an ESCO URI or {CUSTOM_PREFIX}<slug>, got {cid!r}"
    concept = esco_concepts.get(cid)
    if concept is None:
        return f"unknown ESCO URI {cid}"
    if not alias:
        return "alias is required for rows that extend an ESCO concept"
    # canonical_name is optional on ESCO rows; when present it guards against a wrong URI.
    if name and name != concept.name_en:
        return f"canonical_name {name!r} does not match ESCO label {concept.name_en!r} of {cid}"
    return None


def _conflicting_aliases(aliases: list[Alias]) -> list[str]:
    owners: dict[str, set[str]] = {}
    for alias in aliases:
        owners.setdefault(alias.alias_norm, set()).add(alias.canonical_id)
    return [
        f"alias {norm!r} points to several concepts: {', '.join(sorted(ids))}"
        for norm, ids in sorted(owners.items())
        if len(ids) > 1
    ]
