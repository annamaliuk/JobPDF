"""Count how the matcher resolves the skills in the extraction fixtures (JM-18).

Runs the exact and vector tiers against the real built index with the LLM
switched off, so the "would need the LLM" count says how often JM-18's LLM tier
is needed. Prints counts only, never skill text.

Usage (from the repo root, with the db service up and the index built):
    uv run python scripts/match_fixture_skills.py
    uv run python scripts/match_fixture_skills.py --replay-dir path/to/recordings
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import psycopg

from jobpdf.normalization.db import database_url
from jobpdf.normalization.index import SkillIndex
from jobpdf.normalization.matcher import LLM_UNAVAILABLE, SkillMatcher
from jobpdf.normalization.text import normalize

ROOT = Path(__file__).resolve().parents[1]
GOLD_DIR = ROOT / "tests" / "extraction" / "fixtures" / "gold"


def skill_names(profile: dict) -> list[str]:
    """Skill mentions plus each job's skills_used: everything JM-20 will match."""
    names = [s.get("name", "") for s in profile.get("skills", [])]
    for job in profile.get("experience", []):
        names += job.get("skills_used", [])
    return names


def load_profiles(replay_dir: Path | None) -> dict[str, list[dict]]:
    sources = {"gold": [json.loads(p.read_text(encoding="utf-8"))
                        for p in sorted(GOLD_DIR.glob("*.json"))]}
    if replay_dir is not None:
        sources["replay"] = [json.loads(p.read_text(encoding="utf-8"))["tool_input"]
                             for p in sorted(replay_dir.glob("*.json"))]
    return sources


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replay-dir", type=Path, help="recorded extraction responses")
    args = parser.parse_args(argv)

    url = database_url()
    if not url:
        print("DATABASE_URL: missing (environment or .env)")
        return 1
    try:
        index = SkillIndex.connect(url)
    except psycopg.OperationalError as exc:
        print(f"database not reachable ({type(exc).__name__}); start it: docker compose up -d db")
        return 1
    with index:
        if "build" not in index.meta:
            print("index not built; run scripts/build_index.py")
            return 1
        threshold = index.meta.get("thresholds", {}).get("auto_accept_distance")
        print(f"auto_accept_distance: {threshold if threshold is not None else 'missing'}")
        matcher = SkillMatcher(index, tool_caller=None)
        for source, profiles in load_profiles(args.replay_dir).items():
            names = [n for profile in profiles for n in skill_names(profile)]
            # One per normalized form, as the matcher and its cache see them.
            unique = list({normalize(n): n for n in reversed(names) if normalize(n)}.values())
            matches = matcher.match_many(unique)
            tiers = Counter(m.tier.value for m in matches)
            needs_llm = sum(LLM_UNAVAILABLE in m.warnings for m in matches)
            warnings = Counter(w for m in matches for w in m.warnings)
            print(f"{source}: {len(profiles)} profiles, {len(names)} skill mentions, "
                  f"{len(unique)} unique")
            print("  tiers: " + ", ".join(f"{t} {tiers.get(t, 0)}"
                                          for t in ("exact", "vector", "llm", "unmatched")))
            print(f"  would need the LLM: {needs_llm} of {len(unique)}"
                  f" ({needs_llm / len(unique):.0%})" if unique else "  no skills")
            print(f"  warnings: {dict(warnings) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
