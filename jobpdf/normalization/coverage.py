"""How many real job-ad skill tags the taxonomy covers by exact alias lookup (JM-16).

Remotive asks for at most 4 API calls per day, so the response is cached in
data/cache/ and only refetched on request. Tests never call the API.
"""

from __future__ import annotations

import csv
import json
import urllib.request
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from jobpdf.normalization.text import normalize

REMOTIVE_URL = "https://remotive.com/api/remote-jobs?category=software-dev"
DEFAULT_CACHE = Path("data/cache/remotive_software_dev.json")
DEFAULT_ALIASES = Path("data/taxonomy/aliases.csv")
TOP_N = 150
REQUEST_TIMEOUT_S = 60
_USER_AGENT = "JobPDF-taxonomy-coverage/1.0"


@dataclass(frozen=True)
class TagCount:
    tag_norm: str
    tag: str  # the most common spelling, for display
    count: int


@dataclass(frozen=True)
class CoverageReport:
    covered: list[TagCount]
    misses: list[TagCount]
    jobs: int

    @property
    def coverage(self) -> float:
        total = len(self.covered) + len(self.misses)
        return len(self.covered) / total if total else 0.0

    @property
    def weighted_coverage(self) -> float:
        """Share of tag occurrences (not distinct tags) that match."""
        hit = sum(t.count for t in self.covered)
        total = hit + sum(t.count for t in self.misses)
        return hit / total if total else 0.0


def load_jobs(
    cache: Path = DEFAULT_CACHE,
    refresh: bool = False,
    fetch: Callable[[], bytes] | None = None,
) -> dict:
    """The Remotive payload: from the cache, or fetched once and cached."""
    if cache.is_file() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8"))
    raw = (fetch or _fetch_remotive)()
    payload = json.loads(raw)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(raw)
    return payload


def _fetch_remotive() -> bytes:
    request = urllib.request.Request(REMOTIVE_URL, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
        return response.read()


def top_tags(payload: dict, n: int = TOP_N) -> list[TagCount]:
    """Most frequent tags across jobs, merged by normalize() ("Python" == "python")."""
    counts: Counter[str] = Counter()
    spellings: dict[str, Counter[str]] = {}
    for job in payload.get("jobs", []):
        for tag in job.get("tags") or []:
            norm = normalize(tag)
            if not norm:
                continue
            counts[norm] += 1
            spellings.setdefault(norm, Counter())[tag.strip()] += 1
    return [
        TagCount(norm, spellings[norm].most_common(1)[0][0], count)
        for norm, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:n]
    ]


def load_alias_norms(aliases_csv: Path = DEFAULT_ALIASES) -> set[str]:
    with aliases_csv.open(encoding="utf-8", newline="") as f:
        return {row["alias_norm"] for row in csv.DictReader(f)}


def measure(payload: dict, alias_norms: set[str], n: int = TOP_N) -> CoverageReport:
    tags = top_tags(payload, n)
    return CoverageReport(
        covered=[t for t in tags if t.tag_norm in alias_norms],
        misses=[t for t in tags if t.tag_norm not in alias_norms],
        jobs=len(payload.get("jobs", [])),
    )
