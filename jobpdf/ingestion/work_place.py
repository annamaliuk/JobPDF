"""Where a posting is worked: remote, hybrid or on site, and in which country (JM-49).

Both are decided once, at ingestion, from what the source sent, and stored with the vacancy;
ranking then only compares them with where the candidate lives (``ranking/eligibility.py``).

Work mode, most trusted first:

1. The posting says it is not remote ("no remote", "on-site only", "not a remote position").
2. The source's own flag (``remote`` true or false): Remotive, Arbeitnow and The Muse set it.
   A false flag with the word "hybrid" in the posting is hybrid; otherwise it is on site.
3. The words of the posting, for sources that send no flag (Adzuna, Jooble). A phrase that
   only a remote job uses ("fully remote", "work from home", "anywhere in the world") or the
   word "remote" in the title, the place or the tags means remote, and so does the word
   anywhere else in the text. "Hybrid" means hybrid (unless the posting also offers fully
   remote work); "on-site", "in the office" or "relocation package" mean on site.
4. A posting that names a real place and says nothing about remote work is taken as on site:
   these feeds are searched with the word "remote" and it was not found in the excerpt we
   hold, so the job is most likely an office job that mentions it in passing.
5. Otherwise unknown (``None``), and an unknown posting is never hidden.

The country is the one the source states, else the one the place text names
(``places.resolve_country``), else unknown.

Only the posting's own text is read, nothing is logged, and the answer for one posting never
depends on another.
"""

from __future__ import annotations

import re
from typing import Literal

from jobpdf.ingestion.posting_text import posting_lines
from jobpdf.ingestion.sources.models import SourcePosting
from jobpdf.places import resolve_country

REMOTE = "remote"
HYBRID = "hybrid"
ON_SITE = "on_site"
WorkMode = Literal["remote", "hybrid", "on_site"]
MAX_TEXT_CHARS = 20_000  # the description is read up to here; the rest is boilerplate

_NOT_REMOTE = re.compile(
    r"\b(?:no remote|not remote|non[- ]?remote|not a remote|on[- ]?site only|office only|"
    r"remote (?:work )?(?:is )?not (?:possible|available|offered))\b", re.I)
_REMOTE_PHRASE = re.compile(
    r"\b(?:fully[- ]remote|100\s?%\s*remote|completely remote|remote[- ]first|remote[- ]only|"
    r"work(?:ing)? (?:fully |completely )?remotely|work from home|work[- ]from[- ]home|wfh|"
    r"work from anywhere|anywhere in the world|remote (?:position|role|job|work|opportunity)|"
    r"this (?:is )?a remote|distributed team)\b", re.I)
_REMOTE_WORD = re.compile(r"\bremote(?:ly)?\b", re.I)
_HYBRID = re.compile(r"\bhybrid\b", re.I)
_ON_SITE = re.compile(
    r"\b(?:on[- ]?site|in[- ]office|office[- ]based|relocation (?:package|assistance|support)|"
    r"must (?:be|live) (?:located |based )?(?:in|near|within))\b", re.I)


def posting_text(posting: SourcePosting) -> str:
    """The description as one plain string (HTML removed), cut at ``MAX_TEXT_CHARS``."""
    return " ".join(line.text for line in posting_lines(posting.description))[:MAX_TEXT_CHARS]


def classify_work_mode(posting: SourcePosting) -> WorkMode | None:
    """``"remote"``, ``"hybrid"``, ``"on_site"`` or None when the posting does not say."""
    head = " ".join([posting.title, posting.location_raw or "", " ".join(posting.tags)])
    text = posting_text(posting)
    everything = f"{head} {text}"

    if _NOT_REMOTE.search(everything):
        return ON_SITE
    if posting.remote is True:
        return REMOTE
    hybrid = bool(_HYBRID.search(everything))
    if posting.remote is False:
        return HYBRID if hybrid else ON_SITE

    remote_named = bool(_REMOTE_WORD.search(head))
    remote_offered = remote_named or bool(_REMOTE_PHRASE.search(everything))
    if hybrid:
        # "Hybrid or fully remote" offers remote work; plain "hybrid" does not.
        return REMOTE if remote_offered else HYBRID
    if remote_offered or _REMOTE_WORD.search(text):
        return REMOTE
    if _ON_SITE.search(everything):
        return ON_SITE
    if posting.country or resolve_country(posting.location_raw):
        return ON_SITE
    return None


def posting_country(posting: SourcePosting) -> str | None:
    """An ISO code: the one the source states, else the one the place text names."""
    return posting.country or resolve_country(posting.location_raw)
