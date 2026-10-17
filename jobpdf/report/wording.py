"""Small wording helpers for the CV report (JM-41).

Each one turns a number or a list into a fixed English phrase, so every sentence in the
report follows the same rules: natural lists, singular and plural, half years, and
relative dates counted from a given ``today`` (never the clock) so output is repeatable.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from datetime import date, datetime

from jobpdf.extraction.prompt import MASK_CHAR, mask_pii

HIGH_SIMILARITY = 0.6  # calibrated semantic score from which experience reads "high"

_MASKED_RUN = re.compile(f"{MASK_CHAR}+")
_SPACES = re.compile(r"\s+")


def join_natural(items: Sequence[str]) -> str:
    """A natural list: "Python", "Python and SQL", "Python, SQL and Airflow"."""
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def plural(n: int, word: str, many: str | None = None) -> str:
    """A count with its noun: "1 skill", "2 skills"; ``many`` for irregular plurals."""
    return f"{n} {word if n == 1 else (many or word + 's')}"


def half_years(months: int) -> float:
    """Months rounded to the nearest half year (halves round up)."""
    return math.floor(months / 6 + 0.5) / 2


def _years_number(years: float) -> str:
    whole = int(years)
    if years == whole:
        return str(whole)
    return f"{whole}½" if whole else "½"


def years_text(months: int | None) -> str | None:
    """Experience as text: "about 4½ years", "less than a year"; None when unknown."""
    if months is None:
        return None
    if months < 6:
        return "less than a year"
    years = half_years(months)
    if years == 0.5:
        return "about half a year"
    return f"about {_years_number(years)} {'year' if years == 1 else 'years'}"


def years_tile(months: int | None) -> str:
    """The stat tile value: "4½ yrs", "1 yr", "<1 yr", or "—" when unknown."""
    if months is None:
        return "—"
    if months < 6:
        return "<1 yr"
    years = half_years(months)
    return f"{_years_number(years)} {'yr' if years <= 1 else 'yrs'}"


def posted_text(posted_at: datetime | None, today: date) -> str | None:
    """How long ago: "today", "1 day ago", "12 days ago"; None when unknown."""
    if posted_at is None:
        return None
    days = (today - posted_at.date()).days
    if days <= 0:
        return "today"
    return f"{plural(days, 'day')} ago"


def required_sentence(have: int, total: int, preferred_have: int = 0,
                      preferred_total: int = 0) -> str:
    """The skills sentence of a match card, such as "You have all 7 required skills and
    2 of the 4 nice-to-have ones.", with the edge cases: one required skill, none
    matched, none stated."""
    if total == 0:
        return "This listing doesn't state required skills."
    if have == total:
        main = "You have the one required skill" if total == 1 else (
            f"You have all {total} required skills")
    elif have == 0:
        main = "You don't have the one required skill" if total == 1 else (
            f"You have none of the {total} required skills")
    else:
        main = f"You have {have} of the {total} required skills"
    if preferred_total:
        if preferred_have == preferred_total:
            nice = ("the one nice-to-have one" if preferred_total == 1
                    else f"all {preferred_total} nice-to-have ones")
        else:
            nice = f"{preferred_have} of the {preferred_total} nice-to-have ones"
        main += f" and {nice}" if have else f", but {nice}"
    return main + "."


def required_short(have: int, total: int) -> str:
    """The table cell: "6 of 7", or "—" when the listing states none."""
    return f"{have} of {total}" if total else "—"


def similarity_name(score: float | None, sem_min: float) -> str | None:
    """Experience similarity as "high", "medium" or "low"; None when not computed."""
    if score is None:
        return None
    if score >= HIGH_SIMILARITY:
        return "high"
    return "medium" if score >= sem_min else "low"


def percent(x: float) -> str:
    return f"{round(x * 100)}%"


def safe_text(text: str | None) -> str | None:
    """CV text with any email, phone number or URL removed; None if nothing is left.

    Every string that comes from the CV goes through here before it reaches the report.
    """
    if text is None:
        return None
    masked, _ = mask_pii(text)
    cleaned = _SPACES.sub(" ", _MASKED_RUN.sub(" ", masked)).strip(" ,;:|-–—()")
    return cleaned or None
