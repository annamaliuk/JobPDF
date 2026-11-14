"""Which vacancies a candidate should not be shown, or not shown as strong (JM-49, JM-47).

Each rule turns what is known about the candidate into ``VacancyAdjustment`` objects, the shape
``rank_vacancies`` already reads: a vacancy it hides is still fully scored and listed in the
ranking's ``hidden`` list, never dropped, and its ``reasons`` say why.

- On site abroad (JM-49): a vacancy that must be worked on site (or hybrid) in another country
  is hidden.
- Remote outside the region (JM-47): a remote vacancy whose place text limits it to countries
  that do not include the candidate's ("USA only", "EU only") is hidden.
- Experience (JM-47): a vacancy that asks for at least ``EXPERIENCE_MARGIN_MONTHS`` more
  experience than the CV shows is capped at partial. It is never hidden for this.

Nothing happens when something is not known (the mode, the place, the country, the months):
showing a job the person cannot take costs a glance, hiding one they could take costs the job.
The two place rules need the candidate's country (``location_adjustments``); the experience rule
needs only the CV (``experience_adjustments``).

The countries are ISO 3166-1 alpha-2 codes; ``normalize_country`` is the one place that turns
text into one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from jobpdf.places import allowed_countries, is_country
from jobpdf.ranking.models import VacancyFeatures
from jobpdf.ranking.score import BAND_ORDER, Band, VacancyAdjustment

# The reason codes; the report words them.
ON_SITE_ABROAD = "on_site_abroad"
REMOTE_OUTSIDE_REGION = "remote_outside_region"
EXPERIENCE_SHORT = "experience_short"
LOCATION_REASONS = frozenset({ON_SITE_ABROAD, REMOTE_OUTSIDE_REGION})

NOT_REMOTE_MODES = frozenset({"on_site", "hybrid"})
EXPERIENCE_MARGIN_MONTHS = 30  # 2.5 years short of the posting's minimum caps the band


def normalize_country(value: str | None) -> str | None:
    """An upper-case ISO code from the list of known countries, else None."""
    code = (value or "").strip().upper()
    return code if is_country(code) else None


def is_on_site_abroad(vacancy: VacancyFeatures, home_country: str) -> bool:
    return (vacancy.work_mode in NOT_REMOTE_MODES
            and vacancy.country is not None
            and vacancy.country != home_country)


def is_remote_outside_region(vacancy: VacancyFeatures, home_country: str) -> bool:
    """Only the place text counts: a remote job's ``country`` may be where the company is."""
    if vacancy.work_mode != "remote":
        return False
    allowed = allowed_countries(vacancy.location_raw)
    return allowed is not None and home_country not in allowed


def is_experience_short(vacancy: VacancyFeatures, experience_months: int) -> bool:
    years = vacancy.min_years_experience
    return bool(years) and experience_months + EXPERIENCE_MARGIN_MONTHS <= years * 12


def abroad_adjustments(vacancies: Sequence[VacancyFeatures],
                       home_country: str | None) -> Mapping[str, VacancyAdjustment]:
    """Adjustments by vacancy id for the vacancies to hide; empty without a known country.

    Ranking looks adjustments up by ``vacancy.id`` alone, so two vacancies of different sources
    that share an id would share a verdict. That is checked: an id that two vacancies disagree
    on is not hidden for either (the safe side).
    """
    home = normalize_country(home_country)
    if home is None:
        return {}
    return _agreed(vacancies, lambda v: is_on_site_abroad(v, home),
                   VacancyAdjustment(hidden=True, reasons=[ON_SITE_ABROAD]))


def location_adjustments(vacancies: Sequence[VacancyFeatures],
                         home_country: str | None) -> Mapping[str, VacancyAdjustment]:
    """Both place rules for the candidate's country; empty without a known country."""
    home = normalize_country(home_country)
    if home is None:
        return {}
    remote = _agreed(vacancies, lambda v: is_remote_outside_region(v, home),
                     VacancyAdjustment(hidden=True, reasons=[REMOTE_OUTSIDE_REGION]))
    return merge_adjustments(abroad_adjustments(vacancies, home), remote)


def experience_adjustments(vacancies: Sequence[VacancyFeatures],
                           experience_months: int | None) -> Mapping[str, VacancyAdjustment]:
    """Caps for the vacancies that ask for clearly more experience; empty when it is unknown."""
    if experience_months is None:
        return {}
    return _agreed(vacancies, lambda v: is_experience_short(v, experience_months),
                   VacancyAdjustment(band_cap=Band.PARTIAL, reasons=[EXPERIENCE_SHORT]))


def merge_adjustments(*maps: Mapping[str, VacancyAdjustment]) -> dict[str, VacancyAdjustment]:
    """One adjustment per id: hidden if any says so, the lowest cap, every reason once."""
    merged: dict[str, VacancyAdjustment] = {}
    for adjustments in maps:
        for vid, new in adjustments.items():
            old = merged.get(vid)
            if old is None:
                merged[vid] = new
                continue
            caps = [c for c in (old.band_cap, new.band_cap) if c is not None]
            merged[vid] = VacancyAdjustment(
                hidden=old.hidden or new.hidden,
                band_cap=max(caps, key=BAND_ORDER.__getitem__) if caps else None,
                reasons=list(dict.fromkeys([*old.reasons, *new.reasons])))
    return merged


def _agreed(vacancies: Sequence[VacancyFeatures], rule: Callable[[VacancyFeatures], bool],
            adjustment: VacancyAdjustment) -> dict[str, VacancyAdjustment]:
    """``adjustment`` for every id the rule holds for, unless two vacancies with that id
    disagree (then neither gets it: the safe side)."""
    verdicts: dict[str, set[bool]] = {}
    for vacancy in vacancies:
        verdicts.setdefault(vacancy.id, set()).add(rule(vacancy))
    return {vid: adjustment for vid, found in verdicts.items() if found == {True}}
