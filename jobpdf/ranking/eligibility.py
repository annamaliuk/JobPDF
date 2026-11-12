"""Which vacancies a candidate should not be shown for where they live (JM-49).

``abroad_adjustments`` turns the candidate's country into ``VacancyAdjustment`` objects, the
shape ``rank_vacancies`` already reads: a vacancy it hides is still fully scored and listed in
the ranking's ``hidden`` list, never dropped, and its ``reasons`` say why.

The one rule today: a vacancy that must be worked on site (or hybrid) in another country is
hidden. A remote vacancy is never hidden here, and neither is a vacancy whose mode or country
is not known: showing a job the person cannot take costs a glance, hiding one they could take
costs the job. Eligibility of *remote* jobs by where the candidate lives ("USA only") is the
separate eligibility ticket and will add its rules here.

The countries are ISO 3166-1 alpha-2 codes; ``normalize_country`` is the one place that turns
text into one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from jobpdf.places import is_country
from jobpdf.ranking.models import VacancyFeatures
from jobpdf.ranking.score import VacancyAdjustment

ON_SITE_ABROAD = "on_site_abroad"  # the reason code; the report words it
NOT_REMOTE_MODES = frozenset({"on_site", "hybrid"})


def normalize_country(value: str | None) -> str | None:
    """An upper-case ISO code from the list of known countries, else None."""
    code = (value or "").strip().upper()
    return code if is_country(code) else None


def is_on_site_abroad(vacancy: VacancyFeatures, home_country: str) -> bool:
    return (vacancy.work_mode in NOT_REMOTE_MODES
            and vacancy.country is not None
            and vacancy.country != home_country)


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
    verdicts: dict[str, set[bool]] = {}
    for vacancy in vacancies:
        verdicts.setdefault(vacancy.id, set()).add(is_on_site_abroad(vacancy, home))
    reason = VacancyAdjustment(hidden=True, reasons=[ON_SITE_ABROAD])
    return {vid: reason for vid, found in verdicts.items() if found == {True}}
