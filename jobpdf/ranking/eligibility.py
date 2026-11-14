"""Which vacancies a candidate asked not to be shown (JM-49, JM-51).

``abroad_adjustments`` turns the candidate's country, and ``niche_adjustments`` the industries
they chose to avoid, into ``VacancyAdjustment`` objects, the shape ``rank_vacancies`` already
reads: a vacancy it hides is still fully scored and listed in the ranking's ``hidden`` list,
never dropped, and its ``reasons`` say why. ``choice_adjustments`` joins the two: a vacancy
hidden for both carries both reasons.

The one rule today: a vacancy that must be worked on site (or hybrid) in another country is
hidden. A remote vacancy is never hidden here, and neither is a vacancy whose mode or country
is not known: showing a job the person cannot take costs a glance, hiding one they could take
costs the job. Eligibility of *remote* jobs by where the candidate lives ("USA only") is the
separate eligibility ticket and will add its rules here.

The other rule: a vacancy tagged with an industry the candidate chose to avoid (``niches.py``)
is hidden. A vacancy with no tags is never hidden, which includes every vacancy stored before
the tags existed, until it is processed again.

The countries are ISO 3166-1 alpha-2 codes; ``normalize_country`` is the one place that turns
text into one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence

from jobpdf.niches import known as known_niches
from jobpdf.places import is_country
from jobpdf.ranking.models import VacancyFeatures
from jobpdf.ranking.score import VacancyAdjustment

ON_SITE_ABROAD = "on_site_abroad"  # the reason codes; the report words them
RISKY_NICHE = "risky_niche"
REASONS = frozenset({ON_SITE_ABROAD, RISKY_NICHE})
NOT_REMOTE_MODES = frozenset({"on_site", "hybrid"})


def normalize_country(value: str | None) -> str | None:
    """An upper-case ISO code from the list of known countries, else None."""
    code = (value or "").strip().upper()
    return code if is_country(code) else None


def is_on_site_abroad(vacancy: VacancyFeatures, home_country: str) -> bool:
    return (vacancy.work_mode in NOT_REMOTE_MODES
            and vacancy.country is not None
            and vacancy.country != home_country)


def is_in_avoided_niche(vacancy: VacancyFeatures, avoid: Sequence[str]) -> bool:
    return any(code in vacancy.niches for code in avoid)


def _hide(vacancies: Sequence[VacancyFeatures], reason: str,
          hide: Callable[[VacancyFeatures], bool]) -> Mapping[str, VacancyAdjustment]:
    """Adjustments by vacancy id for the vacancies ``hide`` says to hide.

    Ranking looks adjustments up by ``vacancy.id`` alone, so two vacancies of different sources
    that share an id would share a verdict. That is checked: an id that two vacancies disagree
    on is not hidden for either (the safe side).
    """
    verdicts: dict[str, set[bool]] = {}
    for vacancy in vacancies:
        verdicts.setdefault(vacancy.id, set()).add(hide(vacancy))
    adjustment = VacancyAdjustment(hidden=True, reasons=[reason])
    return {vid: adjustment for vid, found in verdicts.items() if found == {True}}


def abroad_adjustments(vacancies: Sequence[VacancyFeatures],
                       home_country: str | None) -> Mapping[str, VacancyAdjustment]:
    """The vacancies to hide for being on site abroad; empty without a known country."""
    home = normalize_country(home_country)
    if home is None:
        return {}
    return _hide(vacancies, ON_SITE_ABROAD, lambda vacancy: is_on_site_abroad(vacancy, home))


def niche_adjustments(vacancies: Sequence[VacancyFeatures],
                      avoid_niches: Iterable[str] | None) -> Mapping[str, VacancyAdjustment]:
    """The vacancies to hide for being in an avoided industry; empty when none is chosen.
    Codes that are not in ``niches.py`` are ignored."""
    avoid = known_niches(avoid_niches)
    if not avoid:
        return {}
    return _hide(vacancies, RISKY_NICHE, lambda vacancy: is_in_avoided_niche(vacancy, avoid))


def choice_adjustments(vacancies: Sequence[VacancyFeatures], home_country: str | None = None,
                       avoid_niches: Iterable[str] | None = None,
                       ) -> Mapping[str, VacancyAdjustment]:
    """Everything the candidate asked to hide, one adjustment per vacancy with all its reasons."""
    joined: dict[str, VacancyAdjustment] = {}
    for rule in (abroad_adjustments(vacancies, home_country),
                 niche_adjustments(vacancies, avoid_niches)):
        for vid, adjustment in rule.items():
            earlier = joined.get(vid)
            joined[vid] = adjustment if earlier is None else VacancyAdjustment(
                hidden=True, reasons=[*earlier.reasons, *adjustment.reasons])
    return joined
