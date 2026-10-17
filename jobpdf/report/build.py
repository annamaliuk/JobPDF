"""Build the readable CV report from the API response (JM-41).

``build_report(response, today=...)`` turns ``AnalysisResponse`` (JM-29, with JM-42's
insights) into a ``Report``: plain-English sentences per section of the report page, in
page order. It is pure and deterministic: no HTML, no LLM, no network, and dates are
counted from ``today``, never the clock.

Why the API response and not ``PipelineResult``: it already carries skill labels,
matched and missing skills, warning sentences and insights, and it leaves out CV quotes
and grounding offsets, so the page and the JSON API show the same facts and the report
never holds text it should not print.

Privacy. The report uses short CV fields only (job titles, skill names), never the
summary, job descriptions or quotes, and passes each through ``safe_text`` so an email,
phone number or URL typed into them is removed. It shows no contact data.

Skill groups. The taxonomy keeps no ESCO skill groups (JM-16 fetches only the skills and
digital-skills files), so "Your skills" is one flat list and there is no "strongest
area" sentence. No categories are invented.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

from jobpdf.api.response import (
    AnalysisResponse,
    Insights,
    MatchOut,
    ProfileOut,
    Scoring,
    SkillOut,
)
from jobpdf.ranking.overlap import PREFERRED_WEIGHT, REQUIRED_WEIGHT
from jobpdf.ranking.score import Band
from jobpdf.report.models import (
    DemandItem,
    LearnItem,
    MatchCard,
    MatchRow,
    ReachCard,
    Report,
    SkillReading,
    SkillsSection,
    StatTile,
)
from jobpdf.report.wording import (
    join_natural,
    percent,
    plural,
    posted_text,
    required_sentence,
    required_short,
    safe_text,
    similarity_name,
    years_text,
    years_tile,
)

BEST_MATCHES = 3
ALL_MATCHES_ROWS = 50
BAND_NAMES: dict[Band, str] = {
    Band.STRONG: "Strong fit",
    Band.PARTIAL: "Partial fit",
    Band.WEAK: "Weak fit",
    Band.UNSCORED: "Not scored",
}
LISTED_BANDS = (Band.STRONG, Band.PARTIAL)  # what "All matches" lists
SOURCE_NOTE = "Vacancies from Remotive.com. Every listing links to its original posting."
SOURCE_ONLY_NOTE = "We only use what's written in your CV and in the vacancy."
DEFAULT_SEM_MIN = 0.3  # used only if a response carries no thresholds


def build_report(response: AnalysisResponse, *, today: date) -> Report:
    """The report for one analysed CV. ``today`` dates the report and "2 days ago"."""
    strong = response.band_counts.get(Band.STRONG, 0)
    partial = response.band_counts.get(Band.PARTIAL, 0)
    vacancies = response.pagination.total + response.hidden_count
    skills = _skills(response.profile)
    sem_min = response.scoring.thresholds.get("sem_min", DEFAULT_SEM_MIN)
    headline, headline_note = _headline(strong, vacancies, response.insights)
    return Report(
        generated_on=today,
        headline=headline,
        headline_note=headline_note,
        summary=_summary(response, skills, strong, partial, vacancies),
        stats=_stats(response.profile, len(skills.recognised), strong, partial),
        notes=_notes(response),
        best_matches=[_card(m, today, sem_min) for m in _strong(response)[:BEST_MATCHES]],
        strong_count=strong,
        within_reach=_within_reach(response),
        skills_to_learn=_skills_to_learn(response.insights),
        in_demand=_in_demand(response.insights),
        skills=skills,
        all_matches_intro=_all_matches_intro(strong + partial),
        all_matches=[_row(m, today) for m in _listed(response)[:ALL_MATCHES_ROWS]],
        scoring=_scoring(response.scoring),
        source_note=SOURCE_NOTE,
    )


# --- summary -----------------------------------------------------------------------


def _headline(strong: int, vacancies: int, insights: Insights) -> tuple[str, str | None]:
    if vacancies == 0:
        return "There are no vacancies to compare your CV with right now", None
    if strong:
        verb = "fits" if strong == 1 else "fit"
        return f"{plural(strong, 'remote job')} {verb} you strongly right now", None
    note = ("See “Within reach” for jobs that one or two more skills would make "
            "strong matches." if insights.within_reach else None)
    return "No remote job fits you strongly yet", note


def _summary(response: AnalysisResponse, skills: SkillsSection, strong: int, partial: int,
             vacancies: int) -> list[str]:
    sentences = []
    years = years_text(response.profile.total_experience_months)
    role = _latest_title(response.profile)
    if years and role:
        sentences.append(f"You have {years} of experience, most recently as {role}.")
    elif years:
        sentences.append(f"You have {years} of experience.")
    elif role:
        sentences.append(f"Your most recent role is {role}.")

    if skills.recognised:
        sentences.append(f"We recognised {plural(len(skills.recognised), 'skill')}.")
    else:
        sentences.append(skills.empty_sentence or "")

    if vacancies:
        sentences.append(
            f"Out of {plural(vacancies, 'current remote vacancy', 'current remote vacancies')}, "
            f"{strong} {'fits' if strong == 1 else 'fit'} you strongly and {partial} partly."
        )
    if response.insights.skills_to_learn:
        skill = _name(response.insights.skills_to_learn[0].skill)
        sentences.append(f"The one skill that would open the most new options for you is "
                         f"{skill}.")
    return [s for s in sentences if s]


def _latest_title(profile: ProfileOut) -> str | None:
    """The job ending "present", else the latest end, else the latest start, else the
    first listed; ties go to the earlier job in the CV."""
    if not profile.experience:
        return None

    def key(item):
        i, job = item
        if job.end == "present":
            return (3, job.start or "", -i)
        if job.end:
            return (2, job.end, -i)
        if job.start:
            return (1, job.start, -i)
        return (0, "", -i)

    for _, job in sorted(enumerate(profile.experience), key=key, reverse=True):
        title = safe_text(job.title)
        if title:
            return title
    return None


def _stats(profile: ProfileOut, recognised: int, strong: int, partial: int) -> list[StatTile]:
    def tile(n: int, word: str) -> StatTile:
        label = plural(n, word).split(" ", 1)[1]
        return StatTile(value=str(n), label=label)

    return [
        tile(strong, "strong fit"),
        tile(partial, "partial fit"),
        StatTile(value=str(recognised),
                 label="skill recognised" if recognised == 1 else "skills recognised"),
        StatTile(value=years_tile(profile.total_experience_months), label="of experience"),
    ]


def _notes(response: AnalysisResponse) -> list[str]:
    """One sentence per warning code, in the order they were raised. The sentences are
    JM-29's fixed ``WARNING_DETAILS`` (an unknown code already has a generic one)."""
    seen: dict[str, str] = {}
    for warning in response.warnings:
        seen.setdefault(warning.code, warning.detail)
    return list(dict.fromkeys(seen.values()))


# --- matches -----------------------------------------------------------------------


def _strong(response: AnalysisResponse) -> list[MatchOut]:
    return [m for m in response.matches if m.band is Band.STRONG]


def _listed(response: AnalysisResponse) -> list[MatchOut]:
    return [m for m in response.matches if m.band in LISTED_BANDS]


def _card(match: MatchOut, today: date, sem_min: float) -> MatchCard:
    v, req, pref = match.vacancy, match.required, match.preferred
    return MatchCard(
        vacancy_id=v.id, title=v.title, company=v.company, location=v.location,
        posted=posted_text(v.posted_at, today), url=v.url, band=match.band,
        band_name=BAND_NAMES[match.band],
        skills_sentence=required_sentence(req.matched_count, req.count,
                                          pref.matched_count, pref.count),
        required_short=required_short(req.matched_count, req.count),
        similarity=similarity_name(match.components.semantic, sem_min),
        missing_required=[_name(s) for s in req.missing],
        missing_preferred=[_name(s) for s in pref.missing],
    )


def _row(match: MatchOut, today: date) -> MatchRow:
    v = match.vacancy
    return MatchRow(
        vacancy_id=v.id, title=v.title, company=v.company, url=v.url, band=match.band,
        band_name=BAND_NAMES[match.band],
        required_short=required_short(match.required.matched_count, match.required.count),
        posted=posted_text(v.posted_at, today),
    )


def _all_matches_intro(listed: int) -> str | None:
    if not listed:
        return None
    intro = f"{plural(listed, 'strong or partial match', 'strong and partial matches')}"
    intro += ", best first." if listed > 1 else "."
    if listed > ALL_MATCHES_ROWS:
        intro += f" Showing the first {ALL_MATCHES_ROWS}."
    return intro


# --- insights ----------------------------------------------------------------------


def _within_reach(response: AnalysisResponse) -> list[ReachCard]:
    by_id = {m.vacancy.id: m for m in response.matches}
    cards = []
    for item in response.insights.within_reach:
        match = by_id.get(item.vacancy_id)
        if match is None or not item.missing:  # not on this page; nothing to add
            continue
        have = (item.required_matched_count if item.required_matched_count is not None
                else match.required.matched_count)
        total = item.required_count if item.required_count is not None else match.required.count
        add = [_name(s) for s in item.missing]
        how = "Adding this one makes it" if len(add) == 1 else "Adding these two makes it"
        v = match.vacancy
        cards.append(ReachCard(
            vacancy_id=v.id, title=v.title, company=v.company, location=v.location, url=v.url,
            band_now=f"{BAND_NAMES[match.band]} now", band_after=BAND_NAMES[Band.STRONG],
            sentence=f"You have {have} of the {total} required skills. {how} a strong match.",
            add=add,
        ))
    return cards


def _closest(n: int, insights: Insights) -> str:
    if insights.working_set_size:
        return f"{n} of your {insights.working_set_size} closest vacancies"
    return plural(n, "vacancy", "vacancies")


def _skills_to_learn(insights: Insights) -> list[LearnItem]:
    return [
        LearnItem(
            skill=_name(item.skill),
            required_in=(f"Required in {_closest(item.required_in, insights)}"
                         if item.required_in is not None else None),
            unlocks=f"+{plural(item.unlocks, 'strong match', 'strong matches')}",
        )
        for item in insights.skills_to_learn
    ]


def _in_demand(insights: Insights) -> list[DemandItem]:
    of = f" of {insights.working_set_size}" if insights.working_set_size else ""
    return [DemandItem(skill=_name(item.skill), count=item.count, text=f"{item.count}{of}")
            for item in insights.in_demand]


# --- your skills -------------------------------------------------------------------


def _all_skills(profile: ProfileOut) -> Iterable[SkillOut]:
    yield from profile.skills
    for job in profile.experience:
        yield from job.skills


def _skills(profile: ProfileOut) -> SkillsSection:
    recognised: dict[str, str] = {}  # concept id -> standard name
    readings: dict[str, SkillReading] = {}  # concept id -> first differing spelling
    unrecognised: dict[str, str] = {}  # casefolded text -> first spelling
    for skill in _all_skills(profile):
        written = safe_text(skill.name)
        if skill.recognised and skill.concept_id:
            label = skill.label or written
            if not label:
                continue
            recognised.setdefault(skill.concept_id, label)
            if (written and not skill.low_confidence
                    and written.casefold() != label.casefold()
                    and skill.concept_id not in readings):
                readings[skill.concept_id] = SkillReading(written=written, read_as=label)
        elif written:
            unrecognised.setdefault(written.casefold(), written)

    reading_list = list(readings.values())
    readings_sentence = None
    if reading_list:
        pairs = join_natural([f"“{r.written}” as {r.read_as}" for r in reading_list])
        which = "that name" if len(reading_list) == 1 else "the standard names"
        readings_sentence = f"We read {pairs}, so vacancies that use {which} still count."

    unrecognised_list = list(unrecognised.values())
    unrecognised_sentence = None
    if len(unrecognised_list) == 1:
        unrecognised_sentence = ("We couldn't match this to a known skill, so it isn't used "
                                 "when comparing required skills.")
    elif unrecognised_list:
        unrecognised_sentence = ("We couldn't match these to known skills, so they aren't "
                                 "used when comparing required skills.")

    empty_sentence = None
    if not recognised:
        empty_sentence = ("We couldn't recognise any skills in your CV, so matches rely on "
                          "experience similarity alone.")
    return SkillsSection(
        recognised=list(recognised.values()), readings=reading_list,
        readings_sentence=readings_sentence, unrecognised=unrecognised_list,
        unrecognised_sentence=unrecognised_sentence, empty_sentence=empty_sentence,
    )


def _name(skill: SkillOut) -> str:
    """The standard name, else the text as written (vacancy skills or unrecognised)."""
    return skill.label or safe_text(skill.name) or skill.name


# --- how we score fit --------------------------------------------------------------


def _scoring(scoring: Scoring) -> list[str]:
    """Generated from the weights and thresholds that produced this result, so the text
    stays true if they are tuned (JM-26) or learned (JM-27)."""
    w_skill = scoring.weights.get("skill")
    w_semantic = scoring.weights.get("semantic")
    strong = scoring.thresholds.get("strong_coverage")
    partial = scoring.thresholds.get("partial_coverage")
    ratio = PREFERRED_WEIGHT / REQUIRED_WEIGHT
    nice = "half as much" if ratio == 0.5 else f"{percent(ratio)} as much"

    paragraphs = []
    if w_skill is not None and w_semantic is not None and w_semantic > 0 and w_skill > 0:
        paragraphs.append(
            f"Fit combines two things. About {percent(w_skill)} comes from how many of a "
            f"vacancy's required skills you have; nice-to-have skills count {nice}. The "
            f"other {percent(w_semantic)} is how similar your experience is to the job's "
            f"responsibilities."
        )
    else:
        paragraphs.append(
            f"Fit comes from how many of a vacancy's required skills you have; "
            f"nice-to-have skills count {nice}."
        )
    if strong is not None and partial is not None:
        paragraphs.append(
            f"A strong fit means you have at least {percent(strong)} of the required "
            f"skills and related experience. A partial fit means you have at least "
            f"{percent(partial)} of them."
        )
    paragraphs.append(SOURCE_ONLY_NOTE)
    return paragraphs
