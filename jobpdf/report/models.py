"""The CV report model (JM-41): what the report page shows, as finished text.

Every field is a sentence, a short label or a value a template needs for a link or a
style (band key, URL). Templates print fields as they are and add no wording of their
own beyond section titles, so every sentence the user reads is tested here.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict

from jobpdf.ranking.score import Band


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class StatTile(_Frozen):
    value: str  # "12", "4½ yrs"
    label: str  # "strong fits"


class MatchCard(_Frozen):
    """One of the best matches."""

    vacancy_id: str
    title: str
    company: str | None
    location: str | None
    posted: str | None  # "2 days ago"
    url: str | None
    band: Band
    band_name: str  # "Strong fit"
    skills_sentence: str  # "You have all 7 required skills and 2 of the 4 nice-to-have ones."
    required_short: str  # "7 of 7"
    similarity: str | None  # "high" | "medium" | "low"; None when not computed
    missing_required: list[str]
    missing_preferred: list[str]


class ReachCard(_Frozen):
    """A partial match that one or two skills would make strong."""

    vacancy_id: str
    title: str
    company: str | None
    location: str | None
    url: str | None
    band_now: str  # "Partial fit now"
    band_after: str  # "Strong fit"
    sentence: str  # "You have 6 of the 8 required skills. Adding these two makes it ..."
    add: list[str]  # the skills to add


class LearnItem(_Frozen):
    skill: str
    required_in: str | None  # "Required in 18 of your 50 closest vacancies"
    unlocks: str  # "+9 strong matches"


class DemandItem(_Frozen):
    skill: str
    count: int
    text: str  # "41 of 50"


class SkillReading(_Frozen):
    written: str  # as in the CV: "GBQ"
    read_as: str  # the standard name: "Google BigQuery"


class SkillsSection(_Frozen):
    recognised: list[str]  # standard names, first-seen order
    readings: list[SkillReading]
    readings_sentence: str | None  # "We read “GBQ” as Google BigQuery, so ..."
    unrecognised: list[str]
    unrecognised_sentence: str | None
    empty_sentence: str | None  # set only when no skill was recognised


class MatchRow(_Frozen):
    vacancy_id: str
    title: str
    company: str | None
    url: str | None
    band: Band
    band_name: str
    required_short: str  # "6 of 7"
    posted: str | None


class Report(_Frozen):
    generated_on: date
    headline: str
    headline_note: str | None  # e.g. the pointer to "Within reach" when nothing is strong
    summary: list[str]  # 0 to 4 sentences
    stats: list[StatTile]
    notes: list[str]  # one sentence per warning
    best_matches: list[MatchCard]
    strong_count: int  # for "See all 12 strong matches"
    within_reach: list[ReachCard]
    skills_to_learn: list[LearnItem]
    in_demand: list[DemandItem]
    skills: SkillsSection
    all_matches_intro: str | None  # "59 strong and partial matches, best first."
    all_matches: list[MatchRow]
    scoring: list[str]  # "How we score fit" paragraphs
    source_note: str
