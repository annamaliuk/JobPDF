"""Shared doubles and builders for the pipeline (JM-28) and API response (JM-29) tests."""

import datetime as dt
from pathlib import Path

from jobpdf.extraction.llm import FakeToolCaller, ToolCallResult
from jobpdf.normalization.match_cache import MatchTier, SkillMatch
from jobpdf.normalization.service import ProfileSkills, ServiceVersions, profile_skill_paths
from jobpdf.pipeline import analyze_cv
from jobpdf.ranking.models import SkillRef, VacancyFeatures
from tests.normalization.fakes import FakeEmbedder

FIXTURES = Path(__file__).parents[1] / "extraction" / "fixtures"
TODAY = dt.date(2026, 10, 3)
PYTHON = "http://data.europa.eu/esco/skill/ccd0a1d9-afda-43d9-b901-96344886e14d"
SQL = "http://data.europa.eu/esco/skill/sql"
VERSIONS = ServiceVersions(taxonomy="v1.2.1", index="abc123", matcher_prompt="1")
# Text that is only in the CV: it must never reach a log line, a warning or an error.
SECRETS = ("Olena", "Testenko", "Acme Widgets", "Globex", "olena.testenko", "single_column")


def cv_bytes(name: str = "single_column.pdf") -> bytes:
    return (FIXTURES / name).read_bytes()


def reply(tool_input: dict, tokens: tuple = (1000, 200)) -> ToolCallResult:
    return ToolCallResult(tool_input=tool_input, stop_reason="tool_use",
                          input_tokens=tokens[0], output_tokens=tokens[1], model="fake-model")


def extracted(**changes) -> dict:
    data = {
        "summary": None,
        "skills": [
            {"name": "Python", "kind": "hard", "found_in": "skills_list",
             "quote": "Python, SQL, Apache Airflow"},
            {"name": "SQL", "kind": "hard", "found_in": "skills_list",
             "quote": "Python, SQL, Apache Airflow"},
        ],
        "experience": [
            {"title": "Data Engineer", "company": "Acme Widgets LLC", "location": None,
             "start": "2021", "end": "2024", "description": None, "skills_used": ["Python"],
             "quote": "Data Engineer\nAcme Widgets LLC"},
        ],
        "education": [], "languages": [], "certifications": [],
    }
    data.update(changes)
    return data


def caller(*responses) -> FakeToolCaller:
    return FakeToolCaller(list(responses) or [reply(extracted())])


class FakeNormalizer:
    """Same call as NormalizationService.normalize_profile; Python and SQL are known."""

    known = {"python": PYTHON, "sql": SQL}

    def __init__(self, warnings: list[str] | None = None) -> None:
        self.profiles: list = []
        self.warnings = warnings or []

    def normalize_profile(self, profile) -> ProfileSkills:
        self.profiles.append(profile)
        matches = {}
        for path, raw in profile_skill_paths(profile):
            concept = self.known.get(raw.lower())
            matches[path] = SkillMatch(
                raw=raw, normalized=raw.lower(), concept_id=concept,
                preferred_label=raw if concept else None,
                tier=MatchTier.EXACT if concept else MatchTier.UNMATCHED,
                warnings=[] if concept else list(self.warnings),
            )
        return ProfileSkills(matches=matches, versions=VERSIONS)


def vacancy(vid: str, required: list[str]) -> VacancyFeatures:
    refs = [SkillRef(concept_id=FakeNormalizer.known.get(s), raw=s, normalized=s)
            for s in required]
    return VacancyFeatures(id=vid, source="synthetic", title=f"Role {vid}", required_skills=refs)


VACANCIES = [vacancy("both", ["python", "sql"]), vacancy("one", ["python", "rust"]),
             vacancy("none", ["rust", "go"])]


def run(content: bytes | None = None, filename: str = "cv.pdf", **kwargs):
    kwargs.setdefault("caller", caller())
    kwargs.setdefault("normalizer", FakeNormalizer())
    kwargs.setdefault("embedder", FakeEmbedder())
    kwargs.setdefault("today", TODAY)
    vacancies = kwargs.pop("vacancies", VACANCIES)
    return analyze_cv(cv_bytes() if content is None else content, filename, vacancies, **kwargs)


