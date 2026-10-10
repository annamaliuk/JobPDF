import psycopg
import pytest

from jobpdf.extraction.llm import FakeToolCaller, ToolCallResult
from jobpdf.extraction.schema import (
    CandidateProfile,
    ExtractedJob,
    ExtractedProfile,
    Skill,
    SkillMention,
)
from jobpdf.normalization.index import Candidate
from jobpdf.normalization.match_cache import (
    InMemorySkillMatchCache,
    MatchTier,
    index_version,
)
from jobpdf.normalization.match_prompt import MATCHER_PROMPT_VERSION
from jobpdf.normalization.matcher import EMPTY_SKILL, SkillMatcher
from jobpdf.normalization.service import (
    LLM_DISABLED,
    NormalizationService,
    ServiceVersions,
)
from jobpdf.normalization.text import normalize

T = MatchTier
META = {
    "model": {"name": "intfloat/multilingual-e5-small", "revision": "abc"},
    "build": {"built_at": "2026-10-01T10:00:00+00:00"},
    "taxonomy": {"manifest_sha256": "f00", "esco_version": "v1.2.1"},
    "thresholds": {"auto_accept_distance": 0.03},
}
BIGQUERY = "jobpdf:skill/google-bigquery"
DOCKER = "jobpdf:skill/docker"
PYTHON = "esco:python"


def cand(cid: str, distance: float = 0.0) -> Candidate:
    return Candidate(canonical_id=cid, canonical_name=cid.upper(), matched_alias="a",
                     distance=distance)


class FakeIndex:
    """Scripted exact hits by normalized text; vector search finds nothing."""

    def __init__(self, exact: dict[str, list[Candidate]] | None = None, meta=None) -> None:
        self.exact = exact if exact is not None else {
            "gbq": [cand(BIGQUERY)], "docker": [cand(DOCKER)], "python": [cand(PYTHON)],
        }
        self.meta = META if meta is None else meta
        self.closed = False

    def lookup_exact(self, raw: str) -> list[Candidate]:
        return list(self.exact.get(normalize(raw), []))

    def search(self, raws: list[str], k: int = 10) -> list[list[Candidate]]:
        return [[] for _ in raws]

    def close(self) -> None:
        self.closed = True


class CountingMatcher(SkillMatcher):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[list[str]] = []

    def match_many(self, raws, context=None):
        self.calls.append(list(raws))
        return super().match_many(raws, context)


def versions() -> ServiceVersions:
    return ServiceVersions(taxonomy="v1.2.1", index=index_version(META),
                           matcher_prompt=MATCHER_PROMPT_VERSION)


def service(index: FakeIndex | None = None, **kwargs) -> tuple[NormalizationService,
                                                               CountingMatcher]:
    index = index or FakeIndex()
    matcher = CountingMatcher(index)
    return NormalizationService(matcher, versions(), index=index, **kwargs), matcher


def job(skills_used: list[str]) -> ExtractedJob:
    return ExtractedJob(title="Engineer", company=None, location=None, start="2021",
                        end="present", description=None, skills_used=skills_used,
                        quote="Engineer")


def candidate_profile() -> CandidateProfile:
    return CandidateProfile(
        summary=None,
        skills=[
            Skill(name="GBQ", kind="hard", found_in=["skills_list"], quotes=["GBQ"]),
            Skill(name="Docker", kind="hard", found_in=["skills_list"], quotes=["Docker"]),
            Skill(name="Teamwork", kind="soft", found_in=["summary"], quotes=["Teamwork"]),
        ],
        experience=[job([]), job(["gbq", "Python"])],
        education=[], languages=[], certifications=[], total_experience_months=None,
    )


# --- normalize_profile -------------------------------------------------------------


def test_normalize_profile_collects_every_location_in_one_call() -> None:
    svc, matcher = service()
    profile = candidate_profile()
    before = profile.model_dump()

    result = svc.normalize_profile(profile)

    assert len(matcher.calls) == 1
    assert list(result.matches) == [
        "skills[0]", "skills[1]", "skills[2]",
        "experience[1].skills_used[0]", "experience[1].skills_used[1]",
    ]
    assert result.matches["skills[0]"].concept_id == BIGQUERY
    assert result.matches["experience[1].skills_used[0]"].concept_id == BIGQUERY
    assert result.matches["skills[0]"].raw == "GBQ"
    assert result.matches["experience[1].skills_used[0]"].raw == "gbq"
    assert result.matches["skills[2]"].tier is T.UNMATCHED  # kept, not dropped
    assert result.versions == versions()
    assert profile.model_dump() == before


def test_normalize_profile_accepts_an_extracted_profile() -> None:
    svc, matcher = service()
    profile = ExtractedProfile(
        summary=None,
        skills=[SkillMention(name="Docker", kind="hard", found_in="skills_list",
                             quote="Docker")],
        experience=[job(["Python"])], education=[], languages=[], certifications=[],
    )

    result = svc.normalize_profile(profile)

    assert {p: m.concept_id for p, m in result.matches.items()} == {
        "skills[0]": DOCKER, "experience[0].skills_used[0]": PYTHON,
    }
    assert len(matcher.calls) == 1


def test_profile_without_skills_makes_no_call() -> None:
    svc, matcher = service()
    profile = candidate_profile().model_copy(update={"skills": [], "experience": [job([])]})

    result = svc.normalize_profile(profile)

    assert result.matches == {}
    assert matcher.calls == []


def test_profile_skills_serialize_for_storage() -> None:
    svc, _ = service()

    dumped = svc.normalize_profile(candidate_profile()).model_dump(mode="json")

    assert dumped["versions"] == {"taxonomy": "v1.2.1", "index": index_version(META),
                                  "matcher_prompt": MATCHER_PROMPT_VERSION}
    assert dumped["matches"]["skills[0]"]["tier"] == "exact"


# --- normalize_skill_groups ----------------------------------------------------------


def test_skill_groups_use_one_call_and_keep_group_and_item_order() -> None:
    svc, matcher = service()
    groups = {"required": ["Python", "GBQ"], "nice_to_have": [], "extra": ["Docker", "", "gbq"]}

    result = svc.normalize_skill_groups(groups)

    assert matcher.calls == [["Python", "GBQ", "Docker", "", "gbq"]]
    assert list(result) == ["required", "nice_to_have", "extra"]
    assert [m.raw for m in result["required"]] == ["Python", "GBQ"]
    assert result["nice_to_have"] == []
    assert [m.concept_id for m in result["extra"]] == [DOCKER, None, BIGQUERY]
    assert result["extra"][1].warnings == [EMPTY_SKILL]


def test_empty_groups_make_no_call() -> None:
    svc, matcher = service()

    assert svc.normalize_skill_groups({}) == {}
    assert svc.normalize_skill_groups({"required": [], "nice_to_have": []}) == {
        "required": [], "nice_to_have": []}
    assert matcher.calls == []


def test_normalize_skills_is_a_pass_through_with_context() -> None:
    index = FakeIndex(exact={})
    index.search = lambda raws, k=10: [[cand(PYTHON, 0.2)] for _ in raws]
    caller = FakeToolCaller([ToolCallResult(
        tool_input={"decisions": [{"item_id": 1, "concept_id": PYTHON}]},
        stop_reason="tool_use", input_tokens=1, output_tokens=1, model="fake-model")])
    matcher = CountingMatcher(index, caller)
    svc = NormalizationService(matcher, versions(), index=index, llm_model="fake-model")

    (m,) = svc.normalize_skills(("Pyhton",), context="vacancy required skills")

    assert (m.tier, m.concept_id) == (T.LLM, PYTHON)
    assert '"context": "vacancy required skills"' in caller.calls[0]["user"]


# --- concept_ids, versions, status ---------------------------------------------------


def test_concept_ids_drop_unmatched_and_duplicates_but_lists_keep_them() -> None:
    svc, _ = service()
    matches = svc.normalize_skills(["GBQ", "gbq", "Teamwork", "Docker"])

    assert NormalizationService.concept_ids(matches) == {BIGQUERY, DOCKER}
    assert len(matches) == 4 and matches[2].tier is T.UNMATCHED


def test_versions_and_status_reflect_the_dependencies() -> None:
    svc, _ = service(llm_model="claude-sonnet-4-6", cache_type="InMemorySkillMatchCache")

    assert svc.versions == versions()
    assert svc.status() == {
        "index": "ok",
        "taxonomy_version": "v1.2.1",
        "index_version": index_version(META),
        "matcher_prompt_version": MATCHER_PROMPT_VERSION,
        "llm": "enabled",
        "llm_model": "claude-sonnet-4-6",
        "cache": "InMemorySkillMatchCache",
    }


def test_status_without_llm_and_never_with_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-dummy-SECRET-value")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:SECRETPW@localhost/db")
    svc, _ = service()

    status = svc.status()

    assert status["llm"] == LLM_DISABLED and status["llm_model"] is None
    assert "SECRET" not in str(status)


class BrokenConnection:
    def execute(self, query: str):
        raise psycopg.OperationalError("server closed the connection")


def test_live_probe_reports_a_dropped_database() -> None:
    index = FakeIndex()
    index.connection = BrokenConnection()
    svc, _ = service(index)

    assert svc.status()["index"] == "unavailable"


def test_close_closes_the_index() -> None:
    index = FakeIndex()
    svc, _ = service(index)

    svc.close()

    assert index.closed


def test_service_uses_the_matchers_cache() -> None:
    cache = InMemorySkillMatchCache()
    index = FakeIndex()
    svc = NormalizationService(SkillMatcher(index, cache=cache), versions(), index=index)

    svc.normalize_skills(["GBQ"])

    assert len(cache) == 1
