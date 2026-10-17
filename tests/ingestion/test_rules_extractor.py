from jobpdf.ingestion.requirement_rules import SkillVocabulary
from jobpdf.ingestion.requirements import (
    NO_REQUIREMENTS,
    PARTIAL_VOCABULARY,
    RULES_VERSION,
    TAGS_ADDED,
    TAGS_FALLBACK,
    RulesExtractor,
)
from tests.ingestion.ingest_fakes import posting

VOCAB = SkillVocabulary.from_norms([
    "python", "django", "postgresql", "docker", "kubernetes", "terraform", "go", "aws",
    "machine learning", "english", "communication",
])

POSTING_HTML = """
<p>We are a fintech founded 20 years ago.</p>
<p><strong>What you'll do</strong></p>
<ul><li>Build APIs in Python and Go</li><li>Run services on AWS</li></ul>
<h3>Requirements:</h3>
<ul><li>5+ years of backend experience, at least 2 years with Django</li>
<li>Strong PostgreSQL and Docker skills</li>
<li>Bachelor's degree in Computer Science or a related field</li>
<li>Fluent English, written and spoken</li></ul>
<p><b>Nice to have</b></p>
<ul><li>Kubernetes, Terraform, Docker</li><li>German (B2) would be great</li>
<li>7+ years in fintech</li></ul>
<h2>What we offer</h2><ul><li>Machine learning courses, fluent German lessons</li></ul>
"""


def extract(description, **changes):
    return RulesExtractor(vocabulary=VOCAB).extract(
        posting(1, description=description, **changes))


def test_a_full_posting_gives_every_field_with_quotes() -> None:
    result = extract(POSTING_HTML, tags=["fintech"])
    assert result.required_skills == ["Python", "Go", "AWS", "Django", "PostgreSQL", "Docker"]
    assert result.preferred_skills == ["Kubernetes", "Terraform"]  # Docker stays required
    assert result.responsibilities == ["Build APIs in Python and Go", "Run services on AWS"]
    assert result.min_years_experience == 5  # not 7 (nice to have) nor 20 (company age)
    assert result.education.level == "bachelor"
    assert result.education.field == "Computer Science"
    assert result.languages == ["English"]  # German is only nice to have
    assert result.warnings == []
    assert result.quotes["Django"].startswith("5+ years")
    assert result.quotes["language:English"] == "Fluent English, written and spoken"


def test_language_names_found_by_the_taxonomy_are_not_skills() -> None:
    text = "<h3>Requirements</h3><ul><li>Fluent English and Python</li></ul>" + "x" * 200
    result = extract(text)
    assert "English" not in result.required_skills
    assert result.languages == ["English"]


def test_an_unusable_description_falls_back_to_tags() -> None:
    result = extract("Python dev.", tags=["Python", "SQL"])
    assert result.required_skills == ["Python", "SQL"]
    assert TAGS_FALLBACK in result.warnings
    assert result.languages is None  # tags say nothing about languages


def test_few_skills_in_the_text_are_topped_up_with_tags() -> None:
    text = "<h3>Requirements</h3><p>You know Python.</p>" + "<p>We value people.</p>" * 15
    result = extract(text, tags=["python", "Airflow"])
    assert result.required_skills == ["Python", "Airflow"]
    assert TAGS_ADDED in result.warnings


def test_no_skills_anywhere_is_flagged() -> None:
    result = extract("<p>We value people.</p>" * 15, tags=[])
    assert result.required_skills == [] and NO_REQUIREMENTS in result.warnings
    assert result.languages == []


def test_without_the_taxonomy_build_the_result_is_flagged(tmp_path) -> None:
    (tmp_path / "custom_skills.csv").write_text(
        "canonical_id,canonical_name,alias,lang\nx:1,Python,py,en\n", encoding="utf-8")
    result = RulesExtractor(taxonomy_dir=tmp_path).extract(posting(1, description=POSTING_HTML))
    assert "Python" in result.required_skills
    assert PARTIAL_VOCABULARY in result.warnings


def test_the_extractor_is_versioned_and_deterministic() -> None:
    assert RulesExtractor.version == RULES_VERSION
    assert extract(POSTING_HTML) == extract(POSTING_HTML)


def test_switching_extractor_rebuilds_stored_vacancies_once() -> None:
    from jobpdf.ingestion.requirements import TagsExtractor
    from jobpdf.ingestion.vacancies import current_versions, ingest
    from tests.ingestion.ingest_fakes import T0, FakeEmbedder, FakeService, MemoryStore

    store, service, embedder = MemoryStore(), FakeService(), FakeEmbedder()
    postings = [posting(1, description=POSTING_HTML), posting(2, description=POSTING_HTML)]

    def run(extractor):
        return ingest(postings, store=store, extractor=extractor, normalizer=service,
                      embedder=embedder, fetched_at=T0)

    assert run(TagsExtractor()).new == 2
    rules = RulesExtractor(vocabulary=VOCAB)
    assert current_versions(service, embedder, rules).extraction == RULES_VERSION
    assert run(rules).stale == 2
    assert run(rules).unchanged == 2
    stored = store.rows[("remotive", "1")]["features"]
    assert stored.extraction_version == RULES_VERSION
    assert stored.min_years_experience == 5


def test_ingestion_uses_the_rules_by_default(monkeypatch) -> None:
    from jobpdf.ingestion import vacancies

    seen = {}

    def fake_ingest(postings, **kwargs):
        seen["extractor"] = kwargs["extractor"]
        raise RuntimeError("stop here")

    monkeypatch.setattr(vacancies, "load_cached_postings", lambda *a, **k: ([posting(1)], None))
    monkeypatch.setattr(vacancies, "ingest", fake_ingest)
    try:
        vacancies.run_ingest(["remotive"], store=object(), normalizer=object())
    except vacancies.IngestError:
        pass
    assert isinstance(seen["extractor"], RulesExtractor)


def test_the_check_script_renders_found_items_with_their_lines() -> None:
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "check_vacancy_requirements.py"
    spec = importlib.util.spec_from_file_location("check_vacancy_requirements", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    p = posting(1, description=POSTING_HTML)
    text = "\n".join(module.render(p, extract(POSTING_HTML)))
    assert ("| required: Django | 5+ years of backend experience, at least 2 years with Django"
            in text)
    assert "| min years: 5 |" in text
    assert "| language: English | Fluent English, written and spoken | |" in text
