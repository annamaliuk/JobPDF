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
    result = extract(text, tags=["python", "Docker", "Remote", "Airflow"])
    assert result.required_skills == ["Python", "Docker"]  # only tags the taxonomy knows
    assert result.quotes["Docker"] == "(source tag)"
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


# --- regressions from the first check on real postings (2026-10-17) -------------------

REAL_VOCAB = SkillVocabulary.from_norms(
    ["python", "sql", "tableau", "ruby", "grpc", "redis", "communication", "innovation",
     "creativity", "reliability", "help customers", "make decisions", "pregnancy",
     "speak spanish", "cantonese", "bookkeeping", "attention to detail"],
    weak=["communication", "innovation", "creativity", "reliability", "help customers",
          "make decisions", "pregnancy", "speak spanish", "bookkeeping",
          "attention to detail"],
)

REAL_HTML = """
<p>Everything we do is the result of the creativity of our team, and its reliability.</p>
<p><strong>What you'll do</strong></p>
<ul><li>Help customers adopt the platform and make decisions with Python</li></ul>
<p><strong>Requirements</strong></p>
<ul><li>Strong communication and attention to detail</li>
<li>Knowledge of Tableau and SQL is a huge plus</li>
<li>You speak Spanish at a native level.</li>
<li>Must speak, read and write Cantonese and English.</li>
<li>Experience with Ruby or gRPC is highly desirable.</li></ul>
<p>Your Future at Acme</p>
<p>Acme is a place where you can grow your career and fuel your innovation.</p>
<p>Our policies are based on merit, regardless of race, sex (including pregnancy), age.</p>
"""


def test_weak_matches_count_only_in_requirement_sections() -> None:
    result = RulesExtractor(vocabulary=REAL_VOCAB).extract(posting(1, description=REAL_HTML))
    assert result.required_skills == ["Python", "communication", "attention to detail"]
    assert result.preferred_skills == ["Tableau", "SQL", "Ruby", "gRPC"]
    assert result.languages == ["Spanish", "English", "Chinese"]
    assert result.responsibilities == [
        "Help customers adopt the platform and make decisions with Python"]


def test_weak_matches_count_anywhere_in_a_posting_without_sections() -> None:
    text = "<p>You will do bookkeeping and invoicing for our clients every day.</p>" * 4
    result = RulesExtractor(vocabulary=REAL_VOCAB).extract(posting(1, description=text))
    assert result.required_skills[0] == "bookkeeping"


def test_boilerplate_sentences_set_nothing() -> None:
    text = ("<h3>Requirements</h3><ul><li>Python</li></ul><p>We are an equal opportunity "
            "employer and offer 401(k), 5 years of tenure bonus, pregnancy leave.</p>" * 3)
    result = RulesExtractor(vocabulary=REAL_VOCAB).extract(posting(1, description=text))
    assert result.min_years_experience is None
    assert "pregnancy" not in result.required_skills


def test_a_language_clause_with_a_cue_is_left_out_but_the_rest_counts() -> None:
    text = ("<h3>Requirements</h3><ul><li>You are fluent in Portuguese and English, German "
            "would be a big plus.</li></ul><p>Find more English Speaking Jobs in Germany on "
            "Arbeitnow</p>" + "<p>We value people.</p>" * 10)
    assert extract(text).languages == ["English", "Portuguese"]


def test_weak_skills_in_a_sentence_stating_a_need_count_outside_requirements() -> None:
    text = ("<p>This role is ideal for someone with strong communication skills.</p>"
            "<h3>Responsibilities</h3><ul><li>Help customers succeed</li></ul>"
            + "<p>We value people.</p>" * 10)
    result = RulesExtractor(vocabulary=REAL_VOCAB).extract(posting(1, description=text))
    assert result.required_skills[:1] == ["communication"]
    assert "Help customers" not in result.required_skills


def test_a_bare_heading_phrase_line_is_a_heading() -> None:
    placed = __import__("jobpdf.ingestion.requirement_rules", fromlist=["x"]).split_sections(
        [__import__("jobpdf.ingestion.posting_text", fromlist=["x"]).Line(t, is_bullet=True)
         for t in ("Responsibilities", "Ship features")])
    assert [(p.line.text, p.section) for p in placed] == [("Ship features", "responsibilities")]
