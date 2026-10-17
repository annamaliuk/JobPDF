import pytest

from jobpdf.ingestion import requirement_rules as rules
from jobpdf.ingestion.posting_text import Line
from jobpdf.ingestion.requirement_rules import SkillVocabulary

VOCAB = SkillVocabulary.from_norms([
    "python", "django", "sql", "postgresql", "docker", "c++", "c#", ".net", "node.js",
    "machine learning", "go", "r", "spring", "java", "ci/cd", "aws", "data", "english",
])


def names(hits) -> list[str]:
    return [h.text for h in hits]


# --- sections -------------------------------------------------------------------------


@pytest.mark.parametrize(("heading", "section"), [
    ("Requirements", rules.REQUIREMENTS),
    ("What you'll bring:", rules.GENERAL),
    ("What we're looking for", rules.REQUIREMENTS),
    ("Nice to have", rules.NICE_TO_HAVE),
    ("Nice-to-have requirements", rules.NICE_TO_HAVE),
    ("Bonus points", rules.NICE_TO_HAVE),
    ("What you'll do", rules.RESPONSIBILITIES),
    ("What you will need", rules.REQUIREMENTS),
    ("Benefits & perks", rules.OTHER),
    ("About us", rules.OTHER),
    ("About you", rules.REQUIREMENTS),
    ("Dein Profil", rules.REQUIREMENTS),
    ("Deine Aufgaben", rules.RESPONSIBILITIES),
    ("Wir bieten", rules.OTHER),
    ("Вимоги:", rules.REQUIREMENTS),
    ("Буде плюсом:", rules.NICE_TO_HAVE),
    ("Обов'язки", rules.RESPONSIBILITIES),
    ("Our story", rules.GENERAL),
])
def test_headings_are_sorted_into_sections(heading: str, section: str) -> None:
    assert rules.classify_heading(heading) == section


def test_lines_take_the_section_of_the_heading_above_them() -> None:
    placed = rules.split_sections([
        Line("Intro"), Line("Requirements", is_heading=True), Line("Python"),
        Line("Benefits", is_heading=True), Line("Gym"),
    ])
    assert [(p.line.text, p.section) for p in placed] == [
        ("Intro", rules.GENERAL), ("Python", rules.REQUIREMENTS), ("Gym", rules.OTHER)]


# --- skills ---------------------------------------------------------------------------


def test_the_longest_alias_wins_and_punctuation_in_skill_names_is_kept() -> None:
    hits = VOCAB.scan("Machine learning with C++, C#, .NET and Node.js.")
    assert names(hits) == ["Machine learning", "C++", "C#", ".NET", "Node.js"]


def test_joined_skills_are_split_and_repeats_dropped() -> None:
    assert names(VOCAB.scan("Python/Django, python, SQL")) == ["Python", "Django", "SQL"]
    assert names(VOCAB.scan("CI/CD on AWS")) == ["CI/CD", "AWS"]


def test_generic_aliases_never_count_as_skills() -> None:
    assert names(VOCAB.scan("Work with data")) == []


@pytest.mark.parametrize(("line", "found"), [
    ("Go and Python services", ["Go", "Python"]),
    ("you will go far", []),
    ("Go", []),  # nothing else on the line backs it up
    ("Statistics in R or Python", ["R", "Python"]),
    ("Spring Boot or Java", ["Spring", "Java"]),
    ("Every spring we meet in Java", ["Java"]),
])
def test_ambiguous_names_need_a_capital_and_another_skill(line: str, found: list[str]) -> None:
    assert names(VOCAB.scan(line)) == found


def test_a_preference_cue_makes_the_sentence_preferred() -> None:
    hits = VOCAB.scan("You know SQL. Docker is a plus.")
    assert [(h.text, h.preferred) for h in hits] == [("SQL", False), ("Docker", True)]
    assert all(h.preferred for h in VOCAB.scan("SQL", preferred=True))


def test_the_vocabulary_falls_back_to_the_committed_custom_skills(tmp_path) -> None:
    (tmp_path / "custom_skills.csv").write_text(
        "canonical_id,canonical_name,alias,lang\nx:1,PostgreSQL,postgres,en\n", encoding="utf-8")
    vocab = SkillVocabulary.load(tmp_path)
    assert not vocab.complete
    assert names(vocab.scan("Postgres and PostgreSQL")) == ["Postgres", "PostgreSQL"]


def test_the_vocabulary_reads_the_taxonomy_build(tmp_path) -> None:
    (tmp_path / "aliases.csv").write_text(
        "canonical_id,alias,alias_norm,lang,alias_kind,exact_only\n"
        "x:1,Apache Kafka,apache kafka,en,preferred,false\n", encoding="utf-8")
    vocab = SkillVocabulary.load(tmp_path)
    assert vocab.complete and vocab.max_words == 2
    assert names(vocab.scan("Streams on Apache Kafka.")) == ["Apache Kafka"]


# --- years, education, languages ------------------------------------------------------


@pytest.mark.parametrize(("sentence", "years"), [
    ("5+ years of backend experience", [5]),
    ("at least 3 years with Python", [3]),
    ("2-4 years in a similar role", [2]),
    ("three years of experience", [3]),
    ("Mindestens 3 Jahre Berufserfahrung", [3]),
    ("Досвід роботи від 2 років", [2, 2]),
    ("3+ роки досвіду", [3]),
    ("we were founded 20 years ago", []),
    ("for 40 years now", []),
    ("candidates 18 years old", []),
    ("Python 3.12 and 2 years of Go", [2]),
])
def test_years_are_lower_bounds_of_stated_spans(sentence: str, years: list[int]) -> None:
    assert rules.years_in(sentence) == years


@pytest.mark.parametrize(("sentence", "level", "field"), [
    ("Bachelor's degree in Computer Science or a related field", "bachelor", "Computer Science"),
    ("BSc or MSc in Mathematics", "bachelor", "Mathematics"),
    ("Master's degree in Data Science", "master", "Data Science"),
    ("PhD in Physics", "phd", "Physics"),
    ("A university degree", "degree", None),
    ("Вища освіта", "degree", None),
    ("Abgeschlossenes Studium der Informatik", "degree", None),
])
def test_education_gives_the_lowest_named_level(sentence, level, field) -> None:
    hit = rules.education_in(sentence)
    assert (hit.level, hit.field) == (level, field)


@pytest.mark.parametrize("sentence", [
    "Bachelor's degree or equivalent experience",
    "Studium oder vergleichbare Ausbildung",
    "You care to a degree about design",
    "Scrum Master experience",
    "Strong Python skills",
])
def test_no_education_when_optional_or_not_mentioned(sentence: str) -> None:
    assert rules.education_in(sentence) is None


@pytest.mark.parametrize(("sentence", "languages"), [
    ("Fluent English, written and spoken", ["English"]),
    ("Sehr gute Deutsch- und Englischkenntnisse", ["English", "German"]),
    ("Англійська на рівні B2", ["English"]),
    ("English C1 and Polish B2", ["English", "Polish"]),
    ("You will polish our UI to a good level", []),
    ("We are an English company", []),
])
def test_languages_need_a_proficiency_word(sentence: str, languages: list[str]) -> None:
    assert rules.languages_in(sentence) == languages


def test_language_names_are_recognised_as_such() -> None:
    assert rules.is_language_name("English")
    assert rules.is_language_name("англійська")
    assert not rules.is_language_name("Python")


# --- responsibilities -----------------------------------------------------------------


def test_responsibilities_are_the_bullets_of_their_section_cut_to_length() -> None:
    long = "Build " + "very " * 60 + "fast APIs"
    placed = rules.split_sections([
        Line("What you'll do", is_heading=True), Line("We move fast."),
        Line("Ship features", is_bullet=True), Line(long, is_bullet=True),
        Line("Requirements", is_heading=True), Line("Python", is_bullet=True),
    ])
    result = rules.responsibilities_of(placed)
    assert result[0] == "Ship features"
    assert len(result) == 2 and result[1].endswith("…")
    assert len(result[1]) <= rules.MAX_RESPONSIBILITY_CHARS + 1


def test_without_bullets_the_section_lines_are_used() -> None:
    placed = rules.split_sections([Line("Your role", is_heading=True), Line("Own the API.")])
    assert rules.responsibilities_of(placed) == ["Own the API."]


def test_non_digital_esco_aliases_load_as_weak(tmp_path) -> None:
    (tmp_path / "aliases.csv").write_text(
        "canonical_id,alias,alias_norm,lang,alias_kind,exact_only\n"
        "e:1,Python,python,en,preferred,false\n"
        "e:2,innovation,innovation,en,preferred,false\n"
        "e:3,teamwork,teamwork,en,custom,false\n", encoding="utf-8")
    (tmp_path / "concepts.csv").write_text(
        "canonical_id,name_en,name_uk,description,skill_type,reuse_level,is_digital,source\n"
        "e:1,Python,,,,,true,esco\ne:2,innovation,,,,,false,esco\n"
        "e:3,teamwork,,,,,false,esco\n", encoding="utf-8")
    vocab = SkillVocabulary.load(tmp_path)
    assert vocab.weak == {"innovation"}  # a custom alias is trusted like a digital skill
    assert [(h.text, h.strong) for h in vocab.scan("Python, innovation and teamwork")] == [
        ("Python", True), ("innovation", False), ("teamwork", True)]


def test_huge_plus_is_a_preference_but_five_plus_years_is_not() -> None:
    assert rules.has_preferred_cue("Tableau is a huge plus")
    assert not rules.has_preferred_cue("5 plus years of SQL")
