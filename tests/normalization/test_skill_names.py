"""JM-46: where the Ukrainian skill names come from, and what happens when they cannot be read."""

import logging
import threading
import time

import pytest

from jobpdf.normalization import skill_names
from jobpdf.normalization.skill_names import (
    IndexSkillNames,
    choose_language,
    postgres_loader,
    preferred_language,
)
from tests.normalization.test_index import (  # noqa: F401  (fixtures and helpers of the index tests)
    build,
    db_url,
    options,
    schema,
    taxonomy_dir,
)

SECRET = "postgresql://jobpdf:hunter2@db.internal/jobpdf"


@pytest.mark.parametrize("asked, language", [
    ("uk", "uk"), ("en", "en"), ("UK", "uk"), (" uk ", "uk"), ("Uk", "uk"),
    ("", "en"), (None, "en"), ("ukr", "en"), ("ru", "en"), ("uk;en", "en"), ("u", "en"),
    ("<script>", "en"),
])
def test_only_a_known_language_code_is_taken_from_the_form(asked, language) -> None:
    assert choose_language(asked) == language


@pytest.mark.parametrize("header, language", [
    ("uk", "uk"),
    ("uk-UA", "uk"),
    ("uk-UA,uk;q=0.9,en;q=0.8", "uk"),
    ("UK-ua", "uk"),
    ("en-US,en;q=0.9,uk;q=0.8", "en"),  # only the first language counts
    ("en", "en"),
    ("ukrainian", "en"),
    ("ukx", "en"),
    ("", "en"),
    (None, "en"),
    ("*", "en"),
])
def test_the_form_preselects_ukrainian_only_when_the_browser_asks_for_it_first(
        header, language) -> None:
    assert preferred_language(header) == language


def test_the_supported_languages_are_exactly_these() -> None:
    assert skill_names.LANGUAGES == ("en", "uk") and skill_names.LANGUAGE_FIELD == "lang"


# --- the cache ---------------------------------------------------------------------------------


def test_the_names_are_loaded_once_and_kept() -> None:
    calls = []

    def load():
        calls.append(1)
        return {"c1": "Один"}

    names = IndexSkillNames(load)

    assert names.get() == {"c1": "Один"} and names.get() == {"c1": "Один"}
    assert calls == [1]


def test_the_cached_names_cannot_be_changed_by_the_loader_afterwards() -> None:
    source = {"c1": "Один"}
    names = IndexSkillNames(lambda: source)
    names.get()

    source["c2"] = "Два"

    assert names.get() == {"c1": "Один"}


def test_a_failed_load_gives_no_names_logs_only_the_error_type_and_is_not_retried_at_once(
        caplog) -> None:
    calls = []
    now = [100.0]

    def load():
        calls.append(1)
        raise RuntimeError(f"cannot connect to {SECRET}")

    names = IndexSkillNames(load, retry_after_s=60, clock=lambda: now[0])
    with caplog.at_level(logging.DEBUG):
        first = names.get()
        now[0] += 59
        second = names.get()

    assert first == {} and second == {} and calls == [1]
    assert "RuntimeError" in caplog.text
    assert "hunter2" not in caplog.text and SECRET not in caplog.text


def test_a_failed_load_is_tried_again_after_the_wait_and_then_kept() -> None:
    now = [0.0]
    answers = [RuntimeError("down"), {"c1": "Один"}]
    calls = []

    def load():
        calls.append(1)
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    names = IndexSkillNames(load, retry_after_s=60, clock=lambda: now[0])

    assert names.get() == {}
    now[0] = 60
    assert names.get() == {"c1": "Один"}
    now[0] = 10_000
    assert names.get() == {"c1": "Один"} and calls == [1, 1]


def test_an_empty_index_is_a_result_not_a_failure() -> None:
    calls = []
    names = IndexSkillNames(lambda: calls.append(1) or {})

    assert names.get() == {} and names.get() == {} and calls == [1]


def test_many_requests_at_once_load_the_names_once() -> None:
    calls = []
    release = threading.Event()

    def load():
        calls.append(1)
        release.wait(5)
        return {"c1": "Один"}

    names = IndexSkillNames(load)
    results = []
    threads = [threading.Thread(target=lambda: results.append(names.get())) for _ in range(8)]
    for thread in threads:
        thread.start()
    time.sleep(0.2)
    release.set()
    for thread in threads:
        thread.join(5)

    assert calls == [1] and results == [{"c1": "Один"}] * 8


def test_without_a_database_there_are_no_names() -> None:
    assert skill_names.no_names() == {}


# --- the index ---------------------------------------------------------------------------------


ROWS = [  # canonical_id, alias, lang, alias_kind
    ("c:python", "Python (computer programming)", "en", "preferred"),
    ("c:python", "Python (комп’ютерне програмування)", "uk", "preferred"),
    ("c:python", "пайтон", "uk", "alt"),
    ("c:team", "work in teams", "en", "preferred"),
    ("c:team", "працювати в команді", "uk", "preferred"),
    ("c:sql", "SQL", "en", "preferred"),  # no Ukrainian row: no name
    ("c:docker", "Docker", "en", "custom"),
    ("c:docker", "докер", "uk", "custom"),  # a Ukrainian alias, but not a preferred name
    ("c:blank", "   ", "uk", "preferred"),
]
EXPECTED = {"c:python": "Python (комп’ютерне програмування)", "c:team": "працювати в команді"}


@pytest.fixture
def plain_schema(db_url: str):  # noqa: F811
    """A throwaway schema holding ``taxonomy_alias`` without the vector column: the loader reads
    only text columns, so these tests need Postgres but not pgvector."""
    import uuid

    import psycopg

    name = f"test_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(db_url, autocommit=True, connect_timeout=10) as admin:
        admin.execute(f'CREATE SCHEMA "{name}"')
        admin.execute(f'CREATE TABLE "{name}".taxonomy_alias (canonical_id text, '
                      "canonical_name text, alias text, lang text, alias_kind text)")
        for cid, alias, lang, kind in ROWS:
            admin.execute(f'INSERT INTO "{name}".taxonomy_alias VALUES (%s, %s, %s, %s, %s)',
                          [cid, "x", alias, lang, kind])
    try:
        yield name
    finally:
        with psycopg.connect(db_url, autocommit=True, connect_timeout=10) as admin:
            admin.execute(f'DROP SCHEMA "{name}" CASCADE')


@pytest.mark.db
def test_the_loader_reads_only_the_preferred_ukrainian_name_of_each_concept(
        db_url, plain_schema, monkeypatch) -> None:  # noqa: F811
    import psycopg

    monkeypatch.setattr(skill_names, "connect", lambda url: psycopg.connect(
        url, options=f"-c search_path={plain_schema}", connect_timeout=10))

    assert postgres_loader(db_url)() == EXPECTED


@pytest.mark.db
def test_the_loader_reads_the_table_it_is_given(db_url, plain_schema, monkeypatch) -> None:  # noqa: F811
    import psycopg

    monkeypatch.setattr(skill_names, "connect", lambda url: psycopg.connect(
        url, options=f"-c search_path={plain_schema}", connect_timeout=10))

    with pytest.raises(psycopg.errors.UndefinedTable):
        postgres_loader(db_url, table="no_such_table")()


@pytest.mark.db
def test_the_loader_reads_an_index_built_by_the_real_build(
        db_url, schema, taxonomy_dir, monkeypatch) -> None:  # noqa: F811
    from jobpdf.normalization.db import connect

    build(db_url, schema, taxonomy_dir)
    monkeypatch.setattr(skill_names, "connect",
                        lambda url: connect(url, options=options(schema)))

    assert postgres_loader(db_url)() == {
        "esco:python": "Python (комп’ютерне програмування)",
        "esco:teamwork": "працювати в команді",
    }


@pytest.mark.db
def test_a_missing_table_is_a_failure_the_cache_survives(
        db_url, plain_schema, monkeypatch, caplog) -> None:  # noqa: F811
    import psycopg

    monkeypatch.setattr(skill_names, "connect", lambda url: psycopg.connect(
        url, options=f"-c search_path={plain_schema}", connect_timeout=10))
    names = IndexSkillNames(postgres_loader(db_url, table="no_such_table"))

    with caplog.at_level(logging.WARNING):
        assert names.get() == {}

    assert "ukrainian skill names unavailable (UndefinedTable)" in caplog.text


def test_an_unreachable_database_gives_no_names_instead_of_an_error() -> None:
    names = IndexSkillNames(postgres_loader("postgresql://nobody:pw@127.0.0.1:1/none"))

    assert names.get() == {}
