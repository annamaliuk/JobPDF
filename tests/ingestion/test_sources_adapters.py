import json
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlsplit

import pytest
from pydantic import ValidationError

from jobpdf.ingestion.sources import adzuna, arbeitnow, jooble, themuse
from jobpdf.ingestion.sources.adzuna import Adzuna
from jobpdf.ingestion.sources.arbeitnow import Arbeitnow
from jobpdf.ingestion.sources.jooble import Jooble
from jobpdf.ingestion.sources.models import (
    FeedError,
    Request,
    SourcePosting,
    parse_time,
    setting,
    stable_id,
)
from jobpdf.ingestion.sources.registry import SOURCE_NAMES, build_adapter
from jobpdf.ingestion.sources.remotive import to_source_posting
from jobpdf.ingestion.sources.themuse import TheMuse
from tests.ingestion.source_fakes import (
    APP_ID,
    APP_KEY,
    KEY,
    adzuna_job,
    adzuna_page,
    arbeitnow_job,
    arbeitnow_page,
    jooble_job,
    jooble_page,
    muse_job,
    muse_page,
)


def query(request: Request) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(urlsplit(request.url).query).items()}


# --- the common shape ----------------------------------------------------------------


def posting(**changes) -> dict:
    data = {"source": "x", "source_id": "1", "url": "https://x.test/1", "title": "T"}
    return {**data, **changes}


@pytest.mark.parametrize("changes", [
    {"url": "javascript:alert(1)"}, {"url": "ftp://x.test/1"}, {"url": ""},
    {"title": "  "}, {"source_id": ""}, {"source": " "},
])
def test_a_posting_needs_a_web_link_title_and_identity(changes) -> None:
    with pytest.raises(ValidationError):
        SourcePosting(**posting(**changes))


def test_a_posting_rejects_unknown_fields_and_is_immutable() -> None:
    with pytest.raises(ValidationError):
        SourcePosting(**posting(extra="x"))
    with pytest.raises(ValidationError):
        SourcePosting(**posting()).title = "other"


def test_a_naive_time_is_read_as_utc() -> None:
    made = SourcePosting(**posting(posted_at=datetime(2026, 9, 1, 12, 0)))

    assert made.posted_at == datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(("value", "expected"), [
    ("2026-09-20T10:00:00Z", datetime(2026, 9, 20, 10, tzinfo=timezone.utc)),
    ("2026-09-20T10:00:00+02:00", datetime(2026, 9, 20, 8, tzinfo=timezone.utc)),
    ("2026-09-20T10:00:00", datetime(2026, 9, 20, 10, tzinfo=timezone.utc)),
    ("2026-09-18T00:00:00.0000000", datetime(2026, 9, 18, tzinfo=timezone.utc)),
    (1_789_900_000, datetime.fromtimestamp(1_789_900_000, tz=timezone.utc)),
    ("not a date", None), ("", None), (None, None), (True, None), ({}, None), (10**30, None),
])
def test_times_are_read_leniently_and_never_raise(value, expected) -> None:
    assert parse_time(value) == expected


def test_a_request_never_shows_its_url_headers_or_body() -> None:
    request = Request("POST", f"https://jooble.org/api/{KEY}", headers={"X-Key": KEY},
                      body=KEY.encode())

    assert KEY not in repr(request) and KEY not in str(request)
    assert "POST" in repr(request)


def test_stable_ids_are_deterministic_and_depend_on_every_part() -> None:
    assert stable_id("a", "b") == stable_id("a", "b")
    assert stable_id("a", "b") != stable_id("b", "a") != stable_id("ab", "")
    assert stable_id("ab", "") != stable_id("a", "b")


def test_settings_come_from_the_given_mapping_only_when_one_is_passed(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("A=from-file\nexport B=1\n# C=commented\nD='quoted'\nE=\n")

    assert setting("A", {"A": " given "}, env_file) == "given"
    assert setting("A", {}, env_file) is None  # a mapping means "only this"
    assert setting("A", {"A": "  "}) is None


def test_settings_fall_back_to_the_env_file_and_a_real_variable_wins(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("A=from-file\nD='quoted'\nE=\n# C=commented\n")
    monkeypatch.delenv("A", raising=False)
    monkeypatch.delenv("C", raising=False)

    assert setting("A", None, env_file) == "from-file"
    assert setting("D", None, env_file) == "quoted"
    assert setting("E", None, env_file) is None
    assert setting("C", None, env_file) is None
    assert setting("MISSING", None, env_file) is None
    monkeypatch.setenv("A", "from-env")
    assert setting("A", None, env_file) == "from-env"
    assert setting("A", None, tmp_path / "nope") == "from-env"
    monkeypatch.delenv("A")
    assert setting("A", None, tmp_path / "nope") is None


# --- Remotive in the common shape -------------------------------------------------------


def test_a_remotive_posting_converts_to_the_common_shape() -> None:
    from jobpdf.ingestion.remotive import RemotivePosting

    original = RemotivePosting(
        id=7, url="https://remotive.com/remote-jobs/7", title="Dev", company_name="Acme",
        category="Software", tags=["python"], job_type="full_time",
        publication_date=datetime(2026, 9, 1, tzinfo=timezone.utc),
        candidate_required_location="Worldwide", salary="$50k", description="<p>x</p>")

    made = to_source_posting(original)

    assert (made.source, made.source_id, made.url, made.title) == (
        "remotive", "7", "https://remotive.com/remote-jobs/7", "Dev")
    assert (made.company, made.location_raw, made.remote) == ("Acme", "Worldwide", True)
    assert (made.tags, made.category, made.employment_type) == (
        ["python"], "Software", "full_time")
    assert made.salary_raw == "$50k" and made.description == "<p>x</p>"
    assert made.posted_at == original.publication_date and not made.description_is_excerpt


# --- The Muse -------------------------------------------------------------------------


def test_the_muse_first_request_asks_for_remote_jobs_without_a_key_by_default() -> None:
    request = TheMuse().next_request([])

    assert request.method == "GET" and request.url.startswith(themuse.API_URL + "?")
    assert query(request) == {"page": "1", "descending": "true",
                              "location": "Flexible / Remote"}


def test_the_muse_sends_its_key_only_when_there_is_one() -> None:
    assert query(TheMuse(api_key=KEY).next_request([]))["api_key"] == KEY
    assert "api_key" not in query(TheMuse().next_request([]))
    assert TheMuse().is_configured()


def test_the_muse_key_comes_from_the_environment_mapping() -> None:
    assert TheMuse.from_environment({"THE_MUSE_API_KEY": KEY}).api_key == KEY
    assert TheMuse.from_environment({}).api_key is None


def test_the_muse_walks_the_pages_it_reports_and_then_stops() -> None:
    adapter = TheMuse()
    first = muse_page([muse_job(1)], page=1, page_count=3)
    second = muse_page([muse_job(2)], page=2, page_count=3)
    third = muse_page([muse_job(3)], page=3, page_count=3)

    assert query(adapter.next_request([first]))["page"] == "2"
    assert query(adapter.next_request([first, second]))["page"] == "3"
    assert adapter.next_request([first, second, third]) is None


def test_the_muse_stops_when_a_page_does_not_say_where_it_is() -> None:
    assert TheMuse().next_request([{"results": []}]) is None
    assert TheMuse().next_request(["junk"]) is None
    assert TheMuse().next_request([muse_page([], page=True, page_count=5)]) is None


def test_the_muse_posting_fields_are_mapped() -> None:
    parsed = TheMuse().parse_page(muse_page([muse_job(1, tags=[{"name": "Remote OK"}])]))

    (made,) = parsed.postings
    assert (made.source, made.source_id) == ("themuse", "1001")
    assert made.url == "https://www.themuse.com/jobs/acme/1" and made.title == "Data Engineer 1"
    assert made.company == "Acme" and made.location_raw == "Flexible / Remote"
    assert made.remote is True and made.category == "Data and Analytics"
    assert made.tags == ["Mid Level", "Remote OK"] and made.employment_type == "external"
    assert made.posted_at == datetime(2026, 9, 20, 10, tzinfo=timezone.utc)
    assert made.description == "<p>Build pipelines 1</p>" and not made.description_is_excerpt
    assert (parsed.skipped, parsed.filtered, parsed.raw_count) == (0, 0, 1)


def test_the_muse_drops_jobs_that_are_not_remote_and_counts_them() -> None:
    onsite = muse_job(2, locations=[{"name": "New York, NY"}])
    no_place = muse_job(3, locations=[])
    flexible = muse_job(4, locations=[{"name": "Flexible / Remote"}, {"name": "Berlin"}])

    parsed = TheMuse().parse_page(muse_page([muse_job(1), onsite, no_place, flexible]))

    assert [p.source_id for p in parsed.postings] == ["1001", "1004"]
    assert parsed.filtered == 2 and parsed.skipped == 0


def test_the_muse_can_keep_on_site_jobs_when_asked() -> None:
    onsite = muse_job(2, locations=[{"name": "New York, NY"}])

    (made,) = TheMuse(remote_only=False).parse_page(muse_page([onsite])).postings

    assert made.remote is False and made.location_raw == "New York, NY"
    assert "location" not in query(TheMuse(remote_only=False).next_request([]))


def test_the_muse_skips_malformed_entries_without_losing_the_rest() -> None:
    bad = [muse_job(2, refs={}), muse_job(3, name=""), muse_job(4, refs={"landing_page": "x"}),
           "junk", None, 5, {"id": 1, "locations": [{"name": "Remote"}]}]

    parsed = TheMuse().parse_page(muse_page([muse_job(1), *bad]))

    assert [p.source_id for p in parsed.postings] == ["1001"]
    assert parsed.skipped == 7 and parsed.raw_count == 8


@pytest.mark.parametrize("page", [None, [], "x", 3, {}, {"results": None}, {"results": {}}])
def test_the_muse_refuses_something_that_is_not_a_feed_page(page) -> None:
    with pytest.raises(FeedError) as caught:
        TheMuse().parse_page(page)

    assert caught.value.code == "invalid_response"


# --- Arbeitnow --------------------------------------------------------------------------


def test_arbeitnow_first_request_is_the_plain_feed_url() -> None:
    request = Arbeitnow().next_request([])

    assert (request.method, request.url, request.body) == ("GET", arbeitnow.API_URL, None)
    assert request.url == "https://www.arbeitnow.com/api/job-board-api"
    assert Arbeitnow().is_configured()


def test_arbeitnow_follows_its_next_link_only_on_its_own_host() -> None:
    own = "https://www.arbeitnow.com/api/job-board-api?page=2"
    adapter = Arbeitnow()

    assert adapter.next_request([arbeitnow_page([], next_url=own)]).url == own
    for elsewhere in ("https://evil.example/api?page=2", "http://www.arbeitnow.com/api?page=2",
                      "https://www.arbeitnow.com.evil.example/x",
                      "https://user@evil.example/www.arbeitnow.com", "//www.arbeitnow.com/x",
                      "/api/job-board-api?page=2", "", None, 5):
        assert adapter.next_request([arbeitnow_page([], next_url=elsewhere)]) is None
    assert adapter.next_request(["junk"]) is None
    assert adapter.next_request([{"links": "x"}]) is None


def test_arbeitnow_posting_fields_are_mapped() -> None:
    parsed = Arbeitnow().parse_page(arbeitnow_page([arbeitnow_job(1)]))

    (made,) = parsed.postings
    assert (made.source, made.source_id) == ("arbeitnow", "python-dev-1-berlin")
    assert made.url == "https://www.arbeitnow.com/jobs/companies/x/1"
    assert made.title == "Python Developer 1" and made.company == "Beispiel GmbH"
    assert made.location_raw == "Berlin" and made.remote is True
    assert made.tags == ["Python", "Backend"] and made.employment_type == "full time"
    assert made.posted_at == datetime.fromtimestamp(1_789_900_000, tz=timezone.utc)
    assert made.description == "<p>Wir suchen 1</p>" and not made.description_is_excerpt


def test_arbeitnow_drops_jobs_that_are_not_marked_remote() -> None:
    jobs = [arbeitnow_job(1), arbeitnow_job(2, remote=False), arbeitnow_job(3, remote=None),
            arbeitnow_job(4, remote="yes")]

    parsed = Arbeitnow().parse_page(arbeitnow_page(jobs))

    assert [p.source_id for p in parsed.postings] == ["python-dev-1-berlin"]
    assert parsed.filtered == 3


def test_arbeitnow_keeps_on_site_jobs_when_asked_and_says_what_it_knows() -> None:
    jobs = [arbeitnow_job(2, remote=False), arbeitnow_job(3, remote="maybe")]

    parsed = Arbeitnow(remote_only=False).parse_page(arbeitnow_page(jobs))

    assert [p.remote for p in parsed.postings] == [False, None]


def test_arbeitnow_skips_malformed_entries_and_tolerates_odd_optional_fields() -> None:
    odd = arbeitnow_job(5, tags="x", job_types=None, created_at="soon", location=3)
    jobs = [arbeitnow_job(1, slug=""), arbeitnow_job(2, url="/relative"),
            arbeitnow_job(3, title=""), {"remote": True}, "junk", odd]

    parsed = Arbeitnow().parse_page(arbeitnow_page(jobs))

    (made,) = parsed.postings
    assert parsed.skipped == 5
    assert (made.tags, made.employment_type, made.posted_at, made.location_raw) == (
        [], None, None, None)


@pytest.mark.parametrize("page", [None, [], "x", {}, {"data": None}, {"data": {}}])
def test_arbeitnow_refuses_something_that_is_not_a_feed_page(page) -> None:
    with pytest.raises(FeedError):
        Arbeitnow().parse_page(page)


# --- Adzuna -----------------------------------------------------------------------------


def adz(**changes) -> Adzuna:
    return Adzuna(**{"app_id": APP_ID, "app_key": APP_KEY, **changes})


def test_adzuna_needs_both_credentials() -> None:
    assert adz().is_configured()
    assert not adz(app_id=None).is_configured()
    assert not adz(app_key="").is_configured()
    assert not Adzuna.from_environment({}).is_configured()
    assert Adzuna.from_environment({"ADZUNA_APP_ID": "a", "ADZUNA_APP_KEY": "b"}).is_configured()


def test_adzuna_first_request_targets_the_country_search_with_credentials() -> None:
    request = adz().next_request([])
    parts = urlsplit(request.url)

    assert request.method == "GET"
    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == \
        "https://api.adzuna.com/v1/api/jobs/gb/search/1"
    assert query(request) == {"app_id": APP_ID, "app_key": APP_KEY, "what": "remote",
                              "results_per_page": "50", "content-type": "application/json"}


def test_adzuna_country_and_query_come_from_the_environment() -> None:
    adapter = Adzuna.from_environment({"ADZUNA_APP_ID": "a", "ADZUNA_APP_KEY": "b",
                                       "ADZUNA_COUNTRY": "DE", "ADZUNA_QUERY": "python remote"})
    request = adapter.next_request([])

    assert urlsplit(request.url).path == "/v1/api/jobs/de/search/1"
    assert query(request)["what"] == "python remote"


@pytest.mark.parametrize("country", ["", "g", "gbr", "g/", "../x", "gb/search/1?x=", "g b", "1b"])
def test_adzuna_refuses_a_country_that_could_change_the_url(country) -> None:
    with pytest.raises(ValueError):
        adz(country=country)
    if country:  # an empty value means "not set": the default applies
        with pytest.raises(ValueError):
            Adzuna.from_environment({"ADZUNA_COUNTRY": country})


def test_adzuna_pages_until_the_reported_count_is_covered() -> None:
    adapter = adz()
    page = adzuna_page([adzuna_job(1)], count=120)  # 3 pages of 50

    assert urlsplit(adapter.next_request([page]).url).path.endswith("/search/2")
    assert urlsplit(adapter.next_request([page, page]).url).path.endswith("/search/3")
    assert adapter.next_request([page, page, page]) is None


def test_adzuna_stops_at_an_empty_page_and_at_exactly_full_counts() -> None:
    adapter = adz()

    assert adapter.next_request([adzuna_page([], count=500)]) is None
    assert adapter.next_request([{"count": 50, "results": [adzuna_job(1)]}]) is None
    assert adapter.next_request([{"count": 51, "results": [adzuna_job(1)]}]) is not None
    assert adapter.next_request(["junk"]) is None
    assert adapter.next_request([{"results": [adzuna_job(1)]}]) is not None  # no count: go on


def test_adzuna_posting_fields_are_mapped_and_the_description_is_an_excerpt() -> None:
    parsed = adz().parse_page(adzuna_page([adzuna_job(1)]))

    (made,) = parsed.postings
    assert (made.source, made.source_id) == ("adzuna", "481")
    assert made.url == "https://www.adzuna.co.uk/jobs/details/481?se=x"
    assert made.title == "Backend Engineer 1" and made.company == "Globex"
    assert made.location_raw == "London" and made.remote is None
    assert made.category == "IT Jobs" and made.employment_type == "full_time"
    assert made.posted_at == datetime(2026, 9, 19, 8, 30, tzinfo=timezone.utc)
    assert made.description == "Snippet of the ad…" and made.description_is_excerpt
    assert made.salary_raw == "50,000 - 60,000"


@pytest.mark.parametrize(("low", "high", "predicted", "expected"), [
    (50000.0, 50000.0, "0", "50,000"), (None, 60000, "0", "60,000"),
    (50000, None, "1", "50,000 (estimated)"), (None, None, "0", None),
    ("x", True, "0", None), (1234567.4, 2e6, "1", "1,234,567 - 2,000,000 (estimated)"),
])
def test_adzuna_salary_is_text_and_flags_estimates(low, high, predicted, expected) -> None:
    job = adzuna_job(1, salary_min=low, salary_max=high, salary_is_predicted=predicted)

    (made,) = adz().parse_page(adzuna_page([job])).postings

    assert made.salary_raw == expected


def test_adzuna_skips_malformed_entries_and_accepts_missing_optional_parts() -> None:
    bare = {"id": 9, "title": "Bare", "redirect_url": "https://www.adzuna.co.uk/9"}
    jobs = [adzuna_job(1, redirect_url="nope"), adzuna_job(2, title=" "), {"title": "x"}, 4, bare]

    parsed = adz().parse_page(adzuna_page(jobs))

    (made,) = parsed.postings
    assert parsed.skipped == 4 and made.source_id == "9"
    assert (made.company, made.location_raw, made.category, made.posted_at) == (None,) * 4


@pytest.mark.parametrize("page", [None, [], "x", {}, {"results": "x"}])
def test_adzuna_refuses_something_that_is_not_a_feed_page(page) -> None:
    with pytest.raises(FeedError):
        adz().parse_page(page)


# --- Jooble -----------------------------------------------------------------------------


def joo(**changes) -> Jooble:
    return Jooble(**{"api_key": KEY, **changes})


def test_jooble_needs_its_key() -> None:
    assert joo().is_configured()
    assert not joo(api_key=None).is_configured() and not joo(api_key="").is_configured()
    assert not Jooble.from_environment({}).is_configured()
    assert Jooble.from_environment({"JOOBLE_API_KEY": KEY}).api_key == KEY


def test_jooble_first_request_posts_json_to_the_key_url() -> None:
    request = joo().next_request([])

    assert request.method == "POST" and request.url == f"https://jooble.org/api/{KEY}"
    assert request.headers == {"Content-Type": "application/json"}
    assert json.loads(request.body) == {"keywords": "remote", "location": "", "page": "1"}
    assert KEY not in repr(request)


def test_jooble_searches_come_from_the_environment() -> None:
    adapter = Jooble.from_environment({"JOOBLE_API_KEY": KEY, "JOOBLE_KEYWORDS": "python",
                                       "JOOBLE_LOCATION": "Kyiv"})

    assert json.loads(adapter.next_request([]).body) == {
        "keywords": "python", "location": "Kyiv", "page": "1"}


def test_jooble_keeps_a_key_with_odd_characters_inside_the_url_path() -> None:
    request = joo(api_key="a/b?c#d e").next_request([])

    assert request.url == "https://jooble.org/api/a%2Fb%3Fc%23d%20e"
    assert urlsplit(request.url).netloc == "jooble.org"


def test_jooble_pages_until_the_total_is_covered() -> None:
    adapter = joo()
    page = jooble_page([jooble_job(1), jooble_job(2)], total=5)

    assert json.loads(adapter.next_request([page]).body)["page"] == "2"
    assert json.loads(adapter.next_request([page, page]).body)["page"] == "3"
    assert adapter.next_request([page, page, page]) is None  # 6 seen >= 5


def test_jooble_stops_on_an_empty_page_and_goes_on_without_a_total() -> None:
    adapter = joo()

    assert adapter.next_request([jooble_page([], total=50)]) is None
    assert adapter.next_request(["junk"]) is None
    assert adapter.next_request([{"jobs": [jooble_job(1)]}]) is not None
    assert adapter.next_request([jooble_page([jooble_job(1)], total=2)]) is not None
    assert adapter.next_request([jooble_page([jooble_job(1)], total=1)]) is None


def test_jooble_posting_fields_are_mapped_and_the_snippet_is_an_excerpt() -> None:
    parsed = joo().parse_page(jooble_page([jooble_job(1, salary="$5k")]))

    (made,) = parsed.postings
    assert (made.source, made.source_id) == ("jooble", "5001")
    assert made.url == "https://jooble.org/desc/100" and made.title == "ML Engineer 1"
    assert made.company == "Initech" and made.location_raw == "Remote"
    assert made.employment_type == "Full-time" and made.salary_raw == "$5k"
    assert made.posted_at == datetime(2026, 9, 18, tzinfo=timezone.utc)
    assert made.description == "We are looking for…" and made.description_is_excerpt
    assert made.remote is None


def test_jooble_without_an_id_derives_a_stable_one_from_the_link() -> None:
    without = jooble_job(1)
    del without["id"]
    blank = jooble_job(2, id="  ")

    a = joo().parse_page(jooble_page([without])).postings[0]
    b = joo().parse_page(jooble_page([without])).postings[0]
    c = joo().parse_page(jooble_page([blank])).postings[0]

    assert a.source_id == b.source_id == stable_id(without["link"]) != c.source_id
    assert c.source_id == stable_id(blank["link"])


def test_jooble_skips_malformed_entries() -> None:
    jobs = [jooble_job(1, link="nope"), jooble_job(2, title=""), {"title": "x"}, "junk",
            jooble_job(3, salary="")]

    parsed = joo().parse_page(jooble_page(jobs))

    (made,) = parsed.postings
    assert parsed.skipped == 4 and made.salary_raw is None


@pytest.mark.parametrize("page", [None, [], "x", {}, {"jobs": "x"}])
def test_jooble_refuses_something_that_is_not_a_feed_page(page) -> None:
    with pytest.raises(FeedError):
        joo().parse_page(page)


# --- registry ---------------------------------------------------------------------------


def test_every_source_has_an_adapter_named_after_it() -> None:
    env = {"JOOBLE_API_KEY": KEY, "ADZUNA_APP_ID": "a", "ADZUNA_APP_KEY": "b"}

    assert SOURCE_NAMES == ("themuse", "arbeitnow", "adzuna", "jooble")
    for name in SOURCE_NAMES:
        adapter = build_adapter(name, env)
        assert adapter.name == name and adapter.is_configured()
        assert adapter.next_request([]) is not None


def test_unconfigured_sources_report_it_and_unknown_names_are_refused() -> None:
    assert not build_adapter("adzuna", {}).is_configured()
    assert not build_adapter("jooble", {}).is_configured()
    assert build_adapter("themuse", {}).is_configured()
    assert build_adapter("arbeitnow", {}).is_configured()
    with pytest.raises(ValueError, match="unknown source"):
        build_adapter("careerjet", {})


def test_the_module_constants_name_their_sources() -> None:
    assert (themuse.NAME, arbeitnow.NAME, adzuna.NAME, jooble.NAME) == (
        "themuse", "arbeitnow", "adzuna", "jooble")


# --- edges the mutation checks asked for -------------------------------------------------


def test_the_muse_treats_flexible_alone_as_remote_and_ignores_blank_names() -> None:
    job = muse_job(1, locations=[{"name": "Flexible"}, {"name": "  "}, {"name": 5}, {}],
                   categories=[{"name": "  "}, {"name": "Design"}], levels="x")

    (made,) = TheMuse().parse_page(muse_page([job])).postings

    assert made.remote is True and made.location_raw == "Flexible"
    assert made.category == "Design" and made.tags == []


def test_text_fields_are_stripped() -> None:
    job = adzuna_job(1, company={"display_name": "  Globex  "}, description="  hi  ")

    (made,) = adz().parse_page(adzuna_page([job])).postings

    assert made.company == "Globex" and made.description == "hi"


def test_the_endpoints_are_the_documented_ones() -> None:
    assert themuse.API_URL == "https://www.themuse.com/api/public/jobs"
    assert adzuna.API_BASE == "https://api.adzuna.com/v1/api/jobs"
    assert jooble.API_BASE == "https://jooble.org/api/"
    assert arbeitnow.HOST == "www.arbeitnow.com"


def test_the_registry_builds_arbeitnow_as_remote_only() -> None:
    adapter = build_adapter("arbeitnow", {})

    parsed = adapter.parse_page(arbeitnow_page([arbeitnow_job(1, remote=False)]))

    assert parsed.postings == [] and parsed.filtered == 1


def test_an_empty_remotive_salary_becomes_none() -> None:
    from jobpdf.ingestion.remotive import RemotivePosting

    made = to_source_posting(RemotivePosting(id=1, url="https://remotive.com/1", title="T",
                                             salary=""))

    assert made.salary_raw is None


def test_an_empty_adzuna_country_setting_means_the_default() -> None:
    assert Adzuna.from_environment({"ADZUNA_COUNTRY": " "}).country == "gb"
