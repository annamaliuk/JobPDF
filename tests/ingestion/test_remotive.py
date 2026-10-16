import json
import logging
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from jobpdf.ingestion import remotive
from jobpdf.ingestion.remotive import (
    API_URL,
    CACHE_KEEP,
    MAX_CACHE_AGE_S,
    MAX_REQUESTS_PER_DAY,
    MIN_INTERVAL_S,
    FeedError,
    FetchError,
    cache_dir,
    fetch_postings,
    parse_feed,
    run_fetch,
)

T0 = 1_790_000_000.0  # a fixed "now" (2026-09-21)
HOUR = 3600.0


def job(i: int = 1, **changes) -> dict:
    data = {
        "id": i, "url": f"https://remotive.com/remote-jobs/software-dev/role-{i}",
        "title": f"Data Engineer {i}", "company_name": "Acme Widgets", "category": "Data",
        "tags": ["python", "sql"], "job_type": "full_time",
        "publication_date": "2026-09-20T08:55:12", "candidate_required_location": "Europe",
        "salary": "$80k-$100k", "description": "<p>Build <b>pipelines</b>.</p>",
    }
    data.update(changes)
    return data


def feed(*jobs: dict, count: object = "auto", **top) -> bytes:
    jobs = jobs or (job(1), job(2))
    body = {"0-legal-notice": "Remotive legal notice", "job-count": len(jobs) if count == "auto"
            else count, "jobs": list(jobs), **top}
    return json.dumps(body).encode("utf-8")


class Clock:
    def __init__(self, now: float = T0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class Transport:
    """Scripted responses (a tuple, or an exception to raise); records every call."""

    def __init__(self, *responses) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, float]] = []

    def __call__(self, url: str, timeout: float):
        self.calls.append((url, timeout))
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(response, Exception):
            raise response
        return response


def ok(body: bytes | None = None) -> tuple[int, bytes]:
    return 200, feed() if body is None else body


@pytest.fixture
def base(tmp_path) -> Path:
    return tmp_path / "base"


def files(base: Path) -> list[str]:
    folder = base / "remotive"
    return sorted(p.name for p in folder.iterdir()) if folder.exists() else []


def cached(base: Path) -> list[str]:
    return [n for n in files(base) if n != "requests.json"]


def get(base, transport, clock, **kwargs):
    return fetch_postings(base=base, transport=transport, clock=clock, **kwargs)


# --- parsing -----------------------------------------------------------------------


def test_a_feed_becomes_typed_postings() -> None:
    parsed = parse_feed(feed(job(1), job(2, tags=None, job_type=None, salary=None)))

    first, second = parsed.postings
    assert (first.id, first.title, first.company_name) == (1, "Data Engineer 1", "Acme Widgets")
    assert first.tags == ["python", "sql"] and second.tags == []
    assert first.description == "<p>Build <b>pipelines</b>.</p>"  # kept as received
    assert (parsed.skipped, parsed.reported_count) == (0, 2)
    assert parsed.legal_notice == "Remotive legal notice"


def test_naive_publication_dates_are_utc() -> None:
    date = parse_feed(feed(job())).postings[0].publication_date

    assert date == datetime(2026, 9, 20, 8, 55, 12, tzinfo=timezone.utc)


def test_unknown_fields_and_missing_optional_ones_are_fine() -> None:
    minimal = {"id": 7, "url": "https://remotive.com/x", "title": "T", "new_field": [1]}

    posting = parse_feed(feed(minimal)).postings[0]

    assert posting.id == 7 and posting.company_name is None and posting.publication_date is None


def test_an_id_sent_as_text_is_still_an_id() -> None:
    assert parse_feed(feed(job("42"))).postings[0].id == 42


@pytest.mark.parametrize("bad", [
    job(2, title=""), job(2, title="   "), job(2, url="javascript:alert(1)"), job(2, url=""),
    {"id": 2, "title": "no url"}, {"url": "https://x.y", "title": "no id"}, "a string", None,
    job(2, id="abc"), job(2, publication_date="not a date"), job(2, tags="python"),
])
def test_one_malformed_posting_is_skipped_and_counted(bad) -> None:
    parsed = parse_feed(feed(job(1), bad, job(3)))

    assert [p.id for p in parsed.postings] == [1, 3] and parsed.skipped == 1


def test_the_first_of_two_postings_with_one_id_wins() -> None:
    parsed = parse_feed(feed(job(1, title="first"), job(1, title="second")))

    assert [p.title for p in parsed.postings] == ["first"] and parsed.skipped == 1


@pytest.mark.parametrize(("body", "code"), [
    (b"not json", "invalid_response"),
    (b"\xff\xfe", "invalid_response"),
    (b"[]", "invalid_response"),
    (b'{"jobs": "x"}', "invalid_response"),
    (b'{"no-jobs": 1}', "invalid_response"),
    (b'{"jobs": []}', "no_postings"),
    (feed({"id": 1}, {"id": 2}), "invalid_response"),  # every posting malformed
])
def test_an_unusable_feed_has_a_code(body, code) -> None:
    with pytest.raises(FeedError) as info:
        parse_feed(body)

    assert info.value.code == code


@pytest.mark.parametrize("count", [True, "5", None, 1.5])
def test_a_nonsense_job_count_is_ignored(count) -> None:
    assert parse_feed(feed(job(), count=count)).reported_count is None


def test_a_posting_cannot_be_changed() -> None:
    posting = parse_feed(feed(job())).postings[0]

    with pytest.raises(ValidationError):
        posting.title = "other"


# --- the happy path and the cache ----------------------------------------------------


def test_the_first_fetch_makes_one_request_and_stores_the_response_as_received(base) -> None:
    body = feed(job(1), job(2))
    transport = Transport(ok(body))

    result = get(base, transport, Clock())

    assert transport.calls == [(API_URL, remotive.REQUEST_TIMEOUT_S)]
    assert [p.id for p in result.postings] == [1, 2]
    assert (result.from_cache, result.stale, result.warnings) == (False, False, [])
    assert result.fetched_at == datetime.fromtimestamp(T0, tz=timezone.utc)
    stored = base / "remotive" / "20260921T141320Z.json"
    assert cached(base) == [stored.name] and stored.read_bytes() == body


def test_within_the_cache_age_no_request_is_made(base) -> None:
    clock, transport = Clock(), Transport(ok())
    first = get(base, transport, clock)

    clock.now += MAX_CACHE_AGE_S - 1
    second = get(base, transport, clock)

    assert len(transport.calls) == 1
    assert second.from_cache and not second.stale
    assert second.fetched_at == first.fetched_at and second.postings == first.postings


def test_the_cache_age_limit_is_inclusive() -> None:
    assert MAX_CACHE_AGE_S == 6 * HOUR


def test_at_exactly_the_cache_age_the_cache_is_still_served(base) -> None:
    clock, transport = Clock(), Transport(ok())
    get(base, transport, clock)

    clock.now += MAX_CACHE_AGE_S
    assert get(base, transport, clock).from_cache
    clock.now += 1
    assert not get(base, transport, clock).from_cache
    assert len(transport.calls) == 2


def test_a_cache_can_be_served_from_a_new_process(base) -> None:
    clock = Clock()
    get(base, Transport(ok()), clock)

    never = Transport(AssertionError("must not be called"))
    result = get(base, never, Clock(T0 + HOUR))

    assert result.from_cache and never.calls == []


def test_stale_means_older_than_the_limit_only_when_a_refresh_failed(base) -> None:
    clock = Clock()
    get(base, Transport(ok()), clock)
    clock.now += 2 * MAX_CACHE_AGE_S

    result = get(base, Transport(OSError("down")), clock)

    assert result.from_cache and result.stale


# --- failures ----------------------------------------------------------------------

FAILURES = [
    (OSError("connection refused"), "network_error"),
    (urllib.error.URLError("dns"), "network_error"),
    (TimeoutError(), "network_error"),
    ((429, b""), "http_error"),
    ((500, b"oops"), "http_error"),
    ((403, b""), "http_error"),
    ((200, b"<html>maintenance</html>"), "invalid_response"),
    ((200, b'{"jobs": []}'), "no_postings"),
    ((200, b"x" * (remotive.MAX_RESPONSE_BYTES + 1)), "response_too_large"),
]


@pytest.mark.parametrize(("failure", "code"), FAILURES)
def test_a_failed_fetch_serves_the_newest_cache_as_stale(base, failure, code) -> None:
    clock = Clock()
    first = get(base, Transport(ok()), clock)
    clock.now += MAX_CACHE_AGE_S + MIN_INTERVAL_S

    result = get(base, Transport(failure), clock)

    assert result.stale and result.from_cache
    assert result.warnings == ["stale_cache", code]
    assert result.fetched_at == first.fetched_at and result.postings == first.postings


@pytest.mark.parametrize(("failure", "code"), FAILURES)
def test_a_failed_fetch_without_a_cache_is_an_error_with_a_code(base, failure, code) -> None:
    with pytest.raises(FetchError) as info:
        get(base, Transport(failure), Clock())

    assert info.value.code == code and cached(base) == []


@pytest.mark.parametrize(("failure", "code"), FAILURES[3:])
def test_a_bad_response_never_replaces_a_good_cache(base, failure, code) -> None:
    clock = Clock()
    get(base, Transport(ok()), clock)
    before = {n: (base / "remotive" / n).read_bytes() for n in cached(base)}
    clock.now += MAX_CACHE_AGE_S + MIN_INTERVAL_S

    get(base, Transport(failure), clock)

    assert {n: (base / "remotive" / n).read_bytes() for n in cached(base)} == before


def test_there_are_no_retries(base) -> None:
    transport = Transport(OSError("down"))

    with pytest.raises(FetchError):
        get(base, transport, Clock())

    assert len(transport.calls) == 1


def test_a_failed_request_still_counts_against_the_limit(base) -> None:
    clock = Clock()
    for _ in range(MAX_REQUESTS_PER_DAY):
        with pytest.raises(FetchError):
            get(base, Transport(OSError("down")), clock)
        clock.now += MIN_INTERVAL_S

    never = Transport(AssertionError("must not be called"))
    with pytest.raises(FetchError) as info:
        get(base, never, clock)

    assert info.value.code == "quota_exhausted" and never.calls == []


def test_the_request_is_recorded_before_it_is_sent(base) -> None:
    seen: list[list[float]] = []

    def spy(url, timeout):
        seen.append(json.loads((base / "remotive" / "requests.json").read_text()))
        return ok()

    get(base, spy, Clock())

    assert seen == [[T0]]


# --- the guard ---------------------------------------------------------------------


def test_four_requests_a_day_and_no_more(base) -> None:
    clock = Clock()
    transport = Transport(ok())
    for _ in range(MAX_REQUESTS_PER_DAY):
        get(base, transport, clock, max_age_s=0)
        clock.now += HOUR  # an hour apart: only the daily cap can stop the next one

    blocked = get(base, transport, clock, max_age_s=0)

    assert len(transport.calls) == MAX_REQUESTS_PER_DAY
    assert blocked.stale and blocked.warnings == ["stale_cache", "quota_exhausted"]


def test_the_oldest_request_leaves_the_window_after_24_hours(base) -> None:
    clock = Clock()
    transport = Transport(ok())
    for _ in range(MAX_REQUESTS_PER_DAY):
        get(base, transport, clock, max_age_s=0)
        clock.now += HOUR
    clock.now = T0 + 24 * HOUR - 1  # the first request is still inside the window
    assert get(base, transport, clock, max_age_s=0).stale

    clock.now = T0 + 24 * HOUR  # now it is not
    assert not get(base, transport, clock, max_age_s=0).stale
    assert len(transport.calls) == MAX_REQUESTS_PER_DAY + 1


def test_two_requests_cannot_be_closer_than_the_minimum_interval(base) -> None:
    clock = Clock()
    transport = Transport(ok())
    get(base, transport, clock, max_age_s=0)

    clock.now += MIN_INTERVAL_S - 1
    soon = get(base, transport, clock, max_age_s=0)
    clock.now += 1
    later = get(base, transport, clock, max_age_s=0)

    assert soon.warnings == ["stale_cache", "too_soon"]
    assert not later.stale and len(transport.calls) == 2


def test_the_guard_limits_are_remotives_advice() -> None:
    assert (MAX_REQUESTS_PER_DAY, MIN_INTERVAL_S) == (4, 30.0)


@pytest.mark.parametrize("content", ["not json", '{"a": 1}', '["x"]', "[null]", ""])
def test_an_unreadable_ledger_is_treated_as_a_request_just_made(base, content) -> None:
    clock = Clock()
    get(base, Transport(ok()), clock)
    (base / "remotive" / "requests.json").write_text(content, encoding="utf-8")
    clock.now += MAX_CACHE_AGE_S + 1
    transport = Transport(AssertionError("must not be called"))

    result = get(base, transport, clock)

    assert result.stale and "too_soon" in result.warnings and transport.calls == []


def test_without_a_ledger_a_request_is_allowed(base) -> None:
    assert not get(base, Transport(ok()), Clock()).stale


# --- the cache folder --------------------------------------------------------------


def test_the_newest_cache_wins_and_unusable_files_are_skipped(base) -> None:
    folder = base / "remotive"
    folder.mkdir(parents=True)
    (folder / "20260920T000000Z.json").write_bytes(feed(job(1, title="old")))
    (folder / "20260921T000000Z.json").write_bytes(b"corrupt")
    (folder / "notes.json").write_bytes(feed(job(9)))
    (folder / "20260922T000000Z.json.tmp").write_bytes(feed(job(9)))

    result = get(base, Transport(OSError("down")), Clock())

    assert [p.title for p in result.postings] == ["old"] and result.stale


def test_only_the_newest_responses_are_kept(base) -> None:
    clock = Clock()
    transport = Transport(ok())
    times = []
    for _ in range(CACHE_KEEP + 3):
        times.append(clock.now)
        get(base, transport, clock, max_age_s=0)
        clock.now += HOUR
        (base / "remotive" / "requests.json").unlink()  # keep the guard out of the way

    stamp = [datetime.fromtimestamp(t, tz=timezone.utc).strftime(remotive.CACHE_STAMP)
             for t in times[-CACHE_KEEP:]]
    assert cached(base) == [f"{s}.json" for s in stamp]
    assert "requests.json" not in cached(base)


def test_no_temporary_files_are_left_behind(base) -> None:
    get(base, Transport(ok()), Clock())

    assert not [n for n in files(base) if n.endswith(".tmp")]


def test_the_default_folder_is_data_cache_remotive_or_the_environment_override(monkeypatch) -> None:
    monkeypatch.delenv("JOBPDF_CACHE_DIR")
    assert cache_dir() == remotive.REPO_ROOT / "data" / "cache" / "remotive"
    monkeypatch.setenv("JOBPDF_CACHE_DIR", "/somewhere")
    assert cache_dir() == Path("/somewhere/remotive")
    assert cache_dir(Path("/explicit")) == Path("/explicit/remotive")


def test_the_default_cache_folder_is_git_ignored() -> None:
    ignore = (remotive.REPO_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "data/cache/" in ignore.splitlines()


# --- what the result says ------------------------------------------------------------


def test_skipped_postings_are_reported_as_a_warning_in_every_case(base) -> None:
    clock = Clock()
    body = feed(job(1), {"id": 2})

    fresh = get(base, Transport(ok(body)), clock)
    again = get(base, Transport(ok(body)), clock)

    assert fresh.skipped == again.skipped == 1
    assert fresh.warnings == again.warnings == ["skipped_postings"]


def test_the_legal_notice_and_urls_travel_with_the_postings(base) -> None:
    result = get(base, Transport(ok()), Clock())

    assert result.legal_notice == "Remotive legal notice"
    assert all(p.url.startswith("https://remotive.com/") for p in result.postings)
    assert remotive.SOURCE == "remotive"


def test_logs_hold_counts_and_codes_never_posting_text(base, caplog) -> None:
    secret = job(1, title="Quantum Llama Wrangler", description="TOPSECRETDESCRIPTION")
    clock = Clock()
    with caplog.at_level(logging.DEBUG):
        get(base, Transport(ok(feed(secret, {"id": 2}))), clock)
        clock.now += MAX_CACHE_AGE_S + MIN_INTERVAL_S
        get(base, Transport((500, b"")), clock)
        get(base, Transport(OSError("secret.example.org refused")), Clock(clock.now + 100))

    assert "fetched 1 Remotive postings (1 skipped)" in caplog.text
    for text in ("Quantum", "TOPSECRET", "remotive.com/remote-jobs", "secret.example"):
        assert text not in caplog.text


# --- the real transport (urlopen replaced: no sockets) -------------------------------


class FakeResponse:
    def __init__(self, status: int, body: bytes) -> None:
        self.status, self.body = status, body

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def read(self, n: int = -1) -> bytes:
        return self.body if n < 0 else self.body[:n]


def test_the_endpoint_is_remotives_job_feed() -> None:
    assert API_URL == "https://remotive.com/api/remote-jobs"


def test_the_default_transport_sends_a_polite_request_and_returns_status_and_body(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"], seen["timeout"] = request.full_url, timeout
        seen["headers"] = {k.lower(): v for k, v in request.header_items()}
        return FakeResponse(200, b"BODY")

    monkeypatch.setattr(remotive.urllib.request, "urlopen", fake_urlopen)

    assert remotive._urllib_transport("https://example.test/x", 7.0) == (200, b"BODY")
    assert seen["url"] == "https://example.test/x" and seen["timeout"] == 7.0
    assert seen["headers"]["user-agent"].startswith("JobPDF/")
    assert "json" in seen["headers"]["accept"]


def test_the_default_transport_turns_an_http_error_into_a_status(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 429, "Too Many", {}, None)

    monkeypatch.setattr(remotive.urllib.request, "urlopen", fake_urlopen)

    assert remotive._urllib_transport("https://example.test/x", 1.0) == (429, b"")


def test_the_default_transport_never_reads_more_than_the_size_limit(monkeypatch) -> None:
    huge = FakeResponse(200, b"x" * (remotive.MAX_RESPONSE_BYTES + 10))
    monkeypatch.setattr(remotive.urllib.request, "urlopen", lambda request, timeout: huge)

    status, body = remotive._urllib_transport("https://example.test/x", 1.0)

    assert len(body) == remotive.MAX_RESPONSE_BYTES + 1


def test_connection_errors_pass_through_to_the_caller(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(remotive.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(urllib.error.URLError):
        remotive._urllib_transport("https://example.test/x", 1.0)


def test_fetch_uses_the_real_transport_by_default(base, monkeypatch) -> None:
    monkeypatch.setattr(remotive, "_urllib_transport", Transport(ok()))

    assert fetch_postings(base=base, clock=Clock()).postings


# --- the Airflow task and the script -------------------------------------------------


def test_the_task_reports_counts_never_postings(monkeypatch) -> None:
    transport = Transport(ok(feed(job(1), job(2), {"id": 3})))
    monkeypatch.setattr(remotive, "_urllib_transport", transport)
    monkeypatch.setattr(remotive.time, "time", Clock())

    summary = run_fetch()

    assert summary == {
        "source": "remotive", "count": 2, "skipped": 1, "from_cache": False, "stale": False,
        "fetched_at": "2026-09-21T14:13:20+00:00", "warnings": ["skipped_postings"],
    }
    json.dumps(summary)


def test_the_scheduled_task_refreshes_data_older_than_its_limit(monkeypatch) -> None:
    clock = Clock()
    transport = Transport(ok())
    monkeypatch.setattr(remotive, "_urllib_transport", transport)
    monkeypatch.setattr(remotive.time, "time", clock)
    run_fetch()

    clock.now += remotive.SCHEDULED_MAX_AGE_S - 1
    assert run_fetch()["from_cache"] is True
    clock.now += 2
    assert run_fetch()["from_cache"] is False
    assert len(transport.calls) == 2


def test_the_scheduled_age_limit_is_shorter_than_the_schedule_interval() -> None:
    assert remotive.SCHEDULED_MAX_AGE_S < 8 * HOUR


def test_the_task_fails_loudly_when_there_is_nothing_to_serve(monkeypatch) -> None:
    monkeypatch.setattr(remotive, "_urllib_transport", Transport(OSError("down")))

    with pytest.raises(FetchError):
        run_fetch()


def _script():
    import importlib.util

    path = Path(__file__).parents[2] / "scripts" / "fetch_remotive.py"
    spec = importlib.util.spec_from_file_location("fetch_remotive", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_script_prints_counts_only(monkeypatch, capsys) -> None:
    monkeypatch.setattr(remotive, "_urllib_transport", Transport(ok(feed(job(1, title="Zorp")))))
    script = _script()

    assert script.main() == 0
    out = capsys.readouterr().out

    assert "1 postings from live request" in out and "Zorp" not in out


def test_the_script_says_so_when_there_is_nothing(monkeypatch, capsys) -> None:
    monkeypatch.setattr(remotive, "_urllib_transport", Transport(OSError("down")))

    assert _script().main() == 1
    assert "network_error" in capsys.readouterr().out
