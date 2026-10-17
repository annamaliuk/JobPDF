import json
import logging
import urllib.error
import urllib.request

import pytest

from jobpdf.ingestion.sources import runner
from jobpdf.ingestion.sources.adzuna import Adzuna
from jobpdf.ingestion.sources.arbeitnow import Arbeitnow
from jobpdf.ingestion.sources.jooble import Jooble
from jobpdf.ingestion.sources.models import FetchError, Policy, Request
from jobpdf.ingestion.sources.runner import (
    SCHEDULED_MAX_AGE_S,
    cache_dir,
    fetch_source,
    run_source,
    urllib_transport,
)
from jobpdf.ingestion.sources.themuse import TheMuse
from tests.ingestion.source_fakes import (
    HOUR,
    KEY,
    T0,
    Clock,
    Transport,
    adzuna_job,
    adzuna_page,
    arbeitnow_job,
    arbeitnow_page,
    fast,
    jooble_job,
    jooble_page,
    muse_job,
    muse_page,
    no_sleep,
    ok,
)


@pytest.fixture
def base(tmp_path):
    return tmp_path / "cache"


def fetch(adapter, transport, base, clock=None, **kwargs):
    return fetch_source(adapter, base=base, transport=transport, clock=clock or Clock(),
                        sleep=kwargs.pop("sleep", no_sleep), **kwargs)


def muse(**changes) -> TheMuse:
    return TheMuse(policy=fast(TheMuse().policy, **changes))


def muse_pages(n: int, per_page: int = 2):
    """n Muse pages with distinct ids."""
    return [ok(muse_page([muse_job(p * 10 + i) for i in range(per_page)], page=p + 1,
                         page_count=n)) for p in range(n)]


def ledger(base, name="themuse") -> list[float]:
    return json.loads((base / name / "requests.json").read_text())


def cache_files(base, name="themuse"):
    return sorted(p.name for p in (base / name).glob("2*.json"))


# --- the happy path -----------------------------------------------------------------------


def test_a_live_run_returns_postings_and_stores_the_responses_as_received(base) -> None:
    transport = Transport(*muse_pages(1))

    result = fetch(muse(), transport, base)

    assert result.source == "themuse" and len(result.postings) == 2
    assert (result.from_cache, result.stale, result.pages, result.warnings) == (
        False, False, 1, [])
    assert (result.skipped, result.filtered) == (0, 0)
    assert result.fetched_at.timestamp() == T0
    (name,) = cache_files(base)
    stored = json.loads((base / "themuse" / name).read_text())
    assert stored["complete"] is True and stored["pages"][0]["results"][0]["id"] == 1000


def test_a_fresh_cache_is_served_without_any_request(base) -> None:
    clock = Clock()
    fetch(muse(), Transport(*muse_pages(1)), base, clock)
    clock.advance(5 * HOUR)
    silent = Transport(ok({}))

    result = fetch(muse(), silent, base, clock)

    assert silent.calls == 0 and result.from_cache and not result.stale
    assert len(result.postings) == 2 and result.warnings == []


def test_an_old_cache_is_refreshed_with_a_live_run(base) -> None:
    clock = Clock()
    fetch(muse(), Transport(*muse_pages(1)), base, clock)
    clock.advance(6 * HOUR + 1)
    second = Transport(ok(muse_page([muse_job(9)])))

    result = fetch(muse(), second, base, clock)

    assert second.calls == 1 and not result.from_cache
    assert [p.source_id for p in result.postings] == ["1009"]
    assert len(cache_files(base)) == 2


def test_the_cache_age_limit_is_inclusive_and_can_be_set_per_call(base) -> None:
    clock = Clock()
    fetch(muse(), Transport(*muse_pages(1)), base, clock)
    clock.advance(6 * HOUR)

    assert fetch(muse(), Transport(ok({})), base, clock).from_cache
    assert not fetch(muse(), Transport(ok(muse_page([muse_job(1)]))), base, clock,
                     max_age_s=HOUR).from_cache


def test_a_run_may_use_several_pages_with_a_pause_between_them(base) -> None:
    pauses = []
    transport = Transport(*muse_pages(3))

    result = fetch(muse(pause_s=2.5), transport, base, sleep=pauses.append)

    assert result.pages == 3 and len(result.postings) == 6 and result.warnings == []
    assert transport.calls == 3 and pauses == [2.5, 2.5]
    assert [q.url.split("page=")[1][0] for q in transport.requests] == ["1", "2", "3"]
    assert len(ledger(base)) == 3


def test_a_run_never_asks_for_more_pages_than_the_policy_allows(base) -> None:
    transport = Transport(*muse_pages(10))

    result = fetch(muse(max_pages=2), transport, base)

    assert transport.calls == 2 and result.pages == 2
    assert result.warnings == ["partial"]
    assert json.loads((base / "themuse" / cache_files(base)[0]).read_text())["complete"] is False


def test_a_run_that_ends_exactly_at_the_page_limit_is_complete(base) -> None:
    result = fetch(muse(max_pages=3), Transport(*muse_pages(3)), base)

    assert result.pages == 3 and result.warnings == []


def test_a_partial_cache_keeps_saying_so(base) -> None:
    clock = Clock()
    fetch(muse(max_pages=2), Transport(*muse_pages(10)), base, clock)

    again = fetch(muse(max_pages=2), Transport(ok({})), base, clock)

    assert again.from_cache and again.warnings == ["partial"]


def test_postings_repeated_across_pages_are_counted_once(base) -> None:
    pages = [ok(muse_page([muse_job(1), muse_job(2)], page=1, page_count=2)),
             ok(muse_page([muse_job(2), muse_job(3)], page=2, page_count=2))]

    result = fetch(muse(), Transport(*pages), base)

    assert [p.source_id for p in result.postings] == ["1001", "1002", "1003"]
    assert result.skipped == 1 and "skipped_postings" in result.warnings


def test_malformed_and_unwanted_postings_are_counted_not_fatal(base) -> None:
    page = muse_page([muse_job(1), muse_job(2, refs={}),
                      muse_job(3, locations=[{"name": "Paris"}])])

    result = fetch(muse(), Transport(ok(page)), base)

    assert [p.source_id for p in result.postings] == ["1001"]
    assert (result.skipped, result.filtered) == (1, 1)
    assert result.warnings == ["skipped_postings"]


# --- failures ------------------------------------------------------------------------------

FAILURES = [
    (OSError("boom"), "network_error"),
    (urllib.error.URLError("dns"), "network_error"),
    (TimeoutError(), "network_error"),
    ((500, b""), "http_error"),
    ((404, b"{}"), "http_error"),
    ((429, b"slow down"), "http_error"),
    ((200, b"<html>captcha</html>"), "invalid_response"),
    ((200, b"\xff\xfe"), "invalid_response"),
    (ok({"unexpected": 1}), "invalid_response"),
    (ok([1, 2]), "invalid_response"),
    (ok(muse_page([muse_job(1, refs={})])), "invalid_response"),
    (ok(muse_page([])), "no_postings"),
    (ok(muse_page([muse_job(1, locations=[{"name": "Paris"}])])), "no_postings"),
]


@pytest.mark.parametrize(("response", "code"), FAILURES)
def test_a_failed_first_request_with_no_cache_raises_its_code(base, response, code) -> None:
    transport = Transport(response)

    with pytest.raises(FetchError) as caught:
        fetch(muse(), transport, base)

    assert caught.value.code == code and transport.calls == 1  # no retry
    assert cache_files(base) == []  # a bad response is never stored


@pytest.mark.parametrize(("response", "code"), FAILURES)
def test_a_failed_request_serves_the_old_cache_marked_stale(base, response, code) -> None:
    clock = Clock()
    fetch(muse(), Transport(*muse_pages(1)), base, clock)
    clock.advance(10 * HOUR)
    transport = Transport(response)

    result = fetch(muse(), transport, base, clock)

    assert result.stale and result.from_cache and len(result.postings) == 2
    assert result.warnings == ["stale_cache", code] and transport.calls == 1
    assert len(cache_files(base)) == 1  # the good cache is untouched


def test_an_oversized_response_is_refused(base) -> None:
    transport = Transport((200, b"x" * 11))

    with pytest.raises(FetchError) as caught:
        fetch(muse(max_response_bytes=10), transport, base)

    assert caught.value.code == "response_too_large"


def test_a_source_without_its_key_is_skipped_before_any_request(base) -> None:
    transport = Transport(ok({}))

    with pytest.raises(FetchError) as caught:
        fetch(Adzuna(app_id="a", app_key=None), transport, base)

    assert caught.value.code == "not_configured" and transport.calls == 0
    assert not (base / "adzuna").exists()


def test_a_later_page_failing_keeps_the_pages_already_fetched(base) -> None:
    pages = muse_pages(3)
    transport = Transport(pages[0], (500, b""))

    result = fetch(muse(), transport, base)

    assert result.pages == 1 and len(result.postings) == 2
    assert result.warnings == ["partial", "http_error"] and not result.stale
    assert transport.calls == 2
    assert json.loads((base / "themuse" / cache_files(base)[0]).read_text())["complete"] is False


def test_a_later_page_that_is_not_a_feed_page_is_dropped_not_stored(base) -> None:
    transport = Transport(muse_pages(3)[0], ok({"unexpected": 1}))

    result = fetch(muse(), transport, base)

    assert result.pages == 1 and result.warnings == ["partial", "invalid_response"]
    assert len(json.loads((base / "themuse" / cache_files(base)[0]).read_text())["pages"]) == 1


def test_an_unusable_cache_file_is_ignored_for_the_next_best(base) -> None:
    clock = Clock()
    fetch(muse(), Transport(*muse_pages(1)), base, clock)
    clock.advance(HOUR)
    (base / "themuse" / "20990101T000000Z.json").write_text("{not json")
    (base / "themuse" / "20990101T000001Z.json").write_text(
        json.dumps({"format": 99, "complete": True, "pages": []}))
    (base / "themuse" / "20990101T000002Z.json").write_text(
        json.dumps({"format": 1, "complete": True, "pages": "x"}))
    (base / "themuse" / "20990101T000003Z.json").write_text(
        json.dumps({"format": 1, "complete": True, "pages": [{"results": []}]}))
    (base / "themuse" / "20990101T000004Z.json").write_text(json.dumps([1]))

    result = fetch(muse(), Transport(ok({})), base, clock)

    assert result.from_cache and len(result.postings) == 2


# --- the request guard ---------------------------------------------------------------------


def test_each_request_is_recorded_before_it_is_sent_and_failures_count(base) -> None:
    seen = []

    def spy(request):
        seen.append(len(ledger(base)))
        return 500, b""

    with pytest.raises(FetchError):
        fetch(muse(), Transport(spy), base)

    assert seen == [1] and ledger(base) == [T0]


def test_the_daily_budget_is_enforced_and_expires_after_a_day(base) -> None:
    clock = Clock()
    adapter = muse(max_requests_per_day=2, min_interval_s=0)
    for _ in range(2):
        with pytest.raises(FetchError):
            fetch(adapter, Transport((500, b"")), base, clock, max_age_s=0)
        clock.advance(60)
    blocked = Transport(ok(muse_page([muse_job(1)])))

    with pytest.raises(FetchError) as caught:
        fetch(adapter, blocked, base, clock, max_age_s=0)
    assert caught.value.code == "quota_exhausted" and blocked.calls == 0

    clock.advance(24 * HOUR)
    assert fetch(adapter, blocked, base, clock, max_age_s=0).pages == 1


def test_two_runs_too_close_together_are_refused(base) -> None:
    clock = Clock()
    adapter = muse(min_interval_s=30)
    with pytest.raises(FetchError):
        fetch(adapter, Transport((500, b"")), base, clock, max_age_s=0)
    clock.advance(29)
    blocked = Transport(ok(muse_page([muse_job(1)])))

    with pytest.raises(FetchError) as caught:
        fetch(adapter, blocked, base, clock, max_age_s=0)
    assert caught.value.code == "too_soon" and blocked.calls == 0

    clock.advance(1)
    assert fetch(adapter, blocked, base, clock, max_age_s=0).pages == 1


def test_a_guard_refusal_serves_the_stale_cache(base) -> None:
    clock = Clock()
    adapter = muse(min_interval_s=3 * HOUR)
    fetch(adapter, Transport(*muse_pages(1)), base, clock)
    clock.advance(2 * HOUR)
    silent = Transport(ok({}))

    result = fetch(adapter, silent, base, clock, max_age_s=HOUR)

    assert result.stale and result.warnings == ["stale_cache", "too_soon"] and silent.calls == 0


def test_the_daily_budget_also_stops_a_run_part_way(base) -> None:
    transport = Transport(*muse_pages(5))

    result = fetch(muse(max_requests_per_day=2, min_interval_s=0), transport, base)

    assert transport.calls == 2 and result.pages == 2
    assert result.warnings == ["partial", "quota_exhausted"] and len(ledger(base)) == 2


def test_pages_within_a_run_are_not_held_back_by_the_minimum_interval(base) -> None:
    transport = Transport(*muse_pages(3))

    result = fetch(muse(min_interval_s=3600), transport, base)

    assert result.pages == 3


def test_an_unreadable_ledger_is_treated_as_a_request_just_made(base) -> None:
    (base / "themuse").mkdir(parents=True)
    (base / "themuse" / "requests.json").write_text("garbage")
    transport = Transport(ok(muse_page([muse_job(1)])))

    with pytest.raises(FetchError) as caught:
        fetch(muse(), transport, base)

    assert caught.value.code == "too_soon" and transport.calls == 0


def test_each_source_has_its_own_ledger_and_cache(base) -> None:
    fetch(muse(), Transport(*muse_pages(1)), base)
    other = Transport(ok(arbeitnow_page([arbeitnow_job(1)])))

    result = fetch(Arbeitnow(policy=fast()), other, base)

    assert other.calls == 1 and result.source == "arbeitnow"
    assert ledger(base, "arbeitnow") == [T0] and ledger(base) == [T0]
    assert cache_dir("a", base) != cache_dir("b", base)


def test_old_cache_files_are_pruned_to_the_newest_few(base) -> None:
    clock = Clock()
    adapter = muse(keep=2, min_interval_s=0)
    for i in range(4):
        fetch(adapter, Transport(ok(muse_page([muse_job(i)]))), base, clock, max_age_s=0)
        clock.advance(HOUR)

    names = cache_files(base)

    assert len(names) == 2 and names == sorted(names)
    assert (base / "themuse" / "requests.json").exists()


def test_the_cache_location_follows_the_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("JOBPDF_CACHE_DIR", str(tmp_path / "elsewhere"))

    assert cache_dir("jooble") == tmp_path / "elsewhere" / "jooble"
    assert cache_dir("jooble", tmp_path / "b") == tmp_path / "b" / "jooble"


# --- keys never leak -------------------------------------------------------------------


def secret_adapters():
    return [
        Jooble(api_key=KEY, policy=fast()),
        Adzuna(app_id=KEY + "-id", app_key=KEY + "-key", policy=fast()),
        TheMuse(api_key=KEY, policy=fast()),
    ]


@pytest.mark.parametrize("adapter", secret_adapters(), ids=lambda a: a.name)
def test_keys_reach_neither_the_logs_nor_the_cache_nor_the_results(base, caplog, adapter) -> None:
    caplog.set_level(logging.DEBUG)
    clock = Clock()
    leaky = OSError(f"connection to {adapter.next_request([]).url} failed")
    ok_body = {"jooble": jooble_page([jooble_job(1)]), "adzuna": adzuna_page([adzuna_job(1)]),
               "themuse": muse_page([muse_job(1)])}[adapter.name]

    fetch(adapter, Transport(ok(ok_body)), base, clock)  # success
    clock.advance(10 * HOUR)
    fetch(adapter, Transport(leaky), base, clock)  # network failure, stale fallback
    clock.advance(10 * HOUR)
    fetch(adapter, Transport((500, b"")), base, clock)  # HTTP failure

    everything = caplog.text + "".join(
        p.read_text() for p in (base / adapter.name).iterdir())
    assert KEY not in everything
    assert len(caplog.records) > 0


def test_a_failure_with_the_key_in_the_exception_is_not_echoed_when_nothing_is_cached(
        base, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    adapter = Jooble(api_key=KEY, policy=fast())

    with pytest.raises(FetchError) as caught:
        fetch(adapter, Transport(OSError(f"https://jooble.org/api/{KEY} unreachable")), base)

    assert KEY not in caplog.text and KEY not in str(caught.value) and KEY not in repr(caught.value)


def test_logs_carry_counts_and_codes_but_no_posting_text(base, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    page = muse_page([muse_job(1, name="ConfidentialTitleXYZ", contents="<p>SecretBodyXYZ</p>")])

    fetch(muse(), Transport(ok(page)), base)

    assert "themuse" in caplog.text and "1 postings" in caplog.text
    assert "ConfidentialTitleXYZ" not in caplog.text and "SecretBodyXYZ" not in caplog.text
    assert "themuse.com" not in caplog.text


# --- scheduled summary -------------------------------------------------------------------------


def test_the_scheduled_summary_has_counts_and_no_postings(base) -> None:
    summary = run_source(muse(), base=base, transport=Transport(*muse_pages(2)),
                         clock=Clock(), sleep=no_sleep)

    assert summary == {
        "source": "themuse", "status": "ok", "count": 4, "pages": 2, "skipped": 0,
        "filtered": 0, "from_cache": False, "stale": False,
        "fetched_at": "2026-09-21T14:13:20+00:00", "warnings": [],
    }


def test_a_source_without_its_key_is_reported_as_skipped_not_failed(base) -> None:
    summary = run_source(Jooble(api_key=None), base=base, transport=Transport(ok({})))

    assert summary == {"source": "jooble", "status": "skipped", "reason": "not_configured"}


def test_a_scheduled_source_that_cannot_be_fetched_fails_the_task(base) -> None:
    with pytest.raises(FetchError):
        run_source(muse(), base=base, transport=Transport((500, b"")), clock=Clock())


def test_the_scheduled_run_refreshes_a_cache_older_than_its_limit(base) -> None:
    clock = Clock()
    run_source(muse(), base=base, transport=Transport(*muse_pages(1)), clock=clock)
    clock.advance(SCHEDULED_MAX_AGE_S - 1)
    assert run_source(muse(), base=base, transport=Transport(ok({})), clock=clock)["from_cache"]

    clock.advance(2)
    refreshed = Transport(ok(muse_page([muse_job(1)])))
    assert not run_source(muse(), base=base, transport=refreshed, clock=clock)["from_cache"]
    assert SCHEDULED_MAX_AGE_S == 7 * HOUR


def test_default_policies_are_modest() -> None:
    for adapter in (TheMuse(), Arbeitnow(), Adzuna(app_id="a", app_key="b"), Jooble(api_key="k")):
        policy = adapter.policy
        assert policy.max_pages <= 5 and policy.max_requests_per_day <= 30
        assert policy.min_interval_s >= 30 and policy.pause_s >= 1
        assert policy.max_response_bytes <= 16 * 1024 * 1024 and policy.max_cache_age_s >= HOUR


# --- the real transport (urllib is faked; the network is blocked in tests) ------------


class FakeResponse:
    def __init__(self, status: int, body: bytes) -> None:
        self.status, self.body, self.asked = status, body, None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n):
        self.asked = n
        return self.body[:n]


def test_the_real_transport_sends_method_body_headers_and_a_user_agent(monkeypatch) -> None:
    sent = {}
    response = FakeResponse(200, b"abcdef")

    def fake_urlopen(request, timeout):
        sent.update(request=request, timeout=timeout)
        return response

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    request = Request("POST", "https://jooble.org/api/k",
                      headers={"Content-Type": "application/json"}, body=b"{}")

    status, body = urllib_transport(request, 7.0, 4)

    sent_request = sent["request"]
    assert (status, body) == (200, b"abcde") and response.asked == 5  # one byte past the cap
    assert sent["timeout"] == 7.0 and sent_request.get_method() == "POST"
    assert sent_request.data == b"{}" and sent_request.full_url == "https://jooble.org/api/k"
    assert sent_request.get_header("Content-type") == "application/json"
    assert sent_request.get_header("User-agent") == runner.USER_AGENT
    assert sent_request.get_header("Accept") == "application/json"


def test_the_real_transport_turns_http_errors_into_a_status(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 503, "busy", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert urllib_transport(Request("GET", "https://x.test/"), 1.0, 10) == (503, b"")


def test_the_real_transport_lets_network_errors_through(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("no route")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(urllib.error.URLError):
        urllib_transport(Request("GET", "https://x.test/"), 1.0, 10)


def test_without_a_transport_the_real_one_is_used_and_the_network_is_blocked(base) -> None:
    # conftest.py makes any socket connect raise AssertionError; where even DNS is down the
    # request fails earlier as a network error. Either way nothing leaves the machine.
    with pytest.raises((AssertionError, FetchError)):
        fetch_source(Arbeitnow(), base=base, clock=Clock(), sleep=no_sleep)


def test_the_policy_is_frozen() -> None:
    with pytest.raises(AttributeError):
        Policy().max_pages = 99


# --- more edges the mutation checks asked for ------------------------------------------


def test_an_exhausted_budget_is_reported_before_the_minimum_interval(base) -> None:
    clock = Clock()
    adapter = muse(max_requests_per_day=1, min_interval_s=3600)
    with pytest.raises(FetchError):
        fetch(adapter, Transport((500, b"")), base, clock)

    with pytest.raises(FetchError) as caught:
        fetch(adapter, Transport(ok({})), base, clock, max_age_s=0)  # same instant: also too soon

    assert caught.value.code == "quota_exhausted"


def test_a_request_exactly_a_day_old_no_longer_counts(base) -> None:
    clock = Clock()
    adapter = muse(max_requests_per_day=1, min_interval_s=0)
    with pytest.raises(FetchError):
        fetch(adapter, Transport((500, b"")), base, clock)
    clock.advance(24 * HOUR - 1)
    with pytest.raises(FetchError) as caught:
        fetch(adapter, Transport(ok({})), base, clock)
    assert caught.value.code == "quota_exhausted"

    clock.advance(1)
    assert fetch(adapter, Transport(ok(muse_page([muse_job(1)]))), base, clock).pages == 1


def test_a_stale_partial_cache_reports_both(base) -> None:
    clock = Clock()
    fetch(muse(max_pages=2), Transport(*muse_pages(10)), base, clock)
    clock.advance(10 * HOUR)

    result = fetch(muse(max_pages=2), Transport((500, b"")), base, clock)

    assert result.stale and result.warnings == ["stale_cache", "http_error", "partial"]


def test_a_response_exactly_at_the_size_limit_is_accepted(base) -> None:
    body = json.dumps(muse_page([muse_job(1)])).encode()

    assert fetch(muse(max_response_bytes=len(body)), Transport((200, body)), base).pages == 1
    with pytest.raises(FetchError) as caught:
        fetch(muse(max_response_bytes=len(body) - 1), Transport((200, body)), base / "other")
    assert caught.value.code == "response_too_large"


def test_the_policy_timeout_and_size_cap_reach_the_transport(base) -> None:
    transport = Transport(ok(muse_page([muse_job(1)])))

    fetch(muse(timeout_s=12.5, max_response_bytes=4096), transport, base)

    assert transport.limits == [(12.5, 4096)]


def test_the_newest_cache_is_the_one_served(base) -> None:
    clock = Clock()
    adapter = muse(min_interval_s=0)
    fetch(adapter, Transport(ok(muse_page([muse_job(1)]))), base, clock)
    clock.advance(HOUR)
    fetch(adapter, Transport(ok(muse_page([muse_job(2)]))), base, clock, max_age_s=0)

    result = fetch(adapter, Transport(ok({})), base, clock)

    assert [p.source_id for p in result.postings] == ["1002"]


def test_pruning_keeps_the_newest_runs_not_the_oldest(base) -> None:
    clock = Clock()
    adapter = muse(keep=2, min_interval_s=0)
    for i in range(4):
        fetch(adapter, Transport(ok(muse_page([muse_job(i)]))), base, clock, max_age_s=0)
        clock.advance(HOUR)

    assert cache_files(base) == ["20260921T161320Z.json", "20260921T171320Z.json"]


def test_a_cache_in_an_unknown_format_is_ignored_even_if_its_pages_are_good(base) -> None:
    clock = Clock()
    folder = base / "themuse"
    folder.mkdir(parents=True)
    page = muse_page([muse_job(1)])
    (folder / "20260921T140000Z.json").write_text(
        json.dumps({"format": 2, "complete": True, "pages": [page]}))

    with pytest.raises(FetchError):
        fetch(muse(), Transport((500, b"")), base, clock)


def test_a_cache_whose_later_page_is_unusable_still_serves_the_good_pages(base) -> None:
    folder = base / "themuse"
    folder.mkdir(parents=True)
    pages = [muse_page([muse_job(1)], page=1, page_count=2), {"unexpected": True}]
    (folder / "20260921T140000Z.json").write_text(
        json.dumps({"format": 1, "complete": True, "pages": pages}))

    result = fetch(muse(), Transport(ok({})), base, Clock(T0))

    assert [p.source_id for p in result.postings] == ["1001"]
