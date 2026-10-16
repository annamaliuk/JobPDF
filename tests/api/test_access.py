"""JM-30: API keys, rate limits and upload size.

Tests marked ``db`` need Postgres through DATABASE_URL and skip without it; each works in a
throwaway schema. The limiter contract runs against both stores.
"""

import hmac
import importlib.util
import io
import logging
import threading
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import psycopg
import pytest

from jobpdf.api import access
from jobpdf.api.access import (
    BODY_OVERHEAD_BYTES,
    HTTP_STATUS,
    AccessControl,
    AccessError,
    ApiKeys,
    Caller,
    ConfigError,
    InMemoryCounterStore,
    LimiterUnavailable,
    PostgresCounterStore,
    RateLimit,
    RateLimiter,
    api_key_from_headers,
    build_access_control,
    check_declared_size,
    hash_key,
    new_api_key,
    read_upload,
)
from jobpdf.normalization.db import database_url
from jobpdf.pipeline import MAX_UPLOAD_BYTES


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRIPTS = Path(__file__).parents[2] / "scripts"
make_api_key = _load_script("make_api_key")
SECRET = "x" * 32
KEY_A, KEY_B = "key-a-" + "a" * 40, "key-b-" + "b" * 40
KEYS = ApiKeys({"alpha": hash_key(KEY_A), "beta": hash_key(KEY_B)})
MINUTE_2 = RateLimit("minute", 60, 2)
DAY_3 = RateLimit("day", 86400, 3)


class Clock:
    def __init__(self, now: float = 1_000_020.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


# --- fixtures ----------------------------------------------------------------------


@pytest.fixture
def db_url() -> str:
    url = database_url()
    if not url:
        pytest.skip("DATABASE_URL not set (environment or .env)")
    return url


@pytest.fixture
def schema(db_url: str) -> Iterator[str]:
    """A throwaway schema per test."""
    name = f"test_{uuid.uuid4().hex[:12]}"
    try:
        admin = psycopg.connect(db_url, autocommit=True, connect_timeout=10)
    except psycopg.OperationalError as exc:
        pytest.skip(f"database not reachable: {type(exc).__name__}")
    with admin:
        admin.execute(f'CREATE SCHEMA "{name}"')
    try:
        yield name
    finally:
        with psycopg.connect(db_url, autocommit=True, connect_timeout=10) as admin:
            admin.execute(f'DROP SCHEMA "{name}" CASCADE')


@pytest.fixture
def make_pg(db_url: str, schema: str) -> Iterator[Callable[..., PostgresCounterStore]]:
    made: list[PostgresCounterStore] = []

    def make(**kwargs: object) -> PostgresCounterStore:
        store = PostgresCounterStore(db_url, options=f"-c search_path={schema}", **kwargs)
        made.append(store)
        return store

    yield make
    for store in made:
        store.close()


@pytest.fixture(params=["memory", pytest.param("postgres", marks=pytest.mark.db)])
def store(request):
    if request.param == "memory":
        return InMemoryCounterStore()
    return request.getfixturevalue("make_pg")()


def limiter(store, clock, *, key=(MINUTE_2, DAY_3), ip=(MINUTE_2, DAY_3), **kwargs) -> RateLimiter:
    return RateLimiter(store, ip_secret=SECRET, key_limits=key, ip_limits=ip, clock=clock,
                       **kwargs)


def refused(call: Callable[[], None]) -> AccessError:
    with pytest.raises(AccessError) as info:
        call()
    return info.value


# --- API keys ----------------------------------------------------------------------


def test_a_known_key_gives_its_name() -> None:
    assert KEYS.authenticate(KEY_A) == "alpha" and KEYS.authenticate(KEY_B) == "beta"


@pytest.mark.parametrize("presented", [None, ""])
def test_no_key_is_missing(presented) -> None:
    assert refused(lambda: KEYS.authenticate(presented)).code == "missing_key"


@pytest.mark.parametrize("presented", ["nope", KEY_A + "x", KEY_A[:-1], KEY_A.upper(), " " + KEY_A])
def test_any_other_key_is_invalid(presented) -> None:
    assert refused(lambda: KEYS.authenticate(presented)).code == "invalid_key"


def test_every_stored_key_is_compared_whatever_the_outcome(monkeypatch) -> None:
    calls = []
    real = hmac.compare_digest
    monkeypatch.setattr(access.hmac, "compare_digest", lambda a, b: calls.append(1) or real(a, b))

    KEYS.authenticate(KEY_A)  # matches the first entry
    refused(lambda: KEYS.authenticate("nope"))

    assert len(calls) == 4  # two keys, two requests; no early exit


def test_status_says_set_or_missing_never_a_value() -> None:
    assert KEYS.status() == "set" and ApiKeys({}).status() == "missing"
    assert len(KEYS) == 2 and KEYS.names == ["alpha", "beta"]


def test_an_empty_key_set_refuses_everything() -> None:
    assert refused(lambda: ApiKeys({}).authenticate(KEY_A)).code == "invalid_key"


def test_a_key_is_never_stored_or_logged(caplog) -> None:
    with caplog.at_level(logging.DEBUG):
        keys = ApiKeys.from_environment({"JOBPDF_API_KEYS": f"alpha:{hash_key(KEY_A)}"})
        keys.authenticate(KEY_A)
        refused(lambda: keys.authenticate("wrong-key-xyz"))

    assert KEY_A not in caplog.text and "wrong-key-xyz" not in caplog.text
    assert KEY_A not in repr(keys.__dict__) and "set" in caplog.text


def test_new_keys_are_long_unique_and_hash_to_64_hex_characters() -> None:
    keys = {new_api_key() for _ in range(50)}

    assert len(keys) == 50 and all(len(k) >= 40 for k in keys)
    assert len(hash_key("x")) == 64


# --- reading the key list ----------------------------------------------------------

DIGEST_1, DIGEST_2 = "ab" * 32, "cd" * 32


def test_the_list_takes_lines_commas_comments_and_any_hex_case() -> None:
    text = f"# who has access\nalpha:{DIGEST_1.upper()}, beta:{DIGEST_2}\n\n"

    keys = ApiKeys.parse(text)

    assert keys.names == ["alpha", "beta"]
    assert ApiKeys.parse("").status() == "missing"


def test_hex_case_does_not_matter_when_checking_a_key() -> None:
    keys = ApiKeys.parse(f"alpha:{hash_key(KEY_A).upper()}")

    assert keys.authenticate(KEY_A) == "alpha"
    assert ApiKeys({"alpha": hash_key(KEY_A).upper()}).authenticate(KEY_A) == "alpha"


@pytest.mark.parametrize("text", [
    "alpha", f"alpha{DIGEST_1}", f"al pha:{DIGEST_1}", f":{DIGEST_1}", "alpha:abc",
    f"alpha:{'zz' * 32}", f"{'n' * 65}:{DIGEST_1}", f"alpha:{DIGEST_1}\nalpha:{DIGEST_2}",
    f"alpha:{DIGEST_1}\nbeta:{DIGEST_1}",
])
def test_a_bad_entry_is_a_config_error_that_does_not_echo_it(text) -> None:
    with pytest.raises(ConfigError) as info:
        ApiKeys.parse(text)

    assert DIGEST_1 not in str(info.value) and "entry" in str(info.value)


def test_keys_come_from_the_variable_and_the_file_together(tmp_path) -> None:
    file = tmp_path / "keys"
    file.write_text(f"beta:{DIGEST_2}\n", encoding="utf-8")
    environ = {"JOBPDF_API_KEYS": f"alpha:{DIGEST_1}", "JOBPDF_API_KEYS_FILE": str(file)}

    assert ApiKeys.from_environment(environ).names == ["alpha", "beta"]
    assert ApiKeys.from_environment({"JOBPDF_API_KEYS_FILE": str(file)}).names == ["beta"]


def test_an_unreadable_keys_file_is_a_config_error(tmp_path) -> None:
    with pytest.raises(ConfigError, match="API keys file"):
        ApiKeys.from_environment({"JOBPDF_API_KEYS_FILE": str(tmp_path / "missing")})


def test_no_keys_is_allowed_but_logged(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        keys = ApiKeys.from_environment({})

    assert keys.status() == "missing" and "missing" in caplog.text


def test_settings_prefer_the_environment_over_the_env_file(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("# JOBPDF_API_KEYS=commented\nJOBPDF_API_KEYS='from-file'\nOTHER=1\n",
                        encoding="utf-8")
    monkeypatch.delenv("JOBPDF_API_KEYS", raising=False)

    assert access._setting("JOBPDF_API_KEYS", None, env_file) == "from-file"
    assert access._setting("OTHER", None, env_file) == "1"
    assert access._setting("ABSENT", None, env_file) is None
    monkeypatch.setenv("JOBPDF_API_KEYS", "from-env")
    assert access._setting("JOBPDF_API_KEYS", None, env_file) == "from-env"
    assert access._setting("JOBPDF_API_KEYS", {}, env_file) is None  # explicit mapping only


@pytest.mark.parametrize(("headers", "expected"), [
    ({"X-API-Key": "k1"}, "k1"),
    ({"x-api-key": "k1"}, "k1"),
    ({"Authorization": "Bearer k2"}, "k2"),
    ({"authorization": "bearer   k2 "}, "k2"),
    ({"X-API-Key": "k1", "Authorization": "Bearer k2"}, "k1"),
    ({"Authorization": "Basic abc"}, None),
    ({"Authorization": "Bearer"}, None),
    ({"X-API-Key": "  "}, None),
    ({}, None),
])
def test_the_key_header(headers, expected) -> None:
    assert api_key_from_headers(headers) == expected


# --- the limiter (both stores) -----------------------------------------------------


def test_uploads_up_to_the_limit_pass_and_the_next_is_refused(store) -> None:
    clock = Clock()
    lim = limiter(store, clock)

    lim.check_key("alpha")
    lim.check_key("alpha")
    err = refused(lambda: lim.check_key("alpha"))

    assert (err.code, err.status) == ("rate_limited", 429)


def test_retry_after_is_the_time_to_the_end_of_the_window(store) -> None:
    clock = Clock(1_000_020.0 + 40.5)  # 1_000_020 starts a minute: it ends at +60
    lim = limiter(store, clock)
    lim.check_key("alpha")
    lim.check_key("alpha")

    assert refused(lambda: lim.check_key("alpha")).retry_after == 20  # ceil(19.5)


def test_the_next_window_starts_fresh(store) -> None:
    clock = Clock()
    lim = limiter(store, clock, key=(MINUTE_2,))
    lim.check_key("alpha")
    lim.check_key("alpha")
    refused(lambda: lim.check_key("alpha"))

    clock.now += 60
    lim.check_key("alpha")
    lim.check_key("alpha")
    assert refused(lambda: lim.check_key("alpha")).code == "rate_limited"


def test_the_daily_cap_holds_across_minutes(store) -> None:
    clock = Clock()
    lim = limiter(store, clock)
    for _ in range(3):
        lim.check_key("alpha")
        clock.now += 60  # a new minute each time, same day

    err = refused(lambda: lim.check_key("alpha"))

    assert err.code == "rate_limited" and err.retry_after > 60


def test_a_refused_upload_does_not_use_up_quota(store) -> None:
    clock = Clock()
    lim = limiter(store, clock)
    lim.check_key("alpha")
    lim.check_key("alpha")
    refused(lambda: lim.check_key("alpha"))  # refused by the minute rule
    refused(lambda: lim.check_key("alpha"))

    clock.now += 60
    lim.check_key("alpha")  # the third *allowed* upload of the day
    assert refused(lambda: lim.check_key("alpha")).code == "rate_limited"
    clock.now += 60
    assert refused(lambda: lim.check_key("alpha")).code == "rate_limited"  # day is full: 3 of 3


def test_a_refusal_by_a_later_rule_uncounts_the_earlier_ones(store) -> None:
    clock = Clock()
    lim = limiter(store, clock, key=(RateLimit("day", 86400, 3), RateLimit("minute", 60, 1)))

    lim.check_key("alpha")  # day 1/3
    refused(lambda: lim.check_key("alpha"))  # the minute rule refuses; day must stay 1/3
    clock.now += 60
    lim.check_key("alpha")  # day 2/3
    refused(lambda: lim.check_key("alpha"))
    clock.now += 60
    lim.check_key("alpha")  # day 3/3: only possible if the refusals counted nothing
    clock.now += 60

    assert refused(lambda: lim.check_key("alpha")).code == "rate_limited"


def test_keys_are_counted_separately(store) -> None:
    lim = limiter(store, Clock(), key=(RateLimit("minute", 60, 1),))

    lim.check_key("alpha")
    lim.check_key("beta")
    assert refused(lambda: lim.check_key("alpha")).code == "rate_limited"


def test_key_and_ip_counters_do_not_mix(store) -> None:
    lim = limiter(store, Clock(), key=(RateLimit("minute", 60, 1),),
                  ip=(RateLimit("minute", 60, 1),))

    lim.check_key("203.0.113.9")  # a key *named* like an address
    lim.check_ip("203.0.113.9")
    assert refused(lambda: lim.check_ip("203.0.113.9")).code == "rate_limited"


def test_ips_are_counted_per_network(store) -> None:
    lim = limiter(store, Clock(), ip=(RateLimit("minute", 60, 1),))

    lim.check_ip("203.0.113.9")
    lim.check_ip("203.0.113.10")
    assert refused(lambda: lim.check_ip("203.0.113.9")).code == "rate_limited"
    lim.check_ip("2001:db8:1:1::1")
    assert refused(lambda: lim.check_ip("2001:db8:1:1:ffff::2")).code == "rate_limited"  # same /64
    lim.check_ip("2001:db8:1:2::1")  # another /64


def test_an_ipv4_mapped_address_counts_as_the_ipv4_one() -> None:
    lim = limiter(InMemoryCounterStore(), Clock())

    assert lim.ip_subject("::ffff:203.0.113.9") == lim.ip_subject(" 203.0.113.9 ")


def test_the_stored_ip_is_a_keyed_hash_not_the_address() -> None:
    a = limiter(InMemoryCounterStore(), Clock())
    b = RateLimiter(InMemoryCounterStore(), ip_secret="y" * 32)

    subject = a.ip_subject("203.0.113.9")

    assert "203" not in subject and len(subject) == 64
    assert subject != b.ip_subject("203.0.113.9")  # a different secret, a different value
    assert subject != a.ip_subject("203.0.113.10")


@pytest.mark.parametrize("bad", ["", "not-an-ip", "999.1.1.1", "203.0.113"])
def test_an_unusable_client_address_is_a_programming_error(bad) -> None:
    with pytest.raises(ValueError):
        limiter(InMemoryCounterStore(), Clock()).check_ip(bad)


def test_the_ip_secret_must_be_long_enough() -> None:
    with pytest.raises(ConfigError) as info:
        RateLimiter(InMemoryCounterStore(), ip_secret="short")

    assert "short" not in str(info.value)


def test_a_rate_limit_must_make_sense() -> None:
    for seconds, limit in [(0, 1), (60, 0), (-1, 5)]:
        with pytest.raises(ValueError):
            RateLimit("x", seconds, limit)


def test_the_default_limits_are_the_documented_ones() -> None:
    assert [(r.name, r.seconds, r.limit) for r in access.KEY_LIMITS] == [
        ("minute", 60, 6), ("day", 86400, 100)]
    assert [(r.name, r.seconds, r.limit) for r in access.IP_LIMITS] == [
        ("hour", 3600, 5), ("day", 86400, 20)]


def test_logs_name_the_key_but_never_the_address(caplog) -> None:
    lim = limiter(InMemoryCounterStore(), Clock(), key=(RateLimit("minute", 60, 1),),
                  ip=(RateLimit("minute", 60, 1),))
    lim.check_key("alpha")
    lim.check_ip("203.0.113.9")

    with caplog.at_level(logging.DEBUG):
        refused(lambda: lim.check_key("alpha"))
        refused(lambda: lim.check_ip("203.0.113.9"))

    assert "key=alpha" in caplog.text and "203.0.113" not in caplog.text
    assert lim.ip_subject("203.0.113.9") not in caplog.text


class BrokenStore:
    def hit(self, *args):
        raise LimiterUnavailable("down")


def test_an_unreachable_store_refuses_by_default() -> None:
    err = refused(lambda: limiter(BrokenStore(), Clock()).check_key("alpha"))

    assert (err.code, err.status, err.retry_after) == ("limiter_unavailable", 503, None)


def test_fail_open_lets_the_request_through_and_says_so(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        limiter(BrokenStore(), Clock(), fail_open=True).check_key("alpha")

    assert "unavailable" in caplog.text


# --- the Postgres store ------------------------------------------------------------


@pytest.mark.db
def test_a_second_connection_shares_the_counts(make_pg) -> None:
    clock = Clock()
    first = limiter(make_pg(), clock, key=(RateLimit("minute", 60, 2),))
    second = limiter(make_pg(), clock, key=(RateLimit("minute", 60, 2),))

    first.check_key("alpha")
    second.check_key("alpha")

    assert refused(lambda: first.check_key("alpha")).code == "rate_limited"


@pytest.mark.db
def test_concurrent_uploads_never_overshoot_the_limit(make_pg) -> None:
    clock = Clock()
    results: list[bool] = []
    lock = threading.Lock()
    limiters = [limiter(make_pg(), clock, key=(RateLimit("minute", 60, 5),)) for _ in range(8)]

    def work(lim: RateLimiter) -> None:
        for _ in range(3):
            try:
                lim.check_key("alpha")
                ok = True
            except AccessError:
                ok = False
            with lock:
                results.append(ok)

    threads = [threading.Thread(target=work, args=(lim,)) for lim in limiters]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 24 and sum(results) == 5


@pytest.mark.db
def test_the_table_holds_hashes_and_never_an_address(make_pg, db_url, schema) -> None:
    lim = limiter(make_pg(), Clock())

    lim.check_ip("203.0.113.9")
    lim.check_key("alpha")

    with psycopg.connect(db_url, options=f"-c search_path={schema}") as conn:
        rows = conn.execute("SELECT scope, subject FROM rate_limit_counters").fetchall()
    assert {scope for scope, _ in rows} == {"ip", "key"}
    assert not any("203.0.113" in subject for _, subject in rows)


@pytest.mark.db
def test_purge_removes_only_long_finished_windows(make_pg) -> None:
    pg = make_pg()
    old, recent = 1_000_000.0, 1_000_000.0 + access.PURGE_AFTER_S + 7200
    pg.hit("key", "alpha", [MINUTE_2], old)
    pg.hit("key", "beta", [MINUTE_2], recent)

    removed = pg.purge(recent)

    assert removed == 1
    assert pg.hit("key", "beta", [MINUTE_2], recent) is None  # still counted: 2 of 2
    assert pg.hit("key", "beta", [MINUTE_2], recent) is not None


@pytest.mark.db
def test_a_lost_connection_refuses_then_recovers(make_pg, db_url) -> None:
    pg = make_pg(retry_after_s=0.0)
    lim = limiter(pg, Clock(), key=(RateLimit("minute", 60, 100),))
    lim.check_key("alpha")
    with psycopg.connect(db_url, autocommit=True) as admin:
        admin.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                      "WHERE pid <> pg_backend_pid() AND application_name = ''"
                      " AND datname = current_database() AND usename = current_user")

    err = refused(lambda: lim.check_key("alpha"))  # the dead connection is noticed here
    lim.check_key("alpha")  # and the next call reconnects

    assert (err.code, err.status) == ("limiter_unavailable", 503)


@pytest.mark.db
def test_after_a_failure_the_store_is_paused_so_requests_do_not_wait(make_pg, db_url) -> None:
    clock = Clock()
    pg = make_pg(retry_after_s=30.0, clock=clock)
    pg.hit("key", "alpha", [MINUTE_2], 1.0)
    with psycopg.connect(db_url, autocommit=True) as admin:
        admin.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                      "WHERE pid <> pg_backend_pid() AND datname = current_database() "
                      "AND usename = current_user")

    with pytest.raises(LimiterUnavailable):
        pg.hit("key", "alpha", [MINUTE_2], 1.0)  # notices the dead connection
    with pytest.raises(LimiterUnavailable):
        pg.hit("key", "beta", [MINUTE_2], 1.0)  # paused: does not even try to reconnect
    clock.now += 31

    assert pg.hit("key", "beta", [MINUTE_2], 1.0) is None  # reconnected


@pytest.mark.db
def test_the_schema_can_be_applied_twice(make_pg) -> None:
    make_pg()
    make_pg().hit("key", "alpha", [MINUTE_2], 1.0)


def test_the_memory_store_forgets_old_windows() -> None:
    mem = InMemoryCounterStore()
    mem.hit("key", "alpha", [MINUTE_2], 1_000_000.0)
    for _ in range(access.PURGE_EVERY_HITS):
        mem.hit("key", "b", [MINUTE_2], 1_000_000.0 + access.PURGE_AFTER_S + 7200)

    assert len(mem) == 1


# --- size --------------------------------------------------------------------------


def test_the_limit_is_five_megabytes_like_the_pipeline() -> None:
    assert MAX_UPLOAD_BYTES == 5 * 1024 * 1024
    assert AccessControl(KEYS, limiter(InMemoryCounterStore(), Clock())).max_bytes == 5 * 1024**2


def test_declared_size_allows_multipart_overhead_but_not_much_more() -> None:
    check_declared_size(None)
    check_declared_size(MAX_UPLOAD_BYTES)
    check_declared_size(MAX_UPLOAD_BYTES + BODY_OVERHEAD_BYTES)
    err = refused(lambda: check_declared_size(MAX_UPLOAD_BYTES + BODY_OVERHEAD_BYTES + 1))

    assert (err.code, err.status) == ("too_large", 413)


def test_the_exact_limit_is_read_in_full_and_one_byte_more_is_refused() -> None:
    assert read_upload(io.BytesIO(b"a" * 100), 100) == b"a" * 100
    assert refused(lambda: read_upload(io.BytesIO(b"a" * 101), 100)).code == "too_large"
    assert read_upload(io.BytesIO(b""), 100) == b""


class Trickle(io.RawIOBase):
    """Returns at most 7 bytes per read and records the biggest total it handed out."""

    def __init__(self, size: int) -> None:
        self.left, self.given = size, 0

    def read(self, n: int = -1) -> bytes:
        n = 7 if n < 0 else min(n, 7, self.left)
        self.left -= n
        self.given += n
        return b"x" * n


def test_short_reads_are_put_together() -> None:
    assert read_upload(Trickle(100), 100) == b"x" * 100


class Endless:
    """A file that always has more; read(n) returns exactly n bytes."""

    def __init__(self) -> None:
        self.given = 0

    def read(self, n: int = -1) -> bytes:
        self.given += n
        return b"x" * n


def test_a_read_never_asks_for_more_than_one_byte_past_the_limit() -> None:
    endless = Endless()

    refused(lambda: read_upload(endless, 1000))

    assert endless.given == 1001


def test_a_huge_body_is_never_read_past_the_limit() -> None:
    huge = Trickle(10**9)

    refused(lambda: read_upload(huge, 1000))

    assert huge.given <= 1000 + 7


# --- putting it together -----------------------------------------------------------


def control(**kwargs) -> AccessControl:
    return AccessControl(KEYS, limiter(InMemoryCounterStore(), Clock(), **kwargs))


def test_a_valid_request_gets_its_caller() -> None:
    assert control().authorize_api({"X-API-Key": KEY_A}, 1000) == Caller("api_key", "alpha")
    assert control().authorize_web("203.0.113.9", 1000) == Caller("web", None)


def test_the_key_is_checked_before_size_and_limits() -> None:
    err = refused(lambda: control().authorize_api({"X-API-Key": "nope"}, 10**9))
    none = refused(lambda: control().authorize_api({}, 10**9))

    assert (err.code, none.code) == ("invalid_key", "missing_key")


def test_a_too_large_upload_does_not_use_up_quota() -> None:
    ctl = control(key=(RateLimit("minute", 60, 1),))

    refused(lambda: ctl.authorize_api({"X-API-Key": KEY_A}, 10**9))

    assert ctl.authorize_api({"X-API-Key": KEY_A}, 10).name == "alpha"


def test_the_api_is_limited_by_key_and_the_web_by_address() -> None:
    ctl = control(key=(RateLimit("minute", 60, 1),), ip=(RateLimit("minute", 60, 2),))

    ctl.authorize_api({"X-API-Key": KEY_A}, 1)
    assert refused(lambda: ctl.authorize_api({"X-API-Key": KEY_A}, 1)).code == "rate_limited"
    ctl.authorize_api({"X-API-Key": KEY_B}, 1)

    ctl.authorize_web("203.0.113.9", 1)
    ctl.authorize_web("203.0.113.9", 1)  # the API calls did not use the address's quota
    assert refused(lambda: ctl.authorize_web("203.0.113.9", 1)).code == "rate_limited"


class Recording:
    def __init__(self) -> None:
        self.inner = InMemoryCounterStore()
        self.scopes: list[str] = []

    def hit(self, scope, subject, rules, now):
        self.scopes.append(scope)
        return self.inner.hit(scope, subject, rules, now)


def test_the_api_touches_only_key_counters_and_the_web_only_ip_counters() -> None:
    rec = Recording()
    ctl = AccessControl(KEYS, limiter(rec, Clock()))

    ctl.authorize_api({"X-API-Key": KEY_A}, 1)
    assert rec.scopes == ["key"]
    ctl.authorize_web("203.0.113.9", 1)
    assert rec.scopes == ["key", "ip"]


def test_web_uploads_are_size_checked_too() -> None:
    assert refused(lambda: control().authorize_web("203.0.113.9", 10**9)).code == "too_large"


def test_reading_an_upload_uses_the_controls_limit() -> None:
    ctl = AccessControl(KEYS, limiter(InMemoryCounterStore(), Clock()), max_bytes=10)

    assert ctl.read_upload(io.BytesIO(b"0123456789")) == b"0123456789"
    assert refused(lambda: ctl.read_upload(io.BytesIO(b"0123456789!"))).code == "too_large"


def test_every_error_code_has_a_status() -> None:
    assert HTTP_STATUS == {"missing_key": 401, "invalid_key": 401, "rate_limited": 429,
                           "too_large": 413, "limiter_unavailable": 503}
    assert AccessError("something_else").status == 403
    assert str(AccessError("rate_limited", 5)) == "rate_limited"


# --- building from the environment ---------------------------------------------------


def test_the_environment_builds_a_working_control() -> None:
    environ = {"JOBPDF_API_KEYS": f"alpha:{hash_key(KEY_A)}", "JOBPDF_RATE_LIMIT_SECRET": SECRET}

    ctl = build_access_control(environ, store=InMemoryCounterStore())

    assert ctl.authorize_api({"X-API-Key": KEY_A}, 1).name == "alpha"


def test_a_missing_or_short_secret_stops_startup() -> None:
    with pytest.raises(ConfigError, match="RATE_LIMIT_SECRET"):
        build_access_control({}, store=InMemoryCounterStore())
    with pytest.raises(ConfigError) as info:
        build_access_control({"JOBPDF_RATE_LIMIT_SECRET": "tiny"}, store=InMemoryCounterStore())

    assert "tiny" not in str(info.value)


def test_without_a_database_url_startup_stops(monkeypatch) -> None:
    monkeypatch.setattr(access, "database_url", lambda: None)

    with pytest.raises(ConfigError, match="DATABASE_URL"):
        build_access_control({"JOBPDF_RATE_LIMIT_SECRET": SECRET})


# --- the key script ----------------------------------------------------------------


def test_the_script_prints_a_key_and_the_line_that_accepts_it(capsys) -> None:
    assert make_api_key.main(["partner-acme"]) == 0
    out = capsys.readouterr().out.splitlines()

    key = out[0].rsplit(": ", 1)[1]
    keys = ApiKeys.parse(out[-1])
    assert keys.names == ["partner-acme"] and keys.authenticate(key) == "partner-acme"
    assert key not in out[-1]  # the line has the hash only


def test_the_script_refuses_a_bad_name(capsys) -> None:
    assert make_api_key.main(["bad name!"]) == 1
    assert "name" in capsys.readouterr().out
