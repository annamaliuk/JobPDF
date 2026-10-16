"""Access control for the API and the web form (JM-30).

Three checks stand between an upload and the pipeline, because every upload costs LLM
money (see "Cost and abuse" in the project description):

- **API key.** Callers send ``X-API-Key`` (or ``Authorization: Bearer``). The server holds
  only SHA-256 hashes, read from the environment (``JOBPDF_API_KEYS``, also found in
  ``.env``) or a secrets file (``JOBPDF_API_KEYS_FILE``), one ``name:sha256hex`` per line.
  Keys are long random strings (``scripts/make_api_key.py``), so a plain hash is enough.
  Code and logs say a key is "set" or "missing" and use its *name*; the key never appears.
- **Rate limits**, per API key (the API) and per client IP (the web form), as fixed
  windows. Counters live in Postgres (``sql/003_rate_limit.sql``) so every worker and every
  restart sees the same numbers. IPs are stored only as a keyed hash (an IPv6 address by
  its /64 network), and never logged.
- **Size.** 5 MB (the pipeline's own limit, ``MAX_UPLOAD_BYTES``), checked first from the
  declared length and then on the bytes actually read.

This module has no web framework in it. The app (JM-36) wraps ``AccessControl`` in a thin
dependency, maps ``AccessError.code`` to a status with ``HTTP_STATUS`` and sends
``Retry-After``. It decides whether ``X-Forwarded-For`` can be trusted; ``authorize_web``
takes the client IP as given.

Decisions to review (all constants or arguments):
- The limits below are the implementer's choice.
- If the counter store is unreachable the request is refused (``limiter_unavailable``),
  because an open door costs money; ``fail_open=True`` flips that.
- A refused request does not use up quota (the counters roll back), so retrying after
  ``retry_after`` works.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import logging
import math
import os
import re
import secrets
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal, Protocol

import psycopg

from jobpdf.normalization.db import (
    CONNECT_TIMEOUT_S,
    ENV_FILE,
    REPO_ROOT,
    apply_schema,
    database_url,
)
from jobpdf.pipeline import MAX_UPLOAD_BYTES

log = logging.getLogger(__name__)

# --- configuration -----------------------------------------------------------------

API_KEYS_ENV = "JOBPDF_API_KEYS"
API_KEYS_FILE_ENV = "JOBPDF_API_KEYS_FILE"
RATE_LIMIT_SECRET_ENV = "JOBPDF_RATE_LIMIT_SECRET"
MIN_SECRET_CHARS = 16
KEY_HEADER = "x-api-key"
KEY_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
SHA256_HEX = re.compile(r"^[0-9a-fA-F]{64}$")

# A multipart body is a little bigger than its file.
BODY_OVERHEAD_BYTES = 64 * 1024
READ_CHUNK_BYTES = 64 * 1024

RATE_LIMIT_SCHEMA_FILE = REPO_ROOT / "sql" / "003_rate_limit.sql"
# After a lost connection the counter store is left alone this long, so each request
# does not wait out the connect timeout.
STORE_RETRY_AFTER_S = 5.0
PURGE_EVERY_HITS = 1000
PURGE_AFTER_S = 3600  # keep a window's row this long after it ended

# Error codes (AccessError.code) and the HTTP status the app should answer with.
MISSING_KEY = "missing_key"
INVALID_KEY = "invalid_key"
RATE_LIMITED = "rate_limited"
TOO_LARGE = "too_large"
LIMITER_UNAVAILABLE = "limiter_unavailable"
HTTP_STATUS = {
    MISSING_KEY: 401,
    INVALID_KEY: 401,
    RATE_LIMITED: 429,
    TOO_LARGE: 413,
    LIMITER_UNAVAILABLE: 503,
}


@dataclass(frozen=True)
class RateLimit:
    """At most ``limit`` uploads per fixed window of ``seconds``."""

    name: str
    seconds: int
    limit: int

    def __post_init__(self) -> None:
        if self.seconds < 1 or self.limit < 1:
            raise ValueError("a rate limit needs seconds >= 1 and limit >= 1")


# Per API key: a short burst limit and a daily cap. Per IP (the anonymous web form) is
# stricter, because nobody has vouched for the caller.
KEY_LIMITS: tuple[RateLimit, ...] = (RateLimit("minute", 60, 6), RateLimit("day", 86400, 100))
IP_LIMITS: tuple[RateLimit, ...] = (RateLimit("hour", 3600, 5), RateLimit("day", 86400, 20))


class AccessError(Exception):
    """The request is refused. ``code`` is stable; the message never holds a key or an IP."""

    def __init__(self, code: str, retry_after: int | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.retry_after = retry_after  # seconds, for a Retry-After header

    @property
    def status(self) -> int:
        return HTTP_STATUS.get(self.code, 403)


class ConfigError(RuntimeError):
    """Access control is misconfigured. Messages never echo configuration values."""


# --- API keys ----------------------------------------------------------------------


def new_api_key() -> str:
    """A fresh key: 256 random bits. Shown once; only its hash is stored."""
    return secrets.token_urlsafe(32)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def api_key_from_headers(headers: Mapping[str, str]) -> str | None:
    """The presented key from ``X-API-Key`` or ``Authorization: Bearer``; header names are
    case-insensitive. None when neither is present."""
    lowered = {name.lower(): value for name, value in headers.items()}
    key = lowered.get(KEY_HEADER)
    if key is None:
        scheme, _, token = (lowered.get("authorization") or "").partition(" ")
        key = token if scheme.lower() == "bearer" else None
    key = key.strip() if key else None
    return key or None


class ApiKeys:
    """The accepted keys, as name -> SHA-256 hex."""

    def __init__(self, hashes: Mapping[str, str]) -> None:
        self._hashes = {name: digest.lower() for name, digest in hashes.items()}

    def __len__(self) -> int:
        return len(self._hashes)

    @property
    def names(self) -> list[str]:
        return sorted(self._hashes)

    def status(self) -> str:
        """"set" or "missing": what /health and logs say instead of any value."""
        return "set" if self._hashes else "missing"

    def authenticate(self, presented: str | None) -> str:
        """The caller's key name, or AccessError(missing_key / invalid_key)."""
        if not presented:
            raise AccessError(MISSING_KEY)
        digest = hash_key(presented)
        found: str | None = None
        for name, stored in self._hashes.items():  # no early exit: same work for any key
            if hmac.compare_digest(digest, stored):
                found = name
        if found is None:
            raise AccessError(INVALID_KEY)
        return found

    @classmethod
    def parse(cls, text: str) -> ApiKeys:
        """Lines or comma-separated ``name:sha256hex``; blank lines and ``#`` comments are
        skipped. Raises ConfigError that says which entry is wrong, never what it holds."""
        hashes: dict[str, str] = {}
        lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
        entries = [part.strip() for line in lines for part in line.split(",")]
        for number, entry in enumerate([e for e in entries if e], 1):
            name, sep, digest = entry.partition(":")
            if not sep or not KEY_NAME_PATTERN.match(name) or not SHA256_HEX.match(digest):
                raise ConfigError(f"API key entry {number} is not name:sha256hex")
            if name in hashes or digest.lower() in hashes.values():
                raise ConfigError(f"API key entry {number} repeats a name or a key")
            hashes[name] = digest.lower()
        return cls(hashes)

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> ApiKeys:
        """From ``JOBPDF_API_KEYS`` and/or the file named by ``JOBPDF_API_KEYS_FILE``.

        With ``environ=None`` the process environment is used, then the repo's ``.env``.
        No keys at all is allowed (every request is then refused) and is logged.
        """
        texts = []
        inline = _setting(API_KEYS_ENV, environ)
        if inline:
            texts.append(inline)
        path = _setting(API_KEYS_FILE_ENV, environ)
        if path:
            try:
                texts.append(Path(path).read_text(encoding="utf-8"))
            except (OSError, UnicodeError) as exc:
                raise ConfigError(
                    f"cannot read the API keys file ({type(exc).__name__})") from None
        keys = cls.parse("\n".join(texts))
        if not keys:
            log.warning("API keys: missing; every API request will be refused")
        else:
            log.info("API keys: set (%d)", len(keys))
        return keys


def _setting(name: str, environ: Mapping[str, str] | None, env_file: Path = ENV_FILE) -> str | None:
    """A setting from ``environ``; with ``environ=None`` from the process environment, then
    ``.env`` (KEY=VALUE lines, like ``database_url``). A real variable always wins."""
    if environ is not None:
        return environ.get(name) or None
    if os.environ.get(name):
        return os.environ[name]
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        key, sep, value = line.strip().partition("=")
        if sep and key.strip() == name and not line.lstrip().startswith("#"):
            return value.strip().strip("'\"") or None
    return None


# --- counters ----------------------------------------------------------------------


@dataclass(frozen=True)
class Rejection:
    rule: RateLimit
    retry_after: int  # whole seconds until the window ends


class LimiterUnavailable(RuntimeError):
    """The counter store cannot be used right now."""


class CounterStore(Protocol):
    def hit(self, scope: str, subject: str, rules: Sequence[RateLimit],
            now: float) -> Rejection | None:
        """Count one upload against every rule at once. Returns None when all rules allow
        it (and all are counted), else the first rule that refuses it (and nothing is
        counted). Raises LimiterUnavailable when the store cannot answer."""


def _window(rule: RateLimit, now: float) -> tuple[int, int]:
    """(first second of the window ``now`` is in, whole seconds until it ends; at least 1)."""
    start = int(now // rule.seconds) * rule.seconds
    return start, max(math.ceil(start + rule.seconds - now), 1)


class InMemoryCounterStore:
    """Process-local counters: for tests and single-process development only. With several
    workers each would have its own counters, so the real limit would be multiplied."""

    def __init__(self) -> None:
        self._hits: dict[tuple[str, str, int, int], int] = {}
        self._lock = threading.Lock()
        self._calls = 0

    def hit(self, scope: str, subject: str, rules: Sequence[RateLimit],
            now: float) -> Rejection | None:
        with self._lock:
            self._calls += 1
            if self._calls % PURGE_EVERY_HITS == 0:
                self._purge(now)
            slots = [(rule, (scope, subject, rule.seconds, _window(rule, now)[0]))
                     for rule in rules]
            for rule, slot in slots:
                if self._hits.get(slot, 0) >= rule.limit:
                    return Rejection(rule, _window(rule, now)[1])
            for _, slot in slots:
                self._hits[slot] = self._hits.get(slot, 0) + 1
            return None

    def _purge(self, now: float) -> None:
        for slot in [s for s in self._hits if s[3] + s[2] < now - PURGE_AFTER_S]:
            del self._hits[slot]

    def __len__(self) -> int:
        return len(self._hits)


_HIT_SQL = (
    "INSERT INTO rate_limit_counters AS c (scope, subject, window_seconds, window_start, hits) "
    "VALUES (%s, %s, %s, %s, 1) "
    "ON CONFLICT (scope, subject, window_seconds, window_start) "
    "DO UPDATE SET hits = c.hits + 1 WHERE c.hits < %s RETURNING hits"
)
_PURGE_SQL = "DELETE FROM rate_limit_counters WHERE window_start + window_seconds < %s"


class PostgresCounterStore:
    """Counters in ``rate_limit_counters``, shared by every worker and kept over restarts.

    Each rule is one atomic upsert that only counts while the window is under its limit,
    inside one transaction for all rules, so concurrent requests cannot overshoot and a
    refusal rolls the other rules' counts back. One autocommit connection behind a lock,
    like the skill index. After a lost connection the store is skipped for
    ``retry_after_s`` seconds (raising LimiterUnavailable at once).
    """

    def __init__(self, url: str, *, retry_after_s: float = STORE_RETRY_AFTER_S,
                 clock: Callable[[], float] = time.monotonic, **connect_kwargs: object) -> None:
        self._url = url
        self._connect_kwargs = connect_kwargs
        self._retry_after_s = retry_after_s
        self._clock = clock
        self._lock = threading.Lock()
        self._conn: psycopg.Connection | None = None
        self._down_until = 0.0
        self._calls = 0
        self._open()  # a misconfiguration shows up at startup

    def hit(self, scope: str, subject: str, rules: Sequence[RateLimit],
            now: float) -> Rejection | None:
        with self._lock:
            if self._clock() < self._down_until:
                raise LimiterUnavailable("counter store paused after a failure")
            conn = self._conn
            try:
                if conn is None or _unusable(conn):
                    conn = self._open()
                rejection = self._count(conn, scope, subject, rules, now)
                self._calls += 1
                if self._calls % PURGE_EVERY_HITS == 0:
                    conn.execute(_PURGE_SQL, [int(now - PURGE_AFTER_S)])
                return rejection
            except psycopg.Error as exc:
                if conn is not None and not _unusable(conn):
                    log.warning("rate limit statement failed (%s)", type(exc).__name__)
                else:
                    log.warning("rate limit database unavailable (%s); pausing %.0f s",
                                type(exc).__name__, self._retry_after_s)
                    self._down_until = self._clock() + self._retry_after_s
                    self._discard_connection()
                raise LimiterUnavailable(type(exc).__name__) from None

    @staticmethod
    def _count(conn: psycopg.Connection, scope: str, subject: str,
               rules: Sequence[RateLimit], now: float) -> Rejection | None:
        rejection: Rejection | None = None
        with conn.transaction():
            for rule in rules:
                start, wait = _window(rule, now)
                row = conn.execute(
                    _HIT_SQL, [scope, subject, rule.seconds, start, rule.limit]).fetchone()
                if row is None:  # at the limit: the WHERE refused the update
                    rejection = Rejection(rule, wait)
                    raise psycopg.Rollback  # un-count the rules already counted
        return rejection

    def purge(self, now: float) -> int:
        """Delete counters whose window ended more than ``PURGE_AFTER_S`` ago."""
        with self._lock:
            conn = self._conn if self._conn is not None else self._open()
            return conn.execute(_PURGE_SQL, [int(now - PURGE_AFTER_S)]).rowcount

    def close(self) -> None:
        with self._lock:
            conn, self._conn = self._conn, None
        if conn is not None:
            conn.close()

    def _open(self) -> psycopg.Connection:
        kwargs = {"connect_timeout": CONNECT_TIMEOUT_S, **self._connect_kwargs}
        conn = psycopg.connect(self._url, autocommit=True, **kwargs)  # type: ignore[arg-type]
        try:
            apply_schema(conn, RATE_LIMIT_SCHEMA_FILE)  # idempotent CREATE TABLE IF NOT EXISTS
        except psycopg.Error:
            conn.close()
            raise
        self._conn = conn
        return conn

    def _discard_connection(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except psycopg.Error:
                pass


def _unusable(conn: psycopg.Connection) -> bool:
    return conn.closed or conn.broken


# --- the limiter -------------------------------------------------------------------


class RateLimiter:
    """Per-key and per-IP limits over a ``CounterStore``."""

    def __init__(self, store: CounterStore, *, ip_secret: str,
                 key_limits: Sequence[RateLimit] = KEY_LIMITS,
                 ip_limits: Sequence[RateLimit] = IP_LIMITS,
                 clock: Callable[[], float] = time.time, fail_open: bool = False) -> None:
        if len(ip_secret) < MIN_SECRET_CHARS:
            raise ConfigError(f"the rate-limit secret needs at least {MIN_SECRET_CHARS} characters")
        self._store = store
        self._ip_secret = ip_secret.encode("utf-8")
        self._key_limits = tuple(key_limits)
        self._ip_limits = tuple(ip_limits)
        self._clock = clock
        self._fail_open = fail_open

    def check_key(self, key_name: str) -> None:
        """Count an upload by this API key, or raise AccessError(rate_limited)."""
        self._hit("key", key_name, self._key_limits)

    def check_ip(self, client_ip: str) -> None:
        """Count an upload from this address, or raise AccessError(rate_limited)."""
        self._hit("ip", self.ip_subject(client_ip), self._ip_limits)

    def ip_subject(self, client_ip: str) -> str:
        """What is stored for an address: a keyed hash of its network (IPv6: the /64)."""
        address = ipaddress.ip_address(client_ip.strip())  # ValueError if it is not one
        if isinstance(address, ipaddress.IPv6Address):
            if address.ipv4_mapped is not None:
                address = address.ipv4_mapped
            else:
                address = ipaddress.ip_network(f"{address}/64", strict=False).network_address
        return hmac.new(self._ip_secret, str(address).encode("ascii"), hashlib.sha256).hexdigest()

    def _hit(self, scope: str, subject: str, rules: Sequence[RateLimit]) -> None:
        try:
            rejection = self._store.hit(scope, subject, rules, self._clock())
        except LimiterUnavailable:
            if self._fail_open:
                log.warning("rate limit store unavailable; letting the request through")
                return
            raise AccessError(LIMITER_UNAVAILABLE) from None
        if rejection is not None:
            if scope == "key":
                log.info("rate limited: key=%s window=%s", subject, rejection.rule.name)
            else:
                log.info("rate limited: ip window=%s", rejection.rule.name)  # never the address
            raise AccessError(RATE_LIMITED, retry_after=rejection.retry_after)


# --- size --------------------------------------------------------------------------


def check_declared_size(content_length: int | None, max_bytes: int = MAX_UPLOAD_BYTES) -> None:
    """Refuse early from the declared body length. A multipart body carries a little
    overhead, so only a body clearly over the limit is refused here; ``read_upload``
    enforces the exact limit on the file itself."""
    if content_length is not None and content_length > max_bytes + BODY_OVERHEAD_BYTES:
        raise AccessError(TOO_LARGE)


def read_upload(file: BinaryIO, max_bytes: int = MAX_UPLOAD_BYTES) -> bytes:
    """All bytes of an upload, reading at most ``max_bytes`` + 1 so a huge body is never
    held in memory; AccessError(too_large) when it is over the limit."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = file.read(min(READ_CHUNK_BYTES, max_bytes + 1 - total))
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > max_bytes:
            raise AccessError(TOO_LARGE)
        chunks.append(chunk)


# --- putting it together -----------------------------------------------------------


@dataclass(frozen=True)
class Caller:
    kind: Literal["api_key", "web"]
    name: str | None  # the API key's name; None for the web form


class AccessControl:
    """What the endpoints call before touching an upload."""

    def __init__(self, keys: ApiKeys, limiter: RateLimiter, *,
                 max_bytes: int = MAX_UPLOAD_BYTES) -> None:
        self.keys = keys
        self.limiter = limiter
        self.max_bytes = max_bytes

    def authorize_api(self, headers: Mapping[str, str], content_length: int | None) -> Caller:
        """Key first (an unknown caller learns nothing else), then size, then the limit, so
        an oversized upload does not use up quota."""
        name = self.keys.authenticate(api_key_from_headers(headers))
        check_declared_size(content_length, self.max_bytes)
        self.limiter.check_key(name)
        return Caller("api_key", name)

    def authorize_web(self, client_ip: str, content_length: int | None) -> Caller:
        """The web form has no key: size, then the per-IP limit."""
        check_declared_size(content_length, self.max_bytes)
        self.limiter.check_ip(client_ip)
        return Caller("web", None)

    def read_upload(self, file: BinaryIO) -> bytes:
        return read_upload(file, self.max_bytes)


def build_access_control(
    environ: Mapping[str, str] | None = None, *, store: CounterStore | None = None,
) -> AccessControl:
    """From the environment: keys, the IP-hash secret and (unless ``store`` is given) a
    Postgres counter store on DATABASE_URL. Raises ConfigError, or psycopg.Error when the
    database is unreachable, so a bad setup stops at startup instead of at the first upload."""
    secret = _setting(RATE_LIMIT_SECRET_ENV, environ)
    if not secret:
        raise ConfigError(f"{RATE_LIMIT_SECRET_ENV} is missing")
    keys = ApiKeys.from_environment(environ)
    if store is None:
        url = database_url()
        if not url:
            raise ConfigError("DATABASE_URL is missing; rate limits need Postgres")
        store = PostgresCounterStore(url)
    return AccessControl(keys, RateLimiter(store, ip_secret=secret))
