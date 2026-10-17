"""JM-38: versioned SQL migrations.

The file-handling tests need nothing. The ones marked ``db`` run against Postgres through
DATABASE_URL, each in a throwaway schema, and skip without it.
"""

import importlib.util
import logging
import threading
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import psycopg
import pytest

from jobpdf import migrations
from jobpdf.migrations import (
    ADVISORY_LOCK_KEY,
    SQL_DIR,
    MigrationError,
    checksum,
    discover,
    migrate,
    pending,
    status,
)
from jobpdf.normalization.db import database_url

ROOT = Path(__file__).resolve().parents[1]
SECRET = "hunter2-SECRET-VALUE"


def load_script():
    spec = importlib.util.spec_from_file_location("migrate_script", ROOT / "scripts" / "migrate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(folder: Path, name: str, sql: str) -> Path:
    path = folder / name
    path.write_text(sql, encoding="utf-8", newline="")
    return path


@pytest.fixture
def folder(tmp_path) -> Path:
    path = tmp_path / "sql"
    path.mkdir()
    return path


def a_b_c(folder: Path) -> None:
    write(folder, "001_alpha.sql", "CREATE TABLE alpha (id int PRIMARY KEY);\n")
    write(folder, "002_beta.sql", "CREATE TABLE beta (id int PRIMARY KEY);\n"
          "INSERT INTO beta VALUES (1);\n")
    write(folder, "010_gamma.sql", "CREATE TABLE gamma (id int PRIMARY KEY);\n")


# --- reading the files -------------------------------------------------------------------------


def test_files_are_found_oldest_first_by_number_not_by_text(folder) -> None:
    for name in ("010_ten.sql", "009_nine.sql", "100_hundred.sql", "002_two.sql"):
        write(folder, name, "SELECT 1;")
    write(folder, "notes.txt", "not a migration")

    found = discover(folder)

    assert [m.label for m in found] == ["002_two", "009_nine", "010_ten", "100_hundred"]
    assert found[0].version == "002" and found[0].name == "two"


def test_four_digit_numbers_come_after_three_digit_ones(folder) -> None:
    for name in ("1000_late.sql", "999_early.sql", "0998_padded.sql"):
        write(folder, name, "SELECT 1;")

    assert [m.version for m in discover(folder)] == ["0998", "999", "1000"]


def test_gaps_in_the_numbering_are_fine(folder) -> None:
    write(folder, "001_a.sql", "SELECT 1;")
    write(folder, "004_d.sql", "SELECT 1;")

    assert [m.version for m in discover(folder)] == ["001", "004"]


@pytest.mark.parametrize("name", ["1_short.sql", "01_two_digits.sql", "001-dash.sql",
                                  "001_Upper.sql", "001_.sql", "readme.sql", "001_a b.sql"])
def test_a_sql_file_with_a_wrong_name_is_an_error_not_skipped(folder, name) -> None:
    write(folder, name, "SELECT 1;")

    with pytest.raises(MigrationError) as raised:
        discover(folder)

    assert raised.value.code == "bad_name" and raised.value.version == name


def test_two_files_with_the_same_number_are_an_error(folder) -> None:
    write(folder, "001_a.sql", "SELECT 1;")
    write(folder, "0001_b.sql", "SELECT 1;")

    with pytest.raises(MigrationError) as raised:
        discover(folder)

    assert (raised.value.code, raised.value.version) == ("duplicate_version", "001")


def test_a_missing_folder_is_an_error_not_an_empty_list(tmp_path) -> None:
    with pytest.raises(MigrationError) as raised:
        discover(tmp_path / "nope")

    assert raised.value.code == "no_directory"


def test_an_empty_folder_has_no_migrations(folder) -> None:
    assert discover(folder) == []


def test_the_checksum_ignores_line_endings_but_not_content() -> None:
    assert checksum("a\nb\n") == checksum("a\r\nb\r\n") == checksum("a\rb\r")
    assert checksum("a\nb\n") != checksum("a\nb")
    assert checksum("a\nb\n") != checksum("a\nc\n")
    assert len(checksum("x")) == 64


def test_the_repository_migrations_are_valid() -> None:
    found = discover(SQL_DIR)

    assert [m.version for m in found][:1] == ["001"]
    assert len({m.version for m in found}) == len(found)


# --- the database -------------------------------------------------------------------------------


@pytest.fixture
def make_conn() -> Iterator[Callable[[], psycopg.Connection]]:
    """Connections that all see one throwaway schema."""
    url = database_url()
    if not url:
        pytest.skip("DATABASE_URL not set (environment or .env)")
    name = f"test_{uuid.uuid4().hex[:12]}"
    try:
        admin = psycopg.connect(url, autocommit=True, connect_timeout=10)
    except psycopg.OperationalError as exc:
        pytest.skip(f"database not reachable: {type(exc).__name__}")
    with admin:
        admin.execute(f'CREATE SCHEMA "{name}"')
    opened: list[psycopg.Connection] = []

    def make() -> psycopg.Connection:
        # public stays on the path: once another test has created the pgvector extension
        # there, CREATE EXTENSION IF NOT EXISTS is a no-op and its types live in public.
        conn = psycopg.connect(url, options=f"-c search_path={name},public", connect_timeout=10)
        opened.append(conn)
        return conn

    yield make
    for conn in opened:
        if not conn.closed:
            conn.close()
    with psycopg.connect(url, autocommit=True, connect_timeout=10) as admin:
        admin.execute(f'DROP SCHEMA "{name}" CASCADE')


@pytest.fixture
def conn(make_conn) -> psycopg.Connection:
    return make_conn()


def tables(conn: psycopg.Connection) -> set[str]:
    rows = conn.execute("SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = current_schema()").fetchall()
    conn.rollback()
    return {r[0] for r in rows}


def lock_is_free(make_conn) -> bool:
    other = make_conn()
    free = other.execute("SELECT pg_try_advisory_lock(%s)", (ADVISORY_LOCK_KEY,)).fetchone()[0]
    if free:
        other.execute("SELECT pg_advisory_unlock(%s)", (ADVISORY_LOCK_KEY,))
    other.commit()
    return free


@pytest.mark.db
def test_all_migrations_are_applied_in_order_and_recorded(conn, folder) -> None:
    a_b_c(folder)

    result = migrate(conn, folder)

    assert result.applied == ("001", "002", "010") and result.already == ()
    assert {"alpha", "beta", "gamma", "schema_migrations"} <= tables(conn)
    rows = conn.execute("SELECT version, name, checksum FROM schema_migrations "
                        "ORDER BY applied_at, version").fetchall()
    assert rows == [(m.version, m.name, m.checksum) for m in discover(folder)]


@pytest.mark.db
def test_a_second_run_changes_nothing(conn, folder) -> None:
    a_b_c(folder)
    migrate(conn, folder)

    again = migrate(conn, folder)

    assert again.applied == () and again.already == ("001", "002", "010")
    assert conn.execute("SELECT count(*) FROM beta").fetchone()[0] == 1  # not run twice


@pytest.mark.db
def test_a_new_file_is_applied_alone(conn, folder) -> None:
    a_b_c(folder)
    migrate(conn, folder)
    write(folder, "011_delta.sql", "CREATE TABLE delta (id int);")

    result = migrate(conn, folder)

    assert result.applied == ("011",) and "delta" in tables(conn)


@pytest.mark.db
def test_an_older_pending_migration_is_still_applied(conn, folder) -> None:
    write(folder, "001_a.sql", "CREATE TABLE a (id int);")
    write(folder, "003_c.sql", "CREATE TABLE c (id int);")
    migrate(conn, folder)
    write(folder, "002_b.sql", "CREATE TABLE b (id int);")  # arrives after 003 is out

    result = migrate(conn, folder)

    assert result.applied == ("002",) and "b" in tables(conn)


@pytest.mark.db
def test_a_failing_migration_rolls_back_whole_and_keeps_earlier_ones(
        conn, folder, caplog) -> None:
    write(folder, "001_ok.sql", "CREATE TABLE ok (id int);")
    write(folder, "002_bad.sql",
          f"CREATE TABLE half (id int);\nSELECT '{SECRET}'::int;\n")  # fails after the table
    write(folder, "003_later.sql", "CREATE TABLE later (id int);")

    with caplog.at_level(logging.DEBUG), pytest.raises(MigrationError) as raised:
        migrate(conn, folder)

    assert (raised.value.code, raised.value.version) == ("migration_failed", "002")
    assert SECRET not in str(raised.value) and SECRET not in caplog.text
    assert raised.value.__cause__ is None and raised.value.__suppress_context__  # no chained text
    assert "DataError" in caplog.text or "InvalidTextRepresentation" in caplog.text
    present = tables(conn)
    assert "ok" in present and "half" not in present and "later" not in present
    rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
    assert [r[0] for r in rows] == ["001"]


@pytest.mark.db
def test_after_fixing_the_failed_file_the_run_continues(conn, folder) -> None:
    write(folder, "001_ok.sql", "CREATE TABLE ok (id int);")
    bad = write(folder, "002_bad.sql", "SELECT 'x'::int;")
    with pytest.raises(MigrationError):
        migrate(conn, folder)
    bad.write_text("CREATE TABLE fixed (id int);")  # never applied, so it may be edited

    result = migrate(conn, folder)

    assert result.applied == ("002",) and result.already == ("001",)


@pytest.mark.db
def test_an_applied_file_that_was_edited_stops_the_run_before_anything_is_applied(
        conn, folder) -> None:
    write(folder, "001_a.sql", "CREATE TABLE a (id int);")
    migrate(conn, folder)
    write(folder, "001_a.sql", "CREATE TABLE a (id int, extra int);")
    write(folder, "002_b.sql", "CREATE TABLE b (id int);")

    with pytest.raises(MigrationError) as raised:
        migrate(conn, folder)

    assert (raised.value.code, raised.value.version) == ("checksum_mismatch", "001")
    assert "b" not in tables(conn)


@pytest.mark.db
def test_an_applied_file_that_was_removed_stops_the_run(conn, folder) -> None:
    gone = write(folder, "001_a.sql", "CREATE TABLE a (id int);")
    migrate(conn, folder)
    gone.unlink()
    write(folder, "002_b.sql", "CREATE TABLE b (id int);")

    with pytest.raises(MigrationError) as raised:
        migrate(conn, folder)

    assert (raised.value.code, raised.value.version) == ("missing_file", "001")
    assert "b" not in tables(conn)


@pytest.mark.db
def test_windows_line_endings_do_not_look_like_an_edit(conn, folder) -> None:
    path = write(folder, "001_a.sql", "CREATE TABLE a (\n  id int\n);\n")
    migrate(conn, folder)
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))

    assert migrate(conn, folder).applied == ()
    assert [s.state for s in status(conn, folder)] == ["applied"]


@pytest.mark.db
def test_status_lists_every_state_and_changes_nothing(conn, folder) -> None:
    write(folder, "001_kept.sql", "CREATE TABLE kept (id int);")
    write(folder, "002_edited.sql", "CREATE TABLE edited (id int);")
    write(folder, "003_removed.sql", "CREATE TABLE removed (id int);")
    migrate(conn, folder)
    write(folder, "002_edited.sql", "CREATE TABLE edited (id int, more int);")
    (folder / "003_removed.sql").unlink()
    write(folder, "004_new.sql", "CREATE TABLE new_one (id int);")

    rows = status(conn, folder)

    assert [(r.version, r.name, r.state) for r in rows] == [
        ("001", "kept", "applied"), ("002", "edited", "changed"),
        ("003", "removed", "missing"), ("004", "new", "pending")]
    assert pending(conn, folder) == ["004"] and "new_one" not in tables(conn)


@pytest.mark.db
def test_status_orders_by_number_and_the_lowest_broken_one_is_reported(conn, folder) -> None:
    write(folder, "999_nine.sql", "CREATE TABLE nine (id int);")
    write(folder, "1000_ten.sql", "CREATE TABLE ten (id int);")
    migrate(conn, folder)
    assert [r.version for r in status(conn, folder)] == ["999", "1000"]

    write(folder, "999_nine.sql", "CREATE TABLE nine (id int, x int);")
    write(folder, "1000_ten.sql", "CREATE TABLE ten (id int, x int);")
    (folder / "1000_ten.sql").unlink()  # one removed, one edited
    with pytest.raises(MigrationError) as raised:
        migrate(conn, folder)

    assert (raised.value.code, raised.value.version) == ("checksum_mismatch", "999")
    assert [r.version for r in status(conn, folder)] == ["999", "1000"]


@pytest.mark.db
def test_status_before_any_run_creates_nothing(conn, folder) -> None:
    a_b_c(folder)

    assert [r.state for r in status(conn, folder)] == ["pending"] * 3
    assert tables(conn) == set()


@pytest.mark.db
def test_concurrent_runs_apply_each_migration_exactly_once(make_conn, folder) -> None:
    # No IF NOT EXISTS: a second application would fail.
    write(folder, "001_once.sql", "CREATE TABLE once (id int PRIMARY KEY);\n"
          "INSERT INTO once VALUES (1);")
    write(folder, "002_twice.sql", "CREATE TABLE twice (id int PRIMARY KEY);")
    start = threading.Barrier(4)
    outcomes: list = []

    def run() -> None:
        own = make_conn()
        start.wait()
        try:
            outcomes.append(migrate(own, folder))
        except Exception as exc:  # noqa: BLE001 - the test reports whatever happened
            outcomes.append(exc)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)

    assert all(isinstance(o, migrations.MigrationResult) for o in outcomes), outcomes
    assert sorted(len(o.applied) for o in outcomes) == [0, 0, 0, 2]
    assert make_conn().execute("SELECT count(*) FROM once").fetchone()[0] == 1


@pytest.mark.db
def test_a_run_gives_up_when_another_holds_the_lock_too_long(make_conn, folder) -> None:
    a_b_c(folder)
    holder = make_conn()
    holder.execute("SELECT pg_advisory_lock(%s)", (ADVISORY_LOCK_KEY,))
    holder.commit()

    with pytest.raises(MigrationError) as raised:
        migrate(make_conn(), folder, lock_timeout_ms=200)

    assert raised.value.code == "lock_timeout"
    assert tables(make_conn()) == set()  # nothing was started
    holder.execute("SELECT pg_advisory_unlock(%s)", (ADVISORY_LOCK_KEY,))
    holder.commit()
    assert migrate(make_conn(), folder).applied == ("001", "002", "010")


@pytest.mark.db
def test_the_lock_is_released_after_success_and_after_failure(make_conn, folder) -> None:
    write(folder, "001_ok.sql", "CREATE TABLE ok (id int);")
    migrate(make_conn(), folder)
    assert lock_is_free(make_conn)

    write(folder, "002_bad.sql", "SELECT 'x'::int;")
    with pytest.raises(MigrationError):
        migrate(make_conn(), folder)
    assert lock_is_free(make_conn)


@pytest.mark.db
def test_the_lock_check_after_a_dead_connection_does_not_raise(make_conn, folder) -> None:
    write(folder, "001_ok.sql", "CREATE TABLE ok (id int);")
    conn = make_conn()
    migrate(conn, folder)
    conn.close()

    migrations._unlock(conn)  # the server already released it with the session


@pytest.mark.db
def test_the_repository_rate_limit_and_vacancy_migrations_apply(conn, tmp_path) -> None:
    only = tmp_path / "sql"
    only.mkdir()
    for path in SQL_DIR.glob("00[34]_*.sql"):
        write(only, path.name, path.read_text(encoding="utf-8"))

    result = migrate(conn, only)

    assert len(result.applied) == 2
    assert {"rate_limit_counters", "vacancies"} <= tables(conn)
    assert migrate(conn, only).applied == ()


@pytest.mark.db
def test_the_whole_repository_set_applies_where_pgvector_exists(conn) -> None:
    if not conn.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'").fetchone():
        pytest.skip("the pgvector extension is not installed")
    conn.rollback()

    result = migrate(conn)

    assert result.applied == tuple(m.version for m in discover(SQL_DIR))
    assert {"taxonomy_alias", "rate_limit_counters", "vacancies"} <= tables(conn)
    assert migrate(conn).applied == ()


# --- the script ---------------------------------------------------------------------------------


@pytest.fixture
def script():
    return load_script()


def run_script(script, make_conn, folder, *args) -> int:
    return script.main(["--dir", str(folder), *args], connect=lambda url: make_conn())


@pytest.mark.db
def test_the_script_applies_and_reports_counts(
        script, make_conn, folder, monkeypatch, capsys) -> None:
    monkeypatch.setattr(script, "database_url", lambda: "postgresql://x")
    a_b_c(folder)

    assert run_script(script, make_conn, folder) == 0
    first = capsys.readouterr().out
    assert "applied 001" in first and "3 applied, 0 already up to date" in first

    assert run_script(script, make_conn, folder) == 0
    assert "0 applied, 3 already up to date" in capsys.readouterr().out


@pytest.mark.db
def test_the_script_lists_status_and_check_fails_while_something_is_pending(
        script, make_conn, folder, monkeypatch, capsys) -> None:
    monkeypatch.setattr(script, "database_url", lambda: "postgresql://x")
    a_b_c(folder)

    assert run_script(script, make_conn, folder, "--check") == 1
    out = capsys.readouterr().out
    assert out.count("pending") == 3 and "alpha" in out
    assert run_script(script, make_conn, folder, "--status") == 0
    capsys.readouterr()

    assert run_script(script, make_conn, folder) == 0
    capsys.readouterr()
    assert run_script(script, make_conn, folder, "--check") == 0
    assert capsys.readouterr().out.count("applied") == 3


@pytest.mark.db
def test_the_script_says_which_migration_stopped_and_nothing_else(
        script, make_conn, folder, monkeypatch, capsys) -> None:
    monkeypatch.setattr(script, "database_url", lambda: "postgresql://x")
    write(folder, "001_bad.sql", f"SELECT '{SECRET}'::int;")

    assert run_script(script, make_conn, folder) == 1
    out = capsys.readouterr().out
    assert "migration stopped: migration_failed 001" in out and SECRET not in out


def test_without_a_database_url_the_script_exits_2(script, monkeypatch, capsys) -> None:
    monkeypatch.setattr(script, "database_url", lambda: None)

    assert script.main([]) == 2
    assert "DATABASE_URL is not set" in capsys.readouterr().out


def test_an_unreachable_database_is_reported_without_the_address(
        script, monkeypatch, capsys) -> None:
    monkeypatch.setattr(script, "database_url", lambda: f"postgresql://u:{SECRET}@nowhere/db")

    def refuse(url: str):
        raise psycopg.OperationalError(f"could not connect to {url}")

    assert script.main([], connect=refuse) == 1
    out = capsys.readouterr().out
    assert out.strip() == "database unreachable" and SECRET not in out


class FakeConn:
    """Stands in for a connection; ``fail`` is raised by every query."""

    def __init__(self, fail: Exception | None = None) -> None:
        self.fail = fail
        self.exited = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.exited = True
        return False

    def execute(self, *args, **kwargs):
        raise self.fail

    def rollback(self) -> None:
        pass


def test_a_bad_folder_is_a_stopped_migration_not_a_traceback(
        script, monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setattr(script, "database_url", lambda: "postgresql://x")
    fake = FakeConn()

    code = script.main(["--dir", str(tmp_path / "nope")], connect=lambda url: fake)

    assert code == 1 and "migration stopped: no_directory" in capsys.readouterr().out
    assert fake.exited  # the connection is closed on every path


@pytest.mark.parametrize("args", [[], ["--status"], ["--check"]])
def test_a_database_error_during_the_run_is_reported_without_its_text(
        script, monkeypatch, capsys, folder, args) -> None:
    monkeypatch.setattr(script, "database_url", lambda: "postgresql://x")
    a_b_c(folder)
    fake = FakeConn(psycopg.OperationalError(f"server said {SECRET}"))

    code = script.main(["--dir", str(folder), *args], connect=lambda url: fake)

    assert code == 1 and capsys.readouterr().out.strip() == "database error"
    assert fake.exited


@pytest.mark.db
@pytest.mark.parametrize("damage", ["edited", "removed"])
def test_check_fails_for_an_applied_file_that_changed_or_vanished(
        script, make_conn, folder, monkeypatch, capsys, damage) -> None:
    monkeypatch.setattr(script, "database_url", lambda: "postgresql://x")
    a_b_c(folder)
    assert run_script(script, make_conn, folder) == 0
    if damage == "edited":
        write(folder, "001_alpha.sql", "CREATE TABLE alpha (id int, x int);")
    else:
        (folder / "001_alpha.sql").unlink()
    capsys.readouterr()

    assert run_script(script, make_conn, folder, "--check") == 1
    assert ("changed" if damage == "edited" else "missing") in capsys.readouterr().out


def test_check_and_status_cannot_be_combined(script) -> None:
    with pytest.raises(SystemExit) as raised:
        script.main(["--check", "--status"])

    assert raised.value.code == 2


# --- deployment wiring --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compose() -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))["services"]


def test_compose_runs_the_migration_script_once_and_keeps_secrets_out(compose) -> None:
    service = compose["migrate"]

    assert service["command"] == ["python", "scripts/migrate.py"]
    assert service["restart"] == "no" and "profiles" not in service
    assert set(service["environment"]) == {"DATABASE_URL"}  # no API keys, no secrets
    assert service["read_only"] is True and service["cap_drop"] == ["ALL"]
    assert service["depends_on"]["db"]["condition"] == "service_healthy"
    assert (ROOT / "scripts" / "migrate.py").is_file()


@pytest.mark.parametrize("name", ["app", "airflow-init"])
def test_the_app_and_airflow_wait_for_the_migrations_to_finish_successfully(compose, name) -> None:
    assert compose[name]["depends_on"]["migrate"]["condition"] == "service_completed_successfully"


def test_the_image_contains_what_the_migration_script_needs() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    for folder in ("jobpdf", "scripts", "sql"):
        assert f"COPY {folder} ./{folder}" in dockerfile
