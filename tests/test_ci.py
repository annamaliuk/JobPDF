"""JM-37: the CI pipeline and the files it depends on.

The workflow cannot run here, so these tests keep it honest: its database guard works, and
the things it names (image targets, DAG ids, versions) still exist where it expects them.
"""

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
DOCKERFILE = ROOT / "Dockerfile"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def junit(*cases: str) -> str:
    """A pytest-style junit report; each case is "run" or "skip"."""
    body = "".join(
        f'<testcase classname="t" name="n{i}">' + ("<skipped/>" if kind == "skip" else "")
        + "</testcase>"
        for i, kind in enumerate(cases)
    )
    return f'<?xml version="1.0"?><testsuites><testsuite>{body}</testsuite></testsuites>'


@pytest.fixture(scope="module")
def guard():
    return load_script("check_db_tests_ran")


def report(tmp_path, *cases: str) -> Path:
    path = tmp_path / "junit.xml"
    path.write_text(junit(*cases), encoding="utf-8")
    return path


# --- the database-tests guard ----------------------------------------------------------------


def test_tests_that_ran_are_counted_and_skipped_ones_are_not(guard, tmp_path) -> None:
    assert guard.count_tests(report(tmp_path, "run", "skip", "run", "skip", "skip")) == (5, 2)


def test_enough_tests_running_passes(guard, tmp_path, capsys) -> None:
    code = guard.main([str(report(tmp_path, "run", "run", "skip")), "--min", "2"])

    assert code == 0 and "2 of 3 database tests ran" in capsys.readouterr().out


def test_too_few_running_fails_even_if_nothing_failed(guard, tmp_path) -> None:
    assert guard.main([str(report(tmp_path, "run", "skip", "skip")), "--min", "2"]) == 1


def test_a_run_where_everything_skipped_fails_with_the_default_floor(guard, tmp_path) -> None:
    assert guard.main([str(report(tmp_path, *["skip"] * 60))]) == 1


def test_the_default_floor_is_met_by_enough_running_tests(guard, tmp_path) -> None:
    assert guard.main([str(report(tmp_path, *["run"] * guard.MIN_DB_TESTS))]) == 0
    assert guard.main([str(report(tmp_path, *["run"] * (guard.MIN_DB_TESTS - 1)))]) == 1


def test_the_floor_is_the_number_the_readme_promises(guard) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert f"fewer than {guard.MIN_DB_TESTS} of them actually ran" in readme


def test_an_empty_report_fails(guard, tmp_path) -> None:
    assert guard.main([str(report(tmp_path))]) == 1


@pytest.mark.parametrize("content", [None, "", "not xml <"])
def test_an_unreadable_report_is_exit_2_with_no_traceback(guard, tmp_path, capsys, content) -> None:
    path = tmp_path / "junit.xml"
    if content is not None:
        path.write_text(content)

    assert guard.main([str(path)]) == 2
    assert "cannot read the test report" in capsys.readouterr().out


# --- the workflow and what it names ------------------------------------------------------------


@pytest.fixture(scope="module")
def workflow() -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def steps_text(workflow: dict, job: str) -> str:
    return "\n".join(
        str(value) for step in workflow["jobs"][job]["steps"] for value in step.values()
    ) + str(workflow["jobs"][job].get("strategy", ""))


def test_the_three_jobs_exist_and_no_secret_is_used(workflow) -> None:
    assert set(workflow["jobs"]) == {"test", "dags", "docker"}
    assert "secrets." not in WORKFLOW.read_text(encoding="utf-8")


def test_the_workflow_runs_on_pull_requests_and_main_with_read_only_rights(workflow) -> None:
    triggers = workflow.get("on", workflow.get(True))  # YAML reads a bare `on` as True

    assert "pull_request" in triggers and triggers["push"]["branches"] == ["main"]
    assert workflow["permissions"] == {"contents": "read"}


def test_every_image_target_it_builds_exists_in_the_dockerfile(workflow) -> None:
    targets = [step["with"]["target"] for step in workflow["jobs"]["docker"]["steps"]
               if "target" in step.get("with", {})]
    stages = re.findall(r"^FROM .* AS (\w+)$", DOCKERFILE.read_text(), flags=re.MULTILINE)

    assert sorted(targets) == ["airflow", "app"] and set(targets) <= set(stages)


def test_the_dag_ids_it_expects_are_the_ones_in_dags() -> None:
    named = re.search(r"assert set\(bag\.dags\) == \{([^}]*)\}", WORKFLOW.read_text()).group(1)
    expected = set(re.findall(r"'([^']+)'", named))
    found = {
        match
        for path in (ROOT / "dags").glob("*.py")
        for match in re.findall(r'dag_id="([^"]+)"', path.read_text())
    }

    assert expected == found


def test_the_airflow_version_is_the_one_in_the_dockerfile(workflow) -> None:
    in_image = re.search(r"apache/airflow:([\d.]+)-python", DOCKERFILE.read_text()).group(1)
    in_ci = steps_text(workflow, "dags")

    assert f"apache-airflow=={in_image}" in in_ci and f"constraints-{in_image}" in in_ci


def test_the_python_versions_cover_the_minimum_and_the_image(workflow) -> None:
    matrix = workflow["jobs"]["test"]["strategy"]["matrix"]["python"]
    minimum = re.search(r'requires-python = ">=([\d.]+)"',
                        (ROOT / "pyproject.toml").read_text()).group(1)
    image = re.search(r"FROM python:([\d.]+)-slim", DOCKERFILE.read_text()).group(1)

    assert minimum in matrix and image in matrix


def test_the_database_tests_are_followed_by_the_guard(workflow) -> None:
    commands = [step.get("run", "") for step in workflow["jobs"]["test"]["steps"]]

    run = next(i for i, c in enumerate(commands) if "pytest -m db" in c)
    guard = next(i for i, c in enumerate(commands) if "check_db_tests_ran.py junit-db.xml" in c)
    rest = next(i for i, c in enumerate(commands) if 'pytest -m "not db"' in c)
    assert run < guard and "--junitxml=junit-db.xml" in commands[run] and guard < rest
    assert "uv sync --locked" in "\n".join(commands) and "ruff check" in "\n".join(commands)


def test_the_database_service_matches_the_url_the_tests_read(workflow) -> None:
    job = workflow["jobs"]["test"]
    env = job["services"]["db"]["env"]

    assert job["services"]["db"]["image"].startswith("pgvector/pgvector")
    assert job["env"]["DATABASE_URL"] == (
        f"postgresql://{env['POSTGRES_USER']}:{env['POSTGRES_PASSWORD']}@localhost:5432/"
        f"{env['POSTGRES_DB']}")
