import json
from pathlib import Path

import pytest

from jobpdf.normalization.coverage import load_jobs, measure, top_tags

PAYLOAD = {
    "jobs": [
        {"title": "Fake job 1", "tags": ["Python", "Docker", "Kubernetes"]},
        {"title": "Fake job 2", "tags": ["python", " PYTHON ", "dbt"]},
        {"title": "Fake job 3", "tags": ["Docker", ""]},
        {"title": "Fake job 4", "tags": None},
    ]
}


def test_top_tags_merge_spellings_and_sort_by_frequency() -> None:
    tags = top_tags(PAYLOAD)

    assert [(t.tag_norm, t.count) for t in tags] == [
        ("python", 3),
        ("docker", 2),
        ("dbt", 1),
        ("kubernetes", 1),
    ]
    assert tags[0].tag in {"Python", "python", "PYTHON"}
    assert len(top_tags(PAYLOAD, n=2)) == 2


def test_measure_coverage() -> None:
    report = measure(PAYLOAD, {"python", "docker"})

    assert [t.tag_norm for t in report.misses] == ["dbt", "kubernetes"]
    assert report.coverage == pytest.approx(0.5)
    assert report.weighted_coverage == pytest.approx(5 / 7)
    assert report.jobs == 4


def never_called() -> bytes:
    raise AssertionError("must not hit the Remotive API")


def test_cache_is_reused(tmp_path: Path) -> None:
    cache = tmp_path / "remotive.json"
    cache.write_text(json.dumps(PAYLOAD), encoding="utf-8")

    assert load_jobs(cache, fetch=never_called) == PAYLOAD


def test_fetch_writes_cache_once(tmp_path: Path) -> None:
    cache = tmp_path / "cache" / "remotive.json"
    raw = json.dumps(PAYLOAD).encode()

    assert load_jobs(cache, fetch=lambda: raw) == PAYLOAD
    assert cache.read_bytes() == raw
    assert load_jobs(cache, fetch=never_called) == PAYLOAD
    assert load_jobs(cache, refresh=True, fetch=lambda: raw) == PAYLOAD
