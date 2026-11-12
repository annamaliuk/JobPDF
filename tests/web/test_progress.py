"""JM-44: progress steps while a CV is analysed. The real pipeline runs on a fixture CV with a
scripted LLM, as in the other web tests, so nothing here needs a network, a key or a model."""

import asyncio
import json
import logging
import threading

import pytest
from fastapi.testclient import TestClient

from jobpdf import pipeline
from jobpdf.api import app as api
from jobpdf.api.app import Services, create_app
from jobpdf.pipeline import MAX_UPLOAD_BYTES, PipelineError
from jobpdf.web import progress
from jobpdf.web.progress import AnalysisStream, Outcome
from tests.api.test_app import analyze_offline, make_services
from tests.normalization.fakes import FakeEmbedder
from tests.pipeline.fakes import SECRETS, TODAY, VACANCIES, FakeNormalizer, caller, cv_bytes
from tests.web.test_routes import XSS, post, with_ip_limit

STREAM = {"Accept": "text/event-stream"}


def analyze_with_stages(content, filename, vacancies, **kwargs):
    """The real pipeline, passing on the progress callback like the production wiring does."""
    return pipeline.analyze_cv(
        content, filename, vacancies, caller=caller(), normalizer=FakeNormalizer(),
        embedder=FakeEmbedder(), today=TODAY, scorer=kwargs.get("scorer"),
        on_stage=kwargs.get("on_stage"))


def streaming_services(**changes) -> Services:
    return make_services(analyze=analyze_with_stages, **changes)


def parse(body: str) -> list[tuple[str, dict]]:
    """The events of a stream, in order; comments (keep-alive) are left out."""
    events = []
    for block in body.split("\n\n"):
        lines = [line for line in block.split("\n") if line]
        if not lines or lines[0].startswith(":"):
            continue
        name = lines[0].removeprefix("event: ")
        events.append((name, json.loads(lines[1].removeprefix("data: "))))
    return events


def stream_of(client, **kwargs):
    response = post(client, headers=STREAM, **kwargs)
    return response, parse(response.text)


@pytest.fixture
def client():
    with TestClient(create_app(streaming_services()), raise_server_exceptions=False) as c:
        yield c


# --- the steps ---------------------------------------------------------------------------------


def test_the_page_lists_five_steps_in_the_order_a_person_expects() -> None:
    assert progress.STEP_IDS == ("read", "skills", "match", "compare", "report")
    assert [s.label for s in progress.STEPS] == [
        "Reading your file", "Finding your skills and experience",
        "Matching your skills to a standard list", "Comparing you with current jobs",
        "Building your report"]


def test_every_pipeline_stage_belongs_to_a_step_and_steps_never_go_backwards() -> None:
    assert set(progress.STAGE_STEP) == {*pipeline.STAGES, progress.RENDER_STAGE}
    order = [progress.STEP_IDS.index(progress.STAGE_STEP[s])
             for s in (*pipeline.STAGES, progress.RENDER_STAGE)]
    assert order == sorted(order)
    assert set(progress.STAGE_STEP.values()) == set(progress.STEP_IDS)


def test_an_event_is_one_json_line_in_the_event_stream_format() -> None:
    assert progress.format_event("step", {"step": "skills"}) == (
        b'event: step\ndata: {"step":"skills"}\n\n')
    raw = progress.format_event("done", {"html": "a\nb é"})
    assert raw.count(b"\n") == 3 and "é".encode() in raw  # no raw line break inside data


@pytest.mark.parametrize("accept, expected", [
    ("text/event-stream", True),
    ("Text/Event-Stream; q=0.9", True),
    ("text/html, text/event-stream", True),
    ("text/html,application/xhtml+xml,*/*;q=0.8", False),
    ("*/*", False),
    ("", False),
    (None, False),
    ("text/event-streams", False),
    ("text/event", False),
    ("event-stream", False),
    ("text", False),
])
def test_only_an_explicit_request_for_a_stream_gets_one(accept, expected) -> None:
    assert progress.wants_stream(accept) is expected


# --- the stream over HTTP ----------------------------------------------------------------------


def test_a_stream_reports_each_step_once_in_order_and_ends_with_the_page(client) -> None:
    response, events = stream_of(client)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert [name for name, _ in events] == ["step"] * 5 + ["done"]
    assert [data["step"] for name, data in events if name == "step"] == list(progress.STEP_IDS)
    done = events[-1][1]
    assert done["status"] == 200 and "<h1>Your CV report</h1>" in done["html"]


def test_the_finished_page_in_the_stream_is_the_page_a_plain_post_gets(client) -> None:
    plain = post(client)
    _, events = stream_of(client)

    assert plain.status_code == 200
    assert events[-1][1]["html"] == plain.text


def test_a_stream_is_not_cached_not_buffered_and_locked_down_like_every_page(client) -> None:
    response, _ = stream_of(client)

    for name, value in routes_headers().items():
        assert response.headers[name] == value, name
    assert response.headers["x-accel-buffering"] == "no"


def routes_headers() -> dict[str, str]:
    from jobpdf.web.routes import PAGE_HEADERS

    return PAGE_HEADERS


def test_step_events_carry_a_step_id_and_nothing_from_the_cv(client) -> None:
    response, events = stream_of(client)

    for name, data in events:
        if name == "step":
            assert set(data) == {"step"} and data["step"] in progress.STEP_IDS
    head = response.text.split("event: done")[0]
    for secret in (*SECRETS, "Python"):
        assert secret not in head


def test_without_the_stream_header_nothing_changes(client) -> None:
    for headers in ({}, {"Accept": "text/html"}, {"Accept": "*/*"}):
        response = post(client, headers=headers)

        assert response.headers["content-type"].startswith("text/html")
        assert "<h1>Your CV report</h1>" in response.text


@pytest.mark.parametrize("error, status", [
    (PipelineError("extraction", "llm_error"), api.PIPELINE_STATUS["llm_error"]),
    (PipelineError("parse", "corrupt"), api.PIPELINE_STATUS["corrupt"]),
])
def test_a_known_failure_ends_the_stream_with_the_same_status_and_sentence_as_a_page(
        error, status) -> None:
    def analyze(*args, **kwargs):
        raise error

    with TestClient(create_app(make_services(analyze=analyze))) as client:
        _, events = stream_of(client)

    name, data = events[-1]
    assert name == "failed" and data["status"] == status
    assert api.ERROR_DETAILS[error.code] in data["html"] and "That did not work" in data["html"]


def test_an_unexpected_error_ends_the_stream_with_a_generic_page_that_repeats_nothing(
        caplog) -> None:
    def analyze(*args, **kwargs):
        raise RuntimeError(f"boom {SECRETS[0]} {XSS}")

    services = make_services(analyze=analyze)
    with TestClient(create_app(services), raise_server_exceptions=False) as client, \
            caplog.at_level(logging.DEBUG):
        response, events = stream_of(client)

    name, data = events[-1]
    assert name == "failed" and data["status"] == 500
    assert api.ERROR_DETAILS["internal_error"] in data["html"]
    assert SECRETS[0] not in response.text and "boom" not in response.text
    assert SECRETS[0] not in caplog.text and "RuntimeError" in caplog.text
    assert services.recorder.snapshot().analyses.total == 1


def test_a_failure_in_the_middle_stops_the_steps_where_they_were() -> None:
    def analyze(content, filename, vacancies, **kwargs):
        kwargs["on_stage"]("parse")
        kwargs["on_stage"]("extraction")
        raise PipelineError("extraction", "llm_error")

    with TestClient(create_app(make_services(analyze=analyze))) as client:
        _, events = stream_of(client)

    assert [(n, d.get("step")) for n, d in events[:-1]] == [("step", "read"), ("step", "skills")]
    assert events[-1][0] == "failed"


def test_a_page_that_cannot_be_drawn_ends_the_stream_with_a_generic_page(monkeypatch, client,
                                                                          caplog) -> None:
    from jobpdf.web import routes

    def broken(response, names=None, home=None):
        raise KeyError(SECRETS[0])

    monkeypatch.setattr(routes, "build_report", broken)
    with caplog.at_level(logging.DEBUG):
        response, events = stream_of(client)

    assert events[-1][0] == "failed" and events[-1][1]["status"] == 500
    assert [d["step"] for n, d in events if n == "step"][-1] == "report"
    assert SECRETS[0] not in response.text and SECRETS[0] not in caplog.text


# --- what a refused upload looks like ----------------------------------------------------------


def test_a_refused_upload_is_a_normal_error_page_even_when_a_stream_was_asked_for() -> None:
    services = streaming_services()
    with TestClient(create_app(services)) as client:
        response = post(client, b"x" * (MAX_UPLOAD_BYTES + 1), "cv.pdf", headers=STREAM)
        bad_form = client.post("/report", data={"nothing": "1"}, headers=STREAM)

    assert response.headers["content-type"].startswith("text/html")
    assert bad_form.status_code == 400 and bad_form.headers["content-type"].startswith("text/html")


def test_the_per_address_limit_still_applies_to_streams_and_the_page_says_when_to_return() -> None:
    services = with_ip_limit(1)
    services.analyze = analyze_with_stages
    with TestClient(create_app(services)) as client:
        first, _ = stream_of(client)
        second = post(client, headers=STREAM)

    assert first.status_code == 200
    assert second.status_code == 429 and "Retry-After" in second.headers
    assert second.headers["content-type"].startswith("text/html")
    assert services.recorder.snapshot().access_denied == {"rate_limited": 1}


def test_when_the_server_is_busy_a_stream_request_gets_the_busy_page() -> None:
    services = streaming_services(max_concurrent=1)
    assert services.slots.acquire(blocking=False)  # another analysis holds the only slot
    with TestClient(create_app(services)) as client:
        response = post(client, headers=STREAM)

    assert response.status_code == 503 and api.ERROR_DETAILS["busy"] in response.text
    assert response.headers["content-type"].startswith("text/html")
    assert services.recorder.snapshot().access_denied == {"busy": 1}


def test_a_stream_is_counted_in_the_metrics_like_any_analysis() -> None:
    services = streaming_services()
    with TestClient(create_app(services)) as client:
        stream_of(client)
        stream_of(client, content=b"not a pdf at all")

    snapshot = services.recorder.snapshot()
    assert snapshot.analyses.total == 2 and snapshot.analyses.ok == 1


# --- the analysis slot -------------------------------------------------------------------------


def test_the_slot_is_free_again_when_the_last_event_arrives() -> None:
    services = streaming_services()
    with TestClient(create_app(services)) as client:
        response, events = stream_of(client)
        again, _ = stream_of(client)  # a second try is not "busy"

    assert events[-1][0] == "done" and again.status_code == 200
    assert services.slots.acquire(blocking=False)


def test_the_slot_is_given_back_after_a_failure() -> None:
    def analyze(*args, **kwargs):
        raise PipelineError("parse", "corrupt")

    services = make_services(analyze=analyze)
    with TestClient(create_app(services)) as client:
        for _ in range(3):
            assert stream_of(client)[1][-1][0] == "failed"
    assert services.slots.acquire(blocking=False)


def test_the_slot_is_given_back_when_the_worker_cannot_start(monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise RuntimeError("no threads")

    services = streaming_services(max_concurrent=1)
    monkeypatch.setattr(progress, "AnalysisStream", refuse)
    with TestClient(create_app(services), raise_server_exceptions=False) as client:
        assert post(client, headers=STREAM).status_code == 500
    assert services.slots.acquire(blocking=False)


# --- AnalysisStream on its own -----------------------------------------------------------------


def run_stream(work, *, release=lambda: None, heartbeat_s=5.0, collect=True):
    async def main():
        stream = AnalysisStream(work, release, heartbeat_s=heartbeat_s)
        return [chunk async for chunk in stream.events()] if collect else stream

    return asyncio.run(main())


def test_a_repeated_or_unknown_stage_does_not_repeat_a_step() -> None:
    def work(on_stage):
        for stage in ("parse", "sections", "bogus", "extraction", "extraction", "grounding"):
            on_stage(stage)
        return Outcome("done", {"status": 200, "html": "ok"})

    chunks = run_stream(work)

    assert chunks == [
        progress.format_event("step", {"step": "read"}),
        progress.format_event("step", {"step": "skills"}),
        progress.format_event("done", {"status": 200, "html": "ok"}),
    ]


def test_the_slot_is_released_before_the_last_event_is_sent() -> None:
    order: list[str] = []

    async def main():
        stream = AnalysisStream(lambda on_stage: Outcome("done", {"html": "x"}),
                                lambda: order.append("released"), heartbeat_s=5.0)
        async for chunk in stream.events():
            order.append(chunk.split(b"\n", 1)[0].decode())

    asyncio.run(main())

    assert order == ["released", "event: done"]


def test_the_slot_is_released_even_when_nobody_reads_the_stream() -> None:
    released = threading.Event()

    async def main():
        AnalysisStream(lambda on_stage: Outcome("done", {"html": "x"}), released.set)
        await asyncio.sleep(0.5)  # the client vanished: events() is never iterated

    asyncio.run(main())

    assert released.is_set()


def test_a_work_function_that_breaks_gives_a_failed_event_and_still_releases(caplog) -> None:
    released: list[bool] = []

    def work(on_stage):
        raise ValueError(f"boom {SECRETS[0]}")

    with caplog.at_level(logging.DEBUG):
        chunks = run_stream(work, release=lambda: released.append(True))

    assert released == [True]
    assert chunks == [progress.format_event("failed", {"status": 500, "html": None})]
    assert SECRETS[0] not in caplog.text and "ValueError" in caplog.text


def test_a_release_that_breaks_does_not_lose_the_result(caplog) -> None:
    def release():
        raise RuntimeError("semaphore")

    with caplog.at_level(logging.ERROR):
        chunks = run_stream(lambda on_stage: Outcome("done", {"html": "x"}), release=release)

    assert chunks == [progress.format_event("done", {"html": "x"})]
    assert "could not be released" in caplog.text


def test_a_quiet_stretch_sends_a_keep_alive_comment_and_then_carries_on() -> None:
    gate = threading.Event()

    def work(on_stage):
        gate.wait(5)
        return Outcome("done", {"html": "x"})

    async def main():
        stream = AnalysisStream(work, lambda: None, heartbeat_s=0.05)
        chunks = []
        async for chunk in stream.events():
            chunks.append(chunk)
            if len(chunks) == 3:
                gate.set()
        return chunks

    chunks = asyncio.run(main())

    assert chunks[:3] == [progress.HEARTBEAT] * 3
    assert chunks[-1] == progress.format_event("done", {"html": "x"})
    assert progress.HEARTBEAT.startswith(b":") and progress.HEARTBEAT.endswith(b"\n\n")


def test_the_stream_ends_after_the_failed_event_too() -> None:
    chunks = run_stream(lambda on_stage: Outcome("failed", {"status": 422, "html": "e"}))

    assert chunks == [progress.format_event("failed", {"status": 422, "html": "e"})]


def test_a_client_that_leaves_mid_stream_does_not_hold_the_slot_or_break_the_worker() -> None:
    started, finish, released = threading.Event(), threading.Event(), threading.Event()

    def work(on_stage):
        on_stage("parse")
        started.set()
        finish.wait(5)
        return Outcome("done", {"html": "x"})

    async def main():
        stream = AnalysisStream(work, released.set, heartbeat_s=5.0)
        events = stream.events()
        first = await events.__anext__()
        await events.aclose()  # what the server does when the connection drops
        finish.set()
        await asyncio.sleep(0.3)
        return first

    first = asyncio.run(main())

    assert first == progress.format_event("step", {"step": "read"})
    assert released.is_set()


def test_the_analysis_uses_one_daemon_thread_per_stream() -> None:
    names: list[tuple[str, bool]] = []

    def work(on_stage):
        current = threading.current_thread()
        names.append((current.name, current.daemon))
        return Outcome("done", {"html": "x"})

    run_stream(work)

    assert names == [("jobpdf-analysis", True)]


def test_the_vacancy_list_is_read_once_per_stream() -> None:
    calls = []
    services = streaming_services(vacancies=lambda: calls.append(1) or VACANCIES)
    with TestClient(create_app(services)) as client:
        stream_of(client)

    assert calls == [1]


def test_the_default_analysis_function_of_the_tests_still_ignores_progress() -> None:
    # JM-43 tests build their services with ``analyze_offline``; it must keep working when the
    # route passes a callback to it.
    services = make_services(analyze=analyze_offline)
    with TestClient(create_app(services)) as client:
        _, events = stream_of(client)

    assert events[-1][0] == "done" and all(n == "done" or n == "step" for n, _ in events)


def test_a_plain_request_gives_the_analysis_no_progress_argument() -> None:
    seen: list[dict] = []

    def strict(content, filename, vacancies, *, caller, normalizer, embedder, scorer):
        seen.append({"called": True})  # would raise TypeError if given ``on_stage``
        return analyze_offline(content, filename, vacancies)

    with TestClient(create_app(make_services(analyze=strict))) as client:
        assert post(client).status_code == 200
        assert client.post("/v1/analyze", files={"file": ("cv.pdf", cv_bytes())}).status_code == 401

    assert seen == [{"called": True}]
