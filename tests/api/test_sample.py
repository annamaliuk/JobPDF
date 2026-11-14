"""JM-52: the bundled sample CV and the memo that keeps the model's answers for it."""

import logging
import threading

import pytest

from jobpdf.api import sample as sample_module
from jobpdf.api.sample import (
    DEFAULT_NAMES,
    MAX_MEMO,
    SAMPLE_CV_ENV,
    SAMPLE_DIR,
    SAMPLE_EMPTY,
    SAMPLE_MISSING,
    SAMPLE_TOO_LARGE,
    SAMPLE_UNREADABLE,
    SAMPLE_UNSUPPORTED,
    MemoCaller,
    SampleCV,
    load_sample,
    request_key,
    sample_path,
)
from jobpdf.extraction.llm import FakeToolCaller, LLMResponseError, MissingAPIKeyError
from jobpdf.pipeline import MAX_UPLOAD_BYTES
from tests.pipeline.fakes import caller, cv_bytes, extracted, reply, run

ASK = {"system": "s", "user": "u", "tool": {"name": "t"}, "max_tokens": 10}


# --- where the file is -------------------------------------------------------------------------


def test_the_bundled_file_lives_inside_the_package_so_the_image_has_it() -> None:
    assert SAMPLE_DIR.parts[-3:] == ("jobpdf", "web", "sample")
    assert DEFAULT_NAMES == ("sample_cv.pdf", "sample_cv.docx")
    assert SAMPLE_CV_ENV == "SAMPLE_CV_PATH"


def test_a_configured_path_wins_over_the_bundled_files(tmp_path) -> None:
    (tmp_path / "sample_cv.pdf").write_bytes(b"%PDF")

    assert sample_path({SAMPLE_CV_ENV: " /srv/demo.docx "}, tmp_path).as_posix() == "/srv/demo.docx"


def test_the_bundled_pdf_is_preferred_then_the_docx(tmp_path) -> None:
    assert sample_path({}, tmp_path) is None
    (tmp_path / "sample_cv.docx").write_bytes(b"PK")
    assert sample_path({}, tmp_path) == tmp_path / "sample_cv.docx"
    (tmp_path / "sample_cv.pdf").write_bytes(b"%PDF")
    assert sample_path({}, tmp_path) == tmp_path / "sample_cv.pdf"
    assert sample_path({SAMPLE_CV_ENV: "  "}, tmp_path) == tmp_path / "sample_cv.pdf"


def test_a_directory_with_the_bundled_name_is_not_a_file(tmp_path) -> None:
    (tmp_path / "sample_cv.pdf").mkdir()

    assert sample_path({}, tmp_path) is None


def test_the_process_environment_is_read_when_none_is_given(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(SAMPLE_CV_ENV, str(tmp_path / "x.pdf"))

    assert sample_path() == tmp_path / "x.pdf"


# --- loading it --------------------------------------------------------------------------------


def test_a_usable_pdf_is_loaded_under_a_fixed_name(tmp_path, caplog) -> None:
    path = tmp_path / "Real Name CV.PDF"
    path.write_bytes(cv_bytes())
    caplog.set_level(logging.INFO)

    sample = load_sample(path, make_caller=caller)

    assert sample.content == cv_bytes()
    assert sample.filename == "sample_cv.pdf"  # the original name is never used or shown
    assert sample.media_type == "application/pdf"
    assert isinstance(sample.caller, MemoCaller)
    assert "sample CV on: pdf" in caplog.text and "Real Name" not in caplog.text


def test_a_docx_is_loaded_with_its_media_type(tmp_path) -> None:
    path = tmp_path / "cv.docx"
    path.write_bytes(b"PK\x03\x04 docx bytes")

    sample = load_sample(path, make_caller=caller)

    assert sample.filename == "sample_cv.docx"
    assert sample.media_type.endswith("wordprocessingml.document")


@pytest.mark.parametrize("name, content, code", [
    ("cv.txt", b"text", SAMPLE_UNSUPPORTED),
    ("cv.doc", b"old word", SAMPLE_UNSUPPORTED),
    ("cv.pdf", b"", SAMPLE_EMPTY),
])
def test_a_file_the_pipeline_would_refuse_turns_the_feature_off(
        tmp_path, caplog, name, content, code) -> None:
    path = tmp_path / name
    path.write_bytes(content)

    assert load_sample(path, make_caller=caller) is None
    assert f"sample CV off: {code}" in caplog.text


def test_a_file_over_the_upload_limit_turns_the_feature_off(tmp_path, caplog) -> None:
    path = tmp_path / "cv.pdf"
    with path.open("wb") as handle:
        handle.truncate(MAX_UPLOAD_BYTES + 1)

    assert load_sample(path, make_caller=caller) is None
    assert f"sample CV off: {SAMPLE_TOO_LARGE}" in caplog.text


def test_a_file_exactly_at_the_upload_limit_is_used(tmp_path) -> None:
    path = tmp_path / "cv.pdf"
    with path.open("wb") as handle:
        handle.truncate(MAX_UPLOAD_BYTES)

    assert load_sample(path, make_caller=caller) is not None


def test_no_file_is_not_a_warning(tmp_path, caplog) -> None:
    caplog.set_level(logging.INFO)

    assert load_sample(tmp_path / "missing.pdf", make_caller=caller) is None

    record = next(r for r in caplog.records if SAMPLE_MISSING in r.getMessage())
    assert record.levelno == logging.INFO


def test_an_unreadable_file_turns_the_feature_off(tmp_path, caplog, monkeypatch) -> None:
    path = tmp_path / "cv.pdf"
    path.write_bytes(b"%PDF")

    def refuse(self):
        raise PermissionError("no")

    monkeypatch.setattr(type(path), "read_bytes", refuse)

    assert load_sample(path, make_caller=caller) is None
    assert f"sample CV off: {SAMPLE_UNREADABLE}" in caplog.text


def test_a_file_that_cannot_be_inspected_turns_the_feature_off(tmp_path, monkeypatch) -> None:
    path = tmp_path / "cv.pdf"
    path.write_bytes(b"%PDF")

    def refuse(self, **kwargs):
        raise PermissionError("no")

    monkeypatch.setattr(type(path), "is_file", lambda self: True)
    monkeypatch.setattr(type(path), "stat", refuse)

    assert sample_module._check(path) == SAMPLE_UNREADABLE


def test_without_a_path_the_configured_one_is_loaded(monkeypatch, tmp_path) -> None:
    path = tmp_path / "demo.pdf"
    path.write_bytes(cv_bytes())
    monkeypatch.setenv(SAMPLE_CV_ENV, str(path))

    assert load_sample(make_caller=caller).content == cv_bytes()


def test_loading_does_not_make_the_model_caller(tmp_path) -> None:
    path = tmp_path / "cv.pdf"
    path.write_bytes(cv_bytes())
    made = []

    load_sample(path, make_caller=lambda: made.append(1) or caller())

    assert made == []  # a missing API key must not stop the app from starting


def test_two_samples_compare_by_their_file_not_their_memo() -> None:
    one = SampleCV(b"x", "sample_cv.pdf", MemoCaller(caller))
    two = SampleCV(b"x", "sample_cv.pdf", MemoCaller(caller))

    assert one == two


# --- the memo ----------------------------------------------------------------------------------


def test_the_first_call_asks_the_model_and_the_same_request_again_does_not() -> None:
    inner = caller()  # one scripted answer: a second real call would fail
    memo = MemoCaller(lambda: inner)

    first = memo.call(**ASK)
    second = memo.call(**ASK)

    assert first is second and len(inner.calls) == 1 and memo.calls == 1 and len(memo) == 1


@pytest.mark.parametrize("change", [{"system": "s2"}, {"user": "u2"},
                                    {"tool": {"name": "t2"}}, {"max_tokens": 11}])
def test_any_change_in_the_request_asks_the_model_again(change) -> None:
    inner = caller(reply({"a": 1}), reply({"a": 2}))
    memo = MemoCaller(lambda: inner)

    memo.call(**ASK)
    other = memo.call(**{**ASK, **change})

    assert other.tool_input == {"a": 2} and len(inner.calls) == 2


def test_a_failed_call_is_not_remembered() -> None:
    inner = caller(LLMResponseError("no tool call"), reply({"a": 1}))
    memo = MemoCaller(lambda: inner)

    with pytest.raises(LLMResponseError):
        memo.call(**ASK)
    assert len(memo) == 0 and memo.calls == 0

    assert memo.call(**ASK).tool_input == {"a": 1} and memo.calls == 1


def test_the_real_caller_is_made_once_and_only_when_needed() -> None:
    made = []
    inner = caller(reply({"a": 1}), reply({"a": 2}))

    memo = MemoCaller(lambda: made.append(1) or inner)
    assert made == []
    assert memo.model == "fake-model"
    memo.call(**ASK)
    memo.call(**{**ASK, "user": "other"})

    assert made == [1]


def test_a_missing_key_surfaces_when_the_model_is_needed() -> None:
    def no_key():
        raise MissingAPIKeyError("GEMINI_API_KEY is missing")

    memo = MemoCaller(no_key)

    with pytest.raises(MissingAPIKeyError):
        memo.call(**ASK)
    with pytest.raises(MissingAPIKeyError):
        _ = memo.model


def test_the_memo_keeps_only_the_most_recent_answers() -> None:
    inner = FakeToolCaller([reply({"n": n}) for n in range(MAX_MEMO + 2)])
    memo = MemoCaller(lambda: inner)

    for n in range(MAX_MEMO + 1):
        memo.call(**{**ASK, "user": str(n)})

    assert len(memo) == MAX_MEMO
    memo.call(**{**ASK, "user": "1"})  # still kept: no new model call
    assert len(inner.calls) == MAX_MEMO + 1
    memo.call(**{**ASK, "user": "0"})  # the oldest was dropped: asked again
    assert len(inner.calls) == MAX_MEMO + 2


def test_reading_an_answer_keeps_it_from_being_dropped_first() -> None:
    inner = FakeToolCaller([reply({"n": n}) for n in range(4)])
    memo = MemoCaller(lambda: inner, max_entries=2)

    memo.call(**{**ASK, "user": "a"})
    memo.call(**{**ASK, "user": "b"})
    memo.call(**{**ASK, "user": "a"})  # "a" is now the most recent
    memo.call(**{**ASK, "user": "c"})  # drops "b", not "a"

    assert len(inner.calls) == 3
    memo.call(**{**ASK, "user": "a"})
    assert len(inner.calls) == 3


def test_requests_are_keyed_on_their_content_not_on_dict_order() -> None:
    one = request_key(system="s", user="u", tool={"a": 1, "b": 2}, max_tokens=1)
    two = request_key(system="s", user="u", tool={"b": 2, "a": 1}, max_tokens=1)

    assert one == two and len(one) == 64
    assert one != request_key(system="s", user="u", tool={"a": 1, "b": 3}, max_tokens=1)


def test_calls_from_several_threads_all_get_an_answer() -> None:
    inner = FakeToolCaller([reply({"a": 1}) for _ in range(8)])
    memo = MemoCaller(lambda: inner)
    results = []

    threads = [threading.Thread(target=lambda: results.append(memo.call(**ASK)))
               for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 8 and all(r.tool_input == {"a": 1} for r in results)
    assert len(memo) == 1


# --- through the real pipeline -----------------------------------------------------------------


def test_the_pipeline_reads_the_sample_twice_with_one_model_call() -> None:
    inner = caller()  # one scripted answer
    memo = MemoCaller(lambda: inner)

    first = run(caller=memo)
    second = run(caller=memo)

    assert len(inner.calls) == 1
    assert first.profile == second.profile
    assert first.ranking.results == second.ranking.results


def test_the_pipeline_with_the_memo_matches_the_pipeline_without_it() -> None:
    plain = run(caller=caller(reply(extracted())))
    memo = run(caller=MemoCaller(lambda: caller(reply(extracted()))))

    assert memo.profile == plain.profile
    assert [r.vacancy.id for r in memo.ranking.results] == [
        r.vacancy.id for r in plain.ranking.results]


def test_a_sample_whose_answer_needed_a_correction_is_read_again_without_the_model() -> None:
    # The extraction asks again with a correction when the first answer is invalid: the memo must
    # hold both answers, or the second report would ask the model for the first one again.
    inner = caller(reply({"not": "a profile"}), reply(extracted()))
    memo = MemoCaller(lambda: inner)

    first = run(caller=memo)
    second = run(caller=memo)

    assert len(inner.calls) == 2 and len(memo) == 2
    assert first.profile == second.profile
    assert MAX_MEMO >= 2


# --- wiring ------------------------------------------------------------------------------------


def test_build_services_loads_the_sample(monkeypatch) -> None:
    from jobpdf.api import app as app_module
    from tests.api.test_app import access
    from tests.pipeline.fakes import FakeNormalizer

    loaded = SampleCV(b"%PDF", "sample_cv.pdf", MemoCaller(caller))
    monkeypatch.setattr(app_module, "build_access_control", access)
    monkeypatch.setattr(app_module, "get_normalization_service", FakeNormalizer)
    monkeypatch.setattr(app_module, "database_url", lambda: None)
    monkeypatch.setattr(app_module, "load_sample", lambda: loaded)

    assert app_module.build_services().sample is loaded


def test_services_have_no_sample_unless_given_one() -> None:
    from tests.api.test_app import make_services

    assert make_services().sample is None


def test_read_fields_returns_text_fields_and_never_a_file() -> None:
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient

    from jobpdf.api.app import read_fields

    app = FastAPI()

    @app.post("/f")
    async def endpoint(request: Request) -> dict:
        return await read_fields(request, ("lang", "home_country", "absent"))

    with TestClient(app) as client:
        body = client.post("/f", data={"lang": "uk"},
                           files={"home_country": ("UA", b"UA", "text/plain")}).json()
        plain = client.post("/f", data={"lang": "en", "home_country": "PL"}).json()

    assert body == {"lang": "uk", "home_country": "", "absent": ""}
    assert plain == {"lang": "en", "home_country": "PL", "absent": ""}
