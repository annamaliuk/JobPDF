"""End-to-end runs of scripts/eval_extraction.py on the synthetic fixture (JM-14)."""

import importlib.util
import json
from pathlib import Path

import pytest

from jobpdf.extraction.eval_dataturks import load_dataturks
from jobpdf.extraction.extract import extract_profile
from jobpdf.extraction.llm import FakeToolCaller, RecordingToolCaller, ToolCallResult
from jobpdf.extraction.plaintext import from_plain_text
from jobpdf.extraction.sections import split_sections

FIXTURE = Path(__file__).parent / "fixtures" / "dataturks_synthetic.jsonl"
SCRIPT = Path(__file__).parents[2] / "scripts" / "eval_extraction.py"

_spec = importlib.util.spec_from_file_location("eval_extraction", SCRIPT)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


def job(title: str, company: str, quote: str) -> dict:
    return {"title": title, "company": company, "location": None, "start": "2020",
            "end": "present", "description": None, "skills_used": [], "quote": quote}


def school(institution: str, degree: str, end: str, quote: str) -> dict:
    return {"institution": institution, "degree": degree, "field": None, "start": None,
            "end": end, "quote": quote}


def hard(name: str, quote: str) -> dict:
    return {"name": name, "kind": "hard", "found_in": "skills_list", "quote": quote}


RESPONSES = {
    0: {
        "summary": None,
        "skills": [hard("Python", "Python, SQL, Tableau")],
        "experience": [
            job("Data Analyst", "Example Corp", "Data Analyst\nExample Corp"),
            job("Junior Analyst", "Sample Labs", "Junior Analyst\nSample Labs"),
        ],
        "education": [school("Placeholder University", "B.Sc. Statistics", "2018",
                             "B.Sc. Statistics\nPlaceholder University, 2018")],
        "languages": [],
        "certifications": [],
    },
    2: {
        "summary": "Alex Example is a QA engineer",  # a leaked name
        "skills": [hard("Selenium", "Selenium, Jira"), hard("Bash", "Bash")],  # Bash: unlocated
        "experience": [
            job("QA Engineer", "Test House Ltd", "QA Engineer, Test House Ltd"),
            # Gold for this company is recovered by the "&amp;" repair.
            job("QA Intern", "Bugs & Fixes", "QA Intern, Bugs &amp; Fixes"),
        ],
        "education": [school("Sample College", "Diploma in Software Testing", "2016",
                             "Diploma in Software Testing - Sample College - 2016")],
        "languages": [],
        "certifications": [],
    },
}
CV_STRINGS = ["Placeholder", "example.com", "Alex Example", "Python, SQL", "Data Analyst",
              "Selenium", "Bash", "Test House", "Bugs"]


def reply(tool_input: dict, stop_reason: str = "tool_use") -> ToolCallResult:
    return ToolCallResult(tool_input=tool_input, stop_reason=stop_reason,
                          input_tokens=1000, output_tokens=200, model="recorded-model")


@pytest.fixture
def recordings(tmp_path: Path) -> Path:
    """Recordings for resumes 0 and 2 only, made through the runner's own pipeline."""
    directory = tmp_path / "recordings"
    resumes, _, _ = load_dataturks(FIXTURE)
    for resume in resumes:
        if resume.index not in RESPONSES:
            continue
        doc = from_plain_text(resume.text)
        sections, _ = split_sections(doc)
        recorder = RecordingToolCaller(FakeToolCaller([reply(RESPONSES[resume.index])]),
                                       directory)
        extract_profile(doc, sections, recorder, use_cache=False)
    return directory


def run(tmp_path: Path, recordings: Path, *extra: str) -> tuple[int, dict | None]:
    out = tmp_path / "report.json"
    code = runner.main(["--dataset", str(FIXTURE), "--recordings", str(recordings),
                        "--out", str(out), *extra])
    report = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else None
    return code, report


def test_replay_end_to_end(tmp_path: Path, recordings: Path, capsys) -> None:
    code, report = run(tmp_path, recordings)

    assert code == 0
    assert report["mode"] == "replay"
    assert report["models"] == ["recorded-model"]
    assert report["match_threshold"] == 85
    assert (report["dataset_records"], report["resumes_selected"]) == (4, 4)
    assert (report["resumes_evaluated"], report["missing_recordings"]) == (2, 2)
    assert (report["invalid_spans"], report["duplicate_spans"]) == (1, 1)
    assert report["amp_repaired_spans"] == 5
    assert report["extraction_errors"] == 0
    assert report["unlocated_quotes"] == 1
    assert report["contact_leaks"] == 1
    assert len(report["dataset_sha256"]) == 64
    fields = {name: (m["tp"], m["fp"], m["fn"]) for name, m in report["fields"].items()}
    assert fields == {
        "designation": (4, 0, 0),
        "companies": (4, 0, 0),
        "degree": (2, 0, 0),
        "college": (2, 0, 0),
        "graduation_year": (2, 0, 0),
    }
    skills = report["skills"]
    assert (skills["quotes"], skills["quotes_inside"], skills["quote_precision"]) == (3, 2, 1.0)
    assert (skills["spans"], skills["spans_covered"], skills["span_coverage"]) == (2, 2, 1.0)
    assert sorted(report["per_resume"]) == ["0", "2"]
    assert report["per_resume"]["2"]["contact_leaks"] == 1
    assert set(report["not_evaluated"]) >= {"Location", "Years of Experience"}

    err = capsys.readouterr().err
    assert "WARN resume 1: no recorded response, skipped" in err
    assert "WARN resume 3: no recorded response, skipped" in err


def test_report_and_output_hold_no_cv_text(tmp_path: Path, recordings: Path, capsys) -> None:
    run(tmp_path, recordings)
    report_text = (tmp_path / "report.json").read_text(encoding="utf-8")
    captured = capsys.readouterr()

    for text in (report_text, captured.out, captured.err):
        for cv_string in CV_STRINGS:
            assert cv_string not in text, cv_string


def test_limit(tmp_path: Path, recordings: Path) -> None:
    _, report = run(tmp_path, recordings, "--limit", "1")

    assert (report["resumes_selected"], report["resumes_evaluated"]) == (1, 1)


def test_live_without_key_is_one_clean_line(tmp_path: Path, capsys) -> None:
    code, report = run(tmp_path, tmp_path / "recordings", "--mode", "live")

    assert code == 2 and report is None
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "ANTHROPIC_API_KEY" in err
    assert "Traceback" not in err
    assert not (tmp_path / "recordings").exists()


def test_missing_dataset(tmp_path: Path, capsys) -> None:
    code = runner.main(["--dataset", str(tmp_path / "nope.jsonl"), "--out",
                        str(tmp_path / "r.json")])

    assert code == 1
    assert capsys.readouterr().err.startswith("ERROR: dataset not found")


def test_replay_or_record_records_once_and_rerecords_truncated(tmp_path: Path) -> None:
    kwargs = {"system": "s", "user": "u", "tool": {"name": "t"}, "max_tokens": 10}
    truncated, full = reply({}, stop_reason="max_tokens"), reply({"summary": None})
    live = FakeToolCaller([truncated, full], model="live-model")
    caller = runner.ReplayOrRecord(live, tmp_path)

    assert caller.model == "live-model"
    assert caller.call(**kwargs) == truncated  # miss: asks the API, records
    assert caller.call(**kwargs) == full  # recorded answer was truncated: asks again
    assert caller.call(**kwargs) == full  # now served from the recording
    assert len(live.calls) == 2
