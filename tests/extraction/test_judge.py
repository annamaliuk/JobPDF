"""LLM-as-judge check and its calibration (JM-15). FakeToolCaller only."""

import datetime as dt
import importlib.util
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from jobpdf.extraction import parse
from jobpdf.extraction.eval_dataturks import GoldSpan, Resume
from jobpdf.extraction.extract import extract_profile
from jobpdf.extraction.judge import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_TOOL_NAME,
    CVJudgement,
    ItemVerdict,
    JudgeError,
    Judgement,
    JudgeResult,
    MissedItem,
    aggregate,
    build_judge_message,
    check_judgement,
    detect_language,
    judge_profile,
    judge_tool,
    weak_grounding_items,
)
from jobpdf.extraction.judge_calibration import (
    agreement_report,
    caught,
    label_agreement,
    plant_copies,
    plant_error,
    plant_report,
)
from jobpdf.extraction.llm import FakeToolCaller, RecordingToolCaller, ToolCallResult
from jobpdf.extraction.plaintext import from_plain_text
from jobpdf.extraction.schema import CandidateProfile, ExtractedJob, Skill
from jobpdf.extraction.sections import split_sections

CV_TEXT = """\
Experience

Data Engineer
Acme Widgets LLC, 2021-2024
Built nightly reporting jobs.

Junior Analyst
Globex Testing Co, 2019-2021

Education

BSc Computer Science, Example State University, 2015-2019

Skills

Python, SQL, Apache Airflow, Docker

Contact: olena.testenko@example.com, +380 44 123 4567
"""
FIXTURE_PDF = Path(__file__).parent / "fixtures" / "single_column.pdf"
SCRIPT = Path(__file__).parents[2] / "scripts" / "judge_extraction.py"
_spec = importlib.util.spec_from_file_location("judge_extraction", SCRIPT)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


def job(title: str, company: str | None, start: str | None, quote: str) -> ExtractedJob:
    return ExtractedJob(title=title, company=company, location=None, start=start, end="2024",
                        description=None, skills_used=[], quote=quote)


def skill(name: str, quote: str | None = None) -> Skill:
    return Skill(name=name, kind="hard", found_in=["skills_list"], quotes=[quote or name])


def make_profile() -> CandidateProfile:
    return CandidateProfile(
        summary=None,
        skills=[skill("Python"), skill("SQL"), skill("Kotlin")],  # Kotlin is not in the CV
        experience=[
            job("Data Engineer", "Acme Widgets LLC", "2021", "Data Engineer\nAcme Widgets LLC"),
            job("Junior Analyst", "Globex Testing Co", "2019", "Junior Analyst"),
        ],
        education=[],
        languages=[],
        certifications=[],
        total_experience_months=None,
    )


@pytest.fixture
def cv():
    doc = from_plain_text(CV_TEXT)
    sections, _ = split_sections(doc)
    return doc, sections


def reply(tool_input: dict, stop_reason: str = "tool_use") -> ToolCallResult:
    return ToolCallResult(tool_input=tool_input, stop_reason=stop_reason,
                          input_tokens=500, output_tokens=100, model="judge-model")


def verdict(path: str, value: str = "supported", wrong: list[str] | None = None) -> dict:
    return {"path": path, "verdict": value, "wrong_fields": wrong or [], "reason": ""}


def all_supported(profile: CandidateProfile) -> list[dict]:
    paths = [f"skills[{i}]" for i in range(len(profile.skills))]
    paths += [f"experience[{i}]" for i in range(len(profile.experience))]
    return [verdict(p) for p in paths]


# --- prompt and tool -------------------------------------------------------


def test_message_holds_masked_cv_and_numbered_items(cv) -> None:
    doc, sections = cv
    message = build_judge_message(doc, sections, make_profile())

    assert "olena.testenko@example.com" not in message
    assert "+380 44 123 4567" not in message
    assert "Acme Widgets LLC, 2021-2024" in message
    assert message.startswith("<cv>\n<section") and "</cv>" in message
    extraction = json.loads(message.split("<extraction>\n", 1)[1].split("\n</extraction>")[0])
    assert [s["path"] for s in extraction["skills"]] == ["skills[0]", "skills[1]", "skills[2]"]
    assert extraction["experience"][1]["path"] == "experience[1]"
    assert "summary" not in extraction


def test_tool_is_self_contained() -> None:
    tool = judge_tool()
    assert tool["name"] == JUDGE_TOOL_NAME
    assert "$defs" not in json.dumps(tool)
    assert set(tool["input_schema"]["properties"]) == {"items", "missed", "text_unreadable"}
    assert "Ukrainian" in JUDGE_SYSTEM_PROMPT


# --- checking the judge ----------------------------------------------------


def test_check_drops_unknown_duplicate_and_ungrounded(cv) -> None:
    doc, sections = cv
    profile = make_profile()
    judgement = Judgement(
        items=[
            ItemVerdict(path="skills[0]", verdict="supported", wrong_fields=[], reason=""),
            ItemVerdict(path="skills[0]", verdict="unsupported", wrong_fields=[], reason="dup"),
            ItemVerdict(path=" skills[2] ", verdict="unsupported", wrong_fields=[], reason="x"),
            ItemVerdict(path="skills[9]", verdict="supported", wrong_fields=[], reason=""),
            ItemVerdict(path="experience[1]", verdict="partly_wrong", wrong_fields=["end"],
                        reason="ends 2021"),
        ],
        missed=[
            MissedItem(field="skills", quote="Apache Airflow"),  # in the CV
            MissedItem(field="skills", quote="Terraform"),  # invented
            MissedItem(field="skills", quote="████████"),  # masked contact
        ],
        text_unreadable=False,
    )

    result = check_judgement(judgement, doc, sections, profile)

    assert [v.path for v in result.verdicts] == ["skills[0]", "skills[2]", "experience[1]"]
    assert result.verdicts[0].verdict == "supported"  # first verdict per path wins
    assert result.unjudged_paths == ["skills[1]", "experience[0]"]
    assert (result.dropped_unknown_paths, result.dropped_duplicate_verdicts) == (1, 1)
    assert [m.quote for m in result.missed] == ["Apache Airflow"]
    assert (result.dropped_ungrounded_missed, result.dropped_masked_missed) == (1, 1)


def test_judge_profile_retries_once_on_invalid_output(cv) -> None:
    doc, sections = cv
    profile = make_profile()
    good = {"items": all_supported(profile), "missed": [], "text_unreadable": False}
    caller = FakeToolCaller([reply({"items": "nope"}), reply(good)])

    result = judge_profile(doc, sections, profile, caller)

    assert len(caller.calls) == 2
    assert "<validation_errors>" in caller.calls[1]["user"]
    assert caller.calls[0]["system"] == JUDGE_SYSTEM_PROMPT
    assert len(result.verdicts) == 5 and not result.unjudged_paths
    assert (result.input_tokens, result.output_tokens, result.model) == (1000, 200, "judge-model")


def test_judge_profile_raises_after_second_invalid_output(cv) -> None:
    doc, sections = cv
    caller = FakeToolCaller([reply({"items": []}), reply({"missed": []})])
    with pytest.raises(JudgeError, match="twice"):
        judge_profile(doc, sections, make_profile(), caller)


def test_judge_profile_retries_truncated_output_with_bigger_budget(cv) -> None:
    doc, sections = cv
    profile = make_profile()
    good = {"items": all_supported(profile), "missed": [], "text_unreadable": False}
    caller = FakeToolCaller([reply({}, stop_reason="max_tokens"), reply(good)])

    judge_profile(doc, sections, profile, caller)

    assert [c["max_tokens"] for c in caller.calls] == [8192, 16384]


# --- aggregate -------------------------------------------------------------


def result_of(verdicts: list[tuple[str, str]], missed: list[str] = (),
              unreadable: bool = False) -> JudgeResult:
    return JudgeResult(
        verdicts=[ItemVerdict(path=p, verdict=v,
                              wrong_fields=["start"] if v == "partly_wrong" else [], reason="")
                  for p, v in verdicts],
        missed=[MissedItem(field=f, quote="q") for f in missed],
        text_unreadable=unreadable,
        item_paths=[p for p, _ in verdicts],
    )


def test_aggregate_rates_languages_and_unreadable() -> None:
    english = CVJudgement("1.pdf", "en", result_of(
        [("skills[0]", "supported"), ("skills[1]", "unsupported"),
         ("experience[0]", "partly_wrong")], missed=["skills"]),
        weak_grounding={"skills[1]", "experience[0]"})
    ukrainian = CVJudgement("2.pdf", "uk", result_of([("skills[0]", "supported")]))
    garbled = CVJudgement("3.pdf", "mixed", result_of([("skills[0]", "unsupported")],
                                                      unreadable=True))

    report = aggregate([english, ukrainian, garbled])

    assert (report["cvs"], report["cvs_unreadable"]) == (3, 1)
    assert report["cvs_by_language"] == {"en": 1, "uk": 1}
    skills = report["overall"]["fields"]["skills"]
    assert (skills["judged"], skills["supported"], skills["unsupported"], skills["missed"]) == (
        3, 2, 1, 1)
    assert skills["supported_rate"] == round(2 / 3, 4)
    assert skills["miss_rate"] == 0.25
    experience = report["overall"]["fields"]["experience"]
    assert experience["error_rate"] == 1.0 and experience["top_wrong_fields"] == {"start": 1}
    education = report["overall"]["fields"]["education"]
    assert education["judged"] == 0 and education["supported_rate"] is None
    assert report["overall"]["all"]["judged"] == 4
    assert report["by_language"]["uk"]["all"]["supported_rate"] == 1.0
    assert report["grounding_agreement"] == {
        "unsupported_items": 1, "weakly_grounded_items": 2, "both": 1,
        "unsupported_also_weakly_grounded": 1.0, "weakly_grounded_also_unsupported": 0.5,
    }


def test_aggregate_holds_no_cv_text() -> None:
    result = result_of([("skills[0]", "unsupported")], missed=["skills"])
    result.verdicts[0].reason = "Kotlin is nowhere in the CV"
    result.missed[0].quote = "Apache Airflow"
    dumped = json.dumps(aggregate([CVJudgement("1.pdf", "en", result)]))
    assert "Kotlin" not in dumped and "Airflow" not in dumped


@pytest.mark.parametrize(("text", "expected"), [
    ("Data Engineer with Python and SQL", "en"),
    ("Інженер даних, досвід роботи з базами даних", "uk"),
    ("Інженер даних, досвід: Python, SQL, Apache Airflow, Docker", "mixed"),
    ("2019 - 2021 ████", "unknown"),
])
def test_detect_language(text: str, expected: str) -> None:
    assert detect_language(text) == expected


def test_weak_grounding_marks_items_with_unfound_quotes(cv) -> None:
    doc, sections = cv
    weak = weak_grounding_items(make_profile(), doc, sections)
    assert weak == {"skills[2]"}  # "Kotlin" is not in the CV


# --- calibration -----------------------------------------------------------


def baseline_all_supported(profile: CandidateProfile) -> JudgeResult:
    return result_of([(v["path"], "supported") for v in all_supported(profile)])


def test_plants_put_one_known_mistake_in_a_copy() -> None:
    profile = make_profile()
    baseline = baseline_all_supported(profile)

    fake = plant_error(profile, "fake_skill", CV_TEXT, baseline)
    assert fake.path == "skills[3]" and fake.profile.skills[3].name == "COBOL"

    dropped = plant_error(profile, "drop_job", CV_TEXT, baseline)
    assert dropped.path is None and [j.title for j in dropped.profile.experience] == [
        "Junior Analyst"]

    shifted = plant_error(profile, "shift_start", CV_TEXT, baseline)
    assert shifted.path == "experience[0]" and shifted.profile.experience[0].start == "2019"

    company = plant_error(profile, "fake_company", CV_TEXT, baseline)
    assert company.profile.experience[0].company == "Northwind Traders"

    assert len(profile.skills) == 3 and profile.experience[0].start == "2021"  # untouched


def test_plants_skip_items_the_baseline_already_flagged() -> None:
    profile = make_profile()
    baseline = result_of([("experience[0]", "partly_wrong"), ("experience[1]", "unsupported")])
    assert plant_error(profile, "shift_start", CV_TEXT, baseline) is None
    assert plant_error(profile, "drop_job", CV_TEXT, baseline) is None


def test_shift_start_keeps_the_month() -> None:
    profile = make_profile()
    profile.experience[0] = profile.experience[0].model_copy(update={"start": "2021-03"})
    plant = plant_error(profile, "shift_start", CV_TEXT, baseline_all_supported(profile))
    assert plant.profile.experience[0].start == "2019-03"


def test_caught_rules() -> None:
    profile = make_profile()
    baseline = baseline_all_supported(profile)
    fake = plant_error(profile, "fake_skill", CV_TEXT, baseline)
    assert caught(fake, result_of([("skills[3]", "unsupported")]), baseline)
    assert not caught(fake, result_of([("skills[3]", "supported")]), baseline)
    assert not caught(fake, result_of([]), baseline)  # no verdict is no catch

    dropped = plant_error(profile, "drop_job", CV_TEXT, baseline)
    assert caught(dropped, missed_jobs("Data Engineer, Acme Widgets LLC"), baseline)
    assert caught(dropped, missed_jobs("Acme Widgets LLC, 2021-2024"), baseline)
    assert not caught(dropped, missed_jobs("Junior Analyst"), baseline)  # another job
    assert not caught(dropped, result_of([], missed=["skills"]), baseline)


def missed_jobs(*quotes: str) -> JudgeResult:
    result = result_of([])
    result.missed = [MissedItem(field="experience", quote=q) for q in quotes]
    return result


def three_job_profile() -> CandidateProfile:
    profile = make_profile()
    profile.experience.append(job("Intern", "Initrode Labs", "2018", "Intern"))
    return profile


def test_plant_copies_pack_one_mistake_per_job_into_one_copy() -> None:
    profile = three_job_profile()
    copies, unfair = plant_copies(profile, CV_TEXT, baseline_all_supported(profile))

    assert unfair == [] and len(copies) == 1
    plants = {p.kind: p for p in copies[0]}
    assert set(plants) == {"fake_skill", "drop_job", "shift_start", "fake_company"}
    copy = plants["drop_job"].profile
    assert all(p.profile is copy for p in plants.values())  # one judge call
    assert plants["drop_job"].dropped.title == "Data Engineer"
    # Job 0 is dropped, so jobs 1 and 2 move up to 0 and 1.
    assert [j.title for j in copy.experience] == ["Junior Analyst", "Intern"]
    assert plants["shift_start"].path == "experience[0]" and copy.experience[0].start == "2017"
    assert plants["fake_company"].path == "experience[1]"
    assert copy.experience[1].company == "Northwind Traders"
    assert plants["fake_skill"].path == "skills[3]"
    assert len(profile.experience) == 3 and profile.experience[0].start == "2021"  # untouched


def test_plant_copies_spill_over_when_jobs_run_out() -> None:
    profile = make_profile()  # two jobs for three job plants
    copies, unfair = plant_copies(profile, CV_TEXT, baseline_all_supported(profile))

    assert unfair == []
    assert [[p.kind for p in c] for c in copies] == [
        ["fake_skill", "drop_job", "shift_start"], ["fake_company"]]
    assert copies[1][0].path == "experience[0]"


def test_plant_copies_report_kinds_without_a_fair_target() -> None:
    profile = make_profile()
    baseline = result_of([("skills[0]", "supported"), ("experience[0]", "unsupported"),
                          ("experience[1]", "partly_wrong")])
    copies, unfair = plant_copies(profile, CV_TEXT, baseline)
    assert unfair == ["drop_job", "shift_start", "fake_company"]
    assert [[p.kind for p in c] for c in copies] == [["fake_skill"]]


def test_drop_job_ignores_a_missed_job_that_only_shares_the_title() -> None:
    profile = make_profile()
    profile.experience[1] = job("Data Engineer", "Globex Testing Co", "2019", "Data Engineer")
    baseline = baseline_all_supported(profile)
    dropped = plant_error(profile, "drop_job", CV_TEXT, baseline)
    assert not caught(dropped, missed_jobs("Data Engineer"), baseline)
    assert caught(dropped, missed_jobs("Data Engineer\nAcme Widgets LLC"), baseline)


def test_plant_report() -> None:
    report = plant_report([("fake_skill", True), ("fake_skill", False), ("drop_job", None)])
    assert report["fake_skill"] == {"planted": 2, "caught": 1, "skipped": 0, "catch_rate": 0.5}
    assert report["drop_job"]["skipped"] == 1 and report["drop_job"]["catch_rate"] is None
    assert report["all"]["planted"] == 2


def test_label_agreement() -> None:
    text = "Data Engineer at Acme Widgets LLC. Junior Analyst at Globex Testing Co."
    resume = Resume(index=0, text=text, spans=[
        GoldSpan(label="Designation", start=0, end=13),
        GoldSpan(label="Companies worked at", start=17, end=33),
    ])
    profile = make_profile()
    judged = result_of([("experience[0]", "supported"), ("experience[1]", "supported"),
                        ("skills[0]", "unsupported")])

    counts = label_agreement(resume, profile, judged)

    assert counts == Counter({"supported_labeled": 1, "supported_unlabeled": 1})
    report = agreement_report(counts)
    assert (report["items"], report["agreement"], report["supported_when_labeled"]) == (
        2, 0.5, 1.0)


# --- the script ------------------------------------------------------------

EXTRACTED = {
    "summary": None,
    "skills": [{"name": "Python", "kind": "hard", "found_in": "skills_list",
                "quote": "Python, SQL"}],
    "experience": [{"title": "Data Engineer", "company": "Acme Widgets LLC", "location": None,
                    "start": "2021", "end": "2024", "description": None, "skills_used": [],
                    "quote": "Data Engineer"}],
    "education": [],
    "languages": [],
    "certifications": [],
}
JUDGED = {
    "items": [verdict("skills[0]"), verdict("experience[0]")],
    "missed": [{"field": "skills", "quote": "Docker"}],
    "text_unreadable": False,
}


@pytest.fixture
def input_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "cvs"
    folder.mkdir()
    shutil.copy(FIXTURE_PDF, folder / "1.pdf")
    shutil.copy(FIXTURE_PDF, folder / "2.pdf")
    return folder


@pytest.fixture
def recordings(tmp_path: Path, input_dir: Path) -> Path:
    """Extraction and baseline judge responses for 1.pdf only, through the real code."""
    directory = tmp_path / "recordings"
    doc = parse(input_dir / "1.pdf")
    sections, _ = split_sections(doc)
    extractor = RecordingToolCaller(FakeToolCaller([reply(EXTRACTED)]), directory)
    result = extract_profile(doc, sections, extractor, use_cache=False, today=dt.date.today())
    judge = RecordingToolCaller(FakeToolCaller([reply(JUDGED)]), directory)
    judge_profile(doc, sections, result.profile, judge)
    return directory


def run(tmp_path: Path, input_dir: Path, recordings: Path, *extra: str):
    out = tmp_path / "report.json"
    code = runner.main(["--input", str(input_dir), "--recordings", str(recordings),
                        "--out", str(out), "--verdicts", str(tmp_path / "verdicts"),
                        "--env-file", str(tmp_path / "no.env"), *extra])
    return code, json.loads(out.read_text(encoding="utf-8")) if out.is_file() else None


def test_replay_end_to_end(tmp_path: Path, input_dir: Path, recordings: Path, capsys) -> None:
    code, report = run(tmp_path, input_dir, recordings)

    assert code == 0
    assert (report["files_selected"], report["files_judged"]) == (2, 2)
    assert report["mode"] == "replay"
    # Both files hold the same CV, so 2.pdf replays 1.pdf's recordings; the four
    # planted copies per CV have none.
    assert report["skipped"] == {"plant_no_recording": 8}
    assert report["planted_errors"]["all"]["planted"] == 0
    assert report["overall"]["all"]["supported"] == 4
    assert report["overall"]["fields"]["skills"]["missed"] == 2
    assert report["per_cv"]["1.pdf"]["language"] == "en"
    assert report["dataturks_agreement"] is None
    dumped = json.dumps(report)
    assert "Docker" not in dumped and "olena" not in dumped.lower()
    spot_check = json.loads((tmp_path / "verdicts" / "1.pdf.json").read_text(encoding="utf-8"))
    assert spot_check["missed"] == [{"field": "skills", "quote": "Docker"}]
    assert "judged 2/2 CVs" in capsys.readouterr().out


def test_replay_skips_files_without_recordings(tmp_path: Path, input_dir: Path) -> None:
    code, report = run(tmp_path, input_dir, tmp_path / "empty", "--no-plants")
    assert code == 0
    assert report["files_judged"] == 0 and report["skipped"] == {"no_recording": 2}
    assert report["planted_errors"] is None


def test_live_without_key_sends_nothing(tmp_path: Path, input_dir: Path) -> None:
    code, report = run(tmp_path, input_dir, tmp_path / "rec", "--mode", "live")
    assert code == 2 and report is None


def test_live_gemini_reads_its_key_from_the_env_file(tmp_path: Path, input_dir: Path,
                                                     monkeypatch, capsys) -> None:
    env = tmp_path / "keys.env"
    env.write_text("LLM_PROVIDER=gemini\n", encoding="utf-8")
    out = tmp_path / "report.json"
    code = runner.main(["--input", str(input_dir), "--mode", "live", "--env-file", str(env),
                        "--out", str(out)])
    assert code == 2 and not out.exists()
    assert "GEMINI_API_KEY" in capsys.readouterr().err
