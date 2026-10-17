"""Judge JM-11 extraction on CVs without labels (JM-15).

Every PDF or DOCX in --input is parsed, sectioned and extracted as in the
pipeline, then a second LLM call judges the extraction against the same masked
CV. Calibration runs alongside: each CV also gets copies of its extraction with
planted mistakes (packed so most CVs need one or two extra judge calls, not
four), and with --dataturks the judge also runs on labeled resumes so its
verdicts can be compared with the labels.

The report holds counts and rates only: no CV text, quotes, names or judge
reasons. Per-CV verdicts with reasons, for the human spot check, go to
--verdicts (default under gitignored data/eval/), keyed by file name.

Replay mode (default) is offline and serves responses recorded by an earlier
live run, skipping CVs without one. Live mode calls the API of LLM_PROVIDER
("anthropic", the default, needs ANTHROPIC_API_KEY; "gemini" needs
GEMINI_API_KEY; "ollama" needs no key and runs a local model) and records every
response, so a re-run never pays twice. Keys and LLM_PROVIDER may sit in the
repo's gitignored .env. JUDGE_MODEL picks the judge's model; extraction keeps
ANTHROPIC_MODEL, GEMINI_MODEL or OLLAMA_MODEL.

Usage (from the repo root):
    uv run python scripts/judge_extraction.py
    uv run python scripts/judge_extraction.py --mode live --limit 3
    LLM_PROVIDER=gemini JUDGE_MODEL=gemini-2.5-flash \\
        uv run python scripts/judge_extraction.py --mode live
    uv run python scripts/judge_extraction.py --mode live \\
        --dataturks "data/external/dataturks/Entity Recognition in Resumes.json"
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

from jobpdf.extraction import parse
from jobpdf.extraction.eval_dataturks import load_dataturks
from jobpdf.extraction.extract import ExtractionError, extract_profile
from jobpdf.extraction.judge import (
    JUDGE_PROMPT_VERSION,
    CVJudgement,
    JudgeError,
    aggregate,
    detect_language,
    judge_profile,
    weak_grounding_items,
)
from jobpdf.extraction.judge_calibration import (
    PLANT_KINDS,
    agreement_report,
    caught,
    label_agreement,
    plant_copies,
    plant_report,
)
from jobpdf.extraction.llm import (
    ENV_FILE,
    LLMResponseError,
    MissingAPIKeyError,
    ReplayMissError,
    ReplayOrRecord,
    ReplayToolCaller,
    ToolCaller,
    api_errors,
    default_caller,
    load_env_file,
    provider,
)
from jobpdf.extraction.models import ParseError
from jobpdf.extraction.plaintext import from_plain_text
from jobpdf.extraction.prompt import PROMPT_VERSION, mask_pii
from jobpdf.extraction.sections import split_sections

DEFAULT_INPUT = Path("data/pdf_dataset")
DEFAULT_RECORDINGS = Path("data/eval/judge/recordings")
DEFAULT_OUT = Path("data/eval/judge/report.json")
DEFAULT_VERDICTS = Path("data/eval/judge/verdicts")
DEFAULT_DATATURKS_LIMIT = 20
SUFFIXES = (".pdf", ".docx")
KEY_NAMES = {"anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}
_SKIPPABLE = (ReplayMissError, ExtractionError, JudgeError, LLMResponseError)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    load_env_file(args.env_file)
    key_name = KEY_NAMES.get(provider())  # None: a local provider needs no key
    if args.mode == "live" and key_name and not os.environ.get(key_name):
        _error(f"--mode live needs {key_name} in the environment or .env; nothing was sent.")
        return 2
    if not args.input.is_dir():
        _error(f"input folder not found: {args.input}")
        return 1
    try:
        extractor, judge, api_errors = _callers(args)
    except MissingAPIKeyError:
        _error(f"{key_name} is not set; nothing was sent.")
        return 2

    files = sorted(
        (p for p in args.input.iterdir() if p.suffix.lower() in SUFFIXES), key=_natural
    )
    if args.limit is not None:
        files = files[: args.limit]

    judged: list[CVJudgement] = []
    outcomes: list[tuple[str, bool | None]] = []
    skipped: Counter = Counter()
    models: set[str] = set()
    try:
        for path in files:
            cv = _judge_file(path, args, extractor, judge, outcomes, skipped, models)
            if cv is not None:
                judged.append(cv)
        dataturks = _dataturks(args, extractor, judge, skipped, models)
    except api_errors as exc:
        _error(
            f"LLM API call failed ({type(exc).__name__}); responses recorded so far "
            f"are kept in {args.recordings}."
        )
        return 3

    report = {
        "input": str(args.input),
        "input_sha256": _folder_sha256(files),
        "files_selected": len(files),
        "files_judged": len(judged),
        "skipped": dict(sorted(skipped.items())),
        "mode": args.mode,
        "models": sorted(models),
        "extraction_prompt_version": PROMPT_VERSION,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        **aggregate(judged),
        "planted_errors": plant_report(outcomes) if not args.no_plants else None,
        "dataturks_agreement": dataturks,
        "per_cv": {
            cv.cv_id: {
                "language": cv.language,
                "unreadable": cv.result.text_unreadable,
                **aggregate([cv])["overall"]["all"],
            }
            for cv in judged
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _print_summary(report, args.out)
    return 0


def _judge_file(path, args, extractor, judge, outcomes, skipped, models) -> CVJudgement | None:
    try:
        doc = parse(path)
    except ParseError:
        skipped["parse_error"] += 1
        _warn(f"{path.name}: could not be parsed, skipped")
        return None
    sections, _ = split_sections(doc)
    try:
        extraction = extract_profile(doc, sections, extractor, use_cache=False)
        baseline = judge_profile(doc, sections, extraction.profile, judge)
    except _SKIPPABLE as exc:
        skipped[_skip_reason(exc)] += 1
        _warn(f"{path.name}: {_skip_reason(exc)}, skipped")
        return None
    models.update({f"extraction:{extraction.model}", f"judge:{baseline.model}"})
    masked_text, _ = mask_pii(doc.full_text)
    cv = CVJudgement(
        cv_id=path.name,
        language=detect_language(masked_text),
        result=baseline,
        weak_grounding=weak_grounding_items(extraction.profile, doc, sections),
    )
    _write_verdicts(args.verdicts, path.name, cv)

    if not args.no_plants and not baseline.text_unreadable:
        copies, unfair = plant_copies(extraction.profile, masked_text, baseline)
        outcomes.extend((kind, None) for kind in unfair)
        for plants in copies:  # one judge call per copy, however many plants it holds
            try:
                result = judge_profile(doc, sections, plants[0].profile, judge)
            except _SKIPPABLE as exc:
                skipped[f"plant_{_skip_reason(exc)}"] += len(plants)
                continue
            outcomes.extend((p.kind, caught(p, result, baseline)) for p in plants)
    return cv


def _dataturks(args, extractor, judge, skipped, models) -> dict | None:
    if args.dataturks is None:
        return None
    if not args.dataturks.is_file():
        _warn(f"DataTurks dataset not found at {args.dataturks}; agreement not measured")
        return None
    resumes, _, _ = load_dataturks(args.dataturks)
    counts: Counter = Counter()
    used = 0
    for resume in resumes[: args.dataturks_limit]:
        doc = from_plain_text(resume.text)
        sections, _ = split_sections(doc)
        try:
            extraction = extract_profile(doc, sections, extractor, use_cache=False)
            result = judge_profile(doc, sections, extraction.profile, judge)
        except _SKIPPABLE as exc:
            skipped[f"dataturks_{_skip_reason(exc)}"] += 1
            continue
        models.update({f"extraction:{extraction.model}", f"judge:{result.model}"})
        counts.update(label_agreement(resume, extraction.profile, result))
        used += 1
    return {"resumes": used, **agreement_report(counts)}


def _write_verdicts(folder: Path, name: str, cv: CVJudgement) -> None:
    """Verdicts with reasons for the human spot check; may hold CV wording, so local only."""
    folder.mkdir(parents=True, exist_ok=True)
    data = {
        "file": name,
        "language": cv.language,
        "text_unreadable": cv.result.text_unreadable,
        "verdicts": [v.model_dump() for v in cv.result.verdicts],
        "missed": [m.model_dump() for m in cv.result.missed],
        "unjudged_paths": cv.result.unjudged_paths,
        "human_check": "mark each verdict right or wrong here",
    }
    (folder / f"{name}.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _skip_reason(exc: Exception) -> str:
    if isinstance(exc, ReplayMissError):
        return "no_recording"
    if isinstance(exc, JudgeError):
        return "judge_error"
    return "extraction_error"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--mode", choices=["replay", "live"], default="replay")
    parser.add_argument("--recordings", type=Path, default=DEFAULT_RECORDINGS)
    parser.add_argument("--limit", type=int, default=None, help="only the first N files")
    parser.add_argument("--no-plants", action="store_true", help="skip the planted errors")
    parser.add_argument("--dataturks", type=Path, default=None,
                        help="DataTurks JSON lines file for the label agreement check")
    parser.add_argument("--dataturks-limit", type=int, default=DEFAULT_DATATURKS_LIMIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verdicts", type=Path, default=DEFAULT_VERDICTS)
    parser.add_argument("--env-file", type=Path, default=ENV_FILE,
                        help="KEY=VALUE file read for API keys and LLM_PROVIDER")
    return parser.parse_args(argv)


def _callers(
    args: argparse.Namespace,
) -> tuple[ToolCaller, ToolCaller, tuple[type[Exception], ...]]:
    """Extraction and judge callers for the mode, and the API errors that stop a live run.

    Both share one recordings folder: replay keys include the system prompt,
    so extraction and judge responses never collide.
    """
    if args.mode == "replay":
        return ReplayToolCaller(args.recordings), ReplayToolCaller(args.recordings), ()
    extractor = ReplayOrRecord(default_caller(), args.recordings)
    judge_model = os.environ.get("JUDGE_MODEL") or None
    judge = ReplayOrRecord(default_caller(model=judge_model), args.recordings)
    return extractor, judge, api_errors()


def _natural(path: Path) -> tuple[int, str]:
    """"2.pdf" before "10.pdf"."""
    stem = path.stem
    return (int(stem), "") if stem.isdigit() else (sys.maxsize, stem)


def _folder_sha256(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8") + b"\x00" + path.read_bytes())
    return digest.hexdigest()


def _print_summary(report: dict, out: Path) -> None:
    print(
        f"judged {report['files_judged']}/{report['files_selected']} CVs "
        f"({report['cvs_unreadable']} unreadable, skipped {report['skipped'] or 'none'})"
    )
    print(f"{'field':<16}{'judged':>7}{'supported':>11}{'errors':>8}{'missed':>8}{'miss%':>7}")
    overall = report["overall"]
    for name, m in [*overall["fields"].items(), ("all", overall["all"])]:
        print(
            f"{name:<16}{m['judged']:>7}{_fmt(m['supported_rate']):>11}"
            f"{_fmt(m['error_rate']):>8}{m['missed']:>8}{_fmt(m['miss_rate']):>7}"
        )
    plants = report["planted_errors"]
    if plants:
        rates = ", ".join(
            f"{kind} {_fmt(plants[kind]['catch_rate'])} ({plants[kind]['planted']})"
            for kind in PLANT_KINDS
        )
        print(f"planted errors caught: {rates}")
    agreement = report["dataturks_agreement"]
    if agreement:
        print(
            f"DataTurks label agreement {_fmt(agreement['agreement'])} "
            f"over {agreement['items']} items in {agreement['resumes']} resumes"
        )
    print(f"report written to {out}")


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _warn(message: str) -> None:
    print(f"WARN {message}", file=sys.stderr)


def _error(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
