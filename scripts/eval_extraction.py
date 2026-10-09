"""Evaluate JM-11 extraction against the DataTurks labeled resumes (JM-14).

Each resume is masked (labeled names and emails, plus any email, phone or URL),
turned into a ParsedDocument, sectioned and extracted, then scored against the
human labels. The JSON report holds counts and metrics only: no CV text,
names, emails or quotes. Per-resume metrics are keyed by line index.

Replay mode (default) is offline: it serves responses recorded by an earlier
live run and skips resumes without one. Live mode calls the Anthropic API
(needs ANTHROPIC_API_KEY in the environment) and records every response, so a
re-run never sends the same prompt twice. Recordings come from real resumes
and stay under gitignored data/eval/, never under tests/.

Usage (from the repo root):
    uv run python scripts/eval_extraction.py
    uv run python scripts/eval_extraction.py --mode live --limit 5
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

from jobpdf.extraction.eval_dataturks import load_dataturks
from jobpdf.extraction.eval_metrics import (
    MATCH_THRESHOLD,
    NOT_EVALUATED,
    TEXT_FIELDS,
    Counts,
    ResumeMetrics,
    evaluate_resume,
)
from jobpdf.extraction.extract import ExtractionError, extract_profile
from jobpdf.extraction.llm import (
    AnthropicToolCaller,
    LLMResponseError,
    MissingAPIKeyError,
    RecordingToolCaller,
    ReplayMissError,
    ReplayToolCaller,
    ToolCaller,
    ToolCallResult,
)
from jobpdf.extraction.plaintext import from_plain_text
from jobpdf.extraction.prompt import PROMPT_VERSION
from jobpdf.extraction.sections import split_sections

DEFAULT_DATASET = Path("data/external/dataturks/Entity Recognition in Resumes.json")
DEFAULT_RECORDINGS = Path("data/eval/extraction/recordings")
DEFAULT_OUT = Path("data/eval/extraction/report.json")
FIELDS = [*TEXT_FIELDS, "graduation_year"]


class ReplayOrRecord:
    """Serve a recorded response if there is one, else ask ``live`` and record it.

    A recorded truncated answer (stop_reason "max_tokens") counts as a miss, so
    the extractor's bigger-budget retry really reaches the API and overwrites it.
    """

    def __init__(self, live: ToolCaller, recordings: Path) -> None:
        self._replay = ReplayToolCaller(recordings)
        self._record = RecordingToolCaller(live, recordings)
        self.model = live.model

    def call(self, *, system: str, user: str, tool: dict, max_tokens: int) -> ToolCallResult:
        kwargs = {"system": system, "user": user, "tool": tool, "max_tokens": max_tokens}
        try:
            recorded = self._replay.call(**kwargs)
        except ReplayMissError:
            return self._record.call(**kwargs)
        if recorded.stop_reason == "max_tokens":
            return self._record.call(**kwargs)
        return recorded


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.mode == "live" and not os.environ.get("ANTHROPIC_API_KEY"):
        _error("--mode live needs ANTHROPIC_API_KEY in the environment; nothing was sent.")
        return 2
    if not args.dataset.is_file():
        _error(f"dataset not found: {args.dataset} (place it under data/external/dataturks/)")
        return 1
    try:
        caller, api_errors = _caller(args)
    except MissingAPIKeyError:
        _error("ANTHROPIC_API_KEY is not set; nothing was sent.")
        return 2

    resumes, stats, load_warnings = load_dataturks(args.dataset)
    for warning in load_warnings:
        _warn(warning)
    selected = resumes[: args.limit] if args.limit is not None else resumes

    total = ResumeMetrics(fields={name: Counts() for name in FIELDS})
    per_resume: dict[str, dict] = {}
    models: set[str] = set()
    missing = errors = extraction_warnings = 0
    for resume in selected:
        doc = from_plain_text(resume.text)
        sections, _ = split_sections(doc)
        try:
            result = extract_profile(doc, sections, caller, use_cache=False)
        except ReplayMissError:
            missing += 1
            _warn(f"resume {resume.index}: no recorded response, skipped")
            continue
        except (ExtractionError, LLMResponseError):
            errors += 1
            _warn(f"resume {resume.index}: extraction failed, skipped")
            continue
        except api_errors as exc:
            _error(
                f"Anthropic API call failed on resume {resume.index} ({type(exc).__name__}); "
                f"responses recorded so far are kept in {args.recordings}."
            )
            return 3
        models.add(result.model)
        extraction_warnings += len(result.warnings)  # counted only: they quote CV text
        metrics = evaluate_resume(resume, result.profile)
        total = total + metrics
        per_resume[str(resume.index)] = metrics.as_report()

    summary = total.as_report()
    report = {
        "dataset_sha256": _sha256(args.dataset),
        "dataset_records": stats.records,
        "resumes_selected": len(selected),
        "resumes_evaluated": len(per_resume),
        "invalid_records": stats.invalid_records,
        "invalid_spans": stats.invalid_spans,
        "amp_repaired_spans": stats.amp_repaired_spans,
        "duplicate_spans": stats.duplicate_spans,
        "unlabeled_annotations": stats.unlabeled_annotations,
        "missing_recordings": missing,
        "extraction_errors": errors,
        "extraction_warnings": extraction_warnings,
        "unlocated_quotes": total.skills.unlocated_quotes,
        "contact_leaks": total.contact_leaks,
        "label_counts": stats.label_counts,
        "mode": args.mode,
        "models": sorted(models),
        "prompt_version": PROMPT_VERSION,
        "match_threshold": MATCH_THRESHOLD,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "fields": summary["fields"],
        "skills": summary["skills"],
        "not_evaluated": NOT_EVALUATED,
        "per_resume": per_resume,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _print_summary(report, args.out)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--mode", choices=["replay", "live"], default="replay")
    parser.add_argument("--recordings", type=Path, default=DEFAULT_RECORDINGS)
    parser.add_argument("--limit", type=int, default=None, help="only the first N resumes")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def _caller(args: argparse.Namespace) -> tuple[ToolCaller, tuple[type[Exception], ...]]:
    """The caller for the mode, and the API errors that should stop a live run."""
    if args.mode == "replay":
        return ReplayToolCaller(args.recordings), ()
    import anthropic  # deferred like in llm.py: replay mode stays offline and cheap

    return ReplayOrRecord(AnthropicToolCaller(), args.recordings), (anthropic.APIError,)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _print_summary(report: dict, out: Path) -> None:
    print(
        f"evaluated {report['resumes_evaluated']}/{report['resumes_selected']} resumes "
        f"(missing recordings {report['missing_recordings']}, "
        f"extraction errors {report['extraction_errors']}); "
        f"invalid spans {report['invalid_spans']}, "
        f"repaired &amp; shifts {report['amp_repaired_spans']}"
    )
    print(f"{'field':<16}{'precision':>10}{'recall':>8}{'f1':>8}{'tp':>6}{'fp':>6}{'fn':>6}")
    for name, m in report["fields"].items():
        print(
            f"{name:<16}{_fmt(m['precision']):>10}{_fmt(m['recall']):>8}{_fmt(m['f1']):>8}"
            f"{m['tp']:>6}{m['fp']:>6}{m['fn']:>6}"
        )
    skills = report["skills"]
    print(
        f"skills: quote precision {_fmt(skills['quote_precision'])}, "
        f"span coverage {_fmt(skills['span_coverage'])}, "
        f"unlocated quotes {skills['unlocated_quotes']}/{skills['quotes']}"
    )
    print(f"contact leaks: {report['contact_leaks']}")
    print(f"report written to {out}")


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _warn(message: str) -> None:
    print(f"WARN {message}", file=sys.stderr)


def _error(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
