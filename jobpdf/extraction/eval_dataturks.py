"""Load and mask the DataTurks "Resume Entities for NER" dataset (JM-14).

The file is JSON lines: ``{"content": str, "annotation": [{"label": [str],
"points": [{"start", "end", "text"}]}], "extras": null}``. End offsets are
inclusive. Spans become half-open ``GoldSpan``s after repair; a span that can't
be repaired is counted and warned about, never silently dropped.

Some ``content`` strings hold the HTML entity "&amp;" where the annotator saw a
plain "&", so every "&amp;" before a span puts it 4 characters later than its
offsets say. ``repair_amp_shift`` maps such offsets back into ``content``.

These are real resumes: warnings name only the resume index, label and
offsets, and the original names/emails live only in memory (excluded from dumps).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from jobpdf.extraction.prompt import MASK_CHAR, mask_pii

__all__ = ["MASK_CHAR", "GoldSpan", "LoadStats", "Resume", "load_dataturks", "mask_contacts"]

# Labels whose text is replaced by MASK_CHAR before anything reaches the LLM.
CONTACT_LABELS = frozenset({"Name", "Email Address"})
# The escaped ampersand that shifts annotator offsets (see module docstring).
AMP_ENTITY = "&amp;"


class GoldSpan(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    start: int
    end: int  # exclusive: text[start:end] is the labeled text


class Resume(BaseModel):
    index: int  # 0-based line number in the dataset file
    text: str  # masked; same length as the original, so offsets stay valid
    spans: list[GoldSpan]
    # The masked names/emails, kept only for the leakage check; never dumped or shown.
    contact_strings: list[str] = Field(default_factory=list, exclude=True, repr=False)


class LoadStats(BaseModel):
    records: int = 0
    invalid_records: int = 0
    label_counts: dict[str, int] = Field(default_factory=dict)
    invalid_spans: int = 0
    amp_repaired_spans: int = 0  # recovered only by repair_amp_shift
    duplicate_spans: int = 0
    unlabeled_annotations: int = 0


class _RawPoint(BaseModel):
    start: int
    end: int
    text: str


class _RawAnnotation(BaseModel):
    label: list[str] | str
    points: list[_RawPoint]


class _RawRecord(BaseModel):
    content: str
    annotation: list[_RawAnnotation] | None = None


def load_dataturks(path: Path) -> tuple[list[Resume], LoadStats, list[str]]:
    """Parse, repair, de-duplicate and mask every record.

    Returns the resumes, counts for the report and warnings (index/offsets only).
    """
    stats = LoadStats()
    labels: Counter[str] = Counter()
    resumes: list[Resume] = []
    warnings: list[str] = []
    # split("\n"), not splitlines(): JSON strings may hold a raw U+2028.
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").split("\n")):
        if not line.strip():
            continue
        stats.records += 1
        try:
            record = _RawRecord.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError):
            stats.invalid_records += 1
            warnings.append(f"resume {index}: record is not valid DataTurks JSON, skipped")
            continue
        spans = _gold_spans(index, record, stats, labels, warnings)
        masked, contacts = mask_contacts(record.content, spans)
        resumes.append(Resume(index=index, text=masked, spans=spans, contact_strings=contacts))
    stats.label_counts = dict(labels.most_common())
    return resumes, stats, warnings


def _gold_spans(
    index: int,
    record: _RawRecord,
    stats: LoadStats,
    labels: Counter[str],
    warnings: list[str],
) -> list[GoldSpan]:
    spans: dict[GoldSpan, None] = {}  # ordered set
    for annotation in record.annotation or []:
        names = [annotation.label] if isinstance(annotation.label, str) else annotation.label
        if not names:
            stats.unlabeled_annotations += 1
            offsets = [(p.start, p.end) for p in annotation.points]
            warnings.append(f"resume {index}: annotation without a label at {offsets}")
            continue
        for point in annotation.points:
            repaired = repair_span(record.content, point.start, point.end, point.text)
            amp_repaired = False
            if repaired is None:
                repaired = repair_amp_shift(record.content, point.start, point.end, point.text)
                amp_repaired = repaired is not None
            for label in names:
                stats.amp_repaired_spans += amp_repaired
                labels[label] += 1
                if repaired is None:
                    stats.invalid_spans += 1
                    warnings.append(
                        f"resume {index}: {label} span [{point.start}, {point.end}] "
                        "does not match its text, excluded from gold"
                    )
                    continue
                span = GoldSpan(label=label, start=repaired[0], end=repaired[1])
                if span in spans:
                    stats.duplicate_spans += 1
                spans[span] = None
    return list(spans)


def repair_span(content: str, start: int, end: int, text: str) -> tuple[int, int] | None:
    """Half-open offsets of ``text`` at the given position, or None.

    Tries the inclusive end (DataTurks' convention), then an exclusive end, then
    the same windows with surrounding whitespace ignored (trimmed bounds).
    """
    if start < 0 or end < start:
        return None
    windows = [(start, end + 1), (start, end)]
    for s, e in windows:
        if e <= len(content) and content[s:e] == text:
            return s, e
    stripped = text.strip()
    if not stripped:
        return None
    for s, e in windows:
        window = content[s:e]
        if window.strip() == stripped:
            lead = len(window) - len(window.lstrip())
            return s + lead, s + lead + len(stripped)
    return None


def repair_amp_shift(content: str, start: int, end: int, text: str) -> tuple[int, int] | None:
    """Half-open offsets in ``content`` for offsets counted as if "&amp;" were "&".

    ``start``/``end`` (inclusive) index the unescaped text. The span is accepted
    only if the mapped window is exactly ``text``, or is ``text`` once its own
    "&amp;" is read as "&"; anything else stays invalid.
    """
    if AMP_ENTITY not in content:
        return None
    # positions[u] = index in content of unescaped character u (plus one past the end).
    positions: list[int] = []
    at = 0
    while at < len(content):
        positions.append(at)
        at += len(AMP_ENTITY) if content.startswith(AMP_ENTITY, at) else 1
    positions.append(len(content))
    if start < 0 or end < start or end + 1 >= len(positions):
        return None
    s, e = positions[start], positions[end + 1]
    window = content[s:e]
    if window == text or window.replace(AMP_ENTITY, "&") == text:
        return s, e
    return None


def mask_contacts(content: str, spans: list[GoldSpan]) -> tuple[str, list[str]]:
    """Mask Name/Email spans and unlabeled contact details; the length is unchanged.

    Returns the masked text and the distinct original Name/Email strings.
    """
    chars = list(content)
    originals: dict[str, None] = {}
    for span in spans:
        if span.label not in CONTACT_LABELS:
            continue
        original = content[span.start : span.end].strip()
        if original:
            originals[original] = None
        chars[span.start : span.end] = MASK_CHAR * (span.end - span.start)
    masked, _ = mask_pii("".join(chars))  # unlabeled emails, phones and URLs
    if len(masked) != len(content):
        raise AssertionError("masking changed the text length; offsets would break")
    return masked, list(originals)
