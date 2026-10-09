"""Metrics comparing JM-11 extraction output with DataTurks gold spans (JM-14).

Text fields are compared through the shared ``normalize()``, with punctuation
that only changes spelling removed ("B.E." == "B.E" == "BE"), and fuzzy-matched
one-to-one; skills are judged by where their quotes sit in the CV. Results are
counts only, so a report built from them never contains CV text.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import astuple, dataclass, field

from rapidfuzz import fuzz

from jobpdf.extraction.eval_dataturks import AMP_ENTITY, MASK_CHAR, Resume
from jobpdf.extraction.prompt import mask_pii
from jobpdf.extraction.schema import CandidateProfile
from jobpdf.normalization.text import normalize

MATCH_THRESHOLD = 85  # rapidfuzz token_set_ratio, 0-100
# A labeled name/email shorter than this (normalized) is too short to search
# for in the output without false alarms.
MIN_LEAK_NEEDLE_CHARS = 4

# Text fields: report name -> DataTurks label.
TEXT_FIELDS = {
    "designation": "Designation",
    "companies": "Companies worked at",
    "degree": "Degree",
    "college": "College Name",
}
YEAR_LABEL = "Graduation Year"
SKILLS_LABEL = "Skills"
NOT_EVALUATED = {
    "Location": (
        "mostly the candidate's own location, which is personal data; the schema has "
        "only per-job location and the prompt forbids addresses"
    ),
    "Years of Experience": (
        "the schema has no stated-years field; total_experience_months is computed "
        "from job dates and is not comparable to a free-text phrase"
    ),
    "Name": "masked before extraction; used only for the contact-leakage check",
    "Email Address": "masked before extraction; used only for the contact-leakage check",
    "UNKNOWN": "no corresponding field",
}
_YEAR = re.compile(r"(?<!\d)(?:19|20)\d\d(?!\d)")
# Dots and apostrophes only change how an abbreviation is spelled, so they are
# dropped: "B.E." -> "be", "M.B.A" -> "mba". Separators become spaces:
# "Pvt. Ltd., Pune" -> "pvt ltd pune". Everything else stays ("c++", "c#", "&"),
# so different names never collapse into the same key.
_SPELLING_ONLY = str.maketrans("", "", ".'’")
_SEPARATORS = re.compile(r"[,;:()\[\]{}/\\|\-–—_\"“”]+")


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: Counts) -> Counts:
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    def as_report(self) -> dict:
        tp, fp, fn = self.tp, self.fp, self.fn
        return {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, tp + fn),
            "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        }


@dataclass
class SkillCounts:
    quotes: int = 0  # distinct skill quotes in the output
    unlocated_quotes: int = 0  # not found verbatim anywhere in the CV
    quotes_inside: int = 0  # located quotes lying inside some Skills span
    spans: int = 0  # gold Skills spans
    spans_covered: int = 0  # Skills spans containing at least one quote

    def __add__(self, other: SkillCounts) -> SkillCounts:
        return SkillCounts(*(a + b for a, b in zip(astuple(self), astuple(other), strict=True)))

    def as_report(self) -> dict:
        located = self.quotes - self.unlocated_quotes
        return {
            "quotes": self.quotes,
            "unlocated_quotes": self.unlocated_quotes,
            "quotes_inside": self.quotes_inside,
            "quote_precision": _ratio(self.quotes_inside, located),  # of located quotes
            "spans": self.spans,
            "spans_covered": self.spans_covered,
            "span_coverage": _ratio(self.spans_covered, self.spans),
        }


@dataclass
class ResumeMetrics:
    fields: dict[str, Counts] = field(default_factory=dict)
    skills: SkillCounts = field(default_factory=SkillCounts)
    contact_leaks: int = 0

    def __add__(self, other: ResumeMetrics) -> ResumeMetrics:
        names = list(dict.fromkeys([*self.fields, *other.fields]))
        return ResumeMetrics(
            fields={n: self.fields.get(n, Counts()) + other.fields.get(n, Counts()) for n in names},
            skills=self.skills + other.skills,
            contact_leaks=self.contact_leaks + other.contact_leaks,
        )

    def as_report(self) -> dict:
        return {
            "fields": {name: counts.as_report() for name, counts in self.fields.items()},
            "skills": self.skills.as_report(),
            "contact_leaks": self.contact_leaks,
        }


def match_strings(
    gold: Iterable[str], predicted: Iterable[str], threshold: float = MATCH_THRESHOLD
) -> Counts:
    """Greedy one-to-one fuzzy matching, best-scoring pairs first.

    Both sides become match keys and are de-duplicated by key, so a gold
    entity labeled twice, or two roles at one employer, count once. Strings
    with an empty key (blank or punctuation only) are ignored.
    """
    gold_keys = _distinct(gold)
    pred_keys = _distinct(predicted)
    pairs = sorted(
        (
            (score, g, p)
            for g, gold_key in enumerate(gold_keys)
            for p, pred_key in enumerate(pred_keys)
            if (score := fuzz.token_set_ratio(gold_key, pred_key)) >= threshold
        ),
        key=lambda pair: (-pair[0], pair[1], pair[2]),
    )
    used_gold: set[int] = set()
    used_pred: set[int] = set()
    for _, g, p in pairs:
        if g not in used_gold and p not in used_pred:
            used_gold.add(g)
            used_pred.add(p)
    tp = len(used_gold)
    return Counts(tp=tp, fp=len(pred_keys) - tp, fn=len(gold_keys) - tp)


def year_counts(gold: set[str], predicted: set[str]) -> Counts:
    return Counts(tp=len(gold & predicted), fp=len(predicted - gold), fn=len(gold - predicted))


def skill_quote_counts(
    full_text: str, quotes: Iterable[str], spans: list[tuple[int, int]]
) -> SkillCounts:
    """Where the distinct skill quotes sit relative to the gold Skills spans."""
    counts = SkillCounts(spans=len(spans))
    covered: set[int] = set()
    for quote in dict.fromkeys(quotes):
        counts.quotes += 1
        # Stand-in until JM-12 grounding exists: exact search, every occurrence.
        occurrences = list(_occurrences(full_text, quote))
        if not occurrences:
            counts.unlocated_quotes += 1
            continue
        inside = {
            i for i, (start, end) in enumerate(spans)
            for at in occurrences if start <= at and at + len(quote) <= end
        }
        counts.quotes_inside += bool(inside)
        covered |= inside
    counts.spans_covered = len(covered)
    return counts


def find_leaks(output: object, contact_strings: Iterable[str]) -> int:
    """Number of output strings holding a masked name/email or any contact detail."""
    needles = [
        n for s in contact_strings if len(n := normalize(s)) >= MIN_LEAK_NEEDLE_CHARS
    ]
    return sum(
        1 for value in _strings(output)
        if MASK_CHAR in value
        or mask_pii(value)[1]
        or any(needle in normalize(value) for needle in needles)
    )


def evaluate_resume(resume: Resume, profile: CandidateProfile) -> ResumeMetrics:
    gold = {label: [] for label in [*TEXT_FIELDS.values(), YEAR_LABEL, SKILLS_LABEL]}
    for span in resume.spans:
        if span.label in gold:
            gold[span.label].append(span)

    def gold_text(label: str) -> list[str]:
        return [resume.text[s.start : s.end] for s in gold[label]]

    jobs, schools = profile.experience, profile.education
    predicted = {
        "designation": [j.title for j in jobs],
        "companies": [j.company for j in jobs if j.company],
        "degree": [e.degree for e in schools if e.degree],
        "college": [e.institution for e in schools if e.institution],
    }
    fields = {
        name: match_strings(gold_text(label), predicted[name])
        for name, label in TEXT_FIELDS.items()
    }
    gold_years = {y for text in gold_text(YEAR_LABEL) for y in _YEAR.findall(text)}
    pred_years = {e.end[:4] for e in schools if e.end and e.end != "present"}
    fields["graduation_year"] = year_counts(gold_years, pred_years)

    quotes = [q for skill in profile.skills for q in skill.quotes]
    skill_spans = [(s.start, s.end) for s in gold[SKILLS_LABEL]]
    return ResumeMetrics(
        fields=fields,
        skills=skill_quote_counts(resume.text, quotes, skill_spans),
        contact_leaks=find_leaks(profile.model_dump(), resume.contact_strings),
    )


def match_key(text: str) -> str:
    """``normalize()``, minus spelling-only punctuation, separators as spaces.

    The dataset's "&amp;" entity reads as "&", whichever side it appears on.
    """
    key = normalize(text).replace(AMP_ENTITY, "&").translate(_SPELLING_ONLY)
    key = _SEPARATORS.sub(" ", key)
    return " ".join(key.split())


def _distinct(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(key for v in values if (key := match_key(v))))


def _occurrences(text: str, quote: str) -> Iterator[int]:
    at = text.find(quote)
    while at != -1:
        yield at
        at = text.find(quote, at + 1)


def _strings(node: object) -> Iterator[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None
