"""LLM-as-judge check of extraction on CVs without labels (JM-15).

JM-14 scores extraction against human labels, but most real uploads have none.
Here a second LLM call reads the same masked CV the extractor saw, gives a
verdict on every extracted item, and lists what the extraction left out, with
a verbatim quote for each omission. Code then checks the judge: an omission
quote that grounding can't find in the CV, or a verdict for an item that does
not exist, is dropped and counted, so the judge can't invent problems.

Like extraction, every call goes through ``ToolCaller``; tests use
``FakeToolCaller`` and never touch the network.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Literal

from pydantic import Field, ValidationError

from jobpdf.extraction.grounding import GroundingStatus, ground
from jobpdf.extraction.llm import ToolCaller, ToolCallResult, default_caller
from jobpdf.extraction.models import ParsedDocument, Section
from jobpdf.extraction.prompt import MASK_CHAR, build_user_message, mask_pii
from jobpdf.extraction.schema import CandidateProfile, Quote, _LLMModel, tool_schema

# Bump on every change to JUDGE_SYSTEM_PROMPT, the message format or the tool:
# replay recordings are keyed on the prompt text.
JUDGE_PROMPT_VERSION = "1"
JUDGE_TOOL_NAME = "record_judgement"
MAX_TOKENS = 8192
MAX_TOKENS_RETRY = 16384
MAX_ERROR_LINES = 20
REASON_MAX_CHARS = 200

# The profile lists the judge gives verdicts on, in report order. The summary
# is free text in the CV's own words and is not judged.
JUDGED_FIELDS = ("skills", "experience", "education", "certifications", "languages")
JudgedField = Literal["skills", "experience", "education", "certifications", "languages"]
Verdict = Literal["supported", "partly_wrong", "unsupported"]

# Share of Cyrillic among all letters: at or above UK_MIN the CV counts as
# Ukrainian, at or below EN_MAX as English, anything between as mixed.
UK_MIN = 0.7
EN_MAX = 0.1
Language = Literal["uk", "en", "mixed", "unknown"]

JUDGE_SYSTEM_PROMPT = """\
You check a structured extraction made from one CV (résumé) and record your
findings by calling the record_judgement tool exactly once.

The user message holds the CV inside <cv>, split into <section> elements, and
the extraction inside <extraction> as JSON. Every extracted item has a "path"
such as "skills[3]" or "experience[0]". The CV may be in English, Ukrainian or
both; judge it in its own language and never translate.

1. Give exactly one verdict for every path in the extraction:
   - "supported": the CV states this item and every field matches the CV.
   - "partly_wrong": the CV states this item, but at least one field is wrong,
     for example a wrong date, company, title, degree, institution, issuer,
     language level, or a skill attributed to the wrong job. List the wrong
     field names in wrong_fields (for example ["end", "company"]).
   - "unsupported": the CV does not state this item at all, or the item is
     something else (for example a spoken language listed as a skill, or a job
     invented from a project).
   reason is one short sentence in English saying why, at most 200 characters.
   For "supported", reason may be empty.
2. These are NOT errors:
   - a description or summary reworded or shortened in the CV's own language;
   - a skill name with a version or minor spelling change ("Python 3" vs "Python");
   - a date given as a year when the CV only gives the year, or null when the CV
     gives no date;
   - null for a field the CV does not state;
   - a debatable choice between "hard" and "soft", or of found_in.
3. In missed, list items the CV clearly states that the extraction left out:
   skills (tools, technologies, programming languages, methods, domain
   knowledge, and clearly named soft skills), jobs, education entries,
   certifications and spoken languages. Each needs a quote copied character for
   character from the CV, at most 200 characters, specific enough to find the
   item. Do not list an item that is already extracted under another spelling,
   and do not list every generic word as a skill.
4. "█" characters are redacted contact data. They are never missed items and
   never errors; never reproduce them.
5. Set text_unreadable to true only if most of the CV text is garbled (broken
   character encoding, random symbols) so that it cannot be checked. Still give
   verdicts for what you can read.
6. Never output personal names, email addresses, phone numbers, street
   addresses or URLs.
"""


class ItemVerdict(_LLMModel):
    path: str = Field(description='The item\'s path from the extraction, e.g. "skills[3]".')
    verdict: Verdict
    wrong_fields: list[str] = Field(
        description='Names of the wrong fields for "partly_wrong", e.g. ["end"]; else [].'
    )
    reason: str = Field(max_length=REASON_MAX_CHARS, description="One short sentence.")


class MissedItem(_LLMModel):
    field: JudgedField = Field(description="Which list the left-out item belongs in.")
    quote: Quote


class Judgement(_LLMModel):
    """What the judge records via the tool."""

    items: list[ItemVerdict]
    missed: list[MissedItem]
    text_unreadable: bool = Field(
        description="True only if most of the CV text is garbled and cannot be checked."
    )


class JudgeError(RuntimeError):
    """The judge's output failed validation even after its single retry."""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        self.errors = errors or []
        detail = "\n  ".join(self.errors)
        super().__init__(f"{message}\n  {detail}" if detail else message)


@dataclass
class JudgeResult:
    """A checked judgement of one CV.

    ``verdicts`` hold only known paths, first verdict per path. ``missed`` holds
    only omissions whose quote grounding located in the CV.
    """

    verdicts: list[ItemVerdict]
    missed: list[MissedItem]
    text_unreadable: bool
    item_paths: list[str]  # every judged path in the extraction, in order
    unjudged_paths: list[str] = field(default_factory=list)
    dropped_unknown_paths: int = 0
    dropped_duplicate_verdicts: int = 0
    dropped_ungrounded_missed: int = 0
    dropped_masked_missed: int = 0
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


def judge_tool() -> dict:
    return {
        **tool_schema(Judgement),
        "name": JUDGE_TOOL_NAME,
        "description": (
            "Record a verdict for every extracted item and the items the extraction "
            "left out. Call it exactly once. Every quote must be copied verbatim from the CV."
        ),
    }


def judged_items(profile: CandidateProfile) -> Iterator[tuple[str, dict]]:
    """(path, item) for every item the judge gives a verdict on, in report order."""
    data = profile.model_dump()
    for name in JUDGED_FIELDS:
        for i, item in enumerate(data[name]):
            yield f"{name}[{i}]", item


def build_judge_message(doc: ParsedDocument, sections: list[Section],
                        profile: CandidateProfile) -> str:
    """The CV exactly as the extractor saw it, then the numbered extraction."""
    extraction = {
        name: [{"path": path, **item} for path, item in judged_items(profile)
               if path.startswith(f"{name}[")]
        for name in JUDGED_FIELDS
    }
    return (
        "<cv>\n"
        + build_user_message(doc, sections)
        + "\n</cv>\n\n<extraction>\n"
        + json.dumps(extraction, ensure_ascii=False, indent=1)
        + "\n</extraction>"
    )


def judge_profile(
    doc: ParsedDocument,
    sections: list[Section],
    profile: CandidateProfile,
    caller: ToolCaller | None = None,
) -> JudgeResult:
    """Ask the judge about one extraction and check what it says.

    ``caller=None`` means the real API of LLM_PROVIDER (``JUDGE_MODEL`` is not
    read here; the script passes a caller built with it).
    """
    caller = caller if caller is not None else default_caller()
    user = build_judge_message(doc, sections, profile)
    tool = judge_tool()
    usage = [0, 0]

    response = _call(caller, user, tool, usage)
    try:
        judgement = Judgement.model_validate(response.tool_input)
    except ValidationError as first_error:
        corrective = user + _correction(response.tool_input, _concise(first_error))
        response = _call(caller, corrective, tool, usage)
        try:
            judgement = Judgement.model_validate(response.tool_input)
        except ValidationError as second_error:
            raise JudgeError(
                "Judge output failed schema validation twice", _concise(second_error)
            ) from second_error

    result = check_judgement(judgement, doc, sections, profile)
    result.model = response.model
    result.input_tokens, result.output_tokens = usage
    return result


def check_judgement(
    judgement: Judgement,
    doc: ParsedDocument,
    sections: list[Section],
    profile: CandidateProfile,
) -> JudgeResult:
    """Keep only verdicts on real paths and omissions grounded in the CV."""
    item_paths = [path for path, _ in judged_items(profile)]
    known = set(item_paths)
    verdicts: dict[str, ItemVerdict] = {}
    unknown = duplicates = 0
    for verdict in judgement.items:
        path = verdict.path.strip()
        if path not in known:
            unknown += 1
        elif path in verdicts:
            duplicates += 1
        else:
            verdicts[path] = verdict.model_copy(update={"path": path})

    masked_text, _ = mask_pii(doc.full_text)
    candidates = [m for m in judgement.missed if MASK_CHAR not in m.quote]
    grounding = ground(
        {"missed": [m.model_dump() for m in candidates]}, doc, sections, masked_text
    )
    found = {
        g.path for g in grounding.groundings if g.status is not GroundingStatus.UNGROUNDED
    }
    missed = [m for i, m in enumerate(candidates) if f"missed[{i}]" in found]

    return JudgeResult(
        verdicts=[verdicts[p] for p in item_paths if p in verdicts],
        missed=missed,
        text_unreadable=judgement.text_unreadable,
        item_paths=item_paths,
        unjudged_paths=[p for p in item_paths if p not in verdicts],
        dropped_unknown_paths=unknown,
        dropped_duplicate_verdicts=duplicates,
        dropped_ungrounded_missed=len(candidates) - len(missed),
        dropped_masked_missed=len(judgement.missed) - len(candidates),
    )


def detect_language(text: str) -> Language:
    """"uk", "en" or "mixed" by the share of Cyrillic letters; "unknown" without letters."""
    cyrillic = latin = 0
    for ch in text:
        if not ch.isalpha():
            continue
        if "Ѐ" <= ch <= "ӿ":
            cyrillic += 1
        elif ch.isascii():
            latin += 1
    if not cyrillic + latin:
        return "unknown"
    share = cyrillic / (cyrillic + latin)
    if share >= UK_MIN:
        return "uk"
    if share <= EN_MAX:
        return "en"
    return "mixed"


def item_field(path: str) -> str:
    return path.split("[", 1)[0]


def weak_grounding_items(
    profile: CandidateProfile, doc: ParsedDocument, sections: list[Section]
) -> set[str]:
    """Item paths with at least one extractor quote grounded only fuzzily or not at all."""
    masked_text, _ = mask_pii(doc.full_text)
    result = ground(profile, doc, sections, masked_text)
    weak = {GroundingStatus.FUZZY, GroundingStatus.UNGROUNDED}
    return {
        g.path.split(".", 1)[0] for g in result.groundings
        if g.status in weak and item_field(g.path) in JUDGED_FIELDS
    }


@dataclass
class CVJudgement:
    """One judged CV as the aggregate sees it: verdicts, omissions, context."""

    cv_id: str
    language: Language
    result: JudgeResult
    weak_grounding: set[str] = field(default_factory=set)


def aggregate(cvs: Iterable[CVJudgement]) -> dict:
    """Counts and rates per field, overall and per language. No CV text or reasons.

    CVs the judge marked unreadable are counted but left out of every rate,
    since verdicts on garbled text say nothing about extraction.
    """
    cvs = list(cvs)
    readable = [cv for cv in cvs if not cv.result.text_unreadable]
    languages = sorted({cv.language for cv in readable})
    return {
        "cvs": len(cvs),
        "cvs_unreadable": len(cvs) - len(readable),
        "cvs_by_language": dict(sorted(Counter(cv.language for cv in readable).items())),
        "overall": _rates(readable),
        "by_language": {
            lang: _rates([cv for cv in readable if cv.language == lang]) for lang in languages
        },
        "grounding_agreement": _grounding_agreement(readable),
        "judge_checks": {
            "unjudged_items": sum(len(cv.result.unjudged_paths) for cv in cvs),
            "dropped_unknown_paths": sum(cv.result.dropped_unknown_paths for cv in cvs),
            "dropped_duplicate_verdicts": sum(cv.result.dropped_duplicate_verdicts for cv in cvs),
            "dropped_ungrounded_missed": sum(cv.result.dropped_ungrounded_missed for cv in cvs),
            "dropped_masked_missed": sum(cv.result.dropped_masked_missed for cv in cvs),
        },
    }


def _rates(cvs: list[CVJudgement]) -> dict:
    fields = {name: _field_rates(cvs, name) for name in JUDGED_FIELDS}
    totals = Counter()
    for counts in fields.values():
        totals.update({k: counts[k] for k in ("judged", "supported", "partly_wrong",
                                              "unsupported", "missed")})
    return {"all": _with_rates(totals), "fields": fields}


def _field_rates(cvs: list[CVJudgement], name: str) -> dict:
    counts: Counter = Counter()
    wrong_fields: Counter = Counter()
    for cv in cvs:
        for verdict in cv.result.verdicts:
            if item_field(verdict.path) != name:
                continue
            counts["judged"] += 1
            counts[verdict.verdict] += 1
            if verdict.verdict == "partly_wrong":
                wrong_fields.update(dict.fromkeys(verdict.wrong_fields, 1))
        counts["missed"] += sum(1 for m in cv.result.missed if m.field == name)
    report = _with_rates(counts)
    report["top_wrong_fields"] = dict(wrong_fields.most_common(5))
    return report


def _with_rates(counts: Counter) -> dict:
    judged, missed = counts["judged"], counts["missed"]
    errors = counts["partly_wrong"] + counts["unsupported"]
    return {
        "judged": judged,
        "supported": counts["supported"],
        "partly_wrong": counts["partly_wrong"],
        "unsupported": counts["unsupported"],
        "missed": missed,
        "supported_rate": _ratio(counts["supported"], judged),  # precision proxy
        "error_rate": _ratio(errors, judged),
        "miss_rate": _ratio(missed, judged + missed),  # 1 - recall proxy
    }


def _grounding_agreement(cvs: list[CVJudgement]) -> dict:
    """How the judge's "unsupported" lines up with JM-12's weak groundings."""
    unsupported = weak = both = 0
    for cv in cvs:
        flagged = {v.path for v in cv.result.verdicts if v.verdict == "unsupported"}
        unsupported += len(flagged)
        weak += len(cv.weak_grounding)
        both += len(flagged & cv.weak_grounding)
    return {
        "unsupported_items": unsupported,
        "weakly_grounded_items": weak,
        "both": both,
        "unsupported_also_weakly_grounded": _ratio(both, unsupported),
        "weakly_grounded_also_unsupported": _ratio(both, weak),
    }


def _call(caller: ToolCaller, user: str, tool: dict, usage: list[int]) -> ToolCallResult:
    """One logical call, retried once with a bigger budget if it was cut off."""
    for max_tokens in (MAX_TOKENS, MAX_TOKENS_RETRY):
        response = caller.call(
            system=JUDGE_SYSTEM_PROMPT, user=user, tool=tool, max_tokens=max_tokens
        )
        usage[0] += response.input_tokens
        usage[1] += response.output_tokens
        if response.stop_reason != "max_tokens":
            return response
    raise JudgeError(f"Judge output exceeded {MAX_TOKENS_RETRY} tokens even after retrying.")


def _concise(error: ValidationError) -> list[str]:
    lines = []
    for item in error.errors(include_url=False)[:MAX_ERROR_LINES]:
        location = ".".join(str(part) for part in item["loc"]) or "(root)"
        lines.append(f"{location}: {item['msg']}")
    return lines


def _correction(previous: dict, errors: list[str]) -> str:
    return (
        "\n\n<previous_output>\n"
        + json.dumps(previous, ensure_ascii=False, indent=1)
        + "\n</previous_output>\n<validation_errors>\n"
        + "\n".join(f"- {line}" for line in errors)
        + "\n</validation_errors>\n"
        f"Your previous {JUDGE_TOOL_NAME} call did not match the schema. "
        "Call the tool again with the complete, corrected judgement."
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None
