"""Show the exact extraction prompt for a CV, without calling any API (JM-11).

Runs parse -> split_sections -> prompt building and prints the system prompt,
the user message and the tool schema, e.g. to review the prompt or to try it
by hand in claude.ai.

Usage (from the repo root):
    uv run python scripts/show_extraction_prompt.py tests/extraction/fixtures/single_column.pdf
    uv run python scripts/show_extraction_prompt.py CV.pdf --json --out prompt.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jobpdf.extraction import parse
from jobpdf.extraction.models import ParseError
from jobpdf.extraction.prompt import PROMPT_VERSION, SYSTEM_PROMPT, build_user_message, mask_pii
from jobpdf.extraction.schema import tool_schema
from jobpdf.extraction.sections import split_sections

CHARS_PER_TOKEN = 4  # rough estimate; no tokenizer offline


def main() -> int:
    parser = argparse.ArgumentParser(description="Print the extraction prompt for a CV.")
    parser.add_argument("cv", type=Path, help="CV file (.pdf or .docx)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--out", type=Path, help="write the output to this file")
    args = parser.parse_args()

    try:
        doc = parse(args.cv)
    except ParseError as exc:
        print(f"ERROR: cannot parse {args.cv}: {exc.reason} ({exc})", file=sys.stderr)
        return 1
    sections, section_warnings = split_sections(doc)
    user = build_user_message(doc, sections)
    tool = tool_schema()
    masked_spans = mask_pii(doc.full_text)[1]
    tool_json = json.dumps(tool, ensure_ascii=False)
    estimate = (len(SYSTEM_PROMPT) + len(user) + len(tool_json)) // CHARS_PER_TOKEN
    info = {
        "prompt_version": PROMPT_VERSION,
        "masked_spans": masked_spans,
        "sections_sent": user.count("<section "),
        "warnings": doc.warnings + section_warnings,
        "estimated_input_tokens": estimate,
    }

    if args.json:
        output = json.dumps(
            {"system": SYSTEM_PROMPT, "user": user, "tool": tool, "info": info},
            indent=2, ensure_ascii=False,
        )
    else:
        output = "\n".join([
            "=" * 30 + " SYSTEM " + "=" * 30, SYSTEM_PROMPT,
            "=" * 30 + " USER " + "=" * 32, user,
            "=" * 30 + " TOOL " + "=" * 32, json.dumps(tool, indent=2, ensure_ascii=False),
            "=" * 30 + " INFO " + "=" * 32,
            *(f"{key}: {value}" for key, value in info.items()),
            f"(token estimate = characters / {CHARS_PER_TOKEN}; no tokenizer offline)",
        ])

    if args.out:
        args.out.write_text(output + "\n", encoding="utf-8")
        print(f"written to {args.out}")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
