"""Text normalisation shared by all format parsers."""

from __future__ import annotations

import re
import unicodedata

# Written as chr() so the invisible character is explicit in source.
SOFT_HYPHEN = chr(0xAD)

_HYPHEN_BREAK = re.compile(rf"([^\W\d_])[-{SOFT_HYPHEN}]\n([^\W\d_])")
_INLINE_SPACE = re.compile(r"[^\S\n]+")


def clean_text(text: str) -> str:
    """Normalise extracted text while keeping one line per visual line.

    Line breaks are kept (single ``\\n``) because JM-10 uses bullet lines; blank
    lines are dropped so a block can never contain the block separator.
    """
    text = unicodedata.normalize("NFKC", text)  # also expands ligatures like "ﬁ"
    text = _HYPHEN_BREAK.sub(_join_hyphenated, text)
    text = text.replace(SOFT_HYPHEN, "")
    lines = (_INLINE_SPACE.sub(" ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def _join_hyphenated(match: re.Match[str]) -> str:
    # Only join when the next word continues in lowercase ("develop-\nment");
    # "Front-\nEnd" or "Senior-\nLevel" keep their hyphen.
    before, after = match.group(1), match.group(2)
    if after.islower():
        return before + after
    return match.group(0)
