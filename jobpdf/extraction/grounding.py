"""Grounding: locate every extracted quote in the CV text (JM-12).

The LLM's "verbatim" quotes differ cosmetically from the parsed text (collapsed
line breaks, straightened quotes, joined hyphenation), so plain ``find`` misses
many of them. Matching therefore runs on a normalized copy of the text that keeps
an index map back to the original, so every hit still yields offsets into
``ParsedDocument.full_text``.

This deliberately does not use ``normalization.text.normalize()``: that one is
for skill lookups and cache keys and cannot map positions back.
"""

from __future__ import annotations

import unicodedata

# Quote and dash variants an LLM tends to straighten. Applied after NFKC, which
# already folds ligatures, full-width forms and the non-breaking space.
_CHAR_MAP = {
    **dict.fromkeys("‘’‚‛", "'"),
    **dict.fromkeys("“”„‟«»", '"'),
    **dict.fromkeys("‐‑‒–—―−", "-"),
}
# Invisible characters that never survive into a quote: soft hyphen, zero-width
# space / non-joiner / joiner, word joiner and BOM.
_INVISIBLE = frozenset(chr(c) for c in (0xAD, 0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF))
_LINE_BREAKS = frozenset("\n\r\v\f\x85  ")


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Normalize ``text`` for matching; ``index_map[i]`` is where char ``i`` came from.

    Steps: NFKC, unified quotes and dashes, invisible characters dropped,
    line-break hyphenation resolved the way ``cleaning.clean_text`` does it, and
    every whitespace run collapsed to one space. The map is non-decreasing, so a
    normalized span ``[s, e)`` maps back to ``[map[s], map[e - 1] + 1)``.
    """
    chars, origin = _fold_characters(text)
    return _fold_whitespace(chars, origin)


def _fold_characters(text: str) -> tuple[list[str], list[int]]:
    # NFKC per base character plus its combining marks, so a decomposed "й"
    # (и + U+0306) composes like the precomposed one without merging clusters.
    chars: list[str] = []
    origin: list[int] = []
    i, n = 0, len(text)
    while i < n:
        j = i + 1
        while j < n and unicodedata.combining(text[j]):
            j += 1
        for ch in unicodedata.normalize("NFKC", text[i:j]):
            ch = _CHAR_MAP.get(ch, ch)
            if ch not in _INVISIBLE:
                chars.append(ch)
                origin.append(i)
        i = j
    return chars, origin


def _fold_whitespace(chars: list[str], origin: list[int]) -> tuple[str, list[int]]:
    out: list[str] = []
    out_map: list[int] = []
    k, n = 0, len(chars)
    while k < n:
        ch = chars[k]
        if ch.isspace():
            run_end = _skip_space(chars, k)
            out.append(" ")
            out_map.append(origin[k])
            k = run_end
            continue
        if ch == "-" and k > 0 and chars[k - 1].isalnum():
            run_end = _skip_space(chars, k + 1)
            breaks_line = any(c in _LINE_BREAKS for c in chars[k + 1 : run_end])
            if breaks_line and run_end < n:
                # "develop-\nment" was one word; "Front-\nEnd" and "2019-\n2021"
                # keep their hyphen (same rule as clean_text).
                if not (chars[k - 1].isalpha() and chars[run_end].islower()):
                    out.append("-")
                    out_map.append(origin[k])
                k = run_end
                continue
        out.append(ch)
        out_map.append(origin[k])
        k += 1
    return "".join(out), out_map


def _skip_space(chars: list[str], k: int) -> int:
    while k < len(chars) and chars[k].isspace():
        k += 1
    return k
