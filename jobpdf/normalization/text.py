"""The one text normalisation used for every skill lookup key."""

from __future__ import annotations

import unicodedata


def normalize(s: str) -> str:
    """Canonical lookup form of a skill label: NFKC, trimmed, lowercased, single-spaced.

    JM-16 ``alias_norm``, JM-18 lookups and JM-19 cache keys MUST all use this
    function. If two of them normalised differently, an alias written by the
    taxonomy build would silently never match at lookup time.

    Punctuation is deliberately kept: "c++", "c#", ".net" and "node.js" are
    distinct skills, and stripping it would merge them with "c" or "net".
    """
    text = unicodedata.normalize("NFKC", s)
    return " ".join(text.lower().split())
