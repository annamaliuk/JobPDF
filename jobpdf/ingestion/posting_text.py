"""A posting's description as clean lines, for requirement extraction (JM-48).

Sources send HTML (Remotive, Arbeitnow, The Muse) or plain text (Adzuna and Jooble
excerpts). Both become a list of ``Line``: one per paragraph, list item or heading, with
whether it looked like a heading or a bullet. HTML structure is the best signal for
sections, so it is kept as flags instead of being flattened away.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser

from jobpdf.extraction.cleaning import clean_text

MAX_INPUT_CHARS = 50_000  # raw description; longer ones are cut before parsing
MIN_USABLE_CHARS = 200  # less cleaned text than this is not a usable description
MAX_HEADING_CHARS = 80
MAX_HEADING_WORDS = 8

_BLOCK_TAGS = {
    "p", "div", "br", "li", "ul", "ol", "tr", "table", "section", "article", "header",
    "footer", "blockquote", "pre", "h1", "h2", "h3", "h4", "h5", "h6", "dt", "dd", "hr",
}
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_BOLD_TAGS = {"strong", "b"}
_SKIP_TAGS = {"script", "style", "noscript", "template"}
_HTML_HINT = re.compile(r"<\s*/?\s*[a-zA-Z][a-zA-Z0-9]*[\s/>]")
_BULLET = re.compile(r"^\s*(?:[-*•–·▪●‣⁃]|\d{1,2}[.)])\s+")


@dataclass(frozen=True)
class Line:
    text: str
    is_heading: bool = False
    is_bullet: bool = False
    # Shaped like a title ("Your Future at Kyndryl") but not marked as a heading: it may
    # open a section, or just be a short item such as "Ruby on Rails".
    is_title: bool = False


class _Collector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[Line] = []
        self._parts: list[str] = []
        self._plain = False  # this line has text outside bold tags
        self._heading = 0
        self._bold = 0
        self._skip = 0
        self._bullet = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag in _BLOCK_TAGS:
            self._flush()
            if tag == "li":
                self._bullet = True
            if tag in _HEADING_TAGS:
                self._heading += 1
        elif tag in _BOLD_TAGS:
            self._bold += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK_TAGS:
            self._flush()
            if tag in _HEADING_TAGS:
                self._heading = max(0, self._heading - 1)
        elif tag in _BOLD_TAGS:
            self._bold = max(0, self._bold - 1)

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if data.strip() and not (self._bold or self._heading):
            self._plain = True
        self._parts.append(data)

    def _flush(self) -> None:
        text = " ".join("".join(self._parts).split())
        if text:
            tagged = bool(self._heading) or not self._plain  # all of it bold, or an <hN>
            self.lines.append(Line(text, is_heading=tagged and _short(text),
                                   is_bullet=self._bullet))
        self._parts, self._plain, self._bullet = [], False, False

    def close(self) -> None:
        super().close()
        self._flush()


def _short(text: str) -> bool:
    return len(text) <= MAX_HEADING_CHARS and len(text.split()) <= MAX_HEADING_WORDS


_SMALL_WORDS = 3  # "at", "of", "the" may stay lowercase in a title


def _looks_like_heading(text: str) -> bool:
    return text.endswith(":") and _short(text)


def _looks_like_title(text: str) -> bool:
    """3 to 6 words, capitalised except small ones, no sentence punctuation."""
    words = text.split()
    return (3 <= len(words) <= 6 and not text.endswith((".", "!", "?", ",", ";", ")"))
            and all(w[:1].isupper() for w in words if len(w) > _SMALL_WORDS))


def posting_lines(description: str | None) -> list[Line]:
    """The description as cleaned lines; [] for nothing. HTML is detected, not assumed."""
    if not description or not description.strip():
        return []
    raw = description[:MAX_INPUT_CHARS]
    if _HTML_HINT.search(raw):
        collector = _Collector()
        collector.feed(raw)
        collector.close()
        found = collector.lines
    else:
        found = [Line(line) for line in raw.splitlines()]
    lines: list[Line] = []
    for line in found:
        text = clean_text(line.text).replace("\n", " ")
        bullet = line.is_bullet
        if match := _BULLET.match(text):
            text, bullet = text[match.end():], True
        text = text.strip()
        if not text:
            continue
        heading = (line.is_heading or _looks_like_heading(text)) and not bullet
        title = not (heading or bullet) and _looks_like_title(text)
        lines.append(Line(text, is_heading=heading, is_bullet=bullet, is_title=title))
    return lines


def is_usable(lines: list[Line]) -> bool:
    """Enough text to read requirements from (else the source's tags are used)."""
    return sum(len(line.text) for line in lines) >= MIN_USABLE_CHARS
