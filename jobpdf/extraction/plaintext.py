"""Build a ParsedDocument from plain text, keeping the text unchanged (JM-14).

Labeled datasets such as DataTurks give character offsets into a plain string,
so ``full_text`` must equal that string exactly. ``parser._assemble`` can't do
that: it always joins blocks with "\\n\\n" and rejects empty blocks, while plain
text has single newlines and runs of three or more. So the blocks are built
here directly, each pointing into the original string; no offset mapping.
"""

from __future__ import annotations

from jobpdf.extraction.models import ParsedDocument, TextBlock

# Every line gets the same size, so JM-10 sees no font signal (only keywords).
DEFAULT_FONT_SIZE = 10.0


def from_plain_text(text: str) -> ParsedDocument:
    """One TextBlock per non-blank line; ``full_text`` is ``text`` itself.

    Blank lines produce no block, the same as empty paragraphs in a DOCX.
    ``source_type`` is "docx" because models.py has no plain-text type, and
    DOCX blocks likewise carry no page or bbox.
    """
    blocks: list[TextBlock] = []
    cursor = 0
    for line in text.split("\n"):
        start, end = cursor, cursor + len(line)
        cursor = end + 1  # past the "\n"
        if not line.strip():
            continue
        blocks.append(
            TextBlock(
                text=line,
                page=None,
                bbox=None,
                font_size=DEFAULT_FONT_SIZE,
                is_bold=False,
                start=start,
                end=end,
            )
        )
    return ParsedDocument(
        source_type="docx",
        blocks=blocks,
        full_text=text,
        has_text_layer=bool(text.strip()),
    )
