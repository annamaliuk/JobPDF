"""DOCX parsing with python-docx: paragraphs and tables in document order."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import docx
from docx.blkcntnr import BlockItemContainer
from docx.document import Document
from docx.oxml.ns import qn
from docx.oxml.xmlchemy import BaseOxmlElement
from docx.styles.style import ParagraphStyle
from docx.table import Table
from docx.text.paragraph import Paragraph

from jobpdf.extraction.cleaning import clean_text
from jobpdf.extraction.models import ParseError, RawBlock

TEXT_BOX_WARNING = (
    "Document contains floating text boxes; their text is not extracted."
)
NO_TEXT_WARNING = "No text found in the document."

# Styles whose paragraphs are headers even without direct bold formatting.
_HEADER_STYLE_PREFIXES = ("Heading", "Title")


def parse_docx(path: Path) -> tuple[list[RawBlock], bool, list[str]]:
    """Extract header paragraphs first, then the body (paragraphs and tables)."""
    doc = _open(path)
    containers = _containers(doc)
    default_size = _default_font_size(doc)
    blocks = [
        block
        for element, parent in containers
        for paragraph in _iter_container(element, parent)
        if (block := _to_raw_block(paragraph, default_size)) is not None
    ]

    warnings: list[str] = []
    if any(element.xpath(".//w:txbxContent") for element, _ in containers):
        warnings.append(TEXT_BOX_WARNING)
    has_text = bool(blocks)
    if not has_text:
        warnings.append(NO_TEXT_WARNING)
    return blocks, has_text, warnings


def _open(path: Path) -> Document:
    try:
        return docx.Document(str(path))
    except Exception as exc:  # bad zip, missing parts, invalid XML all surface differently
        raise ParseError("corrupt", f"Cannot open DOCX: {exc}") from exc


def _containers(doc: Document) -> list[tuple[BaseOxmlElement, BlockItemContainer]]:
    """Story parts in reading order: page header(s), then body.

    Contact details often live in the page header, so it comes first.
    """
    section = doc.sections[0]
    headers = []
    if section.different_first_page_header_footer:
        headers.append(section.first_page_header)
    headers.append(section.header)
    containers: list[tuple[BaseOxmlElement, BlockItemContainer]] = [
        (header._element, header)
        for header in headers
        # A linked header has no definition of its own; touching it would create one.
        if not header.is_linked_to_previous
    ]
    containers.append((doc.element.body, doc._body))
    return containers


def _iter_container(
    element: BaseOxmlElement, parent: BlockItemContainer
) -> Iterator[Paragraph]:
    """Walk block-level children in document order, descending into tables.

    ``doc.paragraphs`` skips tables entirely, and many CV templates are built
    from tables, so we walk the XML ourselves.
    """
    for child in element.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, parent)
        elif child.tag == qn("w:tbl"):
            yield from _iter_table(Table(child, parent))
        elif child.tag == qn("w:sdt"):
            # Content controls (common in Word CV templates) wrap ordinary blocks.
            content = child.find(qn("w:sdtContent"))
            if content is not None:
                yield from _iter_container(content, parent)


def _iter_table(table: Table) -> Iterator[Paragraph]:
    """Row by row, cell by cell; merged cells are visited once."""
    # Hold the elements, not id()s: lxml proxies can be collected and their ids reused.
    seen: set[BaseOxmlElement] = set()
    for row in table.rows:
        for cell in row.cells:
            # A horizontally merged cell is returned once per grid column it spans.
            if cell._tc in seen:
                continue
            seen.add(cell._tc)
            yield from _iter_container(cell._tc, cell)


def _to_raw_block(paragraph: Paragraph, default_size: float | None) -> RawBlock | None:
    text = clean_text(paragraph.text)
    if not text:
        return None
    return RawBlock(
        text=text,
        page=None,
        bbox=None,
        font_size=_font_size(paragraph) or default_size,
        is_bold=_is_bold(paragraph),
    )


def _is_bold(paragraph: Paragraph) -> bool:
    style_name = paragraph.style.name if paragraph.style is not None else ""
    if style_name.startswith(_HEADER_STYLE_PREFIXES):
        return True
    return any(run.bold for run in paragraph.runs if run.text.strip())


def _font_size(paragraph: Paragraph) -> float | None:
    """Most common run size by character count, else the paragraph style's size.

    Headings usually get their size from the style, not from the runs, and
    JM-10 needs that size to tell headers from body text.
    """
    sizes: Counter[float] = Counter()
    for run in paragraph.runs:
        n = len(run.text.strip())
        if n and run.font.size is not None:
            sizes[run.font.size.pt] += n
    if sizes:
        return sizes.most_common(1)[0][0]
    return _style_font_size(paragraph.style)


def _default_font_size(doc: Document) -> float | None:
    """Document-wide default size, used when neither runs nor styles set one.

    Word's built-in Normal style usually has no size of its own; it comes from
    w:docDefaults. Without this, body text would have no size to compare
    headings against.
    """
    path = "/".join(qn(tag) for tag in ("w:docDefaults", "w:rPrDefault", "w:rPr", "w:sz"))
    size = doc.styles.element.find(path)
    if size is None or size.get(qn("w:val")) is None:
        return None
    return int(size.get(qn("w:val"))) / 2  # w:sz is in half-points


def _style_font_size(style: ParagraphStyle | None) -> float | None:
    # Styles inherit size from their base style (e.g. Heading 1 -> Normal).
    while style is not None:
        if style.font.size is not None:
            return style.font.size.pt
        style = style.base_style
    return None
