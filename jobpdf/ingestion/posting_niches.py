"""Which risky industries a posting is about (JM-51), decided once at ingestion.

The industry words are in ``jobpdf/niches.py``. This reads a posting the way
``work_place.py`` does: the head is the title, company, tags and the source's category, the text
is the description as plain text up to ``MAX_TEXT_CHARS``. Only the posting's own text is read,
nothing is logged, and the answer for one posting never depends on another.
"""

from __future__ import annotations

from jobpdf.ingestion.sources.models import SourcePosting
from jobpdf.ingestion.work_place import posting_text
from jobpdf.niches import detect_niches


def posting_niches(posting: SourcePosting) -> list[str]:
    """The industry codes of ``posting`` in the table's order; [] when none or not known."""
    head = " ".join([posting.title, posting.company or "", " ".join(posting.tags),
                     posting.category or ""])
    return detect_niches(head, posting_text(posting))
