"""Job sources besides Remotive: The Muse, Arbeitnow, Adzuna and Jooble.

Every source is a small adapter (``SourceAdapter``) that builds its requests and turns each
response page into the common ``SourcePosting``; ``fetch_source`` is the one place that
caches, rations requests and falls back to stale data. Remotive keeps its own module
(``jobpdf.ingestion.remotive``); ``to_source_posting`` converts its postings to the same
shape, so ingestion downstream handles every source alike.
"""

from jobpdf.ingestion.sources.models import (
    FeedError,
    FetchError,
    ParsedPage,
    Policy,
    Request,
    SourceAdapter,
    SourcePosting,
    SourceResult,
)

__all__ = [
    "FeedError", "FetchError", "ParsedPage", "Policy", "Request", "SourceAdapter",
    "SourcePosting", "SourceResult",
]
