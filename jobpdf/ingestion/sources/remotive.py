"""Remotive postings in the common shape (the fetching stays in jobpdf.ingestion.remotive)."""

from __future__ import annotations

from jobpdf.ingestion.remotive import SOURCE, RemotivePosting
from jobpdf.ingestion.sources.models import SourcePosting


def to_source_posting(posting: RemotivePosting) -> SourcePosting:
    return SourcePosting(
        source=SOURCE, source_id=str(posting.id), url=posting.url, title=posting.title,
        company=posting.company_name, location_raw=posting.candidate_required_location,
        remote=True,  # Remotive lists remote jobs only
        posted_at=posting.publication_date, description=posting.description,
        tags=list(posting.tags), category=posting.category,
        employment_type=posting.job_type, salary_raw=posting.salary or None,
    )
