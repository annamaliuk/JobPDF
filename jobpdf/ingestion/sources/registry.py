"""The sources that can be fetched, by name."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from jobpdf.ingestion.sources.adzuna import Adzuna
from jobpdf.ingestion.sources.arbeitnow import Arbeitnow
from jobpdf.ingestion.sources.jooble import Jooble
from jobpdf.ingestion.sources.models import SourceAdapter
from jobpdf.ingestion.sources.themuse import TheMuse

FACTORIES: dict[str, Callable[[Mapping[str, str] | None], SourceAdapter]] = {
    "themuse": TheMuse.from_environment,
    "arbeitnow": lambda environ: Arbeitnow(),
    "adzuna": Adzuna.from_environment,
    "jooble": Jooble.from_environment,
}
SOURCE_NAMES = tuple(FACTORIES)


def build_adapter(name: str, environ: Mapping[str, str] | None = None) -> SourceAdapter:
    try:
        factory = FACTORIES[name]
    except KeyError:
        raise ValueError(f"unknown source: {name}") from None
    return factory(environ)
