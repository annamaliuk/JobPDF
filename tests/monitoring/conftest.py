import pytest

from jobpdf import pipeline
from jobpdf.extraction import llm


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch) -> None:
    """No monitoring test may reach the LLM or the network, depend on a developer's keys or
    cache, or write into jobpdf/temp/ or data/cache/."""

    def refuse(self, **kwargs) -> None:
        raise AssertionError("AnthropicToolCaller.call() must never run in tests")

    monkeypatch.setattr(llm.AnthropicToolCaller, "call", refuse)
    for name in ("ANTHROPIC_API_KEY", "ADZUNA_APP_ID", "ADZUNA_APP_KEY", "JOOBLE_API_KEY",
                 "THE_MUSE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("JOBPDF_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(pipeline, "TEMP_DIR", tmp_path / "uploads")
