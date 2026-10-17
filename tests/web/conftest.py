import pytest

from jobpdf import pipeline
from jobpdf.extraction import llm


@pytest.fixture(autouse=True)
def isolated_pipeline(tmp_path, monkeypatch) -> None:
    """No API test may reach the Anthropic API, depend on a key, or write into jobpdf/temp/."""

    def refuse(self, **kwargs) -> None:
        raise AssertionError("AnthropicToolCaller.call() must never run in tests")

    monkeypatch.setattr(llm.AnthropicToolCaller, "call", refuse)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(pipeline, "TEMP_DIR", tmp_path / "uploads")
