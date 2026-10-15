import pytest

from jobpdf.extraction import llm


@pytest.fixture(autouse=True)
def no_live_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """No pipeline test may reach the Anthropic API or depend on a developer's key (JM-28)."""

    def refuse(self, **kwargs) -> None:
        raise AssertionError("AnthropicToolCaller.call() must never run in tests")

    monkeypatch.setattr(llm.AnthropicToolCaller, "call", refuse)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def temp_dir(tmp_path, monkeypatch):
    """Uploads go to a throwaway folder, never into the repository's jobpdf/temp/."""
    from jobpdf import pipeline

    folder = tmp_path / "temp"
    monkeypatch.setattr(pipeline, "TEMP_DIR", folder)
    return folder
