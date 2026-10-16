import pytest

from jobpdf.extraction import llm


@pytest.fixture(autouse=True)
def no_live_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test may reach the Gemini API, and every test sees the default provider.

    The per-package conftests already do the same for Anthropic.
    """

    def refuse(self, **kwargs) -> None:
        raise AssertionError("GeminiToolCaller.call() must never run in tests")

    monkeypatch.setattr(llm.GeminiToolCaller, "call", refuse)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
