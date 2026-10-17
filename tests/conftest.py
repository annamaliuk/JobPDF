import pytest

from jobpdf.extraction import llm


@pytest.fixture(autouse=True)
def no_live_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test may reach the Gemini API, and every test sees the default provider.

    The per-package conftests already do the same for Anthropic.
    """

    def refuse(self, **kwargs) -> None:
        raise AssertionError("GeminiToolCaller.call() must never run in tests")

    def refuse_ollama(self, path, body) -> None:
        raise AssertionError("OllamaToolCaller must never reach a server in tests")

    monkeypatch.setattr(llm.GeminiToolCaller, "call", refuse)
    monkeypatch.setattr(llm.OllamaToolCaller, "_post", refuse_ollama)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GEMINI_THINKING_BUDGET", raising=False)
    monkeypatch.delenv("GEMINI_MIN_INTERVAL_S", raising=False)
    for name in ("OLLAMA_MODEL", "OLLAMA_HOST", "OLLAMA_NUM_CTX", "OLLAMA_MAX_PREDICT"):
        monkeypatch.delenv(name, raising=False)
