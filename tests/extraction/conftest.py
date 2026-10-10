import pytest

from jobpdf.extraction import llm


@pytest.fixture(autouse=True)
def no_live_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test may reach the Anthropic API or depend on a developer's key (JM-11).

    Live checks belong to JM-35's opt-in ``live`` marker.
    """

    def refuse(self, **kwargs) -> None:
        raise AssertionError("AnthropicToolCaller.call() must never run in tests")

    monkeypatch.setattr(llm.AnthropicToolCaller, "call", refuse)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
