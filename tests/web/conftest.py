import pytest

from jobpdf import pipeline


@pytest.fixture(autouse=True)
def isolated_pipeline(tmp_path, monkeypatch) -> None:
    """No web test may write into jobpdf/temp/. (tests/conftest.py keeps them off the LLM.)"""
    monkeypatch.setattr(pipeline, "TEMP_DIR", tmp_path / "uploads")
