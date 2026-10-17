import pytest

# tests/conftest.py keeps every pipeline test off the LLM and away from developer keys (JM-28).


@pytest.fixture(autouse=True)
def temp_dir(tmp_path, monkeypatch):
    """Uploads go to a throwaway folder, never into the repository's jobpdf/temp/."""
    from jobpdf import pipeline

    folder = tmp_path / "temp"
    monkeypatch.setattr(pipeline, "TEMP_DIR", folder)
    return folder
