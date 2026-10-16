import socket

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Nothing here may reach Remotive (or anything else): its limit is 4 requests a day, and
    the rules say it is never fetched in tests (JM-31). Caches also go to a tmp folder."""

    def refuse(self, *args, **kwargs):
        raise AssertionError("tests must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setenv("JOBPDF_CACHE_DIR", str(tmp_path / "cache"))
