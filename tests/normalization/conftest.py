import pytest


def pytest_configure(config: pytest.Config) -> None:
    # Registered here rather than in pyproject.toml, which this ticket must not touch.
    config.addinivalue_line(
        "markers", "taxonomy: needs the real built data/taxonomy/aliases.csv (skipped otherwise)"
    )
