"""Print the health report once and exit non-zero when the service is down (JM-34).

Usable by hand and as a container health check. Reads DATABASE_URL and ANTHROPIC_API_KEY
from the environment (as the app does). Statuses, codes and counts only: no secrets.

Usage (from the repo root):
    uv run python scripts/health_check.py            # database, vacancies, llm key, fetch
    uv run python scripts/health_check.py --full     # also loads the normalization service
                                                     # (needs the skill index and the model)
Exit code: 0 ok or degraded, 1 down.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

from jobpdf.monitoring.health import DOWN, HealthReport, build_health


def main(argv: list[str], builder: Callable[..., HealthReport] = build_health) -> int:
    unknown = [a for a in argv if a != "--full"]
    if unknown:
        print("usage: health_check.py [--full]")
        return 2
    kwargs = {}
    if "--full" in argv:
        from jobpdf.normalization.service import get_normalization_service

        kwargs["normalizer"] = get_normalization_service()
    report = builder(**kwargs)
    print(report.model_dump_json(indent=2))
    return 1 if report.status == DOWN else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
