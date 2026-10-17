"""Fail when the database tests were skipped instead of run (JM-37).

The ``db`` tests skip themselves, quietly, when Postgres is missing or unreachable. On a
developer machine that is convenient; in CI it would turn a broken database service into a
green build that tested nothing. CI runs ``pytest -m db --junitxml=FILE`` and then this script.

Usage:
    python scripts/check_db_tests_ran.py junit-db.xml --min 50

Exit code 0 when at least ``--min`` of the tests in the report ran (passed or failed rather
than skipped), 1 when fewer did, 2 when the report cannot be read. Prints counts only.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

MIN_DB_TESTS = 50  # a floor, not an exact count: it only has to catch "nothing ran"


def count_tests(report: Path) -> tuple[int, int]:
    """(tests in the report, tests that were not skipped)."""
    cases = list(ET.parse(report).getroot().iter("testcase"))
    ran = [case for case in cases if case.find("skipped") is None]
    return len(cases), len(ran)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", type=Path, help="the pytest --junitxml file")
    parser.add_argument("--min", type=int, default=MIN_DB_TESTS, dest="minimum")
    args = parser.parse_args(argv)
    try:
        total, ran = count_tests(args.report)
    except (OSError, ET.ParseError):
        print(f"cannot read the test report {args.report}")
        return 2
    print(f"{ran} of {total} database tests ran (at least {args.minimum} are required)")
    return 0 if ran >= args.minimum else 1


if __name__ == "__main__":
    sys.exit(main())
