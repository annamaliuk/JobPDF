"""Make an API key for a caller (JM-30).

Prints the key once, for you to hand to the caller, and the ``name:sha256`` line to add to
``JOBPDF_API_KEYS`` (in .env) or to the file named by ``JOBPDF_API_KEYS_FILE``. The server
keeps only the hash, so a lost key cannot be recovered, only replaced.

Usage (from the repo root):
    uv run python scripts/make_api_key.py partner-acme
"""

from __future__ import annotations

import argparse
import sys

from jobpdf.api.access import KEY_NAME_PATTERN, hash_key, new_api_key


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="who the key is for: letters, digits, _ . - (max 64)")
    args = parser.parse_args(argv)

    if not KEY_NAME_PATTERN.match(args.name):
        print("name: use letters, digits, _ . - and at most 64 characters")
        return 1
    key = new_api_key()
    print(f"API key for {args.name} (shown once; give it to the caller): {key}")
    print("Add this line to JOBPDF_API_KEYS or the keys file:")
    print(f"{args.name}:{hash_key(key)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
