#!/usr/bin/env bash
# Run the extraction tests, including real OCR, inside a container with Tesseract
# (JM-9). Tesseract is not installed on developer machines by design.
#
# Usage (from anywhere in the repo):
#   scripts/test_ocr_docker.sh                 # all tests in tests/extraction
#   scripts/test_ocr_docker.sh -k real_ocr     # extra arguments go to pytest
set -euo pipefail

IMAGE=jobpdf-ocr-test
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Git Bash on Windows: docker needs a Windows path, and MSYS must not rewrite /app.
if host_path="$(cd "$repo_root" && pwd -W 2>/dev/null)"; then
  export MSYS_NO_PATHCONV=1
else
  host_path="$repo_root"
fi

docker build -f "$host_path/docker/Dockerfile.ocr-test" -t "$IMAGE" "$host_path"

# Read-only mount: tests must not write into the repo (they use tmp_path).
docker run --rm -v "$host_path:/app:ro" "$IMAGE" \
  uv run --frozen --no-sync python -m pytest tests/extraction -v -rs -p no:cacheprovider "$@"
