"""The sample CV a visitor can try instead of uploading their own (JM-52).

The file is bundled with the app (``SAMPLE_DIR``, or the path in ``SAMPLE_CV_PATH``) and read
once at startup. When there is no usable file the feature is simply off: the upload page shows
no sample button and the sample routes answer 404.

A sample report runs the whole pipeline on that file, like an upload, with one difference: the
language model's answers are kept in memory by ``MemoCaller`` for the life of the process. The
first sample report costs one model call; later ones reuse its answer, so every visitor sees the
same profile (no model randomness mid-demo) quickly, while the ranking is still computed against
the vacancies of the moment and the visitor's own choices. Only the sample's calls go through the
memo; uploaded CVs never do, and nothing is written to disk.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from jobpdf.extraction.llm import ToolCaller, ToolCallResult, default_caller
from jobpdf.pipeline import ALLOWED_SUFFIXES, MAX_UPLOAD_BYTES

log = logging.getLogger(__name__)

SAMPLE_CV_ENV = "SAMPLE_CV_PATH"  # a file to use instead of the bundled one
SAMPLE_DIR = Path(__file__).resolve().parents[1] / "web" / "sample"
DEFAULT_NAMES = ("sample_cv.pdf", "sample_cv.docx")  # looked for in SAMPLE_DIR, in this order
MAX_MEMO = 8  # model answers kept; one sample needs one or two (a correction retry)
MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

# Why a sample file was not used: logged as a code, never with the file's content.
SAMPLE_MISSING = "sample_missing"
SAMPLE_UNSUPPORTED = "sample_unsupported_type"
SAMPLE_TOO_LARGE = "sample_too_large"
SAMPLE_EMPTY = "sample_empty"
SAMPLE_UNREADABLE = "sample_unreadable"


class MemoCaller:
    """A ``ToolCaller`` that remembers successful answers by their exact request.

    The real caller is made on first use, so a missing API key surfaces inside the extraction
    (where the pipeline turns it into its usual error) and not at startup. Failed calls are not
    remembered. The lock only guards the memo: two first requests at once both ask the model
    rather than one waiting on the other's network call while holding an analysis slot.
    """

    def __init__(self, make: Callable[[], ToolCaller], max_entries: int = MAX_MEMO) -> None:
        self._make = make
        self._inner: ToolCaller | None = None
        self._memo: OrderedDict[str, ToolCallResult] = OrderedDict()
        self._max = max_entries
        self._lock = threading.Lock()
        self.calls = 0  # real model calls made, for tests and the log

    def _caller(self) -> ToolCaller:
        with self._lock:
            if self._inner is None:
                self._inner = self._make()
            return self._inner

    @property
    def model(self) -> str:
        return self._caller().model

    def call(self, *, system: str, user: str, tool: dict, max_tokens: int) -> ToolCallResult:
        key = request_key(system=system, user=user, tool=tool, max_tokens=max_tokens)
        with self._lock:
            if key in self._memo:
                self._memo.move_to_end(key)
                return self._memo[key]
        result = self._caller().call(system=system, user=user, tool=tool, max_tokens=max_tokens)
        with self._lock:
            self.calls += 1
            self._memo[key] = result
            while len(self._memo) > self._max:
                self._memo.popitem(last=False)
        log.info("sample CV: model answer kept in memory (%d kept)", len(self._memo))
        return result

    def __len__(self) -> int:
        return len(self._memo)


def request_key(*, system: str, user: str, tool: dict, max_tokens: int) -> str:
    payload = {"system": system, "user": user, "tool": tool, "max_tokens": max_tokens}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SampleCV:
    content: bytes
    filename: str  # what the pipeline is told; the suffix decides the parser
    caller: MemoCaller = field(compare=False)

    @property
    def media_type(self) -> str:
        return MEDIA_TYPES[Path(self.filename).suffix.lower()]


def sample_path(environ: Mapping[str, str] | None = None,
                sample_dir: Path = SAMPLE_DIR) -> Path | None:
    """The configured file, else the first bundled one that exists, else None."""
    environ = os.environ if environ is None else environ
    configured = (environ.get(SAMPLE_CV_ENV) or "").strip()
    if configured:
        return Path(configured)
    for name in DEFAULT_NAMES:
        candidate = sample_dir / name
        if candidate.is_file():
            return candidate
    return None


def load_sample(path: Path | None = None,
                make_caller: Callable[[], ToolCaller] = default_caller) -> SampleCV | None:
    """The sample, or None (feature off) when there is no usable file; the reason is logged as a
    code. ``path`` defaults to ``sample_path()``."""
    path = sample_path() if path is None else path
    problem = _check(path)
    if problem is not None:
        log_fn = log.info if problem == SAMPLE_MISSING else log.warning
        log_fn("sample CV off: %s", problem)
        return None
    try:
        content = path.read_bytes()
    except OSError:
        log.warning("sample CV off: %s", SAMPLE_UNREADABLE)
        return None
    suffix = path.suffix.lower()
    log.info("sample CV on: %s, %d KB", suffix.lstrip("."), len(content) // 1024)
    return SampleCV(content=content, filename=f"sample_cv{suffix}",
                    caller=MemoCaller(make_caller))


def _check(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return SAMPLE_MISSING
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        return SAMPLE_UNSUPPORTED
    try:
        size = path.stat().st_size
    except OSError:
        return SAMPLE_UNREADABLE
    if size == 0:
        return SAMPLE_EMPTY
    if size > MAX_UPLOAD_BYTES:
        return SAMPLE_TOO_LARGE
    return None
