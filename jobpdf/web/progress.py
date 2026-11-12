"""Progress steps for the web page (JM-44): what the person sees while a CV is analysed.

The upload page can ask for the report as a stream (``Accept: text/event-stream``). The
analysis then runs in one worker thread of this process while the same request stays open and
sends short events:

    event: step    data: {"step": "skills"}            the analysis moved on to this step
    event: done    data: {"status": 200, "html": ...}  the finished page (the last event)
    event: failed  data: {"status": 422, "html": ...}  an error page (the last event)
    : keep-alive                                       a comment, so no proxy closes a quiet line

Nothing is kept: no job id, no result waiting to be fetched, no state shared between requests
or workers. If the person closes the page, the analysis still finishes (a thread cannot be
stopped), its result is dropped, and its slot is given back then.

The seven pipeline stages are grouped into five steps a person understands. A step event holds
a step id and nothing else, so no text from a CV can reach the stream except in the finished page,
which is what the person asked for.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, Literal

log = logging.getLogger(__name__)

STREAM_MEDIA_TYPE = "text/event-stream"
HEARTBEAT_S = 10.0  # a quiet stretch is the language model reading the CV; say we are alive
HEARTBEAT = b": keep-alive\n\n"
RENDER_STAGE = "render"  # not a pipeline stage: the web layer drawing the page


@dataclass(frozen=True)
class Step:
    id: str
    label: str


# The steps, in order, as the page lists them. The labels are the page's own wording.
STEPS: tuple[Step, ...] = (
    Step("read", "Reading your file"),
    Step("skills", "Finding your skills and experience"),
    Step("match", "Matching your skills to a standard list"),
    Step("compare", "Comparing you with current jobs"),
    Step("report", "Building your report"),
)
STEP_IDS = tuple(step.id for step in STEPS)

# Which step each pipeline stage belongs to (the first stage of a step starts that step).
STAGE_STEP: dict[str, str] = {
    "parse": "read",
    "sections": "read",
    "extraction": "skills",
    "grounding": "skills",
    "normalization": "match",
    "embedding": "compare",
    "ranking": "compare",
    RENDER_STAGE: "report",
}

Event = Literal["step", "done", "failed"]
TERMINAL: frozenset[str] = frozenset({"done", "failed"})


@dataclass(frozen=True)
class Outcome:
    """How an analysis ended: the last event of the stream."""

    event: Literal["done", "failed"]
    data: dict[str, Any]


def format_event(event: str, data: dict[str, Any]) -> bytes:
    """One server-sent event. JSON never holds a raw line break, so one ``data:`` line is enough."""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n".encode()


class AnalysisStream:
    """One analysis running in a worker thread and the events it produces.

    ``work(on_stage)`` does the analysis and returns its ``Outcome``; it calls ``on_stage`` with
    the name of each stage as it starts. ``release()`` gives back the analysis slot, and is called
    as soon as ``work`` ends, before the last event is sent, so a person who starts another upload
    the moment the report appears finds the slot free.
    """

    def __init__(self, work: Callable[[Callable[[str], None]], Outcome],
                 release: Callable[[], None], *, heartbeat_s: float = HEARTBEAT_S) -> None:
        self._work = work
        self._release = release
        self._heartbeat_s = heartbeat_s
        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
        self._step: str | None = None
        # Started here, not when the response is first read: the slot is already taken, and a
        # client that vanishes before the first read must still give it back.
        self._thread = threading.Thread(target=self._run, name="jobpdf-analysis", daemon=True)
        self._thread.start()

    def _emit(self, event: str, data: dict[str, Any]) -> None:
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, (event, data))
        except RuntimeError:  # the server is shutting down; nobody is listening
            pass

    def _on_stage(self, stage: str) -> None:
        step = STAGE_STEP.get(stage)
        if step is None or step == self._step:
            return
        self._step = step
        self._emit("step", {"step": step})

    def _run(self) -> None:
        try:
            outcome = self._work(self._on_stage)
        except Exception as exc:  # ``work`` turns known errors into an outcome; this is a bug
            log.error("analysis stream failed unexpectedly (%s)", type(exc).__name__)
            outcome = Outcome("failed", {"status": 500, "html": None})
        finally:
            try:
                self._release()
            except Exception as exc:
                log.error("analysis slot could not be released (%s)", type(exc).__name__)
        self._emit(outcome.event, outcome.data)

    async def events(self) -> AsyncIterator[bytes]:
        """The stream's bytes, until the last event."""
        while True:
            try:
                event, data = await asyncio.wait_for(self._queue.get(), self._heartbeat_s)
            except asyncio.TimeoutError:
                yield HEARTBEAT
                continue
            yield format_event(event, data)
            if event in TERMINAL:
                return


def wants_stream(accept_header: str | None) -> bool:
    """True when the request asks for a stream: only an explicit ``text/event-stream`` counts,
    so a browser form post (``text/html``, ``*/*``) always gets the page."""
    if not accept_header:
        return False
    return any(part.split(";")[0].strip().lower() == STREAM_MEDIA_TYPE
               for part in accept_header.split(","))
