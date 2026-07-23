"""Optional JSONL transcript of every LLM call made during a run.

Distinct from the CypherBench evaluation trace (`evaluation/runner.py`),
which records one summary row per *question*. This records one row per
*LLM call* -- every agent invocation's system/user prompt and raw response
(or error) -- for full auditability of a run (e.g. "show me every request and
response the Query Evaluator made while scoring domain X").
"""

import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

#: Set by callers that know which CypherBench question a call belongs to
#: (`evaluation.runner`), read by `CallLogger.log()` below. `None` (the
#: default) for callers with no such notion, e.g. `multigraphrag ask`. Each
#: `asyncio` task gets its own copy of the context on creation, so concurrent
#: questions never see each other's qid/domain/mode.
_call_context: ContextVar[dict | None] = ContextVar("_call_context", default=None)


@contextmanager
def set_call_context(*, qid: str, domain: str, mode: str):
    """Attach a CypherBench question's id/domain/mode to every call log entry
    written while the context manager is active, so a run's `calls.jsonl` can
    be filtered back to the exact question that produced each LLM call."""
    token = _call_context.set({"qid": qid, "domain": domain, "mode": mode})
    try:
        yield
    finally:
        _call_context.reset(token)


class CallLogger:
    """Appends one JSON line per LLM call to a file.

    Writes are synchronous, unbuffered-by-flush file I/O with no `await` in
    between, so under asyncio's single-threaded cooperative scheduling
    (including when several agent calls run concurrently under a
    semaphore-bounded fan-out) two `log()` calls can never interleave
    mid-write -- no lock is needed.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate, not append: each CLI invocation is a complete, self-contained
        # run, matching how `--trace` is opened in `evaluation/runner.py`. Appending
        # to a stale file left over from an earlier invocation at the same path
        # would silently mix two runs' calls together under the same qids.
        self._file = path.open("w", encoding="utf-8")

    def log(
        self,
        *,
        agent: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response: str | None,
        error: str | None,
    ) -> None:
        context = _call_context.get()
        record = {
            "ts": time.time(),
            "agent": agent,
            "model": model,
            "qid": context["qid"] if context else None,
            "domain": context["domain"] if context else None,
            "mode": context["mode"] if context else None,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response": response,
            "error": error,
        }
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "CallLogger":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
