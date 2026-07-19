"""Optional JSONL transcript of every LLM call made during a run.

Distinct from the CypherBench evaluation trace (`evaluation/runner.py`),
which records one summary row per *question*. This records one row per
*LLM call* -- every agent invocation's system/user prompt and raw response
(or error) -- for full auditability of a run (e.g. "show me every request and
response the Query Evaluator made while scoring domain X").
"""

import json
import time
from pathlib import Path


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
        self._file = path.open("a", encoding="utf-8")

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
        record = {
            "ts": time.time(),
            "agent": agent,
            "model": model,
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
