"""Schema check for `results/**` before it's merged.

Validates every `trace.jsonl` (one row per CypherBench question), `run.json`
(the run manifest), and top-level `calls.jsonl` (the LLM call transcript --
legacy `_legacy-calls/*.jsonl` combined transcripts are exempt from the
qid/domain/mode requirement, since they predate that tagging) found under
`results/`. Checks the whole tree rather than just the PR diff -- simpler and
cheap enough at this repo's current size.

Exits non-zero with a list of every problem found if anything is malformed,
so it can be wired up as a required PR check
(`.github/workflows/validate-results.yml`).
"""

import json
import sys
from pathlib import Path

RESULTS_ROOT = Path("results")

TRACE_REQUIRED_FIELDS = {
    "qid": str,
    "domain": str,
    "mode": str,
    "question": str,
    "gold_cypher": str,
    "answer": str,
    "accepted": bool,
    "iterations": int,
    "similarity": (int, float),
    "correct": bool,
}
TRACE_MODES = {"single", "agentic"}

CALLS_REQUIRED_FIELDS = {
    "agent": str,
    "model": str,
    "system_prompt": str,
    "user_prompt": str,
}

RUN_MANIFEST_REQUIRED_FIELDS = {
    "model": str,
    "temperature": (int, float),
    "split": str,
    "domains": list,
    "modes": list,
    "results": list,
}
RUN_RESULT_ENTRY_REQUIRED_FIELDS = {
    "domain": str,
    "mode": str,
    "total": int,
    "correct": int,
    "accuracy": (int, float),
}


def _check_fields(record: dict, required: dict, *, where: str, errors: list[str]) -> None:
    for field, expected_type in required.items():
        if field not in record:
            errors.append(f"{where}: missing field {field!r}")
            continue
        value = record[field]
        if value is not None and not isinstance(value, expected_type):
            errors.append(
                f"{where}: field {field!r} has type {type(value).__name__}, expected {expected_type}"
            )


def _validate_jsonl(path: Path, required: dict, *, errors: list[str], extra_check=None) -> None:
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path}:{i}"
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{where}: invalid JSON ({exc})")
            continue
        if not isinstance(record, dict):
            errors.append(f"{where}: row is not a JSON object")
            continue
        _check_fields(record, required, where=where, errors=errors)
        if extra_check:
            extra_check(record, where, errors)


def _validate_trace_row(record: dict, where: str, errors: list[str]) -> None:
    mode = record.get("mode")
    if mode is not None and mode not in TRACE_MODES:
        errors.append(f"{where}: mode {mode!r} not one of {TRACE_MODES}")


def _validate_run_json(path: Path, errors: list[str]) -> None:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"{path}: invalid JSON ({exc})")
        return
    if not isinstance(manifest, dict):
        errors.append(f"{path}: root is not a JSON object")
        return
    _check_fields(manifest, RUN_MANIFEST_REQUIRED_FIELDS, where=str(path), errors=errors)
    for i, entry in enumerate(manifest.get("results", [])):
        if not isinstance(entry, dict):
            errors.append(f"{path}: results[{i}] is not a JSON object")
            continue
        _check_fields(entry, RUN_RESULT_ENTRY_REQUIRED_FIELDS, where=f"{path}:results[{i}]", errors=errors)


def main() -> None:
    if not RESULTS_ROOT.exists():
        print("results/ does not exist -- nothing to validate.")
        return

    errors: list[str] = []

    for trace_path in RESULTS_ROOT.rglob("trace.jsonl"):
        _validate_jsonl(trace_path, TRACE_REQUIRED_FIELDS, errors=errors, extra_check=_validate_trace_row)

    for calls_path in (*RESULTS_ROOT.rglob("calls.jsonl"), *RESULTS_ROOT.rglob("judge_calls.jsonl")):
        _validate_jsonl(calls_path, CALLS_REQUIRED_FIELDS, errors=errors)

    for run_json_path in RESULTS_ROOT.rglob("run.json"):
        _validate_run_json(run_json_path, errors=errors)

    if errors:
        print(f"Found {len(errors)} problem(s) in results/:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    print("results/ passed validation.")


if __name__ == "__main__":
    main()
