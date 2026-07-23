"""One-off migration of the single fully-clean exploratory run from this
project's early development (Qwen/Qwen3.5-27B, temperature=1, all 5
CypherBench domains, single+agentic, zero errors across all 800 questions)
into the curated `results/` layout consumed by `scripts/build_recap.py` and
`scripts/build_site.py`.

Not a general-purpose or repeatedly-run tool: every other exploratory trace
produced during that development session had at least one failed question
(see the session's own audit) and is intentionally left out of the public
`results/` directory. Run once, then delete or ignore.

The old `calls_*.jsonl` files predate per-question `qid` tagging in
`llm/call_log.py`, and one call log covers several domains/modes combined --
it cannot be split or attributed to an individual question. Each combined
call log is copied once into a shared `_legacy-calls/` directory per config,
and every affected leaf's `run.json` points at it with `calls_log_scoped:
false` so the site can render an honest "not attributable" notice instead of
pretending it has per-question detail it doesn't.
"""

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

SOURCE_DIR = Path("data/cypherbench")
RESULTS_ROOT = Path("results")
MODEL_DIR = "Qwen--Qwen3.5-27B"
CONFIG_DIR = "temp1.0-reasoning-untracked"
CONFIG_ROOT = RESULTS_ROOT / MODEL_DIR / CONFIG_DIR

RUNS = [
    {
        "trace": SOURCE_DIR / "trace_art_qwen35_t1.jsonl",
        "calls": SOURCE_DIR / "calls" / "calls_art_qwen35_t1.jsonl",
        "split": "train",
        "legacy_calls_name": "art-run-calls.jsonl",
    },
    {
        "trace": SOURCE_DIR / "trace_test4_qwen35_t1.jsonl",
        "calls": SOURCE_DIR / "calls" / "calls_test4_qwen35_t1.jsonl",
        "split": "test",
        "legacy_calls_name": "test4-run-calls.jsonl",
    },
]


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def main() -> None:
    generated_at = datetime.now(UTC).isoformat()
    written_leaves = 0

    for run in RUNS:
        rows = _load_jsonl(run["trace"])
        errored = [r for r in rows if r.get("error")]
        if errored:
            raise SystemExit(
                f"Refusing to migrate {run['trace']}: {len(errored)} row(s) have a non-null "
                "error, so this run is not fully clean (see the session's own audit criteria)."
            )

        legacy_calls_dest = CONFIG_ROOT / "_legacy-calls" / run["legacy_calls_name"]
        legacy_calls_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run["calls"], legacy_calls_dest)

        by_domain_mode: dict[tuple[str, str], list[dict]] = {}
        for row in rows:
            by_domain_mode.setdefault((row["domain"], row["mode"]), []).append(row)

        for (domain, mode), domain_rows in by_domain_mode.items():
            leaf_dir = CONFIG_ROOT / domain / mode
            leaf_dir.mkdir(parents=True, exist_ok=True)

            trace_path = leaf_dir / "trace.jsonl"
            with trace_path.open("w", encoding="utf-8") as fh:
                for row in domain_rows:
                    fh.write(json.dumps(row, default=str) + "\n")

            total = len(domain_rows)
            correct = sum(1 for row in domain_rows if row.get("correct"))

            manifest = {
                "model": "Qwen/Qwen3.5-27B",
                "temperature": 1.0,
                "temperature_source": "inferred_from_filename ('_t1' suffix)",
                "reasoning_enabled": None,
                "reasoning_effort": None,
                "reasoning_note": "predates reasoning_enabled/reasoning_effort config support in this codebase",
                "max_tokens": None,
                "structured_output_mode": None,
                "split": run["split"],
                "domains": [domain],
                "modes": [mode],
                "graph_variant": "simplekg",
                "graph_variant_source": (
                    "assumed from session convention (every model-comparison run in this "
                    "development session used the full-scale simplekg variant); not recorded "
                    "in the legacy run itself"
                ),
                "concurrency": None,
                "concurrency_note": "not recorded for legacy runs",
                "limit": 40,
                "use_judge": False,
                "calls_log": os.path.relpath(legacy_calls_dest, leaf_dir),
                "calls_log_scoped": False,
                "calls_log_note": (
                    "Combined transcript for the whole legacy run (all domains/modes in this "
                    "batch); predates per-question qid tagging, so it cannot be filtered down "
                    "to this specific domain/mode/question."
                ),
                "generated_at": generated_at,
                "migrated": True,
                "migration_note": (
                    f"Migrated from {run['trace']} + {run['calls']}; confirmed zero rows with "
                    "a non-null error before migration."
                ),
                "results": [
                    {
                        "domain": domain,
                        "mode": mode,
                        "total": total,
                        "correct": correct,
                        "accuracy": correct / total if total else 0.0,
                    }
                ],
            }
            (leaf_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            written_leaves += 1

    print(f"Migrated {written_leaves} domain/mode leaves into {CONFIG_ROOT}")


if __name__ == "__main__":
    main()
