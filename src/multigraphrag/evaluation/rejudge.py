"""Re-score already-collected `results/**/trace.jsonl` files with an
LLM-as-a-judge, without re-running the multi-agent/single-pass pipeline.

No Memgraph, no Query Generator/Evaluator/etc. calls needed: the judge only
needs the `question`/`answer` already recorded in `trace.jsonl` plus the gold
answer looked up by `qid` from the CypherBench task file -- one extra LLM
call per question, versus a full re-run of the whole pipeline.
"""

import asyncio
import json
import logging
from pathlib import Path

from multigraphrag.evaluation.judge import LLMJudge
from multigraphrag.llm.call_log import set_call_context

logger = logging.getLogger(__name__)


def _load_gold_answers(dest_dir: Path, split: str) -> dict[str, list]:
    path = dest_dir / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `cypherbench download` first.")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {item["qid"]: json.loads(item["answer_json"]) for item in raw}


async def rejudge_leaf(
    leaf_dir: Path,
    dest_dir: Path,
    judge: LLMJudge,
    *,
    concurrency: int = 10,
) -> dict:
    """Re-score one `results/<model>/<config>/<domain>/<mode>/` leaf in place.

    Overwrites `trace.jsonl`'s `correct`/`judge_reasoning` fields (the
    deterministic `similarity` is left untouched, still useful for
    reference) and updates the matching `results[]` entry in `run.json`.
    Returns a summary dict: `{"total", "correct", "accuracy"}`.
    """
    run_json_path = leaf_dir / "run.json"
    trace_path = leaf_dir / "trace.jsonl"
    manifest = json.loads(run_json_path.read_text(encoding="utf-8"))
    split = manifest["split"]

    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return {"total": 0, "correct": 0, "accuracy": 0.0}

    domain = rows[0]["domain"]
    mode = rows[0]["mode"]
    gold_by_qid = _load_gold_answers(dest_dir, split)

    semaphore = asyncio.Semaphore(concurrency)

    async def _rejudge_row(row: dict) -> dict:
        gold = gold_by_qid.get(row["qid"])
        if gold is None:
            logger.warning("qid %s not found in %s CypherBench tasks; leaving unchanged", row["qid"], split)
            return row
        try:
            async with semaphore:
                with set_call_context(qid=row["qid"], domain=domain, mode=mode):
                    verdict = await judge.judge(
                        question=row["question"], answer=row["answer"], gold_answer_rows=gold
                    )
        except Exception as exc:  # noqa: BLE001 -- one bad judge response must not abort the whole batch
            logger.warning("judge call failed for qid %s (%s): %s", row["qid"], leaf_dir, exc)
            row["judge_error"] = str(exc)
            return row
        row["correct"] = verdict.correct
        row["judge_reasoning"] = verdict.reasoning
        row["judge_error"] = None
        return row

    updated_rows = await asyncio.gather(*(_rejudge_row(row) for row in rows))

    with trace_path.open("w", encoding="utf-8") as fh:
        for row in updated_rows:
            fh.write(json.dumps(row, default=str) + "\n")

    total = len(updated_rows)
    correct = sum(1 for row in updated_rows if row["correct"])
    accuracy = correct / total if total else 0.0

    manifest["use_judge"] = True
    manifest["judge_calls_log"] = "judge_calls.jsonl"
    for entry in manifest.get("results", []):
        if entry["domain"] == domain and entry["mode"] == mode:
            entry["total"] = total
            entry["correct"] = correct
            entry["accuracy"] = accuracy
    run_json_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {"total": total, "correct": correct, "accuracy": accuracy}


def read_leaf_model(leaf_dir: Path) -> str:
    """Return the model recorded in a leaf's `run.json` (used to self-judge)."""
    manifest = json.loads((leaf_dir / "run.json").read_text(encoding="utf-8"))
    return manifest["model"]


def find_leaves(results_root: Path, *, prefix: str | None, force: bool) -> list[Path]:
    """Return leaf directories (each containing `trace.jsonl` + `run.json`)
    to rejudge, filtered by `prefix` (relative to `results_root`) and
    skipping leaves already scored by a judge unless `force` is set."""
    leaves = []
    for run_json_path in sorted(results_root.rglob("run.json")):
        leaf_dir = run_json_path.parent
        relative = leaf_dir.relative_to(results_root)
        if prefix and not str(relative).startswith(prefix):
            continue
        if not (leaf_dir / "trace.jsonl").exists():
            continue
        manifest = json.loads(run_json_path.read_text(encoding="utf-8"))
        if manifest.get("use_judge") and not force:
            continue
        leaves.append(leaf_dir)
    return leaves
