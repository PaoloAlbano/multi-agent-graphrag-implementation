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
    judge_mode: str = "self",
    judge_model: str | None = None,
    concurrency: int = 10,
    only_failed: bool = False,
    only_repaired: bool = False,
    qids: set[str] | None = None,
) -> dict:
    """Re-score one `results/<model>/<config>/<domain>/<mode>/` leaf in place.

    `judge_mode` controls what happens to the leaf's primary score:

    - `"self"` (default): overwrites `trace.jsonl`'s `correct`/
      `judge_reasoning` fields (the deterministic `similarity` is left
      untouched, still useful for reference) and updates the matching
      `results[]` entry in `run.json`, setting `use_judge=true` and
      `judge_mode="self"`. Intended to be run through the leaf's own
      `judge_calls.jsonl` logger.
    - `"fixed"`: does NOT touch the primary score. Instead it records the
      fixed judge's per-question verdicts as `fixed_correct`/`fixed_reasoning`
      on each trace row and a summary under `run.json` `external_judge`
      (`{model, total, correct, accuracy, calls_log, judge_mode: "fixed"}`),
      so the site can show both the primary and the external-judge score side
      by side. Intended to be run through a separate `judge_fixed_calls.jsonl`
      logger.

    `only_failed`: when True, only rows whose judge verdict is missing
    (`fixed_correct is None` in `"fixed"` mode, or `judge_error` set in
    `"self"` mode) are re-scored; rows already judged are left untouched. This
    lets you retry a handful of failed judge calls without re-running (and
    re-logging) the ~99 already-correct rows of a leaf. The `external_judge`
    summary is recomputed from all rows (already-judged + freshly retried).

    Returns a summary dict: `{"total", "correct", "accuracy"}` for whichever
    score was written (primary for `"self"`, external for `"fixed"`).
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

    if qids is not None:
        # Re-score only the rows whose qid was explicitly requested (used to
        # (re)judge freshly repaired rows without touching already-judged ones).
        rows_to_judge = [r for r in rows if r["qid"] in qids]
        if not rows_to_judge:
            logger.info("no rows matching qids in %s; nothing to rejudge", leaf_dir)
            return {
                "total": len(rows),
                "correct": sum(1 for r in rows if r.get("fixed_correct") or (judge_mode == "self" and r.get("correct"))),
                "accuracy": 0.0,
            }
    elif only_repaired:
        # Re-score only rows that were freshly repaired (iterations>0 but no
        # judge verdict yet -- the repair pass runs with --no-judge, so the
        # repaired rows carry the deterministic `correct` but no judge
        # reasoning). In "fixed" mode that's rows without `fixed_correct`; in
        # "self" mode rows without `judge_reasoning`. Already-judged rows are
        # left untouched.
        def _is_repaired(row: dict) -> bool:
            if row.get("iterations", 0) <= 0:
                return False
            return row.get("fixed_correct") is None if judge_mode == "fixed" else not row.get("judge_reasoning")

        repaired = [r for r in rows if _is_repaired(r)]
        if not repaired:
            logger.info("no repaired rows in %s; nothing to rejudge", leaf_dir)
            return {
                "total": len(rows),
                "correct": sum(1 for r in rows if r.get("fixed_correct") or (judge_mode == "self" and r.get("correct"))),
                "accuracy": 0.0,
            }
        rows_to_judge = repaired
    elif only_failed:
        # Re-score only rows whose judge verdict is missing: in "fixed" mode
        # that's rows without `fixed_correct` (a failed call leaves it absent);
        # in "self" mode rows that carry a `judge_error`. Rows already judged
        # are passed through unchanged.
        if judge_mode == "fixed":
            failed = [r for r in rows if r.get("fixed_correct") is None]
        else:
            failed = [r for r in rows if r.get("judge_error")]
        if not failed:
            logger.info("no failed rows in %s; nothing to retry", leaf_dir)
            return {
                "total": len(rows),
                "correct": sum(1 for r in rows if r.get("fixed_correct") or (judge_mode == "self" and r.get("correct"))),
                "accuracy": 0.0,
            }
        rows_to_judge = failed
    else:
        rows_to_judge = rows

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
        if judge_mode == "fixed":
            row["fixed_correct"] = verdict.correct
            row["fixed_reasoning"] = verdict.reasoning
            row["fixed_error"] = None
        else:
            row["correct"] = verdict.correct
            row["judge_reasoning"] = verdict.reasoning
            row["judge_error"] = None
        return row

    if qids is not None or only_repaired:
        # Re-score only the selected rows, then merge them back in original
        # order with the untouched rows.
        retried = await asyncio.gather(*(_rejudge_row(row) for row in rows_to_judge))
        retried_by_id = {id(r): r for r in retried}
        updated_rows = [retried_by_id.get(id(r), r) for r in rows]
    elif only_failed:
        # Re-score only the failed rows, then merge them back in original order
        # with the already-judged rows (left untouched).
        retried = await asyncio.gather(*(_rejudge_row(row) for row in failed))
        retried_by_id = {id(r): r for r in retried}
        updated_rows = [retried_by_id.get(id(r), r) for r in rows]
    else:
        updated_rows = await asyncio.gather(*(_rejudge_row(row) for row in rows))

    with trace_path.open("w", encoding="utf-8") as fh:
        for row in updated_rows:
            fh.write(json.dumps(row, default=str) + "\n")

    total = len(updated_rows)
    if judge_mode == "fixed":
        correct = sum(1 for row in updated_rows if row.get("fixed_correct"))
        accuracy = correct / total if total else 0.0
        manifest["external_judge"] = {
            "model": judge_model,
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "calls_log": "judge_fixed_calls.jsonl",
            "judge_mode": "fixed",
        }
    else:
        correct = sum(1 for row in updated_rows if row["correct"])
        accuracy = correct / total if total else 0.0
        manifest["use_judge"] = True
        manifest["judge_mode"] = "self"
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


def find_leaves(
    results_root: Path,
    *,
    prefix: str | None,
    force: bool,
    judge_mode: str = "self",
    only_failed: bool = False,
    only_repaired: bool = False,
) -> list[Path]:
    """Return leaf directories (each containing `trace.jsonl` + `run.json`)
    to rejudge, filtered by `prefix` (relative to `results_root`).

    Skipping logic depends on `judge_mode`:
    - `"self"`: skip leaves whose `run.json` already has `use_judge=true`,
      unless `force` is set (re-scoring the primary score is destructive).
    - `"fixed"`: never skip on `use_judge` -- an external judge can always be
      (re)run without touching the primary score; only `force` matters if you
      want to overwrite an existing `external_judge` block (otherwise leaves
      that already have one are skipped unless `force` is set).

    `only_failed`: when True, leaves whose rows have no missing verdict are
    skipped even if they already have an `external_judge`/`use_judge` block --
    there is nothing to retry there. Leaves with at least one row whose judge
    verdict is missing (no `fixed_correct` in "fixed" mode, or a `judge_error`
    in "self" mode) are always included, regardless of `force`, so `--only-failed`
    can retry a handful of failed calls without forcing a full leaf re-run.

    `only_repaired`: when True, leaves whose rows have no freshly-repaired rows
    (iterations>0 but no judge verdict yet) are skipped; leaves with at least
    one such row are always included, regardless of `force`. Used after
    `scripts/repair_broken.py` to bring repaired rows' self/fixed verdicts in
    line with the rest of the leaf.
    """
    leaves = []
    for run_json_path in sorted(results_root.rglob("run.json")):
        leaf_dir = run_json_path.parent
        relative = leaf_dir.relative_to(results_root)
        if prefix and not str(relative).startswith(prefix):
            continue
        if not (leaf_dir / "trace.jsonl").exists():
            continue
        manifest = json.loads(run_json_path.read_text(encoding="utf-8"))
        if judge_mode == "fixed":
            already = manifest.get("external_judge") is not None
        else:
            already = manifest.get("use_judge") is True
        if already and not force and not only_failed and not only_repaired:
            continue
        rows = [
            json.loads(line)
            for line in (leaf_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if only_failed:
            if judge_mode == "fixed":
                has_failed = any(r.get("fixed_correct") is None for r in rows)
            else:
                has_failed = any(r.get("judge_error") for r in rows)
            if not has_failed:
                continue
        if only_repaired:
            def _is_repaired(row: dict) -> bool:
                if row.get("iterations", 0) <= 0:
                    return False
                return (
                    row.get("fixed_correct") is None
                    if judge_mode == "fixed"
                    else not row.get("judge_reasoning")
                )

            if not any(_is_repaired(r) for r in rows):
                continue
        leaves.append(leaf_dir)
    return leaves
