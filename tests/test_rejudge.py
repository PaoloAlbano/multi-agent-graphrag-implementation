import json

import pytest

from multigraphrag.composition import build_self_judge
from multigraphrag.config import Settings
from multigraphrag.evaluation.judge import JudgeVerdict, LLMJudge
from multigraphrag.evaluation.rejudge import find_leaves, read_leaf_model, rejudge_leaf


class _FakeLLMClient:
    def __init__(self, verdict: JudgeVerdict):
        self._verdict = verdict

    async def complete_structured(self, *, system_prompt, user_prompt, response_model):
        return self._verdict


class _RaisingLLMClient:
    async def complete_structured(self, *, system_prompt, user_prompt, response_model):
        raise AttributeError("'NoneType' object has no attribute 'strip'")


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _make_leaf(tmp_path, dest_dir):
    leaf = tmp_path / "results" / "modelA" / "temp0.0-reasoning-medium" / "geography" / "single"
    leaf.mkdir(parents=True)
    _write_jsonl(
        leaf / "trace.jsonl",
        [
            {
                "qid": "q1",
                "domain": "geography",
                "mode": "single",
                "question": "How many doors?",
                "gold_cypher": "MATCH (n) RETURN n",
                "generated_cypher": "MATCH (n) RETURN n",
                "answer": "There are 3 doors.",
                "accepted": True,
                "iterations": 1,
                "similarity": 0.1,
                "correct": False,
                "error": None,
                "judge_reasoning": None,
            }
        ],
    )
    (leaf / "run.json").write_text(
        json.dumps(
            {
                "model": "modelA",
                "temperature": 0.0,
                "split": "test",
                "domains": ["geography"],
                "modes": ["single"],
                "use_judge": False,
                "results": [
                    {"domain": "geography", "mode": "single", "total": 1, "correct": 0, "accuracy": 0.0}
                ],
            }
        ),
        encoding="utf-8",
    )

    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "test.json").write_text(
        json.dumps([{"qid": "q1", "graph": "geography", "answer_json": "[[3]]"}]),
        encoding="utf-8",
    )
    return leaf


@pytest.mark.asyncio
async def test_rejudge_leaf_updates_trace_and_manifest(tmp_path):
    dest_dir = tmp_path / "cypherbench"
    leaf = _make_leaf(tmp_path, dest_dir)
    judge = LLMJudge(_FakeLLMClient(JudgeVerdict(correct=True, reasoning="Matches the gold value.")))

    summary = await rejudge_leaf(leaf, dest_dir, judge, concurrency=2)

    assert summary == {"total": 1, "correct": 1, "accuracy": 1.0}

    rows = [json.loads(line) for line in (leaf / "trace.jsonl").read_text().splitlines()]
    assert rows[0]["correct"] is True
    assert rows[0]["judge_reasoning"] == "Matches the gold value."
    assert rows[0]["similarity"] == 0.1  # deterministic score left untouched

    manifest = json.loads((leaf / "run.json").read_text())
    assert manifest["use_judge"] is True
    assert manifest["judge_mode"] == "self"
    assert manifest["results"][0]["correct"] == 1
    assert manifest["results"][0]["accuracy"] == 1.0


@pytest.mark.asyncio
async def test_rejudge_leaf_fixed_records_external_judge_without_touching_primary(tmp_path):
    dest_dir = tmp_path / "cypherbench"
    leaf = _make_leaf(tmp_path, dest_dir)
    judge = LLMJudge(_FakeLLMClient(JudgeVerdict(correct=True, reasoning="Matches the gold value.")))

    summary = await rejudge_leaf(
        leaf, dest_dir, judge, judge_mode="fixed", judge_model="fixed-judge-model", concurrency=2
    )

    assert summary == {"total": 1, "correct": 1, "accuracy": 1.0}

    rows = [json.loads(line) for line in (leaf / "trace.jsonl").read_text().splitlines()]
    assert rows[0]["fixed_correct"] is True
    assert rows[0]["fixed_reasoning"] == "Matches the gold value."
    # primary score untouched by the external judge
    assert rows[0]["correct"] is False
    assert rows[0]["judge_reasoning"] is None

    manifest = json.loads((leaf / "run.json").read_text())
    assert manifest.get("use_judge") is False  # primary score still un-judged
    assert manifest["external_judge"] == {
        "model": "fixed-judge-model",
        "total": 1,
        "correct": 1,
        "accuracy": 1.0,
        "calls_log": "judge_fixed_calls.jsonl",
        "judge_mode": "fixed",
    }
    # primary results[] entry left as-is
    assert manifest["results"][0]["correct"] == 0


@pytest.mark.asyncio
async def test_rejudge_leaf_survives_a_bad_judge_response(tmp_path):
    dest_dir = tmp_path / "cypherbench"
    leaf = _make_leaf(tmp_path, dest_dir)
    judge = LLMJudge(_RaisingLLMClient())

    summary = await rejudge_leaf(leaf, dest_dir, judge, concurrency=2)

    assert summary == {"total": 1, "correct": 0, "accuracy": 0.0}

    rows = [json.loads(line) for line in (leaf / "trace.jsonl").read_text().splitlines()]
    assert rows[0]["correct"] is False  # left at the prior (deterministic) value, unchanged
    assert "NoneType" in rows[0]["judge_error"]


def test_read_leaf_model(tmp_path):
    dest_dir = tmp_path / "cypherbench"
    leaf = _make_leaf(tmp_path, dest_dir)
    assert read_leaf_model(leaf) == "modelA"


def test_build_self_judge_overrides_only_the_model():
    settings = Settings()
    judge, client = build_self_judge(settings, "some/other-model", call_logger=None)
    assert isinstance(judge, LLMJudge)
    assert client._settings.model == "some/other-model"
    assert client._settings.base_url == settings.llm.base_url


def test_find_leaves_filters_prefix_and_already_judged(tmp_path):
    results_root = tmp_path / "results"
    judged = results_root / "modelA" / "cfg" / "domain" / "single"
    unjudged = results_root / "modelB" / "cfg" / "domain" / "single"
    for leaf, use_judge in [(judged, True), (unjudged, False)]:
        leaf.mkdir(parents=True)
        (leaf / "trace.jsonl").write_text("{}\n", encoding="utf-8")
        (leaf / "run.json").write_text(json.dumps({"use_judge": use_judge}), encoding="utf-8")

    assert find_leaves(results_root, prefix=None, force=False) == [unjudged]
    assert set(find_leaves(results_root, prefix=None, force=True)) == {judged, unjudged}
    assert find_leaves(results_root, prefix="modelA", force=False) == []
    assert find_leaves(results_root, prefix="modelA", force=True) == [judged]


def test_find_leaves_fixed_never_skips_on_use_judge(tmp_path):
    results_root = tmp_path / "results"
    self_judged = results_root / "modelA" / "cfg" / "domain" / "single"
    ext_judged = results_root / "modelB" / "cfg" / "domain" / "single"
    for leaf, use_judge, ext in [
        (self_judged, True, None),
        (ext_judged, False, {"model": "fixed"}),
    ]:
        leaf.mkdir(parents=True)
        (leaf / "trace.jsonl").write_text("{}\n", encoding="utf-8")
        manifest = {"use_judge": use_judge}
        if ext is not None:
            manifest["external_judge"] = ext
        (leaf / "run.json").write_text(json.dumps(manifest), encoding="utf-8")

    # fixed mode: a leaf with use_judge=true (self-judged) is still eligible
    assert find_leaves(results_root, prefix=None, force=False, judge_mode="fixed") == [self_judged]
    # ...but one that already has an external_judge block is skipped unless forced
    assert find_leaves(results_root, prefix=None, force=True, judge_mode="fixed") == [self_judged, ext_judged]


def test_find_leaves_only_failed_includes_leaves_with_missing_verdict(tmp_path):
    results_root = tmp_path / "results"
    clean = results_root / "modelA" / "cfg" / "domain" / "single"
    has_failed = results_root / "modelB" / "cfg" / "domain" / "single"
    clean.mkdir(parents=True)
    has_failed.mkdir(parents=True)
    (clean / "trace.jsonl").write_text(
        json.dumps({"qid": "q1", "fixed_correct": True}) + "\n", encoding="utf-8"
    )
    (has_failed / "trace.jsonl").write_text(
        json.dumps({"qid": "q1", "fixed_correct": None}) + "\n", encoding="utf-8"
    )
    for leaf, ext in [(clean, {"model": "fixed"}), (has_failed, {"model": "fixed"})]:
        (leaf / "run.json").write_text(json.dumps({"external_judge": ext}), encoding="utf-8")

    # only_failed: only the leaf with a missing fixed_correct is included, even
    # though both already have an external_judge block (no force needed).
    assert find_leaves(results_root, prefix=None, force=False, judge_mode="fixed", only_failed=True) == [has_failed]


@pytest.mark.asyncio
async def test_rejudge_leaf_only_failed_retries_only_missing_verdicts(tmp_path):
    dest_dir = tmp_path / "cypherbench"
    leaf = _make_leaf(tmp_path, dest_dir)
    # Simulate a leaf that was already externally judged except for one row
    # whose judge call failed (fixed_correct missing).
    trace = leaf / "trace.jsonl"
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    rows[0]["fixed_correct"] = None
    rows[0]["fixed_reasoning"] = None
    rows[0]["fixed_error"] = "LLM endpoint returned empty content (message.content is null)"
    rows.append(
        {
            "qid": "q2",
            "domain": "geography",
            "mode": "single",
            "question": "Second question?",
            "answer": "Two.",
            "similarity": 0.9,
            "correct": True,
            "fixed_correct": True,
            "fixed_reasoning": "ok",
        }
    )
    _write_jsonl(trace, rows)
    (dest_dir / "test.json").write_text(
        json.dumps(
            [
                {"qid": "q1", "graph": "geography", "answer_json": "[[3]]"},
                {"qid": "q2", "graph": "geography", "answer_json": "[[2]]"},
            ]
        ),
        encoding="utf-8",
    )
    manifest = json.loads((leaf / "run.json").read_text())
    manifest["external_judge"] = {"model": "fixed", "total": 2, "correct": 1, "accuracy": 0.5}
    (leaf / "run.json").write_text(json.dumps(manifest), encoding="utf-8")

    judge = LLMJudge(_FakeLLMClient(JudgeVerdict(correct=True, reasoning="Matches the gold value.")))

    summary = await rejudge_leaf(
        leaf, dest_dir, judge, judge_mode="fixed", judge_model="fixed", concurrency=2, only_failed=True
    )

    # q1 (the failed row) is retried and now correct; q2 untouched.
    assert summary == {"total": 2, "correct": 2, "accuracy": 1.0}
    out_rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert out_rows[0]["fixed_correct"] is True
    assert out_rows[0]["fixed_error"] is None
    assert out_rows[1]["fixed_correct"] is True
    assert out_rows[1]["fixed_reasoning"] == "ok"  # untouched

    manifest = json.loads((leaf / "run.json").read_text())
    assert manifest["external_judge"]["correct"] == 2
    assert manifest["external_judge"]["total"] == 2
