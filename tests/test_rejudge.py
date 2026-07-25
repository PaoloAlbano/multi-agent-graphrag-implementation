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
    assert manifest["results"][0]["correct"] == 1
    assert manifest["results"][0]["accuracy"] == 1.0


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
