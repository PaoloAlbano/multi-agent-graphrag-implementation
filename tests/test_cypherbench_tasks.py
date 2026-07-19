import json

from multigraphrag.evaluation.cypherbench import load_tasks


def _make_task(qid: str, graph: str) -> dict:
    return {
        "qid": qid,
        "graph": graph,
        "gold_cypher": "MATCH (n) RETURN n",
        "nl_question": "q",
        "answer_json": "[]",
    }


def test_load_tasks_limit_applies_per_domain_not_globally(tmp_path):
    """A global cap over the combined, multi-domain list would let whichever
    domain appears first in the file crowd out later domains -- e.g. asking
    for 2 domains at limit=40 could silently return 40/0 instead of 40/40.
    """
    raw = [_make_task(f"art{i}", "art") for i in range(5)] + [
        _make_task(f"bio{i}", "biology") for i in range(3)
    ]
    (tmp_path / "test.json").write_text(json.dumps(raw), encoding="utf-8")

    tasks = load_tasks(tmp_path, "test", domains=["art", "biology"], limit=2)

    counts: dict[str, int] = {}
    for task in tasks:
        counts[task.graph] = counts.get(task.graph, 0) + 1

    assert counts == {"art": 2, "biology": 2}


def test_load_tasks_limit_larger_than_available_returns_all(tmp_path):
    raw = [_make_task(f"art{i}", "art") for i in range(3)]
    (tmp_path / "test.json").write_text(json.dumps(raw), encoding="utf-8")

    tasks = load_tasks(tmp_path, "test", domains=["art"], limit=40)

    assert len(tasks) == 3


def test_load_tasks_without_limit_returns_everything_matching_domains(tmp_path):
    raw = [_make_task(f"art{i}", "art") for i in range(3)] + [_make_task("bio0", "biology")]
    (tmp_path / "test.json").write_text(json.dumps(raw), encoding="utf-8")

    tasks = load_tasks(tmp_path, "test", domains=["art"])

    assert len(tasks) == 3
    assert all(t.graph == "art" for t in tasks)
