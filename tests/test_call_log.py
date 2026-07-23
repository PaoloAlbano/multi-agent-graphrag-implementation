import asyncio
import json

from multigraphrag.llm.call_log import CallLogger, set_call_context


def _read_records(path):
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def test_log_without_context_leaves_qid_domain_mode_none(tmp_path):
    logger = CallLogger(tmp_path / "calls.jsonl")
    logger.log(
        agent="query_generator", model="m", system_prompt="s", user_prompt="u", response="r", error=None
    )
    logger.close()

    record = _read_records(tmp_path / "calls.jsonl")[0]
    assert record["qid"] is None
    assert record["domain"] is None
    assert record["mode"] is None


def test_log_inside_context_attaches_qid_domain_mode(tmp_path):
    logger = CallLogger(tmp_path / "calls.jsonl")
    with set_call_context(qid="q1", domain="geography", mode="single"):
        logger.log(
            agent="query_generator", model="m", system_prompt="s", user_prompt="u", response="r", error=None
        )
    logger.close()

    record = _read_records(tmp_path / "calls.jsonl")[0]
    assert record["qid"] == "q1"
    assert record["domain"] == "geography"
    assert record["mode"] == "single"


def test_concurrent_contexts_do_not_leak_across_tasks(tmp_path):
    logger = CallLogger(tmp_path / "calls.jsonl")

    async def _run(qid: str) -> None:
        with set_call_context(qid=qid, domain="geography", mode="agentic"):
            await asyncio.sleep(0)  # yield control so tasks interleave
            logger.log(
                agent="query_generator",
                model="m",
                system_prompt="s",
                user_prompt="u",
                response="r",
                error=None,
            )

    async def _main() -> None:
        await asyncio.gather(*(_run(qid) for qid in ("q1", "q2", "q3")))

    asyncio.run(_main())
    logger.close()

    seen_qids = {record["qid"] for record in _read_records(tmp_path / "calls.jsonl")}
    assert seen_qids == {"q1", "q2", "q3"}
