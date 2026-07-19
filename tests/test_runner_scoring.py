import pytest

from multigraphrag.evaluation.cypherbench import CypherBenchTask
from multigraphrag.evaluation.judge import JudgeVerdict, LLMJudge
from multigraphrag.evaluation.runner import _score


class _FakeLLMClient:
    def __init__(self, verdict: JudgeVerdict):
        self._verdict = verdict

    async def complete_structured(self, *, system_prompt, user_prompt, response_model):
        return self._verdict


def _task() -> CypherBenchTask:
    return CypherBenchTask(
        qid="q1",
        graph="geography",
        gold_cypher="MATCH (n) RETURN n",
        nl_question="How many doors are there?",
        answer_json="[[3]]",
    )


@pytest.mark.asyncio
async def test_score_without_judge_uses_deterministic_heuristic():
    similarity, correct, judge_reasoning = await _score(
        _task(), records=[{"c": 3}], answer="There are 3.", judge=None
    )
    assert correct is True
    assert judge_reasoning is None


@pytest.mark.asyncio
async def test_score_with_judge_overrides_heuristic_verdict():
    # Deterministic heuristic would call this correct (values overlap), but the
    # judge is authoritative when configured.
    judge = LLMJudge(_FakeLLMClient(JudgeVerdict(correct=False, reasoning="Off by context.")))

    similarity, correct, judge_reasoning = await _score(
        _task(), records=[{"c": 3}], answer="There are 3.", judge=judge
    )

    assert correct is False
    assert judge_reasoning == "Off by context."
    assert similarity == 1.0  # heuristic similarity is still computed for reference
