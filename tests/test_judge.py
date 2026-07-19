import pytest

from multigraphrag.evaluation.judge import JudgeVerdict, LLMJudge


class _FakeLLMClient:
    def __init__(self, verdict: JudgeVerdict):
        self._verdict = verdict
        self.captured_user_prompt = None

    async def complete_structured(self, *, system_prompt, user_prompt, response_model):
        self.captured_user_prompt = user_prompt
        return self._verdict


@pytest.mark.asyncio
async def test_judge_returns_llm_verdict_and_includes_gold_and_answer_in_prompt():
    llm = _FakeLLMClient(JudgeVerdict(correct=True, reasoning="Matches the ground truth."))
    judge = LLMJudge(llm)

    verdict = await judge.judge(
        question="How many doors are there?",
        answer="There are 3 doors.",
        gold_answer_rows=[[3]],
    )

    assert verdict.correct is True
    assert "How many doors are there?" in llm.captured_user_prompt
    assert "There are 3 doors." in llm.captured_user_prompt
    assert "[[3]]" in llm.captured_user_prompt
