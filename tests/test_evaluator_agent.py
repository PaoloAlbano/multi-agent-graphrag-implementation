import pytest

from multigraphrag.agents.evaluator import QueryEvaluatorAgent
from multigraphrag.graph.models import QueryOutcome
from multigraphrag.schemas import EvaluationResult, EvaluationStatus


class _CapturingLLMClient:
    def __init__(self, result):
        self.result = result
        self.captured_user_prompt = None

    async def complete_structured(self, *, system_prompt, user_prompt, response_model):
        self.captured_user_prompt = user_prompt
        return self.result


@pytest.mark.asyncio
async def test_evaluate_includes_generator_rationale_in_prompt():
    """The Evaluator must see the Query Generator's own explanation of its query
    (paper's Evaluator criterion #1: consistency with the NL explanation) -- it was
    previously generated but discarded before reaching this agent.
    """
    llm = _CapturingLLMClient(EvaluationResult(status=EvaluationStatus.ACCEPT, feedback="ok"))
    agent = QueryEvaluatorAgent(llm)

    outcome = QueryOutcome(success=True, records=[{"x": 1}])
    await agent.evaluate(
        question="how many?",
        cypher="MATCH (n) RETURN count(n)",
        rationale="Counts all nodes to answer the question.",
        outcome=outcome,
    )

    assert "Counts all nodes to answer the question." in llm.captured_user_prompt


@pytest.mark.asyncio
async def test_evaluate_skips_llm_call_on_execution_error():
    llm = _CapturingLLMClient(None)
    agent = QueryEvaluatorAgent(llm)

    outcome = QueryOutcome(success=False, error_message="boom")
    result = await agent.evaluate(question="q", cypher="BAD", rationale="r", outcome=outcome)

    assert result.status == EvaluationStatus.ERROR_OR_EMPTY
    assert llm.captured_user_prompt is None
