import pytest

from multigraphrag.graph.models import QueryOutcome
from multigraphrag.schemas import (
    AggregatedFeedback,
    EvaluationResult,
    EvaluationStatus,
    ExtractedEntities,
    InterpretedAnswer,
    QueryGeneration,
    VerificationReport,
)
from multigraphrag.workflow.pipeline import AgentBundle, GraphRAGPipeline


class _FakeQueryGenerator:
    async def generate(self, *, question, schema_text, previous_cypher, feedback):
        return QueryGeneration(cypher="MATCH (n) RETURN n", rationale="because")


class _FakeEvaluator:
    def __init__(self, statuses):
        self._statuses = list(statuses)

    async def evaluate(self, *, question, cypher, rationale, outcome):
        status = self._statuses.pop(0)
        return EvaluationResult(status=status, feedback=f"feedback for {status}")


class _FakeEntityExtractor:
    async def extract(self, cypher):
        return ExtractedEntities()


class _FakeVerification:
    def __init__(self, report):
        self._report = report

    async def verify(self, entities, *, question):
        return self._report


class _FailingInstructionsGenerator:
    async def generate(self, report, *, cypher):
        raise AssertionError("Instructions Generator must not be called when nothing is hallucinated")


class _FakeFeedbackAggregator:
    def __init__(self):
        self.calls = []

    async def aggregate(self, *, question, cypher, evaluator_feedback, fix_instructions):
        self.calls.append({"cypher": cypher, "fix_instructions": fix_instructions})
        return AggregatedFeedback(instruction="revise it")


class _FakeInterpreter:
    async def interpret(self, *, question, outcome):
        return InterpretedAnswer(answer="the answer")


class _FakeGraphClient:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)

    async def run_query(self, cypher, max_rows):
        return self._outcomes.pop(0)


@pytest.mark.asyncio
async def test_pipeline_skips_instructions_generator_when_nothing_hallucinated():
    """A legitimately empty result (error_or_empty) with a clean verification report
    must not reach the Instructions Generator -- there is nothing to "fix" -- and the
    Feedback Aggregator must be called with fix_instructions=None in that case.
    """
    empty_report = VerificationReport()
    assert empty_report.has_hallucinations is False

    agents = AgentBundle(
        query_generator=_FakeQueryGenerator(),
        evaluator=_FakeEvaluator([EvaluationStatus.ERROR_OR_EMPTY, EvaluationStatus.ACCEPT]),
        entity_extractor=_FakeEntityExtractor(),
        verification=_FakeVerification(empty_report),
        instructions_generator=_FailingInstructionsGenerator(),
        feedback_aggregator=_FakeFeedbackAggregator(),
        interpreter=_FakeInterpreter(),
    )
    graph_client = _FakeGraphClient(
        [
            QueryOutcome(success=True, records=[]),
            QueryOutcome(success=True, records=[{"x": 1}]),
        ]
    )

    pipeline = GraphRAGPipeline(
        agents, graph_client, llm_clients=[], max_refinement_iterations=4, max_query_result_rows=200
    )

    result = await pipeline.ask("some question", schema_text="schema")

    assert result.accepted is True
    assert agents.feedback_aggregator.calls[0]["fix_instructions"] is None
    assert agents.feedback_aggregator.calls[0]["cypher"] == "MATCH (n) RETURN n"
