"""Query Evaluator agent (paper: Agent Roles and Responsibilities, item 3)."""

import json

from multigraphrag.graph.models import QueryOutcome
from multigraphrag.llm.base import LLMClient
from multigraphrag.prompts.system_prompts import QUERY_EVALUATOR
from multigraphrag.schemas import EvaluationResult, EvaluationStatus


class QueryEvaluatorAgent:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def evaluate(
        self,
        *,
        question: str,
        cypher: str,
        rationale: str | None,
        outcome: QueryOutcome,
    ) -> EvaluationResult:
        if not outcome.success:
            # The database itself already tells us this is Error-or-Empty; skip the
            # LLM call and route straight to verification, saving a round trip.
            return EvaluationResult(
                status=EvaluationStatus.ERROR_OR_EMPTY,
                feedback=f"Query execution failed: {outcome.error_message}",
            )
        if outcome.is_empty:
            return EvaluationResult(
                status=EvaluationStatus.ERROR_OR_EMPTY,
                feedback="Query executed successfully but returned no results.",
            )

        user_prompt = (
            f"User question: {question}\n\n"
            f"Query Generator's natural language explanation of its own query:\n"
            f"{rationale or '(none provided)'}\n\n"
            f"Executed Cypher query:\n{cypher}\n\n"
            f"Query result (JSON, possibly truncated):\n{json.dumps(outcome.records, default=str)}\n"
        )
        return await self._llm.complete_structured(
            system_prompt=QUERY_EVALUATOR,
            user_prompt=user_prompt,
            response_model=EvaluationResult,
        )
