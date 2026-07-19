"""Feedback Aggregator agent (paper: Agent Roles and Responsibilities, item 7)."""

from multigraphrag.llm.base import LLMClient
from multigraphrag.prompts.system_prompts import FEEDBACK_AGGREGATOR
from multigraphrag.schemas import AggregatedFeedback, FixInstructions


class FeedbackAggregatorAgent:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def aggregate(
        self,
        *,
        question: str,
        cypher: str,
        evaluator_feedback: str | None,
        fix_instructions: FixInstructions | None,
    ) -> AggregatedFeedback:
        parts = [f"User question: {question}", f"\nQuery attempt being corrected:\n{cypher}"]
        if evaluator_feedback:
            parts.append(f"\nQuery Evaluator feedback:\n{evaluator_feedback}")
        if fix_instructions and fix_instructions.instructions:
            joined = "\n".join(f"- {i}" for i in fix_instructions.instructions)
            parts.append(f"\nVerification Module correction instructions:\n{joined}")

        return await self._llm.complete_structured(
            system_prompt=FEEDBACK_AGGREGATOR,
            user_prompt="\n".join(parts),
            response_model=AggregatedFeedback,
        )
