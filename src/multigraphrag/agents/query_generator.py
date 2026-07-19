"""Query Generator agent (paper: Agent Roles and Responsibilities, item 1)."""

from multigraphrag.llm.base import LLMClient
from multigraphrag.prompts.system_prompts import QUERY_GENERATOR
from multigraphrag.schemas import QueryGeneration


class QueryGeneratorAgent:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def generate(
        self,
        *,
        question: str,
        schema_text: str,
        previous_cypher: str | None = None,
        feedback: str | None = None,
    ) -> QueryGeneration:
        parts = [
            f"Graph schema:\n{schema_text}",
            f"\nUser question: {question}",
        ]
        if previous_cypher and feedback:
            parts.append(f"\nPrevious query attempt:\n{previous_cypher}")
            parts.append(f"\nAggregated correction feedback:\n{feedback}")
            parts.append("\nRevise the previous query according to the feedback above.")
        else:
            parts.append("\nGenerate the Cypher query to answer the question.")

        return await self._llm.complete_structured(
            system_prompt=QUERY_GENERATOR,
            user_prompt="\n".join(parts),
            response_model=QueryGeneration,
        )
