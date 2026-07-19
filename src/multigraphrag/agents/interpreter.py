"""Interpreter agent (paper: Agent Roles and Responsibilities, item 8)."""

import json

from multigraphrag.graph.models import QueryOutcome
from multigraphrag.llm.base import LLMClient
from multigraphrag.prompts.system_prompts import INTERPRETER
from multigraphrag.schemas import InterpretedAnswer


class InterpreterAgent:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def interpret(self, *, question: str, outcome: QueryOutcome) -> InterpretedAnswer:
        user_prompt = (
            f"User question: {question}\n\n"
            f"Query result (JSON, possibly truncated):\n{json.dumps(outcome.records, default=str)}\n"
        )
        return await self._llm.complete_structured(
            system_prompt=INTERPRETER,
            user_prompt=user_prompt,
            response_model=InterpretedAnswer,
        )
