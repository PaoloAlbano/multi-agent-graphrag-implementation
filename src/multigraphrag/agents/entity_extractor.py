"""Named Entity Extractor agent (paper: Agent Roles and Responsibilities, item 4)."""

from multigraphrag.llm.base import LLMClient
from multigraphrag.prompts.system_prompts import NAMED_ENTITY_EXTRACTOR
from multigraphrag.schemas import ExtractedEntities


class NamedEntityExtractorAgent:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def extract(self, cypher: str) -> ExtractedEntities:
        return await self._llm.complete_structured(
            system_prompt=NAMED_ENTITY_EXTRACTOR,
            user_prompt=f"Cypher query:\n{cypher}",
            response_model=ExtractedEntities,
        )
