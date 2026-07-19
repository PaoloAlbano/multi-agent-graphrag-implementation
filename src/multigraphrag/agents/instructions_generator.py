"""Instructions Generator agent (paper: Agent Roles and Responsibilities, item 6)."""

from multigraphrag.llm.base import LLMClient
from multigraphrag.prompts.system_prompts import INSTRUCTIONS_GENERATOR
from multigraphrag.schemas import FixInstructions, VerificationReport


def _render_report(report: VerificationReport) -> str:
    lines: list[str] = []

    invalid_labels = [v.label for v in report.labels if not v.exists]
    if invalid_labels:
        lines.append(f"Invalid node labels (not present in the graph): {invalid_labels}")

    for verdict in report.property_values:
        if verdict.exists:
            continue
        lines.append(
            f'Property "{verdict.label}.{verdict.property}" used value "{verdict.value}", '
            f"which does not exist. Fuzzy-similar candidates: {verdict.fuzzy_suggestions}. "
            f"LLM-ranked best replacement: {verdict.llm_ranked_suggestion!r}."
        )

    invalid_rels = [v.pattern for v in report.relationships if not v.exists]
    if invalid_rels:
        lines.append(f"Invalid relationship patterns (not present in the graph schema): {invalid_rels}")

    return "\n".join(lines) if lines else "No hallucinated entities detected."


class InstructionsGeneratorAgent:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def generate(self, report: VerificationReport, *, cypher: str) -> FixInstructions:
        user_prompt = f"Query being corrected:\n{cypher}\n\nVerification report:\n{_render_report(report)}"
        return await self._llm.complete_structured(
            system_prompt=INSTRUCTIONS_GENERATOR,
            user_prompt=user_prompt,
            response_model=FixInstructions,
        )
