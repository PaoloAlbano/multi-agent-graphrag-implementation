"""Structured input/output models exchanged between agents.

Each model is passed straight to `LLMClient.complete_structured` as the
`response_model`, so its docstrings/field descriptions double as part of the
instructions the model sees via the generated JSON schema.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class EvaluationStatus(StrEnum):
    """The three discrete grades the Query Evaluator can assign (paper, Query Evaluator)."""

    ACCEPT = "accept"
    INCORRECT = "incorrect"
    ERROR_OR_EMPTY = "error_or_empty"


class QueryGeneration(BaseModel):
    """Output of the Query Generator agent."""

    cypher: str = Field(description="A single executable Cypher query answering the user question.")
    rationale: str = Field(description="Brief explanation of the query logic, in natural language.")


class EvaluationResult(BaseModel):
    """Output of the Query Evaluator agent."""

    status: EvaluationStatus
    feedback: str = Field(
        description=(
            "Structured feedback explaining the grade: semantic/logical issues, "
            "or why the result is invalid/empty."
        )
    )


class NodePropertyValue(BaseModel):
    """A (label, property, value) triple extracted from a generated query."""

    label: str
    property: str
    value: str


class ExtractedEntities(BaseModel):
    """Output of the Named Entity Extractor agent.

    Decomposes a Cypher query into the schema elements susceptible to
    hallucination: node labels, property/value literals, and relationship
    patterns, mirroring Appendix B's `Query Entities Extractor` example.
    """

    node_labels: list[str] = Field(default_factory=list)
    node_property_values: list[NodePropertyValue] = Field(default_factory=list)
    pairwise_relationships: list[str] = Field(
        default_factory=list,
        description='Cypher-like patterns, e.g. "(:Character)-[:hasFather]->(:Character)".',
    )


class LabelVerdict(BaseModel):
    label: str
    exists: bool


class PropertyValueVerdict(BaseModel):
    label: str
    property: str
    value: str
    exists: bool
    fuzzy_suggestions: list[str] = Field(default_factory=list)
    llm_ranked_suggestion: str | None = None


class RelationshipVerdict(BaseModel):
    pattern: str
    exists: bool


class VerificationReport(BaseModel):
    """Output of the Verification Module: existence checks + recovery candidates."""

    labels: list[LabelVerdict] = Field(default_factory=list)
    property_values: list[PropertyValueVerdict] = Field(default_factory=list)
    relationships: list[RelationshipVerdict] = Field(default_factory=list)

    @property
    def has_hallucinations(self) -> bool:
        return (
            any(not v.exists for v in self.labels)
            or any(not v.exists for v in self.property_values)
            or any(not v.exists for v in self.relationships)
        )


class RankedSuggestion(BaseModel):
    """Structured output of the LLM semantic-ranking step over fuzzy candidates."""

    best_candidate: str = Field(description="The single most contextually appropriate replacement value.")


class FixInstructions(BaseModel):
    """Output of the Instructions Generator agent."""

    instructions: list[str] = Field(
        default_factory=list,
        description="Concise, per-entity correction instructions for the Query Generator.",
    )


class AggregatedFeedback(BaseModel):
    """Output of the Feedback Aggregator agent: the single instruction handed
    back to the Query Generator for the next refinement attempt.
    """

    instruction: str


class InterpretedAnswer(BaseModel):
    """Output of the Interpreter agent."""

    answer: str = Field(description="Concise, domain-relevant natural language answer to the user question.")
