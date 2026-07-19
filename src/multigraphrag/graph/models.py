"""Data models shared by the graph client and the agents that consume it."""

from pydantic import BaseModel, Field


class QueryOutcome(BaseModel):
    """Result of executing a single Cypher query against the graph database.

    Mirrors the three outcomes the paper's Query Evaluator distinguishes:
    successful results, an empty result set, or a runtime error.
    """

    success: bool
    records: list[dict] = Field(default_factory=list)
    error_message: str | None = None
    truncated: bool = False

    @property
    def is_empty(self) -> bool:
        return self.success and len(self.records) == 0


class PropertySample(BaseModel):
    key: str
    example_values: list[str] = Field(default_factory=list)


class NodeSchema(BaseModel):
    label: str
    properties: list[PropertySample] = Field(default_factory=list)


class RelationshipPattern(BaseModel):
    start_label: str
    rel_type: str
    end_label: str


class GraphSchema(BaseModel):
    """Aggregated schema description used both for LLM prompting and for
    programmatic existence checks in the Verification Module.
    """

    nodes: list[NodeSchema] = Field(default_factory=list)
    relationships: list[RelationshipPattern] = Field(default_factory=list)

    def node_labels(self) -> list[str]:
        return [n.label for n in self.nodes]

    def relationship_types(self) -> list[str]:
        seen: list[str] = []
        for rel in self.relationships:
            if rel.rel_type not in seen:
                seen.append(rel.rel_type)
        return seen

    def to_cypher_like_prompt(self) -> str:
        """Render the schema in a format resembling Cypher syntax.

        The paper found that presenting the schema close to actual Cypher
        syntax (rather than e.g. a generic JSON dump) improves generation
        quality and token-level alignment with expected outputs (see
        "Incorporating Graph Schema into LLM Prompts").
        """
        lines: list[str] = []

        lines.append("# Node types, properties and sampled example values")
        for node in self.nodes:
            lines.append(f"Node Type: {node.label}")
            lines.append("Properties:")
            for prop in node.properties:
                examples = ", ".join(f'"{v}"' for v in prop.example_values[:3])
                lines.append(f"  .{prop.key}: {examples}")
            lines.append("")

        lines.append("# Relationship types and their valid endpoint patterns")
        by_type: dict[str, list[RelationshipPattern]] = {}
        for rel in self.relationships:
            by_type.setdefault(rel.rel_type, []).append(rel)
        for rel_type, patterns in by_type.items():
            lines.append(f"Type: {rel_type}")
            for pattern in patterns:
                lines.append(f"  - (:{pattern.start_label})-[:{rel_type}]->(:{pattern.end_label})")
            lines.append("")

        return "\n".join(lines)
