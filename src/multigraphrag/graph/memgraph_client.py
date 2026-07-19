"""Async Memgraph client built on the standard `neo4j` Bolt driver.

Memgraph implements the Bolt protocol and is wire-compatible with the
official Neo4j drivers, so no vendor-specific client is required. Only plain
Cypher (no APOC/vendor procedures) is used for schema introspection, keeping
this module portable to other LPG backends such as Neo4j if needed later.
"""

import logging

import neo4j
from neo4j.exceptions import Neo4jError

from multigraphrag.config import MemgraphSettings
from multigraphrag.graph.models import (
    GraphSchema,
    NodeSchema,
    PropertySample,
    QueryOutcome,
    RelationshipPattern,
)

logger = logging.getLogger(__name__)


class MemgraphClient:
    """Thin async wrapper around a Bolt driver pointed at a Memgraph instance."""

    def __init__(self, settings: MemgraphSettings, *, driver: neo4j.AsyncDriver | None = None) -> None:
        self._settings = settings
        self._owns_driver = driver is None
        self._driver = driver or neo4j.AsyncGraphDatabase.driver(
            settings.uri,
            auth=neo4j.basic_auth(settings.username, settings.password),
            encrypted=settings.encrypted,
            connection_timeout=settings.connection_timeout,
        )

    async def aclose(self) -> None:
        if self._owns_driver:
            await self._driver.close()

    async def __aenter__(self) -> "MemgraphClient":
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    async def verify_connectivity(self) -> None:
        await self._driver.verify_connectivity()

    async def run_query(
        self, cypher: str, parameters: dict | None = None, *, max_rows: int = 200
    ) -> QueryOutcome:
        """Execute an arbitrary (LLM-generated) Cypher query.

        Errors are captured rather than raised: the paper's workflow treats a
        runtime error as one of three evaluator outcomes (Accept / Incorrect /
        Error-or-Empty), so callers need the message, not an exception.
        """
        if not cypher or not cypher.strip():
            # The neo4j driver rejects an empty query with a client-side
            # ValueError *before* any server round-trip, which is not a
            # Neo4jError -- left uncaught, it crashed the entire ask() call
            # (observed: a hallucinated empty Cypher string from the Query
            # Generator killed the whole question instead of triggering the
            # self-correction loop). Guard explicitly so this becomes a normal
            # Error-or-Empty outcome the Evaluator/Verification loop can react to.
            return QueryOutcome(
                success=False, error_message="Query Generator produced an empty Cypher query."
            )
        try:
            async with self._driver.session(database=self._settings.database) as session:
                result = await session.run(cypher, parameters or {})
                records = [record.data() async for record in result]
        except Neo4jError as exc:
            logger.debug("Cypher execution failed: %s", exc)
            return QueryOutcome(success=False, error_message=str(exc))
        except ValueError as exc:
            # Other client-side validation errors from the driver (also not
            # Neo4jError subclasses) -- same reasoning as the empty-query
            # guard above: surface as a recoverable outcome, not a crash.
            logger.debug("Cypher execution rejected by driver: %s", exc)
            return QueryOutcome(success=False, error_message=str(exc))

        truncated = len(records) > max_rows
        return QueryOutcome(success=True, records=records[:max_rows], truncated=truncated)

    # -- Schema introspection -------------------------------------------------
    # Deliberately implemented with plain Cypher (no APOC) so it also works
    # unmodified against Neo4j-compatible backends.

    async def get_node_labels(self) -> list[str]:
        outcome = await self.run_query(
            "MATCH (n) UNWIND labels(n) AS label RETURN DISTINCT label ORDER BY label"
        )
        return [row["label"] for row in outcome.records]

    async def get_properties_for_label(self, label: str, *, sample_size: int = 200) -> list[str]:
        outcome = await self.run_query(
            f"MATCH (n:`{label}`) WITH n LIMIT $sample_size UNWIND keys(n) AS k RETURN DISTINCT k ORDER BY k",
            {"sample_size": sample_size},
        )
        # "_"-prefixed keys are internal bookkeeping (e.g. CypherBench's
        # `_eid` join key used to wire up relations at load time) and are not
        # meaningful query targets, so they are hidden from the schema prompt.
        return [row["k"] for row in outcome.records if not row["k"].startswith("_")]

    async def sample_property_values(self, label: str, prop: str, *, limit: int = 3) -> list[str]:
        outcome = await self.run_query(
            f"MATCH (n:`{label}`) WHERE n.`{prop}` IS NOT NULL RETURN DISTINCT n.`{prop}` AS v LIMIT $limit",
            {"limit": limit},
        )
        return [str(row["v"]) for row in outcome.records]

    async def get_all_property_values(self, label: str, prop: str, *, limit: int = 5000) -> list[str]:
        """Full (bounded) set of distinct values, used for fuzzy-matching hallucinated literals."""
        outcome = await self.run_query(
            f"MATCH (n:`{label}`) WHERE n.`{prop}` IS NOT NULL RETURN DISTINCT n.`{prop}` AS v LIMIT $limit",
            {"limit": limit},
        )
        return [str(row["v"]) for row in outcome.records]

    async def get_relationship_patterns(self, *, sample_size: int = 5000) -> list[RelationshipPattern]:
        outcome = await self.run_query(
            "MATCH (a)-[r]->(b) WITH a, r, b LIMIT $sample_size "
            "RETURN DISTINCT labels(a) AS start_labels, type(r) AS rel_type, labels(b) AS end_labels",
            {"sample_size": sample_size},
        )
        patterns: list[RelationshipPattern] = []
        seen: set[tuple[str, str, str]] = set()
        for row in outcome.records:
            for start_label in row["start_labels"] or ["*"]:
                for end_label in row["end_labels"] or ["*"]:
                    key = (start_label, row["rel_type"], end_label)
                    if key not in seen:
                        seen.add(key)
                        patterns.append(
                            RelationshipPattern(
                                start_label=start_label, rel_type=row["rel_type"], end_label=end_label
                            )
                        )
        return patterns

    async def label_exists(self, label: str) -> bool:
        return label in await self.get_node_labels()

    async def relationship_type_exists(self, rel_type: str) -> bool:
        patterns = await self.get_relationship_patterns()
        return any(p.rel_type == rel_type for p in patterns)

    async def relationship_pattern_exists(self, start_label: str, rel_type: str, end_label: str) -> bool:
        patterns = await self.get_relationship_patterns()
        return any(
            p.start_label == start_label and p.rel_type == rel_type and p.end_label == end_label
            for p in patterns
        )

    async def property_value_exists(self, label: str, prop: str, value: str) -> bool:
        outcome = await self.run_query(
            f"MATCH (n:`{label}`) WHERE toLower(toString(n.`{prop}`)) = toLower($value) "
            "RETURN count(n) > 0 AS found",
            {"value": value},
        )
        return bool(outcome.records and outcome.records[0]["found"])

    async def build_schema(self, *, values_per_property: int = 1) -> GraphSchema:
        """Build the full `GraphSchema` used both for prompting and verification."""
        labels = await self.get_node_labels()
        nodes: list[NodeSchema] = []
        for label in labels:
            properties = await self.get_properties_for_label(label)
            prop_samples = [
                PropertySample(
                    key=prop,
                    example_values=await self.sample_property_values(label, prop, limit=values_per_property),
                )
                for prop in properties
            ]
            nodes.append(NodeSchema(label=label, properties=prop_samples))
        relationships = await self.get_relationship_patterns()
        return GraphSchema(nodes=nodes, relationships=relationships)
