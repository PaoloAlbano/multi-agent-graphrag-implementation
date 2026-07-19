"""Verification Module (paper: Agent Roles and Responsibilities, item 5).

Combines programmatic existence checks against the live graph with a
two-step hallucination recovery: (1) normalized-Levenshtein candidate
retrieval (`verification.fuzzy_match`), then (2) LLM-based semantic ranking
of those candidates. The LLM ranking calls for every hallucinated entity are
independent of one another, so they are fanned out concurrently via
`LLMClient.complete_structured_many`, bounded by the configured
`max_concurrency`, instead of being awaited one at a time.
"""

import logging
import re

from multigraphrag.config import WorkflowSettings
from multigraphrag.graph.memgraph_client import MemgraphClient
from multigraphrag.llm.base import LLMClient, Prompt
from multigraphrag.prompts.system_prompts import VERIFICATION_RANKER
from multigraphrag.schemas import (
    ExtractedEntities,
    LabelVerdict,
    NodePropertyValue,
    PropertyValueVerdict,
    RankedSuggestion,
    RelationshipVerdict,
    VerificationReport,
)
from multigraphrag.verification.fuzzy_match import suggest_candidates

logger = logging.getLogger(__name__)

# Node patterns may or may not carry a variable name before the colon (the
# extractor is asked for "(:Label)-[:REL]->(:Label)" but LLMs sometimes still
# echo back the query's own variable names, e.g. "(n:Label)-[r:REL]->(m:Label)").
# Both forms must parse to the same (start_label, rel_type, end_label) triple.
_RELATIONSHIP_PATTERN_RE = re.compile(
    r"\(\s*\w*\s*:\s*([A-Za-z0-9_]+)\s*\)\s*-\s*\[\s*\w*\s*:\s*([A-Za-z0-9_]+)\s*\]\s*-\s*>?\s*"
    r"\(\s*\w*\s*:\s*([A-Za-z0-9_]+)\s*\)"
)


def parse_relationship_pattern(pattern: str) -> tuple[str, str, str] | None:
    """Parse "(:LabelA)-[:REL]->(:LabelB)" (directed/undirected, variable names optional)."""
    match = _RELATIONSHIP_PATTERN_RE.search(pattern)
    if not match:
        return None
    start_label, rel_type, end_label = match.groups()
    return start_label, rel_type, end_label


class VerificationModule:
    def __init__(
        self,
        llm: LLMClient,
        graph_client: MemgraphClient,
        workflow_settings: WorkflowSettings,
    ) -> None:
        self._llm = llm
        self._graph = graph_client
        self._settings = workflow_settings

    async def verify(self, entities: ExtractedEntities, *, question: str) -> VerificationReport:
        labels = await self._verify_labels(entities.node_labels)
        relationships = await self._verify_relationships(entities.pairwise_relationships)
        property_values = await self._verify_property_values(entities.node_property_values, question=question)
        return VerificationReport(labels=labels, relationships=relationships, property_values=property_values)

    async def _verify_labels(self, labels: list[str]) -> list[LabelVerdict]:
        return [LabelVerdict(label=label, exists=await self._graph.label_exists(label)) for label in labels]

    async def _verify_relationships(self, patterns: list[str]) -> list[RelationshipVerdict]:
        verdicts: list[RelationshipVerdict] = []
        for pattern in patterns:
            parsed = parse_relationship_pattern(pattern)
            if parsed is None:
                # Cannot confirm OR deny existence from an unparseable pattern -- treating
                # it as "does not exist" would inject a false hallucination signal into the
                # correction loop. Skip it instead and leave a trace of why.
                logger.warning(
                    "Could not parse relationship pattern %r extracted by the Named Entity "
                    "Extractor (expected '(:Label)-[:REL]->(:Label)'); skipping its "
                    "verification rather than reporting a false hallucination.",
                    pattern,
                )
                continue
            start_label, rel_type, end_label = parsed
            exists = await self._graph.relationship_pattern_exists(start_label, rel_type, end_label)
            verdicts.append(RelationshipVerdict(pattern=pattern, exists=exists))
        return verdicts

    async def _verify_property_values(
        self, node_property_values: list[NodePropertyValue], *, question: str
    ) -> list[PropertyValueVerdict]:
        # First pass: cheap, sequential existence checks straight against the DB.
        pending: list[tuple[NodePropertyValue, list[str]]] = []
        verdicts: list[PropertyValueVerdict] = []
        for npv in node_property_values:
            exists = await self._graph.property_value_exists(npv.label, npv.property, npv.value)
            if exists:
                verdicts.append(
                    PropertyValueVerdict(label=npv.label, property=npv.property, value=npv.value, exists=True)
                )
            else:
                known_values = await self._graph.get_all_property_values(npv.label, npv.property)
                candidates = suggest_candidates(
                    npv.value,
                    known_values,
                    top_k=self._settings.levenshtein_top_k,
                    min_score=self._settings.levenshtein_min_score,
                )
                pending.append((npv, [c.value for c in candidates]))

        if not pending:
            return verdicts

        # Second pass: semantic re-ranking of fuzzy candidates. Independent
        # per-entity, so batch them through the concurrency-bounded LLM helper.
        # Entries with no fuzzy candidates at all are skipped (nothing to rank).
        rankable = [(npv, candidates) for npv, candidates in pending if candidates]
        prompts = [
            Prompt(
                system_prompt=VERIFICATION_RANKER,
                user_prompt=(
                    f"User question: {question}\n"
                    f"Node label: {npv.label}\n"
                    f"Property: {npv.property}\n"
                    f'Hallucinated/mismatched value used in the query: "{npv.value}"\n'
                    f"Candidate existing values: {candidates}\n"
                ),
            )
            for npv, candidates in rankable
        ]
        rankings = await self._llm.complete_structured_many(prompts, RankedSuggestion) if prompts else []
        ranking_by_key = {
            (npv.label, npv.property, npv.value): ranking.best_candidate
            for (npv, _), ranking in zip(rankable, rankings, strict=True)
        }

        for npv, candidates in pending:
            verdicts.append(
                PropertyValueVerdict(
                    label=npv.label,
                    property=npv.property,
                    value=npv.value,
                    exists=False,
                    fuzzy_suggestions=candidates,
                    llm_ranked_suggestion=ranking_by_key.get((npv.label, npv.property, npv.value)),
                )
            )
        return verdicts
