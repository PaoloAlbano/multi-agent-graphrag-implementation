"""The Multi-Agent GraphRAG use case: ask a question, get a Cypher query and
a natural language answer.

`GraphRAGPipeline.ask` is a direct, linear translation of the paper's
Algorithm 1 ("Cypher query refinement with semantic validation and named
entities verification"): generate -> execute -> evaluate, then either accept,
correct via the Evaluator's feedback alone (`incorrect`), or correct via the
full entity-verification path (`error_or_empty`), looping until accepted or
`max_refinement_iterations` is reached, then always interpreting the final
result. No graph/DAG runtime is involved -- the control flow is a bounded
loop with two branches, which a plain `while` expresses more directly than a
state-graph library would.

`GraphRAGPipeline` is a pure orchestrator: it only knows about the `AgentBundle`
it drives, the `MemgraphClient` it needs for query execution and schema
introspection, and the list of `LLMClient`s it must close on shutdown. It has
no knowledge of `Settings` or of how those collaborators were built -- that
wiring lives in the composition root (`composition.py`), which is the only
place that reads config and constructs concrete adapters. This keeps the use
case testable with fakes and free of any config/adapter coupling.
"""

from dataclasses import dataclass

from multigraphrag.agents.entity_extractor import NamedEntityExtractorAgent
from multigraphrag.agents.evaluator import QueryEvaluatorAgent
from multigraphrag.agents.feedback_aggregator import FeedbackAggregatorAgent
from multigraphrag.agents.instructions_generator import InstructionsGeneratorAgent
from multigraphrag.agents.interpreter import InterpreterAgent
from multigraphrag.agents.query_generator import QueryGeneratorAgent
from multigraphrag.agents.verification import VerificationModule
from multigraphrag.graph.memgraph_client import MemgraphClient
from multigraphrag.graph.models import GraphSchema
from multigraphrag.llm.base import LLMClient
from multigraphrag.schemas import EvaluationStatus


@dataclass
class AgentBundle:
    """All seven agents wired for a single pipeline instance.

    Each agent may hold a different `LLMClient` (built from its own
    `AgentModelSettings` entry in `config.py`), which is how per-agent model
    routing (e.g. a stronger model for generation, a cheaper one for
    extraction) is achieved without touching agent or pipeline code.
    """

    query_generator: QueryGeneratorAgent
    evaluator: QueryEvaluatorAgent
    entity_extractor: NamedEntityExtractorAgent
    verification: VerificationModule
    instructions_generator: InstructionsGeneratorAgent
    feedback_aggregator: FeedbackAggregatorAgent
    interpreter: InterpreterAgent


@dataclass
class PipelineResult:
    question: str
    answer: str
    cypher: str | None
    accepted: bool
    iterations: int
    records: list[dict]


class GraphRAGPipeline:
    """Runs the Algorithm-1 self-correction loop for one graph database."""

    def __init__(
        self,
        agents: AgentBundle,
        graph_client: MemgraphClient,
        llm_clients: list[LLMClient],
        *,
        max_refinement_iterations: int,
        max_query_result_rows: int,
    ) -> None:
        self._agents = agents
        self._graph_client = graph_client
        self._llm_clients = llm_clients
        self._max_refinement_iterations = max_refinement_iterations
        self._max_query_result_rows = max_query_result_rows
        self._schema_text: str | None = None

    async def aclose(self) -> None:
        for client in self._llm_clients:
            await client.aclose()

    async def __aenter__(self) -> "GraphRAGPipeline":
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    async def _get_schema_text(self, *, refresh: bool = False) -> str:
        if self._schema_text is None or refresh:
            schema: GraphSchema = await self._graph_client.build_schema()
            self._schema_text = schema.to_cypher_like_prompt()
        return self._schema_text

    async def ask(
        self,
        question: str,
        *,
        schema_text: str | None = None,
        refresh_schema: bool = False,
    ) -> PipelineResult:
        """Answer one question. If `schema_text` is provided (e.g. precomputed
        once by an evaluation harness iterating many questions over the same
        graph), it is used as-is and also cached, skipping re-introspection.
        """
        if schema_text is not None:
            self._schema_text = schema_text
        else:
            schema_text = await self._get_schema_text(refresh=refresh_schema)

        cypher: str | None = None
        feedback_instruction: str | None = None
        iteration = 1

        while True:
            generation = await self._agents.query_generator.generate(
                question=question,
                schema_text=schema_text,
                previous_cypher=cypher,
                feedback=feedback_instruction,
            )
            cypher = generation.cypher
            outcome = await self._graph_client.run_query(cypher, max_rows=self._max_query_result_rows)
            evaluation = await self._agents.evaluator.evaluate(
                question=question, cypher=cypher, rationale=generation.rationale, outcome=outcome
            )

            if evaluation.status == EvaluationStatus.ACCEPT or iteration >= self._max_refinement_iterations:
                break

            if evaluation.status == EvaluationStatus.INCORRECT:
                fix_instructions = None
            else:  # ERROR_OR_EMPTY: verify entities against the live graph first.
                entities = await self._agents.entity_extractor.extract(cypher)
                report = await self._agents.verification.verify(entities, question=question)
                # Only ask the Instructions Generator to act when the report actually
                # flags something hallucinated -- an empty/error result can also be a
                # legitimate empty answer, in which case there is nothing to "fix" here
                # and the Evaluator's own feedback is the only useful signal.
                fix_instructions = (
                    await self._agents.instructions_generator.generate(report, cypher=cypher)
                    if report.has_hallucinations
                    else None
                )

            aggregated = await self._agents.feedback_aggregator.aggregate(
                question=question,
                cypher=cypher,
                evaluator_feedback=evaluation.feedback,
                fix_instructions=fix_instructions,
            )
            feedback_instruction = aggregated.instruction
            iteration += 1

        interpreted = await self._agents.interpreter.interpret(question=question, outcome=outcome)
        return PipelineResult(
            question=question,
            answer=interpreted.answer,
            cypher=cypher,
            accepted=evaluation.status == EvaluationStatus.ACCEPT,
            iterations=iteration,
            records=outcome.records,
        )
