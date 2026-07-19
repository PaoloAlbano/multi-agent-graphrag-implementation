"""Linear-pass baseline: Query Generator -> Executor -> Interpreter, no
Evaluator / Verification Module / Feedback Aggregator involved.

Mirrors the paper's "Single" baseline configuration used in Table 1: blind
retries on execution error/empty result (up to `max_attempts`), with no
semantic evaluation or entity verification loop. Comparing this against
`GraphRAGPipeline` (the "Agentic" configuration) on the same questions/models
is what produces a Single-vs-Agentic accuracy table like the paper's.
"""

from dataclasses import dataclass

from multigraphrag.agents.interpreter import InterpreterAgent
from multigraphrag.agents.query_generator import QueryGeneratorAgent
from multigraphrag.graph.memgraph_client import MemgraphClient
from multigraphrag.graph.models import QueryOutcome


@dataclass
class SinglePassResult:
    question: str
    answer: str
    cypher: str | None
    accepted: bool
    iterations: int
    records: list[dict]


class SinglePassRunner:
    def __init__(
        self,
        query_generator: QueryGeneratorAgent,
        interpreter: InterpreterAgent,
        graph_client: MemgraphClient,
        *,
        max_attempts: int = 4,
        max_query_result_rows: int = 200,
    ) -> None:
        self._query_generator = query_generator
        self._interpreter = interpreter
        self._graph_client = graph_client
        self._max_attempts = max_attempts
        self._max_query_result_rows = max_query_result_rows

    async def ask(self, question: str, schema_text: str) -> SinglePassResult:
        cypher: str | None = None
        outcome: QueryOutcome | None = None

        for attempt in range(1, self._max_attempts + 1):
            feedback = None
            if outcome is not None and not outcome.success:
                feedback = f"Previous attempt failed with: {outcome.error_message}. Try a different query."
            elif outcome is not None and outcome.is_empty:
                feedback = "Previous attempt returned no results. Try a different query."

            generation = await self._query_generator.generate(
                question=question,
                schema_text=schema_text,
                previous_cypher=cypher,
                feedback=feedback,
            )
            cypher = generation.cypher
            outcome = await self._graph_client.run_query(cypher, max_rows=self._max_query_result_rows)

            if outcome.success and not outcome.is_empty:
                break
            if attempt == self._max_attempts:
                break

        outcome = outcome or QueryOutcome(success=False, error_message="No query was executed.")
        interpreted = await self._interpreter.interpret(question=question, outcome=outcome)
        return SinglePassResult(
            question=question,
            answer=interpreted.answer,
            cypher=cypher,
            accepted=outcome.success and not outcome.is_empty,
            iterations=attempt,
            records=outcome.records,
        )
