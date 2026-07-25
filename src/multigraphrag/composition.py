"""Composition root: the one place that reads `Settings` and wires concrete
adapters (vLLM clients, Memgraph) into the use cases (`GraphRAGPipeline`,
`SinglePassRunner`).

Every other module -- agents, the pipeline itself -- depends only on
abstractions (`LLMClient`, `MemgraphClient`) and receives its
collaborators via constructor injection. This module is where those
dependencies are actually constructed and handed out, so config/adapter
concerns stay out of business logic (Clean Architecture-style composition
root, `config.py` stays pure data, and swapping an adapter or a resolution
rule means editing only this file).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from multigraphrag.agents.entity_extractor import NamedEntityExtractorAgent
from multigraphrag.agents.evaluator import QueryEvaluatorAgent
from multigraphrag.agents.feedback_aggregator import FeedbackAggregatorAgent
from multigraphrag.agents.instructions_generator import InstructionsGeneratorAgent
from multigraphrag.agents.interpreter import InterpreterAgent
from multigraphrag.agents.query_generator import QueryGeneratorAgent
from multigraphrag.agents.verification import VerificationModule
from multigraphrag.config import LLMSettings, Settings
from multigraphrag.graph.memgraph_client import MemgraphClient
from multigraphrag.llm.base import LLMClient
from multigraphrag.llm.call_log import CallLogger
from multigraphrag.llm.factory import build_llm_client
from multigraphrag.workflow.pipeline import AgentBundle, GraphRAGPipeline
from multigraphrag.workflow.single_pass import SinglePassRunner

if TYPE_CHECKING:
    from multigraphrag.evaluation.judge import LLMJudge


def load_settings() -> Settings:
    """Read `Settings` from the environment/`.env`.

    Kept as a plain function (not a module-level singleton) so callers --
    tests in particular -- can construct fresh settings from different env
    vars without import-order issues.
    """
    return Settings()


def resolve_agent_llm(settings: Settings, agent_field: str) -> LLMSettings:
    """Return the effective LLM settings for a named agent, falling back to the default `llm` block."""
    override = getattr(settings.agent_models, agent_field)
    return override if override is not None else settings.llm


@dataclass
class _AgentClients:
    bundle: AgentBundle
    llm_clients: list[LLMClient]


def _build_agent_bundle(
    settings: Settings, graph_client: MemgraphClient, *, call_logger: CallLogger | None = None
) -> _AgentClients:
    clients: list[LLMClient] = []

    def make(agent_field: str) -> LLMClient:
        client = build_llm_client(
            resolve_agent_llm(settings, agent_field), agent_name=agent_field, call_logger=call_logger
        )
        clients.append(client)
        return client

    bundle = AgentBundle(
        query_generator=QueryGeneratorAgent(make("query_generator")),
        evaluator=QueryEvaluatorAgent(make("query_evaluator")),
        entity_extractor=NamedEntityExtractorAgent(make("entity_extractor")),
        verification=VerificationModule(make("verification_ranker"), graph_client, settings.workflow),
        instructions_generator=InstructionsGeneratorAgent(make("instructions_generator")),
        feedback_aggregator=FeedbackAggregatorAgent(make("feedback_aggregator")),
        interpreter=InterpreterAgent(make("interpreter")),
    )
    return _AgentClients(bundle=bundle, llm_clients=clients)


def build_pipeline(
    settings: Settings, graph_client: MemgraphClient, *, call_logger: CallLogger | None = None
) -> GraphRAGPipeline:
    """Wire the full Multi-Agent GraphRAG pipeline (the paper's "Agentic" configuration)."""
    agent_clients = _build_agent_bundle(settings, graph_client, call_logger=call_logger)
    return GraphRAGPipeline(
        agent_clients.bundle,
        graph_client,
        agent_clients.llm_clients,
        max_refinement_iterations=settings.workflow.max_refinement_iterations,
        max_query_result_rows=settings.workflow.max_query_result_rows,
    )


def build_single_pass_runner(
    settings: Settings, graph_client: MemgraphClient, *, call_logger: CallLogger | None = None
) -> tuple[SinglePassRunner, list[LLMClient]]:
    """Wire the linear-pass baseline (the paper's "Single" configuration)."""
    clients = [
        build_llm_client(
            resolve_agent_llm(settings, "query_generator"),
            agent_name="query_generator",
            call_logger=call_logger,
        ),
        build_llm_client(
            resolve_agent_llm(settings, "interpreter"), agent_name="interpreter", call_logger=call_logger
        ),
    ]
    runner = SinglePassRunner(
        QueryGeneratorAgent(clients[0]),
        InterpreterAgent(clients[1]),
        graph_client,
        max_attempts=settings.workflow.max_refinement_iterations,
        max_query_result_rows=settings.workflow.max_query_result_rows,
    )
    return runner, clients


def build_llm_judge(
    settings: Settings, *, call_logger: CallLogger | None = None
) -> tuple["LLMJudge", LLMClient]:
    """Wire the LLM-as-a-judge scorer.

    Uses `settings.evaluation.judge` if explicitly configured (e.g. a separate,
    dedicated judge model/endpoint); otherwise falls back to `settings.llm` --
    the same model/endpoint the agents under test use -- so `--use-judge` works
    out of the box without a second model to stand up, at the cost of the judge
    not being fully independent from the model being evaluated.

    Imports `LLMJudge` lazily: `evaluation`'s package `__init__` imports `runner.py`,
    which imports this module, so a top-level import here would be circular.
    """
    from multigraphrag.evaluation.judge import LLMJudge

    judge_settings = settings.evaluation.judge or settings.llm
    client = build_llm_client(judge_settings, agent_name="judge", call_logger=call_logger)
    return LLMJudge(client), client


def build_self_judge(
    settings: Settings, model: str, *, call_logger: CallLogger | None = None
) -> tuple["LLMJudge", LLMClient]:
    """Wire an LLM-as-a-judge scorer using `model` (a specific run's own model)
    rather than `settings.evaluation.judge` -- i.e. each model judges its own
    answers, ignoring any dedicated judge configured via
    `MULTIGRAPHRAG_EVALUATION__JUDGE__*`. Connection details (base URL, API
    key, timeouts, ...) are still taken from `settings.llm`; only the model
    name is overridden. Used by `cypherbench rejudge` to self-judge each
    results leaf with the model that actually produced it.
    """
    from multigraphrag.evaluation.judge import LLMJudge

    judge_settings = settings.llm.model_copy(update={"model": model})
    client = build_llm_client(judge_settings, agent_name="judge", call_logger=call_logger)
    return LLMJudge(client), client
