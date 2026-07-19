"""Evaluation harness comparing the Multi-Agent GraphRAG pipeline (the
paper's "Agentic" configuration) against a linear-pass baseline ("Single")
on CypherBench, mirroring the structure of the paper's Table 1 experiment.

By default, correctness is scored deterministically via
`evaluation.matching.score_records_against_gold` against CypherBench's
`answer_json` ground truth (see that module's docstring for caveats).
Optionally, an LLM-as-a-judge (`evaluation.judge.LLMJudge`) can score the
natural language answer instead -- closer to the paper's own methodology,
though its exact judge prompt is undisclosed -- enabled by configuring
`Settings.evaluation.judge` and passing a built judge into `run_evaluation`.
"""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from multigraphrag.composition import build_pipeline, build_single_pass_runner
from multigraphrag.config import Settings
from multigraphrag.evaluation.cypherbench import (
    CypherBenchTask,
    load_domain_graph,
    load_tasks,
    populate_memgraph,
)
from multigraphrag.evaluation.judge import LLMJudge
from multigraphrag.evaluation.matching import score_records_against_gold
from multigraphrag.graph.memgraph_client import MemgraphClient
from multigraphrag.graph.models import GraphSchema
from multigraphrag.llm.call_log import CallLogger
from multigraphrag.workflow.pipeline import GraphRAGPipeline
from multigraphrag.workflow.single_pass import SinglePassRunner

logger = logging.getLogger(__name__)

EvalMode = Literal["single", "agentic"]
ALL_MODES: tuple[EvalMode, ...] = ("single", "agentic")


@dataclass
class EvalItemResult:
    qid: str
    domain: str
    mode: EvalMode
    question: str
    gold_cypher: str
    generated_cypher: str | None
    answer: str
    accepted: bool
    iterations: int
    similarity: float
    correct: bool
    error: str | None = None
    judge_reasoning: str | None = None

    def to_json(self) -> str:
        return json.dumps(self.__dict__, default=str)


@dataclass
class DomainModeSummary:
    domain: str
    mode: EvalMode
    total: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


@dataclass
class EvalReport:
    per_domain_mode: list[DomainModeSummary] = field(default_factory=list)
    items: list[EvalItemResult] = field(default_factory=list)

    def overall(self, mode: EvalMode) -> DomainModeSummary:
        matching = [s for s in self.per_domain_mode if s.mode == mode]
        return DomainModeSummary(
            domain="__overall__",
            mode=mode,
            total=sum(s.total for s in matching),
            correct=sum(s.correct for s in matching),
        )


async def _score(
    task: CypherBenchTask, records: list[dict], answer: str, judge: LLMJudge | None
) -> tuple[float, bool, str | None]:
    """Score one answer, always computing the deterministic similarity (cheap,
    useful for reference/debugging even when the judge is authoritative) and
    optionally overriding `correct` with the LLM judge's verdict.
    """
    similarity, heuristic_correct = score_records_against_gold(records, task.parsed_answer())
    if judge is None:
        return similarity, heuristic_correct, None
    verdict = await judge.judge(
        question=task.nl_question, answer=answer, gold_answer_rows=task.parsed_answer()
    )
    return similarity, verdict.correct, verdict.reasoning


async def _run_agentic_item(
    pipeline: GraphRAGPipeline,
    task: CypherBenchTask,
    domain: str,
    schema_text: str,
    judge: LLMJudge | None = None,
) -> EvalItemResult:
    try:
        result = await pipeline.ask(task.nl_question, schema_text=schema_text)
        similarity, correct, judge_reasoning = await _score(task, result.records, result.answer, judge)
        return EvalItemResult(
            qid=task.qid,
            domain=domain,
            mode="agentic",
            question=task.nl_question,
            gold_cypher=task.gold_cypher,
            generated_cypher=result.cypher,
            answer=result.answer,
            accepted=result.accepted,
            iterations=result.iterations,
            similarity=similarity,
            correct=correct,
            judge_reasoning=judge_reasoning,
        )
    except Exception as exc:  # noqa: BLE001 -- one bad task must not abort the whole eval run
        logger.exception("Agentic evaluation failed for qid=%s", task.qid)
        return EvalItemResult(
            qid=task.qid,
            domain=domain,
            mode="agentic",
            question=task.nl_question,
            gold_cypher=task.gold_cypher,
            generated_cypher=None,
            answer="",
            accepted=False,
            iterations=0,
            similarity=0.0,
            correct=False,
            error=str(exc),
        )


async def _run_single_pass_item(
    runner: SinglePassRunner,
    task: CypherBenchTask,
    domain: str,
    schema_text: str,
    judge: LLMJudge | None = None,
) -> EvalItemResult:
    try:
        result = await runner.ask(task.nl_question, schema_text)
        similarity, correct, judge_reasoning = await _score(task, result.records, result.answer, judge)
        return EvalItemResult(
            qid=task.qid,
            domain=domain,
            mode="single",
            question=task.nl_question,
            gold_cypher=task.gold_cypher,
            generated_cypher=result.cypher,
            answer=result.answer,
            accepted=result.accepted,
            iterations=result.iterations,
            similarity=similarity,
            correct=correct,
            judge_reasoning=judge_reasoning,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Single-pass evaluation failed for qid=%s", task.qid)
        return EvalItemResult(
            qid=task.qid,
            domain=domain,
            mode="single",
            question=task.nl_question,
            gold_cypher=task.gold_cypher,
            generated_cypher=None,
            answer="",
            accepted=False,
            iterations=0,
            similarity=0.0,
            correct=False,
            error=str(exc),
        )


async def _process_domain_mode(
    domain_tasks: list[CypherBenchTask],
    concurrency: int,
    item_runner: Callable[[CypherBenchTask], Awaitable[EvalItemResult]],
    *,
    report: EvalReport,
    summary: DomainModeSummary,
    trace_handle,
    on_item: Callable[[EvalItemResult], None] | None,
) -> None:
    """Run `item_runner` over every task in a domain/mode, up to `concurrency`
    at a time. Different questions against the same (already-loaded) graph are
    fully independent, so this is a plain semaphore-bounded fan-out -- the same
    pattern used for batching LLM calls in `llm/base.py`.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded(task: CypherBenchTask) -> None:
        async with semaphore:
            item = await item_runner(task)
        summary.total += 1
        summary.correct += int(item.correct)
        report.items.append(item)
        if trace_handle:
            trace_handle.write(item.to_json() + "\n")
        if on_item:
            on_item(item)

    await asyncio.gather(*(_bounded(task) for task in domain_tasks))


async def run_evaluation(
    settings: Settings,
    dataset_dir: Path,
    *,
    split: str,
    domains: list[str] | None = None,
    limit: int | None = None,
    modes: tuple[EvalMode, ...] = ALL_MODES,
    graph_variant: str = "simplekg_sampled",
    trace_path: Path | None = None,
    concurrency: int | None = None,
    judge: LLMJudge | None = None,
    call_logger: CallLogger | None = None,
    on_item: Callable[[EvalItemResult], None] | None = None,
) -> EvalReport:
    """Run the Single vs. Agentic comparison over a CypherBench split.

    Tasks are grouped by domain; for each domain the corresponding graph is
    (re)loaded into Memgraph once, its schema is introspected once, and then
    every requested mode is run over that domain's questions, up to
    `concurrency` questions in flight at a time (falls back to
    `settings.evaluation.concurrency` when not given). If `judge` is given,
    it scores every answer (in addition to the always-computed deterministic
    similarity); otherwise the deterministic heuristic alone determines
    correctness.
    """
    tasks = load_tasks(dataset_dir, split, domains=domains, limit=limit)
    if not tasks:
        raise ValueError("No CypherBench tasks matched the given split/domains/limit.")

    effective_concurrency = concurrency if concurrency is not None else settings.evaluation.concurrency

    tasks_by_domain: dict[str, list[CypherBenchTask]] = {}
    for task in tasks:
        tasks_by_domain.setdefault(task.graph, []).append(task)

    report = EvalReport()
    trace_handle = trace_path.open("w", encoding="utf-8") if trace_path else None

    try:
        async with MemgraphClient(settings.memgraph) as graph_client:
            await graph_client.verify_connectivity()

            for domain, domain_tasks in tasks_by_domain.items():
                logger.info("Loading domain graph '%s' (%d tasks)", domain, len(domain_tasks))
                domain_graph = load_domain_graph(dataset_dir, domain, graph_variant=graph_variant)
                await populate_memgraph(graph_client, domain_graph)
                schema: GraphSchema = await graph_client.build_schema()
                schema_text = schema.to_cypher_like_prompt()

                for mode in modes:
                    summary = DomainModeSummary(domain=domain, mode=mode)

                    if mode == "agentic":
                        async with build_pipeline(
                            settings, graph_client, call_logger=call_logger
                        ) as pipeline:
                            await _process_domain_mode(
                                domain_tasks,
                                effective_concurrency,
                                lambda task: _run_agentic_item(pipeline, task, domain, schema_text, judge),
                                report=report,
                                summary=summary,
                                trace_handle=trace_handle,
                                on_item=on_item,
                            )
                    else:
                        runner, clients = build_single_pass_runner(
                            settings, graph_client, call_logger=call_logger
                        )
                        try:
                            await _process_domain_mode(
                                domain_tasks,
                                effective_concurrency,
                                lambda task: _run_single_pass_item(runner, task, domain, schema_text, judge),
                                report=report,
                                summary=summary,
                                trace_handle=trace_handle,
                                on_item=on_item,
                            )
                        finally:
                            for client in clients:
                                await client.aclose()

                    report.per_domain_mode.append(summary)
    finally:
        if trace_handle:
            trace_handle.close()

    return report
