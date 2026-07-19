"""Command-line entry point for the Multi-Agent GraphRAG pipeline."""

import asyncio
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from multigraphrag.composition import build_llm_judge, build_pipeline, load_settings
from multigraphrag.config import Settings
from multigraphrag.evaluation.cypherbench import CYPHERBENCH_DOMAINS, download_cypherbench
from multigraphrag.evaluation.runner import ALL_MODES, EvalMode, run_evaluation
from multigraphrag.graph.memgraph_client import MemgraphClient
from multigraphrag.llm.call_log import CallLogger

app = typer.Typer(add_completion=False, help="Multi-Agent GraphRAG: text-to-Cypher over a Memgraph LPG.")
cypherbench_app = typer.Typer(
    add_completion=False, help="Download and evaluate against the CypherBench benchmark."
)
app.add_typer(cypherbench_app, name="cypherbench")
console = Console()


async def _ask(question: str, settings: Settings) -> None:
    async with MemgraphClient(settings.memgraph) as graph_client:
        await graph_client.verify_connectivity()
        async with build_pipeline(settings, graph_client) as pipeline:
            result = await pipeline.ask(question)

    console.print(Markdown(f"**Question:** {result.question}"))
    if result.cypher:
        console.print(
            Markdown(
                f"**Cypher ({'accepted' if result.accepted else 'best effort'}):**\n```cypher\n{result.cypher}\n```"
            )
        )
    console.print(Markdown(f"**Answer:** {result.answer}"))
    console.print(f"[dim]iterations: {result.iterations}[/dim]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Natural language question to answer over the graph."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Answer a single natural language question against the configured Memgraph instance."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING)
    settings = load_settings()
    asyncio.run(_ask(question, settings))


@app.command()
def show_schema(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Print the graph schema exactly as it is injected into the Query Generator prompt."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING)
    settings = load_settings()

    async def _run() -> None:
        async with MemgraphClient(settings.memgraph) as graph_client:
            schema = await graph_client.build_schema()
            console.print(schema.to_cypher_like_prompt())

    asyncio.run(_run())


def _parse_csv_option(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


@cypherbench_app.command("download")
def cypherbench_download(
    dest: Path = typer.Option(Path("data/cypherbench"), help="Directory to download the dataset into."),
    domains: str = typer.Option(
        None, help=f"Comma-separated domain subset (default: all). Choices: {', '.join(CYPHERBENCH_DOMAINS)}"
    ),
    splits: str = typer.Option("train,test", help="Comma-separated splits to download."),
    graph_variant: str = typer.Option(
        "simplekg_sampled", help="'simplekg_sampled' (small, default) or 'simplekg' (full-scale, large)."
    ),
    force: bool = typer.Option(False, help="Re-download files even if already present."),
    max_concurrency: int = typer.Option(4, help="Max concurrent downloads."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Download CypherBench task files and per-domain graphs from HuggingFace."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    async def _run() -> None:
        paths = await download_cypherbench(
            dest,
            domains=_parse_csv_option(domains),
            splits=_parse_csv_option(splits),
            graph_variant=graph_variant,
            force=force,
            max_concurrency=max_concurrency,
        )
        console.print(f"Downloaded {len(paths)} file(s) into [bold]{dest}[/bold]")

    asyncio.run(_run())


@cypherbench_app.command("evaluate")
def cypherbench_evaluate(
    dest: Path = typer.Option(Path("data/cypherbench"), help="Directory the dataset was downloaded into."),
    split: str = typer.Option("test", help="'train' or 'test'."),
    domains: str = typer.Option(
        None, help="Comma-separated domain subset (default: all present in the split)."
    ),
    limit: int = typer.Option(None, help="Cap the number of questions per domain (applied before grouping)."),
    mode: str = typer.Option(
        "both", help="'single', 'agentic', or 'both' (compare, like the paper's Table 1)."
    ),
    graph_variant: str = typer.Option("simplekg_sampled", help="Graph variant matching what was downloaded."),
    trace: Path = typer.Option(None, help="Optional path to write a JSONL trace of every scored item."),
    concurrency: int = typer.Option(
        None,
        help=(
            "Max questions processed concurrently per domain/mode (default: "
            "MULTIGRAPHRAG_EVALUATION__CONCURRENCY, itself defaulting to 1). "
            "Different questions against the same graph are independent, so "
            "raising this speeds up evaluation runs substantially."
        ),
    ),
    use_judge: bool = typer.Option(
        False,
        "--use-judge",
        help=(
            "Score answers with an LLM-as-a-judge (requires MULTIGRAPHRAG_EVALUATION__JUDGE__* "
            "to be configured) instead of the deterministic value-overlap heuristic. Closer to "
            "the paper's own scoring methodology, at the cost of one extra LLM call per item."
        ),
    ),
    call_log: Path = typer.Option(
        None,
        "--call-log",
        help=(
            "Optional path to write a JSONL transcript of every LLM call made during the run "
            "(one row per call: agent, model, system/user prompt, raw response or error) -- "
            "distinct from --trace, which records one row per question."
        ),
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the Single-vs-Agentic accuracy comparison on a CypherBench split/subset."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    settings = load_settings()

    modes: tuple[EvalMode, ...]
    if mode == "both":
        modes = ALL_MODES
    elif mode in ("single", "agentic"):
        modes = (mode,)  # type: ignore[assignment]
    else:
        raise typer.BadParameter("mode must be 'single', 'agentic', or 'both'")

    judge_built = None
    if use_judge:
        judge_built = build_llm_judge(settings)
        if judge_built is None:
            raise typer.BadParameter(
                "--use-judge requires MULTIGRAPHRAG_EVALUATION__JUDGE__BASE_URL/MODEL to be set."
            )

    async def _run() -> None:
        judge, judge_client = judge_built if judge_built else (None, None)
        call_logger = CallLogger(call_log) if call_log else None
        try:
            report = await run_evaluation(
                settings,
                dest,
                split=split,
                domains=_parse_csv_option(domains),
                limit=limit,
                modes=modes,
                graph_variant=graph_variant,
                trace_path=trace,
                concurrency=concurrency,
                judge=judge,
                call_logger=call_logger,
                on_item=lambda item: console.print(
                    f"[dim]{item.domain}/{item.mode}[/dim] {item.qid[:8]} "
                    f"{'[green]correct[/green]' if item.correct else '[red]wrong[/red]'} "
                    f"(similarity={item.similarity:.2f}, iterations={item.iterations})"
                ),
            )
        finally:
            if judge_client:
                await judge_client.aclose()
            if call_logger:
                call_logger.close()

        score_label = "LLM-as-a-judge" if judge else "approximate execution-match"
        table = Table(title=f"CypherBench {split} accuracy ({score_label})")
        table.add_column("Domain")
        for m in modes:
            table.add_column(m, justify="right")

        domains_seen = sorted({s.domain for s in report.per_domain_mode})
        by_key = {(s.domain, s.mode): s for s in report.per_domain_mode}
        for domain in domains_seen:
            row = [domain]
            for m in modes:
                summary = by_key.get((domain, m))
                row.append(f"{summary.accuracy:.1%}" if summary else "-")
            table.add_row(*row)

        overall_row = ["Average"]
        for m in modes:
            overall_row.append(f"{report.overall(m).accuracy:.1%}")
        table.add_row(*overall_row, style="bold")

        console.print(table)
        if trace:
            console.print(f"Trace written to [bold]{trace}[/bold]")
        if call_log:
            console.print(f"Call log written to [bold]{call_log}[/bold]")

    asyncio.run(_run())


def main() -> None:
    app()


if __name__ == "__main__":
    main()
