# Multi-Agent GraphRAG

Python implementation of *Multi-Agent GraphRAG: A Text-to-Cypher Framework for
Labeled Property Graphs* (Gusarov et al., 2025, [arXiv:2511.08274](https://arxiv.org/abs/2511.08274)).

A modular agentic pipeline that turns a natural language question into a
Cypher query over a Memgraph (or any Bolt-compatible LPG) database,
iteratively self-corrects it using database-grounded feedback, and returns a
natural language answer.

This is a from-scratch implementation based on reading the paper; it is not
affiliated with the original authors and does not reuse their code.

**[Browse benchmark results](https://paoloalbano.github.io/multi-agent-graphrag-implementation/)**
-- per-model, per-domain Single-vs-Agentic accuracy, with drill-down into
individual questions and (for new runs) their full LLM call transcripts.

## Architecture

Seven cooperating agents plus a graph database executor, driven by a plain
async loop (`GraphRAGPipeline.ask` in `workflow/pipeline.py`) that implements
the paper's Algorithm 1 (self-correction loop, capped at
`MULTIGRAPHRAG_WORKFLOW__MAX_REFINEMENT_ITERATIONS` iterations, default 5 -- 1 initial + 4 corrections, matching Algorithm 1's pseudocode).
No graph/DAG orchestration library is used: the control flow is a bounded
loop with two branches, which a `while` expresses more directly (and with far
fewer moving parts) than a state-graph runtime would -- the paper mentions
using LangGraph for this, but only in one sentence with no further detail on
how, and nothing here needs checkpointing, streaming, or parallel branches.

```
generate_query -> execute_query -> evaluate_query --(accept / iterations exhausted)--> interpret
                                         |
                                         |--(incorrect)--> aggregate_feedback --> generate_query
                                         |
                                         `--(error/empty)--> extract_entities -> verify_entities
                                                              -> generate_instructions
                                                              -> aggregate_feedback --> generate_query
```

| Agent | Module | Role |
|---|---|---|
| Query Generator | `agents/query_generator.py` | NL question + schema (+ feedback) -> Cypher |
| Graph DB Executor | `graph/memgraph_client.py` | Runs Cypher against Memgraph, captures errors/empty results |
| Query Evaluator | `agents/evaluator.py` | Grades a query: `accept` / `incorrect` / `error_or_empty` |
| Named Entity Extractor | `agents/entity_extractor.py` | Pulls node labels, property/value literals, relationship patterns out of the query |
| Verification Module | `agents/verification.py` | Checks each extracted entity against the live graph; on miss, runs RapidFuzz Levenshtein candidate retrieval + LLM semantic re-ranking |
| Instructions Generator | `agents/instructions_generator.py` | Turns verification misses into concrete per-entity fix instructions |
| Feedback Aggregator | `agents/feedback_aggregator.py` | Merges evaluator + verification signals into one instruction for the next attempt |
| Interpreter | `agents/interpreter.py` | Turns the accepted (or best-effort) query result into a natural language answer |

### Model adapter, decoupled from agents

Every agent depends only on the `LLMClient` abstract interface
(`llm/base.py`). The concrete implementation, `VllmClient` (`llm/vllm_client.py`),
talks to vLLM's **OpenAI-compatible** `/chat/completions` endpoint using plain
`httpx` -- no vendor SDK -- and works unmodified against any other
OpenAI-compatible server too. Swapping endpoints, API keys, or even giving
each agent a different model/provider is purely a configuration change (see
below); no agent code changes.

Structured agent outputs (the Cypher query, the evaluation verdict, the
extracted entities, ...) are pydantic models in `schemas.py`. They are
requested one of two ways, controlled by `LLMSettings.structured_output_mode`:

- `json_schema` (default): sends `response_format: json_schema`, relying on
  the endpoint's guided-decoding backend (vLLM + outlines/xgrammar/etc.) to
  constrain generation server-side.
- `prompt`: for endpoints/models that don't support guided decoding (older
  vLLM builds, no backend installed, or a model without JSON-mode support).
  The schema is embedded directly in the prompt text instead, with no
  `response_format` sent.

Either way, the raw response text is always parsed and validated
client-side against the pydantic model, so a model that ignores the
instruction surfaces a clear error instead of failing silently. The mode is
part of `LLMSettings`, so -- like the endpoint/model/API key -- it can be set
globally or overridden per agent (e.g. `json_schema` for agents on a vLLM
model with guided decoding, `prompt` for one routed to a model that lacks
it).

`LLMSettings.max_tokens` defaults to `16000`. Reasoning models can spend a
large, variable share of the completion budget on chain-of-thought before
ever writing the final `content`, so a low/unset cap risks silently
truncating the structured JSON output every agent depends on. Set it to
`None` to let the endpoint apply its own default instead.

Reasoning/"thinking" models are controlled by two dedicated settings, since
every agent here only ever reads the final `content` field, never
`reasoning_content`:

- `LLMSettings.reasoning_enabled` (default `True`): set to `False` to
  actively suppress chain-of-thought via
  `chat_template_kwargs.enable_thinking=false` -- the mechanism for
  Qwen3-style "thinking" models on vLLM (observed: ~101 -> ~2 completion
  tokens per call for a trivial prompt with this set).
- `LLMSettings.reasoning_effort` (default `"medium"`): forwarded verbatim as
  a top-level `reasoning_effort` field when `reasoning_enabled` is `True` --
  the OpenAI/gpt-oss-style knob (`"low"`/`"medium"`/`"high"`). Ignored
  (harmlessly, confirmed empirically) by endpoints/models that don't
  recognize it, e.g. Qwen3. Set to `None` to omit it entirely.

`LLMSettings.extra_body` is a generic dict merged verbatim into every request
payload, applied last so it can override any of the above -- an escape hatch
for anything else vendor/model-specific that doesn't warrant a dedicated
setting.

### Concurrency

`LLMClient.complete_many` / `complete_structured_many` fan out independent
completions concurrently via `asyncio.gather`, bounded by an
`asyncio.Semaphore` sized from `LLMSettings.max_concurrency`. The Verification
Module uses this to semantically re-rank multiple hallucinated entities in
parallel instead of awaiting one LLM call at a time.

## Configuration

All configuration is `pydantic-settings`, so it can come from environment
variables, a `.env` file, or explicit `Settings(...)` construction in tests.
Copy `.env.example` to `.env` and adjust:

```dotenv
MULTIGRAPHRAG_LLM__BASE_URL=http://localhost:8000/v1   # vLLM OpenAI-compatible server
# MULTIGRAPHRAG_LLM__API_KEY=sk-...   # optional -- omit for endpoints that require no auth
MULTIGRAPHRAG_LLM__MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
MULTIGRAPHRAG_LLM__MAX_CONCURRENCY=5

MULTIGRAPHRAG_MEMGRAPH__URI=bolt://localhost:7687
MULTIGRAPHRAG_MEMGRAPH__USERNAME=
MULTIGRAPHRAG_MEMGRAPH__PASSWORD=

MULTIGRAPHRAG_WORKFLOW__MAX_REFINEMENT_ITERATIONS=5
```

Per-agent overrides (different endpoint/model/key per role) live under
`MULTIGRAPHRAG_AGENT_MODELS__<AGENT>__...`, e.g.:

```dotenv
MULTIGRAPHRAG_AGENT_MODELS__QUERY_GENERATOR__MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
MULTIGRAPHRAG_AGENT_MODELS__VERIFICATION_RANKER__BASE_URL=https://api.openai.com/v1
MULTIGRAPHRAG_AGENT_MODELS__VERIFICATION_RANKER__MODEL=gpt-4o-mini
MULTIGRAPHRAG_AGENT_MODELS__VERIFICATION_RANKER__API_KEY=sk-...
```

Any agent left unset falls back to the default `MULTIGRAPHRAG_LLM__*` block
(resolved by `composition.resolve_agent_llm`, see below).

### Composition root

`config.py` holds nothing but plain pydantic-settings data -- no factories, no
resolution logic. All wiring (reading `Settings`, building `VllmClient`s,
constructing agents, assembling a `GraphRAGPipeline`/`SinglePassRunner`)
happens in one place,
`composition.py`, following the Clean Architecture idea of a composition
root: use cases and agents depend only on abstractions (`LLMClient`,
`MemgraphClient`) and receive collaborators via constructor injection, so
they stay free of config/adapter concerns. `cli.py` and
`evaluation/runner.py` are the two callers of `composition.build_pipeline` /
`composition.build_single_pass_runner`.

## Setup

This project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync                       # installs runtime + dev dependencies
cp .env.example .env          # then edit with your vLLM endpoint / Memgraph URI
```

You need a running Memgraph instance (`docker run -p 7687:7687 memgraph/memgraph`)
loaded with your graph, and an OpenAI-compatible LLM endpoint (e.g.
`vllm serve <model> --port 8000`).

## Usage

A `Makefile` wraps the common commands (see `make help`):

```bash
make sync                 # uv sync
make memgraph-up          # start Memgraph locally via Docker (Bolt on :7687)
make ask Q="How many doors exist in the building?"
make show-schema
make test
make lint / make format
make memgraph-down        # stop the local Memgraph container
```

Equivalently, without `make`:

```bash
uv run multigraphrag ask "How many characters have Corlys Velaryon as their father or are married to Daemon Targaryen?"
uv run multigraphrag show-schema   # inspect the schema exactly as injected into the Query Generator prompt
```

> If your shell has an unrelated Python virtualenv active (e.g. conda), `uv`
> may pick it up via `$VIRTUAL_ENV` instead of this project's `.venv`. If you
> hit that, prefix commands with `env -u VIRTUAL_ENV`, e.g.
> `env -u VIRTUAL_ENV uv run pytest`. The Makefile already does this for you.

Programmatically:

```python
from multigraphrag.composition import build_pipeline, load_settings
from multigraphrag.graph.memgraph_client import MemgraphClient

settings = load_settings()
async with MemgraphClient(settings.memgraph) as graph_client:
    async with build_pipeline(settings, graph_client) as pipeline:
        result = await pipeline.ask("How many doors exist in the building?")
        print(result.answer, result.cypher, result.accepted, result.iterations)
```

## Benchmarking against CypherBench

The evaluation harness downloads [CypherBench](https://huggingface.co/datasets/megagonlabs/cypherbench)
(Feng, Papicchio & Rahman, 2025 — the same benchmark used in the paper) and
runs a Single-vs-Agentic comparison like the paper's Table 1, per-domain and
overall.

```bash
make memgraph-up
make cypherbench-download                                   # all 11 domains, train+test
make cypherbench-download CYPHERBENCH_DOMAINS=geography,art  # or just a subset

make cypherbench-eval CYPHERBENCH_SPLIT=test CYPHERBENCH_DOMAINS=geography CYPHERBENCH_LIMIT=50 CYPHERBENCH_CONCURRENCY=5
make cypherbench-eval-single    # only the linear-pass baseline
make cypherbench-eval-agentic   # only the full Multi-Agent GraphRAG pipeline
```

Or directly:

```bash
uv run multigraphrag cypherbench download --dest data/cypherbench
uv run multigraphrag cypherbench evaluate --dest data/cypherbench \
    --split test --domains geography --limit 50 --mode both --trace trace.jsonl
```

Notes:

- **Splits/domains/limit**: `--split train|test`, `--domains a,b,c` (any of
  `art, biology, company, fictional_character, flight_accident, geography,
  movie, nba, politics, soccer, terrorist_attack`), `--limit N` to cap
  questions per domain -- so you can run on just the train split, just the
  test split, or a single category.
- **Why `limit=100` (per domain)?** CypherBench's test split has 189-401
  questions per domain, so a run over every domain is expensive. `limit` is
  applied **per domain** (not globally -- see
  `tests/test_cypherbench_tasks.py`), so each selected domain contributes
  exactly `N` questions. The default is **100 questions per domain**,
  balancing three things: (1) statistical power -- at N=40 the 95% CI on an
  accuracy is roughly ±15pp, at N=100 it narrows to ±10pp; (2) evaluation
  cost -- the agentic pipeline makes ~5-10 LLM calls per question over the
  full-scale graphs; (3) cross-model comparability -- every model in
  `results/` runs the *same* per-domain cap so the numbers line up. It is
  purely a budget knob: override with `LIMIT=N` (Makefile) or `--limit N`
  (CLI) at any time. If you change it, keep it identical across the models
  you want to compare, and the per-run `run.json` records the exact `limit`
  used.
- **Graph loading**: for each domain in scope, its CypherBench graph is loaded
  into the configured Memgraph instance (wiping whatever was there before),
  the schema is introspected once, then every requested mode runs over that
  domain's questions.
- **Graph size**: by default the much smaller `simplekg_sampled` variant is
  downloaded/loaded (a few MB per domain); pass `--graph-variant simplekg`
  for the full-scale graphs (hundreds of MB, matches the paper's "7.8 million
  entities" full-scale setting). Loading is batched and label-indexed so even
  a 580k-node/300k-relationship full graph loads in ~20s.
- **Concurrency**: `--concurrency N` (or `MULTIGRAPHRAG_EVALUATION__CONCURRENCY`,
  default 1) runs up to N questions per domain/mode in parallel -- different
  questions against the same already-loaded graph are fully independent, so
  this can speed up a run substantially. Each agent's own
  `LLMSettings.max_concurrency` still separately bounds in-flight requests to
  its endpoint.
- **Scoring**: by default (`--use-judge` is on unless you pass `--no-judge`),
  correctness is scored by an **LLM-as-a-judge** over the natural language
  answer -- the paper's own methodology. The judge's exact prompt (including
  its few-shot examples) is not published, so this is an independent
  reconstruction of the same idea (`evaluation/judge.py`), not a
  byte-for-byte reproduction of the paper's judge. The judge's model is
  chosen exactly as follows, and the choice is **recorded in every
  `run.json`** so the methodology is never ambiguous:
  - **Self-judge** (`judge_mode: "self"`): each model judges its own
    answers. This is what the current `results/` use. It needs zero extra
    infrastructure, but has a well-known **self-serving bias** risk -- a
    model tends to be lenient with its own mistakes, so absolute numbers are
    *not* directly comparable across models, only within a model across
    domains/configs (and even then only approximately).
  - **Fixed judge** (`judge_mode: "fixed"`): one dedicated judge model
    (`MULTIGRAPHRAG_EVALUATION__JUDGE__*` -- same shape as `LLMSettings`,
    e.g. a strong model like the paper's GigaChat 2 MAX) scores **every**
    model's answers. This is the paper-faithful, cross-model-comparable
    setup, at the cost of standing up one extra model. Run it after the fact
    on existing traces with `make rejudge-fixed` (no pipeline re-run needed).
  Each leaf can carry **both**: `judge_mode: "self"` plus an optional
  `external_judge` block recording the fixed judge's verdicts, so the site
  can show both scores side by side. The deterministic `similarity` score is
  always computed and recorded alongside for reference.
- **Deterministic fallback** (`--no-judge`): scores by comparing the
  pipeline's raw Cypher result values against CypherBench's `answer_json`
  ground truth (normalized value-multiset Jaccard similarity, threshold 0.8;
  see `evaluation/matching.py`) -- free and judge-free, but a coarser
  approximation: it penalizes an otherwise-correct answer whose Cypher
  returns extra context columns beyond the gold query's single column, since
  it compares the *entire* returned value multiset rather than the final
  natural-language answer. The deterministic `similarity` score is always
  computed and recorded alongside the judge's verdict (when a judge is used)
  for reference either way.

## Publishing results / contributing a model

Beyond the ad-hoc `cypherbench-eval` commands above, `results/` is a curated,
published record of Single-vs-Agentic runs, one directory per model/config/
domain/mode, meant to be browsable and diffable in PRs:

```
results/<model-slug>/temp<T>-reasoning-<effort|off>/<domain>/<mode>/
    trace.jsonl   # one row per question: gold/generated Cypher, answer, correct
    calls.jsonl   # every LLM call made, tagged with the question's qid
    run.json      # the settings used (model, temperature, reasoning, ...) plus accuracy
```

To reproduce a run and add your own model's results:

```bash
make memgraph-up
make cypherbench-download-domain DOMAIN=geography
make run-single  MODEL=Qwen/Qwen3.5-27B DOMAIN=geography
make run-agentic MODEL=Qwen/Qwen3.5-27B DOMAIN=geography TEMP=1.0 REASONING=off
```

Each invocation covers exactly one domain and one mode and writes directly
into the `results/` layout above -- no manual `--trace`/`--call-log` path
bookkeeping needed. `REASONING` accepts `off`, `low`, `medium`, or `high`
(forwarded to `MULTIGRAPHRAG_LLM__REASONING_ENABLED`/`__REASONING_EFFORT`);
`TEMP`, `LIMIT` (default 100), `CONCURRENCY` (default 20), and
`GRAPH_VARIANT` (default `simplekg`, full-scale) are all overridable, e.g.
`make run-single MODEL=... DOMAIN=... CONCURRENCY=1` for the larger domains
(`art`, `geography`) if Memgraph runs low on memory.

### What a "config" directory means (`temp<T>-reasoning-<effort|off>`)

Each model's results live under one directory per *config* -- a fixed
`(temperature, reasoning)` pair, i.e. one cell of the following matrix:

| | `reasoning-off` | `reasoning-medium` |
|---|---|---|
| `temp0.0` | `temp0.0-reasoning-off` | `temp0.0-reasoning-medium` |
| `temp1.0` | `temp1.0-reasoning-off` | `temp1.0-reasoning-medium` |

- `temp<T>` is the sampling temperature (`0.0` greedy vs `1.0`).
- `reasoning-off` sets `MULTIGRAPHRAG_LLM__REASONING_ENABLED=false`
  (suppresses chain-of-thought via `enable_thinking=false`).
- `reasoning-medium` sets `REASONING_ENABLED=true` +
  `REASONING_EFFORT=medium` (for gpt-oss-style effort knobs; ignored
  harmlessly by models that don't recognize it, e.g. Qwen3).

The four standard configs in `results/` are exactly the 2×2 cells above.
A `run.json` records the authoritative `temperature` / `reasoning_enabled` /
`reasoning_effort` values, so the directory name is just a human-readable
shorthand -- never parse it as the source of truth.

### Which split a domain uses (train-only vs test)

The 11 CypherBench domains are split unevenly across `train`/`test`: only
`company, fictional_character, flight_accident, geography, movie, nba,
politics` appear in `test`, while `art, biology, soccer, terrorist_attack`
only appear in `train`. The Makefile encodes this (`CYPHERBENCH_TRAIN_ONLY_DOMAINS`)
and picks the split per domain automatically -- you don't need to, but it's
why e.g. `art` results live under `split: train` while `geography` uses
`split: test`. If you invoke the CLI directly, pass the correct `--split` for
the domain you're running (see the table in `data/cypherbench/*.json`).

### Comparability across the published runs

The current `results/` are a mix of two generations:

- **`openai--gpt-oss-20b`** is the oldest: 11 domains, `limit` 40, run before
  the current methodology was settled. It's kept as-is (it was the highest
  budget anyway), but treat its numbers as *indicative* -- its per-domain
  accuracy has a wider CI (±15pp) than the newer runs.
- **Every other model** (`Qwen3.5-27B`, `gpt-oss-120b`, `Nemotron-3-Nano-30B`)
  covers the same **5 shared domains** (`art, company, fictional_character,
  flight_accident, geography`) at `limit` 100, so they're directly comparable
  with each other. The shared-domain subset is what the overview table's
  per-model averages are computed over, keeping the comparison apples-to-apples.

If budget allows later, re-running `gpt-oss-20b` on the shared 5 domains at
`limit` 100 would fold it into the comparable set.

Then:

1. Open a PR adding **only** the new raw `results/<model>/.../{trace,calls}.jsonl`
   and `run.json` files -- do **not** run `make recap`/`make site` or commit
   `results/RECAP.md`, `results/recap.json`, or anything under `docs/` in
   your branch. Those are generated artifacts that `main`'s CI regenerates
   after every merge (see below); committing them yourself just creates
   merge/rebase conflicts against whatever `main` regenerated since your
   branch diverged, with no benefit (your copy gets thrown away and rebuilt
   from the full `results/` tree anyway).
2. `.github/workflows/validate-results.yml` runs `scripts/validate_results.py`
   on the PR, checking every `trace.jsonl`/`calls.jsonl`/`run.json` has the
   expected fields.
3. Once merged to `main`, `.github/workflows/build-site.yml` runs
   `scripts/build_recap.py` (regenerates `results/RECAP.md` +
   `results/recap.json`) and `scripts/build_site.py` (regenerates `docs/`,
   published via GitHub Pages -- configure the repo to serve `/docs` from
   `main`), committing the result back automatically.

`make recap` and `make site` exist for **local preview only** (open
`docs/index.html` with `python -m http.server` from inside `docs/` to see
how your new results will look) -- run them locally to check, but don't
commit what they produce.

**Known limitation**: runs migrated from this project's early exploratory
testing (see `scripts/migrate_legacy_results.py`) predate per-question call
tagging, so their `run.json` sets `calls_log_scoped: false` and the site
shows a combined, non-attributable transcript instead of a per-question
drill-down. Every run produced via `make run-single`/`run-agentic` going
forward has full per-question traceability (`calls.jsonl` rows carry `qid`).

## Project layout

```
src/multigraphrag/
  config.py            # pydantic-settings data only: LLM/agent/Memgraph/workflow config
  composition.py        # composition root: Settings -> LLMClients -> agents -> pipeline
  schemas.py             # structured pydantic I/O models exchanged between agents
  llm/
    base.py             # LLMClient abstract interface + concurrency-bounded batching
    vllm_client.py       # httpx adapter for vLLM's OpenAI-compatible endpoint
    factory.py            # LLMSettings -> LLMClient
  graph/
    memgraph_client.py  # async Bolt client: query execution + schema introspection
    models.py            # QueryOutcome, GraphSchema, ...
  verification/
    fuzzy_match.py       # RapidFuzz normalized-Levenshtein candidate suggestion
  prompts/
    system_prompts.py    # one system prompt per agent role
  agents/                # the seven agents described above
  workflow/
    pipeline.py           # GraphRAGPipeline use case: Algorithm 1's self-correction loop -> `ask()`
    single_pass.py          # linear-pass ("Single") baseline for comparison
  evaluation/
    cypherbench.py        # download + Memgraph loader for the CypherBench benchmark
    matching.py            # approximate execution-accuracy scoring vs. gold answers
    judge.py                # optional LLM-as-a-judge scoring (--use-judge)
    runner.py              # Single-vs-Agentic evaluation loop over a split/domain subset
  cli.py                  # `multigraphrag ask/show-schema/cypherbench download|evaluate`
```

## Status / known gaps vs. the paper

This is a first pass meant to be reviewed and iterated on:

- A CypherBench-based evaluation harness is implemented (see above), but it
  scores answers with a deterministic value-overlap heuristic rather than
  reproducing the paper's undisclosed LLM-as-a-judge prompt; treat numbers as
  relative (Single vs. Agentic), not as a reproduction of the paper's Table 1.
- The IFC/Sample House benchmark from the paper is not implemented.
- Compositional/disjunctive queries and symmetric relationships are called
  out in the paper as open limitations of the approach itself, not something
  this implementation works around.
- No persistence/checkpointing of in-progress questions is wired up; each
  `ask()` call runs the self-correction loop to completion in-process.
