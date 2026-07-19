"""Download and load the CypherBench benchmark (Feng, Papicchio & Rahman, 2025).

Source: https://huggingface.co/datasets/megagonlabs/cypherbench (public, apache-2.0).
The dataset ships two flat task files (`train.json`, `test.json`), each a JSON
array of `{qid, graph, gold_cypher, nl_question, answer_json, from_template}`
records where `graph` is the domain/category (e.g. "geography", "art"). Each
domain's property graph is available in two variants under `graphs/`:

- `simplekg/<domain>_simplekg.json`: the full graph (hundreds of MB for some
  domains).
- `simplekg_sampled/<domain>_sampled_simplekg.json`: a much smaller sampled
  subgraph (a few MB), used here as the default so a full download + local
  Memgraph load stays fast; pass `graph_variant="simplekg"` for full-scale runs.

Both graph files are self-contained: `{"schema": {...}, "entities": [...],
"relations": [...]}`, so no separate schema file needs to be fetched.
"""

import asyncio
import json
import logging
import re
from pathlib import Path

import httpx
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from multigraphrag.graph.memgraph_client import MemgraphClient

logger = logging.getLogger(__name__)

HF_DATASET_BASE_URL = "https://huggingface.co/datasets/megagonlabs/cypherbench/resolve/main"

# The 11 Wikidata-derived domains that make up CypherBench.
CYPHERBENCH_DOMAINS: list[str] = [
    "art",
    "biology",
    "company",
    "fictional_character",
    "flight_accident",
    "geography",
    "movie",
    "nba",
    "politics",
    "soccer",
    "terrorist_attack",
]

CYPHERBENCH_SPLITS: list[str] = ["train", "test"]

_LABEL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class CypherBenchTask(BaseModel):
    """A single (question, gold Cypher, gold answer) example."""

    qid: str
    graph: str = Field(description="Domain/category name, e.g. 'geography'.")
    gold_cypher: str
    nl_question: str
    answer_json: str = Field(description="JSON-encoded list of result rows, e.g. '[[\"x\", 1]]'.")
    from_template: dict | None = None

    def parsed_answer(self) -> list:
        return json.loads(self.answer_json)


def _task_file_name(split: str) -> str:
    if split not in CYPHERBENCH_SPLITS:
        raise ValueError(f"Unknown split {split!r}, expected one of {CYPHERBENCH_SPLITS}")
    return f"{split}.json"


def _graph_file_path(domain: str, graph_variant: str) -> str:
    if domain not in CYPHERBENCH_DOMAINS:
        raise ValueError(f"Unknown domain {domain!r}, expected one of {CYPHERBENCH_DOMAINS}")
    if graph_variant == "simplekg_sampled":
        return f"graphs/simplekg_sampled/{domain}_sampled_simplekg.json"
    if graph_variant == "simplekg":
        return f"graphs/simplekg/{domain}_simplekg.json"
    raise ValueError("graph_variant must be 'simplekg' or 'simplekg_sampled'")


def _is_retryable_download_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return False


async def _download_file(client: httpx.AsyncClient, relative_path: str, dest: Path, *, force: bool) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    url = f"{HF_DATASET_BASE_URL}/{relative_path}"

    # Multi-gigabyte graph files (e.g. biology at ~2.4GB) can hit transient
    # 5xx/transport errors mid-transfer; retry the whole file rather than
    # letting one blip abort the entire (possibly hours-long) download batch.
    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception(_is_retryable_download_error),
    )
    async def _attempt() -> Path:
        async with client.stream("GET", url, follow_redirects=True, timeout=300.0) as response:
            response.raise_for_status()
            tmp_dest = dest.with_suffix(dest.suffix + ".part")
            with tmp_dest.open("wb") as handle:
                async for chunk in response.aiter_bytes(chunk_size=1 << 20):
                    handle.write(chunk)
            tmp_dest.rename(dest)
        return dest

    return await _attempt()


async def download_cypherbench(
    dest_dir: Path,
    *,
    domains: list[str] | None = None,
    splits: list[str] | None = None,
    graph_variant: str = "simplekg_sampled",
    force: bool = False,
    max_concurrency: int = 4,
) -> list[Path]:
    """Download task files and per-domain graphs, skipping files already on disk.

    Downloads are fanned out concurrently (bounded by `max_concurrency`),
    reusing the same gather-with-semaphore pattern used for LLM batching.
    """
    domains = domains or CYPHERBENCH_DOMAINS
    splits = splits or CYPHERBENCH_SPLITS
    dest_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, Path]] = [
        (_task_file_name(split), dest_dir / _task_file_name(split)) for split in splits
    ]
    jobs += [
        (_graph_file_path(domain, graph_variant), dest_dir / _graph_file_path(domain, graph_variant))
        for domain in domains
    ]

    semaphore = asyncio.Semaphore(max_concurrency)
    async with httpx.AsyncClient() as client:

        async def _run(relative_path: str, dest: Path) -> Path:
            async with semaphore:
                return await _download_file(client, relative_path, dest, force=force)

        # return_exceptions=True: a persistent failure on one (possibly huge)
        # file must not discard the other concurrent downloads' progress --
        # every job runs to completion, then failures are reported together.
        results = await asyncio.gather(*(_run(rel, dest) for rel, dest in jobs), return_exceptions=True)

    failures = [
        (rel, result)
        for (rel, _dest), result in zip(jobs, results, strict=True)
        if isinstance(result, Exception)
    ]
    if failures:
        details = "; ".join(f"{rel}: {exc}" for rel, exc in failures)
        raise RuntimeError(f"{len(failures)} of {len(jobs)} CypherBench download(s) failed: {details}")

    return results


def load_tasks(
    dest_dir: Path,
    split: str,
    *,
    domains: list[str] | None = None,
    limit: int | None = None,
) -> list[CypherBenchTask]:
    """Load a split's tasks, optionally filtered to a subset of domains and
    capped at `limit` questions *per domain* (not a global cap across the
    combined, multi-domain list -- with several domains selected, a global
    cap would let questions from whichever domain happens to appear first in
    the file crowd out the rest, e.g. domains later in file order silently
    getting 0 questions even though they matched the filter).
    """
    path = dest_dir / _task_file_name(split)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `download_cypherbench` first.")
    raw = json.loads(path.read_text(encoding="utf-8"))
    tasks = [CypherBenchTask.model_validate(item) for item in raw]
    if domains:
        allowed = set(domains)
        tasks = [t for t in tasks if t.graph in allowed]
    if limit is None:
        return tasks

    capped: list[CypherBenchTask] = []
    counts: dict[str, int] = {}
    for task in tasks:
        count = counts.get(task.graph, 0)
        if count >= limit:
            continue
        counts[task.graph] = count + 1
        capped.append(task)
    return capped


def load_domain_graph(dest_dir: Path, domain: str, *, graph_variant: str = "simplekg_sampled") -> dict:
    path = dest_dir / _graph_file_path(domain, graph_variant)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `download_cypherbench` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_label(label: str) -> str:
    if not _LABEL_RE.match(label):
        raise ValueError(f"Unsafe/unexpected label in CypherBench graph data: {label!r}")
    return label


async def populate_memgraph(
    graph_client: MemgraphClient,
    domain_graph: dict,
    *,
    batch_size: int = 1000,
) -> None:
    """Wipe the target database and load a CypherBench domain graph into it.

    Node properties are flattened (Wikidata `properties` merged with `name`,
    `aliases`, `description`) so the resulting graph matches the flat,
    Cypher-syntax-like schema the agents expect (see `graph/models.py`). An
    internal `_eid` property carries CypherBench's entity id so relations can
    be wired up by matching on it; it is excluded from prompt schemas because
    `MemgraphClient.get_properties_for_label` filters out `_`-prefixed keys.

    Deleting all nodes does not release the memory backing the previous
    domain's `_eid` indexes (still declared for labels no longer present) or
    Memgraph's own internal allocator arenas. Left unaddressed, this compounds
    across an evaluation run that loads several large full-scale domains in
    sequence and can OOM-kill the whole process (observed: reliably crashing
    on the 2nd-3rd large domain, hundreds of thousands of nodes each, in a
    single long-lived Memgraph instance). Dropping every existing index before
    wiping and issuing `FREE MEMORY` afterwards keeps memory bounded to
    roughly one domain's worth at a time.
    """
    existing_indexes = await graph_client.run_query("SHOW INDEX INFO")
    for row in existing_indexes.records:
        label = row.get("label")
        props = row.get("property") or []
        prop = props[0] if isinstance(props, list) else props
        if label and prop:
            await graph_client.run_query(f"DROP INDEX ON :`{label}`({prop})")

    await graph_client.run_query("MATCH (n) DETACH DELETE n")
    await graph_client.run_query("FREE MEMORY")

    entities_by_label: dict[str, list[dict]] = {}
    for entity in domain_graph["entities"]:
        label = _validate_label(entity["label"])
        node = {
            **entity.get("properties", {}),
            "_eid": entity["eid"],
            "name": entity.get("name"),
            "aliases": entity.get("aliases") or [],
            "description": entity.get("description"),
        }
        entities_by_label.setdefault(label, []).append(node)

    for label, nodes in entities_by_label.items():
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i : i + batch_size]
            await graph_client.run_query(
                f"UNWIND $rows AS row CREATE (n:`{label}`) SET n = row", {"rows": batch}
            )
        await graph_client.run_query(f"CREATE INDEX ON :`{label}`(_eid)")

    # Relations must be grouped by (rel_label, subj_label, obj_label), not just
    # rel_label: MATCH (a {_eid: ...}) with no label on `a` cannot use the
    # per-label `:Label(_eid)` index created above, so Memgraph would fall back
    # to a full node scan for every single lookup -- catastrophic on graphs
    # with hundreds of thousands of nodes. Including the literal label (looked
    # up per-entity, since a relation type is not always monomorphic, e.g.
    # CypherBench's `fromUniverse` applies to both Character and Organization
    # subjects) keeps every lookup on the indexed path.
    eid_to_label = {entity["eid"]: _validate_label(entity["label"]) for entity in domain_graph["entities"]}

    relations_by_key: dict[tuple[str, str, str], list[dict]] = {}
    skipped = 0
    for relation in domain_graph["relations"]:
        rel_label = _validate_label(relation["label"])
        subj_label = eid_to_label.get(relation["subj_id"])
        obj_label = eid_to_label.get(relation["obj_id"])
        if subj_label is None or obj_label is None:
            skipped += 1
            continue
        relations_by_key.setdefault((rel_label, subj_label, obj_label), []).append(
            {
                "subj_id": relation["subj_id"],
                "obj_id": relation["obj_id"],
                "properties": relation.get("properties", {}),
            }
        )
    if skipped:
        logger.warning("Skipped %d relations referencing unknown entity ids", skipped)

    for (rel_label, subj_label, obj_label), rels in relations_by_key.items():
        for i in range(0, len(rels), batch_size):
            batch = rels[i : i + batch_size]
            await graph_client.run_query(
                f"""
                UNWIND $rows AS row
                MATCH (a:`{subj_label}` {{_eid: row.subj_id}})
                MATCH (b:`{obj_label}` {{_eid: row.obj_id}})
                CREATE (a)-[r:`{rel_label}`]->(b)
                SET r = row.properties
                """,
                {"rows": batch},
            )
