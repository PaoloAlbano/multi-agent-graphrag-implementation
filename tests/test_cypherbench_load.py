import pytest

from multigraphrag.evaluation.cypherbench import populate_memgraph


class _FakeOutcome:
    def __init__(self, records=None):
        self.records = records or []


class _FakeGraphClient:
    def __init__(self):
        self.queries: list[tuple[str, dict | None]] = []

    async def run_query(self, cypher, params=None):
        self.queries.append((cypher, params))
        return _FakeOutcome()


@pytest.mark.asyncio
async def test_populate_memgraph_relation_match_includes_labels_for_index_usage():
    """MATCH (a {_eid: ...}) with no label cannot use the per-label `:Label(_eid)`
    index, forcing a full node scan on every relation lookup -- catastrophically
    slow on large graphs (confirmed: ~13 relations/sec on a 581k-node graph,
    projected ~6 hours). Both endpoints must carry their literal label.
    """
    graph = _FakeGraphClient()
    domain_graph = {
        "entities": [
            {"eid": "Company#1", "label": "Company", "name": "Acme", "properties": {}},
            {"eid": "Person#1", "label": "Person", "name": "Alice", "properties": {}},
        ],
        "relations": [
            {"label": "hasCEO", "subj_id": "Company#1", "obj_id": "Person#1", "properties": {}},
        ],
    }

    await populate_memgraph(graph, domain_graph)

    relation_queries = [q for q, _ in graph.queries if "CREATE (a)-[r:" in q]
    assert len(relation_queries) == 1
    query = relation_queries[0]
    assert "MATCH (a:`Company` {_eid: row.subj_id})" in query
    assert "MATCH (b:`Person` {_eid: row.obj_id})" in query


@pytest.mark.asyncio
async def test_populate_memgraph_skips_relations_with_unknown_entity_ids():
    graph = _FakeGraphClient()
    domain_graph = {
        "entities": [
            {"eid": "Company#1", "label": "Company", "name": "Acme", "properties": {}},
        ],
        "relations": [
            {"label": "hasCEO", "subj_id": "Company#1", "obj_id": "Person#missing", "properties": {}},
        ],
    }

    await populate_memgraph(graph, domain_graph)

    relation_queries = [q for q, _ in graph.queries if "CREATE (a)-[r:" in q]
    assert relation_queries == []
