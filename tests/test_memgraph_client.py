import pytest

from multigraphrag.config import MemgraphSettings
from multigraphrag.graph.memgraph_client import MemgraphClient


class _DummyDriver:
    async def close(self):
        pass


@pytest.mark.asyncio
async def test_run_query_empty_string_returns_error_outcome_instead_of_raising():
    """The neo4j driver raises a client-side ValueError for an empty query --
    not a Neo4jError -- which previously went uncaught and crashed the whole
    ask() call whenever the Query Generator hallucinated an empty string
    (observed repeatedly against a live gpt-oss-120b run: "Cannot run an
    empty query" killed several questions outright instead of triggering the
    self-correction loop).
    """
    client = MemgraphClient(MemgraphSettings(), driver=_DummyDriver())

    outcome = await client.run_query("")

    assert outcome.success is False
    assert "empty" in outcome.error_message.lower()


@pytest.mark.asyncio
async def test_run_query_whitespace_only_returns_error_outcome():
    client = MemgraphClient(MemgraphSettings(), driver=_DummyDriver())

    outcome = await client.run_query("   \n\t  ")

    assert outcome.success is False


class _RaisingSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def run(self, cypher, parameters):
        raise ValueError("some other client-side driver rejection")


class _RaisingDriver:
    def session(self, database=None):
        return _RaisingSession()

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_run_query_other_value_errors_from_driver_are_caught_too():
    client = MemgraphClient(MemgraphSettings(), driver=_RaisingDriver())

    outcome = await client.run_query("MATCH (n) RETURN n")

    assert outcome.success is False
    assert "some other client-side driver rejection" in outcome.error_message
