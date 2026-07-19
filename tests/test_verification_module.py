import pytest

from multigraphrag.agents.verification import VerificationModule
from multigraphrag.config import WorkflowSettings
from multigraphrag.schemas import ExtractedEntities


class _FakeGraphClient:
    def __init__(self):
        self.checked_patterns = []

    async def label_exists(self, label):
        return True

    async def relationship_pattern_exists(self, start_label, rel_type, end_label):
        self.checked_patterns.append((start_label, rel_type, end_label))
        return True

    async def property_value_exists(self, label, prop, value):
        return True

    async def get_all_property_values(self, label, prop):
        return []


class _UnusedLLMClient:
    async def complete_structured_many(self, prompts, response_model):
        raise AssertionError("should not be called when nothing is hallucinated")


@pytest.mark.asyncio
async def test_unparseable_relationship_pattern_is_skipped_not_marked_false():
    """An extractor output that doesn't match the expected pattern shape must not be
    reported as a hallucination (previously it silently defaulted to exists=False,
    which could feed a false correction signal into the self-correction loop).
    """
    graph = _FakeGraphClient()
    module = VerificationModule(_UnusedLLMClient(), graph, WorkflowSettings())

    entities = ExtractedEntities(
        pairwise_relationships=[
            "(:Character)-[:hasFather]->(:Character)",  # parseable
            "totally not a cypher pattern",  # unparseable -- must be skipped, not flagged
        ]
    )

    report = await module.verify(entities, question="does it matter")

    assert len(report.relationships) == 1
    assert report.relationships[0].pattern == "(:Character)-[:hasFather]->(:Character)"
    assert report.relationships[0].exists is True
    assert graph.checked_patterns == [("Character", "hasFather", "Character")]
