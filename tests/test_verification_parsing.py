from multigraphrag.agents.verification import parse_relationship_pattern


def test_parse_directed_pattern():
    assert parse_relationship_pattern("(:Character)-[:hasFather]->(:Character)") == (
        "Character",
        "hasFather",
        "Character",
    )


def test_parse_undirected_pattern():
    assert parse_relationship_pattern("(:Character)-[:hasSpouse]-(:Character)") == (
        "Character",
        "hasSpouse",
        "Character",
    )


def test_parse_invalid_pattern_returns_none():
    assert parse_relationship_pattern("not a pattern") is None


def test_parse_pattern_with_variable_names():
    assert parse_relationship_pattern("(n:Character)-[r:hasFather]->(m:Character)") == (
        "Character",
        "hasFather",
        "Character",
    )


def test_parse_undirected_pattern_with_variable_names():
    assert parse_relationship_pattern("(n:Character)-[r:hasSpouse]-(m:Character)") == (
        "Character",
        "hasSpouse",
        "Character",
    )
