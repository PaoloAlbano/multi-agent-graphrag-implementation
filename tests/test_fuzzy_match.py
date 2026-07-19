from multigraphrag.verification.fuzzy_match import suggest_candidates


def test_suggest_candidates_ranks_closest_match_first():
    known_values = ["Corlys Velaryon", "Lucerys Velaryon", "Jacaerys Velaryon", "Daemon Targaryen"]

    candidates = suggest_candidates("corlys velaryon", known_values, top_k=3, min_score=50.0)

    assert candidates
    assert candidates[0].value == "Corlys Velaryon"
    assert candidates[0].score > 80


def test_suggest_candidates_empty_when_no_known_values():
    assert suggest_candidates("anything", [], top_k=5) == []


def test_suggest_candidates_filters_below_min_score():
    candidates = suggest_candidates("zzzzzzzzzz", ["Corlys Velaryon"], min_score=90.0)
    assert candidates == []
