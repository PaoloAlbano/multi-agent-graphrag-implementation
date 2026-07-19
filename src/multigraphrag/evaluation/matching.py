"""Approximate execution-accuracy scoring against CypherBench gold answers.

CypherBench's own paper reports an "Execution Accuracy" metric; the
Multi-Agent GraphRAG paper instead uses an LLM-as-a-judge over the final
natural language answer, with an undisclosed judge prompt. Neither is
reproducible here without extra assumptions, so this module implements a
simpler, deterministic, judge-free proxy: flatten both the pipeline's
returned records and the gold `answer_json` rows into a multiset of
normalized scalar values, and score their overlap with a Jaccard-style
similarity. This is intentionally documented as an approximation (see
README), not a reproduction of either paper's exact number.
"""

from collections import Counter

_NULL_TOKENS = {"none", "null", "nan"}


def _normalize_scalar(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        # Normalize 3.0 and 3 to the same token, and round to avoid float noise.
        rounded = round(float(value), 6)
        return f"{rounded:g}"
    text = str(value).strip().lower()
    return None if text in _NULL_TOKENS else text


def _flatten(rows: list) -> Counter:
    counter: Counter = Counter()
    for row in rows:
        cells = row if isinstance(row, list | tuple) else [row]
        for cell in cells:
            normalized = _normalize_scalar(cell)
            if normalized is not None:
                counter[normalized] += 1
    return counter


def flatten_records(records: list[dict]) -> Counter:
    """Flatten `QueryOutcome.records` (list of `{column: value}` dicts) into a value multiset."""
    return _flatten([list(record.values()) for record in records])


def flatten_gold(answer_rows: list) -> Counter:
    """Flatten a parsed `answer_json` (list of result rows) into a value multiset."""
    return _flatten(answer_rows)


def jaccard_similarity(predicted: Counter, gold: Counter) -> float:
    """Multiset Jaccard similarity between predicted and gold value sets.

    Returns 1.0 when both are empty (an empty gold answer correctly matched
    by an empty result), 0.0 if only one side is empty.
    """
    if not predicted and not gold:
        return 1.0
    intersection = sum((predicted & gold).values())
    union = sum((predicted | gold).values())
    return intersection / union if union else 0.0


def score_records_against_gold(
    records: list[dict], answer_rows: list, *, threshold: float = 0.8
) -> tuple[float, bool]:
    """Return (jaccard_similarity, is_correct) for a query result vs. the gold answer."""
    similarity = jaccard_similarity(flatten_records(records), flatten_gold(answer_rows))
    return similarity, similarity >= threshold
