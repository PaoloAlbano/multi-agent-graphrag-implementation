"""Normalized-Levenshtein candidate suggestion, used by the Verification Module.

Implements the first step of the paper's two-step hallucination recovery
process: "retrieves candidate replacements based on the normalized
Levenshtein similarity ratio". The second step (LLM-based semantic ranking)
lives in `agents/verification.py`, which consumes these candidates.
"""

from pydantic import BaseModel
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein


class FuzzyCandidate(BaseModel):
    value: str
    score: float  # normalized similarity ratio, 0-100


def suggest_candidates(
    query_value: str,
    known_values: list[str],
    *,
    top_k: int = 5,
    min_score: float = 60.0,
) -> list[FuzzyCandidate]:
    """Rank `known_values` by normalized Levenshtein similarity to `query_value`."""
    if not known_values:
        return []
    matches = process.extract(
        query_value,
        known_values,
        scorer=Levenshtein.normalized_similarity,
        limit=top_k,
    )
    # Levenshtein.normalized_similarity returns 0-1; scale to 0-100 to match
    # the paper's reported Levenshtein similarity scores (e.g. "86.66").
    return [
        FuzzyCandidate(value=value, score=round(score * 100, 2))
        for value, score, _ in matches
        if score * 100 >= min_score
    ]
