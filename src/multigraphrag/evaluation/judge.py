"""Optional LLM-as-a-judge scoring for the CypherBench evaluation harness.

The paper scores answer correctness with a dedicated judge model (GigaChat 2
MAX) comparing the pipeline's natural language answer against the gold
`answer_json`, following "the LLM-as-a-judge framework (Tiwari et al. 2025)".
Its exact judge prompt (including the few-shot examples it mentions) is not
published, so this is an independent reconstruction of the same idea, not a
reproduction of the paper's specific prompt -- treat verdicts as a second,
model-based scoring option alongside the deterministic heuristic in
`matching.py`, not as a like-for-like replication of the paper's numbers.
"""

import json

from pydantic import BaseModel, Field

from multigraphrag.llm.base import LLMClient

JUDGE_SYSTEM_PROMPT = """\
You are an impartial judge scoring the output of a text-to-Cypher question-answering \
system. You are given the user's question, the ground-truth answer (as JSON rows from \
directly executing a gold Cypher query), and a candidate natural language answer \
produced by the system under test.

Judge whether the candidate answer is semantically correct: it must convey the same \
facts as the ground truth, even if phrased differently, formatted differently, or \
including reasonable additional context. Minor formatting differences (units, casing, \
number formatting) do not make an answer incorrect. Missing facts, wrong facts, or a \
contradictory answer make it incorrect. If the ground truth is empty/null, the answer \
is correct only if it also indicates no result/nothing found.
"""


class JudgeVerdict(BaseModel):
    correct: bool = Field(
        description="Whether the candidate answer is semantically equivalent to the ground truth."
    )
    reasoning: str = Field(description="One or two sentences justifying the verdict.")


class LLMJudge:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def judge(self, *, question: str, answer: str, gold_answer_rows: list) -> JudgeVerdict:
        user_prompt = (
            f"Question: {question}\n\n"
            f"Ground truth (JSON rows from the gold Cypher query): {json.dumps(gold_answer_rows, default=str)}\n\n"
            f"Candidate answer: {answer}\n\n"
            "Is the candidate answer semantically correct?"
        )
        return await self._llm.complete_structured(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=JudgeVerdict,
        )
