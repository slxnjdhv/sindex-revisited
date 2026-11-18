"""Prompt templates (short-answer QA, following Farquhar et al. 2024 / SINdex)."""
from __future__ import annotations

from typing import Optional

QA_INSTRUCTION = "Answer the following question as briefly as possible."


def qa_prompt(question: str, context: Optional[str] = None) -> str:
    parts = [QA_INSTRUCTION]
    if context:
        parts.append(f"Context: {context}")
    parts.append(f"Question: {question}")
    parts.append("Answer:")
    return "\n".join(parts)


# v1 (Farquhar et al. style). Kept for reproducibility of the first runs.
JUDGE_TEMPLATE_V1 = (
    "We are assessing the quality of answers to the following question: {question}\n"
    "The expected answer is: {gold}\n"
    "The proposed answer is: {pred}\n"
    "Within the context of the question, does the proposed answer mean the same as the expected answer? "
    "Respond only with yes or no.\nResponse:"
)

# v2: explicit about extra words / full sentences, which v1 judges penalised
# (e.g. "Dora the Explorer's monkey is named Boots." vs gold "Boots").
JUDGE_TEMPLATE = (
    "You are grading a short answer to a quiz question.\n"
    "Question: {question}\n"
    "Correct answer(s): {gold}\n"
    "Proposed answer: {pred}\n\n"
    "The proposed answer is CORRECT if it states the correct answer or an equivalent "
    "(a synonym, alias, abbreviation, or the correct answer inside a longer sentence). "
    "Extra words or a full sentence are fine. It is WRONG if it gives a different entity, "
    "number or date, contradicts the correct answer, or does not answer the question.\n"
    "Is the proposed answer correct? Reply with exactly one word: yes or no."
)


def judge_prompt(question: str, gold: list, pred: str, version: int = 2) -> str:
    tpl = JUDGE_TEMPLATE if version == 2 else JUDGE_TEMPLATE_V1
    return tpl.format(question=question, gold=" OR ".join(gold[:5]), pred=pred)
