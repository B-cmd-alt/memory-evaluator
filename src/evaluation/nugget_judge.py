"""BEAM-style nugget evaluation using LLM-as-judge.

Decomposes reference answers into atomic nuggets, then scores
each nugget's presence in the generated answer (0, 0.5, or 1).
Primary dependent variable for the experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.evaluation.cost_tracker import CostTracker
from src.llm_client import chat_completion_json


@dataclass
class NuggetScore:
    """Score for a single nugget."""

    nugget: str
    score: float  # 0, 0.5, or 1.0
    reasoning: str


@dataclass
class NuggetEvalResult:
    """Full evaluation result for a single question."""

    question_id: str
    nugget_accuracy: float  # mean of nugget scores
    nugget_scores: list[NuggetScore]
    token_f1: float
    abstention_detected: bool


async def decompose_into_nuggets(
    reference_answer: str,
    cost_tracker: CostTracker | None = None,
) -> list[str]:
    """Decompose a reference answer into atomic nuggets.

    Each nugget is a single, independently verifiable fact.
    """
    result = await chat_completion_json(
        messages=[
            {
                "role": "user",
                "content": f"""Decompose this reference answer into atomic facts (nuggets).
Each nugget should be a single, independently verifiable statement.

Reference answer: {reference_answer}

Return JSON: {{"nuggets": ["nugget1", "nugget2", ...]}}""",
            }
        ],
        model="judge",
        purpose="judge",
        cost_tracker=cost_tracker,
    )
    return result.get("nuggets", [reference_answer])


async def score_nuggets(
    question: str,
    generated_answer: str,
    nuggets: list[str],
    cost_tracker: CostTracker | None = None,
) -> list[NuggetScore]:
    """Score each nugget's presence in the generated answer.

    Scores:
    - 1.0: nugget is fully present and correct
    - 0.5: nugget is partially present or implied
    - 0.0: nugget is absent or contradicted
    """
    nuggets_text = "\n".join(f"[{i}] {n}" for i, n in enumerate(nuggets))

    result = await chat_completion_json(
        messages=[
            {
                "role": "user",
                "content": f"""Score how well the generated answer covers each nugget from the reference.

Question: {question}

Generated answer: {generated_answer}

Reference nuggets:
{nuggets_text}

For each nugget, score:
- 1.0: fully present and correct in the generated answer
- 0.5: partially present or implied but not explicit
- 0.0: absent, wrong, or contradicted

Return JSON: {{"scores": [{{"index": 0, "score": 1.0, "reasoning": "..."}}, ...]}}""",
            }
        ],
        model="judge",
        purpose="judge",
        cost_tracker=cost_tracker,
    )

    scores = []
    for item in result.get("scores", []):
        idx = item.get("index", 0)
        nugget_text = nuggets[idx] if idx < len(nuggets) else ""
        scores.append(
            NuggetScore(
                nugget=nugget_text,
                score=item.get("score", 0.0),
                reasoning=item.get("reasoning", ""),
            )
        )

    # Fill in any missing nuggets with score 0
    scored_indices = {item.get("index") for item in result.get("scores", [])}
    for i, nugget in enumerate(nuggets):
        if i not in scored_indices:
            scores.append(NuggetScore(nugget=nugget, score=0.0, reasoning="not scored"))

    return scores


def compute_token_f1(generated: str, reference: str) -> float:
    """Compute token-level F1 between generated and reference answers."""
    gen_tokens = set(generated.lower().split())
    ref_tokens = set(reference.lower().split())

    if not gen_tokens or not ref_tokens:
        return 0.0

    intersection = gen_tokens & ref_tokens
    precision = len(intersection) / len(gen_tokens) if gen_tokens else 0.0
    recall = len(intersection) / len(ref_tokens) if ref_tokens else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def detect_abstention(answer: str) -> bool:
    """Detect if the answer is an abstention (refusing to answer)."""
    abstention_phrases = [
        "i don't have enough information",
        "i cannot answer",
        "i don't know",
        "not enough context",
        "no information available",
        "unable to determine",
        "insufficient information",
    ]
    lower = answer.lower()
    return any(phrase in lower for phrase in abstention_phrases)


async def evaluate_single(
    question_id: str,
    question: str,
    generated_answer: str,
    reference_answer: str,
    nuggets: list[str] | None = None,
    cost_tracker: CostTracker | None = None,
) -> NuggetEvalResult:
    """Full evaluation of a single question-answer pair."""
    # Decompose if nuggets not provided
    if not nuggets:
        nuggets = await decompose_into_nuggets(reference_answer, cost_tracker)

    # Score nuggets
    nugget_scores = await score_nuggets(
        question, generated_answer, nuggets, cost_tracker
    )

    # Compute nugget accuracy (primary metric)
    if nugget_scores:
        na = sum(ns.score for ns in nugget_scores) / len(nugget_scores)
    else:
        na = 0.0

    # Token F1 (secondary metric)
    f1 = compute_token_f1(generated_answer, reference_answer)

    # Abstention detection
    abstained = detect_abstention(generated_answer)

    return NuggetEvalResult(
        question_id=question_id,
        nugget_accuracy=na,
        nugget_scores=nugget_scores,
        token_f1=f1,
        abstention_detected=abstained,
    )
