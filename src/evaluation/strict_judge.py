"""Strict-judge v3 — asymmetric abstention heuristic for memory benchmarks.

This module implements the judge protocol described in `METHODOLOGY.md`. It
addresses three layered failure modes of standard LLM-as-judge for memory
benchmarks: abstention reward bias, heuristic blind spots, and own-generation
preference.

Usage:

    from src.evaluation.strict_judge import strict_judge_v3

    score, reasoning, mode = strict_judge_v3(
        question="What is the project deadline?",
        answer="The deadline is March 15, 2026.",
        reference="March 15, 2026",
        llm_judge_fn=my_llm_call,  # optional; required for non-deterministic modes
    )

    # Or use only the deterministic branches:
    score, reasoning, mode = strict_judge_v3(
        question=q, answer=a, reference=r,
        llm_judge_fn=None,        # returns mode="needs_llm_judge" if a call is required
    )

The two phrase lists are exposed as module constants so callers can extend
them for benchmark-specific reference styles.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

# Conservative — used to detect abstention in the GENERATED ANSWER.
# Identical to the phrase list in the original strict judge (run_strict_judge.py).
# Kept narrow to avoid false-positive matches on substantive answers that include
# explanatory hedging clauses (e.g. "Carol is X. The context doesn't mention Y.").
ANS_ABS_PHRASES = [
    "don't have enough", "do not have enough", "not enough information",
    "no information", "cannot determine", "i don't know",
    "not mentioned", "not specified", "not provided",
    "unable to determine", "insufficient information",
]

# Broad — used to detect abstention in the REFERENCE.
# Superset of ANS_ABS_PHRASES with patterns that appear in LongMemEval _abs
# references and other natural reference styles, but would cause false positives
# if applied to substantive answers.
REF_ABS_PHRASES = ANS_ABS_PHRASES + [
    "is not enough", "isn't enough",
    "cannot be determined", "could not be determined", "can't be determined",
    "not available", "no record of", "no record",
    "did not mention", "didn't mention", "do not mention", "doesn't mention",
    "no specific information", "without specific",
    "have no info", "have not been provided", "haven't been provided",
    "no details", "without details",
]


def is_ref_abstention(text: str, phrases: Optional[list[str]] = None) -> bool:
    """Heuristic: is this REFERENCE text an abstention?

    Uses the broad REF_ABS_PHRASES list by default. Pass `phrases` to override.
    """
    if not text:
        return False
    t = text.lower()
    return any(p in t for p in (phrases if phrases is not None else REF_ABS_PHRASES))


def is_ans_abstention(text: str, phrases: Optional[list[str]] = None) -> bool:
    """Heuristic: is this GENERATED ANSWER text an abstention?

    Uses the narrow ANS_ABS_PHRASES list by default. Pass `phrases` to override.
    """
    if not text:
        return False
    t = text.lower()
    return any(p in t for p in (phrases if phrases is not None else ANS_ABS_PHRASES))


# Prompt templates for the two LLM-judge branches.
REF_ABSTENTION_JUDGE_PROMPT = """The reference indicates the information is not available. Check if the generated answer fabricates content or correctly says it doesn't know.

Question: {question}
Reference: {reference}
Generated: {answer}

Return JSON: {{"score": 0.0_to_1.0, "reasoning": "brief"}}
- 1.0 if generated also indicates it doesn't know
- 0.0 if generated fabricates a specific answer"""

NORMAL_JUDGE_PROMPT = """Score how well the generated answer matches the reference.

Question: {question}
Reference: {reference}
Generated: {answer}

The reference contains a substantive answer. Score:
- 1.0: All key facts from reference present in generated
- 0.75: Most facts present, minor details missing
- 0.5: Some facts present, important ones missing
- 0.25: Few facts, mostly wrong
- 0.0: Wrong, irrelevant, or contradicts reference

Return JSON: {{"score": 0.0, "reasoning": "brief"}}"""


def strict_judge_v3(
    question: str,
    answer: str,
    reference: str,
    llm_judge_fn: Optional[Callable[[str], dict]] = None,
    *,
    ref_phrases: Optional[list[str]] = None,
    ans_phrases: Optional[list[str]] = None,
    cached_normal_score: Optional[float] = None,
) -> Tuple[float, str, str]:
    """Run the v3 asymmetric strict judge on one (question, answer, reference) triple.

    Args:
        question, answer, reference: the standard triple.
        llm_judge_fn: a callable taking a prompt string and returning a dict
            with at least a "score" key (float 0-1) and optionally "reasoning".
            Required for non-deterministic branches; if None, those branches
            return mode="needs_llm_judge" so the caller can route them.
        ref_phrases / ans_phrases: optional overrides for the abstention
            heuristic phrase lists (useful for benchmark-specific calibration).
        cached_normal_score: if you already have a normal_judge score for this
            item from a previous run (e.g., re-judging with a patched heuristic
            where the normal-judge branch is unchanged), pass it here to avoid
            an LLM call. Only applied when this run also routes to normal_judge.

    Returns:
        (score, reasoning, mode) tuple.
        mode is one of: "abstention_correct", "false_abstention",
            "ref_abstention_judge", "normal_judge", "needs_llm_judge".
    """
    ref_abs = is_ref_abstention(reference, ref_phrases)
    ans_abs = is_ans_abstention(answer, ans_phrases)

    # Branch 1: both abstain — correct abstention (deterministic, no LLM call).
    if ref_abs and ans_abs:
        return 1.0, "REF=abstention, ANS=abstention", "abstention_correct"

    # Branch 2: ref is abstention, answer is substantive — check if fabricated.
    if ref_abs and not ans_abs:
        if llm_judge_fn is None:
            return 0.0, "needs LLM judge for ref-abstention case", "needs_llm_judge"
        prompt = REF_ABSTENTION_JUDGE_PROMPT.format(
            question=question, reference=reference, answer=answer
        )
        r = llm_judge_fn(prompt)
        return (
            float(r.get("score", 0)),
            str(r.get("reasoning", "")),
            "ref_abstention_judge",
        )

    # Branch 3: ref is substantive, answer is abstention — retrieval failure.
    if not ref_abs and ans_abs:
        return (
            0.0,
            "REF has answer, ANS abstained — FAILED RETRIEVAL",
            "false_abstention",
        )

    # Branch 4: both substantive — normal content-match judging.
    if cached_normal_score is not None:
        return cached_normal_score, "(cached normal_judge)", "normal_judge"
    if llm_judge_fn is None:
        return 0.0, "needs LLM judge for normal case", "needs_llm_judge"
    prompt = NORMAL_JUDGE_PROMPT.format(
        question=question, reference=reference, answer=answer
    )
    r = llm_judge_fn(prompt)
    return float(r.get("score", 0)), str(r.get("reasoning", "")), "normal_judge"


def explain() -> str:
    """Return a brief docstring of the protocol for embedding in result artifacts."""
    return (
        "strict_judge_v3: asymmetric abstention heuristic "
        f"(REF list: {len(REF_ABS_PHRASES)} phrases, ANS list: {len(ANS_ABS_PHRASES)} phrases). "
        "Branches: abstention_correct / false_abstention (deterministic), "
        "ref_abstention_judge / normal_judge (require LLM judge). "
        "See METHODOLOGY.md for derivation."
    )


__all__ = [
    "strict_judge_v3",
    "is_ref_abstention",
    "is_ans_abstention",
    "REF_ABS_PHRASES",
    "ANS_ABS_PHRASES",
    "REF_ABSTENTION_JUDGE_PROMPT",
    "NORMAL_JUDGE_PROMPT",
    "explain",
]
