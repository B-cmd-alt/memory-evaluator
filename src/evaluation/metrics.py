"""Aggregate metrics computation.

Computes per-cell and per-architecture aggregate metrics
from individual evaluation results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.evaluation.nugget_judge import NuggetEvalResult


@dataclass
class CellMetrics:
    """Aggregated metrics for a single (architecture, scale, task) cell."""

    architecture: str
    scale: str
    task_type: str
    n_questions: int

    # Primary metrics
    nugget_accuracy_mean: float
    nugget_accuracy_std: float
    nugget_accuracy_ci_low: float
    nugget_accuracy_ci_high: float

    # Secondary metrics
    token_f1_mean: float
    token_f1_std: float

    # Conflict resolution (T4 only)
    cr_single_hop_accuracy: float | None = None
    cr_multi_hop_accuracy: float | None = None

    # Abstention (T5 only)
    abstention_precision: float | None = None
    abstention_recall: float | None = None
    abstention_f1: float | None = None

    # Efficiency
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    cost_per_query_usd: float = 0.0
    retrieval_tier_distribution: dict[str, int] = field(default_factory=dict)


def compute_cell_metrics(
    architecture: str,
    scale: str,
    task_type: str,
    eval_results: list[NuggetEvalResult],
    latencies: list[float],
    costs: list[float],
    tiers: list[str],
) -> CellMetrics:
    """Compute aggregated metrics for a cell."""
    import numpy as np
    from scipy.stats import bootstrap

    n = len(eval_results)
    if n == 0:
        return CellMetrics(
            architecture=architecture,
            scale=scale,
            task_type=task_type,
            n_questions=0,
            nugget_accuracy_mean=0,
            nugget_accuracy_std=0,
            nugget_accuracy_ci_low=0,
            nugget_accuracy_ci_high=0,
            token_f1_mean=0,
            token_f1_std=0,
        )

    na_scores = np.array([r.nugget_accuracy for r in eval_results])
    f1_scores = np.array([r.token_f1 for r in eval_results])

    # Bootstrap CI for nugget accuracy
    ci_low, ci_high = 0.0, 0.0
    if len(na_scores) > 1:
        try:
            ci = bootstrap(
                (na_scores,),
                np.mean,
                n_resamples=10_000,
                confidence_level=0.95,
                method="BCa",
            )
            ci_low = ci.confidence_interval.low
            ci_high = ci.confidence_interval.high
        except Exception:
            ci_low = float(na_scores.mean() - 1.96 * na_scores.std() / np.sqrt(n))
            ci_high = float(na_scores.mean() + 1.96 * na_scores.std() / np.sqrt(n))

    # Latency percentiles
    sorted_lat = sorted(latencies) if latencies else [0]
    p50 = sorted_lat[len(sorted_lat) // 2]
    p95 = sorted_lat[int(len(sorted_lat) * 0.95)]

    # Tier distribution
    tier_dist: dict[str, int] = {}
    for t in tiers:
        tier_dist[t] = tier_dist.get(t, 0) + 1

    # Abstention metrics for T5
    abs_precision = abs_recall = abs_f1 = None
    if task_type == "T5":
        tp = sum(1 for r in eval_results if r.abstention_detected and r.nugget_accuracy < 0.3)
        fp = sum(1 for r in eval_results if r.abstention_detected and r.nugget_accuracy >= 0.3)
        fn = sum(1 for r in eval_results if not r.abstention_detected and r.nugget_accuracy < 0.3)
        abs_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        abs_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        abs_f1 = (
            2 * abs_precision * abs_recall / (abs_precision + abs_recall)
            if (abs_precision + abs_recall) > 0
            else 0.0
        )

    return CellMetrics(
        architecture=architecture,
        scale=scale,
        task_type=task_type,
        n_questions=n,
        nugget_accuracy_mean=float(na_scores.mean()),
        nugget_accuracy_std=float(na_scores.std()),
        nugget_accuracy_ci_low=ci_low,
        nugget_accuracy_ci_high=ci_high,
        token_f1_mean=float(f1_scores.mean()),
        token_f1_std=float(f1_scores.std()),
        abstention_precision=abs_precision,
        abstention_recall=abs_recall,
        abstention_f1=abs_f1,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        cost_per_query_usd=sum(costs) / n if n > 0 else 0.0,
        retrieval_tier_distribution=tier_dist,
    )
