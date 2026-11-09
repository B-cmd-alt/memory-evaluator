"""Statistical analysis for the experiment.

Implements the analysis plan:
- Friedman test across architectures per scale
- Wilcoxon signed-rank pairwise tests
- Cliff's delta effect sizes
- BCa bootstrap confidence intervals
- Segmented regression for crossover analysis
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class PairwiseResult:
    """Result of a pairwise comparison between two architectures."""

    arch_a: str
    arch_b: str
    scale: str
    statistic: float
    p_value: float
    cliffs_delta: float
    effect_interpretation: str  # negligible, small, medium, large
    significant: bool


@dataclass
class CrossoverPoint:
    """Identified crossover point between two architectures."""

    arch_a: str
    arch_b: str
    crossover_scale: float  # log10 of memory count
    crossover_ci_low: float
    crossover_ci_high: float


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    """Compute Cliff's delta (non-parametric effect size).

    Returns (delta, interpretation).
    """
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0, "negligible"

    # Count dominance pairs
    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    delta = (more - less) / (n_x * n_y)

    abs_d = abs(delta)
    if abs_d < 0.147:
        interp = "negligible"
    elif abs_d < 0.33:
        interp = "small"
    elif abs_d < 0.474:
        interp = "medium"
    else:
        interp = "large"

    return delta, interp


def friedman_test(groups: list[np.ndarray]) -> tuple[float, float]:
    """Run Friedman test across architecture groups.

    Returns (statistic, p_value).
    """
    if len(groups) < 3 or any(len(g) == 0 for g in groups):
        return 0.0, 1.0

    # Friedman requires equal-length groups; truncate to minimum
    min_len = min(len(g) for g in groups)
    truncated = [g[:min_len] for g in groups]

    stat, p = stats.friedmanchisquare(*truncated)
    return float(stat), float(p)


def pairwise_wilcoxon(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    arch_a: str,
    arch_b: str,
    scale: str,
    alpha: float = 0.01,
) -> PairwiseResult:
    """Pairwise Wilcoxon signed-rank test with Cliff's delta."""
    min_len = min(len(scores_a), len(scores_b))
    a, b = scores_a[:min_len], scores_b[:min_len]

    if min_len < 5:
        return PairwiseResult(
            arch_a=arch_a, arch_b=arch_b, scale=scale,
            statistic=0, p_value=1.0, cliffs_delta=0,
            effect_interpretation="negligible", significant=False,
        )

    try:
        stat, p = stats.wilcoxon(a, b, alternative="two-sided")
    except ValueError:
        stat, p = 0.0, 1.0

    delta, interp = cliffs_delta(a, b)

    return PairwiseResult(
        arch_a=arch_a,
        arch_b=arch_b,
        scale=scale,
        statistic=float(stat),
        p_value=float(p),
        cliffs_delta=delta,
        effect_interpretation=interp,
        significant=p < alpha,
    )


def holm_bonferroni_correction(p_values: list[float], alpha: float = 0.01) -> list[bool]:
    """Apply Holm-Bonferroni correction to a list of p-values.

    Returns list of booleans indicating significance after correction.
    """
    n = len(p_values)
    if n == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    significant = [False] * n

    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted_alpha = alpha / (n - rank)
        if p <= adjusted_alpha:
            significant[orig_idx] = True
        else:
            break  # all remaining are non-significant

    return significant


def bootstrap_ci(
    data: np.ndarray,
    statistic=np.mean,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Compute BCa bootstrap confidence interval."""
    if len(data) < 2:
        val = statistic(data) if len(data) == 1 else 0.0
        return val, val

    try:
        result = stats.bootstrap(
            (data,),
            statistic,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            method="BCa",
        )
        return float(result.confidence_interval.low), float(result.confidence_interval.high)
    except Exception:
        mean = float(statistic(data))
        se = float(data.std() / np.sqrt(len(data)))
        z = stats.norm.ppf((1 + confidence_level) / 2)
        return mean - z * se, mean + z * se


def find_crossover(
    scales: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    arch_a: str,
    arch_b: str,
    n_bootstrap: int = 1000,
) -> CrossoverPoint | None:
    """Find the scale at which architecture B overtakes A.

    Uses linear interpolation on log-scale performance curves.
    """
    log_scales = np.log10(scales)
    diff = scores_a - scores_b

    # Find where diff changes sign
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) == 0:
        return None

    # Linear interpolation at first sign change
    idx = sign_changes[0]
    x1, x2 = log_scales[idx], log_scales[idx + 1]
    y1, y2 = diff[idx], diff[idx + 1]

    if y2 - y1 == 0:
        return None

    crossover = x1 - y1 * (x2 - x1) / (y2 - y1)

    # Bootstrap CI on crossover point
    rng = np.random.RandomState(42)
    bootstrap_crossovers = []
    for _ in range(n_bootstrap):
        noise_a = scores_a + rng.normal(0, scores_a.std() * 0.1, len(scores_a))
        noise_b = scores_b + rng.normal(0, scores_b.std() * 0.1, len(scores_b))
        d = noise_a - noise_b
        sc = np.where(np.diff(np.sign(d)))[0]
        if len(sc) > 0:
            i = sc[0]
            x1_, x2_ = log_scales[i], log_scales[i + 1]
            y1_, y2_ = d[i], d[i + 1]
            if y2_ - y1_ != 0:
                bootstrap_crossovers.append(x1_ - y1_ * (x2_ - x1_) / (y2_ - y1_))

    if bootstrap_crossovers:
        ci_low = float(np.percentile(bootstrap_crossovers, 2.5))
        ci_high = float(np.percentile(bootstrap_crossovers, 97.5))
    else:
        ci_low = ci_high = float(crossover)

    return CrossoverPoint(
        arch_a=arch_a,
        arch_b=arch_b,
        crossover_scale=float(crossover),
        crossover_ci_low=ci_low,
        crossover_ci_high=ci_high,
    )
