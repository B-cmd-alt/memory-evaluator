"""Scale breakpoint analysis.

Uses segmented (piecewise linear) regression to find the
scale at which each architecture's performance drops below
a threshold, and where one architecture overtakes another.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BreakpointResult:
    """Result of breakpoint analysis for a single architecture."""

    architecture: str
    breakpoint_scale: float  # log10 of memory count where degradation accelerates
    slope_before: float  # performance change rate before breakpoint
    slope_after: float  # performance change rate after breakpoint
    r_squared: float
    threshold_scale: float | None  # scale where NA drops below threshold


def find_degradation_breakpoint(
    scales: np.ndarray,
    scores: np.ndarray,
    architecture: str,
    threshold: float = 0.5,
) -> BreakpointResult:
    """Find where performance degradation accelerates.

    Uses piecewise linear fitting with 1 breakpoint.
    """
    log_scales = np.log10(scales)

    try:
        import pwlf

        pw = pwlf.PiecewiseLinFit(log_scales, scores)
        breaks = pw.fit(2)  # 2 segments = 1 breakpoint
        breakpoint = breaks[1] if len(breaks) > 1 else log_scales[-1]

        # Compute slopes
        slopes = pw.calc_slopes()
        slope_before = slopes[0] if len(slopes) > 0 else 0
        slope_after = slopes[1] if len(slopes) > 1 else 0

        # R-squared
        y_pred = pw.predict(log_scales)
        ss_res = np.sum((scores - y_pred) ** 2)
        ss_tot = np.sum((scores - scores.mean()) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    except ImportError:
        # Fallback: simple split at midpoint
        mid = len(log_scales) // 2
        if mid > 0 and mid < len(log_scales):
            slope_before = (scores[mid] - scores[0]) / (log_scales[mid] - log_scales[0]) if log_scales[mid] != log_scales[0] else 0
            slope_after = (scores[-1] - scores[mid]) / (log_scales[-1] - log_scales[mid]) if log_scales[-1] != log_scales[mid] else 0
        else:
            slope_before = slope_after = 0
        breakpoint = log_scales[mid] if mid < len(log_scales) else log_scales[-1]
        r_sq = 0

    # Find threshold crossing
    threshold_scale = None
    for i in range(len(scores)):
        if scores[i] < threshold:
            if i > 0:
                # Interpolate
                frac = (threshold - scores[i - 1]) / (scores[i] - scores[i - 1])
                threshold_scale = log_scales[i - 1] + frac * (log_scales[i] - log_scales[i - 1])
            else:
                threshold_scale = log_scales[0]
            break

    return BreakpointResult(
        architecture=architecture,
        breakpoint_scale=float(breakpoint),
        slope_before=float(slope_before),
        slope_after=float(slope_after),
        r_squared=float(r_sq),
        threshold_scale=float(threshold_scale) if threshold_scale is not None else None,
    )
