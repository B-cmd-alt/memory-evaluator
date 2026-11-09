"""Latency tracking for experiment runs.

Provides a context manager to measure wall-clock time for
retrieval and inference phases separately.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class LatencyRecord:
    """A single latency measurement."""

    phase: str  # 'retrieval', 'inference', 'ingestion', 'consolidation'
    duration_ms: float
    metadata: dict = field(default_factory=dict)


class LatencyTracker:
    """Accumulates latency measurements and computes statistics."""

    def __init__(self) -> None:
        self.records: list[LatencyRecord] = []

    @contextmanager
    def measure(self, phase: str, **metadata):
        """Context manager to measure and record duration of a phase.

        Usage:
            with tracker.measure("retrieval", query_id="q1"):
                results = await memory.search(query)
        """
        start = time.perf_counter()
        yield
        duration_ms = (time.perf_counter() - start) * 1000
        self.records.append(LatencyRecord(phase=phase, duration_ms=duration_ms, metadata=metadata))

    def percentile(self, phase: str, p: float) -> float:
        """Compute p-th percentile for a given phase."""
        durations = sorted(r.duration_ms for r in self.records if r.phase == phase)
        if not durations:
            return 0.0
        idx = int(len(durations) * p / 100)
        idx = min(idx, len(durations) - 1)
        return durations[idx]

    def summary(self, phase: str) -> dict:
        durations = [r.duration_ms for r in self.records if r.phase == phase]
        if not durations:
            return {"count": 0, "p50": 0, "p95": 0, "p99": 0, "mean": 0}
        durations.sort()
        n = len(durations)
        return {
            "count": n,
            "p50": durations[n // 2],
            "p95": durations[int(n * 0.95)],
            "p99": durations[int(n * 0.99)],
            "mean": sum(durations) / n,
        }

    def reset(self) -> None:
        self.records.clear()
