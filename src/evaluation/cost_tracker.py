"""Token counting and cost tracking for experiment runs.

Tracks all API calls (LLM inference + embeddings) and computes
per-query and per-run costs using the pricing config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class APICall:
    """A single API call record."""

    model: str
    input_tokens: int
    output_tokens: int
    purpose: str  # 'inference', 'judge', 'extraction', 'consolidation', 'embedding'
    cached_input_tokens: int = 0


@dataclass
class CostTracker:
    """Accumulates API calls and computes costs."""

    pricing: dict = field(default_factory=dict)
    calls: list[APICall] = field(default_factory=list)

    @classmethod
    def from_config(cls, pricing_path: Path) -> CostTracker:
        with open(pricing_path) as f:
            pricing = yaml.safe_load(f)
        return cls(pricing=pricing)

    def record(self, call: APICall) -> float:
        """Record an API call and return its cost in USD."""
        self.calls.append(call)
        return self._compute_cost(call)

    def _compute_cost(self, call: APICall) -> float:
        model_pricing = self.pricing.get("models", {}).get(call.model)
        if model_pricing:
            input_cost = (call.input_tokens / 1_000_000) * model_pricing["input"]
            cached_cost = (call.cached_input_tokens / 1_000_000) * model_pricing.get(
                "cached_input", model_pricing["input"]
            )
            output_cost = (call.output_tokens / 1_000_000) * model_pricing["output"]
            return input_cost + cached_cost + output_cost

        emb_pricing = self.pricing.get("embeddings", {}).get(call.model)
        if emb_pricing:
            return (call.input_tokens / 1_000_000) * emb_pricing["per_million_tokens"]

        return 0.0

    @property
    def total_cost(self) -> float:
        return sum(self._compute_cost(c) for c in self.calls)

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    def summary(self) -> dict:
        by_purpose: dict[str, float] = {}
        for call in self.calls:
            cost = self._compute_cost(call)
            by_purpose[call.purpose] = by_purpose.get(call.purpose, 0) + cost
        return {
            "total_cost_usd": self.total_cost,
            "total_calls": len(self.calls),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "cost_by_purpose": by_purpose,
        }

    def reset(self) -> None:
        self.calls.clear()
