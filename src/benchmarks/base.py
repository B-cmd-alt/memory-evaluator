"""Abstract base interface for benchmark loaders.

Each benchmark (BEAM, LongMemEval, MemoryAgentBench, ConvoMem)
implements this interface to provide a unified way to load
memories and evaluation questions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from src.architectures.base import Memory, TaskType


@dataclass
class EvalQuestion:
    """A single evaluation question from a benchmark."""

    id: str
    question: str
    reference_answer: str
    task_type: TaskType
    # For temporal questions, the point-in-time to query at
    query_timestamp: str | None = None
    # For nugget evaluation: pre-decomposed atomic facts in the reference
    nuggets: list[str] = field(default_factory=list)
    # Metadata from the benchmark
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkDataset:
    """A complete dataset from a benchmark at a specific scale."""

    benchmark_name: str
    scale: str  # S0, S1, S2, S3, S4
    memories: list[Memory]
    questions: list[EvalQuestion]
    metadata: dict = field(default_factory=dict)


class Benchmark(ABC):
    """Abstract base for benchmark loaders."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    @property
    @abstractmethod
    def name(self) -> str:
        """Benchmark name (e.g., 'BEAM', 'LongMemEval')."""
        ...

    @abstractmethod
    async def load(self, scale: str) -> BenchmarkDataset:
        """Load benchmark data at a specific scale level.

        Args:
            scale: One of 'S0', 'S1', 'S2', 'S3', 'S4'.

        Returns:
            BenchmarkDataset with memories and evaluation questions.
        """
        ...

    @abstractmethod
    async def download(self) -> None:
        """Download benchmark data if not already present."""
        ...
