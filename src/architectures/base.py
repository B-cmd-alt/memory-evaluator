"""Abstract base interface for all memory architectures.

Every architecture (A through E) implements this interface, ensuring
controlled comparison: same inputs, same outputs, only the memory
mechanism differs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class TaskType(str, Enum):
    SINGLE_FACT = "T1"
    MULTI_SESSION = "T2"
    TEMPORAL = "T3"
    CONFLICT = "T4"
    ABSTENTION = "T5"


@dataclass
class Memory:
    """A single atomic memory unit, normalized across architectures."""

    content: str
    entities: list[str]
    topics: list[str]
    timestamp: datetime
    importance: float  # 1-5
    source_id: str
    # Optional temporal fields (used by architectures that support them)
    valid_until: datetime | None = None
    supersedes_id: str | None = None


@dataclass
class RetrievedItem:
    """A single item returned by the retrieval pipeline."""

    content: str
    source: str  # provenance: which table/store it came from
    score: float  # relevance score (architecture-specific)
    timestamp: datetime | None = None


@dataclass
class QueryResult:
    """The output of a memory query, including answer and instrumentation."""

    answer: str
    retrieved_context: list[RetrievedItem]
    retrieval_tier: str  # which retrieval tier was used (for Lattice: "index"/"bm25"/"deep")
    latency_ms: float
    input_tokens: int
    output_tokens: int
    embedding_tokens: int  # 0 for architectures that don't use embeddings
    cost_usd: float


@dataclass
class SystemStats:
    """Instrumentation stats for a memory system at a point in time."""

    memory_count: int
    storage_bytes: int
    index_entries: int  # how many items in the primary index
    total_ingestion_time_s: float
    total_llm_calls: int
    total_embedding_calls: int


@dataclass
class IngestStats:
    """Stats returned after an ingestion batch."""

    memories_ingested: int
    llm_calls: int
    embedding_calls: int
    duration_s: float
    cost_usd: float


class MemorySystem(ABC):
    """Abstract base for all memory architectures.

    Contract:
    - ingest() stores memories into the system
    - query() retrieves and answers using stored memories
    - get_stats() returns instrumentation data
    - reset() clears all state for a fresh experiment run
    """

    def __init__(self, config: dict):
        self.config = config
        self._total_ingestion_time = 0.0
        self._total_llm_calls = 0
        self._total_embedding_calls = 0

    @abstractmethod
    async def ingest(self, memories: list[Memory]) -> IngestStats:
        """Ingest a batch of memories into the system.

        This includes any architecture-specific processing:
        - A: SQLite insert + BM25 index update + consolidation
        - B: Graphiti entity/relationship extraction + Neo4j writes
        - C: Embedding computation + ChromaDB upsert
        - D: All of the above
        - E: Raw source insert + staged mutations + consolidation cycle
        """
        ...

    @abstractmethod
    async def query(
        self, question: str, timestamp: datetime | None = None
    ) -> QueryResult:
        """Answer a question using stored memories.

        Args:
            question: The question to answer.
            timestamp: Optional point-in-time for temporal queries.
                       If provided, only facts valid at this time are considered.

        Returns:
            QueryResult with answer, retrieved context, and instrumentation.

        The generation prompt is standardized across architectures:
            "Given the following context, answer the question.
             Context: {retrieved_context}
             Question: {question}"
        """
        ...

    @abstractmethod
    async def get_stats(self) -> SystemStats:
        """Return current system instrumentation stats."""
        ...

    @abstractmethod
    async def reset(self) -> None:
        """Clear all state for a fresh experiment run.

        Must leave the system in a clean state equivalent to a fresh __init__.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release resources (DB connections, etc)."""
        ...


# Standard generation prompt — used by ALL architectures for answer generation.
# Only the retrieved_context differs between architectures.
GENERATION_PROMPT = """Given the following context retrieved from memory, answer the question.
If the context does not contain enough information to answer, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""
