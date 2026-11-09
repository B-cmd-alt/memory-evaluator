"""Architecture D: Hybrid Memory.

Combines A (Structured) + B (Graph) + C (Vector RAG) with
Reciprocal Rank Fusion (RRF) to merge results from all three
retrieval channels, then LLM re-ranks the fused candidates.
"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime

from src.architectures.base import (
    GENERATION_PROMPT,
    IngestStats,
    Memory,
    MemorySystem,
    QueryResult,
    RetrievedItem,
    SystemStats,
)
from src.architectures.graph import GraphMemory
from src.architectures.structured import StructuredMemory
from src.architectures.vector_rag import VectorRAGMemory
from src.evaluation.cost_tracker import CostTracker
from src.evaluation.latency_tracker import LatencyTracker
from src.llm_client import chat_completion, chat_completion_json


class HybridMemory(MemorySystem):
    """Architecture D: Triple retrieval with Reciprocal Rank Fusion."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.rrf_k = config.get("rrf_k", 60)
        self.top_k = config.get("top_k", 5)
        self.cost_tracker = CostTracker()
        self.latency_tracker = LatencyTracker()

        # Initialize sub-architectures
        self._structured = StructuredMemory(config.get("structured", {}))
        self._graph = GraphMemory(config.get("graph", {}))
        self._vector = VectorRAGMemory(config.get("vector", {}))

    async def ingest(self, memories: list[Memory]) -> IngestStats:
        """Ingest into all three sub-systems."""
        start = time.perf_counter()

        stats_a = await self._structured.ingest(memories)
        stats_b = await self._graph.ingest(memories)
        stats_c = await self._vector.ingest(memories)

        duration = time.perf_counter() - start
        self._total_ingestion_time += duration

        total_llm = stats_a.llm_calls + stats_b.llm_calls + stats_c.llm_calls
        total_embed = stats_a.embedding_calls + stats_b.embedding_calls + stats_c.embedding_calls
        self._total_llm_calls += total_llm
        self._total_embedding_calls += total_embed

        return IngestStats(
            memories_ingested=len(memories),
            llm_calls=total_llm,
            embedding_calls=total_embed,
            duration_s=duration,
            cost_usd=0.0,
        )

    def _reciprocal_rank_fusion(
        self, result_lists: list[list[RetrievedItem]]
    ) -> list[RetrievedItem]:
        """Merge multiple ranked lists using RRF.

        RRF score for document d:
            score(d) = sum(1 / (k + rank_i(d))) for each list i
        """
        scores: dict[str, float] = defaultdict(float)
        items: dict[str, RetrievedItem] = {}

        for ranked_list in result_lists:
            for rank, item in enumerate(ranked_list):
                key = item.content[:100]  # Use content prefix as dedup key
                scores[key] += 1.0 / (self.rrf_k + rank + 1)
                if key not in items:
                    items[key] = item

        # Sort by RRF score descending
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

        fused = []
        for key in sorted_keys:
            item = items[key]
            item.score = scores[key]
            item.source = "hybrid_rrf"
            fused.append(item)

        return fused

    async def query(
        self, question: str, timestamp: datetime | None = None
    ) -> QueryResult:
        start = time.perf_counter()

        # Retrieve from all three systems
        with self.latency_tracker.measure("retrieval"):
            result_a = await self._structured.query(question, timestamp)
            result_b = await self._graph.query(question, timestamp)
            result_c = await self._vector.query(question, timestamp)

            # Fuse results using RRF
            fused = self._reciprocal_rank_fusion(
                [
                    result_a.retrieved_context,
                    result_b.retrieved_context,
                    result_c.retrieved_context,
                ]
            )

        # LLM re-rank: pick best top-K from fused candidates
        candidates_text = "\n".join(
            f"[{i}] {item.content}" for i, item in enumerate(fused[: self.top_k * 3])
        )

        rerank_result = await chat_completion_json(
            messages=[
                {
                    "role": "user",
                    "content": f"""Given this question and candidate memory items, select the {self.top_k} most relevant items.

Question: {question}

Candidates:
{candidates_text}

Return JSON: {{"selected_indices": [list of indices, most relevant first]}}""",
                }
            ],
            purpose="rerank",
            cost_tracker=self.cost_tracker,
        )
        self._total_llm_calls += 1

        selected_indices = rerank_result.get("selected_indices", list(range(self.top_k)))
        final_items = []
        context_parts = []
        for idx in selected_indices[: self.top_k]:
            if idx < len(fused):
                final_items.append(fused[idx])
                context_parts.append(fused[idx].content)

        context = "\n".join(context_parts)

        # Generate answer
        with self.latency_tracker.measure("inference"):
            answer = await chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": GENERATION_PROMPT.format(
                            context=context, question=question
                        ),
                    }
                ],
                purpose="inference",
                cost_tracker=self.cost_tracker,
            )
        self._total_llm_calls += 1

        total_ms = (time.perf_counter() - start) * 1000

        return QueryResult(
            answer=answer,
            retrieved_context=final_items,
            retrieval_tier="hybrid_rrf",
            latency_ms=total_ms,
            input_tokens=0,
            output_tokens=0,
            embedding_tokens=0,
            cost_usd=0.0,
        )

    async def get_stats(self) -> SystemStats:
        stats_a = await self._structured.get_stats()
        stats_b = await self._graph.get_stats()
        stats_c = await self._vector.get_stats()

        return SystemStats(
            memory_count=stats_a.memory_count,  # same data in all three
            storage_bytes=stats_a.storage_bytes + stats_c.storage_bytes,
            index_entries=stats_a.index_entries + stats_b.index_entries + stats_c.index_entries,
            total_ingestion_time_s=self._total_ingestion_time,
            total_llm_calls=self._total_llm_calls,
            total_embedding_calls=self._total_embedding_calls,
        )

    async def reset(self) -> None:
        await self._structured.reset()
        await self._graph.reset()
        await self._vector.reset()
        self._total_ingestion_time = 0.0
        self._total_llm_calls = 0
        self._total_embedding_calls = 0
        self.cost_tracker.reset()
        self.latency_tracker.reset()

    async def close(self) -> None:
        await self._structured.close()
        await self._graph.close()
        await self._vector.close()
