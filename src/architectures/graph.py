"""Architecture B: Graph-Based Memory.

Graphiti + Neo4j with bi-temporal edges, hybrid BM25+semantic search.
Entities and relationships are extracted by LLM and stored as a
temporal knowledge graph.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from src.architectures.base import (
    GENERATION_PROMPT,
    IngestStats,
    Memory,
    MemorySystem,
    QueryResult,
    RetrievedItem,
    SystemStats,
)
from src.evaluation.cost_tracker import CostTracker
from src.evaluation.latency_tracker import LatencyTracker
from src.llm_client import chat_completion, chat_completion_json


class GraphMemory(MemorySystem):
    """Architecture B: Graphiti + Neo4j temporal knowledge graph.

    Uses Graphiti's episode ingestion to extract entities and
    relationships with bi-temporal metadata. Retrieval uses
    Graphiti's hybrid BM25 + semantic search.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.neo4j_uri = config.get("neo4j_uri", "bolt://localhost:7687")
        self.neo4j_user = config.get("neo4j_user", "neo4j")
        self.neo4j_password = config.get("neo4j_password", "password")
        self.search_limit = config.get("search_limit", 5)
        self.cost_tracker = CostTracker()
        self.latency_tracker = LatencyTracker()
        self._graphiti = None
        self._episode_count = 0

    async def _ensure_graphiti(self):
        """Lazy-init Graphiti to avoid import-time Neo4j connection."""
        if self._graphiti is not None:
            return
        from graphiti_core import Graphiti

        self._graphiti = Graphiti(
            self.neo4j_uri,
            self.neo4j_user,
            self.neo4j_password,
        )
        await self._graphiti.build_indices_and_constraints()

    async def ingest(self, memories: list[Memory]) -> IngestStats:
        await self._ensure_graphiti()
        start = time.perf_counter()
        llm_calls = 0

        for mem in memories:
            # Ingest each memory as an episode
            # Graphiti handles entity extraction and relationship building
            await self._graphiti.add_episode(
                name=f"memory_{mem.source_id}",
                episode_body=mem.content,
                source_description="benchmark_memory",
                reference_time=mem.timestamp,
            )
            self._episode_count += 1
            # Graphiti makes ~3 LLM calls per episode
            # (entity extraction, relationship extraction, dedup)
            llm_calls += 3

        duration = time.perf_counter() - start
        self._total_ingestion_time += duration
        self._total_llm_calls += llm_calls

        return IngestStats(
            memories_ingested=len(memories),
            llm_calls=llm_calls,
            embedding_calls=len(memories),  # Graphiti embeds entities
            duration_s=duration,
            cost_usd=0.0,
        )

    async def query(
        self, question: str, timestamp: datetime | None = None
    ) -> QueryResult:
        await self._ensure_graphiti()
        start = time.perf_counter()

        # Search using Graphiti's hybrid retrieval
        with self.latency_tracker.measure("retrieval"):
            search_results = await self._graphiti.search(
                query=question,
                num_results=self.search_limit,
            )

        retrieved = []
        context_parts = []

        for edge in search_results:
            fact_text = edge.fact if hasattr(edge, "fact") else str(edge)
            item = RetrievedItem(
                content=fact_text,
                source="graphiti",
                score=getattr(edge, "score", 0.0),
                timestamp=getattr(edge, "valid_at", None),
            )
            retrieved.append(item)
            context_parts.append(fact_text)

        context = "\n".join(context_parts[:5])

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
            retrieved_context=retrieved,
            retrieval_tier="graph",
            latency_ms=total_ms,
            input_tokens=0,
            output_tokens=0,
            embedding_tokens=0,
            cost_usd=0.0,
        )

    async def get_stats(self) -> SystemStats:
        await self._ensure_graphiti()
        # Query Neo4j for node/edge counts
        from neo4j import AsyncGraphDatabase

        driver = AsyncGraphDatabase.driver(
            self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
        )
        async with driver.session() as session:
            node_count = (
                await session.run("MATCH (n) RETURN count(n) as c")
            )
            node_record = await node_count.single()
            nodes = node_record["c"] if node_record else 0

        await driver.close()

        return SystemStats(
            memory_count=self._episode_count,
            storage_bytes=0,  # Neo4j doesn't expose this simply
            index_entries=nodes,
            total_ingestion_time_s=self._total_ingestion_time,
            total_llm_calls=self._total_llm_calls,
            total_embedding_calls=self._total_embedding_calls,
        )

    async def reset(self) -> None:
        if self._graphiti:
            # Clear all Neo4j data
            from neo4j import AsyncGraphDatabase

            driver = AsyncGraphDatabase.driver(
                self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
            )
            async with driver.session() as session:
                await session.run("MATCH (n) DETACH DELETE n")
            await driver.close()

        self._episode_count = 0
        self._total_ingestion_time = 0.0
        self._total_llm_calls = 0
        self._total_embedding_calls = 0
        self.cost_tracker.reset()
        self.latency_tracker.reset()

    async def close(self) -> None:
        if self._graphiti:
            await self._graphiti.close()
            self._graphiti = None
