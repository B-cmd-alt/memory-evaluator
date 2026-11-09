"""Architecture C: Vector DB / RAG.

ChromaDB + bge-small-en-v1.5 embeddings, top-K=5 retrieval.
Traditional RAG baseline — embed and retrieve, no consolidation.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

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
from src.llm_client import chat_completion

CHROMA_PATH = Path("results/.scratch/chroma_db")


class VectorRAGMemory(MemorySystem):
    """Architecture C: ChromaDB + embeddings, pure RAG baseline."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.chunk_size = config.get("chunk_size", 512)
        self.top_k = config.get("top_k", 5)
        self.collection_prefix = config.get("collection_prefix", "experiment")
        self.cost_tracker = CostTracker()
        self.latency_tracker = LatencyTracker()

        # Initialize embedding model (local, zero API cost)
        self._embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

        # Initialize ChromaDB
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        self._chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self._collection_name = f"{self.collection_prefix}_{uuid.uuid4().hex[:8]}"
        self._collection = self._chroma_client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using local bge-small model."""
        self._total_embedding_calls += len(texts)
        embeddings = self._embed_model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    async def ingest(self, memories: list[Memory]) -> IngestStats:
        start = time.perf_counter()

        documents = []
        metadatas = []
        ids = []

        for mem in memories:
            doc = mem.content
            metadata = {
                "entities": json.dumps(mem.entities),
                "topics": json.dumps(mem.topics),
                "importance": mem.importance,
                "source_id": mem.source_id,
                "timestamp": mem.timestamp.isoformat(),
            }
            documents.append(doc)
            metadatas.append(metadata)
            ids.append(f"mem_{mem.source_id}_{uuid.uuid4().hex[:8]}")

        # Embed and upsert in batches
        batch_size = 64
        embed_calls = 0
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            embeddings = self._embed(batch_docs)
            embed_calls += len(batch_docs)

            self._collection.add(
                documents=batch_docs,
                embeddings=embeddings,
                metadatas=batch_meta,
                ids=batch_ids,
            )

        duration = time.perf_counter() - start
        self._total_ingestion_time += duration

        return IngestStats(
            memories_ingested=len(memories),
            llm_calls=0,
            embedding_calls=embed_calls,
            duration_s=duration,
            cost_usd=0.0,  # local embeddings = free
        )

    async def query(
        self, question: str, timestamp: datetime | None = None
    ) -> QueryResult:
        start = time.perf_counter()

        # Embed query
        with self.latency_tracker.measure("retrieval"):
            query_embedding = self._embed([question])[0]

            # Retrieve top-K
            where_filter = None
            if timestamp:
                where_filter = {"timestamp": {"$lte": timestamp.isoformat()}}

            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=self.top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

        retrieved = []
        context_parts = []

        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                item = RetrievedItem(
                    content=doc,
                    source="chromadb",
                    score=1.0 - dist,  # cosine distance to similarity
                    timestamp=(
                        datetime.fromisoformat(meta["timestamp"])
                        if meta.get("timestamp")
                        else None
                    ),
                )
                retrieved.append(item)
                context_parts.append(doc)

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

        total_ms = (time.perf_counter() - start) * 1000

        return QueryResult(
            answer=answer,
            retrieved_context=retrieved,
            retrieval_tier="vector",
            latency_ms=total_ms,
            input_tokens=0,
            output_tokens=0,
            embedding_tokens=0,
            cost_usd=0.0,
        )

    async def get_stats(self) -> SystemStats:
        count = self._collection.count()
        # Approximate storage size
        db_size = sum(
            f.stat().st_size
            for f in CHROMA_PATH.rglob("*")
            if f.is_file()
        )

        return SystemStats(
            memory_count=count,
            storage_bytes=db_size,
            index_entries=count,
            total_ingestion_time_s=self._total_ingestion_time,
            total_llm_calls=self._total_llm_calls,
            total_embedding_calls=self._total_embedding_calls,
        )

    async def reset(self) -> None:
        self._chroma_client.delete_collection(self._collection_name)
        self._collection_name = f"{self.collection_prefix}_{uuid.uuid4().hex[:8]}"
        self._collection = self._chroma_client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._total_ingestion_time = 0.0
        self._total_llm_calls = 0
        self._total_embedding_calls = 0
        self.cost_tracker.reset()
        self.latency_tracker.reset()

    async def close(self) -> None:
        pass  # ChromaDB PersistentClient handles cleanup
