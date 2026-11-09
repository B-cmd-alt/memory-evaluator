"""Architecture A: Structured File-Based Memory.

SQLite + BM25 + LLM consolidation + markdown index.
Modeled on Google Always-On Memory Agent / Claude Code patterns.

No vector DB, no embeddings. Uses keyword search (BM25)
and LLM consolidation for knowledge compaction.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime
from pathlib import Path

from rank_bm25 import BM25Okapi

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

DB_PATH = Path("results/.scratch/structured_memory.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    entities TEXT NOT NULL,  -- JSON array
    topics TEXT NOT NULL,    -- JSON array
    importance REAL DEFAULT 3.0,
    source_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    created_at TEXT NOT NULL,
    consolidated INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    summary TEXT NOT NULL,
    last_modified TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, entities, topics,
    content='memories',
    content_rowid='id'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, entities, topics)
    VALUES (new.id, new.content, new.entities, new.topics);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, entities, topics)
    VALUES ('delete', old.id, old.content, old.entities, old.topics);
END;
"""


class StructuredMemory(MemorySystem):
    """Architecture A: SQLite + BM25 + LLM consolidation."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.consolidation_threshold = config.get("consolidation_threshold", 50)
        self.index_max_lines = config.get("index_max_lines", 200)
        self.bm25_top_k = config.get("bm25_top_k", 5)
        self._unconsolidated_count = 0
        self.cost_tracker = CostTracker()
        self.latency_tracker = LatencyTracker()
        self._init_db()

    def _init_db(self) -> None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(DB_PATH))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    async def ingest(self, memories: list[Memory]) -> IngestStats:
        start = time.perf_counter()
        llm_calls = 0

        for mem in memories:
            import json

            self.conn.execute(
                """INSERT INTO memories (content, entities, topics, importance,
                   source_id, timestamp, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    mem.content,
                    json.dumps(mem.entities),
                    json.dumps(mem.topics),
                    mem.importance,
                    mem.source_id,
                    mem.timestamp.isoformat(),
                    datetime.now().isoformat(),
                ),
            )
            self._unconsolidated_count += 1

        self.conn.commit()

        # Trigger consolidation if threshold reached
        if self._unconsolidated_count >= self.consolidation_threshold:
            llm_calls += await self._consolidate()

        duration = time.perf_counter() - start
        self._total_ingestion_time += duration
        self._total_llm_calls += llm_calls

        return IngestStats(
            memories_ingested=len(memories),
            llm_calls=llm_calls,
            embedding_calls=0,
            duration_s=duration,
            cost_usd=0.0,  # computed via cost_tracker
        )

    async def _consolidate(self) -> int:
        """Consolidate unconsolidated memories into topic summaries."""
        import json

        rows = self.conn.execute(
            "SELECT * FROM memories WHERE consolidated = 0 ORDER BY timestamp"
        ).fetchall()

        if not rows:
            return 0

        # Group by topic
        topic_groups: dict[str, list[dict]] = {}
        for row in rows:
            topics = json.loads(row["topics"])
            for topic in topics:
                topic_groups.setdefault(topic, []).append(dict(row))

        llm_calls = 0
        for topic, group_rows in topic_groups.items():
            facts = "\n".join(f"- {r['content']}" for r in group_rows)

            # Check for existing summary
            existing = self.conn.execute(
                "SELECT summary FROM summaries WHERE topic = ?", (topic,)
            ).fetchone()

            if existing:
                prompt = f"""Update this topic summary with new facts.

Existing summary:
{existing['summary']}

New facts:
{facts}

Write an updated summary that incorporates the new facts. If any new facts
contradict the existing summary, use the newer fact. Keep the summary concise."""
            else:
                prompt = f"""Summarize these facts about "{topic}" into a concise paragraph.

Facts:
{facts}

Write a clear, concise summary."""

            summary = await chat_completion(
                messages=[{"role": "user", "content": prompt}],
                purpose="consolidation",
                cost_tracker=self.cost_tracker,
            )
            llm_calls += 1

            if existing:
                self.conn.execute(
                    "UPDATE summaries SET summary = ?, last_modified = ? WHERE topic = ?",
                    (summary, datetime.now().isoformat(), topic),
                )
            else:
                self.conn.execute(
                    "INSERT INTO summaries (topic, summary, last_modified) VALUES (?, ?, ?)",
                    (topic, summary, datetime.now().isoformat()),
                )

        # Mark all as consolidated
        self.conn.execute("UPDATE memories SET consolidated = 1 WHERE consolidated = 0")
        self.conn.commit()
        self._unconsolidated_count = 0

        return llm_calls

    async def query(
        self, question: str, timestamp: datetime | None = None
    ) -> QueryResult:
        start = time.perf_counter()

        # BM25 search over FTS5
        with self.latency_tracker.measure("retrieval"):
            results = self.conn.execute(
                """SELECT m.content, m.entities, m.topics, m.timestamp,
                          rank as score
                   FROM memories_fts f
                   JOIN memories m ON f.rowid = m.id
                   WHERE memories_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (question, self.bm25_top_k),
            ).fetchall()

            # Also check topic summaries
            summary_results = self.conn.execute(
                """SELECT topic, summary FROM summaries
                   ORDER BY last_modified DESC LIMIT 5"""
            ).fetchall()

        retrieved = []
        context_parts = []

        for row in results:
            item = RetrievedItem(
                content=row["content"],
                source="memories_fts",
                score=abs(row["score"]),
                timestamp=datetime.fromisoformat(row["timestamp"]) if row["timestamp"] else None,
            )
            retrieved.append(item)
            context_parts.append(row["content"])

        for row in summary_results:
            context_parts.append(f"[Summary - {row['topic']}]: {row['summary']}")

        context = "\n".join(context_parts[:5])  # Normalize to top-5

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
            retrieval_tier="bm25",
            latency_ms=total_ms,
            input_tokens=0,  # tracked by cost_tracker
            output_tokens=0,
            embedding_tokens=0,
            cost_usd=0.0,
        )

    async def get_stats(self) -> SystemStats:
        count = self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        db_size = DB_PATH.stat().st_size if DB_PATH.exists() else 0
        summary_count = self.conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]

        return SystemStats(
            memory_count=count,
            storage_bytes=db_size,
            index_entries=summary_count,
            total_ingestion_time_s=self._total_ingestion_time,
            total_llm_calls=self._total_llm_calls,
            total_embedding_calls=0,
        )

    async def reset(self) -> None:
        self.conn.executescript("""
            DELETE FROM memories;
            DELETE FROM summaries;
            DELETE FROM memories_fts;
        """)
        self.conn.commit()
        self._unconsolidated_count = 0
        self._total_ingestion_time = 0.0
        self._total_llm_calls = 0
        self.cost_tracker.reset()
        self.latency_tracker.reset()

    async def close(self) -> None:
        self.conn.close()
