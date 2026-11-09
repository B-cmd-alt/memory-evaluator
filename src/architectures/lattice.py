"""Architecture E: Lattice Memory (Novel Contribution).

Best-of-breed synthesis of 6 industry patterns:
1. Always-loaded compact index (Karpathy Wiki + Claude Code)
2. Bi-temporal fact records (Zep/Graphiti)
3. Agent-native tiered retrieval (ByteRover)
4. SQLite + FTS5 storage (Google Always-On)
5. Scheduled consolidation + adaptive merging (Google + Mem0)
6. Linting with contradiction tracking (Karpathy + A-MEM)

Design principle: expensive ingestion, cheap retrieval.
Zero vector DB, zero embeddings.
"""

from __future__ import annotations

import json
import sqlite3
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

DB_PATH = Path("results/.scratch/lattice_memory.db")

SCHEMA = """
-- Layer 1: Immutable raw inputs
CREATE TABLE IF NOT EXISTS raw_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    content TEXT NOT NULL,
    source_type TEXT DEFAULT 'memory',
    t_ingested TEXT NOT NULL,
    processed INTEGER DEFAULT 0
);

-- Layer 2: Atomic facts with bi-temporal fields
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    content TEXT NOT NULL,
    t_event TEXT,           -- when the fact was true (nullable = unknown)
    t_ingested TEXT NOT NULL,
    valid_until TEXT,       -- NULL = still valid
    confidence REAL DEFAULT 1.0,
    source_ids TEXT NOT NULL  -- JSON array of raw_source IDs
);

-- FTS5 index on facts for BM25 search (Tier 2)
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    topic, content,
    content='facts',
    content_rowid='id'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, topic, content) VALUES (new.id, new.topic, new.content);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, topic, content)
    VALUES ('delete', old.id, old.topic, old.content);
END;

CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, topic, content)
    VALUES ('delete', old.id, old.topic, old.content);
    INSERT INTO facts_fts(rowid, topic, content) VALUES (new.id, new.topic, new.content);
END;

-- Layer 3: Pre-synthesized wiki pages per topic
CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT UNIQUE NOT NULL,
    summary TEXT NOT NULL,       -- one-line summary for index.md
    body TEXT NOT NULL,          -- full markdown page
    last_modified TEXT NOT NULL,
    related_topics TEXT DEFAULT '[]'  -- JSON array
);

-- Typed bidirectional links between facts
CREATE TABLE IF NOT EXISTS links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id_a INTEGER NOT NULL REFERENCES facts(id),
    fact_id_b INTEGER NOT NULL REFERENCES facts(id),
    relation TEXT NOT NULL CHECK(relation IN (
        'supports', 'contradicts', 'causes', 'temporal_successor'
    )),
    t_created TEXT NOT NULL
);

-- Staged mutations (applied at next consolidation)
CREATE TABLE IF NOT EXISTS pending_mutations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation TEXT NOT NULL CHECK(operation IN (
        'ADD', 'UPDATE', 'UPSERT', 'DELETE', 'LINK'
    )),
    payload TEXT NOT NULL,   -- JSON
    t_queued TEXT NOT NULL,
    processed INTEGER DEFAULT 0
);

-- FTS on pending mutations for interim search
CREATE VIRTUAL TABLE IF NOT EXISTS pending_fts USING fts5(
    payload,
    content='pending_mutations',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS pending_ai AFTER INSERT ON pending_mutations BEGIN
    INSERT INTO pending_fts(rowid, payload) VALUES (new.id, new.payload);
END;

-- Lint reports
CREATE TABLE IF NOT EXISTS lint_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    t_run TEXT NOT NULL,
    report_type TEXT NOT NULL,  -- 'contradiction', 'orphan', 'gap', 'stale'
    details TEXT NOT NULL,      -- JSON
    resolved INTEGER DEFAULT 0
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_facts_topic ON facts(topic);
CREATE INDEX IF NOT EXISTS idx_facts_valid ON facts(valid_until);
CREATE INDEX IF NOT EXISTS idx_links_fact_a ON links(fact_id_a);
CREATE INDEX IF NOT EXISTS idx_links_fact_b ON links(fact_id_b);
"""


class LatticeMemory(MemorySystem):
    """Architecture E: Lattice Memory — best-of-breed, zero vector DB.

    Retrieval tiers:
    - Tier 1 (Index scan): Read always-loaded index, fetch full pages
    - Tier 2 (BM25 search): FTS5 query over facts + pending mutations
    - Tier 3 (LLM deep search): BM25 candidates + link traversal + LLM re-rank
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.consolidation_trigger_count = config.get("consolidation_trigger_count", 10)
        self.consolidation_trigger_minutes = config.get("consolidation_trigger_minutes", 30)
        self.lint_every_n = config.get("lint_every_n_consolidations", 5)
        self.index_max_lines = config.get("index_max_lines", 200)
        self.bm25_top_k = config.get("bm25_top_k", 20)
        self.deep_search_candidates = config.get("deep_search_candidates", 50)
        self.dedup_bm25_threshold = config.get("dedup_bm25_threshold", 15.0)

        self.cost_tracker = CostTracker()
        self.latency_tracker = LatencyTracker()
        self._unprocessed_count = 0
        self._consolidation_count = 0
        self._last_consolidation: datetime | None = None
        self._cached_index: str = ""

        self._init_db()

    def _init_db(self) -> None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(DB_PATH))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ── Ingestion ──────────────────────────────────────────────

    async def ingest(self, memories: list[Memory]) -> IngestStats:
        start = time.perf_counter()
        llm_calls = 0
        now = datetime.now().isoformat()

        for mem in memories:
            # Insert into raw_sources
            self.conn.execute(
                """INSERT INTO raw_sources (session_id, content, source_type, t_ingested)
                   VALUES (?, ?, ?, ?)""",
                (mem.source_id, mem.content, "memory", now),
            )
            # Also queue as pending mutation for interim searchability
            payload = json.dumps({
                "content": mem.content,
                "entities": mem.entities,
                "topics": mem.topics,
                "timestamp": mem.timestamp.isoformat(),
                "importance": mem.importance,
            })
            self.conn.execute(
                """INSERT INTO pending_mutations (operation, payload, t_queued)
                   VALUES ('ADD', ?, ?)""",
                (payload, now),
            )
            self._unprocessed_count += 1

        self.conn.commit()

        # Check consolidation trigger
        should_consolidate = self._unprocessed_count >= self.consolidation_trigger_count
        if self._last_consolidation:
            elapsed = (datetime.now() - self._last_consolidation).total_seconds() / 60
            should_consolidate = should_consolidate or elapsed >= self.consolidation_trigger_minutes

        if should_consolidate:
            llm_calls += await self._consolidate()

        duration = time.perf_counter() - start
        self._total_ingestion_time += duration
        self._total_llm_calls += llm_calls

        return IngestStats(
            memories_ingested=len(memories),
            llm_calls=llm_calls,
            embedding_calls=0,
            duration_s=duration,
            cost_usd=0.0,
        )

    # ── Consolidation ──────────────────────────────────────────

    async def _consolidate(self) -> int:
        """Full consolidation cycle: extract, match, merge, compile, index."""
        llm_calls = 0

        # Step 1: Extract atomic facts from unprocessed raw sources
        rows = self.conn.execute(
            "SELECT id, content, t_ingested FROM raw_sources WHERE processed = 0"
        ).fetchall()

        if not rows:
            self._last_consolidation = datetime.now()
            return 0

        # Batch extraction: send all unprocessed sources to LLM
        sources_text = "\n---\n".join(
            f"[Source {r['id']}] {r['content']}" for r in rows
        )

        extraction = await chat_completion_json(
            messages=[
                {
                    "role": "user",
                    "content": f"""Extract atomic facts from these sources. For each fact, provide:
- topic: a short topic label (lowercase, underscored)
- content: the atomic fact statement
- t_event: when this fact was true (ISO 8601 if detectable, null otherwise)
- confidence: 0.0-1.0
- source_id: the [Source N] id it came from
- supersedes: if this fact updates/contradicts a previous fact, describe what it supersedes

Sources:
{sources_text}

Return JSON: {{"facts": [{{...}}]}}""",
                }
            ],
            purpose="extraction",
            cost_tracker=self.cost_tracker,
        )
        llm_calls += 1

        extracted_facts = extraction.get("facts", [])
        now = datetime.now().isoformat()
        source_id_map = {r["id"]: r for r in rows}

        for fact in extracted_facts:
            topic = fact.get("topic", "general")
            content = fact.get("content", "")
            t_event = fact.get("t_event")
            confidence = fact.get("confidence", 1.0)
            src_id = fact.get("source_id", 0)

            # Step 2: BM25 match against existing facts on same topic
            existing = self.conn.execute(
                """SELECT f.id, f.content, f.topic, f.valid_until
                   FROM facts_fts fts
                   JOIN facts f ON fts.rowid = f.id
                   WHERE facts_fts MATCH ? AND f.topic = ? AND f.valid_until IS NULL
                   ORDER BY rank
                   LIMIT 5""",
                (content, topic),
            ).fetchall()

            if existing:
                # Ask LLM to decide: Keep, Update, or Delete
                existing_text = "\n".join(
                    f"[Fact {e['id']}] {e['content']}" for e in existing
                )
                decision = await chat_completion_json(
                    messages=[
                        {
                            "role": "user",
                            "content": f"""Compare this new fact against existing facts on the same topic.

New fact: {content}

Existing facts:
{existing_text}

Decide for each existing fact:
- KEEP: new fact is complementary, keep both
- UPDATE: new fact supersedes existing (set valid_until on old)
- DELETE: new fact is redundant (don't insert)

Also identify if the new fact CONTRADICTS any existing fact.

Return JSON: {{
  "action": "KEEP" | "UPDATE" | "DELETE",
  "superseded_fact_ids": [ids to set valid_until on],
  "contradicts_fact_ids": [ids that contradict],
  "insert_new": true | false
}}""",
                        }
                    ],
                    purpose="consolidation",
                    cost_tracker=self.cost_tracker,
                )
                llm_calls += 1

                # Apply decisions
                for sup_id in decision.get("superseded_fact_ids", []):
                    self.conn.execute(
                        "UPDATE facts SET valid_until = ? WHERE id = ?",
                        (now, sup_id),
                    )

                # Create contradiction links
                for contra_id in decision.get("contradicts_fact_ids", []):
                    if decision.get("insert_new", True):
                        # We'll create the link after inserting the new fact
                        pass

                if not decision.get("insert_new", True):
                    continue

            # Insert new fact
            cursor = self.conn.execute(
                """INSERT INTO facts (topic, content, t_event, t_ingested, confidence, source_ids)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (topic, content, t_event, now, confidence, json.dumps([src_id])),
            )
            new_fact_id = cursor.lastrowid

            # Create contradiction links if identified
            if existing:
                for contra_id in decision.get("contradicts_fact_ids", []):
                    self.conn.execute(
                        """INSERT INTO links (fact_id_a, fact_id_b, relation, t_created)
                           VALUES (?, ?, 'contradicts', ?)""",
                        (new_fact_id, contra_id, now),
                    )

                # Create temporal_successor links for superseded facts
                for sup_id in decision.get("superseded_fact_ids", []):
                    self.conn.execute(
                        """INSERT INTO links (fact_id_a, fact_id_b, relation, t_created)
                           VALUES (?, ?, 'temporal_successor', ?)""",
                        (new_fact_id, sup_id, now),
                    )

        # Mark raw sources as processed
        self.conn.execute("UPDATE raw_sources SET processed = 1 WHERE processed = 0")
        # Mark pending mutations as processed
        self.conn.execute("UPDATE pending_mutations SET processed = 1 WHERE processed = 0")

        # Step 3: Compile/update affected wiki pages
        affected_topics = set(f.get("topic", "general") for f in extracted_facts)
        for topic in affected_topics:
            llm_calls += await self._compile_page(topic)

        # Step 4: Regenerate index
        self._regenerate_index()

        self.conn.commit()
        self._unprocessed_count = 0
        self._consolidation_count += 1
        self._last_consolidation = datetime.now()

        # Step 5: Lint every N consolidations
        if self._consolidation_count % self.lint_every_n == 0:
            llm_calls += await self._lint()

        return llm_calls

    async def _compile_page(self, topic: str) -> int:
        """Compile a wiki page from all valid facts on a topic."""
        facts = self.conn.execute(
            """SELECT id, content, t_event, confidence FROM facts
               WHERE topic = ? AND valid_until IS NULL
               ORDER BY t_event, t_ingested""",
            (topic,),
        ).fetchall()

        if not facts:
            return 0

        facts_text = "\n".join(
            f"- [{f['id']}] {f['content']} (confidence: {f['confidence']})"
            for f in facts
        )

        result = await chat_completion_json(
            messages=[
                {
                    "role": "user",
                    "content": f"""Compile these facts about "{topic}" into a wiki page.

Facts:
{facts_text}

Return JSON: {{
  "summary": "one-line summary (max 80 chars)",
  "body": "full markdown page body with key facts, timeline if applicable",
  "related_topics": ["list", "of", "related", "topics"]
}}""",
                }
            ],
            purpose="consolidation",
            cost_tracker=self.cost_tracker,
        )

        existing = self.conn.execute(
            "SELECT id FROM pages WHERE topic = ?", (topic,)
        ).fetchone()

        now = datetime.now().isoformat()
        if existing:
            self.conn.execute(
                """UPDATE pages SET summary = ?, body = ?, last_modified = ?,
                   related_topics = ? WHERE topic = ?""",
                (
                    result.get("summary", topic),
                    result.get("body", ""),
                    now,
                    json.dumps(result.get("related_topics", [])),
                    topic,
                ),
            )
        else:
            self.conn.execute(
                """INSERT INTO pages (topic, summary, body, last_modified, related_topics)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    topic,
                    result.get("summary", topic),
                    result.get("body", ""),
                    now,
                    json.dumps(result.get("related_topics", [])),
                ),
            )

        return 1

    def _regenerate_index(self) -> None:
        """Regenerate the always-loaded index from pages table."""
        pages = self.conn.execute(
            """SELECT topic, summary, last_modified FROM pages
               ORDER BY last_modified DESC
               LIMIT ?""",
            (self.index_max_lines,),
        ).fetchall()

        count = len(pages)
        now = datetime.now().isoformat()
        lines = [f"# Memory Index ({count} topics, updated {now})", ""]
        lines.append("| Topic | Summary | Modified |")
        lines.append("|-------|---------|----------|")

        for p in pages:
            lines.append(f"| {p['topic']} | {p['summary']} | {p['last_modified'][:10]} |")

        self._cached_index = "\n".join(lines)

    # ── Linting ────────────────────────────────────────────────

    async def _lint(self) -> int:
        """Run lint checks: contradictions, orphans, gaps, stale pages."""
        llm_calls = 0
        now = datetime.now().isoformat()

        # 1. Contradiction scan: facts with 'contradicts' links where both are valid
        contradictions = self.conn.execute(
            """SELECT l.fact_id_a, l.fact_id_b, fa.content as content_a, fb.content as content_b
               FROM links l
               JOIN facts fa ON l.fact_id_a = fa.id
               JOIN facts fb ON l.fact_id_b = fb.id
               WHERE l.relation = 'contradicts'
               AND fa.valid_until IS NULL AND fb.valid_until IS NULL"""
        ).fetchall()

        for c in contradictions:
            self.conn.execute(
                """INSERT INTO lint_reports (t_run, report_type, details)
                   VALUES (?, 'contradiction', ?)""",
                (now, json.dumps({
                    "fact_a": {"id": c["fact_id_a"], "content": c["content_a"]},
                    "fact_b": {"id": c["fact_id_b"], "content": c["content_b"]},
                })),
            )

        # 2. Orphan scan: facts not associated with any page topic
        page_topics = set(
            r["topic"] for r in self.conn.execute("SELECT topic FROM pages").fetchall()
        )
        orphan_topics = self.conn.execute(
            "SELECT DISTINCT topic FROM facts WHERE valid_until IS NULL"
        ).fetchall()

        for ot in orphan_topics:
            if ot["topic"] not in page_topics:
                self.conn.execute(
                    """INSERT INTO lint_reports (t_run, report_type, details)
                       VALUES (?, 'orphan', ?)""",
                    (now, json.dumps({"topic": ot["topic"]})),
                )

        # 3. Gap scan: related_topics referenced but no page exists
        for page in self.conn.execute("SELECT related_topics FROM pages").fetchall():
            related = json.loads(page["related_topics"])
            for rt in related:
                if rt not in page_topics:
                    self.conn.execute(
                        """INSERT INTO lint_reports (t_run, report_type, details)
                           VALUES (?, 'gap', ?)""",
                        (now, json.dumps({"missing_topic": rt})),
                    )

        self.conn.commit()
        return llm_calls

    # ── Retrieval (3-tier) ─────────────────────────────────────

    async def query(
        self, question: str, timestamp: datetime | None = None
    ) -> QueryResult:
        start = time.perf_counter()
        tier_used = "index"

        # ── Tier 1: Index scan (free — index is already in memory)
        with self.latency_tracker.measure("retrieval"):
            retrieved, context = await self._tier1_index_scan(question)

            # If index didn't return enough, escalate to Tier 2
            if len(retrieved) < 2:
                tier_used = "bm25"
                t2_retrieved, t2_context = self._tier2_bm25_search(question, timestamp)
                retrieved.extend(t2_retrieved)
                context = context + "\n" + t2_context if context else t2_context

                # If still insufficient, escalate to Tier 3
                if len(retrieved) < 2:
                    tier_used = "deep"
                    t3_retrieved, t3_context = await self._tier3_deep_search(
                        question, timestamp
                    )
                    retrieved.extend(t3_retrieved)
                    context = context + "\n" + t3_context if context else t3_context

        # Deduplicate and cap at 5
        seen = set()
        unique_retrieved = []
        for item in retrieved:
            key = item.content[:100]
            if key not in seen:
                seen.add(key)
                unique_retrieved.append(item)
        retrieved = unique_retrieved[:5]
        context_parts = [r.content for r in retrieved]
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
            retrieved_context=retrieved,
            retrieval_tier=tier_used,
            latency_ms=total_ms,
            input_tokens=0,
            output_tokens=0,
            embedding_tokens=0,
            cost_usd=0.0,
        )

    async def _tier1_index_scan(
        self, question: str
    ) -> tuple[list[RetrievedItem], str]:
        """Tier 1: Read the always-loaded index, fetch relevant pages."""
        if not self._cached_index:
            self._regenerate_index()

        if not self._cached_index:
            return [], ""

        # Use LLM to identify relevant topics from the index
        result = await chat_completion_json(
            messages=[
                {
                    "role": "user",
                    "content": f"""Given this memory index and a question, identify 0-3 relevant topics.

{self._cached_index}

Question: {question}

Return JSON: {{"topics": ["topic1", "topic2"]}}
If no topics are relevant, return {{"topics": []}}""",
                }
            ],
            purpose="retrieval",
            cost_tracker=self.cost_tracker,
        )
        self._total_llm_calls += 1

        topics = result.get("topics", [])
        retrieved = []
        context_parts = []

        for topic in topics[:3]:
            page = self.conn.execute(
                "SELECT body, last_modified FROM pages WHERE topic = ?", (topic,)
            ).fetchone()
            if page:
                item = RetrievedItem(
                    content=page["body"],
                    source="pages",
                    score=1.0,
                    timestamp=datetime.fromisoformat(page["last_modified"]),
                )
                retrieved.append(item)
                context_parts.append(page["body"])

        return retrieved, "\n".join(context_parts)

    def _tier2_bm25_search(
        self, question: str, timestamp: datetime | None = None
    ) -> tuple[list[RetrievedItem], str]:
        """Tier 2: BM25 full-text search over facts + pending mutations."""
        # Search facts
        query_sql = """
            SELECT f.id, f.content, f.topic, f.t_event, f.confidence, rank
            FROM facts_fts fts
            JOIN facts f ON fts.rowid = f.id
            WHERE facts_fts MATCH ?
            AND f.valid_until IS NULL
        """
        params: list = [question]

        if timestamp:
            query_sql += " AND (f.t_event IS NULL OR f.t_event <= ?)"
            params.append(timestamp.isoformat())

        query_sql += " ORDER BY rank LIMIT ?"
        params.append(self.bm25_top_k)

        results = self.conn.execute(query_sql, params).fetchall()

        # Also search pending mutations
        pending = self.conn.execute(
            """SELECT payload FROM pending_fts
               WHERE pending_fts MATCH ?
               LIMIT 5""",
            (question,),
        ).fetchall()

        retrieved = []
        context_parts = []

        for row in results:
            item = RetrievedItem(
                content=row["content"],
                source="facts_fts",
                score=abs(row["rank"]),
                timestamp=datetime.fromisoformat(row["t_event"]) if row["t_event"] else None,
            )
            retrieved.append(item)
            context_parts.append(row["content"])

        for row in pending:
            payload = json.loads(row["payload"])
            item = RetrievedItem(
                content=payload.get("content", ""),
                source="pending",
                score=0.5,
            )
            retrieved.append(item)
            context_parts.append(payload.get("content", ""))

        return retrieved, "\n".join(context_parts[:10])

    async def _tier3_deep_search(
        self, question: str, timestamp: datetime | None = None
    ) -> tuple[list[RetrievedItem], str]:
        """Tier 3: BM25 candidates + link traversal + LLM re-rank."""
        # Get broader candidate set
        candidates = self.conn.execute(
            """SELECT f.id, f.content, f.topic FROM facts_fts fts
               JOIN facts f ON fts.rowid = f.id
               WHERE facts_fts MATCH ? AND f.valid_until IS NULL
               ORDER BY rank LIMIT ?""",
            (question, self.deep_search_candidates),
        ).fetchall()

        # Traverse links from matched facts (1-hop)
        candidate_ids = [c["id"] for c in candidates]
        linked_facts = []
        if candidate_ids:
            placeholders = ",".join("?" * len(candidate_ids))
            linked = self.conn.execute(
                f"""SELECT DISTINCT f.id, f.content, f.topic, l.relation
                    FROM links l
                    JOIN facts f ON (f.id = l.fact_id_a OR f.id = l.fact_id_b)
                    WHERE (l.fact_id_a IN ({placeholders}) OR l.fact_id_b IN ({placeholders}))
                    AND f.valid_until IS NULL
                    AND f.id NOT IN ({placeholders})""",
                candidate_ids * 3,
            ).fetchall()
            linked_facts = list(linked)

        # Combine candidates and linked facts
        all_candidates = list(candidates) + linked_facts
        if not all_candidates:
            return [], ""

        candidates_text = "\n".join(
            f"[{i}] {c['content']}" for i, c in enumerate(all_candidates[:30])
        )

        # LLM re-rank
        rerank = await chat_completion_json(
            messages=[
                {
                    "role": "user",
                    "content": f"""Given this question, select the most relevant facts.

Question: {question}

Candidate facts:
{candidates_text}

Return JSON: {{"selected_indices": [most relevant first], "reasoning": "brief explanation"}}""",
                }
            ],
            purpose="retrieval",
            cost_tracker=self.cost_tracker,
        )
        self._total_llm_calls += 1

        selected = rerank.get("selected_indices", [])
        retrieved = []
        context_parts = []

        for idx in selected[:5]:
            if idx < len(all_candidates):
                c = all_candidates[idx]
                item = RetrievedItem(
                    content=c["content"],
                    source="deep_search",
                    score=1.0 - (selected.index(idx) * 0.1),
                )
                retrieved.append(item)
                context_parts.append(c["content"])

        return retrieved, "\n".join(context_parts)

    # ── Agent mutation tools ───────────────────────────────────

    async def memory_add(self, content: str, topic: str, t_event: str | None = None) -> int:
        """Agent-native ADD: queue a new fact for next consolidation."""
        now = datetime.now().isoformat()
        payload = json.dumps({"content": content, "topic": topic, "t_event": t_event})
        cursor = self.conn.execute(
            "INSERT INTO pending_mutations (operation, payload, t_queued) VALUES ('ADD', ?, ?)",
            (payload, now),
        )
        self.conn.commit()
        return cursor.lastrowid

    async def memory_update(self, fact_id: int, new_content: str) -> None:
        """Agent-native UPDATE: queue a fact update."""
        now = datetime.now().isoformat()
        payload = json.dumps({"fact_id": fact_id, "new_content": new_content})
        self.conn.execute(
            "INSERT INTO pending_mutations (operation, payload, t_queued) VALUES ('UPDATE', ?, ?)",
            (payload, now),
        )
        self.conn.commit()

    async def memory_link(self, fact_id_a: int, fact_id_b: int, relation: str) -> None:
        """Agent-native LINK: create a typed relationship between facts."""
        now = datetime.now().isoformat()
        self.conn.execute(
            "INSERT INTO links (fact_id_a, fact_id_b, relation, t_created) VALUES (?, ?, ?, ?)",
            (fact_id_a, fact_id_b, relation, now),
        )
        self.conn.commit()

    # ── System interface ───────────────────────────────────────

    async def get_stats(self) -> SystemStats:
        fact_count = self.conn.execute(
            "SELECT COUNT(*) FROM facts WHERE valid_until IS NULL"
        ).fetchone()[0]
        page_count = self.conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
        db_size = DB_PATH.stat().st_size if DB_PATH.exists() else 0

        return SystemStats(
            memory_count=fact_count,
            storage_bytes=db_size,
            index_entries=page_count,
            total_ingestion_time_s=self._total_ingestion_time,
            total_llm_calls=self._total_llm_calls,
            total_embedding_calls=0,
        )

    async def reset(self) -> None:
        self.conn.executescript("""
            DELETE FROM raw_sources;
            DELETE FROM facts;
            DELETE FROM facts_fts;
            DELETE FROM pages;
            DELETE FROM links;
            DELETE FROM pending_mutations;
            DELETE FROM pending_fts;
            DELETE FROM lint_reports;
        """)
        self.conn.commit()
        self._unprocessed_count = 0
        self._consolidation_count = 0
        self._last_consolidation = None
        self._cached_index = ""
        self._total_ingestion_time = 0.0
        self._total_llm_calls = 0
        self._total_embedding_calls = 0
        self.cost_tracker.reset()
        self.latency_tracker.reset()

    async def close(self) -> None:
        self.conn.close()
