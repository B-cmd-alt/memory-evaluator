"""Base adapter: instrumentation + the QueryResult/IngestStats bookkeeping every system shares."""
from __future__ import annotations

import time
from datetime import datetime

from src.architectures.base import IngestStats, Memory, MemorySystem, QueryResult, RetrievedItem, SystemStats
from src.systems.common import LLM, answer_from_context, est_tokens


class BaseAdapter(MemorySystem):
    name = "base"
    family = "?"
    year = 0
    kind = "?"            # baseline | classic | modern
    retrieves = True      # False for full-context / compressed-log designs

    def __init__(self, config: dict | None = None):
        super().__init__(config or {})
        self.llm: LLM = self.config.get("llm") or LLM(caller=self.name)
        self.memories: list[Memory] = []
        self._ingest_time = 0.0
        self.workdir = self.config.get("workdir")

    # ── helpers ───────────────────────────────────────────────────────────────────────────────
    def _answer(self, context: str, question: str):
        return answer_from_context(self.llm, context, question)

    def _result(self, answer: str, items: list[RetrievedItem], tier: str, t0: float, in_tok: int, out_tok: int) -> QueryResult:
        return QueryResult(answer=answer, retrieved_context=items, retrieval_tier=tier,
                           latency_ms=(time.time() - t0) * 1000, input_tokens=in_tok, output_tokens=out_tok,
                           embedding_tokens=0, cost_usd=0.0)

    def _ingest_stats(self, n: int, t0: float, llm_before: int) -> IngestStats:
        dt = time.time() - t0
        self._ingest_time += dt
        return IngestStats(memories_ingested=n, llm_calls=self.llm.calls - llm_before, embedding_calls=0,
                           duration_s=dt, cost_usd=0.0)

    # ── contract ──────────────────────────────────────────────────────────────────────────────
    async def ingest(self, memories: list[Memory]) -> IngestStats:
        t0, c0 = time.time(), self.llm.calls
        self.memories.extend(memories)
        self._on_ingest(memories)
        return self._ingest_stats(len(memories), t0, c0)

    def _on_ingest(self, memories: list[Memory]) -> None:  # override
        pass

    async def query(self, question: str, timestamp: datetime | None = None) -> QueryResult:
        raise NotImplementedError

    async def get_stats(self) -> SystemStats:
        return SystemStats(memory_count=len(self.memories), storage_bytes=sum(len(m.content) for m in self.memories),
                           index_entries=len(self.memories), total_ingestion_time_s=self._ingest_time,
                           total_llm_calls=self.llm.calls, total_embedding_calls=0)

    async def reset(self) -> None:
        self.memories = []
        self._ingest_time = 0.0
        self._on_reset()

    def _on_reset(self) -> None:  # override
        pass

    async def close(self) -> None:
        pass

    # deletion contract (SixAxis A3) — default: drop by source_id and rebuild
    async def delete(self, source_ids: list[str]) -> dict:
        before = len(self.memories)
        self.memories = [m for m in self.memories if m.source_id not in set(source_ids)]
        self._on_reset()
        self._on_ingest(self.memories)
        return {"deleted": before - len(self.memories), "method": "rebuild"}

    @staticmethod
    def _item(content: str, source: str, score: float, ts=None) -> RetrievedItem:
        return RetrievedItem(content=content, source=source, score=float(score), timestamp=ts)

    @staticmethod
    def _tok(s: str) -> int:
        return est_tokens(s)
