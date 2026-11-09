"""Baselines + the two RAG family members (vanilla vector RAG, hybrid + reranker).

  bm25           — keyword retrieval, top-5. The 2023 "sparse" default and this repo's old A.
  full_context   — every memory in the prompt, no retrieval. The thing every memory system must beat.
  vector_rag     — dense top-5 (bge-small), the 2022–23 default RAG.
  hybrid_rerank  — BM25 ∪ dense candidates, cross-encoder rerank (bge-reranker-base), top-5:
                   the pattern most production teams actually run in 2026.
"""
from __future__ import annotations

import re
import time
from datetime import datetime

import numpy as np

from src.architectures.base import Memory
from src.systems.base_adapter import BaseAdapter
from src.systems.common import Embedder, fmt_mem

_TOK = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> list[str]:
    return _TOK.findall(s.lower())


class BM25Baseline(BaseAdapter):
    name, family, year, kind = "bm25", "sparse retrieval", 2023, "baseline"
    top_k = 5

    def _on_reset(self):
        self._bm25 = None

    def _on_ingest(self, memories):
        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi([_tokens(m.content) for m in self.memories]) if self.memories else None

    def _rank(self, question: str, k: int) -> list[tuple[int, float]]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tokens(question))
        idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]

    async def query(self, question, timestamp=None):
        t0 = time.time()
        hits = self._rank(question, self.top_k)
        items = [self._item(fmt_mem(self.memories[i]), self.memories[i].source_id, s, self.memories[i].timestamp) for i, s in hits]
        ans, it, ot = self._answer("\n".join(x.content for x in items), question)
        return self._result(ans, items, "bm25", t0, it, ot)


class FullContextBaseline(BaseAdapter):
    name, family, year, kind = "full_context", "no retrieval", 2023, "baseline"
    retrieves = False

    async def query(self, question, timestamp=None):
        t0 = time.time()
        mems = sorted(self.memories, key=lambda m: m.timestamp)
        ctx = "\n".join(fmt_mem(m) for m in mems)
        ans, it, ot = self._answer(ctx, question)
        items = [self._item(f"(all {len(mems)} memories in context)", "all", 1.0)]
        return self._result(ans, items, "full", t0, it, ot)


class VectorRAG(BaseAdapter):
    name, family, year, kind = "vector_rag", "dense retrieval", 2023, "classic"
    top_k = 5

    def _on_reset(self):
        self._vecs = None

    def _on_ingest(self, memories):
        self._vecs = Embedder.get().encode([m.content for m in self.memories]) if self.memories else None

    def _rank(self, question: str, k: int) -> list[tuple[int, float]]:
        if self._vecs is None or not len(self._vecs):
            return []
        q = Embedder.get().encode([question])[0]
        sims = self._vecs @ q
        idx = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in idx]

    async def query(self, question, timestamp=None):
        t0 = time.time()
        hits = self._rank(question, self.top_k)
        items = [self._item(fmt_mem(self.memories[i]), self.memories[i].source_id, s, self.memories[i].timestamp) for i, s in hits]
        ans, it, ot = self._answer("\n".join(x.content for x in items), question)
        return self._result(ans, items, "dense", t0, it, ot)


class HybridRerank(BaseAdapter):
    name, family, year, kind = "hybrid_rerank", "hybrid retrieval + reranker", 2024, "modern"
    top_k, cand_k = 5, 20
    reranker_model = __import__("os").environ.get("RERANKER_MODEL", "BAAI/bge-reranker-base")
    _ce = None

    def _on_reset(self):
        self._bm = BM25Baseline()
        self._vr = VectorRAG()

    def _on_ingest(self, memories):
        self._bm.memories = self.memories
        self._vr.memories = self.memories
        self._bm._on_ingest(memories)
        self._vr._on_ingest(memories)

    def _cross_encoder(self):
        if HybridRerank._ce is None:
            from sentence_transformers import CrossEncoder
            HybridRerank._ce = CrossEncoder(self.reranker_model)
        return HybridRerank._ce

    async def query(self, question, timestamp=None):
        t0 = time.time()
        cands = {i for i, _ in self._bm._rank(question, self.cand_k)} | {i for i, _ in self._vr._rank(question, self.cand_k)}
        cands = sorted(cands)
        if cands:
            scores = self._cross_encoder().predict([(question, self.memories[i].content) for i in cands])
            order = np.argsort(-np.asarray(scores))[: self.top_k]
            hits = [(cands[j], float(scores[j])) for j in order]
        else:
            hits = []
        items = [self._item(fmt_mem(self.memories[i]), self.memories[i].source_id, s, self.memories[i].timestamp) for i, s in hits]
        ans, it, ot = self._answer("\n".join(x.content for x in items), question)
        return self._result(ans, items, "hybrid+rerank", t0, it, ot)
