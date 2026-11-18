"""MinMem — the reference design round 4 pre-registers (experiments/prereg_round4.md, WP2).

Rounds 2-3 found that systems fail on updates when their context can MISS the newest note about the subject, and fail on
deletion when they keep text DERIVED from deleted notes. MinMem is the least machinery that addresses both, with no LLM
call anywhere on the write path:

  write    index the raw notes three ways: BM25, dense vectors (bge-small), and an inverted index from capitalised
           spans found in the note TEXT (a regex; the generator's gold `entities` / `topics` metadata is never read).
  read     candidates = BM25 top-20 ∪ dense top-20 ∪ every note that shares a capitalised span with the question
           (entity-scoped expansion, capped at 200 by BM25 score) → cross-encoder rerank → keep the top 20 → present
           them OLDEST FIRST, followed by one trailer line saying later notes supersede earlier ones.
  answer   the shared answer prompt, like every other system. MinMem only shapes the context.
  delete   drop the notes and rebuild the three indexes. Nothing is derived, so nothing can outlive a delete.

Decomposition (each arm adds one thing; `hybrid_k20` lives in baselines.py):
  hybrid_k20     BM25 ∪ dense candidates, rerank, top 20, reranker order
  minmem_nohdr   + entity expansion
  minmem_order   + oldest-first order
  minmem         + the trailer line
  minmem_noexp   minmem without the expansion (order and trailer kept)
"""
from __future__ import annotations

import re
import threading
import time

import numpy as np

from src.systems.base_adapter import BaseAdapter
from src.systems.baselines import BM25Baseline, HybridRerank, VectorRAG
from src.systems.common import fmt_mem

_SPAN = re.compile(r"[A-Z][\w'’]*\.?(?:\s+[A-Z][\w'’]*\.?)*")
_CE_LOCK = threading.Lock()
# Placed AFTER the notes (a trailer), so that full context + trailer shares its cached prefix with plain full context.
TRAILER = "(The notes above are in date order, oldest first. When two notes disagree, the later note is the current one.)"


def spans(text: str) -> set[str]:
    """Capitalised spans, lower-cased, without a trailing possessive ("Liam O'Brien's" -> "liam o'brien")."""
    return {re.sub(r"['’]s$", "", m.group(0).strip(" .")).lower() for m in _SPAN.finditer(text)}


class MinMem(BaseAdapter):
    name, family, year, kind = "minmem", "hybrid retrieval + entity expansion + latest-wins ordering", 2026, "modern"
    cand_k, exp_cap, top_m = 20, 200, 20
    expand, chrono, trailer = True, True, True

    def _on_reset(self):
        self._bm, self._vr = BM25Baseline(), VectorRAG()
        self._ent: dict[str, list[int]] = {}
        self._ctx: dict[str, tuple] = {}

    def _on_ingest(self, memories):
        self._bm.memories = self._vr.memories = self.memories
        self._bm._on_ingest(memories)
        self._vr._on_ingest(memories)
        self._ent, self._ctx = {}, {}
        for i, m in enumerate(self.memories):
            for s in spans(m.content):
                self._ent.setdefault(s, []).append(i)

    def _retrieve(self, question: str) -> tuple[list, int]:
        """Reader-independent, so cached per question (the runner asks each question once per reader)."""
        if question in self._ctx:
            return self._ctx[question]
        cands = {i for i, _ in self._bm._rank(question, self.cand_k)} | {i for i, _ in self._vr._rank(question, self.cand_k)}
        n_exp = 0
        if self.expand:
            exp = set()
            for s in spans(question):
                exp.update(self._ent.get(s, ()))
            if len(exp) > self.exp_cap:   # a very common span: keep the notes BM25 thinks are closest to the question
                sc = self._bm._bm25.get_scores(re.findall(r"[a-z0-9]+", question.lower()))
                exp = set(sorted(exp, key=lambda i: -sc[i])[: self.exp_cap])
            n_exp = len(exp - cands)
            cands |= exp
        cands = sorted(cands)
        hits = []
        if cands:
            with _CE_LOCK:
                scores = HybridRerank._cross_encoder(HybridRerank).predict([(question, self.memories[i].content) for i in cands], batch_size=64)
            order = np.argsort(-np.asarray(scores))[: self.top_m]
            hits = [(cands[j], float(scores[j])) for j in order]
            if self.chrono:   # oldest first (ties by id)
                hits.sort(key=lambda h: (self.memories[h[0]].timestamp, self.memories[h[0]].source_id))
        self._ctx[question] = (hits, n_exp)
        return self._ctx[question]

    async def query(self, question, timestamp=None):
        t0 = time.time()
        hits, _ = self._retrieve(question)
        items = [self._item(fmt_mem(self.memories[i]), self.memories[i].source_id, s, self.memories[i].timestamp) for i, s in hits]
        body = "\n".join(x.content for x in items)
        ans, it, ot = self._answer((body + "\n" + TRAILER) if (self.trailer and items) else body, question)
        return self._result(ans, items, "minmem", t0, it, ot)


class MinMemNoExpansion(MinMem):
    name = "minmem_noexp"
    expand = False


class MinMemNoHeader(MinMem):
    name = "minmem_nohdr"
    chrono, trailer = False, False


class MinMemOrderOnly(MinMem):
    name = "minmem_order"
    trailer = False
