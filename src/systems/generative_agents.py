"""Generative Agents memory stream (Park et al., UIST 2023) — faithful re-implementation.

Three mechanisms from the paper, nothing else:
  * every memory gets an LLM-rated IMPORTANCE (1–10) at write time;
  * retrieval score = recency (exponential decay) + importance + relevance (embedding cosine),
    each min–max normalised to [0, 1], equal weights, top-k into the prompt;
  * REFLECTION: when accumulated importance since the last reflection exceeds a threshold, the
    agent generates high-level insights from recent memories and stores them as new memories.
Importance rating is batched 10 memories per call (a cost optimisation; the paper rates one
memory per call). Recency: the paper decays by 0.995 per *sandbox hour since last access*; our
memories span months of calendar time, so one sandbox hour is mapped to one calendar day (with
per-hour decay, everything older than a day scores ~0 and recency swamps relevance — observed in
the feasibility round, accuracy 0.38). "Last access" is updated on retrieval, as in the paper.
"""
from __future__ import annotations

import time
from datetime import datetime

import numpy as np

from src.architectures.base import Memory
from src.systems.base_adapter import BaseAdapter
from src.systems.common import Embedder, fmt_mem

_IMPORTANCE_SYS = "You rate the poignancy of memories for an agent, as in Generative Agents (Park et al. 2023)."
_IMPORTANCE = ("On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth) and 10 is extremely "
               "poignant (e.g., a break up, a major decision), rate the likely poignancy of each memory.\n"
               "Return JSON: {{\"ratings\": [int, ...]}} with one integer per memory, in order.\n\nMemories:\n{mems}")
_REFLECT_SYS = "You write reflections for an agent's memory stream, as in Generative Agents (Park et al. 2023)."
_REFLECT = ("Given only the recent memories below, what are 3 high-level insights you can infer? "
            "Each insight must be a single factual sentence grounded in the memories (name people, dates, decisions). "
            "Return JSON: {{\"insights\": [\"...\", \"...\", \"...\"]}}\n\nRecent memories:\n{mems}")


class GenerativeAgentsMemory(BaseAdapter):
    name, family, year, kind = "generative_agents", "scored memory stream + reflection", 2023, "classic"
    top_k = 5
    decay = 0.995           # per sandbox hour (paper); one sandbox hour == one calendar day here
    reflect_threshold = 150  # sum of importance since last reflection (paper value)

    def _on_reset(self):
        self._imp: list[float] = []
        self._vecs = None
        self._since_reflection = 0.0
        self._reflections = 0
        self._last_access: dict[str, datetime] = {}

    def _rate(self, memories: list[Memory]) -> list[float]:
        out = []
        for i in range(0, len(memories), 10):
            batch = memories[i:i + 10]
            r = self.llm.json(_IMPORTANCE_SYS, _IMPORTANCE.format(mems="\n".join(f"{j+1}. {m.content}" for j, m in enumerate(batch))))
            ratings = r.get("ratings") if isinstance(r, dict) else None
            if not isinstance(ratings, list) or len(ratings) != len(batch):
                ratings = [5] * len(batch)
            out += [min(10.0, max(1.0, float(x))) for x in ratings]
        return out

    def _on_ingest(self, memories):
        new = memories if len(self._imp) < len(self.memories) else []
        if new:
            self._imp += self._rate(new)
            self._since_reflection += sum(self._imp[-len(new):])
        self._vecs = Embedder.get().encode([m.content for m in self.memories]) if self.memories else None
        if self._since_reflection >= self.reflect_threshold and self.memories:
            self._reflect()

    def _reflect(self):
        recent = sorted(self.memories, key=lambda m: m.timestamp)[-30:]
        r = self.llm.json(_REFLECT_SYS, _REFLECT.format(mems="\n".join(fmt_mem(m) for m in recent)))
        insights = [s for s in (r.get("insights") or []) if isinstance(s, str) and s.strip()][:3] if isinstance(r, dict) else []
        ts = max(m.timestamp for m in recent)
        for k, s in enumerate(insights):
            self.memories.append(Memory(content=s, entities=[], topics=["reflection"], timestamp=ts, importance=8.0,
                                        source_id=f"reflection_{self._reflections}_{k}"))
            self._imp.append(8.0)
        self._reflections += 1
        self._since_reflection = 0.0
        self._vecs = Embedder.get().encode([m.content for m in self.memories])

    def _scores(self, question: str, now: datetime) -> np.ndarray:
        rel = self._vecs @ Embedder.get().encode([question])[0]
        days = np.array([(now - self._last_access.get(m.source_id, m.timestamp)).total_seconds() / 86400.0 for m in self.memories])
        rec = np.power(self.decay, np.clip(days, 0, None))
        imp = np.array(self._imp[: len(self.memories)]) / 10.0

        def norm(x):
            return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.zeros_like(x)
        return norm(rec) + norm(imp) + norm(rel)

    async def query(self, question, timestamp=None):
        t0 = time.time()
        if not self.memories:
            ans, it, ot = self._answer("", question)
            return self._result(ans, [], "stream", t0, it, ot)
        now = timestamp or max(m.timestamp for m in self.memories)
        s = self._scores(question, now)
        idx = np.argsort(-s)[: self.top_k]
        for i in idx:
            self._last_access[self.memories[i].source_id] = now   # retrieval refreshes recency (paper semantics)
        items = [self._item(fmt_mem(self.memories[i]), self.memories[i].source_id, float(s[i]), self.memories[i].timestamp) for i in idx]
        ans, it, ot = self._answer("\n".join(x.content for x in items), question)
        return self._result(ans, items, "recency+importance+relevance", t0, it, ot)
