"""A-MEM (Xu et al., NeurIPS 2025; agiresearch/A-mem) through the shim: agentic Zettelkasten memory.

Each note is analysed by an LLM (keywords, context, tags), linked to related notes, and may trigger
EVOLUTION of its neighbours (updating their context/tags). Retrieval is embedding search over the
note network (search_agentic). This is the authors' package; we only route its OpenAI client to
the shim (it honours OPENAI_BASE_URL) and use the same bge-small embedder.
"""
from __future__ import annotations

import os
import time

from src.systems.base_adapter import BaseAdapter
from src.systems.common import EMBED_MODEL, SHIM_BASE, fmt_mem


class AMemAdapter(BaseAdapter):
    name, family, year, kind = "a_mem", "agentic note network", 2025, "modern"
    top_k = 5

    def _build(self):
        os.environ["OPENAI_API_KEY"] = "shim"
        os.environ["OPENAI_BASE_URL"] = SHIM_BASE
        from agentic_memory.memory_system import AgenticMemorySystem
        self._s = AgenticMemorySystem(model_name=EMBED_MODEL, llm_backend="openai", llm_model="gpt-4o-mini", evo_threshold=100)
        self.note_ids = {}

    def _on_reset(self):
        self._s = None
        self._build()

    def _on_ingest(self, memories):
        if getattr(self, "_s", None) is None:
            self._build()
        for m in memories:
            nid = self._s.add_note(fmt_mem(m), time=m.timestamp.strftime("%Y%m%d%H%M"))
            self.note_ids[m.source_id] = nid

    async def query(self, question, timestamp=None):
        t0 = time.time()
        res = self._s.search_agentic(question, k=self.top_k)
        items = []
        for r in res or []:
            content = r.get("content") if isinstance(r, dict) else str(r)
            ctx = (r.get("context") if isinstance(r, dict) else "") or ""
            items.append(self._item(f"{content}" + (f"  [context: {ctx}]" if ctx else ""), str(r.get("id", "note")) if isinstance(r, dict) else "note", float(r.get("score", 0) or 0) if isinstance(r, dict) else 0.0))
        ans, it, ot = self._answer("\n".join(x.content for x in items), question)
        return self._result(ans, items, "search_agentic", t0, it, ot)

    async def delete(self, source_ids):
        n = 0
        for sid in source_ids:
            nid = self.note_ids.get(sid)
            if nid is not None:
                try:
                    self._s.delete(nid); n += 1
                except Exception:  # noqa: BLE001
                    pass
        self.memories = [m for m in self.memories if m.source_id not in set(source_ids)]
        return {"deleted": n, "method": "a_mem.delete"}
