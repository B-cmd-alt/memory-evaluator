"""Mem0 (open-source, mem0ai) through the shim: extraction-based memory layer.

Mem0 extracts facts from each message with an LLM, decides ADD / UPDATE / DELETE against similar
existing memories, and stores them in a vector store (Chroma here). All of that is Mem0's own code;
we only route its LLM and embedder to the shim so it uses the same model as every other system.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time

from src.systems.base_adapter import BaseAdapter
from src.systems.common import SHIM_BASE, fmt_mem


class Mem0Adapter(BaseAdapter):
    name, family, year, kind = "mem0", "extracted-facts memory layer", 2024, "modern"
    top_k = 5
    user_id = "u1"

    def _build(self):
        os.environ.setdefault("OPENAI_API_KEY", "shim")
        os.environ["OPENAI_BASE_URL"] = SHIM_BASE
        from mem0 import Memory as M0
        self._dir = tempfile.mkdtemp(prefix="mem0-")
        cfg = {
            "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini", "temperature": 0, "api_key": "shim", "openai_base_url": SHIM_BASE}},
            "embedder": {"provider": "openai", "config": {"model": "bge-small", "embedding_dims": 384, "api_key": "shim", "openai_base_url": SHIM_BASE}},
            "vector_store": {"provider": "chroma", "config": {"collection_name": "sixaxis", "path": self._dir}},
            "version": "v1.1",
        }
        self._m = M0.from_config(cfg)

    def _on_reset(self):
        if getattr(self, "_m", None) is not None:
            try:
                self._m.delete_all(user_id=self.user_id)
            except Exception:  # noqa: BLE001
                pass
            shutil.rmtree(getattr(self, "_dir", ""), ignore_errors=True)
        self._m = None
        self._build()
        self.add_results = []

    def _on_ingest(self, memories):
        if getattr(self, "_m", None) is None:
            self._build()
        for m in memories:
            r = self._m.add(fmt_mem(m), user_id=self.user_id, metadata={"source_id": m.source_id, "date": m.timestamp.isoformat()})
            self.add_results.append({"source_id": m.source_id, "ops": [(x.get("event"), x.get("memory")) for x in (r.get("results") or [])]})

    async def query(self, question, timestamp=None):
        t0 = time.time()
        res = self._m.search(question, filters={"user_id": self.user_id}, limit=self.top_k)
        hits = res.get("results") if isinstance(res, dict) else res
        items = [self._item(h.get("memory", ""), str((h.get("metadata") or {}).get("source_id", h.get("id"))), h.get("score") or 0.0) for h in (hits or [])]
        ans, it, ot = self._answer("\n".join(x.content for x in items), question)
        return self._result(ans, items, "mem0.search", t0, it, ot)

    async def delete(self, source_ids):
        # Mem0 stores EXTRACTED facts, not our source rows; delete every stored memory whose metadata points at a source id.
        n = 0
        allm = self._m.get_all(filters={"user_id": self.user_id})
        for h in (allm.get("results") if isinstance(allm, dict) else allm) or []:
            if (h.get("metadata") or {}).get("source_id") in set(source_ids):
                self._m.delete(h["id"]); n += 1
        self.memories = [m for m in self.memories if m.source_id not in set(source_ids)]
        return {"deleted": n, "method": "mem0.delete by metadata"}

    async def close(self):
        shutil.rmtree(getattr(self, "_dir", ""), ignore_errors=True)
