"""Zep / Graphiti (graphiti-core) through the shim: temporal knowledge-graph memory.

Each memory is an EPISODE; Graphiti extracts entities and bi-temporal relationship edges (facts
with valid_at / invalid_at), deduplicates against the graph and invalidates superseded edges. The
graph lives in an embedded Kuzu database (no Neo4j in this container). Search returns edge facts.
LLM: Graphiti's own OpenAI-compatible client pointed at the shim; embedder: shim; reranker: BGE
cross-encoder in-process (Graphiti's default reranker needs logprobs the shim cannot provide).
"""
from __future__ import annotations

import asyncio
import shutil
import tempfile
import time
from datetime import timezone

from src.systems.base_adapter import BaseAdapter
from src.systems.common import SHIM_BASE, fmt_mem


class GraphitiAdapter(BaseAdapter):
    name, family, year, kind = "zep_graphiti", "temporal knowledge graph", 2024, "modern"
    top_k = 5

    async def _build(self):
        from graphiti_core import Graphiti
        from graphiti_core.driver.kuzu_driver import KuzuDriver
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        from graphiti_core.cross_encoder.bge_reranker_client import BGERerankerClient
        self._dir = tempfile.mkdtemp(prefix="graphiti-")
        llm = OpenAIGenericClient(config=LLMConfig(api_key="shim", base_url=SHIM_BASE, model="gpt-4o-mini", small_model="gpt-4o-mini"))
        emb = OpenAIEmbedder(config=OpenAIEmbedderConfig(api_key="shim", base_url=SHIM_BASE, embedding_model="bge-small", embedding_dim=384))
        self._g = Graphiti(graph_driver=KuzuDriver(db=f"{self._dir}/kuzu"), llm_client=llm, embedder=emb, cross_encoder=BGERerankerClient())
        await self._g.build_indices_and_constraints()
        # graphiti-core's (deprecated) Kuzu driver creates the schema but NOT the full-text indexes its
        # own search queries call by name; create them here so hybrid search works.
        for table, index, cols in (("Episodic", "episode_content", "['content']"), ("Entity", "node_name_and_summary", "['name', 'summary']"),
                                   ("Community", "community_name", "['name']"), ("RelatesToNode_", "edge_name_and_fact", "['name', 'fact']")):
            try:
                await self._g.driver.execute_query(f"CALL CREATE_FTS_INDEX('{table}', '{index}', {cols})")
            except Exception as ex:  # noqa: BLE001 — index may already exist
                if "already" not in str(ex).lower():
                    raise
        self.episodes = 0

    def _on_reset(self):
        if getattr(self, "_g", None) is not None:
            try:
                asyncio.get_event_loop().run_until_complete(self._g.close())
            except Exception:  # noqa: BLE001
                pass
            shutil.rmtree(getattr(self, "_dir", ""), ignore_errors=True)
        self._g = None
        self._episodes_by_source: dict[str, list[str]] = {}

    async def ingest(self, memories):
        from graphiti_core.nodes import EpisodeType
        if getattr(self, "_g", None) is None:
            await self._build()
        t0, c0 = time.time(), self.llm.calls
        for m in memories:
            ts = m.timestamp if m.timestamp.tzinfo else m.timestamp.replace(tzinfo=timezone.utc)
            res = await self._g.add_episode(name=m.source_id, episode_body=m.content, source=EpisodeType.text,
                                            source_description="memory note", reference_time=ts)
            ep = getattr(res, "episode", None)
            if ep is not None and getattr(ep, "uuid", None):
                self._episodes_by_source.setdefault(m.source_id, []).append(ep.uuid)
            self.episodes += 1
        self.memories.extend(memories)
        return self._ingest_stats(len(memories), t0, c0)

    async def delete(self, source_ids):
        """Native deletion: remove the episodes ingested from these notes (Graphiti drops the edges and the entity
        nodes that only those episodes supported). The base-class rebuild path is not usable here — it re-ingests
        through _on_ingest, which this adapter does not implement (the round-2 v1 records show every post-delete
        query erroring with a closed driver), and a rebuild would also cost ~4 LLM calls per memory per case."""
        ids = set(source_ids)
        n = 0
        for sid in ids:
            for uuid in self._episodes_by_source.pop(sid, []):
                try:
                    await self._g.remove_episode(uuid)
                    n += 1
                except Exception:  # noqa: BLE001
                    pass
        self.memories = [m for m in self.memories if m.source_id not in ids]
        return {"deleted": n, "method": "graphiti.remove_episode"}

    async def query(self, question, timestamp=None):
        t0 = time.time()
        edges = await self._g.search(question, num_results=self.top_k)
        items = []
        for e in edges or []:
            fact = getattr(e, "fact", None) or str(e)
            va = getattr(e, "valid_at", None)
            inv = getattr(e, "invalid_at", None)
            tag = f" (valid {va.date() if va else '?'}{' → invalid ' + str(inv.date()) if inv else ''})"
            items.append(self._item(fact + tag, getattr(e, "uuid", "edge"), 1.0, va))
        ans, it, ot = self._answer("\n".join(x.content for x in items), question)
        return self._result(ans, items, "graph.search", t0, it, ot)

    async def close(self):
        if getattr(self, "_g", None) is not None:
            try:
                await self._g.close()
            except Exception:  # noqa: BLE001
                pass
        shutil.rmtree(getattr(self, "_dir", ""), ignore_errors=True)
