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

    # Round-3 fairness arms, evaluated on ONE ingest (the runner re-asks the T4 questions under each):
    #   r2       round-2 configuration: Graphiti.search (EDGE_HYBRID_SEARCH_RRF), 5 edges, invalidation shown as a text tag
    #   default  the library default: Graphiti.search with its default num_results (10)
    #   filter   fetch 30, drop edges Graphiti marked invalid/expired, keep 10 (what an application reading only current facts does)
    #   xenc     Graphiti.search_ with its EDGE_HYBRID_SEARCH_CROSS_ENCODER recipe (local BGE cross-encoder), 10 edges
    read_arm = __import__("os").environ.get("GRAPHITI_READ_ARM", "r2")   # round 4 pre-registers "xenc", round 3's best arm
    read_arms = ("default", "filter", "xenc")

    async def _edges(self, question):
        arm = self.read_arm
        if arm == "r2":
            return await self._g.search(question, num_results=self.top_k)
        if arm == "default":
            return await self._g.search(question)
        if arm == "filter":
            es = await self._g.search(question, num_results=30)
            return [e for e in es if getattr(e, "invalid_at", None) is None and getattr(e, "expired_at", None) is None][:10]
        if arm == "xenc":
            import copy
            from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_CROSS_ENCODER
            cfg = copy.deepcopy(EDGE_HYBRID_SEARCH_CROSS_ENCODER); cfg.limit = 10
            return (await self._g.search_(question, config=cfg)).edges
        raise ValueError(arm)

    async def invalidation_report(self, data: dict) -> dict:
        """Diagnostic (no LLM): for every T4 key, the edges whose fact states the SUPERSEDED value — what share did Graphiti
        mark invalid/expired? Low share = superseded facts stay current in the graph; high share = the failure is retrieval."""
        rows = []
        for q in data["questions"]:
            if q["task"] != "T4":
                continue
            es = await self._g.search(q["question"].replace(" (as of today)", ""), num_results=50)
            st = [e for e in es if q["stale_answer"].lower() in (getattr(e, "fact", "") or "").lower()]
            cur = [e for e in es if q["reference_answer"].lower() in (getattr(e, "fact", "") or "").lower()]
            rows.append({"id": q["id"], "stratum": q.get("stratum", ""), "n_stale_edges": len(st),
                         "n_stale_invalidated": sum(1 for e in st if getattr(e, "invalid_at", None) or getattr(e, "expired_at", None)),
                         "n_current_edges": len(cur),
                         "n_current_invalidated": sum(1 for e in cur if getattr(e, "invalid_at", None) or getattr(e, "expired_at", None)),
                         "stale_facts": [e.fact for e in st][:3], "current_facts": [e.fact for e in cur][:3]})
        have = [r for r in rows if r["n_stale_edges"]]
        return {"rows": rows, "keys_with_stale_edge": len(have), "keys": len(rows),
                "keys_all_stale_invalidated": sum(1 for r in have if r["n_stale_invalidated"] == r["n_stale_edges"]),
                "keys_with_current_edge": sum(1 for r in rows if r["n_current_edges"])}

    async def query(self, question, timestamp=None):
        t0 = time.time()
        edges = await self._edges(question)
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


class GraphitiNeo4jAdapter(GraphitiAdapter):
    """Round 5, Part B: Graphiti on the stack it targets — Neo4j, and an OpenAI model with JSON-schema output for the
    write path (model ids containing "/" pass through the shim to the provider natively). Embedder and reranker stay as
    in every other arm, so only the graph store and the write model differ from `zep_graphiti`.
    Neo4j community has one database: a cell clears it on build, so cells must run one at a time."""
    name = "zep_graphiti_neo4j"

    async def _build(self):
        import os
        from graphiti_core import Graphiti
        from graphiti_core.cross_encoder.bge_reranker_client import BGERerankerClient
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        from graphiti_core.utils.maintenance.graph_data_operations import clear_data
        self._dir = ""
        llm = OpenAIGenericClient(config=LLMConfig(api_key="shim", base_url=SHIM_BASE, model=os.environ.get("GRAPHITI_MODEL", "openai/gpt-6.1-terra"),
                                                   small_model=os.environ.get("GRAPHITI_SMALL_MODEL", "openai/gpt-6-luna")))
        emb = OpenAIEmbedder(config=OpenAIEmbedderConfig(api_key="shim", base_url=SHIM_BASE, embedding_model="bge-small", embedding_dim=384))
        self._g = Graphiti(os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687"), os.environ.get("NEO4J_USER", "neo4j"),
                           os.environ.get("NEO4J_PASSWORD", ""), llm_client=llm, embedder=emb, cross_encoder=BGERerankerClient())
        await clear_data(self._g.driver)
        await self._g.build_indices_and_constraints()
        self.episodes = 0
