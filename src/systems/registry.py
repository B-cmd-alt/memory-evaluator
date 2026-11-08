"""The ten systems + two baselines, with the metadata the evaluator UI renders."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Entry:
    key: str
    cls_path: str
    label: str
    family: str
    year: int
    kind: str            # baseline | classic | modern
    lineage: str         # which paradigm line it sits on
    provenance: str      # official package | faithful re-implementation | baseline
    notes: str = ""


REGISTRY: tuple[Entry, ...] = (
    Entry("full_context", "src.systems.baselines.FullContextBaseline", "Full context (no retrieval)", "no retrieval", 2023, "baseline", "compression", "baseline"),
    Entry("bm25", "src.systems.baselines.BM25Baseline", "BM25 top-5", "sparse retrieval", 2023, "baseline", "retrieval", "baseline"),
    Entry("vector_rag", "src.systems.baselines.VectorRAG", "Vanilla vector RAG", "dense retrieval", 2023, "classic", "retrieval", "baseline"),
    Entry("hybrid_rerank", "src.systems.baselines.HybridRerank", "Hybrid search + reranker", "hybrid retrieval", 2024, "modern", "retrieval", "faithful re-implementation",
          "BM25 ∪ dense candidates, bge-reranker-base cross-encoder"),
    Entry("ms_graphrag", "src.systems.graphrag_adapter.GraphRAGAdapter", "Microsoft GraphRAG", "graph RAG", 2024, "modern", "graph", "official package"),
    Entry("mem0", "src.systems.mem0_adapter.Mem0Adapter", "Mem0 (OSS)", "extracted-facts memory layer", 2024, "modern", "extraction", "official package"),
    Entry("zep_graphiti", "src.systems.graphiti_adapter.GraphitiAdapter", "Zep / Graphiti", "temporal knowledge graph", 2024, "modern", "graph", "official package", "Kuzu embedded driver"),
    Entry("letta", "src.systems.letta_adapter.LettaAdapter", "Letta (MemGPT)", "OS-style self-managed memory", 2023, "modern", "self-managed", "official package", "blocked in this env"),
    Entry("mastra_om", "src.systems.mastra_om.MastraObservationalMemory", "Mastra Observational Memory", "observation log", 2026, "modern", "compression", "faithful re-implementation"),
    Entry("anthropic_memory_tool", "src.systems.anthropic_memory_tool.MemoryToolAdapter", "Anthropic memory tool (files)", "model-managed files", 2025, "modern", "self-managed", "faithful re-implementation",
          "memory_20250818 command set, client-executed"),
    Entry("generative_agents", "src.systems.generative_agents.GenerativeAgentsMemory", "Generative Agents memory stream", "scored stream + reflection", 2023, "classic", "stream", "faithful re-implementation"),
    Entry("a_mem", "src.systems.amem_adapter.AMemAdapter", "A-MEM", "agentic note network", 2025, "modern", "stream", "official package"),
)

BY_KEY = {e.key: e for e in REGISTRY}


def load(key: str):
    import importlib
    e = BY_KEY[key]
    mod, cls = e.cls_path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), cls)
