# Round-1 verdicts → round-2 roster

Round 1 (S1: 90 memories, 25 questions, one Sonnet generator, one bge-small embedder, strict-judge v3)
decides **feasibility and cost class**, not ranking. Full table: `FEASIBILITY.md`; per-question
evidence: `<system>.json`; rendered: `../../site/index.html`.

## What round 1 established

| finding | evidence |
|---|---|
| **H1 holds at S1.** Six systems sit in 0.92–0.94, inside the ~7pp S1 noise band: BM25 (2023) and Mem0 / A-MEM / hybrid+reranker (2024–25) are indistinguishable on recall. | accuracy column |
| **GraphRAG is the outlier on top (0.978)** — but it is the one system whose answer is produced by its *own* local-search prompt and LLM call, not the shared answer prompt, and it reads ~11.5K tokens per query. Its lead is partly an answer-prompt confound; round 2 must use GraphRAG as a context builder and generate with the shared prompt. | `ms_graphrag.json`, read prompt tok/q |
| **Write-path cost spans two orders of magnitude** (H4): 0 calls/memory for retrieval baselines, 0.1 Mastra OM, 0.38 memory tool, 1.0 Mem0 / A-MEM, 3.3 GraphRAG, 4.3 Graphiti. Graphiti spent $7.8 and 36 min on 90 memories. | ingest columns (measured from the shim ledger) |
| **Temporal KG underperforms on updates at this scale**: Graphiti 0.654 overall, T4 0.30 — superseded edges stay valid in the graph (see the T4 cases). | `zep_graphiti.json` |
| **Model-managed files is the most expensive read path and the weakest of the modern systems**: 3 calls and 12K tokens per query, 0.83. | `anthropic_memory_tool.json` |
| **Generative Agents' scoring is not built for factual QA**: recency refresh on access lets a few high-importance reflections dominate every retrieval (0.41). Reported as-is; no tuning to win. | `generative_agents.json` |
| **Three integration faults were environment, not system**: GraphRAG 3.x CLI/config schema (init prompt, `completion_models` block, positional query), graphiti-core's deprecated Kuzu driver not creating its own FTS indexes, Letta not installable on Python 3.13. All recorded with the fix or the block reason. | adapter commit history |

## Roster

| system | round 2 at S1 (3 reps × 2 tiers) | round 2 at S2 | why |
|---|---|---|---|
| full_context, bm25, vector_rag, hybrid_rerank | yes | yes, 3 reps | zero write cost |
| mastra_om | yes | yes, 3 reps | 0.1 calls/memory |
| anthropic_memory_tool | yes | yes, 3 reps | 0.38 calls/memory; read path is the cost, bounded by question count |
| mem0 | yes | yes, 3 reps | 1 call/memory ≈ $10 per S2 ingest |
| a_mem | yes | 1 rep, flagged | 1 call/memory but slow (18 min / 90 memories) |
| ms_graphrag | yes, as context builder + shared prompt | 1 rep, flagged | 3.3 calls/memory ≈ $47 per S2 ingest; own-prompt confound fixed first |
| zep_graphiti | yes | 1 rep, flagged | 4.3 calls/memory ≈ $63 per S2 ingest |
| generative_agents | yes (as the 2023 reference point) | 1 rep | cheap; kept for the lineage comparison |
| letta | no | no | blocked in this environment; needs a 3.11/3.12 env with `letta server` |

Budget estimate (shim-ledger rates): S1 full roster ≈ $55 per tier, S2 as above ≈ $120. Both within
the autonomy band already approved for this work; S2 single-rep rows are reported with that caveat.
