# Feasibility round — S1 (90 memories, 25 questions)

Status: OK = ran end to end · PARTIAL = query errors · FAILED_INGEST · BLOCKED = could not be built (reason recorded).
Judge: strict-judge v3 via the shim (sonnet alias). Costs are measured from the shim log, not estimated.

| system | kind | provenance | status | accuracy | T1 | T2 | T3 | T4 | T5 | ingest LLM calls/mem | ingest s | read calls/q | read prompt tok/q | p50 s | p95 s | $ ingest | $ queries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Full context (no retrieval) | baseline | baseline | OK | **0.94** | 1.0 | 1.0 | 1.0 | 0.7 | 1.0 | 0.0 | 0.0 | 1.0 | 6841 | 4.53 | 5.68 | 0 | 0.702 |
| BM25 top-5 | baseline | baseline | OK | **0.92** | 1.0 | 0.9 | 1.0 | 0.7 | 1.0 | 0.0 | 0.0 | 1.0 | 1754 | 5.25 | 8.84 | 0 | 0.194 |
| Vanilla vector RAG | classic | baseline | OK | **0.93** | 1.0 | 1.0 | 0.98 | 0.67 | 1.0 | 0.0 | 12.2 | 1.0 | 1736 | 5.01 | 6.17 | 0 | 0.192 |
| Hybrid search + reranker | modern | faithful re-implementation | OK | **0.932** | 1.0 | 0.98 | 0.98 | 0.7 | 1.0 | 0.0 | 4.5 | 1.0 | 1762 | 6.58 | 8.42 | 0 | 0.197 |
| Microsoft GraphRAG | modern | official package | OK | **0.978** | 1.0 | 1.0 | 1.0 | 0.9 | 0.99 | 3.3 | 684.1 | 1.0 | 11504 | 19.1 | 21.11 | 5.8536 | 1.279 |
| Mem0 (OSS) | modern | official package | OK | **0.93** | 1.0 | 1.0 | 0.98 | 0.67 | 1.0 | 1.0 | 559.1 | 1.0 | 2670 | 4.6 | 5.45 | 1.1893 | 0.286 |
| Zep / Graphiti | modern | official package | OK | **0.654** | 0.8 | 0.55 | 0.62 | 0.3 | 1.0 | 4.32 | 2169.1 | 1.0 | 1736 | 4.52 | 6.08 | 7.7834 | 0.192 |
| Letta (MemGPT) | modern | official package | **BLOCKED** | — | | | | | | | | | | | | | |  <sub>RuntimeError: letta not installable on Python 3.13 in this container; needs a 3.11/3.12 env + `letta server`</sub>
| Mastra Observational Memory | modern | faithful re-implementation | OK | **0.94** | 1.0 | 1.0 | 1.0 | 0.7 | 1.0 | 0.1 | 70.7 | 1.0 | 6669 | 4.72 | 6.33 | 0.1531 | 0.685 |
| Anthropic memory tool (files) | modern | faithful re-implementation | OK | **0.83** | 0.8 | 1.0 | 0.85 | 0.7 | 0.8 | 0.38 | 326.0 | 2.96 | 12265 | 14.14 | 17.14 | 0.961 | 1.105 |
| Generative Agents memory stream | classic | faithful re-implementation | OK | **0.41** | 0.35 | 0.27 | 0.25 | 0.18 | 1.0 | 0.12 | 71.2 | 1.0 | 2232 | 4.78 | 6.03 | 0.1135 | 0.243 |
| A-MEM | modern | official package | OK | **0.934** | 1.0 | 1.0 | 1.0 | 0.67 | 1.0 | 0.99 | 1077.9 | 1.0 | 1782 | 5.11 | 7.87 | 2.6137 | 0.194 |

Accuracy at S1 has a ~7pp inter-run noise envelope (RESULTS_SUMMARY §1); differences inside it are ties. This round decides feasibility and cost class, not ranking.
