# Memory System Evaluator

**Ten agent-memory systems, one model, one embedder, one calibrated judge — and every number traceable to the question, the retrieved context, the answer and the judge's reasoning that produced it.**

**Try it live: [https://memory-evaluator-862698383291.us-central1.run.app](https://memory-evaluator-862698383291.us-central1.run.app)** — be the judge, ask the memory, break the memory (details below).
Evidence page: **[bb.style/projects/memory-evaluator.html](https://bb.style/projects/memory-evaluator.html)**
(also at [`site/index.html`](site/index.html), self-contained, opens from a file).

## Why

Memory vendors report benchmark scores in the nineties; independent re-runs land in the
thirties to forties. Audits of the most-quoted benchmark found 6.4% of its answer key wrong
and a judge that accepts most intentionally wrong answers. My own earlier study found standard
LLM-as-judge inflating abstention-heavy results by 6–22 points and briefly manufacturing a
finding. Before comparing memory systems, the evaluation itself had to be something I could
trust. This repository is that evaluation, plus the first round of results.

## What is here

| piece | what it does |
|---|---|
| [`BENCHMARK_DESIGN.md`](BENCHMARK_DESIGN.md) | **SixAxis**: a self-built benchmark that takes the core of LongMemEval, MemoryAgentBench, BEAM, ConvoMem, Memora, MemOps, ForgetEval and MemDelta and fixes each one's documented flaw. Six axes: recall, update & forgetting, deletion, adversarial memory, cost, invariance. |
| [`experiments/prereg_sixaxis.md`](experiments/prereg_sixaxis.md) | Four hypotheses pre-registered before any round-2 run. |
| [`src/systems/`](src/systems) | 2 baselines + 10 systems on one `MemorySystem` contract: full context, BM25, vector RAG, hybrid + reranker, **Microsoft GraphRAG**, **Mem0**, **Zep / Graphiti**, **A-MEM** (official packages), and faithful re-implementations of **Mastra Observational Memory**, the **Anthropic memory tool** command set and the **Generative Agents** memory stream; Letta (blocked in the build environment, recorded). [`registry.py`](src/systems/registry.py) carries lineage and provenance. |
| [`tools/openai_shim.py`](tools/openai_shim.py) | An OpenAI-compatible server so every third-party system runs on the *same* generator and embedding model as the baselines (the MemDelta rule), with a per-call cost ledger. |
| [`src/data/sixaxis_generator.py`](src/data/sixaxis_generator.py) | Seeded worlds where every question carries evidence pointers, a stale answer where one exists, two decoys, and a timestamp; deletion and poisoning cases paired with probes; a mechanical `ground_check()` that re-derives every gold answer from its cited memories. |
| [`src/evaluation/strict_judge.py`](src/evaluation/strict_judge.py) | Strict-judge v3: the asymmetric abstention protocol calibrated at κ = 0.87 (binary) / 0.92 (weighted) against independent labels. The methodology and the three judge failure modes: [`docs/JUDGE_METHODOLOGY.md`](docs/JUDGE_METHODOLOGY.md). Calibration data: [`results/judge_calibration/`](results/judge_calibration). |
| [`run_feasibility.py`](run_feasibility.py), [`run_sixaxis.py`](run_sixaxis.py) | Round-1 and round-2 runners. [`tools/render_site.py`](tools/render_site.py) renders the evidence page from the per-question records. |

## Round 1 — feasibility (S1: 90 memories, 25 questions, Claude Sonnet, bge-small, strict-judge v3)

| system | status | accuracy | write LLM calls / memory | ingest | read prompt tokens / query | p50 latency |
|---|---|---|---|---|---|---|
| Full context (no retrieval) | OK | 0.94 | 0 | 0 s | 6,841 | 4.5 s |
| BM25 top-5 | OK | 0.92 | 0 | 0 s | 1,754 | 5.3 s |
| Vanilla vector RAG | OK | 0.93 | 0 | 12 s | 1,736 | 5.0 s |
| Hybrid search + reranker | OK | 0.93 | 0 | 5 s | 1,762 | 6.6 s |
| Microsoft GraphRAG | OK | **0.978** * | 3.3 | 684 s · $5.9 | 11,504 | 19.1 s |
| Mem0 | OK | 0.93 | 1.0 | 559 s · $1.2 | 2,670 | 4.6 s |
| Zep / Graphiti | OK | 0.654 (updates 0.30) | 4.3 | 2,169 s · $7.8 | 1,736 | 4.5 s |
| Mastra Observational Memory (re-impl) | OK | 0.94 | 0.1 | 71 s | 6,669 | 4.7 s |
| Anthropic memory tool (re-impl) | OK | 0.83 | 0.38 | 326 s | 12,265 | 14.1 s |
| Generative Agents memory stream (re-impl) | OK | 0.41 | 0.12 | 71 s | 2,232 | 4.8 s |
| A-MEM | OK | 0.93 | 0.99 | 1,078 s · $2.6 | 1,782 | 5.1 s |
| Letta (MemGPT) | blocked | — | — | no Python 3.13 release in the build container | | |

\* GraphRAG answers with its own local-search prompt and LLM call, not the shared answer prompt; its lead is partly that confound and is fixed in round 2 by using it as a context builder.

What round 1 established (details and the round-2 roster in [`results/feasibility/ROUND2_ROSTER.md`](results/feasibility/ROUND2_ROSTER.md)):

- **Six systems from 2023–2025 tie on recall** inside the S1 noise band. A keyword index and an agentic memory layer are indistinguishable on this axis — pre-registered H1.
- **Write-path cost spans two orders of magnitude**: 0 → 4.3 LLM calls per memory, measured from the shim ledger, not estimated — H4.
- **The temporal knowledge graph fails updates** (0.30): superseded facts stay valid in the graph.
- **Model-managed files** is the most expensive read path (12K tokens, 3 calls per query) and the weakest modern system.
- **Generative Agents' scoring is not built for factual QA**: recency refresh on access lets a few high-importance reflections dominate every retrieval. Reported as-is; nothing was tuned to make any system look good.
- Three integration faults were environment, not system, and are recorded with their fixes (GraphRAG 3.x CLI and config schema, graphiti-core's Kuzu driver omitting its own full-text indexes, Mem0 2.2's search API).

Every row links to its per-question records in [`results/feasibility/`](results/feasibility): question, reference, retrieved context, answer, judge verdict and reasoning, tokens, latency.

## The interactive app

The same code, hosted on Cloud Run (`app/`, scale-to-zero, one small instance):

| tab | what you do | what it costs |
|---|---|---|
| **Be the Judge** | Score a (question, reference, answer) triple from the 100-item calibration set, then see what strict-judge v3, two independent human-proxy labelers and a weaker LLM judge said, and *why the item is hard* (false abstention, correct abstention, …). Your vote is stored anonymously and shown as crowd agreement with each judge. | nothing — all precomputed |
| **Ask the Memory** | Type any question about the S1 world (90 dated engineering-team notes). Six systems answer live — full context, BM25, vector RAG, hybrid + reranker, Mastra-style observational memory, Generative-Agents retrieval — each with the memories it retrieved, latency and dollar cost. The 25 canonical questions also show every round-1 system's recorded answer and strict-judge verdict. | one OpenRouter call per system (Claude Sonnet 5.5) |
| **Break the Memory** | Inject a fact, an update that supersedes it and a poisoned note, then ask. Each system's answer gets a verdict — current / stale / poisoned / abstained — plus whether it even retrieved your notes. The update axis and the adversarial axis of SixAxis, on a world you wrote. | ingest + one call per system |

Guardrails: a daily spend ledger in GCS (default $1/day) and a per-visitor daily cap on live
requests; when the budget is spent the judge game and the precomputed answers keep working.
Votes store a hashed IP and a random session id only. Deploy your own with
[`deploy.sh`](deploy.sh) (Cloud Build → Artifact Registry → Cloud Run, OpenRouter key from Secret Manager).

## Run it

```bash
pip install -e ".[dev]"            # baselines + re-implementations; add [systems] for Mem0/Graphiti/GraphRAG/A-MEM
pytest -q                          # 19 tests: adapter contract, generator ground check, the app offline

# every system needs one OpenAI-compatible endpoint; point OPENAI_BASE_URL at your provider,
# or run the bundled shim (which backs onto the Claude Code CLI and a local bge-small embedder):
python -m tools.openai_shim --port 8008 &
export OPENAI_BASE_URL=http://127.0.0.1:8008/v1 OPENAI_API_KEY=shim

python run_feasibility.py --systems bm25 full_context vector_rag --limit 5     # round 1, cheap subset
python -m src.data.sixaxis_generator --scale S1 --seed 42                      # a fresh SixAxis world
python run_sixaxis.py --data results/sixaxis_S1_seed42.json --systems bm25 full_context   # round 2
python tools/render_site.py                                                    # site/index.html
uvicorn app.main:app --port 8090                                               # the interactive app, locally
```

## Lineage

The judge protocol, the six-axis framing, poisoning and entity-level deletion come from a 2025
study of the same question on hand-built architectures ([`docs/V1_STUDY_RESULTS.md`](docs/V1_STUDY_RESULTS.md)).
That study's own finding — that no architecture dominates, and that the judge moved results by
10–15 points without any architectural change — is why this repository evaluates *real* systems
with a *calibrated* judge and ships the per-question evidence instead of a leaderboard.

## Status and limits

Round 1 is complete. Round 2 (three replications, two model tiers, all six axes, GraphRAG as
context builder) is the next milestone; its hypotheses are pre-registered. Known limits, stated
up front: the judge's human-proxy calibration uses two careful independent labelers, not paid
annotators; the judge is Claude-family only; N = 25–50 questions per scale; Mastra OM, the
Anthropic memory tool and Generative Agents are faithful re-implementations from their public
descriptions, not vendor code.

MIT licensed.
