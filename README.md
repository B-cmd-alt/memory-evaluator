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

## Round 2 — SixAxis at S1 (11 systems × 3 seeded worlds × 2 generator tiers = 66 cells, strict-judge v3 pinned to Sonnet)

Pre-registered in [`experiments/prereg_sixaxis.md`](experiments/prereg_sixaxis.md) before any run; full write-up with
the per-axis tables, the hypothesis verdicts and the nine benchmark corrections the round forced:
**[`docs/ROUND2_RESULTS.md`](docs/ROUND2_RESULTS.md)**. Mean ± std over seeds, Sonnet tier; noise envelope 0.06
(replication std of the baselines; judge-protocol 0.011 and prompt 0.022 are both smaller); decoy acceptance 0.00 on all 66 cells.

| system | A1 recall | A2 current / stale | A3 forgot · ctrl · leak | A4 fooled | write calls/mem | read tok/q |
|---|---|---|---|---|---|---|
| Microsoft GraphRAG (context builder) | **1.00** | 1.00 / 0.00 | 1.00 · 0.94 · 0.00 | 0.00 | 3.10 | 11.8K |
| Mem0 | 0.96 | 0.77 / 0.00 | 1.00 · 1.00 · 0.06 | 0.00 | 1.00 | 2.3K |
| Mastra OM (re-impl) | 0.96 | 0.85 / 0.00 | 1.00 · 1.00 · 0.00 | 0.00 | 0.11 | 4.5K |
| Full context | 0.95 | 0.85 / 0.00 | 1.00 · 1.00 · 0.00 | 0.00 | 0 | 4.8K |
| Hybrid + reranker | 0.88 | 0.92 / 0.00 | 1.00 · 1.00 · 0.00 | 0.00 | 0 | 1.7K |
| Vector RAG | 0.88 | 0.70 / 0.30 | 1.00 · 1.00 · 0.00 | 0.00 | 0 | 1.7K |
| A-MEM | 0.88 | 0.83 / 0.17 | 1.00 · 1.00 · 0.00 | 0.00 | 0.99 | 1.7K |
| Anthropic memory tool (re-impl) | 0.76 | 0.55 / 0.15 | 1.00 · 0.83 · 0.00 | 0.17 | 0.39 | 9.1K |
| Zep / Graphiti | 0.69 | 0.22 / 0.37 | 1.00 · 0.94 · 0.00 | 0.00 | 4.26 | 1.7K |
| BM25 | 0.60 | 0.55 / 0.37 | 1.00 · 1.00 · 0.00 | 0.17 | 0 | 1.6K |
| Generative Agents (re-impl) | 0.41 | 0.15 / 0.15 | 1.00 · 0.67 · **0.45** | **0.83** | 0.12 | 1.9K |

What it says, in one breath: on **recall** the 2023 upper bound (full context) is not beaten — GraphRAG, Mem0 and
Mastra tie it inside the envelope — while the retrieval-window systems lose on multi-session questions and BM25 falls
to 0.60 on this harder world; the generations separate on **updates** (every modern system's beyond-envelope gain is
there; vector RAG and BM25 serve the superseded value 30–37% of the time, Graphiti is the one modern system that is
worse than 2023 here); **deletion** discriminates on ghost residue (Generative Agents' reflections still name the
deleted person 45% of the time) and collateral damage, and on *semantics* — Mem0 by provenance alone forgot only 61%
with 67% leaks until the adapter tracked the ids its `add()` returned; **labelled poison** fools almost nobody whose
answer path shows the generator the whole note; **write cost** spans ×39. Hypotheses: H2 and H4 hold, H3 is falsified
(the architecture gap across the full lineage is an order of magnitude larger than the protocol noise), H1 waits for
S2 and S3. The weaker generator tier flattens the update axis for everyone (Kendall τ 0.60 vs 0.78 on recall).

The benchmark doubted itself first: a deletion-case builder that was not consistent under cumulative application,
leak strings that other notes legitimately contained, poison probes aimed at deleted memories, two adapter lifecycle
faults and one lost-record bug were all caught by reading a record's evidence instead of its number, fixed at the
definition level and re-applied uniformly. They are the first section of the write-up.

## Round 3 — the A2/A3 closing round (one world, 58 update keys, 34 deletion cases, three judge families)

Pre-registered in [`experiments/prereg_round3.md`](experiments/prereg_round3.md) (two amendments, both committed before the
round finished); write-up: **[`docs/ROUND3_RESULTS.md`](docs/ROUND3_RESULTS.md)**. Round 3 exists because round 2's update and
deletion differences rested on 5 and 6–7 items per seed. Paired item-level statistics, Holm-corrected.

All findings are inside one synthetic ~4K-token world (262 dated notes):

- **On updates, no system beats the 2023 upper bound.** Full context scores FAMA 1.00. Hybrid + reranker and Mem0 are
  indistinguishable from it. Mastra and GraphRAG are below it in the primary analysis (−0.10, −0.16) and indistinguishable once
  8 ambiguous items are set aside. Five systems are below it either way (primary Δ FAMA): vector RAG −0.17 [−0.31, −0.03],
  BM25 −0.38, the Anthropic memory tool −0.67 (it does not record later changes in its own files; stale 0.29), Graphiti −0.88,
  Generative Agents −0.83.
- **Graphiti's update deficit holds (H7), but the cause depends on its read configuration.** With round 2's 5-edge read,
  surviving old edges produce stale answers. It invalidates an old edge in 38 of 48 keys but retires all of them in only 5.
  With its best recipe (FAMA 0.45 vs vector RAG's 0.83), the gap is mostly facts missing from the graph.
- **The reranker's stale-answer advantage is in the predicted direction but not significant after Holm** (+0.09, p = 0.06).
  All of vector RAG's stale answers come from notes worded "moved from A to B", and every one is a retrieval miss.
- **Deletion: Mem0 (3/11) and Generative Agents (2/11) are the systems that leak**, with wide CIs. Mem0 leaks through facts it
  merged across notes; Generative Agents through reflections, and it still answers with a deleted city in 3 of 34 cases.
  Rebuild-based systems are clean. Collateral damage is small everywhere; the memory tool lost 1 of 24 valid controls, and
  H9 is unresolved.
- **Judge-robust:** the Sonnet verdicts agree with GPT-6.1 and Gemini-3.8 at κ 0.87–0.94 on 1,350 round-3 judgements, and no
  hypothesis flips.
- **Two more gold problems, and one A4 artifact, were found.**
  - Updates dated before what they supersede. This also invalidates 3 of the 9 round-2 update items in S1 seeds 123 and 456.
  - Implicit role updates that do not entail the old role ended.
  - The A4 "fooled" rule flags abstentions that mention the rumor; no system was actually fooled.

## What the whole study concludes

1. **Where a memory store still fits in a context window, no memory system we tested beats putting everything in the
   prompt, on recall or on updates.** This rests on synthetic worlds of 3.5K–28K tokens and, for updates, on one world.
   What the retrieval systems and Mem0 buy there is read cost: 1.6–2.2K vs 11.5K prompt tokens per query. Mastra (11K) and
   GraphRAG (12K) do not save read tokens either.
2. **What separates memory systems is whether their context reliably contains the newest relevant note, and how deletion
   propagates through what they derive.**
   - A reranker and a fact store (Mem0) keep the newest note in front of the generator.
   - A top-k window (BM25, vector), model-written files that are not updated (the memory tool), an edge graph that misses or
     keeps facts (Graphiti) and importance-weighted reflections (Generative Agents) do not.
   - Extraction and reflection create derived text that a per-note delete does not reach.
3. **The evaluation moves results as much as the systems do.** Over rounds 2 and 3, eleven corrections had to be made by
   reading per-item evidence. One of them, a gold-label error (updates dated before what they supersede), had inflated a
   round-2 headline; an A4 scoring rule may have produced another. A calibrated judge was necessary, and it was not
   sufficient.

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

Rounds 1, 2 (S1) and 3 (the A2/A3 closing round) are complete. The full S2/S3 matrix is deliberately not run: round 3
showed that the recall axis is settled at these scales, and the remaining open questions are scale beyond a context window and
real human labels. Known limits, stated up front: the judge's human-proxy calibration uses two
careful independent LLM labelers, not humans (a blind 100-item human set is exported in `results/round3/human_labels/`);
verdicts are cross-checked by GPT and Gemini judges; N = 22–24 questions
and 1–3 poison probes per S1 seed; A4 measures labelled-poison robustness; Mastra OM, the Anthropic
memory tool and Generative Agents are faithful re-implementations from their public descriptions, not
vendor code; Letta is blocked in the build environment. Every number above links to the record that
produced it.

MIT licensed.
