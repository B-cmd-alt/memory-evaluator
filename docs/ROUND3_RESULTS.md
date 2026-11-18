# SixAxis round 3 — the A2/A3 closing round (one R3 world, Sonnet tier, item-level paired statistics)

Pre-registration: [`experiments/prereg_round3.md`](../experiments/prereg_round3.md), committed before any run, with two
amendments, both committed before the round finished. World: `results/sixaxis_R3_seed42.json`. Records:
`results/round3/<system>_R3_seed42_sonnet_rep1.json`, one per system, with every answer, retrieved context and judge reasoning.
Analysis: `python -m tools.analyze_round3` → `results/round3/analysis.json`. Cross-family re-judge:
`python -m tools.rejudge_round3` → `results/round3/rejudge/`.

## Why this round existed

Round 2 found that recall (A1) ties at the top and that the systems separate on updates (A2), deletion (A3) and write cost
(A5). But A2 rested on 5 update questions per seed and A3 on 6–7 deletion cases, so most of those differences were one or
two items. Round 3 asked one question: do the A2/A3 differences survive at N ≈ 60 update keys and N = 34 deletion cases,
judged by more than one model family?

## Setup

| | |
|---|---|
| world | R3 seed 42: 262 memories (~4K tokens), 30 people, 15 projects; 65 T4 update keys (34 *explicit* "moved from A to B", 31 *implicit* "is now B"), 10 T1, 10 T3, 5 T2, 6 T5; 34 cumulative deletion cases (12 single, 11 entity, 11 intent) in a seeded interleaved order; 2 poison notes |
| systems | full context, BM25, vector RAG, hybrid + reranker, Mastra OM, Generative Agents, Anthropic memory tool, Mem0, Zep/Graphiti (all axes); Microsoft GraphRAG (A1 + A2) |
| generator / judge | Sonnet via the shim; strict-judge v3 pinned to Sonnet; every LLM-branch judgement re-judged by `openai/gpt-6.1-terra` and `google/gemini-3.8-flash` |
| statistics | one world, one replication; paired over items; 95% bootstrap CI (10,000 resamples) + exact McNemar; Holm across H5–H9 |
| cost control | shared SQLite prompt cache across the four runner shims (rebuild-on-delete replays its unchanged prefix; decoy judging is shared). Write cost is the first ingest, which never hits the cache |

## The benchmark doubted itself first: two more gold problems (one also invalidates part of round 2)

**10. Updates dated before what they supersede.** Memory dates are drawn within a session week, so a superseding note
could carry an *earlier* date than the fact it supersedes. In those items the "stale" value is the newer one by date, and
bm25's first "stale" answers were simply reading the dates correctly. This affects 7 of 65 R3 T4 keys and 1 T3 item. It also
affects **round 2: S1 seeds 123 and 456 had 3 of their 9 T4 items and 3 of 10 T3 items inverted, and S2 seed 42 had 1 T4 and 1 T3.**
Round 2's A2 numbers, and its stale rates in particular, are therefore partly gold error (see "What round 3 changes about
round 2" below). Fix: `inadmissible_updates()` excludes those items for every system, `ground_check()` flags them, and the
generator now forces strictly later dates (amendment 1). **The A2 set is 58 keys.**

**11. Implicit role updates are ambiguous.** "X is now data lead of Beacon" does not say the person left their earlier role,
since people can hold two. Both non-Claude judges and the record evidence showed answers naming both roles. Tracker,
city and budget have one value at a time and are not affected. Eight T4 keys, 2 T3 items and 3 deletion controls rest on
such a note. They stay in the primary analysis and are dropped in a reported **sensitivity pass**; the generator wording
is now "X's role changed on <date>: now <role>" (amendment 2). It matters: most of the gap between full context and the
top retrieval systems in the primary A2 numbers is these 8 items.

Two judge-boundary cases also showed up in the records. Neither changes any verdict, and both are reported, not re-scored:

- **Mastra's 3 "lost" controls** are answers that open with "I don't have enough information" and then state the correct
  value. Strict-v3 classifies them as abstentions by design, so Mastra lost no data.
- **Graphiti's one "fooled" poison** is an answer that names the poison value only to call it unverified.

## Judge robustness: pre-declared threshold met

| comparison (LLM-branch items only) | n | κ (binary) |
|---|---|---|
| Sonnet vs GPT-6.1-terra, round-3 records | 1,201 | **0.927** |
| Sonnet vs Gemini-3.8-flash, round-3 records | 1,201 | **0.934** |
| Sonnet strict-v3 vs proxy labels, calibration set (all 100) | 100 | 0.915 |
| GPT strict-v3 vs proxy labels, calibration set (all 100) | 100 | 0.871 |
| Gemini strict-v3 vs proxy labels, calibration set (all 100) | 100 | 0.914 |
| Sonnet / GPT / Gemini vs proxy labels, calibration LLM-branch only | 52 | 0.89 / 0.77 / 0.89 (CIs ≈ ±0.2) |

Both families reach κ ≥ 0.80 against Sonnet on the round-3 items, so by the pre-registered rule the Sonnet tables are primary.
**No hypothesis verdict changes under either non-Claude judge.** When the judges disagree, Sonnet is the stricter one:
both alternatives disagree with it on largely the same items, and in the same direction. On the calibration set's hard,
LLM-branch-only items, GPT agrees least with the proxy labels (0.77, CI 0.51–0.95). The proxy labels are LLM agents, not humans
(see "Human labels" below).

## A2 — updates (58 admissible keys; FAMA = current − stale)

| system | current | stale | FAMA | explicit cur / stale (31) | implicit cur / stale (27) | FAMA, sensitivity (50 keys) | Δ FAMA vs full context, sensitivity [95% CI] |
|---|---|---|---|---|---|---|---|
| Full context | 1.00 | 0.00 | **1.00** | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 | — |
| Hybrid + reranker | 0.95 | 0.00 | 0.95 | 1.00 / 0.00 | 0.89 / 0.00 | 1.00 | 0.00 [0.00, 0.00] |
| Mem0 | 0.95 | 0.00 | 0.95 | 1.00 / 0.00 | 0.89 / 0.00 | 1.00 | 0.00 [0.00, 0.00] |
| Mastra OM | 0.90 | 0.00 | 0.90 | 0.97 / 0.00 | 0.82 / 0.00 | 0.98 | −0.02 [−0.06, 0.00] |
| GraphRAG | 0.85 | 0.00 | 0.85 | 0.94 / 0.00 | 0.74 / 0.00 | 0.96 | −0.04 [−0.10, 0.00] |
| Vector RAG | 0.91 | 0.09 | 0.83 | 0.84 / 0.16 | 1.00 / 0.00 | 0.80 | **−0.20 [−0.36, −0.04]** |
| BM25 | 0.78 | 0.16 | 0.62 | 0.71 / 0.16 | 0.85 / 0.15 | 0.56 | **−0.44 [−0.66, −0.24]** |
| Zep / Graphiti (round-2 read config) | 0.38 | 0.26 | 0.12 | 0.45 / 0.26 | 0.30 / 0.26 | 0.22 | **−0.78 [−0.98, −0.58]** |
| Generative Agents | 0.21 | 0.03 | 0.17 | 0.19 / 0.03 | 0.22 / 0.04 | 0.18 | **−0.82 [−0.94, −0.68]** |
| Anthropic memory tool | *pending* | | | | | | |

What survives at N = 58:

- **Updates are not where 2026 systems beat 2023 ones.** The 2023 upper bound, full context, is at the top. Hybrid +
  reranker and Mem0 tie it, and so do Mastra and GraphRAG once the ambiguous role items are dropped.
- **What separates systems on updates is retrieval architecture, not generation.** Systems whose context can miss the
  newer note fall below the others: a top-5 lexical or vector window (BM25, vector RAG), an edge-fact graph that does not
  retire superseded facts (Graphiti), and importance-weighted reflections (Generative Agents).
- **The stale failures of vector RAG are retrieval misses, not ranking inversions.** In every stale record the newer note
  is not in the top 5. All of them are in the explicit stratum; on implicit updates vector RAG is never stale.

### Hypotheses (Sonnet judge; identical verdicts under GPT and Gemini and in the sensitivity pass)

| | claim | estimate [95% CI] | McNemar p (Holm) | verdict |
|---|---|---|---|---|
| **H5** | stale(vector RAG) > stale(hybrid + reranker) | +0.086 [+0.017, +0.155] | 0.063 (0.19) | **CI excludes 0, but not significant after Holm.** Five discordant keys, all in the predicted direction |
| **H6** | H5 also within the implicit stratum | 0.00 [0, 0] | 1.0 | **unresolved.** Vector RAG is never stale on implicit updates; its stale answers all come from the "moved from A to B" wording |
| **H7** | Graphiti's best read arm has lower FAMA than vector RAG | −0.379 [−0.586, −0.172] | < 0.001 (< 0.001) | **holds** |

### Graphiti: the claim stands, the mechanism is confirmed, and the round-2 magnitude was partly configuration

| Graphiti read arm (one ingest) | FAMA |
|---|---|
| r2 — round-2 config: 5 edges | 0.12 |
| default — library default: 10 edges | 0.38 |
| filter — drop edges Graphiti marked invalid/expired | 0.31 |
| xenc — Graphiti's cross-encoder recipe, 10 edges | **0.45** (best) |
| *vector RAG, for reference* | *0.83* |

The no-LLM invalidation diagnostic answers the mechanism question. For 55 of the 65 keys an edge stating the superseded
value is still in the graph, and **for only 5 of those 55 were all such edges marked invalid**. Superseded facts mostly stay
current in Graphiti's graph, which is the round-2 hypothesis. That is also why the "filter" arm does not help.

Round 2's 5-edge configuration cost Graphiti about 0.26–0.33 FAMA relative to the library defaults. Even its best
configuration stays 0.38 below vector RAG. Graphiti ran on Kuzu, not Neo4j, behind an OpenAI-compatible shim in front of Claude;
that remains a declared limitation, but it would have to explain a gap that the diagnostic attributes to invalidation.

## A3 — deletion (34 cumulative cases; 11 leak-measurable)

| system | forgot | control intact (valid controls) | leaked (of 11) | delete method |
|---|---|---|---|---|
| Full context · BM25 · Hybrid + reranker | 1.00 | 1.00 | 0 | rebuild |
| Vector RAG | 1.00 | 0.97 (30) | 0 | rebuild |
| Mastra OM | 1.00 | 0.90 (30)† | 0 | rebuild (re-observes the log) |
| Zep / Graphiti | 1.00 | 1.00 (21) | 0 | native `remove_episode` |
| Mem0 (provenance ∪ returned ids) | 0.97 | 1.00 (30) | **3** (27%, CI 0–55%) | native, by tracked ids |
| Generative Agents | **0.91** | 1.00 (3)‡ | 2 (18%, CI 0–45%) | rebuild of raw memories; reflections survive |
| Anthropic memory tool | *pending* | | | file edits by the model |

† The three misses are hedged abstentions that state the correct value (see above). ‡ Generative Agents answered only 3
controls correctly *before* any delete, so its collateral-damage number is uninformative.

- **H8** (Generative Agents leaks more than every rebuild system): +0.18 [0.00, +0.45] against each, McNemar p = 0.5.
  **Unresolved.** With 11 leak-measurable cases this test only resolves a large effect, and round 2's 0.45 does not
  reproduce (2/11 here). Generative Agents' ghost residue does show at the answer level: in 3 of 34 cases it still gives the
  deleted city after the delete, because a reflection kept it.
- **Mem0 is the system that leaks.** That is exploratory, not pre-registered, and the CI is wide. The mechanism is visible in
  the records: Mem0's extraction writes facts about the deleted person into *other people's* memories ("Noah Berg became
  design lead of Orbit … succeeding Zofia Nowak"), so deleting the person's own notes, even with every returned id
  tracked, leaves their name served. This is the same delete-semantics problem round 2 found with provenance-only deletion,
  one level further down.
- **H9** (the memory tool loses more valid controls than full context): *pending.*
- Deletion by case position (controls intact, cases 0–11 / 12–23 / 24–33): no system degrades in later cases. Vector
  RAG's one miss is in the first third, and Mastra's hedged abstentions are in the later two thirds. Position does not
  confound shape, because shapes are interleaved.

## A1, A4, A5 (reported for completeness)

- **A1** (T1/T2/T3/T5, admissible): GraphRAG 1.00, hybrid 0.98, Mem0 0.97, full context / vector RAG / Mastra 0.95, BM25 0.86,
  Graphiti 0.63, Generative Agents 0.30. This is the same picture as round 2. Decoy acceptance is 0.00 on every record
  (192 decoys each).
- **A4** (exploratory, 2 poison notes): the only system that served a poison claim was Graphiti, once, and its answer
  called it unverified. Wherever a poison note was retrieved and the context is inspectable (every system except full context and Mastra,
  whose context is the whole store or log), its "unverified" label survived into what the generator saw. Graphiti's edge
  fact reads "An unverified rumor on Slack claims …". A4 does not discriminate between these systems, as expected after its demotion.
- **A5** (first ingest, measured): write LLM calls per memory are 0 for full context, BM25, vector RAG and hybrid; Mastra
  0.12; Generative Agents 0.12; Mem0 1.0; GraphRAG 2.8; Graphiti 3.85. Ingest dollars at Sonnet rates through the CLI:
  Mastra $1.1, Mem0 $3.0, GraphRAG $14.7, Graphiti $18.9, for 262 memories.

## What round 3 changes about round 2

| round-2 statement | round 3 |
|---|---|
| "The generations separate on updates; every modern system's beyond-envelope gain is there" | **Not supported as worded.** The 2023 upper bound (full context) is at the top on updates too. The gains round 2 saw were partly the inverted-date gold (3 of 9 T4 items in S1 seeds 123 and 456) and partly a weak baseline (vector RAG). Updates separate *retrieval-window and graph architectures* from systems whose context reliably contains the newest note, not 2026 from 2023 |
| "Vector RAG and BM25 serve the superseded value 30–37% of the time" | 9% and 16% at N = 58, and only on explicit "moved from" notes for vector RAG. The reranker's advantage over vector RAG is real in direction but not significant after Holm |
| "Graphiti is worse than 2023 systems on updates; superseded edges stay valid" | **Holds** (H7), with the mechanism measured directly. The round-2 magnitude was inflated by a 5-edge read config |
| "Generative Agents' reflections still name the deleted person 45% of the time" | 18% (2/11), unresolved; plus 3/34 cases where the deleted city is still answered |
| "Mem0 leaks 6% even with id tracking" | 27% (3/11, wide CI): merged cross-person facts carry the deleted name |
| strict-v3 judge "κ = 0.87 vs human proxy" (Claude-only) | The verdicts are judge-robust across three families (κ 0.93 on 1,201 round-3 items); the proxy-label κ is still not a human κ |

## Human labels

`python -m tools.export_human_set` writes a blind 100-item set to `results/round3/human_labels/` (`items.csv` and a
self-contained `form.html`; `key.json` holds the hidden verdicts and inclusion weights). The set contains every
cross-judge disagreement, filled to 100 by stratified sampling over axis × verdict. Score a completed sheet with
`python -m tools.export_human_set --score labels.csv`. Until someone labels it, every "human" number in this repository is a
proxy number.

## Limitations

One synthetic world and one replication, so these are item-level, not world-level, inferences. 58 update keys resolve
differences of about 0.1 only when they are one-directional, and 11 leak-measurable cases resolve only large effects. Kuzu
instead of Neo4j for Graphiti. Re-implementations of Mastra OM, the memory tool and Generative Agents. GraphRAG was not run
on A3. A4 is demoted (post hoc, declared). The two gold problems above were found during this round; both are handled by
rules applied to every system, and both were committed before the round finished.
