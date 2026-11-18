# SixAxis round 3 — the A2/A3 closing round (one R3 world, Sonnet tier, item-level paired statistics)

Pre-registration: [`experiments/prereg_round3.md`](../experiments/prereg_round3.md), committed before any run, with two
amendments, both committed before the round finished. World: `results/sixaxis_R3_seed42.json`. Records:
`results/round3/<system>_R3_seed42_sonnet_rep1.json`, one per system, with every answer, retrieved context and judge reasoning.
Analysis: `python -m tools.analyze_round3` → `results/round3/analysis.json`. Cross-family re-judge:
`python -m tools.rejudge_round3` → `results/round3/rejudge/`.

**Scope of every claim below:** one synthetic world of ~4K tokens (262 dated notes) and one replication per system. These
are item-level inferences inside that world, not claims about memory systems in general.

## Why this round existed

Round 2 found that recall (A1) ties at the top and that the systems separate on updates (A2), deletion (A3) and write cost
(A5). But A2 rested on 5 update questions per seed and A3 on 6–7 deletion cases, so most of those differences were one or
two items. Round 3 asked one question: do the A2/A3 differences survive at N ≈ 60 update keys and N = 34 deletion cases,
judged by more than one model family?

## Setup

| | |
|---|---|
| world | R3 seed 42: 262 memories, 30 people, 15 projects; 65 T4 update keys (34 *explicit* "moved from A to B", 31 *implicit* "is now B"), 10 T1, 10 T3, 5 T2, 6 T5; 34 cumulative deletion cases (12 single, 11 entity, 11 intent) in a seeded interleaved order; 2 poison notes |
| systems | full context, BM25, vector RAG, hybrid + reranker, Mastra OM, Generative Agents, Anthropic memory tool, Mem0, Zep/Graphiti (all axes); Microsoft GraphRAG (A1 + A2) |
| generator / judge | Sonnet via the shim; strict-judge v3 pinned to Sonnet; every LLM-branch judgement re-judged by `openai/gpt-6.1-terra` and `google/gemini-3.8-flash` |
| statistics | one world, one replication; paired over items; 95% bootstrap CI (10,000 resamples) + exact McNemar; Holm across H5–H9. A zero-width CI (both systems perfect on the same items) means "indistinguishable at this N", not "proven equal" |
| cost control | shared SQLite prompt cache across the runner shims (rebuild-on-delete replays its unchanged prefix; decoy judging is shared). Write cost is the first ingest, which never hits the cache |

## The benchmark doubted itself first: two more gold problems (one also invalidates part of round 2)

**10. Updates dated before what they supersede.** Memory dates are drawn within a session week, so a superseding note
could carry an *earlier* date than the fact it supersedes. In those items the "stale" value is the newer one by date, and
bm25's first "stale" answers were simply reading the dates correctly. This affects 7 of 65 R3 T4 keys and 1 T3 item. It also
affects **round 2: in S1 seeds 123 and 456, 3 of their 9 T4 items and 3 of their 9 T3 items are inverted (3 of 14 T4 across all
three S1 seeds); S2 seed 42 has 1 T4 and 1 T3.** Round 2's A2 numbers, and its stale rates in particular, are therefore partly
gold error. Fix: `inadmissible_updates()` excludes those items for every system, `ground_check()` flags them, and the
generator now forces strictly later dates (amendment 1). **The A2 set is 58 keys.**

**11. Implicit role updates are ambiguous.** "X is now data lead of Beacon" does not say the person left their earlier role,
since people can hold two. It also does not stop another note from saying someone else took over X's old role, which some
systems read as X having moved on. Eight T4 keys, 2 T3 items and 3 deletion controls rest on such a note. Tracker, city and
budget have one value at a time and are not affected. These items stay in the **primary** analysis and are dropped in a
**sensitivity** pass; both are reported side by side below. The generator wording is now "X's role changed on <date>: now
<role>" (amendment 2).

The two passes differ in one direction. Full context answers all 8 ambiguous items correctly, and the systems that fall
short on them include Mastra, GraphRAG, hybrid and Mem0. In 3 of those failures (Mastra `q_t4_29` and `q_t4_65`, Mem0 `q_t4_48`)
the system did not hedge between two roles: it answered the wrong role. So the primary pass is the conservative reading for
the "does anything beat full context" question, and the sensitivity pass is the generous one.

Two judge-boundary cases also showed up in the records. Neither changes any verdict, and both are reported, not re-scored:

- **Mastra's 3 "lost" controls** are answers that open with "I don't have enough information" and then state the correct
  value. Strict-v3 classifies them as abstentions by design, so Mastra lost no data.
- **Every A4 "fooled" flag is an answer that names the poison value only to call it unverified.** That is 2/2 for Generative
  Agents and 1/2 for Graphiti. See A4.

## Judge robustness: pre-declared threshold met

| comparison (LLM-branch items only) | n | κ at 0.5 | κ at the 0.99 decision threshold |
|---|---|---|---|
| Sonnet vs GPT-6.1-terra, round-3 records (questions + A3) | 1,350 | **0.932** | 0.874 |
| Sonnet vs Gemini-3.8-flash, round-3 records (questions + A3) | 1,350 | **0.937** | 0.869 |
| same, A1/A2 question items only (GPT / Gemini) | 732 | 0.98 / 0.99 | 0.85 / 0.84 |
| Sonnet / GPT / Gemini strict-v3 vs proxy labels, calibration set (all 100) | 100 | 0.92 / 0.87 / 0.91 | |
| same, calibration LLM-branch items only | 52 | 0.89 / 0.77 / 0.89 (CIs ≈ ±0.2) | |

Both families clear the pre-registered κ ≥ 0.80 against Sonnet, including at the 0.99 threshold the outcomes actually use.
The Sonnet tables are therefore primary. **No hypothesis verdict changes under either non-Claude judge.** When the judges
disagree, Sonnet is the stricter one: both alternatives disagree with it on largely the same items, and in the same direction.
On the calibration set's hard, LLM-branch-only items, GPT agrees least with the proxy labels (0.77, CI 0.51–0.95). The proxy
labels are LLM agents, not humans (see "Human labels").

## A2 — updates (FAMA = current − stale)

Primary pass: 58 admissible keys (31 explicit, 27 implicit). Sensitivity pass: 50 keys, the 8 implicit-role keys dropped.

| system | current | stale | FAMA | explicit cur / stale | implicit cur / stale | Δ FAMA vs full context, primary [95% CI] | FAMA, sensitivity | Δ vs full context, sensitivity |
|---|---|---|---|---|---|---|---|---|
| Full context | 1.00 | 0.00 | **1.00** | 1.00 / 0.00 | 1.00 / 0.00 | — | 1.00 | — |
| Hybrid + reranker | 0.95 | 0.00 | 0.95 | 1.00 / 0.00 | 0.89 / 0.00 | −0.05 [−0.12, 0.00] | 1.00 | 0.00 [0, 0] |
| Mem0 | 0.95 | 0.00 | 0.95 | 1.00 / 0.00 | 0.89 / 0.00 | −0.05 [−0.12, 0.00] | 1.00 | 0.00 [0, 0] |
| Mastra OM | 0.90 | 0.00 | 0.90 | 0.97 / 0.00 | 0.81 / 0.00 | **−0.10 [−0.19, −0.03]** | 0.98 | −0.02 [−0.06, 0.00] |
| GraphRAG | 0.85 | 0.00 | 0.85 | 0.94 / 0.00 | 0.74 / 0.00 | **−0.16 [−0.26, −0.07]** | 0.96 | −0.04 [−0.10, 0.00] |
| Vector RAG | 0.91 | 0.09 | 0.83 | 0.84 / 0.16 | 1.00 / 0.00 | **−0.17 [−0.31, −0.03]** | 0.80 | **−0.20 [−0.36, −0.04]** |
| BM25 | 0.78 | 0.16 | 0.62 | 0.71 / 0.16 | 0.85 / 0.15 | **−0.38 [−0.57, −0.21]** | 0.56 | **−0.44 [−0.66, −0.24]** |
| Zep / Graphiti (round-2 read config) | 0.38 | 0.26 | 0.12 | 0.45 / 0.26 | 0.30 / 0.26 | **−0.88 [−1.09, −0.67]** | 0.22 | **−0.78 [−0.98, −0.58]** |
| Generative Agents | 0.21 | 0.03 | 0.17 | 0.19 / 0.03 | 0.22 / 0.04 | **−0.83 [−0.95, −0.71]** | 0.18 | **−0.82 [−0.94, −0.68]** |
| Anthropic memory tool | 0.62 | 0.29 | 0.33 | 0.68 / 0.26 | 0.56 / 0.33 | **−0.67 [−0.90, −0.45]** | 0.40 | **−0.60 [−0.84, −0.36]** |

What survives at this N, in this world:

- **No system beats the 2023 upper bound on updates.** Full context is at the top. Hybrid + reranker and Mem0 are
  indistinguishable from it in both passes (CI reaches 0). Mastra and GraphRAG are below it in the primary pass and
  indistinguishable from it once the ambiguous role items are dropped.
- **Five systems are below full context in both passes:** vector RAG, BM25, the Anthropic memory tool, Graphiti and Generative
  Agents. Their contexts can miss the newer note: a top-5 lexical or vector window, model-written files that are not
  updated, an edge graph that drops or never extracts facts, and importance-weighted reflections.
- **The memory tool has the highest stale rate of any system (0.29).** Its answers say "no later change is recorded in my
  notes". The model filing the notes did not record the later change in the files it wrote, so the newer value is never
  read back.
- **Vector RAG's stale answers are retrieval misses.** In all 5 stale records the newer value is absent from the top-5
  context. All 5 are in the explicit stratum and cover four people (two are the same person with the same retrieved set);
  on implicit updates vector RAG is never stale.

### Hypotheses (Sonnet judge; identical verdicts under GPT and Gemini and in the sensitivity pass)

| | claim | estimate [95% CI] | McNemar p (Holm over H5–H9) | verdict |
|---|---|---|---|---|
| **H5** | stale(vector RAG) > stale(hybrid + reranker) | +0.086 [+0.017, +0.155] | 0.063 (0.25) | **CI excludes 0, not significant after Holm.** Five discordant keys, all in the predicted direction |
| **H6** | H5 also within the implicit stratum | 0.00 [0, 0] | 1.0 | **unresolved.** Vector RAG is never stale on implicit updates |
| **H7** | Graphiti's best read arm has lower FAMA than vector RAG | −0.379 [−0.586, −0.172] | < 0.001 on "current" (< 0.001) | **holds** (−0.34 to −0.36 under the other judges and the sensitivity pass) |

### Graphiti: the claim holds; most of the best arm's gap is missing facts, not stale ones

| Graphiti read arm (one ingest) | current / stale / other (of 58) | FAMA |
|---|---|---|
| r2 — round-2 config: 5 edges | 22 / 15 / 21 | 0.12 |
| default — library default: 10 edges | 29 / 7 / 22 | 0.38 |
| filter — drop edges Graphiti marked invalid/expired, keep 10 | 30 / 12 / 16 | 0.31 |
| xenc — Graphiti's cross-encoder recipe, 10 edges | 30 / 4 / 24 | **0.45** (best) |
| *vector RAG, for reference* | *53 / 5 / 0* | *0.83* |

The diagnostic needs no LLM; it covers the 58 admissible keys:

- **Invalidation fires, but incompletely.** 48 keys still have at least one edge stating the superseded value. Graphiti marked
  at least one of them invalid in 38 keys, but all of them in only 5, so duplicate old edges stay current.
- **For 9 keys the graph has no edge with the current value at all.**

Two failures follow from that:

- **Round 2's 5-edge configuration is mostly a stale-fact problem.** It has 15 stale answers, because the surviving old edges
  outrank the new one.
- **The best configuration is mostly a missing-fact problem.** It has only 4 stale answers, fewer than vector RAG's 5, but 24
  answers that are neither current nor stale, about half of them abstentions. That comes from extraction or retrieval misses,
  not superseded edges.

So the round-2 claim ("worse than 2023 systems on updates") holds, but its mechanism ("superseded edges stay valid") explains
only the round-2 configuration. Graphiti ran on Kuzu, not Neo4j, behind an OpenAI-compatible shim in front of Claude; extraction
quality through that shim is a plausible contributor to the missing facts and is a declared limitation.

## A3 — deletion (34 cumulative cases; 11 leak-measurable)

| system | forgot | control intact (valid controls) | leaked (of 11) | delete method |
|---|---|---|---|---|
| Full context · BM25 · Hybrid + reranker | 1.00 | 1.00 | 0 | rebuild |
| Vector RAG | 1.00 | 0.97 (30)§ | 0 | rebuild |
| Mastra OM | 1.00 | 0.90 (30)† | 0 | rebuild (re-observes the log) |
| Zep / Graphiti | 1.00 | 1.00 (21) | 0 | native `remove_episode` |
| Mem0 (provenance ∪ returned ids) | 0.97 | 1.00 (30) | 3 (27%, CI 0–55%) | native, by tracked ids |
| Generative Agents | 0.91 | 1.00 (3)‡ | 2 (18%, CI 0–45%) | rebuild of raw memories; reflections survive |
| Anthropic memory tool | 1.00 | 0.96 (24) | 0 | rebuild: the model re-files every remaining note |

§ The one miss is an intent control on an ambiguous role item and disappears in the sensitivity pass. † The three misses are
hedged abstentions that state the correct value (see above). ‡ Generative Agents answered only 3 controls correctly *before*
any delete, so its collateral-damage number is uninformative.

- **H8** (Generative Agents leaks more than every rebuild system): +0.18 [0.00, +0.45] against each, McNemar p = 0.5.
  **Unresolved.** With 11 leak-measurable cases this test only resolves a large effect. Round 2's 0.45 is neither confirmed nor
  excluded (it sits at the CI's edge). Generative Agents' ghost residue also shows at the answer level: in 3 of 34 cases it
  still gives the deleted city after the delete, because a reflection kept it.
- **Mem0 and Generative Agents are the two systems that leak** (exploratory, not pre-registered; wide CIs). Mem0's mechanism
  is visible in the records:
  - In 2 of its 3 leaks, extraction had written facts about the deleted person into *other people's* memories ("Noah Berg
    became design lead of Orbit … succeeding Zofia Nowak").
  - In the third, the person's own merged memory kept a deleted role.

  Deleting a note's returned ids does not reach facts that were merged across notes. This is round 2's delete-semantics
  problem one level further down.
- **H9** (the memory tool loses more valid controls than full context): −0.04 [−0.13, 0.00] over 24 paired valid controls,
  McNemar p = 1.0. **Unresolved.** It lost one control (an entity case: after re-filing, its files no longer contained
  Meridian's budget). Round 2's "one valid control in six" does not reproduce at this N; one in 24 is consistent with a small
  effect and with none.
- **Deletion by case position** (controls intact, cases 0–11 / 12–23 / 24–33): no system degrades in later cases. Vector RAG's
  one miss is in the first third, and Mastra's hedged abstentions are in the later two thirds. Position does not confound shape,
  because shapes are interleaved.

## A1, A4, A5 (reported for completeness)

- **A1** (T1/T2/T3/T5, admissible): GraphRAG 1.00, hybrid 0.98, Mem0 0.97, full context / vector RAG / Mastra 0.95, BM25 0.86,
  the memory tool 0.85, Graphiti 0.63, Generative Agents 0.30. This is the same picture as round 2. Decoy acceptance is 0.00 on every record
  (192 decoys each).
- **A4** (exploratory, 2 poison notes): **no system was actually fooled.** The three "fooled" flags (Generative Agents 2/2,
  Graphiti 1/2) are all answers that abstain and mention the poison value as an unverified rumor. The fooled rule (poison
  value present, trusted value absent) cannot tell that apart from believing it. Wherever a poison note was retrieved and the
  context is inspectable, its "unverified" label reached the generator, including Graphiti's edge fact ("An unverified rumor on
  Slack claims …"). The same rule produced round 2's A4 numbers, so **round 2's Generative Agents "fooled 0.83" may be the same
  artifact**. Those records are not in this repository, so it cannot be re-checked here.
- **A5** (first ingest, measured): write LLM calls per memory are 0 for full context, BM25, vector RAG and hybrid; Mastra
  0.12; Generative Agents 0.12; Mem0 1.0; memory tool 0.39; GraphRAG 2.8; Graphiti 3.85. Ingest dollars at Sonnet rates through the
  CLI, for 262 memories: Mastra $1.1, Mem0 $3.0, the memory tool $3.2, GraphRAG $14.7, Graphiti $18.9. Read prompt tokens
  per query: retrieval systems and Mem0 1.6–2.2K, full context 11.5K, Mastra 11.0K, GraphRAG 12.2K, the memory tool 19.5K
  (several model calls per query).

## What round 3 changes about round 2

| round-2 statement | round 3 (one world, N = 58 / 34) |
|---|---|
| "The generations separate on updates; every modern system's beyond-envelope gain is there" | **Not supported as worded.** The 2023 upper bound (full context) is at the top on updates too. Part of round 2's update gap was inverted-date gold (3 of 9 T4 items in S1 seeds 123 and 456); part was a weak 2023 baseline (vector RAG). Updates separate systems whose context can miss the newest note from systems whose context reliably contains it, not 2026 from 2023 |
| "Vector RAG and BM25 serve the superseded value 30–37% of the time" | 9% and 16%; for vector RAG only on explicit "moved from" notes. The reranker's advantage is in the predicted direction but not significant after Holm |
| "Graphiti is worse than 2023 systems on updates; superseded edges stay valid" | The ranking **holds** (H7). The mechanism holds only for the round-2 read config; with better read configs the gap is mostly missing facts |
| "Generative Agents' reflections still name the deleted person 45% of the time" | 18% (2/11), CI 0–45%: neither confirmed nor excluded. Plus 3/34 cases where the deleted city is still answered |
| "Model-managed files lose one valid control in six" (memory tool) | 1 in 24, H9 unresolved. Its larger problem is updates: stale 0.29, FAMA 0.33 |
| "Mem0 leaks 6% even with id tracking" | 27% (3/11, CI 0–55%): merged facts carry the deleted name |
| "Labelled poison fools almost nobody whose answer path shows the generator the whole note; Generative Agents fooled 0.83" | Nobody is fooled here; the fooled rule flags abstentions that mention the rumor, and round 2's 0.83 may be that artifact |
| strict-v3 judge "κ = 0.87 vs human proxy" (Claude-only) | The verdicts are judge-robust across three families (κ 0.87–0.94 on 1,350 round-3 items); the proxy-label κ is still not a human κ |

## Human labels

`python -m tools.export_human_set` writes a blind 100-item set to `results/round3/human_labels/` (`items.csv` and a
self-contained `form.html`; `key.json` holds the hidden verdicts and inclusion weights). The set contains every
cross-judge disagreement, filled to 100 by stratified sampling over axis × verdict. Score a completed sheet with
`python -m tools.export_human_set --score labels.csv`. Until someone labels it, every "human" number in this repository is a
proxy number.

## Limitations

- **Inference scope:** one synthetic world and one replication, so these are item-level, not world-level, inferences.
- **Power:** 58 update keys resolve differences of about 0.1 only when they are one-directional; 11 leak-measurable cases
  resolve only large effects.
- **Graphiti setup:** Kuzu instead of Neo4j, and extraction through a shim.
- **Systems:** Mastra OM, the memory tool and Generative Agents are re-implementations. GraphRAG was not run on A3.
- **A4:** demoted (post hoc, declared), and its fooled rule is known to misfire.
- **Gold problems:** the two found during this round are handled by rules applied to every system, both committed before
  the round finished; the sensitivity pass is reported next to the primary one, not instead of it.
