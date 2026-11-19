# Pre-registration — round 3: the A2/A3 closing round (committed before any round-3 run)

Date: 2026-11-17. Builds on `prereg_sixaxis.md` (round 2) and `docs/ROUND2_RESULTS.md`. Round-2 numbers motivated this round
(they are why A2/A3 are the focus) but are **not pooled** with round-3 numbers.

## Why a round 3

Round 2 found that recall (A1) ties at the top and that the systems differ on updates (A2), deletion (A3) and write cost
(A5). But A2 had 5 update questions per seed and A3 had 6–7 deletion cases per seed, so most A2/A3 differences are one or two
items. Round 3 asks only whether those differences hold at N ≥ 50 update keys and N ≥ 30 deletion cases.

## Fixed before data

- **World**: `results/sixaxis_R3_seed42.json` = `generate("R3", 42)` (`src/data/sixaxis_generator.py`), `ground_check()` clean.
  262 memories (~4K tokens), 30 people, 15 projects, 75% of keys updatable, at most one update per key.
  **65 T4 update keys** (each asked once; 34 *explicit* updates whose note names the old value, 31 *implicit* updates whose note
  states only the new value), 10 T1, 10 T3, 5 T2, 6 T5. **34 deletion cases** (12 single, 11 entity, 11 intent),
  shapes interleaved in a seeded order and applied cumulatively; intent cases delete the person's whole city history.
  2 poison notes (exploratory, see A4).
- **Generator**: Claude Sonnet via the shim (`claude -p`, temperature 0). **Sonnet tier only** — round 2 showed the Haiku tier
  floors A2 for every system, which removes the signal this round is about.
- **Judge**: strict-judge v3 pinned to Sonnet (primary). Every LLM-branch item is re-judged by two non-Claude families via
  OpenRouter at temperature 0: `openai/gpt-6.1-terra` and `google/gemini-3.8-flash`.
- **Embedder / reranker / answer prompt**: unchanged from round 2.
- **Shim prompt cache** (`SHIM_CACHE`): identical requests replay the stored response. This makes rebuild-on-delete affordable
  and keeps decoy judging identical across systems. Hits are logged (`cached=true`). Write cost (A5) is the first ingest,
  which never hits the cache.
- **Systems** (one replication each — the unit of analysis is the item, not the seed):
  full_context, bm25, vector_rag, hybrid_rerank, mastra_om, mem0, generative_agents, anthropic_memory_tool,
  zep_graphiti (all axes); ms_graphrag A1+A2 only (a full re-index per deletion case is unaffordable). a_mem is dropped
  (slow, and it tied vector RAG on every round-2 axis). Letta stays blocked.
- **Graphiti read arms**, all on one ingest, T4 only: `r2` (round-2 config: 5 edges, primary), `default` (library default,
  10 edges), `filter` (drop edges Graphiti marked invalid/expired, 10 edges), `xenc` (Graphiti's cross-encoder recipe,
  10 edges). Plus a no-LLM invalidation diagnostic: of the edges stating the superseded value, the share Graphiti marked
  invalid.

## Statistics

Item-level and paired: every system answers the same 65 keys and 34 cases. For a difference between two systems, the
estimate is the difference in proportions with a **95% paired bootstrap CI** (10,000 resamples of items, seed 0), plus an exact
McNemar p-value. A hypothesis holds when its CI excludes 0 in the predicted direction. It is falsified when the CI
excludes 0 in the other direction, and unresolved otherwise. **Holm correction** applies across H5–H9. With N ≈ 65, differences
under ~0.15–0.20 are expected to stay unresolved; that is reported as such, not as a tie. Results are also reported by deletion
case position, because later cases run on a more degraded state.

## Hypotheses (directional, falsifiable)

- **H5 (stale retrieval)** vector_rag serves the superseded value more often than hybrid_rerank (stale rate, paired over the
  65 T4 keys).
- **H6 (template artifact check)** H5's direction also holds within the *implicit* stratum alone (31 keys). If H5 holds but H6
  does not, the reranker's round-2 advantage is attributed to the explicit "moved from X to Y" wording, not to retrieval.
- **H7 (Graphiti updates)** Graphiti's *best* read arm has a lower FAMA (current − stale) than vector_rag.
  **Retract** the round-2 claim ("Graphiti is worse than 2023 systems on updates") if any arm reaches vector_rag (CI includes 0
  or favours Graphiti). If the invalidation diagnostic shows most stale edges *are* invalidated, the mechanism is reported as
  retrieval, not as "superseded edges stay valid".
- **H8 (ghost residue)** generative_agents leaks deleted strings in retrieved context more often than every rebuild-based
  system (full_context, bm25, vector_rag, hybrid_rerank, mastra_om), over the 11 leak-measurable cases (9 entity, 2 single;
  intent cases have no deleted string that is absent from every surviving memory).
- **H9 (collateral damage)** anthropic_memory_tool keeps fewer valid controls intact after a delete than full_context.

Reported without a test (pre-declared as under-powered): Mem0's leak rate with its CI; GraphRAG's A2 against hybrid_rerank.

## Judge robustness (pre-declared thresholds)

Cohen's κ (binary) between the Sonnet judge and each non-Claude judge, on LLM-branch items only. Deterministic branches agree
by construction and would inflate κ. **κ ≥ 0.80 with both families → "judge-robust"**, Sonnet tables are primary.
**0.60 ≤ κ < 0.80 → both tables are published** and each hypothesis verdict is reported under each judge. **κ < 0.60 → the
round's A1/A2 verdicts are reported as judge-dependent.** A verdict that flips under a non-Claude judge is reported as such.

## A4 (demoted, exploratory — decided post hoc)

Round 2's labelled poison barely discriminates: a strong generator discounts "Unverified rumor: …" by itself. An unlabelled,
later-dated contradiction is by construction an A2 update and has no ground truth. A4 is therefore demoted to an exploratory
appendix. Its one measurement is deterministic: does the "unverified" label survive into the context the generator sees?
This decision was made after seeing round-2 data and is labelled as such.

## Human labels

The two "proxy" labellers are LLM agents and are never called human labels. A 100-item set for a real human pass is
exported blind (`tools/export_human_set.py`). It contains every cross-judge disagreement, filled up by stratified sampling
over axis × verdict towards 50/50 correct/incorrect, with sampling weights recorded for re-weighting.

## Known limitations declared now

One synthetic world; one replication of LLM-write systems; Kuzu instead of Neo4j for Graphiti and an OpenAI-compatible shim
in front of Claude (Graphiti targets OpenAI structured output); Mastra OM, the Anthropic memory tool and Generative Agents are
re-implementations; leak-measurable cases are the subset whose deleted strings appear in no surviving memory.

## Amendment 1 (2026-11-17, after the first system's records (bm25), before any other system finished)

Reading bm25's "stale" answers showed that some of them are right by the dates. Memory dates are drawn within a session
week, so a superseding note could carry an *earlier* date than the fact it supersedes. In those items the "stale" value is
the newer one by date, and the gold is wrong. In the R3 world, 7 of 65 T4 keys (5 strictly earlier, 2 same day) and 1 of 10 T3
items are affected. Round 2 is affected too: in S1 seeds 123 and 456, 3 of 14 T4 and 3 of 15 T3 items; in S2 seed 42, 1 T4 and 1 T3.
Rule, applied identically to every system: an item whose supersession is not dated strictly later is **inadmissible** and
excluded from A1/A2 (`inadmissible_updates()`; `ground_check()` now flags it). The A2 analysis set is therefore **58 keys**
(stratum sizes reported with the results). The generator now forces superseding notes to be dated strictly later. The R3
world file stays the one that was run; regenerating it gives a clean world with the same counts.

## Amendment 2 (2026-11-17, after bm25 / vector_rag / mem0 records and their cross-family re-judge)

The two non-Claude judges disagreed with Sonnet on the same 11 items. Most were intent-case controls where the system
answered "X holds two roles: A and B". The cause is the implicit *role* template: "X is now <role>" does not entail that the
earlier role ended, because a person can hold two roles. The tracker, city and budget templates have no such ambiguity.
This affects 8 of the 58 admissible T4 keys, 2 T3 items and 3 deletion-case controls. These items are **not excluded**: the
primary analysis keeps them. A **sensitivity analysis** without them is reported next to every verdict, and a verdict
that differs between the two is flagged. The generator's implicit role wording is now "X's role changed on <date>: now <role>".

## Erratum to amendment 1 (2026-11-18)

The round-2 counts in amendment 1 read "3 of 14 T4 and 3 of 15 T3 items". The correct figures are 3 of the 9 T4 items and 3 of
the 9 T3 items in S1 seeds 123 and 456, which is 3 of 14 T4 across all three S1 seeds. The R3 counts and the rule are unchanged.
