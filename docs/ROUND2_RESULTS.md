# SixAxis round 2 — results (S1, three seeds, two generator tiers)

Pre-registration: `experiments/prereg_sixaxis.md` (committed before any run). Design: `BENCHMARK_DESIGN.md`.
Records: `results/sixaxis/<system>_S1_seed<seed>_<tier>_rep<n>.json` (every answer, every retrieved context, every
judge reasoning). Aggregation: `tools/aggregate_sixaxis.py` → `results/sixaxis/ROUND2_SUMMARY.md` + `summary.json`.
Site: `site/index.html` (round-2 tables + the update / deletion / adversarial case explorer).

## Setup

| | |
|---|---|
| worlds | SixAxis S1, seeds 42 / 123 / 456 (86–88 memories, 22–24 questions, 6–7 deletion cases, 2–4 poison cases each), `ground_check()` clean |
| systems | 11 of 12 in the registry (Letta blocked: no Python 3.13 release); GraphRAG runs as a **context builder** with the shared answer prompt this round |
| generator tiers | Sonnet (primary) and Haiku, selected by the shim's default model; third-party systems inherit the tier through the shim |
| judge | strict-judge v3, **pinned to Sonnet** by name regardless of tier; deterministic branches first |
| embedder / reranker | bge-small-en-v1.5 / bge-reranker-base for every system that embeds or reranks |
| replications | 3 (one per seed) × 2 tiers = 66 cells |
| noise envelope | max(replication std of the baselines, judge-protocol spread, answer-prompt spread) — `tools/noise_envelopes.py` measured judge 0.011 and prompt 0.022 on the baselines; replication std is the binding term |

## Corrections made during the round (the benchmark doubting itself first)

Each was caught by reading a record's evidence trail instead of its number, fixed at the definition level, and
re-applied uniformly to every record. None re-generated an answer.

1. **Deletion cases were not consistent under cumulative application.** The runner applies the deletion cases in order on one
   system state; the generator had built each case independently, reusing the first stable item / first person as the
   control, which an earlier case had already deleted (15 of 25 cases across the three seeds, 12 in S2). Symptom: the
   full-context system "failed" 61% of controls — *before* the delete too. Fix: cases are built against the running set
   of deleted ids; `ground_check()` replays them in order and now flags control-on-deleted, probe-already-deleted,
   probe-not-in-delete-set and dangling ids (5 and 12 problems on the old files). Worlds regenerated; memories and
   questions unchanged. The contaminated partial run was discarded.
2. **Controls only measure collateral damage if the system could answer them before the delete.** Generative Agents failed
   most controls before any delete (a recall failure). `control_intact` is averaged over controls the system answered
   correctly beforehand, with the count recorded (`tools/judge_control_before.py` filled earlier records).
3. **Leak strings had to be unique to the deleted memories, and an echoed question is not a leak.** A venue city shared by
   other notes ("Osaka" in "the context lists where she presented (Osaka, …)") was counted as a leak of a deleted
   home-city fact, and an abstention naming the person from the question ("I have no information about X") was counted
   as a leak of X. Leak strings are now filtered to strings that occur in no non-deleted memory (cases with nothing unique
   left are `leak_measurable = false`), a string the probe question itself contains counts only when it comes back in
   retrieved context, and `forgot` is the strict judge rejecting the probe's pre-delete answer rather than a substring
   test. `tools/rescore_deletions.py` re-scored every record from its stored artifacts.
4. **Mem0's delete semantics.** Mem0 stores extracted facts, not source notes; a fact merged into a memory created from an
   earlier note keeps that note's provenance, so delete-by-provenance missed it (`{deleted: 0}` while the fact was still
   retrieved; 67% leak, 61–69% forgot — kept as `results/sixaxis/variants/mem0_provenance_only/`). v2 deletes the union of
   provenance matches and every memory id Mem0 returned as ADD/UPDATE for the note, which is what an application tracking
   `add()` results would do; over-deletion of merged memories shows up in the controls.
5. **Lost records.** GraphRAG's in-process config loader changes the working directory; after an hour of indexing the
   runner's relative output path no longer resolved and all six GraphRAG records were lost (the crash also skipped
   Graphiti). Paths are absolute now and the adapter restores the cwd; the cells were re-run with `--skip-existing`.
6. **Poison probes targeted memories a deletion case had removed.** A4 runs after A3 on the same system state, and 10 of 16
   S1/S2 poisons targeted a trusted memory a deletion case deletes, so their "before" answer was already gone. Poisons now
   avoid every deleted memory (and may target person facts, not only project facts, to keep the probe count up — still
   only 1–3 per S1 seed, 6 per system per tier; declared thin); `ground_check` enforces it. Trusted memories, questions
   and deletion cases are byte-identical; A4 was re-run for every cell on a fresh ingest (`tools/rerun_axes.py`).
7. **Graphiti's post-delete queries errored.** The base-class delete rebuilds through a hook Graphiti's adapter never
   implemented, leaving a closed driver: every A3 control after the first delete and every A4 answer was an `<error>`
   (control intact 0.00, fooled 1.00 were artifacts). Graphiti now tracks the episode uuids each note created and
   deletes them natively (`remove_episode`); A3 and A4 re-run for its cells.
8. **GraphRAG's incremental ingest used an invalid 3.x flag**, so every poison ingest failed (`A4 = —`). Incremental
   ingest is now a full re-index over the input folder, consistent with its rebuild-on-delete; A4 re-run.
9. **Infrastructure, not science:** the cloud container is reclaimed within minutes of the session idling (two restarts,
   ~5 h lost; a live waiter now holds the session), three runners were OOM-killed when six GraphRAG/Graphiti cells
   overlapped (resumed), and the reranker's latency was measured under a load average of ~20 (re-measured quietly,
   `tools/remeasure_latency.py`; the under-load value stays in the record).

## Results (S1, mean ± std over seeds 42 / 123 / 456; Sonnet tier unless stated; full tables in `ROUND2_SUMMARY.md`)

Noise envelope 0.059 (Sonnet) / 0.064 (Haiku) = replication std of the baselines; judge-protocol spread 0.011 and
answer-prompt spread 0.022 are both smaller. Decoy acceptance of the judge: **0.00** on every one of the 66 cells
(1,100+ decoys judged). Differences inside the envelope are ties.

### A1 — recall

| | A1 | T2 nuggets | T5 abstain |
|---|---|---|---|
| Microsoft GraphRAG (context builder) | **1.00 ± 0.00** | 1.00 | 1.00 |
| Mem0 | 0.96 ± 0.03 | 1.00 | 0.92 |
| Mastra OM (re-impl) | 0.96 ± 0.03 | 0.99 | 1.00 |
| Full context | 0.95 ± 0.04 | 1.00 | 1.00 |
| Hybrid + reranker | 0.88 ± 0.02 | 0.63 | 1.00 |
| Vector RAG | 0.88 ± 0.04 | 0.68 | 1.00 |
| A-MEM | 0.88 ± 0.04 | 0.67 | 1.00 |
| Anthropic memory tool (re-impl) | 0.76 ± 0.19 | 0.89 | 1.00 |
| Zep / Graphiti | 0.69 ± 0.02 | 0.27 | 1.00 |
| BM25 | 0.60 ± 0.06 | 0.19 | 1.00 |
| Generative Agents (re-impl) | 0.41 ± 0.03 | 0.31 | 1.00 |

Four systems tie at the top inside the envelope: GraphRAG, Mem0, Mastra OM and **full context** (a 2023 entrant).
GraphRAG's 1.00 now comes through the shared answer prompt, so round 1's own-prompt confound is gone; it still reads
~11.8K prompt tokens per query, 2.4× full context. The retrieval-window systems (vector RAG, hybrid, A-MEM, BM25) lose
on **T2 multi-session questions** (nugget recall 0.19–0.68): the answer needs several memories and a top-5 window drops
some — the SixAxis world is deliberately harder here than round 1's (BM25 0.92 → 0.60). Graphiti's recall is bounded by
its edge-fact retrieval (T2 0.27). Abstention (T5) is a solved problem for every system under the shared prompt.

### A2 — update & forgetting (FAMA = current − stale)

| | current | stale | FAMA |
|---|---|---|---|
| GraphRAG | 1.00 | 0.00 | **1.00** |
| Hybrid + reranker | 0.92 | 0.00 | 0.92 |
| Mastra OM · Full context | 0.85 | 0.00 | 0.85 |
| A-MEM | 0.83 | 0.17 | 0.67 |
| Mem0 | 0.77 | 0.00 | 0.77 |
| Vector RAG | 0.70 | 0.30 | 0.40 |
| BM25 | 0.55 | 0.37 | 0.18 |
| Anthropic memory tool | 0.55 | 0.15 | 0.40 |
| Generative Agents | 0.15 | 0.15 | 0.00 |
| Zep / Graphiti | 0.22 | 0.37 | **−0.15** |

This is where the generations separate. Every modern system's largest beyond-envelope gain over vector RAG is on this
axis (H2), and the stale column is the diagnostic: vector RAG and BM25 serve the superseded value 30–37% of the time
because both versions of a fact are equally retrievable; the reranker fixes most of that (stale 0.00). Graphiti is the
one modern system that is *worse* than its 2023 counterparts on updates — superseded edges stay valid in its graph
(stale 0.37, negative FAMA), the same failure round 1 saw. The Haiku tier compresses all of A2 (GraphRAG 1.00 → 0.28
current): the weaker generator does not resolve "as of today" from a dated context even when the context is right.

### A3 — deletion (after the three definition fixes; `forgot` judged, leaks only on strings unique to the deleted memories)

| | forgot | control intact (valid controls) | leaked | delete method |
|---|---|---|---|---|
| Full context · BM25 · Vector RAG · Hybrid · Mastra OM · A-MEM | 1.00 | 1.00 | 0.00 | rebuild |
| Mem0 (provenance ∪ returned ids) | 1.00 | 1.00 | 0.06 | native, by tracked ids |
| GraphRAG | 1.00 | 0.94 | 0.00 | full re-index |
| Zep / Graphiti | 1.00 | 0.94 | 0.00 | native `remove_episode` |
| Anthropic memory tool | 1.00 | 0.83 | 0.00 | file edits by the model |
| Generative Agents | 1.00 | 0.67 | **0.45** | rebuild of raw memories; reflections survive |
| *Mem0, provenance-only delete (variant)* | *0.61* | *1.00* | *0.67* | *metadata match* |

Every system "forgets" at the answer level. The axis discriminates on the two other columns: **ghost residue** —
Generative Agents' reflections synthesised from the deleted memories still name the person in 45% of cases (the
ForgetEval "derived artifact" failure, observed in retrieved context); Mem0 leaks 6% even with id tracking because a
merged memory can still carry the deleted string; and **collateral damage** — the model-managed-files pattern loses one
valid control in six (it edits more than asked), Graphiti and GraphRAG one in sixteen. The Mem0 variant row is the
headline for extraction-based stores: *delete this note* is ill-defined once facts are merged across notes; by
provenance alone the note's facts survived two times in three.

### A4 — adversarial memory (labelled low-trust notes; 1–3 probes per seed)

| | fooled | retrieved the poison |
|---|---|---|
| GraphRAG · Mem0 · Mastra OM · Full context · Hybrid · Vector RAG · A-MEM · Graphiti | 0.00 | 0.00–1.00 |
| BM25 · Anthropic memory tool | 0.17 | 0.28 / 0.00 |
| Generative Agents | **0.83** | 0.83 |

With the poison written as what it is, the shared generator discounts it whenever it sees the whole note: vector RAG and
hybrid retrieve the poison 100% of the time and are fooled 0%. The fooled systems are those whose answer path hides the
note's framing from the generator — Generative Agents (importance-weighted reflections surface the claim without the
"unverified" wrapper). On the Haiku tier Graphiti is fooled 0.83 (edge facts carry the claim, not the caveat) and
Generative Agents 0.39. Declared limitation: this measures labelled-poison robustness; unlabelled contradictions are
future work, and the probe count is thin.

### A5 — cost (measured from the shim ledger)

| | write calls / memory | read prompt tok / query | p50 s |
|---|---|---|---|
| Full context · BM25 · Vector RAG · Hybrid | 0 | 1.6K–4.8K | 4.0–5.0 |
| Mastra OM | 0.11 | 4.5K | 5.2 |
| Generative Agents | 0.12 | 1.9K | 5.4 |
| Anthropic memory tool | 0.39 | 9.1K | 17.4 (3 calls/q) |
| A-MEM | 0.99 | 1.7K | 7.4 |
| Mem0 | 1.00 | 2.3K | 6.1 |
| GraphRAG | 3.10 | 11.8K | 4.2 |
| Zep / Graphiti | **4.26** | 1.7K | 4.3 |

H4: ×39 between the cheapest non-zero (Mastra OM) and the most expensive (Graphiti) write path, with four systems at
zero. The reranker's p50 is the quiet-machine re-measurement (4.0 s); its under-load value (~105 s) stays in the records.

### A6 — invariance across generator tiers (Kendall τ, Sonnet vs Haiku ranking)

A1 **0.78**, A2 FAMA **0.60**, A4 fooled **0.96**. Recall ranking is stable across tiers; the update axis is not — the
weaker generator flattens A2 for every system, so a ranking on updates is partly a ranking of how much a system's
context *forces* the right temporal reading (the reranker and Mastra's observational summaries hold up best: 0.62–0.63
FAMA on Haiku vs GraphRAG's 0.28).

## Hypotheses (pre-registered wording; verdicts at S1 only — H1 needs ≥ 2 of 3 scales)

- **H1** each 2023-era system within the envelope of the best 2026 system on A1. *Falsified if* a 2026 system beats
  every 2023 system by more than the envelope in ≥ 2 of 3 scales. **S1 evidence, both tiers: no 2026 system beats every
  2023 system beyond the envelope** — because full context (2023) sits at 0.95, tied with GraphRAG / Mem0 / Mastra. As
  literally worded H1 is wrong for BM25 and Generative Agents (0.60 / 0.41 are far outside the envelope of the best 2026
  system) and right for full context and vector RAG; the pre-registered falsification clause is not met. Verdict waits
  for S2 and S3; the honest S1 reading is *"the 2023 upper bound (full context) is not beaten; the 2023 lower bound is"*.
- **H2** where a 2026 system beats its 2023 counterpart beyond the envelope, it is on A2 / A3 / A4, not A1. **Holds on
  Sonnet**: the largest beyond-envelope gain is on A2 (FAMA) for A-MEM, hybrid, Mastra, Mem0 and GraphRAG; none on A1.
  **One exception on Haiku**: GraphRAG's largest gain over vector RAG is on A1 (its A2 collapses on the weaker tier).
- **H3** generation spread on A1 smaller than the judge-protocol spread and the prompt spread. **Falsified at S1**:
  0.17 (Sonnet) / 0.19 (Haiku) between the 2023 and 2026 means vs 0.011 (judge) and 0.022 (prompt). The round-1
  intuition behind H3 ("prompt-sensitivity > architecture-sensitivity") survives only for the *upper* half of the table;
  across the full lineage the architecture gap is an order of magnitude larger than the protocol noise.
- **H4** write-path cost separates systems by ≥ 10× between the cheapest and the most expensive runnable system.
  **Holds**: ×39 (0.11 → 4.26 calls per memory), four systems at zero.

Exploratory, labelled as such (not pre-registered): the stale-answer rate as the diagnostic behind A2; ghost residue in
Generative Agents' reflections; the Mem0 delete-semantics comparison; the A2 collapse on the weaker tier.

## Limitations declared

**A4 measures labelled-poison robustness.** The poison memories are written as what they are ("Note from an unverified
forwarded email…", "Unverified Slack rumor…", trust = low): the axis asks whether a system, or the shared generator reading
its context, lets a low-trust note override the trusted record. A strong generator discounts the label on its own, so most
systems score 0 fooled even when they retrieve the poison; the systems that are fooled are those whose answer path does not
show the generator the full note (Generative Agents' reflections, Graphiti's edge facts). An unlabelled contradiction
(recency vs. provenance with no textual cue) is a different, harder test and is future work. Probe count is thin: 1–3
poisons per S1 seed after the disjointness rule.

Human-proxy judge calibration (two independent labelers); Claude-family judge only; N = 22–24 questions per seed at S1;
Letta blocked in this environment; Mastra OM, the Anthropic memory tool and Generative Agents are faithful
re-implementations; S2 and S3 not yet run; GraphRAG as context builder is this round's definition (round 1 let it answer
with its own prompt); Mem0 deletion reported under two documented semantics.
