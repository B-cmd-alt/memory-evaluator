# Pre-registration — round 5: deletion under attack, and Graphiti on its intended stack (committed before any round-5 run)

Date: 2026-11-18. Rounds 2–3 motivated this round and are not pooled with it. Round 4 (the scale ladder) is running while
this is written; no round-4 result is used here.

## Part A — deletion under attack

### Why

Rounds 2–3 measured deletion with one direct question per case and a substring check on the retrieved context. Only 11 of 34
round-3 cases were leak-measurable, and the two systems that leaked did so through *derived* text (Mem0's merged facts,
Generative Agents' reflections). A published study reports that deleting only the raw record leaves the information
recoverable from derived summaries in about 20% of cases, and that a full-pipeline purge removes it (arXiv 2606.10062).
Part A asks how much residue each system leaves when someone actually tries to get the information back, and whether a
purge fixes it.

### Fixed before data

- **Worlds.** `residue_world(seed)` in `src/data/residue.py`: the clean R3 world (262 notes, both round-3 gold fixes) for
  seeds 42, 7 and 2026, with its 34 cumulative deletion cases per seed (102 cases). Entity cases delete only notes whose
  text names the person. (Rounds 2–3 also deleted filler notes that were tagged with the person but never named them.)
- **Probes** (328 in total), built mechanically from the deleted notes:

| case | probe | the deleted string |
|---|---|---|
| entity ("forget everything about P") | direct: "Where is P based?" | the city |
| | event (up to 2): "Who presented <project> at <conf> <year> in <city> on <date>?" | P's full name |
| | role: "Who is the <P's current role>?" | P's full name |
| value (a single fact; "forget where N lives") | direct | the value |
| | paraphrase: "Which city does S live in?" etc. | the value |
| | reverse: "Who is based in <city>?" etc. | S's name |

- **One acceptance rule, with no judge:** a probe is *recovered* at a stage when the deleted string is in the answer. The
  rule is deliberately permissive: a hedged answer that states the deleted value is still a leak.
- **Protocol**, per arm and seed, on one cumulative state: every probe is asked **before** its case's delete; the delete is
  applied; every probe is asked **after**, then the control question. After the last case every probe is asked again
  (**end state**). Then the system is reset and re-ingests only the surviving notes (**full purge**), and every probe is asked
  a last time.
- **Residue (primary).** A probe has residue when it is recovered **before**, recovered at the **end state**, and **not**
  recovered after the **purge**. End state and purge hold exactly the same surviving notes, so the difference is only what
  the system derived and kept. This removes the permissive rule's false positives probe by probe: a surviving note that
  legitimately puts the string in an answer does so after the purge too. A case has residue when any of its probes does.
  Immediate residue (the **after** stage in place of the end state) is reported as secondary; it can be inflated when a
  later case deletes the surviving note that had legitimately supplied the string.
- **Failed calls** are excluded, never counted as "not recovered". An arm with more than 2% failed probe calls or failed
  deletes is PARTIAL and is re-run or reported as not run.
- **Reader:** Sonnet 5.5. **Write model:** Haiku 5.5. **Other settings as in round 4** (Generative Agents' recency refresh
  off; Graphiti read recipe `xenc`; Mastra reflection threshold 40K), set by the runner and recorded in every record.
- **Arms:**

| arm | delete method | role |
|---|---|---|
| `full_context`, `minmem` | drop the notes and rebuild; nothing is derived | control |
| `mastra_om` | re-observe the remaining notes: a purge on every case | control |
| `memory_tool_rebuild` | the model re-files every remaining note: a purge on every case | control |
| `mem0` | delete every memory id `add()` returned for the note | native |
| `generative_agents` | drop the raw memories in place; reflections stay in the stream | native |
| `zep_graphiti` | native `remove_episode` (Kuzu) | native |
| `memory_tool_edit` | the model is shown the notes to forget and edits its own files | native |

  GraphRAG is not run: a full re-index is its only delete path, which is the purge. A separate Mem0 "provenance-only" arm
  is not run: in the installed mem0ai 2.2.1 `add()` only ever creates memories, so provenance and returned ids select the same
  set (see `docs/ERRATA.md` for what this means for round 2's provenance-only figure).

### Statistics

Paired at **probe** level, over probes recovered before the delete in both arms of a comparison; bootstrap 95% CI with
**cases** as clusters, resampled within seed (10,000 resamples); one-sided bootstrap p-values; Holm across the four primary
tests of this document (H16–H19). With about 100 cases the smallest difference in case-level residue rates this design can
resolve is roughly 0.10; smaller true effects will read as unresolved. A sensitivity analysis without the "recovered before"
filter is reported next to each primary result.

### Hypotheses

- **H16 — attack finds more than the direct question.** Pooled over the four native arms, over cases whose direct probe was
  recovered before: case-level residue over all probes − residue over the direct probe alone has a CI lower bound above
  **0.03**. (The difference cannot be negative by construction, hence the preset margin; the control arms' same difference
  is reported as the floor.)
- **H17 — an extraction store keeps what a note-level delete cannot reach.** Probe-level residue(`mem0`) >
  residue(`minmem`). Secondary: the same for `generative_agents`, `zep_graphiti` and `memory_tool_edit`, each against
  `minmem`.
- **H18 — a model editing its own files forgets less than re-filing them.** Probe-level residue(`memory_tool_edit`) >
  residue(`memory_tool_rebuild`).

Reported without a test: each native arm's case-level residue rate with its CI next to the published ~20% raw-only
figure; residue by case shape and by probe kind; immediate vs end-state residue; collateral damage (valid controls lost);
delete cost and latency; and each control arm's residue, which the purge adjustment should hold at zero.

### Changes forced by the pre-commit review

1. **Mem0 adapter.** `get_all()` and `search()` silently returned 20 results; the adapter now passes `top_k`. The
   provenance-only arm was dropped as redundant in this library version.
2. **Generative Agents' delete** re-ingested its own reflections on every delete; it now drops raw memories in place.
3. **Pre-registered settings** are set by the runner itself, not only by the round-4 launcher.
4. **Probe templates.** Budget cases were being built with the role templates ("Who is the $300k?").
5. **Residue is purge-adjusted and uses one acceptance rule.** The draft mixed a judge for value probes with a string test
   for name probes, and counted any name match, so reverse probes had false positives in a third of city cases.
6. **H16 has a margin**, statistics pair at probe level, an MDE is stated, and a no-filter sensitivity is reported.
7. **Runner.** A smoke run can no longer land on a real record's path; failed calls and failed deletes are counted and can
   make an arm PARTIAL; the residue runner has its own shim ledger.
8. `mastra_om` and `memory_tool_rebuild` are labelled as controls, since their delete is a purge.

## Part B — Graphiti on its intended stack

### Why

Rounds 2–3 ran Graphiti on an embedded Kuzu database with Claude behind an OpenAI-compatible shim, and found it far below
vector RAG on updates, with facts missing from the graph. Both choices are declared limitations: Graphiti targets Neo4j and
OpenAI's schema-constrained output. Part B removes them.

### Fixed before data

- **Stack:** Neo4j 5 community (local), `graphiti-core` with `openai/gpt-6.1-terra` as its model and `openai/gpt-6-luna` as
  its small model, called natively through OpenRouter with JSON-schema output. The embedder (bge-small) and reranker (BGE)
  stay as in every other arm, so only the graph store and the write model change. Read recipe `xenc`, as in round 4.
- **Worlds:** the round-4 10K rung (core world), seeds 42, 7, 2026. Readers: Sonnet and Haiku.
- **Comparators:** round 4's `hybrid_k20` and round 4's Kuzu/Haiku `zep_graphiti` on the same worlds.

### Hypothesis

- **H19 — the update deficit survives the intended stack.** With the Sonnet reader, FAMA(Graphiti on Neo4j + GPT) <
  FAMA(`hybrid_k20`), CI excluding 0. **If the CI includes 0 or favours Graphiti, the round-2/3 statement "Graphiti is worse
  on updates" is retracted as a configuration artifact.** Reported with it: the invalidation diagnostic (share of superseded
  edges marked invalid; keys with no current-value edge) for both stacks.

## Part C — reported as run or not run, no hypothesis

- **Letta** in a Python 3.12 environment, on the 10K rung, if it can be installed and served here.
- **Human labels:** the blind 100-item set exported in round 3 (`results/round3/human_labels/`), scored when a human pass
  exists.

## Budget

Shares round 4's $1,000 hard cap and ledger.

## Amendment 1 (2026-11-18, before any round-5 record exists)

The OpenRouter account ran out of credit while round 4 was on its second rung (it held $70 in total; the $1,000 cap was a
ceiling on spending, not a balance). One round-5 arm (`minmem`, seed 42) had started and failed on refused calls; its partial
record was discarded unread. **Part A now runs through the `claude -p` subscription path**, which rounds 1–3 used: the same
shim, with `sonnet` as the reader and judge and `haiku` as the write model. Part A needs only Claude models, so it is
internally consistent on this path; its numbers are not pooled with any OpenRouter-path number. Part B (which needs an OpenAI
model) and the rest of round 4 (which needs four reader families) wait for credit.

## Amendment 2 (2026-11-18, after reading six seed-42 records; before any seed-7 or seed-2026 record is read)

**What was seen.** Reading every seed-42 probe that the pre-registered rule counts as residue (46 probes over `mem0`,
`memory_tool_edit` and `generative_agents`) showed that the purge adjustment does not remove all of the string rule's false
positives. In about a quarter of them the deleted string is in the end-state answer for a reason that has nothing to do with
the deleted notes, and is absent from the purge answer only because the wording differed. Typical case: "forget where Noah
Berg lives" (Lisbon); the end-state answer declines and adds that he *presented at a conference in Lisbon* (a surviving
note); the purge answer declines without saying so. The control arms cannot show this noise: their end-state and purge
prompts are identical, so the cached answer is identical and their floor is zero by construction.

**What is added.** A second residue measure, **disclosure-adjusted residue**, reported next to the pre-registered one:

- For every probe answer at the **after**, **end** and **purge** stages that contains the deleted string, a judge (Sonnet,
  through the same path) is shown the case's deleted notes, the question and the answer, and decides whether the answer
  *discloses a fact that the deleted notes state*. A mention that the deleted notes do not make (another event, another
  person's fact, a talk location rather than a home city) is not a disclosure. `tools/disclosure_judge.py`.
- disclosed(stage) = string in answer ∧ judge says disclosed. Disclosure-adjusted residue = recovered **before** (string
  rule, unchanged) ∧ disclosed(**end**) ∧ ¬disclosed(**purge**).
- The judge is checked against my hand labels of the 46 seed-42 probes (made before the judge was run) and the agreement is
  reported.

**How results are reported.** H16–H18 are tested as registered (string rule) and again on the disclosure-adjusted measure.
The registered verdict stays the registered verdict. The adjusted measure is declared here as the one the write-up's
descriptive rates rely on, and its results are also given for seeds 7 and 2026 alone, which were not seen when this rule was
written. Where the two measures give different verdicts, both are stated.
