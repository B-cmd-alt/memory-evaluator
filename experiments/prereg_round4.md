# Pre-registration — round 4: the scale ladder and a reference design (committed before any round-4 cell is run)

Date: 2026-11-18. Builds on rounds 1–3. Round-3 results motivated this round and are not pooled with it. A first draft of
this document was reviewed by an independent pass before commit; the changes that review forced are listed at the end.

## The question

Rounds 1–3 showed that in a ~10K-token store no memory system beats putting everything in the prompt, on recall or on
updates. That is the regime where a memory system is least needed. Round 4 asks:

1. **At what store size does a memory system start to pay**, and how does that depend on the model that reads the context?
2. **What is the least machinery that works there?** A reference design with no LLM call on the write path ("MinMem") is
   specified below, before any data, and tested against the systems that spend 1–4 LLM calls per memory at write time.

## What was seen before this document was committed

- All of rounds 1–3.
- **Development checks of MinMem's retrieval**, without any reader: on seed 42 at ~200K tokens with the *first* noise design,
  the newest note for each update key was in MinMem's context for 65/65 keys, with and without entity expansion. That noise
  design was then replaced (see "Changes forced by review"), so this number does not describe the worlds that will be run.
- **A 4-question infrastructure smoke test** (bm25, minmem, full context; an earlier ~57K-token rung; four readers, one of
  them since replaced). All 48 answers were correct. It was run to test the API path, token counts and provider caching.

No other round-4 answer has been generated.

## Fixed before data

- **Worlds.** `generate_ladder(seed, target_tokens)` in `src/data/sixaxis_generator.py`. A *core* world (the R3 configuration
  with both round-3 gold fixes) of 260 notes and 91–96 questions (10 T1, 5 T2, 10 T3, 60–65 T4, 6 T5), embedded in noise:
  - other people and projects with their own facts, updates and presentations, written with the same templates and value
    vocabularies over the same dates;
  - **name collisions**: noise people draw first names and surnames from pools that include the core's, so about a quarter
    of noise notes share a name token with a core person, without ever reusing a core first + last pair;
  - **mentions**: about 7% of noise notes name a core person or project in a non-subject position ("X had a 1:1 with Alice
    Chen on …") without stating any fact a core question asks about. Mentions per core entity grow with the rung (median 2,
    8, 32, 133 at 50K, 200K, 800K, 3.2M).

  No noise note is about a core entity, so the core questions and gold are identical on every rung; `ground_check()` must be
  clean. Seeds 42, 7, 2026.
- **Rungs** (store size in Claude tokens; these notes tokenise at ~2 characters per token): **10K** (core only), **50K**,
  **200K**, **800K**, **3.2M**. All four readers have a ~1M-token context window, so full context is run up to 800K and
  cannot be run at 3.2M.
- **Readers** (the model that answers from the context), via OpenRouter at temperature 0:
  `anthropic/claude-sonnet-5.5` (strong), `google/gemini-3.8-flash` (mid), `anthropic/claude-haiku-5.5` and
  `openai/gpt-6-luna` (small, two families). The shared answer prompt is unchanged from rounds 1–3.
- **Write-path model** for every system that calls an LLM at ingest: `anthropic/claude-haiku-5.5`. Rounds 1–3 used Sonnet for
  both; a small write model is what these systems ship with by default, and it is the only affordable choice at these sizes.
- **Judge.** strict-judge v3 with Sonnet 5.5 in the LLM branches. A random 20% of LLM-branch judgements are re-judged by
  `openai/gpt-6.1-terra`; κ ≥ 0.80 at the 0.99 decision threshold keeps the Sonnet tables primary, as in round 3. Two
  judge-free signals are recorded for every T1/T3/T4 answer (reference value present in the text; superseded value
  present), and the false-abstention rate is reported per reader, because strict-v3 scores any answer containing an
  abstention phrase as an abstention and readers may differ in how they hedge.
- **Embedder / reranker.** bge-small-en-v1.5 / bge-reranker-base, as before.
- **Generative Agents** runs with its recency refresh on access switched off, because each question is asked once per
  reader and a refresh would give later readers a different stream. This is a deviation from its published design.

### Systems and the rungs each runs

Zero-write systems and controls run seeds 42, 7 and 2026 on every rung (full-context arms up to 800K):

| arm | what it is |
|---|---|
| `full_context` | every note, oldest first |
| `full_context_hdr` | the same plus MinMem's one-line cue (control: is the cue the effect?) |
| `bm25`, `vector_rag`, `hybrid_rerank` | top-5, as in rounds 1–3 |
| `bm25_k20`, `vector_k20`, `hybrid_k20` | top-20 (control: is the context size the effect?) |
| `minmem_nohdr` | `hybrid_k20` + entity expansion |
| `minmem_order` | + oldest-first order |
| **`minmem`** | + the cue line |
| `minmem_noexp` | `minmem` without the expansion |

LLM-write systems, limited by sequential ingest time. A cell that does not finish is reported as not run, with the reason:

| system | 10K and 50K | 200K | 800K |
|---|---|---|---|
| Mem0 | seeds 42, 7, 2026 | seed 42 | seed 42, attempted |
| Microsoft GraphRAG (context builder) | seeds 42, 7, 2026 | seed 42 | — |
| Zep / Graphiti (round 3's best read recipe, `xenc`) | seeds 42, 7, 2026 | seed 42, attempted | — |
| Generative Agents (re-impl) | seeds 42, 7, 2026 | seed 42 | — |
| Mastra OM (re-impl; reflection threshold at Mastra's published 40K tokens) | seeds 42, 7, 2026 | — | — |
| Anthropic memory tool (re-impl; Sonnet and Haiku readers only) | seeds 42, 7, 2026 | — | — |

Mastra OM is not run above 50K: beyond its reflection threshold its behaviour is set by how the reflector compresses, which a
re-implementation cannot claim to reproduce.

## MinMem, specified before data (`src/systems/minmem.py`)

- **Write:** no LLM call. Index the raw notes with BM25, dense vectors, and an inverted index from capitalised spans found
  in the note text by a regular expression. The generator's gold entity and topic metadata is never read.
- **Read:** candidates = BM25 top-20 ∪ dense top-20 ∪ every note sharing a capitalised span with the question (capped at
  200 by BM25 score). Cross-encoder rerank, keep the top 20, present them oldest first, followed by one line:
  "(The notes above are in date order, oldest first. When two notes disagree, the later note is the current one.)"
- **Answer:** the shared answer prompt. MinMem only shapes the context.
- **Delete:** drop the notes and rebuild the indexes; nothing is derived.
- **No parameter is tuned after this commit.** MinMem was designed by this benchmark's author on these templates, so the
  synthetic ladder is not a held-out test of it. LongMemEval-S (H15) is. On the ladder, MinMem's capitalised-span index
  matches how the generator names its subjects; the name-collision and mention noise exists so that a span match is not an
  answer key at scale, and results are reported separately for person and project questions.

## Metrics

- **Item score:** T1, T3, T4, T5 score 1 when strict-v3 accepts the answer (score ≥ 0.99; T5 by the abstention rule), else 0;
  T2 is nugget recall.
- **Accuracy (primary):** the *task-macro* mean, i.e. the mean of the five per-task means, because T4 is two thirds of the
  items. The item-mean is reported next to it.
- **FAMA:** current − stale over T4 items.
- **Evidence in context:** whether every gold evidence note is in the retrieved context. Reported only for systems whose
  retrieved items are source notes. It separates retrieval misses from reader misses.
- **Cost:** write dollars per store and read dollars per query, both as billed (with provider prompt caching) and at
  uncached list price.

## Error handling

A failed answer (API error after retries, timeout, context overflow, spend cap) is recorded with no score. It is excluded
pairwise from every comparison and is never a zero. A cell where more than 2% of a reader's answers failed is PARTIAL; it is
re-run, or reported as not run. `finish_reason` is logged for every call, and truncated outputs are counted.

## Statistics

- Paired over items, with a **cluster bootstrap** (10,000 resamples): clusters are update keys within a seed (a T4 item and
  the T3 item about the same key are one cluster), resampled within seed. When readers are pooled, the same resampled items
  are used for every reader.
- Each hypothesis has one primary test with a one-sided bootstrap p-value. **Holm** correction applies across the six
  primary p-values at α = 0.05. Confidence intervals are reported unadjusted.
- A conjunction (H12) takes the largest p-value among its parts.
- Seed-level direction (how many of three seeds agree) is reported next to every primary test.

## Hypotheses (directional, falsifiable)

- **H10 — a crossover exists.** Primary: at the 800K rung with the Sonnet reader, accuracy(`minmem`) >
  accuracy(`full_context_hdr`), the matched-cue comparison. Falsified if the CI excludes 0 in the other direction;
  unresolved if it includes 0. Reported with it: the same against plain `full_context`, and `hybrid_k20` against both.
  Descriptive, uncorrected: per reader, the *crossover rung* = the smallest rung at which `minmem` beats `full_context_hdr`
  with a CI excluding 0, given that it does not at 10K; or "none up to 800K".
- **H11 — a weaker reader crosses over earlier.** Primary: at the 200K rung, (`minmem` − `full_context_hdr`) in accuracy is
  larger for Haiku than for Sonnet (difference in differences; CI excludes 0).
- **H12 — MinMem is not inferior to the LLM-write systems.** Primary: at the 50K rung, three seeds, Sonnet reader,
  accuracy(`minmem`) − accuracy(X) has a CI lower bound above −0.05 for **every** X in {Mem0, GraphRAG, Graphiti, memory
  tool, Mastra OM, Generative Agents}. Falsified for X if the CI lies entirely below −0.05. Secondary: the same at the other
  rungs X ran (single-seed where so labelled), for FAMA with a −0.10 margin, and with `hybrid_k20` in place of `minmem`
  (does a plain matched-k retriever already suffice?).
- **H13 — a fixed top-k window degrades with scale on updates.** Primary: FAMA(`vector_rag`) at 3.2M < FAMA(`vector_rag`)
  at 10K (Sonnet reader; CI excludes 0). Secondary: the same for `bm25`, `vector_k20` and `bm25_k20`; and `minmem`'s drop
  over the same range is smaller than `vector_k20`'s (difference in differences, matched k).
- **H14 — both parts of MinMem earn their place.** Primary: at 3.2M, pooled over readers, accuracy(`minmem_nohdr`) >
  accuracy(`hybrid_k20`): entity expansion alone, at matched k. Secondary: at 800K with the Haiku reader,
  FAMA(`minmem`) > FAMA(`minmem_nohdr`), split into the order step (`minmem_order` − `minmem_nohdr`) and the cue step
  (`minmem` − `minmem_order`).
- **H15 — held-out transfer (LongMemEval-S).** Setup: `xiaowu0162/longmemeval-cleaned`, the S split; 120 questions sampled
  proportionally by question type with `random.Random(0)`; each question's own haystack; memory unit = one round (a user
  turn and the assistant reply), dated by its session; the question is asked with its date; every answer is judged as a
  (question, answer, reference) triple by strict-v3 with Sonnet. Primary: with the Sonnet reader, accuracy(`minmem`) −
  accuracy(`hybrid_k20`) has a CI lower bound above −0.03. Reported without a directional prediction: `full_context` and
  `full_context_hdr` against `minmem` and `hybrid_k20` per reader, which tests the published claim that a lean retrieved
  context beats the full history at this size (arXiv 2606.09900: 83.6% vs 73.2%).

Exploratory, labelled as such: the cost–accuracy frontier per rung; where each LLM-write system stops being affordable;
retrieval-miss vs reader-miss decomposition; person vs project questions; entity-collision degree vs accuracy.

## Deviations from rounds 1–3, declared

- Model calls go through OpenRouter, not the `claude -p` CLI, including the judge.
- The write model is Haiku 5.5, not Sonnet.
- The reader set replaces the two-tier Sonnet/Haiku design.
- One replication per cell; world-level variation comes from three seeds.
- Each runner process has its own shim and ledger, so cost windows are exclusive to one cell at a time.
- Deletion (A3) and the adversarial-residue study, the Graphiti-on-Neo4j check, Letta, and the human-label pass are separate
  work packages, to be pre-registered in their own document before they run.

## Budget

Hard cap of $1,000 on OpenRouter for rounds 4–5 together, enforced in the shim (`SHIM_SPEND_CAP_USD`) from a shared ledger.
Full-context reads dominate the cost. Rule, decided by cost before any accuracy is seen: if the billed cost of the 200K
full-context cells projects the 800K full-context cells above $350, `full_context_hdr` at 800K runs with the Sonnet and
Haiku readers only. A cell that cannot run because of the cap is reported as not run.

## Changes forced by the pre-commit review

1. **Matched arms.** The draft compared MinMem (20 notes, a cue line) with baselines at 5 notes and a plain full context.
   Added `bm25_k20`, `vector_k20`, `hybrid_k20`, `minmem_order` and `full_context_hdr`; H10, H13, H14 and H15 now compare
   matched arms.
2. **Noise that collides.** The draft's noise people shared no name token with core people, so MinMem's span index was
   effectively the generator's subject key. Added name collisions and mention notes that grow with the rung.
3. **Errors can no longer score zero.** The draft scored a failed answer as wrong inside an "OK" cell, which at 800K would
   have counted API failures as evidence for H10.
4. **Three seeds for the LLM-write systems at 10K and 50K.** The draft tested H12 on one seed, the one MinMem was developed on.
5. **Generative Agents' recency refresh** would have made later readers see a different stream; it is switched off.
6. **`evidence_in_context`** is reported only for systems whose retrieved items are source notes.
7. **Cost** is reported both as billed and at list price; cost windows are exclusive per runner process.
8. Smaller: task-macro accuracy as the primary metric; cluster bootstrap by update key; Holm on one-sided p-values with
   conjunctions by largest p; per-reader false-abstention rates and judge-free signals; H15's sampling, unit and judge
   specified; a larger name pool, so no middle-initial names are needed.
