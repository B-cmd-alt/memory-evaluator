# SixAxis — a self-built memory benchmark that takes the core of each existing one

> Status: design v1 (2026-11-08), pre-registered before any new system is run.
> Scope: the benchmark the Memory System Evaluator scores its 10 systems + 2 baselines on.
> Everything here is either inherited from a named benchmark (with the reason) or a fix for a
> documented flaw (with the source). Nothing is "our preference".

## 0. Why build our own instead of picking one

No existing benchmark gives all of: (a) ground truth we can audit, (b) a judge we can trust,
(c) forgetting and deletion as first-class, (d) adversarial memory, (e) controlled confounds
across systems, (f) a cost axis. Each has one or two. The evidence:

| benchmark | what it does best (we take) | documented weakness (we fix or avoid) |
|---|---|---|
| **LongMemEval** (ICLR 2025) | the five-ability taxonomy (extraction, multi-session, temporal, knowledge update, abstention); 7 question types; S (~115K tok) / M (~1.5M) scales | binary GPT-4o judge with no wrong-answer control; scores across papers use different judges so they do not compare; vendor numbers self-reported (Mem0 94.8% self-reported vs 32–49% independent) |
| **LoCoMo** (2024) | long multi-session conversational haystacks; the de facto marketing benchmark | two independent audits: **6.4% of gold answers wrong** (99/1,540), scoring ceiling 93.6%; the judge **accepts ~63% of intentionally wrong, on-topic answers**; 444/446 adversarial questions have no gold and are dropped from published evals |
| **MemoryAgentBench** (ICLR 2026) | **incremental multi-turn ingestion** (the system never sees the future); four competencies incl. **selective forgetting / conflict resolution** (FactConsolidation, EventQA); finding: every method fails multi-hop forgetting (<5%) | built from repurposed long-context datasets, so haystacks are documents not lived memory; forgetting measured only through QA |
| **BEAM** (2025) | **nugget-based scoring** for multi-fact answers; a scale ladder up to 10M tokens | scale beyond ~1M is a long-context test, not a memory test; cost prohibitive for a 10-system matrix |
| **ConvoMem** (Salesforce) | evidence-type labels (direct / indirect / temporal / negative / updated); **cost–accuracy crossover across conversation lengths** | QA-only; no adversarial or deletion |
| **Memora** (ACL 2026 Findings) | **FAMA — forgetting-aware accuracy** that penalises an answer built on an invalidated memory; finding: agents frequently reuse invalid memories | personalisation/recommendation tasks out of our scope |
| **MemOps** (2026) | tests memory **operations directly** (did the update land? did the delete propagate?) instead of inferring from QA | new; no system coverage yet |
| **ForgetEval** (June 2026) | **deletion as first-class**: 1,000 templated + 385 adversarial cases (identifier obfuscation, cross-lingual, intent-aware / compound-fact deletion); deterministic substring check; cost + latency of the delete path | deletion only |
| **MemDelta** (June 2026) | **controlled confounds**: fix the embedding model and the generator across systems, stratify by model family, report write-path cost, report full-context refusal rate | one memory system (Mem0), one benchmark |
| **this repo's v1 study** (2025) | six axes incl. poisoning + entity deletion; replication (n=3–4, mean±std); the **strict-judge v3** protocol (κ=0.87 vs independent labels) and the three judge failure modes; prompt-sensitivity > architecture-sensitivity | hypotheses partly post-hoc; human proxy not paid humans; Claude-only judge; N=25–50 |

## 1. Design principles (each one is a lesson with a source)

1. **Incremental, time-stamped ingestion** (MemoryAgentBench). Memories arrive as dated sessions
   in order; a question carries an `asked_at` timestamp; the system may only have ingested what
   came before. This is what makes temporal, update and forgetting questions well-defined. (The
   same as-of discipline the author's earlier work enforces for financial backtests.)
2. **Every gold answer is grounded and auditable** (LoCoMo's 6.4% lesson). Each question stores
   `evidence_ids` (the memory ids that entail the answer) and `stale_evidence_ids` (memories that
   *used to* entail a different answer). A deterministic grounding check regenerates the answer
   from the cited memories only and must agree; a 100-question human-verified subset is the
   audit anchor. A gold answer with no evidence pointer is not admissible.
3. **Judge with controls, not trust** (LoCoMo's 63% lesson + this repo's METHODOLOGY). Every
   question ships **two decoys**: a wrong-but-on-topic answer and a confident stale answer. The
   judge's decoy acceptance rate is a *benchmark health metric* reported next to every table;
   above 5% the table is marked unreliable. Scoring uses strict-judge v3 (asymmetric abstention
   heuristic, LLM only for the residue), with the deterministic-only and LLM-assisted scores
   both reported.
4. **Forgetting is scored, not just recall** (Memora FAMA + MemoryAgentBench). Update questions
   are scored three-way: *current* (correct), *stale* (the superseded answer — penalised
   explicitly, not merely "wrong"), *other*. The headline accuracy for the update axis is
   FAMA-style: correct minus stale.
5. **Deletion is a first-class task with adversarial cases** (ForgetEval). Three deletion
   shapes: single memory, entity-level ("forget everything about Compass"), intent-aware
   (compound facts, identifier variants). After deletion the system is probed by QA **and** its
   state is inspected (index entries, summaries, graph nodes) with a deterministic substring
   leak check — "ghost" residue in derived artifacts counts as failure (MemOps: test the
   operation, not only the answer).
6. **Adversarial memory** (this repo + 2026 poisoning papers). Low-trust memories that
   contradict a trusted fact, and memories that contain instructions ("ignore earlier sources").
   Metric: fooled rate, and whether the answer cites the poisoned source.
7. **Nugget scoring for multi-fact answers** (BEAM). T2 multi-session questions carry 2–5 atomic
   nuggets; score = nuggets recalled / nuggets, so partial aggregation is not binary.
8. **Controlled confounds** (MemDelta). One generator, one embedding model, one answer prompt
   for all systems, delivered through the OpenAI shim so no system can quietly use a different
   model. Results stratified by model tier (Sonnet, Haiku). Full-context baseline reports its
   refusal rate. Write-path cost is *measured* from the shim's call log, never estimated.
9. **Replication and a noise floor** (this repo's v1). n ≥ 3 runs per cell; mean ± std; the
   judge-noise and prompt-noise envelopes are measured once and printed on every ranking; a
   difference inside the envelope is reported as a tie.
10. **Cost crossover** (ConvoMem). Accuracy and cumulative cost plotted against number of
    queries per haystack, so "56× per query" and "ingest once, query many" are both visible.
11. **Pre-registration** (this repo's gap #6). This document and `experiments/prereg_sixaxis.md`
    are committed before the first run; any later hypothesis is labelled exploratory.
12. **A scale ladder, not a scale race** (BEAM/LongMemEval). S1 ≈ 90 memories / 3.5K tokens,
    S2 ≈ 750 / 28K, S3 ≈ LongMemEval-S / 115K. We stop where memory beats long context
    materially and where a 10-system matrix stays under budget.

## 2. The six axes and their metrics

| axis | tasks | metric(s) | inherited from |
|---|---|---|---|
| **A1 Recall** | T1 single fact · T2 multi-session (nuggets) · T3 temporal · T5 abstention | accuracy (strict v3); nugget recall; abstention precision/recall separately | LongMemEval, BEAM, ConvoMem |
| **A2 Update & forgetting** | T4 knowledge update; conflict resolution | three-way (current / stale / other); FAMA = current − stale | Memora, MemoryAgentBench |
| **A3 Deletion** | single / entity / intent-aware deletions, then probes + state inspection | forget rate; ghost-residue rate in derived artifacts; delete-path cost and latency | ForgetEval, MemOps, this repo |
| **A4 Adversarial** | contradicting low-trust memory; instruction-bearing memory | fooled rate; poisoned-source citation rate | this repo, 2026 poisoning work |
| **A5 Cost** | ingest and query, per scale | write-path LLM calls and tokens (measured); read-path tokens; $ per query; crossover point vs full context | MemDelta, ConvoMem, this repo |
| **A6 Invariance** | same questions under 2 model tiers, 3 answer prompts, 2 embedding models (RAG-family only) | rank stability (Kendall τ); spread per system | MemDelta, this repo's elicitation study |

Benchmark health, reported with every table: decoy acceptance rate of the judge; gold-grounding
check pass rate; judge-noise envelope; prompt-noise envelope.

## 3. Data

- **Synthetic structured haystacks** (own generator, extended): persona/company worlds with
  dated sessions; every fact has an id; updates and deletions are injected by construction so
  gold, stale and evidence pointers are exact. This is the only way to get auditable gold at
  scale; LoCoMo shows what happens without it.
- **External anchor**: LongMemEval-S, stratified 50-question sample mapped to T1–T5, judged with
  our protocol, to show the synthetic results transfer. LoCoMo is used only as a reference row
  with its known gold-error ceiling noted, never for ranking.
- **Deletion and poisoning sets** are generated from the same worlds, so a deleted or poisoned
  fact is one the recall set already tests — the before/after is paired.
- Sizes: per scale 50 questions (10 per task type), 20 update questions, 15 deletion cases
  (5 per shape), 20 poisoning cases, 2 decoys per question. Three scales. Two model tiers.

## 4. Protocol

1. Build the trusted bundle: generator and embedding fixed, prompts fixed, judge fixed, seeds
   fixed, shim log reset.
2. For each system: `reset → ingest sessions in order (measure write path) → for each question
   at its asked_at: query (measure read path) → judge with decoys`.
3. Deletion: ingest → probe → delete → probe + inspect state.
4. Poisoning: ingest clean → probe → inject → probe.
5. Three replications; results written as per-question records (question, evidence, retrieved
   context, answer, judge verdict + reasoning, decoy verdicts, tokens, latency) — the evaluator
   UI renders these records directly; aggregates are derived, never hand-entered.

## 5. What we deliberately do not do

10M-token haystacks (BEAM), web-agent environments (LongMemEval-V2), personalised
recommendation (Memora), paid human labels at scale (acknowledged; the 100-question audit subset
is the compromise), cross-family judges (blocked by access; listed as the top open item).

## 6. Pre-registered hypotheses (filled in `experiments/prereg_sixaxis.md`)

H1 Under strict v3, faithful re-implementations of 2023 systems are within the noise envelope of
2026 systems on A1 at S1–S2. H2 The 2026 systems' gains, where real, are on A2–A4 and A5, not A1.
H3 The gap between generations is smaller than the judge-protocol and prompt envelopes.
H4 Write-path cost separates the ten systems by more than an order of magnitude at S2.

## 7. As run in round 2 — amendments to the definitions above (2026-11-09)

Round 2 (`docs/ROUND2_RESULTS.md`) exercised the design for the first time at S1 and changed four definitions.
Each change was forced by a record whose number did not survive reading its evidence; the write-up lists all nine
corrections. The design text above is kept as the target; what was actually measured is this:

- **Sizes at S1 as run:** 22–24 questions (5 per task type, T5 4–5), 5 update questions inside them, 6–7 deletion cases
  (2–3 per shape), 1–3 poison cases per seed, 2 decoys per question; three seeds (42 / 123 / 456); two generator tiers.
  The §3 targets (50 / 20 / 15 / 20) are S2 numbers; S2 and S3 are not yet run.
- **A3 deletion** — cases are applied *cumulatively* on one system state, so every case is built against the running
  set of deleted ids (`ground_check` replays them). `forgot` = the strict judge no longer accepts the probe's pre-delete
  answer (`probe_answer`); abstention is the clean case. The leak check keeps the deterministic substring test but only
  over strings that occur in **no non-deleted memory**, and a string the probe question itself contains counts only when
  it comes back in retrieved context (an abstention naming the subject is not a leak); cases with no unique string are
  `leak_measurable = false`. `control_intact` is averaged over controls the system answered correctly *before* the delete.
  State inspection is through retrieved context only — index entries and graph nodes are not inspected directly yet.
- **A4 adversarial** — poison targets are disjoint from every deletion case (A4 runs after A3). The poison notes are
  *labelled* low-trust text ("unverified forwarded email / Slack rumor / chat"), so the axis measures labelled-poison
  robustness: whether the system, or the shared generator reading its context, lets a low-trust note override the
  trusted record. Instruction-bearing memories ("ignore earlier records") are included in the tool-kind template; an
  unlabelled contradiction is future work. `retrieved_poison` is reported next to `fooled`.
- **GraphRAG** runs as a context builder (local-search context → the shared answer prompt, zero GraphRAG-side LLM calls
  at query time); round 1 had let it answer with its own prompt.
- **Mem0 deletion** is reported under two documented semantics (provenance-only; provenance ∪ returned ids).
- **A6** at S1 covers the two generator tiers; the three-prompt and two-embedder arms were measured only as noise
  envelopes on the baselines (`tools/noise_envelopes.py`), not as a full re-run.

## Sources

LongMemEval (arXiv 2410.10813) · LoCoMo audits: Penfield Labs (dev.to), dial481/locomo-audit,
snap-research/locomo#44 · MemoryAgentBench (arXiv 2507.05257, ICLR 2026) · BEAM (github
mohammadtavakoli78/BEAM) · ConvoMem (HF Salesforce/ConvoMem) · Memora, "From Recall to
Forgetting" (arXiv 2604.20006) · MemOps (arXiv 2607.12893) · ForgetEval, "Control-Plane
Placement Shapes Forgetting" (arXiv 2606.15903) · MemDelta (arXiv 2606.29914) · this repo's
METHODOLOGY.md and RESULTS_SUMMARY.md.
