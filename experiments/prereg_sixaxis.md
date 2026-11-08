# Pre-registration — SixAxis round 2 (committed before any round-2 run)

Date: 2026-11-08. Design: `BENCHMARK_DESIGN.md`. Systems: `src/systems/registry.py` (12 entries).
Round 1 (feasibility, S1, `results/feasibility/`) decides only which systems are runnable and their
cost class; it is NOT used to form or adjust the hypotheses below.

## Fixed before data

- Generator: Claude Sonnet via the shim (`claude -p`, temperature 0); second tier: Haiku.
- Embedder: BAAI/bge-small-en-v1.5 for every system that embeds; reranker: bge-reranker-base.
- Answer prompt: `src/systems/common.py::GENERATION_PROMPT` for every system, verbatim.
- Judge: strict-judge v3 (`src/evaluation/strict_judge.py`), deterministic branches first, LLM residue
  via the shim; decoy acceptance reported per table; a table with decoy acceptance > 5% is flagged.
- Replications: 3 per cell; seeds 42, 123, 456. Report mean ± std. Noise envelopes (judge, prompt)
  measured once on the baselines and printed on every ranking; differences inside = tie.
- Scales: S1 (90 memories), S2 (753), S3 = LongMemEval-S stratified 50.
- Primary metric per axis as in BENCHMARK_DESIGN §2. No metric added after data.

## Hypotheses (directional, falsifiable)

- **H1** On A1 (recall) at S1–S2 under strict v3, each 2023-era system (vector_rag, bm25,
  generative_agents, full_context) is within the noise envelope of the best 2026 system.
  *Falsified if* any 2026 system beats every 2023 system by more than the envelope in ≥2 of 3 scales.
- **H2** Where a 2026 system beats its 2023 counterpart beyond the envelope, it is on A2 (update &
  forgetting), A3 (deletion) or A4 (adversarial), not A1. *Falsified if* the largest beyond-envelope
  gain for any lineage is on A1.
- **H3** The spread between generations on A1 is smaller than the judge-protocol spread (soft vs
  strict v3) and the prompt spread (3 answer prompts). *Falsified if* generation spread > both.
- **H4** Write-path cost (ingest LLM calls per memory, measured) separates the systems by ≥ 10× at
  S2 between the cheapest and the most expensive runnable system. *Falsified if* < 10×.

## Analysis plan

Per axis: table of mean ± std per system per scale per tier; Kendall τ between tiers (A6);
paired comparisons only against the full-context and bm25 baselines; no post-hoc subgroup claims
beyond the T1–T5 task split fixed here. Anything else is labelled exploratory.

## Known limitations declared now

Human-proxy labels (two careful independent labelers), not paid annotators; Claude-family judge
only; N = 50 questions per scale; Letta blocked in this environment; Mastra OM and Generative
Agents are faithful re-implementations, not vendor code.
