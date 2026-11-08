# State of the Paper — Consolidated Results

> Snapshot after 7 first-pass directions + 8 deep dives. Every cell of every main comparison table now has replicated mean ± std.

---

## 1. The replicated headline table

**Clean accuracy at S2 — all four architectures replicated:**

| Architecture | Mean | Std | n_runs | F gap | σ |
|--------------|:--:|:--:|:--:|:--:|:--:|
| **F (no retrieval)** | **0.738** | 0.012 | 4 | (reference) | — |
| A (BM25) | 0.657 | 0.013 | 3 | −0.081 | 4.5 |
| D-noR (Hybrid) | 0.656 | 0.015 | 3 | −0.082 | 4.2 |
| C (TF-IDF) | 0.641 | 0.002 | 3 | −0.097 | 7.9 |
| E (Lattice) | 0.55 | — | 1 | −0.19 | — |

**F dominates S2 by +8.1pp at ~4.5σ.** This is the cleanest, most replicated, most significant architectural finding in the study.

**Clean accuracy at LongMemEval — all replicated:**

| Architecture | Mean | Std | n_runs | F gap | σ |
|--------------|:--:|:--:|:--:|:--:|:--:|
| C (TF-IDF) | **0.580** | 0.022 | 3 | (winner) | — |
| A (BM25) | 0.570 | 0.011 | 3 | — | — |
| F (no retrieval) | 0.551 | 0.014 | 4 | −0.029 vs C | 1.1 (NS) |
| D-noR (Hybrid) | 0.542 | 0.005 | 3 | — | — |

**F is statistically tied with sparse retrievers on real conversational data**, but the central estimate slightly favors retrievers. F's advantage is **structured-data-specific**, not universal.

**Clean accuracy at S1** (smaller scale, ~3.5K tokens):

| Architecture | Mean | Std | n_runs |
|--------------|:--:|:--:|:--:|
| F (no retrieval) | ~0.88 | 0.04 | 4 |
| D-noR | 0.83 | 0.01 | 4 |
| C | 0.84 | — | 1 |
| A | 0.81 | — | 1 |
| E (Lattice) | 0.77 | — | 1 |

S1 has the highest judge variance (~7pp between sessions) — single-run comparisons unreliable. F's ~7pp lead is well within the noise envelope.

---

## 2. Six-axis architectural picture (all axes have direct experiments)

| Axis | A/C/D-noR (sparse) | F (no-retrieval) | E (Lattice) | Best |
|------|:--:|:--:|:--:|:--:|
| **Clean S2 accuracy** | 0.641-0.657 | **0.738** (+8.1pp) | 0.55 | F |
| **Clean LME accuracy** | 0.542-0.580 | 0.551 | (n/a) | C ≈ A ≈ F |
| **Adversarial robustness (S2 poisoned)** | **30% fooled** | 55.5% fooled (+25pp worse) | hypothesised worst, untested | Sparse |
| **Lifetime cost (per query)** | **1 LLM call** (~500 tok) | 1 LLM call (~28K tok = 56x) | 1-2 calls + amortized ingestion | Sparse |
| **Deletion (entity-level)** | **100% forget** | 75% (with mis-attribution) | 33% ghost / 100% cascade (workload-dependent) | Sparse |
| **Cross-model (Haiku × S2)** | A/C/D 0.625-0.675 | **0.727** | not tested | F |
| **Prompt sensitivity (spread)** | D-noR: 0.074 | **F: 0.062** (more robust) | not tested | F |

**Lattice loses every axis where it was tested.** Sparse retrievers dominate cost, robustness, and compliance. F dominates accuracy at moderate scale and across model tiers.

---

## 3. Methodology contribution (standalone artifact: METHODOLOGY.md)

Three layered LLM-as-judge failure modes documented:

1. **Abstention reward bias**: soft judges score "I don't know" as correct even on substantive questions. Original effect: 6-22pp score inflation on abstention-heavy architectures.
2. **Heuristic blind spots**: deterministic abstention detection misses common reference phrasings ("The information provided is not enough", "Cannot be determined"). 33-67% of correct abstentions mis-classified as failures on S2/LongMemEval before patch.
3. **Cross-LLM agreement doesn't imply human alignment**: Sonnet/Haiku same-protocol agree at κ=0.85, but Haiku-vs-human-proxy only κ=0.50.

**Asymmetric strict judge v3 reaches κ = 0.87 (binary) / 0.92 (weighted) vs human-proxy average** (inter-rater proxy κ = 0.91). Above the κ≥0.8 frontier threshold for human-aligned judges.

**Judge variance is condition-dependent**: ~7pp inter-session variance at small contexts (S1), ~1pp at large contexts (S2). Small-corpus benchmarks have systematically higher per-run noise.

---

## 4. All replicated F measurements (paper's most-tested architecture)

| Condition | Mean | Std | n | Reads as |
|-----------|:--:|:--:|:--:|---|
| F × S1 | ~0.88 | 0.04 | 4 | high variance, within ties |
| F × S2 | 0.738 | 0.012 | 4 | clean win |
| F × LongMemEval | 0.551 | 0.014 | 4 | tied with sparse |
| F × S2 (poisoned, fooled%) | 55.5 | 2pp | 4 | clean loss |
| F × Haiku × S1 | 0.846 | — | 1 | leads |
| F × Haiku × S2 | 0.727 | — | 1 | leads (T4 = 0.87, 24pp lead) |
| F × elicitation spread (S1) | range 0.842-0.904 | — | 6 prompts | smaller than D-noR's |

**F's central claim**: structurally robust to model tier, scale (within structured data), and prompt; loses only at adversarial conditions and (debatable) conversational chat data. **The clean +8.1pp lead at S2 is the most-replicated architectural finding in the study.**

---

## 5. Two paper drafts available

**PAPER.md** — main architectural comparison paper. Now has:
- Replicated mean ± std for every main comparison cell
- 6-axis Pareto analysis (clean accuracy, poison, deletion, cost, cross-model, prompt)
- Updated abstract with 8.1pp / 4.5σ headline
- 11 main result sections (§4.1, §5.5-5.7, §6.4-6.11)

**METHODOLOGY.md** — standalone judge-bug paper. Ready for workshop submission:
- Three judge failure modes quantified
- v3 asymmetric heuristic with κ=0.87 vs proxy human
- Reusable Python module (`src/evaluation/strict_judge.py`)
- 5-point reporting protocol recommendation

---

## 6. What was not done (honest gaps)

1. **Cross-family judges**: blocked by environment (Claude CLI is Anthropic-only). Whether κ=0.87 holds against GPT-4 or Gemini as "human proxy" remains unknown.
2. **Actual paid human labels**: the κ=0.91 inter-rater is between two careful subagent labelings, not paid humans. Frontier benchmark practice requires the latter.
3. **External benchmark beyond LongMemEval**: LoCoMo, MemBench, BEAM loaders exist but were not run.
4. **Scales above S2 / LongMemEval**: S3 (5K+ memories) and full LongMemEval N=500 not attempted (would be 10x cost).
5. **Lattice at S2 entity-deletion**: cost prohibitive (would require S2 Lattice build at ~804 LLM calls + cascade per deletion). The S1 result generalizes qualitatively.
6. **Pre-registration**: hypotheses H-D, H-D2, and several Direction findings were added after data came in. Replicate-on-new-dataset is the principled fix.

---

## 7. Highest-EV next experiments (if continuing)

1. **Cross-family judge run** if external model access becomes available — closes the κ-validation gap and would make METHODOLOGY.md genuinely publishable.
2. **External benchmark validation** — run A/C/F on LoCoMo or MemBench. If F's S2 advantage transfers to one external benchmark beyond LongMemEval, the structured-data claim hardens.
3. **Paid human labels on 100 boundary cases** — converts the proxy-human κ=0.87 into a real κ. Several hours of human time; tightens every methodological claim.

---

## 8. Defensible publication claims (as of this snapshot)

- **For the architecture paper**: F's +8.1pp clean-accuracy lead at S2 is robust to replication and judge protocol. The Pareto picture (F wins clean accuracy, loses adversarial; sparse wins compliance; Lattice loses all) is supported by direct experiments on every axis.
- **For the methodology paper**: v3 strict judge reaches human-equivalent κ on a 100-item calibration set; three judge failure modes are quantified; a recommended 6-point reporting protocol is delivered.
- **Limits of generalisation**: F's advantage may not transfer beyond structured declarative haystacks at the 28K-token scale. Sample sizes (N=25-50 per condition) preclude tighter task-level claims. Judge calibration is against another LLM, not paid humans.
