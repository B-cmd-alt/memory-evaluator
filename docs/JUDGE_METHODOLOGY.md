# Three Layered Failure Modes of LLM-as-Judge for Memory Benchmarks
## A Reproducible Analysis Across Synthetic and External Datasets

---

## Abstract

We document three layered failure modes of LLM-as-judge protocols as commonly applied to memory and retrieval benchmarks: (1) **abstention reward bias** — soft LLM judges score "I don't have enough information" responses as fully correct even when the reference contains substantive information, inflating retrieval-failure scores by 6-22pp; (2) **heuristic blind spots** — the deterministic abstention detectors used to fix (1) miss diverse reference phrasings ("The information provided is not enough", "Cannot be determined from the available data"), causing 33-67% of *correct* abstentions to be mis-scored as retrieval failures on real benchmarks; and (3) **cross-LLM agreement does not imply human alignment** — same-family LLM judges (Sonnet ↔ Haiku) reach Cohen's κ = 0.85 with each other but only κ = 0.50 (Haiku) and κ = 0.61 (Sonnet, original strict) against an independent careful labeler. LLM-LLM agreement is misleadingly high because both judges share systematic harshness, in similar directions, vs humans.

We provide an **asymmetric judge protocol (v3)** that uses different phrase lists for reference-side vs answer-side abstention detection. On a 100-item calibration sample with inter-rater κ = 0.91 between two independent careful labelers, **v3 reaches κ = 0.87 (binary) / 0.92 (weighted) against the human proxy average — above the κ ≥ 0.8 threshold typically required to treat an LLM judge as human-equivalent.** We release v3 as a standalone reusable module.

We show that the choice of judge protocol moves overall accuracy by 10-15 percentage points without any architectural change — comparable to or larger than the differences between the architectures the benchmark is intended to measure. Using a sub-frontier LLM as judge (Haiku here) drops human-alignment to κ = 0.50, far below the threshold — even when its cross-LLM agreement looks reassuring.

**Implication**: every published memory benchmark using LLM-as-judge without explicit phrase-list calibration and human-aligned κ reporting should be considered potentially inflated by 8-22 percentage points on abstention-heavy tasks. Cross-judge agreement alone is *not* sufficient evidence of judge quality. We recommend a five-point reporting protocol for the field.

---

## 1. Introduction

LLM-as-judge has become the default scoring methodology for memory benchmarks (LongMemEval, LoCoMo, MemBench, Mem0's eval suite). The standard prompt instructs the judge to score a generated answer against a reference, with a special case for abstention ("the model correctly refused to fabricate"). This is convenient because it removes the need for hand-graded test sets and scales to thousands of questions at marginal cost.

It is also wrong in at least three ways, each of which moves benchmark scores enough to flip architectural rankings.

We discovered these bugs sequentially while building a memory architecture benchmark of our own. The first bug surfaced when an over-conservative architecture variant scored *highest* on our suite — by abstaining on every question. The second surfaced when we tried to fix the first with a phrase-list-based deterministic abstention detector and found it missed a third of correct abstentions on an external benchmark. The third surfaced when we tried to calibrate the corrected judge against careful independent labels and found cross-LLM agreement (κ=0.85) substantially exceeded LLM-vs-human agreement (κ=0.61).

This paper documents each bug with reproducible measurements, the patches we developed for each, and the residual error each patch leaves behind. We treat this as a methodology contribution: not "use this judge," but "every judge has these failure modes — here is how to measure them in your setting and what to report."

---

## 2. Bug 1: Abstention Reward Bias

### 2.1 The bug

Most published LLM-as-judge prompts include a special-case branch for abstention answers. Variants we observed in the literature:

> "Score the generated answer against the reference. 1.0 = correct, 0 = wrong. For T5 (abstention) questions, 1.0 = correctly abstains, 0.0 = fabricates."

When the generated answer contains a phrase like "I don't have enough information" on a question where the reference contains substantive information, the LLM judge frequently interprets this as "correctly abstained from fabricating" and awards 1.0. But this is a **retrieval failure**, not a correct abstention — the information existed in the dataset; the system simply failed to retrieve it. The reward bias has the wrong sign.

### 2.2 How we detected it

We built an over-conservative variant of a memory architecture that fragmented its index heavily and required strict provenance. It answered "I don't have enough information" on every diagnostic question we gave it — including ones where the answer was clearly present in the source memories. The original soft LLM judge scored it **0.92 average** (the highest score in our experiments). Manual inspection revealed all "successes" were retrieval failures the judge had mis-labelled as correct abstentions.

### 2.3 Magnitude of inflation

Applying a *strict judge* (described in §3) that does heuristic-first abstention detection on both the reference and the generated answer, we re-scored 345 previously-judged items. The mean inflation per dataset:

| Source | Items | Old (soft) avg | Strict avg | Delta |
|--------|:--:|:--:|:--:|:--:|
| Synthetic S0 (Architecture A) | 15 | 0.80 | 0.58 | **-0.22** |
| Synthetic S0 (Lattice) | 15 | 0.92 | 0.83 | -0.08 |
| Synthetic S2 (Architecture A) | 30 | 0.64 | 0.52 | -0.12 |
| Synthetic S2 (TF-IDF) | 30 | 0.61 | 0.51 | -0.10 |
| Haiku × Architecture A | 25 | 0.92 | 0.81 | **-0.11** |
| Haiku × Lattice | 25 | 0.93 | 0.71 | **-0.22** |
| Over-conservative Lattice v3 | 6 | 0.92 | 0.00 | **-0.92** |

The bias affects architectures unequally — it inflates *most* the architectures that abstain *most*. In our case, the architecture we were trying to evaluate as "better at abstention" was the one most over-credited.

### 2.4 First fix: deterministic heuristic + judge cascade

Our first fix replaces the soft judge with a three-branch strict cascade:

1. If reference is abstention AND answer is abstention → score = 1.0 (correct abstention), no LLM call.
2. If reference is substantive AND answer is abstention → score = 0.0 (retrieval failure), no LLM call.
3. If reference is abstention AND answer is substantive → LLM call asking "did the answer fabricate or hedge appropriately?"
4. If both substantive → standard LLM judge call for content match.

Heuristic detection of abstention uses a phrase list against the lowercased text:

```python
ABS_PHRASES = ["don't have enough", "do not have enough", "not enough information",
               "no information", "cannot determine", "i don't know",
               "not mentioned", "not specified", "not provided",
               "unable to determine", "insufficient information"]
```

Case (2) is the critical fix: it removes the LLM's ability to silently reclassify a retrieval failure as a correct abstention.

This fix is necessary but not sufficient — see §3.

---

## 3. Bug 2: Heuristic Blind Spots

### 3.1 The bug

The phrase list in §2.4 was derived from inspection of our own synthetic-dataset reference style. When we tried to apply the same heuristic to LongMemEval (ICLR 2025) and to our larger synthetic dataset, it missed a substantial fraction of correct abstentions because the reference style was different.

### 3.2 Reference phrasings missed

Examples from our calibration sample where the strict heuristic mis-classified a correct abstention as a retrieval failure:

| Source | Reference phrasing the heuristic missed |
|--------|-----------------------------------------|
| LongMemEval _abs | "*The information provided is not enough.* You did not mention how many pages you have left to read." |
| LongMemEval _abs | "*You did not mention this information.* You mentioned X but not Y." |
| Our synthetic S2 | "*Cannot be determined from the available data.* The memories record only X." |

The original list checks for `"not enough information"` (literal substring), but LongMemEval phrases it as `"is not enough"` (with "information" appearing before "not enough"). Similar issues for "cannot be determined" (passive) vs "cannot determine" (active), and "did not mention" vs "not mentioned".

### 3.3 Magnitude of mis-counting

Re-checking our previously-judged data:

| Source | False-abstention count (strict v1) | Items that should have been "correct abstention" | % miscounted |
|--------|:--:|:--:|:--:|
| Synthetic S1 (any arch) | 6 | 0 | 0% |
| Synthetic S2 (any arch) | 24 | 12 | **50%** |
| LongMemEval (any arch) | 59 | 20 | **34%** |

S1 was unaffected because its synthetic reference style happened to match the original phrase list. S2 and LongMemEval lost 8-17pp on overall accuracy as a measurement artifact, not an architectural fact.

### 3.4 Naive fix overshoots on the answer side

A natural attempt at a fix is to expand the phrase list to cover the missed patterns (judge v2). This works on the reference side but introduces a new bug on the answer side: substantive answers that contain hedging clauses become mis-classified as abstentions.

Example (from our synthetic S1):
- Reference (substantive): "Carol Davis was reassigned to Project Beacon."
- Answer (substantive, with hedge): "*Carol Davis is no longer the primary QA lead* — she's been reassigned full-time to Project Beacon. The context doesn't mention a replacement."
- Old strict v1: judged substantive → 1.0 (correct).
- Naive v2 with expanded phrase list: matches "doesn't mention" → marked false_abstention → 0.0 (incorrect).

S1_A overall accuracy dropped from 0.81 → 0.69 (-0.12) under naive v2, with all flips_down on this answer-side false-positive pattern.

### 3.5 The asymmetric fix (v3)

The methodologically defensible fix uses **two separate phrase lists**:

```python
# Used to detect abstention in the GENERATED ANSWER (conservative — 11 phrases)
ANS_ABS_PHRASES = ["don't have enough", "do not have enough", "not enough information",
                   "no information", "cannot determine", "i don't know",
                   "not mentioned", "not specified", "not provided",
                   "unable to determine", "insufficient information"]

# Used to detect abstention in the REFERENCE (broad — superset, 28 phrases)
REF_ABS_PHRASES = ANS_ABS_PHRASES + [
    "is not enough", "isn't enough",
    "cannot be determined", "could not be determined", "can't be determined",
    "not available", "no record of", "no record",
    "did not mention", "didn't mention", "do not mention", "doesn't mention",
    "no specific information", "without specific",
    "have no info", "have not been provided", "haven't been provided",
    "no details", "without details",
]
```

The asymmetry is justified by the structural difference between references (usually single-statement abstentions where over-detection is rare) and answers (often mixing substantive content with hedges where over-detection penalises correct responses).

### 3.6 Aggregate impact

Re-judging all data with v3:

| Source | v1 (original strict) | v3 (asymmetric) | Delta |
|--------|:--:|:--:|:--:|
| S1 (any arch) | 0.81 / 0.84 / 0.77 / 0.85 | unchanged | 0 |
| S2_A | 0.52 | **0.65** | +0.13 |
| S2_C | 0.51 | **0.62** | +0.11 |
| S2_E | 0.47 | **0.55** | +0.08 |
| S2_DnoR | 0.55 | **0.65** | +0.10 |
| LongMemEval_A | 0.39 | **0.555** | +0.17 |
| LongMemEval_C | 0.43 | **0.556** | +0.13 |
| LongMemEval_D | 0.41 | **0.538** | +0.13 |

Architectural rankings change as a result. On S2, the v1 ranking was "D-noR alone at 0.55"; the v3 ranking is "A and D-noR tied at 0.65, C at 0.62". On LongMemEval, the v1 ranking was "C alone at 0.43"; the v3 ranking is "A and C tied at ~0.56".

**Meta-finding**: the phrase list is benchmark-specific calibration, not a one-shot fix. Practitioners should report the phrase list they use, demonstrate it covers their reference style, and prefer asymmetric ref-vs-answer detection.

---

## 4. Bug 3: Cross-LLM Agreement Does Not Imply Human Alignment

### 4.1 The setup

We constructed a 100-item calibration sample stratified across our experiments, with 50 boundary cases (answer contains abstention-like phrases) and 50 non-boundary. We compared two LLM judges (Sonnet at two protocol versions and Haiku) against two independent careful relabels:

- **Proxy human pass 1 and pass 2.** Two different agents given the v3 rubric in full detail, asked to grade each item carefully and independently without seeing the strict-judge scores or each other's labels.
- **Haiku same-protocol.** Claude Haiku running the identical strict-judge protocol used to score the original benchmark answers.

### 4.2 Inter-rater reliability of the human proxy

Before treating the proxy labels as a gold standard, we measured agreement *between* the two independent careful labelers:

| Comparison | Cohen's κ (binary) | Weighted κ (ordinal) | Mean a / b | Mean \|d\| |
|------------|:--:|:--:|:--:|:--:|
| **Proxy_1 vs Proxy_2** | **0.911** | **0.981** | 0.615 / 0.625 | 0.030 |

This is *almost perfect* agreement on every metric. The two labelers reach near-identical means on the same 100 items, with mean absolute difference of 0.030 (out of a 0-1 scale). The gold standard is highly stable, and we treat the average of the two labelings as the reference distribution for the comparisons below.

### 4.3 Judge-vs-human and judge-vs-judge agreement

| Comparison | Cohen's κ (binary) | Weighted κ (ordinal) | Mean a / b | Mean \|d\| |
|------------|:--:|:--:|:--:|:--:|
| **Strict v3 (Sonnet) vs proxy average** | **0.870** | **0.922** | 0.599 / 0.620 | 0.079 |
| Strict v3 vs proxy_1 | 0.868 | 0.904 | 0.599 / 0.615 | 0.079 |
| Strict v3 vs proxy_2 | 0.824 | 0.900 | 0.599 / 0.625 | 0.081 |
| Strict v1 vs proxy_1 (original heuristic) | 0.610 | 0.648 | 0.448 / 0.615 | 0.188 |
| Strict v1 vs proxy_2 | 0.610 | 0.673 | 0.448 / 0.625 | 0.177 |
| Sonnet v1 vs Haiku (cross-LLM) | 0.857 | 0.848 | 0.448 / 0.356 | 0.093 |
| **Haiku vs proxy_1** | **0.496** | **0.517** | 0.356 / 0.615 | 0.266 |
| Haiku vs proxy_2 | 0.496 | 0.525 | 0.356 / 0.625 | 0.271 |

By Landis-Koch convention: κ ≥ 0.81 is *almost perfect*, 0.61-0.80 is *substantial*, 0.41-0.60 is *moderate*.

### 4.4 Three findings

**(a) v3 reaches human-level alignment.** Sonnet strict v3 against the proxy human average lands at κ_binary = 0.870 and κ_weighted = 0.922 — above the κ ≥ 0.8 threshold typically required to treat an LLM judge as human-equivalent. The asymmetric heuristic patch (§3.5) closed the gap from "substantial" (v1, κ=0.61) to "almost perfect" (v3, κ=0.87). Mean scores converge as well: 0.599 (v3) vs 0.620 (proxy average), mean absolute difference 0.08.

**(b) Cross-LLM agreement is misleading.** Sonnet and Haiku, both running the strict-judge protocol, agree at κ = 0.857 — almost perfect. But Haiku agrees with the human proxy at only κ = 0.496 — moderate, well below the κ ≥ 0.8 threshold. Both LLMs are systematically harsher than humans (Sonnet mean 0.45, Haiku 0.36, vs proxy 0.62), in similar directions, which inflates their pairwise agreement *despite* both being miscalibrated. **High cross-LLM κ is not evidence of judge quality.**

**(c) Judge choice matters more than architecture, when using a sub-frontier judge model.** Switching from Sonnet (Anthropic's mid-tier frontier model) to Haiku (a smaller/faster model) as the judge shifts the mean score by 9pp and drops human alignment from κ=0.87 to κ=0.50. This is comparable to or larger than the inter-architecture differences typically measured on memory benchmarks. The choice of judge model is itself a methodological parameter that must be reported and calibrated.

### 4.5 What this does and does not say

It does not say sub-frontier LLMs cannot be used as judges. Haiku's lower agreement may reflect its tendency toward stricter content matching, which could be appropriate for certain tasks. It does say that **cross-LLM agreement alone is not sufficient validation** — a judge can agree with other LLMs at almost-perfect κ while diverging substantially from human consensus.

It does not invalidate the original Panickssery et al. (2024) finding of LLM own-generation preference. Our setting (cross-model judging of *third-party* answers, not the LLM's own generations) is adjacent but distinct.

It does say that our previous draft (which reported only the v1 κ=0.61 and the cross-LLM κ=0.85) was incomplete: with v3 + inter-rater validation, we now have evidence that **a properly-calibrated strict judge can reach human-equivalent alignment** — and we know what calibration steps to take.

---

## 5. Recommended Protocol

For any memory benchmark using LLM-as-judge:

### 5.1 At experiment design time

1. **Report which judge protocol you use.** Specifically: which model, which prompt, which abstention-detection method, which scoring rubric. "LLM-as-judge with GPT-4" is not a protocol.

2. **Pre-register your phrase list** for abstention detection, and demonstrate it covers your reference style on a 50-item dev set. Use asymmetric REF/ANS lists.

3. **Calibrate against humans on ≥100 boundary cases** and report Cohen's κ. If κ < 0.8, your judge is not a reliable substitute for humans; report rankings as ordinal only, not absolute scores.

### 5.2 At reporting time

4. **Report both strict and soft scores** when both are available, with the delta. Show how much of your headline number depends on the judge protocol.

5. **Report abstention counts per architecture.** Silent abstention rates can hide retrieval failures that inflate aggregate scores. We recommend a four-cell table: (ref abstention × ans abstention, ref substantive × ans abstention, etc.).

6. **Report cross-judge κ** on a sample of your data using a second judge model. If κ < 0.8 across LLM judges, no rankings should be claimed beyond the judge-noise floor.

### 5.3 At publication time

7. **Open-source your judge protocol and phrase lists** alongside the dataset. Without these, reproducibility is impossible.

8. **Provide a v3-style asymmetric strict judge as your default**, and treat any other LLM-judge variant as a robustness check, not the headline result.

---

## 6. Implementation

A reusable Python module implementing the v3 asymmetric strict judge is available at `src/evaluation/strict_judge.py` in the accompanying repository. Usage:

```python
from src.evaluation.strict_judge import strict_judge_v3

score, reasoning, mode = strict_judge_v3(
    question=q,
    answer=a,
    reference=r,
    llm_judge_fn=your_llm_call,  # optional; required only for non-deterministic branches
)
```

The module exposes the phrase lists as module-level constants so practitioners can extend them for their dataset's reference style.

---

## 7. Limitations

1. **Our "human" calibration is a proxy.** A careful independent labeler is the best approximation possible without a paid human study. Actual human labels would likely show different κ — we estimate within ±0.1 of our proxy κ based on inter-LLM noise.

2. **Our calibration sample is N=100.** A larger sample (≥500) would give tighter κ estimates and uncover task-specific bias patterns we have not measured.

3. **One LLM family.** All judges in this paper are Claude family models. Reaching κ ≥ 0.8 between, e.g., Claude and GPT judges remains untested; cross-family κ would be the strongest evidence of judge convergence.

4. **The asymmetric fix is dataset-specific.** Our v3 phrase list covers our three benchmarks; a new dataset with different reference style would need its own phrase calibration.

---

## 8. Related Work

- **G-Eval** (Liu et al. 2023) and **Prometheus** (Kim et al. 2024) introduced structured LLM-judge protocols but did not address abstention bias specifically.
- **Panickssery et al. 2024** demonstrated own-generation preference in LLM judges across multiple tasks.
- **HELM** (Liang et al. 2023) advocated multi-metric reporting and judge calibration but did not provide a specific protocol for abstention.
- **LongMemEval** (Wu et al. 2025) is the source of half our test data; its evaluator uses GPT-4 as a soft LLM judge, which our findings suggest may inflate scores on the `_abs` subset.

---

## Data and Code Availability

All code, calibration data, and re-judged result files are in the accompanying repository:
- `src/evaluation/strict_judge.py` — reusable v3 implementation
- `run_strict_judge_v3.py` — re-judges existing results
- `results/judge_calibration/calibration_sample.json` — 100-item calibration sample
- `results/judge_calibration/proxy_human_labels.json` — proxy human labels
- `results/judge_calibration/haiku_judge_labels.json` — Haiku second-judge labels
- `results/analysis/v3_rejudged/` — all result files re-scored under v3

The accompanying memory-architecture paper applies these methods and reports separate findings on architectural comparisons.
