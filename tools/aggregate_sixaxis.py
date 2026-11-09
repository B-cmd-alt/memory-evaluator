#!/usr/bin/env python3
"""Aggregate SixAxis round-2 records (results/sixaxis/<system>_<scale>_seed<seed>_<tier>_rep<n>.json) into
per-system, per-tier mean ± std across replications, the baseline-paired comparisons and the tier
invariance (A6) the pre-registration fixes — `experiments/prereg_sixaxis.md` §Analysis plan.

Writes results/sixaxis/summary.json (machine) and results/sixaxis/ROUND2_SUMMARY.md (human). Nothing here
adds a metric: every number is a mean/std of a field run_sixaxis.py already recorded.

    python tools/aggregate_sixaxis.py [--dir results/sixaxis]
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path

AXES = {  # path into rec["summary"] → short label
    "A1": ("A1_recall", "all_non_update"), "A1_T1": ("A1_recall", "T1"), "A1_T2": ("A1_recall", "T2_nugget_recall"),
    "A1_T3": ("A1_recall", "T3"), "A1_T5": ("A1_recall", "T5_abstention"),
    "A2_current": ("A2_update", "current"), "A2_stale": ("A2_update", "stale"), "A2_FAMA": ("A2_update", "FAMA"),
    "A4_fooled": ("A4_adversarial", "fooled"), "A4_retrieved_poison": ("A4_adversarial", "retrieved_poison"),
    "A5_calls_per_memory": ("A5_cost", "ingest_llm_calls_per_memory"), "A5_ingest_usd": ("A5_cost", "ingest_cost_usd"),
    "A5_read_tok_per_q": ("A5_cost", "read_prompt_tokens_per_query"), "A5_read_calls_per_q": ("A5_cost", "read_calls_per_query"),
    "A5_p50_s": ("A5_cost", "latency_p50_s"), "judge_decoy_acceptance": ("judge_health", "decoy_acceptance"),
}
ERA_2023 = {"full_context", "bm25", "vector_rag", "generative_agents"}
NOISE_FILES = sorted(Path("results/sixaxis").glob("noise_envelopes*.json"))   # written by tools/noise_envelopes.py
BASELINES = ("full_context", "bm25")


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    return round(statistics.mean(xs), 3), (round(statistics.stdev(xs), 3) if len(xs) > 1 else 0.0), len(xs)


def a3(rec) -> dict:
    """A3 averaged over shapes (single / entity / intent) so one number per rep; shapes kept too."""
    d = rec["summary"].get("A3_deletion") or {}
    out = {}
    for k in ("forgot", "control_intact", "leaked"):
        vals = [v[k] for v in d.values() if v.get(k) is not None]
        out[k] = round(statistics.mean(vals), 3) if vals else None
    out["by_shape"] = d
    return out


def load(dirp: Path):
    recs = []
    for f in sorted(dirp.glob("*_S*_seed*_rep*.json")):
        r = json.loads(f.read_text())
        if r.get("status") != "OK" or "summary" not in r:
            recs.append({"system": r.get("system"), "tier": r.get("tier", "sonnet"), "scale": r.get("scale"), "seed": r.get("seed"),
                         "status": r.get("status"), "error": r.get("error"), "file": f.name})
            continue
        row = {"system": r["system"], "tier": r.get("tier", "sonnet"), "scale": r["scale"], "seed": r["seed"], "rep": r["rep"], "status": "OK",
               "file": f.name, "wall_seconds": r.get("wall_seconds")}
        for k, (a, b) in AXES.items():
            row[k] = (r["summary"].get(a) or {}).get(b)
        if r.get("latency_quiet"):   # tools/remeasure_latency.py: quiet-machine p50 for CPU-bound read paths
            row["A5_p50_s_under_load"] = row["A5_p50_s"]
            row["A5_p50_s"] = r["latency_quiet"]["p50_s"]
            row["latency_source"] = "quiet"
        row.update({f"A3_{k}": v for k, v in a3(r).items() if k != "by_shape"})
        row["A3_by_shape"] = a3(r)["by_shape"]
        recs.append(row)
    return recs


def kendall_tau(a: list[str], b: list[str]) -> float | None:
    """Rank correlation between two orderings of the same items (ties broken by order)."""
    common = [x for x in a if x in b]
    if len(common) < 3:
        return None
    pa, pb = {x: i for i, x in enumerate(a)}, {x: i for i, x in enumerate(b)}
    conc = disc = 0
    for x, y in combinations(common, 2):
        s = (pa[x] - pa[y]) * (pb[x] - pb[y])
        conc += s > 0
        disc += s < 0
    n = len(common)
    return round((conc - disc) / (n * (n - 1) / 2), 3)


def aggregate(recs: list[dict]) -> dict:
    ok = [r for r in recs if r["status"] == "OK"]
    groups = defaultdict(list)
    for r in ok:
        groups[(r["scale"], r["tier"], r["system"])].append(r)
    cells = {}
    metrics = list(AXES) + ["A3_forgot", "A3_control_intact", "A3_leaked"]
    for (scale, tier, system), rows in groups.items():
        c = {"scale": scale, "tier": tier, "system": system, "n_reps": len(rows), "seeds": sorted(r["seed"] for r in rows)}
        for m in metrics:
            mu, sd, n = mean_std([r.get(m) for r in rows])
            c[m] = {"mean": mu, "std": sd, "n": n}
        cells[f"{scale}|{tier}|{system}"] = c

    # paired comparisons against the baselines, same seed, same tier (prereg: only against full_context / bm25)
    paired = {}
    for (scale, tier, system), rows in groups.items():
        for base in BASELINES:
            if system == base:
                continue
            brows = {r["seed"]: r for r in groups.get((scale, tier, base), [])}
            diffs = [r["A1"] - brows[r["seed"]]["A1"] for r in rows if r["seed"] in brows and r.get("A1") is not None and brows[r["seed"]].get("A1") is not None]
            if diffs:
                mu, sd, n = mean_std(diffs)
                paired[f"{scale}|{tier}|{system}-vs-{base}"] = {"A1_diff_mean": mu, "A1_diff_std": sd, "n_pairs": n}

    # noise envelope = max of (replication std of the baselines, judge-protocol spread, answer-prompt spread);
    # the last two come from tools/noise_envelopes.py (prereg H3), measured once per tier on the baselines
    env = {}
    measured = {}
    for f in NOISE_FILES:
        if f.exists():
            n = json.loads(f.read_text())
            measured[(n.get("scale", "S1"), n.get("tier", "sonnet"))] = n
    for scale in sorted({r["scale"] for r in ok}):
        for tier in sorted({r["tier"] for r in ok if r["scale"] == scale}):
            sds = [cells[f"{scale}|{tier}|{b}"]["A1"]["std"] for b in BASELINES if f"{scale}|{tier}|{b}" in cells]
            sds = [s for s in sds if s is not None]
            rep = round(max(sds), 3) if sds else None
            n = measured.get((scale, tier)) or measured.get((scale, "sonnet"))
            judge = n["judge"]["envelope"] if n else None
            prompt = n["prompt"]["envelope"] if n else None
            parts = [v for v in (rep, judge, prompt) if v is not None]
            env[f"{scale}|{tier}"] = {"replication_std_baselines": rep, "judge_protocol_spread": judge, "answer_prompt_spread": prompt,
                                      "envelope": max(parts) if parts else None,
                                      "note": "envelope = max(replication std, judge spread, prompt spread); judge/prompt from tools/noise_envelopes.py"
                                              + ("" if n and n.get("tier") == tier else " (judge/prompt measured on the sonnet tier)" if n else " (judge/prompt not measured yet)")}

    # A6: ranking agreement between tiers per scale, on A1 and on the update axis
    a6 = {}
    for scale in sorted({r["scale"] for r in ok}):
        tiers = sorted({r["tier"] for r in ok if r["scale"] == scale})
        for t1, t2 in combinations(tiers, 2):
            for m in ("A1", "A2_FAMA", "A4_fooled"):
                r1 = sorted([c for c in cells.values() if c["scale"] == scale and c["tier"] == t1 and c[m]["mean"] is not None], key=lambda c: -c[m]["mean"])
                r2 = sorted([c for c in cells.values() if c["scale"] == scale and c["tier"] == t2 and c[m]["mean"] is not None], key=lambda c: -c[m]["mean"])
                a6[f"{scale}|{t1}-vs-{t2}|{m}"] = kendall_tau([c["system"] for c in r1], [c["system"] for c in r2])

    # H1 / H2 / H4 evidence at each scale+tier (verdicts need ≥2 scales for H1; reported as evidence, never adjusted)
    hyp = {}
    for key, e in env.items():
        scale, tier = key.split("|")
        cs = {c["system"]: c for c in cells.values() if c["scale"] == scale and c["tier"] == tier and c["A1"]["mean"] is not None}
        envelope = e["envelope"]
        if envelope is None or not cs:
            continue
        old = {s: c["A1"]["mean"] for s, c in cs.items() if s in ERA_2023}
        new = {s: c["A1"]["mean"] for s, c in cs.items() if s not in ERA_2023}
        h1 = {s: {"A1": v, "beats_every_2023_by_more_than_envelope": all(v - o > envelope for o in old.values())} for s, v in new.items()} if old else {}
        # H2: for each modern system, where is its largest beyond-envelope gain vs its nearest 2023 counterpart (vector_rag for RAG-likes, bm25 otherwise)?
        h2 = {}
        for s, c in cs.items():
            if s in ERA_2023:
                continue
            base = cs.get("vector_rag") or cs.get("bm25")
            if not base:
                continue
            gains = {"A1": (c["A1"]["mean"] or 0) - (base["A1"]["mean"] or 0),
                     "A2_FAMA": (c["A2_FAMA"]["mean"] or 0) - (base["A2_FAMA"]["mean"] or 0),
                     "A3_forgot": (c["A3_forgot"]["mean"] or 0) - (base["A3_forgot"]["mean"] or 0),
                     "A4_not_fooled": (base["A4_fooled"]["mean"] or 0) - (c["A4_fooled"]["mean"] or 0)}
            beyond = {k: round(v, 3) for k, v in gains.items() if v > envelope}
            h2[s] = {"gains": {k: round(v, 3) for k, v in gains.items()}, "beyond_envelope": beyond,
                     "largest_beyond_envelope_axis": (max(beyond, key=beyond.get) if beyond else None)}
        costs = {s: c["A5_calls_per_memory"]["mean"] for s, c in cs.items() if c["A5_calls_per_memory"]["mean"] is not None}
        nz = [v for v in costs.values() if v and v > 0]
        h4 = {"min_nonzero_calls_per_memory": min(nz) if nz else None, "max_calls_per_memory": max(costs.values()) if costs else None,
              "ratio_max_over_min_nonzero": round(max(costs.values()) / min(nz), 1) if nz else None,
              "zero_cost_systems": sorted(s for s, v in costs.items() if not v)}
        # H3: spread BETWEEN generations on A1 (mean of 2023-era systems vs mean of the rest) against the judge-protocol
        # and answer-prompt spreads measured on the baselines (tools/noise_envelopes.py)
        old_vals, new_vals = list(old.values()), list(new.values())
        gen_spread = round(abs(statistics.mean(new_vals) - statistics.mean(old_vals)), 3) if old_vals and new_vals else None
        h3 = {"generation_spread_A1": gen_spread, "mean_2023": round(statistics.mean(old_vals), 3) if old_vals else None,
              "mean_2026": round(statistics.mean(new_vals), 3) if new_vals else None,
              "judge_protocol_spread": e.get("judge_protocol_spread"), "answer_prompt_spread": e.get("answer_prompt_spread"),
              "holds": (gen_spread is not None and e.get("judge_protocol_spread") is not None and e.get("answer_prompt_spread") is not None
                        and gen_spread < e["judge_protocol_spread"] and gen_spread < e["answer_prompt_spread"]),
              "note": "2023-era = full_context, bm25, vector_rag, generative_agents (prereg H1 list); the mean is dragged by bm25 and generative_agents"}
        hyp[key] = {"envelope_used": envelope, "H1_evidence": h1, "H2_evidence": h2, "H3_evidence": h3, "H4_evidence": h4,
                    "note": "H1 verdict requires ≥2 of 3 scales (prereg); this block is per-scale evidence only"}
    return {"cells": cells, "paired_vs_baselines": paired, "noise_envelopes": env, "A6_tier_kendall_tau": a6, "hypotheses": hyp,
            "non_ok": [r for r in recs if r["status"] != "OK"], "n_records_ok": len(ok), "variants": variants(Path("results/sixaxis/variants"))}


def variants(vdir: Path) -> dict:
    """Documented adapter variants kept outside the main glob (e.g. Mem0 delete-by-provenance-only, round-2 v1):
    A3 numbers per variant and tier so the write-up can show the comparison."""
    out = {}
    if not vdir.exists():
        return out
    for d in sorted(p for p in vdir.iterdir() if p.is_dir()):
        rows = [r for r in load(d) if r["status"] == "OK"]
        for tier in sorted({r["tier"] for r in rows}):
            rs = [r for r in rows if r["tier"] == tier]
            out[f"{d.name}|{tier}"] = {"n_reps": len(rs), **{m: dict(zip(("mean", "std", "n"), mean_std([r.get(m) for r in rs])))
                                                            for m in ("A1", "A3_forgot", "A3_control_intact", "A3_leaked")}}
    return out


def fmt(c, m, pct=True):
    v = c[m]["mean"]
    if v is None:
        return "—"
    s = f"{v:.2f}" if pct else f"{v:g}"
    return s + (f" ± {c[m]['std']:.2f}" if c[m]["n"] > 1 and c[m]["std"] is not None else "")


def markdown(agg: dict) -> str:
    out = ["# SixAxis round 2 — summary (mean ± std across replications)\n",
           "Generated by `tools/aggregate_sixaxis.py` from `results/sixaxis/*.json`. Judge: strict v3 (pinned to Sonnet). "
           "Differences inside the envelope are ties (prereg). Latency: six runners shared a 4-core machine (load avg ~20); "
           "LLM-bound p50s are unaffected, CPU-bound read paths (the cross-encoder reranker) were re-measured on a quiet machine "
           "where the record carries `latency_quiet` (tools/remeasure_latency.py); the under-load value stays in the record.\n"]
    keys = sorted({(c["scale"], c["tier"]) for c in agg["cells"].values()})
    for scale, tier in keys:
        e = agg["noise_envelopes"].get(f"{scale}|{tier}", {})
        out.append(f"\n## {scale} · generator tier: {tier} · envelope = {e.get('envelope')} "
                   f"(replication {e.get('replication_std_baselines')} · judge {e.get('judge_protocol_spread')} · prompt {e.get('answer_prompt_spread')})\n")
        out.append("| system | reps | A1 recall | T2 nuggets | T5 abstain | A2 current | A2 stale | FAMA | A3 forgot | A3 ctrl ok | A3 leaked | A4 fooled | write calls/mem | read tok/q | p50 s | decoy acc |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        rows = sorted([c for c in agg["cells"].values() if c["scale"] == scale and c["tier"] == tier], key=lambda c: -(c["A1"]["mean"] or 0))
        for c in rows:
            out.append(f"| {c['system']} | {c['n_reps']} | {fmt(c,'A1')} | {fmt(c,'A1_T2')} | {fmt(c,'A1_T5')} | {fmt(c,'A2_current')} | {fmt(c,'A2_stale')} | "
                       f"{fmt(c,'A2_FAMA')} | {fmt(c,'A3_forgot')} | {fmt(c,'A3_control_intact')} | {fmt(c,'A3_leaked')} | {fmt(c,'A4_fooled')} | "
                       f"{fmt(c,'A5_calls_per_memory', False)} | {fmt(c,'A5_read_tok_per_q', False)} | {fmt(c,'A5_p50_s', False)} | {fmt(c,'judge_decoy_acceptance')} |")
        h = agg["hypotheses"].get(f"{scale}|{tier}")
        if h:
            beat = [s for s, v in h["H1_evidence"].items() if v["beats_every_2023_by_more_than_envelope"]]
            out.append(f"\n- **H1 evidence ({scale}, {tier})**: 2026 systems beating every 2023 system by more than the envelope: {beat or 'none'}.")
            ax = {s: v["largest_beyond_envelope_axis"] for s, v in h["H2_evidence"].items()}
            out.append(f"- **H2 evidence**: largest beyond-envelope gain per modern system (vs vector_rag): {ax}.")
            h3 = h["H3_evidence"]
            out.append(f"- **H3 evidence**: generation spread on A1 = {h3['generation_spread_A1']} (2023 mean {h3['mean_2023']} vs 2026 mean {h3['mean_2026']}) "
                       f"vs judge spread {h3['judge_protocol_spread']} and prompt spread {h3['answer_prompt_spread']} → {'holds' if h3['holds'] else 'does NOT hold'} at this scale.")
            h4 = h["H4_evidence"]
            out.append(f"- **H4 evidence**: write cost {h4['min_nonzero_calls_per_memory']} → {h4['max_calls_per_memory']} calls/memory "
                       f"(×{h4['ratio_max_over_min_nonzero']} between cheapest non-zero and most expensive; zero-cost: {h4['zero_cost_systems']}).")
    if agg["A6_tier_kendall_tau"]:
        out.append("\n## A6 — ranking agreement between tiers (Kendall τ)\n")
        for k, v in agg["A6_tier_kendall_tau"].items():
            out.append(f"- {k}: τ = {v}")
    if agg["paired_vs_baselines"]:
        out.append("\n## Paired A1 differences vs the baselines (same seed)\n")
        for k, v in sorted(agg["paired_vs_baselines"].items()):
            out.append(f"- {k}: {v['A1_diff_mean']:+.3f} ± {v['A1_diff_std']:.3f} (n={v['n_pairs']})")
    if agg.get("variants"):
        out.append("\n## Adapter variants (kept outside the main table)\n")
        for k, v in agg["variants"].items():
            out.append(f"- {k} (n={v['n_reps']}): A1 {v['A1']['mean']} · A3 forgot {v['A3_forgot']['mean']} · control intact {v['A3_control_intact']['mean']} · leaked {v['A3_leaked']['mean']}")
    if agg["non_ok"]:
        out.append("\n## Not OK\n")
        for r in agg["non_ok"]:
            out.append(f"- {r['file']}: {r['status']} — {r.get('error')}")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/sixaxis")
    a = ap.parse_args()
    d = Path(a.dir)
    agg = aggregate(load(d))
    (d / "summary.json").write_text(json.dumps(agg, indent=1))
    (d / "ROUND2_SUMMARY.md").write_text(markdown(agg))
    print(f"{agg['n_records_ok']} OK records → {d/'summary.json'}, {d/'ROUND2_SUMMARY.md'}; non-OK: {len(agg['non_ok'])}")


if __name__ == "__main__":
    main()
