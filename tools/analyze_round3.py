#!/usr/bin/env python3
"""Round-3 analysis (experiments/prereg_round3.md): per-system tables, H5-H9 with paired bootstrap CIs + exact McNemar +
Holm, Graphiti read arms + invalidation diagnostic, deletion by case position, A4 label survival, and every verdict
re-computed under each non-Claude judge (tools/rejudge_round3.py). Writes results/round3/ANALYSIS.md + analysis.json.

    python -m tools.analyze_round3
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from tools.kappa import kappa_binary

R3 = Path("results/round3")
REBUILD = ["full_context", "bm25", "vector_rag", "hybrid_rerank", "mastra_om"]
ORDER = ["full_context", "bm25", "vector_rag", "hybrid_rerank", "mastra_om", "generative_agents", "anthropic_memory_tool",
         "mem0", "zep_graphiti", "ms_graphrag"]
B = 10_000


def load():
    recs = {}
    for f in sorted(R3.glob("*_R3_seed42_sonnet_rep1.json")):
        r = json.loads(f.read_text())
        if r.get("status") == "OK":
            recs[r["system"]] = r
    rj = {f.stem: json.loads(f.read_text()) for f in sorted((R3 / "rejudge").glob("*.json"))} if (R3 / "rejudge").exists() else {}
    return recs, rj


# ── statistics ──────────────────────────────────────────────────────────────────────────────
def boot_diff(x: list[float], y: list[float], seed=0) -> tuple[float, float, float]:
    """mean(x) - mean(y), paired over items; 95% percentile bootstrap CI."""
    n = len(x)
    d = [a - b for a, b in zip(x, y)]
    est = sum(d) / n
    r = random.Random(seed)
    v = sorted(sum(d[r.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return round(est, 3), round(v[int(0.025 * B)], 3), round(v[int(0.975 * B) - 1], 3)


def mcnemar(x: list[int], y: list[int]) -> float:
    """exact two-sided McNemar p on paired binary outcomes."""
    b = sum(1 for a, c in zip(x, y) if a and not c)
    c = sum(1 for a, cc in zip(x, y) if cc and not a)
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return round(min(1.0, p), 4)


def verdict(ci_lo: float, ci_hi: float, predicted_sign: int) -> str:
    if predicted_sign > 0:
        return "holds" if ci_lo > 0 else ("falsified" if ci_hi < 0 else "unresolved")
    return "holds" if ci_hi < 0 else ("falsified" if ci_lo > 0 else "unresolved")


def holm(ps: dict[str, float]) -> dict[str, float]:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = round(run, 4)
    return out


# ── per-item vectors (optionally re-scored by another judge) ───────────────────────────────
def t4_rows(rec: dict, arm: str | None = None) -> list[dict]:
    return sorted(rec["read_arms"][arm] if arm else [q for q in rec["questions"] if q["task"] == "T4"], key=lambda q: q["id"])


def rescore_q(sysname: str, kind: str, q: dict, rj: dict | None) -> tuple[float, str]:
    """(score, outcome) under an alternative judge: LLM-branch items take that judge's score; deterministic ones keep theirs.
    T4 outcome as in run_sixaxis.score_question: current if accepted, else stale if the superseded value (and not the current
    one) is in the answer, else other."""
    s = q["score"]
    if rj is not None and q.get("judge_mode") in ("normal_judge", "ref_abstention_judge"):
        s = rj.get(f"{sysname}|{kind}|{q['id']}", {"score": s})["score"]
    if q.get("task", "T4") == "T4":
        ans = q["answer"].lower()
        stale = bool(q.get("stale_answer")) and q["stale_answer"].lower() in ans and q["reference"].lower() not in ans
        return s, "current" if s >= 0.99 else ("stale" if stale else "other")
    return s, "correct" if s >= 0.99 else "wrong"


_DATA = json.load(open("results/sixaxis_R3_seed42.json"))
from src.data.sixaxis_generator import inadmissible_updates  # noqa: E402
INADMISSIBLE = inadmissible_updates(_DATA)   # prereg_round3.md amendment 1: excluded from A1/A2 for every system
from src.data.sixaxis_generator import ambiguous_implicit_roles  # noqa: E402
AMBIG_Q, AMBIG_C = ambiguous_implicit_roles(_DATA)   # amendment 2: sensitivity analysis drops these
DROP_Q: set = set()
DROP_C: set = set()


def a2_vectors(rec: dict, rj=None, arm=None, stratum=None):
    rows = [q for q in t4_rows(rec, arm) if q["id"] not in INADMISSIBLE | DROP_Q]
    if arm:   # arm rows lack reference/stale_answer: take them from the main record
        base = {q["id"]: q for q in rec["questions"]}
        rows = [{**q, "reference": base[q["id"]]["reference"], "stale_answer": base[q["id"]]["stale_answer"], "task": "T4"} for q in rows]
    if stratum:
        rows = [q for q in rows if q.get("stratum") == stratum]
    kind = f"arm:{arm}" if arm else "q"
    out = [rescore_q(rec["system"], kind, q, rj)[1] for q in rows]
    return [q["id"] for q in rows], [int(o == "current") for o in out], [int(o == "stale") for o in out]


def a3_vectors(rec: dict, rj=None):
    rows = sorted([d for d in rec.get("deletions") or [] if d["id"] not in DROP_C], key=lambda d: d["id"])
    def jb(d, field, key):
        if rj is None:
            return d[field]
        k = f"{rec['system']}|{key}|{d['id']}"
        if k not in rj:
            return d[field]
        return rj[k]["score"] >= 0.99 if key.startswith("ctrl") else rj[k]["score"] < 0.99
    return [{"id": d["id"], "shape": d["shape"], "position": d.get("position"), "forgot": jb(d, "forgot", "forgot"),
             "control_intact": jb(d, "control_intact", "ctrl_after"), "control_before_ok": jb(d, "control_before_ok", "ctrl_before"),
             "leak": bool(d["leaked_strings"]), "leak_measurable": d.get("leak_measurable", True)} for d in rows]


def mean(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 3) if xs else None


# ── hypotheses ──────────────────────────────────────────────────────────────────────────────
def hypotheses(recs: dict, rj=None) -> dict:
    H, ps = {}, {}
    if {"vector_rag", "hybrid_rerank"} <= recs.keys():
        for h, st in (("H5", None), ("H6", "implicit")):
            _, _, sv = a2_vectors(recs["vector_rag"], rj, stratum=st)
            _, _, sh = a2_vectors(recs["hybrid_rerank"], rj, stratum=st)
            e, lo, hi = boot_diff(sv, sh)
            H[h] = {"claim": f"stale(vector_rag) > stale(hybrid_rerank){' [implicit stratum]' if st else ''}", "n": len(sv),
                    "stale_vector_rag": mean(sv), "stale_hybrid_rerank": mean(sh), "diff": e, "ci95": [lo, hi], "verdict": verdict(lo, hi, +1)}
            ps[h] = mcnemar(sv, sh)
    if {"vector_rag", "zep_graphiti"} <= recs.keys():
        g = recs["zep_graphiti"]
        _, cv, sv = a2_vectors(recs["vector_rag"], rj)
        fv = [c - s for c, s in zip(cv, sv)]
        arms = {"r2": None, **{a: a for a in (g.get("read_arms") or {})}}
        fam = {}
        for name, arm in arms.items():
            _, c, s = a2_vectors(g, rj, arm=arm)
            fam[name] = [x - y for x, y in zip(c, s)]
        best = max(fam, key=lambda k: sum(fam[k]))
        e, lo, hi = boot_diff(fam[best], fv)
        H["H7"] = {"claim": "FAMA(graphiti best arm) < FAMA(vector_rag)", "n": len(fv), "best_arm": best,
                   "FAMA_by_arm": {k: mean(v) for k, v in fam.items()}, "FAMA_vector_rag": mean(fv), "diff": e, "ci95": [lo, hi],
                   "verdict": verdict(lo, hi, -1)}
        # McNemar needs binary: compare "current" indicators of best arm vs vector_rag
        _, cb, _ = a2_vectors(g, rj, arm=arms[best])
        ps["H7"] = mcnemar(cb, cv)
    if "generative_agents" in recs and any(s in recs for s in REBUILD):
        ga = [d for d in a3_vectors(recs["generative_agents"], rj) if d["leak_measurable"]]
        comps, worst_p = {}, 0.0
        for s in REBUILD:
            if s not in recs:
                continue
            o = {d["id"]: d for d in a3_vectors(recs[s], rj)}
            x = [int(d["leak"]) for d in ga]; y = [int(o[d["id"]]["leak"]) for d in ga]
            e, lo, hi = boot_diff(x, y)
            comps[s] = {"leak_other": mean(y), "diff": e, "ci95": [lo, hi], "verdict": verdict(lo, hi, +1)}
            worst_p = max(worst_p, mcnemar(x, y))
        vs = [c["verdict"] for c in comps.values()]
        H["H8"] = {"claim": "leak(generative_agents) > leak(every rebuild system)", "n": len(ga), "leak_generative_agents": mean(int(d["leak"]) for d in ga),
                   "vs": comps, "verdict": "holds" if all(v == "holds" for v in vs) else ("falsified" if any(v == "falsified" for v in vs) else "unresolved")}
        ps["H8"] = worst_p   # intersection-union test: the claim needs every comparison
    if {"anthropic_memory_tool", "full_context"} <= recs.keys():
        m = {d["id"]: d for d in a3_vectors(recs["anthropic_memory_tool"], rj)}
        f = {d["id"]: d for d in a3_vectors(recs["full_context"], rj)}
        ids = [i for i in m if i in f and m[i]["control_before_ok"] and f[i]["control_before_ok"]]
        x = [int(m[i]["control_intact"]) for i in ids]; y = [int(f[i]["control_intact"]) for i in ids]
        e, lo, hi = boot_diff(x, y)
        H["H9"] = {"claim": "control_intact(memory tool) < control_intact(full_context)", "n": len(ids), "memory_tool": mean(x), "full_context": mean(y),
                   "diff": e, "ci95": [lo, hi], "verdict": verdict(lo, hi, -1)}
        ps["H9"] = mcnemar(x, y)
    adj = holm(ps)
    for h in H:
        H[h]["p_mcnemar"] = ps.get(h); H[h]["p_holm"] = adj.get(h)
        if H[h]["verdict"] == "holds" and adj.get(h, 0) >= 0.05:
            H[h]["verdict"] = "holds (CI) but not after Holm"
    return H


def tables(recs: dict) -> dict:
    T = {}
    for s in ORDER:
        if s not in recs:
            continue
        r = recs[s]
        qs = r["questions"]
        a1 = [q["score"] for q in qs if q["task"] != "T4" and q["id"] not in INADMISSIBLE | DROP_Q]
        ids, c, st = a2_vectors(r)
        strata = {}
        for k in ("explicit", "implicit"):
            _, ck, sk = a2_vectors(r, stratum=k)
            strata[k] = {"current": mean(ck), "stale": mean(sk), "n": len(ck)}
        a3 = a3_vectors(r)
        n3 = len(a3)
        pos = {f"{lo}-{hi - 1}": mean(int(d["control_intact"]) for d in a3 if lo <= (d["position"] or 0) < hi and d["control_before_ok"])
               for lo, hi in ((0, 12), (12, 24), (24, 99))} if a3 else {}
        T[s] = {"A1": mean(a1), "A2_current": mean(c), "A2_stale": mean(st), "A2_FAMA": round(mean(c) - mean(st), 3), "A2_by_stratum": strata,
                "A3_forgot": mean(int(d["forgot"]) for d in a3), "A3_control_intact": mean(int(d["control_intact"]) for d in a3 if d["control_before_ok"]),
                "A3_n_valid_controls": sum(d["control_before_ok"] for d in a3),
                "A3_leak": mean(int(d["leak"]) for d in a3 if d["leak_measurable"]), "A3_n_leak_measurable": sum(d["leak_measurable"] for d in a3),
                "A3_control_intact_by_position": pos, "A3_n": n3,
                "A3_by_shape": {sh: {"forgot": mean(int(d["forgot"]) for d in a3 if d["shape"] == sh),
                                     "control_intact": mean(int(d["control_intact"]) for d in a3 if d["shape"] == sh and d["control_before_ok"])}
                                for sh in ("single", "entity", "intent")} if a3 else {},
                "A4_fooled": mean(int(p["fooled"]) for p in r.get("poisons") or [] if "fooled" in p),
                "A4_label_survives": mean(int(any("unverified" in it["content"].lower() for it in p.get("retrieved", [])))
                                          for p in r.get("poisons") or [] if "fooled" in p),
                "A5_write_calls_per_memory": r["ingest"]["llm_calls_per_memory"], "A5_ingest_usd": r["ingest"]["cost_usd"],
                "A5_read_prompt_tok_per_q": r["summary"]["A5_cost"]["read_prompt_tokens_per_query"],
                "judge_decoy_acceptance": r["summary"]["judge_health"]["decoy_acceptance"], "axes": r.get("axes")}
    return T


def judge_agreement(recs: dict, rj: dict) -> dict:
    """κ (binary) between the Sonnet judgement and each alternative judge on LLM-branch items."""
    out = {}
    for name, alt in rj.items():
        a, b = [], []
        for s, r in recs.items():
            for q in r["questions"]:
                k = f"{s}|q|{q['id']}"
                if k in alt and q.get("judge_mode") in ("normal_judge", "ref_abstention_judge"):
                    a.append(q["score"]); b.append(alt[k]["score"])
            for d in r.get("deletions") or []:
                for key, field, inv in (("forgot", "forgot", True), ("ctrl_after", "control_intact", False), ("ctrl_before", "control_before_ok", False)):
                    k = f"{s}|{key}|{d['id']}"
                    if k in alt:
                        a.append(float(d[field] if not inv else not d[field])); b.append(float(alt[k]["score"] >= 0.99))
        # outcomes use score >= 0.99; kappa at 0.5 is the calibration-set convention, kappa at 0.99 the decision threshold
        out[name] = {"n": len(a), "kappa_binary": round(kappa_binary(a, b), 3) if a else None,
                     "kappa_at_decision_threshold": round(kappa_binary(a, b, thr=0.99), 3) if a else None,
                     "disagreements": sum((x >= 0.5) != (y >= 0.5) for x, y in zip(a, b))}
    return out


def main():
    recs, rj = load()
    res = {"systems": sorted(recs), "inadmissible_items": sorted(INADMISSIBLE), "tables": tables(recs), "hypotheses_sonnet": hypotheses(recs)}
    for name, alt in rj.items():
        res[f"hypotheses_{name}"] = hypotheses(recs, alt)
    global DROP_Q, DROP_C
    DROP_Q, DROP_C = AMBIG_Q, AMBIG_C
    res["sensitivity_no_implicit_role"] = {"dropped_questions": sorted(AMBIG_Q), "dropped_cases": sorted(AMBIG_C),
                                           "tables": tables(recs), "hypotheses_sonnet": hypotheses(recs),
                                           **{f"hypotheses_{n}": hypotheses(recs, alt) for n, alt in rj.items()}}
    DROP_Q, DROP_C = set(), set()
    res["judge_agreement"] = judge_agreement(recs, rj)
    if "zep_graphiti" in recs:
        ir = recs["zep_graphiti"].get("invalidation_report", {})
        rows = [r for r in ir.get("rows", []) if r["id"] not in INADMISSIBLE]   # admissible keys only
        have = [r for r in rows if r["n_stale_edges"]]
        res["graphiti_invalidation"] = {"keys": len(rows), "keys_with_stale_edge": len(have),
                                        "keys_all_stale_invalidated": sum(r["n_stale_invalidated"] == r["n_stale_edges"] for r in have),
                                        "keys_any_stale_invalidated": sum(r["n_stale_invalidated"] > 0 for r in have),
                                        "keys_with_current_edge": sum(1 for r in rows if r["n_current_edges"])}
        res["graphiti_arm_outcomes"] = {}
        for arm in [None] + list(recs["zep_graphiti"].get("read_arms") or {}):
            _, c, st = a2_vectors(recs["zep_graphiti"], arm=arm)
            res["graphiti_arm_outcomes"][arm or "r2"] = {"current": sum(c), "stale": sum(st), "other": len(c) - sum(c) - sum(st)}
    (R3 / "analysis.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "tables"}, indent=1)[:6000])


if __name__ == "__main__":
    main()
