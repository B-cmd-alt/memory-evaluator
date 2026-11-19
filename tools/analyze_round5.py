#!/usr/bin/env python3
"""Round-5 analysis, Part A (experiments/prereg_round5.md): purge-adjusted residue per arm and H16-H18.

    python -m tools.analyze_round5      # writes results/round5/analysis.json and prints the summary

Residue of a probe = recovered before AND at the end state AND NOT after the full purge (judge-free string rule).
Statistics: probe-level pairing, cases as bootstrap clusters resampled within seed, one-sided p-values, Holm.
"""
from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

R5 = Path("results/round5")
NATIVE = ["mem0", "generative_agents", "zep_graphiti", "memory_tool_edit"]
CONTROL = ["full_context", "minmem", "mastra_om", "memory_tool_rebuild"]
B = 10_000


def load() -> dict:
    """arms[arm][(seed, case_id)] = case record."""
    arms: dict = defaultdict(dict)
    meta: dict = {}
    for f in sorted(R5.glob("*_s*.json")):
        r = json.loads(f.read_text())
        if not r.get("cases"):
            continue
        meta.setdefault(r["arm"], []).append({"seed": r["seed"], "status": r["status"], "errors": r.get("errors"), "delete_method": r["delete_method"]})
        for c in r["cases"]:
            arms[r["arm"]][(r["seed"], c["id"])] = c
    return arms, meta


def residue(p: dict, stage: str = "end", need_before: bool = True) -> bool | None:
    need = [p.get("before"), p.get(stage), p.get("purge")]
    if any(x is None or x["recovered"] is None for x in need):
        return None
    if need_before and not need[0]["recovered"]:
        return None
    return bool(need[1]["recovered"] and not need[2]["recovered"])


def case_vec(arms, arm: str, stage="end", kinds=None, need_before=True, require_direct_before=False) -> dict:
    """(seed, case) -> 1/0 any-probe residue; cases with no counted probe are absent."""
    out = {}
    for key, c in arms.get(arm, {}).items():
        if require_direct_before and not (c["probes"][0]["before"]["recovered"]):
            continue
        vals = [residue(p, stage, need_before) for p in c["probes"] if kinds is None or p["kind"] in kinds]
        vals = [v for v in vals if v is not None]
        if vals:
            out[key] = float(any(vals))
    return out


def probe_vec(arms, arm: str, stage="end", need_before=True) -> dict:
    """(seed, case, probe index) -> 1/0."""
    out = {}
    for (seed, cid), c in arms.get(arm, {}).items():
        for i, p in enumerate(c["probes"]):
            v = residue(p, stage, need_before)
            if v is not None:
                out[(seed, cid, i)] = float(v)
    return out


def boot(keys: list, x: np.ndarray, y: np.ndarray | None = None, seed: int = 0):
    """mean(x) [- mean(y)] with cases as clusters resampled within seed. keys[i][:2] = (seed, case)."""
    if not len(keys):
        return None
    cl = {k: i for i, k in enumerate(dict.fromkeys(k[:2] for k in keys))}
    ci = np.array([cl[k[:2]] for k in keys])
    cseed = np.array([k[0] for k in cl])
    rng = np.random.default_rng(seed)
    counts = np.zeros((B, len(cl)))
    for s in np.unique(cseed):
        idx = np.where(cseed == s)[0]
        counts[:, idx] = rng.multinomial(len(idx), np.ones(len(idx)) / len(idx), size=B)
    w = counts[:, ci]
    d = x - (y if y is not None else 0)
    est = float(d.mean())
    b = (w * d).sum(1) / np.maximum(w.sum(1), 1e-9)
    return est, b


def rate(arms, arm, **kw) -> dict:
    v = case_vec(arms, arm, **kw)
    if not v:
        return {"n": 0}
    keys = sorted(v)
    est, b = boot(keys, np.array([v[k] for k in keys]))
    return {"n": len(keys), "residue_cases": int(sum(v.values())), "rate": round(est, 3), "ci95": [round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)]}


def compare_probes(arms, a: str, b: str, need_before=True) -> dict:
    va, vb = probe_vec(arms, a, need_before=need_before), probe_vec(arms, b, need_before=need_before)
    keys = sorted(set(va) & set(vb))
    if not keys:
        return {"n": 0, "verdict": "not run"}
    est, bt = boot(keys, np.array([va[k] for k in keys]), np.array([vb[k] for k in keys]))
    lo, hi = np.percentile(bt, [2.5, 97.5])
    return {"n_probes": len(keys), "n_cases": len({k[:2] for k in keys}), f"rate_{a}": round(float(np.mean([va[k] for k in keys])), 3),
            f"rate_{b}": round(float(np.mean([vb[k] for k in keys])), 3), "estimate": round(est, 3), "ci95": [round(float(lo), 3), round(float(hi), 3)],
            "p_one_sided": round(float(np.mean(bt <= 0)), 4), "verdict": "holds" if lo > 0 else ("falsified" if hi < 0 else "unresolved")}


def h16(arms, pool: list[str], margin=0.03) -> dict:
    keys, xa, xd = [], [], []
    for arm in pool:
        allp = case_vec(arms, arm, require_direct_before=True)
        direct = case_vec(arms, arm, kinds={"direct"}, require_direct_before=True)
        for k in sorted(set(allp) & set(direct)):
            keys.append((k[0], f"{arm}|{k[1]}"))
            xa.append(allp[k]); xd.append(direct[k])
    if not keys:
        return {"n": 0, "verdict": "not run"}
    est, bt = boot(keys, np.array(xa), np.array(xd))
    lo, hi = np.percentile(bt, [2.5, 97.5])
    return {"n_case_arm_pairs": len(keys), "residue_all_probes": round(float(np.mean(xa)), 3), "residue_direct_only": round(float(np.mean(xd)), 3),
            "estimate": round(est, 3), "ci95": [round(float(lo), 3), round(float(hi), 3)], "margin": margin,
            "p_one_sided": round(float(np.mean(bt <= margin)), 4), "verdict": "holds" if lo > margin else ("falsified" if hi < margin else "unresolved")}


def holm(ps: dict) -> dict:
    order = sorted((k for k in ps if ps[k] is not None), key=lambda k: ps[k])
    out, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * ps[k]))
        out[k] = round(run, 4)
    return out


def adjusted(arms: dict) -> dict:
    """Amendment 2: at the after/end/purge stages 'recovered' becomes 'string in answer AND the judge says the answer
    discloses a fact the deleted notes state'. A string-positive answer with no judge verdict becomes None (excluded)."""
    out: dict = defaultdict(dict)
    side: dict = {}
    for arm, cases in arms.items():
        for (seed, cid), c in cases.items():
            f = R5 / "disclosure" / f"{arm}_s{seed}.json"
            if f not in side:
                side[f] = json.loads(f.read_text()) if f.exists() else {}
            c2 = copy.deepcopy(c)
            for i, p in enumerate(c2["probes"]):
                for st in ("after", "end", "purge"):
                    if p.get(st) and p[st]["recovered"]:
                        v = side[f].get(f"{cid}|{i}|{st}")
                        p[st]["recovered"] = v["disclosed"] if v else None
            out[arm][(seed, cid)] = c2
    return out


def analyse(arms: dict) -> dict:
    res = {"arms": {}}
    for arm in CONTROL + NATIVE:
        if arm not in arms:
            continue
        cases = arms[arm].values()
        by_kind = defaultdict(list)
        for c in cases:
            for p in c["probes"]:
                v = residue(p)
                if v is not None:
                    by_kind[p["kind"]].append(v)
        valid = [c for c in cases if c["control_before_ok"]]
        res["arms"][arm] = {
            "seeds": sorted({k[0] for k in arms[arm]}),
            "end_state": rate(arms, arm), "immediate": rate(arms, arm, stage="after"), "direct_probe_only": rate(arms, arm, kinds={"direct"}),
            "no_before_filter": rate(arms, arm, need_before=False),
            "by_shape": {sh: rate({arm: {k: c for k, c in arms[arm].items() if c["shape"] == sh}}, arm) for sh in ("single", "entity", "intent")},
            "by_probe_kind": {k: {"n": len(v), "rate": round(float(np.mean(v)), 3)} for k, v in sorted(by_kind.items())},
            "probes_recovered_before": round(float(np.mean([bool(p["before"]["recovered"]) for c in cases for p in c["probes"] if p["before"]["recovered"] is not None])), 3),
            "control_intact": {"n_valid": len(valid), "rate": round(float(np.mean([c["control_after_ok"] for c in valid])), 3) if valid else None},
            "delete_seconds_mean": round(float(np.mean([c["delete_seconds"] for c in cases])), 1),
            "delete_usd_mean": round(float(np.mean([c["delete_cost"]["usd"] for c in cases])), 4),
            "context_leak_cases_end": int(sum(any((p.get("end") or {}).get("context_leak") for p in c["probes"]) for c in cases if c["leak_measurable"]))}
    H = {"H16": {"claim": "native arms: residue(all probes) - residue(direct only) > 0.03", **h16(arms, [a for a in NATIVE if a in arms]),
                 "control_floor": h16(arms, [a for a in CONTROL if a in arms])},
         "H17": {"claim": "probe residue(mem0) > residue(minmem)", **compare_probes(arms, "mem0", "minmem"),
                 "secondary": {a: compare_probes(arms, a, "minmem") for a in ("generative_agents", "zep_graphiti", "memory_tool_edit")},
                 "no_before_filter": compare_probes(arms, "mem0", "minmem", need_before=False)},
         "H18": {"claim": "probe residue(memory_tool_edit) > residue(memory_tool_rebuild)", **compare_probes(arms, "memory_tool_edit", "memory_tool_rebuild"),
                 "no_before_filter": compare_probes(arms, "memory_tool_edit", "memory_tool_rebuild", need_before=False)}}
    res["hypotheses"] = H
    res["p_values_for_holm_with_H19"] = {h: v.get("p_one_sided") for h, v in H.items()}
    return res


def judge_vs_hand() -> dict | None:
    f = R5 / "disclosure" / "hand_labels_s42.json"
    if not f.exists():
        return None
    side: dict = {}
    agree = n = 0
    diff = []
    for k, hand in json.loads(f.read_text())["labels"].items():
        arm, rest = k.split("|", 1)
        sf = R5 / "disclosure" / f"{arm}_s42.json"
        if sf not in side:
            side[sf] = json.loads(sf.read_text()) if sf.exists() else {}
        v = side[sf].get(rest)
        if v is None:
            continue
        n += 1
        agree += v["disclosed"] == hand
        if v["disclosed"] != hand:
            diff.append({"item": k, "hand": hand, "judge": v["disclosed"], "statement": v["statement"]})
    return {"n": n, "agree": agree, "disagreements": diff}


def show(title: str, res: dict):
    print(f"\n=== {title}")
    for arm, a in res["arms"].items():
        e = a["end_state"]
        print(f"{arm:22s} seeds {a['seeds']} residue {e.get('residue_cases')}/{e.get('n')} = {e.get('rate')} {e.get('ci95')} | immediate {a['immediate'].get('rate')} | direct-only {a['direct_probe_only'].get('rate')}"
              f" | controls intact {a['control_intact']['rate']} ({a['control_intact']['n_valid']}) | by kind {a['by_probe_kind']}")
    for h, v in res["hypotheses"].items():
        print(h, v.get("verdict"), v.get("estimate"), v.get("ci95"), "p", v.get("p_one_sided"))


def main():
    arms, meta = load()
    adj = adjusted(arms)
    held = {arm: {k: c for k, c in cases.items() if k[0] != 42} for arm, cases in adj.items()}
    held = {a: c for a, c in held.items() if c}
    res = {"meta": meta, "string_rule_registered": analyse(arms), "disclosure_adjusted": analyse(adj),
           "disclosure_adjusted_seeds_7_2026_only": analyse(held) if held else None, "judge_vs_hand_labels_seed42": judge_vs_hand()}
    (R5 / "analysis.json").write_text(json.dumps(res, indent=1))
    show("string rule (registered)", res["string_rule_registered"])
    show("disclosure-adjusted (amendment 2)", res["disclosure_adjusted"])
    if held:
        show("disclosure-adjusted, seeds 7 and 2026 only", res["disclosure_adjusted_seeds_7_2026_only"])
    print("\njudge vs hand labels:", json.dumps(res["judge_vs_hand_labels_seed42"], indent=1)[:1500])


if __name__ == "__main__":
    main()
