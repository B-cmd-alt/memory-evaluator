#!/usr/bin/env python3
"""Round-4 analysis (experiments/prereg_round4.md): the scale-ladder tables and H10-H15.

    python -m tools.analyze_round4                               # the OpenRouter-path cells, results/round4
    R4_DIR=results/round4_cli python -m tools.analyze_round4     # the subscription-path cells (amendment 3)
Writes <dir>/analysis.json and prints the summary. The two directories are never pooled.

Metrics and statistics exactly as pre-registered: item score (binary for T1/T3/T4/T5, nugget recall for T2), task-macro
accuracy (primary) and item-mean accuracy, FAMA over T4, a cluster bootstrap (update keys within seed, 10,000 resamples,
the same resample for every arm and reader in a comparison), one-sided p-values, Holm over the six primary tests.
Failed answers have no score and drop out pairwise.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

R4 = Path(os.environ.get("R4_DIR", "results/round4"))
SONNET, GEMINI, HAIKU, LUNA = "anthropic/claude-sonnet-5.5", "google/gemini-3.8-flash", "anthropic/claude-haiku-5.5", "openai/gpt-6-luna"
READERS = [SONNET, GEMINI, HAIKU, LUNA]
RUNGS = [0, 50000, 200000, 800000, 3200000]
RUNG_NAME = {0: "10K", 50000: "50K", 200000: "200K", 800000: "800K", 3200000: "3.2M"}
TASKS = ["T1", "T2", "T3", "T4", "T5"]
LLM_WRITE = ["mem0", "ms_graphrag", "zep_graphiti", "anthropic_memory_tool", "mastra_om", "generative_agents"]
LIST_PRICE = {SONNET: (2.0, 10.0), GEMINI: (0.75, 3.75), HAIKU: (0.10, 0.50), LUNA: (0.10, 0.50)}   # $/M tokens in, out (base tier)
B = 10_000
_F = re.compile(r"(.+)_s(\d+)_t(\d+)\.json$")


def load() -> dict:
    """cells[(system, rung)][seed] = record (status OK or PARTIAL with questions)."""
    cells: dict = defaultdict(dict)
    for f in sorted(R4.glob("*_s*_t*.json")):
        m = _F.match(f.name)
        if not m:
            continue
        r = json.loads(f.read_text())
        if r.get("questions"):
            cells[(m.group(1), int(m.group(3)))][int(m.group(2))] = r
    return cells


def cluster_of(q: dict, seed: int) -> str:
    if q["task"] == "T4":
        return f"{seed}|{q['evidence_ids'][0]}"
    if q["task"] == "T3" and len(q["evidence_ids"]) == 2:
        return f"{seed}|{q['evidence_ids'][1]}"
    return f"{seed}|{q['id']}"


def items(cells: dict, system: str, rung: int, reader, seeds=None) -> dict:
    """(seed, qid) -> {task, cluster, score, fama} ; reader may be a list (scores averaged over readers, None if any failed)."""
    out = {}
    rs = reader if isinstance(reader, list) else [reader]
    for seed, rec in cells.get((system, rung), {}).items():
        if seeds and seed not in seeds:
            continue
        for q in rec["questions"]:
            sc, fa = [], []
            for r in rs:
                b = q["by_reader"].get(r)
                if b is None or b["score"] is None:
                    sc = None
                    break
                sc.append(b["score"] if q["task"] == "T2" else float(b["score"] >= 0.99))
                fa.append((b["outcome"] == "current") - (b["outcome"] == "stale") if q["task"] == "T4" else 0.0)
            if sc is None:
                continue
            out[(seed, q["id"])] = {"task": q["task"], "cluster": cluster_of(q, seed), "seed": seed, "score": float(np.mean(sc)), "fama": float(np.mean(fa))}
    return out


class Paired:
    """Items common to several arms, with one shared set of cluster-bootstrap weights."""

    def __init__(self, arms: list[dict], seed: int = 0):
        keys = sorted(set.intersection(*[set(a) for a in arms])) if arms and all(arms) else []
        self.n = len(keys)
        self.keys = keys
        if not keys:
            return
        a0 = arms[0]
        self.task = np.array([TASKS.index(a0[k]["task"]) for k in keys])
        self.score = [np.array([a[k]["score"] for k in keys]) for a in arms]
        self.fama = [np.array([a[k]["fama"] for k in keys]) for a in arms]
        self.seeds = np.array([a0[k]["seed"] for k in keys])
        cl = [a0[k]["cluster"] for k in keys]
        cid = {c: i for i, c in enumerate(dict.fromkeys(cl))}
        cl_idx = np.array([cid[c] for c in cl])
        cl_seed = np.zeros(len(cid), dtype=int)
        for c, s in zip(cl_idx, self.seeds):
            cl_seed[c] = s
        rng = np.random.default_rng(seed)
        counts = np.zeros((B, len(cid)))
        for s in np.unique(cl_seed):   # resample clusters within seed
            idx = np.where(cl_seed == s)[0]
            counts[:, idx] = rng.multinomial(len(idx), np.ones(len(idx)) / len(idx), size=B)
        self.w = counts[:, cl_idx]            # B x n_items
        self.n_clusters = len(cid)

    def macro(self, i: int, w=None) -> np.ndarray | float:
        """task-macro accuracy of arm i (point estimate if w is None, else one value per resample)."""
        x = self.score[i]
        if w is None:
            return float(np.mean([x[self.task == t].mean() for t in range(5) if (self.task == t).any()]))
        vals = []
        for t in range(5):
            m = self.task == t
            if m.any():
                den = w[:, m].sum(1)
                vals.append(np.where(den > 0, (w[:, m] * x[m]).sum(1) / np.maximum(den, 1e-9), np.nan))
        return np.nanmean(np.vstack(vals), axis=0)

    def item_mean(self, i: int) -> float:
        return float(self.score[i].mean())

    def famas(self, i: int, w=None):
        m = self.task == 3
        if not m.any():
            return float("nan") if w is None else np.full(B, np.nan)
        x = self.fama[i][m]
        if w is None:
            return float(x.mean())
        den = w[:, m].sum(1)
        return (w[:, m] * x).sum(1) / np.maximum(den, 1e-9)

    def stat(self, i: int, metric: str, w=None):
        return self.macro(i, w) if metric == "acc" else self.famas(i, w)


def contrast(cells, arms: list[tuple], coef: list[float], metric: str = "acc", seeds=None, margin: float = 0.0, predict: int = +1) -> dict:
    """sum(coef_i * metric(arm_i)) on common items. arms = [(system, rung, reader), ...]. One-sided p for the predicted sign
    (predict=+1: estimate > -margin). Returns estimate, CI, p, n, per-seed direction."""
    data = [items(cells, s, r, rd, seeds) for s, r, rd in arms]
    # arms on different rungs / readers share items by (seed, qid): the core questions are identical on every rung
    out = contrast_data(data, coef, metric, margin, predict)
    if out.get("n"):
        out["arms"] = [f"{s}@{RUNG_NAME[r]}/{(rd if isinstance(rd, str) else 'pooled').split('/')[-1]}" for s, r, rd in arms]
    return out


def contrast_data(data: list[dict], coef: list[float], metric: str = "acc", margin: float = 0.0, predict: int = +1) -> dict:
    P = Paired(data)
    if not P.n:
        return {"n": 0, "verdict": "not run"}
    est = sum(c * P.stat(i, metric) for i, c in enumerate(coef))
    boot = sum(c * P.stat(i, metric, P.w) for i, c in enumerate(coef))
    boot = boot[~np.isnan(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = float(np.mean(boot <= -margin)) if predict > 0 else float(np.mean(boot >= margin))
    by_seed = {}
    for s in np.unique(P.seeds):
        Ps = Paired([{k: v for k, v in d.items() if k[0] == s} for d in data])
        by_seed[int(s)] = round(sum(c * Ps.stat(i, metric) for i, c in enumerate(coef)), 3) if Ps.n else None
    return {"n": P.n, "clusters": P.n_clusters, "estimate": round(float(est), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)],
            "p_one_sided": round(p, 4), "by_seed": by_seed, "item_mean": [round(P.item_mean(i), 3) for i in range(len(data))],
            "macro": [round(P.macro(i), 3) for i in range(len(data))]}


def verdict(c: dict, predict: int = +1, margin: float = 0.0) -> str:
    if not c.get("n"):
        return "not run"
    lo, hi = c["ci95"]
    if predict > 0:
        return "holds" if lo > -margin else ("falsified" if hi < -margin else "unresolved")
    return "holds" if hi < margin else ("falsified" if lo > margin else "unresolved")


def holm(ps: dict) -> dict:
    order = sorted((k for k in ps if ps[k] is not None), key=lambda k: ps[k])
    out, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * ps[k]))
        out[k] = round(run, 4)
    return out


def cell_table(cells) -> dict:
    T = {}
    for (system, rung), by_seed in sorted(cells.items()):
        row = {"seeds": sorted(by_seed), "status": {s: r["status"] for s, r in by_seed.items()}, "readers": {}}
        for rd in READERS:
            it = items(cells, system, rung, rd)
            if not it:
                continue
            P = Paired([it])
            qs = [(rec, q) for rec in by_seed.values() for q in rec["questions"] if rd in q["by_reader"]]
            fa = [q["by_reader"][rd].get("judge_mode") == "false_abstention" for _, q in qs if q["task"] in ("T1", "T3", "T4") and q["by_reader"][rd]["score"] is not None]
            hedged = [q["by_reader"][rd].get("judge_mode") == "false_abstention" and bool(q["by_reader"][rd].get("ref_in_answer")) for _, q in qs if q["task"] in ("T1", "T3", "T4")]
            cost = [rec["read"]["by_reader"].get(rd) for rec in by_seed.values()]
            nq = sum(len(rec["questions"]) for rec in by_seed.values())
            usd = sum(c["usd"] for c in cost if c)
            lp = LIST_PRICE[rd]
            list_usd = sum(c["prompt_tokens"] * lp[0] + c["completion_tokens"] * lp[1] for c in cost if c) / 1e6
            row["readers"][rd] = {"n": P.n, "acc_macro": round(P.macro(0), 3), "acc_item": round(P.item_mean(0), 3), "fama": round(P.famas(0), 3),
                                  "by_task": {TASKS[t]: round(float(P.score[0][P.task == t].mean()), 3) for t in range(5) if (P.task == t).any()},
                                  "false_abstention_rate": round(float(np.mean(fa)), 3) if fa else None,
                                  "hedged_but_contains_reference": int(sum(hedged)),
                                  "errors": sum(rec.get("errors_by_reader", {}).get(rd, 0) for rec in by_seed.values()),
                                  "read_usd_per_query_billed": round(usd / nq, 5), "read_usd_per_query_list": round(list_usd / nq, 5),
                                  "read_prompt_tokens_per_query": round(sum(c["prompt_tokens"] for c in cost if c) / nq)}
        ev = [q["evidence_in_context"] for rec in by_seed.values() for q in rec["questions"] if q.get("evidence_in_context") is not None]
        row["evidence_in_context"] = round(float(np.mean(ev)), 3) if ev else None
        ing = [rec["ingest"] for rec in by_seed.values()]
        row["ingest"] = {"llm_calls_per_memory": round(float(np.mean([i["llm_calls_per_memory"] for i in ing])), 3),
                         "usd_per_store": round(float(np.mean([i["usd"] for i in ing])), 3), "seconds": round(float(np.mean([i["seconds"] for i in ing])))}
        row["store_tokens"] = int(np.mean([rec["store_tokens"] for rec in by_seed.values()]))
        T[f"{system}@{RUNG_NAME[rung]}"] = row
    return T


def load_lme() -> dict:
    out = {}
    for f in sorted((R4 / "lme").glob("*.json")):
        r = json.loads(f.read_text())
        if r.get("questions"):
            out[r["system"]] = r
    return out


def lme_items(lme: dict, system: str, reader: str) -> dict:
    """(0, qid) -> item; every LongMemEval question is its own cluster."""
    out = {}
    for q in lme.get(system, {}).get("questions", []):
        b = q["by_reader"].get(reader)
        if b is None or b["score"] is None:
            continue
        out[(0, q["id"])] = {"task": q["task"], "cluster": q["id"], "seed": 0, "score": float(b["score"] >= 0.99), "fama": 0.0, "stratum": q.get("stratum")}
    return out


def lme_summary(lme: dict, system: str, reader: str) -> dict:
    it = lme_items(lme, system, reader)
    P = Paired([it])
    by = defaultdict(list)
    for v in it.values():
        by[v["stratum"]].append(v["score"])
    asked = sum(1 for q in lme[system]["questions"] if reader in q["by_reader"])
    return {"n": P.n, "failed": asked - P.n, "acc_macro": round(P.macro(0), 3), "acc_item": round(P.item_mean(0), 3),
            "by_question_type": {k: [round(float(np.mean(v)), 3), len(v)] for k, v in sorted(by.items())}}


def path_check(cells) -> dict | None:
    """Amendment 3: the same cell on the other path (OpenRouter vs claude -p), item-level agreement of the binary score."""
    other = Path("results/round4") if R4.name != "round4" else Path("results/round4_cli")
    out = {}
    for (system, rung), by_seed in sorted(cells.items()):
        for seed, rec in by_seed.items():
            f = other / f"{system}_s{seed}_t{rung}.json"
            if not f.exists():
                continue
            o = {q["id"]: q for q in json.loads(f.read_text()).get("questions", [])}
            for rd in (SONNET, HAIKU):
                pairs = [(float(q["by_reader"][rd]["score"] >= 0.99), float(o[q["id"]]["by_reader"][rd]["score"] >= 0.99)) for q in rec["questions"]
                         if q["id"] in o and q["task"] != "T2" and q["by_reader"].get(rd, {}).get("score") is not None and o[q["id"]]["by_reader"].get(rd, {}).get("score") is not None]
                if pairs:
                    a = out.setdefault(f"{system}@{RUNG_NAME[rung]}/{rd.split('/')[-1]}", {"n": 0, "agree": 0, "acc_here": 0.0, "acc_other": 0.0})
                    a["n"] += len(pairs); a["agree"] += sum(x == y for x, y in pairs); a["acc_here"] += sum(x for x, _ in pairs); a["acc_other"] += sum(y for _, y in pairs)
    if not out:
        return None
    for a in out.values():
        a["acc_here"] = round(a["acc_here"] / a["n"], 3); a["acc_other"] = round(a["acc_other"] / a["n"], 3)
    tot = {"n": sum(a["n"] for a in out.values()), "agree": sum(a["agree"] for a in out.values())}
    return {"other_dir": str(other), "total": {**tot, "rate": round(tot["agree"] / tot["n"], 4)}, "cells": out}


def hypotheses(cells) -> dict:
    H, P = {}, {}
    # H10: crossover at 800K, Sonnet, matched cue
    c = contrast(cells, [("minmem", 800000, SONNET), ("full_context_hdr", 800000, SONNET)], [1, -1])
    H["H10"] = {"claim": "acc(minmem) > acc(full_context_hdr) at 800K, Sonnet", **c, "verdict": verdict(c),
                "also": {name: contrast(cells, [(a, 800000, SONNET), (b, 800000, SONNET)], [1, -1])
                         for name, (a, b) in {"minmem - full_context": ("minmem", "full_context"), "hybrid_k20 - full_context_hdr": ("hybrid_k20", "full_context_hdr"),
                                              "hybrid_k20 - full_context": ("hybrid_k20", "full_context"), "full_context_hdr - full_context": ("full_context_hdr", "full_context")}.items()},
                "crossover_rung_by_reader": {}}
    for rd in READERS:
        cross, at10 = "none up to 800K", None
        for rung in RUNGS[:4]:
            cc = contrast(cells, [("minmem", rung, rd), ("full_context_hdr", rung, rd)], [1, -1])
            if not cc.get("n"):
                continue
            if rung == 0:
                at10 = cc["ci95"][0] > 0
            elif cc["ci95"][0] > 0 and not at10:
                cross = RUNG_NAME[rung]
                break
        H["H10"]["crossover_rung_by_reader"][rd.split("/")[-1]] = cross
    P["H10"] = c.get("p_one_sided")
    # H11: weaker reader crosses earlier (diff in diff at 200K)
    c = contrast(cells, [("minmem", 200000, HAIKU), ("full_context_hdr", 200000, HAIKU), ("minmem", 200000, SONNET), ("full_context_hdr", 200000, SONNET)], [1, -1, -1, 1])
    H["H11"] = {"claim": "(minmem - full_context_hdr) larger for Haiku than Sonnet at 200K", **c, "verdict": verdict(c)}
    P["H11"] = c.get("p_one_sided")
    # H12: non-inferiority vs each LLM-write system at 50K, Sonnet, margin 0.05 (conjunction)
    parts = {}
    for x in LLM_WRITE:
        cc = contrast(cells, [("minmem", 50000, SONNET), (x, 50000, SONNET)], [1, -1], margin=0.05)
        parts[x] = {**cc, "verdict": verdict(cc, margin=0.05)}
    vs = [p["verdict"] for p in parts.values()]
    H["H12"] = {"claim": "acc(minmem) - acc(X) > -0.05 for every LLM-write X at 50K, Sonnet", "parts": parts,
                "verdict": "not run" if all(v == "not run" for v in vs) else ("holds" if all(v == "holds" for v in vs) else ("falsified" if any(v == "falsified" for v in vs) else "unresolved")),
                "not_run": [x for x, p in parts.items() if p["verdict"] == "not run"],
                "secondary_hybrid_k20": {x: contrast(cells, [("hybrid_k20", 50000, SONNET), (x, 50000, SONNET)], [1, -1], margin=0.05) for x in LLM_WRITE},
                "secondary_fama": {x: contrast(cells, [("minmem", 50000, SONNET), (x, 50000, SONNET)], [1, -1], metric="fama", margin=0.10) for x in LLM_WRITE},
                "secondary_other_rungs": {f"{x}@{RUNG_NAME[r]}": contrast(cells, [("minmem", r, SONNET), (x, r, SONNET)], [1, -1], margin=0.05)
                                          for x in LLM_WRITE for r in (0, 200000, 800000) if (x, r) in cells}}
    ps = [p.get("p_one_sided") for p in parts.values() if p.get("n")]
    P["H12"] = max(ps) if ps else None
    # H13: vector_rag FAMA drops from 10K to 3.2M
    c = contrast(cells, [("vector_rag", 0, SONNET), ("vector_rag", 3200000, SONNET)], [1, -1], metric="fama")
    H["H13"] = {"claim": "FAMA(vector_rag) at 10K > at 3.2M, Sonnet", **c, "verdict": verdict(c),
                "secondary": {a: contrast(cells, [(a, 0, SONNET), (a, 3200000, SONNET)], [1, -1], metric="fama") for a in ("bm25", "vector_k20", "bm25_k20", "hybrid_k20", "minmem")},
                "secondary_did_minmem_vs_vector_k20": contrast(cells, [("vector_k20", 0, SONNET), ("vector_k20", 3200000, SONNET), ("minmem", 0, SONNET), ("minmem", 3200000, SONNET)], [1, -1, -1, 1], metric="fama")}
    P["H13"] = c.get("p_one_sided")
    # H14: expansion at matched k, 3.2M, pooled readers
    pool = [r for r in READERS if items(cells, "minmem_nohdr", 3200000, r) and items(cells, "hybrid_k20", 3200000, r)] or READERS
    c = contrast(cells, [("minmem_nohdr", 3200000, pool), ("hybrid_k20", 3200000, pool)], [1, -1])
    H["H14"] = {"claim": "acc(minmem_nohdr) > acc(hybrid_k20) at 3.2M, pooled readers", "pooled_readers": [r.split("/")[-1] for r in pool],
                "registered_pool_complete": len(pool) == len(READERS), **c, "verdict": verdict(c),
                "secondary": {"minmem - minmem_nohdr FAMA @800K Haiku": contrast(cells, [("minmem", 800000, HAIKU), ("minmem_nohdr", 800000, HAIKU)], [1, -1], metric="fama"),
                              "order step": contrast(cells, [("minmem_order", 800000, HAIKU), ("minmem_nohdr", 800000, HAIKU)], [1, -1], metric="fama"),
                              "cue step": contrast(cells, [("minmem", 800000, HAIKU), ("minmem_order", 800000, HAIKU)], [1, -1], metric="fama")}}
    P["H14"] = c.get("p_one_sided")
    # H15: LongMemEval-S, Sonnet, non-inferiority of minmem to hybrid_k20 (margin 0.03)
    lme = load_lme()
    c = contrast_data([lme_items(lme, "minmem", SONNET), lme_items(lme, "hybrid_k20", SONNET)], [1, -1], margin=0.03)
    H["H15"] = {"claim": "LongMemEval-S, Sonnet: acc(minmem) - acc(hybrid_k20) > -0.03", **c, "verdict": verdict(c, margin=0.03),
                "systems": {k: {rd.split("/")[-1]: lme_summary(lme, k, rd) for rd in READERS if lme_items(lme, k, rd)} for k in lme},
                "vs_full_context": {f"{a} - {b} / {rd.split('/')[-1]}": contrast_data([lme_items(lme, a, rd), lme_items(lme, b, rd)], [1, -1])
                                    for rd in READERS for a in ("minmem", "hybrid_k20") for b in ("full_context", "full_context_hdr")
                                    if lme_items(lme, a, rd) and lme_items(lme, b, rd)}}
    P["H15"] = c.get("p_one_sided")
    adj = holm(P)
    for h in H:
        H[h]["p_holm"] = adj.get(h)
        if H[h]["verdict"] == "holds" and adj.get(h) is not None and adj[h] >= 0.05:
            H[h]["verdict"] = "holds (CI) but not after Holm"
    return H


def main():
    cells = load()
    res = {"dir": str(R4), "cells": cell_table(cells), "hypotheses": hypotheses(cells), "path_check": path_check(cells)}
    (R4 / "analysis.json").write_text(json.dumps(res, indent=1))
    # compact ladder print: task-macro accuracy, Sonnet | Haiku
    systems = list(dict.fromkeys(s for s, _ in cells))
    for rd in (SONNET, HAIKU):
        print(f"\n== task-macro accuracy / FAMA, reader {rd.split('/')[-1]} ==")
        print(f"{'system':22s}" + "".join(f"{RUNG_NAME[r]:>14s}" for r in RUNGS))
        for s in systems:
            row = ""
            for r in RUNGS:
                c = res["cells"].get(f"{s}@{RUNG_NAME[r]}", {}).get("readers", {}).get(rd)
                row += f"{(str(c['acc_macro']) + ' / ' + str(c['fama'])) if c else '-':>14s}"
            print(f"{s:22s}{row}")
    for h, v in res["hypotheses"].items():
        print(h, v["verdict"], v.get("estimate"), v.get("ci95"), "p", v.get("p_one_sided"), "holm", v.get("p_holm"), "n", v.get("n"))
    if res["path_check"]:
        print("path check vs", res["path_check"]["other_dir"], res["path_check"]["total"])


if __name__ == "__main__":
    main()
