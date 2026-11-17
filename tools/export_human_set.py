#!/usr/bin/env python3
"""Export a blind 100-item human-label set from the round-3 records (prereg_round3.md, "Human labels").

Selection: every LLM-branch item on which the Sonnet judge and a non-Claude judge disagree, then stratified sampling over
(axis x Sonnet verdict) towards 50/50 accepted/rejected. Items are shuffled and carry no system name and no verdict.
Each item records its inclusion probability so agreement can be re-weighted back to the population.

Writes results/round3/human_labels/{items.csv, form.html, key.json}. The labeller fills items.csv (label = 1 the answer
matches the reference / 0 it does not / ? unsure) or uses form.html (opens from a file, downloads labels.csv).
Score with: python -m tools.export_human_set --score path/to/labels.csv
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import random
from collections import defaultdict
from pathlib import Path

from tools.analyze_round3 import load
from tools.kappa import kappa_binary

OUT = Path("results/round3/human_labels")
LLM = ("normal_judge", "ref_abstention_judge")


def pool(recs, data) -> list[dict]:
    dl = {c["id"]: c for c in data["deletions"]}
    items = []
    for s, r in recs.items():
        for q in r["questions"]:
            if q.get("judge_mode") in LLM:
                items.append({"key": f"{s}|q|{q['id']}", "axis": "A2" if q["task"] == "T4" else "A1", "question": q["question"],
                              "reference": q["reference"], "answer": q["answer"], "sonnet": int(q["score"] >= 0.99)})
        for d in r.get("deletions") or []:
            c = dl[d["id"]]
            items.append({"key": f"{s}|ctrl_after|{d['id']}", "axis": "A3", "question": c["control_question"], "reference": c["control_answer"],
                          "answer": d["control_after"], "sonnet": int(d["control_intact"])})
            if not any(p in d["probe_after"].lower() for p in ("don't have enough", "not enough information", "no information")):
                items.append({"key": f"{s}|forgot|{d['id']}", "axis": "A3", "question": c["probe_question"], "reference": c["probe_answer"],
                              "answer": d["probe_after"], "sonnet": int(not d["forgot"])})   # 1 = answer still gives the deleted value
    return items


def export(n=100, seed=0):
    recs, rj = load()
    data = json.load(open("results/sixaxis_R3_seed42.json"))
    P = pool(recs, data)
    rng = random.Random(seed)
    alts = {}
    for it in P:
        k = it["key"]
        it["alt"] = {name: (int(a[k]["score"] >= 0.99) if not k.split("|")[1] == "forgot" else int(a[k]["score"] >= 0.99)) for name, a in rj.items() if k in a}
        it["disagree"] = any(v != it["sonnet"] for v in it["alt"].values())
    pick = [it for it in P if it["disagree"]][:n]
    for it in pick:
        it["p_incl"] = 1.0
    strata = defaultdict(list)
    for it in P:
        if not it["disagree"]:
            strata[(it["axis"], it["sonnet"])].append(it)
    left = n - len(pick)
    keys = sorted(strata)
    for i, k in enumerate(keys):
        want = left // len(keys) + (1 if i < left % len(keys) else 0)
        take = rng.sample(strata[k], min(want, len(strata[k])))
        for it in take:
            it["p_incl"] = round(len(take) / len(strata[k]), 4)
        pick += take
    rng.shuffle(pick)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "items.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["item", "question", "reference", "answer", "label"])
        for i, it in enumerate(pick):
            w.writerow([i, it["question"], it["reference"], it["answer"], ""])
    (OUT / "key.json").write_text(json.dumps([{"item": i, **{k: it[k] for k in ("key", "axis", "sonnet", "alt", "disagree", "p_incl")}}
                                              for i, it in enumerate(pick)], indent=1))
    rows = "".join(f"<tr><td>{i}</td><td>{html.escape(it['question'])}</td><td>{html.escape(it['reference'])}</td><td>{html.escape(it['answer'])}</td>"
                   f"<td><label><input type=radio name=i{i} value=1>yes</label> <label><input type=radio name=i{i} value=0>no</label> "
                   f"<label><input type=radio name=i{i} value=?>unsure</label></td></tr>" for i, it in enumerate(pick))
    (OUT / "form.html").write_text(f"""<!doctype html><meta charset=utf-8><title>Answer labels</title>
<style>body{{font:14px system-ui;margin:16px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:6px;vertical-align:top}}</style>
<h1>Does the answer state the reference?</h1><p>For each row: <b>yes</b> if the answer gives the reference's fact (wording may differ),
<b>no</b> if it gives a different value, abstains, or hedges between values. Then press Download.</p>
<button onclick=dl()>Download labels.csv</button><table><tr><th>#</th><th>question</th><th>reference</th><th>answer</th><th>label</th></tr>{rows}</table>
<script>function dl(){{let s='item,label\\n';for(let i=0;i<{len(pick)};i++){{const e=document.querySelector('input[name=i'+i+']:checked');s+=i+','+(e?e.value:'')+'\\n'}}
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([s],{{type:'text/csv'}}));a.download='labels.csv';a.click()}}</script>""")
    print(f"{len(pick)} items ({sum(it['disagree'] for it in pick)} cross-judge disagreements) -> {OUT}")


def score(path: str):
    key = {k["item"]: k for k in json.load(open(OUT / "key.json"))}
    lab = {int(r["item"]): r["label"] for r in csv.DictReader(open(path)) if r.get("label") in ("0", "1")}
    ids = sorted(lab)
    h = [int(lab[i]) for i in ids]
    out = {"n_labelled": len(ids), "sonnet_kappa": round(kappa_binary([key[i]["sonnet"] for i in ids], h), 3)}
    for name in sorted({n for k in key.values() for n in k["alt"]}):
        sub = [i for i in ids if name in key[i]["alt"]]
        out[f"{name}_kappa"] = round(kappa_binary([key[i]["alt"][name] for i in sub], [int(lab[i]) for i in sub]), 3)
    wt = [1 / key[i]["p_incl"] for i in ids]
    out["sonnet_weighted_agreement"] = round(sum(w for i, w in zip(ids, wt) if key[i]["sonnet"] == int(lab[i])) / sum(wt), 3)
    print(json.dumps(out, indent=1))
    (OUT / "human_scores.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--score"); ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    score(a.score) if a.score else export(a.n)
