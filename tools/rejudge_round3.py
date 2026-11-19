#!/usr/bin/env python3
"""Re-judge every LLM-judged item of the round-3 records with non-Claude judges (prereg_round3.md, judge robustness).

Covers A1/A2 questions whose strict-v3 verdict went through an LLM branch, the Graphiti read arms, and the A3 judgements
(`forgot` = the judge no longer accepts the pre-delete answer; `control_intact`; `control_before_ok`). Deterministic
branches are not re-judged: they agree by construction. Output: results/round3/rejudge/<judge>.json, keyed
"<system>|<kind>|<item id>" -> score (A1/A2) or bool (A3).

    python -m tools.rejudge_round3 --judges or:openai/gpt-6.1-terra or:google/gemini-3.8-flash
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
from pathlib import Path

from src.evaluation.strict_judge import strict_judge_v3, is_ans_abstention
from tools.cross_family_judge import make_judge

R3 = Path("results/round3")
LLM_MODES = ("normal_judge", "ref_abstention_judge")


def tasks(recs: list[dict], data: dict) -> list[tuple[str, str, str, str, str]]:
    """(key, kind, question, answer, reference) for every LLM-branch judgement."""
    dl = {c["id"]: c for c in data["deletions"]}
    out = []
    for r in recs:
        s = r["system"]
        for q in r["questions"]:
            if q.get("judge_mode") in LLM_MODES:
                out.append((f"{s}|q|{q['id']}", "q", q["question"], q["answer"], q["reference"]))
        for arm, rows in (r.get("read_arms") or {}).items():
            for q in rows:
                if q.get("judge_mode") in LLM_MODES:
                    ref = next(x["reference"] for x in r["questions"] if x["id"] == q["id"])
                    qq = next(x["question"] for x in r["questions"] if x["id"] == q["id"])
                    out.append((f"{s}|arm:{arm}|{q['id']}", "q", qq, q["answer"], ref))
        for d in r.get("deletions") or []:
            c = dl[d["id"]]
            if not is_ans_abstention(d["probe_after"]):
                out.append((f"{s}|forgot|{d['id']}", "forgot", c["probe_question"], d["probe_after"], c["probe_answer"]))
            out.append((f"{s}|ctrl_after|{d['id']}", "ctrl", c["control_question"], d["control_after"], c["control_answer"]))
            out.append((f"{s}|ctrl_before|{d['id']}", "ctrl", c["control_question"], d["control_before"], c["control_answer"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="+", default=["or:openai/gpt-6.1-terra", "or:google/gemini-3.8-flash"])
    ap.add_argument("--data", default="results/sixaxis_R3_seed42.json")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    data = json.load(open(a.data))
    recs = [json.loads(f.read_text()) for f in sorted(R3.glob("*_R3_seed42_sonnet_rep1.json"))]
    recs = [r for r in recs if r.get("status") == "OK"]
    todo = tasks(recs, data)
    (R3 / "rejudge").mkdir(parents=True, exist_ok=True)
    for spec in a.judges:
        f = R3 / "rejudge" / f"{spec.replace(':', '__').replace('/', '_')}.json"
        done = json.loads(f.read_text()) if f.exists() else {}
        ledger: list = []
        fn = make_judge(spec, ledger)

        def one(t):
            key, kind, q, ans, ref = t
            s, why, mode = strict_judge_v3(q, ans, ref, fn)
            return key, {"score": s, "mode": mode, "reasoning": why[:200]}

        rest = [t for t in todo if t[0] not in done]
        with cf.ThreadPoolExecutor(a.workers) as ex:
            for k, v in ex.map(one, rest):
                done[k] = v
        f.write_text(json.dumps(done, indent=0))
        print(spec, "items", len(done), "new", len(rest), "cost_usd", round(sum(x["cost_usd"] or 0 for x in ledger), 3))


if __name__ == "__main__":
    main()
