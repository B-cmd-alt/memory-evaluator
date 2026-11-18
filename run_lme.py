#!/usr/bin/env python3
"""Round 4, H15 — the held-out anchor: LongMemEval-S (cleaned) through the same runner as the ladder.

Each LongMemEval-S question has its OWN haystack (~48 sessions, ~115K tokens of real multi-session chat). The memory unit
is one ROUND (a user turn plus the assistant reply that follows it), dated with its session's date; the question is asked
with its question date. A stratified sample of questions is fixed by seed. Per question and system: ingest that haystack,
answer with every reader, judge with strict-judge v3.

    python run_lme.py --n 120 --systems bm25 vector_rag hybrid_rerank minmem full_context

Records: results/round4/lme/<system>.json (one file per system, questions appended; resumable).
Data: data/longmemeval/longmemeval_s_cleaned.json (xiaowu0162/longmemeval-cleaned on Hugging Face; not redistributed here).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import run_ladder as L

DATA = Path("data/longmemeval/longmemeval_s_cleaned.json")
TASK = {"single-session-user": "T1", "single-session-assistant": "T1", "single-session-preference": "T1", "multi-session": "T2",
        "temporal-reasoning": "T3", "knowledge-update": "T4"}


def sample(items: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Stratified by question type, proportional, deterministic."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for x in items:
        by[x["question_type"]].append(x)
    out = []
    for t in sorted(by):
        k = max(1, round(n * len(by[t]) / len(items)))
        out += rng.sample(sorted(by[t], key=lambda x: x["question_id"]), min(k, len(by[t])))
    return sorted(out, key=lambda x: x["question_id"])[:n]


def to_world(x: dict) -> dict:
    mems, evidence = [], []
    for sid, sdate, sess in zip(x["haystack_session_ids"], x["haystack_dates"], x["haystack_sessions"]):
        d = datetime.strptime(sdate.split(" (")[0] + " " + sdate.split(") ")[1], "%Y/%m/%d %H:%M")
        i = 0
        while i < len(sess):
            turns = [sess[i]]
            if sess[i]["role"] == "user" and i + 1 < len(sess) and sess[i + 1]["role"] == "assistant":
                turns.append(sess[i + 1])
            i += len(turns)
            mid = f"{sid}_r{len(mems)}"
            mems.append({"id": mid, "date": d.isoformat(), "content": "\n".join(f"{t['role']}: {t['content']}" for t in turns),
                         "entities": [], "topics": [], "kind": "round"})
            if any(t.get("has_answer") for t in turns):
                evidence.append(mid)
    mems.sort(key=lambda m: (m["date"], m["id"]))
    abstain = x["question_id"].endswith("_abs")
    qd = x["question_date"].split(" (")[0]
    q = {"id": x["question_id"], "task": "T5" if abstain else TASK[x["question_type"]], "stratum": x["question_type"],
         "question": f"{x['question']} (Asked on {qd}.)", "reference_answer": str(x["answer"]), "evidence_ids": evidence,
         "stale_answer": None, "nuggets": []}
    return {"meta": {"seed": 0, "target_tokens": 0, "store_tokens": sum(len(m["content"]) for m in mems) // 4, "n_memories": len(mems)},
            "memories": mems, "questions": [q]}


def score_lme(q: dict, answer: str) -> dict:
    """LongMemEval references are free text: every item goes through strict-judge v3 as a (question, answer, reference) triple
    (its reference-side abstention list covers the `_abs` items)."""
    if answer.startswith("<error"):
        return {"score": None, "outcome": "error", "judge_mode": "error", "judge_reasoning": ""}
    s, why, mode = L.strict_judge_v3(q["question"], answer, q["reference_answer"], L.judge_fn)
    return {"score": s, "outcome": "correct" if s >= 0.99 else "wrong", "judge_mode": mode, "judge_reasoning": why[:300]}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--readers", nargs="+", default=L.READERS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results/round4/lme")
    a = ap.parse_args()
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    items = sample(json.load(open(DATA)), a.n)
    L.score = score_lme
    for k in a.systems:
        f = out / f"{k}.json"
        rec = json.loads(f.read_text()) if f.exists() else {"system": k, "readers": a.readers, "questions": [], "ingest_usd": 0.0, "read": {}}
        done = {q["id"] for q in rec["questions"]}
        for n, x in enumerate(items):
            if x["question_id"] in done:
                continue
            cell = await L.run_cell(k, to_world(x), a.readers, a.workers)
            if not cell.get("questions"):   # blocked or failed ingest; a failed ANSWER is kept with score None and excluded pairwise
                print(f"    {k} {x['question_id']} {cell['status']} {cell['error']}", flush=True)
                continue
            q = cell["questions"][0]
            q["store_tokens"] = cell["store_tokens"]
            rec["questions"].append(q)
            rec["ingest_usd"] += cell["ingest"]["usd"]
            for r, c in cell["read"]["by_reader"].items():
                agg = rec["read"].setdefault(r, {"llm_calls": 0, "prompt_tokens": 0, "usd": 0.0})
                for kk in agg:
                    agg[kk] += c[kk]
            if (n + 1) % 10 == 0 or n + 1 == len(items):
                f.write_text(json.dumps(rec, indent=1, default=str))
                acc = {r.split("/")[-1]: round(sum((q["by_reader"][r]["score"] or 0) >= 0.99 for q in rec["questions"]) / len(rec["questions"]), 3) for r in a.readers}
                print(f"=== {k} {len(rec['questions'])}/{len(items)} {acc}", flush=True)
        f.write_text(json.dumps(rec, indent=1, default=str))


if __name__ == "__main__":
    asyncio.run(main())
