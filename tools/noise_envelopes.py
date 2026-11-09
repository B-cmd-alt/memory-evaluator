#!/usr/bin/env python3
"""Measure the two noise envelopes the pre-registration fixes (experiments/prereg_sixaxis.md, H3) ONCE, on the
baselines, so every round-2 ranking can print them and treat differences inside them as ties:

  judge envelope   — the same baseline answers re-scored by a SOFT LLM judge (the standard symmetric
                     "is the answer correct?" prompt, no abstention asymmetry) vs strict-judge v3:
                     |acc_soft − acc_strict| per baseline, max over baselines.
  prompt envelope  — the same baseline systems re-asked with two alternative answer prompts (terse, verbose),
                     scored by strict v3: max − min accuracy over the three prompts, max over baselines.

Only the zero-write-cost baselines are re-run (their ingest makes no model call). Writes
results/sixaxis/noise_envelopes.json, which tools/aggregate_sixaxis.py folds into the envelope it uses.

    SHIM_LOG=results/sixaxis/logs/shim_noise.jsonl OPENAI_BASE_URL=http://127.0.0.1:8014/v1 \\
      python tools/noise_envelopes.py --tier sonnet [--systems full_context bm25 vector_rag] [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
from pathlib import Path

os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:8008/v1")
os.environ.setdefault("OPENAI_API_KEY", "shim")

from src.systems import common  # noqa: E402
from src.systems.common import LLM  # noqa: E402
from src.systems.registry import load  # noqa: E402
import run_sixaxis as rs  # noqa: E402  (judge_fn, score_question, to_mem, JUDGE pinned to Sonnet)

SOFT_JUDGE_SYSTEM = "You are an impartial grader. Reply with JSON only."
SOFT_JUDGE_PROMPT = """Question: {question}
Reference answer: {reference}
Model answer: {answer}

Is the model answer correct with respect to the reference? Reply as JSON: {{"score": 1 if correct, 0.5 if partially correct, 0 if incorrect, "reasoning": "<one sentence>"}}"""

PROMPTS = {
    "P1_shared": common.GENERATION_PROMPT,
    "P2_terse": """Context:
{context}

Q: {question}
A (one short sentence; if the context lacks the answer, reply exactly "I don't have enough information to answer this question."):""",
    "P3_verbose": """You are a careful assistant answering a question using only the memory context below. Read every entry, note the
dates, prefer the most recent statement when entries conflict, and quote the specific fact you rely on. If the context
does not contain enough information, say "I don't have enough information to answer this question." Do not guess.

Memory context:
{context}

Question: {question}

Answer:""",
}
LLM_TASKS = ("T1", "T3", "T4")   # the tasks strict v3 scores with an LLM residue; T2 (nuggets) and T5 (abstention) are deterministic


def soft_judge(question: str, answer: str, reference: str) -> float:
    r = rs.JUDGE.json(SOFT_JUDGE_SYSTEM, SOFT_JUDGE_PROMPT.format(question=question, reference=reference, answer=answer))
    try:
        return float(r.get("score", 0))
    except (TypeError, ValueError, AttributeError):
        return 0.0


def judge_envelope(recs: list[dict], limit: int | None) -> dict:
    out = {}
    for r in recs:
        qs = [q for q in r["questions"] if q["task"] in LLM_TASKS][:limit]
        if not qs:
            continue
        strict = statistics.mean(q["score"] for q in qs)
        soft = statistics.mean(soft_judge(q["question"], q["answer"], q["reference"]) for q in qs)
        out.setdefault(r["system"], []).append({"seed": r["seed"], "n": len(qs), "strict": round(strict, 3), "soft": round(soft, 3), "spread": round(abs(soft - strict), 3)})
    per_sys = {s: round(statistics.mean(x["spread"] for x in v), 3) for s, v in out.items()}
    return {"per_run": out, "per_system_mean_spread": per_sys, "envelope": max(per_sys.values()) if per_sys else None}


async def prompt_envelope(datasets: dict[int, dict], recs: list[dict], systems: list[str], limit: int | None) -> dict:
    out = {}
    for sysk in systems:
        for seed, data in datasets.items():
            base = next((r for r in recs if r["system"] == sysk and r["seed"] == seed), None)
            qs = data["questions"][:limit]
            accs = {}
            if base:
                bq = {q["id"]: q for q in base["questions"]}
                accs["P1_shared"] = round(statistics.mean(bq[q["id"]]["score"] for q in qs if q["id"] in bq), 3)
            s = load(sysk)()
            await s.reset()
            mems = [rs.to_mem(m) for m in sorted([m for m in data["memories"] if m["kind"] != "poison"], key=lambda m: m["date"])]
            await s.ingest(mems)
            for pname, ptext in PROMPTS.items():
                if pname in accs:
                    continue
                common.GENERATION_PROMPT = ptext
                scores = []
                for q in qs:
                    try:
                        ans = (await s.query(q["question"])).answer
                    except Exception as ex:  # noqa: BLE001
                        ans = f"<error: {type(ex).__name__}>"
                    scores.append(rs.score_question(q, ans)["score"])
                accs[pname] = round(statistics.mean(scores), 3)
            common.GENERATION_PROMPT = PROMPTS["P1_shared"]
            try:
                await s.close()
            except Exception:  # noqa: BLE001
                pass
            out.setdefault(sysk, []).append({"seed": seed, "n": len(qs), "acc_by_prompt": accs, "spread": round(max(accs.values()) - min(accs.values()), 3)})
            print(f"  [prompt] {sysk} seed {seed}: {accs}", flush=True)
    per_sys = {s: round(statistics.mean(x["spread"] for x in v), 3) for s, v in out.items()}
    return {"prompts": {k: v[:160] for k, v in PROMPTS.items()}, "per_run": out, "per_system_mean_spread": per_sys,
            "envelope": max(per_sys.values()) if per_sys else None}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/sixaxis")
    ap.add_argument("--tier", default="sonnet")
    ap.add_argument("--systems", nargs="*", default=["full_context", "bm25", "vector_rag"])
    ap.add_argument("--limit", type=int, default=None, help="questions per run (smoke tests)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Path(a.dir)
    recs = []
    for f in sorted(d.glob("*_S1_seed*_rep*.json")):
        r = json.loads(f.read_text())
        if r.get("status") == "OK" and r.get("tier", "sonnet") == a.tier and r["system"] in a.systems:
            recs.append(r)
    seeds = sorted({r["seed"] for r in recs}) or [42]
    datasets = {s: json.load(open(f"results/sixaxis_S1_seed{s}.json")) for s in seeds}
    print(f"[noise] tier={a.tier} baselines={a.systems} seeds={seeds} records={len(recs)}", flush=True)
    je = judge_envelope(recs, a.limit)
    print(f"  [judge] per-system spread {je['per_system_mean_spread']} → envelope {je['envelope']}", flush=True)
    pe = await prompt_envelope(datasets, recs, a.systems, a.limit)
    print(f"  [prompt] per-system spread {pe['per_system_mean_spread']} → envelope {pe['envelope']}", flush=True)
    res = {"tier": a.tier, "scale": "S1", "systems": a.systems, "seeds": seeds, "limit": a.limit, "judge": je, "prompt": pe,
           "envelope": max(v for v in (je["envelope"], pe["envelope"]) if v is not None) if (je["envelope"] is not None or pe["envelope"] is not None) else None}
    out = Path(a.out or (d / "noise_envelopes.json"))
    out.write_text(json.dumps(res, indent=1))
    print(f"[noise] → {out}: envelope {res['envelope']}")


if __name__ == "__main__":
    asyncio.run(main())
