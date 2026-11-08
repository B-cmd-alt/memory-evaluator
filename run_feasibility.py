#!/usr/bin/env python3
"""Feasibility round: can each registered system ingest S1 and answer its questions, and at what cost?

For every system: reset → ingest the S1 haystack in dated batches ("sessions", incremental) →
answer every question → strict-judge v3 → write per-question records + a summary. Write-path and
read-path LLM calls are read back from the shim log by time window (exact, since systems run
one at a time). A system that cannot be built is recorded as BLOCKED with the reason, never skipped
silently.

    python run_feasibility.py --systems bm25 full_context vector_rag --limit 5
    python run_feasibility.py --all
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
import traceback
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:8008/v1")
os.environ.setdefault("OPENAI_API_KEY", "shim")

from src.architectures.base import Memory  # noqa: E402
from src.evaluation.strict_judge import strict_judge_v3  # noqa: E402
from src.systems.common import LLM  # noqa: E402
from src.systems.registry import REGISTRY, load  # noqa: E402

OUT = Path("results/feasibility")
SHIM_LOG = Path(os.environ.get("SHIM_LOG", "results/feasibility/shim_calls.jsonl"))
JUDGE = LLM(caller="judge", model="gpt-4o")  # sonnet via shim alias; the paper used opus — noted in the report


def load_s1(limit: int | None):
    d = json.load(open("results/s1_dataset.json"))
    mems = [Memory(content=m["content"], entities=[], topics=m.get("topics", []), timestamp=datetime.fromisoformat(m["date"]),
                   importance=3.0, source_id=m["id"]) for m in d["memories"]]
    mems.sort(key=lambda m: m.timestamp)
    qs = d["questions"][:limit] if limit else d["questions"]
    return mems, qs


def shim_calls_since(offset: int) -> tuple[list[dict], int]:
    if not SHIM_LOG.exists():
        return [], 0
    lines = SHIM_LOG.read_text().splitlines()
    recs = []
    for l in lines[offset:]:
        try:
            recs.append(json.loads(l))
        except ValueError:
            pass
    return recs, len(lines)


def judge_fn(prompt: str) -> dict:
    r = JUDGE.json("You are a strict, careful answer judge. Reply with JSON only.", prompt)
    return r if isinstance(r, dict) else {"score": 0, "reasoning": str(r)}


def summarize_calls(recs: list[dict]) -> dict:
    chat = [r for r in recs if "prompt_tokens" in r]
    return {"llm_calls": len(chat), "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in chat),
            "completion_tokens": sum(r.get("completion_tokens", 0) for r in chat),
            "cost_usd": round(sum(r.get("cost_usd") or 0 for r in chat), 4),
            "embed_calls": sum(1 for r in recs if "embed" in r), "errors": sum(1 for r in recs if "error" in r)}


async def run_system(key: str, mems, qs, batch: int) -> dict:
    e = REGISTRY and next(x for x in REGISTRY if x.key == key)
    rec = {"system": key, "label": e.label, "family": e.family, "year": e.year, "kind": e.kind, "provenance": e.provenance,
           "status": "OK", "error": None, "n_memories": len(mems), "n_questions": len(qs), "questions": []}
    t_start = time.time()
    try:
        sysm = load(key)()
        await sysm.reset()
    except Exception as ex:  # noqa: BLE001
        rec.update(status="BLOCKED", error=f"{type(ex).__name__}: {str(ex)[:300]}")
        return rec
    # ── ingest, incremental ───────────────────────────────────────────────────────────────────
    _, off0 = shim_calls_since(0)
    t0 = time.time()
    try:
        step = len(mems) if getattr(sysm, "ingest_whole", False) else batch
        for i in range(0, len(mems), step):
            await sysm.ingest(mems[i:i + step])
    except Exception as ex:  # noqa: BLE001
        rec.update(status="FAILED_INGEST", error=f"{type(ex).__name__}: {str(ex)[:300]}", trace=traceback.format_exc()[-1200:])
        return rec
    ingest_s = time.time() - t0
    ingest_calls, off1 = shim_calls_since(off0)
    rec["ingest"] = {"seconds": round(ingest_s, 1), **summarize_calls(ingest_calls)}
    rec["ingest"]["llm_calls_per_memory"] = round(rec["ingest"]["llm_calls"] / max(1, len(mems)), 2)
    # ── query + judge ─────────────────────────────────────────────────────────────────────────
    errors = 0
    for q in qs:
        t1 = time.time()
        _, qoff = shim_calls_since(0)
        try:
            r = await sysm.query(q["question"])
            answer, items, tier = r.answer, r.retrieved_context, r.retrieval_tier
        except Exception as ex:  # noqa: BLE001
            errors += 1
            answer, items, tier = f"<error: {type(ex).__name__}: {str(ex)[:200]}>", [], "error"
        latency = time.time() - t1
        qcalls, _ = shim_calls_since(qoff)
        try:
            score, reasoning, mode = strict_judge_v3(q["question"], answer, q["reference_answer"], judge_fn)
        except Exception as ex:  # noqa: BLE001
            score, reasoning, mode = 0.0, f"judge error: {ex}", "error"
        rec["questions"].append({
            "id": q["id"], "task": q["task"], "question": q["question"], "reference": q["reference_answer"],
            "answer": answer, "retrieved": [{"content": it.content[:500], "source": it.source, "score": it.score} for it in items],
            "tier": tier, "latency_s": round(latency, 2), "read_path": summarize_calls([c for c in qcalls if c.get("caller") != "judge"]),
            "score": score, "judge_mode": mode, "judge_reasoning": reasoning[:600],
        })
    scores = [x["score"] for x in rec["questions"]]
    lats = [x["latency_s"] for x in rec["questions"]]
    rec["query"] = {"accuracy": round(statistics.mean(scores), 3) if scores else None,
                    "by_task": {t: round(statistics.mean([x["score"] for x in rec["questions"] if x["task"] == t]), 2)
                                for t in sorted({x["task"] for x in rec["questions"]})},
                    "latency_p50_s": round(statistics.median(lats), 2) if lats else None,
                    "latency_p95_s": round(sorted(lats)[int(0.95 * (len(lats) - 1))], 2) if lats else None,
                    "read_llm_calls_per_query": round(sum(x["read_path"]["llm_calls"] for x in rec["questions"]) / max(1, len(qs)), 2),
                    "read_prompt_tokens_per_query": round(sum(x["read_path"]["prompt_tokens"] for x in rec["questions"]) / max(1, len(qs))),
                    "query_errors": errors}
    if errors:
        rec["status"] = "PARTIAL"
    try:
        await sysm.close()
    except Exception:  # noqa: BLE001
        pass
    rec["wall_seconds"] = round(time.time() - t_start, 1)
    return rec


def render(rows: list[dict]) -> str:
    L = ["# Feasibility round — S1 (90 memories, 25 questions)", "",
         "Status: OK = ran end to end · PARTIAL = query errors · FAILED_INGEST · BLOCKED = could not be built (reason recorded).",
         "Judge: strict-judge v3 via the shim (sonnet alias). Costs are measured from the shim log, not estimated.", "",
         "| system | kind | provenance | status | accuracy | T1 | T2 | T3 | T4 | T5 | ingest LLM calls/mem | ingest s | read calls/q | read prompt tok/q | p50 s | p95 s | $ ingest | $ queries |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["status"] in ("BLOCKED", "FAILED_INGEST"):
            L.append(f"| {r['label']} | {r['kind']} | {r['provenance']} | **{r['status']}** | — | | | | | | | | | | | | | |  <sub>{r['error']}</sub>")
            continue
        q, ing = r["query"], r["ingest"]
        bt = q["by_task"]
        qcost = round(sum(x["read_path"]["cost_usd"] for x in r["questions"]), 3)
        L.append(f"| {r['label']} | {r['kind']} | {r['provenance']} | {r['status']} | **{q['accuracy']}** | "
                 + " | ".join(str(bt.get(t, '')) for t in ("T1", "T2", "T3", "T4", "T5")) +
                 f" | {ing['llm_calls_per_memory']} | {ing['seconds']} | {q['read_llm_calls_per_query']} | {q['read_prompt_tokens_per_query']} | "
                 f"{q['latency_p50_s']} | {q['latency_p95_s']} | {ing['cost_usd']} | {qcost} |")
    L += ["", "Accuracy at S1 has a ~7pp inter-run noise envelope (RESULTS_SUMMARY §1); differences inside it are ties. "
          "This round decides feasibility and cost class, not ranking.", ""]
    return "\n".join(L)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="questions per system")
    ap.add_argument("--batch", type=int, default=10, help="memories per ingest session")
    ap.add_argument("--memories", type=int, default=None, help="cap haystack size (smoke)")
    a = ap.parse_args()
    keys = [e.key for e in REGISTRY] if a.all else (a.systems or ["bm25", "full_context"])
    mems, qs = load_s1(a.limit)
    if a.memories:
        mems = mems[: a.memories]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in keys:
        print(f"=== {k}", flush=True)
        r = await run_system(k, mems, qs, a.batch)
        (OUT / f"{k}.json").write_text(json.dumps(r, indent=1, default=str))
        rows.append(r)
        print(f"    {r['status']} acc={r.get('query', {}).get('accuracy')} ingest_calls/mem={r.get('ingest', {}).get('llm_calls_per_memory')} err={r.get('error')}", flush=True)
    existing = {}
    for p in OUT.glob("*.json"):
        if p.name not in ("shim_calls.jsonl",):
            try:
                d = json.loads(p.read_text())
                if "system" in d:
                    existing[d["system"]] = d
            except ValueError:
                pass
    order = [e.key for e in REGISTRY]
    (OUT / "FEASIBILITY.md").write_text(render([existing[k] for k in order if k in existing]))
    print(open(OUT / "FEASIBILITY.md").read())


if __name__ == "__main__":
    asyncio.run(main())
