#!/usr/bin/env python3
"""Round 4 — the scale ladder (experiments/prereg_round4.md).

One cell = (seed, rung, system). The system ingests the rung's store ONCE with the write-path model (the shim's default
model), then every question is answered by every READER through the shared answer prompt. Retrieval is reader-independent,
so a cell costs one ingest plus (questions x readers) answer calls. Answers are judged by strict-judge v3 pinned to Sonnet.

    python run_ladder.py --seed 42 --rung 120000 --systems bm25 minmem full_context --readers anthropic/claude-sonnet-5.5 ...

Records: results/round4/<system>_s<seed>_t<rung>.json — per question: the retrieved source ids, whether the gold evidence
was in the context, and per reader the answer, verdict, tokens and latency. Costs come from the shim ledger (dollars as
billed by the provider, also for calls replayed from the shim cache).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:8008/v1")
os.environ.setdefault("OPENAI_API_KEY", "shim")
for _k in ("HF_HUB_DISABLE_PROGRESS_BARS", "TQDM_DISABLE"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from src.architectures.base import Memory  # noqa: E402
from src.data.sixaxis_generator import generate_ladder  # noqa: E402
from src.evaluation.strict_judge import strict_judge_v3  # noqa: E402
from src.systems.common import LLM, READER  # noqa: E402
from src.systems.registry import BY_KEY, load  # noqa: E402

READERS = ["anthropic/claude-sonnet-5.5", "google/gemini-3.8-flash", "anthropic/claude-haiku-5.5", "openai/gpt-6-luna"]
ZERO_WRITE = {"full_context", "full_context_hdr", "bm25", "bm25_k20", "vector_rag", "vector_k20", "hybrid_rerank", "hybrid_k20",
              "minmem", "minmem_noexp", "minmem_nohdr", "minmem_order"}
THREAD_SAFE = ZERO_WRITE | {"mastra_om"}   # read path is index lookups + one HTTP call, no state change
NOTE_ID_SOURCES = ZERO_WRITE - {"full_context", "full_context_hdr"}   # systems whose retrieved items ARE source notes
MAX_ERROR_RATE = 0.02   # prereg: a cell with more failed answers than this is PARTIAL (re-run, never scored as zeros)
SHIM_LOG = Path(os.environ.get("SHIM_LOG", "results/round4/logs/shim.jsonl")).resolve()
ABSTAIN_MARKERS = ("don't have enough information", "do not have enough information", "not enough information", "no information", "cannot determine", "unknown")
_JUDGE = threading.local()


def judge_fn(prompt: str) -> dict:
    if not hasattr(_JUDGE, "llm"):
        _JUDGE.llm = LLM(caller="judge", model="sonnet")   # pinned, whatever the reader is
    r = _JUDGE.llm.json("You are a strict, careful answer judge. Reply with JSON only.", prompt)
    return r if isinstance(r, dict) else {"score": 0, "reasoning": str(r)}


def is_abstain(ans: str) -> bool:
    a = ans.lower()
    return any(k in a for k in ABSTAIN_MARKERS)


def score(q: dict, answer: str) -> dict:
    """Same task-aware scoring as run_sixaxis.score_question."""
    if answer.startswith("<error"):
        return {"score": None, "outcome": "error", "judge_mode": "error", "judge_reasoning": ""}   # excluded pairwise, never a zero
    out0 = {"ref_in_answer": None, "stale_in_answer": None}
    if q["task"] in ("T1", "T3", "T4"):   # judge-free secondary signal: is the reference value / the superseded value in the text?
        out0 = {"ref_in_answer": q["reference_answer"].lower() in answer.lower(),
                "stale_in_answer": bool(q.get("stale_answer")) and q["stale_answer"].lower() in answer.lower()}
    return {**out0, **_score(q, answer)}


def _score(q: dict, answer: str) -> dict:
    if q["task"] == "T2":
        hit = [n for n in q["nuggets"] if all(p.lower() in answer.lower() for p in n.replace(" at ", "|").replace(" in ", "|").split("|"))]
        s = len(hit) / max(1, len(q["nuggets"]))
        return {"score": s, "outcome": "correct" if s == 1 else ("partial" if s > 0 else "wrong"), "judge_mode": "nugget",
                "judge_reasoning": f"{len(hit)}/{len(q['nuggets'])} nuggets"}
    if q["task"] == "T5":
        ok = is_abstain(answer)
        return {"score": 1.0 if ok else 0.0, "outcome": "correct" if ok else "hallucinated", "judge_mode": "abstention", "judge_reasoning": ""}
    s, why, mode = strict_judge_v3(q["question"], answer, q["reference_answer"], judge_fn)
    out = {"score": s, "judge_mode": mode, "judge_reasoning": why[:300]}
    if q["task"] == "T4":
        stale = bool(q["stale_answer"]) and q["stale_answer"].lower() in answer.lower() and q["reference_answer"].lower() not in answer.lower()
        out["outcome"] = "current" if s >= 0.99 else ("stale" if stale else "other")
    else:
        out["outcome"] = "correct" if s >= 0.99 else "wrong"
    return out


def ledger_window(t0: float, t1: float) -> list[dict]:
    if not SHIM_LOG.exists():
        return []
    out = []
    for line in SHIM_LOG.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if t0 <= r.get("ts", 0) <= t1 and "prompt_tokens" in r:
            out.append(r)
    return out


def cost_of(recs: list[dict]) -> dict:
    return {"llm_calls": len(recs), "prompt_tokens": sum(r["prompt_tokens"] for r in recs), "completion_tokens": sum(r["completion_tokens"] for r in recs),
            "provider_cached_tokens": sum(r.get("provider_cached_tokens", 0) for r in recs),
            "usd": round(sum(r.get("cost_usd_uncached") or r.get("cost_usd") or 0 for r in recs), 4)}


def to_mem(m: dict) -> Memory:
    # entities/topics are the generator's gold metadata: passed for interface compatibility, read by no system under test
    return Memory(content=m["content"], entities=m["entities"], topics=m["topics"], timestamp=datetime.fromisoformat(m["date"]),
                  importance=3.0, source_id=m["id"])


def ask_sync(sysm, question: str, reader: str) -> dict:
    """One (question, reader) query in the calling thread, with its own event loop."""
    tok = READER.set(reader)
    t0 = time.time()
    try:
        for attempt in range(3):   # the shim already retries transport errors; this covers a failed call end to end
            try:
                r = asyncio.run(sysm.query(question))
                return {"answer": r.answer, "sources": [it.source for it in r.retrieved_context], "context_head": [it.content[:200] for it in r.retrieved_context[:25]],
                        "in_tok": r.input_tokens, "out_tok": r.output_tokens, "latency_s": round(time.time() - t0, 2)}
            except Exception as ex:  # noqa: BLE001
                err = f"<error: {type(ex).__name__}: {str(ex)[:300]}>"
                if "spend cap" in err:
                    break
                time.sleep(5 * (attempt + 1))
        return {"answer": err, "sources": [], "context_head": [], "in_tok": 0, "out_tok": 0, "latency_s": round(time.time() - t0, 2)}
    finally:
        READER.reset(tok)


async def ask_async(sysm, question: str, reader: str) -> dict:
    """Same, on the main event loop (systems whose client objects are bound to it: Graphiti, GraphRAG, Mem0, memory tool)."""
    tok = READER.set(reader)
    t0 = time.time()
    try:
        for attempt in range(3):
            try:
                r = await sysm.query(question)
                return {"answer": r.answer, "sources": [it.source for it in r.retrieved_context], "context_head": [it.content[:200] for it in r.retrieved_context[:25]],
                        "in_tok": r.input_tokens, "out_tok": r.output_tokens, "latency_s": round(time.time() - t0, 2)}
            except Exception as ex:  # noqa: BLE001
                err = f"<error: {type(ex).__name__}: {str(ex)[:300]}>"
                if "spend cap" in err:
                    break
                await asyncio.sleep(5 * (attempt + 1))
        return {"answer": err, "sources": [], "context_head": [], "in_tok": 0, "out_tok": 0, "latency_s": round(time.time() - t0, 2)}
    finally:
        READER.reset(tok)


async def run_cell(key: str, data: dict, readers: list[str], workers: int) -> dict:
    meta = data["meta"]
    rec = {"system": key, "label": BY_KEY[key].label, "seed": meta["seed"], "rung": meta["target_tokens"], "store_tokens": meta["store_tokens"],
           "n_memories": meta["n_memories"], "readers": readers, "write_model": os.environ.get("SHIM_DEFAULT_MODEL", "?"), "status": "OK", "error": None}
    try:
        sysm = load(key)()
        await sysm.reset()
    except Exception as ex:  # noqa: BLE001
        rec.update(status="BLOCKED", error=f"{type(ex).__name__}: {str(ex)[:300]}")
        return rec
    mems = [to_mem(m) for m in data["memories"]]   # already in date order
    t0 = time.time()
    try:
        step = len(mems) if (key in ZERO_WRITE or getattr(sysm, "ingest_whole", False)) else 10
        for i in range(0, len(mems), step):
            await sysm.ingest(mems[i:i + step])
            if step == 10 and i and i % 500 == 0:
                print(f"    ingest {i}/{len(mems)} {time.time() - t0:.0f}s", flush=True)
    except Exception as ex:  # noqa: BLE001
        rec.update(status="FAILED_INGEST", error=f"{type(ex).__name__}: {str(ex)[:300]}", trace=traceback.format_exc()[-1500:])
        return rec
    t1 = time.time()
    rec["ingest"] = {"seconds": round(t1 - t0, 1), **cost_of([r for r in ledger_window(t0, t1) if r.get("caller") != "judge"])}
    rec["ingest"]["llm_calls_per_memory"] = round(rec["ingest"]["llm_calls"] / max(1, len(mems)), 3)

    qs = data["questions"]
    pairs = [(q, r) for r in readers for q in qs]
    answers: dict[tuple[str, str], dict] = {}
    tq0 = time.time()
    if key in THREAD_SAFE:
        loop = asyncio.get_running_loop()
        # the first question of each reader goes alone, so a provider-side prompt cache of a long shared prefix is written once
        import contextvars
        with ThreadPoolExecutor(workers) as pool:
            for r in readers:
                answers[(qs[0]["id"], r)] = await loop.run_in_executor(pool, ask_sync, sysm, qs[0]["question"], r)
            futs = {(q["id"], r): loop.run_in_executor(pool, contextvars.copy_context().run, ask_sync, sysm, q["question"], r)
                    for q, r in pairs if (q["id"], r) not in answers}
            for k, f in futs.items():
                answers[k] = await f
    else:
        for q, r in pairs:
            answers[(q["id"], r)] = await ask_async(sysm, q["question"], r)
    tq1 = time.time()
    win = [x for x in ledger_window(tq0, tq1) if x.get("caller") != "judge"]
    rec["read"] = {"seconds": round(tq1 - tq0, 1), "by_reader": {r: cost_of([x for x in win if x.get("model") == r]) for r in readers},
                   "other_models": cost_of([x for x in win if x.get("model") not in readers])}
    try:
        await sysm.close()
    except Exception:  # noqa: BLE001
        pass

    # judging (threads; deterministic branches make no call)
    with ThreadPoolExecutor(workers) as pool:
        verdicts = dict(zip(answers, pool.map(lambda kv: score(next(q for q in qs if q["id"] == kv[0][0]), kv[1]["answer"]), answers.items())))
    core_ids = lambda q: set(q["evidence_ids"])  # noqa: E731
    rec["questions"] = []
    for q in qs:
        first = answers[(q["id"], readers[0])]
        src = set(first["sources"])
        rec["questions"].append({
            "id": q["id"], "task": q["task"], "stratum": q.get("stratum", ""), "question": q["question"], "reference": q["reference_answer"],
            "stale_answer": q.get("stale_answer"), "evidence_ids": q["evidence_ids"], "n_retrieved": len(first["sources"]),
            "evidence_in_context": (core_ids(q) <= src) if (q["evidence_ids"] and key in NOTE_ID_SOURCES) else None,
            "context_head": first["context_head"],
            "by_reader": {r: {**{k: v for k, v in answers[(q["id"], r)].items() if k not in ("sources", "context_head")}, **verdicts[(q["id"], r)]} for r in readers}})
    n_err = {r: sum(q["by_reader"][r]["outcome"] == "error" for q in rec["questions"]) for r in readers}
    rec["errors_by_reader"] = n_err
    if any(n / len(qs) > MAX_ERROR_RATE for n in n_err.values()):
        rec["status"] = "PARTIAL"
        rec["error"] = f"answers failed: {n_err}"
    rec["wall_seconds"] = round(time.time() - t0, 1)
    return rec


def headline(rec: dict) -> str:
    out = []
    for r in rec["readers"]:
        qs = rec["questions"]
        ok = [q for q in qs if q["by_reader"][r]["score"] is not None]
        acc = sum(q["by_reader"][r]["score"] >= 0.99 if q["task"] != "T2" else q["by_reader"][r]["score"] for q in ok) / max(1, len(ok))
        t4 = [q for q in qs if q["task"] == "T4"]
        fama = (sum(q["by_reader"][r]["outcome"] == "current" for q in t4) - sum(q["by_reader"][r]["outcome"] == "stale" for q in t4)) / max(1, len(t4))
        err = sum(q["by_reader"][r]["outcome"] == "error" for q in qs)
        out.append(f"{r.split('/')[-1]}: acc {acc:.2f} FAMA {fama:.2f}" + (f" ERR {err}" if err else ""))
    return " | ".join(out)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rung", type=int, default=0, help="target store tokens (0 = core world only)")
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--readers", nargs="+", default=READERS)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N questions only")
    ap.add_argument("--out", default="results/round4")
    a = ap.parse_args()
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    data = generate_ladder(a.seed, a.rung)
    if a.limit:
        data["questions"] = [q for t in ("T4", "T1", "T2", "T3", "T5") for q in data["questions"] if q["task"] == t][: a.limit]
    for k in a.systems:
        f = out / f"{k}_s{a.seed}_t{a.rung}.json"
        if f.exists() and json.loads(f.read_text()).get("status") == "OK" and not a.limit:
            print(f"=== {k} s{a.seed} t{a.rung} skipped (OK record exists)", flush=True)
            continue
        print(f"=== {k} s{a.seed} t{a.rung} ({data['meta']['n_memories']} notes, ~{data['meta']['store_tokens']} tokens)", flush=True)
        rec = await run_cell(k, data, a.readers, a.workers)
        f.write_text(json.dumps(rec, indent=1, default=str))
        print(f"    {rec['status']} {headline(rec) if rec.get('questions') else rec['error']}" +
              (f" | ingest {rec['ingest']['llm_calls']} calls ${rec['ingest']['usd']}" if rec.get("ingest") else ""), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
