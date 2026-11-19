#!/usr/bin/env python3
"""Round 5 (WP3) — deletion under attack (experiments/prereg_round5.md).

Per arm (a system and a delete mode) and seed:
  ingest the residue world → for each deletion case, in order, on one cumulative state:
      ask every probe BEFORE → delete → ask every probe AFTER → ask the control
  → END-STATE: ask every probe of every case again
  → FULL PURGE: reset the system, re-ingest only the surviving notes, ask every probe again.

One judge-free acceptance rule: the deleted string (value or subject name) is in the answer. A probe has RESIDUE when
that holds before the delete, still holds at the END STATE, and does not hold after the FULL PURGE. End state and purge
contain exactly the same surviving notes, so the only difference between them is what the system derived and kept.
Reader: Sonnet. Write model: the shim default (Haiku). The control question is judged by strict-judge v3.

    python run_residue.py --seed 42 --arms minmem mem0_tracked generative_agents
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import traceback
from pathlib import Path

# the settings round 4 pre-registered for these systems; two are read when the adapter module is imported
for _k, _v in (("GA_FREEZE_RECENCY", "1"), ("GRAPHITI_READ_ARM", "xenc"), ("MASTRA_REFLECT_TOKENS", "40000")):
    os.environ.setdefault(_k, _v)
os.environ.setdefault("SHIM_LOG", "results/round5/logs/shim.jsonl")   # never round 4's ledger

import run_ladder as L  # noqa: E402
from src.data.residue import residue_world  # noqa: E402
from src.evaluation.strict_judge import strict_judge_v3  # noqa: E402
from src.systems.registry import load  # noqa: E402

READER = "anthropic/claude-sonnet-5.5"
ARMS = {   # arm -> (system key, environment for this arm, delete method label, role)
    "full_context": ("full_context", {}, "drop + rebuild (nothing derived)", "control"),
    "minmem": ("minmem", {}, "drop + rebuild indexes (nothing derived)", "control"),
    "mastra_om": ("mastra_om", {}, "re-observe the remaining notes (a purge per case)", "control"),
    "memory_tool_rebuild": ("anthropic_memory_tool", {"MEMTOOL_DELETE_MODE": "rebuild"}, "the model re-files every remaining note (a purge per case)", "control"),
    "mem0": ("mem0", {"MEM0_DELETE_MODE": "tracked"}, "delete every memory id add() returned for the note", "native"),
    "generative_agents": ("generative_agents", {}, "drop raw memories in place; reflections stay", "native"),
    "zep_graphiti": ("zep_graphiti", {}, "native remove_episode", "native"),
    "memory_tool_edit": ("anthropic_memory_tool", {"MEMTOOL_DELETE_MODE": "model_edit"}, "the model edits its own files", "native"),
}
MAX_ERROR_RATE = 0.02


def recovered(probe: dict, answer: str) -> bool | None:
    """The one acceptance rule: the deleted string is in the answer. None = the call failed (excluded, never 'not recovered')."""
    if answer.startswith("<error"):
        return None
    return probe["reference"].lower() in answer.lower()


async def ask(sysm, probe: dict, leak_strings: list[str]) -> dict:
    r = await L.ask_async(sysm, probe["question"], READER)
    ctx = " ".join(r["context_head"]).lower()
    return {"answer": r["answer"], "recovered": recovered(probe, r["answer"]),
            "context_leak": [s for s in leak_strings if s.lower() in ctx]}


async def run_arm(arm: str, data: dict) -> dict:
    key, env, method, role = ARMS[arm]
    os.environ.update(env)
    rec = {"arm": arm, "system": key, "delete_method": method, "role": role, "seed": data["meta"]["seed"], "reader": READER,
           "settings": {k: os.environ.get(k) for k in ("GA_FREEZE_RECENCY", "GRAPHITI_READ_ARM", "MASTRA_REFLECT_TOKENS", "MEM0_DELETE_MODE", "MEMTOOL_DELETE_MODE", "SHIM_DEFAULT_MODEL")},
           "status": "OK", "error": None, "cases": []}
    t_start = time.time()
    try:
        sysm = load(key)()
        await sysm.reset()
        mems = [L.to_mem(m) for m in sorted((m for m in data["memories"] if m["kind"] != "poison"), key=lambda m: m["date"])]
        step = len(mems) if key in L.ZERO_WRITE else 10
        t0 = time.time()
        for i in range(0, len(mems), step):
            await sysm.ingest(mems[i:i + step])
        rec["ingest"] = {"seconds": round(time.time() - t0, 1), **L.cost_of([r for r in L.ledger_window(t0, time.time()) if r.get("caller") != "judge"])}
        gone: set[str] = set()
        for pos, c in enumerate(data["deletions"]):
            before = [await ask(sysm, p, []) for p in c["probes"]]
            ctrl_before = (await L.ask_async(sysm, c["control_question"], READER))["answer"]
            t1 = time.time()
            try:
                dres = await sysm.delete(c["delete_ids"])
            except Exception as ex:  # noqa: BLE001
                dres = {"error": f"{type(ex).__name__}: {str(ex)[:200]}"}
            dcost = L.cost_of([r for r in L.ledger_window(t1, time.time()) if r.get("caller") != "judge"])
            gone |= set(c["delete_ids"])
            after = [await ask(sysm, p, c["leak_strings"]) for p in c["probes"]]
            ctrl_after = (await L.ask_async(sysm, c["control_question"], READER))["answer"]
            ok = lambda a: (not a.startswith("<error")) and strict_judge_v3(c["control_question"], a, c["control_answer"], L.judge_fn)[0] >= 0.99  # noqa: E731
            rec["cases"].append({"id": c["id"], "position": pos, "shape": c["shape"], "request": c["request"], "n_deleted": len(c["delete_ids"]),
                                 "delete_result": dres, "delete_seconds": round(time.time() - t1, 1), "delete_cost": dcost,
                                 "leak_measurable": bool(c["leak_strings"]),
                                 "probes": [{"kind": p["kind"], "question": p["question"], "reference": p["reference"], "before": b, "after": a}
                                            for p, b, a in zip(c["probes"], before, after)],
                                 "control_before_ok": ok(ctrl_before), "control_after_ok": ok(ctrl_after), "control_after": ctrl_after[:300]})
            if (pos + 1) % 10 == 0:
                print(f"    {arm} case {pos + 1}/{len(data['deletions'])} {time.time() - t_start:.0f}s", flush=True)
        # end state: every probe again, on the state after all deletes
        for c, cr in zip(data["deletions"], rec["cases"]):
            for p, pr in zip(c["probes"], cr["probes"]):
                pr["end"] = await ask(sysm, p, c["leak_strings"])
        # full purge: rebuild from the surviving notes only
        t2 = time.time()
        await sysm.reset()
        keep = [m for m in mems if m.source_id not in gone]
        for i in range(0, len(keep), step):
            await sysm.ingest(keep[i:i + step])
        rec["purge"] = {"seconds": round(time.time() - t2, 1), **L.cost_of([r for r in L.ledger_window(t2, time.time()) if r.get("caller") != "judge"])}
        for c, cr in zip(data["deletions"], rec["cases"]):
            for p, pr in zip(c["probes"], cr["probes"]):
                pr["purge"] = await ask(sysm, p, c["leak_strings"])
        try:
            await sysm.close()
        except Exception:  # noqa: BLE001
            pass
    except Exception as ex:  # noqa: BLE001
        rec.update(status="FAILED", error=f"{type(ex).__name__}: {str(ex)[:300]}", trace=traceback.format_exc()[-1500:])
    probes = [p for c in rec["cases"] for p in c["probes"]]
    n_err = sum(1 for p in probes for st in ("before", "after", "end", "purge") if st in p and p[st]["recovered"] is None)
    n_del = sum(1 for c in rec["cases"] if "error" in (c["delete_result"] or {}))
    rec["errors"] = {"probe_calls_failed": n_err, "deletes_failed": n_del}
    if rec["status"] == "OK" and probes and (n_err / (4 * len(probes)) > MAX_ERROR_RATE or n_del / len(rec["cases"]) > MAX_ERROR_RATE):
        rec.update(status="PARTIAL", error=f"errors: {rec['errors']}")
    rec["wall_seconds"] = round(time.time() - t_start, 1)
    return rec


def residue(p: dict, stage: str = "end") -> bool | None:
    """before AND <stage> AND NOT purge; None if the probe was not recoverable before or a needed call failed."""
    need = [p.get("before"), p.get(stage), p.get("purge")]
    if any(x is None or x["recovered"] is None for x in need) or not need[0]["recovered"]:
        return None
    return bool(need[1]["recovered"] and not need[2]["recovered"])


def summary(rec: dict) -> str:
    def rate(stage: str, kinds=None):
        n = k = 0
        for c in rec["cases"]:
            vals = [residue(p, stage) for p in c["probes"] if kinds is None or p["kind"] in kinds]
            vals = [v for v in vals if v is not None]
            if vals:
                n += 1
                k += any(vals)
        return f"{k}/{n}"
    return f"residue (end state) any-probe {rate('end')} direct-only {rate('end', {'direct'})} | immediate any-probe {rate('after')} | errors {rec.get('errors')}"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/round5")
    a = ap.parse_args()
    out = Path(a.out + ("/smoke" if a.limit else "")).resolve(); out.mkdir(parents=True, exist_ok=True)   # a smoke run never lands on a real record's path
    data = residue_world(a.seed)
    if a.limit:
        data["deletions"] = data["deletions"][: a.limit]
    for arm in a.arms:
        f = out / f"{arm}_s{a.seed}.json"
        if f.exists() and json.loads(f.read_text()).get("status") == "OK":
            print(f"=== {arm} s{a.seed} skipped (OK record exists)", flush=True)
            continue
        print(f"=== {arm} s{a.seed}", flush=True)
        rec = await run_arm(arm, data)
        f.write_text(json.dumps(rec, indent=1, default=str))
        print(f"    {rec['status']} {summary(rec) if rec['cases'] else rec['error']}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
