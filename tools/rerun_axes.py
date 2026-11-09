#!/usr/bin/env python3
"""Re-run one or two axes (A3 deletion, A4 adversarial) for existing round-2 records on a FRESH ingest of the same
world, and patch the record (the A1/A2 answers and the ingest cost stay as recorded).

Why this exists: three round-2 defects were confined to these axes — Graphiti's post-delete queries errored (the
base-class rebuild path does not fit its lifecycle; it now deletes episodes natively), GraphRAG's incremental ingest
used an invalid CLI flag (every poison ingest failed), and the generator let poison probes target memories a deletion
case had removed (poisons regenerated, trusted memories/questions/deletions unchanged). Re-running only the affected
axes costs one ingest per cell instead of a full cell.

    SHIM_LOG=... OPENAI_BASE_URL=http://127.0.0.1:80xx/v1 TIER=sonnet \\
      python tools/rerun_axes.py --axes A4 --systems full_context bm25 ... --tier sonnet [--seeds 42 123 456]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:8008/v1")
os.environ.setdefault("OPENAI_API_KEY", "shim")

import run_sixaxis as rs  # noqa: E402
from src.systems.registry import load  # noqa: E402


async def rerun(path: Path, axes: list[str], batch: int) -> None:
    rec = json.loads(path.read_text())
    data = json.load(open(f"results/sixaxis_{rec['scale']}_seed{rec['seed']}.json"))
    sysm = load(rec["system"])()
    await sysm.reset()
    trusted = sorted([m for m in data["memories"] if m["kind"] != "poison"], key=lambda m: m["date"])
    mems = [rs.to_mem(m) for m in trusted]
    _, off = rs.ledger(0)
    t0 = time.time()
    step = len(mems) if getattr(sysm, "ingest_whole", False) else batch
    for i in range(0, len(mems), step):
        await sysm.ingest(mems[i:i + step])
    recs, off = rs.ledger(off)
    sess = rs.Session(sysm, off)
    note = rec.setdefault("reruns", [])
    if "A3" in axes:
        rec["deletions"] = await sess.a3(data)
    if "A4" in axes:
        rec["poisons"] = await sess.a4(data)
    try:
        await sysm.close()
    except Exception:  # noqa: BLE001
        pass
    note.append({"axes": axes, "reingest_seconds": round(time.time() - t0, 1), "reingest": rs.cost(recs), "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "why": "see tools/rerun_axes.py docstring"})
    rec["summary"] = rs.summarize(rec)
    path.write_text(json.dumps(rec, indent=1, default=str))
    s = rec["summary"]
    print(f"    {path.name}: A3={json.dumps(s['A3_deletion'])[:160]} A4={json.dumps(s['A4_adversarial'])}", flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/sixaxis")
    ap.add_argument("--axes", nargs="+", choices=["A3", "A4"], required=True)
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--tier", default=os.environ.get("TIER", "sonnet"))
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--batch", type=int, default=10)
    a = ap.parse_args()
    for f in sorted(Path(a.dir).resolve().glob("*_S*_seed*_rep*.json")):
        r = json.loads(f.read_text())
        if r.get("status") != "OK" or r["system"] not in a.systems or r.get("tier", "sonnet") != a.tier:
            continue
        if a.seeds and r["seed"] not in a.seeds:
            continue
        if any(set(x.get("axes", [])) == set(a.axes) for x in r.get("reruns", [])):   # resumable after a container restart
            print(f"=== rerun {a.axes} {f.name} skipped (already re-run)", flush=True)
            continue
        print(f"=== rerun {a.axes} {f.name}", flush=True)
        try:
            await rerun(f, a.axes, a.batch)
        except Exception as ex:  # noqa: BLE001
            print(f"    FAILED {type(ex).__name__}: {str(ex)[:300]}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
