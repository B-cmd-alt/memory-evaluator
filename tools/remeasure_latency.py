#!/usr/bin/env python3
"""Re-measure query latency on a QUIET machine for systems whose read path is CPU-bound.

Round 2 ran six runners concurrently on a 4-core box (load average ~20); LLM-bound latencies were unaffected
(the shim waits on the model) but the hybrid reranker's cross-encoder pass went from ~2 s to ~100 s per query
under that contention. Accuracy and cost are unaffected (same retrieval, same answers); only the latency
column is. This script re-asks the same questions against a freshly ingested copy of the system with nothing
else running and stores `latency_quiet` {p50_s, p95_s, n, load_avg} next to the original numbers; the
aggregator prefers it where present and labels the column. The original under-load values stay in the record.

    SHIM_LOG=results/sixaxis/logs/shim_noise.jsonl OPENAI_BASE_URL=http://127.0.0.1:8014/v1 \\
      python tools/remeasure_latency.py --systems hybrid_rerank [--tier sonnet haiku] [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:8008/v1")
os.environ.setdefault("OPENAI_API_KEY", "shim")

import run_sixaxis as rs  # noqa: E402
from src.systems.registry import load  # noqa: E402


async def remeasure(path: Path, limit: int | None) -> dict:
    rec = json.loads(path.read_text())
    data = json.load(open(f"results/sixaxis_{rec['scale']}_seed{rec['seed']}.json"))
    s = load(rec["system"])()
    await s.reset()
    mems = [rs.to_mem(m) for m in sorted([m for m in data["memories"] if m["kind"] != "poison"], key=lambda m: m["date"])]
    await s.ingest(mems)
    lat = []
    for q in rec["questions"][:limit]:
        t0 = time.time()
        try:
            await s.query(q["question"])
        except Exception:  # noqa: BLE001
            continue
        lat.append(round(time.time() - t0, 2))
    try:
        await s.close()
    except Exception:  # noqa: BLE001
        pass
    lat.sort()
    out = {"p50_s": round(statistics.median(lat), 2), "p95_s": round(lat[int(0.95 * (len(lat) - 1))], 2), "n": len(lat),
           "load_avg_1m": round(os.getloadavg()[0], 2), "note": "re-measured on a quiet machine; the original p50 was taken under ~20 load avg"}
    rec["latency_quiet"] = out
    path.write_text(json.dumps(rec, indent=1, default=str))
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/sixaxis")
    ap.add_argument("--systems", nargs="+", default=["hybrid_rerank"])
    ap.add_argument("--tier", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    for f in sorted(Path(a.dir).glob("*_S*_seed*_rep*.json")):
        r = json.loads(f.read_text())
        if r.get("status") != "OK" or r["system"] not in a.systems or (a.tier and r.get("tier") not in a.tier):
            continue
        out = await remeasure(f, a.limit)
        print(f"{f.name}: p50 {r['summary']['A5_cost']['latency_p50_s']} s under load → {out['p50_s']} s quiet (load {out['load_avg_1m']})", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
