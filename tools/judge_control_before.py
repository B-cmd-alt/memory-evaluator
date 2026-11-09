#!/usr/bin/env python3
"""Post-hoc: judge each deletion case's `control_before` answer and store `control_before_ok` in the record, then
recompute the record's summary. A control question only measures collateral damage of a delete if the system
could answer it BEFORE the delete (Generative Agents fails most controls before any delete — that is a recall
failure, not deletion damage). run_sixaxis.py records the field itself from this change on; this fills records
written earlier. Idempotent: records that already carry the field are skipped.

    SHIM_LOG=results/sixaxis/logs/shim_noise.jsonl OPENAI_BASE_URL=http://127.0.0.1:8014/v1 python tools/judge_control_before.py
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:8008/v1")
os.environ.setdefault("OPENAI_API_KEY", "shim")

import run_sixaxis as rs  # noqa: E402
from src.evaluation.strict_judge import strict_judge_v3  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/sixaxis")
    a = ap.parse_args()
    n_rec = n_cases = 0
    for f in sorted(Path(a.dir).glob("*_S*_seed*_rep*.json")):
        r = json.loads(f.read_text())
        if r.get("status") != "OK" or all("control_before_ok" in d for d in r["deletions"]):
            continue
        data = json.load(open(f"results/sixaxis_{r['scale']}_seed{r['seed']}.json"))
        cases = {c["id"]: c for c in data["deletions"]}
        for d in r["deletions"]:
            if "control_before_ok" in d:
                continue
            c = cases[d["id"]]
            d["control_before_ok"] = strict_judge_v3(c["control_question"], d["control_before"], c["control_answer"], rs.judge_fn)[0] >= 0.99
            n_cases += 1
        r["summary"] = rs.summarize(r)
        f.write_text(json.dumps(r, indent=1, default=str))
        n_rec += 1
        a3 = r["summary"]["A3_deletion"]
        print(f"{f.name}: control_intact over valid controls = { {k: (v['control_intact'], v['n_valid_controls']) for k, v in a3.items()} }", flush=True)
    print(f"filled {n_cases} cases in {n_rec} records")


if __name__ == "__main__":
    main()
