#!/usr/bin/env python3
"""Post-hoc: re-score the A3 (deletion) cases of existing records against the CURRENT dataset definition.

Two scoring fixes landed during round 2 (both are definition fixes; no answer is re-generated):
  * leak strings are now unique to the deleted memories (a venue city shared by other notes is not a leak); cases
    whose list became empty are `leak_measurable = False` and excluded from the leak rate;
  * `forgot` is the strict judge rejecting the probe's pre-delete answer (`probe_answer`), not a substring test.
The served artifacts (probe/control answers, retrieved context) are in the record, so only judge calls are needed.
Idempotent: a case already carrying `leak_measurable` is skipped.

    SHIM_LOG=results/sixaxis/logs/shim_noise.jsonl OPENAI_BASE_URL=http://127.0.0.1:8014/v1 python tools/rescore_deletions.py [--dir ...]
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

A3_VERSION = 2   # 1: unique leak strings + judged forgot; 2: a question-echoed name counts only via retrieved context


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/sixaxis")
    a = ap.parse_args()
    n_rec = n_cases = 0
    for f in sorted(Path(a.dir).glob("*_S*_seed*_rep*.json")):
        r = json.loads(f.read_text())
        if r.get("status") != "OK" or all(d.get("a3_score_version") == A3_VERSION for d in r["deletions"]):
            continue
        data = json.load(open(f"results/sixaxis_{r['scale']}_seed{r['seed']}.json"))
        cases = {c["id"]: c for c in data["deletions"]}
        for d in r["deletions"]:
            if d.get("a3_score_version") == A3_VERSION:
                continue
            c = cases[d["id"]]
            d["leaked_strings"] = rs.leaked_strings(c, d["probe_after"], d["control_after"], d["retrieved_after"])
            d["leak_measurable"] = bool(c["leak_strings"])
            if "forgot_substring_v1" not in d:   # the judge-based `forgot` needs one call per case; keep it once computed
                d["forgot_substring_v1"] = d["forgot"]
                d["forgot"] = rs.is_abstain(d["probe_after"]) or strict_judge_v3(c["probe_question"], d["probe_after"], c["probe_answer"], rs.judge_fn)[0] < 0.99
            d["a3_score_version"] = A3_VERSION
            n_cases += 1
        r["summary"] = rs.summarize(r)
        f.write_text(json.dumps(r, indent=1, default=str))
        n_rec += 1
        a3 = r["summary"]["A3_deletion"]
        print(f"{f.name}: " + ", ".join(f"{k}: forgot {v['forgot']} leak {v['leaked']} (n_leak {v['n_leak_measurable']})" for k, v in a3.items()), flush=True)
    print(f"rescored {n_cases} cases in {n_rec} records")


if __name__ == "__main__":
    main()
