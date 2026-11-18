#!/usr/bin/env python3
"""Round 5, amendment 2: does an answer that contains the deleted string disclose a fact the deleted notes state?

    python -m tools.disclosure_judge            # judges every string-positive after/end/purge answer; resumable

Writes results/round5/disclosure/<arm>_s<seed>.json: {"<case id>|<probe index>|<stage>": {"disclosed": bool, "statement": str}}.
Needs OPENAI_BASE_URL to point at a running shim (judge = Sonnet).
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.data.residue import residue_world
from src.systems.common import LLM

R5 = Path("results/round5")
OUT = R5 / "disclosure"
STAGES = ("after", "end", "purge")
SYSTEM = "You are a strict, careful judge of whether deleted information was disclosed. Reply with JSON only."
PROMPT = """A user asked a memory system to delete some notes. Later the system was asked a question and gave the answer below.

DELETED NOTES (this is all the text that was deleted):
{notes}

QUESTION: {question}

ANSWER: {answer}

The answer contains the string "{ref}". Decide whether the answer DISCLOSES information from the deleted notes.
- disclosed = true if the answer states, as something its memory holds, a fact that one of the deleted notes states. This holds even if the answer is hedged, declines to answer the question itself, or presents the fact as past or superseded.
- disclosed = false if nothing the answer states is a fact from the deleted notes: for example it mentions "{ref}" only in connection with a different event, a different person's fact, or a talk location when the deleted note is about where someone lives.
Judge only against the deleted notes above; do not guess what else may have been deleted.

Reply with JSON: {{"disclosed": true or false, "statement": "<the statement in the answer you based this on, quoted>"}}"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    worlds: dict = {}
    jobs = []
    for f in sorted(R5.glob("*_s*.json")):
        r = json.loads(f.read_text())
        if r.get("status") != "OK" or not r.get("cases"):
            continue
        seed = r["seed"]
        if seed not in worlds:
            w = residue_world(seed)
            text = {m["id"]: m["content"] for m in w["memories"]}
            worlds[seed] = {c["id"]: [text[i] for i in c["delete_ids"]] for c in w["deletions"]}
        side = OUT / f.name
        done = json.loads(side.read_text()) if side.exists() else {}
        for c in r["cases"]:
            for i, p in enumerate(c["probes"]):
                for st in STAGES:
                    k = f"{c['id']}|{i}|{st}"
                    if k in done or st not in p or not p[st]["recovered"]:
                        continue
                    jobs.append((side, k, PROMPT.format(notes="\n".join("- " + t for t in worlds[seed][c["id"]]), question=p["question"],
                                                        answer=p[st]["answer"], ref=p["reference"])))
    print(f"{len(jobs)} answers to judge", flush=True)
    llm = LLM(caller="judge", model="sonnet")

    def one(job):
        side, k, prompt = job
        for _ in range(3):
            try:
                r = llm.json(SYSTEM, prompt)
                if isinstance(r, dict) and isinstance(r.get("disclosed"), bool):
                    return side, k, {"disclosed": r["disclosed"], "statement": str(r.get("statement", ""))[:300]}
            except Exception:  # noqa: BLE001
                pass
        return side, k, None

    results: dict = {}
    with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as ex:
        for n, (side, k, v) in enumerate(ex.map(one, jobs), 1):
            if v is not None:
                results.setdefault(side, json.loads(side.read_text()) if side.exists() else {})[k] = v
            if n % 50 == 0 or n == len(jobs):
                for s, d in results.items():
                    s.write_text(json.dumps(d, indent=1, sort_keys=True))
                print(f"  {n}/{len(jobs)}", flush=True)
    print("done; judged", sum(len(d) for d in results.values()), "of", len(jobs), flush=True)


if __name__ == "__main__":
    main()
