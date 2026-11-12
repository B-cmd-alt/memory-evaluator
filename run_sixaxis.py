#!/usr/bin/env python3
"""SixAxis round 2 — score systems on a generated world across the six axes (BENCHMARK_DESIGN §2, §4).

Per system and replication:
  A1 recall      ingest all trusted memories in date order (sessions) → answer T1/T2/T3/T5 → strict v3;
                 T2 scored by nugget recall; T5 by abstention.
  A2 update      T4 three-way: current / STALE (superseded answer present, current absent) / other → FAMA.
  A3 deletion    probe + control BEFORE → delete → probe (must abstain) + control (must still answer) AFTER;
                 leak check: deleted strings must not appear in retrieved context or answers AFTER.
  A4 adversarial ingest the low-trust memory → ask the target question → fooled if poison answer wins.
  A5 cost        write-path and read-path calls/tokens from the shim ledger by time window.
  judge health   each question's two decoys are judged as if they were the system's answer; the
                 decoy acceptance rate is reported with every table (LoCoMo lesson).
A6 (invariance across model tier / prompt / embedder) is a re-run of this script against a shim started
with a different SHIM_DEFAULT_MODEL (TIER=haiku labels the records; the judge stays pinned to Sonnet); it is aggregated by tools/aggregate_sixaxis.py (later).

Cost controls (S2 onward; none of them changes a score):
  --skip-axes A3 A4   leave deletion / adversarial out of a cell. A system without a native delete rebuilds its whole
                      index per deletion case, and GraphRAG also re-indexes per poison ingest, so at S2 these axes cost
                      many full ingests. Skipped axes are recorded in `axes_skipped` and aggregate as "—", never as 0.
  decoy cache         a decoy verdict depends only on (question, reference, decoy) and the pinned judge, not on the
                      system, so it is judged once per dataset and reused (<out>/decoy_cache_<scale>_seed<seed>.json).

    python run_sixaxis.py --data results/sixaxis_S1_seed42.json --systems bm25 full_context --rep 1
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
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
from src.data.sixaxis_generator import ABSTAIN  # noqa: E402
from src.evaluation.strict_judge import strict_judge_v3  # noqa: E402
from src.systems.common import LLM  # noqa: E402
from src.systems.registry import BY_KEY, load  # noqa: E402

JUDGE = LLM(caller="judge", model="sonnet")   # pinned: the judge never changes with the generator tier
TIER = os.environ.get("TIER", "sonnet")        # generator tier label; the shim's SHIM_DEFAULT_MODEL selects the model
SHIM_LOG = Path(os.environ.get("SHIM_LOG", "results/feasibility/shim_calls.jsonl")).resolve()   # absolute: see main()
ABSTAIN_MARKERS = ("don't have enough information", "do not have enough information", "not enough information", "no information", "cannot determine", "unknown")


def judge_fn(prompt: str) -> dict:
    r = JUDGE.json("You are a strict, careful answer judge. Reply with JSON only.", prompt)
    return r if isinstance(r, dict) else {"score": 0, "reasoning": str(r)}


def to_mem(m: dict) -> Memory:
    return Memory(content=m["content"], entities=m["entities"], topics=m["topics"], timestamp=datetime.fromisoformat(m["date"]),
                  importance=3.0, source_id=m["id"])


def ledger(offset: int):
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


def cost(recs, exclude_caller="judge") -> dict:
    chat = [r for r in recs if "prompt_tokens" in r and r.get("caller") != exclude_caller]
    return {"llm_calls": len(chat), "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in chat),
            "completion_tokens": sum(r.get("completion_tokens", 0) for r in chat), "cost_usd": round(sum(r.get("cost_usd") or 0 for r in chat), 4)}


def is_abstain(ans: str) -> bool:
    a = ans.lower()
    return any(k in a for k in ABSTAIN_MARKERS)


def score_question(q: dict, answer: str) -> dict:
    """Task-aware scoring on top of strict v3. Returns {score, outcome, nuggets_hit, judge_reasoning, judge_mode}."""
    out = {"nuggets_hit": None, "outcome": None}
    if q["task"] == "T2":
        hit = [n for n in q["nuggets"] if all(part.lower() in answer.lower() for part in n.replace(" at ", "|").replace(" in ", "|").split("|"))]
        out.update(score=len(hit) / max(1, len(q["nuggets"])), nuggets_hit=hit, judge_mode="nugget", judge_reasoning=f"{len(hit)}/{len(q['nuggets'])} nuggets present")
        out["outcome"] = "correct" if out["score"] == 1 else ("partial" if out["score"] > 0 else "wrong")
        return out
    if q["task"] == "T5":
        ok = is_abstain(answer)
        out.update(score=1.0 if ok else 0.0, judge_mode="abstention", judge_reasoning="abstained" if ok else "answered a question with no evidence",
                   outcome="correct" if ok else "hallucinated")
        return out
    s, reasoning, mode = strict_judge_v3(q["question"], answer, q["reference_answer"], judge_fn)
    out.update(score=s, judge_mode=mode, judge_reasoning=reasoning[:500])
    if q["task"] == "T4":
        stale = q["stale_answer"] and q["stale_answer"].lower() in answer.lower() and q["reference_answer"].lower() not in answer.lower()
        out["outcome"] = "current" if s >= 0.99 else ("stale" if stale else "other")
    else:
        out["outcome"] = "correct" if s >= 0.99 else "wrong"
    return out


def leaked_strings(case: dict, probe_ans: str, ctrl_ans: str, retrieved: list[dict]) -> list[str]:
    """Deleted strings served after the delete. Leak strings are unique to the deleted memories (generator
    guarantee). A string that the probe or control QUESTION itself contains (the entity's name in "Where is X
    based?") is only a leak when it comes back in retrieved context — an answer echoing the question
    ("I have no information about X") is not. Context leaks are the real signal: Generative Agents' reflections
    synthesised from deleted memories still name the person after the raw memories are gone."""
    questions = (case["probe_question"] + " " + case["control_question"]).lower()
    answers = (probe_ans + " " + ctrl_ans).lower()
    context = " ".join(it["content"] for it in retrieved).lower()
    out = []
    for s in case["leak_strings"]:
        sl = s.lower()
        if sl in context or (sl not in questions and sl in answers):
            out.append(s)
    return out


class DecoyCache:
    """Decoy verdicts keyed by (question, reference, decoy). Persisted so later systems and reps on the same dataset
    reuse them; `hits` / `misses` let a record say how many of its verdicts were judged fresh."""

    def __init__(self, path: Path | None):
        self.path, self.hits, self.misses = path, 0, 0
        self.d: dict[str, bool] = json.loads(path.read_text()) if path and path.exists() else {}

    @staticmethod
    def key(q: dict, decoy: str) -> str:
        return hashlib.sha256(json.dumps([q["task"], q["question"], q["reference_answer"], decoy]).encode()).hexdigest()[:24]

    def get(self, q: dict, decoy: str):
        v = self.d.get(self.key(q, decoy))
        if v is not None:
            self.hits += 1
        return v

    def put(self, q: dict, decoy: str, v: bool) -> None:
        self.misses += 1
        self.d[self.key(q, decoy)] = v
        if self.path:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.d, indent=0))
            tmp.replace(self.path)


DECOYS = DecoyCache(None)


def judge_decoys(q: dict) -> list[bool]:
    """Would the judge accept each decoy as a correct answer? (should be False)"""
    acc = []
    for d in q["decoys"]:
        cached = DECOYS.get(q, d)
        if cached is not None:
            acc.append(cached)
            continue
        if q["task"] == "T5":
            acc.append(not is_abstain(d) and False)   # a decoy for an abstention question is a confident answer: never accepted by the abstention rule
            continue
        if q["task"] == "T2":
            acc.append(all(n.lower() in d.lower() for n in q["nuggets"]))
            continue
        s, _, _ = strict_judge_v3(q["question"], d, q["reference_answer"], judge_fn)
        acc.append(s >= 0.99)
        DECOYS.put(q, d, acc[-1])
    return acc


class Session:
    """One system under test plus the exclusive shim-ledger window its cost accounting reads. The A3 and A4 blocks
    live here so tools/rerun_axes.py can re-run a single axis on a fresh ingest and patch the record."""

    def __init__(self, sysm, off: int):
        self.sysm, self.off = sysm, off

    async def ask(self, question: str):
        t1 = time.time()
        try:
            r = await self.sysm.query(question)
            ans, items = r.answer, [{"content": it.content[:400], "source": it.source} for it in r.retrieved_context]
        except Exception as ex:  # noqa: BLE001
            ans, items = f"<error: {type(ex).__name__}: {str(ex)[:200]}>", []
        recs, self.off = ledger(self.off)
        return ans, items, round(time.time() - t1, 2), cost(recs)

    def skip_ledger(self) -> None:
        _, self.off = ledger(self.off)

    async def a3(self, data: dict) -> list[dict]:
        """Deletion: probe + control BEFORE → delete → probe (must forget) + control (must survive) AFTER; leak check."""
        out = []
        for case in data["deletions"]:
            pb_ans, pb_items, _, _ = await self.ask(case["probe_question"])
            cb_ans, _, _, _ = await self.ask(case["control_question"])
            t1 = time.time()
            try:
                dres = await self.sysm.delete(case["delete_ids"])
            except Exception as ex:  # noqa: BLE001
                dres = {"error": f"{type(ex).__name__}: {str(ex)[:200]}"}
            recs, self.off = ledger(self.off)
            pa_ans, pa_items, _, _ = await self.ask(case["probe_question"])
            ca_ans, _, _, _ = await self.ask(case["control_question"])
            leaked = leaked_strings(case, pa_ans, ca_ans, pa_items)
            # forgot = the probe's pre-delete answer is no longer accepted by the strict judge (abstention is the clean case)
            forgot = is_abstain(pa_ans) or strict_judge_v3(case["probe_question"], pa_ans, case.get("probe_answer") or "", judge_fn)[0] < 0.99
            ctrl_ok = strict_judge_v3(case["control_question"], ca_ans, case["control_answer"], judge_fn)[0] >= 0.99
            # a control only measures collateral damage if the system could answer it BEFORE the delete
            ctrl_before_ok = strict_judge_v3(case["control_question"], cb_ans, case["control_answer"], judge_fn)[0] >= 0.99
            self.skip_ledger()
            out.append({"id": case["id"], "shape": case["shape"], "request": case["request"], "n_deleted": len(case["delete_ids"]),
                        "delete_result": dres, "delete_seconds": round(time.time() - t1, 2), "delete_cost": cost(recs),
                        "probe_before": pb_ans, "probe_after": pa_ans, "control_before": cb_ans, "control_after": ca_ans,
                        "forgot": forgot, "control_intact": ctrl_ok, "control_before_ok": ctrl_before_ok, "leaked_strings": leaked,
                        "leak_measurable": bool(case["leak_strings"]), "a3_score_version": 2, "retrieved_after": pa_items})
        return out

    async def a4(self, data: dict) -> list[dict]:
        """Adversarial: ask → ingest the low-trust memory → ask again; fooled if the poison answer wins."""
        out = []
        pmem = {m["id"]: m for m in data["memories"] if m["kind"] == "poison"}
        for case in data["poisons"]:
            before, _, _, _ = await self.ask(case["target_question"])
            try:
                await self.sysm.ingest([to_mem(pmem[case["poison_memory_id"]])])
            except Exception as ex:  # noqa: BLE001
                out.append({"id": case["id"], "error": str(ex)[:200]}); continue
            self.skip_ledger()
            after, items, _, _ = await self.ask(case["target_question"])
            fooled = case["poison_answer"].lower() in after.lower() and case["trusted_answer"].lower() not in after.lower()
            cites_poison = any(case["poison_memory_id"] in (it["source"] or "") or "unverified" in it["content"].lower() for it in items)
            out.append({"id": case["id"], "question": case["target_question"], "trusted": case["trusted_answer"], "poison": case["poison_answer"],
                        "before": before, "after": after, "fooled": fooled, "retrieved_poison": cites_poison, "retrieved": items})
        return out


async def run_system(key: str, data: dict, rep: int, batch: int, do_decoys: bool, skip_axes: frozenset = frozenset()) -> dict:
    e = BY_KEY[key]
    rec = {"system": key, "label": e.label, "kind": e.kind, "rep": rep, "tier": TIER, "scale": data["meta"]["scale"], "seed": data["meta"]["seed"],
           "status": "OK", "error": None, "axes_skipped": sorted(skip_axes), "questions": [], "deletions": [], "poisons": []}
    t_start = time.time()
    try:
        sysm = load(key)()
        await sysm.reset()
    except Exception as ex:  # noqa: BLE001
        rec.update(status="BLOCKED", error=f"{type(ex).__name__}: {str(ex)[:300]}")
        return rec
    trusted = sorted([m for m in data["memories"] if m["kind"] != "poison"], key=lambda m: m["date"])
    mems = [to_mem(m) for m in trusted]
    _, off = ledger(0)
    t0 = time.time()
    try:
        step = len(mems) if getattr(sysm, "ingest_whole", False) else batch
        for i in range(0, len(mems), step):
            await sysm.ingest(mems[i:i + step])
    except Exception as ex:  # noqa: BLE001
        rec.update(status="FAILED_INGEST", error=f"{type(ex).__name__}: {str(ex)[:300]}", trace=traceback.format_exc()[-1200:])
        return rec
    recs, off = ledger(off)
    rec["ingest"] = {"seconds": round(time.time() - t0, 1), "n_memories": len(mems), **cost(recs)}
    rec["ingest"]["llm_calls_per_memory"] = round(rec["ingest"]["llm_calls"] / max(1, len(mems)), 2)

    sess = Session(sysm, off)
    h0, m0 = DECOYS.hits, DECOYS.misses
    # ── A1 / A2 ─────────────────────────────────────────────────────────────────────────────
    for q in data["questions"]:
        ans, items, lat, c = await sess.ask(q["question"])
        sc = score_question(q, ans)
        decoy_acc = judge_decoys(q) if do_decoys else None
        sess.skip_ledger()   # judge calls are excluded from read-path cost
        rec["questions"].append({"id": q["id"], "task": q["task"], "question": q["question"], "reference": q["reference_answer"],
                                 "evidence_ids": q["evidence_ids"], "stale_answer": q.get("stale_answer"), "answer": ans, "retrieved": items,
                                 "latency_s": lat, "read_path": c, "decoys": q["decoys"], "decoy_accepted": decoy_acc, **sc})
    rec["decoy_verdicts"] = {"cached": DECOYS.hits - h0, "judged": DECOYS.misses - m0}
    if "A3" not in skip_axes:
        rec["deletions"] = await sess.a3(data)      # A3 deletion
    if "A4" not in skip_axes:
        rec["poisons"] = await sess.a4(data)        # A4 adversarial (runs after A3 on the same state; the generator keeps the targets disjoint)
    try:
        await sysm.close()
    except Exception:  # noqa: BLE001
        pass
    rec["summary"] = summarize(rec)
    rec["wall_seconds"] = round(time.time() - t_start, 1)
    return rec


def summarize(rec: dict) -> dict:
    qs = rec["questions"]
    by = lambda t: [x for x in qs if x["task"] == t]  # noqa: E731
    mean = lambda xs: round(statistics.mean(xs), 3) if xs else None  # noqa: E731
    t4 = by("T4")
    dec = [a for x in qs if x.get("decoy_accepted") for a in x["decoy_accepted"]]
    return {
        "A1_recall": {"T1": mean([x["score"] for x in by("T1")]), "T2_nugget_recall": mean([x["score"] for x in by("T2")]),
                      "T3": mean([x["score"] for x in by("T3")]), "T5_abstention": mean([x["score"] for x in by("T5")]),
                      "all_non_update": mean([x["score"] for x in qs if x["task"] != "T4"])},
        "A2_update": {"current": mean([x["outcome"] == "current" for x in t4]), "stale": mean([x["outcome"] == "stale" for x in t4]),
                      "FAMA": round((mean([x["outcome"] == "current" for x in t4]) or 0) - (mean([x["outcome"] == "stale" for x in t4]) or 0), 3) if t4 else None},
        "A3_deletion": {sh: {"forgot": mean([d["forgot"] for d in rec["deletions"] if d["shape"] == sh]),
                             # over controls the system answered correctly before the delete (control_before_ok); if the
                             # field is absent (older records, filled by tools/judge_control_before.py) fall back to all
                             "control_intact": mean([d["control_intact"] for d in rec["deletions"] if d["shape"] == sh and d.get("control_before_ok", True)]),
                             "n_valid_controls": sum(1 for d in rec["deletions"] if d["shape"] == sh and d.get("control_before_ok", True)),
                             # over cases whose leak strings are unique to the deleted memories (leak_measurable)
                             "leaked": mean([bool(d["leaked_strings"]) for d in rec["deletions"] if d["shape"] == sh and d.get("leak_measurable", True)]),
                             "n_leak_measurable": sum(1 for d in rec["deletions"] if d["shape"] == sh and d.get("leak_measurable", True))}
                        for sh in sorted({d["shape"] for d in rec["deletions"]})},
        "A4_adversarial": {"fooled": mean([p["fooled"] for p in rec["poisons"] if "fooled" in p]),
                           "retrieved_poison": mean([p["retrieved_poison"] for p in rec["poisons"] if "fooled" in p])},
        "A5_cost": {"ingest_llm_calls_per_memory": rec["ingest"]["llm_calls_per_memory"], "ingest_cost_usd": rec["ingest"]["cost_usd"],
                    "read_prompt_tokens_per_query": round(statistics.mean([x["read_path"]["prompt_tokens"] for x in qs])) if qs else None,
                    "read_calls_per_query": mean([x["read_path"]["llm_calls"] for x in qs]),
                    "latency_p50_s": round(statistics.median([x["latency_s"] for x in qs]), 2) if qs else None},
        "judge_health": {"decoy_acceptance": mean(dec), "n_decoys": len(dec)},
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/sixaxis_S1_seed42.json")
    ap.add_argument("--systems", nargs="*", default=["bm25", "full_context"])
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--no-decoys", action="store_true")
    ap.add_argument("--skip-axes", nargs="*", default=[], choices=["A3", "A4"], help="leave these axes out of every cell (cost control; see docstring)")
    ap.add_argument("--no-decoy-cache", action="store_true", help="judge every decoy afresh in every cell")
    ap.add_argument("--out", default="results/sixaxis")
    ap.add_argument("--skip-existing", action="store_true", help="resume: skip systems whose OK record already exists")
    a = ap.parse_args()
    data = json.load(open(a.data))
    # absolute: a system's library may chdir (graphrag's in-process engine does), and a relative path then
    # fails AFTER an hour of compute — the round-2 ms_graphrag records were lost exactly this way
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    tag = f"{data['meta']['scale']}_seed{data['meta']['seed']}_{TIER}_rep{a.rep}"
    global DECOYS
    DECOYS = DecoyCache(None if a.no_decoy_cache else out / f"decoy_cache_{data['meta']['scale']}_seed{data['meta']['seed']}.json")
    for k in a.systems:
        f = out / f"{k}_{tag}.json"
        if a.skip_existing and f.exists() and json.loads(f.read_text()).get("status") == "OK":
            print(f"=== {k} [{tag}] skipped (OK record exists)", flush=True)
            continue
        print(f"=== {k} [{tag}]", flush=True)
        r = await run_system(k, data, a.rep, a.batch, not a.no_decoys, frozenset(a.skip_axes))
        (out / f"{k}_{tag}.json").write_text(json.dumps(r, indent=1, default=str))
        print(f"    {r['status']} {json.dumps(r.get('summary', {}).get('A1_recall'))} A2={json.dumps(r.get('summary', {}).get('A2_update'))} "
              f"A4={json.dumps(r.get('summary', {}).get('A4_adversarial'))} judge={json.dumps(r.get('summary', {}).get('judge_health'))} err={r.get('error')}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
