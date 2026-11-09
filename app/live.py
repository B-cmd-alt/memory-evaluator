"""Ask the Memory and Break the Memory: live runs of the cheap-write systems on the same model, with cost
reported per answer. The expensive systems (GraphRAG, Graphiti, A-MEM, Mem0, memory tool) are served from
their round-1 records for the 25 canonical questions and are not run live.
"""
from __future__ import annotations

import asyncio
import json
import random
import threading
import time
from datetime import datetime, timedelta

from src.architectures.base import Memory
from src.data.sixaxis_generator import ABSTAIN
from src.evaluation.strict_judge import strict_judge_v3
from src.systems.common import LLM
from src.systems.registry import BY_KEY, load

from .config import LIVE_SYSTEMS, LIVE_TIMEOUT_S, PRECOMPUTED_SYSTEMS, RESULTS

_ABSTAIN_MARKERS = ("don't have enough information", "do not have enough information", "not enough information", "cannot determine")


def is_abstain(ans: str) -> bool:
    a = (ans or "").lower()
    return any(k in a for k in _ABSTAIN_MARKERS)


# ── the canonical S1 world (90 memories) + round-1 records ───────────────────────────────────────
S1 = json.loads((RESULTS / "s1_dataset.json").read_text())
S1_MEMS = sorted([Memory(content=m["content"], entities=[], topics=m.get("topics", []), timestamp=datetime.fromisoformat(m["date"]),
                         importance=3.0, source_id=m["id"]) for m in S1["memories"]], key=lambda m: m.timestamp)
S1_QUESTIONS = S1["questions"]


def _load_precomputed() -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for key in PRECOMPUTED_SYSTEMS:
        p = RESULTS / "feasibility" / f"{key}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        if d.get("status") not in ("OK", "PARTIAL"):
            continue
        out[key] = {q["id"]: q for q in d["questions"]}
    return out


PRECOMPUTED = _load_precomputed()


def canonical_questions() -> list[dict]:
    return [{"id": q["id"], "task": q["task"], "question": q["question"], "reference": q["reference_answer"]} for q in S1_QUESTIONS]


def precomputed(qid: str) -> dict:
    rows = {}
    for key, qs in PRECOMPUTED.items():
        r = qs.get(qid)
        if r:
            rows[key] = {"label": BY_KEY[key].label, "answer": r["answer"], "score": r["score"], "retrieved": r["retrieved"][:5],
                         "judge_reasoning": r["judge_reasoning"], "latency_s": r["latency_s"], "read_path": r["read_path"], "live": False}
    return rows


# ── live system pool over the canonical world (lazy, cached per process) ────────────────────────
class Pool:
    def __init__(self):
        self._sys: dict[str, object] = {}
        self._lock = threading.Lock()

    async def get(self, key: str):
        if key not in LIVE_SYSTEMS:
            raise ValueError(f"{key} is not available live")
        with self._lock:
            s = self._sys.get(key)
        if s is None:
            s = load(key)()
            await s.reset()
            await s.ingest(S1_MEMS)
            with self._lock:
                self._sys[key] = s
        return s


POOL = Pool()


async def ask_live(question: str, systems: list[str], judge: bool = True) -> dict:
    """Free-text question over the canonical world, one or more live systems, one shared model."""
    out, total_cost = {}, 0.0
    for key in systems:
        s = await POOL.get(key)
        c0 = s.llm.cost_usd
        t0 = time.time()
        try:
            r = await asyncio.wait_for(s.query(question), timeout=LIVE_TIMEOUT_S)
            ans, items = r.answer, [{"content": it.content[:400], "source": it.source} for it in r.retrieved_context]
        except Exception as ex:  # noqa: BLE001
            ans, items = f"<error: {type(ex).__name__}: {str(ex)[:160]}>", []
        cost = s.llm.cost_usd - c0
        total_cost += cost
        out[key] = {"label": BY_KEY[key].label, "answer": ans, "retrieved": items, "latency_s": round(time.time() - t0, 2),
                    "cost_usd": round(cost, 5), "abstained": is_abstain(ans), "live": True}
    return {"question": question, "systems": out, "cost_usd": round(total_cost, 5)}


async def ask_canonical_live(qid: str, systems: list[str]) -> dict:
    """A canonical question asked live, judged against its reference with strict-judge v3."""
    q = next((x for x in S1_QUESTIONS if x["id"] == qid), None)
    if not q:
        raise KeyError(qid)
    res = await ask_live(q["question"], systems, judge=False)
    jl = LLM(caller="judge")
    for key, r in res["systems"].items():
        if r["answer"].startswith("<error"):
            continue
        score, reasoning, mode = strict_judge_v3(q["question"], r["answer"], q["reference_answer"],
                                                 lambda p: jl.json("You are a strict, careful answer judge. Reply with JSON only.", p))
        r.update(score=score, judge_mode=mode, judge_reasoning=reasoning[:400])
    res["cost_usd"] = round(res["cost_usd"] + jl.cost_usd, 5)
    res["reference"] = q["reference_answer"]
    return res


# ── Break the Memory: a tiny world the visitor writes ────────────────────────────────────────────
def _verdict(ans: str, fact_key: str | None, update_key: str | None, poison_key: str | None) -> str:
    """Which claim does the answer ASSERT? A good answer often mentions the poison only to dismiss it
    ("a forwarded email claimed Zed, but..."), so substring presence is not enough: the key mentioned FIRST is
    taken as the asserted one — answers lead with their claim. Caveat shown in the UI."""
    low = ans.lower()
    if is_abstain(ans):
        return "abstained"
    pos = {}
    for name, k in (("poisoned", poison_key), ("current", update_key), ("stale", fact_key if update_key else None),
                    ("current_fact", fact_key if not update_key else None)):
        if k and k.lower() in low:
            pos[name] = low.index(k.lower())
    if not pos:
        return "other"
    first = min(pos, key=pos.get)
    return "current" if first == "current_fact" else first


async def challenge(fact: str, fact_key: str, update: str | None, update_key: str | None, poison: str | None, poison_key: str | None,
                    question: str, systems: list[str], seed: int = 7) -> dict:
    rng = random.Random(seed)
    filler = rng.sample(S1_MEMS, 15)
    base = datetime(2026, 1, 10)
    mems = [Memory(content=fact, entities=[], topics=["fact"], timestamp=base, importance=3.0, source_id="you_fact")]
    if update:
        mems.append(Memory(content=update, entities=[], topics=["update"], timestamp=base + timedelta(days=40), importance=3.0, source_id="you_update"))
    if poison:
        mems.append(Memory(content=f"Note from an unverified forwarded email: {poison}", entities=[], topics=["unverified"],
                           timestamp=base + timedelta(days=60), importance=3.0, source_id="you_poison"))
    world = sorted(mems + [Memory(content=m.content, entities=[], topics=m.topics, timestamp=m.timestamp, importance=3.0, source_id=m.source_id)
                           for m in filler], key=lambda m: m.timestamp)
    out, total = {}, 0.0
    for key in systems:
        if key not in LIVE_SYSTEMS:
            continue
        s = load(key)()
        await s.reset()
        c0 = 0.0
        t0 = time.time()
        try:
            await asyncio.wait_for(s.ingest(world), timeout=LIVE_TIMEOUT_S)
            r = await asyncio.wait_for(s.query(question), timeout=LIVE_TIMEOUT_S)
            ans, items = r.answer, [{"content": it.content[:300], "source": it.source} for it in r.retrieved_context]
        except Exception as ex:  # noqa: BLE001
            ans, items = f"<error: {type(ex).__name__}: {str(ex)[:160]}>", []
        cost = s.llm.cost_usd - c0
        total += cost
        verdict = _verdict(ans, fact_key, update_key if update else None, poison_key if poison else None)
        saw = {"fact": any(it["source"] == "you_fact" for it in items), "update": any(it["source"] == "you_update" for it in items),
               "poison": any(it["source"] == "you_poison" for it in items)}
        out[key] = {"label": BY_KEY[key].label, "answer": ans, "verdict": verdict, "retrieved": items, "retrieved_yours": saw,
                    "latency_s": round(time.time() - t0, 2), "cost_usd": round(cost, 5)}
        try:
            await s.close()
        except Exception:  # noqa: BLE001
            pass
    return {"question": question, "systems": out, "cost_usd": round(total, 5), "n_memories": len(world)}
