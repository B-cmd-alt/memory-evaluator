"""FastAPI app: the static evidence page + three interactive features.

    /                      the interactive hub (Be the Judge · Ask the Memory · Break the Memory)
    /evidence              the round-1 evidence page (site/index.html)
    /api/judge/*           zero-cost game over the judge calibration set; votes are stored anonymously
    /api/ask/*             canonical questions: precomputed records for 11 systems, or live re-asks of the cheap ones
    /api/challenge         Break the Memory: your fact / update / poison, live on the cheap systems
    /api/budget            today's spend against the hard daily cap
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import judge_game, live
from .budget import budget
from .config import HAS_KEY, LIVE_SYSTEMS, LLM_MODEL, ROOT

app = FastAPI(title="Memory System Evaluator", docs_url=None, redoc_url=None)
STATIC = Path(__file__).parent / "static"
SITE = ROOT / "site"


def _ip(req: Request) -> str:
    fwd = req.headers.get("x-forwarded-for", "")
    return (fwd.split(",")[0].strip() if fwd else (req.client.host if req.client else "?"))


def _ip_hash(req: Request) -> str:
    return hashlib.sha256((_ip(req) + os.environ.get("IP_SALT", "memory-evaluator")).encode()).hexdigest()[:16]


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/evidence")
def evidence():
    return FileResponse(SITE / "index.html")


@app.get("/health")
def health():
    return {"ok": True, "live_enabled": HAS_KEY, "model": LLM_MODEL, "live_systems": LIVE_SYSTEMS}


# ── Be the Judge ─────────────────────────────────────────────────────────────────────────────────
class Vote(BaseModel):
    item_id: str
    score: float = Field(ge=0, le=1)
    session: str = ""


@app.get("/api/judge/next")
def judge_next(task: str | None = None):
    return judge_game.next_item(task)


@app.get("/api/judge/reveal/{item_id}")
def judge_reveal(item_id: str):
    r = judge_game.reveal(item_id)
    if not r:
        raise HTTPException(404, "unknown item")
    return r


@app.post("/api/judge/vote")
def judge_vote(v: Vote, req: Request):
    try:
        judge_game.vote(v.item_id, v.score, v.session, _ip_hash(req))
    except KeyError:
        raise HTTPException(404, "unknown item")
    return {"ok": True, "reveal": judge_game.reveal(v.item_id), "stats": judge_game.stats()}


@app.get("/api/judge/stats")
def judge_stats():
    return judge_game.stats()


# ── Ask the Memory ───────────────────────────────────────────────────────────────────────────────
class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=400)
    systems: list[str] = ["full_context", "bm25", "vector_rag"]


class AskCanonical(BaseModel):
    qid: str
    systems: list[str] = ["full_context", "bm25", "vector_rag"]


@app.get("/api/ask/questions")
def ask_questions():
    return {"questions": live.canonical_questions(), "live_systems": LIVE_SYSTEMS, "live_enabled": HAS_KEY}


@app.get("/api/ask/precomputed/{qid}")
def ask_precomputed(qid: str):
    rows = live.precomputed(qid)
    if not rows:
        raise HTTPException(404, "unknown question")
    q = next(x for x in live.S1_QUESTIONS if x["id"] == qid)
    return {"qid": qid, "question": q["question"], "reference": q["reference_answer"], "task": q["task"], "systems": rows}


def _gate(req: Request, systems: list[str]):
    if not HAS_KEY:
        raise HTTPException(503, "live mode is off on this deployment (no model credential); precomputed evidence still works")
    bad = [s for s in systems if s not in LIVE_SYSTEMS]
    if bad or not systems:
        raise HTTPException(400, f"live systems are {list(LIVE_SYSTEMS)}; not: {bad}")
    ok, why = budget().allow(_ip(req))
    if not ok:
        raise HTTPException(429, why)


@app.post("/api/ask")
async def ask(a: Ask, req: Request):
    _gate(req, a.systems[:4])
    res = await live.ask_live(a.question, a.systems[:4])
    res["budget"] = budget().charge(_ip(req), res["cost_usd"])
    return res


@app.post("/api/ask/canonical")
async def ask_canonical(a: AskCanonical, req: Request):
    _gate(req, a.systems[:4])
    try:
        res = await live.ask_canonical_live(a.qid, a.systems[:4])
    except KeyError:
        raise HTTPException(404, "unknown question")
    res["budget"] = budget().charge(_ip(req), res["cost_usd"])
    return res


# ── Break the Memory ─────────────────────────────────────────────────────────────────────────────
class Challenge(BaseModel):
    fact: str = Field(min_length=5, max_length=300)
    fact_key: str = Field(min_length=1, max_length=60)
    update: str | None = Field(default=None, max_length=300)
    update_key: str | None = Field(default=None, max_length=60)
    poison: str | None = Field(default=None, max_length=300)
    poison_key: str | None = Field(default=None, max_length=60)
    question: str = Field(min_length=3, max_length=300)
    systems: list[str] = ["full_context", "bm25", "vector_rag", "mastra_om"]


@app.post("/api/challenge")
async def challenge(c: Challenge, req: Request):
    _gate(req, c.systems[:4])
    res = await live.challenge(c.fact, c.fact_key, c.update, c.update_key, c.poison, c.poison_key, c.question, c.systems[:4])
    res["budget"] = budget().charge(_ip(req), res["cost_usd"])
    return res


@app.get("/api/budget")
def budget_ledger():
    return budget().ledger()


app.mount("/static", StaticFiles(directory=STATIC), name="static")
