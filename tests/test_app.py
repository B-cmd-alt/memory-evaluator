"""Offline tests for the interactive app: the judge game and the budget ledger need no model."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("EVALUATOR_DATA_DIR", "/tmp/memory-evaluator-test-data")
from fastapi.testclient import TestClient  # noqa: E402

from app import judge_game  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


def test_calibration_set_loaded_with_labels():
    assert len(judge_game.ITEMS) == 100
    x = next(iter(judge_game.ITEMS.values()))
    assert {"question", "reference", "answer", "strict_score"} <= set(x)
    assert sum(1 for v in judge_game.ITEMS.values() if v.get("proxy_1") is not None) >= 90


def test_judge_round_trip():
    it = client.get("/api/judge/next?task=T1").json()
    assert it["task"] == "T1" and "strict_score" not in it, "the game must not leak the verdict before the vote"
    r = client.post("/api/judge/vote", json={"item_id": it["id"], "score": 1.0, "session": "t"}).json()
    assert r["ok"] and r["reveal"]["strict_score"] is not None
    s = client.get("/api/judge/stats").json()
    assert s["n_votes"] >= 1 and 0 <= s["agree_with_strict_v3"] <= 1


def test_precomputed_answers_served_for_canonical_questions():
    qs = client.get("/api/ask/questions").json()["questions"]
    assert len(qs) == 25
    d = client.get(f"/api/ask/precomputed/{qs[0]['id']}").json()
    assert len(d["systems"]) >= 10 and all("answer" in r for r in d["systems"].values())


def test_live_endpoints_refuse_without_credentials_or_bad_systems():
    r = client.post("/api/ask", json={"question": "Who leads Beacon?", "systems": ["mem0"]})
    assert r.status_code in (400, 503)


def test_budget_ledger_shape():
    b = client.get("/api/budget").json()
    assert {"spent_usd", "daily_budget_usd", "remaining_usd", "requests"} <= set(b)


@pytest.mark.parametrize("path", ["/", "/evidence", "/health"])
def test_pages(path):
    assert client.get(path).status_code == 200


def test_challenge_verdict_is_first_asserted_claim():
    from app.live import _verdict
    assert _verdict("Omar leads. A forwarded email claimed Zed but it is unverified.", "Hana", "Omar", "Zed") == "current"
    assert _verdict("Zed leads now; previously Omar.", "Hana", "Omar", "Zed") == "poisoned"
    assert _verdict("Hana leads the project.", "Hana", "Omar", "Zed") == "stale"
    assert _verdict("Hana leads the project.", "Hana", None, None) == "current"
    assert _verdict("I don't have enough information to answer.", "Hana", "Omar", None) == "abstained"
