"""Offline tests for run_sixaxis.py's cost controls: --skip-axes and the per-dataset decoy cache.
A fake LLM answers and a fake judge scores, so nothing reaches the shim."""
from __future__ import annotations

import asyncio
import json

import pytest

import run_sixaxis as rs
from src.systems.baselines import BM25Baseline
from tests.test_systems_contract import FakeLLM

DATA = json.load(open("results/sixaxis_S1_seed42.json"))


@pytest.fixture
def fake_judge(monkeypatch, tmp_path):
    calls = {"n": 0}

    def judge(prompt):
        calls["n"] += 1
        return {"score": 0, "reasoning": "fake"}

    monkeypatch.setattr(rs, "judge_fn", judge)
    monkeypatch.setattr(rs, "SHIM_LOG", tmp_path / "no_ledger.jsonl")
    monkeypatch.setattr(rs, "load", lambda key: (lambda: BM25Baseline({"llm": FakeLLM()})))
    return calls


def run(skip=frozenset()):
    return asyncio.run(rs.run_system("bm25", DATA, 1, 10, True, skip))


def test_skip_axes_leaves_deletion_and_poison_out(fake_judge, monkeypatch, tmp_path):
    monkeypatch.setattr(rs, "DECOYS", rs.DecoyCache(None))
    rec = run(frozenset({"A3", "A4"}))
    assert rec["status"] == "OK" and rec["axes_skipped"] == ["A3", "A4"]
    assert rec["deletions"] == [] and rec["poisons"] == []
    assert len(rec["questions"]) == len(DATA["questions"])
    # a skipped axis aggregates as missing ("—"), never as a zero score
    assert rec["summary"]["A3_deletion"] == {} and rec["summary"]["A4_adversarial"]["fooled"] is None


def test_all_axes_run_by_default(fake_judge, monkeypatch):
    monkeypatch.setattr(rs, "DECOYS", rs.DecoyCache(None))
    rec = run()
    assert rec["axes_skipped"] == []
    assert len(rec["deletions"]) == len(DATA["deletions"]) and len(rec["poisons"]) == len(DATA["poisons"])


def test_decoy_verdicts_are_judged_once_per_dataset(fake_judge, monkeypatch, tmp_path):
    path = tmp_path / "decoy_cache.json"
    monkeypatch.setattr(rs, "DECOYS", rs.DecoyCache(path))
    first = run(frozenset({"A3", "A4"}))
    judged = first["decoy_verdicts"]["judged"]
    assert judged > 0 and first["decoy_verdicts"]["cached"] == 0
    # a second system / rep in a NEW process: reads the persisted cache, judges no decoy again
    monkeypatch.setattr(rs, "DECOYS", rs.DecoyCache(path))
    n_before = fake_judge["n"]
    second = run(frozenset({"A3", "A4"}))
    assert second["decoy_verdicts"] == {"cached": judged, "judged": 0}
    assert [q["decoy_accepted"] for q in second["questions"]] == [q["decoy_accepted"] for q in first["questions"]]
    assert second["summary"]["judge_health"] == first["summary"]["judge_health"]
    # the only judge calls left are the system's own answers
    assert fake_judge["n"] - n_before < n_before
