"""Be the Judge: you score an answer, then see what strict-judge v3, two careful independent labelers and
a weaker LLM judge said. Zero model cost — every item comes from the 100-question calibration set that
backs docs/JUDGE_METHODOLOGY.md. Your vote is stored (anonymously) and becomes a third label.
"""
from __future__ import annotations

import random
import statistics

import json

from .config import RESULTS
from .store import store

MODE_EXPLAIN = {
    "false_abstention": "The answer abstained but the reference is substantive: a retrieval failure. A soft LLM judge "
                        "scores this 1.0 (\"correctly refused to fabricate\") — the abstention-reward bias, 6–22pp of inflation.",
    "correct_abstention": "The reference itself says the information is unavailable, so abstaining is right. Phrase-list "
                          "detectors missed 33–67% of these before the asymmetric v3 lists.",
    "normal_judge": "Neither side abstains; the LLM judge compares the answer to the reference under the strict rubric.",
    "heuristic_match": "Decided by the deterministic branch (exact or normalised match) without an LLM call.",
}


def _load() -> dict[str, dict]:
    """The calibration ids are unique only per (source, id) — e.g. S1 and S2 both have a `q30` — so the game keys
    items on `source:id`. The label files are positionally aligned with calibration_sample.json (same order,
    same ids); they are merged by position with an id check rather than by the non-unique id."""
    base = RESULTS / "judge_calibration"
    cal = json.loads((base / "calibration_sample.json").read_text())
    items: dict[str, dict] = {}
    for x in cal:
        key = f"{x.get('source', '')}:{x['id']}"
        items[key] = {**x, "key": key}
    keys = list(items)

    def _merge(fname: str, field: str, *src_fields: str, extra: dict[str, str] | None = None) -> None:
        try:
            labels = json.loads((base / fname).read_text())
        except (OSError, ValueError):
            return
        if not isinstance(labels, list) or len(labels) != len(keys):
            return
        for key, lab in zip(keys, labels):
            if lab.get("id") != items[key]["id"]:
                continue
            val = next((lab[f] for f in src_fields if f in lab), None)
            items[key][field] = val
            for dst, src in (extra or {}).items():
                items[key][dst] = lab.get(src)

    _merge("proxy_human_labels.json", "proxy_1", "proxy_score", extra={"proxy_1_reasoning": "proxy_reasoning"})
    _merge("proxy_human_labels_pass2.json", "proxy_2", "proxy_score_2", "proxy_score")
    _merge("haiku_judge_labels.json", "haiku", "haiku_score", "score")
    return items


ITEMS = _load()


def next_item(task: str | None = None, seed: int | None = None) -> dict:
    pool = [x for x in ITEMS.values() if not task or x.get("task") == task]
    x = random.Random(seed).choice(pool) if seed is not None else random.choice(pool)
    return {"id": x["key"], "task": x.get("task"), "source": x.get("source"), "question": x["question"],
            "reference": x["reference"], "answer": x["answer"]}


def reveal(item_id: str) -> dict | None:
    x = ITEMS.get(item_id)
    if not x:
        return None
    proxies = [v for v in (x.get("proxy_1"), x.get("proxy_2")) if v is not None]
    mode = x.get("strict_mode", "normal_judge")
    return {
        "id": x["key"], "strict_score": x.get("strict_score"), "strict_mode": mode, "mode_explained": MODE_EXPLAIN.get(mode, mode),
        "soft_judge_would_say": 1.0 if mode == "false_abstention" else None,
        "human_proxy": round(statistics.mean(proxies), 2) if proxies else None, "human_proxy_reasoning": x.get("proxy_1_reasoning"),
        "haiku_judge": x.get("haiku"),
    }


def vote(item_id: str, score: float, session: str, ip_hash: str) -> dict:
    if item_id not in ITEMS:
        raise KeyError(item_id)
    score = min(1.0, max(0.0, float(score)))
    rec = {"item_id": item_id, "score": score, "session": session[:40], "ip_hash": ip_hash[:16]}
    store().append("votes", rec)
    return rec


def stats() -> dict:
    votes = store().list("votes")
    if not votes:
        return {"n_votes": 0}
    agree_strict = agree_proxy = n = 0
    by_mode: dict[str, list[bool]] = {}
    for v in votes:
        x = ITEMS.get(v.get("item_id"))
        if not x or x.get("strict_score") is None:
            continue
        n += 1
        a_strict = (v["score"] >= 0.5) == (float(x["strict_score"]) >= 0.5)
        agree_strict += a_strict
        proxies = [p for p in (x.get("proxy_1"), x.get("proxy_2")) if p is not None]
        if proxies:
            agree_proxy += (v["score"] >= 0.5) == (statistics.mean(proxies) >= 0.5)
        by_mode.setdefault(x.get("strict_mode", "?"), []).append(a_strict)
    return {"n_votes": n, "agree_with_strict_v3": round(agree_strict / n, 3) if n else None,
            "agree_with_human_proxy": round(agree_proxy / n, 3) if n else None,
            "by_mode": {m: {"n": len(v), "agree_with_strict": round(sum(v) / len(v), 3)} for m, v in by_mode.items()},
            "reference_kappas": {"strict_v3_vs_human_proxy_binary": 0.87, "strict_v3_vs_human_proxy_weighted": 0.92,
                                 "inter_rater_binary": 0.91, "haiku_vs_human_proxy_binary": 0.50}}
