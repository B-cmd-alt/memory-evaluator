from collections import Counter

from src.data.sixaxis_generator import ABSTAIN, generate, ground_check


def test_s1_shape_and_grounding():
    d = generate("S1", seed=42)
    assert 70 <= d["meta"]["n_memories"] <= 140
    tasks = Counter(q["task"] for q in d["questions"])
    assert all(tasks[t] >= 3 for t in ("T1", "T2", "T3", "T4", "T5")), tasks
    assert ground_check(d) == []


def test_every_question_is_auditable():
    d = generate("S1", seed=7)
    for q in d["questions"]:
        assert len(q["decoys"]) == 2
        assert q["asked_at"] >= max(m["date"] for m in d["memories"] if m["kind"] != "poison")
        if q["task"] == "T4":
            assert q["stale_answer"] and q["stale_evidence_ids"] and q["stale_answer"] != q["reference_answer"]
        if q["task"] == "T5":
            assert q["reference_answer"] == ABSTAIN and not q["evidence_ids"]


def test_deletions_and_poisons_are_paired_with_probes():
    d = generate("S1", seed=3)
    shapes = Counter(x["shape"] for x in d["deletions"])
    assert shapes["single"] >= 1 and shapes["entity"] >= 1
    for c in d["deletions"]:
        assert c["delete_ids"] and c["probe_question"] and c["control_question"] and c["probe_answer"]
        # leak strings may be empty (a shared city is not a leak) but never non-unique
        others = " ".join(m["content"].lower() for m in d["memories"] if m["id"] not in set(c["delete_ids"]))
        assert all(s.lower() not in others for s in c["leak_strings"])
        assert not set(c["control_evidence_ids"]) & set(c["delete_ids"]), "control must not depend on deleted memories"
    for p in d["poisons"]:
        assert p["poison_answer"] != p["trusted_answer"]
        pm = next(m for m in d["memories"] if m["id"] == p["poison_memory_id"])
        assert pm["trust"] == "low" and pm["kind"] == "poison"


def test_deterministic_for_seed_and_different_across_seeds():
    a, b, c = generate("S1", 1), generate("S1", 1), generate("S1", 2)
    assert a == b and a != c


def test_s2_scale():
    d = generate("S2", seed=42)
    assert 600 <= d["meta"]["n_memories"] <= 900
    assert ground_check(d) == []
