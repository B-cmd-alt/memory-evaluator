"""Offline contract tests for the adapters that need no shim: registry integrity + baseline behaviour
with a fake LLM. The third-party adapters are exercised by run_feasibility.py against the shim."""
from __future__ import annotations

import asyncio
import importlib.util
from datetime import datetime

import pytest

from src.architectures.base import Memory
from src.systems import registry
from src.systems.baselines import BM25Baseline, FullContextBaseline, VectorRAG
from src.systems.common import LLM


class FakeLLM(LLM):
    """Echoes the context it was given so tests can see what the system retrieved."""
    def __init__(self):
        super().__init__(caller="fake")
        self.last_prompt = None

    def chat(self, system, user, *, json_mode=False, tools=None, messages=None, model=None):
        self.calls += 1
        self.last_prompt = user if user else (messages or [{}])[-1].get("content")
        class M:  # noqa: D401
            content = "echo"
            tool_calls = None
        return M()


def mems():
    return [Memory(content=c, entities=[], topics=[], timestamp=datetime(2025, 1, d), importance=3, source_id=f"m{d}")
            for d, c in enumerate(["Linear replaced Jira company-wide in June.", "Carol Davis joined as QA lead.",
                                   "The release train moves to Thursdays."], start=1)]


def test_registry_has_twelve_entries_with_unique_keys():
    assert len(registry.REGISTRY) == 12
    assert len({e.key for e in registry.REGISTRY}) == 12
    for e in registry.REGISTRY:
        assert registry.load(e.key) is not None


@pytest.mark.parametrize("cls", [BM25Baseline, FullContextBaseline])
def test_baselines_ingest_query_delete(cls):
    fake = FakeLLM()
    s = cls({"llm": fake})
    asyncio.run(s.reset())
    asyncio.run(s.ingest(mems()))
    r = asyncio.run(s.query("What replaced Jira?"))
    assert r.answer == "echo" and fake.calls == 1
    assert "Jira" in fake.last_prompt
    d = asyncio.run(s.delete(["m1"]))
    assert d["deleted"] == 1
    asyncio.run(s.query("What replaced Jira?"))
    # the deleted memory's CONTENT must no longer reach the model (the question itself still mentions Jira)
    assert "Linear replaced Jira" not in fake.last_prompt


def test_bm25_retrieves_the_relevant_memory_first():
    fake = FakeLLM()
    s = BM25Baseline({"llm": fake})
    asyncio.run(s.reset()); asyncio.run(s.ingest(mems()))
    r = asyncio.run(s.query("Which tool replaced Jira?"))
    assert r.retrieved_context and "Jira" in r.retrieved_context[0].content


@pytest.mark.skipif(importlib.util.find_spec("sentence_transformers") is None, reason="embedder not installed")
def test_vector_rag_retrieves_semantically():
    fake = FakeLLM()
    s = VectorRAG({"llm": fake})
    asyncio.run(s.reset()); asyncio.run(s.ingest(mems()))
    r = asyncio.run(s.query("Who is the new quality assurance lead?"))
    assert r.retrieved_context and "Carol" in r.retrieved_context[0].content
