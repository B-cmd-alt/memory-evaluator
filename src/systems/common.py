"""Shared plumbing for adapters: shim-backed LLM client, in-process embedder, token estimate."""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field

import numpy as np

SHIM_BASE = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8008/v1")
EMBED_MODEL = os.environ.get("SHIM_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.S)


def est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class LLM:
    """Minimal chat client over the shim. Counts calls and tokens per adapter (`caller`)."""
    caller: str
    model: str = "gpt-4o-mini"
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    _client: object = field(default=None, repr=False)

    def _c(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=SHIM_BASE, api_key="shim", timeout=600, max_retries=1,
                                  default_headers={"X-Caller": self.caller})
        return self._client

    def chat(self, system: str, user: str, *, json_mode: bool = False, tools: list | None = None,
             messages: list | None = None, model: str | None = None):
        msgs = messages or [{"role": "system", "content": system}, {"role": "user", "content": user}]
        kw = {"model": model or self.model, "messages": msgs, "temperature": 0}
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        if tools:
            kw["tools"] = tools
        r = self._c().chat.completions.create(**kw)
        self.calls += 1
        self.prompt_tokens += r.usage.prompt_tokens or 0
        self.completion_tokens += r.usage.completion_tokens or 0
        return r.choices[0].message

    def text(self, system: str, user: str, **kw) -> str:
        return (self.chat(system, user, **kw).content or "").strip()

    def json(self, system: str, user: str, **kw) -> dict:
        t = _FENCE.sub("", self.text(system, user, json_mode=True, **kw))
        try:
            return json.loads(t)
        except ValueError:
            return {"_raw": t}


class Embedder:
    """In-process sentence-transformers embedder; the same model the shim serves."""
    _inst = None
    _lock = threading.Lock()

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(EMBED_MODEL)
        self.calls = 0

    @classmethod
    def get(cls) -> "Embedder":
        with cls._lock:
            if cls._inst is None:
                cls._inst = cls()
        return cls._inst

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, batch_size=64), dtype=np.float32)


GENERATION_SYSTEM = "You answer questions from memory context. Be concise and specific."
GENERATION_PROMPT = """Given the following context retrieved from memory, answer the question.
If the context does not contain enough information to answer, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""


def answer_from_context(llm: LLM, context: str, question: str) -> tuple[str, int, int]:
    """The ONE answer prompt shared by every system (the MemDelta fixed-generator rule)."""
    prompt = GENERATION_PROMPT.format(context=context or "(empty)", question=question)
    before = (llm.prompt_tokens, llm.completion_tokens)
    ans = llm.text(GENERATION_SYSTEM, prompt)
    return ans, llm.prompt_tokens - before[0], llm.completion_tokens - before[1]


def fmt_mem(m) -> str:
    d = m.timestamp.strftime("%Y-%m-%d") if getattr(m, "timestamp", None) else "?"
    return f"[{d}] {m.content}"


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.s = time.time() - self.t0
