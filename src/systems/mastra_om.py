"""Mastra Observational Memory (2026) — faithful Python re-implementation of the published design.

Mastra's description (research page + blog, Feb 2026): no retrieval; two background agents keep a
dated, append-only OBSERVATION log. The OBSERVER compresses each batch of new messages into
compact dated observations; the REFLECTOR condenses the whole log when it grows past a threshold,
preserving dates and superseding facts. Answers are generated with the entire log in context, so
the context is stable (cache-friendly) and bounded (~30K tokens in their LongMemEval run).

Mastra is TypeScript; this is a re-implementation from the public description, not their code.
Thresholds are scaled to our haystack sizes and are configuration, not claims.
"""
from __future__ import annotations

import time

from src.systems.base_adapter import BaseAdapter
from src.systems.common import est_tokens, fmt_mem

_OBS_SYS = ("You are the Observer of an observational memory system. You turn raw dated notes into a compact, "
            "lossless-enough observation log: one line per fact, each line starting with its date (YYYY-MM-DD). "
            "Keep names, numbers, decisions and dates exactly. Do not drop facts. Do not add commentary.")
_OBS = "Existing log tail (for continuity, do not repeat):\n{tail}\n\nNew notes:\n{notes}\n\nWrite ONLY the new observation lines."
_REF_SYS = ("You are the Reflector of an observational memory system. Condense the observation log: merge "
            "duplicates, keep every distinct fact with its date, and when a later line supersedes an earlier "
            "one keep BOTH but mark the earlier as '(superseded on <date>)'. Output the full condensed log, "
            "one dated line per fact, chronological.")


class MastraObservationalMemory(BaseAdapter):
    name, family, year, kind = "mastra_om", "observation log (no retrieval)", 2026, "modern"
    retrieves = False
    batch = 10
    reflect_tokens = 4000

    def _on_reset(self):
        self.log: list[str] = []
        self.reflections = 0

    def _on_ingest(self, memories):
        new = [m for m in memories]
        for i in range(0, len(new), self.batch):
            chunk = new[i:i + self.batch]
            tail = "\n".join(self.log[-15:]) or "(empty)"
            out = self.llm.text(_OBS_SYS, _OBS.format(tail=tail, notes="\n".join(fmt_mem(m) for m in chunk)))
            self.log += [l.strip() for l in out.splitlines() if l.strip() and not l.strip().startswith("```")]
        if est_tokens("\n".join(self.log)) > self.reflect_tokens:
            out = self.llm.text(_REF_SYS, "\n".join(self.log))
            self.log = [l.strip() for l in out.splitlines() if l.strip() and not l.strip().startswith("```")]
            self.reflections += 1

    async def query(self, question, timestamp=None):
        t0 = time.time()
        ctx = "\n".join(self.log)
        ans, it, ot = self._answer(ctx, question)
        items = [self._item(f"(observation log: {len(self.log)} lines, ~{est_tokens(ctx)} tokens, {self.reflections} reflections)", "log", 1.0)]
        return self._result(ans, items, "observation_log", t0, it, ot)
