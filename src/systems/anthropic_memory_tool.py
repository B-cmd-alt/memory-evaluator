"""Model-managed file memory — the Anthropic `memory` tool contract (memory_20250818), executed client-side.

The API's memory tool gives the model six commands over a `/memories` directory — view, create,
str_replace, insert, delete, rename — and the CLIENT executes them. The model decides what to
write, how to organise files, and what to read back. This is the same paradigm as Claude Code's
CLAUDE.md / auto-memory files and Letta's filesystem memory. We implement the command set faithfully
over a sandboxed directory and drive it through the shim's emulated function calling, so it runs on
the same model as every other system. No vector store, no graph: recall quality is entirely the
model's filing discipline.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time

from src.systems.base_adapter import BaseAdapter
from src.systems.common import fmt_mem

TOOL = {"type": "function", "function": {
    "name": "memory",
    "description": "Memory directory tool. Commands: view(path) lists a directory or shows a file; create(path, file_text); "
                   "str_replace(path, old_str, new_str); insert(path, insert_line, insert_text); delete(path); rename(old_path, new_path). "
                   "All paths must start with /memories.",
    "parameters": {"type": "object", "properties": {
        "command": {"type": "string", "enum": ["view", "create", "str_replace", "insert", "delete", "rename"]},
        "path": {"type": "string"}, "file_text": {"type": "string"}, "old_str": {"type": "string"}, "new_str": {"type": "string"},
        "insert_line": {"type": "integer"}, "insert_text": {"type": "string"}, "old_path": {"type": "string"}, "new_path": {"type": "string"}},
        "required": ["command"]}}}

_SYS_INGEST = ("You maintain a personal memory directory at /memories using the `memory` tool. New notes arrive in "
               "batches; file them so you can answer detailed questions later (names, dates, decisions, who-did-what). "
               "Keep /memories/index.md as a short table of contents. Prefer updating existing files over creating many. "
               "When a note supersedes an older fact, record the change with both dates. When done filing, reply 'DONE'.")
_SYS_QUERY = ("You answer questions using ONLY your memory directory at /memories via the `memory` tool. First view "
              "/memories/index.md, then open the relevant files. Then answer concisely. If the files do not contain the "
              "answer, say \"I don't have enough information to answer this question.\"")


class MemoryToolAdapter(BaseAdapter):
    name, family, year, kind = "anthropic_memory_tool", "model-managed files", 2025, "modern"
    batch = 10
    max_turns_ingest = 12
    max_turns_query = 8

    def _on_reset(self):
        if getattr(self, "_root", None):
            shutil.rmtree(self._root, ignore_errors=True)
        self._root = tempfile.mkdtemp(prefix="memtool-")
        self.viewed: list[str] = []

    # ── the six commands, executed client-side on the sandbox ─────────────────────────────────
    def _p(self, path: str) -> str:
        if not path or not path.startswith("/memories"):
            raise ValueError("path must start with /memories")
        full = os.path.normpath(os.path.join(self._root, path[len("/memories"):].lstrip("/")))
        if not full.startswith(self._root):
            raise ValueError("path escapes /memories")
        return full

    def _exec(self, a: dict) -> str:
        try:
            c = a.get("command")
            if c == "view":
                p = self._p(a.get("path") or "/memories")
                if os.path.isdir(p):
                    return "\n".join(sorted(os.path.relpath(os.path.join(r, f), self._root) for r, _, fs in os.walk(p) for f in fs)) or "(empty directory)"
                self.viewed.append(a.get("path"))
                return open(p).read() if os.path.exists(p) else f"error: {a.get('path')} does not exist"
            if c == "create":
                p = self._p(a["path"]); os.makedirs(os.path.dirname(p), exist_ok=True)
                open(p, "w").write(a.get("file_text") or ""); return "created"
            if c == "str_replace":
                p = self._p(a["path"]); s = open(p).read()
                if s.count(a.get("old_str", "")) != 1:
                    return "error: old_str must occur exactly once"
                open(p, "w").write(s.replace(a["old_str"], a.get("new_str") or "", 1)); return "replaced"
            if c == "insert":
                p = self._p(a["path"]); lines = open(p).read().splitlines(keepends=True)
                i = max(0, min(int(a.get("insert_line", len(lines))), len(lines)))
                lines.insert(i, (a.get("insert_text") or "") + "\n"); open(p, "w").write("".join(lines)); return "inserted"
            if c == "delete":
                p = self._p(a["path"]); (shutil.rmtree if os.path.isdir(p) else os.remove)(p); return "deleted"
            if c == "rename":
                os.rename(self._p(a["old_path"]), self._p(a["new_path"])); return "renamed"
            return f"error: unknown command {c}"
        except Exception as e:  # noqa: BLE001
            return f"error: {type(e).__name__}: {e}"

    def _loop(self, system: str, user: str, max_turns: int) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for _ in range(max_turns):
            msg = self.llm.chat("", "", messages=msgs, tools=[TOOL])
            calls = getattr(msg, "tool_calls", None)
            if not calls:
                return (msg.content or "").strip()
            msgs.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]})
            for c in calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                except ValueError:
                    args = {}
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": self._exec(args)})
        return "(turn limit reached)"

    def _on_ingest(self, memories):
        for i in range(0, len(memories), self.batch):
            notes = "\n".join(fmt_mem(m) for m in memories[i:i + self.batch])
            self._loop(_SYS_INGEST, f"New notes to file:\n{notes}", self.max_turns_ingest)

    async def query(self, question, timestamp=None):
        t0 = time.time()
        self.viewed = []
        ans = self._loop(_SYS_QUERY, f"Question: {question}", self.max_turns_query)
        items = [self._item(f"viewed {p}", p, 1.0) for p in self.viewed]
        return self._result(ans, items, "memory_tool", t0, 0, 0)

    def snapshot(self) -> dict:
        return {os.path.relpath(os.path.join(r, f), self._root): open(os.path.join(r, f)).read()
                for r, _, fs in os.walk(self._root) for f in fs}

    async def close(self):
        shutil.rmtree(getattr(self, "_root", ""), ignore_errors=True)
