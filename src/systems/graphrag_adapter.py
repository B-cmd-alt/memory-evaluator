"""Microsoft GraphRAG (graphrag package) through the shim: entity graph + community summaries.

Indexing extracts entities/relationships per text unit, builds communities (Leiden) and writes
community reports; querying uses LOCAL search (entity neighbourhood + text units). This is the
official package driven through its CLI with a settings.yaml that points chat + embeddings at the
shim. Indexing is the expensive step (tens of LLM calls even at S1); incremental ingestion is
approximated by re-indexing with `--method update` where available, else full re-index per batch.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time

import yaml

from src.systems.base_adapter import BaseAdapter
from src.systems.common import SHIM_BASE, fmt_mem


class GraphRAGAdapter(BaseAdapter):
    name, family, year, kind = "ms_graphrag", "graph RAG (communities)", 2024, "modern"
    index_timeout = 3600
    ingest_whole = True   # one indexing pass; per-batch re-indexing would multiply the cost ~9x

    def _on_reset(self):
        if getattr(self, "_root", None):
            shutil.rmtree(self._root, ignore_errors=True)
        self._root = tempfile.mkdtemp(prefix="graphrag-")
        os.makedirs(f"{self._root}/input", exist_ok=True)
        self._indexed = 0
        self.index_log = ""

    def _settings(self):
        env = os.environ.copy()
        env.update({"GRAPHRAG_API_KEY": "shim", "OPENAI_API_KEY": "shim"})
        # `init` prompts for model names unless they are given; stdin is closed so a prompt would abort.
        r = subprocess.run([sys.executable, "-m", "graphrag", "init", "--root", self._root, "--force", "-m", "gpt-4o-mini", "-e", "bge-small"],
                           capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL)
        if r.returncode != 0:
            raise RuntimeError(f"graphrag init failed: {r.stderr[-400:]}")
        cfg_path = f"{self._root}/settings.yaml"
        cfg = yaml.safe_load(open(cfg_path))
        # graphrag 3.x schema: completion_models / embedding_models blocks, input.type = file type,
        # input_storage.base_dir = directory. Point both model blocks at the shim; leave the rest as generated.
        for block, model in (("completion_models", "gpt-4o-mini"), ("embedding_models", "bge-small")):
            for name, m in (cfg.get(block) or {}).items():
                m.update({"model_provider": "openai", "model": model, "api_base": SHIM_BASE, "api_key": "shim", "auth_method": "api_key",
                          "concurrent_requests": 4})
        cfg["input"] = {"type": "text"}
        cfg.setdefault("input_storage", {}).update({"type": "file", "base_dir": "input"})
        yaml.safe_dump(cfg, open(cfg_path, "w"))
        return env

    def _on_ingest(self, memories):
        for m in memories:
            open(f"{self._root}/input/{m.source_id}.txt", "w").write(fmt_mem(m) + "\n")
        env = self._settings()
        method = ["--method", "update"] if self._indexed else []
        r = subprocess.run([sys.executable, "-m", "graphrag", "index", "--root", self._root, *method],
                           capture_output=True, text=True, env=env, timeout=self.index_timeout, stdin=subprocess.DEVNULL)
        self.index_log = (r.stdout + r.stderr)[-2000:]
        if r.returncode != 0:
            raise RuntimeError(f"graphrag index failed: {self.index_log[-600:]}")
        self._indexed = len(self.memories)

    async def query(self, question, timestamp=None):
        t0 = time.time()
        env = os.environ.copy(); env.update({"GRAPHRAG_API_KEY": "shim", "OPENAI_API_KEY": "shim"})
        # graphrag 3.x: the query is a positional argument
        r = subprocess.run([sys.executable, "-m", "graphrag", "query", "--root", self._root, "--method", "local", question],
                           capture_output=True, text=True, env=env, timeout=600, stdin=subprocess.DEVNULL)
        out = r.stdout.strip()
        if "Search Response:" in out:
            ans = out.split("Search Response:", 1)[-1].strip()
        else:
            ans = out or r.stderr[-400:]
        if r.returncode != 0:
            ans = f"<error: graphrag query exit {r.returncode}: {(r.stderr or out)[-300:]}>"
        items = [self._item("(graphrag local search: entities + community reports + text units)", "graphrag", 1.0)]
        return self._result(ans, items, "local_search", t0, 0, 0)

    async def close(self):
        shutil.rmtree(getattr(self, "_root", ""), ignore_errors=True)
