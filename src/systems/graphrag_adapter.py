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
from src.systems.common import SHIM_BASE, answer_from_context, fmt_mem


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
        self._engine = None
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
        # Incremental ingest = full re-index over input/ (graphrag 3.x has a separate `update` command that writes a
        # separate output folder; `index --method update` is not valid and made every round-2 poison ingest fail).
        # Consistent with this adapter's delete (rebuild); the cost is recorded in the ledger either way.
        r = subprocess.run([sys.executable, "-m", "graphrag", "index", "--root", self._root],
                           capture_output=True, text=True, env=env, timeout=self.index_timeout, stdin=subprocess.DEVNULL)
        self.index_log = (r.stdout + r.stderr)[-2000:]
        if r.returncode != 0:
            raise RuntimeError(f"graphrag index failed: {self.index_log[-600:]}")
        self._indexed = len(self.memories)
        self._engine = None

    # ── Round 2: GraphRAG as a CONTEXT BUILDER (roster decision after round 1) ──────────────────────
    # Round 1 let graphrag answer with its own local-search prompt and LLM call, which confounded its lead with
    # the answer prompt. Now its local-search context (entity neighbourhood + community reports + text units)
    # is built in-process with ZERO LLM calls and handed to the shared answer prompt like every other system.
    _engine = None

    def _build_engine(self):
        os.environ.setdefault("GRAPHRAG_API_KEY", "shim")
        from pathlib import Path
        from graphrag.api import query as gq
        from graphrag.cli.query import _resolve_output_files
        from graphrag.config.embeddings import entity_description_embedding
        from graphrag.config.load_config import load_config
        config = load_config(root_dir=Path(self._root))
        dfs = _resolve_output_files(config=config, output_list=["communities", "community_reports", "text_units", "relationships", "entities"],
                                    optional_list=["covariates"])
        level = 2  # the CLI default community level
        store = gq.get_embedding_store(config=config.vector_store, embedding_name=entity_description_embedding)
        cov = gq.read_indexer_covariates(dfs["covariates"]) if dfs.get("covariates") is not None else []
        return gq.get_local_search_engine(
            config=config, reports=gq.read_indexer_reports(dfs["community_reports"], dfs["communities"], level),
            text_units=gq.read_indexer_text_units(dfs["text_units"]), entities=gq.read_indexer_entities(dfs["entities"], dfs["communities"], level),
            relationships=gq.read_indexer_relationships(dfs["relationships"]), covariates={"claims": cov}, description_embedding_store=store,
            response_type="multiple paragraphs", system_prompt=gq.load_search_prompt(config.local_search.prompt), callbacks=[])

    async def query(self, question, timestamp=None):
        t0 = time.time()
        cwd = os.getcwd()   # graphrag's config loading changes the working directory; never let that leak to the caller
        try:
            if self._engine is None:
                self._engine = self._build_engine()
            ctx = self._engine.context_builder.build_context(query=question, **self._engine.context_builder_params)
        finally:
            os.chdir(cwd)
        chunks = ctx.context_chunks
        context = chunks if isinstance(chunks, str) else "\n\n".join(chunks)
        ans, i, o = answer_from_context(self.llm, context, question)
        items = [self._item(context[:4000], "graphrag", 1.0)]
        return self._result(ans, items, "local_context", t0, i, o)

    async def close(self):
        shutil.rmtree(getattr(self, "_root", ""), ignore_errors=True)
