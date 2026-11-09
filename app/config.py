"""Runtime configuration for the interactive app. Everything comes from the environment; no value is secret here."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Model provider: any OpenAI-compatible endpoint. Production = OpenRouter with a key from Secret Manager.
LLM_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-5.5")
HAS_KEY = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY"))

# Cost posture (the whole point of the exercise is that this is bounded and visible).
DAILY_BUDGET_USD = float(os.environ.get("DAILY_BUDGET_USD", "1.00"))
PER_IP_DAILY_LIVE = int(os.environ.get("PER_IP_DAILY_LIVE", "12"))     # live LLM requests per visitor per day
LIVE_TIMEOUT_S = int(os.environ.get("LIVE_TIMEOUT_S", "90"))

# Storage for votes and the budget ledger: a GCS bucket in production, a local directory otherwise.
GCS_BUCKET = os.environ.get("EVALUATOR_BUCKET", "")
LOCAL_DATA = Path(os.environ.get("EVALUATOR_DATA_DIR", str(ROOT / "data")))

# Live systems (cheap write path; the expensive ones are served from round-1 records only).
LIVE_SYSTEMS = ("full_context", "bm25", "vector_rag", "hybrid_rerank", "mastra_om", "generative_agents")
PRECOMPUTED_SYSTEMS = ("full_context", "bm25", "vector_rag", "hybrid_rerank", "ms_graphrag", "mem0", "zep_graphiti",
                       "mastra_om", "anthropic_memory_tool", "generative_agents", "a_mem")
