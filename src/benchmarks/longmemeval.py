"""LongMemEval benchmark loader.

500 curated questions testing 5 memory abilities:
- Information extraction (T1)
- Multi-session reasoning (T2)
- Temporal reasoning (T3)
- Knowledge updates (T3/T4)
- Abstention (T5)

Source: https://github.com/xiaowu0162/LongMemEval
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.architectures.base import Memory, TaskType
from src.benchmarks.base import Benchmark, BenchmarkDataset, EvalQuestion
from src.data.memory_extractor import extract_memories_from_text

# Map LongMemEval categories to our task types
CATEGORY_TO_TASK = {
    "information_extraction": TaskType.SINGLE_FACT,
    "single_session": TaskType.SINGLE_FACT,
    "multi_session": TaskType.MULTI_SESSION,
    "temporal": TaskType.TEMPORAL,
    "knowledge_update": TaskType.TEMPORAL,
    "abstention": TaskType.ABSTENTION,
}


class LongMemEvalBenchmark(Benchmark):
    """Loader for LongMemEval benchmark data."""

    @property
    def name(self) -> str:
        return "LongMemEval"

    async def download(self) -> None:
        """Download LongMemEval data from GitHub."""
        import subprocess

        repo_dir = self.data_dir / "LongMemEval"
        if repo_dir.exists():
            return

        self.data_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git", "clone", "--depth", "1",
                "https://github.com/xiaowu0162/LongMemEval.git",
                str(repo_dir),
            ],
            check=True,
        )

    async def load(self, scale: str) -> BenchmarkDataset:
        """Load LongMemEval at the specified scale.

        S0-S1: Use longmemeval_s.json (~40 sessions, 115K tokens)
        S2+: Use longmemeval_m.json or extend with synthetic padding
        """
        repo_dir = self.data_dir / "LongMemEval"

        if scale in ("S0", "S1"):
            data_file = repo_dir / "data" / "longmemeval_s.json"
        else:
            data_file = repo_dir / "data" / "longmemeval_m.json"
            if not data_file.exists():
                data_file = repo_dir / "data" / "longmemeval_s.json"

        with open(data_file) as f:
            raw_data = json.load(f)

        # Extract conversation sessions as memories
        memories = []
        questions = []

        # Process conversations into memories
        if isinstance(raw_data, dict):
            sessions = raw_data.get("sessions", raw_data.get("conversations", []))
        elif isinstance(raw_data, list):
            sessions = raw_data
        else:
            sessions = []

        for i, session in enumerate(sessions):
            if isinstance(session, dict):
                content = session.get("content", session.get("text", str(session)))
                ts = session.get("timestamp", f"2026-01-{(i % 28) + 1:02d}T10:00:00")
            else:
                content = str(session)
                ts = f"2026-01-{(i % 28) + 1:02d}T10:00:00"

            try:
                timestamp = datetime.fromisoformat(ts)
            except (ValueError, TypeError):
                timestamp = datetime(2026, 1, (i % 28) + 1)

            memories.append(
                Memory(
                    content=content[:2000],  # Truncate very long sessions
                    entities=[],
                    topics=[f"session_{i}"],
                    timestamp=timestamp,
                    importance=3.0,
                    source_id=f"longmemeval_session_{i}",
                )
            )

        # Process questions
        eval_questions = raw_data.get("questions", []) if isinstance(raw_data, dict) else []
        for q in eval_questions:
            category = q.get("category", "information_extraction")
            task_type = CATEGORY_TO_TASK.get(category, TaskType.SINGLE_FACT)

            questions.append(
                EvalQuestion(
                    id=q.get("id", f"lme_{len(questions)}"),
                    question=q.get("question", ""),
                    reference_answer=q.get("answer", q.get("reference", "")),
                    task_type=task_type,
                    query_timestamp=q.get("timestamp"),
                    metadata={"category": category, "benchmark": "LongMemEval"},
                )
            )

        # For S0, subsample to 10 memories
        if scale == "S0":
            memories = memories[:10]
            questions = questions[:20]
        elif scale == "S1":
            memories = memories[:100]

        return BenchmarkDataset(
            benchmark_name=self.name,
            scale=scale,
            memories=memories,
            questions=questions,
            metadata={"source_file": str(data_file)},
        )
