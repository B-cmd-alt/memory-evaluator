"""MemoryAgentBench benchmark loader.

Tests 4 core competencies:
- Accurate Retrieval (RULER-QA, NIAH-MQ)
- Test-Time Learning
- Long-Range Understanding
- Conflict Resolution (FactConsolidation) — the key T4 benchmark

Source: https://github.com/HUST-AI-HYZ/MemoryAgentBench
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.architectures.base import Memory, TaskType
from src.benchmarks.base import Benchmark, BenchmarkDataset, EvalQuestion


class MemoryAgentBenchBenchmark(Benchmark):
    """Loader for MemoryAgentBench data."""

    @property
    def name(self) -> str:
        return "MemoryAgentBench"

    async def download(self) -> None:
        """Download MemoryAgentBench from GitHub."""
        import subprocess

        repo_dir = self.data_dir / "MemoryAgentBench"
        if repo_dir.exists():
            return

        self.data_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git", "clone", "--depth", "1",
                "https://github.com/HUST-AI-HYZ/MemoryAgentBench.git",
                str(repo_dir),
            ],
            check=True,
        )

    async def load(self, scale: str) -> BenchmarkDataset:
        """Load MemoryAgentBench — primarily used for T4 conflict resolution."""
        repo_dir = self.data_dir / "MemoryAgentBench"

        memories = []
        questions = []

        # Load FactConsolidation data (primary use case)
        fact_dir = repo_dir / "data" / "FactConsolidation"
        if not fact_dir.exists():
            fact_dir = repo_dir / "data"

        json_files = list(fact_dir.glob("*.json")) if fact_dir.exists() else []

        for data_file in json_files:
            try:
                with open(data_file) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            items = data if isinstance(data, list) else data.get("items", data.get("data", []))
            for i, item in enumerate(items):
                if isinstance(item, dict):
                    # Original fact
                    original = item.get("original", item.get("fact", ""))
                    if original:
                        memories.append(
                            Memory(
                                content=original,
                                entities=item.get("entities", []),
                                topics=[item.get("topic", "general")],
                                timestamp=datetime(2026, 1, 1),
                                importance=3.0,
                                source_id=f"mab_orig_{data_file.stem}_{i}",
                            )
                        )

                    # Updated/conflicting fact
                    updated = item.get("updated", item.get("counterfactual", ""))
                    if updated:
                        memories.append(
                            Memory(
                                content=updated,
                                entities=item.get("entities", []),
                                topics=[item.get("topic", "general")],
                                timestamp=datetime(2026, 6, 1),  # Later timestamp
                                importance=4.0,
                                source_id=f"mab_upd_{data_file.stem}_{i}",
                                supersedes_id=f"mab_orig_{data_file.stem}_{i}",
                            )
                        )

                    # Evaluation question
                    question = item.get("question", "")
                    answer = item.get("answer", item.get("correct_answer", updated))
                    hop_type = item.get("hop_type", "single")

                    if question:
                        # Determine if single-hop or multi-hop
                        task_type = TaskType.CONFLICT
                        questions.append(
                            EvalQuestion(
                                id=f"mab_{data_file.stem}_{i}",
                                question=question,
                                reference_answer=answer,
                                task_type=task_type,
                                metadata={
                                    "hop_type": hop_type,
                                    "benchmark": "MemoryAgentBench",
                                    "original_fact": original,
                                    "updated_fact": updated,
                                },
                            )
                        )

        # Scale adjustment
        if scale == "S0":
            memories = memories[:10]
            questions = questions[:10]
        elif scale == "S1":
            memories = memories[:100]
            questions = questions[:50]

        return BenchmarkDataset(
            benchmark_name=self.name,
            scale=scale,
            memories=memories,
            questions=questions,
        )
