"""BEAM benchmark loader.

"Beyond a Million Tokens" — tests at 128K, 500K, 1M, 10M tokens.
Uses nugget-based evaluation (fine-grained atomic fact scoring).

Source: https://github.com/mohammadtavakoli78/BEAM
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.architectures.base import Memory, TaskType
from src.benchmarks.base import Benchmark, BenchmarkDataset, EvalQuestion

# Map BEAM question types to our task types
BEAM_TYPE_MAP = {
    "factual": TaskType.SINGLE_FACT,
    "multi_hop": TaskType.MULTI_SESSION,
    "temporal": TaskType.TEMPORAL,
    "contradiction": TaskType.CONFLICT,
    "event_ordering": TaskType.TEMPORAL,
    "abstention": TaskType.ABSTENTION,
}

# Map scales to BEAM context sizes
SCALE_TO_BEAM_SIZE = {
    "S0": "128k",  # subsample
    "S1": "128k",
    "S2": "128k",
    "S3": "500k",
    "S4": "1m",
}


class BEAMBenchmark(Benchmark):
    """Loader for BEAM benchmark data."""

    @property
    def name(self) -> str:
        return "BEAM"

    async def download(self) -> None:
        """Download BEAM data from GitHub."""
        import subprocess

        repo_dir = self.data_dir / "BEAM"
        if repo_dir.exists():
            return

        self.data_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git", "clone", "--depth", "1",
                "https://github.com/mohammadtavakoli78/BEAM.git",
                str(repo_dir),
            ],
            check=True,
        )

    async def load(self, scale: str) -> BenchmarkDataset:
        """Load BEAM at the specified scale."""
        repo_dir = self.data_dir / "BEAM"
        beam_size = SCALE_TO_BEAM_SIZE.get(scale, "128k")

        # Look for conversation data at this scale
        data_dir = repo_dir / "data" / beam_size
        if not data_dir.exists():
            # Try alternative paths
            data_dir = repo_dir / "data"

        memories = []
        questions = []

        # Load conversations
        conv_files = list(data_dir.glob("*.json")) if data_dir.exists() else []
        for conv_file in conv_files:
            try:
                with open(conv_file) as f:
                    conv_data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            # Process conversation turns into memories
            turns = conv_data if isinstance(conv_data, list) else conv_data.get("turns", [])
            for j, turn in enumerate(turns):
                if isinstance(turn, dict):
                    content = turn.get("content", turn.get("text", str(turn)))
                else:
                    content = str(turn)

                memories.append(
                    Memory(
                        content=content[:2000],
                        entities=[],
                        topics=[f"beam_{conv_file.stem}"],
                        timestamp=datetime(2026, 1, 1 + (j % 28)),
                        importance=3.0,
                        source_id=f"beam_{conv_file.stem}_{j}",
                    )
                )

            # Extract probing questions if available
            probes = conv_data.get("probing_questions", []) if isinstance(conv_data, dict) else []
            for q in probes:
                q_type = q.get("type", "factual")
                task_type = BEAM_TYPE_MAP.get(q_type, TaskType.SINGLE_FACT)

                questions.append(
                    EvalQuestion(
                        id=q.get("id", f"beam_{len(questions)}"),
                        question=q.get("question", ""),
                        reference_answer=q.get("answer", ""),
                        task_type=task_type,
                        nuggets=q.get("nuggets", []),
                        metadata={"type": q_type, "benchmark": "BEAM"},
                    )
                )

        # Scale adjustment
        if scale == "S0":
            memories = memories[:10]
            questions = questions[:20]
        elif scale == "S1":
            memories = memories[:100]
        elif scale == "S2":
            memories = memories[:1000]

        return BenchmarkDataset(
            benchmark_name=self.name,
            scale=scale,
            memories=memories,
            questions=questions,
            metadata={"beam_size": beam_size},
        )
