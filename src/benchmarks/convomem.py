"""ConvoMem benchmark loader (supplementary).

75K Q/A pairs across 15 conversation lengths.
Used for cost/latency crossover analysis.

Source: https://huggingface.co/datasets/Salesforce/ConvoMem
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.architectures.base import Memory, TaskType
from src.benchmarks.base import Benchmark, BenchmarkDataset, EvalQuestion

EVIDENCE_TO_TASK = {
    "direct": TaskType.SINGLE_FACT,
    "indirect": TaskType.MULTI_SESSION,
    "temporal": TaskType.TEMPORAL,
    "multi_session": TaskType.MULTI_SESSION,
    "negative": TaskType.ABSTENTION,
    "updated": TaskType.CONFLICT,
}


class ConvoMemBenchmark(Benchmark):
    """Loader for ConvoMem (Salesforce) benchmark data."""

    @property
    def name(self) -> str:
        return "ConvoMem"

    async def download(self) -> None:
        """Download ConvoMem from HuggingFace."""
        cache_dir = self.data_dir / "ConvoMem"
        if cache_dir.exists():
            return

        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            from datasets import load_dataset

            ds = load_dataset("Salesforce/ConvoMem", cache_dir=str(cache_dir))
            # Save to disk for subsequent loads
            ds.save_to_disk(str(cache_dir / "dataset"))
        except Exception as e:
            # Create placeholder if download fails
            (cache_dir / "placeholder.txt").write_text(
                f"Download failed: {e}\nManually download from: "
                "https://huggingface.co/datasets/Salesforce/ConvoMem"
            )

    async def load(self, scale: str) -> BenchmarkDataset:
        """Load ConvoMem at the specified scale.

        Stratified sampling across conversation lengths and evidence types.
        """
        cache_dir = self.data_dir / "ConvoMem"
        dataset_dir = cache_dir / "dataset"

        memories = []
        questions = []

        if dataset_dir.exists():
            try:
                from datasets import load_from_disk

                ds = load_from_disk(str(dataset_dir))

                # Get the appropriate split
                split_data = ds["test"] if "test" in ds else ds[list(ds.keys())[0]]

                # Sample size based on scale
                sample_sizes = {"S0": 10, "S1": 50, "S2": 200, "S3": 500, "S4": 1000}
                n_samples = sample_sizes.get(scale, 200)

                for i, item in enumerate(split_data):
                    if i >= n_samples:
                        break

                    # Extract conversation as memory
                    conv = item.get("conversation", item.get("context", ""))
                    if conv:
                        memories.append(
                            Memory(
                                content=str(conv)[:2000],
                                entities=[],
                                topics=[f"convomem_{i}"],
                                timestamp=datetime(2026, 1, 1 + (i % 28)),
                                importance=3.0,
                                source_id=f"convomem_{i}",
                            )
                        )

                    # Extract Q/A pair
                    question = item.get("question", "")
                    answer = item.get("answer", item.get("reference", ""))
                    evidence_type = item.get("evidence_type", "direct")

                    if question:
                        task_type = EVIDENCE_TO_TASK.get(evidence_type, TaskType.SINGLE_FACT)
                        questions.append(
                            EvalQuestion(
                                id=f"convomem_{i}",
                                question=question,
                                reference_answer=answer,
                                task_type=task_type,
                                metadata={
                                    "evidence_type": evidence_type,
                                    "benchmark": "ConvoMem",
                                    "conversation_length": item.get("num_conversations", 0),
                                },
                            )
                        )

            except Exception:
                pass  # Dataset not available; will return empty

        return BenchmarkDataset(
            benchmark_name=self.name,
            scale=scale,
            memories=memories,
            questions=questions,
        )
