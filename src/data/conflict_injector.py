"""Controlled conflict injection for T4 evaluation.

Generates counterfactual fact pairs using MemoryAgentBench's
methodology: for each original fact, create a temporally-later
fact that contradicts it.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from src.architectures.base import Memory
from src.evaluation.cost_tracker import CostTracker
from src.llm_client import chat_completion_json


async def inject_conflicts(
    memories: list[Memory],
    conflict_rate: float = 0.05,
    cost_tracker: CostTracker | None = None,
    seed: int = 42,
) -> tuple[list[Memory], list[dict]]:
    """Inject controlled conflicts into a memory set.

    For each selected memory, generates a counterfactual update
    that arrives later in time. Returns the augmented memory list
    and a manifest of injected conflicts (for evaluation).

    Args:
        memories: Original memory list.
        conflict_rate: Fraction of memories to create conflicts for.
        cost_tracker: For tracking LLM costs.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (augmented_memories, conflict_manifest).
        Manifest entries: {original_id, conflict_id, original_content,
                          conflict_content, ground_truth: "conflict"}
    """
    rng = random.Random(seed)
    n_conflicts = max(1, int(len(memories) * conflict_rate))
    selected = rng.sample(range(len(memories)), min(n_conflicts, len(memories)))

    conflict_memories = []
    manifest = []

    # Batch process conflicts
    batch_size = 20
    for i in range(0, len(selected), batch_size):
        batch_indices = selected[i : i + batch_size]
        batch_facts = [memories[idx] for idx in batch_indices]

        facts_text = "\n".join(
            f"[{idx}] {mem.content}" for idx, mem in zip(batch_indices, batch_facts)
        )

        result = await chat_completion_json(
            messages=[
                {
                    "role": "user",
                    "content": f"""For each fact below, generate a plausible counterfactual update.
The update should directly contradict the original fact while being realistic.

Original facts:
{facts_text}

Return JSON: {{"conflicts": [{{"index": 0, "updated_content": "...", "reason": "..."}}]}}""",
                }
            ],
            purpose="conflict_injection",
            cost_tracker=cost_tracker,
        )

        for conflict in result.get("conflicts", []):
            idx = conflict.get("index", 0)
            if idx >= len(memories):
                continue

            original = memories[idx]
            # Create the conflict memory with a later timestamp
            conflict_mem = Memory(
                content=conflict.get("updated_content", ""),
                entities=original.entities,
                topics=original.topics,
                timestamp=original.timestamp + timedelta(days=rng.randint(1, 30)),
                importance=original.importance + 0.5,
                source_id=f"conflict_{original.source_id}",
                supersedes_id=original.source_id,
            )
            conflict_memories.append(conflict_mem)

            manifest.append(
                {
                    "original_id": original.source_id,
                    "conflict_id": conflict_mem.source_id,
                    "original_content": original.content,
                    "conflict_content": conflict_mem.content,
                    "ground_truth": "conflict",
                    "correct_answer": conflict_mem.content,  # newer = correct
                }
            )

    # Merge and shuffle
    augmented = list(memories) + conflict_memories
    rng.shuffle(augmented)

    return augmented, manifest
