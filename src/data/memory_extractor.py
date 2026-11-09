"""LLM-based memory extraction from benchmark conversations.

Converts raw benchmark data (dialogues, documents) into
normalized Memory objects usable by all architectures.
"""

from __future__ import annotations

import json
from datetime import datetime

from src.architectures.base import Memory
from src.evaluation.cost_tracker import CostTracker
from src.llm_client import chat_completion_json


async def extract_memories_from_text(
    text: str,
    source_id: str,
    base_timestamp: datetime | None = None,
    cost_tracker: CostTracker | None = None,
) -> list[Memory]:
    """Extract discrete memories from a block of text.

    Uses LLM to identify atomic facts, entities, topics,
    and temporal context from the input text.
    """
    if base_timestamp is None:
        base_timestamp = datetime.now()

    result = await chat_completion_json(
        messages=[
            {
                "role": "user",
                "content": f"""Extract all discrete facts, preferences, events, and relationships from this text.

For each memory, provide:
- content: the atomic fact statement
- entities: list of named entities involved
- topics: list of topic labels (lowercase, underscored)
- timestamp: ISO 8601 if a time is mentioned or inferable, otherwise null
- importance: 1-5 scale (1=trivial, 5=critical)

Text:
{text}

Return JSON: {{"memories": [{{...}}]}}""",
            }
        ],
        purpose="extraction",
        cost_tracker=cost_tracker,
    )

    memories = []
    for i, item in enumerate(result.get("memories", [])):
        ts = base_timestamp
        if item.get("timestamp"):
            try:
                ts = datetime.fromisoformat(item["timestamp"])
            except (ValueError, TypeError):
                pass

        memories.append(
            Memory(
                content=item.get("content", ""),
                entities=item.get("entities", []),
                topics=item.get("topics", []),
                timestamp=ts,
                importance=item.get("importance", 3.0),
                source_id=f"{source_id}_{i}",
            )
        )

    return memories


async def extract_memories_batch(
    texts: list[dict],
    cost_tracker: CostTracker | None = None,
    batch_size: int = 10,
) -> list[Memory]:
    """Extract memories from multiple texts in batches.

    Args:
        texts: List of dicts with 'text', 'source_id', and optional 'timestamp'.
    """
    all_memories = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        combined = "\n---\n".join(
            f"[Source: {t['source_id']}]\n{t['text']}" for t in batch
        )

        base_ts = datetime.now()
        result = await chat_completion_json(
            messages=[
                {
                    "role": "user",
                    "content": f"""Extract all discrete facts from these sources.

For each fact, provide:
- content: atomic fact statement
- entities: named entities
- topics: topic labels
- timestamp: ISO 8601 if detectable, null otherwise
- importance: 1-5
- source_id: the [Source: X] it came from

Sources:
{combined}

Return JSON: {{"memories": [{{...}}]}}""",
                }
            ],
            purpose="extraction",
            cost_tracker=cost_tracker,
        )

        for item in result.get("memories", []):
            ts = base_ts
            if item.get("timestamp"):
                try:
                    ts = datetime.fromisoformat(item["timestamp"])
                except (ValueError, TypeError):
                    pass

            all_memories.append(
                Memory(
                    content=item.get("content", ""),
                    entities=item.get("entities", []),
                    topics=item.get("topics", []),
                    timestamp=ts,
                    importance=item.get("importance", 3.0),
                    source_id=item.get("source_id", f"batch_{i}"),
                )
            )

    return all_memories
