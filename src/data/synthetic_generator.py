"""Synthetic memory generation for S4 (100K memories).

Generates realistic memories from persona templates to
augment benchmark data at the enterprise scale level.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from src.architectures.base import Memory
from src.evaluation.cost_tracker import CostTracker
from src.llm_client import chat_completion_json

TOPIC_CATEGORIES = [
    "preferences", "projects", "relationships", "events",
    "decisions", "goals", "feedback", "technical",
    "meetings", "deliverables",
]


async def generate_persona(
    persona_id: int,
    cost_tracker: CostTracker | None = None,
) -> dict:
    """Generate a detailed persona for synthetic memory creation."""
    result = await chat_completion_json(
        messages=[
            {
                "role": "user",
                "content": f"""Generate a detailed persona for a professional (persona #{persona_id}).

Include:
- name, role, company
- key projects (2-3)
- preferences (communication style, tools, schedule)
- team members (3-5 names)
- current goals (2-3)

Return JSON: {{"persona": {{...}}}}""",
            }
        ],
        purpose="data_generation",
        cost_tracker=cost_tracker,
    )
    return result.get("persona", {"name": f"Person_{persona_id}", "role": "engineer"})


async def generate_memories_for_persona(
    persona: dict,
    n_memories: int,
    start_date: datetime,
    cost_tracker: CostTracker | None = None,
    seed: int = 42,
) -> list[Memory]:
    """Generate n synthetic memories for a persona.

    Spreads memories over a 1-year period from start_date.
    """
    rng = random.Random(seed)
    memories = []

    # Generate in batches
    batch_size = 50
    persona_str = str(persona)

    for batch_start in range(0, n_memories, batch_size):
        batch_n = min(batch_size, n_memories - batch_start)
        topics = rng.choices(TOPIC_CATEGORIES, k=batch_n)

        result = await chat_completion_json(
            messages=[
                {
                    "role": "user",
                    "content": f"""Generate {batch_n} realistic memories for this persona.

Persona: {persona_str}

Topics (one per memory): {topics}

Each memory should be a specific, atomic fact about this persona.
Spread across different aspects of their professional life.

Return JSON: {{"memories": [{{"content": "...", "entities": [...], "topic": "...", "importance": 1-5}}]}}""",
                }
            ],
            purpose="data_generation",
            cost_tracker=cost_tracker,
        )

        for i, item in enumerate(result.get("memories", [])):
            # Spread timestamps over 1 year
            day_offset = rng.randint(0, 365)
            ts = start_date + timedelta(days=day_offset, hours=rng.randint(8, 18))

            memories.append(
                Memory(
                    content=item.get("content", ""),
                    entities=item.get("entities", []),
                    topics=[item.get("topic", topics[i] if i < len(topics) else "general")],
                    timestamp=ts,
                    importance=item.get("importance", 3.0),
                    source_id=f"synth_{persona.get('name', 'unknown')}_{batch_start + i}",
                )
            )

    return memories
