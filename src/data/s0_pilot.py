"""S0 Pilot Dataset: 10 hand-crafted memories + 15 evaluation questions.

Covers all 5 task types (T1-T5) for end-to-end pipeline validation.
Includes 1 conflict pair for T4 testing.

This dataset is small enough to run quickly (~$0.50 API cost)
while exercising every code path in the experiment pipeline.
"""

from __future__ import annotations

from datetime import datetime

from src.architectures.base import Memory, TaskType
from src.benchmarks.base import BenchmarkDataset, EvalQuestion


def create_s0_dataset() -> BenchmarkDataset:
    """Create the hand-crafted S0 pilot dataset."""

    memories = [
        # --- Facts about Alice (project manager) ---
        Memory(
            content="Alice Chen is the project manager for Project Atlas. She joined the company in March 2025.",
            entities=["Alice Chen", "Project Atlas"],
            topics=["alice_chen", "project_atlas"],
            timestamp=datetime(2026, 1, 5),
            importance=4.0,
            source_id="s0_001",
        ),
        Memory(
            content="Alice prefers morning standup meetings at 9 AM and communicates primarily through Slack.",
            entities=["Alice Chen"],
            topics=["alice_chen", "preferences"],
            timestamp=datetime(2026, 1, 10),
            importance=3.0,
            source_id="s0_002",
        ),
        # --- Facts about Project Atlas ---
        Memory(
            content="Project Atlas is a data pipeline migration from Airflow to Dagster, targeting Q2 2026 completion.",
            entities=["Project Atlas", "Airflow", "Dagster"],
            topics=["project_atlas", "technical"],
            timestamp=datetime(2026, 1, 15),
            importance=5.0,
            source_id="s0_003",
        ),
        Memory(
            content="The Project Atlas budget was approved at $150,000 by the VP of Engineering on January 20, 2026.",
            entities=["Project Atlas"],
            topics=["project_atlas", "budget"],
            timestamp=datetime(2026, 1, 20),
            importance=4.0,
            source_id="s0_004",
        ),
        # --- Facts about Bob (engineer) ---
        Memory(
            content="Bob Martinez is the lead engineer on Project Atlas. He has 8 years of experience with distributed systems.",
            entities=["Bob Martinez", "Project Atlas"],
            topics=["bob_martinez", "project_atlas"],
            timestamp=datetime(2026, 1, 12),
            importance=3.5,
            source_id="s0_005",
        ),
        Memory(
            content="Bob reported that the Airflow-to-Dagster migration is 60% complete as of February 15, 2026.",
            entities=["Bob Martinez", "Project Atlas"],
            topics=["project_atlas", "progress"],
            timestamp=datetime(2026, 2, 15),
            importance=4.0,
            source_id="s0_006",
        ),
        # --- Temporal fact (will be superseded) ---
        Memory(
            content="The Project Atlas deadline is June 30, 2026.",
            entities=["Project Atlas"],
            topics=["project_atlas", "deadline"],
            timestamp=datetime(2026, 1, 15),
            importance=5.0,
            source_id="s0_007",
        ),
        # --- Conflict: deadline update (supersedes s0_007) ---
        Memory(
            content="The Project Atlas deadline has been extended to August 15, 2026 due to scope changes.",
            entities=["Project Atlas"],
            topics=["project_atlas", "deadline"],
            timestamp=datetime(2026, 3, 1),
            importance=5.0,
            source_id="s0_008",
            supersedes_id="s0_007",
        ),
        # --- Unrelated fact (for abstention testing) ---
        Memory(
            content="The company holiday party is scheduled for December 20, 2026 at the downtown office.",
            entities=["Company"],
            topics=["events"],
            timestamp=datetime(2026, 2, 1),
            importance=2.0,
            source_id="s0_009",
        ),
        # --- Multi-session reasoning fact ---
        Memory(
            content="Carol Davis from the QA team raised 3 critical bugs in the Dagster integration during the March review.",
            entities=["Carol Davis", "Project Atlas", "Dagster"],
            topics=["project_atlas", "qa", "bugs"],
            timestamp=datetime(2026, 3, 10),
            importance=4.0,
            source_id="s0_010",
        ),
    ]

    questions = [
        # ── T1: Single-fact extraction (3 questions) ──
        EvalQuestion(
            id="s0_q01",
            question="Who is the project manager for Project Atlas?",
            reference_answer="Alice Chen is the project manager for Project Atlas.",
            task_type=TaskType.SINGLE_FACT,
        ),
        EvalQuestion(
            id="s0_q02",
            question="What is the approved budget for Project Atlas?",
            reference_answer="The Project Atlas budget was approved at $150,000 by the VP of Engineering.",
            task_type=TaskType.SINGLE_FACT,
        ),
        EvalQuestion(
            id="s0_q03",
            question="How many years of experience does Bob Martinez have with distributed systems?",
            reference_answer="Bob Martinez has 8 years of experience with distributed systems.",
            task_type=TaskType.SINGLE_FACT,
        ),
        # ── T2: Multi-session reasoning (3 questions) ──
        EvalQuestion(
            id="s0_q04",
            question="What is the relationship between Alice Chen and Bob Martinez on Project Atlas?",
            reference_answer="Alice Chen is the project manager and Bob Martinez is the lead engineer on Project Atlas.",
            task_type=TaskType.MULTI_SESSION,
        ),
        EvalQuestion(
            id="s0_q05",
            question="What were the major developments on Project Atlas in the first quarter of 2026?",
            reference_answer="Project Atlas had its budget approved at $150,000 in January, reached 60% migration completion by February, had its deadline extended to August 2026 in March, and Carol Davis from QA identified 3 critical bugs in the Dagster integration during the March review.",
            task_type=TaskType.MULTI_SESSION,
        ),
        EvalQuestion(
            id="s0_q06",
            question="Who are all the people involved in Project Atlas and what are their roles?",
            reference_answer="Alice Chen is the project manager, Bob Martinez is the lead engineer, and Carol Davis from QA raised critical bugs during the review.",
            task_type=TaskType.MULTI_SESSION,
        ),
        # ── T3: Temporal reasoning (3 questions) ──
        EvalQuestion(
            id="s0_q07",
            question="What was the Project Atlas deadline as of January 2026?",
            reference_answer="As of January 2026, the Project Atlas deadline was June 30, 2026.",
            task_type=TaskType.TEMPORAL,
            query_timestamp="2026-01-31T23:59:59",
        ),
        EvalQuestion(
            id="s0_q08",
            question="What is the current Project Atlas deadline?",
            reference_answer="The current Project Atlas deadline is August 15, 2026, extended from June 30 due to scope changes.",
            task_type=TaskType.TEMPORAL,
        ),
        EvalQuestion(
            id="s0_q09",
            question="What was the migration progress as of mid-February?",
            reference_answer="As of February 15, 2026, the Airflow-to-Dagster migration was 60% complete.",
            task_type=TaskType.TEMPORAL,
            query_timestamp="2026-02-28T23:59:59",
        ),
        # ── T4: Conflict resolution (3 questions) ──
        EvalQuestion(
            id="s0_q10",
            question="When is Project Atlas expected to be completed?",
            reference_answer="Project Atlas is expected to be completed by August 15, 2026. The deadline was originally June 30 but was extended due to scope changes.",
            task_type=TaskType.CONFLICT,
            metadata={
                "original_fact": "The Project Atlas deadline is June 30, 2026.",
                "updated_fact": "The Project Atlas deadline has been extended to August 15, 2026.",
                "hop_type": "single",
            },
        ),
        EvalQuestion(
            id="s0_q11",
            question="Has the project timeline changed since it was first planned?",
            reference_answer="Yes, the Project Atlas deadline was originally June 30, 2026 but was extended to August 15, 2026 in March due to scope changes.",
            task_type=TaskType.CONFLICT,
            metadata={"hop_type": "single"},
        ),
        EvalQuestion(
            id="s0_q12",
            question="Given the deadline extension and the QA bugs found in March, is Project Atlas on track?",
            reference_answer="Project Atlas faces challenges: the deadline was extended from June to August 2026, the migration was 60% complete as of February, and 3 critical bugs were found in March. These factors suggest the project may face further delays.",
            task_type=TaskType.CONFLICT,
            metadata={"hop_type": "multi"},
        ),
        # ── T5: Abstention (3 questions) ──
        EvalQuestion(
            id="s0_q13",
            question="What is the tech stack used by Project Beta?",
            reference_answer="I don't have enough information to answer this question. There is no information about Project Beta in the available context.",
            task_type=TaskType.ABSTENTION,
        ),
        EvalQuestion(
            id="s0_q14",
            question="What is Alice Chen's salary?",
            reference_answer="I don't have enough information to answer this question. No salary information is available.",
            task_type=TaskType.ABSTENTION,
        ),
        EvalQuestion(
            id="s0_q15",
            question="Who is the CEO of the company?",
            reference_answer="I don't have enough information to answer this question. The CEO is not mentioned in the available context.",
            task_type=TaskType.ABSTENTION,
        ),
    ]

    return BenchmarkDataset(
        benchmark_name="S0_Pilot",
        scale="S0",
        memories=memories,
        questions=questions,
        metadata={
            "description": "Hand-crafted pilot dataset for end-to-end validation",
            "n_memories": len(memories),
            "n_questions": len(questions),
            "conflict_pairs": 1,
            "task_distribution": {
                "T1": 3, "T2": 3, "T3": 3, "T4": 3, "T5": 3,
            },
        },
    )
