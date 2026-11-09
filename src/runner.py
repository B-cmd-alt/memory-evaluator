"""Experiment orchestrator.

Runs the full 5x5x5 experiment matrix with 3 replications.
Manages architecture lifecycle, data loading, query execution,
evaluation, and result persistence.

Usage:
    # Full experiment
    python -m src.runner

    # Pilot run (S0 only, single replication)
    python -m src.runner --pilot

    # Specific architecture and scale
    python -m src.runner --arch E_lattice --scale S0 S1

    # Skip architectures that need Neo4j
    python -m src.runner --skip B_graph D_hybrid
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import yaml

from src.architectures.base import MemorySystem, TaskType
from src.benchmarks.base import BenchmarkDataset, EvalQuestion
from src.data.s0_pilot import create_s0_dataset
from src.evaluation.cost_tracker import CostTracker
from src.evaluation.latency_tracker import LatencyTracker
from src.evaluation.nugget_judge import evaluate_single
from src.llm_client import ensure_api_key

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
DATA_DIR = Path("data/benchmarks")


def load_config(config_path: Path = Path("configs/experiment.yaml")) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def instantiate_architecture(arch_key: str, config: dict) -> MemorySystem:
    """Dynamically instantiate a memory architecture from config."""
    arch_config = config["architectures"][arch_key]
    module_path, class_name = arch_config["class"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(config=arch_config.get("params", {}))


async def load_dataset(scale: str, config: dict) -> BenchmarkDataset:
    """Load the appropriate dataset for a scale level."""
    if scale == "S0":
        return create_s0_dataset()

    # For S1+, load from benchmarks
    # Priority: LongMemEval (primary quality), supplemented by BEAM and MemoryAgentBench
    from src.benchmarks.longmemeval import LongMemEvalBenchmark
    from src.benchmarks.beam import BEAMBenchmark
    from src.benchmarks.memoryagentbench import MemoryAgentBenchBenchmark

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Start with LongMemEval as base
    lme = LongMemEvalBenchmark(DATA_DIR)
    try:
        await lme.download()
        base_dataset = await lme.load(scale)
    except Exception as e:
        logger.warning("Could not load LongMemEval: %s. Using S0 pilot as fallback.", e)
        base_dataset = create_s0_dataset()
        base_dataset.scale = scale

    # Supplement with MemoryAgentBench for T4 questions
    try:
        mab = MemoryAgentBenchBenchmark(DATA_DIR)
        await mab.download()
        mab_dataset = await mab.load(scale)
        # Add T4 questions and conflict memories
        base_dataset.questions.extend(mab_dataset.questions)
        base_dataset.memories.extend(mab_dataset.memories)
    except Exception as e:
        logger.warning("Could not load MemoryAgentBench: %s", e)

    # For S3+, inject synthetic conflicts
    if scale in ("S3", "S4"):
        from src.data.conflict_injector import inject_conflicts

        try:
            augmented, manifest = await inject_conflicts(
                base_dataset.memories,
                conflict_rate=0.05,
            )
            base_dataset.memories = augmented
            base_dataset.metadata["conflict_manifest"] = manifest
        except Exception as e:
            logger.warning("Could not inject conflicts: %s", e)

    return base_dataset


async def run_single_run(
    arch_key: str,
    arch: MemorySystem,
    dataset: BenchmarkDataset,
    run_dir: Path,
    cost_tracker: CostTracker,
    run_evaluation: bool = True,
) -> dict:
    """Execute a single experiment run: ingest, query, evaluate."""
    latency_tracker = LatencyTracker()
    run_start = time.perf_counter()

    # ── Phase 1: Ingest ──
    logger.info("  Ingesting %d memories...", len(dataset.memories))
    with latency_tracker.measure("ingestion"):
        ingest_stats = await arch.ingest(dataset.memories)
    logger.info(
        "  Ingested in %.1fs (%d LLM calls, %d embedding calls)",
        ingest_stats.duration_s,
        ingest_stats.llm_calls,
        ingest_stats.embedding_calls,
    )

    # ── Phase 2: Query per task type ──
    all_results = []
    for task_type in TaskType:
        task_questions = [q for q in dataset.questions if q.task_type == task_type]
        if not task_questions:
            continue

        logger.info("  Querying %d %s questions...", len(task_questions), task_type.value)

        for q in task_questions:
            try:
                with latency_tracker.measure("query", question_id=q.id, task=task_type.value):
                    qr = await arch.query(
                        question=q.question,
                        timestamp=(
                            datetime.fromisoformat(q.query_timestamp)
                            if q.query_timestamp
                            else None
                        ),
                    )

                result = {
                    "question_id": q.id,
                    "question": q.question,
                    "reference_answer": q.reference_answer,
                    "generated_answer": qr.answer,
                    "task_type": task_type.value,
                    "nuggets": q.nuggets,
                    "retrieved_context": [asdict(r) for r in qr.retrieved_context],
                    "retrieval_tier": qr.retrieval_tier,
                    "latency_ms": qr.latency_ms,
                    "cost_usd": qr.cost_usd,
                }

                # ── Phase 3: Evaluate (if enabled) ──
                if run_evaluation:
                    eval_result = await evaluate_single(
                        question_id=q.id,
                        question=q.question,
                        generated_answer=qr.answer,
                        reference_answer=q.reference_answer,
                        nuggets=q.nuggets if q.nuggets else None,
                        cost_tracker=cost_tracker,
                    )
                    result["nugget_accuracy"] = eval_result.nugget_accuracy
                    result["token_f1"] = eval_result.token_f1
                    result["abstention_detected"] = eval_result.abstention_detected
                    result["nugget_scores"] = [
                        {"nugget": ns.nugget, "score": ns.score, "reasoning": ns.reasoning}
                        for ns in eval_result.nugget_scores
                    ]

                all_results.append(result)

            except Exception as e:
                logger.error("  Error on question %s: %s", q.id, e)
                all_results.append({
                    "question_id": q.id,
                    "question": q.question,
                    "error": str(e),
                    "task_type": task_type.value,
                })

    # ── Phase 4: Save results ──
    run_duration = time.perf_counter() - run_start
    sys_stats = await arch.get_stats()

    run_summary = {
        "run_id": run_dir.name,
        "architecture": arch_key,
        "scale": dataset.scale,
        "started_at": datetime.now().isoformat(),
        "duration_s": run_duration,
        "n_memories": len(dataset.memories),
        "n_questions": len(dataset.questions),
        "n_results": len(all_results),
        "ingest_stats": asdict(ingest_stats),
        "system_stats": asdict(sys_stats),
        "latency_summary": {
            "ingestion": latency_tracker.summary("ingestion"),
            "query": latency_tracker.summary("query"),
        },
        "cost_summary": cost_tracker.summary(),
    }

    # Compute aggregate accuracy
    scored = [r for r in all_results if "nugget_accuracy" in r]
    if scored:
        run_summary["aggregate_nugget_accuracy"] = (
            sum(r["nugget_accuracy"] for r in scored) / len(scored)
        )
        # Per-task breakdown
        for task_type in TaskType:
            task_scored = [r for r in scored if r["task_type"] == task_type.value]
            if task_scored:
                run_summary[f"nugget_accuracy_{task_type.value}"] = (
                    sum(r["nugget_accuracy"] for r in task_scored) / len(task_scored)
                )

    # Write to disk
    with open(run_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    with open(run_dir / "summary.json", "w") as f:
        json.dump(run_summary, f, indent=2, default=str)

    # Mark as completed
    with open(run_dir / "completed.json", "w") as f:
        json.dump({"completed_at": datetime.now().isoformat()}, f)

    logger.info(
        "  Run complete: %.1fs, %d questions, avg NA=%.3f",
        run_duration,
        len(scored),
        run_summary.get("aggregate_nugget_accuracy", 0),
    )

    return run_summary


async def run_experiment(
    config_path: Path = Path("configs/experiment.yaml"),
    arch_filter: list[str] | None = None,
    scale_filter: list[str] | None = None,
    skip_archs: list[str] | None = None,
    pilot: bool = False,
    no_eval: bool = False,
) -> None:
    """Run the experiment matrix.

    Args:
        config_path: Path to experiment config.
        arch_filter: Only run these architectures (e.g., ["E_lattice"]).
        scale_filter: Only run these scales (e.g., ["S0", "S1"]).
        skip_archs: Skip these architectures (e.g., ["B_graph"]).
        pilot: Run S0 only with single replication.
        no_eval: Skip LLM judge evaluation (faster, no eval costs).
    """
    # Ensure API key is available before starting
    ensure_api_key()

    config = load_config(config_path)
    pricing_path = Path("configs/pricing.yaml")

    arch_keys = list(config["architectures"].keys())
    scale_keys = list(config["scales"].keys())
    seeds = config["experiment"]["random_seeds"]

    # Apply filters
    if arch_filter:
        arch_keys = [k for k in arch_keys if k in arch_filter]
    if skip_archs:
        arch_keys = [k for k in arch_keys if k not in skip_archs]
    if scale_filter:
        scale_keys = [k for k in scale_keys if k in scale_filter]
    if pilot:
        scale_keys = ["S0"]
        seeds = [seeds[0]]  # Single replication

    total_runs = len(arch_keys) * len(scale_keys) * len(seeds)
    completed = 0

    logger.info(
        "Experiment: %d architectures x %d scales x %d replications = %d runs",
        len(arch_keys),
        len(scale_keys),
        len(seeds),
        total_runs,
    )

    for scale_key in scale_keys:
        # Load dataset once per scale (shared across architectures)
        logger.info("Loading dataset for scale %s...", scale_key)
        dataset = await load_dataset(scale_key, config)
        logger.info(
            "Loaded %d memories, %d questions",
            len(dataset.memories),
            len(dataset.questions),
        )

        for arch_key in arch_keys:
            for seed in seeds:
                run_id = f"{arch_key}_{scale_key}_seed{seed}"
                run_dir = RESULTS_DIR / "raw" / run_id
                run_dir.mkdir(parents=True, exist_ok=True)

                # Skip if already completed
                if (run_dir / "completed.json").exists():
                    logger.info("[%d/%d] Skipping completed: %s", completed + 1, total_runs, run_id)
                    completed += 1
                    continue

                logger.info("[%d/%d] Starting: %s", completed + 1, total_runs, run_id)
                cost_tracker = CostTracker.from_config(pricing_path)

                # Instantiate and run
                arch = instantiate_architecture(arch_key, config)
                try:
                    summary = await run_single_run(
                        arch_key=arch_key,
                        arch=arch,
                        dataset=dataset,
                        run_dir=run_dir,
                        cost_tracker=cost_tracker,
                        run_evaluation=not no_eval,
                    )
                    completed += 1
                except Exception as e:
                    logger.error("Run failed: %s — %s", run_id, e, exc_info=True)
                    # Save error
                    with open(run_dir / "error.json", "w") as f:
                        json.dump({"error": str(e), "at": datetime.now().isoformat()}, f)
                finally:
                    await arch.reset()
                    await arch.close()

    logger.info("Experiment complete: %d/%d runs succeeded", completed, total_runs)


def main():
    parser = argparse.ArgumentParser(description="Memory Architecture Benchmark Runner")
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--pilot", action="store_true", help="S0 only, single replication")
    parser.add_argument("--arch", nargs="+", help="Only run these architectures")
    parser.add_argument("--scale", nargs="+", help="Only run these scales")
    parser.add_argument("--skip", nargs="+", help="Skip these architectures")
    parser.add_argument("--no-eval", action="store_true", help="Skip LLM judge evaluation")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    asyncio.run(
        run_experiment(
            config_path=args.config,
            arch_filter=args.arch,
            scale_filter=args.scale,
            skip_archs=args.skip,
            pilot=args.pilot,
            no_eval=args.no_eval,
        )
    )


if __name__ == "__main__":
    main()
