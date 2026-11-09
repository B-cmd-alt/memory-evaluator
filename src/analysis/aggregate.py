"""Aggregate raw results into paper-ready summaries.

Reads per-run results from results/raw/, computes cell metrics,
runs statistical tests, generates figures, and writes summary tables.

Usage:
    python -m src.analysis.aggregate
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.analysis.statistics import (
    bootstrap_ci,
    cliffs_delta,
    find_crossover,
    friedman_test,
    holm_bonferroni_correction,
    pairwise_wilcoxon,
)
from src.analysis.visualization import (
    plot_conflict_resolution,
    plot_latency_distribution,
    plot_pareto_frontier,
    plot_scale_degradation,
)

logger = logging.getLogger(__name__)

RAW_DIR = Path("results/raw")
AGG_DIR = Path("results/aggregated")

ARCH_ORDER = ["A_structured", "B_graph", "C_vector_rag", "D_hybrid", "E_lattice"]
SCALE_ORDER = ["S0", "S1", "S2", "S3", "S4"]
SCALE_VALUES = {"S0": 10, "S1": 100, "S2": 1000, "S3": 10000, "S4": 100000}


def load_all_results() -> dict:
    """Load all completed run results.

    Returns: {arch: {scale: {seed: [results]}}}
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for run_dir in sorted(RAW_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        completed = run_dir / "completed.json"
        results_file = run_dir / "results.json"
        summary_file = run_dir / "summary.json"

        if not completed.exists() or not results_file.exists():
            continue

        # Parse run_id: {arch}_{scale}_seed{N}
        parts = run_dir.name.rsplit("_seed", 1)
        if len(parts) != 2:
            continue

        arch_scale = parts[0]
        seed = parts[1]

        # Find the split between arch and scale
        for scale in SCALE_ORDER:
            if arch_scale.endswith(f"_{scale}"):
                arch = arch_scale[: -(len(scale) + 1)]
                break
        else:
            continue

        with open(results_file) as f:
            results = json.load(f)

        with open(summary_file) as f:
            summary = json.load(f)

        data[arch][scale][seed] = {
            "results": results,
            "summary": summary,
        }

    return dict(data)


def compute_aggregates(data: dict) -> dict:
    """Compute per-cell aggregate metrics.

    Returns: {arch: {scale: {task: {metric: value}}}}
    """
    aggregates = {}

    for arch in ARCH_ORDER:
        if arch not in data:
            continue
        aggregates[arch] = {}

        for scale in SCALE_ORDER:
            if scale not in data[arch]:
                continue
            aggregates[arch][scale] = {}

            # Collect results across seeds
            all_results = []
            for seed_data in data[arch][scale].values():
                all_results.extend(seed_data["results"])

            # Overall metrics
            scored = [r for r in all_results if "nugget_accuracy" in r]
            if scored:
                na_values = np.array([r["nugget_accuracy"] for r in scored])
                ci_low, ci_high = bootstrap_ci(na_values)
                aggregates[arch][scale]["overall"] = {
                    "nugget_accuracy_mean": float(na_values.mean()),
                    "nugget_accuracy_std": float(na_values.std()),
                    "nugget_accuracy_ci": [ci_low, ci_high],
                    "token_f1_mean": float(
                        np.mean([r.get("token_f1", 0) for r in scored])
                    ),
                    "n": len(scored),
                }

            # Per-task metrics
            for task in ["T1", "T2", "T3", "T4", "T5"]:
                task_results = [r for r in scored if r.get("task_type") == task]
                if not task_results:
                    continue

                na_values = np.array([r["nugget_accuracy"] for r in task_results])
                ci_low, ci_high = bootstrap_ci(na_values)

                task_agg = {
                    "nugget_accuracy_mean": float(na_values.mean()),
                    "nugget_accuracy_std": float(na_values.std()),
                    "nugget_accuracy_ci": [ci_low, ci_high],
                    "n": len(task_results),
                }

                # T4-specific: separate single-hop and multi-hop
                if task == "T4":
                    single = [
                        r for r in task_results
                        if r.get("metadata", {}).get("hop_type") == "single"
                        or "single" in str(r.get("question_id", ""))
                    ]
                    multi = [
                        r for r in task_results
                        if r.get("metadata", {}).get("hop_type") == "multi"
                        or "multi" in str(r.get("question_id", ""))
                    ]
                    if single:
                        task_agg["single_hop_accuracy"] = float(
                            np.mean([r["nugget_accuracy"] for r in single])
                        )
                    if multi:
                        task_agg["multi_hop_accuracy"] = float(
                            np.mean([r["nugget_accuracy"] for r in multi])
                        )

                # T5-specific: abstention precision/recall
                if task == "T5":
                    abs_detected = [r for r in task_results if r.get("abstention_detected")]
                    task_agg["abstention_rate"] = len(abs_detected) / len(task_results)

                aggregates[arch][scale][task] = task_agg

            # Efficiency metrics from summaries
            summaries = [
                sd["summary"] for sd in data[arch][scale].values() if "summary" in sd
            ]
            if summaries:
                aggregates[arch][scale]["efficiency"] = {
                    "avg_duration_s": float(
                        np.mean([s.get("duration_s", 0) for s in summaries])
                    ),
                    "avg_cost_usd": float(
                        np.mean([
                            s.get("cost_summary", {}).get("total_cost_usd", 0)
                            for s in summaries
                        ])
                    ),
                    "latency_p50": float(
                        np.mean([
                            s.get("latency_summary", {}).get("query", {}).get("p50", 0)
                            for s in summaries
                        ])
                    ),
                    "latency_p95": float(
                        np.mean([
                            s.get("latency_summary", {}).get("query", {}).get("p95", 0)
                            for s in summaries
                        ])
                    ),
                }

    return aggregates


def run_statistical_tests(data: dict, aggregates: dict) -> dict:
    """Run the full statistical analysis plan."""
    tests = {"friedman": {}, "pairwise": {}, "crossover": []}

    for scale in SCALE_ORDER:
        # Collect NA scores per architecture for this scale
        arch_scores = {}
        for arch in ARCH_ORDER:
            if (
                arch in data
                and scale in data[arch]
            ):
                all_results = []
                for seed_data in data[arch][scale].values():
                    all_results.extend(seed_data["results"])
                scored = [
                    r["nugget_accuracy"]
                    for r in all_results
                    if "nugget_accuracy" in r
                ]
                if scored:
                    arch_scores[arch] = np.array(scored)

        if len(arch_scores) < 3:
            continue

        # Friedman test
        groups = [arch_scores[a] for a in ARCH_ORDER if a in arch_scores]
        stat, p = friedman_test(groups)
        tests["friedman"][scale] = {"statistic": stat, "p_value": p}

        # Pairwise Wilcoxon
        archs_present = [a for a in ARCH_ORDER if a in arch_scores]
        p_values = []
        pairs = []

        for i, a1 in enumerate(archs_present):
            for a2 in archs_present[i + 1 :]:
                result = pairwise_wilcoxon(
                    arch_scores[a1], arch_scores[a2], a1, a2, scale
                )
                pairs.append(asdict(result))
                p_values.append(result.p_value)

        # Holm-Bonferroni correction
        if p_values:
            corrected = holm_bonferroni_correction(p_values)
            for pair, sig in zip(pairs, corrected):
                pair["significant_corrected"] = sig

        tests["pairwise"][scale] = pairs

    # Crossover analysis
    for a1 in ARCH_ORDER:
        for a2 in ARCH_ORDER:
            if a1 >= a2:
                continue
            scales_with_data = []
            scores_a1 = []
            scores_a2 = []

            for scale in SCALE_ORDER:
                if (
                    a1 in aggregates
                    and scale in aggregates[a1]
                    and "overall" in aggregates[a1][scale]
                    and a2 in aggregates
                    and scale in aggregates[a2]
                    and "overall" in aggregates[a2][scale]
                ):
                    scales_with_data.append(SCALE_VALUES[scale])
                    scores_a1.append(
                        aggregates[a1][scale]["overall"]["nugget_accuracy_mean"]
                    )
                    scores_a2.append(
                        aggregates[a2][scale]["overall"]["nugget_accuracy_mean"]
                    )

            if len(scales_with_data) >= 3:
                crossover = find_crossover(
                    np.array(scales_with_data),
                    np.array(scores_a1),
                    np.array(scores_a2),
                    a1,
                    a2,
                )
                if crossover:
                    tests["crossover"].append(asdict(crossover))

    return tests


def generate_paper_tables(aggregates: dict) -> str:
    """Generate LaTeX-ready tables for the paper."""
    lines = []

    # Table 1: Main Results
    lines.append("% Table 1: Nugget Accuracy by Architecture x Scale")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{l" + "c" * len(SCALE_ORDER) + "}")
    lines.append("\\toprule")
    lines.append("Architecture & " + " & ".join(SCALE_ORDER) + " \\\\")
    lines.append("\\midrule")

    for arch in ARCH_ORDER:
        if arch not in aggregates:
            continue
        row = [arch.replace("_", "\\_")]
        for scale in SCALE_ORDER:
            if (
                scale in aggregates[arch]
                and "overall" in aggregates[arch][scale]
            ):
                na = aggregates[arch][scale]["overall"]["nugget_accuracy_mean"]
                ci = aggregates[arch][scale]["overall"]["nugget_accuracy_ci"]
                row.append(f"{na:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
            else:
                row.append("--")
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Nugget Accuracy by Architecture and Scale (95\\% BCa CI)}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def main():
    """Run full analysis pipeline."""
    AGG_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading results...")
    data = load_all_results()
    if not data:
        logger.error("No completed runs found in %s", RAW_DIR)
        return

    logger.info("Found data for architectures: %s", list(data.keys()))

    logger.info("Computing aggregates...")
    aggregates = compute_aggregates(data)

    with open(AGG_DIR / "aggregates.json", "w") as f:
        json.dump(aggregates, f, indent=2, default=str)

    logger.info("Running statistical tests...")
    tests = run_statistical_tests(data, aggregates)

    with open(AGG_DIR / "statistical_tests.json", "w") as f:
        json.dump(tests, f, indent=2, default=str)

    logger.info("Generating tables...")
    tables = generate_paper_tables(aggregates)
    (AGG_DIR / "tables.tex").write_text(tables)

    logger.info("Generating figures...")
    # Extract data for plotting
    na_results = {}
    cost_results = {}
    for arch in ARCH_ORDER:
        if arch not in aggregates:
            continue
        na_results[arch] = {}
        cost_results[arch] = {}
        for scale in SCALE_ORDER:
            if scale in aggregates[arch] and "overall" in aggregates[arch][scale]:
                na_results[arch][scale] = aggregates[arch][scale]["overall"][
                    "nugget_accuracy_mean"
                ]
            if scale in aggregates[arch] and "efficiency" in aggregates[arch][scale]:
                cost_results[arch][scale] = aggregates[arch][scale]["efficiency"][
                    "avg_cost_usd"
                ]

    ci_results = {}
    for arch in ARCH_ORDER:
        if arch not in aggregates:
            continue
        ci_results[arch] = {}
        for scale in SCALE_ORDER:
            if scale in aggregates[arch] and "overall" in aggregates[arch][scale]:
                ci = aggregates[arch][scale]["overall"]["nugget_accuracy_ci"]
                ci_results[arch][scale] = (ci[0], ci[1])

    try:
        plot_scale_degradation(na_results, ci_results)
        logger.info("  Generated fig1_scale_degradation.pdf")
    except Exception as e:
        logger.warning("  Could not generate scale degradation plot: %s", e)

    try:
        plot_pareto_frontier(cost_results, na_results, scale="S2")
        logger.info("  Generated fig2_pareto.pdf")
    except Exception as e:
        logger.warning("  Could not generate Pareto plot: %s", e)

    # Print summary to console
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)

    for arch in ARCH_ORDER:
        if arch not in aggregates:
            continue
        print(f"\n{arch}:")
        for scale in SCALE_ORDER:
            if scale in aggregates[arch] and "overall" in aggregates[arch][scale]:
                overall = aggregates[arch][scale]["overall"]
                print(
                    f"  {scale}: NA={overall['nugget_accuracy_mean']:.3f} "
                    f"[{overall['nugget_accuracy_ci'][0]:.3f}, "
                    f"{overall['nugget_accuracy_ci'][1]:.3f}] "
                    f"(n={overall['n']})"
                )

    # Print crossover points
    if tests["crossover"]:
        print("\nCROSSOVER POINTS:")
        for cp in tests["crossover"]:
            print(
                f"  {cp['arch_a']} vs {cp['arch_b']}: "
                f"10^{cp['crossover_scale']:.2f} memories "
                f"[{cp['crossover_ci_low']:.2f}, {cp['crossover_ci_high']:.2f}]"
            )

    logger.info("Analysis complete. Results in %s", AGG_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    main()
