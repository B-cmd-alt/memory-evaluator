"""Visualization for the research paper.

Generates all figures specified in the plan:
1. Scale degradation curves
2. Cost-accuracy Pareto frontier
3. Conflict resolution deep-dive
4. Crossover point identification
5. Latency distribution
6. Lattice architecture diagram
7. Ingestion vs query cost tradeoff
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ARCH_COLORS = {
    "A_structured": "#1f77b4",
    "B_graph": "#ff7f0e",
    "C_vector_rag": "#2ca02c",
    "D_hybrid": "#d62728",
    "E_lattice": "#9467bd",
}

ARCH_LABELS = {
    "A_structured": "Structured (A)",
    "B_graph": "Graph (B)",
    "C_vector_rag": "RAG (C)",
    "D_hybrid": "Hybrid (D)",
    "E_lattice": "Lattice (E)",
}

SCALE_VALUES = {"S0": 10, "S1": 100, "S2": 1000, "S3": 10000, "S4": 100000}

FIGURES_DIR = Path("results/figures")


def setup_style():
    """Set publication-quality plot style."""
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
    })


def plot_scale_degradation(
    results: dict[str, dict[str, float]],
    ci: dict[str, dict[str, tuple[float, float]]] | None = None,
    metric_name: str = "Nugget Accuracy",
    filename: str = "fig1_scale_degradation.pdf",
) -> None:
    """Figure 1: Scale degradation curves.

    Args:
        results: {arch_key: {scale_key: mean_score}}
        ci: {arch_key: {scale_key: (ci_low, ci_high)}}
    """
    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()

    for arch_key in ARCH_COLORS:
        if arch_key not in results:
            continue

        scales_sorted = sorted(results[arch_key].keys(), key=lambda s: SCALE_VALUES.get(s, 0))
        x = [np.log10(SCALE_VALUES[s]) for s in scales_sorted]
        y = [results[arch_key][s] for s in scales_sorted]

        ax.plot(
            x, y,
            marker="o",
            color=ARCH_COLORS[arch_key],
            label=ARCH_LABELS[arch_key],
            linewidth=2,
            markersize=8,
        )

        if ci and arch_key in ci:
            y_low = [ci[arch_key][s][0] for s in scales_sorted]
            y_high = [ci[arch_key][s][1] for s in scales_sorted]
            ax.fill_between(x, y_low, y_high, alpha=0.15, color=ARCH_COLORS[arch_key])

    ax.set_xlabel("Scale (log₁₀ memories)")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} vs Memory Scale")
    ax.legend(loc="lower left")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["10", "100", "1K", "10K", "100K"])

    fig.savefig(FIGURES_DIR / filename)
    plt.close(fig)


def plot_pareto_frontier(
    costs: dict[str, dict[str, float]],
    accuracies: dict[str, dict[str, float]],
    scale: str = "S2",
    filename: str = "fig2_pareto.pdf",
) -> None:
    """Figure 2: Cost-accuracy Pareto frontier at a given scale."""
    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()

    points = []
    for arch_key in ARCH_COLORS:
        if arch_key in costs and arch_key in accuracies:
            if scale in costs[arch_key] and scale in accuracies[arch_key]:
                c = costs[arch_key][scale]
                a = accuracies[arch_key][scale]
                ax.scatter(c, a, color=ARCH_COLORS[arch_key], s=200, zorder=5)
                ax.annotate(
                    ARCH_LABELS[arch_key],
                    (c, a),
                    textcoords="offset points",
                    xytext=(10, 5),
                    fontsize=10,
                )
                points.append((c, a))

    ax.set_xlabel("Cost per Query (USD)")
    ax.set_ylabel("Nugget Accuracy")
    ax.set_title(f"Cost-Accuracy Tradeoff at {scale}")
    ax.set_xscale("log")

    fig.savefig(FIGURES_DIR / filename)
    plt.close(fig)


def plot_conflict_resolution(
    single_hop: dict[str, dict[str, float]],
    multi_hop: dict[str, dict[str, float]],
    filename: str = "fig3_conflict_resolution.pdf",
) -> None:
    """Figure 3: Conflict resolution accuracy comparison."""
    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, data, title in [
        (axes[0], single_hop, "Single-Hop CR"),
        (axes[1], multi_hop, "Multi-Hop CR"),
    ]:
        for arch_key in ARCH_COLORS:
            if arch_key not in data:
                continue
            scales = sorted(data[arch_key].keys(), key=lambda s: SCALE_VALUES.get(s, 0))
            x_pos = list(range(len(scales)))
            values = [data[arch_key][s] for s in scales]
            ax.bar(
                [p + list(ARCH_COLORS.keys()).index(arch_key) * 0.15 for p in x_pos],
                values,
                width=0.15,
                color=ARCH_COLORS[arch_key],
                label=ARCH_LABELS[arch_key],
            )

        ax.set_xlabel("Scale")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename)
    plt.close(fig)


def plot_latency_distribution(
    latencies: dict[str, dict[str, list[float]]],
    filename: str = "fig5_latency.pdf",
) -> None:
    """Figure 5: Latency violin plots."""
    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    scales = sorted(SCALE_VALUES.keys(), key=lambda s: SCALE_VALUES[s])
    fig, axes = plt.subplots(1, len(scales), figsize=(4 * len(scales), 6), sharey=True)

    for ax, scale in zip(axes, scales):
        data_for_scale = []
        labels = []
        colors = []

        for arch_key in ARCH_COLORS:
            if arch_key in latencies and scale in latencies[arch_key]:
                data_for_scale.append(latencies[arch_key][scale])
                labels.append(ARCH_LABELS[arch_key])
                colors.append(ARCH_COLORS[arch_key])

        if data_for_scale:
            parts = ax.violinplot(data_for_scale, showmedians=True)
            for i, pc in enumerate(parts.get("bodies", [])):
                pc.set_facecolor(colors[i] if i < len(colors) else "gray")
                pc.set_alpha(0.7)

        ax.set_title(f"Scale: {scale}")
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    axes[0].set_ylabel("Latency (ms)")
    fig.suptitle("Query Latency Distribution by Architecture and Scale")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename)
    plt.close(fig)
